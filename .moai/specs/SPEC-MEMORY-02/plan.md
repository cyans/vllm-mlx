---
id: SPEC-MEMORY-02
tags:
  - "@PLAN:MEMORY-02"
---

# Plan — MEMORY-02

## Primary Goal

Operationalize SPEC-MEMORY-01's Path C schema: ship a daily chat
consolidator + weekly vault thematic digest job + tier-aware ranking, all
running inside the existing FastAPI server process with no new
dependencies and no schema migration.

## Predecessors note

This SPEC depends on SPEC-MEMORY-01 being deployed and the schema
(`tier`, `session_id`, `chat_summaries`, `vault_themes`) being live in
production. SPEC-MEMORY-02 writes the first rows to `chat_summaries`
and `vault_themes`; both tables must already exist with the schema
defined in MEMORY-01 § 8.

## Architecture (ASCII)

```
                         FastAPI server process (single Python process)
            +--------------------------------------------------------------+
            |                                                              |
            |   +----------------------------+                             |
            |   | lifespan startup           |                             |
            |   |  - existing MEMORY-01 init |                             |
            |   |  - if CONSOLIDATOR_ENABLED:|                             |
            |   |      asyncio.create_task(  |                             |
            |   |         consolidator_loop) |                             |
            |   +----------------------------+                             |
            |                                                              |
            |   +----------------------------+                             |
            |   | /v1/chat/completions       |   <-- existing chat path    |
            |   | (uses Qwen3.6 engine)      |       acquires NO summarizer|
            |   +----------------------------+       semaphore             |
            |                  |                                           |
            |                  | shared engine pool                        |
            |                  v                                           |
            |   +----------------------------+                             |
            |   | Qwen3.6 (in-process)       |                             |
            |   +----------------------------+                             |
            |                  ^                                           |
            |                  | acquires summarizer_slot (Semaphore=1)    |
            |                  | yields between sessions to chat path      |
            |                  |                                           |
            |   +--------------+-------------+                             |
            |   | consolidator_loop          |                             |
            |   |  asyncio long task         |                             |
            |   |  - sleep until next tick   |                             |
            |   |  - run_chat_consolidation()|                             |
            |   |  - run_vault_digest() (Sun)|                             |
            |   +----+--------------------+--+                             |
            |        |                    |                                |
            |        | reads STM rows     | reads STM rows + writes        |
            |        | (WAL, no lock)     |  vault_themes only             |
            |        v                    v                                |
            |   +-------------+    +-----------------+                     |
            |   | chat        |    | vault           |                     |
            |   | consolidator|    | digest job      |                     |
            |   +------+------+    +--------+--------+                     |
            |          |                    |                              |
            |          | summarize + write  | cluster + write              |
            |          v                    v                              |
            |   +------------------------------------------+               |
            |   | SQLite (WAL): same DB as MEMORY-01       |               |
            |   |   - chat_summaries  (new rows)           |               |
            |   |   - chat_messages   (deleted/redacted)   |               |
            |   |   - vault_themes    (new rows)           |               |
            |   |   - vault_chunks    (UNTOUCHED)          |               |
            |   |   - vec_chunks      (new embeddings)     |               |
            |   +------------------------------------------+               |
            |                                                              |
            |   +----------------------------+                             |
            |   | memory_search (extended)   |                             |
            |   |   4-stream RRF             |                             |
            |   |   tier_filter support      |                             |
            |   +----------------------------+                             |
            +--------------------------------------------------------------+

         Lock-free read pattern: consolidator opens its own connection,
         WAL means readers never block writers and vice versa.

         Transactional write pattern: each session's summary write +
         eviction is one short transaction. Crash mid-tick rolls back
         to a state where either both happen or neither happens.
```

## Technology Decisions

| Concern                  | Pick                                                | Why                                                                                                                       |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Scheduler library        | **Hand-rolled** `asyncio.create_task` + `asyncio.sleep` | Total ~50 LOC, zero new deps, daily-at-hour + weekly-at-day are the only schedules needed. `apscheduler` is overkill.    |
| Summarizer model         | **`MEMORY_SUMMARIZER_MODEL=self`** (default = live Qwen3.6) | Already loaded, zero RAM cost, multilingual, follows the same auto-load lifecycle as the chat path. Smaller model is a one-env-var swap (REQ-O2). |
| Concurrency limit        | **`asyncio.Semaphore(1)`** for summarizer slot       | Eliminates contention with chat completions. Consolidator yields the slot between sessions, never holds across yields.   |
| Batch size               | **10 sessions per inner batch**                      | Small enough that any one batch finishes well under 30s on Apple Silicon, large enough to amortize WAL commit cost.       |
| Per-tick deadline        | **30 minutes** (`MEMORY_CONSOLIDATOR_DEADLINE_SEC=1800`) | Bounded enough to never collide with the next day's tick; sessions roll over cleanly.                                    |
| Embedding pipeline       | **Reuse MEMORY-01 embedder**                         | `chat_summaries` and `vault_themes` get embedded by the same `BAAI/bge-m3` pipeline; same `vec_chunks` virtual table.    |
| Ranking fusion           | **RRF (Reciprocal Rank Fusion), 4 streams**          | Already used in MEMORY-01 across BM25 + dense; extends naturally. No learned reranker — keep it deterministic and explainable. |
| Clustering for vault digest | **k-means on dense embeddings, k=ceil(N/12)**     | N is small (last 7 days of vault chunks); k-means is overkill but simple. Output is 1–N theme rows. Phase 2 may swap for HDBSCAN if quality is poor. |

### Alternatives considered

- **`apscheduler`** (BackgroundScheduler with `IntervalTrigger`):
  full-featured cron, but adds a dep, an extra thread, and persistence
  state we do not need. The scheduling we want is two lines:
  "next 03:00" and "next Sunday 03:00". Hand-rolled wins.
- **External cron / launchd**: works but breaks the "one process, no
  daemons" property of MEMORY-01. Rejected on architectural grounds.
- **Redis-backed queue** for resumability: massive overkill — SQLite +
  WAL gives us atomic per-session transactions for free.
- **Model quantization for the summarizer**: deferred. Self-hosted
  Qwen3.6 is "free" in our setup; if model contention starts hurting
  chat latency, the answer is REQ-O2 (swap to a smaller HF model).
- **HDBSCAN clustering for vault digests**: better quality than
  k-means for varying-density clusters, but adds a dep
  (`hdbscan`/`sklearn`). Defer to Phase 4 if k-means produces poor
  themes in real vaults.

### Risks if these picks are wrong

- Hand-rolled scheduler skews on DST / clock changes: the
  `compute_next_tick` helper must use timezone-aware datetimes and
  recompute on every iteration; covered by unit tests with
  `freezegun`.
- `MEMORY_SUMMARIZER_MODEL=self` may starve the chat path if a user
  fires consolidation manually during peak chat. The semaphore +
  deadline keep it bounded; observability metric in Phase 4 surfaces
  any p95 regression.
- k-means on 7 days of chunks may produce noisy themes if the user
  writes infrequently (e.g. one note per week → k=1). The fix is
  Phase 4 HDBSCAN, but the failure mode is benign — the vault still
  works without themes.

## Phase Breakdown

### Phase 1 — Chat session summarizer + raw eviction

Deliverables:
- `vllm_mlx/memory/scheduler.py`: hand-rolled asyncio loop, daily tick
  computation, integrated with `lifespan`.
- `vllm_mlx/memory/consolidator.py`: STM→LTM session selection,
  per-session summarize-then-evict transaction.
- `vllm_mlx/memory/summarizer.py`: prompt template, JSON output parsing,
  redaction filter pass, `MEMORY_SUMMARIZER_MODEL=self` resolution to
  the live engine, lazy-load path for HF model id alternatives.
- Extend `vllm_mlx/memory/store.py`: `insert_chat_summary`,
  `evict_chat_messages_for_session`, orphan-eviction sweep.
- Unit tests:
  - `tests/test_memory_summarizer.py`: prompt template, JSON parse,
    redaction integration, empty-output handling.
  - `tests/test_memory_consolidator.py`: session selection, idempotency,
    transactional rollback on simulated crash.
  - `tests/test_memory_scheduler.py`: tick computation under DST,
    enabled/disabled gates, `freezegun`-driven runs.
- Manual smoke: seed 5 sessions with timestamps 31+ days ago, run
  `MEMORY_CONSOLIDATOR_DRYRUN=1` and confirm summaries logged; flip
  to live mode and confirm rows summarized + evicted.

Out: vault digests, tier_filter, 4-stream RRF.

Effort: ~3 days.

### Phase 2 — Vault thematic digest

Deliverables:
- `vllm_mlx/memory/digest.py`: 7-day window selection, k-means cluster,
  per-cluster summarization with the digest prompt variant, write to
  `vault_themes`.
- Extend `scheduler.py` to fire the digest job on the configured
  weekday.
- Unit tests:
  - `tests/test_memory_digest.py`: idempotency for repeated weeks,
    `vault_chunks` untouched invariant, JSON array of `source_chunk_ids`
    correctness.
- Manual smoke: seed a week of vault notes, run the digest job,
  confirm `vault_themes` rows appear and `vault_chunks` row count
  is unchanged.

Effort: ~2 days.

### Phase 3 — Tier-aware ranking + tier_filter

Deliverables:
- Extend `vllm_mlx/memory/search.py`:
  - 4 candidate streams (vault STM, chat STM, chat_summary LTM,
    vault_theme LTM).
  - Per-stream weights from § 7 of spec.md.
  - `tier_filter` parameter routed from MCP tool args.
  - `source_type` enum extended with `chat_summary` and `vault_theme`.
- Update MCP tool schema in `vllm_mlx/memory/server.py` to expose
  `tier_filter` (default `both`).
- Unit tests:
  - `tests/test_memory_search_tiered.py`: weight sanity (vault > theme >
    chat_summary > chat_raw), `tier_filter='stm'` excludes summaries,
    `tier_filter='ltm'` excludes raw.

Effort: ~2 days.

### Phase 4 — Polish: metrics, retry, idempotency hardening

Deliverables:
- Metrics: a `/v1/memory/consolidator/stats` admin endpoint exposing
  last tick start/end, sessions processed, sessions skipped, summaries
  shorter than threshold, last error.
- Retry policy: failed sessions tracked in an in-memory backoff map
  (skip for N ticks before retrying), surfaced in stats endpoint.
- Idempotency hardening: orphan-eviction pass runs at every tick start
  (covers crash-after-summary-before-eviction).
- Docs: extend `docs/memory.md` with the consolidation lifecycle, env
  var reference, and operator runbook (manual run, dry-run, force).
- CLI: `vllm-mlx memory consolidate --once --dry-run` for ad-hoc runs.
- Coverage ≥ 85% on `vllm_mlx/memory/{scheduler,consolidator,summarizer,digest}.py`.

Effort: ~2 days.

Total rough effort: ~9 working days.

## Dependencies — additions to `pyproject.toml`

None.

The scheduler is hand-rolled. The summarizer reuses the live engine.
The clustering uses `numpy` (already pulled by `mlx-embeddings`).

If Phase 4 HDBSCAN clustering is later pursued, that adds `scikit-learn`
or `hdbscan`. Out of scope for MVP.

## Integration Points (existing files to touch)

| file                                  | what changes                                                       |
| ------------------------------------- | ------------------------------------------------------------------ |
| `vllm_mlx/server.py` (lifespan)       | Spawn `consolidator_loop` on startup if env says so                |
| `vllm_mlx/memory/server.py`           | MCP tool schema gains `tier_filter`; result envelope gains `chat_summary`/`vault_theme` source types |
| `vllm_mlx/memory/search.py`           | Add streams 3 and 4 to RRF fusion; tier_filter routing             |
| `vllm_mlx/memory/store.py`            | Add `insert_chat_summary`, `insert_vault_theme`, `evict_chat_session_rows`, `select_sessions_due_for_consolidation` |
| `vllm_mlx/cli.py`                     | Add `vllm-mlx memory consolidate [--once] [--dry-run]`             |
| `start-server.sh`                     | Document new env vars in comments                                  |
| `docs/memory.md`                      | New section: STM/LTM lifecycle, consolidator runbook               |

New files:

| file                                       | role                                                            |
| ------------------------------------------ | --------------------------------------------------------------- |
| `vllm_mlx/memory/scheduler.py`             | Hand-rolled asyncio scheduler                                   |
| `vllm_mlx/memory/consolidator.py`          | Daily chat consolidation job                                    |
| `vllm_mlx/memory/summarizer.py`            | Prompt + LLM call + JSON parse + redaction                      |
| `vllm_mlx/memory/digest.py`                | Weekly vault thematic digest                                    |
| `tests/test_memory_scheduler.py`           | Phase 1                                                         |
| `tests/test_memory_consolidator.py`        | Phase 1                                                         |
| `tests/test_memory_summarizer.py`          | Phase 1                                                         |
| `tests/test_memory_digest.py`              | Phase 2                                                         |
| `tests/test_memory_search_tiered.py`       | Phase 3                                                         |
| `tests/test_memory_consolidator_e2e.py`    | Phase 4 — full daily tick on a fixture DB                       |

## Risks & Mitigations

Top 3 risks (and the rest):

1. **Summary quality is too low** (model hallucinates / drops key facts)
   - Mitigation: `MEMORY_SUMMARY_MIN_CHARS` blocks eviction on degenerate
     output (REQ-N4). `MEMORY_CHAT_RETENTION_MODE=redact` lets the user
     keep the row's metadata while losing only the verbatim text — a
     reversible-ish posture during early adoption. Phase 4 dry-run mode
     lets the user manually inspect summaries before flipping to delete
     mode.

2. **Model contention with chat path** (consolidator slows /v1/chat)
   - Mitigation: `summarizer_slot` semaphore is held only by the
     consolidator. Chat path never blocks on it. The consolidator
     yields the slot between sessions and re-acquires; under load,
     chat wins every contest. Per-tick deadline (30 min) bounds
     worst-case stall.

3. **Eviction race** (raw deleted but summary write rolled back)
   - Mitigation: single transaction wraps both `INSERT chat_summaries`
     and `DELETE chat_messages WHERE session_id=?`. SQLite's atomicity
     guarantees both-or-neither. Plus the orphan-eviction sweep at
     every tick start handles the edge case where the summary
     succeeded but the eviction crashed.

Other risks:

| risk                                                          | mitigation                                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Hand-rolled scheduler drifts on DST                           | TZ-aware datetimes, recompute next tick every loop iteration, `freezegun` tests     |
| Long-running sessions (10k+ messages) blow input token budget | Recursive chunk-summarize until under 8000 tokens                                    |
| `MEMORY_SUMMARIZER_MODEL` set to a model that fails to load   | Try once at startup, fail closed (consolidator disabled for the day, chat unaffected)|
| Vault digest creates duplicate themes for the same week       | Window aligned to UTC midnight, idempotent insert keyed on `(period_start, period_end)` |
| User changes `MEMORY_STM_DAYS` mid-deployment                 | Threshold is read fresh per tick — change takes effect on next tick                  |
| User runs `--dry-run` then expects rows already gone          | Dry-run logs explicit "would have evicted N rows" lines; docs make this clear        |
| Large summary backlog after a long downtime                   | Per-tick deadline + roll-over; first few ticks chip away at backlog                  |
| sqlite-vec embedding insert lock contention                   | Same WAL pattern as MEMORY-01; small per-batch transactions                          |

## Out-of-scope (re-stated for clarity)

- Knowledge-graph extraction.
- Cross-device sync.
- Semantic deduplication across sessions.
- Re-summarization of LTM (summaries are immutable once written in MVP).
- Theme migration when `MEMORY_EMBED_MODEL` changes (manual rebuild only).
- Learned reranking (RRF stays).
- Auto-tuning of stream weights from user feedback.

## Expert Consultation

- **expert-backend**: primary author for `scheduler.py`,
  `consolidator.py`, `summarizer.py`, `digest.py`, store extensions,
  lifespan integration.
- **expert-testing**: parametrized scheduler tests with `freezegun`,
  transactional-rollback simulation, e2e fixture DB tests.
- **expert-performance**: bench the consolidator's impact on
  `/v1/chat/completions` p95 latency during a tick; tune semaphore
  yield points.
- **expert-security**: review summary-text redaction pipeline;
  confirm `MEMORY_CHAT_RETENTION_MODE=redact` actually blanks
  content rather than just hiding it.

## Open questions parked for run phase

- **OQ-1**: Should the summarizer call go through the same SSE/streaming
  path as `/v1/chat/completions`, or call the engine directly? (Default:
  call the engine directly with non-streaming inference — simpler and
  the consolidator does not need streaming.)
- **OQ-2**: What is the right `session_id` granularity? Per-request
  `request_id` is too fine; per-day-per-user is too coarse. (Default:
  decided in MEMORY-01 — `session_id` is whatever MEMORY-01's chat
  logger writes; we do not redefine it here.)
- **OQ-3**: Should `MEMORY_VAULT_DIGEST_DAY` accept a number 0-6 or a
  string? (Default: accept both; document the string form as canonical.)
- **OQ-4**: When `MEMORY_SUMMARIZER_MODEL=self` and the live engine is
  swapped at runtime (e.g. user reloads with a different model), do
  in-flight summaries belong to the old model or the new? (Default:
  next tick after a model swap — drop in-flight, restart cleanly.)
- **OQ-5**: Should we expose stream weights as env vars for tuning?
  (Default: hard-coded for MVP; if Phase 4 telemetry shows skew,
  introduce `MEMORY_RANKING_WEIGHTS` JSON env var.)
