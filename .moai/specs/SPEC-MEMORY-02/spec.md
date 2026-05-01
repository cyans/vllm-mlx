---
id: SPEC-MEMORY-02
title: STM/LTM consolidation — daily summarizer cron, chat eviction, vault thematic digests
status: planned
priority: high
mode: ddd
lifecycle: spec-anchored
tags:
  - "@SPEC:MEMORY-02"
predecessors:
  - SPEC-MEMORY-01
related:
  - SPEC-MIGRATE-QWEN36
  - SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
---

# SPEC-MEMORY-02 — STM/LTM consolidation for chat history and vault digests

## 1. Problem

SPEC-MEMORY-01 ships a flat retrieval layer: every vault chunk and every
chat-completion row sits at `tier='stm'` forever. That works for a week but
breaks down on three fronts as the database grows:

- **Privacy**: every user message ever sent through `/v1/chat/completions`
  remains queryable verbatim, indefinitely. The only mitigation in MEMORY-01
  is the `MEMORY_CHAT_RETENTION_DAYS=365` sweeper, which simply deletes rows
  with no recall path. Either we lose the knowledge, or we keep the raw
  conversation indefinitely.
- **Search dilution**: as raw chat rows accumulate, RRF ranking over thousands
  of near-duplicate or low-value conversational utterances starts to crowd
  out high-signal vault notes. "What did I conclude about distillation?"
  retrieves five fragments of one Slack-style back-and-forth instead of the
  one paragraph that captured the conclusion.
- **No forgetting hierarchy**: humans remember last week verbatim and last
  year as themes. The system has no equivalent compression — it is all or
  nothing.

MEMORY-01 anticipated this and adopted **Path C** (forward-compat schema):
the `tier` column on `vault_chunks` and `chat_messages`, the `session_id`
column on `chat_messages`, the empty `chat_summaries` and `vault_themes`
tables, and the `MEMORY_STM_DAYS=30` env var all exist on disk today but are
unused. SPEC-MEMORY-02 turns that infrastructure on.

## 2. Goal

Formalize a two-tier memory model on top of the MEMORY-01 schema and operate
it through an in-process scheduler, with a hard asymmetry between user-authored
sources and machine-generated chat:

- **STM (Short-Term Memory)**: rows with `timestamp >= now() - MEMORY_STM_DAYS`
  (default 30 days). Stored verbatim. Highest ranking weight.
- **LTM (Long-Term Memory)**: rows with `timestamp < now() - MEMORY_STM_DAYS`.
  Behavior depends on source:
  - **Vault notes (user-authored 1차 자료)**: NEVER deleted, NEVER summarized
    away. The raw `vault_chunks` row stays at `tier='stm'` forever — the user
    wrote it, it is ground truth. LTM for the vault is purely **additive**: a
    weekly job *adds* thematic digest rows to `vault_themes` so that when a
    query matches a theme but no individual chunk strongly hits, the digest
    surfaces.
  - **Chat history (machine-mediated, ephemeral)**: a daily job consolidates
    every chat session that crosses the STM/LTM boundary into a single
    `chat_summaries` row, and then — gated by `MEMORY_CHAT_RETENTION_MODE` —
    either deletes or redacts the original `chat_messages` rows. Chat is
    explicitly subject to forgetting; the summary is the only ground truth
    that survives.

The summarizer is the **same Qwen3.6 instance already loaded by the server**.
Cost, in dollars and in dependencies, is zero. The architecture allows
swapping in a smaller model later via a single env var.

## 3. Scope

### In scope

- Daily chat-session consolidator: groups `chat_messages` by `session_id`,
  summarizes sessions whose newest message crosses the STM threshold, writes
  a `chat_summaries` row, then deletes (or redacts) the source rows.
- Weekly vault thematic digest job: clusters STM-window vault chunks (last 7
  days) and produces 1–N rows in `vault_themes`, leaving raw `vault_chunks`
  untouched.
- Tier-aware ranking: extends MEMORY-01's RRF fusion to four streams (STM
  vault chunks, LTM chat summaries, vault thematic digests, raw chat-message
  STM hits) with explicit per-stream weights.
- `tier_filter` argument on `memory_search` (`stm` | `ltm` | `both`).
- In-process asyncio scheduler in the same Python process as the FastAPI
  server. No external cron, no extra binaries.
- Idempotent and resumable jobs: a crash mid-batch leaves the database in a
  consistent state, never deletes a row whose summary failed to write.
- Concurrency lock between summarizer and chat path: chat completions are
  never blocked, summarizer defers when the model is busy.
- Configuration surface: new env vars layered on top of MEMORY-01's existing
  ones; no schema migration.

### Out of scope (deferred)

- Knowledge-graph extraction, entity linking, relation triples.
- Multi-device sync of the consolidated database.
- Semantic deduplication across sessions (e.g. detecting that two sessions
  discussed the same topic and merging summaries).
- Re-summarization of LTM (a summary, once written, is immutable in MVP).
- Theme migration when `MEMORY_EMBED_MODEL` changes (still a manual rebuild).
- Cross-tier reranking with a learned model — RRF stays the only fusion.

## 4. Architecture diff vs MEMORY-01

```
                     +--------------------------------------+
                     |  FastAPI server process              |
                     |                                      |
                     |  +-------------------------------+   |
                     |  | /v1/chat/completions          |   |
                     |  | (existing path, MEMORY-01)    |   |
                     |  +-------------------------------+   |
                     |                                      |
                     |  +-------------------------------+   |
                     |  | MCP memory_search             |   |
                     |  | (extended: tier_filter, 4    |   |
                     |  |  RRF streams)                 |   |
                     |  +-------------------------------+   |
                     |                                      |
                     |  +-------------------------------+   |
                     |  | Consolidator scheduler (NEW)  |   |
                     |  |  asyncio task, lazy hand-     |   |
                     |  |  rolled loop                  |   |
                     |  |                               |   |
                     |  |  daily 03:00  -> chat_consol  |   |
                     |  |  Sunday 03:00 -> vault_themes |   |
                     |  +---------------+---------------+   |
                     |                  |                   |
                     |                  v                   |
                     |  +-------------------------------+   |
                     |  | Summarizer (NEW)              |   |
                     |  |  - reads N chat sessions      |   |
                     |  |  - calls Qwen3.6 in-process   |   |
                     |  |  - writes chat_summaries      |   |
                     |  |  - deletes/redacts sources    |   |
                     |  +-------------------------------+   |
                     +--------------------------------------+

         tier mapping
         ============
         STM region:                       LTM region:
         +---------------------+           +-----------------------+
         | chat_messages       | --(>30d)->| chat_summaries        |
         |   tier='stm'        |  consol.  |   one per session     |
         |   raw text          |           |   raw rows DELETED    |
         +---------------------+           +-----------------------+

         +---------------------+           +-----------------------+
         | vault_chunks        | --(weekly)| vault_themes          |
         |   tier='stm' always |  digest   |   ADDITIVE only       |
         |   raw NEVER deleted | (no edit) |   raw NEVER touched   |
         +---------------------+           +-----------------------+

         memory_search ranking fusion (RRF, 4 streams):

           stream 1 (w=1.0):   vault_chunks (STM, raw)        <-- always
           stream 2 (w=0.6):   chat_messages (STM, raw)       <-- last 30d
           stream 3 (w=0.7):   chat_summaries (LTM, summary)  <-- > 30d
           stream 4 (w=0.8):   vault_themes (LTM, summary)    <-- weekly

         tier_filter overrides:
           'stm'  -> streams 1, 2 only
           'ltm'  -> streams 3, 4 only
           'both' (default) -> all four
```

The chat path and the consolidator never share a write transaction. The
consolidator opens its own SQLite connection in WAL mode and writes summaries
in small batches; the chat path keeps its existing connection. Reads from the
consolidator are non-blocking against writes from the chat path because of
WAL, and the summarizer's calls into Qwen3.6 use the same engine pool that
chat uses (with a concurrency=1 semaphore so it yields to chat).

## 5. EARS Requirements

### Ubiquitous (always active)

- **REQ-U1**: The consolidator SHALL ensure that every `chat_messages` row
  with `tier='stm'` whose `timestamp + MEMORY_STM_DAYS < now()` is either
  (a) summarized and removed, or (b) summarized and redacted, depending on
  `MEMORY_CHAT_RETENTION_MODE`.
- **REQ-U2**: Every `chat_summaries` row SHALL carry a non-null `session_id`,
  `period_start`, `period_end`, and `source_count` matching the source
  messages it consolidated. The mapping MUST be reconstructible from the
  audit log.
- **REQ-U3**: Every `vault_themes` row SHALL store a JSON array of the
  `vault_chunks.id` values that contributed to it, so any digest can be
  traced back to its underlying user-authored notes.
- **REQ-U4**: Embeddings of `chat_summaries` and `vault_themes` rows SHALL
  be produced by the same `MEMORY_EMBED_MODEL` configured in MEMORY-01,
  so they share the same `vec_chunks` virtual table.

### Event-driven (trigger → response)

- **REQ-E1** (chat consolidation trigger): WHEN the consolidator scheduler
  fires (default 03:00 local, configurable via `MEMORY_CONSOLIDATOR_HOUR`),
  the consolidator SHALL select all sessions whose newest message has
  `timestamp + MEMORY_STM_DAYS < now()` AND that have no existing row in
  `chat_summaries`, then process them in batches of `MEMORY_CONSOLIDATOR_BATCH`
  (default 10).
- **REQ-E2** (per-session summarization): WHEN a session is selected for
  summarization, the summarizer SHALL concatenate its messages in
  chronological order, call the configured summarizer model with the prompt
  template defined in §6, and write the result to `chat_summaries` in a
  single transaction.
- **REQ-E3** (raw eviction post-summary): WHEN a `chat_summaries` row has
  been written and verified non-empty (≥ `MEMORY_SUMMARY_MIN_CHARS`, default
  40), the consolidator SHALL evict the source rows according to
  `MEMORY_CHAT_RETENTION_MODE`: `delete` removes them, `redact` blanks
  `messages[*].content` to `[REDACTED-LTM]` while keeping the row's
  metadata for audit.
- **REQ-E4** (vault theme trigger): WHEN the vault digest scheduler fires
  (default Sunday 03:00, configurable via `MEMORY_VAULT_DIGEST_DAY`), the
  digest job SHALL cluster `vault_chunks` rows from the last 7 days
  (timestamp window only, regardless of tier) and produce 1–N
  `vault_themes` rows. It SHALL NOT modify or delete any `vault_chunks`
  row.
- **REQ-E5** (degraded summary): WHEN the summarizer call returns an
  output shorter than `MEMORY_SUMMARY_MIN_CHARS` or fails JSON parsing,
  the consolidator SHALL skip eviction for that session, log a warning,
  and retry on the next scheduler tick.

### State-driven (while X, do Y)

- **REQ-S1**: WHILE the consolidator job is running, the chat path
  (`/v1/chat/completions` and the SSE stream) SHALL continue to serve
  requests with no measurable latency increase (target: p95 latency
  during a consolidation run within 10% of p95 latency outside one).
- **REQ-S2**: WHILE the summarizer model (Qwen3.6 by default) is
  actively serving a `/v1/chat/completions` request, the consolidator
  SHALL defer summary work using a `concurrency=1` asyncio semaphore.
  Defer time SHALL be capped at the per-tick deadline; sessions that
  cannot be processed this tick SHALL be picked up next tick.
- **REQ-S3**: WHILE `MEMORY_CONSOLIDATOR_ENABLED=0` (or when
  `MEMORY_ENABLED=0`), the scheduler SHALL NOT spawn, and the database
  SHALL behave exactly as MEMORY-01 left it.

### Unwanted (shall NOT)

- **REQ-N1**: The consolidator SHALL NOT delete or modify any
  `vault_chunks` row, ever. Vault is user-authored 1차 자료.
- **REQ-N2**: The consolidator SHALL NOT delete or modify any
  `vault_files` row.
- **REQ-N3**: The consolidator SHALL NOT alter `vault_themes` rows once
  written. New digests append; old digests are immutable in MVP.
- **REQ-N4**: The consolidator SHALL NOT delete a `chat_messages` row
  whose corresponding `chat_summaries` row is missing or whose summary
  text is shorter than `MEMORY_SUMMARY_MIN_CHARS`.
- **REQ-N5**: The summary text SHALL NOT bypass MEMORY-01's
  `MEMORY_REDACT_PATTERNS`. Pattern matches in the summary output SHALL
  be replaced with `[REDACTED]` before the row is written.
- **REQ-N6**: The consolidator SHALL NOT make outbound network calls.
  Inherits MEMORY-01 REQ-U2.

### Optional (where possible)

- **REQ-O1**: WHERE the user passes `tier_filter` ∈ {`stm`, `ltm`,
  `both`} to `memory_search`, the server SHALL restrict ranking streams
  per the table in §4. Default is `both`.
- **REQ-O2**: WHERE `MEMORY_SUMMARIZER_MODEL` is set to a Hugging Face
  model id different from the live engine, the consolidator SHALL load
  that model lazily on first use, with the same lazy-load pattern as
  the embedder. This is the swap-out path for using a smaller model.
- **REQ-O3**: WHERE `MEMORY_CONSOLIDATOR_DRYRUN=1` is set, the job
  SHALL produce summary text and log it but SHALL NOT write to the DB
  and SHALL NOT evict any rows. Useful for CI and observability.

## 6. Summarization protocol

### System prompt (chat-session summarizer)

```
You are a memory consolidator. Summarize the following conversation between
a user and an assistant into a compact long-term memory record. Preserve
the original language of the user's messages — if the user wrote in Korean,
your summary stays in Korean.

Output a single JSON object with EXACTLY these keys:
  "key_conclusions": array of 1-5 short strings (≤ 200 chars each)
  "entities":        array of 0-10 short strings (people, projects, concepts)
  "unresolved":      array of 0-5 short strings (questions left open)
  "one_line":        single string ≤ 240 chars summarizing the whole session

Do not include any prose outside the JSON. Do not invent facts not present
in the source. If the conversation is empty or trivial, return all empty
arrays and an empty one_line.
```

### Per-call constraints

- Max input tokens (concatenated session): 8000. Sessions exceeding this
  cap are summarized in chunks; chunk summaries are themselves summarized
  recursively until under cap.
- Max output tokens: 500.
- Temperature: 0.2 (deterministic-leaning).
- Concatenation format: `<|user|>{content}\n<|assistant|>{content}\n` per
  message, in chronological order. Tool calls are flattened to their
  rendered text.

### Vault thematic digest variant

Same JSON schema as above, but the "session" is replaced with a cluster
of vault chunks from the last 7 days. The system prompt's first line
becomes:

```
You are a memory consolidator. Distill the following user notes from the
past week into thematic digests. The user is the author; preserve their
voice and language. Do not invent facts.
```

Each digest row in `vault_themes` produces its own JSON object via the
same schema, with `entities` listing the dominant topics of the cluster.

### Multi-message and multilingual handling

- Sessions are concatenated literally — the model handles ordering.
- Korean stays Korean, English stays English, mixed sessions stay mixed.
  No translation is requested.
- If `key_conclusions` is empty AND `one_line` is empty, the row's
  effective summary text is empty and REQ-N4 prevents eviction.

## 7. Tier-aware ranking

`memory_search` returns up to `top_k` results merged via Reciprocal Rank
Fusion across four streams. Each stream produces its own top-`top_k * 2`
candidate list independently (BM25 + dense, as in MEMORY-01), then the
fused stream applies per-stream weights:

```
RRF_score(item) = sum over streams s of:
                    weight[s] / (k_rrf + rank_in_stream[s](item))

with k_rrf = 60 (standard RRF constant)
```

Default weights:

| stream | source                                  | weight | notes                          |
| ------ | --------------------------------------- | ------ | ------------------------------ |
| 1      | `vault_chunks` STM (always tier='stm')  | 1.0    | user-authored, ground truth    |
| 2      | `chat_messages` STM (last 30 days)      | 0.6    | machine-mediated, recent       |
| 3      | `chat_summaries` LTM (>30 days)         | 0.7    | machine-distilled, older       |
| 4      | `vault_themes` LTM (weekly digests)     | 0.8    | machine-distilled vault themes |

Rationale: vault > themes > chat-summary > chat-raw. Themes outrank chat
summaries because they describe user-authored content. Raw chat outranks
nothing other than itself.

`tier_filter` overrides:
- `tier_filter='stm'` → drops streams 3 and 4. Useful for "what did I just
  say?" queries.
- `tier_filter='ltm'` → drops streams 1 and 2. Useful for "do I have any
  long-running themes about X?" queries.
- `tier_filter='both'` (default) → all four streams.

The result envelope's `source_type` field gains two new values:
`chat_summary` and `vault_theme`. MEMORY-01's `vault` and `chat` values
remain valid for streams 1 and 2.

## 8. Cron / scheduler design

### In-process asyncio scheduler

The scheduler is a single hand-rolled `asyncio.create_task` long-running
coroutine started during FastAPI's `lifespan` startup phase. Pseudo-code:

```
async def consolidator_loop():
    while True:
        next_tick = compute_next_tick(MEMORY_CONSOLIDATOR_HOUR)
        await asyncio.sleep((next_tick - now()).total_seconds())
        if not MEMORY_CONSOLIDATOR_ENABLED:
            continue
        try:
            await run_chat_consolidation()
            if today.weekday() == VAULT_DIGEST_DOW:
                await run_vault_digest()
        except Exception as e:
            log.exception("[memory-02] tick failed: %s", e)
```

Why hand-rolled, not `apscheduler`:
- Zero new dependency.
- Total LOC ~50, fully testable with `freezegun` or a mocked clock.
- No need for cron expressions; daily-at-hour and weekly-at-day are the
  only schedules.
- Resumability is trivial — the next tick re-queries the DB.

### Idempotency and resumability

- Chat consolidation: a session is "ready" when its newest message
  `timestamp + MEMORY_STM_DAYS < now()` AND no row in `chat_summaries`
  exists for that `session_id`. Crash mid-batch: on restart the next
  tick re-queries and resumes from where it left off because:
  - Sessions whose summary was written but rows not yet evicted are
    detected by a second pass that joins `chat_messages` to
    `chat_summaries` on `session_id` and evicts orphans.
  - Sessions whose summary was *not* written remain on the queue.
- Vault digests: each tick computes the previous 7-day window. If a
  digest already exists for that exact window, the job is a no-op.
  Window boundaries are aligned to UTC midnight to avoid duplicates.

### Lock contention with chat path

- Reads (consolidator pulling chat sessions): WAL allows concurrent reads
  alongside writes. No lock contention.
- Writes (consolidator inserting `chat_summaries`, deleting raw rows):
  short transactions, batch size = 10. The chat path's writes
  (chat-message inserts) are also short. WAL handles this natively.
- Model contention (the consolidator wants Qwen3.6, but so does the
  chat path): a process-global `asyncio.Semaphore(1)` named
  `summarizer_slot` is acquired by the consolidator. The chat path does
  NOT acquire this semaphore. If a chat request starts while the
  consolidator holds it, the consolidator releases between sessions
  and re-acquires. Net effect: the consolidator is slower but never
  blocks chat.
- Per-tick deadline: `MEMORY_CONSOLIDATOR_DEADLINE_SEC` (default 1800,
  i.e. 30 minutes). Sessions not processed within the deadline roll
  over to next tick.

## 9. Storage delta

This SPEC adds **zero** new tables and **zero** new columns. Everything
needed is already in MEMORY-01's schema (Path C):

| object                       | created by | first written by | role in MEMORY-02              |
| ---------------------------- | ---------- | ---------------- | ------------------------------ |
| `vault_chunks.tier`          | MEMORY-01  | MEMORY-01 (`'stm'`) | unchanged — vault stays STM forever |
| `chat_messages.tier`         | MEMORY-01  | MEMORY-01 (`'stm'`) | toggled to `'ltm'` then row deleted/redacted |
| `chat_messages.session_id`   | MEMORY-01  | MEMORY-01           | grouping key for consolidator |
| `chat_summaries`             | MEMORY-01  | MEMORY-02           | populated by daily job         |
| `vault_themes`               | MEMORY-01  | MEMORY-02           | populated by weekly job        |
| `meta.schema_version`        | MEMORY-01  | MEMORY-01           | bumped if anything changes — but nothing does in MVP |

Net DDL changes: none. The Path C decision in MEMORY-01 paid for itself.

## 10. Configuration surface

New env vars introduced by MEMORY-02 (layered on top of MEMORY-01's):

| env var                              | default                | meaning                                                         |
| ------------------------------------ | ---------------------- | --------------------------------------------------------------- |
| `MEMORY_CONSOLIDATOR_ENABLED`        | `1` if `MEMORY_ENABLED=1`, else `0` | Master switch for the scheduler                |
| `MEMORY_CONSOLIDATOR_HOUR`           | `3`                    | Hour-of-day (local time, 0-23) for the daily chat job          |
| `MEMORY_VAULT_DIGEST_DAY`            | `sun`                  | `mon`/`tue`/.../`sun` for the weekly vault digest               |
| `MEMORY_SUMMARIZER_MODEL`            | `self`                 | `self` = use the live Qwen3.6 engine; otherwise an HF model id  |
| `MEMORY_SUMMARY_MIN_CHARS`           | `40`                   | Eviction guard — summaries shorter than this block raw deletion |
| `MEMORY_CONSOLIDATOR_BATCH`          | `10`                   | Sessions processed per inner batch                              |
| `MEMORY_CONSOLIDATOR_DEADLINE_SEC`   | `1800`                 | Per-tick wall-clock cap (30 min)                                |
| `MEMORY_CONSOLIDATOR_DRYRUN`         | `0`                    | `1` = produce summaries, log them, but do not write or evict    |

Carried over from MEMORY-01 (already declared, now used for the first time):

| env var                          | default | role in MEMORY-02                                              |
| -------------------------------- | ------- | -------------------------------------------------------------- |
| `MEMORY_STM_DAYS`                | `30`    | The STM/LTM threshold — finally consumed by the consolidator   |
| `MEMORY_CHAT_RETENTION_MODE`     | `delete`| `delete` triggers post-summary eviction; `redact` blanks text  |
| `MEMORY_CHAT_RETENTION_DAYS`     | `365`   | Hard ceiling — even LTM summaries are pruned past this age     |

Note: `MEMORY_CHAT_RETENTION_MODE='redact'` keeps the `chat_messages` row's
metadata (request_id, timestamp, session_id) but blanks `messages[*].content`
to `[REDACTED-LTM]`. The summary remains queryable; the verbatim chat does
not. This is the privacy-vs-audit middle ground.

## 11. Failure modes (and required degradation)

| failure                                              | required behavior                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| Summarizer call fails for one session                | Skip that session, log warning, retry next tick (REQ-E5)                         |
| Summarizer returns malformed JSON                    | Treat as failure → skip session, retry next tick                                  |
| Summarizer returns empty / too-short text            | Block eviction (REQ-N4); session stays in STM until a valid summary lands        |
| Tick exceeds `MEMORY_CONSOLIDATOR_DEADLINE_SEC`      | Stop processing this tick gracefully, commit any complete batches, resume next tick |
| Consolidator crashes mid-batch                       | Transactional rollback — never delete a row whose summary was not committed       |
| `chat_summaries` write succeeds, eviction crashes    | On next tick, the orphan-eviction pass re-runs the eviction step                 |
| `MEMORY_SUMMARIZER_MODEL` load fails                 | Disable the consolidator for the day, log error, do not degrade chat path        |
| Disk full during summary write                       | Roll back transaction, log error, retry next tick                                 |
| Volume disconnect (`/Volumes/data` unmounted)        | Same as MEMORY-01 §10: consolidator becomes a no-op until reconnect              |
| Vault digest job crashes                             | No raw vault data is touched (REQ-N1, N2); next Sunday tick retries cleanly      |

## 12. Privacy posture

The summarizer is an LLM that may hallucinate. The original raw text is the
only ground truth. Therefore:

- A summary MUST be verified non-empty AND ≥ `MEMORY_SUMMARY_MIN_CHARS`
  before any raw row is deleted (REQ-N4).
- The summary SHALL be passed through the same `MEMORY_REDACT_PATTERNS`
  filter as MEMORY-01's pre-write redaction, applied to the summary text
  before write (REQ-N5). A model that absorbs an API key into its summary
  does not leak it.
- `MEMORY_CHAT_RETENTION_MODE='redact'` is the recommended setting for
  users who want auditability without verbatim retention; `'delete'` is
  the recommended setting for maximal privacy.
- Summaries are themselves subject to `MEMORY_CHAT_RETENTION_DAYS` — past
  the hard ceiling (default 365 days), even summaries are removed. This
  caps total retention at one year by default.
- Vault digests describe user-authored content the user already owns;
  no new privacy surface beyond the original notes.
- Inherits all of MEMORY-01's privacy properties: local-only, no
  telemetry, opt-in.

## 13. Traceability

- `@SPEC:MEMORY-02` / `@PLAN:MEMORY-02` / `@ACCEPT:MEMORY-02` — this directory
- `@CODE:MEMORY-02/scheduler` — `vllm_mlx/memory/scheduler.py` (new)
- `@CODE:MEMORY-02/consolidator` — `vllm_mlx/memory/consolidator.py` (new)
- `@CODE:MEMORY-02/summarizer` — `vllm_mlx/memory/summarizer.py` (new)
- `@CODE:MEMORY-02/digest` — `vllm_mlx/memory/digest.py` (new)
- `@CODE:MEMORY-02/search` — extension of `vllm_mlx/memory/search.py` (4-stream RRF + tier_filter)
- `@CODE:MEMORY-02/store` — extension of `vllm_mlx/memory/store.py` (summary inserts, eviction)
- `@TEST:MEMORY-02` — `tests/test_memory_consolidator_*.py` (new)
- **Predecessor**: SPEC-MEMORY-01 supplies the schema (Path C) and the
  embedder/store layer this SPEC composes on top of.

## 14. Success metric

A user types in Korean: "지난 달에 distillation에 대해 어떻게 생각했지?"
where the relevant chat session is now 31 days old.

- The verbatim conversation no longer appears in `chat_messages` (it has
  been deleted, with `MEMORY_CHAT_RETENTION_MODE='delete'`).
- `memory_search` returns:
  - One `chat_summary` row with `source_type='chat_summary'` whose
    `excerpt` contains the model's distilled "key_conclusions" from
    that session, citing `session_id` and `period_start` / `period_end`.
  - Possibly one `vault_theme` row from a recent Sunday digest if the
    topic also appeared in vault notes that week.
  - Plus any current vault chunks that match (`tier='stm'`).
- The model's answer to the user's question grounds itself on the
  summary, with a citation pointing back to the session's date range
  rather than verbatim quotes.

A second success scenario: the user asks "what themes have I been
exploring lately?" with `tier_filter='ltm'`. The server returns recent
`vault_themes` rows even though no individual chunk strongly matched
any single word in the query. Without LTM, this query would have
returned weak random hits or nothing.
