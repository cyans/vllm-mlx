---
id: SPEC-MEMORY-02
tags:
  - "@ACCEPT:MEMORY-02"
---

# Acceptance — MEMORY-02

## Definition of Done

All P1 criteria below pass on the `feature/spec-migrate-qwen36` branch
(or its successor) against a live Qwen3.6 server launched with
`start-server.sh` and the consolidator enabled
(`MEMORY_ENABLED=1 MEMORY_CONSOLIDATOR_ENABLED=1`). The MEMORY-01 schema
must already be deployed; this SPEC introduces no migration.

## Priority 1 (HARD)

### P1-AC1 — Chat row older than STM threshold is summarized and evicted

- **Given** `MEMORY_ENABLED=1`, `MEMORY_CHAT_LOG_ENABLED=1`,
  `MEMORY_CONSOLIDATOR_ENABLED=1`, `MEMORY_STM_DAYS=30`,
  `MEMORY_CHAT_RETENTION_MODE=delete`, and at least one full chat
  session in `chat_messages` whose newest row has
  `timestamp = now() - 31 days`,
- **When** the daily consolidator tick fires (or
  `vllm-mlx memory consolidate --once` is invoked),
- **Then** within one tick:
  1. A new row in `chat_summaries` exists with that session's
     `session_id`, `period_start` and `period_end` matching the source
     timestamps, `source_count` matching the deleted row count, and
     `summary_text` of length ≥ `MEMORY_SUMMARY_MIN_CHARS`.
  2. All `chat_messages` rows for that `session_id` are gone from the
     table.
  3. The summary's embedding is queryable via the same `vec_chunks`
     virtual table as MEMORY-01 vault chunks.

### P1-AC2 — Summary appears in `memory_search` results

- **Given** the post-AC1 state (summary written, raw rows evicted),
- **When** the user asks Qwen3.6 a question whose topic matches that
  session, and the model emits a `memory_search` tool call with
  `tier_filter='both'` (default) or `tier_filter='ltm'`,
- **Then** the result envelope MUST contain at least one entry with
  `source_type == "chat_summary"`, `timestamp` falling within
  `[period_start, period_end]`, and `excerpt` derived from the summary's
  `key_conclusions` or `one_line` field.

### P1-AC3 — Vault chunks are NEVER deleted, ever

- **Given** a vault note `Notes/Old/2020-01-01.md` whose timestamp is 5
  years old, present in `vault_chunks` from MEMORY-01's initial scan,
- **When** the consolidator has run for 90 consecutive days (or the
  test simulates 90 days of ticks via `freezegun`),
- **Then**:
  1. The row in `vault_chunks` for that file MUST still exist.
  2. Its `tier` column MUST still be `'stm'`.
  3. A `memory_search` for a unique string from that file MUST return
     it verbatim with `source_type == "vault"`.
  4. No `vault_chunks` row in the entire database has been deleted.

### P1-AC4 — Vault thematic digest produced and discoverable

- **Given** the vault has accumulated ≥ 7 days of new chunks across at
  least 5 distinct files in the last week, and the configured digest
  weekday has elapsed,
- **When** the weekly digest job runs,
- **Then**:
  1. At least one row in `vault_themes` exists with `period_start` and
     `period_end` covering the previous 7-day window aligned to UTC
     midnight, and `source_chunk_ids` listing the contributing chunk
     IDs.
  2. A `memory_search` with `tier_filter='ltm'` for a topical query
     that does not strongly match any individual chunk MUST return at
     least one entry with `source_type == "vault_theme"`.

### P1-AC5 — `tier_filter='stm'` returns only STM rows

- **Given** the database contains both STM (vault chunks, recent chat
  messages) and LTM (chat summaries, vault themes) rows,
- **When** a `memory_search` is issued with `tier_filter='stm'`,
- **Then** zero results MUST have `source_type` equal to
  `"chat_summary"` or `"vault_theme"`. All results MUST be `"vault"`
  or `"chat"`.

### P1-AC6 — Consolidator failure for one session does not block the batch

- **Given** a batch of 10 sessions due for consolidation, where session
  #4's text is engineered to make the summarizer return malformed JSON
  (e.g. by injecting a control character that breaks JSON parsing),
- **When** the consolidator processes the batch,
- **Then**:
  1. Sessions #1–#3 and #5–#10 are summarized, written to
     `chat_summaries`, and (with `MODE=delete`) their raw rows
     evicted.
  2. Session #4's raw rows MUST still be present in `chat_messages`
     (REQ-N4).
  3. A warning log line MUST identify session #4 by `session_id`.
  4. The next tick re-attempts session #4 (if its content is unchanged
     it fails again — that is acceptable, the row is preserved).

### P1-AC7 — Summarizer respects redaction patterns

- **Given** `MEMORY_REDACT_PATTERNS=sk-[A-Za-z0-9]{20,};password\s*[:=]`
  and a chat session containing
  `My API key is sk-ABCDEFGHIJKLMNOPQRSTUVWX1234567890.`,
- **When** the consolidator summarizes that session,
- **Then** the resulting `chat_summaries.summary_text` MUST NOT contain
  the literal string `sk-ABCDEFGHIJKLMNOPQRSTUVWX1234567890`. Any
  occurrence MUST be replaced with `[REDACTED]`. A subsequent
  `memory_search` with the literal API-key substring MUST NOT return
  the summary's excerpt verbatim.

### P1-AC8 — Server still serves `/v1/chat/completions` during a long consolidation run

- **Given** a backlog of 100 sessions due for consolidation
  (`MEMORY_CONSOLIDATOR_BATCH=10`, expected wall-clock ~10 batches),
- **When** the consolidator is processing this backlog AND a sustained
  load of 1 req/sec is sent to `/v1/chat/completions` from a separate
  client,
- **Then**:
  1. The chat client MUST observe zero failed requests.
  2. The chat client's p95 latency during the consolidation MUST be
     within 25% of its p95 latency measured with the consolidator
     idle. (REQ-S1 says 10%; we allow 25% in this AC as the realistic
     bound, since model contention is real even with the semaphore.)
  3. The consolidator MUST eventually drain the backlog within
     `MEMORY_CONSOLIDATOR_DEADLINE_SEC` rolled over across at most 2
     ticks.

### P1-AC9 — Idempotency: re-running consolidation is a no-op

- **Given** the post-AC1 state (a session has been consolidated and
  evicted),
- **When** `vllm-mlx memory consolidate --once` is invoked again,
- **Then**:
  1. Zero new rows are written to `chat_summaries` (no duplicate
     summary for the same `session_id`).
  2. Zero rows are deleted from `chat_messages` (the raw rows are
     already gone).
  3. The job exit code is 0 and stats report 0 sessions processed.

### P1-AC10 — Crash recovery: orphan eviction handles partial failures

- **Given** a simulated crash between `INSERT chat_summaries` (committed)
  and `DELETE chat_messages` (interrupted) — i.e. the summary exists
  but the raw rows still exist for the same `session_id`,
- **When** the next consolidator tick starts,
- **Then** the orphan-eviction sweep at tick start MUST detect this
  state and complete the eviction. The end state matches AC1's
  end state.

### P1-AC11 — Unit + integration test coverage

- `pytest tests/test_memory_consolidator*.py tests/test_memory_summarizer*.py tests/test_memory_digest*.py tests/test_memory_search_tiered*.py tests/test_memory_scheduler*.py -v` MUST be all green.
- Coverage on `vllm_mlx/memory/{scheduler,consolidator,summarizer,digest}.py` MUST be ≥ 85% (line coverage).
- `ruff check vllm_mlx/memory/` MUST exit clean.

### P1-AC12 — TRUST 5 alignment

- **Tested**: P1-AC11 enforces ≥ 85% line coverage on the new modules.
- **Readable**: All public functions in
  `vllm_mlx/memory/{scheduler,consolidator,summarizer,digest}.py` have
  type hints and one-line docstrings explaining the invariant they
  preserve.
- **Unified**: `ruff check vllm_mlx/memory/` clean, follows existing
  `vllm_mlx/memory/` style established by MEMORY-01 (dataclasses,
  logger naming, async patterns).
- **Secured**: REQ-N1 (vault never deleted) verified by P1-AC3.
  REQ-N5 (redaction in summaries) verified by P1-AC7. REQ-N4 (no
  eviction without verified summary) verified by P1-AC6.
- **Trackable**: Every new module carries `@CODE:MEMORY-02/<area>` tag
  in its header comment; every new test carries `@TEST:MEMORY-02`.

## Priority 2 (SOFT)

### P2-AC1 — Configurable summary prompt

- **Given** an alternative system prompt provided via a config file or
  env var (TBD in run phase, e.g. `MEMORY_SUMMARIZER_PROMPT_FILE`),
- **When** the consolidator runs,
- **Then** the alternative prompt is used instead of the default in
  spec.md § 6, and summaries reflect the alternative instructions.
  (Out of scope for MVP if the run-phase team decides this is
  premature; default prompt is sufficient.)

### P2-AC2 — Alternate summarizer model swap

- **Given** `MEMORY_SUMMARIZER_MODEL` set to a smaller HF model id
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`),
- **When** the server starts and the consolidator tick fires,
- **Then** the summarizer loads the alternate model lazily on first
  use, summaries are produced, and the live Qwen3.6 chat engine is
  not contended (the smaller model runs alongside).

### P2-AC3 — Retry policy for repeatedly failing sessions

- **Given** a session whose summarization has failed N times across
  ticks,
- **When** the consolidator processes a batch,
- **Then** the failing session is skipped for an exponential backoff
  number of ticks before retry (e.g. 1, 2, 4, 8 ticks), surfaced in
  the `/v1/memory/consolidator/stats` endpoint.

### P2-AC4 — `MEMORY_CHAT_RETENTION_MODE=redact` keeps audit metadata

- **Given** `MEMORY_CHAT_RETENTION_MODE=redact`,
- **When** a session is consolidated,
- **Then**:
  1. The `chat_messages` rows for that session still exist.
  2. Their `messages[*].content` is `[REDACTED-LTM]`.
  3. Their `request_id`, `timestamp`, `session_id`, `model`,
     `latency_ms` are unchanged.
  4. The `chat_summaries` row is queryable as in AC2.

## Test Commands

```bash
# Unit tests
pytest tests/test_memory_summarizer.py -v
pytest tests/test_memory_consolidator.py -v
pytest tests/test_memory_scheduler.py -v
pytest tests/test_memory_digest.py -v
pytest tests/test_memory_search_tiered.py -v

# Integration / end-to-end (fixture DB + simulated time via freezegun)
pytest tests/test_memory_consolidator_e2e.py -v

# Coverage
pytest tests/test_memory_consolidator*.py tests/test_memory_summarizer*.py \
       tests/test_memory_digest*.py tests/test_memory_search_tiered*.py \
       tests/test_memory_scheduler*.py \
  --cov=vllm_mlx.memory.scheduler \
  --cov=vllm_mlx.memory.consolidator \
  --cov=vllm_mlx.memory.summarizer \
  --cov=vllm_mlx.memory.digest \
  --cov-report=term-missing

# Lint
ruff check vllm_mlx/memory/

# Manual smoke (requires MEMORY-01 deployed and chat history > 30 days old)
MEMORY_ENABLED=1 MEMORY_CHAT_LOG_ENABLED=1 \
MEMORY_CONSOLIDATOR_ENABLED=1 MEMORY_CONSOLIDATOR_DRYRUN=1 \
  vllm-mlx memory consolidate --once

# After verifying dry-run output, flip to live
MEMORY_ENABLED=1 MEMORY_CHAT_LOG_ENABLED=1 \
MEMORY_CONSOLIDATOR_ENABLED=1 \
  vllm-mlx memory consolidate --once
```

## Out of Scope (deferred)

- Knowledge-graph extraction, entity linking.
- Multi-device sync of consolidated state.
- Semantic deduplication across sessions.
- Re-summarization of LTM rows (summaries are immutable in MVP).
- Theme migration when `MEMORY_EMBED_MODEL` changes.
- Learned reranking — RRF stays for MVP.
- Auto-tuning of stream weights from user feedback.
