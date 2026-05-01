---
id: SPEC-MEMORY-01
tags:
  - "@ACCEPT:MEMORY-01"
---

# Acceptance — MEMORY-01

## Definition of Done

All P1 criteria below pass on the `feature/spec-migrate-qwen36` branch (or its
successor) against a live Qwen3.6 server launched with `start-server.sh` and
`MEMORY_ENABLED=1`.

## Priority 1 (HARD)

### P1-AC1 — End-to-end Korean recall scenario

- **Given** a vault containing a note `Notes/2025-09-13 — distillation.md`
  with text discussing the user's opinion on knowledge distillation,
  **and** the server running with `MEMORY_ENABLED=1`,
  `MEMORY_VAULT_PATH=<vault>`, and `--auto-inject-mcp-tools` enabled,
- **When** the user sends a streaming chat completion with the message
  `"지난 달에 distillation에 대해 어떻게 생각했지?"`,
- **Then** the server log MUST show exactly one tool call to
  `memory__memory_search` (and zero calls to `web-search__tavily_search`),
  the tool result MUST contain at least one entry with
  `source_type == "vault"` and `source_path` ending in
  `Notes/2025-09-13 — distillation.md`, and the final assistant reply MUST
  cite that path verbatim somewhere in `delta.content`.

### P1-AC2 — Vault initial index throughput

- **Given** a fixture vault of 1000 generated `*.md` files (avg 1.5 KB each),
- **When** the server starts cold (no pre-existing DB),
- **Then** the indexer MUST complete the initial scan and embedding pass in
  ≤ 30 seconds on the test machine (Apple Silicon M-series, 16 GB RAM),
  **and** `GET /v1/mcp/status` MUST report `memory.ready=true` within that
  window.

### P1-AC3 — Incremental update latency

- **Given** the indexer is running with `MEMORY_ENABLED=1` and watching a vault,
- **When** a new `*.md` file is written under the vault root,
- **Then** within 5 seconds a subsequent `memory_search` query for a unique
  string from that file MUST return a hit with the new file's path.

### P1-AC4 — Chat persistence + recall

- **Given** `MEMORY_CHAT_LOG_ENABLED=1` and at least one prior chat where the
  user discussed "MoE active parameters",
- **When** in a fresh chat session the user asks "what did I conclude about
  MoE active parameters last time?" and the model emits a `memory_search`
  tool call,
- **Then** at least one returned result MUST have `source_type == "chat"`,
  its `timestamp` MUST be the timestamp of the prior chat row, and its
  `excerpt` MUST contain a paraphrase or quote from that prior conversation.

### P1-AC5 — Privacy opt-out is total

- **Given** `MEMORY_ENABLED=0` (default),
- **When** the server starts,
- **Then** no SQLite file under `MEMORY_DB_PATH` is created or modified,
  `GET /v1/mcp/tools` MUST NOT include `memory__memory_search`, and the
  startup log MUST NOT include any line beginning with `[memory]`. The
  vllm-mlx server MUST behave bit-for-bit identically to a build without
  this SPEC (REQ-S1).

### P1-AC6 — Failure isolation: vault path missing

- **Given** `MEMORY_ENABLED=1` but `MEMORY_VAULT_PATH` points to a
  non-existent directory,
- **When** the server starts,
- **Then** the server MUST start successfully (no crash, exit 0 from
  `start-server.sh` health check), `GET /v1/mcp/tools` MUST still include
  `memory__memory_search`, and a `memory_search` call MUST return
  `{"status": "ok", "results": [], "degraded": false}` with a server-side
  log line warning that the vault path is missing.

### P1-AC7 — Failure isolation: embedding model load fails

- **Given** a forced load failure of bge-m3 (e.g. `MEMORY_EMBED_MODEL` set
  to an invalid HF id),
- **When** the server starts and a `memory_search` is issued,
- **Then** the server MUST return results from FTS5 BM25 with
  `degraded: true` in the result envelope (REQ-O3), and `/v1/chat/completions`
  MUST continue to serve normal requests unaffected (REQ-N4).

### P1-AC8 — Search latency p95

- **Given** a 10,000-chunk vault index built with bge-m3,
- **When** 100 concurrent `memory_search` calls are issued with varied queries,
- **Then** the p95 wall-clock latency from MCP request to MCP response MUST
  be < 500 ms on the test machine.

### P1-AC9 — Redaction blocks secret-shaped content

- **Given** `MEMORY_CHAT_LOG_ENABLED=1` and a chat message containing the
  literal string `sk-ABCDEFGHIJKLMNOPQRSTUVWX1234567890`,
- **When** the chat row is persisted,
- **Then** the row's stored content MUST have that string replaced by
  `[REDACTED]`, and a subsequent `memory_search` with the literal API-key
  substring MUST NOT return that chat row's excerpt verbatim.

### P1-AC10 — Unit + integration test coverage

- `pytest tests/test_memory_*.py -v` MUST be all green.
- Coverage on `vllm_mlx/memory/*` MUST be ≥ 85% (line coverage).
- `ruff check vllm_mlx/memory/` MUST exit clean.
- `bash -n start-server.sh` MUST exit 0 with the new env vars documented in
  comments.

### P1-AC11 — TRUST 5 alignment

- **Tested**: P1-AC10 enforces ≥ 85% line coverage.
- **Readable**: All public functions in `vllm_mlx/memory/` have type hints
  and one-line docstrings.
- **Unified**: `ruff check vllm_mlx/memory/` clean, follows existing
  `vllm_mlx/mcp/` style (dataclasses, logger naming).
- **Secured**: REQ-N1, N2, N3 verified by P1-AC9 + symlink-escape unit test
  in `tests/test_memory_indexer.py::test_refuses_symlink_escape`.
- **Trackable**: Every code module carries `@CODE:MEMORY-01/<area>` tag in
  its header comment.

## Priority 2 (SOFT)

### P2-AC1 — `source_filter="vault"` honored

- **Given** both vault and chat sources have hits for a query,
- **When** the model passes `source_filter="vault"`,
- **Then** zero results with `source_type == "chat"` are returned.

### P2-AC2 — `time_range` honored

- **Given** the vault contains files with frontmatter dates spanning Jan-Dec,
- **When** the model passes `time_range={"after": "2025-09-01"}`,
- **Then** all returned results have `timestamp >= 2025-09-01`.

### P2-AC3 — Retention sweeper

- **Given** `MEMORY_CHAT_RETENTION_DAYS=30` and chat rows older than 30 days,
- **When** the retention sweeper runs (startup or 24h tick),
- **Then** with `MEMORY_CHAT_RETENTION_MODE=delete` the rows are removed,
  and with `=redact` the `excerpt` is replaced with a date stub.

### P2-AC4 — `vllm-mlx memory forget` CLI

- **Given** the CLI command `vllm-mlx memory forget --path 'Personal/**'`,
- **When** the command runs,
- **Then** all chunks whose `source_path` matches the glob are removed from
  both `vault_chunks` and `vec_chunks`, and a follow-up `memory_search` for
  unique strings from those files returns no hits.

## Test Commands

```bash
# Unit tests
pytest tests/test_memory_indexer.py tests/test_memory_chunker.py -v
pytest tests/test_memory_embedder.py tests/test_memory_store.py -v
pytest tests/test_memory_redact.py -v

# Integration / MCP tool surface
pytest tests/test_memory_server.py -v

# End-to-end (requires a real vault fixture + live server)
MEMORY_ENABLED=1 MEMORY_VAULT_PATH=tests/fixtures/vault_1000 \
  pytest tests/test_memory_e2e.py -v

# Coverage
pytest tests/test_memory_*.py \
  --cov=vllm_mlx.memory --cov-report=term-missing

# Lint + script syntax
ruff check vllm_mlx/memory/
bash -n start-server.sh
```

## Out of Scope (deferred)

- Hierarchical / MemGPT-style summarization.
- Knowledge-graph extraction.
- Auto-write-on-saliency.
- Memory curation UI.
- Auto-think gate (separate `SPEC-THINK-02`).
- Cross-device sync, at-rest encryption.
- Re-embedding migrations across embedding models (manual rebuild only).
