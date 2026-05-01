---
id: SPEC-MEMORY-01
title: Long-term semantic memory for Qwen3.6 via MCP (vault + chat history)
status: planned
priority: high
mode: ddd
lifecycle: spec-anchored
tags:
  - "@SPEC:MEMORY-01"
predecessors:
  - SPEC-MIGRATE-QWEN36
  - SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
successors:
  - SPEC-MEMORY-02
related:
  - SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE
---

# SPEC-MEMORY-01 — Long-term semantic memory for Qwen3.6 via MCP

## 1. Problem

Qwen3.6 running on vllm-mlx has no persistent memory of:
- The user's Obsidian vault notes (years of personal knowledge in Markdown).
- Prior chat completions made through this server (everything the user has ever
  asked the model is forgotten the moment a session ends).

Today the model already calls `web-search__tavily_search` via the MCP tavily
server, and a streaming-safe tool-call path landed in
`SPEC-FIX-QWEN36-TOOL-CALL-STREAMING`. We can re-use that exact infrastructure
to expose a second MCP server — a **memory** server — that gives the model a
single tool: `memory_search`. The model decides when to call it; the server
returns ranked snippets with citations.

## 2. Goal

Provide a **read-mostly, retrieval-only** semantic memory MCP server that:
- Indexes the user's Obsidian vault on startup and on file changes.
- Persists every `/v1/chat/completions` request/response (opt-in).
- Exposes a `memory_search` tool callable by Qwen3.6 via the same OpenAI tool
  protocol the Obsidian plugin already speaks.
- Runs entirely locally on Apple Silicon (MLX), no network egress, no telemetry.

## 3. Scope

### In scope (MVP)

- Vault indexer: initial scan of all `*.md` files under a configurable path,
  incremental updates on file changes, deletion handling.
- Chat-log persistence: structured per-request rows including user message,
  assistant message, tool calls, model name, timestamp, latency.
- Embedding pipeline: chunking by Markdown headers + sliding-window fallback,
  batched embedding generation, deterministic chunk IDs.
- Vector store: persistent on-disk store with metadata filters (source,
  time range, path prefix).
- MCP memory server: exactly one tool, `memory_search`, returning ranked
  snippets with `source_path`, `timestamp`, `score`, and a short `excerpt`.
- Server integration: register the memory MCP server alongside `web-search`
  in `mcp.json`, compatible with `--auto-inject-mcp-tools`.
- Configurability: env-var-driven paths, top-K, opt-out, allow/deny lists.
- Failure tolerance: every dependency failure shall produce a degraded but
  functional response — never a server crash.

### Out of scope (deferred to follow-up SPECs)

- Hierarchical / MemGPT-style summarization (`SPEC-MEMORY-02`).
- Knowledge-graph extraction or entity linking.
- Auto-write-on-saliency (auto-saving facts the model "learns").
- Human-in-the-loop memory curation UI.
- Auto-think gate (separate `SPEC-THINK-02`).
- Cross-device sync, encryption-at-rest beyond OS file permissions.
- Re-embedding migrations across embedding models (manual rebuild only).

## 4. Constitution Alignment

Verified against `.moai/project/tech.md` patterns and current dependencies:

- Python ≥ 3.10 (matches `pyproject.toml` `requires-python`).
- MLX-first: embedding model loads through `mlx-embeddings` (already a
  declared dependency at line 62 of `pyproject.toml`). No PyTorch/CUDA path.
- MCP-first: memory server registered through `mcp.json`, follows the same
  `stdio` transport contract used by `web-search` today.
- No new daemons: vector store is an in-process SQLite database.
- No telemetry: zero outbound network calls from the memory server.
- All instruction docs in English; runtime logs follow existing logger style.

## 5. Architecture (high level)

```
+------------------+          +------------------------+
| Obsidian vault   |  watch   |  Vault indexer          |
| (~/Documents/... | -------> |  (mtime + checksum)     |
|  *.md files)     |          +-----------+-------------+
+------------------+                      |
                                          v
                              +------------------------+
                              |  Chunker (markdown HX  |
                              |  + 512-token windows)  |
                              +-----------+------------+
                                          |
                                          v
                              +------------------------+
                              |  Embedder              |
                              |  BAAI/bge-m3           |
                              |  via mlx-embeddings    |
                              +-----------+------------+
                                          |
+------------------+                      v
| /v1/chat/        |              +---------------+
| completions      | ----+        | Vector store  |
| (request hook)   |     |        | sqlite-vec    |
+------------------+     |        | + FTS5        |
                         |        +-------+-------+
                         v                |
              +------------------+        |
              | Chat logger      |--------+
              | (sqlite, opt-in) |
              +------------------+        v
                                  +---------------+
                                  | MCP memory    |
                                  | server        |
                                  | tool:         |
                                  | memory_search |
                                  +-------+-------+
                                          |
                                          v
                                  Qwen3.6 (tool call)
```

Boundary rules:
- The memory server is the **only** writer of search results back to the model.
- The chat logger is the **only** writer of conversation rows to the DB.
- The vault indexer is the **only** writer of vault rows to the DB.
- All three writers share the same SQLite file via WAL mode for safe concurrency.

## 6. EARS Requirements

### Ubiquitous (always active)

- **REQ-U1**: The memory server SHALL expose exactly one MCP tool named
  `memory_search` whose JSON schema is registered through the existing
  `MCPManager` so it is discoverable at `GET /v1/mcp/tools` and injectable
  via `--auto-inject-mcp-tools`.
- **REQ-U2**: All vault notes and chat-log rows SHALL be stored on the local
  filesystem only. The memory server SHALL NOT make outbound network calls.
- **REQ-U3**: Every `memory_search` result SHALL include `source_path`,
  `source_type` (one of `vault` | `chat`), `timestamp` (ISO-8601), `score`
  (float in `[0,1]`), and `excerpt` (≤ 500 chars).
- **REQ-U4**: All embeddings SHALL be produced by exactly one configured
  embedding model per database. Mixing models in one DB is forbidden; the
  index SHALL refuse to start when a model-mismatch is detected.

### Event-driven (trigger → response)

- **REQ-E1** (vault initial index): WHEN the server starts and
  `MEMORY_VAULT_PATH` is set and points to an existing directory, the
  indexer SHALL enumerate every `*.md` file under that path and produce
  embeddings for each chunk before declaring readiness on
  `GET /v1/mcp/status`.
- **REQ-E2** (vault incremental): WHEN a `*.md` file under the watched path
  is created, modified, or deleted, the indexer SHALL update the corresponding
  chunks in the vector store within 5 seconds of the OS notification.
- **REQ-E3** (chat persistence): WHEN `MEMORY_CHAT_LOG_ENABLED=1` and a
  `/v1/chat/completions` request completes (non-streaming) or the SSE stream
  closes (streaming), the chat logger SHALL persist a row containing
  `request_id`, `timestamp`, `model`, `messages`, `assistant_text`,
  `tool_calls`, `latency_ms`.
- **REQ-E4** (chat indexing): WHEN a chat row is persisted, the embedder
  SHALL produce embeddings for the user's last message and the assistant's
  reply, attach them to the chat row, and make them queryable via
  `memory_search` within 10 seconds.
- **REQ-E5** (tool call): WHEN Qwen3.6 emits a `memory_search` tool call,
  the server SHALL return between 0 and `top_k` results (default 5, max 20)
  ranked by descending score.

### State-driven (while X, do Y)

- **REQ-S1**: WHILE `MEMORY_ENABLED=0`, the memory MCP server SHALL NOT
  register, the chat logger SHALL NOT write, and the vault indexer SHALL
  NOT scan. The rest of the vllm-mlx server SHALL behave bit-for-bit
  identically to a build without this SPEC.
- **REQ-S2**: WHILE the embedding model is still warming up (first load
  after process start), `memory_search` SHALL return an empty result list
  with a tool-result message field `status: "warming_up"` rather than
  blocking the model.
- **REQ-S3**: WHILE the vector store file is locked by another writer,
  `memory_search` SHALL retry up to 3 times with exponential backoff (50ms,
  150ms, 450ms) before returning an empty result with `status: "busy"`.

### Unwanted (shall NOT)

- **REQ-N1**: The memory server SHALL NOT expose, log, or return chat rows
  whose `messages[*].content` matches any pattern in
  `MEMORY_REDACT_PATTERNS` (default: `sk-[A-Za-z0-9]{20,}`,
  `(?i)password\s*[:=]`, `(?i)api[_-]?key\s*[:=]`).
- **REQ-N2**: The vault indexer SHALL NOT index files outside the configured
  vault root, even if symlinks point elsewhere.
- **REQ-N3**: The vault indexer SHALL NOT index files matching any pattern
  in `MEMORY_VAULT_DENYLIST` (default: `.obsidian/**`, `**/.trash/**`,
  `**/Templates/**`).
- **REQ-N4**: A failure in the memory subsystem SHALL NOT propagate to
  `/v1/chat/completions`. Failures SHALL be logged and the chat path SHALL
  continue without memory.
- **REQ-N5**: The memory server SHALL NOT return raw chat content from any
  conversation older than `MEMORY_CHAT_RETENTION_DAYS` (default 365). Such
  rows SHALL be either deleted or returned with the `excerpt` field redacted
  to a date stub, depending on `MEMORY_CHAT_RETENTION_MODE`
  (`delete` | `redact`).

### Optional (where possible)

- **REQ-O1**: WHERE the user passes `source_filter` in the tool arguments
  (one of `vault`, `chat`, or both), the memory server SHALL restrict results
  to that source.
- **REQ-O2**: WHERE the user passes `time_range` (`{after: ISO, before: ISO}`),
  the memory server SHALL filter results by `timestamp` accordingly.
- **REQ-O3**: WHERE the embedding model load fails at startup, the server
  SHALL fall back to BM25 (SQLite FTS5) keyword search. Tool results SHALL
  carry a `degraded: true` flag.

## 7. Tool surface — `memory_search`

JSON schema returned at `GET /v1/mcp/tools`:

```json
{
  "type": "function",
  "function": {
    "name": "memory__memory_search",
    "description": "Search the user's long-term memory (Obsidian vault notes and prior conversations). Returns ranked snippets with citations. Use when the user references past notes, prior conversations, or asks 'do you remember...'.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural-language query. May be in any language."
        },
        "top_k": {
          "type": "integer",
          "minimum": 1,
          "maximum": 20,
          "default": 5
        },
        "source_filter": {
          "type": "string",
          "enum": ["vault", "chat", "both"],
          "default": "both"
        },
        "time_range": {
          "type": "object",
          "properties": {
            "after":  {"type": "string", "format": "date-time"},
            "before": {"type": "string", "format": "date-time"}
          }
        }
      },
      "required": ["query"]
    }
  }
}
```

Result envelope (returned as the `tool` message content):

```json
{
  "status": "ok",
  "degraded": false,
  "results": [
    {
      "source_type": "vault",
      "source_path": "Notes/Reading/2025-09-13 — distillation.md",
      "timestamp": "2025-09-13T11:42:00Z",
      "score": 0.83,
      "excerpt": "I think distillation only works when the teacher's logits..."
    }
  ]
}
```

## 8. Storage schema

Single SQLite database at `MEMORY_DB_PATH` (default
`/Volumes/data/vllm-mlx-memory/memory.db` — co-located with the vault on the
external volume so backup of `/Volumes/data` covers both raw notes and memory
state). Schema is forward-compatible with the consolidation logic that
SPEC-MEMORY-02 will add: `tier` columns and the two summary tables exist from
day 1, but MVP only ever writes `tier='stm'` rows and leaves the summary
tables empty. This avoids a schema migration when SPEC-MEMORY-02 lands.

Tables:

| table             | purpose                                         |
| ----------------- | ----------------------------------------------- |
| `meta`            | `embedding_model`, `dim`, `schema_version`      |
| `vault_files`     | one row per indexed file (path, mtime, sha256)  |
| `vault_chunks`    | one row per chunk (file_id, header, text, ts, **tier**) |
| `chat_messages`   | one row per request (request_id, role, payload, ts, **tier**, **session_id**) |
| `chat_summaries`  | (forward-compat, empty in MVP) one row per consolidated session — `session_id`, `summary_text`, `period_start`, `period_end`, `source_count` |
| `vault_themes`    | (forward-compat, empty in MVP) thematic digest — `theme_id`, `summary_text`, `period_start`, `period_end`, `source_chunk_ids` (JSON array) |
| `vec_chunks`      | sqlite-vec virtual table (chunk_id, embedding) |
| `fts_chunks`      | FTS5 virtual table (text)                      |

Column semantics (Path C forward-compat):
- `tier` enum: `'stm'` | `'ltm'`. MVP indexer always writes `'stm'`. SPEC-MEMORY-02
  will introduce a daily job that re-stamps rows older than `MEMORY_STM_DAYS`
  to `'ltm'` after producing the corresponding summary row.
- `session_id` on `chat_messages`: groups messages of one conversation so the
  consolidation job can produce one `chat_summaries` row per session.
- `chat_summaries` and `vault_themes` are also indexed by `vec_chunks` /
  `fts_chunks` (they get their own chunk IDs in the same vector store) so
  `memory_search` retrieves them transparently when SPEC-MEMORY-02 lands.

`schema_version` is checked on startup; mismatch triggers a hard error with a
documented `vllm-mlx memory rebuild` recovery path (manual, not automated in
MVP).

## 9. Configuration surface

| env var                       | default                                  | meaning                                      |
| ----------------------------- | ---------------------------------------- | -------------------------------------------- |
| `MEMORY_ENABLED`              | `0`                                      | Master switch                                |
| `MEMORY_VAULT_PATH`           | `/Volumes/data/Obsidian/obsi`            | Vault root (user-confirmed; external volume) |
| `MEMORY_VAULT_DENYLIST`       | `.obsidian/**,**/.trash/**`              | Glob denylist                                |
| `MEMORY_VAULT_ALLOWLIST`      | empty (= all)                            | If set, only these globs are indexed         |
| `MEMORY_CHAT_LOG_ENABLED`     | `0`                                      | Persist chat completions                     |
| `MEMORY_CHAT_RETENTION_DAYS`  | `365`                                    | Older rows are pruned/redacted               |
| `MEMORY_CHAT_RETENTION_MODE`  | `delete`                                 | `delete` or `redact`                         |
| `MEMORY_DB_PATH`              | `/Volumes/data/vllm-mlx-memory/memory.db`| SQLite file (external volume; co-located with vault) |
| `MEMORY_STM_DAYS`             | `30`                                     | Forward-compat (used by SPEC-MEMORY-02) — STM retention window in days |
| `MEMORY_EMBED_MODEL`          | `BAAI/bge-m3`                            | HF id                                        |
| `MEMORY_EMBED_BATCH`          | `16`                                     | Batch size for embedding                     |
| `MEMORY_TOP_K_DEFAULT`        | `5`                                      | Default `top_k`                              |
| `MEMORY_TOP_K_MAX`            | `20`                                     | Hard ceiling                                 |
| `MEMORY_REDACT_PATTERNS`      | `sk-[A-Za-z0-9]{20,};password\s*[:=]`    | Semicolon-separated regex list               |

## 10. Failure modes (and required degradation)

| failure                              | required behavior                                          |
| ------------------------------------ | ---------------------------------------------------------- |
| `/Volumes/data` not mounted at start | Log warning, skip vault indexer AND chat logger, `memory_search` returns `{status:"unavailable", results:[]}` with `degraded:true` |
| `MEMORY_VAULT_PATH` does not exist (volume mounted but path wrong) | Log warning, skip vault indexer, vault searches return `[]` |
| Embedding model fails to load        | Fall back to FTS5 BM25, set `degraded: true` on results    |
| sqlite-vec extension fails to load   | Same as above; FTS5 keyword search only                    |
| DB file corrupt or schema-mismatched | Hard fail at startup with actionable error; never auto-wipe|
| Partial index (crash mid-scan)       | Resume from last `vault_files.sha256` checkpoint           |
| Disk full during chat-log write      | Drop the row, log error, do not block `/v1/chat/completions` |
| Volume disconnect mid-runtime        | In-flight queries fail with `status:"busy"` per REQ-S3 retry policy; subsequent queries return `unavailable` until reconnect |

## 11. Privacy posture

- Opt-in: `MEMORY_ENABLED=0` by default. Even when enabled, chat logging is
  separately gated by `MEMORY_CHAT_LOG_ENABLED`.
- Local-only: no outbound HTTP from the memory server (REQ-U2).
- Redaction: `MEMORY_REDACT_PATTERNS` applies before the row is written.
- Allow/deny lists: `MEMORY_VAULT_ALLOWLIST` / `MEMORY_VAULT_DENYLIST`.
- Right to forget: `vllm-mlx memory forget --before YYYY-MM-DD` (CLI; MVP)
  and `vllm-mlx memory forget --path <glob>` shall remove rows + embeddings.
- No telemetry, no analytics, no usage counters phoned anywhere.

## 12. Traceability

- `@SPEC:MEMORY-01` / `@PLAN:MEMORY-01` / `@ACCEPT:MEMORY-01` — this directory
- `@CODE:MEMORY-01/server` — `vllm_mlx/memory/server.py` (new)
- `@CODE:MEMORY-01/indexer` — `vllm_mlx/memory/indexer.py` (new)
- `@CODE:MEMORY-01/embedder` — `vllm_mlx/memory/embedder.py` (new)
- `@CODE:MEMORY-01/store` — `vllm_mlx/memory/store.py` (new)
- `@CODE:MEMORY-01/chatlog` — hook in `vllm_mlx/server.py::create_chat_completion`
- `@CODE:MEMORY-01/mcp` — registration in `mcp.json` and `vllm_mlx/mcp/manager.py`
- `@TEST:MEMORY-01` — `tests/test_memory_*.py` (new)
- **Successor**: `SPEC-MEMORY-02` consumes this schema's `tier` columns and
  the `chat_summaries` / `vault_themes` tables to add STM→LTM consolidation
  (daily summarizer cron, raw-chat eviction at `MEMORY_STM_DAYS`, vault
  thematic digests, tier-aware ranking weights). No schema migration required.

## 13. Success metric

A user types in Korean: "지난 달에 distillation에 대해 어떻게 생각했지?"
Qwen3.6 calls `memory__memory_search` with a translated/paraphrased query,
the tavily search is **not** triggered, the model receives 3-5 vault
excerpts dated within the last 35 days, and produces a grounded answer
that quotes the user's own notes by path.
