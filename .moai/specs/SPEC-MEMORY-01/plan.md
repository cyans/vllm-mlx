---
id: SPEC-MEMORY-01
tags:
  - "@PLAN:MEMORY-01"
---

# Plan — MEMORY-01

## Primary Goal

Ship a local, retrieval-only semantic memory MCP server that Qwen3.6 can call
through the existing OpenAI tool-call protocol, indexing the user's Obsidian
vault and (opt-in) chat history.

## Architecture (ASCII)

```
                           +---------------------------+
                           | start-server.sh / cli.py  |
                           +-------------+-------------+
                                         |
                                         v
+--------------+   register   +---------------------------+   tool list
|  mcp.json    | -----------> |   MCPManager (existing)   | -------------+
|  servers:    |              +-----+----------------+----+              |
|   web-search |                    |                |                   |
|   memory     |                    | spawn          | spawn             v
+--------------+                    v                v          /v1/mcp/tools
                       +------------+------+   +-----+--------+ +-------------+
                       | tavily MCP server |   | memory MCP   | |Qwen3.6 chat |
                       | (Node, npx)       |   | server (new) | |+ tool call  |
                       +-------------------+   +------+-------+ +------+------+
                                                      |                |
                                                      | memory_search  |
                                                      v                |
                                            +-------------------+      |
                                            | search.py         |<-----+
                                            | rerank + filters  |
                                            +---------+---------+
                                                      |
                                            +---------+---------+
                                            | sqlite-vec store  |
                                            +---------+---------+
                                                ^     ^     ^
                                                |     |     |
                              +-----------------+     |     +-----------------+
                              |                       |                       |
                    +---------+--------+   +----------+----------+   +--------+--------+
                    | vault indexer    |   | chat logger         |   | embedder        |
                    | (watchdog +      |   | (hook in            |   | bge-m3 via      |
                    | initial scan)    |   | create_chat_         |   | mlx-embeddings  |
                    +------------------+   |  completion)        |   +-----------------+
                                           +---------------------+
```

## Technology Decisions

| Concern              | Pick                              | Why                                                                                                              |
| -------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Embedding model      | **`BAAI/bge-m3`**                 | Multilingual (100+ langs incl. Korean), 1024-dim dense + sparse + colbert in one model, runs on `mlx-embeddings` (already a dep). Strong on MIRACL Korean retrieval. |
| Vector DB            | **`sqlite-vec`** (`/asg017/sqlite-vec`) | Single-file SQLite extension, no daemon, pure-C, MIT-licensed. Co-located with chat-log table. SQLite + FTS5 fallback already available. |
| Indexer trigger      | **`watchdog`**                    | Event-driven via FSEvents on macOS, sub-second detection. Alternative `cron`/poll considered but adds latency and CPU.|
| Chunker              | **Markdown-header-aware + 512-token sliding window** | Headers preserve semantic units; sliding window handles long sections; deterministic chunk IDs.                |
| HTTP framework       | (reuse existing `fastapi`)        | No new framework.                                                                                                 |
| MCP transport        | `stdio` (matches `web-search`)    | Same in-process Python module via `python -m vllm_mlx.memory.server`, simpler than spawning subprocess.          |

### Alternatives considered

- **Embedding**: `intfloat/multilingual-e5-base` is great but smaller and
  weaker on long-context Korean than bge-m3. `jina-embeddings-v3` is excellent
  but lacks an mlx-embeddings recipe today; defer until Phase 4 if bge-m3
  underperforms.
- **Vector DB**: `lancedb` is fast and arrow-native, but adds ~80MB of
  arrow/lance deps and a separate file format. `chromadb` requires a server
  process. Both are reasonable Phase-4 swap-outs if sqlite-vec hits a wall.
- **Indexer**: cron polling is dead simple and zero-dependency. Keep it as
  a fallback when `watchdog` import fails (`MEMORY_INDEXER=poll` env var).

### STM/LTM forward-compat (Path C)

This SPEC adopts **Path C** of the dual-tier memory design: the schema includes
the `tier` column and `chat_summaries` / `vault_themes` tables from day 1,
but MVP only writes `tier='stm'` rows and leaves the summary tables empty.
SPEC-MEMORY-02 then adds the consolidation logic (daily summarizer cron,
raw-chat eviction at 30 days, vault thematic digests) **without any schema
migration**. The cost is ~10 lines of CREATE TABLE plus one column on two
existing tables — paid once now to avoid a more expensive migration later.

### Storage location decision

Both vault notes (`MEMORY_VAULT_PATH=/Volumes/data/Obsidian/obsi`) and the
memory database (`MEMORY_DB_PATH=/Volumes/data/vllm-mlx-memory/memory.db`)
live on the same external volume. Rationale:

- One-shot backup of `/Volumes/data` covers all memory state.
- DB and source data ride together — no risk of a "vault present but DB
  stale" or vice-versa drift after volume disconnect/reconnect.
- Loss of the volume is an explicit failure mode (REQ-N4 + § 10): chat path
  continues to work, memory just becomes unavailable until reconnect.
- The DB lives in a sibling directory (`/Volumes/data/vllm-mlx-memory/`)
  rather than inside the vault, so Obsidian itself never indexes it.

### Risks if these picks are wrong

- bge-m3 is ~2.3 GB on disk; first download may surprise users → mitigate
  with progress bar and clear log line + size warning.
- sqlite-vec extension binary must match Python's SQLite — verified working
  on macOS 14+ with system Python 3.10+.
- watchdog fallback to polling is non-trivial on Linux; macOS-only is fine
  for MVP since the project is Apple-Silicon-first.

## Phase Breakdown

### Phase 1 — Vault read-only PoC (BM25 only)

Deliverables:
- `vllm_mlx/memory/store.py` with SQLite schema (no `sqlite-vec` yet, just FTS5).
- `vllm_mlx/memory/indexer.py` with initial vault scan (no incremental yet).
- `vllm_mlx/memory/server.py` exposing `memory_search` over MCP stdio,
  doing FTS5 BM25 only.
- `mcp.json` updated with `memory` entry (commented-out by default).
- Unit tests: indexer enumerates files, FTS5 finds known phrases.
- Acceptance tests: `tests/test_memory_indexer.py`, `tests/test_memory_server_bm25.py`.
- Manual smoke: `MEMORY_ENABLED=1 MEMORY_VAULT_PATH=/Volumes/data/Obsidian/obsi MEMORY_DB_PATH=/Volumes/data/vllm-mlx-memory/memory.db ./start-server.sh`,
  ask "search my notes for distillation" → tool call → results.

Out: embeddings, chat logging, watchdog, retention.

Effort: ~2 days.

### Phase 2 — Embeddings + vector ANN

Deliverables:
- `vllm_mlx/memory/embedder.py` wrapping `mlx-embeddings` for bge-m3.
- Add `sqlite-vec` virtual table + `vec_chunks`.
- Hybrid search: FTS5 BM25 + dense cosine, RRF (reciprocal rank fusion) merge.
- Replace Phase-1 search path with hybrid; keep BM25-only as
  `degraded: true` fallback.
- Bench: time `memory_search` on a 10k-chunk vault.

Effort: ~3 days.

### Phase 3 — Chat persistence + indexing

Deliverables:
- Hook in `vllm_mlx/server.py::create_chat_completion` (around line 1389) and
  the SSE close path in `stream_chat_completion` to persist chat rows when
  `MEMORY_CHAT_LOG_ENABLED=1`.
- Async embedding job queue so the chat path is not blocked on embedding latency
  (REQ-N4: failures must not propagate).
- Redaction filter applied **before** the row is written (REQ-N1).
- Retention sweeper that runs at startup and every 24h
  (`MEMORY_CHAT_RETENTION_*`).
- CLI: `vllm-mlx memory forget --before <date>`, `--path <glob>`.

Effort: ~2 days.

### Phase 4 — Polish

Deliverables:
- `watchdog` incremental updates (REQ-E2).
- `time_range` and `source_filter` argument support (REQ-O1, REQ-O2).
- `degraded: true` flag plumbed through every fallback path (REQ-O3, REQ-S2).
- Docs: `docs/memory.md`, README section, env-var reference.
- `start-server.sh` flag pass-through for `--memory-enabled`.
- Coverage ≥ 85% on `vllm_mlx/memory/*`.

Effort: ~2 days.

### Chunker quality (followup, landed in Phase 2)

Originally scoped for Phase 4 polish, but pulled forward after the
Phase 1+2 live smoke test exposed a concrete regression: an Obsidian
book-style file produced ~16 chunks where roughly half were 15-30
char header-only fragments (e.g. `## 3부 내면 근력 강화 6단계`) that
lacked any body text. Two consequences:

1. **BM25 dominance** — the tiny header-only chunks scored very high
   on queries that matched the header words verbatim, pushing out body
   chunks that contained the actual content.
2. **Lost parent context for body chunks** — a body chunk like
   `### 6장 1단계: 자기절제의 뇌과학` did not carry its umbrella
   `## 3부 ...` header, so semantic match against `"내면 근력 강화"`
   was weak.

Fix shipped as chunker v2 (`vllm_mlx/memory/indexer.py::chunk_markdown`):

- Every emitted chunk carries its full parent header path as a
  textual prefix (`"# A > ## B > ### C\n\n<body>"`).
- Header-only sections still emit a chunk, but the chunk's text is
  the joined header path itself (gives BM25 enough content to score
  fairly; gives the dense embedder umbrella context for free).
- `meta.chunker_version` is now recorded on first scan; a mismatch at
  startup prints one WARNING line — no auto-rebuild.
- Operator path: `python -m vllm_mlx.memory.indexer --rebuild` wipes
  vault tables and re-chunks from scratch; `python -m vllm_mlx.memory.backfill`
  re-embeds afterwards. The two phases stay separate by design.
- Tests: added 6 chunker-v2 tests + 1 fixture
  (`tests/fixtures/vault_small/multi_level_headers.md`); coverage on
  `indexer.py` rose to 91%, `store.py` stays at 90%.

Schema_version is intentionally NOT bumped — the wire format is
identical, only the *content* of each chunk's `text` column changed.

Total rough effort: ~9 working days.

## Dependencies — additions to `pyproject.toml`

```toml
# main dependencies (always installed)
"watchdog>=4.0.0"           # FSEvents-backed file watcher
"sqlite-vec>=0.1.6"          # vector search SQLite extension

# (mlx-embeddings>=0.0.5 already declared at line 62)
```

No new optional groups. The bge-m3 weights are downloaded lazily by
`mlx-embeddings` on first use and cached in the standard HF cache dir.

## Integration Points (existing files to touch)

| file                                  | what changes                                                   |
| ------------------------------------- | -------------------------------------------------------------- |
| `mcp.json` / `mcp.example.json`       | Add `memory` server entry                                       |
| `vllm_mlx/server.py`                  | Hook chat-log persistence in `create_chat_completion` (~1389)   |
| `vllm_mlx/server.py`                  | Hook chat-log persistence in `stream_chat_completion` SSE close |
| `vllm_mlx/cli.py`                     | Add `vllm-mlx memory rebuild` and `vllm-mlx memory forget`     |
| `pyproject.toml`                      | Add `watchdog`, `sqlite-vec`                                    |
| `start-server.sh`                     | Document the env vars; no flag changes                          |
| `docs/`                               | New `docs/memory.md`                                            |

New files:

| file                                  | role                                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| `vllm_mlx/memory/__init__.py`         | Package marker, public API                                      |
| `vllm_mlx/memory/server.py`           | MCP stdio server exposing `memory_search`                       |
| `vllm_mlx/memory/indexer.py`          | Vault scan + watchdog                                           |
| `vllm_mlx/memory/chunker.py`          | Markdown-header + sliding-window chunking                        |
| `vllm_mlx/memory/embedder.py`         | bge-m3 wrapper, batching, warmup                                |
| `vllm_mlx/memory/store.py`            | sqlite-vec + FTS5 + chat row persistence                        |
| `vllm_mlx/memory/search.py`           | Hybrid retrieval + RRF + filters                                 |
| `vllm_mlx/memory/redact.py`           | Pattern-based redaction                                          |
| `vllm_mlx/memory/cli.py`              | `forget`, `rebuild`, `stats`                                    |
| `tests/test_memory_indexer.py`        | Phase 1                                                          |
| `tests/test_memory_chunker.py`        | Phase 1                                                          |
| `tests/test_memory_embedder.py`       | Phase 2 (mocks model load)                                       |
| `tests/test_memory_store.py`          | Phase 1+2                                                        |
| `tests/test_memory_server.py`         | Phase 1+ end-to-end MCP tool                                     |
| `tests/test_memory_chatlog.py`        | Phase 3                                                          |
| `tests/test_memory_redact.py`         | Phase 3                                                          |
| `tests/test_memory_e2e.py`            | Phase 4 — Korean-query smoke test                                |

## Risks & Mitigations

| risk                                                       | mitigation                                                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Initial index of a 10k-note vault is slow (>5 min)         | Batched embedding (batch=16), progress logging, store partial state, allow resume.               |
| bge-m3 download (~2.3 GB) on first run                     | Document upfront, warn at startup, suggest `huggingface-cli download` pre-fetch, allow offline.  |
| Korean retrieval quality below expectations                | Hybrid BM25+dense via RRF mitigates. If still poor, swap to `jinaai/jina-embeddings-v3` in P4.   |
| sqlite-vec extension fails to load on user's macOS         | Fall back to FTS5-only, surface `degraded: true`. Document in `docs/memory.md`.                  |
| Vault path on iCloud Drive (not yet downloaded files)      | Skip files with size 0 + `.icloud` extension; log a one-line warning per skip.                   |
| Symlink escape from vault root                             | Resolve real path; refuse if outside root (REQ-N2).                                              |
| Chat-log row containing API keys                           | `MEMORY_REDACT_PATTERNS` applied pre-write (REQ-N1); unit-tested with seed strings.              |
| Concurrent writers (indexer + chat logger)                 | SQLite WAL mode, exponential-backoff retries (REQ-S3).                                           |
| Schema drift across versions                               | `meta.schema_version`; mismatch → hard error + `vllm-mlx memory rebuild` instruction.            |
| Memory subsystem crash propagates to chat                  | Top-level try/except in the chat-completion hook with degraded log line (REQ-N4).                |
| Embedding model load on a low-RAM Mac (8GB)                | Lazy load on first call, document RAM footprint, allow `MEMORY_EMBED_MODEL` override.            |

## Out-of-scope (re-stated for clarity)

- MemGPT-style hierarchical summarization
- Knowledge-graph extraction
- Auto-write-on-saliency / agentic memory writes
- Curation UI
- Auto-think gate (separate `SPEC-THINK-02`)
- Cross-device sync
- At-rest encryption beyond OS file permissions

## Expert Consultation

- **expert-backend**: primary author for `server.py`, `store.py`, MCP wiring,
  chat-completion hook.
- **expert-testing**: parametrized chunker tests, redaction unit tests,
  end-to-end Korean-query smoke test.
- **expert-performance**: bench `memory_search` p95, tune RRF weights,
  optimize batch sizes.
- **expert-security**: review `MEMORY_REDACT_PATTERNS`, symlink-escape guard,
  chat-log retention.

## Open questions parked for run phase

- **OQ-1**: Should we use `mlx-embeddings` directly or wrap `mlx_lm`'s
  embedding API? (Default: `mlx-embeddings`, since it is already a dep.)
- **OQ-2**: Is the chat-log hook better placed in middleware or inline in
  `create_chat_completion`? (Default: inline + `try/except`, no middleware.)
- **OQ-3**: Should `memory_search` block the model on cold start, or return
  `warming_up`? (Default: return `warming_up` per REQ-S2 — non-blocking.)
- **OQ-4**: Should we expose a `memory_save(text, tags)` write tool in MVP?
  (Default: **no**, deferred to follow-up SPEC. MVP is read-only.)
