# SPDX-License-Identifier: Apache-2.0
"""Vault indexer: directory walk + chunker + batch upsert.

@CODE:MEMORY-01/indexer

Phase 1 only does an initial scan: enumerate every ``*.md`` file under
``vault_path``, compute SHA-256, chunk by Markdown headers + a 512-token
sliding window fallback, and upsert into the store.

REQ-N2: never traverse symlinks that escape ``vault_root``. We resolve
each candidate path and refuse anything outside the resolved root.
REQ-N3: glob-deny ``.obsidian/**``, ``**/.trash/**``, etc.
REQ-E1: idempotent on sha256 — a second scan over an unchanged vault
performs no writes.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import logging
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .store import MemoryStore, chunk_id_for, file_sha256

logger = logging.getLogger(__name__)


# Bumped manually whenever ``chunk_markdown`` changes in a way that
# alters chunk text or chunk_id derivation. Operators with an existing
# DB will see a single warning at startup if their ``meta.chunker_version``
# does not match this constant; running ``--rebuild`` clears the warning.
#
# Version history:
#   "1" — Phase 1: header section + sliding window only, no parent path.
#   "2" — chunker quality fix: every chunk carries its parent header
#         path as a textual prefix; header-only sections emit a chunk
#         containing the joined path instead of a 15-char fragment.
CHUNKER_VERSION = "2"


# Phase 2 — keep the embedder type loose so this module never imports
# mlx-embeddings at indexing time when the dense path is disabled.
# ``Embedder | None`` is what callers actually pass.
class _SupportsEmbed:  # pragma: no cover - protocol-only sentinel
    """Structural type the indexer expects of any embedder argument.

    We use a small protocol-shaped class instead of typing.Protocol so
    Python 3.10 imports do not pay for ``runtime_checkable`` here.
    """

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        ...  # pragma: no cover

    def available(self) -> bool:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# 512 tokens ≈ 2048 chars under the rough "1 token ≈ 4 chars" heuristic
# from the OpenAI tokenizer FAQ. Phase 1 doesn't run a real tokenizer
# (no embedding model loaded yet) so this approximation is fine.
WINDOW_CHARS = 2048
WINDOW_OVERLAP = 256

# Files larger than this are skipped with a one-line warning. Real-world
# Obsidian notes are well under 100 KB.
MAX_FILE_BYTES = 4 * 1024 * 1024

# Bytes-per-batch transactional commit boundary. We commit every N files
# during a long scan so a crash mid-scan loses at most one batch.
COMMIT_EVERY_FILES = 25


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    text: str
    header: str
    chunk_index: int
    char_offset: int
    char_len: int
    timestamp: str


@dataclass(frozen=True)
class IndexerStats:
    """Summary of an initial-scan run, returned for log + status reporting."""

    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    chunks_written: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Separator placed between path components in the textual prefix.
# Mirrors the convention readers see in Obsidian breadcrumb plugins
# and keeps BM25 tokenization happy: tokens stay whole, the separator
# is stripped by the FTS5 unicode61 tokenizer along with other
# punctuation.
_HEADER_PATH_SEP = " > "


def _build_header_prefix(header_stack: list[tuple[int, str]]) -> str:
    """Render a header stack into ``"## A > ### B"`` form.

    ``header_stack`` entries are ``(level, full_header_line)`` where
    ``full_header_line`` is the original markdown line including the
    ``#`` markers and the title (e.g. ``"## 3부 내면 근력 강화 6단계"``).
    Joining the markdown lines verbatim has two benefits:

    * BM25 sees the same tokens in the prefix that it would see in the
      raw header line, so a query like ``"3부 내면 근력"`` matches both
      header-only chunks AND body chunks under that header.
    * The dense embedder gets the umbrella context for free, even when
      the body chunk would otherwise be a leaf section like
      ``"### 6장 1단계: 자기절제의 뇌과학"``.
    """
    if not header_stack:
        return ""
    return _HEADER_PATH_SEP.join(line for _, line in header_stack)


def chunk_markdown(text: str) -> list[ChunkRow]:
    """Split a Markdown document into header-aware chunks.

    Strategy (chunker v2):

    1. Walk the document line by line, maintaining a header stack so
       every chunk knows its ancestor headers. A new header pops every
       stack entry whose level is ``>= incoming.level`` before pushing
       itself, which gives correct H1>H2>H3 paths even when the doc
       jumps around (e.g. ``## A`` → ``### A1`` → ``## B``).
    2. Body lines accumulate until the next header or EOF; on flush we
       emit a chunk whose ``text`` starts with ``"<header path>\\n\\n"``
       followed by the body. The header line itself is consumed as a
       context marker and never duplicated inside the body.
    3. If a section closes with NO body (header followed immediately by
       another header, or EOF), we still emit a chunk whose ``text`` is
       the header path alone. This keeps ToC entries searchable while
       giving them enough content (parent path) to participate fairly
       in BM25 ranking — fixing the live-smoke regression where bare
       ``"## 3부 ..."`` chunks of 15-30 chars dominated relevance.
    4. When a section's body alone exceeds ``WINDOW_CHARS``, the
       sliding-window splitter takes over and EACH window carries the
       same header-path prefix. The window offsets are computed against
       the body buffer (not the prefix), so ``char_offset`` continues
       to point into the original document.

    Determinism: chunk IDs are still produced from
    ``(path, char_offset, char_len)`` at the caller, so the same input
    file always yields the same chunk IDs *under v2*. Switching from v1
    to v2 changes the chunk text and is expected to change chunk_ids
    for files that contained headers; the ``--rebuild`` flag handles
    the transition cleanly.
    """
    if not text:
        return []

    chunks: list[ChunkRow] = []
    chunk_index = 0
    now = _utc_now_iso()

    # Header stack: list of (level, original_header_line). The original
    # line includes the leading ``#`` markers which we keep in the
    # joined prefix. A separate parallel stack tracks just the title
    # text so the legacy ``ChunkRow.header`` field stays compatible
    # with v1 callers (and with existing tests).
    header_stack: list[tuple[int, str]] = []
    header_titles: list[str] = []

    # Current body buffer: list of (line_offset_in_text, line_text).
    # We track absolute offsets so the resulting ChunkRow.char_offset
    # / char_len keep pointing at the original document, not at the
    # prefixed text.
    body_lines: list[tuple[int, str]] = []
    section_started_after_header: bool = False

    def flush_section() -> None:
        """Emit one chunk for the section that just closed.

        Three cases:

        * Section has body → chunk text is ``"<prefix>\\n\\n<body>"``
          when there is a prefix, else just the body. If the body
          exceeds ``WINDOW_CHARS`` the sliding window kicks in and
          each window carries the same prefix.
        * Section has no body but DOES have a header path → emit a
          single chunk whose text is the prefix alone (the ToC entry).
        * No body and no header path (very start of an empty doc) →
          nothing to emit.
        """
        nonlocal chunk_index

        prefix = _build_header_prefix(header_stack)

        # Stitch body lines back together, preserving original spacing
        # between non-blank lines. We do not strip blank lines inside
        # the body — they belong to the source.
        if body_lines:
            body_start = body_lines[0][0]
            # Reconstruct body by joining with newlines. Each line was
            # captured without its trailing newline so we restore them.
            body_text = "\n".join(line for _, line in body_lines)
            body_text_stripped = body_text.strip()
        else:
            body_start = -1
            body_text = ""
            body_text_stripped = ""

        # Title (without ``#`` markers) of the deepest header on the
        # stack — what gets stored in ``ChunkRow.header`` for back-
        # compat with v1 readers (the FTS layer doesn't use this field
        # for ranking; it lives in ``vault_chunks.header`` for debug).
        leaf_title = header_titles[-1] if header_titles else ""

        if not body_text_stripped:
            # Header-only chunk: emit the prefix as the text body so
            # the chunk has enough content to score fairly.
            if not prefix:
                return  # empty doc, nothing meaningful to emit
            # We anchor a header-only chunk at the start of the most
            # recent header line so char_offset stays meaningful for
            # debugging. char_len is set to the prefix length; the
            # underlying source range is the header line itself.
            anchor_offset = (
                header_stack_offsets[-1] if header_stack_offsets else 0
            )
            chunks.append(
                ChunkRow(
                    chunk_id="",
                    text=prefix,
                    header=leaf_title,
                    chunk_index=chunk_index,
                    char_offset=anchor_offset,
                    char_len=len(prefix),
                    timestamp=now,
                )
            )
            chunk_index += 1
            return

        # Recompute the absolute offset of the trimmed body so
        # char_offset stays pointed at the first non-whitespace byte
        # of the original document (consistent with v1 behaviour).
        leading_strip = len(body_text) - len(body_text.lstrip())
        absolute_offset = body_start + leading_strip
        body_for_chunks = body_text_stripped
        prefix_with_sep = f"{prefix}\n\n" if prefix else ""

        if len(body_for_chunks) <= WINDOW_CHARS:
            chunks.append(
                ChunkRow(
                    chunk_id="",
                    text=f"{prefix_with_sep}{body_for_chunks}",
                    header=leaf_title,
                    chunk_index=chunk_index,
                    char_offset=absolute_offset,
                    char_len=len(body_for_chunks),
                    timestamp=now,
                )
            )
            chunk_index += 1
            return

        # Body exceeds the window: split with sliding windows and
        # prepend the same prefix to each piece.
        for off, piece in _sliding_windows(body_for_chunks):
            chunks.append(
                ChunkRow(
                    chunk_id="",
                    text=f"{prefix_with_sep}{piece}",
                    header=leaf_title,
                    chunk_index=chunk_index,
                    char_offset=absolute_offset + off,
                    char_len=len(piece),
                    timestamp=now,
                )
            )
            chunk_index += 1

    # Iterate the document line by line, tracking byte offsets so we
    # can produce char_offset values that index into the original text.
    pos = 0
    # Parallel stack of header start offsets used by header-only chunks.
    header_stack_offsets: list[int] = []
    for line in text.splitlines(keepends=True):
        # Strip the trailing newline (if any) for matching / storage;
        # we still know the original length via ``line``.
        line_no_nl = line.rstrip("\n").rstrip("\r")
        line_offset = pos

        m = _HEADER_LINE_RE.match(line_no_nl)
        if m is not None:
            # New header: flush the section that just ended, then update
            # the stack and start a fresh body buffer.
            flush_section()
            body_lines = []
            section_started_after_header = True

            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop entries with level >= incoming.level so the new
            # header replaces same-or-deeper ancestors.
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
                header_stack_offsets.pop()
                header_titles.pop()
            header_stack.append((level, line_no_nl))
            header_stack_offsets.append(line_offset)
            header_titles.append(title)
        else:
            # Body line. Skip leading blank lines that come immediately
            # after a header so the body buffer does not start with
            # whitespace garbage; subsequent blank lines are preserved.
            if (
                section_started_after_header
                and not body_lines
                and not line_no_nl.strip()
            ):
                pos += len(line)
                continue
            body_lines.append((line_offset, line_no_nl))
            section_started_after_header = False

        pos += len(line)

    # Flush whatever's left after the last line.
    flush_section()

    return chunks


def _sliding_windows(
    text: str,
    *,
    window: int = WINDOW_CHARS,
    overlap: int = WINDOW_OVERLAP,
) -> Iterator[tuple[int, str]]:
    """Yield (offset, slice) pairs covering ``text`` with overlap."""
    if not text:
        return
    if len(text) <= window:
        yield 0, text
        return

    step = max(1, window - overlap)
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + window)
        yield pos, text[pos:end]
        if end == len(text):
            return
        pos += step


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
class VaultIndexer:
    """Initial-scan indexer (no watchdog, no incremental updates yet).

    Construction does no I/O. Call :meth:`initial_scan` to actually walk
    the vault. Phase 1 always wraps the call in a try/except at the
    server boundary so a memory failure cannot reach the chat path
    (REQ-N4).
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        vault_root: Path,
        denylist: tuple[str, ...] = (),
        allowlist: tuple[str, ...] = (),
        embedder: _SupportsEmbed | None = None,
        embed_disabled: bool = False,
    ):
        self.store = store
        # Resolve symlinks on the root once so per-file resolves can be
        # checked against a stable canonical path (REQ-N2).
        self.vault_root = vault_root.expanduser().resolve()
        self.denylist = denylist
        self.allowlist = allowlist
        # Phase 2 — optional dense indexing. The indexer keeps working
        # when this is None, when the embedder load fails, or when the
        # operator sets MEMORY_EMBED_DISABLED=1.
        self.embedder = embedder
        self.embed_disabled = bool(embed_disabled)

    # -- public API ---------------------------------------------------------

    def initial_scan(self) -> IndexerStats:
        """Walk the vault, chunk + upsert any changed file.

        Returns aggregate statistics for the operator log.
        """
        stats = IndexerStats()

        if not self.vault_root.is_dir():
            logger.warning(
                "[memory] vault path missing or not a directory: %s",
                self.vault_root,
            )
            return stats

        # Record / verify the chunker version so a future operator can
        # detect that existing chunks were produced by a different
        # algorithm. We never auto-rebuild — that's an explicit choice
        # via ``python -m vllm_mlx.memory.indexer --rebuild``.
        self._sync_chunker_version()

        files_seen = 0
        files_indexed = 0
        files_skipped = 0
        files_unchanged = 0
        chunks_written = 0
        errors = 0

        files_in_batch: list[Path] = []
        # We open one transaction per batch of COMMIT_EVERY_FILES files so a
        # crash mid-scan loses at most one batch (resume next start).
        for path in self._walk():
            files_seen += 1
            try:
                rel = self._safe_relpath(path)
                if rel is None:
                    files_skipped += 1
                    continue

                if path.stat().st_size > MAX_FILE_BYTES:
                    logger.warning(
                        "[memory] skipping oversized file (%d bytes): %s",
                        path.stat().st_size,
                        rel,
                    )
                    files_skipped += 1
                    continue

                # Idempotent: only re-chunk when sha256 differs.
                sha = file_sha256(path)
                stored_sha = self.store.get_vault_file_sha(rel)
                if stored_sha == sha:
                    files_unchanged += 1
                    continue

                text = path.read_text(encoding="utf-8", errors="replace")
                chunks = chunk_markdown(text)
                # Stamp deterministic IDs.
                stamped = [
                    {
                        "chunk_id": chunk_id_for(rel, c.char_offset, c.char_len),
                        "text": c.text,
                        "header": c.header,
                        "chunk_index": c.chunk_index,
                        "char_offset": c.char_offset,
                        "char_len": c.char_len,
                        "timestamp": c.timestamp,
                    }
                    for c in chunks
                ]

                files_in_batch.append(path)
                with self.store.transaction():
                    file_id = self.store.upsert_vault_file(
                        path=rel,
                        mtime=path.stat().st_mtime,
                        sha256=sha,
                        size=path.stat().st_size,
                    )
                    inserted = self.store.replace_chunks_for_file(
                        file_id=file_id,
                        source_path=rel,
                        chunks=stamped,
                    )
                    chunks_written += inserted
                    # Phase 2 — embed the freshly written chunks inline
                    # while the chunk_id rows are still in this txn.
                    # Failures are logged but do not abort the txn:
                    # BM25 + the file row remain consistent.
                    self._embed_chunks_safely(stamped)
                files_indexed += 1

                if files_indexed % COMMIT_EVERY_FILES == 0:
                    logger.info(
                        "[memory] initial scan progress: %d files, %d chunks",
                        files_indexed,
                        chunks_written,
                    )
            except Exception:  # noqa: BLE001 - REQ-N4: never propagate
                errors += 1
                logger.exception("[memory] error indexing %s", path)

        stats = IndexerStats(
            files_seen=files_seen,
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            files_unchanged=files_unchanged,
            chunks_written=chunks_written,
            errors=errors,
        )
        logger.info(
            "[memory] initial scan done: seen=%d indexed=%d unchanged=%d "
            "skipped=%d chunks=%d errors=%d",
            stats.files_seen,
            stats.files_indexed,
            stats.files_unchanged,
            stats.files_skipped,
            stats.chunks_written,
            stats.errors,
        )
        return stats

    # -- internals ----------------------------------------------------------

    def _sync_chunker_version(self) -> None:
        """Read / write ``meta.chunker_version`` and warn on mismatch.

        Behaviour:

        * No row yet (fresh DB) → write ``CHUNKER_VERSION`` so future
          opens compare against the correct value.
        * Stored value matches code → no-op.
        * Stored value differs → emit a single WARNING line. We do NOT
          auto-rebuild; the operator decides via ``--rebuild``.
        """
        try:
            stored = self.store.get_meta("chunker_version")
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception("[memory] could not read meta.chunker_version")
            return

        if stored is None:
            try:
                self.store.set_meta("chunker_version", CHUNKER_VERSION)
            except Exception:  # noqa: BLE001 - REQ-N4
                logger.exception(
                    "[memory] could not write meta.chunker_version"
                )
            return

        if stored != CHUNKER_VERSION:
            logger.warning(
                "[memory] chunker version mismatch (db=%s, code=%s); "
                "existing chunks may be stale, consider running "
                "`python -m vllm_mlx.memory.indexer --rebuild`",
                stored,
                CHUNKER_VERSION,
            )

    def _embed_chunks_safely(self, stamped: list[dict[str, object]]) -> None:
        """Best-effort dense indexing for a batch of just-inserted chunks.

        Skips silently when the dense path is unavailable (no embedder,
        embedder failed to load, MEMORY_EMBED_DISABLED=1, or sqlite-vec
        not loaded on the store). REQ-N4: any exception is logged and
        swallowed so the BM25 index stays consistent with vault_files.
        """
        if self.embed_disabled:
            return
        if self.embedder is None:
            return
        if not getattr(self.store, "vec_loaded", False):
            return
        # We do not call ``.available()`` first because it returns False
        # before the very first ``.load()`` call. The embedder itself
        # short-circuits subsequent calls after a failed load.

        try:
            texts = [str(c["text"]) for c in stamped]
            blobs = self.embedder.encode_batch(texts)
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] embedding failed for %d chunks; BM25 still wrote",
                len(stamped),
            )
            return
        if len(blobs) != len(stamped):
            # Embedder degraded mid-batch (returned []). Skip vector
            # writes for this file; backfill will pick them up later.
            if blobs:
                logger.warning(
                    "[memory] embedder returned %d/%d vectors for batch; "
                    "skipping vector insert (backfill required)",
                    len(blobs),
                    len(stamped),
                )
            return
        try:
            self.store.insert_vectors_batch(
                rows=zip(
                    (str(c["chunk_id"]) for c in stamped),
                    blobs,
                ),
                source_type="vault",
            )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] vector insert failed for %d chunks", len(stamped)
            )

    def _walk(self) -> Iterable[Path]:
        """Yield every ``*.md`` candidate under the vault root.

        Uses ``os.walk(followlinks=False)`` so symlinked directories
        outside the root are not silently traversed (REQ-N2). A separate
        per-path resolve check still guards against per-file symlinks.
        """
        for dirpath, dirnames, filenames in os.walk(
            self.vault_root, followlinks=False
        ):
            # Prune denied directories early to save ``stat`` calls.
            pruned = []
            for d in dirnames:
                if not self._is_dir_denied(Path(dirpath) / d):
                    pruned.append(d)
            dirnames[:] = pruned

            for fn in filenames:
                if not fn.lower().endswith(".md"):
                    continue
                yield Path(dirpath) / fn

    def _safe_relpath(self, path: Path) -> str | None:
        """Resolve ``path`` and return the relative form if inside root.

        Returns ``None`` for symlink escapes (REQ-N2) and for paths
        matching the deny / allow lists (REQ-N3). The returned string
        uses POSIX separators so the same row keys work on macOS today
        and Linux tomorrow.
        """
        try:
            real = path.resolve()
        except (OSError, RuntimeError):
            return None
        try:
            rel = real.relative_to(self.vault_root)
        except ValueError:
            logger.warning(
                "[memory] refusing symlink/path escape: %s -> %s",
                path,
                real,
            )
            return None

        rel_posix = rel.as_posix()

        if self._is_path_denied(rel_posix):
            return None
        if self.allowlist and not self._is_path_allowed(rel_posix):
            return None
        return rel_posix

    def _is_dir_denied(self, dirpath: Path) -> bool:
        """Check whether a directory should be pruned from the walk."""
        try:
            real = dirpath.resolve()
            rel = real.relative_to(self.vault_root).as_posix()
        except (OSError, ValueError):
            # If the directory escapes the root or cannot be resolved,
            # do not traverse it.
            return True
        return self._is_path_denied(rel + "/")

    def _is_path_denied(self, rel: str) -> bool:
        return any(_glob_match(p, rel) for p in self.denylist)

    def _is_path_allowed(self, rel: str) -> bool:
        return any(_glob_match(p, rel) for p in self.allowlist)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _glob_match(pattern: str, rel: str) -> bool:
    """Match a path glob, treating ``**`` like ``fnmatch`` does today.

    ``fnmatch.fnmatch`` already treats ``**`` as ``*`` (no recursion)
    but the patterns we use (``.obsidian/**``, ``**/.trash/**``) work
    correctly because each path component is checked along the rel
    string. We additionally check the pattern against every prefix of
    the rel-path so ``.obsidian/**`` matches ``.obsidian/foo`` and
    ``.obsidian/foo/bar.md`` alike.
    """
    if fnmatch.fnmatch(rel, pattern):
        return True
    # Substring check for ``**`` patterns: if a deny like ``**/.trash/**``
    # is supplied we also match any path containing ``/.trash/``.
    if "**" in pattern:
        plain = pattern.replace("**", "").strip("/")
        if plain and (
            rel.startswith(plain + "/")
            or ("/" + plain + "/") in ("/" + rel)
            or rel == plain
        ):
            return True
    return False


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 with ``Z`` suffix."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Rebuild helper
# ---------------------------------------------------------------------------
def rebuild_vault_tables(store: MemoryStore) -> None:
    """Wipe vault-related tables in a single transaction.

    Drops every row from ``vault_files``, ``vault_chunks``, ``fts_chunks``,
    and (if loaded) ``vec_chunks``. Resets ``meta.chunker_version`` to
    the current code value. Does NOT touch ``chat_messages`` /
    ``chat_summaries`` / ``vault_themes`` — those are persisted by the
    chat path and are out of scope for a chunker rebuild.

    Embeddings are dropped because their chunk_ids will no longer exist
    in vault_chunks after the rebuild; the operator runs
    ``python -m vllm_mlx.memory.backfill`` afterwards to repopulate
    ``vec_chunks`` against the newly-emitted chunks.
    """
    with store.transaction():
        store.conn.execute("DELETE FROM fts_chunks")
        store.conn.execute("DELETE FROM vault_chunks")
        store.conn.execute("DELETE FROM vault_files")
        if store.vec_loaded:
            store.conn.execute("DELETE FROM vec_chunks")
        store.set_meta("chunker_version", CHUNKER_VERSION)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser():  # pragma: no cover - thin CLI glue
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m vllm_mlx.memory.indexer",
        description=(
            "Vault chunker / indexer. Without flags this is a noop; the "
            "real indexing path runs inside the live server. Use "
            "``--rebuild`` to wipe the vault tables and re-chunk every "
            "file from scratch (operator action when the chunker "
            "version changes)."
        ),
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "wipe vault_files / vault_chunks / fts_chunks / vec_chunks "
            "and re-run initial_scan from scratch. Required after the "
            "chunker algorithm changes (see meta.chunker_version)."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive 'type YES to confirm' prompt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be deleted, then exit without writing",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """CLI entry point. Returns a process exit code.

    Today's only meaningful operation is ``--rebuild``. We intentionally
    keep this module's CLI tiny: backfill (embeddings) and stats live in
    sibling modules.
    """
    import time

    from .config import resolve_memory_config
    from .store import open_store

    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if not args.rebuild:
        logger.info(
            "[memory] indexer CLI: nothing to do (use --rebuild to "
            "wipe and re-chunk the vault)"
        )
        return 0

    config = resolve_memory_config(os.environ)
    if not config.enabled:
        logger.error(
            "MEMORY_ENABLED is not set. Set MEMORY_ENABLED=1 (and "
            "MEMORY_VAULT_PATH / MEMORY_DB_PATH) before running --rebuild."
        )
        return 2

    if args.dry_run:
        try:
            store = open_store(config.db_path, embed_dim=config.embed_dim)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[memory] failed to open store at %s", config.db_path
            )
            return 1
        try:
            n_files = store.count_vault_files()
            n_chunks = store.count_vault_chunks()
            print(
                f"[memory] dry-run: would delete {n_files} files / "
                f"{n_chunks} chunks from {config.db_path}"
            )
            return 0
        finally:
            store.close()

    if not args.yes:
        try:
            answer = input(
                "[memory] this will delete every vault row in "
                f"{config.db_path}. Type YES to confirm: "
            )
        except (EOFError, KeyboardInterrupt):
            logger.warning("[memory] rebuild cancelled (no tty)")
            return 130
        if answer.strip() != "YES":
            logger.info("[memory] rebuild cancelled")
            return 0

    try:
        store = open_store(config.db_path, embed_dim=config.embed_dim)
    except Exception:  # noqa: BLE001
        logger.exception("[memory] failed to open store at %s", config.db_path)
        return 1

    try:
        before_files = store.count_vault_files()
        before_chunks = store.count_vault_chunks()
        logger.info(
            "[memory] rebuild: dropping %d files / %d chunks",
            before_files,
            before_chunks,
        )
        rebuild_vault_tables(store)

        # Re-run initial_scan so the operator gets a populated DB
        # immediately; embeddings still need a separate ``backfill`` run.
        scanner = VaultIndexer(
            store,
            vault_root=config.vault_path,
            denylist=config.denylist,
            allowlist=config.allowlist,
            # No embedder during rebuild — keep chunk + embed concerns
            # separate so a long re-embed does not block the rebuild
            # itself. Operators run ``backfill`` afterwards.
            embedder=None,
            embed_disabled=True,
        )
        started = time.monotonic()

        # initial_scan already logs every COMMIT_EVERY_FILES files; for
        # a one-shot rebuild we add a single summary line at the end.
        stats = scanner.initial_scan()
        elapsed = max(1e-3, time.monotonic() - started)
        rate = stats.files_indexed / elapsed
        logger.info(
            "[memory] rebuild done: %d files indexed, %d chunks written, "
            "%.1fs elapsed (%.1f files/s)",
            stats.files_indexed,
            stats.chunks_written,
            elapsed,
            rate,
        )
        print(
            f"[memory] rebuild complete: files={stats.files_indexed}, "
            f"chunks={stats.chunks_written}, errors={stats.errors}"
        )
        print(
            "[memory] next step: run "
            "`python -m vllm_mlx.memory.backfill` to embed the new chunks"
        )
        return 0 if stats.errors == 0 else 1
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import sys

    sys.exit(main())


__all__ = [
    "CHUNKER_VERSION",
    "ChunkRow",
    "IndexerStats",
    "VaultIndexer",
    "chunk_markdown",
    "rebuild_vault_tables",
]
