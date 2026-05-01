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


def chunk_markdown(text: str) -> list[ChunkRow]:
    """Split a Markdown document into chunks.

    Strategy:
    1. Find all headers (``# H``..``###### H``) and treat each header
       as the start of a section.
    2. For each section, if the body fits inside ``WINDOW_CHARS``, emit
       one chunk. Otherwise emit overlapping windowed chunks.
    3. Anything before the first header is its own pre-section chunk
       with empty header text.

    Determinism: chunk IDs are produced from (path, char_offset,
    char_len) at the caller, so the same input file always yields the
    same chunk IDs.
    """
    if not text:
        return []

    # Build (start, end, header) sections.
    headers = list(_HEADER_RE.finditer(text))
    sections: list[tuple[int, int, str]] = []

    if not headers:
        sections.append((0, len(text), ""))
    else:
        # Pre-header content (could be frontmatter / intro).
        first = headers[0]
        if first.start() > 0:
            sections.append((0, first.start(), ""))
        for i, m in enumerate(headers):
            section_start = m.start()
            section_end = (
                headers[i + 1].start() if i + 1 < len(headers) else len(text)
            )
            header_text = m.group(2).strip()
            sections.append((section_start, section_end, header_text))

    chunks: list[ChunkRow] = []
    chunk_index = 0
    now = _utc_now_iso()

    for sec_start, sec_end, header in sections:
        section_text = text[sec_start:sec_end].strip()
        if not section_text:
            continue

        # Recover the offset of the trimmed text in the original buffer.
        leading_strip = len(text[sec_start:sec_end]) - len(
            text[sec_start:sec_end].lstrip()
        )
        absolute_offset = sec_start + leading_strip

        if len(section_text) <= WINDOW_CHARS:
            chunks.append(
                ChunkRow(
                    chunk_id="",  # filled in by caller
                    text=section_text,
                    header=header,
                    chunk_index=chunk_index,
                    char_offset=absolute_offset,
                    char_len=len(section_text),
                    timestamp=now,
                )
            )
            chunk_index += 1
        else:
            for off, piece in _sliding_windows(section_text):
                chunks.append(
                    ChunkRow(
                        chunk_id="",
                        text=piece,
                        header=header,
                        chunk_index=chunk_index,
                        char_offset=absolute_offset + off,
                        char_len=len(piece),
                        timestamp=now,
                    )
                )
                chunk_index += 1

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
    ):
        self.store = store
        # Resolve symlinks on the root once so per-file resolves can be
        # checked against a stable canonical path (REQ-N2).
        self.vault_root = vault_root.expanduser().resolve()
        self.denylist = denylist
        self.allowlist = allowlist

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


__all__ = [
    "ChunkRow",
    "IndexerStats",
    "VaultIndexer",
    "chunk_markdown",
]
