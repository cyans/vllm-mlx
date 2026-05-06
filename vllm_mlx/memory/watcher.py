# SPDX-License-Identifier: Apache-2.0
"""Vault watcher: incremental, debounced, FSEvents-backed indexer.

@CODE:MEMORY-01/watcher

Phase 4 implementation of REQ-E2 (vault incremental update within 5s).
Spawns a :mod:`watchdog` ``Observer`` rooted at ``MEMORY_VAULT_PATH`` and
forwards file create / modify / delete / move events to an asyncio
queue. A long-lived consumer task drains the queue, debounces per-path
events within ``MEMORY_WATCHER_DEBOUNCE_MS`` milliseconds, and applies
the corresponding indexer operation atomically against the store.

REQ invariants honored here:

* REQ-N2 — paths whose resolved real form escapes the vault root are
  rejected (symlink-escape guard reuses the same logic as
  :class:`VaultIndexer`).
* REQ-N3 — denylist / allowlist globs filter every event before it
  reaches the DB.
* REQ-N4 — every loop body is wrapped in ``try/except Exception`` so a
  single bad event cannot kill the watcher; the asyncio task survives
  and processes the next event.
* Atomic re-index — modify events DELETE the file's old vec_chunks
  rows, run :meth:`MemoryStore.replace_chunks_for_file` (which already
  deletes vault_chunks + fts_chunks), then re-embed + INSERT. All four
  tables move together inside a single SQLite transaction so a half-
  update is impossible.

The watchdog observer thread NEVER touches the store directly; it only
hands events to the asyncio loop via ``loop.call_soon_threadsafe``,
which keeps the SQLite connection bound to the loop thread (matches the
contract documented in :class:`MemoryStore`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .indexer import (
    CHUNKER_VERSION,
    _glob_match,  # noqa: PLC2701 - intentional reuse of indexer's matcher
    chunk_markdown,
)
from .store import MemoryStore, chunk_id_for, file_sha256

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types we care about. Translated from watchdog so the test suite
# can exercise the consumer without instantiating the real Observer.
# ---------------------------------------------------------------------------
EVENT_ADDED = "added"
EVENT_MODIFIED = "modified"
EVENT_DELETED = "deleted"
EVENT_MOVED = "moved"


@dataclass
class _PendingEvent:
    """One coalesced event waiting for the debounce window to elapse.

    The "last event wins per path" rule is implemented by overwriting
    ``kind`` and ``last_seen`` on each new event for the same path,
    EXCEPT that a delete supersedes any prior add/modify (since the
    file is gone and re-indexing it would crash on missing bytes).
    """

    kind: str
    last_seen: float
    src_path: str
    # Only populated for ``moved`` events; the destination path inside
    # the vault root.
    dst_path: str | None = None


@dataclass
class WatcherStats:
    """Aggregate counters surfaced via :meth:`VaultWatcher.snapshot`.

    Lightweight diagnostic struct; no SQLite reads. Mirrors the style
    of :class:`vllm_mlx.memory.indexer.IndexerStats`.
    """

    events_received: int = 0
    events_processed: int = 0
    events_dropped: int = 0
    add_or_modify_ok: int = 0
    deletes_ok: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Pure event consumer (no watchdog import required at construction time)
# ---------------------------------------------------------------------------
class VaultWatcher:
    """FSEvents-backed incremental indexer for the vault.

    Public surface:

    * :meth:`run` — long-lived asyncio task entry point. The MCP child
      schedules this via ``asyncio.create_task`` after the initial scan.
    * :meth:`stop` — graceful shutdown (cancels the consumer + stops
      the watchdog observer).
    * :meth:`snapshot` — returns a :class:`WatcherStats` copy for logs.
    * :meth:`process_event_for_test` — synchronous helper that dispatches
      a single event (used by tests + the polling fallback). NOT part of
      the runtime path.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        vault_root: Path,
        denylist: tuple[str, ...] = (),
        allowlist: tuple[str, ...] = (),
        embedder: Any | None = None,
        embed_disabled: bool = False,
        debounce_ms: int = 500,
        # Internal: the asyncio queue size cap. We do NOT block the
        # observer thread when full; instead we drop the oldest pending
        # event for that path. The default is generous enough for any
        # human-paced vault edit pattern.
        queue_max: int = 4096,
    ):
        self.store = store
        self.vault_root = vault_root.expanduser().resolve()
        self.denylist = denylist
        self.allowlist = allowlist
        self.embedder = embedder
        self.embed_disabled = bool(embed_disabled)
        self.debounce_seconds = max(0.05, float(debounce_ms) / 1000.0)
        self._queue_max = int(queue_max)

        # Wired up lazily inside :meth:`run` so unit tests of the pure
        # dispatcher do not need a running event loop or real observer.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_RawEvent] | None = None
        self._observer: Any | None = None
        self._stop_event: asyncio.Event | None = None

        # Per-path coalescing dict; flushed by :meth:`_dispatch_ready`.
        self._pending: dict[str, _PendingEvent] = {}
        self._stats = WatcherStats()

    # -------------------------------------------------------- public API

    async def run(self) -> None:
        """Long-lived asyncio task. Never returns under normal operation.

        Spawns a watchdog Observer in a background thread and drains
        events from a thread-safe queue. Wraps every iteration in a
        try/except so a transient failure (DB lock, decode error)
        never kills the loop (REQ-N4).
        """
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_max)
        self._stop_event = asyncio.Event()

        try:
            self._observer = self._build_observer()
        except Exception:  # noqa: BLE001 - REQ-N4: degrade to no-watch
            logger.exception(
                "[memory] failed to start vault watcher; incremental "
                "updates disabled (initial scan still applies)"
            )
            return

        if self._observer is None:
            logger.info(
                "[memory] vault watcher disabled (no observer constructed)"
            )
            return

        logger.info(
            "[memory] vault watcher running: root=%s debounce=%.2fs",
            self.vault_root,
            self.debounce_seconds,
        )

        try:
            while not self._stop_event.is_set():
                try:
                    # Wake at most every half-debounce so pending events
                    # never wait long enough to violate REQ-E2's 5s budget.
                    await self._tick(
                        max_wait=max(0.05, self.debounce_seconds / 2.0)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - REQ-N4
                    logger.exception(
                        "[memory] watcher tick failed; will retry"
                    )
                    self._stats.errors += 1
                    # Avoid a hot crash loop; honour the debounce as a
                    # back-off floor.
                    await asyncio.sleep(self.debounce_seconds)
        finally:
            await self._shutdown_observer()

    async def stop(self) -> None:
        """Request a graceful shutdown of the watcher loop."""
        if self._stop_event is not None:
            self._stop_event.set()

    def snapshot(self) -> WatcherStats:
        """Return a *copy* of the live counters for diagnostic logging."""
        # ``dataclasses.replace`` would import deepcopy machinery; the
        # explicit copy below is allocation-cheap and obvious.
        s = self._stats
        return WatcherStats(
            events_received=s.events_received,
            events_processed=s.events_processed,
            events_dropped=s.events_dropped,
            add_or_modify_ok=s.add_or_modify_ok,
            deletes_ok=s.deletes_ok,
            errors=s.errors,
        )

    # ------------------------------------------------------- test hooks

    def process_event_for_test(self, raw: _RawEvent) -> None:
        """Absorb + dispatch a single event synchronously.

        Used by ``tests/test_memory_watcher.py`` so the suite never
        depends on a real watchdog Observer or a running asyncio loop.
        """
        self._absorb(raw)
        # In tests we want immediate dispatch; pin the clock so the
        # debounce check passes regardless of wall-time.
        self._dispatch_ready(now=time.monotonic() + self.debounce_seconds + 1.0)

    # -------------------------------------------------------- internals

    async def _tick(self, *, max_wait: float) -> None:
        """One iteration of the consumer loop.

        Drains all immediately available events from the queue, then
        processes any pending entries whose debounce window has elapsed.
        ``max_wait`` is the longest we will block waiting for the FIRST
        event of the tick — once the first arrives we drain the queue
        with no further wait so bursts collapse cheaply.
        """
        assert self._queue is not None  # set by run()
        # Wait for at least one event OR a timeout so dispatch runs.
        try:
            raw = await asyncio.wait_for(self._queue.get(), timeout=max_wait)
        except asyncio.TimeoutError:
            self._dispatch_ready(now=time.monotonic())
            return
        self._absorb(raw)
        # Drain anything else already enqueued without blocking.
        while True:
            try:
                raw = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._absorb(raw)
        self._dispatch_ready(now=time.monotonic())

    def _absorb(self, raw: _RawEvent) -> None:
        """Update the pending dict from one raw event.

        Coalescing rules:

        * Multiple modify/add events for the same path within the
          window collapse into one (last wins).
        * A delete for a path supersedes any prior add/modify on that
          path (we cannot index a file that is no longer there).
        * A move with src + dst inside the vault is folded into
          (delete src, add dst).
        * A move with dst outside the vault becomes a single delete
          on src.
        """
        self._stats.events_received += 1
        now = time.monotonic()

        if raw.kind == EVENT_MOVED:
            # Treat as delete(src) + add(dst). dst may already be
            # filtered out by the observer pattern matcher, but we
            # double-check via _safe_relpath.
            self._enqueue_pending(
                path=raw.src_path, kind=EVENT_DELETED, now=now
            )
            if raw.dst_path:
                self._enqueue_pending(
                    path=raw.dst_path, kind=EVENT_ADDED, now=now
                )
            return

        self._enqueue_pending(path=raw.src_path, kind=raw.kind, now=now)

    def _enqueue_pending(self, *, path: str, kind: str, now: float) -> None:
        # Filter: must be inside the vault, must not match denylist,
        # must match allowlist when one is set. Symlink-escape is also
        # caught here.
        rel = self._safe_relpath(Path(path))
        if rel is None:
            self._stats.events_dropped += 1
            return

        existing = self._pending.get(path)
        if existing is None:
            self._pending[path] = _PendingEvent(
                kind=kind, last_seen=now, src_path=path
            )
            return

        # Coalesce: delete always wins over add/modify; otherwise the
        # latest event type takes precedence.
        if kind == EVENT_DELETED or existing.kind == EVENT_DELETED:
            existing.kind = EVENT_DELETED
        else:
            existing.kind = kind
        existing.last_seen = now

    def _dispatch_ready(self, *, now: float) -> None:
        """Process every pending event whose debounce window has elapsed."""
        if not self._pending:
            return
        ready_paths = [
            p
            for p, ev in self._pending.items()
            if (now - ev.last_seen) >= self.debounce_seconds
        ]
        for p in ready_paths:
            ev = self._pending.pop(p)
            try:
                self._dispatch_one(ev)
                self._stats.events_processed += 1
            except Exception:  # noqa: BLE001 - REQ-N4
                logger.exception(
                    "[memory] failed to dispatch %s for %s", ev.kind, p
                )
                self._stats.errors += 1

    def _dispatch_one(self, ev: _PendingEvent) -> None:
        """Apply a single coalesced event to the store."""
        if ev.kind == EVENT_DELETED:
            self._on_deleted(Path(ev.src_path))
        elif ev.kind in (EVENT_ADDED, EVENT_MODIFIED):
            # The on-disk file is the source of truth; treat add and
            # modify identically — both branches re-chunk + re-embed
            # the current contents and replace whatever was previously
            # indexed for that path.
            self._on_added_or_modified(Path(ev.src_path))
        else:  # pragma: no cover - defensive
            logger.warning("[memory] unknown event kind: %r", ev.kind)

    # --------------------------------------------- index ops (DB writes)

    def _on_added_or_modified(self, path: Path) -> None:
        """Re-index a single file (atomic across vault/chunks/fts/vec)."""
        rel = self._safe_relpath(path)
        if rel is None:
            return
        try:
            stat = path.stat()
        except FileNotFoundError:
            # The file disappeared between the event and our handler;
            # treat as a delete instead.
            self._on_deleted(path)
            return
        except OSError:
            logger.exception("[memory] could not stat %s", path)
            return

        if stat.st_size > _MAX_FILE_BYTES:
            logger.warning(
                "[memory] watcher: skipping oversized file (%d bytes): %s",
                stat.st_size,
                rel,
            )
            return

        try:
            sha = file_sha256(path)
        except OSError:
            logger.exception("[memory] could not hash %s", path)
            return

        # Idempotent fast-path: when the on-disk SHA matches what we
        # already stored we can short-circuit BUT only when the chunks
        # are actually present (a previous failed insert could have
        # written the file row without chunks). We check via the chunk
        # count below; for the typical no-op case this saves us from
        # re-chunking and re-embedding on every editor save event.
        stored_sha = self.store.get_vault_file_sha(rel)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.exception("[memory] could not read %s", path)
            return

        chunks = chunk_markdown(text)
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

        if stored_sha == sha:
            existing_file_id = self.store.get_vault_file_id(rel)
            if existing_file_id is not None:
                existing_chunks = self.store.get_chunk_ids_for_file(
                    existing_file_id
                )
                if len(existing_chunks) == len(stamped):
                    return  # truly unchanged; no-op

        with self.store.transaction():
            # upsert_vault_file returns ``cur.lastrowid`` which is the
            # AUTOINCREMENT counter, NOT the existing row's PK on
            # ON CONFLICT UPDATE — see the comment inside store.py.
            # That works for the initial-scan path because the indexer
            # never reuses a file_id within the same transaction, but
            # the watcher's modify flow MUST consult vault_files
            # directly to get the real PK before issuing a chunk
            # replacement against it.
            self.store.upsert_vault_file(
                path=rel,
                mtime=stat.st_mtime,
                sha256=sha,
                size=stat.st_size,
            )
            file_id = self.store.get_vault_file_id(rel)
            if file_id is None:  # pragma: no cover - upsert just wrote it
                logger.error(
                    "[memory] watcher: file row vanished after upsert: %s",
                    rel,
                )
                return
            # Atomic re-index step 1: drop OLD vec rows for the file
            # before replace_chunks_for_file orphans them.
            self.store.delete_vectors_for_file(file_id)
            # Step 2: replace_chunks_for_file already deletes old
            # vault_chunks + fts_chunks for this file_id and inserts
            # the new ones.
            self.store.replace_chunks_for_file(
                file_id=file_id,
                source_path=rel,
                chunks=stamped,
            )
            # Step 3: best-effort embedding. Failures here are logged
            # but do NOT abort the txn — the BM25 index stays consistent
            # with the file row even when the dense path degrades.
            self._embed_chunks_safely(stamped)
            # Mirror the v2 chunker version stamp the indexer would
            # write on initial scan, so the first watcher write to a
            # fresh DB does not appear to "skip" it.
            try:
                if self.store.get_meta("chunker_version") is None:
                    self.store.set_meta("chunker_version", CHUNKER_VERSION)
            except Exception:  # noqa: BLE001 - REQ-N4
                logger.exception(
                    "[memory] watcher: could not stamp meta.chunker_version"
                )

        self._stats.add_or_modify_ok += 1

    def _on_deleted(self, path: Path) -> None:
        """Drop every row for the file that just disappeared."""
        rel = self._safe_relpath(path)
        if rel is None:
            return
        file_id = self.store.get_vault_file_id(rel)
        if file_id is None:
            # Already gone (e.g. delete event after vault wipe); no-op.
            return
        with self.store.transaction():
            self.store.delete_file_and_chunks(file_id)
        self._stats.deletes_ok += 1

    def _embed_chunks_safely(self, stamped: list[dict[str, Any]]) -> None:
        """Best-effort dense indexing for a batch of just-inserted chunks.

        Mirrors :meth:`vllm_mlx.memory.indexer.VaultIndexer._embed_chunks_safely`
        but lives here so the watcher does not depend on the indexer
        class. Skips silently when the dense path is unavailable.
        """
        if self.embed_disabled or self.embedder is None:
            return
        if not getattr(self.store, "vec_loaded", False):
            return
        try:
            texts = [str(c["text"]) for c in stamped]
            blobs = self.embedder.encode_batch(texts)
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] watcher: embed failed for %d chunks; BM25 still wrote",
                len(stamped),
            )
            return
        if not blobs or len(blobs) != len(stamped):
            return
        try:
            self.store.insert_vectors_batch(
                rows=zip(
                    (str(c["chunk_id"]) for c in stamped),
                    blobs,
                    strict=True,
                ),
                source_type="vault",
            )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] watcher: vec insert failed for %d chunks",
                len(stamped),
            )

    # -------------------------------------------------- path safety net

    def _safe_relpath(self, path: Path) -> str | None:
        """Resolve and validate a path against the vault root + glob lists.

        Returns the POSIX-relative form on success, ``None`` for
        symlink escapes (REQ-N2), missing roots, or paths matching a
        deny / allow list (REQ-N3).
        """
        try:
            real = path.resolve()
        except (OSError, RuntimeError):
            return None
        try:
            rel = real.relative_to(self.vault_root)
        except ValueError:
            logger.warning(
                "[memory] watcher: refusing path outside vault: %s -> %s",
                path,
                real,
            )
            return None

        # We only index ``*.md`` files; the observer's pattern filter
        # already drops other extensions but a programmatic test event
        # could slip through.
        if real.suffix.lower() != ".md":
            return None

        rel_posix = rel.as_posix()
        if any(_glob_match(p, rel_posix) for p in self.denylist):
            return None
        if self.allowlist and not any(
            _glob_match(p, rel_posix) for p in self.allowlist
        ):
            return None
        return rel_posix

    # --------------------------------------------- watchdog plumbing

    def _build_observer(self) -> Any | None:
        """Construct + start the watchdog Observer.

        Lazy-imports :mod:`watchdog` so the module import does not pay
        the cost when the operator selects ``MEMORY_INDEXER=poll``.
        Returns ``None`` if the import fails (degraded mode).
        """
        try:
            # Lazy import: watchdog ≥ 6.0 is in pyproject deps but the
            # operator may have removed it on a custom build. Falling
            # back to "no incremental indexer" is preferable to a crash.
            from watchdog.events import (  # noqa: PLC0415
                FileCreatedEvent,
                FileDeletedEvent,
                FileModifiedEvent,
                FileMovedEvent,
                PatternMatchingEventHandler,
            )
            from watchdog.observers import Observer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "[memory] watchdog not installed; vault watcher disabled "
                "(install with: pip install 'watchdog>=6.0')"
            )
            return None

        loop = self._loop
        queue = self._queue
        if loop is None or queue is None:  # pragma: no cover - defensive
            return None

        # Build the handler via a helper instead of an inline class so
        # ruff's "first argument must be self" rule does not collide
        # with the closure capture of ``loop`` and ``queue``.
        handler = _build_event_handler(
            patterns=["*.md"],
            loop=loop,
            queue=queue,
            base_handler_cls=PatternMatchingEventHandler,
            event_classes=(
                FileCreatedEvent,
                FileModifiedEvent,
                FileDeletedEvent,
                FileMovedEvent,
            ),
        )
        observer = Observer()
        # ``recursive=True`` so deep notes (Notes/Reading/2025/...) are
        # observed; the denylist guard prunes ``.obsidian/**`` etc. on
        # the consumer side.
        observer.schedule(handler, str(self.vault_root), recursive=True)
        observer.daemon = True
        observer.start()
        return observer

    async def _shutdown_observer(self) -> None:
        """Stop the watchdog observer cleanly on cancellation/exit."""
        if self._observer is None:
            return
        try:
            self._observer.stop()
            # Block briefly in a thread to avoid blocking the loop while
            # FSEvents drains. ``join`` accepts a timeout so a stuck
            # observer cannot wedge shutdown.
            await asyncio.get_running_loop().run_in_executor(
                None, self._observer.join, 2.0
            )
        except Exception:  # noqa: BLE001 - shutdown best-effort
            logger.exception("[memory] error shutting down vault observer")
        finally:
            self._observer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RawEvent:
    """Plain-data event passed from the observer thread to the asyncio loop.

    Public so tests can construct one without importing watchdog.
    """

    kind: str
    src_path: str
    dst_path: str | None = None
    # Arrival timestamp (monotonic). Optional so tests can pin it.
    received_at: float = field(default_factory=time.monotonic)


def _safe_put_nowait(queue: asyncio.Queue[_RawEvent], event: _RawEvent) -> None:
    """Put an event onto the asyncio queue without raising on a full queue.

    Called from inside ``loop.call_soon_threadsafe`` so we are now on
    the loop thread and safe to touch the queue. On a full queue we
    drop the OLDEST event (FIFO eviction) so a noisy editor cannot
    shadow a later real change.
    """
    try:
        queue.put_nowait(event)
        return
    except asyncio.QueueFull:
        # Make room by discarding the oldest pending raw event. The
        # debounce dict will still see the new one, so this only loses
        # very-low-priority "redundant modify" notifications.
        try:
            queue.get_nowait()
            queue.put_nowait(event)
        except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
            logger.warning(
                "[memory] watcher queue overflow; dropping event for %s",
                event.src_path,
            )


# Match the indexer's ceiling so the watcher and initial scan stay in sync.
# Importing the constant directly creates a hard cycle; keep a local mirror.
_MAX_FILE_BYTES = 4 * 1024 * 1024


def _build_event_handler(
    *,
    patterns: list[str],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[_RawEvent],
    base_handler_cls: Any,
    event_classes: tuple[Any, ...],  # noqa: ARG001 - kept for future filtering
) -> Any:
    """Return a watchdog ``PatternMatchingEventHandler`` instance.

    Encapsulated as a helper so the closure-capturing handler does not
    have to be a nested class (which trips ruff N805 because each handler
    method needs to bind to ``self`` AND the captured queue/loop). The
    helper keeps the lambda-style bridge tiny and unit-testable.
    """

    def _push(raw: _RawEvent) -> None:
        try:
            loop.call_soon_threadsafe(_safe_put_nowait, queue, raw)
        except RuntimeError:
            # Loop already closed (process teardown); drop silently.
            return

    class _Adapter(base_handler_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(
                patterns=patterns,
                ignore_directories=True,
                case_sensitive=False,
            )

        def on_created(self, event: Any) -> None:
            _push(_RawEvent(EVENT_ADDED, str(event.src_path), None))

        def on_modified(self, event: Any) -> None:
            _push(_RawEvent(EVENT_MODIFIED, str(event.src_path), None))

        def on_deleted(self, event: Any) -> None:
            _push(_RawEvent(EVENT_DELETED, str(event.src_path), None))

        def on_moved(self, event: Any) -> None:
            _push(
                _RawEvent(
                    EVENT_MOVED,
                    str(event.src_path),
                    str(event.dest_path),
                )
            )

    return _Adapter()


__all__ = [
    "EVENT_ADDED",
    "EVENT_DELETED",
    "EVENT_MODIFIED",
    "EVENT_MOVED",
    "VaultWatcher",
    "WatcherStats",
    "_RawEvent",
]
