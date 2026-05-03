# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-4 vault watcher (REQ-E2 / REQ-N2 / REQ-N3 / REQ-N4).

@TEST:MEMORY-01/watcher

The real :mod:`watchdog` ``Observer`` is NEVER instantiated by the
test suite — every test calls :meth:`VaultWatcher.process_event_for_test`
directly with a synthesized :class:`_RawEvent`. This keeps the suite
fast, deterministic, and free of FSEvents permission requirements.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vllm_mlx.memory.store import open_store
from vllm_mlx.memory.watcher import (
    EVENT_ADDED,
    EVENT_DELETED,
    EVENT_MODIFIED,
    EVENT_MOVED,
    VaultWatcher,
    _RawEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_vault(tmp_path: Path) -> Path:
    """Build an empty vault directory inside ``tmp_path``."""
    root = tmp_path / "vault"
    root.mkdir()
    return root


def _write_md(root: Path, rel: str, body: str) -> Path:
    """Write a markdown file under ``root`` and return the absolute path."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _build_watcher(
    *,
    tmp_path: Path,
    denylist: tuple[str, ...] = (),
    allowlist: tuple[str, ...] = (),
    debounce_ms: int = 100,
) -> tuple[VaultWatcher, Path, object]:
    """Return ``(watcher, vault_root, store)``.

    The store is opened against ``tmp_path / 'memory.db'`` and must be
    closed by the test (use ``addfinalizer`` or a try/finally).
    """
    vault = _make_vault(tmp_path)
    store = open_store(tmp_path / "memory.db")
    watcher = VaultWatcher(
        store,
        vault_root=vault,
        denylist=denylist,
        allowlist=allowlist,
        debounce_ms=debounce_ms,
    )
    return watcher, vault, store


# ---------------------------------------------------------------------------
# Add events
# ---------------------------------------------------------------------------
class TestAddedEvent:
    def test_added_event_inserts_vault_file_row(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "note1.md", "# Title\n\nbody text")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            assert store.count_vault_files() == 1
            assert store.count_vault_chunks() >= 1
        finally:
            store.close()

    def test_added_event_inserts_fts_row(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "n.md", "# H\n\nunique-token-alpha")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            hits = store.search_bm25("unique-token-alpha", top_k=5)
            assert hits, "expected the new file to be searchable"
        finally:
            store.close()

    def test_added_event_for_non_md_file_ignored(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            # .txt files are NOT vault content (only *.md per SPEC §3).
            path = vault / "ignored.txt"
            path.write_text("nothing")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Modify events
# ---------------------------------------------------------------------------
class TestModifiedEvent:
    def test_modified_replaces_chunks_for_file(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "edit.md", "# A\n\nold body")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            old_chunk_ids = {
                r["chunk_id"]
                for r in store.conn.execute(
                    "SELECT chunk_id FROM vault_chunks"
                ).fetchall()
            }
            assert old_chunk_ids

            # Rewrite with completely different content. Chunk IDs are
            # derived from (path, char_offset, char_len) so the new
            # body MUST yield different IDs (the body length differs).
            path.write_text(
                "# A\n\nthis is a much longer replacement body"
                " with brand-new tokens like fresh-token-beta",
                encoding="utf-8",
            )
            watcher.process_event_for_test(
                _RawEvent(EVENT_MODIFIED, str(path))
            )

            new_chunk_ids = {
                r["chunk_id"]
                for r in store.conn.execute(
                    "SELECT chunk_id FROM vault_chunks"
                ).fetchall()
            }
            # File row count is still 1 (same path) but chunks rotated.
            assert store.count_vault_files() == 1
            assert new_chunk_ids != old_chunk_ids

            # Old chunks must be GONE from FTS too — search for an old
            # token returns nothing while the new token returns hits.
            assert store.search_bm25("fresh-token-beta", top_k=5)
            assert store.search_bm25("old body", top_k=5) == []
        finally:
            store.close()

    def test_modified_with_unchanged_content_is_noop(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "n.md", "# A\n\nsame content")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            chunk_count_before = store.count_vault_chunks()

            # Re-fire MODIFIED without changing the file. The fast-path
            # should detect the matching SHA and skip re-chunking.
            watcher.process_event_for_test(
                _RawEvent(EVENT_MODIFIED, str(path))
            )
            assert store.count_vault_chunks() == chunk_count_before
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Delete events
# ---------------------------------------------------------------------------
class TestDeletedEvent:
    def test_delete_removes_all_rows_for_file(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "doomed.md", "# x\n\nbody-token-gamma")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            assert store.count_vault_files() == 1
            assert store.count_vault_chunks() >= 1
            assert store.search_bm25("body-token-gamma", top_k=5)

            # Real-world: the file is unlinked BEFORE the delete event
            # arrives. Drop it on disk so resolve() returns the (now
            # missing) path; the watcher must still drop the DB rows.
            path.unlink()
            watcher.process_event_for_test(
                _RawEvent(EVENT_DELETED, str(path))
            )

            assert store.count_vault_files() == 0
            assert store.count_vault_chunks() == 0
            assert store.search_bm25("body-token-gamma", top_k=5) == []
        finally:
            store.close()

    def test_delete_for_unknown_file_is_noop(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            # Fire a delete for a path we never indexed; must not raise.
            path = vault / "ghost.md"
            watcher.process_event_for_test(
                _RawEvent(EVENT_DELETED, str(path))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Move events
# ---------------------------------------------------------------------------
class TestMoveEvent:
    def test_move_inside_vault_drops_src_and_indexes_dst(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            src = _write_md(vault, "old.md", "# H\n\nmovable-token")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(src))
            )
            assert store.search_bm25("movable-token", top_k=5)

            # Simulate `mv old.md sub/new.md` on disk before the event.
            dst = vault / "sub" / "new.md"
            dst.parent.mkdir(parents=True)
            src.rename(dst)
            watcher.process_event_for_test(
                _RawEvent(EVENT_MOVED, str(src), str(dst))
            )

            files = [
                r["path"]
                for r in store.conn.execute(
                    "SELECT path FROM vault_files"
                ).fetchall()
            ]
            assert files == ["sub/new.md"], files
            assert store.search_bm25("movable-token", top_k=5)
        finally:
            store.close()

    def test_move_outside_vault_is_a_delete(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            src = _write_md(vault, "leaving.md", "# H\n\nleaving-token")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(src))
            )
            assert store.count_vault_files() == 1

            # Move out of the vault. Watcher should treat as delete.
            outside_dst = tmp_path / "outside.md"
            src.rename(outside_dst)
            watcher.process_event_for_test(
                _RawEvent(EVENT_MOVED, str(src), str(outside_dst))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# REQ-N3 — denylist enforcement
# ---------------------------------------------------------------------------
class TestDenylist:
    def test_denylisted_obsidian_path_is_ignored(self, tmp_path):
        watcher, vault, store = _build_watcher(
            tmp_path=tmp_path, denylist=(".obsidian/**", "**/.trash/**")
        )
        try:
            obsi = _write_md(
                vault, ".obsidian/internal.md", "# config\n\nblob"
            )
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(obsi))
            )
            assert store.count_vault_files() == 0

            trashed = _write_md(
                vault, "Notes/.trash/old.md", "# bin\n\nblob"
            )
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(trashed))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()

    def test_allowlist_filters_out_unmatched_paths(self, tmp_path):
        watcher, vault, store = _build_watcher(
            tmp_path=tmp_path,
            allowlist=("Notes/**",),
        )
        try:
            allowed = _write_md(vault, "Notes/keep.md", "# K\n\nkeep")
            blocked = _write_md(vault, "Other/skip.md", "# S\n\nskip")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(allowed))
            )
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(blocked))
            )
            paths = sorted(
                r["path"]
                for r in store.conn.execute(
                    "SELECT path FROM vault_files"
                ).fetchall()
            )
            assert paths == ["Notes/keep.md"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# REQ-N2 — symlink escape rejection
# ---------------------------------------------------------------------------
class TestSymlinkEscape:
    def test_symlink_pointing_outside_vault_is_rejected(self, tmp_path):
        # File lives outside the vault; a symlink inside the vault
        # points at it. resolve() returns the OUTSIDE real path so
        # _safe_relpath must reject.
        outside = tmp_path / "outside" / "secret.md"
        outside.parent.mkdir()
        outside.write_text("# secret\nbody", encoding="utf-8")

        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            link = vault / "linked.md"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                pytest.skip("symlink not supported on this platform")

            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(link))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()

    def test_path_completely_outside_vault_is_rejected(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            external = tmp_path / "external.md"
            external.write_text("body", encoding="utf-8")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(external))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------
class TestDebounce:
    def test_rapid_modifies_collapse_to_one_dispatch(self, tmp_path):
        # Use a long debounce (1s) and an explicit dispatch trigger so
        # we can prove only ONE dispatch happens per path within the
        # window.
        watcher, vault, store = _build_watcher(
            tmp_path=tmp_path, debounce_ms=1000
        )
        try:
            path = _write_md(vault, "rapid.md", "# H\n\nfirst")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            chunk_count = store.count_vault_chunks()
            assert chunk_count >= 1

            # Hammer 10 modifies very quickly. Each event lands in the
            # debounce dict but only the LAST one should actually run
            # against the store. Because process_event_for_test uses a
            # forced "now" that satisfies the debounce, each call here
            # would normally dispatch — to test the coalescing we drive
            # _absorb directly.
            for _ in range(10):
                watcher._absorb(
                    _RawEvent(EVENT_MODIFIED, str(path))
                )
            # At this point ALL 10 events sit in the pending dict
            # under the same path. Only one dispatch should run.
            assert len(watcher._pending) == 1

            # Force the debounce to elapse.
            watcher._dispatch_ready(
                now=time.monotonic() + watcher.debounce_seconds + 1.0
            )
            # Pending must now be empty AND we must have processed
            # exactly one event (the coalesced final modify).
            assert watcher._pending == {}
            stats = watcher.snapshot()
            # one initial ADDED + one coalesced MODIFIED = 2 processed
            assert stats.events_processed == 2
        finally:
            store.close()

    def test_delete_supersedes_pending_add(self, tmp_path):
        watcher, vault, store = _build_watcher(
            tmp_path=tmp_path, debounce_ms=1000
        )
        try:
            path = _write_md(vault, "ephemeral.md", "# x\n\nbody")
            # Push add then delete into the dict in rapid succession.
            watcher._absorb(_RawEvent(EVENT_ADDED, str(path)))
            path.unlink()
            watcher._absorb(_RawEvent(EVENT_DELETED, str(path)))
            assert len(watcher._pending) == 1
            pending = next(iter(watcher._pending.values()))
            assert pending.kind == EVENT_DELETED

            watcher._dispatch_ready(
                now=time.monotonic() + watcher.debounce_seconds + 1.0
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# REQ-N4 — failure isolation
# ---------------------------------------------------------------------------
class TestFailureIsolation:
    def test_handler_exception_does_not_kill_loop(self, tmp_path, monkeypatch):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            path = _write_md(vault, "ok.md", "# H\n\nbody")

            # Make _on_added_or_modified raise on the FIRST call only.
            calls = {"n": 0}
            real_handler = watcher._on_added_or_modified

            def flaky(p):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("boom")
                real_handler(p)

            monkeypatch.setattr(
                watcher, "_on_added_or_modified", flaky
            )

            # First event raises → counted as error, NOT propagated.
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            stats1 = watcher.snapshot()
            assert stats1.errors == 1
            assert store.count_vault_files() == 0

            # Second event still succeeds — the loop survived.
            watcher.process_event_for_test(
                _RawEvent(EVENT_MODIFIED, str(path))
            )
            stats2 = watcher.snapshot()
            assert stats2.errors == 1
            assert store.count_vault_files() == 1
        finally:
            store.close()

    def test_disappeared_file_falls_back_to_delete(self, tmp_path):
        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            # Index a file then unlink it BEFORE re-firing ADDED. The
            # handler should detect the missing file and treat it as a
            # delete instead of crashing.
            path = _write_md(vault, "race.md", "# H\n\nbody")
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            assert store.count_vault_files() == 1

            path.unlink()
            watcher.process_event_for_test(
                _RawEvent(EVENT_ADDED, str(path))
            )
            assert store.count_vault_files() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Asyncio loop integration (mocked observer, no real watchdog)
# ---------------------------------------------------------------------------
class TestAsyncioLoop:
    def test_run_with_no_observer_exits_cleanly(self, tmp_path, monkeypatch):
        """When the observer cannot be built we degrade to no-watch + exit."""
        import asyncio as _asyncio

        watcher, vault, store = _build_watcher(tmp_path=tmp_path)
        try:
            monkeypatch.setattr(watcher, "_build_observer", lambda: None)
            # Should return without raising even though no observer ran.
            _asyncio.run(watcher.run())
        finally:
            store.close()

    def test_run_drains_queue_and_dispatches(self, tmp_path, monkeypatch):
        """End-to-end: enqueue a synthetic event, run the loop briefly."""
        import asyncio as _asyncio

        watcher, vault, store = _build_watcher(
            tmp_path=tmp_path, debounce_ms=50
        )
        try:
            # Replace _build_observer with a noop sentinel object so the
            # loop body actually runs (it returns early when None).
            class _FakeObserver:
                def stop(self):
                    pass

                def join(self, timeout=None):
                    pass

            monkeypatch.setattr(
                watcher, "_build_observer", lambda: _FakeObserver()
            )

            path = _write_md(vault, "live.md", "# A\n\nasync-run-token")

            async def driver():
                task = _asyncio.create_task(watcher.run())
                # Wait for the loop to wire up its queue.
                while watcher._queue is None:
                    await _asyncio.sleep(0.01)
                # Now drop an event into the queue; the loop should
                # absorb + dispatch within one debounce window.
                watcher._queue.put_nowait(
                    _RawEvent(EVENT_ADDED, str(path))
                )
                # 50ms debounce + 50ms dispatch slack.
                await _asyncio.sleep(0.3)
                import contextlib as _contextlib

                await watcher.stop()
                try:
                    await _asyncio.wait_for(task, timeout=2.0)
                except _asyncio.TimeoutError:
                    task.cancel()
                    with _contextlib.suppress(_asyncio.CancelledError):
                        await task

            _asyncio.run(driver())
            assert store.count_vault_files() == 1
        finally:
            store.close()

    def test_safe_put_nowait_evicts_oldest_when_full(self):
        """Queue overflow drops the OLDEST event, not the new one."""
        import asyncio as _asyncio

        from vllm_mlx.memory.watcher import _safe_put_nowait

        async def driver():
            queue: _asyncio.Queue[_RawEvent] = _asyncio.Queue(maxsize=2)
            queue.put_nowait(_RawEvent(EVENT_ADDED, "/a"))
            queue.put_nowait(_RawEvent(EVENT_ADDED, "/b"))
            # Now full — _safe_put_nowait should drop /a and accept /c.
            _safe_put_nowait(queue, _RawEvent(EVENT_ADDED, "/c"))
            kept = []
            while not queue.empty():
                kept.append(queue.get_nowait().src_path)
            assert kept == ["/b", "/c"]

        _asyncio.run(driver())
