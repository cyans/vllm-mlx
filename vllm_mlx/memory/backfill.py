# SPDX-License-Identifier: Apache-2.0
"""Backfill embeddings for already-indexed vault chunks.

@CODE:MEMORY-01/backfill

Phase 1 indexed every chunk into ``fts_chunks`` but skipped embeddings
(the dense path didn't exist yet). Phase 2 introduces the dense path,
so users with a populated Phase-1 DB need a one-time pass that:

* Walks ``vault_chunks WHERE chunk_id NOT IN vec_chunks``
* Embeds in batches of ``MEMORY_EMBED_BATCH``
* INSERTs each batch into ``vec_chunks`` inside its own transaction
* Resumes safely after kill -9 (the next run picks up where the
  previous one stopped because the WHERE clause is idempotent)

This module is also runnable as a script::

    python -m vllm_mlx.memory.backfill                  # reads MEMORY_* env
    python -m vllm_mlx.memory.backfill --batch 32        # ad-hoc override
    python -m vllm_mlx.memory.backfill --force-rebuild   # REQ-U4 recovery

REQ-N4 isolation: this module never touches the chat path. It is a
standalone CLI that talks to the same SQLite file the live server
uses; SQLite WAL mode keeps the live server's reads consistent.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

from .config import resolve_memory_config
from .embedder import Embedder
from .store import MemoryStore, open_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackfillStats:
    """Summary of a backfill run for the operator log."""

    total: int
    succeeded: int
    failed: int
    duration_s: float
    skipped_extension_missing: bool = False
    aborted_reason: str | None = None

    def as_log(self) -> str:
        if self.aborted_reason:
            return f"[memory] backfill aborted: {self.aborted_reason}"
        return (
            f"[memory] backfill done: total={self.total} "
            f"succeeded={self.succeeded} failed={self.failed} "
            f"duration={self.duration_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def backfill(
    store: MemoryStore,
    embedder: object,  # _SupportsEmbed; loose typing avoids hard import
    *,
    batch: int = 16,
    log_every: int = 250,
) -> BackfillStats:
    """Re-embed every ``vault_chunks`` row that lacks a vector.

    Pre-conditions:
    * ``store.open()`` has been called.
    * ``store.vec_loaded`` is True (sqlite-vec is attached).

    The embedder's API is duck-typed: it must expose ``encode_batch``
    returning a list of float32-packed blobs. Pass either a real
    :class:`vllm_mlx.memory.embedder.Embedder` or a fake in tests.

    Returns a :class:`BackfillStats` describing the run. Errors during
    individual batches are logged and counted; they do not abort the
    overall run unless the embedder enters a permanent failed state.
    """
    started = time.monotonic()

    if not store.vec_loaded:
        logger.warning(
            "[memory] backfill skipped: sqlite-vec is not loaded on this "
            "connection (BM25 still works). Install sqlite-vec and retry."
        )
        return BackfillStats(
            total=0,
            succeeded=0,
            failed=0,
            duration_s=time.monotonic() - started,
            skipped_extension_missing=True,
        )

    total_target = store.count_chunks_missing_vectors()
    if total_target == 0:
        logger.info("[memory] backfill: nothing to do; all chunks have vectors")
        return BackfillStats(
            total=0,
            succeeded=0,
            failed=0,
            duration_s=time.monotonic() - started,
        )

    logger.info(
        "[memory] backfill: %d chunks to embed (batch=%d)",
        total_target,
        batch,
    )

    succeeded = 0
    failed = 0

    for chunk_batch in store.iter_chunks_missing_vectors(batch_size=batch):
        chunk_ids = [c[0] for c in chunk_batch]
        texts = [c[1] for c in chunk_batch]

        try:
            blobs = embedder.encode_batch(texts)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] backfill: encode_batch raised for %d chunks",
                len(texts),
            )
            blobs = []

        if not blobs:
            failed += len(chunk_ids)
            # Embedder is dead. Bail out — retrying every batch in a
            # tight loop is wasteful when the underlying error is
            # likely permanent (model load failed, OOM, etc.).
            if getattr(embedder, "disabled", lambda: False)():
                logger.error(
                    "[memory] backfill aborted: embedder reported permanent "
                    "failure. Restart the server after addressing the load "
                    "error (see prior log lines)."
                )
                return BackfillStats(
                    total=total_target,
                    succeeded=succeeded,
                    failed=failed,
                    duration_s=time.monotonic() - started,
                    aborted_reason="embedder_disabled",
                )
            continue

        if len(blobs) != len(chunk_ids):
            # Partial response — log and skip this batch, backfill
            # itself remains resumable on the next run because we did
            # not insert anything for these chunk_ids.
            logger.warning(
                "[memory] backfill: embedder returned %d/%d vectors; "
                "skipping batch",
                len(blobs),
                len(chunk_ids),
            )
            failed += len(chunk_ids)
            continue

        try:
            with store.transaction():
                store.insert_vectors_batch(
                    rows=zip(chunk_ids, blobs),
                    source_type="vault",
                )
            succeeded += len(chunk_ids)
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] backfill: insert failed for batch of %d",
                len(chunk_ids),
            )
            failed += len(chunk_ids)
            continue

        if succeeded and (succeeded % log_every == 0 or succeeded < log_every):
            logger.info(
                "[memory] backfill progress: %d/%d done (%.1f%%)",
                succeeded,
                total_target,
                100.0 * succeeded / max(1, total_target),
            )

    return BackfillStats(
        total=total_target,
        succeeded=succeeded,
        failed=failed,
        duration_s=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vllm_mlx.memory.backfill",
        description=(
            "Embed every vault_chunks row that lacks a vector. Reads its "
            "configuration from the same MEMORY_* environment variables "
            "the live server uses (SPEC-MEMORY-01 §9)."
        ),
    )
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="override MEMORY_EMBED_BATCH for this run (default: env value)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="emit a progress line every N successful embeddings",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "DROP and recreate vec_chunks before backfilling. Required "
            "when switching MEMORY_EMBED_MODEL or MEMORY_EMBED_DIM "
            "(REQ-U4 recovery path)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many chunks need embedding, then exit without writing",
    )
    return p


def _force_rebuild_vec_table(store: MemoryStore) -> None:
    """Drop ``vec_chunks`` and the ``meta.embedding_*`` keys.

    Used by ``--force-rebuild`` to reset the embedding state when the
    operator wants to switch models. The vault_chunks rows are left
    alone — the next backfill will re-embed them.
    """
    if not store.vec_loaded:
        logger.warning(
            "[memory] --force-rebuild: vec extension not loaded, nothing to drop"
        )
        return
    with store.transaction():
        store.conn.execute("DROP TABLE IF EXISTS vec_chunks")
        store.conn.execute(
            "DELETE FROM meta WHERE key IN "
            "('embedding_model','embedding_dim','embedding_initialized')"
        )
    # Re-create at the configured dim by reopening the connection.
    # Simpler: just call _try_load_sqlite_vec again so the IF NOT
    # EXISTS template runs for the same dim.
    store._try_load_sqlite_vec()  # noqa: SLF001 - intentional re-init
    logger.info("[memory] --force-rebuild: vec_chunks recreated")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    config = resolve_memory_config(os.environ)
    if not config.enabled:
        logger.error(
            "MEMORY_ENABLED is not set. Set MEMORY_ENABLED=1 (and "
            "MEMORY_VAULT_PATH / MEMORY_DB_PATH) before running backfill."
        )
        return 2

    logger.info(
        "[memory] backfill starting: db=%s model=%s dim=%d batch=%d",
        config.db_path,
        config.embed_model,
        config.embed_dim,
        config.embed_batch,
    )

    try:
        store = open_store(config.db_path, embed_dim=config.embed_dim)
    except Exception:  # noqa: BLE001
        logger.exception("[memory] failed to open store at %s", config.db_path)
        return 1

    try:
        if args.force_rebuild:
            _force_rebuild_vec_table(store)

        # REQ-U4 — verify model identity before we touch any vectors.
        ok, reason = store.assert_embed_compat(
            model=config.embed_model, dim=config.embed_dim
        )
        if not ok:
            logger.error("[memory] %s", reason)
            return 3

        if args.dry_run:
            n = store.count_chunks_missing_vectors()
            print(
                f"[memory] dry-run: {n} chunks need embedding "
                f"(model={config.embed_model}, dim={config.embed_dim})"
            )
            return 0

        embedder = Embedder.from_config(config)
        # Eagerly load so an unrunnable model fails fast rather than
        # after the first batch attempt.
        if not embedder.load():
            logger.error(
                "[memory] embedder failed to load; backfill cannot proceed. "
                "Hint: pre-fetch with `huggingface-cli download %s`",
                config.embed_model,
            )
            return 4

        # Record identity now that we have a confirmed-loadable model.
        store.record_embedding_identity(
            model=config.embed_model, dim=config.embed_dim
        )

        stats = backfill(
            store,
            embedder,
            batch=int(args.batch or config.embed_batch),
            log_every=int(args.log_every),
        )
        print(stats.as_log())
        if stats.aborted_reason:
            return 5
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
