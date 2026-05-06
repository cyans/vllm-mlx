# SPDX-License-Identifier: Apache-2.0
"""Tests for the vault indexer and chunker.

@TEST:MEMORY-01/indexer

Phase 1 scope:
- chunker is deterministic and respects the sliding-window contract;
- initial scan finds nested ``*.md`` files;
- denylist (REQ-N3) prevents indexing ``.obsidian/**``;
- symlink escapes (REQ-N2) are refused;
- rescan over an unchanged vault is a no-op (sha256 idempotency,
  REQ-E1 setup for Phase 4 incremental).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vllm_mlx.memory.config import MEMORY_DEFAULT_DENYLIST
from vllm_mlx.memory.indexer import (
    CHUNKER_VERSION,
    WINDOW_CHARS,
    WINDOW_OVERLAP,
    VaultIndexer,
    chunk_markdown,
    rebuild_vault_tables,
)
from vllm_mlx.memory.store import chunk_id_for, open_store

# Cache the path so each test does not have to rebuild it.
FIXTURES_VAULT = Path(__file__).parent / "fixtures" / "vault_small"


@pytest.fixture
def store(tmp_path):
    s = open_store(tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def indexer(store):
    return VaultIndexer(
        store,
        vault_root=FIXTURES_VAULT,
        denylist=MEMORY_DEFAULT_DENYLIST,
    )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class TestChunker:
    def test_empty_text_yields_no_chunks(self):
        assert chunk_markdown("") == []

    def test_no_headers_yields_one_chunk(self):
        text = "Just a few lines of prose without any header.\nLine two."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].header == ""
        assert chunks[0].chunk_index == 0
        assert "header" in chunks[0].text or "prose" in chunks[0].text

    def test_single_header_yields_one_chunk_with_header(self):
        text = "# Title\n\nBody text under the header."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].header == "Title"

    def test_multiple_headers_yield_multiple_chunks(self):
        text = "# A\n\nbody A.\n\n## B\n\nbody B.\n\n## C\n\nbody C.\n"
        chunks = chunk_markdown(text)
        assert len(chunks) == 3
        assert [c.header for c in chunks] == ["A", "B", "C"]
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_pre_header_content_becomes_anonymous_chunk(self):
        # YAML frontmatter / intro paragraph before the first header.
        text = "---\ntitle: x\n---\n\nintro line\n\n# H1\n\nbody"
        chunks = chunk_markdown(text)
        assert len(chunks) >= 2
        # The first chunk is the pre-header content with empty header.
        assert chunks[0].header == ""
        # The next chunk is the section under H1.
        assert chunks[1].header == "H1"

    def test_long_section_is_windowed(self):
        # 6_000 chars should produce multiple windowed chunks.
        body = "x" * 6000
        text = f"# Long\n\n{body}\n"
        chunks = chunk_markdown(text)
        assert len(chunks) >= 3
        # Each chunk must respect the window cap.
        for c in chunks:
            assert c.char_len <= WINDOW_CHARS
        # Adjacent chunks overlap by approximately ``WINDOW_OVERLAP``.
        deltas = [
            chunks[i + 1].char_offset - chunks[i].char_offset
            for i in range(len(chunks) - 1)
        ]
        for delta in deltas:
            assert delta < WINDOW_CHARS, "windows must overlap"
            assert delta >= WINDOW_CHARS - WINDOW_OVERLAP - 1, (
                f"unexpectedly small step: {delta}"
            )

    def test_korean_text_chunks_correctly(self):
        text = "# 한국어 제목\n\n한국어 본문이 여기에 있다."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].header == "한국어 제목"
        assert "본문" in chunks[0].text

    def test_chunk_indices_are_dense_and_zero_based(self):
        text = "# A\n\nfoo\n\n# B\n\nbar\n\n# C\n\nbaz"
        chunks = chunk_markdown(text)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Initial scan
# ---------------------------------------------------------------------------


class TestInitialScan:
    def test_scan_indexes_all_md_files_in_fixture_vault(self, indexer, store):
        stats = indexer.initial_scan()
        # Fixture has 21 .md files total but 1 lives under .obsidian/ so
        # the deny list (REQ-N3 default) must skip it: 20 indexed.
        # (Includes ``multi_level_headers.md`` added for the v2 chunker
        # tests; the .obsidian/ deny list is still asserted below.)
        assert stats.errors == 0
        assert stats.files_indexed >= 18
        assert stats.files_indexed < 21  # .obsidian/plugins.md must be excluded
        assert stats.chunks_written > 0

    def test_obsidian_files_are_not_indexed(self, indexer, store):
        indexer.initial_scan()
        # The unique marker is in tests/fixtures/vault_small/.obsidian/plugins.md
        hits = store.search_bm25(
            "obsidian-internal-marker-do-not-index", top_k=10
        )
        assert hits == [], (
            "REQ-N3 violation: .obsidian/ content leaked into the index"
        )

    def test_nested_files_are_indexed(self, indexer, store):
        indexer.initial_scan()
        # "rutabaga aurora" and "magnolia tungsten" are unique markers
        # in deeply nested fixture files.
        hits = store.search_bm25("rutabaga aurora", top_k=5)
        assert hits, "nested file not indexed"
        assert any("nested-1" in h.source_path for h in hits)
        deep = store.search_bm25("magnolia tungsten", top_k=5)
        assert deep
        assert any("deep-note" in h.source_path for h in deep)

    def test_rescan_is_idempotent(self, indexer, store):
        first = indexer.initial_scan()
        assert first.files_indexed > 0

        second = indexer.initial_scan()
        # Nothing changed on disk — every file should be reported as
        # "unchanged" and zero chunks should be (re)written.
        assert second.files_indexed == 0
        assert second.chunks_written == 0
        assert second.files_unchanged == first.files_indexed

    def test_modified_file_is_reindexed(
        self, indexer, store, tmp_path, monkeypatch
    ):
        # Build a writable temp vault with a copy of one fixture so we
        # can mutate it without polluting the repo.
        vault = tmp_path / "vault"
        vault.mkdir()
        target = vault / "note.md"
        target.write_text("# H\n\nfirst body\n", encoding="utf-8")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        s1 = scoped.initial_scan()
        assert s1.files_indexed == 1

        target.write_text("# H\n\nsecond body different\n", encoding="utf-8")
        s2 = scoped.initial_scan()
        assert s2.files_indexed == 1
        # The new content must be searchable; the old content must not.
        assert store.search_bm25("second", top_k=3)
        assert store.search_bm25("first", top_k=3) == []

    def test_missing_vault_root_logs_and_returns_empty_stats(
        self, store, tmp_path
    ):
        scoped = VaultIndexer(
            store,
            vault_root=tmp_path / "does-not-exist",
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        stats = scoped.initial_scan()
        assert stats.files_seen == 0
        assert stats.files_indexed == 0
        assert stats.errors == 0


# ---------------------------------------------------------------------------
# Symlink escape (REQ-N2)
# ---------------------------------------------------------------------------


class TestSymlinkEscape:
    def test_refuses_symlink_escape(self, store, tmp_path):
        """REQ-N2: a symlink that points outside the vault root must not
        be indexed, even if its name has a ``.md`` extension.

        @TEST:MEMORY-01/security/symlink-escape
        """
        # Outside content that should be unreachable.
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.md"
        secret.write_text(
            "# Secret\n\ndo not index this content unique-token-zelnodorm\n",
            encoding="utf-8",
        )

        # Vault that contains a real note plus a symlink to the outside file.
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "real.md").write_text(
            "# Real\n\nlegit content here\n", encoding="utf-8"
        )
        try:
            os.symlink(secret, vault / "leaked.md")
        except (OSError, NotImplementedError):
            pytest.skip("symlink unsupported on this filesystem")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        scoped.initial_scan()

        # Real file is indexed.
        assert store.search_bm25("legit", top_k=3)
        # Outside-content marker must NOT be reachable from the index.
        leaked = store.search_bm25("zelnodorm", top_k=3)
        assert leaked == [], f"REQ-N2 violation: symlink escape leaked: {leaked}"


# ---------------------------------------------------------------------------
# Glob denylist + allowlist
# ---------------------------------------------------------------------------


class TestPathFilters:
    def test_custom_denylist_excludes_matching_files(self, store, tmp_path):
        vault = tmp_path / "vault"
        (vault / "private").mkdir(parents=True)
        (vault / "public.md").write_text("# Public\n\nworld\n", encoding="utf-8")
        (vault / "private" / "secret.md").write_text(
            "# Private\n\nshouldnotappear\n", encoding="utf-8"
        )

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=("private/**",),
        )
        scoped.initial_scan()
        assert store.count_vault_files() == 1
        assert store.search_bm25("shouldnotappear", top_k=3) == []

    def test_oversized_files_are_skipped(self, store, tmp_path, monkeypatch):
        from vllm_mlx.memory import indexer as indexer_mod

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "small.md").write_text(
            "# Small\n\ntiny content\n", encoding="utf-8"
        )
        # Force the threshold low so we don't have to write a real 4MB file.
        monkeypatch.setattr(indexer_mod, "MAX_FILE_BYTES", 32)
        (vault / "big.md").write_text(
            "# Big\n\n" + ("padding " * 200) + "\n", encoding="utf-8"
        )

        scoped = indexer_mod.VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        stats = scoped.initial_scan()
        assert stats.files_indexed == 1
        assert stats.files_skipped >= 1

    def test_unreadable_file_does_not_crash_scan(
        self, store, tmp_path, monkeypatch
    ):
        from vllm_mlx.memory.indexer import VaultIndexer

        vault = tmp_path / "vault"
        vault.mkdir()
        good = vault / "good.md"
        bad = vault / "bad.md"
        good.write_text("# Good\n\nworking content\n", encoding="utf-8")
        bad.write_text("# Bad\n\ndoesn't matter\n", encoding="utf-8")

        # Force ``file_sha256`` to raise on the bad path so we exercise
        # the per-file try/except (REQ-N4).
        from vllm_mlx.memory import indexer as indexer_mod

        original = indexer_mod.file_sha256

        def explode(path):
            if path.name == "bad.md":
                raise OSError("simulated read failure")
            return original(path)

        monkeypatch.setattr(indexer_mod, "file_sha256", explode)
        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        stats = scoped.initial_scan()
        assert stats.errors == 1
        assert stats.files_indexed == 1

    def test_allowlist_restricts_files(self, store, tmp_path):
        vault = tmp_path / "vault"
        (vault / "keep").mkdir(parents=True)
        (vault / "skip").mkdir(parents=True)
        (vault / "keep" / "k.md").write_text(
            "# Keep\n\nincluded keepword\n", encoding="utf-8"
        )
        (vault / "skip" / "s.md").write_text(
            "# Skip\n\nignored skipword\n", encoding="utf-8"
        )

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            allowlist=("keep/**",),
        )
        scoped.initial_scan()
        assert store.search_bm25("keepword", top_k=3)
        assert store.search_bm25("skipword", top_k=3) == []


# ---------------------------------------------------------------------------
# Chunker v2: header-path prefix
# ---------------------------------------------------------------------------


class TestHeaderPathPrefix:
    """v2 chunker: every chunk text starts with its parent header path.

    @TEST:MEMORY-01/chunker-v2
    """

    def test_header_path_prefix_attaches_to_body_chunk(self):
        """A single ``## H2`` followed by body produces ``"## H2\\n\\n<body>"``."""
        text = "## H2\n\nbody"
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].text.startswith("## H2\n\n"), chunks[0].text
        assert "body" in chunks[0].text

    def test_nested_h1_h2_h3_path_in_chunk(self):
        """``# A`` > ``## B`` > ``### C`` paths join with ``" > "``."""
        text = "# A\n## B\n### C\nbody"
        chunks = chunk_markdown(text)
        # Each section flushes when the next header arrives, so we get
        # one header-only chunk for A, one for B, then C with its body.
        c_chunk = next(c for c in chunks if "body" in c.text)
        assert c_chunk.text.startswith("# A > ## B > ### C\n\n"), c_chunk.text

    def test_header_only_section_emits_path_chunk(self):
        """A header with no body still emits a chunk containing the path."""
        text = "# A\n## B\n## C\nbody"
        chunks = chunk_markdown(text)
        # B has no body before C so we expect a header-only chunk for B
        # whose text is just the path.
        b_chunks = [c for c in chunks if c.text == "# A > ## B"]
        assert len(b_chunks) == 1, [c.text for c in chunks]
        # C has body so it carries the prefix + body.
        c_chunks = [c for c in chunks if c.text.startswith("# A > ## C\n\n")]
        assert len(c_chunks) == 1
        assert c_chunks[0].text == "# A > ## C\n\nbody"

    def test_consecutive_h2_levels_pop_h3_correctly(self):
        """An H2 after an H3 must pop the H3 from the stack.

        For ``## A`` → ``### A1`` → body1 → ``## B`` → body2:
        - A1's body chunk has prefix ``"## A > ### A1"``
        - B's body chunk has prefix ``"## B"`` only (A1 popped, A popped).
        """
        text = "## A\n### A1\nbody1\n## B\nbody2"
        chunks = chunk_markdown(text)
        a1_chunk = next(c for c in chunks if "body1" in c.text)
        b_chunk = next(c for c in chunks if "body2" in c.text)
        assert a1_chunk.text.startswith("## A > ### A1\n\n"), a1_chunk.text
        assert b_chunk.text.startswith("## B\n\n"), b_chunk.text
        # Ensure the H2 ``## A`` did NOT bleed into the B chunk.
        assert "## A" not in b_chunk.text.split("\n\n")[0]

    def test_pre_header_intro_has_no_prefix(self):
        """Content before the first header still emits with empty prefix."""
        text = "intro paragraph\n\n# H1\nbody"
        chunks = chunk_markdown(text)
        # The intro chunk has an empty header path so its text is just
        # the body content (matches v1 behaviour for pre-header text).
        intro = chunks[0]
        assert "intro" in intro.text
        assert not intro.text.startswith("# "), (
            "pre-header chunk must not carry a prefix"
        )

    def test_long_section_windows_each_carry_prefix(self):
        """Sliding-window pieces of a long section all share the prefix."""
        body = "x" * 6000
        text = f"# Long\n## Sub\n{body}"
        chunks = chunk_markdown(text)
        body_chunks = [c for c in chunks if "x" in c.text]
        assert len(body_chunks) >= 3
        for c in body_chunks:
            assert c.text.startswith("# Long > ## Sub\n\n"), c.text

    def test_chunker_version_constant_is_v2(self):
        """The version constant must be at v2 now that this code shipped."""
        assert CHUNKER_VERSION == "2"


# ---------------------------------------------------------------------------
# Chunker v2: meta + rebuild plumbing
# ---------------------------------------------------------------------------


class TestChunkerVersionMeta:
    def test_chunker_version_recorded_in_meta(self, store, tmp_path):
        """``initial_scan`` writes ``meta.chunker_version`` on first scan."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "n.md").write_text("# N\n\nbody\n", encoding="utf-8")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        scoped.initial_scan()
        assert store.get_meta("chunker_version") == CHUNKER_VERSION

    def test_mismatch_emits_warning_but_does_not_rebuild(
        self, store, tmp_path, caplog
    ):
        """A stale ``chunker_version`` triggers a warning, not a rebuild."""
        import logging

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "n.md").write_text("# N\n\nbody\n", encoding="utf-8")

        # Pre-populate meta with an OLD version.
        store.set_meta("chunker_version", "1")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        with caplog.at_level(logging.WARNING):
            scoped.initial_scan()
        # The warning was emitted...
        assert any(
            "chunker version mismatch" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
        # ...but the stored version stayed at "1" — we never auto-bump.
        assert store.get_meta("chunker_version") == "1"


class TestRebuild:
    def test_rebuild_wipes_existing_data(self, store, tmp_path):
        """``rebuild_vault_tables`` empties vault tables in one transaction."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "a.md").write_text("# A\n\nfirst\n", encoding="utf-8")
        (vault / "b.md").write_text("# B\n\nsecond\n", encoding="utf-8")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        scoped.initial_scan()
        assert store.count_vault_files() == 2
        assert store.count_vault_chunks() >= 2

        rebuild_vault_tables(store)

        assert store.count_vault_files() == 0
        assert store.count_vault_chunks() == 0
        # FTS must be wiped too — a search for old content returns nothing.
        assert store.search_bm25("first", top_k=3) == []
        assert store.search_bm25("second", top_k=3) == []
        # The chunker_version key is reset to the current code value.
        assert store.get_meta("chunker_version") == CHUNKER_VERSION

        # Re-indexing afterwards repopulates everything from scratch.
        scoped.initial_scan()
        assert store.count_vault_files() == 2
        assert store.search_bm25("first", top_k=3)
        assert store.search_bm25("second", top_k=3)


class TestChunkIdDeterminism:
    def test_chunk_id_deterministic_under_new_chunker(self, tmp_path):
        """Same fixture twice → identical chunk_ids both runs.

        Regression guard: chunker v2 changes the *text* of chunks but
        the chunk_id derivation (path + char_offset + char_len) must
        still be stable so a re-scan with no source change is a no-op.
        """
        vault = tmp_path / "vault"
        vault.mkdir()
        src = vault / "doc.md"
        src.write_text(
            "# H1\n## H2\nbody body body\n## H3\nmore body here\n",
            encoding="utf-8",
        )

        # Two independent stores, same input file.
        store_a = open_store(tmp_path / "a.db")
        store_b = open_store(tmp_path / "b.db")
        try:
            for s in (store_a, store_b):
                VaultIndexer(
                    s,
                    vault_root=vault,
                    denylist=MEMORY_DEFAULT_DENYLIST,
                ).initial_scan()

            ids_a = [
                row["chunk_id"]
                for row in store_a.conn.execute(
                    "SELECT chunk_id FROM vault_chunks ORDER BY chunk_index"
                ).fetchall()
            ]
            ids_b = [
                row["chunk_id"]
                for row in store_b.conn.execute(
                    "SELECT chunk_id FROM vault_chunks ORDER BY chunk_index"
                ).fetchall()
            ]
            assert ids_a == ids_b
            assert len(ids_a) > 0
            # Spot-check: each ID matches the path/offset/len recipe.
            row = store_a.conn.execute(
                "SELECT chunk_id, char_offset, char_len FROM vault_chunks "
                "ORDER BY chunk_index LIMIT 1"
            ).fetchone()
            expected = chunk_id_for(
                "doc.md", int(row["char_offset"]), int(row["char_len"])
            )
            assert row["chunk_id"] == expected
        finally:
            store_a.close()
            store_b.close()


class TestMultiLevelHeadersFixture:
    def test_fixture_chunks_carry_full_paths(self, store, tmp_path):
        """The new fixture exercises the full v2 chunker behaviour.

        @TEST:MEMORY-01/chunker-v2/fixture
        """
        # Copy the canonical fixture into a writable scratch vault so we
        # do not pollute the shared FIXTURES_VAULT counts.
        vault = tmp_path / "vault"
        vault.mkdir()
        src = (FIXTURES_VAULT / "multi_level_headers.md").read_text(
            encoding="utf-8"
        )
        (vault / "multi_level_headers.md").write_text(src, encoding="utf-8")

        scoped = VaultIndexer(
            store,
            vault_root=vault,
            denylist=MEMORY_DEFAULT_DENYLIST,
        )
        scoped.initial_scan()

        rows = store.conn.execute(
            "SELECT text FROM vault_chunks ORDER BY chunk_index"
        ).fetchall()
        texts = [r["text"] for r in rows]

        # 1. Section A's body carries the H1 prefix.
        assert any(
            t.startswith("# Top > ## Section A\n\n") and "body of A" in t
            for t in texts
        ), texts

        # 2. Subsection A1 keeps the full path under Section A.
        assert any(
            t.startswith("# Top > ## Section A > ### Subsection A1\n\n")
            and "detail for A1" in t
            for t in texts
        ), texts

        # 3. Subsection A2 has no body before Section B closes it,
        #    so it emits a header-only chunk with the path alone.
        assert (
            "# Top > ## Section A > ### Subsection A2" in texts
        ), texts

        # 4. Section B has no body before Subsection B1, but Subsection
        #    B1 itself does — so we expect a header-only chunk for B
        #    plus a body chunk for B1 with the right path (no Section A!).
        assert "# Top > ## Section B" in texts, texts
        assert any(
            t.startswith("# Top > ## Section B > ### Subsection B1\n\n")
            and "some content for B1" in t
            for t in texts
        ), texts
        # Crucially: B1 must NOT carry Section A as an ancestor.
        b1_chunk = next(
            t for t in texts if "some content for B1" in t
        )
        assert "Section A" not in b1_chunk

        # 5. Section C's body carries the simple path under Top.
        assert any(
            t.startswith("# Top > ## Section C\n\n") and "final content" in t
            for t in texts
        ), texts
