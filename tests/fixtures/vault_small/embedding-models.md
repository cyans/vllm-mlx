# Embedding model survey

Compared bge-m3, multilingual-e5-base, and jina-embeddings-v3 on a
500-doc Korean corpus. bge-m3 wins on long-context recall by a small
margin but the difference is within noise.

## Decision

Use bge-m3 for the memory subsystem because mlx-embeddings already
ships a recipe. Revisit if Phase 4 evals show degradation.
