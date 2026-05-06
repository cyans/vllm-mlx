# API design notes

The memory_search tool returns a flat list of results to keep the model's
parsing job simple. Each result carries source_path, source_type,
timestamp, score, and excerpt.

Score is normalized to [0, 1] so the model has a stable confidence signal
across BM25 and dense backends.
