# Long document for sliding window test

Section one introduces the problem space and motivation for the work.
We discuss why naive approaches fail and why a more principled solution
is required. The introduction also covers the historical context.

## Section two

Section two derives the algorithm from first principles. Each step is
motivated by an empirical observation from prior work. We avoid premature
optimization in this section to keep the exposition clean.

## Section three

Section three benchmarks the implementation on three datasets of
increasing size. Wall-clock latency stays roughly linear in input
length, with a small constant overhead from the sliding window
machinery. Memory usage is dominated by the embedding cache.

## Section four

Section four discusses limitations and threats to validity. The most
important caveat is that all benchmarks were run on a single machine
configuration. Cross-machine variance is documented in the appendix.

## Section five

Section five concludes with three open problems and an invitation to
collaborate. The accompanying code is released under Apache 2.0.
