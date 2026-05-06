# Inference latency observations

Mac mini M4 Pro running Qwen3.6 35B-A3B 4-bit:

- TTFT around 280ms with prefix cache enabled
- Throughput sits at 18-22 tok/s in continuous batching
- Memory pressure peaks during the first prefix lookup

The unified memory architecture is a big help here.
