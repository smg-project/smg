# tokenizers v1 encode benchmark — 2026-10-13

Tested the optional backend against the default wrapper on upstream base `d52de821`, including the v1 1,024-entry word-cache bound and borrowed serial encode path.

Xeon Platinum 8573C, Linux x86_64, **4 CPU quota**, Rust 1.98.0, repository release profile. Model revisions/hashes and environment are in [environment.json](environment.json). All four models selected the v1 path in the enabled builds. Measurements are CPU text encode calls, not HTTP latency.

## Single caller, no L0/L1

P50 in ms, median of three 300 ms rounds. 128/2,048-token rows use flat text; 50k/80k rows use eight chat boundaries with a changing leading nonce. Word/piece caches are warm. “Saved” and “Less time” compare the current wrapper with v1 internal single-thread encoding; “Speedup” is current / v1.

| Model | Tokens | Current ms | v1 / 1 thread ms | Saved ms | Less time | Speedup | v1 / 4 threads ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax-m3 | 128 | 0.015 | 0.003 | 0.012 | 82.8% | 5.83x | 0.003 |
| minimax-m3 | 2,048 | 0.263 | 0.038 | 0.224 | 85.4% | 6.83x | 0.039 |
| minimax-m3 | 50,000 | 7.199 | 1.072 | 6.127 | 85.1% | 6.71x | 0.651 |
| minimax-m3 | 80,000 | 10.851 | 1.651 | 9.200 | 84.8% | 6.57x | 2.015 |
| glm-5.3 | 128 | 0.012 | 0.001 | 0.010 | 88.0% | 8.32x | 0.001 |
| glm-5.3 | 2,048 | 0.198 | 0.019 | 0.179 | 90.3% | 10.31x | 0.019 |
| glm-5.3 | 50,000 | 5.852 | 0.498 | 5.354 | 91.5% | 11.75x | 0.421 |
| glm-5.3 | 80,000 | 8.785 | 0.791 | 7.994 | 91.0% | 11.10x | 0.644 |
| qwen3-8b | 128 | 0.012 | 0.002 | 0.010 | 86.7% | 7.50x | 0.002 |
| qwen3-8b | 2,048 | 0.202 | 0.023 | 0.178 | 88.5% | 8.66x | 0.023 |
| qwen3-8b | 50,000 | 5.229 | 0.583 | 4.646 | 88.9% | 8.97x | 0.368 |
| qwen3-8b | 80,000 | 8.697 | 0.915 | 7.782 | 89.5% | 9.50x | 0.538 |
| deepseek-v3.2 | 128 | 0.010 | 0.002 | 0.009 | 84.4% | 6.40x | 0.002 |
| deepseek-v3.2 | 2,048 | 0.236 | 0.023 | 0.213 | 90.3% | 10.36x | 0.023 |
| deepseek-v3.2 | 50,000 | 6.692 | 0.599 | 6.093 | 91.0% | 11.17x | 0.490 |
| deepseek-v3.2 | 80,000 | 11.027 | 1.019 | 10.008 | 90.8% | 10.82x | 0.767 |

## Cache behavior at 80,000 tokens

L0: default 256 MiB / 10,000 entries. L1: 1 GiB logical budget. `repeat` has 100% L0 hits; `prefix90`/`prefix99` have 100% L1 hits and zero L0 hits; `unique` has no hits. Both levels prefer L0. Entries/counters are verified by the summarizer. Each row compares the same input workload and backend; all values are P50 ms.

| Model | Backend | Scenario | Off | L0 | L1 | Both |
|---|---|---|---|---|---|---|
| MiniMax M3 | Current | unique | 10.851 | 12.342 | 14.620 | 17.825 |
| MiniMax M3 | Current | repeat | 15.365 | 0.074 | 2.789 | 0.073 |
| MiniMax M3 | Current | prefix90 | 10.904 | 11.557 | 2.813 | 3.451 |
| MiniMax M3 | Current | prefix99 | 11.939 | 12.807 | 2.017 | 2.468 |
| MiniMax M3 | v1 / 1 thread | unique | 1.651 | 2.177 | 4.439 | 4.714 |
| MiniMax M3 | v1 / 1 thread | repeat | 1.730 | 0.074 | 1.843 | 0.073 |
| MiniMax M3 | v1 / 1 thread | prefix90 | 1.644 | 2.216 | 1.960 | 2.551 |
| MiniMax M3 | v1 / 1 thread | prefix99 | 1.672 | 2.240 | 1.744 | 2.368 |
| GLM-5.3 | Current | unique | 8.785 | 8.770 | 9.999 | 10.048 |
| GLM-5.3 | Current | repeat | 8.879 | 0.069 | 1.508 | 0.069 |
| GLM-5.3 | Current | prefix90 | 8.234 | 8.810 | 1.393 | 1.869 |
| GLM-5.3 | Current | prefix99 | 8.096 | 8.727 | 0.612 | 1.093 |
| GLM-5.3 | v1 / 1 thread | unique | 0.791 | 1.319 | 1.986 | 2.581 |
| GLM-5.3 | v1 / 1 thread | repeat | 0.822 | 0.070 | 0.618 | 0.070 |
| GLM-5.3 | v1 / 1 thread | prefix90 | 0.786 | 1.430 | 0.651 | 1.245 |
| GLM-5.3 | v1 / 1 thread | prefix99 | 0.807 | 1.306 | 0.555 | 1.097 |

With v1, full-repeat L0 hits take about 0.07 ms for both models. L1 alone improves GLM prefix90/prefix99 from 0.786/0.807 ms to 0.651/0.555 ms, but does not improve these MiniMax prefixes. Enabling both caches can cost more than L1 alone on an L0 miss.

Full-repeat L0 hits avoid encoding but still hash the input and copy IDs. L1 scans boundaries, hashes prefixes and assembles output even on a hit; a faster encoder makes that overhead harder to amortize. The shared-prefix working set fits in a few MiB, so this does not establish a benefit from a larger capacity or from enabling L1 for every workload. Unique-input insertion/eviction costs are included.

## Concurrent callers, 80,000 tokens, caches off

Closed-loop `unique` workload. Throughput is total requests / wall time; P95 is the median round P95. Compare internal 1 versus 4 threads under the same 4-CPU quota.

| Model | Callers | Current req/s | v1 1T req/s | v1 4T req/s | Current P95 ms | v1 1T P95 ms | v1 4T P95 ms |
|---|---|---|---|---|---|---|---|
| MiniMax M3 | 4 | 231 | 1316 | 459 | 21.02 | 3.62 | 30.75 |
| MiniMax M3 | 16 | 60 | 1389 | 532 | 111.25 | 29.72 | 66.01 |
| MiniMax M3 | 32 | 231 | 1537 | 865 | 188.25 | 46.33 | 69.02 |
| GLM-5.3 | 4 | 284 | 3195 | 1800 | 18.65 | 1.91 | 5.65 |
| GLM-5.3 | 16 | 284 | 3014 | 1633 | 108.34 | 20.80 | 33.26 |
| GLM-5.3 | 32 | 292 | 3056 | 1023 | 143.23 | 33.42 | 82.70 |

The current MiniMax 16-caller throughput is anomalously below its 4- and 32-caller results; treat that row as exploratory rather than a stable scaling estimate.

Internal parallelism is a deployment choice, not an unconditional speedup. Extra workers can compete with caller threads; use the throughput/tail measurements for the intended concurrency rather than extrapolating single-caller latency.

## Batch API and model load

Batch size 8, each input 80k tokens; median of five warm calls, complete returned IDs checked. Load times are one observation after the reference model has been loaded (warm filesystem), not a cold-start distribution. Both libraries/models are retained in the opt-in implementation; loading additionally converts/rebuilds the v1 pipeline.

| Model | Current batch ms | v1 1T batch ms | v1 4T batch ms | Current load ms | v1 load ms |
|---|---|---|---|---|---|
| MiniMax M3 | 92.11 | 15.09 | 8.45 | 542 | 1894 |
| GLM-5.3 | 71.81 | 8.13 | 6.96 | 754 | 2247 |
| Qwen3-8B | 69.45 | 7.59 | 3.99 | 518 | 2054 |
| DeepSeek-V3.2 | 89.29 | 9.39 | 5.34 | 320 | 1228 |

## Validation and limits

- Default tokenizer suite passed (473 tests), followed by all six final adapter tests, including the two added normalization/fallback cases. The complete v1 suite passed **475 tests** with internal parallelism enabled; 16 pre-existing/opt-in fixture tests were skipped in that suite.
- The explicit four-model parity test passed separately with internal parallelism off and on: Unicode fuzzing, long inputs, special tokens, both special-token flags, batch ordering, decoding, and 16-way cache calls. It asserts the v1 backend was actually selected.
- Existing cache-retention, dropped-tokenizer reclamation and warmed long-piece allocation tests pass. The v1 cache is bounded to 1,024 entries per scratch context. Peak production RSS across many resident models is not measured.
- Tokenizer all-target/all-feature Clippy, nightly formatting, Python Ruff and gateway `cargo check -p smg --features tokenizers-v1 --lib` passed. Workspace all-feature Clippy was attempted but blocked by missing system `opencv4` (the optional video module); that check is not claimed as passing.
- This is an opt-in deterministic BPE encode path. Unsupported models/configurations keep the existing backend. Existing decoding, chat rendering, metadata and EOS semantics remain in the shared wrapper.
- Synthetic inputs reuse words while changing a nonce; no claim of fully novel documents. Three fixed-order rounds on one host; small differences and sparse high-concurrency P99s are not statistically conclusive. L0/L1 first-miss samples do not clear the model word cache.

Measured 648 configurations and 5,934,990 timed calls. See [results.csv](results.csv) for all sizes, scenarios, cache settings, P50/P95/P99, sample counts, round ranges, first-miss costs, hit rates and absolute/relative comparisons. [README](README.md) documents the complete protocol and commands; `run.py` retains per-round JSONL and model hashes when reproduced.
