# Full ByteLevel decode results

Measured 2026-10-13 on macOS-26.5.2-arm64-arm-64bit, Rust 1.98.0, system allocator.
Base: `fc27e1e23da4441f843a8987ee1392bc14179834`. Source/binary/model identities are in
[metadata.json](metadata.json); the final patch is the opt-in table decoder in this PR.
The workspace bench profile uses opt-level=3, thin LTO and one codegen unit.

Default build versus `tokenizers-v1` build, three alternating warm process pairs,
200 ms per configuration with complete 32-call batches. Each number below is the
median of three mean closed-loop observations. All 340 configurations and their
paired min/max speedups are in [comparison.csv](comparison.csv); the 2,040 raw
observations and call counts are in [rounds.csv](rounds.csv). See
[README.md](README.md) for the complete measurement contract and reproduction.

## Main workload

Mixed English/Chinese/code/emoji, skip specials, one caller. These component
speedups do not establish HTTP latency, model throughput or TTFT gains.

| Model | Actual IDs | Default us | Opt-in us | Speedup |
|---|---:|---:|---:|---:|
| minimax-m3 | 128 | 8.20 | 0.46 | 17.81x |
| minimax-m3 | 2048 | 121.38 | 7.03 | 17.26x |
| minimax-m3 | 50000 | 2860.02 | 171.03 | 16.72x |
| minimax-m3 | 80000 | 4629.16 | 276.80 | 16.72x |
| glm-5.3 | 128 | 8.37 | 0.49 | 17.25x |
| glm-5.3 | 2048 | 121.97 | 7.07 | 17.25x |
| glm-5.3 | 50000 | 2874.25 | 171.17 | 16.79x |
| glm-5.3 | 80000 | 4611.15 | 269.45 | 17.11x |
| qwen3-8b | 128 | 8.22 | 0.44 | 18.52x |
| qwen3-8b | 2048 | 117.31 | 6.74 | 17.40x |
| qwen3-8b | 50000 | 2838.20 | 163.03 | 17.41x |
| qwen3-8b | 80000 | 4530.08 | 262.63 | 17.25x |
| deepseek-v3.2 | 128 | 8.06 | 0.46 | 17.49x |
| deepseek-v3.2 | 2048 | 120.31 | 6.90 | 17.44x |
| deepseek-v3.2 | 50000 | 2894.57 | 164.03 | 17.65x |
| deepseek-v3.2 | 80000 | 4505.39 | 268.77 | 16.76x |

## Input-shape coverage

Ranges across four models and both skip-special settings, one caller. Random
vocabulary IDs include unknown IDs. Byte fragments include incomplete UTF-8 and
invalid bytes. Empty/single-token results are available in the CSV and especially
sensitive to loop/timer overhead.

| Scenario | IDs | Minimum speedup | Maximum speedup |
|---|---:|---:|---:|
| mixed | 2048 | 16.84x | 17.44x |
| mixed | 80000 | 16.43x | 17.25x |
| ascii | 2048 | 20.73x | 23.03x |
| ascii | 80000 | 20.03x | 23.15x |
| unicode | 2048 | 12.16x | 16.02x |
| unicode | 80000 | 14.01x | 16.46x |
| special | 2048 | 13.99x | 19.73x |
| special | 80000 | 15.10x | 19.40x |
| random | 2048 | 13.85x | 15.16x |
| random | 80000 | 7.45x | 9.01x |
| byte-fragments | 2048 | 16.69x | 18.84x |
| byte-fragments | 80000 | 16.92x | 18.71x |

## Concurrent callers

Shared immutable tokenizer, mixed inputs, skip specials. Mean worker time includes
contention; throughput is aggregate completed calls divided by the longest worker
interval. Threads are created and warmed before the timed barrier. This laptop is
not a server scalability prediction.

| Model | IDs | Callers | Default us | Opt-in us | Speedup | Default requests/s | Opt-in requests/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax-m3 | 2048 | 4 | 201.47 | 7.34 | 27.46x | 19702 | 545007 |
| minimax-m3 | 2048 | 16 | 444.51 | 11.14 | 39.91x | 34989 | 1435358 |
| minimax-m3 | 80000 | 4 | 6901.39 | 274.76 | 25.12x | 550 | 14509 |
| minimax-m3 | 80000 | 16 | 11857.89 | 433.80 | 27.34x | 1248 | 36037 |
| glm-5.3 | 2048 | 4 | 127.94 | 7.31 | 17.51x | 30937 | 547194 |
| glm-5.3 | 2048 | 16 | 288.19 | 11.16 | 25.82x | 53674 | 1431711 |
| glm-5.3 | 80000 | 4 | 4960.39 | 278.32 | 17.82x | 796 | 14325 |
| glm-5.3 | 80000 | 16 | 10083.33 | 432.58 | 23.31x | 1292 | 35932 |
| qwen3-8b | 2048 | 4 | 126.16 | 6.92 | 18.22x | 31508 | 577511 |
| qwen3-8b | 2048 | 16 | 234.12 | 10.67 | 21.93x | 67273 | 1497815 |
| qwen3-8b | 80000 | 4 | 4712.58 | 266.69 | 17.67x | 846 | 14917 |
| qwen3-8b | 80000 | 16 | 8836.14 | 402.09 | 21.98x | 1636 | 38676 |
| deepseek-v3.2 | 2048 | 4 | 123.47 | 6.91 | 17.88x | 32301 | 578966 |
| deepseek-v3.2 | 2048 | 16 | 225.83 | 10.97 | 20.59x | 69862 | 1458234 |
| deepseek-v3.2 | 80000 | 4 | 4721.98 | 264.84 | 17.83x | 845 | 14947 |
| deepseek-v3.2 | 80000 | 16 | 8574.25 | 415.32 | 20.64x | 1646 | 37613 |

## Streaming: unchanged implementation

No streaming optimization is included. Below are negative controls for 2,048 IDs,
one caller, skip specials; the CSV also includes skip=false. Variations can come
from feature-dependent binary layout, extra encoder state and host noise; they
must not be presented as a streaming optimization. All individual output pieces
and finishes match between builds, including malformed and split-byte scenarios.

| Model | Scenario | Default us | Opt-in us | Ratio |
|---|---|---:|---:|---:|
| minimax-m3 | mixed | 45.59 | 45.06 | 1.01x |
| minimax-m3 | special | 44.46 | 43.29 | 1.03x |
| minimax-m3 | random | 71.97 | 69.49 | 1.04x |
| minimax-m3 | byte-fragments | 71.17 | 67.97 | 1.05x |
| glm-5.3 | mixed | 45.72 | 45.84 | 1.00x |
| glm-5.3 | special | 44.77 | 44.73 | 1.00x |
| glm-5.3 | random | 69.43 | 70.93 | 0.98x |
| glm-5.3 | byte-fragments | 69.65 | 66.69 | 1.04x |
| qwen3-8b | mixed | 45.48 | 44.13 | 1.03x |
| qwen3-8b | special | 43.84 | 43.54 | 1.01x |
| qwen3-8b | random | 70.24 | 68.17 | 1.03x |
| qwen3-8b | byte-fragments | 70.34 | 67.23 | 1.05x |
| deepseek-v3.2 | mixed | 46.28 | 46.27 | 1.00x |
| deepseek-v3.2 | special | 43.15 | 43.43 | 0.99x |
| deepseek-v3.2 | random | 69.58 | 69.38 | 1.00x |
| deepseek-v3.2 | byte-fragments | 69.62 | 68.31 | 1.02x |

## Nonselected decoder controls

Synthetic 2,048-ID inputs. Sequence(ByteLevel), ByteFallback+Fuse and no-decoder
keep the original 0.23 path. The plain ByteLevel fixture is a positive control.

| Decoder | Default us | Opt-in us | Ratio |
|---|---:|---:|---:|
| `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}` | 103.43 | 7.91 | 13.07x |
| `{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]}` | 102.68 | 101.25 | 1.01x |
| `{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}` | 68.23 | 69.25 | 0.99x |
| `none` | 44.48 | 45.91 | 0.97x |

## Load and memory

No table builder or storage changes are introduced: the decoder reuses the
existing Arc-backed offsets/kind/byte slab, so incremental persistent decode
storage is zero. Output storage is per call, initially four bytes per ID and grown
as needed; invalid UTF-8 can require a second lossy String allocation. This is a
source-level accounting statement, not an allocation-counter or RSS measurement.

Wrapper construction below excludes the already-loaded HF tokenizer clone.
The opt-in constructor also builds the v1 **encoder** introduced by #3356; its
extra time and memory are not caused by this decode patch. JSON-load observations
and every wrapper-load sample are in [loads.csv](loads.csv). Filesystem caches
were warm; no cold-start or peak-RSS claim is made.

| Model | Default wrapper ms | Opt-in wrapper ms |
|---|---:|---:|
| minimax-m3 | 65.78 | 614.00 |
| glm-5.3 | 54.63 | 580.27 |
| qwen3-8b | 51.12 | 447.21 |
| deepseek-v3.2 | 45.02 | 375.59 |

## Validation and limits

- Default tokenizer suite: **479 passed, 0 failed, 17 ignored, 0 filtered**.
  Opt-in suite: **479 passed, 0 failed, 17 ignored, 0 filtered**. Counts include
  integration targets and doctests; external fixture tests retain their existing
  ignores/conditional skips, so this is not full Bellwether coverage.
- Explicit four-model `v1_encode --include-ignored`: **8 passed, 0 ignored**,
  covering Unicode, batch ordering, cache use and concurrent calls.
- Each benchmark build: **3,892,414** full-decode parity cases plus **3,200**
  concurrent comparisons, all identical to 0.23.2. Every vocabulary ID alone and
  with context, unknown IDs, both special-token settings, and 10,000 deterministic
  multi-ID sequences per model. Not exhaustive over all token combinations.
- Added unit regressions enumerate every byte pair with both skip settings,
  special/Unicode added tokens, unknown IDs, sparse vocabulary and fallback
  decoders. Complete tokenizer targets cover existing stop/accounting/recovery.
- All full-output and length-delimited stream-piece digests agree across the six
  timed processes. Synthetic controls are validated before timing starts.
- Workspace nightly fmt, tokenizer all-target Clippy in both feature modes, and
  runner Ruff checks pass. Existing Cargo-manifest/Clippy config diagnostics are
  not new source warnings. Local workspace/GPU/server tests were not run.
- Builds and correctness finish before timing. A one-second Unix process-list
  monitor rejects a run if it sees a compiler. It cannot eliminate other host
  load or catch every brief process; three warm alternating rounds and raw
  observations expose, but do not remove, measurement noise.

Only these four pinned models and synthetic inputs were measured. Linux,
production jemalloc, production traffic, engine E2E and GPU/cluster throughput are
not verified. No cluster was started. The release profile differs from this bench
profile; use the supplied harness on the intended deployment host before making
production claims.
