# Optional tokenizers v1 encoding

Build the gateway with `cargo build --release -p smg --features tokenizers-v1`.
The default build continues using tokenizers 0.23. The optional dependency is
aliased as `tokenizers_v1` and pinned to `1.0.0-rc.2`. Enabling it accelerates
single and batch text encoding inside `HuggingFaceTokenizer`. Full decode with a
plain ByteLevel decoder also reuses SMG's existing byte table (see
[decode benchmarks](../decode/README.md)). Streaming, other decoders, model metadata
and chat rendering retain their existing implementation. Both libraries are
present in the opt-in build; tokenizers 0.23 is still required. The default build
does not enable either acceleration.

The pipeline is constructed **after** tokenizer configuration overrides, so
configured BOS/EOS processing is preserved. Results use `Encoding::Plain`, like
the existing native encoder. Non-BPE models, stochastic BPE, padding/truncation,
Split configurations without a usable engine and configurations rejected by the v1
reader retain the existing encoder. `uses_v1_encoder()` exposes the selection.
The Split guard avoids an rc.2 reader issue: it can accept a regex without a
usable engine, even with `regex` enabled. V1 word caches are limited to the same
1,024 entries as the existing fallback. The RC reader ignores cache capacity, so
the adapter rebuilds its BPE model with that bound and reuses the parsed pipeline
configuration. Short/serial calls use the borrowed `encode_into` API to avoid
input copies and batch-handle allocations.

No per-model changes to the process-wide thread pool are made. Set
`TOKENIZERS_PARALLELISM=false` for caller-thread encoding; unset/true uses the
library's internal parallelism. The benchmark explicitly compares 1 and 4 internal
threads. More internal threads can reduce single-request latency while hurting
throughput under concurrent requests and a limited CPU quota.

## Reproduce

Run from the repository root with the pinned Rust toolchain. The Python helper
uses only the standard library. Model revisions are pinned in `models.json`;
only tokenizer JSON is downloaded, never model weights.

```sh
python crates/tokenizer/benches/v1/run.py prepare
python crates/tokenizer/benches/v1/run.py run
python crates/tokenizer/benches/v1/run.py summarize
```

Override `--models` / `--output` to keep artifacts elsewhere. `run` builds both
versions before measuring and writes `current-1.jsonl`, `v1-1.jsonl`, `v1-4.jsonl`.
Model records contain revision, BLAKE3, encoder selection and load time. Load time
follows reference loading, so it is not cold-filesystem startup time.
`summarize` validates cache counters and produces `summary.csv`, `comparison.csv`,
`auxiliary.json` (load and batch observations), and a compact `table.md`.

## Measurement contract

- Four models: MiniMax M3, GLM-5.3, Qwen3-8B, DeepSeek-V3.2.
- **Actual** 128 / 2,048 / 50,000 / 80,000 tokens, calibrated per model and checked
  on every timed output. Mixed English, Chinese and code; full IDs checked against
  0.23 before each configuration on four nonce variants, twice each.
- Short inputs: flat text, caches off. Long inputs: flat text plus chat text with
  eight real special-token boundaries. Internal parallelism depends on these
  boundaries; flat text is a separate control.
- `unique`: mutate a 64-token nonce before all boundaries (no L0/L1 reuse).
  `repeat`: identical input (L0 hits). `prefix90` / `prefix99`: mutate the tail
  after a 90% / 99% token prefix (L1 hits, L0 misses).
- Long chat inputs cover off / L0 / L1 / both. L0 uses its default 256 MiB and
  10,000-entry limits; L1 has a **1 GiB budget**, not a 1 GiB preallocation.
  Budgeted bytes are the cache's logical accounting, not RSS.
- MiniMax/GLM additionally cover 4 / 16 / 32 caller threads for unique, repeat
  and 90% prefixes with caches off/both. Other models cover single requests.
- Each configuration: three 300 ms closed-loop rounds, waiting for started
  requests to finish; two warm calls per worker. Latency covers the encode call,
  including cache operations and CPU contention. It excludes input mutation,
  thread startup and result destruction; throughput includes their loop costs.
- P50/P95/P99 are medians of the three round quantiles; throughput is total
  requests / total wall time. Round P50 min/max and sample counts are retained.
  Sparse high-concurrency tails are exploratory, not production SLO estimates.
- Three first-miss observations clear L0/L1, not the model's internal piece cache.
  Full process/model startup is separate. Batch size 8 is timed separately, with
  full ID checks and five observations after warm-up.
- Model word/piece caches are warm. Nonce mutation prevents whole-input/prefix
  reuse but does **not** make every word novel. Synthetic workload, fixed-order
  runs, one host; small differences are not statistical evidence. No HTTP,
  template rendering, network or model inference is timed.

Latency saved = current − v1; time reduction = saved / current; speedup = current /
v1. Cache comparisons use the same backend, scenario, length and concurrency.
A large L1 budget helps only when reusable prefixes otherwise cannot stay resident;
this single-prefix workload fits in a few MiB and does not test multi-session
capacity benefits. It must not be used to recommend enabling L1 universally.

## Functional validation

```sh
cargo test --locked -p llm-tokenizer
TOKENIZERS_PARALLELISM=false cargo test --locked -p llm-tokenizer --features tokenizers-v1
SMG_V1_TOKENIZER_DIRS=/tmp/smg-tokenizer-models/minimax-m3,/tmp/smg-tokenizer-models/glm-5.3,/tmp/smg-tokenizer-models/qwen3-8b,/tmp/smg-tokenizer-models/deepseek-v3.2 \
  TOKENIZERS_PARALLELISM=false cargo test --locked -p llm-tokenizer --features tokenizers-v1 \
  --test v1_encode -- --ignored
# Repeat with TOKENIZERS_PARALLELISM=true to exercise the internal pool.
cargo clippy --locked -p llm-tokenizer --all-targets --all-features -- -D warnings
cargo +nightly fmt --all -- --check
```

The new tests exercise empty/Unicode inputs, added-token matching flags,
`encode_special_tokens`, both `add_special_tokens` values, batch ordering and
empty batches, BOS/EOS config overrides, chat rendering, EOS metadata, full and
incremental decoding, cache hit/miss paths, and fallback configuration semantics.
The explicitly invoked real-model test adds deterministic Unicode fuzzing, long
inputs, model special tokens, 16-way concurrent cache calls and full ID/decode
comparisons. Existing tokenizer tests run with both default and v1 features in CI.
Prefix-cache comparisons use rendered chat semantics; arbitrary post-processors
that append EOS to each encoded segment are outside this cache workload.
