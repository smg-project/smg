# Opt-in full ByteLevel decoding

`cargo build --release -p smg --features tokenizers-v1` enables both the v1
encoder and the full ByteLevel decode fast path. Neither is enabled by default.
The decoder reuses SMG's immutable byte table, which was already built for
streaming. It does not invoke the RC decoder, allocate a second vocabulary, or
change incremental decoding. Non-ByteLevel decoders retain tokenizers 0.23.

## Reproduce

Run from the repository root with the pinned Rust toolchain. The Python runner
uses only the standard library. It downloads tokenizer JSON files at the exact
revisions in `models.json` and verifies their SHA-256; no model weights are needed.
In an isolated checkout, extract `dependencies.lock.gz` from this directory
to the repository root as `Cargo.lock` to reproduce the recorded dependency graph.
Alternatively, resolve dependencies and generate your own local Cargo.lock; that
can select different versions and should be reported as a new experiment.
The build phase uses `--locked --offline` and therefore requires cached dependencies.

```sh
gzip -dc crates/tokenizer/benches/decode/dependencies.lock.gz > Cargo.lock
python crates/tokenizer/benches/decode/run.py prepare
python crates/tokenizer/benches/decode/run.py run
python crates/tokenizer/benches/decode/run.py summarize
```

Use `--models PATH` and `--output PATH` for existing model files and results.
`--duration-ms` defaults to 200 per configuration. `CARGO_TARGET_DIR` can point at
an existing cache. Where sccache rejects incremental compilation, set
`RUSTC_WRAPPER=` for these commands rather than changing global configuration.

The runner builds default and opt-in executables **before** correctness or timing.
It then checks every vocabulary ID alone and in context, both special-token
settings, deterministic multi-ID sequences with unknown IDs, and eight concurrent
readers. Any failure stops the run. Three process pairs run in alternating order
(default/opt-in, opt-in/default, default/opt-in). Run on an otherwise idle machine;
do not compile or run other benchmarks at the same time. A one-second Unix `ps`
monitor aborts and marks a timing file as contaminated if it sees a compiler;
other host activity and brief processes can still add noise.

## Measurement contract

- Four pinned models: MiniMax M3, GLM-5.3, Qwen3-8B and DeepSeek-V3.2.
- Full decode at 0, 1, 128, 2,048, 50,000 and 80,000 actual IDs. These are token-ID
  prefixes and can end in a partial character; no re-encoding or truncation is timed.
- Mixed English/Chinese/code/emoji text; special tokens inserted every 17 IDs;
  deterministic random vocabulary IDs with unknown IDs; ASCII; multilingual text;
  split UTF-8 byte tokens interleaved with invalid bytes. Both values of
  `skip_special_tokens` are measured for every input shape.
- Shared-tokenizer concurrency of 4 and 16 callers for 2k/80k mixed inputs;
  internal encoding parallelism is disabled. Decode has no L0/L1 text cache.
- Streaming is a negative control: 2k mixed/special/random/byte-fragment IDs, both skip settings,
  including Sequence construction, every append, and finish. Length-delimited
  output-piece digests must match across builds, including emission boundaries.
  This preserves existing stream behavior rather than assuming arbitrary malformed
  stream output equals full decode.
- Synthetic plain ByteLevel, Sequence(ByteLevel), ByteFallback+Fuse and no-decoder
  controls exercise malformed byte runs and nonselected paths.
- Each worker warms eight calls, synchronizes at a barrier, and measures batches
  of 32 calls to amortize clock overhead. An entire batch finishes before the
  duration check; long/concurrent cases can exceed the requested duration.
- Mean closed-loop time includes decoding, output allocation/destruction and loop
  overhead, but excludes loading, input construction and thread startup. It is
  accumulated worker time divided by calls. Throughput divides calls by the
  longest worker duration. Under contention it is not single-thread CPU time.
- Medians of three process observations are reported; raw rounds, sample counts
  and paired speedup min/max are retained. This does not estimate P95/P99 or prove
  that small timing differences are significant. Empty/single-token results remain
  especially sensitive to loop/timer overhead.
- Model loading is recorded separately. Opt-in loading also constructs the v1
  **encoder**, so default/opt-in load differences cannot be attributed to decode.
  The decode change adds no constructor work or persistent table storage.
- The workspace `bench` profile uses opt-level=3, thin LTO and one codegen unit;
  the allocator is the system allocator. Gateway release/production allocator
  settings differ. This is not an engine/HTTP/TTFT benchmark.

`metadata.json` records revisions, hashes, toolchain, host and binary identities.
`comparison.csv` has all 340 paired configurations, `rounds.csv` preserves every
sample, and `loads.csv` separates load observations. The summarizer fails on
missing/duplicate configurations, incorrect feature selection, zero samples, or
full/stream output differences. Local results are in [RESULTS.md](RESULTS.md).

## Verification

```sh
cargo +nightly fmt --all -- --check
cargo clippy -p llm-tokenizer --all-targets -- -D warnings
cargo clippy -p llm-tokenizer --all-targets --features tokenizers-v1 -- -D warnings
cargo test -p llm-tokenizer
cargo test -p llm-tokenizer --features tokenizers-v1
```

Repository tokenizer tests need the fixtures documented by their respective
helpers/CI configuration. Missing external fixtures do not establish model parity.
The runner's `check-default.jsonl` and `check-opt-in.jsonl` record the explicitly
verified four-model cases independently of those external fixtures.
