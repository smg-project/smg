# llm-tokenizer

## Overview
The `llm-tokenizer` crate exposes a single `Tokenizer` facade around multiple backends
(Hugging Face JSON tokenizers, OpenAI/tiktoken models, and an in-memory mock). It packages the
shared behaviours needed by LLM applications—encoding user text, incrementally decoding streamed tokens,
tracking per-request state, and detecting stop conditions—behind trait objects so consuming code
can remain backend-agnostic.

Key capabilities:
- trait-based split between `Encoder`, `Decoder`, and `Tokenizer` for shared APIs across backends
- Hugging Face tokenizer loading (with optional chat templates) and HF Hub downloads
- heuristic selection of OpenAI/tiktoken encodings for GPT model names
- incremental decoding utilities (`DecodeStream`, `Sequence`) that handle UTF-8 boundaries
- stop sequence handling via `StopSequenceDecoder` with token-level and string-level triggers
- optional Jinja2 chat-template rendering that matches Hugging Face semantics

The implementation deliberately keeps the surface area small—metrics, batching, or SentencePiece
support mentioned in earlier drafts do **not** exist today. This document reflects the actual code
as of `tokenizer/src/*`.

## Source Map
- `lib.rs` – module exports and the `Tokenizer` wrapper around `Arc<dyn Tokenizer>`
- `traits.rs` – shared traits and the `Encoding`/`SpecialTokens` helper types
- `factory.rs` – backend discovery, file/model heuristics, and tokio-aware creation helpers
- `hub.rs` – Hugging Face Hub downloads via `hf_hub`
- `huggingface.rs` – wrapper over `tokenizers::Tokenizer`, chat template loading, vocab access
- `tiktoken.rs` – wrapper over `tiktoken-rs` encoders for OpenAI model families and hub-loaded tiktoken models (includes `tokenizer_config.json` parsing for the tiktoken path)
- `tekken.rs` – the `tekken.json` reader (base64 ranks, the file's own split pattern, the special-token block below the ranks) that feeds the tiktoken backend
- `chat_template.rs` – AST-driven Jinja template inspection, rendering utilities, shared `ChatTemplateState`, and template file loading
- `registry.rs` – thread-safe tokenizer registry with deduplication for IGW mode, supporting lookup by UUID or name
- `sequence.rs` – stateful incremental decoding helper used by router sequences
- `stream.rs` – stateless streaming decoder that yields textual chunks from token streams
- `stop.rs` – stop-sequence detection with "jail" buffering and a builder API
- `mock.rs` – lightweight tokenizer used by unit tests
- `tests.rs` – smoke tests covering the trait facade and helpers (largely with the mock backend)
- `cache/` – multi-level caching infrastructure (L0 in-memory, L1 prefix-based)

## Core Traits and Types (`traits.rs`)
- `Encoder`, `Decoder`, and `Tokenizer` traits stay `Send + Sync` so instances can be shared across
  threads. Concrete backends implement the minimal methods: `encode`, `encode_batch`, `decode`,
  `vocab_size`, special-token lookup, and optional token↔id conversions.
- `Encoding` wraps backend-specific results: `Hf` holds the Hugging Face encoding object,
  `Plain` is a general-purpose `Vec<u32>` container, and `Tiktoken` stores u32 IDs
  from `tiktoken-rs`. `Encoding::token_ids()` is the zero-copy accessor used everywhere.
- `SpecialTokens` collects optional BOS/EOS/etc. markers so upstream code can make backend-agnostic
  decisions.
- `Tokenizer` (in `lib.rs`) is a thin `Arc<dyn Tokenizer>` newtype that exposes convenience methods
  (`encode`, `decode`, `decode_stream`, etc.) while keeping cloning cheap.

## Backend Implementations
### HuggingFaceTokenizer (`huggingface.rs`)
- Loads `tokenizer.json` (or similar) using `tokenizers::Tokenizer::from_file`.
- Caches vocab forward and reverse maps for `token_to_id`/`id_to_token` support.
- Extracts special tokens using common patterns (e.g. `<s>`, `[CLS]`).
- Adds special tokens on request the way the checkpoint's configuration prescribes: `encode(text, true)`
  carries the configured BOS/EOS when `tokenizer_config.json` sets `add_bos_token` / `add_eos_token`
  to `true`, through the post-processor the `transformers` Llama-style fast classes install at load
  (`update_post_processor`), even where `tokenizer.json` alone has none or a plain `ByteLevel` one;
  `encode(text, false)` gives the ids of the file alone. The serving engine's tokenizer, not the bare
  file, is the reference, so a cross-check against the `tokenizers` crate compares with
  `HuggingFaceTokenizer::tokenizers_pipeline()` (the same post-processor) or with the flag off.
- Supports optional chat templates: either auto-discovered next to the tokenizer via
  `tokenizer_config.json` or overridable with an explicit template path.
- Exposes `apply_chat_template` for JSON message payloads and template parameters.
- GLM-5.3 and GLM-5.3-Flash use native Rust chat rendering when checkpoint metadata and
  the template matches the current official one. Any other template keeps MiniJinja.

### TiktokenTokenizer (`tiktoken.rs`)
- Wraps the `tiktoken-rs` `CoreBPE` builders (`cl100k_base`, `p50k_base`, `p50k_edit`, `r50k_base`).
- `from_model_name` heuristically maps OpenAI model IDs (e.g. `gpt-4`, `text-davinci-003`) to those
  bases. Unknown model names return an error rather than silently defaulting.
- `from_dir` loads hub-hosted tiktoken models (e.g. Kimi K2, DeepSeek) from a directory containing
  `tiktoken.model` and `tokenizer_config.json`, with full vocab maps and chat template support.
- `from_tekken_file` loads a `tekken.json` vocabulary into the same engine, laid out as the file's
  reference reader lays it out: an id is a rank plus the size of the special-token block, the
  vocabulary is cut at the configured size, `encode(text, true)` prepends the file's BOS and
  nothing else, and `skip_special_tokens` drops every id of the block. The chat templates render
  under the structural form (the file's own reader writes its control tokens structurally and
  encodes every request string as plain text): the template's control tokens are tokens and the
  same spelling typed into a message stays text; the flat `encode` reads a spelling as its token,
  as the checkpoint's conversion does.
- Implements encode/decode operations; batch encode simply iterates sequentially.
- Built-in OpenAI models provide approximate vocab sizes and common GPT special tokens.
  Hub-loaded models build full `token_to_id`/`id_to_token` mappings from the BPE file.

### MockTokenizer (`mock.rs`)
- Purely for tests; hard-codes a tiny vocabulary and simple whitespace tokenization.
- Implements the same trait surface so helpers can be exercised without pulling real tokenizer data.

## Factory and Backend Discovery (`factory.rs`)
- `create_tokenizer{,_async}` accept either a filesystem path or a model identifier. Logic:
   1. Paths are loaded directly; a directory takes `tekken.json`, then `tokenizer.json`, then
      `vocab.json` + `merges.txt` under a Qwen2 class, then a tiktoken vocabulary; a single file's
      extension (or JSON autodetection) selects the backend, `tekken.json` by its name. Both
      backends read the processor's legacy `chat_template.json` beside the files for the requests
      without tools (`processor_chat_template`, one read): the tiktoken backend does so for every
      vocabulary format it loads (`tiktoken.model`, `*.tiktoken`, `tekken.json`), as the engine's
      processor and the conversion of the same checkpoint do.
   2. Strings that look like OpenAI model names (`gpt-*`, `davinci`, `curie`, `babbage`, `ada`) use
      `TiktokenTokenizer`.
   3. Everything else attempts a Hugging Face Hub download via `download_tokenizer_from_hf`.
- Chat templates can be injected with `create_tokenizer_with_chat_template`.
- Async creation uses `tokio` for network access. The blocking variant reuses or spins up a runtime
  when called from synchronous contexts.
- SentencePiece (`.model`) and GGUF files are detected but currently return a clear `not supported`
  error.

## Hugging Face Hub Integration (`hub.rs`)
- Uses the async `hf_hub` API to list and download tokenizer-related files
  (`tokenizer.json`, `merges.txt`, `.model`, etc.), filtering out weights and docs.
- The helper returns the HF cache directory containing the fetched files; the factory then loads
  from disk using standard file paths.
- Honour the `HF_TOKEN` environment variable for private or rate-limited models. Without it the
  download may fail with an authorization error.

## Chat Template Support (`chat_template.rs`)
- Detects whether a template expects raw string content or the structured OpenAI-style `content`
  list by walking the minijinja AST. This matches the Python-side detection logic of the serving
  engines.
- `ChatTemplateProcessor` (constructed per call) renders templates against JSON `messages` and
  `ChatTemplateParams` (system prompt, tools, EOS token handling, etc.). Errors surface as
  `anyhow::Error`, keeping parity with Hugging Face error messages.
- The tokenizer wrapper stores both the template string and its detected content format so callers
  can pre-transform message content correctly.

## Streaming and Stateful Helpers
### `DecodeStream` (`stream.rs`)
- Maintains a sliding window (`prefix_offset`, `read_offset`) over accumulated token IDs.
- Each `step` decodes the known prefix and the new slice; when the new slice produces additional
  UTF-8 text (and does not end in the replacement character `�`), it returns the incremental chunk
  and updates offsets. Otherwise it returns `None` and waits for more tokens.
- `step_batch` and `flush` offer convenience for batching and draining remaining text.

### `Sequence` (`sequence.rs`)
- Holds per-request decoding state: accumulated IDs plus offsets mirroring `DecodeStream`.
- `append_text` encodes extra prompt text; `append_token` decodes incremental output while
  respecting UTF-8 boundaries and replacing stray `�` characters.
- Designed for integration with router sequence management where decoded text must be replayed.

### `StopSequenceDecoder` (`stop.rs`)
- Extends the incremental decoding approach with a "jail" buffer that holds potential partial
  matches against configured stop sequences.
- Supports both token-level stops (visible or hidden) and arbitrary string sequences. When a string
  stop is configured, the decoder emits only the safe prefix and keeps a suffix jailed until it can
  decide whether it completes a stop sequence.
- Provides `StopSequenceDecoderBuilder` for ergonomic configuration and exposes `process_token`,
  `process_tokens`, `flush`, `reset`, and `is_stopped` helpers.

## Caching (`cache/`)
The caching subsystem provides multi-level caching for tokenizer results:
- `L0Cache`: In-memory exact-match cache with approximate LRU eviction for token ID lookups
- `L1Cache`: Prefix-based cache that can reuse partial encoding results
- `CachedTokenizer`: Wrapper that adds caching to any tokenizer implementation
- `TokenizerFingerprint`: Content-based fingerprinting for cache key generation

## Testing
- Unit tests cover the mock tokenizer, the `Tokenizer` wrapper, incremental decoding helpers, and
  stop-sequence behaviour (`tests.rs`, `sequence.rs`, `stop.rs`, `tiktoken.rs`, `factory.rs`,
  `hub.rs`). Network-dependent Hugging Face downloads are exercised behind a best-effort async test
  that skips in CI without credentials.
- Use `cargo test -p llm-tokenizer` to run the crate's test suite.

## Benchmarks

### The crate's own benches

`cargo bench -p llm-tokenizer --bench <name>` runs one of the crate's criterion groups: `stop_decoder` (a stream's stop
rules built from the configuration and from a copy, the per-token cost with and without stop strings),
`incremental_decode` (a full decode, one decode step, `Sequence::append_token_into`, the stop decoder's step),
`stop_sequence_search` and `byte_fallback_invalid_run`. The table benches that need model files read them from the
environment: `encode_volume` (`VOLUME_BENCH_TOKENIZERS`) and `parallel_encode` (`PARALLEL_BENCH_TOKENIZERS`) print one
line and exit without them; `encode_cost` (`ENCODE_BENCH_TOKENIZERS`, `ENCODE_BENCH_PROMPTS`), `prefix_cache`
(`PREFIX_BENCH_TOKENIZERS`) and `v1_compare` (`SMG_BENCH_MODEL_ROOT`) panic without them, and `decode_compare` takes a model root and a duration in milliseconds as positional arguments (`cargo bench -p llm-tokenizer --bench decode_compare -- <root> <ms>`; `cargo bench` appends `--bench` behind them, so the duration is not optional there) or runs through `benches/decode/run.py`, which builds it with `--no-run` and runs the binary itself. Each file's header documents its variables.
`benches/v1/README.md` and `benches/decode/README.md` describe the two runners that compare the default and the
`tokenizers-v1` build on pinned model revisions. None of these gates a pull request: the gate is the gateway's
`tokenizer_benchmark` (`model_gateway/benches/tokenizer_benchmark.rs`), run by `.github/workflows/benchmark-tokenizer.yml`
against the thresholds in `.github/bench-gates/tokenizer.toml`; every binary that job builds has
`RUSTFLAGS="-C llvm-args=-enable-machine-outliner=never -C llvm-args=-align-all-functions=6"` (every function on a 64-byte
boundary, the placement rule the gate file describes), so a pair built without them can read differently. On a pull request
the workflow builds the pull request's bench binary and the base branch's (the merge commit's first parent) with
`scripts/bench_gate.py build --gates .github/bench-gates/tokenizer.toml --out <binary> -- cargo bench -p smg --features tokenizers-v1 --bench tokenizer_benchmark --no-run`
(`--out` is the path the built executable is copied to; the base with `--sources-from <head>`, so the two differ in the
measured code alone), then runs them interleaved and compares them with
`scripts/bench_gate.py run --gates .github/bench-gates/tokenizer.toml --base <binary> --pr <binary> --out <dir> --cwd model_gateway`
(`--out` is the directory for `results.json` and `gate.md`); the job fails on a gated regression. A push to `main`, the
weekly schedule and a manual dispatch run the single `cargo bench -p smg --features tokenizers-v1 --bench tokenizer_benchmark`
without the gate.

### Against other implementations

The tables below come from a comparative harness kept outside this repository (test data and reference binaries live
outside the crates): one shared cell module compiled behind a small adapter per implementation, so every
implementation runs the same cells on the same texts. What a cell measures:

- **encode**: one process per implementation, family, text size and mode, pinned to one core; 3 warm-ups and 30
  timed encodes; the p50 of nanoseconds per token. *Cold* = every encode a different text (the class's 33 texts),
  *warm* = the same text every time. Three rounds, a round being one pass over the family list with the
  implementations interleaved in a rotating order inside each cell; a cell is the median of its rounds. A margin is
  the reference's time divided by SMG's, minus one: positive means SMG is faster.
- **decode**: a full decode of 2,048 ids of a 32k-token text (30 repetitions); the same ids stepped through one
  `Sequence` (incremental decode and the latency of the first delta); 64 interleaved streams of 256 steps (the step
  and the first delta as wall time).
- **batch**: the first 1,000 texts of the parity corpus through `encode_batch` on one thread.

Implementations: **SMG 1.15.0** (this crate's release build); **SMG v1.14.0** (the previous release's crate under the
same harness); **fastokens**: the open-source Rust tokenizer stack built on `fastokens` 0.3.5 (`riptoken` 0.3.0 for
tiktoken files, `tekken` 0.2.0 for `tekken.json`, `tokenizers` 0.22.2 where those decline a file), pinned at one
commit; **v1**: this crate built with `--features tokenizers-v1`, so the `tokenizers` 1.0.0-rc.3 pipeline encodes and
everything else is shared, which isolates the encoder. Where the v1 pipeline declines a file (an NFC normalizer, a
tiktoken file, `tekken.json`, WordPiece, Unigram) its build runs SMG's own path and the column reads `-`.

Corpus: 33 texts per size class of about 256, 2,000, 32,000, 128,000 and 500,000 tokens (counted with the DeepSeek
V4.1 vocabulary), assembled from public text: XL-Sum news articles in Arabic, Chinese, English, Hindi, Japanese,
Korean, Russian and Thai, and open-source code in six languages; 14 to 19 % of the bytes are outside ASCII.

Families: the tokenizer files of sixteen public checkpoints. Ten byte-level BPE vocabularies (DeepSeek V4.1 Flash,
DeepSeek V3.2, Llama 3.2, Llama 4 Scout, gpt-oss, Qwen3.6, GLM-5.3 Flash, MiniMax M3, MiMo V2.6, and Ministral 3 through
its `tekken.json`), three byte-fallback BPE vocabularies (Gemma 3, Gemma 4, Llama 2), a tiktoken file (Kimi K3), and
two of the `tokenizers` crate's other model types (BERT base, WordPiece; T5 small, Unigram), which SMG and the
reference both hand to that crate.

Setup: one x86-64 core per cell with its sibling idle, release builds (opt-level 3, fat LTO, one codegen unit), three
rounds, medians. Measured on the SMG 1.15.0 release candidate: the encode tables on commit 1e34e4481 (2026-11-09), the decode and streaming table on 173a21639 (2026-11-10; the byte-level decode's buffer sizing is the only change to this crate between the two); v1.14.0 is the crate at the `v1.14.0` tag. Numbers are nanoseconds per token unless the column says otherwise. Two builds of one source differ by up to about
20 % on a cell from code placement alone (the v1 build shares SMG's decode and streaming code, and those cells read -20 to
+9 % between the two builds), so a margin inside that band, the placement band below, is not a difference between implementations.

#### Encode, cold (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 114.7 (-52 / +5) | 105.8 (-50 / -2) | 98.3 (-46 / +36) | 85.4 (-42 / +47) | 68.6 (-46 / +79) |
| DeepSeek V3.2 | byte-level BPE | 115.8 (-54 / -0) | 102.1 (-42 / -2) | 87.9 (-35 / +39) | 78.4 (-35 / +54) | 66.6 (-41 / +90) |
| Llama 3.2 | byte-level BPE | 107.8 (-56 / -6) | 98.7 (-42 / +0) | 82.7 (-40 / +35) | 73.0 (-43 / +58) | 55.5 (-48 / +101) |
| Llama 4 | byte-level BPE | 120.4 (-55 / -5) | 127.6 (-34 / +7) | 114.5 (-35 / +36) | 91.2 (-25 / +73) | 79.1 (-39 / +86) |
| gpt-oss | byte-level BPE | 117.9 (-61 / +17) | 104.9 (-39 / +21) | 94.0 (-30 / +60) | 80.4 (-21 / +93) | 72.8 (-37 / +105) |
| Qwen 3.6 | byte-level BPE | 120.2 (-52 / -) | 115.1 (-41 / -) | 104.3 (-29 / -) | 93.6 (-24 / -) | 85.7 (-37 / -) |
| GLM-5.3 | byte-level BPE | 126.5 (-62 / -10) | 95.1 (-55 / -3) | 76.4 (-40 / +43) | 59.4 (-40 / +76) | 53.1 (-48 / +77) |
| MiniMax M3 | byte-level BPE | 109.2 (-58 / +5) | 97.5 (-24 / +10) | 93.7 (-36 / +97) | 77.8 (-29 / +113) | 69.0 (-29 / +149) |
| MiMo V2.6 | byte-level BPE | 102.2 (-50 / -0) | 97.0 (-37 / +7) | 76.3 (-35 / +114) | 72.0 (-31 / +126) | 64.2 (-38 / +146) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 114.2 (+626 / -) | 111.2 (+572 / -) | 92.2 (+716 / -) | 74.0 (+930 / -) | 64.8 (+1052 / -) |
| Gemma 3 | byte-fallback BPE | 213.2 (+81 / +177) | 139.2 (+156 / +242) | 125.0 (+151 / +262) | 109.8 (+146 / +296) | 104.4 (+126 / +321) |
| Gemma 4 | byte-fallback BPE | 186.5 (+92 / +189) | 135.8 (+146 / +285) | 109.9 (+174 / +448) | 105.5 (+136 / +592) | 83.6 (+151 / +871) |
| Llama 2 (byte fallback) | byte-fallback BPE | 136.7 (+132 / +151) | 54.7 (+122 / +305) | 29.9 (+229 / +778) | 25.4 (+232 / +1399) | 21.9 (+217 / +1757) |
| Kimi K3 | tiktoken file | 202.6 (+32 / -) | 159.9 (+38 / -) | 136.7 (+84 / -) | 129.8 (+99 / -) | 106.7 (+137 / -) |
| BERT base | WordPiece | 1219.4 (+5 / -) | 1055.6 (+1 / -) | 1085.3 (+2 / -) | 1158.6 (+6 / -) | 1400.3 (-3 / -) |
| T5 small | Unigram | 1867.7 (+8 / -) | 1628.3 (+7 / -) | 1829.4 (+11 / -) | 1938.7 (+5 / -) | 2017.2 (+5 / -) |

#### Encode, warm (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 14.9 (+28 / +36) | 13.4 (+39 / +18) | 24.6 (-20 / +493) | 28.5 (-24 / +335) | 33.5 (-35 / +255) |
| DeepSeek V3.2 | byte-level BPE | 14.7 (+38 / +37) | 13.0 (+97 / +21) | 23.7 (-2 / +557) | 28.2 (-9 / +373) | 33.7 (-27 / +266) |
| Llama 3.2 | byte-level BPE | 11.6 (+17 / +19) | 8.7 (+96 / +28) | 22.1 (-29 / +461) | 24.1 (-28 / +382) | 29.5 (-37 / +254) |
| Llama 4 | byte-level BPE | 17.8 (+18 / +71) | 9.9 (+103 / +74) | 22.3 (-14 / +718) | 28.1 (+18 / +463) | 32.9 (-31 / +375) |
| gpt-oss | byte-level BPE | 15.7 (+18 / +64) | 9.2 (+73 / +72) | 22.0 (-16 / +728) | 25.6 (+9 / +531) | 30.9 (+0 / +401) |
| Qwen 3.6 | byte-level BPE | 22.9 (+53 / -) | 11.3 (+113 / -) | 33.8 (-8 / -) | 38.5 (-24 / -) | 42.2 (-15 / -) |
| GLM-5.3 | byte-level BPE | 14.7 (+19 / +46) | 9.7 (+95 / +42) | 20.2 (-28 / +382) | 26.3 (-30 / +312) | 29.2 (-34 / +260) |
| MiniMax M3 | byte-level BPE | 17.6 (+48 / +178) | 12.6 (+127 / +141) | 28.2 (-8 / +524) | 38.7 (-27 / +374) | 38.5 (+17 / +372) |
| MiMo V2.6 | byte-level BPE | 16.1 (+46 / +122) | 10.5 (+109 / +76) | 29.8 (-22 / +444) | 38.8 (-36 / +362) | 38.7 (-21 / +343) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 13.0 (+5713 / -) | 8.8 (+6692 / -) | 19.5 (+3718 / -) | 27.6 (+2547 / -) | 30.4 (+2419 / -) |
| Gemma 3 | byte-fallback BPE | 12.0 (+396 / +30) | 8.9 (+427 / +71) | 22.7 (+381 / +1749) | 34.3 (+252 / +1173) | 36.0 (+228 / +1107) |
| Gemma 4 | byte-fallback BPE | 11.9 (+295 / -10) | 8.2 (+300 / -9) | 15.5 (+458 / -19) | 19.8 (+414 / +3535) | 25.8 (+263 / +2833) |
| Llama 2 (byte fallback) | byte-fallback BPE | 9.5 (+243 / -38) | 8.1 (+234 / -50) | 14.7 (+208 / -60) | 17.4 (+229 / +1796) | 18.8 (+179 / +1884) |
| Kimi K3 | tiktoken file | 37.0 (+478 / -) | 51.1 (+158 / -) | 61.3 (+357 / -) | 77.4 (+225 / -) | 79.2 (+223 / -) |
| BERT base | WordPiece | 694.9 (+3 / -) | 1006.0 (+2 / -) | 1016.2 (+11 / -) | 1148.5 (+5 / -) | 1432.8 (-2 / -) |
| T5 small | Unigram | 442.9 (+7 / -) | 1161.6 (+6 / -) | 1402.9 (+23 / -) | 1684.6 (+4 / -) | 1993.3 (+7 / -) |

#### v1.14.0 to 1.15.0 (change in ns per token; negative = 1.15.0 faster)

| family | cold: median change (range over the five sizes) | warm: median change (range) |
|---|---|---|
| DeepSeek V4.1 | -82.9 % (-86.9 % .. -79.2 %) | -90.8 % (-94.4 % .. -82.3 %) |
| DeepSeek V3.2 | -84.2 % (-87.6 % .. -80.8 %) | -91.0 % (-94.4 % .. -82.7 %) |
| Llama 3.2 | -94.6 % (-96.7 % .. -92.1 %) | -98.5 % (-99.0 % .. -98.3 %) |
| Llama 4 | -94.0 % (-95.7 % .. -91.0 %) | -98.4 % (-99.0 % .. -98.2 %) |
| gpt-oss | -82.1 % (-85.7 % .. -70.7 %) | -89.1 % (-94.6 % .. -82.2 %) |
| Qwen 3.6 | -86.5 % (-96.5 % .. -82.0 %) | -78.1 % (-98.3 % .. -76.8 %) |
| GLM-5.3 | -76.0 % (-84.6 % .. -67.4 %) | -79.5 % (-92.7 % .. -64.8 %) |
| MiniMax M3 | -87.6 % (-96.7 % .. -84.1 %) | -87.4 % (-98.1 % .. -82.7 %) |
| MiMo V2.6 | -87.7 % (-97.0 % .. -80.9 %) | -79.0 % (-98.2 % .. -75.9 %) |
| Mistral (tekken) | -77.9 % (-85.6 % .. -67.9 %) | -89.4 % (-93.7 % .. -81.2 %) |
| Gemma 3 | -90.3 % (-92.5 % .. -86.8 %) | -98.0 % (-99.0 % .. -97.2 %) |
| Gemma 4 | -93.7 % (-96.6 % .. -88.3 %) | -99.0 % (-99.2 % .. -98.4 %) |
| Llama 2 (byte fallback) | -96.7 % (-98.6 % .. -85.5 %) | -98.1 % (-98.5 % .. -97.7 %) |
| Kimi K3 | -59.6 % (-67.3 % .. -53.1 %) | -76.8 % (-85.5 % .. -76.1 %) |
| BERT base | +1.0 % (-3.9 % .. +2.7 %) | +0.2 % (-0.1 % .. +5.2 %) |
| T5 small | +0.6 % (-0.6 % .. +4.4 %) | -1.3 % (-2.8 % .. +1.8 %) |

#### Decode and streaming (read on commit 173a21639; ns per token or ns per step; in brackets the margin vs fastokens, in %; the v1 build shares SMG's decode and streaming code, so it has no margin there; the batch column, an `encode_batch`, carries the margin vs fastokens / vs v1; the v1 half reads `-` for Qwen 3.6, Mistral (tekken), Kimi K3, BERT base and T5 small, whose files the v1 pipeline declines: both builds ran the same encoder there, as in the encode tables)

| family | full decode, 2,048 ids | incremental decode, 2,048 ids | first delta of one stream | 64 streams: step | 64 streams: first delta | batch of 1,000 texts, one thread (Mtok/s) |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | 9.5 (+227) | 90.3 (+275) | 50.0 (+440) | 50.0 (+380) | 110.0 (+739) | 40.2 (-10 / +310) |
| DeepSeek V3.2 | 9.8 (+222) | 89.6 (+275) | 50.0 (+442) | 50.0 (+380) | 110.0 (+720) | 41.5 (+4 / +320) |
| Llama 3.2 | 8.7 (+243) | 90.3 (+254) | 50.0 (+440) | 41.0 (+485) | 110.0 (+291) | 35.0 (-29 / +186) |
| Llama 4 | 9.4 (+249) | 90.1 (+282) | 50.0 (+440) | 50.0 (+382) | 110.0 (+328) | 31.3 (-14 / +332) |
| gpt-oss | 9.5 (+222) | 90.9 (+266) | 50.0 (+440) | 50.0 (+380) | 110.0 (+492) | 29.4 (-8 / +279) |
| Qwen 3.6 | 9.9 (+228) | 88.8 (+278) | 50.0 (+422) | 40.0 (+475) | 130.0 (+262) | 20.4 (-15 / -) |
| GLM-5.3 | 6.9 (+263) | 88.0 (+224) | 50.0 (+442) | 50.0 (+362) | 120.0 (+250) | 45.0 (-26 / +242) |
| MiniMax M3 | 7.1 (+241) | 94.9 (+210) | 50.0 (+422) | 50.0 (+380) | 120.0 (+325) | 22.5 (-15 / +216) |
| MiMo V2.6 | 8.7 (+223) | 90.5 (+238) | 50.0 (+440) | 41.0 (+463) | 121.0 (+240) | 27.8 (-27 / +173) |
| Mistral (tekken) | 8.5 (-) | 95.6 (-) | 70.0 (-) | 40.0 (+775) | 120.0 (+1052) | 32.8 (+2408 / -) |
| Gemma 3 | 16.7 (+518) | 106.0 (+489) | 261.0 (+211) | 60.0 (+835) | 430.0 (+68) | 28.0 (+297 / +1001) |
| Gemma 4 | 16.2 (+565) | 105.8 (+487) | 251.0 (+208) | 60.0 (+835) | 421.0 (+64) | 25.7 (+226 / +985) |
| Llama 2 (byte fallback) | 14.2 (+648) | 122.7 (+511) | 251.0 (+283) | 60.0 (+1035) | 401.0 (+130) | 64.9 (+139 / +376) |
| Kimi K3 | 7.5 (+19) | 94.6 (+180) | 50.0 (+340) | 40.0 (+425) | 130.0 (+224) | 14.9 (+211 / -) |
| BERT base | 686.9 (-19) | 2168.7 (-18) | 1522.0 (-22) | 2053.0 (-20) | 1682.0 (-14) | 1.3 (+1 / -) |
| T5 small | 110.0 (+13) | 428.9 (+4) | 380.0 (-8) | 350.0 (+12) | 531.0 (+13) | 0.7 (+9 / -) |

Ministral (`tekken.json`): the reference's single-sequence decode cell aborts in every round (its decode of a window that
ends inside a character panics), so the full decode, incremental decode and first-delta cells have no reference value; its
64-stream cell ran to the end on this text with every stream's text equal to the full decode, so those two margins are
like-for-like.

#### Reading the tables

- Against v1.14.0 every encode cell of the fourteen vocabularies whose encoder changed is faster (all 140 cells; BERT and T5, 20 cells, run the `tokenizers` crate in both builds and read within their round spreads): 86 % less time per token cold and 92 % warm on the byte-level vocabularies (medians over 50 cells each), 92 % and 98 % on the byte-fallback vocabularies, which 1.14.0 still ran through the generic pipeline, as it did the Llama 3.2 and Llama 4 files.
- Cold text is where the fastokens stack is still ahead on the nine non-tekken byte-level vocabularies (-62 % to -21 %, every cell beyond the placement band; the first-sight merge of unseen pieces). On byte-fallback vocabularies, the tiktoken file and `tekken.json` SMG is ahead on every cold cell.
- Warm text at 256 and 2k tokens: SMG ahead of fastokens beyond the placement band on 14 of 18 cells of the nine non-tekken byte-level vocabularies (+28 % to +127 %), the 256-token cells of Llama 3.2, gpt-oss, GLM-5.3 and Llama 4 inside it (+17 % to +19 %; on `tekken.json` the tekken crate is far behind on every cell); at 32k tokens and above fastokens is ahead beyond the band on 15 of 27 cells (-37 % to -21 %) and the other 12 read -20 % to +18 %, inside it.
- The v1 pipeline is behind SMG by more than the placement band on all 24 cold byte-level cells at 32k tokens and above (+35 % to +149 %) and on 38 of 40 warm byte-level cells (DeepSeek V4.1 at 2k and Llama 3.2 at 256 read +18 % to +19 %, inside it); of the 16 cold cells at 256 and 2k tokens, 15 read -10 % to +17 %, inside the band, and gpt-oss at 2k reads +21 %, just beyond it. Its whole-text word cache wins the Llama 2 warm cells at 256, 2k and 32k tokens (-60 % to -38 %), a lookup rather than an encode (this crate's exact-match cache, where a deployment enables it, answers a repeated text the same way); Gemma 4's at 256, 2k and 32k tokens (-19 % to -9 %) sit inside the band.
- On the nine non-tekken byte-level vocabularies the decode and streaming margins are wider than encode's: a full decode runs at 6.9 to 9.9 ns per token against the fastokens stack's 24.1 to 32.8, and the first delta of 64 interleaved streams arrives in 110 to 130 ns against 411 to 923; the byte-fallback vocabularies read 14.2 to 16.7 against 103.0 to 107.8 on the full decode; BERT, on the `tokenizers` crate in both, reads 687 against 554 (-19 %), inside the band.

## Known Limitations & Future Work
- SentencePiece (`.model`) and GGUF tokenizers are detected but deliberately unimplemented.
- `Encoding::Plain` is a general-purpose `Vec<u32>` container used by mock tokenizers and cache merge logic.
- Built-in `TiktokenTokenizer` models (via `from_model_name`) keep no text-to-id map, so
  `token_to_id` returns `None`; `id_to_token` reads the encoder and answers. Hub-loaded models (via `from_dir`) answer both.
- There is no metrics or batching layer inside this module; the router records metrics elsewhere.
- Dynamic batching / sequence pooling code that earlier READMEs mentioned never landed in Rust.

## Usage Examples
```rust
use std::sync::Arc;
use llm_tokenizer::{
    create_tokenizer_from_file, create_tokenizer, SequenceDecoderOutput, Tokenizer,
    stop::StopSequenceDecoderBuilder,
};

// Load a tokenizer from disk (Hugging Face JSON)
// create_tokenizer_from_file returns Arc<dyn Tokenizer>
let inner = create_tokenizer_from_file("/path/to/tokenizer.json")?;
let tokenizer = Tokenizer::from_arc(Arc::clone(&inner));
let encoding = tokenizer.encode("Hello, world!", false)?;
assert!(!encoding.token_ids().is_empty());

// Auto-detect OpenAI GPT tokenizer (returns Arc<dyn Tokenizer>)
let openai = Tokenizer::from_arc(create_tokenizer("gpt-4")?);
let text = openai.decode(&[1, 2, 3], true)?;

// Incremental decoding with stop sequences
let mut stream = tokenizer.decode_stream(&[], true);
let mut stop = StopSequenceDecoderBuilder::new(Arc::clone(&inner))
    .stop_sequence("\nHuman:")
    .build();
for &token in encoding.token_ids() {
    if let Some(chunk) = stream.step(token)? {
        match stop.process_token(token)? {
            SequenceDecoderOutput::Text(t) => println!("{}", t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                println!("{}", t);
                break;
            }
            SequenceDecoderOutput::Held | SequenceDecoderOutput::Stopped => {}
        }
    }
}
```

```rust
// Apply a chat template when one is bundled with the tokenizer
use llm_tokenizer::{chat_template::ChatTemplateParams, HuggingFaceTokenizer};

let hf = HuggingFaceTokenizer::from_file_with_chat_template(
    "./tokenizer.json",
    Some("./chat_template.jinja"),
)?;
let messages = vec![
    serde_json::json!({"role": "system", "content": "You are concise."}),
    serde_json::json!({"role": "user", "content": "Summarise Rust traits."}),
];
let prompt = hf.apply_chat_template(
    &messages,
    ChatTemplateParams {
        add_generation_prompt: true,
        tools: None,
        documents: None,
        template_kwargs: None,
    },
)?;
```

Set `HF_TOKEN` in the environment if you need to download private models from the Hugging Face Hub.

## Prometheus cache activity

The gateway exports process-lifetime totals across tokenizer instances on `/metrics`:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `smg_tokenizer_cache_lookups_total` | `layer=l0` or `l1`, `result=hit` or `miss` | One outcome per lookup, including L1 misses without special-token boundaries |
| `smg_tokenizer_cache_evictions_total` | `layer=l0` or `l1` | Entries actually removed for capacity, excluding clear, drop and replacement |
| `smg_tokenizer_cache_reused_bytes_total` | `layer=l0` or `l1` | UTF-8 input bytes served by hits: whole input for L0, matched prefix for L1 |
| `smg_tokenizer_encode_path_loads_total` | `path=v1`, `native` or `tokenizers` | Hugging Face tokenizers loaded, by the encode path chosen at load (see `encode_path.rs`); a tokenizer loaded twice counts twice |
| `smg_tokenizer_encode_fallbacks_total` | `reason=piece_declined` | Inputs the direct byte-level encoder gave up to `tokenizers` after the load's choice |

Labels are fixed; no model IDs or prompt text are retained. L0 hits bypass L1,
so each layer's hit ratio uses its own lookup count. Disabled layers have no
activity. Totals survive cache clear/drop and are available even if the metrics
recorder starts after tokenization. Scraping reads atomics without scanning caches.
Reused bytes measure reuse volume, not resident memory or CPU time saved; in
particular, a prefix hit can still do suffix tokenization and cache population.

Per-layer hit ratio (undefined until that layer receives lookups):

```promql
sum by (layer) (rate(smg_tokenizer_cache_lookups_total{result="hit"}[5m]))
/
sum by (layer) (rate(smg_tokenizer_cache_lookups_total[5m]))
```

Compare this with `rate(smg_tokenizer_cache_evictions_total[5m])` when tuning
capacity, and `rate(smg_tokenizer_cache_reused_bytes_total[5m])` to distinguish
small prefix hits from reuse of large prompts.

`smg_tokenizer_encode_path_loads_total` says which encoder each loaded
tokenizer's encodes go through (the load also logs the path once, with the
reasons the faster paths declined); a non-zero `tokenizers` series under the
`tokenizers-v1` feature names the tokenizers the opt-in pipeline does not take.
`smg_tokenizer_encode_fallbacks_total` counts the inputs that still left the
direct encoder afterwards, each encoded a second time by `tokenizers`.
