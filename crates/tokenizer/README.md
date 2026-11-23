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
- Reads `tokenizer.json` (or similar) into memory and deserializes it there (the two steps of
  `tokenizers::Tokenizer::from_file`), keeping the text so a Unigram model section can be read
  a second time for the scores the reference reads.
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
- Tells the control tokens a template writes from the same spellings typed into a request, for two
  kinds of checkpoint: one that ships its own vocabulary file beside the conversion, and one
  whose template builds its control spellings (it writes no control
  token whole by itself, in one piece or in pieces side by side: it binds each to a name, out of
  a constant of its own or spelled whole in that one statement, and writes the names; a control
  may also be kept in an attribute of a namespace that the top level makes once, and written by
  that attribute). On
  `apply_chat_template_with_encoding` the template's tokens are tokens and a spelling typed into
  a message, a tool, a document or a template argument stays text, also when it is typed in two
  parts the template joins, and so does a text that only the pipeline's normalizer makes an
  added token's spelling. Under the second kind a typed added token keeps its id when the
  vocabulary does not flag it special, the template writes it nowhere as a control and the flat
  `encode` reads it alone and as it stands. The second kind is recognised at load from the
  template itself, and
  only when its syntax shows that every value it builds is only written (alone, in a join, or
  asked whether it is empty), each of those values is empty, one control spelling, or a string
  that begins with a control spelling and ends with one, each fits in one plain string literal
  (it holds no newline or other control character, no backslash and not both kinds of quote),
  the pipeline matches every added token as it is spelled (under one that matches an added
  token after normalization the template is left as it is), and the marks cover every control
  it writes on a few probe conversations; any other checkpoint renders
  and encodes as before. The flat `encode` of a rendered text reads a spelling as its token.
- GLM-5.3 and GLM-5.3-Flash use native Rust chat rendering when checkpoint metadata and
  the template matches the current official one. Any other template keeps MiniJinja.

### Native Inkling2 requests (`encoders/inkling2.rs`)

The tokenizer selects the native renderer only for `Inkling2ForConditionalGeneration` or model type
`inkling2`. It uses the checkpoint's `tokenizer.json`, not a built-in
OpenAI vocabulary or a Jinja template. Loading validates the structural token IDs.

Use `apply_chat_template_with_encoding` and run its deferred `EncodeJob`. The
returned text is diagnostic: a flat re-encode cannot distinguish literal caller
text from structural tokens. The renderer encodes each text payload as ordinary text.
It inserts typed message headers, content headers and turn endings as IDs.
There is no BOS or assistant generation opener. The final assistant can continue
only as string text without reasoning or tool calls.

JSON response formats require separated reasoning and cannot continue nonempty assistant text.
The gateway checks these renderer capabilities against the actual response format before rendering.
An empty assistant prefix is permitted. This validation does not establish constrained-generation correctness.
Other renderers retain their existing JSON continuation and reasoning behavior.

Request rendering follows the model's reference request format, not the standalone renderer's
tool-author convention: a tool call carries its name in JSON, without adding a
tool-name author to the assistant header. Tool results retain an explicit name
if supplied. A call ID is metadata, not prompt text. Tool schemas use sorted,
compact Unicode JSON with round-trip float parsing. Effort defaults to 0.9, follows leading system messages,
and accepts the native names or numbers in `[0, 0.99]`.

An image part emits header ID 200005 followed by soft anchor ID 200054
(`<|unused_200054|>`). The common multimodal
hydration layer must expand only the soft anchor and retain the header. A literal
anchor spelling in caller text is not an image. Image bytes, geometry, feature
generation and cache identity belong to that layer. Audio, video, document
rendering and non-function tools are not supported here.

Native image processing requires the checkpoint's fixed image policy: its
`image_preprocessing.json`, or the `Inkling2ImageProcessor` section of the
released `processor_config.json`. Both forms give the same features. It retains
original dimensions and rejects request-level `max_long_side_pixel` overrides
before fetching media. Grid cells contain duplicated temporal RGB patches with
fixed normalization. The tokenizer bundle includes both policy files.

`cargo test -p llm-tokenizer --test inkling2_renderer` runs self-contained tests.
The ignored oracle test additionally requires `INKLING2_MODEL_DIR` and
`INKLING2_RENDER_FIXTURES` to name a local checkpoint and a file of reference
renderings. It validates exact text and IDs without downloading model files.
Direct typed thinking, tool-error and tool-call parts also require support in the
gateway's request schema.

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
- Takes a `StopSequenceConfig`, whose `with_*` setters add the stops, and exposes `process_token`,
  `process_tokens`, `flush`, `finish`, `reset`, and `is_stopped` helpers.

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
`stop_sequence_search`, `byte_fallback_invalid_run` and `encode_blocks` (one row: a 10 KB text of long words through
the native encoder, where its block pass is most of the work). The table benches that need model files read them from the
environment: `encode_volume` (`VOLUME_BENCH_TOKENIZERS`) and `parallel_encode` (`PARALLEL_BENCH_TOKENIZERS`) print one
line and exit without them; `encode_cost` (`ENCODE_BENCH_TOKENIZERS`, `ENCODE_BENCH_PROMPTS`), `prefix_cache`
(`PREFIX_BENCH_TOKENIZERS`) and `v1_compare` (`SMG_BENCH_MODEL_ROOT`) panic without them, and `decode_compare` takes a model root and a duration in milliseconds as positional arguments (`cargo bench -p llm-tokenizer --bench decode_compare -- <root> <ms>`; `cargo bench` appends `--bench` behind them, so the duration is not optional there) or runs through `benches/decode/run.py`, which builds it with `--no-run` and runs the binary itself. Each file's header documents its variables.
`benches/v1/README.md` and `benches/decode/README.md` describe the two runners that compare the default and the
`tokenizers-v1` build on pinned model revisions. One of these gates a pull request: `encode_blocks`, against
`.github/bench-gates/tokenizer-blocks.toml`, which says what the row pins. The other gate is the gateway's
`tokenizer_benchmark` (`model_gateway/benches/tokenizer_benchmark.rs`), against the thresholds in
`.github/bench-gates/tokenizer.toml`. `.github/workflows/benchmark-tokenizer.yml` runs both, a row of its matrix each; every binary that job builds has
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

The tables below come from the comparative harness in [`benches/compare`](benches/compare/README.md), which says how
to run it (the corpus is test data and lives outside the crates): one shared cell module compiled behind a small
adapter per implementation, so every implementation runs the same cells on the same texts. What a cell measures:

- **encode**: one process per implementation, family, text size and mode, pinned to one core; 3 warm-ups and 30
  timed encodes; the p50 of nanoseconds per token. *Cold* = every encode a different text (the class's 33 texts),
  *warm* = the same text every time. Three rounds, a round being one pass over the family list with the
  implementations interleaved in a rotating order inside each cell, and five more rounds on the cells where SMG stands
  within 10 % of its best reference; a cell is the median of its rounds. A margin is
  the reference's time divided by SMG's, minus one: positive means SMG is faster.
- **decode**: a full decode of 2,048 ids of a 32k-token text (30 repetitions); the same ids stepped through one
  `Sequence` (incremental decode and the latency of the first delta); 64 interleaved streams of 256 steps (the step
  and the first delta as wall time).
- **batch**: the first 1,000 texts of the parity corpus through `encode_batch` on one thread.
- **load and memory**: one process per implementation and family: the load time and the resident memory after it, the
  high-water mark after one encode of a 1M-token text, and the resident memory per thread with 64 threads alive: its
  growth on one shared tokenizer, and its level with a tokenizer per thread, both counted from before the first thread
  starts. These cells were read again in a second window with the corrected memory cell, which reads the resident
  memory while every thread still holds its tokenizer.

Implementations: **SMG 1.16.0** (this crate's release build); **SMG v1.15.0** (the previous release's crate under the
same harness); **fastokens**: the open-source Rust tokenizer stack built on `fastokens` 0.3.5 (`riptoken` 0.3.0 for
tiktoken files, `tekken` 0.2.0 for `tekken.json`, `tokenizers` 0.22.2 where those decline a file), pinned at one
commit; **v1**: this crate built with `--features tokenizers-v1`, so the `tokenizers` 1.0.0-rc.3 pipeline encodes and
everything else is shared, which isolates the encoder. Where the v1 pipeline declines a file (a
tiktoken file, `tekken.json`, WordPiece, Unigram) its build takes its next encode path (SMG's own; the `tokenizers`
crate's for Unigram) and the column reads `-`. The csv files hold two more reference columns that the tables leave
out: the `tokenizers` crate 0.23.2 through its default pipeline, and `tiktoken-rs` 0.12.1 on gpt-oss and Kimi K3.

Corpus: 33 texts per size class of about 256, 2,000, 32,000, 128,000, 500,000 and 1,000,000 tokens (counted with the
DeepSeek V4.1 vocabulary; a 1M text is two 500k texts joined), assembled from public text: XL-Sum news articles in Arabic, Chinese, English, Hindi, Japanese,
Korean, Russian and Thai, and open-source code and configuration (Rust, Python, Go, shell, C, Jinja templates, TOML and YAML,
protobuf, JSON) with generated tool-call payloads; 17 to 39 % of the bytes of a size class are outside ASCII.
A warm cell encodes one text, the first of its size class, again and again: a warm figure is the speed of that one
text with hot caches, not the speed of mixed traffic. Those six texts are 1,548 bytes (256), 8,873 (2k), 151,173 (32k),
599,401 (128k), 2,289,914 (500k) and 4,613,798 (1M), with 90.1, 41.2, 50.1, 42.3, 39.4 and 40.9 % of their bytes
outside ASCII, where their classes hold 17.1, 31.1, 37.6, 38.3, 38.7 and 38.7 %. Such a text can read slower warm
than its class reads cold, as some cells from the 32k class up do in `benches/compare/encode-rounds.csv`.

Families: the tokenizer files of sixteen public checkpoints. Ten byte-level BPE vocabularies (DeepSeek V4.1 Flash,
DeepSeek V3.2, Llama 3.2, Llama 4 Scout, gpt-oss, Qwen3.6, GLM-5.3 Flash, MiniMax M3, MiMo V2.6, and Ministral 3 through
its `tekken.json`), three byte-fallback BPE vocabularies (Gemma 3, Gemma 4, Llama 2), a tiktoken file (Kimi K3), and
two of the `tokenizers` crate's other model types (BERT base, WordPiece; T5 small, Unigram): SMG encodes
BERT base itself since 1.16.0 and hands T5 small to that crate, and the reference hands it both.

Setup: one x86-64 core per cell with its sibling idle, release builds (opt-level 3, fat LTO, one codegen unit; the
fastokens stack under its own workspace's profile, thin LTO), three rounds and eight on the edge cells, medians. Measured
on the SMG 1.16.0 release candidate, commit 378df2a52 (2026-11-18); v1.15.0 is the crate at the `v1.15.0` tag, built by
the same toolchain. Other work ran on the machine's other cores during part of the run, so absolute figures may read
high; the implementations of a cell run within the same minute, so the margins hold. Numbers are nanoseconds per token
unless the column says otherwise. Two builds of one source differ by about 20 % on a cell from code placement alone (the
reading list below gives this run's range), so a margin inside that band, the placement band below, is not a
difference between implementations.

#### Encode, cold (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k | 1m |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 81.5 (-40 / +43) | 81.7 (-39 / +9) | 61.8 (-22 / +65) | 61.9 (-28 / +80) | 52.7 (-34 / +115) | 39.3 (-33 / +179) |
| DeepSeek V3.2 | byte-level BPE | 77.7 (-24 / +56) | 83.6 (-29 / +14) | 67.5 (-20 / +83) | 62.1 (-21 / +84) | 51.6 (-28 / +116) | 36.7 (-14 / +216) |
| Llama 3.2 | byte-level BPE | 73.8 (-40 / +36) | 79.9 (-36 / +17) | 63.7 (-24 / +58) | 52.2 (-26 / +89) | 43.5 (-36 / +137) | 33.8 (-29 / +202) |
| Llama 4 | byte-level BPE | 85.3 (-36 / +35) | 102.7 (-28 / +17) | 89.3 (-27 / +58) | 73.9 (-10 / +96) | 69.0 (-34 / +116) | 48.1 (-17 / +201) |
| gpt-oss | byte-level BPE | 82.2 (-38 / +43) | 87.8 (-26 / +35) | 86.7 (-26 / +65) | 79.0 (-20 / +92) | 62.4 (-29 / +147) | 46.5 (-22 / +224) |
| Qwen 3.6 | byte-level BPE | 73.6 (-18 / +261) | 75.8 (-3 / +238) | 81.3 (-8 / +299) | 70.5 (+0 / +331) | 64.0 (-13 / +386) | 47.4 (-4 / +565) |
| GLM-5.3 | byte-level BPE | 78.1 (-39 / +37) | 74.9 (-43 / +18) | 57.6 (-20 / +84) | 44.3 (-22 / +120) | 41.8 (-35 / +135) | 32.8 (-27 / +198) |
| MiniMax M3 | byte-level BPE | 83.8 (-39 / +33) | 93.8 (-24 / +16) | 79.9 (-23 / +123) | 65.2 (-14 / +160) | 53.1 (-10 / +215) | 42.2 (-1 / +291) |
| MiMo V2.6 | byte-level BPE | 69.0 (-36 / +54) | 76.1 (-23 / +29) | 62.6 (-19 / +165) | 50.8 (-7 / +192) | 46.2 (-12 / +229) | 38.9 (+10 / +298) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 79.4 (+867 / -) | 99.2 (+645 / -) | 74.1 (+920 / -) | 61.5 (+1138 / -) | 52.2 (+1342 / -) | 42.0 (+1647 / -) |
| Gemma 3 | byte-fallback BPE | 166.5 (+160 / +232) | 126.7 (+161 / +287) | 133.2 (+148 / +231) | 99.0 (+177 / +326) | 87.4 (+178 / +410) | 64.8 (+185 / +582) |
| Gemma 4 | byte-fallback BPE | 168.5 (+104 / +231) | 130.9 (+138 / +301) | 104.7 (+171 / +458) | 95.6 (+164 / +674) | 82.8 (+166 / +855) | 60.3 (+175 / +1250) |
| Llama 2 (byte fallback) | byte-fallback BPE | 115.9 (+136 / +168) | 46.0 (+150 / +368) | 28.0 (+232 / +914) | 20.6 (+276 / +1505) | 19.1 (+247 / +1935) | 17.7 (+299 / +2120) |
| Kimi K3 | tiktoken file | 142.9 (+87 / -) | 132.9 (+66 / -) | 109.2 (+131 / -) | 101.3 (+144 / -) | 85.2 (+199 / -) | 81.3 (+206 / -) |
| BERT base | WordPiece | 106.8 (+1080 / -) | 88.3 (+1085 / -) | 96.4 (+1076 / -) | 103.1 (+1078 / -) | 104.6 (+1301 / -) | 101.2 (+1368 / -) |
| T5 small | Unigram | 1563.8 (+9 / -) | 1528.2 (+15 / -) | 1668.3 (+25 / -) | 1559.9 (+23 / -) | 1610.1 (+26 / -) | 1721.2 (+22 / -) |

#### Encode, warm (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k | 1m |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 11.6 (+70 / +79) | 9.4 (+106 / +68) | 12.7 (+56 / +955) | 16.2 (+27 / +735) | 20.0 (+5 / +488) | 21.5 (+1 / +441) |
| DeepSeek V3.2 | byte-level BPE | 11.7 (+79 / +77) | 9.4 (+188 / +69) | 12.7 (+87 / +803) | 16.1 (+53 / +670) | 19.8 (+22 / +455) | 22.5 (+14 / +449) |
| Llama 3.2 | byte-level BPE | 9.2 (+48 / +55) | 7.1 (+146 / +59) | 14.2 (+9 / +630) | 15.2 (+10 / +650) | 19.2 (-2 / +430) | 22.5 (-14 / +366) |
| Llama 4 | byte-level BPE | 14.3 (+47 / +117) | 9.1 (+130 / +91) | 16.4 (+21 / +1054) | 19.6 (+52 / +700) | 23.0 (-4 / +530) | 25.3 (+9 / +489) |
| gpt-oss | byte-level BPE | 12.4 (+48 / +112) | 8.2 (+93 / +90) | 15.6 (+13 / +970) | 18.1 (+53 / +779) | 23.2 (+35 / +516) | 26.9 (+21 / +483) |
| Qwen 3.6 | byte-level BPE | 18.1 (+85 / +691) | 8.5 (+194 / +1164) | 17.0 (+84 / +1881) | 18.9 (+64 / +1593) | 25.8 (+45 / +1160) | 30.2 (+26 / +958) |
| GLM-5.3 | byte-level BPE | 11.6 (+51 / +89) | 7.7 (+151 / +82) | 12.8 (+11 / +653) | 15.6 (+15 / +613) | 19.1 (-5 / +404) | 22.5 (-13 / +348) |
| MiniMax M3 | byte-level BPE | 14.6 (+69 / +229) | 10.8 (+176 / +178) | 17.6 (+50 / +880) | 22.0 (+30 / +730) | 31.7 (+41 / +455) | 28.6 (+44 / +528) |
| MiMo V2.6 | byte-level BPE | 13.2 (+68 / +161) | 7.9 (+196 / +130) | 17.0 (+40 / +861) | 18.1 (+31 / +833) | 24.8 (+29 / +553) | 26.8 (+20 / +524) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 10.7 (+6744 / -) | 7.8 (+6873 / -) | 14.3 (+5284 / -) | 16.8 (+4318 / -) | 20.3 (+3587 / -) | 22.4 (+3179 / -) |
| Gemma 3 | byte-fallback BPE | 10.0 (+499 / +57) | 7.7 (+515 / +88) | 15.4 (+762 / +2660) | 20.4 (+508 / +2138) | 23.9 (+356 / +1697) | 26.5 (+389 / +1538) |
| Gemma 4 | byte-fallback BPE | 9.6 (+396 / +10) | 7.0 (+368 / -2) | 13.4 (+358 / -7) | 16.4 (+520 / +4465) | 22.3 (+321 / +3397) | 22.4 (+400 / +3186) |
| Llama 2 (byte fallback) | byte-fallback BPE | 7.7 (+309 / -40) | 7.3 (+276 / -55) | 13.3 (+183 / -63) | 14.1 (+297 / +1854) | 16.1 (+217 / +2141) | 15.6 (+290 / +2212) |
| Kimi K3 | tiktoken file | 24.1 (+792 / -) | 35.3 (+260 / -) | 42.9 (+546 / -) | 51.9 (+413 / -) | 56.7 (+355 / -) | 60.5 (+339 / -) |
| BERT base | WordPiece | 103.6 (+602 / -) | 69.6 (+1266 / -) | 99.0 (+1120 / -) | 99.5 (+1099 / -) | 101.9 (+1184 / -) | 102.3 (+1288 / -) |
| T5 small | Unigram | 375.1 (+26 / -) | 1012.2 (+22 / -) | 1281.5 (+35 / -) | 1392.2 (+21 / -) | 1592.8 (+30 / -) | 1641.2 (+24 / -) |

#### v1.15.0 to 1.16.0 (change in ns per token; negative = 1.16.0 faster)

| family | cold: median change (range over the six sizes) | warm: median change (range) |
|---|---|---|
| DeepSeek V4.1 | -18.2 % (-26.1 % .. -12.9 %) | -35.2 % (-45.5 % .. -20.3 %) |
| DeepSeek V3.2 | -21.7 % (-29.9 % .. -10.6 %) | -34.8 % (-45.4 % .. -16.8 %) |
| Llama 3.2 | -20.1 % (-24.4 % .. -12.8 %) | -26.3 % (-33.6 % .. -14.2 %) |
| Llama 4 | -16.2 % (-20.8 % .. -9.6 %) | -19.4 % (-25.6 % .. -5.7 %) |
| gpt-oss | -11.8 % (-20.3 % .. -5.4 %) | -21.9 % (-27.4 % .. -8.2 %) |
| Qwen 3.6 | -22.9 % (-29.5 % .. -14.6 %) | -33.2 % (-50.7 % .. -12.8 %) |
| GLM-5.3 | -17.8 % (-28.7 % .. -17.1 %) | -24.3 % (-32.4 % .. -14.2 %) |
| MiniMax M3 | -17.8 % (-21.0 % .. +2.0 %) | -22.8 % (-37.8 % .. -12.0 %) |
| MiMo V2.6 | -23.5 % (-26.6 % .. -13.9 %) | -29.9 % (-44.3 % .. -13.8 %) |
| Mistral (tekken) | -11.9 % (-22.8 % .. -0.2 %) | -23.1 % (-25.0 % .. -6.5 %) |
| Gemma 3 | -14.6 % (-19.9 % .. -8.2 %) | -31.3 % (-37.9 % .. -12.1 %) |
| Gemma 4 | -9.6 % (-13.0 % .. -4.4 %) | -13.8 % (-21.1 % .. -10.5 %) |
| Llama 2 (byte fallback) | -10.6 % (-12.2 % .. -5.7 %) | -9.9 % (-16.9 % .. -3.3 %) |
| Kimi K3 | -21.1 % (-24.2 % .. -17.6 %) | -29.8 % (-37.6 % .. -24.3 %) |
| BERT base | -91.4 % (-92.9 % .. -91.0 %) | -91.9 % (-92.8 % .. -85.2 %) |
| T5 small | -11.2 % (-15.4 % .. +1.1 %) | -12.8 % (-14.6 % .. -8.8 %) |

#### Decode and streaming (ns per token or ns per step; in brackets the margin vs fastokens, in %; the batch column, in Mtok/s, carries the margin vs fastokens / vs v1)

| family | full decode, 2,048 ids | incremental decode, 2,048 ids | first delta of one stream | 64 streams: step | 64 streams: first delta | batch of 1,000 texts, one thread (Mtok/s) |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | 8.6 (+268) | 90.9 (+270) | 50.0 (+460) | 41.0 (+485) | 110.0 (+619) | 48.1 (+4 / +372) |
| DeepSeek V3.2 | 8.4 (+281) | 89.5 (+283) | 50.0 (+460) | 50.0 (+382) | 111.0 (+821) | 56.0 (+36 / +426) |
| Llama 3.2 | 7.8 (+284) | 89.1 (+267) | 50.0 (+460) | 40.0 (+500) | 120.0 (+251) | 44.3 (-13 / +277) |
| Llama 4 | 8.1 (+303) | 89.2 (+295) | 50.0 (+460) | 50.0 (+400) | 110.0 (+401) | 40.9 (+9 / +424) |
| gpt-oss | 8.4 (+273) | 88.9 (+278) | 50.0 (+462) | 41.0 (+488) | 110.0 (+474) | 36.9 (+7 / +372) |
| Qwen 3.6 | 8.3 (+296) | 88.5 (+292) | 50.0 (+460) | 40.0 (+500) | 121.0 (+298) | 31.2 (+28 / +687) |
| GLM-5.3 | 6.5 (+290) | 87.7 (+235) | 50.0 (+460) | 41.0 (+485) | 110.0 (+274) | 53.6 (-9 / +293) |
| MiniMax M3 | 6.6 (+268) | 94.9 (+210) | 50.0 (+460) | 40.0 (+500) | 121.0 (+305) | 31.2 (+16 / +328) |
| MiMo V2.6 | 7.6 (+274) | 88.6 (+257) | 50.0 (+460) | 41.0 (+488) | 121.0 (+240) | 38.6 (-1 / +268) |
| Mistral (tekken) | 7.6 (-) | 89.1 (-) | 50.0 (-) | 40.0 (+752) | 241.0 (+453) | 37.7 (+2693 / -) |
| Gemma 3 | 15.9 (+562) | 105.6 (+498) | 260.0 (+212) | 51.0 (+1000) | 391.0 (+82) | 36.1 (+389 / +1296) |
| Gemma 4 | 15.7 (+564) | 104.5 (+502) | 250.0 (+216) | 51.0 (+980) | 390.0 (+75) | 44.7 (+440 / +1642) |
| Llama 2 (byte fallback) | 13.5 (+668) | 119.8 (+532) | 270.0 (+263) | 60.0 (+1070) | 401.0 (+130) | 70.5 (+150 / +403) |
| Kimi K3 | 6.9 (+32) | 91.4 (+195) | 50.0 (+340) | 40.0 (+428) | 130.0 (+224) | 23.4 (+386 / -) |
| BERT base | 6.6 (+8399) | 96.2 (+1765) | 211.0 (+464) | 50.0 (+3286) | 320.0 (+367) | 9.9 (+682 / -) |
| T5 small | 6.7 (+1756) | 108.0 (+337) | 210.0 (+72) | 60.0 (+600) | 361.0 (+78) | 0.7 (+13 / -) |

#### Decode, streaming and batch: v1.15.0 to 1.16.0 (change in time; negative = 1.16.0 faster; the batch column is the change in tokens per second, where positive = 1.16.0 faster)

| family | full decode, 2,048 ids | incremental decode, 2,048 ids | first delta of one stream | 64 streams: step | 64 streams: first delta | batch of 1,000 texts, one thread |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | -9.9 % | +1.5 % | +0.0 % | -18.0 % | -0.9 % | +15.6 % |
| DeepSeek V3.2 | -5.3 % | +0.3 % | +0.0 % | +0.0 % | +0.0 % | +28.3 % |
| Llama 3.2 | -8.7 % | +0.2 % | +0.0 % | -20.0 % | +9.1 % | +14.7 % |
| Llama 4 | -12.2 % | -1.3 % | +0.0 % | +0.0 % | -0.9 % | +19.2 % |
| gpt-oss | -7.0 % | -0.3 % | +0.0 % | -18.0 % | -0.9 % | +14.1 % |
| Qwen 3.6 | -7.5 % | -0.3 % | +0.0 % | +0.0 % | +0.0 % | +40.0 % |
| GLM-5.3 | -9.2 % | +0.2 % | +0.0 % | -18.0 % | -15.4 % | +13.9 % |
| MiniMax M3 | -4.9 % | +0.4 % | +0.0 % | -20.0 % | +0.8 % | +29.2 % |
| MiMo V2.6 | -6.8 % | +0.3 % | +0.0 % | -18.0 % | -6.9 % | +32.6 % |
| Mistral (tekken) | -12.9 % | +0.1 % | +0.0 % | -2.4 % | +100.8 % | +10.8 % |
| Gemma 3 | -0.8 % | +0.9 % | +0.0 % | -15.0 % | -7.1 % | +8.9 % |
| Gemma 4 | +2.5 % | +1.6 % | -0.4 % | -15.0 % | -5.1 % | +49.1 % |
| Llama 2 (byte fallback) | +2.0 % | +0.7 % | -3.6 % | +0.0 % | +2.8 % | +6.6 % |
| Kimi K3 | -9.7 % | -0.3 % | +0.0 % | +0.0 % | +0.0 % | +48.6 % |
| BERT base | -99.0 % | -95.7 % | -86.7 % | -97.6 % | -81.5 % | +689.8 % |
| T5 small | -93.6 % | -75.2 % | -46.3 % | -83.4 % | -33.1 % | +0.9 % |

#### Load and memory (each cell: SMG 1.16.0 / SMG v1.15.0 / fastokens)

| family | load, ms | resident memory after the load, MB | high-water mark after a 1m-token encode, MB | growth per thread at 64 threads, one shared tokenizer, kB | resident level per thread, a tokenizer per thread, MB |
|---|---|---|---|---|---|
| DeepSeek V4.1 | 196.5 / 442.5 / 678.1 | 72.2 / 138.4 / 112.8 | 94 / 145 / 197 | 697 / 969 / 16562 | 70.6 / 124.0 / 141.5 |
| DeepSeek V3.2 | 199.4 / 430.6 / 669.0 | 71.0 / 136.8 / 112.8 | 93 / 143 / 197 | 698 / 969 / 16559 | 72.9 / 119.9 / 131.8 |
| Llama 3.2 | 264.1 / 747.7 / 1127.4 | 93.3 / 184.4 / 284.0 | 114 / 250 / 337 | 707 / 996 / 16553 | 98.7 / 248.5 / 309.4 |
| Llama 4 | 458.3 / 1225.2 / 2043.8 | 116.8 / 284.4 / 439.7 | 133 / 390 / 526 | 708 / 985 / 32950 | 110.1 / 345.4 / 498.1 |
| gpt-oss | 460.8 / 1263.7 / 2094.7 | 113.9 / 287.1 / 442.1 | 133 / 396 / 528 | 718 / 986 / 32965 | 120.4 / 354.8 / 495.5 |
| Qwen 3.6 | 413.1 / 1038.0 / 1492.4 | 112.7 / 237.6 / 186.1 | 133 / 243 / 270 | 655 / 892 / 32950 | 150.1 / 209.7 / 208.2 |
| GLM-5.3 | 313.4 / 885.4 / 1391.2 | 107.4 / 211.0 / 325.9 | 122 / 289 / 377 | 717 / 999 / 16568 | 113.1 / 269.1 / 349.6 |
| MiniMax M3 | 310.1 / 753.8 / 1107.4 | 87.3 / 172.2 / 158.9 | 111 / 177 / 276 | 724 / 989 / 32985 | 104.0 / 157.4 / 223.3 |
| MiMo V2.6 | 221.3 / 603.0 / 888.5 | 71.8 / 141.0 / 212.9 | 95 / 162 / 266 | 704 / 997 / 16553 | 79.7 / 174.9 / 243.8 |
| Mistral (tekken) | 317.2 / 355.4 / 183.6 | 81.2 / 99.6 / 72.1 | 112 / 130 / 94 | 690 / 926 / 461 | 72.6 / 91.1 / 91.1 |
| Gemma 3 | 595.8 / 1614.2 / 2411.0 | 156.9 / 289.1 / 473.2 | 176 / 466 / 506 | 605 / 695 / 1883 | 172.2 / 417.7 / 463.6 |
| Gemma 4 | 571.2 / 1627.9 / 2446.3 | 156.8 / 288.6 / 467.8 | 173 / 464 / 499 | 599 / 689 / 1831 | 173.9 / 417.3 / 464.0 |
| Llama 2 (byte fallback) | 40.2 / 106.0 / 133.2 | 26.4 / 40.2 / 42.7 | 52 / 59 / 119 | 632 / 974 / 1819 | 20.5 / 29.7 / 41.9 |
| Kimi K3 | 363.4 / 412.1 / 2574.7 | 83.3 / 101.9 / 71.9 | 112 / 129 / 79 | 704 / 996 / 179 | 76.4 / 95.5 / 66.5 |
| BERT base | 16.6 / 22.1 / 52.4 | 14.5 / 20.1 / 19.5 | 24 / 602 / 596 | 49 / 1882 / 1929 | 7.4 / 13.9 / 15.8 |
| T5 small | 42.4 / 51.9 / 89.4 | 35.3 / 43.0 / 44.5 | 393 / 462 / 455 | 1660 / 1658 / 1665 | 33.0 / 40.6 / 37.5 |

#### Reading the tables

- Of 192 encode cells SMG 1.16.0 reads a lower time than every reference in the csv files (fastokens, tiktoken-rs, tokenizers, v1) on 127: 42 of 96 cold and 85 of 96 warm; SMG v1.15.0 in the same rounds reads 83.
- Against v1.15.0, cold: of 96 encode cells 91 read faster and 0 slower by more than 5 %, 5 within it; the median change is -17.3 %.
- Against v1.15.0, warm: of 96 encode cells 95 read faster and 0 slower by more than 5 %, 1 within it; the median change is -23.7 %.
- Against v1.15.0, decode, streaming and batch: of 96 cells 47 read faster and 2 slower by more than 5 %, 47 within it. Slower: Mistral (tekken), 64 streams: first delta, 241.0 against 120.0 ns (+100.8 %); Llama 3.2, 64 streams: first delta, 120.0 against 110.0 ns (+9.1 %, one step of the clock). The step and first-delta cells read a clock that moves in steps of about 10 ns; 9 of the faster cells differ by one step or less.
- Against fastokens, cold: of 96 encode cells SMG is ahead beyond the placement band (20 %) on 40 and behind beyond it on 36; 20 read inside the band (the margins run from -43 % to +1647 %).
- Against fastokens, warm: of 96 encode cells SMG is ahead beyond the placement band (20 %) on 82 and behind beyond it on 0; 14 read inside the band (the margins run from -14 % to +6873 %).
- Against v1, cold: of 72 encode cells SMG is ahead beyond the placement band (20 %) on 66 and behind beyond it on 0; 6 read inside the band (the margins run from +9 % to +2120 %).
- Against v1, warm: of 72 encode cells SMG is ahead beyond the placement band (20 %) on 66 and behind beyond it on 3; 3 read inside the band (the margins run from -63 % to +4465 %).
- Full decode, 2,048 ids, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +32 % to +8399 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- Incremental decode, 2,048 ids, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +195 % to +1765 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- First delta of one stream, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +72 % to +464 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- 64 streams: step, against fastokens: SMG is ahead beyond the band on 16 of 16 families, behind beyond it on 0, inside it on 0 (the margins run from +382 % to +3286 %).
- 64 streams: first delta, against fastokens: SMG is ahead beyond the band on 16 of 16 families, behind beyond it on 0, inside it on 0 (the margins run from +75 % to +821 %).
- Batch of 1,000 texts, one thread, against fastokens: SMG is ahead beyond the band on 8 of 16 families, behind beyond it on 0, inside it on 8 (the margins run from -13 % to +2693 %).
- Batch of 1,000 texts, one thread, against v1: SMG is ahead beyond the band on 12 of 12 families, behind beyond it on 0, inside it on 0 (the margins run from +268 % to +1642 %).
- Load and memory, against v1.15.0: SMG 1.16.0 reads lower by more than 5 % on 16 of 16 families in the load time; on 16 of 16 families in the resident memory after the load; on 16 of 16 families in the high-water mark after a 1m-token encode; on 15 of 16 families in the growth per thread on one shared tokenizer; on 16 of 16 families in the resident level per thread with a tokenizer per thread.
- Load and memory, against fastokens: SMG 1.16.0 reads lower by more than 5 % on 15 of 16 families in the load time (higher on Mistral (tekken)); on 14 of 16 families in the resident memory after the load (higher on Mistral (tekken), Kimi K3); on 14 of 16 families in the high-water mark after a 1m-token encode (higher on Mistral (tekken), Kimi K3); on 13 of 16 families in the growth per thread on one shared tokenizer (higher on Mistral (tekken), Kimi K3); on 15 of 16 families in the resident level per thread with a tokenizer per thread (higher on Kimi K3).
- The v1 column reads `-` for Mistral (tekken), Kimi K3, BERT base, T5 small: the v1 pipeline declines their files, and that build takes its next encode path there (SMG's own for Mistral (tekken), Kimi K3, BERT base; the `tokenizers` crate's for T5 small).
- The placement band: the v1 build shares SMG's decode and streaming code, and between the two builds those 80 cells read -18 % to +25 %.
- The corpus: 17 to 39 % of the bytes of a size class are outside ASCII (`corpus.json`).

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
    create_tokenizer_from_file, create_tokenizer, SequenceDecoderOutput, StopSequenceConfig,
    StopSequenceDecoder, Tokenizer,
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

// Incremental decoding with stop sequences: the stop decoder takes every
// token id and decodes incrementally itself
let mut stop = StopSequenceDecoder::new(
    Arc::clone(&inner),
    StopSequenceConfig::default().with_stop_sequence("\nHuman:"),
    true, // skip special tokens
);
for &token in encoding.token_ids() {
    match stop.process_token(token)? {
        SequenceDecoderOutput::Text(t) => print!("{t}"),
        SequenceDecoderOutput::StoppedWithText(t) => {
            print!("{t}");
            break;
        }
        SequenceDecoderOutput::Stopped => break,
        SequenceDecoderOutput::Held => {}
    }
}
// End of input: release the text still held as a possible start of a stop
match stop.finish()? {
    SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) => println!("{t}"),
    SequenceDecoderOutput::Held | SequenceDecoderOutput::Stopped => println!(),
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
