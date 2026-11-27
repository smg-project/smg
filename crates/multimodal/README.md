# llm-multimodal

## Overview

`llm-multimodal` is the media side of a chat request. It takes the images, video clips and
audio clips a request names, fetches and decodes them, runs the preprocessing the model's
family defines, and returns the encoder inputs and the prompt replacements, so a gateway
can hand a serving engine tensors in place of media.

Key capabilities:

- a registry of model families (`ModelRegistry`): a model id and its `config.json` resolve
  to a `ModelProcessorSpec`, which names the modalities and their limits, the placeholder
  tokens and their expansion, and builds the family's processors
- image, video and audio processors that follow each family's reference processor: the
  resize, the tiling or the patch layout, the normalize, the log-mel front ends
- the encoder input written in one pass in the dtype its destination takes
  (`EncoderDtype`: `float32`, `bfloat16`, `float16`, `uint8`)
- media from URLs, `data:` URLs, base64 payloads, inline bytes and files (`MediaSource`),
  fetched under size limits, an optional allow-list of hosts and an address policy that
  refuses hosts that are not public
- image decode for PNG, JPEG, GIF, BMP, ICO, TIFF and WebP; video through `ffmpeg`, or
  OpenCV behind a feature; audio through an in-process decoder with `ffmpeg` behind it
- large buffers kept between requests in a bounded pool, and a preprocessing pool the
  host sizes once at start-up

## Source map

- `lib.rs` – module exports
- `registry/` – `ModelRegistry`, the `ModelProcessorSpec` trait (`traits.rs`) and one module per family
- `vision/` – the `VisionPreProcessor` trait (`processor.rs`), the processors (`processors/`), the shared transforms and resamplers, and where preprocessing runs (`execution.rs`)
- `audio/` – the `AudioPreProcessor` trait (`processor.rs`), the decode (`decode.rs`) and the front ends (`processors/`)
- `media.rs` – `MediaConnector`: the fetch, the image decode, the video frame sampling
- `encoder_inputs.rs` – `EncoderInput`, `EncoderDtype`, `PreprocessedEncoderInputs` and the one-pass writes
- `types.rs` – what a request's media becomes (`ImageFrame`, `VideoClip`, `AudioClip`), `Modality`, `TransparencyPolicy`, `PromptReplacement`
- `buffer_pool.rs`, `wire_bytes.rs` – the buffers kept between requests; owned wire bytes and their shared views
- `jpeg_turbo.rs`, `webp_native.rs` – libjpeg-turbo and libwebp, loaded at runtime where the host has them
- `data_url.rs`, `fetch_policy.rs` – the size of an inline payload and its readers; which hosts a media URL may be fetched from
- `hub.rs` – a model's config files from the Hugging Face Hub (feature `hf-hub`)
- `error.rs` – the error types, their codes and whose fault an error is

`AsyncMultiModalTracker`, exported at the root, is the request-side entry: it takes a
request's media parts one by one (`push_part`), fetches and decodes them through a
`MediaConnector`, and returns them from `finalize`.

## Model families and processors

`ModelRegistry::lookup` returns the first registered family whose `matches` accepts the
model's id and config. The families of `ModelRegistry::new` in registration order, each by
the name its `name()` returns, with the processors it builds:

| family | vision processor | audio processor |
|---|---|---|
| `deepseek_v41` | `DeepseekV41Processor` (image) | |
| `gemma4` | `Gemma4Processor` (image, video) | |
| `glm53_flash` | `Glm53FlashProcessor` (image, video) | |
| `inkling2` | `Inkling2ImageProcessor` (image) | |
| `inkling` | `InklingImageProcessor` (image) | `InklingAudioProcessor` |
| `kimi_k3` | `KimiK3Processor` (image) | |
| `kimi_k25` | `KimiK25Processor` (image) | |
| `llama4` | `Llama4VisionProcessor` (image) | |
| `minimax_m3` | `MiniMaxM3VisionProcessor` (image, video) | |
| `muse_spark` | `MuseSparkVisionProcessor` (image, video) | `MuseSparkAudioProcessor` |
| `nemotron_omni` | `NemotronOmniProcessor` (image, video) | `ParakeetAudioProcessor` |
| `nemotron_transcribe` | | `NemotronTranscribeAudioProcessor` |
| `llava_next` | `LlavaNextProcessor` (image) | |
| `llava` | `LlavaProcessor`, or `PixtralProcessor` where the config's vision tower is `pixtral` (image) | |
| `qwen3_asr` | | `Qwen3AudioProcessor` |
| `qwen3_omni` | `Qwen3OmniVisionProcessor` (image, video) | `Qwen3AudioProcessor` |
| `qwen3_vl` | `Qwen3VLProcessor` (image, video) | |
| `qwen_vl` | `Qwen2VLProcessor` (image, video) | |
| `phi3_v` | `Phi3VisionProcessor` (image) | |
| `step3_vl` | `Step3VLProcessor` (image) | |

Which of a family's modalities one model serves is read from its config: most families
take video or audio only where the config declares the modality's tokens or its tower.
`QwenVLProcessorBase` is the base that `Qwen2VLProcessor`, `Qwen3VLProcessor`,
`Qwen3OmniVisionProcessor` and `MiniMaxM3VisionProcessor` wrap.

`ModelRegistry::for_format` keeps the families whose processors emit one
`EncoderInputFormat`. For `Reference` it also registers `nemotron_vl_tiled` and
`nemotron_vl` ahead of them, which `new` does not hold: the Nemotron VL and Omni
checkpoints in the reference tensor layout, image and video through
`NemotronOmniProcessor::reference`, no audio.

## Media decoding

- **Images.** PNG, JPEG, GIF, BMP, ICO, TIFF and WebP. A JPEG decodes through
  libjpeg-turbo and a still WebP through libwebp where the host has the library; both
  are loaded at runtime and nothing links against them. Without them the pure-Rust
  decoders of the `image` crate take the file, and a JPEG's pixels then differ from
  libjpeg-turbo's by a few levels. AVIF has no decoder here and is refused with a typed
  error.
- **Video.** A clip is decoded by the `ffmpeg` executable on the path; with the
  `opencv-video` feature OpenCV is tried first. `SMG_VIDEO_DECODE_BACKEND` (`auto`,
  `opencv`, `ffmpeg`) picks one. Frames are sampled by the family's own rule
  (`FrameSampling`).
- **Audio.** An in-process decoder first, `ffmpeg` for what it does not take.
- **Transparency.** `TransparencyPolicy` says what an alpha channel becomes: `keep_alpha`
  (the default), `composite_over_white` or `drop_alpha`.

## Parallelism

Preprocessing is CPU-bound, and the crate never touches rayon's global pool. A host calls
`configure_parallelism` once at start-up and chooses a dedicated pool of the size it names
or inline execution on the calling thread. A host that configures nothing gets a small
dedicated pool on first use, sized by `SMG_MM_THREADS` where that is set.

## Environment variables

The crate reads fifteen variables. Each is read once, on first use, and kept for the life
of the process, so a change takes a restart. A number that is unset, unreadable or zero
falls back to its default.

| Variable | Default | What it sets |
|---|---|---|
| `SMG_IMAGE_MAX_INPUT_BYTES` | 256 MiB | The most bytes one image may have as it arrives (fetched, read from a file or decoded from base64), before it is decoded. |
| `SMG_VIDEO_MAX_INPUT_BYTES` | 256 MiB | The same for one video. |
| `SMG_AUDIO_MAX_INPUT_BYTES` | 256 MiB | The same for one audio clip. |
| `SMG_IMAGE_MAX_COUNT`, `SMG_VIDEO_MAX_COUNT`, `SMG_AUDIO_MAX_COUNT` | the model family's own limit | How many images, videos or audio clips one request may carry. It replaces the family's limit, up or down, and never enables a modality the family does not serve. |
| `SMG_VIDEO_DECODE_BACKEND` | `auto` | The video decoder: `auto`, `opencv` or `ffmpeg` (see Media decoding). `opencv` needs the `opencv-video` feature. Any other value fails every video decode with an error that names it. |
| `SMG_VIDEO_PROCESS_TIMEOUT_SECS` | 30 | The time one decoder run may spend on a clip, in seconds; a fraction is allowed. With the `opencv-video` feature, a clip the buffered OpenCV pass fails on, a timeout included, gets the same budget again in OpenCV from a temporary file, and under `auto` once more in `ffmpeg`. |
| `SMG_VIDEO_MAX_DECODED_BYTES` | 1 GiB | The most bytes the decoded frames of one clip may take. |
| `SMG_VIDEO_MAX_DECODE_PIXELS` | 10^12 | The most pixels one video decode may run through: the source frames it has to decode, times the clip's width and height. A decode past it is refused before a decoder starts. |
| `SMG_AUDIO_DECODE_BACKEND` | `auto` | The audio decoder: `auto` (the in-process decoder first, then `ffmpeg`), `symphonia` or `ffmpeg`. Any other value fails every audio decode with an error that names it. |
| `SMG_AUDIO_PROCESS_TIMEOUT_SECS` | 30 | The time one decoder may spend on a clip, in seconds; a fraction is allowed. Under `auto` a clip the in-process decoder fails on, a timeout included, gets the same budget again in `ffmpeg`. |
| `SMG_AUDIO_MAX_DECODED_BYTES` | 256 MiB | The most bytes the decoded samples of one clip may take. |
| `SMG_MM_THREADS` | the machine's parallelism, at most 8 | The threads of the dedicated pool where the host names no size (see Parallelism). |
| `SMG_LOG_MM_TIMING` | off | `1`, `true`, `yes` or `on` logs how long each video decode took. A host that calls `init_log_video_decode_timing` before the first decode decides instead. |

The gateway is such a host: its `--log-mm-timing` flag wins over `SMG_LOG_MM_TIMING`, and
the media pipeline it runs worker-side preprocesses inline unless `SMG_MM_THREADS` is set.
The test run has two settings of its own (`SMG_REQUIRE_TURBOJPEG` and
`SMG_REQUIRE_VIDEO_TOOLS`, see Testing), and so has the concurrency bench (see Benchmarks).

## Cargo features

- `hf-hub` (default) – the `hub` module: `config.json` and the processor configs of a
  model id, downloaded into the local Hub cache. A host that resolves model files itself
  builds with `default-features = false`.
- `opencv-video` – the OpenCV video backend; it needs OpenCV's headers and libraries at
  build time.

## Testing

`cargo test -p llm-multimodal` runs the unit tests and the suites under `tests/` (fetch,
preprocess, video). A test that needs libjpeg-turbo or the video tools skips, and says
so, where the host lacks them; `SMG_REQUIRE_TURBOJPEG` and `SMG_REQUIRE_VIDEO_TOOLS` turn
those skips into failures, as CI sets them. The scripts under `scripts/` regenerate the
golden fixtures from the reference processors.

## Benchmarks

### The crate's own benches

`cargo bench -p llm-multimodal --bench <name>` runs one of the crate's criterion benches:
`image_preprocess` (the preprocessing of `deepseek_v41`, `gemma4`, `llama4`, `qwen_vl`,
`qwen3_vl` and `step3_vl` at several image sizes, the shared steps one at a time in the
`step_*` groups, and `decode_preprocess`), `minimax_m3_preprocess` (`m3_image_batch`,
`m3_source_resize` and `m3_video_clips`, serial against parallel), `pixel_cache_hit` (what
a hit of the gateway's pixel cache copies, for one image and for four, as the processor's
own bytes and as bfloat16 and float32 wire bytes, with the allocation counts beside),
`audio_clip` (a 30 s WAV clip through the decode and the audio front end's batch entry,
alone and eight at once, with the allocation counts; `audio_clip_siblings` runs the other
front ends on the same clip) and `audio_preprocess` (the resample and the log-mel front
end; its header says why it tracks one build over time and is not the tool to compare two
builds with). `minimax_m3_concurrency` is a closed-loop request bench that prints one JSON
record per arm; it runs only under `SMG_M3_CONCURRENCY_BENCH=1` and reads `SMG_MM_THREADS`
and `SMG_M3_BENCH_CASE` (`1080p`, `4k`, `mixed` or `rgba1080p`).

Three timing tests are ignored by default and run by name:
`cargo test -p llm-multimodal --release --test image_data_url_bench -- --ignored --nocapture`
(the request-side entry over inline images; it needs no file),
`--test decode_parallel_bench` (a batch of photos decoded one after another against
`ImageFrame::decode_all`; `REAL_JPEG` names the file) and `--test decode_preprocess_bench`
(`REAL_JPEG` and `PP_CONFIG`).

The examples are the tools for two builds side by side:
`cargo run --release -p llm-multimodal --example mm_baseline -- --manifest <classes.json> --out <out.jsonl> --threads 1 --repeats 10`
is the harness (every stage of the pipeline over the classes of a manifest, one JSON line
per class with the stage times, the CPU and the digests of what it wrote), and
`--example video_sampling_timing -- <clip>...` times the video path and digests the frames
it samples. `scripts/bench/multimodal/` holds the baseline built on the harness: one
command over a corpus, with the reference arm beside it, and the same classes through the
gateway.

### Release against release

The tables below are the recorded run of `benches/release`, the comparison a release is
read with. `run.py build` compiles one source of the harness against both releases,
`run.py rounds` runs the two executables over the same class lists in interleaved rounds on
pinned cores, `run.py parity` and `run.py reference` add the byte comparison on a wider
class list and the reference column, and `run.py summarize` writes the tables.
[benches/release/README.md](benches/release/README.md) has the commands and the whole
measurement contract, [RESULTS.md](benches/release/RESULTS.md) the table of every
configuration, and `cells.csv` beside it every cell. What a cell measures:

- **image**: one processor configuration by one class, on one thread: the header read and
  the decode, the processor's one-pass write in the dtype of the wire, and the move of the
  wire bytes, per image. One harness process runs the 20 classes of a configuration, 10
  timed repeats a class; a round runs both releases, in an order that swaps every round; a
  cell is the median of its rounds. A change is this release's time over the previous
  one's, minus one: negative means this release is faster. Within 2 % either way a cell
  reads as level.
- **same bytes**: the shape, the grid and the SHA-256 of the decoded pixels, of the float32
  write and of the wire bytes are equal between the releases. The wider list compares 59
  classes on the bf16 and the float32 wire, with alpha composited over white and with alpha
  kept: 236 outputs a configuration.
- **reference**: the image processor class the pinned `transformers` holds for the
  checkpoint's config, on the same files with one compute thread: its time per image over
  this crate's, and whether the decoded pixels and the float32 output are its bytes. The
  crate's figure also carries the write of the wire bytes; the reference's does not. A
  processor that is code shipped inside a checkpoint is never run, so such a configuration
  has no reference column.
- **audio**: one audio front end by one uncompressed WAV clip, read the same way, per clip.

Classes: 19 image files and one batch of ten, written by `benches/release/make_images.py`
from one seeded synthetic scene: JPEG from 256x256 to 8192x8192, progressive, CMYK,
greyscale and EXIF-rotated; PNG from 256x256 to 3840x2160 and with alpha; WebP lossy and
lossless; GIF, BMP and TIFF. 19 WAV clips written by `make_audio.py`: 16-, 24- and 32-bit
PCM, float and A-law, 8 to 96 kHz, one, two and six channels, 0.1 s to 600 s.

Configurations: 20, from 16 of the families above. `llava` runs once with each of its two
processors, `qwen3_vl` and `qwen_vl` with the configs of two checkpoints each, and
`step3_vl` with each of its two model types. The image processors of `inkling` and
`inkling2` are not in the tables.

Setup: One x86-64 core, release builds, one harness source on both releases (the gateway's allocator with its options, the decoded frame borrowed, the consuming write on another thread as the gateway does it), the bf16 wire, alpha composited over white, 10 timed repeats per class, 15 interleaved rounds per build with the order swapped (medians). 1.16.0 = `322350993`, 1.15.0 = `82c0f1fcd`. "Same bytes" compares the SHA-256 of the decoded pixels, of the float32 write and of the wire bytes. The reference column runs the processor class each checkpoint's config names (transformers 5.17.0, torch 2.13.0+cpu, Pillow 12.3.0, numpy 2.2.6), one compute thread, its decode and preprocess stages; this release's figure beside it also carries the write of the wire bytes.

#### `qwen3_vl`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference, one thread | reference over 1.16.0 |
|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.51 | -31 % | 0.50 | -31 % | yes | 2.42 | 4.73x |
| JPEG 1024x768 | 3.72 | 0 % | 3.69 | +1 % | yes | 19.3 | 5.18x |
| JPEG 1920x1080 | 11.2 | -17 % | 11.1 | -17 % | yes | 63.1 | 5.65x |
| JPEG 4096x2304 | 76.1 | -2 % | 75.6 | -2 % | yes | 319 | 4.20x |
| JPEG 8192x8192 | 451 | -26 % | 448 | -26 % | yes | 1011 | 2.24x |
| JPEG progressive | 6.89 | -1 % | 6.83 | -1 % | yes | 19.2 | 2.78x |
| JPEG CMYK | 8.21 | 0 % | 8.15 | 0 % | yes | 23.1 | 2.81x |
| JPEG greyscale | 3.29 | -2 % | 3.26 | -1 % | yes | 15.3 | 4.65x |
| JPEG EXIF-rotated | 2.03 | -20 % | 2.01 | -21 % | yes | 7.31 | 3.60x |
| PNG 256x256 | 0.61 | -27 % | 0.59 | -27 % | yes | 2.90 | 4.78x |
| PNG 1024x768 | 5.94 | -12 % | 5.89 | -11 % | yes | 25.6 | 4.32x |
| PNG 1920x1080 | 16.2 | -23 % | 16.1 | -23 % | yes | 85.6 | 5.30x |
| PNG 3840x2160 | 98.9 | -14 % | 98.2 | -14 % | yes | 387 | 3.91x |
| PNG RGBA (alpha) | 8.48 | -19 % | 8.37 | -19 % | yes | 32.2 | 3.79x |
| WebP lossy | 5.32 | -1 % | 5.27 | -1 % | yes | 19.3 | 3.63x |
| WebP lossless | 16.1 | 0 % | 16.0 | 0 % | yes | 29.5 | 1.83x |
| GIF | 8.15 | -15 % | 8.09 | -16 % | yes | 17.3 | 2.12x |
| BMP | 3.72 | -7 % | 3.69 | -7 % | yes | 13.1 | 3.51x |
| TIFF | 3.06 | -2 % | 3.03 | -2 % | yes | 15.4 | 5.03x |
| batch of ten 1024x768 JPEGs | 6.54 | +1 % | 6.49 | +1 % | yes | 23.6 | 3.61x |

#### The 20 processor configurations, medians over the 19 image classes (change against 1.15.0)

| processor | time per image vs 1.15.0 | least .. most | CPU per image vs 1.15.0 | batch of ten | same bytes, the 20 standard classes | same bytes, 59 classes on both wires, alpha composited and kept | reference over 1.16.0 (least .. most) | the reference's bytes: decoded pixels, float32 |
|---|---|---|---|---|---|---|---|---|
| `deepseek_v41` | -15 % | -22 % .. -1 % | -15 % | -44 % | 20 of 20 | 236 of 236 | - (no such class in the library) | - |
| `gemma4` | -47 % | -75 % .. -13 % | -47 % | -67 % | 20 of 20 | 236 of 236 | 1.89x (1.26x .. 4.91x) | 20 of 20, 20 of 20 |
| `glm53_flash` | -51 % | -62 % .. -26 % | -51 % | -45 % | 20 of 20 | 224 of 224 (12 refused by both) | 2.78x (1.62x .. 4.79x) | 20 of 20, 20 of 20 |
| `kimi_k25` | -35 % | -47 % .. -8 % | -35 % | -62 % | 20 of 20 | 236 of 236 | 2.56x (1.48x .. 3.98x) | 20 of 20, 0 of 20 |
| `kimi_k3` | -36 % | -61 % .. -12 % | -36 % | -61 % | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - |
| `llama4` | -2 % | -22 % .. +7 % | -2 % | -4 % | 0 of 20 | 0 of 236 | 0.88x (0.49x .. 4.22x) | 20 of 20, 0 of 20 |
| `llava_next` | -19 % | -37 % .. +4 % | -19 % | -18 % | 0 of 20 | 0 of 236 | 1.95x (1.28x .. 3.37x) | 20 of 20, 20 of 20 |
| `llava` | -17 % | -33 % .. +6 % | -17 % | -9 % | 0 of 20 | 0 of 236 | 1.98x (1.10x .. 3.66x) | 20 of 20, 20 of 20 |
| `llava` with the `pixtral` processor | -29 % | -41 % .. -9 % | -29 % | -25 % | 20 of 20 | 236 of 236 | 3.31x (1.52x .. 5.47x) | 20 of 20, 20 of 20 |
| `minimax_m3` | -36 % | -49 % .. -11 % | -36 % | -47 % | 20 of 20 | 224 of 224 (12 refused by both) | 2.25x (1.37x .. 3.78x) | 20 of 20, 20 of 20 |
| `muse_spark` | -28 % | -32 % .. -16 % | -28 % | -22 % | 20 of 20 | 236 of 236 | - (no such class in the library) | - |
| `nemotron_omni` | -34 % | -47 % .. -2 % | -34 % | -40 % | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - |
| `phi3_v` | -48 % | -58 % .. -32 % | -48 % | -44 % | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - |
| `qwen3_omni` | -13 % | -30 % .. +1 % | -13 % | 0 % | 20 of 20 | 224 of 224 (12 refused by both) | 3.76x (1.85x .. 5.52x) | 20 of 20, 20 of 20 |
| `qwen3_vl` | -12 % | -31 % .. 0 % | -11 % | +1 % | 20 of 20 | 224 of 224 (12 refused by both) | 3.91x (1.83x .. 5.65x) | 20 of 20, 20 of 20 |
| `qwen3_vl` (second config) | -13 % | -30 % .. 0 % | -12 % | -3 % | 20 of 20 | 224 of 224 (12 refused by both) | 3.73x (1.74x .. 5.69x) | 20 of 20, 20 of 20 |
| `qwen_vl` | -41 % | -50 % .. -22 % | -41 % | -36 % | 20 of 20 | 224 of 224 (12 refused by both) | 2.98x (1.68x .. 4.08x) | 20 of 20, 20 of 20 |
| `qwen_vl` (second config) | -43 % | -52 % .. -23 % | -43 % | -35 % | 20 of 20 | 224 of 224 (12 refused by both) | 3.09x (1.68x .. 4.39x) | 20 of 20, 20 of 20 |
| `step3_vl` | -4 % | -16 % .. 0 % | -4 % | -2 % | 20 of 20 | 228 of 228 (8 refused by both) | - (no such class in the library) | - |
| `step3_vl` (model type `step5_vl`) | -4 % | -14 % .. +3 % | -4 % | -4 % | 20 of 20 | 228 of 228 (8 refused by both) | - (no such class in the library) | - |

A class refused by both releases lies outside the processor's own bounds (an extreme aspect
ratio). `llama4` ran with the checkpoint's published processor configuration and
`muse_spark` with the processor values of the crate's golden fixture.

#### `qwen3_asr`, milliseconds per clip

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per clip | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| 16-bit 16 kHz mono, 0.1 s | 0.16 | +2 % | 0.15 | +2 % | yes |
| 16-bit 16 kHz mono, 1 s | 0.59 | +1 % | 0.57 | 0 % | yes |
| 16-bit 16 kHz mono, 30 s | 14.9 | -3 % | 14.7 | -3 % | yes |
| 16-bit 16 kHz mono, 600 s | 324 | 0 % | 321 | 0 % | yes |
| 16-bit 16 kHz mono, clipped signal, 30 s | 14.9 | -2 % | 14.7 | -2 % | yes |
| 16-bit 16 kHz mono, constant offset, 30 s | 14.7 | -3 % | 14.6 | -3 % | yes |
| 16-bit 16 kHz mono, silence, 30 s | 14.9 | -1 % | 14.7 | -2 % | yes |
| 16-bit 8 kHz mono, 30 s | 36.8 | +1 % | 36.5 | +1 % | yes |
| 16-bit 22.05 kHz mono, 30 s | 690 | +3 % | 686 | +3 % | yes |
| 16-bit 44.1 kHz mono, 30 s | 716 | +3 % | 711 | +3 % | yes |
| 16-bit 48 kHz mono, 30 s | 80.3 | +2 % | 79.6 | +1 % | yes |
| 16-bit 96 kHz mono, 30 s | 146 | 0 % | 145 | 0 % | yes |
| 16-bit 44.1 kHz stereo, 30 s | 718 | +3 % | 713 | +3 % | yes |
| 16-bit 48 kHz stereo, 30 s | 83.0 | +1 % | 82.4 | +1 % | yes |
| 16-bit 48 kHz six channels, 30 s | 93.7 | +1 % | 92.9 | +1 % | yes |
| 24-bit 16 kHz mono, 30 s | 15.1 | -2 % | 15.0 | -2 % | yes |
| 32-bit 16 kHz mono, 30 s | 15.1 | -1 % | 15.0 | -1 % | yes |
| float 16 kHz mono, 30 s | 14.8 | -3 % | 14.7 | -2 % | yes |
| A-law 8 kHz mono, 30 s | 37.2 | +1 % | 36.8 | +1 % | yes |

#### Reading the tables

- **Faster per image, the same bytes.** On one core, 338 of the 380 image cells (20 processor configurations, 19 image classes each) take over 2 % less time per image than in v1.15.0, 37 read within 2 % and 5 read slower; every cell whose output was not changed on purpose writes the same bytes as v1.15.0. Medians over the image classes: `glm53_flash` -51 %, `phi3_v` -48 %, `gemma4` -47 %, `qwen_vl` (second config) -43 %, `qwen_vl` -41 %, `kimi_k3` -36 %, `minimax_m3` -36 %, `kimi_k25` -35 %, `nemotron_omni` -34 %, `llava` with the `pixtral` processor -29 %, `muse_spark` -28 %, `llava_next` -19 %, `llava` -17 %, `deepseek_v41` -15 %, `qwen3_omni` -13 %, `qwen3_vl` (second config) -13 %, `qwen3_vl` -12 %, `step3_vl` (model type `step5_vl`) -4 %, `step3_vl` -4 %, `llama4` -2 %.
- **Changed on purpose.** `llama4`, `llava`, `llava_next`: their tensors differ from v1.15.0 in every class, the decoded pixels do not. The `llava` and `llava_next` processors resize and normalize as their reference processor does; v1.15.0 listed both as not yet bit-identical. The `llama4` processor normalizes in a new order: every tensor changed, and its float32 output is the bytes of its reference class in 0 of 20 classes (0 of 20 in v1.15.0).
- **Cells that read slower (5 of 380, over 2 % slower).** Slower in every round: `llava` JPEG 8192x8192 +6 % (345 ms against 325 ms, slower in 15 of 15 rounds, the rounds from 1.02 to 1.13). Slower by the medians, with rounds on both sides of level: `llama4` BMP +7 % (28.5 ms against 26.7 ms, slower in 10 of 15 rounds, the rounds from 0.83 to 1.47); `llava_next` JPEG 8192x8192 +4 % (377 ms against 362 ms, slower in 14 of 15 rounds, the rounds from 0.93 to 1.08); `step3_vl` (model type `step5_vl`) JPEG 256x256 +3 % (3.76 ms against 3.66 ms, slower in 7 of 15 rounds, the rounds from 0.86 to 1.46); `llama4` JPEG greyscale +3 % (26.6 ms against 25.9 ms, slower in 7 of 15 rounds, the rounds from 0.84 to 1.46).
- **Reading a slower cell.** A cell slower in every round is a cost. A cell whose rounds
  fall on both sides of level moves from one harness process to the next by more than the
  difference it reads: it is listed as it reads and is not shown to be a cost.
  `benches/release/rounds.csv.gz` holds the decode, preprocess and serialize medians of
  every window, which show the stage a difference sits in.
- **Against the reference processors.** For the 13 configurations whose processor is a class of the pinned reference library, the reference's time per image over this release's, one compute thread each (median over the image classes, least .. most): `qwen3_vl` 3.91x (1.83x .. 5.65x), `qwen3_omni` 3.76x (1.85x .. 5.52x), `qwen3_vl` (second config) 3.73x (1.74x .. 5.69x), `llava` with the `pixtral` processor 3.31x (1.52x .. 5.47x), `qwen_vl` (second config) 3.09x (1.68x .. 4.39x), `qwen_vl` 2.98x (1.68x .. 4.08x), `glm53_flash` 2.78x (1.62x .. 4.79x), `kimi_k25` 2.56x (1.48x .. 3.98x), `minimax_m3` 2.25x (1.37x .. 3.78x), `llava` 1.98x (1.10x .. 3.66x), `llava_next` 1.95x (1.28x .. 3.37x), `gemma4` 1.89x (1.26x .. 4.91x), `llama4` 0.88x (0.49x .. 4.22x). A figure under 1x is a class where the reference is the faster one. Of 260 outputs read against the reference, the decoded pixels are the same bytes in 260 and the float32 output in 220 (v1.15.0: 180 of 260); the ones that differ are every class of `kimi_k25` and of `llama4`, in v1.15.0 and in this release alike (for `kimi_k25` the reference read here is the class the pinned library holds for the checkpoint's config; the processor code the checkpoint itself ships is never run here). No reference column: `deepseek_v41` (no such class in the library); `kimi_k3` (its processor is checkpoint code); `muse_spark` (no such class in the library); `nemotron_omni` (its processor is checkpoint code); `phi3_v` (its processor is checkpoint code); `step3_vl` (no such class in the library); `step3_vl` (model type `step5_vl`) (no such class in the library).
- **Audio.** 19 uncompressed WAV classes (16-, 24- and 32-bit PCM, float and A-law; 8 to 96 kHz; one, two and six channels; 0.1 s to 600 s) through the audio front end of `qwen3_asr` and `qwen3_omni`, time per clip on one core, 15 interleaved rounds: 7 of the 38 cells take over 2 % less time than in v1.15.0, 25 read within 2 % and 6 read slower; 38 of 38 write the same bytes as v1.15.0. Medians over the classes: `qwen3_asr` +1 %, `qwen3_omni` 0 %. Slower: `qwen3_omni` 16-bit 44.1 kHz stereo, 30 s +3 % (718 ms against 694 ms, slower in 15 of 15 rounds, the rounds from 1.02 to 1.10); `qwen3_asr` 16-bit 44.1 kHz mono, 30 s +3 % (716 ms against 693 ms, slower in 15 of 15 rounds, the rounds from 1.03 to 1.10); `qwen3_asr` 16-bit 44.1 kHz stereo, 30 s +3 % (718 ms against 695 ms, slower in 15 of 15 rounds, the rounds from 1.03 to 1.10); `qwen3_omni` 16-bit 44.1 kHz mono, 30 s +3 % (714 ms against 692 ms, slower in 15 of 15 rounds, the rounds from 1.03 to 1.10); `qwen3_asr` 16-bit 22.05 kHz mono, 30 s +3 % (690 ms against 669 ms, slower in 15 of 15 rounds, the rounds from 1.02 to 1.11); `qwen3_omni` 16-bit 22.05 kHz mono, 30 s +3 % (688 ms against 668 ms, slower in 15 of 15 rounds, the rounds from 1.02 to 1.08). 6 of the 6 slower cells are clips at another rate than 16 kHz, which pass through the resampler; their rounds read, this release's time over the previous one's, `qwen3_asr` from 1.02 to 1.11 (median 1.03) and `qwen3_omni` from 1.02 to 1.10 (median 1.03). The resampler's source is the same in both releases and each release is built by its own compiler: the cause is not established. Compressed audio (FLAC, MP3, AAC, Opus, Vorbis) and video were not measured for this release: the run had neither the clips nor the decoder they need.
- **Not covered.** One thread on one machine: no cell under load and none through the
  gateway. A class is read in a process that ran the classes before it, the same on both
  sides; `benches/release/README.md` says what a class read in a process of its own
  changes. Video, compressed audio, and the image processors of `inkling` and `inkling2`
  have no cell.
