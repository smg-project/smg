//! The multimodal baseline of record: SMG's media pipeline stage by stage over the
//! classes of a manifest (image formats and sizes, a batch, video, audio), one item at a
//! time or on several threads at once, with the digests that make a run comparable
//! with the reference processor's output on the same files.
//!
//! The entry point is `scripts/bench/multimodal/mm_baseline.py`: it resolves the classes
//! manifest against a corpus directory, runs this example at each thread count, runs
//! the reference processor on the same files and writes the table. By hand:
//!
//! ```text
//! cargo run --release -p llm-multimodal --example mm_baseline -- \
//!     --manifest resolved.json --out crate.jsonl --threads 1 --repeats 20
//! ```
//!
//! The manifest is a JSON object `{"classes": [...]}`. A class has `name`, `kind`
//! (`image`, `batch`, `video` or `audio`), `model_id` (the id the registry resolves the
//! family from), `config_dir` (the checkpoint's `config.json` and preprocessor configs)
//! and `files` (absolute paths, one per item of the request). Every stage runs as the
//! gateway runs it on the router placement: the header read and the decode of every
//! frame (`ImageFrame::decode_all`), the processor's one-pass write over the frames'
//! own decoded pixels, borrowed as the gateway borrows them, in the destination's dtype
//! (bf16 unless `--wire` says otherwise; the two-pass path where a processor has no
//! one-pass write),
//! and the move of the wire bytes (`into_bytes_as`); the wire then returns to the buffer
//! pool as a sent payload returns it (`recycled_bytes`), so a run measures the pool's
//! steady state rather than a fresh zeroed buffer per iteration. The pool is
//! process-wide and bounded, so per-class numbers are read one class per process
//! (`--only`, or a one-class manifest): in a multi-class run a later class inherits
//! the pool the classes before it filled, and its row depends on their order. The
//! pool keeps at most its bound of buffers, so at a thread count whose wires exceed
//! it some iterations take a fresh buffer, as the gateway's requests do under the
//! same concurrency: the row reports that steady state, and at a low `--repeats`
//! its averaged columns and its p90 carry each thread's first cold take (the median
//! only at `--repeats 1`).
//! Video decodes through the media connector under the family's own sampling rule;
//! audio through the in-process decoder.
//!
//! The harness runs the gateway's allocator: jemalloc with the options the `smg`
//! executable and both bindings export, so a cell reads the page-return policy
//! production runs. Under the system allocator the large transient buffers of a 1080p
//! image (about 20 MB on the torch-exact path) went back to the OS after every image
//! and were faulted in again on the next, which charged a kernel for that policy
//! rather than for its own work.
//!
//! The output is JSON Lines. The first line, `{"kind":"meta",...}`, names the
//! architecture, the core count, the decoders found at runtime (TurboJPEG, libwebp),
//! the crate's preprocessing pool, the vector-kernel paths this build takes and the
//! allocator. Then one
//! `{"kind":"row",...}` per class: the per-item latency of every stage (median and p90
//! over all threads' iterations), the wall time, the throughput, the process CPU time
//! per item (every thread and the pool included), the byte counts (encoded, decoded,
//! wire), the output shape and grid, and the digests: `rgb_sha256` of the first item's
//! decoded RGB bytes, `f32_sha256` of the two-pass float32 write (what the reference
//! processor produces), `wire_sha256` of the one-pass wire bytes and whether they equal
//! the two-pass write in the same dtype. A class that fails gets `error` instead, and
//! the run goes on. The last line on standard error is
//! `BASELINE_DONE classes=<n> ok=<n> err=<n>`.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool: progress on stderr, the usage text on stdout"
)]

use std::{
    collections::HashSet,
    env,
    error::Error,
    fs,
    io::{BufWriter, Cursor, Write},
    mem::size_of,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Barrier},
    thread,
    time::Instant,
};

use bytes::Bytes;
use image::{DynamicImage, ImageReader};
use llm_multimodal::{
    audio::{decode_audio_mono_f32, Qwen3AudioProcessor},
    buffer_pool::{recycle, recycled_bytes},
    encoder_inputs::{EncoderDtype, EncoderInput, ModelSpecificValue, PreprocessedEncoderInputs},
    jpeg_turbo, parallelism,
    registry::{ModelMetadata, ModelRegistry, Tokenizer},
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource, Modality, TransparencyPolicy},
    vision::{PreProcessorConfig, VideoClipOptions, VisionPreProcessor},
    webp_native, MediaConnector, MediaConnectorConfig, MediaSource, Parallelism, VideoClip,
    VideoFetchConfig,
};
use rustix::time::{clock_gettime, ClockId};
use rustls::crypto::ring;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};

// The gateway's allocator and its options (`SERVER_MALLOC_CONF` in the
// gateway's metrics module, the bytes the `smg` executable and both bindings
// export; this crate sits below the gateway, so the string is repeated here).
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[expect(
    unsafe_code,
    reason = "jemalloc reads its options from this exported symbol; an immutable NUL-terminated byte string, read alone by jemalloc and the meta line"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

/// The allocator this build runs, for the meta line.
fn allocator() -> &'static str {
    if cfg!(all(not(target_env = "msvc"), not(target_env = "musl"))) {
        "jemalloc"
    } else {
        "system: jemalloc is not built in on this target"
    }
}

/// The options exported to jemalloc, the bytes themselves without the NUL
/// (`None` where jemalloc is not built in). The environment's
/// `_RJEM_MALLOC_CONF`, read after them and winning per key, is recorded
/// beside them so a results file names both inputs.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
fn allocator_conf() -> Option<&'static str> {
    std::str::from_utf8(&MALLOC_CONF[..MALLOC_CONF.len() - 1]).ok()
}

/// See the jemalloc form above: nothing is exported on this target.
#[cfg(not(all(not(target_env = "msvc"), not(target_env = "musl"))))]
fn allocator_conf() -> Option<&'static str> {
    None
}

const USAGE: &str = "mm_baseline: SMG's media pipeline per class, stage by stage\n\
    \n\
    Options:\n\
    \x20 --manifest <file>   the resolved classes manifest (JSON)\n\
    \x20 --out <file>        the JSON Lines written (one meta line, one row per class)\n\
    \x20 --threads <n>       threads running the class at once [1]\n\
    \x20 --repeats <n>       iterations per thread [10]\n\
    \x20 --wire <dtype>      bf16, f16, f32 or u8: the destination's dtype [bf16]\n\
    \x20 --transparency <p>  composite_white, drop_alpha or keep_alpha [composite_white]\n\
    \x20 --only <a,b,...>    run these classes only\n\
    \x20 --help              this text";

/// A fault in the run's setup, reported on standard error with a non-zero exit status.
type Fault = Box<dyn Error>;

#[derive(Deserialize)]
struct Manifest {
    classes: Vec<Class>,
}

#[derive(Deserialize)]
struct Class {
    name: String,
    kind: String,
    model_id: String,
    config_dir: PathBuf,
    files: Vec<PathBuf>,
    /// Iterations per thread for this class when fewer than `--repeats` (a long clip).
    #[serde(default)]
    repeats: Option<usize>,
}

struct Options {
    manifest: PathBuf,
    out: PathBuf,
    threads: usize,
    repeats: usize,
    wire: EncoderDtype,
    transparency: TransparencyPolicy,
    only: Option<HashSet<String>>,
}

impl Options {
    /// `None` when `--help` was asked for and printed.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Option<Self>, Fault> {
        let mut manifest = None;
        let mut out = None;
        let mut threads = 1;
        let mut repeats = 10;
        let mut wire = EncoderDtype::BFloat16;
        let mut transparency = TransparencyPolicy::CompositeOverWhite;
        let mut only = None;
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                println!("{USAGE}");
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{arg} takes a value\n{USAGE}"))?;
            match arg.as_str() {
                "--manifest" => manifest = Some(PathBuf::from(value)),
                "--out" => out = Some(PathBuf::from(value)),
                "--threads" => threads = value.parse()?,
                "--repeats" => repeats = value.parse()?,
                "--wire" => {
                    wire = match value.as_str() {
                        "bf16" => EncoderDtype::BFloat16,
                        "f16" => EncoderDtype::Float16,
                        "f32" => EncoderDtype::Float32,
                        "u8" => EncoderDtype::Uint8,
                        other => return Err(format!("unknown wire dtype {other:?}").into()),
                    }
                }
                "--transparency" => {
                    transparency = match value.as_str() {
                        "composite_white" => TransparencyPolicy::CompositeOverWhite,
                        "drop_alpha" => TransparencyPolicy::DropAlpha,
                        "keep_alpha" => TransparencyPolicy::KeepAlpha,
                        other => return Err(format!("unknown transparency {other:?}").into()),
                    }
                }
                "--only" => only = Some(value.split(',').map(str::to_string).collect()),
                other => return Err(format!("unknown argument {other:?}\n{USAGE}").into()),
            }
        }
        Ok(Some(Self {
            manifest: manifest.ok_or("--manifest is required")?,
            out: out.ok_or("--out is required")?,
            threads: threads.max(1),
            repeats: repeats.max(1),
            wire,
            transparency,
            only,
        }))
    }
}

/// The registry resolves a family from the model id and `config.json` alone here; no
/// token of the vocabulary is read.
struct NoTokenizer;

impl Tokenizer for NoTokenizer {
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }
}

/// The checkpoint's configs, read the way the gateway reads a local model directory.
struct ModelConfigs {
    config: Value,
    image: PreProcessorConfig,
    video: Option<PreProcessorConfig>,
}

impl ModelConfigs {
    fn load(dir: &Path) -> Result<Self, Fault> {
        let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let image = PreProcessorConfig::from_json(&fs::read_to_string(
            dir.join("preprocessor_config.json"),
        )?)?;
        let video = fs::read_to_string(dir.join("video_preprocessor_config.json"))
            .ok()
            .map(|text| PreProcessorConfig::from_json(&text))
            .transpose()?;
        Ok(Self {
            config,
            image,
            video,
        })
    }
}

/// One class, prepared: its processor and the bytes of its files.
enum Work {
    Images {
        processor: Arc<dyn VisionPreProcessor>,
        files: Vec<Bytes>,
    },
    Video {
        processor: Arc<dyn VisionPreProcessor>,
        connector: Arc<MediaConnector>,
        config: VideoFetchConfig,
        bytes: Vec<u8>,
    },
    Audio {
        processor: Arc<Qwen3AudioProcessor>,
        bytes: Bytes,
    },
}

/// The per-item stage times of one iteration, in milliseconds, with the byte counts.
#[derive(Clone, Copy, Default)]
struct Sample {
    decode_ms: f64,
    preprocess_ms: f64,
    serialize_ms: f64,
    decoded_bytes: usize,
    wire_bytes: usize,
}

impl Sample {
    fn total_ms(&self) -> f64 {
        self.decode_ms + self.preprocess_ms + self.serialize_ms
    }
}

/// What one class reports besides its timings.
struct Facts {
    shape: Vec<usize>,
    grid: Vec<i64>,
    items: usize,
    encoded_bytes: usize,
    rgb_sha256: Option<String>,
    f32_sha256: Option<String>,
    u8_sha256: Option<String>,
    wire_sha256: String,
    one_pass_equals_two_pass: Option<bool>,
    video_frames: Option<usize>,
    /// A clip's `video_second_per_grid`, the time one temporal patch spans as the processor reports it.
    video_second_per_grid: Option<f32>,
}

#[expect(
    clippy::disallowed_methods,
    reason = "an example binary's client factory; the process TLS crypto provider is installed first"
)]
fn http_client_builder() -> reqwest::ClientBuilder {
    let _ = ring::default_provider().install_default();
    reqwest::Client::builder()
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// SHA-256 of `bytes` as lowercase hex.
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The process's CPU time (every thread), in milliseconds.
fn process_cpu_ms() -> f64 {
    let t = clock_gettime(ClockId::ProcessCPUTime);
    t.tv_sec as f64 * 1000.0 + t.tv_nsec as f64 / 1_000_000.0
}

/// The decoded frames of `files`, header read and decode included, as the gateway makes
/// them for one request: the frames are built from the bytes and decoded together on the
/// preprocessing pool; the processor then borrows their pixels.
fn decode_images(
    files: &[Bytes],
    transparency: TransparencyPolicy,
) -> Result<(Vec<Arc<ImageFrame>>, f64), String> {
    let started = Instant::now();
    let mut frames = Vec::with_capacity(files.len());
    for bytes in files {
        let (width, height) = ImageReader::new(Cursor::new(&bytes[..]))
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .into_dimensions()
            .map_err(|e| e.to_string())?;
        frames.push(Arc::new(
            ImageFrame::encoded(
                bytes.clone(),
                ImageDetail::Auto,
                ImageSource::InlineBytes,
                String::new(),
                ImageSize::new(width, height),
                None,
            )
            .with_transparency(transparency),
        ));
    }
    ImageFrame::decode_all(&frames).map_err(|e| e.to_string())?;
    Ok((frames, ms(started)))
}

/// The gateway's `preprocess_images`: the one-pass write in the wire's dtype when the
/// processor has one, the two-pass path otherwise.
fn preprocess(
    processor: &dyn VisionPreProcessor,
    images: &[&DynamicImage],
    wire: EncoderDtype,
) -> Result<PreprocessedEncoderInputs, String> {
    processor
        .preprocess_wire(images, wire)
        .unwrap_or_else(|| processor.preprocess(images))
        .map_err(|e| e.to_string())
}

/// The gateway's video preprocessing of one decoded clip.
fn preprocess_clip(
    processor: &dyn VisionPreProcessor,
    clip: &VideoClip,
) -> Result<PreprocessedEncoderInputs, String> {
    let options = VideoClipOptions {
        sample_fps: clip.sample_fps(),
        max_long_side_pixel: clip.max_long_side_pixel(),
    };
    if !clip.frames().is_empty() {
        return processor
            .preprocess_video(clip.frames(), &options)
            .map_err(|e| e.to_string());
    }
    if let Some(rgb) = clip.rgb_video() {
        if let Ok(refs) = rgb.frame_refs() {
            if let Ok(inputs) = processor.preprocess_video_rgb(&refs, &options) {
                return Ok(inputs);
            }
        }
    }
    let frames = clip.materialized_frames()?;
    processor
        .preprocess_video(&frames, &options)
        .map_err(|e| e.to_string())
}

fn clip_frame_count(clip: &VideoClip) -> usize {
    match clip.rgb_video() {
        Some(rgb) => rgb.frame_refs().map_or(0, |refs| refs.len()),
        None => clip.frames().len(),
    }
}

/// The image grid the processor reports, whichever integer tensor it uses for it.
fn grid_of(inputs: &PreprocessedEncoderInputs) -> Vec<i64> {
    ["image_grid_thw", "video_grid_thw"]
        .iter()
        .find_map(|key| match inputs.model_specific.get(*key) {
            Some(ModelSpecificValue::IntTensor { data, .. }) => Some(data.clone()),
            Some(ModelSpecificValue::UintTensor { data, .. }) => {
                Some(data.iter().map(|&v| i64::from(v)).collect())
            }
            _ => None,
        })
        .unwrap_or_default()
}

/// The time one temporal patch of a clip spans, as the processor reports it.
fn second_per_grid_of(inputs: &PreprocessedEncoderInputs) -> Option<f32> {
    match inputs.model_specific.get("video_second_per_grid") {
        Some(ModelSpecificValue::Tensor { data, .. }) => data.first().copied(),
        _ => None,
    }
}

/// One timed iteration of a class: the stage times and the wire bytes.
fn iterate(
    work: &Work,
    wire: EncoderDtype,
    transparency: TransparencyPolicy,
    runtime: &Runtime,
) -> Result<(Sample, Vec<u8>), String> {
    let mut sample = Sample::default();
    let (inputs, started) = match work {
        Work::Images { processor, files } => {
            let (frames, decode_ms) = decode_images(files, transparency)?;
            sample.decode_ms = decode_ms;
            // Borrowed from the frames' own cells, as the gateway's batch path borrows them.
            let images = frames
                .iter()
                .map(|frame| frame.image().map_err(|e| e.to_string()))
                .collect::<Result<Vec<&DynamicImage>, _>>()?;
            sample.decoded_bytes = images.iter().map(|image| image.as_bytes().len()).sum();
            let started = Instant::now();
            let inputs = preprocess(processor.as_ref(), &images, wire)?;
            sample.preprocess_ms = ms(started);
            (inputs, Instant::now())
        }
        Work::Video {
            processor,
            connector,
            config,
            bytes,
        } => {
            let started = Instant::now();
            let clip = runtime
                .block_on(connector.fetch_video(MediaSource::InlineBytes(bytes.clone()), *config))
                .map_err(|e| e.to_string())?;
            sample.decode_ms = ms(started);
            sample.decoded_bytes = match clip.rgb_video() {
                Some(rgb) => rgb
                    .frame_refs()
                    .map(|refs| refs.iter().map(|frame| frame.data.len()).sum())
                    .unwrap_or(0),
                None => clip
                    .frames()
                    .iter()
                    .map(|frame| frame.as_bytes().len())
                    .sum(),
            };
            let started = Instant::now();
            let inputs = preprocess_clip(processor.as_ref(), &clip)?;
            sample.preprocess_ms = ms(started);
            (inputs, Instant::now())
        }
        Work::Audio { processor, bytes } => {
            let started = Instant::now();
            let decoded = runtime
                .block_on(decode_audio_mono_f32(bytes.clone()))
                .map_err(|e| e.to_string())?;
            sample.decode_ms = ms(started);
            sample.decoded_bytes = decoded.samples.len() * size_of::<f32>();
            let started = Instant::now();
            let inputs = processor
                .preprocess_decoded_clips(&[&decoded])
                .map_err(|e| e.to_string())?;
            sample.preprocess_ms = ms(started);
            (inputs, Instant::now())
        }
    };
    let bytes = inputs.encoder_input.into_bytes_as(wire)?;
    sample.serialize_ms = ms(started);
    sample.wire_bytes = bytes.len();
    Ok((sample, bytes))
}

/// The facts of a class that do not change between iterations: the shape, the grid and
/// the digests, from one untimed pass on the main thread.
fn facts(
    work: &Work,
    wire: EncoderDtype,
    transparency: TransparencyPolicy,
    runtime: &Runtime,
) -> Result<Facts, String> {
    let encoded_bytes = match work {
        Work::Images { files, .. } => files.iter().map(Bytes::len).sum(),
        Work::Video { bytes, .. } => bytes.len(),
        Work::Audio { bytes, .. } => bytes.len(),
    };
    let (_, one_pass) = iterate(work, wire, transparency, runtime)?;
    let wire_sha256 = sha256(&one_pass);
    // The warm-up's pooled buffers return to the pool as the timed loop's wire does,
    // so the pool is warm before the timed loop starts.
    let facts = match work {
        Work::Images { processor, files } => {
            let (frames, _) = decode_images(files, transparency)?;
            let refs = frames
                .iter()
                .map(|frame| frame.image().map_err(|e| e.to_string()))
                .collect::<Result<Vec<&DynamicImage>, _>>()?;
            let rgb = refs.first().map(|image| sha256(image.to_rgb8().as_raw()));
            let two_pass = processor.preprocess(&refs).map_err(|e| e.to_string())?;
            let u8_sha256 = match &two_pass.encoder_input {
                EncoderInput::U8 { patches, .. } => Some(sha256(
                    &patches
                        .as_standard_layout()
                        .iter()
                        .copied()
                        .collect::<Vec<u8>>(),
                )),
                _ => None,
            };
            let two_pass_wire = two_pass.encoder_input.write_as(wire);
            let f32_bytes = two_pass.encoder_input.write_as(EncoderDtype::Float32);
            let facts = Facts {
                shape: two_pass.encoder_input.shape().to_vec(),
                grid: grid_of(&two_pass),
                items: files.len(),
                encoded_bytes,
                rgb_sha256: rgb,
                f32_sha256: Some(sha256(&f32_bytes)),
                u8_sha256,
                wire_sha256,
                one_pass_equals_two_pass: Some(two_pass_wire == one_pass),
                video_frames: None,
                video_second_per_grid: None,
            };
            recycle(two_pass_wire);
            recycle(f32_bytes);
            facts
        }
        Work::Video {
            processor,
            connector,
            config,
            bytes,
        } => {
            let clip = runtime
                .block_on(connector.fetch_video(MediaSource::InlineBytes(bytes.clone()), *config))
                .map_err(|e| e.to_string())?;
            let inputs = preprocess_clip(processor.as_ref(), &clip)?;
            let f32_bytes = inputs.encoder_input.write_as(EncoderDtype::Float32);
            let facts = Facts {
                shape: inputs.encoder_input.shape().to_vec(),
                grid: grid_of(&inputs),
                items: 1,
                encoded_bytes,
                rgb_sha256: None,
                f32_sha256: Some(sha256(&f32_bytes)),
                u8_sha256: None,
                wire_sha256,
                one_pass_equals_two_pass: None,
                video_frames: Some(clip_frame_count(&clip)),
                video_second_per_grid: second_per_grid_of(&inputs),
            };
            recycle(f32_bytes);
            facts
        }
        Work::Audio { processor, bytes } => {
            let decoded = runtime
                .block_on(decode_audio_mono_f32(bytes.clone()))
                .map_err(|e| e.to_string())?;
            let inputs = processor
                .preprocess_decoded_clips(&[&decoded])
                .map_err(|e| e.to_string())?;
            let f32_bytes = inputs.encoder_input.write_as(EncoderDtype::Float32);
            let facts = Facts {
                shape: inputs.encoder_input.shape().to_vec(),
                grid: Vec::new(),
                items: 1,
                encoded_bytes,
                rgb_sha256: None,
                f32_sha256: Some(sha256(&f32_bytes)),
                u8_sha256: None,
                wire_sha256,
                one_pass_equals_two_pass: None,
                video_frames: None,
                video_second_per_grid: None,
            };
            recycle(f32_bytes);
            facts
        }
    };
    recycle(one_pass);
    Ok(facts)
}

/// Build the class's processor through the registry, as the gateway does, and read its
/// files.
fn prepare(class: &Class, connector: &Arc<MediaConnector>) -> Result<Work, Fault> {
    let configs = ModelConfigs::load(&class.config_dir)?;
    let registry = ModelRegistry::new();
    let metadata = ModelMetadata {
        model_id: &class.model_id,
        tokenizer: &NoTokenizer,
        config: &configs.config,
    };
    let spec = registry
        .lookup(&metadata)
        .ok_or_else(|| format!("no family spec matches model id {:?}", class.model_id))?;
    let read = |path: &PathBuf| -> Result<Vec<u8>, Fault> {
        fs::read(path).map_err(|e| format!("{}: {e}", path.display()).into())
    };
    Ok(match class.kind.as_str() {
        "image" | "batch" => {
            let processor = spec
                .vision_processor(&configs.config, &configs.image, Modality::Image)
                .map_err(|e| e.to_string())?;
            let files = class
                .files
                .iter()
                .map(|path| read(path).map(Bytes::from))
                .collect::<Result<Vec<_>, _>>()?;
            Work::Images {
                processor: Arc::from(processor),
                files,
            }
        }
        "video" => {
            // The sampler reads the config the processor is built from, as the gateway
            // does: the checkpoint's video config, else its shared one.
            let video_config = configs.video.as_ref().unwrap_or(&configs.image);
            let processor = spec
                .vision_processor(&configs.config, video_config, Modality::Video)
                .map_err(|e| e.to_string())?;
            let mut config = VideoFetchConfig {
                sampling: spec.video_frame_sampling_with(video_config),
                ..VideoFetchConfig::default()
            };
            if let Some(fps) = spec.default_video_sample_fps_with(video_config) {
                config.sample_fps = fps;
            }
            let path = class.files.first().ok_or("a video class names one file")?;
            Work::Video {
                processor: Arc::from(processor),
                connector: connector.clone(),
                config,
                bytes: read(path)?,
            }
        }
        "audio" => {
            // the crate arm runs the Qwen3 audio frontend: the families whose audio
            // processor it is are named here, so another family's clip cannot run
            // through it under that family's name
            if !matches!(spec.name(), "qwen3_asr" | "qwen3_omni") {
                return Err(format!("no audio arm for the {} family", spec.name()).into());
            }
            let path = class.files.first().ok_or("an audio class names one file")?;
            Work::Audio {
                processor: Arc::new(Qwen3AudioProcessor::from_configs(
                    &configs.config,
                    &configs.image,
                )),
                bytes: Bytes::from(read(path)?),
            }
        }
        other => return Err(format!("unknown class kind {other:?}").into()),
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn stage_summary(samples: &[Sample], pick: fn(&Sample) -> f64) -> Value {
    let mut values: Vec<f64> = samples.iter().map(pick).collect();
    values.sort_by(f64::total_cmp);
    json!({
        "median_ms": percentile(&values, 0.5),
        "p90_ms": percentile(&values, 0.9),
        "min_ms": values.first().copied().unwrap_or(0.0),
    })
}

/// Run one class on `threads` threads, `repeats` iterations each, after one warm-up
/// iteration, and report the row.
fn run_class(
    class: &Class,
    work: Arc<Work>,
    options: &Options,
    runtime: &Runtime,
) -> Result<Value, String> {
    let facts = facts(&work, options.wire, options.transparency, runtime)?;
    let repeats = class
        .repeats
        .map_or(options.repeats, |limit| limit.clamp(1, options.repeats));
    let barrier = Arc::new(Barrier::new(options.threads));
    let cpu_before = process_cpu_ms();
    let started = Instant::now();
    let workers: Vec<_> = (0..options.threads)
        .map(|_| {
            let work = work.clone();
            let barrier = barrier.clone();
            let wire = options.wire;
            let transparency = options.transparency;
            thread::spawn(move || -> Result<Vec<Sample>, String> {
                let runtime = RuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                let mut samples = Vec::with_capacity(repeats);
                barrier.wait();
                for _ in 0..repeats {
                    let (sample, bytes) = iterate(&work, wire, transparency, &runtime)?;
                    // The wire back to the pool, as a sent payload returns it.
                    drop(recycled_bytes(bytes));
                    samples.push(sample);
                }
                Ok(samples)
            })
        })
        .collect();
    let mut samples = Vec::new();
    for worker in workers {
        samples.extend(
            worker
                .join()
                .map_err(|_| "a worker thread panicked".to_string())??,
        );
    }
    let wall_ms = ms(started);
    let cpu_ms = process_cpu_ms() - cpu_before;
    let iterations = samples.len();
    let items = iterations * facts.items;
    let first = samples.first().copied().unwrap_or_default();
    Ok(json!({
        "kind": "row",
        "class": class.name,
        "class_kind": class.kind,
        "model_id": class.model_id,
        "threads": options.threads,
        "repeats": repeats,
        "items_per_request": facts.items,
        "iterations": iterations,
        "wall_ms": wall_ms,
        "requests_per_s": iterations as f64 / (wall_ms / 1000.0),
        "items_per_s": items as f64 / (wall_ms / 1000.0),
        "cpu_ms_per_request": cpu_ms / iterations as f64,
        "cpu_ms_per_item": cpu_ms / items as f64,
        "stage_ms": {
            "decode": stage_summary(&samples, |s| s.decode_ms),
            "preprocess": stage_summary(&samples, |s| s.preprocess_ms),
            "serialize": stage_summary(&samples, |s| s.serialize_ms),
            "total": stage_summary(&samples, Sample::total_ms),
        },
        "bytes": {
            "encoded": facts.encoded_bytes,
            "decoded": first.decoded_bytes,
            "wire": first.wire_bytes,
        },
        "shape": facts.shape,
        "grid": facts.grid,
        "video_frames": facts.video_frames,
        "video_second_per_grid": facts.video_second_per_grid,
        "digests": {
            "rgb_sha256": facts.rgb_sha256,
            "f32_sha256": facts.f32_sha256,
            "u8_sha256": facts.u8_sha256,
            "wire_sha256": facts.wire_sha256,
        },
        "one_pass_equals_two_pass": facts.one_pass_equals_two_pass,
    }))
}

/// The arm of the Pillow-exact passes (`vision/transforms.rs`) this host
/// resizes through, decided as the crate decides it: the AVX-512 arm needs
/// the five features its `detected` reads, else the AVX2 arm, else the
/// portable rows (the only ones off x86-64).
fn resample_arm() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
        {
            "avx512"
        } else if std::arch::is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "portable"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "portable"
    }
}

fn meta(options: &Options) -> Value {
    let pool = match parallelism() {
        Parallelism::Pool(threads) => threads.get(),
        Parallelism::Inline => 0,
    };
    let vector_kernels = if cfg!(target_arch = "aarch64") {
        "aarch64: the crate's vector paths (pixel conversion, resample, orientation, RGB runs, JPEG colour conversion)"
    } else {
        match resample_arm() {
            "avx512" => "x86-64: the AVX-512 arm of the Pillow-exact passes for the uint8 antialiased resample and the AVX2 wire formula of the pixel conversion; orientation, RGB runs and the JPEG colour conversion run scalar",
            "avx2" => "x86-64: the AVX2 arm of the Pillow-exact passes for the uint8 antialiased resample and the AVX2 wire formula of the pixel conversion; orientation, RGB runs and the JPEG colour conversion run scalar",
            _ => "scalar: the crate has no vector path for this architecture",
        }
    };
    json!({
        "kind": "meta",
        "arch": env::consts::ARCH,
        "os": env::consts::OS,
        "cores": thread::available_parallelism().map_or(0, NonZeroUsize::get),
        "turbojpeg": jpeg_turbo::turbojpeg_available(),
        "libwebp": webp_native::webp_available(),
        "libwebp_version": webp_native::webp_version(),
        "pool_threads": pool,
        "vector_kernels": vector_kernels,
        "allocator": allocator(),
        "allocator_conf": allocator_conf(),
        "allocator_env": env::var("_RJEM_MALLOC_CONF").ok(),
        "wire": format!("{:?}", options.wire),
        "transparency": format!("{:?}", options.transparency),
        "threads": options.threads,
        "repeats": options.repeats,
        "build": if cfg!(debug_assertions) { "debug" } else { "release" },
    })
}

fn run(options: &Options) -> Result<(usize, usize), Fault> {
    let manifest: Manifest = serde_json::from_str(&fs::read_to_string(&options.manifest)?)?;
    let connector = Arc::new(MediaConnector::new(
        http_client_builder(),
        MediaConnectorConfig::default(),
    )?);
    let runtime = RuntimeBuilder::new_current_thread().enable_all().build()?;
    let mut out = BufWriter::new(fs::File::create(&options.out)?);
    serde_json::to_writer(&mut out, &meta(options))?;
    out.write_all(b"\n")?;
    let (mut ok, mut err) = (0, 0);
    for class in &manifest.classes {
        if options
            .only
            .as_ref()
            .is_some_and(|only| !only.contains(&class.name))
        {
            continue;
        }
        eprintln!("mm_baseline: {} ({} threads)", class.name, options.threads);
        let row = match prepare(class, &connector) {
            Ok(work) => run_class(class, Arc::new(work), options, &runtime),
            Err(fault) => Err(fault.to_string()),
        };
        let line = match row {
            Ok(row) => {
                ok += 1;
                row
            }
            Err(error) => {
                err += 1;
                eprintln!("mm_baseline: {} failed: {error}", class.name);
                json!({"kind": "row", "class": class.name, "class_kind": class.kind, "threads": options.threads, "error": error})
            }
        };
        serde_json::to_writer(&mut out, &line)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok((ok, err))
}

fn main() -> ExitCode {
    let options = match Options::parse(env::args().skip(1)) {
        Ok(Some(options)) => options,
        Ok(None) => return ExitCode::SUCCESS,
        Err(fault) => {
            eprintln!("mm_baseline: {fault}");
            return ExitCode::FAILURE;
        }
    };
    match run(&options) {
        Ok((ok, err)) => {
            eprintln!("BASELINE_DONE classes={} ok={ok} err={err}", ok + err);
            ExitCode::SUCCESS
        }
        Err(fault) => {
            eprintln!("mm_baseline: {fault}");
            ExitCode::FAILURE
        }
    }
}
