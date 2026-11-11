//! Benchmark: SMG image preprocessing vs HF processor baseline.
//!
//! Measures the time for model-specific image preprocessing (resize, normalize,
//! patchify) at various image sizes. Compare results with the companion Python
//! script `scripts/bench_image_preprocess.py` which benchmarks HF transformers.
//!
//! Run:  cargo bench -p llm-multimodal --bench image_preprocess

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use bytes::Bytes;
use criterion::{
    criterion_group, criterion_main,
    measurement::{Measurement, ValueFormatter},
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType, DynamicImage, RgbImage};
use llm_multimodal::{
    buffer_pool,
    encoder_inputs::{EncoderDtype, EncoderInput, PreprocessedEncoderInputs},
    jpeg_turbo,
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processors::{
            DeepseekV41Processor, Gemma4Processor, KimiK3Processor, Llama4VisionProcessor,
            Qwen2VLProcessor, Qwen3VLProcessor, Step3VLProcessor,
        },
        transforms, VideoClipOptions, VisionPreProcessor,
    },
    Modality,
};

/// Returns an iteration's output buffer to the crate's pool, as a request does
/// once its tensor has been sent: the patch tensor or the wire goes back
/// through the crate's own return path, a float tensor is dropped. Without
/// this every iteration takes the pool's cold path (a class-sized zeroed
/// allocation that the bench's allocator maps fresh each time), which is not
/// what a request pays after the first one of its class.
fn returned(inputs: PreprocessedEncoderInputs) {
    match inputs.encoder_input {
        EncoderInput::U8 { patches, .. } => {
            buffer_pool::recycle(patches.into_raw_vec_and_offset().0);
        }
        EncoderInput::Wire { bytes, .. } => buffer_pool::recycle(bytes),
        EncoderInput::SharedWire { bytes, .. } => bytes.recycle(),
        EncoderInput::F32(_) => {}
    }
}

/// Create a synthetic RGB image with some variation (not all zeros).
fn make_test_image(width: u32, height: u32) -> DynamicImage {
    let img = RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    DynamicImage::ImageRgb8(img)
}

fn load_preprocessor_config(model_path: &str) -> Option<PreProcessorConfig> {
    let config_path = format!("{model_path}/preprocessor_config.json");
    let json = std::fs::read_to_string(&config_path).ok()?;
    PreProcessorConfig::from_json(&json).ok()
}

// ── Full pipeline benchmarks ─────────────────────────────────────

fn bench_qwen3_vl(c: &mut Criterion) {
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen3-VL-8B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });
    let processor = Qwen3VLProcessor::from_config_for(&config, Modality::Image);

    let sizes: &[(u32, u32)] = &[
        (224, 224),
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (3840, 2160),
    ];

    let mut group = c.benchmark_group("qwen3_vl_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| returned(processor.preprocess(imgs).unwrap()));
            },
        );
    }
    group.finish();

    // Batch benchmarks
    let mut group = c.benchmark_group("qwen3_vl_batch");
    for batch_size in [3, 5, 10] {
        let owned: Vec<DynamicImage> = (0..batch_size)
            .map(|i| make_test_image(640 + i * 10, 480 + i * 10))
            .collect();
        let images: Vec<&DynamicImage> = owned.iter().collect();
        group.bench_with_input(
            BenchmarkId::new("640x480", format!("batch{batch_size}")),
            &images,
            |b, imgs| {
                b.iter(|| returned(processor.preprocess(imgs).unwrap()));
            },
        );
    }
    group.finish();

    // Extreme: very small and very large
    let mut group = c.benchmark_group("qwen3_vl_extreme");
    let extremes: &[(u32, u32, &str)] = &[
        (32, 32, "tiny_32x32"),
        (50, 50, "small_50x50"),
        (100, 2000, "tall_100x2000"),
        (2000, 100, "wide_2000x100"),
        (4096, 4096, "huge_4096x4096"),
    ];
    for &(w, h, label) in extremes {
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(BenchmarkId::new("single", label), &images, |b, imgs| {
            b.iter(|| returned(processor.preprocess(imgs).unwrap()));
        });
    }
    group.finish();
}

/// A clip's frames through the video path at three frame sizes off the family's
/// 32-pixel grid, so that every frame is resized to it (a frame already on the
/// grid is patchified from its borrowed bytes and never reaches the resize),
/// the frames patchified two by two into one pooled buffer. The throughput is
/// per frame, in the process's CPU time.
fn bench_qwen3_vl_video(c: &mut Criterion<CpuTime>) {
    const FRAMES: usize = 8;
    let config = PreProcessorConfig::from_json(
        r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}, "temporal_patch_size": 2}"#,
    )
    .unwrap();
    let processor = Qwen3VLProcessor::from_config_for(&config, Modality::Video);
    let clip = VideoClipOptions::sampled_at(2.0);

    let mut group = c.benchmark_group("qwen3_vl_video");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(FRAMES as u64));
    for &(w, h) in &[(650, 490), (1000, 750), (1920, 1080)] {
        let frames: Vec<DynamicImage> = (0..FRAMES).map(|_| make_test_image(w, h)).collect();
        group.bench_with_input(
            BenchmarkId::new("frame", format!("{w}x{h}")),
            &frames,
            |b, frames| {
                b.iter(|| returned(processor.preprocess_video(frames, &clip).unwrap()));
            },
        );
    }
    group.finish();
}

fn bench_qwen2_vl(c: &mut Criterion) {
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen2-VL-2B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });
    let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

    let sizes: &[(u32, u32)] = &[(224, 224), (640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("qwen2_vl_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| returned(processor.preprocess(imgs).unwrap()));
            },
        );
    }
    group.finish();
}

fn bench_llama4(c: &mut Criterion) {
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });
    let processor = Llama4VisionProcessor::from_preprocessor_config(&config);

    let sizes: &[(u32, u32)] = &[
        (224, 224),
        (336, 336),
        (640, 480),
        (1024, 768),
        (1920, 1080),
    ];

    let mut group = c.benchmark_group("llama4_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| returned(processor.preprocess(imgs).unwrap()));
            },
        );
    }
    group.finish();
}

// ── Per-step profiling benchmarks ────────────────────────────────

fn bench_individual_steps(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    // Step 1: Resize only
    let mut group = c.benchmark_group("step_resize");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        // Qwen3-VL target: smart_resize result
        let processor = Qwen3VLProcessor::new();
        let (th, tw) = processor.smart_resize(h as usize, w as usize).unwrap();
        group.bench_with_input(
            BenchmarkId::new("fir_bilinear", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, tw as u32, th as u32, FilterType::Triangle));
            },
        );
    }
    group.finish();

    // Step 2: to_tensor only
    let mut group = c.benchmark_group("step_to_tensor");
    for &(w, h) in sizes {
        // Use a pre-resized image to isolate to_tensor cost
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("rgb8", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::to_tensor(img));
            },
        );
    }
    group.finish();

    // Step 3: normalize only
    let mut group = c.benchmark_group("step_normalize");
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let tensor = transforms::to_tensor(&image);
        group.bench_with_input(
            BenchmarkId::new("f32", format!("{w}x{h}")),
            &tensor,
            |b, t| {
                b.iter_batched(
                    || t.clone(),
                    |mut fresh| transforms::normalize(&mut fresh, &mean, &std),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_llama4_steps(c: &mut Criterion) {
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });
    let processor = Llama4VisionProcessor::from_preprocessor_config(&config);

    // 1024x768 is the worst case (1.8x slower than HF)
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("llama4_steps");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);

        // Full preprocess
        group.bench_with_input(
            BenchmarkId::new("full_preprocess", format!("{w}x{h}")),
            &image,
            |b, img| {
                let imgs = [img];
                b.iter(|| returned(processor.preprocess(&imgs).unwrap()));
            },
        );

        // Just resize
        group.bench_with_input(
            BenchmarkId::new("resize_only", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, 336, 336, FilterType::Triangle));
            },
        );

        // to_tensor_and_normalize on tile-sized image
        let tile_img = make_test_image(336, 336);
        group.bench_with_input(
            BenchmarkId::new("tensor_normalize_336", format!("{w}x{h}")),
            &tile_img,
            |b, img| {
                b.iter(|| {
                    transforms::to_tensor_and_normalize(img, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qwen3_vl,
    bench_qwen2_vl,
    bench_llama4,
    bench_llama4_steps,
    bench_individual_steps,
    bench_fused_to_tensor_normalize,
    bench_to_rgb8,
    bench_resize_detailed,
);
criterion_group!(
    name = wire_benches;
    config = Criterion::default().with_measurement(CpuTime);
    targets = bench_wire_write, bench_deepseek_v41, bench_step3_vl, bench_gemma4, bench_decode_preprocess, bench_qwen3_vl_video
);
criterion_main!(benches, wire_benches);

fn bench_fused_to_tensor_normalize(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];

    let mut group = c.benchmark_group("step_to_tensor_normalize_fused");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::to_tensor_and_normalize(img, &mean, &std));
            },
        );
    }
    group.finish();
}

fn bench_to_rgb8(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("step_to_rgb8");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("rgb8", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| img.to_rgb8());
            },
        );
    }
    group.finish();

    // Also test when image is already RGB8 (should be free)
    let mut group = c.benchmark_group("step_to_rgb8_noop");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
        group.bench_with_input(
            BenchmarkId::new("already_rgb8", format!("{w}x{h}")),
            &rgb,
            |b, img| {
                b.iter(|| img.to_rgb8());
            },
        );
    }
    group.finish();
}

fn bench_resize_detailed(c: &mut Criterion) {
    let processor = Qwen3VLProcessor::new();

    // Benchmark: make_test_image + resize + convert back to DynamicImage
    let mut group = c.benchmark_group("step_resize_full_pipeline");
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let (th, tw) = processor.smart_resize(h as usize, w as usize).unwrap();

        // fir resize (our path)
        group.bench_with_input(
            BenchmarkId::new("fir", format!("{w}x{h}->{tw}x{th}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, tw as u32, th as u32, FilterType::Triangle));
            },
        );

        // image crate resize (old path, for comparison)
        group.bench_with_input(
            BenchmarkId::new("image_crate", format!("{w}x{h}->{tw}x{th}")),
            &image,
            |b, img| {
                b.iter(|| img.resize_exact(tw as u32, th as u32, FilterType::Triangle));
            },
        );
    }
    group.finish();
}

// ── The wire write: two passes against one ───────────────────────

/// The process's CPU time as the measurement of the wire groups. On a loaded
/// host the wall clock counts the other tenants' slices of the core; the CPU
/// clock counts this process's work alone, on every thread it ran on (the
/// preprocessing pool's included), which is also the cost the gateway pays
/// per image. Pinned to one core it is the wall time of the work.
struct CpuTime;

fn process_cpu_time() -> Duration {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

impl Measurement for CpuTime {
    type Intermediate = Duration;
    type Value = Duration;

    fn start(&self) -> Duration {
        process_cpu_time()
    }

    fn end(&self, start: Duration) -> Duration {
        process_cpu_time().saturating_sub(start)
    }

    fn add(&self, v1: &Duration, v2: &Duration) -> Duration {
        *v1 + *v2
    }

    fn zero(&self) -> Duration {
        Duration::ZERO
    }

    fn to_f64(&self, value: &Duration) -> f64 {
        value.as_nanos() as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CpuTimeFormatter
    }
}

/// Nanoseconds of CPU time in the unit of their magnitude.
struct CpuTimeFormatter;

impl ValueFormatter for CpuTimeFormatter {
    fn scale_values(&self, typical: f64, values: &mut [f64]) -> &'static str {
        let (factor, unit) = if typical < 1e3 {
            (1.0, "cpu-ns")
        } else if typical < 1e6 {
            (1e-3, "cpu-µs")
        } else if typical < 1e9 {
            (1e-6, "cpu-ms")
        } else {
            (1e-9, "cpu-s")
        };
        for value in values {
            *value *= factor;
        }
        unit
    }

    fn scale_throughputs(
        &self,
        typical: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        // Bytes (or elements) per CPU-second, scaled by the typical rate.
        let (count, what) = match *throughput {
            Throughput::Bytes(bytes) | Throughput::BytesDecimal(bytes) => (bytes as f64, "B"),
            Throughput::Bits(bits) => (bits as f64, "bit"),
            Throughput::Elements(elements) | Throughput::ElementsAndBytes { elements, .. } => {
                (elements as f64, "elem")
            }
        };
        let per_second = count * 1e9 / typical;
        let (factor, unit) = match (per_second, what) {
            (rate, "B") if rate >= 1e9 => (1e-9, "GB/cpu-s"),
            (rate, "B") if rate >= 1e6 => (1e-6, "MB/cpu-s"),
            (_, "B") => (1e-3, "KB/cpu-s"),
            (rate, "bit") if rate >= 1e9 => (1e-9, "Gbit/cpu-s"),
            (rate, "bit") if rate >= 1e6 => (1e-6, "Mbit/cpu-s"),
            (_, "bit") => (1e-3, "Kbit/cpu-s"),
            (rate, _) if rate >= 1e9 => (1e-9, "Gelem/cpu-s"),
            (rate, _) if rate >= 1e6 => (1e-6, "Melem/cpu-s"),
            (_, _) => (1e-3, "Kelem/cpu-s"),
        };
        for value in values {
            *value = count * 1e9 / *value * factor;
        }
        unit
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "cpu-ns"
    }
}

/// The wire dtypes the arms are measured in: the half-width float the
/// vector formula writes, the single, and (on the classes the table below
/// marks) the half the tables write.
const WIRE_DTYPES: [EncoderDtype; 2] = [EncoderDtype::BFloat16, EncoderDtype::Float32];

/// One image class through both arms, per wire dtype: the two passes
/// (`preprocess` to the patch bytes, then the write in the wire's dtype, as
/// assembly performs it) and the one pass (`preprocess_wire`). Both arms end
/// with the wire bytes going back to the pool; the bytes are the same.
fn bench_wire_arms<P: VisionPreProcessor>(
    group: &mut BenchmarkGroup<'_, CpuTime>,
    processor: &P,
    class: &str,
    images: &[&DynamicImage],
    with_float16: bool,
) {
    let mut dtypes = WIRE_DTYPES.to_vec();
    if with_float16 {
        dtypes.push(EncoderDtype::Float16);
    }
    for dtype in dtypes {
        let short = match dtype {
            EncoderDtype::BFloat16 => "bf16",
            EncoderDtype::Float16 => "f16",
            EncoderDtype::Float32 => "f32",
            EncoderDtype::Uint8 => "u8",
        };
        group.bench_with_input(
            BenchmarkId::new(format!("two_pass_{short}"), class),
            images,
            |b, imgs| {
                b.iter(|| {
                    buffer_pool::recycle(
                        processor
                            .preprocess(imgs)
                            .unwrap()
                            .encoder_input
                            .into_bytes_as(dtype)
                            .unwrap(),
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("one_pass_{short}"), class),
            images,
            |b, imgs| {
                b.iter(|| {
                    buffer_pool::recycle(
                        processor
                            .preprocess_wire(imgs, dtype)
                            .expect("the pixel-byte processors write wires")
                            .unwrap()
                            .encoder_input
                            .into_bytes_as(dtype)
                            .unwrap(),
                    );
                });
            },
        );
    }
}

/// The class label of a source size: the size alone when the size rule
/// leaves it (the arms then differ by the patch layout and the write alone),
/// else the size and its target (a resize in both arms first, as most
/// images have).
fn wire_class(w: u32, h: u32, target: (usize, usize)) -> String {
    let (th, tw) = target;
    if (tw as u32, th as u32) == (w, h) {
        format!("{w}x{h}")
    } else {
        format!("{w}x{h}-to-{tw}x{th}")
    }
}

// ── DeepSeek V4.1: one image per size class ──────────────────────

/// The per-image cost of the patch processor (the contain fit or stretch to
/// the planned grid, then the normalised patches) on one image of each size
/// class, in CPU time: what an image pays before its request can leave.
fn bench_deepseek_v41(c: &mut Criterion<CpuTime>) {
    let config = PreProcessorConfig::from_json(
        r#"{"patch_size": 14, "min_pixels": 295936, "downsample_ratio": 3, "max_image_tokens": 1024}"#,
    )
    .unwrap();
    let processor = DeepseekV41Processor::from_preprocessor_config(&config);
    // (class, width, height): the medians of the size classes an image
    // corpus falls into, from a small diagram to a 4k-side scan.
    let classes: &[(&str, u32, u32)] = &[
        ("small", 300, 232),
        ("medium", 640, 480),
        ("large", 1024, 768),
        ("document", 1240, 1754),
        ("scan", 2048, 1536),
        ("huge", 4096, 3072),
    ];
    let mut group = c.benchmark_group("deepseek_v41");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for &(class, w, h) in classes {
        let plan = processor.plan_image_grid(w, h);
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new(
                "single",
                format!(
                    "{class}-{w}x{h}-to-{}x{}",
                    plan.best_width, plan.best_height
                ),
            ),
            &images,
            |b, imgs| b.iter(|| returned(processor.preprocess(imgs).unwrap())),
        );
    }
    group.finish();
}

// ── Step3-VL: one image per size class ───────────────────────────

/// The per-image cost of the global-view-and-crops processor (the Pillow
/// resizes, the crops, the normalized antialiased resize of the view to 728
/// and of every crop to 504) on one image of each size class, in CPU time:
/// the crop count grows with the size, from none on a small diagram to
/// thirty on a capped 4k-side scan.
fn bench_step3_vl(c: &mut Criterion<CpuTime>) {
    let processor = Step3VLProcessor::new();
    let classes: &[(&str, u32, u32)] = &[
        ("small", 300, 232),
        ("medium", 640, 480),
        ("large", 1024, 768),
        ("document", 1240, 1754),
        ("scan", 2048, 1536),
        ("huge", 4096, 3072),
    ];
    let mut group = c.benchmark_group("step3_vl");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for &(class, w, h) in classes {
        let plan = processor.plan(w, h);
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new(
                "single",
                format!("{class}-{w}x{h}-crops{}", plan.num_patches()),
            ),
            &images,
            |b, imgs| b.iter(|| returned(processor.preprocess(imgs).unwrap())),
        );
    }
    group.finish();
}

// ── One image request at one thread: decode and preprocess ───────

/// `image` encoded as a baseline JPEG at quality 90, the form most requests
/// carry.
fn jpeg_bytes(image: &DynamicImage) -> Bytes {
    let mut encoded = Vec::new();
    JpegEncoder::new_with_quality(&mut encoded, 90)
        .encode_image(image)
        .unwrap();
    Bytes::from(encoded)
}

/// The crate's share of one image request, as a request pays it: the encoded
/// bytes become a frame, the frame decodes on first use (through TurboJPEG
/// when its library loads, else the pure-Rust decoder), the family's
/// processor writes the bfloat16 wire in one pass where it has one (else the
/// patch tensor and then the wire), and the wire bytes go back to the pool.
fn decode_and_preprocess<P: VisionPreProcessor>(processor: &P, encoded: &Bytes, size: ImageSize) {
    let frame = ImageFrame::encoded(
        encoded.clone(),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        String::new(),
        size,
        None,
    );
    let images = [frame.image().unwrap()];
    wire_request(processor, &images);
}

/// One request's share of a processor: the one-pass bfloat16 wire where the
/// processor writes it, else the patch tensor and the wire write; the wire
/// bytes go back to the pool as a sent payload returns them.
fn wire_request<P: VisionPreProcessor>(processor: &P, images: &[&DynamicImage]) {
    let inputs = processor
        .preprocess_wire(images, EncoderDtype::BFloat16)
        .unwrap_or_else(|| processor.preprocess(images))
        .unwrap();
    buffer_pool::recycle(
        inputs
            .encoder_input
            .into_bytes_as(EncoderDtype::BFloat16)
            .unwrap(),
    );
}

/// The Gemma 4 processor on the request's path per image class, in CPU time
/// at one thread: the plain sizes, the budget's own square (768x768 maps to
/// itself under the resize rule, so that row is the one-pass write alone), a
/// square over the budget (896x896 is 3136 patches against 2520: a downscale
/// to 768x768 plus the write) and a batch of ten; every other image resizes
/// to the fixed patch budget, so those rows read the resample and the patch
/// write of about 1.9 M elements each.
fn bench_gemma4(c: &mut Criterion<CpuTime>) {
    let processor = Gemma4Processor::new();
    let mut group = c.benchmark_group("gemma4_preprocess");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for (w, h) in [
        (256, 256),
        (1024, 768),
        (1920, 1080),
        (768, 768),
        (896, 896),
    ] {
        let image = make_test_image(w, h);
        let images = [&image];
        group.bench_with_input(
            BenchmarkId::new("one_image", format!("{w}x{h}")),
            &images,
            |b, imgs| b.iter(|| wire_request(&processor, imgs)),
        );
    }
    let image = make_test_image(1024, 768);
    let batch: Vec<&DynamicImage> = (0..10).map(|_| &image).collect();
    group.bench_with_input(
        BenchmarkId::new("batch", "10x1024x768"),
        &batch,
        |b, imgs| b.iter(|| wire_request(&processor, imgs)),
    );
    group.finish();
}

/// One 1024x768 image through decode and preprocess per family, as a colour
/// JPEG and as a greyscale JPEG, in CPU time at one thread: the class a
/// comparison with the published crate once read slower at one thread while
/// the larger sizes read faster, so a change to the decode or the wire path
/// shows here before it shows in a service. The source is the synthetic
/// image of the other groups, encoded here (the greyscale from its luma).
#[expect(
    clippy::print_stderr,
    reason = "the decoder the rows ran through is printed once beside them"
)]
fn bench_decode_preprocess(c: &mut Criterion<CpuTime>) {
    eprintln!(
        "decode_preprocess rows decode through {}",
        if jpeg_turbo::turbojpeg_available() {
            "TurboJPEG"
        } else {
            "the pure-Rust JPEG decoder (libturbojpeg.so.0 did not load)"
        }
    );
    let (w, h) = (1024, 768);
    let colour = make_test_image(w, h);
    let grey = DynamicImage::ImageLuma8(colour.to_luma8());
    let size = ImageSize::new(w, h);
    let classes = [
        ("jpeg", jpeg_bytes(&colour)),
        ("jpeg_grey", jpeg_bytes(&grey)),
    ];
    let deepseek = DeepseekV41Processor::from_preprocessor_config(
        &PreProcessorConfig::from_json(
            r#"{"patch_size": 14, "min_pixels": 295936, "downsample_ratio": 3, "max_image_tokens": 1024}"#,
        )
        .unwrap(),
    );
    let kimi = KimiK3Processor::from_preprocessor_config(&PreProcessorConfig::default());
    let gemma4 = Gemma4Processor::new();
    let mut group = c.benchmark_group("decode_preprocess");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for (class, encoded) in &classes {
        group.bench_with_input(
            BenchmarkId::new(format!("deepseek_v41_{class}"), format!("{w}x{h}")),
            encoded,
            |b, encoded| b.iter(|| decode_and_preprocess(&deepseek, encoded, size)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("kimi_k3_{class}"), format!("{w}x{h}")),
            encoded,
            |b, encoded| b.iter(|| decode_and_preprocess(&kimi, encoded, size)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("gemma4_{class}"), format!("{w}x{h}")),
            encoded,
            |b, encoded| b.iter(|| decode_and_preprocess(&gemma4, encoded, size)),
        );
    }
    group.finish();
}

/// The processors that emit pixel bytes write the wire in two passes or in
/// one; this group is the speed of each per image class and wire dtype on
/// the sizes of the groups above (the plain sizes, the extremes) and the
/// sizes just off a target that resize to it.
fn bench_wire_write(c: &mut Criterion<CpuTime>) {
    let config = PreProcessorConfig::from_json(
        r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
    )
    .unwrap();

    let qwen3 = Qwen3VLProcessor::from_config_for(&config, Modality::Image);
    // (width, height, measured in float16 too)
    let qwen3_classes: &[(u32, u32, bool)] = &[
        (320, 256, false),
        (640, 480, true),
        (648, 486, false),
        (1280, 960, true),
        (1296, 972, false),
        (1920, 1080, false),
        (3840, 2160, false),
        (32, 32, false),
        (100, 2000, false),
        (2000, 100, false),
        (4096, 4096, false),
    ];
    let mut group = c.benchmark_group("qwen3_vl_wire");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for &(w, h, with_float16) in qwen3_classes {
        let target = qwen3.smart_resize(h as usize, w as usize).unwrap();
        let class = wire_class(w, h, target);
        let image = make_test_image(w, h);
        let images = [&image];
        bench_wire_arms(&mut group, &qwen3, &class, &images, with_float16);
    }
    group.finish();

    let qwen2 = Qwen2VLProcessor::from_preprocessor_config(&config);
    let qwen2_classes: &[(u32, u32, bool)] = &[
        (644, 476, true),
        (640, 480, false),
        (1288, 952, false),
        (1920, 1080, false),
    ];
    let mut group = c.benchmark_group("qwen2_vl_wire");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for &(w, h, with_float16) in qwen2_classes {
        let target = qwen2.smart_resize(h as usize, w as usize).unwrap();
        let class = wire_class(w, h, target);
        let image = make_test_image(w, h);
        let images = [&image];
        bench_wire_arms(&mut group, &qwen2, &class, &images, with_float16);
    }
    group.finish();
}
