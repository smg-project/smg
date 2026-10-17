//! Benchmark: SMG image preprocessing vs HF processor baseline.
//!
//! Measures the time for model-specific image preprocessing (resize, normalize,
//! patchify) at various image sizes. Compare results with the companion Python
//! script `scripts/bench_image_preprocess.py` which benchmarks HF transformers.
//!
//! Run:  cargo bench -p llm-multimodal --bench image_preprocess

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main,
    measurement::{Measurement, ValueFormatter},
    BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use llm_multimodal::{
    encoder_inputs::EncoderDtype,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processors::{Llama4VisionProcessor, Qwen2VLProcessor, Qwen3VLProcessor},
        transforms, VisionPreProcessor,
    },
};

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
    let processor = Qwen3VLProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen3-VL-8B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });

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
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();

    // Batch benchmarks
    let mut group = c.benchmark_group("qwen3_vl_batch");
    for batch_size in [3, 5, 10] {
        let images: Vec<DynamicImage> = (0..batch_size)
            .map(|i| make_test_image(640 + i * 10, 480 + i * 10))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("640x480", format!("batch{batch_size}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
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
        let images = [image];
        group.bench_with_input(BenchmarkId::new("single", label), &images, |b, imgs| {
            b.iter(|| processor.preprocess(imgs, &config).unwrap());
        });
    }
    group.finish();
}

fn bench_qwen2_vl(c: &mut Criterion) {
    let processor = Qwen2VLProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen2-VL-2B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });

    let sizes: &[(u32, u32)] = &[(224, 224), (640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("qwen2_vl_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_llama4(c: &mut Criterion) {
    let processor = Llama4VisionProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });

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
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
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
    let processor = Llama4VisionProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });

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
                let imgs = [img.clone()];
                b.iter(|| processor.preprocess(&imgs, &config).unwrap());
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
    targets = bench_wire_write
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
/// with the owned wire bytes; the bytes are the same.
fn bench_wire_arms<P: VisionPreProcessor>(
    group: &mut BenchmarkGroup<'_, CpuTime>,
    processor: &P,
    config: &PreProcessorConfig,
    class: &str,
    images: &[DynamicImage],
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
                    processor
                        .preprocess(imgs, config)
                        .unwrap()
                        .encoder_input
                        .into_bytes_as(dtype)
                        .unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("one_pass_{short}"), class),
            images,
            |b, imgs| {
                b.iter(|| {
                    processor
                        .preprocess_wire(imgs, config, dtype)
                        .expect("the pixel-byte processors write wires")
                        .unwrap()
                        .encoder_input
                        .into_bytes_as(dtype)
                        .unwrap()
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

/// The processors that emit pixel bytes write the wire in two passes or in
/// one; this group is the speed of each per image class and wire dtype on
/// the sizes of the groups above (the plain sizes, the extremes) and the
/// sizes just off a target that resize to it.
fn bench_wire_write(c: &mut Criterion<CpuTime>) {
    let config = PreProcessorConfig::from_json(
        r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
    )
    .unwrap();

    let qwen3 = Qwen3VLProcessor::new();
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
        let images = [make_test_image(w, h)];
        bench_wire_arms(&mut group, &qwen3, &config, &class, &images, with_float16);
    }
    group.finish();

    let qwen2 = Qwen2VLProcessor::new();
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
        let images = [make_test_image(w, h)];
        bench_wire_arms(&mut group, &qwen2, &config, &class, &images, with_float16);
    }
    group.finish();
}
