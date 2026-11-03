//! Benchmark: what a pixel-cache hit copies.
//!
//! The gateway's pixel cache keeps a preprocessed image behind a shared pointer and hands a hit
//! out as a clone of the whole preprocessed item (the encoder input and its side tensors), and a
//! request with several cached images clones each hit and then joins the parts into one batch, a
//! copy of every byte again. Both copies are of this crate's types, so the rows here price them on
//! a 1,024-patch image (512 x 512 under the Qwen3-VL geometry: 32 x 32 patches of 16 px, 256
//! merged tokens): one image, the clone a hit returns, and four images, four clones and the join;
//! as the image's own bytes, the form the cache holds when no wire dtype is resolved, and as the
//! bfloat16 and float32 wires it holds otherwise. The rows time the copies; what they produce is
//! dropped outside the measurement. Beside the rows, the allocation calls one hit makes, counted
//! by this binary's own allocator.
//!
//! Run:  cargo bench -p llm-multimodal --bench pixel_cache_hit

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator reports the allocation calls of a hit beside its time"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashMap,
    hint::black_box,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    encoder_inputs::EncoderDtype,
    vision::{preprocessor_config::PreProcessorConfig, processors::Qwen3VLProcessor},
    FieldLayout, Modality, ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor,
};

/// Allocation and reallocation calls made through the global allocator so far.
static CALLS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes (this binary is single-threaded while it runs).
fn allocations_during<T>(work: impl FnOnce() -> T) -> usize {
    let before = CALLS.load(Ordering::Relaxed);
    black_box(work());
    CALLS.load(Ordering::Relaxed) - before
}

/// A 512 x 512 image with some variation: 32 x 32 patches of 16 px under the processor's
/// geometry, which keeps a multiple of 32 as it is.
fn image() -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(512, 512, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    }))
}

/// The image-side layouts of the Qwen3-VL spec, as the gateway hands them to the join: the
/// grid and the per-image patch counts stack one row per image.
fn layouts() -> HashMap<String, FieldLayout> {
    HashMap::from([
        ("image_grid_thw".to_string(), FieldLayout::Batched),
        ("patches_per_image".to_string(), FieldLayout::Batched),
    ])
}

/// The cache's entry for the image in one form: the processor's own bytes, or the wire bytes.
fn entry(processor: &Qwen3VLProcessor, wire: Option<EncoderDtype>) -> PreprocessedEncoderInputs {
    let images = [image()];
    let preprocessed = match wire {
        Some(dtype) => processor
            .preprocess_wire(&images, dtype)
            .expect("the processor writes the wire dtype in one pass")
            .unwrap(),
        None => processor.preprocess(&images).unwrap(),
    };
    assert_eq!(
        preprocessed.feature_token_counts,
        vec![256],
        "1,024 patches merge into 256 tokens"
    );
    assert!(
        matches!(
            preprocessed.model_specific.get("image_grid_thw"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data.as_slice() == [1, 32, 32]
        ),
        "the grid is 1 x 32 x 32"
    );
    assert_eq!(
        preprocessed.encoder_input.shape()[0],
        1_024,
        "the image is 1,024 patches"
    );
    preprocessed
}

/// What the gateway does on a hit: the cached item cloned out of its shared pointer.
fn hit(cached: &Arc<PreprocessedEncoderInputs>) -> PreprocessedEncoderInputs {
    (**cached).clone()
}

/// What the gateway does for a request of several cached images: one hit per image, then the
/// parts joined into the request's batch.
fn batch(
    cached: &[Arc<PreprocessedEncoderInputs>],
    layouts: &HashMap<String, FieldLayout>,
) -> PreprocessedEncoderInputs {
    let parts = cached.iter().map(hit).collect();
    PreprocessedEncoderInputs::concat(parts, layouts).unwrap()
}

#[expect(
    clippy::print_stdout,
    reason = "the allocation calls of a hit are printed beside the timing rows"
)]
fn bench_pixel_cache_hit(c: &mut Criterion) {
    let config = PreProcessorConfig::from_json(
        r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
    )
    .unwrap();
    let processor = Qwen3VLProcessor::from_config_for(&config, Modality::Image);
    let layouts = layouts();
    let forms = [
        ("u8", None),
        ("bf16", Some(EncoderDtype::BFloat16)),
        ("f32", Some(EncoderDtype::Float32)),
    ];
    let mut group = c.benchmark_group("pixel_cache_hit");
    group
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    for (form, wire) in forms {
        let cached = Arc::new(entry(&processor, wire));
        let four: Vec<Arc<PreprocessedEncoderInputs>> =
            (0..4).map(|_| Arc::clone(&cached)).collect();
        let bytes = cached.encoder_input.nbytes() as u64;
        println!(
            "pixel_cache_hit/{form}: {bytes} encoder bytes per image; one hit: {} allocation \
             calls; four hits and the join: {} allocation calls",
            allocations_during(|| hit(&cached)),
            allocations_during(|| batch(&four, &layouts)),
        );
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new(form, "1_image"), &cached, |b, cached| {
            b.iter_with_large_drop(|| hit(cached));
        });
        group.throughput(Throughput::Bytes(bytes * 4));
        group.bench_with_input(BenchmarkId::new(form, "4_images"), &four, |b, four| {
            b.iter_with_large_drop(|| batch(four, &layouts));
        });
    }
    group.finish();
}

criterion_group!(pixel_cache, bench_pixel_cache_hit);
criterion_main!(pixel_cache);
