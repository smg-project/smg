//! Benchmark: MiniMax-M3 image and video preprocessing.
//!
//! Run:  cargo bench -p llm-multimodal --bench minimax_m3_preprocess

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::{
    preprocessor_config::PreProcessorConfig, processors::MiniMaxM3VisionProcessor,
    VisionPreProcessor,
};
use rayon::prelude::*;

fn make_test_image(width: u32, height: u32) -> DynamicImage {
    let img = RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    DynamicImage::ImageRgb8(img)
}

fn config() -> PreProcessorConfig {
    PreProcessorConfig::from_json(r#"{"do_resize": true, "do_normalize": true}"#).unwrap()
}

/// How much of an image batch is spent deep-copying the decoded frames.
fn bench_clone_overhead(c: &mut Criterion) {
    let processor = MiniMaxM3VisionProcessor::new();
    let cfg = config();

    let mut group = c.benchmark_group("m3_image_batch");
    for &count in &[1usize, 4, 20] {
        let images: Vec<DynamicImage> = (0..count).map(|_| make_test_image(1024, 768)).collect();

        group.bench_with_input(
            BenchmarkId::new("preprocess_only", count),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &cfg).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("clone_then_preprocess", count),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let copies: Vec<DynamicImage> = imgs.to_vec();
                    processor.preprocess(&copies, &cfg).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Several video clips in one request: one at a time versus all at once.
fn bench_video_clips(c: &mut Criterion) {
    let processor = MiniMaxM3VisionProcessor::new();
    let cfg = config();

    let mut group = c.benchmark_group("m3_video_clips");
    group.sample_size(20);
    for &clips in &[2usize, 4, 8] {
        let batch: Vec<Vec<DynamicImage>> = (0..clips)
            .map(|_| (0..8).map(|_| make_test_image(640, 480)).collect())
            .collect();

        group.bench_with_input(BenchmarkId::new("serial", clips), &batch, |b, batch| {
            b.iter(|| {
                batch
                    .iter()
                    .map(|frames| processor.preprocess_video(frames, &cfg).unwrap())
                    .collect::<Vec<_>>()
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", clips), &batch, |b, batch| {
            b.iter(|| {
                batch
                    .par_iter()
                    .map(|frames| processor.preprocess_video(frames, &cfg).unwrap())
                    .collect::<Vec<_>>()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_clone_overhead, bench_video_clips);
criterion_main!(benches);
