//! Benchmark: MiniMax-M3 image and video preprocessing.
//!
//! Run:  cargo bench -p llm-multimodal --bench minimax_m3_preprocess

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::{
    preprocessor_config::PreProcessorConfig, processors::MiniMaxM3VisionProcessor,
    VideoClipOptions, VisionPreProcessor,
};
use rayon::prelude::*;
use rustls::crypto::ring;

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
    let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config()).unwrap();

    let mut group = c.benchmark_group("m3_image_batch");
    for &count in &[1usize, 4, 20] {
        let images: Vec<DynamicImage> = (0..count).map(|_| make_test_image(1024, 768)).collect();

        group.bench_with_input(
            BenchmarkId::new("preprocess_only", count),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("clone_then_preprocess", count),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let copies: Vec<DynamicImage> = imgs.to_vec();
                    processor.preprocess(&copies).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Several video clips in one request: one at a time versus all at once.
fn bench_video_clips(c: &mut Criterion) {
    let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config()).unwrap();
    let clip = VideoClipOptions::sampled_at(2.0);

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
                    .map(|frames| processor.preprocess_video(frames, &clip).unwrap())
                    .collect::<Vec<_>>()
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", clips), &batch, |b, batch| {
            b.iter(|| {
                batch
                    .par_iter()
                    .map(|frames| processor.preprocess_video(frames, &clip).unwrap())
                    .collect::<Vec<_>>()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_clone_overhead,
    bench_video_clips,
    bench_source_resize
);
criterion_main!(benches);

/// Encoded-image fetching, lazy decode, resize, normalization and patchification.
/// Both arms use the same large checkpoint budget; the old connector and processor
/// remain available so the baseline is executable alongside the source policy.
fn bench_source_resize(c: &mut Criterion) {
    use std::{hint::black_box, sync::Arc, time::Duration};

    use base64::{engine::general_purpose::STANDARD, Engine};
    use llm_multimodal::{
        encoder_inputs::ModelSpecificValue, AsyncMultiModalTracker, MediaConnector,
        MediaConnectorConfig, MediaContentPart, Modality, TrackedMedia,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let connector = Arc::new(
        MediaConnector::new(http_client_builder(), MediaConnectorConfig::default()).unwrap(),
    );
    let mut group = c.benchmark_group("m3_source_resize");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for (w, h, cap, count, expected_grids) in [
        (224, 224, 2016, 1, [[16, 16], [16, 16]]),
        (1000, 500, 1008, 1, [[36, 72], [36, 72]]),
        (1920, 1080, 1008, 1, [[40, 72], [40, 72]]),
        (3840, 2160, 2016, 1, [[80, 144], [80, 144]]),
        (1000, 50, 1008, 1, [[8, 160], [8, 160]]),
        (2000, 100, 1008, 1, [[8, 162], [4, 72]]),
        (1920, 1080, 1008, 4, [[40, 72], [40, 72]]),
        (3840, 2160, 2016, 4, [[80, 144], [80, 144]]),
    ] {
        let mut encoded = std::io::Cursor::new(Vec::new());
        make_test_image(w, h)
            .write_to(&mut encoded, image::ImageFormat::Jpeg)
            .unwrap();
        let url = format!(
            "data:image/jpeg;base64,{}",
            STANDARD.encode(encoded.get_ref())
        );
        for source_policy in [false, true] {
            let mut cfg = PreProcessorConfig {
                max_pixels: Some(12_845_056),
                ..config()
            };
            cfg.extra.insert(
                llm_multimodal::minimax_resize::RESIZED_KEY.into(),
                source_policy.into(),
            );
            let processor =
                Arc::new(MiniMaxM3VisionProcessor::from_preprocessor_config(&cfg).unwrap());
            let run = || {
                runtime.block_on(async {
                    let jobs = (0..count).map(|_| {
                        let connector = connector.clone();
                        let url = url.clone();
                        let processor = processor.clone();
                        async move {
                            let mut tracker = AsyncMultiModalTracker::new(connector)
                                .with_minimax_image_resize(source_policy);
                            tracker
                                .push_part(MediaContentPart::ImageUrl {
                                    url,
                                    detail: None,
                                    uuid: None,
                                    max_long_side_pixel: Some(cap),
                                })
                                .unwrap();
                            let mut fetched = tracker.finalize().await.unwrap();
                            let frame = fetched
                                .data
                                .remove(&Modality::Image)
                                .unwrap()
                                .into_iter()
                                .find_map(|media| match media {
                                    TrackedMedia::Image(frame) => Some(frame),
                                    _ => None,
                                })
                                .expect("image fixture");
                            tokio::task::spawn_blocking(move || {
                                let pixels = frame.image().unwrap();
                                processor.preprocess(std::slice::from_ref(pixels)).unwrap()
                            })
                            .await
                            .unwrap()
                        }
                    });
                    futures::future::join_all(jobs).await
                })
            };
            // Pin token counts as well as timing: thin images deliberately differ
            // because the mutually exclusive source branches replace two stages.
            let [grid_h, grid_w] = expected_grids[usize::from(source_policy)];
            for out in run() {
                assert_eq!(out.feature_token_counts, vec![grid_h * grid_w / 4]);
                assert!(matches!(out.model_specific.get("image_grid_thw"),
                    Some(ModelSpecificValue::IntTensor { data, shape })
                    if data.as_slice() == [1, grid_h as i64, grid_w as i64] && shape.as_slice() == [1, 3]));
            }
            let label = format!(
                "{}_{w}x{h}_cap{cap}_c{count}",
                if source_policy { "direct" } else { "legacy" }
            );
            group.bench_function(label, |b| b.iter(|| black_box(run())));
        }
    }
    group.finish();
}

/// Install the TLS provider before building the benchmark's media client.
#[expect(
    clippy::disallowed_methods,
    reason = "benchmark client factory installs the TLS provider first"
)]
fn http_client_builder() -> reqwest::ClientBuilder {
    let _ = ring::default_provider().install_default();
    reqwest::Client::builder()
}
