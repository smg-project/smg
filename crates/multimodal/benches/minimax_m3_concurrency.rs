//! Closed-loop CPU request benchmark, emitting one JSON record per measured arm.
//! Run with SMG_M3_CONCURRENCY_BENCH=1 SMG_MM_THREADS=4 cargo bench
//! -p llm-multimodal --bench minimax_m3_concurrency.
//! Admission latency includes blocking-pool queueing, but excludes waiting for
//! a concurrency slot. This is not HTTP latency or an open-loop overload test.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::{sync::Arc, time::Instant};

use base64::{engine::general_purpose::STANDARD, Engine};
use futures::{stream, StreamExt};
use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    encoder_inputs::ModelSpecificValue,
    vision::{
        preprocessor_config::PreProcessorConfig, processors::MiniMaxM3VisionProcessor,
        VisionPreProcessor,
    },
    AsyncMultiModalTracker, MediaConnector, MediaConnectorConfig, MediaContentPart, Modality,
    TrackedMedia,
};
use rustls::crypto::ring;

struct Fixture {
    url: String,
    cap: u32,
    grid: [i64; 3],
}

/// Encode deterministic fixtures outside all timed regions.
fn fixture(w: u32, h: u32, cap: u32, grid: [i64; 3]) -> Fixture {
    let image = DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    }));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Jpeg)
        .unwrap();
    Fixture {
        url: format!(
            "data:image/jpeg;base64,{}",
            STANDARD.encode(encoded.get_ref())
        ),
        cap,
        grid,
    }
}

/// Execute fresh fetch/decode/resize/tensor requests with bounded admission.
async fn run(
    connector: &Arc<MediaConnector>,
    processor: &Arc<MiniMaxM3VisionProcessor>,
    fixtures: &[Fixture],
    direct: bool,
    concurrency: usize,
    count: usize,
) -> Vec<f64> {
    stream::iter(0..count)
        .map(|i| {
            let fixture = &fixtures[i % fixtures.len()];
            let connector = connector.clone();
            let processor = processor.clone();
            async move {
                let start = Instant::now();
                let mut tracker =
                    AsyncMultiModalTracker::new(connector).with_minimax_image_resize(direct);
                tracker
                    .push_part(MediaContentPart::ImageUrl {
                        url: fixture.url.clone(),
                        detail: None,
                        uuid: None,
                        max_long_side_pixel: Some(fixture.cap),
                    })
                    .unwrap();
                let mut fetched = tracker.finalize().await.unwrap();
                let frame = fetched
                    .data
                    .remove(&Modality::Image)
                    .unwrap()
                    .into_iter()
                    .find_map(|media| {
                        if let TrackedMedia::Image(frame) = media {
                            Some(frame)
                        } else {
                            None
                        }
                    })
                    .unwrap();
                let output = tokio::task::spawn_blocking(move || {
                    processor
                        .preprocess(std::slice::from_ref(frame.image().unwrap()))
                        .unwrap()
                })
                .await
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    output.feature_token_counts,
                    vec![(fixture.grid[1] * fixture.grid[2] / 4) as usize]
                );
                assert!(matches!(output.model_specific.get("image_grid_thw"),
                Some(ModelSpecificValue::IntTensor { data, shape })
                if data.as_slice() == fixture.grid && shape.as_slice() == [1, 3]));
                elapsed
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await
}

/// Install the TLS provider before constructing the benchmark connector.
#[expect(
    clippy::disallowed_methods,
    reason = "benchmark client installs TLS provider first"
)]
fn connector() -> Arc<MediaConnector> {
    let _ = ring::default_provider().install_default();
    Arc::new(
        MediaConnector::new(reqwest::Client::builder(), MediaConnectorConfig::default()).unwrap(),
    )
}

/// Alternate arm order across three rounds; retain every admitted request latency.
#[expect(
    clippy::print_stdout,
    reason = "benchmark emits machine-readable measurements"
)]
fn main() {
    if std::env::var("SMG_M3_CONCURRENCY_BENCH").as_deref() != Ok("1") {
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let connector = connector();
    let processors = [false, true].map(|direct| {
        let mut config = PreProcessorConfig::from_json(
            r#"{"do_resize":true,"do_normalize":true,"max_pixels":12845056}"#,
        )
        .unwrap();
        config.extra.insert(
            llm_multimodal::minimax_resize::RESIZED_KEY.into(),
            direct.into(),
        );
        Arc::new(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap())
    });
    let fixtures = [
        fixture(224, 224, 2016, [1, 16, 16]),
        fixture(1000, 500, 1008, [1, 36, 72]),
        fixture(1920, 1080, 1008, [1, 40, 72]),
        fixture(3840, 2160, 2016, [1, 80, 144]),
    ];
    for round in 0..3 {
        for (name, inputs) in [
            ("1080p", &fixtures[2..3]),
            ("4k", &fixtures[3..4]),
            ("mixed", &fixtures[..]),
        ] {
            for concurrency in [1, 2, 4, 8, 16] {
                for index in [round % 2, 1 - round % 2] {
                    let direct = index == 1;
                    runtime.block_on(run(
                        &connector,
                        &processors[index],
                        inputs,
                        direct,
                        concurrency,
                        concurrency.max(4),
                    ));
                    let start = Instant::now();
                    let latencies = runtime.block_on(run(
                        &connector,
                        &processors[index],
                        inputs,
                        direct,
                        concurrency,
                        64,
                    ));
                    let seconds = start.elapsed().as_secs_f64();
                    println!(
                        "{}",
                        serde_json::json!({"round":round+1,"case":name,"concurrency":concurrency,
                        "arm":if direct {"direct"} else {"legacy"},"requests":64,"seconds":seconds,
                        "requests_per_second":64.0/seconds,"latency_ms":latencies})
                    );
                }
            }
        }
    }
}
