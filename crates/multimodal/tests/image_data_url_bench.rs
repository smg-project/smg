//! Self-contained tracker benchmark; no model, network, or image fixtures needed.
//! Run on base and candidate with the same file and build profile:
//! `cargo test -p llm-multimodal --test image_data_url_bench --release -- --ignored --nocapture`
//! Fixture creation, warmup, and output hashing are outside the measured region.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::{sync::Arc, time::Instant};

use base64::{engine::general_purpose::STANDARD, Engine};
use image::{codecs::jpeg::JpegEncoder, RgbImage};
use llm_multimodal::{
    AsyncMultiModalTracker, MediaConnector, MediaConnectorConfig, MediaContentPart, Modality,
    TrackedMedia,
};

fn fixtures() -> Vec<String> {
    (0..70)
        .map(|index| {
            let mut state = index as u32 + 1;
            let image = RgbImage::from_fn(512, 512, |_, _| {
                image::Rgb(std::array::from_fn(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                }))
            });
            let mut jpeg = Vec::new();
            JpegEncoder::new_with_quality(&mut jpeg, 85)
                .encode_image(&image)
                .expect("encode fixture");
            format!("data:image/jpeg;base64,{}", STANDARD.encode(jpeg))
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "performance benchmark; run explicitly on base and candidate"]
async fn image_data_url_tracker() {
    let urls = fixtures();
    let connector = Arc::new(
        MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
            .expect("connector"),
    );
    for (count, unique) in [(1, 1), (8, 8), (32, 32), (70, 70), (70, 1)] {
        let url_bytes: usize = (0..count).map(|i| urls[i % unique].len()).sum();
        let mut expected_digest = None;
        for iteration in 0..7 {
            let mut tracker = AsyncMultiModalTracker::new(connector.clone());
            let started = Instant::now();
            for index in 0..count {
                tracker
                    .push_part(MediaContentPart::ImageUrl {
                        url: urls[index % unique].clone(),
                        detail: None,
                        uuid: None,
                        max_long_side_pixel: None,
                    })
                    .expect("enqueue image");
            }
            let enqueue_ms = started.elapsed().as_secs_f64() * 1000.0;
            let output = tracker.finalize().await.expect("decode images");
            let total_ms = started.elapsed().as_secs_f64() * 1000.0;
            let images = output.data.get(&Modality::Image).expect("image output");
            assert_eq!(images.len(), count);
            let mut hasher = blake3::Hasher::new();
            for image in images {
                let TrackedMedia::Image(frame) = image else {
                    panic!("expected an image");
                };
                assert_eq!((frame.image.width(), frame.image.height()), (512, 512));
                hasher.update(frame.image.to_rgb8().as_raw());
            }
            let digest = hasher.finalize().to_hex().to_string();
            assert_eq!(
                expected_digest.get_or_insert_with(|| digest.clone()),
                &digest
            );
            if iteration >= 2 {
                println!(
                    "BENCH_JSON {}",
                    serde_json::json!({
                        "images": count, "unique": unique, "url_bytes": url_bytes,
                        "sample": iteration - 2, "enqueue_ms": enqueue_ms,
                        "tracker_ms": total_ms, "digest": digest,
                    })
                );
            }
        }
    }
}
