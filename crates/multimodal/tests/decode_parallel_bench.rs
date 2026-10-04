//! Microbench: decoding a batch of photos one after another versus
//! `ImageFrame::decode_all` on the preprocessing pool.
//!
//! Run:
//!   REAL_JPEG=/path/x.jpg cargo test -p llm-multimodal --release \
//!     --test decode_parallel_bench -- --ignored --nocapture
#![allow(clippy::expect_used, clippy::print_stderr)]

use std::{sync::Arc, time::Instant};

use llm_multimodal::{
    ImageDetail, ImageFetchConfig, ImageFrame, MediaConnector, MediaConnectorConfig, MediaSource,
};

async fn frames(connector: &MediaConnector, bytes: &[u8], count: usize) -> Vec<Arc<ImageFrame>> {
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        frames.push(
            connector
                .fetch_image(
                    MediaSource::InlineBytes(bytes.to_vec()),
                    ImageFetchConfig {
                        detail: ImageDetail::default(),
                        max_long_side_pixel: None,
                    },
                )
                .await
                .expect("frame from bytes"),
        );
    }
    frames
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "perf microbench; needs REAL_JPEG"]
async fn bench_decode_all() {
    let path = std::env::var("REAL_JPEG").expect("set REAL_JPEG to a JPEG path");
    let bytes = std::fs::read(&path).expect("read jpeg");
    let connector = MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
        .expect("connector");
    // Warm up the decoder and the pool.
    ImageFrame::decode_all(&frames(&connector, &bytes, 2).await).expect("warm-up");

    for count in [1usize, 4, 8] {
        let mut serial = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..3 {
            let batch = frames(&connector, &bytes, count).await;
            let started = Instant::now();
            for frame in &batch {
                frame.image().expect("decode");
            }
            serial.push(started.elapsed().as_secs_f64() * 1000.0);

            let batch = frames(&connector, &bytes, count).await;
            let started = Instant::now();
            ImageFrame::decode_all(&batch).expect("decode_all");
            parallel.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        serial.sort_by(f64::total_cmp);
        parallel.sort_by(f64::total_cmp);
        eprintln!(
            "{count} images: one after another {:.1} ms, decode_all {:.1} ms (medians of 3)",
            serial[1], parallel[1]
        );
    }
}
