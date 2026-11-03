//! The media pipeline's per-request families: histograms with their own
//! buckets (not summaries), carrying what the record calls say.
//!
//! The recorder is this binary's, installed once for all its tests (see
//! `recorder()` in the group's `mod.rs`).

use std::time::Duration;

use llm_multimodal::{FetchOutcome, FetchReport, FetchSourceKind, Modality};
use smg::observability::metrics::Metrics;

/// The lines of one family.
fn family<'a>(body: &'a str, name: &str) -> Vec<&'a str> {
    body.lines().filter(|line| line.contains(name)).collect()
}

/// The value of the sample whose line carries every one of `labels`.
#[expect(
    clippy::expect_used,
    reason = "a test helper: a missing or non-numeric sample is the test's failure"
)]
fn sample(lines: &[&str], labels: &[&str]) -> f64 {
    let line = lines
        .iter()
        .find(|line| !line.starts_with('#') && labels.iter().all(|label| line.contains(label)));
    assert!(
        line.is_some(),
        "no sample with {labels:?} in:\n{}",
        lines.join("\n")
    );
    line.and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
        .expect("a numeric sample value")
}

#[test]
fn the_media_pipeline_families_render_as_histograms_with_their_values() {
    let handle = crate::metrics::recorder();
    Metrics::record_mm_processing_stage("mm-series-model", "preprocess", 0.004);
    Metrics::record_mm_processing_stage("mm-series-model", "total", 0.3);
    Metrics::record_mm_request_media(
        "mm-series-model",
        Modality::Image,
        3,
        &[("url", 2), ("data_url", 0), ("inline", 1)],
    );
    Metrics::record_mm_fetch(
        "mm-series-model",
        &FetchReport {
            modality: Modality::Image,
            source: FetchSourceKind::Url,
            outcome: FetchOutcome::Ok,
            duration: Duration::from_millis(20),
            bytes: 4096,
        },
    );
    Metrics::record_mm_fetch(
        "mm-series-model",
        &FetchReport {
            modality: Modality::Image,
            source: FetchSourceKind::DataUrl,
            outcome: FetchOutcome::TooLarge,
            duration: Duration::from_millis(1),
            bytes: 0,
        },
    );
    let body = handle.render();

    for name in [
        "smg_mm_processing_seconds",
        "smg_mm_media_per_request",
        "smg_mm_media_fetch_seconds",
    ] {
        let lines = family(&body, name);
        let type_line = format!("# TYPE {name} histogram");
        assert!(
            lines.contains(&type_line.as_str()),
            "{name} is not a histogram:\n{}",
            lines.join("\n")
        );
        assert!(
            !lines.iter().any(|line| line.contains("quantile=")),
            "{name} renders as a summary:\n{}",
            lines.join("\n")
        );
    }

    let stages = family(&body, "smg_mm_processing_seconds");
    let preprocess = ["model=\"mm-series-model\"", "stage=\"preprocess\""];
    assert_eq!(
        sample(
            &stages,
            &["_bucket", preprocess[0], preprocess[1], "le=\"0.005\""]
        ),
        1.0
    );
    assert_eq!(
        sample(
            &stages,
            &["_bucket", preprocess[0], preprocess[1], "le=\"0.0025\""]
        ),
        0.0
    );
    assert_eq!(
        sample(&stages, &["_count", preprocess[0], "stage=\"total\""]),
        1.0
    );

    let per_request = family(&body, "smg_mm_media_per_request");
    let image = ["model=\"mm-series-model\"", "modality=\"image\""];
    assert_eq!(sample(&per_request, &["_sum", image[0], image[1]]), 3.0);
    assert_eq!(
        sample(&per_request, &["_bucket", image[0], image[1], "le=\"3\""]),
        1.0
    );
    assert_eq!(
        sample(&per_request, &["_bucket", image[0], image[1], "le=\"2\""]),
        0.0
    );

    let items = family(&body, "smg_mm_media_items_total");
    assert_eq!(sample(&items, &[image[0], image[1], "source=\"url\""]), 2.0);
    assert_eq!(
        sample(&items, &[image[0], image[1], "source=\"inline\""]),
        1.0
    );
    assert!(
        !items
            .iter()
            .any(|line| line.contains("source=\"data_url\"")),
        "a zero count makes no series:\n{}",
        items.join("\n")
    );

    let fetches = family(&body, "smg_mm_media_fetch_seconds");
    assert_eq!(
        sample(
            &fetches,
            &["_count", image[0], "outcome=\"ok\"", "source=\"url\""]
        ),
        1.0
    );
    assert_eq!(
        sample(
            &fetches,
            &["_bucket", image[0], "outcome=\"ok\"", "le=\"0.025\""]
        ),
        1.0
    );
    assert_eq!(
        sample(
            &fetches,
            &[
                "_count",
                image[0],
                "outcome=\"too_large\"",
                "source=\"data_url\""
            ]
        ),
        1.0
    );

    let bytes = family(&body, "smg_mm_media_fetch_bytes_total");
    assert_eq!(
        sample(&bytes, &[image[0], image[1], "source=\"url\""]),
        4096.0
    );
    assert!(
        !bytes
            .iter()
            .any(|line| line.contains("source=\"data_url\"")),
        "a failed fetch brought no bytes:\n{}",
        bytes.join("\n")
    );
}
