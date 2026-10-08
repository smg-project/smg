use std::{path::PathBuf, sync::Arc, time::Duration};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use llm_multimodal::{
    AsyncMultiModalTracker, AudioSource, ImageFetchConfig, ImageSource, MediaConnector,
    MediaConnectorConfig, MediaContentPart, MediaSource, Modality,
};
use reqwest::Client;
use tempfile::tempdir;

const TINY_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

#[expect(
    clippy::expect_used,
    reason = "test helper: panic on failure is intentional"
)]
fn tiny_png_bytes() -> Vec<u8> {
    BASE64_STANDARD
        .decode(TINY_PNG_BASE64)
        .expect("decode tiny png fixture")
}

fn wav_i16_mono(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_bytes = samples.len() as u32 * 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[expect(
    clippy::expect_used,
    reason = "test helper: panic on failure is intentional"
)]
fn test_connector(allowed_path: Option<PathBuf>) -> MediaConnector {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("client");
    MediaConnector::new(
        client,
        MediaConnectorConfig {
            allowed_domains: None,
            allowed_local_media_path: allowed_path,
            fetch_timeout: Duration::from_secs(5),
        },
    )
    .expect("media connector")
}

#[tokio::test]
async fn fetch_image_from_inline_bytes() {
    let connector = test_connector(None);
    let bytes = tiny_png_bytes();
    let frame = connector
        .fetch_image(
            MediaSource::InlineBytes(bytes.clone()),
            ImageFetchConfig::default(),
        )
        .await
        .expect("inline image");
    assert_eq!(frame.size().width, 1);
    assert_eq!(frame.size().height, 1);
    assert_eq!(frame.raw_bytes(), bytes.as_slice());
}

#[tokio::test]
async fn fetch_image_from_data_url() {
    let connector = test_connector(None);
    let bytes = tiny_png_bytes();
    let data_url = format!(
        "data:image/png;base64,{}",
        BASE64_STANDARD.encode(bytes.clone())
    );

    let frame = connector
        .fetch_image(MediaSource::DataUrl(data_url), ImageFetchConfig::default())
        .await
        .expect("data url");
    assert_eq!(frame.size().width, 1);
    assert_eq!(frame.raw_bytes(), bytes.as_slice());
}

#[tokio::test]
async fn fetch_image_from_file() {
    let tmp = tempdir().expect("tempdir");
    let allowed_root = std::fs::canonicalize(tmp.path()).expect("canonical tmp path");
    let file_path = allowed_root.join("tiny.png");
    std::fs::write(&file_path, tiny_png_bytes()).expect("write png");

    let connector = test_connector(Some(allowed_root));
    let frame = connector
        .fetch_image(
            MediaSource::File(file_path.clone()),
            ImageFetchConfig::default(),
        )
        .await
        .expect("file png");
    assert_eq!(frame.size().width, 1);
    let expected = std::fs::canonicalize(&file_path).expect("canonical path");
    match frame.source() {
        ImageSource::File { path } => assert_eq!(path, &expected),
        other => panic!("expected file source, got {other:?}"),
    }
}

#[tokio::test]
async fn fetch_audio_from_inline_bytes_decodes_samples() {
    let connector = test_connector(None);
    let bytes = wav_i16_mono(16_000, &[0, 16_384, -16_384]);
    let clip = connector
        .fetch_audio(MediaSource::InlineBytes(bytes.clone()))
        .await
        .expect("inline audio");

    assert_eq!(clip.raw_bytes(), bytes.as_slice());
    assert_eq!(clip.decoded().sample_rate, 16_000);
    assert_eq!(clip.decoded().samples.len(), 3);
    assert!(clip.decoded().samples[0].abs() < 1e-6);
    assert!((clip.decoded().samples[1] - 0.5).abs() < 1e-4);
    assert!((clip.decoded().samples[2] + 0.5).abs() < 1e-4);
    assert!(matches!(clip.source(), AudioSource::InlineBytes));
}

#[tokio::test]
async fn tracker_fetches_images_and_records_uuids() {
    let connector = Arc::new(test_connector(None));
    let mut tracker = AsyncMultiModalTracker::new(connector);

    tracker
        .push_part(MediaContentPart::Text {
            text: "before".into(),
        })
        .expect("text part");
    tracker
        .push_part(MediaContentPart::ImageData {
            data: tiny_png_bytes(),
            mime_type: Some("image/png".into()),
            uuid: Some("img-1".into()),
            detail: None,
        })
        .expect("image part");
    tracker
        .push_part(MediaContentPart::Text {
            text: "after".into(),
        })
        .expect("text part");

    let output = tracker.finalize().await.expect("tracker finalize");

    let images = output.data.get(&Modality::Image).expect("image entry");
    assert_eq!(images.len(), 1);

    let uuids = output.uuids.get(&Modality::Image).expect("uuid entry");
    assert_eq!(uuids, &vec![Some("img-1".into())]);
}
