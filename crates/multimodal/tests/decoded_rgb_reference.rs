//! The decoded RGB of images in the sample formats the pipeline converts,
//! pinned to the bytes the serving engine's own loader hands its processor:
//! 16-bit samples narrow as that loader narrows them, and an image that
//! carries transparency is composited over opaque white when it is decoded
//! (dropping the alpha channel would read a transparent pixel as black). The expected bytes come
//! from `scripts/generate_decoded_rgb_fixtures.py` with the pinned Pillow and
//! live in `tests/fixtures/golden/decoded_rgb/manifest.json`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::Path;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use llm_multimodal::types::{ImageDetail, ImageFrame, ImageSize, ImageSource};
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    image: String,
    width: u32,
    height: u32,
    note: String,
    expected_rgb_base64: String,
}

/// The gateway's decode of an inline body: the header's size, then the
/// frame's lazy decode (orientation applied, no long-side cap).
fn decoded_rgb(bytes: &[u8]) -> image::RgbImage {
    let (width, height) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .unwrap()
        .into_dimensions()
        .unwrap();
    let frame = ImageFrame::encoded(
        bytes::Bytes::copy_from_slice(bytes),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        String::new(),
        ImageSize::new(width, height),
        None,
    );
    frame.image().expect("the fixture decodes").to_rgb8()
}

#[test]
fn decoded_rgb_matches_the_reference_loader() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(fixtures.join("golden/decoded_rgb/manifest.json")).unwrap(),
    )
    .unwrap();
    assert!(!manifest.cases.is_empty());
    let mut failures = Vec::new();
    for case in &manifest.cases {
        let bytes = std::fs::read(fixtures.join("images/decoded_rgb").join(&case.image)).unwrap();
        let expected = BASE64_STANDARD.decode(&case.expected_rgb_base64).unwrap();
        let rgb = decoded_rgb(&bytes);
        if (rgb.width(), rgb.height()) != (case.width, case.height) {
            failures.push(format!(
                "{}: decoded {}x{}, reference {}x{} ({})",
                case.name,
                rgb.width(),
                rgb.height(),
                case.width,
                case.height,
                case.note
            ));
            continue;
        }
        let decoded = rgb.as_raw();
        assert_eq!(decoded.len(), expected.len(), "{}: byte count", case.name);
        if let Some(i) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
            let differing = decoded.iter().zip(&expected).filter(|(a, b)| a != b).count();
            let pixel = u32::try_from(i / 3).unwrap();
            failures.push(format!(
                "{}: pixel ({}, {}) channel {}: decoded {}, reference {}; {} of {} bytes differ ({})",
                case.name,
                pixel % case.width,
                pixel / case.width,
                i % 3,
                decoded[i],
                expected[i],
                differing,
                expected.len(),
                case.note
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "decoded RGB differs from the reference loader:\n{}",
        failures.join("\n")
    );
}
