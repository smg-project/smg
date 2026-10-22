//! The decoded RGB of a frame under each transparency policy, pinned to the
//! bytes of the two reference loader behaviours (Pillow, through
//! `scripts/generate_decoded_rgb_fixtures.py`): `CompositeOverWhite` composites
//! an image that carries transparency over opaque white (a paste through the
//! alpha mask); `DropAlpha` keeps every pixel's stored colour
//! (`convert("RGB")`); `KeepAlpha` decodes the file as it is. The expected
//! bytes live in `tests/fixtures/golden/decoded_rgb/manifest.json`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::Path;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use llm_multimodal::{
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource},
    TransparencyPolicy,
};
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
    /// Whether the reference loader sees transparency in the file (an alpha
    /// channel, an `LA`/`PA` mode or a colour key); the decoder here expands
    /// a colour key into an alpha channel, so the two agree.
    transparent: bool,
    expected_rgb_base64: String,
    expected_dropped_rgb_base64: String,
}

/// The gateway's frame for an inline body under `policy`: the header's size,
/// the lazy decode (orientation applied, no long-side cap).
fn frame(bytes: &[u8], policy: TransparencyPolicy) -> ImageFrame {
    let (width, height) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .unwrap()
        .into_dimensions()
        .unwrap();
    ImageFrame::encoded(
        bytes::Bytes::copy_from_slice(bytes),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        String::new(),
        ImageSize::new(width, height),
        None,
    )
    .with_transparency(policy)
}

fn cases() -> Vec<(Case, Vec<u8>)> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(fixtures.join("golden/decoded_rgb/manifest.json")).unwrap(),
    )
    .unwrap();
    assert!(!manifest.cases.is_empty());
    manifest
        .cases
        .into_iter()
        .map(|case| {
            let bytes =
                std::fs::read(fixtures.join("images/decoded_rgb").join(&case.image)).unwrap();
            (case, bytes)
        })
        .collect()
}

/// Every case decoded under `policy`, against the expected bytes `expected` picks.
fn check(policy: TransparencyPolicy, expected: impl Fn(&Case) -> &str) {
    let mut failures = Vec::new();
    for (case, bytes) in cases() {
        let want = BASE64_STANDARD.decode(expected(&case)).unwrap();
        let rgb = frame(&bytes, policy).image().unwrap().to_rgb8();
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
        let got = rgb.as_raw();
        assert_eq!(got.len(), want.len(), "{}: byte count", case.name);
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            let pixel = u32::try_from(i / 3).unwrap();
            failures.push(format!(
                "{}: pixel ({}, {}) channel {}: got {}, reference {}; {} of {} bytes differ ({})",
                case.name,
                pixel % case.width,
                pixel / case.width,
                i % 3,
                got[i],
                want[i],
                differing,
                want.len(),
                case.note
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{policy:?} differs from the reference loader:\n{}",
        failures.join("\n")
    );
}

#[test]
fn composited_over_white_matches_the_reference_loader() {
    check(TransparencyPolicy::CompositeOverWhite, |case| {
        &case.expected_rgb_base64
    });
}

#[test]
fn dropped_alpha_keeps_the_stored_colour() {
    check(TransparencyPolicy::DropAlpha, |case| {
        &case.expected_dropped_rgb_base64
    });
}

#[test]
fn keep_alpha_decodes_the_file_as_it_is() {
    for (case, bytes) in cases() {
        let kept = frame(&bytes, TransparencyPolicy::KeepAlpha)
            .image()
            .unwrap()
            .clone();
        assert_eq!(
            kept.color().has_alpha(),
            case.transparent,
            "{}: the alpha channel is there exactly when the file has transparency",
            case.name
        );
        let dropped = frame(&bytes, TransparencyPolicy::DropAlpha)
            .image()
            .unwrap()
            .to_rgb8();
        assert_eq!(
            kept.to_rgb8().as_raw(),
            dropped.as_raw(),
            "{}: the stored colour",
            case.name
        );
    }
}

#[test]
fn with_transparency_sets_the_policy_the_frame_decodes_under() {
    let (_, bytes) = cases().into_iter().next().unwrap();
    for policy in [
        TransparencyPolicy::KeepAlpha,
        TransparencyPolicy::CompositeOverWhite,
        TransparencyPolicy::DropAlpha,
    ] {
        assert_eq!(frame(&bytes, policy).transparency(), policy);
    }
}
