//! Source geometry, pixel identity and lazy decode regressions.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use crate::{ImageDetail, ImageSource};
/// Exercise mutually exclusive branches, exact-cap equality and unaligned caps.
#[test]
fn branches_on_source_geometry_then_aligns_like_the_model() {
    for ((w, h, cap), expected) in [
        ((2000, 100, 1000), (1008, 56)), // shrink only; do not raise short side
        ((1000, 50, 1000), (2240, 112)), // equality takes the short-side branch
        ((50, 1000, 1000), (112, 2240)),
        ((1000, 500, 1000), (1008, 504)),
        ((100, 50, 1000), (224, 112)),
        ((1000, 500, 1009), (1008, 504)),
        ((70, 112, 1000), (112, 168)),
        ((1125, 500, 1022), (1008, 448)),
        ((2067, 1000, 2030), (2016, 980)),
    ] {
        assert_eq!(
            dimensions(w, h, cap).unwrap(),
            expected,
            "{w}x{h} cap {cap}"
        );
    }
    let error = dimensions(100, 100, 0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "image max_long_side_pixel must be positive, got 0"
    );
    assert_eq!(error.code(), crate::MediaErrorCode::FetchFailed);
}

/// Ensure planned grids bypass legacy short-side staging and area shrinking.
#[test]
fn processor_does_not_repeat_resize_or_shrink_to_legacy_area_budget() {
    use crate::vision::{processors::minimax_m3::MiniMaxM3VisionProcessor, VisionPreProcessor};
    let mut config = crate::PreProcessorConfig::default();
    config
        .extra
        .insert(RESIZED_KEY.into(), serde_json::json!(true));
    let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
    for (w, h) in [(1008, 56), (1008, 1008)] {
        let image = DynamicImage::new_rgb8(w, h);
        let output = processor.preprocess(&[image]).unwrap();
        assert_eq!(
            output.feature_token_counts,
            vec![(w as usize / 28) * (h as usize / 28)]
        );
        assert_eq!(
            processor.calculate_num_tokens(w, h).unwrap(),
            (w as usize / 28) * (h as usize / 28)
        );
    }
}

/// Encode a small source fixture for lazy decode.
fn encoded(w: u32, h: u32) -> Arc<ImageFrame> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    DynamicImage::new_rgb8(w, h)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    Arc::new(ImageFrame::encoded(
        bytes.into_inner().into(),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        "source".into(),
        ImageSize::new(w, h),
        None,
    ))
}

/// Check lazy decode dimensions and cache identities for equivalent grids.
#[test]
fn lazy_pixels_match_header_and_cache_identity() {
    let original = encoded(2000, 100);
    let resized = image(original.clone(), 1000).unwrap();
    assert_eq!(resized.size(), ImageSize::new(1008, 56));
    assert_eq!(
        (
            resized.image().unwrap().width(),
            resized.image().unwrap().height()
        ),
        (1008, 56)
    );
    assert_ne!(resized.hash, original.hash);
    assert_eq!(resized.hash, image(original, 1009).unwrap().hash);
}

/// Reject oversized headers before trying to decode an empty payload.
#[test]
fn rejects_image_budget_before_decoding_pixels() {
    let frame = Arc::new(ImageFrame::encoded(
        bytes::Bytes::new(),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        "source".into(),
        ImageSize::new(4000, 4000),
        None,
    ));
    assert!(matches!(
        image(frame, 4000),
        Err(MediaConnectorError::PixelBudgetExceeded {
            modality: "image",
            ..
        })
    ));
    assert!(budget(IMAGE_MAX_TOTAL_PIXELS, IMAGE_MAX_TOTAL_PIXELS, "image").is_ok());
}

/// Check source planning across shrink and enlargement half-grid ties.
#[test]
fn geometry_matrix_and_ties_obey_the_source_contract() {
    // Calculate the branch's rational scale separately, including caps exactly
    // halfway between grid points after shrinking and sub-patch inputs.
    let round_grid = |side: u64, numerator: u64, denominator: u64| {
        let n = side * numerator;
        let d = denominator * 28;
        let q = n / d;
        let r = n % d;
        let rounded = q + u64::from(r * 2 > d || (r * 2 == d && q % 2 == 1));
        (rounded.max(1) * 28) as u32
    };
    for w in [
        1, 27, 28, 42, 70, 111, 112, 113, 999, 1000, 1008, 1125, 2067, 3584, 8192,
    ] {
        for h in [
            1, 27, 28, 42, 70, 111, 112, 113, 999, 1000, 1008, 1125, 2067, 3584, 8192,
        ] {
            for cap in [
                1, 27, 112, 150, 999, 1000, 1008, 1022, 2016, 2030, 3584, 8192,
            ] {
                let (num, den) = if w.max(h) > cap {
                    (cap, w.max(h))
                } else if w.min(h) < 112 {
                    (112, w.min(h))
                } else {
                    (1, 1)
                };
                let expected = (
                    round_grid(w.into(), num.into(), den.into()),
                    round_grid(h.into(), num.into(), den.into()),
                );
                assert_eq!(dimensions(w, h, cap).unwrap(), expected, "{w}x{h}/{cap}");
            }
        }
    }
    assert!(dimensions(0, 112, 1000).is_err());
    assert!(dimensions(112, 0, 1000).is_err());
    assert!(dimensions(u32::MAX, 1, u32::MAX).is_err());
}

/// Check direct kernel pixels for each supported encoded format and alpha policy.
#[tokio::test]
async fn encoded_pixels_match_direct_resize_after_transparency() {
    use image::{ImageFormat, Rgba, RgbaImage};

    use crate::{
        ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource, TransparencyPolicy,
    };
    let connector = MediaConnector::new(
        crate::test_support::http_client_builder(),
        MediaConnectorConfig::default(),
    )
    .unwrap();
    for format in [ImageFormat::Png, ImageFormat::WebP, ImageFormat::Jpeg] {
        let pixels = DynamicImage::ImageRgba8(RgbaImage::from_fn(400, 50, |x, y| {
            Rgba([x as u8, y as u8, 70, (x + y) as u8])
        }));
        let pixels = if format == ImageFormat::Jpeg {
            DynamicImage::ImageRgb8(pixels.to_rgb8())
        } else {
            pixels
        };
        let mut bytes = std::io::Cursor::new(Vec::new());
        pixels.write_to(&mut bytes, format).unwrap();
        for transparency in [
            TransparencyPolicy::KeepAlpha,
            TransparencyPolicy::CompositeOverWhite,
            TransparencyPolicy::DropAlpha,
        ] {
            let original = connector
                .fetch_image(
                    MediaSource::InlineBytes(bytes.get_ref().clone()),
                    ImageFetchConfig {
                        transparency,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            for cap in [301, 1000] {
                let (w, h) = dimensions(400, 50, cap).unwrap();
                let expected = resize_image(original.image().unwrap().clone(), w, h);
                let actual = image(original.clone(), cap).unwrap();
                assert_eq!(actual.image().unwrap().as_bytes(), expected.as_bytes());
                assert_eq!(actual.transparency(), transparency);
                let decoded = actual.image().unwrap();
                assert_eq!(
                    decoded.color().has_alpha(),
                    original.image().unwrap().color().has_alpha()
                );
                let rgb_reference = resize_bicubic_torch_aa(original.image().unwrap(), w, h);
                assert_eq!(decoded.to_rgb8(), rgb_reference.to_rgb8());
                if decoded.color().has_alpha() {
                    assert!(decoded.to_rgba8().pixels().any(|p| p[3] != 255));
                }
            }
        }
    }
}

/// Duplicate requests share fetching; equivalent grids share pixel-cache identities.
#[tokio::test]
async fn tracker_deduplicates_without_confusing_effective_sizes() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    use crate::{
        AsyncMultiModalTracker, MediaConnector, MediaConnectorConfig, MediaContentPart, Modality,
        TrackedMedia,
    };
    let connector = Arc::new(
        MediaConnector::new(
            crate::test_support::http_client_builder(),
            MediaConnectorConfig::default(),
        )
        .unwrap(),
    );
    let source = encoded(2000, 100);
    let url = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(&source.raw_bytes)
    );
    let mut tracker = AsyncMultiModalTracker::new(connector).with_minimax_image_resize(true);
    for cap in [1000, 1000, 1009, 2016] {
        tracker
            .push_part(MediaContentPart::ImageUrl {
                url: url.clone(),
                detail: None,
                uuid: None,
                max_long_side_pixel: Some(cap),
            })
            .unwrap();
    }
    let mut result = tracker.finalize().await.unwrap();
    let frames: Vec<_> = result
        .data
        .remove(&Modality::Image)
        .unwrap()
        .into_iter()
        .map(|m| match m {
            TrackedMedia::Image(f) => f,
            _ => panic!("image expected"),
        })
        .collect();
    assert!(Arc::ptr_eq(&frames[0], &frames[1]));
    assert_eq!(frames[0].hash, frames[2].hash);
    assert_ne!(frames[0].hash, frames[3].hash);
    assert_eq!(frames[0].size(), ImageSize::new(1008, 56));
    assert_eq!(frames[3].size(), ImageSize::new(2240, 112));
}

/// A custom grid must not silently resize already planned source pixels again.
#[test]
fn source_policy_rejects_an_incompatible_processor_grid() {
    use crate::vision::processors::MiniMaxM3VisionProcessor;
    let mut config = crate::PreProcessorConfig::from_json(r#"{"patch_size":16}"#).unwrap();
    assert!(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).is_ok());
    config.extra.insert(RESIZED_KEY.into(), true.into());
    assert!(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).is_err());
}

/// The exact fixed area is admitted using only headers; one grid step more is refused.
#[test]
fn image_budget_boundary_needs_no_decode() {
    for (side, valid) in [(3584, true), (3612, false)] {
        let frame = Arc::new(ImageFrame::encoded(
            bytes::Bytes::new(),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            "boundary".into(),
            ImageSize::new(side, side),
            None,
        ));
        assert_eq!(image(frame, side).is_ok(), valid);
    }
}

/// Day-0 clamps the short-side raise and keeps engine-specific pixel identities.
#[test]
fn sglang_day0_geometry_and_cache_identity() {
    for ((w, h, cap), expected) in [
        ((1000, 50, 2016), (2016, 112)),
        ((320, 240, 756), (308, 252)),
        ((56, 56, 2016), (112, 112)),
        ((1024, 1024, 2016), (1036, 1036)),
    ] {
        assert_eq!(sglang_dimensions(w, h, cap).unwrap(), expected);
    }
    for cap in [0, 56, 1000, 3612] {
        assert!(sglang_dimensions(512, 512, cap).is_err());
    }
    assert!(sglang_dimensions(201, 1, 2016).is_err());
    let frame = encoded(512, 512);
    let vllm = image(frame.clone(), 2016).unwrap();
    let sglang = image_for_runtime(frame, 2016, true).unwrap();
    assert_eq!(vllm.size(), sglang.size());
    assert_ne!(vllm.hash, sglang.hash);
}
