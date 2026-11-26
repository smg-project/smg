//! Source geometry, pixel identity and lazy decode regressions.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use crate::{test_support::http_client_builder, ImageDetail, ImageSource};
/// The aligned size against the model processor's arithmetic, row for row:
/// both floors, a float product that lands above the floor, the grid rounded
/// up, a cap rounded down to the grid and the scale back under it.
#[test]
fn aligned_sizes_match_the_model_processor() {
    for ((w, h, cap), expected) in [
        // The default cap: a floor, the grid alone, the scale back.
        ((1, 1, 2016), (168, 168)),
        ((500, 400, 2016), (504, 420)),
        ((512, 512, 2016), (532, 532)),
        ((2000, 2000, 2016), (2016, 2016)),
        ((2000, 1334, 2016), (2016, 1344)),
        ((4284, 5712, 2016), (1512, 2016)),
        // The area floor, then the short-side floor on what it left.
        ((224, 112, 2016), (336, 168)),
        ((167, 168, 2016), (168, 196)),
        // 153 * (168.0 / 153) lands above 168: one grid step more.
        ((153, 153, 2016), (196, 196)),
        // A floor first, then the cap on the enlarged long side.
        ((1000, 50, 2016), (2016, 112)),
        ((50, 1000, 1000), (56, 980)),
        // A cap off the grid is rounded down; a cap on it stands.
        ((1000, 500, 1000), (980, 504)),
        ((1000, 500, 1009), (1008, 504)),
        // 2016 * 2016 / 2044 = 1988.38 is cut to 1988, on the grid already.
        ((2017, 2016, 2016), (2016, 1988)),
        ((3600, 3600, 3600), (3584, 3584)),
        // A side the scale takes to nothing keeps one cell.
        ((33600, 168, 140), (140, 28)),
        ((100, 100, 27), (28, 28)),
    ] {
        assert_eq!(
            dimensions(w, h, cap).unwrap(),
            expected,
            "{w}x{h} cap {cap}"
        );
    }
    assert!(dimensions(0, 112, 1000).is_err());
    assert!(dimensions(112, 0, 1000).is_err());
    assert!(dimensions(u32::MAX, 1, u32::MAX).is_err());
    let error = dimensions(100, 100, 0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "image max_long_side_pixel must be positive, got 0"
    );
    assert_eq!(error.code(), crate::MediaErrorCode::InvalidRequest);
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
        let output = processor.preprocess(&[&image]).unwrap();
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

/// No aligned size is planned over the processor's 200:1 aspect limit, nor
/// for a source over it: the nearest-grid size stands, and the processor
/// serves or refuses it as before.
#[test]
fn a_planned_size_stays_within_the_processors_aspect_limit() {
    use crate::vision::{processors::minimax_m3::MiniMaxM3VisionProcessor, VisionPreProcessor};
    let mut config = crate::PreProcessorConfig::default();
    config
        .extra
        .insert(RESIZED_KEY.into(), serde_json::json!(true));
    let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
    for ((w, h, cap), planned) in [
        // Aligned 5628x28 is 201:1; the nearest-grid size keeps 199:1.
        ((5572, 28, 5640), (22_288, 112)),
        ((28, 5572, 5640), (112, 22_288)),
        ((4095, 21, 5628), (21_840, 112)),
        // Aligned 5600x28 is 200:1 and stands.
        ((5600, 28, 5600), (5600, 28)),
    ] {
        let size = image(encoded(w, h), cap).unwrap().size();
        assert_eq!((size.width, size.height), planned, "{w}x{h} cap {cap}");
        let output = processor
            .preprocess(&[&DynamicImage::new_rgb8(size.width, size.height)])
            .unwrap_or_else(|err| panic!("{w}x{h} cap {cap}: {err}"));
        assert_eq!(
            output.feature_token_counts,
            vec![(size.width as usize / 28) * (size.height as usize / 28)]
        );
    }
    // The model processor sizes no source over 200:1: the plan is the
    // nearest-grid one, which the processor refuses as before.
    let size = image(encoded(1, 499), 2016).unwrap().size();
    assert_eq!((size.width, size.height), (112, 55_888));
    assert!(processor.calculate_num_tokens(112, 55_888).is_err());
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
    assert_eq!(resized.size(), ImageSize::new(980, 56));
    assert_eq!(
        (
            resized.image().unwrap().width(),
            resized.image().unwrap().height()
        ),
        (980, 56)
    );
    assert_ne!(resized.hash, original.hash);
    assert_eq!(resized.hash, image(original, 1007).unwrap().hash);
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

/// Check source planning across both rules. The aligned size, where the
/// budget and the aspect limit admit it, is on the grid, holds one cell at
/// least and stays under the cap rounded down to the grid. Everywhere else the
/// nearest-grid size stands to the pixel, half-grid ties included.
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
    const SIDES: [u32; 17] = [
        1, 27, 28, 42, 70, 111, 112, 113, 167, 168, 999, 1000, 1008, 1125, 2067, 3584, 8192,
    ];
    let mut nearest = 0;
    for w in SIDES {
        for h in SIDES {
            // 3654 and 3682 are half-grid ties over the budget: one rounds
            // down to an even cell count, one up.
            for cap in [
                1, 27, 112, 150, 999, 1000, 1008, 1022, 2016, 2030, 3584, 3640, 3654, 3682, 4004,
                8192,
            ] {
                let sides = dimensions(w, h, cap).unwrap();
                let (aw, ah) = aligned(w, h, cap);
                if u64::from(aw) * u64::from(ah) <= IMAGE_MAX_TOTAL_PIXELS
                    && w.max(h) <= 200 * w.min(h)
                    && aw.max(ah) <= 200 * aw.min(ah)
                {
                    assert_eq!(sides, (aw, ah), "{w}x{h}/{cap}");
                    let limit = (cap - cap % 28).max(28);
                    for side in [sides.0, sides.1] {
                        assert!(
                            side % 28 == 0 && (28..=limit).contains(&side),
                            "{w}x{h}/{cap}: {sides:?}"
                        );
                    }
                    continue;
                }
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
                assert_eq!(sides, expected, "{w}x{h}/{cap}");
                nearest += 1;
            }
        }
    }
    // 288 sources are over the aspect limit; nine sizes pass the budget.
    assert_eq!(nearest, 297);
}

/// Check direct kernel pixels for each supported encoded format and alpha policy.
#[tokio::test]
async fn encoded_pixels_match_direct_resize_after_transparency() {
    use image::{ImageFormat, Rgba, RgbaImage};

    use crate::{
        ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource, TransparencyPolicy,
    };
    let connector =
        MediaConnector::new(http_client_builder(), MediaConnectorConfig::default()).unwrap();
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
        MediaConnector::new(http_client_builder(), MediaConnectorConfig::default()).unwrap(),
    );
    let source = encoded(2000, 100);
    let url = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(&source.raw_bytes)
    );
    let mut tracker = AsyncMultiModalTracker::new(connector).with_minimax_image_resize(true);
    for cap in [1000, 1000, 1007, 2016] {
        tracker
            .push_part(MediaContentPart::ImageUrl {
                url: url.clone().into(),
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
    assert_eq!(frames[0].size(), ImageSize::new(980, 56));
    assert_eq!(frames[3].size(), ImageSize::new(2016, 112));
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

/// The exact fixed area is admitted using only headers; one grid step more is
/// refused. Where the aligned size passes the budget the nearest-grid size
/// stands, and a source that fits the budget with it is admitted.
#[test]
fn image_budget_boundary_needs_no_decode() {
    for ((w, h, cap), planned) in [
        ((3584, 3584, 3584), Some((3584, 3584))),
        ((3612, 3612, 3612), None),
        // The cap is rounded down to the grid, so the aligned size fits.
        ((3600, 3600, 3600), Some((3584, 3584))),
        // Aligned 3612x3612, 3556x3640 and 168x78820 pass the budget.
        ((3590, 3590, 4000), Some((3584, 3584))),
        ((3557, 3656, 3640), Some((3528, 3640))),
        ((1, 499, 100_000), Some((112, 55_888))),
    ] {
        let frame = Arc::new(ImageFrame::encoded(
            bytes::Bytes::new(),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            "boundary".into(),
            ImageSize::new(w, h),
            None,
        ));
        assert_eq!(
            image(frame, cap).ok().map(|frame| frame.size()),
            planned.map(|(w, h)| ImageSize::new(w, h)),
            "{w}x{h} cap {cap}"
        );
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
    let frame = encoded(1008, 1008);
    let vllm = image(frame.clone(), 2016).unwrap();
    let sglang = image_for_runtime(frame, 2016, true).unwrap();
    assert_eq!(vllm.size(), sglang.size());
    assert_ne!(vllm.hash, sglang.hash);
}

/// Preserve vLLM's source geometry, exact cache key and Torch-AA pixels.
#[test]
fn vllm_source_contract_is_unchanged_by_sglang_opt_in() {
    let frame = encoded(1000, 50);
    let resized = image(frame.clone(), 1000).unwrap();
    assert_eq!(resized.size(), ImageSize::new(980, 56));
    let mut hash = blake3::Hasher::new();
    hash.update(b"minimax-m3-source-resize-v1");
    hash.update(frame.hash.as_bytes());
    hash.update(&980u32.to_le_bytes());
    hash.update(&56u32.to_le_bytes());
    assert_eq!(resized.hash, hash.finalize().to_hex().to_string());
    assert_eq!(
        resized.image().unwrap().to_rgb8(),
        resize_image(frame.image().unwrap().clone(), 980, 56).to_rgb8()
    );
    let error = sglang_dimensions(512, 512, 1000).unwrap_err();
    assert!(error
        .to_string()
        .contains("multiple of 28 within 112..=3584"));
}

/// The frame holds the resized image, so the decoder's buffer is back in the
/// pool when the resize returns, with or without an alpha channel; the pixels
/// are the same as from a buffer the pool does not keep. A frame already at
/// its planned size keeps its buffer.
#[test]
fn a_resized_frame_returns_the_decoders_buffer() {
    static MINE: buffer_pool::BufferPool = buffer_pool::BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let pooled = |pixels: &[u8]| {
            let mut buffer = buffer_pool::take(pixels.len());
            buffer.copy_from_slice(pixels);
            buffer
        };
        let rgb = DynamicImage::ImageRgb8(image::RgbImage::from_fn(320, 320, |x, y| {
            image::Rgb([x as u8, y as u8, (x ^ y) as u8])
        }));
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(320, 320, |x, y| {
            image::Rgba([x as u8, y as u8, (x ^ y) as u8, (x + y) as u8])
        }));
        let expected_rgb = resize_image(rgb.clone(), 160, 160);
        let expected_rgba = resize_image(rgba.clone(), 160, 160);
        assert_eq!(MINE.held(), 0);

        let frame = image::RgbImage::from_raw(320, 320, pooled(rgb.as_bytes())).unwrap();
        let resized = resize_image(DynamicImage::ImageRgb8(frame), 160, 160);
        assert!(resized.as_bytes() == expected_rgb.as_bytes());
        assert_eq!(MINE.held(), 512 << 10);

        let frame = image::RgbaImage::from_raw(320, 320, pooled(rgba.as_bytes())).unwrap();
        let resized = resize_image(DynamicImage::ImageRgba8(frame), 160, 160);
        assert!(resized.color().has_alpha());
        assert!(resized.as_bytes() == expected_rgba.as_bytes());
        assert_eq!(MINE.held(), 512 << 10);

        let frame = image::RgbImage::from_raw(320, 320, pooled(rgb.as_bytes())).unwrap();
        let same = resize_image(DynamicImage::ImageRgb8(frame), 320, 320);
        assert!(same.as_bytes() == rgb.as_bytes());
        assert_eq!(MINE.held(), 0);
    });
}
