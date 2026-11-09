use super::super::*;

fn image(width: u32, height: u32) -> image::DynamicImage {
    image::DynamicImage::new_rgb8(width, height)
}

#[test]
fn accepts_multiples_of_the_patch_factor() {
    for value in [28, 252, 504, 1008] {
        assert!(validate_max_long_side_pixel(Some(value)).is_ok(), "{value}");
    }
    // Absent is always fine.
    assert!(validate_max_long_side_pixel(None).is_ok());
}

#[test]
fn rejects_zero_and_non_multiples() {
    // The tiers the M3 contract suite exercises as invalid.
    for value in [0, 100, 251, 1009] {
        let err = validate_max_long_side_pixel(Some(value)).unwrap_err();
        assert!(
            matches!(err, MediaConnectorError::InvalidMaxLongSidePixel { .. }),
            "expected rejection for {value}, got {err:?}"
        );
    }
}

#[test]
fn caps_the_long_side_preserving_aspect_ratio() {
    // 5000x3000 is the contract suite's stress image.
    let capped = apply_max_long_side_pixel(image(5000, 3000), Some(504));
    assert_eq!(capped.width(), 504);
    // 3000/5000 * 504 = 302.4, floor-ish via the resize box fit.
    assert!(
        (302..=303).contains(&capped.height()),
        "height {} should preserve the 5:3 ratio",
        capped.height()
    );
}

#[test]
fn caps_the_long_side_when_portrait() {
    let capped = apply_max_long_side_pixel(image(3000, 5000), Some(504));
    assert_eq!(capped.height(), 504);
    assert!((302..=303).contains(&capped.width()));
}

#[test]
fn smaller_images_are_left_untouched() {
    // The cap only ever removes resolution; it must not upscale.
    let original = apply_max_long_side_pixel(image(200, 100), Some(1008));
    assert_eq!((original.width(), original.height()), (200, 100));
}

#[test]
fn a_fetched_frame_carries_its_policy_in_its_hash() {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![0, 0, 0, 0, 10, 20, 30, 255]).expect("2x1");
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    let frame = |policy| {
        MediaConnector::decode_image(
            Bytes::from(png.clone()),
            ImageDetail::Auto,
            None,
            policy,
            ImageSource::InlineBytes,
        )
        .expect("decodes")
    };
    let keep = frame(TransparencyPolicy::KeepAlpha);
    let white = frame(TransparencyPolicy::CompositeOverWhite);
    assert_eq!(keep.transparency(), TransparencyPolicy::KeepAlpha);
    assert_eq!(white.transparency(), TransparencyPolicy::CompositeOverWhite);
    assert_eq!(keep.hash, crate::hasher::hash_image(&png));
    assert_eq!(
        white.hash,
        crate::hasher::hash_image_with_transparency(
            &png,
            None,
            TransparencyPolicy::CompositeOverWhite
        )
    );
    assert_ne!(keep.hash, white.hash);
    assert!(keep.image().expect("decode").color().has_alpha());
    assert_eq!(
        white.image().expect("decode").to_rgb8().as_raw(),
        &[255, 255, 255, 10, 20, 30]
    );
}

#[test]
fn different_caps_hash_differently() {
    // The pixel cache and the backend's mm cache key off this hash; two
    // tiers of the same bytes must not collide.
    let bytes = b"fake-encoded-image-bytes";
    let low = crate::hasher::hash_image_with_resolution_cap(bytes, Some(252));
    let high = crate::hasher::hash_image_with_resolution_cap(bytes, Some(1008));
    assert_ne!(low, high);
}

#[test]
fn absent_cap_keeps_the_plain_byte_hash() {
    // Uncapped requests keep their existing identity so cached entries
    // stay valid.
    let bytes = b"fake-encoded-image-bytes";
    assert_eq!(
        crate::hasher::hash_image_with_resolution_cap(bytes, None),
        crate::hasher::hash_image(bytes)
    );
}

#[test]
fn same_cap_hashes_stably() {
    let bytes = b"fake-encoded-image-bytes";
    assert_eq!(
        crate::hasher::hash_image_with_resolution_cap(bytes, Some(504)),
        crate::hasher::hash_image_with_resolution_cap(bytes, Some(504))
    );
}

#[test]
fn absent_cap_is_a_no_op() {
    let original = apply_max_long_side_pixel(image(5000, 3000), None);
    assert_eq!((original.width(), original.height()), (5000, 3000));
}

#[test]
fn larger_caps_keep_more_pixels() {
    // Monotonicity is what the contract suite asserts via prompt_tokens.
    let low = apply_max_long_side_pixel(image(5000, 3000), Some(252));
    let mid = apply_max_long_side_pixel(image(5000, 3000), Some(504));
    let high = apply_max_long_side_pixel(image(5000, 3000), Some(1008));

    assert!(low.width() < mid.width());
    assert!(mid.width() < high.width());
}
