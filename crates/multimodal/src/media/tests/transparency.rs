use super::super::*;

#[test]
fn png_colour_type_comes_from_the_header_alone() {
    let mut png = PNG_SIGNATURE.to_vec();
    png.extend_from_slice(&[0, 0, 0, 13]);
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
    assert_eq!(png_colour_type(&png), Some(PNG_COLOUR_TYPE_TRUECOLOUR));
    assert_eq!(png_colour_type(&png[..20]), None);
    assert_eq!(png_colour_type(b"RIFF\0\0\0\0WEBPVP8 "), None);
}

#[test]
fn the_blend_is_the_reference_loader_s() {
    // Pillow's paste through the alpha mask over (255, 255, 255).
    let rgba = image::RgbaImage::from_raw(
        4,
        1,
        vec![
            10, 20, 30, 255, 10, 20, 30, 0, 10, 20, 30, 128, 200, 100, 0, 1,
        ],
    )
    .expect("4x1 RGBA");
    let rgb = composite_over_white(&rgba);
    assert_eq!(
        rgb.as_raw(),
        &[10, 20, 30, 255, 255, 255, 132, 137, 142, 255, 254, 254]
    );
}

#[test]
fn opaque_images_pass_through_untouched() {
    for policy in [
        TransparencyPolicy::KeepAlpha,
        TransparencyPolicy::CompositeOverWhite,
        TransparencyPolicy::DropAlpha,
    ] {
        let grey = image::DynamicImage::new_luma8(2, 2);
        assert!(matches!(
            apply_transparency(grey, &[], policy),
            image::DynamicImage::ImageLuma8(_)
        ));
    }
}

#[test]
fn each_policy_treats_a_transparent_pixel_its_own_way() {
    let rgba = || {
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(2, 1, vec![0, 0, 0, 0, 10, 20, 30, 255]).expect("2x1 RGBA"),
        )
    };
    let white = apply_transparency(rgba(), &[], TransparencyPolicy::CompositeOverWhite);
    assert_eq!(white.to_rgb8().as_raw(), &[255, 255, 255, 10, 20, 30]);
    assert!(!white.color().has_alpha());
    let dropped = apply_transparency(rgba(), &[], TransparencyPolicy::DropAlpha);
    assert_eq!(dropped.to_rgb8().as_raw(), &[0, 0, 0, 10, 20, 30]);
    assert!(!dropped.color().has_alpha());
    let kept = apply_transparency(rgba(), &[], TransparencyPolicy::KeepAlpha);
    assert!(kept.color().has_alpha());
}
