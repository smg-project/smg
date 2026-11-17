use super::super::*;
use crate::buffer_pool::BufferPool;

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

/// The reference's integer blend of one pixel over white.
fn blend(px: [u8; 4]) -> [u8; 3] {
    let alpha = u32::from(px[3]);
    [px[0], px[1], px[2]].map(|c| ((u32::from(c) * alpha + 255 * (255 - alpha) + 127) / 255) as u8)
}

/// Every channel value at every alpha, 65,536 pixels: the composite is the
/// reference's blend.
#[test]
fn the_blend_holds_for_every_value_at_every_alpha() {
    let rgba = image::RgbaImage::from_fn(256, 256, |value, alpha| {
        image::Rgba([value as u8, !(value as u8), value as u8 ^ 0x5a, alpha as u8])
    });
    let rgb = composite_over_white(&rgba);
    let (blended, _) = rgb.as_raw().as_chunks::<3>();
    let (pixels, _) = rgba.as_raw().as_chunks::<4>();
    assert_eq!(blended.len(), pixels.len());
    for (out, px) in blended.iter().zip(pixels) {
        assert_eq!(*out, blend(*px), "{px:?}");
    }
}

/// The same four pixels in a container seven bytes longer than the image:
/// the composite ends at the image's pixels.
#[test]
fn the_blend_ends_at_the_images_pixels() {
    let mut container = vec![
        10, 20, 30, 255, 10, 20, 30, 0, 10, 20, 30, 128, 200, 100, 0, 1,
    ];
    container.extend_from_slice(&[9; 7]);
    let rgba = image::RgbaImage::from_raw(4, 1, container).expect("4x1 RGBA");
    let rgb = composite_over_white(&rgba);
    assert_eq!(
        rgb.as_raw(),
        &[10, 20, 30, 255, 255, 255, 132, 137, 142, 255, 254, 254]
    );
}

/// A 320x320 image, 307,200 bytes of output (the pool's 512 KiB class): the
/// composite is written into a buffer of the pool, where a debug build's
/// buffer arrives holding the sentinel, and the image returns that buffer.
#[test]
fn the_composite_is_written_into_a_pool_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let rgba = image::RgbaImage::from_fn(320, 320, |x, y| {
            image::Rgba([x as u8, y as u8, (x ^ y) as u8, (x + y) as u8])
        });
        let rgb = composite_over_white(&rgba);
        let (blended, _) = rgb.as_raw().as_chunks::<3>();
        let (pixels, _) = rgba.as_raw().as_chunks::<4>();
        assert_eq!(blended.len(), pixels.len());
        assert!(blended
            .iter()
            .zip(pixels)
            .all(|(out, px)| *out == blend(*px)));
        buffer_pool::recycle_image(image::DynamicImage::ImageRgb8(rgb));
        assert_eq!(MINE.held(), 512 << 10);
    });
}
