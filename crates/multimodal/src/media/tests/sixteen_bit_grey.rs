use super::super::*;

#[test]
fn greyscale_samples_are_clipped_not_scaled() {
    let wide = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        6,
        1,
        vec![0, 1, 254, 255, 256, 0xFFFF],
    )
    .expect("6x1 grey16");
    match clip_sixteen_bit_grey(image::DynamicImage::ImageLuma16(wide), &[]) {
        image::DynamicImage::ImageLuma8(narrow) => {
            assert_eq!(narrow.as_raw(), &[0, 1, 254, 255, 255, 255]);
        }
        other => panic!("expected grey8, got {:?}", other.color()),
    }
}

#[test]
fn other_layouts_pass_through_untouched() {
    let rgb = image::DynamicImage::new_rgb16(2, 2);
    assert!(matches!(
        clip_sixteen_bit_grey(rgb, &[]),
        image::DynamicImage::ImageRgb16(_)
    ));
    // Greyscale with alpha from a container that is not a greyscale PNG
    // (here: no container at all) is the narrowing's, not the clip's.
    let grey_alpha = image::DynamicImage::new_luma_a16(2, 2);
    assert!(matches!(
        clip_sixteen_bit_grey(grey_alpha, &[]),
        image::DynamicImage::ImageLumaA16(_)
    ));
}

/// A 2x2 16-bit greyscale PNG (colour type 0) with samples 1, 255, 256
/// and 65535 and a `tRNS` key of 256. The reference loader opens it in
/// its 16-bit greyscale mode with the key as metadata: its RGB is the
/// clipped sample on every channel, the key ignored.
const GREY16_WITH_KEY: [u8; 86] = [
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x10, 0x00, 0x00, 0x00, 0x00, 0x07, 0x4d, 0x8e,
    0xbb, 0x00, 0x00, 0x00, 0x02, 0x74, 0x52, 0x4e, 0x53, 0x01, 0x00, 0x6f, 0x88, 0xfc, 0x79, 0x00,
    0x00, 0x00, 0x0f, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0x60, 0x64, 0xf8, 0x0f, 0xc2,
    0xff, 0x01, 0x09, 0x0d, 0x03, 0x00, 0x03, 0x69, 0x79, 0x60, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

/// A 2x2 16-bit greyscale-with-alpha PNG (colour type 4) with the pairs
/// (1, 65535), (255, 65535), (256, 65535) and (65535, 32768). The
/// reference loader unpacks it by the high byte of each sample.
const GREY_ALPHA16: [u8; 79] = [
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x10, 0x04, 0x00, 0x00, 0x00, 0x88, 0x2f, 0x19,
    0xec, 0x00, 0x00, 0x00, 0x16, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0x60, 0xfc, 0xff,
    0x9f, 0xe1, 0x3f, 0x10, 0x33, 0x82, 0xc8, 0xff, 0x0d, 0x0c, 0x00, 0x50, 0xda, 0x09, 0x7a, 0xc6,
    0xe8, 0x50, 0x15, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

/// The key of a 16-bit greyscale PNG is not an alpha channel to the
/// reference loader's RGB conversion: the decode drops the alpha the
/// decoder expands from it and clips the samples, whatever the
/// transparency policy, so the pixels are that loader's RGB.
#[test]
fn a_sixteen_bit_greyscale_pngs_key_is_not_an_alpha_channel() {
    for policy in [
        TransparencyPolicy::CompositeOverWhite,
        TransparencyPolicy::DropAlpha,
        TransparencyPolicy::KeepAlpha,
    ] {
        let image = decode_image_bytes(&GREY16_WITH_KEY, None, policy).expect("a PNG");
        assert!(!image.color().has_alpha(), "{policy:?}: no alpha channel");
        assert_eq!(
            image.to_rgb8().as_raw(),
            &[1, 1, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "{policy:?}"
        );
    }
}

/// Greyscale stored with an alpha channel at 16 bits is another layout to
/// the reference loader, read by the high byte of every sample: the clip
/// leaves it to the narrowing, and its alpha stays.
#[test]
fn a_sixteen_bit_grey_alpha_png_keeps_its_high_bytes() {
    let image =
        decode_image_bytes(&GREY_ALPHA16, None, TransparencyPolicy::KeepAlpha).expect("a PNG");
    assert_eq!(
        image.to_rgba8().as_raw(),
        &[0, 0, 0, 255, 0, 0, 0, 255, 1, 1, 1, 255, 255, 255, 255, 128]
    );
}
