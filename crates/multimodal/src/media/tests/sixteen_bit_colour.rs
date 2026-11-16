use super::super::*;

#[test]
fn truecolour_samples_keep_their_high_byte() {
    // Low bytes of 128 and more are where rounding would land one higher.
    let wide = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(
        2,
        1,
        vec![0x0080, 0x00FF, 0x8080, 0xFF7F, 0x0100, 0xFFFF],
    )
    .expect("2x1 RGB16");
    match narrow_sixteen_bit_colour(image::DynamicImage::ImageRgb16(wide)) {
        image::DynamicImage::ImageRgb8(narrow) => {
            assert_eq!(narrow.as_raw(), &[0, 0, 0x80, 0xFF, 1, 0xFF]);
        }
        other => panic!("expected RGB8, got {:?}", other.color()),
    }
}

#[test]
fn alpha_samples_are_narrowed_the_same_way() {
    let wide =
        image::ImageBuffer::<image::LumaA<u16>, Vec<u16>>::from_raw(1, 1, vec![0x1234, 0xFF80])
            .expect("1x1 LA16");
    match narrow_sixteen_bit_colour(image::DynamicImage::ImageLumaA16(wide)) {
        image::DynamicImage::ImageLumaA8(narrow) => assert_eq!(narrow.as_raw(), &[0x12, 0xFF]),
        other => panic!("expected LA8, got {:?}", other.color()),
    }
}

#[test]
fn other_depths_pass_through_untouched() {
    let grey = image::DynamicImage::new_luma16(2, 2);
    assert!(matches!(
        narrow_sixteen_bit_colour(grey),
        image::DynamicImage::ImageLuma16(_)
    ));
    let rgb = image::DynamicImage::new_rgb8(2, 2);
    assert!(matches!(
        narrow_sixteen_bit_colour(rgb),
        image::DynamicImage::ImageRgb8(_)
    ));
}
