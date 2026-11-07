use image::{codecs::jpeg::JpegEncoder, GenericImageView, Rgb};

use super::super::*;

/// A 4x2 JPEG, red on the left and blue on the right, carrying the given
/// EXIF orientation in an APP1 segment (TIFF little-endian, one IFD0
/// entry: tag 0x0112, SHORT, count 1).
fn jpeg_with_orientation(orientation: u8) -> Vec<u8> {
    let mut img = image::RgbImage::new(4, 2);
    for (x, _, pixel) in img.enumerate_pixels_mut() {
        *pixel = if x < 2 {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        };
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 100)
        .encode_image(&img)
        .expect("encode");
    let mut app1 = vec![0xFF, 0xE1, 0x00, 0x22];
    app1.extend_from_slice(b"Exif\0\0");
    app1.extend_from_slice(b"II*\0\x08\0\0\0");
    app1.extend_from_slice(&[0x01, 0x00]);
    app1.extend_from_slice(&[0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00]);
    app1.extend_from_slice(&[orientation, 0x00, 0x00, 0x00]);
    app1.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
}

fn is_red(pixel: image::Rgba<u8>) -> bool {
    pixel[0] > 128 && pixel[2] < 128
}

fn is_blue(pixel: image::Rgba<u8>) -> bool {
    pixel[2] > 128 && pixel[0] < 128
}

#[test]
fn a_jpeg_stored_sideways_is_decoded_upright_like_the_engine_loader() {
    // Orientation 6: rotate 90 degrees clockwise for display, so the
    // 4x2 picture is a 2x4 one with the red half on top.
    let bytes = jpeg_with_orientation(6);
    let size = encoded_image_size(&bytes, None).expect("size");
    assert_eq!((size.width, size.height), (2, 4));
    let img = decode_image_bytes(&bytes, None, TransparencyPolicy::KeepAlpha).expect("decode");
    assert_eq!(img.dimensions(), (2, 4));
    assert!(is_red(img.get_pixel(0, 0)) && is_red(img.get_pixel(1, 0)));
    assert!(is_blue(img.get_pixel(0, 3)) && is_blue(img.get_pixel(1, 3)));
}

#[test]
fn a_mirrored_jpeg_keeps_its_size_and_swaps_its_halves() {
    // Orientation 2: flipped horizontally.
    let bytes = jpeg_with_orientation(2);
    let size = encoded_image_size(&bytes, None).expect("size");
    assert_eq!((size.width, size.height), (4, 2));
    let img = decode_image_bytes(&bytes, None, TransparencyPolicy::KeepAlpha).expect("decode");
    assert!(is_blue(img.get_pixel(0, 0)) && is_red(img.get_pixel(3, 0)));
}

#[test]
fn an_upright_jpeg_and_a_png_are_left_alone() {
    let bytes = jpeg_with_orientation(1);
    let img = decode_image_bytes(&bytes, None, TransparencyPolicy::KeepAlpha).expect("decode");
    assert_eq!(img.dimensions(), (4, 2));
    assert!(is_red(img.get_pixel(0, 0)) && is_blue(img.get_pixel(3, 0)));

    let mut png = Vec::new();
    image::DynamicImage::new_rgb8(4, 2)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    assert_eq!(
        decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha)
            .expect("decode")
            .dimensions(),
        (4, 2)
    );
    let size = encoded_image_size(&png, None).expect("size");
    assert_eq!((size.width, size.height), (4, 2));
}
