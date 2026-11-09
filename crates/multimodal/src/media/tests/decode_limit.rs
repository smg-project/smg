use super::super::*;

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn png_chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// A PNG whose header declares a `side` x `side` RGB8 image, with the
/// smallest IDAT a decoder accepts as a start (the stored zlib block of a
/// single filter byte): a few dozen bytes on the wire, `side * side * 3`
/// bytes once decoded.
fn png_declaring(side: u32) -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&side.to_be_bytes());
    ihdr.extend_from_slice(&side.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut out, *b"IHDR", &ihdr);
    png_chunk(
        &mut out,
        *b"IDAT",
        &[
            0x78, 0x01, 0x01, 0x01, 0x00, 0xFE, 0xFF, 0x00, 0x00, 0x01, 0x00, 0x01,
        ],
    );
    png_chunk(&mut out, *b"IEND", &[]);
    out
}

#[test]
fn a_decoded_size_over_the_allocation_limit_is_refused_before_the_allocation() {
    // 30000 x 30000 x 3 = 2.7 GB declared by a 70-byte file.
    let bomb = png_declaring(30_000);
    match decode_image_bytes(&bomb, None, TransparencyPolicy::KeepAlpha) {
        Err(MediaConnectorError::Image(image::ImageError::Limits(_))) => {}
        other => panic!("expected the crate's limit error, got {other:?}"),
    }
    // The header is still readable on its own: the size probe does not
    // allocate the pixels.
    let size = encoded_image_size(&bomb, None).expect("header read");
    assert_eq!((size.width, size.height), (30_000, 30_000));
}

#[test]
fn a_small_png_still_decodes_through_the_limit_check() {
    let mut png = Vec::new();
    image::RgbImage::from_pixel(3, 2, image::Rgb([10, 20, 30]))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode");
    let image = decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha).expect("decode");
    assert_eq!((image.width(), image.height()), (3, 2));
}
