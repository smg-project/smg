use super::super::*;
use crate::buffer_pool::BufferPool;

/// A 320x320 RGBA image, 409,600 bytes of pixels (the pool's 512 KiB class),
/// and its PNG.
fn rgba_square_and_its_png() -> (image::DynamicImage, Vec<u8>) {
    let square = image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(320, 320, |x, y| {
        image::Rgba([x as u8, y as u8, (x ^ y) as u8, (x + y) as u8])
    }));
    let mut png = Vec::new();
    square
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode");
    (square, png)
}

/// The decoded PNG is a pooled buffer: the second decode takes the buffer the
/// first one returned and writes the same pixels into it (the decoder writes
/// every byte of the image it reports).
#[test]
fn a_png_decode_reuses_the_returned_buffer_and_writes_the_same_pixels() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let (square, png) = rgba_square_and_its_png();
        let first = decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert!(first.as_bytes() == square.as_bytes());
        let address = first.as_bytes().as_ptr();
        buffer_pool::recycle_image(first);
        assert_eq!(MINE.held(), 512 << 10);
        let second =
            decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert_eq!(
            second.as_bytes().as_ptr(),
            address,
            "the second decode took the first's buffer"
        );
        assert!(second.as_bytes() == square.as_bytes());
    });
}

/// A policy that flattens the alpha hands the frame a new image, so the
/// decoder's buffer is back in the pool when the decode returns.
#[test]
fn a_flattened_png_returns_the_decoders_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let (_, png) = rgba_square_and_its_png();
        for policy in [
            TransparencyPolicy::DropAlpha,
            TransparencyPolicy::CompositeOverWhite,
        ] {
            let flat = decode_image_bytes(&png, None, policy).expect("decoded");
            assert!(!flat.color().has_alpha(), "{policy:?}");
            assert_eq!(MINE.held(), 512 << 10, "{policy:?}");
        }
    });
}

/// A file cut short takes its buffer and then fails in the decoder: the
/// buffer goes back to the pool, as a refused JPEG's or WebP's does.
#[test]
fn a_refused_png_returns_its_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let (_, png) = rgba_square_and_its_png();
        let cut = &png[..png.len() / 2];
        assert!(decode_image_bytes(cut, None, TransparencyPolicy::KeepAlpha).is_err());
        assert_eq!(MINE.held(), 512 << 10);
    });
}

/// CRC-32 of a PNG chunk's type and data.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

/// Appends one PNG chunk: length, type, data, CRC.
fn push_chunk(png: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(&kind);
    png.extend_from_slice(data);
    let crc = crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

/// The first column, first row, column step and row step of each Adam7 pass.
const ADAM7: [(u32, u32, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// `square` as an Adam7 PNG. The `image` crate's encoder writes no interlaced
/// file, so the seven passes go into stored deflate blocks here.
fn interlaced_png(square: &image::RgbaImage) -> Vec<u8> {
    let (width, height) = square.dimensions();
    let mut scanlines = Vec::new();
    for (first_column, first_row, column_step, row_step) in ADAM7 {
        for y in (first_row..height).step_by(row_step) {
            // Filter type 0: the row as it is.
            scanlines.push(0);
            for x in (first_column..width).step_by(column_step) {
                scanlines.extend_from_slice(&square.get_pixel(x, y).0);
            }
        }
    }
    let mut zlib = vec![0x78, 0x01];
    let last = scanlines.len().div_ceil(0xFFFF) - 1;
    for (index, block) in scanlines.chunks(0xFFFF).enumerate() {
        let len = block.len() as u16;
        zlib.push(u8::from(index == last));
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (low, high) = scanlines.iter().fold((1u32, 0u32), |(low, high), &byte| {
        let low = (low + u32::from(byte)) % 65_521;
        (low, (high + low) % 65_521)
    });
    zlib.extend_from_slice(&((high << 16) | low).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // Eight bits a channel, RGBA, deflate, adaptive filtering, Adam7.
    header.extend_from_slice(&[8, 6, 0, 0, 1]);
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    push_chunk(&mut png, *b"IHDR", &header);
    push_chunk(&mut png, *b"IDAT", &zlib);
    push_chunk(&mut png, *b"IEND", &[]);
    png
}

/// An Adam7 file takes the decoder's other write path, pass by pass. It too
/// writes every byte of the image: in a debug build each taken buffer arrives
/// holding the sentinel, so a pixel a pass left unwritten fails the
/// comparison.
#[test]
fn an_interlaced_png_writes_every_pixel_of_its_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let (square, _) = rgba_square_and_its_png();
        let png = interlaced_png(square.as_rgba8().expect("an RGBA square"));
        let first = decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert!(first.as_bytes() == square.as_bytes());
        buffer_pool::recycle_image(first);
        assert_eq!(
            MINE.held(),
            512 << 10,
            "the decode took a buffer of the pool"
        );
        let second =
            decode_image_bytes(&png, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert!(second.as_bytes() == square.as_bytes());
    });
}
