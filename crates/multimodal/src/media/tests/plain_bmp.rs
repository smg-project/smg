use image::codecs::bmp::BmpDecoder;

use super::super::*;
use crate::buffer_pool::BufferPool;

/// A bitmap file: the 14-byte file header, an info header of `header_len`
/// bytes (its first 40 written, the rest zero), `gap` bytes, then the rows
/// (`height` of them, or its negation when it is negative: a top-down file)
/// of `width` pixels of `bits` bits, each padded to four bytes. Every byte of
/// the gap and of a row, the padding too, is a function of its place.
fn bitmap(
    header_len: u32,
    width: i32,
    height: i32,
    bits: u16,
    compression: u32,
    gap: usize,
) -> Vec<u8> {
    let stride = (usize::from(bits / 8) * width.unsigned_abs() as usize + 3) & !3;
    let rows = stride * height.unsigned_abs() as usize;
    let offset = 14 + header_len as usize + gap;
    let mut file = Vec::with_capacity(offset + rows);
    file.extend_from_slice(b"BM");
    file.extend_from_slice(&((offset + rows) as u32).to_le_bytes());
    file.extend_from_slice(&[0; 4]);
    file.extend_from_slice(&(offset as u32).to_le_bytes());
    file.extend_from_slice(&header_len.to_le_bytes());
    file.extend_from_slice(&width.to_le_bytes());
    file.extend_from_slice(&height.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes());
    file.extend_from_slice(&bits.to_le_bytes());
    file.extend_from_slice(&compression.to_le_bytes());
    file.resize(14 + header_len as usize, 0);
    file.extend((0..gap).map(|at| (at * 5 + 3) as u8));
    file.extend((0..rows).map(|at| (at * 7 + at / stride * 13 + 1) as u8));
    file
}

/// What the `image` crate's own decoder makes of the file.
fn general(file: &[u8]) -> image::DynamicImage {
    let decoder = BmpDecoder::new(std::io::Cursor::new(file)).expect("a header");
    image::DynamicImage::from_decoder(decoder).expect("the pixels")
}

/// What the general branch makes of a bitmap without the direct reader: the
/// crate's decoder behind the same reservation, or `None` for its refusal.
fn general_or_refusal(file: &[u8]) -> Option<image::DynamicImage> {
    let mut decoder = BmpDecoder::new(std::io::Cursor::new(file)).ok()?;
    let mut limits = image::Limits::default();
    limits.reserve(decoder.total_bytes()).ok()?;
    decoder.set_limits(limits).ok()?;
    image::DynamicImage::from_decoder(decoder).ok()
}

/// The plain kind in every header size, row padding and row order, with and
/// without a gap before the rows: the direct reader takes it and gives the
/// pixels of the crate's decoder, and so does the whole decode.
#[test]
fn a_plain_bitmap_reads_the_general_decoders_pixels() {
    for header_len in [40, 52, 56, 108, 124] {
        for width in [1, 2, 3, 4, 5, 67] {
            for height in [1, 3, -1, -3] {
                for gap in [0, 5] {
                    let file = bitmap(header_len, width, height, 24, 0, gap);
                    let case = format!("{header_len} {width}x{height} gap {gap}");
                    let expected = general(&file);
                    let dimensions = (width.unsigned_abs(), height.unsigned_abs());
                    let direct = plain_bmp_into_pool(&file, dimensions).expect(&case);
                    assert_eq!(direct.color(), image::ColorType::Rgb8, "{case}");
                    assert_eq!((direct.width(), direct.height()), dimensions, "{case}");
                    assert!(direct.as_bytes() == expected.as_bytes(), "{case}");
                    let whole = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha)
                        .expect(&case);
                    assert_eq!(whole.color(), image::ColorType::Rgb8, "{case}");
                    assert!(whole.as_bytes() == expected.as_bytes(), "{case}");
                }
            }
        }
    }
}

/// A bitmap of another kind is not the direct reader's: the crate's decoder
/// reads it as before.
#[test]
fn a_bitmap_of_another_kind_stays_with_the_general_decoder() {
    for (kind, file) in [
        ("32 bits a pixel", bitmap(40, 5, 3, 32, 0, 0)),
        ("16 bits a pixel", bitmap(40, 5, 3, 16, 0, 0)),
        ("a palette", bitmap(40, 5, 3, 8, 0, 1024)),
    ] {
        assert!(plain_bmp_into_pool(&file, (5, 3)).is_none(), "{kind}");
        let whole = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect(kind);
        assert!(whole.as_bytes() == general(&file).as_bytes(), "{kind}");
    }
    let plain = bitmap(40, 5, 3, 24, 0, 0);
    assert!(plain_bmp_into_pool(&plain, (5, 3)).is_some());
    let mut two_planes = plain.clone();
    two_planes[26] = 2;
    let mut rows_past_the_end = plain.clone();
    rows_past_the_end[10..14].copy_from_slice(&(plain.len() as u32 + 1).to_le_bytes());
    for (kind, file, dimensions) in [
        (
            "a run-length compression",
            bitmap(40, 5, 3, 8, 1, 1024),
            (5, 3),
        ),
        ("the 12-byte header", bitmap(12, 5, 3, 24, 0, 0), (5, 3)),
        ("two planes", two_planes, (5, 3)),
        ("another width than the decoder read", plain.clone(), (4, 3)),
        (
            "another height than the decoder read",
            plain.clone(),
            (5, 4),
        ),
        ("no pixels", plain.clone(), (0, 0)),
        ("rows that start past the end", rows_past_the_end, (5, 3)),
        ("less than a header", plain[..53].to_vec(), (5, 3)),
    ] {
        assert!(plain_bmp_into_pool(&file, dimensions).is_none(), "{kind}");
    }
}

/// A file one byte short of its last row's padding is the crate's decoder's
/// to refuse: the direct reader takes no file short of its rows.
#[test]
fn a_bitmap_short_of_its_rows_is_refused_as_before() {
    let plain = bitmap(40, 5, 3, 24, 0, 0);
    let short = &plain[..plain.len() - 1];
    assert!(plain_bmp_into_pool(short, (5, 3)).is_none());
    assert!(decode_image_bytes(short, None, TransparencyPolicy::KeepAlpha).is_err());
}

/// Every byte of the two headers set to every value, in a bottom-up and a
/// top-down file. Whatever the change makes of the file (another size, kind
/// or place of the rows, rows short of the file, a header the crate's decoder
/// refuses), the whole decode gives what that decoder gives: its pixels, or a
/// refusal where it refuses.
#[test]
fn every_one_byte_change_of_the_header_reads_as_the_general_decoder_reads_it() {
    for height in [3, -3] {
        let plain = bitmap(40, 5, height, 24, 0, 0);
        for at in 0..54 {
            for value in 0..=u8::MAX {
                let mut file = plain.clone();
                file[at] = value;
                let whole = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).ok();
                let general = general_or_refusal(&file);
                assert_eq!(
                    whole.is_some(),
                    general.is_some(),
                    "byte {at} set to {value}, height {height}: decoded"
                );
                if let (Some(whole), Some(general)) = (whole, general) {
                    assert_eq!(
                        (whole.color(), whole.width(), whole.height()),
                        (general.color(), general.width(), general.height()),
                        "byte {at} set to {value}, height {height}"
                    );
                    assert!(
                        whole.as_bytes() == general.as_bytes(),
                        "byte {at} set to {value}, height {height}: the pixels"
                    );
                }
            }
        }
    }
}

/// The decoded bitmap is a pooled buffer: the second decode takes the buffer
/// the first one returned and writes the same pixels into it.
#[test]
fn a_plain_bitmap_decode_reuses_the_returned_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        // 320x320 pixels of three bytes: 307,200 bytes, the pool's 512 KiB
        // class.
        let file = bitmap(40, 320, 320, 24, 0, 0);
        let expected = general(&file);
        let first =
            decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert!(first.as_bytes() == expected.as_bytes());
        let address = first.as_bytes().as_ptr();
        buffer_pool::recycle_image(first);
        assert_eq!(
            MINE.held(),
            512 << 10,
            "the decoded bitmap was a buffer of the pool"
        );
        let second =
            decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert_eq!(
            second.as_bytes().as_ptr(),
            address,
            "the second decode took the first's buffer"
        );
        assert!(second.as_bytes() == expected.as_bytes());
    });
}
