use image::codecs::jpeg::JpegEncoder;

use super::super::*;

/// A 16x8 JPEG with a left-to-right ramp, written by the `image` crate's
/// encoder, which ends it with the end-of-image marker.
fn ramp_jpeg() -> Vec<u8> {
    let mut img = image::RgbImage::new(16, 8);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x * 16) as u8, (y * 32) as u8, 128]);
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 90)
        .encode_image(&img)
        .expect("encode");
    assert!(jpeg.ends_with(&[0xFF, 0xD9]), "the encoder ends the file");
    jpeg
}

/// The offset of the scan header: the segment whose marker is SOS.
fn scan_header(jpeg: &[u8]) -> usize {
    (2..jpeg.len() - 1)
        .find(|&at| jpeg[at] == 0xFF && jpeg[at + 1] == 0xDA)
        .expect("a scan header")
}

/// A JPEG whose data ends before its end-of-image marker is refused on both
/// decode paths, as the reference loader refuses it, where a complete file
/// and one with stray bytes before its scan header (a defect both decoders
/// read past) decode.
#[test]
fn a_jpeg_cut_before_its_end_marker_is_refused() {
    let jpeg = ramp_jpeg();
    let image = decode_image_bytes(&jpeg, None, TransparencyPolicy::KeepAlpha).expect("decode");
    assert_eq!((image.width(), image.height()), (16, 8));

    let cut = &jpeg[..jpeg.len() - 2];
    let error = decode_image_bytes(cut, None, TransparencyPolicy::KeepAlpha)
        .expect_err("a cut file is refused");
    assert!(
        matches!(
            error,
            MediaConnectorError::Image(image::ImageError::Decoding(_))
        ),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("ends in a form this decoder refuses"),
        "{error}"
    );

    let mut stray = jpeg.clone();
    let at = scan_header(&stray);
    stray.splice(at..at, [0x12, 0x34]);
    let image = decode_image_bytes(&stray, None, TransparencyPolicy::KeepAlpha)
        .expect("stray bytes before a marker decode");
    assert_eq!((image.width(), image.height()), (16, 8));
}

/// The reference loader's rule behind a single-scan file's scan, measured on
/// Pillow 12.3 with its bundled libjpeg-turbo: in place of the end marker, a
/// comment segment whose payload holds the marker's two bytes, with or
/// without a byte after it, or a comment with another payload, and the file
/// is served with the clean file's pixels (the comment's marker ended the
/// entropy-coded data with every row out, and the missing end marker is
/// ignored). With a library that reads warnings (TurboJPEG 2.0 and later)
/// the frame it decoded is kept; otherwise the pure-Rust decoder reads to
/// the end marker and refuses the file itself.
#[test]
fn a_single_scan_file_ending_in_a_comment_is_served_as_the_reference_serves_it() {
    let jpeg = ramp_jpeg();
    let clean = decode_image_bytes(&jpeg, None, TransparencyPolicy::KeepAlpha)
        .expect("decode")
        .to_rgb8();
    let cut = &jpeg[..jpeg.len() - 2];
    let tails: [&[u8]; 3] = [
        &[0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9],
        &[0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9, 0x00],
        &[0xFF, 0xFE, 0x00, 0x05, 0x61, 0x62, 0x63],
    ];
    for tail in tails {
        let file = [cut, tail].concat();
        let decoded = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha);
        if crate::jpeg_turbo::turbojpeg_reads_warnings() {
            let served = decoded.expect("served with the library").to_rgb8();
            assert!(
                served.as_raw() == clean.as_raw(),
                "the clean file's pixels for the tail {tail:02X?}"
            );
        } else {
            let error = decoded.expect_err("the pure-Rust decoder reads to the end marker");
            assert!(
                !error
                    .to_string()
                    .contains("ends in a form this decoder refuses"),
                "refused by the decoder, not before it: {error}"
            );
        }
    }
}

/// A progressive file is read through to its end marker before its first
/// row, so the reference loader refuses it with a comment in place of the
/// marker, and so does this path. The copy of the ramp below stands in by
/// its markers alone: its frame marker is rewritten to the progressive one
/// over a baseline scan, which the library refuses outright as a bad
/// progression (no frame, no warning), and the fallback arm's walk then
/// refuses the file before the fallback decoder; a real progressive file
/// ending in a comment is the warned case the walk refuses as well. The
/// refusal before the fallback decoder is told from the decoders' own by
/// its message.
#[test]
fn a_progressive_file_ending_in_a_comment_is_refused() {
    let mut jpeg = ramp_jpeg();
    let frame = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xC0])
        .expect("the frame header");
    jpeg[frame + 1] = 0xC2;
    if let Err(error) = decode_image_bytes(&jpeg, None, TransparencyPolicy::KeepAlpha) {
        assert!(
            !error
                .to_string()
                .contains("ends in a form this decoder refuses"),
            "with its end marker the file reaches the decoders: {error}"
        );
    }
    let mut comment = jpeg[..jpeg.len() - 2].to_vec();
    comment.extend([0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9]);
    let error = decode_image_bytes(&comment, None, TransparencyPolicy::KeepAlpha)
        .expect_err("refused before the fallback decoder");
    assert!(
        error
            .to_string()
            .contains("ends in a form this decoder refuses"),
        "{error}"
    );
}

/// One of the residuals against the reference loader (`end_is_served`
/// names them), which serves a single-scan file whose entropy-coded data
/// is followed by zero bytes in place of the end marker when its decoder's
/// read past the data is satisfied: refused
/// here on both paths, like the bare end, since the walk cannot tell those
/// bytes from the data and whether the read was satisfied is the decoder's
/// to know (the library reports its first warning only, the same for both).
#[test]
fn a_single_scan_file_padded_behind_its_data_is_refused() {
    let jpeg = ramp_jpeg();
    let mut padded = jpeg[..jpeg.len() - 2].to_vec();
    padded.extend([0; 64]);
    let error = decode_image_bytes(&padded, None, TransparencyPolicy::KeepAlpha)
        .expect_err("the padded file is refused");
    assert!(
        error
            .to_string()
            .contains("ends in a form this decoder refuses"),
        "{error}"
    );
}

/// Behind a single-scan frame's scan libjpeg parses a second scan header
/// and the table segments, where it steps over a comment or an application
/// segment, and the reference loader refuses what libjpeg refuses there
/// ("broken data stream"): a scan cut halfway and followed by a copy of the
/// scan header, by a Huffman table with a bad length or by a restart
/// interval with a bad length, and a complete scan followed by a copy of the
/// scan header with no end marker, are refused on both paths. A comment or
/// an application segment with a length short of its own two bytes before
/// the end marker is stepped over as nothing, as libjpeg steps over it, and
/// a warned file with one is served with the clean file's pixels through
/// the library; the pure-Rust decoder's reading of such a length is its own.
#[test]
fn segments_libjpeg_parses_behind_a_single_scan_are_refused_as_the_reference_refuses_them() {
    let jpeg = ramp_jpeg();
    let scan = scan_header(&jpeg);
    let length = usize::from(u16::from_be_bytes([jpeg[scan + 2], jpeg[scan + 3]]));
    let header = &jpeg[scan..scan + 2 + length];
    let data = scan + 2 + length;
    let half = &jpeg[..data + (jpeg.len() - 2 - data) / 2];
    let cut = &jpeg[..jpeg.len() - 2];
    let refused: [Vec<u8>; 4] = [
        [half, header].concat(),
        [half, &[0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00][..]].concat(),
        [half, &[0xFF, 0xDD, 0x00, 0x03, 0x00][..]].concat(),
        [cut, header].concat(),
    ];
    for file in &refused {
        let error = decode_image_bytes(file, None, TransparencyPolicy::KeepAlpha)
            .expect_err("refused before the fallback decoder");
        assert!(
            error
                .to_string()
                .contains("ends in a form this decoder refuses"),
            "{error}"
        );
    }
    if !crate::test_support::warned_decode_or_skip("the served short-length tails") {
        return;
    }
    let clean = decode_image_bytes(&jpeg, None, TransparencyPolicy::KeepAlpha)
        .expect("decode")
        .to_rgb8();
    let mut stray = jpeg.clone();
    stray.splice(scan..scan, [0x12, 0x34]);
    let tails: [&[u8]; 2] = [
        &[0xFF, 0xFE, 0x00, 0x01, 0xFF, 0xD9],
        &[0xFF, 0xE1, 0x00, 0x00, 0xFF, 0xD9],
    ];
    for tail in tails {
        let file = [&stray[..stray.len() - 2], tail].concat();
        let served = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha)
            .expect("served through the library")
            .to_rgb8();
        assert!(
            served.as_raw() == clean.as_raw(),
            "the clean file's pixels for the tail {tail:02X?}"
        );
    }
}

/// Behind a single-scan frame's scan libjpeg parses a table segment and refuses
/// a malformed one, where the walk reads the segment's length alone, so a table
/// there is refused before the fallback decoder, end marker or not. The
/// reference loader (Pillow 12.3 with its bundled libjpeg-turbo) refuses a
/// Huffman table of length 3, one of length 4 with two zero bytes, one with a
/// table index of 5 and a quantization table of length 3 with an index of 5,
/// each behind a warned or a cut scan and before the end marker, and loads the
/// file's own Huffman table copied there, an empty table of length 2 and a
/// restart interval of length 4: those are the documented residual, refused
/// here where the loader serves them. A clean scan followed by a valid table
/// and the marker warns the library of nothing, so with the library it is
/// served as the loader serves it, without the reading; without the library the
/// reading refuses it.
#[test]
fn a_table_behind_a_single_scan_frames_scan_is_refused_before_the_fallback_decoder() {
    let jpeg = ramp_jpeg();
    let scan = scan_header(&jpeg);
    let length = usize::from(u16::from_be_bytes([jpeg[scan + 2], jpeg[scan + 3]]));
    let data = scan + 2 + length;
    let half = &jpeg[..data + (jpeg.len() - 2 - data) / 2];
    let mut stray = jpeg.clone();
    stray.splice(scan..scan, [0x12, 0x34]);
    let warned = &stray[..stray.len() - 2];
    let huffman = jpeg
        .windows(2)
        .position(|pair| pair == [0xFF, 0xC4])
        .expect("a Huffman table");
    let table_length = usize::from(u16::from_be_bytes([jpeg[huffman + 2], jpeg[huffman + 3]]));
    let own_table = &jpeg[huffman..huffman + 2 + table_length];
    let marker: &[u8] = &[0xFF, 0xD9];
    let mut index_five = vec![0xFF, 0xC4, 0x00, 0x13, 0x05];
    index_five.extend([0; 16]);
    let tables: [&[u8]; 7] = [
        &[0xFF, 0xC4, 0x00, 0x03, 0x00],
        &[0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00],
        &index_five,
        &[0xFF, 0xDB, 0x00, 0x03, 0x05],
        own_table,
        &[0xFF, 0xC4, 0x00, 0x02],
        &[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x10],
    ];
    for table in tables {
        for scan in [warned, half] {
            let file = [scan, table, marker].concat();
            let error = decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha)
                .expect_err("refused before the fallback decoder");
            assert!(
                error
                    .to_string()
                    .contains("ends in a form this decoder refuses"),
                "{error} for the table {table:02X?}"
            );
        }
    }
    let complete = [&jpeg[..jpeg.len() - 2], own_table, marker].concat();
    let decoded = decode_image_bytes(&complete, None, TransparencyPolicy::KeepAlpha);
    if crate::jpeg_turbo::turbojpeg_available() {
        let clean = decode_image_bytes(&jpeg, None, TransparencyPolicy::KeepAlpha)
            .expect("decode")
            .to_rgb8();
        let served = decoded.expect("served with the library").to_rgb8();
        assert!(
            served.as_raw() == clean.as_raw(),
            "the clean file's pixels behind a clean scan"
        );
    } else {
        let error = decoded.expect_err("refused before the decoder");
        assert!(
            error
                .to_string()
                .contains("ends in a form this decoder refuses"),
            "{error}"
        );
    }
}
