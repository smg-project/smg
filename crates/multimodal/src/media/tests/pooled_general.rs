use super::super::*;
use crate::buffer_pool::BufferPool;

/// 320 pixels a side: 307,200 bytes of RGB and 409,600 of RGBA, both in the
/// pool's 512 KiB class, and 102,400 of grey, which the pool does not serve.
const SIDE: u32 = 320;

/// 256 blocks of 20 pixels a side, each of its own colour, the first one
/// transparent: few enough colours for a palette to hold them all.
fn blocks() -> image::DynamicImage {
    image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(SIDE, SIDE, |x, y| {
        let [index, ..] = (x / 20 + 16 * (y / 20)).to_le_bytes();
        let alpha = if index == 0 { 0 } else { 255 };
        image::Rgba([index, !index, index ^ 0x5A, alpha])
    }))
}

fn encoded(image: &image::DynamicImage, format: image::ImageFormat) -> Vec<u8> {
    let mut file = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut file), format)
        .expect("encoded");
    file
}

/// Files no other road of the decode takes: each reaches the general branch.
fn files() -> Vec<(&'static str, Vec<u8>)> {
    let rgba = blocks();
    let rgb = image::DynamicImage::ImageRgb8(rgba.to_rgb8());
    let grey = image::DynamicImage::ImageLuma8(rgba.to_luma8());
    vec![
        ("an RGB TIFF", encoded(&rgb, image::ImageFormat::Tiff)),
        ("an RGBA TIFF", encoded(&rgba, image::ImageFormat::Tiff)),
        ("a grey TIFF", encoded(&grey, image::ImageFormat::Tiff)),
        ("a GIF", encoded(&rgba, image::ImageFormat::Gif)),
        ("a 32-bit bitmap", encoded(&rgba, image::ImageFormat::Bmp)),
    ]
}

/// What the general branch made of a file before the pool: the crate's
/// decoder behind the same reservation, into the crate's own zeroed vector.
fn general(file: &[u8]) -> image::DynamicImage {
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(file))
        .with_guessed_format()
        .expect("a format")
        .into_decoder()
        .expect("a header");
    let mut limits = image::Limits::default();
    limits
        .reserve(decoder.total_bytes())
        .expect("inside the limit");
    decoder.set_limits(limits).expect("limits taken");
    image::DynamicImage::from_decoder(decoder).expect("the pixels")
}

/// Every file of the general branch decodes to the layout and the pixels the
/// crate's own vector held.
#[test]
fn the_general_branch_writes_the_crates_pixels() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        for (name, file) in files() {
            let expected = general(&file);
            let decoded =
                decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
            assert_eq!(decoded.color(), expected.color(), "{name}");
            assert!(decoded.as_bytes() == expected.as_bytes(), "{name}");
        }
    });
}

/// The decoded image is a pooled buffer: the second decode takes the buffer
/// the first one returned and writes the same pixels into it.
#[test]
fn the_general_branch_decodes_into_the_returned_buffer() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        for (name, file) in files() {
            let expected = general(&file);
            if expected.as_bytes().len() < buffer_pool::SMALLEST {
                // Below the pooled range: a zeroed allocation, as before.
                continue;
            }
            let first =
                decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
            let address = first.as_bytes().as_ptr();
            buffer_pool::recycle_image(first);
            assert_eq!(
                MINE.held(),
                512 << 10,
                "{name}: the decoded image was a buffer of the pool"
            );
            let second =
                decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
            assert_eq!(
                second.as_bytes().as_ptr(),
                address,
                "{name}: the second decode took the first's buffer"
            );
            assert!(second.as_bytes() == expected.as_bytes(), "{name}");
        }
    });
}

/// A decoder that reports an RGB image and writes no byte of it.
struct Silent;

impl image::ImageDecoder for Silent {
    fn dimensions(&self) -> (u32, u32) {
        (SIDE, SIDE)
    }

    fn color_type(&self) -> image::ColorType {
        image::ColorType::Rgb8
    }

    fn read_image(self, _buf: &mut [u8]) -> image::ImageResult<()> {
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> image::ImageResult<()> {
        (*self).read_image(buf)
    }
}

/// A byte the decoder leaves alone reads zero, as in the crate's own vector,
/// though the buffer it is in last held another image.
#[test]
fn a_byte_the_decoder_leaves_alone_reads_zero() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let len = usize::try_from(Silent.total_bytes()).expect("a length");
        let mut used = buffer_pool::take(len);
        used.fill(0x17);
        let address = used.as_ptr();
        buffer_pool::recycle(used);
        let image = eight_bit_into_pool(Silent, true).expect("decoded");
        assert_eq!(image.as_bytes().as_ptr(), address, "the returned buffer");
        assert!(image.as_bytes().iter().all(|&byte| byte == 0));
    });
}

/// A GIF whose every pixel names index 3 of a colour table of two entries,
/// in one frame over the whole image: its decoder writes no byte for such a
/// pixel. The codes are three bits each, the least significant bit first,
/// and a clear code (4) before every two pixels keeps them at that width.
fn a_gif_beyond_its_colour_table() -> Vec<u8> {
    let [low, high, ..] = SIDE.to_le_bytes();
    let mut file = b"GIF89a".to_vec();
    // The screen with its table of two colours, then the frame: no table of
    // its own, and the smallest code size of the format.
    file.extend([low, high, low, high, 0x80, 0, 0]);
    file.extend([0, 0, 0, 255, 255, 255]);
    file.extend([0x2C, 0, 0, 0, 0, low, high, low, high, 0, 2]);
    let codes = (0..SIDE * SIDE / 2).flat_map(|_| [4_u32, 3, 3]).chain([5]);
    let (mut data, mut bits, mut held) = (Vec::new(), 0_u32, 0);
    for code in codes {
        bits |= code << held;
        held += 3;
        while held >= 8 {
            let [byte, ..] = bits.to_le_bytes();
            data.push(byte);
            bits >>= 8;
            held -= 8;
        }
    }
    if held > 0 {
        let [byte, ..] = bits.to_le_bytes();
        data.push(byte);
    }
    for block in data.chunks(255) {
        file.push(u8::try_from(block.len()).expect("a sub-block"));
        file.extend(block);
    }
    file.extend([0, 0x3B]);
    file
}

/// A pixel beyond a GIF's colour table is a byte the decoder leaves alone:
/// the decode of the file reads zero there, as in the crate's own vector,
/// though the buffer it takes last held another image.
#[test]
fn a_gif_index_beyond_its_colour_table_reads_zero() {
    static MINE: BufferPool = BufferPool::new();
    buffer_pool::with_pool(&MINE, || {
        let file = a_gif_beyond_its_colour_table();
        let expected = general(&file);
        let mut used = buffer_pool::take(expected.as_bytes().len());
        used.fill(0x17);
        let address = used.as_ptr();
        buffer_pool::recycle(used);
        let decoded =
            decode_image_bytes(&file, None, TransparencyPolicy::KeepAlpha).expect("decoded");
        assert_eq!(decoded.as_bytes().as_ptr(), address, "the returned buffer");
        assert_eq!(decoded.color(), expected.color());
        assert!(decoded.as_bytes() == expected.as_bytes());
        assert!(decoded.as_bytes().iter().all(|&byte| byte == 0));
    });
}
