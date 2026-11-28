use super::super::*;

/// The blend the composite stands for, one channel at one alpha.
fn blended(value: u8, alpha: u8) -> u8 {
    let (value, alpha) = (u32::from(value), u32::from(alpha));
    u8::try_from((value * alpha + 255 * (255 - alpha) + 127) / 255).expect("a channel value")
}

/// What the composite owes for `rgba`, pixel by pixel.
fn expected(rgba: &image::RgbaImage) -> Vec<u8> {
    rgba.pixels()
        .flat_map(|px| [px.0[0], px.0[1], px.0[2]].map(|value| blended(value, px.0[3])))
        .collect()
}

/// An opaque image whose colour changes from pixel to pixel.
fn opaque(width: u32, height: u32) -> image::RgbaImage {
    image::RgbaImage::from_fn(width, height, |x, y| {
        let [low, high, ..] = (y * width + x).to_le_bytes();
        image::Rgba([low, high, low ^ high, u8::MAX])
    })
}

#[test]
fn an_opaque_image_keeps_its_stored_colour() {
    // One pixel, the sizes around one block of the opacity read, several
    // blocks with a partial one behind them, and a size the pool serves.
    for (width, height) in [
        (1, 1),
        (1023, 1),
        (1024, 1),
        (1025, 1),
        (257, 13),
        (320, 320),
    ] {
        let rgba = opaque(width, height);
        let stored: Vec<u8> = rgba
            .pixels()
            .flat_map(|px| [px.0[0], px.0[1], px.0[2]])
            .collect();
        let out = composite_over_white(&rgba);
        assert_eq!(out.as_raw(), &stored, "{width}x{height}");
        assert_eq!(out.as_raw(), &expected(&rgba), "{width}x{height}");
    }
}

#[test]
fn one_transparent_pixel_anywhere_reads_the_blend_at_every_pixel() {
    let (width, height) = (257, 13);
    let count = width * height;
    // The first pixel and the last, the pixels around a block's edge, and
    // one of the partial block at the end; an alpha one step under full too.
    for at in [0, 1, 1023, 1024, 1025, count - 2, count - 1] {
        for alpha in [0, 1, 128, 254] {
            let mut rgba = opaque(width, height);
            rgba.get_pixel_mut(at % width, at / width).0[3] = alpha;
            let out = composite_over_white(&rgba);
            assert_eq!(
                out.as_raw(),
                &expected(&rgba),
                "pixel {at} at alpha {alpha}"
            );
        }
    }
}

#[test]
fn an_opaque_lead_keeps_its_stored_colour_and_the_rest_reads_the_blend() {
    let (width, height) = (257, 13);
    let count = width * height;
    // Two opaque blocks, then the first transparent pixel at a block's edge,
    // inside the block and in the partial block at the end; every pixel
    // behind it is transparent to a degree of its own.
    for first in [2048, 2049, 3071, 3072, count - 1] {
        let mut rgba = opaque(width, height);
        for at in first..count {
            let [alpha, ..] = (at % 255).to_le_bytes();
            rgba.get_pixel_mut(at % width, at / width).0[3] = alpha;
        }
        let stored: Vec<u8> = rgba
            .pixels()
            .flat_map(|px| [px.0[0], px.0[1], px.0[2]])
            .collect();
        let lead = 3 * usize::try_from(first).expect("a pixel index");
        let out = composite_over_white(&rgba);
        assert_eq!(out.as_raw()[..lead], stored[..lead], "the lead of {first}");
        assert_eq!(out.as_raw(), &expected(&rgba), "transparent from {first}");
    }
}
