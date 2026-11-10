use super::super::*;

/// An icon file the way the serving engine's own image library writes
/// one from an image: one PNG entry per size in the image's own colour
/// type, each declared at 32 bits per pixel, the smallest first, a side
/// of 256 written as 0.
fn icon_of_png_entries(entries: &[image::DynamicImage]) -> Vec<u8> {
    let declared: Vec<_> = entries.iter().map(|entry| (entry, 0, 32)).collect();
    icon_of_declared_png_entries(&declared)
}

/// An icon file of PNG entries in the order given, each with the palette
/// colour count and bits per pixel its directory entry declares.
fn icon_of_declared_png_entries(entries: &[(&image::DynamicImage, u8, u16)]) -> Vec<u8> {
    let pngs: Vec<Vec<u8>> = entries
        .iter()
        .map(|(entry, _, _)| {
            let mut png = Vec::new();
            entry
                .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .expect("PNG encode");
            png
        })
        .collect();
    let mut ico = ICO_SIGNATURE.to_vec();
    ico.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * entries.len();
    for ((entry, colours, bits), png) in entries.iter().zip(&pngs) {
        let side = |pixels: u32| if pixels == 256 { 0 } else { pixels as u8 };
        ico.extend_from_slice(&[side(entry.width()), side(entry.height()), *colours, 0]);
        ico.extend_from_slice(&0u16.to_le_bytes());
        ico.extend_from_slice(&bits.to_le_bytes());
        ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
        ico.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += png.len();
    }
    for png in &pngs {
        ico.extend_from_slice(png);
    }
    ico
}

fn gradient(side: u32) -> image::RgbImage {
    image::RgbImage::from_fn(side, side, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    })
}

/// A second image of the same side as [`gradient`], every pixel different
/// from it.
fn inverse_gradient(side: u32) -> image::RgbImage {
    let mut image = gradient(side);
    image
        .pixels_mut()
        .for_each(|pixel| pixel.0 = pixel.0.map(|channel| 255 - channel));
    image
}

/// The RGB PNG entry of an icon saved from an RGB image decodes to the
/// PNG's pixels, as the serving engine's loader reads it; the `image`
/// crate's own ICO decoder refuses the entry for not being RGBA. The
/// header read names the same entry.
#[test]
fn an_icons_rgb_png_entry_decodes_as_the_png() {
    let largest = gradient(256);
    let ico = icon_of_png_entries(&[
        image::DynamicImage::ImageRgb8(gradient(16)),
        image::DynamicImage::ImageRgb8(largest.clone()),
    ]);
    let image = decode_image_bytes(&ico, None, TransparencyPolicy::CompositeOverWhite)
        .expect("an icon whose entry is an RGB PNG");
    assert_eq!((image.width(), image.height()), (256, 256));
    assert_eq!(image.to_rgb8().as_raw(), largest.as_raw());
    assert_eq!(
        encoded_image_size(&ico, None).expect("the icon's header"),
        crate::types::ImageSize::new(256, 256)
    );
}

/// An RGBA entry, the one form the crate's decoder took already, keeps
/// its pixels and its alpha on the same path.
#[test]
fn an_icons_rgba_png_entry_keeps_its_pixels() {
    let rgba = image::RgbaImage::from_fn(32, 32, |x, y| {
        image::Rgba([(x * 8) as u8, (y * 8) as u8, 7, (x * y) as u8])
    });
    let ico = icon_of_png_entries(&[image::DynamicImage::ImageRgba8(rgba.clone())]);
    let image = decode_image_bytes(&ico, None, TransparencyPolicy::KeepAlpha)
        .expect("an icon whose entry is an RGBA PNG");
    assert_eq!(image.to_rgba8().as_raw(), rgba.as_raw());
}

/// The largest entry is the one decoded wherever the directory lists it;
/// listed first, its PNG runs on into the next entry's bytes, which the
/// PNG decoder leaves alone past its own end.
#[test]
fn the_largest_entry_is_the_one_decoded_wherever_it_is_listed() {
    let largest = gradient(48);
    for entries in [
        vec![
            image::DynamicImage::ImageRgb8(gradient(16)),
            image::DynamicImage::ImageRgb8(largest.clone()),
        ],
        vec![
            image::DynamicImage::ImageRgb8(largest.clone()),
            image::DynamicImage::ImageRgb8(gradient(16)),
        ],
    ] {
        let ico = icon_of_png_entries(&entries);
        let image = decode_image_bytes(&ico, None, TransparencyPolicy::DropAlpha)
            .expect("an icon of two RGB PNG entries");
        assert_eq!(image.to_rgb8().as_raw(), largest.as_raw());
    }
}

/// The entry is the one the serving engine's loader opens, where the
/// crate's own order would name another: the largest area whatever its
/// declared depth, the lowest depth among equal areas (a palette's colour
/// count standing for its bits when no depth is declared), the first
/// listed on a full tie.
#[test]
fn the_entry_is_the_loaders_choice_on_an_icon_that_mixes_depths() {
    let large = image::DynamicImage::ImageRgb8(gradient(256));
    let small = image::DynamicImage::ImageRgb8(gradient(48));
    let first = image::DynamicImage::ImageRgb8(gradient(32));
    let second = image::DynamicImage::ImageRgb8(inverse_gradient(32));
    for (entries, expected) in [
        // The crate takes the 32-bit entry; the loader the larger one.
        (vec![(&small, 0, 32), (&large, 0, 0)], &large),
        // Equal areas: the lower declared depth, listed second.
        (vec![(&first, 0, 32), (&second, 0, 24)], &second),
        // Equal areas: sixteen palette colours are four bits, the lowest.
        (vec![(&first, 0, 32), (&second, 16, 0)], &second),
        // A palette count of one reads as 256 colours, like no depth.
        (vec![(&first, 1, 0), (&second, 0, 32)], &second),
        // A full tie: the first listed.
        (vec![(&first, 0, 32), (&second, 0, 32)], &first),
    ] {
        let ico = icon_of_declared_png_entries(&entries);
        let image = decode_image_bytes(&ico, None, TransparencyPolicy::DropAlpha)
            .expect("an icon of two RGB PNG entries");
        assert_eq!(image.to_rgb8().as_raw(), expected.to_rgb8().as_raw());
    }
}

/// A PNG entry runs from its offset to the end of the file, as both
/// readers take it: a declared length shorter than the PNG, or one past
/// the end of the file, cuts nothing.
#[test]
fn a_png_entrys_declared_length_does_not_cut_it() {
    let only = gradient(24);
    let ico = icon_of_png_entries(&[image::DynamicImage::ImageRgb8(only.clone())]);
    let png = &ico[6 + 16..];
    for declared in [(png.len() / 2) as u32, 8, u32::MAX] {
        let mut ico = ico.clone();
        ico[6 + 8..6 + 12].copy_from_slice(&declared.to_le_bytes());
        assert_eq!(ico_png_entry(&ico), Some(png));
        let image = decode_image_bytes(&ico, None, TransparencyPolicy::DropAlpha)
            .expect("an icon whose entry declares the wrong length");
        assert_eq!(image.to_rgb8().as_raw(), only.as_raw());
    }
}

/// Anything but an icon whose chosen entry is a PNG inside the file is
/// left to the crate's decoder: another container, a directory the file
/// cuts short, an entry past the end, a bitmap entry, a bitmap chosen
/// over a smaller PNG entry.
#[test]
fn only_a_png_entry_within_the_file_is_taken() {
    assert!(ico_png_entry(&PNG_SIGNATURE).is_none());
    assert!(ico_png_entry(&ICO_SIGNATURE).is_none());
    assert!(ico_png_entry(&[0, 0, 1, 0, 1, 0]).is_none());
    let mut ico = icon_of_png_entries(&[image::DynamicImage::ImageRgb8(gradient(8))]);
    assert!(ico_png_entry(&ico).is_some());
    assert!(ico_png_entry(&ico[..6 + 16 - 1]).is_none());
    let mut past_the_end = ico.clone();
    past_the_end[6 + 12..6 + 16].copy_from_slice(&(ico.len() as u32).to_le_bytes());
    assert!(ico_png_entry(&past_the_end).is_none());
    ico[6 + 16] = b'B';
    assert!(ico_png_entry(&ico).is_none());
    let mut bitmap_on_top = icon_of_png_entries(&[
        image::DynamicImage::ImageRgb8(gradient(16)),
        image::DynamicImage::ImageRgb8(gradient(48)),
    ]);
    let larger_entry = usize::try_from(u32::from_le_bytes(
        bitmap_on_top[6 + 16 + 12..6 + 16 + 16].try_into().unwrap(),
    ))
    .unwrap();
    bitmap_on_top[larger_entry] = b'B';
    assert!(ico_png_entry(&bitmap_on_top).is_none());
}
