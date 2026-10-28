//! MiniMax-M3 image source-size policy, shared by router and Rust worker fetching.
use std::sync::Arc;

use image::{
    error::{ParameterError, ParameterErrorKind},
    DynamicImage, ImageError,
};

use crate::{
    types::{ImageFrame, ImageSize},
    vision::transforms::{resize_bicubic_torch_aa, round_half_to_even},
    MediaConnectorError,
};

/// Short-side floor, applied only when the source does not exceed the long-side cap.
pub const MIN_SHORT_SIDE: u32 = 112;
/// Hard limit on aligned image pixels, checked before decoding.
pub const IMAGE_MAX_TOTAL_PIXELS: u64 = 12_845_056;
/// Default image long-side cap.
pub const IMAGE_DEFAULT_LONG_SIDE: u32 = 2016;
/// Internal processor marker for images whose source geometry was already planned.
pub const RESIZED_KEY: &str = "minimax_media_resized";
/// Internal selector for the SGLang MiniMax M3.1 day-0 resize contract.
pub const SGLANG_RESIZE_KEY: &str = "minimax_sglang_resize";

#[cfg(test)]
#[path = "minimax_resize_tests.rs"]
mod tests;

/// Branch on original dimensions, then use the model's nearest patch-grid
/// alignment (ties to even, like Python round). Never round the input cap.
pub(crate) fn dimensions(
    width: u32,
    height: u32,
    cap: u32,
) -> Result<(u32, u32), MediaConnectorError> {
    if cap == 0 {
        return Err(MediaConnectorError::InvalidImageLongSideCap { value: cap });
    }
    if width == 0 || height == 0 {
        return Err(MediaConnectorError::Image(ImageError::Parameter(
            ParameterError::from_kind(ParameterErrorKind::DimensionMismatch),
        )));
    }
    let long = width.max(height);
    let short = width.min(height);
    let (numerator, denominator) = if long > cap {
        (cap, long)
    } else if short < MIN_SHORT_SIDE {
        (MIN_SHORT_SIDE, short)
    } else {
        (1, 1)
    };
    // A float scale can move an exact half-grid tie above or below the tie.
    // u32 * u32 fits u64, including the largest positive source and cap.
    let align = |side: u32| {
        let n = u64::from(side) * u64::from(numerator);
        let d = u64::from(denominator) * 28;
        let q = n / d;
        let r = n % d;
        let rounded = q + u64::from(r * 2 > d || (r * 2 == d && q % 2 == 1));
        rounded.max(1) * 28
    };
    let w = align(width);
    let h = align(height);
    if w > u64::from(u32::MAX) || h > u64::from(u32::MAX) {
        return Err(MediaConnectorError::PixelBudgetExceeded {
            modality: "image",
            pixels: w.saturating_mul(h),
            limit: IMAGE_MAX_TOTAL_PIXELS,
        });
    }
    Ok((w as u32, h as u32))
}

/// Match SGLang day-0 nearest-grid alignment with a bounded short-side floor.
pub(crate) fn sglang_dimensions(
    width: u32,
    height: u32,
    cap: u32,
) -> Result<(u32, u32), MediaConnectorError> {
    if !(112..=3584).contains(&cap) || !cap.is_multiple_of(28) {
        return Err(MediaConnectorError::InvalidImageLongSideCap { value: cap });
    }
    if width == 0
        || height == 0
        || f64::from(width.max(height)) / f64::from(width.min(height)) > 200.0
    {
        return Err(MediaConnectorError::Image(ImageError::Parameter(
            ParameterError::from_kind(ParameterErrorKind::DimensionMismatch),
        )));
    }
    let long = f64::from(width.max(height));
    let short = f64::from(width.min(height));
    let ratio = if long > f64::from(cap) {
        f64::from(cap) / long
    } else if short < 112.0 {
        (112.0 / short).min(f64::from(cap) / long)
    } else {
        1.0
    };
    let align = |side| (round_half_to_even(f64::from(side) * ratio / 28.0) as u32).max(1) * 28;
    Ok((align(width), align(height)))
}

/// Reject oversized output before allocating pixels.
fn budget(pixels: u64, limit: u64, modality: &'static str) -> Result<(), MediaConnectorError> {
    if pixels > limit {
        return Err(MediaConnectorError::PixelBudgetExceeded {
            modality,
            pixels,
            limit,
        });
    }
    Ok(())
}

/// Resize directly to the planned patch grid with the M3 uint8 bicubic kernel.
pub(crate) fn resize_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    if (image.width(), image.height()) == (width, height) {
        image
    } else if image.color().has_alpha() {
        // Resize straight RGB and alpha independently: the model consumes RGB,
        // while the shared frame must still honor KeepAlpha for other readers.
        let rgba = image.to_rgba8();
        let alpha = image::RgbImage::from_fn(image.width(), image.height(), |x, y| {
            image::Rgb([rgba.get_pixel(x, y)[3]; 3])
        });
        let rgb = resize_bicubic_torch_aa(&image, width, height).into_rgb8();
        let alpha =
            resize_bicubic_torch_aa(&DynamicImage::ImageRgb8(alpha), width, height).into_rgb8();
        DynamicImage::ImageRgba8(image::RgbaImage::from_fn(width, height, |x, y| {
            let p = rgb.get_pixel(x, y);
            image::Rgba([p[0], p[1], p[2], alpha.get_pixel(x, y)[0]])
        }))
    } else {
        resize_bicubic_torch_aa(&image, width, height)
    }
}

/// Include effective geometry in the pixel-cache identity.
fn resized_hash(hash: &str, width: u32, height: u32, sglang: bool) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"minimax-m3-source-resize-v1");
    if sglang {
        hasher.update(b"sglang-pillow-day0");
    }
    hasher.update(hash.as_bytes());
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Plan from the original header while retaining lazy decode and transparency.
pub(crate) fn image(
    frame: Arc<ImageFrame>,
    cap: u32,
) -> Result<Arc<ImageFrame>, MediaConnectorError> {
    image_for_runtime(frame, cap, false)
}

/// Plan the engine's source geometry while retaining lazy image decoding.
pub(crate) fn image_for_runtime(
    frame: Arc<ImageFrame>,
    cap: u32,
    sglang: bool,
) -> Result<Arc<ImageFrame>, MediaConnectorError> {
    let size = frame.size();
    let (w, h) = if sglang {
        sglang_dimensions(size.width, size.height, cap)?
    } else {
        dimensions(size.width, size.height, cap)?
    };
    budget(u64::from(w) * u64::from(h), IMAGE_MAX_TOTAL_PIXELS, "image")?;
    Ok(Arc::new(
        ImageFrame::encoded(
            frame.raw_bytes.clone(),
            frame.detail,
            frame.source.clone(),
            resized_hash(&frame.hash, w, h, sglang),
            ImageSize::new(w, h),
            None,
        )
        .with_transparency(frame.transparency())
        .with_minimax_resize()
        .with_minimax_sglang_resize(sglang),
    ))
}
