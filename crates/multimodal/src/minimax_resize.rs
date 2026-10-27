//! MiniMax-M3 image source-size policy, shared by router and Rust worker fetching.
use std::sync::Arc;

use image::{
    error::{ParameterError, ParameterErrorKind},
    DynamicImage, ImageError,
};

use crate::{
    types::{ImageFrame, ImageSize},
    vision::transforms::resize_bicubic_torch_aa,
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
        return Err(MediaConnectorError::InvalidMaxLongSidePixel {
            value: cap,
            factor: 28,
        });
    }
    if width == 0 || height == 0 {
        return Err(MediaConnectorError::Image(ImageError::Parameter(
            ParameterError::from_kind(ParameterErrorKind::DimensionMismatch),
        )));
    }
    let long = width.max(height);
    let short = width.min(height);
    let scale = if long > cap {
        f64::from(cap) / f64::from(long)
    } else if short < MIN_SHORT_SIDE {
        f64::from(MIN_SHORT_SIDE) / f64::from(short)
    } else {
        1.0
    };
    let align =
        |side: u32| ((f64::from(side) * scale / 28.0).round_ties_even().max(1.0) as u64) * 28;
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
    } else {
        resize_bicubic_torch_aa(&image, width, height)
    }
}

/// Include effective geometry in the pixel-cache identity.
fn resized_hash(hash: &str, width: u32, height: u32) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"minimax-m3-source-resize-v1");
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
    let size = frame.size();
    let (w, h) = dimensions(size.width, size.height, cap)?;
    budget(u64::from(w) * u64::from(h), IMAGE_MAX_TOTAL_PIXELS, "image")?;
    Ok(Arc::new(
        ImageFrame::encoded(
            frame.raw_bytes.clone(),
            frame.detail,
            frame.source.clone(),
            resized_hash(&frame.hash, w, h),
            ImageSize::new(w, h),
            None,
        )
        .with_transparency(frame.transparency())
        .with_minimax_resize(),
    ))
}
