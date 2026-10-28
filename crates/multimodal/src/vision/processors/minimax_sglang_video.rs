//! SGLang day-0 aligns video frames to the nearest grid with Pillow bicubic resizing.

use super::MiniMaxM3VisionProcessor;
use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    types::RgbFrameRef,
    vision::{
        processor::{VideoClipOptions, VisionPreProcessor},
        transforms::{resize_bicubic_pil_rgb, TransformError},
    },
};

impl MiniMaxM3VisionProcessor {
    /// Mirror SGLang `get_video_tensor` followed by the builtin day-0 processor.
    pub(super) fn preprocess_sglang_video(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let first = frames.first().ok_or(TransformError::EmptyBatch)?;
        let (width, height) = (first.width, first.height);
        if width == 0
            || height == 0
            || frames
                .iter()
                .any(|f| f.width != width || f.height != height)
        {
            return Err(TransformError::ShapeError(
                "MiniMax video frames must share nonzero dimensions".into(),
            ));
        }
        if !clip.sample_fps.is_finite() || clip.sample_fps <= 0.0 {
            return Err(TransformError::ShapeError(
                "MiniMax video fps must be positive and finite".into(),
            ));
        }
        let factor = self.inner.get_factor();
        let cap = self.inner.video_max_pixels().isqrt() / factor * factor;
        if cap == 0 {
            return Err(TransformError::ShapeError(
                "MiniMax video pixel budget is below one patch group".into(),
            ));
        }
        let cap = clip.max_long_side_pixel.unwrap_or(cap as u32);
        let (stage_w, stage_h) = crate::minimax_resize::sglang_dimensions(width, height, cap)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        let total_pixels = u64::from(stage_w) * u64::from(stage_h) * frames.len() as u64;
        if total_pixels > 301_056_000 {
            return Err(TransformError::ShapeError(format!(
                "MiniMax video has {total_pixels} pixels, exceeding 301056000"
            )));
        }
        let (stage_w, stage_h) = (stage_w as usize, stage_h as usize);
        let (out_h, out_w) = if self.inner.resizes() {
            self.inner
                .smart_resize_video(frames.len(), stage_h, stage_w)?
        } else {
            (stage_h, stage_w)
        };
        let mut resized = Vec::with_capacity(frames.len());
        for frame in frames {
            let mut pixels =
                resize_bicubic_pil_rgb(frame.data, width, height, stage_w as u32, stage_h as u32)?;
            if self.inner.resizes() && (stage_w != out_w || stage_h != out_h) {
                pixels = resize_bicubic_pil_rgb(
                    pixels.as_raw(),
                    stage_w as u32,
                    stage_h as u32,
                    out_w as u32,
                    out_h as u32,
                )?;
            }
            resized.push(pixels);
        }
        let frames: Vec<_> = resized
            .iter()
            .map(|frame| RgbFrameRef {
                width: frame.width(),
                height: frame.height(),
                data: frame.as_raw(),
            })
            .collect();
        // The native loader already selected the grid and Pillow pixels. Keep
        // bytes through patchification; normalize only at the wire boundary.
        let mut output = self
            .inner
            .clone()
            .with_aligned_video_frames(out_w * out_h)
            .preprocess_video_rgb(&frames, clip)?;
        output.item_sizes = vec![(width, height)];
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encoder_inputs::{EncoderInput, ModelSpecificValue},
        vision::{PreProcessorConfig, VisionPreProcessor},
    };

    /// Compare frame alignment, patch order and Pillow pixels with SGLang/HF output.
    #[test]
    fn sglang_video_matches_independent_torch_reference() {
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/golden/minimax_sglang_video.json"
        ))
        .unwrap();
        let mut config = PreProcessorConfig {
            do_resize: Some(false),
            ..PreProcessorConfig::default()
        };
        config
            .extra
            .insert(crate::minimax_resize::SGLANG_RESIZE_KEY.into(), true.into());
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        let frames: Vec<_> = golden["sampled_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| {
                let frame = i.as_u64().unwrap() as u32;
                image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(320, 240, |x, y| {
                    image::Rgb(std::array::from_fn(|c| {
                        ((x * 17 + y * 29 + c as u32 * 71 + frame * 13) % 256) as u8
                    }))
                }))
            })
            .collect();
        let output = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(1.0))
            .unwrap();
        assert!(matches!(output.encoder_input, EncoderInput::U8 { .. }));
        assert_eq!(output.item_sizes, [(320, 240)]);
        assert_eq!(output.encoder_input.shape(), [792, 1176]);
        assert_eq!(output.feature_token_counts, [198]);
        let ModelSpecificValue::IntTensor { data, shape } =
            &output.model_specific["video_grid_thw"]
        else {
            panic!("integer video grid required")
        };
        assert_eq!(data, &[2, 18, 22]);
        assert_eq!(shape, &[1, 3]);
        let flat = output.encoder_input.flat_f32();
        for (i, v) in golden["indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(golden["values"].as_array().unwrap())
        {
            let index = i.as_u64().unwrap() as usize;
            let expected = v.as_f64().unwrap() as f32;
            assert!(
                (flat[index] - expected).abs() < 2e-5,
                "pixel {index}: {} versus {expected}",
                flat[index]
            );
        }
        let rgb: Vec<_> = frames.iter().map(image::DynamicImage::to_rgb8).collect();
        let borrowed: Vec<_> = rgb
            .iter()
            .map(|f| RgbFrameRef {
                width: f.width(),
                height: f.height(),
                data: f.as_raw(),
            })
            .collect();
        let fast = processor
            .preprocess_video_rgb(&borrowed, &VideoClipOptions::sampled_at(1.0))
            .unwrap();
        assert_eq!(fast.encoder_input.flat_f32(), flat);
    }
    /// An explicit loader cap survives byte patchification without an area resize.
    #[test]
    fn sglang_explicit_video_cap_keeps_source_size_and_loader_grid() {
        let mut config = PreProcessorConfig {
            do_resize: Some(false),
            ..PreProcessorConfig::default()
        };
        config
            .extra
            .insert(crate::minimax_resize::SGLANG_RESIZE_KEY.into(), true.into());
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        let output = processor
            .preprocess_video(
                &[image::DynamicImage::new_rgb8(1920, 1080)],
                &VideoClipOptions {
                    sample_fps: 1.0,
                    max_long_side_pixel: Some(1008),
                },
            )
            .unwrap();
        assert_eq!(output.item_sizes, [(1920, 1080)]);
        assert_eq!(output.encoder_input.shape(), [2880, 1176]);
        assert_eq!(output.feature_token_counts, [720]);
        assert!(matches!(output.encoder_input, EncoderInput::U8 { .. }));
    }

    /// The combined HF processor disables a second resize after loader alignment.
    #[test]
    fn sglang_small_video_keeps_loader_grid() {
        let mut config = PreProcessorConfig {
            do_resize: Some(false),
            ..PreProcessorConfig::default()
        };
        config
            .extra
            .insert(crate::minimax_resize::SGLANG_RESIZE_KEY.into(), true.into());
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        let output = processor
            .preprocess_video(
                &[image::DynamicImage::new_rgb8(14, 14)],
                &VideoClipOptions::sampled_at(1.0),
            )
            .unwrap();
        assert_eq!(output.encoder_input.shape(), [64, 1176]);
        assert_eq!(output.feature_token_counts, [16]);
    }
}
