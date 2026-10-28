//! SGLang day-0 aligns video frames to the nearest grid with Pillow bicubic resizing.
use ndarray::{Array2, Array3};

use super::MiniMaxM3VisionProcessor;
use crate::{
    encoder_inputs::{EncoderInput, ModelSpecificValue, PreprocessedEncoderInputs},
    types::RgbFrameRef,
    vision::{
        processor::VideoClipOptions,
        transforms::{resize_bicubic_pil_rgb, TransformError},
    },
};

impl MiniMaxM3VisionProcessor {
    /// Mirror SGLang `get_video_tensor` followed by the checkpoint HF processor.
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
        let norm = self.inner.normalization();
        let mut tensors = Vec::with_capacity(frames.len() + self.inner.temporal_patch_size());
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
            tensors.push(Array3::from_shape_fn((3, out_h, out_w), |(c, y, x)| {
                (f32::from(pixels.as_raw()[(y * out_w + x) * 3 + c]) * norm.scale - norm.offset[c])
                    / norm.divisor[c]
            }));
        }
        let temporal = self.inner.temporal_patch_size();
        while tensors.len() % temporal != 0 {
            let last = tensors.last().ok_or(TransformError::EmptyBatch)?.clone();
            tensors.push(last);
        }
        let (t, h, w) = self.inner.calculate_grid_thw(out_h, out_w, tensors.len());
        let mut patches = Vec::new();
        self.inner
            .patchify_video_into(&tensors, t, h, w, &mut patches)?;
        let features = 3 * temporal * self.inner.patch_size() * self.inner.patch_size();
        let array = Array2::from_shape_vec((t * h * w, features), patches)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        Ok(PreprocessedEncoderInputs::new(
            EncoderInput::from(array),
            vec![self.inner.calculate_tokens_from_grid(t, h, w)],
            vec![(width, height)],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![t as i64, h as i64, w as i64], 1, 3),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![(t * h * w) as i64]),
        )
        .with_extra(
            "video_second_per_grid",
            ModelSpecificValue::Tensor {
                data: vec![(temporal as f64 / clip.sample_fps) as f32],
                shape: vec![1],
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::{PreProcessorConfig, VisionPreProcessor};

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
