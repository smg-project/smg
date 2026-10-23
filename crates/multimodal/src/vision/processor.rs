//! Model-owned vision processor trait.
//!
//! Shared encoder output types live in [`crate::encoder_inputs`] and are re-exported
//! here for compatibility.

use image::DynamicImage;

use super::transforms::TransformError;
pub use crate::encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
use crate::{encoder_inputs::EncoderDtype, types::RgbFrameRef};

/// Helper to extract a dimension from encoder_input given an ndim-dependent axis index.
/// Returns `Err` if the ndim is not 4 or 5.
fn dim_for_ndim(
    ndim: usize,
    axis_4d: usize,
    axis_5d: usize,
    shape: &[usize],
) -> Result<usize, TransformError> {
    match ndim {
        4 => Ok(shape[axis_4d]),
        5 => Ok(shape[axis_5d]),
        _ => Err(TransformError::InvalidShape {
            expected: format!("4D or 5D encoder_input tensor, got {ndim}D"),
            actual: shape.to_vec(),
        }),
    }
}

impl PreprocessedEncoderInputs {
    /// Get the number of channels.
    ///
    /// For 4D tensors `[B, C, H, W]`, returns `shape[1]`.
    /// For 5D tensors `[B, N, C, H, W]` (Phi3-Vision), returns `shape[2]`.
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn channels(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 1, 2, self.encoder_input.shape())
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors `[B, C, H, W]`, returns `shape[2]`.
    /// For 5D tensors `[B, N, C, H, W]` (Phi3-Vision), returns `shape[3]`.
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn height(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 2, 3, self.encoder_input.shape())
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors `[B, C, H, W]`, returns `shape[3]`.
    /// For 5D tensors `[B, N, C, H, W]` (Phi3-Vision), returns `shape[4]`.
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn width(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 3, 4, self.encoder_input.shape())
    }
}

/// Request-scoped inputs to one image preprocessing call, beside the
/// configuration the processor resolved at construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisionPreprocessingContext {
    /// Context tokens left for the call's images: the engine's context length
    /// less the prompt's text tokens. A processor that sizes images by it keeps
    /// back its own reserve and shares the rest among the call's images, so a
    /// caller that sets it preprocesses a request's images in one call.
    /// `None` sizes each image as if it were alone.
    pub token_budget: Option<usize>,
}

/// What a video preprocessor reads about one clip besides its frames.
///
/// Both are set per request (by the media connector's sampling and the
/// caller's `max_long_side_pixel`), not by the checkpoint, so they are
/// arguments of each call rather than part of the configuration a processor
/// resolves at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoClipOptions {
    /// The rate the clip's frames were sampled at, in frames per second, as
    /// the double the request or the checkpoint named it.
    pub sample_fps: f64,
    /// The long-side cap the caller asked for, when it named one; the frames
    /// were already scaled to it.
    pub max_long_side_pixel: Option<u32>,
}

impl VideoClipOptions {
    /// A clip sampled at `sample_fps` with no long-side cap.
    pub fn sampled_at(sample_fps: f64) -> Self {
        Self {
            sample_fps,
            max_long_side_pixel: None,
        }
    }
}

/// Trait for model-specific vision preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline. An instance is built by its
/// model's spec ([`ModelProcessorSpec::vision_processor`](crate::ModelProcessorSpec::vision_processor))
/// for one modality's preprocessor config and retains the parameters it
/// resolved from it; request methods receive only the media and what is set
/// per request.
pub trait VisionPreProcessor: Send + Sync {
    /// Whether preprocessing individual images and concatenating their outputs
    /// is equivalent to preprocessing the whole batch, including metadata.
    ///
    /// Opt in only when resize, padding and normalization are independent of
    /// other images. An output field layout alone does not guarantee this.
    fn supports_per_image_preprocessing(&self) -> bool {
        false
    }

    /// Whether this processor produces the image's own bytes
    /// ([`EncoderInput::U8`](crate::encoder_inputs::EncoderInput::U8)), so
    /// a destination that normalizes on device can be served raw pixels
    /// exactly; `false` means it produces normalized floats.
    fn emits_pixel_bytes(&self) -> bool {
        false
    }

    /// Whether this processor takes an image that carries an alpha channel as
    /// decoded and composites or drops the channel itself, because its
    /// reference is handed the image as stored by the serving engines'
    /// loaders. A frame for such a processor is decoded under
    /// [`TransparencyPolicy::KeepAlpha`](crate::TransparencyPolicy::KeepAlpha)
    /// whatever the destination engine's own rule.
    fn keeps_alpha(&self) -> bool {
        false
    }

    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images to preprocess
    ///
    /// # Returns
    /// Preprocessed encoder inputs ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError>;

    /// [`preprocess`](Self::preprocess) under a request's context. A processor
    /// whose output does not depend on the request keeps this default, which
    /// ignores the context.
    fn preprocess_with_context(
        &self,
        images: &[DynamicImage],
        _context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess(images)
    }

    /// Preprocess a batch of images straight into `dtype`, the wire's dtype,
    /// when the processor can lay the patches out and normalize them in one
    /// pass (an [`EncoderInput::Wire`](crate::encoder_inputs::EncoderInput::Wire)
    /// result); `None` when it cannot, which leaves the caller
    /// [`preprocess`](Self::preprocess) and the per-destination write. The
    /// bytes a `Wire` carries are the bytes `preprocess` followed by a write
    /// in `dtype` would produce.
    fn preprocess_wire(
        &self,
        _images: &[DynamicImage],
        _dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        None
    }

    /// Preprocess one decoded video clip represented as sampled frames.
    ///
    /// Implementations that support video should emit the same primary
    /// `encoder_input` tensor shape used by the image path, plus video-specific
    /// model metadata such as `video_grid_thw`.
    fn preprocess_video(
        &self,
        _frames: &[DynamicImage],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support video preprocessing",
            self.model_name()
        )))
    }

    /// Preprocess one decoded video clip represented as borrowed RGB frame
    /// buffers. Implementations can override this to avoid materializing
    /// `DynamicImage` objects after media decode.
    fn preprocess_video_rgb(
        &self,
        _frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support RGB video preprocessing",
            self.model_name()
        )))
    }

    /// Calculate the number of vision tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed. A size
    /// its resize rule refuses is the error the preprocessing reports for it,
    /// never a count of zero or one taken from other settings; the arithmetic
    /// itself cannot fail.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError>;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;

    #[test]
    fn test_preprocessed_encoder_inputs_geometry_accessors() {
        let encoder_input = Array4::<f32>::zeros((2, 3, 336, 336));
        let inputs = PreprocessedEncoderInputs::new(
            encoder_input,
            vec![576, 576],
            vec![(640, 480), (800, 600)],
        );

        assert_eq!(inputs.channels().unwrap(), 3);
        assert_eq!(inputs.height().unwrap(), 336);
        assert_eq!(inputs.width().unwrap(), 336);
    }
}
