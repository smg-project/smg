//! Vision processor trait and registry.
//!
//! Shared encoder output types live in [`crate::encoder_inputs`] and are re-exported
//! here for compatibility.

use std::collections::HashMap;

use image::DynamicImage;

use super::{preprocessor_config::PreProcessorConfig, transforms::TransformError};
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
    /// For 4D tensors [B, C, H, W], returns shape[1].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[2].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn channels(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 1, 2, self.encoder_input.shape())
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[2].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[3].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn height(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 2, 3, self.encoder_input.shape())
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[3].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[4].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn width(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 3, 4, self.encoder_input.shape())
    }
}

/// Trait for model-specific vision preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline.
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
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed encoder inputs ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError>;

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
        _config: &PreProcessorConfig,
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
        _config: &PreProcessorConfig,
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
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support RGB video preprocessing",
            self.model_name()
        )))
    }

    /// Calculate the number of vision tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed. A
    /// config the implementation cannot read, or a size its resize rule
    /// refuses, is the error the preprocessing reports for it, never a count
    /// of zero or one taken from other settings; the arithmetic itself
    /// cannot fail.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<usize, TransformError>;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        config.get_target_size()
    }
}

/// Registry of available vision processors.
pub struct VisionProcessorRegistry {
    processors: HashMap<String, Box<dyn VisionPreProcessor>>,
}

impl VisionProcessorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a model pattern.
    pub fn register(&mut self, pattern: impl Into<String>, processor: Box<dyn VisionPreProcessor>) {
        // Matching is case-insensitive on both sides; lowercase the pattern
        // once here rather than every pattern on every lookup.
        self.processors
            .insert(pattern.into().to_lowercase(), processor);
    }

    /// Find a processor for the given model ID, falling back to model_type.
    ///
    /// Matches by substring containment (case-insensitive).
    pub fn find(
        &self,
        model_id: &str,
        model_type: Option<&str>,
    ) -> Option<&dyn VisionPreProcessor> {
        self.find_in_candidate(model_id)
            .or_else(|| model_type.and_then(|mt| self.find_in_candidate(mt)))
    }

    fn find_in_candidate(&self, candidate: &str) -> Option<&dyn VisionPreProcessor> {
        let candidate = candidate.to_lowercase();
        self.processors
            .iter()
            .find(|(pattern, _)| candidate.contains(pattern.as_str()))
            .map(|(_, processor)| processor.as_ref())
    }

    /// Get list of supported model patterns.
    pub fn supported_patterns(&self) -> Vec<&str> {
        self.processors.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for VisionProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The constructor of a built-in processor, as the table of defaults holds it.
type ProcessorConstructor = fn() -> Box<dyn VisionPreProcessor>;

/// The built-in processors and the patterns that select them: one row per
/// processor family, the patterns a model id or model type is matched against
/// (case-insensitively, as substrings) with the processor they select.
/// [`VisionProcessorRegistry::with_defaults`] registers every row.
const DEFAULT_PROCESSORS: &[(&[&str], ProcessorConstructor)] = &[
    // DeepSeek-V4.1 vision variant (patch=14, 3x3 aligner downsample,
    // gray-padded contain fit): the HF model_type and the marketing id.
    (&["deepseek_v41", "deepseek-v4.1"], || {
        Box::new(super::processors::DeepseekV41Processor::new())
    }),
    // Nemotron-H Omni (the HF model_type and the released ids' spelling).
    (&["nemotronh_nano_omni", "nemotron-3-nano-omni"], || {
        Box::new(super::processors::NemotronOmniProcessor::new())
    }),
    // Gemma 4 (the HF model_type and the released ids' spelling).
    (&["gemma4", "gemma-4"], || {
        Box::new(super::processors::Gemma4Processor::new())
    }),
    (
        &[
            "glm-5.3-flash",
            "glm5.3-flash",
            "glm53_flash",
            "glm-5-next",
            "glm5-next",
            "glm5_next",
        ],
        || Box::new(super::processors::Glm53FlashProcessor::new()),
    ),
    // LLaVA-NeXT (v1.6+, anyres multi-crop).
    (&["llava-next", "llava_next", "llava-v1.6"], || {
        Box::new(super::processors::LlavaNextProcessor::new())
    }),
    // Standard LLaVA (v1.5, single-patch), under patterns specific enough not
    // to match LLaVA-NeXT model ids like "llava-v1.6-*".
    (&["llava-1.5", "llava-v1.5"], || {
        Box::new(super::processors::LlavaProcessor::new())
    }),
    // Muse Spark (the HF model_type and the released ids' spelling).
    (&["muse_spark", "muse-spark"], || {
        Box::new(super::processors::MuseSparkVisionProcessor::new())
    }),
    (&["qwen3-vl", "qwen3_vl"], || {
        Box::new(super::processors::Qwen3VLProcessor::new())
    }),
    // Qwen3-Omni uses the same patchification and normalization contract as
    // Qwen3-VL for its image and video towers.
    (&["qwen3-omni", "qwen3_omni"], || {
        Box::new(super::processors::Qwen3OmniVisionProcessor::new())
    }),
    // The Qwen3.5 family (and Qwen3.6: same arch) reuses Qwen3-VL
    // preprocessing.
    (&["qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6"], || {
        Box::new(super::processors::Qwen3VLProcessor::new())
    }),
    // Qwen4-Exp: the same vision tower as Qwen3.5.
    (&["qwen4_exp", "qwen4-exp"], || {
        Box::new(super::processors::Qwen3VLProcessor::new())
    }),
    // Inkling/TML multimodal model.
    (&["inkling"], || {
        Box::new(super::processors::InklingImageProcessor::new())
    }),
    // MiniMax-M3 preprocesses images the Qwen2-VL way but with its own pixel
    // bounds, so it gets a dedicated processor rather than reusing the
    // Qwen2-VL defaults.
    (&["minimax_m3", "minimax-m3", "minimax_m3_vl"], || {
        Box::new(super::processors::MiniMaxM3VisionProcessor::new())
    }),
    // Qwen2-VL (matches Qwen/Qwen2-VL-*, etc.) and Qwen2.5-VL, whose
    // preprocessing is identical.
    (
        &[
            "qwen2-vl",
            "qwen2_vl",
            "qwen2.5-vl",
            "qwen2_5-vl",
            "qwen2_5_vl",
        ],
        || Box::new(super::processors::Qwen2VLProcessor::new()),
    ),
    (&["phi-3-vision", "phi3-vision", "phi3_v"], || {
        Box::new(super::processors::Phi3VisionProcessor::new())
    }),
    (&["llama-4", "llama4"], || {
        Box::new(super::processors::Llama4VisionProcessor::new())
    }),
    // The Kimi MoonViT family. K3 shares the stack but ships a larger patch
    // budget and chessboard alpha compositing, so it gets its own processor
    // rather than reusing K2.5's.
    (&["kimi-k2", "kimi_k2"], || {
        Box::new(super::processors::KimiK25Processor::new())
    }),
    (&["kimi-k3", "kimi_k3"], || {
        Box::new(super::processors::KimiK3Processor::new())
    }),
];

impl VisionProcessorRegistry {
    /// Create a registry with every built-in processor registered, each under
    /// the patterns of its row in the table of defaults (`DEFAULT_PROCESSORS`).
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        for (patterns, constructor) in DEFAULT_PROCESSORS {
            for pattern in *patterns {
                registry.register(*pattern, constructor());
            }
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;
    use crate::vision::processors::LlavaProcessor;

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

    #[test]
    fn test_registry_with_defaults() {
        let registry = VisionProcessorRegistry::with_defaults();

        // Should find LLaVA processor
        assert!(registry.find("llava-hf/llava-1.5-7b-hf", None).is_some());
        assert!(registry.find("liuhaotian/llava-v1.5-7b", None).is_some());

        // Should find LLaVA-NeXT processor
        assert!(registry
            .find("llava-hf/llava-v1.6-mistral-7b-hf", None)
            .is_some());
        assert!(registry
            .find("lmms-lab/llava-next-interleave-qwen-7b", None)
            .is_some());

        // Get the processor and check model name
        let processor = registry.find("llava-hf/llava-1.5-7b-hf", None).unwrap();
        assert_eq!(processor.model_name(), "llava");

        let processor = registry
            .find("org/inkling-chat", None)
            .expect("Inkling model family");
        assert_eq!(processor.model_name(), "inkling");

        for (model_id, model_type) in [
            ("zai-org/GLM-5.3-Flash", None),
            ("internal/checkpoint", Some("glm53_flash")),
            ("internal/checkpoint", Some("glm5_next")),
        ] {
            let processor = registry
                .find(model_id, model_type)
                .expect("GLM-5.3-Flash processor");
            assert_eq!(processor.model_name(), "glm-5.3-flash");
        }
    }

    /// Every pattern of the table of defaults with the processor it selects,
    /// as the registry resolved them when each was registered by hand.
    #[test]
    fn with_defaults_resolves_every_pattern_to_its_processor() {
        let registry = VisionProcessorRegistry::with_defaults();
        let expected = [
            ("deepseek_v41", "deepseek_v41"),
            ("deepseek-v4.1", "deepseek_v41"),
            ("nemotronh_nano_omni", "nemotron_omni"),
            ("nemotron-3-nano-omni", "nemotron_omni"),
            ("gemma4", "gemma4"),
            ("gemma-4", "gemma4"),
            ("glm-5.3-flash", "glm-5.3-flash"),
            ("glm5.3-flash", "glm-5.3-flash"),
            ("glm53_flash", "glm-5.3-flash"),
            ("glm-5-next", "glm-5.3-flash"),
            ("glm5-next", "glm-5.3-flash"),
            ("glm5_next", "glm-5.3-flash"),
            ("llava-next", "llava-next"),
            ("llava_next", "llava-next"),
            ("llava-v1.6", "llava-next"),
            ("llava-1.5", "llava"),
            ("llava-v1.5", "llava"),
            ("muse_spark", "muse_spark"),
            ("muse-spark", "muse_spark"),
            ("qwen3-vl", "qwen3-vl"),
            ("qwen3_vl", "qwen3-vl"),
            ("qwen3-omni", "qwen3-omni"),
            ("qwen3_omni", "qwen3-omni"),
            ("qwen3.5", "qwen3-vl"),
            ("qwen3_5", "qwen3-vl"),
            ("qwen3.6", "qwen3-vl"),
            ("qwen3_6", "qwen3-vl"),
            ("qwen4_exp", "qwen3-vl"),
            ("qwen4-exp", "qwen3-vl"),
            ("inkling", "inkling"),
            ("minimax_m3", "minimax_m3"),
            ("minimax-m3", "minimax_m3"),
            ("minimax_m3_vl", "minimax_m3"),
            ("qwen2-vl", "qwen2-vl"),
            ("qwen2_vl", "qwen2-vl"),
            ("qwen2.5-vl", "qwen2-vl"),
            ("qwen2_5-vl", "qwen2-vl"),
            ("qwen2_5_vl", "qwen2-vl"),
            ("phi-3-vision", "phi3-vision"),
            ("phi3-vision", "phi3-vision"),
            ("phi3_v", "phi3-vision"),
            ("llama-4", "llama4-vision"),
            ("llama4", "llama4-vision"),
            ("kimi-k2", "kimi-k2.5"),
            ("kimi_k2", "kimi-k2.5"),
            ("kimi-k3", "kimi-k3"),
            ("kimi_k3", "kimi-k3"),
        ];
        assert_eq!(registry.supported_patterns().len(), expected.len());
        for (pattern, model_name) in expected {
            let processor = registry
                .find(pattern, None)
                .unwrap_or_else(|| panic!("{pattern} selects no processor"));
            assert_eq!(processor.model_name(), model_name, "{pattern}");
        }
    }

    #[test]
    fn test_registry_finds_gemma4_by_id_and_model_type() {
        let registry = VisionProcessorRegistry::with_defaults();
        for (model_id, model_type) in [
            ("google/gemma-4-26B-A4B-it", None),
            ("/models/local-checkpoint", Some("gemma4")),
        ] {
            let processor = registry
                .find(model_id, model_type)
                .expect("Gemma 4 vision processor");
            assert_eq!(processor.model_name(), "gemma4", "{model_id}");
        }
        assert!(registry
            .find("/models/local-checkpoint", Some("gemma3"))
            .is_none());
    }

    #[test]
    fn test_registry_separates_kimi_k25_and_k3() {
        // K3's pixel pipeline differs from K2.5's (larger patch budget, alpha
        // compositing), so resolving a K3 id to the K2.5 processor is a bug.
        let registry = VisionProcessorRegistry::with_defaults();

        for id in ["moonshotai/Kimi-K3", "moonshotai/Kimi_K3-Instruct"] {
            let processor = registry.find(id, None).expect("K3 vision processor");
            assert_eq!(processor.model_name(), "kimi-k3", "{id}");
        }
        for id in ["moonshotai/Kimi-K2.5", "moonshotai/Kimi_K2-VL"] {
            let processor = registry.find(id, None).expect("K2.5 vision processor");
            assert_eq!(processor.model_name(), "kimi-k2.5", "{id}");
        }

        // model_type fallback must split the same way.
        assert_eq!(
            registry
                .find("internal/checkpoint-final", Some("kimi_k3"))
                .expect("K3 by model_type")
                .model_name(),
            "kimi-k3"
        );
        assert_eq!(
            registry
                .find("internal/checkpoint-final", Some("kimi_k2"))
                .expect("K2.5 by model_type")
                .model_name(),
            "kimi-k2.5"
        );
    }

    #[test]
    fn test_registry_find() {
        let mut registry = VisionProcessorRegistry::new();

        // Create a mock processor using LlavaProcessor
        registry.register("test-model", Box::new(LlavaProcessor::new()));

        assert!(registry.find("test-model-7b", None).is_some());
        assert!(registry.find("TEST-MODEL", None).is_some());
        assert!(registry.find("other-model", None).is_none());
    }

    #[test]
    fn test_registry_find_falls_back_to_model_type() {
        let registry = VisionProcessorRegistry::with_defaults();

        assert!(registry.find("custom-model", None).is_none());

        let processor = registry
            .find("custom-model", Some("qwen3_vl"))
            .expect("qwen3 processor by model_type");
        assert_eq!(processor.model_name(), "qwen3-vl");
    }

    #[test]
    fn test_registry_find_preserves_fast_path() {
        let registry = VisionProcessorRegistry::with_defaults();

        let processor = registry
            .find("Qwen3-VL-30B-A3B-Instruct", Some("qwen2_vl"))
            .expect("qwen3 processor by model_id");
        assert_eq!(processor.model_name(), "qwen3-vl");
    }

    #[test]
    fn test_registry_find_phi3_model_type_fallback() {
        let registry = VisionProcessorRegistry::with_defaults();

        let processor = registry
            .find("custom-model", Some("phi3_v"))
            .expect("phi3 processor by model_type");
        assert_eq!(processor.model_name(), "phi3-vision");
    }

    #[test]
    fn test_registry_find_qwen4_exp_model_type_fallback() {
        let registry = VisionProcessorRegistry::with_defaults();

        let processor = registry
            .find("custom-model", Some("qwen4_exp"))
            .expect("qwen3 processor by qwen4_exp model_type");
        assert_eq!(processor.model_name(), "qwen3-vl");
    }
}
