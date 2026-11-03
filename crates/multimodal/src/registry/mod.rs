mod deepseek_v41;
mod gemma4;
mod glm53_flash;
mod inkling;
mod kimi_k25;
mod kimi_k3;
mod llama4;
mod llava;
mod minimax_m3;
mod muse_spark;
mod nemotron_omni;
mod phi3_v;
pub mod qwen3_asr;
mod qwen3_omni;
mod qwen3_vl;
mod qwen_vl;
mod traits;
pub mod transcription;

use deepseek_v41::DeepseekV41VisionSpec;
pub use deepseek_v41::DEEPSEEK_V41_IMAGE_PLACEHOLDER;
use gemma4::Gemma4Spec;
use glm53_flash::Glm53FlashSpec;
use inkling::InklingSpec;
use kimi_k25::KimiK25VisionSpec;
use kimi_k3::KimiK3VisionSpec;
use llama4::Llama4Spec;
use llava::{LlavaNextSpec, LlavaSpec};
use minimax_m3::MiniMaxM3VisionSpec;
use muse_spark::MuseSparkSpec;
use nemotron_omni::NemotronOmniSpec;
use once_cell::sync::Lazy;
use phi3_v::Phi3VisionSpec;
use qwen3_asr::Qwen3AsrSpec;
use qwen3_omni::Qwen3OmniSpec;
use qwen3_vl::Qwen3VLVisionSpec;
use qwen_vl::QwenVLVisionSpec;
// Re-export public API from traits.
pub use traits::{
    modality_limit_override, EncoderInputFormat, MediaItemInfo, MediaPartOrder, ModalityHooks,
    ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult, ReplacementsHook,
    Tokenizer,
};

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                // DeepSeek-V4.1 matches only its own model_type / id, so it can lead.
                LazySpec::new(|| Box::new(DeepseekV41VisionSpec)),
                // Gemma 4 matches its own model_type / id only.
                LazySpec::new(|| Box::new(Gemma4Spec)),
                LazySpec::new(|| Box::new(Glm53FlashSpec)),
                LazySpec::new(|| Box::new(InklingSpec)),
                // Kimi-K3 must be registered before Kimi-K2.5: the two families
                // share a transport layout but not a prompt shape.
                LazySpec::new(|| Box::new(KimiK3VisionSpec)),
                LazySpec::new(|| Box::new(KimiK25VisionSpec)),
                LazySpec::new(|| Box::new(Llama4Spec)),
                LazySpec::new(|| Box::new(MiniMaxM3VisionSpec)),
                // Muse Spark matches its own model_type / architecture / id only.
                LazySpec::new(|| Box::new(MuseSparkSpec)),
                // Nemotron-H Omni matches its own model_type only.
                LazySpec::new(|| Box::new(NemotronOmniSpec)),
                // LlavaNext must be registered before Llava so "llava_next" model_type matches first.
                LazySpec::new(|| Box::new(LlavaNextSpec)),
                LazySpec::new(|| Box::new(LlavaSpec)),
                LazySpec::new(|| Box::new(Qwen3AsrSpec)),
                LazySpec::new(|| Box::new(Qwen3OmniSpec)),
                // Qwen3-VL must be registered before QwenVL so "qwen3" matches first.
                LazySpec::new(|| Box::new(Qwen3VLVisionSpec)),
                LazySpec::new(|| Box::new(QwenVLVisionSpec)),
                LazySpec::new(|| Box::new(Phi3VisionSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }

    /// The names of the model families this registry has a spec for, in
    /// lookup order: what a refusal can list for the operator.
    pub fn spec_names(&self) -> Vec<&'static str> {
        self.specs.iter().map(|spec| spec.get().name()).collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

#[cfg(test)]
pub(super) mod test_helpers {
    use std::collections::HashMap;

    use crate::{
        encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
        types::ImageSize,
    };

    pub struct TestTokenizer {
        vocab: HashMap<String, u32>,
        text_base: Option<u32>,
    }

    impl TestTokenizer {
        pub fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self {
                vocab,
                text_base: None,
            }
        }

        /// Encode text as one id per byte, offset by `base`.
        ///
        /// Off by default so specs that only look up special tokens are
        /// unaffected; specs that splice encoded text into a replacement
        /// enable it to assert the exact layout.
        pub fn with_byte_encoder(mut self, base: u32) -> Self {
            self.text_base = Some(base);
            self
        }
    }

    impl crate::registry::Tokenizer for TestTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
        }

        fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
            // Mirrors the pre-seam stub behavior: without a byte encoder the
            // stub encodes everything to nothing rather than failing.
            Some(match self.text_base {
                Some(base) => text.bytes().map(|b| base + u32::from(b)).collect(),
                None => Vec::new(),
            })
        }
    }

    pub fn test_preprocessed_with_tokens(
        item_sizes: &[ImageSize],
        feature_token_counts: &[usize],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::<f32>::zeros(vec![1, 3, 336, 336]).into(),
            feature_token_counts: feature_token_counts.to_vec(),
            item_sizes: sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Build `PreprocessedEncoderInputs` with explicit aspect_ratios (for Llama4 tests).
    pub fn test_preprocessed_with_aspects(
        item_sizes: &[ImageSize],
        aspect_ratios: &[(i64, i64)],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        let flat: Vec<i64> = aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h, w])
            .collect();
        let batch = aspect_ratios.len();
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: flat,
                shape: vec![batch, 2],
            },
        );
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::<f32>::zeros(vec![1, 3, 336, 336]).into(),
            feature_token_counts: vec![0; sizes.len()],
            item_sizes: sizes,
            model_specific,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::test_helpers::TestTokenizer,
        vision::{
            processors::llava::{CLIP_MEAN, CLIP_STD},
            transforms::mean_to_rgb,
        },
        Modality, ModelSpecificValue, PreProcessorConfig, VideoClipOptions,
    };

    /// A checkpoint of this family resolves to a spec by its `model_type`
    /// alone (the served id is often an opaque path).
    #[test]
    fn the_registry_resolves_the_gemma4_family() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({
            "model_type": "gemma4",
            "image_token_id": 258880,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "video_token_id": 258884
        });
        let metadata = ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("the gemma4 family has a spec");
        assert_eq!(spec.name(), "gemma4");
    }

    /// The vision processors come from the spec a model resolves to, so they
    /// preprocess the way that spec lays out the prompt: a checkpoint known
    /// only by its `model_type` gets one, and an id that names another
    /// family's release does not pick that family's processor.
    #[test]
    fn a_model_preprocesses_with_the_processors_of_the_spec_it_resolves_to() {
        let tokenizer = TestTokenizer::new(&[]);
        let preprocessor_config = PreProcessorConfig::default();
        for (model_id, model_type, spec_name, image, video) in [
            ("/models/checkpoint", "llava", "llava", "llava", None),
            // A LLaVA config under an id that names a LLaVA-NeXT release.
            ("org/llava-v1.6-vicuna-7b", "llava", "llava", "llava", None),
            (
                "/models/checkpoint",
                "llava_next",
                "llava_next",
                "llava-next",
                None,
            ),
            (
                "/models/checkpoint",
                "qwen2_vl",
                "qwen_vl",
                "qwen2-vl",
                None,
            ),
            (
                "/models/checkpoint",
                "qwen3_vl",
                "qwen3_vl",
                "qwen3-vl",
                Some("qwen3-vl"),
            ),
        ] {
            let config = json!({ "model_type": model_type });
            let metadata = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &config,
            };
            let registry = ModelRegistry::new();
            let spec = registry.lookup(&metadata).expect("a vision spec");
            assert_eq!(spec.name(), spec_name, "{model_id} ({model_type})");
            let processor = |modality| {
                spec.vision_processor(&config, &preprocessor_config, modality)
                    .map(|processor| processor.model_name())
            };
            assert_eq!(processor(Modality::Image), Ok(image), "{model_id}");
            match video {
                Some(video) => assert_eq!(processor(Modality::Video), Ok(video), "{model_id}"),
                None => assert_eq!(
                    processor(Modality::Video),
                    Err(ModelRegistryError::UnsupportedModality {
                        spec: spec_name,
                        modality: Modality::Video,
                    }),
                    "{model_id}"
                ),
            }
        }
    }

    fn white_image() -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(128, 128, image::Rgb([255; 3])))
    }

    fn metadata<'a>(
        tokenizer: &'a TestTokenizer,
        config: &'a serde_json::Value,
    ) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer,
            config,
        }
    }

    /// Each processor keeps the geometry and normalization of the config it
    /// was built from, whatever another instance built from another config
    /// does between its calls.
    #[test]
    fn vision_processors_retain_the_parameters_of_their_own_config() {
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({"model_type": "qwen3_vl"});
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata(&tokenizer, &model_config))
            .expect("qwen3_vl spec");
        let mut config = PreProcessorConfig::from_json(
            r#"{"min_pixels":1024,"max_pixels":1024,"image_mean":[0,0,0],"image_std":[1,1,1]}"#,
        )
        .unwrap();
        let small = spec
            .vision_processor(&model_config, &config, Modality::Image)
            .unwrap();
        config.min_pixels = Some(4096);
        config.max_pixels = Some(4096);
        config.image_mean = Some(vec![1.0; 3]);
        let large = spec
            .vision_processor(&model_config, &config, Modality::Image)
            .unwrap();
        drop(config);

        for _ in 0..2 {
            let small_output = small.preprocess(&[white_image()]).unwrap();
            let large_output = large.preprocess(&[white_image()]).unwrap();
            assert_eq!(small_output.feature_token_counts, vec![1]);
            assert_eq!(large_output.feature_token_counts, vec![4]);
            assert!(small_output
                .encoder_input
                .flat_f32()
                .iter()
                .all(|&v| v == 1.0));
            assert!(large_output
                .encoder_input
                .flat_f32()
                .iter()
                .all(|&v| v == 0.0));
            assert_eq!(small.calculate_num_tokens(128, 128).unwrap(), 1);
            assert_eq!(large.calculate_num_tokens(128, 128).unwrap(), 4);
        }
    }

    /// The image and video processors are built from their own configs, and
    /// the rate a clip was sampled at is the call's, not the processor's.
    #[test]
    fn image_and_video_processors_resolve_their_own_configs() {
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({"model_type": "qwen3_vl"});
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata(&tokenizer, &model_config))
            .expect("qwen3_vl spec");
        let image_config =
            PreProcessorConfig::from_json(r#"{"min_pixels":1024,"max_pixels":1024}"#).unwrap();
        let video_config =
            PreProcessorConfig::from_json(r#"{"min_pixels":8192,"max_pixels":8192}"#).unwrap();
        let images = spec
            .vision_processor(&model_config, &image_config, Modality::Image)
            .unwrap();
        let videos = spec
            .vision_processor(&model_config, &video_config, Modality::Video)
            .unwrap();
        assert_eq!(
            images
                .preprocess(&[white_image()])
                .unwrap()
                .feature_token_counts,
            vec![1]
        );
        for (fps, second_per_grid) in [(8.0, 0.25), (2.0, 1.0)] {
            let video = videos
                .preprocess_video(
                    &[white_image(), white_image()],
                    &VideoClipOptions::sampled_at(fps),
                )
                .unwrap();
            assert_eq!(video.feature_token_counts, vec![4]);
            assert!(matches!(
                video.model_specific.get("video_second_per_grid"),
                Some(ModelSpecificValue::Tensor { data, shape })
                    if data == &[second_per_grid] && shape == &[1]
            ));
        }
        assert_eq!(
            spec.vision_processor(&model_config, &image_config, Modality::Audio)
                .err(),
            Some(ModelRegistryError::UnsupportedModality {
                spec: "qwen3_vl",
                modality: Modality::Audio,
            })
        );
    }

    /// The LLaVA processor takes its geometry from the model config the spec
    /// builds it with.
    #[test]
    fn the_llava_processor_takes_its_geometry_from_the_model_config() {
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({
            "model_type": "llava",
            "vision_config": {"image_size": 28, "patch_size": 14},
            "image_aspect_ratio": "pad",
        });
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata(&tokenizer, &model_config))
            .expect("llava spec");
        let processor = spec
            .vision_processor(
                &model_config,
                &PreProcessorConfig::default(),
                Modality::Image,
            )
            .unwrap();
        // A 28x14 image, white on its left half and black on its right:
        // `pad` centres it on a 28x28 canvas filled with the mean colour,
        // where `square` would resize and crop it to its content.
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(28, 14, |x, _| {
            image::Rgb([if x < 14 { 255 } else { 0 }; 3])
        }));
        let output = processor.preprocess(&[image]).unwrap();
        assert_eq!(output.encoder_input.shape(), &[1, 3, 28, 28]);
        assert_eq!(output.feature_token_counts, vec![4]);
        assert_eq!(processor.get_processed_size(), Some((28, 28)));
        let values = output.encoder_input.flat_f32();
        let fill = mean_to_rgb(&CLIP_MEAN);
        for channel in 0..3 {
            let at = |x: usize, y: usize| values[channel * 28 * 28 + y * 28 + x];
            let normalized = |byte: u8| {
                ((f64::from(byte) / 255.0 - CLIP_MEAN[channel]) / CLIP_STD[channel]) as f32
            };
            for (x, y, byte) in [
                (2, 2, fill[channel]),
                (25, 25, fill[channel]),
                (2, 14, 255),
                (25, 14, 0),
            ] {
                assert!(
                    (at(x, y) - normalized(byte)).abs() < 0.05,
                    "channel {channel} at ({x}, {y}): {} != {}",
                    at(x, y),
                    normalized(byte)
                );
            }
        }
    }

    /// Each modality's processor validates only the settings its own path
    /// reads: a video budget the image defaults cannot hold still builds the
    /// video processor, and an image budget the video path never reads does
    /// not stop it.
    #[test]
    fn a_modality_processor_validates_only_its_own_settings() {
        let tokenizer = TestTokenizer::new(&[]);
        let registry = ModelRegistry::new();
        for (model_type, spec_name, config) in [
            // 8,001 tokens: within the 30,000-token video budget, above the
            // 8,000-token image maximum.
            (
                "glm53_flash",
                "glm53_flash",
                r#"{"min_pixels":12545568,"max_video_tokens":30000}"#,
            ),
            // A budget the image path refuses; video frames of an image
            // config keep the video default.
            ("gemma4", "gemma4", r#"{"max_soft_tokens":100}"#),
        ] {
            let model_config = json!({ "model_type": model_type });
            let spec = registry
                .lookup(&metadata(&tokenizer, &model_config))
                .expect("a vision spec");
            let config = PreProcessorConfig::from_json(config).unwrap();
            assert!(
                spec.vision_processor(&model_config, &config, Modality::Video)
                    .is_ok(),
                "{model_type}: the video processor"
            );
            assert!(
                matches!(
                    spec.vision_processor(&model_config, &config, Modality::Image),
                    Err(ModelRegistryError::InvalidPreprocessorConfig {
                        spec,
                        modality: Modality::Image,
                        ..
                    }) if spec == spec_name
                ),
                "{model_type}: the image processor"
            );
        }
    }

    /// A preprocessor config the processor cannot serve is refused when the
    /// processor is built, not on a request.
    #[test]
    fn a_preprocessor_config_the_processor_cannot_serve_is_refused_at_construction() {
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({"model_type": "minimax_m3_vl"});
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata(&tokenizer, &model_config))
            .expect("minimax spec");
        let config = PreProcessorConfig::from_json(
            r#"{"img_token_compression_config": {"spatial_merge_size": "two"}}"#,
        )
        .unwrap();
        assert!(matches!(
            spec.vision_processor(&model_config, &config, Modality::Image),
            Err(ModelRegistryError::InvalidPreprocessorConfig {
                spec: "minimax_m3",
                modality: Modality::Image,
                ..
            })
        ));
    }
}
