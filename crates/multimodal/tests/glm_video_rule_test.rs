//! A video for the GLM 5 family costs what it costs on the serving engine's
//! own server: its default loader's frames (32, spread from the first to the
//! last), its serving cap on the clip's vision tokens (30,000), and the
//! stamp of each temporal group taken from the sampled frames.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use llm_multimodal::{
    vision::Glm53FlashProcessor, FrameSampling, MediaItemInfo, Modality, ModelMetadata,
    ModelRegistry, ModelSpecificValue, PreProcessorConfig, RgbFrameRef, TokenId, Tokenizer,
    VideoSamplingInfo, VisionPreProcessor,
};

const IMAGE_ID: u32 = 154854;
const BEGIN_ID: u32 = 154830;
const END_ID: u32 = 154831;

/// Knows the family's media markers; encodes any other text byte by byte.
struct MarkerTokenizer;

impl Tokenizer for MarkerTokenizer {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|image|>" => Some(IMAGE_ID),
            "<|begin_of_image|>" => Some(BEGIN_ID),
            "<|end_of_image|>" => Some(END_ID),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match id {
            IMAGE_ID => Some("<|image|>".to_string()),
            BEGIN_ID => Some("<|begin_of_image|>".to_string()),
            END_ID => Some("<|end_of_image|>".to_string()),
            _ => None,
        }
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        Some(text.bytes().map(|byte| 1000 + u32::from(byte)).collect())
    }
}

/// The video processor config the family's checkpoint ships.
fn video_config() -> PreProcessorConfig {
    PreProcessorConfig::from_json(
        r#"{"do_rescale":true,"video_processor_type":"Glm5NextVideoProcessor","patch_expand_factor":1,
        "merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],
        "image_std":[0.26862954,0.26130258,0.27577711],"temporal_patch_size":2,"patch_size":14,
        "min_image_tokens":16,"max_image_tokens":240000,"fps":2}"#,
    )
    .expect("video processor config")
}

/// The loader's picks for a 278-frame clip: `linspace(0, 277, 32)` truncated.
fn loader_indices() -> Vec<usize> {
    (0..32).map(|i| i * 277 / 31).collect()
}

fn grid(output: &llm_multimodal::PreprocessedEncoderInputs) -> Vec<i64> {
    match output
        .model_specific
        .get("video_grid_thw")
        .expect("video grid")
    {
        ModelSpecificValue::IntTensor { data, .. } => data.clone(),
        other => panic!("unexpected grid value: {other:?}"),
    }
}

/// 32 frames of a 1708x1030 clip (the issue's aspect, half its size: 1.76
/// million pixels a frame against the 1.47 million the serving cap leaves
/// each of 32 frames) fit the engine's canvas, 952x1540: grid
/// `[16, 68, 110]`, 29,920 image tokens, and the stamps follow the frames.
#[test]
fn a_clip_costs_what_it_costs_on_the_engines_own_server() {
    let (width, height) = (1708u32, 1030u32);
    let pixels = vec![128u8; (width * height * 3) as usize];
    let frames: Vec<RgbFrameRef<'_>> = (0..32)
        .map(|_| RgbFrameRef {
            width,
            height,
            data: &pixels,
        })
        .collect();
    let config = video_config();
    let output = Glm53FlashProcessor::new()
        .preprocess_video_rgb(&frames, &config)
        .expect("preprocessed clip");
    assert_eq!(grid(&output), vec![16, 68, 110]);
    assert_eq!(output.feature_token_counts, vec![16 * 34 * 55]);

    let tokenizer = MarkerTokenizer;
    let model_config = serde_json::json!({"model_type": "glm53_flash", "image_token_id": IMAGE_ID});
    let metadata = ModelMetadata {
        model_id: "zai-org/GLM-5.3-Flash",
        tokenizer: &tokenizer,
        config: &model_config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("the family's spec");
    assert_eq!(
        spec.video_frame_sampling(),
        FrameSampling::UpTo { max_frames: 32 }
    );

    let media = [MediaItemInfo {
        video_sampling: Some(VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: loader_indices(),
        }),
    }];
    let replacement = spec
        .prompt_replacements_with_media(&metadata, &output, Modality::Video, &media, &config)
        .expect("video replacement")
        .pop()
        .expect("one clip");
    let stamps = [
        "0.0", "0.0", "1.0", "1.0", "2.0", "2.0", "3.0", "4.0", "4.0", "5.0", "5.0", "6.0", "7.0",
        "7.0", "8.0", "8.0",
    ];
    let expected: Vec<TokenId> = stamps
        .iter()
        .flat_map(|second| {
            [BEGIN_ID]
                .into_iter()
                .chain(std::iter::repeat_n(IMAGE_ID, 34 * 55))
                .chain([END_ID])
                .chain(
                    tokenizer
                        .encode_text(&format!("{second} seconds"))
                        .expect("stamp"),
                )
                .map(|id| id as TokenId)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(replacement.tokens, expected);
}
