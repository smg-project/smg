//! A video for the GLM 5 family costs what it costs on the serving engine's
//! own server: its default loader's frames (at most 32, spread from the first
//! to the last), resampled at the reference video processor's own rate when
//! the loader kept the clip whole, its serving cap on the clip's vision tokens
//! (30,000), and the stamp of each temporal group taken from the sampled
//! frames.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use llm_multimodal::{
    media::sampled_frame_indices, vision::Glm53FlashProcessor, FrameSampling, MediaItemInfo,
    Modality, ModelMetadata, ModelProcessorSpec, ModelRegistry, ModelSpecificValue,
    PreProcessorConfig, RgbFrameRef, TokenId, Tokenizer, VideoClipOptions, VideoFetchConfig,
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

/// The family's sampling under its checkpoint's video processor config: the
/// engine's default loader keeps at most 32 frames spread from the first to
/// the last, and when that kept the whole clip the reference processor
/// resamples it at the config's rate (2 frames a second), at most 2,048.
fn family_sampling(spec: &dyn ModelProcessorSpec, config: &PreProcessorConfig) -> VideoFetchConfig {
    let sampling = spec.video_frame_sampling_with(Some(config));
    assert_eq!(
        sampling,
        FrameSampling::UpToThenRate {
            max_frames: 32,
            rate_max_frames: 2048,
        }
    );
    let sample_fps = spec
        .default_video_sample_fps_with(Some(config))
        .expect("the family's sampling rate");
    assert_eq!(sample_fps, 2.0);
    VideoFetchConfig {
        sample_fps,
        sampling,
        ..VideoFetchConfig::default()
    }
}

/// The prompt a clip becomes: one temporal group per entry of `stamps`,
/// `tokens` image tokens each, followed by its stamp.
fn stamped_prompt(tokenizer: &MarkerTokenizer, stamps: &[&str], tokens: usize) -> Vec<TokenId> {
    stamps
        .iter()
        .flat_map(|second| {
            [BEGIN_ID]
                .into_iter()
                .chain(std::iter::repeat_n(IMAGE_ID, tokens))
                .chain([END_ID])
                .chain(
                    tokenizer
                        .encode_text(&format!("{second} seconds"))
                        .expect("stamp"),
                )
                .map(|id| id as TokenId)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// 32 frames of a 1708x1030 clip (the issue's aspect, half its size: 1.76
/// million pixels a frame against the 1.47 million the serving cap leaves
/// each of 32 frames) fit the engine's canvas, 952x1540: grid
/// `[16, 68, 110]`, 29,920 image tokens, and the stamps follow the frames.
/// The clip has 278 frames at 30 fps, more than the loader keeps, so the
/// loader's picks stand as they are: the reference processor resamples only
/// a clip the loader kept whole.
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
    let output = Glm53FlashProcessor::from_config_for(&config, Modality::Video)
        .expect("video processor")
        .preprocess_video_rgb(&frames, &VideoClipOptions::sampled_at(2.0))
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
    let sampling = family_sampling(spec, &config);
    assert_eq!(sampled_frame_indices(278, 30.0, sampling), loader_indices());

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
    assert_eq!(
        replacement.tokens,
        stamped_prompt(&tokenizer, &stamps, 34 * 55)
    );
}

/// A 4-second clip of 32 frames at 8 fps is kept whole by the loader, so the
/// reference processor resamples it at the config's 2 frames a second: its
/// `sample_frames` picks every fourth frame, `[0, 4, 8, ..., 28]`, eight
/// frames in four temporal groups stamped 0.0 to 3.0 seconds. Four groups
/// leave each frame 5.88 million pixels of the serving cap, so a 1708x1030
/// frame keeps its size, aligned to the 28-pixel factor: grid `[4, 74, 122]`,
/// 9,028 image tokens (four groups of 2,257 at the frame's own size) against
/// the 29,920 (sixteen groups of 1,870 on the shrunk canvas) the loader's 32
/// frames cost under the serving cap.
#[test]
fn a_clip_the_loader_keeps_whole_is_resampled_at_the_processors_rate() {
    let tokenizer = MarkerTokenizer;
    let model_config = serde_json::json!({"model_type": "glm53_flash", "image_token_id": IMAGE_ID});
    let metadata = ModelMetadata {
        model_id: "zai-org/GLM-5.3-Flash",
        tokenizer: &tokenizer,
        config: &model_config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("the family's spec");
    let config = video_config();
    let sampling = family_sampling(spec, &config);
    let picks = sampled_frame_indices(32, 8.0, sampling);
    assert_eq!(picks, vec![0, 4, 8, 12, 16, 20, 24, 28]);

    let (width, height) = (1708u32, 1030u32);
    let pixels = vec![128u8; (width * height * 3) as usize];
    let frames: Vec<RgbFrameRef<'_>> = picks
        .iter()
        .map(|_| RgbFrameRef {
            width,
            height,
            data: &pixels,
        })
        .collect();
    let output = Glm53FlashProcessor::from_config_for(&config, Modality::Video)
        .expect("video processor")
        .preprocess_video_rgb(&frames, &VideoClipOptions::sampled_at(2.0))
        .expect("preprocessed clip");
    assert_eq!(grid(&output), vec![4, 74, 122]);
    assert_eq!(output.feature_token_counts, vec![4 * 37 * 61]);

    let media = [MediaItemInfo {
        video_sampling: Some(VideoSamplingInfo {
            source_fps: 8.0,
            frame_indices: picks,
        }),
    }];
    let replacement = spec
        .prompt_replacements_with_media(&metadata, &output, Modality::Video, &media, &config)
        .expect("video replacement")
        .pop()
        .expect("one clip");
    assert_eq!(
        replacement.tokens,
        stamped_prompt(&tokenizer, &["0.0", "1.0", "2.0", "3.0"], 37 * 61)
    );
}

/// The rate the checkpoint's video processor config carries reaches the picks
/// as the double the reference reads: a config naming 0.3 frames a second
/// picks `[0, 11]` of 20 frames at 3 fps, as the reference's `sample_frames`
/// does, not the `[0, 10]` of the rate narrowed to binary32 on the way.
#[test]
fn the_configs_rate_reaches_the_picks_as_the_double_the_reference_reads() {
    let tokenizer = MarkerTokenizer;
    let model_config = serde_json::json!({"model_type": "glm53_flash", "image_token_id": IMAGE_ID});
    let metadata = ModelMetadata {
        model_id: "zai-org/GLM-5.3-Flash",
        tokenizer: &tokenizer,
        config: &model_config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("the family's spec");
    let config = PreProcessorConfig::from_json(r#"{"fps":0.3}"#).expect("video processor config");
    let sampling = VideoFetchConfig {
        sample_fps: spec
            .default_video_sample_fps_with(Some(&config))
            .expect("the config's rate"),
        sampling: spec.video_frame_sampling_with(Some(&config)),
        ..VideoFetchConfig::default()
    };
    assert_eq!(sampled_frame_indices(20, 3.0, sampling), [0, 11]);
}
