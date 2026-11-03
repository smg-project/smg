//! A video for the GLM 5 family costs what it costs on the serving engine's
//! own server: the frames its own video loader picks for this family
//! (`trunc(duration x 2)` of them, at most 2,048, from a walk at twice the
//! rate re-spread evenly over the clip; a longer clip is no longer cut to the
//! generic loader's 32), its serving cap on the clip's vision tokens (30,000),
//! and the stamp of each temporal group taken from the sampled frames.
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
/// family's own frame sampler at the config's rate (2 frames a second), at
/// most 2,048 frames, its walk scaled by the temporal patch size of 2.
fn family_sampling(spec: &dyn ModelProcessorSpec, config: &PreProcessorConfig) -> VideoFetchConfig {
    let sampling = spec.video_frame_sampling_with(Some(config));
    assert_eq!(
        sampling,
        FrameSampling::RateWalk {
            max_frames: 2048,
            temporal_patch_size: 2,
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

/// The family's spec and the metadata its prompt replacements read.
fn family<'a>(
    tokenizer: &'a MarkerTokenizer,
    model_config: &'a serde_json::Value,
) -> ModelMetadata<'a> {
    ModelMetadata {
        model_id: "zai-org/GLM-5.3-Flash",
        tokenizer,
        config: model_config,
    }
}

/// `picks` frames of `width` x `height` grey pixels through the family's
/// processor, then its prompt replacement for a clip at `fps` whose decoder
/// reported those picks: the grid, the image token count and the stamped
/// prompt are compared with the engine's own numbers for the same clip.
fn clip_through_the_family(
    picks: &[usize],
    fps: f64,
    (width, height): (u32, u32),
) -> (Vec<i64>, Vec<usize>, Vec<TokenId>) {
    let tokenizer = MarkerTokenizer;
    let model_config = serde_json::json!({"model_type": "glm53_flash", "image_token_id": IMAGE_ID});
    let metadata = family(&tokenizer, &model_config);
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("the family's spec");
    let config = video_config();
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
    let media = [MediaItemInfo {
        video_sampling: Some(VideoSamplingInfo {
            source_fps: fps,
            frame_indices: picks.to_vec(),
        }),
    }];
    let replacement = spec
        .prompt_replacements_with_media(&metadata, &output, Modality::Video, &media, &config)
        .expect("video replacement")
        .pop()
        .expect("one clip");
    (
        grid(&output),
        output.feature_token_counts.clone(),
        replacement.tokens,
    )
}

fn picks_for(total_frames: usize, fps: f64) -> Vec<usize> {
    let tokenizer = MarkerTokenizer;
    let model_config = serde_json::json!({"model_type": "glm53_flash", "image_token_id": IMAGE_ID});
    let metadata = family(&tokenizer, &model_config);
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("the family's spec");
    sampled_frame_indices(total_frames, fps, family_sampling(spec, &video_config()))
}

/// A 278-frame clip at 30 fps (9.27 seconds) is sampled by the family's own
/// loader to `trunc(9.27 x 2) = 18` frames, `linspace(0, 277, 18)` truncated
/// (its walk at a quarter second over-collects and is re-spread): nine
/// temporal groups where the generic 32-frame loader rule gave sixteen. 18
/// frames of 1708x1030 (the issue's aspect, half its size) fit the engine's
/// serving cap at their own size, aligned to the 28-pixel factor: grid
/// `[9, 74, 122]`, 20,313 image tokens (nine groups of 2,257) against the
/// 29,920 (sixteen groups of 1,870 on a 952x1540 canvas) the 32 frames cost;
/// each group is stamped with the whole second of its first frame.
#[test]
fn a_clip_costs_what_it_costs_on_the_engines_own_server() {
    let picks = picks_for(278, 30.0);
    assert_eq!(
        picks,
        vec![0, 16, 32, 48, 65, 81, 97, 114, 130, 146, 162, 179, 195, 211, 228, 244, 260, 277]
    );
    let (grid, tokens, prompt) = clip_through_the_family(&picks, 30.0, (1708, 1030));
    assert_eq!(grid, vec![9, 74, 122]);
    assert_eq!(tokens, vec![9 * 37 * 61]);
    let stamps = [
        "0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0",
    ];
    assert_eq!(prompt, stamped_prompt(&MarkerTokenizer, &stamps, 37 * 61));
}

/// A 4-second clip of 32 frames at 8 fps: eight frames, `linspace(0, 31, 8)`
/// truncated, `[0, 4, 8, 13, 17, 22, 26, 31]`, the engine's own picks for
/// this clip (the generic loader rule with the reference processor's
/// resample picked every fourth frame, `[0, 4, ..., 28]`: the same count and
/// cost, five different frames). Four groups leave each 1708x1030 frame its
/// own size: grid `[4, 74, 122]`, 9,028 image tokens, stamped 0.0 to 3.0
/// seconds (the groups start at frames 0, 8, 17 and 26).
#[test]
fn a_short_clip_takes_the_engines_own_frames() {
    let picks = picks_for(32, 8.0);
    assert_eq!(picks, vec![0, 4, 8, 13, 17, 22, 26, 31]);
    let (grid, tokens, prompt) = clip_through_the_family(&picks, 8.0, (1708, 1030));
    assert_eq!(grid, vec![4, 74, 122]);
    assert_eq!(tokens, vec![4 * 37 * 61]);
    assert_eq!(
        prompt,
        stamped_prompt(&MarkerTokenizer, &["0.0", "1.0", "2.0", "3.0"], 37 * 61)
    );
}

/// The two clips the engine's own server was measured on beside this
/// pipeline: 50 frames at 10 fps (5 seconds, 640x360) become ten frames in
/// five groups, grid `[5, 26, 46]` and 1,495 image tokens; 36 frames at 12
/// fps (3 seconds, 1280x720) become six frames in three groups, grid
/// `[3, 52, 92]` and 3,588 image tokens: the engine's own picks, grids and
/// counts for these clips, where the generic 32-frame loader rule sampled 32
/// frames of each (4,901 and 19,253 prompt tokens against the engine's 1,546
/// and 3,627).
#[test]
fn clips_longer_than_the_generic_loader_budget_keep_the_engines_count() {
    let picks = picks_for(50, 10.0);
    assert_eq!(picks, vec![0, 5, 10, 16, 21, 27, 32, 38, 43, 49]);
    let (grid, tokens, prompt) = clip_through_the_family(&picks, 10.0, (640, 360));
    assert_eq!(grid, vec![5, 26, 46]);
    assert_eq!(tokens, vec![5 * 13 * 23]);
    assert_eq!(
        prompt,
        stamped_prompt(
            &MarkerTokenizer,
            &["0.0", "1.0", "2.0", "3.0", "4.0"],
            13 * 23
        )
    );

    let picks = picks_for(36, 12.0);
    assert_eq!(picks, vec![0, 7, 14, 21, 28, 35]);
    let (grid, tokens, prompt) = clip_through_the_family(&picks, 12.0, (1280, 720));
    assert_eq!(grid, vec![3, 52, 92]);
    assert_eq!(tokens, vec![3 * 26 * 46]);
    assert_eq!(
        prompt,
        stamped_prompt(&MarkerTokenizer, &["0.0", "1.0", "2.0"], 26 * 46)
    );
}

/// The rate the checkpoint's video processor config carries reaches the picks
/// as the double the reference reads: a config naming 0.3 frames a second is
/// `0.3` at the sampler, not the `0.300000011920929` of the rate narrowed to
/// binary32 on the way. The generic loader rule's resample at that rate picks
/// `[0, 11]` of 20 frames at 3 fps, as the reference's `sample_frames` does,
/// not the `[0, 10]` of the narrowed rate; the family's own rule picks
/// `trunc(6.67 s x 0.3) = 2` frames of the same clip, `[0, 19]`.
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
    let rate = spec
        .default_video_sample_fps_with(Some(&config))
        .expect("the config's rate");
    assert_eq!(rate, 0.3);
    let generic = VideoFetchConfig {
        sample_fps: rate,
        sampling: FrameSampling::UpToThenRate {
            max_frames: 32,
            rate_max_frames: 2048,
        },
        ..VideoFetchConfig::default()
    };
    assert_eq!(sampled_frame_indices(20, 3.0, generic), [0, 11]);
    let family = VideoFetchConfig {
        sample_fps: rate,
        sampling: spec.video_frame_sampling_with(Some(&config)),
        ..VideoFetchConfig::default()
    };
    assert_eq!(sampled_frame_indices(20, 3.0, family), [0, 19]);
}
