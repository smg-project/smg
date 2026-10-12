//! A video prompt for the Qwen3-VL family is laid out as the family's
//! reference processor lays it out: the media connector picks the frames its
//! reference video processor picks, and the spec stamps every temporal group
//! with those frames' times and wraps it in its own markers. The golden
//! (`tests/fixtures/golden/qwen3_vl_video_prompt.json`) is written by
//! `scripts/generate_qwen3_vl_video_prompt_golden.py` from the reference
//! processors themselves.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use llm_multimodal::{
    media::sampled_frame_indices, MediaItemInfo, Modality, ModelMetadata, ModelRegistry,
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, TokenId, Tokenizer,
    VideoFetchConfig, VideoSamplingInfo,
};
use serde::Deserialize;

const VISION_START: u32 = 1;
const VISION_END: u32 = 2;
const VIDEO_PAD: u32 = 3;
const IMAGE_PAD: u32 = 4;
/// Where the marker tokenizer puts plain text, one id per byte.
const TEXT_BASE: u32 = 1000;

/// Knows the family's markers; encodes any other text byte by byte, so a
/// replacement can be rendered back to the text the reference wrote.
struct MarkerTokenizer;

impl Tokenizer for MarkerTokenizer {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|vision_start|>" => Some(VISION_START),
            "<|vision_end|>" => Some(VISION_END),
            "<|video_pad|>" => Some(VIDEO_PAD),
            "<|image_pad|>" => Some(IMAGE_PAD),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match id {
            VISION_START => Some("<|vision_start|>".to_string()),
            VISION_END => Some("<|vision_end|>".to_string()),
            VIDEO_PAD => Some("<|video_pad|>".to_string()),
            IMAGE_PAD => Some("<|image_pad|>".to_string()),
            _ => None,
        }
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        Some(
            text.bytes()
                .map(|byte| TEXT_BASE + u32::from(byte))
                .collect(),
        )
    }
}

#[derive(Deserialize)]
struct Golden {
    sampling: Sampling,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Sampling {
    fps: f32,
    min_frames: usize,
    max_frames: usize,
    temporal_patch_size: usize,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    total_frames: usize,
    fps: f64,
    frame_indices: Vec<usize>,
    grid_thw: [i64; 3],
    frame_seqlen: usize,
    timestamps: Vec<String>,
    placeholder: String,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("fixtures/golden/qwen3_vl_video_prompt.json"))
        .expect("golden JSON")
}

/// The replacement's text: markers by name, everything else by byte.
fn render(tokens: &[TokenId]) -> String {
    let mut text = String::new();
    let mut bytes = Vec::new();
    let flush = |bytes: &mut Vec<u8>, text: &mut String| {
        if !bytes.is_empty() {
            text.push_str(std::str::from_utf8(bytes).expect("utf-8 stamp"));
            bytes.clear();
        }
    };
    for &token in tokens {
        let id = u32::try_from(token).expect("non-negative token id");
        match MarkerTokenizer.id_to_token(id) {
            Some(marker) => {
                flush(&mut bytes, &mut text);
                text.push_str(&marker);
            }
            None => bytes.push(u8::try_from(id - TEXT_BASE).expect("a text byte")),
        }
    }
    flush(&mut bytes, &mut text);
    text
}

#[test]
fn a_video_prompt_is_laid_out_as_the_reference_processor_lays_it_out() {
    let golden = golden();
    let tokenizer = MarkerTokenizer;
    let config = serde_json::json!({
        "model_type": "qwen3_vl",
        "image_token_id": IMAGE_PAD,
        "video_token_id": VIDEO_PAD,
        "vision_start_token_id": VISION_START,
        "vision_end_token_id": VISION_END,
        "vision_config": {"patch_size": 16, "temporal_patch_size": golden.sampling.temporal_patch_size},
    });
    let metadata = ModelMetadata {
        model_id: "Qwen/Qwen3-VL-8B-Instruct",
        tokenizer: &tokenizer,
        config: &config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("qwen3_vl spec");
    assert_eq!(spec.name(), "qwen3_vl");
    let preprocessor_config = PreProcessorConfig::from_json(&format!(
        r#"{{"patch_size":16,"merge_size":2,"temporal_patch_size":{},"fps":{}}}"#,
        golden.sampling.temporal_patch_size, golden.sampling.fps
    ))
    .expect("video preprocessor config");
    // The connector samples the way the spec asks, at the reference's rate
    // and frame bounds.
    let fetch = VideoFetchConfig {
        min_frames: golden.sampling.min_frames,
        max_frames: golden.sampling.max_frames,
        sample_fps: golden.sampling.fps,
        max_long_side_pixel: None,
        sampling: spec.video_frame_sampling(),
    };

    for case in &golden.cases {
        // The reference pads its picks to whole temporal groups by repeating
        // the last frame; the connector leaves that repeat to the processor's
        // own temporal padding, so the picks match up to that tail.
        let indices = sampled_frame_indices(case.total_frames, case.fps, fetch);
        let picked = indices.len();
        assert!(
            picked > 0 && picked <= case.frame_indices.len(),
            "{}: {picked} frames picked for {} frames at {} fps, the reference picked {}",
            case.name,
            case.total_frames,
            case.fps,
            case.frame_indices.len()
        );
        assert_eq!(
            indices[..],
            case.frame_indices[..picked],
            "{}: the frames picked for {} frames at {} fps differ from the reference's",
            case.name,
            case.total_frames,
            case.fps
        );
        let last = indices[picked - 1];
        assert!(
            case.frame_indices[picked..]
                .iter()
                .all(|&index| index == last),
            "{}: the reference's tail {:?} is not a repeat of the last pick {last}",
            case.name,
            &case.frame_indices[picked..]
        );

        let [grid_t, grid_h, grid_w] = case.grid_thw;
        let num_tokens = usize::try_from(grid_t).unwrap() * case.frame_seqlen;
        let preprocessed = PreprocessedEncoderInputs::new(
            ndarray::Array4::<f32>::zeros((1, 1, 1, 1)),
            vec![num_tokens],
            vec![(1, 1)],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![grid_t, grid_h, grid_w], 1, 3),
        );
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: case.fps,
                frame_indices: indices,
            }),
        }];
        let replacements = spec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &preprocessor_config,
            )
            .expect("video replacements");
        assert_eq!(replacements.len(), 1, "{}", case.name);
        let rendered = render(&replacements[0].tokens);
        let stamps: Vec<String> = rendered
            .split('<')
            .filter_map(|part| part.strip_suffix(" seconds>"))
            .map(str::to_string)
            .collect();
        assert_eq!(
            stamps, case.timestamps,
            "{}: the stamps differ from the reference's",
            case.name
        );
        assert_eq!(
            rendered, case.placeholder,
            "{}: the video placeholder differs from the reference's",
            case.name
        );
        // The replacement is the placeholder range, as on the reference, with
        // the chat template's outer markers around it.
        assert_eq!(replacements[0].structural_prefix, 0, "{}", case.name);
    }
}
