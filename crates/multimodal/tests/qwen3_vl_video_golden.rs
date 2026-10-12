//! The Qwen3-VL family's video path at the crate level, against the reference
//! video processor: a short clip is decoded and sampled by the media connector,
//! preprocessed by the family's processor and laid out by the family's spec.
//! The shapes, the grid and the token count must equal what the reference
//! processor produces for the same frames (`fixtures/golden/qwen3_vl_video.json`,
//! written by `scripts/generate_qwen3_vl_video_golden.py`), and every tensor
//! the spec's video layout names must be one the video batch carries, with the
//! sizes tensor covering every encoder row: that is what assembly slices by.
//!
//! The clip decode needs ffmpeg on PATH; without it the test returns early,
//! like the other video tests.
#![allow(clippy::expect_used, clippy::panic, clippy::print_stderr)]

use std::{fs, path::Path, process::Command};

use llm_multimodal::{
    vision::Qwen3VLProcessor, FieldLayout, MediaConnector, MediaConnectorConfig, MediaSource,
    Modality, ModelMetadata, ModelRegistry, ModelSpecificValue, PreProcessorConfig, Tokenizer,
    VideoFetchConfig, VisionPreProcessor,
};
use serde_json::{json, Value};

mod common;

const GOLDEN: &str = "tests/fixtures/golden/qwen3_vl_video.json";

/// The spec lookup reads the model config only; no token is resolved here.
struct NoTokenizer;

impl Tokenizer for NoTokenizer {
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn as_usizes(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_u64().expect("unsigned") as usize)
        .collect()
}

fn int_values(value: &ModelSpecificValue) -> Vec<i64> {
    match value {
        ModelSpecificValue::IntTensor { data, .. } => data.clone(),
        ModelSpecificValue::UintTensor { data, .. } => data.iter().map(|&v| i64::from(v)).collect(),
        other => panic!("expected an integer tensor, got {other:?}"),
    }
}

#[tokio::test]
async fn qwen3_vl_video_matches_the_reference_layout() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available; skipping");
        return;
    }
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let golden: Value =
        serde_json::from_str(&fs::read_to_string(crate_root.join(GOLDEN)).expect("golden"))
            .expect("golden json");
    let clip =
        fs::read(crate_root.join(golden["clip"].as_str().expect("clip path"))).expect("clip bytes");

    // The connector's own decode and sampling (2 fps, at least 4 frames).
    let connector = MediaConnector::new(
        common::http_client_builder(),
        MediaConnectorConfig::default(),
    )
    .expect("media connector");
    let video = connector
        .fetch_video(MediaSource::InlineBytes(clip), VideoFetchConfig::default())
        .await
        .expect("the clip decodes");
    let frames = video.materialized_frames().expect("decoded frames");
    assert_eq!(
        frames.len(),
        golden["sampled_frames"].as_u64().expect("sampled_frames") as usize,
        "sampled frame count"
    );
    assert_eq!(
        (frames[0].width(), frames[0].height()),
        (
            golden["frame"]["width"].as_u64().expect("width") as u32,
            golden["frame"]["height"].as_u64().expect("height") as u32,
        ),
        "frame size"
    );

    // The family's processor on those frames, with the family's video values.
    let config =
        PreProcessorConfig::from_value(golden["preprocessor_config"].clone()).expect("config");
    let processor = Qwen3VLProcessor::from_preprocessor_config(&config);
    let output = processor
        .preprocess_video(&frames, &config)
        .expect("video preprocess");

    let expected_shape = as_usizes(&golden["outputs"]["pixel_values_videos"]["shape"]);
    assert_eq!(
        output.encoder_input.shape(),
        &expected_shape[..],
        "pixel_values_videos shape"
    );
    let expected_grid: Vec<i64> = golden["outputs"]["video_grid_thw"]["values"][0]
        .as_array()
        .expect("grid row")
        .iter()
        .map(|v| v.as_i64().expect("grid value"))
        .collect();
    assert_eq!(
        int_values(&output.model_specific["video_grid_thw"]),
        expected_grid,
        "video_grid_thw"
    );
    assert_eq!(
        output.feature_token_counts,
        vec![golden["tokens_per_video"].as_u64().expect("tokens") as usize],
        "video tokens"
    );

    // The spec's video layout names only tensors this batch carries, and its
    // sizes tensor accounts for every encoder row of the one clip.
    let model_config = json!({
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
    });
    let registry = ModelRegistry::new();
    let spec = registry
        .lookup(&ModelMetadata {
            model_id: "qwen3-vl",
            tokenizer: &NoTokenizer,
            config: &model_config,
        })
        .expect("the family's spec");
    let layouts = spec.encoder_field_layouts_for(Modality::Video);
    let mut carried: Vec<&String> = output.model_specific.keys().collect();
    carried.sort();
    let FieldLayout::Flat { sizes_key } = &layouts.encoder_input else {
        panic!("video patches are laid out flat over a sizes tensor, got {layouts:?}");
    };
    let sizes = output
        .model_specific
        .get(sizes_key)
        .unwrap_or_else(|| {
            panic!("the video layout slices by {sizes_key:?}, the batch carries {carried:?}")
        })
        .as_flat_sizes()
        .expect("flat sizes");
    assert_eq!(
        sizes,
        vec![expected_shape[0]],
        "one clip owns every encoder row"
    );
    for key in layouts.model_specific.keys() {
        assert!(
            output.model_specific.contains_key(key),
            "the video layout names {key:?}, the batch carries {carried:?}"
        );
    }
}
