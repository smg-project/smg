//! Muse Spark image and video preprocessing pinned to the family's reference
//! processors (transformers' `MuseGlimmerImageProcessor` and
//! `MuseGlimmerVideoProcessor`, the torchvision backend) and to the prompt
//! rewriting of its `MuseSparkProcessor`.
//!
//! `scripts/generate_muse_spark_preprocess_fingerprints.py` runs the reference
//! over synthetic seeded images (rebuilt here with the same formula), PNG
//! fixtures (PNG so PIL and the `image` crate decode identical pixels) and
//! synthetic frame clips sampled by the reference's own `sample_frames`,
//! recording the grid, the token count, the shape, FNV-1a fingerprints of the
//! exact float32 patch bytes and of the bfloat16 bits the engine receives,
//! and the prompt expansion. This test reproduces every one of them: the
//! processor's bytes, the connector's frame sampling and the spec's
//! replacement, stamps included.
#![allow(clippy::expect_used, clippy::panic)]

use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    media::{sampled_frame_indices, FrameSampling, VideoFetchConfig},
    registry::{MediaItemInfo, ModelMetadata, ModelRegistry, Tokenizer},
    types::VideoSamplingInfo,
    vision::{
        preprocessor_config::PreProcessorConfig, processor::ModelSpecificValue,
        processors::MuseSparkVisionProcessor, PreprocessedEncoderInputs, VisionPreProcessor,
        VisionProcessorRegistry,
    },
    Modality, PromptReplacement, TokenId,
};
use serde::Deserialize;
use serde_json::json;

const GOLDEN: &str = include_str!("fixtures/golden/muse_spark_preprocess_fingerprints.json");

#[derive(Deserialize)]
struct GoldenDocument {
    reference: String,
    image_config: serde_json::Value,
    video_config: serde_json::Value,
    cases: Vec<GoldenCase>,
    video_cases: Vec<GoldenVideoCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    source: Source,
    width: u32,
    height: u32,
    grid_thw: serde_json::Value,
    tokens: serde_json::Value,
    shape: Vec<usize>,
    expansion: String,
    fnv1a_f32: String,
    fnv1a_bf16: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Source {
    Seed {
        seed: u8,
    },
    File {
        file: String,
    },
    Seeds {
        seeds: Vec<u8>,
        sizes: Vec<[u32; 2]>,
    },
}

#[derive(Deserialize)]
struct GoldenVideoCase {
    name: String,
    source: ClipSource,
    width: u32,
    height: u32,
    frame_indices: Vec<usize>,
    grid_thw: Vec<i64>,
    tokens: usize,
    shape: Vec<usize>,
    stamps: Vec<String>,
    expansion: String,
    fnv1a_f32: String,
    fnv1a_bf16: String,
}

#[derive(Deserialize)]
struct ClipSource {
    seed: u8,
    frames: usize,
    fps: f64,
}

/// The reference's media tokens, and plain text as one id per byte above
/// `TEXT_BASE`, so a replacement decodes back to the string the reference
/// writes.
struct ByteTokenizer;

const TEXT_BASE: u32 = 1_000_000;
const SPECIAL: &[(&str, u32)] = &[
    ("<|patch|>", 200_092),
    ("<|video|>", 200_091),
    ("<|image_start|>", 200_080),
    ("<|image_end|>", 200_081),
    ("<|vid_start|>", 200_082),
    ("<|vid_end|>", 200_083),
    ("<|vid_frame_separator|>", 200_087),
];

impl Tokenizer for ByteTokenizer {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        SPECIAL
            .iter()
            .find(|(text, _)| *text == token)
            .map(|(_, id)| *id)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        SPECIAL
            .iter()
            .find(|(_, known)| *known == id)
            .map(|(text, _)| (*text).to_string())
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        Some(
            text.bytes()
                .map(|byte| TEXT_BASE + u32::from(byte))
                .collect(),
        )
    }
}

fn decoded(replacement: &PromptReplacement) -> String {
    let mut out = String::new();
    let mut bytes = Vec::new();
    for &token in &replacement.tokens {
        let token = token as u32;
        if token >= TEXT_BASE {
            bytes.push((token - TEXT_BASE) as u8);
            continue;
        }
        if !bytes.is_empty() {
            out.push_str(&String::from_utf8(std::mem::take(&mut bytes)).expect("utf-8 stamp"));
        }
        out.push_str(&ByteTokenizer.id_to_token(token).expect("a known token"));
    }
    if !bytes.is_empty() {
        out.push_str(&String::from_utf8(bytes).expect("utf-8 stamp"));
    }
    out
}

fn make_seeded_image(width: u32, height: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            seed.wrapping_add(((x * 7 + y * 3) % 256) as u8),
            seed.wrapping_add(((x * 5 + y * 11) % 256) as u8),
            seed.wrapping_add(((x + y * 2) % 256) as u8),
        ])
    }))
}

fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn images_for(case: &GoldenCase) -> Vec<DynamicImage> {
    match &case.source {
        Source::Seed { seed } => vec![make_seeded_image(case.width, case.height, *seed)],
        Source::Seeds { seeds, sizes } => seeds
            .iter()
            .zip(sizes)
            .map(|(seed, [width, height])| make_seeded_image(*width, *height, *seed))
            .collect(),
        Source::File { file } => {
            let path = format!(
                "{}/tests/fixtures/images/{file}",
                env!("CARGO_MANIFEST_DIR")
            );
            vec![image::open(&path).unwrap_or_else(|e| panic!("{}: decode {path}: {e}", case.name))]
        }
    }
}

fn int_tensor<'a>(out: &'a PreprocessedEncoderInputs, key: &str) -> &'a [i64] {
    match out.model_specific.get(key) {
        Some(ModelSpecificValue::IntTensor { data, .. }) => data,
        other => panic!("{key} must be an int tensor, got {other:?}"),
    }
}

fn check_bytes(
    name: &str,
    out: &PreprocessedEncoderInputs,
    shape: &[usize],
    f32: &str,
    bf16: &str,
) {
    assert_eq!(out.encoder_input.shape(), shape, "{name}: shape");
    let flat = out.encoder_input_flat();
    let values: &[f32] = flat.as_ref();
    assert_eq!(
        fnv1a(values.iter().flat_map(|v| v.to_le_bytes())),
        f32,
        "{name}: float32 patch bytes differ from the reference"
    );
    assert_eq!(
        fnv1a(
            values
                .iter()
                .flat_map(|v| llm_multimodal::f32_to_bf16_bits(*v).to_le_bytes())
        ),
        bf16,
        "{name}: bfloat16 patch bits differ from the reference"
    );
}

fn model_config() -> serde_json::Value {
    json!({
        "model_type": "muse_spark",
        "architectures": ["MuseSparkForConditionalGeneration"],
        "image_token_id": 200092,
        "video_token_id": 200091,
    })
}

#[test]
fn images_match_the_reference_fingerprints_and_expansions() {
    let document: GoldenDocument =
        serde_json::from_str(GOLDEN).expect("golden document must match the schema");
    assert!(
        document.reference.contains("MuseGlimmerImageProcessor"),
        "unexpected reference {}",
        document.reference
    );
    let config = PreProcessorConfig::from_value(document.image_config).expect("image config");
    let processor = MuseSparkVisionProcessor::new();
    let registry = ModelRegistry::new();
    let model_config = model_config();
    let metadata = ModelMetadata {
        model_id: "muse-spark",
        tokenizer: &ByteTokenizer,
        config: &model_config,
    };
    let spec = registry.lookup(&metadata).expect("the family's spec");

    for case in &document.cases {
        let name = &case.name;
        let images = images_for(case);
        let out = processor
            .preprocess(&images, &config)
            .unwrap_or_else(|e| panic!("{name}: preprocess: {e}"));
        let grid: Vec<i64> = match &case.grid_thw {
            serde_json::Value::Array(rows) if rows.first().is_some_and(|v| v.is_array()) => rows
                .iter()
                .flat_map(|row| {
                    row.as_array()
                        .expect("row")
                        .iter()
                        .map(|v| v.as_i64().expect("int"))
                })
                .collect(),
            other => serde_json::from_value(other.clone()).expect("grid"),
        };
        assert_eq!(
            int_tensor(&out, "image_grid_thw"),
            grid,
            "{name}: image_grid_thw"
        );
        let tokens: Vec<usize> = match &case.tokens {
            serde_json::Value::Number(n) => vec![n.as_u64().expect("tokens") as usize],
            other => serde_json::from_value(other.clone()).expect("tokens"),
        };
        assert_eq!(out.feature_token_counts, tokens, "{name}: tokens");
        for (image, &count) in images.iter().zip(&tokens) {
            assert_eq!(
                processor
                    .calculate_num_tokens(image.width(), image.height(), &config)
                    .expect("count"),
                count,
                "{name}: calculate_num_tokens"
            );
        }
        check_bytes(name, &out, &case.shape, &case.fnv1a_f32, &case.fnv1a_bf16);
        if !case.expansion.is_empty() {
            let replacements = spec
                .prompt_replacements_with_media(
                    &metadata,
                    &out,
                    Modality::Image,
                    &[MediaItemInfo::default()],
                    &config,
                )
                .expect("replacements");
            assert_eq!(
                decoded(&replacements[0]),
                case.expansion,
                "{name}: expansion"
            );
            assert_eq!(
                replacements[0]
                    .feature_ranges
                    .as_ref()
                    .map(|r| (r[0].offset, r[0].length)),
                Some((1, tokens[0])),
                "{name}: feature span"
            );
        }
    }

    // The registry resolves the family's processor by model_type and by id.
    let processors = VisionProcessorRegistry::with_defaults();
    for (model_id, model_type) in [
        ("/models/local-checkpoint", Some("muse_spark")),
        ("meta-models/Muse-Spark-open-816B-A42B-NVFP4", None),
    ] {
        let found = processors
            .find(model_id, model_type)
            .unwrap_or_else(|| panic!("no processor for {model_id}"));
        assert_eq!(found.model_name(), "muse_spark");
    }
}

#[test]
fn clips_match_the_reference_sampling_fingerprints_and_stamps() {
    let document: GoldenDocument =
        serde_json::from_str(GOLDEN).expect("golden document must match the schema");
    let config = PreProcessorConfig::from_value(document.video_config).expect("video config");
    let processor = MuseSparkVisionProcessor::new();
    let registry = ModelRegistry::new();
    let model_config = model_config();
    let metadata = ModelMetadata {
        model_id: "muse-spark",
        tokenizer: &ByteTokenizer,
        config: &model_config,
    };
    let spec = registry.lookup(&metadata).expect("the family's spec");
    let sampling = spec.video_frame_sampling_with(Some(&config));
    assert_eq!(
        sampling,
        FrameSampling::TemporalGroups {
            frame_factor: 2,
            max_frames: 600
        }
    );
    let fetch = VideoFetchConfig {
        sample_fps: spec
            .default_video_sample_fps_with(Some(&config))
            .expect("the reference's rate"),
        sampling,
        ..VideoFetchConfig::default()
    };

    for case in &document.video_cases {
        let name = &case.name;
        // The connector's pick of source frames is the reference's.
        let indices = sampled_frame_indices(case.source.frames, case.source.fps, fetch);
        assert_eq!(indices, case.frame_indices, "{name}: sampled frames");
        let frames: Vec<DynamicImage> = indices
            .iter()
            .map(|&index| {
                make_seeded_image(
                    case.width,
                    case.height,
                    case.source.seed.wrapping_add(index as u8),
                )
            })
            .collect();
        let out = processor
            .preprocess_video(&frames, &config)
            .unwrap_or_else(|e| panic!("{name}: preprocess_video: {e}"));
        assert_eq!(
            int_tensor(&out, "video_grid_thw"),
            case.grid_thw,
            "{name}: video_grid_thw"
        );
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: tokens"
        );
        check_bytes(name, &out, &case.shape, &case.fnv1a_f32, &case.fnv1a_bf16);

        // The spec's replacement from the decoder's record of the sampling.
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: case.source.fps,
                frame_indices: indices.clone(),
            }),
        }];
        let replacements = spec
            .prompt_replacements_with_media(&metadata, &out, Modality::Video, &media, &config)
            .expect("replacements");
        assert_eq!(
            decoded(&replacements[0]),
            case.expansion,
            "{name}: expansion"
        );
        let ranges = replacements[0]
            .feature_ranges
            .as_ref()
            .expect("feature ranges");
        assert_eq!(
            ranges.len(),
            case.stamps.len(),
            "{name}: one range per group"
        );
        let per_group = case.tokens / case.grid_thw[0] as usize;
        for range in ranges {
            assert_eq!(range.length, per_group, "{name}: range length");
            assert!(
                replacements[0].tokens[range.offset..range.offset + range.length]
                    .iter()
                    .all(|&token| token == 200_091 as TokenId),
                "{name}: ranges cover the video tokens"
            );
        }
    }
}
