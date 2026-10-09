//! Gemma 4 image preprocessing pinned to transformers' `Gemma4ImageProcessor`
//! (the torchvision backend, the processor the engine runs for images and
//! for video frames).
//!
//! `scripts/generate_gemma4_preprocess_fingerprints.py` runs the reference
//! over synthetic seeded images (rebuilt here with the same formula) and PNG
//! fixtures (PNG so PIL and the `image` crate decode identical pixels), at the
//! image budget and at the video-frame budget, recording the resized size, the
//! soft-token count and FNV-1a fingerprints of the exact float32 patch bytes,
//! of the bfloat16 bits the engine receives and of the int64 position ids.
//! This test reproduces every one of them, padding included.
#![allow(clippy::expect_used, clippy::panic)]

use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    registry::{ModelMetadata, ModelRegistry, Tokenizer},
    vision::{
        preprocessor_config::PreProcessorConfig, processor::ModelSpecificValue,
        processors::Gemma4Processor, PreprocessedEncoderInputs, VisionPreProcessor,
        VisionProcessorRegistry,
    },
};
use serde::Deserialize;
use serde_json::json;

/// SHA-256 of the reference `image_processing_gemma4.py` the fixtures were
/// recorded from; a regeneration against a different file fails here.
const REFERENCE_SHA256: &str = "5d280d5448b1c219183a27e95b6aa7178b350275772d07881c91c65f13fa815e";

/// The released checkpoints' `processor_config.json` sections.
const IMAGE_CONFIG: &str = r#"{
    "do_convert_rgb": true, "do_normalize": false, "do_rescale": true, "do_resize": true,
    "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
    "image_processor_type": "Gemma4ImageProcessor", "image_seq_length": 280,
    "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3, "resample": 3,
    "rescale_factor": 0.00392156862745098
}"#;
const VIDEO_CONFIG: &str = r#"{
    "do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
    "do_sample_frames": true, "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
    "max_soft_tokens": 70, "num_frames": 32, "patch_size": 16, "pooling_kernel_size": 3,
    "resample": 3, "rescale_factor": 0.00392156862745098, "return_metadata": false,
    "video_processor_type": "Gemma4VideoProcessor"
}"#;

#[derive(Deserialize)]
struct GoldenDocument {
    reference: String,
    reference_sha256: String,
    cases: Vec<GoldenCase>,
    video_cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    source: Source,
    width: u32,
    height: u32,
    max_soft_tokens: usize,
    target_hw: Target,
    num_soft_tokens: Tokens,
    shape: Vec<usize>,
    positions_shape: Vec<usize>,
    fnv1a_f32: String,
    fnv1a_bf16: String,
    fnv1a_positions_i64: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Source {
    Seed { seed: u8 },
    File { file: String },
    Seeds { seeds: Vec<u8> },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Target {
    One([usize; 2]),
    Many(Vec<[usize; 2]>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Tokens {
    One(usize),
    Many(Vec<usize>),
}

const GOLDEN: &str = include_str!("fixtures/golden/gemma4_preprocess_fingerprints.json");

/// The seeded pattern shared with the other preprocessing goldens and the
/// generator: `R=(x*7+y*3)%256`, `G=(x*5+y*11)%256`, `B=(x+y*2)%256`, each
/// plus the seed with u8 wraparound.
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
        Source::Seeds { seeds } => seeds
            .iter()
            .map(|seed| make_seeded_image(case.width, case.height, *seed))
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

fn positions<'a>(out: &'a PreprocessedEncoderInputs, key: &str) -> (&'a [i64], &'a [usize]) {
    match out.model_specific.get(key) {
        Some(ModelSpecificValue::IntTensor { data, shape }) => (data, shape),
        other => panic!("{key} must be an int tensor, got {other:?}"),
    }
}

fn check(case: &GoldenCase, out: &PreprocessedEncoderInputs, positions_key: &str) {
    let name = &case.name;
    let expected_tokens = match &case.num_soft_tokens {
        Tokens::One(count) => vec![*count],
        Tokens::Many(counts) => counts.clone(),
    };
    let targets = match &case.target_hw {
        Target::One(target) => vec![*target],
        Target::Many(targets) => targets.clone(),
    };
    for (target, &tokens) in targets.iter().zip(&expected_tokens) {
        assert_eq!(
            (target[0] / 16) * (target[1] / 16) / 9,
            tokens,
            "{name}: recorded target"
        );
    }
    assert_eq!(
        out.encoder_input.shape(),
        case.shape.as_slice(),
        "{name}: shape"
    );
    let (position_data, position_shape) = positions(out, positions_key);
    assert_eq!(
        position_shape,
        case.positions_shape.as_slice(),
        "{name}: positions shape"
    );
    assert_eq!(
        fnv1a(position_data.iter().flat_map(|v| v.to_le_bytes())),
        case.fnv1a_positions_i64,
        "{name}: position ids differ from the reference"
    );
    let flat = out.encoder_input_flat();
    let values: &[f32] = flat.as_ref();
    assert_eq!(
        fnv1a(values.iter().flat_map(|v| v.to_le_bytes())),
        case.fnv1a_f32,
        "{name}: float32 patch bytes differ from the reference"
    );
    assert_eq!(
        fnv1a(
            values
                .iter()
                .flat_map(|v| llm_multimodal::f32_to_bf16_bits(*v).to_le_bytes())
        ),
        case.fnv1a_bf16,
        "{name}: bfloat16 patch bits differ from the reference"
    );
}

#[test]
fn preprocess_matches_the_reference_fingerprints() {
    let document: GoldenDocument =
        serde_json::from_str(GOLDEN).expect("golden document must match the schema");
    assert!(
        document.reference.contains("image_processing_gemma4.py"),
        "unexpected reference {}",
        document.reference
    );
    assert_eq!(
        document.reference_sha256, REFERENCE_SHA256,
        "golden fixtures were generated from an unexpected reference"
    );
    assert!(document.cases.len() >= 40, "expected the recorded case set");

    let processor = Gemma4Processor::new();
    let image_config = PreProcessorConfig::from_json(IMAGE_CONFIG).expect("image config");
    let video_config = PreProcessorConfig::from_json(VIDEO_CONFIG).expect("video config");
    for case in &document.cases {
        let images = images_for(case);
        assert_eq!(
            (images[0].width(), images[0].height()),
            (case.width, case.height),
            "{}",
            case.name
        );
        let Tokens::One(tokens) = case.num_soft_tokens else {
            panic!("{}: one image, one count", case.name);
        };
        match case.max_soft_tokens {
            280 => {
                let out = processor
                    .preprocess(&images, &image_config)
                    .unwrap_or_else(|e| panic!("{}: preprocess failed: {e}", case.name));
                assert_eq!(
                    out.feature_token_counts,
                    vec![tokens],
                    "{}: tokens",
                    case.name
                );
                assert_eq!(
                    processor.calculate_num_tokens(case.width, case.height, &image_config),
                    tokens,
                    "{}: calculate_num_tokens",
                    case.name
                );
                check(case, &out, "pixel_position_ids");
            }
            70 => {
                // The engine's video frames: the image processor at the video budget.
                let out = processor
                    .preprocess_video(&images, &video_config)
                    .unwrap_or_else(|e| panic!("{}: preprocess_video failed: {e}", case.name));
                assert_eq!(
                    out.feature_token_counts,
                    vec![tokens],
                    "{}: tokens",
                    case.name
                );
                check(case, &out, "pixel_position_ids_videos");
            }
            other => panic!("{}: unexpected budget {other}", case.name),
        }
    }
    for case in &document.video_cases {
        let frames = images_for(case);
        let out = processor
            .preprocess_video(&frames, &video_config)
            .unwrap_or_else(|e| panic!("{}: preprocess_video failed: {e}", case.name));
        let Tokens::Many(per_frame) = &case.num_soft_tokens else {
            panic!("{}: a clip records one count per frame", case.name);
        };
        assert_eq!(
            out.feature_token_counts,
            vec![per_frame.iter().sum::<usize>()],
            "{}: tokens",
            case.name
        );
        let (counts, _) = positions(&out, "video_frame_counts");
        assert_eq!(
            counts,
            &[frames.len() as i64],
            "{}: frame counts",
            case.name
        );
        check(case, &out, "pixel_position_ids_videos");
    }
}

struct VocabTokenizer;

impl Tokenizer for VocabTokenizer {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|image|>" => Some(258880),
            "<|video|>" => Some(258884),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match id {
            258880 => Some("<|image|>".to_string()),
            258884 => Some("<|video|>".to_string()),
            _ => None,
        }
    }

    fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
        Some(Vec::new())
    }
}

/// Both registries resolve the family: the model spec by `model_type`, the
/// vision processor by id or `model_type`.
#[test]
fn both_registries_resolve_the_family() {
    let config = json!({
        "model_type": "gemma4",
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "video_token_id": 258884
    });
    let tokenizer = VocabTokenizer;
    let metadata = ModelMetadata {
        model_id: "/models/local-checkpoint",
        tokenizer: &tokenizer,
        config: &config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).expect("gemma4 spec");
    assert_eq!(spec.name(), "gemma4");
    assert_eq!(spec.placeholder_token(&metadata).unwrap(), "<|image|>");
    let processors = VisionProcessorRegistry::with_defaults();
    assert_eq!(
        processors
            .find("/models/local-checkpoint", Some("gemma4"))
            .map(|processor| processor.model_name()),
        Some("gemma4")
    );
}
