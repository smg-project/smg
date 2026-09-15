//! Golden parity checks against the pinned DeepSeek-V4-Vision reference run.
#![allow(clippy::expect_used, clippy::panic)]

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use llm_multimodal::{
    jpeg_turbo,
    vision::{deepseek_v4_geometry, transforms::pad_contain_pil, ModelSpecificValue},
    DeepseekV4VisionProcessor, PreProcessorConfig, VisionPreProcessor,
};
use serde::Deserialize;

const FIXTURE_ROOT_ENV: &str = "DSV4_VISION_FIXTURE_ROOT";
const REFERENCE_ROOT_ENV: &str = "DSV4_VISION_REFERENCE_ROOT";

#[derive(Debug, Deserialize)]
struct GoldenDocument {
    fixtures: BTreeMap<String, GoldenFixture>,
}

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    input_width: u32,
    input_height: u32,
    best_width: usize,
    best_height: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,
    num_tokens: usize,
    initial_num_tokens: usize,
    safe_resize_iterations: usize,
    aspect_clamped: bool,
    min_pixel_upscale: bool,
    patch_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct PadGoldenDocument {
    pillow: String,
    cases: Vec<PadGoldenCase>,
}

#[derive(Debug, Deserialize)]
struct PadGoldenCase {
    input: [u32; 2],
    output: [u32; 2],
    fnv1a_rgb8: String,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn deepseek_v4_vision_pillow_pad_golden() {
    let fixture: PadGoldenDocument =
        serde_json::from_str(include_str!("fixtures/deepseek_v4_vision/pillow_pad.json"))
            .expect("invalid Pillow pad fixture");
    assert_eq!(fixture.pillow, "12.3.0");
    assert_eq!(fixture.cases.len(), 12);
    for case in fixture.cases {
        let source = image::RgbImage::from_fn(case.input[0], case.input[1], |x, y| {
            image::Rgb([
                ((x * 37 + y * 11 + 3) % 256) as u8,
                ((x * 17 + y * 29 + 5) % 256) as u8,
                ((x * 7 + y * 43 + 9) % 256) as u8,
            ])
        });
        let result = pad_contain_pil(
            &image::DynamicImage::ImageRgb8(source),
            case.output[0],
            case.output[1],
            image::Rgb([127, 127, 127]),
        )
        .to_rgb8();
        let expected = u64::from_str_radix(&case.fnv1a_rgb8, 16).unwrap();
        assert_eq!(
            fnv1a(result.as_raw()),
            expected,
            "Pillow pad {:?} -> {:?}",
            case.input,
            case.output
        );
    }
}

fn uint_tensor<'a>(result: &'a llm_multimodal::PreprocessedEncoderInputs, name: &str) -> &'a [u32] {
    match result.model_specific.get(name) {
        Some(ModelSpecificValue::UintTensor { data, shape }) => {
            assert_eq!(shape, &[1], "{name} must be item-batched");
            data
        }
        value => panic!("expected {name} UintTensor, got {value:?}"),
    }
}

#[expect(
    clippy::print_stderr,
    reason = "portable golden tests report explicit skips"
)]
fn external_fixture_roots(test_name: &str) -> Option<(PathBuf, PathBuf)> {
    let fixture_root = env::var_os(FIXTURE_ROOT_ENV);
    let reference_root = env::var_os(REFERENCE_ROOT_ENV);
    let (Some(fixture_root), Some(reference_root)) = (fixture_root, reference_root) else {
        eprintln!(
            "skipping {test_name}: set {FIXTURE_ROOT_ENV} and {REFERENCE_ROOT_ENV} to the approved external DeepSeek artifacts"
        );
        return None;
    };
    Some((PathBuf::from(fixture_root), PathBuf::from(reference_root)))
}

fn image_path(id: &str, fixture_root: &Path, official_images: &Path) -> PathBuf {
    match id {
        "F10" => official_images.join("carrots.jpeg"),
        "F11" => official_images.join("corn.jpeg"),
        _ => fixture_root.join("images").join(format!("{id}.png")),
    }
}

fn bf16_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let bits = (value.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

#[test]
fn deepseek_v4_vision_golden_all_reference_fixtures() {
    let Some((fixture_root, reference_root)) =
        external_fixture_roots("deepseek_v4_vision_golden_all_reference_fixtures")
    else {
        return;
    };
    let official_images = reference_root.join("harness/examples/images");
    let metadata_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/deepseek_v4_vision/preprocessing.json");
    let document: GoldenDocument = serde_json::from_slice(
        &fs::read(&metadata_path).expect("missing checked-in DeepSeek V4 metadata fixture"),
    )
    .expect("invalid DeepSeek V4 metadata fixture");
    let layout_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/deepseek_v4_vision/layout.json");
    let layout: serde_json::Value = serde_json::from_slice(
        &fs::read(&layout_path).expect("missing checked-in DeepSeek V4 layout fixture"),
    )
    .expect("invalid DeepSeek V4 layout fixture");
    let processor = DeepseekV4VisionProcessor::new();

    for (id, expected) in document.fixtures {
        let source = image_path(&id, &fixture_root, &official_images);
        let source_bytes = fs::read(&source).unwrap_or_else(|error| {
            panic!(
                "missing approved DeepSeek reference image {}: {error}",
                source.display()
            )
        });
        let image = if jpeg_turbo::is_jpeg(&source_bytes) {
            jpeg_turbo::decode_jpeg_rgb(&source_bytes).unwrap_or_else(|| {
                panic!(
                    "{} requires libjpeg-turbo for Pillow-compatible JPEG decode",
                    source.display()
                )
            })
        } else {
            image::load_from_memory(&source_bytes)
                .unwrap_or_else(|error| panic!("failed to decode {}: {error}", source.display()))
        };
        let result = processor
            .preprocess(&[image], &PreProcessorConfig::default())
            .unwrap_or_else(|error| panic!("{id} preprocessing failed: {error}"));
        let geometry = deepseek_v4_geometry(expected.input_width, expected.input_height);
        assert_eq!(geometry.best_width, expected.best_width, "{id} best_width");
        assert_eq!(
            geometry.best_height, expected.best_height,
            "{id} best_height"
        );
        assert_eq!(geometry.n_vit_h, expected.n_vit_h, "{id} n_vit_h");
        assert_eq!(geometry.n_vit_w, expected.n_vit_w, "{id} n_vit_w");
        assert_eq!(geometry.n_llm_h, expected.n_llm_h, "{id} n_llm_h");
        assert_eq!(geometry.n_llm_w, expected.n_llm_w, "{id} n_llm_w");
        assert_eq!(geometry.num_tokens, expected.num_tokens, "{id} num_tokens");
        assert_eq!(
            geometry.initial_num_tokens, expected.initial_num_tokens,
            "{id} initial_num_tokens"
        );
        assert_eq!(
            geometry.safe_resize_iterations, expected.safe_resize_iterations,
            "{id} safe_resize_iterations"
        );
        assert_eq!(geometry.aspect_clamped, expected.aspect_clamped, "{id}");
        assert_eq!(
            geometry.min_pixel_upscale, expected.min_pixel_upscale,
            "{id}"
        );
        assert_eq!(
            result.encoder_input.shape(),
            expected.patch_shape,
            "{id} shape"
        );
        for residue in 0..4 {
            let expected_layout = &layout["fixtures"][&id][residue.to_string()];
            let compress_pad = 3 - residue;
            assert_eq!(
                expected_layout["compress_pad"].as_u64().unwrap() as usize,
                compress_pad,
                "{id} residue {residue} compress_pad"
            );
            assert_eq!(
                expected_layout["block_length"].as_u64().unwrap() as usize,
                geometry.num_tokens + compress_pad,
                "{id} residue {residue} block length"
            );
            assert_eq!(
                expected_layout["num_tokens"].as_u64().unwrap() as usize,
                geometry.num_tokens,
                "{id} residue {residue} feature tokens"
            );
        }
        assert_eq!(result.feature_token_counts, [expected.num_tokens], "{id}");
        assert_eq!(
            uint_tensor(&result, "patches_per_image"),
            &[expected.n_vit_h as u32 * expected.n_vit_w as u32],
            "{id}"
        );
        assert_eq!(
            uint_tensor(&result, "n_vit_h"),
            &[expected.n_vit_h as u32],
            "{id}"
        );
        assert_eq!(
            uint_tensor(&result, "n_vit_w"),
            &[expected.n_vit_w as u32],
            "{id}"
        );
        assert_eq!(
            uint_tensor(&result, "n_llm_h"),
            &[expected.n_llm_h as u32],
            "{id}"
        );
        assert_eq!(
            uint_tensor(&result, "n_llm_w"),
            &[expected.n_llm_w as u32],
            "{id}"
        );
        assert_eq!(
            uint_tensor(&result, "safe_resize_iterations"),
            &[expected.safe_resize_iterations as u32],
            "{id}"
        );

        let golden_path = fixture_root.join("patches").join(format!("{id}.bf16.bin"));
        let golden = fs::read(&golden_path).unwrap_or_else(|error| {
            panic!(
                "missing approved DeepSeek reference tensor {}: {error}",
                golden_path.display()
            )
        });
        let actual = bf16_le_bytes(
            result
                .encoder_input
                .as_slice_memory_order()
                .expect("DeepSeek patches must be contiguous"),
        );
        assert_eq!(actual.len(), golden.len(), "{id} BF16 tensor byte length");
        if let Some(index) = actual
            .iter()
            .zip(&golden)
            .position(|(actual, expected)| actual != expected)
        {
            let start = index.saturating_sub(8);
            let end = (index + 8).min(actual.len());
            panic!(
                "{id} BF16 patch tensor differs at byte {index}; actual={:?}, expected={:?}",
                &actual[start..end],
                &golden[start..end]
            );
        }
    }
}
