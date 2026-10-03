//! Mandatory HuggingFace golden checks for Qwen image preprocessing.
#![allow(clippy::expect_used, clippy::panic)]

use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::{
    processor::ModelSpecificValue, PreProcessorConfig, Qwen2VLProcessor, Qwen3VLProcessor,
    VisionPreProcessor,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenDocument {
    generator: String,
    pillow: String,
    transformers: String,
    cases: Vec<GoldenCase>,
    video_cases: Vec<GoldenVideoCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    model: String,
    width: u32,
    height: u32,
    shape: Vec<usize>,
    grid_thw: Vec<i64>,
    fnv1a_patch_u8: String,
}

#[derive(Deserialize)]
struct GoldenVideoCase {
    model: String,
    width: u32,
    height: u32,
    frame_count: usize,
    shape: Vec<usize>,
    grid_thw: Vec<i64>,
    fnv1a_patch_u8: String,
}

fn make_image(width: u32, height: u32) -> DynamicImage {
    make_seeded_image(width, height, 0)
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

fn image_grid(result: &llm_multimodal::vision::PreprocessedEncoderInputs) -> &[i64] {
    match result.model_specific.get("image_grid_thw") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[1, 3]);
            data
        }
        value => panic!("expected image_grid_thw IntTensor, got {value:?}"),
    }
}

fn video_grid(result: &llm_multimodal::vision::PreprocessedEncoderInputs) -> &[i64] {
    match result.model_specific.get("video_grid_thw") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[1, 3]);
            data
        }
        value => panic!("expected video_grid_thw IntTensor, got {value:?}"),
    }
}

fn fnv1a_patch_u8(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        let byte = (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn config(mean: [f64; 3], std: [f64; 3]) -> PreProcessorConfig {
    PreProcessorConfig {
        do_resize: Some(true),
        do_normalize: Some(false),
        image_mean: Some(mean.to_vec()),
        image_std: Some(std.to_vec()),
        resampling: Some(3),
        ..Default::default()
    }
}

fn check_case(processor: &dyn VisionPreProcessor, config: &PreProcessorConfig, case: &GoldenCase) {
    let result = processor
        .preprocess(&[make_image(case.width, case.height)], config)
        .expect("Qwen golden preprocessing failed");
    assert_eq!(result.encoder_input.shape(), case.shape);

    assert_eq!(image_grid(&result), &case.grid_thw);

    let expected =
        u64::from_str_radix(&case.fnv1a_patch_u8, 16).expect("invalid golden FNV-1a fingerprint");
    let encoder_f32 = result.encoder_input.as_f32();
    let values = encoder_f32
        .as_slice_memory_order()
        .expect("Qwen encoder input must be contiguous");
    assert_eq!(
        fnv1a_patch_u8(values),
        expected,
        "{} {}x{} patchified pixels differ from HuggingFace; first values: {:?}",
        case.model,
        case.width,
        case.height,
        // Keep a small sample in failure output to distinguish resize/layout
        // regressions from one-bit normalization differences.
        &values[..values.len().min(24)]
    );
}

fn check_video_case(
    processor: &Qwen3VLProcessor,
    config: &PreProcessorConfig,
    case: &GoldenVideoCase,
) {
    assert_eq!(case.model, "qwen3_vl");
    let seeds = [3, 101, 177];
    assert_eq!(case.frame_count, seeds.len());
    let frames = seeds
        .into_iter()
        .map(|seed| make_seeded_image(case.width, case.height, seed))
        .collect::<Vec<_>>();
    let result = processor
        .preprocess_video(&frames, config)
        .expect("Qwen video golden preprocessing failed");
    assert_eq!(result.encoder_input.shape(), case.shape);
    assert_eq!(video_grid(&result), &case.grid_thw);

    let expected = u64::from_str_radix(&case.fnv1a_patch_u8, 16)
        .expect("invalid video golden FNV-1a fingerprint");
    let encoder_f32 = result.encoder_input.as_f32();
    let values = encoder_f32
        .as_slice_memory_order()
        .expect("Qwen video encoder input must be contiguous");
    assert_eq!(
        fnv1a_patch_u8(values),
        expected,
        "Qwen3-VL {}x{}x{} patchified video differs from HF/Pillow",
        case.width,
        case.height,
        case.frame_count
    );
}

#[test]
fn qwen_preprocessing_matches_huggingface_golden() {
    let golden: GoldenDocument = serde_json::from_str(include_str!(
        "fixtures/golden/qwen_preprocess_fingerprints.json"
    ))
    .expect("invalid checked-in Qwen golden fixture");
    assert_eq!(golden.generator, "generate_qwen_preprocess_fingerprints.py");
    assert!(!golden.transformers.is_empty());
    assert!(!golden.pillow.is_empty());
    assert_eq!(golden.cases.len(), 4, "Qwen golden coverage changed");
    assert_eq!(
        golden.video_cases.len(),
        1,
        "Qwen video golden coverage changed"
    );

    let qwen2 = Qwen2VLProcessor::new();
    let qwen2_config = config(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    );
    let qwen3 = Qwen3VLProcessor::new();
    let qwen3_config = config([0.5; 3], [0.5; 3]);

    for case in &golden.cases {
        match case.model.as_str() {
            "qwen2_vl" => check_case(&qwen2, &qwen2_config, case),
            "qwen3_vl" => check_case(&qwen3, &qwen3_config, case),
            model => panic!("unknown Qwen golden model {model}"),
        }
    }
    for case in &golden.video_cases {
        check_video_case(&qwen3, &qwen3_config, case);
    }
}
