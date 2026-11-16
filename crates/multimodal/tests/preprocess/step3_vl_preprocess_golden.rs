//! Step3-VL image preprocessing pinned to the serving engine's own processor
//! module for the family (the checkpoints ship no preprocessor config).
//!
//! `scripts/generate_step3_vl_preprocess_fingerprints.py` runs that module
//! over synthetic seeded images (rebuilt here with the same formula),
//! recording per image the crop count, the newline flags, the token count,
//! the tensor shapes and FNV-1a fingerprints of the exact float32 bytes of
//! the global view and of the crops, and of the bfloat16 bits the engine
//! casts them to. This test reproduces every one of them.

use llm_multimodal::vision::{
    processor::ModelSpecificValue,
    processors::step3_vl::{
        IMAGE_FEATURE_TOKENS, NUM_PATCHES, PATCH_FEATURE_TOKENS, PATCH_NEWLINE_MASK,
        PATCH_PIXEL_VALUES,
    },
    Step3VLProcessor, VisionPreProcessor,
};
use serde::Deserialize;

use super::{fnv1a, fnv1a_f32, make_seeded_image};

/// SHA-256 of the reference module the fixtures were recorded from; a
/// regeneration against a different file fails here.
const REFERENCE_SHA256: &str = "ce0265261a6bffaf40b98911ef5b9babe30d20443af83a4c8ecf251ce6a861fa";

#[derive(Deserialize)]
struct GoldenDocument {
    reference: String,
    reference_sha256: String,
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    source: Source,
    width: u32,
    height: u32,
    num_patches: usize,
    newline_mask: Vec<i64>,
    num_tokens: usize,
    shape: Vec<usize>,
    patches_shape: Vec<usize>,
    fnv1a_f32: String,
    fnv1a_bf16: String,
    patches_fnv1a_f32: String,
    patches_fnv1a_bf16: String,
}

#[derive(Deserialize)]
struct Source {
    seed: u8,
}

const GOLDEN: &str = include_str!("../fixtures/golden/step3_vl_preprocess_fingerprints.json");

/// float32 -> bfloat16 bits with round-to-nearest-even, the conversion
/// `torch.Tensor.to(torch.bfloat16)` applies to finite values.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn fnv1a_bf16(values: &[f32]) -> String {
    fnv1a(
        values
            .iter()
            .flat_map(|&value| bf16_bits(value).to_le_bytes()),
    )
}

#[test]
fn preprocess_matches_the_reference_fingerprints() {
    let document: GoldenDocument =
        serde_json::from_str(GOLDEN).expect("golden document must match the schema");
    assert!(
        document.reference.ends_with("step3_vl.py"),
        "unexpected reference {}",
        document.reference
    );
    assert_eq!(
        document.reference_sha256, REFERENCE_SHA256,
        "golden fixtures were generated from an unexpected reference"
    );
    assert!(document.cases.len() >= 14, "expected the recorded case set");

    let processor = Step3VLProcessor::new();
    for case in &document.cases {
        let image = make_seeded_image(case.width, case.height, case.source.seed);
        let out = processor
            .preprocess(&[&image])
            .unwrap_or_else(|e| panic!("{}: preprocess failed: {e}", case.name));

        assert_eq!(
            out.encoder_input.shape(),
            case.shape.as_slice(),
            "{}: shape",
            case.name
        );
        // The fixture's token count is the reference's prompt layout; the
        // features behind it are the image tokens inside that layout.
        assert_eq!(
            processor.plan(case.width, case.height).num_tokens(),
            case.num_tokens,
            "{}: tokens",
            case.name
        );
        let features = case.num_patches * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS;
        assert_eq!(
            out.feature_token_counts,
            vec![features],
            "{}: features",
            case.name
        );
        assert_eq!(
            processor
                .calculate_num_tokens(case.width, case.height)
                .unwrap(),
            features,
            "{}: features from the size",
            case.name
        );
        let Some(ModelSpecificValue::IntTensor { data: counts, .. }) =
            out.model_specific.get(NUM_PATCHES)
        else {
            panic!("{}: {NUM_PATCHES} must be an int tensor", case.name);
        };
        assert_eq!(counts, &[case.num_patches as i64], "{}: crops", case.name);
        let Some(ModelSpecificValue::IntTensor { data: newlines, .. }) =
            out.model_specific.get(PATCH_NEWLINE_MASK)
        else {
            panic!("{}: {PATCH_NEWLINE_MASK} must be an int tensor", case.name);
        };
        assert_eq!(newlines, &case.newline_mask, "{}: newlines", case.name);
        let Some(ModelSpecificValue::Tensor {
            data: patches,
            shape: patches_shape,
        }) = out.model_specific.get(PATCH_PIXEL_VALUES)
        else {
            panic!("{}: {PATCH_PIXEL_VALUES} must be a float tensor", case.name);
        };
        assert_eq!(
            patches_shape, &case.patches_shape,
            "{}: crops shape",
            case.name
        );

        let view = out.encoder_input.flat_f32();
        assert_eq!(fnv1a_f32(&view), case.fnv1a_f32, "{}: view f32", case.name);
        assert_eq!(
            fnv1a_bf16(&view),
            case.fnv1a_bf16,
            "{}: view bf16",
            case.name
        );
        assert_eq!(
            fnv1a_f32(patches),
            case.patches_fnv1a_f32,
            "{}: crops f32",
            case.name
        );
        assert_eq!(
            fnv1a_bf16(patches),
            case.patches_fnv1a_bf16,
            "{}: crops bf16",
            case.name
        );
    }
}
