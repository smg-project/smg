//! DeepSeek-V4.1 image preprocessing pinned to the checkpoint's reference
//! `inference/image_processor.py`.
//!
//! `scripts/generate_deepseek_v41_preprocess_fingerprints.py` runs the
//! reference over synthetic seeded images (rebuilt here with the same formula)
//! and PNG fixtures (PNG so PIL and the `image` crate decode identical
//! pixels), recording the grids, the token count and FNV-1a fingerprints of
//! the exact float32 patch bytes before the bfloat16 cast and of the bfloat16
//! bits the engine receives. This test reproduces every one of them.

use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::{
    preprocessor_config::PreProcessorConfig, processor::ModelSpecificValue,
    processors::DeepseekV41Processor, VisionPreProcessor,
};
use serde::Deserialize;

/// SHA-256 of the reference `inference/image_processor.py` the fixtures were
/// recorded from; a regeneration against a different file fails here.
const REFERENCE_SHA256: &str = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272";

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
    shape: Vec<usize>,
    vit_grid: Vec<i64>,
    llm_grid: Vec<i64>,
    num_tokens: usize,
    fnv1a_types_i64: String,
    fnv1a_f32: String,
    fnv1a_bf16: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Source {
    Seed { seed: u8 },
    File { file: String },
}

const GOLDEN: &str = include_str!("fixtures/golden/deepseek_v41_preprocess_fingerprints.json");

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

/// float32 -> bfloat16 bits with round-to-nearest-even, the conversion
/// `torch.Tensor.to(torch.bfloat16)` applies to finite values.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

#[expect(
    clippy::panic,
    reason = "test helper — a missing tensor is a test failure"
)]
fn int_tensor<'a>(
    out: &'a llm_multimodal::vision::PreprocessedEncoderInputs,
    key: &str,
) -> &'a [i64] {
    match out.model_specific.get(key) {
        Some(ModelSpecificValue::IntTensor { data, .. }) => data,
        other => panic!("{key} must be an int tensor, got {other:?}"),
    }
}

#[test]
fn preprocess_matches_the_reference_fingerprints() {
    let document: GoldenDocument =
        serde_json::from_str(GOLDEN).expect("golden document must match the schema");
    assert!(
        document.reference.contains("image_processor.py"),
        "unexpected reference {}",
        document.reference
    );
    assert_eq!(
        document.reference_sha256, REFERENCE_SHA256,
        "golden fixtures were generated from an unexpected reference"
    );
    assert!(document.cases.len() >= 12, "expected the recorded case set");

    let processor = DeepseekV41Processor::new();
    let config = PreProcessorConfig::default();
    for case in &document.cases {
        let image = match &case.source {
            Source::Seed { seed } => make_seeded_image(case.width, case.height, *seed),
            Source::File { file } => {
                let path = format!(
                    "{}/tests/fixtures/images/{file}",
                    env!("CARGO_MANIFEST_DIR")
                );
                image::open(&path).unwrap_or_else(|e| panic!("{}: decode {path}: {e}", case.name))
            }
        };
        assert_eq!(
            (image.width(), image.height()),
            (case.width, case.height),
            "{}",
            case.name
        );

        let out = processor
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap_or_else(|e| panic!("{}: preprocess failed: {e}", case.name));

        assert_eq!(
            out.encoder_input.shape(),
            case.shape.as_slice(),
            "{}: shape",
            case.name
        );
        assert_eq!(
            int_tensor(&out, "vit_grid"),
            case.vit_grid.as_slice(),
            "{}: vit_grid",
            case.name
        );
        assert_eq!(
            int_tensor(&out, "llm_grid"),
            case.llm_grid.as_slice(),
            "{}: llm_grid",
            case.name
        );
        assert_eq!(
            out.feature_token_counts,
            vec![case.num_tokens],
            "{}: tokens",
            case.name
        );
        assert_eq!(
            fnv1a(
                int_tensor(&out, "types")
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
            ),
            case.fnv1a_types_i64,
            "{}: types",
            case.name
        );

        let flat = out.encoder_input_flat();
        let values: &[f32] = flat.as_ref();
        assert_eq!(
            fnv1a(values.iter().flat_map(|v| v.to_le_bytes())),
            case.fnv1a_f32,
            "{}: float32 patch bytes differ from the reference",
            case.name
        );
        assert_eq!(
            fnv1a(values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes())),
            case.fnv1a_bf16,
            "{}: bfloat16 patch bits differ from the reference",
            case.name
        );
    }
}

#[test]
fn bf16_rounding_is_round_to_nearest_even() {
    // 1.0 is exact; the value just above 1.0 in bf16 terms rounds by RNE.
    assert_eq!(bf16_bits(1.0), 0x3F80);
    assert_eq!(bf16_bits(-1.0), 0xBF80);
    // 1 + 2^-8 sits exactly halfway between 1.0 and the next bf16 (1 + 2^-7):
    // ties go to the even mantissa, i.e. 1.0.
    assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3F80);
    // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 (odd) and 1 + 2^-6 (even).
    assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3F82);
    // 1 + 5 * 2^-8 is halfway between 1 + 2^-6 (even) and 1 + 3 * 2^-7 (odd).
    assert_eq!(bf16_bits(1.0 + 5.0 / 256.0), 0x3F82);
    assert_eq!(bf16_bits(0.5), 0x3F00);
    assert_eq!(bf16_bits(-0.5), 0xBF00);
}
