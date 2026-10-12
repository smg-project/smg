//! The pixel values the Qwen vision families hand the engine, bit for bit
//! against the reference image processors (the torchvision-backed ones the
//! serving engines run): the per-byte normalization table each channel maps
//! through, the patch bytes, and the normalized tensor as `f32` and as
//! `bfloat16` (what the engine casts it to), on a synthetic gradient holding
//! every byte value in every channel at a size neither family resizes
//! (`fixtures/golden/qwen_pixel_norm.json`, written by
//! `scripts/generate_qwen_pixel_norm_golden.py`).
#![allow(clippy::expect_used, clippy::panic)]

use std::{fs, path::Path};

use image::{DynamicImage, Rgb, RgbImage};
use llm_multimodal::{
    vision::{Qwen2VLProcessor, Qwen3VLProcessor},
    EncoderDtype, PreProcessorConfig, VisionPreProcessor,
};
use serde_json::Value;

const GOLDEN: &str = "tests/fixtures/golden/qwen_pixel_norm.json";

/// Channel 0 is the row, channel 1 the column, channel 2 their sum, each
/// modulo 256: every byte value in every channel.
fn gradient(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        Rgb([(y % 256) as u8, (x % 256) as u8, ((x + y) % 256) as u8])
    }))
}

fn fnv1a(bytes: &[u8]) -> String {
    let mut value: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        value ^= u64::from(byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{value:016x}")
}

fn processor_for(name: &str, config: &PreProcessorConfig) -> Box<dyn VisionPreProcessor> {
    match name {
        "qwen2_vl" => Box::new(Qwen2VLProcessor::from_preprocessor_config(config)),
        "qwen3_vl" => Box::new(Qwen3VLProcessor::from_preprocessor_config(config)),
        other => panic!("no processor for golden entry {other}"),
    }
}

#[test]
fn qwen_pixel_values_match_the_reference_processors() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let golden: Value =
        serde_json::from_str(&fs::read_to_string(crate_root.join(GOLDEN)).expect("golden"))
            .expect("golden json");
    let image = gradient(
        golden["image"]["width"].as_u64().expect("width") as u32,
        golden["image"]["height"].as_u64().expect("height") as u32,
    );

    for (name, entry) in golden["models"].as_object().expect("models") {
        let config = PreProcessorConfig::from_value(entry["preprocessor_config"].clone())
            .expect("preprocessor config");
        let processor = processor_for(name, &config);
        let output = processor
            .preprocess(std::slice::from_ref(&image), &config)
            .expect("preprocess");

        let expected_shape: Vec<usize> = entry["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().expect("dim") as usize)
            .collect();
        assert_eq!(
            output.encoder_input.shape(),
            &expected_shape[..],
            "{name}: shape"
        );

        // The table every byte maps through, per channel, as the reference
        // computes it.
        let norm = output
            .encoder_input
            .pixel_norm()
            .unwrap_or_else(|| panic!("{name}: the processor keeps the pixel bytes"));
        assert_eq!(
            norm.channel_block,
            entry["channel_block"].as_u64().expect("channel_block") as usize,
            "{name}: channel block"
        );
        let table = norm.table_f32();
        let mut mismatches = Vec::new();
        for (channel, bits) in entry["table_bits"]
            .as_array()
            .expect("table")
            .iter()
            .enumerate()
        {
            for (value, want) in bits.as_array().expect("channel table").iter().enumerate() {
                let want = u32::from_str_radix(want.as_str().expect("hex"), 16).expect("bits");
                let got = table[channel][value].to_bits();
                if got != want {
                    mismatches.push((channel, value, got, want));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "{name}: {} of 768 byte values map to another f32 than the reference's; first: \
             channel {} byte {} -> {:08x} ({}), reference {:08x} ({})",
            mismatches.len(),
            mismatches[0].0,
            mismatches[0].1,
            mismatches[0].2,
            f32::from_bits(mismatches[0].2),
            mismatches[0].3,
            f32::from_bits(mismatches[0].3),
        );

        // The whole tensor: the bytes in patch order, then the floats the
        // engine receives in either width.
        let bytes = output.encoder_input.write_as(EncoderDtype::Uint8);
        assert_eq!(
            fnv1a(&bytes),
            entry["fnv1a_patch_u8"].as_str().expect("u8 fingerprint"),
            "{name}: patch bytes"
        );
        assert_eq!(
            fnv1a(&output.encoder_input.write_as(EncoderDtype::Float32)),
            entry["fnv1a_f32"].as_str().expect("f32 fingerprint"),
            "{name}: normalized f32 tensor"
        );
        assert_eq!(
            fnv1a(&output.encoder_input.write_as(EncoderDtype::BFloat16)),
            entry["fnv1a_bf16"].as_str().expect("bf16 fingerprint"),
            "{name}: normalized bfloat16 tensor"
        );
    }
}
