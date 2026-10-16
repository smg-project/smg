//! The one-pass wire write against the two-pass path over the crate's image
//! fixtures: the same bytes, byte for byte, for every image, pixel-byte
//! processor and wire dtype. `--nocapture` prints each wire's digest, for a
//! comparison across builds.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use image::DynamicImage;
use llm_multimodal::{
    encoder_inputs::EncoderDtype,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processors::{Qwen2VLProcessor, Qwen3VLProcessor},
        VisionPreProcessor,
    },
};

fn image_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            image_files(&path, out);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("jpg" | "jpeg" | "png")
        ) {
            out.push(path);
        }
    }
}

/// Every decodable image under the fixtures, by its path there.
fn fixture_images() -> Vec<(String, DynamicImage)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images");
    let mut paths = Vec::new();
    image_files(&root, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| {
            let name = path.strip_prefix(&root).unwrap().display().to_string();
            image::open(&path).ok().map(|image| (name, image))
        })
        .collect()
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the wires' digests, for a comparison across builds"
)]
fn the_one_pass_wire_of_every_fixture_image_equals_the_two_pass_bytes() {
    let config = PreProcessorConfig::from_json(
        r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
    )
    .unwrap();
    let processors: [(&str, Box<dyn VisionPreProcessor>); 2] = [
        ("qwen3_vl", Box::new(Qwen3VLProcessor::new())),
        ("qwen2_vl", Box::new(Qwen2VLProcessor::new())),
    ];
    let images = fixture_images();
    assert!(
        images.len() >= 10,
        "{} fixture images decoded",
        images.len()
    );
    let mut compared = 0usize;
    for (processor_name, processor) in &processors {
        for (file, image) in &images {
            let batch = std::slice::from_ref(image);
            let two_pass = match processor.preprocess(batch, &config) {
                Ok(two_pass) => two_pass,
                Err(error) => {
                    // A size the rule refuses is refused on both paths.
                    let one_pass = processor
                        .preprocess_wire(batch, &config, EncoderDtype::BFloat16)
                        .expect("the pixel-byte processors write wires");
                    assert!(one_pass.is_err(), "{processor_name} {file}: {error}");
                    continue;
                }
            };
            for dtype in [
                EncoderDtype::BFloat16,
                EncoderDtype::Float16,
                EncoderDtype::Float32,
            ] {
                let one_pass = processor
                    .preprocess_wire(batch, &config, dtype)
                    .expect("the pixel-byte processors write wires")
                    .unwrap();
                assert_eq!(
                    one_pass.encoder_input.shape(),
                    two_pass.encoder_input.shape(),
                    "{processor_name} {file} {dtype}"
                );
                let expected = two_pass.encoder_input.write_as(dtype);
                let wire = one_pass.encoder_input.into_bytes_as(dtype).unwrap();
                assert!(
                    wire == expected,
                    "{processor_name} {file} {dtype}: the one-pass wire differs from the two-pass bytes"
                );
                println!(
                    "{processor_name} {file} {dtype} {} bytes {}",
                    wire.len(),
                    blake3::hash(&wire).to_hex()
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 30, "{compared} wires compared");
}
