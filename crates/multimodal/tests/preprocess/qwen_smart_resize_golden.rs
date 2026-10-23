//! The resize target of the Qwen-family `smart_resize`, pinned to the
//! reference image processor's values, including the degenerate shapes where
//! a side rounds to zero and the order of the rounding, the floor to one
//! factor and the pixel-budget test decides the grid. The golden comes from
//! `scripts/generate_qwen_smart_resize_golden.py`; its MiniMax-M3 section,
//! whose reference floors first, is checked by the base processor's own
//! unit tests.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::BTreeMap;

use llm_multimodal::vision::{PreProcessorConfig, Qwen2VLProcessor, Qwen3VLProcessor};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    families: BTreeMap<String, Family>,
}

#[derive(Deserialize)]
struct Family {
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    height: usize,
    width: usize,
    #[serde(default)]
    resized: Option<[usize; 2]>,
    #[serde(default)]
    error: Option<String>,
}

type Resize = Box<dyn Fn(usize, usize) -> Option<(usize, usize)>>;

fn preprocessor_config(family: &Family) -> PreProcessorConfig {
    PreProcessorConfig::from_json(&format!(
        r#"{{"patch_size":{},"merge_size":{},"min_pixels":{},"max_pixels":{},"temporal_patch_size":2}}"#,
        family.patch_size, family.merge_size, family.min_pixels, family.max_pixels
    ))
    .unwrap()
}

/// The crate's resize for a Qwen family of the golden, through the processor
/// built from the family's preprocessor configuration.
fn resize_for(name: &str, family: &Family) -> Resize {
    match name {
        "qwen2_vl" => {
            let processor =
                Qwen2VLProcessor::from_preprocessor_config(&preprocessor_config(family));
            Box::new(move |h, w| processor.smart_resize(h, w).ok())
        }
        "qwen3_vl" => {
            let processor =
                Qwen3VLProcessor::from_preprocessor_config(&preprocessor_config(family));
            Box::new(move |h, w| processor.smart_resize(h, w).ok())
        }
        other => panic!("no processor for golden family {other}"),
    }
}

#[test]
fn smart_resize_matches_the_reference_processor() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden/qwen_smart_resize.json"
    );
    let golden: Golden = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for name in ["qwen2_vl", "qwen3_vl"] {
        let family = golden.families.get(name).expect("golden family present");
        assert!(!family.cases.is_empty(), "{name}: no cases");
        let resize = resize_for(name, family);
        for case in &family.cases {
            let got = resize(case.height, case.width);
            let want = case.resized.map(|[h, w]| (h, w));
            if got != want {
                failures.push(format!(
                    "{name}: {}x{} -> {:?}, reference {:?}{}",
                    case.height,
                    case.width,
                    got,
                    want,
                    case.error
                        .as_deref()
                        .map(|e| format!(" (reference error: {e})"))
                        .unwrap_or_default()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "smart_resize differs from the reference processor (height x width -> (height, width)):\n{}",
        failures.join("\n")
    );
}
