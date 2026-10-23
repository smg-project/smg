//! Microbench: SMG (Rust) JPEG decode (libjpeg-turbo) + Qwen3-VL preprocess,
//! on a real image + the real model preprocessor config. Compare against the
//! HF/PIL path that vLLM uses (scripts: bench_hf_preprocess.py).
//!
//! Run:
//!   REAL_JPEG=/path/x.jpg PP_CONFIG=/path/preprocessor_config.json \
//!     cargo test -p llm-multimodal --test decode_preprocess_bench -- --ignored --nocapture
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]
use std::time::Instant;

use llm_multimodal::{
    jpeg_turbo,
    types::Modality,
    vision::{
        preprocessor_config::PreProcessorConfig, processors::Qwen3VLProcessor, VisionPreProcessor,
    },
};

#[test]
#[ignore = "perf microbench; needs REAL_JPEG + PP_CONFIG"]
fn bench_decode_preprocess() {
    let jpeg_path = std::env::var("REAL_JPEG").expect("set REAL_JPEG to a JPEG path");
    let cfg_path = std::env::var("PP_CONFIG").expect("set PP_CONFIG to preprocessor_config.json");

    let bytes = std::fs::read(&jpeg_path).expect("read jpeg");
    let config =
        PreProcessorConfig::from_json(&std::fs::read_to_string(&cfg_path).expect("read config"))
            .expect("parse preprocessor config");
    let proc = Qwen3VLProcessor::from_config_for(&config, Modality::Image);

    // warmup
    let img = jpeg_turbo::decode_jpeg_rgb(&bytes).expect("turbojpeg decode");
    let _ = proc
        .preprocess(std::slice::from_ref(&img))
        .expect("preprocess");

    let n_dec = 300usize;
    let t0 = Instant::now();
    for _ in 0..n_dec {
        let _ = jpeg_turbo::decode_jpeg_rgb(&bytes).unwrap();
    }
    let dec_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_dec as f64;

    let n_pp = 200usize;
    let t1 = Instant::now();
    for _ in 0..n_pp {
        let _ = proc.preprocess(std::slice::from_ref(&img)).unwrap();
    }
    let pp_ms = t1.elapsed().as_secs_f64() * 1000.0 / n_pp as f64;

    eprintln!(
        "image: {}x{}  ({} bytes jpeg)",
        img.width(),
        img.height(),
        bytes.len()
    );
    eprintln!("SMG(Rust) decode  (libjpeg-turbo): {dec_ms:.3} ms/img  [{n_dec} iters]");
    eprintln!("SMG(Rust) preprocess (Qwen3-VL)  : {pp_ms:.3} ms/img  [{n_pp} iters]");
    eprintln!(
        "SMG(Rust) decode+preprocess total: {:.3} ms/img",
        dec_ms + pp_ms
    );
}
