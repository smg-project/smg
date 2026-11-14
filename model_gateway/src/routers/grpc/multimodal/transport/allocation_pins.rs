//! What the encoder dtype resolution allocates per media item, pinned at
//! zero on the default and override paths so the next change that copies a
//! cached name fails a test rather than moving a bench inside its noise. The
//! lib's test binary runs under the executable's allocator (`lib.rs`), and
//! each pin reads that allocator's counter for its own thread
//! (`routers::grpc::testing::bytes_allocated_during`), so the shared binary
//! is read one thread at a time.

use std::sync::Arc;

use llm_multimodal::{EncoderDtype, Modality};
use openai_protocol::worker::RuntimeType;

use super::{
    mm_encoder_input_dtype, mm_sglang_encoder_input_dtype, mm_vllm_encoder_input_dtype,
    shared_image_wire_dtype_over, DtypeOverrides,
};
use crate::{
    routers::grpc::{context::WorkerSelection, mode::Mode, testing::bytes_allocated_during},
    worker::{
        test_support::no_health_check, BasicWorkerBuilder, ConnectionMode, ModelCard, Worker,
        WorkerRegistry, WorkerType,
    },
};

/// The label the labelled workers carry; the one copy an assembly keeps of it
/// is the label's own bytes.
const LABEL: &str = "bfloat16";

fn worker(url: &str, runtime: RuntimeType, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new("qwen3-vl"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(runtime)
        .health_config(no_health_check());
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    Arc::new(builder.build())
}

fn selection(url: &str, labels: &[(&str, &str)]) -> WorkerSelection {
    WorkerSelection::Single {
        worker: worker(url, RuntimeType::TokenSpeed, labels),
    }
}

/// The dtype an assembly writes a media item in is read from values cached
/// once (the router's overrides, the worker's label) and kept as a borrowed
/// name: resolving it allocates nothing on the default path for the vLLM,
/// TokenSpeed and SGLang shapes alike, and a worker's label costs the one
/// copy an assembly keeps (none for SGLang, which parses it in place).
#[test]
fn resolving_a_media_items_dtype_allocates_nothing_on_the_default_path() {
    let plain = selection("grpc://plain:8000", &[]);
    let labelled = selection(
        "grpc://labelled:8000",
        &[("multimodal_encoder_dtype", LABEL)],
    );
    // Warm: the overrides' cells read the environment once; that read is not the pin's.
    let _ = mm_encoder_input_dtype(Modality::Image, Some(&plain));
    let _ = mm_vllm_encoder_input_dtype(Some(&plain));
    let _ = mm_sglang_encoder_input_dtype(Some(&plain));

    let tokenspeed_default = bytes_allocated_during(|| {
        assert_eq!(
            mm_encoder_input_dtype(Modality::Image, Some(&plain)),
            "bfloat16"
        );
    });
    let vllm_default = bytes_allocated_during(|| {
        assert_eq!(mm_vllm_encoder_input_dtype(Some(&plain)), "float32");
    });
    let sglang_default = bytes_allocated_during(|| {
        assert_eq!(
            mm_sglang_encoder_input_dtype(Some(&plain)),
            EncoderDtype::Float32
        );
    });
    let sglang_label = bytes_allocated_during(|| {
        assert_eq!(
            mm_sglang_encoder_input_dtype(Some(&labelled)),
            EncoderDtype::BFloat16
        );
    });
    let tokenspeed_label = bytes_allocated_during(|| {
        assert_eq!(
            mm_encoder_input_dtype(Modality::Image, Some(&labelled)),
            LABEL
        );
    });

    // One assertion over every count, so a failure names them all.
    let read = [
        ("TokenSpeed, the default", tokenspeed_default),
        ("vLLM, the default", vllm_default),
        ("SGLang, the default", sglang_default),
        ("SGLang, a worker's label", sglang_label),
        ("TokenSpeed, a worker's label", tokenspeed_label),
    ];
    let pinned = [
        ("TokenSpeed, the default", 0),
        ("vLLM, the default", 0),
        ("SGLang, the default", 0),
        ("SGLang, a worker's label", 0),
        ("TokenSpeed, a worker's label", LABEL.len() as u64),
    ];
    assert_eq!(
        read, pinned,
        "bytes allocated per resolution (read, pinned)"
    );
}

/// The one-pass wire read over a mode's pools walks the registry's snapshots
/// in place: once the snapshots are built, reading the dtype a fleet of four
/// labelled workers agrees on allocates nothing.
#[test]
fn the_pools_one_pass_dtype_read_allocates_nothing() {
    let registry = WorkerRegistry::new();
    for url in [
        "grpc://a:8000",
        "grpc://b:8000",
        "grpc://c:8000",
        "grpc://d:8000",
    ] {
        registry
            .register(worker(
                url,
                RuntimeType::TokenSpeed,
                &[("multimodal_encoder_dtype", LABEL)],
            ))
            .expect("register");
    }
    let pools = Mode::Regular.multimodal_wire_pools();
    let plain = DtypeOverrides::default();
    // Warm: the first read after a membership change builds the snapshots; that build is not the pin's.
    assert_eq!(
        shared_image_wire_dtype_over(&registry, "qwen3-vl", pools, &plain),
        Some(EncoderDtype::BFloat16)
    );
    let bytes = bytes_allocated_during(|| {
        assert_eq!(
            shared_image_wire_dtype_over(&registry, "qwen3-vl", pools, &plain),
            Some(EncoderDtype::BFloat16)
        );
    });
    assert_eq!(bytes, 0, "the pools are walked in place");
}
