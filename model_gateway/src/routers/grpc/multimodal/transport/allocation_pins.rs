//! What the encoder dtype resolution allocates per media item, pinned at
//! zero on the default and override paths so the next change that copies a
//! cached name fails a test rather than moving a bench inside its noise. The
//! lib's test binary runs under the executable's allocator (`lib.rs`), and
//! each pin reads that allocator's counter for its own thread
//! (`routers::grpc::testing::bytes_allocated_during`), so the shared binary
//! is read one thread at a time.

use std::sync::Arc;

use llm_multimodal::{EncoderDtype, Modality};
use openai_protocol::{model_card::ModelCard, worker::RuntimeType};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker, WorkerType},
    test_support::no_health_check,
};

use super::{
    mm_encoder_input_dtype, mm_sglang_encoder_input_dtype, mm_vllm_encoder_input_dtype,
    shared_image_wire_dtype_over, DtypeOverrides,
};
use crate::routers::{
    common::snapshot::WorkerRegistry,
    grpc::{
        backend_client::BasicWorkerBuilder, context::WorkerSelection, mode::Mode,
        testing::bytes_allocated_during,
    },
};

/// The label the labelled workers carry. An assembly keeps one copy of it; what
/// that copy costs the allocator is read in the pin, not taken from its length.
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
        worker: worker(url, RuntimeType::TokenSpeed, labels).into(),
    }
}

/// The dtype an assembly writes a media item in is read from values cached
/// once (the router's overrides, the worker's label) and kept as a borrowed
/// name: resolving it allocates nothing on the default path for the vLLM,
/// TokenSpeed and SGLang shapes alike, and a worker's label costs the one
/// copy an assembly keeps (none for SGLang, which parses it in place).
///
/// The expectations follow the process's own overrides, read from the cells
/// the resolvers read: under an inherited override the answer is the override,
/// a borrowed name, and a worker's label is not copied.
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
    // The process's own overrides, from the cells the resolvers read: an
    // inherited one is the answer, and the pins below follow it.
    let overrides = DtypeOverrides::from_env();
    let tokenspeed_override = overrides.image.or(overrides.default);
    // The allocator's counter adds a copy's size class, not its length: read
    // what one copy of the label costs. Under a TokenSpeed override the name
    // is the process's own and the label is not copied. The copy is at least
    // the label's length: a counter that stopped moving fails here, and not
    // as `0 == 0` on every row below.
    let one_copy = bytes_allocated_during(|| drop(std::hint::black_box(LABEL.to_owned())));
    assert!(
        one_copy >= LABEL.len() as u64,
        "the counter reads this thread's allocations: one copy of {LABEL:?} read {one_copy} bytes"
    );
    let label_copy = if tokenspeed_override.is_some() {
        0
    } else {
        one_copy
    };

    let tokenspeed_default = bytes_allocated_during(|| {
        assert_eq!(
            mm_encoder_input_dtype(Modality::Image, Some(&plain)),
            tokenspeed_override.unwrap_or("bfloat16")
        );
    });
    let vllm_default = bytes_allocated_during(|| {
        assert_eq!(
            mm_vllm_encoder_input_dtype(Some(&plain)),
            overrides.vllm.unwrap_or("float32")
        );
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
            tokenspeed_override.unwrap_or(LABEL)
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
        ("TokenSpeed, a worker's label", label_copy),
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
