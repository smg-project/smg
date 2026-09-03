//! Microbenchmark for the `GET /workers` response path.
//!
//! Measures the full per-request cost (build + serialize) across three points,
//! all using the real `WorkerInfo` type (no parallel/duplicate response shape):
//!
//!   1. `original_value_plus_deepclone` — deep-clone each `WorkerSpec` into a
//!      `WorkerInfo`, then build an intermediate `serde_json::Value` (`json!`)
//!      and serialize that. A per-worker hash map plus a double pass.
//!   2. `serde_direct_plus_deepclone` — still deep-clones the spec, but
//!      serializes a typed struct directly (single `serde_json::to_vec`).
//!   3. `serde_direct_plus_arc` — `WorkerSpec` is shared via `Arc`, so building
//!      each `WorkerInfo` (`worker_to_info`) is a refcount bump, then serialize
//!      directly. This is the shipped path.
//!
//! Worker data is entirely synthetic/generic (no proprietary specs).
//! Run: `cargo bench -p smg --bench workers_endpoint`

use std::{collections::HashMap, hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use openai_protocol::{
    model_card::ModelCard,
    model_type::ModelType,
    worker::{WorkerInfo, WorkerStatus},
};
use serde::Serialize;
use smg::worker::{worker::worker_to_info, BasicWorkerBuilder, Worker, WorkerType};

/// ~16 generic labels approximating a realistic worker's label payload size.
fn make_labels() -> HashMap<String, String> {
    let pairs = [
        ("tensor_parallel", "4"),
        ("pipeline_parallel", "1"),
        ("data_parallel", "1"),
        ("max_seq_len", "262144"),
        ("max_running_requests", "512"),
        ("is_generation", "true"),
        ("is_embedding", "false"),
        ("has_image_understanding", "true"),
        ("has_audio_understanding", "false"),
        ("load_balance_method", "round_robin"),
        ("weight_version", "default"),
        ("runtime_version", "0.0.0"),
        ("arch_family", "example_arch"),
        ("served_model_name", "example-org/example-model-v1"),
        (
            "model_path",
            "storage://example-bucket/models/example-model-v1",
        ),
        ("deploy_zone", "zone-a"),
    ];
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn make_model() -> ModelCard {
    let mut card = ModelCard::new("example-org/example-model-v1")
        .with_model_type(ModelType::VISION_LLM)
        .with_context_length(262_144)
        .with_tokenizer_path("example-org/example-model-tokenizer");
    card.hf_model_type = Some("example_arch".to_string());
    card.architectures = vec!["ExampleForCausalLM".to_string()];
    card
}

fn make_worker(i: usize) -> Arc<dyn Worker> {
    let worker_type = match i % 3 {
        0 => WorkerType::Prefill,
        1 => WorkerType::Decode,
        _ => WorkerType::Regular,
    };
    let worker =
        BasicWorkerBuilder::new(format!("http://worker-{i}.example.svc.cluster.local:30000"))
            .worker_type(worker_type)
            .status(WorkerStatus::Ready)
            .labels(make_labels())
            .model(make_model())
            .build();
    Arc::new(worker)
}

fn build_workers(n: usize) -> Vec<Arc<dyn Worker>> {
    (0..n).map(make_worker).collect()
}

/// Replicates the OLD per-worker cost: deep-clone the whole `WorkerSpec`.
fn deep_clone_info(w: &Arc<dyn Worker>) -> WorkerInfo {
    let meta = w.metadata();
    let status = w.status();
    WorkerInfo {
        id: w.url().to_string(),
        model_id: meta.spec.models.primary().map(|m| m.id.clone()),
        spec: Arc::new((*meta.spec).clone()), // deep clone, then wrap
        is_healthy: status == WorkerStatus::Ready,
        status: Some(status),
        load: w.load(),
        http2: meta.http2,
        job_status: None,
    }
}

#[derive(Serialize)]
struct Stats {
    prefill_count: usize,
    decode_count: usize,
    regular_count: usize,
}
const STATS: Stats = Stats {
    prefill_count: 0,
    decode_count: 0,
    regular_count: 0,
};

#[derive(Serialize)]
struct Body<'a> {
    workers: &'a [WorkerInfo],
    total: usize,
    stats: Stats,
}

fn serialize_value(infos: &[WorkerInfo]) -> Vec<u8> {
    let value = serde_json::json!({
        "workers": infos,
        "total": infos.len(),
        "stats": { "prefill_count": 0, "decode_count": 0, "regular_count": 0 }
    });
    serde_json::to_vec(&value).unwrap_or_default()
}

fn serialize_direct(infos: &[WorkerInfo]) -> Vec<u8> {
    serde_json::to_vec(&Body {
        workers: infos,
        total: infos.len(),
        stats: STATS,
    })
    .unwrap_or_default()
}

// Stage 1: deep clone + serde_json::Value.
fn stage_original(workers: &[Arc<dyn Worker>]) -> Vec<u8> {
    let infos: Vec<WorkerInfo> = workers.iter().map(deep_clone_info).collect();
    serialize_value(&infos)
}

// Stage 2: deep clone + direct serialize.
fn stage_direct_deepclone(workers: &[Arc<dyn Worker>]) -> Vec<u8> {
    let infos: Vec<WorkerInfo> = workers.iter().map(deep_clone_info).collect();
    serialize_direct(&infos)
}

// Stage 3: Arc-shared spec (worker_to_info) + direct serialize. (Shipped.)
fn stage_arc_direct(workers: &[Arc<dyn Worker>]) -> Vec<u8> {
    let infos: Vec<WorkerInfo> = workers.iter().map(worker_to_info).collect();
    serialize_direct(&infos)
}

fn bench_workers(c: &mut Criterion) {
    const N: usize = 2000;
    let workers = build_workers(N);

    // Sanity: all three implementations produce equivalent JSON.
    let s1 = stage_original(&workers);
    let s2 = stage_direct_deepclone(&workers);
    let s3 = stage_arc_direct(&workers);
    let v1: serde_json::Value = serde_json::from_slice(&s1).unwrap_or_default();
    let v2: serde_json::Value = serde_json::from_slice(&s2).unwrap_or_default();
    let v3: serde_json::Value = serde_json::from_slice(&s3).unwrap_or_default();
    assert_eq!(v1, v2, "direct must match original");
    assert_eq!(v1, v3, "arc must match original");

    let mut group = c.benchmark_group("workers_endpoint");
    group.throughput(Throughput::Bytes(s3.len() as u64));
    group.bench_function("1_original_value_plus_deepclone", |b| {
        b.iter(|| black_box(stage_original(black_box(&workers))));
    });
    group.bench_function("2_serde_direct_plus_deepclone", |b| {
        b.iter(|| black_box(stage_direct_deepclone(black_box(&workers))));
    });
    group.bench_function("3_serde_direct_plus_arc", |b| {
        b.iter(|| black_box(stage_arc_direct(black_box(&workers))));
    });
    group.finish();
}

criterion_group!(benches, bench_workers);
criterion_main!(benches);
