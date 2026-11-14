//! Managed-only policy and coverage regressions; these do not stand in for native GPU tests.
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use smg_grpc_client::sglang_managed::proto;

use super::{planner::*, registry::LoadTracker};

/// Distinct endpoints share a fixed compatible deployment contract.
fn worker(name: &str) -> Worker {
    Worker {
        instance: proto::Instance {
            worker_id: name.into(),
            generation: "g".into(),
        },
        control_endpoint: format!("http://{name}:1"),
        receive_endpoint: format!("http://{name}:2"),
        contract: Contract {
            model_revision: "m".into(),
            processor: "p".into(),
            layout: "l".into(),
            embedding_dtype: "bf16".into(),
            kv: "k".into(),
            transfer: "z".into(),
            topology: "t".into(),
        },
    }
}

/// Lower GPU work alone must not win when its embedding reaches P later.
#[test]
fn media_arrival_accounts_for_transfer_and_overlap() {
    let fast_gpu_slow_link = EncodeCost {
        cpu_wait_ms: 10,
        preprocess_ms: 20,
        gpu_available_ms: 40,
        encode_ms: 5,
        receive_ready_ms: 5,
        send_available_ms: 10,
        transfer_startup_ms: 2,
        embedding_bytes: 10000,
        bytes_per_ms: 10,
        ..Default::default()
    };
    assert_eq!(fast_gpu_slow_link.media_ready_ms(), Some(1047));
    let slow_gpu_fast_link = EncodeCost {
        encode_ms: 30,
        bytes_per_ms: 1000,
        ..fast_gpu_slow_link
    };
    let pair = PdCandidate {
        prefill: worker("p"),
        decode: worker("d"),
        encoders: vec![
            EncoderCandidate {
                worker: worker("e1"),
                cost: fast_gpu_slow_link,
                fresh: true,
                has_capacity: true,
                stable_tie_break: 1,
            },
            EncoderCandidate {
                worker: worker("e2"),
                cost: slow_gpu_fast_link,
                fresh: true,
                has_capacity: true,
                stable_tie_break: 2,
            },
        ],
        has_capacity: true,
        layout_ms: 1,
        manifest_delivery_ms: 1,
        decode_registration_ms: 5,
        bind_submit_ms: 1,
        native_pd_ms: 10,
    };
    let planner = Planner::default();
    assert_eq!(
        planner
            .choose(std::slice::from_ref(&pair), 2, Duration::from_secs(2))
            .unwrap()
            .1,
        Some(1)
    );
    assert!(planner
        .choose(std::slice::from_ref(&pair), 2, Duration::from_millis(50))
        .is_err());
    assert_eq!(
        planner
            .choose(&[pair], 0, Duration::from_secs(2))
            .unwrap()
            .1,
        None
    );
}

/// A covering snapshot may precede Accepted; unknown work remains charged meanwhile.
#[test]
fn snapshot_before_ack_deduplicates_without_releasing_unknown_work() {
    let now = Instant::now();
    let instance = worker("e").instance;
    let mut tracker = LoadTracker::new(instance.clone(), 4);
    tracker.reserve("a".into(), 30).unwrap();
    tracker
        .observe(
            proto::LoadSnapshot {
                instance: Some(instance.clone()),
                snapshot_seq: 1,
                covered_admission_seq: 7,
                cpu_work: 30,
                capacity: 2,
                active: 1,
                ..Default::default()
            },
            now,
        )
        .unwrap();
    assert_eq!(
        tracker.stage_work(now, Duration::from_secs(1)),
        Some((60, 30, 0))
    );
    assert!(!tracker.has_capacity(now, Duration::from_secs(1)));
    tracker.accepted("a", &instance, 7).unwrap();
    assert_eq!(
        tracker.stage_work(now, Duration::from_secs(1)),
        Some((30, 0, 0))
    );
    assert!(tracker.has_capacity(now, Duration::from_secs(1)));
    assert_eq!(
        tracker.stage_work(now + Duration::from_secs(2), Duration::from_secs(1)),
        None
    );
    let mut restart = instance;
    restart.generation = "restarted".into();
    assert!(tracker
        .observe(
            proto::LoadSnapshot {
                instance: Some(restart),
                snapshot_seq: 2,
                ..Default::default()
            },
            now
        )
        .is_err());
}

/// Independent TP retains model and native transfer compatibility checks.
#[test]
fn independent_encoder_tp_preserves_pd_and_model_fences() {
    let mut p = worker("p");
    p.contract.topology = "single-host-tp2".into();
    p.contract.kv = "mooncake".into();
    let mut d = worker("d");
    d.contract = p.contract.clone();
    let mut e = worker("e");
    e.contract = p.contract.clone();
    e.contract.topology = "single-host-tp1".into();
    let mut pair = PdCandidate {
        prefill: p,
        decode: d,
        encoders: vec![EncoderCandidate {
            worker: e,
            cost: EncodeCost {
                bytes_per_ms: 1,
                ..Default::default()
            },
            fresh: true,
            has_capacity: true,
            stable_tie_break: 0,
        }],
        has_capacity: true,
        layout_ms: 1,
        manifest_delivery_ms: 1,
        decode_registration_ms: 1,
        bind_submit_ms: 1,
        native_pd_ms: 1,
    };
    let choose = |pair: &PdCandidate| {
        Planner::default().choose(std::slice::from_ref(pair), 1, Duration::from_secs(1))
    };
    assert!(choose(&pair).is_ok());
    pair.encoders[0].worker.contract.model_revision = "other-weights".into();
    assert!(choose(&pair).is_err());
    pair.encoders[0].worker.contract.model_revision = pair.prefill.contract.model_revision.clone();
    pair.encoders[0].worker.contract.topology = "single-host-tp8".into();
    assert!(choose(&pair).is_ok());
    pair.encoders[0].worker.contract.topology = "unknown".into();
    assert!(choose(&pair).is_err());
    pair.encoders[0].worker.contract.topology = "single-host-tp1".into();
    pair.decode.contract.topology = "single-host-tp1".into();
    assert!(choose(&pair).is_ok());
    // D never receives embeddings, so its unused native embedding backend is irrelevant.
    pair.decode.contract.transfer = "unused".into();
    assert!(choose(&pair).is_ok());
    pair.encoders[0].worker.contract.transfer = "different".into();
    assert!(choose(&pair).is_err());
    pair.encoders[0].worker.contract.transfer = pair.prefill.contract.transfer.clone();
    pair.prefill.contract.kv = "nixl".into();
    pair.decode.contract.kv = "nixl".into();
    assert!(choose(&pair).is_err());
}

/// Only the explicitly invoked real-worker server enables the hash ablation.
/// This switch and its environment inputs do not exist in production builds.
pub(super) static BENCHMARK_HASH: AtomicBool = AtomicBool::new(false);

/// Serve the normal HTTP stack against real workers for reproducible selection ablations.
/// Run this test alone in a release test binary, with SMG_MANAGED_BENCH_CONFIG,
/// SMG_MANAGED_BENCH_PORT and SMG_MANAGED_BENCH_POLICY=arrival|hash explicitly set.
/// Both modes retain the same admission checks, immutable plans and executor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "explicit benchmark server requiring real E/P/D workers and pinned weights"]
async fn real_worker_benchmark_server() {
    let policy = std::env::var("SMG_MANAGED_BENCH_POLICY").expect("explicit benchmark policy");
    BENCHMARK_HASH.store(
        match policy.as_str() {
            "arrival" => false,
            "hash" => true,
            _ => panic!("benchmark policy must be arrival or hash"),
        },
        Ordering::Relaxed,
    );
    // JSON is only a test-harness input, not a production deployment interface.
    let deployment: super::router::Config = serde_json::from_slice(
        &std::fs::read(std::env::var("SMG_MANAGED_BENCH_CONFIG").expect("benchmark fixture"))
            .expect("read benchmark fixture"),
    )
    .expect("valid benchmark fixture");
    let router = crate::config::RouterConfig {
        sglang_managed_epd: true,
        connection_mode: crate::worker::ConnectionMode::Grpc,
        model_path: Some(deployment.model_path.clone()),
        model_aliases: std::collections::HashMap::from([(deployment.model, deployment.model_path)]),
        mode: crate::config::RoutingMode::EncodePrefillDecode {
            encode_urls: deployment
                .encoders
                .into_iter()
                .map(|e| (e.endpoint, None))
                .collect(),
            prefill_urls: deployment
                .prefill_urls
                .into_iter()
                .map(|p| (p, None))
                .collect(),
            decode_urls: deployment.decode_urls,
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        },
        host: "0.0.0.0".into(),
        port: std::env::var("SMG_MANAGED_BENCH_PORT")
            .expect("explicit benchmark port")
            .parse()
            .expect("valid benchmark port"),
        runtime_worker_threads: Some(4),
        ..Default::default()
    };
    crate::server::startup(crate::server::ServerConfig::minimal(router))
        .await
        .expect("real-worker benchmark server");
}
