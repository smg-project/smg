//! Managed-only policy and coverage regressions; these do not stand in for native GPU tests.
use std::time::{Duration, Instant};

use smg_grpc_client::sglang_managed::proto;

use super::{load_tracker::LoadTracker, planner::*};

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
