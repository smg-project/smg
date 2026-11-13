use std::time::Instant;

use engine_servicer::managed::{
    decode::{DecodeGate, State},
    ledger::Ledger,
};
use smg_grpc_client::{
    sglang_managed::{budget::Deadline, proto::*, Limits, LAYOUT},
    sglang_proto as sg,
};

fn operation() -> OperationRef {
    OperationRef {
        identity: Some(Identity {
            logical_request_id: "r".into(),
            attempt_id: "a".into(),
            branch_id: 0,
        }),
        target: Some(Instance {
            worker_id: "d".into(),
            generation: "g1".into(),
        }),
        operation_id: "op".into(),
        request_digest: vec![1; 32],
        plan_digest: vec![2; 32],
    }
}

fn budget() -> Budget {
    Budget {
        original_expiry_unix_ms: 2000,
        remaining_timeout_ms: 1000,
        estimated_work: 10,
        context_bytes: 100,
    }
}

#[expect(clippy::expect_used, reason = "construct known valid test inputs")]
fn gate(deferred: bool) -> DecodeGate {
    let op = operation();
    let instance = op.target.clone().expect("valid test fixture");
    let now = Instant::now();
    let deadline = Deadline::admit(&budget(), 1000, now, 0, 20, 3000).expect("valid test fixture");
    let start = DecodeStart {
        version: 1,
        binding: Some(InputBinding {
            decode: Some(op),
            prefill: Some(Instance {
                worker_id: "p".into(),
                generation: "pg1".into(),
            }),
            prefill_rid: "a-p".into(),
            bootstrap: Some(Bootstrap {
                host: "p".into(),
                port: 8998,
                room: 1,
            }),
            layout_contract: LAYOUT.into(),
            processor_contract: "processor".into(),
            model_revision: "revision".into(),
            media_routing_labels: vec![],
        }),
        budget: Some(budget()),
        generation: Some(sg::GenerateRequest {
            request_id: "a-d".into(),
            tokenized: Some(sg::TokenizedInput {
                input_ids: if deferred { vec![] } else { vec![1, 2] },
                ..Default::default()
            }),
            ..Default::default()
        }),
        deferred_from_prefill: deferred,
    };
    DecodeGate::register(start, &instance, deadline, Limits::default(), now)
        .expect("valid test fixture")
}

#[test]
fn waiting_input_never_submits_and_cancel_never_creates_an_abort_only_request() {
    let mut gate = gate(true);
    assert!(gate
        .take_submission(Instant::now())
        .expect("valid test fixture")
        .is_none());
    assert!(!gate.cancel());
    assert_eq!(gate.state(), State::Terminal);
    assert!(gate
        .take_submission(Instant::now())
        .expect("valid test fixture")
        .is_none());
}

#[test]
fn cancellation_during_send_requires_generate_then_abort_once() {
    let mut gate = gate(false);
    assert!(gate
        .take_submission(Instant::now())
        .expect("valid test fixture")
        .is_some());
    assert!(!gate.cancel());
    assert_eq!(gate.state(), State::NativeSubmitting);
    assert!(gate.submitted().expect("valid test fixture"));
    assert_eq!(gate.state(), State::Terminal);
    assert!(!gate.cancel());
    assert!(gate
        .take_submission(Instant::now())
        .expect("valid test fixture")
        .is_none());
}

#[test]
fn complete_half_close_keeps_generation_alive_but_incomplete_half_close_fails() {
    let mut text = gate(false);
    text.half_close().expect("valid test fixture");
    assert!(text
        .take_submission(Instant::now())
        .expect("valid test fixture")
        .is_some());
    assert!(!text.submitted().expect("valid test fixture"));
    assert!(text.cancel());
    let mut deferred = gate(true);
    assert!(deferred.half_close().is_err());
    assert_eq!(deferred.state(), State::Terminal);
}

#[test]
fn cancel_before_start_survives_until_expiry_and_does_not_allow_late_resurrection() {
    let op = operation();
    let mut ledger = Ledger::new(
        op.target.clone().expect("valid test fixture"),
        8,
        2,
        100,
        1000,
        100,
        3000,
    )
    .expect("valid test fixture");
    ledger
        .cancel(op.clone(), 2000, 1000)
        .expect("valid test fixture");
    assert!(ledger.admit(op.clone(), &budget(), 1001).is_err());
    ledger.collect(2200);
    assert!(ledger.admit(op, &budget(), 2200).is_err());
}

#[test]
fn cancellation_remains_charged_until_safe_completion_and_snapshots_cover_admissions() {
    let op = operation();
    let mut ledger = Ledger::new(
        op.target.clone().expect("valid test fixture"),
        8,
        2,
        100,
        1000,
        100,
        3000,
    )
    .expect("valid test fixture");
    assert_eq!(
        ledger
            .admit(op.clone(), &budget(), 1000)
            .expect("valid test fixture"),
        (1, true)
    );
    assert_eq!(
        ledger
            .admit(op.clone(), &budget(), 1001)
            .expect("valid test fixture"),
        (1, false)
    );
    ledger
        .cancel(op.clone(), 2000, 1002)
        .expect("valid test fixture");
    let snapshot = ledger.snapshot().expect("valid test fixture");
    assert_eq!(
        (
            snapshot.covered_admission_seq,
            snapshot.active,
            snapshot.draining_work
        ),
        (1, 1, 10)
    );
    ledger.collect(9999);
    assert_eq!(ledger.snapshot().expect("valid test fixture").active, 1);
    ledger.quiesce(&op).expect("valid test fixture");
    assert_eq!(ledger.snapshot().expect("valid test fixture").active, 0);
}

#[test]
fn drain_retains_existing_operations_and_rejects_new_ones() {
    let op = operation();
    let mut ledger = Ledger::new(
        op.target.clone().expect("valid test fixture"),
        8,
        2,
        100,
        1000,
        100,
        3000,
    )
    .expect("valid test fixture");
    ledger
        .admit(op.clone(), &budget(), 1000)
        .expect("valid test fixture");
    ledger.drain();
    assert_eq!(
        ledger
            .admit(op.clone(), &budget(), 1001)
            .expect("valid test fixture"),
        (1, false)
    );
    let mut other = op.clone();
    other.operation_id = "other".into();
    assert!(ledger.admit(other, &budget(), 1001).is_err());
    ledger.cancel(op, 2000, 1001).expect("valid test fixture");
}
