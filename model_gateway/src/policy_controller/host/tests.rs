//! Cases of the host over the module's test support: rows written by hand
//! through the store's own door, beside workers registered by hand.

use std::{
    collections::BTreeSet,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use axum::{
    body::to_bytes,
    http::{header::RETRY_AFTER, StatusCode},
};
use kv_index::compute_request_content_hashes;
use openai_protocol::worker::WorkerStatus;
use serde_json::Value;
use smg_policy_controller::{
    decision::ShedReason,
    door::Refused,
    lease::{ReleaseReason, TransportClass},
    store::{
        BatchOutcome, KvEventSink, LeasesStore, LoadSink, LoadSource, MemoryStore, WorkerFacts,
        WorkerStatus as RowStatus, WorkersStore, PUBLISH_WINDOW,
    },
    view::LoadRow,
};
use smg_worker::identity::basic::WorkerLoadGuard;

use super::{
    left_no_worker,
    testing::{bed, cache_bed, store_blocks, MODEL},
    Ask, End, Hashes, Held, Prompt, Unplaced, GATEWAY,
};

/// A first attempt's ask of the request `id`.
pub(super) fn first(id: &[u8]) -> Ask<'_> {
    Ask {
        id,
        model: MODEL,
        streaming: false,
        tried: &[],
        hashes: None,
    }
}

/// A plain end with status 200.
fn ok() -> End {
    End {
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// The gateway's live leases, read until they number `expected` or the
/// patience ends: a report is applied by whoever holds the lease's file.
fn live(store: &MemoryStore, expected: usize) -> usize {
    let patience = Instant::now() + Duration::from_secs(5);
    loop {
        let live = LeasesStore::live(store, GATEWAY);
        if live == expected || Instant::now() > patience {
            return live;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[tokio::test]
async fn a_pick_lands_on_the_gateways_worker_of_its_row() {
    let (host, store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let placed = host
        .place(&first(b"r1"), Instant::now())
        .await
        .expect("a worker");
    let row = WorkersStore::get(&*store, placed.id()).expect("the target's row");
    assert_eq!(placed.worker().url(), row.facts.key);
}

#[tokio::test]
async fn a_drawn_pick_is_not_the_same_worker_for_every_request() {
    let (host, _store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let mut picked = BTreeSet::new();
    for turn in 0..64u8 {
        let placed = host
            .place(&first(&[turn]), Instant::now())
            .await
            .expect("a worker");
        picked.insert(placed.worker().url().to_owned());
    }
    // Each ask draws its own seed. Under one seed the default profile's
    // draw lands every request on the same worker of an unchanged pool.
    assert_eq!(picked.len(), 2, "64 requests landed on {picked:?}");
}

#[tokio::test]
async fn a_returning_prefix_lands_on_the_worker_that_holds_it() {
    let tokens: Vec<u32> = (1..=64).collect();
    for block_size in [32, 16] {
        let (host, store, holder) = cache_bed();
        store.set_block_size(MODEL, block_size);
        let blocks = compute_request_content_hashes(&tokens, block_size as usize);
        assert_eq!(store_blocks(&store, holder, &blocks), BatchOutcome::Applied);
        let hashes = Hashes::of(
            &store,
            MODEL,
            &Prompt {
                tokens: Some(&tokens),
                namespace: None,
            },
        );
        let now = Instant::now();
        // Give plain asks a lower-load choice while affinity can still win.
        for row in store.snapshot().1 {
            store.record(
                row.id,
                0,
                LoadRow {
                    running: u32::from(row.id == holder),
                    ..LoadRow::default()
                },
                now,
                LoadSource::Stream,
            );
        }
        let now = now + PUBLISH_WINDOW;
        store.publish(now);
        let mut plain_matches = 0;
        for turn in 0..32u8 {
            let id = [turn];
            let plain = first(&id);
            let returning = Ask {
                hashes: Some(&hashes),
                ..plain
            };
            assert_eq!(
                host.place(&returning, now).await.expect("a worker").id(),
                holder,
                "block size {block_size}, request {turn}"
            );
            let placed = host.place(&plain, now).await.expect("a worker");
            plain_matches += usize::from(placed.id() == holder);
        }
        assert!(plain_matches < 32, "block size {block_size}");
        let returning = Ask {
            hashes: Some(&hashes),
            ..first(b"returning")
        };
        assert_eq!(returning.request(&[]).facts.block_size, block_size);
    }
}

#[tokio::test]
async fn a_worker_the_gateway_cannot_send_to_is_passed_over() {
    let (host, _store, workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    for turn in 0..8u8 {
        let placed = host
            .place(&first(&[turn]), Instant::now())
            .await
            .expect("the other worker");
        assert_eq!(placed.worker().url(), "http://w1:8000");
    }
}

#[tokio::test]
async fn a_target_the_routes_rule_refuses_is_passed_over() {
    let (host, _store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    for turn in 0..8u8 {
        let placed = host
            .place_where(
                &first(&[turn]),
                |worker| worker.url() != "http://w0:8000",
                Instant::now(),
            )
            .await
            .expect("the worker the rule holds");
        assert_eq!(placed.worker().url(), "http://w1:8000");
    }
}

#[tokio::test]
async fn a_rule_that_refuses_every_target_ends_the_attempt_and_its_leases() {
    let (host, store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let unplaced = host
        .place_where(&first(b"r1"), |_| false, Instant::now())
        .await
        .err();
    assert_eq!(unplaced, Some(Unplaced::NoTargetHolds));
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_store_with_no_row_sheds_by_the_controllers_reason() {
    let (host, _store, _workers) = bed(&[]);
    let unplaced = host
        .place(&first(b"r1"), Instant::now())
        .await
        .expect_err("no row, no worker");
    assert!(
        matches!(
            unplaced,
            Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            }
        ),
        "{unplaced:?}"
    );
}

#[tokio::test]
async fn an_attempt_the_gateway_can_send_nowhere_ends_after_its_asks() {
    let (host, _store, workers) = bed(&["http://w0:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    let unplaced = host.place(&first(b"r1"), Instant::now()).await.err();
    assert_eq!(unplaced, Some(Unplaced::NoTargetHolds));
}

#[tokio::test]
async fn the_commit_gives_the_lease_up_and_names_its_misses() {
    let (host, store, workers) = bed(&["http://w0:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    let now = Instant::now();
    let asked = host.ask(&first(b"r1"), &[], now).await.expect("a decision");
    assert_eq!(live(&store, 1), 1);
    let mut missed = Vec::new();
    assert!(host.commit(asked, &mut missed, now).is_none());
    assert_eq!(missed.len(), 1, "the one target is a miss: {missed:?}");
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_lease_ends_once_by_its_completion_and_by_a_drop() {
    let (host, store, _workers) = bed(&["http://w0:8000"]);
    let now = Instant::now();
    let mut placed = host.place(&first(b"r1"), now).await.expect("a worker");
    assert_eq!(live(&store, 1), 1);
    placed.dispatched(now);
    placed.completed(ok(), now);
    assert_eq!(live(&store, 0), 0);
    // Said once: nothing more is owed, and a release says nothing.
    placed.released(ReleaseReason::ClientGone, now);
    drop(placed);
    // A placement dropped before any send gives its lease up.
    drop(host.place(&first(b"r2"), now).await.expect("a worker"));
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_held_request_dropped_after_its_send_ends_its_lease_and_its_load() {
    let (host, store, workers) = bed(&["http://w0:8000"]);
    let now = Instant::now();
    let placed = host.place(&first(b"r1"), now).await.expect("a worker");
    let mut held = Held::new(placed, |worker| {
        WorkerLoadGuard::with_key(Arc::clone(worker), None)
    });
    held.placed().dispatched(now);
    assert_eq!((live(&store, 1), workers[0].load()), (1, 1));
    // A response dropped with no end said: the lease completes as cancelled
    // and the worker's load is given back.
    drop(held);
    assert_eq!((live(&store, 0), workers[0].load()), (0, 0));
}

#[tokio::test]
async fn each_reason_an_attempt_has_no_worker_reads_its_own_answer() {
    let shed = |reason| Unplaced::Shed {
        reason,
        emptied_by: "not-overloaded",
        retry_after_secs: 7,
    };
    let unavailable = StatusCode::SERVICE_UNAVAILABLE;
    let shed_code = "worker_overload_protection_shed";
    let rows = [
        (
            shed(ShedReason::Saturated),
            unavailable,
            shed_code,
            Some("7"),
        ),
        (Unplaced::Queued, unavailable, shed_code, Some("1")),
        (
            shed(ShedReason::NoEndpoints),
            unavailable,
            "no_available_workers",
            Some("1"),
        ),
        (
            Unplaced::NoTargetHolds,
            unavailable,
            "no_available_workers",
            Some("1"),
        ),
        (
            shed(ShedReason::FlowLimit),
            StatusCode::TOO_MANY_REQUESTS,
            "admission_queue_full",
            Some("2"),
        ),
        (
            shed(ShedReason::Deadline),
            unavailable,
            "admission_queue_timeout",
            Some("2"),
        ),
        (
            shed(ShedReason::PolicyDeclined),
            unavailable,
            "policy_declined",
            None,
        ),
        (
            Unplaced::Refused(Refused::Faulted),
            unavailable,
            "policy_controller_unavailable",
            Some("1"),
        ),
    ];
    for (unplaced, status, code, retry_after) in rows {
        let answer = unplaced.answer(MODEL);
        let hint = answer
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        assert_eq!(answer.status(), status, "{unplaced:?}");
        assert_eq!(hint.as_deref(), retry_after, "{unplaced:?}");
        let body = to_bytes(answer.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let error: Value = serde_json::from_slice(&body).expect("a JSON error object");
        assert_eq!(error["error"]["code"], code, "{unplaced:?}");
    }
}

#[tokio::test]
async fn a_placement_names_the_rank_its_decision_named() {
    let (host, store, _workers) = bed(&["http://w0:8000"]);
    // The bed's row is rank 0 of 1, which a constant would answer too: the
    // row is written again as rank 1 of 2.
    let unranked = host
        .place(&first(b"r0"), Instant::now())
        .await
        .expect("a worker");
    let row = WorkersStore::get(&*store, unranked.id()).expect("the bed's row");
    drop(unranked);
    let id = store
        .upsert(WorkerFacts {
            dp_rank: 1,
            dp_size: 2,
            ..row.facts
        })
        .id();
    store.publish_now();
    let row = WorkersStore::get(&*store, id).expect("the ranked row");
    if row.status != RowStatus::Ready {
        store
            .set_status(id, RowStatus::Ready, row.revision)
            .expect("the row stands");
        store.publish_now();
    }
    let placed = host
        .place(&first(b"r1"), Instant::now())
        .await
        .expect("a worker");
    assert_eq!((placed.id(), placed.rank()), (id, 1));
}

#[test]
fn a_pool_no_stage_emptied_reads_its_own_reason() {
    assert_eq!(
        left_no_worker("none"),
        "the policy controller holds no candidate for the model"
    );
    assert_eq!(
        left_no_worker("healthy"),
        "the policy controller's healthy stage left no worker"
    );
}
