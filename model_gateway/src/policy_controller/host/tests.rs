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
use futures::FutureExt;
use kv_index::compute_request_content_hashes;
use openai_protocol::worker::WorkerStatus;
use serde_json::{json, Value};
use smg_policy_controller::{
    decision::{Decision, Shed, ShedReason, Trace},
    door::Refused,
    lease::{ReleaseReason, TransportClass},
    store::{
        BatchOutcome, KvEventSink, LeasesStore, LoadSink, LoadSource, MemoryStore, WorkerFacts,
        WorkerStatus as RowStatus, WorkersStore, PUBLISH_WINDOW,
    },
    view::LoadRow,
    AppliedSpec, PolicySpec,
};
use smg_worker::identity::basic::WorkerLoadGuard;

use super::{
    left_no_worker, profile_of,
    testing::{bed, cache_bed, store_blocks, MODEL},
    Ask, End, Hashes, Held, Prompt, Unplaced, GATEWAY,
};
use crate::observability::{cache_trace, metrics::test_support::render_with_recorder};

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

fn assert_decisions(rendered: &str, profile: &str, outcome: &str, count: u32) {
    for expected in [
        format!(
            "smg_policy_decisions_total{{model=\"{MODEL}\",profile=\"{profile}\",outcome=\"{outcome}\",mode=\"embedded\"}} {count}"
        ),
        format!(
            "smg_policy_decision_seconds_count{{mode=\"embedded\",outcome=\"{outcome}\"}} {count}"
        ),
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing {expected} in:\n{rendered}"
        );
    }
}

#[test]
fn a_placement_counts_and_times_its_documents_profile() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let document: PolicySpec = serde_yaml::from_str(
        "apiVersion: smg.policy/v1
generation: 2
profiles:
  batch: { pick: random }
  chat: { pick: random }
bindings:
  - match: { model: 'batch-*' }
    profile: batch
  - match: { model: '*' }
    profile: chat",
    )
    .unwrap();
    store
        .apply(AppliedSpec::validate(&document).unwrap())
        .unwrap();
    assert!(host
        .profile_name(&Trace {
            generation: 1,
            ..Trace::default()
        })
        .is_none());
    let before = Instant::now();
    let rendered = render_with_recorder(|| {
        let now = before - Duration::from_secs(60);
        assert!(host
            .place(&first(b"r1"), now)
            .now_or_never()
            .unwrap()
            .is_ok());
    });
    assert_decisions(&rendered, "chat", "pick", 1);
    let seconds: f64 = rendered
        .lines()
        .find_map(|line| {
            line.strip_prefix(
                "smg_policy_decision_seconds_sum{mode=\"embedded\",outcome=\"pick\"} ",
            )
        })
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        seconds > 0.0 && seconds <= before.elapsed().as_secs_f64(),
        "{seconds}"
    );
}

#[test]
fn a_shed_and_each_attempt_to_take_a_target_are_counted() {
    let (empty, _, _) = bed(&[]);
    let (unhealthy, _, workers) = bed(&["http://w0:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    let rendered = render_with_recorder(|| {
        assert!(matches!(
            empty
                .place(&first(b"r1"), Instant::now())
                .now_or_never()
                .unwrap(),
            Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            })
        ));
        assert_eq!(
            unhealthy
                .place(&first(b"r2"), Instant::now())
                .now_or_never()
                .unwrap()
                .err(),
            Some(Unplaced::NoTargetHolds)
        );
    });
    assert_decisions(&rendered, "random", "shed", 1);
    assert_decisions(&rendered, "random", "pick", 3);
}

#[test]
fn a_request_no_binding_covers_is_shed_under_no_profile() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let document: PolicySpec = serde_yaml::from_str(
        "apiVersion: smg.policy/v1
generation: 2
profiles:
  batch: { pick: random }
bindings:
  - match: { model: 'batch-*' }
    profile: batch",
    )
    .unwrap();
    store
        .apply(AppliedSpec::validate(&document).unwrap())
        .unwrap();
    let rendered = render_with_recorder(|| {
        cache_trace::scope(true, async {
            let unplaced = host.place(&first(b"r1"), Instant::now()).await.err();
            assert!(
                matches!(
                    unplaced,
                    Some(Unplaced::Shed {
                        reason: ShedReason::PolicyDeclined,
                        ..
                    })
                ),
                "{unplaced:?}"
            );
            let rows = cache_trace::captured_selections();
            assert_eq!(rows.len(), 1);
            // The one profile of the document never ran: the row names none.
            assert_eq!(rows[0]["policy"], json!(null));
            assert_eq!(rows[0]["decision"]["profile"], json!(null));
            assert_eq!(rows[0]["decision"]["emptied_by"], "binding");
        })
        .now_or_never()
        .unwrap();
    });
    assert_decisions(&rendered, "", "shed", 1);
}

#[test]
fn only_a_shed_ahead_of_every_profile_names_no_profile() {
    let shed = |reason, emptied_by| {
        Decision::Shed(Shed {
            reason,
            emptied_by,
            retry_after_secs: 1,
            wait_allowed: false,
            trace: Trace {
                profile: 3,
                ..Trace::default()
            },
        })
    };
    for (reason, stage) in [
        (ShedReason::PolicyDeclined, "binding"),
        (ShedReason::PolicyDeclined, "affinity-column"),
        (ShedReason::PolicyDeclined, "written-rows"),
        (ShedReason::NoEndpoints, "wires"),
        (ShedReason::NoEndpoints, "subset-hint"),
    ] {
        assert_eq!(profile_of(&shed(reason, stage)), None, "{stage}");
    }
    // A profile's own stage that declines, and a pool a filter emptied.
    let by_a_picker = shed(ShedReason::PolicyDeclined, "random");
    assert_eq!(profile_of(&by_a_picker), Some(3));
    let emptied = shed(ShedReason::NoEndpoints, "healthy");
    assert_eq!(profile_of(&emptied), Some(3));
}

#[test]
fn a_refused_ask_is_counted_without_a_profile() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    // A term above the lease id's term bits cannot mint a lease.
    LeasesStore::set_term(&*store, 1 << 31);
    let rendered = render_with_recorder(|| {
        let refused = host
            .ask(&first(b"r1"), &[], Instant::now())
            .now_or_never()
            .unwrap()
            .err();
        assert_eq!(refused, Some(Unplaced::Refused(Refused::Unleased)));
    });
    assert_decisions(&rendered, "", "refused", 1);
}

#[tokio::test]
async fn a_trace_records_the_worker_taken_each_miss_and_the_shed() {
    let (host, store, workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let ask = first(b"r1");
    let now = Instant::now();
    let asked = host.ask(&ask, &[], now).await.unwrap();
    let targets = asked.targets;
    let preferred = host.worker(targets.first().unwrap().id).unwrap();
    preferred.set_status(WorkerStatus::NotReady);
    drop(asked);
    cache_trace::scope(true, async {
        let placed = host.place(&ask, now).await.unwrap();
        assert_ne!(placed.worker().url(), preferred.url());
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["worker"], placed.worker().url());
        assert_eq!(row["policy"], "random");
        assert_eq!(row["decision"]["outcome"], "pick");
        assert_eq!(row["decision"]["generation"], 1);
        assert_eq!(row["decision"]["survivors"], 2);
        assert!(row["decision"]["top"]
            .as_array()
            .unwrap()
            .iter()
            .any(|candidate| candidate["worker"] == preferred.url()));
        assert_eq!(row["decision"]["lifted"], "none");
    })
    .await;
    for worker in workers {
        worker.set_status(WorkerStatus::NotReady);
    }
    cache_trace::scope(true, async {
        assert_eq!(
            host.place(&first(b"r2"), Instant::now()).await.err(),
            Some(Unplaced::NoTargetHolds)
        );
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row["decision"]["outcome"], "pick");
            assert_eq!(row["worker"], json!(null));
        }
    })
    .await;
    for target in targets.iter() {
        let row = WorkersStore::get(&*store, target.id).unwrap();
        store
            .set_status(target.id, RowStatus::NotReady, row.revision)
            .unwrap();
    }
    store.publish_now();
    cache_trace::scope(true, async {
        let result = host.place(&first(b"r3"), Instant::now()).await;
        assert!(matches!(
            result,
            Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            })
        ));
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["worker"], json!(null));
        assert_eq!(row["decision"]["outcome"], "shed");
        assert_eq!(row["decision"]["reason"], "no_endpoints");
        assert_eq!(row["decision"]["emptied_by"], "healthy");
    })
    .await;
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
