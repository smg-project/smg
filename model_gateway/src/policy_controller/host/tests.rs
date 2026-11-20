//! Cases of the host over the module's test support: rows written by hand
//! through the store's own door, beside workers registered by hand.

use std::{
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    io,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

use axum::{
    body::{to_bytes, Body},
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures::FutureExt;
use futures_util::{stream, StreamExt};
use kv_index::compute_request_content_hashes;
use openai_protocol::worker::WorkerStatus;
use serde_json::{json, Value};
use smg_policy_controller::{
    decision::{Decision, Shed, ShedReason, Trace},
    door::Refused,
    lease::{ReleaseReason, TransportClass},
    spec::{KeySourceSpec, RoutingKeySpec},
    store::{
        BatchOutcome, HealthSink, KvEventSink, LeasesStore, LoadSink, LoadSource, MemoryStore,
        SessionsStore, WorkerFacts, WorkerStatus as RowStatus, WorkersStore, PUBLISH_WINDOW,
    },
    view::{fingerprint, Identity, LoadRow},
    AppliedSpec, DerivedKey, KeySource, PolicySpec,
};
use smg_worker::identity::basic::WorkerLoadGuard;

use super::{
    elapsed_micros, left_no_worker, mark, profile_of,
    testing::{bed, bed_under, cache_bed, store_blocks, MODEL},
    transport_class, Ask, End, Hashes, Held, Prompt, Sent, Unplaced, GATEWAY, NO_STAGE,
};
use crate::{
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig},
    middleware::{
        ADMISSION_QUEUE_FULL_CODE, ADMISSION_QUEUE_FULL_STATUS, ADMISSION_QUEUE_TIMEOUT_CODE,
        ADMISSION_QUEUE_TIMEOUT_STATUS,
    },
    observability::{cache_trace, metrics::test_support::render_with_recorder},
    routers::common::{
        placement::refusal_reason, retry::is_retryable_response, stream::GuardedStream,
    },
};

/// The workers of a keyed case.
const URLS: [&str; 4] = [
    "http://w0:8000",
    "http://w1:8000",
    "http://w2:8000",
    "http://w3:8000",
];

/// The header a `manual` rule reads its key from, as the gateway's
/// configuration names it where it names none.
const KEY_HEADER: &str = "x-smg-routing-key";

/// A request's headers with `key` under `name`.
fn keyed(name: &'static str, key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = HeaderValue::from_str(key).expect("a header's value");
    headers.insert(name, key);
    headers
}

/// A key's two parts, owned: a derived key borrows its document.
fn owned(key: Identity<'_>) -> (u64, Option<String>) {
    (key.hash, key.plain.map(str::to_owned))
}

/// A source that is the request header `name`.
fn header(name: &str) -> KeySourceSpec {
    KeySourceSpec {
        header: Some(name.to_owned()),
        ..KeySourceSpec::default()
    }
}

/// Hands `store` the document of `policy` again, a generation later, with
/// `block` as the sources of its binding's key: an operator's apply.
fn name_sources(store: &MemoryStore, policy: PolicyConfig, block: RoutingKeySpec) {
    let config = RouterConfig {
        policy_controller: true,
        policy,
        ..RouterConfig::default()
    };
    let mut document = config.policy_document().expect("the routing's document");
    document.generation += 1;
    document.bindings[0].routing_key = Some(block);
    let generation = config
        .hosted_policy(&document)
        .expect("a document the gateway hosts");
    store.apply(generation).expect("a later generation");
}

/// The manual policy, its first worker drawn.
fn manual() -> PolicyConfig {
    PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    }
}

#[test]
fn each_mark_is_new() {
    let start = Arc::new(Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                (0..250).map(|_| mark()).collect::<Vec<_>>()
            })
        })
        .collect();
    let marks: HashSet<_> = threads
        .into_iter()
        .flat_map(|thread| thread.join().expect("the thread finished"))
        .collect();
    assert_eq!(marks.len(), 1000);
}

#[test]
fn each_status_reads_its_class() {
    for (status, answered, class) in [
        (StatusCode::OK, true, TransportClass::Ok),
        (StatusCode::FOUND, true, TransportClass::Ok),
        (StatusCode::NOT_FOUND, false, TransportClass::ClientError),
        (StatusCode::GATEWAY_TIMEOUT, true, TransportClass::Timeout),
        (StatusCode::GATEWAY_TIMEOUT, false, TransportClass::Timeout),
        (
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            TransportClass::ServerError,
        ),
        (
            StatusCode::BAD_GATEWAY,
            false,
            TransportClass::ConnectFailed,
        ),
    ] {
        assert_eq!(
            transport_class(status, answered),
            class,
            "{status}, {answered}"
        );
    }
}

#[test]
fn elapsed_microseconds_saturate_at_the_limits() {
    let sent = Instant::now();
    let later = sent + Duration::from_micros(17);
    assert_eq!(elapsed_micros(later, sent), 17);
    assert_eq!(elapsed_micros(sent, later), 0);
    let too_late = sent + Duration::from_micros(u64::from(u32::MAX) + 1);
    assert_eq!(elapsed_micros(too_late, sent), u32::MAX);
}

#[test]
fn an_http_end_times_the_send_and_any_first_data_frame() {
    let sent = Instant::now();
    let first = sent + Duration::from_micros(17);
    let now = sent + Duration::from_micros(41);
    let whole = End::http(StatusCode::OK, TransportClass::Ok, sent, None, now);
    let streamed = End::http(
        StatusCode::ACCEPTED,
        TransportClass::Ok,
        sent,
        Some(first),
        now,
    );
    assert_eq!(whole.status, 200);
    assert_eq!(whole.class, TransportClass::Ok);
    assert_eq!(whole.ttft_micros, 0);
    assert_eq!(streamed.status, 202);
    assert_eq!(streamed.ttft_micros, 17);
    for end in [whole, streamed] {
        assert_eq!(
            (end.tokens_in, end.tokens_out, end.duration_micros),
            (0, 0, 41)
        );
    }
}

/// A first attempt's ask of the request `id`.
pub(super) fn first(id: &[u8]) -> Ask<'_> {
    Ask::of(id, MODEL)
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
async fn a_placement_never_sent_says_nothing_at_its_end() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let now = Instant::now();
    let placed = Sent::from(host.place(&first(b"r1"), now).await.expect("a worker"));
    let row = placed.id();
    assert_eq!(live(&store, 1), 1);

    placed.ended(StatusCode::OK, true, now);
    assert_eq!(live(&store, 0), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, now));
}

#[tokio::test]
async fn a_whole_end_records_a_success_at_its_instant() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let sent = Instant::now();
    let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
    let row = placed.id();
    placed.dispatched(sent);
    let end = sent + Duration::from_millis(1);

    placed.ended(StatusCode::OK, true, end);
    assert_eq!(live(&store, 0), 0);
    assert!(store.completed_ok_since(GATEWAY, row, end));
    assert!(!store.completed_ok_since(GATEWAY, row, end + Duration::from_micros(1)));
}

#[tokio::test]
async fn a_body_keeps_its_head_and_reports_what_its_frames_show() {
    for failed in [false, true] {
        let (host, store, _) = bed(&["http://w0:8000"]);
        let sent = Instant::now();
        let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
        let row = placed.id();
        placed.dispatched(sent);

        let data = stream::iter([Ok(Bytes::from_static(b"a"))]);
        let error = stream::iter(failed.then(|| Err(io::Error::other("body failed"))));
        let mut response = Response::new(Body::from_stream(data.chain(error)));
        *response.status_mut() = StatusCode::ACCEPTED;
        response
            .headers_mut()
            .insert("x-test", HeaderValue::from_static("worker"));
        response.extensions_mut().insert(42u32);

        let response = placed.held_to_the_end_of(response);
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(response.headers()["x-test"], "worker");
        assert_eq!(response.extensions().get::<u32>(), Some(&42));
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
        let mut body = response.into_body().into_data_stream();
        let first = body.next().await.expect("the data frame").expect("data");
        assert_eq!(first.as_ref(), b"a");
        if failed {
            let error = body
                .next()
                .await
                .expect("the error frame")
                .expect_err("body failure");
            assert_eq!(error.to_string(), "body failed");
            // A server drops a failed body here, so its lease ends here.
            assert_eq!(live(&store, 0), 0);
        } else {
            assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
        }
        assert!(body.next().await.is_none());
        assert_eq!(live(&store, 0), 0);
        assert_eq!(store.completed_ok_since(GATEWAY, row, sent), !failed);
    }
}

#[tokio::test]
async fn a_body_the_relay_marked_cut_ends_as_a_failure() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let sent = Instant::now();
    let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
    let row = placed.id();
    placed.dispatched(sent);
    // The relay ends a cut body cleanly and writes the cut on the mark.
    placed.cut_mark().store(502, Ordering::Relaxed);

    let response = placed.held_to_the_end_of(Response::new(Body::from("data: a\n\n")));
    to_bytes(response.into_body(), 64)
        .await
        .expect("a clean end");
    assert_eq!(live(&store, 0), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, sent));
}

#[tokio::test]
async fn the_placement_drops_before_its_body() {
    struct ReadLiveOnDrop {
        store: Arc<MemoryStore>,
        live: Arc<AtomicUsize>,
    }

    impl Drop for ReadLiveOnDrop {
        fn drop(&mut self) {
            self.live
                .store(LeasesStore::live(&*self.store, GATEWAY), Ordering::SeqCst);
        }
    }

    let (host, store, workers) = bed(&["http://w0:8000"]);
    let sent = Instant::now();
    let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
    let row = placed.id();
    placed.dispatched(sent);
    let observed = Arc::new(AtomicUsize::new(1));
    let probe = ReadLiveOnDrop {
        store: Arc::clone(&store),
        live: Arc::clone(&observed),
    };
    let guard = WorkerLoadGuard::with_key(Arc::clone(placed.worker()), None);
    let body = Body::from_stream(GuardedStream::new(
        stream::pending::<Result<Bytes, Infallible>>(),
        (probe, guard),
    ));

    let response = placed.held_to_the_end_of(Response::new(body));
    assert_eq!(
        (LeasesStore::live(&*store, GATEWAY), workers[0].load()),
        (1, 1)
    );
    drop(response);
    assert_eq!(observed.load(Ordering::SeqCst), 0);
    assert_eq!(workers[0].load(), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, sent));
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
        assert_eq!(
            returning.request(Identity::NONE, &[]).facts.block_size,
            block_size
        );
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
                emptied_by: NO_STAGE,
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
            None,
        ),
        (Unplaced::Queued, unavailable, shed_code, Some("1"), None),
        (
            shed(ShedReason::NoEndpoints),
            unavailable,
            "no_available_workers",
            Some("1"),
            None,
        ),
        (
            Unplaced::NoTargetHolds,
            unavailable,
            "no_available_workers",
            Some("1"),
            None,
        ),
        (
            shed(ShedReason::FlowLimit),
            ADMISSION_QUEUE_FULL_STATUS,
            ADMISSION_QUEUE_FULL_CODE,
            Some("2"),
            Some("a queue of the policy controller is at its limit"),
        ),
        (
            shed(ShedReason::Deadline),
            ADMISSION_QUEUE_TIMEOUT_STATUS,
            ADMISSION_QUEUE_TIMEOUT_CODE,
            Some("2"),
            Some("the decision's budget passed before a worker was picked"),
        ),
        (
            shed(ShedReason::PolicyDeclined),
            unavailable,
            "policy_declined",
            None,
            None,
        ),
        (
            Unplaced::Refused(Refused::Faulted),
            unavailable,
            "policy_controller_unavailable",
            Some("1"),
            None,
        ),
    ];
    for (unplaced, status, code, retry_after, message) in rows {
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
        if let Some(message) = message {
            assert_eq!(error["error"]["message"], message, "{unplaced:?}");
        }
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

#[tokio::test]
async fn a_declined_answer_names_no_cause_and_its_line_names_what_declined() {
    for emptied_by in ["binding", "affinity-column", "random"] {
        let declined = Unplaced::Shed {
            reason: ShedReason::PolicyDeclined,
            emptied_by,
            retry_after_secs: 0,
        };
        let answer = declined.answer(MODEL);
        assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!answer.headers().contains_key(RETRY_AFTER));
        assert!(!is_retryable_response(&answer));
        let reason = format!("the policy declined at {emptied_by}");
        assert_eq!(refusal_reason(&answer), Some(reason.as_str()));
        let body = to_bytes(answer.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let error: Value = serde_json::from_slice(&body).expect("a JSON error object");
        assert_eq!(error["error"]["code"], "policy_declined");
        assert_eq!(
            error["error"]["message"],
            format!("The policy declined the request for model '{MODEL}'"),
            "{emptied_by}"
        );
    }
}

#[test]
fn a_pool_no_stage_emptied_reads_its_own_reason() {
    assert_eq!(
        left_no_worker(NO_STAGE),
        "the policy controller holds no candidate for the model"
    );
    assert_eq!(
        left_no_worker("healthy"),
        "the policy controller's healthy stage left no worker"
    );
}

#[tokio::test]
async fn an_ask_states_the_key_its_bindings_sources_hold() {
    let headers = keyed(KEY_HEADER, "conv-1");
    let ask = Ask {
        headers: Some(&headers),
        ..first(b"request")
    };
    let (_, store, _) = bed_under(manual(), &URLS[..1]);
    store.deciding(|world| {
        let derived = ask.key(world.spec()).expect("the header's key");
        assert_eq!(derived.key, Identity::plain("conv-1"));
        assert_eq!(derived.source, KeySource::Header(KEY_HEADER));
        let stated = ask.request(derived.key, &[]).facts.session_key;
        assert_eq!(stated, derived.key);
        let bare = first(b"request");
        assert_eq!(bare.key(world.spec()), None, "no header");
    });
    // A binding that names no source gives its requests no key.
    let (_, store, _) = bed(&URLS[..1]);
    store.deciding(|world| assert_eq!(ask.key(world.spec()), None));
}

#[tokio::test]
async fn a_key_is_read_by_the_documents_sources_in_their_order() {
    let (_, store, _) = bed_under(PolicyConfig::ConsistentHashing, &URLS[..1]);
    let block = RoutingKeySpec {
        sources: vec![
            header("x-conversation"),
            header(KEY_HEADER),
            KeySourceSpec {
                rid: true,
                ..KeySourceSpec::default()
            },
        ],
        strip_lineage: true,
    };
    name_sources(&store, PolicyConfig::ConsistentHashing, block);
    // A header the view does not list leaves as its fingerprint alone.
    let hashed = |key: &str| Identity::hashed(fingerprint(key.as_bytes()));
    let key_of = |headers: Option<&HeaderMap>, rid: Option<&str>| {
        let ask = Ask {
            headers,
            rid,
            ..first(b"request")
        };
        let key = |derived: DerivedKey<'_>| owned(derived.key);
        store.deciding(|world| {
            ask.key(world.spec())
                .map_or_else(|| owned(Identity::NONE), key)
        })
    };

    let mut headers = keyed("x-conversation", "conv_t2_r1");
    headers.insert(KEY_HEADER, HeaderValue::from_static("hint_t2"));
    let rid = Some("rid-key_r1");
    assert_eq!(key_of(Some(&headers), rid), owned(hashed("conv")));
    headers.remove("x-conversation");
    assert_eq!(key_of(Some(&headers), rid), owned(hashed("hint")));
    assert_eq!(key_of(None, rid), owned(Identity::plain("rid-key")));
    assert_eq!(key_of(Some(&HeaderMap::new()), None), owned(Identity::NONE));
}

/// A keyed attempt counts where its key came from, under the series the
/// gateway's own keyed selection counts by: a header, or the request's own
/// id. It counts once, however many asks its commits take, and an attempt
/// with no key counts none.
#[tokio::test]
async fn an_attempt_counts_the_source_of_its_key_once() {
    let (host, store, _) = bed_under(PolicyConfig::ConsistentHashing, &URLS);
    let block = RoutingKeySpec {
        sources: vec![
            header(KEY_HEADER),
            KeySourceSpec {
                rid: true,
                ..KeySourceSpec::default()
            },
        ],
        strip_lineage: false,
    };
    name_sources(&store, PolicyConfig::ConsistentHashing, block);
    let headers = keyed(KEY_HEADER, "conv-1");
    let by_header = Ask {
        headers: Some(&headers),
        ..first(b"by header")
    };
    let by_rid = Ask {
        rid: Some("rid-1"),
        ..first(b"by rid")
    };
    let now = Instant::now();
    let rendered = render_with_recorder(|| {
        // An attempt's first ask counts its key's source. Its ask again,
        // behind a commit that missed, names the missed target and counts
        // nothing.
        let asked = host.ask(&by_header, &[], now).now_or_never();
        let asked = asked.expect("the door decides at once").expect("a pick");
        let missed = [asked.targets.first().expect("a target").id];
        drop(asked);
        let again = host.ask(&by_header, &missed, now).now_or_never();
        assert!(again.expect("the door decides at once").is_ok());
        for ask in [by_rid, first(b"no key")] {
            let placed = host.place(&ask, now).now_or_never();
            assert!(placed.expect("the door decides at once").is_ok());
        }
    });
    for expected in [
        "smg_routing_key_source_total{source=\"header\"} 1",
        "smg_routing_key_source_total{source=\"rid\"} 1",
    ] {
        let counted = rendered.lines().any(|line| line == expected);
        assert!(counted, "{expected} in:\n{rendered}");
    }
}

#[tokio::test]
async fn a_key_keeps_its_worker_across_requests_under_the_manual_policy() {
    let (host, store, _) = bed_under(manual(), &URLS);
    let now = Instant::now();
    let headers = keyed(KEY_HEADER, "conv-1");
    let ask = Ask {
        headers: Some(&headers),
        ..first(b"first")
    };
    let mut placed = host.place(&ask, now).await.expect("a worker");
    placed.dispatched(now);
    placed.completed(ok(), now);

    for turn in 0..8u8 {
        let mut next = host
            .place(&Ask { id: &[turn], ..ask }, now)
            .await
            .expect("the pinned worker");
        assert!(Arc::ptr_eq(placed.worker(), next.worker()));
        assert_eq!(placed.id(), next.id());
        next.dispatched(now);
        next.completed(ok(), now);
    }
    assert_eq!(SessionsStore::len(&*store), 1);
}

#[tokio::test]
async fn a_key_keeps_its_worker_under_consistent_hashing() {
    let ring = PolicyConfig::ConsistentHashing;
    let (host, store, _) = bed_under(ring.clone(), &URLS);
    // The converter names no source for the ring yet: the document names
    // the header here, as an operator's would.
    let block = RoutingKeySpec {
        sources: vec![header(KEY_HEADER)],
        strip_lineage: false,
    };
    name_sources(&store, ring, block);
    let now = Instant::now();
    let headers = keyed(KEY_HEADER, "conv-1");
    let ask = Ask {
        headers: Some(&headers),
        ..first(b"first")
    };
    let mut placed = host.place(&ask, now).await.expect("a worker");
    placed.released(ReleaseReason::ClientGone, now);
    for turn in 0..8u8 {
        let mut next = host
            .place(&Ask { id: &[turn], ..ask }, now)
            .await
            .expect("the ring's owner");
        assert_eq!(placed.id(), next.id());
        next.released(ReleaseReason::ClientGone, now);
    }

    let mut owners = BTreeSet::new();
    for turn in 8..28u8 {
        let (id, headers) = ([turn], keyed(KEY_HEADER, &format!("conv-{turn}")));
        let ask = Ask {
            headers: Some(&headers),
            ..first(&id)
        };
        let mut placed = host.place(&ask, now).await.expect("a worker");
        owners.insert(placed.id());
        placed.released(ReleaseReason::ClientGone, now);
    }
    assert!(owners.len() >= 2, "ring owners: {owners:?}");
    assert!(SessionsStore::is_empty(&*store));
}

#[tokio::test]
async fn a_retry_of_a_key_steps_off_the_worker_it_tried() {
    let headers = keyed(KEY_HEADER, "conv-1");
    for urls in [&URLS[..2], &URLS[..1]] {
        let (host, _, _) = bed_under(manual(), urls);
        let ask = Ask {
            headers: Some(&headers),
            ..first(b"request")
        };
        let now = Instant::now();
        let mut placed = host.place(&ask, now).await.expect("a worker");
        placed.dispatched(now);
        placed.completed(
            End {
                status: 503,
                class: TransportClass::ServerError,
                ..ok()
            },
            now,
        );

        let tried = [placed.id()];
        let now = Instant::now();
        let mut retry = host
            .place(
                &Ask {
                    tried: &tried,
                    ..ask
                },
                now,
            )
            .await
            .expect("a worker for the retry");
        assert_eq!(retry.id() == placed.id(), urls.len() == 1);
        retry.dispatched(now);
        retry.completed(ok(), now);
    }
}

#[tokio::test]
async fn a_key_whose_client_left_before_the_send_keeps_no_pin() {
    let (host, store, _) = bed_under(manual(), &URLS);
    let now = Instant::now();
    let headers = keyed(KEY_HEADER, "conv-1");
    let ask = Ask {
        headers: Some(&headers),
        ..first(b"left")
    };
    let mut placed = host.place(&ask, now).await.expect("a worker");
    placed.released(ReleaseReason::ClientGone, now);
    assert!(SessionsStore::is_empty(&*store));

    let mut sent = host
        .place(&Ask { id: b"sent", ..ask }, now)
        .await
        .expect("a worker");
    sent.dispatched(now);
    assert_eq!(SessionsStore::len(&*store), 1);
    sent.completed(ok(), now);
}
