//! Cases of the host over the module's test support: rows written by hand
//! through the store's own door, beside workers registered by hand.

use std::{
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error::Error,
    future::{ready, Future},
    io,
    ops::Range,
    pin::pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Barrier,
    },
    task::{Context, Poll},
    thread,
    time::{Duration, Instant},
};

use axum::{
    body::{to_bytes, Body},
    http::{
        header::{CONTENT_LENGTH, RETRY_AFTER},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::Response,
};
use bytes::Bytes;
use futures::{poll, task::noop_waker, FutureExt};
use futures_util::{stream, StreamExt};
use kv_index::compute_request_content_hashes;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use parking_lot::Mutex;
use serde_json::{json, Value};
use smg_policy_controller::{
    decision::{Decision, Shed, ShedReason, Trace},
    door::Refused,
    spec::{KeySourceSpec, Pairing, RoutingKeySpec},
    store::{
        Ask as StoreAsk, BatchOutcome, HealthSink, KvEventSink, LeaseState, LeasesStore, LoadSink,
        LoadSource, MemoryStore, PairBuffers, PairingDescriptor, SessionsStore, WorkerFacts,
        WorkerStatus as RowStatus, WorkersStore, PUBLISH_WINDOW,
    },
    view::{
        fingerprint, Candidate, Identity, LoadRow, MediaCounts, Phase, RequestFacts, Role,
        WireModes, WorkerId,
    },
    AppliedSpec, Catalog, DerivedKey, KeySource, Lease, LeaseId, Lifecycle, Membership, Outcome,
    Picker, PolicySpec, ReleaseReason, Side, Target, Targets, TieBand, TransportClass,
};
use smg_worker::{
    health::circuit_breaker::CircuitBreakerConfig,
    identity::basic::{Worker, WorkerLoadGuard, WorkerType},
};

use super::{
    elapsed_micros, left_no_worker, mark, profile_of,
    testing::{bed, bed_under, cache_bed, record_bed, store_blocks, MODEL},
    transport_class, Ask, End, Hashes, Held, Host, Placed, Prompt, Sent, Unplaced, GATEWAY,
    NO_STAGE,
};
use crate::{
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig},
    middleware::{
        ADMISSION_QUEUE_FULL_CODE, ADMISSION_QUEUE_FULL_STATUS, ADMISSION_QUEUE_TIMEOUT_CODE,
        ADMISSION_QUEUE_TIMEOUT_STATUS,
    },
    observability::{cache_trace, metrics::test_support::render_with_recorder},
    routers::{
        common::{
            placement::refusal_reason,
            prefill_admission::{
                acquire_decided, test_support::select_worker, PrefillAdmission,
                PrefillAdmissionAttempt, PrefillAdmissionRejection, PrefillCandidateError,
            },
            retry::is_retryable_response,
            snapshot::WorkerRegistry,
            stream::GuardedStream,
        },
        grpc::backend_client::BasicWorkerBuilder,
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

/// An HTTP/1.1 server writes a declared length and asks for no frame after
/// it. So the lease ends at the last declared byte, and a body dropped
/// before that byte is cancelled.
#[tokio::test]
async fn a_sized_body_ends_its_lease_at_its_last_declared_byte() {
    for whole in [true, false] {
        let (host, store, _) = bed(&["http://w0:8000"]);
        let sent = Instant::now();
        let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
        let row = placed.id();
        placed.dispatched(sent);

        let frames = [b"ab", b"cd"].map(|data| Ok::<_, Infallible>(Bytes::from_static(data)));
        let mut response = Response::new(Body::from_stream(stream::iter(frames)));
        response
            .headers_mut()
            .insert(CONTENT_LENGTH, HeaderValue::from_static("4"));
        let response = placed.held_to_the_end_of(response);
        let mut body = response.into_body().into_data_stream();

        body.next().await.expect("the first frame").expect("data");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
        if whole {
            body.next().await.expect("the last frame").expect("data");
            assert_eq!(live(&store, 0), 0);
        }
        drop(body);
        assert_eq!(live(&store, 0), 0);
        assert_eq!(store.completed_ok_since(GATEWAY, row, sent), whole);
    }
}

/// A body with bytes still to come: a stream that never yields.
fn unended() -> Body {
    Body::from_stream(stream::pending::<Result<Bytes, Infallible>>())
}

/// A server asks a body with nothing to send for no frame: a body at its
/// end already, a declared length of zero, a status that declares no body.
/// So the lease ends at the head, while the response is still held, as the
/// answer it was. A cut the relay marked still decides the class.
#[tokio::test]
async fn a_response_with_nothing_to_send_ends_its_lease_at_its_head() {
    const OK: StatusCode = StatusCode::OK;
    for (status, at_end, declared, cut, ok) in [
        (OK, true, None, None, true),
        // The body's own word decides over a declared length.
        (OK, true, Some("5"), None, true),
        (OK, false, Some("0"), None, true),
        (StatusCode::CONTINUE, false, None, None, true),
        (StatusCode::NO_CONTENT, false, None, None, true),
        (StatusCode::NOT_MODIFIED, false, None, None, true),
        (OK, true, None, Some(StatusCode::GATEWAY_TIMEOUT), false),
    ] {
        let (host, store, _) = bed(&["http://w0:8000"]);
        let sent = Instant::now();
        let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
        let row = placed.id();
        placed.dispatched(sent);
        if let Some(cut) = cut {
            placed.cut_mark().store(cut.as_u16(), Ordering::Relaxed);
        }

        let mut response = Response::new(if at_end { Body::empty() } else { unended() });
        *response.status_mut() = status;
        if let Some(length) = declared {
            response
                .headers_mut()
                .insert(CONTENT_LENGTH, HeaderValue::from_static(length));
        }
        let held = placed.held_to_the_end_of(response);
        assert_eq!(live(&store, 0), 0, "{status} {at_end} {declared:?}");
        assert_eq!(
            store.completed_ok_since(GATEWAY, row, sent),
            ok,
            "{status} {at_end} {declared:?} {cut:?}"
        );
        drop(held);
    }
}

/// A stream with bytes to come, dropped before its first poll, went out to
/// nobody. Its lease ends cancelled, like one dropped mid-way.
#[tokio::test]
async fn a_pending_stream_dropped_unpolled_is_cancelled() {
    let (host, store, _) = bed(&["http://w0:8000"]);
    let sent = Instant::now();
    let mut placed = Sent::from(host.place(&first(b"r1"), sent).await.expect("a worker"));
    let row = placed.id();
    placed.dispatched(sent);

    let response = placed.held_to_the_end_of(Response::new(unended()));
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    drop(response);
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
    assert!(placed.decode().is_none());
    assert_eq!((placed.decode_id(), placed.decode_rank()), (None, None));
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
    let shed = |profile, emptied_by| {
        Decision::Shed(Shed {
            reason: ShedReason::PolicyDeclined,
            emptied_by,
            retry_after_secs: 1,
            wait_allowed: false,
            trace: Trace {
                profile,
                ..Trace::default()
            },
        })
    };
    // The trace's mark decides, and a stage's word decides nothing: rows
    // not written decline under their profile, and so does a plugin stage
    // that carries the word of a shed the engine makes ahead of a profile.
    for stage in ["written-rows", "binding", "random"] {
        assert_eq!(profile_of(&shed(3, stage)), Some(3), "{stage}");
        assert_eq!(profile_of(&shed(Trace::NO_PROFILE, stage)), None, "{stage}");
    }
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
                ..Prompt::default()
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
            returning.request(Identity::NONE, 0, &[]).facts.block_size,
            block_size
        );
    }
}

#[tokio::test]
async fn a_returning_prompt_lands_on_the_worker_it_was_sent_to() {
    // No size of record: a block is the document's 16 tokens, and a window
    // those tokens at four bytes.
    let text: String = ('a'..='z').cycle().take(256).collect();
    let tokens: Vec<u32> = (1..=64).collect();
    let as_text = Prompt {
        text: Some(&text),
        ..Prompt::default()
    };
    let as_tokens = Prompt {
        tokens: Some(&tokens),
        ..Prompt::default()
    };
    for prompt in [as_text, as_tokens] {
        let (host, store) = record_bed();
        let hashes = Hashes::of(&store, MODEL, &prompt);
        assert!(!hashes.blocks.is_empty());
        let now = Instant::now();
        let cold = Ask {
            hashes: Some(&hashes),
            ..first(b"cold")
        };
        let sent = host.place(&cold, now).await.expect("a worker").id();
        // Every other row reports less load: the record alone brings it back.
        for row in store.snapshot().1 {
            let running = u32::from(row.id == sent);
            let load = LoadRow {
                running,
                ..LoadRow::default()
            };
            store.record(row.id, 0, load, now, LoadSource::Stream);
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
            let placed = host.place(&returning, now).await.expect("a worker");
            assert_eq!(placed.id(), sent, "request {turn}");
            let placed = host.place(&plain, now).await.expect("a worker");
            plain_matches += usize::from(placed.id() == sent);
        }
        assert!(plain_matches < 32);
    }
}

#[tokio::test]
async fn a_worker_the_gateway_cannot_send_to_is_passed_over() {
    // The filter's two halves: a worker that is not ready, and a ready
    // worker whose breaker is open.
    for open_breaker in [false, true] {
        let (host, _store, workers) = bed(&["http://w0:8000", "http://w1:8000"]);
        if open_breaker {
            for _ in 0..CircuitBreakerConfig::default().failure_threshold {
                workers[0].record_outcome(503);
            }
            assert!(!workers[0].circuit_breaker_can_execute());
        } else {
            workers[0].set_status(WorkerStatus::NotReady);
        }
        for turn in 0..8u8 {
            let placed = host
                .place(&first(&[turn]), Instant::now())
                .await
                .expect("the other worker");
            assert_eq!(placed.worker().url(), "http://w1:8000");
        }
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
        (Unplaced::NoRoom, unavailable, shed_code, Some("2"), None),
        (
            Unplaced::Admission(PrefillAdmissionRejection::QueueFull),
            StatusCode::TOO_MANY_REQUESTS,
            "pd_prefill_queue_full",
            None,
            None,
        ),
        (
            Unplaced::Admission(PrefillAdmissionRejection::QueueTimeout),
            StatusCode::TOO_MANY_REQUESTS,
            "pd_prefill_queue_timeout",
            None,
            None,
        ),
        (
            Unplaced::Admission(PrefillAdmissionRejection::Unavailable),
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
        assert_eq!(unplaced.is_at_capacity(), unplaced == Unplaced::NoRoom);
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
        let (derived, client) = ask.keys(world.spec());
        let derived = derived.expect("the header's key");
        assert_eq!(derived.key, Identity::plain("conv-1"));
        assert_eq!(derived.source, KeySource::Header(KEY_HEADER));
        assert_eq!(client, 0);
        let stated = ask.request(derived.key, client, &[]).facts.session_key;
        assert_eq!(stated, derived.key);
        let bare = first(b"request");
        assert_eq!(bare.keys(world.spec()), (None, 0), "no header");
    });
    // A binding that names no source gives its requests no key.
    let (_, store, _) = bed(&URLS[..1]);
    store.deciding(|world| assert_eq!(ask.keys(world.spec()), (None, 0)));
}

#[tokio::test]
async fn an_ask_with_no_routing_key_states_its_clients_key() {
    // The hash ring's document names the two client headers behind the
    // routing key header.
    let (_, store, _) = bed_under(PolicyConfig::ConsistentHashing, &URLS[..1]);
    let keys_of = |headers: &HeaderMap| {
        let ask = Ask {
            headers: Some(headers),
            ..first(b"request")
        };
        store.deciding(|world| {
            let (derived, client) = ask.keys(world.spec());
            let stated = ask.request(Identity::NONE, client, &[]).facts.client_key;
            assert_eq!(stated, client);
            (derived.is_none(), client)
        })
    };
    let mut headers = keyed("cookie", "session=abc");
    assert_eq!(keys_of(&headers), (true, fingerprint(b"session=abc")));
    headers.insert("x-forwarded-for", HeaderValue::from_static("10.0.0.1"));
    assert_eq!(keys_of(&headers), (true, fingerprint(b"10.0.0.1")));
    // A routing key leaves the client's unread.
    headers.insert(KEY_HEADER, HeaderValue::from_static("conv-1"));
    assert_eq!(keys_of(&headers), (false, 0));
    assert_eq!(keys_of(&HeaderMap::new()), (true, 0));
}

/// An ask states its request's media, and the request carries the counts
/// as stated. An ask that states none carries zero.
#[test]
fn an_ask_states_its_requests_media() {
    let media = MediaCounts {
        images: 2,
        audio: 1,
        bytes: 4_096,
        ..MediaCounts::NONE
    };
    let ask = Ask {
        media,
        ..first(b"media")
    };
    assert_eq!(ask.request(Identity::NONE, 0, &[]).facts.media, media);
    let bare = first(b"bare").request(Identity::NONE, 0, &[]);
    assert_eq!(bare.facts.media, MediaCounts::NONE);
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
            ask.keys(world.spec())
                .0
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
    let (host, store, _) = bed_under(PolicyConfig::ConsistentHashing, &URLS);
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
async fn a_clients_key_keeps_its_worker_under_consistent_hashing() {
    let (host, store, _) = bed_under(PolicyConfig::ConsistentHashing, &URLS);
    let now = Instant::now();
    // No routing key: the ring places each client's requests by its cookie.
    let mut owners = BTreeSet::new();
    for client in 0..20u8 {
        let headers = keyed("cookie", &format!("session={client}"));
        let mut placed_at = BTreeSet::new();
        for turn in 0..4u8 {
            let id = [client, turn];
            let ask = Ask {
                headers: Some(&headers),
                ..first(&id)
            };
            let mut placed = host.place(&ask, now).await.expect("a worker");
            placed_at.insert(placed.id());
            placed.released(ReleaseReason::ClientGone, now);
        }
        assert_eq!(placed_at.len(), 1, "client {client}");
        owners.extend(placed_at);
    }
    assert!(owners.len() >= 2, "ring owners: {owners:?}");
    // A client's key pins no session.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairCall {
    Issued(Lease),
    Dispatched(Side, WorkerId, u32),
    Completed(Outcome),
    Released(LeaseId, Side, ReleaseReason),
}

type PairCalls = Arc<Mutex<Vec<PairCall>>>;

#[derive(Debug)]
struct Publication {
    entered: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}

#[derive(Clone, Debug)]
struct PairPicker {
    name: &'static str,
    limit: usize,
    calls: Option<PairCalls>,
    publication: Option<Arc<Publication>>,
}

impl PairPicker {
    fn record(&self, call: PairCall) {
        if let Some(calls) = &self.calls {
            calls.lock().push(call);
        }
    }
}

impl Picker for PairPicker {
    fn name(&self) -> &'static str {
        self.name
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let mut kept = kept.to_vec();
        kept.sort_unstable_by_key(|&i| candidates[i as usize].dp_rank);
        for i in kept.into_iter().take(self.limit) {
            let row = &candidates[i as usize];
            out.push(Target {
                position: row.position,
                id: row.id,
                dp_rank: row.dp_rank,
                pair: None,
            });
        }
    }
}

impl Lifecycle for PairPicker {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.record(PairCall::Issued(*lease));
    }

    fn on_dispatched(&self, _: &Lease, target: &Target, side: Side, since: u32) {
        self.record(PairCall::Dispatched(side, target.id, since));
    }

    fn on_completed(&self, _: &Lease, outcome: &Outcome) {
        self.record(PairCall::Completed(*outcome));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.record(PairCall::Released(lease.id, lease.side, reason));
    }

    fn on_membership(&self, change: &Membership<'_>) {
        if let (Membership::Snapshot(_), Some(publication)) = (change, &self.publication) {
            let _ = publication.entered.send(());
            let _ = publication.resume.lock().recv();
        }
    }
}

fn pair_spec(
    mode: Pairing,
    decode_targets: usize,
    generation: u64,
    calls: &PairCalls,
    publication: Option<Arc<Publication>>,
    key: Option<&RoutingKeySpec>,
) -> Result<AppliedSpec, Box<dyn Error>> {
    let mut document = PolicySpec::pair_presets("least_load", "least_load")?;
    document.plans.pd_pair.pairing = mode;
    document.generation = generation;
    for binding in &mut document.bindings {
        binding.routing_key = key.cloned();
    }
    let mut catalog = Catalog::builtin();
    let prefill = PairPicker {
        name: "ordered-prefill",
        limit: 4,
        calls: Some(Arc::clone(calls)),
        publication,
    };
    let decode = PairPicker {
        name: "ordered-decode",
        limit: decode_targets,
        calls: None,
        publication: None,
    };
    for (profile, picker) in [("prefill", prefill), ("decode", decode)] {
        let profile = document
            .profiles
            .get_mut(profile)
            .ok_or("the side's profile is missing")?;
        profile.filter = vec!["healthy".into(), "not-tried".into()];
        profile.score.clear();
        profile.pick = picker.name.into();
        catalog.picker(picker.name, move || Box::new(picker.clone()))?;
    }
    AppliedSpec::validate_in(&document, &catalog, &[]).map_err(Into::into)
}

struct PairBed {
    host: Host,
    store: Arc<MemoryStore>,
    prefills: Vec<(WorkerId, Arc<dyn Worker>)>,
    decodes: Vec<(WorkerId, Arc<dyn Worker>)>,
    calls: PairCalls,
}

impl PairBed {
    fn new(
        mode: Pairing,
        prefills: u16,
        decodes: u16,
        decode_targets: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let store = Arc::new(MemoryStore::new());
        let registry = Arc::new(WorkerRegistry::new());
        let add = |name,
                   role,
                   worker_type,
                   ranks: Range<u16>,
                   protocol: Option<&str>|
         -> Result<Vec<_>, Box<dyn Error>> {
            ranks
                .map(|rank| {
                    let url = format!("http://{name}-{rank}:8000");
                    let id = store
                        .upsert(WorkerFacts {
                            key: url.clone(),
                            engine: url.clone(),
                            engine_kind: "runtime".to_owned(),
                            incarnation: 1,
                            role,
                            wire_modes: WireModes::HTTP,
                            models: vec![MODEL.to_owned()],
                            dp_rank: rank,
                            dp_size: rank + 1,
                            pairing: protocol.map(|protocol: &str| {
                                Arc::new(PairingDescriptor {
                                    protocol: Some(protocol.to_owned()),
                                    ..PairingDescriptor::default()
                                })
                            }),
                            ..WorkerFacts::default()
                        })
                        .id();
                    let row = WorkersStore::get(&*store, id).ok_or("the row is missing")?;
                    store
                        .set_status(id, RowStatus::Ready, row.revision)
                        .ok_or("the row did not become ready")?;
                    let worker: Arc<dyn Worker> = Arc::new(
                        BasicWorkerBuilder::new(&url)
                            .model(ModelCard::new(MODEL))
                            .worker_type(worker_type)
                            .status(WorkerStatus::Ready)
                            .circuit_breaker_config(CircuitBreakerConfig {
                                failure_threshold: 1,
                                ..CircuitBreakerConfig::default()
                            })
                            .build(),
                    );
                    registry
                        .register(Arc::clone(&worker))
                        .ok_or("the worker did not register")?;
                    Ok((id, worker))
                })
                .collect()
        };
        let protocol = match mode {
            Pairing::Strict | Pairing::Off => Some("prefill-wire"),
            Pairing::Lenient => None,
        };
        let decode_protocol = match mode {
            Pairing::Off => Some("decode-wire"),
            Pairing::Strict | Pairing::Lenient => Some("prefill-wire"),
        };
        let prefills = add(
            "p",
            Role::Prefill,
            WorkerType::Prefill,
            1..1 + prefills,
            protocol,
        )?;
        let decodes = add(
            "d",
            Role::Decode,
            WorkerType::Decode,
            4..4 + decodes,
            decode_protocol,
        )?;
        let calls = PairCalls::default();
        store.apply(pair_spec(mode, decode_targets, 1, &calls, None, None)?)?;
        store.publish_now();
        Ok(Self {
            host: Host::at_keys(Arc::clone(&store), registry),
            store,
            prefills,
            decodes,
            calls,
        })
    }

    async fn commit(&self) -> Result<Placed, Unplaced> {
        let now = Instant::now();
        let asked = self.host.ask(&pair_ask(), &[], now).await?;
        self.host.commit_pair(asked, None, &mut Vec::new(), now)
    }

    fn issued(&self, side: Side) -> Vec<Lease> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match *call {
                PairCall::Issued(lease) if lease.side == side => Some(lease),
                _ => None,
            })
            .collect()
    }

    fn counts(&self, ids: &[WorkerId]) -> Option<Vec<u32>> {
        self.store.with_world(|world| {
            let ask = StoreAsk {
                gateway: GATEWAY,
                now: Instant::now(),
                pinned: None,
                affinity: false,
                signal_cells: false,
            };
            let mut rows = PairBuffers::default();
            let built = world.build_pair(MODEL, &ask, &mut rows)?;
            world.count_pair_in_flight(&built, &ask, &mut rows);
            let pools = rows.pools();
            ids.iter()
                .map(|id| {
                    pools
                        .prefills
                        .iter()
                        .chain(pools.decodes)
                        .find(|row| row.id == *id)
                        .map(|row| row.in_flight)
                })
                .collect()
        })
    }
}

fn pair_ask() -> Ask<'static> {
    Ask {
        phase: Phase::Prefill,
        prompt_tokens: 37,
        ..first(b"pair")
    }
}

#[tokio::test]
async fn a_pair_lands_on_a_prefill_and_its_decode_with_the_ranks_its_targets_name() {
    for mode in [Pairing::Strict, Pairing::Lenient, Pairing::Off] {
        let bed = PairBed::new(mode, 1, 1, 1).expect("a pair fleet");
        let (prefill, worker) = &bed.prefills[0];
        let (decode, decoder) = &bed.decodes[0];
        let now = Instant::now();
        let asked = bed.host.ask(&pair_ask(), &[], now).await.expect("a pair");
        let target = *asked.targets.first().expect("a target");
        let partner = target.pair.expect("a decode target");
        let placed = bed
            .host
            .commit_pair(asked, None, &mut Vec::new(), now)
            .expect("a placement");
        assert_eq!((placed.id(), placed.rank()), (*prefill, target.dp_rank));
        assert!(Arc::ptr_eq(placed.worker(), worker));
        assert_eq!(placed.decode_id(), Some(*decode));
        assert_eq!(placed.decode_rank(), Some(partner.decode_dp_rank));
        assert_eq!((placed.rank(), placed.decode_rank()), (1, Some(4)));
        assert!(Arc::ptr_eq(placed.decode().expect("a decode"), decoder));
        for side in [Side::Prefill, Side::Decode] {
            assert_eq!(bed.issued(side)[0].tokens, 37);
        }
    }
}

#[test]
fn an_asks_own_token_count_wins_over_its_hashes_count() {
    let tokens = Hashes {
        prompt_tokens: 88,
        ..Hashes::default()
    };
    let ask = Ask {
        hashes: Some(&tokens),
        ..pair_ask()
    };
    assert_eq!(ask.request(Identity::NONE, 0, &[]).facts.prompt_tokens, 37);
}

#[test]
fn an_asks_stated_attempt_wins_over_the_count_of_its_rows() {
    // A failed pair names its prefill and its decode: two rows, one attempt.
    let rows = [WorkerId(3), WorkerId(4)];
    let counted = Ask {
        tried: &rows,
        ..pair_ask()
    };
    assert_eq!(counted.request(Identity::NONE, 0, &[]).facts.attempt, 2);
    let stated = Ask {
        attempt: Some(1),
        ..counted
    };
    let request = stated.request(Identity::NONE, 0, &[]);
    assert_eq!((request.facts.attempt, request.facts.tried), (1, &rows[..]));
}

#[tokio::test]
async fn a_prefill_without_room_is_passed_over_for_the_next_pair_target() {
    let bed = PairBed::new(Pairing::Lenient, 2, 2, 2).expect("a pair fleet");
    let (full, prefill) = &bed.prefills[0];
    bed.decodes[1].1.record_circuit_breaker_outcome(false);
    let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
    let _occupied = admission
        .admit(None, select_worker(prefill))
        .await
        .expect("held load");
    let now = Instant::now();
    let asked = bed
        .host
        .ask(&pair_ask(), &[], now)
        .await
        .expect("the first pair");
    assert_eq!(asked.targets.first().expect("a target").id, *full);
    let mut missed = vec![WorkerId(0)];
    let mut asked = Some(asked);
    let result = admission
        .admit(None, |room| {
            let result = bed.host.commit_pair(
                asked.take().expect("one commit"),
                Some(room),
                &mut missed,
                now,
            );
            assert_eq!(result.expect_err("the prefill is full"), Unplaced::NoRoom);
            PrefillAdmissionAttempt::<()>::AtCapacity
        })
        .await;
    assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
    assert_eq!(missed, [WorkerId(0)]);
    assert_eq!(live(&bed.store, 0), 0);
    let other = bed
        .host
        .ask(&pair_ask(), &[*full], now)
        .await
        .expect("the other pair");
    let mut other = Some(other);
    let admitted = admission
        .admit(None, |room| {
            let placed = bed
                .host
                .commit_pair(
                    other.take().expect("one commit"),
                    Some(room),
                    &mut missed,
                    now,
                )
                .expect("the second prefill has room");
            room.select(Arc::clone(placed.worker()), placed)
        })
        .await
        .expect("the other pair is admitted");
    let placed = &admitted.selected;
    assert_eq!(placed.id(), bed.prefills[1].0);
    assert_eq!(placed.decode_id(), Some(bed.decodes[0].0));
}

/// The commit names the target it took to the handle, so the decode's
/// dispatch names the decode that is sent and not the pick's first. A
/// pick's targets share its prefill and differ by decode.
#[tokio::test]
async fn a_pair_taken_past_the_picks_first_target_dispatches_under_the_one_taken() {
    let bed = PairBed::new(Pairing::Lenient, 1, 2, 2).expect("a pair fleet");
    let now = Instant::now();
    let asked = bed.host.ask(&pair_ask(), &[], now).await.expect("a pair");
    let mut sides = asked.targets.iter().filter_map(|target| target.pair);
    let (first, second) = (
        sides.next().expect("the pick's first target").decode_id,
        sides.next().expect("a target behind it").decode_id,
    );
    assert_ne!(first, second);
    let cut = bed.decodes.iter().find(|(id, _)| *id == first);
    cut.expect("the first decode")
        .1
        .record_circuit_breaker_outcome(false);
    let mut missed = Vec::new();
    let mut placed = bed
        .host
        .commit_pair(asked, None, &mut missed, now)
        .expect("the target behind the first");
    let prefill = bed.prefills[0].0;
    assert_eq!((placed.id(), placed.decode_id()), (prefill, Some(second)));
    assert!(missed.is_empty());
    bed.calls.lock().clear();
    placed.dispatched(now + Duration::from_micros(5));
    placed.decode_dispatched(now + Duration::from_micros(8));
    assert_eq!(
        *bed.calls.lock(),
        [
            PairCall::Dispatched(Side::Prefill, prefill, 5),
            PairCall::Dispatched(Side::Decode, second, 8),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn a_pair_that_finds_no_room_queues_and_is_decided_again_at_its_admission() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
    let occupied = admission
        .admit(None, select_worker(&bed.prefills[0].1))
        .await
        .expect("held load");
    let ask = pair_ask();
    let decided_at = Instant::now();
    let mut clock = [decided_at, decided_at + Duration::from_millis(1)].into_iter();
    let attempt = acquire_decided(
        Some(&admission),
        None,
        Placed::worker,
        || ready(None),
        || {
            bed.host
                .ask(&ask, &[], clock.next().expect("one decision per wake"))
        },
        |asked, room| {
            bed.host
                .commit_pair(asked, room, &mut Vec::new(), Instant::now())
        },
    );
    tokio::pin!(attempt);
    assert!(poll!(attempt.as_mut()).is_pending());
    assert_eq!(admission.queued_requests(), 1);
    let first = bed.issued(Side::Prefill)[0].id;
    {
        let calls = bed.calls.lock();
        assert_eq!(calls.len(), 4);
        assert_eq!(
            calls[2],
            PairCall::Released(first, Side::Prefill, ReleaseReason::CommitFailed)
        );
        assert_eq!(
            calls[3],
            PairCall::Released(first, Side::Decode, ReleaseReason::CommitFailed)
        );
    }
    assert_eq!(live(&bed.store, 0), 0);
    drop(occupied);
    let Ok((placed, _guard)) = attempt.await else {
        panic!("a new decision is admitted");
    };
    let second = placed.handle.lease().expect("a lease").lease.id;
    assert_eq!(
        bed.issued(Side::Prefill)
            .iter()
            .map(|lease| lease.id)
            .collect::<Vec<_>>(),
        [first, second]
    );
    assert_ne!(first, second);
    assert!(LeasesStore::get(&*bed.store, GATEWAY, first).is_none());
    let record =
        LeasesStore::get(&*bed.store, GATEWAY, second).expect("the second decision's lease");
    assert_eq!(record.issued_at, decided_at + Duration::from_millis(1));
    assert_eq!(live(&bed.store, 1), 1);
}

#[tokio::test]
async fn a_decode_the_gateway_cannot_send_to_is_passed_over_and_named_in_the_next_ask() {
    for targets in [1, 2] {
        let bed = PairBed::new(Pairing::Lenient, 1, 2, targets).expect("a pair fleet");
        if targets == 1 {
            bed.decodes[0].1.set_status(WorkerStatus::NotReady);
        } else {
            bed.decodes[0].1.record_circuit_breaker_outcome(false);
        }
        let (placed, _) = bed
            .host
            .place_pair(&pair_ask(), None, None)
            .await
            .expect("the other decode");
        assert_eq!(placed.decode_id(), Some(bed.decodes[1].0));
        let issued: Vec<_> = bed
            .issued(Side::Decode)
            .iter()
            .map(|lease| lease.booked)
            .collect();
        if targets == 1 {
            assert_eq!(issued, [bed.decodes[0].0, bed.decodes[1].0]);
        } else {
            assert_eq!(issued, [bed.decodes[0].0]);
        }
        assert_eq!(live(&bed.store, 1), 1);
    }
}

#[tokio::test]
async fn a_missing_prefill_and_an_open_decode_breaker_are_both_misses() {
    let bed = PairBed::new(Pairing::Lenient, 1, 2, 2).expect("a pair fleet");
    let (prefill, worker) = &bed.prefills[0];
    let (decode, decoder) = &bed.decodes[0];
    bed.host
        .registry()
        .remove_by_url(worker.url())
        .expect("the prefill leaves the registry");
    decoder.record_circuit_breaker_outcome(false);
    let now = Instant::now();
    let asked = bed
        .host
        .ask(&pair_ask(), &[], now)
        .await
        .expect("the store still sees the pair");
    let mut missed = Vec::new();
    let result = bed.host.commit_pair(asked, None, &mut missed, now);
    assert_eq!(
        result.expect_err("no target holds"),
        Unplaced::NoTargetHolds
    );
    assert_eq!(
        missed.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([*prefill, *decode])
    );
    assert_eq!(live(&bed.store, 0), 0);
}

#[tokio::test]
async fn three_passes_with_no_target_end_the_attempt() {
    let bed = PairBed::new(Pairing::Lenient, 1, 4, 1).expect("a pair fleet");
    for (_, worker) in &bed.decodes {
        worker.set_status(WorkerStatus::NotReady);
    }
    let result = bed.host.place_pair(&pair_ask(), None, None).await;
    assert!(matches!(result, Err(Unplaced::NoTargetHolds)));
    let issued: Vec<_> = bed
        .issued(Side::Decode)
        .iter()
        .map(|lease| lease.booked)
        .collect();
    assert_eq!(
        issued,
        bed.decodes[..3]
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
    );
    assert_eq!(live(&bed.store, 0), 0);
}

#[tokio::test(start_paused = true)]
async fn a_queued_head_whose_decode_cannot_be_sent_to_asks_again_in_its_place() {
    let bed = PairBed::new(Pairing::Lenient, 1, 2, 1).expect("a pair fleet");
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
    let occupied = admission
        .admit(None, select_worker(&bed.prefills[0].1))
        .await
        .expect("held load");
    let ask = pair_ask();
    let head = bed.host.place_pair(&ask, Some(&admission), None);
    tokio::pin!(head);
    assert!(poll!(head.as_mut()).is_pending());
    let behind = bed.host.place_pair(&ask, Some(&admission), None);
    tokio::pin!(behind);
    assert!(poll!(behind.as_mut()).is_pending());
    assert_eq!(admission.queued_requests(), 2);

    // The controller still picks the first decode: its breaker is the
    // gateway's to know.
    bed.decodes[0].1.record_circuit_breaker_outcome(false);
    drop(occupied);
    assert!(poll!(behind.as_mut()).is_pending());
    let Poll::Ready(Ok((placed, _guard))) = poll!(head.as_mut()) else {
        panic!("the head asks again in its place and is admitted");
    };
    assert_eq!(placed.decode_id(), Some(bed.decodes[1].0));
    assert_eq!(admission.queued_requests(), 1);
    assert!(poll!(behind.as_mut()).is_pending());
    assert_eq!(live(&bed.store, 1), 1);
}

/// A keyed pair counts the source of its key once an attempt. Admission
/// decides again at each wake, from a list a wait on room leaves empty: the
/// attempt's first ask counts, and the asks of a later wake count nothing.
#[tokio::test]
async fn a_keyed_pair_that_wakes_twice_counts_its_keys_source_once() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let block = RoutingKeySpec {
        sources: vec![header(KEY_HEADER)],
        strip_lineage: false,
    };
    let keyed_pair = pair_spec(Pairing::Lenient, 1, 2, &bed.calls, None, Some(&block));
    let applied = bed
        .store
        .apply(keyed_pair.expect("a keyed pair's document"));
    applied.expect("a later generation");
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
    let headers = keyed(KEY_HEADER, "conv-1");
    let ask = Ask {
        headers: Some(&headers),
        ..pair_ask()
    };
    let prefill = &bed.prefills[0].1;
    let rendered = render_with_recorder(|| {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        // The prefill reads full twice over: an admitted request holds it,
        // and one more load stands beside.
        let occupied = admission.admit(None, select_worker(prefill));
        let occupied = occupied.now_or_never().expect("no waiter stands ahead");
        let occupied = occupied.expect("held load");
        let _beside = WorkerLoadGuard::with_key(Arc::clone(prefill), None);
        let mut head = pin!(bed.host.place_pair(&ask, Some(&admission), None));
        assert!(head.as_mut().poll(&mut cx).is_pending());
        let decided = bed.issued(Side::Prefill).len();
        assert_ne!(decided, 0, "the head decides as it arrives");
        // The admitted request ends: the head wakes, decides again and
        // still reads its prefill full.
        drop(occupied);
        assert!(head.as_mut().poll(&mut cx).is_pending());
        assert!(bed.issued(Side::Prefill).len() > decided, "a second wake");
    });
    let counted: Vec<_> = rendered
        .lines()
        .filter(|line| line.starts_with("smg_routing_key_source_total"))
        .collect();
    let once = ["smg_routing_key_source_total{source=\"header\"} 1"];
    assert_eq!(counted, once, "{rendered}");
}

#[tokio::test(start_paused = true)]
async fn the_queues_refusals_leave_no_lease_on_file() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let timeout = Duration::from_secs(5);
    let admission = PrefillAdmission::new(1, 1, timeout);
    let _occupied = admission
        .admit(None, select_worker(&bed.prefills[0].1))
        .await
        .expect("held load");
    let ask = pair_ask();
    let waiting = bed.host.place_pair(&ask, Some(&admission), None);
    tokio::pin!(waiting);
    assert!(poll!(waiting.as_mut()).is_pending());
    assert_eq!(live(&bed.store, 0), 0);
    let decided = bed.issued(Side::Prefill).len();
    assert!(matches!(
        bed.host
            .place_pair(&pair_ask(), Some(&admission), None)
            .await,
        Err(Unplaced::Admission(PrefillAdmissionRejection::QueueFull))
    ));
    assert_eq!(admission.queued_requests(), 1);
    assert_eq!(
        bed.issued(Side::Prefill).len(),
        decided,
        "a live fleet joins without deciding"
    );
    assert_eq!(live(&bed.store, 0), 0);
    tokio::time::advance(timeout).await;
    assert!(matches!(
        waiting.await,
        Err(Unplaced::Admission(PrefillAdmissionRejection::QueueTimeout))
    ));
    assert_eq!(admission.queued_requests(), 0);
    assert_eq!(live(&bed.store, 0), 0);
}

#[tokio::test]
async fn each_side_of_a_placed_pair_reports_its_own_dispatch_and_end() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let now = Instant::now();
    let asked = bed.host.ask(&pair_ask(), &[], now).await.expect("a pair");
    let mut placed = bed
        .host
        .commit_pair(asked, None, &mut Vec::new(), now)
        .expect("a placement");
    let (prefill, decode) = (bed.prefills[0].0, bed.decodes[0].0);
    let lease = placed.handle.lease().expect("a lease").lease.id;
    bed.calls.lock().clear();
    placed.dispatched(now + Duration::from_micros(5));
    placed.decode_dispatched(now + Duration::from_micros(8));
    let end = End {
        ttft_micros: 13,
        tokens_in: 37,
        tokens_out: 5,
        duration_micros: 233,
        ..ok()
    };
    placed.completed(end, now);
    let sides = bed
        .store
        .sides(GATEWAY, lease)
        .expect("decode still holds the lease");
    assert_eq!(sides.record.state, LeaseState::Completed);
    assert_eq!(
        sides.decode.expect("decode still stands").state,
        LeaseState::Dispatched {
            at: now + Duration::from_micros(8)
        }
    );
    placed.decode_completed(end, now);
    placed.completed(end, now);
    placed.decode_completed(end, now);
    assert_eq!(live(&bed.store, 0), 0);
    assert_eq!(
        *bed.calls.lock(),
        [
            PairCall::Dispatched(Side::Prefill, prefill, 5),
            PairCall::Dispatched(Side::Decode, decode, 8),
            PairCall::Completed(Outcome {
                side: Side::Prefill,
                served: prefill,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 13,
                tokens_in: 37,
                tokens_out: 5,
                duration_micros: 233
            }),
            PairCall::Completed(Outcome {
                side: Side::Decode,
                served: decode,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 13,
                tokens_in: 37,
                tokens_out: 5,
                duration_micros: 233
            }),
        ]
    );
}

/// A route's rule at the pair's commit: a target whose decode the rule
/// refuses is passed over like one the gateway cannot send to. With every
/// target refused the pick is released and the refused rows are missed.
#[tokio::test]
async fn the_routes_rule_passes_over_a_pair_at_its_commit() {
    let bed = PairBed::new(Pairing::Lenient, 1, 2, 2).expect("a pair fleet");
    let now = Instant::now();
    let url_of = |row: WorkerId| {
        let decode = bed.decodes.iter().find(|(id, _)| *id == row);
        decode.expect("a decode of the bed").1.url().to_owned()
    };
    let asked = bed.host.ask(&pair_ask(), &[], now).await.expect("a pair");
    let rows: Vec<_> = asked
        .targets
        .iter()
        .filter_map(|target| target.pair)
        .collect();
    let (first, second) = (rows[0].decode_id, rows[1].decode_id);
    let refused = url_of(first);
    let mut missed = Vec::new();
    let placed = bed
        .host
        .commit_pair_where(
            asked,
            |worker| worker.url() != refused,
            None,
            &mut missed,
            now,
        )
        .expect("the target behind the first");
    assert_eq!(placed.decode_id(), Some(second));
    assert!(missed.is_empty());
    drop(placed);

    let asked = bed.host.ask(&pair_ask(), &[], now).await.expect("a pair");
    let rows = asked.targets.iter().filter_map(|target| target.pair);
    let rows: Vec<_> = rows.map(|pair| pair.decode_id).collect();
    let refused = [url_of(first), url_of(second)];
    let none = bed.host.commit_pair_where(
        asked,
        |worker| !refused.iter().any(|url| url == worker.url()),
        None,
        &mut missed,
        now,
    );
    assert_eq!(none.err(), Some(Unplaced::NoTargetHolds));
    assert_eq!(missed, rows);
    assert_eq!(live(&bed.store, 0), 0);
}

/// A pair's router says the prefill's end where the prefill is done with,
/// ahead of the answer its decode writes. The answer's end then ends the
/// decode alone, so the prefill hears one completion.
#[tokio::test]
async fn a_sent_pairs_prefill_ends_ahead_of_its_answer() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let placed = bed.commit().await.expect("a placement");
    let lease = placed.handle.lease().expect("a lease").lease.id;
    let mut sent = Sent::from(placed);
    let now = Instant::now();
    bed.calls.lock().clear();
    sent.dispatched(now);
    sent.decode_dispatched(now);
    sent.prefill_ended(StatusCode::OK, now + Duration::from_micros(5));
    let sides = bed
        .store
        .sides(GATEWAY, lease)
        .expect("the decode still holds the lease");
    assert_eq!(sides.record.state, LeaseState::Completed);
    let decode = sides.decode.expect("the decode side").state;
    assert_eq!(decode, LeaseState::Dispatched { at: now });
    sent.ended(StatusCode::OK, true, now + Duration::from_micros(9));
    assert_eq!(live(&bed.store, 0), 0);
    let ended = |call: &PairCall| match call {
        PairCall::Completed(outcome) => Some(outcome.side),
        _ => None,
    };
    let ends: Vec<_> = bed.calls.lock().iter().filter_map(ended).collect();
    assert_eq!(ends, [Side::Prefill, Side::Decode]);
}

/// A pair's router that sends its decode alone says no dispatch of the
/// prefill. The answer's end then ends the decode, and the placement's drop
/// releases the prefill: its row hears no dispatch and no completion for a
/// request it never saw.
#[tokio::test]
async fn a_prefill_that_was_not_sent_hears_no_dispatch_and_no_end() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let placed = bed.commit().await.expect("a placement");
    let lease = placed.handle.lease().expect("a lease").lease.id;
    let decode = bed.decodes[0].0;
    let mut sent = Sent::from(placed);
    let now = Instant::now();
    bed.calls.lock().clear();
    sent.decode_dispatched(now);
    sent.prefill_ended(StatusCode::OK, now + Duration::from_micros(5));
    sent.ended(StatusCode::OK, true, now + Duration::from_micros(9));
    assert_eq!(live(&bed.store, 0), 0);
    let calls = bed.calls.lock();
    assert_eq!(calls.len(), 3, "{calls:?}");
    assert!(matches!(calls[0], PairCall::Dispatched(Side::Decode, id, _) if id == decode));
    assert!(matches!(calls[1], PairCall::Completed(outcome) if outcome.side == Side::Decode));
    assert_eq!(
        calls[2],
        PairCall::Released(lease, Side::Prefill, ReleaseReason::Abandoned)
    );
}

#[tokio::test]
async fn releasing_the_decode_leaves_the_prefill_on_file() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let mut placed = bed.commit().await.expect("a placement");
    let lease = placed.handle.lease().expect("a lease").lease.id;
    placed.decode_released(ReleaseReason::ClientGone, Instant::now());
    let sides = bed.store.sides(GATEWAY, lease).expect("the prefill stands");
    assert_eq!(sides.record.state, LeaseState::ISSUED);
    assert_eq!(
        sides.decode.expect("decode state").state,
        LeaseState::Released(ReleaseReason::ClientGone)
    );
    placed.released(ReleaseReason::ClientGone, Instant::now());
    assert_eq!(live(&bed.store, 0), 0);
}

#[tokio::test]
async fn the_next_decode_of_the_same_prefill_is_taken_before_the_decode_is_sent() {
    let bed = PairBed::new(Pairing::Lenient, 2, 3, 3).expect("a pair fleet");
    let mut placed = bed.commit().await.expect("a placement");
    let prefill = placed.id();
    let rank = placed.rank();
    let decodes = [bed.decodes[0].0, bed.decodes[1].0, bed.decodes[2].0];
    assert_eq!(placed.decode_id(), Some(decodes[0]));
    let ids = [prefill, decodes[0], decodes[1], decodes[2]];
    assert_eq!(bed.counts(&ids), Some(vec![1, 1, 0, 0]));
    assert!(!placed.next_decode(&bed.host), "the prefill must go first");
    placed.dispatched(Instant::now());
    bed.decodes[1].1.set_status(WorkerStatus::NotReady);
    assert!(placed.next_decode(&bed.host));
    assert_eq!((placed.id(), placed.rank()), (prefill, rank));
    assert_eq!(
        (placed.decode_id(), placed.decode_rank()),
        (Some(decodes[2]), Some(6))
    );
    assert!(Arc::ptr_eq(
        placed.decode().expect("the alternate"),
        &bed.decodes[2].1
    ));
    assert!(
        !placed.next_decode(&bed.host),
        "the earlier decodes stay behind"
    );
    assert_eq!(bed.counts(&ids), Some(vec![1, 1, 0, 0]));
    placed.decode_dispatched(Instant::now());
    assert_eq!(bed.counts(&ids), Some(vec![1, 0, 0, 1]));
    assert_eq!(live(&bed.store, 1), 1);
}

#[tokio::test]
async fn a_pair_without_admission_is_one_ask_and_one_commit() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let prefill = &bed.prefills[0].1;
    let _occupied = WorkerLoadGuard::with_key(Arc::clone(prefill), None);
    let (placed, guard) = bed
        .host
        .place_pair(&pair_ask(), None, Some("request"))
        .await
        .expect("a placement");
    assert_eq!(bed.issued(Side::Prefill).len(), 1);
    assert_eq!(live(&bed.store, 1), 1);
    assert_eq!(
        (
            prefill.load(),
            prefill.routing_key_load(),
            bed.decodes[0].1.load()
        ),
        (2, 1, 0)
    );
    assert_eq!(placed.id(), bed.prefills[0].0);
    drop(guard);
    assert_eq!((prefill.load(), prefill.routing_key_load()), (1, 0));
}

/// A route states the wire it dials, and the controller places among the
/// rows that serve it. No row of this fleet serves the other wire.
#[tokio::test]
async fn an_ask_that_states_its_wire_is_placed_among_the_rows_that_serve_it() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let ask = |wires| Ask {
        wires,
        ..pair_ask()
    };
    let now = Instant::now();
    let unplaced = bed.host.ask(&ask(WireModes::GRPC), &[], now).await.err();
    assert!(
        matches!(
            unplaced,
            Some(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "wires",
                ..
            })
        ),
        "{unplaced:?}"
    );
    assert_eq!(live(&bed.store, 0), 0);

    let asked = bed.host.ask(&ask(WireModes::HTTP), &[], now).await;
    let asked = asked.expect("the rows serve the wire");
    let placed = bed
        .host
        .commit_pair(asked, None, &mut Vec::new(), now)
        .expect("a placement");
    assert_eq!(placed.id(), bed.prefills[0].0);
}

#[tokio::test]
async fn a_commit_under_a_held_publication_returns() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let now = Instant::now();
    let asked = bed
        .host
        .ask(&pair_ask(), &[], now)
        .await
        .expect("a decision before publication");
    thread::scope(|scope| {
        let (entered, publishing) = mpsc::channel();
        let (resume, held) = mpsc::channel();
        let publication = Arc::new(Publication {
            entered,
            resume: Mutex::new(held),
        });
        let spec = pair_spec(Pairing::Lenient, 1, 2, &bed.calls, Some(publication), None)
            .expect("a new generation");
        let store = &bed.store;
        let writer = scope.spawn(move || store.apply(spec));
        let deadline = Duration::from_secs(3);
        publishing
            .recv_timeout(deadline)
            .expect("the writer holds membership");
        let (returned, result) = mpsc::channel();
        let host = &bed.host;
        scope.spawn(move || {
            let placement = host.commit_pair(asked, None, &mut Vec::new(), now);
            let _ = returned.send(placement);
        });
        let committed = result.recv_timeout(deadline);
        drop(resume);
        writer
            .join()
            .expect("the publication thread")
            .expect("the publication finished");
        assert_eq!(
            committed
                .expect("the commit returned while membership was locked")
                .expect("a placement")
                .id(),
            bed.prefills[0].0
        );
    });
}

#[tokio::test]
async fn a_decision_whose_targets_read_full_asks_again_and_names_them() {
    let bed = PairBed::new(Pairing::Lenient, 2, 1, 1).expect("a pair fleet");
    let (first, prefill) = &bed.prefills[0];
    let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
    let _occupied = admission
        .admit(None, select_worker(prefill))
        .await
        .expect("held load");
    let (placed, _guard) = bed
        .host
        .place_pair(&pair_ask(), Some(&admission), None)
        .await
        .expect("the next decision finds room");
    let issued: Vec<_> = bed
        .issued(Side::Prefill)
        .iter()
        .map(|lease| lease.booked)
        .collect();
    assert_eq!(issued, [*first, bed.prefills[1].0]);
    assert_eq!(placed.id(), bed.prefills[1].0);
    assert_eq!(live(&bed.store, 1), 1);
}

#[tokio::test(start_paused = true)]
async fn a_request_behind_a_waiter_whose_fleet_is_gone_is_answered_at_once() {
    for role in [Role::Prefill, Role::Decode] {
        let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let prefill = &bed.prefills[0].1;
        let _occupied = admission
            .admit(None, select_worker(prefill))
            .await
            .expect("held load");
        let older = admission.admit(None, select_worker(prefill));
        tokio::pin!(older);
        assert!(poll!(older.as_mut()).is_pending());
        let (id, worker) = if role == Role::Prefill {
            &bed.prefills[0]
        } else {
            &bed.decodes[0]
        };
        bed.host
            .registry()
            .remove_by_url(worker.url())
            .expect("the worker leaves the registry");
        let row = WorkersStore::get(&*bed.store, *id).expect("the row");
        bed.store
            .remove(*id, row.revision)
            .expect("the row leaves the store");
        let ask = pair_ask();
        let request = bed.host.place_pair(&ask, Some(&admission), None);
        tokio::pin!(request);
        assert!(matches!(
            poll!(request.as_mut()),
            Poll::Ready(Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            }))
        ));
        assert_eq!(admission.queued_requests(), 1);
        assert_eq!(live(&bed.store, 0), 0);
    }
}

#[tokio::test(start_paused = true)]
async fn a_probe_refuses_a_decode_ask_and_releases_a_pick_before_queuing() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
    let prefill = &bed.prefills[0].1;
    let _occupied = admission
        .admit(None, select_worker(prefill))
        .await
        .expect("held load");
    let older = admission.admit(None, select_worker(prefill));
    tokio::pin!(older);
    assert!(poll!(older.as_mut()).is_pending());
    bed.decodes[0].1.set_status(WorkerStatus::NotReady);

    let side = Ask {
        phase: Phase::Decode,
        ..pair_ask()
    };
    let refused = bed.host.place_pair(&side, Some(&admission), None);
    tokio::pin!(refused);
    assert!(matches!(
        poll!(refused.as_mut()),
        Poll::Ready(Err(Unplaced::Refused(Refused::PairSide)))
    ));
    assert_eq!(admission.queued_requests(), 1);

    let ask = pair_ask();
    let picked = bed.host.place_pair(&ask, Some(&admission), None);
    tokio::pin!(picked);
    assert!(poll!(picked.as_mut()).is_pending());
    assert_eq!(admission.queued_requests(), 2);
    assert_eq!(live(&bed.store, 0), 0);
    let first = bed.issued(Side::Prefill)[0].id;
    assert_eq!(
        &bed.calls.lock()[2..],
        [
            PairCall::Released(first, Side::Prefill, ReleaseReason::CommitFailed),
            PairCall::Released(first, Side::Decode, ReleaseReason::CommitFailed),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn a_decision_that_reads_every_target_full_waits_and_is_admitted_at_a_release() {
    let bed = PairBed::new(Pairing::Lenient, 1, 1, 1).expect("a pair fleet");
    let prefill = &bed.prefills[0].1;
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
    let occupied = admission
        .admit(None, select_worker(prefill))
        .await
        .expect("held load");
    assert!(admission.reads_full(prefill));
    let ask = pair_ask();
    let request = bed.host.place_pair(&ask, Some(&admission), None);
    tokio::pin!(request);
    assert!(poll!(request.as_mut()).is_pending());
    assert_eq!(admission.queued_requests(), 1);
    assert_eq!(bed.issued(Side::Prefill).len(), 3);
    assert_eq!(live(&bed.store, 0), 0);
    drop(occupied);
    assert!(!admission.reads_full(prefill));
    let (placed, _guard) = request
        .await
        .expect("the released slot admits the new decision");
    assert_eq!(placed.id(), bed.prefills[0].0);
    assert_eq!(bed.issued(Side::Prefill).len(), 4);
    assert_eq!(admission.queued_requests(), 0);
    assert_eq!(live(&bed.store, 1), 1);
}
