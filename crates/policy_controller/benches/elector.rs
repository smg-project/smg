//! The election's costs: the fence read a shard thread pays before every
//! decision (`policy_elector/fence/leader_term`), a holder's renewal over
//! the in-process record and over the lease file
//! (`policy_elector/<memory|file>/renew`: the read, the lock, the write and
//! the rename), and a follower's look at the file
//! (`policy_elector/file/follow`), and with the `kubernetes` feature a
//! holder's renewal over an API server in this process behind the real
//! client (`policy_elector/kubernetes/renew`: the GET, the PUT, JSON both
//! ways, no network). The renewals and looks run once per renew period in
//! production; the fence read runs per decision. The last group is the
//! decision beside the election
//! (`policy_elector/decide_beside/<alone|elected|alone_again>/<n>`): the
//! decision bench's least-load cell with no election thread, with one
//! renewing a lease file at the design's settings, and with none again, the
//! same binary's control. Not a pull-request gate: the rows are I/O bound and
//! read for their order of magnitude.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a lease that fails to open is a bug the unit tests catch"
)]

mod common;

use std::{hint::black_box, sync::mpsc, thread, time::Instant};

use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    engine::{Constraints, PoolFacts, RequestFacts, Scratch},
    service::elector::{
        ConditionalWriteOutcome, ElectionSettings, Elector, FileLease, LeaseBackend, LeaseRecord,
        MemoryLease, TickOutcome,
    },
};

/// A holder whose fence is open, as a leader's is. It takes a lease that a
/// leader released, which a candidate takes and leads at once: a take of a
/// lease that no record held leads one renew deadline later by the clock.
fn holder(mut backend: Box<dyn LeaseBackend>, identity: &str) -> Elector {
    let released = LeaseRecord {
        holder: String::new(),
        term: 1,
        acquired_at_micros: 1,
        renewed_at_micros: 1,
        duration_ms: 0,
    };
    assert_eq!(
        backend.write(None, &released).expect("the release"),
        ConditionalWriteOutcome::Done
    );
    let mut elector =
        Elector::new(backend, identity, ElectionSettings::default()).expect("valid settings");
    let now = Instant::now();
    let take = elector.tick(now).expect("the take");
    assert!(matches!(take, TickOutcome::Acquired { .. }), "{take:?}");
    assert!(elector.fence().leader_term(now).is_some());
    elector
}

fn fence(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_elector/fence");
    let elector = holder(Box::new(MemoryLease::new()), "a@host:1");
    let fence = elector.fence();
    group.bench_function("leader_term", |b| {
        b.iter(|| fence.leader_term(black_box(Instant::now())));
    });
    group.finish();
}

fn memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_elector/memory");
    let mut elector = holder(Box::new(MemoryLease::new()), "a@host:1");
    group.bench_function("renew", |b| {
        b.iter(|| elector.tick(black_box(Instant::now())).expect("renewed"));
    });
    group.finish();
}

fn file(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_elector/file");
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let mut elector = holder(Box::new(FileLease::new(&path)), "a@host:1");
    group.bench_function("renew", |b| {
        b.iter(|| elector.tick(black_box(Instant::now())).expect("renewed"));
    });
    let mut follower = Elector::new(
        Box::new(FileLease::new(&path)),
        "b@host:2",
        ElectionSettings::default(),
    )
    .expect("valid settings");
    let t0 = Instant::now();
    group.bench_function("follow", |b| {
        b.iter(|| follower.tick(black_box(t0)).expect("following"));
    });
    group.finish();
}

/// A holder renewing its lease file every renew period on a thread of its
/// own, the way the controller's election thread steps its elector; stopped
/// and joined when dropped.
struct Renewing {
    stop: mpsc::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
    _dir: tempfile::TempDir,
}

impl Renewing {
    fn start() -> Self {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut elector = holder(
            Box::new(FileLease::new(dir.path().join("lease"))),
            "a@host:1",
        );
        let period = ElectionSettings::default().renew_period;
        let (stop, stopped) = mpsc::channel();
        let thread = thread::spawn(move || {
            while stopped.recv_timeout(period) == Err(mpsc::RecvTimeoutError::Timeout) {
                elector.tick(Instant::now()).expect("renewed");
            }
        });
        Self {
            stop,
            thread: Some(thread),
            _dir: dir,
        }
    }
}

impl Drop for Renewing {
    fn drop(&mut self) {
        // A thread that is gone has nothing to stop.
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The decision beside the election: the election is one thread that wakes
/// once a renew period, and no read of the decision path is its. The three
/// arms run in one process, pinned by the caller to one core, so the thread
/// renews on the core the decisions run on.
fn decide_beside(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_elector/decide_beside");
    let engine = engine_of("least_load");
    for arm in ["alone", "elected", "alone_again"] {
        let _renewing = (arm == "elected").then(Renewing::start);
        for n in [8usize, 64, 512] {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            group.bench_with_input(BenchmarkId::new(arm, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        model: "m",
                        prompt_tokens: 512,
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    };
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
    }
    group.finish();
}

#[cfg(feature = "kubernetes")]
mod kubernetes {
    use std::{convert::Infallible, hint::black_box, sync::Arc, time::Instant};

    use criterion::Criterion;
    use http::{Method, Request, Response, StatusCode};
    use k8s_openapi::api::coordination::v1::Lease;
    use kube::{client::Body, core::Status, Client};
    use parking_lot::Mutex;
    use smg_policy_controller::service::elector::KubernetesLease;

    /// One `Lease` object and its resource version: a GET answers it, a
    /// POST creates it, a PUT replaces it under the version it names.
    #[derive(Default)]
    struct Server {
        state: Mutex<(Option<Lease>, u64)>,
    }

    impl Server {
        async fn answer(&self, request: Request<Body>) -> Response<Body> {
            let (parts, body) = request.into_parts();
            let bytes = body.collect_bytes().await.expect("the request body");
            let mut state = self.state.lock();
            match parts.method {
                Method::GET => match &state.0 {
                    Some(object) => json(StatusCode::OK, object),
                    None => failure(StatusCode::NOT_FOUND, "NotFound"),
                },
                Method::POST | Method::PUT => {
                    let mut object: Lease = serde_json::from_slice(&bytes).expect("a Lease");
                    let current = state
                        .0
                        .as_ref()
                        .and_then(|o| o.metadata.resource_version.clone());
                    if parts.method == Method::PUT && object.metadata.resource_version != current {
                        return failure(StatusCode::CONFLICT, "Conflict");
                    }
                    state.1 += 1;
                    object.metadata.resource_version = Some(state.1.to_string());
                    state.0 = Some(object.clone());
                    json(StatusCode::OK, &object)
                }
                _ => failure(StatusCode::METHOD_NOT_ALLOWED, "MethodNotAllowed"),
            }
        }
    }

    fn json<T: serde::Serialize>(status: StatusCode, value: &T) -> Response<Body> {
        Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(value).expect("json")))
            .expect("a response")
    }

    fn failure(status: StatusCode, reason: &str) -> Response<Body> {
        json(
            status,
            &Status::failure(reason, reason).with_code(status.as_u16()),
        )
    }

    pub fn renew(c: &mut Criterion) {
        let mut group = c.benchmark_group("policy_elector/kubernetes");
        let runtime = tokio::runtime::Runtime::new().expect("a runtime");
        let server = Arc::new(Server::default());
        let client = runtime.block_on(async {
            let server = Arc::clone(&server);
            let service = tower::service_fn(move |request: Request<Body>| {
                let server = Arc::clone(&server);
                async move { Ok::<_, Infallible>(server.answer(request).await) }
            });
            Client::new(service, "policy")
        });
        let backend =
            KubernetesLease::new(client, "policy", "controller", runtime.handle().clone());
        let mut elector = super::holder(Box::new(backend), "a@host:1");
        group.bench_function("renew", |b| {
            b.iter(|| elector.tick(black_box(Instant::now())).expect("renewed"));
        });
        group.finish();
    }
}

#[cfg(feature = "kubernetes")]
criterion_group!(
    benches,
    fence,
    memory,
    file,
    kubernetes::renew,
    decide_beside
);
#[cfg(not(feature = "kubernetes"))]
criterion_group!(benches, fence, memory, file, decide_beside);
criterion_main!(benches);
