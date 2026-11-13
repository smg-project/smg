//! The election's costs: the fence read a shard thread pays before every
//! decision (`policy_elector/fence/leader_term`), a holder's renewal over
//! the in-process record and over the lease file
//! (`policy_elector/<memory|file>/renew`: the read, the lock, the write and
//! the rename), and a follower's look at the file
//! (`policy_elector/file/follow`), and with the `kubernetes` feature a
//! holder's renewal over an API server in this process behind the real
//! client (`policy_elector/kubernetes/renew`: the GET, the PUT, JSON both
//! ways, no network). The renewals and looks run once per renew period in
//! production; the fence read runs per decision. Not a pull-request gate:
//! the rows are I/O bound and read for their order of magnitude.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a lease that fails to open is a bug the unit tests catch"
)]

use std::{hint::black_box, time::Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use smg_policy_controller::elector::{
    ElectionSettings, Elector, FileLease, LeaseBackend, MemoryLease, Tick,
};

fn holder(backend: Box<dyn LeaseBackend>, identity: &str) -> Elector {
    let mut elector =
        Elector::new(backend, identity, ElectionSettings::default()).expect("valid settings");
    assert_eq!(
        elector.tick(Instant::now()).expect("the take"),
        Tick::Acquired { term: 1 }
    );
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

#[cfg(feature = "kubernetes")]
mod kubernetes {
    use std::{convert::Infallible, hint::black_box, sync::Arc, time::Instant};

    use criterion::Criterion;
    use http::{Method, Request, Response, StatusCode};
    use k8s_openapi::api::coordination::v1::Lease;
    use kube::{client::Body, core::Status, Client};
    use parking_lot::Mutex;
    use smg_policy_controller::elector::KubernetesLease;

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
criterion_group!(benches, fence, memory, file, kubernetes::renew);
#[cfg(not(feature = "kubernetes"))]
criterion_group!(benches, fence, memory, file);
criterion_main!(benches);
