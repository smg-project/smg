//! The election's costs: the fence read a shard thread pays before every
//! decision (`policy_elector/fence/leader_term`), a holder's renewal over
//! the in-process record and over the lease file
//! (`policy_elector/<memory|file>/renew`: the read, the lock, the write and
//! the rename), and a follower's look at the file
//! (`policy_elector/file/follow`). The renewals and looks run once per renew
//! period in production; the fence read runs per decision. Not a
//! pull-request gate: the rows are I/O bound and read for their order of
//! magnitude.
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

criterion_group!(benches, fence, memory, file);
criterion_main!(benches);
