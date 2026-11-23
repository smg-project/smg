//! The cost of the sent-prefix predictor's two calls,
//! `PrefixPredictor::deepest` and `PrefixPredictor::record`, on a predictor at
//! its bound: a write there takes out as many old blocks as it adds. One
//! predictor holds 4,096 requests of 64 blocks, sent to 8 workers, behind a
//! head of 8 blocks they all share. The `cold` turns run on another, which
//! holds requests that share nothing.
//!
//! - `read/sent_before/64`: the read of a request sent before.
//! - `read/cold/64`: the read of a request none of whose blocks was sent.
//! - `turn/sent_before/64`: the read and the write one decision makes for
//!   a request sent before. The write adds nothing.
//! - `turn/grown/64`: the same for a request whose last four blocks are
//!   new. The write adds the four and takes four older ones out.
//! - `turn/cold/64` and `turn/cold/256`: the same for a request never
//!   sent, of that many blocks.
//!
//! Ids: `policy_predictor/<call>/<request>/<blocks>`. The requests go
//! round in the order they were sent, so the oldest path is always the
//! request's own or, on a cold turn, the oldest request's. A read folds
//! the request's keys into its caller's buffer and then reads by them, as
//! a decision does, and the write behind it takes the same keys at a new
//! tick.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg_policy_controller::{engine::WorkerId, tracking::PrefixPredictor};

/// The namespace of every request.
const SEED: u64 = 0;
/// The requests the first predictor holds.
const REQUESTS: usize = 4_096;
/// The workers they were sent to, in turn.
const WORKERS: u64 = 8;
/// The blocks of a request, the shared head among them.
const BLOCKS: u64 = 64;
/// The blocks of the head every request shares.
const HEAD: u64 = 8;
/// The blocks a turn adds.
const GROWN: u64 = 4;
/// The blocks the second predictor holds for each length of a cold turn.
const COLD_BLOCKS: u64 = 131_072;

fn mix(mut bits: u64) -> u64 {
    bits ^= bits >> 30;
    bits = bits.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    bits ^= bits >> 27;
    bits = bits.wrapping_mul(0x94D0_49BB_1331_11EB);
    bits ^ (bits >> 31)
}

/// The hash of a block: one for each request, turn and place.
fn block(request: u64, turn: u64, at: u64) -> u64 {
    mix(mix(mix(request) ^ turn) ^ at)
}

/// A request of the first predictor: the shared head, its own blocks, and
/// the last four as its turn `turn` wrote them.
fn request(of: usize, turn: u64) -> Vec<u64> {
    let of = of as u64;
    let head = (0..HEAD).map(|at| block(u64::MAX, 0, at));
    let own = (HEAD..BLOCKS - GROWN).map(|at| block(of, 0, at));
    let grown = (BLOCKS - GROWN..BLOCKS).map(|at| block(of, turn, at));
    head.chain(own).chain(grown).collect()
}

/// A request that shares no block with any other.
fn apart(of: u64, blocks: u64) -> Vec<u64> {
    (0..blocks).map(|at| block(of, u64::MAX, at)).collect()
}

fn worker(of: usize) -> WorkerId {
    WorkerId(of as u64 % WORKERS)
}

/// A read as a decision makes it: the request's keys folded into the
/// caller's buffer, then the read by them.
fn read(held: &PrefixPredictor, keys: &mut Vec<u64>, request: &[u64]) -> u32 {
    PrefixPredictor::fold_into(keys, SEED, request);
    held.deepest(keys).blocks()
}

/// The write behind a read, by the keys the read folded, at a new tick.
fn write(held: &mut PrefixPredictor, keys: &[u64], worker: WorkerId, tick: &mut u16) {
    *tick = tick.wrapping_add(1);
    held.record(keys, worker, *tick);
}

/// The predictor that holds every request's first turn, at its bound.
fn sent_before() -> PrefixPredictor {
    let mut predictor = PrefixPredictor::new(usize::MAX);
    let (mut keys, mut tick) = (Vec::new(), 0);
    for of in 0..REQUESTS {
        PrefixPredictor::fold_into(&mut keys, SEED, &request(of, 1));
        write(&mut predictor, &keys, worker(of), &mut tick);
    }
    predictor.set_bound(predictor.len());
    predictor
}

/// A predictor at its bound that holds requests of `blocks` blocks which
/// share nothing; and the number the next such request takes.
fn cold(blocks: u64) -> (PrefixPredictor, u64) {
    let held = COLD_BLOCKS / blocks;
    let mut predictor = PrefixPredictor::new(usize::MAX);
    let (mut keys, mut tick) = (Vec::new(), 0);
    for of in 0..held {
        PrefixPredictor::fold_into(&mut keys, SEED, &apart(of, blocks));
        write(&mut predictor, &keys, WorkerId(of % WORKERS), &mut tick);
    }
    predictor.set_bound(predictor.len());
    (predictor, held)
}

fn predictor(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_predictor");
    let requests: Vec<Vec<u64>> = (0..REQUESTS).map(|of| request(of, 1)).collect();
    let never: Vec<Vec<u64>> = (0..REQUESTS)
        .map(|of| apart(u64::MAX - of as u64, BLOCKS))
        .collect();
    let mut held = sent_before();
    let mut next = 0;
    let (mut keys, mut tick) = (Vec::new(), 0);

    group.bench_function("read/sent_before/64", |b| {
        b.iter(|| {
            next = (next + 1) % REQUESTS;
            black_box(read(&held, &mut keys, black_box(&requests[next])))
        });
    });
    group.bench_function("read/cold/64", |b| {
        b.iter(|| {
            next = (next + 1) % REQUESTS;
            black_box(read(&held, &mut keys, black_box(&never[next])))
        });
    });
    next = REQUESTS - 1;
    group.bench_function("turn/sent_before/64", |b| {
        b.iter(|| {
            next = (next + 1) % REQUESTS;
            let depth = read(&held, &mut keys, black_box(&requests[next]));
            write(&mut held, &keys, worker(next), &mut tick);
            black_box(depth)
        });
    });
    let mut turn = 2;
    group.bench_function("turn/grown/64", |b| {
        b.iter_batched(
            || {
                next = (next + 1) % REQUESTS;
                // A request's next turn comes once the others had theirs.
                turn += u64::from(next == 0);
                (request(next, turn), worker(next))
            },
            |(request, worker)| {
                let depth = read(&held, &mut keys, &request);
                write(&mut held, &keys, worker, &mut tick);
                black_box(depth)
            },
            BatchSize::SmallInput,
        );
    });
    for blocks in [64, 256] {
        let (mut held, mut of) = cold(blocks);
        group.bench_function(format!("turn/cold/{blocks}"), |b| {
            b.iter_batched(
                || {
                    of += 1;
                    (apart(of, blocks), WorkerId(of % WORKERS))
                },
                |(request, worker)| {
                    let depth = read(&held, &mut keys, &request);
                    write(&mut held, &keys, worker, &mut tick);
                    black_box(depth)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, predictor);
criterion_main!(benches);
