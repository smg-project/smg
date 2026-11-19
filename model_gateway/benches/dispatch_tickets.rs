//! The dispatch tickets under many threads.
//!
//! One streaming request's tickets on ONE worker for their whole life (the
//! booking, the first response, the drop), from 1, 8 and 32 threads at once.
//! No stream and no network: the cell reads what the notes and the handles
//! cost when requests land on one worker together. A round is one life on
//! each thread, so the throughput is lives a second over all threads. The
//! threads wait at a barrier until the last of them is spawned. Each stamps
//! the start of its first round and the end of its last, and the timed span
//! runs from the earliest start to the latest end: it holds every round and
//! no spawn.
//!
//! Run: `cargo bench -p smg --bench dispatch_tickets`

use std::{
    hint::black_box,
    panic::resume_unwind,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg::routers::grpc::{backend_client::BasicWorkerBuilder, dispatch_ticket_life};
use smg_worker::identity::basic::Worker;

/// The prompt each request books. The backlog's arithmetic is not what the
/// cell measures, so any size that is not empty serves.
const PROMPT_TOKENS: u64 = 4_096;

fn bench_ticket_life(c: &mut Criterion) {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://worker:8000").build());
    let mut group = c.benchmark_group("dispatch_tickets");
    for threads in [1_usize, 8, 32] {
        group.throughput(Throughput::Elements(threads as u64));
        group.bench_function(format!("one_worker/{threads}_threads"), |b| {
            b.iter_custom(|rounds| {
                let barrier = Barrier::new(threads);
                // The scope returns once every thread has ended its rounds,
                // with the earliest start and the latest end among them.
                let span = thread::scope(|scope| {
                    let stamped: Vec<_> = (0..threads)
                        .map(|_| {
                            scope.spawn(|| {
                                barrier.wait();
                                let started = Instant::now();
                                for _ in 0..rounds {
                                    dispatch_ticket_life(black_box(&worker), PROMPT_TOKENS);
                                }
                                (started, Instant::now())
                            })
                        })
                        .collect();
                    stamped
                        .into_iter()
                        .map(|thread| thread.join().unwrap_or_else(|panic| resume_unwind(panic)))
                        .reduce(|(first, last), (started, ended)| {
                            (first.min(started), last.max(ended))
                        })
                });
                span.map_or(Duration::ZERO, |(first, last)| last.duration_since(first))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ticket_life);
criterion_main!(benches);
