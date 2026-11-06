//! What one load record pushed on a worker's KV-event stream costs the
//! worker monitor in allocation calls, pinned at today's count, so the next
//! change that builds a report, a key or a map per record fails a test rather
//! than moving the gateway's idle allocation rate inside its noise: the
//! records arrive per scheduler step from every engine, request or no
//! request. A separate executable, so it can declare the counting allocator
//! the pin needs.

#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a record allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
    time::{Duration, Instant},
};

use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg::{
    config::PolicyConfig,
    policies::PolicyRegistry,
    worker::{
        monitor::WorkerMonitor, registry::WorkerRegistry, BasicWorkerBuilder, ConnectionMode,
        Worker, WorkerType,
    },
};
use smg_grpc_client::common_proto::EngineLoad;

thread_local! {
    /// Allocation and reallocation calls this thread made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let out = work();
    (out, CALLS.with(Cell::get) - before)
}

/// A monitor with nothing spawned, as the gateway holds it before the event
/// loop starts; the pin drives `apply_pushed_load` directly, as the KV-event
/// subscription does for every record its stream carries.
fn monitor() -> Arc<WorkerMonitor> {
    let registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    Arc::new(WorkerMonitor::new(
        registry,
        policy_registry,
        1,
        false,
        false,
    ))
}

fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// A heartbeat-shaped record: the core figures with the telemetry an engine
/// reports beside them, so the report carries absolute token counts and the
/// DP-rank token cache takes it, as it takes a poll.
fn record(running: u32, sample: u64) -> EngineLoad {
    EngineLoad {
        running_requests: running,
        waiting_requests: 1,
        token_usage: 0.25,
        gen_throughput: 100.0,
        max_running_requests: 64,
        sample,
        cache_hit_rate: Some(0.5),
        num_used_tokens: Some(1024),
        max_total_num_tokens: Some(65_536),
        ..Default::default()
    }
}

/// A record for a rank already on file, the steady state of every stream
/// (each record after a rank's first), allocates for two things: the
/// `smg_engine_*` gauges, which take the rank label's text and one label
/// vector per gauge for every rank of the worker (five gauges, plus the
/// capacity gauge when the record carries the capacity), and the DP-rank
/// token cache's map of the report's ranks. Nothing per record for the
/// report itself: the rank's entry is replaced in place, and the poll-shaped
/// copy that the shared snapshot takes is built once per publish window.
#[tokio::test(flavor = "current_thread")]
async fn a_record_for_a_rank_on_file_allocates_for_its_gauges_and_the_rank_cache_alone() {
    /// The rank label's text and six gauge keys, per rank of the report.
    const GAUGES_PER_RANK: usize = 7;
    /// The DP-rank token cache's map of the report's ranks, per record.
    const RANK_CACHE: usize = 1;

    let monitor = monitor();
    let t0 = Instant::now();

    let single = worker("http://engine-a:8000");
    // The worker's first record makes its entries and starts the publisher.
    monitor.apply_pushed_load(&single, 0, &record(2, 1), t0);
    let ((), one_rank) = allocations_during(|| {
        monitor.apply_pushed_load(&single, 0, &record(3, 2), t0 + Duration::from_millis(1));
    });

    let two_ranks = worker("http://engine-b:8000");
    monitor.apply_pushed_load(&two_ranks, 0, &record(2, 1), t0);
    let ((), second_rank) = allocations_during(|| {
        monitor.apply_pushed_load(&two_ranks, 1, &record(5, 2), t0 + Duration::from_millis(1));
    });
    let ((), both_on_file) = allocations_during(|| {
        monitor.apply_pushed_load(&two_ranks, 0, &record(4, 3), t0 + Duration::from_millis(2));
    });

    eprintln!(
        "pushed record: {one_rank} allocation calls for the one rank on file, \
         {both_on_file} with two ranks on file, {second_rank} for a worker's second rank"
    );
    assert_eq!(
        one_rank,
        GAUGES_PER_RANK + RANK_CACHE,
        "a record for the one rank of a worker on file: the gauges' seven and the rank cache's one"
    );
    assert_eq!(
        both_on_file,
        2 * GAUGES_PER_RANK + RANK_CACHE,
        "two ranks on file: the gauges' per-rank train twice and the rank cache once"
    );
}
