//! The warm-up slice of the classic cache-aware policy over real workers on
//! their own clocks: a worker that holds nothing gets the slice for its
//! warm-up window alone, the window starts again when its index is emptied
//! underneath, and an emptied worker that regrew past the thinness ratio is
//! not young again. One test in its own binary, as the warm-up budgets are
//! process-wide and the first call sets them.

use std::{sync::Arc, thread, time::Duration};

use kv_index::{compute_content_hash, SequenceHash, StoredBlock};
use openai_protocol::worker::HealthCheckConfig;
use smg::{
    policies::{
        kv_index::{
            event_monitor::KvEventMonitor,
            index_backend::{KvIndex, WorkerBlocks},
        },
        CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo,
    },
    routers::grpc::backend_client::BasicWorkerBuilder,
};
use smg_worker::{
    health::liveness,
    identity::basic::{Worker, WorkerType},
};

#[test]
fn a_worker_that_holds_nothing_gets_the_slice_for_its_window_alone() {
    // A window of one second, and every thin-overlap request offered to the
    // slice, so 64 requests decide between two tied candidates.
    liveness::configure_from(liveness::Budgets {
        warmup_secs: Duration::from_secs(1),
        warmup_share: 1.0,
        ..liveness::budgets()
    });
    let urls = [
        "http://w0:8000",
        "http://w1:8000",
        "http://w2:8000",
        "http://w3:8000",
    ];
    let workers: Vec<Arc<dyn Worker>> = urls
        .iter()
        .map(|url| {
            Arc::new(
                BasicWorkerBuilder::new(*url)
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        block_size: 4,
        ..Default::default()
    });
    policy.init_workers(&workers);
    // w0 and w1 hold a cache worth catching up to (1,100 blocks each, the
    // pool's level), w2 never held a block and w3 holds four.
    let index = Arc::new(KvIndex::chain());
    let ids: Vec<u32> = urls
        .iter()
        .map(|url| index.intern_worker(url).unwrap())
        .collect();
    let store = |worker: u32, tokens: &[u32], seq: u64, held: &mut WorkerBlocks| {
        let blocks: Vec<StoredBlock> = tokens
            .chunks(4)
            .enumerate()
            .map(|(i, chunk)| StoredBlock {
                seq_hash: SequenceHash(seq + i as u64),
                content_hash: compute_content_hash(chunk),
            })
            .collect();
        index.apply_stored(worker, &blocks, None, held).unwrap();
    };
    let head: Vec<u32> = (1..=4_400).collect();
    let other: Vec<u32> = (10_001..=14_400).collect();
    let few: Vec<u32> = (50_001..=50_016).collect();
    store(ids[0], &head, 1, &mut WorkerBlocks::default());
    store(ids[1], &other, 10_000, &mut WorkerBlocks::default());
    let mut held = WorkerBlocks::default();
    store(ids[3], &few, 50_000, &mut held);
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.set_index("unknown", Arc::clone(&index));
    policy.set_kv_event_monitor(Some(monitor));
    // w0's first block followed by eleven novel ones: a thin overlap.
    let mut request: Vec<u32> = (1..=4).collect();
    request.extend(90_000..90_044);
    let info = SelectWorkerInfo {
        tokens: Some(&request),
        ..Default::default()
    };
    let picks = || -> Vec<usize> {
        (0..64)
            .map(|_| policy.select_worker(&workers, &info).unwrap())
            .collect()
    };
    // Inside its window the worker that never held a block gets its slice.
    let young = picks();
    assert!(young.contains(&2), "{young:?}");
    // Both joined over a second ago now, and w3's blocks go: its window
    // starts again at the emptying, the window of w2 does not.
    thread::sleep(Duration::from_millis(1_050));
    let gone: Vec<SequenceHash> = (50_000..50_004).map(SequenceHash).collect();
    index.apply_removed(ids[3], &gone, &mut held);
    let old = picks();
    assert!(old.contains(&3), "the emptied worker warms again: {old:?}");
    assert!(
        !old.contains(&2),
        "it never held a cache and is past its window: {old:?}"
    );
    // Regrown past the thinness ratio inside its new window (600 blocks of
    // a level of 1,100) the emptied worker is not thin, and it is not young
    // either: the young rule reads its admission, not the emptying.
    let back: Vec<u32> = (60_001..=62_400).collect();
    store(ids[3], &back, 60_000, &mut held);
    let regrown = picks();
    assert!(
        !regrown.contains(&3),
        "regrown past the ratio it is not young again: {regrown:?}"
    );
}
