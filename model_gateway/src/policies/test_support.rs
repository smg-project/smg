//! What the policy tests line their workers up with, the routing-key
//! header they send and the selection info that carries it.

use std::sync::Arc;

use kv_index::{compute_content_hash, SequenceHash, StoredBlock};

use super::{normalize_model_key, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo};
use crate::worker::{
    load_state::TestFeed, test_support::worker_of, KvEventMonitor, KvIndex, Worker, WorkerBlocks,
    WorkerType,
};

/// A healthy worker (health checks disabled) at `url`.
pub(crate) fn make_worker(url: &str) -> Arc<dyn Worker> {
    worker_of(url, WorkerType::Regular)
}

/// Healthy workers (health checks disabled) for the given URLs.
pub(crate) fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter().map(|url| make_worker(url)).collect()
}

/// The request headers that carry `key` as the routing key.
pub(crate) fn headers_with_routing_key(key: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", key.parse().unwrap());
    headers
}

/// The selection info of a request carrying `headers` and nothing else.
pub(crate) fn with_headers(headers: &http::HeaderMap) -> SelectWorkerInfo<'_> {
    SelectWorkerInfo {
        headers: Some(headers),
        ..Default::default()
    }
}

/// The `a`/`b` worker pair under `policy` with a test feed attached; the
/// test publishes the pair's reports through the feed.
pub(crate) fn fed_pair<P: LoadBalancingPolicy>(policy: P) -> (P, TestFeed, Vec<Arc<dyn Worker>>) {
    let feed = TestFeed::attached_to(&policy);
    let workers = make_workers(&["http://a:8000", "http://b:8000"]);
    (policy, feed, workers)
}

/// A prompt of four blocks of four tokens that `holder` holds in its
/// model's event index, wired into `policies`: the prompt. The holder has
/// a request in flight, so a placement by load alone never lands on it:
/// only the prompt's affinity does.
pub(crate) fn hold_prompt(policies: &PolicyRegistry, holder: &dyn Worker) -> Vec<u32> {
    holder.increment_load();
    let tokens: Vec<u32> = (1..=16).collect();
    let indexer = Arc::new(KvIndex::chain());
    let id = indexer.intern_worker(holder.url()).expect("interned");
    let blocks: Vec<StoredBlock> = tokens
        .chunks(4)
        .enumerate()
        .map(|(i, chunk)| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: compute_content_hash(chunk),
        })
        .collect();
    indexer
        .apply_stored(id, &blocks, None, &mut WorkerBlocks::default())
        .expect("applied");
    let key = normalize_model_key(holder.model_id()).to_string();
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert(key.clone(), indexer);
    monitor.set_block_size(&key, 4);
    policies.set_kv_event_monitor(Some(monitor));
    tokens
}
