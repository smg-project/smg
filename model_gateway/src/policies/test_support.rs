//! What the policy tests line their workers up with, and the routing-key
//! header they send.

use std::sync::Arc;

use super::LoadBalancingPolicy;
use crate::worker::{load_state::TestFeed, test_support::worker_of, Worker, WorkerType};

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

/// The `a`/`b` worker pair under `policy` with a test feed attached; the
/// test publishes the pair's reports through the feed.
pub(crate) fn fed_pair<P: LoadBalancingPolicy>(policy: P) -> (P, TestFeed, Vec<Arc<dyn Worker>>) {
    let feed = TestFeed::attached_to(&policy);
    let workers = make_workers(&["http://a:8000", "http://b:8000"]);
    (policy, feed, workers)
}
