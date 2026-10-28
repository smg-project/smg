//! What the policy tests line their workers up with.

use std::sync::Arc;

use super::LoadBalancingPolicy;
use crate::worker::{
    load_state::TestFeed, test_support::no_health_check, BasicWorkerBuilder, Worker, WorkerType,
};

/// A healthy worker (health checks disabled) at `url`.
pub(crate) fn make_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    )
}

/// Healthy workers (health checks disabled) for the given URLs.
pub(crate) fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter().map(|url| make_worker(url)).collect()
}

/// The `a`/`b` worker pair under `policy` with a test feed attached; the
/// test publishes the pair's reports through the feed.
pub(crate) fn fed_pair<P: LoadBalancingPolicy>(policy: P) -> (P, TestFeed, Vec<Arc<dyn Worker>>) {
    let feed = TestFeed::attached_to(&policy);
    let workers = make_workers(&["http://a:8000", "http://b:8000"]);
    (policy, feed, workers)
}
