//! What the policy tests line their workers up with, the routing-key
//! header they send and the selection info that carries it.

use std::sync::Arc;

use smg_policy_controller::PolicySpec;

use super::{engine_host::Decider, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::{load_state::TestFeed, test_support::worker_of, Worker, WorkerType};

/// A decider over a document the policy configuration cannot express: the
/// `default` profile, bound to every model's regular placement, scores
/// `default`; the two pair sides, bound to every model's pair phases, score
/// `prefill` and `decode`. A side with no scorer picks round-robin.
pub(crate) fn decider_over(default: &[&str], prefill: &[&str], decode: &[&str]) -> Decider {
    let profile = |filters: &str, score: &[&str]| {
        let pick = if score.is_empty() {
            "round-robin"
        } else {
            "best"
        };
        let score = serde_json::to_string(score).expect("names");
        format!(r#"{{ "filter": {filters}, "score": {score}, "pick": "{pick}" }}"#)
    };
    let text = format!(
        r#"{{
          "apiVersion": "smg.policy/v1",
          "profiles": {{
            "default": {},
            "prefill": {},
            "decode": {}
          }},
          "bindings": [
            {{ "match": {{ "model": "*", "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
              "profiles": {{ "prefill": "prefill", "decode": "decode" }} }},
            {{ "match": {{ "model": "*" }}, "profile": "default" }}
          ]
        }}"#,
        profile(r#"["healthy"]"#, default),
        profile(r#"["healthy", "has-capacity"]"#, prefill),
        profile(r#"["healthy", "has-room"]"#, decode),
    );
    let spec: PolicySpec = serde_json::from_str(&text).expect("a document");
    Decider::from_spec(&spec).expect("validates")
}

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
