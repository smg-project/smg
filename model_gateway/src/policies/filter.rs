//! Pre-scoring candidate filters for policy-based worker selection.
//!
//! Filters run between eligibility (health + circuit breaker) and policy
//! scoring: each filter answers keep/drop for one candidate worker given the
//! per-request selection info. Filters express HARD requirements — a worker
//! that fails one may not serve the request at all; preferences (ordering,
//! weighting) belong in policies.
//!
//! The chain lives on the `PolicyRegistry` and applies to every policy-based
//! selection path uniformly. An empty chain adds no cost. When a non-empty
//! candidate set is filtered down to nothing, selection reports
//! "all filtered" — an unavailability outcome (503), deliberately distinct
//! from an empty pool (the model may not exist → 404 on some paths) and from
//! backpressure (429): retrying elsewhere just bounces, and a 404 would lie
//! to the client about the model's existence.

use std::{fmt::Debug, sync::Arc};

use super::SelectWorkerInfo;
use crate::{config::types::RouterConfig, worker::Worker};

/// A hard candidate requirement applied before policy scoring.
pub trait WorkerFilter: Send + Sync + Debug {
    /// Filter name for logs and debugging.
    fn name(&self) -> &'static str;

    /// Whether `worker` may remain in the candidate set for this request.
    fn keep(&self, worker: &dyn Worker, info: &SelectWorkerInfo) -> bool;
}

/// Label filter driven by a request header.
///
/// The header value is parsed as comma-separated `key=value` pairs; only
/// workers whose spec labels contain ALL pairs survive. Requests without the
/// header keep every candidate, so the filter is inert until a client asks
/// for narrowing. Malformed pairs (no `=`, empty key) are ignored.
///
/// Worker labels already flow end to end (k8s discovery selectors, the
/// `/workers` management API), so this enables tenant pinning, canary
/// cohorts, and hardware classes with no new worker-side surface.
#[derive(Debug)]
pub struct LabelHeaderFilter {
    /// Stored lowercase; `HeaderMap` lookups are case-insensitive by name.
    header_name: String,
}

impl LabelHeaderFilter {
    pub fn new(header_name: impl AsRef<str>) -> Self {
        Self {
            header_name: header_name.as_ref().to_ascii_lowercase(),
        }
    }
}

impl WorkerFilter for LabelHeaderFilter {
    fn name(&self) -> &'static str {
        "label_header"
    }

    fn keep(&self, worker: &dyn Worker, info: &SelectWorkerInfo) -> bool {
        let Some(required) = info
            .headers
            .and_then(|headers| headers.get(&self.header_name))
            .and_then(|value| value.to_str().ok())
        else {
            return true;
        };
        let labels = &worker.metadata().spec.labels;
        label_pairs(required)
            .all(|(key, value)| labels.get(key).is_some_and(|actual| actual == value))
    }
}

/// Build the configured filter chain: currently the label header filter when
/// `worker_filter_header` is set (validated as a header name at config
/// time). Empty when nothing is configured — filtering costs nothing.
pub fn worker_filters_from_config(config: &RouterConfig) -> Vec<Arc<dyn WorkerFilter>> {
    let mut filters: Vec<Arc<dyn WorkerFilter>> = Vec::new();
    if let Some(header) = config.worker_filter_header.as_deref() {
        let header = header.trim();
        if !header.is_empty() {
            filters.push(Arc::new(LabelHeaderFilter::new(header)));
        }
    }
    filters
}

/// Parse `k=v,k2=v2` into pairs, skipping malformed entries.
fn label_pairs(value: &str) -> impl Iterator<Item = (&str, &str)> {
    value.split(',').filter_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        let (key, value) = (key.trim(), value.trim());
        (!key.is_empty()).then_some((key, value))
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn labeled_worker(labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        let mut map = HashMap::new();
        for (k, v) in labels {
            map.insert((*k).to_string(), (*v).to_string());
        }
        Arc::new(
            BasicWorkerBuilder::new("http://w:8000")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .labels(map)
                .build(),
        )
    }

    fn info_with_headers(headers: &http::HeaderMap) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            headers: Some(headers),
            ..Default::default()
        }
    }

    #[test]
    fn missing_header_keeps_everything() {
        let filter = LabelHeaderFilter::new("X-SMG-Worker-Labels");
        let worker = labeled_worker(&[]);
        let headers = http::HeaderMap::new();
        assert!(filter.keep(worker.as_ref(), &info_with_headers(&headers)));
        assert!(filter.keep(worker.as_ref(), &SelectWorkerInfo::default()));
    }

    #[test]
    fn requires_all_pairs() {
        let filter = LabelHeaderFilter::new("X-SMG-Worker-Labels");
        let full_match = labeled_worker(&[("tenant", "acme"), ("tier", "gpu-h100"), ("x", "y")]);
        let partial = labeled_worker(&[("tenant", "acme")]);
        let wrong_value = labeled_worker(&[("tenant", "acme"), ("tier", "cpu")]);
        let unlabeled = labeled_worker(&[]);

        let mut headers = http::HeaderMap::new();
        headers.insert(
            "x-smg-worker-labels",
            "tenant=acme, tier=gpu-h100".parse().unwrap(),
        );
        let info = info_with_headers(&headers);

        assert!(filter.keep(full_match.as_ref(), &info));
        assert!(!filter.keep(partial.as_ref(), &info));
        assert!(!filter.keep(wrong_value.as_ref(), &info));
        assert!(!filter.keep(unlabeled.as_ref(), &info));
    }

    #[test]
    fn malformed_pairs_are_ignored() {
        let filter = LabelHeaderFilter::new("X-SMG-Worker-Labels");
        let worker = labeled_worker(&[("tenant", "acme")]);

        let mut headers = http::HeaderMap::new();
        // Bare token and empty key are skipped; the valid pair still applies.
        headers.insert(
            "x-smg-worker-labels",
            "garbage, =nokey, tenant=acme".parse().unwrap(),
        );
        assert!(filter.keep(worker.as_ref(), &info_with_headers(&headers)));

        // Only malformed entries → no requirements → keep.
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-worker-labels", "garbage".parse().unwrap());
        assert!(filter.keep(worker.as_ref(), &info_with_headers(&headers)));
    }

    #[test]
    fn empty_value_matches_empty_label() {
        let filter = LabelHeaderFilter::new("X-SMG-Worker-Labels");
        let worker = labeled_worker(&[("flag", "")]);
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-worker-labels", "flag=".parse().unwrap());
        assert!(filter.keep(worker.as_ref(), &info_with_headers(&headers)));
    }
}
