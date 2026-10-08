//! Opt-in routing evidence, scoped to one pipeline run across async retries.

use std::{
    cell::RefCell,
    future::Future,
    sync::{Arc, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};

use crate::worker::Worker;

#[derive(Default)]
struct Capture {
    prediction: Value,
    selections: Vec<Value>,
    truncated: bool,
    gates: Vec<Value>,
    candidates: Vec<Value>,
    candidates_complete: bool,
    observation_started_ns: u64,
    observation_finished_ns: u64,
    scores: Vec<Value>,
}

impl Capture {
    fn observe_candidates(&mut self, workers: &[Arc<dyn Worker>]) {
        self.observation_started_ns = timestamp_ns();
        self.candidates = workers
            .iter()
            .take(32)
            .map(|worker| {
                json!({
                    "worker": worker.url(), "load": worker.load(), "healthy": worker.is_healthy(),
                    "overloaded": worker.is_overloaded(), "registry_revision": worker.revision(),
                    "backend_cache_epoch": null,
                })
            })
            .collect();
        self.candidates_complete = workers.len() <= 32;
        self.observation_finished_ns = timestamp_ns();
    }
}

tokio::task_local! {
    static CAPTURE: RefCell<Capture>;
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SMG_CACHE_TRACE").is_ok_and(|v| v == "1"))
}

fn header_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1"))
}

fn timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

pub(crate) fn begin_selection(workers: &[Arc<dyn Worker>]) {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| {
            let mut capture = capture.borrow_mut();
            capture.prediction = Value::Null;
            capture.gates.clear();
            capture.scores.clear();
            capture.observe_candidates(workers);
        });
    }
}

pub(crate) async fn scope<F: Future>(future: F) -> F::Output {
    if enabled() {
        CAPTURE
            .scope(RefCell::new(Capture::default()), future)
            .await
    } else {
        future.await
    }
}

pub(crate) fn prediction(value: Value) {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| capture.borrow_mut().prediction = value);
    }
}

pub(crate) fn gate(value: Value) {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| {
            let mut capture = capture.borrow_mut();
            if capture.gates.len() < 32 {
                capture.gates.push(value);
            } else {
                capture.truncated = true;
            }
        });
    }
}

/// Record a score at its existing evaluation site, before dispatch credit.
pub(crate) fn score(value: Value) {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| {
            let mut capture = capture.borrow_mut();
            if capture.scores.len() < 64 {
                capture.scores.push(value);
            } else {
                capture.truncated = true;
            }
        });
    }
}

pub(crate) fn mark_truncated() {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| capture.borrow_mut().truncated = true);
    }
}

pub(crate) fn selection(
    policy: &str,
    origin: &str,
    workers: &[Arc<dyn Worker>],
    selected: Option<usize>,
) {
    if !enabled() {
        return;
    }
    let _ = CAPTURE.try_with(|capture| {
        let mut capture = capture.borrow_mut();
        if capture.selections.len() >= 16 {
            capture.truncated = true;
            return;
        }
        let value = json!({
            "policy": policy, "origin": origin,
            "worker": selected.map(|idx| workers[idx].url()),
            "prediction": std::mem::take(&mut capture.prediction),
            "candidates": std::mem::take(&mut capture.candidates),
            "candidates_complete": capture.candidates_complete,
            "observation_started_ns": capture.observation_started_ns,
            "observation_finished_ns": capture.observation_finished_ns,
            "snapshot_atomic": false,
            "scores": std::mem::take(&mut capture.scores),
            "load_source": "router_inflight", "eligibility_scope": "policy_candidates",
            "load_observation_phase": "before_selection",
            "gates": std::mem::take(&mut capture.gates),
        });
        capture.selections.push(value);
    });
}

pub(crate) fn dispatch(
    root_id: Option<&str>,
    attempt: u32,
    engine_ids: Vec<String>,
    engine_ids_complete: bool,
    mode: &str,
) -> Option<String> {
    if !enabled() {
        return None;
    }
    static ROUTER_EPOCH: OnceLock<String> = OnceLock::new();
    let router_epoch = ROUTER_EPOCH.get_or_init(|| uuid::Uuid::now_v7().to_string());
    CAPTURE.try_with(|capture| {
        let mut capture = capture.borrow_mut();
        let value = json!({
            "schema": 1, "router_epoch": router_epoch, "dispatch_timestamp_ns": timestamp_ns(), "root_id": root_id, "dispatch_id": uuid::Uuid::now_v7().to_string(),
            "attempt": attempt, "engine_ids": engine_ids, "mode": mode,
            "engine_ids_complete": engine_ids_complete,
            "selections": std::mem::take(&mut capture.selections), "truncated": std::mem::take(&mut capture.truncated),
            "unattributed_gates": std::mem::take(&mut capture.gates),
            "unattributed_prediction": std::mem::take(&mut capture.prediction),
            "cache_evidence": "unknown",
        });
        let encoded = value.to_string();
        tracing::info!(target: "smg::cache_trace", evidence = %encoded, "Cache routing dispatch");
        if header_enabled() {
            let header = gateway_header(&value);
            if header.is_none() {
                tracing::info!(target: "smg::cache_trace", "Cache trace header omitted: size or encoding limit");
            }
            header
        } else {
            None
        }
    }).ok().flatten()
}

fn gateway_header(value: &Value) -> Option<String> {
    let selections: Vec<_> = value["selections"]
        .as_array()?
        .iter()
        .map(|selection| {
            json!({
                "policy": selection["policy"], "origin": selection["origin"],
                "prediction": selection["prediction"],
            })
        })
        .collect();
    let header = json!({
        "schema": value["schema"], "root_id": value["root_id"],
        "dispatch_id": value["dispatch_id"], "attempt": value["attempt"],
        "engine_ids": value["engine_ids"], "engine_ids_complete": value["engine_ids_complete"],
        "selections": selections, "truncated": value["truncated"],
    })
    .to_string();
    (header.len() <= 2048 && header.is_ascii()).then_some(header)
}

pub(crate) fn failure(root_id: Option<&str>, status: u16) {
    if enabled() {
        let _ = CAPTURE.try_with(|capture| {
            let capture = capture.borrow();
            let evidence = json!({"root_id": root_id, "status": status,
                "selections": capture.selections, "gates": capture.gates,
                "truncated": capture.truncated});
            tracing::info!(target: "smg::cache_trace", evidence = %evidence, "Cache routing failure");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_observation_survives_dispatch_load_credit() {
        use crate::worker::BasicWorkerBuilder;
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8000").build());
        let workers = vec![worker];
        let mut capture = Capture::default();
        capture.observe_candidates(&workers);
        workers[0].increment_load();
        assert_eq!(capture.candidates[0]["load"], json!(0));
        assert_eq!(workers[0].load(), 1);
        assert!(capture.candidates_complete);
        assert!(capture.observation_started_ns <= capture.observation_finished_ns);
    }

    #[test]
    fn gateway_header_keeps_join_fields_without_candidate_state() {
        let mut evidence = json!({
            "schema": 1, "root_id": "root", "dispatch_id": "dispatch", "attempt": 1,
            "engine_ids": ["engine"], "engine_ids_complete": true, "truncated": false,
            "selections": [{"policy": "cache_aware", "origin": "policy",
                "prediction": {"source": "approximate_tree"},
                "worker": "selected-worker", "candidates": [{"worker": "candidate-worker", "load": 99}],
                "gates": [{"spill": true}]}],
        });
        let encoded = gateway_header(&evidence).unwrap();
        assert!(!encoded.contains("worker"));
        assert!(!encoded.contains("gates"));
        let decoded: Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded["engine_ids"], json!(["engine"]));
        assert_eq!(decoded["attempt"], 1);
        evidence["root_id"] = Value::String("x".repeat(2049));
        assert!(gateway_header(&evidence).is_none());
    }
}
