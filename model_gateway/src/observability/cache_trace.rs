//! Opt-in routing evidence, scoped to one pipeline run across async retries.

use std::{
    cell::RefCell,
    future::Future,
    sync::{Arc, OnceLock},
};

use serde_json::{json, Value};

use crate::worker::Worker;

#[derive(Default)]
struct Capture {
    prediction: Value,
    selections: Vec<Value>,
    truncated: bool,
    gates: Vec<Value>,
}

tokio::task_local! {
    static CAPTURE: RefCell<Capture>;
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SMG_CACHE_TRACE").is_ok_and(|v| v == "1"))
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

pub(crate) fn selection(
    policy: &str,
    origin: &str,
    workers: &[Arc<dyn Worker>],
    selected: Option<usize>,
) {
    if !enabled() {
        return;
    }
    let _ =
        CAPTURE.try_with(|capture| {
            let mut capture = capture.borrow_mut();
            if capture.selections.len() >= 16 {
                capture.truncated = true;
                return;
            }
            let candidates: Vec<_> = workers.iter().take(32).map(|worker| json!({
            "worker": worker.url(), "load": worker.load(), "healthy": worker.is_healthy(),
            "overloaded": worker.is_overloaded(),
        })).collect();
            let value = json!({
                "policy": policy, "origin": origin,
                "worker": selected.map(|idx| workers[idx].url()),
                "prediction": std::mem::take(&mut capture.prediction),
                "candidates": candidates, "candidates_complete": workers.len() <= 32,
                "load_source": "router_inflight", "eligibility_scope": "policy_candidates",
                "load_observation_phase": "after_selection",
                "gates": std::mem::take(&mut capture.gates),
            });
            capture.selections.push(value);
        });
}

pub(crate) fn dispatch(
    root_id: Option<&str>,
    attempt: u32,
    engine_ids: Vec<String>,
    mode: &str,
) -> Option<String> {
    if !enabled() {
        return None;
    }
    CAPTURE.try_with(|capture| {
        let mut capture = capture.borrow_mut();
        let value = json!({
            "schema": 1, "root_id": root_id, "dispatch_id": uuid::Uuid::now_v7().to_string(),
            "attempt": attempt, "engine_ids": engine_ids, "mode": mode,
            "engine_ids_complete": engine_ids.len() < 32,
            "selections": std::mem::take(&mut capture.selections), "truncated": std::mem::take(&mut capture.truncated),
            "cache_evidence": "unknown",
        });
        let encoded = value.to_string();
        tracing::info!(target: "smg::cache_trace", evidence = %encoded, "Cache routing dispatch");
        (encoded.len() <= 16384).then_some(encoded)
    }).ok().flatten()
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
