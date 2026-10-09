//! Opt-in routing evidence, scoped to one pipeline run across async retries.

use std::{
    cell::RefCell,
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use http::HeaderMap;
use serde_json::{json, Value};

use crate::worker::Worker;

#[derive(Default)]
struct Capture {
    /// The request asked for its decision record (`x-smg-cache-trace: 1`).
    header_requested: bool,
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

/// Evidence is captured when one of its two consumers is switched on for
/// the process, the INFO line (`SMG_CACHE_TRACE=1`) or the response header
/// (`SMG_CACHE_TRACE_HEADER=1`), and for the request that asked for its own
/// record with the `x-smg-cache-trace: 1` request header (see [`scope`]).
/// Unset both and nothing is observed for the requests that did not ask.
pub(crate) fn enabled() -> bool {
    configured() || CAPTURE.try_with(|_| ()).is_ok()
}

/// The process-wide switches: either one captures every request.
fn configured() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| log_enabled() || header_enabled())
}

/// The request header that asks for this one request's decision record in
/// the response, whatever the process switches say and with no log line.
pub(crate) const REQUEST_HEADER: &str = "x-smg-cache-trace";

/// Whether the request asked for its decision record (`x-smg-cache-trace: 1`).
pub(crate) fn requested(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|headers| headers.get(REQUEST_HEADER))
        .is_some_and(|value| value.as_bytes().trim_ascii() == b"1")
}

/// `SMG_CACHE_TRACE=1`: the evidence of a dispatch (one in every
/// `SMG_CACHE_TRACE_SAMPLE`) and of a failed request as an INFO line.
fn log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SMG_CACHE_TRACE").is_ok_and(|v| v == "1"))
}

/// `SMG_CACHE_TRACE_HEADER=1`: the decision record of a dispatch in the
/// `x-smg-cache-trace` response header, with or without the log line.
fn header_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1"))
}

/// One dispatch line in every `every` dispatches, counted per process.
struct Sampler {
    every: u64,
    dispatches: AtomicU64,
}

impl Sampler {
    const fn new(every: u64) -> Self {
        Self {
            every,
            dispatches: AtomicU64::new(0),
        }
    }

    fn admit(&self) -> bool {
        self.every <= 1
            || self
                .dispatches
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(self.every)
    }
}

/// `SMG_CACHE_TRACE_SAMPLE=N`: log the evidence of one dispatch in every N
/// (default 1: every dispatch). The response header is not sampled.
fn sampler() -> &'static Sampler {
    static SAMPLER: OnceLock<Sampler> = OnceLock::new();
    SAMPLER.get_or_init(|| {
        Sampler::new(parse_sample(
            std::env::var("SMG_CACHE_TRACE_SAMPLE").ok().as_deref(),
        ))
    })
}

/// `SMG_CACHE_TRACE_MAX_BYTES=B`: an evidence line longer than B bytes drops
/// its candidate, score and gate lists and keeps the decision record (see
/// [`compact`]). Unset or unparsable: no cap.
fn max_bytes() -> Option<usize> {
    static MAX_BYTES: OnceLock<Option<usize>> = OnceLock::new();
    *MAX_BYTES
        .get_or_init(|| parse_max_bytes(std::env::var("SMG_CACHE_TRACE_MAX_BYTES").ok().as_deref()))
}

fn parse_sample(value: Option<&str>) -> u64 {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|every| *every > 0)
        .unwrap_or(1)
}

fn parse_max_bytes(value: Option<&str>) -> Option<usize> {
    value.and_then(|value| value.trim().parse().ok())
}

/// The dispatch's evidence line: `None` when the sampler skips it, compacted
/// when it is longer than the cap.
fn log_line(value: Value, sampler: &Sampler, cap: Option<usize>) -> Option<String> {
    sampler.admit().then(|| encode_capped(value, cap))
}

/// What one dispatch's evidence becomes: the response header when the header
/// switch is on, the INFO line when the log switch is on and the sampler
/// admits the dispatch. The header depends on neither the log switch nor the
/// sampler.
fn emit(
    value: Value,
    log: bool,
    header: bool,
    sampler: &Sampler,
    cap: Option<usize>,
) -> (Option<String>, Option<String>) {
    let header = header.then(|| gateway_header(&value)).flatten();
    let line = log.then(|| log_line(value, sampler, cap)).flatten();
    (header, line)
}

fn encode_capped(mut value: Value, cap: Option<usize>) -> String {
    let encoded = value.to_string();
    if cap.is_some_and(|cap| encoded.len() > cap) {
        compact(&mut value);
        return value.to_string();
    }
    encoded
}

/// Drop the lists that grow with the fleet (per-candidate state, scores and
/// gates: ~116 bytes per candidate) and keep the decision record: ids,
/// policy, origin, the chosen worker and the prediction. Each selection
/// records how many entries of each list it dropped; the line is marked
/// `capped`.
fn compact(value: &mut Value) {
    const LISTS: [&str; 3] = ["candidates", "scores", "gates"];
    if let Some(selections) = value.get_mut("selections").and_then(Value::as_array_mut) {
        for selection in selections.iter_mut().filter_map(Value::as_object_mut) {
            let elided = LISTS
                .iter()
                .filter_map(|list| {
                    let dropped = selection.remove(*list)?;
                    Some((
                        (*list).to_string(),
                        json!(dropped.as_array().map_or(0, Vec::len)),
                    ))
                })
                .collect();
            selection.insert("elided".to_string(), Value::Object(elided));
        }
    }
    if let Some(object) = value.as_object_mut() {
        object.remove("unattributed_gates");
        object.remove("gates");
        object.insert("capped".to_string(), json!(true));
    }
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

/// Run `future` with a capture in scope when a process switch is on or the
/// request asked for its header (`header_requested`); otherwise as it is.
pub(crate) async fn scope<F: Future>(header_requested: bool, future: F) -> F::Output {
    if configured() || header_requested {
        let capture = Capture {
            header_requested,
            ..Capture::default()
        };
        CAPTURE.scope(RefCell::new(capture), future).await
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
        let header_wanted = header_enabled() || capture.header_requested;
        let (header, line) = emit(value, log_enabled(), header_wanted, sampler(), max_bytes());
        if header_wanted && header.is_none() {
            // The process switch asked: say so at INFO. Only the request asked: the
            // request's own header stays its only trace, so the omission is at DEBUG.
            if header_enabled() {
                tracing::info!(target: "smg::cache_trace", "Cache trace header omitted: size or encoding limit");
            } else {
                tracing::debug!(target: "smg::cache_trace", "Cache trace header omitted: size or encoding limit");
            }
        }
        if let Some(encoded) = line {
            tracing::info!(target: "smg::cache_trace", evidence = %encoded, "Cache routing dispatch");
        }
        header
    }).ok().flatten()
}

fn gateway_header(value: &Value) -> Option<String> {
    let selections: Vec<_> = value["selections"]
        .as_array()?
        .iter()
        .map(|selection| {
            json!({
                "policy": selection["policy"], "origin": selection["origin"],
                "worker": selection["worker"], "prediction": selection["prediction"],
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
    if log_enabled() {
        let _ = CAPTURE.try_with(|capture| {
            let capture = capture.borrow();
            let evidence = json!({"root_id": root_id, "status": status,
                "selections": capture.selections, "gates": capture.gates,
                "truncated": capture.truncated});
            let evidence = encode_capped(evidence, max_bytes());
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
        assert!(!encoded.contains("candidate-worker"));
        assert!(!encoded.contains("gates"));
        let decoded: Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded["engine_ids"], json!(["engine"]));
        assert_eq!(decoded["attempt"], 1);
        assert_eq!(
            decoded["selections"][0]["worker"], "selected-worker",
            "the chosen worker joins the record"
        );
        evidence["root_id"] = Value::String("x".repeat(2049));
        assert!(gateway_header(&evidence).is_none());
    }

    fn fleet_evidence(candidates: usize) -> Value {
        let candidates: Vec<Value> = (0..candidates)
            .map(|index| {
                json!({
                    "worker": format!("http://10.0.0.{index}:8000"), "load": index, "healthy": true,
                    "overloaded": false, "registry_revision": 7, "backend_cache_epoch": null,
                })
            })
            .collect();
        json!({
            "schema": 1, "root_id": "root", "dispatch_id": "dispatch", "attempt": 1,
            "engine_ids": ["engine"], "engine_ids_complete": true, "truncated": false,
            "selections": [{"policy": "cache_aware", "origin": "policy", "worker": "http://10.0.0.3:8000",
                "prediction": {"source": "approximate_tree", "overlap_blocks": 4},
                "candidates": candidates, "candidates_complete": true,
                "scores": [{"worker": "http://10.0.0.3:8000", "score": 0.9}],
                "gates": [{"spill": false}]}],
            "unattributed_gates": [{"spill": true}],
        })
    }

    #[test]
    fn sample_knob_logs_one_dispatch_in_every_n() {
        assert_eq!(parse_sample(None), 1);
        assert_eq!(parse_sample(Some("0")), 1);
        assert_eq!(parse_sample(Some("many")), 1);
        assert_eq!(parse_sample(Some(" 100 ")), 100);
        let every_request = Sampler::new(1);
        assert!((0..10).all(|_| every_request.admit()));
        let one_in_hundred = Sampler::new(100);
        let logged = (0..1000)
            .filter(|_| log_line(fleet_evidence(32), &one_in_hundred, None).is_some())
            .count();
        assert_eq!(logged, 10);
        assert!(
            Sampler::new(100).admit(),
            "the first dispatch is always logged"
        );
    }

    #[test]
    fn byte_cap_keeps_the_decision_record_and_drops_the_candidate_lists() {
        assert_eq!(parse_max_bytes(None), None);
        assert_eq!(parse_max_bytes(Some("bytes")), None);
        assert_eq!(parse_max_bytes(Some("1024")), Some(1024));
        let full = encode_capped(fleet_evidence(32), None);
        assert!(
            full.len() > 3000,
            "32 candidates make the line {} bytes",
            full.len()
        );
        assert_eq!(encode_capped(fleet_evidence(32), Some(full.len())), full);
        let capped = encode_capped(fleet_evidence(32), Some(1024));
        assert!(capped.len() < 1024, "capped line is {} bytes", capped.len());
        let decoded: Value = serde_json::from_str(&capped).unwrap();
        assert_eq!(decoded["capped"], json!(true));
        assert_eq!(decoded["root_id"], "root");
        assert_eq!(decoded["engine_ids"], json!(["engine"]));
        let selection = &decoded["selections"][0];
        assert_eq!(selection["policy"], "cache_aware");
        assert_eq!(selection["worker"], "http://10.0.0.3:8000");
        assert_eq!(selection["prediction"]["overlap_blocks"], 4);
        assert!(selection.get("candidates").is_none());
        assert!(selection.get("scores").is_none());
        assert_eq!(
            selection["elided"],
            json!({"candidates": 32, "scores": 1, "gates": 1})
        );
        assert!(decoded.get("unattributed_gates").is_none());
        let failure = encode_capped(
            json!({"root_id": "root", "status": 503, "selections": [], "gates": [{"spill": true}], "truncated": false}),
            Some(0),
        );
        let decoded: Value = serde_json::from_str(&failure).unwrap();
        assert_eq!(decoded["status"], 503);
        assert!(decoded.get("gates").is_none());
    }

    #[test]
    fn the_header_switch_alone_yields_the_header_and_no_log_line() {
        let every_dispatch = Sampler::new(1);
        let (header, line) = emit(fleet_evidence(2), false, true, &every_dispatch, None);
        let header: Value = serde_json::from_str(&header.unwrap()).unwrap();
        assert_eq!(header["selections"][0]["prediction"]["overlap_blocks"], 4);
        assert!(line.is_none(), "the log switch is off");
        let (header, line) = emit(fleet_evidence(2), true, false, &every_dispatch, None);
        assert!(header.is_none(), "the header switch is off");
        assert!(line.is_some());
        let (header, line) = emit(fleet_evidence(2), true, true, &every_dispatch, None);
        assert!(header.is_some() && line.is_some());
        let one_in_hundred = Sampler::new(100);
        let outcomes: Vec<_> = (0..200)
            .map(|_| emit(fleet_evidence(2), true, true, &one_in_hundred, None))
            .collect();
        assert!(
            outcomes.iter().all(|(header, _)| header.is_some()),
            "sampling the log line leaves the header on every dispatch"
        );
        assert_eq!(
            outcomes.iter().filter(|(_, line)| line.is_some()).count(),
            2
        );
    }

    #[test]
    fn the_request_header_asks_for_this_requests_record() {
        let mut headers = HeaderMap::new();
        assert!(!requested(None));
        assert!(!requested(Some(&headers)));
        headers.insert(REQUEST_HEADER, http::HeaderValue::from_static("0"));
        assert!(!requested(Some(&headers)));
        headers.insert(REQUEST_HEADER, http::HeaderValue::from_static(" 1 "));
        assert!(requested(Some(&headers)));
    }

    #[tokio::test]
    async fn a_requesting_scope_captures_its_request_alone() {
        use crate::worker::BasicWorkerBuilder;
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("grpc://worker:50051").build());
        let workers = vec![worker];
        scope(true, async {
            assert!(enabled(), "the request asked: its evidence is captured");
            begin_selection(&workers);
            selection("random", "policy", &workers, Some(0));
            let header = dispatch(Some("root"), 0, vec!["engine".into()], true, "regular")
                .expect("the header for the request that asked");
            let header: Value = serde_json::from_str(&header).unwrap();
            assert_eq!(header["root_id"], "root");
            assert_eq!(header["selections"][0]["worker"], "grpc://worker:50051");
            assert_eq!(header["selections"][0]["policy"], "random");
        })
        .await;
        assert_eq!(
            enabled(),
            configured(),
            "outside that request only the process switches count"
        );
        scope(false, async { assert_eq!(enabled(), configured()) }).await;
    }
}
