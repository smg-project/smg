//! Opt-in routing evidence, scoped to one pipeline run across async retries.
//! Backend epochs come from registration metadata and are withdrawn after a
//! health failure until worker removal and registration. An undetected restart still needs
//! native epoch evidence; several epochs do not identify a request's DP replica.

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
use serde::{ser::SerializeMap, Serialize, Serializer};
use serde_json::{json, Value};
use smg_worker::identity::basic::Worker;

#[derive(Default)]
struct Capture {
    /// The request asked for its decision record (`x-smg-cache-trace: 1`).
    header_requested: bool,
    prediction: Value,
    selections: Vec<Selection>,
    truncated: bool,
    gates: Vec<Value>,
    candidates: Vec<Candidate>,
    candidates_complete: bool,
    observation_started_ns: u64,
    observation_finished_ns: u64,
    scores: Vec<Value>,
}

/// One candidate as a selection observed it: the worker, and the state read
/// from it where the pick runs, which may be under a caller's lock. Nothing
/// here is text: the candidate's JSON object is written from these values
/// when the evidence is (see its [`Serialize`]), after the pick and its lock.
/// The URL and the advertised epochs are the worker's own and never change,
/// so they are read at the write; whether those epochs were current is read
/// here.
struct Candidate {
    worker: Arc<dyn Worker>,
    load: usize,
    healthy: bool,
    overloaded: bool,
    revision: u64,
    epochs_valid: bool,
}

impl Serialize for Candidate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let epochs = if self.epochs_valid {
            self.worker.advertised_cache_trace_epochs()
        } else {
            Vec::new()
        };
        let epoch = if epochs.len() == 1 {
            epochs.first()
        } else {
            None
        };
        let mut row = serializer.serialize_map(Some(8))?;
        row.serialize_entry("worker", self.worker.redacted_url())?;
        row.serialize_entry("load", &self.load)?;
        row.serialize_entry("healthy", &self.healthy)?;
        row.serialize_entry("overloaded", &self.overloaded)?;
        row.serialize_entry("registry_revision", &self.revision)?;
        row.serialize_entry("backend_cache_epoch", &epoch)?;
        row.serialize_entry("backend_cache_epochs", &epochs)?;
        row.serialize_entry("backend_cache_epoch_source", "registration_metadata")?;
        row.end()
    }
}

/// One selection's row as it was observed, kept as values until the evidence
/// is written.
struct Selection {
    /// The policy's name and the seam's: static, so a row copies neither.
    policy: &'static str,
    origin: &'static str,
    worker: Option<Arc<dyn Worker>>,
    prediction: Value,
    candidates: Vec<Candidate>,
    candidates_complete: bool,
    observation_started_ns: u64,
    observation_finished_ns: u64,
    scores: Vec<Value>,
    gates: Vec<Value>,
}

impl Serialize for Selection {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let worker = self.worker.as_ref().map(|worker| worker.redacted_url());
        let mut row = serializer.serialize_map(Some(14))?;
        row.serialize_entry("policy", &self.policy)?;
        row.serialize_entry("origin", &self.origin)?;
        row.serialize_entry("worker", &worker)?;
        row.serialize_entry("prediction", &self.prediction)?;
        row.serialize_entry("candidates", &self.candidates)?;
        row.serialize_entry("candidates_complete", &self.candidates_complete)?;
        row.serialize_entry("observation_started_ns", &self.observation_started_ns)?;
        row.serialize_entry("observation_finished_ns", &self.observation_finished_ns)?;
        row.serialize_entry("snapshot_atomic", &false)?;
        row.serialize_entry("scores", &self.scores)?;
        row.serialize_entry("load_source", "router_inflight")?;
        row.serialize_entry("eligibility_scope", "policy_candidates")?;
        row.serialize_entry("load_observation_phase", "before_selection")?;
        row.serialize_entry("gates", &self.gates)?;
        row.end()
    }
}

impl Capture {
    /// Read the candidates where the pick is about to run: each worker's
    /// handle and its live state, no text. A caller may hold a lock here.
    fn observe_candidates(&mut self, workers: &[Arc<dyn Worker>]) {
        self.observation_started_ns = timestamp_ns();
        self.candidates = workers
            .iter()
            .take(32)
            .map(|worker| Candidate {
                epochs_valid: worker.cache_trace_epochs_valid(),
                load: worker.load(),
                healthy: worker.is_healthy(),
                overloaded: worker.is_overloaded(),
                revision: worker.revision(),
                worker: Arc::clone(worker),
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
pub(crate) fn configured() -> bool {
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
    policy: &'static str,
    origin: &'static str,
    workers: &[Arc<dyn Worker>],
    selected: Option<usize>,
) {
    if !enabled() {
        return;
    }
    let _ = CAPTURE.try_with(|capture| {
        let mut capture = capture.borrow_mut();
        let capture = &mut *capture;
        if capture.selections.len() >= 16 {
            capture.truncated = true;
            return;
        }
        // The row's values, moved: its text is written with the evidence.
        capture.selections.push(Selection {
            policy,
            origin,
            worker: selected.map(|idx| Arc::clone(&workers[idx])),
            prediction: std::mem::take(&mut capture.prediction),
            candidates: std::mem::take(&mut capture.candidates),
            candidates_complete: capture.candidates_complete,
            observation_started_ns: capture.observation_started_ns,
            observation_finished_ns: capture.observation_finished_ns,
            scores: std::mem::take(&mut capture.scores),
            gates: std::mem::take(&mut capture.gates),
        });
    });
}

/// The rows a request's capture keeps at most. A row past them is dropped
/// and marks the record truncated. [`selection`] states the same bound in
/// place.
const MAX_SELECTIONS: usize = 16;

/// What a selection observed of its candidates, held by its caller until
/// the selection's row is written ([`selection_of`]): the candidates,
/// whether they are all of the list, and the reading's two instants.
pub(crate) struct Observed {
    candidates: Vec<Candidate>,
    candidates_complete: bool,
    started_ns: u64,
    finished_ns: u64,
}

/// A list's candidates as a selection observes them where its pick is about
/// to run ([`begin_selection`]), handed to the caller and not kept in the
/// request's capture. It is for a call that may pick from two lists and
/// answers workers, not places in a list: its caller reads each list ahead
/// of the call and writes a row for each pick behind it
/// ([`selection_of`]). `None` where nothing is captured for the request,
/// which is one lookup.
pub(crate) fn observe(workers: &[Arc<dyn Worker>]) -> Option<Observed> {
    CAPTURE
        .try_with(|_| {
            let mut read = Capture::default();
            read.observe_candidates(workers);
            Observed {
                candidates: read.candidates,
                candidates_complete: read.candidates_complete,
                started_ns: read.observation_started_ns,
                finished_ns: read.observation_finished_ns,
            }
        })
        .ok()
}

/// The row of the selection whose candidates `observed` holds: `policy` and
/// `origin` as [`selection`] takes them, and `worker` the worker picked,
/// `None` where the call picked nobody. The row holds no prediction, score
/// or gate: the call that picked wrote none.
pub(crate) fn selection_of(
    policy: &'static str,
    origin: &'static str,
    observed: Observed,
    worker: Option<&Arc<dyn Worker>>,
) {
    let _ = CAPTURE.try_with(|capture| {
        let mut capture = capture.borrow_mut();
        if capture.selections.len() >= MAX_SELECTIONS {
            capture.truncated = true;
            return;
        }
        capture.selections.push(Selection {
            policy,
            origin,
            worker: worker.map(Arc::clone),
            prediction: Value::Null,
            candidates: observed.candidates,
            candidates_complete: observed.candidates_complete,
            observation_started_ns: observed.started_ns,
            observation_finished_ns: observed.finished_ns,
            scores: Vec::new(),
            gates: Vec::new(),
        });
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
            let evidence = failure_evidence(&capture.borrow(), root_id, status);
            let evidence = encode_capped(evidence, max_bytes());
            tracing::info!(target: "smg::cache_trace", evidence = %evidence, "Cache routing failure");
        });
    }
}

/// A failed request's evidence: its rows so far, written from the capture
/// and left in it, where a retry's dispatch still finds them.
fn failure_evidence(capture: &Capture, root_id: Option<&str>, status: u16) -> Value {
    json!({"root_id": root_id, "status": status,
        "selections": capture.selections, "gates": capture.gates,
        "truncated": capture.truncated})
}

/// The rows the capture in scope holds, each as the evidence line lists it
/// (its candidates, scores and gates included), for a test that reads what
/// a selection observed; empty outside a capture.
#[cfg(test)]
pub(crate) fn captured_selections() -> Vec<Value> {
    CAPTURE
        .try_with(|capture| {
            let capture = capture.borrow();
            capture.selections.iter().map(|row| json!(row)).collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "../../tests/unit/observability/cache_trace/tests.rs"]
mod tests;
