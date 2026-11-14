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

use crate::worker::Worker;

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
mod tests {
    use super::*;

    /// The first observed candidate as the evidence lists it.
    fn first_candidate(capture: &Capture) -> Value {
        json!(capture.candidates[0])
    }

    /// A row is written with the evidence from the values its selection
    /// kept, and it is the row the selection used to write as text where it
    /// ran: the same keys, order and values for a candidate that advertises
    /// one epoch, one that advertises two and one that advertises none, with
    /// the prediction, score and gate recorded for it. What a worker shows
    /// when the evidence is written is not in it: a load credit, an overload
    /// flag and a health failure after the selection leave the row as it was
    /// observed.
    #[tokio::test]
    async fn a_row_written_with_the_evidence_is_the_row_as_it_was_observed() {
        use openai_protocol::worker::WorkerStatus;

        use crate::worker::BasicWorkerBuilder;
        let worker = |url: &str, epochs: Option<&str>| -> Arc<dyn Worker> {
            // Ready, so that the health failure below moves what a worker
            // shows after its row was observed.
            let mut builder = BasicWorkerBuilder::new(url).status(WorkerStatus::Ready);
            if let Some(epochs) = epochs {
                builder = builder.labels(std::collections::HashMap::from([(
                    "cache_trace_epochs".to_string(),
                    epochs.to_string(),
                )]));
            }
            Arc::new(builder.build())
        };
        let workers = vec![
            worker("grpc://one:50051", Some(r#"["first"]"#)),
            worker("grpc://two:50051", Some(r#"["first","second"]"#)),
            worker("grpc://none:50051", None),
        ];
        workers[2].increment_load();
        scope(true, async {
            // The rows as the selection wrote them before: one object per
            // candidate, read from the workers where the pick ran.
            let observed: Vec<Value> = workers
                .iter()
                .map(|worker| {
                    let epochs = if worker.cache_trace_epochs_valid() {
                        worker.advertised_cache_trace_epochs()
                    } else {
                        Vec::new()
                    };
                    let epoch = if epochs.len() == 1 {
                        epochs.first()
                    } else {
                        None
                    };
                    json!({
                        "worker": worker.redacted_url(), "load": worker.load(), "healthy": worker.is_healthy(),
                        "overloaded": worker.is_overloaded(), "registry_revision": worker.revision(),
                        "backend_cache_epoch": epoch,
                        "backend_cache_epochs": epochs,
                        "backend_cache_epoch_source": "registration_metadata",
                    })
                })
                .collect();
            assert_eq!(observed[0]["backend_cache_epoch"], "first");
            assert_ne!(observed[0]["registry_revision"], 41);
            assert_eq!(observed[1]["healthy"], true);
            assert_eq!(observed[2]["load"], 1);
            begin_selection(&workers);
            prediction(json!({"source": "approximate_tree", "overlap_blocks": 4}));
            score(json!({"worker": "grpc://two:50051", "score": 0.9}));
            gate(json!({"spill": false}));
            selection("cache_aware", "policy", &workers, Some(1));
            // After the selection, before the evidence is written: a load
            // credit, a revision raised, the overload flag, and a health
            // failure, which also withdraws the epochs.
            workers[0].increment_load();
            workers[0].raise_revision_to(41);
            workers[1].set_overloaded(true);
            workers[1].set_status(WorkerStatus::NotReady);
            assert_eq!(workers[0].revision(), 41);
            assert!(workers[1].is_overloaded());
            assert!(!workers[1].is_healthy());
            assert!(!workers[1].cache_trace_epochs_valid());
            let rows = captured_selections();
            let row = &rows[0];
            let expected = json!({
                "policy": "cache_aware", "origin": "policy",
                "worker": "grpc://two:50051",
                "prediction": {"source": "approximate_tree", "overlap_blocks": 4},
                "candidates": observed,
                "candidates_complete": true,
                "observation_started_ns": row["observation_started_ns"],
                "observation_finished_ns": row["observation_finished_ns"],
                "snapshot_atomic": false,
                "scores": [{"worker": "grpc://two:50051", "score": 0.9}],
                "load_source": "router_inflight", "eligibility_scope": "policy_candidates",
                "load_observation_phase": "before_selection",
                "gates": [{"spill": false}],
            });
            assert_eq!(*row, expected);
            assert_eq!(row.to_string(), expected.to_string());
        })
        .await;
    }

    /// A row with no worker (a shed, a refused commit) stays in the capture
    /// in pick order: the failure line writes it and leaves it, and the
    /// dispatch that follows a re-decision writes it before the row of the
    /// pick that held.
    #[tokio::test]
    async fn a_row_without_a_worker_is_written_in_pick_order_by_the_failure_and_the_dispatch() {
        use crate::worker::BasicWorkerBuilder;
        let workers: Vec<Arc<dyn Worker>> = ["grpc://one:50051", "grpc://two:50051"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        scope(true, async {
            begin_selection(&workers);
            selection("round_robin", "policy", &workers, None);
            let failed =
                CAPTURE.with(|capture| failure_evidence(&capture.borrow(), Some("root"), 503));
            assert_eq!(failed["status"], 503);
            assert_eq!(failed["selections"].as_array().map(Vec::len), Some(1));
            assert!(failed["selections"][0]["worker"].is_null());
            assert_eq!(
                failed["selections"][0]["candidates"][1]["worker"],
                "grpc://two:50051"
            );
            begin_selection(&workers[1..]);
            selection("round_robin", "policy", &workers[1..], Some(0));
            let rows = captured_selections();
            assert_eq!(rows.len(), 2, "the failure line left the first row");
            assert!(rows[0]["worker"].is_null());
            assert_eq!(rows[0]["candidates"].as_array().map(Vec::len), Some(2));
            assert_eq!(rows[1]["worker"], "grpc://two:50051");
            assert_eq!(rows[1]["candidates"].as_array().map(Vec::len), Some(1));
            let header = dispatch(Some("root"), 1, vec!["engine".into()], true, "regular")
                .expect("the header for the request that asked");
            let header: Value = serde_json::from_str(&header).unwrap();
            assert!(header["selections"][0]["worker"].is_null());
            assert_eq!(header["selections"][1]["worker"], "grpc://two:50051");
            assert!(captured_selections().is_empty(), "the dispatch took them");
        })
        .await;
    }

    /// A selection observes 32 candidates at most and says whether that was
    /// all of them; a pick beyond the 32 observed is still named.
    #[tokio::test]
    async fn a_row_observes_32_candidates_and_says_whether_that_was_all() {
        use crate::worker::BasicWorkerBuilder;
        let workers: Vec<Arc<dyn Worker>> = (0..33)
            .map(|i| {
                Arc::new(BasicWorkerBuilder::new(format!("grpc://w{i}:50051")).build())
                    as Arc<dyn Worker>
            })
            .collect();
        scope(true, async {
            begin_selection(&workers[..32]);
            selection("round_robin", "policy", &workers[..32], Some(31));
            begin_selection(&workers);
            selection("round_robin", "policy", &workers, Some(32));
            let rows = captured_selections();
            assert_eq!(rows[0]["candidates"].as_array().map(Vec::len), Some(32));
            assert_eq!(rows[0]["candidates_complete"], true);
            assert_eq!(rows[0]["worker"], "grpc://w31:50051");
            assert_eq!(rows[1]["candidates"].as_array().map(Vec::len), Some(32));
            assert_eq!(rows[1]["candidates_complete"], false);
            assert_eq!(rows[1]["candidates"][31]["worker"], "grpc://w31:50051");
            assert_eq!(rows[1]["worker"], "grpc://w32:50051");
        })
        .await;
    }

    #[test]
    fn candidate_observation_survives_dispatch_load_credit() {
        use crate::worker::BasicWorkerBuilder;
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8000").build());
        let workers = vec![worker];
        let mut capture = Capture::default();
        capture.observe_candidates(&workers);
        workers[0].increment_load();
        assert_eq!(first_candidate(&capture)["load"], json!(0));
        assert_eq!(workers[0].load(), 1);
        assert!(capture.candidates_complete);
        assert!(capture.observation_started_ns <= capture.observation_finished_ns);
    }

    #[test]
    fn candidate_epochs_are_replica_scoped_and_invalidated_after_failure() {
        use openai_protocol::worker::WorkerStatus;

        use crate::worker::BasicWorkerBuilder;
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8000")
                .labels(std::collections::HashMap::from([(
                    "cache_trace_epochs".to_string(),
                    r#"["one","two"]"#.to_string(),
                )]))
                .build(),
        );
        let mut capture = Capture::default();
        capture.observe_candidates(std::slice::from_ref(&worker));
        assert_eq!(
            first_candidate(&capture)["backend_cache_epochs"],
            json!(["one", "two"])
        );
        assert!(first_candidate(&capture)["backend_cache_epoch"].is_null());
        worker.set_status(WorkerStatus::NotReady);
        worker.set_status(WorkerStatus::Ready);
        capture.observe_candidates(&[worker]);
        assert_eq!(first_candidate(&capture)["backend_cache_epochs"], json!([]));
        assert!(first_candidate(&capture)["backend_cache_epoch"].is_null());
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
