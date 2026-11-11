//! The decision ring and `Explain`: a decision the request asked to see
//! (`x-smg-cache-trace: 1`), or one a process-wide cache-trace switch
//! captures, leaves its record in a bounded ring, and `GET
//! /policy/explain/{id}` on the admin plane reads it back by the decision id
//! the request's trace header names. An untraced decision writes nothing.
//! The record holds the trace and what the host knew at the decision, with
//! worker ids in place of addresses; the read resolves the ids against the
//! registry, so a worker gone since the decision stays an id.

use std::{collections::VecDeque, sync::Arc};

use axum::{
    response::{IntoResponse, Response},
    Json,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use smg_policy_controller::{Engine, Phase, Targets, WorkerId, TRACE_TOP};

use crate::{
    app_context::AppContext, policies::engine_host::worker_id, routers::error as route_error,
    worker::Worker,
};

/// Traced decisions the ring keeps; the oldest leaves at the cap.
pub const RING_CAPACITY: usize = 1024;

/// One decision as the ring keeps it.
#[derive(Clone, Debug)]
pub struct DecisionRecord {
    /// The lease id of the decision, the id the trace header names.
    pub decision_id: u64,
    pub at_unix_micros: u64,
    pub model: Arc<str>,
    pub phase: Phase,
    /// The profile that decided, as the engine's position.
    pub profile: u16,
    /// The `outcome` label of the decision counter.
    pub outcome: &'static str,
    /// The stage that settled the decision.
    pub stage: &'static str,
    pub survivors: u32,
    pub pool_size: u32,
    pub attempt: u8,
    /// The ids a commit of this request had refused before this decision.
    pub excluded: u8,
    pub seed: u64,
    pub generation: u64,
    pub top: [(WorkerId, f32); TRACE_TOP],
    pub top_len: u8,
    /// The ordered targets of a pick; empty for a shed.
    pub targets: Targets,
}

/// The ring of traced decisions, newest last.
#[derive(Debug)]
pub struct DecisionRing {
    entries: Mutex<VecDeque<DecisionRecord>>,
}

impl Default for DecisionRing {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionRing {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
        }
    }

    /// Keep `record`; the oldest record leaves at the cap.
    pub fn record(&self, record: DecisionRecord) {
        let mut entries = self.entries.lock();
        if entries.len() == RING_CAPACITY {
            entries.pop_front();
        }
        entries.push_back(record);
    }

    /// The record of decision `id`, searched newest first: decisions end out
    /// of order across threads, so the ring is not sorted by id.
    pub fn find(&self, id: u64) -> Option<DecisionRecord> {
        self.entries
            .lock()
            .iter()
            .rev()
            .find(|record| record.decision_id == id)
            .cloned()
    }

    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Single => "single",
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
        Phase::Encode => "encode",
    }
}

/// The record as the admin plane answers it: the profile by name and every
/// worker id with the registered worker's address when one still carries it.
pub fn render(record: &DecisionRecord, engine: &Engine, workers: &[Arc<dyn Worker>]) -> Value {
    let address = |id: WorkerId| {
        workers
            .iter()
            .find(|worker| worker_id(worker.as_ref()) == id)
            .map(|worker| worker.redacted_url().to_owned())
    };
    let worker = |id: WorkerId| {
        json!({
            "worker_id": format!("{:016x}", id.0),
            "worker": address(id),
        })
    };
    let top: Vec<Value> = record.top[..usize::from(record.top_len)]
        .iter()
        .map(|&(id, score)| {
            let mut row = worker(id);
            row["score"] = json!(score);
            row
        })
        .collect();
    let targets: Vec<Value> = record
        .targets
        .iter()
        .map(|target| {
            let mut row = worker(target.id);
            row["position"] = json!(target.position);
            row["dp_rank"] = json!(target.dp_rank);
            if let Some(pair) = target.pair {
                let mut decode = worker(pair.decode_id);
                decode["position"] = json!(pair.decode_position);
                decode["dp_rank"] = json!(pair.decode_dp_rank);
                row["decode"] = decode;
            }
            row
        })
        .collect();
    json!({
        "decision_id": record.decision_id,
        "at_unix_micros": record.at_unix_micros,
        "model": &*record.model,
        "phase": phase_name(record.phase),
        "profile": engine
            .profiles()
            .get(usize::from(record.profile))
            .map(|profile| profile.name.as_str()),
        "outcome": record.outcome,
        "stage": record.stage,
        "survivors": record.survivors,
        "pool_size": record.pool_size,
        "attempt": record.attempt,
        "excluded": record.excluded,
        "seed": record.seed,
        "generation": record.generation,
        "top": top,
        "targets": targets,
    })
}

/// `GET /policy/explain/{id}`: the ring's record of decision `id`. 400 for
/// an id that is not a number; 404 when no engine is hosted or no traced
/// decision carries the id.
pub(crate) fn respond(context: &AppContext, id: &str) -> Response {
    let Ok(id) = id.parse::<u64>() else {
        return route_error::bad_request(
            "invalid_request",
            "the decision id is the number the x-smg-cache-trace response header names",
        );
    };
    let Some(decider) = context.policy_registry.decider() else {
        return route_error::not_found(
            "policy_engine_not_hosted",
            "no policy engine is hosted: the gateway decides through the legacy policies \
             (--policy-engine legacy)",
        );
    };
    let Some(record) = decider.explain(id) else {
        return route_error::not_found(
            "decision_not_found",
            format!(
                "no record of decision {id}: the request did not ask for its trace \
                 (x-smg-cache-trace: 1), or the ring of the last {RING_CAPACITY} traced \
                 decisions has passed it"
            ),
        );
    };
    let workers = context.worker_registry.get_all();
    Json(render(&record, decider.engine(), &workers)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::PolicyConfig,
        observability::cache_trace,
        policies::{
            engine_host::{Decider, HostRequest},
            test_support,
        },
    };

    fn decider() -> Decider {
        let round_robin = PolicyConfig::RoundRobin;
        Decider::from_configs(&round_robin, &round_robin, &round_robin).expect("round robin")
    }

    fn record(id: u64) -> DecisionRecord {
        DecisionRecord {
            decision_id: id,
            at_unix_micros: 0,
            model: Arc::from("m"),
            phase: Phase::Single,
            profile: 0,
            outcome: "pick",
            stage: "round-robin",
            survivors: 1,
            pool_size: 1,
            attempt: 0,
            excluded: 0,
            seed: 0,
            generation: 1,
            top: [(WorkerId(0), 0.0); TRACE_TOP],
            top_len: 0,
            targets: Targets::new(),
        }
    }

    /// A decision taken while the request's capture is in scope is read back
    /// by its own id after later decisions, with its own target; a decision
    /// outside any capture leaves no record.
    #[tokio::test]
    async fn a_traced_decision_is_read_back_by_its_id_and_an_untraced_one_is_not() {
        let decider = decider();
        let workers = test_support::make_workers(&["http://w0:8000", "http://w1:8000"]);
        let request = HostRequest {
            model: "m-explain",
            ..HostRequest::default()
        };
        let (_, untraced) = decider.decide(&request, &workers, &[]);
        assert!(
            decider.explain(untraced.id.0).is_none(),
            "a request that did not ask for its trace leaves no record"
        );

        let ((first, first_lease), (_, second_lease)) = cache_trace::scope(true, async {
            (
                decider.decide(&request, &workers, &[]),
                decider.decide(&request, &workers, &[]),
            )
        })
        .await;
        let record = decider
            .explain(first_lease.id.0)
            .expect("the traced decision is in the ring");
        assert_eq!(record.decision_id, first_lease.id.0);
        assert_eq!(&*record.model, "m-explain");
        assert_eq!(record.outcome, "pick");
        assert_eq!(record.stage, "round-robin");
        assert_eq!(record.survivors, 2);
        assert_eq!(record.pool_size, 2);
        let picked = first.pick().and_then(|pick| pick.targets.first().copied());
        assert_eq!(
            record.targets.first().map(|target| target.id),
            picked.map(|target| target.id),
            "the record carries the first decision's target, not a later one's"
        );
        assert_eq!(
            decider.explain(second_lease.id.0).map(|r| r.decision_id),
            Some(second_lease.id.0)
        );

        let rendered = render(&record, decider.engine(), &workers);
        assert_eq!(rendered["profile"], "default");
        assert_eq!(rendered["phase"], "single");
        let address = picked.map(|target| workers[target.position as usize].url().to_owned());
        assert_eq!(rendered["targets"][0]["worker"], json!(address));
        assert_eq!(rendered["top"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn the_ring_keeps_the_newest_records_up_to_its_capacity() {
        let ring = DecisionRing::new();
        assert!(ring.is_empty());
        for id in 1..=(RING_CAPACITY as u64 + 1) {
            ring.record(record(id));
        }
        assert_eq!(ring.len(), RING_CAPACITY);
        assert!(ring.find(1).is_none(), "the oldest record left at the cap");
        assert_eq!(ring.find(2).map(|r| r.decision_id), Some(2));
        assert_eq!(
            ring.find(RING_CAPACITY as u64 + 1).map(|r| r.decision_id),
            Some(RING_CAPACITY as u64 + 1)
        );
    }
}
