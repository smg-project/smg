//! A kept decision as JSON, for a host that answers one outside the control
//! call: what the record holds, under the field names of the contract's
//! `DecisionRecord` at every level, each number in the form of its width
//! in the contract.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::DecisionRecord;
use crate::engine::{
    decision::{Decision, LiftedVeto, ShedReason, Target},
    framework::Engine,
    view::WorkerId,
};

/// A shed's reason, in the contract's word for it.
const fn shed_reason(reason: ShedReason) -> &'static str {
    match reason {
        ShedReason::Saturated => "SHED_REASON_SATURATED",
        ShedReason::FlowLimit => "SHED_REASON_FLOW_LIMIT",
        ShedReason::NoEndpoints => "SHED_REASON_NO_ENDPOINTS",
        ShedReason::PolicyDeclined => "SHED_REASON_POLICY_DECLINED",
        ShedReason::Deadline => "SHED_REASON_DEADLINE",
    }
}

/// The veto a pick lifted, in the contract's word for it.
const fn lifted(veto: LiftedVeto) -> &'static str {
    match veto {
        LiftedVeto::None => "LIFTED_VETO_NONE",
        LiftedVeto::Overload => "LIFTED_VETO_OVERLOAD",
        LiftedVeto::Liveness => "LIFTED_VETO_LIVENESS",
    }
}

/// `value` with every field that holds no value left out, at every level.
fn stated(value: Value) -> Value {
    match value {
        Value::Object(fields) => {
            let held = fields.into_iter().filter(|(_, held)| !held.is_null());
            Value::Object(held.map(|(name, held)| (name, stated(held))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(stated).collect()),
        other => other,
    }
}

impl DecisionRecord {
    /// The record as JSON, under the field names of the contract's
    /// `DecisionRecord` at every level. The names of its plan, its profile
    /// and that profile's stages stand where `engine` holds the record's
    /// generation ([`Self::names`]).
    ///
    /// The JSON holds what the record holds. It states neither `request_id`
    /// nor `gateway_id`: the record holds neither, and a caller holds the
    /// id it asked by. A field of the contract that no kept record holds is
    /// left out as well.
    ///
    /// A number is written by the width the contract carries it in. One of
    /// 64 bits is a decimal string, since it does not fit a JSON number: a
    /// worker's id, a seed, a ticket, a generation and the instant. A
    /// narrower one and a score are JSON numbers. An enum is the contract's
    /// word. An option that holds no value is left out, and a zero, a
    /// `false` and an empty list are written. The instant is the wall
    /// clock's, in microseconds since the epoch.
    ///
    /// Three names stand beside the contract's, which has no field for
    /// them. `shadow` at the record is what the binding's shadow said
    /// beside the decision. `url` beside a target's `id` and a top row's
    /// `worker_id`, and `decode_url` beside a pair's `decode_id`, are what
    /// `url` answers for the worker of that id: its host's address for it,
    /// where the host still holds one.
    ///
    /// A record holds no fact of its request: no id, no routing value, no
    /// session key, no header, no prompt and no credential stand in it.
    #[must_use]
    pub fn json(&self, engine: &Engine, url: &dyn Fn(WorkerId) -> Option<String>) -> Value {
        let trace = self.decision.trace();
        let age = Instant::now().saturating_duration_since(self.at);
        let at = SystemTime::now().checked_sub(age);
        let since = at.and_then(|at| at.duration_since(UNIX_EPOCH).ok());
        let micros = since.and_then(|since| i64::try_from(since.as_micros()).ok());
        let names = self.names(engine);
        let names = names.as_ref();
        let target = |target: &Target| {
            json!({
                "id": target.id.0.to_string(),
                "dp_rank": target.dp_rank,
                "url": url(target.id),
                "pair": target.pair.map(|pair| json!({
                    "decode_id": pair.decode_id.0.to_string(),
                    "decode_dp_rank": pair.decode_dp_rank,
                    "decode_url": url(pair.decode_id),
                })),
            })
        };
        let (outcome, held) = match &self.decision {
            Decision::Pick(pick) => {
                let targets: Vec<Value> = pick.targets.iter().map(target).collect();
                ("pick", json!({ "targets": targets }))
            }
            Decision::Shed(shed) => (
                "shed",
                json!({
                    "reason": shed_reason(shed.reason),
                    "emptied_by": shed.emptied_by,
                    "retry_after_secs": shed.retry_after_secs,
                    "wait_allowed": shed.wait_allowed,
                }),
            ),
            Decision::Queue(queue) => (
                "queue",
                json!({
                    "ticket": queue.ticket.to_string(),
                    "max_wait_micros": queue.max_wait_micros,
                }),
            ),
        };
        let top = trace.top.iter().take(usize::from(trace.top_len));
        let top: Vec<Value> = top
            .map(|&(id, score)| {
                json!({ "worker_id": id.0.to_string(), "score": score, "url": url(id) })
            })
            .collect();
        stated(json!({
            "generation": trace.generation.to_string(),
            "binding": self.binding,
            "decision": {
                (outcome): held,
                "trace": {
                    "profile": trace.made_under().unwrap_or_default(),
                    "no_profile": trace.made_under().is_none(),
                    "survivors": trace.survivors.get(),
                    "drew_first": trace.drew_first(),
                    "top": top,
                    "seed": trace.seed.to_string(),
                    "generation": trace.generation.to_string(),
                    "lifted": lifted(trace.lifted),
                },
            },
            "at_unix_micros": micros.map(|micros| micros.to_string()),
            "plan_name": names.map(|names| names.plan),
            "profile_name": names.map(|names| names.profile.as_str()),
            "picker_name": names.map(|names| names.picker.as_str()),
            "filter_names": names.map(|names| names.filters.as_slice()),
            "scorer_names": names.map(|names| names.scorers.as_slice()),
            "shadow": self.shadow.map(|said| said.as_str()),
        }))
    }
}

#[cfg(test)]
mod tests;
