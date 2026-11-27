//! `GET /policy/explain/{request_id}`: the newest routing decision Grand
//! Central kept of one request, in the JSON the Grand Central crate writes
//! a kept decision in. The gateway adds each worker's address and writes
//! no other field.
//!
//! The engine keeps a sample of its picks and sheds where the policy spec
//! sets `explain.sample`. A kept decision stands under the id its route
//! stated as the request's own, which is the id in the request's body where
//! the route reads the body and hands that id on. Where the route stated
//! none, it stands under the gateway's own id of the request: the one its
//! response's `x-request-id` header shows. The gateway restates that id on
//! the request under a policy spec that keeps a sample, and the placement
//! reads it off the headers its route hands it
//! (`middleware::request_id`). Of a request that states no id of its own
//! and whose route hands its placement no headers, no decision is kept. An
//! id that finds no record is answered 404 with the reason, in the control
//! plane's error envelope.
//!
//! A decision kept under the gateway's own id may be one its route let go:
//! a request forwarded unread that is handed back to the buffered path
//! behind its pick is placed again, and where its body states an id the
//! second decision stands under that id.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use smg_grand_central::engine::{worker_id, WorkerId};
use smg_worker::identity::basic::Worker;

use super::AppState;
use crate::admin::{json::refusal, params::AdminPath};

/// The gateway does not run Grand Central: the classic policies place
/// its requests, and they keep no decision.
const OFF: (&str, &str) = (
    "GRAND_CENTRAL_OFF",
    "Grand Central is off: no routing decision is kept",
);

/// The engine runs under a policy spec that keeps no decision.
const NO_SAMPLE: (&str, &str) = (
    "EXPLAIN_SAMPLE_OFF",
    "The policy spec keeps no sample: it sets no explain.sample",
);

/// The engine keeps a sample, and no record of it answers the id: the
/// request was not one of the sample, stated another id or none, or its
/// record has left the ring.
const NOT_KEPT: (&str, &str) = (
    "DECISION_NOT_KEPT",
    "No decision of that request id is in the ring",
);

/// The 404 of an id that finds no record, with its code and its reason.
fn not_found((code, reason): (&str, &str)) -> Response {
    refusal(StatusCode::NOT_FOUND, code, reason.to_string())
}

/// Each worker of `workers` by the id the engine knows it by, with the
/// address a request's success path shows for it (`Worker::redacted_url`):
/// a credential in a registered URL is not in it.
fn addresses(workers: &[Arc<dyn Worker>]) -> HashMap<WorkerId, &str> {
    workers
        .iter()
        .map(|worker| (worker_id(worker.as_ref()), worker.redacted_url()))
        .collect()
}

/// `GET /policy/explain/{request_id}`: 200 with the kept decision of the
/// request the id names, or 404 with one of three reasons ([`OFF`],
/// [`NO_SAMPLE`], [`NOT_KEPT`]). This handler refuses no id for its form:
/// an id the path extractor read is looked up as it is given.
///
/// The answer is the Grand Central crate's JSON of the record
/// (`DecisionRecord::json`), under the engine of the policy spec in force.
/// A worker is written with its address ([`addresses`]), read once from
/// the registry. A worker that has left the registry since is written with
/// its id and no address.
///
/// An id is no secret and no proof of who asked. A client may state any
/// text as its request's id, in the body or in a request-id header the
/// gateway reads, and of the records kept under one text the newest is
/// read.
pub(super) async fn get(
    State(state): State<Arc<AppState>>,
    AdminPath(request_id): AdminPath<String>,
) -> Response {
    let Some(engine) = state.context.policy_registry.engine() else {
        return not_found(OFF);
    };
    let spec = engine.spec();
    if spec.engine().explain_sample() == 0 {
        return not_found(NO_SAMPLE);
    }
    let Some(record) = engine.explained(request_id.as_bytes()) else {
        return not_found(NOT_KEPT);
    };
    let workers = state.context.worker_registry.get_all();
    let addresses = addresses(&workers);
    let address = |id: WorkerId| addresses.get(&id).map(|url| (*url).to_string());
    Json(record.json(spec.engine(), &address)).into_response()
}

#[cfg(test)]
#[path = "../../tests/unit/server/policy_explain/tests.rs"]
mod tests;
