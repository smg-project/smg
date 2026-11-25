//! `GET /policy/explain/{request_id}`: the newest routing decision Grand
//! Central kept of one request, in the JSON the Grand Central crate writes
//! a kept decision in. The gateway adds each worker's address and writes
//! no other field.
//!
//! The engine keeps a sample of its picks and sheds where the policy spec
//! sets `explain.sample`, each under the id its request states. An id that
//! finds no record is answered 404 with the reason, in the control plane's
//! error envelope.

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
mod tests {
    use super::*;
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    /// A worker is found by the id the engine knows it by and written with
    /// the address a request's success path shows. A registered URL that
    /// carries a credential is written without it.
    #[test]
    fn a_worker_is_written_with_the_address_a_success_path_shows() {
        let plain = "http://127.0.0.1:8001";
        let keyed = "http://user:secret@127.0.0.1:8002";
        let workers = [plain, keyed]
            .map(|url| -> Arc<dyn Worker> { Arc::new(BasicWorkerBuilder::new(url).build()) });
        let found = addresses(&workers);
        assert_eq!(found.len(), 2);
        assert_eq!(found[&worker_id(workers[0].as_ref())], plain);
        let shown = found[&worker_id(workers[1].as_ref())];
        assert_eq!(shown, workers[1].redacted_url());
        assert!(!shown.contains("secret"), "{shown}");
        assert!(shown.contains("127.0.0.1:8002"), "{shown}");
    }
}
