//! The `signal_cost` cells: what a decision pays to read a named signal.
//!
//! A cell's document declares one signal and costs each worker by it, so
//! every decision is lent its list's signal cells: a single request's in
//! `signal_cost`, a pair's decode side in `pair_signal_cost`, and the
//! decode side of an encode request decided over its three lists in
//! `encode_signal_cost`. Under
//! such a document each worker's load report states a value of the signal,
//! beside what every cell's report states. Any other cell's document
//! declares no signal, and its reports are left as they were made.

use std::{collections::HashMap, sync::Arc};

use smg_policy_controller::document::PolicySpec;
use smg_worker::{
    identity::basic::Worker,
    worker::{NamedValue, WorkerLoadResponse},
};

/// The named signal the cell's document declares, in tokens.
const SIGNAL: &str = "kv_tokens_held";

/// The `signal_cost` cell's document: the `least_load` preset with another
/// scorer ([`costed`]). The document reads a load report and a signal, so
/// each decision reads its list's signal cells.
pub(super) fn document() -> PolicySpec {
    let mut document = super::preset("least_load");
    costed(&mut document, "least_load");
    document
}

/// The `pair_signal_cost` cell's document: the pair of `least_load`
/// presets with another scorer on its decode side ([`costed`]). The decode
/// side reads a signal and the prefill side none, so each decision reads
/// its decode list's signal cells and no cell of its prefill list.
pub(super) fn pair_document() -> PolicySpec {
    let mut document = super::pair::of("least_load");
    costed(&mut document, "decode");
    document
}

/// The `encode_signal_cost` cell's document: the encode document of
/// `least_load` presets with another scorer on its decode side
/// ([`costed`]). The decode side reads a signal and the two other sides
/// none, so each decision reads its decode list's signal cells and no cell
/// of its prefill list or of its encoders.
pub(super) fn encode_document() -> PolicySpec {
    let mut document = super::encode::of("least_load");
    costed(&mut document, "decode");
    document
}

/// Declares [`SIGNAL`] in `document` and costs each worker of `profile` by
/// it: the tokens its report says wait in its queue and the tokens the
/// report states under the signal.
fn costed(document: &mut PolicySpec, profile: &str) {
    let tokens = serde_json::from_str(r#"{"unit":"tokens"}"#).expect("a signal");
    document.signals.insert(SIGNAL.into(), tokens);
    let cost = format!(
        r#"{{"least-cost":{{"terms":{{"load.waiting_uncached_tokens":1,"signal.{SIGNAL}":1}},
            "unreported":"missing-as-zero"}}}}"#
    );
    let profile = document.profiles.get_mut(profile);
    let profile = profile.expect("the document's profile");
    profile.score = vec![serde_json::from_str(&cost).expect("a scorer")];
}

/// `loads`, the reports of `workers`, with a value of [`SIGNAL`] stated in
/// each where `document` declares the signal, and as they came where it
/// does not.
pub(super) fn stated(
    document: &PolicySpec,
    workers: &[Arc<dyn Worker>],
    mut loads: HashMap<String, WorkerLoadResponse>,
) -> HashMap<String, WorkerLoadResponse> {
    if !document.signals.contains_key(SIGNAL) {
        return loads;
    }
    for (at, worker) in workers.iter().enumerate() {
        let report = loads.get_mut(worker.url()).expect("a report a worker");
        report.loads[0].values.push(NamedValue {
            name: SIGNAL.into(),
            unit: "tokens".into(),
            value: (super::mix(at as u64 * 16 + 5) % 65_536) as f64,
            age_ms: None,
        });
    }
    loads
}
