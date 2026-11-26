//! The facts a worker object states of its row.

use std::{collections::BTreeMap, sync::Arc};

use smg_worker::{
    identity::basic::Worker,
    pd::pairing::PAIRING_PROTOCOL_LABEL,
    worker::{HealthCheckConfig, WorkerSpec},
};

use super::{membership_writer::Over, object::role};
use crate::{
    engine::view::{threshold_permille, WireModes},
    store::{
        pairing::{PairingDescriptor, LAYOUT_FACTS},
        HealthPolicy, Overload, WorkerFacts,
    },
};

/// The engine a worker's rows stand under: the address its ranks share, or
/// its own, as it was registered. A join base is not that key: it drops an
/// HTTP address's trailing slash.
pub(super) fn engine_of(worker: &dyn Worker) -> &str {
    let spec = &worker.metadata().spec;
    spec.dp_base_url.as_deref().unwrap_or(&spec.url)
}

/// The facts a provider states of a worker it found, from the object to the
/// row as the module's table describes. Facts the object does not carry
/// come from the standing row, or from the store's defaults. The models are
/// the ones the object serves now, and a threshold of its own registration
/// is stated over the standing row's. A threshold it leaves out stays the
/// standing row's: this reading keeps it under every verb. A door that
/// writes a replacement by id ([`Over::Replaced`]) clears it past this
/// reading.
///
/// The pairing descriptor is the standing row's here too. A host that
/// states the worker's own reads it of the worker ([`pairing_descriptor`]).
/// The controller's door states none: its log carries no descriptor. A row
/// that states none pairs by what its labels report, so the row's labels
/// hold every pairing fact the worker's spec holds: its pairing protocol
/// and its KV connector are laid among them.
pub fn facts(worker: &dyn Worker, standing: Option<&WorkerFacts>) -> WorkerFacts {
    let defaults = WorkerFacts::default();
    let standing = standing.unwrap_or(&defaults);
    let metadata = worker.metadata();
    let spec = &metadata.spec;
    let checks = &metadata.health_config;
    WorkerFacts {
        key: spec.url.clone(),
        engine: engine_of(worker).to_owned(),
        incarnation: standing.incarnation,
        role: role(spec.worker_type),
        wire_modes: WireModes::from(spec.connection_mode),
        engine_kind: if spec.runtime_type.is_specified() {
            spec.runtime_type.as_str().to_owned()
        } else {
            String::new()
        },
        models: worker.models().into_iter().map(|card| card.id).collect(),
        // Values wider than a row's fields take the largest value they hold.
        dp_rank: u16::try_from(spec.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX),
        dp_size: u16::try_from(spec.dp_size.unwrap_or(1)).unwrap_or(u16::MAX),
        labels: labels(spec),
        zone: standing.zone.clone(),
        pairing: standing.pairing.clone(),
        encode: standing.encode.clone(),
        vision: standing.vision,
        max_running: standing.max_running,
        kv_block_size: spec
            .kv_block_size
            .map(|size| u32::try_from(size).unwrap_or(u32::MAX)),
        bootstrap_port: spec.bootstrap_port,
        priority: standing.priority,
        reachable_from: standing.reachable_from,
        signals: standing.signals,
        health: HealthPolicy {
            failure_threshold: checks.failure_threshold,
            success_threshold: checks.success_threshold,
            check_interval_ms: milliseconds(checks.check_interval_secs),
            timeout_ms: milliseconds(checks.timeout_secs),
            probes: !checks.disable_health_check,
        },
        overload: {
            let own = thresholds(spec);
            Overload {
                kv_usage_permille: own
                    .kv_usage_permille
                    .or(standing.overload.kv_usage_permille),
                waiting_requests: own.waiting_requests.or(standing.overload.waiting_requests),
            }
        },
        // A window the object's registration left at the default is not a
        // fact of the row: the object built from the row takes the default.
        drain_settle_secs: (checks.drain_settle_secs
            != HealthCheckConfig::default().drain_settle_secs)
            .then(|| u32::try_from(checks.drain_settle_secs).unwrap_or(u32::MAX)),
        demand_facts: standing.demand_facts,
    }
}

/// The label a worker's KV connector is read from where its spec names
/// none.
const KV_CONNECTOR_LABEL: &str = "kv_connector";

/// The labels a row takes of a worker: the worker's own, and among them the
/// two pairing facts its spec may hold, its operator's pairing protocol and
/// its KV connector, each under the label a worker's pairing facts read it
/// from. A worker reads its spec ahead of that label, so the spec's value
/// stands over a label of the name, and a blank protocol on the spec, which
/// a worker reads as none, takes the label out. The object a holder builds
/// of the row holds the labels alone, and offers a partner what this worker
/// offers ([`pairing_descriptor`]).
fn labels(spec: &WorkerSpec) -> BTreeMap<String, String> {
    let mut labels: BTreeMap<String, String> = spec
        .labels
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    match spec.pairing_protocol.as_deref().map(str::trim) {
        Some("") => {
            labels.remove(PAIRING_PROTOCOL_LABEL);
        }
        Some(protocol) => {
            labels.insert(PAIRING_PROTOCOL_LABEL.to_owned(), protocol.to_owned());
        }
        None => {}
    }
    if let Some(connector) = spec.kv_connector.as_deref().filter(|name| !name.is_empty()) {
        labels.insert(KV_CONNECTOR_LABEL.to_owned(), connector.to_owned());
    }
    labels
}

/// The thresholds a worker's own registration states.
fn thresholds(spec: &WorkerSpec) -> Overload {
    Overload {
        kv_usage_permille: spec.overload.token_usage.map(threshold_permille),
        waiting_requests: spec
            .overload
            .waiting_requests
            .map(|waiting| u32::try_from(waiting).unwrap_or(u32::MAX)),
    }
}

/// The facts `built` states at a write that names `over`: [`facts`], and at
/// a replacement by id the thresholds as its own registration states them.
/// A replacement is its maker's whole statement of what a registration can
/// state, so a threshold it leaves out is cleared. An add finds no row to
/// keep one from. An upsert and an update keep the standing row's.
#[cfg_attr(
    not(any(test, feature = "discovery")),
    expect(dead_code, reason = "the replica's own door reads a worker so")
)]
pub(crate) fn stated(
    built: &dyn Worker,
    standing: Option<&WorkerFacts>,
    over: Option<Over<'_>>,
) -> WorkerFacts {
    let facts = facts(built, standing);
    match over {
        Some(Over::Replaced(_)) => WorkerFacts {
            overload: thresholds(&built.metadata().spec),
            ..facts
        },
        Some(Over::Updated(_)) | None => facts,
    }
}

/// The pairing facts `worker` offers a partner, as a row states them: the
/// protocol its operator set, its KV transport, its version, and its KV
/// layout facts in the store's order. The worker derived them once, at its
/// build, and a gateway's own pair index reads the same descriptor. So the
/// fields are mapped here and no label is read: one rule says what a worker
/// reports. A layout fact the store keeps no slot for is left out, since
/// its rule could not read it. `None` where the worker reports nothing, as
/// the rule reads a row with no descriptor.
///
/// A host that tells a worker it keeps no row of states them over
/// [`facts`]' reading of the worker alone.
pub fn pairing_descriptor(worker: &dyn Worker) -> Option<Arc<PairingDescriptor>> {
    let offered = worker.pd_pairing();
    let mut descriptor = PairingDescriptor {
        protocol: offered.explicit().map(str::to_owned),
        transport: offered.transport().map(str::to_owned),
        version: offered.version().map(str::to_owned),
        ..PairingDescriptor::default()
    };
    for (name, value) in offered.kv_layout() {
        let slot = LAYOUT_FACTS.iter().position(|&(_, fact)| fact == name);
        if let Some(slot) = slot {
            descriptor.kv_layout[slot] = Some(value.to_owned());
        }
    }
    descriptor.normalize();
    (descriptor != PairingDescriptor::default()).then(|| Arc::new(descriptor))
}

/// Convert the worker's seconds to the policy's milliseconds, up to the
/// largest interval the row can hold.
fn milliseconds(seconds: u64) -> u32 {
    u32::try_from(seconds.saturating_mul(1_000)).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/found/tests.rs"]
mod tests;
