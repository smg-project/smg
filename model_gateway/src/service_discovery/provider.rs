//! The contract between a worker-discovery provider and the shared reconciler.
//!
//! A provider turns its source — Kubernetes Pods today; files, Slurm and Consul
//! next — into these types and nothing else. The reconciler sees only these, so
//! it never learns which kind of source produced a worker beyond the
//! [`DiscoveryKind`] it is handed.

use std::collections::BTreeMap;

use openai_protocol::worker::WorkerType;

use crate::{observability::metrics::metrics_labels, worker::endpoint::Endpoint};

/// Which provider produced a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DiscoveryKind {
    Kubernetes,
}

impl DiscoveryKind {
    /// The value written to the provider label on every worker this provider
    /// registers, and matched to decide which workers it owns.
    ///
    /// A literal rather than the metric label, even though the two agree today:
    /// this is an identity stamped on registered workers, and renaming a metric
    /// must not silently orphan them.
    pub(super) fn as_label(self) -> &'static str {
        match self {
            DiscoveryKind::Kubernetes => "kubernetes",
        }
    }

    /// The `discovery` value on discovery metrics.
    pub(super) fn metric_label(self) -> &'static str {
        match self {
            DiscoveryKind::Kubernetes => metrics_labels::DISCOVERY_KUBERNETES,
        }
    }
}

/// One worker exactly as a provider describes it: the whole contract between a
/// provider and the reconciler.
///
/// Flat on purpose. Which fields count as the worker's *configuration* is
/// decided in [`Self::fingerprint`], which names every field — so a new one
/// cannot be added without choosing, at the hash, whether it belongs to the
/// configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredWorker {
    /// The provider's stable identity for this worker instance, unique within a
    /// snapshot. A different value at an endpoint means a different instance
    /// there, which the reconciler replaces.
    ///
    /// Kubernetes uses `{pod_uid}:{port}`: one Pod can serve several ports, and
    /// a Pod *recreated* at the same address — a rescheduled StatefulSet Pod,
    /// say — gets a new UID. A container restart inside one Pod (a crash, an
    /// OOM kill, a failed liveness probe) keeps the UID and the IP, so it is
    /// the same instance here and is not replaced; recovering from it is the
    /// health checker's job.
    pub(super) discovery_id: String,
    /// Parsed once by the provider, so a bad address is reported against the
    /// record that published it — and held parsed so a derived `Debug` goes
    /// through [`Endpoint`]'s credential masking.
    pub(super) endpoint: Endpoint,
    pub(super) worker_type: WorkerType,
    pub(super) bootstrap_port: Option<u16>,
    pub(super) model_id_override: Option<String>,
    pub(super) kv_connector: Option<String>,
    pub(super) kv_role: Option<String>,
    pub(super) kv_engine_id: Option<String>,
    /// Labels a provider keeps writing for compatibility with what it wrote
    /// before discovery became provider-neutral — Kubernetes' Pod name and UID.
    /// The reconciler writes them verbatim and never interprets them. They are
    /// not configuration, so they do not move the fingerprint.
    pub(super) compat_labels: BTreeMap<String, String>,
}

impl DiscoveredWorker {
    /// A deterministic BLAKE3 fingerprint of the worker's configuration.
    ///
    /// The encoding is explicit rather than derived: versioned, with each field
    /// written as present-or-absent plus a length prefix, so no value can forge
    /// a field boundary and a later change to the projection cannot silently
    /// collide with this one.
    ///
    /// The endpoint is hashed as rendered — scheme included — because the
    /// spelling decides how the worker is registered (`h:p` dual-probes HTTP
    /// and gRPC; `grpc://h:p` does not). Two spellings that parse to the same
    /// endpoint render, and so hash, identically.
    ///
    /// Never log this next to the endpoint: the rendered form can carry
    /// credentials, and the pair would let a reader test guesses offline.
    pub(super) fn fingerprint(&self) -> String {
        // Exhaustive on purpose, with no `..`: adding a field is a compile
        // error here until it is either hashed or explicitly set aside.
        let DiscoveredWorker {
            discovery_id: _,  // identity, not configuration
            compat_labels: _, // provider compatibility, not configuration
            endpoint,
            worker_type,
            bootstrap_port,
            model_id_override,
            kv_connector,
            kv_role,
            kv_engine_id,
        } = self;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"smg-discovered-worker-spec/v1");
        hash_field(&mut hasher, Some(&endpoint.render()));
        hash_field(&mut hasher, Some(&worker_type.to_string()));
        hash_field(
            &mut hasher,
            bootstrap_port.map(|p| p.to_string()).as_deref(),
        );
        hash_field(&mut hasher, model_id_override.as_deref());
        hash_field(&mut hasher, kv_connector.as_deref());
        hash_field(&mut hasher, kv_role.as_deref());
        hash_field(&mut hasher, kv_engine_id.as_deref());
        hasher.finalize().to_hex().to_string()
    }
}

/// Write one field of the fingerprint: a presence byte, then — when present —
/// the value's length and bytes. The presence byte separates absent from
/// empty; the length is what stops a value containing that byte from forging
/// the boundary to the next field.
fn hash_field(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_at(url: &str) -> DiscoveredWorker {
        DiscoveredWorker {
            discovery_id: "id-1".to_string(),
            endpoint: Endpoint::parse_with_rank(url).expect(url).0,
            worker_type: WorkerType::Regular,
            bootstrap_port: None,
            model_id_override: None,
            kv_connector: None,
            kv_role: None,
            kv_engine_id: None,
            compat_labels: BTreeMap::new(),
        }
    }

    #[test]
    fn fingerprint_is_deterministic_and_spelling_insensitive() {
        let a = worker_at("10.0.0.1:8080");
        assert_eq!(a.fingerprint(), a.clone().fingerprint());
        // Two spellings that parse to one endpoint register identically.
        assert_eq!(
            worker_at("[::1]:8080").fingerprint(),
            worker_at("[0:0:0:0:0:0:0:1]:8080").fingerprint()
        );
    }

    /// Every configuration field moves the fingerprint, so a change to any of
    /// them is a change the source made to this worker.
    #[test]
    fn every_configuration_field_moves_the_fingerprint() {
        let base = worker_at("10.0.0.1:8080");
        let variants = [
            DiscoveredWorker {
                endpoint: Endpoint::parse("grpc://10.0.0.1:8080").unwrap(),
                ..base.clone()
            },
            DiscoveredWorker {
                worker_type: WorkerType::Prefill,
                ..base.clone()
            },
            DiscoveredWorker {
                bootstrap_port: Some(9000),
                ..base.clone()
            },
            DiscoveredWorker {
                model_id_override: Some("m".to_string()),
                ..base.clone()
            },
            DiscoveredWorker {
                kv_connector: Some("c".to_string()),
                ..base.clone()
            },
            DiscoveredWorker {
                kv_role: Some("r".to_string()),
                ..base.clone()
            },
            DiscoveredWorker {
                kv_engine_id: Some("e".to_string()),
                ..base.clone()
            },
        ];
        let mut seen = vec![base.fingerprint()];
        for variant in &variants {
            let hash = variant.fingerprint();
            assert!(!seen.contains(&hash), "{variant:?} collided");
            seen.push(hash);
        }
    }

    /// Identity and compatibility labels are set aside in the hash. Otherwise a
    /// new instance at the same address and an edit to the configuration would
    /// be indistinguishable, and a provider's bookkeeping labels could force
    /// replacements.
    #[test]
    fn identity_and_compat_labels_do_not_move_the_fingerprint() {
        let base = worker_at("10.0.0.1:8080");
        let renamed = DiscoveredWorker {
            discovery_id: "id-2".to_string(),
            compat_labels: BTreeMap::from([("k".to_string(), "v".to_string())]),
            ..base.clone()
        };
        assert_eq!(base.fingerprint(), renamed.fingerprint());
    }

    /// The length prefix is what keeps a value from forging a field boundary.
    /// The presence byte alone separates `"ab"|"c"` from `"a"|"bc"`, but not a
    /// value that contains the presence byte itself: without lengths, both of
    /// these encode as `\x01 a \x01 b \x01 c`. Provider input is operator-written
    /// text, so a control byte in it is not hypothetical.
    #[test]
    fn a_value_cannot_forge_a_field_boundary() {
        let left = DiscoveredWorker {
            kv_connector: Some("a\u{1}b".to_string()),
            kv_role: Some("c".to_string()),
            ..worker_at("10.0.0.1:8080")
        };
        let right = DiscoveredWorker {
            kv_connector: Some("a".to_string()),
            kv_role: Some("b\u{1}c".to_string()),
            ..worker_at("10.0.0.1:8080")
        };
        assert_ne!(left.fingerprint(), right.fingerprint());

        // Absent and empty are different answers, which the presence byte
        // (not the length) guarantees.
        let absent = worker_at("10.0.0.1:8080");
        let empty = DiscoveredWorker {
            kv_role: Some(String::new()),
            ..absent.clone()
        };
        assert_ne!(absent.fingerprint(), empty.fingerprint());
    }
}
