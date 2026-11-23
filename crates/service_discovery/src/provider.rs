//! The contract between a worker-discovery provider and the shared reconciler.
//!
//! A provider turns its source — Kubernetes Pods, a JSON manifest, Slurm jobs
//! or a Consul service — into these types and nothing else. The reconciler
//! sees only these, so it never learns which kind of source produced a worker
//! beyond the [`DiscoveryKind`] it is handed.

use std::collections::BTreeMap;
#[cfg(any(feature = "consul", feature = "slurm"))]
use std::net::Ipv6Addr;

use openai_protocol::worker::WorkerType;
use smg_worker::identity::endpoint::Endpoint;
#[cfg(any(feature = "consul", feature = "slurm"))]
use smg_worker::identity::endpoint::EndpointError;

#[cfg(any(
    test,
    feature = "consul",
    feature = "file",
    feature = "kubernetes",
    feature = "slurm"
))]
use super::series;

/// Which provider produced a snapshot. A kind stands in a build that holds
/// its provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DiscoveryKind {
    #[cfg(any(test, feature = "kubernetes"))]
    Kubernetes,
    #[cfg(any(test, feature = "file"))]
    File,
    #[cfg(any(test, feature = "slurm"))]
    Slurm,
    #[cfg(any(test, feature = "consul"))]
    Consul,
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
            #[cfg(any(test, feature = "kubernetes"))]
            DiscoveryKind::Kubernetes => "kubernetes",
            #[cfg(any(test, feature = "file"))]
            DiscoveryKind::File => "file",
            #[cfg(any(test, feature = "slurm"))]
            DiscoveryKind::Slurm => "slurm",
            #[cfg(any(test, feature = "consul"))]
            DiscoveryKind::Consul => "consul",
        }
    }

    /// The `discovery` value on discovery metrics.
    pub(super) fn metric_label(self) -> &'static str {
        match self {
            #[cfg(any(test, feature = "kubernetes"))]
            DiscoveryKind::Kubernetes => series::DISCOVERY_KUBERNETES,
            #[cfg(any(test, feature = "file"))]
            DiscoveryKind::File => series::DISCOVERY_FILE,
            #[cfg(any(test, feature = "slurm"))]
            DiscoveryKind::Slurm => series::DISCOVERY_SLURM,
            #[cfg(any(test, feature = "consul"))]
            DiscoveryKind::Consul => series::DISCOVERY_CONSUL,
        }
    }
}

/// The worker types a routing mode accepts, for a provider whose records name
/// their own type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Roles {
    /// Regular workers alone.
    Regular,
    /// Prefill and decode workers.
    PrefillDecode,
    /// Encode, prefill and decode workers.
    EncodePrefillDecode,
}

impl Roles {
    #[cfg(any(feature = "consul", feature = "file"))]
    pub(super) fn allows(self, worker_type: WorkerType) -> bool {
        matches!(
            (self, worker_type),
            (Self::Regular, WorkerType::Regular)
                | (
                    Self::PrefillDecode,
                    WorkerType::Prefill | WorkerType::Decode
                )
                | (
                    Self::EncodePrefillDecode,
                    WorkerType::Encode | WorkerType::Prefill | WorkerType::Decode
                )
        )
    }

    /// The type of a record that names none. Regular routing has only one; a
    /// disaggregated record must say which pool it joins.
    #[cfg(any(feature = "consul", feature = "file"))]
    pub(super) fn default_type(self) -> Option<WorkerType> {
        (self == Self::Regular).then_some(WorkerType::Regular)
    }

    #[cfg(any(feature = "consul", feature = "file"))]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::PrefillDecode => "prefill-decode",
            Self::EncodePrefillDecode => "encode-prefill-decode",
        }
    }
}

/// The endpoint of a server on `host` (a name or an address, IPv6 bracketed
/// or not) and `port`, with no scheme, so registration probes for HTTP and
/// gRPC both.
#[cfg(any(feature = "consul", feature = "slurm"))]
pub(super) fn endpoint_at(host: &str, port: u16) -> Result<Endpoint, EndpointError> {
    let authority = if host.parse::<Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    Endpoint::parse(&authority)
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
    /// there, which the reconciler removes and re-registers. The same instance
    /// republished with a different [`Self::fingerprint`] is updated in place.
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
    /// The worker's explicit prefill/decode pairing protocol, where the
    /// provider's source states one. Kubernetes reads it from a Pod label.
    /// The reconciler puts it on the worker spec, where it outranks the
    /// protocol an engine reports about itself.
    pub(super) pairing_protocol: Option<String>,
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
    /// The reconciler stamps this on every worker it registers. When the stamp
    /// on the same instance no longer matches the published record, it
    /// re-registers the record with an Upsert, which updates the worker in
    /// place: no removal and no drain, unlike a new `discovery_id`.
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
            pairing_protocol,
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
        // Written only where stated, under a tag of its own: a worker that
        // states no pairing protocol keeps the fingerprint it had before
        // the field was read, so adding the field re-registers nobody.
        if let Some(protocol) = pairing_protocol {
            hasher.update(b"pairing_protocol");
            hash_field(&mut hasher, Some(protocol));
        }
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

    /// Each provider's kind stamps its own name on a worker and on a series.
    #[test]
    fn each_kind_has_its_own_labels() {
        let kinds = [
            DiscoveryKind::Kubernetes,
            DiscoveryKind::File,
            DiscoveryKind::Slurm,
            DiscoveryKind::Consul,
        ];
        let labels = kinds.map(|kind| (kind.as_label(), kind.metric_label()));
        assert_eq!(
            labels,
            [
                ("kubernetes", "kubernetes"),
                ("file", "file"),
                ("slurm", "slurm"),
                ("consul", "consul"),
            ]
        );
    }

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
            pairing_protocol: None,
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
            DiscoveredWorker {
                pairing_protocol: Some("p".to_string()),
                ..base.clone()
            },
            // A change of the value, not only its arrival: a Pod relabelled
            // in the middle of a rollout updates its worker in place.
            DiscoveredWorker {
                pairing_protocol: Some("q".to_string()),
                ..base.clone()
            },
            // Stated and empty is not the same answer as not stated.
            DiscoveredWorker {
                pairing_protocol: Some(String::new()),
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

    /// A worker that states no pairing protocol keeps the fingerprint it had
    /// before the field was read: these two are the fingerprints of the same
    /// two records on the commit before it. Reading the field therefore
    /// re-registers no worker that does not use it.
    #[test]
    fn a_worker_with_no_pairing_protocol_keeps_its_earlier_fingerprint() {
        assert_eq!(
            worker_at("10.0.0.1:8080").fingerprint(),
            "3a04468c1f7c7f257b3ff4794b6ec558a281805bd5d103b6d71bf6ff85444e8e"
        );
        let full = DiscoveredWorker {
            worker_type: WorkerType::Decode,
            bootstrap_port: Some(9000),
            model_id_override: Some("m".to_string()),
            kv_connector: Some("c".to_string()),
            kv_role: Some("r".to_string()),
            kv_engine_id: Some("e".to_string()),
            ..worker_at("grpc://10.0.0.1:8080")
        };
        assert_eq!(
            full.fingerprint(),
            "d3c6de55f4e2dcf4ac367bb58751b33a2f3b2bc74cf42c6b87a4f30c2870d88c"
        );
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
