//! PD pairing descriptors: what a prefill and a decode must agree on for a
//! KV handoff to work (#2483).
//!
//! A rendezvous only succeeds between two workers of one runtime that share
//! a KV transport (NIXL vs Mooncake) and a compatible KV layout, and normally
//! an engine version. Each PD worker carries a [`PdPairing`] derived from the
//! labels its engine reports at discovery, or from an explicit
//! `pairing_protocol` the operator sets on the worker; placement pairs a
//! prefill only with a decode whose descriptor is compatible.
//! [`PdPairingMode::Lenient`] refuses only a known difference in runtime,
//! transport or a KV layout fact, so a fleet that reports nothing keeps
//! working and a rolling engine upgrade keeps pairing;
//! [`PdPairingMode::Strict`] also refuses a component one side does not
//! report and a version difference; [`PdPairingMode::Off`] pairs on nothing.

use std::collections::{BTreeMap, HashMap};

use openai_protocol::worker::{RuntimeType, WorkerSpec};

pub use crate::config::types::PdPairingMode;

/// Label under which an operator may set the pairing protocol explicitly,
/// alongside the `pairing_protocol` field on the worker spec.
pub const PAIRING_PROTOCOL_LABEL: &str = "pairing_protocol";
/// The Kubernetes annotation flattened into the same label.
pub const PAIRING_PROTOCOL_ANNOTATION_LABEL: &str = "smg.ai/pairing-protocol";

/// The KV layout facts engines report: the short name used in the pairing
/// key, the canonical label, and alias labels (vLLM's `block_size` is the
/// page size before canonicalisation).
const LAYOUT_FACTS: [(&str, &str, &[&str]); 4] = [
    ("dtype", "kv_cache_dtype", &[]),
    ("page", "page_size", &["block_size"]),
    ("attn", "attention_backend", &[]),
    ("model", "model_dtype", &[]),
];

/// The component two descriptors disagree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingMismatch {
    /// Different runtimes never pair.
    Runtime,
    /// Both sides set an explicit protocol and they differ.
    Explicit,
    /// Different KV transports (NIXL vs Mooncake).
    Transport,
    /// Different engine versions (strict mode only).
    Version,
    /// Both sides report the named KV layout fact and disagree.
    KvLayout(&'static str),
    /// A component one side does not report, rejected under strict mode.
    Unknown(&'static str),
}

impl PairingMismatch {
    /// The component, for logs and the placement failure.
    pub fn describe(self) -> String {
        match self {
            Self::Runtime => "runtime".to_string(),
            Self::Explicit => "pairing_protocol".to_string(),
            Self::Transport => "transport".to_string(),
            Self::Version => "version".to_string(),
            Self::KvLayout(fact) => fact.to_string(),
            Self::Unknown(component) => format!("unknown {component}"),
        }
    }
}

/// What one PD worker offers a rendezvous partner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdPairing {
    runtime: RuntimeType,
    explicit: Option<String>,
    transport: Option<String>,
    version: Option<String>,
    /// The reported KV layout facts by short name (see `LAYOUT_FACTS`).
    kv_layout: BTreeMap<&'static str, String>,
}

impl PdPairing {
    /// Derive the descriptor from a worker's spec and discovered labels.
    pub fn derive(spec: &WorkerSpec) -> Self {
        let labels = &spec.labels;
        let explicit = spec
            .pairing_protocol
            .as_deref()
            .or_else(|| labels.get(PAIRING_PROTOCOL_LABEL).map(String::as_str))
            .or_else(|| {
                labels
                    .get(PAIRING_PROTOCOL_ANNOTATION_LABEL)
                    .map(String::as_str)
            })
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        Self {
            runtime: spec.runtime_type,
            explicit,
            transport: transport_of(spec),
            version: non_empty(labels.get("version")),
            kv_layout: kv_layout_of(labels),
        }
    }

    /// The operator-set protocol, when there is one.
    pub fn explicit(&self) -> Option<&str> {
        self.explicit.as_deref()
    }

    /// The KV transport, when known.
    pub fn transport(&self) -> Option<&str> {
        self.transport.as_deref()
    }

    /// The engine version, when reported.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// The group a worker pairs in: the explicit protocol, else
    /// `runtime/transport/layout` with `?` for an unknown component. Version
    /// is left out: it only counts under strict mode, and a rolling upgrade
    /// would otherwise show two keys for workers that do pair.
    pub fn key(&self) -> String {
        if let Some(explicit) = &self.explicit {
            return explicit.clone();
        }
        let layout: Vec<String> = LAYOUT_FACTS
            .iter()
            .filter_map(|(short, _, _)| {
                self.kv_layout
                    .get(short)
                    .map(|value| format!("{short}={value}"))
            })
            .collect();
        format!(
            "{}/{}/{}",
            self.runtime.as_str(),
            self.transport.as_deref().unwrap_or("?"),
            if layout.is_empty() {
                "?".to_string()
            } else {
                layout.join(",")
            },
        )
    }

    /// Whether the two workers may form a PD pair under `mode`.
    pub fn compatible(&self, other: &Self, mode: PdPairingMode) -> bool {
        self.mismatch(other, mode).is_none()
    }

    /// The first component the two descriptors disagree on, if any.
    ///
    /// Runtime, transport and each KV layout fact are hard facts: a known
    /// difference refuses the pair in every mode but `Off`, a fact one side
    /// reports and the other does not only under `Strict`, and a layout fact
    /// neither side reports is not compared. Version is a soft fact: engines
    /// report it unevenly (vLLM's gRPC server info carries none) and a
    /// rolling upgrade legitimately mixes versions, so it counts only under
    /// `Strict`, and only when both sides report one.
    pub fn mismatch(&self, other: &Self, mode: PdPairingMode) -> Option<PairingMismatch> {
        if mode == PdPairingMode::Off {
            return None;
        }
        let strict = mode == PdPairingMode::Strict;
        let unknown = |name: &'static str| strict.then_some(PairingMismatch::Unknown(name));
        match (self.runtime, other.runtime) {
            (RuntimeType::Unspecified, _) | (_, RuntimeType::Unspecified) => {
                if let Some(mismatch) = unknown("runtime") {
                    return Some(mismatch);
                }
            }
            (mine, theirs) if mine != theirs => return Some(PairingMismatch::Runtime),
            _ => {}
        }
        // Two explicit protocols are the operators' assertion, compared
        // whole. One explicit protocol cannot be checked against derived
        // facts: strict refuses, lenient falls through to the facts rather
        // than pairing blindly.
        match (&self.explicit, &other.explicit) {
            (Some(a), Some(b)) => return (a != b).then_some(PairingMismatch::Explicit),
            (None, None) => {}
            _ => {
                if let Some(mismatch) = unknown("pairing_protocol") {
                    return Some(mismatch);
                }
            }
        }
        match (&self.transport, &other.transport) {
            (Some(a), Some(b)) if a != b => return Some(PairingMismatch::Transport),
            (Some(_), Some(_)) => {}
            _ => {
                if let Some(mismatch) = unknown("transport") {
                    return Some(mismatch);
                }
            }
        }
        for (short, label, _) in LAYOUT_FACTS {
            match (self.kv_layout.get(short), other.kv_layout.get(short)) {
                (Some(a), Some(b)) if a != b => return Some(PairingMismatch::KvLayout(label)),
                (Some(_), Some(_)) | (None, None) => {}
                _ => {
                    if let Some(mismatch) = unknown(label) {
                        return Some(mismatch);
                    }
                }
            }
        }
        if strict {
            if let (Some(mine), Some(theirs)) = (&self.version, &other.version) {
                if mine != theirs {
                    return Some(PairingMismatch::Version);
                }
            }
        }
        None
    }
}

fn non_empty(value: Option<&String>) -> Option<String> {
    value
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The KV transport by runtime: vLLM names a connector, SGLang a transfer
/// backend, TokenSpeed has only Mooncake.
fn transport_of(spec: &WorkerSpec) -> Option<String> {
    let labels = &spec.labels;
    match spec.runtime_type {
        RuntimeType::Vllm => spec
            .kv_connector
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| non_empty(labels.get("kv_connector")))
            .map(|connector| normalise_connector(&connector)),
        RuntimeType::Sglang => {
            non_empty(labels.get("disaggregation_transfer_backend")).map(|s| s.to_ascii_lowercase())
        }
        RuntimeType::TokenSpeed => Some(
            non_empty(labels.get("disaggregation_transfer_backend"))
                .map_or_else(|| "mooncake".to_string(), |s| s.to_ascii_lowercase()),
        ),
        RuntimeType::Trtllm
        | RuntimeType::Mlx
        | RuntimeType::Generic
        | RuntimeType::External
        | RuntimeType::Unspecified => None,
    }
}

/// `NixlConnector` / `MooncakeConnector` (and their store variants) fold to
/// the transport name; anything else is kept lower-cased.
fn normalise_connector(connector: &str) -> String {
    let lower = connector.to_ascii_lowercase();
    if lower.contains("nixl") {
        "nixl".to_string()
    } else if lower.contains("mooncake") {
        "mooncake".to_string()
    } else {
        lower
    }
}

/// The KV layout facts the engine reports, by short name; a fact the labels
/// do not carry is absent.
fn kv_layout_of(labels: &HashMap<String, String>) -> BTreeMap<&'static str, String> {
    LAYOUT_FACTS
        .iter()
        .filter_map(|(short, label, aliases)| {
            std::iter::once(label)
                .chain(aliases.iter())
                .find_map(|name| non_empty(labels.get(*name)))
                .map(|value| (*short, value.to_ascii_lowercase()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(runtime: RuntimeType, labels: &[(&str, &str)]) -> WorkerSpec {
        let mut spec = WorkerSpec::new("grpc://w:1");
        spec.runtime_type = runtime;
        spec.labels = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        spec
    }

    #[test]
    fn vllm_derives_transport_from_the_connector_and_layout_from_the_labels() {
        let mut s = spec(
            RuntimeType::Vllm,
            &[
                ("version", "0.27.1"),
                ("kv_cache_dtype", "auto"),
                ("block_size", "16"),
                ("attention_backend", "FLASH_ATTN"),
                ("model_dtype", "torch.bfloat16"),
            ],
        );
        s.kv_connector = Some("NixlConnector".to_string());
        let pairing = PdPairing::derive(&s);
        assert_eq!(pairing.transport(), Some("nixl"));
        assert_eq!(pairing.version(), Some("0.27.1"));
        assert_eq!(
            pairing.key(),
            "vllm/nixl/dtype=auto,page=16,attn=flash_attn,model=torch.bfloat16"
        );
        // The connector may also arrive as a discovered label.
        let s = spec(
            RuntimeType::Vllm,
            &[("kv_connector", "MooncakeStoreConnector")],
        );
        assert_eq!(PdPairing::derive(&s).transport(), Some("mooncake"));
    }

    #[test]
    fn sglang_and_tokenspeed_derive_transport_from_the_transfer_backend() {
        let sglang = spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "0.5.18"),
                ("page_size", "64"),
            ],
        );
        assert_eq!(PdPairing::derive(&sglang).key(), "sglang/nixl/page=64");
        // TokenSpeed has only Mooncake, so an unreported backend is Mooncake.
        let tokenspeed = spec(RuntimeType::TokenSpeed, &[("version", "0.3.0")]);
        assert_eq!(
            PdPairing::derive(&tokenspeed).key(),
            "tokenspeed/mooncake/?"
        );
    }

    #[test]
    fn an_explicit_protocol_is_the_whole_key() {
        let mut s = spec(RuntimeType::Vllm, &[("version", "0.27.1")]);
        s.pairing_protocol = Some("blue".to_string());
        assert_eq!(PdPairing::derive(&s).key(), "blue");
        let labelled = spec(
            RuntimeType::Vllm,
            &[("smg.ai/pairing-protocol", "green"), ("version", "0.1.0")],
        );
        assert_eq!(PdPairing::derive(&labelled).key(), "green");
    }

    #[test]
    fn transports_versions_and_layouts_must_match_when_both_are_known() {
        let nixl = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "1"),
                ("kv_cache_dtype", "auto"),
            ],
        ));
        let mooncake = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "mooncake"),
                ("version", "1"),
                ("kv_cache_dtype", "auto"),
            ],
        ));
        assert_eq!(
            nixl.mismatch(&mooncake, PdPairingMode::Lenient),
            Some(PairingMismatch::Transport)
        );
        let older = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "0"),
                ("kv_cache_dtype", "auto"),
            ],
        ));
        // A version difference is tolerated leniently (rolling upgrades) and
        // refused strictly.
        assert!(nixl.compatible(&older, PdPairingMode::Lenient));
        assert_eq!(
            nixl.mismatch(&older, PdPairingMode::Strict),
            Some(PairingMismatch::Version)
        );
        let fp8 = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "1"),
                ("kv_cache_dtype", "fp8_e5m2"),
            ],
        ));
        assert_eq!(
            fp8.mismatch(&nixl, PdPairingMode::Lenient),
            Some(PairingMismatch::KvLayout("kv_cache_dtype"))
        );
        assert_eq!(
            fp8.mismatch(&nixl, PdPairingMode::Strict),
            Some(PairingMismatch::KvLayout("kv_cache_dtype"))
        );
        assert!(nixl.compatible(&nixl.clone(), PdPairingMode::Strict));
    }

    #[test]
    fn a_layout_fact_one_side_omits_is_unknown_and_one_neither_reports_is_not_a_fact() {
        let requested = PdPairing::derive(&spec(
            RuntimeType::Vllm,
            &[
                ("kv_connector", "NixlConnector"),
                ("kv_cache_dtype", "auto"),
                ("attention_backend", "FLASH_ATTN"),
            ],
        ));
        // The auto path reports no attention backend.
        let auto = PdPairing::derive(&spec(
            RuntimeType::Vllm,
            &[
                ("kv_connector", "NixlConnector"),
                ("kv_cache_dtype", "auto"),
            ],
        ));
        assert!(requested.compatible(&auto, PdPairingMode::Lenient));
        assert_eq!(
            requested.mismatch(&auto, PdPairingMode::Strict),
            Some(PairingMismatch::Unknown("attention_backend"))
        );
        // Neither side reports the backend: nothing to compare, even strictly.
        assert!(auto.compatible(&auto.clone(), PdPairingMode::Strict));
    }

    #[test]
    fn unknown_components_pair_leniently_and_fail_strictly() {
        let known = PdPairing::derive(&spec(
            RuntimeType::Vllm,
            &[("kv_connector", "NixlConnector"), ("version", "0.27.1")],
        ));
        let silent = PdPairing::derive(&spec(RuntimeType::Vllm, &[]));
        assert!(known.compatible(&silent, PdPairingMode::Lenient));
        assert_eq!(
            known.mismatch(&silent, PdPairingMode::Strict),
            Some(PairingMismatch::Unknown("transport"))
        );
        // Explicit on one side only: strict refuses, lenient still compares
        // the derived facts, so a known transport difference is refused.
        let mut asserted = spec(RuntimeType::Vllm, &[("kv_connector", "NixlConnector")]);
        asserted.pairing_protocol = Some("blue".to_string());
        let asserted = PdPairing::derive(&asserted);
        assert!(asserted.compatible(&known, PdPairingMode::Lenient));
        assert_eq!(
            asserted.mismatch(&known, PdPairingMode::Strict),
            Some(PairingMismatch::Unknown("pairing_protocol"))
        );
        let mooncake = PdPairing::derive(&spec(
            RuntimeType::Vllm,
            &[("kv_connector", "MooncakeConnector")],
        ));
        assert_eq!(
            asserted.mismatch(&mooncake, PdPairingMode::Lenient),
            Some(PairingMismatch::Transport)
        );
        let other = PdPairing::derive(&{
            let mut s = spec(RuntimeType::Vllm, &[]);
            s.pairing_protocol = Some("red".to_string());
            s
        });
        assert_eq!(
            asserted.mismatch(&other, PdPairingMode::Lenient),
            Some(PairingMismatch::Explicit)
        );
    }

    #[test]
    fn different_runtimes_never_pair_but_an_undetected_one_is_unknown() {
        let sglang = PdPairing::derive(&spec(RuntimeType::Sglang, &[]));
        let vllm = PdPairing::derive(&spec(RuntimeType::Vllm, &[]));
        assert_eq!(
            sglang.mismatch(&vllm, PdPairingMode::Lenient),
            Some(PairingMismatch::Runtime)
        );
        // A worker whose runtime probe failed has an unknown runtime, not a
        // different one.
        let undetected = PdPairing::derive(&spec(RuntimeType::Unspecified, &[]));
        assert!(vllm.compatible(&undetected, PdPairingMode::Lenient));
        assert_eq!(
            vllm.mismatch(&undetected, PdPairingMode::Strict),
            Some(PairingMismatch::Unknown("runtime"))
        );
    }

    #[test]
    fn off_pairs_anything() {
        let nixl = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[("disaggregation_transfer_backend", "nixl")],
        ));
        let mooncake = PdPairing::derive(&spec(
            RuntimeType::Vllm,
            &[("kv_connector", "MooncakeConnector")],
        ));
        assert_eq!(
            nixl.mismatch(&mooncake, PdPairingMode::Lenient),
            Some(PairingMismatch::Runtime)
        );
        assert!(nixl.compatible(&mooncake, PdPairingMode::Off));
    }

    #[test]
    fn a_mismatch_describes_its_component() {
        assert_eq!(PairingMismatch::Transport.describe(), "transport");
        assert_eq!(
            PairingMismatch::KvLayout("page_size").describe(),
            "page_size"
        );
        assert_eq!(
            PairingMismatch::Unknown("attention_backend").describe(),
            "unknown attention_backend"
        );
    }
}
