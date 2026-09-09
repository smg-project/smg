//! PD pairing descriptors: what a prefill and a decode must agree on for a
//! KV handoff to work (#2483).
//!
//! A rendezvous only succeeds between two workers that share a transport
//! (NIXL vs Mooncake) and a compatible KV layout, and normally an engine
//! version. Each PD worker carries a [`PdPairing`] derived from the labels its
//! engine reports at discovery, or from an explicit `pairing_protocol` the
//! operator sets on the worker; placement pairs a prefill only with a decode
//! whose descriptor is compatible. [`PdPairingMode::Lenient`] refuses only a
//! known difference in transport or KV layout, so a fleet that reports
//! nothing keeps working and a rolling engine upgrade keeps pairing;
//! [`PdPairingMode::Strict`] also refuses unknown components and version
//! differences; [`PdPairingMode::Off`] pairs on nothing.

use std::collections::HashMap;

use openai_protocol::worker::{RuntimeType, WorkerSpec};

pub use crate::config::types::PdPairingMode;

/// Label under which an operator may set the pairing protocol explicitly,
/// alongside the `pairing_protocol` field on the worker spec.
pub const PAIRING_PROTOCOL_LABEL: &str = "pairing_protocol";
/// The Kubernetes annotation flattened into the same label.
pub const PAIRING_PROTOCOL_ANNOTATION_LABEL: &str = "smg.ai/pairing-protocol";

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
    /// Different KV cache layouts (dtype, page size, attention backend).
    KvLayout,
    /// A component one side does not report, rejected under strict mode.
    Unknown(&'static str),
}

/// What one PD worker offers a rendezvous partner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdPairing {
    runtime: RuntimeType,
    explicit: Option<String>,
    transport: Option<String>,
    version: Option<String>,
    kv_layout: Option<String>,
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

    /// One string that names the pairing group: the explicit protocol, else
    /// `runtime/transport/version/layout` with `?` for an unknown component.
    pub fn key(&self) -> String {
        if let Some(explicit) = &self.explicit {
            return explicit.clone();
        }
        let part = |value: &Option<String>| value.as_deref().unwrap_or("?").to_string();
        format!(
            "{}/{}/{}/{}",
            self.runtime.as_str(),
            part(&self.transport),
            part(&self.version),
            part(&self.kv_layout),
        )
    }

    /// Whether the two workers may form a PD pair under `mode`.
    pub fn compatible(&self, other: &Self, mode: PdPairingMode) -> bool {
        self.mismatch(other, mode).is_none()
    }

    /// The first component the two descriptors disagree on, if any.
    ///
    /// Runtime, transport and KV layout are hard facts: a known difference
    /// refuses the pair in every mode but `Off`, an unknown one only under
    /// `Strict`. Version is a soft fact: engines report it unevenly (vLLM's
    /// gRPC server info carries none) and a rolling upgrade legitimately
    /// mixes versions, so it counts only under `Strict`, and only when both
    /// sides report one.
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
        // An explicit protocol is the operator's assertion: it is compared
        // whole and nothing derived is consulted.
        match (&self.explicit, &other.explicit) {
            (Some(a), Some(b)) => return (a != b).then_some(PairingMismatch::Explicit),
            (None, None) => {}
            _ => return unknown("pairing_protocol"),
        }
        let hard_facts = [
            (
                "transport",
                &self.transport,
                &other.transport,
                PairingMismatch::Transport,
            ),
            (
                "kv_layout",
                &self.kv_layout,
                &other.kv_layout,
                PairingMismatch::KvLayout,
            ),
        ];
        for (name, mine, theirs, mismatch) in hard_facts {
            match (mine, theirs) {
                (Some(a), Some(b)) if a != b => return Some(mismatch),
                (Some(_), Some(_)) => {}
                _ => {
                    if let Some(mismatch) = unknown(name) {
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

/// The KV layout facts the engines report: cache dtype, page or block size,
/// attention backend, model dtype. `None` when the worker reports none of
/// them.
fn kv_layout_of(labels: &HashMap<String, String>) -> Option<String> {
    const FACTS: [(&str, &str); 5] = [
        ("kv_cache_dtype", "dtype"),
        ("page_size", "page"),
        ("block_size", "page"),
        ("attention_backend", "attn"),
        ("model_dtype", "model"),
    ];
    let parts: Vec<String> = FACTS
        .iter()
        .filter_map(|(label, short)| {
            non_empty(labels.get(*label))
                .map(|value| format!("{short}={}", value.to_ascii_lowercase()))
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(","))
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
        assert_eq!(
            pairing.key(),
            "vllm/nixl/0.27.1/dtype=auto,page=16,attn=flash_attn,model=torch.bfloat16"
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
        assert_eq!(
            PdPairing::derive(&sglang).key(),
            "sglang/nixl/0.5.18/page=64"
        );
        // TokenSpeed has only Mooncake, so an unreported backend is Mooncake.
        let tokenspeed = spec(RuntimeType::TokenSpeed, &[("version", "0.3.0")]);
        assert_eq!(
            PdPairing::derive(&tokenspeed).key(),
            "tokenspeed/mooncake/0.3.0/?"
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
        let auto = PdPairing::derive(&spec(
            RuntimeType::Sglang,
            &[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "1"),
                ("kv_cache_dtype", "auto"),
            ],
        ));
        assert_eq!(
            fp8.mismatch(&auto, PdPairingMode::Strict),
            Some(PairingMismatch::KvLayout)
        );
        assert!(nixl.compatible(&nixl.clone(), PdPairingMode::Strict));
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
        // Explicit on one side only: the operator's assertion cannot be
        // checked against a derived descriptor.
        let mut asserted = spec(RuntimeType::Vllm, &[("kv_connector", "NixlConnector")]);
        asserted.pairing_protocol = Some("blue".to_string());
        let asserted = PdPairing::derive(&asserted);
        assert!(asserted.compatible(&known, PdPairingMode::Lenient));
        assert_eq!(
            asserted.mismatch(&known, PdPairingMode::Strict),
            Some(PairingMismatch::Unknown("pairing_protocol"))
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
}
