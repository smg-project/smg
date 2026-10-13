//! PD pairing descriptors: what a prefill and a decode must agree on for a
//! KV handoff to work (#2483).
//!
//! A rendezvous only succeeds between two workers of one runtime that share
//! a KV transport (NIXL vs Mooncake) and a compatible KV layout, and normally
//! an engine version. Each PD worker carries a [`PdPairing`] derived from the
//! labels its engine reports at discovery, or from an explicit
//! `pairing_protocol` the operator sets: on the worker spec, as a worker
//! label, or in the engine's own environment (`SMG_PAIRING_PROTOCOL`, which
//! the gRPC servicers report in server info and so reaches the same label).
//! Placement pairs a prefill only with a decode whose descriptor is
//! compatible.
//! [`PdPairingMode::Lenient`] refuses only a known difference in runtime,
//! transport or a KV layout fact, so a fleet that reports nothing keeps
//! working and a rolling engine upgrade keeps pairing;
//! [`PdPairingMode::Strict`] also refuses a component one side does not
//! report and a version difference; [`PdPairingMode::Off`] pairs on nothing.

use std::collections::{BTreeMap, HashMap};

use openai_protocol::worker::{RuntimeType, WorkerSpec};

use super::MORIIO_MODE_LABEL;
pub use crate::config::types::PdPairingMode;

/// Label under which the pairing protocol arrives when it is not on the
/// worker spec: set by the operator as a worker label, or reported by the
/// engine's servicer from its `SMG_PAIRING_PROTOCOL` environment. A worker
/// label wins over the reported value, since config labels are merged over
/// discovered ones.
pub const PAIRING_PROTOCOL_LABEL: &str = "pairing_protocol";

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

/// How a KV transport moves blocks between the legs of a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTransferProfile {
    /// A rendezvous or side channel per pair: the prefill hands its blocks
    /// to the one decode the request names (a bootstrap address, a peer's
    /// side channel, a transfer id), so the gateway tags the legs and probes
    /// the port the partners meet on.
    P2p,
    /// A store the deployment shares, keyed by prefix hash: the prefill
    /// writes its blocks to the store and whichever decode follows reads
    /// them back. Nothing is agreed per pair and nothing listens for a
    /// partner, so the gateway mints nothing: it tags no leg, hands no
    /// bootstrap port and probes none; it only runs the decode after the
    /// prefill. Whether the blocks arrive is the store's business.
    Store,
}

/// How the two sides of a would-be pair report one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Agreement {
    /// Both report it, the same.
    Agree,
    /// Both report it, differently.
    Differ,
    /// One side reports it, the other does not.
    OneSided,
    /// Neither reports it.
    Unreported,
}

impl Agreement {
    fn of(mine: Option<&str>, theirs: Option<&str>) -> Self {
        match (mine, theirs) {
            (Some(a), Some(b)) if a == b => Self::Agree,
            (Some(_), Some(_)) => Self::Differ,
            (None, None) => Self::Unreported,
            _ => Self::OneSided,
        }
    }

    /// The verdict on a hard fact: a known difference is `differ`; a side
    /// that does not report the fact leaves it unknown, which only `Strict`
    /// refuses; a fact nobody reports is what `unreported` says of the
    /// component.
    fn hard_fact(
        self,
        differ: PairingMismatch,
        name: &'static str,
        strict: bool,
        unreported: Unreported,
    ) -> Option<PairingMismatch> {
        let unknown = strict.then_some(PairingMismatch::Unknown(name));
        match self {
            Self::Differ => Some(differ),
            Self::OneSided => unknown,
            Self::Unreported if unreported == Unreported::Unknown => unknown,
            Self::Agree | Self::Unreported => None,
        }
    }
}

/// What it means when neither side of a pair reports a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unreported {
    /// The component always exists (a runtime, a transport): two workers
    /// that name neither leave it unknown, which only `Strict` refuses.
    Unknown,
    /// The engines may not carry the fact (a KV layout fact): nothing to
    /// compare.
    NotAFact,
}

/// What one PD worker offers a rendezvous partner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdPairing {
    runtime: RuntimeType,
    explicit: Option<String>,
    transport: Option<String>,
    profile: Option<KvTransferProfile>,
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
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let transport = transport_of(spec);
        let profile = profile_of(spec, transport.is_some());
        Self {
            runtime: spec.runtime_type,
            explicit,
            transport,
            profile,
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

    /// How the transport moves blocks between the legs, when the transport
    /// is known.
    pub fn profile(&self) -> Option<KvTransferProfile> {
        self.profile
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
        // The hard facts, judged in this order by [`Agreement::hard_fact`].
        // A worker whose runtime probe failed has an unknown runtime, not a
        // different one.
        let runtime = |runtime: RuntimeType| runtime.is_specified().then(|| runtime.as_str());
        if let Some(mismatch) = Agreement::of(runtime(self.runtime), runtime(other.runtime))
            .hard_fact(
                PairingMismatch::Runtime,
                "runtime",
                strict,
                Unreported::Unknown,
            )
        {
            return Some(mismatch);
        }
        // Two explicit protocols are the operators' assertion, compared
        // whole: equal ones settle the pair. One explicit protocol cannot be
        // checked against derived facts: strict refuses, lenient falls
        // through to the facts rather than pairing blindly.
        match Agreement::of(self.explicit.as_deref(), other.explicit.as_deref()) {
            Agreement::Agree => return None,
            Agreement::Differ => return Some(PairingMismatch::Explicit),
            Agreement::OneSided if strict => {
                return Some(PairingMismatch::Unknown("pairing_protocol"))
            }
            Agreement::OneSided | Agreement::Unreported => {}
        }
        if let Some(mismatch) = Agreement::of(self.transport.as_deref(), other.transport.as_deref())
            .hard_fact(
                PairingMismatch::Transport,
                "transport",
                strict,
                Unreported::Unknown,
            )
        {
            return Some(mismatch);
        }
        for (short, label, _) in LAYOUT_FACTS {
            let agreement = Agreement::of(
                self.kv_layout.get(short).map(String::as_str),
                other.kv_layout.get(short).map(String::as_str),
            );
            if let Some(mismatch) = agreement.hard_fact(
                PairingMismatch::KvLayout(label),
                label,
                strict,
                Unreported::NotAFact,
            ) {
                return Some(mismatch);
            }
        }
        // The soft fact: a version difference counts only under `Strict`, and
        // only when both sides report one.
        let versions = Agreement::of(self.version.as_deref(), other.version.as_deref());
        (strict && versions == Agreement::Differ).then_some(PairingMismatch::Version)
    }
}

fn non_empty(value: Option<&String>) -> Option<String> {
    value
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The connector a vLLM worker names: on its spec, else as a discovered label.
fn vllm_connector(spec: &WorkerSpec) -> Option<String> {
    spec.kv_connector
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| non_empty(spec.labels.get("kv_connector")))
}

/// The transfer profile by runtime: a vLLM connector's name says whether it
/// writes to a store (see [`connector_profile`]); the transfer backends of
/// SGLang and TokenSpeed all rendezvous per pair. Unknown when the transport
/// is.
fn profile_of(spec: &WorkerSpec, transport_known: bool) -> Option<KvTransferProfile> {
    match spec.runtime_type {
        RuntimeType::Vllm => vllm_connector(spec).map(|connector| connector_profile(&connector)),
        RuntimeType::Sglang | RuntimeType::TokenSpeed => {
            transport_known.then_some(KvTransferProfile::P2p)
        }
        RuntimeType::Trtllm
        | RuntimeType::Mlx
        | RuntimeType::Generic
        | RuntimeType::External
        | RuntimeType::Unspecified => None,
    }
}

/// The connectors whose legs meet in a store the deployment shares, by
/// exact name or family prefix: the Mooncake store, the shared-storage
/// connector, the FlexKV and 3FS families. Connectors whose blocks stay in
/// the engine (offloading) or whose disaggregated mode is a per-pair push
/// the request has to address (the LMCache family) are not listed: they
/// keep the unknown-connector handling.
const STORE_CONNECTORS: [&str; 2] = ["MooncakeStoreConnector", "SharedStorageConnector"];
const STORE_CONNECTOR_FAMILIES: [&str; 2] = ["FlexKV", "HF3FS"];

/// The profile a connector's name implies. A name not in the store table
/// is a rendezvous connector: what every P2P connector is, and what the
/// request path's passthrough handling of an unknown connector assumes.
pub fn connector_profile(connector: &str) -> KvTransferProfile {
    let connector = connector.trim();
    if STORE_CONNECTORS.contains(&connector)
        || STORE_CONNECTOR_FAMILIES
            .iter()
            .any(|family| connector.starts_with(family))
    {
        KvTransferProfile::Store
    } else {
        KvTransferProfile::P2p
    }
}

/// The KV transport by runtime: vLLM names a connector, SGLang a transfer
/// backend, TokenSpeed has only Mooncake.
fn transport_of(spec: &WorkerSpec) -> Option<String> {
    let labels = &spec.labels;
    match spec.runtime_type {
        RuntimeType::Vllm => vllm_connector(spec)
            .map(|connector| normalise_connector(&connector))
            // MoRI-IO READ and WRITE engines cannot hand KV to each other.
            .map(|transport| {
                match (transport.as_str(), non_empty(labels.get(MORIIO_MODE_LABEL))) {
                    ("moriio", Some(mode)) => format!("moriio-{}", mode.to_ascii_lowercase()),
                    _ => transport,
                }
            }),
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

/// The transports a connector name may carry: `NixlConnector`,
/// `MooncakeConnector` and `MoRIIOConnector`, with their variants, name the
/// transport they contain.
const CONNECTOR_TRANSPORTS: [&str; 3] = ["nixl", "mooncake", "moriio"];

/// A connector name folded to its transport, and a store variant to
/// `<transport>-store`: legs that meet in a store cannot hand KV to legs
/// that rendezvous, so the two never pair. A name that carries no known
/// transport is kept lower-cased, so two workers naming the same unknown
/// connector still pair.
fn normalise_connector(connector: &str) -> String {
    let lower = connector.to_ascii_lowercase();
    let Some(family) = CONNECTOR_TRANSPORTS
        .iter()
        .find(|transport| lower.contains(*transport))
    else {
        return lower;
    };
    match connector_profile(connector) {
        KvTransferProfile::P2p => (*family).to_string(),
        KvTransferProfile::Store => format!("{family}-store"),
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
        assert_eq!(PdPairing::derive(&s).transport(), Some("mooncake-store"));
    }

    /// The connector's name says how its legs meet: the P2P connectors
    /// rendezvous per pair, the store connectors meet in a shared store; the
    /// other runtimes' transfer backends all rendezvous, and a worker with no
    /// known transport has no profile.
    #[test]
    fn a_connectors_name_says_whether_its_legs_rendezvous_or_meet_in_a_store() {
        let vllm = |connector: &str| {
            let mut s = spec(RuntimeType::Vllm, &[]);
            s.kv_connector = Some(connector.to_string());
            PdPairing::derive(&s)
        };
        for (connector, profile) in [
            ("MooncakeConnector", KvTransferProfile::P2p),
            ("NixlConnector", KvTransferProfile::P2p),
            ("NixlPullConnector", KvTransferProfile::P2p),
            ("MoRIIOConnector", KvTransferProfile::P2p),
            ("P2pNcclConnector", KvTransferProfile::P2p),
            ("MooncakeStoreConnector", KvTransferProfile::Store),
            ("SharedStorageConnector", KvTransferProfile::Store),
            ("FlexKVConnectorV1", KvTransferProfile::Store),
            ("HF3FSKVConnector", KvTransferProfile::Store),
            // Their blocks stay in the engine, or their disaggregated mode is
            // a per-pair push the request has to address: not a shared store.
            ("OffloadingConnector", KvTransferProfile::P2p),
            ("SimpleCPUOffloadConnector", KvTransferProfile::P2p),
            ("LMCacheConnectorV1", KvTransferProfile::P2p),
            ("LMCacheMPConnector", KvTransferProfile::P2p),
        ] {
            assert_eq!(connector_profile(connector), profile, "{connector}");
            assert_eq!(vllm(connector).profile(), Some(profile), "{connector}");
        }
        let sglang = spec(
            RuntimeType::Sglang,
            &[("disaggregation_transfer_backend", "mooncake")],
        );
        assert_eq!(
            PdPairing::derive(&sglang).profile(),
            Some(KvTransferProfile::P2p)
        );
        assert_eq!(
            PdPairing::derive(&spec(RuntimeType::TokenSpeed, &[])).profile(),
            Some(KvTransferProfile::P2p)
        );
        assert_eq!(
            PdPairing::derive(&spec(RuntimeType::Vllm, &[])).profile(),
            None
        );
        assert_eq!(
            PdPairing::derive(&spec(RuntimeType::Sglang, &[])).profile(),
            None
        );
    }

    /// A store variant of a rendezvous connector family is its own transport:
    /// a store leg and a rendezvous leg cannot hand KV to each other, so they
    /// never pair in a mode that compares transports.
    #[test]
    fn a_store_connector_is_its_own_transport_and_never_pairs_with_a_rendezvous_leg() {
        let vllm = |connector: &str| {
            let mut s = spec(RuntimeType::Vllm, &[]);
            s.kv_connector = Some(connector.to_string());
            PdPairing::derive(&s)
        };
        let store = vllm("MooncakeStoreConnector");
        let rendezvous = vllm("MooncakeConnector");
        assert_eq!(store.transport(), Some("mooncake-store"));
        assert_eq!(store.key(), "vllm/mooncake-store/?");
        assert_eq!(rendezvous.transport(), Some("mooncake"));
        for mode in [PdPairingMode::Lenient, PdPairingMode::Strict] {
            assert_eq!(
                store.mismatch(&rendezvous, mode),
                Some(PairingMismatch::Transport)
            );
            assert_eq!(
                rendezvous.mismatch(&store, mode),
                Some(PairingMismatch::Transport)
            );
            assert!(store.compatible(&store.clone(), mode));
        }
        assert!(store.compatible(&rendezvous, PdPairingMode::Off));
        // A connector outside the Mooncake and NIXL families keeps its own
        // name, as before.
        assert_eq!(
            vllm("SharedStorageConnector").transport(),
            Some("sharedstorageconnector")
        );
    }

    #[test]
    fn moriio_read_and_write_engines_never_pair() {
        let moriio = |mode: Option<&str>| {
            let labels: Vec<(&str, &str)> = mode.map(|m| ("moriio_mode", m)).into_iter().collect();
            let mut s = spec(RuntimeType::Vllm, &labels);
            s.kv_connector = Some("MoRIIOConnector".to_string());
            PdPairing::derive(&s)
        };
        let (read, write, unknown) = (moriio(Some("read")), moriio(Some("WRITE")), moriio(None));
        assert_eq!(read.transport(), Some("moriio-read"));
        assert_eq!(write.transport(), Some("moriio-write"));
        assert_eq!(unknown.transport(), Some("moriio"));
        assert_eq!(
            read.mismatch(&write, PdPairingMode::Lenient),
            Some(PairingMismatch::Transport)
        );
        assert!(read.compatible(&moriio(Some("Read")), PdPairingMode::Strict));
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
        // The label path, which both a worker label and the engine's
        // SMG_PAIRING_PROTOCOL environment (via server info) arrive through.
        let labelled = spec(
            RuntimeType::Vllm,
            &[("pairing_protocol", " green "), ("version", "0.1.0")],
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
