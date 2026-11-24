//! The compatible prefill/decode pairs of a membership snapshot (#2483).
//!
//! Pairing descriptors ([`smg_worker::pd::pairing`]) never change after a worker is
//! built and membership changes rarely, so which decodes a prefill may pair
//! with is decided once per snapshot, here, and shared by every request.
//! Placement then reads a prefill's partner list instead of comparing
//! descriptors on the request path.

use std::sync::{Arc, OnceLock};

use smg_policy_controller::engine::{LentList, ListCell, PartnerClasses};
use smg_worker::{
    identity::basic::Worker,
    pd::pairing::{PairingMismatch, PdPairing, PdPairingMode},
};

use crate::routers::common::{pair_classes, snapshot::RoutingPool};

/// A shared, immutable list of workers, as the routing snapshot holds them.
type Workers = Arc<[Arc<dyn Worker>]>;

/// The wire a PD pair is drawn from: each has its own prefill and decode
/// pools, since the rendezvous rides the transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdWire {
    Grpc,
    Http,
}

impl PdWire {
    pub(crate) const COUNT: usize = 2;

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Grpc => 0,
            Self::Http => 1,
        }
    }

    /// The (prefill, decode) pools of this wire.
    pub(crate) const fn pools(self) -> (RoutingPool, RoutingPool) {
        match self {
            Self::Grpc => (RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode),
            Self::Http => (RoutingPool::HttpPrefill, RoutingPool::HttpDecode),
        }
    }

    /// Whether a pair on this wire must share a runtime. The gRPC wire
    /// builds one request for the pair, in the message type of the prefill's
    /// runtime, and each worker's client sends only its own runtime's type:
    /// a pair of two runtimes cannot be dispatched at all. The HTTP wire
    /// forwards one JSON body to both sides and has no such fact.
    pub(crate) const fn runtime_specific(self) -> bool {
        match self {
            Self::Grpc => true,
            Self::Http => false,
        }
    }

    /// Why `prefill` and `decode` cannot pair on this wire under `mode`: the
    /// wire's own fact first, then the mode's rule over the descriptors
    /// ([`PdPairing::mismatch`]). The pairing mode says how much of the KV
    /// handoff is compared and `Off` compares none of it; a known runtime
    /// difference on a runtime-specific wire is not the handoff's fact but
    /// the wire's, so no mode switches it off. An unknown runtime is not a
    /// different one and pairs as the mode says.
    fn mismatch(
        self,
        prefill: &PdPairing,
        decode: &PdPairing,
        mode: PdPairingMode,
    ) -> Option<PairingMismatch> {
        if self.runtime_specific() && prefill.runtime_differs(decode) {
            return Some(PairingMismatch::Runtime);
        }
        prefill.mismatch(decode, mode)
    }
}

/// Why no prefill in a snapshot can pair with any decode: each pool's distinct
/// pairing keys and the components the pools disagreed on, sorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PairingRefusal {
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
    pub mismatches: Vec<String>,
}

/// One wire's prefill and decode pools with, for every prefill that has at
/// least one compatible decode, that prefill's partners.
#[derive(Debug)]
pub(crate) struct PdPairIndex {
    /// The wire the pairs were made for. Placement reads the wire's facts
    /// from here, so they are the ones the lists were built by.
    pub wire: PdWire,
    /// The prefill pool as the snapshot holds it, for role verdicts.
    pub prefill_pool: Arc<[Arc<dyn Worker>]>,
    /// The decode pool as the snapshot holds it, for role verdicts.
    pub decode_pool: Arc<[Arc<dyn Worker>]>,
    /// Prefill workers with at least one compatible decode, in pool order.
    pub prefill: Arc<[Arc<dyn Worker>]>,
    /// Parallel to `prefill`: each worker's compatible decodes, in pool
    /// order. Prefills with equal descriptors share one list.
    pub partners: Vec<Arc<[Arc<dyn Worker>]>>,
    /// Set when both pools have workers and no pair is compatible.
    pub refusal: Option<PairingRefusal>,
    /// The pair classes of the two pools ([`Self::classes`]): empty until
    /// the policy engine first places a pair over this index.
    classes: OnceLock<PartnerClasses>,
    /// The cells the two pools are lent to the policy engine with
    /// ([`Self::lent`]): the prefill pool's, then the decode pool's.
    cells: [ListCell; 2],
}

impl PdPairIndex {
    /// Pair every prefill with the decodes it can pair with on `wire` under
    /// `mode` (see [`PdWire::mismatch`]). Every placement of the wire reads
    /// these lists, the overload default included, so a pair the wire cannot
    /// carry is handed out by none of them.
    pub(crate) fn build(
        prefill_pool: Arc<[Arc<dyn Worker>]>,
        decode_pool: Arc<[Arc<dyn Worker>]>,
        wire: PdWire,
        mode: PdPairingMode,
    ) -> Self {
        let mut by_descriptor: Vec<(&PdPairing, Workers)> = Vec::new();
        let mut prefill: Vec<Arc<dyn Worker>> = Vec::new();
        let mut partners: Vec<Workers> = Vec::new();
        for worker in prefill_pool.iter() {
            let descriptor = worker.pd_pairing();
            let list = match by_descriptor.iter().find(|(d, _)| *d == descriptor) {
                Some((_, list)) => Arc::clone(list),
                None => {
                    let pairs = |d: &Arc<dyn Worker>| {
                        wire.mismatch(descriptor, d.pd_pairing(), mode).is_none()
                    };
                    // `Off` sorts nothing of the handoff: where the wire
                    // keeps no decode apart either, the pool itself is the
                    // list and none is built.
                    let list: Workers =
                        if mode == PdPairingMode::Off && decode_pool.iter().all(pairs) {
                            Arc::clone(&decode_pool)
                        } else {
                            decode_pool.iter().filter(|d| pairs(d)).cloned().collect()
                        };
                    by_descriptor.push((descriptor, Arc::clone(&list)));
                    list
                }
            };
            if !list.is_empty() {
                prefill.push(Arc::clone(worker));
                partners.push(list);
            }
        }
        let refusal = (prefill.is_empty() && !prefill_pool.is_empty() && !decode_pool.is_empty())
            .then(|| PairingRefusal {
                prefill: pairing_keys(&prefill_pool),
                decode: pairing_keys(&decode_pool),
                mismatches: pairing_mismatches(&prefill_pool, &decode_pool, wire, mode),
            });
        Self {
            wire,
            prefill_pool,
            decode_pool,
            prefill: prefill.into(),
            partners,
            refusal,
            classes: OnceLock::new(),
            cells: Default::default(),
        }
    }

    /// The class of every worker of the two pools, in each pool's order,
    /// and the table of the classes that pair: these partner lists, as the
    /// policy engine's pair decision reads them. Numbered at the first call
    /// and kept with the index, so a membership pays for it once and a
    /// gateway whose pairs no engine places never does. A call that arrives
    /// during the first waits for it.
    pub(crate) fn classes(&self) -> &PartnerClasses {
        self.classes.get_or_init(|| pair_classes::of(self))
    }

    /// The two pools as the policy engine is lent them, the prefill pool
    /// and then the decode pool: each with a cell that stands beside it in
    /// this index. The engine keeps its value of the list in that cell from
    /// the index's first pair on, and the value falls with the index, as
    /// the lists do.
    pub(crate) fn lent(&self) -> [LentList<'_>; 2] {
        let [prefill, decode] = &self.cells;
        let lent = |workers, cell| LentList {
            workers,
            cell,
            facts: None,
        };
        [
            lent(&self.prefill_pool, prefill),
            lent(&self.decode_pool, decode),
        ]
    }

    /// Whether some prefill can pair at all: both pools have workers and at
    /// least one pair is compatible. Availability is not consulted.
    pub(crate) fn can_pair(&self) -> bool {
        !self.prefill.is_empty()
    }

    /// An index over two empty pools of `wire`.
    pub(crate) fn empty(wire: PdWire) -> Self {
        Self {
            wire,
            prefill_pool: Arc::from([]),
            decode_pool: Arc::from([]),
            prefill: Arc::from([]),
            partners: Vec::new(),
            refusal: None,
            classes: OnceLock::new(),
            cells: Default::default(),
        }
    }
}

/// The distinct pairing keys of one pool, sorted.
fn pairing_keys(pool: &[Arc<dyn Worker>]) -> Vec<String> {
    let mut keys: Vec<String> = pool.iter().map(|w| w.pd_pairing().key()).collect();
    keys.sort();
    keys.dedup();
    keys
}

/// The distinct components on which the pools' workers cannot pair, sorted.
fn pairing_mismatches(
    prefill: &[Arc<dyn Worker>],
    decode: &[Arc<dyn Worker>],
    wire: PdWire,
    mode: PdPairingMode,
) -> Vec<String> {
    let mut mismatches: Vec<String> = prefill
        .iter()
        .flat_map(|p| {
            decode
                .iter()
                .filter_map(move |d| wire.mismatch(p.pd_pairing(), d.pd_pairing(), mode))
        })
        .map(PairingMismatch::describe)
        .collect();
    mismatches.sort();
    mismatches.dedup();
    mismatches
}

#[cfg(test)]
mod tests {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::{
        identity::basic::{ConnectionMode, RuntimeType, WorkerType},
        test_support::no_health_check,
    };

    use super::*;
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    fn worker(url: &str, worker_type: WorkerType, connector: Option<&str>) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .health_config(no_health_check());
        if let Some(connector) = connector {
            builder = builder.kv_connector(connector);
        }
        Arc::new(builder.build())
    }

    fn urls(workers: &[Arc<dyn Worker>]) -> Vec<&str> {
        workers.iter().map(|w| w.url()).collect()
    }

    #[test]
    fn each_prefill_lists_the_decodes_it_can_hand_off_to() {
        let prefill: Arc<[Arc<dyn Worker>]> = Arc::from([
            worker("grpc://p:nixl", WorkerType::Prefill, Some("NixlConnector")),
            worker(
                "grpc://p:mooncake",
                WorkerType::Prefill,
                Some("MooncakeConnector"),
            ),
            worker("grpc://p:silent", WorkerType::Prefill, None),
        ]);
        let decode: Arc<[Arc<dyn Worker>]> = Arc::from([
            worker("grpc://d:nixl", WorkerType::Decode, Some("NixlConnector")),
            worker(
                "grpc://d:mooncake",
                WorkerType::Decode,
                Some("MooncakeConnector"),
            ),
        ]);

        let index = PdPairIndex::build(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            PdWire::Grpc,
            PdPairingMode::Lenient,
        );
        assert_eq!(
            urls(&index.prefill),
            ["grpc://p:nixl", "grpc://p:mooncake", "grpc://p:silent"]
        );
        assert_eq!(urls(&index.partners[0]), ["grpc://d:nixl"]);
        assert_eq!(urls(&index.partners[1]), ["grpc://d:mooncake"]);
        // An unknown transport pairs with either side under lenient mode.
        assert_eq!(
            urls(&index.partners[2]),
            ["grpc://d:nixl", "grpc://d:mooncake"]
        );
        assert!(index.refusal.is_none());

        // Strict mode drops the prefill that reports no transport.
        let strict = PdPairIndex::build(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            PdWire::Grpc,
            PdPairingMode::Strict,
        );
        assert_eq!(
            urls(&strict.prefill),
            ["grpc://p:nixl", "grpc://p:mooncake"]
        );

        // Off compares nothing of the handoff: every prefill of this
        // one-runtime fleet shares the whole decode pool.
        let off = PdPairIndex::build(
            prefill,
            Arc::clone(&decode),
            PdWire::Grpc,
            PdPairingMode::Off,
        );
        assert_eq!(off.prefill.len(), 3);
        assert!(off.partners.iter().all(|list| Arc::ptr_eq(list, &decode)));
    }

    #[test]
    fn prefills_with_one_descriptor_share_one_partner_list() {
        let prefill: Arc<[Arc<dyn Worker>]> = Arc::from([
            worker("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            worker("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
        ]);
        let decode: Arc<[Arc<dyn Worker>]> = Arc::from([worker(
            "grpc://d:1",
            WorkerType::Decode,
            Some("NixlConnector"),
        )]);
        let index = PdPairIndex::build(prefill, decode, PdWire::Grpc, PdPairingMode::Lenient);
        assert!(Arc::ptr_eq(&index.partners[0], &index.partners[1]));
    }

    #[test]
    fn a_fleet_without_a_shared_transport_records_the_refusal() {
        let prefill: Arc<[Arc<dyn Worker>]> = Arc::from([worker(
            "grpc://p:1",
            WorkerType::Prefill,
            Some("NixlConnector"),
        )]);
        let decode: Arc<[Arc<dyn Worker>]> = Arc::from([worker(
            "grpc://d:1",
            WorkerType::Decode,
            Some("MooncakeConnector"),
        )]);
        let index = PdPairIndex::build(prefill, decode, PdWire::Grpc, PdPairingMode::Lenient);
        assert!(index.prefill.is_empty());
        assert_eq!(
            index.refusal,
            Some(PairingRefusal {
                prefill: vec!["vllm/nixl/?".to_string()],
                decode: vec!["vllm/mooncake/?".to_string()],
                mismatches: vec!["transport".to_string()],
            })
        );
        // A pool with no workers at all is not a refusal.
        assert!(PdPairIndex::empty(PdWire::Grpc).refusal.is_none());
    }

    fn worker_on(url: &str, worker_type: WorkerType, runtime: RuntimeType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(runtime)
                .health_config(no_health_check())
                .build(),
        )
    }

    /// Three prefills on three runtimes over decodes on two of them.
    fn three_runtimes() -> (Workers, Workers) {
        let prefill = Arc::from([
            worker_on("grpc://p:a", WorkerType::Prefill, RuntimeType::Vllm),
            worker_on("grpc://p:b", WorkerType::Prefill, RuntimeType::Sglang),
            worker_on("grpc://p:c", WorkerType::Prefill, RuntimeType::TokenSpeed),
        ]);
        let decode = Arc::from([
            worker_on("grpc://d:a", WorkerType::Decode, RuntimeType::Vllm),
            worker_on("grpc://d:b", WorkerType::Decode, RuntimeType::Sglang),
        ]);
        (prefill, decode)
    }

    #[test]
    fn pairing_off_keeps_known_runtimes_apart_where_the_wire_is_runtime_specific() {
        let (prefill, decode) = three_runtimes();
        let index = PdPairIndex::build(
            Arc::clone(&prefill),
            Arc::clone(&decode),
            PdWire::Grpc,
            PdPairingMode::Off,
        );
        // Each prefill lists the decodes of its own runtime; the one with no
        // decode on its runtime is in no pair, though the pool has decodes.
        assert_eq!(urls(&index.prefill), ["grpc://p:a", "grpc://p:b"]);
        assert_eq!(urls(&index.partners[0]), ["grpc://d:a"]);
        assert_eq!(urls(&index.partners[1]), ["grpc://d:b"]);
        assert!(index.refusal.is_none());

        // The other wire forwards one body to both sides: off pairs
        // everything there, as it did.
        let http = PdPairIndex::build(
            prefill,
            Arc::clone(&decode),
            PdWire::Http,
            PdPairingMode::Off,
        );
        assert_eq!(http.prefill.len(), 3);
        assert!(http.partners.iter().all(|list| Arc::ptr_eq(list, &decode)));
    }

    #[test]
    fn a_fleet_with_no_shared_runtime_is_refused_under_pairing_off_as_under_lenient() {
        let pools = || -> (Workers, Workers) {
            (
                Arc::from([worker_on(
                    "grpc://p:1",
                    WorkerType::Prefill,
                    RuntimeType::Vllm,
                )]),
                Arc::from([worker_on(
                    "grpc://d:1",
                    WorkerType::Decode,
                    RuntimeType::Sglang,
                )]),
            )
        };
        let build = |wire: PdWire, mode: PdPairingMode| {
            let (prefill, decode) = pools();
            PdPairIndex::build(prefill, decode, wire, mode)
        };
        let off = build(PdWire::Grpc, PdPairingMode::Off);
        assert!(!off.can_pair());
        assert_eq!(
            off.refusal,
            Some(PairingRefusal {
                prefill: vec!["vllm/?/?".to_string()],
                decode: vec!["sglang/?/?".to_string()],
                mismatches: vec!["runtime".to_string()],
            })
        );
        // One cause, one answer: the mode's own refusal of the same fleet.
        assert_eq!(
            off.refusal,
            build(PdWire::Grpc, PdPairingMode::Lenient).refusal
        );
        // Where the wire has no such fact the pair stands under off.
        let http = build(PdWire::Http, PdPairingMode::Off);
        assert!(http.can_pair());
        assert!(http.refusal.is_none());
    }

    #[test]
    fn an_unknown_runtime_pairs_as_the_mode_says() {
        let build = |mode: PdPairingMode| {
            PdPairIndex::build(
                Arc::from([worker_on(
                    "grpc://p:1",
                    WorkerType::Prefill,
                    RuntimeType::Unspecified,
                )]),
                Arc::from([worker_on(
                    "grpc://d:1",
                    WorkerType::Decode,
                    RuntimeType::Vllm,
                )]),
                PdWire::Grpc,
                mode,
            )
        };
        // Unknown is not different: the wire's fact holds nothing apart.
        assert!(build(PdPairingMode::Off).can_pair());
        assert!(build(PdPairingMode::Lenient).can_pair());
        let strict = build(PdPairingMode::Strict);
        assert!(!strict.can_pair());
        assert_eq!(
            strict.refusal.map(|refusal| refusal.mismatches),
            Some(vec!["unknown runtime".to_string()])
        );
    }
}

#[cfg(test)]
#[expect(clippy::print_stdout, reason = "a recorder says where its sheet went")]
mod classic_pair_rows;
