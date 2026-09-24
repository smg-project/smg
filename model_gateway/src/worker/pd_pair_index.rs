//! The compatible prefill/decode pairs of a membership snapshot (#2483).
//!
//! Pairing descriptors ([`super::pd_pairing`]) never change after a worker is
//! built and membership changes rarely, so which decodes a prefill may pair
//! with is decided once per snapshot, here, and shared by every request.
//! Placement then reads a prefill's partner list instead of comparing
//! descriptors on the request path.

use std::sync::Arc;

use super::{
    pd_pairing::{PairingMismatch, PdPairing, PdPairingMode},
    registry::RoutingPool,
    Worker,
};

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
}

/// Why no prefill in a snapshot can pair with any decode: each leg's distinct
/// pairing keys and the components the legs disagreed on, sorted.
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
    /// The prefill pool as the snapshot holds it, for leg verdicts.
    pub prefill_pool: Arc<[Arc<dyn Worker>]>,
    /// The decode pool as the snapshot holds it, for leg verdicts.
    pub decode_pool: Arc<[Arc<dyn Worker>]>,
    /// Prefill workers with at least one compatible decode, in pool order.
    pub prefill: Arc<[Arc<dyn Worker>]>,
    /// Parallel to `prefill`: each worker's compatible decodes, in pool
    /// order. Prefills with equal descriptors share one list.
    pub partners: Vec<Arc<[Arc<dyn Worker>]>>,
    /// Set when both pools have workers and no pair is compatible.
    pub refusal: Option<PairingRefusal>,
}

impl PdPairIndex {
    /// Pair every prefill with its compatible decodes under `mode`.
    pub(crate) fn build(
        prefill_pool: Arc<[Arc<dyn Worker>]>,
        decode_pool: Arc<[Arc<dyn Worker>]>,
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
                    let list: Workers = if mode == PdPairingMode::Off {
                        Arc::clone(&decode_pool)
                    } else {
                        decode_pool
                            .iter()
                            .filter(|d| descriptor.compatible(d.pd_pairing(), mode))
                            .cloned()
                            .collect()
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
                mismatches: pairing_mismatches(&prefill_pool, &decode_pool, mode),
            });
        Self {
            prefill_pool,
            decode_pool,
            prefill: prefill.into(),
            partners,
            refusal,
        }
    }

    /// Whether some prefill can pair at all: both legs have workers and at
    /// least one pair is compatible. Availability is not consulted.
    pub(crate) fn can_pair(&self) -> bool {
        !self.prefill.is_empty()
    }

    /// An index over two empty pools.
    pub(crate) fn empty() -> Self {
        Self::build(Arc::from([]), Arc::from([]), PdPairingMode::Off)
    }
}

/// The distinct pairing keys of one leg, sorted.
fn pairing_keys(leg: &[Arc<dyn Worker>]) -> Vec<String> {
    let mut keys: Vec<String> = leg.iter().map(|w| w.pd_pairing().key()).collect();
    keys.sort();
    keys.dedup();
    keys
}

/// The distinct components on which the legs' descriptors disagree, sorted.
fn pairing_mismatches(
    prefill: &[Arc<dyn Worker>],
    decode: &[Arc<dyn Worker>],
    mode: PdPairingMode,
) -> Vec<String> {
    let mut mismatches: Vec<String> = prefill
        .iter()
        .flat_map(|p| {
            decode
                .iter()
                .filter_map(move |d| p.pd_pairing().mismatch(d.pd_pairing(), mode))
        })
        .map(PairingMismatch::describe)
        .collect();
    mismatches.sort();
    mismatches.dedup();
    mismatches
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, RuntimeType, WorkerType};

    fn worker(url: &str, worker_type: WorkerType, connector: Option<&str>) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
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
            PdPairingMode::Strict,
        );
        assert_eq!(
            urls(&strict.prefill),
            ["grpc://p:nixl", "grpc://p:mooncake"]
        );

        // Off pairs everything: every prefill shares the whole decode pool.
        let off = PdPairIndex::build(prefill, Arc::clone(&decode), PdPairingMode::Off);
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
        let index = PdPairIndex::build(prefill, decode, PdPairingMode::Lenient);
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
        let index = PdPairIndex::build(prefill, decode, PdPairingMode::Lenient);
        assert!(index.prefill.is_empty());
        assert_eq!(
            index.refusal,
            Some(PairingRefusal {
                prefill: vec!["vllm/nixl/?".to_string()],
                decode: vec!["vllm/mooncake/?".to_string()],
                mismatches: vec!["transport".to_string()],
            })
        );
        // A leg with no workers at all is not a refusal.
        assert!(PdPairIndex::empty().refusal.is_none());
    }
}
