//! Worker placement shared by the HTTP and gRPC families.
//!
//! Every family used to carry its own copy of the same sequences: take the
//! routing pool for the model, narrow it to the retained wire on a retry,
//! drop unavailable workers unless the policy does that itself, hand the
//! survivors to the policy registry, record the selection; and for a
//! disaggregated request, do that per leg from one snapshot under one
//! runtime. This module is those sequences written once. A family still
//! decides which pools it draws from and what a failure means on its wire.
//!
//! Distinct from [`super::worker_selection`], the least-load selector with
//! refresh-on-miss that the provider and realtime paths use; this is the
//! policy-registry path over the self-hosted routing pools.

use std::sync::Arc;

use axum::{http::HeaderMap, response::Response};
use tracing::{debug, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{
        policy_filters_unavailable_workers, CacheNamespace, PolicyRegistry, SelectWorkerInfo,
        WorkerLeg,
    },
    routers::common::overload,
    worker::{ConnectionMode, ConnectionModeExt, RoutingPool, RuntimeType, Worker, WorkerRegistry},
};

/// The wire a retained plan was built for. Retry re-selection filters
/// candidates to this (runtime, transport): the plan's proto flavor and its
/// stop-resolution are wire-specific and cannot be rebuilt post-drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WireConstraint {
    pub runtime: RuntimeType,
    pub connection: ConnectionMode,
}

/// Everything a single-worker placement reads from the request.
#[derive(Clone, Copy, Default)]
pub(crate) struct PlacementInputs<'a> {
    /// Request text for cache-aware routing.
    pub text: Option<&'a str>,
    /// Tokenized request, or a valid routing-tokens hint, for prefix hashing.
    pub tokens: Option<&'a [u32]>,
    /// Request headers for header-based policies and the routing-key hint.
    pub headers: Option<&'a HeaderMap>,
    /// Session key derived from the body's `rid`.
    pub rid_key: Option<&'a str>,
    /// The request's cache partition, when set.
    pub cache_namespace: Option<CacheNamespace>,
}

/// The pool a placement draws from, before the availability filter.
///
/// Borrowed straight from the registry when no wire is pinned, so the hot
/// path does not clone the snapshot per request.
pub(crate) enum Candidates {
    Shared(Arc<[Arc<dyn Worker>]>),
    Pinned(Vec<Arc<dyn Worker>>),
}

impl Candidates {
    pub(crate) fn as_slice(&self) -> &[Arc<dyn Worker>] {
        match self {
            Self::Shared(shared) => shared,
            Self::Pinned(pinned) => pinned,
        }
    }
}

/// Why a placement produced nothing, judged from exactly the pool selection
/// drew from.
pub(crate) enum PlacementFailure {
    /// Nothing serves the model on this wire.
    NoCandidates,
    /// Every candidate vetoed the request as overloaded; carries the shed.
    AllOverloaded(Response),
    /// Candidates exist but none is available (health, circuit breaker).
    Unavailable,
    /// Available candidates existed but the named policy picked none.
    PolicyDeclined(&'static str),
}

/// The two legs of a disaggregated placement, each the pool it selects over
/// before the availability filter. Both must come from one registry snapshot
/// so the pair cannot straddle a membership change.
pub(crate) struct PairCandidates<'a> {
    pub prefill: &'a [Arc<dyn Worker>],
    pub decode: &'a [Arc<dyn Worker>],
}

/// A selected prefill/decode pair.
pub(crate) struct Pair {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    /// The runtime both legs run when homogeneity was required. Otherwise
    /// it is the first available prefill worker's, which says nothing about
    /// the selected pair; only a homogeneous caller should read it.
    pub runtime: RuntimeType,
}

/// Which leg of a pair placement failed, and why. Boxed at the `Err`
/// site: a shed verdict carries a whole response.
pub(crate) struct PairFailure {
    pub leg: WorkerLeg,
    pub verdict: PlacementFailure,
}

/// The candidates for `model_id` in `pool`, narrowed to `wire` when a retry
/// pins the retained plan's runtime and transport.
pub(crate) fn candidates(
    registry: &WorkerRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
) -> Candidates {
    let pool = registry.get_routing_pool(model_id, pool);
    match wire {
        None => Candidates::Shared(pool),
        Some(wire) => Candidates::Pinned(
            pool.iter()
                .filter(|w| {
                    w.metadata().spec.runtime_type == wire.runtime
                        && *w.connection_mode() == wire.connection
                })
                .cloned()
                .collect(),
        ),
    }
}

/// Pick one worker for `model_id` from `pool`, or `None` when nothing is
/// selectable. Records the selection metric under the chosen worker's own
/// transport label.
pub(crate) fn select_single(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let candidates = candidates(registry, model_id, pool, wire);
    select_from(registry, policies, model_id, candidates.as_slice(), inputs)
}

/// [`select_single`] over an explicit candidate slice: the entry for a caller
/// with a candidate rule the pools cannot express, such as dropping DP-aware
/// workers from a path that cannot pin a rank.
pub(crate) fn select_from(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let policy = policies.get_policy_or_default(model_id);

    // Most policies already apply the complete availability predicate. Give
    // them the shared snapshot directly instead of cloning every available
    // worker into a second per-request Vec. Hash policies use a weaker health
    // predicate and keep the pre-filter.
    let filtered;
    let available: &[Arc<dyn Worker>] = if policy_filters_unavailable_workers(policy.as_ref()) {
        candidates
    } else {
        filtered = candidates
            .iter()
            .filter(|worker| worker.is_available())
            .cloned()
            .collect::<Vec<_>>();
        &filtered
    };
    if available.is_empty() {
        return None;
    }

    // Cached hash ring for consistent hashing (O(log n) lookup).
    let hash_ring = registry.get_hash_ring(model_id);

    // The registry applies the routing-key sticky override when enabled and
    // otherwise delegates to the configured policy.
    let idx = policies.select_worker(
        &policy,
        available,
        &SelectWorkerInfo {
            request_text: inputs.text,
            tokens: inputs.tokens,
            headers: inputs.headers,
            routing_key: policies.resolve_routing_key(inputs.headers),
            rid_key: inputs.rid_key,
            cache_namespace: inputs.cache_namespace,
            hash_ring,
            leg: WorkerLeg::Single,
        },
    )?;
    let selected = available[idx].clone();

    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        selected.connection_mode().as_metric_label(),
        model_id,
        policy.name(),
    );

    Some(selected)
}

/// Classify a failed single-worker placement from the same pool it drew from.
pub(crate) fn single_failure(
    registry: &WorkerRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
) -> PlacementFailure {
    let candidates = candidates(registry, model_id, pool, wire);
    failure_from(candidates.as_slice(), model_id)
}

/// Classify a failed placement from the candidates it drew from.
pub(crate) fn failure_from(candidates: &[Arc<dyn Worker>], model_id: &str) -> PlacementFailure {
    if candidates.is_empty() {
        return PlacementFailure::NoCandidates;
    }
    if let Some(shed) = overload::shed_if_all_overloaded(candidates, model_id) {
        return PlacementFailure::AllOverloaded(shed);
    }
    PlacementFailure::Unavailable
}

/// Pick a prefill/decode pair for `model_id`, one worker per leg, each under
/// its own policy and sticky namespace.
///
/// Every leg is judged live for availability; `wire` pins both legs to the
/// retained plan's runtime and transport on a retry; `homogeneous_runtime` narrows both
/// legs to the first available prefill worker's runtime, which the gRPC
/// wire needs because its rendezvous is runtime-specific. A miss names the
/// leg and carries the verdict judged from that leg's own candidates.
pub(crate) fn select_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: PairCandidates<'_>,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, Box<PairFailure>> {
    let available = |leg: &[Arc<dyn Worker>]| -> Vec<Arc<dyn Worker>> {
        leg.iter()
            .filter(|w| {
                w.is_available()
                    && wire.is_none_or(|wire| {
                        w.metadata().spec.runtime_type == wire.runtime
                            && *w.connection_mode() == wire.connection
                    })
            })
            .cloned()
            .collect()
    };
    let mut prefill = available(candidates.prefill);
    let mut decode = available(candidates.decode);

    if prefill.is_empty() {
        debug!("No available prefill workers");
        return Err(Box::new(PairFailure {
            leg: WorkerLeg::Prefill,
            verdict: failure_from(candidates.prefill, model_id),
        }));
    }
    if decode.is_empty() {
        debug!("No available decode workers");
        return Err(Box::new(PairFailure {
            leg: WorkerLeg::Decode,
            verdict: failure_from(candidates.decode, model_id),
        }));
    }

    // Where the wire's rendezvous is runtime-specific, both legs must share a
    // runtime: take the first prefill worker's and narrow both legs to it.
    let runtime = prefill[0].metadata().spec.runtime_type;
    if homogeneous_runtime {
        let prefill_mixed = prefill
            .iter()
            .skip(1)
            .any(|w| w.metadata().spec.runtime_type != runtime);
        let decode_mixed = decode
            .iter()
            .any(|w| w.metadata().spec.runtime_type != runtime);
        if prefill_mixed || decode_mixed {
            warn!(
                "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
                prefill_mixed, decode_mixed, runtime
            );
        }
        prefill.retain(|w| w.metadata().spec.runtime_type == runtime);
        decode.retain(|w| w.metadata().spec.runtime_type == runtime);
        if decode.is_empty() {
            debug!("No available PD pair for runtime {:?}", runtime);
            return Err(Box::new(PairFailure {
                leg: WorkerLeg::Decode,
                verdict: PlacementFailure::Unavailable,
            }));
        }
    }

    // Independent prefill/decode policies so stateful ones (round robin) do
    // not share a counter; each leg tags the sticky key with its own prefix.
    let prefill_policy = policies.get_prefill_policy();
    let decode_policy = policies.get_decode_policy();
    let hash_ring = registry.get_hash_ring(model_id);
    let mut info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.resolve_routing_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        hash_ring,
        leg: WorkerLeg::Prefill,
    };
    // Both legs were filtered for availability above, so a miss here is the
    // policy's own decision, never an overloaded pool.
    let declined = |leg: WorkerLeg, policy: &'static str| {
        Box::new(PairFailure {
            leg,
            verdict: PlacementFailure::PolicyDeclined(policy),
        })
    };
    let Some(prefill_idx) = policies.select_worker(&prefill_policy, &prefill, &info) else {
        return Err(declined(WorkerLeg::Prefill, prefill_policy.name()));
    };
    info.leg = WorkerLeg::Decode;
    let Some(decode_idx) = policies.select_worker(&decode_policy, &decode, &info) else {
        return Err(declined(WorkerLeg::Decode, decode_policy.name()));
    };

    let selected_prefill = prefill[prefill_idx].clone();
    let selected_decode = decode[decode_idx].clone();
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        selected_prefill.connection_mode().as_metric_label(),
        model_id,
        prefill_policy.name(),
    );
    Metrics::record_worker_selection(
        metrics_labels::WORKER_DECODE,
        selected_decode.connection_mode().as_metric_label(),
        model_id,
        decode_policy.name(),
    );

    Ok(Pair {
        prefill: selected_prefill,
        decode: selected_decode,
        runtime,
    })
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        worker::{BasicWorkerBuilder, ModelCard, WorkerType},
    };

    const MODEL: &str = "m";

    fn registry_with(workers: &[(&str, ConnectionMode, RuntimeType)]) -> WorkerRegistry {
        let typed: Vec<_> = workers
            .iter()
            .map(|(url, connection, runtime)| (*url, WorkerType::Regular, *connection, *runtime))
            .collect();
        registry_of(&typed)
    }

    fn registry_of(workers: &[(&str, WorkerType, ConnectionMode, RuntimeType)]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type, connection, runtime) in workers {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .model(ModelCard::new(MODEL))
                        .worker_type(*worker_type)
                        .connection_mode(*connection)
                        .runtime_type(*runtime)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .expect("worker registers");
        }
        registry
    }

    fn urls(candidates: &Candidates) -> Vec<String> {
        let mut urls: Vec<String> = candidates
            .as_slice()
            .iter()
            .map(|w| w.url().to_string())
            .collect();
        urls.sort();
        urls
    }

    #[test]
    fn each_pool_sees_only_its_own_transport() {
        let registry = registry_with(&[
            ("http://h:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("grpc://g:1", ConnectionMode::Grpc, RuntimeType::Sglang),
            ("zmq://z:1", ConnectionMode::Zmq, RuntimeType::Vllm),
        ]);

        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::HttpRegular,
                None
            )),
            ["http://h:1"]
        );
        // The gRPC pipeline serves both gRPC and direct-ZMQ workers.
        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::GrpcPipelineRegular,
                None
            )),
            ["grpc://g:1", "zmq://z:1"]
        );
    }

    #[test]
    fn a_pinned_wire_narrows_to_its_runtime_and_transport() {
        let registry = registry_with(&[
            ("grpc://g:1", ConnectionMode::Grpc, RuntimeType::Sglang),
            ("grpc://g:2", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("zmq://z:1", ConnectionMode::Zmq, RuntimeType::Vllm),
        ]);
        let wire = Some(WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
        });

        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::GrpcPipelineRegular,
                wire
            )),
            ["grpc://g:2"]
        );
    }

    #[test]
    fn a_pair_shares_the_first_prefill_runtime_and_names_a_missing_leg() {
        let registry = registry_of(&[
            (
                "grpc://p:1",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "grpc://p:2",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:1",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:2",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let snapshot = registry.get_routing_snapshot(MODEL);
        let prefill = snapshot.pool(RoutingPool::GrpcPrefill);
        let decode = snapshot.pool(RoutingPool::GrpcDecode);

        let pair = select_pair(
            &registry,
            &policies,
            MODEL,
            PairCandidates {
                prefill: &prefill,
                decode: &decode,
            },
            None,
            true,
            PlacementInputs::default(),
        )
        .ok()
        .expect("a pair exists under either runtime");
        assert_eq!(pair.prefill.metadata().spec.runtime_type, pair.runtime);
        assert_eq!(pair.decode.metadata().spec.runtime_type, pair.runtime);

        // A model with no decode worker at all names the leg.
        let only_prefill = registry_of(&[(
            "grpc://p:1",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            RuntimeType::Sglang,
        )]);
        let snapshot = only_prefill.get_routing_snapshot(MODEL);
        let prefill = snapshot.pool(RoutingPool::GrpcPrefill);
        let decode = snapshot.pool(RoutingPool::GrpcDecode);
        let failure = select_pair(
            &only_prefill,
            &policies,
            MODEL,
            PairCandidates {
                prefill: &prefill,
                decode: &decode,
            },
            None,
            true,
            PlacementInputs::default(),
        )
        .err()
        .expect("no decode worker exists");
        assert_eq!(failure.leg, WorkerLeg::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::NoCandidates));
    }

    #[test]
    fn a_pinned_pair_keeps_the_retained_runtime_and_transport() {
        // Same runtime on both legs, but the only decode worker speaks ZMQ.
        let registry = registry_of(&[
            (
                "grpc://p:1",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "zmq://d:1",
                WorkerType::Decode,
                ConnectionMode::Zmq,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let by_type = |worker_type: WorkerType| -> Vec<Arc<dyn Worker>> {
            registry
                .get_all()
                .into_iter()
                .filter(|w| *w.worker_type() == worker_type)
                .collect()
        };
        let prefill = by_type(WorkerType::Prefill);
        let decode = by_type(WorkerType::Decode);
        let candidates = || PairCandidates {
            prefill: &prefill,
            decode: &decode,
        };

        // Unpinned, the ZMQ decode worker is a candidate like any other.
        assert!(select_pair(
            &registry,
            &policies,
            MODEL,
            candidates(),
            None,
            false,
            PlacementInputs::default(),
        )
        .is_ok());

        // A retry that retained a gRPC plan must not land on it.
        let failure = select_pair(
            &registry,
            &policies,
            MODEL,
            candidates(),
            Some(WireConstraint {
                runtime: RuntimeType::Sglang,
                connection: ConnectionMode::Grpc,
            }),
            false,
            PlacementInputs::default(),
        )
        .err()
        .expect("the retained transport has no decode worker");
        assert_eq!(failure.leg, WorkerLeg::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::Unavailable));
    }

    #[test]
    fn a_caller_can_narrow_the_candidates_itself() {
        let registry = registry_with(&[
            ("http://h:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://h:2", ConnectionMode::Http, RuntimeType::Sglang),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let pool = registry.get_routing_pool(MODEL, RoutingPool::HttpRegular);
        let only_second: Vec<Arc<dyn Worker>> = pool
            .iter()
            .filter(|w| w.url() == "http://h:2")
            .cloned()
            .collect();

        let selected = select_from(
            &registry,
            &policies,
            MODEL,
            &only_second,
            PlacementInputs::default(),
        )
        .expect("the narrowed slice still has a worker");
        assert_eq!(selected.url(), "http://h:2");
        assert!(matches!(
            failure_from(&[], MODEL),
            PlacementFailure::NoCandidates
        ));
    }

    #[test]
    fn selection_and_failure_read_the_same_pool() {
        let registry = registry_with(&[("http://h:1", ConnectionMode::Http, RuntimeType::Sglang)]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let selected = select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs::default(),
        )
        .expect("the HTTP worker is selectable from its own pool");
        assert_eq!(selected.url(), "http://h:1");

        // The gRPC pool holds nothing for this model, so selection fails and
        // the verdict says why.
        assert!(select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::GrpcPipelineRegular,
            None,
            PlacementInputs::default(),
        )
        .is_none());
        assert!(matches!(
            single_failure(&registry, MODEL, RoutingPool::GrpcPipelineRegular, None),
            PlacementFailure::NoCandidates
        ));
    }
}
