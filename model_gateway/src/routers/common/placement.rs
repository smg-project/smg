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
    worker::{
        ConnectionMode, ConnectionModeExt, PdPairIndex, RoutingPool, RuntimeType, Worker,
        WorkerRegistry,
    },
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
    /// Both legs have workers, whatever their availability, but no prefill
    /// shares a KV transfer protocol with any decode (#2483): judged from
    /// membership alone, since it cannot change until membership does.
    /// Carries each leg's pairing keys and the components the legs
    /// disagreed on.
    NoCompatiblePair {
        prefill: Vec<String>,
        decode: Vec<String>,
        mismatches: Vec<String>,
    },
}

/// A selected prefill/decode pair.
pub(crate) struct Pair {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    /// The selected prefill worker's runtime, which under a homogeneous
    /// caller is the runtime both legs run.
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
/// `pairs` is the snapshot's compatibility index (see [`PdPairIndex`]):
/// which decodes each prefill may hand off to was decided when membership
/// last changed, so this path compares no descriptors. Every leg is judged
/// live for availability, and a prefill counts only while one of its
/// partners is available, so the policy's pick can always be paired; `wire`
/// pins both legs to the retained plan's runtime and transport on a retry;
/// `homogeneous_runtime` narrows both legs to the runtime of the first
/// prefill worker open under its own runtime, which the gRPC wire needs
/// because its rendezvous is runtime-specific. A miss names the leg and carries the verdict judged
/// from that leg's own candidates.
pub(crate) fn select_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, Box<PairFailure>> {
    let eligible = |w: &Arc<dyn Worker>| {
        w.is_available()
            && wire.is_none_or(|wire| {
                w.metadata().spec.runtime_type == wire.runtime
                    && *w.connection_mode() == wire.connection
            })
    };
    let fail = |leg: WorkerLeg, verdict: PlacementFailure| Box::new(PairFailure { leg, verdict });

    // A leg with no worker at all names itself before pairing is judged.
    if pairs.prefill_pool.is_empty() {
        debug!("No prefill workers");
        return Err(fail(WorkerLeg::Prefill, PlacementFailure::NoCandidates));
    }
    if pairs.decode_pool.is_empty() {
        debug!("No decode workers");
        return Err(fail(WorkerLeg::Decode, PlacementFailure::NoCandidates));
    }
    if let Some(refusal) = &pairs.refusal {
        warn!(
            model_id,
            mode = policies.pd_pairing_mode().as_str(),
            mismatches = ?refusal.mismatches,
            prefill = ?refusal.prefill,
            decode = ?refusal.decode,
            "No prefill/decode pair shares a KV transfer protocol"
        );
        return Err(fail(
            WorkerLeg::Decode,
            PlacementFailure::NoCompatiblePair {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            },
        ));
    }

    // Live state: a prefill is open when it is available and so is one of
    // its partners, so the policy's pick can always be paired. Where the
    // wire's rendezvous is runtime-specific, both legs must also share a
    // runtime: that of the first prefill worker open under its own runtime,
    // so a prefill whose partners are all down never shuts out a healthy
    // pair on another runtime. The index keeps runtimes apart already
    // unless pairing is off or a runtime is unknown.
    let partner_open = |d: &Arc<dyn Worker>, runtime: Option<RuntimeType>| {
        eligible(d) && runtime.is_none_or(|r| d.metadata().spec.runtime_type == r)
    };
    let can_pair_on = |i: usize, runtime: Option<RuntimeType>| {
        pairs.partners[i].iter().any(|d| partner_open(d, runtime))
    };
    if !pairs.prefill.iter().any(&eligible) {
        debug!("No available prefill workers");
        return Err(fail(
            WorkerLeg::Prefill,
            failure_from(&pairs.prefill, model_id),
        ));
    }
    let leg_runtime = homogeneous_runtime
        .then(|| {
            pairs.prefill.iter().enumerate().find_map(|(i, p)| {
                let runtime = p.metadata().spec.runtime_type;
                (eligible(p) && can_pair_on(i, Some(runtime))).then_some(runtime)
            })
        })
        .flatten();
    // One pass: the open prefills on the leg's runtime, counting the
    // pairable ones the narrowing excluded so the exclusion leaves a trace.
    let mut excluded = 0usize;
    let open: Vec<usize> = (0..pairs.prefill.len())
        .filter(|&i| {
            let p = &pairs.prefill[i];
            if !eligible(p) {
                return false;
            }
            let own = p.metadata().spec.runtime_type;
            if leg_runtime.is_some_and(|runtime| own != runtime) {
                if can_pair_on(i, Some(own)) {
                    excluded += 1;
                }
                return false;
            }
            can_pair_on(i, leg_runtime)
        })
        .collect();
    if excluded > 0 {
        warn!(
            model_id,
            ?leg_runtime,
            excluded,
            "Mixed runtime types in PD workers; pairable prefill workers on another runtime were excluded"
        );
    }
    if open.is_empty() {
        debug!(?leg_runtime, "No available PD pair");
        return Err(fail(
            WorkerLeg::Decode,
            failure_from(&pairs.decode_pool, model_id),
        ));
    }
    let prefill: Vec<Arc<dyn Worker>> = open
        .iter()
        .map(|&i| Arc::clone(&pairs.prefill[i]))
        .collect();

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
    let selected_prefill = prefill[prefill_idx].clone();
    // The pick's open partners: non-empty by construction, short of an
    // availability flip between the two reads.
    let decode: Vec<Arc<dyn Worker>> = pairs.partners[open[prefill_idx]]
        .iter()
        .filter(|d| partner_open(d, leg_runtime))
        .cloned()
        .collect();
    if decode.is_empty() {
        debug!(
            ?leg_runtime,
            "The selected prefill's partners went unavailable"
        );
        return Err(fail(WorkerLeg::Decode, PlacementFailure::Unavailable));
    }
    info.leg = WorkerLeg::Decode;
    let Some(decode_idx) = policies.select_worker(&decode_policy, &decode, &info) else {
        return Err(declined(WorkerLeg::Decode, decode_policy.name()));
    };
    let selected_decode = decode[decode_idx].clone();
    let runtime = selected_prefill.metadata().spec.runtime_type;
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

    debug!(
        prefill = %selected_prefill.url(),
        decode = %selected_decode.url(),
        "Selected PD pair"
    );
    Ok(Pair {
        prefill: selected_prefill,
        decode: selected_decode,
        runtime,
    })
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::{
        config::types::{PdPairingMode, PolicyConfig},
        worker::{BasicWorkerBuilder, ModelCard, PdWire, WorkerType},
    };

    const MODEL: &str = "m";

    /// A vLLM gRPC PD fleet whose legs carry the given KV connectors
    /// (`None` leaves the transport unknown).
    fn pd_registry(workers: &[(&str, WorkerType, Option<&str>)]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type, connector) in workers {
            let mut builder = BasicWorkerBuilder::new(*url)
                .model(ModelCard::new(MODEL))
                .worker_type(*worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                });
            if let Some(connector) = connector {
                builder = builder.kv_connector(*connector);
            }
            registry
                .register(Arc::new(builder.build()))
                .expect("worker registers");
        }
        registry
    }

    fn pairs_of(registry: &WorkerRegistry, policies: &PolicyRegistry) -> Arc<PdPairIndex> {
        registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode())
    }

    fn pair_from(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
    ) -> Result<Pair, Box<PairFailure>> {
        select_pair(
            registry,
            policies,
            MODEL,
            &pairs_of(registry, policies),
            None,
            true,
            PlacementInputs::default(),
        )
    }

    #[test]
    fn a_pair_shares_its_kv_transport() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // The snapshot's index pairs each prefill with the decode that
        // speaks its transport, once, for every request to read.
        let pairs = pairs_of(&registry, &policies);
        assert_eq!(pairs.prefill.len(), 2);
        for (prefill, partners) in pairs.prefill.iter().zip(&pairs.partners) {
            assert_eq!(partners.len(), 1, "{}", prefill.url());
            assert_eq!(
                prefill.pd_pairing().transport(),
                partners[0].pd_pairing().transport()
            );
        }
        assert!(Arc::ptr_eq(&pairs, &pairs_of(&registry, &policies)));

        // And a placement lands on a matching pair.
        let pair = pair_from(&registry, &policies)
            .ok()
            .expect("the fleet has matching pairs");
        assert_eq!(
            pair.prefill.pd_pairing().transport(),
            pair.decode.pd_pairing().transport()
        );
    }

    #[test]
    fn a_homogeneous_pair_skips_a_prefill_whose_only_partner_runs_another_runtime() {
        // P1 (NIXL) pairs only with D1, whose runtime probe failed; P2
        // (Mooncake) pairs with D1 and D2. On the gRPC wire both legs must
        // share a runtime, so P1 is never open and every placement lands on
        // P2/D2 instead of failing every other request.
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://d:1")
                    .model(ModelCard::new(MODEL))
                    .worker_type(WorkerType::Decode)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Unspecified)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .expect("worker registers");
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        for _ in 0..4 {
            let pair = pair_from(&registry, &policies)
                .ok()
                .expect("P2 and D2 pair on one runtime");
            assert_eq!(pair.prefill.url(), "grpc://p:2");
            assert_eq!(pair.decode.url(), "grpc://d:2");
            assert_eq!(pair.runtime, RuntimeType::Vllm);
        }
    }

    #[test]
    fn a_pair_whose_partners_are_down_does_not_shut_out_a_healthy_pair_on_another_runtime() {
        // One model served by a vLLM pair and an SGLang pair mid-migration.
        // The vLLM decode goes down: the vLLM prefill, first in pool order,
        // must not dictate the runtime, so the SGLang pair keeps serving.
        let registry = registry_of(&[
            (
                "grpc://p:vllm",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://p:sglang",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "grpc://d:vllm",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:sglang",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let vllm_decode = registry
            .get_all()
            .into_iter()
            .find(|w| w.url() == "grpc://d:vllm")
            .expect("registered");
        vllm_decode.set_status(WorkerStatus::NotReady);

        for _ in 0..3 {
            let pair = pair_from(&registry, &policies)
                .ok()
                .expect("the SGLang pair is healthy");
            assert_eq!(pair.prefill.url(), "grpc://p:sglang");
            assert_eq!(pair.decode.url(), "grpc://d:sglang");
            assert_eq!(pair.runtime, RuntimeType::Sglang);
        }
    }

    #[test]
    fn a_fleet_without_a_shared_transport_names_both_legs() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let failure = pair_from(&registry, &policies)
            .err()
            .expect("no pair shares a transport");
        assert_eq!(failure.leg, WorkerLeg::Decode);
        match failure.verdict {
            PlacementFailure::NoCompatiblePair {
                prefill,
                decode,
                mismatches,
            } => {
                assert_eq!(prefill, ["vllm/nixl/?"]);
                assert_eq!(decode, ["vllm/mooncake/?"]);
                assert_eq!(mismatches, ["transport"]);
            }
            _ => panic!("expected NoCompatiblePair"),
        }

        // `off` restores pre-pairing placement for the same fleet.
        let off =
            PolicyRegistry::new(PolicyConfig::RoundRobin).with_pd_pairing_mode(PdPairingMode::Off);
        assert!(pair_from(&registry, &off).is_ok());
    }

    #[test]
    fn an_unknown_transport_pairs_leniently_and_fails_strictly() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, None),
        ]);

        let lenient = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(pair_from(&registry, &lenient).is_ok());

        let strict = PolicyRegistry::new(PolicyConfig::RoundRobin)
            .with_pd_pairing_mode(PdPairingMode::Strict);
        let failure = pair_from(&registry, &strict)
            .err()
            .expect("strict mode refuses an unknown transport");
        assert!(matches!(
            failure.verdict,
            PlacementFailure::NoCompatiblePair { .. }
        ));
    }

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

        let pair = pair_from(&registry, &policies)
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
        let failure = pair_from(&only_prefill, &policies)
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
        // Built from raw legs, as a caller with its own candidate lists would.
        let pairs = PdPairIndex::build(
            by_type(WorkerType::Prefill).into(),
            by_type(WorkerType::Decode).into(),
            policies.pd_pairing_mode(),
        );

        // Unpinned, the ZMQ decode worker is a candidate like any other.
        assert!(select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs,
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
            &pairs,
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
