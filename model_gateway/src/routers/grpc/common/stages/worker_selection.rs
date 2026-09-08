//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use tracing::{error, warn};

use super::PipelineStage;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{CacheNamespace, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        common::placement::{self, PairCandidates, PairFailure, PlacementFailure, PlacementInputs},
        error,
        grpc::{
            context::{
                DispatchContext, EncodeWorkerAssignment, RequestContext, RequestType,
                RoutingSnapshot, WireConstraint, WorkerSelection,
            },
            multimodal,
        },
    },
    worker::{
        ConnectionModeExt, HashRing, ModelWorkerSnapshot, RoutingPool, RuntimeType, Worker,
        WorkerMode, WorkerRegistry, WorkerType,
    },
};

/// Result type for PD worker pair selection: (prefill, decode, runtime_type)
type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);

/// Result type for EPD worker selection: (encode assignments, prefill, decode, runtime_type).
type EncodePrefillDecodeWorkerSelection = (
    Vec<EncodeWorkerAssignment>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    RuntimeType,
);

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
    /// EPD mode: select encode + prefill + decode workers
    EncodePrefillDecode,
}

/// Which candidates a single-worker selection may consider.
#[derive(Clone, Copy, Debug)]
enum LaneFilter {
    /// Any worker in the model's regular gRPC/ZMQ pool.
    Any,
    /// Direct engine workers only; see [`smg_ineligible`].
    EngineOnly,
    /// A retry: only workers matching the retained plan's wire.
    Wire(WireConstraint),
}

impl LaneFilter {
    fn admits(self, worker: &Arc<dyn Worker>) -> bool {
        match self {
            Self::Any => true,
            Self::EngineOnly => worker.worker_mode() != WorkerMode::Smg,
            Self::Wire(c) => c.admits(worker),
        }
    }
}

/// Whether a request must stay off two-tier SMG Workers: `WorkerInference` v1
/// carries text generation only, so embeddings, classify, and multimodal
/// inputs would fail there with 501 at request building.
fn smg_ineligible(request_type: &RequestType, has_multimodal: bool) -> bool {
    has_multimodal
        || matches!(
            request_type,
            RequestType::Embedding(_) | RequestType::Classify(_)
        )
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let intermediate = ctx.state.multimodal_intermediate.as_ref();

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();
        let rid_key = self
            .policy_registry
            .derive_rid_key(ctx.input.request_type.rid())
            .map(str::to_string);
        ctx.state.sticky_key = rid_key.clone().or_else(|| {
            self.policy_registry
                .sticky_header_key(headers)
                .map(str::to_string)
        });

        // Selection inputs that survive the request drop: retry attempts
        // re-select from these. Text is copied only when a configured policy
        // would actually read it (tokens win otherwise).
        let keep_text =
            tokens.is_none() || self.policy_registry.any_policy_needs_request_text(headers);
        // The typed engine wires (gRPC servicers, direct ZMQ) do not carry the
        // request's cache-partition fields yet, so the engine's prefix cache
        // is unpartitioned on this path and the router must not split
        // affinity for it: a namespace here would cost cache hits for every
        // request that sets a salt today. The snapshot carries the field so
        // the forwarding change can derive it (as the HTTP proxy path does via
        // `GenerationRequest::cache_partition`) without touching selection.
        let cache_namespace: Option<CacheNamespace> = None;
        ctx.state.routing_snapshot = Some(RoutingSnapshot {
            routing_text: keep_text.then(|| text.map(str::to_string)).flatten(),
            token_ids: ids.to_vec(),
            rid_key: rid_key.clone(),
            cache_namespace,
        });
        let rid_key = rid_key.as_deref();

        let model_id = ctx.input.model_id.as_str();
        // `WorkerInference` v1 is text generation only. In a pool that mixes
        // two-tier SMG Workers with direct engine workers, an embedding,
        // classify, or multimodal request placed on an SMG Worker would fail
        // with 501 at request building instead of reaching an engine that can
        // serve it -- so keep those requests off SMG Workers here.
        let lane = if smg_ineligible(&ctx.input.request_type, intermediate.is_some()) {
            LaneFilter::EngineOnly
        } else {
            LaneFilter::Any
        };
        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    lane,
                ) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        return Err(self.selection_failure(model_id, &[WorkerType::Regular], None))
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self.select_pd_pair(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    None,
                ) {
                    Ok((prefill, decode, runtime_type)) => WorkerSelection::Disaggregated {
                        encode_assignments: None,
                        prefill,
                        decode,
                        runtime_type,
                    },
                    Err(response) => return Err(response),
                }
            }
            WorkerSelectionMode::EncodePrefillDecode => {
                let encode_item_hashes = match encode_item_hashes(intermediate) {
                    Ok(hashes) => hashes,
                    Err(err) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            error = %err,
                            "Failed to derive encode item routing hashes"
                        );
                        return Err(error::internal_error(
                            "encode_routing_hash_failed",
                            format!("Failed to derive encode routing hashes: {err}"),
                        ));
                    }
                };
                match self.select_encode_prefill_decode_workers(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    &encode_item_hashes,
                ) {
                    Some((encode_assignments, prefill, decode, runtime_type)) => {
                        WorkerSelection::Disaggregated {
                            encode_assignments: if encode_assignments.is_empty() {
                                None
                            } else {
                                Some(encode_assignments)
                            },
                            prefill,
                            decode,
                            runtime_type,
                        }
                    }
                    None => {
                        // Encode is a demanded leg only when the request
                        // carries encode items; an idle-but-vetoed encode pool
                        // must not shed a text-only request.
                        let legs: &[WorkerType] = if encode_item_hashes.is_empty() {
                            &[WorkerType::Prefill, WorkerType::Decode]
                        } else {
                            &[WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode]
                        };
                        return Err(self.selection_failure(model_id, legs, None));
                    }
                }
            }
        };

        // Reject an unsupported (backend, modality) combination now that the
        // runtime is known, before request building fetches/preprocesses media
        // only to fail deep in assembly. The prefill leg builds the request in
        // disaggregated mode, so its runtime is the one that must support the
        // request's modalities.
        if let Some(intermediate) = intermediate {
            if let Err(err) = multimodal::ensure_backend_supports_modalities(
                selection_runtime(&workers),
                intermediate,
            ) {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    format!("{err}"),
                ));
            }
        }

        ctx.state.workers = Some(workers);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

impl WorkerSelectionStage {
    #[cfg(test)]
    pub(crate) fn signature(&self) -> String {
        format!("WorkerSelectionStage({:?})", self.mode)
    }

    /// Per-attempt re-selection from the routing snapshot, after the request
    /// dropped at build. Candidates are pinned to the retained plan's wire
    /// (runtime + transport): the plan cannot be rebuilt for another flavor.
    /// EPD re-selects only the prefill/decode pair — the first dispatch
    /// already launched the encode jobs, and the plan carries their
    /// bootstrap rooms.
    pub(crate) fn reselect(&self, ctx: &mut DispatchContext) -> Result<(), Response> {
        let text = ctx.routing.routing_text.as_deref();
        let tokens = if ctx.routing.token_ids.is_empty() {
            None
        } else {
            Some(ctx.routing.token_ids.as_slice())
        };
        let rid_key = ctx.routing.rid_key.as_deref();
        let cache_namespace = ctx.routing.cache_namespace;
        let headers = ctx.headers.as_ref();
        let model_id = ctx.model_id.as_str();
        let wire = Some(ctx.wire);

        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    LaneFilter::Wire(ctx.wire),
                ) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        return Err(self.selection_failure(model_id, &[WorkerType::Regular], wire))
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode | WorkerSelectionMode::EncodePrefillDecode => {
                match self.select_pd_pair(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    wire,
                ) {
                    Ok((prefill, decode, runtime_type)) => WorkerSelection::Disaggregated {
                        encode_assignments: None,
                        prefill,
                        decode,
                        runtime_type,
                    },
                    Err(response) => return Err(response),
                }
            }
        };

        ctx.workers = Some(workers);
        Ok(())
    }
}

/// Runtime of the leg that builds the generate request: the sole worker in
/// regular mode, the prefill worker in disaggregated (PD/EPD) mode.
fn selection_runtime(workers: &WorkerSelection) -> RuntimeType {
    match workers {
        WorkerSelection::Single { worker } => worker.metadata().spec.runtime_type,
        WorkerSelection::Disaggregated { runtime_type, .. } => *runtime_type,
    }
}

impl WorkerSelectionStage {
    /// Response for a selection that produced nothing: a 503 shed when a leg's
    /// whole candidate pool is vetoed, the existing 404 otherwise.
    ///
    /// `legs` must be exactly the legs this selection demanded — the verdict is
    /// per leg because a whole-model predicate is false exactly when one
    /// saturated leg made the set unselectable, and an undemanded leg (EPD
    /// without encode items) must not be able to shed a request that never
    /// needed it.
    /// `wire` narrows the verdict to the retained plan's runtime/transport on
    /// the retry path: a drained pinned pool must not answer 404 just because
    /// other runtimes still serve the model.
    fn selection_failure(
        &self,
        model_id: &str,
        legs: &[WorkerType],
        wire: Option<WireConstraint>,
    ) -> Response {
        let mut unavailable = false;
        for leg in legs {
            let verdict = match leg {
                // The regular leg is judged from exactly the pool the shared
                // placement drew from.
                WorkerType::Regular => placement::single_failure(
                    &self.worker_registry,
                    model_id,
                    RoutingPool::GrpcPipelineRegular,
                    wire,
                ),
                WorkerType::Prefill => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcPrefill, wire)
                }
                WorkerType::Decode => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcDecode, wire)
                }
                WorkerType::Encode => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcEncode, wire)
                }
            };
            match verdict {
                PlacementFailure::AllOverloaded(shed) => return shed,
                PlacementFailure::Unavailable | PlacementFailure::PolicyDeclined(_) => {
                    unavailable = true;
                }
                PlacementFailure::NoCandidates => {}
            }
        }
        if unavailable {
            return self.workers_unavailable(model_id);
        }
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No worker serves model"
        );
        error::model_not_found(model_id)
    }

    /// Workers serve the model but none can take the request right now
    /// (unhealthy, circuit breaker open, or the policy declined). A 503 with
    /// the same code the HTTP router uses: the model exists, the client should
    /// retry, and nothing about its request is wrong. Answering 404 here told
    /// clients the model was gone while its workers restarted.
    fn workers_unavailable(&self, model_id: &str) -> Response {
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No available workers for model"
        );
        error::service_unavailable(
            "no_available_workers",
            format!("All workers for model '{model_id}' are unavailable (unhealthy or circuit breaker open)"),
        )
    }

    /// The response for a failed pair placement. The verdict was judged from
    /// the leg's own candidates inside the placement, so a shed is answered
    /// as it was built and counted once; a leg nobody serves is a 404, and a
    /// leg whose workers are all unavailable is the 503 the HTTP router gives.
    fn pair_failure(&self, model_id: &str, failure: PairFailure) -> Response {
        match failure.verdict {
            PlacementFailure::AllOverloaded(shed) => shed,
            PlacementFailure::Unavailable | PlacementFailure::PolicyDeclined(_) => {
                self.workers_unavailable(model_id)
            }
            PlacementFailure::NoCandidates => {
                error!(
                    function = "WorkerSelectionStage::execute",
                    mode = ?self.mode,
                    model_id = %model_id,
                    leg = ?failure.leg,
                    "No worker serves model"
                );
                error::model_not_found(model_id)
            }
        }
    }

    /// The verdict for one disaggregated leg, judged from the pool it
    /// selected over *before* the `is_available()` filter. The legs are
    /// gRPC-only (no KV rendezvous on ZMQ), so a retry pins the runtime alone;
    /// the regular leg is judged by [`placement::single_failure`] instead.
    /// Failure path only.
    fn disaggregated_leg_verdict(
        &self,
        model_id: &str,
        pool: RoutingPool,
        wire: Option<WireConstraint>,
    ) -> PlacementFailure {
        let candidates: Vec<Arc<dyn Worker>> = self
            .worker_registry
            .get_routing_pool(model_id, pool)
            .iter()
            .filter(|w| wire.is_none_or(|c| w.metadata().spec.runtime_type == c.runtime))
            .cloned()
            .collect();
        placement::failure_from(&candidates, model_id)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "selection threads every routing input the policy consumes"
    )]
    fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
        cache_namespace: Option<CacheNamespace>,
        lane: LaneFilter,
    ) -> Option<Arc<dyn Worker>> {
        // The gRPC router serves both gRPC and direct-ZMQ workers, so the pool
        // accepts either transport (not HTTP). A retry pins the retained wire;
        // a request the two-tier lane cannot serve keeps off SMG Workers.
        let pool = self
            .worker_registry
            .get_routing_pool(model_id, RoutingPool::GrpcPipelineRegular);
        let filtered;
        let candidates: &[Arc<dyn Worker>] = match lane {
            LaneFilter::Any => &pool,
            lane => {
                filtered = pool
                    .iter()
                    .filter(|w| lane.admits(w))
                    .cloned()
                    .collect::<Vec<_>>();
                &filtered
            }
        };
        placement::select_from(
            &self.worker_registry,
            &self.policy_registry,
            model_id,
            candidates,
            PlacementInputs {
                text,
                tokens,
                headers,
                rid_key,
                cache_namespace,
            },
        )
    }

    /// Workers from one leg pool of `snapshot` that also pass the live
    /// `is_available()` check (health, circuit breaker, overload veto).
    fn available_workers(
        snapshot: &ModelWorkerSnapshot,
        pool: RoutingPool,
    ) -> Vec<Arc<dyn Worker>> {
        snapshot
            .pool(pool)
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "selection threads every routing input the policy consumes"
    )]
    fn select_pd_pair(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
        cache_namespace: Option<CacheNamespace>,
        wire: Option<WireConstraint>,
    ) -> Result<PdWorkerPair, Response> {
        // Both legs derive from ONE membership snapshot: separate pool
        // lookups could straddle a concurrent replacement and pair workers
        // that never coexisted. The pools are strictly gRPC (a ZMQ leg would
        // silently drop the PD bootstrap info, see `RoutingPool::GrpcPrefill`;
        // the wildcard model maps to the global snapshot). The legs must
        // share a runtime, the rendezvous being runtime-specific, and a retry
        // pins both to the retained plan's runtime.
        let snapshot = self.worker_registry.get_routing_snapshot(model_id);
        let prefill = snapshot.pool(RoutingPool::GrpcPrefill);
        let decode = snapshot.pool(RoutingPool::GrpcDecode);
        let pair = placement::select_pair(
            &self.worker_registry,
            &self.policy_registry,
            model_id,
            PairCandidates {
                prefill: &prefill,
                decode: &decode,
            },
            wire,
            true,
            PlacementInputs {
                text,
                tokens,
                headers,
                rid_key,
                cache_namespace,
            },
        )
        .map_err(|failure| self.pair_failure(model_id, *failure))?;
        Ok((pair.prefill, pair.decode, pair.runtime))
    }

    /// Select per-item encode workers + a prefill/decode pair for EPD routing.
    ///
    /// Mirrors `select_pd_pair` but also assigns each multimodal item to an
    /// encode worker. prefill+decode are selected as a normal PD pair. All pools
    /// are filtered to a runtime shared by the selected encode/prefill/decode
    /// legs.
    #[expect(
        clippy::too_many_arguments,
        reason = "selection threads every routing input the policy consumes"
    )]
    fn select_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
        cache_namespace: Option<CacheNamespace>,
        encode_item_hashes: &[Vec<u8>],
    ) -> Option<EncodePrefillDecodeWorkerSelection> {
        // All three legs derive from ONE membership snapshot (see
        // select_pd_pair). The pools are strictly gRPC — encode dispatch is
        // a gRPC encoder RPC the direct-ZMQ worker has no path for, and the
        // ZMQ wire carries no KV-transfer rendezvous for the prefill/decode
        // legs; the wildcard model maps to the global snapshot. Availability
        // stays a live per-request check.
        let snapshot = self.worker_registry.get_routing_snapshot(model_id);
        let all_encode = Self::available_workers(&snapshot, RoutingPool::GrpcEncode);
        let all_prefill = Self::available_workers(&snapshot, RoutingPool::GrpcPrefill);
        let all_decode = Self::available_workers(&snapshot, RoutingPool::GrpcDecode);

        let needs_encode = !encode_item_hashes.is_empty();
        if needs_encode && all_encode.is_empty() {
            warn!("No available encode workers");
            return None;
        }
        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }
        if all_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Disaggregated legs must share a runtime. Pick a runtime that has at
        // least one available worker in every required EPD pool instead of
        // blindly using the first prefill runtime.
        let Some(target_runtime) = all_prefill
            .iter()
            .map(|w| w.metadata().spec.runtime_type)
            .find(|runtime| {
                // The current EPD multimodal encoder adapter is TokenSpeed-
                // specific. Do not select a shared SGLang/vLLM runtime only to
                // reject it later during request building.
                (!needs_encode || *runtime == RuntimeType::TokenSpeed)
                    && all_decode
                        .iter()
                        .any(|w| w.metadata().spec.runtime_type == *runtime)
                    && (!needs_encode
                        || all_encode
                            .iter()
                            .any(|w| w.metadata().spec.runtime_type == *runtime))
            })
        else {
            warn!("No available encode/prefill/decode worker set with a shared runtime");
            return None;
        };

        let mixed = all_prefill
            .iter()
            .chain(all_decode.iter())
            .any(|w| w.metadata().spec.runtime_type != target_runtime)
            || (needs_encode
                && all_encode
                    .iter()
                    .any(|w| w.metadata().spec.runtime_type != target_runtime));
        if mixed {
            warn!(
                "Mixed runtime types in encode/prefill/decode workers. Using {:?}.",
                target_runtime
            );
        }

        // Filter all three pools to the target runtime
        let available_encode: Vec<_> = all_encode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if (needs_encode && available_encode.is_empty())
            || available_prefill.is_empty()
            || available_decode.is_empty()
        {
            warn!(
                "No available encode/prefill/decode worker set for runtime {:?}",
                target_runtime
            );
            return None;
        }

        // Select encode, prefill, and decode via their per-role policies. Encode
        // defaults to consistent hashing over each item's content hash; prefill
        // and decode fall back to the main policy when unset.
        let encode_policy = self.policy_registry.get_encode_policy();
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let mut info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            routing_key: self.policy_registry.resolve_routing_key(headers),
            rid_key,
            cache_namespace,
            hash_ring: hash_ring.clone(),
            leg: WorkerLeg::Prefill,
        };
        let prefill_idx =
            self.policy_registry
                .select_worker(&prefill_policy, &available_prefill, &info)?;
        info.leg = WorkerLeg::Decode;
        let decode_idx =
            self.policy_registry
                .select_worker(&decode_policy, &available_decode, &info)?;

        let encode_assignments = assign_encode_workers(
            &available_encode,
            encode_item_hashes,
            model_id,
            encode_policy.as_ref(),
            hash_ring.clone(),
        )?;

        // Record worker selection metrics for prefill and decode, each tagged
        // with the policy that picked it. Encode item assignment metrics are
        // recorded in assign_encode_workers.
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            available_prefill[prefill_idx]
                .connection_mode()
                .as_metric_label(),
            model_id,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            available_decode[decode_idx]
                .connection_mode()
                .as_metric_label(),
            model_id,
            decode_policy.name(),
        );

        Some((
            encode_assignments,
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
            target_runtime,
        ))
    }
}

fn encode_item_hashes(
    intermediate: Option<&multimodal::MultimodalIntermediate>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let Some(intermediate) = intermediate else {
        return Ok(Vec::new());
    };
    multimodal::encode_routing_hashes(intermediate)
}

fn assign_encode_workers(
    encode_workers: &[Arc<dyn Worker>],
    item_hashes: &[Vec<u8>],
    model_id: &str,
    policy: &dyn LoadBalancingPolicy,
    hash_ring: Option<Arc<HashRing>>,
) -> Option<Vec<EncodeWorkerAssignment>> {
    if item_hashes.is_empty() {
        return Some(Vec::new());
    }

    item_hashes
        .iter()
        .enumerate()
        .map(|(item_index, content_hash)| {
            let routing_headers = encode_routing_headers(content_hash);
            let info = SelectWorkerInfo {
                request_text: None,
                tokens: None,
                headers: Some(&routing_headers),
                routing_key: None,
                // Encode items key by media-content hash; a conversation key
                // here would defeat per-item encode reuse.
                rid_key: None,
                cache_namespace: None,
                hash_ring: hash_ring.clone(),
                leg: WorkerLeg::Single,
            };
            let worker_idx = policy.select_worker(encode_workers, &info)?;
            let worker = encode_workers[worker_idx].clone();
            Metrics::record_worker_selection(
                metrics_labels::WORKER_ENCODE,
                metrics_labels::CONNECTION_GRPC,
                model_id,
                policy.name(),
            );
            Some(EncodeWorkerAssignment { item_index, worker })
        })
        .collect()
}

fn encode_routing_headers(content_hash: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = hex_encode(content_hash);
    if let Ok(value) = HeaderValue::from_str(&key) {
        headers.insert("x-smg-routing-key", value);
    }
    headers
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::http::StatusCode;
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyFactory,
        routers::common::retry::is_retryable_response,
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn register_pd_workers(
        registry: &WorkerRegistry,
        model_id: &str,
        n: usize,
    ) -> (Vec<String>, Vec<String>) {
        let mut prefill_urls = Vec::with_capacity(n);
        let mut decode_urls = Vec::with_capacity(n);

        for i in 0..n {
            let url = format!("grpc://127.0.0.1:{}", 8000 + i);
            prefill_urls.push(url.clone());
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Prefill)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        for i in 0..n {
            let url = format!("grpc://127.0.0.1:{}", 8100 + i);
            decode_urls.push(url.clone());
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Decode)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        (prefill_urls, decode_urls)
    }

    fn hit_counts_in_order(urls: &[String], hits: &HashMap<String, usize>) -> Vec<usize> {
        urls.iter()
            .map(|url| hits.get(url).copied().unwrap_or(0))
            .collect()
    }

    /// Correctness bar for PD round-robin: every worker in both pools is hit
    /// equally across 40 `select_pd_pair` calls.
    fn assert_even_pd_round_robin_coverage(
        prefill_urls: &[String],
        decode_urls: &[String],
        prefill_hits: &HashMap<String, usize>,
        decode_hits: &HashMap<String, usize>,
    ) {
        assert_eq!(
            hit_counts_in_order(prefill_urls, prefill_hits),
            vec![10, 10, 10, 10],
            "even PD round-robin coverage: every prefill worker should get 10/40"
        );
        assert_eq!(
            hit_counts_in_order(decode_urls, decode_hits),
            vec![10, 10, 10, 10],
            "even PD round-robin coverage: every decode worker should get 10/40"
        );
    }

    /// Drive `select_pd_pair` through the stage (uses `get_prefill_policy` /
    /// `get_decode_policy` internally) and count selections by worker URL.
    fn count_select_pd_pair_hits(
        stage: &WorkerSelectionStage,
        model_id: &str,
        iterations: usize,
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let mut prefill_hits = HashMap::new();
        let mut decode_hits = HashMap::new();
        for _ in 0..iterations {
            let (prefill, decode, _) = stage
                .select_pd_pair(model_id, None, None, None, None, None, None)
                .expect("select_pd_pair should return a pair");
            *prefill_hits.entry(prefill.url().to_string()).or_default() += 1;
            *decode_hits.entry(decode.url().to_string()).or_default() += 1;
        }
        (prefill_hits, decode_hits)
    }

    /// A saturated prefill leg is a pressure condition, not model absence.
    ///
    /// The model's decode workers stay unflagged, so a whole-model shed
    /// predicate reads "not all overloaded" and the request would fall through
    /// to 404 — the exact answer this feature exists to replace, and worse than
    /// the pre-feature behaviour where the prefill worker still served.
    #[test]
    fn a_fully_vetoed_prefill_leg_sheds_rather_than_404s() {
        let model_id = "test-model-prefill-veto";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, _) = register_pd_workers(&worker_registry, model_id, 4);

        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
        );
        assert!(stage
            .select_pd_pair(model_id, None, None, None, None, None, None)
            .is_ok());

        for url in &prefill_urls {
            let worker = worker_registry.get_by_url(url).expect("registered");
            worker_registry.set_worker_overloaded(&worker, true);
        }

        assert!(
            stage
                .select_pd_pair(model_id, None, None, None, None, None, None)
                .is_err(),
            "the veto empties the prefill pool"
        );
        let response =
            stage.selection_failure(model_id, &[WorkerType::Prefill, WorkerType::Decode], None);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert!(
            !is_retryable_response(&response),
            "a shed must be terminal for the retry layer"
        );
    }

    /// An undemanded leg cannot shed: a text-only EPD request that fails for a
    /// non-overload reason must not 503 just because the (unused) encode pool
    /// is saturated.
    #[test]
    fn an_undemanded_encode_leg_cannot_shed() {
        let model_id = "test-model-encode-veto";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let encode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8460")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Encode)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        worker_registry.register(Arc::clone(&encode)).unwrap();
        worker_registry.set_worker_overloaded(&encode, true);

        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::EncodePrefillDecode,
        );

        // No prefill/decode workers registered: with encode undemanded this is
        // model absence (404), not pressure.
        let text_only =
            stage.selection_failure(model_id, &[WorkerType::Prefill, WorkerType::Decode], None);
        assert_eq!(text_only.status(), StatusCode::NOT_FOUND);

        // With encode demanded, the saturated encode pool is a shed.
        let with_encode = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode],
            None,
        );
        assert_eq!(with_encode.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A model nobody serves is still a 404 — the shed must not swallow real
    /// misconfiguration.
    #[test]
    fn an_unserved_model_still_reports_not_found() {
        let stage = WorkerSelectionStage::new(
            Arc::new(WorkerRegistry::new()),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
        );
        assert_eq!(
            stage
                .selection_failure("nobody", &[WorkerType::Prefill, WorkerType::Decode], None)
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    #[should_panic(expected = "even PD round-robin coverage")]
    fn select_pd_pair_shared_round_robin_fails_even_coverage() {
        // Same correctness bar as the independent test. One shared RoundRobin
        // Arc for P/D advances the counter twice per request, so even coverage
        // must fail (this test is expected to panic on that assertion).
        let model_id = "test-model-shared";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let shared = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        policy_registry.set_prefill_policy(Arc::clone(&shared));
        policy_registry.set_decode_policy(shared);
        assert!(Arc::ptr_eq(
            &policy_registry.get_prefill_policy(),
            &policy_registry.get_decode_policy()
        ));

        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
        );
        let (prefill_hits, decode_hits) = count_select_pd_pair_hits(&stage, model_id, 40);
        assert_even_pd_round_robin_coverage(
            &prefill_urls,
            &decode_urls,
            &prefill_hits,
            &decode_hits,
        );
    }

    #[test]
    fn select_pd_pair_independent_round_robin_passes_even_coverage() {
        // Production PD startup: two independent RoundRobinPolicy instances.
        // Same correctness bar; this configuration must pass.
        let model_id = "test-model-independent";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        assert!(!Arc::ptr_eq(
            &policy_registry.get_prefill_policy(),
            &policy_registry.get_decode_policy()
        ));

        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
        );
        let (prefill_hits, decode_hits) = count_select_pd_pair_hits(&stage, model_id, 40);
        assert_even_pd_round_robin_coverage(
            &prefill_urls,
            &decode_urls,
            &prefill_hits,
            &decode_hits,
        );
    }

    #[test]
    fn select_pd_pair_ignores_zmq_legs() {
        // The ZMQ wire carries no KV-transfer rendezvous, so ZMQ prefill/decode
        // workers must never be paired even if they reach the registry.
        let model_id = "test-model-zmq";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (port, worker_type) in [(9000, WorkerType::Prefill), (9100, WorkerType::Decode)] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("ipc:///tmp/smg-zmq/{port}.ipc"))
                        .model(ModelCard::new(model_id))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Zmq)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::PrefillDecode,
        );

        assert!(
            stage
                .select_pd_pair(model_id, None, None, None, None, None, None)
                .is_err(),
            "ZMQ-only PD pools must not yield a pair"
        );

        // Adding gRPC legs makes selection succeed, and it never picks the ZMQ ones.
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);
        let (prefill, decode, _) = stage
            .select_pd_pair(model_id, None, None, None, None, None, None)
            .expect("gRPC PD pair should be selected");
        assert!(prefill_urls.contains(&prefill.url().to_string()));
        assert!(decode_urls.contains(&decode.url().to_string()));
    }

    /// gRPC selection pins by the rid-derived key under the override: repeats
    /// of one conversation land on one worker even as a poisoned per-request
    /// header key rotates; the header only keys requests without a rid.
    #[test]
    fn grpc_selection_pins_by_rid_key_under_override() {
        use crate::config::types::{ManualAssignmentMode, RoutingKeyOverrideConfig};

        let model_id = "test-model-rid-sticky";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for i in 0..2 {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", 8300 + i))
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let policy_registry = Arc::new(PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::Delegate,
                ..Default::default()
            },
        ));
        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry.clone(),
            WorkerSelectionMode::Regular,
        );

        let rid_key = policy_registry.derive_rid_key(Some("conv7_t1"));
        assert_eq!(rid_key, Some("conv7"));

        let mut poison = HeaderMap::new();
        poison.insert("x-smg-routing-key", "req-unique-1".parse().unwrap());
        let first = stage
            .select_single_worker(
                model_id,
                None,
                None,
                Some(&poison),
                rid_key,
                None,
                LaneFilter::Any,
            )
            .unwrap();
        for (i, rid) in ["conv7_t2", "conv7_t2_r1", "conv7_t3"].iter().enumerate() {
            let mut rotated = HeaderMap::new();
            rotated.insert(
                "x-smg-routing-key",
                format!("req-unique-{}", i + 2).parse().unwrap(),
            );
            let again = stage
                .select_single_worker(
                    model_id,
                    None,
                    None,
                    Some(&rotated),
                    policy_registry.derive_rid_key(Some(rid)),
                    None,
                    LaneFilter::Any,
                )
                .unwrap();
            assert_eq!(again.url(), first.url(), "follow-up must pin by rid key");
        }
    }

    /// The gRPC transport must shed an all-overloaded model with the same
    /// distinct overload 503 the HTTP router uses. Its usual empty-pool answer
    /// is a 404, which would both misreport pressure as model absence and skip
    /// the retry path (404 is not retryable).
    #[test]
    fn grpc_all_overloaded_sheds_503_instead_of_404() {
        use crate::routers::error::extract_error_code_from_response;

        let model_id = "test-model-overload-shed";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let mut workers = Vec::new();
        for i in 0..2 {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", 8400 + i))
                    .model(ModelCard::new(model_id))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .health_config(no_health_check())
                    .build(),
            );
            worker_registry.register(Arc::clone(&worker)).unwrap();
            workers.push(worker);
        }
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            policy_registry,
            WorkerSelectionMode::Regular,
        );

        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
            .is_some());

        worker_registry.set_worker_overloaded(&workers[0], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
                .is_some(),
            "one eligible worker left still serves"
        );

        worker_registry.set_worker_overloaded(&workers[1], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
                .is_none(),
            "the veto empties the candidate pool"
        );

        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );

        // Recovery re-admits, and the failure response goes back to 404 for a
        // genuinely absent model.
        worker_registry.set_worker_overloaded(&workers[0], false);
        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
            .is_some());
        assert_eq!(
            stage
                .selection_failure("no-such-model", &[WorkerType::Regular], None)
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    /// Workers serve the model but none is available: a 503 with the HTTP
    /// router's code, not the 404 that told clients the model was gone while
    /// its workers restarted.
    #[test]
    fn an_unavailable_regular_worker_answers_503_not_404() {
        use openai_protocol::worker::WorkerStatus;

        use crate::routers::error::extract_error_code_from_response;
        let model_id = "test-model-unavailable";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8470")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        worker_registry.register(Arc::clone(&worker)).unwrap();
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
        );
        // Any status but Ready is unavailable to routing.
        worker.set_status(WorkerStatus::NotReady);
        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
            .is_none());

        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "no_available_workers"
        );
        // A model nobody serves stays a 404.
        assert_eq!(
            stage
                .selection_failure("no-such-model", &[WorkerType::Regular], None)
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    /// A disaggregated leg whose only worker is down is the same 503, both
    /// through the per-leg fallback and through the pair verdict.
    #[test]
    fn an_unavailable_decode_leg_answers_503_not_404() {
        use openai_protocol::worker::WorkerStatus;

        use crate::{policies::WorkerLeg, routers::error::extract_error_code_from_response};
        let model_id = "test-model-decode-down";
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_pd_workers(&worker_registry, model_id, 1);
        let decode = worker_registry
            .get_routing_pool(model_id, RoutingPool::GrpcDecode)
            .first()
            .cloned()
            .expect("one decode worker registered");
        decode.set_status(WorkerStatus::NotReady);
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
        );

        let fallback =
            stage.selection_failure(model_id, &[WorkerType::Prefill, WorkerType::Decode], None);
        let pair = stage.pair_failure(
            model_id,
            PairFailure {
                leg: WorkerLeg::Decode,
                verdict: PlacementFailure::Unavailable,
            },
        );

        for response in [fallback, pair] {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                extract_error_code_from_response(&response),
                "no_available_workers"
            );
        }
        let absent = stage.pair_failure(
            model_id,
            PairFailure {
                leg: WorkerLeg::Decode,
                verdict: PlacementFailure::NoCandidates,
            },
        );
        assert_eq!(absent.status(), StatusCode::NOT_FOUND);
    }

    fn dispatch_ctx(model_id: &str, wire: WireConstraint) -> DispatchContext {
        DispatchContext {
            model_id: model_id.to_string(),
            dispatch_model: model_id.to_string(),
            streaming: false,
            headers: None,
            rate_limit_cell: None,
            routing: RoutingSnapshot {
                routing_text: None,
                token_ids: vec![1, 2, 3],
                rid_key: None,
                cache_namespace: None,
            },
            wire,
            tokenizer: None,
            workers: None,
            sticky_key: None,
            clients: None,
            encode_outputs: None,
            dispatch: None,
            load_guards: None,
            response: Default::default(),
        }
    }

    /// Retry re-selection must never leave the retained plan's wire: the
    /// runtime AND transport filters both apply, or a retry could pick a
    /// worker the plan's proto flavor cannot be dispatched to.
    #[test]
    fn reselect_pins_regular_candidates_to_the_retained_wire() {
        let model_id = "wire-pin-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, runtime, connection) in [
            (
                "grpc://127.0.0.1:9300",
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
            ),
            (
                "grpc://127.0.0.1:9301",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
            ),
            (
                "ipc:///tmp/smg-wire-pin",
                RuntimeType::Vllm,
                ConnectionMode::Zmq,
            ),
        ] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(connection)
                        .runtime_type(runtime)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
        );

        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                mode: WorkerMode::Engine,
            },
        );
        for _ in 0..8 {
            stage.reselect(&mut ctx).unwrap();
            match ctx.workers.as_ref().unwrap() {
                WorkerSelection::Single { worker } => {
                    assert_eq!(
                        worker.url(),
                        "grpc://127.0.0.1:9301",
                        "reselect must stay on the retained runtime and transport"
                    );
                }
                WorkerSelection::Disaggregated { .. } => panic!("expected single selection"),
            }
        }
    }

    #[test]
    fn reselect_pins_pd_pair_to_the_retained_runtime() {
        let model_id = "wire-pin-pd-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, runtime) in [
            (
                "grpc://127.0.0.1:9310",
                WorkerType::Prefill,
                RuntimeType::Sglang,
            ),
            (
                "grpc://127.0.0.1:9311",
                WorkerType::Decode,
                RuntimeType::Sglang,
            ),
            (
                "grpc://127.0.0.1:9312",
                WorkerType::Prefill,
                RuntimeType::Vllm,
            ),
            (
                "grpc://127.0.0.1:9313",
                WorkerType::Decode,
                RuntimeType::Vllm,
            ),
        ] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(runtime)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
        );

        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                mode: WorkerMode::Engine,
            },
        );
        for _ in 0..8 {
            stage.reselect(&mut ctx).unwrap();
            match ctx.workers.as_ref().unwrap() {
                WorkerSelection::Disaggregated {
                    prefill,
                    decode,
                    runtime_type,
                    ..
                } => {
                    assert_eq!(*runtime_type, RuntimeType::Vllm);
                    assert_eq!(prefill.url(), "grpc://127.0.0.1:9312");
                    assert_eq!(decode.url(), "grpc://127.0.0.1:9313");
                }
                WorkerSelection::Single { .. } => panic!("expected PD selection"),
            }
        }
    }

    /// A drained pinned pool must shed (503), not answer 404 just because
    /// another runtime still serves the model: 404 is non-retryable and would
    /// end the retry loop on a lie.
    #[test]
    fn reselect_verdict_reflects_the_pinned_pool() {
        let model_id = "wire-verdict-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let mut pinned = None;
        for (url, runtime) in [
            ("grpc://127.0.0.1:9320", RuntimeType::Sglang),
            ("grpc://127.0.0.1:9321", RuntimeType::Vllm),
        ] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(model_id))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(runtime)
                    .health_config(no_health_check())
                    .build(),
            );
            worker_registry.register(worker.clone()).unwrap();
            if runtime == RuntimeType::Vllm {
                pinned = Some(worker);
            }
        }
        worker_registry.set_worker_overloaded(&pinned.expect("vllm worker registered"), true);

        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
        );
        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                mode: WorkerMode::Engine,
            },
        );

        let response = stage
            .reselect(&mut ctx)
            .expect_err("the pinned pool is fully vetoed");
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "verdict must reflect the pinned pool, not every runtime"
        );
    }

    /// `WorkerInference` v1 is text generation only. In a pool mixing two-tier
    /// SMG Workers with direct engine workers, an embedding request must land
    /// on an engine worker rather than 501 at request building on an SMG one,
    /// while generation may use either.
    #[test]
    fn requests_the_two_tier_lane_cannot_serve_stay_off_smg_workers() {
        let model_id = "test-model-mixed-pool";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let smg: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8470")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .worker_mode(WorkerMode::Smg)
                .health_config(no_health_check())
                .build(),
        );
        let engine: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8471")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        worker_registry.register(Arc::clone(&smg)).unwrap();
        worker_registry.register(Arc::clone(&engine)).unwrap();
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
        );

        // Generation may use either worker: round-robin over both reaches the
        // SMG Worker within two picks.
        let generation_picks: Vec<String> = (0..4)
            .map(|_| {
                stage
                    .select_single_worker(model_id, None, None, None, None, None, LaneFilter::Any)
                    .expect("either worker serves generation")
                    .url()
                    .to_string()
            })
            .collect();
        assert!(generation_picks.iter().any(|url| url == smg.url()));
        assert!(generation_picks.iter().any(|url| url == engine.url()));

        for _ in 0..4 {
            let picked = stage
                .select_single_worker(
                    model_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    LaneFilter::EngineOnly,
                )
                .expect("the engine worker can serve");
            assert_eq!(
                picked.url(),
                engine.url(),
                "excluded lane must never be picked"
            );
        }

        // A retry pins the mode it planned for: an SMG plan cannot be replayed
        // against a direct engine worker's proto even on the same runtime and
        // transport, and vice versa.
        let pinned = |mode| {
            stage
                .select_single_worker(
                    model_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    LaneFilter::Wire(WireConstraint {
                        runtime: RuntimeType::Unspecified,
                        connection: ConnectionMode::Grpc,
                        mode,
                    }),
                )
                .expect("pinned mode has a worker")
        };
        assert_eq!(pinned(WorkerMode::Smg).url(), smg.url());
        assert_eq!(pinned(WorkerMode::Engine).url(), engine.url());

        // Only SMG Workers left: the excluded request has nowhere to go.
        worker_registry.remove_by_url(engine.url());
        assert!(stage
            .select_single_worker(
                model_id,
                None,
                None,
                None,
                None,
                None,
                LaneFilter::EngineOnly
            )
            .is_none());
    }

    /// The namespace lives on the routing snapshot so a retry re-selects in
    /// the same cache partition: the retained snapshot, not a re-derivation
    /// from a request that may already be released, decides the key.
    #[test]
    fn reselect_keys_affinity_under_the_retained_cache_namespace() {
        use openai_protocol::common::CachePartition;

        let model_id = "namespace-retry-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let workers: Vec<Arc<dyn Worker>> = ["grpc://127.0.0.1:9401", "grpc://127.0.0.1:9402"]
            .iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Vllm)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        for worker in &workers {
            worker_registry.register(Arc::clone(worker)).unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 0,
                max_tree_size: 4096,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
            })),
            WorkerSelectionMode::Regular,
        );
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            mode: WorkerMode::Engine,
        };
        let namespace = |salt: &str| {
            CacheNamespace::derive(&CachePartition {
                cache_salt: Some(salt),
                extra_key: None,
                lora_path: None,
            })
        };
        let selected = |ctx: &DispatchContext| match ctx.workers.as_ref().unwrap() {
            WorkerSelection::Single { worker } => worker.url().to_string(),
            WorkerSelection::Disaggregated { .. } => panic!("expected single selection"),
        };
        let prompt: Vec<u32> = (1..33).collect();

        // Worker 1 is busier, so tenant A's first selection lands on worker 2.
        workers[0].increment_load();
        let mut ctx = dispatch_ctx(model_id, wire);
        ctx.routing.token_ids = prompt.clone();
        ctx.routing.cache_namespace = namespace("tenant-a");
        stage.reselect(&mut ctx).unwrap();
        assert_eq!(selected(&ctx), "grpc://127.0.0.1:9402");

        // A retry from the retained snapshot stays in tenant A's partition
        // even once worker 2 is the busier one...
        workers[1].increment_load();
        workers[1].increment_load();
        stage.reselect(&mut ctx).unwrap();
        assert_eq!(selected(&ctx), "grpc://127.0.0.1:9402");

        // ...while a retained snapshot for tenant B misses and takes the
        // least-loaded worker 1.
        let mut other = dispatch_ctx(model_id, wire);
        other.routing.token_ids = prompt;
        other.routing.cache_namespace = namespace("tenant-b");
        stage.reselect(&mut other).unwrap();
        assert_eq!(selected(&other), "grpc://127.0.0.1:9401");
    }
}
