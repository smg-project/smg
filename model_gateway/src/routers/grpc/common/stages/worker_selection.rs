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
    policies::{
        policy_filters_unavailable_workers, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo,
        WorkerLeg,
    },
    routers::{
        common::overload,
        error,
        grpc::{
            context::{EncodeWorkerAssignment, RequestContext, WorkerSelection},
            multimodal,
        },
    },
    worker::{
        ConnectionModeExt, HashRing, ModelWorkerSnapshot, RoutingPool, RuntimeType, Worker,
        WorkerRegistry, WorkerType,
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
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
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
        let rid_key = rid_key.as_deref();

        let model_id = ctx.input.model_id.as_str();
        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(model_id, text, tokens, headers, rid_key) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => return Err(self.selection_failure(model_id, &[WorkerType::Regular])),
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self.select_pd_pair(model_id, text, tokens, headers, rid_key) {
                    Some((prefill, decode, runtime_type)) => WorkerSelection::Disaggregated {
                        encode_assignments: None,
                        prefill,
                        decode,
                        runtime_type,
                    },
                    None => {
                        return Err(self.selection_failure(
                            model_id,
                            &[WorkerType::Prefill, WorkerType::Decode],
                        ))
                    }
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
                        return Err(self.selection_failure(model_id, legs));
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
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!("WorkerSelectionStage({:?})", self.mode)
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
    fn selection_failure(&self, model_id: &str, legs: &[WorkerType]) -> Response {
        for leg in legs {
            let candidates = self.leg_candidates(model_id, *leg);
            if let Some(shed) = overload::shed_if_all_overloaded(&candidates, model_id) {
                return shed;
            }
        }
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No available workers for model"
        );
        error::model_not_found(model_id)
    }

    /// The pool one leg selected over, *before* the `is_available()` filter,
    /// under the same worker-type and transport rules selection applied.
    /// Failure path only.
    fn leg_candidates(&self, model_id: &str, worker_type: WorkerType) -> Vec<Arc<dyn Worker>> {
        // One definition shared with selection: the same routing-pool
        // projection, before the `is_available()` filter. Regular selection
        // takes either gRPC-pipeline transport; the disaggregated legs are
        // gRPC-only (no KV rendezvous on ZMQ). The wildcard model maps to
        // the global snapshot — not the `unknown` model-index entry.
        let pool = match worker_type {
            WorkerType::Regular => RoutingPool::GrpcPipelineRegular,
            WorkerType::Prefill => RoutingPool::GrpcPrefill,
            WorkerType::Decode => RoutingPool::GrpcDecode,
            WorkerType::Encode => RoutingPool::GrpcEncode,
        };
        self.worker_registry
            .get_routing_pool(model_id, pool)
            .to_vec()
    }

    fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model. The gRPC router serves both gRPC
        // and direct-ZMQ workers, so accept either transport (not HTTP).
        let candidates = self
            .worker_registry
            .get_routing_pool(model_id, RoutingPool::GrpcPipelineRegular);

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        let filtered;
        let available: &[Arc<dyn Worker>] = if policy_filters_unavailable_workers(policy.as_ref()) {
            &candidates
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

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Select worker via the registry (applies the routing-key sticky override
        // when enabled; otherwise delegates to the configured policy).
        let idx = self.policy_registry.select_worker(
            &policy,
            available,
            &SelectWorkerInfo {
                request_text: text,
                tokens,
                headers,
                routing_key: self.policy_registry.resolve_routing_key(headers),
                rid_key,
                hash_ring,
                leg: WorkerLeg::Single,
            },
        )?;
        let selected = available[idx].clone();

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            selected.connection_mode().as_metric_label(),
            model_id,
            policy.name(),
        );

        Some(selected)
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

    fn select_pd_pair(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
    ) -> Option<PdWorkerPair> {
        // Both legs derive from ONE membership snapshot: separate pool
        // lookups could straddle a concurrent replacement and pair workers
        // that never coexisted. The pools are strictly gRPC (a ZMQ leg would
        // silently drop the PD bootstrap info, see `RoutingPool::GrpcPrefill`;
        // the wildcard model maps to the global snapshot), and availability
        // stays a live per-request check.
        let snapshot = self.worker_registry.get_routing_snapshot(model_id);
        let all_prefill = Self::available_workers(&snapshot, RoutingPool::GrpcPrefill);
        let all_decode = Self::available_workers(&snapshot, RoutingPool::GrpcDecode);

        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }

        if all_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Determine the runtime type from prefill workers.
        // All workers in a PD pair must use the same runtime.
        let first_runtime = all_prefill.first()?.metadata().spec.runtime_type;

        // Check for mixed runtimes in both prefill and decode pools
        let prefill_mixed = all_prefill
            .iter()
            .skip(1)
            .any(|w| w.metadata().spec.runtime_type != first_runtime);
        let decode_mixed = all_decode
            .iter()
            .any(|w| w.metadata().spec.runtime_type != first_runtime);

        if prefill_mixed || decode_mixed {
            warn!(
                "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
                prefill_mixed,
                decode_mixed,
                first_runtime
            );
        }

        let target_runtime = first_runtime;

        // Filter both pools to the target runtime
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if available_prefill.is_empty() || available_decode.is_empty() {
            warn!("No available PD pair for runtime {:?}", target_runtime);
            return None;
        }

        // Independent P/D policies so stateful ones (e.g. round_robin) don't share a counter.
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Prefill and decode are separate pools; tag each leg so the routing-key
        // override keys its sticky map per leg (a key sticks independently).
        let mut info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            routing_key: self.policy_registry.resolve_routing_key(headers),
            rid_key,
            hash_ring,
            leg: WorkerLeg::Prefill,
        };
        let prefill_idx =
            self.policy_registry
                .select_worker(&prefill_policy, &available_prefill, &info)?;
        info.leg = WorkerLeg::Decode;
        let decode_idx =
            self.policy_registry
                .select_worker(&decode_policy, &available_decode, &info)?;

        let model = model_id;

        // Record worker selection metrics for both prefill and decode
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            available_prefill[prefill_idx]
                .connection_mode()
                .as_metric_label(),
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            available_decode[decode_idx]
                .connection_mode()
                .as_metric_label(),
            model,
            decode_policy.name(),
        );

        Some((
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
            target_runtime,
        ))
    }

    /// Select per-item encode workers + a prefill/decode pair for EPD routing.
    ///
    /// Mirrors `select_pd_pair` but also assigns each multimodal item to an
    /// encode worker. prefill+decode are selected as a normal PD pair. All pools
    /// are filtered to a runtime shared by the selected encode/prefill/decode
    /// legs.
    fn select_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
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
                .select_pd_pair(model_id, None, None, None, None)
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
            .select_pd_pair(model_id, None, None, None, None)
            .is_some());

        for url in &prefill_urls {
            let worker = worker_registry.get_by_url(url).expect("registered");
            worker_registry.set_worker_overloaded(&worker, true);
        }

        assert!(
            stage
                .select_pd_pair(model_id, None, None, None, None)
                .is_none(),
            "the veto empties the prefill pool"
        );
        let response =
            stage.selection_failure(model_id, &[WorkerType::Prefill, WorkerType::Decode]);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_available_workers"
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
            stage.selection_failure(model_id, &[WorkerType::Prefill, WorkerType::Decode]);
        assert_eq!(text_only.status(), StatusCode::NOT_FOUND);

        // With encode demanded, the saturated encode pool is a shed.
        let with_encode = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode],
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
                .selection_failure("nobody", &[WorkerType::Prefill, WorkerType::Decode])
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
                .select_pd_pair(model_id, None, None, None, None)
                .is_none(),
            "ZMQ-only PD pools must not yield a pair"
        );

        // Adding gRPC legs makes selection succeed, and it never picks the ZMQ ones.
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);
        let (prefill, decode, _) = stage
            .select_pd_pair(model_id, None, None, None, None)
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
            .select_single_worker(model_id, None, None, Some(&poison), rid_key)
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
                )
                .unwrap();
            assert_eq!(again.url(), first.url(), "follow-up must pin by rid key");
        }
    }

    /// The gRPC transport must shed an all-overloaded model with the same 503
    /// `no_available_workers` the HTTP router uses. Its usual empty-pool answer
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
            .select_single_worker(model_id, None, None, None, None)
            .is_some());

        worker_registry.set_worker_overloaded(&workers[0], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None)
                .is_some(),
            "one eligible worker left still serves"
        );

        worker_registry.set_worker_overloaded(&workers[1], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None)
                .is_none(),
            "the veto empties the candidate pool"
        );

        let response = stage.selection_failure(model_id, &[WorkerType::Regular]);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "no_available_workers"
        );

        // Recovery re-admits, and the failure response goes back to 404 for a
        // genuinely absent model.
        worker_registry.set_worker_overloaded(&workers[0], false);
        assert!(stage
            .select_single_worker(model_id, None, None, None, None)
            .is_some());
        assert_eq!(
            stage
                .selection_failure("no-such-model", &[WorkerType::Regular])
                .status(),
            StatusCode::NOT_FOUND
        );
    }
}
