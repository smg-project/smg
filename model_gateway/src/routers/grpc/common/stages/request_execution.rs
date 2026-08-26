//! Request execution stage: execute gRPC requests from an execution plan.

use std::time::Instant;

use async_trait::async_trait;
use axum::response::Response;
use futures::future::{join_all, try_join_all};
use tracing::{debug, error, info_span, Instrument};

use super::{
    pd_protocol::{DpPlacement, PdDispatch, PdProtocol},
    PipelineStage,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::kv_transfer::{
            connector_mode_for_worker, mooncake_decode_params, mooncake_prefill_params,
            KvConnectorMode, NIXL_PREFILL_KV_PARAMS,
        },
        error,
        grpc::{
            common::stages::encode::EncodeDispatchPlan,
            context::{
                ClientSelection, ExecutionPlan, ExecutionPlanKind, ExecutionResult, LoadGuards,
                PdTiming, RequestContext, WorkerSelection,
            },
            proto_wrapper::{
                ProtoEmbedRequest, ProtoGenerateRequest, ProtoRequest, ProtoResponseVariant,
                ProtoStream,
            },
            utils::tonic_ext::{TonicResultExt, TonicStatusExt},
        },
    },
    worker::ConnectionModeExt,
};

type StreamResult = Result<ProtoStream, tonic::Status>;

/// Metric connection labels for the PD legs (a leg can be gRPC or ZMQ).
fn pd_leg_labels(workers: &WorkerSelection) -> (&'static str, &'static str) {
    match workers {
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => (
            prefill.connection_mode().as_metric_label(),
            decode.connection_mode().as_metric_label(),
        ),
        WorkerSelection::Single { worker } => {
            let label = worker.connection_mode().as_metric_label();
            (label, label)
        }
    }
}

/// Request execution stage: execute the plan produced by request building.
pub(crate) struct RequestExecutionStage;

impl RequestExecutionStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let execution_plan = ctx.state.execution_plan.take().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Execution plan not built"
            );
            error::internal_error("execution_plan_not_built", "Execution plan not built")
        })?;

        // `None` for non-EPD or text-only EPD. Taking it transfers the encode
        // jobs' SHM Drop guards here: dispatch consumes them, while an early
        // error before dispatch drops them and reclaims the SHM.
        let encode_dispatch = ctx.state.encode_outputs.take().map(|o| o.dispatch);

        let clients = ctx.state.clients.as_mut().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        // Create load guards for worker load tracking (increment load when created)
        // They will be automatically dropped (and decrement load) when RequestContext is dropped
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Worker selection not completed"
            );
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
        })?;

        let sub_requests = match &execution_plan {
            ExecutionPlan::Batch { requests, .. } => requests.len(),
            _ => 1,
        };
        ctx.state.load_guards = Some(LoadGuards::scaled(
            workers,
            ctx.state.sticky_key.as_deref(),
            sub_requests,
        ));

        // Extract dispatch metadata for tracing span
        let dispatch = ctx.state.dispatch.as_ref();
        let request_id = dispatch.map(|d| d.request_id.as_str()).unwrap_or("unknown");
        let model = dispatch.map(|d| d.model.as_str()).unwrap_or("unknown");
        let request_type = execution_plan.request_type();
        let mode = execution_plan.mode_label();

        // Create OTEL span for gRPC request execution
        let span = info_span!(
            target: "smg::otel-trace",
            "grpc_execute",
            request_type,
            request_id = %request_id,
            model = %model,
            mode = %mode,
        );

        let result = async {
            match execution_plan {
                ExecutionPlan::Single(request) => match request {
                    ProtoRequest::Generate(req) => self.execute_single(req, clients, workers).await,
                    ProtoRequest::Embed(req) => {
                        self.execute_single_embed(req, clients, workers).await
                    }
                },
                ExecutionPlan::PrefillDecode(req) => {
                    self.execute_pd_dispatch(req, clients, workers, model).await
                }
                ExecutionPlan::EncodePrefillDecode { request } => {
                    // Bootstrap info was injected into the prefill request during
                    // request building; dispatch the encode jobs with the
                    // prefill+decode leg.
                    self.execute_epd_dispatch(request, clients, workers, model, encode_dispatch)
                        .await
                }
                ExecutionPlan::Batch { kind, requests, .. } => {
                    self.execute_batch_dispatch(kind, requests, clients, workers, model)
                        .await
                }
            }
        }
        .instrument(span)
        .await?;

        // Store result in context for ResponseProcessingStage
        ctx.state.response.execution_result = Some(result);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}

impl RequestExecutionStage {
    async fn execute_pd_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
        model: &str,
    ) -> Result<ExecutionResult, Response> {
        let Some(runtime_type) = workers.disaggregated_runtime_type() else {
            error!(
                function = "RequestExecutionStage::execute",
                "PD mode requires disaggregated worker selection"
            );
            return Err(error::internal_error(
                "pd_mode_requires_disaggregated_workers",
                "PD mode requires disaggregated worker selection",
            ));
        };
        let Some(protocol) = PdProtocol::for_runtime(*runtime_type) else {
            error!(
                function = "RequestExecutionStage::execute",
                runtime_type = ?runtime_type,
                "Runtime does not support PD disaggregated mode"
            );
            return Err(error::bad_request(
                "runtime_pd_not_supported",
                "This runtime does not support PD disaggregated mode",
            ));
        };
        // Dispatch shape comes from the per-runtime PD protocol table (see
        // `PdProtocol::for_runtime`): sequential legs relay the KV handoff
        // through the router, parallel legs rendezvous on bootstrap info
        // carried in the request.
        match protocol.dispatch {
            PdDispatch::Sequential => {
                self.execute_sequential_pd(proto_request, clients, workers, model)
                    .await
            }
            PdDispatch::Parallel => {
                self.execute_parallel_pd(proto_request, clients, workers, protocol)
                    .await
            }
        }
    }

    async fn execute_epd_dispatch(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
        model: &str,
        encode_dispatch: Option<EncodeDispatchPlan>,
    ) -> Result<ExecutionResult, Response> {
        if let Some(encode_dispatch) = encode_dispatch {
            Self::spawn_encode_dispatch(encode_dispatch);
        }
        proto_request.clear_mm_pixel_values();
        self.execute_pd_dispatch(proto_request, clients, workers, model)
            .await
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "EPD encode dispatch is intentionally supervised in the background while the prefill leg blocks on embedding receive."
    )]
    fn spawn_encode_dispatch(encode_dispatch: EncodeDispatchPlan) {
        if encode_dispatch.is_empty() {
            return;
        }

        let num_encode_items = encode_dispatch.len();
        let sends: Vec<_> = encode_dispatch
            .into_jobs()
            .into_iter()
            .map(|job| tokio::spawn(async move { job.dispatch().await }))
            .collect();

        debug!(
            num_encode_items,
            "EPD encode dispatch issued with prefill/decode"
        );

        tokio::spawn(async move {
            for join_res in join_all(sends).await {
                match join_res {
                    Ok(Ok(())) => {}
                    Ok(Err(message)) => {
                        error!(
                            function = "RequestExecutionStage::execute_epd_dispatch",
                            error = %message,
                            "Backend encode dispatch failed after EPD dispatch; embedding-receive timeout will abort the request"
                        );
                    }
                    Err(join_err) => {
                        error!(
                            function = "RequestExecutionStage::execute_epd_dispatch",
                            error = %join_err,
                            "Encode dispatch task panicked after EPD dispatch"
                        );
                    }
                }
            }
        });
    }

    /// Dispatch one backend request per batched prompt concurrently, preserving
    /// prompt order. Fail-fast: the first failed dispatch fails the batch and
    /// drops the remaining streams (abort-on-drop reclaims them backend-side).
    async fn execute_batch_dispatch(
        &self,
        kind: ExecutionPlanKind,
        requests: Vec<ProtoGenerateRequest>,
        clients: &ClientSelection,
        workers: &WorkerSelection,
        model: &str,
    ) -> Result<ExecutionResult, Response> {
        let dispatches = requests.into_iter().map(|request| {
            let mut clients = clients.clone();
            async move {
                match kind {
                    ExecutionPlanKind::Single => {
                        self.execute_single(request, &mut clients, workers).await
                    }
                    // Completion EPD carries no encode jobs; sub-requests dispatch as PD.
                    ExecutionPlanKind::PrefillDecode | ExecutionPlanKind::EncodePrefillDecode => {
                        self.execute_pd_dispatch(request, &mut clients, workers, model)
                            .await
                    }
                }
            }
        });

        let results = try_join_all(dispatches).await?;
        Ok(ExecutionResult::Batch { results })
    }

    async fn execute_single(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single",
                "Expected single client but got disaggregated"
            );
            error::internal_error(
                "expected_single_client_got_disaggregated",
                "Expected single client but got disaggregated",
            )
        })?;

        if let Some(rank) = workers.single().and_then(|w| w.dp_rank()) {
            proto_request.set_data_parallel_rank(rank as i32);
        }

        let result = client.generate(proto_request).await;
        workers.record_outcome(result.cb_status_code());

        let stream = result.map_err(|e| {
            error!(function = "execute_single", error = %e, "Failed to start generation");
            e.to_http_error(
                "start_generation_failed",
                format!("Failed to start generation: {}", e.message()),
            )
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_single_embed(
        &self,
        proto_request: ProtoEmbedRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single_embed",
                "Expected single client but got disaggregated"
            );
            error::internal_error(
                "expected_single_client_got_disaggregated",
                "Expected single client but got disaggregated",
            )
        })?;

        let result = client.embed(proto_request).await;
        workers.record_outcome(result.cb_status_code());

        let complete = result.map_err(|e| {
            error!(function = "execute_single_embed", error = %e, "Failed to start embedding");
            e.to_http_error(
                "start_embedding_failed",
                format!("Failed to start embedding: {}", e.message()),
            )
        })?;

        Ok(ExecutionResult::Embedding { response: complete })
    }

    async fn execute_parallel_pd(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
        protocol: PdProtocol,
    ) -> Result<ExecutionResult, Response> {
        let runtime = workers
            .disaggregated_runtime_type()
            .map(|r| r.as_str())
            .unwrap_or("");
        let (prefill_client, decode_client) = clients.disaggregated_mut().ok_or_else(|| {
            error!(
                function = "execute_parallel_pd",
                "Expected disaggregated clients but got single"
            );
            error::internal_error(
                "expected_disaggregated_clients_got_single",
                "Expected disaggregated clients but got single",
            )
        })?;

        // Decode consumes the KV handoff from prefill, but TokenSpeed still
        // needs multimodal metadata to pad placeholders and compute MRoPE in
        // the same way as prefill. Drop raw pixels and prefill-only encode
        // rooms, but keep the per-item metadata. The pixel-free leg is the
        // clone, so pixel tensors are never duplicated.
        let mut prefill_request = proto_request;
        let mut decode_request = prefill_request.clone_without_mm_pixels();
        decode_request.clear_encode_bootstrap_info();
        // Pin each leg's DP rank only when the engine reads placement from
        // the request field; `RoomResidue` engines take it from the bootstrap
        // room minted in maybe_inject_pd_rendezvous, ignore the pin, and spam
        // conflict warnings on a mismatched decode-leg pin.
        if protocol.dp_placement == DpPlacement::PinField {
            if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
                prefill_request.set_data_parallel_rank(rank as i32);
            }
            if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
                decode_request.set_data_parallel_rank(rank as i32);
            }
        }

        // `generate` only establishes the prefill stream here (SMG does not drain
        // it on this path), so prefill duration cannot be measured — only TTFT,
        // recorded at the first decode token in streaming. prefill_start anchors it.
        let prefill_start = Instant::now();
        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Record circuit breaker outcomes (client errors don't count as failures)
        workers.record_prefill_decode_outcomes(
            prefill_result.cb_status_code(),
            decode_result.cb_status_code(),
        );

        let (prefill_label, decode_label) = pd_leg_labels(workers);

        // Handle prefill result
        let prefill_stream = prefill_result.map_err(|e| {
            Metrics::record_worker_error(
                metrics_labels::WORKER_PREFILL,
                prefill_label,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_parallel_pd", error = %e, "Prefill worker failed to start");
            e.to_http_error(
                "prefill_worker_failed_to_start",
                format!("Prefill worker failed to start: {}", e.message()),
            )
        })?;

        // Handle decode result
        let decode_stream = decode_result.map_err(|e| {
            Metrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                decode_label,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_parallel_pd", error = %e, "Decode worker failed to start");
            e.to_http_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e.message()),
            )
        })?;

        // A client disconnect drops both leg streams, which normally fires an
        // immediate abort to each worker. The decode leg must not be aborted
        // while it is still receiving the KV handoff from prefill — tearing
        // the request down mid-transfer can crash or leak on the engine — so
        // its abort is deferred until the first decode response (the proof
        // the handoff completed). The prefill leg keeps the immediate abort:
        // if prefill is still running there is nothing to hand off yet, and
        // stopping it promptly frees capacity.
        let decode_stream = decode_stream.defer_abort_until_first_item();

        Ok(ExecutionResult::PrefillDecode {
            prefill: prefill_stream,
            decode: Box::new(decode_stream),
            pd_timing: PdTiming {
                prefill_start,
                runtime,
            },
        })
    }

    /// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
    /// then send original request to decode.
    ///
    /// For Mooncake: injects bootstrap_host/port from prefill worker metadata into
    /// the decode request. For NIXL: tags the prefill request with do_remote_decode,
    /// then relays the kv_transfer_params returned by the prefill engine to decode.
    async fn execute_sequential_pd(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
        model: &str,
    ) -> Result<ExecutionResult, Response> {
        let runtime = workers
            .disaggregated_runtime_type()
            .map(|r| r.as_str())
            .unwrap_or("");
        let (prefill_client, decode_client) = clients.disaggregated_mut().ok_or_else(|| {
            error!(
                function = "execute_sequential_pd",
                "Expected disaggregated clients but got single"
            );
            error::internal_error(
                "expected_disaggregated_clients_got_single",
                "Expected disaggregated clients but got single",
            )
        })?;

        let mode = workers
            .prefill_worker()
            .map(|w| connector_mode_for_worker(w.as_ref()))
            .unwrap_or(KvConnectorMode::Passthrough);

        // Recorded on the success path (after decode established) so failed
        // requests don't pollute success metrics; captured here before use of mode.
        let kv_connector_label = mode.metrics_label();

        match &mode {
            KvConnectorMode::Mooncake {
                host,
                port,
                engine_id,
            } => debug!(
                bootstrap_host = %host,
                bootstrap_port = port,
                engine_id_known = engine_id.is_some(),
                "vLLM PD (Mooncake): will inject kv_transfer_params into decode request"
            ),
            KvConnectorMode::Nixl => debug!(
                "vLLM PD (NIXL): will tag prefill with do_remote_decode and relay returned kv_transfer_params to decode"
            ),
            KvConnectorMode::Passthrough => {
                // Warn once: PD without a discovered connector usually means GetServerInfo
                // lacks kv fields or labels.kv_connector is missing in worker config
                static WARN_ONCE: std::sync::Once = std::sync::Once::new();
                WARN_ONCE.call_once(|| {
                    tracing::warn!(
                        "vLLM PD: no kv_connector detected on prefill worker; KV transfer params \
                         will only be relayed if the engine returns them"
                    );
                });
            }
        }

        // The KV handoff is single-consumer: with n>1 each fan-out child on decode
        // would pull, and the first completion frees the prefill blocks under its
        // siblings (same hazard for NIXL and Mooncake)
        let relay_kv_params = proto_request.sampling_n() <= 1;

        // Mooncake is push-based: the engine returns nothing, so the router mints
        // the transfer correlation id and synthesizes decode params from metadata
        let mooncake_transfer_id = match &mode {
            KvConnectorMode::Mooncake {
                engine_id: Some(_), ..
            } if relay_kv_params => Some(format!("xfer-{}", uuid::Uuid::now_v7())),
            _ => None,
        };

        // Decode reuses the request minus pixels: it receives KV via the P/D
        // transfer, and prefill reads and unlinks any /dev/shm segments, so a
        // reused ShmHandle would be unreadable. Same request_id on both legs
        // is load-bearing for NIXL P/D correlation on vLLM < 0.13. The
        // pixel-free leg is the clone, so pixel tensors are never duplicated
        // and die with the prefill send.
        let mut decode_request = proto_request.clone_without_mm_pixels();
        // Sanitize prefill sampling (max_tokens=1, n=1), stream=false.
        let mut prefill_request = proto_request;
        prefill_request.sanitize_sampling_for_prefill(1);
        prefill_request.set_stream(false);
        if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
            prefill_request.set_data_parallel_rank(rank as i32);
        }
        if mode == KvConnectorMode::Nixl {
            if relay_kv_params {
                prefill_request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
            } else {
                debug!(
                    request_id = %prefill_request.request_id(),
                    "vLLM PD (NIXL): n>1 request, skipping kv_transfer_params relay \
                     (decode recomputes the prompt locally)"
                );
            }
        }
        if let Some(ref transfer_id) = mooncake_transfer_id {
            prefill_request.set_kv_transfer_params_json(mooncake_prefill_params(transfer_id));
        }

        debug!(
            request_id = %prefill_request.request_id(),
            "vLLM PD: sending prefill request (max_tokens=1)"
        );

        // Send to prefill, wait for completion
        let (prefill_label, decode_label) = pd_leg_labels(workers);
        let prefill_start = Instant::now();
        let mut prefill_stream = prefill_client
            .generate(prefill_request)
            .await
            .map_err(|e| {
                workers.record_outcome_prefill(e.http_status().as_u16());
                Metrics::record_worker_error(
                    metrics_labels::WORKER_PREFILL,
                    prefill_label,
                    metrics_labels::ERROR_BACKEND,
                );
                error!(function = "execute_sequential_pd", error = %e, "Prefill worker failed to start");
                e.to_http_error("prefill_worker_failed_to_start", format!("Prefill worker failed to start: {}", e.message()))
            })?;

        // Drain prefill response, harvesting connector params from the Complete frame
        let mut prefill_kv_params: Option<String> = None;
        while let Some(result) = prefill_stream.next().await {
            match result {
                Ok(response) => {
                    if let ProtoResponseVariant::Complete(complete) = response.into_response() {
                        if let Some(json) = complete.kv_transfer_params_json() {
                            prefill_kv_params = Some(json.to_owned());
                        }
                    }
                }
                Err(e) => {
                    workers.record_outcome_prefill(e.http_status().as_u16());
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_PREFILL,
                        prefill_label,
                        metrics_labels::ERROR_BACKEND,
                    );
                    error!(function = "execute_sequential_pd", error = %e, "Prefill stream error");
                    return Err(e.to_http_error(
                        "prefill_stream_error",
                        format!("Prefill stream error: {}", e.message()),
                    ));
                }
            }
        }
        prefill_stream.mark_completed();
        workers.record_outcome_prefill(200);
        // Captured at drain; recorded below only once decode is established.
        let prefill_duration = prefill_start.elapsed();

        // KV-transfer window: prefill drain complete to decode send complete.
        let kv_window_start = Instant::now();

        debug!("vLLM PD: prefill completed, sending decode request");

        if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
            decode_request.set_data_parallel_rank(rank as i32);
        }
        match (&mode, prefill_kv_params) {
            // Modern Mooncake: synthesized params under the minted transfer_id
            (
                KvConnectorMode::Mooncake {
                    host,
                    port,
                    engine_id: Some(engine_id),
                },
                _,
            ) if mooncake_transfer_id.is_some() => {
                let transfer_id = mooncake_transfer_id.as_deref().unwrap_or_default();
                debug!(
                    request_id = %decode_request.request_id(),
                    transfer_id = %transfer_id,
                    "vLLM PD (Mooncake): injecting minted kv_transfer_params into decode request"
                );
                decode_request.set_kv_transfer_params_json(mooncake_decode_params(
                    transfer_id,
                    engine_id,
                    host,
                    *port,
                ));
            }
            // Legacy Mooncake (no engine_id discovered, or n>1): typed host/port injection
            (KvConnectorMode::Mooncake { host, port, .. }, _) => {
                debug!(
                    remote_host = %host,
                    remote_port = port,
                    "vLLM PD: injecting kv_transfer_params into decode request"
                );
                decode_request.set_kv_transfer_params(host.clone(), *port);
            }
            (KvConnectorMode::Nixl | KvConnectorMode::Passthrough, Some(json))
                if relay_kv_params =>
            {
                debug!(
                    request_id = %decode_request.request_id(),
                    params_len = json.len(),
                    "vLLM PD: relaying prefill kv_transfer_params to decode request"
                );
                decode_request.set_kv_transfer_params_json(json);
            }
            (KvConnectorMode::Nixl, None) if relay_kv_params => {
                Metrics::record_pd_kv_transfer_failure();
                tracing::warn!(
                    request_id = %decode_request.request_id(),
                    "vLLM PD (NIXL): prefill returned no kv_transfer_params; decode will \
                     recompute the prompt locally (outdated smg-grpc-servicer or missing \
                     kv-transfer-config?)"
                );
            }
            _ => {}
        }

        // Send request to decode
        let decode_stream = decode_client.generate(decode_request).await.map_err(|e| {
            workers.record_outcome_decode(e.http_status().as_u16());
            Metrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                decode_label,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_sequential_pd", error = %e, "Decode worker failed to start");
            e.to_http_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e.message()),
            )
        })?;

        workers.record_outcome_decode(200);
        // Decode established: record the success-only PD metrics here.
        Metrics::record_pd_kv_connector_mode(kv_connector_label);
        Metrics::record_pd_prefill_duration(
            metrics_labels::BACKEND_PD,
            model,
            runtime,
            prefill_duration,
        );
        Metrics::record_pd_kv_transfer_duration(
            metrics_labels::BACKEND_PD,
            model,
            runtime,
            kv_window_start.elapsed(),
        );

        // Prefill has completed and its KV blocks are held pending decode's
        // transfer; aborting decode mid-transfer on a client disconnect can
        // crash or leak on the engine side. Defer the abort until the first
        // decode response proves the handoff finished (see the parallel PD
        // path for the same invariant).
        let decode_stream = decode_stream.defer_abort_until_first_item();

        Ok(ExecutionResult::Single {
            stream: decode_stream,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as vllm};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType};

    #[test]
    fn pd_leg_labels_reflect_each_legs_transport() {
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("ipc:///tmp/smg-test-prefill")
                .worker_type(WorkerType::Prefill)
                .connection_mode(ConnectionMode::Zmq)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        let selection = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode,
            runtime_type: RuntimeType::TokenSpeed,
        };
        assert_eq!(
            pd_leg_labels(&selection),
            (
                metrics_labels::CONNECTION_ZMQ,
                metrics_labels::CONNECTION_GRPC
            ),
            "each PD leg must carry its own transport label"
        );
    }

    #[test]
    fn mooncake_prefill_params_carry_transfer_id() {
        let value: serde_json::Value =
            serde_json::from_str(&mooncake_prefill_params("xfer-abc")).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value.as_object().unwrap().len(), 3);
    }

    #[test]
    fn mooncake_decode_params_synthesize_full_handoff() {
        let value: serde_json::Value = serde_json::from_str(&mooncake_decode_params(
            "xfer-abc", "engine-1", "10.0.0.1", 8998,
        ))
        .unwrap();
        assert_eq!(value["do_remote_decode"], false);
        assert_eq!(value["do_remote_prefill"], true);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value["remote_engine_id"], "engine-1");
        assert_eq!(value["remote_bootstrap_addr"], "http://10.0.0.1:8998");
        assert_eq!(value.as_object().unwrap().len(), 5);
    }

    #[test]
    fn nixl_prefill_kv_params_is_valid_json() {
        let value: serde_json::Value = serde_json::from_str(NIXL_PREFILL_KV_PARAMS).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value.as_object().unwrap().len(), 2);
    }

    #[test]
    fn clone_without_mm_pixels_keeps_pixels_on_the_original_only() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "pd-1".to_string(),
            mm_inputs: Some(vllm::MultimodalInputs::default()),
            ..Default::default()
        }));
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::Vllm(original) = request else {
            panic!("expected vLLM request");
        };
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };
        assert!(original.mm_inputs.is_some(), "prefill leg keeps pixels");
        assert!(
            decode.mm_inputs.is_none(),
            "decode leg never carries pixels"
        );
        assert_eq!(decode.request_id, "pd-1");
    }

    #[test]
    fn clone_without_mm_pixels_keeps_tokenspeed_item_metadata() {
        let item = ts::MultimodalItem {
            encoder_input: Some(ts::TensorData::default()),
            placeholder_token_id: Some(7),
            ..Default::default()
        };
        let mut request = ProtoGenerateRequest::TokenSpeed(Box::new(ts::GenerateRequest {
            mm_inputs: Some(ts::MultimodalInputs { items: vec![item] }),
            ..Default::default()
        }));
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::TokenSpeed(original) = request else {
            panic!("expected TokenSpeed request");
        };
        let ProtoGenerateRequest::TokenSpeed(decode) = clone else {
            panic!("expected TokenSpeed clone");
        };
        let original_item = &original.mm_inputs.as_ref().unwrap().items[0];
        assert!(
            original_item.encoder_input.is_some(),
            "prefill leg keeps encoder input"
        );
        let decode_item = &decode.mm_inputs.as_ref().unwrap().items[0];
        assert!(
            decode_item.encoder_input.is_none(),
            "decode leg drops encoder input"
        );
        assert_eq!(
            decode_item.placeholder_token_id,
            Some(7),
            "decode leg keeps per-item metadata"
        );
    }

    #[test]
    fn sanitize_sampling_for_prefill_forces_length_capped_finish() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(128),
                min_tokens: 16,
                n: 4,
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![2],
                ignore_eos: false,
                ..Default::default()
            }),
            ..Default::default()
        }));
        request.sanitize_sampling_for_prefill(1);
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        let params = req.sampling_params.unwrap();
        assert_eq!(params.max_tokens, Some(1));
        assert_eq!(params.min_tokens, 0);
        assert_eq!(params.n, 1);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
        assert!(params.ignore_eos);
    }

    #[test]
    fn sampling_n_defaults_to_one() {
        let unset = ProtoGenerateRequest::Vllm(Box::default());
        assert_eq!(unset.sampling_n(), 1);

        let zero = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 0,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(zero.sampling_n(), 1);

        let four = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 4,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(four.sampling_n(), 4);
    }

    #[test]
    fn kv_transfer_params_json_request_roundtrip() {
        let mut request = ProtoGenerateRequest::Vllm(Box::default());
        request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        assert_eq!(
            req.kv_transfer_params_json.as_deref(),
            Some(NIXL_PREFILL_KV_PARAMS)
        );
    }

    #[test]
    fn kv_transfer_params_json_complete_accessor_filters_empty() {
        use crate::routers::grpc::proto_wrapper::ProtoGenerateComplete;

        let complete = ProtoGenerateComplete::Vllm(vllm::GenerateComplete {
            kv_transfer_params_json: Some(r#"{"do_remote_prefill":true}"#.to_string()),
            ..Default::default()
        });
        assert_eq!(
            complete.kv_transfer_params_json(),
            Some(r#"{"do_remote_prefill":true}"#)
        );

        let empty = ProtoGenerateComplete::Vllm(vllm::GenerateComplete {
            kv_transfer_params_json: Some(String::new()),
            ..Default::default()
        });
        assert_eq!(empty.kv_transfer_params_json(), None);

        let unset = ProtoGenerateComplete::Vllm(vllm::GenerateComplete::default());
        assert_eq!(unset.kv_transfer_params_json(), None);
    }
}
