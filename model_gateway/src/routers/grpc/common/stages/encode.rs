//! EPD encode stage: plan the encode handoff the selection's runtime takes.
//!
//! Runs after client acquisition and before request building and lands its
//! plan in `ProcessingState::encode_outputs`:
//!
//! - a rendezvous handoff: borrows the multimodal intermediate, mints one
//!   bootstrap room per encode item and serializes the encode payload with
//!   pixels for the encode workers; request building injects the per-item
//!   bootstrap info into the prefill request and drops the prefill pixels,
//!   request execution dispatches the encode jobs alongside the prefill/decode
//!   dispatch;
//! - a push handoff: names the encoder the built prompt goes to first (under
//!   an image anchor, the encoders its images fan out to) and the consumer
//!   destination its transfers go to, and refuses what the handoff cannot
//!   carry; request execution runs that encode call before the
//!   prefill/decode dispatch.
//!
//! Also owns the rendezvous items' assembly and transport-specific tensor
//! payloads; the sends to the encode workers live in the backend layer.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use axum::response::Response;
use rand::RngExt;
use smg_grpc_client::{
    common_proto, tokenspeed_encoder::tokenspeed_encoder_proto as tokenspeed_encoder,
};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerLoadGuard};
use tracing::error;
use uuid::Uuid;

use super::{helpers::bootstrap_rendezvous, pd_protocol::EncodeHandoff, PipelineStage};
use crate::routers::{
    error,
    grpc::{
        backend_client::{
            backend_client_if_initialized, send_tokenspeed_encode_frame,
            send_tokenspeed_encode_rpc, BackendClient,
        },
        client::GrpcClient,
        context::{ClientSelection, EncodeOutputs, RequestContext, WorkerSelection},
        multimodal::{
            assemble_tokenspeed_for_encode, assembly_error_response,
            ec_push::{push_destination, ImageFanout, PushEncodePlan},
            mm_rdma_exporter, worker_supports_ec_transfer, MultimodalIntermediate,
        },
        proto_wrapper::{
            cleanup_mm_shm_handles, cleanup_tokenspeed_items_encoder_shm,
            collect_tokenspeed_multimodal_inputs_shm_handles, stage_tokenspeed_tensor_rdma,
            EncodeItemBootstrapInfo, TokenSpeedMultimodalData, TokenSpeedMultimodalItem,
        },
        zmq_client::EncodeRendezvous,
    },
};

/// No-op unless the request is multimodal and worker selection produced encode
/// assignments; otherwise `encode_outputs` stays `None` and downstream stages
/// take the plain prefill path.
pub(crate) struct EncodeStage;

impl EncodeStage {
    pub(crate) fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PipelineStage for EncodeStage {
    /// Plan the handoff the selection takes: the rendezvous rooms and encode
    /// jobs, or the push arm's encoder and destination.
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let Some(workers) = ctx.state.workers.as_ref() else {
            return Ok(());
        };
        if workers
            .encode_assignments()
            .is_none_or(|assignments| assignments.is_empty())
        {
            return Ok(());
        }

        // Under SGLang prepared-input EPD, media encoding is planned ahead of
        // preparation and dispatched directly during request execution. If a
        // prepared-input encoder is selected without the SGLang EPD adapter
        // enabled (e.g., via dynamic registration without `--backend sglang`),
        // refuse explicitly instead of falling through to Rendezvous.
        if workers.encode_handoff() == Some(EncodeHandoff::PreparedInput) {
            if matches!(
                ctx.state.encode_outputs,
                Some(EncodeOutputs::PreparedInput(_))
            ) {
                return Ok(());
            }
            return Err(error::bad_request(
                "sglang_epd_not_enabled",
                "SGLang prepared-input EPD requires --epd-disaggregation with --backend sglang",
            ));
        }

        // Under a push handoff the encoder takes the built prompt during
        // request execution; the stage names the encoder and the consumer
        // destination and refuses what the handoff cannot carry. There is no
        // per-item bootstrap rendezvous.
        if workers.encode_handoff() == Some(EncodeHandoff::Push) {
            let plan = plan_push(
                ctx.input.request_type.sampling_n(),
                workers,
                ctx.state.image_anchor,
            )?;
            ctx.state.encode_outputs = Some(EncodeOutputs::Push(plan));
            return Ok(());
        }

        let Some(intermediate) = ctx.state.multimodal_intermediate.as_ref() else {
            return Ok(());
        };

        let plan = build_plan(
            intermediate,
            ctx.state.clients.as_ref(),
            ctx.state.workers.as_ref(),
        )
        .map_err(|e| {
            error!(function = "EncodeStage::execute", error = %e, "Failed to plan EPD encode");
            assembly_error_response(&e)
        })?;

        // No items resolved to encode work: leave `encode_outputs` unset so
        // request building takes the plain prefill path (with pixels) rather
        // than the pixel-drop encode path.
        if plan.is_empty() {
            return Ok(());
        }

        let (bootstrap_info, dispatch) = plan.into_parts();
        ctx.state.encode_outputs = Some(EncodeOutputs::Rendezvous {
            bootstrap_info,
            dispatch,
        });
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Encode"
    }
}

pub(crate) struct EncodePlan {
    bootstrap_info: Vec<EncodeItemBootstrapInfo>,
    dispatch: EncodeDispatchPlan,
}

pub(crate) struct EncodeDispatchPlan {
    jobs: Vec<PreparedEncodeJob>,
}

pub(crate) struct PreparedEncodeJob {
    item: PreparedEncodeItem,
    target: EncodeTarget,
    /// The encode worker the job is assigned to: the send goes to it, and
    /// its load counts the send.
    worker: Arc<dyn Worker>,
    bootstrap_room: i64,
}

/// The transport that reaches the encode worker of a job.
pub(crate) enum EncodeTarget {
    /// A gRPC encode worker, by its URL.
    Grpc,
    /// A TokenSpeed encode server on the direct ZMQ lane: the worker's
    /// connection carries the request, with the rendezvous the request names
    /// (the server's own bootstrap address, which the prefill side pulls the
    /// embedding from).
    Zmq { rendezvous: EncodeRendezvous },
}

impl EncodePlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.dispatch.is_empty()
    }

    pub(crate) fn into_parts(self) -> (Vec<EncodeItemBootstrapInfo>, EncodeDispatchPlan) {
        (self.bootstrap_info, self.dispatch)
    }
}

impl EncodeDispatchPlan {
    fn new(jobs: Vec<PreparedEncodeJob>) -> Self {
        Self { jobs }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.jobs.len()
    }

    pub(crate) fn into_jobs(self) -> Vec<PreparedEncodeJob> {
        self.jobs
    }
}

impl PreparedEncodeJob {
    pub(crate) async fn dispatch(self) -> std::result::Result<(), String> {
        // Booked on its encode worker until the send ends, as a generate
        // call is on its worker.
        let _load = WorkerLoadGuard::with_key(Arc::clone(&self.worker), None);
        self.item
            .dispatch(self.worker.as_ref(), self.target, self.bootstrap_room)
            .await
    }
}

pub(crate) enum PreparedEncodeItem {
    TokenSpeed {
        item: Option<TokenSpeedMultimodalItem>,
        shm_enabled: bool,
        shm_min_bytes: usize,
        cleanup_on_drop: bool,
    },
}

impl PreparedEncodeItem {
    fn tokenspeed(item: TokenSpeedMultimodalItem, shm_enabled: bool, shm_min_bytes: usize) -> Self {
        Self::TokenSpeed {
            item: Some(item),
            shm_enabled,
            shm_min_bytes,
            cleanup_on_drop: true,
        }
    }

    pub(crate) async fn dispatch(
        mut self,
        worker: &dyn Worker,
        target: EncodeTarget,
        bootstrap_room: i64,
    ) -> std::result::Result<(), String> {
        match &mut self {
            Self::TokenSpeed {
                item,
                shm_enabled,
                shm_min_bytes,
                cleanup_on_drop,
            } => {
                let mut item = item
                    .take()
                    .ok_or_else(|| "encode item was already dispatched".to_string())?;
                *cleanup_on_drop = false;
                // Stage with bootstrap_room as the slot key (load-bearing), so
                // into_proto(false) below does not re-stage with a random key.
                // The ZMQ lane carries the tensor inline, never a remote pull.
                if let (EncodeTarget::Grpc, Some(exporter)) = (&target, mm_rdma_exporter()) {
                    item.encoder_input =
                        stage_tokenspeed_tensor_rdma(exporter, bootstrap_room, item.encoder_input);
                }
                let request = tokenspeed_encoder::EncodeRequest {
                    request_id: format!("encode-{}", Uuid::now_v7()),
                    mm_inputs: Some(
                        TokenSpeedMultimodalData {
                            items: vec![item],
                            shm_enabled: *shm_enabled,
                            shm_min_bytes: *shm_min_bytes,
                        }
                        .into_proto(false),
                    ),
                    items: vec![tokenspeed_encoder::EncodeItemAssignment { bootstrap_room }],
                };
                let shm_handles = request
                    .mm_inputs
                    .as_ref()
                    .map(collect_tokenspeed_multimodal_inputs_shm_handles)
                    .unwrap_or_default();
                let _shm_guard = TokenSpeedShmCleanupGuard(shm_handles);
                match target {
                    EncodeTarget::Grpc => {
                        send_tokenspeed_encode_rpc(worker.url().to_string(), request).await
                    }
                    EncodeTarget::Zmq { rendezvous } => {
                        send_tokenspeed_encode_frame(worker, &rendezvous, request).await
                    }
                }
            }
        }
    }
}

struct TokenSpeedShmCleanupGuard(Vec<common_proto::ShmHandle>);

impl Drop for TokenSpeedShmCleanupGuard {
    fn drop(&mut self) {
        cleanup_mm_shm_handles(&self.0);
    }
}

impl Drop for PreparedEncodeItem {
    fn drop(&mut self) {
        if let Self::TokenSpeed {
            item: Some(item),
            cleanup_on_drop: true,
            ..
        } = self
        {
            cleanup_tokenspeed_items_encoder_shm(std::slice::from_ref(item), None);
        }
    }
}

/// The push arm's plan from the selection: the encoder of the prompt (under
/// an image anchor, the encoders its images fan out to) and the consumer
/// side's destination, refused where the handoff cannot carry the request
/// (more than one sample, more than one encoder without an anchor to fan out
/// on, a consumer without the transfer).
fn plan_push(
    sampling_n: u32,
    workers: &WorkerSelection,
    image_anchor: Option<u32>,
) -> Result<PushEncodePlan, Response> {
    if sampling_n != 1 {
        return Err(error::bad_request(
            "ec_multiple_samples",
            "a push encode handoff requires n=1",
        ));
    }
    let WorkerSelection::Disaggregated {
        encode_assignments: Some(assignments),
        prefill,
        ..
    } = workers
    else {
        return Err(error::service_unavailable(
            "ec_encoder_unavailable",
            "No encoder selected for the push handoff",
        ));
    };
    let Some(first) = assignments.first() else {
        return Err(error::internal_error(
            "ec_encoder_assignment",
            "a push handoff takes an encoder",
        ));
    };
    if image_anchor.is_none() && assignments.len() > 1 {
        return Err(error::internal_error(
            "ec_encoder_assignment",
            "a push handoff takes one encoder per prompt",
        ));
    }
    if !worker_supports_ec_transfer(prefill.as_ref()) {
        return Err(error::service_unavailable(
            "ec_consumer_unsupported",
            "Prefill worker does not support EC transfer",
        ));
    }
    Ok(PushEncodePlan {
        encoder: Arc::clone(&first.worker),
        destination: push_destination(prefill.as_ref())?,
        fanout: image_anchor.map(|anchor| ImageFanout {
            anchor,
            assignments: assignments.clone(),
        }),
    })
}

fn build_plan(
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<EncodePlan> {
    let workers = workers.ok_or_else(|| anyhow!("Worker selection stage not completed"))?;
    // The ZMQ lane carries an encode request's tensors inline, so a request
    // with an encode server on it assembles without /dev/shm.
    let inline = workers.encode_assignments().is_some_and(|assignments| {
        assignments
            .iter()
            .any(|assignment| *assignment.worker.connection_mode() == ConnectionMode::Zmq)
    });
    let items = prepare_items(intermediate, clients, Some(workers), inline)?;
    if items.is_empty() {
        return Ok(EncodePlan {
            bootstrap_info: Vec::new(),
            dispatch: EncodeDispatchPlan::new(Vec::new()),
        });
    }

    let encode_assignments = workers
        .encode_assignments()
        .filter(|assignments| !assignments.is_empty())
        .ok_or_else(|| anyhow!("Encode planning requires EPD worker selection"))?
        .to_vec();

    if encode_assignments.len() != items.len() {
        return Err(anyhow!(
            "EPD encode item/assignment count mismatch: {} items, {} assignments",
            items.len(),
            encode_assignments.len()
        ));
    }

    let mut bootstrap_info = Vec::with_capacity(items.len());
    let mut jobs = Vec::with_capacity(items.len());
    for (global_index, (item, assignment)) in items.into_iter().zip(encode_assignments).enumerate()
    {
        if assignment.item_index != global_index {
            return Err(anyhow!(
                "EPD encode assignment order mismatch: expected item {}, got {}",
                global_index,
                assignment.item_index
            ));
        }

        // 63-bit room: no in-flight dedup, so a 2^31 space birthday-collides
        // under load (silent embedding cross-wire). See the proto field doc.
        let bootstrap_room = rand::rng().random_range(0..i64::MAX);

        let worker = &assignment.worker;
        let backend = backend_client_if_initialized(worker.as_ref())?;
        let (bootstrap_host, bootstrap_port) = bootstrap_rendezvous(
            worker.bootstrap_host(),
            worker.bootstrap_port(),
            backend
                .as_deref()
                .and_then(BackendClient::declared_bootstrap),
        );
        bootstrap_info.push(EncodeItemBootstrapInfo {
            item_index: global_index as u32,
            bootstrap_host: bootstrap_host.to_string(),
            bootstrap_port: i32::from(bootstrap_port),
            bootstrap_room,
        });
        let target = match worker.connection_mode() {
            ConnectionMode::Zmq => EncodeTarget::Zmq {
                rendezvous: EncodeRendezvous {
                    bootstrap_host: bootstrap_host.to_string(),
                    bootstrap_port: i32::from(bootstrap_port),
                },
            },
            ConnectionMode::Grpc | ConnectionMode::Http => EncodeTarget::Grpc,
        };
        jobs.push(PreparedEncodeJob {
            item,
            target,
            worker: Arc::clone(worker),
            bootstrap_room,
        });
    }

    Ok(EncodePlan {
        bootstrap_info,
        dispatch: EncodeDispatchPlan::new(jobs),
    })
}

/// The encode items of an EPD request, assembled for its encode workers
/// (`inline`: without /dev/shm, for an encode server on the ZMQ lane). A
/// TokenSpeed prefill worker takes the items' embeddings out of band over
/// either transport.
fn prepare_items(
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
    inline: bool,
) -> Result<Vec<PreparedEncodeItem>> {
    let clients = clients.ok_or_else(|| anyhow!("Client acquisition stage not completed"))?;
    match clients {
        ClientSelection::Disaggregated {
            prefill: BackendClient::Grpc(GrpcClient::TokenSpeed(_)),
            ..
        } => prepare_tokenspeed_items(intermediate, workers, inline),
        ClientSelection::Disaggregated {
            prefill: prefill @ BackendClient::Zmq(_),
            ..
        } if prefill.runtime_type() == RuntimeType::TokenSpeed => {
            prepare_tokenspeed_items(intermediate, workers, inline)
        }
        ClientSelection::Disaggregated { prefill, .. } => Err(anyhow!(
            "EPD encode is not implemented for {} backend",
            backend_name(prefill)
        )),
        ClientSelection::Single { .. } => {
            Err(anyhow!("Encode planning requires EPD client selection"))
        }
    }
}

fn prepare_tokenspeed_items(
    intermediate: &MultimodalIntermediate,
    workers: Option<&WorkerSelection>,
    inline: bool,
) -> Result<Vec<PreparedEncodeItem>> {
    let tokenspeed_mm = assemble_tokenspeed_for_encode(intermediate, workers, inline)?;
    let shm_enabled = tokenspeed_mm.shm_enabled;
    let shm_min_bytes = tokenspeed_mm.shm_min_bytes;
    Ok(tokenspeed_mm
        .items
        .into_iter()
        .map(|item| PreparedEncodeItem::tokenspeed(item, shm_enabled, shm_min_bytes))
        .collect())
}

/// Display name of the engine runtime behind a client, for diagnostics. Keyed on
/// the runtime, not the transport: a ZMQ worker reports its engine runtime
/// (vLLM or TokenSpeed) the same as a gRPC worker for that engine.
fn backend_name(client: &BackendClient) -> &'static str {
    match client.runtime_type() {
        RuntimeType::Sglang => "SGLang",
        RuntimeType::Vllm => "vLLM",
        RuntimeType::Trtllm => "TRT-LLM",
        RuntimeType::Mlx => "MLX",
        RuntimeType::TokenSpeed => "TokenSpeed",
        // A gRPC/ZMQ worker is always a local runtime; External (an upstream
        // OpenAI-compatible API, HTTP-proxied), Generic (an unidentified
        // OpenAI-compatible HTTP backend), and Unspecified never reach here.
        RuntimeType::External | RuntimeType::Generic | RuntimeType::Unspecified => "unknown",
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/stages/encode/tests.rs"]
mod tests;
