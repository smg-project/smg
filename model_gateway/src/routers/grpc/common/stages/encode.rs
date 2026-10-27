//! EPD encode stage: plan the encode-worker rendezvous and dispatch payloads.
//!
//! Runs after client acquisition and before request building. Borrows the
//! multimodal intermediate, mints one bootstrap room per encode item, and
//! serializes the encode payload with pixels for the encode workers, landing the
//! results in `ProcessingState::encode_outputs`:
//!
//! - request building injects the per-item bootstrap info into the prefill
//!   request and drops the prefill pixels;
//! - request execution dispatches the encode jobs alongside the prefill/decode
//!   dispatch.
//!
//! Also owns the backend wire details for EPD encode: item assembly,
//! transport-specific tensor payloads, and the request to the encode worker
//! (the encoder RPC of a gRPC worker, the encode frame of a TokenSpeed encode
//! server on the direct ZMQ lane).

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use axum::response::Response;
use rand::RngExt;
use smg_grpc_client::{
    common_proto,
    tokenspeed_encoder::{tokenspeed_encoder_proto as tokenspeed_encoder, TokenSpeedEncoderClient},
};
use tracing::error;
use uuid::Uuid;

use super::{helpers::bootstrap_rendezvous, PipelineStage};
use crate::{
    routers::grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        context::{ClientSelection, EncodeOutputs, RequestContext, WorkerSelection},
        multimodal::{
            assemble_tokenspeed_for_encode, assembly_error_response, mm_rdma_exporter,
            MultimodalIntermediate,
        },
        proto_wrapper::{
            cleanup_mm_shm_handles, cleanup_tokenspeed_items_encoder_shm,
            collect_tokenspeed_multimodal_inputs_shm_handles, stage_tokenspeed_tensor_rdma,
            EncodeItemBootstrapInfo, TokenSpeedMultimodalData, TokenSpeedMultimodalItem,
        },
        zmq_client::EncodeRendezvous,
    },
    worker::{ConnectionMode, RuntimeType, Worker},
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
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        if ctx
            .state
            .workers
            .as_ref()
            .and_then(WorkerSelection::encode_assignments)
            .is_none_or(|assignments| assignments.is_empty())
        {
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
        ctx.state.encode_outputs = Some(EncodeOutputs {
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
    bootstrap_room: i64,
}

/// The encode worker a job goes to, by the transport that reaches it.
pub(crate) enum EncodeTarget {
    /// A gRPC encode worker, by its URL.
    Grpc(String),
    /// A TokenSpeed encode server on the direct ZMQ lane: its worker, whose
    /// connection carries the request, and the rendezvous the request names
    /// (the server's own bootstrap address, which the prefill side pulls the
    /// embedding from).
    Zmq {
        worker: Arc<dyn Worker>,
        rendezvous: EncodeRendezvous,
    },
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
        self.item.dispatch(self.target, self.bootstrap_room).await
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
                if let (EncodeTarget::Grpc(_), Some(exporter)) = (&target, mm_rdma_exporter()) {
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
                    EncodeTarget::Grpc(endpoint) => {
                        send_tokenspeed_encode_rpc(endpoint, request).await
                    }
                    EncodeTarget::Zmq { worker, rendezvous } => {
                        send_tokenspeed_encode_frame(worker.as_ref(), &rendezvous, request).await
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
        let backend = worker.backend_client_if_initialized();
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
                worker: Arc::clone(worker),
                rendezvous: EncodeRendezvous {
                    bootstrap_host: bootstrap_host.to_string(),
                    bootstrap_port: i32::from(bootstrap_port),
                },
            },
            ConnectionMode::Grpc | ConnectionMode::Http => {
                EncodeTarget::Grpc(worker.url().to_string())
            }
        };
        jobs.push(PreparedEncodeJob {
            item,
            target,
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

/// Hand an encode request to a TokenSpeed encode server over its worker's
/// ZMQ connection. Nothing answers it: the server ships the embedding to the
/// prefill side over the transfer engine, as the gRPC encoder acks after its
/// send.
async fn send_tokenspeed_encode_frame(
    worker: &dyn Worker,
    rendezvous: &EncodeRendezvous,
    request: tokenspeed_encoder::EncodeRequest,
) -> std::result::Result<(), String> {
    let url = worker.url();
    let client = worker
        .get_backend_client()
        .await
        .map_err(|e| format!("encode worker {url} is not connected: {e}"))?
        .ok_or_else(|| format!("encode worker {url} has no ZMQ connection"))?;
    let BackendClient::Zmq(client) = client.as_ref() else {
        return Err(format!("encode worker {url} is not a direct-ZMQ worker"));
    };
    client
        .encode_tokenspeed(request, rendezvous)
        .await
        .map_err(|status| format!("encode request to {url} failed: {}", status.message()))
}

async fn send_tokenspeed_encode_rpc(
    endpoint: String,
    request: tokenspeed_encoder::EncodeRequest,
) -> std::result::Result<(), String> {
    let client = TokenSpeedEncoderClient::connect_cached(&endpoint)
        .await
        .map_err(|e| format!("connect to encode worker {endpoint} failed: {e}"))?;
    let response = client
        .encode(request)
        .await
        .map_err(|e| format!("encode RPC to {endpoint} failed: {}", e.message()))?;
    if !response.accepted {
        return Err(format!(
            "encode worker {endpoint} did not accept the request"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_plan_is_empty_and_yields_no_jobs() {
        let plan = EncodePlan {
            bootstrap_info: Vec::new(),
            dispatch: EncodeDispatchPlan::new(Vec::new()),
        };
        assert!(plan.is_empty());
        let (bootstrap_info, dispatch) = plan.into_parts();
        assert!(bootstrap_info.is_empty());
        assert_eq!(dispatch.len(), 0);
        assert!(dispatch.into_jobs().is_empty());
    }

    /// Real `/dev/shm` segments back the encode jobs, so these run only on
    /// Linux. They prove the encode jobs' SHM Drop guards move with the owning
    /// state: dropping it before dispatch reclaims the segment, while a
    /// dispatched item transfers ownership off the guard.
    #[cfg(target_os = "linux")]
    mod shm_lifecycle {
        use std::collections::HashMap;

        use smg_grpc_client::common_proto::ShmHandle;

        use super::*;
        use crate::routers::grpc::{
            context::{EncodeOutputs, ProcessingState},
            proto_wrapper::{
                mm_shm_dev_writable, write_tokenspeed_shm_with, TokenSpeedModality,
                TokenSpeedTensor,
            },
        };

        fn shm_path(name: &str) -> std::path::PathBuf {
            std::path::PathBuf::from("/dev/shm").join(name)
        }

        /// Build a `PreparedEncodeItem` whose encoder input is a freshly-created
        /// `/dev/shm` segment, plus the path to that segment. The item's `Drop`
        /// (cleanup_on_drop=true) is expected to unlink the segment.
        fn tokenspeed_item_backed_by_shm() -> (PreparedEncodeItem, std::path::PathBuf) {
            let payload = vec![7u8; 4096];
            let handle: ShmHandle = write_tokenspeed_shm_with(payload.len(), |out| {
                out.copy_from_slice(&payload);
                Ok(())
            })
            .expect("write /dev/shm segment");
            let path = shm_path(&handle.name);
            assert!(path.exists(), "SHM segment must exist after creation");

            let item = TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::shm(handle, vec![1, 4096], "float32".to_string()),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: None,
                mm_placeholders: Vec::new(),
                content_hash: vec![1, 2, 3],
            };
            (
                PreparedEncodeItem::tokenspeed(
                    item, /*shm_enabled=*/ true, /*shm_min_bytes=*/ 0,
                ),
                path,
            )
        }

        /// Dropping the owning `ProcessingState` (as a `RequestContext` drop
        /// would, on early return / cancellation) before dispatch must reclaim
        /// the encode jobs' `/dev/shm` segments via `PreparedEncodeItem`'s `Drop`.
        #[test]
        fn dropping_state_before_dispatch_reclaims_encode_shm() {
            if !mm_shm_dev_writable() {
                // Linux container without a usable /dev/shm; skip rather than fail.
                return;
            }

            let (item, path) = tokenspeed_item_backed_by_shm();
            let dispatch = EncodeDispatchPlan::new(vec![PreparedEncodeJob {
                item,
                target: EncodeTarget::Grpc("http://encode-worker:9000".to_string()),
                bootstrap_room: 42,
            }]);
            assert!(!dispatch.is_empty());

            let state = ProcessingState {
                encode_outputs: Some(EncodeOutputs {
                    bootstrap_info: Vec::new(),
                    dispatch,
                }),
                ..ProcessingState::default()
            };

            // Segment is still live while the state owns the plan.
            assert!(
                path.exists(),
                "SHM segment must stay live until the owning state drops"
            );

            // Drop the state without ever dispatching (cancellation / early return).
            drop(state);

            assert!(
                !path.exists(),
                "dropping the owning state before dispatch must unlink the encode SHM segment"
            );
        }

        /// dispatch() disarms the item's own Drop (cleanup_on_drop=false) and
        /// hands cleanup to the send-path guard, which reclaims the segment after
        /// the send attempt — so even a failed dispatch reclaims exactly once
        /// (no leak, and the disarmed Drop cannot double-unlink).
        #[tokio::test]
        async fn dispatch_reclaims_shm_via_send_path_guard() {
            if !mm_shm_dev_writable() {
                return;
            }

            let (item, path) = tokenspeed_item_backed_by_shm();

            // Bogus endpoint: the encode RPC fails, but the send-path guard still
            // reclaims the segment when dispatch() returns.
            let _ = item
                .dispatch(EncodeTarget::Grpc("http://127.0.0.1:1".to_string()), 7)
                .await;

            assert!(
                !path.exists(),
                "dispatch must reclaim the encode SHM segment (no leak, no double-unlink)"
            );
        }
    }
}
