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
use tracing::error;
use uuid::Uuid;

use super::{helpers::bootstrap_rendezvous, pd_protocol::EncodeHandoff, PipelineStage};
use crate::{
    routers::{
        error,
        grpc::{
            backend_client::{
                send_tokenspeed_encode_frame, send_tokenspeed_encode_rpc, BackendClient,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::grpc::context::EncodeWorkerAssignment;

    fn ec_worker(url: &str, role: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        let mut builder = crate::worker::BasicWorkerBuilder::new(url)
            .runtime_type(RuntimeType::Vllm)
            .label("ec_role", role)
            .label("ec_connector", "ECMooncakeConnector");
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn push_selection(encoders: Vec<Arc<dyn Worker>>, prefill: Arc<dyn Worker>) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: Some(
                encoders
                    .into_iter()
                    .enumerate()
                    .map(|(item_index, worker)| EncodeWorkerAssignment { item_index, worker })
                    .collect(),
            ),
            decode: Arc::clone(&prefill),
            prefill,
            runtime_type: RuntimeType::Vllm,
        }
    }

    const CONSUMER: [(&str, &str); 2] = [
        ("supports_ec_transfer", "true"),
        ("ec_zmq_addrs", r#"["tcp://127.0.0.1:9101"]"#),
    ];

    /// A push handoff plans the one encoder and the consumer's destination;
    /// a pull connector plans no destination.
    #[test]
    fn a_push_handoff_plans_one_encoder_and_the_consumers_destination() {
        let encoder = ec_worker("grpc://127.0.0.1:9000", "ec_producer", &[]);
        let consumer = ec_worker("grpc://127.0.0.1:9001", "ec_consumer", &CONSUMER);
        let plan = plan_push(1, &push_selection(vec![encoder], consumer), None).expect("a plan");
        assert_eq!(plan.encoder.url(), "grpc://127.0.0.1:9000");
        assert_eq!(plan.destination.as_deref(), Some("tcp://127.0.0.1:9101"));
        assert!(plan.fanout.is_none());

        let pull = ec_worker(
            "grpc://127.0.0.1:9002",
            "ec_consumer",
            &[
                ("supports_ec_transfer", "true"),
                ("ec_connector", "ECCPUConnector"),
            ],
        );
        let encoder = ec_worker("grpc://127.0.0.1:9000", "ec_producer", &[]);
        let plan = plan_push(1, &push_selection(vec![encoder], pull), None).expect("a plan");
        assert!(plan.destination.is_none());
    }

    /// The push arm refuses what the handoff cannot carry: more than one
    /// sample, more than one encoder without an anchor to fan out on, a
    /// consumer without the transfer; under an anchor the encoders are the
    /// fan-out.
    #[test]
    fn a_push_handoff_refuses_what_it_cannot_carry() {
        use axum::http::StatusCode;
        let producer = || ec_worker("grpc://127.0.0.1:9000", "ec_producer", &[]);
        let consumer = ec_worker("grpc://127.0.0.1:9001", "ec_consumer", &CONSUMER);
        let refused = |n, encoders, prefill, anchor| {
            plan_push(n, &push_selection(encoders, prefill), anchor)
                .err()
                .expect("refused")
                .status()
        };
        assert_eq!(
            refused(2, vec![producer()], Arc::clone(&consumer), None),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            refused(1, vec![producer(), producer()], Arc::clone(&consumer), None),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let no_transfer = ec_worker("grpc://127.0.0.1:9003", "ec_consumer", &[]);
        assert_eq!(
            refused(1, vec![producer()], no_transfer, None),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let fanned = plan_push(
            1,
            &push_selection(vec![producer(), producer()], consumer),
            Some(7),
        )
        .expect("a fan-out plan");
        assert_eq!(
            fanned.fanout.map(|fanout| fanout.assignments.len()),
            Some(2)
        );
    }

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

    /// An EPD plan whose encode server is on the direct ZMQ lane assembles
    /// its items inline (no /dev/shm), targets the server over its own
    /// connection with the rendezvous its engine declared, and the dispatch
    /// sends the encode frame on that connection with the rendezvous in it;
    /// a prefill side on the same lane takes the TokenSpeed items.
    #[tokio::test]
    async fn a_zmq_encode_server_takes_its_items_inline_at_its_declared_rendezvous() {
        use std::time::Duration;

        use engine_zmq_client::{
            mock_engine::{connect_to_frontend, default_ready_response},
            protocol::{
                handshake::EngineCoreReadyResponse, tokenspeed::request::TokenSpeedRequestType,
            },
            EngineId,
        };

        use crate::{
            routers::grpc::{
                context::EncodeWorkerAssignment,
                multimodal::single_image_intermediate,
                proto_wrapper::TokenSpeedTensorStorage,
                zmq_client::{EosTokenIds, ZmqEngineClient},
            },
            worker::{BasicWorkerBuilder, WorkerType},
        };

        let dir = tempfile::tempdir().unwrap();
        // A rank that declared `ready`, behind a direct-ZMQ worker of `role`,
        // the backend injected as the connect driver leaves it.
        let connected = |name: &'static str, role: WorkerType, ready: EngineCoreReadyResponse| {
            let dir = dir.path().join(name);
            async move {
                std::fs::create_dir_all(&dir).unwrap();
                let ep = |socket: &str| format!("ipc://{}", dir.join(socket).display());
                let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
                let (client, rank) = tokio::join!(
                    ZmqEngineClient::connect(
                        &handshake,
                        &input,
                        &output,
                        1,
                        "m".to_string(),
                        EosTokenIds::default(),
                        RuntimeType::TokenSpeed,
                        Duration::from_secs(10)
                    ),
                    connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
                );
                let client = BackendClient::Zmq(client.expect("adapter connect"));
                let worker = BasicWorkerBuilder::new(ep("worker.ipc"))
                    .connection_mode(ConnectionMode::Zmq)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .worker_type(role)
                    .build();
                worker
                    .backend_client
                    .load_full()
                    .set(Arc::new(client.clone()))
                    .ok()
                    .expect("cell empty");
                (
                    Arc::new(worker) as Arc<dyn Worker>,
                    client,
                    rank.expect("mock engine"),
                )
            }
        };
        let encoder_ready = EngineCoreReadyResponse {
            request_types: vec!["encode".to_string()],
            disaggregation_mode: Some("encode".to_string()),
            bootstrap_host: Some("10.0.0.9".to_string()),
            bootstrap_port: Some(8998),
            ..default_ready_response()
        };
        let (encoder, _, mut encoder_rank) =
            connected("encode", WorkerType::Encode, encoder_ready).await;
        let (prefill, prefill_client, _prefill_rank) =
            connected("prefill", WorkerType::Prefill, default_ready_response()).await;
        let (decode, decode_client, _decode_rank) =
            connected("decode", WorkerType::Decode, default_ready_response()).await;
        let workers = WorkerSelection::Disaggregated {
            encode_assignments: Some(vec![EncodeWorkerAssignment {
                item_index: 0,
                worker: Arc::clone(&encoder),
            }]),
            prefill,
            decode,
            runtime_type: RuntimeType::TokenSpeed,
        };
        let clients = ClientSelection::Disaggregated {
            prefill: prefill_client,
            decode: decode_client,
        };
        let intermediate =
            MultimodalIntermediate::try_new(vec![single_image_intermediate()]).unwrap();

        let plan = build_plan(&intermediate, Some(&clients), Some(&workers)).expect("the plan");
        let (bootstrap_info, dispatch) = plan.into_parts();
        assert_eq!(bootstrap_info.len(), 1);
        assert_eq!(
            (
                bootstrap_info[0].item_index,
                bootstrap_info[0].bootstrap_host.as_str(),
                bootstrap_info[0].bootstrap_port
            ),
            (0, "10.0.0.9", 8998)
        );
        let mut jobs = dispatch.into_jobs();
        assert_eq!(jobs.len(), 1);
        let job = jobs.pop().unwrap();
        assert_eq!(job.bootstrap_room, bootstrap_info[0].bootstrap_room);
        match &job.target {
            EncodeTarget::Zmq { worker, rendezvous } => {
                assert_eq!(worker.url(), encoder.url());
                assert_eq!(
                    (
                        rendezvous.bootstrap_host.as_str(),
                        rendezvous.bootstrap_port
                    ),
                    ("10.0.0.9", 8998)
                );
            }
            EncodeTarget::Grpc(endpoint) => panic!("a gRPC target {endpoint}"),
        }
        let PreparedEncodeItem::TokenSpeed {
            item: Some(item),
            shm_enabled,
            ..
        } = &job.item
        else {
            panic!("an item")
        };
        assert!(!*shm_enabled, "inline on the ZMQ lane");
        assert!(matches!(
            item.encoder_input.storage,
            TokenSpeedTensorStorage::Inline(_)
        ));

        let (sent, frames) = tokio::join!(job.dispatch(), encoder_rank.recv_request());
        sent.expect("the frame left on the server's connection");
        let frames = frames.unwrap();
        assert_eq!(
            TokenSpeedRequestType::from_frame(frames[0].as_ref()),
            Some(TokenSpeedRequestType::Encode)
        );
        assert!(
            frames[1].windows(8).any(|window| window == b"10.0.0.9"),
            "the rendezvous rides the frame"
        );
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
                encode_outputs: Some(EncodeOutputs::Rendezvous {
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
