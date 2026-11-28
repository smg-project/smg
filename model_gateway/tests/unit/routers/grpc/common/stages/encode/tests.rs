use super::*;
use crate::routers::grpc::{backend_client::BasicWorkerBuilder, context::EncodeWorkerAssignment};

fn ec_worker(url: &str, role: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
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
        decode: Arc::clone(&prefill).into(),
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
    use smg_worker::identity::basic::WorkerType;

    use crate::routers::grpc::{
        backend_client::BasicWorkerBuilder,
        context::EncodeWorkerAssignment,
        multimodal::single_image_intermediate,
        proto_wrapper::TokenSpeedTensorStorage,
        zmq_client::{EosTokenIds, ZmqEngineClient},
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
                .engine_connection
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
        decode: decode.into(),
        runtime_type: RuntimeType::TokenSpeed,
    };
    let clients = ClientSelection::Disaggregated {
        prefill: prefill_client,
        decode: decode_client,
    };
    let intermediate = MultimodalIntermediate::try_new(vec![single_image_intermediate()]).unwrap();

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
        EncodeTarget::Zmq { rendezvous } => {
            assert_eq!(job.worker.url(), encoder.url());
            assert_eq!(
                (
                    rendezvous.bootstrap_host.as_str(),
                    rendezvous.bootstrap_port
                ),
                ("10.0.0.9", 8998)
            );
        }
        EncodeTarget::Grpc => panic!("a gRPC target"),
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

/// A rendezvous plan's encode call ends with its request: once the
/// request has let go of the dispatch's watch, a call the encoder still
/// holds is cancelled there.
#[tokio::test]
async fn an_encode_call_ends_with_its_request() {
    use std::{collections::HashMap, time::Duration};

    use tokenspeed_encoder::token_speed_encoder_server::{
        TokenSpeedEncoder, TokenSpeedEncoderServer,
    };
    use tokio::sync::{mpsc, oneshot};
    use tonic::transport::Server;

    use crate::routers::grpc::{
        common::stages::request_execution::launch_encode_dispatch,
        proto_wrapper::{TokenSpeedModality, TokenSpeedTensor},
        test_engine::serve_on_loopback,
    };

    /// An encoder that holds every call, and hands out what closes when
    /// the held call is dropped.
    struct Encoder {
        calls: mpsc::UnboundedSender<oneshot::Receiver<()>>,
    }

    #[tonic::async_trait]
    impl TokenSpeedEncoder for Encoder {
        async fn encode(
            &self,
            _request: tonic::Request<tokenspeed_encoder::EncodeRequest>,
        ) -> std::result::Result<tonic::Response<tokenspeed_encoder::EncodeResponse>, tonic::Status>
        {
            let (_held, dropped) = oneshot::channel();
            let _ = self.calls.send(dropped);
            std::future::pending().await
        }
    }

    let (calls, mut held) = mpsc::unbounded_channel();
    let service = TokenSpeedEncoderServer::new(Encoder { calls });
    let (address, _server) = serve_on_loopback(Server::builder().add_service(service)).await;
    let item = TokenSpeedMultimodalItem {
        modality: TokenSpeedModality::Image,
        encoder_input: TokenSpeedTensor::inline(vec![0u8; 4], vec![1], "float32".to_string()),
        model_specific_tensors: HashMap::new(),
        placeholder_token_id: None,
        mm_placeholders: Vec::new(),
        content_hash: Vec::new(),
    };
    let worker: Arc<dyn Worker> =
        Arc::new(BasicWorkerBuilder::new(format!("http://{address}")).build());
    let watch = launch_encode_dispatch(EncodeDispatchPlan::new(vec![PreparedEncodeJob {
        item: PreparedEncodeItem::tokenspeed(item, false, 0),
        target: EncodeTarget::Grpc,
        worker: Arc::clone(&worker),
        bootstrap_room: 42,
    }]));
    let mut dropped = tokio::time::timeout(Duration::from_secs(5), held.recv())
        .await
        .expect("the encode send never reached the encoder")
        .expect("the encoder holds the call");
    assert_eq!(worker.load(), 1, "a call in flight counts in the load");

    // While the request holds the watch, its call stays open.
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut dropped)
            .await
            .is_err(),
        "the call was dropped while its request still held the watch"
    );

    // The request is over.
    drop(watch);
    tokio::time::timeout(Duration::from_secs(5), dropped)
        .await
        .expect("the encoder still holds the call of a request that is over")
        .expect_err("a held call is dropped, never released");
    assert_eq!(worker.load(), 0, "the load goes with the call");
}

/// An encode send that fails names its worker with the address masked:
/// its text is the line the dispatch's supervisor writes at error level.
/// Every worker here is registered with a password and a token in its
/// query. The encoder call meets a refused connection, a failed call and
/// a declined request; the frame's sender meets a worker out of reach,
/// one without a backend connection and one off the ZMQ lane.
#[tokio::test]
async fn a_failed_encode_send_names_its_worker_masked() {
    use tokenspeed_encoder::token_speed_encoder_server::{
        TokenSpeedEncoder, TokenSpeedEncoderServer,
    };
    use tonic::transport::Server;

    use crate::routers::grpc::test_engine::serve_on_loopback;

    /// An encoder that fails every call, or declines it.
    struct Encoder {
        fails: bool,
    }

    #[tonic::async_trait]
    impl TokenSpeedEncoder for Encoder {
        async fn encode(
            &self,
            _request: tonic::Request<tokenspeed_encoder::EncodeRequest>,
        ) -> std::result::Result<tonic::Response<tokenspeed_encoder::EncodeResponse>, tonic::Status>
        {
            if self.fails {
                return Err(tonic::Status::internal("the encoder failed"));
            }
            Ok(tonic::Response::new(tokenspeed_encoder::EncodeResponse {
                accepted: false,
            }))
        }
    }

    let serving = |fails| async move {
        let service = TokenSpeedEncoderServer::new(Encoder { fails });
        let (address, _server) = serve_on_loopback(Server::builder().add_service(service)).await;
        address.port()
    };
    let (failing, declining) = (serving(true).await, serving(false).await);
    // Nothing listens on port 1: the connection is refused.
    let refusing = 1;
    let with_secrets = |port: u16| format!("http://user:s3cret@127.0.0.1:{port}/?token=s3cret");
    let worker = |port, mode| {
        BasicWorkerBuilder::new(with_secrets(port))
            .connection_mode(mode)
            .runtime_type(RuntimeType::TokenSpeed)
            .build()
    };
    let request = tokenspeed_encoder::EncodeRequest::default;
    let rendezvous = EncodeRendezvous {
        bootstrap_host: "10.0.0.9".to_string(),
        bootstrap_port: 8998,
    };

    let mut failures = Vec::new();
    for (what, port) in [
        ("a refused connection", refusing),
        ("a failed call", failing),
        ("a declined request", declining),
    ] {
        let sent = send_tokenspeed_encode_rpc(with_secrets(port), request()).await;
        failures.push((what, port, sent.expect_err(what)));
    }
    for (what, port, mode) in [
        ("a worker out of reach", refusing, ConnectionMode::Grpc),
        (
            "a worker without a backend connection",
            refusing,
            ConnectionMode::Http,
        ),
        ("a worker off the ZMQ lane", declining, ConnectionMode::Grpc),
    ] {
        let sent = send_tokenspeed_encode_frame(&worker(port, mode), &rendezvous, request()).await;
        failures.push((what, port, sent.expect_err(what)));
    }
    let unmasked: Vec<_> = failures
        .iter()
        .filter(|(_, port, text)| {
            text.contains("s3cret") || !text.contains(&format!("http://***@127.0.0.1:{port}"))
        })
        .collect();
    assert!(
        unmasked.is_empty(),
        "encode failures that do not name their worker masked: {unmasked:#?}"
    );
}

/// Encode sends that reach a new endpoint together dial its pool of four
/// connections between them, not a pool each.
#[tokio::test]
async fn encode_sends_that_meet_a_new_endpoint_together_share_its_pool() {
    use std::time::Duration;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback listener");
    let endpoint = format!("http://{}", listener.local_addr().expect("its address"));
    // The peer answers nothing, so each send ends at its bound.
    let sends = futures::future::join_all((0..32).map(|_| {
        tokio::time::timeout(
            Duration::from_millis(200),
            send_tokenspeed_encode_rpc(
                endpoint.clone(),
                tokenspeed_encoder::EncodeRequest::default(),
            ),
        )
    }));
    // Every dialled connection is taken and held: one the peer closed
    // would be dialled again.
    let dialled = async {
        let mut held = Vec::new();
        while let Ok(Ok((socket, _))) =
            tokio::time::timeout(Duration::from_millis(400), listener.accept()).await
        {
            held.push(socket);
        }
        held.len()
    };
    let (_, dialled) = tokio::join!(sends, dialled);
    assert_eq!(dialled, 4);
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
            mm_shm_dev_writable, write_tokenspeed_shm_with, TokenSpeedModality, TokenSpeedTensor,
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
            target: EncodeTarget::Grpc,
            worker: Arc::new(BasicWorkerBuilder::new("http://encode-worker:9000").build()),
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
        let worker = BasicWorkerBuilder::new("http://127.0.0.1:1").build();
        let _ = item.dispatch(&worker, EncodeTarget::Grpc, 7).await;

        assert!(
            !path.exists(),
            "dispatch must reclaim the encode SHM segment (no leak, no double-unlink)"
        );
    }
}
