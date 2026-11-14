//! Real tonic transport tests of the input gate; no claim about a GPU or KV transport.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use engine_servicer::{
    managed::{
        decode::NativeInput,
        ledger::Ledger,
        service::{ManagedService, NativeRuntime},
    },
    BoxFuture,
};
use smg_grpc_client::{
    sglang_managed::{
        budget::Deadline,
        proto::{
            self, control_event::Event, decode_command::Command,
            managed_sglang_client::ManagedSglangClient, managed_sglang_server::ManagedSglangServer,
        },
        Limits, LAYOUT,
    },
    sglang_proto as sg,
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Status};

type BoxStream<T> = std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send>>;

#[derive(Default)]
struct Runtime {
    submitted: AtomicUsize,
    terminated: AtomicUsize,
}

impl NativeRuntime for Runtime {
    fn prefill(
        &self,
        _: proto::PrefillStart,
        _: Deadline,
    ) -> BoxFuture<Result<BoxStream<proto::ControlEvent>, Status>> {
        Box::pin(async { Ok(Box::pin(futures::stream::pending()) as BoxStream<_>) })
    }
    fn decode(
        &self,
        _: proto::OperationRef,
        _: proto::InputBinding,
        _: Deadline,
        _: NativeInput,
    ) -> BoxFuture<Result<BoxStream<sg::GenerateResponse>, Status>> {
        self.submitted.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(Box::pin(futures::stream::pending()) as BoxStream<_>) })
    }
    fn terminate(&self, _: proto::OperationRef) -> BoxFuture<Result<bool, Status>> {
        self.terminated.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(true) })
    }
}

#[expect(clippy::expect_used, reason = "test host clock is after Unix epoch")]
fn start(deferred: bool) -> proto::DecodeStart {
    let expiry = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("valid test fixture")
        .as_millis() as u64
        + 10000;
    proto::DecodeStart {
        version: 1,
        binding: Some(proto::InputBinding {
            decode: Some(proto::OperationRef {
                identity: Some(proto::Identity {
                    logical_request_id: "r".into(),
                    attempt_id: "a".into(),
                    branch_id: 0,
                }),
                target: Some(proto::Instance {
                    worker_id: "d".into(),
                    generation: "g".into(),
                }),
                operation_id: "op".into(),
                request_digest: vec![1; 32],
                plan_digest: vec![2; 32],
            }),
            prefill: Some(proto::Instance {
                worker_id: "p".into(),
                generation: "pg".into(),
            }),
            prefill_rid: "a-p".into(),
            bootstrap: Some(proto::Bootstrap {
                host: "p".into(),
                port: 8998,
                room: 9,
            }),
            layout_contract: LAYOUT.into(),
            processor_contract: "processor".into(),
            model_revision: "weights".into(),
            media_routing_labels: vec![],
        }),
        budget: Some(proto::Budget {
            original_expiry_unix_ms: expiry,
            remaining_timeout_ms: 10000,
            estimated_work: 1,
            context_bytes: 5 * 1024 * 1024,
        }),
        generation: Some(sg::GenerateRequest {
            request_id: "op".into(),
            tokenized: Some(sg::TokenizedInput {
                input_ids: if deferred { vec![] } else { vec![1, 2] },
                ..Default::default()
            }),
            ..Default::default()
        }),
        deferred_from_prefill: deferred,
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "test server is explicitly stopped and joined"
)]
#[expect(
    clippy::expect_used,
    reason = "fail the test if its local server fixture cannot start"
)]
async fn serve() -> (
    ManagedSglangClient<tonic::transport::Channel>,
    Arc<Runtime>,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let runtime = Arc::new(Runtime::default());
    let instance = start(true)
        .binding
        .expect("valid test fixture")
        .decode
        .expect("valid test fixture")
        .target
        .expect("valid test fixture");
    let ledger = Ledger::new(instance.clone(), 16, 8, 100, 64 * 1024 * 1024, 1000, 60000)
        .expect("valid test fixture");
    let service = ManagedService::new(
        Arc::clone(&runtime),
        instance,
        ledger,
        Limits::default(),
        60000,
    );
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("valid test fixture");
    let address = listener.local_addr().expect("valid test fixture");
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(async move {
        Server::builder()
            .add_service(ManagedSglangServer::new(service))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = stopped.await;
            })
            .await
            .expect("valid test fixture");
    });
    let client = ManagedSglangClient::connect(format!("http://{address}"))
        .await
        .expect("valid test fixture");
    (client, runtime, stop, task)
}

#[tokio::test]
async fn accepted_is_available_before_input_and_cancel_does_not_submit_native_work() {
    let (mut client, runtime, stop, task) = serve().await;
    let (sender, receiver) = mpsc::channel(2);
    sender
        .send(proto::DecodeCommand {
            command: Some(Command::Start(start(true))),
        })
        .await
        .expect("valid test fixture");
    let mut events = tokio::time::timeout(
        Duration::from_secs(1),
        client.generate_decode(ReceiverStream::new(receiver)),
    )
    .await
    .expect("valid test fixture")
    .expect("valid test fixture")
    .into_inner();
    let accepted = events
        .message()
        .await
        .expect("valid test fixture")
        .expect("valid test fixture");
    assert_eq!(
        accepted.event,
        Some(Event::Stage(proto::Stage::Accepted as i32))
    );
    assert_eq!(runtime.submitted.load(Ordering::SeqCst), 0);
    sender
        .send(proto::DecodeCommand {
            command: Some(Command::Cancel("client cancelled".into())),
        })
        .await
        .expect("valid test fixture");
    assert_eq!(
        events.message().await.unwrap_err().code(),
        tonic::Code::Cancelled
    );
    assert_eq!(runtime.submitted.load(Ordering::SeqCst), 0);
    assert_eq!(runtime.terminated.load(Ordering::SeqCst), 0);
    drop(events);
    drop(sender);
    drop(client);
    stop.send(()).expect("valid test fixture");
    task.await.expect("valid test fixture");
}

#[tokio::test]
async fn inline_text_submits_once_and_normal_half_close_keeps_the_response_alive() {
    let (mut client, runtime, stop, task) = serve().await;
    let request = start(false);
    let termination = proto::Termination {
        operation: request
            .binding
            .as_ref()
            .expect("valid test fixture")
            .decode
            .clone(),
        original_expiry_unix_ms: request
            .budget
            .as_ref()
            .expect("valid test fixture")
            .original_expiry_unix_ms,
        native_rid: "op".into(),
        reason: "test".into(),
    };
    let mut events = client
        .generate_decode(tokio_stream::iter([proto::DecodeCommand {
            command: Some(Command::Start(request)),
        }]))
        .await
        .expect("valid test fixture")
        .into_inner();
    for stage in [
        proto::Stage::Accepted,
        proto::Stage::InputBound,
        proto::Stage::NativeSubmitted,
    ] {
        assert_eq!(
            events
                .message()
                .await
                .expect("valid test fixture")
                .expect("valid test fixture")
                .event,
            Some(Event::Stage(stage as i32))
        );
    }
    assert_eq!(runtime.submitted.load(Ordering::SeqCst), 1);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), events.message())
            .await
            .is_err()
    );
    assert!(
        client
            .terminate(termination)
            .await
            .expect("valid test fixture")
            .into_inner()
            .recorded
    );
    assert_eq!(
        events.message().await.unwrap_err().code(),
        tonic::Code::Cancelled
    );
    drop(events);
    drop(client);
    stop.send(()).expect("valid test fixture");
    task.await.expect("valid test fixture");
}

/// Actual ZMQ transport with a scripted scheduler: proves ownership, not GPU resource release.
#[tokio::test]
async fn native_abort_receipt_waits_for_scheduler_terminal_output() {
    use std::time::Instant;

    use bytes::Bytes;
    use engine_servicer::managed::native::ManagedNativeRuntime;
    use engine_zmq_adapter::{EosTokenIds, ZmqEngineClient};
    use engine_zmq_client::{
        codec::encode_msgpack,
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::{handshake::ManagedSglangHandshake, sglang::output::BatchTokenIDSlimOutput},
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    let directory = tempfile::tempdir().unwrap();
    let endpoint = |name| format!("ipc://{}/{name}", directory.path().display());
    let (handshake, input, output) = (endpoint("hs"), endpoint("in"), endpoint("out"));
    let mut ready = default_ready_response();
    ready.sglang_managed = Some(ManagedSglangHandshake {
        version: 1,
        worker_id: "d".into(),
        generation: "g".into(),
        role: "decode".into(),
        layout: LAYOUT.into(),
    });
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".into(),
            EosTokenIds::default(),
            RuntimeType::Sglang,
            Duration::from_secs(5)
        ),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
    );
    let client = client.unwrap();
    let (mut input, mut output) = engine.unwrap().split();
    let runtime =
        ManagedNativeRuntime::new(Arc::new(move || Some(client.clone())), 8, Limits::default())
            .unwrap();
    let request = start(false);
    let binding = request.binding.unwrap();
    let operation = binding.decode.clone().unwrap();
    let budget = request.budget.unwrap();
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let deadline = Deadline::admit(&budget, wall, Instant::now(), 0, 0, 60000).unwrap();
    let responses = runtime
        .decode(
            operation.clone(),
            binding,
            deadline,
            NativeInput::Text(request.generation.unwrap()),
        )
        .await
        .unwrap();
    let generate = tokio::time::timeout(Duration::from_secs(2), input.recv_frames())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(generate[0].as_ref(), b"\x00");
    let mut termination = runtime.terminate(operation);
    let abort = tokio::time::timeout(Duration::from_secs(2), input.recv_frames())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(abort[0].as_ref(), b"\x01");
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut termination)
            .await
            .is_err()
    );
    // A delayed scheduler receipt must remain observable past the former 30s cutoff.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(31)).await;
    tokio::task::yield_now().await;
    tokio::time::resume();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut termination)
            .await
            .is_err()
    );
    let acknowledged = BatchTokenIDSlimOutput {
        rids: vec!["op".into()],
        output_ids: vec![vec![]],
        finished_reasons: vec!["abort".into()],
        finished_messages: vec![None],
        finished_matched: vec![None],
        prompt_tokens: vec![2],
        completion_tokens: vec![0],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        ..Default::default()
    };
    output
        .send_frames(vec![Bytes::from(encode_msgpack(&acknowledged).unwrap())])
        .await
        .unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(2), termination)
        .await
        .unwrap()
        .unwrap());
    drop(responses);
}
