//! Round-trip tests for the vLLM native-endpoint client against an in-process
//! mock server implementing the vendored `vllm.Inference` / `vllm.Control`
//! services.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use smg_grpc_client::vllm_native::{proto, VllmNativeClient, DATA_PARALLEL_RANK_METADATA_KEY};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

#[derive(Default)]
struct MockState {
    /// dp-rank metadata observed on the last GenerateStream call.
    seen_dp_rank: Mutex<Option<String>>,
    /// Request ids passed to Abort.
    aborted: Mutex<Vec<String>>,
    /// Whether any RPC arrived at all.
    touched: AtomicBool,
}

#[derive(Clone, Default)]
struct MockVllm {
    state: Arc<MockState>,
}

#[tonic::async_trait]
impl proto::control_server::Control for MockVllm {
    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::ServerInfo>, Status> {
        self.state.touched.store(true, Ordering::Relaxed);
        Ok(Response::new(proto::ServerInfo {
            engine_version: "0.0-test".to_string(),
            api_version: "1".to_string(),
            instance_id: "mock-1".to_string(),
            parallelism: Some(proto::ParallelismInfo {
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                data_parallel_size: 2,
                data_parallel_rank: 0,
                decode_context_parallel_size: 1,
            }),
            max_model_len: 4096,
            kv_block_size: 16,
            total_kv_blocks: 1000,
            max_running_requests: 8,
            max_batched_tokens: 8192,
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::ModelInfo>, Status> {
        Ok(Response::new(proto::ModelInfo {
            model_id: "mock/model".to_string(),
            served_model_name: "mock-model".to_string(),
            served_model_aliases: vec![],
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_multimodal: false,
            reasoning_parser: "basic".to_string(),
            tool_call_parser: "json".to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.state
            .aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(request.into_inner().request_ids);
        Ok(Response::new(proto::AbortResponse {}))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<proto::GetKvEventSourcesRequest>,
    ) -> Result<Response<proto::GetKvEventSourcesResponse>, Status> {
        Ok(Response::new(proto::GetKvEventSourcesResponse {
            sources: vec![proto::KvEventSource {
                transport: "zmq".to_string(),
                endpoint: "tcp://127.0.0.1:5557".to_string(),
                topic: String::new(),
                replay_endpoint: "tcp://127.0.0.1:5558".to_string(),
                data_parallel_rank: Some(0),
                encoding: "msgpack".to_string(),
                schema_version: 1,
                buffer_steps: 64,
                hwm: 1000,
                max_queue_size: 10_000,
            }],
        }))
    }
}

#[tonic::async_trait]
impl proto::inference_server::Inference for MockVllm {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<proto::GenerateResponse>, Status> {
        Err(Status::unimplemented("unary generate unused in tests"))
    }

    type GenerateStreamStream =
        futures::stream::Iter<std::vec::IntoIter<Result<proto::GenerateResponse, Status>>>;

    async fn generate_stream(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        *self
            .state
            .seen_dp_rank
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = request
            .metadata()
            .get(DATA_PARALLEL_RANK_METADATA_KEY)
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        let _ = request.into_inner();
        let chunk = |text: &str, tokens: u32| proto::GenerateResponse {
            prompt_info: None,
            outputs: Some(proto::SequenceOutput {
                index: 0,
                text: text.to_string(),
                num_tokens: tokens,
                ..Default::default()
            }),
        };
        let chunks = vec![Ok(chunk("hel", 1)), Ok(chunk("lo", 1))];
        Ok(Response::new(futures::stream::iter(chunks)))
    }
}

async fn spawn_mock() -> Result<(String, Arc<MockState>), Box<dyn std::error::Error>> {
    let mock = MockVllm::default();
    let state = Arc::clone(&mock.state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    #[expect(
        clippy::disallowed_methods,
        reason = "mock server for one test; lives until the test process exits"
    )]
    tokio::spawn(
        Server::builder()
            .add_service(proto::control_server::ControlServer::new(mock.clone()))
            .add_service(proto::inference_server::InferenceServer::new(mock))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    Ok((format!("http://{addr}"), state))
}

#[tokio::test]
async fn control_surface_round_trips() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmNativeClient::connect(&endpoint).await.unwrap();

    let server_info = client.get_server_info().await.unwrap();
    assert_eq!(server_info.kv_block_size, 16);
    assert_eq!(
        server_info.parallelism.as_ref().unwrap().data_parallel_size,
        2
    );
    assert!(state.touched.load(Ordering::Relaxed));

    let model_info = client.get_model_info().await.unwrap();
    assert_eq!(model_info.served_model_name, "mock-model");
    // The engine declares its own parser names — the native source for
    // per-model parser overrides.
    assert_eq!(model_info.tool_call_parser, "json");
    assert_eq!(model_info.reasoning_parser, "basic");

    let sources = client.get_kv_event_sources().await.unwrap();
    assert_eq!(sources.sources.len(), 1);
    assert_eq!(sources.sources[0].transport, "zmq");
    assert_eq!(sources.sources[0].data_parallel_rank, Some(0));
}

#[tokio::test]
async fn generate_stream_carries_dp_rank_metadata() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmNativeClient::connect(&endpoint).await.unwrap();

    let request = proto::GenerateRequest {
        request_id: "req-1".to_string(),
        model: "mock-model".to_string(),
        ..Default::default()
    };

    let mut stream = client
        .generate_stream(request.clone(), Some(1))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.message().await.unwrap() {
        text.push_str(&chunk.outputs.unwrap().text);
    }
    assert_eq!(text, "hello");
    assert_eq!(
        state
            .seen_dp_rank
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref(),
        Some("1")
    );

    // Without a rank, no metadata header is sent.
    let mut stream = client.generate_stream(request, None).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
    assert_eq!(
        state
            .seen_dp_rank
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref(),
        None
    );
}

#[tokio::test]
async fn abort_reaches_the_control_service() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmNativeClient::connect(&endpoint).await.unwrap();

    client.abort(vec!["req-9".to_string()]).await.unwrap();
    assert_eq!(
        state
            .aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        ["req-9"]
    );
}
