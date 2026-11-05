//! What the generate path allocates per token on a streamed request without
//! fan-out, over the crate's mock engine, pinned at today's count so the next
//! change that adds a per-token copy fails a test rather than moving a bench
//! inside its noise. A separate executable, so it can declare the counting
//! allocator the pin needs; the bed is the one `benches/generate_path.rs`
//! times.
#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "a test that stops on a broken setup and prints the count it pins"
)]

mod common;

use std::{
    collections::BTreeSet,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use common::counting_alloc::{self, Counting};
use engine_servicer::{ServicerConfig, VllmModelInfo, VllmServicerConfig, VllmServicerServer};
use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound, MockEngine},
    protocol::vllm::output::{
        EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    },
    EngineId,
};
use futures::StreamExt;
use smg_grpc_client::{vllm_proto as vllm, vllm_proto::vllm_engine_client::VllmEngineClient};
use tonic::transport::Channel;

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The tokens of the long request; the short one has one, so the difference
/// is the per-token cost alone.
const LONG: u32 = 256;

/// The mock engine's side of the wire: every add is answered with as many
/// single-token chunks as its `max_tokens` asks for and a `length` finish.
async fn answer_adds(engine: MockEngine) {
    let (mut input, mut output) = engine.split();
    while let Ok(inbound) = input.recv().await {
        let EngineInbound::Add(request) = inbound else {
            continue;
        };
        let id = request.request_id.clone();
        let tokens = request
            .sampling_params
            .as_ref()
            .map_or(1, |params| params.max_tokens)
            .max(1);
        for token in 0..tokens {
            let finish = (token + 1 == tokens).then_some(EngineCoreFinishReason::Length);
            let batch = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                engine_index: 0,
                outputs: vec![EngineCoreOutput {
                    request_id: id.clone(),
                    new_token_ids: vec![token + 10],
                    finish_reason: finish,
                    ..Default::default()
                }],
                finished_requests: finish.map(|_| BTreeSet::from([id.clone()])),
                ..Default::default()
            });
            output
                .send_outputs(&batch)
                .await
                .expect("the mock engine's output is sent");
        }
    }
}

struct Bed {
    server: VllmServicerServer,
    client: VllmEngineClient<Channel>,
    sent: u64,
    _dir: tempfile::TempDir,
}

impl Bed {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let handshake = format!("ipc://{}", dir.path().join("handshake").display());
        let servicer = ServicerConfig {
            engine_startup_timeout: Duration::from_secs(30),
            engine_startup_ceiling: None,
            ..ServicerConfig::new(
                "127.0.0.1:0",
                format!("ipc://{}", dir.path().join("engine").display()),
                &handshake,
            )
        };
        let model = VllmModelInfo {
            model_path: "org/model".to_string(),
            served_model_name: "served".to_string(),
            tokenizer_path: "org/model".to_string(),
            is_generation: true,
            max_context_length: 4096,
            vocab_size: 1024,
            data_parallel_size: 1,
            ..Default::default()
        };
        let server = VllmServicerServer::start(VllmServicerConfig {
            servicer,
            model,
            media_processor: None,
        })
        .expect("the servicer starts");
        let engine = connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response(),
        )
        .await
        .expect("the mock engine handshakes");
        while !server.engine_ready() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "the mock engine answers until the test's runtime drops"
        )]
        let _mock = tokio::spawn(answer_adds(engine));
        let channel = Channel::from_shared(format!("http://{}", server.address()))
            .unwrap()
            .connect()
            .await
            .expect("the gRPC channel connects");
        Self {
            server,
            client: VllmEngineClient::new(channel),
            sent: 0,
            _dir: dir,
        }
    }

    /// One streamed request of `tokens` tokens, read to its end.
    async fn request(&mut self, tokens: u32) {
        self.sent += 1;
        let request = vllm::GenerateRequest {
            request_id: format!("r{}", self.sent),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(tokens),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = self
            .client
            .generate(request)
            .await
            .expect("the call is accepted")
            .into_inner();
        let mut received = 0;
        while let Some(item) = stream.next().await {
            if let Some(vllm::generate_response::Response::Chunk(chunk)) =
                item.expect("a response").response
            {
                received += chunk.token_ids.len() as u32;
            }
        }
        assert_eq!(received, tokens);
    }
}

/// The pins of this binary run one at a time: the counter is process-wide,
/// so a bed starting beside a measured window would be counted with it.
static ONE_PIN_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The allocation calls one request of `tokens` tokens makes on every
/// thread: the servicer serves on its own runtime thread, the client and the
/// mock engine on the test's; nothing else runs.
fn counted(runtime: &tokio::runtime::Runtime, bed: &mut Bed, tokens: u32) -> usize {
    let before = counting_alloc::total_calls();
    runtime.block_on(bed.request(tokens));
    counting_alloc::total_calls() - before
}

#[test]
fn a_streamed_request_without_fan_out_allocates_this_much_per_token() {
    let _one_at_a_time = ONE_PIN_AT_A_TIME
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut bed = runtime.block_on(Bed::start());
    // The connection's and the first requests' one-time costs are paid first.
    for _ in 0..3 {
        runtime.block_on(bed.request(1));
    }
    let short = counted(&runtime, &mut bed, 1);
    let long = counted(&runtime, &mut bed, LONG);
    let short_again = counted(&runtime, &mut bed, 1);
    let long_again = counted(&runtime, &mut bed, LONG);
    let per_token = (long.min(long_again) - short.max(short_again)) as f64 / f64::from(LONG - 1);
    eprintln!(
        "allocation calls per request: {short} and {short_again} for 1 token, {long} and \
         {long_again} for {LONG} tokens: {per_token:.2} per token"
    );
    bed.server.stop(Duration::from_secs(5)).unwrap();
    assert!(
        per_token <= 20.5,
        "allocation calls per token: {per_token:.2} (20.07 with the one choice polled in place, \
         21.07 through the merged stream; counted on every thread: the servicer, the tonic/h2 \
         client and server and the mock engine's own per-token work, so a dependency bump that \
         moves it re-pins here)"
    );
}
