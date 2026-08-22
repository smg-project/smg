// Live wire-validation probe: play the SMG frontend against a real headless
// vLLM EngineCore over ZMQ. Binds the handshake/input/output sockets, drives the
// handshake, submits one tokenized generate request, and prints the streamed
// token ids.
//
// Usage:
//   live_probe <handshake_addr> <input_ipc> <output_ipc> [tok1 tok2 ...]
//
// Then launch the engine so it dials <handshake_addr>:
//   vllm serve <model> --headless --data-parallel-size 1 \
//     --data-parallel-size-local 1 \
//     --data-parallel-address 127.0.0.1 --data-parallel-rpc-port <port> \
//     --enforce-eager

// This is a dev-only CLI probe, not library code: printing to stderr, `expect`
// on argument/handshake errors, and `process::exit` on failure are all
// appropriate here and intentionally exempt from the workspace restriction lints.
#![expect(
    clippy::print_stderr,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "dev-only live-validation CLI example"
)]

use std::time::Duration;

use engine_zmq_client::{
    connect_handshake,
    protocol::vllm::{request::EngineCoreRequest, sampling::EngineCoreSamplingParams},
    EngineCoreClient,
};
use futures::StreamExt;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let mut args = std::env::args().skip(1);
    let handshake = args
        .next()
        .expect("arg1: handshake address (tcp://host:port)");
    let input = args.next().expect("arg2: input ipc:// address");
    let output = args.next().expect("arg3: output ipc:// address");
    let tokens: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();
    let prompt_token_ids = if tokens.is_empty() {
        vec![1, 2, 3, 4]
    } else {
        tokens
    };

    // Model load + KV profiling happen between INIT and READY, so allow a long
    // handshake window.
    let timeout = Duration::from_secs(600);

    eprintln!("[probe] binding sockets; waiting for engine to connect on {handshake}");
    let transport = connect_handshake(&handshake, 1, &input, &output, timeout)
        .await
        .expect("handshake with engine");

    let engine = &transport.engines[0];
    eprintln!(
        "[probe] engine registered: vllm_version={} max_model_len={} num_gpu_blocks={} block_size={} dtype={} engine_index={:?}",
        engine.ready_response.vllm_version,
        engine.ready_response.max_model_len,
        engine.ready_response.num_gpu_blocks,
        engine.ready_response.block_size,
        engine.ready_response.dtype.as_str(),
        engine.engine_id.engine_index(),
    );

    let client = EngineCoreClient::new(transport);

    let request = EngineCoreRequest {
        request_id: "probe-1".to_string(),
        prompt_token_ids: Some(prompt_token_ids.clone()),
        sampling_params: Some(EngineCoreSamplingParams {
            temperature: 0.0,
            max_tokens: 32,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..EngineCoreRequest::default()
    };
    eprintln!("[probe] submitting request with prompt_token_ids={prompt_token_ids:?}");

    let mut stream = client.submit(request).await.expect("submit");
    let mut all_tokens = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(output) => {
                all_tokens.extend(output.new_token_ids.iter().copied());
                eprintln!(
                    "[probe] +{:?} finish={:?}",
                    output.new_token_ids, output.finish_reason
                );
                if output.finished() {
                    eprintln!(
                        "[probe] DONE finish={:?} total_tokens={} ids={:?}",
                        output.finish_reason,
                        all_tokens.len(),
                        all_tokens
                    );
                    break;
                }
            }
            Err(error) => {
                eprintln!("[probe] ERROR: {error}");
                std::process::exit(1);
            }
        }
    }
    eprintln!("[probe] success: streamed {} tokens", all_tokens.len());
}
