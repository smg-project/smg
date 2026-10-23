//! Drive cancellation through the gateway's gRPC client, including its
//! abort-on-drop path, and check the simulated engine's own load.

use std::{sync::Arc, time::Duration};

use futures::StreamExt;
use mock_worker::{
    config::Config,
    engine::{EngineParams, TimingModel},
    grpc::serve_with_listener,
};
use smg_grpc_client::{
    tokenspeed_proto as ts, tokenspeed_scheduler::AbortOnDropStream, TokenSpeedSchedulerClient,
};
use tokio::{net::TcpListener, time::timeout};
use ts::generate_response::Response as GenResp;

async fn start(max_running: usize, overhead_ms: f64) -> TokenSpeedSchedulerClient {
    let cfg = Config {
        realistic: true,
        engine: EngineParams {
            max_running,
            request_overhead_ms: overhead_ms,
            timing: TimingModel::Linear {
                prefill_tps: 8000.0,
                decode_base_ms: 10.0,
                decode_per_req_ms: 0.0,
            },
            ..EngineParams::default()
        },
        ..Config::default()
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    TokenSpeedSchedulerClient::connect(&format!("http://{address}"))
        .await
        .unwrap()
}

fn request(id: &str, max_new: u32, stream: bool) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: vec![1; 64],
            ..Default::default()
        }),
        sampling_params: Some(ts::SamplingParams {
            max_new_tokens: Some(max_new),
            ..Default::default()
        }),
        stream,
        ..Default::default()
    }
}

async fn complete(stream: &mut AbortOnDropStream) -> ts::GenerateComplete {
    timeout(Duration::from_secs(2), async {
        while let Some(frame) = stream.next().await {
            if let Some(GenResp::Complete(done)) = frame.unwrap().response {
                assert!(stream.next().await.is_none(), "nothing follows completion");
                stream.mark_completed();
                return done;
            }
        }
        panic!("stream ended without a completion");
    })
    .await
    .expect("the worker ends the stream promptly")
}

async fn load(client: &TokenSpeedSchedulerClient, running: i32, waiting: i32) -> ts::SchedulerLoad {
    timeout(Duration::from_secs(2), async {
        loop {
            let load = client.get_loads(vec![]).await.unwrap().loads.remove(0);
            if (load.num_running_reqs, load.num_waiting_reqs) == (running, waiting) {
                return load;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the engine releases cancelled work")
}

#[tokio::test]
async fn abort_ends_generation_without_affecting_another_request() {
    // Delayed delivery must keep tokens before the abort completion.
    let client = start(2, 50.0).await;
    let mut cancelled = client
        .generate(request("cancelled", 4096, true))
        .await
        .unwrap();
    let mut other = client.generate(request("other", 16, true)).await.unwrap();
    let first = cancelled.next().await.unwrap().unwrap();
    assert!(matches!(first.response, Some(GenResp::Chunk(_))));
    client
        .abort_request("cancelled".into(), String::new())
        .await
        .unwrap();

    let done = complete(&mut cancelled).await;
    assert_eq!(done.finish_reason, "abort");
    assert!(done.completion_tokens < 4096);
    assert_eq!(done.output_ids.len(), done.completion_tokens as usize);
    assert_eq!(done.prompt_tokens, 64);

    // Repeated and unknown IDs are harmless; a completed request stays done.
    for id in ["cancelled", "missing"] {
        client
            .abort_request(id.into(), String::new())
            .await
            .unwrap();
    }
    let done = complete(&mut other).await;
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.completion_tokens, 16);
    client
        .abort_request("other".into(), String::new())
        .await
        .unwrap();
    assert_eq!(load(&client, 0, 0).await.num_used_tokens, 0);
}

#[tokio::test]
async fn abort_removes_a_queued_non_streaming_request() {
    let client = start(1, 0.0).await;
    let mut running = client
        .generate(request("running", 4096, true))
        .await
        .unwrap();
    running.next().await.unwrap().unwrap();
    let mut queued = client
        .generate(request("queued", 4096, false))
        .await
        .unwrap();
    load(&client, 1, 1).await;

    client
        .abort_request("queued".into(), String::new())
        .await
        .unwrap();
    let done = complete(&mut queued).await;
    assert_eq!(done.finish_reason, "abort");
    assert_eq!(done.completion_tokens, 0);
    assert!(done.output_ids.is_empty());
    assert_eq!(load(&client, 1, 0).await.num_waiting_uncached_tokens, 0);

    drop(running);
    assert_eq!(load(&client, 0, 0).await.num_used_tokens, 0);
}

#[tokio::test]
async fn dropping_the_client_stream_releases_engine_work() {
    let client = start(1, 0.0).await;
    let mut stream = client
        .generate(request("dropped", 4096, true))
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    load(&client, 1, 0).await;
    drop(stream);
    assert_eq!(load(&client, 0, 0).await.num_used_tokens, 0);
}
