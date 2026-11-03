//! The `Generate` of a long prompt reaches the mock worker. A gateway sends a
//! prompt's token ids as one message; a million of them are a few megabytes,
//! which tonic's default message limit (4 MiB) refuses.

use futures::StreamExt;
use mock_worker::config::Config;
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use ts::generate_response::Response as GenResp;

mod common;
use common::start;

/// A million-token prompt as a gateway sends it: ids of a six-figure
/// vocabulary (three bytes each as varints) with the prompt text alongside,
/// about 7 MB on the wire.
fn million_token_request() -> ts::GenerateRequest {
    let input_ids: Vec<u32> = (0..1_048_576u32).map(|i| 100_000 + (i % 50_000)).collect();
    ts::GenerateRequest {
        request_id: "million".to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids,
            original_text: "a".repeat(4 << 20),
        }),
        sampling_params: Some(ts::SamplingParams {
            max_new_tokens: Some(2),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    }
}

/// By default the worker takes a message of any size, as the engine
/// servicers do, and answers the stream through to its `Complete`.
#[tokio::test]
async fn a_million_token_generate_round_trips_by_default() {
    let client = start(Config::default()).await;

    let mut stream = client
        .generate(million_token_request())
        .await
        .expect("the worker accepts a million-token Generate");
    let mut last = None;
    while let Some(frame) = stream.next().await {
        last = Some(frame.expect("a response frame"));
    }
    stream.mark_completed();

    assert!(
        matches!(
            last.as_ref().and_then(|frame| frame.response.as_ref()),
            Some(GenResp::Complete(_))
        ),
        "the stream ends with a Complete: {last:?}"
    );
}

/// `--grpc-max-message-bytes` sets the limit; a message above it is refused
/// as tonic refuses it, naming the size it found.
#[tokio::test]
async fn a_configured_limit_refuses_a_larger_message() {
    let client = start(Config {
        grpc_max_message_bytes: 4 << 20,
        ..Config::default()
    })
    .await;

    let err = match client.generate(million_token_request()).await {
        Ok(_) => panic!("a message above the configured limit is refused"),
        Err(err) => err,
    };

    assert!(
        err.message().contains("message length too large"),
        "the refusal names the size: {err:?}"
    );
}
