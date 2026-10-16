use super::*;
use crate::routers::grpc::regular::reasoning_usage::tests::CountingTokenizer;

#[derive(Clone, Copy, Debug)]
enum CompleteIds {
    Full,
    LastDelta,
    Empty,
}

fn frames(
    tokenizer: &dyn Tokenizer,
    text: &str,
    size: usize,
    finish: &str,
    complete_ids: CompleteIds,
) -> Vec<proto::GenerateResponse> {
    let ids = tokenizer.encode(text, false).unwrap().token_ids().to_vec();
    let mut frames = Vec::new();
    // Two interleaved choices share a prompt, but each has its own reasoning.
    for chunk in ids.chunks(size) {
        for index in 0..2 {
            frames.push(proto::GenerateResponse {
                response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                    index,
                    token_ids: chunk.to_vec(),
                    prompt_tokens: 5,
                    ..Default::default()
                })),
            });
        }
    }
    let final_ids = match complete_ids {
        CompleteIds::Full => ids.clone(),
        CompleteIds::LastDelta => ids.chunks(size).last().unwrap_or_default().to_vec(),
        CompleteIds::Empty => Vec::new(),
    };
    for index in 0..2 {
        frames.push(proto::GenerateResponse {
            response: Some(GenerationEvent::Complete(proto::GenerateComplete {
                index,
                output_ids: final_ids.clone(),
                completion_tokens: final_ids.len() as u32,
                prompt_tokens: 5,
                finish_reason: finish.into(),
                ..Default::default()
            })),
        });
    }
    frames
}

#[tokio::test]
async fn vllm_final_reasoning_usage_matches_streaming_and_collected_choices() {
    for parser_name in ["qwen3", "minimax_m3"] {
        for (text, thinking, reason, expected) in [
            ("<think>ab</think>answer", false, "stop", 4),
            ("<think>想🌍</think>answer", false, "stop", 4),
            ("想🌍</think>answer", true, "stop", 4),
            ("<think>想🌍</think></think>answer", false, "stop", 4),
            ("想🌍", true, "length", 4),
            ("answer", false, "stop", 0),
            ("</think>answer", true, "stop", 0),
        ] {
            let text = if parser_name == "minimax_m3" {
                text.replace("<think>", "<mm:think>")
                    .replace("</think>", "</mm:think>")
            } else {
                text.to_string()
            };
            for separate_reasoning in [false, true] {
                let tokenizer: Arc<dyn Tokenizer> = Arc::new(if parser_name == "minimax_m3" {
                    CountingTokenizer::minimax()
                } else {
                    CountingTokenizer::default()
                });
                let request: ChatCompletionRequest = serde_json::from_value(json!({
                    "model": "eof-test", "messages": [], "n": 2,
                    "separate_reasoning": separate_reasoning,
                    "chat_template_kwargs": {"enable_thinking": thinking},
                    "stream_options": {"include_usage": true, "continuous_usage_stats": true}
                }))
                .unwrap();
                let spec = ChatResponseSpec::from(&request);
                let (stream, server) = scripted_stream(
                    frames(tokenizer.as_ref(), &text, 1, reason, CompleteIds::Full),
                    "0",
                )
                .await;
                let streaming = StreamingProcessor::new(
                    ToolParserFactory::new(),
                    ReasoningParserFactory::new(),
                    utils::ParserResolver::new(
                        Arc::new(WorkerRegistry::new()),
                        None,
                        Some(parser_name.to_string()),
                    ),
                    "vllm",
                );
                let non_streaming = ResponseProcessor::new(
                    ToolParserFactory::new(),
                    ReasoningParserFactory::new(),
                    streaming.parser_resolver.clone(),
                );
                let mut decoder =
                    utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
                let response = non_streaming
                    .process_non_streaming_chat_response(
                        context::ExecutionResult::Single { stream },
                        spec.clone(),
                        dispatch(),
                        tokenizer.clone(),
                        &mut decoder,
                        false,
                    )
                    .await
                    .unwrap_or_else(|error| panic!("{}", error.status()));
                server.abort();
                let usage = serde_json::to_value(response.usage.unwrap()).unwrap();
                assert_eq!(
                    usage["completion_tokens_details"]["reasoning_tokens"], expected,
                    "{text}"
                );
                assert_eq!(usage["prompt_tokens"], 5);

                // A marker may share a chunk with both reasoning and answer tokens.
                for (chunk_size, complete_ids) in [1, 3, 64].into_iter().flat_map(|size| {
                    [
                        CompleteIds::Full,
                        CompleteIds::LastDelta,
                        CompleteIds::Empty,
                    ]
                    .map(|mode| (size, mode))
                }) {
                    let (stream, server) = scripted_stream(
                        frames(tokenizer.as_ref(), &text, chunk_size, reason, complete_ids),
                        "0",
                    )
                    .await;
                    let (tx, rx) = sse_channel();
                    streaming
                        .process_streaming_chunks(
                            stream,
                            dispatch(),
                            tokenizer.clone(),
                            (None, None, false, false, false),
                            spec.clone(),
                            &tx,
                            None,
                        )
                        .await
                        .unwrap();
                    drop(tx);
                    let events = collect_events(rx).await;
                    server.abort();
                    let final_event = events
                        .iter()
                        .rev()
                        .find(|event| event["usage"].is_object())
                        .unwrap();
                    assert_eq!(
                        serde_json::to_value(
                            serde_json::from_value::<Usage>(final_event["usage"].clone(),)
                                .unwrap()
                                .with_complete_details(),
                        )
                        .unwrap(),
                        usage,
                        "{text}, chunk_size={chunk_size}, complete_ids={complete_ids:?}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn vllm_missing_complete_keeps_parser_fallback() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CountingTokenizer::default());
    let mut responses = frames(
        tokenizer.as_ref(),
        "<think>abc</think>x",
        2,
        "stop",
        CompleteIds::LastDelta,
    );
    responses.pop(); // choice 1 never completed
    let (stream, server) = scripted_stream(responses, "0").await;
    let mut spec = chat_spec(false);
    spec.stream_options = Some(openai_protocol::common::StreamOptions {
        include_usage: Some(true),
        continuous_usage_stats: Some(true),
        ..Default::default()
    });
    let (tx, rx) = sse_channel();
    let _ = processor(false)
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer,
            (None, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(!events.is_empty());
    // Keep the community parser estimate for these mixed chunks (eight tokens),
    // rather than claiming the exact six-token aggregate before both choices complete.
    let usage = &events
        .iter()
        .rev()
        .find(|event| event["usage"].is_object())
        .unwrap()["usage"];
    assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 8);
}

#[tokio::test]
async fn vllm_nonstreaming_invalid_exact_count_keeps_parser_fallback() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CountingTokenizer::default());
    let mut responses = frames(
        tokenizer.as_ref(),
        "<think>ab</think>answer",
        2,
        "stop",
        CompleteIds::Full,
    );
    for response in &mut responses {
        if let Some(GenerationEvent::Complete(c)) = &mut response.response {
            c.completion_tokens += 1; // Complete IDs are not a trustworthy full sequence.
        }
    }
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"eof-test", "messages":[], "n":2,
        "separate_reasoning":true, "chat_template_kwargs":{"enable_thinking":false}
    }))
    .unwrap();
    let (stream, server) = scripted_stream(responses, "0").await;
    let streaming = processor(false);
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        streaming.parser_resolver.clone(),
    );
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let response = processor
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            ChatResponseSpec::from(&request),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("{}", e.status()));
    server.abort();
    assert_eq!(
        response
            .usage
            .unwrap()
            .completion_tokens_details
            .unwrap()
            .reasoning_tokens,
        Some(4)
    );
}
