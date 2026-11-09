//! The generate endpoint's token series: a generate answer counts its prompt
//! and its cache hit in `smg_router_tokens_total` once per request, beside
//! the completion tokens of every choice, streamed (on the plain loop and on
//! the one that carries logprobs) and collected alike, as the chat answers
//! do; the figures are the ones the client reads in `meta_info`.

use std::time::Instant;

use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::{
    super::*,
    eof::{
        chunk, collect_events, complete_with_prompt, runtime, scripted_stream, CharacterTokenizer,
    },
};
use crate::{
    observability::metrics::test_support::render_with_recorder,
    routers::grpc::regular::processor::ResponseProcessor, worker::WorkerRegistry,
};

const MODEL: &str = "generate-tokens-test";

/// The `Complete` of index `index` on `stop`, reporting the request's prompt
/// of 7 tokens, 3 of them a cache hit, as every choice's `Complete` reports
/// the one prompt.
fn complete_with_cache_hit(index: u32) -> proto::GenerateResponse {
    let mut terminal = complete_with_prompt(index, "stop", 7);
    if let Some(GenerationEvent::Complete(complete)) = &mut terminal.response {
        complete.cached_tokens = 3;
    }
    terminal
}

/// The value of the generate endpoint's `smg_router_tokens_total` series of
/// `token_type` for this file's model in `rendered`, if the series exists.
fn tokens(rendered: &str, token_type: &str) -> Option<f64> {
    let model = format!("model=\"{MODEL}\"");
    let token_type = format!("token_type=\"{token_type}\"");
    rendered
        .lines()
        .find(|line| {
            line.starts_with("smg_router_tokens_total{")
                && line.contains("endpoint=\"generate\"")
                && line.contains(&model)
                && line.contains(&token_type)
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// The scrape after a two-choice generate stream of one chunk per choice,
/// driven over the plain loop or, with `with_logprobs`, the loop that
/// carries logprobs.
fn streamed_scrape(with_logprobs: bool) -> String {
    let rt = runtime();
    render_with_recorder(|| {
        rt.block_on(async {
            let (stream, server) = scripted_stream(
                vec![
                    chunk(0, "hi"),
                    chunk(1, "yo"),
                    complete_with_cache_hit(0),
                    complete_with_cache_hit(1),
                ],
                "0",
            )
            .await;
            let ctx = GenerateStreamContext {
                request_id: "gen-tokens".to_string(),
                weight_version: "default".to_string(),
                return_logprob: with_logprobs,
                backend_type: "test",
                model: MODEL.to_string(),
                expected_choices: 2,
            };
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
            let (tx, rx) = sse_channel();
            let result = if with_logprobs {
                StreamingProcessor::process_generate_streaming_with_input_logprobs(
                    tokenizer, stream, ctx, None, &tx, None, None,
                )
                .await
            } else {
                StreamingProcessor::process_generate_streaming(tokenizer, stream, ctx, &tx, None)
                    .await
            };
            drop(tx);
            let _events = collect_events(rx).await;
            server.abort();
            assert!(result.is_ok(), "{result:?}");
        });
    })
}

#[test]
fn a_generate_stream_counts_its_prompt_and_cache_hit_once_per_request() {
    for with_logprobs in [false, true] {
        let rendered = streamed_scrape(with_logprobs);
        assert_eq!(
            tokens(&rendered, "input"),
            Some(7.0),
            "the prompt once, not once per choice (logprobs {with_logprobs}): {rendered}"
        );
        assert_eq!(
            tokens(&rendered, "cached"),
            Some(3.0),
            "the cache hit once (logprobs {with_logprobs}): {rendered}"
        );
        assert_eq!(
            tokens(&rendered, "output"),
            Some(4.0),
            "the completion tokens of both choices (logprobs {with_logprobs}): {rendered}"
        );
    }
}

#[test]
fn a_collected_generate_answer_counts_its_prompt_and_cache_hit_once_per_request() {
    let rt = runtime();
    let rendered = render_with_recorder(|| {
        rt.block_on(async {
            let (stream, server) = scripted_stream(
                vec![complete_with_cache_hit(0), complete_with_cache_hit(1)],
                "0",
            )
            .await;
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
            let mut decoder =
                utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
            let dispatch = context::DispatchMetadata {
                request_id: "gen-tokens".to_string(),
                model: MODEL.to_string(),
                created: 1,
                weight_version: None,
            };
            let responses = ResponseProcessor::new(
                ToolParserFactory::new(),
                ReasoningParserFactory::new(),
                utils::ParserResolver::with_flags(Arc::new(WorkerRegistry::new()), None, None),
            )
            .process_non_streaming_generate_response(
                context::ExecutionResult::Single {
                    stream,
                    dispatch_start: Instant::now(),
                },
                dispatch,
                &mut decoder,
                false,
                Instant::now(),
                metrics_labels::ENDPOINT_GENERATE,
            )
            .await
            .unwrap_or_else(|response| panic!("{}", response.status()));
            server.abort();
            // The figures the scrape counts are the ones every choice's
            // `meta_info` reports to the client.
            assert_eq!(responses.len(), 2, "one answer per choice");
            for response in &responses {
                assert_eq!(response.meta_info.prompt_tokens, 7);
                assert_eq!(response.meta_info.cached_tokens, 3);
            }
        });
    });
    assert_eq!(
        tokens(&rendered, "input"),
        Some(7.0),
        "the prompt once, not once per choice: {rendered}"
    );
    assert_eq!(
        tokens(&rendered, "cached"),
        Some(3.0),
        "the cache hit once: {rendered}"
    );
    assert_eq!(
        tokens(&rendered, "output"),
        Some(24.0),
        "the completion tokens of both choices: {rendered}"
    );
}
