//! Chat Completions logprobs over a real gRPC worker connection: a unary
//! response's `logprobs.content` covers the same tokens as the logprobs on a
//! stream's content deltas, whether the older parsers or a Symphony table read
//! the output.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_tokenizer.rs"]
#[expect(dead_code, reason = "this test decodes chunks only")]
mod scripted_tokenizer;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{sync::Arc, time::Duration};

use axum::body::to_bytes;
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{ParserChoice, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, time::timeout};

const MODEL: &str = "chat-logprobs-model";

/// The model's output, one chunk per token; the worker sends one more token
/// that decodes to nothing, like an EOS token.
const CHUNKS: &[&str] = &[
    "<think>", "check", " the", " map", "</think>", "Hello", " my", " friend", ".",
];

/// A worker's output and how the gateway is set up to read it.
struct Script {
    /// The model the worker serves.
    model: &'static str,
    /// The model's output, one chunk per token (see [`CHUNKS`]).
    chunks: &'static [&'static str],
    /// The gateway's `--parser`, which the response's `x-smg-parser` header
    /// names back.
    parser: ParserChoice,
    /// The reasoning parser the gateway is configured with, for the older
    /// parsers.
    reasoning_parser: Option<&'static str>,
    /// Whether the request declares the `lookup` tool.
    tools: bool,
}

/// The older parsers read the output, as by default.
const OLDER: Script = Script {
    model: MODEL,
    chunks: CHUNKS,
    parser: ParserChoice::Legacy,
    reasoning_parser: Some("qwen3"),
    tools: false,
};

/// Models Symphony's registry knows, each read by its table under
/// `--parser symphony`, with a thought, the content `Hello my friend.` and a
/// call in the table's syntax: JSON (Qwen3), tagged (Qwen 3.5), Gemma's own
/// (Gemma 4) and a Python list (LFM2.5).
const SYMPHONY: &[Script] = &[
    Script {
        model: "Qwen/Qwen3-8B",
        chunks: &[
            "<think>",
            "check",
            " the",
            " map",
            "</think>",
            "Hello",
            " my",
            " friend",
            ".",
            "<tool_call>",
            "\n{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n",
            "</tool_call>",
        ],
        parser: ParserChoice::Symphony,
        reasoning_parser: None,
        tools: true,
    },
    Script {
        model: "Qwen/Qwen3.5-9B",
        chunks: &[
            "<think>",
            "check",
            " the",
            " map",
            "</think>",
            "Hello",
            " my",
            " friend",
            ".",
            "<tool_call>",
            "\n<function=lookup>\n<parameter=q>\n1\n</parameter>\n</function>\n",
            "</tool_call>",
        ],
        parser: ParserChoice::Symphony,
        reasoning_parser: None,
        tools: true,
    },
    Script {
        model: "google/gemma-4-E4B-it",
        chunks: &[
            "<|channel>",
            "thought",
            "\n",
            "check",
            " the",
            " map",
            "<channel|>",
            "<|tool_call>",
            "call:lookup{q:1}",
            "<tool_call|>",
            "Hello",
            " my",
            " friend",
            ".",
        ],
        parser: ParserChoice::Symphony,
        reasoning_parser: None,
        tools: true,
    },
    Script {
        model: "LiquidAI/LFM2.5-1.2B-Instruct",
        chunks: &[
            "<think>",
            "check",
            " the",
            " map",
            "</think>",
            "Hello",
            " my",
            " friend",
            ".",
            "<|tool_call_start|>",
            "[lookup(q=1)]",
            "<|tool_call_end|>",
        ],
        parser: ParserChoice::Symphony,
        reasoning_parser: None,
        tools: true,
    },
];

/// A chat response's message and the `(token, logprob)` pairs of its content:
/// a unary response's message and `logprobs.content`, or the message a
/// stream's content and reasoning deltas make and the logprobs on its content
/// deltas. The response's `x-smg-parser` header names the script's parser.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn chat_logprobs(script: &Script, stream: bool) -> (Value, Vec<(String, f64)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(script.chunks.len() + 1).unwrap(),
            finish_reasons: vec!["stop"],
            generation: Default::default(),
        }
        .serve(listener),
    );
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(script.model))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.parser = script.parser;
    config.reasoning_parser = script.reasoning_parser.map(Into::into);
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(
            script
                .chunks
                .iter()
                .map(|chunk| chunk.to_string())
                .collect(),
        ));
    tokenizers
        .load("tokenizer-id", script.model, "test", || async move {
            Ok(tokenizer)
        })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let mut request = json!({
        "model": script.model, "max_tokens": 16, "stream": stream, "logprobs": true,
        "messages": [{"role": "user", "content": "Hello"}],
    });
    if script.tools {
        request["tools"] = json!([{
            "type": "function", "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"q": {"type": "integer"}}},
            }
        }]);
    }
    let request = serde_json::from_value(request).unwrap();
    let bytes = timeout(Duration::from_secs(30), async {
        let response = router
            .route_chat(None, &tenant, request, script.model)
            .await;
        assert_eq!(
            response.headers().get("x-smg-parser").unwrap(),
            script.parser.name(),
            "{}",
            script.model
        );
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap()
    })
    .await
    .expect("chat response should finish");
    server.abort();
    let body = std::str::from_utf8(&bytes).unwrap();
    let items = |logprobs: &Value| -> Vec<(String, f64)> {
        logprobs["content"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|item| {
                let token = item["token"].as_str().unwrap().to_string();
                (token, item["logprob"].as_f64().unwrap())
            })
            .collect()
    };
    if !stream {
        let response: Value = serde_json::from_str(body).unwrap();
        let choice = &response["choices"][0];
        return (choice["message"].clone(), items(&choice["logprobs"]));
    }
    let (mut content, mut reasoning, mut logprobs) = (String::new(), String::new(), Vec::new());
    for choice in body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .filter_map(|event| event["choices"].get(0).cloned())
    {
        if let Some(text) = choice["delta"]["reasoning_content"].as_str() {
            reasoning.push_str(text);
        }
        if let Some(text) = choice["delta"]["content"]
            .as_str()
            .filter(|text| !text.is_empty())
        {
            content.push_str(text);
            logprobs.extend(items(&choice["logprobs"]));
        }
    }
    let message = json!({"content": content, "reasoning_content": reasoning});
    (message, logprobs)
}

/// A unary response's logprobs cover its content's tokens, as a stream's
/// content deltas do: not the reasoning, its delimiters or the EOS token.
#[tokio::test]
async fn unary_logprobs_cover_the_content_as_a_stream_does() {
    // Token `100 + i` (chunk `i`) reports logprob `-(i + 1)`.
    let content: Vec<(String, f64)> = ["Hello", " my", " friend", "."]
        .iter()
        .map(|token| token.to_string())
        .zip([-6.0, -7.0, -8.0, -9.0])
        .collect();
    let (message, logprobs) = chat_logprobs(&OLDER, false).await;
    assert_eq!(message["content"], "Hello my friend.", "{message}");
    assert_eq!(message["reasoning_content"], "check the map", "{message}");
    assert_eq!(logprobs, content);
    assert_eq!(chat_logprobs(&OLDER, true).await.1, content);
}

/// With a Symphony table reading the output, a unary response's logprobs
/// cover its content's tokens as well, as the stream's content deltas do: not
/// the thought, the table's markers, the call or the EOS token.
#[tokio::test]
async fn a_symphony_tables_unary_logprobs_cover_the_content_as_its_stream_does() {
    for script in SYMPHONY {
        // The content is four chunks from `Hello` on; token `100 + i` (chunk
        // `i`) reports logprob `-(i + 1)`.
        let first = script
            .chunks
            .iter()
            .position(|chunk| *chunk == "Hello")
            .expect("the content's first chunk");
        let content: Vec<(String, f64)> = script.chunks[first..first + 4]
            .iter()
            .zip(first..)
            .map(|(chunk, i)| (chunk.to_string(), -((i + 1) as f64)))
            .collect();
        for stream in [false, true] {
            let (message, logprobs) = chat_logprobs(script, stream).await;
            let case = format!("{} (stream: {stream}): {message}", script.model);
            assert_eq!(message["content"], "Hello my friend.", "{case}");
            assert_eq!(message["reasoning_content"], "check the map", "{case}");
            if !stream {
                assert_eq!(
                    message["tool_calls"][0]["function"]["name"], "lookup",
                    "{case}"
                );
            }
            assert_eq!(logprobs, content, "{case}");
        }
    }
}
