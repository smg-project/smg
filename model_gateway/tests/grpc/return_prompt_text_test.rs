//! `return_prompt_text` on chat completions over gRPC workers: when the
//! request asks, the answer carries the prompt as the gateway rendered it for
//! the model, at the top level of a collected answer and on the first event
//! of a stream alone; a request that does not ask reads no such key. Run
//! against the mock engine on each wire, on the regular route and on the
//! prefill and decode route, for a model whose stream goes out as it is
//! written and for one whose stream is re-sliced on its way out.

use std::{sync::Arc, time::Duration};

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_worker, serve_mock_worker_with,
};
use mock_worker::config::{Config, SchedulerWire};
use openai_protocol::model_card::ModelCard;
use serde_json::{json, Value};
use smg::{middleware::TenantRequestMeta, tenant::TenantKey};
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerType};

use super::{events, read};
use crate::common;

/// A model on the wire that sends a stream's events as they are written.
const MODEL: &str = "return-prompt-text-test-model";

/// A model on the wire that re-slices a stream's events on their way out
/// (`routers::common::sse_rechunk`).
const RESLICED_MODEL: &str = "MiniMax-M3";

/// What the mock tokenizer's template renders for the test's one user turn:
/// a line per message, then the assistant's opening.
const PROMPT: &str = "user: say \"hi\"\nassistant: ";

#[tokio::test]
async fn a_chat_answer_carries_the_rendered_prompt_only_when_the_request_asks() {
    let tenant = TenantRequestMeta::new(TenantKey::new("return-prompt-text-test-tenant"));
    for (wire, runtime) in [
        (SchedulerWire::Vllm, RuntimeType::Vllm),
        (SchedulerWire::TokenSpeed, RuntimeType::TokenSpeed),
    ] {
        for (roles, model) in [
            (&[WorkerType::Regular][..], MODEL),
            (&[WorkerType::Prefill, WorkerType::Decode][..], MODEL),
            (&[WorkerType::Regular][..], RESLICED_MODEL),
            (
                &[WorkerType::Prefill, WorkerType::Decode][..],
                RESLICED_MODEL,
            ),
        ] {
            let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
            for role in roles {
                let port = serve_mock_worker_with(|port| Config {
                    http_count: 0,
                    grpc_base_port: port,
                    grpc_count: 1,
                    grpc_wire: wire,
                    model_id: model.to_string(),
                    tokenizer_path: model.to_string(),
                    gen_delay: Duration::ZERO,
                    output_tokens: 3,
                    realistic: false,
                    ..Config::default()
                })
                .await;
                workers.push(Arc::new(
                    grpc_worker(port, *role, runtime, ModelCard::new(model)).build(),
                ));
            }
            let config = if roles.len() == 1 {
                grpc_router_config().regular_mode(vec![])
            } else {
                grpc_router_config().prefill_decode_mode(vec![], vec![])
            }
            .round_robin_policy()
            .build_unchecked();
            let router = build_grpc_router(config, model, workers).await;
            for (stream, asks) in [(false, true), (true, true), (false, false), (true, false)] {
                let mut body = json!({
                    "model": model, "max_tokens": 8, "stream": stream,
                    "messages": [{"role": "user", "content": "say \"hi\""}],
                });
                if stream {
                    // Usage on every chunk: the stream's first event carries
                    // a usage object, and the re-sliced wire splits such an
                    // event into an opening and a closing one.
                    body["stream_options"] =
                        json!({"include_usage": true, "continuous_usage_stats": true});
                }
                if asks {
                    body["return_prompt_text"] = json!(true);
                }
                let request = serde_json::from_value(body).unwrap();
                let answer = read(router.route_chat(None, &tenant, Arc::new(request), model)).await;
                // A collected answer is one object; a stream is its events,
                // the first of them the one that opens the choice.
                let events = if stream {
                    events(&answer)
                } else {
                    vec![serde_json::from_str(&answer).unwrap()]
                };
                let prompts: Vec<Option<&Value>> = events
                    .iter()
                    .map(|event| event.get("prompt_text"))
                    .collect();
                let case =
                    format!("{model}, {wire:?}, {roles:?}, stream {stream}, asks {asks}: {answer}");
                assert_eq!(prompts[0], asks.then_some(&json!(PROMPT)), "{case}");
                assert!(prompts[1..].iter().all(Option::is_none), "{case}");
            }
        }
    }
}
