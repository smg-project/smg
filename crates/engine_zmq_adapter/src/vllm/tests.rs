use engine_zmq_client::{
    mock_engine::EngineInbound,
    protocol::vllm::{
        logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
        output::{EngineCoreOutputs, RequestBatchOutputs},
    },
};
use openai_protocol::worker::RuntimeType;

use super::*;
use crate::{eos::EosTokenIds, test_support::connected_pair};

fn batch(
    request_id: &str,
    token: u32,
    logprob: Option<f32>,
    finish: Option<EngineCoreFinishReason>,
) -> EngineCoreOutputs {
    let finished = finish.map(|_| BTreeSet::from([request_id.to_string()]));
    let new_logprobs = logprob.map(|lp| Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![TokenLogprob {
                token_id: token,
                logprob: lp,
                rank: 1,
            }],
        }],
    });
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: vec![token],
            new_logprobs,
            finish_reason: finish,
            ..Default::default()
        }],
        finished_requests: finished,
        ..Default::default()
    })
}

/// End-to-end over ipc://: the adapter translates a vLLM-proto request to
/// EngineCore, and maps the engine's outputs back to vLLM-proto responses.
#[tokio::test]
async fn generate_e2e_translates_and_streams_vllm_proto() {
    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;

    #[expect(
        clippy::disallowed_methods,
        reason = "engine task ends after responding"
    )]
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let inbound = input.recv().await.unwrap();
        let request = match inbound {
            EngineInbound::Add(request) => request,
            other => panic!("expected Add, got {other:?}"),
        };
        assert_eq!(request.request_id, "r1");
        assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
        assert_eq!(request.sampling_params.as_ref().unwrap().max_tokens, 2);
        output
            .send_outputs(&batch("r1", 10, Some(-0.5), None))
            .await
            .unwrap();
        output
            .send_outputs(&batch(
                "r1",
                11,
                Some(-1.25),
                Some(EngineCoreFinishReason::Length),
            ))
            .await
            .unwrap();
        // Keep the peer alive until the frontend has consumed the replies.
        (input, output)
    });

    let req = vllm::GenerateRequest {
        request_id: "r1".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            max_tokens: Some(2),
            logprobs: Some(1),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    };
    let mut stream = client.generate_vllm(req).await.expect("generate");

    let first = stream.next().await.expect("chunk item").expect("chunk ok");
    match first.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            assert_eq!(chunk.token_ids, vec![10]);
            let logprobs = chunk.output_logprobs.expect("chunk logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-0.5]);
            assert_eq!(logprobs.token_ids, vec![10]);
        }
        other => panic!("expected chunk, got {other:?}"),
    }
    // The finish tick carried a new token, so its delta is emitted as a
    // chunk before the (cumulative) terminal complete.
    let second = stream.next().await.expect("chunk item").expect("chunk ok");
    match second.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            assert_eq!(chunk.token_ids, vec![11]);
            let logprobs = chunk.output_logprobs.expect("chunk logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-1.25]);
            assert_eq!(logprobs.token_ids, vec![11]);
        }
        other => panic!("expected chunk, got {other:?}"),
    }
    let third = stream
        .next()
        .await
        .expect("complete item")
        .expect("complete ok");
    match third.response {
        Some(vllm::generate_response::Response::Complete(complete)) => {
            assert_eq!(complete.output_ids, vec![10, 11]);
            assert_eq!(complete.finish_reason, "length");
            assert_eq!(complete.completion_tokens, 2);
            let logprobs = complete.output_logprobs.expect("complete logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
            assert_eq!(logprobs.token_ids, vec![10, 11]);
        }
        other => panic!("expected complete, got {other:?}"),
    }
    assert!(stream.next().await.is_none());

    engine_task.await.unwrap();
}

/// The sampled token also ranks first — the common case under greedy or
/// low-temperature decoding. The engine repeats it in the ranked columns,
/// so the shaped list must carry it once and still return `k` candidates.
#[test]
fn shape_top_logprobs_dedups_sampled_token_at_rank_one() {
    let entries = vec![
        TokenLogprob {
            token_id: 10,
            logprob: -0.1,
            rank: 1,
        },
        TokenLogprob {
            token_id: 10,
            logprob: -0.1,
            rank: 1,
        },
        TokenLogprob {
            token_id: 20,
            logprob: -0.3,
            rank: 2,
        },
    ];
    assert_eq!(
        shape_top_logprobs(&entries, 2),
        vllm::TopLogProbs {
            values: vec![-0.1, -0.3],
            token_ids: vec![10, 20],
        }
    );
    assert_eq!(
        shape_top_logprobs(&entries, 1),
        vllm::TopLogProbs {
            values: vec![-0.1],
            token_ids: vec![10],
        }
    );
}

/// A sampled token outside the top-k leads the list and the ranked
/// candidates follow in order, truncated to the requested count.
#[test]
fn shape_top_logprobs_keeps_sampled_token_outside_top_k() {
    let entries = vec![
        TokenLogprob {
            token_id: 10,
            logprob: -0.5,
            rank: 5,
        },
        TokenLogprob {
            token_id: 20,
            logprob: -0.1,
            rank: 1,
        },
        TokenLogprob {
            token_id: 30,
            logprob: -0.3,
            rank: 2,
        },
    ];
    assert_eq!(
        shape_top_logprobs(&entries, 2),
        vllm::TopLogProbs {
            values: vec![-0.5, -0.1],
            token_ids: vec![10, 20],
        }
    );
}

/// With `logprobs=k`, each position's ranked candidates are shaped into
/// `top_logprobs`, taking the sampled entry plus the leading candidates up
/// to the requested count (matching the gRPC servicer's `islice` behaviour).
#[tokio::test]
async fn generate_shapes_top_logprobs_to_requested_count() {
    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;

    // One position with the sampled token (actual vocab rank) first, then
    // the engine's ranked candidates. The wire carries `k + 1` entries.
    let position = PositionLogprobs {
        entries: vec![
            TokenLogprob {
                token_id: 10,
                logprob: -0.5,
                rank: 5,
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 30,
                logprob: -0.3,
                rank: 2,
            },
        ],
    };
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "r1".to_string(),
            new_token_ids: vec![10],
            new_logprobs: Some(Logprobs {
                positions: vec![position],
            }),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }],
        finished_requests: Some(BTreeSet::from(["r1".to_string()])),
        ..Default::default()
    });

    #[expect(
        clippy::disallowed_methods,
        reason = "engine task ends after responding"
    )]
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let inbound = input.recv().await.unwrap();
        let request = match inbound {
            EngineInbound::Add(request) => request,
            other => panic!("expected Add, got {other:?}"),
        };
        assert_eq!(request.sampling_params.as_ref().unwrap().logprobs, Some(2));
        output.send_outputs(&outputs).await.unwrap();
        // Keep the peer alive until the frontend has consumed the replies.
        (input, output)
    });

    let req = vllm::GenerateRequest {
        request_id: "r1".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            max_tokens: Some(1),
            logprobs: Some(2),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    };
    let mut stream = client.generate_vllm(req).await.expect("generate");

    // The requested count is 2, so `top_logprobs` keeps the sampled entry
    // plus the leading candidate (the third entry is dropped).
    let expected_top = vec![vllm::TopLogProbs {
        values: vec![-0.5, -0.1],
        token_ids: vec![10, 20],
    }];

    // The finish tick carried a token, so the delta streams as a chunk.
    let chunk = stream.next().await.expect("chunk item").expect("chunk ok");
    match chunk.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            let logprobs = chunk.output_logprobs.expect("chunk logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-0.5]);
            assert_eq!(logprobs.token_ids, vec![10]);
            assert_eq!(logprobs.top_logprobs, expected_top);
        }
        other => panic!("expected chunk, got {other:?}"),
    }
    let complete = stream
        .next()
        .await
        .expect("complete item")
        .expect("complete ok");
    match complete.response {
        Some(vllm::generate_response::Response::Complete(complete)) => {
            let logprobs = complete.output_logprobs.expect("complete logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-0.5]);
            assert_eq!(logprobs.token_ids, vec![10]);
            assert_eq!(logprobs.top_logprobs, expected_top);
        }
        other => panic!("expected complete, got {other:?}"),
    }
    assert!(stream.next().await.is_none());

    engine_task.await.unwrap();
}

/// Prompt logprobs end to end: the request carries `prompt_logprobs` to the
/// engine, and the engine's prompt tensors come back as `input_logprobs` on
/// the first token-bearing chunk (once) and on the terminal `Complete`.
#[tokio::test]
async fn generate_streams_prompt_logprobs() {
    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;

    // Prompt position for input token 2 (the engine scores every prompt
    // token but the first), with one ranked candidate behind it.
    let prompt_tensors = Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: 2,
                    logprob: -0.5,
                    rank: 3,
                },
                TokenLogprob {
                    token_id: 20,
                    logprob: -0.1,
                    rank: 1,
                },
            ],
        }],
    };
    let prefill = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "r1".to_string(),
            new_token_ids: vec![10],
            new_prompt_logprobs_tensors: Some(prompt_tensors),
            ..Default::default()
        }],
        ..Default::default()
    });
    let decode = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "r1".to_string(),
            new_token_ids: vec![11],
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }],
        finished_requests: Some(BTreeSet::from(["r1".to_string()])),
        ..Default::default()
    });

    #[expect(
        clippy::disallowed_methods,
        reason = "engine task ends after responding"
    )]
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let inbound = input.recv().await.unwrap();
        let request = match inbound {
            EngineInbound::Add(request) => request,
            other => panic!("expected Add, got {other:?}"),
        };
        assert_eq!(
            request.sampling_params.as_ref().unwrap().prompt_logprobs,
            Some(1),
            "prompt_logprobs reaches the engine"
        );
        output.send_outputs(&prefill).await.unwrap();
        output.send_outputs(&decode).await.unwrap();
        // Keep the peer alive until the frontend has consumed the replies.
        (input, output)
    });

    let req = vllm::GenerateRequest {
        request_id: "r1".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            max_tokens: Some(2),
            prompt_logprobs: Some(1),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    };
    let mut stream = client.generate_vllm(req).await.expect("generate");

    // Prompt token 1 leads with a null logprob (nothing precedes it); the
    // requested count of 1 keeps the prompt token's own ranked entry.
    let expect_input_logprobs = |logprobs: Option<vllm::InputLogProbs>, whose: &str| {
        let logprobs = logprobs.unwrap_or_else(|| panic!("{whose} input logprobs"));
        assert_eq!(logprobs.token_ids, vec![1, 2]);
        assert_eq!(
            logprobs.token_logprobs,
            vec![
                vllm::InputTokenLogProb { value: None },
                vllm::InputTokenLogProb { value: Some(-0.5) },
            ]
        );
        assert_eq!(
            logprobs.top_logprobs,
            vec![
                vllm::TopLogProbs::default(),
                vllm::TopLogProbs {
                    values: vec![-0.5],
                    token_ids: vec![2],
                },
            ]
        );
    };

    let first = stream.next().await.expect("chunk item").expect("chunk ok");
    match first.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            expect_input_logprobs(chunk.input_logprobs, "first chunk");
        }
        other => panic!("expected chunk, got {other:?}"),
    }
    // The finish tick carried a token, so its delta streams as a chunk
    // first — without repeating the prompt logprobs.
    let second = stream.next().await.expect("chunk item").expect("chunk ok");
    match second.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            assert!(
                chunk.input_logprobs.is_none(),
                "prompt logprobs ride the first chunk only"
            );
        }
        other => panic!("expected chunk, got {other:?}"),
    }
    let complete = stream
        .next()
        .await
        .expect("complete item")
        .expect("complete ok");
    match complete.response {
        Some(vllm::generate_response::Response::Complete(complete)) => {
            expect_input_logprobs(complete.input_logprobs, "complete");
        }
        other => panic!("expected complete, got {other:?}"),
    }

    engine_task.await.unwrap();
}

#[test]
fn finish_reasons_map_to_vllm_strings() {
    assert_eq!(finish_reason_str(EngineCoreFinishReason::Length), "length");
    assert_eq!(
        finish_reason_str(EngineCoreFinishReason::Repetition),
        "stop"
    );
    assert_eq!(finish_reason_str(EngineCoreFinishReason::Abort), "abort");
}

fn tokenized_req(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
    vllm::GenerateRequest {
        request_id: "r1".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(sampling),
        stream: true,
        ..Default::default()
    }
}

/// A grammar the gateway led for a first-token start reaches the engine
/// with `reasoning_ended = true`, the gate behind its reasoning parser
/// taken as passed; a grammar the gateway did not shape, and a request
/// that constrains nothing, say nothing of it.
#[test]
fn a_grammar_led_for_the_first_token_starts_there_whatever_the_engines_parser() {
    let translated = |sampling| {
        translate_request(
            tokenized_req(sampling),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("translated")
    };
    let led = translated(vllm::SamplingParams {
        constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
        grammar_from_first_token: true,
        ..Default::default()
    });
    assert_eq!(led.reasoning_ended, Some(true));
    let unshaped = translated(vllm::SamplingParams {
        constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
        ..Default::default()
    });
    assert_eq!(unshaped.reasoning_ended, None);
    let free = translated(vllm::SamplingParams {
        grammar_from_first_token: true,
        ..Default::default()
    });
    assert_eq!(free.reasoning_ended, None, "no grammar, nothing to start");
}

#[test]
fn vllm_forwards_prompt_logprobs() {
    // Prompt logprobs ride the wire sampling params verbatim.
    let request = translate_request(
        tokenized_req(vllm::SamplingParams {
            prompt_logprobs: Some(2),
            ..Default::default()
        }),
        4096,
        ModelDtype::BFloat16,
        &EosTokenIds::default(),
    )
    .expect("translated");
    assert_eq!(
        request.sampling_params.as_ref().unwrap().prompt_logprobs,
        Some(2)
    );
}

#[test]
fn kv_transfer_params_prefer_json_then_the_legacy_pair() {
    let mut req = tokenized_req(vllm::SamplingParams::default());
    assert_eq!(kv_transfer_params(&req).unwrap(), None);

    req.kv_transfer_params = Some(vllm::KvTransferParams {
        remote_host: "10.0.0.1".to_string(),
        remote_port: 5600,
    });
    assert_eq!(
        kv_transfer_params(&req).unwrap(),
        Some(serde_json::json!({"remote_host": "10.0.0.1", "remote_port": 5600}))
    );
    // The JSON field wins over the legacy pair.
    req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
    assert_eq!(
        kv_transfer_params(&req).unwrap(),
        Some(serde_json::json!({"do_remote_decode": true}))
    );
    // Malformed params are the caller's error, as on the Python servicer.
    req.kv_transfer_params_json = Some("[1]".to_string());
    assert!(kv_transfer_params(&req)
        .unwrap_err()
        .contains("JSON object"));
    req.kv_transfer_params_json = Some("{".to_string());
    assert!(kv_transfer_params(&req).unwrap_err().contains("invalid"));
    req.kv_transfer_params_json = None;
    req.kv_transfer_params = Some(vllm::KvTransferParams {
        remote_host: String::new(),
        remote_port: 5600,
    });
    assert!(kv_transfer_params(&req)
        .unwrap_err()
        .contains("remote_host"));
}

#[test]
fn kv_transfer_params_ride_the_sampling_extra_args() {
    let mut req = tokenized_req(vllm::SamplingParams::default());
    req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
    let request = translate_request(req, 4096, ModelDtype::BFloat16, &EosTokenIds::default())
        .expect("translated");
    let extra = request
        .sampling_params
        .as_ref()
        .and_then(|sp| sp.extra_args.as_ref())
        .expect("extra_args");
    assert_eq!(
        extra["kv_transfer_params"],
        serde_json::json!({"do_remote_decode": true})
    );
    // A request without sampling params still carries them.
    let mut req = tokenized_req(vllm::SamplingParams::default());
    req.sampling_params = None;
    req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
    let request = translate_request(req, 4096, ModelDtype::BFloat16, &EosTokenIds::default())
        .expect("translated");
    assert!(request
        .sampling_params
        .and_then(|sp| sp.extra_args)
        .is_some_and(|extra| extra.contains_key("kv_transfer_params")));
}

#[test]
fn returned_kv_transfer_params_land_on_the_complete() {
    let params = serde_json::json!({
        "do_remote_prefill": true,
        "remote_block_ids": [1, 2],
        "remote_host": "10.0.0.1",
        "remote_port": 5600,
    });
    // Direct Complete.
    let mut response = vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Complete(
            vllm::GenerateComplete::default(),
        )),
    };
    let mut pending = None;
    attach_kv_transfer_params(&mut response, &mut pending, Some(params.clone()));
    let Some(vllm::generate_response::Response::Complete(complete)) = response.response else {
        panic!("complete");
    };
    let json: serde_json::Value =
        serde_json::from_str(complete.kv_transfer_params_json.as_deref().unwrap()).unwrap();
    assert_eq!(json, params);
    let legacy = complete.kv_transfer_params.unwrap();
    assert_eq!(
        (legacy.remote_host.as_str(), legacy.remote_port),
        ("10.0.0.1", 5600)
    );

    // Complete parked behind the finish tick's chunk.
    let mut response = vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk::default(),
        )),
    };
    let mut pending = Some(vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Complete(
            vllm::GenerateComplete::default(),
        )),
    });
    attach_kv_transfer_params(
        &mut response,
        &mut pending,
        Some(serde_json::json!({"remote_engine_id": "eng-a"})),
    );
    let Some(vllm::generate_response::Response::Complete(parked)) = pending.unwrap().response
    else {
        panic!("parked complete");
    };
    assert_eq!(
        parked.kv_transfer_params_json.as_deref(),
        Some(r#"{"remote_engine_id":"eng-a"}"#)
    );
    // No host/port: no legacy mirror.
    assert!(parked.kv_transfer_params.is_none());

    // A non-finish chunk carries nothing.
    let mut response = vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk::default(),
        )),
    };
    let mut pending = None;
    attach_kv_transfer_params(&mut response, &mut pending, Some(params));
    assert!(pending.is_none());
}

#[test]
fn structured_output_backend_follows_the_engine_config() {
    use engine_zmq_client::protocol::vllm::structured_outputs::{
        StructuredOutputBackend, StructuredOutputConstraint,
    };
    use StructuredOutputsBackendConfig::{Auto, Pinned};

    assert_eq!(
        structured_outputs_backend_from_config("guidance"),
        Pinned(StructuredOutputBackend::Guidance)
    );
    assert_eq!(
        structured_outputs_backend_from_config(" outlines "),
        Pinned(StructuredOutputBackend::Outlines)
    );
    assert_eq!(
        structured_outputs_backend_from_config("lm-format-enforcer"),
        Pinned(StructuredOutputBackend::LmFormatEnforcer)
    );
    // vLLM reads the backend off an option-carrying spelling's prefix.
    assert_eq!(
        structured_outputs_backend_from_config("xgrammar:disable-any-whitespace"),
        Pinned(StructuredOutputBackend::Xgrammar)
    );
    // `auto` (and anything unknown) is the engine's own default.
    assert_eq!(structured_outputs_backend_from_config("auto"), Auto);
    assert_eq!(structured_outputs_backend_from_config(""), Auto);
    assert_eq!(StructuredOutputsBackendConfig::default(), Auto);

    let constrained = || {
        tokenized_req(vllm::SamplingParams {
            constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
            ..Default::default()
        })
    };
    let backend_of = |request: EngineCoreRequest| {
        request
            .sampling_params
            .and_then(|sp| sp.structured_outputs)
            .map(|so| so.backend)
            .expect("structured outputs")
    };
    // Pinned by the caller: every constraint carries the engine's backend.
    let request = translate_request_with_backend(
        constrained(),
        4096,
        ModelDtype::BFloat16,
        &EosTokenIds::default(),
        Pinned(StructuredOutputBackend::Guidance),
    )
    .expect("translated");
    assert_eq!(backend_of(request), StructuredOutputBackend::Guidance);
    // Nothing configured: the engine's `auto`.
    let request = translate_request(
        constrained(),
        4096,
        ModelDtype::BFloat16,
        &EosTokenIds::default(),
    )
    .expect("translated");
    assert_eq!(backend_of(request), StructuredOutputBackend::Xgrammar);

    // `auto`: xgrammar for every constraint, the engine's one backend
    // being the one structural tags compile on. A Lark grammar goes to
    // xgrammar unchanged (vLLM 0.30 parses Lark there); a choice is
    // lowered to the grammar vLLM's frontend substitutes.
    let auto = |constraint| {
        translate_request_with_backend(
            tokenized_req(vllm::SamplingParams {
                constraint: Some(constraint),
                ..Default::default()
            }),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
            Auto,
        )
        .expect("translated")
        .sampling_params
        .and_then(|sp| sp.structured_outputs)
        .expect("structured outputs")
    };
    let lark = "start: \"yes\" | \"no\"";
    let params = auto(vllm::sampling_params::Constraint::Grammar(lark.to_string()));
    assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
    assert_eq!(
        params.constraint,
        StructuredOutputConstraint::Grammar(lark.to_string())
    );
    let params = auto(vllm::sampling_params::Constraint::Choice(
        vllm::ChoiceConstraint {
            choices: vec!["a".to_string(), "b".to_string()],
        },
    ));
    assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
    assert_eq!(
        params.constraint,
        StructuredOutputConstraint::Grammar("root ::= \"a\" | \"b\"".to_string())
    );
}

/// A JSON schema with a feature xgrammar lacks (vLLM's own static check)
/// is refused before it reaches the engine wherever the resolved backend
/// is xgrammar, `auto` included, as vLLM's frontend refuses it under
/// `backend=xgrammar`: xgrammar would compile it with the keyword dropped
/// and the engine would answer off the schema. A pinned guidance enforces
/// such a schema and takes it as is. The tool parameters of a forced tool
/// call reach this translation as the same `JsonSchema` constraint (a
/// parser without structural tags builds one from the tools), or inside
/// a `StructuralTag` (a parser with structural tags embeds each tool's
/// parameters as a `json_schema` element), so a tool whose parameters use
/// such a keyword is refused the same way on both; the message names the
/// keyword and where it stands.
#[test]
fn a_schema_xgrammar_cannot_enforce_is_refused_wherever_xgrammar_is_the_backend() {
    use engine_zmq_client::protocol::vllm::structured_outputs::StructuredOutputConstraint;
    use StructuredOutputsBackendConfig::{Auto, Pinned};
    let schema = |text: &str| {
        tokenized_req(vllm::SamplingParams {
            constraint: Some(vllm::sampling_params::Constraint::JsonSchema(
                text.to_string(),
            )),
            ..Default::default()
        })
    };
    let translate = |request, config| {
        translate_request_with_backend(
            request,
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
            config,
        )
    };
    let multiple_of = r#"{"type":"integer","multipleOf":3}"#;
    for config in [Auto, Pinned(StructuredOutputBackend::Xgrammar)] {
        let refused = translate(schema(multiple_of), config).expect_err("refused");
        assert!(
            refused.contains("not supported by xgrammar"),
            "{config:?}: {refused}"
        );
    }
    // The parameters of a forced tool call, as a tool-call parser without
    // structural tags sends them: an array parameter with `uniqueItems`.
    let tool_parameters = r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"required":["tags"]}"#;
    let refused = translate(schema(tool_parameters), Auto).expect_err("refused");
    assert!(
        refused.contains("tool parameters of a forced tool call")
            && refused.contains("`uniqueItems` at `/properties/tags`"),
        "{refused}"
    );
    let refused = translate(schema(multiple_of), Auto).expect_err("refused");
    assert!(refused.contains("`multipleOf` at its root"), "{refused}");
    // The same parameters inside a forced call's structural tag, as a
    // parser with structural tags sends them (one `tag` per tool, the
    // parameters its `json_schema` content): refused under `auto` and a
    // pinned xgrammar, the message naming the keyword and where it
    // stands in the tag; a pinned guidance takes the tag as it is; a tag
    // whose schemas xgrammar compiles whole passes, stamped xgrammar.
    let tag = |parameters: serde_json::Value| {
        let tag = serde_json::json!({"format": {"type": "triggered_tags",
            "triggers": ["<tool_call>"], "tags": [
                {"begin": "<tool_call>{\"name\": \"plain\", \"arguments\": ",
                 "content": {"type": "json_schema", "json_schema":
                    {"type": "object", "properties": {"city": {"type": "string"}}}},
                 "end": "}</tool_call>"},
                {"begin": "<tool_call>{\"name\": \"tagged\", \"arguments\": ",
                 "content": {"type": "json_schema", "json_schema": parameters},
                 "end": "}</tool_call>"},
            ], "at_least_one": true}});
        tokenized_req(vllm::SamplingParams {
            constraint: Some(vllm::sampling_params::Constraint::StructuralTag(
                tag.to_string(),
            )),
            ..Default::default()
        })
    };
    let unique_items: serde_json::Value = serde_json::from_str(tool_parameters).unwrap();
    for config in [Auto, Pinned(StructuredOutputBackend::Xgrammar)] {
        let refused = translate(tag(unique_items.clone()), config).expect_err("refused");
        assert!(
            refused.contains("structural tag constraint")
                && refused.contains(
                    "`uniqueItems` at `/format/tags/1/content/json_schema/properties/tags` \
                     (inside the tag that begins `<tool_call>{\"name\": \"tagged\", \
                     \"arguments\": `)"
                )
                && refused.contains("not supported by xgrammar"),
            "{config:?}: {refused}"
        );
    }
    let params = translate(tag(unique_items), Pinned(StructuredOutputBackend::Guidance))
        .expect("translated")
        .sampling_params
        .and_then(|sp| sp.structured_outputs)
        .expect("structured outputs");
    assert_eq!(params.backend, StructuredOutputBackend::Guidance);
    assert!(matches!(
        params.constraint,
        StructuredOutputConstraint::StructuralTag(_)
    ));
    let params = translate(
        tag(serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}})),
        Auto,
    )
    .expect("translated")
    .sampling_params
    .and_then(|sp| sp.structured_outputs)
    .expect("structured outputs");
    assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
    assert!(matches!(
        params.constraint,
        StructuredOutputConstraint::StructuralTag(_)
    ));
    let params = translate(
        schema(multiple_of),
        Pinned(StructuredOutputBackend::Guidance),
    )
    .expect("translated")
    .sampling_params
    .and_then(|sp| sp.structured_outputs)
    .expect("structured outputs");
    assert_eq!(params.backend, StructuredOutputBackend::Guidance);
    assert_eq!(
        params.constraint,
        StructuredOutputConstraint::Json(serde_json::json!({"type": "integer", "multipleOf": 3}))
    );
    // A schema xgrammar compiles whole passes under `auto`.
    let params = translate(
        schema(r#"{"type":"object","properties":{"city":{"type":"string"}}}"#),
        Auto,
    )
    .expect("translated")
    .sampling_params
    .and_then(|sp| sp.structured_outputs)
    .expect("structured outputs");
    assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
}

#[test]
fn spec_decode_counts_land_on_the_complete() {
    let metrics = SpecDecodeMetrics {
        num_spec_tokens: 3,
        histogram: vec![1, 2, 0, 1],
        num_draft_tokens: 9,
        ..Default::default()
    };
    // 0*1 + 1*2 + 2*0 + 3*1 accepted draft tokens, as the Python servicer sums.
    assert_eq!(metrics.accepted_tokens(), 5);
    let mut response = vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Complete(
            vllm::GenerateComplete::default(),
        )),
    };
    let mut pending = None;
    attach_spec_decode_counts(&mut response, &mut pending, Some(metrics));
    let Some(vllm::generate_response::Response::Complete(complete)) = response.response else {
        panic!("complete");
    };
    assert_eq!(complete.spec_accepted_tokens, 5);
    assert_eq!(complete.spec_draft_tokens, 9);
}

#[test]
fn ranked_candidate_count_maps_the_sentinels() {
    assert_eq!(ranked_candidate_count(None), 0);
    assert_eq!(ranked_candidate_count(Some(0)), 0);
    assert_eq!(ranked_candidate_count(Some(3)), 3);
    // -1 asks for every candidate the engine returned.
    assert_eq!(ranked_candidate_count(Some(-1)), usize::MAX);
}

/// Prompt bounds apply before engine admission, with or without an output limit.
#[test]
fn vllm_validates_prompt_length_even_with_explicit_max_tokens() {
    for sampling in [
        None,
        Some(vllm::SamplingParams::default()),
        Some(vllm::SamplingParams {
            max_tokens: Some(16_384),
            ..Default::default()
        }),
    ] {
        for (prompt_len, max_model_len, accepted) in [
            (0, 4, false),
            (3, 4, true),
            (4, 4, false),
            (5, 4, false),
            (1_049_692, 1_048_576, false),
            // An engine that reported no window bounds nothing but
            // emptiness (unreachable from vLLM, which always reports one).
            (1_049_692, 0, true),
            (0, 0, false),
        ] {
            let mut req = tokenized_req(vllm::SamplingParams::default());
            req.sampling_params = sampling.clone();
            req.input = Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    input_ids: vec![1; prompt_len],
                    ..Default::default()
                },
            ));
            let result = translate_request(
                req,
                max_model_len,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
            );
            assert_eq!(
                result.is_ok(),
                accepted,
                "prompt_len={prompt_len}, max_model_len={max_model_len}, max_tokens={:?}",
                sampling.as_ref().and_then(|sp| sp.max_tokens)
            );
            match result {
                Err(error) => assert!(error.contains("prompt"), "{error}"),
                Ok(request) => {
                    // The request's own limit is kept; an unset one is the
                    // window's remainder, and no bound under no window.
                    if let Some(sp) = sampling.as_ref() {
                        let remainder = if max_model_len == 0 {
                            u32::MAX
                        } else {
                            u32::try_from(max_model_len - prompt_len as u64).unwrap()
                        };
                        assert_eq!(
                            request.sampling_params.expect("params").max_tokens,
                            sp.max_tokens.unwrap_or(remainder)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn vllm_defaults_unset_max_tokens_to_remaining_context() {
    let max_tokens = |sampling, max_model_len| {
        translate_request(
            tokenized_req(sampling),
            max_model_len,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("request translated")
        .sampling_params
        .expect("sampling params present")
        .max_tokens
    };

    // Unset max_tokens defaults to `max_model_len - prompt_len` (prompt is
    // 3 tokens), mirroring vLLM's bypassed OpenAI frontend.
    assert_eq!(max_tokens(vllm::SamplingParams::default(), 100), 97);
    // An explicit value is always honored.
    assert_eq!(
        max_tokens(
            vllm::SamplingParams {
                max_tokens: Some(8),
                ..Default::default()
            },
            100,
        ),
        8,
    );
}

#[test]
fn vllm_attaches_eos_stop_ids() {
    let eos = EosTokenIds::new(Some(5), vec![7]);
    let sampling = |sp| {
        translate_request(tokenized_req(sp), 4096, ModelDtype::BFloat16, &eos)
            .expect("request translated")
            .sampling_params
            .expect("sampling params present")
    };

    // Primary rides `_eos_token_id`, extras merge into `stop_token_ids`
    // without duplicating, and the union lands in `_all_stop_token_ids`.
    let sp = sampling(vllm::SamplingParams {
        stop_token_ids: vec![7, 9],
        ..Default::default()
    });
    assert_eq!(sp.eos_token_id, Some(5));
    assert_eq!(sp.stop_token_ids, vec![7, 9]);
    assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));

    // ignore_eos drops the EOS stops from the wire but keeps the
    // bookkeeping set (mirrors the reference frontend).
    let sp = sampling(vllm::SamplingParams {
        stop_token_ids: vec![9],
        ignore_eos: true,
        ..Default::default()
    });
    assert_eq!(sp.eos_token_id, None);
    assert_eq!(sp.stop_token_ids, vec![9]);
    assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));
}

#[test]
fn vllm_translates_structured_output_constraints() {
    use engine_zmq_client::protocol::vllm::structured_outputs::{
        StructuredOutputBackend, StructuredOutputConstraint,
    };

    let translate = |constraint| {
        translate_request(
            tokenized_req(vllm::SamplingParams {
                constraint: Some(constraint),
                ..Default::default()
            }),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("constraint translated")
        .sampling_params
        .expect("sampling params present")
        .structured_outputs
    };

    // Each constraint mode maps onto its typed counterpart, under the
    // engine's `auto` backend when nothing configured one.
    let json_object = translate(vllm::sampling_params::Constraint::JsonObject(true))
        .expect("json_object translated");
    assert_eq!(
        json_object.constraint,
        StructuredOutputConstraint::JsonObject
    );
    assert_eq!(json_object.backend, StructuredOutputBackend::Xgrammar);

    let regex = translate(vllm::sampling_params::Constraint::Regex("a.*".to_string()))
        .expect("regex translated");
    assert_eq!(
        regex.constraint,
        StructuredOutputConstraint::Regex("a.*".to_string())
    );

    // A choice is lowered to the grammar xgrammar compiles, as vLLM's
    // frontend does before it sends one there.
    let choice = translate(vllm::sampling_params::Constraint::Choice(
        vllm::ChoiceConstraint {
            choices: vec!["yes".to_string(), "no".to_string()],
        },
    ))
    .expect("choice translated");
    assert_eq!(
        choice.constraint,
        StructuredOutputConstraint::Grammar("root ::= \"yes\" | \"no\"".to_string())
    );

    // A JSON schema string is parsed to preserve object shape.
    let json = translate(vllm::sampling_params::Constraint::JsonSchema(
        r#"{"type":"object"}"#.to_string(),
    ))
    .expect("json schema translated");
    assert_eq!(
        json.constraint,
        StructuredOutputConstraint::Json(serde_json::json!({"type": "object"}))
    );

    // json_object=false means the caller opted out: no constraint.
    assert!(translate(vllm::sampling_params::Constraint::JsonObject(false)).is_none());
}

/// n=3 fans out into 3 single-sample wire requests with unique sub-rids.
/// An explicit seed derives per-sub seeds (`seed + i`) so the samples
/// differ deterministically; no seed stays `None` per sub (the engine
/// seeds each rid independently).
#[test]
fn fan_out_splits_n_into_single_sample_requests() {
    let mut req = tokenized_req(vllm::SamplingParams {
        n: 3,
        seed: Some(7),
        ..Default::default()
    });
    req.request_id = "r1".to_string();

    let subs = fan_out_requests(req);
    assert_eq!(subs.len(), 3);
    let rids: Vec<&str> = subs.iter().map(|sub| sub.request_id.as_str()).collect();
    assert_eq!(rids, vec!["r1-0", "r1-1", "r1-2"]);
    for (i, sub) in subs.iter().enumerate() {
        let sp = sub.sampling_params.as_ref().unwrap();
        assert_eq!(sp.n, 1);
        assert_eq!(sp.seed, Some(7 + i as i32));
        // Everything else is shared verbatim.
        assert_eq!(
            sub.input,
            Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                }
            ))
        );
    }

    // No explicit seed: every sub keeps None.
    let subs = fan_out_requests(tokenized_req(vllm::SamplingParams {
        n: 2,
        ..Default::default()
    }));
    assert!(subs
        .iter()
        .all(|sub| sub.sampling_params.as_ref().unwrap().seed.is_none()));

    // n<=1 passes through untouched (rid keeps its original form).
    let single = fan_out_requests(tokenized_req(vllm::SamplingParams::default()));
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].request_id, "r1");
}

/// n=2 over a vLLM EngineCore: two engine-side requests with distinct
/// sub-rids, and the merged stream yields two `Complete`s tagged with the
/// proto `index` (0 and 1) the pipeline demuxes choices by.
#[tokio::test]
async fn generate_e2e_fans_out_n2_vllm() {
    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;

    #[expect(
        clippy::disallowed_methods,
        reason = "engine task ends after responding"
    )]
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let mut rids = Vec::new();
        for _ in 0..2 {
            let request = match input.recv().await.unwrap() {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            // Each sub is a single-sample request with a derived seed.
            let sp = request.sampling_params.as_ref().unwrap();
            assert_eq!(sp.max_tokens, 4);
            rids.push((request.request_id.clone(), sp.seed));
        }
        assert_eq!(
            rids,
            vec![("r1-0".to_string(), Some(5)), ("r1-1".to_string(), Some(6))],
            "sub-rids must be unique and seeds derived per sub"
        );
        output
            .send_outputs(&batch("r1-0", 10, None, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        output
            .send_outputs(&batch("r1-1", 11, None, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        // Keep the peer alive until the frontend has consumed the replies.
        (input, output)
    });

    let mut req = tokenized_req(vllm::SamplingParams {
        n: 2,
        seed: Some(5),
        max_tokens: Some(4),
        ..Default::default()
    });
    req.request_id = "r1".to_string();
    let mut stream = client.generate_vllm(req).await.expect("generate");

    let mut completes = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item").response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                completes.push(complete);
            }
            Some(vllm::generate_response::Response::Chunk(_)) | None => {}
        }
    }
    completes.sort_by_key(|complete| complete.index);
    assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
    assert_eq!(completes[0].index, 0);
    assert_eq!(completes[0].output_ids, vec![10]);
    assert_eq!(completes[1].index, 1);
    assert_eq!(completes[1].output_ids, vec![11]);

    engine_task.await.unwrap();
}

/// Dropping the merged stream before completion aborts EVERY fanned-out
/// engine-side sub-request, not just one.
#[tokio::test]
async fn dropping_fanned_out_stream_aborts_all_subs() {
    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;
    let (mut engine_input, _engine_output) = engine.split();

    let stream = client
        .generate_vllm(tokenized_req(vllm::SamplingParams {
            n: 2,
            ..Default::default()
        }))
        .await
        .expect("generate");

    // Consume both Adds first so the drop-triggered aborts are the next
    // inbound messages.
    for _ in 0..2 {
        match engine_input.recv().await.unwrap() {
            EngineInbound::Add(_) => {}
            other => panic!("expected Add, got {other:?}"),
        }
    }

    drop(stream); // unfinished -> every sub auto-aborts

    let mut aborted = BTreeSet::new();
    while aborted.len() < 2 {
        match engine_input.recv().await.unwrap() {
            EngineInbound::Abort(rids) => aborted.extend(rids),
            other => panic!("expected Abort, got {other:?}"),
        }
    }
    assert_eq!(
        aborted,
        BTreeSet::from(["r1-0".to_string(), "r1-1".to_string()])
    );
}

fn sampling_request(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
    vllm::GenerateRequest {
        request_id: "s".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(sampling),
        ..Default::default()
    }
}

/// Proto "unset" zeros for `top_p`/`repetition_penalty` mean disabled, as
/// the Python servicer translates them; the engine rejects literal zeros.
#[test]
fn translate_normalizes_unset_top_p_and_repetition_penalty() {
    let request = translate_request(
        sampling_request(vllm::SamplingParams::default()),
        1024,
        ModelDtype::Float32,
        &EosTokenIds::default(),
    )
    .expect("defaults translate");
    let sampling = request.sampling_params.expect("sampling params");
    assert_eq!(sampling.top_p, 1.0);
    assert_eq!(sampling.repetition_penalty, 1.0);
    assert_eq!(sampling.temperature, 1.0);
    // Unset max_tokens defaults to the remaining context.
    assert_eq!(sampling.max_tokens, 1024 - 3);
}

/// vLLM's parameter checks run before submit: a request the engine would
/// reject on its input thread (killing it) is an invalid-argument error.
#[test]
fn translate_rejects_sampling_the_engine_would_refuse() {
    let cases: Vec<(&str, vllm::SamplingParams)> = vec![
        (
            "temperature",
            vllm::SamplingParams {
                temperature: Some(-0.5),
                ..Default::default()
            },
        ),
        (
            "top_p",
            vllm::SamplingParams {
                top_p: 1.5,
                ..Default::default()
            },
        ),
        (
            "min_p",
            vllm::SamplingParams {
                min_p: 2.0,
                ..Default::default()
            },
        ),
        (
            "presence_penalty",
            vllm::SamplingParams {
                presence_penalty: 3.0,
                ..Default::default()
            },
        ),
        (
            "repetition_penalty",
            vllm::SamplingParams {
                repetition_penalty: -1.0,
                ..Default::default()
            },
        ),
        (
            "max_tokens",
            vllm::SamplingParams {
                max_tokens: Some(0),
                ..Default::default()
            },
        ),
        (
            "min_tokens",
            vllm::SamplingParams {
                max_tokens: Some(4),
                min_tokens: 8,
                ..Default::default()
            },
        ),
        (
            "logprobs",
            vllm::SamplingParams {
                logprobs: Some(-2),
                ..Default::default()
            },
        ),
        // vLLM's `_verify_greedy_sampling`: no distinct choices at temperature 0.
        (
            "n must be 1 when using greedy sampling, got 2.",
            vllm::SamplingParams {
                temperature: Some(0.0),
                n: 2,
                ..Default::default()
            },
        ),
    ];
    for (field, sampling) in cases {
        let error = translate_request(
            sampling_request(sampling),
            1024,
            ModelDtype::Float32,
            &EosTokenIds::default(),
        )
        .expect_err(field);
        assert!(error.contains(field), "{field}: {error}");
    }
}

/// This wire's engine names no cause: an `Error` finish on a constrained
/// request is as often a decode side's failed KV load (the request
/// carries `kv_transfer_params`) as a grammar rejection, and an `Abort` the
/// frontend did not ask for is a pause or a shutdown. Both keep the shape
/// they have on every request: the `Error` is the engine's own report (the
/// internal status the router retries, charges and probes), the `Abort` a
/// `Complete` with finish reason `abort`; neither is the constraint-failure
/// status, which only an engine's own attribution raises.
#[tokio::test]
async fn an_error_or_abort_finish_of_a_constrained_request_keeps_the_engines_own_shape() {
    use smg_grpc_client::is_constraint_failure;

    let (client, engine, _dir) = connected_pair(RuntimeType::Vllm).await;
    // (rid, the finish, constrained, a decode side carrying its KV handoff)
    let cases = [
        ("r1", EngineCoreFinishReason::Error, true, true),
        ("r2", EngineCoreFinishReason::Error, true, false),
        ("r3", EngineCoreFinishReason::Abort, true, false),
        ("r4", EngineCoreFinishReason::Error, false, false),
        ("r5", EngineCoreFinishReason::Abort, false, false),
    ];

    #[expect(
        clippy::disallowed_methods,
        reason = "engine task ends after responding"
    )]
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        for (rid, finish, _, _) in cases {
            let request = loop {
                match input.recv().await.unwrap() {
                    EngineInbound::Add(request) => break request,
                    // A dropped stream's abort of a request the engine
                    // already finished.
                    EngineInbound::Abort(_) => continue,
                    other => panic!("expected Add, got {other:?}"),
                }
            };
            assert_eq!(request.request_id, rid);
            output
                .send_outputs(&batch(rid, 10, None, Some(finish)))
                .await
                .unwrap();
        }
        // Keep the peer alive until the frontend has consumed the replies.
        (input, output)
    });

    for (rid, finish, constrained, decode_side) in cases {
        let req = vllm::GenerateRequest {
            request_id: rid.to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(8),
                constraint: constrained.then(|| {
                    vllm::sampling_params::Constraint::JsonSchema(
                        r#"{"type":"object"}"#.to_string(),
                    )
                }),
                ..Default::default()
            }),
            kv_transfer_params_json: decode_side.then(|| {
                r#"{"do_remote_prefill":true,"remote_host":"prefill","remote_port":5600}"#
                    .to_string()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_vllm(req).await.expect("generate");
        let item = stream.next().await.expect("an item");
        match finish {
            EngineCoreFinishReason::Error => {
                let error = item.expect_err("an error");
                assert_eq!(error.code(), tonic::Code::Internal, "{rid}");
                assert_eq!(
                    error.message(),
                    "engine finished the request with an error (see engine logs)",
                    "{rid}: the engine's own report, word for word"
                );
                assert!(!is_constraint_failure(&error), "{rid}: {error}");
            }
            _ => {
                // The tick's token goes out as a chunk; the parked
                // `Complete` keeps the abort finish.
                assert!(
                    matches!(
                        item.expect("a chunk").response,
                        Some(vllm::generate_response::Response::Chunk(_))
                    ),
                    "{rid}"
                );
                let complete = stream.next().await.expect("an item").expect("a response");
                match complete.response {
                    Some(vllm::generate_response::Response::Complete(complete)) => {
                        assert_eq!(complete.finish_reason, "abort", "{rid}");
                    }
                    other => panic!("{rid}: expected a Complete, got {other:?}"),
                }
            }
        }
    }
    engine_task.await.unwrap();
}
