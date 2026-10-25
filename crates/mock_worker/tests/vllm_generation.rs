//! The actual SMG client must see the shared engine's tokens and cancellation.
use std::{sync::Arc, time::Duration};

use futures::StreamExt;
use mock_worker::{
    config::{Config, ModelTokens},
    grpc::serve_with_listener_until,
};
use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as v, VllmEngineClient};
use tokio::{
    net::TcpListener,
    sync::oneshot,
    time::{sleep, timeout},
};
use tonic::Code;

fn request(id: &str, stream: bool, max: u32) -> v::GenerateRequest {
    let body = serde_json::from_value(serde_json::json!({
        "model": "mock", "prompt": "fixture", "temperature": 0.0,
        "max_tokens": max, "stream": stream
    }))
    .unwrap();
    VllmEngineClient::build_generate_request_from_completion(
        id.into(),
        &body,
        "fixture".into(),
        vec![100; 64],
    )
    .unwrap()
}

async fn completion(client: &VllmEngineClient, req: v::GenerateRequest) -> v::GenerateComplete {
    let streaming = req.stream;
    let mut output = client.generate(req).await.unwrap();
    let mut ids = Vec::new();
    let mut complete = None;
    while let Some(frame) = output.next().await {
        assert!(complete.is_none(), "no output may follow a terminal");
        match frame.unwrap().response.unwrap() {
            v::generate_response::Response::Chunk(chunk) => {
                assert!(streaming);
                ids.extend(chunk.token_ids);
                assert_eq!(chunk.completion_tokens as usize, ids.len());
                assert_eq!(chunk.prompt_tokens, 64);
            }
            v::generate_response::Response::Complete(done) => {
                if streaming {
                    assert_eq!(done.output_ids, ids);
                }
                assert_eq!(done.completion_tokens as usize, done.output_ids.len());
                assert_eq!(done.prompt_tokens, 64);
                complete = Some(done);
            }
        }
    }
    output.mark_completed();
    complete.expect("a terminal response is required")
}

#[tokio::test]
async fn vllm_generation_preserves_tokens_cache_and_multiplexed_cancellation() {
    timeout(Duration::from_secs(15), async {
        let cfg = Config {
            realistic: true,
            grpc_vllm: true,
            context_length: 8192,
            ..Config::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins its server after shutdown"
        )]
        let server = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
            let _ = stopped.await;
        }));
        let client = VllmEngineClient::connect(&endpoint).await.unwrap();
        let cold = completion(&client, request("same", true, 4)).await;
        assert_eq!((cold.completion_tokens, cold.cached_tokens), (4, 0));
        assert_eq!(cold.finish_reason, "length");
        let warm = completion(&client, request("same", false, 4)).await;
        assert_eq!(warm.output_ids, cold.output_ids);
        assert!(warm.cached_tokens > 0 && warm.cached_tokens <= 64);

        let mut scheduler =
            ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::connect(endpoint)
                .await
                .unwrap();
        let mut same = scheduler
            .generate(ts::GenerateRequest {
                request_id: "same".into(),
                tokenized: Some(ts::TokenizedInput {
                    input_ids: vec![100; 64],
                    ..Default::default()
                }),
                sampling_params: Some(ts::SamplingParams {
                    max_new_tokens: Some(4),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let Some(ts::generate_response::Response::Complete(done)) =
            same.message().await.unwrap().unwrap().response
        else {
            panic!("terminal expected");
        };
        assert_eq!(done.output_ids, cold.output_ids);
        assert_eq!((done.prompt_tokens, done.completion_tokens), (64, 4));
        assert!(same.message().await.unwrap().is_none());

        // A dropped request must release its work while the same channel remains usable.
        let mut cancelled = client
            .generate(request("cancelled", true, 4096))
            .await
            .unwrap();
        assert!(cancelled.next().await.unwrap().is_ok());
        let survivor = client.generate(request("survivor", true, 4)).await.unwrap();
        drop(cancelled);
        let mut survivor = survivor;
        let mut survived = false;
        while let Some(frame) = survivor.next().await {
            if let Some(v::generate_response::Response::Complete(done)) = frame.unwrap().response {
                assert_eq!(done.finish_reason, "length");
                assert_eq!(done.completion_tokens, 4);
                survived = true;
            }
        }
        survivor.mark_completed();
        assert!(survived);
        while client.get_server_info().await.unwrap().active_requests != 0 {
            sleep(Duration::from_millis(5)).await;
        }

        // Reject options before they reach the shared engine, including NaN controls.
        for case in 0..15 {
            let mut req = request("rejected", true, 4);
            let p = req.sampling_params.as_mut().unwrap();
            let code = match case {
                0 => {
                    req.request_id.clear();
                    Code::InvalidArgument
                }
                1 => {
                    req.input = Some(v::generate_request::Input::Text("text".into()));
                    Code::InvalidArgument
                }
                2 => {
                    req.input = Some(v::generate_request::Input::Tokenized(v::TokenizedInput {
                        input_ids: vec![32000],
                        ..Default::default()
                    }));
                    Code::InvalidArgument
                }
                3 => {
                    p.max_tokens = Some(0);
                    Code::InvalidArgument
                }
                4 => {
                    p.max_tokens = Some(8192);
                    Code::InvalidArgument
                }
                5 => {
                    p.temperature = Some(f32::NAN);
                    Code::Unimplemented
                }
                6 => {
                    p.n = 2;
                    Code::Unimplemented
                }
                7 => {
                    p.stop_token_ids = vec![101];
                    Code::Unimplemented
                }
                8 => {
                    p.seed = Some(1);
                    Code::Unimplemented
                }
                9 => {
                    p.logprobs = Some(0);
                    Code::Unimplemented
                }
                10 => {
                    p.constraint = Some(v::sampling_params::Constraint::JsonObject(true));
                    Code::Unimplemented
                }
                11 => {
                    p.truncate_prompt_tokens = Some(8);
                    Code::Unimplemented
                }
                12 => {
                    req.data_parallel_rank = Some(1);
                    Code::InvalidArgument
                }
                13 => {
                    req.kv_transfer_params_json = Some("{}".into());
                    Code::Unimplemented
                }
                _ => {
                    req.media_refs = Some(v::MediaRefs::default());
                    Code::Unimplemented
                }
            };
            match client.generate(req).await {
                Err(error) => assert_eq!(error.code(), code, "case {case}"),
                Ok(_) => panic!("case {case} accepted"),
            }
        }
        assert_eq!(client.get_server_info().await.unwrap().active_requests, 0);
        stop.send(()).unwrap();
        server.await.unwrap();
    })
    .await
    .expect("generation, rejection and shutdown remain bounded");
}

#[tokio::test]
async fn declared_model_tokens_are_shared_with_discovery_and_prompt_validation() {
    timeout(Duration::from_secs(15), async {
        let cfg = Config {
            realistic: true,
            grpc_vllm: true,
            context_length: 40960,
            model_tokens: ModelTokens::new(151936, vec![151645, 151643], 151643, 151643).unwrap(),
            ..Config::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins its server after shutdown"
        )]
        let server = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
            let _ = stopped.await;
        }));
        let client = VllmEngineClient::connect(&endpoint).await.unwrap();
        let info = client.get_model_info().await.unwrap();
        let mut scheduler =
            ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::connect(endpoint)
                .await
                .unwrap();
        let ts_info = scheduler
            .get_model_info(ts::GetModelInfoRequest {})
            .await
            .unwrap()
            .into_inner();
        assert_eq!((info.vocab_size, info.max_context_length), (151936, 40960));
        assert_eq!(info.vocab_size as i32, ts_info.vocab_size);
        assert_eq!(info.eos_token_ids, vec![151645, 151643]);
        assert_eq!(ts_info.eos_token_ids, vec![151645, 151643]);
        assert_eq!((info.bos_token_id, info.pad_token_id), (151643, 151643));
        assert_eq!(
            (ts_info.bos_token_id, ts_info.pad_token_id),
            (151643, 151643)
        );
        let mut high = request("expanded-vocabulary", false, 4);
        if let Some(v::generate_request::Input::Tokenized(input)) = &mut high.input {
            input.input_ids = vec![151644; 64];
        } else {
            panic!("the actual client builder must emit tokenized input");
        }
        let mut invalid = high.clone();
        if let Some(v::generate_request::Input::Tokenized(input)) = &mut invalid.input {
            input.input_ids[0] = info.vocab_size;
        }
        let done = completion(&client, high).await;
        assert_eq!(done.output_ids.len(), 4);
        assert!(done
            .output_ids
            .iter()
            .all(|id| (100..30100).contains(id) && !info.eos_token_ids.contains(id)));
        match client.generate(invalid).await {
            Err(error) => assert_eq!(error.code(), Code::InvalidArgument),
            Ok(_) => panic!("the exclusive vocabulary bound must reject the prompt"),
        }
        stop.send(()).unwrap();
        server.await.unwrap();
    })
    .await
    .expect("metadata, generation, rejection and shutdown remain bounded");
}
