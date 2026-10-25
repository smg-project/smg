//! Native unary/streaming wire semantics, cancellation and connection isolation.
use std::{sync::Arc, time::Duration};

use mock_worker::{
    config::Config,
    engine::{EngineParams, TimingModel},
    grpc::serve_with_listener_until,
};
use smg_grpc_client::tokenspeed_proto as ts;
use tokio::{net::TcpListener, sync::oneshot, time::timeout};
use tonic::{transport::Channel, Code, Request};
use vllm_proto::{self as v, control_client::ControlClient, inference_client::InferenceClient};

struct Worker {
    inference: InferenceClient<Channel>,
    control: ControlClient<Channel>,
    scheduler: ts::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl Worker {
    async fn start() -> Self {
        let cfg = Config {
            realistic: true,
            grpc_native_vllm: true,
            model_id: "native-fixture".into(),
            context_length: 4096,
            engine: EngineParams {
                block_size: 8,
                kv_capacity_tokens: 65536,
                timing: TimingModel::Linear {
                    prefill_tps: 1_000_000.0,
                    decode_base_ms: 5.0,
                    decode_per_req_ms: 0.0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "joined on shutdown and aborted on assertion failure"
        )]
        let task = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
            let _ = stopped.await;
        }));
        let channel = Channel::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        Self {
            inference: InferenceClient::new(channel.clone()),
            control: ControlClient::new(channel.clone()),
            scheduler: ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel),
            stop: Some(stop),
            task,
        }
    }
    async fn finish(mut self) {
        self.stop.take().unwrap().send(()).unwrap();
        (&mut self.task).await.unwrap();
    }
    async fn idle(&mut self) {
        loop {
            let loads = self
                .scheduler
                .get_loads(ts::GetLoadsRequest::default())
                .await
                .unwrap()
                .into_inner()
                .loads;
            if loads
                .iter()
                .all(|load| load.num_running_reqs == 0 && load.num_waiting_reqs == 0)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn request(id: &str, max_new_tokens: u32) -> v::GenerateRequest {
    v::GenerateRequest {
        request_id: id.into(),
        model: "native-fixture".into(),
        prompt: Some(v::generate_request::Prompt::TokenIds(v::TokenIds {
            ids: vec![100; 16],
        })),
        stopping: Some(v::StoppingCriteria {
            max_new_tokens,
            ..Default::default()
        }),
        response: Some(v::ResponseOptions {
            output_text: Some(false),
            output_token_ids: true,
            prompt_token_ids: true,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn native_unary_and_streaming_preserve_tokens_counts_and_cache_metadata() {
    timeout(Duration::from_secs(15), async {
        let mut worker = Worker::start().await;
        let unary = worker
            .inference
            .generate(request("parity", 17))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(unary.prompt_info.unwrap().token_ids, vec![100; 16]);
        let unary = unary.outputs.unwrap();
        assert_eq!(unary.num_tokens, 17);
        let finish = unary.finish_info.unwrap();
        assert_eq!(finish.num_output_tokens, 17);
        assert_eq!(finish.num_cached_tokens, Some(0));
        assert_eq!(
            finish.finish_reason,
            v::finish_info::FinishReason::Length as i32
        );
        let mut stream = worker
            .inference
            .generate_stream(request("parity", 17))
            .await
            .unwrap()
            .into_inner();
        let mut ids = Vec::new();
        let mut total = 0;
        let mut frames = 0;
        let mut done = false;
        while let Some(frame) = stream.message().await.unwrap() {
            assert!(!done, "no frames after terminal");
            if frames == 0 {
                assert_eq!(frame.prompt_info.unwrap().token_ids, vec![100; 16]);
            } else {
                assert!(frame.prompt_info.is_none());
            }
            let output = frame.outputs.unwrap();
            assert!(output.text.is_empty());
            assert_eq!(output.index, 0);
            assert_eq!(output.num_tokens as usize, output.token_ids.len());
            ids.extend(output.token_ids);
            total += output.num_tokens;
            frames += 1;
            if let Some(finish) = output.finish_info {
                assert_eq!(
                    output.num_tokens, 0,
                    "Complete must not repeat streamed IDs"
                );
                assert_eq!(finish.num_output_tokens, total);
                // A fully cached prompt recomputes its final block: one of two
                // eight-token blocks is reused, exactly as TokenSpeed reports.
                assert_eq!(finish.num_cached_tokens, Some(8));
                done = true;
            }
        }
        assert!(done);
        assert!(frames > 1);
        assert_eq!(total, 17);
        assert_eq!(ids, unary.token_ids);
        let mut core = worker
            .scheduler
            .generate(ts::GenerateRequest {
                request_id: "parity".into(),
                tokenized: Some(ts::TokenizedInput {
                    input_ids: vec![100; 16],
                    ..Default::default()
                }),
                sampling_params: Some(ts::SamplingParams {
                    max_new_tokens: Some(17),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let Some(ts::generate_response::Response::Complete(core_done)) =
            core.message().await.unwrap().unwrap().response
        else {
            panic!("non-streaming core request must return its completion");
        };
        assert_eq!(core_done.output_ids, unary.token_ids);
        assert_eq!(
            (
                core_done.prompt_tokens,
                core_done.completion_tokens,
                core_done.cached_tokens
            ),
            (16, 17, 8)
        );
        assert!(core.message().await.unwrap().is_none());
        let mut no_prompt = request("no-prompt-ids", 1);
        no_prompt.response.as_mut().unwrap().prompt_token_ids = false;
        let prompt = worker
            .inference
            .generate(no_prompt)
            .await
            .unwrap()
            .into_inner()
            .prompt_info
            .unwrap();
        assert_eq!(prompt.num_prompt_tokens, 16);
        assert!(prompt.token_ids.is_empty());
        worker.idle().await;
        worker.finish().await;
    })
    .await
    .expect("native parity fixture finishes within bound");
}

#[tokio::test]
async fn native_abort_and_drop_preserve_other_streams_on_the_same_connection() {
    timeout(Duration::from_secs(15), async {
        let mut worker = Worker::start().await;
        let mut aborted = worker
            .inference
            .generate_stream(request("abort-me", 2000))
            .await
            .unwrap()
            .into_inner();
        let first = aborted.message().await.unwrap().unwrap().outputs.unwrap();
        let mut total = first.num_tokens;
        assert!(first.finish_info.is_none());
        let mut survivor = worker
            .inference
            .generate_stream(request("survivor", 17))
            .await
            .unwrap()
            .into_inner();
        worker
            .control
            .abort(v::AbortRequest {
                request_ids: vec!["abort-me".into()],
            })
            .await
            .unwrap();
        let mut done = false;
        while let Some(frame) = aborted.message().await.unwrap() {
            let output = frame.outputs.unwrap();
            total += output.num_tokens;
            if let Some(finish) = output.finish_info {
                assert_eq!(
                    finish.finish_reason,
                    v::finish_info::FinishReason::Aborted as i32
                );
                assert_eq!(finish.num_output_tokens, total);
                done = true;
            }
        }
        assert!(done);
        assert!(total < 2000);
        let mut dropped = worker
            .inference
            .generate_stream(request("drop-me", 2000))
            .await
            .unwrap()
            .into_inner();
        assert!(dropped.message().await.unwrap().is_some());
        drop(dropped);
        let mut survivor_total = 0;
        let mut survivor_done = false;
        while let Some(frame) = survivor.message().await.unwrap() {
            let output = frame.outputs.unwrap();
            survivor_total += output.num_tokens;
            if let Some(finish) = output.finish_info {
                assert_eq!(
                    finish.finish_reason,
                    v::finish_info::FinishReason::Length as i32
                );
                assert_eq!(finish.num_output_tokens, 17);
                survivor_done = true;
            }
        }
        assert!(survivor_done);
        assert_eq!(survivor_total, 17);
        worker.idle().await;
        worker.finish().await;
    })
    .await
    .expect("cancellation, stream isolation and drain finish within bound");
}

#[tokio::test]
async fn native_rejects_default_text_output_and_invalid_rank_on_both_methods() {
    timeout(Duration::from_secs(10), async {
        let mut worker = Worker::start().await;
        for stream in [false, true] {
            for rank in ["1", "-1", "bad"] {
                let mut r = Request::new(request("bad-rank", 1));
                r.metadata_mut()
                    .insert("x-data-parallel-rank", rank.parse().unwrap());
                let code = if stream {
                    worker
                        .inference
                        .generate_stream(r)
                        .await
                        .err()
                        .unwrap()
                        .code()
                } else {
                    worker.inference.generate(r).await.err().unwrap().code()
                };
                assert_eq!(code, Code::InvalidArgument);
            }
            let mut r = request("default-text", 1);
            r.response = None;
            let code = if stream {
                worker
                    .inference
                    .generate_stream(r)
                    .await
                    .err()
                    .unwrap()
                    .code()
            } else {
                worker.inference.generate(r).await.err().unwrap().code()
            };
            assert_eq!(code, Code::Unimplemented);
        }
        let mut valid = Request::new(request("valid-rank", 1));
        valid
            .metadata_mut()
            .insert("x-data-parallel-rank", "0".parse().unwrap());
        assert_eq!(
            worker
                .inference
                .generate(valid)
                .await
                .unwrap()
                .into_inner()
                .outputs
                .unwrap()
                .num_tokens,
            1
        );
        worker.idle().await;
        worker.finish().await;
    })
    .await
    .expect("rejections finish within bound");
}
