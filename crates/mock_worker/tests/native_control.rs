//! The pinned native client controls the existing engine without advertising inference.
use std::{sync::Arc, time::Duration};

use mock_worker::{
    config::Config, engine::EngineParams, grpc::serve_with_listener_until, overrides,
};
use smg_grpc_client::tokenspeed_proto as ts;
use tokio::{net::TcpListener, sync::oneshot, time::timeout};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{health_client::HealthClient, HealthCheckRequest};
use vllm_proto::{self as v, control_client::ControlClient};

#[tokio::test]
async fn native_control_uses_shared_state_and_rejects_unimplemented_capabilities() {
    timeout(Duration::from_secs(15), async {
        for realistic in [false, true] {
            let cfg = Config {
                realistic,
                grpc_native_vllm: true,
                model_id: "shared-native".into(),
                context_length: 4096,
                engine: EngineParams {
                    block_size: 8,
                    kv_capacity_tokens: 81,
                    max_running: 3,
                    max_batched_tokens: 32,
                    ..Default::default()
                },
                ..Config::default()
            };
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = oneshot::channel();
            #[expect(
                clippy::disallowed_methods,
                reason = "the test joins the server after shutdown"
            )]
            let server = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
                let _ = stopped.await;
            }));
            let channel = Channel::from_shared(format!("http://{address}"))
                .unwrap()
                .connect()
                .await
                .unwrap();
            let mut native = ControlClient::new(channel.clone());
            let mut scheduler =
                ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel.clone());
            let mut health = HealthClient::new(channel.clone());
            let info = native
                .get_server_info(v::GetServerInfoRequest {})
                .await
                .unwrap()
                .into_inner();
            assert_eq!(info.proto_version, "0.6.0");
            assert!(info.engine_version.starts_with("mock-worker/"));
            assert_eq!(info.max_model_len, 4096);
            assert_eq!(
                (
                    info.kv_block_size,
                    info.total_kv_blocks,
                    info.max_running_requests,
                    info.max_batched_tokens
                ),
                if realistic {
                    (8, 10, 3, 32)
                } else {
                    (0, 0, 0, 0)
                }
            );
            let parallelism = info.parallelism.unwrap();
            assert_eq!(
                (
                    parallelism.data_parallel_size,
                    parallelism.data_parallel_rank,
                    parallelism.world_size
                ),
                (1, 0, 1)
            );
            assert_eq!(info.max_loras, 0);
            assert!(info.rl_capabilities.is_none());
            let model = native
                .get_model_info(v::GetModelInfoRequest {})
                .await
                .unwrap()
                .into_inner();
            assert_eq!(model.model_id, "shared-native");
            assert_eq!(
                model.served_model_name,
                scheduler
                    .get_model_info(ts::GetModelInfoRequest {})
                    .await
                    .unwrap()
                    .into_inner()
                    .served_model_name
            );
            assert!(
                !model.supports_text_input
                    && !model.supports_token_ids_input
                    && !model.supports_multimodal
                    && !model.supports_lora
            );
            let check = HealthCheckRequest {
                service: "vllm.Control".into(),
            };
            assert_eq!(
                health
                    .check(check.clone())
                    .await
                    .unwrap()
                    .into_inner()
                    .status,
                1
            );
            let serving = overrides::for_port(address.port());
            struct RestoreServing(Arc<overrides::WorkerOverrides>);
            impl Drop for RestoreServing {
                fn drop(&mut self) {
                    self.0.set_serving(true);
                }
            }
            let _restore = RestoreServing(Arc::clone(&serving));
            serving.set_serving(false);
            assert_eq!(
                health
                    .check(check.clone())
                    .await
                    .unwrap()
                    .into_inner()
                    .status,
                2
            );
            serving.set_serving(true);
            assert_eq!(
                health
                    .check(HealthCheckRequest {
                        service: "vllm.Inference".into()
                    })
                    .await
                    .unwrap_err()
                    .code(),
                Code::NotFound
            );
            let mut inference = v::inference_client::InferenceClient::new(channel);
            assert_eq!(
                inference
                    .generate(v::GenerateRequest::default())
                    .await
                    .unwrap_err()
                    .code(),
                Code::Unimplemented
            );
            assert_eq!(
                native
                    .shutdown(v::ShutdownRequest {})
                    .await
                    .unwrap_err()
                    .code(),
                Code::FailedPrecondition
            );
            assert_eq!(
                native
                    .get_kv_event_sources(v::GetKvEventSourcesRequest {})
                    .await
                    .unwrap_err()
                    .code(),
                Code::Unimplemented
            );
            assert_eq!(
                native
                    .list_loras(v::ListLorasRequest {})
                    .await
                    .unwrap_err()
                    .code(),
                Code::Unimplemented
            );
            assert_eq!(
                native
                    .is_paused(v::IsPausedRequest {})
                    .await
                    .unwrap_err()
                    .code(),
                Code::Unimplemented
            );
            for ids in [
                vec!["".into()],
                vec!["x".repeat(257)],
                vec!["x".into(); 1025],
            ] {
                assert_eq!(
                    native
                        .abort(v::AbortRequest { request_ids: ids })
                        .await
                        .unwrap_err()
                        .code(),
                    Code::InvalidArgument
                );
            }
            if realistic {
                let mut output = scheduler
                    .generate(ts::GenerateRequest {
                        request_id: "native-abort".into(),
                        stream: true,
                        tokenized: Some(ts::TokenizedInput {
                            input_ids: vec![100; 16],
                            ..Default::default()
                        }),
                        sampling_params: Some(ts::SamplingParams {
                            max_new_tokens: Some(4096),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .await
                    .unwrap()
                    .into_inner();
                assert!(output.message().await.unwrap().is_some());
                native
                    .abort(v::AbortRequest {
                        request_ids: vec![
                            "native-abort".into(),
                            "missing".into(),
                            "native-abort".into(),
                        ],
                    })
                    .await
                    .unwrap();
                loop {
                    let frame = output
                        .message()
                        .await
                        .unwrap()
                        .expect("terminal response before EOF");
                    if let Some(ts::generate_response::Response::Complete(done)) = frame.response {
                        assert_eq!(done.finish_reason, "abort");
                        break;
                    }
                }
                assert!(output.message().await.unwrap().is_none());
            } else {
                assert_eq!(
                    native
                        .abort(v::AbortRequest {
                            request_ids: vec!["x".into()]
                        })
                        .await
                        .unwrap_err()
                        .code(),
                    Code::FailedPrecondition
                );
            }
            let mut watch = health.watch(check).await.unwrap().into_inner();
            assert_eq!(watch.message().await.unwrap().unwrap().status, 1);
            stop.send(()).unwrap();
            assert_eq!(watch.message().await.unwrap().unwrap().status, 2);
            assert!(watch.message().await.unwrap().is_none());
            server.await.unwrap();
        }
    })
    .await
    .expect("native control and shutdown finish within the bound");
}
