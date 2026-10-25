//! Both gRPC control surfaces must observe the same simulated worker.
use std::{sync::Arc, time::Duration};

use mock_worker::{config::Config, grpc::serve_with_listener_until, overrides};
use smg_grpc_client::{
    common_proto as common, tokenspeed_proto as ts, vllm_proto as v, VllmEngineClient,
};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{health_client::HealthClient, HealthCheckRequest};

#[tokio::test]
async fn vllm_control_shares_health_load_cache_and_abort_with_tokenspeed() {
    timeout(Duration::from_secs(15), async {
        let cfg = Config {
            realistic: true,
            grpc_vllm: true,
            model_id: "shared-mock".into(),
            ..Config::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins this server after shutdown"
        )]
        let server = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
            let _ = stopped.await;
        }));
        let channel = Channel::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut vllm = v::vllm_engine_client::VllmEngineClient::new(channel.clone());
        let mut scheduler =
            ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel.clone());
        let mut health = HealthClient::new(channel);
        let service = "vllm.grpc.engine.VllmEngine";
        assert_eq!(
            health
                .check(HealthCheckRequest {
                    service: service.into()
                })
                .await
                .unwrap()
                .into_inner()
                .status,
            1
        );
        let serving = overrides::for_port(address.port());
        serving.set_serving(false);
        assert!(
            !vllm
                .health_check(v::HealthCheckRequest {})
                .await
                .unwrap()
                .into_inner()
                .healthy
        );
        assert_eq!(
            health
                .check(HealthCheckRequest {
                    service: service.into()
                })
                .await
                .unwrap()
                .into_inner()
                .status,
            2
        );
        serving.set_serving(true);
        let info = vllm
            .get_model_info(v::GetModelInfoRequest {})
            .await
            .unwrap()
            .into_inner();
        assert_eq!(info.served_model_name, "shared-mock");
        assert_eq!(info.vocab_size, 32000);
        for supported in [true, false] {
            serving.patch_info(
                serde_json::json!({"supports_vision": supported})
                    .as_object()
                    .unwrap(),
            );
            assert_eq!(
                vllm.get_model_info(v::GetModelInfoRequest {})
                    .await
                    .unwrap()
                    .into_inner()
                    .supports_vision,
                supported
            );
            assert_eq!(
                scheduler
                    .get_model_info(ts::GetModelInfoRequest {})
                    .await
                    .unwrap()
                    .into_inner()
                    .supports_vision,
                supported
            );
        }
        let info = vllm
            .get_server_info(v::GetServerInfoRequest {})
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            (info.data_parallel_size, info.block_size, info.max_num_seqs),
            (1, 16, 256)
        );
        // Unsupported capabilities must fail, never return fake successful outputs.
        assert_eq!(
            vllm.generate(v::GenerateRequest::default())
                .await
                .unwrap_err()
                .code(),
            Code::Unimplemented
        );
        assert_eq!(
            vllm.embed(v::EmbedRequest::default())
                .await
                .unwrap_err()
                .code(),
            Code::Unimplemented
        );
        assert_eq!(
            vllm.get_loads(v::GetLoadsRequest {
                dp_rank: Some(1),
                include: vec![]
            })
            .await
            .unwrap_err()
            .code(),
            Code::InvalidArgument
        );
        let mut events = vllm
            .subscribe_kv_events(common::SubscribeKvEventsRequest {
                load_records_before_first_batch: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let mut output = scheduler
            .generate(ts::GenerateRequest {
                request_id: "shared".into(),
                tokenized: Some(ts::TokenizedInput {
                    input_ids: vec![100; 64],
                    ..Default::default()
                }),
                sampling_params: Some(ts::SamplingParams {
                    max_new_tokens: Some(4096),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert!(output.message().await.unwrap().is_some());
        // Exercise the actual client with the same sections the gateway asks for.
        let client = VllmEngineClient::connect(&format!("http://{address}"))
            .await
            .unwrap();
        let load = client
            .get_loads(
                ["core", "disagg", "queues", "memory"]
                    .map(String::from)
                    .to_vec(),
            )
            .await
            .unwrap()
            .loads
            .remove(0);
        assert_eq!(load.num_running_reqs, 1);
        assert_eq!(
            (load.num_used_tokens, load.num_waiting_uncached_tokens),
            (0, 0)
        );
        assert_eq!(load.gen_throughput, 0.0);
        loop {
            let batch = events.message().await.unwrap().unwrap();
            assert_eq!(batch.load.as_ref().unwrap().gen_throughput, 0.0);
            if batch
                .events
                .iter()
                .any(|event| matches!(event.data, Some(common::kv_cache_event::Data::Stored(_))))
            {
                break;
            }
        }
        vllm.abort(v::AbortRequest {
            request_ids: vec!["shared".into(), "missing".into(), "shared".into()],
        })
        .await
        .unwrap();
        loop {
            let frame = output.message().await.unwrap().unwrap();
            if let Some(ts::generate_response::Response::Complete(done)) = frame.response {
                assert_eq!(done.finish_reason, "abort");
                break;
            }
        }
        assert!(output.message().await.unwrap().is_none());
        // Open subscriptions must not prevent the shared listener from draining.
        stop.send(()).unwrap();
        while events.message().await.unwrap().is_some() {}
        server.await.unwrap();
    })
    .await
    .expect("control-plane operations and shutdown complete within the bound");
}
