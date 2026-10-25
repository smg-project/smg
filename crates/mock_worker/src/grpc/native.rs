//! Native vLLM control discovery and cancellation for the same mock engine.
//! Inference supports a restricted token-ID-only profile on the same scheduler.

mod inference;
pub(super) use inference::NativeInferenceServer;
use vllm_proto::{
    self as v,
    control_server::{Control, ControlServer},
};

use super::*;

pub(super) type NativeServer = ControlServer<MockNative>;

#[derive(Clone)]
pub(super) struct MockNative(pub(super) MockScheduler);

#[tonic::async_trait]
impl Control for MockNative {
    async fn get_server_info(
        &self,
        _request: Request<v::GetServerInfoRequest>,
    ) -> Result<Response<v::ServerInfo>, Status> {
        let cfg = &self.0.cfg;
        let engine = self.0.engine.as_ref().map(|_| &cfg.engine);
        Ok(Response::new(v::ServerInfo {
            engine_version: concat!("mock-worker/", env!("CARGO_PKG_VERSION")).into(),
            api_version: "mock-control".into(),
            instance_id: format!("{}:{}", std::process::id(), self.0.name),
            parallelism: Some(v::ParallelismInfo {
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                data_parallel_size: 1,
                data_parallel_rank: 0,
                decode_context_parallel_size: 1,
                world_size: 1,
                data_parallel_size_local: 1,
            }),
            max_model_len: cfg.context_length.min(i32::MAX as u32),
            kv_block_size: engine.map_or(0, |p| p.block_size.max(1)),
            total_kv_blocks: engine.map_or(0, |p| p.capacity_blocks()),
            max_running_requests: engine.map_or(0, |p| p.max_running as u64),
            max_batched_tokens: engine.map_or(0, |p| u64::from(p.max_batched_tokens)),
            proto_version: v::VERSION.into(),
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<v::GetModelInfoRequest>,
    ) -> Result<Response<v::ModelInfo>, Status> {
        Ok(Response::new(v::ModelInfo {
            model_id: self.0.cfg.model_id.clone(),
            served_model_name: self.0.cfg.model_id.clone(),
            // Canned/scripted workers do not expose native inference.
            supports_token_ids_input: self.0.cfg.realistic && self.0.cfg.replay.script.is_none(),
            ..Default::default()
        }))
    }

    async fn abort(
        &self,
        request: Request<v::AbortRequest>,
    ) -> Result<Response<v::AbortResponse>, Status> {
        let ids = request.into_inner().request_ids;
        if ids.len() > 1024 || ids.iter().any(|id| id.is_empty() || id.len() > 256) {
            return Err(Status::invalid_argument(
                "abort accepts at most 1024 request IDs, each 1..=256 bytes",
            ));
        }
        self.0.engine.as_ref().ok_or_else(|| {
            Status::failed_precondition("native abort requires --engine realistic")
        })?;
        // Validate the entire batch before changing the shared engine. Unknown
        // and repeated IDs are harmless, matching the native cancellation API.
        for id in ids {
            self.0.abort_request(id);
        }
        Ok(Response::new(v::AbortResponse {}))
    }

    async fn shutdown(
        &self,
        _request: Request<v::ShutdownRequest>,
    ) -> Result<Response<v::ShutdownResponse>, Status> {
        Err(Status::failed_precondition(
            "this mock does not manage an engine process",
        ))
    }

    async fn load_lora(
        &self,
        _request: Request<v::LoadLoraRequest>,
    ) -> Result<Response<v::LoadLoraResponse>, Status> {
        Err(Status::unimplemented("native LoadLora is not implemented"))
    }

    async fn unload_lora(
        &self,
        _request: Request<v::UnloadLoraRequest>,
    ) -> Result<Response<v::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented(
            "native UnloadLora is not implemented",
        ))
    }

    async fn list_loras(
        &self,
        _request: Request<v::ListLorasRequest>,
    ) -> Result<Response<v::ListLorasResponse>, Status> {
        Err(Status::unimplemented("native ListLoras is not implemented"))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<v::GetKvEventSourcesRequest>,
    ) -> Result<Response<v::GetKvEventSourcesResponse>, Status> {
        if *self.0.stopped.borrow() {
            return Err(Status::unavailable("worker is stopping"));
        }
        let cfg = &self.0.cfg;
        if !cfg.realistic || !cfg.engine.prefix_cache || cfg.kv_events_zmq_base_port.is_none() {
            return Ok(Response::new(v::GetKvEventSourcesResponse {
                sources: vec![],
            }));
        }
        if cfg.kv_events_wire != crate::kv_zmq::Wire::Vllm {
            return Err(Status::failed_precondition(
                "native KV discovery requires the vllm event wire",
            ));
        }
        let source = self
            .0
            .kv_source
            .as_ref()
            .filter(|source| source.is_live())
            .ok_or_else(|| Status::unavailable("configured KV publisher is not live"))?;
        // The pinned zeromq revision 67dcba9 PUB fanout and per-peer queues each configure
        // 100000 entries. These are queue settings, not total-memory bounds.
        const PUB_QUEUE_CAPACITY: u32 = 100_000;
        Ok(Response::new(v::GetKvEventSourcesResponse {
            sources: vec![v::KvEventSource {
                transport: "zmq".into(),
                endpoint: source.endpoint.clone(),
                topic: source.cfg.topic.clone(),
                replay_endpoint: source.replay_endpoint.clone(),
                data_parallel_rank: Some(
                    u32::try_from(source.cfg.dp_rank).map_err(|_| {
                        Status::failed_precondition("KV rank cannot be represented")
                    })?,
                ),
                encoding: "msgpack".into(),
                schema_version: 1,
                buffer_steps: u32::try_from(source.cfg.buffer_steps.max(1))
                    .map_err(|_| Status::failed_precondition("KV buffer cannot be represented"))?,
                hwm: PUB_QUEUE_CAPACITY,
                max_queue_size: PUB_QUEUE_CAPACITY,
            }],
        }))
    }

    async fn pause_generation(
        &self,
        _request: Request<v::PauseGenerationRequest>,
    ) -> Result<Response<v::PauseGenerationResponse>, Status> {
        Err(Status::unimplemented(
            "native PauseGeneration is not implemented",
        ))
    }

    async fn resume_generation(
        &self,
        _request: Request<v::ResumeGenerationRequest>,
    ) -> Result<Response<v::ResumeGenerationResponse>, Status> {
        Err(Status::unimplemented(
            "native ResumeGeneration is not implemented",
        ))
    }

    async fn is_paused(
        &self,
        _request: Request<v::IsPausedRequest>,
    ) -> Result<Response<v::IsPausedResponse>, Status> {
        Err(Status::unimplemented("native IsPaused is not implemented"))
    }

    async fn sleep(
        &self,
        _request: Request<v::SleepRequest>,
    ) -> Result<Response<v::SleepResponse>, Status> {
        Err(Status::unimplemented("native Sleep is not implemented"))
    }

    async fn wake_up(
        &self,
        _request: Request<v::WakeUpRequest>,
    ) -> Result<Response<v::WakeUpResponse>, Status> {
        Err(Status::unimplemented("native WakeUp is not implemented"))
    }

    async fn is_sleeping(
        &self,
        _request: Request<v::IsSleepingRequest>,
    ) -> Result<Response<v::IsSleepingResponse>, Status> {
        Err(Status::unimplemented(
            "native IsSleeping is not implemented",
        ))
    }

    async fn init_weight_transfer_engine(
        &self,
        _request: Request<v::InitWeightTransferEngineRequest>,
    ) -> Result<Response<v::InitWeightTransferEngineResponse>, Status> {
        Err(Status::unimplemented(
            "native InitWeightTransferEngine is not implemented",
        ))
    }

    async fn start_weight_update(
        &self,
        _request: Request<v::StartWeightUpdateRequest>,
    ) -> Result<Response<v::StartWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native StartWeightUpdate is not implemented",
        ))
    }

    async fn start_draft_weight_update(
        &self,
        _request: Request<v::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<v::StartDraftWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native StartDraftWeightUpdate is not implemented",
        ))
    }

    async fn update_weights(
        &self,
        _request: Request<v::UpdateWeightsRequest>,
    ) -> Result<Response<v::UpdateWeightsResponse>, Status> {
        Err(Status::unimplemented(
            "native UpdateWeights is not implemented",
        ))
    }

    async fn finish_weight_update(
        &self,
        _request: Request<v::FinishWeightUpdateRequest>,
    ) -> Result<Response<v::FinishWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native FinishWeightUpdate is not implemented",
        ))
    }

    async fn update_weight_version(
        &self,
        _request: Request<v::UpdateWeightVersionRequest>,
    ) -> Result<Response<v::UpdateWeightVersionResponse>, Status> {
        Err(Status::unimplemented(
            "native UpdateWeightVersion is not implemented",
        ))
    }

    async fn get_weight_version(
        &self,
        _request: Request<v::GetWeightVersionRequest>,
    ) -> Result<Response<v::GetWeightVersionResponse>, Status> {
        Err(Status::unimplemented(
            "native GetWeightVersion is not implemented",
        ))
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::vllm_proto::{self as smg_vllm, vllm_engine_server::VllmEngine};
    use vllm_proto::control_server::Control;

    use super::*;

    #[tokio::test]
    async fn discovery_expires_when_an_unpolled_or_running_publisher_is_dropped() {
        for poll in [false, true] {
            let cfg = Config {
                host: "127.0.0.1".into(),
                realistic: true,
                kv_events_zmq_base_port: Some(0),
                kv_events_replay: false,
                kv_events_buffer_steps: 0,
                ..Default::default()
            };
            let engine = Engine::spawn(cfg.engine.clone());
            let publisher =
                crate::kv_zmq::BoundPublisher::bind(engine.clone(), cfg.kv_zmq_for(0, 0).unwrap())
                    .await
                    .unwrap();
            let (_stop, stopped) = watch::channel(false);
            let native = MockNative(MockScheduler {
                cfg: Arc::new(cfg),
                name: "native-kv-lease".into(),
                engine: Some(engine),
                capture: None,
                kv_source: Some(publisher.source()),
                stopped,
                prefill_pending: Arc::new(AtomicU64::new(0)),
                admissions: admission::Admissions::default(),
                overrides: crate::overrides::for_port(0),
            });
            let source = native
                .get_kv_event_sources(Request::new(v::GetKvEventSourcesRequest {}))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                source.sources[0].buffer_steps, 1,
                "effective replay retention"
            );
            assert!(source.sources[0].replay_endpoint.is_empty());
            let mut running = Box::pin(publisher.run());
            if poll {
                assert!(futures::poll!(&mut running).is_pending());
            }
            drop(running);
            assert_eq!(
                native
                    .get_kv_event_sources(Request::new(v::GetKvEventSourcesRequest {}))
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::Unavailable
            );
        }
    }

    #[tokio::test]
    async fn all_grpc_abort_paths_cancel_delayed_admission_without_submitting() {
        tokio::time::timeout(Duration::from_secs(5), async {
            for protocol in ["native", "tokenspeed", "smg-vllm"] {
                let engine = Engine::spawn(engine::EngineParams::default());
                engine.fault_admit_delay_ms(60_000);
                let (_stop, stopped) = watch::channel(false);
                let service = MockScheduler {
                    cfg: Arc::new(Config {
                        realistic: true,
                        ..Default::default()
                    }),
                    name: format!("admission-test-{protocol}"),
                    engine: Some(engine.clone()),
                    capture: None,
                    kv_source: None,
                    stopped,
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                    admissions: admission::Admissions::default(),
                    overrides: crate::overrides::for_port(0),
                };
                let request = ts::GenerateRequest {
                    request_id: "delayed".into(),
                    stream: true,
                    tokenized: Some(ts::TokenizedInput {
                        input_ids: vec![100; 16],
                        ..Default::default()
                    }),
                    sampling_params: Some(ts::SamplingParams {
                        max_new_tokens: Some(1),
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let mut generation = Box::pin(service.generate(Request::new(request.clone())));
                // Poll into the real admission wait before aborting; no scheduling sleep.
                assert!(futures::poll!(&mut generation).is_pending());
                let native = MockNative(service.clone());
                let invalid = native
                    .abort(Request::new(v::AbortRequest {
                        request_ids: vec!["delayed".into(), "".into()],
                    }))
                    .await
                    .unwrap_err();
                assert_eq!(invalid.code(), tonic::Code::InvalidArgument);
                assert!(
                    futures::poll!(&mut generation).is_pending(),
                    "invalid batches must not partially cancel"
                );
                match protocol {
                    "native" => {
                        native
                            .abort(Request::new(v::AbortRequest {
                                request_ids: vec!["delayed".into()],
                            }))
                            .await
                            .unwrap();
                    }
                    "tokenspeed" => {
                        service
                            .abort(Request::new(ts::AbortRequest {
                                request_id: "delayed".into(),
                                ..Default::default()
                            }))
                            .await
                            .unwrap();
                    }
                    _ => {
                        vllm::MockVllm::new(service.clone())
                            .abort(Request::new(smg_vllm::AbortRequest {
                                request_ids: vec!["delayed".into()],
                            }))
                            .await
                            .unwrap();
                    }
                }
                match generation.await {
                    Err(error) => assert_eq!(error.code(), tonic::Code::Cancelled),
                    Ok(_) => panic!("aborted admission returned a generation stream"),
                }
                assert_eq!(service.prefill_pending.load(Ordering::Relaxed), 0);
                assert_eq!(engine.load().num_running_reqs, 0);
                assert_eq!(engine.load().num_waiting_reqs, 0);

                // Completed cancellation leaves no tombstone: the same ID can be reused.
                engine.fault_admit_delay_ms(0);
                let mut output = service
                    .generate(Request::new(request))
                    .await
                    .unwrap()
                    .into_inner();
                let mut completed = false;
                while let Some(frame) = output.next().await {
                    if let Some(GenResp::Complete(done)) = frame.unwrap().response {
                        assert_ne!(done.finish_reason, "abort");
                        assert_eq!(done.output_ids.len(), 1);
                        completed = true;
                    }
                }
                assert!(completed);
                assert_eq!(service.prefill_pending.load(Ordering::Relaxed), 0);
            }
        })
        .await
        .expect("cancelled admission must not wait for the 60-second delay");
    }
}
