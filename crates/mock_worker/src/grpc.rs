//! Mock gRPC worker implementing the TokenSpeed scheduler service. The gateway
//! tokenizes and sends token ids; this service streams back canned token ids.

use std::{
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{
    common_proto as common,
    tokenspeed_scheduler::tokenspeed_proto as ts,
    worker_inference::{
        from_tokenspeed_response, into_tokenspeed_request, proto as worker_inference,
    },
};
use tokio::sync::mpsc;
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};
use worker_inference::worker_inference_server::{WorkerInference, WorkerInferenceServer};

use crate::{
    config::Config,
    control::MockWorkerControl,
    engine::{self, Engine, NewRequest},
};

/// Serve the mock TokenSpeed gRPC service on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let ip = match host.parse::<IpAddr>() {
        Ok(ip) => ip,
        Err(e) => {
            tracing::error!("grpc worker host {host} invalid: {e}");
            return;
        }
    };
    let addr = SocketAddr::new(ip, port);
    // One simulated engine per listener (i.e. per virtual worker).
    let engine = cfg.realistic.then(|| Engine::spawn(cfg.engine.clone()));
    let control = MockWorkerControl::new(&cfg, &host, port);
    let service = MockScheduler { cfg, engine };
    let worker_inference = MockWorkerInference {
        scheduler: service.clone(),
    };
    if let Err(e) = Server::builder()
        .add_service(TokenSpeedSchedulerServer::new(service))
        .add_service(WorkerInferenceServer::new(worker_inference))
        .add_service(control.into_server())
        .serve(addr)
        .await
    {
        tracing::error!("grpc worker {port} stopped: {e}");
    }
}

#[derive(Clone)]
struct MockScheduler {
    cfg: Arc<Config>,
    /// Present iff the worker runs the realistic engine simulator.
    engine: Option<Engine>,
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type WorkerGenStream =
    Pin<Box<dyn Stream<Item = Result<worker_inference::GenerateResponse, Status>> + Send>>;
type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

#[derive(Clone)]
struct MockWorkerInference {
    scheduler: MockScheduler,
}

#[tonic::async_trait]
impl WorkerInference for MockWorkerInference {
    type GenerateStream = WorkerGenStream;

    async fn generate(
        &self,
        request: Request<worker_inference::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = Request::new(into_tokenspeed_request(request.into_inner()));
        let response = TokenSpeedScheduler::generate(&self.scheduler, request).await?;
        let stream = response
            .into_inner()
            .map(|item| item.map(from_tokenspeed_response));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn abort(
        &self,
        request: Request<worker_inference::AbortRequest>,
    ) -> Result<Response<worker_inference::AbortResponse>, Status> {
        let request = request.into_inner();
        let response = TokenSpeedScheduler::abort(
            &self.scheduler,
            Request::new(ts::AbortRequest {
                request_id: request.request_id,
                reason: request.reason,
            }),
        )
        .await?
        .into_inner();
        Ok(Response::new(worker_inference::AbortResponse {
            success: response.success,
            message: response.message,
        }))
    }
}

#[tonic::async_trait]
impl TokenSpeedScheduler for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        // Realistic mode: submit to the engine simulator and stream its output.
        if let Some(engine) = &self.engine {
            let req = request.into_inner();
            let request_id = req.request_id;
            let prompt_token_ids = req.tokenized.map(|t| t.input_ids).unwrap_or_default();
            // Omitted limit falls back to the worker default, matching the HTTP
            // path; `unwrap_or(0)` here would make unbounded requests generate
            // nothing (zero tokens), starving the routing signals.
            let max_new = req
                .sampling_params
                .and_then(|s| s.max_new_tokens)
                .unwrap_or(self.cfg.output_tokens);
            let stream_chunks = req.stream;
            let (tx, rx) = mpsc::unbounded_channel();
            engine.submit(NewRequest {
                request_id: request_id.clone(),
                prompt_token_ids,
                max_new,
                events: tx,
            });
            return Ok(Response::new(generate_stream(
                rx,
                stream_chunks,
                request_id,
            )));
        }

        // Canned mode: a single up-front delay, then synthetic token ids.
        let request_id = request.into_inner().request_id;
        if !self.cfg.gen_delay.is_zero() {
            tokio::time::sleep(self.cfg.gen_delay).await;
        }
        let ids: Vec<u32> = (0..self.cfg.output_tokens).map(|i| 100 + i).collect();

        let mut items: Vec<Result<ts::GenerateResponse, Status>> = Vec::new();
        for id in &ids {
            items.push(Ok(ts::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![*id],
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    index: 0,
                })),
            }));
        }
        items.push(Ok(ts::GenerateResponse {
            request_id,
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: ids,
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: self.cfg.output_tokens,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
            })),
        }));

        Ok(Response::new(Box::pin(stream::iter(items))))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(ts::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            served_model_name: self.cfg.model_id.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: 32768,
            max_req_input_len: 32768,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: String::new(),
            supports_vision: false,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args: None,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        let load = match &self.engine {
            Some(engine) => snapshot_to_scheduler_load(&engine.load()),
            None => ts::SchedulerLoad {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens: 0,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 1_000_000,
                max_running_requests: 0,
                token_usage: 0.0,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                memory: None,
                queues: None,
            },
        };
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: 1,
            loads: vec![load],
            aggregate: None,
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        match &self.engine {
            // Realistic mode with prefix caching: stream the engine's KV events.
            Some(engine) if engine.kv_enabled() => {
                let start = request.into_inner().start_sequence_number;
                Ok(Response::new(engine.subscribe_kv(start)))
            }
            // Otherwise Unimplemented makes the gateway's KvEventMonitor give up
            // cleanly (no idle per-worker task), exactly as before this RPC existed.
            _ => Err(Status::unimplemented(
                "mock-worker (KV events require --engine realistic with --prefix-cache true)",
            )),
        }
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        // The mock has no tokenizer artifacts to serve; the gateway's
        // remote-tokenizer fallback treats Unimplemented as "not supported".
        Err(Status::unimplemented("mock-worker"))
    }
}

/// Map the engine's [`engine::GenEvent`] channel to the gRPC generate stream.
/// In streaming mode each token becomes a `Chunk`; otherwise tokens are
/// accumulated and only the final `Complete` is sent. After `Complete` the
/// engine has dropped the sender, so the next `recv()` yields `None` and the
/// stream ends.
fn generate_stream(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    stream_chunks: bool,
    request_id: String,
) -> GenStream {
    let init = (rx, Vec::<u32>::new(), stream_chunks, request_id);
    Box::pin(stream::unfold(
        init,
        |(mut rx, mut output_ids, stream_chunks, request_id)| async move {
            loop {
                match rx.recv().await {
                    Some(engine::GenEvent::Token {
                        token_id,
                        prompt_tokens,
                        cached_tokens,
                    }) => {
                        output_ids.push(token_id);
                        if stream_chunks {
                            let resp = ts::GenerateResponse {
                                request_id: request_id.clone(),
                                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                                    token_ids: vec![token_id],
                                    prompt_tokens,
                                    completion_tokens: output_ids.len() as u32,
                                    cached_tokens,
                                    output_logprobs: None,
                                    index: 0,
                                })),
                            };
                            return Some((Ok(resp), (rx, output_ids, stream_chunks, request_id)));
                        }
                        // Non-streaming: keep accumulating until Done.
                    }
                    Some(engine::GenEvent::Done {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                    }) => {
                        let resp = ts::GenerateResponse {
                            request_id: request_id.clone(),
                            response: Some(GenResp::Complete(ts::GenerateComplete {
                                output_ids: std::mem::take(&mut output_ids),
                                finish_reason: finish_reason.to_string(),
                                prompt_tokens,
                                completion_tokens,
                                cached_tokens,
                                output_logprobs: None,
                                matched_stop: None,
                                index: 0,
                            })),
                        };
                        return Some((Ok(resp), (rx, output_ids, stream_chunks, request_id)));
                    }
                    None => return None,
                }
            }
        },
    ))
}

/// Map an engine load snapshot to the TokenSpeed `SchedulerLoad` wire type.
fn snapshot_to_scheduler_load(s: &engine::LoadSnapshot) -> ts::SchedulerLoad {
    ts::SchedulerLoad {
        dp_rank: 0,
        num_running_reqs: s.num_running_reqs,
        num_waiting_reqs: s.num_waiting_reqs,
        num_waiting_uncached_tokens: s.num_waiting_uncached_tokens,
        num_total_reqs: s.num_running_reqs + s.num_waiting_reqs,
        num_used_tokens: s.num_used_tokens,
        max_total_num_tokens: s.max_total_num_tokens,
        max_running_requests: s.max_running_requests,
        token_usage: s.token_usage,
        gen_throughput: s.gen_throughput,
        cache_hit_rate: s.cache_hit_rate,
        utilization: s.token_usage,
        memory: None,
        queues: None,
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::worker_inference::TokenSpeedWorkerInference;
    use tokio_stream::wrappers::TcpListenerStream;
    use worker_inference::generate_response::Response as WorkerResp;

    use super::*;
    use crate::engine::EngineParams;

    fn config(output_tokens: u32) -> Arc<Config> {
        Arc::new(Config {
            host: "127.0.0.1".to_string(),
            http_base_port: 0,
            http_count: 0,
            grpc_base_port: 19_000,
            grpc_count: 1,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: "mock-model".to_string(),
            gen_delay: std::time::Duration::ZERO,
            output_tokens,
            realistic: false,
            engine: EngineParams::default(),
        })
    }

    /// Drives the Worker's TokenSpeed gRPC adapter through a real stream. The
    /// scheduler wire already emits delta chunks (one new token per frame
    /// here, as `tokenspeed_scheduler.proto` specifies), so every chunk must
    /// reach the `WorkerInference` wire intact. An adapter that re-derived
    /// deltas from a cumulative assumption would drain every chunk after the
    /// first to nothing while `Complete` still looked right.
    #[tokio::test]
    async fn tokenspeed_adapter_passes_delta_chunks_through_unchanged() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let scheduler = MockScheduler {
            cfg: config(4),
            engine: None,
        };
        let mut servers = tokio::task::JoinSet::new();
        servers.spawn(async move {
            Server::builder()
                .add_service(TokenSpeedSchedulerServer::new(scheduler))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
        });

        let adapter = TokenSpeedWorkerInference::connect(&format!("http://{address}"))
            .await
            .unwrap();
        let request = worker_inference::GenerateRequest {
            request_id: "delta-passthrough".to_string(),
            tokenized: Some(worker_inference::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(worker_inference::SamplingParams {
                max_new_tokens: Some(4),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = WorkerInference::generate(&adapter, Request::new(request))
            .await
            .unwrap()
            .into_inner();

        let mut chunk_tokens = Vec::new();
        let mut complete = None;
        while let Some(item) = stream.next().await {
            let response = item.unwrap();
            assert_eq!(response.request_id, "delta-passthrough");
            match response.response.unwrap() {
                WorkerResp::Chunk(chunk) => {
                    assert_eq!(chunk.index, 0);
                    chunk_tokens.push(chunk.token_ids);
                }
                WorkerResp::Complete(done) => {
                    assert!(complete.replace(done).is_none(), "one Complete per stream");
                }
            }
        }

        // Every frame keeps its own token; nothing is drained as a prefix.
        assert_eq!(
            chunk_tokens,
            vec![vec![100], vec![101], vec![102], vec![103]]
        );
        let complete = complete.expect("stream ends with Complete");
        assert_eq!(complete.output_ids, vec![100, 101, 102, 103]);
        assert_eq!(complete.finish_reason, "stop");
        assert_eq!(complete.completion_tokens, 4);
        servers.abort_all();
    }
}
