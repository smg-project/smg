//! Worker-local engine transports.
//!
//! The public Router-to-Worker boundary remains `WorkerInference` gRPC. This
//! module adapts the existing same-host ZMQ engine client to that stable wire,
//! so colocated vLLM and TokenSpeed schedulers can avoid a Python/gRPC hop.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use smg_grpc_client::{
    worker_inference::{
        from_vllm_response, into_tokenspeed_request, into_vllm_request, EngineTransport,
        EngineTransportStream,
    },
    worker_inference_proto as proto,
};
use tokio::sync::oneshot;
use tonic::Status;

use crate::{
    routers::grpc::{
        proto_wrapper::ProtoGenerateRequest,
        zmq_client::{connect_for_worker, ZmqEngineClient, ZmqGenerateStream},
    },
    worker::RuntimeType,
};

type ActiveRequests = Arc<Mutex<HashMap<String, oneshot::Sender<()>>>>;

/// Same-host ZMQ IPC implementation of the Worker-local engine boundary.
#[derive(Clone)]
pub struct ZmqWorkerTransport {
    client: ZmqEngineClient,
    runtime: RuntimeType,
    active: ActiveRequests,
}

impl ZmqWorkerTransport {
    /// Bind the Worker-side ZMQ sockets and await the engine handshake.
    pub async fn connect(
        base_url: &str,
        model_id: String,
        runtime: RuntimeType,
        handshake_override: Option<&str>,
        engine_count: usize,
    ) -> Result<Self, String> {
        if !matches!(runtime, RuntimeType::Vllm | RuntimeType::TokenSpeed) {
            return Err(format!(
                "Worker ZMQ transport supports only vllm and tokenspeed, not {runtime}"
            ));
        }
        if engine_count == 0 {
            return Err("Worker ZMQ engine_count must be positive".to_string());
        }
        let client = connect_for_worker(
            base_url,
            model_id,
            runtime,
            handshake_override,
            engine_count,
        )
        .await?;
        Ok(Self {
            client,
            runtime,
            active: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

struct ZmqStreamState {
    request_id: String,
    stream: ZmqGenerateStream,
    cancel: oneshot::Receiver<()>,
    active: ActiveRequests,
    done: bool,
}

impl Drop for ZmqStreamState {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active.lock() {
            active.remove(&self.request_id);
        }
        // Dropping `stream` before its terminal frame triggers the existing
        // engine-zmq-client auto-abort path.
    }
}

/// Terminal status of a stream ended by `WorkerInference.Abort`. The Router
/// distinguishes this from an engine failure and from a stream that merely
/// ran out of frames.
fn cancelled_status() -> Status {
    Status::cancelled("request aborted on the Worker")
}

#[tonic::async_trait]
impl EngineTransport for ZmqWorkerTransport {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        let request_id = request.request_id.clone();
        let engine_request = match self.runtime {
            RuntimeType::Vllm => ProtoGenerateRequest::Vllm(Box::new(into_vllm_request(request)?)),
            RuntimeType::TokenSpeed => {
                ProtoGenerateRequest::TokenSpeed(Box::new(into_tokenspeed_request(request)))
            }
            other => {
                return Err(Status::failed_precondition(format!(
                    "Worker ZMQ transport is unavailable for {other}"
                )))
            }
        };
        // Register before submitting, not after: an `abort` landing in the gap
        // would find no entry, still answer `success: true`, and leave the
        // engine generating to completion behind an accepted cancellation.
        let (cancel_tx, mut cancel_rx) = oneshot::channel();
        {
            let mut active = self
                .active
                .lock()
                .map_err(|_| Status::internal("Worker ZMQ request registry is poisoned"))?;
            // A second stream under the same id would overwrite this sender,
            // and the first stream's `Drop` would then remove the *new* entry,
            // turning a later `abort` into a silent no-op. The id is the
            // Router's cancellation handle, so it has to be unique here.
            if active.contains_key(&request_id) {
                return Err(Status::already_exists(format!(
                    "request {request_id} is already in flight on this Worker"
                )));
            }
            active.insert(request_id.clone(), cancel_tx);
        }

        let stream = match self.client.generate(engine_request).await {
            Ok(stream) => stream,
            Err(error) => {
                // Nothing else will drop the registration: `ZmqStreamState` is
                // never built on this path.
                if let Ok(mut active) = self.active.lock() {
                    active.remove(&request_id);
                }
                return Err(error);
            }
        };
        // An abort that raced the submission has already fired the sender. The
        // engine has the request by now, so surface the cancellation as the
        // stream's only item and let the drop-driven auto-abort reach the
        // engine. A clean EOF here would look to the Router like a stream that
        // ended without `Complete`, not like the abort it asked for.
        if cancel_rx.try_recv().is_ok() {
            drop(stream);
            return Ok(Box::pin(futures::stream::once(async {
                Err(cancelled_status())
            })));
        }

        let state = ZmqStreamState {
            request_id,
            stream,
            cancel: cancel_rx,
            active: Arc::clone(&self.active),
            done: false,
        };
        let stream = futures::stream::unfold(state, |mut state| async move {
            if state.done {
                return None;
            }
            tokio::select! {
                _ = &mut state.cancel => {
                    state.done = true;
                    Some((Err(cancelled_status()), state))
                }
                item = state.stream.next() => match item {
                    Some(Ok(response)) => {
                        let response = from_vllm_response(&state.request_id, response);
                        Some((Ok(response), state))
                    }
                    Some(Err(error)) => {
                        state.done = true;
                        Some((Err(error), state))
                    }
                    None => None,
                },
            }
        });
        Ok(Box::pin(stream))
    }

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status> {
        let sender = self
            .active
            .lock()
            .map_err(|_| Status::internal("Worker ZMQ request registry is poisoned"))?
            .remove(&request.request_id);
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
        // Abort is idempotent: an already-finished or already-cancelled
        // request is a successful no-op.
        Ok(proto::AbortResponse {
            success: true,
            message: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
        EngineId,
    };
    use futures::StreamExt;

    use super::*;
    use crate::routers::grpc::zmq_client::zmq_handshake_address;

    #[tokio::test]
    async fn vllm_zmq_transport_maps_generate_and_propagates_abort() {
        let dir = tempfile::tempdir().expect("temporary socket directory");
        let base_url = format!("ipc://{}", dir.path().join("worker").display());
        let handshake = zmq_handshake_address(&base_url, None).expect("handshake address");

        let (transport, engine) = tokio::join!(
            ZmqWorkerTransport::connect(
                &base_url,
                "org/model".to_string(),
                RuntimeType::Vllm,
                None,
                1,
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response(),
            ),
        );
        let transport = transport.expect("Worker ZMQ transport");
        let mut engine = engine.expect("mock vLLM engine");

        let request_id = "worker-zmq-1".to_string();
        let mut stream = transport
            .generate(proto::GenerateRequest {
                request_id: request_id.clone(),
                tokenized: Some(proto::TokenizedInput {
                    original_text: "hello".to_string(),
                    input_ids: vec![1, 2, 3],
                }),
                sampling_params: Some(proto::SamplingParams {
                    max_new_tokens: Some(4),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            })
            .await
            .expect("generate stream");

        let inbound = tokio::time::timeout(Duration::from_secs(2), engine.recv())
            .await
            .expect("engine request timeout")
            .expect("engine request");
        let EngineInbound::Add(request) = inbound else {
            panic!("expected add request, got {inbound:?}");
        };
        assert_eq!(request.request_id, request_id);
        assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
        assert_eq!(request.sampling_params.expect("sampling").max_tokens, 4);

        transport
            .abort(proto::AbortRequest {
                request_id: request_id.clone(),
                reason: "test cancellation".to_string(),
            })
            .await
            .expect("abort");
        let status = stream
            .next()
            .await
            .expect("cancelled stream reports the abort")
            .expect_err("cancellation is a status, not a frame");
        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert!(stream.next().await.is_none(), "cancelled stream must close");

        let inbound = tokio::time::timeout(Duration::from_secs(2), engine.recv())
            .await
            .expect("engine abort timeout")
            .expect("engine abort");
        let EngineInbound::Abort(request_ids) = inbound else {
            panic!("expected abort request, got {inbound:?}");
        };
        assert_eq!(request_ids, vec![request_id]);

        // Checked after the abort above has reached the engine, so the
        // duplicate's own submission cannot race that inbound frame.
        // The id is the Router's cancellation handle: a second registration
        // under it must be refused rather than silently replace the first.
        let dup = proto::GenerateRequest {
            request_id: "worker-zmq-dup".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "hello".to_string(),
                input_ids: vec![1],
            }),
            sampling_params: Some(proto::SamplingParams::default()),
            stream: true,
            ..Default::default()
        };
        let _first = transport.generate(dup.clone()).await.expect("first stream");
        let code = match transport.generate(dup).await {
            Ok(_) => panic!("duplicate request id must be rejected"),
            Err(status) => status.code(),
        };
        assert_eq!(code, tonic::Code::AlreadyExists);
    }
}
