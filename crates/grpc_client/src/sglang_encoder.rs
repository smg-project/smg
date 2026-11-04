//! SGLang IMAGE encode RPCs coordinated with one prefill attempt.
//!
//! Without embedding ports, zmq_to_scheduler Encode waits for scheduler
//! registration and delivery, so dispatch it alongside prefill Generate.

use std::sync::Arc;

use tokio::time::{sleep_until, Instant};
use tonic::{transport::Channel, Request, Status};

use crate::{BoxedTraceInjector, NoopTraceInjector, RequestDeadline};

#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.encoder");
}

#[derive(Clone)]
pub struct SglangEncoderClient {
    client: proto::sglang_encoder_client::SglangEncoderClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl SglangEncoderClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let channel = crate::connect_channel(endpoint).await?;
        Ok(Self {
            client: proto::sglang_encoder_client::SglangEncoderClient::new(channel),
            trace_injector,
        })
    }

    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Complete Encode; only zmq_to_scheduler includes scheduler delivery.
    /// The caller owns the deadline, shared with its other request work.
    pub async fn encode(
        &self,
        request: proto::EncodeRequest,
        deadline: RequestDeadline,
    ) -> Result<proto::EncodeResponse, Status> {
        tokio::select! {
            biased;
            () = sleep_until(deadline.at()) => Err(deadline.status()),
            result = self.encode_until(request, deadline) => result,
        }
    }

    async fn encode_until(
        &self,
        request: proto::EncodeRequest,
        deadline: RequestDeadline,
    ) -> Result<proto::EncodeResponse, Status> {
        let remaining = deadline.at().saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(deadline.status());
        }
        let mut client = self.client.clone();
        let mut request = Request::new(request);
        // The caller owns expiration; grpc-timeout would arm tonic's second,
        // rounded local timer and lose the gateway timeout marker.
        if let Err(error) = self.trace_injector.inject(request.metadata_mut()) {
            tracing::warn!("Failed to inject trace context: {error}");
        }
        client
            .encode(request)
            .await
            .map(tonic::Response::into_inner)
    }
}
