//! SGLang encoder RPC client.

use std::time::Duration;

use tokio::time::{sleep_until, Instant};
use tonic::{transport::Channel, Request, Status};

#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.encoder");
}

#[derive(Clone)]
pub struct SglangEncoderClient {
    client: proto::sglang_encoder_client::SglangEncoderClient<Channel>,
}

impl SglangEncoderClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let channel = crate::connect_channel(endpoint).await?;
        Ok(Self {
            client: proto::sglang_encoder_client::SglangEncoderClient::new(channel),
        })
    }

    /// Complete the Encode RPC without asserting a remote transfer backend.
    /// With zmq_to_scheduler this includes scheduler delivery; mooncake still
    /// requires Send. Verify the configured backend separately.
    pub async fn encode(
        &self,
        request: proto::EncodeRequest,
        timeout: Duration,
    ) -> Result<proto::EncodeResponse, Status> {
        if timeout.is_zero() {
            return Err(Status::invalid_argument("encode deadline must be nonzero"));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Status::invalid_argument("encode deadline overflow"))?;
        tokio::select! {
            biased;
            _ = sleep_until(deadline) => Err(Status::deadline_exceeded("SGLang encode deadline elapsed")),
            result = self.encode_until(request, deadline) => result,
        }
    }

    async fn encode_until(
        &self,
        request: proto::EncodeRequest,
        deadline: Instant,
    ) -> Result<proto::EncodeResponse, Status> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(Status::deadline_exceeded("SGLang encode deadline elapsed"));
        }
        let mut client = self.client.clone();
        let mut request = Request::new(request);
        request.set_timeout(remaining);
        client
            .encode(request)
            .await
            .map(tonic::Response::into_inner)
    }
}
