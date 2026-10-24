//! SGLang IMAGE encode RPCs coordinated with one prefill attempt.
//!
//! Without embedding ports, zmq_to_scheduler Encode waits for scheduler
//! registration and delivery, so dispatch it alongside prefill Generate.

use std::{collections::HashSet, future::Future, time::Duration};

use futures::future::try_join_all;
use tokio::time::{sleep_until, Instant};
use tonic::{transport::Channel, Request, Status};

use crate::sglang_proto;

#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.encoder");
}

#[derive(Clone)]
pub struct SglangEncoderClient {
    endpoint: String,
    client: proto::sglang_encoder_client::SglangEncoderClient<Channel>,
}

impl SglangEncoderClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let channel = crate::connect_channel(endpoint).await?;
        Ok(Self {
            endpoint: endpoint.to_string(),
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

/// Run active encoders concurrently with one prefill attempt.
///
/// Construct every Encode from this attempt's final ID and ordered IMAGE slices.
/// Reject stale handoffs after IdStamp changes. Zero assignments start no RPC.
/// Timeout/error drops local sibling futures; backend release is not proven.
/// Requires an encoder configured for zmq_to_scheduler; that needs server-side
/// verification. This does not connect EncodeStage or change gateway retries.
pub async fn encode_with_prefill<F, Fut, T>(
    encoders: Vec<SglangEncoderClient>,
    prefill: sglang_proto::GenerateRequest,
    timeout: Duration,
    dispatch_prefill: F,
) -> Result<(Vec<proto::EncodeResponse>, T), Status>
where
    F: FnOnce(sglang_proto::GenerateRequest) -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| Status::invalid_argument("attempt deadline overflow"))?;
    let handoff = prefill
        .encode_handoff
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("prefill encoder handoff missing"))?;
    if timeout.is_zero()
        || prefill.request_id.is_empty()
        || handoff.request_id != prefill.request_id
        || handoff.encoder_urls.len() != handoff.num_items_assigned.len()
        || handoff.encoder_urls.iter().collect::<HashSet<_>>().len() != handoff.encoder_urls.len()
        || encoders.is_empty()
    {
        return Err(Status::invalid_argument("invalid SGLang encode attempt"));
    }
    let images = prefill
        .mm_inputs
        .as_ref()
        .filter(|media| {
            !media.image_urls.is_empty()
                && media.video_urls.is_empty()
                && media.audio_urls.is_empty()
                && media.pixel_values.is_none()
        })
        .ok_or_else(|| Status::invalid_argument("encoder handoff needs image URL inputs"))?;
    let total = handoff
        .num_items_assigned
        .iter()
        .try_fold(0usize, |sum, count| sum.checked_add(*count as usize));
    if total != Some(images.image_urls.len()) {
        return Err(Status::invalid_argument(
            "encoder assignments do not cover the images",
        ));
    }
    let assignments: Vec<_> = handoff
        .encoder_urls
        .iter()
        .zip(&handoff.num_items_assigned)
        .filter(|(_, count)| **count != 0)
        .collect();
    let parts = u32::try_from(encoders.len())
        .map_err(|_| Status::invalid_argument("too many encoder parts"))?;
    if assignments.len() != encoders.len() {
        return Err(Status::invalid_argument(
            "encoder client/assignment count differs",
        ));
    }
    let mut jobs = Vec::with_capacity(encoders.len());
    let mut offset = 0;
    for (index, (client, (url, count))) in encoders.into_iter().zip(assignments).enumerate() {
        if client.endpoint != *url {
            return Err(Status::invalid_argument(
                "encoder endpoint order differs from handoff",
            ));
        }
        let end = offset + *count as usize;
        let request = proto::EncodeRequest {
            req_id: prefill.request_id.clone(),
            mm_items: images.image_urls[offset..end].to_vec(),
            num_parts: parts,
            part_idx: index as u32,
            ..Default::default()
        };
        jobs.push((client, request));
        offset = end;
    }
    let encode = try_join_all(
        jobs.into_iter()
            .map(|(client, request)| async move { client.encode_until(request, deadline).await }),
    );
    // Tonic reports its transport timeout as Cancelled; poll our deadline first.
    tokio::select! {
        biased;
        _ = sleep_until(deadline) => Err(Status::deadline_exceeded("SGLang EPD attempt deadline elapsed")),
        result = async { futures::try_join!(encode, dispatch_prefill(prefill)) } => result,
    }
}
