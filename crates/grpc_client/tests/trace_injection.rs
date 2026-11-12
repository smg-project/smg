//! The trace context the client's injector writes reaches the engine on
//! every call that injects it: `generate`, `embed`, `flush_cache`,
//! `start_profile` and `stop_profile`. Driven through the real
//! `VllmEngineClient` against the in-process mock engine, which records the
//! trace header each call carried; the no-op injector the plain `connect`
//! installs is the control.

use std::sync::Arc;

use smg_grpc_client::{common_proto, vllm_proto as proto, TraceInjector, VllmEngineClient};
use tonic::metadata::{MetadataMap, MetadataValue};

mod common;
use common::{generate_request, spawn_mock, MockState, TRACE_HEADER};

/// A W3C trace context (the specification's own example), the value the
/// gateway's injector writes under the same header from the active span.
const TRACE_PARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

/// Writes `traceparent`, the header the gateway's injector carries the span
/// in (it writes `tracestate` beside it, which these tests do not read).
struct ProbeInjector;

impl TraceInjector for ProbeInjector {
    fn inject(
        &self,
        metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        metadata.insert(TRACE_HEADER, MetadataValue::from_static(TRACE_PARENT));
        Ok(())
    }
}

type Failure = Box<dyn std::error::Error + Send + Sync>;

async fn traced_client() -> Result<(VllmEngineClient, Arc<MockState>), Failure> {
    let (endpoint, state) = spawn_mock().await?;
    let client =
        VllmEngineClient::connect_with_trace_injector(&endpoint, Arc::new(ProbeInjector)).await?;
    Ok((client, state))
}

fn carried(call: &'static str) -> Vec<(&'static str, Option<String>)> {
    vec![(call, Some(TRACE_PARENT.to_owned()))]
}

#[tokio::test]
async fn generate_carries_the_trace_context() {
    let (client, state) = traced_client().await.unwrap();
    let _stream = client.generate(generate_request("traced-1")).await.unwrap();
    assert_eq!(state.traced(), carried("Generate"));
}

#[tokio::test]
async fn embed_carries_the_trace_context() {
    let (client, state) = traced_client().await.unwrap();
    client.embed(proto::EmbedRequest::default()).await.unwrap();
    assert_eq!(state.traced(), carried("Embed"));
}

#[tokio::test]
async fn flush_cache_carries_the_trace_context() {
    let (client, state) = traced_client().await.unwrap();
    client.flush_cache(1.5).await.unwrap();
    assert_eq!(state.traced(), carried("FlushCache"));
}

#[tokio::test]
async fn start_profile_carries_the_trace_context() {
    let (client, state) = traced_client().await.unwrap();
    client
        .start_profile(common_proto::StartProfileRequest::default())
        .await
        .unwrap();
    assert_eq!(state.traced(), carried("StartProfile"));
}

#[tokio::test]
async fn stop_profile_carries_the_trace_context() {
    let (client, state) = traced_client().await.unwrap();
    client.stop_profile().await.unwrap();
    assert_eq!(state.traced(), carried("StopProfile"));
}

#[tokio::test]
async fn the_plain_connect_injects_nothing() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let _stream = client.generate(generate_request("plain-1")).await.unwrap();
    assert_eq!(state.traced(), [("Generate", None)]);
}
