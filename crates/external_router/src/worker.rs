//! What an external router sees of the gateway's workers.

use std::{any::Any, fmt::Debug, sync::Arc};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::worker::{ConnectionMode, ProviderType, RuntimeType, WorkerType};

use crate::{ExternalRouterSpec, RetryConfig};

/// Keeps a worker's load accounted for as long as it is alive; dropping it
/// releases the load.
pub type LoadHold = Box<dyn Any + Send + Sync>;

/// One worker as an external router may use it.
pub trait ExternalWorker: Send + Sync + Debug {
    fn url(&self) -> &str;
    fn api_key(&self) -> Option<&String>;
    fn model_id(&self) -> &str;
    fn is_healthy(&self) -> bool;
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType>;
    /// Report the upstream status so health and circuit state follow it.
    fn record_outcome(&self, status_code: u16);
    fn http_client(&self) -> &reqwest::Client;
    /// Hold the worker's load for a long-lived session.
    #[must_use = "dropping the hold releases the worker's load at once"]
    fn hold_load(&self, headers: Option<&HeaderMap>) -> LoadHold;
}

/// What a worker selection asks for.
///
/// Combines the model to resolve with optional registry filters and
/// the caller's HTTP headers (used for auth passthrough during
/// upstream model refresh).
#[derive(Debug, Default)]
pub struct SelectWorkerRequest<'a> {
    /// Model ID to select a worker for (required).
    pub model_id: &'a str,

    /// Caller's HTTP headers — used to extract the auth token for
    /// upstream `/v1/models` refresh on cache miss.
    pub headers: Option<&'a HeaderMap>,

    /// The router selecting, so only workers it takes are candidates: a
    /// caller's credentials must not reach a worker of another provider.
    pub router: Option<ExternalRouterSpec>,

    /// Filter by worker type (Regular, Prefill, Decode). `None` = any.
    pub worker_type: Option<WorkerType>,

    /// Filter by connection mode (Http, Grpc). `None` = any.
    pub connection_mode: Option<ConnectionMode>,

    /// Filter by runtime type (External, Sglang, Vllm, Trtllm). `None` = any.
    pub runtime_type: Option<RuntimeType>,

    /// When `true`, restrict candidates to workers advertising realtime
    /// capability (the `realtime` label). Used by the realtime routes so
    /// they never proxy to a worker that can't serve realtime.
    pub require_realtime_capable: bool,
}

/// A snapshot of the worker population.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerStats {
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub unhealthy_workers: usize,
}

/// The gateway's worker registry as an external router may use it.
#[async_trait]
pub trait WorkerSource: Send + Sync + Debug {
    /// Pick a worker for a request, or answer with the response that says
    /// why none fits.
    async fn select(
        &self,
        req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn ExternalWorker>, Response>;
    /// A retry policy a worker registered for the model, if any.
    fn retry_config(&self, model_id: &str) -> Option<RetryConfig>;
    fn stats(&self) -> WorkerStats;
    /// Every worker that fronts a third-party provider.
    fn external_workers(&self) -> Vec<Arc<dyn ExternalWorker>>;
}
