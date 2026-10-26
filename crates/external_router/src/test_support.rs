//! What the crate's capture tests drive a request through: a worker at an
//! address of the test's choosing, a worker source over such workers, and
//! the HTTP client they share.

use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::worker::ProviderType;
use rustls::crypto::ring::default_provider;

use crate::{
    error,
    worker::{ExternalWorker, LoadHold, SelectWorkerRequest, WorkerSource, WorkerStats},
    RetryConfig,
};

/// A worker registered with credentials in its URL, at a port nobody listens
/// on: a request to it fails at the connection, and every line written on
/// the way must show the address as [`MASKED_WORKER_URL`].
pub(crate) const CREDENTIALED_WORKER_URL: &str = "https://user:secret@127.0.0.1:1";

/// [`CREDENTIALED_WORKER_URL`] as a log line or a message shows it.
pub(crate) const MASKED_WORKER_URL: &str = "https://***@127.0.0.1:1";

/// The tests' HTTP client, the process TLS crypto provider installed first.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub(crate) fn http_client() -> reqwest::Client {
    let _ = default_provider().install_default();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("a client")
}

/// A worker at one address, serving model `m`, with no API key of its own.
pub(crate) struct StubWorker {
    url: String,
    redacted: String,
    healthy: bool,
    client: reqwest::Client,
}

impl StubWorker {
    /// A healthy worker at `url`.
    pub(crate) fn at(url: &str) -> Arc<dyn ExternalWorker> {
        Arc::new(Self {
            url: url.to_string(),
            redacted: crate::worker::redacted_url(url).into_owned(),
            healthy: true,
            client: http_client(),
        })
    }

    /// A worker at `url` whose health check has failed.
    #[cfg_attr(
        not(feature = "openai"),
        expect(dead_code, reason = "read by the health answers' tests alone")
    )]
    pub(crate) fn unhealthy_at(url: &str) -> Arc<dyn ExternalWorker> {
        Arc::new(Self {
            url: url.to_string(),
            redacted: crate::worker::redacted_url(url).into_owned(),
            healthy: false,
            client: http_client(),
        })
    }
}

impl fmt::Debug for StubWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StubWorker")
            .field("url", &self.url)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl ExternalWorker for StubWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn redacted_url(&self) -> &str {
        &self.redacted
    }

    fn api_key(&self) -> Option<&String> {
        None
    }

    fn model_id(&self) -> &str {
        "m"
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }

    fn provider_for_model(&self, _model_id: &str) -> Option<&ProviderType> {
        None
    }

    fn record_outcome(&self, _status_code: u16) {}

    fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    fn hold_load(&self, _headers: Option<&HeaderMap>) -> LoadHold {
        Box::new(())
    }
}

/// A worker source over a fixed set of workers; it selects none of them.
#[cfg_attr(
    not(any(feature = "openai", feature = "gemini")),
    expect(
        dead_code,
        reason = "read by the health answers' and the Gemini step's tests alone"
    )
)]
#[derive(Debug, Default)]
pub(crate) struct StubWorkers(pub(crate) Vec<Arc<dyn ExternalWorker>>);

#[async_trait]
impl WorkerSource for StubWorkers {
    async fn select(
        &self,
        _req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn ExternalWorker>, Response> {
        Err(error::service_unavailable(
            "service_unavailable",
            "No worker selected",
        ))
    }

    fn retry_config(&self, _model_id: &str) -> Option<RetryConfig> {
        None
    }

    fn stats(&self) -> WorkerStats {
        WorkerStats::default()
    }

    fn external_workers(&self) -> Vec<Arc<dyn ExternalWorker>> {
        self.0.clone()
    }
}
