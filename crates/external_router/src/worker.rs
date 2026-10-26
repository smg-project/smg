//! What an external router sees of the gateway's workers.

use std::{any::Any, borrow::Cow, fmt::Debug, sync::Arc};

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

/// A worker's URL as a log line or a message shows it.
///
/// A URL without userinfo, query or fragment is returned as it is, byte for
/// byte and without an allocation: the common case, so a plain address
/// reads in the logs exactly as it was configured. Otherwise the userinfo is
/// masked whole and the address is cut at its path:
/// `https://user:secret@host:8000/v1/messages?k=v` reads
/// `https://***@host:8000/v1/messages`. The username goes with the secret,
/// since a token is often the username (`https://token@host`,
/// `https://token:x-oauth-basic@host`); a query or a fragment is no part of
/// the address and may carry a token. The gateway renders a worker's own URL
/// the same way through its worker type; until that rendering is reachable
/// from a worker held here, every site writes through this rule, which then
/// stays for the strings that are not a worker's rendering: an MCP server
/// URL, a request URL composed from a worker's address and a path. It works
/// on the string alone: nothing is parsed, and an `@` before the first `/` of
/// a schemeless string counts as userinfo as well.
pub(crate) fn redacted_url(url: &str) -> Cow<'_, str> {
    // The authority runs from the scheme separator (when the text before it
    // is a scheme) to the first `/`, `?` or `#`.
    let authority_start = url
        .find("://")
        .filter(|&at| {
            url[..at]
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"+-.".contains(&byte))
        })
        .map_or(0, |at| at + 3);
    let authority_end = url[authority_start..]
        .find(['/', '?', '#'])
        .map_or(url.len(), |at| authority_start + at);
    let host_start = url[authority_start..authority_end]
        .rfind('@')
        .map(|at| authority_start + at + 1);
    let cut = url[authority_end..]
        .find(['?', '#'])
        .map(|at| authority_end + at);
    if host_start.is_none() && cut.is_none() {
        return Cow::Borrowed(url);
    }
    let mut path_end = cut.unwrap_or(url.len());
    // As the gateway renders it: a lone `/` is no path.
    if &url[authority_end..path_end] == "/" {
        path_end = authority_end;
    }
    let mut masked = String::with_capacity(path_end);
    masked.push_str(&url[..authority_start]);
    if host_start.is_some() {
        masked.push_str("***@");
    }
    masked.push_str(&url[host_start.unwrap_or(authority_start)..path_end]);
    Cow::Owned(masked)
}

#[cfg(test)]
mod tests {
    use super::redacted_url;

    /// A URL without userinfo, query or fragment is the text as given,
    /// borrowed: whatever its scheme, host form, path or trailing slash.
    #[test]
    fn a_plain_url_reads_byte_for_byte() {
        for plain in [
            "https://api.example.com",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1/messages",
            "http://Worker:3000/",
            "wss://host/v1/realtime",
            "127.0.0.1:8000",
            "",
        ] {
            let shown = redacted_url(plain);
            assert!(matches!(shown, std::borrow::Cow::Borrowed(_)), "{plain}");
            assert_eq!(shown, plain);
        }
    }

    /// A userinfo is masked whole, whatever its shape, and a query or a
    /// fragment is cut; nothing of the credential survives.
    #[test]
    fn a_userinfo_is_masked_whole_and_a_query_is_cut() {
        for (url, masked) in [
            (
                "https://user:secret@127.0.0.1:1/v1/messages",
                "https://***@127.0.0.1:1/v1/messages",
            ),
            ("https://api-token@host:8000", "https://***@host:8000"),
            ("https://tok-123:x-oauth-basic@host", "https://***@host"),
            (
                "wss://user:secret@127.0.0.1:1/v1/realtime?model=m",
                "wss://***@127.0.0.1:1/v1/realtime",
            ),
            ("http://user:secret@host:8000/", "http://***@host:8000"),
            ("http://host:8000/v1?token=secret", "http://host:8000/v1"),
            ("http://host:8000/v1#secret", "http://host:8000/v1"),
            ("http://host:8000/?x=1", "http://host:8000"),
            // Schemeless: an `@` before the first `/` is a userinfo too.
            ("user:secret@host/sse", "***@host/sse"),
            // A `://` inside a path or a query is not a scheme separator.
            ("host/x?next=http://user:secret@y", "host/x"),
        ] {
            let shown = redacted_url(url);
            assert_eq!(shown, masked, "{url}");
            for credential in ["secret", "user", "api-token", "tok-123", "x-oauth-basic"] {
                assert!(!shown.contains(credential), "{url} -> {shown}");
            }
        }
    }
}
