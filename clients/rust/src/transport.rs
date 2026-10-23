use reqwest::{Client, RequestBuilder, Response};
use serde::Serialize;

use crate::{config::ClientConfig, SmgError};

/// HTTP transport with retry logic and auth.
#[derive(Debug, Clone)]
pub(crate) struct Transport {
    client: Client,
    config: ClientConfig,
}

/// HTTP status codes that are safe to retry.
const RETRYABLE_STATUSES: &[u16] = &[429, 500, 503];

impl Transport {
    #[expect(
        clippy::disallowed_methods,
        reason = "this crate bundles reqwest's own TLS provider (its `rustls` feature); the lint guards the crates built with `rustls-no-provider`"
    )]
    pub(crate) fn new(config: &ClientConfig) -> Result<Self, SmgError> {
        let mut builder = Client::builder().timeout(config.timeout);

        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref key) = config.api_key {
            let val = format!("Bearer {key}");
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&val)
                    .map_err(|e| SmgError::Stream(format!("invalid api key header: {e}")))?,
            );
        }
        builder = builder.default_headers(headers);

        let client = builder
            .build()
            .map_err(|e| SmgError::Stream(format!("failed to build HTTP client: {e}")))?;

        Ok(Self {
            client,
            config: config.clone(),
        })
    }

    /// Build a full URL from a path.
    fn url(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}{path}")
    }

    /// Send a GET request with retry.
    pub(crate) async fn get(&self, path: &str) -> Result<Response, SmgError> {
        let req = self.client.get(self.url(path));
        self.send_with_retry(req, None::<()>).await
    }

    /// Send a POST request with a JSON body with retry.
    pub(crate) async fn post<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<Response, SmgError> {
        let req = self.client.post(self.url(path));
        self.send_with_retry(req, Some(body)).await
    }

    /// Send a PUT request with a JSON body with retry.
    pub(crate) async fn put<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<Response, SmgError> {
        let req = self.client.put(self.url(path));
        self.send_with_retry(req, Some(body)).await
    }

    /// Send a DELETE request with retry.
    pub(crate) async fn delete(&self, path: &str) -> Result<Response, SmgError> {
        let req = self.client.delete(self.url(path));
        self.send_with_retry(req, None::<()>).await
    }

    /// Send a POST request for streaming (no retry once connected).
    ///
    /// Streaming requests are not retried because once the server starts generating
    /// tokens the response cannot be replayed. Connection-level failures before any
    /// bytes are received propagate as `SmgError::Connection`.
    pub(crate) async fn post_stream<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<Response, SmgError> {
        let resp = self.client.post(self.url(path)).json(body).send().await?;
        check_status(resp).await
    }

    /// Send a request with exponential backoff retry for transient errors.
    async fn send_with_retry<T: Serialize>(
        &self,
        builder: RequestBuilder,
        body: Option<T>,
    ) -> Result<Response, SmgError> {
        let max_attempts = self.config.max_retries + 1;
        let mut last_err = None;

        for attempt in 0..max_attempts {
            let req = match &body {
                Some(b) => builder
                    .try_clone()
                    .ok_or_else(|| SmgError::Stream("request cannot be cloned".into()))?
                    .json(b),
                None => builder
                    .try_clone()
                    .ok_or_else(|| SmgError::Stream("request cannot be cloned".into()))?,
            };

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if RETRYABLE_STATUSES.contains(&status) && attempt < max_attempts - 1 {
                        // Parse Retry-After header as seconds if present.
                        // HTTP-date values (e.g. "Wed, 21 Oct 2015 07:28:00 GMT")
                        // fall through to backoff_delay, which is acceptable.
                        let delay = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<f64>().ok())
                            .filter(|&v| v.is_finite() && v >= 0.0)
                            .map(std::time::Duration::from_secs_f64)
                            .unwrap_or_else(|| backoff_delay(attempt));

                        let body_text = resp.text().await.unwrap_or_default();
                        last_err = Some(SmgError::from_status(status, &body_text));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return check_status(resp).await;
                }
                Err(e) => {
                    if attempt < max_attempts - 1 && is_retryable_error(&e) {
                        last_err = Some(SmgError::Connection(e));
                        tokio::time::sleep(backoff_delay(attempt)).await;
                        continue;
                    }
                    return Err(SmgError::Connection(e));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| SmgError::Stream("max retries exceeded".into())))
    }
}

/// Check the HTTP response status and return an error for non-2xx.
async fn check_status(resp: Response) -> Result<Response, SmgError> {
    let status = resp.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    Err(SmgError::from_status(status, &body))
}

/// Exponential backoff with jitter: base of 0.5s, 1s, 2s, 4s, ... plus
/// random jitter up to 50% of the base delay to avoid thundering-herd effects.
fn backoff_delay(attempt: u32) -> std::time::Duration {
    let base_ms = 500u64.saturating_mul(2u64.saturating_pow(attempt));
    let capped_ms = base_ms.min(30_000);
    // Simple jitter using system time to avoid adding a `rand` dependency.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let jitter_ms = nanos % (capped_ms / 2 + 1);
    std::time::Duration::from_millis((capped_ms + jitter_ms).min(30_000))
}

/// Check if a reqwest error is transient and worth retrying.
fn is_retryable_error(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}
