use std::time::Duration;

use axum::{http::StatusCode, response::Response};
use rand::RngExt;
use tracing::debug;

use crate::{error::gateway_timeout, RetryConfig};

/// Check if an HTTP status code indicates a retryable error
pub fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

/// Response extension marking a response terminal for the retry layer
/// regardless of its status code — for outcomes whose cause cannot clear
/// inside a backoff window (the load-shed 503: its veto moves at the poll
/// interval, not the retry interval).
#[derive(Debug, Clone, Copy)]
pub struct NonRetryable;

/// Mark `response` terminal for [`is_retryable_response`].
pub fn mark_non_retryable(response: &mut Response) {
    response.extensions_mut().insert(NonRetryable);
}

/// The code of the gateway's answer for an upstream call its request timeout
/// cut, the same on every path, so a client keys on one code whatever the
/// transport behind the gateway.
pub const REQUEST_TIMEOUT_CODE: &str = "call_upstream_timeout";

/// Whether `error` is the request's own timeout on an upstream call: the
/// client's budget (`--request-timeout-secs`, a worker's `timeout_secs`, a
/// call's own bound) ran out while the request connected, waited for the
/// head or read the body.
///
/// Not a connect timeout, which is a connect failure like a refused connect
/// (the worker was never reached, and another may be), and not a timeout
/// below the client (the OS giving up on a peer that stopped answering),
/// which is the worker's connection failing under the request. Both stay the
/// worker's fault, which the retry layer routes around.
///
/// A total timeout that runs out while the connect is still pending reads as
/// the request's own as well (the client marks no phase): the budget is per
/// attempt, and a replay could not fit in what is left of it. The gateway's
/// clients keep their connect timeout below the request timeout for that
/// reason (the worker clients and the shared client its external providers
/// send through, clamped where a configuration would not), so a host that has
/// gone dark fails as a connect timeout and is routed around. Whole seconds
/// cannot go below one: under a 1 s request timeout the two timers race, and a
/// dark host may read as the request's own timeout instead.
pub fn is_request_timeout(error: &reqwest::Error) -> bool {
    if !error.is_timeout() || error.is_connect() {
        return false;
    }
    let mut cause = std::error::Error::source(error);
    while let Some(err) = cause {
        if err
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        {
            return false;
        }
        cause = err.source();
    }
    true
}

/// The answer for an upstream call the request timeout cut: a `504` under
/// [`REQUEST_TIMEOUT_CODE`], terminal for the retry layer. The timeout is the
/// operator's bound on what a request may take, not a transient fault of the
/// worker: a replay would run the work again on another worker and hold the
/// client for a second timeout.
pub fn request_timeout_response(message: impl Into<String>) -> Response {
    let mut response = gateway_timeout(REQUEST_TIMEOUT_CODE, message);
    mark_non_retryable(&mut response);
    response
}

/// Whether a response should be retried: a retryable status that has not been
/// explicitly marked terminal by the code that produced it.
///
/// This is what `should_retry` predicates should call. `is_retryable_status`
/// alone cannot see the marker and will retry a shed.
pub fn is_retryable_response(response: &Response) -> bool {
    response.extensions().get::<NonRetryable>().is_none() && is_retryable_status(response.status())
}

/// Computes exponential backoff with optional jitter.
#[derive(Debug, Clone)]
pub struct BackoffCalculator;

impl BackoffCalculator {
    /// Calculate backoff delay for a given attempt index (0-based).
    pub fn calculate_delay(config: &RetryConfig, attempt: u32) -> Duration {
        let pow = config.backoff_multiplier.powi(attempt as i32);
        let delay_ms = ((config.initial_backoff_ms as f32 * pow) as u64).min(config.max_backoff_ms);

        let jitter = config.jitter_factor.clamp(0.0, 1.0);
        if jitter > 0.0 {
            let mut rng = rand::rng();
            let jitter_scale: f32 = rng.random_range(-jitter..=jitter);
            let jitter_ms = (delay_ms as f32 * jitter_scale)
                .round()
                .max(-(delay_ms as f32));
            let adjusted = (delay_ms as i64 + jitter_ms as i64).max(0) as u64;
            return Duration::from_millis(adjusted);
        }

        Duration::from_millis(delay_ms)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetryError {
    #[error("maximum retry attempts exceeded")]
    MaxRetriesExceeded,
}

/// A thin async retry executor for generic operations.
#[derive(Debug, Clone, Default)]
pub struct RetryExecutor;

impl RetryExecutor {
    /// Execute an async operation with retries and backoff.
    /// The `operation` closure is invoked each attempt with the attempt index.
    pub async fn execute_with_retry<F, Fut, T>(
        config: &RetryConfig,
        mut operation: F,
    ) -> Result<T, RetryError>
    where
        F: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Result<T, ()>>,
    {
        let max = config.max_retries.max(1);
        let mut attempt: u32 = 0;
        loop {
            if let Ok(val) = operation(attempt).await {
                return Ok(val);
            }
            let is_last = attempt + 1 >= max;
            if is_last {
                return Err(RetryError::MaxRetriesExceeded);
            }
            let delay = BackoffCalculator::calculate_delay(config, attempt);
            attempt += 1;
            tokio::time::sleep(delay).await;
        }
    }

    /// Execute an operation that returns an HTTP Response with retries and backoff.
    ///
    /// Usage pattern:
    /// - `operation(attempt)`: perform one attempt (0-based). Construct and send the request,
    ///   then return the `Response`. Do any per-attempt bookkeeping (e.g., load tracking,
    ///   circuit-breaker outcome recording) inside this closure.
    /// - `should_retry(&response, attempt)`: decide if the given response should be retried
    ///   (e.g., based on HTTP status). Returning false short-circuits and returns the response.
    /// - `on_backoff(delay, next_attempt)`: called before sleeping between attempts.
    ///   Use this to record metrics.
    /// - `on_exhausted()`: called when the executor has exhausted all retry attempts.
    ///
    /// Example:
    /// ```ignore
    /// let resp = RetryExecutor::execute_response_with_retry(
    ///     &retry_cfg,
    ///     |attempt| async move {
    ///         let worker = select_cb_aware_worker()?;
    ///         let resp = send_request(worker).await;
    ///         worker.record_outcome(resp.status().is_success());
    ///         resp
    ///     },
    ///     |res, _| matches!(res.status(), StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS | StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT),
    ///     |delay, _attempt| { /* record backoff metrics */ },
    ///     || { /* record retries exhausted */ },
    /// ).await;
    /// ```
    pub async fn execute_response_with_retry<Op, Fut, ShouldRetry, OnBackoff, OnExhausted>(
        config: &RetryConfig,
        mut operation: Op,
        should_retry: ShouldRetry,
        on_backoff: OnBackoff,
        mut on_exhausted: OnExhausted,
    ) -> Response
    where
        Op: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Response>,
        ShouldRetry: Fn(&Response, u32) -> bool,
        OnBackoff: Fn(Duration, u32),
        OnExhausted: FnMut(),
    {
        let max = config.max_retries.max(1);

        let mut attempt: u32 = 0;
        loop {
            let response = operation(attempt).await;
            let is_last = attempt + 1 >= max;

            if !should_retry(&response, attempt) {
                return response;
            }

            if is_last {
                on_exhausted();
                return response;
            }

            let next_attempt = attempt + 1;
            let delay = BackoffCalculator::calculate_delay(config, attempt);
            debug!(
                attempt = attempt,
                next_attempt = next_attempt,
                delay_ms = delay.as_millis() as u64,
                "Retry backoff"
            );
            on_backoff(delay, next_attempt);
            tokio::time::sleep(delay).await;

            attempt = next_attempt;
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/retry/tests.rs"]
mod tests;
