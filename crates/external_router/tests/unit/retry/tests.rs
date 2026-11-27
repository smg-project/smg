use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use axum::{http::StatusCode, response::IntoResponse};

use super::*;
use crate::error::extract_error_code_from_response;

fn base_retry_config() -> RetryConfig {
    RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 4,
        backoff_multiplier: 2.0,
        jitter_factor: 0.0,
    }
}

#[test]
fn test_backoff_no_jitter_progression_and_cap() {
    let cfg = RetryConfig {
        max_retries: 10,
        initial_backoff_ms: 100,
        max_backoff_ms: 250,
        backoff_multiplier: 2.0,
        jitter_factor: 0.0,
    };
    assert_eq!(
        BackoffCalculator::calculate_delay(&cfg, 0),
        Duration::from_millis(100)
    );
    assert_eq!(
        BackoffCalculator::calculate_delay(&cfg, 1),
        Duration::from_millis(200)
    );
    assert_eq!(
        BackoffCalculator::calculate_delay(&cfg, 2),
        Duration::from_millis(250)
    );
    assert_eq!(
        BackoffCalculator::calculate_delay(&cfg, 10),
        Duration::from_millis(250)
    );
}

#[test]
fn test_backoff_with_jitter_within_bounds() {
    let cfg = RetryConfig {
        max_retries: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 10_000,
        backoff_multiplier: 2.0,
        jitter_factor: 0.5,
    };
    let base = 400.0;
    for _ in 0..50 {
        let d = BackoffCalculator::calculate_delay(&cfg, 2).as_millis() as f32;
        assert!(d >= base * 0.5 - 1.0 && d <= base * 1.5 + 1.0);
    }
}

#[tokio::test]
async fn test_execute_with_retry_success_after_failures() {
    let cfg = base_retry_config();
    let remaining = Arc::new(AtomicU32::new(2));
    let calls = Arc::new(AtomicU32::new(0));

    let res: Result<u32, RetryError> = RetryExecutor::execute_with_retry(&cfg, {
        let remaining = remaining.clone();
        let calls = calls.clone();
        move |_attempt| {
            calls.fetch_add(1, Ordering::Relaxed);
            let remaining = remaining.clone();
            async move {
                if remaining
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1))
                    .is_ok()
                {
                    Err(())
                } else {
                    Ok(42u32)
                }
            }
        }
    })
    .await;

    assert!(res.is_ok());
    assert_eq!(res.unwrap(), 42);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn test_execute_with_retry_exhausted() {
    let cfg = base_retry_config();
    let calls = Arc::new(AtomicU32::new(0));
    let res: Result<u32, RetryError> = RetryExecutor::execute_with_retry(&cfg, {
        let calls = calls.clone();
        move |_attempt| {
            calls.fetch_add(1, Ordering::Relaxed);
            async move { Err(()) }
        }
    })
    .await;

    assert!(matches!(res, Err(RetryError::MaxRetriesExceeded)));
    assert_eq!(calls.load(Ordering::Relaxed), cfg.max_retries);
}

/// The gateway's answer for a request its timeout cut carries a status
/// the retry layer would replay and the mark that it must not.
#[test]
fn a_request_timeout_answer_is_a_504_the_retry_layer_does_not_replay() {
    let response = request_timeout_response("the worker did not answer in time");

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        extract_error_code_from_response(&response),
        REQUEST_TIMEOUT_CODE
    );
    assert!(is_retryable_status(response.status()));
    assert!(!is_retryable_response(&response));
}

#[tokio::test]
async fn test_execute_response_with_retry_success_path_and_hooks() {
    let cfg = base_retry_config();
    let remaining = Arc::new(AtomicU32::new(2));
    let calls = Arc::new(AtomicU32::new(0));
    let backoffs = Arc::new(AtomicU32::new(0));
    let exhausted = Arc::new(AtomicU32::new(0));

    let response = RetryExecutor::execute_response_with_retry(
        &cfg,
        {
            let remaining = remaining.clone();
            let calls = calls.clone();
            move |_attempt| {
                calls.fetch_add(1, Ordering::Relaxed);
                let remaining = remaining.clone();
                async move {
                    if remaining
                        .try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1))
                        .is_ok()
                    {
                        (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response()
                    } else {
                        (StatusCode::OK, "ok").into_response()
                    }
                }
            }
        },
        |res, _attempt| !res.status().is_success(),
        {
            let backoffs = backoffs.clone();
            move |_delay, _next_attempt| {
                backoffs.fetch_add(1, Ordering::Relaxed);
            }
        },
        {
            let exhausted = exhausted.clone();
            move || {
                exhausted.fetch_add(1, Ordering::Relaxed);
            }
        },
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(backoffs.load(Ordering::Relaxed), 2);
    assert_eq!(exhausted.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_execute_response_with_retry_non_retryable_short_circuit() {
    let cfg = base_retry_config();
    let calls = Arc::new(AtomicU32::new(0));
    let backoffs = Arc::new(AtomicU32::new(0));
    let exhausted = Arc::new(AtomicU32::new(0));

    let response = RetryExecutor::execute_response_with_retry(
        &cfg,
        {
            let calls = calls.clone();
            move |_attempt| {
                calls.fetch_add(1, Ordering::Relaxed);
                async move { (StatusCode::BAD_REQUEST, "bad").into_response() }
            }
        },
        |_res, _attempt| false,
        {
            let backoffs = backoffs.clone();
            move |_delay, _next_attempt| {
                backoffs.fetch_add(1, Ordering::Relaxed);
            }
        },
        {
            let exhausted = exhausted.clone();
            move || {
                exhausted.fetch_add(1, Ordering::Relaxed);
            }
        },
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(backoffs.load(Ordering::Relaxed), 0);
    assert_eq!(exhausted.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_execute_response_with_retry_exhausted_hooks() {
    let cfg = base_retry_config();
    let calls = Arc::new(AtomicU32::new(0));
    let backoffs = Arc::new(AtomicU32::new(0));
    let exhausted = Arc::new(AtomicU32::new(0));

    let response = RetryExecutor::execute_response_with_retry(
        &cfg,
        {
            let calls = calls.clone();
            move |_attempt| {
                calls.fetch_add(1, Ordering::Relaxed);
                async move { (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response() }
            }
        },
        |_res, _attempt| true,
        {
            let backoffs = backoffs.clone();
            move |_delay, _next_attempt| {
                backoffs.fetch_add(1, Ordering::Relaxed);
            }
        },
        {
            let exhausted = exhausted.clone();
            move || {
                exhausted.fetch_add(1, Ordering::Relaxed);
            }
        },
    )
    .await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(calls.load(Ordering::Relaxed), cfg.max_retries);
    assert_eq!(backoffs.load(Ordering::Relaxed), cfg.max_retries - 1);
    assert_eq!(exhausted.load(Ordering::Relaxed), 1);
}
