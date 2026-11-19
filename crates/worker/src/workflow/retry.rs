//! The retry policies the worker workflows give their steps.

use std::time::Duration;

use wfaas::{BackoffStrategy, RetryPolicy};

/// One attempt and no wait: a step given this policy is not run again.
pub fn single_attempt() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 1,
        backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
    }
}
