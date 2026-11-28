//! Expected-wait arithmetic shared by routing and admission.
//!
//! Callers own signal validation and missing-data fallbacks. This module only
//! evaluates the common score once those inputs have been resolved.

/// Default KV-pressure time penalty, in seconds.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;
/// Token estimate for a request without a known token count.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;
/// Fallback aggregate throughput, in tokens per second.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;

/// Prepared expected-wait score for one worker load snapshot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedWait {
    queued_tokens: f64,
    throughput: f64,
    kv_wait: f64,
}

impl ExpectedWait {
    pub(crate) fn new(queued_tokens: f64, throughput: f64, usage: f64, weight: f64) -> Self {
        let k = usage.clamp(0.0, 0.999);
        Self {
            queued_tokens,
            throughput,
            kv_wait: weight * k / (1.0 - k),
        }
    }

    pub(crate) fn seconds(self, dispatched_tokens: u64) -> f64 {
        (self.queued_tokens + dispatched_tokens as f64) / self.throughput + self.kv_wait
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/expected_wait/tests.rs"]
mod tests;
