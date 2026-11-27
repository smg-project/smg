//! RL control-plane configuration, embedded in the gateway's `RouterConfig`.

use serde::{Deserialize, Serialize};

/// Configuration for the RL control plane. Inert unless `enabled`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlConfig {
    /// Mount `/v1/rl/*`.
    pub enabled: bool,
    /// Total timeout for one proxied engine call (refits can take minutes).
    pub control_timeout_secs: u64,
    /// Maximum concurrent engine calls during a fan-out.
    pub fanout_concurrency: usize,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            control_timeout_secs: 600,
            fanout_concurrency: 32,
        }
    }
}

impl RlConfig {
    /// Validate the values that matter when the control plane is enabled.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.control_timeout_secs == 0 {
            return Err("rl.control_timeout_secs must be >= 1".to_string());
        }
        if self.fanout_concurrency == 0 {
            return Err("rl.fanout_concurrency must be >= 1".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/config/tests.rs"]
mod tests;
