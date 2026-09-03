//! RL control-plane configuration, embedded in the gateway's `RouterConfig`.

use serde::{Deserialize, Serialize};

/// Configuration for the RL control plane. Inert unless `enabled`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlConfig {
    /// Mount `/v1/rl/*` and build the control-plane client.
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
mod tests {
    use super::*;

    #[test]
    fn defaults_are_inert() {
        let cfg = RlConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.control_timeout_secs, 600);
        assert_eq!(cfg.fanout_concurrency, 32);
        assert_eq!(cfg.validate(), Ok(()));
    }

    #[test]
    fn serde_roundtrip_and_partial_json() {
        let cfg: RlConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.control_timeout_secs, 600);
        let back = serde_json::to_string(&cfg).unwrap();
        assert_eq!(serde_json::from_str::<RlConfig>(&back).unwrap(), cfg);
    }

    #[test]
    fn validate_rejects_zero_values_only_when_enabled() {
        let mut cfg = RlConfig {
            enabled: true,
            control_timeout_secs: 0,
            fanout_concurrency: 32,
        };
        assert!(cfg.validate().unwrap_err().contains("control_timeout_secs"));
        cfg.control_timeout_secs = 600;
        cfg.fanout_concurrency = 0;
        assert!(cfg.validate().unwrap_err().contains("fanout_concurrency"));
        cfg.enabled = false;
        assert_eq!(cfg.validate(), Ok(()));
    }
}
