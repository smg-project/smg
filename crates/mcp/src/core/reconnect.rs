//! Reconnection management for MCP servers.

use std::time::Duration;

use tracing::{error, info, warn};

use crate::error::{McpError, McpResult};

/// Manages automatic reconnection to MCP servers with exponential backoff.
pub struct ReconnectionManager {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for ReconnectionManager {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl ReconnectionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// delay = min(base_delay * 2^(attempt-1), max_delay)
    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let exponent = attempt.saturating_sub(1).min(63);
        let factor = 2u64.saturating_pow(exponent);
        let delay = (self.base_delay.as_millis() as u64).saturating_mul(factor);
        Duration::from_millis(delay.min(self.max_delay.as_millis() as u64))
    }

    pub async fn reconnect<F, Fut, T>(&self, server_name: &str, mut connect_fn: F) -> McpResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = McpResult<T>>,
    {
        for attempt in 1..=self.max_retries {
            match connect_fn().await {
                Ok(val) => {
                    info!(
                        "Successfully reconnected to '{}' on attempt {}",
                        server_name, attempt
                    );
                    return Ok(val);
                }
                Err(e) => {
                    if attempt >= self.max_retries {
                        error!(
                            "Failed to reconnect to '{}' after {} retries: {}",
                            server_name, attempt, e
                        );
                        return Err(e);
                    }
                    let delay = self.calculate_backoff(attempt);
                    warn!(
                        "Reconnect attempt {} for '{}' failed: {}. Retrying in {:?}",
                        attempt, server_name, e, delay
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(McpError::ConnectionFailed(format!(
            "Max retries reached for {server_name}"
        )))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/core/reconnect/tests.rs"]
mod tests;
