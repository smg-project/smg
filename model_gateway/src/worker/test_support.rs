//! What the crate's tests build their workers with: a health-check
//! configuration that leaves the worker as the test built it.

use openai_protocol::worker::HealthCheckConfig;

/// The health checker disabled, everything else default: a test's worker
/// keeps the state it was built with, with no probe behind it.
pub(crate) fn no_health_check() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}
