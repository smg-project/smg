pub(crate) mod bounds;
pub mod builder;
mod policy_document;
pub(crate) mod redacted;
pub mod secrets;
#[cfg(test)]
#[path = "../../tests/unit/config/test_support.rs"]
pub(crate) mod test_support;
pub mod types;
pub(crate) mod validation;
mod worker_settings;

pub use builder::*;
use smg_service_discovery::DiscoveryConfigError;
pub use types::*;
pub(crate) use validation::parse_listen_ip;
pub use validation::{
    available_cpus, bind_socket_addr, dp_aware_deprecation_warning,
    dp_minimum_tokens_scheduler_scope_warning, parse_bind_host,
    runtime_worker_threads_oversubscription, unused_credentials_warning,
    unused_credentials_warnings, validate_encode_policy, validate_mesh_listener_port,
    validate_mesh_server_name, validate_worker_url, RUNTIME_WORKER_THREADS_CEILING,
    RUNTIME_WORKER_THREADS_PER_CPU,
};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required field: {field}")]
    MissingRequired { field: String },
}

impl ConfigError {
    /// An invalid value of `field`: the `value` as it reads, and the `reason`.
    pub fn invalid_value(
        field: impl Into<String>,
        value: impl std::fmt::Display,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidValue {
            field: field.into(),
            value: value.to_string(),
            reason: reason.into(),
        }
    }
}

impl From<DiscoveryConfigError> for ConfigError {
    fn from(error: DiscoveryConfigError) -> Self {
        match error {
            DiscoveryConfigError::ValidationFailed { reason } => Self::ValidationFailed { reason },
            DiscoveryConfigError::InvalidValue {
                field,
                value,
                reason,
            } => Self::InvalidValue {
                field,
                value,
                reason,
            },
        }
    }
}

pub type ConfigResult<T> = Result<T, ConfigError>;

#[cfg(test)]
#[path = "../../tests/unit/config/config_error_tests.rs"]
mod config_error_tests;
