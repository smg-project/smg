pub(crate) mod bounds;
pub mod builder;
pub mod types;
pub(crate) mod validation;

pub use builder::*;
pub use types::*;
pub(crate) use validation::parse_listen_ip;
pub use validation::{
    available_cpus, bind_socket_addr, dp_minimum_tokens_scheduler_scope_warning, parse_bind_host,
    runtime_worker_threads_oversubscription, validate_mesh_listener_port,
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

pub type ConfigResult<T> = Result<T, ConfigError>;

#[cfg(test)]
mod config_error_tests {
    use super::ConfigError;

    #[test]
    fn the_constructor_builds_the_same_error_as_the_literal() {
        let built = ConfigError::invalid_value("port", 0u16, "Must be > 0");
        let literal = ConfigError::InvalidValue {
            field: "port".to_string(),
            value: 0u16.to_string(),
            reason: "Must be > 0".to_string(),
        };
        assert_eq!(built.to_string(), literal.to_string());
        assert_eq!(
            built.to_string(),
            "Invalid value for field 'port': 0 - Must be > 0"
        );
        let ConfigError::InvalidValue {
            field,
            value,
            reason,
        } = built
        else {
            panic!("an invalid value");
        };
        assert_eq!(
            (field.as_str(), value.as_str(), reason.as_str()),
            ("port", "0", "Must be > 0")
        );
    }
}
