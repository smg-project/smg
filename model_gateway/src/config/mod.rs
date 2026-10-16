pub(crate) mod bounds;
pub mod builder;
pub mod types;
pub(crate) mod validation;

pub use builder::*;
pub use types::*;
pub(crate) use validation::parse_listen_ip;
pub use validation::{
    available_cpus, bind_socket_addr, parse_bind_host, runtime_worker_threads_oversubscription,
    validate_mesh_listener_port, validate_mesh_server_name, validate_worker_url,
    RUNTIME_WORKER_THREADS_CEILING, RUNTIME_WORKER_THREADS_PER_CPU,
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

pub type ConfigResult<T> = Result<T, ConfigError>;
