//! What every engine servicer is configured with, whichever engine it serves:
//! the gRPC listener, the engines' ZMQ endpoints, the tokenizer directory and
//! the start-up bounds. The per-engine configurations embed it next to the
//! model facts they advertise, and the per-engine servers validate it the
//! same way before they bind.

use std::time::Duration;

use crate::{ServicerError, DEFAULT_ENGINE_STARTUP_CEILING, DEFAULT_ENGINE_STARTUP_TIMEOUT};

/// How a servicer binds, where its engines dial in, and how long their start
/// may take: the part of a servicer's configuration that does not depend on
/// the engine it serves.
#[derive(Debug, Clone)]
pub struct ServicerConfig {
    /// `host:port` for the gRPC listener.
    pub bind_address: String,
    /// `ipc://<path>` base for the data-plane sockets (`<path>-in.sock` and
    /// `<path>-out.sock`), owned by this process's uid.
    pub ipc_base_url: String,
    /// `tcp://host:port` the headless engine dials for the handshake, as the
    /// launcher hands it to the engine. An `ipc://<path>` endpoint is
    /// accepted too: one per test, so parallel tests never share a probed
    /// TCP port.
    pub handshake_address: String,
    /// Engines that will dial in: the engine's data-parallel size as the
    /// launcher started it.
    pub engine_count: usize,
    /// Local directory holding the model's tokenizer files, bundled by
    /// `GetTokenizer`; a servicer that matches string stops itself loads it
    /// and refuses requests carrying string stops without it.
    pub tokenizer_dir: Option<String>,
    /// Bound on silence during the engine's startup handshake: a handshake
    /// message or the lifecycle owner's report that the engine process is
    /// alive (each servicer's `note_engine_alive`) resets it; see
    /// [`DEFAULT_ENGINE_STARTUP_TIMEOUT`].
    pub engine_startup_timeout: Duration,
    /// Bound on the whole startup handshake however alive the engine is;
    /// `None` leaves that to the silence bound and the lifecycle owner. See
    /// [`DEFAULT_ENGINE_STARTUP_CEILING`].
    pub engine_startup_ceiling: Option<Duration>,
}

impl ServicerConfig {
    /// A servicer on `bind_address` for one engine that dials
    /// `handshake_address`, with its data plane under `ipc_base_url`, no
    /// tokenizer directory, and the crate's start-up bounds
    /// ([`DEFAULT_ENGINE_STARTUP_TIMEOUT`], [`DEFAULT_ENGINE_STARTUP_CEILING`]).
    pub fn new(
        bind_address: impl Into<String>,
        ipc_base_url: impl Into<String>,
        handshake_address: impl Into<String>,
    ) -> Self {
        Self {
            bind_address: bind_address.into(),
            ipc_base_url: ipc_base_url.into(),
            handshake_address: handshake_address.into(),
            engine_count: 1,
            tokenizer_dir: None,
            engine_startup_timeout: DEFAULT_ENGINE_STARTUP_TIMEOUT,
            engine_startup_ceiling: Some(DEFAULT_ENGINE_STARTUP_CEILING),
        }
    }

    /// The checks every servicer runs before it binds: the shape of the
    /// three endpoints, at least one engine, and positive bounds. The first
    /// check that fails names its field.
    pub(crate) fn validate(&self) -> Result<(), ServicerError> {
        let invalid = |message: &str| ServicerError::InvalidConfig(message.to_string());
        if self.bind_address.rsplit_once(':').is_none() {
            return Err(invalid("bind_address must be host:port"));
        }
        if !self.ipc_base_url.starts_with("ipc://") {
            return Err(invalid("ipc_base_url must be ipc://<path>"));
        }
        if !self.handshake_address.starts_with("tcp://")
            && !self.handshake_address.starts_with("ipc://")
        {
            return Err(invalid(
                "handshake_address must be tcp://host:port or ipc://<path>",
            ));
        }
        if self.engine_count == 0 {
            return Err(invalid("engine_count must be positive"));
        }
        if self.engine_startup_timeout.is_zero() {
            return Err(invalid("engine_startup_timeout must be positive"));
        }
        if self
            .engine_startup_ceiling
            .is_some_and(|ceiling| ceiling.is_zero())
        {
            return Err(invalid("engine_startup_ceiling must be positive when set"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ServicerConfig;
    use crate::{ServicerError, DEFAULT_ENGINE_STARTUP_CEILING, DEFAULT_ENGINE_STARTUP_TIMEOUT};

    fn valid() -> ServicerConfig {
        ServicerConfig::new("127.0.0.1:0", "ipc:///tmp/engine", "tcp://127.0.0.1:1")
    }

    /// What a lifecycle owner gets when it says nothing beyond the endpoints:
    /// one engine, no tokenizer directory, and the crate's start-up bounds.
    #[test]
    fn a_new_config_keeps_the_endpoints_and_carries_the_crate_defaults() {
        let config = valid();
        assert_eq!(config.bind_address, "127.0.0.1:0");
        assert_eq!(config.ipc_base_url, "ipc:///tmp/engine");
        assert_eq!(config.handshake_address, "tcp://127.0.0.1:1");
        assert_eq!(config.engine_count, 1);
        assert_eq!(config.tokenizer_dir, None);
        assert_eq!(
            config.engine_startup_timeout,
            DEFAULT_ENGINE_STARTUP_TIMEOUT
        );
        assert_eq!(
            config.engine_startup_ceiling,
            Some(DEFAULT_ENGINE_STARTUP_CEILING)
        );
        config.validate().expect("the defaults are valid");
    }

    /// An IPC handshake endpoint (the tests' choice) and no ceiling are valid
    /// too.
    #[test]
    fn an_ipc_handshake_and_no_ceiling_are_valid() {
        let mut config = valid();
        config.handshake_address = "ipc:///tmp/handshake".to_string();
        config.engine_startup_ceiling = None;
        config.validate().expect("valid");
    }

    /// Each check refuses with the message that names its field, the texts
    /// the lifecycle owner shows as they were.
    #[test]
    fn validate_names_the_field_at_fault() {
        let refusal = |config: ServicerConfig| match config.validate() {
            Err(ServicerError::InvalidConfig(message)) => message,
            other => panic!("expected an invalid config, got {other:?}"),
        };
        let with = |edit: fn(&mut ServicerConfig)| {
            let mut config = valid();
            edit(&mut config);
            config
        };
        assert_eq!(
            refusal(with(|config| config.bind_address = "nope".to_string())),
            "bind_address must be host:port"
        );
        assert_eq!(
            refusal(with(
                |config| config.ipc_base_url = "/tmp/not-ipc".to_string()
            )),
            "ipc_base_url must be ipc://<path>"
        );
        assert_eq!(
            refusal(with(|config| {
                config.handshake_address = "udp://127.0.0.1:1".to_string();
            })),
            "handshake_address must be tcp://host:port or ipc://<path>"
        );
        assert_eq!(
            refusal(with(|config| config.engine_count = 0)),
            "engine_count must be positive"
        );
        assert_eq!(
            refusal(with(|config| config.engine_startup_timeout = Duration::ZERO)),
            "engine_startup_timeout must be positive"
        );
        assert_eq!(
            refusal(with(|config| {
                config.engine_startup_ceiling = Some(Duration::ZERO);
            })),
            "engine_startup_ceiling must be positive when set"
        );
    }
}
