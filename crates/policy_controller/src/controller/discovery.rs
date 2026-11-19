//! The elected controller's discovery task. The election wakes it, and
//! the fence tells it which term the replica leads under.

mod flags;
mod host;

use std::{
    error::Error,
    fmt,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use smg_service_discovery::{
    start_service_discovery, DiscoveryConfig, Roles, RuntimeDiscoveryConfig,
    DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS,
};
use tokio::{sync::watch, task::JoinHandle, time};
use tracing::error;

use self::flags::Credential;
pub(super) use self::{flags::DiscoveryFlags, host::Host};
use super::{election::Standing, ConfigError};
use crate::elector::Fence;

/// The selected worker discovery provider and the credential it stamps.
#[derive(Clone)]
pub struct DiscoverySettings(Box<Settings>);

#[derive(Clone)]
struct Settings {
    config: DiscoveryConfig,
    roles: Roles,
    runtime: RuntimeDiscoveryConfig,
    credential: Option<Credential>,
}

impl DiscoverySettings {
    pub(super) fn new(
        config: DiscoveryConfig,
        roles: Option<Roles>,
        credential: Option<Credential>,
    ) -> Result<Self, ConfigError> {
        // Providers that need roles take regular workers by default.
        let roles = roles.unwrap_or(Roles::Regular);
        let runtime =
            RuntimeDiscoveryConfig::from_config(&config, Some(roles)).map_err(|error| {
                ConfigError::DiscoveryFlag {
                    flag: "--discovery-provider",
                    reason: error.to_string(),
                }
            })?;
        Ok(Self(Box::new(Settings {
            config,
            roles,
            runtime,
            credential,
        })))
    }

    pub(super) fn runtime(&self) -> &RuntimeDiscoveryConfig {
        &self.0.runtime
    }

    pub(super) fn credential(&self) -> Option<&Credential> {
        self.0.credential.as_ref()
    }
}

impl fmt::Debug for DiscoverySettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscoverySettings")
            .field("runtime", self.runtime())
            .field("roles", &self.0.roles)
            .field("credential", &self.credential())
            .finish()
    }
}

impl PartialEq for DiscoverySettings {
    fn eq(&self, other: &Self) -> bool {
        // Compare the supplied settings. Some providers resolve environment
        // and file values into private runtime fields.
        self.0.config == other.0.config
            && self.0.roles == other.0.roles
            && self.0.credential == other.0.credential
    }
}

impl Eq for DiscoverySettings {}

/// Worker discovery under the controller: the provider runs while this replica leads.
pub(crate) struct Discovery {
    settings: RuntimeDiscoveryConfig,
    host: Arc<Host>,
    fence: Arc<Fence>,
}

impl Discovery {
    pub(crate) fn new(
        settings: RuntimeDiscoveryConfig,
        host: Arc<Host>,
        fence: Arc<Fence>,
    ) -> Self {
        Self {
            settings,
            host,
            fence,
        }
    }

    pub(crate) async fn run(
        &self,
        mut standing: watch::Receiver<Standing>,
        mut stop: watch::Receiver<bool>,
    ) {
        let stopped = async {
            let _ = stop.wait_for(|stop| *stop).await;
        };
        tokio::pin!(stopped);
        loop {
            let Some(term) = self.leader_term() else {
                tokio::select! {
                    () = &mut stopped => return,
                    changed = standing.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                continue;
            };

            // The watch can skip a loss followed by a take. A new term
            // ends the old provider even when both reads find a leader.
            let left = self.left(term, &mut standing);
            tokio::pin!(left);
            let start = || start_service_discovery(self.settings.clone(), self.host.clone());
            let mut task = tokio::select! {
                biased;
                () = &mut stopped => return,
                open = &mut left => {
                    if !open {
                        return;
                    }
                    continue;
                }
                started = self.start(term, start) => {
                    let Some(task) = started else {
                        continue;
                    };
                    task
                }
            };
            let open = tokio::select! {
                biased;
                () = &mut stopped => false,
                open = &mut left => open,
            };
            task.0.abort();
            let _ = (&mut task.0).await;
            if !open {
                return;
            }
        }
    }

    fn leader_term(&self) -> Option<u64> {
        self.fence.leader_term(Instant::now())
    }

    /// Wait for a different term, returning whether the watch is still open.
    async fn left(&self, term: u64, standing: &mut watch::Receiver<Standing>) -> bool {
        loop {
            if self.leader_term() != Some(term) {
                return true;
            }
            if standing.changed().await.is_err() {
                return false;
            }
        }
    }

    async fn start<F>(&self, term: u64, start: impl Fn() -> F + Send) -> Option<ProviderTask>
    where
        F: Future<Output = Result<JoinHandle<()>, Box<dyn Error + Send + Sync>>> + Send,
    {
        let (provider, interval) = match &self.settings {
            RuntimeDiscoveryConfig::Kubernetes(config) => ("kubernetes", config.check_interval),
            RuntimeDiscoveryConfig::File(config) => ("file", config.check_interval),
            RuntimeDiscoveryConfig::Slurm(config) => ("slurm", config.check_interval),
            RuntimeDiscoveryConfig::Consul(config) => ("consul", config.check_interval),
            // The enum is open to providers this build does not hold.
            _ => (
                "other",
                Duration::from_secs(DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS),
            ),
        };
        while self.leader_term() == Some(term) {
            match start().await {
                Ok(task) => return Some(ProviderTask(task)),
                Err(error) => error!(provider, %error, "worker discovery could not start"),
            }
            time::sleep(interval).await;
        }
        None
    }
}

/// If the serve is cancelled, its provider must stop with it.
struct ProviderTask(JoinHandle<()>);

impl Drop for ProviderTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests;
