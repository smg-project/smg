//! The elected controller's discovery task. The election wakes it, and
//! the fence tells it which term the replica leads under. A lead the
//! election renews no more ends by the fence's clock, with no further word.

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
use super::{election::Standing, readiness::reached, ConfigError};
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
        standing: watch::Receiver<Standing>,
        stop: watch::Receiver<bool>,
    ) {
        let start = || start_service_discovery(self.settings.clone(), self.host.clone());
        self.run_with(standing, stop, start).await;
    }

    /// [`Discovery::run`] over `start`, the start of the provider.
    async fn run_with<F>(
        &self,
        mut standing: watch::Receiver<Standing>,
        mut stop: watch::Receiver<bool>,
        start: impl Fn() -> F + Send + Sync,
    ) where
        F: Future<Output = Started> + Send,
    {
        let (provider, _) = self.provider();
        let stopped = async {
            let _ = stop.wait_for(|stop| *stop).await;
        };
        tokio::pin!(stopped);
        // The term whose provider ended by itself: its next start waits the
        // check interval first, as a failed start does.
        let mut ended = None;
        loop {
            let now = Instant::now();
            let Some(term) = self.fence.leader_term(now) else {
                tokio::select! {
                    () = &mut stopped => return,
                    open = self.turned(now, &mut standing) => {
                        if !open {
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
            let mut task = tokio::select! {
                biased;
                () = &mut stopped => return,
                open = &mut left => {
                    if !open {
                        return;
                    }
                    continue;
                }
                started = self.start(term, &start, ended == Some(term)) => {
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
                end = &mut task.0 => {
                    // A provider that stops by itself would leave the
                    // workers as they stood: it is said and started again.
                    match end {
                        Ok(()) => error!(
                            provider,
                            "worker discovery ended by itself and starts again after the check interval"
                        ),
                        Err(error) => error!(
                            provider,
                            %error,
                            "worker discovery failed and starts again after the check interval"
                        ),
                    }
                    ended = Some(term);
                    continue;
                }
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

    /// The provider's name in the logs and its check interval.
    fn provider(&self) -> (&'static str, Duration) {
        match &self.settings {
            RuntimeDiscoveryConfig::Kubernetes(config) => ("kubernetes", config.check_interval),
            RuntimeDiscoveryConfig::File(config) => ("file", config.check_interval),
            RuntimeDiscoveryConfig::Slurm(config) => ("slurm", config.check_interval),
            RuntimeDiscoveryConfig::Consul(config) => ("consul", config.check_interval),
            // The enum is open to providers this build does not hold.
            _ => (
                "other",
                Duration::from_secs(DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS),
            ),
        }
    }

    /// Waits for the election's next word, or for the instant the clock
    /// alone turns the fence's answer as it stood at `now`: a lead the
    /// election renews no more ends there with no word. Returns whether
    /// the watch is still open.
    async fn turned(&self, now: Instant, standing: &mut watch::Receiver<Standing>) -> bool {
        tokio::select! {
            changed = standing.changed() => changed.is_ok(),
            () = reached(self.fence.turns_at(now)) => true,
        }
    }

    /// Waits for a different term, returning whether the watch is still open.
    async fn left(&self, term: u64, standing: &mut watch::Receiver<Standing>) -> bool {
        loop {
            // One read of the clock serves the term and its next turn, so
            // no end of the window falls between two reads.
            let now = Instant::now();
            if self.fence.leader_term(now) != Some(term) {
                return true;
            }
            if !self.turned(now, standing).await {
                return false;
            }
        }
    }

    /// Starts the provider while the replica leads under `term`. A start
    /// that fails is said and tried again after the check interval, and
    /// `paused` holds the first try back by the same interval.
    async fn start<F>(
        &self,
        term: u64,
        start: impl Fn() -> F + Send,
        mut paused: bool,
    ) -> Option<ProviderTask>
    where
        F: Future<Output = Started> + Send,
    {
        let (provider, interval) = self.provider();
        loop {
            if paused {
                time::sleep(interval).await;
            }
            if self.leader_term() != Some(term) {
                return None;
            }
            match start().await {
                Ok(task) => return Some(ProviderTask(task)),
                Err(error) => error!(provider, %error, "worker discovery could not start"),
            }
            paused = true;
        }
    }
}

/// What the provider's start returns: its task, or why it did not start.
type Started = Result<JoinHandle<()>, Box<dyn Error + Send + Sync>>;

/// If the serve is cancelled, its provider must stop with it.
struct ProviderTask(JoinHandle<()>);

impl Drop for ProviderTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests;
