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

use self::flags::{refuse_settings, Credential};
pub(super) use self::{flags::DiscoveryFlags, host::Host};
use super::{election::ElectionObservation, readiness::reached, ConfigError};
use crate::service::elector::Fence;

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
        let runtime = RuntimeDiscoveryConfig::from_config(&config, Some(roles))
            .map_err(|error| refuse_settings(&config, roles, error))?;
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

    /// Runs the provider while this replica leads, until the stop. The
    /// election's own end closes `standing`: the provider ends there, and
    /// the task still stays until the stop, as the serve's other tasks do.
    pub(crate) async fn run(
        &self,
        standing: watch::Receiver<ElectionObservation>,
        stop: watch::Receiver<bool>,
    ) {
        let start = || start_service_discovery(self.settings.clone(), self.host.clone());
        self.run_with(standing, stop, start).await;
    }

    /// [`Discovery::run`] over `start`, the start of the provider.
    async fn run_with<F>(
        &self,
        mut standing: watch::Receiver<ElectionObservation>,
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
            let (leads, turn) = self.fence.term_and_turn(Instant::now());
            let Some(term) = leads else {
                tokio::select! {
                    () = &mut stopped => return,
                    open = Self::turned(turn, &mut standing) => {
                        if !open {
                            break;
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
                        break;
                    }
                    continue;
                }
                started = self.start(term, &start, ended == Some(term)) => {
                    let Some(task) = started else {
                        continue;
                    };
                    // The pause is owed once, to the start behind the end.
                    ended = None;
                    task
                }
            };
            let open = tokio::select! {
                biased;
                () = &mut stopped => None,
                open = &mut left => Some(open),
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
            match open {
                None => return,
                Some(true) => {}
                Some(false) => break,
            }
        }
        // The election's watch closed: its thread is gone, and the serve
        // says so itself. No provider runs here. The task stays until the
        // stop, so the serve does not read its end as a stopped listener.
        stopped.await;
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

    /// Waits for the election's next word, or for `turn`, the instant the
    /// clock alone turns the fence's answer: a lead the election renews no
    /// more ends there with no word. Returns whether the watch is still
    /// open.
    async fn turned(
        turn: Option<Instant>,
        standing: &mut watch::Receiver<ElectionObservation>,
    ) -> bool {
        tokio::select! {
            changed = standing.changed() => changed.is_ok(),
            () = reached(turn) => true,
        }
    }

    /// Waits for a different term, returning whether the watch is still open.
    async fn left(&self, term: u64, standing: &mut watch::Receiver<ElectionObservation>) -> bool {
        loop {
            // One read of the fence gives the term and its next turn. No
            // release and no end of the window falls between the two, so a
            // lead always comes with the instant to sleep to.
            let (leads, turn) = self.fence.term_and_turn(Instant::now());
            if leads != Some(term) {
                return true;
            }
            if !Self::turned(turn, standing).await {
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
