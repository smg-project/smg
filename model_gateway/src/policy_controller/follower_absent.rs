//! The follower's names on a target the controller crate's client does not
//! build for. Startup refuses a controller's address there, so nothing here
//! runs: it keeps the gateway's other modules one text for every target.

use std::sync::Arc;

use smg_policy_controller::store::MemoryStore;

use crate::config::RouterConfig;

/// What the follower has reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Followed {
    /// No snapshot was taken yet.
    Nothing,
    /// A snapshot was taken.
    #[expect(dead_code, reason = "no follower runs on this target")]
    Fleet,
}

/// The follower's settings. None is read on this target.
pub(crate) struct Settings;

impl Settings {
    pub(crate) fn new(_address: String, _config: &RouterConfig) -> Self {
        Self
    }
}

/// A follower that follows nothing.
pub(crate) struct Follower;

impl Follower {
    pub(crate) fn start(_settings: Settings, _store: Arc<MemoryStore>) -> Self {
        Self
    }

    pub(crate) fn followed(&self) -> Followed {
        Followed::Nothing
    }

    pub(crate) async fn stop(self) {}
}
