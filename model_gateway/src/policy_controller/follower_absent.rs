//! The follower's names on a target the controller crate's client does not
//! build for. Startup refuses a controller's address there, so nothing here
//! runs: it keeps the gateway's other modules one text for every target.

use std::{sync::Arc, time::Instant};

use smg_policy_controller::{
    door::{Door, Request},
    handle::Handle,
};

use super::{feed::Feed, host::NoAnswer};
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

    #[expect(
        clippy::unnecessary_wraps,
        clippy::unused_self,
        reason = "the follower's settings read their files; none is read here"
    )]
    pub(crate) fn readable(&self) -> Result<(), String> {
        Ok(())
    }
}

/// The link's name on a target that dials none.
#[derive(Default)]
pub(crate) struct Linked;

impl Linked {
    #[expect(
        clippy::unused_async,
        reason = "the follower's link waits for its answer; nothing is asked here"
    )]
    pub(crate) async fn asked(
        &self,
        _request: &Request<'_>,
        _now: Instant,
    ) -> Result<Handle<Arc<Door>>, NoAnswer> {
        Err(NoAnswer::NoLink)
    }

    #[expect(
        clippy::unused_self,
        reason = "the follower's link reads its task's progress; nothing is read here"
    )]
    pub(crate) fn fleet(&self) -> bool {
        false
    }
}

/// A follower that follows nothing.
pub(crate) struct Follower;

impl Follower {
    pub(crate) fn start(_settings: Settings, _feed: Feed) -> Self {
        Self
    }

    #[expect(
        clippy::unused_self,
        reason = "the follower reads its task's progress; nothing is read here"
    )]
    pub(crate) fn followed(&self) -> Followed {
        Followed::Nothing
    }

    #[expect(
        clippy::unused_self,
        reason = "the follower hands out the link its task holds; none is held here"
    )]
    pub(crate) fn linked(&self) -> Arc<Linked> {
        Arc::new(Linked)
    }

    #[expect(
        clippy::unused_async,
        reason = "the follower's stop awaits its task; nothing runs here"
    )]
    pub(crate) async fn stop(self) {}
}
