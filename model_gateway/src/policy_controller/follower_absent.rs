//! The follower's names on a target the controller crate's client does not
//! build for. Startup refuses a controller's address there, so nothing here
//! runs: it keeps the gateway's other modules one text for every target.

use super::feed::Feed;
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
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the follower's settings refuse TLS files that do not pair; none is read here"
    )]
    pub(crate) fn new(_address: String, _config: &RouterConfig) -> Result<Self, String> {
        Ok(Self)
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
        clippy::unused_async,
        reason = "the follower's stop awaits its task; nothing runs here"
    )]
    pub(crate) async fn stop(self) {}
}
