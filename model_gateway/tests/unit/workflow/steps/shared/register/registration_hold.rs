use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use tokio::sync::Notify;

/// The hold on one URL's registry step: `reached` fires when the step is
/// in its window, `release` lets it write.
pub(crate) struct RegistrationHold {
    pub(crate) reached: Notify,
    pub(crate) release: Notify,
}

static HOLDS: LazyLock<DashMap<String, Arc<RegistrationHold>>> = LazyLock::new(DashMap::new);

/// Places a hold on the next registry step for `url`.
pub(crate) fn place(url: &str) -> Arc<RegistrationHold> {
    let hold = Arc::new(RegistrationHold {
        reached: Notify::new(),
        release: Notify::new(),
    });
    HOLDS.insert(url.to_string(), Arc::clone(&hold));
    hold
}

/// The step's side: under a hold placed for `url`, says the window is
/// reached and waits for the release; without one, returns at once.
pub(crate) async fn honour(url: &str) {
    let Some((_, hold)) = HOLDS.remove(url) else {
        return;
    };
    hold.reached.notify_one();
    hold.release.notified().await;
}
