//! Optional observation of the engine wire for native engine statistics.
//!
//! Observers must not retain tensor frames or perform blocking I/O. No observer
//! is installed on the ordinary inference path.

use std::sync::{Arc, OnceLock};

/// Synchronous metadata callback; implementations must return without blocking I/O.
pub type WireObserver = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;
/// Shared installation slot, empty unless a statistics consumer is enabled.
pub type ObserverSlot = Arc<OnceLock<WireObserver>>;

/// Deliver metadata only when a consumer is installed.
pub fn observe(slot: &ObserverSlot, event: &str, payload: &[u8]) {
    if let Some(observer) = slot.get() {
        observer(event, payload);
    }
}
