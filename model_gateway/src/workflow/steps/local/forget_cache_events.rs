//! Telling the cache-event monitor that a worker left.

use crate::app_context::AppContext;

/// Tell the cache-event monitor the worker at `url` left: its subscription
/// stops and its blocks leave the index.
pub(super) async fn forget_cache_events(app_context: &AppContext, url: &str) {
    if let Some(monitor) = &app_context.kv_event_monitor {
        monitor.on_worker_removed(url).await;
    }
}
