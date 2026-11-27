//! What the stock controller's cases read of the follow.

use std::sync::Arc;

use smg_worker::{kv_events::KvEventCollector, registry::Generation};

use super::KvFeed;
use crate::worker_host::KvHost;

impl<G: Generation> KvFeed<G> {
    /// The collector, for the stock controller's cases of the follow:
    /// shared, so a case reads it behind the serve that takes the follow.
    pub(crate) fn collector(&self) -> &Arc<KvEventCollector<KvHost<G>>> {
        &self.collector
    }
}
