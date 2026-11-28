//! The lead of a replica that elects.

use std::{sync::Arc, time::Instant};

use tokio::sync::watch;

use crate::{service::elector::Fence, worker_host::Lead};

/// The lead of a replica that elects: the term the role's word names, while
/// its fence admits that term now.
///
/// The word is the one the serve tells its watches: the term this replica
/// leads, if it leads one. It names none from the replica's stop on, where
/// the fence may still stand. The fence is read by the clock, as before
/// every write a leader makes: past its renew deadline it admits nothing,
/// told or not. A term the fence took and the word has not told yet is no
/// lead's: the task of that term starts at its word.
pub(crate) struct ElectedLead {
    fence: Arc<Fence>,
    led: watch::Receiver<Option<u64>>,
}

impl ElectedLead {
    /// The lead under `fence`, by the word `led` holds.
    pub(crate) fn new(fence: Arc<Fence>, led: watch::Receiver<Option<u64>>) -> Self {
        Self { fence, led }
    }
}

impl Lead for ElectedLead {
    /// One read of the word and one of the fence, and no wait: the word's
    /// lock is held for the read alone, and its writer holds it for a write
    /// of the word alone.
    fn term(&self) -> Option<u64> {
        let told = (*self.led.borrow())?;
        (self.fence.leader_term(Instant::now()) == Some(told)).then_some(told)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/service/controller/worker_objects/lead/tests.rs"]
pub(super) mod tests;
