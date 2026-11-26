//! The load reports a deciding thread gives its frames' requests.
//!
//! The replica's load monitor publishes its workers' reports as one snapshot
//! on one channel ([`LoadMonitor::loads`](super::worker_objects::LoadMonitor::loads)).
//! Every deciding thread holds that channel from its start and keeps the
//! newest snapshot at hand ([`FrameLoads`]). A frame's request is given it
//! ([`FrameRequest::reading`](super::frame_request::FrameRequest::reading)),
//! so its decision reads the reports the monitor last published and waits
//! for no poll.

use std::sync::Arc;

use smg_worker::load::state::{LoadReceiver, LoadSnapshot};

/// The newest snapshot of the replica's load monitor, as one deciding
/// thread holds it between its frames.
///
/// A read looks at the channel's count of publications, and takes the
/// snapshot again only behind a publication. A monitor that is gone leaves
/// its last snapshot standing here.
#[derive(Debug)]
pub(crate) struct FrameLoads {
    channel: LoadReceiver,
    newest: Arc<LoadSnapshot>,
}

impl FrameLoads {
    /// The thread's hold of `channel`, with the snapshot that stands on it.
    pub(crate) fn of(mut channel: LoadReceiver) -> Self {
        let newest = Arc::clone(&channel.borrow_and_update());
        Self { channel, newest }
    }

    /// The newest snapshot the monitor published.
    pub(crate) fn newest(&mut self) -> &Arc<LoadSnapshot> {
        match self.channel.has_changed() {
            Ok(false) => {}
            // A publication, or a monitor that is gone: the channel says
            // nothing of a last publication then, so its snapshot is read.
            Ok(true) | Err(_) => {
                self.newest = Arc::clone(&self.channel.borrow_and_update());
            }
        }
        &self.newest
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/frame_loads/tests.rs"]
pub(super) mod tests;
