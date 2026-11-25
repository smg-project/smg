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
#[cfg(test)]
use tokio::sync::watch;

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

/// A channel nobody publishes on, with an empty snapshot: a case's
/// deciders read no load report.
#[cfg(test)]
pub(crate) fn silent() -> LoadReceiver {
    watch::channel(Arc::new(LoadSnapshot::default())).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_holds_the_snapshot_the_monitor_last_published() {
        let first = Arc::new(LoadSnapshot::default());
        let (monitor, channel) = watch::channel(Arc::clone(&first));
        let mut loads = FrameLoads::of(channel);
        assert!(Arc::ptr_eq(loads.newest(), &first));
        // A publication: the next read holds it, and so does the one after.
        let second = Arc::new(LoadSnapshot::default());
        monitor.send_replace(Arc::clone(&second));
        assert!(Arc::ptr_eq(loads.newest(), &second));
        assert!(Arc::ptr_eq(loads.newest(), &second));
        // Two publications between two reads: the newest alone is held.
        let third = Arc::new(LoadSnapshot::default());
        monitor.send_replace(Arc::new(LoadSnapshot::default()));
        monitor.send_replace(Arc::clone(&third));
        assert!(Arc::ptr_eq(loads.newest(), &third));
    }

    /// A thread that takes its channel behind a publication starts with
    /// that snapshot: the channel counts no publication against a reader
    /// that came after it, so the thread reads what stands at its start.
    #[test]
    fn a_publication_ahead_of_the_threads_start_is_the_one_it_starts_with() {
        let (monitor, _first) = watch::channel(Arc::new(LoadSnapshot::default()));
        let published = Arc::new(LoadSnapshot::default());
        monitor.send_replace(Arc::clone(&published));
        let mut loads = FrameLoads::of(monitor.subscribe());
        assert!(Arc::ptr_eq(loads.newest(), &published));
    }

    #[test]
    fn a_monitor_that_is_gone_leaves_its_last_snapshot() {
        let (monitor, channel) = watch::channel(Arc::new(LoadSnapshot::default()));
        let mut loads = FrameLoads::of(channel);
        // The last publication, which the thread has not read, and the
        // monitor's end behind it.
        let last = Arc::new(LoadSnapshot::default());
        monitor.send_replace(Arc::clone(&last));
        drop(monitor);
        assert!(Arc::ptr_eq(loads.newest(), &last));
        assert!(Arc::ptr_eq(loads.newest(), &last));
    }
}
