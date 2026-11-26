use tokio::sync::watch;

use super::*;

/// A channel nobody publishes on, with an empty snapshot: a case's
/// deciders read no load report.
pub(in crate::service::controller) fn silent() -> LoadReceiver {
    watch::channel(Arc::new(LoadSnapshot::default())).1
}

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
