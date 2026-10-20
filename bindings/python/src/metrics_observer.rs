//! Bounded delivery of metrics metadata outside inference runtime threads.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, SyncSender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_EVENTS: usize = 4096;
const MAX_BYTES: usize = 64 * 1024 * 1024;

pub(crate) type Event = (String, Vec<u8>, f64);

#[derive(Default)]
struct State {
    stopped: AtomicBool,
    failed: AtomicBool,
    bytes: AtomicUsize,
}

/// A nonblocking producer and a dedicated consumer; overflow invalidates collection.
pub(crate) struct MetricsObserver {
    sender: SyncSender<Event>,
    max_bytes: usize,
    state: Arc<State>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl MetricsObserver {
    /// Start one callback worker with bounded event count and retained payload bytes.
    pub(crate) fn new(callback: impl Fn(&[Event]) + Send + 'static) -> std::io::Result<Self> {
        Self::with_limits(callback, MAX_EVENTS, MAX_BYTES)
    }

    /// Construct the same worker with smaller limits for deterministic overload tests.
    fn with_limits(
        callback: impl Fn(&[Event]) + Send + 'static,
        capacity: usize,
        max_bytes: usize,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Event>(capacity);
        let state = Arc::new(State::default());
        let consumer = state.clone();
        let worker = thread::Builder::new()
            .name("vllm-metrics".into())
            .spawn(move || {
                while !consumer.stopped.load(Ordering::Acquire) {
                    if consumer.failed.load(Ordering::Acquire) {
                        callback(&[("overflow".into(), Vec::new(), timestamp())]);
                        break;
                    }
                    match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(first) => {
                            let mut batch = vec![first];
                            batch.extend(receiver.try_iter().take(255));
                            let bytes = batch.iter().map(|(_, payload, _)| payload.len()).sum();
                            consumer.bytes.fetch_sub(bytes, Ordering::Relaxed);
                            callback(&batch);
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })?;
        // The payload limit is stored alongside the sender rather than in each event.
        Ok(Self {
            sender,
            state,
            worker: Mutex::new(Some(worker)),
            max_bytes,
        })
    }

    /// Copy metadata without acquiring the GIL or waiting for the consumer.
    pub(crate) fn observe(&self, event: &str, payload: &[u8]) {
        if self.state.stopped.load(Ordering::Acquire) || self.state.failed.load(Ordering::Acquire) {
            return;
        }
        let time = timestamp();
        if self
            .state
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bytes| {
                bytes
                    .checked_add(payload.len())
                    .filter(|&total| total <= self.max_bytes)
            })
            .is_err()
        {
            self.state.failed.store(true, Ordering::Release);
            return;
        }
        if self
            .sender
            .try_send((event.into(), payload.to_vec(), time))
            .is_err()
        {
            self.state.bytes.fetch_sub(payload.len(), Ordering::Relaxed);
            self.state.failed.store(true, Ordering::Release);
        }
    }

    /// Let an idle worker exit while the server drains its connections.
    pub(crate) fn signal_stop(&self) {
        self.state.stopped.store(true, Ordering::Release);
    }

    /// Stop outside the GIL, respecting the remaining server shutdown budget.
    pub(crate) fn close(&self, timeout: Duration) -> bool {
        self.signal_stop();
        let start = Instant::now();
        let mut slot = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(worker) = slot.as_ref() {
            while !worker.is_finished() {
                if start.elapsed() >= timeout {
                    return false;
                }
                thread::sleep(
                    Duration::from_millis(1).min(timeout.saturating_sub(start.elapsed())),
                );
            }
        }
        if let Some(worker) = slot.take() {
            let _ = worker.join();
        }
        true
    }
}

impl Drop for MetricsObserver {
    /// Signal shutdown without joining while a Python destructor may hold the GIL.
    fn drop(&mut self) {
        self.state.stopped.store(true, Ordering::Release);
    }
}

/// Capture wall time at the inference event, excluding queue and GIL delays.
fn timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stalled consumer must not stall producers or silently lose accounting.
    #[test]
    fn blocked_callback_does_not_block_producers_and_overflow_is_terminal() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (events_tx, events_rx) = mpsc::channel();
        let observer = MetricsObserver::with_limits(
            move |events| {
                for (event, _, time) in events {
                    if event == "first" {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    }
                    events_tx.send((event.to_owned(), *time)).unwrap();
                }
            },
            1,
            16,
        )
        .unwrap();
        observer.observe("first", b"1");
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        observer.observe("queued", b"2");
        observer.observe("full", b"3");
        observer.observe("ignored", b"4");
        release_tx.send(()).unwrap();
        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(2)).unwrap().0,
            "first"
        );
        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(2)).unwrap().0,
            "overflow"
        );
        assert!(observer.close(Duration::from_secs(2)));
        assert!(events_rx.try_recv().is_err());
    }

    /// Oversized metadata fails collection before allocating a retained copy.
    #[test]
    fn payload_limit_fails_without_retaining_large_frames() {
        let (tx, rx) = mpsc::channel();
        let observer = MetricsObserver::with_limits(
            move |events| {
                tx.send(events[0].0.clone()).unwrap();
            },
            8,
            2,
        )
        .unwrap();
        observer.observe("output", b"123");
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), "overflow");
        assert_eq!(observer.state.bytes.load(Ordering::Relaxed), 0);
        assert!(observer.close(Duration::from_secs(2)));
    }

    /// Worker scheduling must not alter event order or measured latency.
    #[test]
    fn delivery_preserves_event_order_and_producer_time() {
        let (tx, rx) = mpsc::channel();
        let observer = MetricsObserver::new(move |events| {
            for (event, _, time) in events {
                tx.send((event.to_owned(), *time)).unwrap();
            }
        })
        .unwrap();
        let start = timestamp();
        observer.observe("add", b"a");
        observer.observe("consume", b"a");
        let end = timestamp();
        for expected in ["add", "consume"] {
            let (event, time) = rx.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(event, expected);
            assert!((start..=end).contains(&time));
        }
        assert!(observer.close(Duration::from_secs(2)));
    }
    /// A blocked Python callback must not remove the server's shutdown deadline.
    #[test]
    fn close_respects_timeout_and_can_be_retried() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let observer = MetricsObserver::new(move |_| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
        observer.observe("add", b"");
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(!observer.close(Duration::ZERO));
        release_tx.send(()).unwrap();
        assert!(observer.close(Duration::from_secs(2)));
    }
}
