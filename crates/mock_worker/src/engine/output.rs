//! Bounded, nonblocking output, including events awaiting simulated delivery.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use tokio::sync::{mpsc, Notify, OwnedSemaphorePermit, Semaphore};

use super::GenEvent;

struct Shared {
    slots: Arc<Semaphore>,
    overflow: AtomicBool,
    failed: Notify,
}

struct Buffered {
    event: GenEvent,
    at: Instant,
    // Held through the delivery delay, not just while inside the channel.
    _slot: OwnedSemaphorePermit,
}

/// A nonblocking sink; overflow closes this request to further generation.
#[derive(Clone)]
pub struct EventSender {
    tx: mpsc::Sender<Buffered>,
    shared: Arc<Shared>,
}

/// The request's output. Overflow is a terminal event, never a successful EOF.
pub struct EventReceiver {
    rx: mpsc::Receiver<Buffered>,
    shared: Arc<Shared>,
    pending: Option<Buffered>,
    ended: bool,
}

/// Create an output queue with a finite event budget (clamped to 2..=65536).
/// Tokens and terminal events both consume a slot, including delayed delivery.
pub fn output_channel(capacity: usize) -> (EventSender, EventReceiver) {
    let capacity = capacity.clamp(2, 65536);
    let (tx, rx) = mpsc::channel(capacity);
    let shared = Arc::new(Shared {
        slots: Arc::new(Semaphore::new(capacity)),
        overflow: AtomicBool::new(false),
        failed: Notify::new(),
    });
    (
        EventSender {
            tx,
            shared: shared.clone(),
        },
        EventReceiver {
            rx,
            shared,
            pending: None,
            ended: false,
        },
    )
}

impl EventSender {
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed() || self.shared.overflow.load(Ordering::Acquire)
    }

    pub fn send(&self, event: GenEvent) -> Result<(), mpsc::error::SendError<GenEvent>> {
        self.send_at(event, Instant::now())
    }

    pub(super) fn send_at(
        &self,
        event: GenEvent,
        at: Instant,
    ) -> Result<(), mpsc::error::SendError<GenEvent>> {
        if self.is_closed() {
            return Err(mpsc::error::SendError(event));
        }
        let Ok(slot) = self.shared.slots.clone().try_acquire_owned() else {
            if !self.shared.overflow.swap(true, Ordering::AcqRel) {
                // One receiver: notify_one retains a permit across registration.
                self.shared.failed.notify_one();
            }
            return Err(mpsc::error::SendError(event));
        };
        self.tx
            .try_send(Buffered {
                event,
                at,
                _slot: slot,
            })
            .map_err(|error| mpsc::error::SendError(error.into_inner().event))
    }
}

impl EventReceiver {
    fn overflow(&mut self) -> GenEvent {
        self.ended = true;
        self.pending = None;
        self.rx.close();
        while self.rx.try_recv().is_ok() {}
        GenEvent::OutputOverflow
    }

    /// Receive in FIFO order after the simulated delay; overflow interrupts it.
    pub async fn recv(&mut self) -> Option<GenEvent> {
        if self.ended {
            return None;
        }
        if self.shared.overflow.load(Ordering::Acquire) {
            return Some(self.overflow());
        }
        if self.pending.is_none() {
            self.pending = tokio::select! {
                biased;
                _ = self.shared.failed.notified() => return Some(self.overflow()),
                item = self.rx.recv() => item,
            };
        }
        // Keep the event and its permit in self if this receive is cancelled.
        let at = self.pending.as_ref()?.at;
        if at > Instant::now() {
            tokio::select! {
                biased;
                _ = self.shared.failed.notified() => return Some(self.overflow()),
                _ = tokio::time::sleep_until(at.into()) => {}
            }
        }
        if self.shared.overflow.load(Ordering::Acquire) {
            return Some(self.overflow());
        }
        self.pending.take().map(|buffered| buffered.event)
    }

    /// The pure scheduler tests only enqueue events with immediate delivery.
    #[cfg(test)]
    pub(super) fn try_recv(&mut self) -> Result<GenEvent, mpsc::error::TryRecvError> {
        if self.ended {
            return Err(mpsc::error::TryRecvError::Disconnected);
        }
        if self.shared.overflow.load(Ordering::Acquire) {
            return Ok(self.overflow());
        }
        let buffered = self.rx.try_recv()?;
        assert!(buffered.at <= Instant::now());
        Ok(buffered.event)
    }
}

#[cfg(test)]
mod tests {
    use std::{task::Poll, time::Duration};

    use super::*;

    fn token(id: u32) -> GenEvent {
        GenEvent::Token {
            token_id: id,
            prompt_tokens: 4,
            cached_tokens: 0,
        }
    }

    #[test]
    fn ready_events_do_not_register_a_timer() {
        // No time driver: an immediate event must not depend on a timer tick.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let (tx, mut rx) = output_channel(2);
            let mut receiving = Box::pin(rx.recv());
            assert!(futures::poll!(&mut receiving).is_pending());
            tx.send(token(1)).unwrap();
            assert!(matches!(
                futures::poll!(&mut receiving),
                Poll::Ready(Some(GenEvent::Token { token_id: 1, .. }))
            ));
            drop(receiving);

            tx.send_at(token(2), Instant::now() - Duration::from_millis(1))
                .unwrap();
            let mut expired = Box::pin(rx.recv());
            assert!(matches!(
                futures::poll!(&mut expired),
                Poll::Ready(Some(GenEvent::Token { token_id: 2, .. }))
            ));
        });
    }

    #[tokio::test]
    async fn delayed_events_keep_their_budget_and_overflow_interrupts_delivery() {
        let (tx, mut rx) = output_channel(2);
        tx.send_at(token(1), Instant::now() + Duration::from_secs(3600))
            .unwrap();
        let mut receiving = Box::pin(rx.recv());
        assert!(futures::poll!(&mut receiving).is_pending());
        tx.send(token(2)).unwrap();
        assert!(
            tx.send(token(3)).is_err(),
            "the event awaiting delivery still counts"
        );
        assert!(tx.is_closed());
        assert!(matches!(
            futures::poll!(&mut receiving),
            Poll::Ready(Some(GenEvent::OutputOverflow))
        ));
        drop(receiving);
        assert!(rx.recv().await.is_none());
        assert_eq!(tx.shared.slots.available_permits(), 2);
    }

    #[tokio::test]
    async fn cancelling_a_delayed_receive_preserves_fifo_and_its_event() {
        let (tx, mut rx) = output_channel(2);
        let at = Instant::now() + Duration::from_millis(100);
        tx.send_at(token(1), at).unwrap();
        tx.send(token(2)).unwrap();
        let mut receiving = Box::pin(rx.recv());
        assert!(futures::poll!(&mut receiving).is_pending());
        drop(receiving);
        assert!(matches!(
            rx.recv().await,
            Some(GenEvent::Token { token_id: 1, .. })
        ));
        assert!(Instant::now() >= at);
        assert!(matches!(
            rx.recv().await,
            Some(GenEvent::Token { token_id: 2, .. })
        ));
        drop(rx);
        assert!(tx.is_closed());
    }
}
