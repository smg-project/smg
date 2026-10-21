//! A backend that records what the request paths settle, for the tests of
//! the settle sites.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;
use parking_lot::Mutex;
use uuid::Uuid;

use super::{
    backend::{RateLimitBackend, ReserveOutcome, ReserveRequest, UsageSettlement},
    SharedReservationHandle,
};

/// Admits everything and records every settlement and close.
#[derive(Default)]
pub(crate) struct RecordingBackend {
    settlements: Mutex<Vec<UsageSettlement>>,
    closes: AtomicUsize,
}

impl RecordingBackend {
    /// A reservation on a fresh recording backend, and the backend to read
    /// back what the path under test did with it.
    pub(crate) fn reservation() -> (Arc<SharedReservationHandle>, Arc<Self>) {
        let backend = Arc::new(Self::default());
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        (handle, backend)
    }

    /// The settlements recorded so far, in order.
    pub(crate) fn settlements(&self) -> Vec<UsageSettlement> {
        self.settlements.lock().clone()
    }

    /// How many reservations were closed without a settlement.
    pub(crate) fn closes(&self) -> usize {
        self.closes.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl RateLimitBackend for RecordingBackend {
    async fn reserve(&self, _request: ReserveRequest) -> ReserveOutcome {
        ReserveOutcome::Admitted
    }

    async fn settle_success(&self, _request_charge_id: Uuid, usage: UsageSettlement) {
        self.settlements.lock().push(usage);
    }

    async fn close_reserved_only(&self, _request_charge_id: Uuid) {
        self.closes.fetch_add(1, Ordering::SeqCst);
    }

    async fn abandon(&self, _request_charge_id: Uuid) {
        self.closes.fetch_add(1, Ordering::SeqCst);
    }
}
