//! A backend that records what the request paths settle, for the tests of
//! the settle sites, and the reservation request the backends' tests
//! reserve with.

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
use crate::tenant::TenantKey;

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
    async fn reserve(&self, _request: ReserveRequest<'_>) -> ReserveOutcome {
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

/// A reservation of `tokens` input tokens for `tenant`: a fresh charge id and
/// no model, as the backends' tests reserve.
pub(crate) fn reserve_request(tenant: &str, tokens: u32) -> ReserveRequest<'static> {
    ReserveRequest {
        request_charge_id: Uuid::now_v7(),
        tenant_key: TenantKey::new(tenant),
        model_id: None,
        estimated_input_tokens: tokens,
    }
}
