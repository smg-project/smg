//! Cap on the preprocessed media bytes the gateway holds in flight for engines,
//! and the gauge of what it holds, budget or not.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::response::Response;
use http::{header::RETRY_AFTER, HeaderValue, StatusCode};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{middleware::SHED_RETRY_AFTER_SECS, observability::metrics::Metrics, routers::error};

/// Granularity of the budget, so the permit count stays within the semaphore's range;
/// a budget below one unit rounds to nothing, which the configuration refuses.
pub(crate) const UNIT_BYTES: usize = 1024;
/// How long a request waits for bytes to free up before it is refused.
const WAIT: Duration = Duration::from_secs(2);

/// Why a reservation was refused.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum InflightRefusal {
    /// Larger than the whole budget; waiting would never help.
    TooLarge,
    /// The budget did not free up in time.
    Busy,
}

/// Bytes of preprocessed media the gateway may hold in flight at once.
pub(crate) struct MultimodalInflight {
    budget_bytes: usize,
    units: usize,
    semaphore: Arc<Semaphore>,
    /// What the requests currently queued for room are holding.
    waiting: AtomicUsize,
    wait: Duration,
}

impl MultimodalInflight {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        // Rounded down, so a budget that is not a whole number of units is
        // enforced at the nearest value below it rather than above. A budget
        // under one unit therefore admits nothing, which is the honest
        // reading of asking for less than the smallest amount that can be
        // handed out.
        let units = (budget_bytes / UNIT_BYTES).min(Semaphore::MAX_PERMITS);
        Self {
            budget_bytes: units * UNIT_BYTES,
            units,
            semaphore: Arc::new(Semaphore::new(units)),
            waiting: AtomicUsize::new(0),
            wait: WAIT,
        }
    }

    #[cfg(test)]
    fn with_wait(mut self, wait: Duration) -> Self {
        self.wait = wait;
        self
    }

    pub(crate) fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Hold `bytes` of the budget until the returned permit drops.
    pub(crate) async fn reserve(&self, bytes: usize) -> Result<InflightPermit, InflightRefusal> {
        let units = bytes.div_ceil(UNIT_BYTES);
        if units > self.units {
            return Err(InflightRefusal::TooLarge);
        }
        let Ok(permit_units) = u32::try_from(units) else {
            return Err(InflightRefusal::TooLarge);
        };
        // A request keeps its media while it queues, so the queue weighs as
        // much as the budget does. Turning arrivals away past one budget's
        // worth of queue keeps what the gateway holds bounded, instead of
        // letting it grow with however many callers happen to be waiting.
        let queued = Waiting::enter(&self.waiting, units);
        if queued.total() > self.units {
            return Err(InflightRefusal::Busy);
        }
        let acquire = Arc::clone(&self.semaphore).acquire_many_owned(permit_units);
        match tokio::time::timeout(self.wait, acquire).await {
            Ok(Ok(permit)) => Ok(InflightPermit::counting(bytes, Some(permit))),
            Ok(Err(_)) | Err(_) => Err(InflightRefusal::Busy),
        }
    }
}

/// One queued request's share of the waiting total, given back on drop.
struct Waiting<'a> {
    waiting: &'a AtomicUsize,
    units: usize,
    total: usize,
}

impl<'a> Waiting<'a> {
    fn enter(waiting: &'a AtomicUsize, units: usize) -> Self {
        let total = waiting.fetch_add(units, Ordering::AcqRel) + units;
        Self {
            waiting,
            units,
            total,
        }
    }

    fn total(&self) -> usize {
        self.total
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.waiting.fetch_sub(self.units, Ordering::AcqRel);
    }
}

/// Releases its share of the budget when dropped and takes its bytes off the
/// in-flight gauge. Without a budget it holds nothing and counts alone.
#[derive(Debug)]
pub(crate) struct InflightPermit {
    _permit: Option<OwnedSemaphorePermit>,
    bytes: usize,
}

impl InflightPermit {
    /// `bytes` on the gauge (`smg_mm_inflight_bytes`) until the drop.
    fn counting(bytes: usize, permit: Option<OwnedSemaphorePermit>) -> Self {
        Metrics::record_mm_inflight_reserved(bytes);
        Self {
            _permit: permit,
            bytes,
        }
    }
}

impl Drop for InflightPermit {
    fn drop(&mut self) {
        Metrics::record_mm_inflight_released(self.bytes);
    }
}

/// Reserve room for a request's media bytes, or the response that refuses it.
/// Without a budget nothing is held: the bytes go on the gauge until the
/// permit drops, so the gauge reads either way.
pub(crate) async fn reserve_multimodal_inflight(
    inflight: Option<&MultimodalInflight>,
    bytes: usize,
) -> Result<Option<InflightPermit>, Response> {
    let Some(inflight) = inflight else {
        return Ok(Some(InflightPermit::counting(bytes, None)));
    };
    match inflight.reserve(bytes).await {
        Ok(permit) => Ok(Some(permit)),
        Err(InflightRefusal::TooLarge) => {
            Metrics::record_admission_rejected("multimodal_too_large");
            Err(error::create_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "multimodal_payload_too_large",
                format!(
                    "the request carries {bytes} bytes of preprocessed media, more than the {} bytes this gateway holds in flight",
                    inflight.budget_bytes()
                ),
            ))
        }
        Err(InflightRefusal::Busy) => {
            Metrics::record_admission_rejected("multimodal_inflight");
            let mut response = error::too_many_requests(
                "multimodal_inflight_budget",
                "the gateway is already holding its budget of preprocessed media in flight; retry shortly",
            );
            // "Retry shortly" says how short, the way the gateway's other
            // sheds do, so a client that honours the header paces itself
            // instead of meeting the same full budget again at once.
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from(SHED_RETRY_AFTER_SECS));
            Err(response)
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/inflight/tests.rs"]
mod tests;
