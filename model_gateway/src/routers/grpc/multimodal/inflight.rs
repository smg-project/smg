//! Cap on the preprocessed media bytes the gateway holds in flight for engines.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::response::Response;
use http::StatusCode;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{observability::metrics::Metrics, routers::error};

/// Granularity of the budget, so the permit count stays within the semaphore's range.
const UNIT_BYTES: usize = 1024;
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
            Ok(Ok(permit)) => Ok(InflightPermit { _permit: permit }),
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

/// Releases its share of the budget when dropped.
#[derive(Debug)]
pub(crate) struct InflightPermit {
    _permit: OwnedSemaphorePermit,
}

/// Reserve room for a request's media bytes, or the response that refuses it.
pub(crate) async fn reserve_multimodal_inflight(
    inflight: Option<&MultimodalInflight>,
    bytes: usize,
) -> Result<Option<InflightPermit>, Response> {
    let Some(inflight) = inflight else {
        return Ok(None);
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
            Err(error::too_many_requests(
                "multimodal_inflight_budget",
                "the gateway is already holding its budget of preprocessed media in flight; retry shortly",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(budget: usize) -> MultimodalInflight {
        MultimodalInflight::new(budget).with_wait(Duration::from_millis(20))
    }

    #[tokio::test]
    async fn bytes_are_held_until_the_permit_drops() {
        let inflight = quick(4096);
        let first = inflight.reserve(3000).await.unwrap();
        assert_eq!(
            inflight.reserve(2000).await.unwrap_err(),
            InflightRefusal::Busy
        );
        drop(first);
        assert!(inflight.reserve(2000).await.is_ok());
    }

    #[tokio::test]
    async fn a_request_above_the_whole_budget_is_refused_at_once() {
        let inflight = quick(4096);
        let started = std::time::Instant::now();
        assert_eq!(
            inflight.reserve(5000).await.unwrap_err(),
            InflightRefusal::TooLarge
        );
        assert!(started.elapsed() < Duration::from_millis(15));
        assert!(inflight.reserve(0).await.is_ok());
    }

    /// A budget that is not a whole number of units is enforced at the value
    /// below it, so the gateway never admits more than it was told to.
    #[tokio::test]
    async fn a_budget_between_units_is_rounded_down() {
        let inflight = quick(4097);
        assert_eq!(inflight.budget_bytes(), 4096);
        assert_eq!(
            inflight.reserve(5000).await.unwrap_err(),
            InflightRefusal::TooLarge
        );

        let under_one_unit = quick(1);
        assert_eq!(under_one_unit.budget_bytes(), 0);
        assert_eq!(
            under_one_unit.reserve(1).await.unwrap_err(),
            InflightRefusal::TooLarge
        );
        assert!(under_one_unit.reserve(0).await.is_ok());
    }

    /// Queued requests still hold their media, so the gateway turns arrivals
    /// away once the queue is as heavy as the budget rather than letting the
    /// two of them add up without limit.
    #[tokio::test]
    async fn arrivals_past_a_budget_of_waiting_are_refused_without_waiting() {
        let inflight = Arc::new(MultimodalInflight::new(4096).with_wait(Duration::from_secs(5)));
        let held = inflight.reserve(4096).await.unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins this handle before it returns"
        )]
        let queued = tokio::spawn({
            let inflight = Arc::clone(&inflight);
            async move { inflight.reserve(4096).await }
        });
        // Give the queued request time to register before the next arrives.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let started = std::time::Instant::now();
        assert_eq!(
            inflight.reserve(1024).await.unwrap_err(),
            InflightRefusal::Busy
        );
        assert!(started.elapsed() < Duration::from_millis(500));

        drop(held);
        assert!(queued.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn refusals_answer_413_and_429_and_no_budget_means_no_permit() {
        assert!(reserve_multimodal_inflight(None, usize::MAX)
            .await
            .unwrap()
            .is_none());

        let inflight = quick(4096);
        let too_large = reserve_multimodal_inflight(Some(&inflight), 5000)
            .await
            .unwrap_err();
        assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let _held = inflight.reserve(4096).await.unwrap();
        let busy = reserve_multimodal_inflight(Some(&inflight), 1024)
            .await
            .unwrap_err();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
