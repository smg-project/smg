//! The bound on a request's wait inside the gateway ahead of worker selection
//! (`--queue-timeout-secs`): one implementation for the gRPC selection stage,
//! the regular HTTP router and the HTTP prefill-decode router.
//!
//! The admission bound caps how many requests the gateway holds, not for how
//! long: a gateway that falls behind its own ingress (tokenization, selection)
//! would otherwise dispatch requests that waited minutes to workers that are
//! not overloaded at all, so neither the overload shed nor the admission queue
//! ever answers. The wait is read from the request's [`AcceptedAt`] stamp (the
//! admission grant where there is one, else the gateway's acceptance), once
//! per request; a request held past the bound is refused with a 503 under
//! [`SELECTION_QUEUE_TIMEOUT_ERROR_CODE`] before any worker is read, terminal
//! for the retry layer like every shed, with the admission queue's back-off
//! hint and counted among its rejections. A disaggregated request's prefill
//! admission wait is held to the remaining budget
//! ([`QueueTimeout::admit_selection_within_budget`]).

use std::{
    future::Future,
    time::{Duration, Instant},
};

use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};
use tracing::debug;

use crate::{
    middleware::{tenant_resolution::AcceptedAt, TenantRequestMeta, SHED_RETRY_AFTER_SECS},
    observability::metrics::{metrics_labels, Metrics},
    routers::{common::retry::mark_non_retryable, error},
};

/// Client-visible, gateway-owned code for a request refused ahead of worker
/// selection because it had already waited longer than the queue timeout
/// inside the gateway.
pub(crate) const SELECTION_QUEUE_TIMEOUT_ERROR_CODE: &str = "selection_queue_timeout";

/// The longest a request may have waited inside the gateway when it reaches
/// worker selection, as a router holds it: `--queue-timeout-secs`, zero
/// leaving the wait unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueueTimeout(Option<Duration>);

impl QueueTimeout {
    /// No bound: the wait ahead of selection is never refused.
    pub(crate) const UNBOUNDED: Self = Self(None);

    /// The bound `timeout` sets; zero leaves the wait unbounded.
    pub(crate) fn new(timeout: Duration) -> Self {
        Self((!timeout.is_zero()).then_some(timeout))
    }

    /// The bound `--queue-timeout-secs` sets; zero leaves the wait unbounded.
    pub(crate) fn from_secs(secs: u64) -> Self {
        Self::new(Duration::from_secs(secs))
    }

    /// The acceptance stamp this selection is bounded by, taken from the
    /// request's meta once: the first selection of a request takes it, a later
    /// run of the same request (a Responses tool loop iteration, with the same
    /// meta) finds it spent and runs unbounded, as it did not wait inside the
    /// gateway. `None` without a bound, a meta or a fresh stamp.
    pub(crate) fn acceptance_for_selection(
        self,
        meta: Option<&TenantRequestMeta>,
    ) -> Option<Instant> {
        self.0?;
        meta?.extension::<AcceptedAt>()?.take_for_selection()
    }

    /// The 503 for a request held past the queue timeout since `accepted`
    /// (the admission grant where there is one, else the gateway's acceptance),
    /// before any worker is read. Checked again after the prefill admission
    /// wait of a disaggregated request, which can outlast the bound. Terminal
    /// for the retry layer, like every shed, with the same back-off hint as
    /// the admission queue's refusals.
    pub(crate) fn refuse_if_expired(self, model_id: &str, accepted: Instant) -> Option<Response> {
        let timeout = self.0?;
        let waited = waited_past_timeout(accepted, AcceptedAt::clock_now(), timeout)?;
        Some(queue_timeout_refusal(model_id, waited, timeout))
    }

    /// `admission`, a prefill admission wait, held to the remaining budget of
    /// the bound: when the budget runs out the waiter is dropped (its queue
    /// ticket goes with it) and the request is refused as one held past the
    /// queue timeout; without a bound the wait is the prefill queue's own.
    async fn admit_within_budget<T, F>(
        self,
        model_id: &str,
        accepted: Option<Instant>,
        admission: F,
    ) -> Result<T, Response>
    where
        F: Future<Output = Result<T, Response>>,
    {
        let (Some(timeout), Some(accepted)) = (self.0, accepted) else {
            return admission.await;
        };
        let budget =
            timeout.saturating_sub(AcceptedAt::clock_now().saturating_duration_since(accepted));
        match tokio::time::timeout(budget, admission).await {
            Ok(result) => result,
            Err(_) => Err(queue_timeout_refusal(
                model_id,
                AcceptedAt::clock_now().saturating_duration_since(accepted),
                timeout,
            )),
        }
    }

    /// A disaggregated selection admitted within the budget
    /// ([`Self::admit_within_budget`]) and still within it afterwards: an
    /// admission that lands right at the bound gives its claim back (the
    /// selection is dropped with the load guards it holds) instead of taking
    /// a worker.
    pub(crate) async fn admit_selection_within_budget<T, F>(
        self,
        model_id: &str,
        accepted: Option<Instant>,
        admission: F,
    ) -> Result<T, Response>
    where
        F: Future<Output = Result<T, Response>>,
    {
        let selected = self
            .admit_within_budget(model_id, accepted, admission)
            .await?;
        if let Some(refused) =
            accepted.and_then(|accepted| self.refuse_if_expired(model_id, accepted))
        {
            drop(selected);
            return Err(refused);
        }
        Ok(selected)
    }
}

/// The 503 of a request that waited `waited`, past the queue timeout.
fn queue_timeout_refusal(model_id: &str, waited: Duration, timeout: Duration) -> Response {
    Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_SELECTION_TIMEOUT);
    debug!(
        model_id = %model_id,
        waited_secs = waited.as_secs_f64(),
        timeout_secs = timeout.as_secs_f64(),
        "Refused ahead of worker selection: waited longer than the queue timeout"
    );
    let mut response = error::service_unavailable(
        SELECTION_QUEUE_TIMEOUT_ERROR_CODE,
        format!(
            "The request waited {:.1} s inside the gateway before worker selection, longer \
             than the queue timeout of {} s",
            waited.as_secs_f64(),
            timeout.as_secs()
        ),
    );
    response
        .headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from(SHED_RETRY_AFTER_SECS));
    mark_non_retryable(&mut response);
    response
}

/// How long a request accepted at `accepted` has waited by `now`, when that
/// exceeds `timeout`.
pub(crate) fn waited_past_timeout(
    accepted: Instant,
    now: Instant,
    timeout: Duration,
) -> Option<Duration> {
    let waited = now.saturating_duration_since(accepted);
    (waited > timeout).then_some(waited)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wait_counts_only_once_it_exceeds_the_timeout() {
        let accepted = Instant::now();
        let timeout = Duration::from_secs(60);
        assert_eq!(waited_past_timeout(accepted, accepted, timeout), None);
        assert_eq!(
            waited_past_timeout(accepted, accepted + Duration::from_secs(60), timeout),
            None,
            "exactly the timeout is still inside the bound"
        );
        assert_eq!(
            waited_past_timeout(accepted, accepted + Duration::from_secs(61), timeout),
            Some(Duration::from_secs(61))
        );
        assert_eq!(
            waited_past_timeout(
                accepted + Duration::from_secs(5),
                accepted,
                Duration::from_secs(1)
            ),
            None,
            "a clock that reads before the acceptance is a zero wait, not a refusal"
        );
    }

    /// Zero is the flag's way of switching the bound off: no stamp is taken
    /// and nothing is refused, however long the request waited.
    #[test]
    fn zero_leaves_the_wait_unbounded() {
        assert_eq!(QueueTimeout::from_secs(0), QueueTimeout::UNBOUNDED);
        let long_ago = Instant::now()
            .checked_sub(Duration::from_secs(5))
            .expect("the clock has been running for five seconds");
        let meta = TenantRequestMeta::new(crate::tenant::TenantKey::new("t"))
            .with_extension(AcceptedAt::at(long_ago));
        assert_eq!(
            QueueTimeout::UNBOUNDED.acceptance_for_selection(Some(&meta)),
            None,
            "without a bound the stamp is left for a bounded selection"
        );
        assert!(QueueTimeout::UNBOUNDED
            .refuse_if_expired("m", long_ago)
            .is_none());
        assert!(
            QueueTimeout::from_secs(1)
                .acceptance_for_selection(Some(&meta))
                .is_some(),
            "the stamp was not spent by the unbounded read"
        );
    }
}
