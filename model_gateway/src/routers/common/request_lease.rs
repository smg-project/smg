//! Dispatch-phase ownership of request memory, shared by every router.

use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;

use super::trim_serialization_slack;
use crate::{
    config::types::RetryConfig, observability::metrics::Metrics, policies::CacheNamespace,
};

/// When a [`RequestLease`] lets go of the parsed request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReleasePoint {
    /// Retries disabled: release at dispatch, once the upstream bytes exist.
    AfterDispatch,
    /// Retries enabled: keep the request for replay; released when the lease
    /// drops, at retry-window close (first non-retryable response).
    AtRetryClose,
}

impl ReleasePoint {
    pub(crate) fn from_retry_config(config: &RetryConfig) -> Self {
        if config.max_retries.max(1) <= 1 {
            Self::AfterDispatch
        } else {
            Self::AtRetryClose
        }
    }
}

/// Routing inputs derived from the request body before dispatch.
#[derive(Default)]
pub(crate) struct RoutingDerivatives {
    pub tokens: Option<Vec<u32>>,
    pub text: Option<String>,
    pub rid_key: Option<String>,
    pub cache_namespace: Option<CacheNamespace>,
}

/// Borrowed view of the leased request and its routing derivatives.
#[derive(Clone, Copy)]
pub(crate) struct LeaseView<'a, T> {
    pub request: &'a T,
    pub tokens: Option<&'a [u32]>,
    pub text: Option<&'a str>,
    pub rid_key: Option<&'a str>,
    pub cache_namespace: Option<CacheNamespace>,
}

/// Single owner of a request's dispatch-phase memory: the parsed request,
/// its routing derivatives, and the memoized serialized upstream body.
///
/// Invariant: the parsed request and its derivatives live exactly until the
/// lease's release point — [`ReleasePoint::AfterDispatch`] frees them the
/// moment the upstream bytes exist, [`ReleasePoint::AtRetryClose`] keeps them
/// for retry replay until the lease drops.
pub(crate) struct RequestLease<T> {
    inner: Mutex<Inner<T>>,
    release: ReleasePoint,
}

struct Inner<T> {
    held: Option<Held<T>>,
    body: SerializedBody,
}

struct Held<T> {
    request: T,
    routing: RoutingDerivatives,
}

enum SerializedBody {
    None,
    Single(Bytes),
    Sides(Bytes, Bytes),
}

impl SerializedBody {
    /// One upstream rendering of the request; PD sides differ only in
    /// injected fields, so the larger side stands for both.
    fn released_len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Single(body) => body.len(),
            Self::Sides(prefill, decode) => prefill.len().max(decode.len()),
        }
    }
}

impl<T> RequestLease<T> {
    pub(crate) fn new(request: T, routing: RoutingDerivatives, release: ReleasePoint) -> Self {
        Self {
            inner: Mutex::new(Inner {
                held: Some(Held { request, routing }),
                body: SerializedBody::None,
            }),
            release,
        }
    }

    pub(crate) fn release_point(&self) -> ReleasePoint {
        self.release
    }

    /// Run `f` over the leased request and derivatives. Valid only before
    /// release; the closure keeps the borrow synchronous by construction.
    pub(crate) fn with_view<R>(&self, f: impl FnOnce(LeaseView<'_, T>) -> R) -> R {
        let inner = self.lock();
        f(Self::view(&inner))
    }

    /// Serialize the upstream body from the leased request, memoizing the
    /// bytes for [`Self::body`]. Calling again (a later retry attempt whose
    /// worker may shape the body differently) refreshes the memo.
    pub(crate) fn serialize_with<E>(
        &self,
        f: impl FnOnce(LeaseView<'_, T>) -> Result<Vec<u8>, E>,
    ) -> Result<Bytes, E> {
        let mut inner = self.lock();
        let mut buf = f(Self::view(&inner))?;
        // The frozen memo can outlive dispatch (retry replay); never let it
        // drag growth slack along.
        trim_serialization_slack(&mut buf);
        let body = Bytes::from(buf);
        inner.body = SerializedBody::Single(body.clone());
        Ok(body)
    }

    /// Two-side (PD) variant of [`Self::serialize_with`]: both side bodies come
    /// from one pass over the leased request, and the closure's intermediate
    /// trees die with it.
    pub(crate) fn serialize_sides_with<E>(
        &self,
        f: impl FnOnce(LeaseView<'_, T>) -> Result<(Vec<u8>, Vec<u8>), E>,
    ) -> Result<(Bytes, Bytes), E> {
        let mut inner = self.lock();
        let (mut prefill, mut decode) = f(Self::view(&inner))?;
        trim_serialization_slack(&mut prefill);
        trim_serialization_slack(&mut decode);
        let sides = (Bytes::from(prefill), Bytes::from(decode));
        inner.body = SerializedBody::Sides(sides.0.clone(), sides.1.clone());
        Ok(sides)
    }

    /// Memoized upstream body from the last single-body serialization; the
    /// handle is cheap and safe to hand to (re)dispatch attempts.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "accessor for dispatch paths that resend an unchanged memoized body"
        )
    )]
    pub(crate) fn body(&self) -> Option<Bytes> {
        match &self.lock().body {
            SerializedBody::Single(body) => Some(body.clone()),
            SerializedBody::None | SerializedBody::Sides(..) => None,
        }
    }

    /// Under `AfterDispatch`, free the parsed request and derivatives and
    /// count the serialized size as released early; under `AtRetryClose` a
    /// no-op — release happens when the lease drops.
    pub(crate) fn release_dispatch(&self) {
        if self.release != ReleasePoint::AfterDispatch {
            return;
        }
        let mut inner = self.lock();
        if inner.held.take().is_some() {
            Metrics::record_request_buffers_released_early(inner.body.released_len());
        }
        // In-flight sends hold their own handles to the serialized bytes.
        inner.body = SerializedBody::None;
    }

    fn lock(&self) -> MutexGuard<'_, Inner<T>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn view<'a>(inner: &'a Inner<T>) -> LeaseView<'a, T> {
        #[expect(
            clippy::expect_used,
            reason = "using a lease after release_dispatch is a dispatch-order bug, not a runtime condition"
        )]
        let held = inner
            .held
            .as_ref()
            .expect("request lease used after release");
        LeaseView {
            request: &held.request,
            tokens: held.routing.tokens.as_deref(),
            text: held.routing.text.as_deref(),
            rid_key: held.routing.rid_key.as_deref(),
            cache_namespace: held.routing.cache_namespace,
        }
    }
}

/// Shared drop-probe idiom for release tests: a probed request type plus
/// loopback stubs gated on the probe's weak count.
#[cfg(test)]
#[path = "../../../tests/unit/routers/common/request_lease/test_probe.rs"]
pub(crate) mod test_probe;

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/request_lease/tests.rs"]
mod tests;
