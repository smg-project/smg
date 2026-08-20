//! Dispatch-phase ownership of request memory, shared by every router.

use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;

use crate::{config::types::RetryConfig, observability::metrics::Metrics};

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
}

/// Borrowed view of the leased request and its routing derivatives.
#[derive(Clone, Copy)]
pub(crate) struct LeaseView<'a, T> {
    pub request: &'a T,
    pub tokens: Option<&'a [u32]>,
    pub text: Option<&'a str>,
    pub rid_key: Option<&'a str>,
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
    Legs(Bytes, Bytes),
}

impl SerializedBody {
    /// One upstream rendering of the request; PD legs differ only in
    /// injected fields, so the larger leg stands for both.
    fn released_len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Single(body) => body.len(),
            Self::Legs(prefill, decode) => prefill.len().max(decode.len()),
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
        let body = Bytes::from(f(Self::view(&inner))?);
        inner.body = SerializedBody::Single(body.clone());
        Ok(body)
    }

    /// Two-leg (PD) variant of [`Self::serialize_with`]: both leg bodies come
    /// from one pass over the leased request, and the closure's intermediate
    /// trees die with it.
    pub(crate) fn serialize_legs_with<E>(
        &self,
        f: impl FnOnce(LeaseView<'_, T>) -> Result<(Vec<u8>, Vec<u8>), E>,
    ) -> Result<(Bytes, Bytes), E> {
        let mut inner = self.lock();
        let (prefill, decode) = f(Self::view(&inner))?;
        let legs = (Bytes::from(prefill), Bytes::from(decode));
        inner.body = SerializedBody::Legs(legs.0.clone(), legs.1.clone());
        Ok(legs)
    }

    /// Memoized upstream body from the last single-leg serialization; the
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
            SerializedBody::None | SerializedBody::Legs(..) => None,
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
        }
    }
}

/// Shared drop-probe idiom for release tests: a probed request type plus
/// loopback stubs gated on the probe's weak count.
#[cfg(test)]
pub(crate) mod test_probe {
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, Weak,
        },
        time::Duration,
    };

    use axum::http::header::CONTENT_TYPE;
    use openai_protocol::common::GenerationRequest;

    /// Typed request carrying a drop probe; tests watch the `Arc` count to
    /// observe exactly when a router frees the parsed body.
    #[derive(serde::Serialize)]
    pub(crate) struct DropProbeRequest {
        pub text: String,
        #[serde(skip)]
        pub _probe: Arc<()>,
    }

    impl GenerationRequest for DropProbeRequest {
        fn is_stream(&self) -> bool {
            false
        }

        fn get_model(&self) -> Option<&str> {
            None
        }

        fn extract_text_for_routing(&self) -> String {
            self.text.clone()
        }
    }

    /// Loopback POST /generate stub that answers `{}` only after every probe
    /// clone outside the test is gone (or after a deadline, leaving
    /// `released` false).
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    pub(crate) async fn spawn_release_gated_stub(probe: Weak<()>) -> (String, Arc<AtomicBool>) {
        let released = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&released);
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move || {
                let probe = probe.clone();
                let flag = Arc::clone(&flag);
                async move {
                    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                    while probe.strong_count() > 1 && tokio::time::Instant::now() < deadline {
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    flag.store(probe.strong_count() <= 1, Ordering::SeqCst);
                    ([(CONTENT_TYPE, "application/json")], "{}")
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), released)
    }

    /// Loopback POST /generate stub answering `{}` immediately.
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    pub(crate) async fn spawn_immediate_stub() -> String {
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(|| async { ([(CONTENT_TYPE, "application/json")], "{}") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, sync::Arc};

    use super::*;

    fn lease(release: ReleasePoint) -> (RequestLease<Arc<()>>, std::sync::Weak<()>) {
        let probe = Arc::new(());
        let weak = Arc::downgrade(&probe);
        (
            RequestLease::new(probe, RoutingDerivatives::default(), release),
            weak,
        )
    }

    #[test]
    fn after_dispatch_release_frees_request_and_memo() {
        let (lease, probe) = lease(ReleasePoint::AfterDispatch);
        let body = lease
            .serialize_with(|_| Ok::<_, Infallible>(b"abcd".to_vec()))
            .unwrap();
        assert_eq!(probe.strong_count(), 1);
        assert_eq!(lease.body(), Some(body.clone()));

        lease.release_dispatch();

        assert_eq!(probe.strong_count(), 0, "release must free the request");
        assert_eq!(lease.body(), None, "release must drop the memo handle");
        assert_eq!(body.as_ref(), b"abcd", "caller handles stay valid");
    }

    #[test]
    fn at_retry_close_keeps_request_until_drop() {
        let (lease, probe) = lease(ReleasePoint::AtRetryClose);
        lease
            .serialize_with(|_| Ok::<_, Infallible>(b"abcd".to_vec()))
            .unwrap();

        lease.release_dispatch();

        assert_eq!(probe.strong_count(), 1, "request must survive for replay");
        assert!(lease.body().is_some());

        drop(lease);
        assert_eq!(probe.strong_count(), 0, "drop closes the lease");
    }

    #[test]
    fn serialize_refreshes_the_memo_per_attempt() {
        let (lease, _probe) = lease(ReleasePoint::AtRetryClose);
        lease
            .serialize_with(|_| Ok::<_, Infallible>(b"first".to_vec()))
            .unwrap();
        lease
            .serialize_with(|_| Ok::<_, Infallible>(b"second".to_vec()))
            .unwrap();
        assert_eq!(lease.body().as_deref(), Some(b"second".as_slice()));
    }

    #[test]
    fn legs_serialize_from_one_view_and_release_together() {
        let (lease, probe) = lease(ReleasePoint::AfterDispatch);
        let (prefill, decode) = lease
            .serialize_legs_with(|_| Ok::<_, Infallible>((b"pp".to_vec(), b"dddd".to_vec())))
            .unwrap();
        assert_eq!(lease.body(), None, "legs are not a single-body memo");

        lease.release_dispatch();

        assert_eq!(probe.strong_count(), 0);
        assert_eq!(prefill.as_ref(), b"pp");
        assert_eq!(decode.as_ref(), b"dddd");
    }
}
