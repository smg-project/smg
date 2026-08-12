//! Generic abort-on-drop wrapper for engine streaming responses.
//!
//! Each engine's `generate()` returns a server-streaming RPC; if the caller
//! drops the stream early (client disconnect, error, panic), the backend
//! has no way to know it should stop scheduling work. The wrapper here
//! sends an explicit `abort_request` from `Drop` so resources are
//! reclaimed even when the request never completes normally.
//!
//! Engines plug in via the `AbortOnDropClient` trait — they describe how
//! to translate `(client, request_id)` into the abort future. This keeps
//! the (large) Drop / Stream impl in one place instead of replicated
//! across `mlx_engine`, `sglang_scheduler`, `vllm_engine`, and
//! `trtllm_service`.

use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use tonic::Streaming;
use tracing::{debug, warn};

/// Upper bound on how long a deferred abort waits for the stream's first
/// item before aborting anyway. Bounds the detached drain task when the
/// backend never produces output (e.g. its upstream handoff peer died).
const DEFERRED_ABORT_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Bridge between the generic [`AbortOnDropStream`] and an engine-specific
/// client. Implementors provide an async function that the wrapper calls
/// from `Drop` to release backend resources.
pub trait AbortOnDropClient: Clone + Send + Sync + 'static {
    /// Returned future is awaited by the spawned cleanup task — engines
    /// are free to attach the appropriate `reason` string (or omit it,
    /// for proto schemas that don't carry one).
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>>;
}

/// Smart wrapper around `tonic::Streaming<T>` that fires an abort RPC on
/// `Drop` unless [`mark_completed`](Self::mark_completed) was called.
///
/// `T` is the engine-specific stream item (typically `proto::GenerateResponse`).
/// `C` is the engine client implementing [`AbortOnDropClient`].
pub struct AbortOnDropStream<T: Send + 'static, C: AbortOnDropClient> {
    /// `Some` while the wrapper owns the stream; `Drop` may `take()` it into
    /// a detached drain task when the abort is deferred.
    inner: Option<Streaming<T>>,
    request_id: String,
    client: C,
    aborted: Arc<AtomicBool>,
    /// When set, a drop that happens before the stream yielded its first
    /// item drains the stream in the background until that first item (or a
    /// terminal event / [`DEFERRED_ABORT_MAX_WAIT`]) before sending the
    /// abort. For workers whose request must not be torn down mid-handoff —
    /// e.g. a disaggregated decode leg receiving KV state from its prefill
    /// peer — an early abort can kill the transfer while the peer is still
    /// writing; the first generated item is the proof the handoff completed.
    defer_abort_until_first_item: bool,
    /// Whether `poll_next` has yielded at least one item (data or error).
    saw_item: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Send + 'static, C: AbortOnDropClient> AbortOnDropStream<T, C> {
    /// Wrap a streaming response so it auto-aborts on drop.
    pub fn new(stream: Streaming<T>, request_id: String, client: C) -> Self {
        debug!("Created AbortOnDropStream for request {}", request_id);
        Self {
            inner: Some(stream),
            request_id,
            client,
            aborted: Arc::new(AtomicBool::new(false)),
            defer_abort_until_first_item: false,
            saw_item: false,
            _marker: PhantomData,
        }
    }

    /// Defer the abort-on-drop until the stream has produced its first item.
    ///
    /// Default behavior (off) aborts immediately on drop. See the field docs
    /// for when deferral is required; `mark_completed` still suppresses the
    /// abort entirely in either mode.
    #[must_use]
    pub fn defer_abort_until_first_item(mut self) -> Self {
        self.defer_abort_until_first_item = true;
        self
    }

    /// Suppress the abort-on-drop. Call after the stream completes
    /// successfully so the backend isn't told to abort an already-finished
    /// request.
    pub fn mark_completed(&self) {
        // Release pairs with AcqRel in `Drop::drop` so the cleanup task
        // observes this write.
        self.aborted.store(true, Ordering::Release);
        debug!("Request {} marked as completed", self.request_id);
    }
}

impl<T: Send + 'static, C: AbortOnDropClient> Drop for AbortOnDropStream<T, C> {
    fn drop(&mut self) {
        // Atomically claim the "send abort" responsibility. If
        // `mark_completed` already ran, `compare_exchange` fails and we
        // bail out; otherwise we own the cleanup.
        if self
            .aborted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let request_id = self.request_id.clone();
        let request_id_for_log = request_id.clone();
        let client = self.client.clone();

        // Deferred mode, dropped before the first item: keep the stream
        // alive in the background until the backend produces its first item
        // (or terminates, or the wait cap fires), THEN abort. The request
        // still dies — the deferral only moves the abort past the window
        // where tearing it down would interrupt an in-flight handoff.
        let drain = if self.defer_abort_until_first_item && !self.saw_item {
            self.inner.take()
        } else {
            None
        };

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            if let Some(mut stream) = drain {
                debug!(
                    "Stream dropped before first item for request {}, draining before abort",
                    request_id_for_log
                );
                let first_item = tokio::time::timeout(DEFERRED_ABORT_MAX_WAIT, async {
                    // `Streaming::message` resolves on the next message,
                    // stream end, or transport error — any of which ends the
                    // protected window.
                    let _ = stream.message().await;
                })
                .await;
                if first_item.is_err() {
                    warn!(
                        "Deferred abort for request {} timed out waiting for the first item",
                        request_id_for_log
                    );
                }
            } else {
                debug!(
                    "Stream dropped without completion for request {}, sending abort",
                    request_id_for_log
                );
            }
            if let Err(e) = client.abort_for_drop(request_id).await {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

// `Streaming<T>` is `Unpin` regardless of `T`, and we never project a
// pinned reference to any field. Marking the wrapper `Unpin` lets us
// use `Pin<&mut Self>::deref_mut` to reach `inner` without needing
// `pin-project` machinery.
impl<T: Send + 'static, C: AbortOnDropClient> Unpin for AbortOnDropStream<T, C> {}

impl<T: Send + 'static, C: AbortOnDropClient> futures::Stream for AbortOnDropStream<T, C> {
    type Item = Result<T, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match this.inner.as_mut() {
            Some(inner) => {
                let polled = Pin::new(inner).poll_next(cx);
                if matches!(polled, Poll::Ready(Some(_))) {
                    // Any yielded item — data or error — ends the deferred
                    // window: the backend has responded, so its handoff
                    // state is resolved either way.
                    this.saw_item = true;
                }
                polled
            }
            // Only reachable if polled after Drop took the stream, which
            // cannot happen for an owned value; return end-of-stream rather
            // than panicking to keep the impl total.
            None => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-only test: any `Clone + Send + Sync + 'static` type can
    /// implement [`AbortOnDropClient`] with a single async method.
    /// `Drop` and `Stream` semantics are exercised via the engine-level
    /// integration tests that hit a real gRPC server.
    #[test]
    fn trait_is_implementable_by_simple_client() {
        #[derive(Clone)]
        struct DummyClient;

        impl AbortOnDropClient for DummyClient {
            fn abort_for_drop(
                self,
                _request_id: String,
            ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
                Box::pin(async { Ok(()) })
            }
        }

        // `_marker` is `PhantomData<fn() -> T>`, so the struct itself is
        // `Send + Sync` regardless of `T`.
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<AbortOnDropStream<(), DummyClient>>();
    }
}
