//! Generic abort-on-drop wrapper for engine streaming responses.
//!
//! Each engine's `generate()` returns a server-streaming RPC; if the caller
//! drops the stream early (client disconnect, error, panic), the backend
//! has no way to know it should stop scheduling work. The wrapper here
//! sends an explicit `abort_request` from `Drop` so resources are
//! reclaimed even when the request never completes normally. The stream
//! itself is released by the worker's own end of it: the cleanup task reads
//! it, bounded, until the worker ends it after the abort, and resets it only
//! when the worker does not (see `Drop`). A request's deadline
//! ([`AbortOnDropStream::with_deadline`]) ends the stream the same way from
//! the inside: at the deadline the stream is released as a dropped one is,
//! and its reader meets the deadline's status.
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
    time::Duration,
};

use tokio::time::Sleep;
use tonic::Streaming;
use tracing::{debug, warn};

use crate::request_deadline::{far_future, RequestDeadline, TimeoutKind};

/// Upper bound on how long a deferred abort waits for the stream's first
/// item before aborting anyway. Bounds the detached drain task when the
/// backend never produces output (e.g. its upstream handoff peer died).
const DEFERRED_ABORT_MAX_WAIT: Duration = Duration::from_secs(30);

/// Upper bound on how long a dropped stream is read for the worker's own end
/// of it, after the abort RPC, before it is released by a reset. A worker
/// that took the abort ends the stream within its next scheduling step; one
/// that is wedged does not, and the reset after this bound is what frees the
/// stream on this side.
const RELEASE_DRAIN_MAX_WAIT: Duration = Duration::from_secs(30);

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
    /// e.g. a disaggregated decode side receiving KV state from its prefill
    /// peer — an early abort can kill the transfer while the peer is still
    /// writing; the first generated item is the proof the handoff completed.
    defer_abort_until_first_item: bool,
    /// Whether `poll_next` has yielded at least one item (data or error).
    saw_item: bool,
    /// How long a drop waits for the worker to end the stream after the
    /// abort before releasing it with a reset (see `Drop`).
    release_drain_bound: Duration,
    /// The request's bounds and their one timer, when the stream is bounded
    /// (see [`Self::with_deadline`]).
    deadline: Option<Bounded>,
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
            release_drain_bound: RELEASE_DRAIN_MAX_WAIT,
            deadline: None,
            _marker: PhantomData,
        }
    }

    /// Bound the stream by the request's `deadline` (its total and, when
    /// set, its first-token and inter-token bounds): a poll at or past the
    /// total ends the stream, whatever the worker has sent since (as an HTTP
    /// client's timeout ends a body); a poll past a silence bound ends it
    /// only when the worker has nothing waiting. At the cut the stream is
    /// released as a dropped one is (the abort goes to the worker at once,
    /// the deferral and the drain apply as on a drop), and the reader meets
    /// the bound's status ([`RequestDeadline::status`]), then the end. One
    /// timer serves every bound: it is re-armed when a message moves the
    /// next applicable instant (the first message, and each message under an
    /// inter-token bound). A stream that yielded a message before the
    /// bounds were attached starts in its second phase.
    #[must_use]
    pub fn with_deadline(mut self, deadline: RequestDeadline) -> Self {
        // A stream whose first message was read before the bounds were
        // attached (a side read ahead during the dispatch) is in its second
        // phase: the silence after that message is the inter-token bound's,
        // counted from here, not the first-token bound's.
        let last = self.saw_item.then(tokio::time::Instant::now);
        let armed = deadline.next_at(last);
        let at = armed.map_or_else(far_future, |(at, _)| at);
        self.deadline = Some(Bounded {
            deadline,
            armed,
            sleep: Box::pin(tokio::time::sleep_until(at)),
        });
        self
    }

    /// Bound the wait for the worker's own end of a dropped stream (default
    /// `RELEASE_DRAIN_MAX_WAIT`, 30 s) before it is released with a reset.
    #[must_use]
    pub fn with_release_drain_bound(mut self, bound: Duration) -> Self {
        self.release_drain_bound = bound;
        self
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
        self.release();
    }
}

impl<T: Send + 'static, C: AbortOnDropClient> AbortOnDropStream<T, C> {
    /// Let go of the stream before its end: the abort goes to the worker
    /// (deferred past the first item when so configured) and the stream is
    /// read up to the worker's own end of it, both detached. What a drop
    /// does, and what the request's deadline does when it passes; nothing
    /// after `mark_completed`, nothing twice.
    fn release(&mut self) {
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
        let client = self.client.clone();
        let defer = self.defer_abort_until_first_item && !self.saw_item;
        let release_bound = self.release_drain_bound;
        // The stream goes to the cleanup task whole: it is released there by
        // the worker's own end of it (see below), not by the reset that
        // dropping it here would send.
        let stream = self.inner.take();

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            let mut stream = stream;
            // Deferred mode, dropped before the first item: keep the stream
            // alive until the backend produces its first item (or
            // terminates, or the wait cap fires), THEN abort. The request
            // still dies — the deferral only moves the abort past the window
            // where tearing it down would interrupt an in-flight handoff.
            if defer {
                if let Some(stream) = stream.as_mut() {
                    debug!(
                        "Stream dropped before first item for request {}, draining before abort",
                        request_id
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
                            request_id
                        );
                    }
                }
            }
            // The abort tells the worker to stop; the worker then ends the
            // stream with its own final frames, and reading up to them
            // releases the stream without a RST_STREAM from this side. A
            // reset here would leave the stream locally reset, which h2
            // remembers for one second only (fifty at a time): a frame the
            // worker still sends on it after that is a frame on an unknown
            // stream, answered with a library reset that counts toward the
            // connection's limit of 1024, past which the whole connection
            // goes away (GOAWAY ENHANCE_YOUR_CALM "too_many_internal_resets")
            // and every call in flight on it fails. A worker that does not
            // end the stream within the bound is released by the reset.
            futures::future::join(
                abort_with_deadline(client, request_id.clone()),
                release_by_draining(stream, release_bound, &request_id),
            )
            .await;
        });
    }
}

/// Read `stream` up to the worker's own end of it (its last message, an
/// error, or end of stream) for at most `bound`; it is dropped either way,
/// which resets it only when it is still open then.
async fn release_by_draining<T>(stream: Option<Streaming<T>>, bound: Duration, request_id: &str) {
    let Some(mut stream) = stream else {
        return;
    };
    let ended = tokio::time::timeout(bound, async {
        while let Ok(Some(_)) = stream.message().await {}
    })
    .await;
    if ended.is_err() {
        debug!(
            "Dropped stream for request {} still open after {:?}; releasing it with a reset",
            request_id, bound
        );
    }
}

/// Send the drop-abort RPC under a local deadline.
///
/// The abort runs in a detached task that no caller can cancel, so without a
/// bound a wedged backend (whose gRPC layer answers pings while its handler
/// hangs) would let these tasks accumulate without limit — precisely during the
/// mass-disconnect a backend brownout causes. See [`crate::ABORT_RPC_DEADLINE`].
async fn abort_with_deadline<C: AbortOnDropClient>(client: C, request_id: String) {
    abort_within(crate::ABORT_RPC_DEADLINE, client, request_id).await;
}

/// Deadline-bounded abort, parameterized on `deadline` for testability.
async fn abort_within<C: AbortOnDropClient>(deadline: Duration, client: C, request_id: String) {
    debug!(
        "Stream dropped without completion for request {}, sending abort",
        request_id
    );
    match tokio::time::timeout(deadline, client.abort_for_drop(request_id.clone())).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!(
            "Failed to send abort on drop for request {}: {}",
            request_id, e
        ),
        Err(_elapsed) => warn!(
            "Abort-on-drop RPC for request {} timed out after {:?}; giving up",
            request_id, deadline
        ),
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
        // The stream went with the deadline's cut (or, unreachably for an
        // owned value, with Drop): end-of-stream rather than a panic keeps
        // the impl total.
        if this.inner.is_none() {
            return Poll::Ready(None);
        }
        // The bound is read first, and the timer's waker sees to the poll
        // when the worker sends nothing at all. The total ends a stream
        // polled at or past it, a message the worker sent since included.
        // A silence bound asks the worker first: a message already waiting
        // in the transport is no silence of the worker's, whatever the
        // reader's own pace (a slow client holds the read loop in the
        // response channel while the worker's messages queue up), so it cuts
        // only a stream that has nothing when its timer has fired. After
        // `mark_completed` the request is done and no bound speaks again: a
        // later poll reaches the stream's own end.
        let completed = this.aborted.load(Ordering::Acquire);
        let passed = this
            .deadline
            .as_mut()
            .filter(|_| !completed)
            .and_then(|bounded| bounded.passed(cx));
        if matches!(passed, Some(TimeoutKind::Total)) {
            return this.cut(TimeoutKind::Total);
        }
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        let polled = Pin::new(inner).poll_next(cx);
        match polled {
            Poll::Ready(Some(_)) => {
                // Any yielded item — data or error — ends the deferred
                // window: the backend has responded, so its handoff
                // state is resolved either way.
                this.saw_item = true;
                if let Some(bounded) = this.deadline.as_mut() {
                    bounded.item_arrived();
                }
            }
            Poll::Pending => {
                if let Some(kind) = passed {
                    return this.cut(kind);
                }
            }
            Poll::Ready(None) => {}
        }
        polled
    }
}

impl<T: Send + 'static, C: AbortOnDropClient> AbortOnDropStream<T, C> {
    /// The bound `kind` passed: the stream ends here with the bound's status
    /// and the worker's request is released (the abort at once).
    fn cut(&mut self, kind: TimeoutKind) -> Poll<Option<Result<T, tonic::Status>>> {
        let deadline = self
            .deadline
            .as_ref()
            .map(|bounded| bounded.deadline)
            .unwrap_or_else(|| RequestDeadline::after(Duration::ZERO));
        let status = deadline.status(kind);
        warn!(
            request_id = %self.request_id,
            bound = kind.label(),
            timeout = ?deadline.timeout(kind),
            "Request timed out: the worker did not answer within its bound; aborting it on the worker"
        );
        self.release();
        Poll::Ready(Some(Err(status)))
    }
}

/// A stream's bounds with their one timer (see
/// [`AbortOnDropStream::with_deadline`]).
struct Bounded {
    deadline: RequestDeadline,
    /// The instant the timer is armed for and the bound that passes there;
    /// `None` when nothing bounds the current phase (the timer then sleeps
    /// past any request's life).
    armed: Option<(tokio::time::Instant, TimeoutKind)>,
    sleep: Pin<Box<Sleep>>,
}

impl Bounded {
    /// A message arrived: the next applicable bound is counted from it. The
    /// timer moves only when the instant does (the total's never does), so a
    /// stream under the total alone costs no re-arm per message.
    fn item_arrived(&mut self) {
        let next = self.deadline.next_at(Some(tokio::time::Instant::now()));
        if next.map(|(at, _)| at) != self.armed.map(|(at, _)| at) {
            let at = next.map_or_else(far_future, |(at, _)| at);
            self.sleep.as_mut().reset(at);
        }
        self.armed = next;
    }

    /// Whether the armed bound has passed: polled, so its waker sees to the
    /// next poll when the worker sends nothing at all.
    fn passed(&mut self, cx: &mut Context<'_>) -> Option<TimeoutKind> {
        let (_, kind) = self.armed?;
        self.sleep.as_mut().poll(cx).is_ready().then_some(kind)
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

    /// A backend whose abort handler is wedged (answers the transport but never
    /// completes the RPC) must not hang the detached drop-abort task forever.
    /// The abort here never resolves, so the only way `abort_within` can return
    /// is the deadline firing — an unbounded await would deadlock this test.
    #[tokio::test]
    async fn abort_within_gives_up_on_a_hung_backend() {
        #[derive(Clone)]
        struct HangingClient;

        impl AbortOnDropClient for HangingClient {
            fn abort_for_drop(
                self,
                _request_id: String,
            ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
                Box::pin(async {
                    std::future::pending::<()>().await;
                    Ok(())
                })
            }
        }

        // Short real deadline keeps the test fast; it returns only because the
        // timeout fires against the never-resolving abort.
        abort_within(
            Duration::from_millis(10),
            HangingClient,
            "req-1".to_string(),
        )
        .await;
    }
}
