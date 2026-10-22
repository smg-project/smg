//! In-flight request registry: the cancellation handles of the `Abort` RPC
//! and of shutdown, keyed by request id. Engine-neutral; every servicer
//! registers its streams here, so the registry is also the admission gate:
//! a drain closes it, and a readiness hold (the lifecycle owner's planned
//! warm-up before SERVING) closes it to every request but the warm-up's own.
//! A cancellation names its cause ([`CancelReason`]), so a stream can end a
//! caller's abort as a completion and a shutdown under an open request as
//! the failure it is.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use futures::stream::{SelectAll, Stream};
use tokio::sync::oneshot;
use tonic::Status;

/// Why a registered request's stream is being cancelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelReason {
    /// An `Abort` RPC named the request: the caller's own abort (the Router
    /// drops the stream and sends it). The stream ends the way the Python
    /// servicer ends it, with each open choice's `Complete(finish_reason =
    /// "abort")`.
    Abort,
    /// The servicer is shutting down with the request still open: its engine
    /// side is going away (the lifecycle owner stopping the pair after the
    /// engine exited, a drain whose grace ran out). Nobody asked for an abort
    /// and the request did not complete, so the stream ends with a status,
    /// never with a completion the caller could read as a short answer.
    Shutdown,
}

/// The message of the `UNAVAILABLE` status a shutdown ends an open stream
/// with.
pub(crate) const SHUTDOWN_MESSAGE: &str =
    "servicer shutting down: the engine went away before the request completed";

/// The request-id prefix of the lifecycle owner's own requests: its planned
/// warm-up names them `smg-warmup-<n>` (the Python servicer package's
/// `warmup.py` builds them; its tests pin the form) and sends them over the
/// servicer's own port, from the process that holds readiness. While
/// readiness is held, these are the only requests admitted; anything else
/// the gateway forwards in that window would share the engine's steps with
/// the warm-up's.
pub(crate) const WARMUP_REQUEST_ID_PREFIX: &str = "smg-warmup-";

/// Whether `request_id` has the exact form of a warm-up request, the prefix
/// followed by the request's ordinal and nothing else. The gateway derives a
/// request's id from a caller's `x-request-id` with a suffix of its own, so
/// an id a client chose to start with the prefix does not take this form.
pub(crate) fn is_warmup_request_id(request_id: &str) -> bool {
    request_id
        .strip_prefix(WARMUP_REQUEST_ID_PREFIX)
        .is_some_and(|ordinal| !ordinal.is_empty() && ordinal.bytes().all(|b| b.is_ascii_digit()))
}

/// Registered in-flight requests. The generation disambiguates an id reused
/// after an abort from the stream that registered it.
#[derive(Default)]
pub(crate) struct RequestRegistry {
    entries: Mutex<HashMap<String, (u64, oneshot::Sender<CancelReason>)>>,
    generation: AtomicU64,
    draining: AtomicBool,
    /// Readiness held for the planned warm-up: admission is the warm-up's
    /// own requests only.
    warming: AtomicBool,
    active: AtomicUsize,
}

/// Keeps a request's registry entry for the stream's lifetime; dropping it
/// removes the entry unless an abort already did (and the id was re-registered
/// since, which the generation tells apart).
pub(crate) struct Registration {
    registry: Arc<RequestRegistry>,
    request_id: String,
    generation: u64,
}

impl Drop for Registration {
    /// Release this lifetime independently of the cancellation-map entry.
    fn drop(&mut self) {
        self.registry.active.fetch_sub(1, Ordering::AcqRel);
        if let Ok(mut entries) = self.registry.entries.lock() {
            if entries
                .get(&self.request_id)
                .is_some_and(|(generation, _)| *generation == self.generation)
            {
                entries.remove(&self.request_id);
            }
        }
    }
}

impl RequestRegistry {
    /// Register `request_id` before its engine submit, so an `Abort` landing
    /// in the gap finds the entry. A live duplicate is refused: the id is the
    /// Router's cancellation handle, and the engine would refuse it too.
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: &str,
    ) -> Result<(Registration, oneshot::Receiver<CancelReason>), Status> {
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let generation = self.generation.fetch_add(1, Ordering::Relaxed);
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| Status::internal("request registry is poisoned"))?;
        if entries.contains_key(request_id) {
            return Err(Status::already_exists(format!(
                "request {request_id} is already in flight"
            )));
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::unavailable("servicer is draining"));
        }
        if self.warming.load(Ordering::Acquire) && !is_warmup_request_id(request_id) {
            return Err(Status::unavailable("servicer is warming up"));
        }
        entries.insert(request_id.to_string(), (generation, cancel_tx));
        self.active.fetch_add(1, Ordering::AcqRel);
        drop(entries);
        Ok((
            Registration {
                registry: Arc::clone(self),
                request_id: request_id.to_string(),
                generation,
            },
            cancel_rx,
        ))
    }

    /// Serialize admission with the drain boundary. Once closed, a zero
    /// active count cannot race with a newly admitted request.
    pub(crate) fn set_accepting(&self, accepting: bool) {
        let _entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        self.draining.store(!accepting, Ordering::Release);
    }

    /// Whether admission is open: no drain has closed it.
    pub(crate) fn is_accepting(&self) -> bool {
        !self.draining.load(Ordering::Acquire)
    }

    /// Serialize admission with the warm-up boundary: while readiness is
    /// held (`true`), only the lifecycle owner's own requests (the planned
    /// warm-up's, by their id) are admitted, so nothing else reaches the
    /// engine beside the warm-up's steps; released (`false`), every request
    /// is admitted again.
    pub(crate) fn set_warming(&self, warming: bool) {
        let _entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        self.warming.store(warming, Ordering::Release);
    }

    /// Count request lifetimes, including aborted streams still unwinding.
    /// The cancellation map alone loses those entries before their cleanup.
    pub(crate) fn active(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    /// Fire the cancellation of every given id that is live, as the caller's
    /// abort ([`CancelReason::Abort`]); returns how many were. An unknown or
    /// already-finished id is a no-op: cleanup is idempotent.
    pub(crate) fn abort(&self, request_ids: &[String]) -> Result<usize, Status> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| Status::internal("request registry is poisoned"))?;
        let mut found = 0;
        for request_id in request_ids {
            if let Some((_, cancel)) = entries.remove(request_id) {
                let _ = cancel.send(CancelReason::Abort);
                found += 1;
            }
        }
        Ok(found)
    }

    /// Fire every registered cancellation as the shutdown it is
    /// ([`CancelReason::Shutdown`]): the streams end before the listener
    /// closes, each with the status a request the engine left behind gets.
    pub(crate) fn cancel_all(&self) -> Result<(), crate::ServicerError> {
        let mut entries = crate::lock(&self.entries)?;
        for (_, (_, cancel)) in entries.drain() {
            let _ = cancel.send(CancelReason::Shutdown);
        }
        Ok(())
    }

    /// In-flight requests.
    pub(crate) fn len(&self) -> u32 {
        self.entries
            .lock()
            .map(|entries| u32::try_from(entries.len()).unwrap_or(u32::MAX))
            .unwrap_or(0)
    }
}

/// A choice of a request's stream as a shutdown sees it: it may already hold
/// its terminal (the engine finished it on the tick before, or a stop string
/// matched, and the `Complete` waits for the next poll), still be open, or
/// have ended.
pub(crate) trait HeldTerminal {
    type Response;
    /// The terminal this choice already holds, taken; `None` for a choice
    /// still open (left as it is) or already ended.
    fn finished(&mut self) -> Option<Self::Response>;
    /// Whether the choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool;
}

/// What a shutdown yields for a request's choices, in this order: every
/// terminal a choice already holds (those completions go out as they would
/// have), then whether a choice was still really open (only then does the
/// stream end with a status), then the choices are dropped (which aborts the
/// engine side). Taking the terminals first is the point: a request the
/// engine had finished on the tick before the shutdown keeps its completion.
pub(crate) fn settle_shutdown<C: HeldTerminal + Stream + Unpin>(
    choices: &mut SelectAll<C>,
) -> (VecDeque<C::Response>, bool) {
    let finished = choices
        .iter_mut()
        .filter_map(HeldTerminal::finished)
        .collect();
    let cut = choices.iter().any(HeldTerminal::is_open);
    choices.clear();
    (finished, cut)
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        task::{Context, Poll},
    };

    use super::*;

    /// Aborting removes the handle, but only dropping the guard finishes a request.
    #[test]
    fn drain_closes_admission_and_counts_aborted_lifetimes() {
        let registry = Arc::new(RequestRegistry::default());
        let (first, _cancel) = registry.register("first").unwrap();
        assert_eq!(registry.active(), 1);
        registry.abort(&["first".to_string()]).unwrap();
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.active(), 1);
        let (second, _cancel) = registry.register("first").unwrap();
        registry.set_accepting(false);
        assert!(
            matches!(registry.register("late"), Err(e) if e.code() == tonic::Code::Unavailable)
        );
        drop(first);
        assert_eq!(registry.active(), 1);
        assert_eq!(registry.len(), 1);
        registry.cancel_all().unwrap();
        assert_eq!(registry.active(), 1);
        drop(second);
        assert_eq!(registry.active(), 0);
        registry.set_accepting(true);
        assert!(registry.register("resumed").is_ok());
    }

    /// While readiness is held for the planned warm-up, the warm-up's own
    /// requests are admitted and every other id is refused as not serving;
    /// the drain refuses the warm-up's own as well; the release admits all.
    #[test]
    fn a_readiness_hold_admits_only_the_warm_ups_own_requests() {
        let registry = Arc::new(RequestRegistry::default());
        registry.set_warming(true);
        assert!(registry.is_accepting(), "a hold is not a drain");
        let refused = registry
            .register("chatcmpl-1")
            .err()
            .expect("a foreign request is refused");
        assert_eq!(refused.code(), tonic::Code::Unavailable);
        assert_eq!(refused.message(), "servicer is warming up");
        assert_eq!(registry.active(), 0, "a refused request has no lifetime");
        let (own, _cancel) = registry
            .register(&format!("{WARMUP_REQUEST_ID_PREFIX}7"))
            .expect("the warm-up's own request passes");
        assert_eq!(registry.active(), 1);
        drop(own);
        // The prefix alone is not the form: a caller's id the gateway stamped
        // with its own suffix, or the bare prefix, is a foreign request.
        for crafted in [
            "smg-warmup-x-01a16586-c52f-76e8-bb8f-9f407fd9f836",
            "smg-warmup-1-01a16586-c52f-76e8-bb8f-9f407fd9f836",
            "smg-warmup-",
            "SMG-WARMUP-1",
        ] {
            let refused = registry
                .register(crafted)
                .err()
                .expect("an id that only starts with the prefix is refused");
            assert_eq!(refused.message(), "servicer is warming up", "{crafted}");
        }

        registry.set_accepting(false);
        let drained = registry
            .register(&format!("{WARMUP_REQUEST_ID_PREFIX}8"))
            .err()
            .expect("a drain closes admission to the warm-up as well");
        assert_eq!(drained.message(), "servicer is draining");
        registry.set_accepting(true);

        registry.set_warming(false);
        let (late, _cancel) = registry
            .register("chatcmpl-1")
            .expect("the release admits every request");
        drop(late);
        assert_eq!(registry.active(), 0);
    }

    /// A choice as the shutdown sees it, in the shape of the servicers' own:
    /// a `Complete` queued for the next poll, an engine stream that may hold
    /// the engine's parked `Complete`, or no engine stream once the
    /// completion is out (or the choice ended). Never polled here.
    struct Choice {
        pending: Option<&'static str>,
        inner: Option<Inner>,
    }

    struct Inner {
        parked: Option<&'static str>,
    }

    impl Choice {
        fn open() -> Self {
            Self {
                pending: None,
                inner: Some(Inner { parked: None }),
            }
        }

        fn queued(terminal: &'static str) -> Self {
            Self {
                pending: Some(terminal),
                inner: Some(Inner { parked: None }),
            }
        }

        fn parked(terminal: &'static str) -> Self {
            Self {
                pending: None,
                inner: Some(Inner {
                    parked: Some(terminal),
                }),
            }
        }

        /// The completion already went out (or the choice ended).
        fn done() -> Self {
            Self {
                pending: None,
                inner: None,
            }
        }
    }

    impl Stream for Choice {
        type Item = &'static str;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl HeldTerminal for Choice {
        type Response = &'static str;

        fn finished(&mut self) -> Option<&'static str> {
            if let Some(terminal) = self.pending.take() {
                self.inner = None;
                return Some(terminal);
            }
            let parked = self.inner.as_mut()?.parked.take()?;
            self.inner = None;
            Some(parked)
        }

        fn is_open(&self) -> bool {
            self.pending.is_none() && self.inner.is_some()
        }
    }

    fn choices(items: Vec<Choice>) -> SelectAll<Choice> {
        let mut all = SelectAll::new();
        for item in items {
            all.push(item);
        }
        all
    }

    #[test]
    fn a_held_terminal_goes_out_and_an_open_choice_is_cut() {
        let mut all = choices(vec![Choice::queued("length"), Choice::open()]);
        let (finished, cut) = settle_shutdown(&mut all);
        assert_eq!(finished, ["length"]);
        assert!(cut, "the choice still open is what the shutdown cuts");
        assert!(all.is_empty(), "the choices are dropped afterwards");
    }

    #[test]
    fn a_request_whose_every_choice_had_finished_is_not_cut() {
        let mut all = choices(vec![Choice::queued("stop"), Choice::parked("length")]);
        let (finished, cut) = settle_shutdown(&mut all);
        // `SelectAll` has no order of its own; both completions are there.
        let mut finished: Vec<_> = finished.into_iter().collect();
        finished.sort_unstable();
        assert_eq!(finished, ["length", "stop"]);
        assert!(
            !cut,
            "nothing was open: the stream ends with its completions, no status"
        );
    }

    #[test]
    fn an_ended_choice_neither_yields_nor_cuts() {
        let mut all = choices(vec![Choice::done()]);
        let (finished, cut) = settle_shutdown(&mut all);
        assert!(finished.is_empty());
        assert!(!cut);
    }

    /// A choice whose completion already went out next to one still
    /// generating: only the latter is cut. (That a choice stream drops its
    /// engine stream as the engine's own `Complete` goes out, so it is
    /// `done` here and not `open`, is pinned per servicer by
    /// `a_shutdown_after_the_engines_completion_ends_the_stream_cleanly`.)
    #[test]
    fn a_choice_whose_completion_already_went_out_is_not_cut() {
        let mut mixed = choices(vec![Choice::done(), Choice::open()]);
        let (finished, cut) = settle_shutdown(&mut mixed);
        assert!(finished.is_empty());
        assert!(cut, "the choice still generating is the one cut");
    }
}
