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

use futures::stream::Stream;
use tokio::sync::oneshot;
use tonic::Status;

use crate::stream::Choices;

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
/// warm-up names them `smg-warmup-<nonce>-<n>` (the Python servicer package's
/// `warmup.py` builds them; its tests pin the form) and sends them over the
/// servicer's own port, from the process that holds readiness. While
/// readiness is held, these are the only requests admitted; anything else
/// the gateway forwards in that window would share the engine's steps with
/// the warm-up's.
pub(crate) const WARMUP_REQUEST_ID_PREFIX: &str = "smg-warmup-";

/// Whether `request_id` has the exact form of a warm-up request: the prefix,
/// the `nonce` the servicer was constructed with and a `-` when it has one,
/// the request's ordinal and nothing else. The gateway derives a request's
/// id from a caller's `x-request-id` with a suffix of its own, so that path
/// cannot take this form; a body `rid` outside PD is forwarded as sent and
/// can spell the bare form, which is why the lifecycle owner draws a nonce
/// per process and hands it to the servicer and to its warm-up alone.
pub(crate) fn is_warmup_request_id(request_id: &str, nonce: Option<&str>) -> bool {
    let Some(rest) = request_id.strip_prefix(WARMUP_REQUEST_ID_PREFIX) else {
        return false;
    };
    let ordinal = match nonce {
        Some(nonce) => match rest
            .strip_prefix(nonce)
            .and_then(|rest| rest.strip_prefix('-'))
        {
            Some(ordinal) => ordinal,
            None => return false,
        },
        None => rest,
    };
    !ordinal.is_empty() && ordinal.bytes().all(|b| b.is_ascii_digit())
}

/// A registration's generation and the sender that cancels its stream.
type RegistryEntry = (u64, oneshot::Sender<CancelReason>);

/// Registered in-flight requests. The generation disambiguates an id reused
/// after an abort from the stream that registered it.
#[derive(Default)]
pub(crate) struct RequestRegistry {
    /// Keyed by the id the registration shares (one copy of the id per
    /// request, held by the map and by the registration that removes it).
    entries: Mutex<HashMap<Arc<str>, RegistryEntry>>,
    generation: AtomicU64,
    draining: AtomicBool,
    /// Readiness held for the planned warm-up: admission is the warm-up's
    /// own requests only.
    warming: AtomicBool,
    /// The mark the warm-up's own request ids carry
    /// (`ServicerConfig::warmup_nonce`); `None` admits the bare form.
    warmup_nonce: Option<String>,
    active: AtomicUsize,
}

/// Keeps a request's registry entry for the stream's lifetime; dropping it
/// removes the entry unless an abort already did (and the id was re-registered
/// since, which the generation tells apart).
pub(crate) struct Registration {
    registry: Arc<RequestRegistry>,
    request_id: Arc<str>,
    generation: u64,
    /// Whether this lifetime is in the active count (a client's request is,
    /// the servicer's own health probe is not).
    counted: bool,
}

impl Drop for Registration {
    /// Release this lifetime independently of the cancellation-map entry.
    fn drop(&mut self) {
        if self.counted {
            self.registry.active.fetch_sub(1, Ordering::AcqRel);
        }
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
    /// An empty registry whose readiness hold admits the warm-up ids that
    /// carry `warmup_nonce` (the bare `smg-warmup-<n>` under `None`).
    pub(crate) fn with_warmup_nonce(warmup_nonce: Option<String>) -> Self {
        Self {
            warmup_nonce,
            ..Self::default()
        }
    }

    /// Register `request_id` before its engine submit, so an `Abort` landing
    /// in the gap finds the entry. A live duplicate is refused: the id is the
    /// Router's cancellation handle, and the engine would refuse it too.
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: &str,
    ) -> Result<(Registration, oneshot::Receiver<CancelReason>), Status> {
        self.register_as(request_id, true)
    }

    /// [`register`](Self::register) for the servicer's own health probe: the
    /// entry is cancelled and settled like any request's, and the lifetime
    /// is outside the active count ([`active`](Self::active)), so a scrape
    /// during the probe's few milliseconds reads no request of a client's.
    pub(crate) fn register_uncounted(
        self: &Arc<Self>,
        request_id: &str,
    ) -> Result<(Registration, oneshot::Receiver<CancelReason>), Status> {
        self.register_as(request_id, false)
    }

    fn register_as(
        self: &Arc<Self>,
        request_id: &str,
        counted: bool,
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
        if self.warming.load(Ordering::Acquire)
            && !is_warmup_request_id(request_id, self.warmup_nonce.as_deref())
        {
            return Err(Status::unavailable("servicer is warming up"));
        }
        let request_id: Arc<str> = Arc::from(request_id);
        entries.insert(Arc::clone(&request_id), (generation, cancel_tx));
        if counted {
            self.active.fetch_add(1, Ordering::AcqRel);
        }
        drop(entries);
        Ok((
            Registration {
                registry: Arc::clone(self),
                request_id,
                generation,
                counted,
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
    /// The servicer's own health probe is registered outside this count.
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
            if let Some((_, cancel)) = entries.remove(request_id.as_str()) {
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
    choices: &mut Choices<C>,
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
#[path = "../tests/unit/requests/tests.rs"]
mod tests;
