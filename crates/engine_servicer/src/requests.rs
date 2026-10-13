//! In-flight request registry: the cancellation handles of the `Abort` RPC
//! and of shutdown, keyed by request id. Engine-neutral; every servicer
//! registers its streams here. A cancellation names its cause
//! ([`CancelReason`]), so a stream can end a caller's abort as a completion
//! and a shutdown under an open request as the failure it is.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

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

/// Registered in-flight requests. The generation disambiguates an id reused
/// after an abort from the stream that registered it.
#[derive(Default)]
pub(crate) struct RequestRegistry {
    entries: Mutex<HashMap<String, (u64, oneshot::Sender<CancelReason>)>>,
    generation: AtomicU64,
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
    fn drop(&mut self) {
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
        entries.insert(request_id.to_string(), (generation, cancel_tx));
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
