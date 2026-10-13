//! In-flight request registry: the cancellation handles of the `Abort` RPC
//! and of shutdown, keyed by request id. Engine-neutral; every servicer
//! registers its streams here. A cancellation names its cause
//! ([`CancelReason`]), so a stream can end a caller's abort as a completion
//! and a shutdown under an open request as the failure it is.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
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

    /// A choice as the shutdown sees it; never polled here.
    struct Choice {
        terminal: Option<&'static str>,
        open: bool,
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
            let terminal = self.terminal.take();
            if terminal.is_some() {
                self.open = false;
            }
            terminal
        }

        fn is_open(&self) -> bool {
            self.open
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
        let mut all = choices(vec![
            Choice {
                terminal: Some("length"),
                open: true,
            },
            Choice {
                terminal: None,
                open: true,
            },
        ]);
        let (finished, cut) = settle_shutdown(&mut all);
        assert_eq!(finished, ["length"]);
        assert!(cut, "the choice still open is what the shutdown cuts");
        assert!(all.is_empty(), "the choices are dropped afterwards");
    }

    #[test]
    fn a_request_whose_every_choice_had_finished_is_not_cut() {
        let mut all = choices(vec![
            Choice {
                terminal: Some("stop"),
                open: true,
            },
            Choice {
                terminal: Some("length"),
                open: true,
            },
        ]);
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
        let mut all = choices(vec![Choice {
            terminal: None,
            open: false,
        }]);
        let (finished, cut) = settle_shutdown(&mut all);
        assert!(finished.is_empty());
        assert!(!cut);
    }
}
