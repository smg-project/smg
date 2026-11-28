//! One relay of a worker's streaming body to the client, for the HTTP
//! routers: the regular router's answer and the disaggregated router's decode
//! side read through the same task. A closed breaker samples a stream once,
//! at its end; on a breaker that is not closed the head's success is the
//! worker's before the task can sample the body's failure; a body the worker
//! ends with an error after a successful head is accounted to its source
//! and, when the head was SSE, ended with the typed error event on its own
//! frame; the load guard rides in the body's own stream; a client that
//! leaves ends the upstream read. The regular relay re-chunks SSE deltas
//! when the wire of the model the client asked for asks, by the wire's
//! numbers; the decode relay reads each chunk through a hook (the prefill
//! log-probability merge and the `[DONE]` cut).

use std::time::Duration;

use axum::{
    body::Body,
    http::{header::CONTENT_LENGTH, HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use smg_worker::{health::circuit_breaker::CircuitState, identity::basic::WorkerLoadGuard};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

use super::stream_failure::{
    mark_head_success_recorded, note_sent, RelaySource, SentTail, UpstreamCut,
};
use crate::routers::common::{
    placement::RequestEnd,
    sse::SSE_CHANNEL_BUFFER,
    sse_rechunk::{NoopObserver, SseRechunker},
    stream::GuardedStream,
};

/// What a chunk became through the relay's hook, and whether it ends the
/// relay (the decode side ends at the `[DONE]` sentinel).
pub(crate) struct ChunkStep {
    pub(crate) bytes: Bytes,
    pub(crate) last: bool,
}

/// The decode side's reading of its stream: what each chunk becomes, whether
/// it ends the relay, and what the hook still holds when the upstream ends.
/// A closure is a hook that holds nothing.
pub(crate) trait ChunkHook: Send {
    /// One upstream chunk, read.
    fn step(&mut self, chunk: Bytes) -> ChunkStep;

    /// The bytes held back for a later chunk (a frame cut short of its
    /// delimiter) when the upstream ends: sent before the stream's end or
    /// its error event. Nothing by default.
    fn finish(&mut self) -> Bytes {
        Bytes::new()
    }
}

impl<F: FnMut(Bytes) -> ChunkStep + Send> ChunkHook for F {
    fn step(&mut self, chunk: Bytes) -> ChunkStep {
        self(chunk)
    }
}

/// How the relay reads each upstream chunk: as it is; through the SSE
/// re-chunker the regular router's wire asks for; or through
/// the decode side's hook (the prefill log-probability merge and the
/// `[DONE]` cut). One of the three, so a re-chunker's held text can never be
/// cut short by a hook that ends the relay.
pub(crate) enum ChunkMode {
    Plain,
    Rechunk(Box<SseRechunker<NoopObserver>>),
    Hook(Box<dyn ChunkHook>),
}

impl ChunkMode {
    /// Whether the re-chunker's observer ended the stream (its error event
    /// was the last frame).
    fn stopped(&self) -> bool {
        matches!(self, Self::Rechunk(rechunker) if rechunker.stopped().is_some())
    }

    /// Whether the re-chunker holds text an idle flush should release.
    fn has_pending(&self) -> bool {
        matches!(self, Self::Rechunk(rechunker) if rechunker.has_pending())
    }

    /// The re-chunker's idle flush; the timer is never armed without one.
    fn idle_flush(&self) -> Duration {
        match self {
            Self::Rechunk(rechunker) => rechunker.policy().idle_flush,
            Self::Plain | Self::Hook(_) => Duration::ZERO,
        }
    }

    /// The re-chunker's or the hook's tail at the stream's end, or nothing.
    fn finish(&mut self) -> Bytes {
        match self {
            Self::Rechunk(rechunker) => rechunker.finish(),
            Self::Hook(hook) => hook.finish(),
            Self::Plain => Bytes::new(),
        }
    }
}

/// The relay of one streaming answer.
pub(crate) struct Relay {
    /// The head's status, answered as it came.
    pub(crate) status: StatusCode,
    /// Whether the upstream body is SSE (the worker said so, or a successful
    /// stream said nothing): a cut after a successful head is then ended
    /// with the typed error event.
    pub(crate) upstream_sse: bool,
    /// Where the body comes from, for the accounting of a cut.
    pub(crate) source: RelaySource,
    /// The worker's load, released when the body is dropped.
    pub(crate) guard: WorkerLoadGuard,
    /// The handle the relay reports the stream's end through: its caller's
    /// word on whether its route tells Grand Central its requests'
    /// ends. A route that tells them asks it of `guard`
    /// ([`RequestEnd::of`]). A route that tells none hands the empty
    /// handle, and the relay says nothing: the pair route, until it tells
    /// every answer of its two sides.
    pub(crate) end: RequestEnd,
    /// The response headers, prepared by the caller. Under `upstream_sse` the
    /// relay removes the upstream's `Content-Length` from them, since it may
    /// lengthen the body.
    pub(crate) headers: HeaderMap,
    /// How each chunk is read before it is sent.
    pub(crate) mode: ChunkMode,
}

/// Relay `stream` to the client as the response's body. A closed breaker
/// samples the stream once, at its end; on a breaker that is not closed the
/// head's success is recorded before the task starts; the task runs under
/// the request's span; the response carries the head-success mark the
/// callers read.
pub(crate) fn relay_streaming_body<S>(stream: S, relay: Relay) -> Response
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let Relay {
        status,
        upstream_sse,
        source,
        guard,
        end,
        mut headers,
        mut mode,
    } = relay;
    // The relay may change an SSE body's length: re-chunking re-slices it,
    // a decode-side hook re-encodes it, and a cut after a successful head
    // appends the error event and the stream's end. So a streamed SSE body
    // never carries the upstream's `Content-Length`, whichever router
    // relays it.
    if upstream_sse {
        headers.remove(CONTENT_LENGTH);
    }
    // Bounded channel applies backpressure: a slow client makes the
    // relay await on `send` instead of buffering the whole response.
    let (tx, rx) = mpsc::channel(SSE_CHANNEL_BUFFER);

    // A good head's stream is one sample of a closed breaker, taken by the
    // task at the stream's end: the cut's failure alone, or the success of
    // an end the worker did not cut. A success recorded at the head would
    // reset the failure streak the cut then counts into, and a worker that
    // cuts every stream behind a good head would read one failure for ever
    // and never open. On a breaker that is not closed the head's success
    // stays the worker's before the relay can sample the body's failure: a
    // trial is answered by its head, and on a multi-threaded runtime the
    // task may run first with a body error already buffered behind the
    // head, so the head's success recorded by the caller afterwards would
    // reset the breaker's failure streak and erase the cut's sample.
    // The two ends differ there on purpose: a good head is the worker
    // answering now, whichever request it answers, and a trial cannot wait
    // for a stream's end, so the head counts; the end's success is the
    // closed breaker's sample of the whole stream, and a stream that began
    // before a trip is no part of the trial.
    // A failed head is the caller's to record (through its ledger on
    // the typed route); the relay never accounts its body.
    // The read below is the worker's one read of its breaker, and it moves
    // an open breaker whose timeout has passed to half-open as a
    // selection's read does: the trial then starts at this head, and the
    // routing state's series follows at the next selection's read, which
    // publishes the move it finds made.
    let good_head = status.is_success();
    let sampled_at_end =
        good_head && source.worker().circuit_breaker_state() == CircuitState::Closed;
    if good_head && !sampled_at_end {
        source.worker().record_outcome(status.as_u16());
    }
    // A good head's end is the relay's to report, as its cut is the
    // relay's to account: the task holds the handle its caller handed. A
    // failed head is the caller's to report with its record, so the handle
    // is let go here and that report waits for no task. A client that
    // leaves is reported by nobody.
    let end = if good_head {
        end
    } else {
        drop(end);
        RequestEnd::default()
    };

    // The relay runs under the request's span, so a failure after
    // the head went out is logged with the request's id.
    let span = tracing::Span::current();
    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
    )]
    tokio::spawn(
        async move {
            let mut stream = std::pin::pin!(stream);
            // What the client has been sent, for the record of a cut, and
            // its last bytes, so the event that ends a cut SSE body is
            // written on a frame boundary.
            let mut relayed: u64 = 0;
            let mut sent_tail = SentTail::default();
            // One timer, reset per chunk, instead of a fresh sleep per token.
            let idle_flush = mode.idle_flush();
            let idle = tokio::time::sleep(idle_flush);
            tokio::pin!(idle);
            // One wait for the client's leaving, kept across the chunks: a
            // `closed()` built per `select!` round registers a waiter on the
            // channel and drops it on every chunk; kept, it registers once.
            let client_gone = tx.closed();
            tokio::pin!(client_gone);
            // The success a closed breaker is owed at the stream's end,
            // unless the worker cuts the stream: the cut is then its sample.
            let mut owes_success = sampled_at_end;
            loop {
                tokio::select! {
                    chunk = stream.next() => {
                        if let ChunkMode::Rechunk(_) = mode {
                            idle.as_mut().reset(tokio::time::Instant::now() + idle_flush);
                        }
                        match chunk {
                            // An empty upstream chunk must not become an empty
                            // h2 DATA frame toward the client.
                            Some(Ok(bytes)) if bytes.is_empty() => {}
                            Some(Ok(bytes)) => {
                                let (bytes, last) = match &mut mode {
                                    ChunkMode::Plain => (bytes, false),
                                    ChunkMode::Rechunk(rechunker) => (rechunker.feed(bytes), false),
                                    ChunkMode::Hook(hook) => {
                                        let step = hook.step(bytes);
                                        (step.bytes, step.last)
                                    }
                                };
                                if !bytes.is_empty() {
                                    relayed += bytes.len() as u64;
                                    note_sent(&mut sent_tail, &bytes);
                                    if tx.send(Ok(bytes)).await.is_err() {
                                        break;
                                    }
                                }
                                // The hook ended the stream (the decode side's
                                // sentinel), or the observer did: its error
                                // event was the last frame, and letting go of
                                // the upstream cancels the engine's work.
                                if last || mode.stopped() {
                                    if last {
                                        end.report(status.as_u16(), || true);
                                    }
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                let tail = mode.finish();
                                if !tail.is_empty() {
                                    note_sent(&mut sent_tail, &tail);
                                    let _ = tx.send(Ok(tail)).await;
                                }
                                // The observer ended the stream on its tail: its
                                // error event was the last frame. The cut stays the
                                // worker's to account.
                                if mode.stopped() {
                                    if status.is_success() {
                                        let cut = UpstreamCut::of(&e);
                                        source.account(&cut, relayed);
                                        owes_success = false;
                                        end.report(cut.status().as_u16(), || false);
                                    }
                                    break;
                                }
                                // A head that was a success was answered as one; the
                                // body ending in an error after it is the worker's
                                // failure, counted here, and an SSE response is ended
                                // with the typed event and the stream's own end. A
                                // failed head was counted when it arrived; its body
                                // is cut as before.
                                if status.is_success() {
                                    let cut = UpstreamCut::of(&e);
                                    source.account(&cut, relayed);
                                    owes_success = false;
                                    end.report(cut.status().as_u16(), || false);
                                    if upstream_sse {
                                        // On its own frame: a cut inside a frame leaves
                                        // a partial line the event must not merge into.
                                        let event =
                                            source.error_event_on_its_own_frame(&cut, sent_tail.as_slice());
                                        let _ = tx.send(Ok(event)).await;
                                        break;
                                    }
                                }
                                let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                                break;
                            }
                            None => {
                                let tail = mode.finish();
                                if !tail.is_empty() {
                                    let _ = tx.send(Ok(tail)).await;
                                }
                                end.report(status.as_u16(), || true);
                                break;
                            }
                        }
                    },
                    () = &mut idle, if mode.has_pending() => {
                        idle.as_mut().reset(tokio::time::Instant::now() + idle_flush);
                        if let ChunkMode::Rechunk(rechunker) = &mut mode {
                            let tail = rechunker.flush_pending();
                            if !tail.is_empty() && tx.send(Ok(tail)).await.is_err() {
                                break;
                            }
                        }
                        if mode.stopped() {
                            break;
                        }
                    }
                    // Client gone with no chunk in flight (long prefill,
                    // stalled upstream): break so the upstream stream drops,
                    // closing the upstream connection and letting the
                    // engine abort generation.
                    () = &mut client_gone => break,
                }
            }
            // The closed breaker's one sample of a stream the worker did
            // not cut: a clean end, a hook's or an observer's stop, a
            // client that left. A breaker that is not closed at the stream's
            // end takes none: the success belongs to the closed breaker the
            // head found, and on a tripped one it would count toward a
            // trial this stream never was. The state is read at the two ends
            // alone, so a breaker that tripped and closed again in between
            // takes the success as one that stayed closed does. The read is
            // the head's: it moves an open breaker past its timeout to
            // half-open, with no sample.
            if owes_success && source.worker().circuit_breaker_state() == CircuitState::Closed {
                source.worker().record_outcome(status.as_u16());
            }
        }
        .instrument(span),
    );

    // The load guard rides in the body's own stream: released when
    // the body is dropped, at its end or when the client leaves.
    let body = Body::from_stream(GuardedStream::new(ReceiverStream::new(rx), guard));

    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    // Every good head carries the mark: the breaker's sample of its stream
    // is the relay's, at the head or at the end, and the caller records none.
    if good_head {
        mark_head_success_recorded(&mut response);
    }
    response
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/http/stream_relay/tests.rs"]
mod tests;
