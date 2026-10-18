//! One relay of a worker's streaming body to the client, for the HTTP
//! routers: the regular router's answer and the disaggregated router's decode
//! leg read through the same task. The head's success is the worker's before
//! the task can sample the body's failure; a body the worker ends with an
//! error after a successful head is accounted to its source and, when the
//! head was SSE, ended with the typed error event on its own frame; the load
//! guard rides in the body's own stream; a client that leaves ends the
//! upstream read. The regular relay re-chunks SSE deltas when the provider
//! profile asks; the decode relay reads each chunk through a hook (the
//! prefill log-probability merge and the `[DONE]` cut).

use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

use super::{
    sse::SSE_CHANNEL_BUFFER,
    sse_rechunk::{NoopObserver, SseRechunker, IDLE_FLUSH},
};
use crate::{
    routers::http::stream_failure::{
        mark_head_success_recorded, note_sent, RelaySource, UpstreamCut,
    },
    worker::{GuardedStream, WorkerLoadGuard},
};

/// What a chunk became through the relay's hook, and whether it ends the
/// relay (the decode leg ends at the `[DONE]` sentinel).
pub(crate) struct ChunkStep {
    pub(crate) bytes: Bytes,
    pub(crate) last: bool,
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
    /// The response headers, prepared by the caller.
    pub(crate) headers: HeaderMap,
    /// The regular relay's SSE re-chunker, when its provider asks for one.
    pub(crate) rechunker: Option<SseRechunker<NoopObserver>>,
    /// The decode relay's per-chunk read, before anything is sent.
    pub(crate) per_chunk: Option<Box<dyn FnMut(Bytes) -> ChunkStep + Send>>,
}

/// Relay `stream` to the client as the response's body. The head's success
/// is recorded before the task starts; the task runs under the request's
/// span; the response carries the head-success mark the callers read.
pub(crate) fn relay_streaming_body<S>(stream: S, relay: Relay) -> Response
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let Relay {
        status,
        upstream_sse,
        source,
        guard,
        headers,
        mut rechunker,
        mut per_chunk,
    } = relay;
    // Bounded channel applies backpressure: a slow client makes the
    // relay await on `send` instead of buffering the whole response.
    let (tx, rx) = mpsc::channel(SSE_CHANNEL_BUFFER);

    // The head's success is the worker's before the relay can sample
    // the body's failure: on a multi-threaded runtime the task may
    // run first with a body error already buffered behind the head,
    // and the head's success recorded by the caller afterwards would
    // reset the breaker's failure streak and erase the cut's sample.
    // A failed head is the caller's to record (through its ledger on
    // the typed route); the relay never accounts its body.
    let head_success_recorded = status.is_success();
    if head_success_recorded {
        source.worker().record_outcome(status.as_u16());
    }

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
            let mut sent_tail: Vec<u8> = Vec::with_capacity(4);
            // One timer, reset per chunk, instead of a fresh sleep per token.
            let idle = tokio::time::sleep(IDLE_FLUSH);
            tokio::pin!(idle);
            loop {
                tokio::select! {
                    chunk = stream.next() => {
                        if rechunker.is_some() {
                            idle.as_mut().reset(tokio::time::Instant::now() + IDLE_FLUSH);
                        }
                        match chunk {
                            // An empty upstream chunk must not become an empty
                            // h2 DATA frame toward the client.
                            Some(Ok(bytes)) if bytes.is_empty() => {}
                            Some(Ok(bytes)) => {
                                let (bytes, last) = match per_chunk.as_mut() {
                                    Some(read) => {
                                        let step = read(bytes);
                                        (step.bytes, step.last)
                                    }
                                    None => (bytes, false),
                                };
                                let bytes = match rechunker.as_mut() {
                                    Some(r) => r.feed(bytes),
                                    None => bytes,
                                };
                                if !bytes.is_empty() {
                                    relayed += bytes.len() as u64;
                                    note_sent(&mut sent_tail, &bytes);
                                    if tx.send(Ok(bytes)).await.is_err() {
                                        break;
                                    }
                                }
                                // The hook ended the stream (the decode leg's
                                // sentinel), or the observer did: its error
                                // event was the last frame, and letting go of
                                // the upstream cancels the engine's work.
                                if last || rechunker.as_ref().is_some_and(|r| r.stopped().is_some()) {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                if let Some(tail) = rechunker.as_mut().map(SseRechunker::finish) {
                                    if !tail.is_empty() {
                                        note_sent(&mut sent_tail, &tail);
                                        let _ = tx.send(Ok(tail)).await;
                                    }
                                }
                                // The observer ended the stream on its tail: its
                                // error event was the last frame. The cut stays the
                                // worker's to account.
                                if rechunker.as_ref().is_some_and(|r| r.stopped().is_some()) {
                                    if status.is_success() {
                                        source.account(&UpstreamCut::of(&e), relayed);
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
                                    if upstream_sse {
                                        // On its own frame: a cut inside a frame leaves
                                        // a partial line the event must not merge into.
                                        let event =
                                            source.error_event_on_its_own_frame(&cut, &sent_tail);
                                        let _ = tx.send(Ok(event)).await;
                                        break;
                                    }
                                }
                                let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                                break;
                            }
                            None => {
                                if let Some(tail) = rechunker.as_mut().map(SseRechunker::finish) {
                                    if !tail.is_empty() {
                                        let _ = tx.send(Ok(tail)).await;
                                    }
                                }
                                break;
                            }
                        }
                    },
                    () = &mut idle, if rechunker.as_ref().is_some_and(SseRechunker::has_pending) => {
                        idle.as_mut().reset(tokio::time::Instant::now() + IDLE_FLUSH);
                        if let Some(tail) = rechunker.as_mut().map(SseRechunker::flush_pending) {
                            if !tail.is_empty() && tx.send(Ok(tail)).await.is_err() {
                                break;
                            }
                        }
                        if rechunker.as_ref().is_some_and(|r| r.stopped().is_some()) {
                            break;
                        }
                    }
                    // Client gone with no chunk in flight (long prefill,
                    // stalled upstream): break so the upstream stream drops,
                    // closing the upstream connection and letting the
                    // engine abort generation.
                    () = tx.closed() => break,
                }
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
    if head_success_recorded {
        mark_head_success_recorded(&mut response);
    }
    response
}
