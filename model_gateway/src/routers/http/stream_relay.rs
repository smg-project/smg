//! One relay of a worker's streaming body to the client, for the HTTP
//! routers: the regular router's answer and the disaggregated router's decode
//! side read through the same task. The head's success is the worker's before
//! the task can sample the body's failure; a body the worker ends with an
//! error after a successful head is accounted to its source and, when the
//! head was SSE, ended with the typed error event on its own frame; the load
//! guard rides in the body's own stream; a client that leaves ends the
//! upstream read. The regular relay re-chunks SSE deltas when the wire of
//! the model the client asked for asks, by the wire's numbers; the decode
//! relay reads each chunk through a hook (the prefill log-probability merge
//! and the `[DONE]` cut).

use std::time::Duration;

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

use super::stream_failure::{
    mark_head_success_recorded, note_sent, RelaySource, SentTail, UpstreamCut,
};
use crate::{
    routers::common::{
        sse::SSE_CHANNEL_BUFFER,
        sse_rechunk::{NoopObserver, SseRechunker},
    },
    worker::{GuardedStream, WorkerLoadGuard},
};

/// What a chunk became through the relay's hook, and whether it ends the
/// relay (the decode side ends at the `[DONE]` sentinel).
pub(crate) struct ChunkStep {
    pub(crate) bytes: Bytes,
    pub(crate) last: bool,
}

/// How the relay reads each upstream chunk: as it is; through the SSE
/// re-chunker the regular router's wire asks for; or through
/// the decode side's hook (the prefill log-probability merge and the
/// `[DONE]` cut). One of the three, so a re-chunker's held text can never be
/// cut short by a hook that ends the relay.
pub(crate) enum ChunkMode {
    Plain,
    Rechunk(Box<SseRechunker<NoopObserver>>),
    Hook(Box<dyn FnMut(Bytes) -> ChunkStep + Send>),
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

    /// The re-chunker's tail at the stream's end, or nothing.
    fn finish(&mut self) -> Bytes {
        match self {
            Self::Rechunk(rechunker) => rechunker.finish(),
            Self::Plain | Self::Hook(_) => Bytes::new(),
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
    /// The response headers, prepared by the caller.
    pub(crate) headers: HeaderMap,
    /// How each chunk is read before it is sent.
    pub(crate) mode: ChunkMode,
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
        mut mode,
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
                                    ChunkMode::Hook(read) => {
                                        let step = read(bytes);
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

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

    use axum::{
        http::{HeaderMap, StatusCode},
        response::Response,
    };
    use bytes::Bytes;
    use futures_util::{stream, Stream, StreamExt};
    use http_body_util::BodyExt;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use super::{relay_streaming_body, ChunkMode, ChunkStep, Relay};
    use crate::{
        routers::{
            common::{
                sse::SSE_CHANNEL_BUFFER,
                sse_rechunk::{RechunkPolicy, SseRechunker},
            },
            grpc::wire::Wire,
            http::stream_failure::RelaySource,
        },
        worker::{test_support::RefusedAddress, BasicWorkerBuilder, Worker, WorkerLoadGuard},
    };

    /// The relay of `upstream` in `mode`: a successful SSE head from a worker
    /// at a port nobody listens on.
    fn relay_of<S>(upstream: S, mode: ChunkMode) -> Response
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:1").build());
        relay_streaming_body(
            upstream,
            Relay {
                status: StatusCode::OK,
                upstream_sse: true,
                source: RelaySource::new("m", "/v1/chat/completions", Arc::clone(&worker)),
                guard: WorkerLoadGuard::new(worker, None),
                headers: HeaderMap::new(),
                mode,
            },
        )
    }

    /// The frames the client gets when `upstream` runs through the relay in
    /// `mode`, read to the body's end.
    async fn frames_from<S>(upstream: S, mode: ChunkMode) -> Vec<Bytes>
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut body = relay_of(upstream, mode).into_body();
        let mut frames = Vec::new();
        while let Some(frame) = body.frame().await {
            if let Ok(data) = frame.expect("a relayed frame").into_data() {
                frames.push(data);
            }
        }
        frames
    }

    /// The frames the client gets when the upstream yields `chunks` through
    /// the relay in `mode`.
    async fn frames_through(chunks: Vec<&'static [u8]>, mode: ChunkMode) -> Vec<Bytes> {
        let upstream = stream::iter(
            chunks
                .into_iter()
                .map(|chunk| Ok::<_, reqwest::Error>(Bytes::from_static(chunk))),
        );
        frames_from(upstream, mode).await
    }

    /// The one re-slicing policy in the tree.
    fn re_slicing_policy() -> RechunkPolicy {
        Wire::Minimax
            .rechunk_policy()
            .expect("the wire re-slices its streams")
    }

    /// A relay through the re-chunker of that policy.
    fn re_sliced() -> ChunkMode {
        ChunkMode::Rechunk(Box::new(SseRechunker::new(re_slicing_policy())))
    }

    /// One chat chunk frame carrying `text` as its content delta.
    fn content_event(text: &str) -> Bytes {
        Bytes::from(format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        ))
    }

    /// Two content deltas the re-chunker holds: together under the emit
    /// threshold, and at least the minimum tail an idle flush emits.
    const HELD: [&str; 2] = ["abc", "def"];
    /// The text of [`HELD`] once it reaches the client.
    const HELD_TEXT: &str = "abcdef";

    /// [`HELD`] as chat chunk frames, the policy's bounds checked once.
    fn held_deltas() -> Vec<Bytes> {
        let policy = re_slicing_policy();
        let chars = HELD_TEXT.chars().count();
        assert!(
            policy.min_tail_chars <= chars && chars < policy.emit_threshold,
            "the held text must reach the minimum tail and stay under the emit threshold"
        );
        HELD.iter().map(|text| content_event(text)).collect()
    }

    /// The content text of every chat chunk event in `frames`, each with the
    /// index of the frame it came in.
    fn content_of(frames: &[Bytes]) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let text = String::from_utf8_lossy(frame);
            for data in text
                .split("\n\n")
                .filter_map(|f| f.trim_start().strip_prefix("data: "))
                .filter(|d| *d != "[DONE]")
            {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = value["choices"][0]["delta"]["content"].as_str() {
                        out.push((index, content.to_string()));
                    }
                }
            }
        }
        out
    }

    /// A transport error from a connect refused at an address nobody
    /// listens at, as `reqwest` reports it.
    async fn refused_call() -> reqwest::Error {
        let refused = RefusedAddress::reserve();
        crate::server::http_client_builder()
            .build()
            .unwrap()
            .get(format!("http://{}/v1/chat/completions", refused.addr()))
            .send()
            .await
            .unwrap_err()
    }

    /// An upstream that closes its body without a closing frame still
    /// delivers the text the re-chunker held under its emit threshold: the
    /// relay writes the re-chunker's tail at the clean end.
    #[tokio::test]
    async fn a_clean_upstream_end_delivers_the_re_chunkers_held_text() {
        let upstream = stream::iter(held_deltas().into_iter().map(Ok::<_, reqwest::Error>));
        let frames = frames_from(upstream, re_sliced()).await;
        let text: String = content_of(&frames).into_iter().map(|(_, t)| t).collect();
        assert_eq!(text, HELD_TEXT);
    }

    /// An upstream cut by a transport error delivers the held text before
    /// the typed error event that ends the body.
    #[tokio::test]
    async fn a_cut_writes_the_held_text_before_the_error_event() {
        let error = refused_call().await;
        let upstream = stream::iter(held_deltas().into_iter().map(Ok))
            .chain(stream::once(async move { Err(error) }));
        let frames = frames_from(upstream, re_sliced()).await;
        let content = content_of(&frames);
        let text: String = content.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, HELD_TEXT);
        let last = frames.len() - 1;
        let event = String::from_utf8_lossy(&frames[last]);
        assert!(
            event.contains("error"),
            "the body ends with the error event: {event}"
        );
        assert!(
            content.iter().all(|(index, _)| *index < last),
            "the held text goes out before the event"
        );
    }

    /// The idle flush waits for the policy's silence since the last chunk,
    /// not since the relay started: a second delta half a window in moves
    /// the flush by half a window.
    #[tokio::test(start_paused = true)]
    async fn the_idle_flush_waits_for_silence_since_the_last_chunk() {
        let idle_flush = re_slicing_policy().idle_flush;
        let (tx, rx) = mpsc::channel::<Result<Bytes, reqwest::Error>>(4);
        let mut body = relay_of(ReceiverStream::new(rx), re_sliced()).into_body();
        let started = tokio::time::Instant::now();
        tx.send(Ok(content_event(HELD[0]))).await.unwrap();
        tokio::time::sleep(idle_flush / 2).await;
        tx.send(Ok(content_event(HELD[1]))).await.unwrap();
        let frame = body
            .frame()
            .await
            .expect("a frame")
            .expect("a relayed frame")
            .into_data()
            .expect("data");
        let text: String = content_of(&[frame]).into_iter().map(|(_, t)| t).collect();
        assert_eq!(text, HELD_TEXT);
        assert_eq!(
            started.elapsed(),
            idle_flush / 2 + idle_flush,
            "the flush runs one window after the last chunk"
        );
        drop(tx);
    }

    /// An empty upstream chunk becomes no frame at all, on the plain path and
    /// through the decode side's hook alike; the bytes around it are relayed as
    /// they came.
    #[tokio::test]
    async fn an_empty_upstream_chunk_is_no_frame_toward_the_client() {
        let chunks: Vec<&'static [u8]> = vec![b"data: a\n\n", b"", b"data: [DONE]\n\n"];
        let plain = frames_through(chunks.clone(), ChunkMode::Plain).await;
        assert_eq!(
            plain,
            [
                Bytes::from_static(b"data: a\n\n"),
                Bytes::from_static(b"data: [DONE]\n\n")
            ]
        );

        let hook: Box<dyn FnMut(Bytes) -> ChunkStep + Send> = Box::new(|bytes: Bytes| ChunkStep {
            last: bytes.as_ref() == b"data: [DONE]\n\n",
            bytes,
        });
        let hooked = frames_through(chunks, ChunkMode::Hook(hook)).await;
        assert_eq!(hooked, plain);
    }

    /// The hook's `last` ends the relay: a chunk after it is not read.
    #[tokio::test]
    async fn the_hooks_last_chunk_ends_the_relay() {
        let hook: Box<dyn FnMut(Bytes) -> ChunkStep + Send> = Box::new(|bytes: Bytes| ChunkStep {
            last: bytes.as_ref() == b"data: [DONE]\n\n",
            bytes,
        });
        let frames = frames_through(
            vec![b"data: a\n\n", b"data: [DONE]\n\n", b"data: after\n\n"],
            ChunkMode::Hook(hook),
        )
        .await;
        assert_eq!(
            frames,
            [
                Bytes::from_static(b"data: a\n\n"),
                Bytes::from_static(b"data: [DONE]\n\n")
            ]
        );
    }

    /// A client that stops reading stops the upstream read: the relay holds
    /// at most `SSE_CHANNEL_BUFFER` chunks for it, plus the one its parked
    /// `send` carries, and never the rest of the stream. One frame the client
    /// reads frees one slot.
    #[tokio::test(start_paused = true)]
    async fn a_client_that_stops_reading_bounds_the_upstream_read_at_the_relays_buffer() {
        let pulled = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&pulled);
        let upstream = stream::repeat_with(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok::<_, reqwest::Error>(Bytes::from_static(b"data: x\n\n"))
        });
        let mut body = relay_of(upstream, ChunkMode::Plain).into_body();

        // The relay fills its channel and parks on the next send.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let unread = pulled.load(Ordering::SeqCst);
        assert!(
            (SSE_CHANNEL_BUFFER..=SSE_CHANNEL_BUFFER + 1).contains(&unread),
            "the relay pulled {unread} chunks for a client that read none"
        );

        // One frame read frees one slot: one more chunk, not the stream.
        body.frame()
            .await
            .expect("a frame")
            .expect("a relayed frame");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let after_one = pulled.load(Ordering::SeqCst);
        assert!(
            (unread + 1..=unread + 2).contains(&after_one),
            "the relay pulled {after_one} chunks after one frame was read, {unread} before"
        );
    }
}
