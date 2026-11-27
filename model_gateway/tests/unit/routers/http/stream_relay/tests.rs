use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::Body,
    http::{header::CONTENT_LENGTH, HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{stream, Stream, StreamExt};
use http_body_util::BodyExt;
use smg_worker::{
    health::circuit_breaker::{CircuitBreakerConfig, CircuitState},
    identity::basic::{Worker, WorkerLoadGuard},
    test_support::RefusedAddress,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{relay_streaming_body, ChunkHook, ChunkMode, ChunkStep, Relay, RequestEnd};
use crate::{
    routers::{
        common::{sse::SSE_CHANNEL_BUFFER, sse_rechunk::SseRechunker},
        grpc::{backend_client::BasicWorkerBuilder, wire::Wire},
        http::stream_failure::RelaySource,
    },
    server::http_client_builder,
};

/// The relay of `upstream` in `mode`: a successful SSE head from a worker
/// at a port nobody listens on.
fn relay_of<S>(upstream: S, mode: ChunkMode) -> Response
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:1").build());
    relay_streaming_body(
        upstream,
        Relay {
            status: StatusCode::OK,
            upstream_sse: true,
            source: RelaySource::new("m", "/v1/chat/completions", Arc::clone(&worker)),
            guard: WorkerLoadGuard::with_key(worker, None),
            end: RequestEnd::default(),
            headers: HeaderMap::new(),
            mode,
        },
    )
}

/// The frames the client gets when the upstream yields `chunks` through
/// the relay in `mode`.
async fn frames_through(chunks: Vec<&'static [u8]>, mode: ChunkMode) -> Vec<Bytes> {
    let upstream = stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, reqwest::Error>(Bytes::from_static(chunk))),
    );
    frames_of(relay_of(upstream, mode).into_body()).await
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

    let hook = Box::new(|bytes: Bytes| ChunkStep {
        last: bytes.as_ref() == b"data: [DONE]\n\n",
        bytes,
    });
    let hooked = frames_through(chunks, ChunkMode::Hook(hook)).await;
    assert_eq!(hooked, plain);
}

/// The hook's `last` ends the relay: a chunk after it is not read.
#[tokio::test]
async fn the_hooks_last_chunk_ends_the_relay() {
    let hook = Box::new(|bytes: Bytes| ChunkStep {
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

/// A hook that holds every chunk back for the next one: what a frame
/// cut across chunks needs. Its `finish` is the held chunk.
struct HoldsOne(Option<Bytes>);

impl ChunkHook for HoldsOne {
    fn step(&mut self, chunk: Bytes) -> ChunkStep {
        ChunkStep {
            bytes: self.0.replace(chunk).unwrap_or_default(),
            last: false,
        }
    }

    fn finish(&mut self) -> Bytes {
        self.0.take().unwrap_or_default()
    }
}

/// What a hook still holds at a clean upstream end follows the last
/// chunk it let go.
#[tokio::test]
async fn the_hooks_held_tail_leaves_at_a_clean_upstream_end() {
    let frames = frames_through(
        vec![b"data: a\n\n", b"data: b\n\n"],
        ChunkMode::Hook(Box::new(HoldsOne(None))),
    )
    .await;
    assert_eq!(
        frames,
        [
            Bytes::from_static(b"data: a\n\n"),
            Bytes::from_static(b"data: b\n\n")
        ]
    );
}

/// What a hook still holds when the upstream is cut goes out before the
/// typed error event, as a re-chunker's held text does.
#[tokio::test]
async fn the_hooks_held_tail_leaves_before_a_cuts_error_event() {
    let error = upstream_error().await;
    let upstream = stream::iter(vec![
        Ok(Bytes::from_static(b"data: a\n\n")),
        Ok(Bytes::from_static(b"data: b\n\n")),
        Err(error),
    ]);
    let events = events_of(
        &frames_of(relay_of(upstream, ChunkMode::Hook(Box::new(HoldsOne(None)))).into_body()).await,
    );
    let [first, second, error_event, done] = events.as_slice() else {
        panic!("{events:?}");
    };
    assert_eq!((first.as_str(), second.as_str()), ("data: a", "data: b"));
    assert!(
        error_event.starts_with("data: {\"error\":"),
        "{error_event}"
    );
    assert_eq!(done, "data: [DONE]");
}

/// A client that stops reading stops the upstream read: the relay holds
/// at most `SSE_CHANNEL_BUFFER` chunks for it, plus the one its parked
/// `send` carries, and never the rest of the stream. One frame the client
/// reads frees one slot.
#[tokio::test(start_paused = true)]
async fn a_client_that_stops_reading_bounds_the_upstream_read_at_the_relays_buffer() {
    let pulled = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&pulled);
    // Capped, then parked: a relay with no bound at all pulls the cap and
    // fails the count below, instead of spinning on an upstream that is
    // always ready.
    let upstream = stream::repeat_with(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        Ok::<_, reqwest::Error>(Bytes::from_static(b"data: x\n\n"))
    })
    .take(4 * SSE_CHANNEL_BUFFER)
    .chain(stream::pending());
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

/// A re-chunker on the one policy in the tree (MiniMax's).
fn rechunking() -> ChunkMode {
    let policy = Wire::Minimax
        .rechunk_policy()
        .expect("MiniMax's streams are re-sliced");
    ChunkMode::Rechunk(Box::new(SseRechunker::new(policy)))
}

/// One chat completion chunk whose content delta is `text`.
fn delta(text: &str) -> Bytes {
    Bytes::from(format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    ))
}

/// Every data frame of `body`, to its end.
async fn frames_of(mut body: Body) -> Vec<Bytes> {
    let mut frames = Vec::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.expect("a relayed frame").into_data() {
            frames.push(data);
        }
    }
    frames
}

/// The events of `frames`, blank-line separated; a final fragment without
/// its terminator is a failure, since a client cannot dispatch it.
fn events_of(frames: &[Bytes]) -> Vec<String> {
    let text = String::from_utf8(frames.concat()).expect("UTF-8 SSE");
    let (events, rest) = text.rsplit_once("\n\n").unwrap_or(("", text.as_str()));
    assert!(rest.is_empty(), "an unterminated event: {rest:?}");
    events
        .split("\n\n")
        .filter(|event| !event.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The content deltas among `events`, in order.
fn contents(events: &[String]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| event.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|json| {
            json["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// A body error as reqwest reports one: the connect to a port nobody
/// listens on is refused. The client takes no proxy from the
/// environment, so the refusal comes from the loopback socket on every
/// host.
async fn upstream_error() -> reqwest::Error {
    let refused = RefusedAddress::reserve();
    http_client_builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(format!("http://{}/v1/chat/completions", refused.addr()))
        .send()
        .await
        .unwrap_err()
}

/// Text under the policy's emit threshold is held for a larger slice; an
/// upstream that closes its body cleanly, with no `[DONE]` and no closing
/// event, still gets it to the client, whole.
#[tokio::test]
async fn the_held_text_of_a_re_sliced_stream_leaves_at_a_clean_upstream_end() {
    let upstream =
        stream::iter(["Hel", "lo, ", "world"].map(|text| Ok::<_, reqwest::Error>(delta(text))));
    let events = events_of(&frames_of(relay_of(upstream, rechunking()).into_body()).await);
    assert_eq!(contents(&events).concat(), "Hello, world", "{events:?}");
}

/// The text held when the upstream is cut goes out before the typed
/// error event, which ends the stream with `[DONE]` as on a plain relay.
#[tokio::test]
async fn the_held_text_of_a_re_sliced_stream_leaves_before_a_cuts_error_event() {
    let error = upstream_error().await;
    let upstream = stream::iter(vec![Ok(delta("Hel")), Ok(delta("lo")), Err(error)]);
    let events = events_of(&frames_of(relay_of(upstream, rechunking()).into_body()).await);
    assert_eq!(contents(&events), ["Hello"], "{events:?}");
    let [.., error_event, done] = events.as_slice() else {
        panic!("{events:?}");
    };
    let error: serde_json::Value =
        serde_json::from_str(error_event.strip_prefix("data: ").unwrap())
            .expect("the error event is JSON");
    assert_eq!(error["error"]["type"], "stream_error", "{error_event}");
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.starts_with("Stream error: ")),
        "{error_event}"
    );
    assert_eq!(done, "data: [DONE]");
}

/// A worker with `breaker` as its circuit breaker's settings.
fn worker_with(breaker: CircuitBreakerConfig) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:1")
            .circuit_breaker_config(breaker)
            .build(),
    )
}

/// A worker whose closed breaker opens at its second failure in a row.
fn worker_opening_at_two() -> Arc<dyn Worker> {
    worker_with(CircuitBreakerConfig {
        failure_threshold: 2,
        ..Default::default()
    })
}

/// A stream of `worker` that runs until the returned sender is dropped:
/// its good head is out, its body is still open.
fn a_running_stream(
    worker: &Arc<dyn Worker>,
) -> (mpsc::Sender<Result<Bytes, reqwest::Error>>, Response) {
    let (worker_side, upstream) = mpsc::channel(1);
    let response = relay_from(worker, ReceiverStream::new(upstream));
    (worker_side, response)
}

/// The plain relay of `upstream` from `worker`: a successful SSE head.
fn relay_from<S>(worker: &Arc<dyn Worker>, upstream: S) -> Response
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    relay_streaming_body(
        upstream,
        Relay {
            status: StatusCode::OK,
            upstream_sse: true,
            source: RelaySource::new("m", "/v1/chat/completions", Arc::clone(worker)),
            guard: WorkerLoadGuard::with_key(Arc::clone(worker), None),
            end: RequestEnd::default(),
            headers: HeaderMap::new(),
            mode: ChunkMode::Plain,
        },
    )
}

/// One stream `worker` cuts behind a good head, read to its end: the
/// client reads a 200 that ends in the error event and the done mark.
async fn a_cut_stream(worker: &Arc<dyn Worker>) {
    let error = upstream_error().await;
    let upstream = stream::iter(vec![Ok(delta("Hel")), Err(error)]);
    let events = events_of(&frames_of(relay_from(worker, upstream).into_body()).await);
    assert_eq!(
        events.last().map(String::as_str),
        Some("data: [DONE]"),
        "{events:?}"
    );
}

/// A worker that cuts every stream behind a good head fails every
/// request it takes: its breaker opens at the failure threshold, as it
/// does where the same cuts end bodies that are not streamed.
#[tokio::test]
async fn a_worker_that_cuts_every_stream_behind_a_good_head_opens_its_breaker() {
    let worker = worker_opening_at_two();
    a_cut_stream(&worker).await;
    a_cut_stream(&worker).await;
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
}

/// A stream the worker ends cleanly is the closed breaker's one success,
/// taken at the stream's end: between two cut streams it starts the
/// failure streak anew, and the breaker stays closed.
#[tokio::test]
async fn a_clean_stream_between_two_cut_ones_starts_the_failure_streak_anew() {
    let worker = worker_opening_at_two();
    a_cut_stream(&worker).await;
    let upstream = stream::iter([Ok::<_, reqwest::Error>(delta("Hello"))]);
    let events = events_of(&frames_of(relay_from(&worker, upstream).into_body()).await);
    assert_eq!(contents(&events), ["Hello"], "{events:?}");
    a_cut_stream(&worker).await;
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// A stream its client leaves is no cut of the worker's: the closed
/// breaker takes its one success when the relay ends, as it does at a
/// clean end.
#[tokio::test]
async fn a_stream_its_client_leaves_between_two_cut_ones_starts_the_failure_streak_anew() {
    let worker = worker_opening_at_two();
    a_cut_stream(&worker).await;
    let (worker_side, response) = a_running_stream(&worker);
    drop(response);
    // The relay lets go of the upstream once its sample is in.
    worker_side.closed().await;
    a_cut_stream(&worker).await;
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// A good head that finds an open breaker past its timeout, before any
/// selection read it, starts the trial itself: the relay reads the state
/// with the worker's own read, which moves such a breaker to half-open
/// as a selection's read does, and the head is the trial's success.
#[tokio::test]
async fn a_good_head_on_an_open_breaker_past_its_timeout_starts_its_trial_with_a_success() {
    let worker = worker_with(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_duration: Duration::ZERO,
        ..Default::default()
    });
    // Open and past its timeout of zero; nothing has read the state since.
    worker.record_circuit_breaker_outcome(false);
    let upstream = stream::iter([Ok::<_, reqwest::Error>(delta("Hello"))]);
    let response = relay_from(&worker, upstream);
    // The head is out and the relay task has not run: one success closed the trial.
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    let events = events_of(&frames_of(response.into_body()).await);
    assert_eq!(contents(&events), ["Hello"], "{events:?}");
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// The same head where the trial needs two successes: the breaker is
/// half-open behind it with one of the two counted, and a second good
/// head closes it.
#[tokio::test]
async fn a_good_head_on_an_open_breaker_past_its_timeout_counts_toward_the_trials_close() {
    let worker = worker_with(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 2,
        timeout_duration: Duration::ZERO,
        ..Default::default()
    });
    worker.record_circuit_breaker_outcome(false);
    let _first = a_running_stream(&worker);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    let _second = a_running_stream(&worker);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// The good head of a stream that still runs is no sample of a closed
/// breaker: a failed request before it and one behind it are two
/// failures in a row, and the breaker opens.
#[tokio::test]
async fn a_running_streams_head_leaves_a_failure_streak_unbroken() {
    let worker = worker_opening_at_two();
    worker.record_outcome(503);
    let _running = a_running_stream(&worker);
    worker.record_outcome(503);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
}

/// The closed breaker's success of a stream lands at the stream's end,
/// not at its head: a failed request while the stream runs and one
/// behind its clean end are not two failures in a row.
#[tokio::test]
async fn a_streams_success_lands_at_its_end_between_two_failed_requests() {
    let worker = worker_opening_at_two();
    let (worker_side, response) = a_running_stream(&worker);
    worker.record_outcome(503);
    drop(worker_side);
    let _frames = frames_of(response.into_body()).await;
    worker.record_outcome(503);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// A breaker that tripped while a stream ran takes no success from that
/// stream's end: half-open, it still waits for its trial.
#[tokio::test]
async fn a_streams_end_on_a_breaker_that_tripped_meanwhile_is_no_trial_success() {
    let worker = worker_with(CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration: Duration::ZERO,
        ..Default::default()
    });
    let (worker_side, response) = a_running_stream(&worker);
    worker.record_outcome(503);
    worker.record_outcome(503);
    // Tripped, and past its timeout of zero: this read starts the trial.
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    drop(worker_side);
    let _frames = frames_of(response.into_body()).await;
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
}

/// The idle flush waits for the policy's silence since the last upstream
/// chunk, not since the relay started: a chunk late in the window starts
/// a new one.
#[tokio::test(start_paused = true)]
async fn the_idle_flush_of_a_re_sliced_stream_waits_for_silence_since_the_last_chunk() {
    let idle_flush = Wire::Minimax
        .rechunk_policy()
        .expect("MiniMax's streams are re-sliced")
        .idle_flush;
    let (tx, rx) = mpsc::channel::<Result<Bytes, reqwest::Error>>(4);
    let mut body = relay_of(ReceiverStream::new(rx), rechunking()).into_body();
    tx.send(Ok(delta("abc"))).await.unwrap();
    tokio::time::sleep(idle_flush * 4 / 5).await;
    tx.send(Ok(delta("def"))).await.unwrap();

    // Six characters are pending now; the window counts from this chunk.
    let early = tokio::time::timeout(idle_flush - Duration::from_millis(10), body.frame()).await;
    assert!(
        early.is_err(),
        "a flush before the window since the last chunk had passed: {early:?}"
    );
    let frame = tokio::time::timeout(Duration::from_millis(20), body.frame())
        .await
        .expect("the idle flush")
        .expect("a frame")
        .expect("a relayed frame");
    let events = events_of(&[frame.into_data().expect("a data frame")]);
    assert_eq!(contents(&events), ["abcdef"], "{events:?}");
    drop(tx);
}

/// The relay owns the header rule for every caller: a sized SSE body
/// loses the upstream's `Content-Length` here, and a body that is not
/// SSE keeps the length its caller prepared.
#[tokio::test]
async fn a_sized_sse_bodys_content_length_leaves_with_the_relay_and_another_bodys_stays() {
    for (upstream_sse, kept) in [(true, false), (false, true)] {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:1").build());
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_LENGTH, "9".parse().unwrap());
        let upstream = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: x\n\n",
        ))]);
        let response = relay_streaming_body(
            upstream,
            Relay {
                status: StatusCode::OK,
                upstream_sse,
                source: RelaySource::new("m", "/v1/chat/completions", Arc::clone(&worker)),
                guard: WorkerLoadGuard::with_key(worker, None),
                end: RequestEnd::default(),
                headers,
                mode: ChunkMode::Plain,
            },
        );
        assert_eq!(
            response.headers().contains_key(CONTENT_LENGTH),
            kept,
            "upstream_sse {upstream_sse}: the upstream's Content-Length"
        );
    }
}
