use std::sync::Arc;

use smg_worker::{identity::basic::Worker, runtime::StallReason};

use super::{
    sglang, tokenspeed, vllm, DispatchBooking, FanoutStream, FirstItem, PrefillTicket,
    ProtoGenerateResponse, ProtoStream,
};
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

#[test]
fn the_cache_hit_a_first_response_reports_is_read_from_a_chunk_or_a_complete() {
    // The prefill ticket hands the worker the hit with the first
    // response, whichever shape it has; a response that carries no
    // count reads as no hit, and the prompt teaches its full length.
    let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk {
                cached_tokens: 1_900,
                ..Default::default()
            },
        )),
    }));
    assert_eq!(chunk.cached_tokens(), 1_900);
    let complete = ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
        response: Some(sglang::generate_response::Response::Complete(
            sglang::GenerateComplete {
                cached_tokens: 64,
                ..Default::default()
            },
        )),
        ..Default::default()
    }));
    assert_eq!(complete.cached_tokens(), 64);
    let empty = ProtoGenerateResponse::Vllm(Box::default());
    assert_eq!(empty.cached_tokens(), 0);
}

#[test]
fn a_tracked_stream_is_one_of_its_workers_pile_for_its_life() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
    let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new())).tracked(
        Arc::clone(&worker),
        0,
        true,
    );
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        1,
        "counted at dispatch"
    );
    let stream = stream.defer_abort_until_first_item();
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        1,
        "and across the stream's rewraps"
    );
    drop(stream);
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        0,
        "released with the stream"
    );
}

#[tokio::test]
async fn a_tracked_stream_inside_a_tracked_stream_ends_and_releases_both_piles() {
    // The in-place poll of `next` covers one tracked layer over a
    // transport stream; a second layer takes the recursive path. Both
    // layers' bookkeeping holds either way.
    let outer: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
    let inner: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w2:8000").build());
    let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked(Arc::clone(&inner), 0, true)
        .tracked(Arc::clone(&outer), 0, true);
    assert_eq!(
        (
            inner.runtime_state().unwrap().tracked_load(),
            outer.runtime_state().unwrap().tracked_load()
        ),
        (1, 1)
    );
    assert!(
        stream.next().await.is_none(),
        "an empty fanout ends at once"
    );
    drop(stream);
    assert_eq!(
        (
            inner.runtime_state().unwrap().tracked_load(),
            outer.runtime_state().unwrap().tracked_load()
        ),
        (0, 0)
    );
}

/// A stream over one chunk, read through a fan-out so the chunk reaches
/// this stream's tracked layer as a response and not as a replay.
fn one_chunk_tracked(worker: &Arc<dyn Worker>) -> ProtoStream {
    let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk::default(),
        )),
    }));
    let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let mut child =
        ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked(Arc::clone(&engine), 0, false);
    if let ProtoStream::Tracked(tracked) = &mut child {
        tracked.replay = Some(Some(Ok(chunk)));
    }
    ProtoStream::Fanout(FanoutStream::new(vec![child])).tracked(Arc::clone(worker), 0, true)
}

/// Review finding on the second head: the veto clear on the captured
/// runtime path had no case of its own (the veto pins drive the worker
/// path), so a stream's response clears a veto here.
#[tokio::test]
async fn a_tracked_streams_response_clears_the_workers_veto() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://w1:30000").build());
    let mut stream = one_chunk_tracked(&worker);
    assert!(worker.set_stall(Some(StallReason::Wedged)));
    assert!(stream.next().await.is_some_and(|item| item.is_ok()));
    assert!(
        worker.stall_reason().is_none(),
        "the response clears the veto"
    );
}

/// The output ticket lands on the state its stream captured at its start,
/// as the stream's other notes do, and not on what the door answers at
/// the first response; its end gives the output back there and leaves
/// none open on either state. The case swaps the worker's state under a
/// live stream, which the door's doc forbids, to pin where the output
/// goes.
#[tokio::test]
async fn a_streams_output_is_noted_on_the_state_the_stream_captured() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://w1:30000").build());
    let other: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://w1:30000").build());
    let captured = worker.runtime_state().unwrap();
    let mut stream = one_chunk_tracked(&worker);

    assert!(worker.inherit_shared_state_from(other.as_ref()));
    let swapped_in = worker.runtime_state().unwrap();
    assert!(
        !Arc::ptr_eq(&captured, &swapped_in),
        "the state was swapped"
    );

    assert!(stream.next().await.is_some_and(|item| item.is_ok()));
    assert!(
        captured.output_silence().is_some(),
        "the output is open on the state the stream captured"
    );
    assert!(
        swapped_in.output_silence().is_none(),
        "and not on the state swapped in"
    );

    drop(stream);
    assert!(
        captured.output_silence().is_none() && swapped_in.output_silence().is_none(),
        "given back where it was opened: none is left open on either state"
    );
}

/// Review finding on the first head of this change: a millisecond stamp
/// kept per worker handle in front of the runtime load left a failure
/// noted through another handle of the same runtime (a clone, or the
/// handle a monitor holds) pending until the next millisecond. The
/// stream notes the shared runtime itself, so the runtime's own rule
/// decides, whichever handle noted the failure.
#[tokio::test]
async fn a_tracked_streams_response_clears_a_failure_noted_through_another_handle() {
    let basic = BasicWorkerBuilder::new("grpc://w1:30000").build();
    let other: Arc<dyn Worker> = Arc::new(basic.clone());
    let worker: Arc<dyn Worker> = Arc::new(basic);
    let mut stream = one_chunk_tracked(&worker);
    // A token through this handle, then a failure through the other,
    // within one millisecond as far as the clock allows.
    worker.runtime_state().unwrap().note_token_progress();
    other.runtime_state().unwrap().note_transport_failure();
    assert!(
        worker.runtime_state().unwrap().transport_failure_pending(),
        "the handles share the runtime"
    );
    assert!(stream.next().await.is_some_and(|item| item.is_ok()));
    assert!(
        !worker.runtime_state().unwrap().transport_failure_pending(),
        "the response clears it"
    );
    assert!(!other.runtime_state().unwrap().transport_failure_pending());
}

#[test]
fn a_non_streaming_generation_joins_no_pile_but_keeps_its_prefill_and_its_progress() {
    // Review finding on the pushed head: left untracked altogether, a
    // non-streaming generation's one answer no longer counted as progress
    // and its prompt left the prefill backlog, so a saturated engine
    // finishing such requests while streaming ones waited was wedged. It
    // is tracked like any other, minus the pile slot.
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
    let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new())).tracked(
        Arc::clone(&worker),
        64,
        false,
    );
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        0,
        "nothing to see progress on until it answers"
    );
    match &stream {
        ProtoStream::Tracked(tracked) => {
            assert!(tracked.progress.is_none(), "no pile slot");
            assert!(
                tracked.prefill.as_ref().is_some_and(|ticket| !ticket.timed),
                "pending prefill like any other, untimed: its first response is its whole answer"
            );
            // Its one answer goes through the same `Tracked` arm of
            // `next` as a token, which is where progress is recorded.
        }
        _ => panic!("a tracked stream"),
    }
    let stream = stream.defer_abort_until_first_item();
    assert_eq!(worker.runtime_state().unwrap().tracked_load(), 0);
    drop(stream);
    assert_eq!(worker.runtime_state().unwrap().tracked_load(), 0);
}

/// A disaggregated decode side earns its worker's breaker success at its
/// first response, never at its dispatch: a side whose stream ends with
/// nothing on it leaves a failure before it standing, so the next one
/// opens a breaker set to two.
#[tokio::test]
async fn a_decode_side_that_ends_with_nothing_on_it_earns_no_breaker_success() {
    use smg_worker::health::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:30000")
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            })
            .build(),
    );
    worker.record_outcome(504);
    let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked_decode_side(Arc::clone(&worker));
    match &stream {
        ProtoStream::Tracked(tracked) => {
            assert!(tracked.samples_success, "the success waits for a response");
            assert!(tracked.progress.is_none(), "no pile slot");
            assert!(tracked.prefill.is_none(), "no prompt booked");
        }
        _ => panic!("a tracked stream"),
    }
    assert!(
        stream.next().await.is_none(),
        "an empty fanout ends at once"
    );
    drop(stream);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    worker.record_outcome(504);
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Open,
        "two failures with no success earned between them"
    );
}

/// A single dispatch earns its worker's breaker success at its stream's
/// first response, as a decode side does: between two failures it keeps
/// a breaker set to two closed.
#[tokio::test]
async fn a_single_dispatch_earns_its_breaker_success_at_its_first_response() {
    use smg_worker::health::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    // A stream of one response: a tracked layer of its own replays it.
    let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let mut answering =
        ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked(engine, 0, false);
    if let ProtoStream::Tracked(tracked) = &mut answering {
        tracked.replay = Some(Some(Ok(ProtoGenerateResponse::Vllm(Box::default()))));
    }
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://w1:30000")
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            })
            .build(),
    );
    let booking = DispatchBooking::book(&worker, 16, true);
    let mut stream = answering.tracked_single_booked(Arc::clone(&worker), booking);
    worker.record_outcome(504);
    assert!(stream.next().await.is_some_and(|item| item.is_ok()));
    worker.record_outcome(504);
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Closed,
        "a success, at the first response, between the two failures"
    );
}

#[test]
fn a_prefill_side_booked_at_dispatch_is_pending_before_its_stream_exists() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    assert!(
        PrefillTicket::book(&worker, 0).is_none(),
        "an empty prompt books nothing"
    );
    let ticket = PrefillTicket::book(&worker, 4_096);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::from_millis(409),
        "pending from the dispatch, before the call has answered"
    );
    let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked_prefill_side_booked(Arc::clone(&worker), ticket);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::from_millis(409),
        "handed to the stream: booked once, not again"
    );
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        0,
        "a side is never part of a pile"
    );
    match &stream {
        ProtoStream::Tracked(tracked) => {
            assert!(tracked.prefill.as_ref().is_some_and(|ticket| ticket.timed));
        }
        _ => panic!("a tracked stream"),
    }
    drop(stream);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::ZERO,
        "released with the stream"
    );
}

#[test]
fn a_side_booked_at_dispatch_whose_call_fails_releases_its_prompt() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    let ticket = PrefillTicket::book(&worker, 1_000);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::from_millis(100)
    );
    drop(ticket);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::ZERO,
        "a call that fails or is retired gives the prompt back untimed"
    );
}

#[test]
fn a_prefill_side_books_its_prompt_timed_and_joins_no_pile() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked_prefill_side_booked(Arc::clone(&worker), PrefillTicket::book(&worker, 4_096));
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        0,
        "a side is never part of a pile"
    );
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::from_millis(409),
        "its prompt is pending prefill on the prefill worker (cold prior)"
    );
    match &stream {
        ProtoStream::Tracked(tracked) => {
            assert!(tracked.progress.is_none());
            assert!(tracked.prefill.as_ref().is_some_and(|ticket| ticket.timed));
        }
        _ => panic!("a tracked stream"),
    }
    drop(stream);
    assert_eq!(
        worker.runtime_state().unwrap().prefill_backlog(),
        std::time::Duration::ZERO,
        "released with the stream"
    );
}

/// 0 is the state's unset stamp, so a ticket notes its start and its
/// first response no lower than one (the worker's own notes held that
/// floor before the tickets wrote the state): a prompt booked and
/// answered at stamp 0 reads as answered, and as having taken no time.
#[test]
fn a_prefill_ticket_notes_its_start_and_its_answer_no_lower_than_stamp_one() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    let state = worker.runtime_state().unwrap();
    let mut ticket = PrefillTicket::book_at(&worker, 2_001, 0).unwrap();
    ticket.first_response_at(0, 0);
    assert_eq!(
        state.prefill_answered_ago_at(1),
        Some(std::time::Duration::ZERO),
        "answered at one: an answer at 0 reads as none"
    );
    // Booked at one too, so the prompt put no time in the rate's window:
    // the next one's 2,000 ms are the whole span its sample is taken over.
    let mut next = PrefillTicket::book_at(&worker, 2_001, 1).unwrap();
    next.first_response_at(0, 2_001);
    assert_eq!(
        state.prefill_rate_tps(),
        2_001,
        "4,002 tokens over 2,000 ms: a start at 0 makes the span 2,001 ms"
    );
}

#[tokio::test]
async fn the_first_response_pulled_ahead_is_replayed_to_the_reader_once() {
    // An empty fan-out ends at once: the end is what is pulled ahead, and
    // the reader sees it once more, then the stream as it is.
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked_prefill_side_booked(Arc::clone(&worker), PrefillTicket::book(&worker, 16));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    assert_eq!(
        stream.first_response_within(deadline).await.ok(),
        Some(FirstItem::End),
        "what was pulled ahead is named: the end, not a response"
    );
    assert_eq!(
        stream.first_response_within(deadline).await.ok(),
        Some(FirstItem::End),
        "a second wait does not pull past the held item"
    );
    assert!(stream.next().await.is_none(), "the held end is replayed");
    assert!(stream.next().await.is_none(), "and the stream stays ended");
}

/// The parallel decode side's first item, pulled ahead while its prefill
/// side is awaited, is held unbooked and replayed once to its reader,
/// then the stream as it was; whatever tracks the stream then books it
/// at the replay: the decode worker's output, and its success.
#[tokio::test]
async fn a_first_item_pulled_ahead_is_replayed_once_and_booked_by_its_reader() {
    use smg_worker::health::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    type Item = Result<ProtoGenerateResponse, tonic::Status>;
    // Chunks told apart by the count each carries.
    let chunk = |cached_tokens| -> Item {
        let chunk = vllm::GenerateStreamChunk {
            cached_tokens,
            ..Default::default()
        };
        let response = Some(vllm::generate_response::Response::Chunk(chunk));
        Ok(ProtoGenerateResponse::Vllm(Box::new(
            vllm::GenerateResponse { response },
        )))
    };
    let which = |item: Option<&Item>| Some(item?.as_ref().ok()?.cached_tokens());
    // A child that yields chunks 1 and 2 (each replayed by a tracked
    // layer of its own, on an engine of its own) and ends, read through
    // a fan-out: its first item is pulled from it.
    let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let empty = ProtoStream::Fanout(FanoutStream::new(Vec::new()));
    let child = [2, 1].into_iter().fold(empty, |inner, id| {
        let mut layer = inner.tracked(Arc::clone(&engine), 0, false);
        if let ProtoStream::Tracked(tracked) = &mut layer {
            tracked.replay = Some(Some(chunk(id)));
        }
        layer
    });
    let mut stream = ProtoStream::Fanout(FanoutStream::new(vec![child]));
    assert_eq!(which(stream.first_item_ahead().await), Some(1), "pulled");
    assert_eq!(
        which(stream.first_item_ahead().await),
        Some(1),
        "held, not read again"
    );
    assert!(stream.worker().is_none());
    let stream = stream.defer_abort_until_first_item();
    assert!(stream.holds_item_ahead(), "rewrapped with what it holds");
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:30000")
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            })
            .build(),
    );
    worker.record_outcome(504);
    let mut stream = stream.tracked_decode_side(Arc::clone(&worker));
    assert!(worker.output_silence().is_none(), "nothing booked unread");
    assert_eq!(which(stream.next().await.as_ref()), Some(1), "replayed");
    assert!(worker.output_silence().is_some(), "booked at the replay");
    assert_eq!(which(stream.next().await.as_ref()), Some(2), "then on");
    assert!(stream.next().await.is_none(), "to the stream's end");
    worker.record_outcome(504);
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Closed,
        "a success, booked at the replay, between the two failures"
    );
}

/// A decode side restamps every response the stream it wraps hands on,
/// a held first item as the ones read after it, and once: a wrapper
/// inside a wrapper answers with the outer side's figure.
#[tokio::test]
async fn a_decode_side_restamps_every_response_of_the_stream_it_wraps() {
    type Item = Result<ProtoGenerateResponse, tonic::Status>;
    let chunk = |cached_tokens| -> Item {
        let chunk = vllm::GenerateStreamChunk {
            cached_tokens,
            ..Default::default()
        };
        let response = Some(vllm::generate_response::Response::Chunk(chunk));
        Ok(ProtoGenerateResponse::Vllm(Box::new(
            vllm::GenerateResponse { response },
        )))
    };
    let which = |item: Option<&Item>| Some(item?.as_ref().ok()?.cached_tokens());
    let prefill = |cached_tokens| {
        let mut prefill = super::PrefillCachedTokens::default();
        prefill.note(&super::ProtoGenerateComplete::Vllm(Box::new(
            vllm::GenerateComplete {
                cached_tokens,
                ..Default::default()
            },
        )));
        prefill
    };
    // Two tracked layers, each holding one chunk for its reader: the
    // stream yields the chunks 1 and 2 and ends.
    let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let two_chunks = || {
        let empty = ProtoStream::Fanout(FanoutStream::new(Vec::new()));
        [2, 1].into_iter().fold(empty, |inner, id| {
            let mut layer = inner.tracked(Arc::clone(&engine), 0, false);
            if let ProtoStream::Tracked(tracked) = &mut layer {
                tracked.replay = Some(Some(chunk(id)));
            }
            layer
        })
    };
    let mut stream = two_chunks().decode_side(prefill(7));
    assert!(matches!(stream, ProtoStream::DecodeSide(_)), "wrapped");
    assert_eq!(which(stream.next().await.as_ref()), Some(7), "the held");
    assert_eq!(which(stream.next().await.as_ref()), Some(7), "the next");
    assert!(stream.next().await.is_none(), "to the stream's end");
    let mut stream = two_chunks().decode_side(prefill(7)).decode_side(prefill(9));
    assert_eq!(which(stream.next().await.as_ref()), Some(9), "the outer");
    assert_eq!(which(stream.next().await.as_ref()), Some(9), "the outer");
    assert!(stream.next().await.is_none(), "to the stream's end");
}

/// A decode side's first item is its prefill worker's handoff only when
/// it is output: a chunk, or a complete that ended neither in an abort
/// (an aborted choice, as the Rust servicers end one) nor in an error.
#[test]
fn output_is_a_chunk_or_a_complete_the_engine_neither_aborted_nor_failed() {
    use sglang::generate_response::Response as Sg;
    use tokenspeed::generate_response::Response as Ts;
    let ts = |response: Ts| {
        let response = Some(response);
        let response = tokenspeed::GenerateResponse {
            response,
            ..Default::default()
        };
        ProtoGenerateResponse::TokenSpeed(Box::new(response)).is_output()
    };
    let sg = |response: Sg| {
        let response = Some(response);
        let response = sglang::GenerateResponse {
            response,
            ..Default::default()
        };
        ProtoGenerateResponse::Sglang(Box::new(response)).is_output()
    };
    assert!(ts(Ts::Chunk(Default::default())) && sg(Sg::Chunk(Default::default())));
    for (reasons, output) in [(["stop", "length"], true), (["abort", "error"], false)] {
        for reason in reasons {
            let finish_reason = reason.to_string();
            let complete = sglang::GenerateComplete {
                finish_reason,
                ..Default::default()
            };
            assert_eq!(sg(Sg::Complete(complete)), output, "{reason}");
            let finish_reason = reason.to_string();
            let complete = tokenspeed::GenerateComplete {
                finish_reason,
                ..Default::default()
            };
            assert_eq!(ts(Ts::Complete(complete)), output, "{reason}");
        }
    }
    let empty = ProtoGenerateResponse::TokenSpeed(Box::default());
    assert!(!empty.is_output(), "an empty response");
}

#[test]
fn a_dispatch_booking_holds_the_prompt_and_the_pile_until_its_stream_takes_over() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
    let books = |worker: &Arc<dyn Worker>| {
        (
            worker.runtime_state().unwrap().prefill_backlog(),
            worker.runtime_state().unwrap().tracked_load(),
        )
    };
    // 4,096 prompt tokens at the cold prior read as 409 ms of backlog.
    let pending = std::time::Duration::from_millis(409);
    let none = (std::time::Duration::ZERO, 0);
    // A streaming generation: its prompt timed, its place in the pile.
    let booking = DispatchBooking::book(&worker, 4_096, true);
    assert_eq!(
        books(&worker),
        (pending, 1),
        "booked at the dispatch, before the call has answered"
    );
    assert!(booking.prefill.as_ref().is_some_and(|ticket| ticket.timed));
    let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
        .tracked_booked(Arc::clone(&worker), booking);
    assert_eq!(
        books(&worker),
        (pending, 1),
        "handed to the stream: booked once, not again"
    );
    drop(stream);
    assert_eq!(books(&worker), none, "released with the stream");
    // A non-streaming generation: its prompt pending untimed, no pile slot.
    let booking = DispatchBooking::book(&worker, 4_096, false);
    assert_eq!(books(&worker), (pending, 0));
    assert!(booking.prefill.as_ref().is_some_and(|ticket| !ticket.timed));
    assert!(booking.progress.is_none());
    // Dropped before a stream took it over (the call failed): given back.
    drop(booking);
    assert_eq!(books(&worker), none);
}
