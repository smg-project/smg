use std::{sync::Arc, time::Duration};

use smg_worker::identity::basic::Worker;

use super::{vllm, FanoutStream, PrefillTicket, ProtoGenerateResponse, ProtoStream};
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

/// A timed stream's first response ends its prompt's prefill with the
/// cache hit the response reports, so the worker's rate learns the
/// tokens the engine computed and not the prompt's length. A prompt of
/// 100,000 tokens answered after a second with 90,000 of them cached
/// teaches 10,000 tokens a second at most; read at full length it would
/// teach some 95,000.
#[tokio::test]
async fn a_streams_first_response_teaches_the_rate_its_prompt_less_the_cache_hit() {
    const PROMPT: u64 = 100_000;
    const HIT: u32 = 90_000;
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://w1:30000").build());
    let state = worker.runtime_state().unwrap();
    let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk {
                cached_tokens: HIT,
                ..Default::default()
            },
        )),
    }));
    // The chunk reaches the stream's tracked layer as a response and not
    // as a replay: a child stream hands it over through a fan-out.
    let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let mut child = ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked(engine, 0, false);
    if let ProtoStream::Tracked(tracked) = &mut child {
        tracked.replay = Some(Some(Ok(chunk)));
    }
    let mut stream = ProtoStream::Fanout(FanoutStream::new(vec![child])).tracked(
        Arc::clone(&worker),
        PROMPT,
        true,
    );
    assert_eq!(state.prefill_rate_tps(), 0, "no rate before an answer");

    // The rate takes its first sample from a second of prefill.
    tokio::time::sleep(Duration::from_millis(1_050)).await;
    assert!(stream.next().await.is_some_and(|item| item.is_ok()));

    let computed = PROMPT - u64::from(HIT);
    let rate = state.prefill_rate_tps();
    assert!(
        (computed / 10..=computed).contains(&rate),
        "{computed} tokens computed in a second or more teach at most {computed} a second: {rate}"
    );
}

/// A prompt is ended once: answered at its first response, a ticket
/// notes nothing more at its drop. Two prompts are booked on one
/// worker; with the first answered and dropped, the second is still
/// pending at its whole length.
#[test]
fn a_prefill_ticket_answered_and_dropped_ends_its_prompt_once() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
    let state = worker.runtime_state().unwrap();
    let mut first = PrefillTicket::book_at(&worker, 8_192, 1).unwrap();
    let second = PrefillTicket::book_at(&worker, 4_096, 1).unwrap();
    assert_eq!(
        state.prefill_backlog(),
        Duration::from_millis(1_228),
        "both prompts are pending"
    );

    first.first_response_at(0, 2);
    drop(first);
    assert_eq!(
        state.prefill_backlog(),
        Duration::from_millis(409),
        "the second prompt's 4,096 tokens are all still pending"
    );

    drop(second);
    assert_eq!(state.prefill_backlog(), Duration::ZERO);
}
