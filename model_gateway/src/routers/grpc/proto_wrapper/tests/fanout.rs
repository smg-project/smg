//! The fan-out stream: children polled from a rotating cursor and retired as they end, prefill
//! drained through it, and the test-only child impl that lets `drain_prefill` take a bare fan-out.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use super::super::*;
use crate::{
    routers::grpc::common::response_collection::drain_prefill,
    worker::{
        BasicWorkerBuilder, PrefillAdmission, PrefillLoadGuard, Worker, WorkerLoadGuard, WorkerType,
    },
};

impl<C: FanoutChild> FanoutChild for FanoutStream<C> {
    fn next_item(
        &mut self,
    ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send {
        self.next()
    }

    fn mark_completed(&mut self) {
        Self::mark_completed(self);
    }

    fn defer_abort_until_first_item(self) -> Self {
        Self::defer_abort_until_first_item(self)
    }
}

/// A child that yields a script, then ends.
struct Scripted {
    items: VecDeque<Result<ProtoGenerateResponse, tonic::Status>>,
    completed: Arc<AtomicBool>,
}

impl FanoutChild for Scripted {
    fn next_item(
        &mut self,
    ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send {
        let item = self.items.pop_front();
        async move { item }
    }

    fn mark_completed(&mut self) {
        self.completed.store(true, Ordering::SeqCst);
    }

    fn defer_abort_until_first_item(self) -> Self {
        self
    }
}

fn chunk(text: &str) -> ProtoGenerateResponse {
    ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
        response: Some(sglang::generate_response::Response::Chunk(
            sglang::GenerateStreamChunk {
                token_ids: vec![text.len() as u32],
                ..Default::default()
            },
        )),
        ..Default::default()
    }))
}

fn complete() -> ProtoGenerateResponse {
    ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
        response: Some(sglang::generate_response::Response::Complete(
            sglang::GenerateComplete::default(),
        )),
        ..Default::default()
    }))
}

fn child(items: Vec<ProtoGenerateResponse>) -> (Scripted, Arc<AtomicBool>) {
    let completed = Arc::new(AtomicBool::new(false));
    let child = Scripted {
        items: items.into_iter().map(Ok).collect(),
        completed: Arc::clone(&completed),
    };
    (child, completed)
}

fn index_of(response: ProtoGenerateResponse) -> u32 {
    match response.into_response() {
        ProtoResponseVariant::Chunk(chunk) => chunk.index(),
        ProtoResponseVariant::Complete(complete) => complete.index(),
        ProtoResponseVariant::None => u32::MAX,
    }
}

#[tokio::test]
async fn fanout_stamps_each_childs_position_and_ends_with_the_last_child() {
    let (first, first_done) = child(vec![chunk("a"), chunk("aa"), complete()]);
    let (second, second_done) = child(vec![chunk("b"), complete()]);
    let mut fanout = FanoutStream::new(vec![first, second]);

    let mut indices = Vec::new();
    while let Some(item) = fanout.next().await {
        indices.push(index_of(item.unwrap()));
    }
    assert_eq!(indices.len(), 5);
    assert_eq!(indices.iter().filter(|&&i| i == 0).count(), 3);
    assert_eq!(indices.iter().filter(|&&i| i == 1).count(), 2);
    // The rotating cursor lets the second child in before the first drains.
    assert_ne!(indices[0], indices[1]);
    assert_eq!(fanout.open_children(), 0);
    assert!(fanout.next().await.is_none());

    assert!(!first_done.load(Ordering::SeqCst));
    fanout.mark_completed();
    // Ended children were retired; only open children get marked.
    assert!(!first_done.load(Ordering::SeqCst));
    assert!(!second_done.load(Ordering::SeqCst));
}

#[tokio::test]
async fn fanout_mark_completed_reaches_every_open_child() {
    let (first, first_done) = child(vec![chunk("a")]);
    let (second, second_done) = child(vec![chunk("b")]);
    let mut fanout = FanoutStream::new(vec![first, second]);
    assert_eq!(fanout.open_children(), 2);
    fanout.mark_completed();
    assert!(first_done.load(Ordering::SeqCst));
    assert!(second_done.load(Ordering::SeqCst));
}

#[tokio::test]
async fn fanout_passes_a_child_error_through() {
    let (first, _) = child(vec![chunk("a")]);
    let second = Scripted {
        items: VecDeque::from([Err(tonic::Status::internal("side died"))]),
        completed: Arc::new(AtomicBool::new(false)),
    };
    let mut fanout = FanoutStream::new(vec![second, first]);
    let first_item = fanout.next().await.unwrap();
    assert!(matches!(first_item, Err(status) if status.message() == "side died"));
}

#[tokio::test]
async fn prefill_fanout_retains_unfinished_sample_guards() {
    for finish_at_complete in [false, true] {
        for bounded in [false, true] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://prefill-fanout")
                    .worker_type(WorkerType::Prefill)
                    .build(),
            );
            let admission = PrefillAdmission::new(1, 0, std::time::Duration::from_secs(1));
            let guard = if bounded {
                let admitted = admission
                    .admit(None, |capacity| capacity.select(Arc::clone(&worker), ()))
                    .await
                    .unwrap();
                PrefillLoadGuard::Admission {
                    _reservation: Arc::new(admitted.reservation),
                }
            } else {
                PrefillLoadGuard::Unbounded {
                    _guard: WorkerLoadGuard::new(Arc::clone(&worker), None),
                }
            };
            let guards = guard.replicate_to(2).into_iter().map(Some).collect();
            assert_eq!(worker.load(), if bounded { 1 } else { 2 });
            // Each n=1 Prefill dispatch returns one terminal response.
            // The second dispatch is unfinished when the first completes.
            let (first, _) = child(vec![complete()]);
            let (second, _) = child(vec![complete()]);
            let mut stream = FanoutStream::new(vec![first, second]);
            let mut indices = Vec::new();
            drain_prefill(&mut stream, guards, finish_at_complete, |complete| {
                indices.push(complete.index());
                assert_eq!(worker.load(), if complete.index() == 0 { 1 } else { 0 });
            })
            .await
            .unwrap();
            assert_eq!(indices, vec![0, 1]);
            assert_eq!(worker.load(), 0);
        }
    }
}

#[test]
fn set_index_restamps_chunks_and_completes() {
    let mut response = chunk("x");
    response.set_index(3);
    assert_eq!(index_of(response), 3);
    let mut response = complete();
    response.set_index(2);
    assert_eq!(index_of(response), 2);
}

#[test]
fn sampling_n_reads_and_writes_the_room_based_engines() {
    let mut sglang_req = ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
        sampling_params: Some(sglang::SamplingParams {
            n: 4,
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert_eq!(sglang_req.sampling_n(), 4);
    sglang_req.set_sampling_n(1);
    assert_eq!(sglang_req.sampling_n(), 1);
    assert!(!sglang_req.has_mm_inputs());

    let mut tokenspeed_req =
        ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed::GenerateRequest {
            sampling_params: Some(tokenspeed::SamplingParams {
                n: 0,
                sampling_seed: Some(5),
                ..Default::default()
            }),
            mm_inputs: Some(tokenspeed::MultimodalInputs::default()),
            ..Default::default()
        }));
    assert_eq!(tokenspeed_req.sampling_n(), 1, "0 means unset");
    assert!(tokenspeed_req.has_mm_inputs());
    tokenspeed_req.offset_sampling_seed(2);
    let ProtoGenerateRequest::TokenSpeed(req) = &tokenspeed_req else {
        panic!("runtime changed");
    };
    assert_eq!(req.sampling_params.as_ref().unwrap().sampling_seed, Some(7));
}
