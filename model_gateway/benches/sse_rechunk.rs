//! Per-frame cost of the SSE re-chunk stage with nobody observing it: a
//! stream of one-token content events (the common shape), one large delta
//! that is re-sliced, and a tool call's arguments in pieces. Every case
//! feeds a whole stream through a fresh stage and flushes its tail.

use std::hint::black_box;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg::routers::common::sse_rechunk::SseRechunker;

fn content_event(text: &str) -> Bytes {
    Bytes::from(format!(
        "data: {{\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}},\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap_or_default()
    ))
}

fn arguments_event(text: &str) -> Bytes {
    Bytes::from(format!(
        "data: {{\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{}}}}}]}},\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap_or_default()
    ))
}

/// Feeds `frames` through a fresh stage; the bytes it produced.
fn run(frames: &[Bytes]) -> usize {
    let mut stage = SseRechunker::new();
    let mut produced = 0;
    for frame in frames {
        produced += stage.feed(frame.clone()).len();
    }
    produced + stage.finish().len()
}

fn bench_rechunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("sse_rechunk");

    let tokens: Vec<Bytes> = (0..1000)
        .map(|i| content_event(&format!("tok{i:03} ")))
        .collect();
    group.throughput(Throughput::Elements(tokens.len() as u64));
    group.bench_function("content/one_token_per_frame", |b| {
        b.iter(|| black_box(run(black_box(&tokens))));
    });

    let large = vec![content_event(&"x".repeat(4000))];
    group.throughput(Throughput::Elements(1));
    group.bench_function("content/one_4000_char_frame", |b| {
        b.iter(|| black_box(run(black_box(&large))));
    });

    let arguments: Vec<Bytes> = (0..1000)
        .map(|i| arguments_event(&format!("\"k{i}\":1,")))
        .collect();
    group.throughput(Throughput::Elements(arguments.len() as u64));
    group.bench_function("tool_arguments/one_piece_per_frame", |b| {
        b.iter(|| black_box(run(black_box(&arguments))));
    });

    group.finish();
}

criterion_group!(benches, bench_rechunk);
criterion_main!(benches);
