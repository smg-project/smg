//! A Messages stream under a Symphony table, per token: the reader's events
//! through the content-block machine to the SSE channel, over a scripted
//! engine stream of 300 single-token chunks and a completion, as nanoseconds
//! per chunk (criterion, `Throughput::Elements`). One row per shape of
//! answer: prose alone, where every chunk is a text delta into the open text
//! block, and a `<think>` block read as thinking before the prose.
//!
//! A bench target of its own, beside `stream_state_machines`, whose seam it
//! drives: that binary's rows are gated
//! (`.github/bench-gates/stream-state-machines.toml`) and its sources stay
//! as they are.
#![expect(
    clippy::expect_used,
    reason = "a benchmark: its own plumbing fails loudly"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg::routers::grpc::{EngineScript, StreamBench};

/// The gateway binary runs on jemalloc with its own run-time options (see
/// `src/main.rs`); the benchmark links the same allocator under the same
/// options, as `stream_state_machines` does.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
mod allocator {
    use smg::observability::metrics::SERVER_MALLOC_CONF;
    use tikv_jemallocator::Jemalloc;

    /// The gateway's jemalloc options, exported as `src/main.rs` exports
    /// them.
    #[expect(
        unsafe_code,
        reason = "jemalloc reads its options from this exported symbol, as src/main.rs exports it"
    )]
    #[export_name = "_rjem_malloc_conf"]
    pub static MALLOC_CONF: &[u8; 100] = SERVER_MALLOC_CONF;

    #[global_allocator]
    static GLOBAL_ALLOCATOR: Jemalloc = Jemalloc;
}

/// The tokens of one stream: the chunks before the completion.
const TOKENS: usize = 300;

/// `TOKENS` characters of an answer: a `<think>` block first when `thought`,
/// then prose.
fn answer(thought: bool) -> String {
    const THOUGHT: &str = "<think>The question asks for a greeting; a short one will do.</think>";
    const SENTENCE: &str = "Hello! The gateway streams each token as the engine produces it. ";
    let thought = if thought { THOUGHT } else { "" };
    thought
        .chars()
        .chain(SENTENCE.chars().cycle())
        .take(TOKENS)
        .collect()
}

fn bench_streams(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let bench = runtime.block_on(StreamBench::start());
    let mut group = c.benchmark_group("symphony_messages_stream");
    group.throughput(Throughput::Elements(
        u64::try_from(TOKENS + 1).unwrap_or(u64::MAX),
    ));
    for (name, thought) in [("prose", false), ("thinking", true)] {
        let script = EngineScript::single_tokens(&answer(thought));
        // The first stream pays for what is built lazily.
        runtime.block_on(bench.messages_symphony(&script));
        group.bench_function(name, |b| {
            b.iter(|| black_box(runtime.block_on(bench.messages_symphony(&script))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_streams);
criterion_main!(benches);
