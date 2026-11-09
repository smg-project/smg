//! The per-token paths of the gateway's stream state machines over a
//! scripted engine stream: a chat under the legacy parsers and under a
//! Symphony table, a completion, a generate and a Messages stream, each of
//! 300 single-token chunks and a completion, as nanoseconds per chunk
//! (criterion, `Throughput::Elements`) and allocations per stream (a
//! counting allocator over the binary's jemalloc, printed once per row).
//! A pull request is gated on the criterion figures by
//! `.github/bench-gates/stream-state-machines.toml`.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a benchmark: its own plumbing fails loudly and its figures go to stdout"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg::routers::grpc::{EngineScript, StreamBench};

/// The tokens of one stream: the chunks before the completion.
const TOKENS: usize = 300;

/// One row of the group: its id, the script it streams and the driver that
/// runs the script once, returning the bytes written.
type Row<'a> = (
    &'a str,
    &'a EngineScript,
    &'a dyn Fn(&EngineScript) -> usize,
);

/// The gateway binary runs on jemalloc with its own run-time options (see
/// `src/main.rs`); the benchmark links the same allocator under the same
/// options, counting every allocation it is asked for.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
mod counting {
    use std::{
        alloc::{GlobalAlloc, Layout},
        sync::atomic::{AtomicU64, Ordering},
    };

    use smg::observability::metrics::SERVER_MALLOC_CONF;
    use tikv_jemallocator::Jemalloc;

    /// The gateway's jemalloc options, exported as `src/main.rs` exports
    /// them, so the rows run under production's decay and purge behaviour
    /// and not under jemalloc's defaults.
    #[expect(
        unsafe_code,
        reason = "jemalloc reads its options from this exported symbol, as src/main.rs exports it"
    )]
    #[export_name = "_rjem_malloc_conf"]
    pub static MALLOC_CONF: &[u8; 100] = SERVER_MALLOC_CONF;

    /// Allocations since the process started (allocations and reallocations;
    /// frees are not counted).
    pub(super) static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

    struct CountingJemalloc;

    #[expect(
        unsafe_code,
        reason = "a global allocator implements the unsafe `GlobalAlloc`; every call goes to jemalloc unchanged, the count beside it"
    )]
    unsafe impl GlobalAlloc for CountingJemalloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            unsafe { Jemalloc.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { Jemalloc.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            unsafe { Jemalloc.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            unsafe { Jemalloc.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingJemalloc = CountingJemalloc;
}

/// Allocations since the process started; zero where the counting allocator
/// is not linked.
fn allocations() -> u64 {
    #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
    {
        use std::sync::atomic::Ordering;
        counting::ALLOCATIONS.load(Ordering::Relaxed)
    }
    #[cfg(any(target_env = "msvc", target_env = "musl"))]
    {
        0
    }
}

/// `tokens` characters of an answer: a `<think>` block the chat and Messages
/// streams read as reasoning, then prose.
fn thinking_text(tokens: usize) -> String {
    let mut text =
        String::from("<think>The question asks for a greeting; a short one will do.</think>");
    fill(&mut text, tokens)
}

/// `tokens` characters of prose.
fn prose_text(tokens: usize) -> String {
    fill(&mut String::new(), tokens)
}

fn fill(text: &mut String, tokens: usize) -> String {
    const SENTENCE: &str = "Hello! The gateway streams each token as the engine produces it. ";
    while text.chars().count() < tokens {
        text.push_str(SENTENCE);
    }
    text.chars().take(tokens).collect()
}

fn bench_streams(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let bench = runtime.block_on(StreamBench::start());
    let thinking = EngineScript::single_tokens(&thinking_text(TOKENS));
    let prose = EngineScript::single_tokens(&prose_text(TOKENS));
    let rows: [Row<'_>; 5] = [
        ("chat_legacy", &thinking, &|script| {
            runtime.block_on(bench.chat(script))
        }),
        ("chat_symphony", &thinking, &|script| {
            runtime.block_on(bench.chat_symphony(script))
        }),
        ("completion", &prose, &|script| {
            runtime.block_on(bench.completion(script))
        }),
        ("generate", &prose, &|script| {
            runtime.block_on(bench.generate(script))
        }),
        ("messages", &thinking, &|script| {
            runtime.block_on(bench.messages(script))
        }),
    ];
    let mut group = c.benchmark_group("stream_state_machines");
    group.throughput(Throughput::Elements(
        u64::try_from(TOKENS + 1).unwrap_or(u64::MAX),
    ));
    for (name, script, run) in rows {
        // The first stream pays for what is built lazily; the second is the
        // steady state the count describes.
        run(script);
        let before = allocations();
        let bytes = run(script);
        let allocations = allocations() - before;
        println!("{name}: {bytes} bytes written, {allocations} allocations per stream");
        group.bench_function(name, |b| b.iter(|| black_box(run(script))));
    }
    group.finish();
}

criterion_group!(benches, bench_streams);
criterion_main!(benches);
