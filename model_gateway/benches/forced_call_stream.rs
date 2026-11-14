//! A forced call's stream and its assignment: a chat request whose
//! `tool_choice` names a function or requires a call, read under the JSON
//! schema the gateway constrains such a call with, by the reader the gateway
//! assigns the request. One row per way the call is written (the arguments
//! of the named function, the list a required call writes) and per shape of
//! table: one that reads a thought, where the output is the thought and then
//! the JSON, and one that reads none, where it is the JSON alone. A stream
//! is 300 single-token chunks and a completion, as nanoseconds per chunk
//! (criterion, `Throughput::Elements`) and as the allocation calls and bytes
//! of one stream (a counting allocator over the binary's jemalloc, printed
//! once per row). The assignment rows time what a request pays before its
//! stream to be given its reader.
//!
//! A bench target of its own, beside `stream_state_machines`, whose seam it
//! drives: that binary's rows are gated
//! (`.github/bench-gates/stream-state-machines.toml`) and its sources stay
//! as they are.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a benchmark: its own plumbing fails loudly and its figures go to stdout"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg::routers::grpc::{EngineScript, ForcedChat, StreamBench};

/// The tokens of one stream: the chunks before the completion.
const TOKENS: usize = 300;

/// The gateway binary runs on jemalloc with its own run-time options (see
/// `src/main.rs`); the benchmark links the same allocator under the same
/// options, counting every allocation it is asked for and its bytes, as
/// `stream_state_machines` counts the calls.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
mod counting {
    use std::{
        alloc::{GlobalAlloc, Layout},
        sync::atomic::{AtomicU64, Ordering},
    };

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

    /// Allocations since the process started (allocations and reallocations;
    /// frees are not counted).
    pub(super) static CALLS: AtomicU64 = AtomicU64::new(0);
    /// The bytes those allocations asked for (a reallocation's new size).
    pub(super) static BYTES: AtomicU64 = AtomicU64::new(0);

    struct CountingJemalloc;

    fn count(bytes: usize) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    #[expect(
        unsafe_code,
        reason = "a global allocator implements the unsafe `GlobalAlloc`; every call goes to jemalloc unchanged, the counts beside it"
    )]
    unsafe impl GlobalAlloc for CountingJemalloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count(layout.size());
            unsafe { Jemalloc.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { Jemalloc.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count(layout.size());
            unsafe { Jemalloc.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count(new_size);
            unsafe { Jemalloc.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingJemalloc = CountingJemalloc;
}

/// The allocation calls since the process started and the bytes they asked
/// for; zeros where the counting allocator is not linked.
fn allocated() -> (u64, u64) {
    #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
    {
        use std::sync::atomic::Ordering;
        (
            counting::CALLS.load(Ordering::Relaxed),
            counting::BYTES.load(Ordering::Relaxed),
        )
    }
    #[cfg(any(target_env = "msvc", target_env = "musl"))]
    {
        (0, 0)
    }
}

/// `TOKENS` characters of a forced call's output: the thought a table reads
/// first when `thought`, then the JSON, the named function's arguments when
/// `named` and a required call's list otherwise, around one long string.
fn output(thought: bool, named: bool) -> String {
    const THOUGHT: &str = "<think>The question asks for a lookup; one call will do.</think>";
    const SENTENCE: &str = "The gateway streams each token as the engine produces it. ";
    let (open, close) = if named {
        (r#"{"q":""#, r#""}"#)
    } else {
        (r#"[{"name":"lookup","parameters":{"q":""#, r#""}}]"#)
    };
    let mut text = String::from(if thought { THOUGHT } else { "" });
    text.push_str(open);
    let filler = TOKENS - text.chars().count() - close.chars().count();
    text.extend(SENTENCE.chars().cycle().take(filler));
    text.push_str(close);
    text
}

fn bench_forced_calls(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let bench = runtime.block_on(StreamBench::start());
    let request = |thought: bool, named: bool| {
        if thought {
            ForcedChat::on_a_thought_table(named)
        } else {
            ForcedChat::on_a_plain_table(named)
        }
    };

    let mut streams = c.benchmark_group("forced_call_stream");
    streams.throughput(Throughput::Elements(
        u64::try_from(TOKENS + 1).unwrap_or(u64::MAX),
    ));
    for (name, thought, named) in [
        ("named/thought_table", true, true),
        ("named/plain_table", false, true),
        ("required/thought_table", true, false),
        ("required/plain_table", false, false),
    ] {
        let forced = request(thought, named);
        let script = EngineScript::single_tokens(&output(thought, named));
        let run = || runtime.block_on(bench.chat_forced(&forced, &script));
        // The first stream pays for what is built lazily; the second is the
        // steady state the counts describe.
        run();
        let before = allocated();
        let written = run();
        let after = allocated();
        println!(
            "{name}: a table reads: {}, {written} bytes written, {} allocations of {} bytes per stream",
            forced.assign(),
            after.0 - before.0,
            after.1 - before.1,
        );
        streams.bench_function(name, |b| b.iter(|| black_box(run())));
    }
    streams.finish();

    let mut assignments = c.benchmark_group("forced_call_assignment");
    for (name, thought) in [("thought_table", true), ("plain_table", false)] {
        let forced = request(thought, true);
        let before = allocated();
        let assigned = forced.assign();
        let after = allocated();
        println!(
            "{name}: a table reads: {assigned}, {} allocations of {} bytes per assignment",
            after.0 - before.0,
            after.1 - before.1,
        );
        assignments.bench_function(name, |b| b.iter(|| black_box(forced.assign())));
    }
    assignments.finish();
}

criterion_group!(benches, bench_forced_calls);
criterion_main!(benches);
