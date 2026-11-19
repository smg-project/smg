//! A tokenizer load leaves no thread pool behind. The direct encoder reads
//! the vocabulary into its seed on the caller's rayon pool when the load runs
//! on a worker of one, else on a pool built for that read and joined before
//! the load returns. Read on rayon's global pool instead, the first load
//! would build that pool at the machine's parallelism and keep its threads,
//! idle, for the life of the process: a gateway with a CPU quota of sixteen
//! carried sixteen such threads. This binary holds the one test, so nothing
//! else in the process can have built the global pool before it, and the
//! subscriber that reads the seed's debug line can be the process's default.
//!
//! The default build only: the opt-in `tokenizers-v1` pipeline builds the
//! global pool at its own load (a dependency of that pipeline builds the
//! added-vocabulary hash on it), which is that pipeline's, not the seed's.
#![cfg(not(feature = "tokenizers-v1"))]

mod common;

use std::sync::{Arc, Mutex, PoisonError};

use llm_tokenizer::HuggingFaceTokenizer;

/// A tokenizer loaded as the loaders build it.
fn load() -> HuggingFaceTokenizer {
    HuggingFaceTokenizer::from_tokenizer(common::byte_level_bpe(
        &common::FEW_MERGES,
        &["<|endoftext|>"],
    ))
}

/// The `threads` of every vocabulary seed's debug line, in order: the size of
/// the pool the seed read on. A pool worker's load logs it from that pool's
/// thread, so this subscriber is the process's default, not one thread's.
struct SeedThreads(Arc<Mutex<Vec<u64>>>);

impl tracing::Subscriber for SeedThreads {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        /// The `threads` field of one event.
        struct Threads(Option<u64>);
        impl tracing::field::Visit for Threads {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                if field.name() == "threads" {
                    self.0 = Some(value);
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        // The seed's line is the one that says whether its table covers the
        // short entries.
        if event
            .metadata()
            .fields()
            .field("covers_short_entries")
            .is_none()
        {
            return;
        }
        let mut threads = Threads(None);
        event.record(&mut threads);
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(threads.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn a_load_builds_no_global_thread_pool() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    tracing::subscriber::set_global_default(SeedThreads(Arc::clone(&seen)))
        .expect("the process's first subscriber");
    // A load from a plain thread, as the gateway's: the seed reads on four
    // threads of its own.
    drop(load());
    assert_eq!(
        std::mem::take(&mut *seen.lock().unwrap_or_else(PoisonError::into_inner)),
        [4],
        "a plain thread's load reads on four threads of its own"
    );
    // A load from a worker of a pool of the caller's own, as the Python
    // binding's: the seed reads on that pool.
    let callers = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("a two-thread pool");
    drop(callers.install(load));
    drop(callers);
    assert_eq!(
        std::mem::take(&mut *seen.lock().unwrap_or_else(PoisonError::into_inner)),
        [2],
        "a pool worker's load reads on that pool"
    );
    // The global pool can be built only while none exists: this succeeds
    // exactly when the loads left none behind (the pool it builds is the
    // test's own, one thread, and nothing here uses it).
    assert!(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok(),
        "the load built rayon's global thread pool"
    );
}
