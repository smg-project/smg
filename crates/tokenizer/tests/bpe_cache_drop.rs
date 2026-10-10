//! A tokenizer the gateway removes (an unload, a re-registration, a reload)
//! must take its per-thread caches with it: the threads that encoded with it
//! are the runtime's and the blocking pool's, and they live as long as the
//! process. The `tokenizers` BPE model keeps the words it merged in a
//! thread-local the crate never clears, so every thread kept its words for
//! every tokenizer ever dropped. For a byte-level BPE of the common shape the
//! wrapper's own piece cache, owned by the encoder and freed with it, answers
//! in front of the model, and the model's cache stays empty.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a thread kept"
)]
#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]
#![expect(
    clippy::print_stdout,
    reason = "the bytes each dropped tokenizer left behind are the test's finding"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
};

use llm_tokenizer::{traits::Encoder, HuggingFaceTokenizer};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    processors::byte_level::ByteLevel as ByteLevelProcessor,
    AddedToken, Tokenizer,
};

/// Bytes currently allocated through the global allocator.
static LIVE: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= layout.size() {
            LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed);
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Threads that encode with every tokenizer they are handed, as the
/// gateway's runtime and blocking threads do for each tokenizer registered
/// while they live.
const THREADS: usize = 4;
/// Distinct words each thread encodes with one tokenizer: more than the
/// model's per-thread word cache holds, so a thread that keeps that cache
/// keeps it full.
const WORDS: usize = 2_000;
/// Tokenizers loaded, encoded with and dropped in turn while the threads
/// live on: a reload each.
const CYCLES: usize = 5;
/// What the threads together may keep of one dropped tokenizer.
const LEFT_BEHIND_MAX: usize = THREADS * 8 * 1024;

/// The 256 characters a byte-level BPE spells its bytes with: the printable
/// bytes as themselves, the others from U+0100 on.
fn byte_level_alphabet() -> Vec<char> {
    let mut next = 0x100;
    (0u32..256)
        .map(|byte| {
            let printable = (0x21..=0x7E).contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            let code = if printable {
                byte
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(code).expect("a character of the byte-level alphabet")
        })
        .collect()
}

/// A byte-level BPE of the common shape with no merges: every word is one
/// piece and one symbol per byte, and each distinct word a thread encodes
/// is one entry of the model's per-thread word cache.
fn byte_level_bpe() -> Tokenizer {
    let mut vocab = Vocab::default();
    for (id, letter) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(letter.to_string(), id as u32);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("a byte-level BPE with no merges");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        ByteLevelPreTokenizer::default().add_prefix_space(false),
    ));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer.with_post_processor(Some(ByteLevelProcessor::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|endoftext|>", true)])
        .expect("a special token");
    tokenizer
}

/// The `i`-th of the twelve-letter words, all distinct.
fn word(mut i: usize) -> String {
    let mut word = String::with_capacity(12);
    for _ in 0..12 {
        word.push((b'a' + (i % 26) as u8) as char);
        i /= 26;
    }
    word
}

#[test]
fn a_dropped_tokenizer_leaves_nothing_on_the_threads_that_encoded_with_it() {
    let (done, reports) = mpsc::channel::<()>();
    let workers: Vec<_> = (0..THREADS)
        .map(|thread| {
            let (jobs, tokenizers) = mpsc::channel::<Arc<HuggingFaceTokenizer>>();
            let done = done.clone();
            let handle = thread::spawn(move || {
                while let Ok(tokenizer) = tokenizers.recv() {
                    for i in 0..WORDS {
                        let text = word(thread * WORDS + i);
                        let encoding = tokenizer.encode(&text, false).expect("an encode");
                        assert_eq!(encoding.token_ids().len(), 12, "one id per letter");
                    }
                    drop(tokenizer);
                    done.send(()).expect("the test is waiting");
                }
            });
            (jobs, handle)
        })
        .collect();

    let mut left_behind = Vec::with_capacity(CYCLES);
    for _ in 0..CYCLES {
        let before = LIVE.load(Ordering::Relaxed);
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(byte_level_bpe()));
        for (jobs, _) in &workers {
            jobs.send(Arc::clone(&tokenizer)).expect("a live worker");
        }
        for _ in 0..THREADS {
            reports.recv().expect("a worker's report");
        }
        assert_eq!(Arc::strong_count(&tokenizer), 1, "every worker let go");
        drop(tokenizer);
        left_behind.push(LIVE.load(Ordering::Relaxed).saturating_sub(before));
    }
    for (jobs, handle) in workers {
        drop(jobs);
        handle.join().expect("a worker's end");
    }

    println!("bytes the {THREADS} threads kept of each dropped tokenizer: {left_behind:?}");
    // The first tokenizer pays each thread's one-time tables; the rest must
    // leave nothing but the model's empty per-thread slot.
    assert!(
        left_behind
            .iter()
            .skip(1)
            .all(|&bytes| bytes < LEFT_BEHIND_MAX),
        "the {THREADS} threads kept {left_behind:?} bytes of the dropped tokenizers"
    );
}
