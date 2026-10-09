//! The `tokenizers` BPE model remembers the words it merged in a cache per
//! thread, and the gateway encodes on every runtime and blocking thread it
//! has: left at the crate's default, those caches grew with the thread count
//! (a few MB per thread) and never settled. The wrapper bounds them, so a
//! thread that encodes many distinct words keeps a few hundred KB.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a thread's cache kept"
)]
#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use llm_tokenizer::{traits::Encoder, HuggingFaceTokenizer};
use tokenizers::Tokenizer;

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

/// A BPE over the lower-case letters with no merges, split on whitespace:
/// every word is one symbol per letter, and each distinct word the model
/// sees is one entry of its per-thread cache.
fn letters_bpe() -> Tokenizer {
    let vocab: Vec<String> = ('a'..='z')
        .enumerate()
        .map(|(id, letter)| format!("\"{letter}\":{id}"))
        .collect();
    let json = format!(
        concat!(
            "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],",
            "\"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},",
            "\"post_processor\":null,\"decoder\":null,",
            "\"model\":{{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,",
            "\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\"fuse_unk\":false,",
            "\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{{{}}},\"merges\":[]}}}}"
        ),
        vocab.join(",")
    );
    Tokenizer::from_bytes(json.as_bytes()).expect("a letters-only BPE tokenizer")
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
fn a_thread_that_encodes_many_distinct_words_keeps_a_bounded_cache() {
    let tokenizer = HuggingFaceTokenizer::from_tokenizer(letters_bpe());
    // The first encode on a thread sets up its tables; measure from there.
    tokenizer
        .encode(&word(usize::MAX / 7), false)
        .expect("the warm-up encode");
    let before = LIVE.load(Ordering::Relaxed);
    for i in 0..20_000 {
        let text = word(i);
        let encoding = tokenizer.encode(&text, false).expect("an encode");
        assert_eq!(encoding.token_ids().len(), 12, "one id per letter");
    }
    let kept = LIVE.load(Ordering::Relaxed).saturating_sub(before);
    assert!(
        kept < 1 << 20,
        "the thread's BPE word cache kept {kept} bytes after 20,000 distinct words"
    );
}
