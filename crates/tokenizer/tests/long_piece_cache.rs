//! A prompt made of long pieces (clauses in a script without word spaces, a
//! long identifier, a run of digits) went through the model's merges on
//! every encode: the encoder's piece cache held pieces of up to 64 bytes
//! only, and the model's own word cache, which used to hold the rest, stays
//! empty behind the direct path because it outlived the tokenizer. The
//! encoder caches long pieces too now, within a byte budget, so re-encoding
//! such a prompt touches the model for none of its pieces: the second encode
//! allocates the ids and nothing else.

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]
#![expect(
    clippy::print_stdout,
    reason = "the bytes each encode allocated are the test's finding"
)]

mod common;

use std::sync::atomic::Ordering;

use common::counting_alloc::{Counting, ALLOCATED};
use llm_tokenizer::{traits::Encoder, HuggingFaceTokenizer};
use tokenizers::Tokenizer;

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Clauses in the prompt, and characters in each: pieces of 90 bytes, well
/// above the 64 the encoder caches by count.
const CLAUSES: usize = 6;
const CHARACTERS: usize = 30;

/// A byte-level BPE of the common shape with no merges: one id per byte, so
/// the ids of a prompt are as many as its bytes.
fn byte_level_bpe() -> Tokenizer {
    common::byte_level_bpe(&[], &["<|endoftext|>"])
}

/// A prompt of `CLAUSES` clauses of `CHARACTERS` characters of a script
/// without word spaces, each closed by a punctuation mark: the byte-level
/// split makes one piece of each clause and one of each mark.
fn prompt() -> String {
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut prompt = String::new();
    for clause in 0..CLAUSES {
        for _ in 0..CHARACTERS {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            prompt.push(char::from_u32(0x4E00 + (seed % 0x51A5) as u32).expect("a character"));
        }
        prompt.push(if clause % 2 == 0 { '，' } else { '。' });
    }
    prompt
}

#[test]
fn a_prompt_of_long_pieces_is_encoded_again_without_the_model() {
    let prompt = prompt();
    let reference = byte_level_bpe()
        .encode(prompt.as_str(), false)
        .expect("the reference encode");
    let tokenizer = HuggingFaceTokenizer::from_tokenizer(byte_level_bpe());

    let before = ALLOCATED.load(Ordering::Relaxed);
    let first = tokenizer.encode(&prompt, false).expect("the first encode");
    let first_bytes = ALLOCATED.load(Ordering::Relaxed) - before;
    let before = ALLOCATED.load(Ordering::Relaxed);
    let second = tokenizer.encode(&prompt, false).expect("the second encode");
    let second_bytes = ALLOCATED.load(Ordering::Relaxed) - before;
    println!("bytes allocated by the first encode {first_bytes}, by the second {second_bytes}");

    assert_eq!(first.token_ids(), reference.get_ids());
    assert_eq!(second.token_ids(), reference.get_ids());
    assert_eq!(first.token_ids().len(), prompt.len(), "one id per byte");
    // The ids and nothing of the model's: no word of symbols, no heap of
    // merges, no token strings for any of the clauses.
    assert!(
        second_bytes * 8 < first_bytes,
        "the second encode allocated {second_bytes} bytes against {first_bytes} for the first"
    );
}
