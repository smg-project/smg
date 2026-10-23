//! A prompt made of long pieces (clauses in a script without word spaces, a
//! long identifier, a run of digits) went through the model's merges on
//! every encode: the encoder's piece cache held pieces of up to 64 bytes
//! only, and the model's own word cache, which used to hold the rest, stays
//! empty behind the direct path because it outlived the tokenizer. The
//! encoder caches long pieces too now, within a byte budget, so re-encoding
//! such a prompt touches the model for none of its pieces: the second encode
//! allocates the ids and nothing else.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what an encode allocates"
)]
#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]
#![expect(
    clippy::print_stdout,
    reason = "the bytes each encode allocated are the test's finding"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use llm_tokenizer::{traits::Encoder, HuggingFaceTokenizer};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    processors::byte_level::ByteLevel as ByteLevelProcessor,
    AddedToken, Tokenizer,
};

/// Bytes requested from the global allocator so far.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Clauses in the prompt, and characters in each: pieces of 90 bytes, well
/// above the 64 the encoder caches by count.
const CLAUSES: usize = 6;
const CHARACTERS: usize = 30;

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

/// A byte-level BPE of the common shape with no merges: one id per byte, so
/// the ids of a prompt are as many as its bytes.
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
