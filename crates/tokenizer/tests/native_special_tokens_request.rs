//! A tokenizer the direct encode path accepts has no post-processor that
//! adds tokens (none, or the byte-level one, which only reshapes offsets), so
//! a request for special tokens changes nothing in its ids: the direct path
//! serves that request too, with the ids `tokenizers` gives for it, instead
//! of sending every such encode through `tokenizers`.

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use llm_tokenizer::{
    traits::{Encoder, Encoding},
    HuggingFaceTokenizer,
};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Merges, Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    processors::{byte_level::ByteLevel as ByteLevelProcessor, template::TemplateProcessing},
    AddedToken, Tokenizer,
};

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

/// A byte-level BPE of the common shape with a few merges and one special
/// token, with the byte-level post-processor, which adds no tokens.
fn byte_level_bpe() -> Tokenizer {
    let mut vocab = Vocab::default();
    for (id, letter) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(letter.to_string(), id as u32);
    }
    let mut merges = Merges::new();
    for (a, b) in [
        ("t", "h"),
        ("th", "e"),
        ("Ġ", "t"),
        ("Ġt", "h"),
        ("Ġth", "e"),
        ("l", "l"),
        ("ll", "o"),
    ] {
        let next = vocab.len() as u32;
        vocab.insert(format!("{a}{b}"), next);
        merges.push((a.to_string(), b.to_string()));
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("a byte-level BPE");
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

const TEXTS: &[&str] = &[
    "hello world",
    "the theory, then: hello<|endoftext|>the end\n",
    "  leading spaces, tabs\t and Ünïcödé, 1234 numbers",
    "",
];

#[test]
fn a_request_for_special_tokens_takes_the_direct_path_with_the_same_ids() {
    let reference = byte_level_bpe();
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    for text in TEXTS {
        let without = ours.encode(text, false).expect("an encode");
        assert!(
            matches!(without, Encoding::Plain(_)),
            "{text:?}: the direct path serves the tokenizer"
        );
        let with = ours.encode(text, true).expect("an encode");
        let expected = reference.encode(*text, true).expect("the reference encode");
        assert_eq!(with.token_ids(), expected.get_ids(), "{text:?}");
        assert_eq!(with.token_ids(), without.token_ids(), "{text:?}");
        assert!(
            matches!(with, Encoding::Plain(_)),
            "{text:?}: the direct path serves the request for special tokens"
        );
    }
}

/// The negative case the direct path's correctness now rests on: a tokenizer
/// whose post-processor adds a token (`add_bos` / `add_eos` style) must not
/// take the direct path, so a request for special tokens still gets the
/// added token from `tokenizers`.
#[test]
fn a_tokenizer_whose_post_processor_adds_a_token_keeps_the_tokenizers_path() {
    let mut reference = byte_level_bpe();
    let eot = reference
        .token_to_id("<|endoftext|>")
        .expect("the special token's id");
    let template = TemplateProcessing::builder()
        .try_single("<|endoftext|> $A:0")
        .expect("a single-sequence template")
        .special_tokens(vec![("<|endoftext|>", eot)])
        .build()
        .expect("a template post-processor");
    reference.with_post_processor(Some(template));
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    for text in TEXTS {
        let with = ours.encode(text, true).expect("an encode");
        let expected = reference.encode(*text, true).expect("the reference encode");
        assert_eq!(with.token_ids(), expected.get_ids(), "{text:?}");
        assert_eq!(
            with.token_ids().first(),
            Some(&eot),
            "{text:?}: the added token leads the ids"
        );
        assert!(
            !matches!(with, Encoding::Plain(_)),
            "{text:?}: a token-adding post-processor keeps the request off the direct path"
        );
        let without = ours.encode(text, false).expect("an encode");
        assert_eq!(
            without.token_ids(),
            reference
                .encode(*text, false)
                .expect("the reference encode")
                .get_ids(),
            "{text:?}"
        );
    }
}
