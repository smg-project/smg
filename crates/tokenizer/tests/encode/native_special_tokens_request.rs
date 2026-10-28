//! A tokenizer the direct encode path accepts has no post-processor that
//! adds tokens (none, or the byte-level one, which only reshapes offsets), so
//! a request for special tokens changes nothing in its ids: the direct path
//! serves that request too, with the ids `tokenizers` gives for it, instead
//! of sending every such encode through `tokenizers`.

use llm_tokenizer::{
    traits::{Encoder, Encoding},
    HuggingFaceTokenizer,
};
use tokenizers::{processors::template::TemplateProcessing, Tokenizer};

use crate::common::FEW_MERGES;

/// A byte-level BPE of the common shape with a few merges and one special
/// token, with the byte-level post-processor, which adds no tokens.
fn byte_level_bpe() -> Tokenizer {
    crate::common::byte_level_bpe(&FEW_MERGES, &["<|endoftext|>"])
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
        assert_eq!(
            matches!(with, Encoding::Plain(_)),
            ours.uses_v1_encoder(),
            "{text:?}: only v1 returns plain ids for a token-adding post-processor"
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
