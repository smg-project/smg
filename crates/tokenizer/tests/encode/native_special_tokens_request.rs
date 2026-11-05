//! A tokenizer the direct encode path accepts adds, on a request for special
//! tokens, what its post-processor prescribes: nothing for none or the
//! byte-level one (which only reshapes offsets), the frame of a
//! `TemplateProcessing` otherwise. Both requests are served by the direct
//! path, with the ids `tokenizers` gives for them, instead of sending every
//! encode with the flag through `tokenizers`.

use llm_tokenizer::{
    traits::{Encoder, Encoding},
    EncodePath, HuggingFaceTokenizer,
};
use tokenizers::{
    processors::{
        byte_level::ByteLevel as ByteLevelProcessor, sequence::Sequence,
        template::TemplateProcessing,
    },
    Tokenizer,
};

use crate::common::FEW_MERGES;

/// A byte-level BPE of the common shape with a few merges and one special
/// token, with the byte-level post-processor, which adds no tokens.
fn byte_level_bpe() -> Tokenizer {
    crate::common::byte_level_bpe(&FEW_MERGES, &["<|endoftext|>"])
}

/// The template that puts the special token (`id` in the vocabulary) in
/// front of the sequence and after it.
#[expect(
    clippy::expect_used,
    reason = "a helper of tests that stop on a broken setup"
)]
fn frame(id: u32) -> TemplateProcessing {
    TemplateProcessing::builder()
        .try_single("<|endoftext|> $A:0 <|endoftext|>")
        .expect("a single-sequence template")
        .special_tokens(vec![("<|endoftext|>", id)])
        .build()
        .expect("a template post-processor")
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

/// A tokenizer whose post-processor adds tokens (`add_bos` / `add_eos`
/// style) takes the direct path too: the path frames the ids itself on
/// request, so they are the ones `tokenizers` gives either way.
#[test]
fn a_tokenizer_whose_post_processor_adds_tokens_takes_the_direct_path_with_its_frame() {
    let mut reference = byte_level_bpe();
    let eot = reference
        .token_to_id("<|endoftext|>")
        .expect("the special token's id");
    reference.with_post_processor(Some(frame(eot)));
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    assert_ne!(ours.encode_path(), EncodePath::Tokenizers);
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
            with.token_ids().last(),
            Some(&eot),
            "{text:?}: and closes them"
        );
        assert!(
            matches!(with, Encoding::Plain(_)),
            "{text:?}: the direct path serves the request for special tokens"
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
        assert_eq!(
            without.token_ids().len() + 2,
            with.token_ids().len(),
            "{text:?}: the frame is the two ids"
        );
    }
}

/// The form the Llama-3-class files carry, read from a `tokenizer.json`: a
/// `Sequence` of the byte-level processor and a template. The loader puts
/// the file on the direct path, and the ids with and without special tokens
/// are those `tokenizers` reads from the same file.
#[test]
fn a_file_with_a_template_post_processor_loads_onto_the_direct_path() {
    let mut written = byte_level_bpe();
    let eot = written
        .token_to_id("<|endoftext|>")
        .expect("the special token's id");
    written.with_post_processor(Some(Sequence::new(vec![
        ByteLevelProcessor::default().trim_offsets(false).into(),
        frame(eot).into(),
    ])));
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("tokenizer.json");
    written.save(&path, false).expect("the file is written");
    let path = path.to_str().expect("a utf-8 path");
    let reference = Tokenizer::from_file(path).expect("the reference reads the file");
    let ours = HuggingFaceTokenizer::from_file(path).expect("the loader reads the file");
    assert_ne!(
        ours.encode_path(),
        EncodePath::Tokenizers,
        "the direct path takes the file"
    );
    for text in TEXTS {
        for add_special_tokens in [false, true] {
            let encoding = ours.encode(text, add_special_tokens).expect("an encode");
            let expected = reference
                .encode(*text, add_special_tokens)
                .expect("the reference encode");
            assert_eq!(
                encoding.token_ids(),
                expected.get_ids(),
                "{text:?} with special tokens {add_special_tokens}"
            );
            assert!(
                matches!(encoding, Encoding::Plain(_)),
                "{text:?} with special tokens {add_special_tokens}: the direct path serves it"
            );
        }
    }
}
