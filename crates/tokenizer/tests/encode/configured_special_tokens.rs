//! The configuration's special tokens on request. An encode with
//! `add_special_tokens` carries the BOS and EOS `tokenizer_config.json`
//! prescribes (`add_bos_token` / `add_eos_token`), through the post-processor
//! the `transformers` Llama-style fast classes install at load, even where
//! `tokenizer.json` alone has a plain `ByteLevel` post-processor and adds
//! none; without the flag the ids are the file's alone. So a cross-check
//! against the `tokenizers` crate compares with the same pipeline
//! ([`HuggingFaceTokenizer::tokenizers_pipeline`]) or with the flag off,
//! and the configured BOS does not read as one extra leading id.

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use llm_tokenizer::{traits::Encoder, HuggingFaceTokenizer};
use serde_json::{json, Map, Value};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Merges, Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    processors::byte_level::ByteLevel as ByteLevelProcessor,
    AddedToken, Tokenizer,
};

const BOS: &str = "<|startoftext|>";
const EOS: &str = "<|endoftext|>";

/// A `tokenizer_config.json` shape: none, or its `add_bos_token` and
/// `add_eos_token`, each `None` when the key is left out.
type ConfigFlags = Option<(Option<bool>, Option<bool>)>;

const TEXTS: &[&str] = &[
    "hello world",
    "the theory, then: hello<|endoftext|>the end\n",
    "  leading spaces, tabs\t and Ünïcödé, 1234 numbers",
    "",
];

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

/// A byte-level BPE of the common shape with a few merges, a BOS and an EOS
/// in the added vocabulary, and the byte-level post-processor, which adds no
/// token: the shape of the checkpoints whose configuration alone asks for
/// the BOS.
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
        .add_special_tokens([AddedToken::from(BOS, true), AddedToken::from(EOS, true)])
        .expect("the special tokens");
    tokenizer
}

/// A checkpoint directory: the tokenizer above as `tokenizer.json` and, when
/// the flags are given, a `tokenizer_config.json` naming the BOS and EOS with
/// `add_bos_token` / `add_eos_token` as given (a `None` leaves the key out).
fn checkpoint(flags: ConfigFlags) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("tokenizer.json");
    byte_level_bpe().save(&path, false).expect("the file");
    if let Some((add_bos, add_eos)) = flags {
        let mut config = Map::new();
        config.insert("bos_token".into(), json!(BOS));
        config.insert("eos_token".into(), json!(EOS));
        if let Some(add_bos) = add_bos {
            config.insert("add_bos_token".into(), json!(add_bos));
        }
        if let Some(add_eos) = add_eos {
            config.insert("add_eos_token".into(), json!(add_eos));
        }
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            Value::Object(config).to_string(),
        )
        .expect("the configuration");
    }
    let path = path.to_str().expect("a UTF-8 path").to_owned();
    (dir, path)
}

fn ids(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    tokenizer
        .encode(text, add_special_tokens)
        .expect("the reference encode")
        .get_ids()
        .to_vec()
}

#[test]
fn the_file_alone_adds_no_token_on_request() {
    let (_dir, path) = checkpoint(None);
    let bare = Tokenizer::from_file(&path).expect("the file loads");
    for text in TEXTS {
        assert_eq!(ids(&bare, text, true), ids(&bare, text, false), "{text:?}");
    }
}

/// Every configuration shape, with the ids a request for special tokens
/// adds around the file's own: the configured BOS and EOS when the flags
/// say so, nothing unless one of them is an explicit `true`.
#[test]
fn the_configured_bos_and_eos_are_added_on_request_where_the_file_adds_none() {
    let shapes: &[(ConfigFlags, bool, bool)] = &[
        (Some((Some(true), None)), true, false),
        (Some((Some(true), Some(false))), true, false),
        (Some((Some(true), Some(true))), true, true),
        // With the BOS unset, the file's post-processor decides it: this one
        // adds nothing, so only the configured EOS comes.
        (Some((None, Some(true))), false, true),
        (Some((Some(false), Some(false))), false, false),
        (Some((None, None)), false, false),
        (None, false, false),
    ];
    for &(flags, bos, eos) in shapes {
        let (_dir, path) = checkpoint(flags);
        let bare = Tokenizer::from_file(&path).expect("the file loads");
        let ours = HuggingFaceTokenizer::from_file(&path).expect("the crate loads it");
        let bos_id = bare.token_to_id(BOS).expect("the BOS id");
        let eos_id = bare.token_to_id(EOS).expect("the EOS id");
        for text in TEXTS {
            let file_ids = ids(&bare, text, false);
            let mut expected = Vec::with_capacity(file_ids.len() + 2);
            if bos {
                expected.push(bos_id);
            }
            expected.extend_from_slice(&file_ids);
            if eos {
                expected.push(eos_id);
            }
            assert_eq!(
                ours.encode(text, true).expect("an encode").token_ids(),
                expected.as_slice(),
                "{flags:?} {text:?}: the ids with special tokens requested"
            );
            assert_eq!(
                ours.encode(text, false).expect("an encode").token_ids(),
                file_ids.as_slice(),
                "{flags:?} {text:?}: without the flag the ids are the file's"
            );
        }
    }
}

/// The alignment a cross-check against the `tokenizers` crate needs: the
/// crate's pipeline carries the configured post-processor, so comparing with
/// it agrees with every flag, while the file read alone differs by the
/// configured BOS exactly when special tokens are requested.
#[test]
fn a_cross_check_compares_with_the_same_pipeline_or_with_the_flag_off() {
    let (_dir, path) = checkpoint(Some((Some(true), None)));
    let bare = Tokenizer::from_file(&path).expect("the file loads");
    let ours = HuggingFaceTokenizer::from_file(&path).expect("the crate loads it");
    let pipeline = ours.tokenizers_pipeline();
    let bos_id = bare.token_to_id(BOS).expect("the BOS id");
    for text in TEXTS {
        for add_special_tokens in [false, true] {
            let got = ours
                .encode(text, add_special_tokens)
                .expect("an encode")
                .token_ids()
                .to_vec();
            assert_eq!(
                got,
                ids(pipeline, text, add_special_tokens),
                "{text:?} ({add_special_tokens}): the crate's own pipeline agrees"
            );
            let file = ids(&bare, text, add_special_tokens);
            if add_special_tokens {
                assert_eq!(
                    got.first(),
                    Some(&bos_id),
                    "{text:?}: the configured BOS leads"
                );
                assert_eq!(
                    got[1..],
                    file[..],
                    "{text:?}: one leading id, the rest the file's"
                );
            } else {
                assert_eq!(
                    got, file,
                    "{text:?}: the file read alone agrees with the flag off"
                );
            }
        }
    }
}
