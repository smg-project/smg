//! Compatibility of the optional encode pipeline with the public wrapper.
#![expect(
    clippy::expect_used,
    reason = "test setup and reference failures must fail the test"
)]

use std::sync::Arc;

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    traits::{Decoder, Encoder, Tokenizer as _},
    CacheConfig, CachedTokenizer, DecodeStream, HuggingFaceTokenizer,
};
use tokenizers::{
    decoders::byte_level::ByteLevel,
    models::bpe::{Merges, Vocab, BPE},
    normalizers::{unicode::NFC, utils::Lowercase, Sequence as NormalizerSequence},
    pre_tokenizers::split::{Split, SplitPattern},
    processors::template::TemplateProcessing,
    AddedToken, PaddingParams, PaddingStrategy, SplitDelimiterBehavior, Tokenizer,
    TruncationParams,
};

fn fixture() -> Tokenizer {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = Vocab::default();
    for (id, ch) in alphabet.into_iter().enumerate() {
        vocab.insert(ch.to_string(), id as u32);
    }
    let mut merges = Merges::new();
    for (a, b) in [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")] {
        vocab.insert(format!("{a}{b}"), vocab.len() as u32);
        merges.push((a.to_owned(), b.to_owned()));
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("BPE");
    let mut t = Tokenizer::new(bpe);
    t.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    t.with_decoder(Some(ByteLevel::default()));
    t.add_special_tokens([
        AddedToken::from("<s>", true),
        AddedToken::from("</s>", true),
    ])
    .expect("special tokens");
    t.add_tokens([AddedToken::from("tag", false)
        .single_word(true)
        .lstrip(true)
        .rstrip(true)])
        .expect("added token flags");
    let bos = t.token_to_id("<s>").expect("BOS");
    let eos = t.token_to_id("</s>").expect("EOS");
    t.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<s> $A </s>")
            .expect("template")
            .special_tokens(vec![("<s>", bos), ("</s>", eos)])
            .build()
            .expect("post processor"),
    ));
    t
}

const TEXTS: &[&str] = &[
    "",
    " ",
    "hello",
    "hello\r\n\tworld",
    " 中文🙂 café e\u{301} ไทย العربية ",
    "<s>hello</s><s>",
    "  tag  ",
    "tagged tag tag",
    "\0\u{fffd}",
];

fn compare(reference: &Tokenizer, ours: &HuggingFaceTokenizer) {
    for specials in [false, true] {
        let batch = ours.encode_batch(TEXTS, specials).expect("batch");
        let expected = reference
            .encode_batch(TEXTS.to_vec(), specials)
            .expect("reference batch");
        for ((text, batch), expected) in TEXTS.iter().zip(batch).zip(expected) {
            let encoded = ours.encode(text, specials).expect("encode");
            assert_eq!(
                encoded.token_ids(),
                expected.get_ids(),
                "{text:?}, specials={specials}"
            );
            assert_eq!(batch.token_ids(), expected.get_ids(), "batch: {text:?}");
            for skip in [false, true] {
                assert_eq!(
                    ours.decode(encoded.token_ids(), skip).expect("decode"),
                    reference
                        .decode(expected.get_ids(), skip)
                        .expect("reference decode")
                );
            }
        }
    }
    assert!(ours
        .encode_batch(&[], false)
        .expect("empty batch")
        .is_empty());
}

#[test]
fn encode_batch_decode_and_added_token_flags_match() {
    for split_specials in [false, true] {
        let mut reference = fixture();
        reference.set_encode_special_tokens(split_specials);
        let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
        assert_eq!(ours.uses_v1_encoder(), cfg!(feature = "tokenizers-v1"));
        compare(&reference, &ours);
    }
}

#[test]
fn normalized_added_tokens_match_after_normalization() {
    let mut reference = fixture();
    reference
        .with_normalizer(Some(NormalizerSequence::new(vec![
            NFC.into(),
            Lowercase.into(),
        ])))
        .expect("normalizers");
    reference
        .add_tokens([AddedToken::from("CAFÉ", false).normalized(true)])
        .expect("normalized token");
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    assert_eq!(ours.uses_v1_encoder(), cfg!(feature = "tokenizers-v1"));
    compare(&reference, &ours);
}

#[test]
fn a_split_without_a_v1_engine_falls_back_before_encoding() {
    let mut reference = fixture();
    reference.with_pre_tokenizer(Some(
        Split::new(
            SplitPattern::Regex(r"\w+|\s+".into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .expect("custom regex"),
    ));
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    assert!(!ours.uses_v1_encoder());
    compare(&reference, &ours);
}

#[test]
fn padding_truncation_and_dropout_keep_the_existing_encoder() {
    let mut padded = fixture();
    padded.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(64),
        ..Default::default()
    }));
    let mut truncated = fixture();
    truncated
        .with_truncation(Some(TruncationParams {
            max_length: 8,
            ..Default::default()
        }))
        .expect("truncation");
    for reference in [padded, truncated] {
        let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
        assert!(!ours.uses_v1_encoder());
        compare(&reference, &ours);
    }
    let mut reference = fixture();
    if let tokenizers::models::ModelWrapper::BPE(model) = reference.get_model() {
        let mut model = model.clone();
        model.dropout = Some(0.5);
        reference.with_model(model);
    }
    assert!(!HuggingFaceTokenizer::from_tokenizer(reference).uses_v1_encoder());
}

#[test]
fn configured_bos_eos_template_and_streaming_survive_the_adapter() {
    let dir = tempfile::tempdir().expect("directory");
    let mut reference = fixture();
    reference.with_post_processor(None::<TemplateProcessing>);
    let path = dir.path().join("tokenizer.json");
    reference.save(&path, false).expect("save");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token":"<s>", "eos_token":"</s>", "add_bos_token":true, "add_eos_token":true,
            "chat_template":"{% for m in messages %}{{ m.content }}{% endfor %}"
        })
        .to_string(),
    )
    .expect("config");
    let ours =
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().expect("path")).expect("load"));
    assert_eq!(ours.uses_v1_encoder(), cfg!(feature = "tokenizers-v1"));
    let text = "hello 中文🙂";
    let encoded = ours.encode(text, true).expect("encode");
    assert_eq!(
        encoded.token_ids().first().copied(),
        reference.token_to_id("<s>")
    );
    assert_eq!(
        encoded.token_ids().last().copied(),
        reference.token_to_id("</s>")
    );
    assert!(ours
        .eos_token_ids()
        .contains(&reference.token_to_id("</s>").expect("EOS")));
    assert_eq!(
        ours.apply_chat_template(
            &[serde_json::json!({"role":"user","content":text})],
            ChatTemplateParams::default()
        )
        .expect("render"),
        text
    );
    let mut stream = DecodeStream::new(ours.clone(), &[], true);
    let mut decoded = stream
        .step_batch(encoded.token_ids())
        .expect("stream")
        .concat();
    decoded.push_str(&stream.flush().expect("flush").unwrap_or_default());
    assert_eq!(decoded, text);
    let mut incremental = ours.incremental_decoder(true).expect("byte-level decoder");
    let mut output = String::new();
    for &id in encoded.token_ids() {
        output.push_str(&incremental.step(id).expect("step"));
    }
    assert_eq!(output, text);
}

#[test]
fn both_cache_levels_preserve_ids_on_hits_and_changing_suffixes() {
    let mut reference = fixture();
    // Prefix caching is used on rendered chat text (without an EOS-appending processor).
    reference.with_post_processor(None::<TemplateProcessing>);
    let ours = Arc::new(HuggingFaceTokenizer::from_tokenizer(reference.clone()));
    for (l0, l1) in [(false, false), (true, false), (false, true), (true, true)] {
        let cached = CachedTokenizer::new(
            ours.clone(),
            CacheConfig {
                enable_l0: l0,
                enable_l1: l1,
                l1_max_memory: 1024 * 1024 * 1024,
                ..Default::default()
            },
        );
        for specials in [false, true] {
            for text in [
                "hello</s>world",
                "hello</s>world",
                "hello</s>中文🙂",
                "hello</s>",
            ] {
                assert_eq!(
                    cached.encode(text, specials).expect("cached").token_ids(),
                    reference
                        .encode(text, specials)
                        .expect("reference")
                        .get_ids()
                );
            }
        }
    }
}

/// Long cache batches preserve IDs and ordering under the gateway's outer pool.
#[test]
fn cached_long_batch_matches_reference_inside_rayon_pool() {
    let reference = fixture();
    let ours = Arc::new(HuggingFaceTokenizer::from_tokenizer(reference.clone()));
    assert_eq!(ours.uses_v1_encoder(), cfg!(feature = "tokenizers-v1"));
    let cached = CachedTokenizer::new(
        ours,
        CacheConfig {
            enable_l0: false,
            enable_l1: false,
            ..Default::default()
        },
    );
    let texts = ["hello 中文🙂 ".repeat(2048), "world café ".repeat(2048)];
    let inputs: Vec<_> = texts.iter().map(String::as_str).collect();
    assert!(inputs.iter().all(|text| text.len() >= 16 * 1024));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("outer pool");
    for specials in [false, true] {
        let actual = pool.install(|| {
            cached
                .encode_batch(&inputs, specials)
                .expect("cached batch")
        });
        for (text, actual) in inputs.iter().zip(actual) {
            assert_eq!(
                actual.token_ids(),
                reference
                    .encode(*text, specials)
                    .expect("reference")
                    .get_ids()
            );
        }
    }
}

/// Run explicitly with the four pinned model directories from the benchmark.
#[test]
#[ignore = "requires SMG_V1_TOKENIZER_DIRS (comma-separated tokenizer directories)"]
fn real_models_match_on_unicode_batches_caches_and_concurrent_calls() {
    let dirs = std::env::var("SMG_V1_TOKENIZER_DIRS").expect("set model directories");
    let alphabet: Vec<char> = "abcXYZ019 !\r\n\t\0éß中日ไทยالعربية🙂\u{301}\u{200d}\u{fffd}"
        .chars()
        .collect();
    let mut seed = 42u64;
    let mut texts: Vec<String> = TEXTS.iter().map(|s| (*s).to_owned()).collect();
    for n in 0..128 {
        let text = (0..n * 17)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                alphabet[seed as usize % alphabet.len()]
            })
            .collect();
        texts.push(text);
    }
    texts.push("hello 中文🙂 fn main() {}\n".repeat(8000));
    for dir in dirs.split(',') {
        let reference = Tokenizer::from_file(std::path::Path::new(dir).join("tokenizer.json"))
            .expect("reference model");
        let ours = Arc::new(HuggingFaceTokenizer::from_tokenizer(reference.clone()));
        assert_eq!(
            ours.uses_v1_encoder(),
            cfg!(feature = "tokenizers-v1"),
            "{dir}"
        );
        let specials: Vec<_> = reference
            .get_added_tokens_decoder()
            .values()
            .filter(|t| t.special)
            .map(|t| t.content.clone())
            .collect();
        let mut model_texts = texts.clone();
        for marker in specials.iter().take(32) {
            model_texts.push(format!("{marker} hello 中文🙂{marker} tail"));
        }
        for add in [false, true] {
            for chunk in model_texts.chunks(16) {
                let inputs: Vec<_> = chunk.iter().map(String::as_str).collect();
                let batch = ours.encode_batch(&inputs, add).expect("batch");
                for (text, batched) in inputs.iter().zip(batch) {
                    let expected = reference.encode(*text, add).expect("reference");
                    assert_eq!(
                        ours.encode(text, add).expect("encode").token_ids(),
                        expected.get_ids(),
                        "{dir}: {text:?}"
                    );
                    assert_eq!(batched.token_ids(), expected.get_ids(), "{dir}: batch");
                    for skip in [false, true] {
                        assert_eq!(
                            ours.decode(expected.get_ids(), skip).expect("decode"),
                            reference
                                .decode(expected.get_ids(), skip)
                                .expect("reference decode")
                        );
                    }
                }
            }
        }
        let marker = specials.first().expect("special token");
        for (l0, l1) in [(false, false), (true, false), (false, true), (true, true)] {
            let cached = Arc::new(CachedTokenizer::new(
                ours.clone(),
                CacheConfig {
                    enable_l0: l0,
                    enable_l1: l1,
                    l1_max_memory: 1024 * 1024 * 1024,
                    ..Default::default()
                },
            ));
            std::thread::scope(|scope| {
                for worker in 0..16 {
                    let cached = &cached;
                    let reference = &reference;
                    scope.spawn(move || {
                        for n in 0..8 {
                            let text = format!("hello 中文🙂{marker} worker {worker} turn {n}");
                            for add in [false, true] {
                                let expected =
                                    reference.encode(text.as_str(), add).expect("reference");
                                for _ in 0..2 {
                                    assert_eq!(
                                        cached.encode(&text, add).expect("cached").token_ids(),
                                        expected.get_ids()
                                    );
                                }
                            }
                        }
                    });
                }
            });
        }
    }
}
