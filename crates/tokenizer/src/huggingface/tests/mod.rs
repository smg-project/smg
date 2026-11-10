mod segment;
mod structural;

use std::{collections::HashMap, fs};

use serde_json::json;
use tempfile::TempDir;

use super::{derive_thinking_mode, detect_renderer_from_config, Renderer};
use crate::{
    chat_template::ChatTemplateParams,
    encode_path::EncodePath,
    encoders::deepseek_v32::ThinkingMode,
    traits::{Decoder as _, Encoder as _, Tokenizer as _},
    HuggingFaceTokenizer,
};

const MIN_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "BPE",
        "vocab": { "hello": 0 },
        "merges": []
    }
}"#;

/// A byte-level BPE of the shape the direct encode path takes: two
/// letters, the space mark and the merges that join it to a letter.
const BYTE_LEVEL_TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
    "post_processor": null,
    "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
    "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
        "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
        "vocab": {"a": 0, "b": 1, "Ġ": 2, "Ġa": 3, "Ġb": 4},
        "merges": [["Ġ", "a"], ["Ġ", "b"]]}
}"#;

/// A word-level vocabulary: no direct encode path takes it, so its
/// encodes run in `tokenizers`.
const WORD_LEVEL_TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
    "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
    "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "<unk>": 2}, "unk_token": "<unk>"}
}"#;

/// The word-level vocabulary with two added tokens matched after
/// normalization that share one normalized spelling: a letter (U+05D0)
/// and its compatibility form (U+FB21) under NFKC, their ids past the
/// model's vocabulary as a checkpoint's added tokens are.
const NORMALIZED_TIE_TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": [
        {"id": 3, "content": "\u05d0", "single_word": false, "lstrip": false, "rstrip": false,
         "normalized": true, "special": false},
        {"id": 4, "content": "\ufb21", "single_word": false, "lstrip": false, "rstrip": false,
         "normalized": true, "special": false}
    ],
    "normalizer": {"type": "NFKC"},
    "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
    "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "<unk>": 2}, "unk_token": "<unk>"}
}"#;

/// The added tokens a prefix cache may match are those of the raw pass
/// (`normalized: false`); the tokens matched after normalization, special
/// or not, are reported apart, so the cache does not split at any of them.
#[test]
fn the_cache_sees_the_raw_pass_added_tokens_and_the_normalized_ones_apart() {
    use tokenizers::{
        models::bpe::{Merges, Vocab, BPE},
        AddedToken,
    };

    use super::HuggingFaceTokenizer;
    use crate::traits::Tokenizer as TokenizerTrait;
    let mut vocab = Vocab::default();
    for (id, piece) in ["<unk>", "a", "b", "<", ">"].iter().enumerate() {
        vocab.insert((*piece).to_string(), id as u32);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Merges::new())
        .unk_token("<unk>".into())
        .build()
        .expect("model");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer
        .add_special_tokens([
            AddedToken::from("<|s|>", true),
            AddedToken::from("<|n|>", true).normalized(true),
        ])
        .expect("special tokens added");
    tokenizer
        .add_tokens([
            AddedToken::from("</think>", false).normalized(false),
            AddedToken::from("a<", false),
        ])
        .expect("added tokens added");
    let wrapped = HuggingFaceTokenizer::from_tokenizer(tokenizer);
    assert_eq!(wrapped.added_token_strings(), vec!["</think>".to_owned()]);
    let mut after_normalization = wrapped.added_tokens_after_normalization();
    after_normalization.sort();
    assert_eq!(
        after_normalization,
        vec!["<|n|>".to_owned(), "a<".to_owned()]
    );
}

/// The one-shot decode through the piece table is byte for byte the
/// `tokenizers` decode: random id sequences of word pieces, byte runs
/// (valid, incomplete, lost), specials, a lone marker and a piece ending
/// in U+FFFD, with and without the trailing strip, under both
/// special-token settings, long and short.
#[test]
fn the_piece_table_decodes_what_the_tokenizers_decode_decodes() {
    use super::HuggingFaceTokenizer;
    use crate::{
        byte_fallback::tests::{random_ids, synthetic as synthetic_for_decode},
        traits::Decoder as _,
    };
    for strip in [false, true] {
        for skip in [false, true] {
            let with = HuggingFaceTokenizer::from_tokenizer(synthetic_for_decode(strip));
            assert!(with.pieces.is_some(), "the synthetic chain has a table");
            let mut without = HuggingFaceTokenizer::from_tokenizer(synthetic_for_decode(strip));
            without.pieces = None;
            let mut state = 0x2545_F491_4F6C_DD1Du64 ^ u64::from(strip) ^ (u64::from(skip) << 1);
            for round in 0..2_000usize {
                let ids = random_ids(&mut state, 1 + round % 40);
                assert_eq!(
                    with.decode(&ids, skip).unwrap(),
                    without.decode(&ids, skip).unwrap(),
                    "ids {ids:?} strip {strip} skip {skip}"
                );
            }
            // A long sequence, as a one-shot decode of a whole output is.
            let long: Vec<u32> = (0..400)
                .flat_map(|r| random_ids(&mut state, 1 + r % 40))
                .collect();
            assert_eq!(
                with.decode(&long, skip).unwrap(),
                without.decode(&long, skip).unwrap(),
                "{} ids, strip {strip} skip {skip}",
                long.len()
            );
        }
    }
}

/// With the piece table the step algorithm streams byte for byte what it
/// streams through the `tokenizers` window decode. Random streams of word
/// pieces, byte runs (valid, incomplete, lost), specials, a lone marker
/// and a piece ending in U+FFFD go through the raw client step, the eager
/// step and `finish_decode`, fresh and seeded, with and without the
/// trailing strip, under both special-token settings; the raw step keeps
/// the process-wide recovery counters still, which other tests of this
/// binary read. Streams without a lost run also go through `Sequence`,
/// the gateway's path.
#[test]
fn the_piece_table_streams_what_the_tokenizers_window_decode_streams() {
    use std::sync::Arc;

    use super::HuggingFaceTokenizer;
    use crate::{
        byte_fallback::tests::{random_ids, random_ids_without_lost_runs, synthetic},
        sequence::Sequence,
        traits::{Decoder, Tokenizer as TokenizerTrait},
    };
    type Window = (Vec<u32>, String, usize);
    for strip in [false, true] {
        for skip in [false, true] {
            let with = HuggingFaceTokenizer::from_tokenizer(synthetic(strip));
            assert!(with.pieces.is_some(), "the synthetic chain has a table");
            let mut without = HuggingFaceTokenizer::from_tokenizer(synthetic(strip));
            without.pieces = None;
            let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(strip) ^ (u64::from(skip) << 1);
            for round in 0..1_200usize {
                let seed = random_ids(&mut state, round % 5);
                let ids = random_ids(&mut state, 1 + round % 24);
                let eager = round % 3 == 2;
                // The window a seeded sequence starts from: the seed as
                // context, nothing emitted.
                let mut a: Window = (seed.clone(), String::new(), seed.len());
                let mut b: Window = (seed.clone(), String::new(), seed.len());
                let step = |tok: &HuggingFaceTokenizer, w: &mut Window, id: u32| {
                    if eager {
                        tok.decode_step_eager(id, &mut w.0, &mut w.1, &mut w.2, skip)
                    } else {
                        tok.decode_step(id, &mut w.0, &mut w.1, &mut w.2, skip)
                    }
                };
                for &id in &ids {
                    assert_eq!(
                        step(&with, &mut a, id).unwrap(),
                        step(&without, &mut b, id).unwrap(),
                        "seed {seed:?} ids {ids:?} at {id} strip {strip} skip {skip} eager {eager}"
                    );
                }
                // The end of input, where nothing is pending that a lost
                // run would make the counters record: the last id closes
                // any run (a skipped special does not).
                if ids
                    .last()
                    .is_some_and(|&id| (3..=7).contains(&id) || (!skip && (1..=2).contains(&id)))
                {
                    let fa = with
                        .finish_decode(&mut a.0, &mut a.1, &mut a.2, skip)
                        .unwrap();
                    let fb = without
                        .finish_decode(&mut b.0, &mut b.1, &mut b.2, skip)
                        .unwrap();
                    assert_eq!(
                        (fa.text, fa.prefix_mismatch),
                        (fb.text, fb.prefix_mismatch),
                        "finish of {seed:?} + {ids:?}"
                    );
                }
            }
            let with: Arc<dyn TokenizerTrait> = Arc::new(with);
            let without: Arc<dyn TokenizerTrait> = Arc::new(without);
            for round in 0..300usize {
                let ids = random_ids_without_lost_runs(&mut state, 1 + round % 30);
                let mut a = Sequence::new_with_options(Arc::clone(&with), skip);
                let mut b = Sequence::new_with_options(Arc::clone(&without), skip);
                for &id in &ids {
                    assert_eq!(
                        a.append_token(id).unwrap(),
                        b.append_token(id).unwrap(),
                        "{ids:?} at {id} strip {strip} skip {skip}"
                    );
                }
                assert_eq!(
                    a.finish().unwrap(),
                    b.finish().unwrap(),
                    "finish of {ids:?}"
                );
                assert_eq!(
                    a.prefix_mismatches(),
                    0,
                    "{ids:?}: no lost run, no recovery"
                );
            }
        }
    }
}

fn thinking_kwargs(value: bool) -> HashMap<String, serde_json::Value> {
    HashMap::from([("thinking".to_string(), serde_json::Value::Bool(value))])
}

/// A Llama-style checkpoint as a legacy conversion writes it: a
/// single-character vocabulary with byte fallback, the `Prepend` +
/// `Replace` normalizer and no pre-tokenizer, two special tokens matched
/// raw and a marker added as a normalized token.
const LLAMA_STYLE_TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": [
        {"id": 3, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 4, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 5, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}],
    "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
    "pre_tokenizer": null, "post_processor": null,
    "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
    "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
        "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<|im_start|>": 3, "<|im_end|>": 4, "</think>": 5, "▁": 6,
            "a": 7, "b": 8, "<": 9, "/": 10, ">": 11, "t": 12, "h": 13, "i": 14, "n": 15, "k": 16, "<0x0A>": 17},
        "merges": []}
}"#;

/// Writes [`LLAMA_STYLE_TOKENIZER_JSON`] with a `tokenizer_config.json`
/// naming `class` and the two flags (absent when `None`) and a
/// `config.json` of `model_type`; returns the tokenizer file's path.
fn llama_style_checkpoint(
    dir: &std::path::Path,
    class: &str,
    legacy: Option<bool>,
    add_prefix_space: Option<bool>,
    model_type: &str,
) -> String {
    let tokenizer_path = dir.join("tokenizer.json");
    fs::write(&tokenizer_path, LLAMA_STYLE_TOKENIZER_JSON).unwrap();
    let mut config = json!({
        "tokenizer_class": class, "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "add_bos_token": false, "add_eos_token": false, "clean_up_tokenization_spaces": false,
    });
    if let Some(legacy) = legacy {
        config["legacy"] = json!(legacy);
    }
    if let Some(add_prefix_space) = add_prefix_space {
        config["add_prefix_space"] = json!(add_prefix_space);
    }
    fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    fs::write(
        dir.join("config.json"),
        json!({"model_type": model_type, "architectures": ["LlamaForCausalLM"]}).to_string(),
    )
    .unwrap();
    tokenizer_path.to_string_lossy().into_owned()
}

fn ids_of(tokenizer: &HuggingFaceTokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).unwrap().token_ids().to_vec()
}

/// The expected ids below are the reference class's (transformers 5.17
/// `LlamaTokenizer` over this checkpoint, which loads it with no
/// normalizer and `Metaspace(prepend_scheme=first, split=False)`); the
/// file's own pipeline gives the ids in the comments.
#[test]
fn a_llama_class_checkpoint_encodes_as_its_reference_class_not_as_its_file() {
    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(
        temp.path(),
        "LlamaTokenizer",
        Some(false),
        Some(true),
        "llama",
    );
    let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert!(tokenizer.tokenizers_pipeline().get_normalizer().is_none());
    // A space before a normalized marker is its own piece and the marker
    // is matched (file: [6, 7, 5, 6, 8], the marker's pattern took the
    // space).
    assert_eq!(ids_of(&tokenizer, "a </think> b"), [6, 7, 6, 5, 6, 8]);
    // The marker is matched after a newline and between letters too
    // (file: the marker's characters, [.., 9, 10, 12, 13, 14, 15, 16, 11]).
    assert_eq!(ids_of(&tokenizer, "a\n</think>"), [6, 7, 17, 5]);
    assert_eq!(ids_of(&tokenizer, "a</think>b"), [6, 7, 5, 8]);
    // Text after a special token is not prefixed with the word marker
    // (file: [3, 6, 7]); text at the start is.
    assert_eq!(ids_of(&tokenizer, "<|im_start|>a"), [3, 7]);
    assert_eq!(ids_of(&tokenizer, "<|im_start|> b"), [3, 6, 8]);
    assert_eq!(ids_of(&tokenizer, "ab<|im_end|>"), [6, 7, 8, 4]);
    // A text that starts with a space gets one word marker, not two
    // (file: [6, 6, 7]).
    assert_eq!(ids_of(&tokenizer, " a"), [6, 7]);
    assert_eq!(ids_of(&tokenizer, "a  b"), [6, 7, 6, 6, 8]);
    assert_eq!(ids_of(&tokenizer, ""), [0u32; 0]);
    // The marker decodes as its content: the loader keeps no normalized
    // form of it once the normalizer is gone.
    assert_eq!(
        tokenizer.decode(&[7, 6, 5, 6, 8], false).unwrap(),
        "a </think> b"
    );
}

/// `legacy: true` prepends the word marker before every split, as the
/// file does, and `add_prefix_space: false` nowhere; the marker's match
/// and the space before it are the class's in both. The `Fast` spelling
/// of the class and absent flags read as the class's defaults.
#[test]
fn the_llama_class_flags_pick_the_prepend_scheme() {
    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(
        temp.path(),
        "LlamaTokenizer",
        Some(true),
        Some(true),
        "llama",
    );
    let always = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_eq!(ids_of(&always, "a </think> b"), [6, 7, 6, 5, 6, 8]);
    assert_eq!(ids_of(&always, "<|im_start|>a"), [3, 6, 7]);
    assert_eq!(ids_of(&always, "a</think>b"), [6, 7, 5, 6, 8]);
    assert_eq!(ids_of(&always, " a"), [6, 7]);

    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(
        temp.path(),
        "LlamaTokenizer",
        Some(false),
        Some(false),
        "llama",
    );
    let never = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_eq!(ids_of(&never, "a </think> b"), [7, 6, 5, 6, 8]);
    assert_eq!(ids_of(&never, "<|im_start|>a"), [3, 7]);
    assert_eq!(ids_of(&never, "ab<|im_end|>"), [7, 8, 4]);
    assert_eq!(ids_of(&never, " a"), [6, 7]);
    // The class's decoder strips no leading space where no marker is
    // prepended, so a text that starts with a space round-trips; the
    // file's chain (kept before) ended in the `Strip` and gave `a`. With
    // the marker prepended the one space goes, as the class's does.
    assert_eq!(never.decode(&[6, 7], false).unwrap(), " a");
    assert_eq!(
        never.decode(&[7, 6, 5, 6, 8], false).unwrap(),
        "a </think> b"
    );
    assert_eq!(always.decode(&[6, 7], false).unwrap(), "a");

    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(temp.path(), "LlamaTokenizerFast", None, None, "llama");
    let first = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_eq!(ids_of(&first, "a </think> b"), [6, 7, 6, 5, 6, 8]);
    assert_eq!(ids_of(&first, "<|im_start|>a"), [3, 7]);
}

/// A checkpoint of a model type the reference reads as its file (`phi3`)
/// keeps its file's pipeline, and so does one whose configuration names
/// another class or is missing.
#[test]
fn other_checkpoints_keep_their_file_pipeline() {
    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(
        temp.path(),
        "LlamaTokenizer",
        Some(false),
        Some(true),
        "phi3",
    );
    let other_model_type = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert!(other_model_type
        .tokenizers_pipeline()
        .get_normalizer()
        .is_some());
    assert_eq!(ids_of(&other_model_type, "a </think> b"), [6, 7, 5, 6, 8]);
    assert_eq!(ids_of(&other_model_type, "<|im_start|>a"), [3, 6, 7]);

    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(temp.path(), "PreTrainedTokenizerFast", None, None, "llama");
    let other_class = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_eq!(ids_of(&other_class, "<|im_start|>a"), [3, 6, 7]);

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("tokenizer.json");
    fs::write(&path, LLAMA_STYLE_TOKENIZER_JSON).unwrap();
    let no_configuration = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    assert_eq!(ids_of(&no_configuration, "<|im_start|>a"), [3, 6, 7]);
}

/// The table the gate reads is searched by `binary_search`, so it stays
/// strictly sorted in byte order and without repeats; a refresh that
/// misplaces an entry fails here instead of falling through to the class
/// pipeline.
#[test]
fn the_reference_s_file_table_is_strictly_sorted() {
    assert!(super::REFERENCE_READS_AS_FILE.is_sorted_by(|a, b| a < b));
}

/// A model type the reference leaves to the named class (`minicpm`,
/// `zamba2`, or one it has no tokenizer class of its own for) is the
/// class's too, with the scheme the flags select; a type the reference
/// reads as its file (`mistral`, `granite`, `llava`) keeps the file.
#[test]
fn the_reference_s_model_type_table_decides_between_the_class_and_the_file() {
    for model_type in ["minicpm", "zamba2", "a_type_the_reference_has_no_class_for"] {
        let temp = TempDir::new().unwrap();
        let path =
            llama_style_checkpoint(temp.path(), "LlamaTokenizer", Some(true), None, model_type);
        let class = HuggingFaceTokenizer::from_file(&path).unwrap();
        assert!(
            class.tokenizers_pipeline().get_normalizer().is_none(),
            "{model_type}"
        );
        // `legacy: true`: the marker before every split (file: [6, 7, 5, 6, 8]);
        // a text starting with a space gets one marker, not two (file: [6, 6, 7]).
        assert_eq!(
            ids_of(&class, "a </think> b"),
            [6, 7, 6, 5, 6, 8],
            "{model_type}"
        );
        assert_eq!(ids_of(&class, " a"), [6, 7], "{model_type}");
    }
    for model_type in ["mistral", "granite", "llava"] {
        let temp = TempDir::new().unwrap();
        let path =
            llama_style_checkpoint(temp.path(), "LlamaTokenizer", Some(true), None, model_type);
        let file = HuggingFaceTokenizer::from_file(&path).unwrap();
        assert!(
            file.tokenizers_pipeline().get_normalizer().is_some(),
            "{model_type}"
        );
        assert_eq!(
            ids_of(&file, "a </think> b"),
            [6, 7, 5, 6, 8],
            "{model_type}"
        );
        assert_eq!(ids_of(&file, " a"), [6, 6, 7], "{model_type}");
    }
}

/// A Llama-class file whose normalizer replaces something other than the
/// space is not the legacy conversion: it keeps its own pipeline, so the
/// replacement it names is applied (`ba` reads `▁b▁`, not the class's
/// `▁ba`), as the reference applies the file as written.
#[test]
fn a_llama_class_file_replacing_more_than_the_space_keeps_its_own_pipeline() {
    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(temp.path(), "LlamaTokenizer", None, None, "llama");
    fs::write(
        &path,
        LLAMA_STYLE_TOKENIZER_JSON.replace(
            r#"{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}"#,
            r#"{"type": "Replace", "pattern": {"String": "a"}, "content": "▁"}"#,
        ),
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_eq!(ids_of(&tokenizer, "ba"), [6, 8, 6]);
    assert!(tokenizer.core.tokenizer.get_normalizer().is_some());
}

/// A prompt rendered by segments encodes as its flat text does: the plain
/// region after a control is not prefixed with the word marker, the one
/// at the start of the prompt is.
#[test]
fn a_llama_class_prompt_rendered_by_segments_agrees_with_its_flat_encode() {
    use crate::encoders::PromptSegment;
    let temp = TempDir::new().unwrap();
    let path = llama_style_checkpoint(
        temp.path(),
        "LlamaTokenizer",
        Some(false),
        Some(true),
        "llama",
    );
    let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
    let after_control = [
        PromptSegment::control("<|im_start|>"),
        PromptSegment::text("a b"),
    ];
    assert_eq!(
        tokenizer.core.encode_segments(&after_control).unwrap(),
        [3, 7, 6, 8]
    );
    assert_eq!(ids_of(&tokenizer, "<|im_start|>a b"), [3, 7, 6, 8]);
    let at_start = [
        PromptSegment::text("a b"),
        PromptSegment::control("<|im_end|>"),
    ];
    assert_eq!(
        tokenizer.core.encode_segments(&at_start).unwrap(),
        [6, 7, 6, 8, 4]
    );
    assert_eq!(ids_of(&tokenizer, "a b<|im_end|>"), [6, 7, 6, 8, 4]);
    let two_regions = [
        PromptSegment::text(" a"),
        PromptSegment::control("<|im_end|>"),
        PromptSegment::text(" b"),
    ];
    assert_eq!(
        tokenizer.core.encode_segments(&two_regions).unwrap(),
        [6, 7, 4, 6, 8]
    );
    assert_eq!(ids_of(&tokenizer, " a<|im_end|> b"), [6, 7, 4, 6, 8]);
}

/// A byte-fallback fixture vocabulary as a legacy conversion writes it:
/// the `Prepend` + `Replace` normalizer, no pre-tokenizer, the template
/// post-processor, and the pieces of the text `> <s> </s> <|` under the
/// ids a 32k byte-fallback vocabulary gives them, so that the ids below
/// are the reference's on that text.
const BYTE_FALLBACK_TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
    "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
    "pre_tokenizer": null,
    "post_processor": {"type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<s>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
    "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
    "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
        "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "▁▁": 259, "▁<": 529, "▁>": 1405, "▁": 29871,
            "<": 29966, ">": 29958, "|": 29989},
        "merges": ["▁ <", "▁ ▁", "▁ >"]}
}"#;

/// The reference picks the class from `tokenizer_config.json` before it
/// reads any `config.json`, so a checkpoint shipped as the tokenizer's
/// files alone is the class's too: after a special token spelled in the
/// text, the class prepends no word marker where the file's normalizer
/// prepends one (the file: `[1405, 29871, 1, 259, 2, 29871, 529, 29989]`,
/// one id more). A `config.json` that says no model type reads the same.
#[test]
fn a_llama_class_checkpoint_shipped_without_config_json_encodes_as_its_class() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("tokenizer.json");
    fs::write(&path, BYTE_FALLBACK_TOKENIZER_JSON).unwrap();
    fs::write(
        temp.path().join("tokenizer_config.json"),
        json!({
            "tokenizer_class": "LlamaTokenizer", "legacy": false, "bos_token": "<s>",
            "eos_token": "</s>", "unk_token": "<unk>", "clean_up_tokenization_spaces": false,
        })
        .to_string(),
    )
    .unwrap();
    let text = "> <s> </s> <|";
    let class = [1405, 29871, 1, 29871, 2, 529, 29989];
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    assert!(tokenizer.tokenizers_pipeline().get_normalizer().is_none());
    assert_eq!(ids_of(&tokenizer, text), class);

    fs::write(
        temp.path().join("config.json"),
        json!({"architectures": ["LlamaForCausalLM"]}).to_string(),
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    assert_eq!(ids_of(&tokenizer, text), class);
}

#[test]
fn response_template_is_retained_from_tokenizer_config() {
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    let expected = json!({
        "defaults": {"thinking": "", "content": "", "tool_calls": []},
        "start_anchor_pattern": "anchor",
        "fields": {"sentinel": "raw-value-is-not-rewritten"}
    });
    fs::write(
        temp.path().join("tokenizer_config.json"),
        json!({"response_template": expected}).to_string(),
    )
    .unwrap();

    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(tokenizer.response_template(), Some(&expected));
}

#[test]
fn tokenizer_without_response_template_reports_none() {
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    fs::write(
        temp.path().join("tokenizer_config.json"),
        json!({"chat_template": "{{ messages }}"}).to_string(),
    )
    .unwrap();

    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(tokenizer.response_template(), None);
}

/// `tokenizer.json`'s `truncation` and `padding` blocks are the defaults
/// the engine's tokenizer keeps for a call that asks; a plain encode runs
/// with both off. The loader clears them: a prompt longer than
/// `max_length` encodes whole on the direct path and on the `tokenizers`
/// path alike, and the direct path takes a file whose only obstacle was
/// the block.
#[test]
fn the_files_truncation_and_padding_blocks_do_not_apply_to_an_encode() {
    let twelve_words = "a b a b a b a b a b a b";
    let byte_level_ids = [0, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4];
    let word_level_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let truncation = json!({
        "direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0
    });
    let padding = json!({
        "strategy": {"Fixed": 16}, "direction": "Right", "pad_to_multiple_of": null,
        "pad_id": 1, "pad_type_id": 0, "pad_token": "b"
    });
    for (block, params) in [("truncation", &truncation), ("padding", &padding)] {
        for (file, expected, direct) in [
            (BYTE_LEVEL_TOKENIZER_JSON, &byte_level_ids[..], true),
            (WORD_LEVEL_TOKENIZER_JSON, &word_level_ids[..], false),
        ] {
            let temp = TempDir::new().unwrap();
            let tokenizer_path = temp.path().join("tokenizer.json");
            let mut definition: serde_json::Value = serde_json::from_str(file).unwrap();
            definition[block] = params.clone();
            fs::write(&tokenizer_path, definition.to_string()).unwrap();

            let tokenizer =
                HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
            assert_eq!(
                tokenizer.encode_path() == EncodePath::Tokenizers,
                !direct,
                "a {block} block: the encode path"
            );
            let ids = tokenizer.encode(twelve_words, false).unwrap();
            assert_eq!(
                ids.token_ids(),
                expected,
                "a {block} block, direct path {direct}"
            );
        }
    }
}

/// A checkpoint's own tokenizer gives the letter one of the two ids per
/// process; the loader settles the pair, so every process gives the
/// letter, and any text whose normalized form is the letter, the lowest
/// id, and the compatibility form spelled as such its own.
#[test]
fn two_added_tokens_with_one_normalized_spelling_encode_the_same_in_every_process() {
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, NORMALIZED_TIE_TOKENIZER_JSON).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(tokenizer.encode_path(), EncodePath::Tokenizers);
    for (text, ids) in [
        ("\u{5d0}", &[3][..]),
        ("\u{fb21}", &[4][..]),
        ("a \u{fb21} \u{5d0} b", &[0, 4, 3, 1][..]),
    ] {
        assert_eq!(
            tokenizer.encode(text, false).unwrap().token_ids(),
            ids,
            "{text:?}"
        );
    }
}

/// The letter keeps the normalized match whichever id the pair gave it,
/// so plain text reads as before, longer normalized tokens included; the
/// compatibility form is matched by its own spelling, before
/// normalization, so a longer normalized token no longer spans it (the
/// one reading that changes beyond the choice of id).
#[test]
fn the_letter_keeps_the_normalized_match_whichever_id_is_lower() {
    // The compatibility form at the lower id, and a two-letter normalized
    // token that holds the letter.
    let definition = r#"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 3, "content": "\ufb21", "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": true, "special": false},
            {"id": 4, "content": "\u05d0", "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": true, "special": false},
            {"id": 5, "content": "\u05d0\u05d1", "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": true, "special": false}
        ],
        "normalizer": {"type": "NFKC"},
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "<unk>": 2}, "unk_token": "<unk>"}
    }"#;
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, definition).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    for (text, ids) in [
        ("\u{5d0}", &[4][..]),
        ("\u{5d0}\u{5d1}", &[5][..]),
        ("\u{fb21}", &[3][..]),
        ("\u{fb21}\u{5d1}", &[3, 2][..]),
    ] {
        assert_eq!(
            tokenizer.encode(text, false).unwrap().token_ids(),
            ids,
            "{text:?}"
        );
    }
}

/// Two compatibility forms of a letter the vocabulary has no token for,
/// so no member spells the normalized form: the lowest id keeps the
/// normalized match, as the fallback.
#[test]
fn the_lowest_id_keeps_the_normalized_match_when_no_member_spells_the_form() {
    let definition = r#"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 3, "content": "\uff46\uff49", "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": true, "special": false},
            {"id": 4, "content": "\ufb01", "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": true, "special": false}
        ],
        "normalizer": {"type": "NFKC"},
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "<unk>": 2}, "unk_token": "<unk>"}
    }"#;
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, definition).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    for (text, ids) in [
        ("fi", &[3][..]),
        ("\u{ff46}\u{ff49}", &[3][..]),
        ("\u{fb01}", &[4][..]),
    ] {
        assert_eq!(
            tokenizer.encode(text, false).unwrap().token_ids(),
            ids,
            "{text:?}"
        );
    }
}

/// Settling re-adds the compatibility form of a pair matched on the raw
/// text and nothing else: the ids decode as before, a second pass finds
/// nothing to settle, and a vocabulary without such a pair is not touched.
#[test]
fn settling_a_normalized_tie_keeps_decoding_and_leaves_other_vocabularies_alone() {
    let mut tokenizer = tokenizers::Tokenizer::from_bytes(NORMALIZED_TIE_TOKENIZER_JSON).unwrap();
    let decoded = |tokenizer: &tokenizers::Tokenizer| -> Vec<String> {
        [3, 4]
            .iter()
            .map(|&id| tokenizer.decode(&[id], false).unwrap())
            .collect()
    };
    let before = decoded(&tokenizer);
    assert_eq!(
        HuggingFaceTokenizer::settle_normalized_added_token_ties(&mut tokenizer).unwrap(),
        1
    );
    assert_eq!(decoded(&tokenizer), before);
    let added = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    assert!(
        added.get(&3).unwrap().normalized,
        "the letter keeps the normalized match"
    );
    assert!(
        !added.get(&4).unwrap().normalized,
        "the compatibility form is matched on the raw text"
    );
    assert_eq!(
        HuggingFaceTokenizer::settle_normalized_added_token_ties(&mut tokenizer).unwrap(),
        0
    );
    let mut plain = tokenizers::Tokenizer::from_bytes(WORD_LEVEL_TOKENIZER_JSON).unwrap();
    assert_eq!(
        HuggingFaceTokenizer::settle_normalized_added_token_ties(&mut plain).unwrap(),
        0
    );
}

/// The DeepSeek V4 encoder port stands in for the encoder module those
/// checkpoints ship in place of a chat template. A checkpoint of the
/// architecture that ships a template of its own renders through the
/// template, as the engine's own renderer does.
#[test]
fn a_v4_architecture_checkpoint_with_its_own_template_renders_through_it() {
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    fs::write(
        temp.path().join("config.json"),
        json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})
            .to_string(),
    )
    .unwrap();
    assert!(matches!(
        detect_renderer_from_config(temp.path(), false),
        Renderer::DeepseekV4
    ));
    assert!(matches!(
        detect_renderer_from_config(temp.path(), true),
        Renderer::Jinja
    ));

    fs::write(
        temp.path().join("chat_template.jinja"),
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}",
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(
        tokenizer
            .apply_chat_template(
                &[json!({"role": "user", "content": "Hi"})],
                ChatTemplateParams::default()
            )
            .unwrap(),
        "<user>Hi"
    );
}

/// The loader reads a configuration's named templates: `default` renders
/// a plain request, `tool_use` one that declares tools; a template file
/// beside the checkpoint (or the operator's) stands alone for both.
#[test]
fn the_loader_reads_a_named_template_list_and_a_template_file_stands_alone() {
    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    fs::write(
        temp.path().join("tokenizer_config.json"),
        json!({"chat_template": [
            {"name": "default", "template": "d"},
            {"name": "tool_use", "template": "t"}
        ]})
        .to_string(),
    )
    .unwrap();
    let messages = [json!({"role": "user", "content": "Hi"})];
    let tools: [serde_json::Value; 0] = [];
    let with_tools = || ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };

    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(
        tokenizer
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap(),
        "d"
    );
    assert_eq!(
        tokenizer
            .apply_chat_template(&messages, with_tools())
            .unwrap(),
        "t"
    );

    let file = temp.path().join("own.jinja");
    fs::write(&file, "f").unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_path.to_str().unwrap(),
        file.to_str(),
    )
    .unwrap();
    assert_eq!(
        tokenizer
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap(),
        "f"
    );
    assert_eq!(
        tokenizer
            .apply_chat_template(&messages, with_tools())
            .unwrap(),
        "f"
    );
}

/// A multimodal checkpoint may ship the processor's legacy
/// `chat_template.json` beside the tokenizer's template, and the two can
/// differ: the engine's server renders a request without tools through
/// the processor's template and one with tools through the tokenizer's,
/// and so does the loader, the content shape following the template
/// selected. The checkpoint's own `chat_template.jinja` is the
/// tokenizer's template and keeps the processor's for the tool-less
/// requests; equal templates, a processor file beside a configuration
/// without a template, an operator's template file and a replacement
/// template change nothing.
#[test]
fn the_processor_s_template_renders_a_request_without_tools() {
    use crate::chat_template::ChatTemplateContentFormat;

    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    let config_path = temp.path().join("tokenizer_config.json");
    let processor_path = temp.path().join("chat_template.json");
    let jinja_path = temp.path().join("chat_template.jinja");
    let write = |path: &std::path::Path, template: &str| {
        fs::write(path, json!({"chat_template": template}).to_string()).unwrap();
    };
    let string_template =
        "tokenizer:{% for message in messages %}{{ message['content'] }}{% endfor %}";
    let parts_template = "processor:{% for message in messages %}\
         {% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endfor %}";
    let messages = [json!({"role": "user", "content": "Hi"})];
    let tools: [serde_json::Value; 0] = [];
    let with_tools = || ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    let load = || HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    let render = |tokenizer: &HuggingFaceTokenizer, params: ChatTemplateParams| {
        tokenizer.apply_chat_template(&messages, params).unwrap()
    };

    // Both, differing: the processor's for a request without tools, with
    // the content shape of that template; the tokenizer's with tools.
    write(&config_path, string_template);
    write(&processor_path, parts_template);
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "processor:Hi"
    );
    assert_eq!(render(&tokenizer, with_tools()), "tokenizer:Hi");
    assert_eq!(
        tokenizer.chat_template_content_format_for_request(false),
        ChatTemplateContentFormat::OpenAI
    );
    assert_eq!(
        tokenizer.chat_template_content_format_for_request(true),
        ChatTemplateContentFormat::String
    );
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::String
    );

    // The checkpoint's own `chat_template.jinja` is the tokenizer's
    // template (the discovery prefers it): the processor's still renders
    // the tool-less requests.
    fs::write(&jinja_path, "jinja:{{ messages[0]['content'] }}").unwrap();
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "processor:Hi"
    );
    assert_eq!(render(&tokenizer, with_tools()), "jinja:Hi");
    fs::remove_file(&jinja_path).unwrap();

    // Equal templates: one template, whatever the request declares.
    write(&processor_path, string_template);
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "tokenizer:Hi"
    );
    assert_eq!(render(&tokenizer, with_tools()), "tokenizer:Hi");
    assert_eq!(
        tokenizer.chat_template_content_format_for_request(false),
        tokenizer.chat_template_content_format()
    );

    // The processor's file beside a configuration without a template is
    // the one template of the checkpoint, as before.
    write(&processor_path, parts_template);
    fs::write(
        &config_path,
        json!({"tokenizer_class": "PreTrainedTokenizerFast"}).to_string(),
    )
    .unwrap();
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "processor:Hi"
    );
    assert_eq!(render(&tokenizer, with_tools()), "processor:Hi");

    // An operator's template file stands alone, as the request's own
    // template outranks the processor's in the engine's server; so does
    // a replacement template.
    write(&config_path, string_template);
    let own = temp.path().join("own.jinja");
    fs::write(&own, "own:{{ messages[0]['content'] }}").unwrap();
    let mut tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_path.to_str().unwrap(),
        own.to_str(),
    )
    .unwrap();
    assert_eq!(render(&tokenizer, ChatTemplateParams::default()), "own:Hi");
    assert_eq!(render(&tokenizer, with_tools()), "own:Hi");
    let mut tokenizer_with_both = load();
    assert_eq!(
        render(&tokenizer_with_both, ChatTemplateParams::default()),
        "processor:Hi"
    );
    tokenizer_with_both
        .set_chat_template("alone:{{ messages[0]['content'] }}".to_string())
        .unwrap();
    assert_eq!(
        render(&tokenizer_with_both, ChatTemplateParams::default()),
        "alone:Hi"
    );
    assert_eq!(render(&tokenizer_with_both, with_tools()), "alone:Hi");
    tokenizer
        .set_chat_template("alone:{{ messages[0]['content'] }}".to_string())
        .unwrap();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "alone:Hi"
    );

    // A processor file that does not parse, or a processor template that
    // does not compile, is no processor template: the tokenizer loads and
    // its own template serves every request.
    fs::write(&processor_path, "{not json").unwrap();
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "tokenizer:Hi"
    );
    write(&processor_path, "{{ messages[0]['content'] ");
    let tokenizer = load();
    assert_eq!(
        render(&tokenizer, ChatTemplateParams::default()),
        "tokenizer:Hi"
    );
    assert_eq!(render(&tokenizer, with_tools()), "tokenizer:Hi");
}

// Regression: DeepSeek V3.2/V4 bypass ChatTemplateState::apply, so the
// resolved `params.thinking` (from reasoning_effort / Anthropic ThinkingConfig)
// must be honored here — with an explicit `template_kwargs["thinking"]` still
// winning. Same precedence as the Jinja path.
#[test]
fn derive_thinking_mode_honors_params_thinking_and_explicit_override() {
    // No signal at all -> Chat (default off).
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams::default()),
        ThinkingMode::Chat
    ));

    // params.thinking is the fallback when there is no explicit kwarg.
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            thinking: Some(true),
            ..Default::default()
        }),
        ThinkingMode::Thinking
    ));
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            thinking: Some(false),
            ..Default::default()
        }),
        ThinkingMode::Chat
    ));

    // An explicit template_kwargs["thinking"] wins over params.thinking.
    let on = thinking_kwargs(true);
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            thinking: Some(false),
            template_kwargs: Some(&on),
            ..Default::default()
        }),
        ThinkingMode::Thinking
    ));
    let off = thinking_kwargs(false);
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            thinking: Some(true),
            template_kwargs: Some(&off),
            ..Default::default()
        }),
        ThinkingMode::Chat
    ));
}

/// The engine's tokenizer wrapper switches V3.2 thinking on from the
/// `thinking` kwarg or its `enable_thinking` alias; the shim reads the
/// alias too, and the alias, like the key, outranks the projected
/// `params.thinking`. Deliberate divergence from the engine's wrapper,
/// which ORs the two keys: `thinking` wins here when both are present,
/// the order the gateway arms the reasoning parser in (as in the V4
/// shim), so a contradictory pair renders chat mode here where the
/// engine would think, and the prompt and the parser agree. (That the
/// profile declares the alias is checked through the detected renderer
/// in the render tests.)
#[test]
fn derive_thinking_mode_reads_the_enable_thinking_alias() {
    let alias = |value: bool| {
        HashMap::from([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(value),
        )])
    };
    let on = alias(true);
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            template_kwargs: Some(&on),
            ..Default::default()
        }),
        ThinkingMode::Thinking
    ));
    let off = alias(false);
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            thinking: Some(true),
            template_kwargs: Some(&off),
            ..Default::default()
        }),
        ThinkingMode::Chat
    ));
    let both = HashMap::from([
        ("thinking".to_string(), serde_json::Value::Bool(false)),
        ("enable_thinking".to_string(), serde_json::Value::Bool(true)),
    ]);
    assert!(matches!(
        derive_thinking_mode(&ChatTemplateParams {
            template_kwargs: Some(&both),
            ..Default::default()
        }),
        ThinkingMode::Chat
    ));
}

/// A string-format template reads each content as the string it is, and
/// the owned entry wraps nothing for it, as the borrowed one wraps nothing:
/// the two render one prompt with the strings in it. A pre-wrap under every
/// template would hand this one a part list in place of each string.
#[test]
fn an_owned_render_of_a_string_format_template_leaves_the_strings_alone() {
    use crate::chat_template::ChatTemplateContentFormat;

    let temp = TempDir::new().unwrap();
    let tokenizer_path = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON).unwrap();
    let mut tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    tokenizer
        .set_chat_template(
            "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
                .to_string(),
        )
        .unwrap();
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::String
    );

    let messages = vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    let borrowed = tokenizer
        .apply_chat_template_with_encoding(&messages, ChatTemplateParams::default(), None)
        .unwrap();
    let owned = tokenizer
        .apply_chat_template_with_encoding_owned(messages, ChatTemplateParams::default(), None)
        .unwrap();
    assert_eq!(owned.text, borrowed.text, "the same prompt either way");
    assert_eq!(borrowed.text, "system:Be brief.\nuser:Hi\n");
}
