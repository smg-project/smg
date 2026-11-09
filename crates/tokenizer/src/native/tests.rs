use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, byte_level::ByteLevel as ByteLevelDecoder, fuse::Fuse,
        sequence::Sequence as DecoderSequence, strip::Strip,
    },
    models::{
        bpe::{Merges, Vocab, BPE},
        wordlevel::WordLevel,
    },
    normalizers::{
        unicode::{NFC, NFKC},
        NormalizerWrapper, Replace,
    },
    pre_tokenizers::{
        byte_level::ByteLevel as ByteLevelPreTokenizer,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::{
        bert::BertProcessing,
        byte_level::ByteLevel as ByteLevelProcessor,
        sequence::Sequence as ProcessorSequence,
        template::{SpecialToken, TemplateProcessing},
        PostProcessorWrapper,
    },
    AddedToken, AddedVocabulary, DecoderWrapper, SplitDelimiterBehavior, Tokenizer as HfTokenizer,
};
use unicode_normalization_alignments::{is_nfc_quick, IsNormalized};

use super::{
    sentencepiece::{MERGE_SCRATCH_SYMBOLS, WORD_MARKER},
    short_pieces::{ShortPieceKey, ShortPieceTable, MIN_SLOTS},
    *,
};
use crate::{
    encode_path::{encode_fallbacks, EncodeFallbacks, EncodePath},
    huggingface::HuggingFaceTokenizer,
    kimi_k2_tokenizer::KIMI_K2_PATTERN,
};

const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DIGITS: &str = r"\p{N}{1,3}";
const CJK: &str = "[一-龥぀-ゟ゠-ヿ]+";
const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The cased template without the clitic group and with one digit at a
/// time.
const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const POOL: &[char] = &[
    'a',
    'b',
    'e',
    'd',
    'l',
    'm',
    'r',
    's',
    't',
    'v',
    'A',
    'E',
    'L',
    'S',
    'T',
    'Z',
    'ſ',
    'ß',
    'é',
    'Ö',
    'ǅ',
    'ʰ',
    'ª',
    '0',
    '1',
    '9',
    '٣',
    '๒',
    '²',
    '½',
    ' ',
    '\t',
    '\n',
    '\r',
    '\u{0B}',
    '\u{85}',
    '\u{A0}',
    '\u{2003}',
    '\u{3000}',
    '\'',
    '"',
    ',',
    '.',
    '-',
    '/',
    '_',
    '~',
    '[',
    ']',
    '\\',
    '^',
    '`',
    '{',
    '|',
    '}',
    '<',
    '>',
    '!',
    '(',
    ')',
    '€',
    '→',
    '\u{301}',
    '\u{93E}',
    '\u{E31}',
    '\u{E48}',
    '\u{9BE}',
    'ก',
    'ข',
    'ा',
    'क',
    'ب',
    'я',
    'Я',
    'α',
    'Ω',
    '中',
    '文',
    '龥',
    'あ',
    'ん',
    'ア',
    'ー',
    '한',
    '😀',
    '👍',
    '\u{1F3FD}',
    '\u{200D}',
    '\u{FE0F}',
    '\u{FFFD}',
    '\u{0}',
    '\u{7F}',
];

fn pseudo_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Random text from the pool, with added tokens dropped in now and then
/// (whole, and cut so only a prefix of one appears).
fn random_text(state: &mut u64, len: usize, added: &[&str]) -> String {
    let mut text = String::new();
    for _ in 0..len {
        let roll = pseudo_random(state);
        if !added.is_empty() && roll.is_multiple_of(11) {
            let token = added[(roll / 11) as usize % added.len()];
            if roll.is_multiple_of(3) {
                let half = token
                    .char_indices()
                    .nth(token.chars().count() / 2)
                    .map_or(token.len(), |(i, _)| i);
                text.push_str(&token[..half]);
            } else {
                text.push_str(token);
            }
        } else {
            text.push(POOL[(roll % POOL.len() as u64) as usize]);
        }
    }
    text
}

/// A byte-level BPE over the whole alphabet with merges for the pairs
/// that occur in a corpus of pool text, so pieces really do merge.
fn model(state: &mut u64, ignore_merges: bool) -> BPE {
    let alphabet: Vec<char> = {
        let table = bytes_char();
        let mut chars: Vec<char> = table.to_vec();
        chars.sort_unstable();
        chars
    };
    let mut vocab = Vocab::default();
    for ch in &alphabet {
        let next = vocab.len() as u32;
        vocab.insert(ch.to_string(), next);
    }
    let mut merges = Merges::new();
    let table = bytes_char();
    let corpus = random_text(state, 4000, &[]);
    let mapped: String = corpus.bytes().map(|b| table[usize::from(b)]).collect();
    let chars: Vec<char> = mapped.chars().collect();
    let mut seen = std::collections::HashSet::new();
    for pair in chars.windows(2) {
        let (a, b) = (pair[0].to_string(), pair[1].to_string());
        if a == " " || b == " " || !seen.insert((a.clone(), b.clone())) {
            continue;
        }
        let merged = format!("{a}{b}");
        if vocab.contains_key(&merged) {
            continue;
        }
        let next = vocab.len() as u32;
        vocab.insert(merged, next);
        merges.push((a, b));
        if merges.len() >= 600 {
            break;
        }
    }
    // A few longer merges on top of the pairs.
    for (a, b) in [
        ("Ġt", "h"),
        ("Ġth", "e"),
        ("ĠT", "he"),
        ("Ã", "©"),
        ("ð", "Ł"),
    ] {
        if vocab.contains_key(a) && vocab.contains_key(b) {
            let merged = format!("{a}{b}");
            if !vocab.contains_key(&merged) {
                let next = vocab.len() as u32;
                vocab.insert(merged, next);
                merges.push((a.to_string(), b.to_string()));
            }
        }
    }
    BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(ignore_merges)
        .build()
        .expect("bpe")
}

fn split(pattern: &str, behavior: SplitDelimiterBehavior, invert: bool) -> PreTokenizerWrapper {
    PreTokenizerWrapper::Split(
        Split::new(SplitPattern::Regex(pattern.to_owned()), behavior, invert).expect("split"),
    )
}

fn byte_level(use_regex: bool) -> PreTokenizerWrapper {
    PreTokenizerWrapper::ByteLevel(
        ByteLevelPreTokenizer::default()
            .add_prefix_space(false)
            .use_regex(use_regex),
    )
}

struct Shape {
    name: &'static str,
    nfc: bool,
    pre_tokenizers: Vec<PreTokenizerWrapper>,
    ignore_merges: bool,
    special: &'static [&'static str],
    added: &'static [(&'static str, bool)],
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "qwen3.5",
            nfc: true,
            pre_tokenizers: vec![
                split(QWEN35, SplitDelimiterBehavior::Isolated, false),
                byte_level(false),
            ],
            ignore_merges: false,
            special: &[
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "<think>",
                "</think>",
            ],
            added: &[("<tool_call>", false), ("</tool_call>", false)],
        },
        Shape {
            name: "cl100k-ignore-merges",
            nfc: false,
            pre_tokenizers: vec![
                split(CL100K, SplitDelimiterBehavior::Isolated, false),
                byte_level(false),
            ],
            ignore_merges: true,
            special: &["<|endoftext|>", "[gMASK]", "<sop>"],
            added: &[],
        },
        Shape {
            name: "three-splits",
            nfc: false,
            pre_tokenizers: vec![
                split(DIGITS, SplitDelimiterBehavior::Isolated, false),
                split(CJK, SplitDelimiterBehavior::Isolated, false),
                split(DEEPSEEK, SplitDelimiterBehavior::Isolated, false),
                byte_level(false),
            ],
            ignore_merges: false,
            special: &[
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<｜User｜>",
            ],
            added: &[("<｜tool▁call▁begin｜>", true), ("<｜fim▁hole｜>", true)],
        },
        Shape {
            name: "inverted-removed",
            nfc: true,
            pre_tokenizers: vec![
                split(CASED, SplitDelimiterBehavior::Removed, true),
                byte_level(false),
            ],
            ignore_merges: false,
            special: &["<|im_start|>", "<|im_end|>", "[e~[", "]~b]"],
            added: &[],
        },
        Shape {
            name: "byte-level-regex",
            nfc: false,
            pre_tokenizers: vec![byte_level(true)],
            ignore_merges: false,
            special: &["<|endoftext|>"],
            added: &[],
        },
        // Added tokens with a space inside, matched after normalization
        // (the default of `add_tokens`): a cut at that space would turn
        // the token into pieces.
        Shape {
            name: "normalized-words",
            nfc: false,
            pre_tokenizers: vec![
                split(CL100K, SplitDelimiterBehavior::Isolated, false),
                byte_level(false),
            ],
            ignore_merges: false,
            special: &["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            added: &[
                ("foo bar", true),
                ("### Instruction", true),
                ("<tool_call>", false),
            ],
        },
    ]
}

fn tokenizer_of(shape: &Shape, state: &mut u64) -> HfTokenizer {
    let mut tokenizer = HfTokenizer::new(model(state, shape.ignore_merges));
    if shape.nfc {
        tokenizer
            .with_normalizer(Some(NormalizerWrapper::NFC(NFC)))
            .expect("normalizer");
    }
    let pre = if shape.pre_tokenizers.len() == 1 {
        shape.pre_tokenizers[0].clone()
    } else {
        PreTokenizerWrapper::Sequence(Sequence::new(shape.pre_tokenizers.clone()))
    };
    tokenizer.with_pre_tokenizer(Some(pre));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer.with_post_processor(Some(ByteLevelProcessor::default().trim_offsets(false)));
    tokenizer
        .add_special_tokens(shape.special.iter().map(|s| AddedToken::from(*s, true)))
        .expect("special tokens");
    tokenizer
        .add_tokens(
            shape
                .added
                .iter()
                .map(|(s, normalized)| AddedToken::from(*s, false).normalized(*normalized)),
        )
        .expect("added tokens");
    tokenizer
}

/// The native encoder of `tokenizer` with its seed read, as the loaders
/// build it.
fn seeded(tokenizer: &HfTokenizer) -> NativeEncoder {
    let mut native = NativeEncoder::from_tokenizer(tokenizer).expect("native path");
    native.seed_vocabulary(tokenizer.get_model());
    native
}

#[test]
fn every_shape_gets_a_native_path() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer);
        assert!(
            native.is_ok(),
            "{}: {:?}",
            shape.name,
            native.as_ref().err()
        );
        let expected_stages = shape
            .pre_tokenizers
            .iter()
            .filter(|p| matches!(p, PreTokenizerWrapper::Split(_)))
            .count()
            + usize::from(shape.name == "byte-level-regex");
        assert_eq!(
            native.expect("native").stages.len(),
            expected_stages,
            "{}",
            shape.name
        );
    }
}

#[test]
fn shapes_outside_the_path_are_declined() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let base = &shapes()[0];
    let declined = |tokenizer: &HfTokenizer| {
        NativeEncoder::from_tokenizer(tokenizer)
            .expect_err("declined")
            .to_string()
    };
    // Prefix space.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        split(QWEN35, SplitDelimiterBehavior::Isolated, false),
        PreTokenizerWrapper::ByteLevel(ByteLevelPreTokenizer::default().add_prefix_space(true)),
    ]))));
    assert_eq!(
        declined(&tokenizer),
        "the pre-tokenizer does not end in ByteLevel without a prefix space"
    );
    // A normalizer other than NFC.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer
        .with_normalizer(Some(NormalizerWrapper::NFKC(NFKC)))
        .expect("normalizer");
    assert_eq!(declined(&tokenizer), "normalizer NFKC (only NFC or none)");
    // A split behaviour the path does not reproduce.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        split(QWEN35, SplitDelimiterBehavior::MergedWithNext, false),
        byte_level(false),
    ]))));
    assert_eq!(
        declined(&tokenizer),
        "Split behavior Some(\"MergedWithNext\") with invert Some(false) (only Isolated, or \
             Removed inverted)"
    );
    // An added token with lstrip.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer
        .add_special_tokens([AddedToken::from("<pad>", true).lstrip(true)])
        .expect("special token");
    assert_eq!(
        declined(&tokenizer),
        "added token \"<pad>\" with single_word, lstrip or rstrip, or without content"
    );
    // A regex outside the subset.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        split(r"\w+|\s+", SplitDelimiterBehavior::Isolated, false),
        byte_level(false),
    ]))));
    assert_eq!(
        declined(&tokenizer),
        r"a split pattern outside the supported regex subset: \w+|\s+"
    );
    // A post-processor of another kind, and a model that is not BPE.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer.with_post_processor(Some(BertProcessing::new(
        ("[SEP]".to_owned(), 1),
        ("[CLS]".to_owned(), 2),
    )));
    assert_eq!(
        declined(&tokenizer),
        "post-processor BertProcessing (only ByteLevel, TemplateProcessing or none)"
    );
    let tokenizer = HfTokenizer::new(WordLevel::default());
    assert_eq!(declined(&tokenizer), "the model is not BPE");
    // No pre-tokenizer at all.
    let mut tokenizer = tokenizer_of(base, &mut state);
    tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
    assert_eq!(
        declined(&tokenizer),
        "no pre-tokenizer (the path needs ByteLevel, after any Split stages)"
    );
}

#[test]
fn a_batch_takes_the_direct_path_input_by_input() {
    use crate::traits::{Encoder, Encoding};
    let mut state = 0x5A5A_1234_5678_9ABCu64;
    for shape in shapes() {
        let tokenizer = HuggingFaceTokenizer::from_tokenizer(tokenizer_of(&shape, &mut state));
        let added: Vec<&str> = shape.special.to_vec();
        let mut texts: Vec<String> = (0..8)
            .map(|round| random_text(&mut state, 1 + round * 5, &added))
            .collect();
        // Under an NFC shape, one input not in NFC sits in the middle of
        // the batch: it is normalized and encoded on the direct path like
        // its neighbours.
        if shape.nfc {
            texts.insert(texts.len() / 2, "e\u{301}".to_string());
        }
        let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let batch = tokenizer.encode_batch(&inputs, false).expect(shape.name);
        assert_eq!(batch.len(), inputs.len(), "{}", shape.name);
        for (input, encoding) in inputs.iter().zip(&batch) {
            let single = tokenizer.encode(input, false).expect("encode");
            assert_eq!(
                encoding.token_ids(),
                single.token_ids(),
                "{}: {input:?}",
                shape.name
            );
            assert!(
                matches!(encoding, Encoding::Plain(_)),
                "{}: the batch left the direct path for {input:?}",
                shape.name
            );
        }
    }
}

/// A text long enough for the pool, with spaces, words, digits, CJK and
/// added tokens, so that stable cuts exist and the units are several.
fn long_text(state: &mut u64, added: &[&str], bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + 64);
    while text.len() < bytes {
        let words = 1 + (pseudo_random(state) % 12) as usize;
        text.push_str(&random_text(state, words, added));
        text.push(' ');
        text.push_str(
            ["word", "Wort", "parole", "palabra", "slovo"][(pseudo_random(state) % 5) as usize],
        );
        if pseudo_random(state).is_multiple_of(7) {
            text.push_str(" 12345 ");
        }
    }
    text
}

#[test]
fn a_long_text_encodes_the_same_on_the_pool_as_in_one_piece() {
    let mut state = 0x5A5A_1234_5678_9ABCu64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let added: Vec<&str> = shape
            .special
            .iter()
            .copied()
            .chain(shape.added.iter().map(|(s, _)| *s))
            .collect();
        let mut compared = 0;
        for _ in 0..3 {
            let text = long_text(&mut state, &added, 3 * PARALLEL_MIN_BYTES);
            let units = native.units(&text, true, true).as_slice().len();
            assert!(
                units >= 4,
                "{}: {units} units for {} bytes",
                shape.name,
                text.len()
            );
            let pooled = native
                .encode_with(Some(tokenizer.get_model()), &text, true, false)
                .expect("pooled");
            let serial = native
                .encode_with(Some(tokenizer.get_model()), &text, false, false)
                .expect("serial");
            compared += 1;
            assert_eq!(pooled, serial, "{}: the pool changed the ids", shape.name);
            let reference = tokenizer
                .encode(text.as_str(), false)
                .expect("encode")
                .get_ids()
                .to_vec();
            assert_eq!(pooled, reference, "{}: against the reference", shape.name);
        }
        assert!(
            compared >= 2,
            "{}: {compared} long texts compared",
            shape.name
        );
    }
}

/// The added-token search through the tokens' prefixes finds what the
/// automaton's unanchored leftmost-longest search finds: the same ranges and
/// the same tokens, on texts made of the tokens, their prefixes, other text
/// and characters sharing a token's first bytes; for a set of ASCII first
/// bytes only (the automaton's own search), one with a token in fullwidth
/// bars among `<` tokens (the chain shape), one of tokens beyond ASCII only,
/// one with more than three first bytes (tab and newline runs, `<`, `[` and
/// runs of a mark beyond ASCII), one whose prefixes stand whole (every
/// token shorter than the prefix length), one the library declines as it
/// stands (one-byte tab and newline tokens among twenty-two prefixes: the
/// tokens' five first bytes stand in), and two sets of added words whose
/// prefixes would be under the floor (they keep the automaton's search).
#[test]
fn added_tokens_are_found_where_the_unanchored_search_finds_them() {
    let sets: [(&[&str], bool); 5] = [
        (&["<a>", "<ab>", "<abc>", "<|x|>", "<"], false),
        (
            &[
                "<a>",
                "<ab>",
                "<abc>",
                "<\u{FF5C}x\u{FF5C}>",
                "<\u{FF5C}xy\u{FF5C}>",
                "\u{FF5C}x\u{FF5C}",
                "</t>",
                "<t>",
            ],
            true,
        ),
        (&["\u{FF5C}x\u{FF5C}", "\u{FF0C}", "\u{4E2D}\u{6587}"], true),
        (
            &[
                "\t\t",
                "\t\t\t",
                "\n\n",
                "\n\n\n",
                "<a>",
                "[b]",
                "\u{2581}\u{2581}",
                "\u{2581}\u{2581}\u{2581}",
            ],
            true,
        ),
        (
            &["<a>", "\u{4E2D}", "\u{0416}a", "\u{00E9}", "e\u{FF5C}"],
            true,
        ),
    ];
    let parts = [
        "<a>",
        "<ab>",
        "<abc>",
        "<|x|>",
        "<",
        "<a",
        "<ab",
        "</t>",
        "<t>",
        "<t",
        "</tx",
        "<u3>",
        "<u1",
        "<\u{FF5C}x",
        "<\u{FF5C}x\u{FF5C}>",
        "<\u{FF5C}xy\u{FF5C}>",
        "\u{FF5C}",
        "\u{FF5C}x",
        "\u{FF5C}x\u{FF5C}",
        "\u{FF0C}",
        "[b]",
        "[",
        "\t",
        "\t\t",
        "\t\t\t\t",
        "\n",
        "\n\n",
        "\n\n\n\n",
        "\u{2581}",
        "\u{2581}\u{2581}",
        "\u{2581}\u{2581}\u{2581}\u{2581}",
        "\u{201C}",
        "\u{4E2D}",
        "\u{4E2D}\u{6587}",
        "\u{6587}",
        "\u{0416}",
        "\u{0416}a",
        "\u{00E9}",
        "e\u{FF5C}",
        "xyz",
        " ",
        "a",
        "e",
        "abing",
        "ab",
        "ing",
        "zaed",
        "ed",
        "qa",
    ];
    // Sets of added words: the prefixes that fit are one byte (70 words with
    // distinct two-byte prefixes) or two bytes (60 pairs sharing one), under
    // the floor, so both keep the automaton's own search.
    let letter = |i: usize| (b'a' + (i % 26) as u8) as char;
    let words_one: Vec<String> = (0..70)
        .map(|i| format!("{}{}ing", letter(i), letter(i / 26)))
        .collect();
    let words_two: Vec<String> = (0..60)
        .flat_map(|i| {
            [
                format!("{}{}ing", letter(i), letter(i / 26)),
                format!("{}{}ed", letter(i), letter(i / 26)),
            ]
        })
        .collect();
    // A set the library declines as it stands (one-byte tab and newline
    // tokens among twenty-two prefixes): the tokens' five first bytes stand
    // in, and the anchored automaton decides at each.
    let tags: Vec<String> = (0..16).map(|i| format!("<u{i}>")).collect();
    let declined: Vec<&str> = ["\t", "\t\t", "\n", "\n\n", "[b]", "\u{2581}\u{2581}"]
        .into_iter()
        .chain(tags.iter().map(String::as_str))
        .collect();
    let mut all: Vec<(Vec<&str>, bool)> = sets
        .iter()
        .map(|(tokens, by_prefixes)| (tokens.to_vec(), *by_prefixes))
        .collect();
    all.push((words_one.iter().map(String::as_str).collect(), false));
    all.push((words_two.iter().map(String::as_str).collect(), false));
    all.push((declined, true));
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for (tokens, by_prefixes) in &all {
        let list: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| ((*token).to_owned(), 1000 + i as u32))
            .collect();
        let added = AddedTokens::build(&list)
            .expect("the tokens form an automaton")
            .expect("a non-empty set");
        assert_eq!(added.prefixes.is_some(), *by_prefixes, "{tokens:?}");
        let reference = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens)
            .expect("the reference automaton");
        for round in 0..3000 {
            let text: String = (0..1 + round % 12)
                .map(|_| parts[(pseudo_random(&mut state) % parts.len() as u64) as usize])
                .collect();
            let expected: Vec<(Range<usize>, usize)> = reference
                .find_iter(&text)
                .map(|found| (found.range(), found.pattern().as_usize()))
                .collect();
            assert_eq!(
                added.matches(&text).collect::<Vec<_>>(),
                expected,
                "{tokens:?} on {text:?}"
            );
            assert!(
                added
                    .ranges(&text)
                    .eq(expected.iter().map(|(range, _)| range.clone())),
                "{tokens:?} on {text:?}"
            );
        }
    }
}

/// An added token matched after normalization whose inner space is the
/// first cut candidate past the chunk size: it comes out as a unit of its
/// own before any cut, so the pooled ids are the one-piece ids.
#[test]
fn a_normalized_added_token_across_a_cut_stays_whole() {
    let mut state = 0x0F1E_2D3C_4B5A_6978u64;
    let shape = shapes()
        .into_iter()
        .find(|shape| shape.name == "normalized-words")
        .expect("the shape");
    let tokenizer = tokenizer_of(&shape, &mut state);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    let foo_bar = tokenizer.token_to_id("foo bar").expect("the added token");
    // No candidate before the token (letters only), the token's space as
    // the first candidate past the chunk size, then text enough for cuts.
    let mut text = "x".repeat(PARALLEL_CHUNK_BYTES + 5);
    text.push_str("foo bar");
    text.push_str(&long_text(&mut state, &[], 3 * PARALLEL_CHUNK_BYTES));
    let units = native.units(&text, true, true);
    let units = units.as_slice();
    assert!(
        units
            .iter()
            .any(|unit| matches!(unit, Segment::Added(id) if *id == foo_bar)),
        "the added token is a unit of its own ({} units)",
        units.len()
    );
    assert!(units.len() >= 4, "{} units", units.len());
    let pooled = native
        .encode_with(Some(tokenizer.get_model()), &text, true, false)
        .expect("pooled");
    let serial = native
        .encode_with(Some(tokenizer.get_model()), &text, false, false)
        .expect("serial");
    let reference = tokenizer
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert_eq!(pooled, serial, "the pool changed the ids");
    assert_eq!(pooled, reference, "against the reference");
    assert!(pooled.contains(&foo_bar), "the token's id is in the ids");
}

/// A stretch without a cut candidate (a table of numbers: spaces before
/// digits, never before a letter) longer than two chunks, then prose: the
/// search goes on past the stretch and the prose is chunked.
#[test]
fn chunking_goes_on_past_a_stretch_without_candidates() {
    let mut state = 0x7E57_C0DE_0BAD_F00Du64;
    let shape = shapes()
        .into_iter()
        .find(|shape| shape.name == "cl100k-ignore-merges")
        .expect("the shape");
    let tokenizer = tokenizer_of(&shape, &mut state);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    let mut text = "0.125, -3.5, 42, ".repeat(PARALLEL_CHUNK_BYTES / 8);
    assert!(text.len() > 2 * PARALLEL_CHUNK_BYTES);
    text.push_str(&long_text(&mut state, &[], 4 * PARALLEL_CHUNK_BYTES));
    let units = native.units(&text, true, true);
    let units = units.as_slice();
    assert!(
        units.len() > 2,
        "{} units: the prose after the table is chunked",
        units.len()
    );
    let pooled = native
        .encode_with(Some(tokenizer.get_model()), &text, true, false)
        .expect("pooled");
    let serial = native
        .encode_with(Some(tokenizer.get_model()), &text, false, false)
        .expect("serial");
    let reference = tokenizer
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert_eq!(pooled, serial, "the pool changed the ids");
    assert_eq!(pooled, reference, "against the reference");
}

#[test]
fn stable_cuts_leave_the_pieces_as_they_are() {
    let mut state = 0x0BAD_CAFE_F00D_0001u64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let mut text = long_text(&mut state, &[], 2 * PARALLEL_MIN_BYTES);
        if shape.nfc {
            text = nfc(&text).into_owned();
        }
        let whole: Vec<String> = {
            let mut out = Vec::new();
            native.pieces(0, &text, &mut |piece| out.push(piece.to_owned()));
            out
        };
        let mut chunked = Vec::new();
        let units = native.units(&text, true, true);
        let units = units.as_slice();
        assert!(
            units.len() >= 2,
            "{}: no cut in {} bytes",
            shape.name,
            text.len()
        );
        for unit in units {
            let Segment::Text(chunk) = unit else {
                panic!("no added tokens in this text");
            };
            native.pieces(0, chunk, &mut |piece| chunked.push(piece.to_owned()));
        }
        assert_eq!(chunked, whole, "{}: a cut split a piece", shape.name);
    }
}

#[test]
fn ids_match_tokenizers_on_random_text_for_every_shape() {
    let mut state = 0x0F0F_1234_ABCD_9876u64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = seeded(&tokenizer);
        let added: Vec<&str> = shape
            .special
            .iter()
            .copied()
            .chain(shape.added.iter().map(|(s, _)| *s))
            .collect();
        for round in 0..2000 {
            let text = random_text(&mut state, 1 + round % 40, &added);
            let reference = tokenizer
                .encode(text.as_str(), false)
                .expect("encode")
                .get_ids()
                .to_vec();
            let ids = native
                .encode(Some(tokenizer.get_model()), &text, false)
                .unwrap_or_else(|fallback| {
                    panic!("{}: declined {text:?} as {fallback:?}", shape.name)
                });
            assert_eq!(ids, reference, "{}: {text:?}", shape.name);
        }
    }
}

#[test]
fn fixed_texts_match_tokenizers_for_every_shape() {
    let texts = [
        "",
        " ",
        "\n\n",
        "Hello, world!",
        "I'm testing: 12345\nsecond line.",
        "Cafe\u{301} 中文🙂 — naïve",
        "  leading  and trailing  ",
        "<|im_start|>assistant\nHello<|im_end|>",
        "<|im_start|>user\n<tool_call>{\"a\": 1}</tool_call><|im_end|>\n",
        "WE'LL see what IT'S worth; they'd know, I'VE heard, you're, he's, isn'T",
        "x'ſ y'ſ",
        "1234567890 ١٢٣ ๑๒๓๔",
        "日本語のテキスト、한국어 텍스트, 中文文本。",
        "ครับ ผม ชื่อ นาย คำ ไทย",
        "कृपया यह पढ़ें और समझें",
        "একটি নতুন প্রোগ্রাম 16টি",
        "ఒక కొత్త కార్యక్రమం",
        "يَتَلَقَّى مَرْحَبًا",
        "fn main() { println!(\"hi\"); }\n\n\n// done\r\n",
        "a\tb\u{A0}c\u{3000}d\u{2028}e",
        "<｜begin▁of▁sentence｜>hello<｜User｜>q<｜tool▁call▁begin｜>x<｜fim▁hole｜>",
        "[e~[hello]~b] and [e~ half",
    ];
    let mut state = 0x7777_1111_2222_3333u64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = seeded(&tokenizer);
        for text in texts {
            let reference = tokenizer
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            let ids = native
                .encode(Some(tokenizer.get_model()), text, false)
                .unwrap_or_else(|fallback| {
                    panic!("{}: declined {text:?} as {fallback:?}", shape.name)
                });
            assert_eq!(ids, reference, "{}: {text:?}", shape.name);
        }
    }
}

/// The letters-only split of the Qwen2 family, under which a combining
/// mark (`\p{M}`) starts a new piece; the Qwen3.5-family files carry
/// [`QWEN35`], whose `[\p{L}\p{M}]+` keeps a letter and its marks together.
const LETTERS_ONLY: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A byte-level BPE that learned `letter` and `marks` byte by byte and
/// then their merge, the way a vocabulary that keeps letters and marks
/// together holds a cluster: the letter, the marks and the cluster are
/// tokens, and the cluster is reached only from the other two.
fn cluster_model(letter: &str, marks: &str) -> BPE {
    let table = bytes_char();
    let mut alphabet: Vec<char> = table.to_vec();
    alphabet.sort_unstable();
    let mut vocab = Vocab::default();
    for ch in alphabet {
        let next = vocab.len() as u32;
        vocab.insert(ch.to_string(), next);
    }
    let mut merges = Merges::new();
    let mut learn = |left: String, right: String| {
        let merged = format!("{left}{right}");
        if !vocab.contains_key(&merged) {
            let next = vocab.len() as u32;
            vocab.insert(merged.clone(), next);
            merges.push((left, right));
        }
        merged
    };
    let mut chain = |word: &str| {
        let mut mapped = word.bytes().map(|b| table[usize::from(b)].to_string());
        let mut token = mapped.next().expect("a non-empty word");
        for right in mapped {
            token = learn(token, right);
        }
        token
    };
    let letter = chain(letter);
    let marks = chain(marks);
    learn(letter, marks);
    BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("bpe")
}

/// An NFC byte-level tokenizer over `model` with one split `pattern`.
fn split_tokenizer(pattern: &str, model: BPE) -> HfTokenizer {
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer
        .with_normalizer(Some(NormalizerWrapper::NFC(NFC)))
        .expect("normalizer");
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        split(pattern, SplitDelimiterBehavior::Isolated, false),
        byte_level(false),
    ]))));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer.with_post_processor(Some(ByteLevelProcessor::default().trim_offsets(false)));
    tokenizer
}

/// The pieces the native pre-tokenizer cuts `text` into.
fn native_pieces(native: &NativeEncoder, text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    native.pieces(0, text, &mut |piece| pieces.push(piece.to_owned()));
    pieces
}

/// A letter followed by its combining `marks` is one piece under a split
/// that admits marks, and so one token when the vocabulary has it, and two
/// pieces and two tokens under the letters-only split: the native path
/// follows the file's pattern either way and gives `tokenizers`' ids on
/// the cluster and on `sentence`.
fn marks_stay_with_their_letter(letter: &str, marks: &str, sentence: &str) {
    let cluster = format!("{letter}{marks}");
    let model = || cluster_model(letter, marks);
    let with_marks = split_tokenizer(QWEN35, model());
    let letters_only = split_tokenizer(LETTERS_ONLY, model());
    let native_with_marks = NativeEncoder::from_tokenizer(&with_marks).expect("native path");
    let native_letters_only = NativeEncoder::from_tokenizer(&letters_only).expect("native path");
    assert_eq!(
        native_pieces(&native_with_marks, &cluster),
        std::slice::from_ref(&cluster)
    );
    assert_eq!(
        native_pieces(&native_letters_only, &cluster),
        [letter.to_owned(), marks.to_owned()]
    );
    let mut lengths = Vec::new();
    for (tokenizer, native) in [
        (&with_marks, &native_with_marks),
        (&letters_only, &native_letters_only),
    ] {
        for text in [cluster.as_str(), sentence] {
            let reference = tokenizer
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            let ids = native
                .encode(Some(tokenizer.get_model()), text, false)
                .expect("the native path takes NFC text");
            assert_eq!(ids, reference, "{text:?}");
        }
        lengths.push(
            native
                .encode(Some(tokenizer.get_model()), &cluster, false)
                .unwrap()
                .len(),
        );
    }
    assert_eq!(lengths, [1, 2], "{cluster:?}");
}

#[test]
fn bengali_marks_stay_with_their_letter() {
    marks_stay_with_their_letter("ট", "\u{9BF}", "একটি নতুন প্রোগ্রাম 16টি");
}

#[test]
fn telugu_marks_stay_with_their_letter() {
    marks_stay_with_their_letter("క", "\u{C3E}", "ఒక కొత్త కార్యక్రమం");
}

#[test]
fn thai_marks_stay_with_their_letter() {
    marks_stay_with_their_letter("น", "\u{E47}", "โปรแกรมใหม่หนึ่ง เน็ต");
}

#[test]
fn devanagari_marks_stay_with_their_letter() {
    marks_stay_with_their_letter("क", "\u{948}", "चाय कैसे बनाएँ");
}

#[test]
fn arabic_marks_stay_with_their_letter() {
    marks_stay_with_their_letter("ق", "\u{64E}\u{651}", "يَتَلَقَّى مَرْحَبًا");
}

#[test]
fn text_not_in_nfc_is_normalized_gap_by_gap_and_encoded_natively() {
    let mut state = 0x5555_6666_7777_8888u64;
    let shape = &shapes()[0];
    let mut tokenizer = tokenizer_of(shape, &mut state);
    // A special token (matched on the raw text) whose last letter would
    // compose with a mark after it, and a token matched after normalization
    // whose last letter would too.
    tokenizer
        .add_special_tokens([AddedToken::from("USER", true)])
        .expect("special token");
    tokenizer
        .add_tokens([AddedToken::from("foo bar", false).normalized(true)])
        .expect("added token");
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    let model = tokenizer.get_model();
    let user = tokenizer
        .token_to_id("USER")
        .expect("the special token's id");
    let foo_bar = tokenizer
        .token_to_id("foo bar")
        .expect("the added token's id");
    for text in [
        // "e" + combining acute composes under NFC.
        "Cafe\u{301}",
        // Marks reorder, then compose.
        "a\u{323}\u{301}",
        // Hangul jamo compose into a syllable.
        "\u{1100}\u{1161}",
        "\u{AC00}\u{11A8}",
        // Thai marks reorder.
        "ก\u{E48}\u{E38}",
        // A mark right after a raw token: cut first, the mark stays alone.
        "e<|im_start|>\u{301}x",
        "USER\u{301} and Cafe\u{301}<|im_end|>\u{301}",
        // A raw token that is not special.
        "\u{1100}<tool_call>\u{1161}",
        // A token matched after normalization: found after a composed
        // letter, not found when the mark composes with its last letter.
        "e\u{301}foo bar",
        "foo bar\u{301}",
        "e\u{301}foo bar\u{301} and foo bar",
    ] {
        let reference = tokenizer.encode(text, false).expect("encode");
        let ids = native
            .encode(Some(model), text, false)
            .unwrap_or_else(|fallback| panic!("{text:?} declined as {fallback:?}"));
        assert_eq!(ids, reference.get_ids(), "{text:?}");
    }
    // The ids are those of the normalized text.
    assert_eq!(
        native.encode(Some(model), "Cafe\u{301}", false),
        native.encode(Some(model), "Caf\u{E9}", false)
    );
    // The raw token is cut before normalization: a mark after it never
    // composes with its last letter (normalizing the whole text first
    // would turn "USER" + acute into "USEŔ" and lose the token).
    let ids = native
        .encode(Some(model), "USER\u{301}", false)
        .expect("ids");
    assert_eq!(ids.first(), Some(&user));
    // The token matched after normalization sees the normalized gap: the
    // composed last letter is no longer its spelling.
    assert!(native
        .encode(Some(model), "e\u{301}foo bar", false)
        .expect("ids")
        .contains(&foo_bar));
    assert!(!native
        .encode(Some(model), "foo bar\u{301}", false)
        .expect("ids")
        .contains(&foo_bar));
}

#[test]
fn marks_and_compositions_newer_than_the_library_tables_read_as_the_library_does() {
    // `tokenizers` normalizes with the Unicode 9.0 tables: a mark they do
    // not know is a starter to them, which blocks a composition across it,
    // and a pair they do not know never composes. A newer crate's tables
    // compose both, and the ids would be those of other bytes.
    let mut state = 0x7A7A_6B6B_5C5C_4D4Du64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    let model = tokenizer.get_model();
    for text in [
        // A letter, a mark from Unicode 14 (U+0899, combining class 220 in
        // the current tables), an acute: the current tables compose the
        // acute with the letter across the mark.
        "a\u{899}\u{301}",
        // The same with marks from Unicode 10 and 14 (U+1DF9, U+1DFA).
        "a\u{1DF9}\u{301}",
        "a\u{1DFA}\u{323}",
        // Pairs the current tables compose into one Unicode 16 character
        // (Kirat Rai, Gurung Khema, Tulu-Tigalari).
        "\u{16D63}\u{16D67}",
        "\u{1611E}\u{1611E}",
        "\u{11382}\u{113C9}",
        // Where both tables agree: a mark of class 230 blocks the acute in
        // both, and an acute right after the letter composes in both.
        "a\u{898}\u{301}",
        "a\u{301}\u{899}",
        // In a prompt, next to a raw added token.
        "<|im_start|>user\nCafe\u{301} a\u{899}\u{301} \u{16D63}\u{16D67}<|im_end|>\n",
    ] {
        let reference = tokenizer.encode(text, false).expect("encode");
        let ids = native
            .encode(Some(model), text, false)
            .unwrap_or_else(|fallback| panic!("{text:?} declined as {fallback:?}"));
        assert_eq!(ids, reference.get_ids(), "{text:?}");
    }
    // The ids are those of the text's own bytes, not of the composed form.
    assert_ne!(
        native.encode(Some(model), "a\u{899}\u{301}", false),
        native.encode(Some(model), "\u{E1}\u{899}", false)
    );
}

#[test]
fn the_nfc_check_and_copy_read_every_stretch_as_the_library_does() {
    // The stretch-wise check and copy against the library's crate run over
    // the whole text, on texts that put non-ASCII runs at the start, at the
    // end, right after one ASCII character, across the eight-byte steps of
    // the scan, next to each other, and nowhere.
    use unicode_normalization_alignments::is_nfc as library_is_nfc;
    let library_nfc = |text: &str| -> String { text.nfc().map(|(ch, _)| ch).collect() };
    let quick = NfcQuick::read();
    let pieces = [
        "a",
        "b",
        " ",
        "1234567",
        "12345678",
        "e",
        "\u{301}",
        "\u{323}",
        "\u{E9}",
        "\u{1EA1}",
        "ก",
        "\u{E48}",
        "\u{1100}",
        "\u{1161}",
        "\u{AC00}",
        "中文",
        "\u{4000}",
        "\u{9FFF}",
        "\u{3FFF}",
        "\u{A000}",
        "か",
        "\u{3099}",
        "\u{11A8}",
        "\u{FF76}",
        "、",
        "🙂",
        "\u{899}",
        "<|im_start|>",
        "\n",
    ];
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    let mut changed = 0;
    for _ in 0..3000 {
        let len = (pseudo_random(&mut state) % 9) as usize;
        let text: String = (0..len)
            .map(|_| pieces[(pseudo_random(&mut state) % pieces.len() as u64) as usize])
            .collect();
        let expected = library_nfc(&text);
        assert_eq!(is_nfc(&text), library_is_nfc(&text), "{text:?}");
        assert_eq!(
            quick.is_nfc(&text),
            library_is_nfc(&text),
            "{text:?} (quick)"
        );
        let normalized = nfc(&text);
        assert_eq!(normalized.as_ref(), expected, "{text:?}");
        assert_eq!(
            matches!(normalized, Cow::Borrowed(_)),
            text == expected,
            "{text:?}: borrowed only when nothing changed"
        );
        changed += usize::from(text != expected);
    }
    assert!(changed > 150, "{changed} of 3000 texts changed");
    // The shapes by name: ASCII alone, a run at the start, a mark after a
    // letter it composes with or not, the letter at an eight-byte step, a
    // run already in NFC next to one that is not.
    for (text, in_nfc) in [
        ("", true),
        ("plain ascii", true),
        ("\u{301}", true),
        ("\u{301}x", true),
        ("x\u{301}", true),
        ("e\u{301}", false),
        ("Cafe\u{301}", false),
        ("1234567e\u{301}", false),
        ("12345678e\u{301}", false),
        ("123456789e\u{301}", false),
        ("中文e\u{301}", false),
        ("中文", true),
        ("🙂", true),
        ("a\u{899}\u{301}", true),
        ("\u{1100}\u{1161} and \u{AC00}", false),
        // Ideographs: a mark after one composes with nothing, a kana or a
        // syllable after one composes as it would alone, runs at the edges.
        ("漢\u{301}", true),
        ("a漢\u{301}", true),
        ("e\u{301}漢", false),
        ("漢e\u{301}", false),
        ("漢\u{3099}", true),
        ("漢か\u{3099}", false),
        ("漢が", true),
        ("漢字\u{AC00}\u{11A8}", false),
        ("漢字\u{AC01}\u{11A8}", true),
        ("\u{4000}\u{9FFF}", true),
        ("\u{3FFF}\u{A000}", true),
        ("abc漢字", true),
        ("漢字abc", true),
        ("漢字 and 漢字\u{323}\u{301}", true),
        ("漢字\u{301}\u{323}", false),
    ] {
        assert_eq!(is_nfc(text), in_nfc, "{text:?}");
        assert_eq!(quick.is_nfc(text), in_nfc, "{text:?} (quick)");
        assert_eq!(nfc(text).as_ref(), library_nfc(text), "{text:?}");
    }
}

#[test]
fn every_shape_merges_its_misses_natively() {
    let mut state = 0x6666_7777_8888_9999u64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let table = native.merges.as_ref();
        assert!(table.is_some(), "{}", shape.name);
        assert_eq!(
            table.map(|table| table.ignore_merges),
            Some(shape.ignore_merges),
            "{}",
            shape.name
        );
    }
}

#[test]
fn a_model_the_merge_table_declines_keeps_the_path_through_the_model() {
    // A vocabulary without the token of byte 0x00: the table declines the
    // model, the path stays, and its misses take the model's own merge.
    let mut state = 0x1212_3434_5656_7878u64;
    let shape = &shapes()[0];
    let mut tokenizer = tokenizer_of(shape, &mut state);
    let table = bytes_char();
    let mut vocab = Vocab::default();
    let mut alphabet: Vec<char> = table.to_vec();
    alphabet.sort_unstable();
    for ch in alphabet.into_iter().filter(|&ch| ch != table[0]) {
        let id = vocab.len() as u32;
        vocab.insert(ch.to_string(), id);
    }
    let mut merges = Merges::new();
    for (a, b) in [("a", "b"), ("b", "c"), ("ab", "c"), ("a", "bc")] {
        let merged = format!("{a}{b}");
        if !vocab.contains_key(&merged) {
            let id = vocab.len() as u32;
            vocab.insert(merged, id);
        }
        merges.push((a.to_owned(), b.to_owned()));
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("bpe");
    tokenizer.with_model(bpe);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    assert!(native.merges.is_none());
    for text in [
        "abc",
        "abcabc xabc",
        "cab abcd",
        "<|im_start|>abc<|im_end|>",
    ] {
        let reference = tokenizer.encode(text, false).expect("encode");
        assert_eq!(
            native.encode(Some(tokenizer.get_model()), text, false),
            Ok(reference.get_ids().to_vec()),
            "{text:?}"
        );
    }
}

#[test]
fn a_piece_the_model_declines_is_reported_after_the_work() {
    // A vocabulary without the character for byte 0x7F whose `unk_token`
    // is not in it either: `tokenizers` fails such a piece, and so does
    // this path, once the pieces before it have been tokenized.
    let mut state = 0x4444_3333_2222_1111u64;
    let shape = &shapes()[0];
    let mut tokenizer = tokenizer_of(shape, &mut state);
    let table = bytes_char();
    let mut vocab = Vocab::default();
    for ch in table.iter().filter(|&&ch| ch != table[0x7F]) {
        let id = vocab.len() as u32;
        vocab.insert(ch.to_string(), id);
    }
    let mut bpe = BPE::builder()
        .vocab_and_merges(vocab, Merges::new())
        .build()
        .expect("bpe");
    bpe.unk_token = Some("<unk>".to_owned());
    tokenizer.with_model(bpe);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    assert!(native
        .encode(Some(tokenizer.get_model()), "plain text", false)
        .is_ok());
    assert_eq!(
        native.encode(Some(tokenizer.get_model()), "plain text\u{7F}", false),
        Err(Fallback::PieceDeclined)
    );
    assert!(tokenizer.encode("plain text\u{7F}", false).is_err());
}

#[test]
fn an_input_the_direct_encoder_gives_up_is_counted_and_encoded_by_tokenizers() {
    use crate::traits::Encoder;
    let mut state = 0x8888_9999_AAAA_BBBBu64;
    let shape = &shapes()[0];
    let mut reference = tokenizer_of(shape, &mut state);
    // The model of `a_piece_the_model_declines_is_reported_after_the_work`:
    // no character for byte 0x7F and no `unk_token` in the vocabulary.
    let table = bytes_char();
    let mut vocab = Vocab::default();
    for ch in table.iter().filter(|&&ch| ch != table[0x7F]) {
        let id = vocab.len() as u32;
        vocab.insert(ch.to_string(), id);
    }
    let mut bpe = BPE::builder()
        .vocab_and_merges(vocab, Merges::new())
        .build()
        .expect("bpe");
    bpe.unk_token = Some("<unk>".to_owned());
    reference.with_model(bpe);
    let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
    // The count applies to the direct path: with the v1 feature on, the
    // shape's split pattern is one the pipeline declines today.
    assert_eq!(ours.encode_path(), EncodePath::Native);
    let declined = |stats: &[EncodeFallbacks; 1]| {
        stats
            .iter()
            .find(|s| s.reason == "piece_declined")
            .map_or(0, |s| s.encodes)
    };
    let before = declined(&encode_fallbacks());
    for text in ["plain text\u{7F}", "\u{7F}", "a\u{7F}b"] {
        // The piece fails the `tokenizers` encode too: the error is its.
        assert!(ours.encode(text, false).is_err(), "{text:?}");
        assert!(reference.encode(text, false).is_err(), "{text:?}");
    }
    // Three inputs left the direct path here; other tests may add theirs.
    assert!(declined(&encode_fallbacks()) >= before + 3);
}

#[test]
fn the_piece_cache_returns_the_same_ids() {
    let mut state = 0xABCD_EF01_2345_6789u64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let native = seeded(&tokenizer);
    let text = "the same words the same words, again and again and again";
    let first = native
        .encode(Some(tokenizer.get_model()), text, false)
        .expect("ids");
    let second = native
        .encode(Some(tokenizer.get_model()), text, false)
        .expect("ids");
    assert_eq!(first, second);
    assert_eq!(
        first,
        tokenizer.encode(text, false).expect("encode").get_ids()
    );
}

#[test]
fn a_split_pattern_that_can_match_empty_declines_the_tokenizer() {
    let mut state = 0x2222_3333_4444_5555u64;
    let base = &shapes()[0];
    let mut tokenizer = tokenizer_of(base, &mut state);
    // `\p{N}*` matches empty between letters, and the engine splits there.
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        split(r"\p{N}*|\s+", SplitDelimiterBehavior::Isolated, false),
        byte_level(false),
    ]))));
    if let Ok(native) = NativeEncoder::from_tokenizer(&tokenizer) {
        for text in ["ab1", "the same words 12"] {
            let reference = tokenizer.encode(text, false).expect("encode");
            assert_eq!(
                native.encode(Some(tokenizer.get_model()), text, false),
                Ok(reference.get_ids().to_vec()),
                "{text:?}"
            );
        }
    }
    assert!(NativeEncoder::from_tokenizer(&tokenizer).is_err());
}

#[test]
fn piece_caches_belong_to_their_encoder() {
    // Each thread that encodes gets its own cache, held by the encoder
    // rather than by the thread, so a removed tokenizer frees all of them.
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let threads = 4;
    let barrier = std::sync::Barrier::new(threads);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (native, barrier) = (&native, &barrier);
            scope.spawn(move || {
                native
                    .encode(Some(model), "the same words again", false)
                    .expect("ids");
                barrier.wait();
            });
        }
    });
    let caches: Vec<usize> = native
        .piece_cache
        .iter_mut()
        .map(|cache| cache.get_mut().short_entries())
        .collect();
    assert_eq!(caches.len(), threads, "{caches:?}");
    assert!(caches.iter().all(|&entries| entries > 0), "{caches:?}");
    drop(native);
}

/// The raw bytes a vocabulary entry stands for, back through the byte-level
/// alphabet; `None` for an entry with a character outside it.
fn raw_bytes(entry: &str) -> Option<Vec<u8>> {
    let table = bytes_char();
    entry
        .chars()
        .map(|c| table.iter().position(|&a| a == c).map(|b| b as u8))
        .collect()
}

#[test]
fn the_seed_holds_every_short_vocabulary_entry_with_the_models_ids() {
    let mut state = 0x5EED_5EED_5EED_5EEDu64;
    for shape in shapes() {
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        let mut seeded = 0;
        for entry in tokenizer.get_vocab(false).keys() {
            let Some(key) = raw_bytes(entry).and_then(|bytes| ShortPieceKey::of(&bytes)) else {
                continue;
            };
            let expected: Vec<u32> = model
                .tokenize(entry)
                .expect("the model tokenizes its own entry")
                .iter()
                .map(|token| token.id)
                .collect();
            let mut ids = Vec::new();
            assert!(
                native.seed.ids_into(key, &mut ids),
                "{}: {entry:?} is not seeded",
                shape.name
            );
            assert_eq!(ids, expected, "{}: {entry:?}", shape.name);
            seeded += 1;
        }
        assert_eq!(native.seed.len(), seeded, "{}", shape.name);
        assert!(seeded > 256, "{}: {seeded}", shape.name);
    }
}

#[test]
fn an_entry_the_merges_do_not_reach_is_seeded_as_the_model_tokenizes_it() {
    // The vocabulary has "xyz" but the merges make only "xy": the model
    // tokenizes its own entry as two ids under the merges and as itself under
    // `ignore_merges`; the seed holds what the model gives, either way, and
    // so does the encode.
    for ignore_merges in [false, true] {
        let mut vocab = Vocab::default();
        for (id, c) in bytes_char().iter().enumerate() {
            vocab.insert(c.to_string(), id as u32);
        }
        let xy = vocab.len() as u32;
        vocab.insert("xy".to_owned(), xy);
        let xyz = vocab.len() as u32;
        vocab.insert("xyz".to_owned(), xyz);
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![("x".to_owned(), "y".to_owned())])
            .ignore_merges(ignore_merges)
            .build()
            .expect("bpe");
        let mut tokenizer = HfTokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(byte_level(true)));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        let z = model.token_to_id("z").expect("z");
        let expected = if ignore_merges {
            vec![xyz]
        } else {
            vec![xy, z]
        };
        let mut ids = Vec::new();
        assert!(native
            .seed
            .ids_into(ShortPieceKey::of(b"xyz").expect("key"), &mut ids));
        assert_eq!(ids, expected, "ignore_merges {ignore_merges}");
        assert_eq!(
            tokenizer.encode("xyz", false).expect("encode").get_ids(),
            expected.as_slice(),
            "ignore_merges {ignore_merges}"
        );
        assert_eq!(
            native.encode(Some(model), "xyz", false),
            Ok(expected),
            "ignore_merges {ignore_merges}"
        );
    }
}

#[test]
fn a_vocabulary_with_gaps_in_its_ids_keeps_the_whole_piece_lookup() {
    // The seed reads the ids `0..vocab_size`: with "xy" at 300 and "xyz" at
    // 301 in a vocabulary of 258 entries both lie beyond it, so the seed does
    // not cover the short entries and the whole-piece lookup of
    // `ignore_merges` stays on; "xyz" is then its own entry, as the model
    // says, where the lookup skipped would give the merge's two ids.
    let mut vocab = Vocab::default();
    for (id, c) in bytes_char().iter().enumerate() {
        vocab.insert(c.to_string(), id as u32);
    }
    vocab.insert("xy".to_owned(), 300);
    vocab.insert("xyz".to_owned(), 301);
    assert_eq!(vocab.len(), 258);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("x".to_owned(), "y".to_owned())])
        .ignore_merges(true)
        .build()
        .expect("bpe");
    let mut tokenizer = HfTokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level(true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let mut ids = Vec::new();
    assert!(!native
        .seed
        .ids_into(ShortPieceKey::of(b"xyz").expect("key"), &mut ids));
    assert!(!native.seed_covers_short_entries);
    assert_eq!(
        tokenizer.encode("xyz", false).expect("encode").get_ids(),
        [301]
    );
    assert_eq!(native.encode(Some(model), "xyz", false), Ok(vec![301]));
}

#[test]
fn a_long_entry_the_merges_do_not_reach_is_its_own_id_under_ignore_merges() {
    // A 20-byte entry the merges never build lies past the seed's keys: the
    // whole-piece lookup of `ignore_merges` finds it, and the filter over
    // the long entries, built from the bytes the pass read back from the
    // alphabet, must admit the bytes the piece arrives with. A 20-byte piece
    // that is no entry merges (to its bytes here: no merge reaches it).
    let long = "abcdefghijklmnopqrst";
    let other = "abcdefghijklmnopqrsu";
    let mut vocab = Vocab::default();
    for (id, c) in bytes_char().iter().enumerate() {
        vocab.insert(c.to_string(), id as u32);
    }
    let xy = vocab.len() as u32;
    vocab.insert("xy".to_owned(), xy);
    let entry = vocab.len() as u32;
    vocab.insert(long.to_owned(), entry);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("x".to_owned(), "y".to_owned())])
        .ignore_merges(true)
        .build()
        .expect("bpe");
    let mut tokenizer = HfTokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level(true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    assert!(native
        .long_entries
        .as_ref()
        .is_some_and(|filter| filter.may_hold(long.as_bytes())));
    assert_eq!(
        tokenizer.encode(long, false).expect("encode").get_ids(),
        [entry]
    );
    assert_eq!(native.encode(Some(model), long, false), Ok(vec![entry]));
    let merged: Vec<u32> = tokenizer
        .encode(other, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert_eq!(merged.len(), other.len());
    assert_eq!(native.encode(Some(model), other, false), Ok(merged));
}

#[test]
fn a_vocabulary_with_gaps_in_its_ids_keeps_the_whole_piece_lookup_for_long_entries_too() {
    // As above with the long entry at 301, past the 258 ids the seed's pass
    // reads: the pass did not read every entry, so there is no filter and
    // the lookup stays for long pieces as for short ones.
    let long = "abcdefghijklmnopqrst";
    let mut vocab = Vocab::default();
    for (id, c) in bytes_char().iter().enumerate() {
        vocab.insert(c.to_string(), id as u32);
    }
    vocab.insert("xy".to_owned(), 300);
    vocab.insert(long.to_owned(), 301);
    assert_eq!(vocab.len(), 258);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("x".to_owned(), "y".to_owned())])
        .ignore_merges(true)
        .build()
        .expect("bpe");
    let mut tokenizer = HfTokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level(true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    assert!(native.long_entries.is_none());
    assert_eq!(
        tokenizer.encode(long, false).expect("encode").get_ids(),
        [301]
    );
    assert_eq!(native.encode(Some(model), long, false), Ok(vec![301]));
}

#[test]
fn the_seed_answers_a_vocabulary_piece_before_the_model_does() {
    // The encoder is built over a model with `ignore_merges` on, so its seed
    // holds "xyz" as the entry's own id; the encode then runs over a copy of
    // the model with `ignore_merges` off, which merges "xyz" into two ids.
    // A vocabulary word is answered by the seed (the entry's id), a piece the
    // vocabulary lacks by the model; both land in the thread's packed tier.
    let mut vocab = Vocab::default();
    for (id, c) in bytes_char().iter().enumerate() {
        vocab.insert(c.to_string(), id as u32);
    }
    let xy = vocab.len() as u32;
    vocab.insert("xy".to_owned(), xy);
    let xyz = vocab.len() as u32;
    vocab.insert("xyz".to_owned(), xyz);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("x".to_owned(), "y".to_owned())])
        .ignore_merges(true)
        .build()
        .expect("bpe");
    let mut merging = bpe.clone();
    merging.ignore_merges = false;
    let merging = ModelWrapper::BPE(merging);
    let mut tokenizer = HfTokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level(true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let mut native = seeded(&tokenizer);
    let z = merging.token_to_id("z").expect("z");
    let q = merging.token_to_id("q").expect("q");
    assert_eq!(
        merging
            .tokenize("xyz")
            .expect("tokenize")
            .iter()
            .map(|t| t.id)
            .collect::<Vec<_>>(),
        [xy, z],
        "the merging copy merges the entry"
    );
    assert_eq!(
        native.encode(Some(&merging), "xyz", false),
        Ok(vec![xyz]),
        "the seed answered"
    );
    assert_eq!(
        native.encode(Some(&merging), "xyz", false),
        Ok(vec![xyz]),
        "the thread tier answered"
    );
    assert_eq!(
        native.encode(Some(&merging), "xyq", false),
        Ok(vec![xy, q]),
        "the model answered"
    );
    let caches: Vec<(usize, usize)> = native
        .piece_cache
        .iter_mut()
        .map(|cache| {
            let cache = cache.get_mut();
            (cache.packed.len(), cache.short.len())
        })
        .collect();
    assert_eq!(caches, [(2, 0)]);
}

/// Distinct words for distinct `i`: distinct pieces.
fn word(mut i: u32) -> String {
    let mut word = String::new();
    loop {
        word.push(char::from(b'a' + (i % 26) as u8));
        i /= 26;
        if i == 0 {
            return word;
        }
    }
}

#[test]
fn piece_caches_stay_within_their_bound_across_threads() {
    let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let threads = 24u32;
    // The encoder counts 4,096 threads already, as a gateway with many
    // runtime and blocking threads would: each thread's share is a thousand
    // pieces, which its 15,000 distinct words exceed many times over.
    let counted = 4_096;
    native.threads.store(counted, Ordering::Relaxed);
    let barrier = std::sync::Barrier::new(threads as usize);
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (native, barrier) = (&native, &barrier);
            scope.spawn(move || {
                native.encode(Some(model), "warm", false).expect("ids");
                barrier.wait();
                let text: String = (0..15_000)
                    .map(|i| format!(" {}", word(i * threads + t)))
                    .collect();
                // On this thread, not the pool: the test is about the
                // caches the encoding threads hold.
                native
                    .encode_with(Some(model), &text, false, false)
                    .expect("ids");
            });
        }
    });
    assert_eq!(
        native.threads.load(Ordering::Relaxed),
        counted + threads as usize
    );
    let share = native.piece_cache_share();
    assert!(share < PIECE_CACHE_CAPACITY, "{share}");
    let long_share = native.long_piece_bytes_share();
    assert!(long_share < LONG_PIECE_BYTES_CAPACITY, "{long_share}");
    let caches: Vec<usize> = native
        .piece_cache
        .iter_mut()
        .map(|cache| cache.get_mut().short_entries())
        .collect();
    assert_eq!(caches.len(), threads as usize, "{caches:?}");
    assert!(
        caches.iter().all(|&entries| entries <= share),
        "share {share}, caches {caches:?}"
    );
    assert!(
        caches.iter().sum::<usize>() <= PIECE_CACHE_TOTAL_CAPACITY,
        "{caches:?}"
    );
    // Every thread joined before any table grew, so none has more slots
    // than its share takes.
    let slots: Vec<usize> = native
        .piece_cache
        .iter_mut()
        .map(|cache| cache.get_mut().packed.slots())
        .collect();
    assert!(
        slots
            .iter()
            .all(|&slots| slots <= ShortPieceTable::slots_for(share)),
        "share {share}, slots {slots:?}"
    );
}

/// A thread's packed table starts with the fewest slots, grows with the
/// distinct pieces the thread meets up to its share, and is made anew for a
/// smaller share (threads joined since it grew) when it is cleared.
#[test]
fn a_threads_packed_table_grows_with_use_and_follows_a_fallen_share() {
    let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let packed = |native: &mut NativeEncoder| -> (usize, usize) {
        let caches: Vec<(usize, usize)> = native
            .piece_cache
            .iter_mut()
            .map(|cache| {
                let packed = &cache.get_mut().packed;
                (packed.slots(), packed.len())
            })
            .collect();
        assert_eq!(caches.len(), 1, "{caches:?}");
        caches[0]
    };
    let short = native
        .encode(Some(model), "a few words", false)
        .expect("ids");
    let (slots, len) = packed(&mut native);
    assert_eq!(slots, MIN_SLOTS, "a short prompt grows nothing");
    assert!(len <= 4, "{len}");
    // More distinct pieces than the bound of 16,384 the cache had (a text of
    // 32k tokens cleared it on every pass), fewer than the share: the table
    // grows to the slots they take and keeps every one of them.
    let text: String = (0..20_000).map(|i| format!(" {}", word(i))).collect();
    native
        .encode_with(Some(model), &text, false, false)
        .expect("ids");
    let (slots, len) = packed(&mut native);
    assert!(len >= 20_000, "{len}");
    assert!(len < PIECE_CACHE_CAPACITY, "{len}");
    assert_eq!(slots, ShortPieceTable::slots_for(len));
    assert!(slots < ShortPieceTable::slots_for(PIECE_CACHE_CAPACITY));
    // Threads join (the share falls to a fifth of the entries held): the
    // clear at the next piece makes the table anew for the new share, and
    // the thread's pieces are answered as before.
    native.threads.store(1_024, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(share < len / 4, "{share} < {len} / 4");
    let more: String = (20_000..25_000).map(|i| format!(" {}", word(i))).collect();
    native
        .encode_with(Some(model), &more, false, false)
        .expect("ids");
    let (slots, len) = packed(&mut native);
    assert_eq!(slots, ShortPieceTable::slots_for(share));
    assert!(len <= share, "{len} <= {share}");
    assert_eq!(native.encode(Some(model), "a few words", false), Ok(short));
}

/// A thread whose share fell (threads joined) since it last read it is
/// within the new share from the start of its next encode call, with no
/// miss in that call.
#[test]
fn a_thread_whose_share_fell_is_within_it_from_its_next_encode_on() {
    let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let entries = |native: &mut NativeEncoder| -> usize {
        native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().short_entries())
            .sum()
    };
    let head: String = (0..200).map(|i| format!(" {}", word(i))).collect();
    let text: String = (0..20_000).map(|i| format!(" {}", word(i))).collect();
    let before = native
        .encode_with(Some(model), &head, false, false)
        .expect("ids");
    native
        .encode_with(Some(model), &text, false, false)
        .expect("ids");
    assert!(entries(&mut native) >= 20_000);
    // Threads join: the share falls below what this thread holds, which it
    // keeps until its next encode.
    native.threads.store(1_024, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(share < 20_000, "{share}");
    assert!(entries(&mut native) >= 20_000);
    // The next encode call, of pieces the thread has all met, starts within
    // the share and answers as before.
    let after = native
        .encode_with(Some(model), &head, false, false)
        .expect("ids");
    assert_eq!(after, before);
    let held = entries(&mut native);
    assert!(held <= share, "{held} <= {share}");
}

/// A thread's packed table grown for an earlier share keeps its slots
/// through a clear at the bound; once the share fell below them (threads
/// joined) the thread's next encode call makes the table anew for the
/// share, entries within it or not.
#[test]
fn a_thread_whose_packed_table_outgrew_its_share_remakes_it_at_its_next_encode() {
    let mut state = 0xF00D_CAFE_DEAD_BEEFu64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let slots = |native: &mut NativeEncoder| -> usize {
        native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().packed_slots())
            .sum()
    };
    let entries = |native: &mut NativeEncoder| -> usize {
        native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().short_entries())
            .sum()
    };
    let head: String = (0..200).map(|i| format!(" {}", word(i))).collect();
    let text: String = (0..20_000).map(|i| format!(" {}", word(i))).collect();
    let before = native
        .encode_with(Some(model), &head, false, false)
        .expect("ids");
    native
        .encode_with(Some(model), &text, false, false)
        .expect("ids");
    let grown = slots(&mut native);
    // Threads join: the share falls below the entries, and the clear at the
    // start of the next call keeps the slots, which still serve the share.
    native.threads.store(256, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(entries(&mut native) > share, "{share}");
    assert_eq!(
        native
            .encode_with(Some(model), &head, false, false)
            .expect("ids"),
        before
    );
    assert!(entries(&mut native) <= share);
    assert_eq!(slots(&mut native), grown);
    // More threads join: the entries are within the new share, the slots are
    // not; the next call makes the table anew for the share.
    native.threads.store(1_024, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(entries(&mut native) <= share, "{share}");
    assert!(grown > ShortPieceTable::slots_for(share), "{grown} slots");
    assert_eq!(
        native
            .encode_with(Some(model), &head, false, false)
            .expect("ids"),
        before
    );
    let kept = slots(&mut native);
    assert!(
        kept <= ShortPieceTable::slots_for(share),
        "{kept} slots for a share of {share}"
    );
}

/// A thread's map of the 16-to-64-byte pieces grown for an earlier share
/// keeps its buckets through a clear at the bound (the map's own clear
/// keeps the allocation); once the share fell below half its capacity
/// (threads joined) the thread's next encode call makes the map anew for
/// the share, entries within it or not.
#[test]
fn a_thread_whose_map_outgrew_its_share_remakes_it_at_its_next_encode() {
    let mut state = 0xCAFE_F00D_BEEF_DEADu64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let mut native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let capacity = |native: &mut NativeEncoder| -> usize {
        native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().map_capacity())
            .sum()
    };
    let entries = |native: &mut NativeEncoder| -> usize {
        native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().short_entries())
            .sum()
    };
    // Pieces of 21 bytes (a space and twenty letters): the map tier.
    let head: String = (0..200).map(|i| format!(" {}", letters(20, i))).collect();
    let text: String = (0..20_000)
        .map(|i| format!(" {}", letters(20, i)))
        .collect();
    let before = native
        .encode_with(Some(model), &head, false, false)
        .expect("ids");
    native
        .encode_with(Some(model), &text, false, false)
        .expect("ids");
    assert!(entries(&mut native) >= 20_000);
    let grown = capacity(&mut native);
    assert!(grown >= 20_000, "{grown}");
    // Threads join: the share falls below the entries; the clear at the
    // start of the next call keeps the buckets, which still serve the share.
    native.threads.store(256, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(grown <= 2 * share, "{grown} for a share of {share}");
    assert_eq!(
        native
            .encode_with(Some(model), &head, false, false)
            .expect("ids"),
        before
    );
    assert!(entries(&mut native) <= share);
    assert_eq!(capacity(&mut native), grown);
    // More threads join: the entries are within the new share, the buckets
    // are more than twice it; the next call makes the map anew for the share.
    native.threads.store(1_024, Ordering::Relaxed);
    let share = native.piece_cache_share();
    assert!(entries(&mut native) <= share, "{share}");
    assert!(grown > 2 * share, "{grown} for a share of {share}");
    assert_eq!(
        native
            .encode_with(Some(model), &head, false, false)
            .expect("ids"),
        before
    );
    let kept = capacity(&mut native);
    assert!(kept <= 2 * share, "{kept} for a share of {share}");
}

/// `len` letters, distinct for distinct `i`: one piece of `len` bytes
/// under the byte-level split (letters only, no space). The padding is
/// a letter `word` never uses, so no two differ only in where it starts.
fn letters(len: usize, i: u32) -> String {
    let mut text = word(i);
    while text.len() < len {
        text.push('X');
    }
    text
}

/// What the long tier accounts for one entry.
fn long_entry_bytes(piece: &[u8], ids: &[u32]) -> usize {
    LONG_PIECE_ENTRY_BYTES + piece.len() + size_of_val(ids)
}

#[test]
fn long_pieces_are_cached_by_their_bytes() {
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    let short = letters(CACHED_PIECE_MAX_BYTES, 1);
    let long = letters(CACHED_PIECE_MAX_BYTES + 1, 2);
    let longest = letters(LONG_PIECE_MAX_BYTES, 3);
    let beyond = letters(LONG_PIECE_MAX_BYTES + 1, 4);
    // Newlines split the words without joining a space to them.
    let text = format!("{short}\n{long}\n{longest}\n{beyond}");
    let reference = tokenizer.encode(text.as_str(), false).expect("encode");
    let first = native.encode(Some(model), &text, false).expect("ids");
    assert_eq!(first, reference.get_ids());
    {
        let cache = native
            .piece_cache
            .get()
            .expect("this thread's cache")
            .borrow();
        assert!(cache.short.contains_key(short.as_bytes()));
        assert!(!cache.long.contains_key(short.as_bytes()));
        assert!(cache.long.contains_key(long.as_bytes()));
        assert!(cache.long.contains_key(longest.as_bytes()));
        assert!(!cache.long.contains_key(beyond.as_bytes()));
        assert!(!cache.short.contains_key(beyond.as_bytes()));
        assert_eq!(cache.long.len(), 2);
        let accounted: usize = cache
            .long
            .iter()
            .map(|(piece, ids)| long_entry_bytes(piece, ids))
            .sum();
        assert_eq!(cache.long_bytes, accounted);
    }
    // The second encode is answered from the cache: the same ids, and
    // the cache as it was.
    let second = native.encode(Some(model), &text, false).expect("ids");
    assert_eq!(second, first);
    let cache = native
        .piece_cache
        .get()
        .expect("this thread's cache")
        .borrow();
    assert_eq!(cache.long.len(), 2);
    assert_eq!(
        cache.long_bytes,
        long_entry_bytes(long.as_bytes(), &cache.long[long.as_bytes()])
            + long_entry_bytes(longest.as_bytes(), &cache.long[longest.as_bytes()])
    );
}

#[test]
fn long_pieces_stay_within_their_byte_share() {
    let mut state = 0x7A7A_5B5B_3C3C_1D1Du64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let native = seeded(&tokenizer);
    let model = tokenizer.get_model();
    // Distinct long pieces worth more than one thread's share: each
    // entry takes its 80 bytes, its ids (at least 27, the model merges
    // pairs) and its slot, so 16,000 of them are over 3 MB.
    let words = 16_000u32;
    let text: String = (0..words)
        .map(|i| format!("{}\n", letters(80, i)))
        .collect();
    native.encode(Some(model), &text, false).expect("ids");
    let share = native.long_piece_bytes_share();
    let cache = native
        .piece_cache
        .get()
        .expect("this thread's cache")
        .borrow();
    let accounted: usize = cache
        .long
        .iter()
        .map(|(piece, ids)| long_entry_bytes(piece, ids))
        .sum();
    assert_eq!(cache.long_bytes, accounted);
    assert!(cache.long_bytes <= share, "{} > {share}", cache.long_bytes);
    assert!(cache.long.len() < words as usize, "{}", cache.long.len());
    assert!(!cache.long.is_empty());
}

#[test]
fn text_in_nfc_takes_the_native_path_when_the_quick_check_is_unsure() {
    let mut state = 0x6666_7777_8888_9999u64;
    let shape = &shapes()[0];
    let tokenizer = tokenizer_of(shape, &mut state);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    // Marks the quick check cannot vouch for, on text that is already in
    // NFC: a Bengali vowel sign, a Devanagari nukta (its composition is
    // excluded), an acute on a letter with no precomposed form, a lone
    // Hangul trailing consonant, a prompt in Thai and Hindi.
    for text in [
        "কা",
        "क़ पढ़ें",
        "x\u{301}",
        "\u{11A8}",
        "<|im_start|>user\nครับ ผม ชื่อ นาย คำ ไทย, कृपया यह पढ़ें<|im_end|>\n",
    ] {
        assert_ne!(
            is_nfc_quick(text.chars()),
            IsNormalized::Yes,
            "{text:?}: the quick check should be unsure here"
        );
        let ids = native
            .encode(Some(tokenizer.get_model()), text, false)
            .unwrap_or_else(|fallback| {
                panic!("{text:?} is in NFC and must take the native path, not {fallback:?}")
            });
        assert_eq!(
            ids,
            tokenizer.encode(text, false).expect("encode").get_ids(),
            "{text:?}"
        );
    }
    // Not in NFC: composition or reordering changes the text; the path
    // normalizes it and the ids are those `tokenizers` gives.
    for text in [
        "Cafe\u{301}",
        "a\u{323}\u{301}",
        "\u{1100}\u{1161}",
        "\u{AC00}\u{11A8}",
        "ก\u{E48}\u{E38}",
    ] {
        assert_eq!(
            native
                .encode(Some(tokenizer.get_model()), text, false)
                .unwrap_or_else(|fallback| panic!("{text:?} declined as {fallback:?}")),
            tokenizer.encode(text, false).expect("encode").get_ids(),
            "{text:?}"
        );
    }
}

/// `post_processor` built from a single template and its special tokens.
fn template(single: &str, tokens: Vec<SpecialToken>) -> TemplateProcessing {
    TemplateProcessing::builder()
        .try_single(single)
        .expect("a single template")
        .special_tokens(tokens)
        .build()
        .expect("a template post-processor")
}

#[test]
fn a_template_post_processor_frames_the_ids_on_request() {
    let mut state = 0x1F2E_3D4C_5B6A_7988u64;
    let shape = &shapes()[0];
    let base = tokenizer_of(shape, &mut state);
    let id = |token: &str| base.token_to_id(token).expect("a special token's id");
    let (start, end, eot) = (id("<|im_start|>"), id("<|im_end|>"), id("<|endoftext|>"));
    // A token before and one after; the byte-level processor in front of
    // a template, the form the Llama-3-class files carry; a special token
    // standing for two ids.
    let forms: [(&str, PostProcessorWrapper); 3] = [
        (
            "before and after",
            template(
                "<|im_start|> $A <|im_end|>",
                vec![("<|im_start|>", start).into(), ("<|im_end|>", end).into()],
            )
            .into(),
        ),
        (
            "byte-level then template",
            ProcessorSequence::new(vec![
                ByteLevelProcessor::default().trim_offsets(false).into(),
                template("<|endoftext|> $A", vec![("<|endoftext|>", eot).into()]).into(),
            ])
            .into(),
        ),
        (
            "a two-id token",
            template(
                "$A <|turn|>",
                vec![SpecialToken::new(
                    "<|turn|>".to_owned(),
                    vec![end, eot],
                    vec!["<|im_end|>".to_owned(), "<|endoftext|>".to_owned()],
                )
                .expect("two ids")],
            )
            .into(),
        ),
    ];
    let added: Vec<&str> = shape
        .special
        .iter()
        .copied()
        .chain(shape.added.iter().map(|(s, _)| *s))
        .collect();
    for (name, post_processor) in forms {
        let mut tokenizer = tokenizer_of(shape, &mut state);
        tokenizer.with_post_processor(Some(post_processor));
        let native = NativeEncoder::from_tokenizer(&tokenizer)
            .unwrap_or_else(|declined| panic!("{name}: {declined}"));
        let model = tokenizer.get_model();
        for round in 0..300 {
            let text = if round == 0 {
                String::new()
            } else {
                random_text(&mut state, 1 + round % 40, &added)
            };
            for add_special_tokens in [false, true] {
                let reference = tokenizer
                    .encode(text.as_str(), add_special_tokens)
                    .expect("encode");
                let ids = native
                    .encode(Some(model), &text, add_special_tokens)
                    .unwrap_or_else(|fallback| panic!("{name}: {text:?} declined as {fallback:?}"));
                assert_eq!(
                    ids,
                    reference.get_ids(),
                    "{name}: {text:?} with special tokens {add_special_tokens}"
                );
            }
        }
        // The pooled encode frames the whole result once.
        let long = long_text(&mut state, &added, 3 * PARALLEL_MIN_BYTES);
        assert_eq!(
            native
                .encode_with(Some(model), &long, true, true)
                .expect("pooled"),
            tokenizer
                .encode(long.as_str(), true)
                .expect("encode")
                .get_ids(),
            "{name}: pooled"
        );
        // The plain encode of a segment never frames.
        assert_eq!(
            native
                .encode_plain(Some(model), "hello", true)
                .expect("plain"),
            native.encode(Some(model), "hello", false).expect("ids"),
            "{name}"
        );
    }
}

#[test]
fn a_template_before_the_byte_level_processor_frames_the_ids_too() {
    let mut state = 0x2A2B_2C2D_2E2F_3031u64;
    let shape = shapes()
        .into_iter()
        .find(|shape| shape.name == "cl100k-ignore-merges")
        .expect("the shape");
    let mut tokenizer = tokenizer_of(&shape, &mut state);
    let id = |token: &str| tokenizer.token_to_id(token).expect("a special token's id");
    let (gmask, sop) = (id("[gMASK]"), id("<sop>"));
    tokenizer.with_post_processor(Some(ProcessorSequence::new(vec![
        template(
            "[gMASK] $A <sop>",
            vec![("[gMASK]", gmask).into(), ("<sop>", sop).into()],
        )
        .into(),
        ByteLevelProcessor::default().trim_offsets(true).into(),
    ])));
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
    assert_eq!(
        native.frame,
        Frame {
            before: vec![gmask],
            after: vec![sop],
        }
    );
    for text in ["", "hello world", "x[gMASK]y<sop>"] {
        for add_special_tokens in [false, true] {
            assert_eq!(
                native
                    .encode(Some(tokenizer.get_model()), text, add_special_tokens)
                    .expect("ids"),
                tokenizer
                    .encode(text, add_special_tokens)
                    .expect("encode")
                    .get_ids(),
                "{text:?} with special tokens {add_special_tokens}"
            );
        }
    }
}

#[test]
fn template_shapes_outside_the_frame_are_declined() {
    use serde_json::json;
    let declined = |post_processor: Value| {
        NativeEncoder::frame_from(&post_processor)
            .expect_err("declined")
            .to_string()
    };
    let tokens = json!({
        "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
        "<none>": {"id": "<none>", "ids": [], "tokens": []},
    });
    let bos = json!({"SpecialToken": {"id": "<s>", "type_id": 0}});
    let none = json!({"SpecialToken": {"id": "<none>", "type_id": 0}});
    let a = json!({"Sequence": {"id": "A", "type_id": 0}});
    let b = json!({"Sequence": {"id": "B", "type_id": 1}});
    let template = |single: Vec<Value>| {
        json!({
            "type": "TemplateProcessing",
            "single": single,
            "pair": [bos, a, b],
            "special_tokens": tokens,
        })
    };
    // The frame the file reader of `tokenizers` takes as it is but the
    // library's builder would refuse: the body dropped, the body twice, a
    // second sequence, a token outside the map.
    assert_eq!(
        declined(template(vec![bos.clone()])),
        "a TemplateProcessing single template without one $A sequence"
    );
    assert_eq!(
        declined(template(vec![a.clone(), a.clone()])),
        "a TemplateProcessing single template without one $A sequence"
    );
    assert_eq!(
        declined(template(vec![a.clone(), b.clone()])),
        "a TemplateProcessing single template with a $B sequence"
    );
    assert_eq!(
        declined(template(vec![
            json!({"SpecialToken": {"id": "<unk>", "type_id": 0}}),
            a.clone(),
        ])),
        "a TemplateProcessing special token \"<unk>\" missing from its map"
    );
    assert_eq!(
        declined(json!({"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]})),
        "post-processor RobertaProcessing (only ByteLevel, TemplateProcessing or none)"
    );
    // A template after one with special-token pieces: the library hands
    // that one's pieces to the next unmerged (two take its pair template,
    // three hit a `todo!()` in `TemplateProcessing::process_encodings`),
    // whether the next adds ids or not, and a piece with an empty id list
    // is a piece too; a template of `$A` alone passes one encoding through
    // and may come before one that adds.
    let two_templates = |first: Vec<Value>, second: Vec<Value>| {
        json!({
            "type": "Sequence",
            "processors": [template(first), template(second)],
        })
    };
    let after_pieces =
        "a post-processor sequence with a template after one with special-token pieces";
    assert_eq!(
        declined(two_templates(
            vec![bos.clone(), a.clone()],
            vec![a.clone(), bos.clone()]
        )),
        after_pieces
    );
    assert_eq!(
        declined(two_templates(vec![bos.clone(), a.clone()], vec![a.clone()])),
        after_pieces
    );
    assert_eq!(
        declined(two_templates(vec![a.clone(), none], vec![a.clone()])),
        after_pieces
    );
    assert_eq!(
        NativeEncoder::frame_from(&two_templates(
            vec![a.clone()],
            vec![a.clone(), bos.clone()]
        ))
        .expect("frame"),
        Frame {
            before: vec![],
            after: vec![1],
        }
    );
    // A sequence inside a sequence, and a member of another kind.
    assert_eq!(
        declined(json!({
            "type": "Sequence",
            "processors": [{"type": "Sequence", "processors": [template(vec![a.clone()])]}],
        })),
        "a post-processor sequence inside a sequence"
    );
    assert_eq!(
        declined(json!({
            "type": "Sequence",
            "processors": [{"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]}],
        })),
        "post-processor RobertaProcessing in a sequence (only ByteLevel or TemplateProcessing)"
    );
    // The shapes taken: the Llama-3-class form, and a pair template that
    // is never read.
    assert_eq!(
        NativeEncoder::frame_from(&template(vec![bos.clone(), a.clone()])).expect("frame"),
        Frame {
            before: vec![1],
            after: vec![],
        }
    );
    assert_eq!(
        NativeEncoder::frame_from(&json!({
            "type": "Sequence",
            "processors": [
                {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
                template(vec![a.clone(), bos.clone()]),
            ],
        }))
        .expect("frame"),
        Frame {
            before: vec![],
            after: vec![1],
        }
    );
}

#[test]
fn the_ideograph_run_is_exact_at_its_utf8_edges() {
    // U+4000 is E4 80 80, U+9FFF is E9 BF BF; U+3FFF (E3 BF BF) and U+A000
    // (EA 80 80) are outside, as are the kana under E3 and the Hangul
    // syllables under EA..=ED. A continuation byte never starts a run.
    assert_eq!(ideograph_run("\u{4000}".as_bytes()), 3);
    assert_eq!(ideograph_run("\u{9FFF}".as_bytes()), 3);
    assert_eq!(ideograph_run("\u{3FFF}".as_bytes()), 0);
    assert_eq!(ideograph_run("\u{A000}".as_bytes()), 0);
    assert_eq!(ideograph_run("か".as_bytes()), 0);
    assert_eq!(ideograph_run("\u{AC00}".as_bytes()), 0);
    assert_eq!(ideograph_run("漢字か漢".as_bytes()), 6);
    assert_eq!(ideograph_run("漢字abc".as_bytes()), 6);
    assert_eq!(ideograph_run(&"漢".as_bytes()[1..]), 0);
    assert_eq!(checked_run("か漢".as_bytes()), 3);
    assert_eq!(checked_run("かな abc".as_bytes()), 6);
    assert_eq!(checked_run("\u{301}\u{3099}漢".as_bytes()), 5);
    assert_eq!(checked_run("漢".as_bytes()), 0);
    // The stretches of a mixed text: settled runs on their own, the ASCII
    // character before a run of other characters kept with it.
    let text = "ab漢字e\u{301}か\u{3099}漢 x";
    let got: Vec<(usize, &str, bool)> = stretches(text).collect();
    assert_eq!(
        got,
        [
            (0, "ab", true),
            (2, "漢字", true),
            (8, "e\u{301}か\u{3099}", false),
            (17, "漢", true),
            (20, " x", true),
        ]
    );
}

#[test]
fn the_ideograph_range_is_settled_by_the_crates_own_tables() {
    // Every code point of U+4000..=U+9FFF, read from the tables the native
    // path normalizes with: combining class 0, no canonical decomposition,
    // the quick check's Yes (so not the second element of any composition:
    // those are the quick check's Maybe, by its definition), and no
    // composition with any possible second element after it. The candidate
    // second elements are every code point the quick check calls Maybe;
    // three known compositions outside the range show the walk sees them;
    // the counts are asserted at the end.
    use unicode_normalization_alignments::char::{
        canonical_combining_class, compose, decompose_canonical,
    };
    let quick = |ch: char| is_nfc_quick(std::iter::once(ch));
    let seconds: Vec<char> = (0..=0x10FFFFu32)
        .filter_map(char::from_u32)
        .filter(|&ch| quick(ch) == IsNormalized::Maybe)
        .collect();
    assert!(seconds.contains(&'\u{301}'));
    assert!(seconds.contains(&'\u{3099}'));
    assert!(seconds.contains(&'\u{11A8}'));
    assert_eq!(compose('e', '\u{301}'), Some('\u{E9}'));
    assert_eq!(compose('か', '\u{3099}'), Some('が'));
    assert_eq!(compose('\u{AC00}', '\u{11A8}'), Some('\u{AC01}'));
    let (mut walked, mut compositions) = (0usize, 0usize);
    for ch in '\u{4000}'..='\u{9FFF}' {
        walked += 1;
        assert_eq!(canonical_combining_class(ch), 0, "{ch:?}");
        let mut parts = Vec::new();
        decompose_canonical(ch, |part| parts.push(part));
        assert_eq!(parts, [ch], "{ch:?}");
        assert_eq!(quick(ch), IsNormalized::Yes, "{ch:?}");
        compositions += seconds
            .iter()
            .filter(|&&second| compose(ch, second).is_some())
            .count();
    }
    // The counts of the walk: the range, the crate's Maybe set (Unicode 9.0
    // tables), and no composition between them.
    assert_eq!(walked, 0x6000);
    assert_eq!(seconds.len(), 110);
    assert_eq!(compositions, 0);
}

#[test]
fn an_ascii_free_line_splits_into_its_pieces_in_one_pass() {
    // Chinese prose with full-width punctuation and no ASCII at all: every
    // mark is a checked stretch on its own and every run of ideographs a
    // settled one. `checked_run` ends at the first ASCII or ideograph lead
    // byte; scanning the whole non-ASCII run first and trimming it to the
    // first ideograph read the rest of the line again for every mark.
    let line = "漢字中文测试，漢字中文测试。".repeat(512);
    let got: Vec<(usize, &str, bool)> = stretches(&line).collect();
    assert_eq!(got.len(), 2048);
    let mut at = 0;
    for (start, stretch, settled) in &got {
        assert_eq!(*start, at);
        at += stretch.len();
        if *settled {
            assert_eq!(*stretch, "漢字中文测试");
        } else {
            assert!(*stretch == "，" || *stretch == "。", "{stretch:?}");
        }
    }
    assert_eq!(at, line.len());
    assert!(is_nfc(&line));
    assert!(matches!(nfc(&line), Cow::Borrowed(_)));
    // The checked run ends at an ideograph, at an ASCII byte, or at the end.
    assert_eq!(checked_run("，漢".as_bytes()), 3);
    assert_eq!(checked_run("，。、漢字".as_bytes()), 9);
    assert_eq!(checked_run("，。x".as_bytes()), 6);
    assert_eq!(checked_run("，。".as_bytes()), 6);
    assert_eq!(checked_run("x".as_bytes()), 0);
}

/// A tiktoken vocabulary over the pool: the 256 bytes and `merges` tokens
/// learnt from a corpus by the byte-pair rule (the most frequent adjacent
/// pair inside a word becomes the next token), the byte tokens ranked
/// first or, as in a file whose byte tokens are not its first lines, last.
fn tiktoken_ranks(state: &mut u64, merges: usize, bytes_last: bool) -> FxHashMap<Vec<u8>, u32> {
    let corpus = random_text(state, 6000, &[]);
    let mut symbols: Vec<Vec<u8>> = corpus.bytes().map(|b| vec![b]).collect();
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    while tokens.len() < merges {
        let mut counts: FxHashMap<(Vec<u8>, Vec<u8>), usize> = FxHashMap::default();
        for pair in symbols.windows(2) {
            if pair[0] == b" " || pair[1] == b" " || pair[0] == b"\n" || pair[1] == b"\n" {
                continue;
            }
            *counts
                .entry((pair[0].clone(), pair[1].clone()))
                .or_default() += 1;
        }
        let Some(((a, b), _)) = counts
            .into_iter()
            .max_by_key(|((a, b), n)| (*n, std::cmp::Reverse((a.clone(), b.clone()))))
        else {
            break;
        };
        let token = [a.as_slice(), b.as_slice()].concat();
        if seen.insert(token.clone()) {
            tokens.push(token.clone());
        }
        let mut merged = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                merged.push(token.clone());
                i += 2;
            } else {
                merged.push(symbols[i].clone());
                i += 1;
            }
        }
        symbols = merged;
    }
    let mut ranks = FxHashMap::default();
    let (byte_base, token_base) = if bytes_last {
        (tokens.len() as u32, 0)
    } else {
        (0, 256)
    };
    for byte in 0..=255u8 {
        ranks.insert(vec![byte], byte_base + u32::from(byte));
    }
    for (i, token) in tokens.into_iter().enumerate() {
        ranks.insert(token, token_base + i as u32);
    }
    ranks
}

/// A tiktoken vocabulary rides the direct path as the file's own encoder
/// encodes it: the Kimi template with the byte tokens first and last, and
/// the cl100k template; special tokens cut out of the raw text (`encode`)
/// or read as text (`encode_plain`); random text of every pool character
/// with the specials sprinkled in, short, and long enough for the thread
/// pool; against `tiktoken_rs::CoreBPE` over the same ranks.
#[test]
fn a_tiktoken_vocabulary_encodes_on_the_direct_path_as_its_own_encoder_does() {
    const SPECIALS: [&str; 4] = ["[BOS]", "[EOS]", "<|open|>", "<|reserved_token_1|>"];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for (pattern, bytes_last) in [
        (KIMI_K2_PATTERN, false),
        (KIMI_K2_PATTERN, true),
        (CL100K, false),
    ] {
        let ranks = tiktoken_ranks(&mut state, 600, bytes_last);
        let next = ranks.len() as u32 + 7;
        let specials: Vec<(String, u32)> = SPECIALS
            .iter()
            .enumerate()
            .map(|(i, special)| ((*special).to_owned(), next + i as u32))
            .collect();
        let native =
            NativeEncoder::from_ranks(&ranks, pattern, &specials).expect("the direct path");
        assert!(native.chain.is_none());
        assert_eq!(native.stages.len(), 1);
        let own = CoreBPE::new(ranks.clone(), specials.iter().cloned().collect(), pattern)
            .expect("the file's encoder");
        for round in 0..1500 {
            let text = random_text(&mut state, 1 + round % 48, &SPECIALS);
            assert_eq!(
                native.encode(None, &text, false),
                Ok(own.encode_with_special_tokens(&text)),
                "bytes last {bytes_last}: {text:?}"
            );
            assert_eq!(
                native.encode_plain(None, &text, true),
                Ok(own.encode_ordinary(&text)),
                "bytes last {bytes_last}, plain: {text:?}"
            );
        }
        let long = long_text(&mut state, &SPECIALS, 3 * PARALLEL_MIN_BYTES);
        assert_eq!(
            native.encode(None, &long, false),
            Ok(own.encode_with_special_tokens(&long))
        );
        assert_eq!(
            native.encode_plain(None, &long, true),
            Ok(own.encode_ordinary(&long))
        );
    }
}

/// Characters of the pool the sentencepiece vocabulary below has no token
/// for: they fall to their bytes' tokens.
const BYTE_FALLBACK_CHARS: &[char] = &['😀', '👍', '\u{1F3FD}', 'Ω', 'ǅ', '\u{0}', '\u{7F}'];

/// Added tokens of the sentencepiece shapes: one matched on the raw text,
/// one after normalization.
const SENTENCEPIECE_ADDED: &[(&str, bool)] = &[("<think>", false), ("[note]", true)];

/// A sentencepiece-style BPE with byte fallback: `<unk>`, the sentinels,
/// the 256 byte tokens, the word marker, one token per other character of
/// the pool but [`BYTE_FALLBACK_CHARS`], and merges for the pairs of a
/// corpus of pool text with its spaces as markers (as a trainer leaves
/// them), so pieces merge and the marker merges with the words it starts;
/// one merge joins a word to the marker after it, so a token spans a word
/// boundary.
fn sentencepiece_model(state: &mut u64) -> BPE {
    let mut vocab = Vocab::default();
    let add = |vocab: &mut Vocab, token: String| {
        if !vocab.contains_key(&token) {
            let id = vocab.len() as u32;
            vocab.insert(token, id);
        }
    };
    for token in ["<unk>", "<s>", "</s>"] {
        add(&mut vocab, token.to_owned());
    }
    for byte in 0..=255u8 {
        add(&mut vocab, format!("<0x{byte:02X}>"));
    }
    add(&mut vocab, WORD_MARKER.to_string());
    for &c in POOL {
        if c != ' ' && !BYTE_FALLBACK_CHARS.contains(&c) {
            add(&mut vocab, c.to_string());
        }
    }
    let corpus = random_text(state, 4000, &[]).replace(' ', "\u{2581}");
    let chars: Vec<char> = corpus.chars().collect();
    let mut merges = Merges::new();
    let mut seen = std::collections::HashSet::new();
    for pair in chars.windows(2) {
        let (a, b) = (pair[0].to_string(), pair[1].to_string());
        if !vocab.contains_key(&a)
            || !vocab.contains_key(&b)
            || !seen.insert((a.clone(), b.clone()))
        {
            continue;
        }
        let merged = format!("{a}{b}");
        if vocab.contains_key(&merged) {
            continue;
        }
        add(&mut vocab, merged);
        merges.push((a, b));
        if merges.len() >= 600 {
            break;
        }
    }
    for (a, b) in [
        ("\u{2581}t", "h"),
        ("\u{2581}th", "e"),
        ("t", "he"),
        ("\u{2581}", "the"),
        ("\u{2581}the", "\u{2581}"),
        ("e", "d"),
    ] {
        if vocab.contains_key(a) && vocab.contains_key(b) {
            let merged = format!("{a}{b}");
            if !vocab.contains_key(&merged) {
                add(&mut vocab, merged);
                merges.push((a.to_owned(), b.to_owned()));
            }
        }
    }
    BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .unk_token("<unk>".to_owned())
        .fuse_unk(true)
        .build()
        .expect("a byte-fallback BPE")
}

/// The pipelines of the byte-fallback families: the marker by a `Replace`
/// normalizer with the no-op `Split` behind it (the Gemma class) or alone,
/// or by the Llama class's `Metaspace` under each prepend scheme.
struct SentencepieceShape {
    name: &'static str,
    replace: bool,
    split: bool,
    metaspace: Option<PrependScheme>,
    /// The post-processor frames the ids with `<s>`.
    bos: bool,
}

fn sentencepiece_shapes() -> Vec<SentencepieceShape> {
    vec![
        SentencepieceShape {
            name: "replace-split",
            replace: true,
            split: true,
            metaspace: None,
            bos: true,
        },
        SentencepieceShape {
            name: "replace",
            replace: true,
            split: false,
            metaspace: None,
            bos: false,
        },
        SentencepieceShape {
            name: "metaspace-first",
            replace: false,
            split: false,
            metaspace: Some(PrependScheme::First),
            bos: true,
        },
        SentencepieceShape {
            name: "metaspace-always",
            replace: false,
            split: false,
            metaspace: Some(PrependScheme::Always),
            bos: false,
        },
        SentencepieceShape {
            name: "metaspace-never",
            replace: false,
            split: false,
            metaspace: Some(PrependScheme::Never),
            bos: false,
        },
    ]
}

fn sentencepiece_tokenizer(shape: &SentencepieceShape, state: &mut u64) -> HfTokenizer {
    let mut tokenizer = HfTokenizer::new(sentencepiece_model(state));
    if shape.replace {
        tokenizer
            .with_normalizer(Some(
                Replace::new(" ", "\u{2581}").expect("a Replace normalizer"),
            ))
            .expect("normalizer");
    }
    if shape.split {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::String(" ".to_owned()),
                SplitDelimiterBehavior::MergedWithPrevious,
                false,
            )
            .expect("split"),
        )));
    }
    if let Some(scheme) = shape.metaspace {
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', scheme, false)));
    }
    let mut decoder: Vec<DecoderWrapper> = vec![
        Replace::new("\u{2581}", " ")
            .expect("a Replace decoder")
            .into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
    ];
    if shape
        .metaspace
        .is_some_and(|scheme| scheme != PrependScheme::Never)
    {
        decoder.push(Strip::new(' ', 1, 0).into());
    }
    tokenizer.with_decoder(Some(DecoderSequence::new(decoder)));
    if shape.bos {
        let bos = tokenizer.get_model().token_to_id("<s>").expect("<s>");
        tokenizer.with_post_processor(Some(template("<s> $A", vec![("<s>", bos).into()])));
    }
    tokenizer
        .add_special_tokens([
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ])
        .expect("special tokens");
    tokenizer
        .add_tokens(
            SENTENCEPIECE_ADDED
                .iter()
                .map(|(s, normalized)| AddedToken::from(*s, false).normalized(*normalized)),
        )
        .expect("added tokens");
    tokenizer
}

/// The spellings [`random_text`] drops into a text of a sentencepiece shape.
fn sentencepiece_added() -> Vec<&'static str> {
    ["<s>", "</s>"]
        .into_iter()
        .chain(SENTENCEPIECE_ADDED.iter().map(|(s, _)| *s))
        .collect()
}

/// The plain tokenizer the segment encode falls back to: the pipeline
/// without its added vocabulary, and after a control under `Metaspace`
/// `First` the scheme `Never` (as `huggingface` builds it).
fn plain_tokenizer(tokenizer: &HfTokenizer, at_start: bool) -> HfTokenizer {
    let mut plain = tokenizer.clone();
    plain.with_added_vocabulary(AddedVocabulary::new());
    if let Some(PreTokenizerWrapper::Metaspace(metaspace)) = tokenizer.get_pre_tokenizer() {
        if !at_start && metaspace.get_prepend_scheme() == PrependScheme::First {
            plain.with_pre_tokenizer(Some(Metaspace::new(
                metaspace.get_replacement(),
                PrependScheme::Never,
                metaspace.get_split(),
            )));
        }
    }
    plain
}

#[test]
fn every_sentencepiece_shape_gets_a_native_path() {
    let mut state = 0x5EED_5EED_5EED_5EEDu64;
    for shape in sentencepiece_shapes() {
        let tokenizer = sentencepiece_tokenizer(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer);
        assert!(
            native.is_ok(),
            "{}: {:?}",
            shape.name,
            native.as_ref().err()
        );
        let native = native.expect("native");
        assert!(native.chars.is_some(), "{}", shape.name);
        assert!(
            native.stages.is_empty() && native.merges.is_none(),
            "{}",
            shape.name
        );
    }
}

/// A byte-level vocabulary whose file sets `byte_fallback` keeps the
/// byte-level path: the shape decides, not the flag.
#[test]
fn a_byte_level_vocabulary_with_the_byte_fallback_flag_keeps_its_path() {
    let mut state = 0xB17E_1E7E_1F1A_6000u64;
    let shape = &shapes()[0];
    let mut tokenizer = tokenizer_of(shape, &mut state);
    let mut model = model(&mut state, shape.ignore_merges);
    model.byte_fallback = true;
    tokenizer.with_model(model);
    let native = NativeEncoder::from_tokenizer(&tokenizer).expect("the byte-level path");
    assert!(native.chars.is_none() && !native.stages.is_empty());
}

#[test]
fn ids_match_tokenizers_on_random_text_for_every_sentencepiece_shape() {
    let mut state = 0x0DD5_EED1_2345_6789u64;
    let added = sentencepiece_added();
    for shape in sentencepiece_shapes() {
        let tokenizer = sentencepiece_tokenizer(&shape, &mut state);
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        for round in 0..300 {
            let mut text = random_text(&mut state, 1 + round % 90, &added);
            // Texts that start with a space, a marker or an added token read
            // the prepend rule.
            match round % 5 {
                1 => text.insert(0, ' '),
                2 => text.insert(0, '\u{2581}'),
                3 => text.insert_str(0, "<s>"),
                _ => {}
            }
            for special in [false, true] {
                let expected = tokenizer
                    .encode(text.as_str(), special)
                    .expect("tokenizers encodes")
                    .get_ids()
                    .to_vec();
                assert_eq!(
                    native.encode(Some(model), &text, special),
                    Ok(expected.clone()),
                    "{}: {text:?} (special {special})",
                    shape.name
                );
                // Again, from the thread's piece cache.
                assert_eq!(
                    native.encode(Some(model), &text, special),
                    Ok(expected),
                    "{}: {text:?} again",
                    shape.name
                );
            }
        }
    }
}

#[test]
fn fixed_texts_match_tokenizers_for_every_sentencepiece_shape() {
    let mut state = 0xF1DE_D5EE_D000_0001u64;
    for shape in sentencepiece_shapes() {
        let tokenizer = sentencepiece_tokenizer(&shape, &mut state);
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        for text in [
            "",
            " ",
            "  ",
            "\u{2581}",
            "the",
            " the",
            "the the  the",
            "<s>the</s> the<think> the [note]the",
            "the\u{2581}the",
            "ed\n\nthe\tthe\r\n",
            "中文 字 😀 the Ω",
            "e\u{301}d \u{E31}\u{E48}",
            "123 ٣ ² the",
            "a\u{0}b\u{7F}c",
            "the\u{2581}",
            "\u{2581}\u{2581}the",
        ] {
            for special in [false, true] {
                let expected = tokenizer
                    .encode(text, special)
                    .expect("tokenizers encodes")
                    .get_ids()
                    .to_vec();
                assert_eq!(
                    native.encode(Some(model), text, special),
                    Ok(expected),
                    "{}: {text:?} (special {special})",
                    shape.name
                );
            }
        }
    }
}

#[test]
fn plain_segments_match_tokenizers_without_its_added_vocabulary_for_every_sentencepiece_shape() {
    let mut state = 0x7A1E_5EED_0000_0001u64;
    let added = sentencepiece_added();
    for shape in sentencepiece_shapes() {
        let tokenizer = sentencepiece_tokenizer(&shape, &mut state);
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        for at_start in [true, false] {
            let plain = plain_tokenizer(&tokenizer, at_start);
            // An empty segment: no marker, as `tokenizers` drops the empty
            // split before its pre-tokenizer.
            assert_eq!(
                native.encode_plain(Some(model), "", at_start),
                Ok(Vec::new()),
                "{}: empty (at_start {at_start})",
                shape.name
            );
            for round in 0..120 {
                let mut text = random_text(&mut state, 1 + round % 60, &added);
                if round % 3 == 1 {
                    text.insert(0, ' ');
                }
                let expected = plain
                    .encode(text.as_str(), false)
                    .expect("tokenizers encodes")
                    .get_ids()
                    .to_vec();
                assert_eq!(
                    native.encode_plain(Some(model), &text, at_start),
                    Ok(expected),
                    "{}: {text:?} (at_start {at_start})",
                    shape.name
                );
            }
        }
    }
}

/// A long text of a sentencepiece shape encodes the same on the pool as in
/// one piece: the chunks after the first get no word marker under any
/// scheme, and the cuts between them cross no merge.
#[test]
fn a_long_sentencepiece_text_encodes_the_same_on_the_pool_as_in_one_piece() {
    let mut state = 0x1017_5EED_C0FF_EE00u64;
    let added = sentencepiece_added();
    for shape in sentencepiece_shapes() {
        let tokenizer = sentencepiece_tokenizer(&shape, &mut state);
        let native = seeded(&tokenizer);
        let model = tokenizer.get_model();
        for special in [false, true] {
            let text = long_text(&mut state, &added, 3 * PARALLEL_MIN_BYTES);
            let units = native.units(&text, true, true).as_slice().len();
            assert!(units >= 4, "{}: {units} units", shape.name);
            let reference = tokenizer
                .encode(text.as_str(), special)
                .expect("tokenizers encodes")
                .get_ids()
                .to_vec();
            assert_eq!(
                native.encode_with(Some(model), &text, false, special),
                Ok(reference.clone()),
                "{}: in one piece",
                shape.name
            );
            assert_eq!(
                native.encode_with(Some(model), &text, true, special),
                Ok(reference),
                "{}: on the pool",
                shape.name
            );
        }
    }
}

#[test]
fn a_tiktoken_vocabulary_outside_the_path_declines() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let ranks = tiktoken_ranks(&mut state, 50, false);
    let declined = |ranks: &FxHashMap<Vec<u8>, u32>, pattern: &str, specials: &[(String, u32)]| {
        NativeEncoder::from_ranks(ranks, pattern, specials)
            .expect_err("declined")
            .to_string()
    };
    assert_eq!(
        declined(&ranks, r"\w+|\s+", &[]),
        r"a split pattern outside the supported regex subset: \w+|\s+"
    );
    assert_eq!(
        declined(&ranks, KIMI_K2_PATTERN, &[(String::new(), 9)]),
        "special token \"\" without content"
    );
    assert_eq!(
        declined(
            &ranks,
            KIMI_K2_PATTERN,
            &[
                ("<|b|>".to_owned(), 300),
                ("<|a|>".to_owned(), 301),
                ("<|a|>x".to_owned(), 302),
            ]
        ),
        "special token \"<|a|>\" is a prefix of \"<|a|>x\""
    );
    let mut incomplete = ranks.clone();
    incomplete.remove([0x7Fu8].as_slice());
    assert_eq!(
        declined(&incomplete, KIMI_K2_PATTERN, &[]),
        "byte 0x7f has no token"
    );
    assert!(NativeEncoder::from_ranks(&ranks, KIMI_K2_PATTERN, &[]).is_ok());
}

/// The texts of the fixture tree the repository's fixture tests read, when
/// `BELLWETHER_FIXTURES` names an unpacked one: the reference text of every
/// render and parse case (`<model>/render/*.jsonl`, `<model>/parse/*.jsonl`).
/// Empty without the variable, so the tests that read them hold the scanners
/// against the fuzz inputs alone; with it, a tree that cannot be read, a line
/// that is not a JSON case or a tree without texts fails the test instead of
/// passing it on the fuzz inputs alone.
fn fixture_texts() -> Vec<String> {
    let Ok(root) = std::env::var("BELLWETHER_FIXTURES") else {
        return Vec::new();
    };
    let mut texts = Vec::new();
    let models =
        std::fs::read_dir(&root).unwrap_or_else(|e| panic!("BELLWETHER_FIXTURES={root}: {e}"));
    for model in models.flatten() {
        for kind in ["render", "parse"] {
            let Ok(files) = std::fs::read_dir(model.path().join(kind)) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
                    continue;
                }
                let content = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                for (number, line) in content.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let case: Value = serde_json::from_str(line).unwrap_or_else(|e| {
                        panic!(
                            "{} line {}: not a JSON case: {e}",
                            path.display(),
                            number + 1
                        )
                    });
                    if let Some(text) = case
                        .get("reference")
                        .and_then(|reference| reference.get("text"))
                        .and_then(Value::as_str)
                    {
                        if !text.is_empty() {
                            texts.push(text.to_owned());
                        }
                    }
                }
            }
        }
    }
    assert!(
        !texts.is_empty(),
        "BELLWETHER_FIXTURES={root}: no render or parse case with a text under it"
    );
    texts
}

/// How many fuzz inputs the scanner differentials draw: `SCANNER_FUZZ_ROUNDS`
/// when set (the long runs are opt-in; it must be a count), else three
/// thousand.
fn fuzz_rounds() -> usize {
    std::env::var("SCANNER_FUZZ_ROUNDS").map_or(3000, |rounds| {
        rounds
            .parse()
            .unwrap_or_else(|e| panic!("SCANNER_FUZZ_ROUNDS={rounds}: not a count: {e}"))
    })
}

/// A random code point: ASCII a quarter of the time, the Basic Multilingual
/// Plane half of it, any plane the rest.
fn random_code_point(state: &mut u64) -> char {
    loop {
        let roll = pseudo_random(state);
        let code = match roll % 4 {
            0 => (roll >> 2) % 0x80,
            1 | 2 => (roll >> 2) % 0x1_0000,
            _ => (roll >> 2) % 0x11_0000,
        };
        if let Some(c) = char::from_u32(code as u32) {
            return c;
        }
    }
}

/// A fuzz input for the scanners: pool text, random code points of every
/// plane, or a window of a fixture text with one character changed.
fn fuzz_text(state: &mut u64, fixtures: &[String]) -> String {
    let roll = pseudo_random(state);
    match roll % 3 {
        1 => {
            let len = 1 + (roll / 3) as usize % 24;
            (0..len).map(|_| random_code_point(state)).collect()
        }
        2 if !fixtures.is_empty() => {
            let chars: Vec<char> = fixtures[(roll / 3) as usize % fixtures.len()]
                .chars()
                .collect();
            let start = pseudo_random(state) as usize % chars.len();
            let end = (start + 1 + pseudo_random(state) as usize % 64).min(chars.len());
            let mut window = chars[start..end].to_vec();
            let at = pseudo_random(state) as usize % window.len();
            window[at] = random_code_point(state);
            window.into_iter().collect()
        }
        _ => random_text(state, 1 + (roll / 3) as usize % 48, &[]),
    }
}

/// The pieces of `stages` applied in turn by their match ranges, as byte
/// ranges: the walk the encoder made before the scanners.
fn staged(stages: &[Stage], text: &str) -> Vec<(usize, usize)> {
    fn walk(stages: &[Stage], text: &str, offset: usize, out: &mut Vec<(usize, usize)>) {
        if text.is_empty() {
            return;
        }
        let Some((stage, rest)) = stages.split_first() else {
            out.push((offset, offset + text.len()));
            return;
        };
        let mut cursor = 0;
        for (start, end) in stage.pattern.find_iter(text) {
            if stage.keep_gaps && cursor < start {
                walk(rest, &text[cursor..start], offset + cursor, out);
            }
            walk(rest, &text[start..end], offset + start, out);
            cursor = end;
        }
        if stage.keep_gaps && cursor < text.len() {
            walk(rest, &text[cursor..], offset + cursor, out);
        }
    }
    let mut out = Vec::new();
    walk(stages, text, 0, &mut out);
    out
}

/// Holds the scanner of each of `patterns` against the pattern's program:
/// the same match ranges on every fixture text and on the fuzz inputs.
fn scanner_differential(patterns: &[&str]) {
    let fixtures = fixture_texts();
    let rounds = fuzz_rounds();
    let mut state = 0x7F4A_7C15_9E37_79B9u64;
    for pattern in patterns {
        let parsed = Pattern::parse(pattern).expect("parses");
        assert!(parsed.has_scanner(), "{pattern}");
        let program = parsed.program_only();
        let check = |text: &str| {
            let expected = program.find_iter(text).collect::<Vec<_>>();
            assert_eq!(
                parsed.find_iter(text).collect::<Vec<_>>(),
                expected,
                "pattern {pattern:?} text {text:?}"
            );
            let mut each = Vec::new();
            parsed.for_each_match(text, |start, end| each.push((start, end)));
            assert_eq!(
                each, expected,
                "for_each_match: pattern {pattern:?} text {text:?}"
            );
        };
        for text in &fixtures {
            check(text);
        }
        for _ in 0..rounds {
            check(&fuzz_text(&mut state, &fixtures));
        }
    }
}

#[test]
fn the_gpt4_template_scanner_cuts_the_fixture_and_fuzz_texts_as_its_program_does() {
    scanner_differential(&[CL100K, QWEN35]);
}

#[test]
fn the_cased_template_scanner_cuts_the_fixture_and_fuzz_texts_as_its_program_does() {
    scanner_differential(&[CASED, TEKKEN]);
}

#[test]
fn the_chain_scanner_cuts_the_fixture_and_fuzz_texts_as_its_stage_programs_do() {
    let stages: Vec<Stage> = [DIGITS, CJK, DEEPSEEK]
        .iter()
        .map(|pattern| Stage {
            pattern: Pattern::parse(pattern).expect("parses"),
            keep_gaps: true,
        })
        .collect();
    let scanner = chain::Scanner::recognize(&stages).expect("the chain takes the scanner");
    let programs: Vec<Stage> = stages
        .iter()
        .map(|stage| Stage {
            pattern: stage.pattern.program_only(),
            keep_gaps: stage.keep_gaps,
        })
        .collect();
    let fixtures = fixture_texts();
    let rounds = fuzz_rounds();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let check = |text: &str| {
        let mut ours = Vec::new();
        scanner.pieces(text, &mut |start, end| ours.push((start, end)));
        assert_eq!(ours, staged(&programs, text), "text {text:?}");
    };
    for text in &fixtures {
        check(text);
    }
    for _ in 0..rounds {
        check(&fuzz_text(&mut state, &fixtures));
    }
}

#[test]
fn sentencepiece_shapes_outside_the_path_are_declined() {
    let mut state = 0xDEC1_1DED_5EED_0001u64;
    let shapes = sentencepiece_shapes();
    let declined = |tokenizer: &HfTokenizer| {
        NativeEncoder::from_tokenizer(tokenizer)
            .expect_err("declined")
            .to_string()
    };
    // A normalizer other than the Replace.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[0], &mut state);
    tokenizer
        .with_normalizer(Some(NormalizerWrapper::NFC(NFC)))
        .expect("normalizer");
    assert_eq!(
        declined(&tokenizer),
        "a byte-fallback vocabulary with a normalizer other than Replace(space, marker) or none"
    );
    // Neither the Replace nor a Metaspace.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[1], &mut state);
    tokenizer
        .with_normalizer(None::<NormalizerWrapper>)
        .expect("normalizer");
    assert_eq!(
        declined(&tokenizer),
        "a byte-fallback vocabulary with neither a Replace normalizer nor a Metaspace \
         pre-tokenizer"
    );
    // A Metaspace that splits, and one under the Replace.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[2], &mut state);
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', PrependScheme::First, true)));
    assert_eq!(
        declined(&tokenizer),
        "a byte-fallback vocabulary with the pre-tokenizer Metaspace (only none or a Split on \
         the space after the Replace normalizer, or Metaspace without split)"
    );
    let mut tokenizer = sentencepiece_tokenizer(&shapes[1], &mut state);
    tokenizer.with_pre_tokenizer(Some(Metaspace::new(
        '\u{2581}',
        PrependScheme::Never,
        false,
    )));
    assert!(declined(&tokenizer)
        .starts_with("a byte-fallback vocabulary with the pre-tokenizer Metaspace"));
    // A Split on something else behind the Replace.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[0], &mut state);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
        Split::new(
            SplitPattern::String("\n".to_owned()),
            SplitDelimiterBehavior::MergedWithPrevious,
            false,
        )
        .expect("split"),
    )));
    assert!(
        declined(&tokenizer).starts_with("a byte-fallback vocabulary with the pre-tokenizer Split")
    );
    // An added token with a space, matched after the Replace.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[0], &mut state);
    tokenizer
        .add_tokens([AddedToken::from("end of text", false).normalized(true)])
        .expect("added token");
    assert_eq!(
        declined(&tokenizer),
        "a byte-fallback vocabulary with an added token matched after the Replace normalizer \
         that holds a space or the word marker"
    );
    // The same token under Metaspace is matched on the same text either way.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[2], &mut state);
    tokenizer
        .add_tokens([AddedToken::from("end of text", false).normalized(true)])
        .expect("added token");
    assert!(NativeEncoder::from_tokenizer(&tokenizer).is_ok());
    // ignore_merges.
    let mut tokenizer = sentencepiece_tokenizer(&shapes[0], &mut state);
    let mut model = sentencepiece_model(&mut state);
    model.ignore_merges = true;
    tokenizer.with_model(model);
    assert_eq!(
        declined(&tokenizer),
        "a byte-fallback vocabulary with ignore_merges (the whole text is looked up)"
    );
    // A byte token missing, and a token that spells one inside it.
    let small = |extra: &[&str], drop: &str| {
        let mut vocab = Vocab::default();
        for token in ["<unk>", "a", "b", "\u{2581}"]
            .into_iter()
            .chain(
                (0..=255u8)
                    .map(|byte| format!("<0x{byte:02X}>"))
                    .collect::<Vec<_>>()
                    .iter()
                    .map(String::as_str),
            )
            .chain(extra.iter().copied())
        {
            if token != drop {
                let id = vocab.len() as u32;
                vocab.insert(token.to_owned(), id);
            }
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Merges::new())
            .byte_fallback(true)
            .unk_token("<unk>".to_owned())
            .build()
            .expect("a small byte-fallback BPE");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer
            .with_normalizer(Some(
                Replace::new(" ", "\u{2581}").expect("a Replace normalizer"),
            ))
            .expect("normalizer");
        tokenizer
    };
    assert_eq!(declined(&small(&[], "<0x7F>")), "byte 0x7f has no token");
    assert_eq!(
        declined(&small(&["a<0x41>"], "")),
        "the token \"a<0x41>\" spells a byte token inside it"
    );
    assert!(NativeEncoder::from_tokenizer(&small(&["ab"], "")).is_ok());
}

/// A piece no vocabulary token cuts can be the whole text: the thread's
/// scratch list of initial symbols grows to it for the merge and is handed
/// back afterwards, so a thread keeps `MERGE_SCRATCH_SYMBOLS` of scratch,
/// not the longest piece it ever met.
#[test]
fn a_long_uncut_piece_leaves_the_thread_a_bounded_scratch() {
    let mut vocab = Vocab::default();
    let add = |vocab: &mut Vocab, token: String| {
        if !vocab.contains_key(&token) {
            let id = vocab.len() as u32;
            vocab.insert(token, id);
        }
    };
    add(&mut vocab, "<unk>".to_owned());
    for byte in 0..=255u8 {
        add(&mut vocab, format!("<0x{byte:02X}>"));
    }
    for token in [
        "\u{4e2d}",
        "\u{6587}",
        "\u{4e2d}\u{6587}",
        "\u{6587}\u{4e2d}",
    ] {
        add(&mut vocab, token.to_owned());
    }
    let merges: Merges = vec![
        ("\u{4e2d}".to_owned(), "\u{6587}".to_owned()),
        ("\u{6587}".to_owned(), "\u{4e2d}".to_owned()),
    ];
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("a byte-fallback BPE");
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer
        .with_normalizer(Some(
            Replace::new(" ", "\u{2581}").expect("a Replace normalizer"),
        ))
        .expect("normalizer");
    let native = seeded(&tokenizer);
    // Every adjacent pair of bytes of the text is in a vocabulary token, so
    // the text is one piece, merged on this thread.
    let text = "\u{4e2d}\u{6587}".repeat(4 * MERGE_SCRATCH_SYMBOLS);
    assert_eq!(
        native.units(&text, true, true).as_slice().len(),
        1,
        "nothing cuts the text"
    );
    let expected = tokenizer
        .encode(text.as_str(), false)
        .expect("tokenizers encodes")
        .get_ids()
        .to_vec();
    assert_eq!(
        native.encode(Some(tokenizer.get_model()), &text, false),
        Ok(expected)
    );
    let scratch = native.thread_piece_cache().borrow().symbols.capacity();
    assert!(
        scratch <= MERGE_SCRATCH_SYMBOLS,
        "the thread kept a scratch of {scratch} symbols"
    );
}
