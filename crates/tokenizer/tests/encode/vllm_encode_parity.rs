//! The backend against `tokenizers` on real vocabularies.
//!
//! Opt-in: `SMG_VLLM_ENCODE_TOKENIZER_DIRS` lists tokenizer directories
//! (comma-separated, each with a `tokenizer.json`). For each the backend
//! takes, the crate's tokenizer must give the same ids as the `tokenizers`
//! pipeline the crate configures from the same files (the file with the
//! special-token rule of `tokenizer_config.json`, see
//! [`HuggingFaceTokenizer::tokenizers_pipeline`]; the file read alone would
//! show a configured BOS as a difference with special tokens on), over
//! mixed-script random text, texts with the tokenizer's own special tokens,
//! and long prompts, with and without special tokens. Without the variable
//! the test prints a notice and passes; the in-repo unit tests cover the
//! same comparison on synthetic vocabularies, and the bellwether fixture
//! test covers recorded prompts.
#![expect(clippy::print_stdout, reason = "the test reports what it compared")]

use llm_tokenizer::{
    create_tokenizer,
    traits::{Encoder, Encoding},
    EncodePath, HuggingFaceTokenizer,
};

const DIRS_ENV: &str = "SMG_VLLM_ENCODE_TOKENIZER_DIRS";

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

const TEXTS: &[&str] = &[
    "",
    " ",
    "\n\n",
    "Hello, world!",
    "I'm testing: 12345\nsecond line.",
    "Caf\u{E9} 中文🙂 — naïve",
    "  leading  and trailing  ",
    "<|im_start|>assistant\nHello<|im_end|>",
    "<|im_start|>user\n<tool_call>{\"a\": 1}</tool_call><|im_end|>\n",
    "WE'LL see what IT'S worth; they'd know, I'VE heard, you're, he's, isn'T",
    "x'ſ y'ſ",
    "1234567890 ١٢٣ ๑๒๓๔",
    "日本語のテキスト、한국어 텍스트, 中文文本。",
    "ครับ ผม ชื่อ นาย คำ ไทย",
    "कृपया यह पढ़ें और समझें",
    "fn main() { println!(\"hi\"); }\n\n\n// done\r\n",
    "a\tb\u{A0}c\u{3000}d\u{2028}e",
    "<｜begin▁of▁sentence｜>hello<｜User｜>q<｜Assistant｜>",
    "[e~[hello]~b] and [e~ half",
    "<|endoftext|>[gMASK]<sop><|user|>\nhi<|assistant|>\n",
];

fn pseudo_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_text(state: &mut u64, len: usize, specials: &[String]) -> String {
    let mut text = String::new();
    for _ in 0..len {
        let roll = pseudo_random(state);
        if !specials.is_empty() && roll.is_multiple_of(13) {
            let token = &specials[(roll / 13) as usize % specials.len()];
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

#[test]
fn the_backend_matches_tokenizers_on_real_vocabularies() {
    let Ok(dirs) = std::env::var(DIRS_ENV) else {
        println!("{DIRS_ENV} is not set: skipping the real-vocabulary comparison");
        return;
    };
    let mut state = 0x6A09_E667_F3BC_C908u64;
    for dir in dirs.split(',').map(str::trim).filter(|d| !d.is_empty()) {
        let ours = create_tokenizer(dir).expect("the crate loads the tokenizer");
        let Some(ours) = ours.as_any().downcast_ref::<HuggingFaceTokenizer>() else {
            println!(
                "{dir}: no Hugging Face tokenizer (a tiktoken vocabulary), nothing to compare"
            );
            continue;
        };
        if ours.encode_path() != EncodePath::Vllm {
            println!("{dir}: the backend declined the tokenizer, nothing to compare");
            continue;
        }
        let on_tokenizers =
            HuggingFaceTokenizer::from_file_on_tokenizers(&format!("{dir}/tokenizer.json"))
                .expect("the crate loads the file on tokenizers");
        let reference = on_tokenizers
            .tokenizers_pipeline()
            .expect("the pipeline alone was asked for");
        let specials: Vec<String> = reference
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .map(|token| token.content.clone())
            .collect();
        let mut compared = 0usize;
        let mut check = |text: &str| {
            for add_special_tokens in [false, true] {
                let encoding = ours.encode(text, add_special_tokens).expect("encode");
                let expected = reference
                    .encode(text, add_special_tokens)
                    .expect("reference encode");
                assert_eq!(
                    encoding.token_ids(),
                    expected.get_ids(),
                    "{dir}: {text:?} (special tokens {add_special_tokens})"
                );
                assert!(
                    matches!(encoding, Encoding::Plain(_)),
                    "{dir}: {text:?} (special tokens {add_special_tokens}) left the backend"
                );
            }
            compared += 1;
        };
        for text in TEXTS {
            check(text);
        }
        for round in 0..3000 {
            let text = random_text(&mut state, 1 + round % 60, &specials);
            check(&text);
        }
        // Long prompts: the random text repeated, with special tokens between
        // turns, well past any per-piece cache.
        let turn = random_text(&mut state, 2000, &[]);
        let long: String = (0..40)
            .map(|i| {
                format!(
                    "{}\n{turn}\n",
                    specials
                        .get(i % specials.len().max(1))
                        .map_or("", String::as_str)
                )
            })
            .collect();
        check(&long);
        // A long document without special tokens: one text the pre-tokenizer
        // cuts into many pieces, past the size of any pretoken cache.
        let document: String = (0..60)
            .map(|_| random_text(&mut state, 2000, &[]))
            .collect::<Vec<_>>()
            .join(" ");
        check(&document);
        println!("{dir}: {compared} texts compared through the backend");
        assert!(compared > 0, "{dir}: nothing compared");
    }
}
