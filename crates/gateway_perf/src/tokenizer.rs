//! A small, self-contained tokenizer directory for the gateway perf bed.
//!
//! The bed needs a tokenizer with a chat template so the gateway renders and encodes
//! every request the way it does in production; fetching one from a model hub makes a
//! CI job depend on the network and on a third party's rate limits. This writes one
//! from nothing, in the shape the gateway's own encoder takes (the shape of the common
//! byte-level BPE tokenizers: a BPE model over the 256 byte symbols, the ByteLevel
//! pre-tokenizer with its word-splitting pattern, an NFC normalizer, a ByteLevel decoder),
//! with merges that make every word of the load client's deterministic prompts, tool
//! schemas and scripted answers one token, in both spellings (after a space and bare)
//! and in title case. Anything else encodes to its bytes: no unknown token, every text
//! decodes back to itself. Role markers are added special tokens.
//!
//! The numbers measured on it are relative (base against head on the same tokenizer).
//! A real tokenizer directory can be passed instead (`--tokenizer <dir or hub id>`) when
//! absolute per-request costs with a production template are the question.

use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::Path,
};

use anyhow::Result;
use serde_json::{json, Map, Value};

/// The load client's prompt vocabulary (crates/mock_worker/src/bin/stress_load.rs,
/// `VOCAB_TEXT`), kept in one place here: the bed's prompts encode without an unknown.
pub const PROMPT_WORDS: &str = "the of and to in is that for it as with on be by this are from or an at which have not were but \
one all their there when who will more if out so what up its about into than them can only other \
new some could time these two may then do first any my now such like our over man me even most \
made after also did many before must through back years where much your way well down should \
because each just those people how too little state good very make world still own see men work \
long get here between both life being under never day same another know while last might us \
great old year off come since against go came right used take three states himself few house use \
during without again place around however home small found thought went say part once general \
high upon school every does got united left number course war until always away something fact \
though water less public put think almost hand enough far took head yet government system better \
set told nothing night end why called eyes find going look asked later knew point next program \
city business give group toward young days let room president side social given present several \
order national second possible rather per face among form important often things looked early \
white case become large need big four within felt along children saw best church ever least \
power development light thing seemed family interest want members mind country area others \
turned although open service certain kind problem began different door thus help means sense \
whole matter perhaps itself times human law line above name example action company local show \
whether five history gave today either act feet across taken past quite anything seen having \
death week body word field experience really class already money free information table value \
cost market growth design test data model result process report rate level change read write \
file code error check build run";

/// Words of the request bodies around the prompts: the roles, the tool schemas and the
/// tool results the load client builds, the JSON-schema answer format, the scripted
/// answers, and the template's own words.
pub const STRUCTURE_WORDS: &str = "
system user assistant tool tools Tools developer content role function type object string integer
array items properties required additionalProperties minimum maximum enum format description
parameters arguments name key fields limit filter since status open closed any ok unsure answer
confidence strict schema json_schema date time record returns looks asked lookup Tool call id
call_id tool_call_id tool_calls a b c k true false null quick brown fox jumps over lazy dog
The An A is was ready done fine lookup_0 lookup_1 lookup_2 lookup_3 lookup_4 lookup_5 lookup_6
lookup_7 lookup_8 lookup_9 lookup_10 lookup_11 lookup_12 lookup_13 lookup_14 lookup_15 mock
model m1 text image_url input_text output_text
";

pub const SPECIAL: [&str; 3] = ["<|start|>", "<|end|>", "<|pad|>"];
/// The byte-level symbol of a space (the GPT-2 byte alphabet).
pub const SPACE: &str = "Ġ";
/// The word-splitting pattern of the common byte-level tokenizers: contractions, a
/// letter run with one leading non-letter, single digits, punctuation runs, newlines,
/// then spaces (the space before a word stays with the word).
pub const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub const CHAT_TEMPLATE: &str = concat!(
    "{%- if tools %}",
    r"{{ '<|start|>system\n' }}",
    r"{%- if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{{ '\n' }}{%- endif %}",
    r"{{ 'Tools:\n' }}",
    r"{%- for tool in tools %}{{ tool | tojson }}{{ '\n' }}{%- endfor %}",
    r"{{ '<|end|>\n' }}",
    "{%- endif %}",
    "{%- for message in messages %}",
    "{%- if not (message['role'] == 'system' and tools) %}",
    r"{{ '<|start|>' + message['role'] + '\n' }}",
    "{%- if message['content'] is string %}{{ message['content'] }}",
    "{%- elif message['content'] is iterable %}",
    "{%- for part in message['content'] %}",
    "{%- if part['type'] == 'text' %}{{ part['text'] }}{%- endif %}",
    "{%- endfor %}",
    "{%- endif %}",
    "{%- if message['tool_calls'] %}",
    r"{%- for call in message['tool_calls'] %}{{ '\n' }}{{ call['function'] | tojson }}{%- endfor %}",
    "{%- endif %}",
    r"{{ '<|end|>\n' }}",
    "{%- endif %}",
    "{%- endfor %}",
    r"{%- if add_generation_prompt %}{{ '<|start|>assistant\n' }}{%- endif %}",
);

/// The 256 byte symbols of byte-level BPE (GPT-2's `bytes_to_unicode`): printable
/// ASCII and Latin-1 letters stand for themselves, the other bytes get code points
/// from 256 up, in byte order.
pub fn byte_alphabet() -> Vec<String> {
    let keep = |b: u32| {
        (b'!'.into()..=b'~'.into()).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
    };
    let mut next = 256u32;
    (0u32..256)
        .map(|b| {
            let code = if keep(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(code).map(String::from).unwrap_or_default()
        })
        .collect()
}

fn to_symbols(word: &str, alphabet: &[String]) -> Vec<String> {
    word.bytes()
        .map(|b| alphabet[usize::from(b)].clone())
        .collect()
}

/// Python's `str.capitalize()`: the first character upper, the rest lower.
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

/// Every word that gets a merge chain: the prompt vocabulary, the structure words,
/// the extras, each also in title case; sorted so the output is deterministic.
pub fn words_of(extra_words: &[String]) -> Vec<String> {
    let mut words: BTreeSet<String> = PROMPT_WORDS
        .split_whitespace()
        .chain(STRUCTURE_WORDS.split_whitespace())
        .map(str::to_string)
        .chain(extra_words.iter().cloned())
        .collect();
    let title: Vec<String> = words.iter().map(|w| capitalize(w)).collect();
    words.extend(title);
    words.into_iter().collect()
}

/// The vocabulary and the ordered merges: for every word, after a space (the
/// common spelling inside a text) and bare (the first word of a text), the chain
/// that joins its byte symbols left to right. A BPE applies the lowest-ranked pair
/// present, so the order of the chains decides what wins where words overlap: every
/// spaced spelling's chain before every bare one (a chain of a spaced word only
/// ever meets its own prefixes), and longer words before shorter ones within each
/// group, so a word's own chain outranks the chains of the shorter words inside it.
/// Every spaced spelling is then one token; a bare word that contains a prefix of
/// a longer listed word may take two, at the start of a text only.
pub fn merges_for(words: &[String], alphabet: &[String]) -> (Vec<String>, Vec<(String, String)>) {
    let mut by_length: Vec<&String> = words.iter().collect();
    by_length.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let spaced = by_length.iter().map(|w| {
        let mut chain = vec![SPACE.to_string()];
        chain.extend(to_symbols(w, alphabet));
        chain
    });
    let bare = by_length.iter().map(|w| to_symbols(w, alphabet));
    let mut vocab: Vec<String> = alphabet.to_vec();
    let mut seen: HashSet<String> = vocab.iter().cloned().collect();
    let mut merges: Vec<(String, String)> = Vec::new();
    let mut seen_merges: HashSet<(String, String)> = HashSet::new();
    for chain in spaced.chain(bare) {
        for position in 0..chain.len().saturating_sub(1) {
            let left: String = chain[..=position].concat();
            let right = chain[position + 1].clone();
            let pair = (left, right);
            if !seen_merges.insert(pair.clone()) {
                continue;
            }
            let joined = format!("{}{}", pair.0, pair.1);
            merges.push(pair);
            if seen.insert(joined.clone()) {
                vocab.push(joined);
            }
        }
    }
    (vocab, merges)
}

/// `tokenizer.json`.
pub fn tokenizer_json(vocab: &[String], merges: &[(String, String)]) -> Value {
    let ids: Map<String, Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), json!(i)))
        .collect();
    let added: Vec<Value> = SPECIAL
        .iter()
        .enumerate()
        .map(|(n, token)| {
            json!({
                "id": vocab.len() + n,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": SPLIT_PATTERN}, "behavior": "Isolated", "invert": false},
                byte_level,
            ],
        },
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": ids,
            "merges": merges.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>(),
        },
    })
}

/// `tokenizer_config.json`.
pub fn tokenizer_config() -> Value {
    json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 262144,
        "add_bos_token": false,
        "add_prefix_space": false,
        "bos_token": null,
        "eos_token": "<|end|>",
        "pad_token": "<|pad|>",
        "unk_token": null,
        "additional_special_tokens": ["<|start|>", "<|end|>"],
        "clean_up_tokenization_spaces": false,
        "split_special_tokens": false,
        "chat_template": CHAT_TEMPLATE,
    })
}

/// Writes the directory; returns the vocabulary (the specials last).
pub fn write(out: &Path, extra_words: &[String]) -> Result<Vec<String>> {
    fs::create_dir_all(out)?;
    let alphabet = byte_alphabet();
    let (vocab, merges) = merges_for(&words_of(extra_words), &alphabet);
    let pretty = |v: &Value| -> Result<String> { Ok(serde_json::to_string_pretty(v)? + "\n") };
    fs::write(
        out.join("tokenizer.json"),
        pretty(&tokenizer_json(&vocab, &merges))?,
    )?;
    fs::write(
        out.join("tokenizer_config.json"),
        pretty(&tokenizer_config())?,
    )?;
    fs::write(
        out.join("special_tokens_map.json"),
        pretty(&json!({"eos_token": "<|end|>", "pad_token": "<|pad|>"}))?,
    )?;
    let end_id = vocab.len() + SPECIAL.iter().position(|s| *s == "<|end|>").unwrap_or(1);
    fs::write(
        out.join("config.json"),
        pretty(
            &json!({"model_type": "synthetic", "eos_token_id": end_id, "vocab_size": vocab.len() + SPECIAL.len()}),
        )?,
    )?;
    let mut tokens = vocab;
    tokens.extend(SPECIAL.iter().map(|s| (*s).to_string()));
    Ok(tokens)
}

#[cfg(test)]
#[path = "../tests/unit/tokenizer/tests.rs"]
mod tests;
