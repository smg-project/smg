//! The Lark spelling of a grammar for the llguidance grammar backend.
//!
//! That backend reads every added token of the worker's vocabulary as a special token: no byte
//! literal and no text lexeme matches one, a token reference does. So a fixed word the vocabulary
//! has as a token is spelled as the token beside its bytes, `(<tool_call> | "<tool_call>")`; free
//! text is a text lexeme interleaved with the added tokens it may hold (every added token except
//! one equal to, holding or standing inside a word the text may not hold), `T (A T)*` with `A` a
//! range of token ids; and a fixed word that follows free text is folded into the text's head,
//! since the lexer takes the longest match and a greedy lexeme would run past the word: the text
//! ends at the word's bytes through a lazy lexeme, or at the word's leading token by itself; a
//! token node that follows the text is reached by the token alone, never by its bytes, so a
//! vocabulary without the token refuses the grammar. The words a text may not hold are the `& ~`
//! operators of its lexeme's regex. A token that begins or ends a word the text may not hold,
//! spelled across several tokens, gets one transition in every state of the text: the text
//! before it may not end with the word's leading bytes, or, in the state of the token the word
//! begins with, may not be the bytes between the two tokens; the text after a token the word
//! begins with may not start with the word's remaining bytes, a later token of the word read as
//! its bytes (the head spells a later token as itself or its bytes). A token the text leaves out
//! (one equal to a word, holding one, inside one between bytes, or beside another token in one)
//! is read as bytes wherever a word holds it, so it never names a state. The spelling is one Lark
//! text, what the engines' grammar fields take; a node the Lark cannot carry refuses the grammar
//! by name.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
};

use serde_json::Value;

use super::super::{Bound, Grammar, Tag};
use crate::format::Whitespace;

/// The added tokens of a worker's vocabulary, by text, with its end-of-sequence token: the tokens
/// the llguidance backend reads as special, which the Lark names wherever free text may hold them
/// and wherever a fixed word is one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AddedTokens {
    ids: HashMap<String, u32>,
    /// Every text, longest first: the order a word is scanned for the tokens it holds.
    longest_first: Vec<String>,
    eos: Option<u32>,
}

impl AddedTokens {
    /// The tokens from (text, id) pairs and the end-of-sequence id; an empty text is dropped, a
    /// repeated text keeps its first id.
    pub fn new(tokens: impl IntoIterator<Item = (String, u32)>, eos: Option<u32>) -> Self {
        let mut ids: HashMap<String, u32> = HashMap::new();
        for (text, id) in tokens {
            if !text.is_empty() {
                ids.entry(text).or_insert(id);
            }
        }
        let mut longest_first: Vec<String> = ids.keys().cloned().collect();
        longest_first.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        Self {
            ids,
            longest_first,
            eos,
        }
    }

    /// The token's id, when the vocabulary has `text` as one token.
    pub fn id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// How many tokens the vocabulary adds.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the vocabulary adds no token.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// What the Lark cannot carry, by name: the grammar is refused for the backend, never loosened in
/// silence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inexpressible(pub String);

impl fmt::Display for Inexpressible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Inexpressible {}

/// A grammar's Lark text, with the loosenings the spelling took, each named: a JSON schema's
/// excluded words are not held out of its strings (the backend's JSON has no such rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lark {
    pub text: String,
    pub notes: Vec<String>,
}

impl Grammar {
    /// The grammar as one Lark text for the llguidance backend, spelled for the vocabulary's
    /// added `tokens`; the error names what the Lark cannot carry (a JSON schema in a style other
    /// than JSON, a schema that accepts nothing, an embedded EBNF grammar, a token the vocabulary
    /// lacks).
    pub fn to_lark(&self, tokens: &AddedTokens) -> Result<Lark, Inexpressible> {
        let mut lowering = Lowering::new(tokens);
        let (start, _) = lowering.lower(self, None, &[])?;
        let text = lowering.text(&start);
        Ok(Lark {
            text,
            notes: lowering.notes,
        })
    }
}

/// The characters a Rust regex reads as syntax (and the Lark's own `/`, `#`, `&`, `~`).
const REGEX_META: &str = "\\.+*?()|[]{}^$/#&~-";

fn regex_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if REGEX_META.contains(c) => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\x7f' => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// A regex node as the Lark's `/.../` terminal: a slash the pattern does not escape is escaped;
/// a backslash and the character after it are kept as they are, so an escaped slash stays one.
/// An empty pattern is the empty literal: `//` opens a Lark comment, and an empty pattern
/// matches the empty text, as the empty literal does.
fn regex_lark(pattern: &str) -> String {
    if pattern.is_empty() {
        return "\"\"".to_string();
    }
    let mut out = String::with_capacity(pattern.len() + 2);
    out.push('/');
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                out.push('\\');
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            }
            '/' => out.push_str("\\/"),
            ch => out.push(ch),
        }
    }
    out.push('/');
    out
}

/// A Lark string literal in JSON escape syntax, non-ASCII kept as it is.
fn literal(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string())
}

/// Token ids as the backend's inclusive range list, `<[a-b,c,d-e]>`.
fn ranges(ids: &[u32]) -> String {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let mut j = i;
        while j + 1 < ids.len() && ids[j + 1] == ids[j] + 1 {
            j += 1;
        }
        parts.push(if i == j {
            ids[i].to_string()
        } else {
            format!("{}-{}", ids[i], ids[j])
        });
        i = j + 1;
    }
    format!("<[{}]>", parts.join(","))
}

/// Whether a token's text is its own reference in the Lark, `<name>`.
fn is_token_shaped(text: &str) -> bool {
    let Some(inner) = text.strip_prefix('<').and_then(|t| t.strip_suffix('>')) else {
        return false;
    };
    !inner.is_empty()
        && !inner
            .chars()
            .any(|c| c == '<' || c == '>' || c.is_whitespace())
}

fn join_escaped<'a>(words: impl IntoIterator<Item = &'a str>) -> String {
    words
        .into_iter()
        .map(regex_escape)
        .collect::<Vec<_>>()
        .join("|")
}

fn sorted_unique(words: &[String]) -> BTreeSet<&str> {
    words.iter().map(String::as_str).collect()
}

/// The texts a bound names, the empty ones left out.
fn bound_words(end: &Bound) -> Vec<String> {
    match end {
        Bound::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![text.clone()]
            }
        }
        Bound::Texts(texts) => texts.iter().filter(|t| !t.is_empty()).cloned().collect(),
    }
}

/// One part of the fixed word free text runs up to: bytes the tree writes as a constant or a
/// tag's begin (spelled as bytes, or as the tokens the vocabulary has for pieces of them), or a
/// token node, spelled by the token alone and never by its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Part {
    Bytes(String),
    Token(String),
}

/// A fixed word as its parts, adjacent byte parts joined.
type Word = Vec<Part>;

/// `parts` with the empty ones dropped and adjacent byte parts joined.
fn merge_parts(parts: Vec<Part>) -> Word {
    let mut out: Word = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            Part::Bytes(text) | Part::Token(text) if text.is_empty() => {}
            Part::Bytes(text) => match out.last_mut() {
                Some(Part::Bytes(last)) => last.push_str(&text),
                _ => out.push(Part::Bytes(text)),
            },
            token @ Part::Token(_) => out.push(token),
        }
    }
    out
}

/// The word's text, a token part by the token's text.
fn word_text(parts: &[Part]) -> String {
    parts
        .iter()
        .map(|part| match part {
            Part::Bytes(text) | Part::Token(text) => text.as_str(),
        })
        .collect()
}

/// A word the tree writes as bytes (a tag's end, a trigger) as parts; empty for an empty text.
fn bytes_word(text: &str) -> Word {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![Part::Bytes(text.to_string())]
    }
}

fn bytes_words(words: &[String]) -> Vec<Word> {
    words.iter().map(|word| bytes_word(word.as_str())).collect()
}

/// The fixed word a node begins with, when the node is anchored on one, as parts: a constant's
/// text, a tag's begin, a token node (kept as a token: it is spelled by the token alone), the
/// leading constants and tokens of a sequence.
fn word_of(node: &Grammar) -> Option<Word> {
    let parts = match node {
        Grammar::ConstString(value) => vec![Part::Bytes(value.clone())],
        Grammar::Token(text) => vec![Part::Token(text.clone())],
        Grammar::Tag(tag) => {
            if tag.begin.is_empty() {
                return word_of(&tag.content);
            }
            vec![Part::Bytes(tag.begin.clone())]
        }
        Grammar::Sequence(elements) => {
            let mut parts: Vec<Part> = Vec::new();
            for element in elements {
                match element {
                    Grammar::ConstString(value) => {
                        if !value.is_empty() {
                            parts.push(Part::Bytes(value.clone()));
                        }
                    }
                    Grammar::Token(text) => parts.push(Part::Token(text.clone())),
                    other => {
                        match word_of(other) {
                            Some(head) if !matches!(other, Grammar::Tag(_)) => parts.extend(head),
                            Some(head) if parts.is_empty() => return Some(head),
                            _ => {}
                        }
                        break;
                    }
                }
            }
            parts
        }
        _ => return None,
    };
    let parts = merge_parts(parts);
    (!parts.is_empty()).then_some(parts)
}

/// `node` with its leading word `parts` taken away (the head of a text consumed them); `None`
/// when the node does not begin with them. A token part is consumed by a token node alone.
fn after_word(node: &Grammar, parts: &[Part]) -> Option<Grammar> {
    let parts = merge_parts(parts.to_vec());
    let [first, more @ ..] = parts.as_slice() else {
        return Some(node.clone());
    };
    let alone = more.is_empty();
    match node {
        Grammar::ConstString(value) => match first {
            Part::Bytes(bytes) if alone => value
                .strip_prefix(bytes.as_str())
                .map(|rest| Grammar::ConstString(rest.to_string())),
            _ => None,
        },
        Grammar::Token(text) => match first {
            Part::Token(token) if alone && token == text => {
                Some(Grammar::ConstString(String::new()))
            }
            _ => None,
        },
        Grammar::Tag(tag) => {
            if tag.begin.is_empty() {
                return after_word(&tag.content, &parts).map(|inner| {
                    Grammar::Tag(Tag {
                        begin: String::new(),
                        content: Box::new(inner),
                        end: tag.end.clone(),
                    })
                });
            }
            match first {
                Part::Bytes(bytes) if alone => tag.begin.strip_prefix(bytes.as_str()).map(|rest| {
                    Grammar::Tag(Tag {
                        begin: rest.to_string(),
                        content: tag.content.clone(),
                        end: tag.end.clone(),
                    })
                }),
                _ => None,
            }
        }
        Grammar::Sequence(elements) if !elements.is_empty() => {
            let mut remaining: Vec<Part> = parts;
            let mut elements: &[Grammar] = elements;
            let mut out: Vec<Grammar> = Vec::new();
            while !remaining.is_empty() && !elements.is_empty() {
                let element = &elements[0];
                let part = remaining[0].clone();
                match element {
                    Grammar::ConstString(value) => {
                        if value.is_empty() {
                            elements = &elements[1..];
                            continue;
                        }
                        let Part::Bytes(bytes) = part else {
                            return None;
                        };
                        if let Some(rest) = bytes.strip_prefix(value.as_str()) {
                            remaining.remove(0);
                            if !rest.is_empty() {
                                remaining.insert(0, Part::Bytes(rest.to_string()));
                            }
                            elements = &elements[1..];
                            continue;
                        }
                        let tail = value.strip_prefix(bytes.as_str())?;
                        out.push(Grammar::ConstString(tail.to_string()));
                        remaining.remove(0);
                        elements = &elements[1..];
                        if !remaining.is_empty() {
                            return None;
                        }
                        break;
                    }
                    Grammar::Token(text) => match part {
                        Part::Token(token) if token == *text => {
                            remaining.remove(0);
                            elements = &elements[1..];
                        }
                        _ => return None,
                    },
                    _ => {
                        out.push(after_word(element, &remaining)?);
                        remaining.clear();
                        elements = &elements[1..];
                    }
                }
            }
            if !remaining.is_empty() {
                return None;
            }
            out.extend(elements.iter().cloned());
            Some(Grammar::Sequence(out))
        }
        _ => None,
    }
}

/// A piece of a fixed word: bytes, a piece the vocabulary has as one token (spelled as the token
/// or its bytes), or a token node of the tree (spelled by the token alone).
enum Piece {
    Bytes(String),
    Token(String),
    Node(String),
}

impl Piece {
    fn text(&self) -> &str {
        match self {
            Self::Bytes(text) | Self::Token(text) | Self::Node(text) => text,
        }
    }
}

/// The vocabulary's added tokens against the words free text may not hold.
#[derive(Default)]
struct Classes {
    /// The tokens the text may hold anywhere.
    free: Vec<u32>,
    /// Token text -> the remaining bytes of the forbidden words it begins, a later token of the
    /// word as its bytes: the text after the token may not start with one.
    prefix: BTreeMap<String, Vec<String>>,
    /// Token text -> the leading bytes of the forbidden words it ends, each with the token that
    /// stood before those bytes in the word (`None` when none did): the text before the token may
    /// not end with one, or, in the state after that token, may not be one (the bytes between the
    /// two tokens are the whole lexeme there).
    suffix: Suffixes,
}

type Suffixes = BTreeMap<String, Vec<(String, Option<String>)>>;

/// The rule names of free text, one per state: the start, and after each token that begins a
/// forbidden word.
type StateNames = BTreeMap<Option<String>, String>;

struct Lowering<'a> {
    tokens: &'a AddedTokens,
    rules: Vec<String>,
    /// Terminal body -> name.
    terminals: HashMap<String, String>,
    terminal_lines: Vec<String>,
    /// (excludes, forbid) -> the names of the free-text rules.
    text_memo: HashMap<(Vec<String>, Vec<String>), StateNames>,
    /// (excludes, forbid, word) -> the name of the head rule.
    head_memo: HashMap<(Vec<String>, Vec<String>, Word), String>,
    notes: Vec<String>,
    n: usize,
}

impl<'a> Lowering<'a> {
    fn new(tokens: &'a AddedTokens) -> Self {
        Self {
            tokens,
            rules: Vec::new(),
            terminals: HashMap::new(),
            terminal_lines: Vec::new(),
            text_memo: HashMap::new(),
            head_memo: HashMap::new(),
            notes: Vec::new(),
            n: 0,
        }
    }

    // -- names ------------------------------------------------------------------------------

    fn fresh(&mut self, prefix: &str) -> String {
        self.n += 1;
        format!("{prefix}{}", self.n)
    }

    fn terminal(&mut self, body: String) -> String {
        if let Some(name) = self.terminals.get(&body) {
            return name.clone();
        }
        let name = format!("T{}", self.terminals.len() + 1);
        self.terminal_lines.push(format!("{name}: {body}"));
        self.terminals.insert(body, name.clone());
        name
    }

    fn rule(&mut self, body: &str, options: Option<&str>, prefix: &str) -> String {
        let name = self.fresh(prefix);
        let head = match options {
            Some(options) => format!("{name}[{options}]"),
            None => name.clone(),
        };
        self.rules.push(format!("{head}: {body}"));
        name
    }

    // -- tokens and literals ----------------------------------------------------------------

    fn token_ref(&self, text: &str) -> Result<String, Inexpressible> {
        let id = self
            .tokens
            .id(text)
            .ok_or_else(|| Inexpressible(format!("the vocabulary has no token {text:?}")))?;
        if is_token_shaped(text) {
            Ok(text.to_string())
        } else {
            Ok(format!("<[{id}]>"))
        }
    }

    /// A fixed word split into the pieces the vocabulary has as one token (the longest at every
    /// position) and the bytes between them.
    fn segments(&self, text: &str) -> Vec<Piece> {
        if text.is_empty() {
            return Vec::new();
        }
        let candidates: Vec<&str> = self
            .tokens
            .longest_first
            .iter()
            .map(String::as_str)
            .filter(|t| text.contains(t))
            .collect();
        if candidates.is_empty() {
            return vec![Piece::Bytes(text.to_string())];
        }
        let mut out = Vec::new();
        let mut bytes = String::new();
        let mut at = 0;
        while at < text.len() {
            let rest = &text[at..];
            if let Some(hit) = candidates.iter().find(|t| rest.starts_with(*t)) {
                if !bytes.is_empty() {
                    out.push(Piece::Bytes(std::mem::take(&mut bytes)));
                }
                out.push(Piece::Token(hit.to_string()));
                at += hit.len();
            } else if let Some(ch) = rest.chars().next() {
                bytes.push(ch);
                at += ch.len_utf8();
            } else {
                break;
            }
        }
        if !bytes.is_empty() {
            out.push(Piece::Bytes(bytes));
        }
        out
    }

    fn piece(&self, piece: &Piece) -> Result<String, Inexpressible> {
        match piece {
            Piece::Bytes(text) => Ok(literal(text)),
            Piece::Token(text) => Ok(format!("({} | {})", self.token_ref(text)?, literal(text))),
            Piece::Node(text) => self.token_ref(text),
        }
    }

    /// A fixed word, the pieces the vocabulary has as tokens spelled as the token or its bytes.
    fn lit(&self, text: &str) -> Result<String, Inexpressible> {
        if text.is_empty() {
            return Ok("\"\"".to_string());
        }
        let pieces = self
            .segments(text)
            .iter()
            .map(|p| self.piece(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(pieces.join(" "))
    }

    // -- free text --------------------------------------------------------------------------

    /// The regex body of free text holding none of `excludes` and ending with `suffix`: the
    /// words are held out of the text before the suffix only, never out of the suffix itself nor
    /// across the boundary, since the suffix is a fixed word's bytes and not the text's.
    fn text_body(excludes: &[String], suffix: &str) -> String {
        let suffix = regex_escape(suffix);
        let mut body = format!("/(?s:.*){suffix}/");
        let words: Vec<&str> = excludes
            .iter()
            .map(String::as_str)
            .filter(|e| !e.is_empty())
            .collect();
        if !words.is_empty() {
            body.push_str(&format!(
                " & ~/(?s:.*)(?:{})(?s:.*){suffix}/",
                join_escaped(words)
            ));
        }
        body
    }

    /// The vocabulary's added tokens against the words free text may not hold. A token equal to a
    /// word, holding one, standing inside one between bytes, or beside another token in one is
    /// left out of the text altogether; the left-out tokens are found first (a pass that leaves a
    /// token out repeats the reading without it), and a word is read as bytes across them, so
    /// none of them names a state. A token piece equal to or holding a word never stands in the
    /// text, so a word that runs through such a piece is read as bytes there and the piece
    /// constrains nothing.
    fn classify(&self, forbid: &[String]) -> Classes {
        let words: Vec<&str> = forbid
            .iter()
            .map(String::as_str)
            .filter(|w| !w.is_empty())
            .collect();
        // the end-of-sequence token never stands in free text: a word is read as bytes across
        // it, as across a left-out token, so it never cuts a word nor names a state
        let mut skip: HashSet<&str> = self
            .tokens
            .longest_first
            .iter()
            .map(String::as_str)
            .filter(|t| {
                self.tokens.id(t) == self.tokens.eos
                    || words.iter().any(|w| t == w || t.contains(*w))
            })
            .collect();
        loop {
            let mut classes = Classes::default();
            let mut left_out: Vec<&str> = Vec::new();
            for text in &self.tokens.longest_first {
                let Some(id) = self.tokens.id(text) else {
                    continue;
                };
                if skip.contains(text.as_str()) {
                    continue;
                }
                let mut out = false;
                let mut prefix: Vec<String> = Vec::new();
                let mut suffix: Vec<(String, Option<String>)> = Vec::new();
                'words: for w in &words {
                    let mut start = 0;
                    while let Some(found) = w[start..].find(text.as_str()) {
                        let i = start + found;
                        let mut lead = &w[..i];
                        let mut rest = &w[i + text.len()..];
                        let mut adjacent = false;
                        let mut before: Option<&str> = None;
                        let mut cut: Option<usize> = None;
                        for t in &self.tokens.longest_first {
                            if skip.contains(t.as_str()) {
                                continue;
                            }
                            if let Some(j) = lead.rfind(t.as_str()) {
                                let end = j + t.len();
                                if cut.is_none_or(|c| end > c) {
                                    cut = Some(end);
                                    before = Some(t.as_str());
                                }
                            }
                        }
                        if let Some(c) = cut {
                            adjacent |= c == lead.len();
                            lead = &lead[c..];
                        }
                        let mut cut: Option<usize> = None;
                        for t in &self.tokens.longest_first {
                            if skip.contains(t.as_str()) {
                                continue;
                            }
                            if let Some(k) = rest.find(t.as_str()) {
                                if cut.is_none_or(|c| k < c) {
                                    cut = Some(k);
                                }
                            }
                        }
                        if let Some(c) = cut {
                            adjacent |= c == 0;
                            rest = &rest[..c];
                        }
                        if adjacent
                            || (!lead.is_empty() && !rest.is_empty())
                            || (lead.is_empty() && rest.is_empty())
                        {
                            out = true;
                            break 'words;
                        } else if !lead.is_empty() {
                            suffix.push((lead.to_string(), before.map(str::to_string)));
                        } else {
                            // the text after this token may not start with the rest of the word
                            // as bytes (the head spells a later token as itself or its bytes);
                            // where the word goes on with another token, that token's suffix
                            // entry also holds the bytes between them, exactly, in this state
                            prefix.push(w[i + text.len()..].to_string());
                        }
                        start = i + text.chars().next().map_or(1, char::len_utf8);
                    }
                }
                if out {
                    left_out.push(text.as_str());
                    continue;
                }
                if prefix.is_empty() && suffix.is_empty() {
                    classes.free.push(id);
                }
                if !prefix.is_empty() {
                    classes.prefix.insert(text.clone(), prefix);
                }
                if !suffix.is_empty() {
                    classes.suffix.insert(text.clone(), suffix);
                }
            }
            if left_out.is_empty() {
                return classes;
            }
            skip.extend(left_out);
        }
    }

    /// The text lexeme holding none of `excludes`, not starting with one of `starts`, not ending
    /// with one of `ends`, not one of `exact`.
    fn lexeme(
        &mut self,
        excludes: &[String],
        starts: &[String],
        ends: &[String],
        exact: &[String],
    ) -> String {
        let mut body = Self::text_body(excludes, "");
        if !starts.is_empty() {
            body.push_str(&format!(
                " & ~/(?:{})(?s:.*)/",
                join_escaped(sorted_unique(starts))
            ));
        }
        if !ends.is_empty() {
            body.push_str(&format!(
                " & ~/(?s:.*)(?:{})/",
                join_escaped(sorted_unique(ends))
            ));
        }
        if !exact.is_empty() {
            body.push_str(&format!(" & ~/(?:{})/", join_escaped(sorted_unique(exact))));
        }
        self.terminal(body)
    }

    /// The states of free text after the tokens that begin a forbidden word, the start first. A
    /// token that stood before a suffix piece in a word begins that word, so it has a prefix
    /// entry of its own (the word's remaining bytes); the entry is only guaranteed here, so the
    /// piece's lexeme can name the state it follows.
    fn tails(classes: &mut Classes) -> Vec<Option<String>> {
        let preceding: BTreeSet<String> = classes
            .suffix
            .values()
            .flatten()
            .filter_map(|(_, before)| before.clone())
            .collect();
        for before in preceding {
            classes.prefix.entry(before).or_default();
        }
        let mut tails = vec![None];
        tails.extend(classes.prefix.keys().cloned().map(Some));
        tails
    }

    /// The alternatives every state of free text shares after its own lexeme: a free token and
    /// the start state; then one transition per token that begins or ends a forbidden word: the
    /// lexeme before it may not end with the leading bytes no token preceded, nor be the bytes
    /// that followed this state's token in a word, and the token lands in a state of its own when
    /// it begins a word, in the start state otherwise.
    fn continuations(
        &mut self,
        excludes: &[String],
        classes: &Classes,
        names: &StateNames,
        state: Option<&String>,
        starts: &[String],
        plain: &str,
    ) -> Result<Vec<String>, Inexpressible> {
        let mut alts = Vec::new();
        if !classes.free.is_empty() {
            alts.push(format!(
                "{plain} {} {}",
                ranges(&classes.free),
                names[&None]
            ));
        }
        let tokens: BTreeSet<&String> =
            classes.prefix.keys().chain(classes.suffix.keys()).collect();
        for token in tokens {
            let entries = || classes.suffix.get(token).into_iter().flatten();
            let ends: Vec<String> = entries()
                .filter(|(_, before)| before.is_none())
                .map(|(lead, _)| lead.clone())
                .collect();
            let exact: Vec<String> = entries()
                .filter(|(_, before)| before.is_some() && before.as_ref() == state)
                .map(|(lead, _)| lead.clone())
                .collect();
            let lexeme = if ends.is_empty() && exact.is_empty() {
                plain.to_string()
            } else {
                self.lexeme(excludes, starts, &ends, &exact)
            };
            let next = if classes.prefix.contains_key(token) {
                &names[&Some(token.clone())]
            } else {
                &names[&None]
            };
            alts.push(format!("{lexeme} {} {next}", self.token_ref(token)?));
        }
        Ok(alts)
    }

    /// The free-text rules for (`excludes`, `forbid`), one per state: the text lexeme, then any
    /// run of (token, text) pairs where a free token takes a plain text, a prefix-piece token a
    /// text not starting with its remaining bytes, and a suffix-piece token follows a text not
    /// ending with its leading bytes.
    fn text_rules(
        &mut self,
        excludes: &[String],
        forbid: &[String],
    ) -> Result<StateNames, Inexpressible> {
        let key = (excludes.to_vec(), forbid.to_vec());
        if let Some(names) = self.text_memo.get(&key) {
            return Ok(names.clone());
        }
        let mut all: Vec<String> = excludes.to_vec();
        all.extend(forbid.iter().cloned());
        let mut classes = self.classify(&all);
        if classes.free.is_empty() && classes.prefix.is_empty() && classes.suffix.is_empty() {
            let lexeme = self.lexeme(excludes, &[], &[], &[]);
            let names = StateNames::from([(None, lexeme)]);
            self.text_memo.insert(key, names.clone());
            return Ok(names);
        }
        let tails = Self::tails(&mut classes);
        let mut names = StateNames::new();
        for state in &tails {
            names.insert(state.clone(), self.fresh("x"));
        }
        for state in &tails {
            let starts: Vec<String> = state
                .as_ref()
                .and_then(|p| classes.prefix.get(p))
                .cloned()
                .unwrap_or_default();
            let plain = self.lexeme(excludes, &starts, &[], &[]);
            let mut alts = vec![plain.clone()];
            alts.extend(self.continuations(
                excludes,
                &classes,
                &names,
                state.as_ref(),
                &starts,
                &plain,
            )?);
            self.rules
                .push(format!("{}: {}", names[state], alts.join(" | ")));
        }
        self.text_memo.insert(key, names.clone());
        Ok(names)
    }

    /// Free text: the text lexeme, interleaved with the added tokens it may hold.
    fn text_expr(
        &mut self,
        excludes: &[String],
        forbid: &[String],
    ) -> Result<String, Inexpressible> {
        Ok(self.text_rules(excludes, forbid)?[&None].clone())
    }

    /// For each word, the rule of free text (holding none of `excludes`) up to and including the
    /// word, after any run of text and the added tokens the text may hold: the byte spelling as a
    /// lazy lexeme, and the token spelling, where the text ends at the word's leading token by
    /// itself; a word that holds a token node has the token spelling alone, the token required.
    fn heads(
        &mut self,
        excludes: &[String],
        words: &[Word],
        forbid: &[String],
    ) -> Result<Vec<String>, Inexpressible> {
        let texts: Vec<String> = words.iter().map(Vec::as_slice).map(word_text).collect();
        let mut all_forbid: Vec<String> = excludes.to_vec();
        all_forbid.extend(forbid.iter().cloned());
        all_forbid.extend(texts.iter().cloned());
        let mut classes = self.classify(&all_forbid);
        let tails = Self::tails(&mut classes);
        let mut out = Vec::with_capacity(words.len());
        for (word, text) in words.iter().zip(&texts) {
            let key = (excludes.to_vec(), all_forbid.clone(), word.clone());
            if let Some(name) = self.head_memo.get(&key) {
                out.push(name.clone());
                continue;
            }
            let mut names = StateNames::new();
            for state in &tails {
                names.insert(state.clone(), self.fresh("h"));
            }
            let mut pieces: Vec<Piece> = Vec::new();
            for part in word {
                match part {
                    Part::Bytes(bytes) => pieces.extend(self.segments(bytes)),
                    Part::Token(token) => pieces.push(Piece::Node(token.clone())),
                }
            }
            let first_token = pieces
                .iter()
                .position(|piece| !matches!(piece, Piece::Bytes(_)));
            let token_node = pieces.iter().any(|piece| matches!(piece, Piece::Node(_)));
            for state in &tails {
                let starts: Vec<String> = state
                    .as_ref()
                    .and_then(|p| classes.prefix.get(p))
                    .cloned()
                    .unwrap_or_default();
                let plain = self.lexeme(excludes, &starts, &[], &[]);
                let mut alts = Vec::new();
                if !token_node {
                    let lazy = format!("{plain} {}", literal(text));
                    alts.push(self.rule(&lazy, Some("lazy"), "l"));
                }
                if let Some(first) = first_token {
                    let lead: String = pieces[..first].iter().map(Piece::text).collect();
                    let mut parts = Vec::new();
                    if lead.is_empty() {
                        parts.push(plain.clone());
                    } else {
                        // the start ban binds the text before the lead, as the exclusion ban
                        // does: a lexeme text + lead starts with a rest exactly when the text does
                        let mut body = Self::text_body(excludes, &lead);
                        if !starts.is_empty() {
                            body.push_str(&format!(
                                " & ~/(?:{})(?s:.*){}/",
                                join_escaped(sorted_unique(&starts)),
                                regex_escape(&lead)
                            ));
                        }
                        parts.push(self.terminal(body));
                    }
                    parts.push(self.token_ref(pieces[first].text())?);
                    for piece in &pieces[first + 1..] {
                        parts.push(self.piece(piece)?);
                    }
                    let body = parts.join(" ");
                    alts.push(self.rule(&body, None, "k"));
                }
                alts.extend(self.continuations(
                    excludes,
                    &classes,
                    &names,
                    state.as_ref(),
                    &starts,
                    &plain,
                )?);
                self.rules
                    .push(format!("{}: {}", names[state], alts.join(" | ")));
            }
            let name = names[&None].clone();
            self.head_memo.insert(key, name.clone());
            out.push(name);
        }
        Ok(out)
    }

    /// Free text up to and including the first of the `follow` words.
    fn lazy_text(
        &mut self,
        excludes: &[String],
        follow: &[Word],
        forbid: &[String],
    ) -> Result<String, Inexpressible> {
        let words: Vec<Word> = follow.iter().filter(|w| !w.is_empty()).cloned().collect();
        if words.is_empty() {
            return self.text_expr(excludes, forbid);
        }
        let exprs = self.heads(excludes, &words, forbid)?;
        Ok(if exprs.len() == 1 {
            exprs.into_iter().next().unwrap_or_default()
        } else {
            format!("({})", exprs.join(" | "))
        })
    }

    // -- leaves -----------------------------------------------------------------------------

    fn json_schema(
        &mut self,
        schema: &Value,
        style: Option<&str>,
        excludes: &[String],
        whitespace: Whitespace,
    ) -> Result<String, Inexpressible> {
        if let Some(style) = style.filter(|s| *s != "json") {
            return Err(Inexpressible(format!(
                "a JSON schema in the {style} style has no Lark spelling"
            )));
        }
        if !excludes.is_empty() {
            let note = "a JSON schema's excluded words are not held out of its strings";
            if !self.notes.iter().any(|n| n == note) {
                self.notes.push(note.to_string());
            }
        }
        let bound = match whitespace {
            Whitespace::Free => None,
            Whitespace::Bounded(run) => Some(run),
        };
        let object = |text: &str| match bound {
            None => text.to_string(),
            Some(run) => {
                let guidance = format!(
                    "{{\"x-guidance\":{{\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{{0,{run}}}\"}}"
                );
                if text == "{}" {
                    format!("{guidance}}}")
                } else {
                    format!("{guidance},{}", &text[1..])
                }
            }
        };
        let text = match schema {
            Value::Bool(false) => {
                return Err(Inexpressible(
                    "a JSON schema that accepts nothing has no Lark spelling".to_string(),
                ))
            }
            Value::Bool(true) => object("{}"),
            Value::Object(_) => object(&serde_json::to_string(schema).unwrap_or_default()),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        Ok(format!("%json {text}"))
    }

    // -- the tree ---------------------------------------------------------------------------

    /// The elements of a sequence; free text before a fixed word (the next constant, the next
    /// tag's begin, the leading word of the next sequence) folds the word into its head, and the
    /// last text takes the enclosing `follow` words. Returns the expressions and whether the
    /// `follow` words were consumed.
    fn elements(
        &mut self,
        nodes: &[Grammar],
        follow: Option<&[String]>,
        forbid: &[String],
    ) -> Result<(Vec<String>, bool), Inexpressible> {
        let mut nodes: Vec<Cow<'_, Grammar>> = nodes.iter().map(Cow::Borrowed).collect();
        let mut out = Vec::with_capacity(nodes.len());
        let mut i = 0;
        while i < nodes.len() {
            let text = match nodes[i].as_ref() {
                Grammar::AnyText { excludes } => Some(excludes.clone()),
                _ => None,
            };
            if let Some(excludes) = text {
                let word = nodes.get(i + 1).and_then(|next| word_of(next.as_ref()));
                if let Some(word) = word {
                    let Some(rest) = after_word(nodes[i + 1].as_ref(), &word) else {
                        return Err(Inexpressible(format!(
                            "the text's following word {:?} is not where the tree begins",
                            word_text(&word)
                        )));
                    };
                    out.push(self.lazy_text(&excludes, std::slice::from_ref(&word), forbid)?);
                    nodes[i + 1] = Cow::Owned(rest);
                    i += 1;
                    continue;
                }
                if i + 1 == nodes.len() {
                    if let Some(follow) = follow.filter(|f| !f.is_empty()) {
                        out.push(self.lazy_text(&excludes, &bytes_words(follow), forbid)?);
                        return Ok((out, true));
                    }
                }
                out.push(self.text_expr(&excludes, forbid)?);
                i += 1;
                continue;
            }
            let last = i + 1 == nodes.len();
            let (expr, consumed) =
                self.lower(nodes[i].as_ref(), if last { follow } else { None }, forbid)?;
            out.push(expr);
            if consumed {
                return Ok((out, true));
            }
            i += 1;
        }
        Ok((out, false))
    }

    /// A node's expression and whether it consumed the `follow` words (the fixed words that end
    /// the enclosing tag, folded into the head of a text that stands last).
    fn lower(
        &mut self,
        node: &Grammar,
        follow: Option<&[String]>,
        forbid: &[String],
    ) -> Result<(String, bool), Inexpressible> {
        let follow = follow.filter(|f| !f.is_empty());
        Ok(match node {
            Grammar::ConstString(value) => (self.lit(value)?, false),
            Grammar::AnyText { excludes } => match follow {
                Some(follow) => (
                    self.lazy_text(excludes, &bytes_words(follow), forbid)?,
                    true,
                ),
                None => (self.text_expr(excludes, forbid)?, false),
            },
            Grammar::Regex(pattern) => (regex_lark(pattern), false),
            Grammar::JsonSchema {
                schema,
                style,
                excludes,
                whitespace,
            } => (
                self.json_schema(schema, style.as_deref(), excludes, *whitespace)?,
                false,
            ),
            Grammar::Ebnf(_) => {
                return Err(Inexpressible(
                    "an embedded EBNF grammar has no Lark spelling".to_string(),
                ))
            }
            Grammar::Sequence(elements) => {
                let (parts, consumed) = self.elements(elements, follow, forbid)?;
                (format!("({})", parts.join(" ")), consumed)
            }
            Grammar::Or(elements) => {
                let mut alts = Vec::with_capacity(elements.len());
                for element in elements {
                    alts.push(self.lower(element, follow, forbid)?);
                }
                let any = alts.iter().any(|(_, consumed)| *consumed);
                let all = !alts.is_empty() && alts.iter().all(|(_, consumed)| *consumed);
                if let Some(follow) = follow.filter(|_| any && !all) {
                    // the alternatives that did not take the follow words take them here, so
                    // every alternative ends the same way
                    let tail = self.ends(follow)?;
                    alts = alts
                        .into_iter()
                        .map(|(expr, consumed)| {
                            if consumed {
                                (expr, true)
                            } else {
                                (format!("{expr} {tail}"), true)
                            }
                        })
                        .collect();
                }
                let all = !alts.is_empty() && alts.iter().all(|(_, consumed)| *consumed);
                let exprs: Vec<String> = alts.into_iter().map(|(expr, _)| expr).collect();
                (format!("({})", exprs.join(" | ")), all)
            }
            Grammar::Optional(content) => (
                format!("({})?", self.lower(content, None, forbid)?.0),
                false,
            ),
            Grammar::Star(content) => (
                format!("({})*", self.lower(content, None, forbid)?.0),
                false,
            ),
            Grammar::Plus(content) => (
                format!("({})+", self.lower(content, None, forbid)?.0),
                false,
            ),
            Grammar::Tag(tag) => (format!("({})", self.tag_body(tag, 0)?), false),
            Grammar::Token(text) => (self.token_ref(text)?, false),
            Grammar::TriggeredTags {
                triggers,
                tags,
                at_least_one,
                stop_after_first,
            } => (
                self.triggered_tags(triggers, tags, *at_least_one, *stop_after_first)?,
                false,
            ),
            Grammar::TagsWithSeparator {
                tags,
                separator,
                at_least_one,
                stop_after_first,
            } => (
                self.tags_with_separator(tags, separator, *at_least_one, *stop_after_first)?,
                false,
            ),
        })
    }

    /// The fixed words that end a tag, as one literal or an alternation.
    fn ends(&self, words: &[String]) -> Result<String, Inexpressible> {
        let words: Vec<&String> = words.iter().filter(|w| !w.is_empty()).collect();
        match words.as_slice() {
            [] => Ok(String::new()),
            [word] => self.lit(word),
            _ => {
                let literals = words
                    .iter()
                    .map(|w| self.lit(w))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!("({})", literals.join(" | ")))
            }
        }
    }

    /// `begin content end`, the begin's first `skip` bytes already consumed (by a trigger).
    fn tag_body(&mut self, tag: &Tag, skip: usize) -> Result<String, Inexpressible> {
        let begin = tag.begin.get(skip..).unwrap_or_default();
        let words = bound_words(&tag.end);
        let mut parts = Vec::with_capacity(3);
        if !begin.is_empty() {
            parts.push(self.lit(begin)?);
        }
        let follow = (!words.is_empty()).then_some(words.as_slice());
        let (inner, consumed) = self.lower(&tag.content, follow, &words)?;
        parts.push(inner);
        if !consumed {
            let tail = self.ends(&words)?;
            if !tail.is_empty() {
                parts.push(tail);
            }
        }
        Ok(parts.join(" "))
    }

    /// Free text until a trigger opens one of the tags, then the tag; with `at_least_one` the
    /// first tag at once; with `stop_after_first` nothing after the first tag.
    fn triggered_tags(
        &mut self,
        triggers: &[String],
        tags: &[Tag],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, Inexpressible> {
        let text = self.text_expr(&[], triggers)?;
        let mut heads: HashMap<&str, String> = HashMap::new();
        let mut first_alts = Vec::with_capacity(tags.len());
        let mut later_alts = Vec::with_capacity(tags.len());
        for tag in tags {
            let Some(trigger) = triggers.iter().find(|t| tag.begin.starts_with(t.as_str())) else {
                return Err(Inexpressible(format!(
                    "the tag opening with {:?} matches no trigger",
                    tag.begin
                )));
            };
            if !heads.contains_key(trigger.as_str()) {
                let head = self.lazy_text(&[], &[bytes_word(trigger)], triggers)?;
                heads.insert(trigger.as_str(), head);
            }
            let body = self.tag_body(tag, trigger.len())?;
            let opener = self.lit(trigger)?;
            let first = if body.is_empty() {
                opener
            } else {
                format!("{opener} {body}")
            };
            first_alts.push(self.rule(&first, None, "f"));
            let head = heads[trigger.as_str()].clone();
            let later = if body.is_empty() {
                head
            } else {
                format!("{head} {body}")
            };
            later_alts.push(self.rule(&later, None, "t"));
        }
        let first = format!("({})", first_alts.join(" | "));
        let later = format!("({})", later_alts.join(" | "));
        Ok(match (at_least_one, stop_after_first) {
            (true, true) => first,
            (true, false) => format!("({first} {later}* {text})"),
            (false, true) => format!("({later} | {text})"),
            (false, false) => format!("({later}* {text})"),
        })
    }

    /// The tags one after another with the separator between them, no free text.
    fn tags_with_separator(
        &mut self,
        tags: &[Tag],
        separator: &str,
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, Inexpressible> {
        let mut bodies = Vec::with_capacity(tags.len());
        for tag in tags {
            bodies.push(format!("({})", self.tag_body(tag, 0)?));
        }
        let alts = format!("({})", bodies.join(" | "));
        let sep = if separator.is_empty() {
            String::new()
        } else {
            self.lit(separator)?
        };
        let one = self.rule(&alts, None, "s");
        if stop_after_first {
            return Ok(if at_least_one {
                one
            } else {
                format!("({one})?")
            });
        }
        let more = if sep.is_empty() {
            format!("({one})*")
        } else {
            format!("({sep} {one})*")
        };
        Ok(if at_least_one {
            format!("({one} {more})")
        } else {
            format!("({one} {more})?")
        })
    }

    // -- the text ---------------------------------------------------------------------------

    fn text(&self, start: &str) -> String {
        let mut out = String::from("%llguidance {}\nstart: ");
        out.push_str(start);
        out.push('\n');
        for rule in &self.rules {
            out.push_str(rule);
            out.push('\n');
        }
        for line in &self.terminal_lines {
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests;
