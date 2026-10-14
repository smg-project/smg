//! The pattern syntax of the `regex` module as transformers compiles it
//! (`regex.compile(pattern, regex.DOTALL)`, version 0 behaviour), ported from
//! the parser in `_regex_core.py` (`regex` 2026.9.10). Every construct is read
//! into a [`Node`] tree, also those this crate's matcher does not run; the
//! matcher refuses those when it compiles a tree ([`Pattern::new`]).
//!
//! The functions keep the module's names and order of tests, so a pattern is
//! read, or rejected, as the module reads it. A few constructs are not read
//! (listed in the README): Unicode properties in braces, named characters,
//! POSIX classes other than the 14 POSIX names, group names outside ASCII, a
//! group inside a group of the same name, version 1 behaviour, and groups
//! nested more than 100 deep.
//!
//! [`Pattern::new`]: super::Pattern::new

use std::{collections::HashMap, fmt};

use super::{Escape, Look, Node, Other, Quantifier, Set};
use crate::py;

const ASCII: u32 = 1 << 0;
const BESTMATCH: u32 = 1 << 1;
const ENHANCEMATCH: u32 = 1 << 2;
const FULLCASE: u32 = 1 << 3;
const IGNORECASE: u32 = 1 << 4;
const LOCALE: u32 = 1 << 5;
const MULTILINE: u32 = 1 << 6;
const POSIX: u32 = 1 << 7;
const REVERSE: u32 = 1 << 8;
const DOTALL: u32 = 1 << 9;
const UNICODE: u32 = 1 << 10;
const VERSION0: u32 = 1 << 11;
const VERSION1: u32 = 1 << 12;
const WORD: u32 = 1 << 13;
const VERBOSE: u32 = 1 << 14;

/// Flags whose inline use reads the whole pattern again with them.
const GLOBAL_FLAGS: u32 = VERSION0 | VERSION1 | BESTMATCH | ENHANCEMATCH | POSIX | REVERSE;
const ALL_ENCODINGS: u32 = ASCII | LOCALE | UNICODE;

/// `REGEX_FLAGS`: the letters of inline flags (`V0` and `V1` take two).
fn flag_bit(name: &str) -> Option<u32> {
    Some(match name {
        "a" => ASCII,
        "b" => BESTMATCH,
        "e" => ENHANCEMATCH,
        "f" => FULLCASE,
        "i" => IGNORECASE,
        "L" => LOCALE,
        "m" => MULTILINE,
        "p" => POSIX,
        "r" => REVERSE,
        "s" => DOTALL,
        "u" => UNICODE,
        "V0" => VERSION0,
        "V1" => VERSION1,
        "w" => WORD,
        "x" => VERBOSE,
        _ => return None,
    })
}

/// Inline flags that change what a pattern matches when turned on: case
/// folding, multiline anchors, Unicode word boundaries, ASCII or locale
/// classes, and the global matching modes. `s` changes it only when turned
/// off (transformers turns it on). `f` alone, `u`, `x` and `V0` change nothing
/// a match returns (`x` changes how the pattern reads, which the reader does).
const MEANINGFUL: [(u32, char); 9] = [
    (IGNORECASE, 'i'),
    (MULTILINE, 'm'),
    (WORD, 'w'),
    (ASCII, 'a'),
    (LOCALE, 'L'),
    (BESTMATCH, 'b'),
    (ENHANCEMATCH, 'e'),
    (POSIX, 'p'),
    (REVERSE, 'r'),
];

/// The deepest nesting of groups read. The reader, and every walk over a
/// tree, recurses once per level; a template from a model's
/// `tokenizer_config.json` must not be able to overflow the stack.
const MAX_GROUP_DEPTH: usize = 100;

/// The 14 POSIX class names, as `standardise_name` writes them.
const POSIX_NAMES: [&str; 14] = [
    "ALNUM", "ALPHA", "ASCII", "BLANK", "CNTRL", "DIGIT", "GRAPH", "LOWER", "PRINT", "PUNCT",
    "SPACE", "UPPER", "WORD", "XDIGIT",
];

/// Why a pattern was not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReadError {
    /// The `regex` module rejects the pattern too (its message, without the
    /// position).
    Syntax(String),
    /// A construct the reader does not read, which the module may accept.
    NotRead(&'static str),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(message) => f.write_str(message),
            Self::NotRead(what) => f.write_str(what),
        }
    }
}

type Read<T> = Result<T, ReadError>;

fn syntax<T>(message: impl Into<String>) -> Read<T> {
    Err(ReadError::Syntax(message.into()))
}

/// A pattern as the `regex` module reads it.
#[derive(Debug, Clone)]
pub(crate) struct Regex {
    pub(crate) node: Node,
    /// Named groups and their numbers, in the order the module lists them
    /// (`pattern.groupindex`); a name used twice names one group.
    pub(crate) names: Vec<(String, usize)>,
    /// The number of capturing groups.
    pub(crate) groups: usize,
    /// Per group number (0 is the whole match): whether a match can leave the
    /// group unset.
    pub(crate) optional: Vec<bool>,
    /// The inline flags that change what the pattern matches, as letters
    /// (`i`, `m`, ...; `-s` when `s` is turned off); empty when there are none.
    pub(crate) flags: String,
}

impl Regex {
    /// `regex.compile(source, regex.DOTALL)`, read into a tree.
    pub(crate) fn parse(source: &str) -> Read<Self> {
        let mut global = DOTALL;
        // A global flag turned on inline reads the whole pattern again with it;
        // version 1 reads another syntax, which the reader does not read.
        loop {
            let mut reader = Reader::new(source, global);
            match reader.pattern() {
                Err(Restart(flags)) if flags & VERSION1 != 0 => {
                    return Err(ReadError::NotRead("version 1 behaviour (?V1)"));
                }
                Err(Restart(flags)) => global = flags,
                Ok(Err(error)) => return Err(error),
                Ok(Ok(node)) => return reader.finish(node),
            }
        }
    }

    /// The names of the named groups.
    pub(crate) fn group_names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(|(name, _)| name.as_str())
    }
}

/// `_UnscopedFlagSet`: read the pattern again with these global flags.
struct Restart(u32);

/// A result of reading, or a restart with new global flags.
type Step<T> = Result<Read<T>, Restart>;

/// What a name or number left for the end of reading refers to (`fix_groups`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reference {
    Backref,
    Call,
    Condition,
}

/// What an escape reads as: a character by its code point (a surrogate can
/// bound a range in a class), or another node.
enum Escaped {
    Char(u32),
    Node(Node),
}

/// `Source` and `Info` of `_regex_core.py`: the pattern, the position, and
/// what reading found so far.
struct Reader {
    chars: Vec<char>,
    pos: usize,
    ignore_space: bool,
    flags: u32,
    global_flags: u32,
    group_count: usize,
    /// `group_index`: name to number, in the order names are first used.
    group_index: Vec<(String, usize)>,
    /// `group_name`: number to name.
    group_name: HashMap<usize, String>,
    open_groups: Vec<usize>,
    open_group_count: HashMap<usize, usize>,
    /// Names and numbers to resolve at the end. A backreference's node holds
    /// its index here until then.
    references: Vec<(Reference, String)>,
    flags_used: String,
    depth: usize,
}

impl Reader {
    fn new(source: &str, global_flags: u32) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            ignore_space: false,
            flags: global_flags,
            global_flags,
            group_count: 0,
            group_index: Vec::new(),
            group_name: HashMap::new(),
            open_groups: Vec::new(),
            open_group_count: HashMap::new(),
            references: Vec::new(),
            flags_used: String::new(),
            depth: 0,
        }
    }

    /// From `at`, past whitespace and comments when `ignore`.
    fn skip(&self, mut at: usize, ignore: bool) -> usize {
        if ignore {
            while let Some(&c) = self.chars.get(at) {
                if py::is_space(c) {
                    at += 1;
                } else if c == '#' {
                    at = self.chars[at..]
                        .iter()
                        .position(|&c| c == '\n')
                        .map_or(self.chars.len(), |k| at + k);
                } else {
                    break;
                }
            }
        }
        at
    }

    /// `source.get(override_ignore)`; `None` at the end.
    fn get_with(&mut self, override_ignore: bool) -> Option<char> {
        let at = self.skip(self.pos, self.ignore_space && !override_ignore);
        let c = self.chars.get(at).copied();
        self.pos = if c.is_some() { at + 1 } else { at };
        c
    }

    fn get(&mut self) -> Option<char> {
        self.get_with(false)
    }

    fn peek(&self) -> Option<char> {
        self.chars
            .get(self.skip(self.pos, self.ignore_space))
            .copied()
    }

    /// `source.match(s)`: each character after whitespace when verbose.
    fn eat(&mut self, s: &str) -> bool {
        let mut at = self.pos;
        for c in s.chars() {
            at = self.skip(at, self.ignore_space);
            if self.chars.get(at) != Some(&c) {
                return false;
            }
            at += 1;
        }
        self.pos = at;
        true
    }

    fn expect(&mut self, s: &str) -> Read<()> {
        if self.eat(s) {
            Ok(())
        } else {
            syntax(format!("missing {s}"))
        }
    }

    /// `source.get_while(test)`; verbose patterns skip whitespace and
    /// comments unless `keep_spaces`.
    fn get_while(&mut self, test: impl Fn(char) -> bool, keep_spaces: bool) -> String {
        let ignore = self.ignore_space && !keep_spaces;
        let mut out = String::new();
        loop {
            let at = self.skip(self.pos, ignore);
            match self.chars.get(at) {
                Some(&c) if test(c) => {
                    out.push(c);
                    self.pos = at + 1;
                }
                _ => {
                    if ignore {
                        self.pos = at;
                    }
                    return out;
                }
            }
        }
    }

    fn digits(&mut self) -> String {
        self.get_while(|c| c.is_ascii_digit(), false)
    }

    fn at_end(&self) -> bool {
        self.skip(self.pos, self.ignore_space) >= self.chars.len()
    }

    /// Note the flags that change what the pattern matches.
    fn use_flags(&mut self, on: u32, off: u32) {
        for &(bit, letter) in &MEANINGFUL {
            if on & bit != 0 && !self.flags_used.contains(letter) {
                self.flags_used.push(letter);
            }
        }
        if off & DOTALL != 0 && !self.flags_used.contains("-s") {
            self.flags_used.push_str("-s");
        }
    }

    fn enter(&mut self) -> Read<()> {
        if self.depth == MAX_GROUP_DEPTH {
            return Err(ReadError::NotRead("groups nested more than 100 deep"));
        }
        self.depth += 1;
        Ok(())
    }

    /// `_parse_pattern`: alternatives.
    fn pattern(&mut self) -> Step<Node> {
        let mut branches = match self.sequence()? {
            Ok(node) => vec![node],
            Err(e) => return Ok(Err(e)),
        };
        while self.eat("|") {
            match self.sequence()? {
                Ok(node) => branches.push(node),
                Err(e) => return Ok(Err(e)),
            }
        }
        Ok(Ok(if branches.len() == 1 {
            branches.remove(0)
        } else {
            Node::Alt(branches)
        }))
    }

    /// `parse_sequence`. A `None` item marks where a quantifier has nothing
    /// to repeat: the start, and after a quantifier.
    fn sequence(&mut self) -> Step<Node> {
        let mut items: Vec<Option<Node>> = vec![None];
        loop {
            let saved = self.pos;
            let Some(c) = self.get() else {
                self.pos = saved;
                break;
            };
            let item = match c {
                ')' | '|' => {
                    self.pos = saved;
                    break;
                }
                '\\' => self.escape(false).map(|e| match e {
                    Escaped::Char(c) => Node::One(code_point(c)),
                    Escaped::Node(node) => node,
                }),
                '(' => match self.paren()? {
                    Ok(Some(node)) => Ok(node),
                    Ok(None) => continue,
                    Err(e) => Err(e),
                },
                '.' => Ok(Node::One(Set::Any)),
                '[' => self.set(),
                '^' => Ok(Node::Look(Look::Start)),
                '$' => Ok(Node::Look(Look::Dollar)),
                '?' | '*' | '+' | '{' => {
                    match self.quantifier(c) {
                        Ok(Some((min, max))) => match self.apply_quantifier(&mut items, min, max) {
                            Ok(()) => {
                                items.push(None);
                                continue;
                            }
                            Err(e) => Err(e),
                        },
                        Ok(None) => match self.fuzzy(c) {
                            // A setting that allows no error is dropped.
                            Ok(Some(false)) => continue,
                            Ok(Some(true)) => match items.pop() {
                                Some(Some(node)) => {
                                    items.push(Some(Node::Other(Other::Fuzzy, vec![node])));
                                    items.push(None);
                                    continue;
                                }
                                _ => syntax("nothing for fuzzy constraint"),
                            },
                            Ok(None) => Ok(Node::One(Set::Char(c))),
                            Err(e) => Err(e),
                        },
                        Err(e) => Err(e),
                    }
                }
                c => Ok(Node::One(Set::Char(c))),
            };
            match item {
                Ok(node) => items.push(Some(node)),
                Err(e) => return Ok(Err(e)),
            }
        }
        let mut items: Vec<Node> = items.into_iter().flatten().collect();
        Ok(Ok(if items.len() == 1 {
            items.remove(0)
        } else {
            Node::Cat(items)
        }))
    }

    /// `apply_quantifier`, once the counts are read.
    fn apply_quantifier(
        &mut self,
        items: &mut Vec<Option<Node>>,
        min: u32,
        max: Option<u32>,
    ) -> Read<()> {
        let element = match items.pop() {
            Some(Some(node)) => node,
            _ if items.is_empty() => return syntax("nothing to repeat"),
            _ => return syntax("multiple repeat"),
        };
        if matches!(element, Node::Repeat { .. }) {
            return syntax("multiple repeat");
        }
        let saved = self.pos;
        let kind = match self.get() {
            Some('?') => Quantifier::Lazy,
            Some('+') => Quantifier::Possessive,
            _ => {
                self.pos = saved;
                Quantifier::Greedy
            }
        };
        // The module ignores a quantifier of an empty item, or of exactly one.
        items.push(Some(
            if element.is_empty() || (min == 1 && max == Some(1)) {
                element
            } else {
                Node::Repeat {
                    node: Box::new(element),
                    min,
                    max,
                    kind,
                }
            },
        ));
        Ok(())
    }

    /// `parse_quantifier`: the counts of `?`, `*`, `+` or `{m,n}`; `None` when
    /// a `{` starts no counted repeat.
    fn quantifier(&mut self, c: char) -> Read<Option<(u32, Option<u32>)>> {
        Ok(match c {
            '?' => Some((0, Some(1))),
            '*' => Some((0, None)),
            '+' => Some((1, None)),
            _ => self.limited_quantifier()?,
        })
    }

    /// `parse_limited_quantifier`, after `{`.
    fn limited_quantifier(&mut self) -> Read<Option<(u32, Option<u32>)>> {
        let saved = self.pos;
        let first = self.digits();
        let (min, max) = if self.eat(",") {
            let second = self.digits();
            let min = if first.is_empty() { "0".into() } else { first };
            (min, (!second.is_empty()).then_some(second))
        } else if first.is_empty() {
            self.pos = saved;
            return Ok(None);
        } else {
            (first.clone(), Some(first))
        };
        if !self.eat("}") {
            self.pos = saved;
            return Ok(None);
        }
        // `UNLIMITED` (2**32 - 1) and more are too many.
        let count = |digits: &str| digits.parse::<u32>().ok().filter(|&n| n < u32::MAX);
        let Some(min) = count(&min) else {
            return syntax("repeat count too big");
        };
        let max = match max.as_deref().map(count) {
            None => None,
            Some(Some(max)) => Some(max),
            Some(None) => return syntax("repeat count too big"),
        };
        if max.is_some_and(|max| min > max) {
            return syntax("min repeat greater than max repeat");
        }
        Ok(Some((min, max)))
    }

    /// `parse_fuzzy`, after `{`: whether the setting allows errors, or `None`
    /// when this `{` starts no fuzzy setting (it is then a literal).
    fn fuzzy(&mut self, c: char) -> Read<Option<bool>> {
        if c != '{' {
            return Ok(None);
        }
        let saved = self.pos;
        let mut constraints = Constraints::default();
        let mut items = || -> Result<(), Fuzzy> {
            self.fuzzy_item(&mut constraints)?;
            while self.eat(",") {
                self.fuzzy_item(&mut constraints)?;
            }
            Ok(())
        };
        match items() {
            Ok(()) => {}
            Err(Fuzzy::NotFuzzy) => {
                self.pos = saved;
                return Ok(None);
            }
            Err(Fuzzy::Error(e)) => return Err(e),
        }
        let set_test = if self.eat(":") {
            self.fuzzy_test()?
        } else {
            true
        };
        if !self.eat("}") {
            return syntax("expected }");
        }
        // `is_actually_fuzzy`.
        let none = Some((0, Some(0)));
        let fuzzy = constraints.e != none
            && (constraints.s, constraints.i, constraints.d) != (none, none, none);
        // The module compiles a test only when the setting allows errors, and
        // raises `RuntimeError` for one that is not a character or a set.
        if fuzzy && !set_test {
            return syntax("invalid RE code");
        }
        Ok(Some(fuzzy))
    }

    /// `parse_fuzzy_item`.
    fn fuzzy_item(&mut self, constraints: &mut Constraints) -> Result<(), Fuzzy> {
        let saved = self.pos;
        match self.cost_constraint(constraints) {
            Err(Fuzzy::NotFuzzy) => {
                self.pos = saved;
                self.cost_equation(constraints)
            }
            other => other,
        }
    }

    /// `parse_cost_constraint`.
    fn cost_constraint(&mut self, constraints: &mut Constraints) -> Result<(), Fuzzy> {
        let saved = self.pos;
        match self.get() {
            Some(c) if c.is_ascii_alphabetic() => {
                constraints.check(c)?;
                let limit = match self.fuzzy_compare() {
                    None => (0, None),
                    Some(inclusive) => {
                        let mut max = self.cost_limit()?;
                        if !inclusive {
                            max -= 1;
                        }
                        if max < 0 {
                            return Err(Fuzzy::error("bad fuzzy cost limit"));
                        }
                        (0, Some(max))
                    }
                };
                constraints.set(c, limit);
                Ok(())
            }
            Some(c) if c.is_ascii_digit() => {
                self.pos = saved;
                let mut min = self.cost_limit()?;
                let min_inclusive = self.fuzzy_compare().ok_or(Fuzzy::NotFuzzy)?;
                let c = self.get().unwrap_or('\0');
                constraints.check(c)?;
                let max_inclusive = self.fuzzy_compare().ok_or(Fuzzy::NotFuzzy)?;
                let mut max = self.cost_limit()?;
                if !min_inclusive {
                    min += 1;
                }
                if !max_inclusive {
                    max -= 1;
                }
                if !(0 <= min && min <= max) {
                    return Err(Fuzzy::error("bad fuzzy cost limit"));
                }
                constraints.set(c, (min, Some(max)));
                Ok(())
            }
            _ => Err(Fuzzy::NotFuzzy),
        }
    }

    /// `parse_cost_limit`.
    fn cost_limit(&mut self) -> Result<i64, Fuzzy> {
        self.digits()
            .parse::<i64>()
            .map_err(|_| Fuzzy::error("bad fuzzy cost limit"))
    }

    /// `parse_fuzzy_compare`: inclusive, exclusive, or no comparison.
    fn fuzzy_compare(&mut self) -> Option<bool> {
        if self.eat("<=") {
            Some(true)
        } else if self.eat("<") {
            Some(false)
        } else {
            None
        }
    }

    /// `parse_cost_equation`.
    fn cost_equation(&mut self, constraints: &mut Constraints) -> Result<(), Fuzzy> {
        if constraints.cost {
            return Err(Fuzzy::error("more than one cost equation"));
        }
        let mut terms = String::new();
        self.cost_term(&mut terms)?;
        while self.eat("+") {
            self.cost_term(&mut terms)?;
        }
        let inclusive = self.fuzzy_compare().ok_or(Fuzzy::NotFuzzy)?;
        // `int(parse_count(source))`: an empty count raises a ValueError,
        // which transformers' loading lets through.
        let mut max = self
            .digits()
            .parse::<i64>()
            .map_err(|_| Fuzzy::error("invalid cost limit"))?;
        if !inclusive {
            max -= 1;
        }
        if max < 0 {
            return Err(Fuzzy::error("bad fuzzy cost limit"));
        }
        constraints.cost = true;
        Ok(())
    }

    /// `parse_cost_term`: a coefficient (always digits) and `d`, `i` or `s`.
    fn cost_term(&mut self, terms: &mut String) -> Result<(), Fuzzy> {
        self.digits();
        let Some(c @ ('d' | 'i' | 's')) = self.get() else {
            return Err(Fuzzy::NotFuzzy);
        };
        if terms.contains(c) {
            return Err(Fuzzy::error("repeated fuzzy cost"));
        }
        terms.push(c);
        Ok(())
    }

    /// `parse_fuzzy_test`: a character, `.`, an escape or a class; whether it
    /// is a character or a set (an escape can also be an assertion or a
    /// backreference). The module never resolves a backreference here.
    fn fuzzy_test(&mut self) -> Read<bool> {
        match self.get() {
            Some('\\') => {
                let references = self.references.len();
                let escaped = self.escape(false)?;
                self.references.truncate(references);
                Ok(matches!(
                    escaped,
                    Escaped::Char(_) | Escaped::Node(Node::One(_))
                ))
            }
            Some('.') => Ok(true),
            Some('[') => self.set().map(|_| true),
            Some('(' | ')' | '|' | '?' | '*' | '+' | '{' | '^' | '$' | '#') | None => {
                syntax("expected character set")
            }
            Some(_) => Ok(true),
        }
    }

    /// `parse_paren`, after `(`: a group, an assertion or an extension; `None`
    /// for inline flags and comments.
    fn paren(&mut self) -> Step<Option<Node>> {
        let saved = self.pos;
        let c = self.get_with(true);
        if c == Some('?') {
            let saved2 = self.pos;
            match self.get_with(true) {
                Some('<') => {
                    let saved3 = self.pos;
                    match self.get() {
                        Some(c @ ('=' | '!')) => return self.lookaround(true, c == '='),
                        _ => self.pos = saved3,
                    }
                    return match self.name(false) {
                        Ok(name) => self.capture(Some(name)),
                        Err(e) => Ok(Err(e)),
                    };
                }
                Some(c @ ('=' | '!')) => return self.lookaround(false, c == '='),
                Some('P') => return self.extension(),
                Some('#') => return Ok(self.comment().map(|()| None)),
                Some('(') => return self.conditional(),
                Some('>') => return self.atomic(),
                Some('|') => return self.branch_reset(),
                Some('R') => return Ok(self.call("0".into())),
                Some(c) if c.is_ascii_digit() => {
                    let number = format!("{c}{}", self.digits());
                    return Ok(self.call(number));
                }
                Some('&') => {
                    return Ok(match self.name(false) {
                        Ok(name) => self.call(name),
                        Err(e) => Err(e),
                    })
                }
                Some(c @ ('+' | '-')) if self.peek().is_some_and(|d| d.is_ascii_digit()) => {
                    return Ok(self.relative_call(c == '+'));
                }
                _ => {}
            }
            self.pos = saved2;
            return self.flags_subpattern();
        }
        if c == Some('*') {
            let word = self.get_while(|c| c != ')' && c != '>', false);
            if word.chars().next().is_some_and(char::is_alphabetic) {
                let verb = match word.as_str() {
                    "FAIL" | "F" => Other::Fail,
                    "PRUNE" => Other::Prune,
                    "SKIP" => Other::Skip,
                    _ => return Ok(syntax("unknown verb")),
                };
                return Ok(self
                    .expect(")")
                    .map(|()| Some(Node::Other(verb, Vec::new()))));
            }
        }
        self.pos = saved;
        self.capture(None)
    }

    /// A capturing group after its opening (and name).
    fn capture(&mut self, name: Option<String>) -> Step<Option<Node>> {
        let group = match self.open_group(name.as_deref()) {
            Ok(group) => group,
            Err(e) => return Ok(Err(e)),
        };
        if name.is_some() {
            if let Err(e) = self.expect(">") {
                return Ok(Err(e));
            }
        }
        let inner = self.scoped(Self::pattern)?;
        self.open_groups.pop();
        Ok(inner.map(|inner| Some(Node::Group(Some(group), Box::new(inner)))))
    }

    /// A group's body and its `)`, with the flags restored after it.
    fn scoped(&mut self, body: impl FnOnce(&mut Self) -> Step<Node>) -> Step<Node> {
        if let Err(e) = self.enter() {
            return Ok(Err(e));
        }
        let saved_flags = self.flags;
        let inner = match body(self)? {
            Ok(node) => self.expect(")").map(|()| node),
            Err(e) => Err(e),
        };
        self.flags = saved_flags;
        self.ignore_space = self.flags & VERBOSE != 0;
        self.depth -= 1;
        Ok(inner)
    }

    /// `Info.open_group`.
    fn open_group(&mut self, name: Option<&str>) -> Read<usize> {
        let known = name.and_then(|name| self.group_number(name));
        let group = match known {
            Some(group) => group,
            None => {
                loop {
                    self.group_count += 1;
                    if name.is_none() || !self.group_name.contains_key(&self.group_count) {
                        break;
                    }
                }
                if let Some(name) = name {
                    self.group_index.push((name.to_owned(), self.group_count));
                    self.group_name.insert(self.group_count, name.to_owned());
                }
                self.group_count
            }
        };
        if self.open_groups.contains(&group) {
            return Err(ReadError::NotRead(
                "a group inside a group of the same name",
            ));
        }
        self.open_groups.push(group);
        *self.open_group_count.entry(group).or_default() += 1;
        Ok(group)
    }

    fn group_number(&self, name: &str) -> Option<usize> {
        self.group_index
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, g)| g)
    }

    /// `Info.is_open_group` (version 0).
    fn is_open_group(&self, name: &str) -> bool {
        let group = if name.bytes().all(|b| b.is_ascii_digit()) {
            name.parse::<usize>().ok()
        } else {
            self.group_number(name)
        };
        group.is_some_and(|g| self.open_groups.contains(&g))
    }

    /// `parse_name`. Names outside ASCII are not read: Python's
    /// `str.isidentifier` and `str.isdigit` would need Unicode tables.
    fn name(&mut self, allow_numeric: bool) -> Read<String> {
        let name = self.get_while(|c| c != ')' && c != '>', false);
        if name.is_empty() {
            return syntax("missing group name");
        }
        if !name.is_ascii() {
            return Err(ReadError::NotRead("a group name outside ASCII"));
        }
        let valid = if name.bytes().all(|b| b.is_ascii_digit()) {
            // A number names a group from 1.
            allow_numeric && name.bytes().any(|b| b != b'0')
        } else {
            let mut bytes = name.bytes();
            bytes
                .next()
                .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
        };
        if valid {
            Ok(name)
        } else {
            syntax("bad character in group name")
        }
    }

    /// A name or number resolved at the end; returns its index.
    fn reference(&mut self, kind: Reference, name: String) -> usize {
        self.references.push((kind, name));
        self.references.len() - 1
    }

    /// `make_ref_group`: the node holds the reference's index until `finish`.
    fn backref(&mut self, name: String) -> Node {
        Node::Backref(self.reference(Reference::Backref, name))
    }

    /// `parse_call_group` and `parse_call_named_group`, after the name.
    fn call(&mut self, group: String) -> Read<Option<Node>> {
        self.expect(")")?;
        self.reference(Reference::Call, group);
        Ok(Some(Node::Other(Other::GroupCall, Vec::new())))
    }

    /// `parse_rel_call_group`, after `(?+` or `(?-`.
    fn relative_call(&mut self, forward: bool) -> Read<Option<Node>> {
        let digits = self.digits();
        if digits.is_empty() {
            return syntax("missing relative group number");
        }
        let offset = digits.parse::<usize>().unwrap_or(usize::MAX);
        let group = if forward {
            self.group_count.checked_add(offset)
        } else {
            (self.group_count + 1).checked_sub(offset)
        };
        match group.filter(|&g| g > 0) {
            Some(group) => self.call(group.to_string()),
            None => syntax("invalid relative group number"),
        }
    }

    /// `parse_extension`, after `(?P`.
    fn extension(&mut self) -> Step<Option<Node>> {
        let saved = self.pos;
        match self.get() {
            Some('<') => match self.name(false) {
                Ok(name) => self.capture(Some(name)),
                Err(e) => Ok(Err(e)),
            },
            Some('=') => Ok(self.named_backref()),
            Some('>' | '&') => Ok(match self.name(false) {
                Ok(name) => self.call(name),
                Err(e) => Err(e),
            }),
            _ => {
                self.pos = saved;
                Ok(syntax("unknown extension"))
            }
        }
    }

    /// `(?P=name)`, after `=`.
    fn named_backref(&mut self) -> Read<Option<Node>> {
        let name = self.name(true)?;
        self.expect(")")?;
        if self.is_open_group(&name) {
            return syntax("cannot refer to an open group");
        }
        Ok(Some(self.backref(name)))
    }

    /// `parse_comment`, after `(?#`.
    fn comment(&mut self) -> Read<()> {
        loop {
            let saved = self.pos;
            match self.get_with(true) {
                None | Some(')') => {
                    self.pos = saved;
                    break;
                }
                Some('\\') => {
                    self.get_with(true);
                }
                Some(_) => {}
            }
        }
        self.expect(")")
    }

    /// `parse_lookaround`.
    fn lookaround(&mut self, behind: bool, positive: bool) -> Step<Option<Node>> {
        let inner = self.scoped(Self::pattern)?;
        Ok(inner.map(|inner| {
            Some(Node::Around {
                behind,
                negated: !positive,
                node: Box::new(inner),
            })
        }))
    }

    /// `parse_atomic`.
    fn atomic(&mut self) -> Step<Option<Node>> {
        let inner = self.scoped(Self::pattern)?;
        Ok(inner.map(|inner| Some(Node::Other(Other::Atomic, vec![inner]))))
    }

    /// `parse_common`, after `(?|`: alternatives whose groups are numbered
    /// from the same number.
    fn branch_reset(&mut self) -> Step<Option<Node>> {
        if let Err(e) = self.enter() {
            return Ok(Err(e));
        }
        let initial = self.group_count;
        let mut branches = match self.sequence()? {
            Ok(node) => vec![node],
            Err(e) => return Ok(Err(e)),
        };
        let mut last = self.group_count;
        while self.eat("|") {
            self.group_count = initial;
            match self.sequence()? {
                Ok(node) => branches.push(node),
                Err(e) => return Ok(Err(e)),
            }
            last = last.max(self.group_count);
        }
        self.group_count = last;
        self.depth -= 1;
        if let Err(e) = self.expect(")") {
            return Ok(Err(e));
        }
        Ok(Ok(Some(if branches.len() == 1 {
            branches.remove(0)
        } else {
            Node::Other(Other::BranchReset, branches)
        })))
    }

    /// `parse_conditional`, after `(?(`.
    fn conditional(&mut self) -> Step<Option<Node>> {
        let saved_flags = self.flags;
        let saved = self.pos;
        if self.get() == Some('?') {
            let assertion = match self.get() {
                Some(c @ ('=' | '!')) => Some((false, c == '=')),
                Some('<') => match self.get() {
                    Some(c @ ('=' | '!')) => Some((true, c == '=')),
                    _ => None,
                },
                _ => None,
            };
            let Some((behind, positive)) = assertion else {
                self.pos = saved;
                return Ok(syntax("expected lookaround conditional"));
            };
            // `parse_lookaround_conditional`: the flags come back after the test only.
            let test = match self.scoped(Self::pattern)? {
                Ok(test) => test,
                Err(e) => return Ok(Err(e)),
            };
            let (yes, no) = match self.branches()? {
                Ok(branches) => branches,
                Err(e) => return Ok(Err(e)),
            };
            let test = Node::Around {
                behind,
                negated: !positive,
                node: Box::new(test),
            };
            return Ok(Ok(Some(Node::Other(
                Other::Conditional,
                vec![test, yes, no],
            ))));
        }
        self.pos = saved;
        let condition = (|| {
            let name = self.name(true)?;
            self.expect(")")?;
            Ok(name)
        })();
        let name = match condition {
            Ok(name) => name,
            Err(e) => return Ok(Err(e)),
        };
        let branches = self.branches()?;
        self.flags = saved_flags;
        self.ignore_space = self.flags & VERBOSE != 0;
        let (yes, no) = match branches {
            Ok(branches) => branches,
            Err(e) => return Ok(Err(e)),
        };
        // An empty conditional is nothing, and its group is not checked.
        if yes.is_empty() && no.is_empty() {
            return Ok(Ok(Some(Node::Cat(Vec::new()))));
        }
        self.reference(Reference::Condition, name);
        Ok(Ok(Some(Node::Other(Other::Conditional, vec![yes, no]))))
    }

    /// The `yes|no` sequences of a conditional, and its `)`.
    fn branches(&mut self) -> Step<(Node, Node)> {
        if let Err(e) = self.enter() {
            return Ok(Err(e));
        }
        let yes = match self.sequence()? {
            Ok(node) => node,
            Err(e) => return Ok(Err(e)),
        };
        let no = if self.eat("|") {
            match self.sequence()? {
                Ok(node) => node,
                Err(e) => return Ok(Err(e)),
            }
        } else {
            Node::Cat(Vec::new())
        };
        self.depth -= 1;
        Ok(self.expect(")").map(|()| (yes, no)))
    }

    /// `parse_flags_subpattern`: inline flags, or a group with scoped flags.
    fn flags_subpattern(&mut self) -> Step<Option<Node>> {
        let (on, off) = match self.inline_flags() {
            Ok(flags) => flags,
            Err(e) => return Ok(Err(e)),
        };
        if off & GLOBAL_FLAGS != 0 {
            return Ok(syntax("bad inline flags: cannot turn off global flag"));
        }
        if on & off != 0 {
            return Ok(syntax("bad inline flags: flag turned on and off"));
        }
        let new_global = on & !self.global_flags & GLOBAL_FLAGS;
        if new_global != 0 {
            return Err(Restart(self.global_flags | new_global));
        }
        let on = on & !GLOBAL_FLAGS;
        self.use_flags(on, off);
        if self.eat(":") {
            // `parse_subpattern`.
            let saved_flags = self.flags;
            self.flags = (self.flags | on) & !off;
            if self.flags & ALL_ENCODINGS != 0 {
                self.flags = (self.flags & !ALL_ENCODINGS) | on;
            }
            self.ignore_space = self.flags & VERBOSE != 0;
            let inner = self.scoped(Self::pattern)?;
            self.flags = saved_flags;
            self.ignore_space = self.flags & VERBOSE != 0;
            return Ok(inner.map(|inner| Some(Node::Group(None, Box::new(inner)))));
        }
        if self.eat(")") {
            // `parse_positional_flags`.
            self.flags = (self.flags | on) & !off;
            self.ignore_space = self.flags & VERBOSE != 0;
            return Ok(Ok(None));
        }
        Ok(syntax("unknown extension"))
    }

    /// `parse_flags`: the flags turned on and off.
    fn inline_flags(&mut self) -> Read<(u32, u32)> {
        let on = self.flag_set();
        if !self.eat("-") {
            return Ok((on, 0));
        }
        match self.flag_set() {
            0 => syntax("bad inline flags: no flags after '-'"),
            off => Ok((on, off)),
        }
    }

    /// `parse_flag_set`.
    fn flag_set(&mut self) -> u32 {
        let mut flags = 0;
        loop {
            let saved = self.pos;
            let mut name = String::new();
            if let Some(c) = self.get() {
                name.push(c);
                if c == 'V' {
                    name.extend(self.get());
                }
            }
            match flag_bit(&name) {
                Some(bit) => flags |= bit,
                None => {
                    self.pos = saved;
                    return flags;
                }
            }
        }
    }

    /// `parse_escape`, after `\`.
    fn escape(&mut self, in_set: bool) -> Read<Escaped> {
        let saved_ignore = self.ignore_space;
        self.ignore_space = false;
        let c = self.get();
        self.ignore_space = saved_ignore;
        let Some(c) = c else {
            return syntax("bad escape (end of pattern)");
        };
        let node = |node| Ok(Escaped::Node(node));
        let other = |kind| Ok(Escaped::Node(Node::Other(kind, Vec::new())));
        match c {
            'x' => self.hex_escape('x', 2),
            'u' => self.hex_escape('u', 4),
            'U' => self.hex_escape('U', 8),
            'g' if !in_set => {
                let saved = self.pos;
                // A reference the module rejects is a literal `g`. A name
                // outside ASCII is not read: Python's `str.isidentifier` and
                // `str.isdigit` decide whether the module reads it as a
                // reference, as a literal, or rejects the pattern.
                match self.group_ref() {
                    Ok(backref) => node(backref),
                    Err(not_read @ ReadError::NotRead(_)) => Err(not_read),
                    Err(ReadError::Syntax(_)) => {
                        self.pos = saved;
                        Ok(Escaped::Char(u32::from('g')))
                    }
                }
            }
            'G' if !in_set => other(Other::SearchAnchor),
            // A named list needs keyword arguments, which transformers does not pass.
            'L' if !in_set => syntax("undefined named list"),
            'N' => self.named_char(),
            'p' | 'P' => self.property(c == 'p'),
            'R' if !in_set => other(Other::LineBreak),
            'X' if !in_set => other(Other::Grapheme),
            c if c.is_ascii_alphabetic() => {
                if !in_set {
                    match c {
                        'A' => return node(Node::Look(Look::Start)),
                        'b' => return node(Node::Look(Look::WordBoundary)),
                        'B' => return node(Node::Look(Look::NotWordBoundary)),
                        'K' => return other(Other::Keep),
                        'm' => return other(Other::WordStart),
                        'M' => return other(Other::WordEnd),
                        'Z' | 'z' => return node(Node::Look(Look::End)),
                        _ => {}
                    }
                }
                let set = match c {
                    'd' => Set::Escape(Escape::Digit),
                    'D' => Set::Escape(Escape::NotDigit),
                    'h' => Set::Property("\\h".into()),
                    's' => Set::Escape(Escape::Space),
                    'S' => Set::Escape(Escape::NotSpace),
                    'w' => Set::Escape(Escape::Word),
                    'W' => Set::Escape(Escape::NotWord),
                    'a' => return Ok(Escaped::Char(0x7)),
                    'b' => return Ok(Escaped::Char(0x8)),
                    'f' => return Ok(Escaped::Char(0xc)),
                    'n' => return Ok(Escaped::Char(0xa)),
                    'r' => return Ok(Escaped::Char(0xd)),
                    't' => return Ok(Escaped::Char(0x9)),
                    'v' => return Ok(Escaped::Char(0xb)),
                    c => return syntax(format!("bad escape \\{c}")),
                };
                node(Node::One(set))
            }
            c if c.is_ascii_digit() => self.numeric_escape(c, in_set),
            c => Ok(Escaped::Char(u32::from(c))),
        }
    }

    /// `parse_numeric_escape`: octal, or a backreference by number.
    fn numeric_escape(&mut self, c: char, in_set: bool) -> Read<Escaped> {
        if in_set || c == '0' {
            return self.octal_escape(c);
        }
        let mut digits = c.to_string();
        let mut saved = self.pos;
        if let Some(d) = self.get().filter(char::is_ascii_digit) {
            digits.push(d);
            saved = self.pos;
            if let Some(o) = self.get().filter(|&o| is_octal_digit(o)) {
                if digits.chars().all(is_octal_digit) {
                    let value = u32::from_str_radix(&format!("{digits}{o}"), 8).unwrap_or(0);
                    return Ok(Escaped::Char(value & 0x1ff));
                }
            }
        }
        self.pos = saved;
        if self.is_open_group(&digits) {
            return syntax("cannot refer to an open group");
        }
        Ok(Escaped::Node(self.backref(digits)))
    }

    /// `parse_octal_escape`: up to three octal digits.
    fn octal_escape(&mut self, first: char) -> Read<Escaped> {
        let mut digits = first.to_string();
        let mut saved = self.pos;
        while digits.len() < 3 {
            match self.get() {
                Some(c) if is_octal_digit(c) => {
                    digits.push(c);
                    saved = self.pos;
                }
                _ => break,
            }
        }
        self.pos = saved;
        match u32::from_str_radix(&digits, 8) {
            Ok(value) => Ok(Escaped::Char(value)),
            Err(_) if is_octal_digit(first) => syntax(format!("incomplete escape \\{digits}")),
            Err(_) => syntax(format!("bad escape \\{first}")),
        }
    }

    /// `parse_hex_escape`.
    fn hex_escape(&mut self, escape: char, len: usize) -> Read<Escaped> {
        let mut digits = String::new();
        for _ in 0..len {
            match self.get() {
                Some(c) if c.is_ascii_hexdigit() => digits.push(c),
                _ => return syntax(format!("incomplete escape \\{escape}{digits}")),
            }
        }
        match u32::from_str_radix(&digits, 16) {
            Ok(value) if value < 0x11_0000 => Ok(Escaped::Char(value)),
            _ => syntax(format!("bad hex escape \\{escape}{digits}")),
        }
    }

    /// `parse_group_ref`, after `\g`.
    fn group_ref(&mut self) -> Read<Node> {
        self.expect("<")?;
        let name = self.name(true)?;
        self.expect(">")?;
        if self.is_open_group(&name) {
            return syntax("cannot refer to an open group");
        }
        Ok(self.backref(name))
    }

    /// `parse_named_char`, after `\N`: a named character is not read (its
    /// name needs the Unicode character database); else a literal `N`.
    fn named_char(&mut self) -> Read<Escaped> {
        let saved = self.pos;
        if self.eat("{") {
            self.get_while(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '-', true);
            if self.eat("}") {
                return Err(ReadError::NotRead("a named character \\N{...}"));
            }
        }
        self.pos = saved;
        Ok(Escaped::Char(u32::from('N')))
    }

    /// `parse_property`, after `\p` or `\P`: `\pL` and the other one-letter
    /// categories; a property in braces is not read (its name needs the
    /// module's property tables); else a literal `p` or `P`.
    fn property(&mut self, positive: bool) -> Read<Escaped> {
        let saved = self.pos;
        let letter = if positive { 'p' } else { 'P' };
        match self.get() {
            Some('{') => {
                self.eat("^");
                self.property_name();
                if self.eat("}") {
                    return Err(ReadError::NotRead("a Unicode property \\p{...}"));
                }
            }
            Some(c) if "CLMNPSZ".contains(c) => {
                let property = Set::Property(format!("\\{letter}{c}").into());
                return Ok(Escaped::Node(Node::One(property)));
            }
            _ => {}
        }
        self.pos = saved;
        Ok(Escaped::Char(u32::from(letter)))
    }

    /// `parse_property_name`: an optional `name:` or `name=` qualifier and a name.
    fn property_name(&mut self) -> (Option<String>, String) {
        let part = |c: char| c.is_ascii_alphanumeric() || " &_-.".contains(c);
        let name = self.get_while(part, false);
        let mut saved = self.pos;
        let named = match self.get() {
            Some(':' | '=') => {
                let value = self.get_while(|c| part(c) || c == '/', false);
                let value = value.trim_matches(' ');
                if value.is_empty() {
                    (None, name)
                } else {
                    saved = self.pos;
                    (Some(name), value.to_owned())
                }
            }
            _ => (None, name),
        };
        self.pos = saved;
        named
    }

    /// `parse_set`, after `[` (version 0: no nested sets or set operations).
    fn set(&mut self) -> Read<Node> {
        let saved_ignore = self.ignore_space;
        self.ignore_space = false;
        let set = self.set_body();
        self.ignore_space = saved_ignore;
        set
    }

    fn set_body(&mut self) -> Read<Node> {
        let negated = self.eat("^");
        let mut class = Class::default();
        // `parse_set_imp_union`: one member, then members up to `]`.
        self.set_member(&mut class)?;
        loop {
            let saved = self.pos;
            if self.eat("]") {
                self.pos = saved;
                break;
            }
            self.set_member(&mut class)?;
        }
        if !self.eat("]") {
            return syntax("missing ]");
        }
        Ok(Node::One(class.into_set(negated)))
    }

    /// `parse_set_member`: an item, or a range of two characters.
    fn set_member(&mut self, class: &mut Class) -> Read<()> {
        let start = self.set_item()?;
        let SetItem::Char(lo) = start else {
            class.add(start);
            return Ok(());
        };
        if !self.eat("-") {
            class.add(start);
            return Ok(());
        }
        let saved = self.pos;
        if self.eat("]") {
            self.pos = saved;
            class.add(start);
            class.add(SetItem::Char(u32::from('-')));
            return Ok(());
        }
        let end = self.set_item()?;
        let SetItem::Char(hi) = end else {
            class.add(start);
            class.add(SetItem::Char(u32::from('-')));
            class.add(end);
            return Ok(());
        };
        if lo > hi {
            return syntax("bad character range");
        }
        class.ranges.push((lo, hi));
        Ok(())
    }

    /// `parse_set_item`: an escape, a POSIX class or one character.
    fn set_item(&mut self) -> Read<SetItem> {
        if self.eat("\\") {
            return Ok(match self.escape(true)? {
                Escaped::Char(c) => SetItem::Char(c),
                Escaped::Node(Node::One(Set::Escape(e))) => SetItem::Escape(e),
                Escaped::Node(Node::One(Set::Property(p))) => SetItem::Property(p),
                // In a set an escape is a character, a class or a property.
                Escaped::Node(_) => return syntax("bad escape in a set"),
            });
        }
        let saved = self.pos;
        if self.eat("[:") {
            match self.posix_class()? {
                Some(name) => return Ok(SetItem::Property(name.into())),
                None => self.pos = saved,
            }
        }
        match self.get() {
            Some(c) => Ok(SetItem::Char(u32::from(c))),
            None => syntax("unterminated character set"),
        }
    }

    /// `parse_posix_class`, after `[:`: the class, or `None` when this is no
    /// POSIX class (the `[` is then a literal). Only the POSIX names are read.
    fn posix_class(&mut self) -> Read<Option<String>> {
        let negated = self.eat("^");
        let (qualifier, name) = self.property_name();
        if !self.eat(":]") {
            return Ok(None);
        }
        // `standardise_name`.
        let standard = name
            .chars()
            .filter(|c| !"_- ".contains(*c))
            .collect::<String>()
            .to_uppercase();
        if qualifier.is_some() || !POSIX_NAMES.contains(&standard.as_str()) {
            return Err(ReadError::NotRead(
                "a POSIX class other than the POSIX names",
            ));
        }
        let caret = if negated { "^" } else { "" };
        Ok(Some(format!("[:{caret}{}:]", standard.to_lowercase())))
    }

    /// Resolve names and numbers, and check what only the whole pattern
    /// shows (`fix_groups` and the end of `_compile`).
    fn finish(self, mut node: Node) -> Read<Regex> {
        if !self.at_end() {
            return syntax("unbalanced parenthesis");
        }
        if ![0, ASCII, LOCALE, UNICODE].contains(&(self.flags & ALL_ENCODINGS)) {
            return syntax("ASCII, LOCALE and UNICODE flags are mutually incompatible");
        }
        let mut numbers = Vec::with_capacity(self.references.len());
        for (kind, name) in &self.references {
            let group = if name.bytes().all(|b| b.is_ascii_digit()) {
                Some(name.parse::<usize>().unwrap_or(usize::MAX))
            } else {
                self.group_number(name)
            };
            let count = self.group_count;
            numbers.push(match (*kind, group) {
                (Reference::Backref | Reference::Condition, None) if name != "DEFINE" => {
                    return syntax("unknown group")
                }
                (Reference::Backref, None) => return syntax("unknown group"),
                (Reference::Backref, Some(g)) if !(1..=count).contains(&g) => {
                    return syntax("invalid group reference")
                }
                (Reference::Call, None) => return syntax("invalid group reference"),
                (Reference::Call, Some(g)) if g > count => return syntax("unknown group"),
                (Reference::Call, Some(g))
                    if g > 0 && self.open_group_count.get(&g).copied().unwrap_or(0) > 1 =>
                {
                    return syntax("ambiguous group reference")
                }
                (Reference::Condition, None) => 0,
                (Reference::Condition, Some(g)) if g > count => {
                    return syntax("invalid group reference")
                }
                (_, Some(g)) => g,
            });
        }
        resolve(&mut node, &numbers);
        let mut optional = vec![false; self.group_count + 1];
        mark_optional(&node, false, &mut optional);
        let mut flags = self.flags_used;
        for &(bit, letter) in &MEANINGFUL {
            if self.global_flags & bit != 0 && !flags.contains(letter) {
                flags.push(letter);
            }
        }
        Ok(Regex {
            node,
            names: self.group_index,
            groups: self.group_count,
            optional,
            flags,
        })
    }
}

/// Put the group numbers into the backreferences, which hold the index of
/// their name or number until the whole pattern is read.
fn resolve(node: &mut Node, numbers: &[usize]) {
    match node {
        Node::Backref(group) => *group = numbers.get(*group).copied().unwrap_or(0),
        Node::Group(_, inner)
        | Node::Around { node: inner, .. }
        | Node::Repeat { node: inner, .. } => resolve(inner, numbers),
        Node::Alt(nodes) | Node::Cat(nodes) | Node::Other(_, nodes) => {
            nodes.iter_mut().for_each(|n| resolve(n, numbers));
        }
        Node::One(_) | Node::Look(_) => {}
    }
}

/// Mark the groups a match can leave unset: those in an alternative, an
/// optional repeat, a lookaround, or a construct of another kind.
fn mark_optional(node: &Node, optional: bool, out: &mut [bool]) {
    match node {
        Node::Group(group, inner) => {
            if let Some(slot) = group.and_then(|g| out.get_mut(g)) {
                *slot |= optional;
            }
            mark_optional(inner, optional, out);
        }
        Node::Alt(nodes) | Node::Other(_, nodes) => {
            nodes.iter().for_each(|n| mark_optional(n, true, out));
        }
        Node::Cat(nodes) => nodes.iter().for_each(|n| mark_optional(n, optional, out)),
        Node::Repeat { node, min, .. } => mark_optional(node, optional || *min == 0, out),
        Node::Around { node, .. } => mark_optional(node, true, out),
        Node::One(_) | Node::Look(_) | Node::Backref(_) => {}
    }
}

fn is_octal_digit(c: char) -> bool {
    ('0'..='7').contains(&c)
}

/// A character by its code point. A lone surrogate (`\ud800`) is a character
/// no text of this crate holds, so it matches nothing.
fn code_point(value: u32) -> Set {
    match char::from_u32(value) {
        Some(c) => Set::Char(c),
        None => Class::default().into_set(false),
    }
}

/// One item of a class.
enum SetItem {
    /// A character by its code point; a surrogate can bound a range.
    Char(u32),
    Escape(Escape),
    Property(Box<str>),
}

/// The members of a class as they are read.
#[derive(Default)]
struct Class {
    ranges: Vec<(u32, u32)>,
    escapes: Vec<Escape>,
    properties: Vec<Box<str>>,
}

impl Class {
    fn add(&mut self, item: SetItem) {
        match item {
            SetItem::Char(c) => self.ranges.push((c, c)),
            SetItem::Escape(e) => {
                if !self.escapes.contains(&e) {
                    self.escapes.push(e);
                }
            }
            SetItem::Property(p) => self.properties.push(p),
        }
    }

    /// The class, without the surrogate code points no text holds. A class
    /// with a class escape and its complement (`[\s\S]`, also `[^\d\D]`) is
    /// any character: the module's `SetUnion.optimise` makes it `AnyAll`,
    /// and drops the negation with it.
    fn into_set(self, negated: bool) -> Set {
        let complement = |e: &Escape| match e {
            Escape::Word => Escape::NotWord,
            Escape::NotWord => Escape::Word,
            Escape::Space => Escape::NotSpace,
            Escape::NotSpace => Escape::Space,
            Escape::Digit => Escape::NotDigit,
            Escape::NotDigit => Escape::Digit,
        };
        if self
            .escapes
            .iter()
            .any(|e| self.escapes.contains(&complement(e)))
        {
            return Set::Any;
        }
        let ranges: Vec<(char, char)> = self
            .ranges
            .into_iter()
            .filter_map(|(lo, hi)| {
                let lo = if (0xd800..=0xdfff).contains(&lo) {
                    0xe000
                } else {
                    lo
                };
                let hi = if (0xd800..=0xdfff).contains(&hi) {
                    0xd7ff
                } else {
                    hi
                };
                Some((char::from_u32(lo)?, char::from_u32(hi)?)).filter(|(lo, hi)| lo <= hi)
            })
            .collect();
        Set::Class {
            negated,
            ranges: ranges.into(),
            escapes: self.escapes.into(),
            properties: self.properties.into(),
        }
    }
}

/// The constraints of a fuzzy setting read so far: `e`, `s`, `i`, `d` and a
/// cost equation.
#[derive(Default)]
struct Constraints {
    e: Option<(i64, Option<i64>)>,
    s: Option<(i64, Option<i64>)>,
    i: Option<(i64, Option<i64>)>,
    d: Option<(i64, Option<i64>)>,
    cost: bool,
}

impl Constraints {
    fn slot(&mut self, c: char) -> Option<&mut Option<(i64, Option<i64>)>> {
        match c {
            'd' => Some(&mut self.d),
            'e' => Some(&mut self.e),
            'i' => Some(&mut self.i),
            's' => Some(&mut self.s),
            _ => None,
        }
    }

    /// `parse_constraint`: one of `d`, `e`, `i`, `s`, not given before.
    fn check(&mut self, c: char) -> Result<(), Fuzzy> {
        match self.slot(c) {
            Some(slot) if slot.is_none() => Ok(()),
            _ => Err(Fuzzy::NotFuzzy),
        }
    }

    fn set(&mut self, c: char, limit: (i64, Option<i64>)) {
        if let Some(slot) = self.slot(c) {
            *slot = Some(limit);
        }
    }
}

/// While reading a fuzzy setting: `ParseError` (this `{` starts none) or a
/// module error.
enum Fuzzy {
    NotFuzzy,
    Error(ReadError),
}

impl Fuzzy {
    fn error(message: &str) -> Self {
        Self::Error(ReadError::Syntax(message.into()))
    }
}

impl From<ReadError> for Fuzzy {
    fn from(e: ReadError) -> Self {
        Self::Error(e)
    }
}
