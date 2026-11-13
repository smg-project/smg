//! The pre-tokenizer patterns of byte-level BPE tokenizers, matched natively.
//!
//! The patterns in use (GPT-2, Qwen, GLM, DeepSeek, MiniMax and their kin)
//! draw on a small regex subset: alternation, concatenation, character
//! classes built from Unicode categories, `\s`, ranges, literals and `\uHHHH`
//! code points, a group of single-character alternatives (read as the class
//! of their union), greedy `?` `*` `+` `{m,n}` and the possessive `?+` `*+`
//! `++` on a class, a case-insensitive group of literal alternatives (or a
//! class of letters) for the English clitics, the lookahead `(?!\S)`, the
//! end anchor `$` (the engine's end of line: the end of the text, or
//! before a newline), the digit class `\d` and the digit-grouping
//! lookahead `(?=(?:\d{3})*\b)` after a digit, read as the count of the
//! digit run ahead and the word boundary after it. This module parses that
//! subset and matches it the way the engine behind `tokenizers`' `Split`
//! does: leftmost-first alternation, greedy quantifiers that back off one
//! repetition at a time. The engine also yields empty matches, and `Split`
//! cuts the text at them; this matcher does not reproduce that, so a pattern
//! that can match the empty string is declined like everything else outside
//! the subset, and the caller keeps the `tokenizers` pipeline for it.
//!
//! The pre-tokenizer of the GPT-4 family (Qwen 2 and 3, Llama 3, GLM 4.6
//! and their kin ship one template, the digit rule's bound apart) is matched
//! by a [`Scanner`] instead of the program, and the cased-letter template
//! (gpt-oss, Llama 4, MiniMax and their kin) by [`cased::Scanner`]: each
//! alternative is decided at a position by a few class tests and class runs,
//! which is what the backtracking engine arrives at for that shape, so the
//! pieces are the same and no character is interpreted op by op. A scanner
//! reads a character's classes as role bits ([`scan::Roles`]). The classes
//! are the parser's, so the matchers cannot read a category differently.
//!
//! Character categories come from [`super::unicode`], generated from that
//! same engine, so `\p{L}` or `\s` cannot mean one thing here and another
//! there.

use super::{
    cased, chain,
    scan::{Bits, Roles, WordScan},
    unicode,
};

/// A set of characters: Unicode categories, code point ranges, negation.
#[derive(Debug, Clone, Default)]
pub(super) struct Class {
    negated: bool,
    categories: u16,
    /// Code point ranges, sorted and disjoint once [`Class::finish`] has run.
    ranges: Vec<(u32, u32)>,
    /// Categories taken out of the class before any negation: the engine's
    /// intersection with a negated property class, `[...&&[^\p{Han}]]`.
    without: u16,
    /// Membership of U+0000..=U+007F, resolved once.
    ascii: AsciiSet,
    /// Membership of U+0000..=U+FFFF, resolved once (empty until
    /// [`Class::finish`]).
    plane: PlaneSet,
}

/// A set of ASCII characters as a membership table, one entry per code
/// point. The matcher asks it once per character it tests (every class test
/// and every first-set test on ASCII text), and a table read is one load
/// where a bit of a 128-bit word costs a two-word select and a variable shift.
#[derive(Clone, Copy)]
struct AsciiSet([bool; 128]);

impl Default for AsciiSet {
    fn default() -> Self {
        Self([false; 128])
    }
}

impl std::fmt::Debug for AsciiSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let members: String = (0u8..128)
            .filter(|&code| self.0[usize::from(code)])
            .map(char::from)
            .collect();
        f.debug_tuple("AsciiSet").field(&members).finish()
    }
}

impl AsciiSet {
    /// Membership of `code`, which the caller has checked is below U+0080.
    #[inline]
    fn contains(&self, code: u32) -> bool {
        debug_assert!(code < 128);
        self.0[(code & 0x7f) as usize]
    }

    fn insert(&mut self, code: u32) {
        self.0[(code & 0x7f) as usize] = true;
    }

    fn union(mut self, other: Self) -> Self {
        for (mine, theirs) in self.0.iter_mut().zip(other.0) {
            *mine |= theirs;
        }
        self
    }
}

/// Words of a [`PlaneSet`]: one bit per code point of the Basic
/// Multilingual Plane.
const PLANE_WORDS: usize = 0x1_0000 / 64;

/// A set of characters of the Basic Multilingual Plane as a bitmap, 8 KiB.
/// A class test on a character beyond ASCII was a binary search over the
/// range table of each of the class's categories, once per character per
/// class test (measured: half of a warm encode of prose with a sixth of its
/// bytes outside ASCII); a bit test is one load. The planes above keep the
/// tables: rare in prompts, and a bit per code point of them would be 136
/// KiB per class.
#[derive(Clone)]
struct PlaneSet(Box<[u64; PLANE_WORDS]>);

impl Default for PlaneSet {
    fn default() -> Self {
        Self(Box::new([0; PLANE_WORDS]))
    }
}

impl std::fmt::Debug for PlaneSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let members: u32 = self.0.iter().map(|word| word.count_ones()).sum();
        f.debug_tuple("PlaneSet").field(&members).finish()
    }
}

impl PlaneSet {
    /// Membership of `code`, which the caller has checked is below U+10000.
    #[inline]
    fn contains(&self, code: u32) -> bool {
        debug_assert!(code < 0x1_0000);
        (self.0[(code >> 6) as usize & (PLANE_WORDS - 1)] >> (code & 63)) & 1 != 0
    }

    /// Inserts every code point of `ranges` that is in the plane.
    fn insert_ranges(&mut self, ranges: &[(u32, u32)]) {
        for &(start, end) in ranges {
            for code in start..=end.min(0xFFFF) {
                self.0[(code >> 6) as usize] |= 1 << (code & 63);
            }
        }
    }

    /// Removes every code point of `ranges` that is in the plane.
    fn remove_ranges(&mut self, ranges: &[(u32, u32)]) {
        for &(start, end) in ranges {
            for code in start..=end.min(0xFFFF) {
                self.0[(code >> 6) as usize] &= !(1 << (code & 63));
            }
        }
    }

    /// The complement within the plane.
    fn negate(&mut self) {
        for word in self.0.iter_mut() {
            *word = !*word;
        }
    }
}

const CAT_LETTER: u16 = 1 << 0;
const CAT_MARK: u16 = 1 << 1;
const CAT_NUMBER: u16 = 1 << 2;
const CAT_PUNCTUATION: u16 = 1 << 3;
const CAT_SYMBOL: u16 = 1 << 4;
const CAT_UPPERCASE: u16 = 1 << 5;
const CAT_LOWERCASE: u16 = 1 << 6;
const CAT_TITLECASE: u16 = 1 << 7;
const CAT_MODIFIER_LETTER: u16 = 1 << 8;
const CAT_OTHER_LETTER: u16 = 1 << 9;
const CAT_SPACE: u16 = 1 << 10;
/// The `Han` script: the one script property the patterns use (a template
/// cuts ideograph runs out before its letter classes and takes them out of
/// those classes).
const CAT_HAN: u16 = 1 << 11;
/// `\d`: the engine's decimal digits.
const CAT_DIGIT: u16 = 1 << 12;
/// `\w`: the engine's word characters, the class its word boundary `\b`
/// reads on either side of a position.
const CAT_WORD: u16 = 1 << 13;

/// Pairs of ASCII letters that are the full case folding of one character
/// (ß ẞ: ss; ﬀ: ff; ﬁ: fi; ﬂ: fl; ﬃ ﬄ: ffi, ffl; ﬅ ﬆ: st), which the engine
/// matches under `(?i:)` and this matcher does not.
const MULTI_CHARACTER_FOLDS: [[char; 2]; 5] =
    [['s', 's'], ['s', 't'], ['f', 'f'], ['f', 'i'], ['f', 'l']];

/// Adds to `class` the ASCII letter `c` in both cases and the characters
/// beyond ASCII the engine folds together with it under `(?i:...)`.
fn fold_letter(class: &mut Class, c: char) {
    let lower = c.to_ascii_lowercase();
    let upper = c.to_ascii_uppercase();
    class.ranges.push((u32::from(lower), u32::from(lower)));
    class.ranges.push((u32::from(upper), u32::from(upper)));
    if let Some((_, extra)) = unicode::FOLD_EXTRA
        .iter()
        .find(|(letter, _)| *letter == lower as u8)
    {
        for &e in *extra {
            class.ranges.push((u32::from(e), u32::from(e)));
        }
    }
}

/// A bracket class under `(?i:...)`: accepted when every member is an ASCII
/// letter, each folded the way a literal letter is (the engine folds a
/// class's members too); a range, a property, a negation or a non-ASCII
/// member would match differently and is declined.
fn folded_letters(class: Class) -> Option<Class> {
    if class.negated || class.categories != 0 || class.without != 0 {
        return None;
    }
    let mut folded = Class::default();
    for &(low, high) in &class.ranges {
        if low != high {
            return None;
        }
        let letter = char::from_u32(low).filter(char::is_ascii_alphabetic)?;
        fold_letter(&mut folded, letter);
    }
    Some(folded)
}

const CATEGORY_TABLES: [(u16, &[(u32, u32)]); 14] = [
    (CAT_LETTER, unicode::LETTER),
    (CAT_MARK, unicode::MARK),
    (CAT_NUMBER, unicode::NUMBER),
    (CAT_PUNCTUATION, unicode::PUNCTUATION),
    (CAT_SYMBOL, unicode::SYMBOL),
    (CAT_UPPERCASE, unicode::UPPERCASE),
    (CAT_LOWERCASE, unicode::LOWERCASE),
    (CAT_TITLECASE, unicode::TITLECASE),
    (CAT_MODIFIER_LETTER, unicode::MODIFIER_LETTER),
    (CAT_OTHER_LETTER, unicode::OTHER_LETTER),
    (CAT_SPACE, unicode::SPACE),
    (CAT_HAN, unicode::HAN),
    (CAT_DIGIT, unicode::DIGIT),
    (CAT_WORD, unicode::WORD),
];

#[inline]
fn in_ranges(ranges: &[(u32, u32)], code: u32) -> bool {
    let index = ranges.partition_point(|&(_, end)| end < code);
    ranges.get(index).is_some_and(|&(start, _)| start <= code)
}

/// Whether `code` is in the table of any category of `categories`.
fn in_categories(categories: u16, code: u32) -> bool {
    CATEGORY_TABLES
        .iter()
        .any(|&(bit, table)| categories & bit != 0 && in_ranges(table, code))
}

impl Class {
    fn finish(mut self) -> Self {
        // `in_ranges` binary-searches on the ends in start order: a range
        // inside another breaks that, so the ranges are sorted here, once
        // for every way a class is built, and the overlapping ones become
        // one.
        self.ranges.sort_unstable();
        self.ranges.dedup_by(|next, kept| {
            if next.0 <= kept.1 {
                kept.1 = kept.1.max(next.1);
                true
            } else {
                false
            }
        });
        let mut ascii = AsciiSet::default();
        for code in 0u32..128 {
            if self.contains_slow(code) {
                ascii.insert(code);
            }
        }
        self.ascii = ascii;
        self.plane = self.plane_set();
        self
    }

    /// The members of the class in the Basic Multilingual Plane: the class's
    /// own ranges and the tables of its categories, the complement for a
    /// negated class (the `hit != negated` of [`Class::contains_slow`]).
    fn plane_set(&self) -> PlaneSet {
        let mut plane = PlaneSet::default();
        plane.insert_ranges(&self.ranges);
        for &(bit, table) in &CATEGORY_TABLES {
            if self.categories & bit != 0 {
                plane.insert_ranges(table);
            }
        }
        for &(bit, table) in &CATEGORY_TABLES {
            if self.without & bit != 0 {
                plane.remove_ranges(table);
            }
        }
        if self.negated {
            plane.negate();
        }
        plane
    }

    fn contains_slow(&self, code: u32) -> bool {
        let mut hit = in_ranges(&self.ranges, code);
        if !hit && self.categories != 0 {
            hit = in_categories(self.categories, code);
        }
        if hit && self.without != 0 {
            hit = !in_categories(self.without, code);
        }
        hit != self.negated
    }

    #[inline]
    pub(super) fn contains(&self, c: char) -> bool {
        let code = u32::from(c);
        if code < 128 {
            self.ascii.contains(code)
        } else if code < 0x1_0000 {
            self.plane.contains(code)
        } else {
            self.contains_slow(code)
        }
    }

    /// Whether the class may hold a character beyond ASCII.
    pub(super) fn may_hold_beyond_ascii(&self) -> bool {
        self.first().non_ascii
    }

    /// Whether every character the class can hold is printable ASCII
    /// (U+0021..=U+007E): no category, no negation, every range within.
    pub(super) fn within_printable_ascii(&self) -> bool {
        !self.negated
            && self.categories == 0
            && self
                .ranges
                .iter()
                .all(|&(start, end)| start >= 0x21 && end <= 0x7E)
    }

    /// The characters the class may hold, exact below U+0080 and one bit for
    /// the rest, set unless every member of the class is ASCII.
    fn first(&self) -> First {
        First {
            ascii: self.ascii,
            non_ascii: self.negated
                || self.categories != 0
                || self.ranges.iter().any(|&(_, end)| end >= 128),
            nullable: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum Node {
    /// One character of the class.
    Class(usize),
    /// Greedy repetition of one class character, `min..=max` times. A
    /// `possessive` run (`?+`, `*+`, `++`) never gives a character back once
    /// taken: when what follows fails, the match backs off to a choice made
    /// before the run, if there is one.
    ClassRepeat {
        class: usize,
        min: u32,
        max: u32,
        possessive: bool,
    },
    /// Greedy repetition of a group, `min..=max` times; `first` is what the
    /// group's body can begin with ([`first_sets`] fills it).
    Repeat {
        node: Box<Node>,
        min: u32,
        max: u32,
        first: First,
    },
    Concat(Vec<Node>),
    /// Alternatives tried in order; the first that lets the rest match wins.
    /// `firsts` holds, per alternative, the characters a match of it can
    /// begin with ([`first_sets`] fills them once the classes are final), so
    /// an alternative that cannot start at a position is not stepped into.
    Alt {
        alternatives: Vec<Node>,
        firsts: Vec<First>,
    },
    /// `(?!...)` over one class: succeeds when the next character is not in it.
    NotAhead(usize),
    /// `$`: succeeds at the end of the text and before a newline (the
    /// engine's end of line, in its default syntax).
    End,
    /// `(?=(?:\d{size})*\b)`: succeeds when the run of the class ahead
    /// holds a multiple of `size` characters and the word boundary holds
    /// after it (the text ends there, or the next character is outside
    /// the `word` class), or when the boundary holds at the position
    /// itself (see [`Op::GroupsAhead`]). The boundary at the position reads
    /// the character before it, so the node stands only behind an atom
    /// that must consume in its alternative ([`groups_ahead_leads`]): the
    /// character it reads is one the match consumed, inside any unit of
    /// the thread pool as in the whole text.
    GroupsAhead {
        class: usize,
        size: u32,
        word: usize,
    },
    /// `([class])\1*` or `([class])\1+`, the capture read back at once: one
    /// character of the class, then that same character while it repeats,
    /// `at_least` characters in all. Taken only as the last atom of a
    /// top-level alternative, where the engine's greedy backreference has
    /// nothing after it to give characters back to, and once per pattern:
    /// `\1` names the pattern's first group wherever it stands, so a second
    /// group declines.
    RunOfOne {
        class: usize,
        at_least: u32,
    },
}

/// The characters a match of a node can begin with: a table of the ASCII
/// code points, one bit for every character beyond ASCII (set when a class
/// may hold any of them), and whether the node can match without consuming a
/// character, in which case any character may follow it.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct First {
    ascii: AsciiSet,
    non_ascii: bool,
    nullable: bool,
}

impl First {
    #[inline]
    fn may_start(&self, c: char) -> bool {
        let code = u32::from(c);
        self.nullable
            || if code < 128 {
                self.ascii.contains(code)
            } else {
                self.non_ascii
            }
    }

    /// The set of this node followed by `next`: `next` widens it only when
    /// this node can match empty.
    fn then(self, next: First) -> First {
        if self.nullable {
            First {
                ascii: self.ascii.union(next.ascii),
                non_ascii: self.non_ascii || next.non_ascii,
                nullable: next.nullable,
            }
        } else {
            self
        }
    }

    fn or(self, other: First) -> First {
        First {
            ascii: self.ascii.union(other.ascii),
            non_ascii: self.non_ascii || other.non_ascii,
            nullable: self.nullable || other.nullable,
        }
    }
}

/// A parsed pattern: the program the matcher runs, compiled once from the
/// tree the parser built.
#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    classes: Vec<Class>,
    program: Vec<Op>,
    /// The first-character sets the `Split` ops index.
    firsts: Vec<First>,
    /// How the pattern is matched: by the scanner of its template's shape,
    /// else by the program.
    matcher: Matcher,
    /// What the pattern is as a stage of a three-stage chain, when it is
    /// one (see [`chain`]).
    part: Option<chain::Part>,
    /// Whether the pattern holds the end anchor (see [`Self::end_anchored`]).
    end_anchored: bool,
    /// Whether the pattern holds the digit-grouping lookahead (see
    /// [`Self::reads_word_boundary`]).
    reads_word_boundary: bool,
}

/// The matcher of a pattern: a hand-written scanner for the two template
/// shapes in use, the program for every other pattern.
#[derive(Debug, Clone)]
enum Matcher {
    /// The compiled program, interpreted (see [`Pattern::match_at`]).
    Program,
    /// The GPT-4 family's template.
    Template(Scanner),
    /// The cased-letter template.
    Cased(cased::Scanner),
}

/// One instruction of a compiled pattern. The tree is flattened into a
/// sequence so that a match is one loop over it: every point where the tree
/// matcher recursed with a continuation (an alternative, a greedy run) is an
/// entry on an explicit stack of [`Choice`]s instead of a stack frame.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// One character of the class.
    Char(usize),
    /// A greedy run of `min..=max` characters of the class; when what follows
    /// fails it gives back one character at a time, down to `min`, unless it
    /// is `possessive`: such a run leaves no choice to come back to.
    Run {
        class: usize,
        min: u32,
        max: u32,
        possessive: bool,
    },
    /// Succeeds when the next character is not in the class.
    NotAhead(usize),
    /// Succeeds at the end of the text and before a newline (`$`).
    End,
    /// The digit-grouping lookahead (see [`Node::GroupsAhead`]).
    GroupsAhead {
        class: usize,
        size: u32,
        word: usize,
    },
    /// One character of the class, then the same character while it repeats
    /// (`at_least` in all); nothing follows it in its alternative, so the run
    /// is taken whole and leaves no choice.
    RunOfOne { class: usize, at_least: u32 },
    /// A choice: the ops that follow are tried first; `next` is where to go
    /// when they fail (the following alternative's `Split`, the op after an
    /// optional group, or [`NO_NEXT`] for the last alternative). `first`
    /// indexes the set of characters the ops that follow can begin with, and
    /// a character outside it takes `next` without an entry on the stack.
    Split { next: usize, first: usize },
    /// The end of an alternative: continue after the alternation.
    Jump(usize),
    /// The whole pattern matched up to the position.
    Match,
}

/// `Split.next` of the last alternative: a failure there backtracks.
const NO_NEXT: usize = usize::MAX;

/// A choice the matcher may come back to: `pc` is the `Split` whose `next`
/// is the way not taken, or the `Run` that may give back characters, in which
/// case `pos` is where its run ends and `count` how many characters it holds.
#[derive(Debug, Clone, Copy, Default)]
struct Choice {
    pc: usize,
    pos: usize,
    count: u32,
}

/// The run of a class the digit-grouping lookahead measured last
/// ([`Op::GroupsAhead`]): the position it measured from, the run's end and
/// the count of the class's characters between. The end is the same for
/// every position inside the run, so an evaluation nearby (a back-off of the
/// digit run before the lookahead moves at most a few characters, the scanner
/// moves forward one) reuses it with the count moved by the characters
/// between, and a long digit run costs one scan instead of one per position.
/// Keyed by the class's index: the parser gives every lookahead of a pattern
/// the same classes, so the ops share the memo.
#[derive(Clone, Copy)]
struct RunMemo {
    class: usize,
    from: usize,
    end: usize,
    count: u32,
}

impl Default for RunMemo {
    /// No run measured: a class index no pattern has, and an empty span.
    fn default() -> Self {
        Self {
            class: usize::MAX,
            from: 1,
            end: 0,
            count: 0,
        }
    }
}

/// Choices a match can hold at once: one per alternation and per run on the
/// deepest path of the tree (four for the GPT-2 and Qwen patterns, six for the
/// largest in use). A pattern that needs more is declined.
const MAX_CHOICES: usize = 32;

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    classes: Vec<Class>,
    case_insensitive: bool,
    /// The digit class and the word class of the first digit-grouping
    /// lookahead, reused by every later one: the ops then read the same
    /// classes under the same indices and share the match loop's one run
    /// memo ([`RunMemo`] is keyed by the index).
    groups_ahead: Option<(usize, usize)>,
    /// Whether the pattern's capturing group has been read: `\1` names the
    /// first group wherever it stands, so a second group declines
    /// ([`Parser::parse_repeated_character`]).
    captured: bool,
}

impl Pattern {
    /// `None` when the pattern uses anything outside the supported subset.
    pub(crate) fn parse(pattern: &str) -> Option<Self> {
        let mut parser = Parser {
            chars: pattern.chars().peekable(),
            classes: Vec::new(),
            case_insensitive: false,
            groups_ahead: None,
            captured: false,
        };
        let mut root = parser.parse_alternation()?;
        if parser.chars.next().is_some() {
            return None;
        }
        if can_match_empty(&root) || groups_ahead_leads(&root, true) || misplaced_run(&root, true) {
            return None;
        }
        first_sets(&mut root, &parser.classes);
        if choices(&root) > MAX_CHOICES {
            return None;
        }
        let matcher = match Scanner::recognize(&root, &parser.classes) {
            Some(scanner) => Matcher::Template(scanner),
            None => match cased::Scanner::recognize(&root, &parser.classes) {
                Some(scanner) => Matcher::Cased(scanner),
                None => Matcher::Program,
            },
        };
        let part = chain::Part::recognize(&root);
        let end_anchored = holds_end(&root);
        let reads_word_boundary = holds_groups_ahead(&root);
        let mut program = Vec::new();
        let mut firsts = Vec::new();
        compile(&root, &mut program, &mut firsts);
        program.push(Op::Match);
        Some(Self {
            classes: parser.classes,
            program,
            firsts,
            matcher,
            part,
            end_anchored,
            reads_word_boundary,
        })
    }

    /// The pattern's classes, which its [`Part`](chain::Part) indexes.
    pub(super) fn classes(&self) -> &[Class] {
        &self.classes
    }

    /// What the pattern is as a stage of a three-stage chain, if anything.
    pub(super) fn part(&self) -> Option<&chain::Part> {
        self.part.as_ref()
    }

    /// Whether every class of the pattern is within printable ASCII and the
    /// pattern holds no end anchor, so that it cuts a text and its byte-level
    /// image at the same places: the anchor reads a `\n`, which the map moves
    /// beyond ASCII, so on the raw text it would hold where the reference, on
    /// the mapped text, sees the middle of a piece.
    pub(super) fn within_printable_ascii(&self) -> bool {
        !self.end_anchored && self.classes.iter().all(Class::within_printable_ascii)
    }

    /// Whether the pattern holds the end anchor `$`: it reads the end of
    /// the text it runs over, so a text it cuts is never split into units
    /// (see `NativeEncoder::stable_cut`).
    pub(super) fn end_anchored(&self) -> bool {
        self.end_anchored
    }

    /// Whether the pattern holds the digit-grouping lookahead, whose word
    /// boundary reads the Hugging Face engine's word class: a file whose own
    /// encoder reads `\b` with another class is declined on its path (see
    /// `NativeEncoder::from_ranks`).
    pub(super) fn reads_word_boundary(&self) -> bool {
        self.reads_word_boundary
    }

    /// This pattern matched by its program alone, the scanner set aside: the
    /// tests hold the two against each other.
    #[cfg(test)]
    pub(super) fn program_only(&self) -> Self {
        Self {
            matcher: Matcher::Program,
            ..self.clone()
        }
    }

    /// Whether a scanner matches this pattern (else the program does).
    #[cfg(test)]
    pub(super) fn has_scanner(&self) -> bool {
        !matches!(self.matcher, Matcher::Program)
    }

    /// The matches in `text`, leftmost first, each continuing after the last.
    pub(crate) fn find_iter<'p, 't>(&'p self, text: &'t str) -> Matches<'p, 't> {
        Matches {
            pattern: self,
            text,
            pos: 0,
            choices: [Choice::default(); MAX_CHOICES],
            memo: RunMemo::default(),
        }
    }

    /// Calls `f` with each match of `text`, leftmost first, each continuing
    /// after the last (what [`Self::find_iter`] yields), the matcher
    /// dispatched once per text: a scanner runs its own loop with `f`
    /// compiled into it, where the iterator costs a call, a frame and an
    /// `Option` written to the stack per piece (measured: a tenth of the
    /// instructions of a warm long encode); the program keeps the iterator.
    /// `words` says the scanners' run loops take their word scans: the loop
    /// is compiled both ways, and the encoder chooses once per call on the
    /// input's length (its `WORD_SCAN_MIN_BYTES`).
    #[inline(always)]
    pub(crate) fn for_each_match<F: FnMut(usize, usize)>(&self, text: &str, words: bool, mut f: F) {
        match &self.matcher {
            Matcher::Template(scanner) if words => {
                scanner.for_each_match::<true, F>(&self.classes, text, &mut f);
            }
            Matcher::Template(scanner) => {
                scanner.for_each_match::<false, F>(&self.classes, text, &mut f);
            }
            Matcher::Cased(scanner) if words => {
                scanner.for_each_match::<true, F>(&self.classes, text, &mut f);
            }
            Matcher::Cased(scanner) => {
                scanner.for_each_match::<false, F>(&self.classes, text, &mut f);
            }
            Matcher::Program => {
                for (start, end) in self.find_iter(text) {
                    f(start, end);
                }
            }
        }
    }

    /// The run of `class` characters from `pos`: its end and their count,
    /// through `memo` (see [`RunMemo`]). A position inside the run measured
    /// last reuses that run's end, the count moved by the characters between
    /// the two positions (forward, every character between is in the run;
    /// backward, each is checked); any other position scans the run.
    fn class_run(&self, text: &str, pos: usize, class: usize, memo: &mut RunMemo) -> (usize, u32) {
        let set = &self.classes[class];
        if memo.class == class && memo.from <= pos && pos <= memo.end {
            let mut p = memo.from;
            let mut between = 0u32;
            while p < pos {
                match next_char(text, p) {
                    Some((_, len)) => {
                        p += len;
                        between += 1;
                    }
                    None => break,
                }
            }
            if p == pos {
                let count = memo.count - between;
                *memo = RunMemo {
                    class,
                    from: pos,
                    end: memo.end,
                    count,
                };
                return (memo.end, count);
            }
        } else if memo.class == class && pos < memo.from && memo.from <= memo.end {
            let mut p = pos;
            let mut between = 0u32;
            while p < memo.from {
                match next_char(text, p) {
                    Some((c, len)) if set.contains(c) => {
                        p += len;
                        between += 1;
                    }
                    _ => break,
                }
            }
            if p == memo.from {
                let count = memo.count + between;
                *memo = RunMemo {
                    class,
                    from: pos,
                    end: memo.end,
                    count,
                };
                return (memo.end, count);
            }
        }
        let mut count = 0u32;
        let mut end = pos;
        while let Some((c, len)) = next_char(text, end) {
            if !set.contains(c) {
                break;
            }
            end += len;
            count += 1;
        }
        *memo = RunMemo {
            class,
            from: pos,
            end,
            count,
        };
        (end, count)
    }

    /// The end of the match that starts exactly at `pos`, if any: the program
    /// run from its first op, alternatives in order and runs longest first,
    /// the first path that reaches `Match` winning, as the engine does it.
    fn match_at(
        &self,
        text: &str,
        pos: usize,
        choices: &mut [Choice; MAX_CHOICES],
        memo: &mut RunMemo,
    ) -> Option<usize> {
        let mut depth = 0usize;
        let mut pc = 0usize;
        let mut pos = pos;
        loop {
            let failed = match self.program[pc] {
                Op::Char(class) => match next_char(text, pos) {
                    Some((c, len)) if self.classes[class].contains(c) => {
                        pos += len;
                        pc += 1;
                        false
                    }
                    _ => true,
                },
                Op::Run {
                    class,
                    min,
                    max,
                    possessive,
                } => {
                    let class = &self.classes[class];
                    let mut count = 0u32;
                    let mut p = pos;
                    while count < max {
                        match next_char(text, p) {
                            Some((c, len)) if class.contains(c) => {
                                p += len;
                                count += 1;
                            }
                            _ => break,
                        }
                    }
                    if count < min {
                        true
                    } else {
                        if !possessive {
                            choices[depth] = Choice { pc, pos: p, count };
                            depth += 1;
                        }
                        pos = p;
                        pc += 1;
                        false
                    }
                }
                Op::NotAhead(class) => match next_char(text, pos) {
                    Some((c, _)) if self.classes[class].contains(c) => true,
                    _ => {
                        pc += 1;
                        false
                    }
                },
                Op::End => {
                    if pos == text.len() || text.as_bytes()[pos] == b'\n' {
                        pc += 1;
                        false
                    } else {
                        true
                    }
                }
                Op::GroupsAhead { class, size, word } => {
                    // The engine takes the whole run of the class for the
                    // repeated group and backs off one group at a time. A
                    // boundary never falls between two characters of the
                    // class (each is a word character), so it holds either
                    // after the whole run, when the run holds a multiple of
                    // `size` characters and ends the text or meets a
                    // character outside the word class, or at the position
                    // itself, between the character before it and the one
                    // at it. The run is measured once per run, not once per
                    // position (see [`RunMemo`]).
                    let (end, run) = self.class_run(text, pos, class, memo);
                    let word = &self.classes[word];
                    let word_at =
                        |p: usize| next_char(text, p).is_some_and(|(c, _)| word.contains(c));
                    let word_before = text[..pos]
                        .chars()
                        .next_back()
                        .is_some_and(|c| word.contains(c));
                    let after_run = run > 0 && run.is_multiple_of(size) && !word_at(end);
                    if after_run || word_before != word_at(pos) {
                        pc += 1;
                        false
                    } else {
                        true
                    }
                }
                Op::RunOfOne { class, at_least } => match next_char(text, pos) {
                    Some((c, len)) if self.classes[class].contains(c) => {
                        let mut count = 1u32;
                        let mut p = pos + len;
                        while let Some((d, n)) = next_char(text, p) {
                            if d != c {
                                break;
                            }
                            p += n;
                            count += 1;
                        }
                        if count < at_least {
                            true
                        } else {
                            pos = p;
                            pc += 1;
                            false
                        }
                    }
                    _ => true,
                },
                Op::Split { next, first } => {
                    let excluded = match next_char(text, pos) {
                        Some((c, _)) => !self.firsts[first].may_start(c),
                        None => false,
                    };
                    if excluded {
                        if next == NO_NEXT {
                            true
                        } else {
                            pc = next;
                            false
                        }
                    } else {
                        if next != NO_NEXT {
                            choices[depth] = Choice { pc, pos, count: 0 };
                            depth += 1;
                        }
                        pc += 1;
                        false
                    }
                }
                Op::Jump(target) => {
                    pc = target;
                    false
                }
                Op::Match => return Some(pos),
            };
            if !failed {
                continue;
            }
            // Back to the latest choice: a run gives back one character and
            // goes on after itself; an exhausted run or a split is dropped,
            // the split's other way taken.
            loop {
                if depth == 0 {
                    return None;
                }
                let choice = &mut choices[depth - 1];
                match self.program[choice.pc] {
                    Op::Run { min, .. } if choice.count > min => {
                        let mut p = choice.pos - 1;
                        while !text.is_char_boundary(p) {
                            p -= 1;
                        }
                        choice.pos = p;
                        choice.count -= 1;
                        pos = p;
                        pc = choice.pc + 1;
                        break;
                    }
                    Op::Run { .. } => depth -= 1,
                    Op::Split { next, .. } => {
                        pos = choice.pos;
                        pc = next;
                        depth -= 1;
                        break;
                    }
                    // No other op leaves a choice.
                    _ => depth -= 1,
                }
            }
        }
    }
}

/// A part of the GPT-4 family's template that one class fills: its bit in
/// a character's role bits.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Role {
    /// The optional lead-in of a letter run (`[^\r\n\p{L}\p{N}]?`).
    WordPrefix,
    /// A letter run (`\p{L}+`).
    Word,
    /// A digit (`\p{N}`, up to the digit cap at once).
    Digit,
    /// The optional space before a punctuation run (` ?`).
    PunctPrefix,
    /// A punctuation run (`[^\s\p{L}\p{N}]+`).
    Punct,
    /// The line breaks after a punctuation run (`[\r\n]*`).
    PunctBreak,
    /// The whitespace before line breaks (`\s*`).
    BreakSpace,
    /// A run of line breaks (`[\r\n]+`).
    Break,
    /// Whitespace that leaves its last character to what follows (`\s+`
    /// before `(?!\S)`).
    TrailSpace,
    /// What may not follow that whitespace (`\S`).
    NotTrail,
    /// A whitespace run (`\s+`).
    Space,
}

/// The roles, in the order of their bits.
const ROLES: [Role; 11] = [
    Role::WordPrefix,
    Role::Word,
    Role::Digit,
    Role::PunctPrefix,
    Role::Punct,
    Role::PunctBreak,
    Role::BreakSpace,
    Role::Break,
    Role::TrailSpace,
    Role::NotTrail,
    Role::Space,
];

impl Role {
    #[inline(always)]
    const fn bit(self) -> Bits {
        1 << (self as u32)
    }
}

/// The matcher of the GPT-4 family's pre-tokenizer template,
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// with whatever classes stand in those places and whatever bound the digit
/// rule has (Qwen 2 and 3 take one digit, Llama 3 and GLM 4.6 up to three,
/// Qwen 3.5 counts marks among the letters). For this shape, leftmost-first
/// alternation over greedy runs comes down to a decision per alternative
/// that one scan settles:
///
/// 1. a clitic: the first alternative whose characters follow in order;
/// 2. a letter run, with the lead-in when a letter follows it;
/// 3. up to the bound of digits;
/// 4. a punctuation run, with the space when punctuation follows it, and
///    the line breaks after the run;
/// 5. whitespace ending in line breaks: the whitespace run up to and
///    including its last line break, when it holds one;
/// 6. whitespace before a non-space: the run short of its last character,
///    or whole at the end of the text, when it has a character to give;
/// 7. the whitespace run.
///
/// Each is what the engine's backtracking arrives at: a greedy optional
/// lead-in is kept only when the run after it can start, a greedy run before
/// a required class gives back characters until the class follows, and the
/// first alternative to reach the end of the pattern wins.
#[derive(Debug, Clone)]
struct Scanner {
    roles: Roles,
    /// The clitic alternatives, each the classes of its characters in order.
    clitics: Vec<Vec<usize>>,
    /// Per ASCII code point, whether a clitic alternative may begin with it.
    clitic_start: [bool; 128],
    /// Whether a clitic alternative may begin beyond ASCII.
    clitic_beyond_ascii: bool,
    /// The most digits the digit alternative takes at once.
    digit_cap: u32,
}

impl Scanner {
    /// The scanner of `root`, when it is the template; `None` for every other
    /// shape.
    fn recognize(root: &Node, classes: &[Class]) -> Option<Self> {
        let Node::Alt { alternatives, .. } = root else {
            return None;
        };
        let [clitics, word, digits, punct, breaks, trail, space] = alternatives.as_slice() else {
            return None;
        };
        let Node::Alt {
            alternatives: clitic_alternatives,
            ..
        } = clitics
        else {
            return None;
        };
        let clitics = clitic_alternatives
            .iter()
            .map(|alternative| match alternative {
                Node::Class(class) => Some(vec![*class]),
                Node::Concat(nodes) => nodes.iter().map(plain_class).collect(),
                _ => None,
            })
            .collect::<Option<Vec<Vec<usize>>>>()?;
        if clitics.iter().any(Vec::is_empty) {
            return None;
        }
        let [word_prefix, word_run] = concat(word)? else {
            return None;
        };
        let (digit, digit_cap) = match digits {
            Node::Class(class) => (*class, 1),
            Node::ClassRepeat {
                class,
                min: 1,
                max,
                possessive: false,
            } => (*class, *max),
            _ => return None,
        };
        let [punct_prefix, punct_run, punct_break] = concat(punct)? else {
            return None;
        };
        let [break_space, break_run] = concat(breaks)? else {
            return None;
        };
        let [trail_space, not_trail] = concat(trail)? else {
            return None;
        };
        let Node::NotAhead(not_trail) = not_trail else {
            return None;
        };
        let of = |role: Role| -> Option<usize> {
            Some(match role {
                Role::WordPrefix => repeat(word_prefix, 0, 1)?,
                Role::Word => repeat(word_run, 1, u32::MAX)?,
                Role::Digit => digit,
                Role::PunctPrefix => repeat(punct_prefix, 0, 1)?,
                Role::Punct => repeat(punct_run, 1, u32::MAX)?,
                Role::PunctBreak => repeat(punct_break, 0, u32::MAX)?,
                Role::BreakSpace => repeat(break_space, 0, u32::MAX)?,
                Role::Break => repeat(break_run, 1, u32::MAX)?,
                Role::TrailSpace => repeat(trail_space, 1, u32::MAX)?,
                Role::NotTrail => *not_trail,
                Role::Space => repeat(space, 1, u32::MAX)?,
            })
        };
        let role_classes = ROLES
            .iter()
            .map(|&role| of(role).map(|class| &classes[class]))
            .collect::<Option<Vec<&Class>>>()?;
        let mut clitic_start = [false; 128];
        let mut clitic_beyond_ascii = false;
        for alternative in &clitics {
            let class = &classes[alternative[0]];
            for (code, start) in clitic_start.iter_mut().enumerate() {
                *start |= class.contains(char::from(code as u8));
            }
            clitic_beyond_ascii |= class.may_hold_beyond_ascii();
        }
        Some(Self {
            roles: Roles::build(&role_classes),
            clitics,
            clitic_start,
            clitic_beyond_ascii,
            digit_cap,
        })
    }

    /// The end of the piece that starts at `pos`, if one does: the first of
    /// the seven alternatives that matches there, as the engine picks it.
    /// Compiled into its callers' loops: out of line it was a call per
    /// piece, and every value the loop keeps across a call is a spill.
    #[inline(always)]
    fn match_at<const WORDS: bool>(
        &self,
        classes: &[Class],
        text: &str,
        pos: usize,
    ) -> Option<usize> {
        let (c, len) = self.roles.at(text, pos)?;
        let after = pos + len;
        // 1. A clitic: the first alternative whose characters follow in order.
        let first = text.as_bytes()[pos];
        let may_start_clitic = if first < 0x80 {
            self.clitic_start[usize::from(first)]
        } else {
            self.clitic_beyond_ascii
        };
        if may_start_clitic {
            if let Some(end) = clitic_match(classes, text, pos, &self.clitics) {
                return Some(end);
            }
        }
        // 2. A letter run, with the lead-in when a letter follows it.
        if c & Role::WordPrefix.bit() != 0 {
            if let Some((d, n)) = self.roles.at(text, after) {
                if d & Role::Word.bit() != 0 {
                    return Some(self.roles.run_end_as::<{ WordScan::LETTERS }, WORDS>(
                        text,
                        after + n,
                        Role::Word.bit(),
                    ));
                }
            }
        }
        if c & Role::Word.bit() != 0 {
            return Some(self.roles.run_end_as::<{ WordScan::LETTERS }, WORDS>(
                text,
                after,
                Role::Word.bit(),
            ));
        }
        // 3. Up to the bound of digits.
        if c & Role::Digit.bit() != 0 {
            let mut end = after;
            let mut count = 1;
            while count < self.digit_cap {
                match self.roles.at(text, end) {
                    Some((d, n)) if d & Role::Digit.bit() != 0 => {
                        end += n;
                        count += 1;
                    }
                    _ => break,
                }
            }
            return Some(end);
        }
        // 4. A punctuation run, with the space when punctuation follows it,
        //    and the line breaks after the run.
        if c & Role::PunctPrefix.bit() != 0 {
            if let Some((d, n)) = self.roles.at(text, after) {
                if d & Role::Punct.bit() != 0 {
                    let end = self.roles.run_end(text, after + n, Role::Punct.bit());
                    return Some(self.roles.run_end(text, end, Role::PunctBreak.bit()));
                }
            }
        }
        if c & Role::Punct.bit() != 0 {
            let end = self.roles.run_end(text, after, Role::Punct.bit());
            return Some(self.roles.run_end(text, end, Role::PunctBreak.bit()));
        }
        // 5. Whitespace ending in line breaks: the greedy whitespace run
        //    gives back to the last position where a line break stands,
        //    inside the run or right after it.
        let mut end = pos;
        let mut last_break = None;
        loop {
            match self.roles.at(text, end) {
                Some((d, n)) if d & Role::BreakSpace.bit() != 0 => {
                    if d & Role::Break.bit() != 0 {
                        last_break = Some(end);
                    }
                    end += n;
                }
                Some((d, _)) if d & Role::Break.bit() != 0 => {
                    last_break = Some(end);
                    break;
                }
                _ => break,
            }
        }
        if let Some(start) = last_break {
            return Some(self.roles.run_end(text, start, Role::Break.bit()));
        }
        // 6. Whitespace before a non-space: the greedy run gives back its last
        //    character while a non-space follows it, down to one.
        if c & Role::TrailSpace.bit() != 0 {
            let mut end = self.roles.run_end_as::<{ WordScan::SPACES }, WORDS>(
                text,
                after,
                Role::TrailSpace.bit(),
            );
            loop {
                match self.roles.at(text, end) {
                    Some((d, _)) if d & Role::NotTrail.bit() != 0 => {}
                    _ => return Some(end),
                }
                let shorter = previous_boundary(text, end);
                if shorter == pos {
                    break;
                }
                end = shorter;
            }
        }
        // 7. The whitespace run.
        if c & Role::Space.bit() != 0 {
            return Some(self.roles.run_end_as::<{ WordScan::SPACES }, WORDS>(
                text,
                after,
                Role::Space.bit(),
            ));
        }
        None
    }

    /// Calls `f` with each piece of `text` in order (see
    /// [`Pattern::for_each_match`]): a position no alternative matches at
    /// is passed over by one character, as the iterator does.
    #[inline(always)]
    fn for_each_match<const WORDS: bool, F: FnMut(usize, usize)>(
        &self,
        classes: &[Class],
        text: &str,
        f: &mut F,
    ) {
        // The closure stays inside the encoder's loop (the compiler's budget
        // would leave it out of line with the word scans compiled into
        // `match_at`, and the call per piece would be back).
        each_match(
            text,
            #[inline(always)]
            |pos| self.match_at::<WORDS>(classes, text, pos),
            f,
        );
    }
}

/// The end of the first of `clitics` (each the classes of its characters in
/// order) whose characters follow at `pos`, if one does.
#[inline(never)]
pub(super) fn clitic_match(
    classes: &[Class],
    text: &str,
    pos: usize,
    clitics: &[Vec<usize>],
) -> Option<usize> {
    'alternatives: for alternative in clitics {
        let mut p = pos;
        for &class in alternative {
            match next_char(text, p) {
                Some((d, n)) if classes[class].contains(d) => p += n,
                _ => continue 'alternatives,
            }
        }
        return Some(p);
    }
    None
}

/// The class of a node that is one character of a class.
pub(super) fn plain_class(node: &Node) -> Option<usize> {
    match node {
        Node::Class(class) => Some(*class),
        _ => None,
    }
}

/// The class of a node that is a run of `min..=max` characters of one class.
pub(super) fn repeat(node: &Node, min: u32, max: u32) -> Option<usize> {
    match node {
        Node::ClassRepeat {
            class,
            min: node_min,
            max: node_max,
            possessive: false,
        } if *node_min == min && *node_max == max => Some(*class),
        _ => None,
    }
}

/// The nodes of a concatenation.
pub(super) fn concat(node: &Node) -> Option<&[Node]> {
    match node {
        Node::Concat(nodes) => Some(nodes),
        _ => None,
    }
}

/// The start of the character before `pos`, a character boundary past 0.
#[inline]
pub(super) fn previous_boundary(text: &str, pos: usize) -> usize {
    let mut p = pos - 1;
    while !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// Appends the ops of `node` to `program`; the first-character sets the
/// splits index go to `firsts`. Alternatives are laid out in order, each
/// ending in a jump past the alternation, so that the program tries them the
/// way the tree matcher did.
fn compile(node: &Node, program: &mut Vec<Op>, firsts: &mut Vec<First>) {
    match node {
        Node::Class(class) => program.push(Op::Char(*class)),
        Node::ClassRepeat {
            class,
            min,
            max,
            possessive,
        } => program.push(Op::Run {
            class: *class,
            min: *min,
            max: *max,
            possessive: *possessive,
        }),
        Node::Repeat {
            node,
            min,
            max,
            first,
        } => match (*min, *max) {
            // `{0}`: matches empty, nothing to run.
            (_, 0) => {}
            // The optional group: its body first, the way past it when the
            // body fails (or the character cannot begin it).
            (0, _) => {
                let split = program.len();
                let set = firsts.len();
                firsts.push(*first);
                program.push(Op::Split {
                    next: NO_NEXT,
                    first: set,
                });
                compile(node, program, firsts);
                program[split] = Op::Split {
                    next: program.len(),
                    first: set,
                };
            }
            // `{1}`: the body itself.
            _ => compile(node, program, firsts),
        },
        Node::Concat(nodes) => {
            for node in nodes {
                compile(node, program, firsts);
            }
        }
        Node::Alt {
            alternatives,
            firsts: sets,
        } => {
            // The sets are filled by `first_sets` before the compile; an
            // alternation without them would compile to no ops and match
            // empty. Once per pattern, so the check stays in release builds.
            assert_eq!(sets.len(), alternatives.len());
            let mut previous_split = None;
            let mut jumps = Vec::new();
            for (index, (alternative, first)) in alternatives.iter().zip(sets).enumerate() {
                let split = program.len();
                if let Some(previous) = previous_split {
                    if let Op::Split { first: set, .. } = program[previous] {
                        program[previous] = Op::Split {
                            next: split,
                            first: set,
                        };
                    }
                }
                program.push(Op::Split {
                    next: NO_NEXT,
                    first: firsts.len(),
                });
                firsts.push(*first);
                previous_split = Some(split);
                compile(alternative, program, firsts);
                if index + 1 < alternatives.len() {
                    jumps.push(program.len());
                    program.push(Op::Jump(NO_NEXT));
                }
            }
            let end = program.len();
            for jump in jumps {
                program[jump] = Op::Jump(end);
            }
        }
        Node::NotAhead(class) => program.push(Op::NotAhead(*class)),
        Node::End => program.push(Op::End),
        Node::GroupsAhead { class, size, word } => program.push(Op::GroupsAhead {
            class: *class,
            size: *size,
            word: *word,
        }),
        Node::RunOfOne { class, at_least } => program.push(Op::RunOfOne {
            class: *class,
            at_least: *at_least,
        }),
    }
}

/// The most choices a match of `node` can hold at once: every run and every
/// alternation on a path leaves one until the path is done or backs off.
fn choices(node: &Node) -> usize {
    match node {
        Node::Class(_)
        | Node::NotAhead(_)
        | Node::End
        | Node::GroupsAhead { .. }
        | Node::RunOfOne { .. } => 0,
        Node::ClassRepeat { possessive, .. } => usize::from(!*possessive),
        Node::Repeat { node, max, .. } => {
            if *max == 0 {
                0
            } else {
                1 + choices(node)
            }
        }
        Node::Concat(nodes) => nodes.iter().map(choices).sum(),
        Node::Alt { alternatives, .. } => 1 + alternatives.iter().map(choices).max().unwrap_or(0),
    }
}

/// Iterator over the matches of a [`Pattern`] in a text.
pub(crate) struct Matches<'p, 't> {
    pattern: &'p Pattern,
    text: &'t str,
    pos: usize,
    /// The matcher's stack, held here so that one text's matches share it.
    choices: [Choice; MAX_CHOICES],
    /// The digit run measured last, shared by one text's matches the same way.
    memo: RunMemo,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        while self.pos < self.text.len() {
            let start = self.pos;
            let end = match &self.pattern.matcher {
                Matcher::Template(scanner) => {
                    scanner.match_at::<false>(&self.pattern.classes, self.text, start)
                }
                Matcher::Cased(scanner) => {
                    scanner.match_at::<false>(&self.pattern.classes, self.text, start)
                }
                Matcher::Program => {
                    self.pattern
                        .match_at(self.text, start, &mut self.choices, &mut self.memo)
                }
            };
            match end {
                Some(end) if end > start => {
                    self.pos = end;
                    return Some((start, end));
                }
                _ => {
                    // No match starts here: move on by one character. (A
                    // pattern that can match empty is declined at parse time,
                    // so an empty match does not occur; it would count as no
                    // match.)
                    let (_, len) = next_char(self.text, start)?;
                    self.pos = start + len;
                }
            }
        }
        None
    }
}

/// Whether `node` can match the empty string. The engine yields empty
/// matches (all but one right after the previous match), and `Split` cuts the
/// text at them: `\p{N}*` over "ab1" gives the pieces "a", "b", "1". This
/// matcher does not reproduce that, so a pattern whose root can match empty is
/// declined at parse time and no stage ever sees an empty match.
fn can_match_empty(node: &Node) -> bool {
    match node {
        Node::Class(_) => false,
        Node::ClassRepeat { min, .. } => *min == 0,
        Node::Repeat { node, min, .. } => *min == 0 || can_match_empty(node),
        Node::Concat(nodes) => nodes.iter().all(can_match_empty),
        Node::Alt { alternatives, .. } => alternatives.iter().any(can_match_empty),
        Node::NotAhead(_) | Node::End | Node::GroupsAhead { .. } => true,
        Node::RunOfOne { .. } => false,
    }
}

/// Whether a [`Node::GroupsAhead`] under `node` can be evaluated at the
/// start of a match (`leading` says nothing that must consume stands before
/// `node` in its alternative): its word boundary would read the character
/// before the match, which a unit of the thread pool does not hold where the
/// whole text does, so such a pattern declines.
fn groups_ahead_leads(node: &Node, leading: bool) -> bool {
    match node {
        Node::GroupsAhead { .. } => leading,
        Node::Concat(nodes) => {
            let mut leading = leading;
            nodes.iter().any(|node| {
                let leads = groups_ahead_leads(node, leading);
                leading = leading && can_match_empty(node);
                leads
            })
        }
        Node::Alt { alternatives, .. } => alternatives
            .iter()
            .any(|alternative| groups_ahead_leads(alternative, leading)),
        Node::Repeat { node, .. } => groups_ahead_leads(node, leading),
        Node::Class(_)
        | Node::ClassRepeat { .. }
        | Node::NotAhead(_)
        | Node::End
        | Node::RunOfOne { .. } => false,
    }
}

/// Whether `node` holds the digit-grouping lookahead anywhere.
fn holds_groups_ahead(node: &Node) -> bool {
    match node {
        Node::GroupsAhead { .. } => true,
        Node::Class(_)
        | Node::ClassRepeat { .. }
        | Node::NotAhead(_)
        | Node::End
        | Node::RunOfOne { .. } => false,
        Node::Repeat { node, .. } => holds_groups_ahead(node),
        Node::Concat(nodes) => nodes.iter().any(holds_groups_ahead),
        Node::Alt { alternatives, .. } => alternatives.iter().any(holds_groups_ahead),
    }
}

/// Whether `node` holds the end anchor anywhere: such a pattern reads the
/// end of the text it runs over.
fn holds_end(node: &Node) -> bool {
    match node {
        Node::End => true,
        Node::Class(_)
        | Node::ClassRepeat { .. }
        | Node::NotAhead(_)
        | Node::GroupsAhead { .. }
        | Node::RunOfOne { .. } => false,
        Node::Repeat { node, .. } => holds_end(node),
        Node::Concat(nodes) => nodes.iter().any(holds_end),
        Node::Alt { alternatives, .. } => alternatives.iter().any(holds_end),
    }
}

/// Whether a [`Node::RunOfOne`] sits anywhere but at the end of a top-level
/// alternative (`last` says the node ends one): there the engine's greedy
/// backreference could give characters back to what follows, which the run
/// never does, so such a pattern declines.
fn misplaced_run(node: &Node, last: bool) -> bool {
    match node {
        Node::RunOfOne { .. } => !last,
        Node::Concat(nodes) => nodes
            .iter()
            .enumerate()
            .any(|(i, node)| misplaced_run(node, last && i + 1 == nodes.len())),
        Node::Alt { alternatives, .. } => alternatives
            .iter()
            .any(|alternative| misplaced_run(alternative, last)),
        Node::Repeat { node, .. } => misplaced_run(node, false),
        Node::Class(_)
        | Node::ClassRepeat { .. }
        | Node::NotAhead(_)
        | Node::End
        | Node::GroupsAhead { .. } => false,
    }
}

/// Fills the first-character sets of every alternation under `node` and
/// returns `node`'s own. It runs once per pattern, after the parser has
/// finished the classes.
fn first_sets(node: &mut Node, classes: &[Class]) -> First {
    match node {
        Node::Class(class) => classes[*class].first(),
        Node::ClassRepeat { class, min, .. } => First {
            nullable: *min == 0,
            ..classes[*class].first()
        },
        Node::Repeat {
            node, min, first, ..
        } => {
            *first = first_sets(node, classes);
            First {
                nullable: *min == 0 || first.nullable,
                ..*first
            }
        }
        Node::Concat(nodes) => {
            // Every node is visited (an alternation deeper in the run needs
            // its sets too); only the nullable prefix widens the run's own.
            let mut first = First {
                nullable: true,
                ..First::default()
            };
            for node in nodes {
                let next = first_sets(node, classes);
                first = first.then(next);
            }
            first
        }
        Node::Alt {
            alternatives,
            firsts,
        } => {
            *firsts = alternatives
                .iter_mut()
                .map(|alternative| first_sets(alternative, classes))
                .collect();
            firsts
                .iter()
                .fold(First::default(), |all, first| all.or(*first))
        }
        Node::NotAhead(_) | Node::End | Node::GroupsAhead { .. } => First {
            nullable: true,
            ..First::default()
        },
        Node::RunOfOne { class, .. } => classes[*class].first(),
    }
}

/// Calls `f` with each match of `text` in order, each continuing after the
/// last, `match_at(pos)` giving the end of the match that starts at `pos`:
/// a position no alternative matches at is passed over by one character,
/// as [`Matches::next`] does. The one loop behind the scanners'
/// `for_each_match`, compiled into each with its own `match_at`.
#[inline(always)]
pub(super) fn each_match<M, F>(text: &str, mut match_at: M, f: &mut F)
where
    M: FnMut(usize) -> Option<usize>,
    F: FnMut(usize, usize),
{
    let mut pos = 0;
    while pos < text.len() {
        match match_at(pos) {
            Some(end) if end > pos => {
                f(pos, end);
                pos = end;
            }
            _ => {
                let Some((_, len)) = next_char(text, pos) else {
                    return;
                };
                pos += len;
            }
        }
    }
}

#[inline]
pub(super) fn next_char(text: &str, pos: usize) -> Option<(char, usize)> {
    let byte = *text.as_bytes().get(pos)?;
    if byte < 0x80 {
        return Some((byte as char, 1));
    }
    let c = text[pos..].chars().next()?;
    Some((c, c.len_utf8()))
}

impl Parser<'_> {
    fn add_class(&mut self, class: Class) -> usize {
        self.classes.push(class.finish());
        self.classes.len() - 1
    }

    /// The ASCII letters a class holds, in lower case, once each.
    fn ascii_letters(&self, class: usize) -> Vec<char> {
        let class = &self.classes[class];
        ('a'..='z')
            .filter(|&letter| class.contains(letter) || class.contains(letter.to_ascii_uppercase()))
            .collect()
    }

    fn parse_alternation(&mut self) -> Option<Node> {
        let mut alternatives = vec![self.parse_concat()?];
        while self.chars.peek() == Some(&'|') {
            self.chars.next();
            alternatives.push(self.parse_concat()?);
        }
        Some(if alternatives.len() == 1 {
            alternatives.pop()?
        } else {
            Node::Alt {
                alternatives,
                firsts: Vec::new(),
            }
        })
    }

    fn parse_concat(&mut self) -> Option<Node> {
        let mut nodes = Vec::new();
        let mut previous_letters: Vec<char> = Vec::new();
        while let Some(&c) = self.chars.peek() {
            if c == '|' || c == ')' {
                break;
            }
            nodes.push(self.parse_quantified()?);
            // Under `(?i:)` the engine also matches a run of letters against
            // the one character whose full case folding is that run (ß and ẞ
            // for "ss", ﬁ for "fi", ﬆ for "st", ...), which a matcher that
            // folds letter by letter cannot do: such runs are declined, a
            // letter class standing for each letter it holds.
            let letters = match nodes.last() {
                Some(Node::Class(class)) if self.case_insensitive => self.ascii_letters(*class),
                _ => Vec::new(),
            };
            let folds_as_one = previous_letters.iter().any(|&first| {
                letters
                    .iter()
                    .any(|&second| MULTI_CHARACTER_FOLDS.contains(&[first, second]))
            });
            if folds_as_one {
                return None;
            }
            previous_letters = letters;
        }
        Some(if nodes.len() == 1 {
            nodes.pop()?
        } else {
            Node::Concat(nodes)
        })
    }

    fn parse_quantified(&mut self) -> Option<Node> {
        let atom = self.parse_atom()?;
        let (min, max) = match self.chars.peek() {
            Some('?') => (0, 1),
            Some('*') => (0, u32::MAX),
            Some('+') => (1, u32::MAX),
            Some('{') => {
                self.chars.next();
                let min = self.parse_number()?;
                let max = match self.chars.next()? {
                    '}' => min,
                    ',' => {
                        if self.chars.peek() == Some(&'}') {
                            u32::MAX
                        } else {
                            self.parse_number()?
                        }
                    }
                    _ => return None,
                };
                if max != min && self.chars.next()? != '}' {
                    return None;
                }
                if min > max {
                    return None;
                }
                return self.quantified(atom, min, max, true);
            }
            _ => return Some(atom),
        };
        self.chars.next();
        self.quantified(atom, min, max, false)
    }

    fn quantified(&mut self, atom: Node, min: u32, max: u32, interval: bool) -> Option<Node> {
        // The lazy forms (`??`, `*?`, `+?`) are not supported. The possessive
        // ones (`?+`, `*+`, `++`) are, on a class: the run is taken whole and
        // never gives a character back, as the engine takes it. After an
        // interval the engine reads a `+` as a second quantifier (a repeat of
        // the interval), not as the possessive mark: declined.
        let possessive = match self.chars.peek() {
            Some('?') => return None,
            Some('+') if interval => return None,
            Some('+') => {
                self.chars.next();
                true
            }
            _ => false,
        };
        Some(match atom {
            Node::Class(class) => Node::ClassRepeat {
                class,
                min,
                max,
                possessive,
            },
            Node::NotAhead(_) | Node::End | Node::GroupsAhead { .. } | Node::RunOfOne { .. } => {
                return None
            }
            // A possessive group would need the engine's rule for the choices
            // inside it (an atomic group): declined.
            _ if possessive => return None,
            // A group that repeats would need the engine's rule for an empty
            // iteration and an unbounded choice stack: only the optional form
            // is accepted, longer runs stay with `tokenizers`.
            _ if max > 1 => return None,
            node => Node::Repeat {
                node: Box::new(node),
                min,
                max,
                first: First::default(),
            },
        })
    }

    fn parse_number(&mut self) -> Option<u32> {
        let mut digits = String::new();
        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                digits.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        digits.parse().ok()
    }

    fn parse_atom(&mut self) -> Option<Node> {
        let c = self.chars.next()?;
        // Under `(?i:...)` only ASCII letters and plain ASCII literals are
        // folded the way the engine folds them, letter by letter, and a class
        // whose members are ASCII letters the same way (`folded_letters`); an
        // escape, a non-ASCII literal or any other class there would match
        // differently, so it is declined (so are lookaheads and the letter
        // runs the engine folds as one character; see `parse_group` and
        // `parse_concat`).
        if self.case_insensitive && (c == '\\' || !c.is_ascii()) {
            return None;
        }
        match c {
            '(' => self.parse_group(),
            '[' => {
                let class = self.parse_class_body()?;
                let class = if self.case_insensitive {
                    folded_letters(class)?
                } else {
                    class
                };
                Some(Node::Class(self.add_class(class)))
            }
            '\\' => {
                let class = self.parse_escape_as_class()?;
                Some(Node::Class(self.add_class(class)))
            }
            '$' => Some(Node::End),
            '.' | '^' | '*' | '+' | '?' | '{' | '}' | ')' | '|' | ']' => None,
            literal => Some(Node::Class(self.literal_class(literal))),
        }
    }

    fn parse_group(&mut self) -> Option<Node> {
        let first = self.chars.next()?;
        if first != '?' {
            return self.parse_repeated_character(first);
        }
        match self.chars.next()? {
            ':' => {
                let inner = self.parse_alternation()?;
                if self.chars.next()? != ')' {
                    return None;
                }
                Some(self.single_characters(inner))
            }
            'i' => {
                if self.chars.next()? != ':' {
                    return None;
                }
                let was = std::mem::replace(&mut self.case_insensitive, true);
                let inner = self.parse_alternation();
                self.case_insensitive = was;
                let inner = inner?;
                (self.chars.next()? == ')').then_some(inner)
            }
            '!' => {
                // The option would reach into the lookahead's class, which
                // this matcher does not fold: declined under `(?i:)`.
                if self.case_insensitive {
                    return None;
                }
                let class = match self.chars.next()? {
                    '\\' => self.parse_escape_as_class()?,
                    '[' => self.parse_class_body()?,
                    _ => return None,
                };
                if self.chars.next()? != ')' {
                    return None;
                }
                let class = self.add_class(class);
                Some(Node::NotAhead(class))
            }
            '=' => self.parse_groups_ahead(),
            _ => None,
        }
    }

    /// `(?=(?:\d{n})*\b)`, the digit-grouping lookahead: a `Split` that cuts
    /// a digit run into groups of `n` from its end puts it after `\d{1,n}`,
    /// so the first group takes the remainder. Read as one op over the digit
    /// run ahead ([`Node::GroupsAhead`]); any other positive lookahead is
    /// outside the subset.
    fn parse_groups_ahead(&mut self) -> Option<Node> {
        if self.case_insensitive {
            return None;
        }
        for expected in ['(', '?', ':', '\\', 'd', '{'] {
            if self.chars.next()? != expected {
                return None;
            }
        }
        let size = self.parse_number().filter(|&size| size > 0)?;
        for expected in ['}', ')', '*', '\\', 'b', ')'] {
            if self.chars.next()? != expected {
                return None;
            }
        }
        // One pair of classes for every lookahead of the pattern (see
        // `Parser::groups_ahead`).
        let (class, word) = match self.groups_ahead {
            Some(pair) => pair,
            None => {
                let class = self.add_class(Class {
                    categories: CAT_DIGIT,
                    ..Class::default()
                });
                let word = self.add_class(Class {
                    categories: CAT_WORD,
                    ..Class::default()
                });
                self.groups_ahead = Some((class, word));
                (class, word)
            }
        };
        Some(Node::GroupsAhead { class, size, word })
    }

    /// A capturing group read back at once, `([class])\1*` or `([class])\1+`
    /// (a run of one repeated character, as a parse family's punctuation
    /// stage writes it): the group holds one class, the backreference follows
    /// it with its quantifier, and nothing else may follow in the alternative
    /// (`Pattern::parse` checks the placement). Every other capture or
    /// backreference, a second capturing group (`\1` names the pattern's
    /// first group wherever it stands) and the shape under `(?i:)` decline.
    fn parse_repeated_character(&mut self, first: char) -> Option<Node> {
        if self.case_insensitive {
            return None;
        }
        // A second group's `\1` reads the first one, unset in another
        // alternative, where the engine's `\1*` then matches nothing while a
        // run of the second group would take characters: one group per
        // pattern.
        if std::mem::replace(&mut self.captured, true) {
            return None;
        }
        let class = match first {
            '\\' => self.parse_escape_as_class()?,
            '[' => self.parse_class_body()?,
            _ => return None,
        };
        if self.chars.next()? != ')' || self.chars.next()? != '\\' || self.chars.next()? != '1' {
            return None;
        }
        let at_least = match self.chars.next()? {
            '*' => 1,
            '+' => 2,
            _ => return None,
        };
        let class = self.add_class(class);
        Some(Node::RunOfOne { class, at_least })
    }

    /// A group whose alternatives are each one character of a class
    /// (`(?:\p{L}|\p{M}|\u200C|\u200D)`) is the class of their union: the
    /// engine matches one character of whichever alternative, as it matches
    /// one character of the class, so the group then repeats as a class does
    /// and a letter run the hub files write this way stays native. Any other
    /// group stays as parsed, as does one under `(?i:)`, a negated alternative
    /// or an intersection (their union is not the union of the ranges).
    fn single_characters(&mut self, inner: Node) -> Node {
        let Node::Alt { alternatives, .. } = &inner else {
            return inner;
        };
        // The alternatives' classes are the last ones added, in order; the
        // union takes their place.
        let Some(first) = self.classes.len().checked_sub(alternatives.len()) else {
            return inner;
        };
        if self.case_insensitive {
            return inner;
        }
        for (i, alternative) in alternatives.iter().enumerate() {
            match alternative {
                Node::Class(index)
                    if *index == first + i
                        && !self.classes[*index].negated
                        && self.classes[*index].without == 0 => {}
                _ => return inner,
            }
        }
        let mut class = Class::default();
        for member in &self.classes[first..] {
            class.categories |= member.categories;
            class.ranges.extend_from_slice(&member.ranges);
        }
        self.classes.truncate(first);
        Node::Class(self.add_class(class))
    }

    /// A literal character as a class: itself, or under `(?i:)` every
    /// character the engine folds together with it.
    fn literal_class(&mut self, c: char) -> usize {
        let mut class = Class::default();
        if self.case_insensitive && c.is_ascii_alphabetic() {
            fold_letter(&mut class, c);
        } else {
            class.ranges.push((u32::from(c), u32::from(c)));
        }
        self.add_class(class)
    }

    /// The code point of a `\uHHHH` escape (four hex digits, as the engine
    /// reads the form the hub files use for the joiners); a surrogate is no
    /// character and declines.
    fn code_point(&mut self) -> Option<char> {
        let mut code = 0;
        for _ in 0..4 {
            code = (code << 4) | self.chars.next()?.to_digit(16)?;
        }
        char::from_u32(code)
    }

    /// An escape outside a class, as a class of its own.
    fn parse_escape_as_class(&mut self) -> Option<Class> {
        let mut class = Class::default();
        match self.chars.next()? {
            'p' => class.categories |= self.parse_property()?,
            'P' => {
                class.categories |= self.parse_property()?;
                class.negated = true;
            }
            'd' => class.categories |= CAT_DIGIT,
            's' => class.categories |= CAT_SPACE,
            'S' => {
                class.categories |= CAT_SPACE;
                class.negated = true;
            }
            'u' => {
                let c = self.code_point()?;
                class.ranges.push((u32::from(c), u32::from(c)));
            }
            other => {
                let c = escaped_literal(other)?;
                class.ranges.push((u32::from(c), u32::from(c)));
            }
        }
        Some(class)
    }

    /// `{L}`, `{Lu}`, ... after `\p`.
    fn parse_property(&mut self) -> Option<u16> {
        if self.chars.next()? != '{' {
            return None;
        }
        let mut name = String::new();
        loop {
            match self.chars.next()? {
                '}' => break,
                c => name.push(c),
            }
        }
        Some(match name.as_str() {
            "L" => CAT_LETTER,
            "M" => CAT_MARK,
            "N" => CAT_NUMBER,
            "P" => CAT_PUNCTUATION,
            "S" => CAT_SYMBOL,
            "Lu" => CAT_UPPERCASE,
            "Ll" => CAT_LOWERCASE,
            "Lt" => CAT_TITLECASE,
            "Lm" => CAT_MODIFIER_LETTER,
            "Lo" => CAT_OTHER_LETTER,
            "Han" => CAT_HAN,
            _ => return None,
        })
    }

    /// `[^\p{X}...]` after `&&` inside a class: the properties it takes out
    /// of the class. Nothing else may stand on either side of the `&&`.
    fn parse_subtracted_properties(&mut self) -> Option<u16> {
        if self.chars.next()? != '[' || self.chars.next()? != '^' {
            return None;
        }
        let mut categories = 0;
        loop {
            match self.chars.next()? {
                ']' if categories != 0 => return Some(categories),
                '\\' if self.chars.next()? == 'p' => categories |= self.parse_property()?,
                _ => return None,
            }
        }
    }

    /// The body of `[...]`, after the opening bracket.
    fn parse_class_body(&mut self) -> Option<Class> {
        let mut class = Class::default();
        if self.chars.peek() == Some(&'^') {
            self.chars.next();
            class.negated = true;
        }
        let mut first = true;
        loop {
            let c = self.chars.next()?;
            let item = match c {
                ']' if !first => break,
                '\\' => match self.chars.next()? {
                    'p' => {
                        class.categories |= self.parse_property()?;
                        first = false;
                        continue;
                    }
                    'd' => {
                        class.categories |= CAT_DIGIT;
                        first = false;
                        continue;
                    }
                    's' => {
                        class.categories |= CAT_SPACE;
                        first = false;
                        continue;
                    }
                    'P' | 'S' | 'D' | 'w' | 'W' | 'h' | 'H' => return None,
                    'u' => self.code_point()?,
                    other => escaped_literal(other)?,
                },
                // An intersection with a negated property class closes the
                // class: `[...&&[^\p{Han}]]`. Nested classes, other
                // intersections and POSIX brackets are not supported.
                '&' if self.chars.peek() == Some(&'&') => {
                    // Nothing before the intersection would be the empty
                    // set: declined, as every other shape of it is.
                    if first {
                        return None;
                    }
                    self.chars.next();
                    class.without = self.parse_subtracted_properties()?;
                    if self.chars.next()? != ']' {
                        return None;
                    }
                    break;
                }
                '[' => return None,
                literal => literal,
            };
            first = false;
            // A range `a-b`, unless the dash ends the class.
            if self.chars.peek() == Some(&'-') {
                self.chars.next();
                match self.chars.peek() {
                    Some(&']') => {
                        class.ranges.push((u32::from(item), u32::from(item)));
                        class.ranges.push((u32::from('-'), u32::from('-')));
                        continue;
                    }
                    Some(&'\\') => {
                        self.chars.next();
                        let end = match self.chars.next()? {
                            'p' | 's' | 'P' | 'S' => return None,
                            'u' => self.code_point()?,
                            other => escaped_literal(other)?,
                        };
                        if end < item {
                            return None;
                        }
                        class.ranges.push((u32::from(item), u32::from(end)));
                    }
                    Some(_) => {
                        let end = self.chars.next()?;
                        if end == '[' || end < item {
                            return None;
                        }
                        class.ranges.push((u32::from(item), u32::from(end)));
                    }
                    None => return None,
                }
            } else {
                class.ranges.push((u32::from(item), u32::from(item)));
            }
        }
        Some(class)
    }
}

/// The character an escape stands for, when it is a plain literal escape.
fn escaped_literal(c: char) -> Option<char> {
    Some(match c {
        'r' => '\r',
        'n' => '\n',
        't' => '\t',
        'f' => '\x0C',
        'v' => '\x0B',
        'e' => '\x1B',
        'a' => '\x07',
        // Any escaped non-alphanumeric character is itself.
        c if !c.is_alphanumeric() => c,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use tiktoken_rs::CoreBPE;
    use tokenizers::utils::SysRegex;

    use super::*;

    const GPT2: &str =
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    /// The template with a digit rule of up to two digits, and with an
    /// unbounded one: the other two bounds the scanner takes.
    const TWO_DIGITS: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const ANY_DIGITS: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DIGITS: &str = r"\p{N}{1,3}";
    const CJK: &str = "[一-龥぀-ゟ゠-ヿ]+";
    const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    /// The cased template without the clitic group and with one digit at a
    /// time (the Mistral tekken template), and with the group and one digit.
    const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CASED_ONE_DIGIT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// The Kimi template: ideograph runs first, the cased letter runs
    /// without the ideographs, the rest as cl100k's.
    const KIMI: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// The clitic group as a letter class with the apostrophe outside it,
    /// and possessive runs; the same with greedy runs, a digit cap and a
    /// one-space last alternative.
    const POSSESSIVE_CLITIC_CLASS: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    const CLITIC_CLASS: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s";
    /// A letter run whose class is written as an alternation of single
    /// characters, the joiners as code-point escapes.
    const JOINED_LETTERS: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// A parse family's punctuation stage: a bracket, or a capturing group
    /// of one punctuation character read back at once (a run of it).
    const PUNCTUATION_RUNS: &str = r#"[\(\)\[\]\{\}]|([!"\#\$%\&'\*\+,\-\./:;<=>\?\\\^_`\|\~])\1*"#;

    const PATTERNS: [&str; 21] = [
        GPT2,
        QWEN2,
        QWEN35,
        CL100K,
        TWO_DIGITS,
        ANY_DIGITS,
        DIGITS,
        CJK,
        DEEPSEEK,
        CASED,
        TEKKEN,
        CASED_ONE_DIGIT,
        KIMI,
        POSSESSIVE_CLITIC_CLASS,
        CLITIC_CLASS,
        JOINED_LETTERS,
        END_SPACES,
        END_LETTERS,
        END_GROUP,
        DIGIT_GROUPS,
        PUNCTUATION_RUNS,
    ];

    /// `\s+$` as deepseek-prover-v2-7b's fourth stage spells it: a whitespace
    /// run that ends the text or stands before a newline.
    const END_SPACES: &str = r"\s+$";
    /// A letter run at the end beside one anywhere: the anchored alternative
    /// is tried first and backs off to the other.
    const END_LETTERS: &str = r"[a-z]+$|[a-z]+";
    /// The anchor after a group whose alternatives nest a prefix.
    const END_GROUP: &str = r" ?(?:ab|b)$| ?[a-z]+";
    /// The digit-grouping split of the 255k command-a family, before its
    /// cased template: a digit run ends in groups of three, the first group
    /// taking the remainder, and a run that meets a word character stays
    /// whole.
    const DIGIT_GROUPS: &str = r"\d{1,3}(?=(?:\d{3})*\b)";

    /// Characters that exercise every class the patterns use: ASCII of all
    /// kinds, spaces and line breaks of several scripts, letters with and
    /// without case, combining marks, digits beyond ASCII, CJK, kana, emoji,
    /// and the clitic letters in their folded forms.
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
        '\u{0C}',
        '\u{85}',
        '\u{A0}',
        '\u{2003}',
        '\u{2028}',
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
        '@',
        '#',
        '$',
        '%',
        '&',
        '*',
        '+',
        ':',
        ';',
        '<',
        '=',
        '>',
        '?',
        '!',
        '(',
        ')',
        '€',
        '©',
        '→',
        '\u{301}',
        '\u{93E}',
        '\u{E31}',
        '\u{E48}',
        '\u{9BE}',
        '\u{C4D}',
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
        '々',
        '〇',
        '〡',
        '⺀',
        '𠀀',
        '𮯰',
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
        '\u{AD}',
        '\u{1F}',
    ];

    fn oracle_matches(pattern: &str, text: &str) -> Vec<(usize, usize)> {
        let re = SysRegex::new(pattern).expect("the engine compiles the pattern");
        re.find_iter(text).collect()
    }

    fn pseudo_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_text(state: &mut u64, len: usize) -> String {
        (0..len)
            .map(|_| POOL[(pseudo_random(state) % POOL.len() as u64) as usize])
            .collect()
    }

    #[test]
    fn every_known_pattern_parses() {
        for pattern in PATTERNS {
            assert!(Pattern::parse(pattern).is_some(), "{pattern}");
        }
    }

    /// The first-character sets of the alternatives whose first `Split` is
    /// at `pc`, in order (each `Split` names the next one).
    fn split_chain(pattern: &Pattern, mut pc: usize) -> Vec<First> {
        let mut sets = Vec::new();
        loop {
            let Op::Split { next, first } = pattern.program[pc] else {
                panic!("op {pc} is not a split: {:?}", pattern.program[pc]);
            };
            sets.push(pattern.firsts[first]);
            if next == NO_NEXT {
                return sets;
            }
            pc = next;
        }
    }

    #[test]
    fn alternatives_know_the_characters_a_match_of_theirs_can_begin_with() {
        let parsed = Pattern::parse(QWEN2).expect("parses");
        // The program begins with the root alternation's first split; the
        // alternatives in the pattern's order: the clitics, a letter run with
        // an optional lead-in, a digit, punctuation with an optional space,
        // whitespace ending in a line break, whitespace before a non-space,
        // whitespace.
        let firsts = split_chain(&parsed, 0);
        assert_eq!(firsts.len(), 7);
        let starts = |c: char| -> Vec<usize> {
            firsts
                .iter()
                .enumerate()
                .filter(|(_, first)| first.may_start(c))
                .map(|(index, _)| index)
                .collect()
        };
        assert_eq!(starts('\''), vec![0, 1, 3]);
        assert_eq!(starts('a'), vec![1]);
        assert_eq!(starts('7'), vec![2]);
        assert_eq!(starts(','), vec![1, 3]);
        assert_eq!(starts(' '), vec![1, 3, 4, 5, 6]);
        assert_eq!(starts('\n'), vec![4, 5, 6]);
        // Beyond ASCII the sets are exact only in the negative: a letter run
        // may start with a non-ASCII letter, a digit alternative built from a
        // category may hold one, and no set excludes what it cannot see.
        assert!(starts('\u{E9}').contains(&1));
        assert!(!starts('\u{E9}').contains(&0));
        // The clitic group is an alternation of its own, laid out right after
        // the root's first split; each alternative begins with the apostrophe
        // and with nothing else.
        let clitics = split_chain(&parsed, 1);
        assert_eq!(clitics.len(), 7);
        assert!(clitics
            .iter()
            .all(|first| first.may_start('\'') && !first.may_start('s')));
        assert!(matches!(parsed.program.last(), Some(Op::Match)));
    }

    #[test]
    fn unsupported_syntax_is_rejected() {
        for pattern in [
            r"a.b",
            r"^a",
            r"(a)",
            r"a*?",
            r"a+?",
            r"(?:ab)?+",
            r"(?:ab|c)++",
            r"a{1,3}+",
            r"[[:alpha:]]",
            r"[a&&b]",
            r"[\p{L}&&\p{Han}]",
            r"[\p{L}&&[^a]]",
            r"[\p{L}&&[^\p{Han}]a]",
            r"[&&[^\p{Han}]]",
            r"[&&a]",
            r"[^&&a]",
            r"[\p{L}&&[^]]",
            r"(?=a)",
            r"\b",
            r"(?=\d)",
            r"(?=(?:\d{3})*)",
            r"(?=(?:\d{3})*\b)",
            r"(?=(?:\d{2,3})*\b)",
            r"(?=(?:[0-9]{3})*\b)",
            r"(?=(?:\p{N}{3})*\b)",
            r"(?=(?:\d{3})+\b)",
            r"(?=(?:\d{0})*\b)",
            r"\d(?=(?:\d{3})*\b)?",
            r"(?i:\d(?=(?:\d{3})*\b))",
            // the lookahead at a match's start: its boundary would read the
            // character before the match, which a unit of the pool lacks
            r"(?=(?:\d{3})*\b)\d",
            r"(?=(?:\d{3})*\b) [a-z]+|[a-z]+",
            r"\d?(?=(?:\d{3})*\b)\d",
            r"\d*(?=(?:\d{3})*\b)\d",
            r"(?:\d|(?=(?:\d{3})*\b))\d",
            r"\d|(?=(?:\d{3})*\b)\d",
            r"(?:(?=(?:\d{3})*\b)\d)?\d",
            r"\p{Hiragana}",
            r"a{3,1}",
            r"[z-a]",
            r"(?:ab)+",
            r"(?: ?\p{L})*",
            r"(?:ab){2}",
            r"(?i:[a-z]|ll)",
            r"(?i:\p{Ll})",
            r"(?i:é)",
            r"\p{N}*",
            r"a?",
            r"(?!\S)",
            r"\s*|\p{L}+",
            r"(?:\p{L}+)?",
            r"a|",
            r"",
            r"(?i:'s(?![a-z]))",
            r"(?i:'s(?!\S))",
            r"(?i:ss)",
            r"(?i:'st)",
            r"(?i:fi|fl|ff)",
            r"$",
            r"^\s+",
            r"\s+$|^a",
            r"a$?",
            r"(?:a|$)",
            // a capture or a backreference outside the one shape taken
            r"(a)\1*",
            r"([ab])\1",
            r"([ab])\1?",
            r"([ab])\2*",
            r"([ab]|c)\1*",
            r"(\p{L}\p{M})\1*",
            r"\1",
            r"([ab])\1*+",
            r"(?i:([ab])\1*)",
            // the run with something after it in its alternative
            r"([ab])\1*c",
            r"x|([ab])\1*y",
            r"(?:([ab])\1*)?",
            r"(?:x([ab])\1*)y",
            // a zero-width anchor after it too: the engine backs `\1*` off so
            // that `$` holds before a `\n` of the run
            r"([\n])\1*$",
            // a second capturing group: its `\1` names the first group, unset
            // in another alternative, where the engine's `\1*` matches nothing
            // (the last row declines for that before the placement check)
            r"([a])\1*|([b])\1*",
            r"([ab])\1*|([cd])\1*",
            r"(?:x|([ab])\1*)|([cd])\1*",
            r"([ab])\1*([cd])\1*",
            r"([ab])\1*|([cd])\2*",
            r"([ab])\1*|c([ab])\1*d",
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
        }
    }

    /// Our matches of `pattern` in `text`.
    fn matches_of(pattern: &str, text: &str) -> Vec<(usize, usize)> {
        Pattern::parse(pattern)
            .expect("parses")
            .find_iter(text)
            .collect()
    }

    /// `\d` is the engine's decimal digit class: the digits of every
    /// script and none of the other numbers (a superscript, a fraction, a
    /// circled digit, a Roman numeral).
    #[test]
    fn the_digit_class_is_the_engines() {
        let text = POOL.iter().collect::<String>() + "0123456789\u{663}\u{664}\u{6F4}\u{7C0}\u{969}\u{E52}\u{1D7D8}\u{2460}\u{216B}\u{B2}\u{BD}\u{2469}";
        for pattern in [r"\d+", r"\d", r"\d{1,3}", r"[a-z]\d*", r"\d{2}|[^\d]+"] {
            assert_eq!(
                matches_of(pattern, &text),
                oracle_matches(pattern, &text),
                "{pattern}"
            );
        }
    }

    /// The digit-grouping lookahead counts the digit run ahead as the
    /// engine does: a run of a multiple of the group's size that ends the
    /// text or meets a character outside the word class holds the
    /// boundary, so the groups are cut from the end of the run, the first
    /// taking the remainder; a run that meets a word character (a letter,
    /// a mark, an underscore, a joiner, a Roman numeral) matches nowhere;
    /// after a character outside the word class the boundary holds at the
    /// position itself.
    #[test]
    fn the_digit_grouping_lookahead_counts_the_run_as_the_engine_does() {
        let runs = [
            "",
            "1",
            "12",
            "123",
            "1234",
            "12345",
            "123456",
            "1234567",
            "123456789012",
            "\u{663}\u{664}\u{665}",
            "\u{663}\u{664}\u{665}\u{666}",
            "1\u{663}4",
            "\u{969}\u{969}\u{969}",
        ];
        let prefixes = ["", "x", " ", ",", "\u{663}", "_", "\u{301}"];
        let suffixes = [
            "",
            " ",
            "a",
            "\u{E9}",
            "\u{301}",
            "_",
            ",",
            ".",
            "$",
            "\u{2460}",
            "\u{216B}",
            "\u{B2}",
            "\u{BD}",
            "\u{200D}",
            "\u{AA}",
            "\u{2B0}",
            "\u{4E2D}",
            "\u{1F600}",
            "\n",
            "\u{A0}",
            "\u{663}",
        ];
        for pattern in [
            DIGIT_GROUPS,
            r"\d{1,3}(?=(?:\d{2})*\b)",
            r"\d(?=(?:\d{4})*\b)",
            r"[,a](?=(?:\d{3})*\b)",
            r"\d{2}(?=(?:\d{3})*\b)|\d",
        ] {
            for prefix in prefixes {
                for run in runs {
                    for suffix in suffixes {
                        let text = format!("{prefix}{run}{suffix}{run}{suffix}");
                        assert_eq!(
                            matches_of(pattern, &text),
                            oracle_matches(pattern, &text),
                            "{pattern} on {text:?}"
                        );
                    }
                }
            }
        }
        for text in [
            "12,345,678.90",
            "1 000 000",
            "2026-11-12T03:00Z",
            "v1.2.3",
            "1234abc 5678 90",
            "123_456",
            "007",
            "\u{663}\u{664}\u{665}\u{666}\u{667}\u{668}\u{669} \u{661}\u{662}\u{663}",
            "1\u{2460}2",
            "x1_",
            "99\n100",
        ] {
            assert_eq!(
                matches_of(DIGIT_GROUPS, text),
                oracle_matches(DIGIT_GROUPS, text),
                "{text:?}"
            );
        }
    }

    /// A capturing group read back at once is a run of one repeated
    /// character, cut as the engine's greedy backreference cuts it; the run
    /// may end a top-level alternative anywhere in the alternation.
    #[test]
    fn a_repeated_character_run_is_cut_as_the_engine_cuts_it() {
        fn runs(pattern: &str, text: &str) -> Vec<(usize, usize)> {
            Pattern::parse(pattern)
                .expect("parses")
                .find_iter(text)
                .collect()
        }
        for (pattern, text) in [
            (PUNCTUATION_RUNS, "a...b, c!!!d (()) ... ,,,. .,"),
            (PUNCTUATION_RUNS, "..."),
            (PUNCTUATION_RUNS, ".a.a."),
            (PUNCTUATION_RUNS, "[[]]{}()"),
            (
                PUNCTUATION_RUNS,
                "---\\\\ ``` ~~ || ^^ __ // :: ;; << == >> ?? ** ++ ,, '' \"\" ## $$ %% && ..",
            ),
            (PUNCTUATION_RUNS, "a\u{0301}..\u{0301}.\u{00e9}"),
            (r"([ab])\1*", "aaabbbab"),
            (r"([ab])\1+", "aaabbbab"),
            (r"x|([ab])\1*", "xaaxbb"),
            (r"(?:x|([ab])\1*)", "xaaxbb"),
            (r"(\p{L})\1*", "aa\u{00e9}\u{00e9}bb ccc"),
            (r"(\s)\1+", "a  b \t\t c"),
        ] {
            assert_eq!(
                runs(pattern, text),
                oracle_matches(pattern, text),
                "{pattern} on {text:?}"
            );
        }
        assert_eq!(
            runs(r"([ab])\1*", "aaabbbab"),
            vec![(0, 3), (3, 6), (6, 7), (7, 8)]
        );
        assert_eq!(runs(r"([ab])\1+", "aaabbbab"), vec![(0, 3), (3, 6)]);
        let parsed = Pattern::parse(PUNCTUATION_RUNS).expect("parses");
        assert!(matches!(parsed.matcher, Matcher::Program));
        assert!(parsed
            .program
            .iter()
            .any(|op| matches!(op, Op::RunOfOne { at_least: 1, .. })));
    }

    #[test]
    fn a_single_character_alternation_is_its_class() {
        // The alternation form and the class form parse to the same cuts,
        // and both agree with the engine, over every swept code point and
        // over texts of letters, marks, joiners, digits and spaces.
        let pairs = [
            (
                r"(?:\p{L}|\p{M}|\u200C|\u200D)+",
                r"[\p{L}\p{M}\u200C\u200D]+",
            ),
            (r" ?(?:a|b|\u00E9)+|\p{N}", r" ?[ab\u00E9]+|\p{N}"),
            (r"(?:\p{N}|[,.])+\p{L}?", r"[\p{N},.]+\p{L}?"),
            // A range inside another: the union's ranges merge in
            // `Class::finish`, or `d`..`z` would be read as absent.
            (r"(?:[a-z]|c)+", r"[a-zc]+"),
        ];
        let mut texts: Vec<String> = code_points_swept()
            .filter_map(char::from_u32)
            .map(String::from)
            .collect();
        let pool: &[char] = &[
            'a', 'b', '\u{E9}', '\u{E01}', '\u{301}', '\u{E31}', '\u{200C}', '\u{200D}', '1',
            '\u{663}', ' ', ',', '.', '\n',
        ];
        let mut state = 0x6A09_E667_F3BC_C908u64;
        for round in 0..2000 {
            texts.push(
                (0..1 + round % 16)
                    .map(|_| pool[(pseudo_random(&mut state) % pool.len() as u64) as usize])
                    .collect(),
            );
        }
        for (alternation, class) in pairs {
            let ours = Pattern::parse(alternation).expect("the alternation parses");
            let theirs = Pattern::parse(class).expect("the class parses");
            let engine = SysRegex::new(alternation).expect("the engine compiles the alternation");
            for text in &texts {
                let cuts: Vec<(usize, usize)> = ours.find_iter(text).collect();
                assert_eq!(
                    cuts,
                    theirs.find_iter(text).collect::<Vec<_>>(),
                    "{alternation:?} on {text:?}"
                );
                assert_eq!(
                    cuts,
                    engine.find_iter(text).collect::<Vec<_>>(),
                    "the engine on {alternation:?} {text:?}"
                );
            }
        }
        // A `\u` escape is a code point in an atom, in a class and as a
        // range end; the other shapes stay declined.
        let cuts = |pattern: &str, text: &str| -> Vec<(usize, usize)> {
            Pattern::parse(pattern)
                .expect("parses")
                .find_iter(text)
                .collect()
        };
        assert_eq!(cuts(r"\u0041+", "AAb"), oracle_matches(r"\u0041+", "AAb"));
        assert_eq!(cuts(r"\u0041+", "AAb"), vec![(0, 2)]);
        assert_eq!(cuts(r"[\u0041-\u0043]+", "ABCD"), vec![(0, 3)]);
        for pattern in [
            r"\u00",
            r"\u00G1",
            r"\uD800",
            r"(?:ab|c)+",
            r"(?:[^a]|b)+",
            r"(?:\p{L}|\p{M}+)+",
            r"(?i:(?:a|b)+)",
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
        }
    }

    /// The code points the two cross-engine sweeps below check: every one of
    /// the four planes that hold assigned scripts (0 to 3) and plane 14's
    /// first block (tags and variation selectors), then a stride over the
    /// unassigned and private planes. A script a Unicode version adds,
    /// however small, lies inside the part swept whole.
    fn code_points_swept() -> impl Iterator<Item = u32> + Clone {
        (0u32..0x4_0000)
            .chain(0xE_0000..0xE_1000)
            .chain((0x4_0000..=0x10_FFFF).step_by(97))
    }

    /// The `Han` property and a class with it taken out agree with the
    /// engine character by character over [`code_points_swept`], for the
    /// property alone, negated, taken out of the Kimi template's two letter
    /// classes, and taken out under an outer negation.
    #[test]
    fn han_and_the_classes_it_is_taken_out_of_agree_with_the_engine() {
        let patterns = [
            r"\p{Han}",
            r"[^\p{Han}]",
            r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[^\p{L}\p{N}&&[^\p{Han}]]",
            r"[a-z\p{M}&&[^\p{Han}\p{Lo}]]",
        ];
        let codes = code_points_swept();
        let mut buf = String::new();
        for pattern in patterns {
            let parsed = Pattern::parse(pattern).expect("parses");
            let engine = SysRegex::new(pattern).expect("the engine compiles the pattern");
            for code in codes.clone() {
                let Some(c) = char::from_u32(code) else {
                    continue;
                };
                buf.clear();
                buf.push(c);
                let ours = parsed.find_iter(&buf).next() == Some((0, buf.len()));
                let theirs = engine.find_iter(&buf).next() == Some((0, buf.len()));
                assert_eq!(ours, theirs, "{pattern} at U+{code:04X}");
            }
        }
    }

    /// The classes of the tiktoken templates the direct path reads, checked
    /// over [`code_points_swept`] against the file's own encoder (a `CoreBPE`
    /// whose pattern is the class alone encodes a code point it matches and
    /// drops one it does not): the two engines' Unicode tables agree today,
    /// and a version apart between them, a new script included, would show
    /// here.
    #[test]
    fn the_tiktoken_template_classes_agree_with_the_files_own_encoder() {
        let ranks: FxHashMap<Vec<u8>, u32> = (0..=255u8).map(|b| (vec![b], u32::from(b))).collect();
        let classes = [
            r"[\p{Han}]",
            r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[^\r\n\p{L}\p{N}]",
            r"\p{L}",
            r"\p{N}",
            r"[^\s\p{L}\p{N}]",
            r"\s",
        ];
        let codes = code_points_swept();
        let mut buf = String::new();
        for class in classes {
            let parsed = Pattern::parse(class).expect("parses");
            let own = CoreBPE::new(ranks.clone(), FxHashMap::default(), class)
                .expect("the file's own encoder compiles the class");
            for code in codes.clone() {
                let Some(c) = char::from_u32(code) else {
                    continue;
                };
                buf.clear();
                buf.push(c);
                let ours = parsed.find_iter(&buf).next() == Some((0, buf.len()));
                let theirs = !own.encode_ordinary(&buf).is_empty();
                assert_eq!(ours, theirs, "{class} at U+{code:04X}");
            }
        }
    }

    #[test]
    fn matches_agree_with_the_engine_on_random_text() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for pattern in PATTERNS {
            let parsed = Pattern::parse(pattern).expect("parses");
            for round in 0..1500 {
                let text = random_text(&mut state, 1 + round % 24);
                let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                assert_eq!(
                    ours,
                    oracle_matches(pattern, &text),
                    "pattern {pattern:?} text {text:?}"
                );
                if !matches!(parsed.matcher, Matcher::Program) {
                    let program: Vec<(usize, usize)> =
                        parsed.program_only().find_iter(&text).collect();
                    assert_eq!(ours, program, "pattern {pattern:?} text {text:?}");
                }
            }
        }
    }

    #[test]
    fn the_gpt4_family_template_takes_the_scanner_and_no_other_shape_does() {
        for (pattern, digit_cap) in [
            (QWEN2, 1),
            (QWEN35, 1),
            (CL100K, 3),
            (TWO_DIGITS, 2),
            (ANY_DIGITS, u32::MAX),
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            let Matcher::Template(scanner) = &parsed.matcher else {
                panic!("the template takes the scanner: {pattern}");
            };
            assert_eq!(scanner.digit_cap, digit_cap, "{pattern}");
            assert_eq!(scanner.clitics.len(), 7, "{pattern}");
            assert!(!scanner.clitic_beyond_ascii, "{pattern}");
        }
        // The last two alternatives the other way round, a digit rule of two
        // or more, a lead-in run: the same classes in another shape.
        const SWAPPED: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+|\s+(?!\S)";
        const TWO_OR_MORE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{2,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const LEAD_IN_RUN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]*\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        for pattern in [
            GPT2,
            DIGITS,
            CJK,
            DEEPSEEK,
            KIMI,
            SWAPPED,
            TWO_OR_MORE,
            LEAD_IN_RUN,
        ] {
            assert!(
                matches!(
                    Pattern::parse(pattern).expect("parses").matcher,
                    Matcher::Program
                ),
                "{pattern}"
            );
        }
    }

    /// The run sites name the word scans their roles carry on the real
    /// templates: the template's letter run is the letters, the cased
    /// template's upper and lower runs are the upper-case and the lower-case
    /// letters, and the white-space runs of both are the white space; the
    /// digit, punctuation and line-break roles run byte by byte. A class
    /// change that dropped a scan would keep the ids and lose the gain; it
    /// fails here instead.
    #[test]
    fn the_run_sites_name_the_scans_their_roles_carry() {
        for pattern in [CL100K, QWEN2, QWEN35, TWO_DIGITS, ANY_DIGITS] {
            let parsed = Pattern::parse(pattern).expect("parses");
            let Matcher::Template(scanner) = &parsed.matcher else {
                panic!("the template takes its scanner: {pattern}");
            };
            let roles = &scanner.roles;
            assert_eq!(
                roles.scan_of(Role::Word.bit()),
                WordScan::LETTERS,
                "{pattern}"
            );
            for role in [Role::TrailSpace, Role::Space] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::SPACES,
                    "{pattern} {role:?}"
                );
            }
            for role in [Role::Digit, Role::Punct, Role::PunctBreak, Role::Break] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::NONE,
                    "{pattern} {role:?}"
                );
            }
        }
        for pattern in [CASED, TEKKEN, CASED_ONE_DIGIT] {
            let parsed = Pattern::parse(pattern).expect("parses");
            let Matcher::Cased(scanner) = &parsed.matcher else {
                panic!("the cased template takes its scanner: {pattern}");
            };
            let roles = scanner.roles();
            for role in [cased::Role::Upper1, cased::Role::Upper2] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::UPPER,
                    "{pattern} {role:?}"
                );
            }
            for role in [cased::Role::Lower1, cased::Role::Lower2] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::LOWER,
                    "{pattern} {role:?}"
                );
            }
            for role in [cased::Role::TrailSpace, cased::Role::Space] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::SPACES,
                    "{pattern} {role:?}"
                );
            }
            for role in [
                cased::Role::Digit,
                cased::Role::Punct,
                cased::Role::PunctBreak,
                cased::Role::Break,
            ] {
                assert_eq!(
                    roles.scan_of(role.bit()),
                    WordScan::NONE,
                    "{pattern} {role:?}"
                );
            }
        }
    }

    /// The cased-letter template with the clitic group, without it and with
    /// one digit at a time, and no other shape: the group in one letter
    /// alternative only, the letter alternatives the other way round, a
    /// required lead-in, a non-optional group.
    #[test]
    fn the_cased_template_takes_its_scanner_and_no_other_shape_does() {
        for (pattern, digit_cap, clitics) in
            [(CASED, 3, 7), (TEKKEN, 1, 0), (CASED_ONE_DIGIT, 1, 7)]
        {
            let parsed = Pattern::parse(pattern).expect("parses");
            let Matcher::Cased(scanner) = &parsed.matcher else {
                panic!("the cased template takes its scanner: {pattern}");
            };
            assert_eq!(scanner.digit_cap(), digit_cap, "{pattern}");
            assert_eq!(scanner.clitics(), clitics, "{pattern}");
        }
        const ONE_GROUP: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const UPPER_FIRST: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const LEAD_IN_REQUIRED: &str = r"[^\r\n\p{L}\p{N}][\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}][\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const GROUP_REQUIRED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        for pattern in [
            ONE_GROUP,
            UPPER_FIRST,
            LEAD_IN_REQUIRED,
            GROUP_REQUIRED,
            CL100K,
            KIMI,
        ] {
            assert!(
                !matches!(
                    Pattern::parse(pattern).expect("parses").matcher,
                    Matcher::Cased(_)
                ),
                "{pattern}"
            );
        }
    }

    #[test]
    fn the_scanner_agrees_with_the_program_and_the_engine_on_shaped_text() {
        // Text shaped like JSON, code and prose: runs of spaces and line
        // breaks before letters, digits, quotes and brackets, where the
        // whitespace alternatives give characters back, clitics in every
        // case, lead-ins of every kind, and the characters beyond ASCII
        // that the classes hold or exclude.
        const PARTS: &[&str] = &[
            " ", "  ", "   ", "\n", "\n\n", "\r\n", "\t", "\u{A0}", "\u{2028}", "\u{3000}", "\"",
            "'", "'s", "'S", "'ll", "'LL", "'ve", "'re", "'d", "'m", "'t", "'x", "'ſ", ":", ",",
            ".", "{", "}", "[", "]", "(", ")", "-", "_", "a", "Word", "ſ", "é", "ß", "1", "22",
            "333", "4444", "٣", "²", "中", "ก", "\u{301}", "\u{93E}", "😀", "\u{200D}", "\u{FE0F}",
            "x", "ABC", "AbC", "aBC", "ǅx", "xǅ", "ʰa", "Aʰ", "中A", "A中", "\u{301}A", "/", "//",
            ".../", "\n/", "I'M", "WE'LL", "'S",
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for pattern in [
            QWEN2,
            QWEN35,
            CL100K,
            TWO_DIGITS,
            ANY_DIGITS,
            CASED,
            TEKKEN,
            CASED_ONE_DIGIT,
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            assert!(!matches!(parsed.matcher, Matcher::Program), "{pattern}");
            let program = parsed.program_only();
            for round in 0..2000 {
                let text: String = (0..1 + round % 40)
                    .map(|_| PARTS[(pseudo_random(&mut state) % PARTS.len() as u64) as usize])
                    .collect();
                let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                assert_eq!(
                    ours,
                    program.find_iter(&text).collect::<Vec<_>>(),
                    "the program; pattern {pattern:?} text {text:?}"
                );
                assert_eq!(
                    ours,
                    oracle_matches(pattern, &text),
                    "the engine; pattern {pattern:?} text {text:?}"
                );
            }
        }
    }

    #[test]
    fn matches_agree_with_the_engine_on_clitics_and_whitespace_runs() {
        let texts = [
            "I'm sure they'RE here, it'S fine, we'll see, you'd know, I've, don't, isn'T",
            "I'ſ and WE'LL and it'ſ and 'ſ'ſ",
            "a  b   c\n\n\nd \n e\r\n\r\nf  \n  g",
            "   leading and trailing   ",
            "tabs\t\tand\u{A0}nbsp\u{3000}ideographic\u{2028}sep",
            "123456789 ١٢٣٤ ๑๒๓ 1,234.56",
            "日本語のテキストと한국어와中文混在",
            "e\u{301}cole ka\u{93E} th\u{E31}\u{E48} b\u{9BE}",
            "snake_case-kebab.dot/slash\\back `tick` ~tilde~ [br] {cu} (pa) <an>",
            "emoji 😀👍🏽 and ✨ and \u{FE0F} and \u{200D}",
            "",
            "\n",
            " ",
            "'",
            "''s",
        ];
        for pattern in PATTERNS {
            let parsed = Pattern::parse(pattern).expect("parses");
            for text in texts {
                let ours: Vec<(usize, usize)> = parsed.find_iter(text).collect();
                assert_eq!(
                    ours,
                    oracle_matches(pattern, text),
                    "pattern {pattern:?} text {text:?}"
                );
            }
        }
    }

    #[test]
    fn long_runs_do_not_recurse_per_character() {
        let parsed = Pattern::parse(QWEN35).expect("parses");
        let spaces = " ".repeat(2_000_000);
        let ours: Vec<(usize, usize)> = parsed.find_iter(&spaces).collect();
        assert_eq!(ours, vec![(0, 2_000_000)]);
        let digits = "7".repeat(300_000);
        assert_eq!(parsed.find_iter(&digits).count(), 300_000);
        let mixed = "\n".repeat(100_000) + &" ".repeat(100_000) + "x";
        let ours: Vec<(usize, usize)> = parsed.find_iter(&mixed).collect();
        assert_eq!(ours, oracle_matches(QWEN35, &mixed));
    }

    #[test]
    fn patterns_that_can_match_empty_are_declined() {
        // The engine yields empty matches and `Split` cuts the text at them:
        // `\p{N}*` over "ab1" matches at 0, at 1 and at 2..3, so the pieces
        // are "a", "b" and "1". A matcher that skipped the empty matches
        // would give "ab" and "1", and the ids could differ.
        assert_eq!(
            oracle_matches(r"\p{N}*", "ab1"),
            vec![(0, 0), (1, 1), (2, 3)]
        );
        for (pattern, text) in [
            (r"\p{N}*", "ab1"),
            (r"a?", "ab1"),
            (r"(?i:'s)?", "x's y"),
            (r"\s*|\p{L}+", "ab 1"),
            (r"\p{N}{0,3}", "ab12"),
            (r"(?!\S)", "a b"),
            (r"a*b*", "ab1"),
            (r"(?:\p{L}+)?", "ab 1"),
            (r"", "ab"),
            (r"a|", "ab"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} can match empty and must be declined"
            );
        }
    }

    #[test]
    fn the_choice_stack_holds_a_pattern_at_its_limit_and_declines_one_past_it() {
        // Every optional class leaves one choice on the stack while the rest
        // of the pattern runs: a pattern of MAX_CHOICES of them fills the
        // stack exactly at the match, one more is declined at parse time.
        // (A text of a's with no b would back off through every one of the
        // 2^MAX_CHOICES ways, in this matcher and in the engine alike.)
        let at_limit = "a?".repeat(MAX_CHOICES) + "b";
        let parsed = Pattern::parse(&at_limit).expect("parses at the limit");
        for text in [
            "a".repeat(MAX_CHOICES) + "b",
            "a".repeat(MAX_CHOICES - 1) + "b",
            "ab b".into(),
        ] {
            assert_eq!(
                parsed.find_iter(&text).collect::<Vec<_>>(),
                oracle_matches(&at_limit, &text),
                "{text:?}"
            );
        }
        assert!(Pattern::parse(&("a?".repeat(MAX_CHOICES + 1) + "b")).is_none());
        // The bound counts the deepest path, not the whole tree: an
        // alternation of MAX_CHOICES such runs needs two choices.
        let wide = (0..MAX_CHOICES)
            .map(|_| "a?b")
            .collect::<Vec<_>>()
            .join("|");
        assert!(Pattern::parse(&wide).is_some());
    }

    #[test]
    fn optional_groups_whose_body_can_match_empty_agree_with_the_engine() {
        // The body of an optional group may match empty; the group then
        // matches like the engine's, which takes the empty body and goes on.
        for (pattern, text) in [
            (r"(?:\p{N}*|x)?x", "xx 1x x"),
            (r"(?:\p{N}*|x)?y", "xy 2y y"),
            (r"(?:\p{N}*){1}x", "x 1x 22x"),
            (r"(?:\p{N}*)?x", "x 1x 22x"),
            (r"(?:x|\p{N}*)?x", "xx 1x x"),
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            assert_eq!(
                parsed.find_iter(text).collect::<Vec<_>>(),
                oracle_matches(pattern, text),
                "pattern {pattern:?} text {text:?}"
            );
        }
    }

    #[test]
    fn possessive_runs_never_give_back() {
        // `a++a` cannot match "aaa": the run takes every a and gives none
        // back, so the alternative fails where the greedy `a+a` matches.
        assert_eq!(oracle_matches(r"a++a", "aaa"), Vec::<(usize, usize)>::new());
        assert_eq!(oracle_matches(r"a+a", "aaa"), vec![(0, 3)]);
        for (pattern, text) in [
            (r"a++a|a", "aaa b"),
            (r"a*+a|a", "aaa"),
            (r"a?+a|x", "aa x a"),
            (r"[^x]?+x|x", "ax xx x"),
            (r"\p{L}++\p{N}|\p{L}", "ab1 ab a"),
            (r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s", " !!\n ?\r\n x"),
            (
                POSSESSIVE_CLITIC_CLASS,
                "it's 'S don't 'LL x'veY  a1 2,3 \n\n  x \r\n 'ſ",
            ),
            (
                CLITIC_CLASS,
                "it's 'S don't 'LL x'veY  a1234 2,3 \n\n  x \r\n 'ſ",
            ),
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            assert_eq!(
                parsed.find_iter(text).collect::<Vec<_>>(),
                oracle_matches(pattern, text),
                "pattern {pattern:?} text {text:?}"
            );
        }
        // A possessive run holds no choice: the stack a pattern needs is
        // counted without it.
        let mut parser = Parser {
            chars: "a++b+".chars().peekable(),
            classes: Vec::new(),
            case_insensitive: false,
            groups_ahead: None,
            captured: false,
        };
        let root = parser.parse_alternation().expect("parses");
        assert_eq!(choices(&root), 1);
        // The engine's possessive mark on a group, and a `+` after an
        // interval (a second quantifier to the engine), stay outside the
        // subset.
        for pattern in [
            r"(?:ab)++",
            r"(?:ab|c)?+",
            r"(?:'s)*+",
            r"a{2}+",
            r"a{1,3}+",
            r"a{2,}+",
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
        }
    }

    #[test]
    fn case_insensitive_letter_classes_fold_each_letter() {
        // The engine folds a class's members like literals: `[sdmt]` under
        // `(?i:)` takes S, D, M, T and the long s, and nothing of ß.
        for pattern in [
            r"'(?i:[sdmt]|ll|ve|re)",
            r"(?i:[sdmt])",
            r"(?i:[kK]|x)",
            r"(?i:[tT]|a[sd])",
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            for text in ["'S 'd 'LL 'ſ 'ß 'K '\u{212A}", "x K \u{212A} k t as AD aß"] {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "pattern {pattern:?} text {text:?}"
                );
            }
            let mut state = 0x5EED_0000_1234_5678u64;
            for round in 0..500 {
                let text = random_text(&mut state, 1 + round % 24);
                assert_eq!(
                    parsed.find_iter(&text).collect::<Vec<_>>(),
                    oracle_matches(pattern, &text),
                    "pattern {pattern:?} text {text:?}"
                );
            }
        }
        // A class with anything but ASCII letters stays declined, and so does
        // a letter class beside a literal letter the engine folds with one of
        // its letters as one character.
        for pattern in [
            r"(?i:[a-z])",
            r"(?i:[s\p{L}])",
            r"(?i:[^s])",
            r"(?i:[sé])",
            r"(?i:[s1])",
            r"(?i:[s\s])",
            r"(?i:[st]s)",
            r"(?i:f[fil])",
            r"(?i:[sx][st])",
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
        }
    }

    #[test]
    fn case_insensitive_groups_decline_classes_and_non_ascii_literals() {
        // Under `(?i:)` the engine folds classes, properties and non-ASCII
        // literals as well; this matcher folds ASCII letters only, so a
        // group with anything else is declined rather than matched
        // differently.
        assert_eq!(
            oracle_matches(r"(?i:[a-z]+)", "ABC \u{17F} \u{212A}"),
            vec![(0, 3), (4, 6), (7, 10)]
        );
        assert_eq!(oracle_matches(r"(?i:é)", "É"), vec![(0, 2)]);
        for (pattern, text) in [
            (r"(?i:[a-z]+)", "ABC \u{17F} \u{212A}"),
            (r"(?i:[s\p{Ll}]|ll)", "'S 'LL"),
            (r"(?i:\p{Ll})", "A"),
            (r"(?i:é)", "É"),
            (r"(?i:'[a-z])", "'S"),
            (r"(?i:ǆ)", "Ǆ ǅ"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} must be declined"
            );
        }
    }

    #[test]
    fn case_insensitive_groups_decline_lookaheads_and_multi_character_folds() {
        // The option reaches into a lookahead: `(?![a-z])` under `(?i:)`
        // also rejects A-Z, ſ and K.
        assert_eq!(
            oracle_matches(r"(?i:'s(?![a-z]))", "'sA 'sb 's"),
            vec![(8, 10)]
        );
        // A run of ASCII letters also matches the one character whose full
        // case folding is that run: ß and ẞ for "ss", ﬁ for "fi", ﬆ for "st".
        assert_eq!(oracle_matches(r"(?i:ss)", "ß"), vec![(0, 2)]);
        assert_eq!(oracle_matches(r"(?i:'st)", "'ﬆ"), vec![(0, 4)]);
        for (pattern, text) in [
            (r"(?i:'s(?![a-z]))", "'sA 'sb 's"),
            (r"(?i:'s(?!\p{Ll}))", "'sA"),
            (r"(?i:'s(?!\S))", "'sA 's"),
            (r"(?i:ss)", "ß ẞ ss"),
            (r"(?i:'st|'ss)", "'ﬆ 'ß"),
            (r"(?i:fi|fl|ff)", "ﬁ ﬂ ﬀ"),
            (r"(?i:ffi)", "ﬃ"),
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d|'st)", "'ﬅ"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} must be declined"
            );
        }
        // The clitic groups in use fold letter by letter and stay accepted.
        for pattern in [QWEN2, QWEN35, CL100K, CASED] {
            assert!(Pattern::parse(pattern).is_some(), "{pattern}");
        }
    }

    /// A long digit run costs one scan whatever follows it (the memo of the
    /// run's end moves with the match loop through back-offs and starts), and
    /// cuts as the engine cuts it: a run that meets a letter (the boundary
    /// fails after the run, every start backs off three times), a run that
    /// ends the text, one that meets a space, runs of a multiple of three,
    /// and digits of another script (two bytes each, the memo counts
    /// characters).
    #[test]
    fn a_long_digit_run_is_measured_once_whatever_follows() {
        let run = "1".repeat(4096);
        let arabic = "\u{663}".repeat(301);
        for pattern in [
            DIGIT_GROUPS,
            r"\d{2}(?=(?:\d{3})*\b)|\d|[a-z]+",
            // two lookaheads share the classes and the memo
            r"\d{1,3}(?=(?:\d{3})*\b)|\d{1,2}(?=(?:\d{2})*\b)|[a-z]+",
        ] {
            let parsed = Pattern::parse(pattern).expect("parses");
            for text in [
                format!("{run}a"),
                run.clone(),
                format!("{run} x"),
                format!("{}a", &run[..3999]),
                format!("{arabic}a {arabic}"),
                format!("{run}a{}", &run[..300]),
                "12 123 1234a 12345 123456 1234567a 1,234,567.89x".to_owned(),
            ] {
                let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                assert_eq!(
                    ours,
                    oracle_matches(pattern, &text),
                    "{pattern} on {} bytes",
                    text.len()
                );
            }
        }
    }

    #[test]
    fn group_repeats_and_long_runs_on_a_small_stack() {
        // A group under `+`, `*` or `{n,}` is declined (its choices would
        // grow with the run); the optional form and class runs each leave one
        // choice on a stack sized at parse time, so a match uses no stack
        // frame per repetition or per character and long inputs match on a
        // 256 KiB stack.
        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                for pattern in [
                    r"(?:\r?\n)+",
                    r"(?: ?\p{L})+",
                    r"(?:ab)*",
                    r"(?:ab|c){2,}",
                    r"(?:'s){1,3}",
                ] {
                    assert!(Pattern::parse(pattern).is_none(), "{pattern}");
                }
                let text = "\r\n".repeat(50_000) + &" x".repeat(50_000) + &"'s".repeat(5_000);
                for pattern in [GPT2, QWEN35, CASED] {
                    let parsed = Pattern::parse(pattern).expect("parses");
                    let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                    assert_eq!(ours, oracle_matches(pattern, &text), "{pattern}");
                }
            })
            .expect("spawn");
        worker
            .join()
            .expect("the matcher must not overflow a small stack");
    }

    /// The plane bitmap of a class answers as its range and category tables
    /// do: every category table over the whole plane, a negated class with
    /// ranges and categories over the whole plane, and every class of every
    /// pattern over a stride of the plane, the pool and the planes above.
    #[test]
    fn the_plane_bitmap_agrees_with_the_tables() {
        let plane = || (0u32..0x1_0000).filter_map(char::from_u32);
        for &(bit, table) in &CATEGORY_TABLES {
            let class = Class {
                categories: bit,
                ..Class::default()
            }
            .finish();
            for c in plane() {
                let code = u32::from(c);
                assert_eq!(
                    class.contains(c),
                    in_ranges(table, code),
                    "category {bit:#x} at U+{code:04X}"
                );
            }
        }
        let negated = Class {
            negated: true,
            categories: CAT_LETTER | CAT_MARK,
            ranges: vec![(0x80, 0x9F), (0x3000, 0x303F), (0x1F600, 0x1F64F)],
            ..Class::default()
        }
        .finish();
        for c in plane() {
            let code = u32::from(c);
            assert_eq!(
                negated.contains(c),
                negated.contains_slow(code),
                "the negated class at U+{code:04X}"
            );
        }
        let beyond = [
            '\u{1F600}',
            '\u{1F3FD}',
            '\u{20000}',
            '\u{E0001}',
            '\u{10FFFF}',
        ];
        for pattern in PATTERNS {
            let parsed = Pattern::parse(pattern).expect("the pattern parses");
            for (i, class) in parsed.classes().iter().enumerate() {
                let sample = (0u32..0x1_0000)
                    .step_by(7)
                    .filter_map(char::from_u32)
                    .chain(POOL.iter().copied())
                    .chain(beyond);
                for c in sample {
                    let code = u32::from(c);
                    assert_eq!(
                        class.contains(c),
                        class.contains_slow(code),
                        "{pattern}: class {i} at U+{code:04X}"
                    );
                }
            }
        }
    }
}
