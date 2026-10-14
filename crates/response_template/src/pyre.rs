//! The part of Python's `regex` module that response templates use.
//! transformers compiles every pattern with `regex.DOTALL` and calls `search`,
//! `search(..., partial=True)` and `finditer`. A pattern is read (`parse`)
//! into a tree of every construct the module reads, and compiled here into a
//! small program for a backtracking matcher that returns what the `regex`
//! module returns: the leftmost match in backtracking order, its named groups,
//! and the start of a partial match. What the matcher runs is listed in the
//! README; a tree with anything else is refused with the construct's name
//! when it compiles.

pub(crate) mod parse;

pub(crate) use parse::Regex;

use crate::error::Unsupported;

/// `\w` of the `regex` module: Rust's word characters and those its newer
/// Unicode tables add.
pub(crate) fn is_word(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    let code = u32::from(c);
    let i = WORD_ADDED.partition_point(|&(lo, _)| lo <= code);
    regex_syntax::is_word_character(c) || (i > 0 && code <= WORD_ADDED[i - 1].1)
}

/// Word characters of the `regex` module (Unicode 17) that regex-syntax's
/// tables (Unicode 16) lack; a unit test compares every code point.
const WORD_ADDED: [(u32, u32); 27] = [
    (0x88f, 0x88f),
    (0xc5c, 0xc5c),
    (0xcdc, 0xcdc),
    (0x1acf, 0x1add),
    (0x1ae0, 0x1aeb),
    (0xa7ce, 0xa7cf),
    (0xa7d2, 0xa7d2),
    (0xa7d4, 0xa7d4),
    (0xa7f1, 0xa7f1),
    (0x10940, 0x10959),
    (0x10ec5, 0x10ec7),
    (0x10efa, 0x10efb),
    (0x11b60, 0x11b67),
    (0x11db0, 0x11ddb),
    (0x11de0, 0x11de9),
    (0x16ea0, 0x16eb8),
    (0x16ebb, 0x16ed3),
    (0x16ff2, 0x16ff6),
    (0x187f8, 0x187ff),
    (0x18d09, 0x18d1e),
    (0x18d80, 0x18df2),
    (0x1e6c0, 0x1e6de),
    (0x1e6e0, 0x1e6f5),
    (0x1e6fe, 0x1e6ff),
    (0x2b73a, 0x2b73f),
    (0x2cea2, 0x2cead),
    (0x323b0, 0x33479),
];

/// `\d` of the `regex` module: the decimal digits of Unicode 17; a unit test
/// compares every code point.
pub(crate) fn is_digit(c: char) -> bool {
    let code = u32::from(c);
    let i = DIGIT.partition_point(|&(lo, _)| lo <= code);
    i > 0 && code <= DIGIT[i - 1].1
}

const DIGIT: [(u32, u32); 72] = [
    (0x30, 0x39),
    (0x660, 0x669),
    (0x6f0, 0x6f9),
    (0x7c0, 0x7c9),
    (0x966, 0x96f),
    (0x9e6, 0x9ef),
    (0xa66, 0xa6f),
    (0xae6, 0xaef),
    (0xb66, 0xb6f),
    (0xbe6, 0xbef),
    (0xc66, 0xc6f),
    (0xce6, 0xcef),
    (0xd66, 0xd6f),
    (0xde6, 0xdef),
    (0xe50, 0xe59),
    (0xed0, 0xed9),
    (0xf20, 0xf29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17e0, 0x17e9),
    (0x1810, 0x1819),
    (0x1946, 0x194f),
    (0x19d0, 0x19d9),
    (0x1a80, 0x1a89),
    (0x1a90, 0x1a99),
    (0x1b50, 0x1b59),
    (0x1bb0, 0x1bb9),
    (0x1c40, 0x1c49),
    (0x1c50, 0x1c59),
    (0xa620, 0xa629),
    (0xa8d0, 0xa8d9),
    (0xa900, 0xa909),
    (0xa9d0, 0xa9d9),
    (0xa9f0, 0xa9f9),
    (0xaa50, 0xaa59),
    (0xabf0, 0xabf9),
    (0xff10, 0xff19),
    (0x104a0, 0x104a9),
    (0x10d30, 0x10d39),
    (0x10d40, 0x10d49),
    (0x11066, 0x1106f),
    (0x110f0, 0x110f9),
    (0x11136, 0x1113f),
    (0x111d0, 0x111d9),
    (0x112f0, 0x112f9),
    (0x11450, 0x11459),
    (0x114d0, 0x114d9),
    (0x11650, 0x11659),
    (0x116c0, 0x116c9),
    (0x116d0, 0x116e3),
    (0x11730, 0x11739),
    (0x118e0, 0x118e9),
    (0x11950, 0x11959),
    (0x11bf0, 0x11bf9),
    (0x11c50, 0x11c59),
    (0x11d50, 0x11d59),
    (0x11da0, 0x11da9),
    (0x11de0, 0x11de9),
    (0x11f50, 0x11f59),
    (0x16130, 0x16139),
    (0x16a60, 0x16a69),
    (0x16ac0, 0x16ac9),
    (0x16b50, 0x16b59),
    (0x16d70, 0x16d79),
    (0x1ccf0, 0x1ccf9),
    (0x1d7ce, 0x1d7ff),
    (0x1e140, 0x1e149),
    (0x1e2f0, 0x1e2f9),
    (0x1e4f0, 0x1e4f9),
    (0x1e5f1, 0x1e5fa),
    (0x1e950, 0x1e959),
    (0x1fbf0, 0x1fbf9),
];

/// A class escape: `\w`, `\W`, `\s`, `\S`, `\d`, `\D`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Escape {
    Word,
    NotWord,
    Space,
    NotSpace,
    Digit,
    NotDigit,
}

impl Escape {
    /// With the `regex` module's tables: `\s` is Unicode White_Space.
    fn matches(self, c: char) -> bool {
        match self {
            Self::Word => is_word(c),
            Self::NotWord => !is_word(c),
            Self::Space => c.is_whitespace(),
            Self::NotSpace => !c.is_whitespace(),
            Self::Digit => is_digit(c),
            Self::NotDigit => !is_digit(c),
        }
    }
}

/// One character: a literal, `.`, a class escape, a property, or a class of
/// characters, ranges of them, class escapes and properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Set {
    Char(char),
    Any,
    Escape(Escape),
    /// A Unicode property or POSIX class as written (`\pL`, `[:alpha:]`,
    /// `\h`): read, not run.
    Property(Box<str>),
    Class {
        negated: bool,
        ranges: Box<[(char, char)]>,
        escapes: Box<[Escape]>,
        properties: Box<[Box<str>]>,
    },
}

impl Set {
    /// Whether `c` is in the set; a set with a property is never run.
    fn matches(&self, c: char) -> bool {
        match self {
            Self::Char(x) => *x == c,
            Self::Any => true,
            Self::Escape(e) => e.matches(c),
            Self::Property(_) => false,
            Self::Class {
                negated,
                ranges,
                escapes,
                ..
            } => {
                (ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&c))
                    || escapes.iter().any(|e| e.matches(c)))
                    != *negated
            }
        }
    }

    /// The property this set holds, if any: the matcher does not run it.
    fn property(&self) -> Option<&str> {
        match self {
            Self::Property(name) => Some(name),
            Self::Class { properties, .. } => properties.first().map(|p| &**p),
            _ => None,
        }
    }
}

/// The characters a match can start with. A start from which no match or
/// partial match can begin is skipped without running the matcher there; no
/// state it would have entered is reachable from a later start.
#[derive(Debug)]
enum First {
    Char(char),
    Chars(FirstChars),
}

#[derive(Debug)]
struct FirstChars {
    ascii: [bool; 128],
    /// Other characters, sorted.
    chars: Vec<char>,
    /// Sets other than one character.
    sets: Vec<Set>,
}

impl FirstChars {
    fn matches(&self, c: char) -> bool {
        match self.ascii.get(c as usize) {
            Some(&first) => first,
            None => {
                self.chars.binary_search(&c).is_ok() || self.sets.iter().any(|set| set.matches(c))
            }
        }
    }
}

impl First {
    /// `None` when the pattern can match the empty string, can start with
    /// any character, or starts with a backreference.
    fn of(node: &Node) -> Option<Self> {
        let (mut chars, mut sets) = (Vec::new(), Vec::new());
        if node.nullable() || !first_sets(node, &mut chars, &mut sets) || sets.contains(&Set::Any) {
            return None;
        }
        chars.sort_unstable();
        chars.dedup();
        if let ([c], []) = (chars.as_slice(), sets.as_slice()) {
            return Some(Self::Char(*c));
        }
        let mut ascii = [false; 128];
        for (c, first) in (0..128u8).map(char::from).zip(ascii.iter_mut()) {
            *first = chars.binary_search(&c).is_ok() || sets.iter().any(|set: &Set| set.matches(c));
        }
        chars.retain(|c| !c.is_ascii());
        Some(Self::Chars(FirstChars { ascii, chars, sets }))
    }
}

/// Add the characters and other sets that hold the first character of a
/// match of `node`; false when a backreference can come first.
fn first_sets(node: &Node, chars: &mut Vec<char>, sets: &mut Vec<Set>) -> bool {
    match node {
        Node::One(Set::Char(c)) => {
            chars.push(*c);
            true
        }
        Node::One(set) => {
            if !sets.contains(set) {
                sets.push(set.clone());
            }
            true
        }
        Node::Look(_) | Node::Around { .. } => true,
        Node::Backref(_) | Node::Other(..) => false,
        Node::Group(_, inner) | Node::Repeat { node: inner, .. } => first_sets(inner, chars, sets),
        Node::Alt(branches) => branches
            .iter()
            .all(|branch| first_sets(branch, chars, sets)),
        Node::Cat(nodes) => {
            for node in nodes {
                if !first_sets(node, chars, sets) {
                    return false;
                }
                if !node.nullable() {
                    break;
                }
            }
            true
        }
    }
}

/// `^` and `\A` (start of text), `$` (end, or before a final newline), `\Z`
/// and `\z` (end), `\b`, `\B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Look {
    Start,
    Dollar,
    End,
    WordBoundary,
    NotWordBoundary,
}

impl Look {
    /// The end of the haystack is the end of the text, also in a partial search.
    fn at(self, hay: &str, at: usize) -> bool {
        match self {
            Self::Start => at == 0,
            Self::End => at == hay.len(),
            Self::Dollar => at == hay.len() || (at + 1 == hay.len() && hay.ends_with('\n')),
            Self::WordBoundary => {
                hay[..at].chars().next_back().is_some_and(is_word)
                    != hay[at..].chars().next().is_some_and(is_word)
            }
            Self::NotWordBoundary => !Self::WordBoundary.at(hay, at),
        }
    }
}

/// How a quantifier repeats: greedy, lazy (`*?`) or possessive (`*+`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantifier {
    Greedy,
    Lazy,
    Possessive,
}

/// Constructs the reader reads that the matcher does not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Other {
    /// `(?>...)`.
    Atomic,
    /// `(?|...|...)`.
    BranchReset,
    /// `(?(1)yes|no)`, `(?(?=...)yes|no)`.
    Conditional,
    /// `a{e<=1}`.
    Fuzzy,
    /// `(?R)`, `(?1)`, `(?&name)`.
    GroupCall,
    /// `\K`.
    Keep,
    /// `\G`.
    SearchAnchor,
    /// `\X`.
    Grapheme,
    /// `\R`.
    LineBreak,
    /// `\m`.
    WordStart,
    /// `\M`.
    WordEnd,
    /// `(*FAIL)`, `(*F)`.
    Fail,
    /// `(*PRUNE)`.
    Prune,
    /// `(*SKIP)`.
    Skip,
}

impl Other {
    fn name(self) -> &'static str {
        match self {
            Self::Atomic => "atomic group",
            Self::BranchReset => "branch reset group",
            Self::Conditional => "conditional",
            Self::Fuzzy => "fuzzy matching",
            Self::GroupCall => "group call",
            Self::Keep => "escape \\K",
            Self::SearchAnchor => "escape \\G",
            Self::Grapheme => "escape \\X",
            Self::LineBreak => "escape \\R",
            Self::WordStart => "escape \\m",
            Self::WordEnd => "escape \\M",
            Self::Fail => "verb (*FAIL)",
            Self::Prune => "verb (*PRUNE)",
            Self::Skip => "verb (*SKIP)",
        }
    }
}

/// A pattern as the `regex` module reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Node {
    One(Set),
    Look(Look),
    /// `(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)`.
    Around {
        behind: bool,
        negated: bool,
        node: Box<Node>,
    },
    /// A capturing group (its number) or `(?:...)`.
    Group(Option<usize>, Box<Node>),
    Alt(Vec<Node>),
    Cat(Vec<Node>),
    /// `node` from `min` to `max` times (`None`: no limit): `*`, `+`, `?`,
    /// `{m,n}`.
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        kind: Quantifier,
    },
    /// A backreference to a group number.
    Backref(usize),
    /// Another construct, with what it holds.
    Other(Other, Vec<Node>),
}

impl Node {
    /// Whether the `regex` module can reduce this alternative to a negated
    /// class of one character (after moving a common prefix out of the
    /// alternatives, merging nested ones and dropping empty groups). A class
    /// with a class escape is not such a class; one of a lone surrogate is.
    fn negated_char(&self) -> bool {
        match self {
            Self::One(Set::Class {
                negated,
                ranges,
                escapes,
                properties,
            }) => {
                *negated
                    && escapes.is_empty()
                    && properties.is_empty()
                    && (ranges.is_empty() || (ranges.len() == 1 && ranges[0].0 == ranges[0].1))
            }
            Self::Group(None, node) => node.negated_char(),
            Self::Cat(nodes) => nodes
                .iter()
                .rev()
                .find(|node| !node.empty_group())
                .is_some_and(Self::negated_char),
            Self::Alt(nodes) => nodes.iter().any(Self::negated_char),
            _ => false,
        }
    }

    /// Whether this is a non-capturing group of nothing (such as `(?:)` or
    /// `(?:(?:))`), which the `regex` module drops when it parses.
    fn empty_group(&self) -> bool {
        match self {
            Self::Group(None, node) => node.empty_group(),
            Self::Cat(nodes) => nodes.iter().all(Self::empty_group),
            _ => false,
        }
    }

    fn has_backref(&self) -> bool {
        match self {
            Self::Backref(_) => true,
            Self::Group(_, node) | Self::Repeat { node, .. } | Self::Around { node, .. } => {
                node.has_backref()
            }
            Self::Alt(nodes) | Self::Cat(nodes) | Self::Other(_, nodes) => {
                nodes.iter().any(Self::has_backref)
            }
            Self::One(_) | Self::Look(_) => false,
        }
    }

    fn nullable(&self) -> bool {
        match self {
            Self::One(_) => false,
            Self::Look(_) | Self::Around { .. } | Self::Backref(_) | Self::Other(..) => true,
            Self::Group(_, node) => node.nullable(),
            Self::Repeat { node, min, .. } => *min == 0 || node.nullable(),
            Self::Alt(nodes) => nodes.iter().any(Self::nullable),
            Self::Cat(nodes) => nodes.iter().all(Self::nullable),
        }
    }

    /// The `regex` module's `is_empty`: whether the item reads as nothing,
    /// so that a quantifier of it is dropped.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Group(None, node) | Self::Repeat { node, .. } => node.is_empty(),
            Self::Around { negated, node, .. } => !negated && node.is_empty(),
            Self::Alt(nodes) | Self::Cat(nodes) => nodes.iter().all(Self::is_empty),
            Self::Other(Other::Atomic | Other::Fuzzy | Other::BranchReset, nodes) => {
                nodes.iter().all(Self::is_empty)
            }
            // A lookaround conditional reads as nothing when its test and its
            // `yes` branch do.
            Self::Other(Other::Conditional, nodes) => match nodes.as_slice() {
                [Self::Around { node: test, .. }, yes, _] => test.is_empty() && yes.is_empty(),
                nodes => nodes.iter().all(Self::is_empty),
            },
            _ => false,
        }
    }
}

/// What the tail of a lazy repeat starts with, as the `regex` module tests
/// it before it lets the repeat take another character (the `test` node of
/// its `LAZY_REPEAT_ONE`): one literal character, a literal string, or
/// something else.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Test {
    Char(char),
    Str(Box<str>),
    Other,
}

#[derive(Debug, Clone)]
enum Inst {
    One(Set),
    /// `x*` (greedy) or `x*?` (lazy) of one character. `memo` numbers the
    /// repeats and branch points of a pattern for [`Run::visit`].
    Star {
        set: Set,
        greedy: bool,
        memo: usize,
    },
    /// `x{min,max}` (greedy) of one character.
    Count {
        set: Set,
        min: u32,
        max: Option<u32>,
        memo: usize,
    },
    /// A lazy repeat of one character, run as the `regex` module runs its
    /// `LAZY_REPEAT_ONE`: `body` and `tail` number the positions it was
    /// entered at and its tail was tried at.
    Lazy {
        set: Set,
        min: u32,
        max: Option<u32>,
        test: Test,
        body: usize,
        tail: usize,
    },
    Look(Look),
    /// `(?=...)`: its body follows, ending in a `Match`; when the body
    /// matches, continue at `end` from where the lookahead started.
    Ahead {
        end: usize,
    },
    /// Try the next instruction, then the one at `second`. `memo` as for
    /// `Star`.
    Split {
        second: usize,
        memo: usize,
    },
    Jmp(usize),
    Save(usize),
    Backref(usize),
    Match,
}

/// How transformers uses a pattern: open and close patterns are also searched
/// with `partial=True`; start anchors and tag patterns only with `finditer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Delimiter,
    Finditer,
}

/// A compiled pattern, as `regex.compile(pattern, regex.DOTALL)`.
#[derive(Debug)]
pub(crate) struct Pattern {
    prog: Vec<Inst>,
    /// Named groups: name and group number.
    names: Vec<(String, usize)>,
    /// Per group number: whether a match can leave the group unset.
    optional: Vec<bool>,
    nullable: bool,
    /// The characters every match starts with, to skip ahead to.
    first: Option<First>,
    /// A literal every match contains: without it there is no complete match.
    required: Option<String>,
    /// The number of repeats and branch points; the positions a search
    /// entered them at are remembered (sound without backreferences, which
    /// then set this to 0).
    memos: usize,
    /// Per instruction: for a greedy repeat, whether what follows it can
    /// start with a character it takes. When it cannot, giving characters
    /// back could only end where what follows fails at once, as in `\s*<` or
    /// `[^>]+>`, so the repeat gives none back.
    gives_back: Vec<bool>,
}

/// A complete match with its group spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Match {
    pub(crate) start: usize,
    pub(crate) end: usize,
    slots: Vec<Option<usize>>,
}

/// A result of `search(..., partial=True)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Found {
    Complete(Match),
    /// The text from this position to the end could still become a match.
    Partial(usize),
}

/// Searches of one unchanged haystack, remembered: a result from an earlier
/// position holds while the position has not passed its start, and nothing
/// found stays nothing.
#[derive(Debug, Default, Clone)]
pub(crate) struct Memo {
    complete: Option<(usize, Option<Match>)>,
    partial: Option<(usize, Option<usize>)>,
}

/// Why a lazy repeat in an open or close pattern is refused.
const LAZY_POSITION: &str = "lazy repeat in this position of an open or close pattern";

impl Pattern {
    /// Read and compile a Python pattern, or name the construct that is not
    /// ported. A `finditer` pattern must not match the empty string (see the
    /// README).
    pub(crate) fn new(source: &str, role: Role) -> Result<Self, Unsupported> {
        let regex = Regex::parse(source).map_err(|e| Unsupported::Regex(e.to_string()))?;
        Self::compile(&regex, role)
    }

    /// Compile a pattern the reader read, or name the construct the matcher
    /// does not run.
    pub(crate) fn compile(regex: &Regex, role: Role) -> Result<Self, Unsupported> {
        check(regex, role).map_err(Unsupported::Regex)?;
        if role == Role::Finditer && regex.node.nullable() {
            return Err(Unsupported::EmptyMatch);
        }
        let mut pattern = Self::build(
            &regex.node,
            role,
            regex.names.clone(),
            regex.optional.clone(),
        );
        // A lazy repeat in an open or close pattern runs as the module's only
        // before a literal tail, and not first: where a partial search starts
        // depends on the module's own first sets there.
        if role == Role::Delimiter && lazy_first(&pattern.prog) {
            return Err(Unsupported::Regex(LAZY_POSITION.into()));
        }
        for pc in 0..pattern.prog.len() {
            if !matches!(pattern.prog[pc], Inst::Lazy { .. }) {
                continue;
            }
            let (test, branch_follows) = tail_test(&pattern.prog, pc + 1);
            if role == Role::Delimiter && (test == Test::Other || branch_follows) {
                return Err(Unsupported::Regex(LAZY_POSITION.into()));
            }
            if let Inst::Lazy { test: slot, .. } = &mut pattern.prog[pc] {
                *slot = test;
            }
        }
        Ok(pattern)
    }

    /// `"|".join(regex.escape(s) for s in literals)`.
    pub(crate) fn literals(literals: &[String]) -> Self {
        let node = Node::Alt(
            literals
                .iter()
                .map(|s| Node::Cat(s.chars().map(|c| Node::One(Set::Char(c))).collect()))
                .collect(),
        );
        // A search enters no branch point of a list twice at one position.
        Self {
            memos: 0,
            ..Self::build(&node, Role::Delimiter, Vec::new(), vec![false])
        }
    }

    fn build(node: &Node, role: Role, names: Vec<(String, usize)>, optional: Vec<bool>) -> Self {
        let (mut prog, mut memos) = (Vec::new(), 0);
        compile(node, role, None, &mut prog, &mut memos);
        prog.push(Inst::Match);
        let first = First::of(node);
        let (mut run, mut longest) = (String::new(), String::new());
        required_literal(node, &mut run, &mut longest);
        end_run(&mut run, &mut longest);
        if node.has_backref() {
            memos = 0;
        }
        let gives_back = (0..prog.len())
            .map(|pc| match &prog[pc] {
                Inst::Star {
                    set, greedy: true, ..
                }
                | Inst::Count { set, .. } => literal_firsts(&prog, pc + 1)
                    .is_none_or(|next| next.iter().any(|&c| set.matches(c))),
                _ => false,
            })
            .collect();
        Self {
            prog,
            names,
            optional,
            nullable: node.nullable(),
            first,
            required: (longest.len() > 1).then_some(longest),
            memos,
            gives_back,
        }
    }

    /// Whether the pattern can match the empty string.
    pub(crate) fn nullable(&self) -> bool {
        self.nullable
    }

    /// The names of the named groups (`pattern.groupindex`).
    #[cfg(test)]
    pub(crate) fn group_names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(|(name, _)| name.as_str())
    }

    /// Whether the named group exists and every match sets it.
    pub(crate) fn always_sets(&self, name: &str) -> Option<bool> {
        let (_, group) = self.names.iter().find(|(n, _)| n == name)?;
        Some(!self.optional[*group])
    }

    /// `m.groupdict()`: every named group, `None` where the match left it unset.
    pub(crate) fn groupdict<'h>(&self, hay: &'h str, m: &Match) -> Vec<(&str, Option<&'h str>)> {
        self.names
            .iter()
            .map(|(name, group)| {
                let span = m.slots[2 * group].zip(m.slots[2 * group + 1]);
                (name.as_str(), span.map(|(a, b)| &hay[a..b]))
            })
            .collect()
    }

    /// `pattern.search(hay, pos)`.
    pub(crate) fn search(&self, hay: &str, pos: usize, memo: &mut Memo) -> Option<Match> {
        if let Some((from, found)) = &memo.complete {
            if pos >= *from && found.as_ref().is_none_or(|m| m.start >= pos) {
                return found.clone();
            }
        }
        let found = match self.find(hay, pos, false) {
            Some((start, Step::Match(end, slots))) => Some(Match { start, end, slots }),
            _ => None,
        };
        memo.complete = Some((pos, found.clone()));
        found
    }

    /// `pattern.search(hay, pos, partial=True)`: a complete match if there is
    /// one; else the first start, before the end of the haystack, from which
    /// the matcher reads past the end. (At the end itself the `regex` module
    /// reports an empty partial match or odd spans; transformers never
    /// commits or holds anything for a match that starts there.)
    pub(crate) fn search_partial(&self, hay: &str, pos: usize, memo: &mut Memo) -> Option<Found> {
        if let Some(m) = self.search(hay, pos, memo) {
            return Some(Found::Complete(m));
        }
        if let Some((from, found)) = memo.partial {
            if pos >= from && found.is_none_or(|start| start >= pos) {
                return found.map(Found::Partial);
            }
        }
        let found = self.find(hay, pos, true).map(|(start, _)| start);
        memo.partial = Some((pos, found));
        found.map(Found::Partial)
    }

    /// `pattern.finditer(hay)`, for patterns that cannot match the empty string.
    pub(crate) fn finditer<'h>(&'h self, hay: &'h str) -> impl Iterator<Item = Match> + 'h {
        let mut at = Some(0);
        std::iter::from_fn(move || {
            let (start, step) = self.find(hay, at?, false)?;
            let Step::Match(end, slots) = step else {
                return None;
            };
            at = (end > start).then_some(end);
            Some(Match { start, end, slots })
        })
    }

    /// The first start from `pos` with a match (or, in a partial search, a
    /// partial match), trying starts in order like the `regex` module.
    fn find(&self, hay: &str, pos: usize, partial: bool) -> Option<(usize, Step)> {
        if let (Some(literal), false) = (&self.required, partial) {
            if !hay[pos..].contains(literal.as_str()) {
                return None;
            }
        }
        let mut run = Run {
            slots: vec![None; 2 * self.optional.len()],
            stack: Vec::new(),
            visited: vec![Vec::new(); self.memos],
            base: pos,
        };
        let mut start = pos;
        loop {
            match &self.first {
                Some(First::Char(c)) => start += hay[start..].find(*c)?,
                Some(First::Chars(first)) => start += hay[start..].find(|c| first.matches(c))?,
                None => {}
            }
            if partial && start == hay.len() {
                return None;
            }
            match self.run(hay, 0, start, partial, &mut run) {
                Step::Fail => {}
                step => return Some((start, step)),
            }
            start += hay[start..].chars().next()?.len_utf8();
        }
    }

    /// One backtracking run of the program from `pc` at `start`, up to the
    /// next `Match`. In a partial search, reading past the end of the
    /// haystack is a partial match, except when a lazy repeat would extend
    /// there; entering a lazy repeat at the end is one too. A lazy repeat
    /// before a literal follows the `regex` module's own rules instead
    /// ([`Pattern::lazy`]).
    fn run(&self, hay: &str, pc: usize, start: usize, partial: bool, run: &mut Run) -> Step {
        run.stack.push(Frame::Try(pc, start));
        while let Some(frame) = run.stack.pop() {
            let (mut pc, mut at) = match frame {
                Frame::Try(pc, at) => (pc, at),
                Frame::Restore(slot, old) => {
                    run.slots[slot] = old;
                    continue;
                }
                // A greedy repeat gives back one more character.
                Frame::GiveBack { pc, floor, at } => {
                    if at > floor {
                        let back = hay[..at].chars().next_back().map_or(0, char::len_utf8);
                        run.stack.push(Frame::GiveBack {
                            pc,
                            floor,
                            at: at - back,
                        });
                    }
                    (pc + 1, at)
                }
                // A lazy repeat takes one more character.
                Frame::Extend { pc, at } => {
                    let Inst::Star { set, memo, .. } = &self.prog[pc] else {
                        continue;
                    };
                    let Some(c) = hay[at..].chars().next().filter(|&c| set.matches(c)) else {
                        continue;
                    };
                    let next = at + c.len_utf8();
                    if !run.visit(*memo, next) {
                        continue;
                    }
                    run.stack.push(Frame::Extend { pc, at: next });
                    (pc + 1, next)
                }
                // A lazy repeat run as the module's takes more characters.
                Frame::Lazy { pc, pos, count } => {
                    match self.lazy(hay, pc, pos, count, partial, run) {
                        Lazy::Partial => {
                            run.stack.clear();
                            return Step::Partial;
                        }
                        Lazy::Fail => continue,
                        Lazy::Match(pos, count) => {
                            self.push_lazy(hay, pc, pos, count, run);
                            (pc + 1, pos)
                        }
                    }
                }
            };
            loop {
                match &self.prog[pc] {
                    Inst::Match => {
                        run.stack.clear();
                        return Step::Match(at, run.slots.clone());
                    }
                    Inst::One(set) => match hay[at..].chars().next() {
                        Some(c) if set.matches(c) => {
                            at += c.len_utf8();
                            pc += 1;
                        }
                        None if partial => {
                            run.stack.clear();
                            return Step::Partial;
                        }
                        _ => break,
                    },
                    Inst::Star {
                        set,
                        greedy: true,
                        memo,
                    } => {
                        if !run.visit(*memo, at) {
                            break;
                        }
                        // Take as many as match, stopping where an earlier
                        // entry already took over.
                        let run_end = hay[at..]
                            .find(|c| !set.matches(c))
                            .map_or(hay.len(), |k| at + k);
                        let end = run.claim(*memo, at, run_end, hay);
                        if partial && end == hay.len() {
                            run.stack.clear();
                            return Step::Partial;
                        }
                        if end > at && self.gives_back[pc] {
                            let back = hay[..end].chars().next_back().map_or(0, char::len_utf8);
                            run.stack.push(Frame::GiveBack {
                                pc,
                                floor: at,
                                at: end - back,
                            });
                        }
                        at = end;
                        pc += 1;
                    }
                    Inst::Star { memo, .. } => {
                        if partial && at == hay.len() {
                            run.stack.clear();
                            return Step::Partial;
                        }
                        if !run.visit(*memo, at) {
                            break;
                        }
                        run.stack.push(Frame::Extend { pc, at });
                        pc += 1;
                    }
                    // The module's `count_one` up to the maximum, then giving
                    // back down to the minimum: reaching the end below the
                    // maximum is a partial match.
                    Inst::Count {
                        set,
                        min,
                        max,
                        memo,
                    } => {
                        if !run.visit(*memo, at) {
                            break;
                        }
                        let (mut end, mut count, mut floor) = (at, 0, at);
                        while max.is_none_or(|max| count < max) {
                            let Some(c) = hay[end..].chars().next().filter(|&c| set.matches(c))
                            else {
                                break;
                            };
                            end += c.len_utf8();
                            count += 1;
                            if count == *min {
                                floor = end;
                            }
                        }
                        if partial && end == hay.len() && max.is_none_or(|max| count < max) {
                            run.stack.clear();
                            return Step::Partial;
                        }
                        if count < *min {
                            break;
                        }
                        if end > floor && self.gives_back[pc] {
                            let back = hay[..end].chars().next_back().map_or(0, char::len_utf8);
                            run.stack.push(Frame::GiveBack {
                                pc,
                                floor,
                                at: end - back,
                            });
                        }
                        at = end;
                        pc += 1;
                    }
                    // `count_one` up to the minimum (reaching the end below
                    // it is a partial match); backtracking takes more
                    // ([`Pattern::lazy`]).
                    Inst::Lazy { set, min, body, .. } => {
                        if !run.visit(*body, at) {
                            break;
                        }
                        let (mut end, mut count) = (at, 0);
                        while count < *min {
                            let Some(c) = hay[end..].chars().next().filter(|&c| set.matches(c))
                            else {
                                break;
                            };
                            end += c.len_utf8();
                            count += 1;
                        }
                        if count < *min {
                            if partial && end == hay.len() {
                                run.stack.clear();
                                return Step::Partial;
                            }
                            break;
                        }
                        self.push_lazy(hay, pc, end, count, run);
                        at = end;
                        pc += 1;
                    }
                    Inst::Look(look) if look.at(hay, at) => pc += 1,
                    Inst::Look(_) => break,
                    // The body runs with its own stack and memory: whether it
                    // matches depends only on the position, and once it has
                    // matched it is not tried again (it sets no groups).
                    Inst::Ahead { end } => {
                        let mut body = Run {
                            slots: Vec::new(),
                            stack: Vec::new(),
                            visited: vec![Vec::new(); run.visited.len()],
                            base: at,
                        };
                        match self.run(hay, pc + 1, at, partial, &mut body) {
                            Step::Match(..) => pc = *end,
                            Step::Partial => {
                                run.stack.clear();
                                return Step::Partial;
                            }
                            Step::Fail => break,
                        }
                    }
                    Inst::Split { second, memo } => {
                        if !run.visit(*memo, at) {
                            break;
                        }
                        run.stack.push(Frame::Try(*second, at));
                        pc += 1;
                    }
                    Inst::Jmp(to) => pc = *to,
                    Inst::Save(slot) => {
                        run.stack.push(Frame::Restore(*slot, run.slots[*slot]));
                        run.slots[*slot] = Some(at);
                        pc += 1;
                    }
                    Inst::Backref(group) => {
                        let Some((a, b)) = run.slots[2 * group].zip(run.slots[2 * group + 1])
                        else {
                            break;
                        };
                        if !hay[at..].starts_with(&hay[a..b]) {
                            break;
                        }
                        at += b - a;
                        pc += 1;
                    }
                }
            }
        }
        Step::Fail
    }

    /// Remember the lazy repeat at `pc`, now at `pos` with `count`
    /// characters, so that backtracking can take more: while it is below its
    /// maximum and characters are left.
    fn push_lazy(&self, hay: &str, pc: usize, pos: usize, count: u32, run: &mut Run) {
        if let Inst::Lazy { max, .. } = &self.prog[pc] {
            if pos < hay.len() && max.is_none_or(|max| count < max) {
                run.stack.push(Frame::Lazy { pc, pos, count });
            }
        }
    }

    /// Backtracking into the lazy repeat at `pc`, now at `pos` with `count`
    /// characters: the next position from which its tail can match, found as
    /// the `regex` module's `LAZY_REPEAT_ONE` finds it. Before a literal
    /// character, a partial search is a partial match once at most one
    /// character is left, whatever it is. Before a literal string, it is one
    /// once no later copy of the string is left in the text (within the
    /// maximum), whatever comes before.
    fn lazy(
        &self,
        hay: &str,
        pc: usize,
        mut pos: usize,
        mut count: u32,
        partial: bool,
        run: &mut Run,
    ) -> Lazy {
        let Inst::Lazy {
            set,
            max,
            test,
            tail,
            ..
        } = &self.prog[pc]
        else {
            return Lazy::Fail;
        };
        let below_max = |count: u32| max.is_none_or(|max| count < max);
        match test {
            Test::Char(ch) => loop {
                let mut rest = hay[pos..].chars();
                let next = rest.next();
                let last = rest.next().is_none();
                if partial && last {
                    return Lazy::Partial;
                }
                if last || !below_max(count) {
                    return Lazy::Fail;
                }
                let Some(c) = next.filter(|&c| set.matches(c)) else {
                    return Lazy::Fail;
                };
                pos += c.len_utf8();
                count += 1;
                if hay[pos..].starts_with(*ch) && run.visit(*tail, pos) {
                    return Lazy::Match(pos, count);
                }
            },
            Test::Str(s) => loop {
                if partial && pos >= hay.len() {
                    return Lazy::Partial;
                }
                let from = pos + hay[pos..].chars().next().map_or(1, char::len_utf8);
                let found = hay
                    .get(from..)
                    .and_then(|rest| rest.find(&**s))
                    .map(|k| from + k)
                    .filter(|&at| {
                        max.is_none_or(|max| {
                            u64::from(count) + hay[pos..at].chars().count() as u64 <= u64::from(max)
                        })
                    });
                let Some(at) = found else {
                    return if partial { Lazy::Partial } else { Lazy::Fail };
                };
                for c in hay[pos..at].chars() {
                    if !set.matches(c) {
                        return Lazy::Fail;
                    }
                    count += 1;
                }
                pos = at;
                if run.visit(*tail, pos) {
                    return Lazy::Match(pos, count);
                }
            },
            Test::Other => loop {
                if !below_max(count) {
                    return Lazy::Fail;
                }
                let Some(c) = hay[pos..].chars().next().filter(|&c| set.matches(c)) else {
                    return Lazy::Fail;
                };
                pos += c.len_utf8();
                count += 1;
                if run.visit(*tail, pos) {
                    return Lazy::Match(pos, count);
                }
            },
        }
    }
}

#[derive(Debug)]
enum Step {
    Match(usize, Vec<Option<usize>>),
    Partial,
    Fail,
}

/// Backtracking into a lazy repeat: its tail can be tried at this position
/// with this many characters, there is no such position, or the search is a
/// partial match.
enum Lazy {
    Match(usize, u32),
    Fail,
    Partial,
}

enum Frame {
    Try(usize, usize),
    Restore(usize, Option<usize>),
    GiveBack { pc: usize, floor: usize, at: usize },
    Extend { pc: usize, at: usize },
    Lazy { pc: usize, pos: usize, count: u32 },
}

/// State of one search: group slots, the backtracking stack, and the
/// positions at which it entered each repeat and branch point.
struct Run {
    slots: Vec<Option<usize>>,
    stack: Vec<Frame>,
    /// Per repeat or branch point, one bit per position from `base`, grown
    /// as far as the search gets; empty when not remembering.
    visited: Vec<Vec<u64>>,
    base: usize,
}

impl Run {
    /// The bits of `memo`, covering position `at`.
    fn bits(&mut self, memo: usize, at: usize) -> &mut Vec<u64> {
        let words = (at - self.base) / 64 + 1;
        let bits = &mut self.visited[memo];
        if bits.len() < words {
            bits.resize(words, 0);
        }
        bits
    }

    /// Mark `memo` entered at `at`; false if it was before. What follows
    /// depends only on the pair, so a second entry would fail again.
    fn visit(&mut self, memo: usize, at: usize) -> bool {
        if self.visited.is_empty() {
            return true;
        }
        let bit = at - self.base;
        let word = &mut self.bits(memo, at)[bit / 64];
        let mask = 1u64 << (bit % 64);
        let fresh = *word & mask == 0;
        *word |= mask;
        fresh
    }

    /// A greedy repeat entered at `at` whose characters run to `run_end`:
    /// mark the positions it takes, up to the first one an earlier entry
    /// took, and return where it stops.
    fn claim(&mut self, memo: usize, at: usize, run_end: usize, hay: &str) -> usize {
        if self.visited.is_empty() || run_end == at {
            return run_end;
        }
        let (lo, hi) = (at - self.base + 1, run_end - self.base);
        let bits = self.bits(memo, run_end);
        let mut first_taken = None;
        let mut bit = lo;
        while bit <= hi {
            let (word, offset) = (bit / 64, bit % 64);
            let width = (64 - offset).min(hi - bit + 1);
            let mask = (u64::MAX >> (64 - width)) << offset;
            let taken = bits[word] & mask;
            if taken != 0 {
                let first = taken.trailing_zeros();
                bits[word] |= mask & ((1u64 << first) - 1);
                first_taken = Some(word * 64 + first as usize);
                break;
            }
            bits[word] |= mask;
            bit += width;
        }
        match first_taken {
            None => run_end,
            Some(bit) => {
                let taken_at = bit + self.base;
                let back = hay[..taken_at]
                    .chars()
                    .next_back()
                    .map_or(0, char::len_utf8);
                taken_at - back
            }
        }
    }
}

/// The characters the first character a run from `pc` reads must be, when
/// each is a literal; `None` when it can be another, or when the run can end
/// without reading one.
fn literal_firsts(prog: &[Inst], pc: usize) -> Option<Vec<char>> {
    let (mut firsts, mut seen, mut todo) = (Vec::new(), vec![false; prog.len()], vec![pc]);
    while let Some(pc) = todo.pop() {
        if std::mem::replace(&mut seen[pc], true) {
            continue;
        }
        match &prog[pc] {
            Inst::One(Set::Char(c)) => firsts.push(*c),
            Inst::Look(_) | Inst::Save(_) => todo.push(pc + 1),
            Inst::Jmp(to) => todo.push(*to),
            Inst::Split { second, .. } => todo.extend([pc + 1, *second]),
            _ => return None,
        }
    }
    Some(firsts)
}

/// Whether a run from the start can reach a lazy repeat run as the module's
/// before it reads a character.
fn lazy_first(prog: &[Inst]) -> bool {
    let (mut seen, mut todo) = (vec![false; prog.len()], vec![0]);
    while let Some(pc) = todo.pop() {
        if std::mem::replace(&mut seen[pc], true) {
            continue;
        }
        match &prog[pc] {
            Inst::Lazy { .. } => return true,
            Inst::Look(_) | Inst::Save(_) | Inst::Star { .. } => todo.push(pc + 1),
            Inst::Count { min: 0, .. } => todo.push(pc + 1),
            Inst::Ahead { end } => todo.extend([pc + 1, *end]),
            Inst::Jmp(to) => todo.push(*to),
            Inst::Split { second, .. } => todo.extend([pc + 1, *second]),
            _ => {}
        }
    }
    false
}

/// The test of the lazy repeat whose tail starts at `pc` (past the starts and
/// ends of groups, as the module looks past them), and whether a branch
/// point follows its literals (where the module could move a common prefix
/// of alternatives into them).
fn tail_test(prog: &[Inst], mut pc: usize) -> (Test, bool) {
    while let Some(Inst::Save(_)) = prog.get(pc) {
        pc += 1;
    }
    let mut literals = String::new();
    while let Some(Inst::One(Set::Char(c))) = prog.get(pc) {
        literals.push(*c);
        pc += 1;
    }
    let branch_follows = matches!(prog.get(pc), Some(Inst::Split { .. } | Inst::Jmp(_)));
    let mut chars = literals.chars();
    let test = match (chars.next(), chars.next()) {
        (None, _) => Test::Other,
        (Some(c), None) => Test::Char(c),
        _ => Test::Str(literals.into()),
    };
    (test, branch_follows)
}

/// The longest run of literal characters in `node` that every match contains.
fn required_literal(node: &Node, run: &mut String, longest: &mut String) {
    match node {
        Node::One(Set::Char(c)) => run.push(*c),
        Node::Look(_) => {}
        Node::Group(_, inner) => required_literal(inner, run, longest),
        Node::Cat(nodes) => nodes.iter().for_each(|n| required_literal(n, run, longest)),
        _ => end_run(run, longest),
    }
}

/// Keep `run` if it is longer than `longest` (in characters), and start a new one.
fn end_run(run: &mut String, longest: &mut String) {
    if run.chars().count() > longest.chars().count() {
        std::mem::swap(run, longest);
    }
    run.clear();
}

/// The number of the next repeat or branch point, counting it.
fn next_memo(memos: &mut usize) -> usize {
    *memos += 1;
    *memos - 1
}

/// A branch point before the next instruction; [`set_second`] sets the other.
fn split(prog: &mut Vec<Inst>, memos: &mut usize) -> usize {
    let memo = next_memo(memos);
    prog.push(Inst::Split { second: 0, memo });
    prog.len() - 1
}

fn set_second(prog: &mut [Inst], split: usize) {
    let end = prog.len();
    if let Inst::Split { second, .. } = &mut prog[split] {
        *second = end;
    }
}

/// Whether the `*?` at `items[i]` keeps the matcher's own rule in an open or
/// close pattern: `.*?` before a literal, or after another item and before
/// `\b` and a literal (where partial searches match the `regex` module's).
fn plain_lazy(items: &[Node], i: usize) -> bool {
    let literal = |k: usize| matches!(items.get(k), Some(Node::One(Set::Char(_))));
    let dot_literal = matches!(
        &items[i],
        Node::Repeat { node, .. } if **node == Node::One(Set::Any)
    ) && literal(i + 1);
    let boundary =
        i > 0 && items.get(i + 1) == Some(&Node::Look(Look::WordBoundary)) && literal(i + 2);
    dot_literal || boundary
}

/// Compile `node` into `prog`, numbering its repeats and branch points from
/// `memos`; `items` is the sequence `node` is in, with its index.
fn compile(
    node: &Node,
    role: Role,
    items: Option<(&[Node], usize)>,
    prog: &mut Vec<Inst>,
    memos: &mut usize,
) {
    match node {
        Node::One(set) => prog.push(Inst::One(set.clone())),
        Node::Look(look) => prog.push(Inst::Look(*look)),
        Node::Around { node: body, .. } => {
            let at = prog.len();
            prog.push(Inst::Ahead { end: 0 });
            compile(body, role, None, prog, memos);
            prog.push(Inst::Match);
            prog[at] = Inst::Ahead { end: prog.len() };
        }
        Node::Backref(group) => prog.push(Inst::Backref(*group)),
        Node::Group(group, inner) => {
            if let Some(g) = group {
                prog.push(Inst::Save(2 * g));
            }
            compile(inner, role, None, prog, memos);
            if let Some(g) = group {
                prog.push(Inst::Save(2 * g + 1));
            }
        }
        Node::Cat(nodes) => {
            for (i, n) in nodes.iter().enumerate() {
                compile(n, role, Some((nodes, i)), prog, memos);
            }
        }
        Node::Alt(branches) => {
            let mut jumps = Vec::new();
            for (k, branch) in branches.iter().enumerate() {
                if k + 1 == branches.len() {
                    compile(branch, role, None, prog, memos);
                    break;
                }
                let at = split(prog, memos);
                compile(branch, role, None, prog, memos);
                jumps.push(prog.len());
                prog.push(Inst::Jmp(0));
                set_second(prog, at);
            }
            let end = prog.len();
            for j in jumps {
                prog[j] = Inst::Jmp(end);
            }
        }
        Node::Repeat {
            node: body,
            min,
            max,
            kind,
        } => match (body.as_ref(), *min, *max, *kind) {
            (Node::One(set), 0 | 1, None, Quantifier::Greedy) => {
                if *min == 1 {
                    prog.push(Inst::One(set.clone()));
                }
                let memo = next_memo(memos);
                prog.push(Inst::Star {
                    set: set.clone(),
                    greedy: true,
                    memo,
                });
            }
            (Node::One(set), 0, None, Quantifier::Lazy)
                if role == Role::Finditer
                    || items.is_some_and(|(items, i)| plain_lazy(items, i)) =>
            {
                let memo = next_memo(memos);
                prog.push(Inst::Star {
                    set: set.clone(),
                    greedy: false,
                    memo,
                });
            }
            (body, 0, Some(1), Quantifier::Greedy) => {
                let at = split(prog, memos);
                compile(body, role, None, prog, memos);
                set_second(prog, at);
            }
            (Node::One(set), min, max, Quantifier::Greedy) => {
                let memo = next_memo(memos);
                prog.push(Inst::Count {
                    set: set.clone(),
                    min,
                    max,
                    memo,
                });
            }
            (Node::One(set), min, max, Quantifier::Lazy) => {
                let body = next_memo(memos);
                let tail = next_memo(memos);
                prog.push(Inst::Lazy {
                    set: set.clone(),
                    min,
                    max,
                    test: Test::Other,
                    body,
                    tail,
                });
            }
            // Refused by `check`.
            _ => {}
        },
        // Refused by `check`.
        Node::Other(..) => {}
    }
}

/// What the matcher does not run in a pattern used for `role`, named, in
/// the order the pattern reads.
fn check(regex: &Regex, role: Role) -> Result<(), String> {
    if !regex.flags.is_empty() {
        return Err(format!("inline flags (?{})", regex.flags));
    }
    let mut checker = Checker {
        role,
        closed: vec![false; regex.groups + 1],
        ahead: false,
        enclosed: false,
    };
    checker.node(&regex.node, None)
}

struct Checker {
    role: Role,
    /// Per group number: whether the group closed before this point.
    closed: Vec<bool>,
    /// Inside a lookahead.
    ahead: bool,
    /// Inside an alternative, a repeat or a lookahead, where the `regex`
    /// module may rebuild what follows a lazy repeat.
    enclosed: bool,
}

impl Checker {
    fn node(&mut self, node: &Node, items: Option<(&[Node], usize)>) -> Result<(), String> {
        match node {
            Node::One(set) => set.property().map_or(Ok(()), |name| {
                Err(format!("Unicode property or POSIX class {name}"))
            }),
            Node::Look(Look::NotWordBoundary) => Err("escape \\B".into()),
            Node::Look(_) => Ok(()),
            Node::Around {
                behind,
                negated,
                node,
            } => {
                // Patterns used only with `finditer` are never searched
                // partially, where a zero-width assertion would change the result.
                if self.role == Role::Delimiter || *behind || *negated {
                    return Err("look-around".into());
                }
                let saved = (self.ahead, self.enclosed);
                (self.ahead, self.enclosed) = (true, true);
                let checked = self.node(node, None);
                (self.ahead, self.enclosed) = saved;
                checked
            }
            Node::Group(group, inner) => {
                if group.is_some() && self.ahead {
                    return Err("group in a lookahead".into());
                }
                self.node(inner, None)?;
                if let Some(closed) = group.and_then(|g| self.closed.get_mut(g)) {
                    *closed = true;
                }
                Ok(())
            }
            Node::Alt(branches) => {
                let enclosed = std::mem::replace(&mut self.enclosed, true);
                for branch in branches {
                    self.node(branch, None)?;
                }
                self.enclosed = enclosed;
                // The `regex` module merges alternatives of one character into
                // one class, and a merged class with two negated characters
                // matches as if both were negated in one: `[^a]|[^b]` matches
                // neither `a` nor `b`.
                if branches.iter().filter(|b| b.negated_char()).count() > 1 {
                    return Err(
                        "alternatives that can end in a negated class of one character".into(),
                    );
                }
                Ok(())
            }
            Node::Cat(nodes) => {
                for (i, node) in nodes.iter().enumerate() {
                    self.node(node, Some((nodes, i)))?;
                }
                Ok(())
            }
            Node::Repeat {
                node: body,
                min,
                max,
                kind,
            } => {
                let enclosed = std::mem::replace(&mut self.enclosed, true);
                self.node(body, None)?;
                self.enclosed = enclosed;
                if *kind == Quantifier::Possessive {
                    return Err("possessive quantifier".into());
                }
                let optional = (*min, *max, *kind) == (0, Some(1), Quantifier::Greedy);
                match body.as_ref() {
                    Node::One(_) => {}
                    // The `regex` module does not retry a repeat's body at a
                    // position where it failed, even when the groups a
                    // backreference in it reads have changed since.
                    body if optional && body.has_backref() => {
                        return Err("backreference inside an optional group".into())
                    }
                    Node::Group(..) if optional => {}
                    _ => return Err("repeat of a group or an assertion".into()),
                }
                // In an open or close pattern, partial searches match the
                // module's for `*?` in the matcher's own positions, and for a
                // lazy repeat without a maximum before a literal, outside
                // alternatives, repeats and lookaheads (its tail is checked
                // once compiled). With a maximum, the module's search for a
                // literal that follows skips starts by the maximum's width.
                let plain = (*min, *max) == (0, None)
                    && items.is_some_and(|(items, i)| plain_lazy(items, i));
                if *kind == Quantifier::Lazy
                    && self.role == Role::Delimiter
                    && !plain
                    && (self.enclosed || max.is_some())
                {
                    return Err(LAZY_POSITION.into());
                }
                Ok(())
            }
            Node::Backref(group) => {
                if self.role == Role::Delimiter {
                    return Err("backreference in an open or close pattern".into());
                }
                if self.ahead {
                    return Err("backreference in a lookahead".into());
                }
                if !self.closed.get(*group).copied().unwrap_or(false) {
                    return Err("backreference to a later group".into());
                }
                Ok(())
            }
            Node::Other(kind, _) => Err(kind.name().into()),
        }
    }
}
