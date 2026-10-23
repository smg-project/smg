//! The pre-tokenizer patterns of byte-level BPE tokenizers, matched natively.
//!
//! The patterns in use (GPT-2, Qwen, GLM, DeepSeek, MiniMax and their kin)
//! draw on a small regex subset: alternation, concatenation, character
//! classes built from Unicode categories, `\s`, ranges and literals, greedy
//! `?` `*` `+` `{m,n}`, a case-insensitive group of literal alternatives for
//! the English clitics, and the lookahead `(?!\S)`. This module parses that
//! subset and matches it the way the engine behind `tokenizers`' `Split`
//! does: leftmost-first alternation, greedy quantifiers that back off one
//! repetition at a time. The engine also yields empty matches, and `Split`
//! cuts the text at them; this matcher does not reproduce that, so a pattern
//! that can match the empty string is declined like everything else outside
//! the subset, and the caller keeps the `tokenizers` pipeline for it.
//!
//! Character categories come from [`super::unicode`], generated from that
//! same engine, so `\p{L}` or `\s` cannot mean one thing here and another
//! there.

use super::unicode;

/// A set of characters: Unicode categories, code point ranges, negation.
#[derive(Debug, Clone, Default)]
struct Class {
    negated: bool,
    categories: u16,
    ranges: Vec<(u32, u32)>,
    /// Membership of U+0000..=U+007F, resolved once.
    ascii: AsciiSet,
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

/// Pairs of ASCII letters that are the full case folding of one character
/// (ß ẞ: ss; ﬀ: ff; ﬁ: fi; ﬂ: fl; ﬃ ﬄ: ffi, ffl; ﬅ ﬆ: st), which the engine
/// matches under `(?i:)` and this matcher does not.
const MULTI_CHARACTER_FOLDS: [[char; 2]; 5] =
    [['s', 's'], ['s', 't'], ['f', 'f'], ['f', 'i'], ['f', 'l']];

const CATEGORY_TABLES: [(u16, &[(u32, u32)]); 11] = [
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
];

#[inline]
fn in_ranges(ranges: &[(u32, u32)], code: u32) -> bool {
    let index = ranges.partition_point(|&(_, end)| end < code);
    ranges.get(index).is_some_and(|&(start, _)| start <= code)
}

impl Class {
    fn finish(mut self) -> Self {
        let mut ascii = AsciiSet::default();
        for code in 0u32..128 {
            if self.contains_slow(code) {
                ascii.insert(code);
            }
        }
        self.ascii = ascii;
        self
    }

    fn contains_slow(&self, code: u32) -> bool {
        let mut hit = in_ranges(&self.ranges, code);
        if !hit && self.categories != 0 {
            hit = CATEGORY_TABLES
                .iter()
                .any(|&(bit, table)| self.categories & bit != 0 && in_ranges(table, code));
        }
        hit != self.negated
    }

    #[inline]
    fn contains(&self, c: char) -> bool {
        let code = u32::from(c);
        if code < 128 {
            self.ascii.contains(code)
        } else {
            self.contains_slow(code)
        }
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
enum Node {
    /// One character of the class.
    Class(usize),
    /// Greedy repetition of one class character, `min..=max` times.
    ClassRepeat {
        class: usize,
        min: u32,
        max: u32,
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
}

/// The characters a match of a node can begin with: a table of the ASCII
/// code points, one bit for every character beyond ASCII (set when a class
/// may hold any of them), and whether the node can match without consuming a
/// character, in which case any character may follow it.
#[derive(Debug, Clone, Copy, Default)]
struct First {
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
    /// fails it gives back one character at a time, down to `min`.
    Run { class: usize, min: u32, max: u32 },
    /// Succeeds when the next character is not in the class.
    NotAhead(usize),
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

/// Choices a match can hold at once: one per alternation and per run on the
/// deepest path of the tree (four for the GPT-2 and Qwen patterns, six for the
/// largest in use). A pattern that needs more is declined.
const MAX_CHOICES: usize = 32;

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    classes: Vec<Class>,
    case_insensitive: bool,
}

impl Pattern {
    /// `None` when the pattern uses anything outside the supported subset.
    pub(crate) fn parse(pattern: &str) -> Option<Self> {
        let mut parser = Parser {
            chars: pattern.chars().peekable(),
            classes: Vec::new(),
            case_insensitive: false,
        };
        let mut root = parser.parse_alternation()?;
        if parser.chars.next().is_some() {
            return None;
        }
        if can_match_empty(&root) {
            return None;
        }
        first_sets(&mut root, &parser.classes);
        if choices(&root) > MAX_CHOICES {
            return None;
        }
        let mut program = Vec::new();
        let mut firsts = Vec::new();
        compile(&root, &mut program, &mut firsts);
        program.push(Op::Match);
        Some(Self {
            classes: parser.classes,
            program,
            firsts,
        })
    }

    /// The matches in `text`, leftmost first, each continuing after the last.
    pub(crate) fn find_iter<'p, 't>(&'p self, text: &'t str) -> Matches<'p, 't> {
        Matches {
            pattern: self,
            text,
            pos: 0,
            choices: [Choice::default(); MAX_CHOICES],
        }
    }

    /// The end of the match that starts exactly at `pos`, if any: the program
    /// run from its first op, alternatives in order and runs longest first,
    /// the first path that reaches `Match` winning, as the engine does it.
    fn match_at(
        &self,
        text: &str,
        pos: usize,
        choices: &mut [Choice; MAX_CHOICES],
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
                Op::Run { class, min, max } => {
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
                        choices[depth] = Choice { pc, pos: p, count };
                        depth += 1;
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

/// Appends the ops of `node` to `program`; the first-character sets the
/// splits index go to `firsts`. Alternatives are laid out in order, each
/// ending in a jump past the alternation, so that the program tries them the
/// way the tree matcher did.
fn compile(node: &Node, program: &mut Vec<Op>, firsts: &mut Vec<First>) {
    match node {
        Node::Class(class) => program.push(Op::Char(*class)),
        Node::ClassRepeat { class, min, max } => program.push(Op::Run {
            class: *class,
            min: *min,
            max: *max,
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
    }
}

/// The most choices a match of `node` can hold at once: every run and every
/// alternation on a path leaves one until the path is done or backs off.
fn choices(node: &Node) -> usize {
    match node {
        Node::Class(_) | Node::NotAhead(_) => 0,
        Node::ClassRepeat { .. } => 1,
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
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        while self.pos < self.text.len() {
            let start = self.pos;
            match self.pattern.match_at(self.text, start, &mut self.choices) {
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
        Node::NotAhead(_) => true,
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
        Node::NotAhead(_) => First {
            nullable: true,
            ..First::default()
        },
    }
}

#[inline]
fn next_char(text: &str, pos: usize) -> Option<(char, usize)> {
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
        let mut previous_letter = None;
        while let Some(&c) = self.chars.peek() {
            if c == '|' || c == ')' {
                break;
            }
            nodes.push(self.parse_quantified()?);
            // Under `(?i:)` the engine also matches a run of letters against
            // the one character whose full case folding is that run (ß and ẞ
            // for "ss", ﬁ for "fi", ﬆ for "st", ...), which a matcher that
            // folds letter by letter cannot do: such runs are declined.
            let letter = (self.case_insensitive
                && c.is_ascii_alphabetic()
                && matches!(nodes.last(), Some(Node::Class(_))))
            .then(|| c.to_ascii_lowercase());
            if let (Some(first), Some(second)) = (previous_letter, letter) {
                if MULTI_CHARACTER_FOLDS.contains(&[first, second]) {
                    return None;
                }
            }
            previous_letter = letter;
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
                return self.quantified(atom, min, max);
            }
            _ => return Some(atom),
        };
        self.chars.next();
        self.quantified(atom, min, max)
    }

    fn quantified(&mut self, atom: Node, min: u32, max: u32) -> Option<Node> {
        // Lazy (`??`, `*?`) and possessive (`?+`, `*+`) forms are not supported.
        if matches!(self.chars.peek(), Some('?' | '+')) {
            return None;
        }
        Some(match atom {
            Node::Class(class) => Node::ClassRepeat { class, min, max },
            Node::NotAhead(_) => return None,
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
        // folded the way the engine folds them, letter by letter; a class, an
        // escape or a non-ASCII literal there would match differently, so it
        // is declined (so are lookaheads and the letter runs the engine folds
        // as one character; see `parse_group` and `parse_concat`).
        if self.case_insensitive && (c == '[' || c == '\\' || !c.is_ascii()) {
            return None;
        }
        match c {
            '(' => self.parse_group(),
            '[' => {
                let class = self.parse_class_body()?;
                Some(Node::Class(self.add_class(class)))
            }
            '\\' => {
                let class = self.parse_escape_as_class()?;
                Some(Node::Class(self.add_class(class)))
            }
            '.' | '^' | '$' | '*' | '+' | '?' | '{' | '}' | ')' | '|' | ']' => None,
            literal => Some(Node::Class(self.literal_class(literal))),
        }
    }

    fn parse_group(&mut self) -> Option<Node> {
        if self.chars.next()? != '?' {
            return None;
        }
        match self.chars.next()? {
            ':' => {
                let inner = self.parse_alternation()?;
                (self.chars.next()? == ')').then_some(inner)
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
            _ => None,
        }
    }

    /// A literal character as a class: itself, or under `(?i:)` every
    /// character the engine folds together with it.
    fn literal_class(&mut self, c: char) -> usize {
        let mut class = Class::default();
        if self.case_insensitive && c.is_ascii_alphabetic() {
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
            class.ranges.sort_unstable();
        } else {
            class.ranges.push((u32::from(c), u32::from(c)));
        }
        self.add_class(class)
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
            's' => class.categories |= CAT_SPACE,
            'S' => {
                class.categories |= CAT_SPACE;
                class.negated = true;
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
            _ => return None,
        })
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
                    's' => {
                        class.categories |= CAT_SPACE;
                        first = false;
                        continue;
                    }
                    'P' | 'S' | 'd' | 'D' | 'w' | 'W' | 'h' | 'H' => return None,
                    other => escaped_literal(other)?,
                },
                // Nested classes, intersections and POSIX brackets are not supported.
                '[' => return None,
                '&' if self.chars.peek() == Some(&'&') => return None,
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
        class.ranges.sort_unstable();
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
    use tokenizers::utils::SysRegex;

    use super::*;

    const GPT2: &str =
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DIGITS: &str = r"\p{N}{1,3}";
    const CJK: &str = "[一-龥぀-ゟ゠-ヿ]+";
    const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    const PATTERNS: [&str; 8] = [GPT2, QWEN2, QWEN35, CL100K, DIGITS, CJK, DEEPSEEK, CASED];

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
            r"a$",
            r"(a)",
            r"a*?",
            r"a++",
            r"[[:alpha:]]",
            r"[a&&b]",
            r"\d+",
            r"(?=a)",
            r"\p{Han}",
            r"a{3,1}",
            r"[z-a]",
            r"(?:ab)+",
            r"(?: ?\p{L})*",
            r"(?:ab){2}",
            r"(?i:[sdmt]|ll|ve|re)",
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
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
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
            (r"(?i:[sdmt]|ll|ve|re)", "'S 'LL"),
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
                    r"(?:a|b){2,}",
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
}
