//! The part of Python's `regex` module that response templates use.
//! transformers compiles every pattern with `regex.DOTALL` and calls `search`,
//! `search(..., partial=True)` and `finditer`. A pattern is parsed here into a
//! small program for a backtracking matcher that returns what the `regex`
//! module returns: the leftmost match in backtracking order, its named groups,
//! and the start of a partial match. The syntax accepted is what the templates
//! of transformers' tests and of `transformers serve` and the public
//! checkpoint shapes the parity fixtures record use, plus positive classes,
//! ranges in classes and `\w` and `\s` in classes (listed in the README);
//! anything else is refused with the construct's name.

use crate::error::Unsupported;

/// `\w` of the `regex` module: Rust's word characters and those its newer
/// Unicode tables add.
pub(crate) fn is_word(c: char) -> bool {
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

/// One character: a literal, `.`, `\w`, `\s`, or a class of literals,
/// ranges of them, `\w` and `\s`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Set {
    Char(char),
    Any,
    Word,
    Space,
    Class {
        negated: bool,
        ranges: Box<[(char, char)]>,
        word: bool,
        space: bool,
    },
}

impl Set {
    fn matches(&self, c: char) -> bool {
        match self {
            Self::Char(x) => *x == c,
            Self::Any => true,
            Self::Word => is_word(c),
            Self::Space => c.is_whitespace(),
            Self::Class {
                negated,
                ranges,
                word,
                space,
            } => {
                (ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&c))
                    || (*word && is_word(c))
                    || (*space && c.is_whitespace()))
                    != *negated
            }
        }
    }
}

/// `^` (start of text), `$` (end, or before a final newline), `\Z`, `\b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Look {
    Start,
    Dollar,
    End,
    WordBoundary,
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rep {
    Star,
    Plus,
    Opt,
    LazyStar,
}

#[derive(Debug)]
enum Node {
    One(Set),
    Look(Look),
    Group(Option<usize>, Box<Node>),
    Alt(Vec<Node>),
    Cat(Vec<Node>),
    /// `?` after one character or a group.
    Opt(Box<Node>),
    /// `*`, `+` or `*?` after one character.
    Repeat(Set, Rep),
    Backref(usize),
}

impl Node {
    /// Whether the `regex` module can reduce this alternative to a negated
    /// class of one character (after moving a common prefix out of the
    /// alternatives, merging nested ones and dropping empty groups). A class
    /// with `\w` or `\s` is not such a class.
    fn negated_char(&self) -> bool {
        match self {
            Self::One(Set::Class {
                negated,
                ranges,
                word,
                space,
            }) => *negated && !*word && !*space && ranges.len() == 1 && ranges[0].0 == ranges[0].1,
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
    /// `(?:(?:))?`), which the `regex` module drops when it parses.
    fn empty_group(&self) -> bool {
        match self {
            Self::Group(None, node) | Self::Opt(node) => node.empty_group(),
            Self::Cat(nodes) => nodes.iter().all(Self::empty_group),
            _ => false,
        }
    }

    fn has_backref(&self) -> bool {
        match self {
            Self::Backref(_) => true,
            Self::Group(_, node) | Self::Opt(node) => node.has_backref(),
            Self::Alt(nodes) | Self::Cat(nodes) => nodes.iter().any(Self::has_backref),
            _ => false,
        }
    }

    fn nullable(&self) -> bool {
        match self {
            Self::One(_) => false,
            Self::Look(_) | Self::Backref(_) => true,
            Self::Group(_, node) => node.nullable(),
            Self::Repeat(_, rep) => *rep != Rep::Plus,
            Self::Opt(_) => true,
            Self::Alt(nodes) => nodes.iter().any(Self::nullable),
            Self::Cat(nodes) => nodes.iter().all(Self::nullable),
        }
    }
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
    Look(Look),
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
    /// A literal every match starts with, to skip ahead to.
    first: Option<char>,
    /// A literal every match contains: without it there is no complete match.
    required: Option<String>,
    /// The number of repeats and branch points; the positions a search
    /// entered them at are remembered (sound without backreferences, which
    /// then set this to 0).
    memos: usize,
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

impl Pattern {
    /// Compile a Python pattern, or name the construct that is not ported.
    /// A `finditer` pattern must not match the empty string (see the README).
    pub(crate) fn new(source: &str, role: Role) -> Result<Self, Unsupported> {
        let mut parser = Parser {
            chars: source.chars().collect(),
            i: 0,
            role,
            groups: 0,
            names: Vec::new(),
            closed: vec![false],
            optional: vec![false],
        };
        let node = parser.alt().map_err(Unsupported::Regex)?;
        if parser.i != parser.chars.len() {
            return Err(Unsupported::Regex("unbalanced parenthesis".into()));
        }
        if role == Role::Finditer && node.nullable() {
            return Err(Unsupported::EmptyMatch);
        }
        Ok(Self::build(&node, parser.names, parser.optional))
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
            ..Self::build(&node, Vec::new(), vec![false])
        }
    }

    fn build(node: &Node, names: Vec<(String, usize)>, optional: Vec<bool>) -> Self {
        let mut prog = Vec::new();
        compile(node, &mut prog);
        prog.push(Inst::Match);
        let first = match prog.first() {
            Some(Inst::One(Set::Char(c))) => Some(*c),
            _ => None,
        };
        let (mut run, mut longest) = (String::new(), String::new());
        required_literal(node, &mut run, &mut longest);
        let memos = if node.has_backref() {
            0
        } else {
            next_memo(&prog)
        };
        Self {
            prog,
            names,
            optional,
            nullable: node.nullable(),
            first,
            required: (longest.len() > 1).then_some(longest),
            memos,
        }
    }

    /// Whether the pattern can match the empty string.
    pub(crate) fn nullable(&self) -> bool {
        self.nullable
    }

    /// The names of the named groups (`pattern.groupindex`).
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
            if let Some(c) = self.first {
                start += hay[start..].find(c)?;
            }
            if partial && start == hay.len() {
                return None;
            }
            match self.run(hay, start, partial, &mut run) {
                Step::Fail => {}
                step => return Some((start, step)),
            }
            start += hay[start..].chars().next()?.len_utf8();
        }
    }

    /// One backtracking run from `start`. In a partial search, reading past
    /// the end of the haystack is a partial match, except when a lazy repeat
    /// would extend there; entering a lazy repeat at the end is one too.
    fn run(&self, hay: &str, start: usize, partial: bool, run: &mut Run) -> Step {
        run.stack.push(Frame::Try(0, start));
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
                        if end > at {
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
                    Inst::Look(look) if look.at(hay, at) => pc += 1,
                    Inst::Look(_) => break,
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
}

#[derive(Debug)]
enum Step {
    Match(usize, Vec<Option<usize>>),
    Partial,
    Fail,
}

enum Frame {
    Try(usize, usize),
    Restore(usize, Option<usize>),
    GiveBack { pc: usize, floor: usize, at: usize },
    Extend { pc: usize, at: usize },
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

/// The longest run of literal characters in `node` that every match contains.
fn required_literal(node: &Node, run: &mut String, longest: &mut String) {
    match node {
        Node::One(Set::Char(c)) => run.push(*c),
        Node::Look(_) => {}
        Node::Group(_, inner) => required_literal(inner, run, longest),
        Node::Cat(nodes) => nodes.iter().for_each(|n| required_literal(n, run, longest)),
        _ => {
            if run.chars().count() > longest.chars().count() {
                longest.clone_from(run);
            }
            run.clear();
        }
    }
    if run.chars().count() > longest.chars().count() {
        longest.clone_from(run);
    }
}

/// The number of the next repeat or branch point in `prog`.
fn next_memo(prog: &[Inst]) -> usize {
    prog.iter()
        .filter(|i| matches!(i, Inst::Star { .. } | Inst::Split { .. }))
        .count()
}

/// A branch point before the next instruction; [`set_second`] sets the other.
fn split(prog: &mut Vec<Inst>) -> usize {
    let memo = next_memo(prog);
    prog.push(Inst::Split { second: 0, memo });
    prog.len() - 1
}

fn set_second(prog: &mut [Inst], split: usize) {
    let end = prog.len();
    if let Inst::Split { second, .. } = &mut prog[split] {
        *second = end;
    }
}

fn compile(node: &Node, prog: &mut Vec<Inst>) {
    match node {
        Node::One(set) => prog.push(Inst::One(set.clone())),
        Node::Look(look) => prog.push(Inst::Look(*look)),
        Node::Backref(group) => prog.push(Inst::Backref(*group)),
        Node::Group(group, inner) => {
            if let Some(g) = group {
                prog.push(Inst::Save(2 * g));
            }
            compile(inner, prog);
            if let Some(g) = group {
                prog.push(Inst::Save(2 * g + 1));
            }
        }
        Node::Cat(nodes) => nodes.iter().for_each(|n| compile(n, prog)),
        Node::Alt(branches) => {
            let mut jumps = Vec::new();
            for (k, branch) in branches.iter().enumerate() {
                if k + 1 == branches.len() {
                    compile(branch, prog);
                    break;
                }
                let at = split(prog);
                compile(branch, prog);
                jumps.push(prog.len());
                prog.push(Inst::Jmp(0));
                set_second(prog, at);
            }
            let end = prog.len();
            for j in jumps {
                prog[j] = Inst::Jmp(end);
            }
        }
        Node::Opt(body) => {
            let at = split(prog);
            compile(body, prog);
            set_second(prog, at);
        }
        Node::Repeat(set, rep) => {
            if *rep == Rep::Plus {
                prog.push(Inst::One(set.clone()));
            }
            let memo = next_memo(prog);
            prog.push(Inst::Star {
                set: set.clone(),
                greedy: *rep != Rep::LazyStar,
                memo,
            });
        }
    }
}

/// A parser for the supported subset of the `regex` module's syntax. Errors
/// name the construct that is not supported.
struct Parser {
    chars: Vec<char>,
    i: usize,
    role: Role,
    groups: usize,
    names: Vec<(String, usize)>,
    closed: Vec<bool>,
    optional: Vec<bool>,
}

type Parsed = Result<Node, String>;

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        let found = self.peek() == Some(c);
        self.i += usize::from(found);
        found
    }

    /// A literal character at `k` that no quantifier follows.
    fn literal_at(&self, k: usize) -> bool {
        let next = match self.chars.get(k..) {
            Some(['\\', c, rest @ ..]) if c.is_ascii_punctuation() || *c == 'n' => rest,
            Some([c, rest @ ..]) if !".^$*+?{}[]\\|()".contains(*c) => rest,
            _ => return false,
        };
        !matches!(next.first(), Some('*' | '+' | '?' | '{'))
    }

    fn rest_starts_with(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(k, c)| self.chars.get(self.i + k) == Some(&c))
    }

    /// Groups from number `first` on can be left unset by a match.
    fn mark_optional(&mut self, first: usize) {
        for flag in &mut self.optional[first + 1..] {
            *flag = true;
        }
    }

    fn alt(&mut self) -> Parsed {
        let first = self.groups;
        let mut branches = vec![self.seq()?];
        while self.eat('|') {
            branches.push(self.seq()?);
        }
        if branches.len() == 1 {
            return Ok(branches.remove(0));
        }
        // The `regex` module merges alternatives of one character into one
        // class, and a merged class with two negated characters matches as if
        // both were negated in one: `[^a]|[^b]` matches neither `a` nor `b`.
        if branches.iter().filter(|b| b.negated_char()).count() > 1 {
            return Err("alternatives that can end in a negated class of one character".into());
        }
        self.mark_optional(first);
        Ok(Node::Alt(branches))
    }

    fn seq(&mut self) -> Parsed {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let first = self.groups;
            let atom = self.atom()?;
            let rep = match self.peek() {
                Some('*') => Rep::Star,
                Some('+') => Rep::Plus,
                Some('?') => Rep::Opt,
                _ => {
                    items.push(atom);
                    continue;
                }
            };
            self.i += 1;
            let rep = match (rep, self.eat('?')) {
                (rep, false) => rep,
                (Rep::Star, true) => Rep::LazyStar,
                _ => return Err("lazy quantifier other than *?".into()),
            };
            if matches!(self.peek(), Some('*' | '+' | '?' | '{')) {
                return Err("repeated or possessive quantifier".into());
            }
            let node = match (atom, rep) {
                (Node::One(set), Rep::Star | Rep::Plus | Rep::LazyStar) => Node::Repeat(set, rep),
                // The `regex` module does not retry a repeat's body at a
                // position where it failed, even when the groups a
                // backreference in it reads have changed since.
                (atom, Rep::Opt) if atom.has_backref() => {
                    return Err("backreference inside an optional group".into())
                }
                (atom @ (Node::One(_) | Node::Group(..)), Rep::Opt) => Node::Opt(Box::new(atom)),
                _ => return Err("repeat of a group or an assertion".into()),
            };
            // Partial searches match the `regex` module for `.*?` before a
            // literal, and for a lazy repeat after another item and before `\b`
            // and a literal (see the README).
            if rep == Rep::LazyStar && self.role == Role::Delimiter {
                let dot_literal =
                    matches!(node, Node::Repeat(Set::Any, _)) && self.literal_at(self.i);
                let boundary = !items.is_empty()
                    && self.rest_starts_with("\\b")
                    && self.literal_at(self.i + 2);
                if !dot_literal && !boundary {
                    return Err("lazy repeat in this position of an open or close pattern".into());
                }
            }
            if rep == Rep::Opt {
                self.mark_optional(first);
            }
            items.push(node);
        }
        Ok(if items.len() == 1 {
            items.remove(0)
        } else {
            Node::Cat(items)
        })
    }

    fn atom(&mut self) -> Parsed {
        let c = self.peek().unwrap_or_default();
        self.i += 1;
        Ok(match c {
            '.' => Node::One(Set::Any),
            '^' => Node::Look(Look::Start),
            '$' => Node::Look(Look::Dollar),
            '\\' => return self.escape(false),
            '[' => return self.class(),
            '(' => return self.group(),
            '*' | '+' | '?' | '{' | '}' | ']' => return Err(format!("'{c}' without an atom")),
            c => Node::One(Set::Char(c)),
        })
    }

    fn escape(&mut self, in_class: bool) -> Parsed {
        let c = self.peek().ok_or("trailing backslash")?;
        self.i += 1;
        Ok(match c {
            c if c.is_ascii_punctuation() => Node::One(Set::Char(c)),
            'n' => Node::One(Set::Char('\n')),
            'w' => Node::One(Set::Word),
            's' => Node::One(Set::Space),
            'b' if !in_class => Node::Look(Look::WordBoundary),
            'Z' if !in_class => Node::Look(Look::End),
            '1'..='9' if !in_class && !self.peek().is_some_and(|d| d.is_ascii_digit()) => {
                let group = c as usize - '0' as usize;
                if self.role == Role::Delimiter {
                    return Err("backreference in an open or close pattern".into());
                }
                if !self.closed.get(group).copied().unwrap_or(false) {
                    return Err("backreference to a later group".into());
                }
                Node::Backref(group)
            }
            c => return Err(format!("escape \\{c}")),
        })
    }

    /// `[...]` or `[^...]` of literal characters, ranges of them, `\w` and
    /// `\s`. A `-` is literal only as the last item; one next to `\w` or `\s`
    /// elsewhere is refused (the `regex` module reads it as a literal, Python's
    /// `re` rejects it).
    fn class(&mut self) -> Parsed {
        let negated = self.eat('^');
        let mut ranges = Vec::new();
        let (mut word, mut space) = (false, false);
        loop {
            let c = self.peek().ok_or("unterminated character class")?;
            self.i += 1;
            let empty = ranges.is_empty() && !word && !space;
            let lo = match c {
                ']' if !empty => break,
                '-' if self.peek() == Some(']') && !empty => '-',
                c => match self.class_item(c)? {
                    Set::Char(lo) => lo,
                    set => {
                        if self.range_follows() {
                            let e = self.chars[self.i - 1];
                            return Err(format!("'-' after \\{e} in a character class"));
                        }
                        word |= set == Set::Word;
                        space |= set == Set::Space;
                        continue;
                    }
                },
            };
            let mut hi = lo;
            if self.range_follows() {
                let c = self.chars[self.i + 1];
                self.i += 2;
                hi = match self.class_item(c)? {
                    Set::Char(hi) => hi,
                    _ => {
                        let e = self.chars[self.i - 1];
                        return Err(format!("'-' before \\{e} in a character class"));
                    }
                };
                if hi < lo {
                    return Err("character range out of order".into());
                }
            }
            ranges.push((lo, hi));
        }
        Ok(Node::One(Set::Class {
            negated,
            ranges: ranges.into(),
            word,
            space,
        }))
    }

    /// A `-` that makes a range (not one before the closing `]`) comes next.
    fn range_follows(&self) -> bool {
        self.peek() == Some('-') && !matches!(self.chars.get(self.i + 1), Some(']') | None)
    }

    /// One item of a class: a literal character, `\w` or `\s`.
    fn class_item(&mut self, c: char) -> Result<Set, String> {
        match c {
            '\\' => match self.escape(true)? {
                Node::One(set) => Ok(set),
                _ => Err("character class".into()),
            },
            '[' | ']' | '-' | '^' | '&' | '|' | '~' => Err(format!("'{c}' in a character class")),
            c => Ok(Set::Char(c)),
        }
    }

    fn group(&mut self) -> Parsed {
        let group = if self.rest_starts_with("?:") {
            self.i += 2;
            None
        } else if self.rest_starts_with("?P<") {
            self.i += 3;
            let end = self.chars[self.i..]
                .iter()
                .position(|&c| c == '>')
                .ok_or("group name")?;
            let name: String = self.chars[self.i..self.i + end].iter().collect();
            self.i += end + 1;
            let valid = name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !valid || self.names.iter().any(|(n, _)| *n == name) {
                return Err(format!("group name {name:?}"));
            }
            self.groups += 1;
            self.names.push((name, self.groups));
            self.closed.push(false);
            self.optional.push(false);
            Some(self.groups)
        } else if ["?=", "?!", "?<=", "?<!"]
            .iter()
            .any(|s| self.rest_starts_with(s))
        {
            return Err("look-around".into());
        } else if self.peek() == Some('?') {
            return Err("group syntax".into());
        } else {
            return Err("unnamed capturing group".into());
        };
        let inner = self.alt()?;
        if !self.eat(')') {
            return Err("unbalanced parenthesis".into());
        }
        if let Some(g) = group {
            self.closed[g] = true;
        }
        Ok(Node::Group(group, Box::new(inner)))
    }
}
