//! The matcher: a [`Pattern`] run over text as it arrives, decided as soon as the text decides it;
//! and the same machine run as a search, for where a pattern that must end at a given place starts.

use std::ops::Range;

use super::{Atom, Class, End, Pattern, Slot};

/// The most a matcher holds undecided by default, in bytes: 8 KiB. A pattern whose decision needs
/// more text than this, a header longer than any a template writes, is no match.
pub const HOLD_CAP: usize = 8 * 1024;

/// What the text fed so far says about the pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// More text can still change the answer.
    Pending,
    /// The pattern matches, and no continuation of the text can change the match.
    Matched(Match),
    /// The pattern does not match, whatever follows.
    NoMatch,
}

/// A match, in byte offsets of the text the matcher was fed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    end: usize,
    ending: Option<usize>,
    spans: [Option<(usize, usize)>; SLOTS],
}

impl Match {
    /// Where the match ends. The bytes from here on follow it: they are not part of it, a
    /// lookahead's included.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Which of the matcher's ends closed the match, by its place in the list; `None` for a
    /// matcher given no ends.
    pub fn ending(&self) -> Option<usize> {
        self.ending
    }

    /// The bytes the run captured as `slot`, if a run did.
    pub fn span(&self, slot: Slot) -> Option<Range<usize>> {
        self.spans[slot_index(slot)].map(|(start, end)| start..end)
    }
}

/// A pattern run over text as it arrives.
///
/// The answer is the one Python's `regex` module gives when it matches the pattern, followed by
/// one of its ends, at the start of the whole text: `match` under `DOTALL`, the dialect response
/// templates are written in. That is the first match in backtracking order: a greedy run would
/// rather take one more character and a lazy one would rather stop, an optional literal is taken
/// before it is skipped, and of the ends the first in the list is tried first. The matcher keeps
/// every alternative alive at once, in that order, one thread per place in the pattern (a small
/// Pike VM), so it reads each character once and holds no text but what it is fed.
///
/// It answers as soon as no continuation of the text can change the answer:
///
/// - a match, when a thread has reached the end of the pattern and no thread before it in that
///   order is still alive: `\w+` that has read `get` waits for the character after it, while a
///   pattern that ends in a literal is decided by the literal's last character;
/// - no match, when every thread has died;
/// - `\b` at the edge of the text fed so far waits for the next character, or for the end of the
///   text; it is never read as false for want of one;
/// - past [`HOLD_CAP`] bytes still undecided, or the cap [`hold_cap`](Self::hold_cap) sets, no
///   match.
///
/// Fed past its answer, it keeps the text and gives the same answer.
#[derive(Clone, Debug)]
pub struct Matcher<'p> {
    machine: Machine<'p>,
    cap: Option<usize>,
    /// Every byte fed.
    held: String,
    /// How far into `held` the threads have read.
    at: usize,
    /// The character before `at`, which `\b` reads; at the start, the one before the text.
    before: Option<char>,
    /// The threads alive at `at`, in priority order, each at the instruction it reads next.
    threads: Vec<Thread>,
    /// The best match found so far. Every thread in `threads` comes before it in priority, and
    /// it stands once they have all died.
    found: Option<Match>,
    /// The answer, once there is one: the match, or none.
    answer: Option<Option<Match>>,
    /// Room the threads reuse, so that reading a character allocates nothing.
    spare: Vec<Thread>,
}

/// How many slots a match captures.
const SLOTS: usize = 3;

fn slot_index(slot: Slot) -> usize {
    match slot {
        Slot::Name => 0,
        Slot::Key => 1,
        Slot::Value => 2,
    }
}

/// One instruction of a matcher's program.
#[derive(Clone, Copy, Debug)]
enum Op<'p> {
    /// Read this character.
    Char(char),
    /// Read a character of the class.
    Class(&'p Class),
    /// Go on at both, the first before the second.
    Split(usize, usize),
    /// Go on there.
    Jump(usize),
    /// `\b`.
    Boundary,
    /// A captured run starts.
    Open(usize),
    /// A captured run ends.
    Close(usize),
    /// The match ends here; what follows is looked ahead at.
    Mark,
    /// A match, through the end at this place in the matcher's list, if it has ends.
    Done(Option<usize>),
}

/// One alternative: where it is in the program, where it started, and what it has captured.
#[derive(Clone, Copy, Debug)]
struct Thread {
    pc: usize,
    /// Where its match starts: the start of the text for a matcher, the place it was started at
    /// for a search.
    start: usize,
    /// Where its match ends, once it has passed the mark.
    end: usize,
    /// Where each slot's run started.
    open: [usize; SLOTS],
    spans: [Option<(usize, usize)>; SLOTS],
}

impl Thread {
    /// A thread at the program's start, its match starting at `start`.
    fn at_start(start: usize) -> Self {
        Self {
            pc: 0,
            start,
            end: start,
            open: [start; SLOTS],
            spans: [None; SLOTS],
        }
    }
}

/// What [`Machine::follow`] knows of the next character.
#[derive(Clone, Copy)]
enum Next {
    /// This character, or the end of the text.
    Known(Option<char>),
    /// Not fed yet.
    Unknown,
}

/// A program, the word class its `\b` reads, and the room following its threads reuses, shared by
/// [`Matcher`] and [`Suffix`].
#[derive(Clone, Debug)]
struct Machine<'p> {
    program: Vec<Op<'p>>,
    word: &'p Class,
    /// Where the threads go next, in priority order: filled by [`follow`](Self::follow).
    leaves: Vec<Thread>,
    stack: Vec<Thread>,
    /// The instructions [`follow`](Self::follow) has reached, by its generation.
    visited: Vec<u32>,
    generation: u32,
}

impl<'p> Machine<'p> {
    fn new(program: Vec<Op<'p>>, word: &'p Class) -> Self {
        let visited = vec![0; program.len()];
        Self {
            program,
            word,
            leaves: Vec::new(),
            stack: Vec::new(),
            visited,
            generation: 0,
        }
    }

    /// Follow every thread of `threads` through the instructions that read no character, and leave
    /// in `leaves`, in priority order, where they arrive: at a character to read, at a match, and,
    /// while the next character is unknown, at `\b`, which waits for it. `before` is the character
    /// before `at`, the place the threads stand. An instruction is reached once, by the first
    /// thread to get there: any later one would do the same from there, with less priority.
    fn follow(&mut self, threads: &[Thread], next: Next, before: Option<char>, at: usize) {
        self.generation = match self.generation.checked_add(1) {
            Some(generation) => generation,
            None => {
                self.visited.fill(0);
                1
            }
        };
        self.leaves.clear();
        for &start in threads {
            self.stack.push(start);
            while let Some(mut thread) = self.stack.pop() {
                let pc = thread.pc;
                if self.visited[pc] == self.generation {
                    continue;
                }
                self.visited[pc] = self.generation;
                match self.program[pc] {
                    Op::Char(_) | Op::Class(_) | Op::Done(_) => self.leaves.push(thread),
                    Op::Split(first, second) => {
                        self.stack.push(Thread {
                            pc: second,
                            ..thread
                        });
                        self.stack.push(Thread {
                            pc: first,
                            ..thread
                        });
                    }
                    Op::Jump(to) => self.stack.push(Thread { pc: to, ..thread }),
                    Op::Boundary => match next {
                        Next::Unknown => self.leaves.push(thread),
                        Next::Known(after) => {
                            let word = |c: Option<char>| c.is_some_and(|c| self.word.contains(c));
                            if word(before) != word(after) {
                                thread.pc += 1;
                                self.stack.push(thread);
                            }
                        }
                    },
                    Op::Open(slot) => {
                        thread.open[slot] = at;
                        thread.pc += 1;
                        self.stack.push(thread);
                    }
                    Op::Close(slot) => {
                        thread.spans[slot] = Some((thread.open[slot], at));
                        thread.pc += 1;
                        self.stack.push(thread);
                    }
                    Op::Mark => {
                        thread.end = at;
                        thread.pc += 1;
                        self.stack.push(thread);
                    }
                }
            }
        }
    }

    /// Whether `thread`, at a leaf, reads `next`.
    fn reads(&self, thread: &Thread, next: char) -> bool {
        match self.program[thread.pc] {
            Op::Char(expected) => expected == next,
            Op::Class(class) => class.contains(next),
            _ => false,
        }
    }
}

impl<'p> Matcher<'p> {
    /// A matcher for `pattern` followed by one of `ends` (none: the pattern alone), at the start
    /// of the text it is fed. `word` is the class `\b` reads, and `before` the character just
    /// before the text (`None` at the start of the output), which `\b` at the text's start reads.
    /// It holds at most [`HOLD_CAP`] bytes undecided.
    pub fn new(
        pattern: &'p Pattern,
        ends: &'p [End],
        word: &'p Class,
        before: Option<char>,
    ) -> Self {
        Self {
            machine: Machine::new(compile(pattern, ends), word),
            cap: Some(HOLD_CAP),
            held: String::new(),
            at: 0,
            before,
            threads: vec![Thread::at_start(0)],
            found: None,
            answer: None,
            spare: Vec::new(),
        }
    }

    /// The same matcher holding at most `cap` bytes undecided, or any number with `None`.
    #[must_use]
    pub fn hold_cap(mut self, cap: Option<usize>) -> Self {
        self.cap = cap;
        self
    }

    /// Append the next bytes of the text and say what the text fed so far decides.
    pub fn feed(&mut self, text: &str) -> Verdict {
        self.held.push_str(text);
        if self.answer.is_none() {
            self.answer = self.read();
        }
        self.verdict()
    }

    /// No more text will come: the answer for the text fed.
    pub fn finish(&mut self) -> Verdict {
        if self.answer.is_none() {
            self.answer = Some(self.at_end());
        }
        self.verdict()
    }

    /// Every byte fed so far, decided or not: what a caller gives back when there is no match, and
    /// what follows a match from its [`end`](Match::end) on.
    pub fn held(&self) -> &str {
        &self.held
    }

    /// The bytes fed, taken back.
    pub fn into_held(self) -> String {
        self.held
    }

    fn verdict(&self) -> Verdict {
        match &self.answer {
            None => Verdict::Pending,
            Some(Some(found)) => Verdict::Matched(found.clone()),
            Some(None) => Verdict::NoMatch,
        }
    }

    /// Read the held text from where the threads stand, and the answer, if it is decided.
    fn read(&mut self) -> Option<Option<Match>> {
        while let Some(next) = self.held[self.at..].chars().next() {
            if let Some(answer) = self.step(next) {
                return Some(answer);
            }
            if self.cap.is_some_and(|cap| self.at > cap) {
                return Some(None);
            }
        }
        self.at_edge()
    }

    /// Read `next`: the answer, if the threads that read it leave one, or `None` once they have
    /// moved past it.
    fn step(&mut self, next: char) -> Option<Option<Match>> {
        self.machine
            .follow(&self.threads, Next::Known(Some(next)), self.before, self.at);
        let mut alive = std::mem::take(&mut self.spare);
        alive.clear();
        for thread in &self.machine.leaves {
            if let Op::Done(ending) = self.machine.program[thread.pc] {
                // Every thread after this one comes after its match in priority: they are cut.
                self.found = Some(matched(thread, ending));
                break;
            }
            if self.machine.reads(thread, next) {
                alive.push(Thread {
                    pc: thread.pc + 1,
                    ..*thread
                });
            }
        }
        if alive.is_empty() {
            self.spare = alive;
            return Some(self.found.take());
        }
        self.spare = std::mem::replace(&mut self.threads, alive);
        self.at += next.len_utf8();
        self.before = Some(next);
        None
    }

    /// The answer at the edge of the text fed, if no next character can change it: a thread
    /// that has reached the end of the pattern, with none before it still alive.
    fn at_edge(&mut self) -> Option<Option<Match>> {
        self.machine
            .follow(&self.threads, Next::Unknown, self.before, self.at);
        match self.machine.leaves.first() {
            Some(thread) => match self.machine.program[thread.pc] {
                Op::Done(ending) => Some(Some(matched(thread, ending))),
                _ => None,
            },
            None => Some(self.found.take()),
        }
    }

    /// The answer at the end of the text: the first thread to reach the end of the pattern, or
    /// the match found before.
    fn at_end(&mut self) -> Option<Match> {
        self.machine
            .follow(&self.threads, Next::Known(None), self.before, self.at);
        let done =
            self.machine
                .leaves
                .iter()
                .find_map(|thread| match self.machine.program[thread.pc] {
                    Op::Done(ending) => Some(matched(thread, ending)),
                    _ => None,
                });
        done.or_else(|| self.found.take())
    }
}

fn matched(thread: &Thread, ending: Option<usize>) -> Match {
    Match {
        end: thread.end,
        ending,
        spans: thread.spans,
    }
}

/// A pattern searched for by where its match ends.
///
/// Fed a text as it arrives, a search starts the pattern at every place of it, and says, for the
/// text fed so far, where the earliest match that ends exactly at its end starts, given the
/// character that comes next, which a `\b` at the end reads. That is the start Python's `regex`
/// module gives when it searches the text for the pattern followed by what comes next (under
/// `DOTALL`, with that next part matched where the text ends): a region whose closing pattern is
/// text before a marker (`</function>\s*` before `</tool_call>`) closes at the marker only where
/// that text ends right at it, and the search says where it began. Of two matches ending there the
/// one that starts first wins, as a search tries the places in order; which of the pattern's
/// alternatives took the bytes does not matter, so nothing is captured.
///
/// Every thread keeps the place it started at. Of two threads at one place in the pattern only
/// the one that started first is kept, since they would go on alike: so the search reads each
/// character once and holds one thread per place in the pattern, whatever the text's length. A
/// match longer than the cap, [`HOLD_CAP`] by default, is none: its threads are dropped once the
/// text has run past it.
#[derive(Clone, Debug)]
pub struct Suffix<'p> {
    machine: Machine<'p>,
    cap: Option<usize>,
    /// Bytes read so far.
    at: usize,
    /// The character before `at`; at the start, the one before the text.
    before: Option<char>,
    /// The threads alive at `at`, the earliest started first.
    threads: Vec<Thread>,
    /// Room the threads reuse.
    spare: Vec<Thread>,
}

impl<'p> Suffix<'p> {
    /// A search for `pattern` over a text whose first byte follows `before` (`None` at the start
    /// of the output), with `word` the class `\b` reads. A match is at most [`HOLD_CAP`] bytes.
    pub fn new(pattern: &'p Pattern, word: &'p Class, before: Option<char>) -> Self {
        Self {
            machine: Machine::new(compile(pattern, &[]), word),
            cap: Some(HOLD_CAP),
            at: 0,
            before,
            threads: Vec::new(),
            spare: Vec::new(),
        }
    }

    /// The same search, its matches at most `cap` bytes, or of any length with `None`.
    #[must_use]
    pub fn hold_cap(mut self, cap: Option<usize>) -> Self {
        self.cap = cap;
        self
    }

    /// Append the next bytes of the text.
    pub fn feed(&mut self, text: &str) {
        for next in text.chars() {
            self.threads.push(Thread::at_start(self.at));
            self.machine
                .follow(&self.threads, Next::Known(Some(next)), self.before, self.at);
            let mut alive = std::mem::take(&mut self.spare);
            alive.clear();
            let read = self.at + next.len_utf8();
            let oldest = self.cap.map_or(0, |cap| read.saturating_sub(cap));
            for thread in &self.machine.leaves {
                // A thread at a match has nothing more to read; one that would make the match
                // longer than the cap is no match.
                if thread.start >= oldest && self.machine.reads(thread, next) {
                    alive.push(Thread {
                        pc: thread.pc + 1,
                        ..*thread
                    });
                }
            }
            self.spare = std::mem::replace(&mut self.threads, alive);
            self.at = read;
            self.before = Some(next);
        }
    }

    /// Where the earliest match that ends at the end of the text fed so far starts, in bytes of
    /// that text, when `next` comes after it (`None`: the text ends there). A search may be asked
    /// again after more bytes.
    pub fn start(&mut self, next: Option<char>) -> Option<usize> {
        let mut threads = std::mem::take(&mut self.spare);
        threads.clear();
        threads.extend_from_slice(&self.threads);
        threads.push(Thread::at_start(self.at));
        self.machine
            .follow(&threads, Next::Known(next), self.before, self.at);
        self.spare = threads;
        self.machine
            .leaves
            .iter()
            .find(|thread| matches!(self.machine.program[thread.pc], Op::Done(_)))
            .map(|thread| thread.start)
    }
}

/// Whether some way through `pattern` reads every character of `text`, priorities aside: from the
/// pattern's start, or, `anywhere`, from any place inside it, as a match that began before the
/// text would. `before` is the character before the text, which a `\b` reads.
pub(crate) fn reads_through(
    pattern: &Pattern,
    word: &Class,
    before: Option<char>,
    text: &str,
    anywhere: bool,
) -> bool {
    let mut machine = Machine::new(compile(pattern, &[]), word);
    let starts = if anywhere { machine.program.len() } else { 1 };
    let mut threads: Vec<Thread> = (0..starts)
        .map(|pc| Thread {
            pc,
            ..Thread::at_start(0)
        })
        .collect();
    let (mut at, mut before) = (0, before);
    for next in text.chars() {
        machine.follow(&threads, Next::Known(Some(next)), before, at);
        threads = machine
            .leaves
            .iter()
            .filter(|thread| machine.reads(thread, next))
            .map(|thread| Thread {
                pc: thread.pc + 1,
                ..*thread
            })
            .collect();
        if threads.is_empty() {
            return false;
        }
        at += next.len_utf8();
        before = Some(next);
    }
    true
}

/// The program for `pattern` and its `ends`: the pattern, then each end in turn, its consumed part,
/// the mark, the part looked ahead at and the match.
fn compile<'p>(pattern: &'p Pattern, ends: &'p [End]) -> Vec<Op<'p>> {
    let mut program = Vec::new();
    emit(&mut program, &pattern.0);
    if ends.is_empty() {
        program.extend([Op::Mark, Op::Done(None)]);
        return program;
    }
    for (index, end) in ends.iter().enumerate() {
        let split = (index + 1 < ends.len()).then(|| {
            program.push(Op::Jump(0));
            program.len() - 1
        });
        emit(&mut program, &end.consume.0);
        program.push(Op::Mark);
        emit(&mut program, &end.ahead.0);
        program.push(Op::Done(Some(index)));
        if let Some(split) = split {
            program[split] = Op::Split(split + 1, program.len());
        }
    }
    program
}

fn emit<'p>(program: &mut Vec<Op<'p>>, atoms: &'p [Atom]) {
    for atom in atoms {
        match atom {
            Atom::Lit(text) => program.extend(text.chars().map(Op::Char)),
            Atom::Maybe(text) => {
                let split = program.len();
                program.push(Op::Jump(0));
                program.extend(text.chars().map(Op::Char));
                program[split] = Op::Split(split + 1, program.len());
            }
            Atom::Boundary => program.push(Op::Boundary),
            Atom::Run {
                class,
                min,
                max,
                greedy,
                capture,
            } => {
                let slot = capture.map(slot_index);
                if let Some(slot) = slot {
                    program.push(Op::Open(slot));
                }
                program.extend((0..*min).map(|_| Op::Class(class)));
                let choose = |take: usize, leave: usize| {
                    if *greedy {
                        Op::Split(take, leave)
                    } else {
                        Op::Split(leave, take)
                    }
                };
                match max {
                    None => {
                        let split = program.len();
                        program.extend([Op::Jump(0), Op::Class(class), Op::Jump(split)]);
                        program[split] = choose(split + 1, program.len());
                    }
                    Some(max) => {
                        let splits: Vec<usize> = (*min..*max)
                            .map(|_| {
                                program.extend([Op::Jump(0), Op::Class(class)]);
                                program.len() - 2
                            })
                            .collect();
                        let after = program.len();
                        for split in splits {
                            program[split] = choose(split + 1, after);
                        }
                    }
                }
                if let Some(slot) = slot {
                    program.push(Op::Close(slot));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(text: &str) -> Atom {
        Atom::Lit(text.into())
    }

    fn run(class: &Class, min: u16, max: Option<u16>, greedy: bool) -> Atom {
        Atom::Run {
            class: class.clone(),
            min,
            max,
            greedy,
            capture: None,
        }
    }

    fn captured(slot: Slot, class: &Class, min: u16, max: Option<u16>, greedy: bool) -> Atom {
        Atom::Run {
            class: class.clone(),
            min,
            max,
            greedy,
            capture: Some(slot),
        }
    }

    fn pattern(atoms: impl IntoIterator<Item = Atom>) -> Pattern {
        Pattern(atoms.into_iter().collect())
    }

    fn word() -> Class {
        Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
    }

    fn space() -> Class {
        Class::new([('\t', '\r'), (' ', ' ')], false)
    }

    fn none_of(chars: &str) -> Class {
        Class::new(chars.chars().map(|c| (c, c)), true)
    }

    fn end(consume: impl IntoIterator<Item = Atom>, ahead: impl IntoIterator<Item = Atom>) -> End {
        End {
            consume: pattern(consume),
            ahead: pattern(ahead),
        }
    }

    /// The verdict for `text` fed whole, cut once at every place, and a character at a time, then
    /// finished: one verdict, whatever the cuts, which this returns. A verdict given before the end
    /// must be the final one.
    fn verdict(pattern: &Pattern, ends: &[End], before: Option<char>, text: &str) -> Verdict {
        let word = word();
        let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).skip(1).collect();
        let mut plans: Vec<Vec<usize>> = vec![vec![], starts.clone()];
        plans.extend(starts.iter().map(|&cut| vec![cut]));
        let mut verdicts = plans.iter().map(|cuts| {
            let mut matcher = Matcher::new(pattern, ends, &word, before);
            let mut decided: Option<Verdict> = None;
            let mut from = 0;
            for &cut in cuts.iter().chain([&text.len()]) {
                let said = matcher.feed(&text[from..cut]);
                from = cut;
                if let Some(earlier) = &decided {
                    assert_eq!(&said, earlier, "a verdict stays, cut at {cuts:?}");
                } else if said != Verdict::Pending {
                    decided = Some(said);
                }
            }
            let last = matcher.finish();
            if let Some(earlier) = decided {
                assert_eq!(last, earlier, "a verdict stays, cut at {cuts:?}");
            }
            assert_eq!(matcher.held(), text, "every byte is held, cut at {cuts:?}");
            last
        });
        let first = verdicts.next().expect("the whole text is one plan");
        for (other, cuts) in verdicts.zip(plans.iter().skip(1)) {
            assert_eq!(other, first, "cut at {cuts:?}");
        }
        first
    }

    fn matched(verdict: Verdict) -> Match {
        match verdict {
            Verdict::Matched(found) => found,
            other => panic!("expected a match, got {other:?}"),
        }
    }

    #[test]
    fn a_literal_is_decided_by_its_last_character_or_its_first_wrong_one() {
        let tag = pattern([lit("<a>")]);
        let word = word();
        let mut matcher = Matcher::new(&tag, &[], &word, None);
        assert_eq!(matcher.feed("<a"), Verdict::Pending);
        let found = matched(matcher.feed(">tail"));
        assert_eq!(found.end(), 3);
        assert_eq!(
            &matcher.held()[found.end()..],
            "tail",
            "what follows the match is held"
        );
        let mut matcher = Matcher::new(&tag, &[], &word, None);
        assert_eq!(matcher.feed("<b"), Verdict::NoMatch);
        assert_eq!(
            matcher.feed("more"),
            Verdict::NoMatch,
            "fed past its answer"
        );
        assert_eq!(matcher.into_held(), "<bmore", "and every byte kept");
        assert_eq!(
            verdict(&tag, &[], None, "<a"),
            Verdict::NoMatch,
            "cut short"
        );
    }

    #[test]
    fn a_greedy_run_waits_for_the_character_after_it() {
        let name = pattern([captured(Slot::Name, &word(), 1, None, true)]);
        let word = word();
        let mut matcher = Matcher::new(&name, &[], &word, Some(':'));
        assert_eq!(matcher.feed("get"), Verdict::Pending);
        assert_eq!(matcher.feed("_weather"), Verdict::Pending);
        let found = matched(matcher.feed("{"));
        assert_eq!((found.end(), found.span(Slot::Name)), (11, Some(0..11)));
        assert_eq!(found.span(Slot::Key), None);
        let found = matched(verdict(&name, &[], Some(':'), "get"));
        assert_eq!(
            found.span(Slot::Name),
            Some(0..3),
            "the end of the text ends the run"
        );
        assert_eq!(verdict(&name, &[], Some(':'), "{"), Verdict::NoMatch);
    }

    #[test]
    fn greedy_and_lazy_runs_split_the_text_as_backtracking_does() {
        let word = word();
        let greedy = pattern([
            captured(Slot::Name, &word, 1, None, true),
            captured(Slot::Key, &word, 0, None, true),
        ]);
        let found = matched(verdict(&greedy, &[], None, "abc"));
        assert_eq!(
            (found.span(Slot::Name), found.span(Slot::Key)),
            (Some(0..3), Some(3..3))
        );
        let lazy = pattern([
            captured(Slot::Name, &word, 1, None, false),
            captured(Slot::Key, &word, 0, None, true),
        ]);
        let found = matched(verdict(&lazy, &[], None, "abc"));
        assert_eq!(
            (found.span(Slot::Name), found.span(Slot::Key)),
            (Some(0..1), Some(1..3))
        );
        let to_close = |greedy: bool| pattern([run(&Class::any(), 0, None, greedy), lit("</a>")]);
        let text = "x</a>y</a>z";
        assert_eq!(matched(verdict(&to_close(false), &[], None, text)).end(), 5);
        assert_eq!(matched(verdict(&to_close(true), &[], None, text)).end(), 10);
        let lazy = to_close(false);
        let mut matcher = Matcher::new(&lazy, &[], &word, None);
        assert_eq!(
            matched(matcher.feed("x</a>")).end(),
            5,
            "a lazy run decides at once"
        );
        let greedy = to_close(true);
        let mut matcher = Matcher::new(&greedy, &[], &word, None);
        assert_eq!(
            matcher.feed(text),
            Verdict::Pending,
            "a later close may still come"
        );
        assert_eq!(matched(matcher.finish()).end(), 10);
    }

    #[test]
    fn a_run_that_takes_too_much_gives_back_what_the_rest_needs() {
        let word = word();
        let a = Class::new([('a', 'a')], false);
        let bounded = |greedy: bool| pattern([run(&a, 2, Some(3), greedy), lit("a")]);
        assert_eq!(matched(verdict(&bounded(true), &[], None, "aaaa")).end(), 4);
        assert_eq!(matched(verdict(&bounded(true), &[], None, "aaa")).end(), 3);
        assert_eq!(
            matched(verdict(&bounded(false), &[], None, "aaaa")).end(),
            3
        );
        assert_eq!(verdict(&bounded(true), &[], None, "aa"), Verdict::NoMatch);
        let lines = pattern([run(&space(), 0, None, true), lit("\n<v>")]);
        assert_eq!(matched(verdict(&lines, &[], None, " \n\n<v>")).end(), 6);
        let exact = pattern([run(&word, 2, Some(2), true)]);
        assert_eq!(matched(verdict(&exact, &[], None, "abc")).end(), 2);
    }

    #[test]
    fn an_optional_literal_is_taken_before_it_is_skipped() {
        let word = word();
        let maybe = pattern([Atom::Maybe("ab".into()), lit("a")]);
        assert_eq!(matched(verdict(&maybe, &[], None, "aba")).end(), 3);
        assert_eq!(matched(verdict(&maybe, &[], None, "ac")).end(), 1);
        assert_eq!(
            matched(verdict(&maybe, &[], None, "ab")).end(),
            1,
            "taken, then given up"
        );
        let mut matcher = Matcher::new(&maybe, &[], &word, None);
        assert_eq!(
            matcher.feed("a"),
            Verdict::Pending,
            "the optional `ab` may be coming"
        );
        assert_eq!(matched(matcher.feed("c")).end(), 1);
    }

    #[test]
    fn a_word_boundary_at_the_edge_of_the_text_waits_for_the_next_character() {
        let attribute = pattern([
            Atom::Boundary,
            run(&none_of(">"), 0, None, false),
            Atom::Boundary,
            lit("k=\""),
            captured(Slot::Key, &none_of("\""), 1, None, true),
            lit("\">"),
        ]);
        let word = word();
        let mut matcher = Matcher::new(&attribute, &[], &word, Some('a'));
        assert_eq!(
            matcher.feed(" "),
            Verdict::Pending,
            "after a space, a word may begin: the boundary is not false"
        );
        let found = matched(matcher.feed("k=\"x\">"));
        assert_eq!((found.end(), found.span(Slot::Key)), (7, Some(4..5)));
        assert_eq!(
            verdict(&attribute, &[], Some('a'), "k=\"x\">"),
            Verdict::NoMatch,
            "no boundary between two word characters"
        );
        let found = matched(verdict(&attribute, &[], None, "k=\"x\">"));
        assert_eq!(
            found.end(),
            6,
            "the start of the output is no word character"
        );
        let found = matched(verdict(&attribute, &[], Some('a'), " id=1 k=\"x\">"));
        assert_eq!(found.span(Slot::Key), Some(9..10));
        let at_end = pattern([lit("a"), Atom::Boundary]);
        let mut matcher = Matcher::new(&at_end, &[], &word, None);
        assert_eq!(matcher.feed("a"), Verdict::Pending);
        assert_eq!(
            matched(matcher.finish()).end(),
            1,
            "the end of the text is no word character"
        );
    }

    #[test]
    fn a_match_waits_for_every_alternative_before_it_and_no_other() {
        let word = word();
        // The greedy run's longer match comes first: `ab` waits for what follows it.
        let longest = pattern([
            captured(Slot::Name, &word, 1, None, true),
            Atom::Maybe("!".into()),
        ]);
        let mut matcher = Matcher::new(&longest, &[], &word, None);
        assert_eq!(matcher.feed("ab"), Verdict::Pending);
        let found = matched(matcher.feed("!x"));
        assert_eq!((found.end(), found.span(Slot::Name)), (3, Some(0..2)));
        // The lazy run's shorter match comes first: decided without the character after it.
        let shortest = pattern([captured(Slot::Name, &word, 1, None, false), lit(">")]);
        let mut matcher = Matcher::new(&shortest, &[], &word, None);
        assert_eq!(matched(matcher.feed("ab>")).end(), 3);
        // A match found early stands once the alternatives before it die.
        let mut matcher = Matcher::new(&longest, &[], &word, None);
        assert_eq!(matcher.feed("ab"), Verdict::Pending);
        assert_eq!(matched(matcher.feed(" ")).end(), 2);
    }

    #[test]
    fn past_the_cap_an_undecided_pattern_is_no_match() {
        let word = word();
        let name = pattern([captured(Slot::Name, &word, 1, None, true)]);
        let mut matcher = Matcher::new(&name, &[], &word, None).hold_cap(Some(8));
        assert_eq!(
            matcher.feed("abcdefgh"),
            Verdict::Pending,
            "eight bytes are held"
        );
        assert_eq!(
            matcher.feed("i"),
            Verdict::NoMatch,
            "the ninth is past the cap"
        );
        let mut matcher = Matcher::new(&name, &[], &word, None).hold_cap(Some(8));
        assert_eq!(
            matched(matcher.feed("abcdefgh(")).end(),
            8,
            "decided by the character after the cap, which it does not take"
        );
        let long = "a".repeat(HOLD_CAP + 1);
        let mut matcher = Matcher::new(&name, &[], &word, None);
        assert_eq!(matcher.feed(&long[..HOLD_CAP]), Verdict::Pending);
        assert_eq!(
            matcher.feed(&long[HOLD_CAP..]),
            Verdict::NoMatch,
            "the default cap"
        );
        let mut matcher = Matcher::new(&name, &[], &word, None).hold_cap(None);
        assert_eq!(matcher.feed(&long), Verdict::Pending, "no cap");
        assert_eq!(matched(matcher.finish()).end(), HOLD_CAP + 1);
    }

    #[test]
    fn a_value_ends_at_the_first_end_that_matches_with_or_without_a_lookahead() {
        let word = word();
        let parameter = pattern([
            lit("<k="),
            captured(Slot::Key, &word, 1, None, true),
            lit(">"),
            captured(Slot::Value, &Class::any(), 0, None, false),
        ]);
        let ends = [end([lit("</k>")], []), end([], [lit("<k=")])];
        let found = matched(verdict(&parameter, &ends, None, "<k=a>1</k>"));
        assert_eq!(found.end(), 10);
        assert_eq!(found.ending(), Some(0));
        assert_eq!(
            (found.span(Slot::Key), found.span(Slot::Value)),
            (Some(3..4), Some(5..6))
        );
        let text = "<k=a>1<k=b>2</k>";
        let found = matched(verdict(&parameter, &ends, None, text));
        assert_eq!(
            found.ending(),
            Some(1),
            "the next tag comes first, so it ends the value"
        );
        assert_eq!((found.end(), found.span(Slot::Value)), (6, Some(5..6)));
        assert_eq!(
            &text[found.end()..],
            "<k=b>2</k>",
            "the lookahead is not taken"
        );
        let mut matcher = Matcher::new(&parameter, &ends, &word, None);
        assert_eq!(
            matcher.feed("<k=a>1<k"),
            Verdict::Pending,
            "`<k` may begin the next tag"
        );
        assert_eq!(matched(matcher.feed("=b")).end(), 6);
        let mut matcher = Matcher::new(&parameter, &ends, &word, None);
        assert_eq!(
            matcher.feed("<k=a>1<kx"),
            Verdict::Pending,
            "not a tag: still the value"
        );
        assert_eq!(matched(matcher.feed("</k>")).span(Slot::Value), Some(5..9));
        assert_eq!(
            verdict(&parameter, &ends, None, "<k=a>1"),
            Verdict::NoMatch,
            "a value that never ends is no parameter"
        );
        let trimmed = pattern([
            lit("<k="),
            captured(Slot::Key, &word, 1, None, true),
            lit(">"),
            run(&space(), 0, None, true),
            captured(Slot::Value, &Class::any(), 0, None, false),
        ]);
        let spaced = [end([run(&space(), 0, None, true), lit("</k>")], [])];
        let found = matched(verdict(&trimmed, &spaced, None, "<k=a>\n 1 2 \n</k>"));
        assert_eq!(
            found.span(Slot::Value),
            Some(7..10),
            "the whitespace around the value is the tag's"
        );
    }

    #[test]
    fn of_two_ends_at_one_place_the_first_in_the_list_wins() {
        let value = pattern([captured(Slot::Value, &Class::any(), 0, None, false)]);
        let close_first = [end([lit("</k>")], []), end([], [lit("</")])];
        let found = matched(verdict(&value, &close_first, None, "1</k>"));
        assert_eq!((found.ending(), found.end()), (Some(0), 5));
        let ahead_first = [end([], [lit("</")]), end([lit("</k>")], [])];
        let found = matched(verdict(&value, &ahead_first, None, "1</k>"));
        assert_eq!((found.ending(), found.end()), (Some(0), 1));
    }

    #[test]
    fn an_empty_pattern_matches_where_it_starts() {
        let word = word();
        let empty = Pattern::default();
        let mut matcher = Matcher::new(&empty, &[], &word, None);
        assert_eq!(matched(matcher.feed("")).end(), 0);
        assert_eq!(matched(verdict(&empty, &[], None, "abc")).end(), 0);
    }

    /// The start a search gives for `text` fed whole, cut once at every place, and a character at
    /// a time, asked with `next` after it: one start, whatever the cuts, which this returns.
    fn suffix(
        pattern: &Pattern,
        before: Option<char>,
        text: &str,
        next: Option<char>,
    ) -> Option<usize> {
        let word = word();
        let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).skip(1).collect();
        let mut plans: Vec<Vec<usize>> = vec![vec![], starts.clone()];
        plans.extend(starts.iter().map(|&cut| vec![cut]));
        let mut found = plans.iter().map(|cuts| {
            let mut search = Suffix::new(pattern, &word, before);
            let mut from = 0;
            for &cut in cuts.iter().chain([&text.len()]) {
                search.feed(&text[from..cut]);
                from = cut;
            }
            let start = search.start(next);
            assert_eq!(
                search.start(next),
                start,
                "asking again gives the same start"
            );
            start
        });
        let first = found.next().expect("the whole text is one plan");
        for (other, cuts) in found.zip(plans.iter().skip(1)) {
            assert_eq!(other, first, "cut at {cuts:?}");
        }
        first
    }

    #[test]
    fn a_search_finds_the_earliest_start_of_a_match_that_ends_at_the_end() {
        let trailer = pattern([lit("</f>"), run(&space(), 0, None, true)]);
        assert_eq!(suffix(&trailer, None, "x</f>\n", Some('<')), Some(1));
        assert_eq!(suffix(&trailer, None, "x</f>", Some('<')), Some(1));
        assert_eq!(
            suffix(&trailer, None, "</f> \n</f>\n", Some('<')),
            Some(6),
            "the first close is followed by more than whitespace"
        );
        assert_eq!(suffix(&trailer, None, "</f>x", Some('<')), None);
        assert_eq!(suffix(&trailer, None, "", Some('<')), None);
        let spaces = pattern([run(&space(), 0, None, true)]);
        assert_eq!(
            suffix(&spaces, None, "ab \n ", None),
            Some(2),
            "the run's earliest start"
        );
        assert_eq!(
            suffix(&spaces, None, "ab", None),
            Some(2),
            "an empty match ends where it starts"
        );
        let lazy = pattern([lit("<"), run(&Class::any(), 0, None, false), lit(">")]);
        assert_eq!(
            suffix(&lazy, None, "<a> <b>", Some('!')),
            Some(0),
            "a lazy run may still take what a match needs to end here"
        );
    }

    #[test]
    fn a_word_boundary_at_the_end_of_a_search_reads_what_comes_next() {
        let word = word();
        let ends_a_word = pattern([captured(Slot::Name, &word, 1, None, true), Atom::Boundary]);
        assert_eq!(suffix(&ends_a_word, None, "ab", Some('<')), Some(0));
        assert_eq!(suffix(&ends_a_word, None, "ab", None), Some(0));
        assert_eq!(
            suffix(&ends_a_word, None, "ab", Some('c')),
            None,
            "the word goes on"
        );
        let starts_a_word = pattern([Atom::Boundary, lit("x")]);
        assert_eq!(suffix(&starts_a_word, Some('a'), "x", None), None);
        assert_eq!(suffix(&starts_a_word, Some(' '), "x", None), Some(0));
        assert_eq!(suffix(&starts_a_word, None, "ax", None), None);
        assert_eq!(suffix(&starts_a_word, None, " x", None), Some(1));
    }

    #[test]
    fn a_pattern_reads_through_a_text_from_its_start_or_from_inside_it() {
        let word = word();
        let trailer = pattern([lit("</f>"), run(&space(), 0, None, true)]);
        assert!(reads_through(&trailer, &word, None, "</f> \n", false));
        assert!(!reads_through(&trailer, &word, None, "f>", false));
        assert!(
            reads_through(&trailer, &word, None, "f>", true),
            "from inside its literal"
        );
        assert!(!reads_through(&trailer, &word, None, "</k>", true));
        assert!(!reads_through(&trailer, &word, None, "</f>x", true));
        let skipping = pattern([run(&Class::any(), 0, None, false), lit("</f>")]);
        assert!(reads_through(&skipping, &word, None, "</k>", false));
        let starts_a_word = pattern([Atom::Boundary, lit("x")]);
        assert!(!reads_through(&starts_a_word, &word, Some('a'), "x", false));
        assert!(reads_through(&starts_a_word, &word, Some(' '), "x", false));
    }

    #[test]
    fn a_search_drops_a_match_longer_than_its_cap() {
        let word = word();
        let tail = pattern([lit("<"), run(&word, 0, None, true)]);
        let mut search = Suffix::new(&tail, &word, None).hold_cap(Some(4));
        search.feed("<abc");
        assert_eq!(search.start(None), Some(0), "four bytes");
        search.feed("d");
        assert_eq!(search.start(None), None, "five are past the cap");
        let mut search = Suffix::new(&tail, &word, None).hold_cap(None);
        search.feed(
            &"<".to_string()
                .chars()
                .chain("a".repeat(HOLD_CAP).chars())
                .collect::<String>(),
        );
        assert_eq!(search.start(None), Some(0), "no cap");
        let mut search = Suffix::new(&tail, &word, None);
        search.feed(
            &"<".to_string()
                .chars()
                .chain("a".repeat(HOLD_CAP).chars())
                .collect::<String>(),
        );
        assert_eq!(search.start(None), None, "the default cap");
    }

    #[test]
    fn characters_of_any_width_are_read_whole() {
        let word = Class::new(
            [
                ('a', 'z'),
                ('\u{0301}', '\u{0301}'),
                ('\u{4E00}', '\u{9FFF}'),
            ],
            false,
        );
        let name = pattern([captured(Slot::Name, &word, 1, None, true), lit("\u{1F600}")]);
        let text = "e\u{0301}\u{4E2D}\u{1F600}!";
        let mut matcher = Matcher::new(&name, &[], &word, None);
        let mut found = Verdict::Pending;
        for (at, c) in text.char_indices() {
            if found == Verdict::Pending {
                found = matcher.feed(&text[at..at + c.len_utf8()]);
            }
        }
        let found = matched(found);
        assert_eq!((found.end(), found.span(Slot::Name)), (10, Some(0..6)));
    }
}
