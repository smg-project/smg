//! Opener and close matching over a buffer that is still growing.

use regex::{Captures, Regex};
use regex_automata::{
    dfa::{dense, Automaton, StartKind},
    util::{primitives::StateID, syntax},
    Anchored, Input, PatternID,
};

use crate::compiled::Field;

/// Where the earliest opener starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opener {
    /// `field` opens at `start..end`, and no further input can change that.
    Complete {
        field: Field,
        start: usize,
        end: usize,
    },
    /// The text from `start` on may still become an opener, so hold it.
    Pending { start: usize },
}

/// Result of matching one opener at one position.
enum Probe {
    Dead,
    Alive,
    Match(usize),
}

/// Bound on the opener DFA, so a pathological pattern is rejected, not built.
const DFA_SIZE_LIMIT: usize = 1 << 20;

/// Anchored opener matching. The DFA tells whether an opener that reaches the
/// end of the buffer could still grow; the regexes extract its captures.
#[derive(Debug)]
pub(crate) struct Openers {
    dfa: dense::DFA<Vec<u32>>,
    regexes: Vec<Regex>,
}

impl Openers {
    /// `patterns` and `regexes` are indexed like [`Field::ALL`].
    pub(crate) fn new(patterns: &[&str], regexes: Vec<Regex>) -> Result<Self, String> {
        let config = dense::Config::new()
            .start_kind(StartKind::Anchored)
            .starts_for_each_pattern(true)
            .dfa_size_limit(Some(DFA_SIZE_LIMIT))
            .determinize_size_limit(Some(DFA_SIZE_LIMIT));
        let dfa = dense::Builder::new()
            .configure(config)
            .syntax(syntax::Config::new().dot_matches_new_line(true))
            .build_many(patterns)
            .map_err(|error| error.to_string())?;
        Ok(Self { dfa, regexes })
    }

    /// The earliest opener of one of `fields` at or after `from`; `hay[..from]`
    /// is look-behind context only. An opener that more input could still
    /// extend is pending unless `eof` says no more input will come. On a tie
    /// the longest opener wins, then the first of `fields`.
    pub(crate) fn scan(
        &self,
        hay: &str,
        from: usize,
        fields: &[Field],
        eof: bool,
    ) -> Option<Opener> {
        for (offset, _) in hay[from..].char_indices() {
            let start = from + offset;
            let mut best: Option<(Field, usize)> = None;
            for &field in fields {
                match self.probe(field, hay.as_bytes(), start, eof) {
                    Probe::Alive => return Some(Opener::Pending { start }),
                    Probe::Match(end) if best.is_none_or(|(_, longest)| end > longest) => {
                        best = Some((field, end));
                    }
                    _ => {}
                }
            }
            if let Some((field, end)) = best {
                return Some(Opener::Complete { field, start, end });
            }
        }
        None
    }

    /// The captures of `field`'s opener at `start`.
    pub(crate) fn captures<'h>(
        &self,
        field: Field,
        hay: &'h str,
        start: usize,
    ) -> Option<Captures<'h>> {
        self.regexes[field as usize]
            .captures_at(hay, start)
            .filter(|captures| captures.get(0).is_some_and(|m| m.start() == start))
    }

    fn probe(&self, field: Field, hay: &[u8], start: usize, eof: bool) -> Probe {
        let input = Input::new(hay)
            .range(start..)
            .anchored(Anchored::Pattern(PatternID::must(field as usize)));
        let Ok(mut state) = self.dfa.start_state_forward(&input) else {
            return Probe::Dead;
        };
        let mut end = None;
        for (at, &byte) in hay.iter().enumerate().skip(start) {
            state = self.dfa.next_state(state, byte);
            if self.dfa.is_match_state(state) {
                // The DFA reports a match one byte after it ends.
                end = Some(at);
            } else if self.dfa.is_dead_state(state) || self.dfa.is_quit_state(state) {
                return end.map_or(Probe::Dead, Probe::Match);
            }
        }
        if self.dfa.is_match_state(self.dfa.next_eoi_state(state)) {
            end = Some(hay.len());
        }
        if !eof && self.can_grow(state) {
            return Probe::Alive;
        }
        end.map_or(Probe::Dead, Probe::Match)
    }

    /// Whether some continuation of the input keeps the match alive. A
    /// successor that is a match state only reports the match that ends here,
    /// so it counts only if it has a live successor of its own.
    fn can_grow(&self, state: StateID) -> bool {
        let alive = |state| !self.dfa.is_dead_state(state) && !self.dfa.is_quit_state(state);
        self.bytes().any(|byte| {
            let next = self.dfa.next_state(state, byte);
            alive(next)
                && (!self.dfa.is_match_state(next)
                    || self
                        .bytes()
                        .any(|byte| alive(self.dfa.next_state(next, byte))))
        })
    }

    /// One byte per DFA byte class.
    fn bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.dfa
            .byte_classes()
            .representatives(0..=255)
            .filter_map(|unit| unit.as_u8())
    }
}

/// The earliest close in `hay` as `start..end`; the longest wins a tie.
pub fn find_close(hay: &str, closes: &[String]) -> Option<(usize, usize)> {
    closes
        .iter()
        .filter_map(|close| hay.find(close.as_str()).map(|at| (at, at + close.len())))
        .min_by_key(|&(start, end)| (start, std::cmp::Reverse(end)))
}

/// Length of the longest suffix of `hay` that is a proper prefix of a close.
pub fn partial_close_len(hay: &str, closes: &[String]) -> usize {
    closes
        .iter()
        .filter_map(|close| {
            (1..close.len())
                .rev()
                .find(|&len| close.is_char_boundary(len) && hay.ends_with(&close[..len]))
        })
        .max()
        .unwrap_or(0)
}

/// Drop `buffer[..end]` except its last character, which stays as look-behind
/// context for the next scan. Returns the length of that context.
pub fn consume(buffer: &mut String, end: usize) -> usize {
    let context = buffer[..end].chars().next_back().map_or(0, char::len_utf8);
    buffer.drain(..end - context);
    context
}
