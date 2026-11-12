//! The cased-letter template, the pre-tokenizer of the families whose letter
//! runs are cut by case, matched by a scanner:
//!
//! ```text
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! with whatever classes stand in those places, the clitic group in both
//! letter alternatives or in neither (the Mistral tekken template has none
//! and takes one digit at a time), and whatever bound the digit rule has.
//! Run as the backtracking program the template costs 750 instructions a
//! piece (the optional lead-in and the two overlapping runs are entered and
//! given back at most positions); decided alternative by alternative it
//! costs a third of that, and the pieces are the same:
//!
//! 1. the letter alternative whose lower run is required: the lead-in is
//!    kept when the body after it matches, else the body is tried at the
//!    lead-in itself (a combining mark is a lead-in and a letter-run
//!    character at once); the body is the greedy upper run and a lower run
//!    of at least one character after it, and when no lower character
//!    follows the upper run, that run gives back to its last character that
//!    is also of the lower class (the two classes share Lm, Lo and M), where
//!    the lower run then is that character alone; the clitic is taken when
//!    its letters follow;
//! 2. the letter alternative whose upper run is required: the same lead-in
//!    rule; the upper run, the lower run after it, the clitic; nothing after
//!    the upper run can fail, so it never gives back;
//! 3. up to the bound of digits;
//! 4. a punctuation run, with the space when punctuation follows it, and
//!    the run of the break class after it;
//! 5. whitespace ending in line breaks: the whitespace run up to and
//!    including the last line break it holds, when it holds one;
//! 6. whitespace before a non-space: the run short of its last character,
//!    or whole at the end of the text, when it has a character to give;
//! 7. the whitespace run.
//!
//! Each is what the engine's backtracking arrives at: a greedy optional
//! lead-in is kept only when the rest matches after it, a greedy run before a
//! required class gives back characters until the class follows, and the
//! first alternative to reach the end of the pattern wins. The classes are
//! the parser's ([`Roles`] reads them), so the scanner and the program cannot
//! read a category differently, and the program stays as the matcher the
//! tests hold the scanner against.

use super::{
    pattern::{
        clitic_match, concat, each_match, plain_class, previous_boundary, repeat, Class, Node,
    },
    scan::{Bits, Roles, WordScan},
};

/// A part of the template that one class fills: its bit in a character's
/// role bits.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub(super) enum Role {
    /// The optional lead-in of the first letter alternative.
    Lead1,
    /// The upper run of the first letter alternative (`*`).
    Upper1,
    /// The lower run of the first letter alternative (`+`).
    Lower1,
    /// The optional lead-in of the second letter alternative.
    Lead2,
    /// The upper run of the second letter alternative (`+`).
    Upper2,
    /// The lower run of the second letter alternative (`*`).
    Lower2,
    /// A digit (`\p{N}`, up to the digit cap at once).
    Digit,
    /// The optional space before a punctuation run (` ?`).
    PunctPrefix,
    /// A punctuation run (`[^\s\p{L}\p{N}]+`).
    Punct,
    /// The run after a punctuation run (`[\r\n/]*`).
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
const ROLES: [Role; 15] = [
    Role::Lead1,
    Role::Upper1,
    Role::Lower1,
    Role::Lead2,
    Role::Upper2,
    Role::Lower2,
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
    pub(super) const fn bit(self) -> Bits {
        1 << (self as u32)
    }
}

/// The scanner; see the module documentation.
#[derive(Debug, Clone)]
pub(super) struct Scanner {
    roles: Roles,
    /// The clitic alternatives of the first and the second letter
    /// alternative, each the classes of its characters in order; empty for a
    /// template without the group.
    clitics1: Vec<Vec<usize>>,
    clitics2: Vec<Vec<usize>>,
    /// Per ASCII code point, whether a clitic alternative may begin with it.
    clitic_start: [bool; 128],
    /// Whether a clitic alternative may begin beyond ASCII.
    clitic_beyond_ascii: bool,
    /// The most digits the digit alternative takes at once.
    digit_cap: u32,
}

/// The lead-in, upper and lower classes of a letter alternative with runs
/// of the given minimums, and its clitic group (empty when it has none).
fn letters(
    node: &Node,
    upper_min: u32,
    lower_min: u32,
) -> Option<(usize, usize, usize, Vec<Vec<usize>>)> {
    let (core, clitics) = match concat(node)? {
        [lead, upper, lower] => ([lead, upper, lower], Vec::new()),
        [lead, upper, lower, clitics] => ([lead, upper, lower], clitic_group(clitics)?),
        _ => return None,
    };
    Some((
        repeat(core[0], 0, 1)?,
        repeat(core[1], upper_min, u32::MAX)?,
        repeat(core[2], lower_min, u32::MAX)?,
        clitics,
    ))
}

/// The alternatives of an optional group of literal sequences, each the
/// classes of its characters in order.
fn clitic_group(node: &Node) -> Option<Vec<Vec<usize>>> {
    let Node::Repeat {
        node,
        min: 0,
        max: 1,
        ..
    } = node
    else {
        return None;
    };
    let Node::Alt { alternatives, .. } = &**node else {
        return None;
    };
    let clitics = alternatives
        .iter()
        .map(|alternative| match alternative {
            Node::Class(class) => Some(vec![*class]),
            Node::Concat(nodes) => nodes.iter().map(plain_class).collect(),
            _ => None,
        })
        .collect::<Option<Vec<Vec<usize>>>>()?;
    (!clitics.is_empty() && clitics.iter().all(|alternative| !alternative.is_empty()))
        .then_some(clitics)
}

impl Scanner {
    /// The scanner of `root`, when it is the template; `None` for every
    /// other shape.
    pub(super) fn recognize(root: &Node, classes: &[Class]) -> Option<Self> {
        let Node::Alt { alternatives, .. } = root else {
            return None;
        };
        let [first, second, digits, punct, breaks, trail, space] = alternatives.as_slice() else {
            return None;
        };
        let (lead1, upper1, lower1, clitics1) = letters(first, 0, 1)?;
        let (lead2, upper2, lower2, clitics2) = letters(second, 1, 0)?;
        if clitics1.is_empty() != clitics2.is_empty() {
            return None;
        }
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
                Role::Lead1 => lead1,
                Role::Upper1 => upper1,
                Role::Lower1 => lower1,
                Role::Lead2 => lead2,
                Role::Upper2 => upper2,
                Role::Lower2 => lower2,
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
        for alternative in clitics1.iter().chain(&clitics2) {
            let class = &classes[alternative[0]];
            for (code, start) in clitic_start.iter_mut().enumerate() {
                *start |= class.contains(char::from(code as u8));
            }
            clitic_beyond_ascii |= class.may_hold_beyond_ascii();
        }
        Some(Self {
            roles: Roles::build(&role_classes),
            clitics1,
            clitics2,
            clitic_start,
            clitic_beyond_ascii,
            digit_cap,
        })
    }

    /// The most digits the digit alternative takes at once.
    #[cfg(test)]
    pub(super) fn digit_cap(&self) -> u32 {
        self.digit_cap
    }

    /// The role tables; for the tests that pin the templates' scans.
    #[cfg(test)]
    pub(super) fn roles(&self) -> &Roles {
        &self.roles
    }

    /// The number of clitic alternatives of a letter alternative.
    #[cfg(test)]
    pub(super) fn clitics(&self) -> usize {
        self.clitics1.len()
    }

    /// The end of the clitic that follows `end`, if one does, else `end`.
    #[inline(always)]
    fn clitic_end(
        &self,
        classes: &[Class],
        text: &str,
        end: usize,
        clitics: &[Vec<usize>],
    ) -> usize {
        if clitics.is_empty() {
            return end;
        }
        let Some(&byte) = text.as_bytes().get(end) else {
            return end;
        };
        let may_start = if byte < 0x80 {
            self.clitic_start[usize::from(byte)]
        } else {
            self.clitic_beyond_ascii
        };
        if may_start {
            clitic_match(classes, text, end, clitics).unwrap_or(end)
        } else {
            end
        }
    }

    /// The body of the first letter alternative at `q`: the upper run, the
    /// lower run that must follow it (the upper run giving back to its last
    /// character of the lower class when none follows), then the clitic.
    #[inline(always)]
    fn lower_body<const WORDS: bool>(
        &self,
        classes: &[Class],
        text: &str,
        q: usize,
        upper: Role,
        lower: Role,
        clitics: &[Vec<usize>],
    ) -> Option<usize> {
        let (p, last_lower_end) = self.roles.run_end_marked(text, q, upper.bit(), lower.bit());
        if let Some((bits, len)) = self.roles.at(text, p) {
            if bits & lower.bit() != 0 {
                let end =
                    self.roles
                        .run_end_as::<{ WordScan::LOWER }, WORDS>(text, p + len, lower.bit());
                return Some(self.clitic_end(classes, text, end, clitics));
            }
        }
        last_lower_end.map(|end| self.clitic_end(classes, text, end, clitics))
    }

    /// The body of the second letter alternative at `q`: the upper run of at
    /// least one character, the lower run after it, then the clitic.
    #[inline(always)]
    fn upper_body<const WORDS: bool>(
        &self,
        classes: &[Class],
        text: &str,
        q: usize,
        upper: Role,
        lower: Role,
        clitics: &[Vec<usize>],
    ) -> Option<usize> {
        let (bits, len) = self.roles.at(text, q)?;
        if bits & upper.bit() == 0 {
            return None;
        }
        let end = self
            .roles
            .run_end_as::<{ WordScan::UPPER }, WORDS>(text, q + len, upper.bit());
        let end = self
            .roles
            .run_end_as::<{ WordScan::LOWER }, WORDS>(text, end, lower.bit());
        Some(self.clitic_end(classes, text, end, clitics))
    }

    /// The end of the piece that starts at `pos`, if one does: the first of
    /// the seven alternatives that matches there, as the engine picks it.
    /// Compiled into its callers' loops: out of line it was a call per
    /// piece, and every value the loop keeps across a call is a spill.
    #[inline(always)]
    pub(super) fn match_at<const WORDS: bool>(
        &self,
        classes: &[Class],
        text: &str,
        pos: usize,
    ) -> Option<usize> {
        let (c, len) = self.roles.at(text, pos)?;
        let after = pos + len;
        // 1. The first letter alternative: with the lead-in, then without it.
        if c & Role::Lead1.bit() != 0 {
            if let Some(end) = self.lower_body::<WORDS>(
                classes,
                text,
                after,
                Role::Upper1,
                Role::Lower1,
                &self.clitics1,
            ) {
                return Some(end);
            }
        }
        if c & (Role::Upper1.bit() | Role::Lower1.bit()) != 0 {
            if let Some(end) = self.lower_body::<WORDS>(
                classes,
                text,
                pos,
                Role::Upper1,
                Role::Lower1,
                &self.clitics1,
            ) {
                return Some(end);
            }
        }
        // 2. The second letter alternative, the same way.
        if c & Role::Lead2.bit() != 0 {
            if let Some(end) = self.upper_body::<WORDS>(
                classes,
                text,
                after,
                Role::Upper2,
                Role::Lower2,
                &self.clitics2,
            ) {
                return Some(end);
            }
        }
        if c & Role::Upper2.bit() != 0 {
            return self.upper_body::<WORDS>(
                classes,
                text,
                pos,
                Role::Upper2,
                Role::Lower2,
                &self.clitics2,
            );
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
        //    and the break run after it.
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
    /// `Pattern::for_each_match`): a position no alternative matches at is
    /// passed over by one character, as the iterator does.
    #[inline(always)]
    pub(super) fn for_each_match<const WORDS: bool, F: FnMut(usize, usize)>(
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
