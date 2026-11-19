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
//! and takes one digit at a time), whatever bound the digit rule has, and
//! in front of the seven, or not, an alternative that is a run of one class
//! (the Kimi template opens with its ideographs, `[\p{Han}]+`, and takes
//! them out of both letter classes).
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
//! first alternative to reach the end of the pattern wins.
//!
//! The opening run, in a template that has one, is the first alternative: a
//! piece that starts at a character of its class is that run, whatever else
//! the character is (a Han radical is punctuation and a Han numeral a digit
//! inside a piece that began before it, and a run of their own where one
//! begins). Such a template is the scanner's when no character of the run's
//! class is one a letter body can begin with (the upper and the lower class
//! of the first letter alternative, the upper class of the second). A
//! character of the run then reaches a letter alternative as its lead-in
//! alone, a role it never plays, since the run is tried first: the role
//! table holds no lead-in bit for it, and the scanner reads the run in the
//! digit test, one mask for both. The letter, punctuation and whitespace
//! pieces of a template with no opening run take the tests they took, and a
//! digit piece one more. A template whose run shares a character with those
//! letter classes keeps the program. The classes are
//! the parser's ([`Roles`] reads them), so the scanner and the program cannot
//! read a category differently, and the program stays as the matcher the
//! tests hold the scanner against.

use super::{
    block,
    pattern::{
        clitic_match, concat, each_match, each_match_by_blocks, plain_class, previous_boundary,
        repeat, Class, Node,
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
    /// The run a template opens with (`[\p{Han}]+`): the last role, which a
    /// template without the run has no class for.
    Run,
}

/// The roles, in the order of their bits.
const ROLES: [Role; 16] = [
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
    Role::Run,
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
    /// Whether [`block`]'s pass cuts ASCII text as this scanner does
    /// ([`Self::reads_ascii_as_blocks`]) and the alternatives leave no gap.
    blocks: bool,
    /// Whether the run behind punctuation takes the slash (`[\r\n/]*`; the
    /// Kimi template's takes the line breaks alone).
    take_slash: bool,
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
        // An eighth alternative in front is the opening run.
        let (run, alternatives) = match alternatives.as_slice() {
            [run, seven @ ..] if seven.len() == 7 => (Some(repeat(run, 1, u32::MAX)?), seven),
            alternatives => (None, alternatives),
        };
        let [first, second, digits, punct, breaks, trail, space] = alternatives else {
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
                Role::Run => run?,
            })
        };
        // The run's role is the last: a template without the run has the
        // tables it had, and no character holds the role's bit.
        let role_classes = ROLES[..ROLES.len() - usize::from(run.is_none())]
            .iter()
            .map(|&role| of(role).map(|class| &classes[class]))
            .collect::<Option<Vec<&Class>>>()?;
        let mut roles = Roles::build(&role_classes);
        if run.is_some() {
            // See the module documentation: no character of the run may
            // begin a letter body, and none is a lead-in.
            let bodies = Role::Upper1.bit() | Role::Lower1.bit() | Role::Upper2.bit();
            if roles.some_char_holds_both(Role::Run.bit(), bodies) {
                return None;
            }
            roles.clear_where(Role::Run.bit(), Role::Lead1.bit() | Role::Lead2.bit());
        }
        let mut clitic_start = [false; 128];
        let mut clitic_beyond_ascii = false;
        for alternative in clitics1.iter().chain(&clitics2) {
            let class = &classes[alternative[0]];
            for (code, start) in clitic_start.iter_mut().enumerate() {
                *start |= class.contains(char::from(code as u8));
            }
            clitic_beyond_ascii |= class.may_hold_beyond_ascii();
        }
        let mut scanner = Self {
            roles,
            clitics1,
            clitics2,
            clitic_start,
            clitic_beyond_ascii,
            digit_cap,
            blocks: false,
            take_slash: false,
        };
        scanner.take_slash = (scanner.roles.ascii_members(Role::PunctBreak.bit()) >> b'/') & 1 != 0;
        scanner.blocks = scanner.tiles() && scanner.reads_ascii_as_blocks(classes);
        Some(scanner)
    }

    /// Whether the encoder may read a text of this template by blocks.
    pub(super) fn blocks(&self) -> bool {
        self.blocks
    }

    /// Whether the alternatives leave no gap in any text: every character
    /// holds a role that starts a piece of one character at least (the
    /// encoder's block routine keeps no cursor to find a gap by).
    fn tiles(&self) -> bool {
        self.roles.every_char_holds(
            Role::Lower1.bit()
                | Role::Upper2.bit()
                | Role::Digit.bit()
                | Role::Punct.bit()
                | Role::Space.bit()
                | Role::Run.bit(),
        )
    }

    /// Whether [`block`]'s pass cuts ASCII text as this scanner does. The
    /// pass computes the starts from the plain ASCII sets, so every role
    /// must hold exactly its set of them (the opening run none); the two
    /// letter alternatives must carry one clitic group, whose clitics begin
    /// with the apostrophe alone and go on in letters; and the digit rule's
    /// bound must fit a block's shifts.
    fn reads_ascii_as_blocks(&self, classes: &[Class]) -> bool {
        let ascii = |test: &dyn Fn(u8) -> bool| {
            (0..128u8)
                .filter(|&code| test(code))
                .fold(0u128, |set, code| set | (1 << code))
        };
        let capitals = ascii(&|code| code.is_ascii_uppercase());
        let smalls = ascii(&|code| code.is_ascii_lowercase());
        let letters = capitals | smalls;
        let digits = ascii(&block::is_digit);
        let spaces = ascii(&block::is_space);
        let breaks = ascii(&block::is_break);
        let lead = !(letters | digits | breaks);
        let roles = [
            (Role::Lead1, lead),
            (Role::Upper1, capitals),
            (Role::Lower1, smalls),
            (Role::Lead2, lead),
            (Role::Upper2, capitals),
            (Role::Lower2, smalls),
            (Role::Digit, digits),
            (Role::PunctPrefix, 1 << b' '),
            (Role::Punct, !(letters | digits | spaces)),
            (
                Role::PunctBreak,
                breaks | (u128::from(self.take_slash) << b'/'),
            ),
            (Role::BreakSpace, spaces),
            (Role::Break, breaks),
            (Role::TrailSpace, spaces),
            (Role::NotTrail, !spaces),
            (Role::Space, spaces),
            (Role::Run, 0),
        ];
        let clitics_go_on_in_letters = self.clitics1.iter().all(|clitic| {
            (2..=block::BLOCK).contains(&clitic.len())
                && clitic[1..].iter().all(|&class| {
                    ascii(&|code| classes[class].contains(char::from(code))) & !letters == 0
                })
        });
        // The two groups are two sets of classes to the parser: the same
        // group when they hold the same ASCII, place by place.
        let members = |class: usize| ascii(&|code| classes[class].contains(char::from(code)));
        let same_clitics = self.clitics1.len() == self.clitics2.len()
            && self
                .clitics1
                .iter()
                .zip(&self.clitics2)
                .all(|(one, other)| {
                    one.len() == other.len()
                        && one
                            .iter()
                            .zip(other)
                            .all(|(&a, &b)| members(a) == members(b))
                });
        let quote = !self.clitics1.is_empty();
        roles
            .into_iter()
            .all(|(role, set)| self.roles.ascii_members(role.bit()) == set)
            && same_clitics
            && (0..128).all(|code| self.clitic_start[code] == (quote && code == usize::from(b'\'')))
            && clitics_go_on_in_letters
            && (self.digit_cap < u64::BITS || self.digit_cap == u32::MAX)
    }

    /// [`block::cased_pieces`] from `from`, a piece's start, with this
    /// scanner's digit bound and clitics.
    #[inline(always)]
    pub(super) fn block_pass<F: FnMut(usize, usize)>(
        &self,
        classes: &[Class],
        text: &str,
        from: usize,
        f: &mut F,
    ) -> usize {
        block::cased_pieces(
            text.as_bytes(),
            from,
            self.digit_cap,
            self.take_slash,
            |at| clitic_match(classes, text, at, &self.clitics1),
            f,
        )
    }

    /// [`Self::for_each_match`] through [`block`]'s pass, for a text
    /// `Pattern::by_cased_blocks` gives a reader of: the templates' one walk
    /// (`pattern::each_match_by_blocks`) with this template's pass
    /// ([`Self::block_pass`]) and its `match_at`.
    #[inline(always)]
    pub(super) fn for_each_match_by_blocks<const WORDS: bool, F: FnMut(usize, usize)>(
        &self,
        classes: &[Class],
        text: &str,
        f: &mut F,
    ) {
        each_match_by_blocks(
            text,
            #[inline(always)]
            |from, f| self.block_pass(classes, text, from, f),
            #[inline(always)]
            |pos| self.match_at::<WORDS>(classes, text, pos),
            f,
        );
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
    /// the template's alternatives that matches there, as the engine picks
    /// it.
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
        // 3. Up to the bound of digits; and, behind the same test, the
        //    opening run of a template that has one: no letter alternative
        //    takes a character of it (see the module documentation), so the
        //    first alternative is read here and the pieces of every other
        //    kind pay no test for it.
        if c & (Role::Digit.bit() | Role::Run.bit()) != 0 {
            if c & Role::Run.bit() != 0 {
                return Some(self.roles.run_end(text, after, Role::Run.bit()));
            }
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
