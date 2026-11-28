//! A pre-tokenizer chain of three `Split` stages, matched with one scan.
//!
//! One family of byte-level BPE tokenizers pre-tokenizes in three stages,
//! each a `Split` that cuts the pieces of the stage before it:
//!
//! ```text
//! \p{N}{1,3}
//! [一-龥぀-ゟ゠-ヿ]+
//! [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Digit runs are cut into groups of up to three, runs of one class (CJK
//! ideographs and kana in the files that ship it) are cut out of what is
//! left, and a six-alternative template cuts the rest, every stage applied
//! to each piece of the stage before it on its own: a lead-in, a lookahead
//! or a run that gives characters back never reaches past the piece it is
//! in. Run as three patterns that is three passes over the text, each
//! interpreting its program character by character, with the pieces of one
//! handed to the next.
//!
//! [`Scanner`] produces the same pieces in one pass. The first two stages
//! come down to character classes read left to right: a character of the
//! digit class opens a group of up to the bound, a character of the run
//! class a run that a digit ends (the first stage cut there), anything else
//! a stretch that the next digit or run character ends. The template is
//! then decided inside each of those pieces, alternative by alternative,
//! the way [`super::pattern::Scanner`] decides the GPT-4 family's template:
//! a punctuation character followed by letters takes the letter run; a
//! letter run keeps its lead-in when a letter follows it; a punctuation run
//! keeps its space when punctuation follows it and takes the line breaks
//! after it; whitespace runs up to and including the last line break it
//! holds, or short of its last character when a non-space follows, or
//! whole. Each is what the backtracking engine arrives at for this shape,
//! so the pieces are the same, and the piece's end is the end of the text
//! for every decision, as it is for the engine: a group or a run is cut
//! first and the template reads it on its own, while a stretch between them
//! is cut by the template as it is read, the next digit or run character
//! standing in for the end of the text in every decision, so the stretch is
//! read once, not scanned to its end and then read. A character's classes
//! are read as role bits in one table lookup ([`Roles`]), so a run is a
//! loop over bytes.
//!
//! The shape is recognized from the parsed patterns, with whatever classes
//! stand in its places and whatever bound the digit rule has; the classes
//! are the parser's, so the scan and the programs cannot read a Unicode
//! category differently, and the programs stay as the second matcher the
//! tests hold the scan against.

use super::{
    block,
    pattern::{concat, plain_class, previous_boundary, repeat, Class, Node},
    scan::{Bits, Roles, WordScan},
    Pieces, Stage,
};

/// A part of the template that one class fills: its bit in a character's
/// role bits.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Role {
    /// The first stage's class (`\p{N}`), taken in groups up to its bound.
    Digit,
    /// The second stage's class, taken in runs.
    Run,
    /// The punctuation before a letter run (`[!-/:-@\[-`{-~]`).
    PunctLead,
    /// The letters after it (`[A-Za-z]+`).
    PunctLetter,
    /// The optional lead-in of a letter run (`[^\r\n\p{L}\p{P}\p{S}]?`).
    Lead,
    /// A letter run (`[\p{L}\p{M}]+`).
    Word,
    /// The optional space before a punctuation run (` ?`).
    PunctSpace,
    /// A punctuation run (`[\p{P}\p{S}]+`).
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

/// The roles, so many bits of a character.
const ROLES: usize = 14;

/// The characters that end a stretch between the first two stages' pieces:
/// read as the end of the text by every decision of the template inside
/// such a stretch.
const STAGE_STOP: Bits = Role::Digit.bit() | Role::Run.bit();

/// The role bits that let the stages after the first cut a group of the
/// first: a character of the second stage's class, or one that opens a match
/// of the template, alone (a letter, a punctuation mark, whitespace) or
/// before a letter (the punctuation lead). A group whose characters hold
/// none of them is one piece: the second stage finds no run in it and the
/// template no match at any of its characters. The lead-in of a letter run
/// and the space before punctuation are not among them: each opens a match
/// only before a character that holds one of these bits.
const GROUP_CUTS: Bits = Role::Run.bit()
    | Role::PunctLead.bit()
    | Role::Word.bit()
    | Role::Punct.bit()
    | Role::BreakSpace.bit()
    | Role::Break.bit()
    | Role::TrailSpace.bit()
    | Role::Space.bit();

/// The template's roles in the order [`Part::Template`] lists their classes,
/// which is the order of their bits after the two stage classes.
const TEMPLATE_ROLES: [Role; ROLES - 2] = [
    Role::PunctLead,
    Role::PunctLetter,
    Role::Lead,
    Role::Word,
    Role::PunctSpace,
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

/// What one pattern is as a stage of the chain, read from its parsed tree
/// once; `Scanner::recognize` puts three of them together.
#[derive(Debug, Clone)]
pub(super) enum Part {
    /// A run of one class, `1..=max` characters at a time.
    Run { class: usize, max: u32 },
    /// The six-alternative template: the class of each role, in the order
    /// of [`TEMPLATE_ROLES`].
    Template([usize; ROLES - 2]),
}

impl Part {
    /// The part `root` is, if it is one.
    pub(super) fn recognize(root: &Node) -> Option<Self> {
        match root {
            Node::Class(class) => Some(Self::Run {
                class: *class,
                max: 1,
            }),
            Node::ClassRepeat {
                class,
                min: 1,
                max,
                possessive: false,
            } => Some(Self::Run {
                class: *class,
                max: *max,
            }),
            Node::Alt { alternatives, .. } => {
                let [punct_letters, word, punct, breaks, trail, space] = alternatives.as_slice()
                else {
                    return None;
                };
                let [punct_lead, letter_run] = concat(punct_letters)? else {
                    return None;
                };
                let [lead, word_run] = concat(word)? else {
                    return None;
                };
                let [punct_space, punct_run, punct_break] = concat(punct)? else {
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
                Some(Self::Template([
                    plain_class(punct_lead)?,
                    repeat(letter_run, 1, u32::MAX)?,
                    repeat(lead, 0, 1)?,
                    repeat(word_run, 1, u32::MAX)?,
                    repeat(punct_space, 0, 1)?,
                    repeat(punct_run, 1, u32::MAX)?,
                    repeat(punct_break, 0, u32::MAX)?,
                    repeat(break_space, 0, u32::MAX)?,
                    repeat(break_run, 1, u32::MAX)?,
                    repeat(trail_space, 1, u32::MAX)?,
                    *not_trail,
                    repeat(space, 1, u32::MAX)?,
                ]))
            }
            _ => None,
        }
    }
}

/// The one-pass matcher of a three-stage chain; see the module
/// documentation.
#[derive(Debug, Clone)]
pub(super) struct Scanner {
    /// The role bits of a character: the two stage classes, then the
    /// template's roles in the order of [`TEMPLATE_ROLES`].
    roles: Roles,
    /// The most characters of the digit class one group holds.
    digits: u32,
    /// Whether the letter and white-space runs may take the word scans
    /// (`Roles::run_end_as_in`): no stop class holds an ASCII character of
    /// the scanned sets, so a word of members never hides a stop.
    word_scans: bool,
    /// Whether the encoder may read a text by blocks: [`block`]'s pass cuts
    /// plain ASCII text as this scanner does
    /// ([`Self::reads_ascii_as_blocks`]).
    blocks: bool,
}

impl Scanner {
    /// The scanner of `stages`, when they are a chain of a bounded run, an
    /// unbounded run and the template, each passing its matches and gaps
    /// on; `None` for every other stage list, which the stage patterns
    /// match in turn.
    pub(super) fn recognize(stages: &[Stage]) -> Option<Self> {
        let [first, second, third] = stages else {
            return None;
        };
        if stages
            .iter()
            .any(|stage| stage.pieces != Pieces::MatchesAndGaps)
        {
            return None;
        }
        let Some(Part::Run {
            class: digit,
            max: digits,
        }) = first.pattern.part()
        else {
            return None;
        };
        let Some(Part::Run {
            class: run,
            max: u32::MAX,
        }) = second.pattern.part()
        else {
            return None;
        };
        let Some(Part::Template(template)) = third.pattern.part() else {
            return None;
        };
        let mut classes: Vec<&Class> = Vec::with_capacity(ROLES);
        classes.push(&first.pattern.classes()[*digit]);
        classes.push(&second.pattern.classes()[*run]);
        for (role, class) in TEMPLATE_ROLES.iter().zip(template) {
            // the bit of a role is its place in this list
            debug_assert_eq!(*role as usize, classes.len());
            classes.push(&third.pattern.classes()[*class]);
        }
        let roles = Roles::build(&classes);
        let word_scans = roles.ascii_members(STAGE_STOP) & WordScan::MEMBERS == 0;
        let blocks = Self::reads_ascii_as_blocks(&roles, *digits);
        tracing::debug!(
            word_scans,
            blocks,
            "the native encode path takes the chain scanner"
        );
        Some(Self {
            roles,
            digits: *digits,
            word_scans,
            blocks,
        })
    }

    /// Whether [`block`]'s pass cuts plain ASCII text as this scanner does.
    /// The pass computes the starts from the plain ASCII sets, so every
    /// role must hold, of ASCII, exactly its set of them: the ten digits in
    /// the first stage's class and nothing in the second's, the letters,
    /// the punctuation marks and the whitespace in the template's places,
    /// and every other character as a lead-in alone (a control character:
    /// the pass leaves a block that holds one to the scanner). The digit
    /// bound must fit a block's shifts. A chain with another class in any
    /// place is matched by the scanner alone.
    fn reads_ascii_as_blocks(roles: &Roles, digits: u32) -> bool {
        let ascii = |test: &dyn Fn(u8) -> bool| {
            (0..128u8)
                .filter(|&code| test(code))
                .fold(0u128, |set, code| set | (1 << code))
        };
        let letters = ascii(&block::is_letter);
        let digit_set = ascii(&block::is_digit);
        let spaces = ascii(&block::is_space);
        let breaks = ascii(&block::is_break);
        let controls = ascii(&|code| code.is_ascii_control()) & !spaces;
        let punct = !(letters | digit_set | spaces | controls);
        [
            (Role::Digit, digit_set),
            (Role::Run, 0),
            (Role::PunctLead, punct),
            (Role::PunctLetter, letters),
            (Role::Lead, !(letters | punct | breaks)),
            (Role::Word, letters),
            (Role::PunctSpace, 1 << b' '),
            (Role::Punct, punct),
            (Role::PunctBreak, breaks),
            (Role::BreakSpace, spaces),
            (Role::Break, breaks),
            (Role::TrailSpace, spaces),
            (Role::NotTrail, !spaces),
            (Role::Space, spaces),
        ]
        .into_iter()
        .all(|(role, set)| roles.ascii_members(role.bit()) == set)
            && (digits < u64::BITS || digits == u32::MAX)
    }

    /// Calls `f` with each piece of `text` as its byte range, in order; the
    /// pieces tile the text, as three `Split` stages that keep their gaps
    /// leave no character out. `words` says the run loops take their word
    /// scans (a long input, or a unit that opens as ASCII text; see the
    /// encoder's `WORD_SCAN_MIN_BYTES`): the walk is compiled both ways and
    /// the choice is made once per unit. A unit [`block`]'s pass serves
    /// ([`Self::by_blocks`]) is read by blocks, any other by the scanner's
    /// own loop. `f` is taken by value, so that a caller's closure can ask
    /// to be compiled into the walk (`#[inline(always)]` on the closure, as
    /// the encoder's probe does).
    pub(super) fn pieces<F: FnMut(usize, usize)>(&self, text: &str, words: bool, mut f: F) {
        if self.by_blocks(text, words) {
            if words {
                self.pieces_by_blocks::<true, F>(text, &mut f);
            } else {
                self.pieces_by_blocks::<false, F>(text, &mut f);
            }
        } else if words {
            self.pieces_as::<true, F>(text, &mut f);
        } else {
            self.pieces_as::<false, F>(text, &mut f);
        }
    }

    /// Whether `text` is read by blocks, decided as the pass's other caller
    /// decides it (`Pattern::by_blocks`): [`block`]'s pass cuts as this
    /// scanner does, the text is no shorter than the pass's bound, and one
    /// of the places it is asked at begins an ASCII stretch
    /// ([`block::has_stretch`]). A text of under one step between two such
    /// places is asked at its opening alone, and one that takes no word
    /// scans (`words`) has answered that, it does not open as ASCII text:
    /// it is not asked, and runs the code it ran before the pass. A longer
    /// text in another script pays the test and keeps the scanner's own
    /// loop.
    #[inline]
    fn by_blocks(&self, text: &str, words: bool) -> bool {
        self.blocks
            && text.len() >= block::MIN_BYTES
            && (words || text.len() >= block::SAMPLE)
            && block::has_stretch(text.as_bytes())
    }

    /// [`Self::pieces`] through [`block`]'s pass, word scans on or off. The
    /// pass cuts each stretch of plain ASCII. What lies between two
    /// stretches and behind the last one is cut here by the scanner's loop
    /// with a bound, from the piece the pass left open up to the first
    /// piece start a walk further on:
    /// text beyond ASCII, a block with a control character, a blank run the
    /// pass declines, the text's last block. Behind a stretch of four
    /// blocks or more the pass is tried again two blocks on; where it cut
    /// less the walk goes twice as far as the one before, up to
    /// [`block::MAX_WALK`]. The pass begins at any piece start and the
    /// scanner at any other: no rule of the template reads the text before
    /// a piece's start, and a group of the first stage begins at each of
    /// its cuts. Out of line: compiled into [`Self::pieces`], it made the
    /// scanner's own loop there dearer (counted on a memorized text of
    /// 1.5 KB in another script: about a hundred instructions a text).
    #[inline(never)]
    fn pieces_by_blocks<const WORDS: bool, F: FnMut(usize, usize)>(&self, text: &str, f: &mut F) {
        let mut pos = 0;
        let mut walk = 2 * block::BLOCK;
        while pos < text.len() {
            let open = block::chain_pieces(text.as_bytes(), pos, self.digits, f);
            // The pass reads two blocks before it cuts one, so a short
            // stretch pays for more blocks than it cuts: only a stretch of
            // four blocks says the text reads by blocks here.
            walk = if open - pos >= 4 * block::BLOCK {
                2 * block::BLOCK
            } else {
                (2 * walk).min(block::MAX_WALK)
            };
            pos = self.pieces_from::<WORDS, true, F>(text, open, open + walk, f);
        }
    }

    /// A piece end off the walk's main path (a digit group or a run taken
    /// whole, the text before a match or after the last one): the callback
    /// out of line, so that a callback compiled into the walk stands at the
    /// template's match alone.
    #[inline(never)]
    fn aside<F: FnMut(usize, usize)>(f: &mut F, start: usize, end: usize) {
        f(start, end);
    }

    /// [`Self::pieces`] by the scanner's own loop, with the word scans on or
    /// off.
    fn pieces_as<const WORDS: bool, F: FnMut(usize, usize)>(&self, text: &str, f: &mut F) {
        self.pieces_from::<WORDS, false, F>(text, 0, usize::MAX, f);
    }

    /// The pieces of `text` from `from`, a piece's start, to its end; with
    /// `BOUNDED`, to the first piece start at or behind `until`, which is
    /// answered (the text's length when there is none): the walk by blocks
    /// reads the text between two stretches of the pass this way.
    #[inline(always)]
    fn pieces_from<const WORDS: bool, const BOUNDED: bool, F: FnMut(usize, usize)>(
        &self,
        text: &str,
        from: usize,
        until: usize,
        f: &mut F,
    ) -> usize {
        let mut pos = from;
        while let Some((c, len)) = self.roles.at(text, pos) {
            if BOUNDED && pos >= until {
                break;
            }
            let after = pos + len;
            pos = if c & Role::Digit.bit() != 0 {
                // The first stage: a group of up to the bound; the stages
                // after it see the group on its own. `any` gathers the role
                // bits of its characters.
                let mut end = after;
                let mut count = 1;
                let mut any = c;
                while count < self.digits {
                    match self.roles.at(text, end) {
                        Some((d, n)) if d & Role::Digit.bit() != 0 => {
                            any |= d;
                            end += n;
                            count += 1;
                        }
                        _ => break,
                    }
                }
                if any & GROUP_CUTS == 0 {
                    // No character of the group is of the second stage's
                    // class or opens a match of the template: the group is
                    // one piece, and the two stages' reads of it are left
                    // out.
                    Self::aside(f, pos, end);
                } else {
                    self.group_pieces::<WORDS, F>(text, pos, end, f);
                }
                end
            } else if c & Role::Run.bit() != 0 {
                // The second stage, inside a gap of the first: a run that
                // the next digit ends, since the first stage cut there.
                // `every` keeps the role bits all its characters share.
                let mut end = after;
                let mut every = c;
                while let Some((d, n)) = self.roles.at(text, end) {
                    if d & Role::Run.bit() == 0 || d & Role::Digit.bit() != 0 {
                        break;
                    }
                    every &= d;
                    end += n;
                }
                if every & Role::Word.bit() != 0 && c & Role::PunctLead.bit() == 0 {
                    // Every character is a letter of the template and the
                    // first opens no punctuation match: the template's
                    // letter run takes the run whole, so the run is one
                    // piece and the template's read of it is left out.
                    Self::aside(f, pos, end);
                    end
                } else {
                    self.template_pieces::<WORDS, false, F>(text, pos, end, 0, usize::MAX, f)
                }
            } else {
                // A gap of both: the stretch to the next digit or run
                // character, which the template cuts as it reads it.
                self.template_pieces::<WORDS, BOUNDED, F>(
                    text,
                    pos,
                    text.len(),
                    STAGE_STOP,
                    until,
                    f,
                )
            };
        }
        pos
    }

    /// The second and third stages over one group of the first,
    /// `text[start..end]`: the runs of the second stage's class in it and
    /// the stretches between them, each cut by the template. The walk brings
    /// only a group that holds a character of [`GROUP_CUTS`]: any other is
    /// one piece, as a group of a digit class is, and the stages applied
    /// here would cut nothing in it. So the pieces are the stages' for any
    /// classes.
    fn group_pieces<const WORDS: bool, F: FnMut(usize, usize)>(
        &self,
        text: &str,
        start: usize,
        end: usize,
        f: &mut F,
    ) {
        let mut pos = start;
        while pos < end {
            let Some((c, len)) = self.roles.at(text, pos) else {
                break;
            };
            let run = c & Role::Run.bit() != 0;
            let mut piece_end = pos + len;
            while piece_end < end {
                match self.roles.at(text, piece_end) {
                    Some((d, n)) if (d & Role::Run.bit() != 0) == run => piece_end += n,
                    _ => break,
                }
            }
            self.template_pieces::<WORDS, false, F>(text, pos, piece_end, 0, usize::MAX, f);
            pos = piece_end;
        }
    }

    /// The template's pieces of `text[start..end]`, a piece of the stages
    /// before it, or, with `stop` bits, of the text from `start` up to the
    /// first character holding one of them (a stretch between the first two
    /// stages' pieces, which the template cuts with that character as the
    /// end of the text): its matches, and the text between them as pieces
    /// of their own. Answers where the piece ended.
    fn template_pieces<const WORDS: bool, const BOUNDED: bool, F: FnMut(usize, usize)>(
        &self,
        text: &str,
        start: usize,
        end: usize,
        stop: Bits,
        until: usize,
        f: &mut F,
    ) -> usize {
        let piece = &text[start..end];
        let mut gap = 0;
        let mut pos = 0;
        while let Some((c, len)) = self.at_in(piece, pos, stop) {
            match self.template_match::<WORDS>(piece, pos, c, len, stop) {
                Some(matched) => {
                    if gap < pos {
                        Self::aside(f, start + gap, start + pos);
                    }
                    f(start + pos, start + matched);
                    pos = matched;
                    gap = matched;
                    // The walk by blocks stops at the first piece start at
                    // or behind its bound (`BOUNDED`).
                    if BOUNDED && start + matched >= until {
                        break;
                    }
                }
                None => pos += len,
            }
        }
        if gap < pos {
            Self::aside(f, start + gap, start + pos);
        }
        start + pos
    }

    /// The role bits of the character at `pos` and its length, `None` at
    /// the end of the text or at a character holding one of `stop`.
    #[inline(always)]
    fn at_in(&self, text: &str, pos: usize, stop: Bits) -> Option<(Bits, usize)> {
        self.roles
            .at(text, pos)
            .filter(|(bits, _)| bits & stop == 0)
    }

    /// The end of the run of characters holding `bit` that starts at `pos`,
    /// a character holding one of `stop` ending it as the text's end would
    /// ([`Roles::run_end_in`], the one byte loop with stop bits).
    #[inline(always)]
    fn run_end_in(&self, text: &str, pos: usize, bit: Bits, stop: Bits) -> usize {
        self.roles.run_end_in(text, pos, bit, stop)
    }

    /// [`Self::run_end_in`] through the word scan `SCAN` of the role's ASCII
    /// members, eight bytes at a time, when no stop class has a character
    /// among them (read at recognition) and the encoder passes the scans on
    /// (`WORDS`); byte by byte otherwise.
    #[inline(always)]
    fn run<const SCAN: u8, const WORDS: bool>(
        &self,
        text: &str,
        pos: usize,
        bit: Bits,
        stop: Bits,
    ) -> usize {
        if WORDS && self.word_scans {
            self.roles.run_end_as_in::<SCAN>(text, pos, bit, stop)
        } else {
            self.run_end_in(text, pos, bit, stop)
        }
    }

    /// The end of the template's match that starts at `pos`, where a
    /// character with the role bits `c` of `len` bytes stands, if one does:
    /// the first of the six alternatives that matches there, as the engine
    /// picks it; a character holding one of `stop` is the end of the text.
    /// Compiled into each copy of the template's loop: the loop has a
    /// bounded copy for the walk by blocks, and as a routine with two
    /// callers this was a call a piece (counted: about twenty instructions
    /// a piece on the walk, two on a short text).
    #[inline(always)]
    fn template_match<const WORDS: bool>(
        &self,
        text: &str,
        pos: usize,
        c: Bits,
        len: usize,
        stop: Bits,
    ) -> Option<usize> {
        let after = pos + len;
        // 1. Punctuation followed by letters: the letter run after it.
        if c & Role::PunctLead.bit() != 0 {
            if let Some((d, n)) = self.at_in(text, after, stop) {
                if d & Role::PunctLetter.bit() != 0 {
                    return Some(self.run::<{ WordScan::LETTERS }, WORDS>(
                        text,
                        after + n,
                        Role::PunctLetter.bit(),
                        stop,
                    ));
                }
            }
        }
        // 2. A letter run, with the lead-in when a letter follows it.
        if c & Role::Lead.bit() != 0 {
            if let Some((d, n)) = self.at_in(text, after, stop) {
                if d & Role::Word.bit() != 0 {
                    return Some(self.run::<{ WordScan::LETTERS }, WORDS>(
                        text,
                        after + n,
                        Role::Word.bit(),
                        stop,
                    ));
                }
            }
        }
        if c & Role::Word.bit() != 0 {
            return Some(self.run::<{ WordScan::LETTERS }, WORDS>(
                text,
                after,
                Role::Word.bit(),
                stop,
            ));
        }
        // 3. A punctuation run, with the space when punctuation follows it,
        //    and the line breaks after the run.
        if c & Role::PunctSpace.bit() != 0 {
            if let Some((d, n)) = self.at_in(text, after, stop) {
                if d & Role::Punct.bit() != 0 {
                    let end = self.run_end_in(text, after + n, Role::Punct.bit(), stop);
                    return Some(self.run_end_in(text, end, Role::PunctBreak.bit(), stop));
                }
            }
        }
        if c & Role::Punct.bit() != 0 {
            let end = self.run_end_in(text, after, Role::Punct.bit(), stop);
            return Some(self.run_end_in(text, end, Role::PunctBreak.bit(), stop));
        }
        // 4. Whitespace ending in line breaks: the greedy whitespace run
        //    gives back to the last position where a line break stands,
        //    inside the run or right after it.
        let mut end = pos;
        let mut last_break = None;
        loop {
            match self.at_in(text, end, stop) {
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
            return Some(self.run_end_in(text, start, Role::Break.bit(), stop));
        }
        // 5. Whitespace before a non-space: the greedy run gives back its
        //    last character while a non-space follows it, down to one.
        if c & Role::TrailSpace.bit() != 0 {
            let mut end =
                self.run::<{ WordScan::SPACES }, WORDS>(text, after, Role::TrailSpace.bit(), stop);
            loop {
                match self.at_in(text, end, stop) {
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
        // 6. The whitespace run.
        if c & Role::Space.bit() != 0 {
            return Some(self.run::<{ WordScan::SPACES }, WORDS>(
                text,
                after,
                Role::Space.bit(),
                stop,
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        pre_tokenizers::{
            sequence::Sequence,
            split::{Split, SplitPattern},
            PreTokenizerWrapper,
        },
        OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior,
    };

    use super::{
        super::{
            pattern::Pattern,
            tests::{pseudo_random, staged},
            Pieces, Stage,
        },
        *,
    };

    const DIGITS: &str = r"\p{N}{1,3}";
    const ONE_DIGIT: &str = r"\p{N}";
    const RUNS: &str = "[一-龥぀-ゟ゠-ヿ]+";
    /// A bounded stage and a run stage whose classes overlap (`a` to `f`
    /// are in both), so that a digit group holds characters of the run
    /// class and a run meets a character of the digit class.
    const HEX: &str = "[0-9a-f]{1,3}";
    const LETTERS: &str = "[a-z]+";
    const TEMPLATE: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    const GPT4: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn stage(pattern: &str, pieces: Pieces) -> Stage {
        Stage {
            pattern: Pattern::parse(pattern).expect("parses"),
            pieces,
            digits: false,
        }
    }

    fn chain(patterns: &[&str]) -> Vec<Stage> {
        patterns
            .iter()
            .map(|p| stage(p, Pieces::MatchesAndGaps))
            .collect()
    }

    /// The pieces of the scanner, as byte ranges of `text`.
    fn scanned(scanner: &Scanner, text: &str) -> Vec<(usize, usize)> {
        let mut pieces = Vec::new();
        scanner.pieces(text, true, &mut |start, end| pieces.push((start, end)));
        let mut bytewise = Vec::new();
        scanner.pieces(text, false, &mut |start, end| bytewise.push((start, end)));
        assert_eq!(
            pieces, bytewise,
            "the word scans and the byte loop cut {text:?} alike"
        );
        pieces
    }

    /// The pieces of the engine: a `Sequence` of `Split`s over the text.
    fn engine(patterns: &[&str], text: &str) -> Vec<(usize, usize)> {
        let stages: Vec<(&str, SplitDelimiterBehavior)> = patterns
            .iter()
            .map(|p| (*p, SplitDelimiterBehavior::Isolated))
            .collect();
        engine_with(&stages, text)
    }

    /// The pieces of the engine: a `Sequence` of `Split`s over the text,
    /// each with its behavior.
    fn engine_with(stages: &[(&str, SplitDelimiterBehavior)], text: &str) -> Vec<(usize, usize)> {
        let sequence = Sequence::new(
            stages
                .iter()
                .map(|(p, behavior)| {
                    PreTokenizerWrapper::Split(
                        Split::new(SplitPattern::Regex((*p).to_owned()), *behavior, false)
                            .expect("split"),
                    )
                })
                .collect(),
        );
        let mut pretokenized = PreTokenizedString::from(text);
        sequence
            .pre_tokenize(&mut pretokenized)
            .expect("the engine splits");
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, (start, end), _)| (start, end))
            .collect()
    }

    /// The walk of a `MergedWithNext` stage glues each match to the text
    /// after it as the engine does: a run of line breaks before the byte-level
    /// regex, on texts with runs between words, at both ends, adjacent, and
    /// none.
    #[test]
    fn a_merged_stage_walks_as_the_engine_glues() {
        const LINE_BREAKS: &str = r"(?:\r?\n)+(?!\r?\n)";
        let stages = [
            stage(LINE_BREAKS, Pieces::MatchesWithNextGap),
            stage(GPT4, Pieces::MatchesAndGaps),
        ];
        let patterns = [
            (LINE_BREAKS, SplitDelimiterBehavior::MergedWithNext),
            (GPT4, SplitDelimiterBehavior::Isolated),
        ];
        for text in [
            "one\ntwo\r\nthree\rfour",
            "\n\nlead and trail\n",
            "\r\n\r\n\n",
            "a\n1\r\nb 22\n\n33 x",
            "    indented\n\tcode\n\n",
            "no break at all",
            "\n",
        ] {
            assert_eq!(
                staged(&stages, text),
                engine_with(&patterns, text),
                "text {text:?}"
            );
            assert_eq!(
                staged(&stages[..1], text),
                engine_with(&patterns[..1], text),
                "the stage alone; text {text:?}"
            );
        }
    }

    /// Text shaped for the chain's stages: digit runs of every length up to
    /// and past the bound, digits beyond ASCII, ideographs and kana with
    /// the punctuation and symbols their blocks hold, Thai and Indic
    /// letters with marks, punctuation before letters in every case,
    /// clitic spellings, and runs of spaces, tabs and line breaks before
    /// letters, digits, quotes and brackets.
    const PARTS: &[&str] = &[
        "1",
        "22",
        "333",
        "4444",
        "55555",
        "123456789012",
        "٣",
        "١٢٣٤",
        "๒๓",
        "²",
        "½",
        "中",
        "中文",
        "日本語の文章",
        "龥",
        "あ",
        "ん",
        "カタ",
        "カタ・カナ",
        "ー",
        "゛",
        "゜",
        "゠",
        "ヽ",
        "\u{3040}",
        "か\u{3099}",
        "ก",
        "กข",
        "ก\u{E31}\u{E48}",
        "क\u{93E}",
        "b\u{9BE}",
        "é",
        "e\u{301}",
        "한",
        "я",
        "α",
        " ",
        "  ",
        "   ",
        "\n",
        "\n\n",
        "\r\n",
        "\t",
        "\u{A0}",
        "\u{2028}",
        "\u{3000}",
        "\"",
        "'",
        "'s",
        "'S",
        "'ll",
        ".",
        ",",
        ":",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "-",
        "_",
        "/",
        "~",
        "@x",
        "#tag",
        "a",
        "Word",
        "ſ",
        "ß",
        "😀",
        "👍🏽",
        "\u{200D}",
        "\u{FE0F}",
        "\u{FFFD}",
        "\u{0}",
        "\u{7F}",
        "\u{AD}",
        "x",
    ];

    fn shaped_text(state: &mut u64, parts: usize) -> String {
        (0..parts)
            .map(|_| PARTS[(pseudo_random(state) % PARTS.len() as u64) as usize])
            .collect()
    }

    /// The walk by blocks cuts as the scanner alone does. The text is
    /// plain ASCII in stretches, shaped around the three rules in which
    /// the chain's template and the GPT-4 family's differ there (no
    /// clitic, a run of blanks before a digit stays whole, a control
    /// character is no punctuation), with a part beyond the pass's classes
    /// thrown in never, rarely or often, so the pass begins and hands back
    /// at every kind of piece start and at every block edge.
    #[test]
    fn the_walk_by_blocks_cuts_as_the_scanner_alone() {
        const PLAIN: &[&str] = &[
            "a",
            "b",
            "word",
            "Word",
            "x",
            "I",
            " ",
            " ",
            " ",
            "  ",
            "   ",
            "\t",
            "\t\t",
            "\n",
            "\n",
            "\r\n",
            "\n\n",
            " \n",
            "\n ",
            "\n  ",
            "\n\t",
            "1",
            "22",
            "333",
            "4444",
            "55555",
            "7777777",
            " 1",
            "  1",
            "   22",
            "\t1",
            "\n 1",
            "\n  1",
            ".",
            ",",
            "!",
            "?",
            "'",
            "'s",
            "'re",
            "'ll",
            "(",
            ")",
            "((",
            " (",
            " .",
            "._",
            "-",
            "--",
            "\"",
            ".\n",
            "!\r\n\r\n",
            "don't",
            "x.y",
            "_",
            "$",
            "+",
            "=",
            "\u{b}",
            "\u{c}",
        ];
        const ODD: &[&str] = &[
            "\u{1}",
            "\u{7f}",
            "\u{1}a",
            "\u{0}",
            "\u{e9}",
            "\u{65e5}\u{672c}",
            "\u{30ab}\u{30fb}",
            "\u{44f}",
            "\u{2014}",
            "\u{a0}",
            "\u{663}",
            "\u{3000}",
            "\u{1f600}",
        ];
        let shipped = Scanner::recognize(&chain(&[DIGITS, RUNS, TEMPLATE])).expect("the scanner");
        assert!(
            shipped.blocks,
            "the shipped chain reads plain ASCII by blocks"
        );
        let one = Scanner::recognize(&chain(&[ONE_DIGIT, RUNS, TEMPLATE])).expect("the scanner");
        assert!(one.blocks, "a one-digit first stage reads by blocks too");
        let overlapping =
            Scanner::recognize(&chain(&[HEX, LETTERS, TEMPLATE])).expect("the scanner");
        assert!(
            !overlapping.blocks,
            "a digit class holding a-f keeps the scanner"
        );
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for scanner in [&shipped, &one] {
            for round in 0..3000usize {
                let odd_every = [0u64, 64, 8][round % 3];
                let mut text = String::new();
                // One round in eight is long, so that its walks double
                // further.
                let parts = (150 + round % 250) * if round % 8 == 7 { 8 } else { 1 };
                for _ in 0..parts {
                    let pick = pseudo_random(&mut state);
                    let pool = if odd_every != 0 && (pick >> 32).is_multiple_of(odd_every) {
                        ODD
                    } else {
                        PLAIN
                    };
                    text.push_str(pool[(pick % pool.len() as u64) as usize]);
                }
                let mut alone = Vec::new();
                scanner.pieces_as::<false, _>(&text, &mut |start, end| alone.push((start, end)));
                let mut by_blocks = Vec::new();
                scanner.pieces_by_blocks::<false, _>(&text, &mut |start, end| {
                    by_blocks.push((start, end));
                });
                assert_eq!(by_blocks, alone, "text {text:?}");
                by_blocks.clear();
                scanner.pieces_by_blocks::<true, _>(&text, &mut |start, end| {
                    by_blocks.push((start, end));
                });
                assert_eq!(by_blocks, alone, "with the word scans; text {text:?}");
            }
        }
        // The pass itself cuts plain text up to its last blocks.
        let plain = "word, 12  345 (x) don't\n  go ".repeat(40);
        let mut cut = 0;
        let open = block::chain_pieces(plain.as_bytes(), 0, 3, &mut |_, _| cut += 1);
        assert!(
            cut > 100 && open + 3 * block::BLOCK >= plain.len(),
            "{cut} pieces, open at {open} of {}",
            plain.len()
        );
    }

    /// A run of blanks behind a line break that fills the whole block behind
    /// its first blank: whether that blank starts a piece hangs on the byte
    /// that ends the run, and the pass then reads it past the next block
    /// (`Edge::starts` of `block`). The generated texts above hold no run
    /// that long. Here the break stands at every offset of a block, the run
    /// is from just under one block to three blocks long, and it ends in
    /// each kind of byte the pass can meet there: the text's end; a letter;
    /// a digit, which the chain's template never sees, alone and before a
    /// line break (the run ends at the digit all the same); a control
    /// character, whose block is the scanner's; a line break of either
    /// kind, with which the run is one piece from its first break on; and
    /// beyond ASCII, where the pass declines, a letter, a digit, and
    /// whitespace before a line break (the run goes on through it). The
    /// scanner alone and the walk by blocks cut each text as the engine
    /// does.
    #[test]
    fn a_block_of_blanks_behind_a_line_break_is_cut_as_the_engine_cuts_it() {
        const ENDS: &[&str] = &[
            "", "x", "1", "1\n", "\u{1}", "\n", "\r", "\u{e9}", "\u{663}", "\u{a0}\n",
        ];
        const BLANKS: [usize; 5] = [
            block::BLOCK - 1,
            block::BLOCK,
            block::BLOCK + 1,
            2 * block::BLOCK,
            3 * block::BLOCK,
        ];
        let words = "word ".repeat(40);
        for patterns in [[DIGITS, RUNS, TEMPLATE], [ONE_DIGIT, RUNS, TEMPLATE]] {
            let scanner = Scanner::recognize(&chain(&patterns)).expect("the scanner");
            assert!(scanner.blocks, "the chain reads plain ASCII by blocks");
            for shift in 0..block::BLOCK {
                let lead = &words[..2 * block::BLOCK + shift];
                for blanks in BLANKS {
                    let run = " ".repeat(blanks);
                    for ends_in in ENDS {
                        for tail in ["", words.as_str()] {
                            let text = format!("{lead}\n{run}{ends_in}{tail}");
                            let cut = engine(&patterns, &text);
                            let mut alone = Vec::new();
                            scanner.pieces_as::<false, _>(&text, &mut |start, end| {
                                alone.push((start, end));
                            });
                            assert_eq!(alone, cut, "the scanner alone; text {text:?}");
                            let mut by_blocks = Vec::new();
                            scanner.pieces_by_blocks::<false, _>(&text, &mut |start, end| {
                                by_blocks.push((start, end));
                            });
                            assert_eq!(by_blocks, cut, "by blocks; text {text:?}");
                            by_blocks.clear();
                            scanner.pieces_by_blocks::<true, _>(&text, &mut |start, end| {
                                by_blocks.push((start, end));
                            });
                            assert_eq!(by_blocks, cut, "with the word scans; text {text:?}");
                        }
                    }
                }
            }
        }
    }

    /// A text is read by blocks only where the pass serves it: from the
    /// pass's bound on, with an ASCII stretch at one of the places the text
    /// is asked at. A text in another script keeps the scanner's own loop,
    /// and a short one that takes no word scans is not asked at all.
    #[test]
    fn a_text_is_read_by_blocks_only_with_an_ascii_stretch() {
        let scanner = Scanner::recognize(&chain(&[DIGITS, RUNS, TEMPLATE])).expect("the scanner");
        let ascii = "word ".repeat(block::MIN_BYTES);
        assert!(scanner.by_blocks(&ascii, true));
        assert!(!scanner.by_blocks(&ascii[..block::MIN_BYTES - 1], true));
        // A text under one step between two places is asked at its opening
        // alone: one that takes no word scans has answered, and is not
        // asked.
        assert!(ascii.len() < block::SAMPLE);
        assert!(!scanner.by_blocks(&ascii, false));
        let cyrillic = "слово ".repeat(block::MIN_BYTES);
        assert!(cyrillic.len() >= block::SAMPLE);
        assert!(!scanner.by_blocks(&cyrillic, true));
        assert!(!scanner.by_blocks(&cyrillic, false));
        // The stretch may stand at any place a longer text is asked at,
        // whether the text takes the word scans or not.
        let late = format!("{}{ascii}", "я".repeat(block::SAMPLE / 2));
        assert!(scanner.by_blocks(&late, true));
        assert!(scanner.by_blocks(&late, false));
        // A chain the pass does not cut as keeps the scanner on any text.
        let overlapping =
            Scanner::recognize(&chain(&[HEX, LETTERS, TEMPLATE])).expect("the scanner");
        assert!(!overlapping.by_blocks(&ascii, true));
    }

    #[test]
    fn the_chain_takes_the_scanner_and_other_stage_lists_do_not() {
        let scanner = Scanner::recognize(&chain(&[DIGITS, RUNS, TEMPLATE]))
            .expect("the chain takes the scanner");
        assert_eq!(scanner.digits, 3);
        let one = Scanner::recognize(&chain(&[ONE_DIGIT, RUNS, TEMPLATE]))
            .expect("a one-digit first stage takes the scanner");
        assert_eq!(one.digits, 1);
        let overlapping = Scanner::recognize(&chain(&[HEX, LETTERS, TEMPLATE]))
            .expect("overlapping classes take the scanner");
        assert_eq!(overlapping.digits, 3);
        // The gain of the word scans rests on these: a class change that
        // kept the ids and lost the scans fails here, not in a benchmark.
        assert!(scanner.word_scans, "the shipped shape takes the word scans");
        assert!(
            !overlapping.word_scans,
            "a digit class holding a-f keeps the byte loop"
        );
        for role in [Role::PunctLetter, Role::Word] {
            assert_eq!(
                scanner.roles.scan_of(role.bit()),
                WordScan::LETTERS,
                "{role:?}"
            );
        }
        for role in [Role::TrailSpace, Role::Space] {
            assert_eq!(
                scanner.roles.scan_of(role.bit()),
                WordScan::SPACES,
                "{role:?}"
            );
        }
        // Fewer or more stages, the stages in another order, the GPT-4
        // family's template as the third stage, a bounded second stage, a
        // stage that drops its gaps, a stage that drops its matches.
        for stages in [
            chain(&[DIGITS, TEMPLATE]),
            chain(&[TEMPLATE]),
            chain(&[DIGITS, RUNS, TEMPLATE, GPT4]),
            chain(&[RUNS, DIGITS, TEMPLATE]),
            chain(&[DIGITS, TEMPLATE, RUNS]),
            chain(&[DIGITS, RUNS, GPT4]),
            chain(&[DIGITS, r"[一-龥]{1,3}", TEMPLATE]),
            vec![
                stage(DIGITS, Pieces::MatchesAndGaps),
                stage(RUNS, Pieces::MatchesAndGaps),
                stage(TEMPLATE, Pieces::Matches),
            ],
            vec![
                stage(DIGITS, Pieces::Gaps),
                stage(RUNS, Pieces::MatchesAndGaps),
                stage(TEMPLATE, Pieces::MatchesAndGaps),
            ],
        ] {
            assert!(
                Scanner::recognize(&stages).is_none(),
                "{} stages",
                stages.len()
            );
        }
    }

    #[test]
    fn the_scanner_agrees_with_the_stage_programs_and_the_engine_on_shaped_text() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for patterns in [
            [DIGITS, RUNS, TEMPLATE],
            [ONE_DIGIT, RUNS, TEMPLATE],
            [HEX, LETTERS, TEMPLATE],
        ] {
            let stages = chain(&patterns);
            let scanner = Scanner::recognize(&stages).expect("the scanner");
            for round in 0..3000 {
                let text = shaped_text(&mut state, 1 + round % 40);
                let ours = scanned(&scanner, &text);
                assert_eq!(
                    ours,
                    staged(&stages, &text),
                    "the programs; chain {patterns:?} text {text:?}"
                );
                assert_eq!(
                    ours,
                    engine(&patterns, &text),
                    "the engine; chain {patterns:?} text {text:?}"
                );
            }
        }
    }

    /// With `a` to `f` in both the digit and the run class, a digit group
    /// is cut by the run stage inside it and a run ends at a character of
    /// the digit class, as the stages applied in turn cut; the pieces here
    /// are the ones only those two cuts produce (`1a2` reads `1a|2` without
    /// the first, `xd1` reads `xd|1` without the second).
    #[test]
    fn overlapping_classes_cut_a_group_by_the_run_class_and_end_a_run_at_the_digit_class() {
        let patterns = [HEX, LETTERS, TEMPLATE];
        let stages = chain(&patterns);
        let scanner = Scanner::recognize(&stages).expect("the scanner");
        for (text, pieces) in [
            ("1a2", vec![(0, 1), (1, 2), (2, 3)]),
            ("ab12cd", vec![(0, 2), (2, 3), (3, 4), (4, 6)]),
            ("xd1", vec![(0, 1), (1, 2), (2, 3)]),
            ("abcdef", vec![(0, 3), (3, 6)]),
        ] {
            let ours = scanned(&scanner, text);
            assert_eq!(ours, pieces, "text {text:?}");
            assert_eq!(ours, staged(&stages, text), "the programs; text {text:?}");
            assert_eq!(ours, engine(&patterns, text), "the engine; text {text:?}");
        }
    }

    /// A run of the second stage's class whose characters are all letters
    /// of the template, and a group of the first stage that holds no
    /// character of `GROUP_CUTS`, are one piece without the later stages'
    /// read of them; the stages still cut every other run and group: a run
    /// with a punctuation mark or a symbol of its block in it, a run that a
    /// punctuation lead opens, a group with a letter, a punctuation mark or
    /// a space in it.
    #[test]
    fn a_run_of_letters_and_a_plain_group_are_one_piece_and_the_stages_cut_the_others() {
        // A punctuation lead that is a letter too, with a run class that
        // holds it: `ab` is the first alternative's match inside the run.
        const LEAD_A: &str = "[!a][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
        for (patterns, text, pieces) in [
            ([DIGITS, RUNS, TEMPLATE], "日本語の文章", vec![(0, 18)]),
            ([DIGITS, RUNS, TEMPLATE], "か\u{3099}き", vec![(0, 9)]),
            (
                [DIGITS, RUNS, TEMPLATE],
                "カタ・カナ",
                vec![(0, 6), (6, 9), (9, 15)],
            ),
            (
                [DIGITS, RUNS, TEMPLATE],
                "あ゛い",
                vec![(0, 3), (3, 6), (6, 9)],
            ),
            ([DIGITS, RUNS, TEMPLATE], "123", vec![(0, 3)]),
            ([DIGITS, RUNS, TEMPLATE], "١٢٣٤", vec![(0, 6), (6, 8)]),
            ([DIGITS, "[a-zé]+", LEAD_A], "abé", vec![(0, 2), (2, 4)]),
            ([HEX, RUNS, TEMPLATE], "1a2", vec![(0, 2), (2, 3)]),
            // A group class that holds a punctuation mark and a space: each
            // is a match of the template inside the group. The full stop is
            // a punctuation lead too; the ideographic one is not.
            (
                [r"[0-9 .]{1,3}", RUNS, TEMPLATE],
                "1.2",
                vec![(0, 1), (1, 2), (2, 3)],
            ),
            (
                [r"[0-9 .]{1,3}", RUNS, TEMPLATE],
                "1 2",
                vec![(0, 1), (1, 2), (2, 3)],
            ),
            (
                ["[0-9。]{1,3}", RUNS, TEMPLATE],
                "1。2",
                vec![(0, 1), (1, 4), (4, 5)],
            ),
        ] {
            let stages = chain(&patterns);
            let scanner = Scanner::recognize(&stages).expect("the scanner");
            let ours = scanned(&scanner, text);
            assert_eq!(ours, pieces, "chain {patterns:?} text {text:?}");
            assert_eq!(
                ours,
                staged(&stages, text),
                "the programs; chain {patterns:?} text {text:?}"
            );
            assert_eq!(
                ours,
                engine(&patterns, text),
                "the engine; chain {patterns:?} text {text:?}"
            );
        }
    }

    #[test]
    fn the_scanner_agrees_with_the_engine_on_prose_code_and_scripts() {
        let texts = [
            "The year 2024 had 366 days; 1,234,567 people and 3.14159 as pi.",
            "中文和English混合, 日本語のテキスト、カタカナ・ひらがな。한국어도 있다.",
            "deepseek_v3.2-exp: x = foo(a, b) + 42 * 7 // 3; return [x, y]\n\tif (x > 0) {\n\n\t\treturn;\n\t}\n",
            "ภาษาไทยไม่มีช่องว่างระหว่างคำ และตัวเลข ๑๒๓ กับ 456",
            "I'm sure they'RE here, it'S fine, we'll see, you'd know, I've, don't",
            "a  b   c\n\n\nd \n e\r\n\r\nf  \n  g 1 22 333 4444 55555 666666",
            "   leading and trailing   ",
            "snake_case-kebab.dot/slash\\back `tick` ~tilde~ [br] {cu} (pa) <an> @at #hash",
            "emoji 😀👍🏽 and ✨ and \u{FE0F} and \u{200D} and 中😀中 and 1😀2",
            "e\u{301}cole ka\u{93E} th\u{E31}\u{E48} b\u{9BE} か\u{3099}き\u{309A}",
            "２０２４年の数字と全角の１２３、半角の123、混在の1２3",
            "",
            "\n",
            " ",
            "'",
            "7",
            "中",
            "・",
        ];
        let patterns = [DIGITS, RUNS, TEMPLATE];
        let stages = chain(&patterns);
        let scanner = Scanner::recognize(&stages).expect("the scanner");
        for text in texts {
            let ours = scanned(&scanner, text);
            assert_eq!(ours, staged(&stages, text), "the programs; text {text:?}");
            assert_eq!(ours, engine(&patterns, text), "the engine; text {text:?}");
        }
    }

    #[test]
    fn long_runs_do_not_recurse_per_character() {
        let patterns = [DIGITS, RUNS, TEMPLATE];
        let scanner = Scanner::recognize(&chain(&patterns)).expect("the scanner");
        let spaces = " ".repeat(2_000_000);
        assert_eq!(scanned(&scanner, &spaces), vec![(0, 2_000_000)]);
        let digits = "7".repeat(300_000);
        assert_eq!(scanned(&scanner, &digits).len(), 100_000);
        let ideographs = "中".repeat(100_000);
        assert_eq!(scanned(&scanner, &ideographs), vec![(0, 300_000)]);
        let mixed = "\n".repeat(100_000) + &" ".repeat(100_000) + "x1中";
        assert_eq!(scanned(&scanner, &mixed), engine(&patterns, &mixed));
    }
}
