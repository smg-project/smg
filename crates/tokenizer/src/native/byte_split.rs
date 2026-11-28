//! The byte-level split, the pattern the `ByteLevel` pre-tokenizer applies
//! itself, matched by a scanner:
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! with whatever classes stand in those places and whatever clitics lead
//! it, as long as they all begin with one class. The pattern has no nested
//! group and no bounded run, so the template scanners decline it, and as the
//! backtracking program every piece is interpreted op by op. Decided
//! alternative by alternative a long warm encode costs a third to a half
//! fewer instructions on the vocabularies that carry the split (counted),
//! and the pieces are the same:
//!
//! 1. a clitic: the first leading alternative whose characters follow in
//!    order;
//! 2. a letter run, with the lead-in when a letter follows it;
//! 3. a digit run, with the lead-in when a digit follows it;
//! 4. a punctuation run, with the lead-in when punctuation follows it;
//! 5. whitespace before a non-space: the run short of its last character,
//!    or whole at the end of the text, when it has a character to give;
//! 6. the whitespace run.
//!
//! Each is what the engine's backtracking arrives at: a greedy optional
//! lead-in is kept only when the run after it can start, a run that ends its
//! alternative never gives a character back, a greedy run before a negative
//! lookahead gives back characters until the lookahead holds, and the first
//! alternative to reach the end of the pattern wins. The classes are the
//! parser's ([`Roles`] reads them), so the scanner and the program cannot
//! read a category differently, and the program stays as the matcher the
//! tests hold the scanner against. Its iterator ([`Matches`]) yields what the
//! pattern's own yields by the program, and the walk over a pipeline's stages
//! takes it in that one's place.

use super::{
    pattern::{clitic_match, concat, plain_class, previous_boundary, repeat, Class, Node},
    scan::{Bits, Roles},
};

/// A part of the pattern that one class fills: its bit in a character's
/// role bits.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Role {
    /// What every clitic begins with (`'`).
    Clitic,
    /// The optional lead-in of a letter run (` ?`).
    LetterPrefix,
    /// A letter run (`\p{L}+`).
    Letter,
    /// The optional lead-in of a digit run (` ?`).
    DigitPrefix,
    /// A digit run (`\p{N}+`).
    Digit,
    /// The optional lead-in of a punctuation run (` ?`).
    PunctPrefix,
    /// A punctuation run (`[^\s\p{L}\p{N}]+`).
    Punct,
    /// Whitespace that leaves its last character to what follows (`\s+`
    /// before `(?!\S)`).
    TrailSpace,
    /// What may not follow that whitespace (`\S`).
    NotTrail,
    /// A whitespace run (`\s+`).
    Space,
}

/// The roles, in the order of their bits.
const ROLES: [Role; 10] = [
    Role::Clitic,
    Role::LetterPrefix,
    Role::Letter,
    Role::DigitPrefix,
    Role::Digit,
    Role::PunctPrefix,
    Role::Punct,
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

/// The scanner; see the module documentation.
#[derive(Debug, Clone)]
pub(super) struct Scanner {
    roles: Roles,
    /// The clitic alternatives, each the classes of its characters in order.
    clitics: Vec<Vec<usize>>,
}

/// The lead-in and the run class of a run alternative, `lead?run+`.
fn led_run(node: &Node) -> Option<(usize, usize)> {
    let [lead, run] = concat(node)? else {
        return None;
    };
    Some((repeat(lead, 0, 1)?, repeat(run, 1, u32::MAX)?))
}

impl Scanner {
    /// The scanner of `root`, when it is the byte-level split's form; `None`
    /// for every other shape.
    pub(super) fn recognize(root: &Node, classes: &[Class]) -> Option<Self> {
        let Node::Alt { alternatives, .. } = root else {
            return None;
        };
        let (clitics, [letters, digits, punct, trail, space]) = alternatives.split_last_chunk()?;
        let clitics = clitics
            .iter()
            .map(|alternative| match alternative {
                Node::Class(class) => Some(vec![*class]),
                Node::Concat(nodes) => nodes.iter().map(plain_class).collect(),
                _ => None,
            })
            .collect::<Option<Vec<Vec<usize>>>>()?;
        // One role says where a clitic may begin, so the clitics must all
        // begin with the class of the first one.
        let opening = *clitics.first()?.first()?;
        let open_alike = clitics.iter().all(|clitic| {
            clitic
                .first()
                .is_some_and(|&class| classes[class].same_set(&classes[opening]))
        });
        if !open_alike {
            return None;
        }
        let (letter_prefix, letter) = led_run(letters)?;
        let (digit_prefix, digit) = led_run(digits)?;
        let (punct_prefix, punct) = led_run(punct)?;
        let [trail_space, Node::NotAhead(not_trail)] = concat(trail)? else {
            return None;
        };
        let trail_space = repeat(trail_space, 1, u32::MAX)?;
        let space = repeat(space, 1, u32::MAX)?;
        let role_classes = ROLES.map(|role| {
            &classes[match role {
                Role::Clitic => opening,
                Role::LetterPrefix => letter_prefix,
                Role::Letter => letter,
                Role::DigitPrefix => digit_prefix,
                Role::Digit => digit,
                Role::PunctPrefix => punct_prefix,
                Role::Punct => punct,
                Role::TrailSpace => trail_space,
                Role::NotTrail => *not_trail,
                Role::Space => space,
            }]
        });
        Some(Self {
            roles: Roles::build(&role_classes),
            clitics,
        })
    }

    /// The number of clitic alternatives.
    #[cfg(test)]
    pub(super) fn clitics(&self) -> usize {
        self.clitics.len()
    }

    /// The end of a run alternative, `lead?run+`, at a character that holds
    /// the roles `c` and ends at `after`: with the lead-in when a character
    /// of the run follows it, else from the character itself when it is one
    /// of the run.
    #[inline(always)]
    fn led_run_end(
        &self,
        text: &str,
        c: Bits,
        after: usize,
        lead: Role,
        run: Role,
    ) -> Option<usize> {
        if c & lead.bit() != 0 {
            if let Some((d, n)) = self.roles.at(text, after) {
                if d & run.bit() != 0 {
                    return Some(self.roles.run_end(text, after + n, run.bit()));
                }
            }
        }
        (c & run.bit() != 0).then(|| self.roles.run_end(text, after, run.bit()))
    }

    /// The end of the piece that starts at `pos`, if one does: the first of
    /// the alternatives that matches there, as the engine picks it.
    #[inline(always)]
    pub(super) fn match_at(&self, classes: &[Class], text: &str, pos: usize) -> Option<usize> {
        let (c, len) = self.roles.at(text, pos)?;
        let after = pos + len;
        // 1. A clitic: the first alternative whose characters follow in order.
        if c & Role::Clitic.bit() != 0 {
            if let Some(end) = clitic_match(classes, text, pos, &self.clitics) {
                return Some(end);
            }
        }
        // 2 to 4. A run of letters, of digits, of punctuation, each with its
        //    lead-in when a character of the run follows it.
        for (lead, run) in [
            (Role::LetterPrefix, Role::Letter),
            (Role::DigitPrefix, Role::Digit),
            (Role::PunctPrefix, Role::Punct),
        ] {
            if let Some(end) = self.led_run_end(text, c, after, lead, run) {
                return Some(end);
            }
        }
        // 5. Whitespace before a non-space: the greedy run gives back its last
        //    character while a non-space follows it, down to one.
        if c & Role::TrailSpace.bit() != 0 {
            let mut end = self.roles.run_end(text, after, Role::TrailSpace.bit());
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
        // 6. The whitespace run.
        if c & Role::Space.bit() != 0 {
            return Some(self.roles.run_end(text, after, Role::Space.bit()));
        }
        None
    }

    /// The matches in `text`, leftmost first, each continuing after the
    /// last: what the pattern's iterator yields by the program.
    #[inline(always)]
    pub(super) fn matches<'s, 't>(
        &'s self,
        classes: &'s [Class],
        text: &'t str,
    ) -> Matches<'s, 't> {
        Matches {
            scanner: self,
            classes,
            text,
            pos: 0,
        }
    }
}

/// Iterator over the matches of a [`Scanner`] in a text.
pub(super) struct Matches<'s, 't> {
    scanner: &'s Scanner,
    classes: &'s [Class],
    text: &'t str,
    pos: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize)> {
        while self.pos < self.text.len() {
            let start = self.pos;
            if let Some(end) = self.scanner.match_at(self.classes, self.text, start) {
                debug_assert!(end > start, "no alternative matches empty");
                self.pos = end;
                return Some((start, end));
            }
            // No alternative starts with this character (the split's own
            // classes hold every character; other classes may not): the
            // search passes over it, as the program's does.
            self.pos += self.text[start..].chars().next().map_or(1, char::len_utf8);
        }
        None
    }
}
