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
//! for every decision, as it is for the engine.
//!
//! The shape is recognized from the parsed patterns, with whatever classes
//! stand in its places and whatever bound the digit rule has; the classes
//! are the parser's, so the scan and the programs cannot read a Unicode
//! category differently, and the programs stay as the second matcher the
//! tests hold the scan against.

use super::{
    pattern::{concat, next_char, plain_class, previous_boundary, repeat, Class, Node},
    Stage,
};

/// A part of the template that one class fills: the bit of
/// [`Scanner::ascii`] and the slot of [`Scanner::roles`] that hold it.
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

/// The roles, so many slots of [`Scanner::roles`].
const ROLES: usize = 14;

/// The template's roles in the order [`Part::Template`] lists their classes.
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
    fn bit(self) -> u16 {
        1 << (self as u16)
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
            Node::ClassRepeat { class, min: 1, max } => Some(Self::Run {
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
    /// The classes of the three stages, each role's under its slot.
    classes: Vec<Class>,
    /// Per ASCII code point, a bit per [`Role`] whose class holds it.
    ascii: [u16; 128],
    /// The class of each role, read for characters beyond ASCII.
    roles: [usize; ROLES],
    /// The most characters of the digit class one group holds.
    digits: u32,
}

impl Scanner {
    /// The scanner of `stages`, when they are a chain of a bounded run, an
    /// unbounded run and the template, each keeping its gaps; `None` for
    /// every other stage list, which the stage patterns match in turn.
    pub(super) fn recognize(stages: &[Stage]) -> Option<Self> {
        let [first, second, third] = stages else {
            return None;
        };
        if !(first.keep_gaps && second.keep_gaps && third.keep_gaps) {
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
        let mut classes = Vec::with_capacity(ROLES);
        let mut roles = [0usize; ROLES];
        let mut take = |role: Role, class: &Class| {
            roles[role as usize] = classes.len();
            classes.push(class.clone());
        };
        take(Role::Digit, &first.pattern.classes()[*digit]);
        take(Role::Run, &second.pattern.classes()[*run]);
        for (role, class) in TEMPLATE_ROLES.iter().zip(template) {
            take(*role, &third.pattern.classes()[*class]);
        }
        let mut ascii = [0u16; 128];
        for (code, bits) in ascii.iter_mut().enumerate() {
            let c = char::from(code as u8);
            for (slot, class) in roles.iter().enumerate() {
                if classes[*class].contains(c) {
                    *bits |= 1 << slot;
                }
            }
        }
        Some(Self {
            classes,
            ascii,
            roles,
            digits: *digits,
        })
    }

    /// Whether `c` is in the class of `role`.
    #[inline(always)]
    fn has(&self, role: Role, c: char) -> bool {
        let code = u32::from(c);
        if code < 128 {
            self.ascii[code as usize] & role.bit() != 0
        } else {
            self.classes[self.roles[role as usize]].contains(c)
        }
    }

    /// The end of the run of `role`'s class that starts at `pos`.
    #[inline(always)]
    fn run_end(&self, text: &str, mut pos: usize, role: Role) -> usize {
        while let Some((c, len)) = next_char(text, pos) {
            if !self.has(role, c) {
                break;
            }
            pos += len;
        }
        pos
    }

    /// Calls `f` with each piece of `text`, in order; the pieces tile the
    /// text, as three `Split` stages that keep their gaps leave no character
    /// out.
    pub(super) fn pieces<F: FnMut(&str)>(&self, text: &str, f: &mut F) {
        let mut pos = 0;
        while let Some((c, len)) = next_char(text, pos) {
            let after = pos + len;
            let end = if self.has(Role::Digit, c) {
                // The first stage: a group of up to the bound; the stages
                // after it see the group on its own.
                let mut end = after;
                let mut count = 1;
                while count < self.digits {
                    match next_char(text, end) {
                        Some((d, n)) if self.has(Role::Digit, d) => {
                            end += n;
                            count += 1;
                        }
                        _ => break,
                    }
                }
                self.group_pieces(&text[pos..end], f);
                end
            } else if self.has(Role::Run, c) {
                // The second stage, inside a gap of the first: a run that
                // the next digit ends, since the first stage cut there.
                let mut end = after;
                while let Some((d, n)) = next_char(text, end) {
                    if !self.has(Role::Run, d) || self.has(Role::Digit, d) {
                        break;
                    }
                    end += n;
                }
                self.template_pieces(&text[pos..end], f);
                end
            } else {
                // A gap of both: the stretch to the next digit or run
                // character, which the template cuts.
                let mut end = after;
                while let Some((d, n)) = next_char(text, end) {
                    if self.has(Role::Digit, d) || self.has(Role::Run, d) {
                        break;
                    }
                    end += n;
                }
                self.template_pieces(&text[pos..end], f);
                end
            };
            pos = end;
        }
    }

    /// The second and third stages over one group of the first: the runs of
    /// the second stage's class in it and the stretches between them, each
    /// cut by the template. For a digit class the template cuts nothing and
    /// the group is one piece; the stages are applied all the same, so the
    /// pieces are theirs for any classes.
    fn group_pieces<F: FnMut(&str)>(&self, group: &str, f: &mut F) {
        let mut pos = 0;
        while let Some((c, len)) = next_char(group, pos) {
            let run = self.has(Role::Run, c);
            let mut piece_end = pos + len;
            while let Some((d, n)) = next_char(group, piece_end) {
                if self.has(Role::Run, d) != run {
                    break;
                }
                piece_end += n;
            }
            self.template_pieces(&group[pos..piece_end], f);
            pos = piece_end;
        }
    }

    /// The template's pieces of one piece of the stages before it: its
    /// matches, and the text between them as pieces of their own.
    fn template_pieces<F: FnMut(&str)>(&self, piece: &str, f: &mut F) {
        let mut gap = 0;
        let mut pos = 0;
        while let Some((c, len)) = next_char(piece, pos) {
            match self.template_match(piece, pos, c, len) {
                Some(end) => {
                    if gap < pos {
                        f(&piece[gap..pos]);
                    }
                    f(&piece[pos..end]);
                    pos = end;
                    gap = end;
                }
                None => pos += len,
            }
        }
        if gap < piece.len() {
            f(&piece[gap..]);
        }
    }

    /// The end of the template's match that starts at `pos`, where `c` of
    /// `len` bytes stands, if one does: the first of the six alternatives
    /// that matches there, as the engine picks it.
    fn template_match(&self, text: &str, pos: usize, c: char, len: usize) -> Option<usize> {
        let after = pos + len;
        // 1. Punctuation followed by letters: the letter run after it.
        if self.has(Role::PunctLead, c) {
            if let Some((d, n)) = next_char(text, after) {
                if self.has(Role::PunctLetter, d) {
                    return Some(self.run_end(text, after + n, Role::PunctLetter));
                }
            }
        }
        // 2. A letter run, with the lead-in when a letter follows it.
        if self.has(Role::Lead, c) {
            if let Some((d, n)) = next_char(text, after) {
                if self.has(Role::Word, d) {
                    return Some(self.run_end(text, after + n, Role::Word));
                }
            }
        }
        if self.has(Role::Word, c) {
            return Some(self.run_end(text, after, Role::Word));
        }
        // 3. A punctuation run, with the space when punctuation follows it,
        //    and the line breaks after the run.
        if self.has(Role::PunctSpace, c) {
            if let Some((d, n)) = next_char(text, after) {
                if self.has(Role::Punct, d) {
                    let end = self.run_end(text, after + n, Role::Punct);
                    return Some(self.run_end(text, end, Role::PunctBreak));
                }
            }
        }
        if self.has(Role::Punct, c) {
            let end = self.run_end(text, after, Role::Punct);
            return Some(self.run_end(text, end, Role::PunctBreak));
        }
        // 4. Whitespace ending in line breaks: the greedy whitespace run
        //    gives back characters until a line break follows it.
        let mut end = self.run_end(text, pos, Role::BreakSpace);
        loop {
            if let Some((d, n)) = next_char(text, end) {
                if self.has(Role::Break, d) {
                    return Some(self.run_end(text, end + n, Role::Break));
                }
            }
            if end == pos {
                break;
            }
            end = previous_boundary(text, end);
        }
        // 5. Whitespace before a non-space: the greedy run gives back its
        //    last character while a non-space follows it, down to one.
        if self.has(Role::TrailSpace, c) {
            let mut end = self.run_end(text, after, Role::TrailSpace);
            loop {
                match next_char(text, end) {
                    Some((d, _)) if self.has(Role::NotTrail, d) => {}
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
        if self.has(Role::Space, c) {
            return Some(self.run_end(text, after, Role::Space));
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
        super::{pattern::Pattern, Stage},
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

    fn stage(pattern: &str, keep_gaps: bool) -> Stage {
        Stage {
            pattern: Pattern::parse(pattern).expect("parses"),
            keep_gaps,
        }
    }

    fn chain(patterns: &[&str]) -> Vec<Stage> {
        patterns.iter().map(|p| stage(p, true)).collect()
    }

    /// The pieces of the scanner, as byte ranges of `text`.
    fn scanned(scanner: &Scanner, text: &str) -> Vec<(usize, usize)> {
        let mut pieces = Vec::new();
        scanner.pieces(text, &mut |piece: &str| {
            let start = piece.as_ptr() as usize - text.as_ptr() as usize;
            pieces.push((start, start + piece.len()));
        });
        pieces
    }

    /// The pieces of the stage patterns run in turn, each over the pieces
    /// of the one before (the recursion of `NativeEncoder::pieces`).
    fn staged(stages: &[Stage], text: &str) -> Vec<(usize, usize)> {
        fn walk(stages: &[Stage], text: &str, offset: usize, out: &mut Vec<(usize, usize)>) {
            if text.is_empty() {
                return;
            }
            let Some((stage, rest)) = stages.split_first() else {
                out.push((offset, offset + text.len()));
                return;
            };
            let mut cursor = 0;
            for (start, end) in stage.pattern.find_iter(text) {
                if stage.keep_gaps && cursor < start {
                    walk(rest, &text[cursor..start], offset + cursor, out);
                }
                walk(rest, &text[start..end], offset + start, out);
                cursor = end;
            }
            if stage.keep_gaps && cursor < text.len() {
                walk(rest, &text[cursor..], offset + cursor, out);
            }
        }
        let mut out = Vec::new();
        walk(stages, text, 0, &mut out);
        out
    }

    /// The pieces of the engine: a `Sequence` of `Split`s over the text.
    fn engine(patterns: &[&str], text: &str) -> Vec<(usize, usize)> {
        let sequence = Sequence::new(
            patterns
                .iter()
                .map(|p| {
                    PreTokenizerWrapper::Split(
                        Split::new(
                            SplitPattern::Regex((*p).to_owned()),
                            SplitDelimiterBehavior::Isolated,
                            false,
                        )
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

    fn pseudo_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
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
        // Fewer or more stages, the stages in another order, the GPT-4
        // family's template as the third stage, a bounded second stage, a
        // stage that drops its gaps.
        for stages in [
            chain(&[DIGITS, TEMPLATE]),
            chain(&[TEMPLATE]),
            chain(&[DIGITS, RUNS, TEMPLATE, GPT4]),
            chain(&[RUNS, DIGITS, TEMPLATE]),
            chain(&[DIGITS, TEMPLATE, RUNS]),
            chain(&[DIGITS, RUNS, GPT4]),
            chain(&[DIGITS, r"[一-龥]{1,3}", TEMPLATE]),
            vec![
                stage(DIGITS, true),
                stage(RUNS, true),
                stage(TEMPLATE, false),
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
