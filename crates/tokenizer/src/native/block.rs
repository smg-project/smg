//! The piece starts of the GPT-4 family's pre-tokenizer template on ASCII
//! text, a block of 64 bytes at a time.
//!
//! The template's scanner (`pattern::Scanner`) decides one piece at a
//! position: a role read, a dispatch over the alternatives, a run loop,
//! every branch following the text. This pass decides the starts of a whole
//! block at once. Each class of the template becomes one bit per byte of
//! the block ([`Masks`]), and whether a byte starts a piece is a few word
//! operations on those masks ([`Edge::starts`]), because every alternative
//! comes down to a rule on a character's class and its neighbours':
//!
//! - a punctuation run starts a piece unless a literal space stands before
//!   it (the space starts that piece), and takes the line breaks behind it;
//! - a whitespace run starts a piece at its first character, at its last
//!   blank when a visible character follows it, and behind its last line
//!   break when it goes on from there;
//! - a letter run starts a piece unless its lead-in does (a blank, or a
//!   punctuation character that starts a piece itself); a clitic is such an
//!   apostrophe, and the letter behind a clitic starts a piece;
//! - a digit run starts a piece at every digit the rule's bound apart.
//!
//! A blank is whitespace that is no line break. The rules are the
//! template's only where its classes hold, of ASCII, the plain sets below:
//! the scanner checks that before it takes the pass
//! (`Scanner::reads_ascii_as_blocks`) and stays the definition of the
//! template, and a test cuts generated text both ways. No rule reads the
//! text before a piece's start, so the pass can begin at any such start
//! ([`pieces`]) and hand the text back to the scanner at any other.
//!
//! Two more templates take the same pass, each by three rules of its own.
//! The three-stage chain's (`chain::Scanner`, [`chain_pieces`]) has no
//! clitics, keeps a run of blanks before a digit whole and reads a control
//! character as no punctuation. In the cased-letter template
//! (`cased::Scanner`, [`cased_pieces`]) a capital behind a small letter
//! starts a piece, a clitic belongs to the letter piece before it, and the
//! run behind punctuation takes the slash as well as the line breaks where
//! the template says so.

/// Bytes per block: one bit of a mask per byte.
pub(super) const BLOCK: usize = 64;

/// The shortest text the encoder reads by blocks. Under it the scanner's
/// own loop is as fast on text the processor has learned: the pass runs
/// fewer instructions from a few blocks on, but it reads a block ahead and
/// leaves a stretch's last block to the scanner (measured: a bound of two
/// blocks costs a memorized unit of 160 to 320 bytes 3 to 5 % of its time
/// and saves new text of 320 bytes a third).
pub(super) const MIN_BYTES: usize = 8 * BLOCK;

/// The farthest the scanner walks before it tries the pass again where the
/// pass found no ASCII stretch. The walk doubles from two blocks to this
/// cap: a long part beyond ASCII pays one test of two blocks per walk, and
/// the ASCII behind such a part waits for the pass no longer than the cap.
/// The cap is chosen, not tuned. Counted on mixed text: a cap of 8 blocks
/// runs 1.7 to 2.3 % fewer instructions at 32k to 500k tokens and within
/// 0.4 % either way at 2k tokens; a cap of 512 blocks runs 2.3 to 3.6 %
/// more at 32k to 500k tokens and the same at 2k. Time was measured at
/// neither.
pub(super) const MAX_WALK: usize = 64 * BLOCK;

/// Moves the flag in byte `i` of a word to bit `56 + i` of the product, no
/// two flags of the eight on one bit.
const GATHER: u64 = 0x0102_0408_1020_4080;

/// The ASCII letters.
#[inline(always)]
pub(super) fn is_letter(byte: u8) -> bool {
    (byte | 0x20).wrapping_sub(b'a') < 26
}

/// The ASCII digits.
#[inline(always)]
pub(super) fn is_digit(byte: u8) -> bool {
    byte.wrapping_sub(b'0') < 10
}

/// The ASCII whitespace: the space, and `\t` to `\r`.
#[inline(always)]
pub(super) fn is_space(byte: u8) -> bool {
    byte == b' ' || byte.wrapping_sub(b'\t') < 5
}

/// The line breaks.
#[inline(always)]
pub(super) fn is_break(byte: u8) -> bool {
    byte == b'\n' || byte == b'\r'
}

/// One bit per flag byte of `flags`, bit `i` for byte `i`: eight flags at a
/// time into eight bits by one multiply.
#[inline(always)]
fn gather(flags: &[u8; BLOCK]) -> u64 {
    let mut mask = 0;
    for (eighth, lanes) in flags.as_chunks::<8>().0.iter().enumerate() {
        mask |= (u64::from_le_bytes(*lanes).wrapping_mul(GATHER) >> 56) << (8 * eighth);
    }
    mask
}

/// The class masks of one block.
#[derive(Clone, Copy)]
struct Masks {
    letters: u64,
    digits: u64,
    /// The literal space: the one whitespace a punctuation run takes.
    spaces: u64,
    whitespace: u64,
    breaks: u64,
}

impl Masks {
    /// A flag byte per byte first, in a loop with no exit and no table,
    /// which the compiler emits as vector compares; then the gathers, the
    /// larger half of the cost. So three flags a byte carry the five
    /// classes: whether the byte is whitespace, and one of two kinds on
    /// either side of that (a letter or the literal space, a digit or a
    /// line break).
    #[inline(always)]
    fn of(block: &[u8; BLOCK]) -> Self {
        let mut flags = [[0u8; BLOCK]; 3];
        for (at, &byte) in block.iter().enumerate() {
            flags[0][at] = u8::from(is_space(byte));
            flags[1][at] = u8::from(is_letter(byte) || byte == b' ');
            flags[2][at] = u8::from(is_digit(byte) || is_break(byte));
        }
        let whitespace = gather(&flags[0]);
        let first = gather(&flags[1]);
        let second = gather(&flags[2]);
        Self {
            letters: first & !whitespace,
            digits: second & !whitespace,
            spaces: first & whitespace,
            whitespace,
            breaks: second & whitespace,
        }
    }
}

/// What a block hands to the next one: the masks whose top bits the next
/// block's first characters read, and the runs open at the edge.
#[derive(Default)]
struct Edge {
    letters: u64,
    digits: u64,
    spaces: u64,
    breaks: u64,
    punct: u64,
    /// The whitespace no punctuation run takes.
    rest: u64,
    lead_ins: u64,
    /// The starts of the digit rule.
    groups: u64,
    /// The block ended in line breaks that a punctuation run takes.
    taking_breaks: bool,
    /// The starts behind clitics that reach over the edge.
    behind_clitics: u64,
}

/// `mask` one character on: bit `i` is the bit of the character before `i`,
/// the first one's from the block before, `last`.
#[inline(always)]
fn before(mask: u64, last: u64) -> u64 {
    (mask << 1) | (last >> 63)
}

/// `mask` one character back: bit `i` is the bit of the character after
/// `i`, the last one's from the block after, `first`.
#[inline(always)]
fn after(mask: u64, first: u64) -> u64 {
    (mask >> 1) | (first << 63)
}

impl Edge {
    /// The piece starts of the block at `base`, bit `i` for byte `base + i`,
    /// from the block's masks, the next block's (a start can hang on the
    /// character behind it, and a blank's on where its run ends) and the
    /// block before, `self`, which this block replaces; `None` when a run
    /// of blanks behind a line break ends at a character beyond ASCII,
    /// which may be whitespace too: the scanner's to say. `clitic_end(pos)`
    /// is the end of the clitic at `pos`, if one begins there.
    /// `rules` are the template's own beside these ([`Rules`]), with what
    /// they keep from the block before.
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn starts<const CHAIN: bool, C: Fn(usize) -> Option<usize>, R: Rules>(
        &mut self,
        text: &[u8],
        base: usize,
        block: &Masks,
        next: &Masks,
        digit_cap: u32,
        clitic_end: &C,
        rules: &mut R,
    ) -> Option<u64> {
        let (letters, digits, breaks) = (block.letters, block.digits, block.breaks);
        let blanks = block.whitespace & !breaks;
        let punct = !(letters | digits | block.whitespace);

        // Punctuation, and the line breaks behind it: one addition carries
        // through each run of breaks that begins behind punctuation (or
        // goes on from the block before) and clears it.
        let behind_punct = before(punct, self.punct);
        let punct_starts = punct & !behind_punct & !before(block.spaces, self.spaces);
        let (sum, taking_breaks) =
            breaks.overflowing_add(breaks & (behind_punct | u64::from(self.taking_breaks)));
        let rest = block.whitespace & !(breaks & !sum);
        // A template with a rule of its own for that run cuts by it.
        let (punct_starts, rest, taking_breaks) = rules.punct(
            block,
            self,
            punct,
            behind_punct,
            (punct_starts, rest, taking_breaks),
        );

        // Whitespace. A blank behind a line break starts a piece when that
        // break is the run's last, which is where the blanks from it end:
        // in this block, in the next or beyond it.
        let run_starts = rest & !before(rest, self.rest);
        // The chain's template never sees a digit (the stage before it cut
        // the digits out), so there a digit ends the text for this rule as
        // whitespace does: the blank before one starts no piece of its own.
        let (unseen, next_unseen) = if CHAIN { (digits, next.digits) } else { (0, 0) };
        let last_blanks = blanks & !after(block.whitespace | unseen, next.whitespace | next_unseen);
        let mut starts = punct_starts | run_starts | last_blanks;
        let mut behind_break = blanks & before(breaks, self.breaks) & !starts;
        while behind_break != 0 {
            let at = behind_break.trailing_zeros();
            behind_break &= behind_break - 1;
            let end = at + (blanks >> at).trailing_ones();
            let ends_in_break = if end < u64::BITS {
                (breaks >> end) & 1 != 0
            } else {
                let next_blanks = next.whitespace & !next.breaks;
                if next_blanks == u64::MAX {
                    let beyond = text[base + 2 * BLOCK..]
                        .iter()
                        .find(|&&byte| !is_space(byte) || is_break(byte));
                    match beyond {
                        Some(&byte) if byte.is_ascii() => is_break(byte),
                        Some(_) => return None,
                        None => false,
                    }
                } else {
                    (next.breaks >> next_blanks.trailing_ones()) & 1 != 0
                }
            };
            starts |= u64::from(!ends_in_break) << at;
        }

        // Letters. A clitic can begin only at an apostrophe that starts a
        // piece before a letter, which is a lead-in to the masks: the piece
        // starts there either way, and the clitic adds the start behind it.
        let lead_ins = after(letters, next.letters) & (blanks | punct_starts);
        let word_starts =
            letters & !before(letters, self.letters) & !before(lead_ins, self.lead_ins);
        let mut behind_clitics = std::mem::take(&mut self.behind_clitics);
        // The chain's template has no clitics, and a template with a letter
        // rule of its own reads them there.
        let mut quotes = if CHAIN || R::OWN_CLITICS {
            0
        } else {
            lead_ins & punct_starts
        };
        while quotes != 0 {
            let at = base + quotes.trailing_zeros() as usize;
            quotes &= quotes - 1;
            if text[at] != b'\'' {
                continue;
            }
            let behind =
                clitic_end(at).filter(|&end| text.get(end).is_some_and(|&byte| is_letter(byte)));
            match behind.map(|end| end - base) {
                Some(bit) if bit < BLOCK => behind_clitics |= 1 << bit,
                Some(bit) => self.behind_clitics |= 1 << (bit - BLOCK),
                None => {}
            }
        }

        // Digits: a run's first, then every `digit_cap`th behind a start,
        // inside the run (`full`: the digits with `digit_cap - 1` digits
        // before them).
        let mut groups = digits & !before(digits, self.digits);
        if digit_cap == 1 {
            groups = digits;
        } else if digit_cap < u64::BITS {
            let full = (1..digit_cap).fold(digits, |full, back| {
                full & ((digits << back) | (self.digits >> (u64::BITS - back)))
            });
            let mut more =
                ((groups << digit_cap) | (self.groups >> (u64::BITS - digit_cap))) & full;
            while more != 0 {
                groups |= more;
                more = (more << digit_cap) & full;
            }
        }

        *self = Self {
            letters,
            digits,
            spaces: block.spaces,
            breaks,
            punct,
            rest,
            lead_ins,
            groups,
            taking_breaks,
            behind_clitics: self.behind_clitics,
        };
        let starts = starts | word_starts | groups | behind_clitics;
        Some(rules.letters(
            text,
            base,
            letters,
            lead_ins & punct_starts,
            clitic_end,
            starts,
        ))
    }
}

/// The rules of a template beside the ones [`Edge::starts`] cuts by, with
/// what they keep from block to block. `()` has none: the GPT-4 family's
/// template and the chain's are cut by [`Edge::starts`] alone, and each call
/// below is nothing in their copies of the pass.
trait Rules {
    /// Whether [`Self::letters`] reads the clitics: [`Edge::starts`] then
    /// leaves them to it.
    const OWN_CLITICS: bool = false;

    /// Reads what the rules tell apart in the block the pass cuts next.
    #[inline(always)]
    fn read(&mut self, _block: &[u8; BLOCK], _masks: &Masks) {}

    /// The punctuation rule's starts, the whitespace no punctuation run
    /// takes and whether the block ends inside such a run: `cut`, as
    /// [`Edge::starts`] found them, or the rule's own, from the block's
    /// masks and the block before (`edge`).
    #[inline(always)]
    fn punct(
        &mut self,
        _block: &Masks,
        _edge: &Edge,
        _punct: u64,
        _behind_punct: u64,
        cut: (u64, u64, bool),
    ) -> (u64, u64, bool) {
        cut
    }

    /// The block's starts: `starts`, as [`Edge::starts`] found them, or
    /// those with the letter rule's own. `quotes` are the punctuation
    /// starts right before a letter.
    #[inline(always)]
    fn letters<C: Fn(usize) -> Option<usize>>(
        &mut self,
        _text: &[u8],
        _base: usize,
        _letters: u64,
        _quotes: u64,
        _clitic_end: &C,
        starts: u64,
    ) -> u64 {
        starts
    }
}

impl Rules for () {}

/// The rules of the cased-letter template (`cased`), and what a block of it
/// hands to the next one beside [`Edge`].
#[derive(Default)]
struct Cased {
    /// Whether the run behind punctuation takes the slash.
    take_slash: bool,
    /// The capitals and the slashes of the block the pass cuts next.
    capitals: u64,
    slashes: u64,
    /// Of the block before: the letters, the small letters and the
    /// punctuation no run before it takes.
    letters: u64,
    smalls: u64,
    free: u64,
    /// The characters of a clitic that reaches over the edge, and the start
    /// behind that clitic.
    in_clitics: u64,
    clitic_ends: u64,
}

impl Rules for Cased {
    const OWN_CLITICS: bool = true;

    /// One more flag a byte and one more gather, for this template alone:
    /// the capitals among the letters and the slash among the punctuation.
    #[inline(always)]
    fn read(&mut self, block: &[u8; BLOCK], masks: &Masks) {
        let mut flags = [0u8; BLOCK];
        for (at, &byte) in block.iter().enumerate() {
            flags[at] = u8::from(byte.wrapping_sub(b'A') < 26 || byte == b'/');
        }
        let either = gather(&flags);
        self.capitals = either & masks.letters;
        self.slashes = either & !masks.letters;
    }

    /// The run behind punctuation takes the slash as well as the line
    /// breaks, in a template that says so. A slash is punctuation itself,
    /// so that run begins at a line break (a slash right behind punctuation
    /// is of the punctuation run still) and goes on through breaks and
    /// slashes. One addition clears each run of the two from its first
    /// break behind punctuation (the later such breaks of a run come back
    /// as bits of the sum, and are taken all the same). Punctuation behind
    /// a taken slash starts a piece: the run before it has ended.
    #[inline(always)]
    fn punct(
        &mut self,
        block: &Masks,
        edge: &Edge,
        punct: u64,
        behind_punct: u64,
        _: (u64, u64, bool),
    ) -> (u64, u64, bool) {
        let breaks = block.breaks;
        let joiners = breaks | if self.take_slash { self.slashes } else { 0 };
        let behind = (breaks & behind_punct) | (joiners & u64::from(edge.taking_breaks));
        let (sum, taking) = joiners.overflowing_add(behind);
        let taken = (joiners & !sum) | behind;
        let free = punct & !taken;
        let starts = free & !before(free, self.free) & !before(block.spaces, edge.spaces);
        self.free = free;
        (starts, block.whitespace & !taken, taking)
    }

    /// A capital behind a small letter starts a piece, and a clitic belongs
    /// to the letter piece before it. Such an apostrophe stands between two
    /// letters; one right behind a clitic begins a piece of its own (a
    /// piece takes one clitic). No start lies inside a clitic, and one lies
    /// behind it whatever stands there.
    #[inline(always)]
    fn letters<C: Fn(usize) -> Option<usize>>(
        &mut self,
        text: &[u8],
        base: usize,
        letters: u64,
        quotes: u64,
        clitic_end: &C,
        starts: u64,
    ) -> u64 {
        let smalls = letters & !self.capitals;
        let case_starts = self.capitals & before(smalls, self.smalls);
        let mut in_clitics = std::mem::take(&mut self.in_clitics);
        let mut clitic_ends = std::mem::take(&mut self.clitic_ends);
        let mut quotes = quotes & before(letters, self.letters);
        while quotes != 0 {
            let bit = quotes.trailing_zeros();
            quotes &= quotes - 1;
            let at = base + bit as usize;
            if text[at] != b'\'' || (clitic_ends >> bit) & 1 != 0 {
                continue;
            }
            let Some(end) = clitic_end(at) else {
                continue;
            };
            let behind = (1u128 << (end - at)) << bit;
            let inside = behind - (1u128 << bit);
            in_clitics |= inside as u64;
            self.in_clitics |= (inside >> u64::BITS) as u64;
            clitic_ends |= behind as u64;
            self.clitic_ends |= (behind >> u64::BITS) as u64;
        }
        self.letters = letters;
        self.smalls = smalls;
        ((starts | case_starts) & !in_clitics) | clitic_ends
    }
}

/// Cuts the ASCII stretch of `text` that begins at `from`, a piece's start,
/// by the template with the digit bound `digit_cap`: calls `piece` with the
/// start and the end of each piece, in order, and returns the start of the
/// piece it leaves open, which is the caller's to cut by its own matcher
/// (`from` itself when the pass cut nothing). A block is cut with the masks
/// of the next one in hand, so the pass stops before the last whole block
/// ahead of the text's end or of a byte beyond ASCII, and where
/// [`Edge::starts`] declines; it takes no stretch under two blocks.
/// `clitic_end(pos)` is the end of the clitic at `pos`, if one begins there.
/// `rules` are the template's own beside those of [`Edge::starts`]
/// ([`Rules`]): `()` for a template that has none.
/// Compiled into its caller: as a routine of its own it took `piece` by
/// reference, and the encoder's probe then kept its captures in memory
/// through the caller's walk between two stretches as well (counted: 1.4
/// instructions per token on a Cyrillic text, 2 to 4 on long mixed text).
#[inline(always)]
fn pieces_of<const CHAIN: bool, C, F, R>(
    text: &[u8],
    from: usize,
    digit_cap: u32,
    clitic_end: C,
    mut rules: R,
    piece: &mut F,
) -> usize
where
    C: Fn(usize) -> Option<usize>,
    F: FnMut(usize, usize),
    R: Rules,
{
    let [first, rest @ ..] = text[from..].as_chunks::<BLOCK>().0 else {
        return from;
    };
    if !text[from..]
        .first_chunk::<{ 2 * BLOCK }>()
        .is_some_and(|stretch| reads::<CHAIN>(stretch))
    {
        return from;
    }
    let mut open = from;
    let mut edge = Edge::default();
    let mut masks = Masks::of(first);
    rules.read(first, &masks);
    // The stretch's first byte starts the open piece and ends none.
    let mut opening = 1;
    for (index, block) in rest.iter().enumerate() {
        if !reads::<CHAIN>(block) {
            break;
        }
        let next = Masks::of(block);
        let base = from + index * BLOCK;
        let Some(starts) = edge.starts::<CHAIN, C, R>(
            text,
            base,
            &masks,
            &next,
            digit_cap,
            &clitic_end,
            &mut rules,
        ) else {
            break;
        };
        let mut starts = starts & !opening;
        opening = 0;
        while starts != 0 {
            let start = base + starts.trailing_zeros() as usize;
            starts &= starts - 1;
            piece(open, start);
            open = start;
        }
        masks = next;
        rules.read(block, &masks);
    }
    open
}

/// Whether the pass reads `bytes`: all ASCII, and for the chain's template
/// (`CHAIN`) no control character either. There a control character is a
/// lead-in of letters and no punctuation, which the masks do not say, so a
/// block that holds one is the scanner's, as a block beyond ASCII is. The
/// ASCII test stands first and alone: it ends at the first bytes beyond
/// ASCII, so a try of the pass on text in another script costs little.
#[inline(always)]
fn reads<const CHAIN: bool>(bytes: &[u8]) -> bool {
    bytes.is_ascii()
        && !(CHAIN
            && bytes.iter().fold(false, |odd, &byte| {
                odd | (byte.is_ascii_control() && !is_space(byte))
            }))
}

/// [`pieces_of`] by the GPT-4 family's template.
#[inline(always)]
pub(super) fn pieces<C, F>(
    text: &[u8],
    from: usize,
    digit_cap: u32,
    clitic_end: C,
    piece: &mut F,
) -> usize
where
    C: Fn(usize) -> Option<usize>,
    F: FnMut(usize, usize),
{
    pieces_of::<false, C, F, ()>(text, from, digit_cap, clitic_end, (), piece)
}

/// [`pieces_of`] by the cased-letter template (`cased`). On plain ASCII it
/// cuts as the GPT-4 family's template but for three rules ([`Cased`]): a
/// capital behind a small letter starts a piece; a clitic belongs to the
/// letter piece before it and is no piece of its own; and, in a template
/// that says so (`take_slash`), the run behind punctuation takes the slash
/// as well as the line breaks.
/// `clitic_end(pos)` is the end of the clitic at `pos`, if one begins there.
#[inline(always)]
pub(super) fn cased_pieces<C, F>(
    text: &[u8],
    from: usize,
    digit_cap: u32,
    take_slash: bool,
    clitic_end: C,
    piece: &mut F,
) -> usize
where
    C: Fn(usize) -> Option<usize>,
    F: FnMut(usize, usize),
{
    let rules = Cased {
        take_slash,
        ..Cased::default()
    };
    pieces_of::<false, C, F, Cased>(text, from, digit_cap, clitic_end, rules, piece)
}

/// [`pieces_of`] by the template of the three-stage chain (`chain`), whose
/// first stage cuts the digits by the bound `digit_cap`. On plain ASCII it
/// cuts as the GPT-4 family's template but for three rules: it has no
/// clitics, a run of blanks before a digit stays whole, and a control
/// character is no punctuation (such a block is left to the scanner).
#[inline(always)]
pub(super) fn chain_pieces<F: FnMut(usize, usize)>(
    text: &[u8],
    from: usize,
    digit_cap: u32,
    piece: &mut F,
) -> usize {
    pieces_of::<true, _, F, ()>(text, from, digit_cap, |_| None, (), piece)
}

/// Bytes between two places a text is asked at for an ASCII stretch
/// ([`has_stretch`]), on a text of up to [`PLACES`] such steps.
pub(super) const SAMPLE: usize = 64 * BLOCK;

/// The most places a text is asked at: a longer text is asked at this many,
/// further apart. A place of a long text is memory the scan has not reached
/// yet, a cache miss ahead of the scan each (measured on a text of 4 MB
/// with no stretch, asked every [`SAMPLE`] bytes: 1,024 places, 0.5 to
/// 2.3 % of the encode's time).
pub(super) const PLACES: usize = 64;

/// Whether one of the places `text` is asked at begins a stretch [`pieces`]
/// cuts (two blocks of ASCII): one place per [`SAMPLE`] bytes, on a text of
/// more than [`PLACES`] such steps that many places evenly apart. The
/// encoder reads a text by blocks only then: between two stretches the caller of
/// [`pieces`] walks piece by piece, and on a text with no stretch that walk
/// is all there is, slower than the scanner's own loop (measured on a
/// Cyrillic text: 1 % more instructions, 4 to 7 % more time). The first
/// place alone would not do: a long text of sections in several scripts
/// opens with any of them. It is a sample and no search, so a text whose
/// stretches every place misses keeps the scanner's loop and loses nothing.
pub(super) fn has_stretch(text: &[u8]) -> bool {
    let step = text.len().div_ceil(PLACES).max(SAMPLE);
    text.chunks(step).any(|part| {
        part.first_chunk::<{ 2 * BLOCK }>()
            .is_some_and(|stretch| stretch.is_ascii())
    })
}

/// The punctuation pieces after which the lean scan of an all-ASCII text
/// leaves for the pass, when they end within [`HEAD`] bytes of the text.
pub(super) const MARKS: u32 = 2;

/// How far into a text the lean scan looks for its [`MARKS`].
pub(super) const HEAD: usize = 4 * BLOCK;

/// A text under this many bytes that a thread reads again is one whose
/// branches the lean scan no longer misses: the pass has to beat a scan at
/// its best there, which takes more pieces a block.
pub(super) const SHORT_TEXT: usize = 32 * BLOCK;

/// Whether the all-ASCII `text` opens as short pieces: one byte in eight
/// of its first two blocks is no letter, three in sixteen in a text under
/// [`SHORT_TEXT`]. Read once a text, off the scan's loop.
#[inline(never)]
pub(super) fn is_broken(text: &[u8]) -> bool {
    text.first_chunk::<{ 2 * BLOCK }>().is_some_and(|head| {
        // Counted in a byte, so the loop is byte compares: two blocks are
        // fewer than 256 bytes.
        let mut no_letter = 0u8;
        for &byte in head {
            no_letter += u8::from(!is_letter(byte));
        }
        let bar = if text.len() < SHORT_TEXT {
            3 * BLOCK / 8
        } else {
            2 * BLOCK / 8
        };
        usize::from(no_letter) >= bar
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_masks_hold_one_bit_per_byte_in_the_bytes_order() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..500 {
            let mut block = [0u8; BLOCK];
            for byte in &mut block {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state.to_le_bytes()[0] & 0x7F;
            }
            let bits = |test: fn(u8) -> bool| {
                block
                    .iter()
                    .rev()
                    .fold(0u64, |mask, &byte| (mask << 1) | u64::from(test(byte)))
            };
            let masks = Masks::of(&block);
            assert_eq!(masks.letters, bits(is_letter), "{block:?}");
            assert_eq!(masks.digits, bits(is_digit), "{block:?}");
            assert_eq!(masks.spaces, bits(|byte| byte == b' '), "{block:?}");
            assert_eq!(masks.whitespace, bits(is_space), "{block:?}");
            assert_eq!(masks.breaks, bits(is_break), "{block:?}");
        }
    }
}
