//! Characters pre-formed as one symbol before a byte-level merge, where the
//! vocabulary proves it exact: a piece of a multi-byte script then merges
//! from its characters' tokens instead of from their bytes, and is cut
//! between two characters no token holds adjacent, so a run of ideographs
//! merges as segments of a character or two instead of as one long piece.
//!
//! A character `c` with a token of its own is *pre-formable* when (1) its
//! bytes alone merge to that token and (2) every merge that takes the token
//! as an operand ranks after the merges that form it; an occurrence of `c`
//! is pre-formed when also (3) no token covers part of `c` together with
//! the bytes beside it. For (3) the *fragment* tokens are read at load (a
//! token that starts with continuation bytes, a token that ends in an
//! incomplete character) and the bytes around the occurrence are compared
//! with them. Then no merge crosses `c`'s boundary while `c` is in parts
//! (its product would be a fragment token, excluded by 3), so the merges
//! inside `c` commute with every other merge of the piece, and no merge
//! taking `c` whole fires before `c` would have been whole (by 2): the
//! sequence of merges, and so the ids, are the model's own. Two adjacent
//! pre-formed characters whose pair no token holds cannot merge, directly
//! or inside larger symbols (the product would hold both adjacent), so the
//! piece is cut there and each segment merges on its own.
//!
//! The reference encoders of the engines this crate is measured against
//! pre-form characters under the same three conditions.

use rustc_hash::FxHashMap;

use super::{Merge, MergeTable};

/// Bits of a token id in a BMP entry; the rule index takes the rest.
const ID_BITS: u32 = 22;
/// Fragment rules the entry's high bits can index; a character whose new
/// rule would be one more is declined (it merges from its bytes).
const MAX_RULES: usize = 1 << (u32::BITS - ID_BITS);
/// The id bits of a BMP entry.
const ID_MASK: u32 = (1 << ID_BITS) - 1;
/// A BMP entry of a character that does not pre-form.
const NO_ENTRY: u32 = u32::MAX;
/// No character: the symbol before was a byte, or there was none.
pub(super) const NO_CHAR: u32 = u32::MAX;
/// Filter bits per character pair, about one false positive in thirty (a
/// false positive only forgoes a cut).
const BIGRAM_BITS_PER_PAIR: usize = 32;
/// Bytes past which a run merges from its characters wherever its script
/// has any (see [`Preformed::pays_off`]).
const LONG_RUN: usize = 32;

/// The fragment tokens that could cover part of one pre-formable
/// character: `heads` are the byte strings that, right before it, complete
/// a token ending inside it; `tails` those that, right after it, complete a
/// token starting inside it. `before` and `after` hold the heads' last and
/// the tails' first bytes, so most occurrences are cleared by one bit test.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct Rule {
    before: [u64; 4],
    after: [u64; 4],
    heads: Vec<Box<[u8]>>,
    tails: Vec<Box<[u8]>>,
}

/// The pre-formable characters of one vocabulary; see the module
/// documentation. Empty for a vocabulary where none pre-forms.
pub(super) struct Preformed {
    /// Code point (BMP) -> `rule << ID_BITS | id` of the character's token,
    /// or [`NO_ENTRY`]; `rule` indexes [`Self::rules`] (0: no fragment can
    /// cover part of the character, it pre-forms anywhere). One load per
    /// character. Empty when no character pre-forms.
    bmp: Box<[u32]>,
    /// The fragment contexts, deduplicated; `rules[0]` is empty.
    rules: Vec<Rule>,
    /// Every pair of adjacent whole characters inside some token, as the
    /// bits of a one-hash filter over `first << 21 | second`.
    bigrams: Box<[u64]>,
    /// Bits per 64-code-point block of the BMP (see [`Self::pays_off`]):
    /// `good`, at least eight pre-formable characters whose fragment checks
    /// are cheap; `fair`, checks up to four times as costly; `some`, any.
    good: [u64; 16],
    fair: [u64; 16],
    some: [u64; 16],
}

impl Preformed {
    /// No pre-formable character.
    pub(super) fn empty() -> Self {
        Self {
            bmp: Box::default(),
            rules: vec![Rule::default()],
            bigrams: vec![0; 1].into_boxed_slice(),
            good: [0; 16],
            fair: [0; 16],
            some: [0; 16],
        }
    }

    /// Whether no character pre-forms.
    pub(super) fn is_empty(&self) -> bool {
        self.bmp.is_empty()
    }

    /// Characters that pre-form (somewhere).
    pub(super) fn characters(&self) -> usize {
        self.bmp.iter().filter(|&&e| e != NO_ENTRY).count()
    }

    /// Bytes the tables hold, for the load log.
    pub(super) fn bytes(&self) -> usize {
        size_of_val(&*self.bmp)
            + size_of_val(&*self.bigrams)
            + self
                .rules
                .iter()
                .map(|r| {
                    size_of::<Rule>()
                        + r.heads
                            .iter()
                            .chain(&r.tails)
                            .map(|f| f.len() + 16)
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// The pre-formable characters of a vocabulary under `table`: `raws`
    /// are its tokens as the raw bytes they stand for, with their ids (a
    /// model's alphabet spellings mapped back, a rank file's bytes as they
    /// are); `min_operand` holds, per id, the least rank of a merge that
    /// takes the id as an operand (`u32::MAX` for none).
    pub(super) fn analyse(
        table: &MergeTable,
        mut raws: Vec<(u32, Vec<u8>)>,
        min_operand: &[u32],
    ) -> Self {
        raws.retain(|(_, raw)| !raw.is_empty());
        raws.sort_unstable_by_key(|(id, _)| *id);
        // The fragment tokens: leading continuation bytes -> the rest;
        // a trailing incomplete character -> the bytes before it.
        let mut heads: FxHashMap<&[u8], Vec<Box<[u8]>>> = FxHashMap::default();
        let mut tails: FxHashMap<&[u8], Vec<Box<[u8]>>> = FxHashMap::default();
        for (_, raw) in &raws {
            let lead = raw
                .iter()
                .take_while(|&&b| is_continuation(b))
                .count()
                .min(3);
            if lead > 0 && lead < raw.len() {
                tails
                    .entry(&raw[..lead])
                    .or_default()
                    .push(raw[lead..].into());
            }
            if let Some(p) = raw.iter().rposition(|&b| !is_continuation(b)) {
                if raw.len() - p < char_len(raw[p]) && p > 0 {
                    heads.entry(&raw[p..]).or_default().push(raw[..p].into());
                }
            }
        }
        let mut bmp = vec![NO_ENTRY; 0x1_0000];
        let mut rules = vec![Rule::default()];
        let mut rule_index: FxHashMap<Rule, u32> = FxHashMap::default();
        let mut any = false;
        for (id, raw) in &raws {
            if !(2..=4).contains(&raw.len()) || *id > ID_MASK {
                continue;
            }
            let Some(c) = one_char(raw) else { continue };
            let cp = u32::from(c);
            if cp >= 0x1_0000 {
                continue;
            }
            // (1) the bytes alone merge to the token, and the greatest rank
            // on the way.
            let mut word: Vec<u32> = raw.iter().map(|&b| table.byte_id(b)).collect();
            let mut max_rank = 0u32;
            loop {
                let mut best: Option<(u32, usize, u32)> = None;
                for at in 0..word.len() - 1 {
                    let merge = table.merge_of(word[at], word[at + 1]);
                    if merge != Merge::NONE && best.is_none_or(|(rank, _, _)| merge.rank() < rank) {
                        best = Some((merge.rank(), at, merge.product()));
                    }
                }
                let Some((rank, at, product)) = best else {
                    break;
                };
                max_rank = max_rank.max(rank);
                word[at] = product;
                word.remove(at + 1);
            }
            if word != [*id] {
                continue;
            }
            // (2) every merge that takes the token ranks after those.
            if min_operand
                .get(*id as usize)
                .is_some_and(|&rank| rank <= max_rank)
            {
                continue;
            }
            // (3) the fragments that could cover part of it, by its proper
            // prefixes and suffixes.
            let mut rule = Rule::default();
            for k in 1..raw.len() {
                for head in heads.get(&raw[..k]).into_iter().flatten() {
                    set_bit(&mut rule.before, head[head.len() - 1]);
                    rule.heads.push(head.clone());
                }
                for tail in tails.get(&raw[k..]).into_iter().flatten() {
                    set_bit(&mut rule.after, tail[0]);
                    rule.tails.push(tail.clone());
                }
            }
            let index = if rule.heads.is_empty() && rule.tails.is_empty() {
                0
            } else {
                rule.heads.sort();
                rule.heads.dedup();
                rule.tails.sort();
                rule.tails.dedup();
                match rule_index.get(&rule) {
                    Some(&index) => index,
                    None if rules.len() >= MAX_RULES => continue,
                    None => {
                        let index = rules.len() as u32;
                        rules.push(rule.clone());
                        rule_index.insert(rule, index);
                        index
                    }
                }
            };
            bmp[cp as usize] = index << ID_BITS | *id;
            any = true;
        }
        if !any {
            return Self::empty();
        }
        // A block is good with at least eight pre-formable characters whose
        // fragment checks stay cheap: at most eight heads to compare on
        // average where the byte before the character (inside a word of the
        // script, a continuation byte) passes the bit test.
        let (mut good, mut fair, mut some) = ([0u64; 16], [0u64; 16], [0u64; 16]);
        for block in 0..1024usize {
            let (mut characters, mut heads_n) = (0usize, 0usize);
            for &entry in &bmp[block << 6..(block + 1) << 6] {
                if entry == NO_ENTRY {
                    continue;
                }
                characters += 1;
                let rule = &rules[(entry >> ID_BITS) as usize];
                if rule.before[2] != 0 {
                    heads_n += rule.heads.len();
                }
            }
            if characters > 0 {
                some[block >> 6] |= 1 << (block & 63);
            }
            if characters >= 8 && heads_n <= 8 * characters {
                good[block >> 6] |= 1 << (block & 63);
            }
            if characters >= 8 && heads_n <= 32 * characters {
                fair[block >> 6] |= 1 << (block & 63);
            }
        }
        // Adjacent whole characters inside any token (a fragment token's
        // whole characters count too).
        let mut pairs: Vec<u64> = Vec::new();
        for (_, raw) in &raws {
            let mut prev: Option<(u32, usize)> = None;
            let mut i = 0;
            while i < raw.len() {
                match whole_char(&raw[i..]) {
                    Some((c, len)) => {
                        if let Some((p, end)) = prev {
                            if end == i {
                                pairs.push(u64::from(p) << 21 | u64::from(c));
                            }
                        }
                        prev = Some((c, i + len));
                        i += len;
                    }
                    None => {
                        prev = None;
                        i += 1;
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let bits = (pairs.len() * BIGRAM_BITS_PER_PAIR)
            .next_power_of_two()
            .max(64);
        let mut bigrams = vec![0u64; bits / 64];
        for &pair in &pairs {
            let bit = bigram_bit(pair, bits);
            bigrams[bit >> 6] |= 1 << (bit & 63);
        }
        Self {
            bmp: bmp.into_boxed_slice(),
            rules,
            bigrams: bigrams.into_boxed_slice(),
            good,
            fair,
            some,
        }
    }

    /// Whether `bytes` (not all ASCII) merges faster from its characters
    /// than from its bytes, by the block of its first non-ASCII character:
    /// an ideograph, a kana or a Hangul syllable (lead byte E3 to ED: the
    /// frequent ones pre-form, however sparse their block), or a block of
    /// cheap characters; the longer the run, the costlier the checks may be.
    #[inline(always)]
    pub(super) fn pays_off(&self, bytes: &[u8]) -> bool {
        let Some(k) = bytes.iter().position(|&b| b >= 0x80) else {
            return false;
        };
        let b0 = u32::from(bytes[k]);
        let at = |j: usize| bytes.get(k + j).map_or(0, |&b| u32::from(b) & 0x3F);
        let cp = match b0 {
            0xC0..=0xDF => (b0 & 0x1F) << 6 | at(1),
            0xE3..=0xED => return true,
            0xE0..=0xEF => (b0 & 0x0F) << 12 | at(1) << 6 | at(2),
            _ => return bytes.len() > LONG_RUN,
        };
        let (word, bit) = ((cp >> 12) as usize, (cp >> 6) & 63);
        self.good[word] >> bit & 1 != 0
            || (bytes.len() > 16 && self.fair[word] >> bit & 1 != 0)
            || (bytes.len() > LONG_RUN && self.some[word] >> bit & 1 != 0)
    }

    /// Whether some token may hold the character `a` right before `b`
    /// (never `false` when one does).
    #[inline(always)]
    pub(super) fn joinable(&self, a: u32, b: u32) -> bool {
        let bit = bigram_bit(u64::from(a) << 21 | u64::from(b), self.bigrams.len() * 64);
        self.bigrams[bit >> 6] >> (bit & 63) & 1 != 0
    }

    /// The token of the character at `bytes[i..i + len]` (code point `cp`)
    /// when this occurrence pre-forms (see the module documentation).
    #[inline(always)]
    pub(super) fn token_at(&self, bytes: &[u8], i: usize, len: usize, cp: u32) -> Option<u32> {
        let entry = *self.bmp.get(cp as usize)?;
        if entry == NO_ENTRY {
            return None;
        }
        let (id, rule) = (entry & ID_MASK, entry >> ID_BITS);
        if rule != 0 {
            let rule = &self.rules[rule as usize];
            let j = i + len;
            if i > 0
                && bit(&rule.before, bytes[i - 1])
                && rule
                    .heads
                    .iter()
                    .any(|head| head.len() <= i && bytes[i - head.len()..i] == **head)
            {
                return None;
            }
            if j < bytes.len()
                && bit(&rule.after, bytes[j])
                && rule
                    .tails
                    .iter()
                    .any(|tail| j + tail.len() <= bytes.len() && bytes[j..j + tail.len()] == **tail)
            {
                return None;
            }
        }
        Some(id)
    }
}

/// Bit of the pair `key` in a filter of `bits` (a power of two) bits.
#[inline(always)]
fn bigram_bit(key: u64, bits: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & (bits - 1)
}

#[inline(always)]
fn bit(set: &[u64; 4], byte: u8) -> bool {
    set[usize::from(byte >> 6)] >> (byte & 63) & 1 != 0
}

fn set_bit(set: &mut [u64; 4], byte: u8) {
    set[usize::from(byte >> 6)] |= 1 << (byte & 63);
}

/// Whether `byte` continues a UTF-8 character.
#[inline(always)]
pub(super) fn is_continuation(byte: u8) -> bool {
    (0x80..0xC0).contains(&byte)
}

/// Bytes of the character `lead` starts (1 for a byte that starts none).
#[inline(always)]
pub(super) fn char_len(lead: u8) -> usize {
    match lead {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

/// The character `bytes` spells, when it spells exactly one.
fn one_char(bytes: &[u8]) -> Option<char> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// The whole character at the start of `bytes` and its length, if one is
/// there.
fn whole_char(bytes: &[u8]) -> Option<(u32, usize)> {
    let len = char_len(bytes[0]);
    if bytes[0] >= 0x80 && len == 1 || len > bytes.len() {
        return None;
    }
    let c = one_char(&bytes[..len])?;
    Some((u32::from(c), len))
}

/// The code point of the character of `len` bytes at `bytes` (valid UTF-8).
#[inline(always)]
pub(super) fn code_point(bytes: &[u8], len: usize) -> u32 {
    let b0 = u32::from(bytes[0]);
    let at = |j: usize| u32::from(bytes[j]) & 0x3F;
    match len {
        2 => (b0 & 0x1F) << 6 | at(1),
        3 => (b0 & 0x0F) << 12 | at(1) << 6 | at(2),
        _ => (b0 & 0x07) << 18 | at(1) << 12 | at(2) << 6 | at(3),
    }
}
