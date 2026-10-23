//! A set of characters, as whoever lowered a template's pattern resolved it.

use std::cmp::Ordering;

/// A set of characters: the characters in sorted, disjoint ranges, or every character outside
/// them.
///
/// What `\w`, `\s` and `\d` hold is the regular-expression dialect's to say, and dialects differ:
/// Python's `regex` counts a combining mark as a word character and not `²`, and leaves out of
/// `\s` four control characters that Python's own `str.isspace` counts. So whoever lowers a
/// template's pattern resolves such a class into ranges with the template's own dialect, and the
/// crate carries no character tables of its own.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Class {
    ranges: Box<[(char, char)]>,
    negated: bool,
}

impl Class {
    /// The characters in `ranges`, each range holding both its ends, or, `negated`, every
    /// character outside them. The ranges may come in any order and overlap; a range whose start
    /// is past its end holds nothing.
    pub fn new(ranges: impl IntoIterator<Item = (char, char)>, negated: bool) -> Self {
        let mut ranges: Vec<(char, char)> =
            ranges.into_iter().filter(|(lo, hi)| lo <= hi).collect();
        ranges.sort_unstable();
        let mut merged: Vec<(char, char)> = Vec::with_capacity(ranges.len());
        for (lo, hi) in ranges {
            match merged.last_mut() {
                Some((_, last)) if successor(*last).is_none_or(|next| lo <= next) => {
                    *last = (*last).max(hi);
                }
                _ => merged.push((lo, hi)),
            }
        }
        Self {
            ranges: merged.into_boxed_slice(),
            negated,
        }
    }

    /// Every character: `.` under `DOTALL`, `[\s\S]`.
    pub fn any() -> Self {
        Self {
            ranges: Box::default(),
            negated: true,
        }
    }

    /// Whether `c` is in the set.
    pub fn contains(&self, c: char) -> bool {
        let inside = self
            .ranges
            .binary_search_by(|&(lo, hi)| {
                if hi < c {
                    Ordering::Less
                } else if lo > c {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok();
        inside != self.negated
    }

    /// Whether the set holds every character, however it was spelled.
    pub(super) fn is_any(&self) -> bool {
        if self.negated {
            self.ranges.is_empty()
        } else {
            *self.ranges == [('\0', char::MAX)]
        }
    }
}

/// The character after `c`, across the gap the surrogates leave; none after the last.
fn successor(c: char) -> Option<char> {
    match c {
        '\u{D7FF}' => Some('\u{E000}'),
        _ => char::from_u32(u32::from(c) + 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_sorted_merged_and_read_by_search() {
        let class = Class::new(
            [('x', 'z'), ('a', 'c'), ('b', 'd'), ('e', 'e'), ('q', 'p')],
            false,
        );
        assert_eq!(
            *class.ranges,
            [('a', 'e'), ('x', 'z')],
            "overlapping and adjacent ranges merge"
        );
        for c in ['a', 'c', 'e', 'x', 'y', 'z'] {
            assert!(class.contains(c), "{c:?}");
        }
        for c in ['f', 'p', 'q', 'w', '{', '\0', char::MAX] {
            assert!(!class.contains(c), "{c:?}");
        }
        assert_eq!(
            Class::new([('a', 'e'), ('x', 'z')], false),
            class,
            "one set, one spelling"
        );
    }

    #[test]
    fn a_negated_class_holds_every_character_outside_its_ranges() {
        let class = Class::new([('>', '>'), ('\n', '\n')], true);
        assert!(!class.contains('>'));
        assert!(!class.contains('\n'));
        for c in ['a', ' ', '<', '\u{0301}', '\u{1F600}'] {
            assert!(class.contains(c), "{c:?}");
        }
    }

    #[test]
    fn every_character_is_any_however_it_is_spelled() {
        assert!(Class::any().is_any());
        assert!(Class::any().contains('\n'));
        assert!(Class::any().contains(char::MAX));
        let space = [('\t', '\r'), (' ', ' ')];
        let rest = [('\0', '\u{8}'), ('\u{E}', '\u{1F}'), ('!', char::MAX)];
        let both = Class::new(space.into_iter().chain(rest), false);
        assert!(both.is_any(), "a class and its complement");
        assert!(
            Class::new([('\u{E000}', char::MAX), ('\0', '\u{D7FF}')], false).is_any(),
            "the surrogates' gap is no gap between characters"
        );
        assert!(!Class::new([('\0', '\u{D7FF}')], false).is_any());
        assert!(!Class::new([('a', 'a')], true).is_any());
        assert!(
            !Class::new([('\0', char::MAX)], true).is_any(),
            "that is the empty set"
        );
    }
}
