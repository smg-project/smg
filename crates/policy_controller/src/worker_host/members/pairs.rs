//! The pair classes a generation keeps for its prefill and decode lists, and
//! a pair's admission as a controller reads it.

use std::{fmt, iter, sync::OnceLock};

use super::Members;
use crate::{
    document::spec::Pairing,
    engine::{PairAdmission, PartnerClasses},
};

/// A generation's sets of pair classes: each under the pairing mode and the
/// holder's count of in-place moves its ask read.
pub(super) type PairSets = Kept<(u64, Pairing), PartnerClasses>;

/// Values made one behind another for as long as a generation stands, each
/// under a key: an ask reads the value of its key, and makes it first where
/// none stands.
///
/// A value is neither freed nor moved while the generation stands, so a
/// decision under way keeps the one it read, as a list's cell keeps its
/// values. A read costs one step for each value made before its own.
pub(super) struct Kept<K, V>(OnceLock<Box<Link<K, V>>>);

/// One kept value under its key, and the values made behind it.
struct Link<K, V> {
    key: K,
    value: V,
    behind: Kept<K, V>,
}

impl<K, V> Default for Kept<K, V> {
    fn default() -> Self {
        Self(OnceLock::new())
    }
}

/// The values are counted and not printed: a print takes no deeper a stack
/// than their fall does.
impl<K, V> fmt::Debug for Kept<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let links = iter::successors(self.0.get(), |link| link.behind.0.get());
        f.debug_struct("Kept")
            .field("values", &links.count())
            .finish()
    }
}

impl<K, V> Drop for Kept<K, V> {
    /// The values fall one behind the other, and not each inside the fall
    /// of the one made before it: their fall takes no deeper a stack.
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(mut link) = next {
            next = link.behind.0.take();
        }
    }
}

impl<K: Clone + PartialEq, V> Kept<K, V> {
    /// The value kept under `key`, which `make` makes where none stands.
    /// Where several ask at once for a value that does not stand, one of
    /// them makes it and the others wait for it and read the same one.
    pub(super) fn under(&self, key: &K, make: impl Fn() -> V) -> &V {
        let mut kept = self;
        loop {
            let link = kept.0.get_or_init(|| {
                Box::new(Link {
                    key: key.clone(),
                    value: make(),
                    behind: Self::default(),
                })
            });
            if link.key == *key {
                return &link.value;
            }
            kept = &link.behind;
        }
    }
}

impl Members {
    /// The pair classes of the generation's prefill and decode lists under
    /// `key`: the holder's count of in-place moves as the ask read it, and
    /// a pairing mode. `derive` derives them where no set stands under that
    /// key.
    /// A set made under another key stays where it is until the generation
    /// falls.
    pub(in crate::worker_host) fn pair_classes(
        &self,
        key: (u64, Pairing),
        derive: impl Fn() -> PartnerClasses,
    ) -> &PartnerClasses {
        self.pairs.under(&key, derive)
    }
}

/// A pair's admission as a controller reads it: a prefill always has
/// capacity and a decode always has a room.
///
/// A controller knows no admission window of a gateway. So it reads both
/// bits open, as the rows its store builds state them, and the gateway that
/// asks gates capacity at its own commit.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAdmission;

impl PairAdmission for OpenAdmission {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, ptr};

    use super::*;

    /// Values kept under keys: each is made once and stands where it is, and
    /// another key makes another behind it.
    #[test]
    fn a_value_is_kept_under_its_key_and_made_once() {
        let kept = Kept::<u8, String>::default();
        let made = Cell::new(0);
        let make = |name: &'static str| {
            let made = &made;
            move || {
                made.set(made.get() + 1);
                name.to_string()
            }
        };
        let first = kept.under(&1, make("one"));
        let second = kept.under(&2, make("two"));
        assert_eq!((first.as_str(), second.as_str()), ("one", "two"));
        assert!(ptr::eq(first, kept.under(&1, make("one again"))));
        assert!(ptr::eq(second, kept.under(&2, make("two again"))));
        assert_eq!(made.get(), 2, "a value that stands is not made again");
    }

    /// The values of a generation that stood long fall one behind the other,
    /// and a print counts them: neither takes a deeper stack than one value's.
    #[test]
    fn a_long_line_of_kept_values_falls_on_a_flat_stack() {
        let kept = Kept::<u32, u32>::default();
        let mut last = &kept;
        for n in 0..200_000 {
            let link = Link {
                key: n,
                value: n,
                behind: Kept::default(),
            };
            last = &last.0.get_or_init(|| Box::new(link)).behind;
        }
        assert_eq!(*kept.under(&7, || 0), 7);
        assert_eq!(format!("{kept:?}"), "Kept { values: 200000 }");
        drop(kept);
    }
}
