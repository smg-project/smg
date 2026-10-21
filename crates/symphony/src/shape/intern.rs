//! One copy of each distinct value, for the life of the process.

use std::{
    collections::HashSet,
    hash::{BuildHasherDefault, DefaultHasher, Hash},
    sync::{Mutex, PoisonError},
};

/// One copy of each distinct value, kept for the life of the process: a value built at run time,
/// a call's shape compiled from a model's template, becomes `&'static` data, so that a type that
/// holds it stays `Copy` as one that holds a table written in the crate does.
///
/// A value seen before is not stored again: the first copy is returned and the new one dropped. So
/// what is kept is bounded by the distinct values a process builds (the templates of the models it
/// loads, whose tables are built once per tokenizer), and nothing is freed. An interner is a
/// `static`, one per type it keeps.
pub struct Interner<T: 'static> {
    values: Mutex<HashSet<&'static T, BuildHasherDefault<DefaultHasher>>>,
}

impl<T: Eq + Hash + Sync + 'static> Interner<T> {
    /// An interner that holds nothing yet.
    pub const fn new() -> Self {
        Self {
            values: Mutex::new(HashSet::with_hasher(BuildHasherDefault::new())),
        }
    }

    /// The process's one copy of `value`.
    pub fn intern(&self, value: T) -> &'static T {
        // A panic elsewhere while the lock was held leaves the set as it was: every insert is one
        // step, so the set is never half-written and is safe to keep using.
        let mut values = self.values.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&kept) = values.get(&value) {
            return kept;
        }
        let kept: &'static T = Box::leak(Box::new(value));
        values.insert(kept);
        kept
    }
}

impl<T: Eq + Hash + Sync + 'static> Default for Interner<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{ptr, thread};

    use super::*;
    use crate::shape::{Atom, Pattern};

    fn literal(text: &str) -> Pattern {
        Pattern([Atom::Lit(text.into())].into())
    }

    #[test]
    fn equal_values_share_one_copy_and_different_ones_do_not() {
        static PATTERNS: Interner<Pattern> = Interner::new();
        let first = PATTERNS.intern(literal("<a>"));
        let again = PATTERNS.intern(literal("<a>"));
        let other = PATTERNS.intern(literal("</a>"));
        assert!(ptr::eq(first, again), "one copy of an equal value");
        assert!(!ptr::eq(first, other));
        assert_eq!(*first, literal("<a>"));
        assert_eq!(*other, literal("</a>"));
    }

    #[test]
    fn threads_that_intern_one_value_get_the_one_copy() {
        static PATTERNS: Interner<Pattern> = Interner::new();
        let kept: Vec<usize> = thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| ptr::from_ref(PATTERNS.intern(literal("<k>"))) as usize))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("an interning thread panicked"))
                .collect()
        });
        assert!(kept.windows(2).all(|pair| pair[0] == pair[1]), "{kept:?}");
    }
}
