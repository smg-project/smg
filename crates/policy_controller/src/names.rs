//! Names kept for the life of the process.

use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

/// `name` as a string that lives as long as the process. A trace, a shed
/// and a metric's label carry `&'static str` names, and a name read from a
/// document is not one: it is kept here, one allocation a distinct name
/// however often it is asked for. A spec that reloads a plugin under the
/// same name reuses the allocation.
pub(crate) fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    #[cfg(test)]
    ASKED.with(|asked| asked.set(asked.get() + 1));
    let names = NAMES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = names
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(known) = guard.get(name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    guard.insert(leaked);
    leaked
}

#[cfg(test)]
thread_local! {
    /// How many names this thread asked [`intern`] for: a case reads it
    /// around a call to show that the call takes no name from the table,
    /// and so not the table's lock.
    pub(crate) static ASKED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::intern;

    #[test]
    fn a_name_is_interned_once() {
        let a = intern("plugin-a");
        let b = intern("plugin-a");
        assert!(std::ptr::eq(a, b), "one allocation per distinct name");
        assert_ne!(intern("plugin-b"), a);
    }
}
