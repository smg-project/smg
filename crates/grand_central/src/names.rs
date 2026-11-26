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
    // The cases' count of calls stays here, since only this function sees
    // every call: a case reads it to show that a placement asks for no name.
    #[cfg(test)]
    tests::ASKED.with(|asked| asked.set(asked.get() + 1));
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
#[path = "../tests/unit/names/tests.rs"]
pub(crate) mod tests;
