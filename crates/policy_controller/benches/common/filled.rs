//! Where a bench's fill ends: the store lists the rows the fill registered.
//!
//! A membership write leaves its build owed, so a fill that asks for no
//! build would time pools with no rows, and no failure would say so. A
//! bench that fills a store through its doors ends the fill here.

use smg_policy_controller::{engine::Role, store::MemoryStore};

/// The pool of `model` and `role` lists `rows` members.
pub fn filled(store: &MemoryStore, model: &str, role: Role, rows: usize) {
    let listed = store.snapshot().members(model, role);
    assert_eq!(
        listed,
        Some(rows),
        "the {role:?} pool of {model} after the fill"
    );
}
