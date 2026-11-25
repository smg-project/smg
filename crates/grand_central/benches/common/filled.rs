//! Where a bench's fill ends: the store holds the rows the fill registered,
//! and their build is paid.
//!
//! A membership write leaves its build owed, so a fill that asks for no
//! build would time a world built of no member, and no failure would say
//! so. A bench that fills a store through its doors ends the fill here.

use smg_grand_central::{
    engine::Role,
    store::{MemoryStore, WorkersStore},
};

/// `rows` members on file serve `model` in `role`, and no build is owed.
pub fn filled(store: &MemoryStore, model: &str, role: Role, rows: usize) {
    let (_, on_file) = WorkersStore::all_rows(store);
    let serving = on_file
        .iter()
        .filter(|row| row.facts.role == role && row.facts.models.iter().any(|named| named == model))
        .count();
    assert_eq!(
        serving, rows,
        "the {role:?} members of {model} after the fill"
    );
    assert_eq!(
        store.publish_due(),
        None,
        "the fill of {model} left its build owed"
    );
}
