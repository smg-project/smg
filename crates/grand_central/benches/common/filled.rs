//! Where a bench's fill ends: the fill named no member twice, and the build
//! of its rows is paid.
//!
//! A membership write leaves its build owed, so a fill that asks for no
//! build would time a world built of no member, and no failure would say
//! so. A bench that fills a store through its doors ends the fill here.
//!
//! A store gives its host no read of a row. That each write of a fill
//! landed is the fill's own word: the store's epoch counts the writes it
//! took. A holder of worker objects cuts the rows that it and the store
//! both hold.

use std::collections::BTreeSet;

use smg_grand_central::{engine::WorkerId, store::MemoryStore};

/// No id of the fill of `model` is named twice, and no build is owed.
pub fn filled(store: &MemoryStore, ids: &[WorkerId], model: &str) {
    let distinct: BTreeSet<WorkerId> = ids.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        ids.len(),
        "the fill of {model} names a member twice"
    );
    assert_eq!(
        store.publish_due(),
        None,
        "the fill of {model} left its build owed"
    );
}
