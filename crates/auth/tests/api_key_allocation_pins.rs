//! What a control-plane key lookup allocates, pinned at today's count, so
//! the next change that copies the matched entry fails a test rather than
//! moving a number inside a benchmark's noise. A separate executable, so it
//! can declare the counting allocator the pin needs (`common`).

#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
mod common;

use common::counting_alloc::{allocations_during, Counting};
use smg_auth::{ApiKeyEntry, ApiKeySet, Role};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// A lookup hashes the presented key on the stack and reads the matched
/// entry through the set's snapshot: nothing is allocated, whether the key
/// matches the first entry, the last one or none.
#[test]
fn a_key_lookup_allocates_nothing() {
    let set = ApiKeySet::new(vec![
        ApiKeyEntry::new("ops", "Ops", "ops-key", Role::Admin),
        ApiKeyEntry::new("reader", "Reader", "reader-key", Role::User),
        ApiKeyEntry::new("ci", "CI", "ci-key", Role::User),
    ]);
    // Warm: whatever a first lookup builds once is not the pin's.
    assert!(set.find("ci-key").is_some());
    let (found, last) = allocations_during(|| set.find("ci-key"));
    assert_eq!(found.map(|entry| entry.id.clone()).as_deref(), Some("ci"));
    let (found, first) = allocations_during(|| set.find("ops-key"));
    assert_eq!(found.map(|entry| entry.id.clone()).as_deref(), Some("ops"));
    let (missed, miss) = allocations_during(|| set.find("no-such-key"));
    assert!(missed.is_none());
    eprintln!(
        "allocation calls per key lookup: matching the last entry {last}, the first entry {first}, none {miss}"
    );
    assert_eq!(last, 0, "a lookup that matches the last entry");
    assert_eq!(first, 0, "a lookup that matches the first entry");
    assert_eq!(miss, 0, "a lookup that matches no entry");
}
