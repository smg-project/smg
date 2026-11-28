//! Under a shared lease: the source writes while its replica leads, and
//! the fence is read again behind a pass and behind each lock.

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use super::{
    super::{Generations, Leading},
    applied, election, election_under, followed, generations, generations_under, sent, yaml,
};
use crate::{
    document::{
        applied::AppliedSpec,
        operator::{
            test_support::{bound_to, standing, stock, LEAST_LOAD, RANDOM},
            ApplyOutcome,
        },
        spec::{Digest, PolicySpec, SpecError},
    },
    service::elector::Elector,
    stages::registry::StageRegistry,
    store::MemoryStore,
};

/// Under a shared lease the door holds who writes: its source, while
/// the replica's fence shows a term. A document the source hands while
/// the replica does not lead is not read, whether it runs or not and
/// whatever its watch saw ahead of a look, and nothing is counted. The
/// log writes there all the same.
#[test]
fn under_a_shared_lease_the_source_writes_while_its_replica_leads() {
    let mut elector = election();
    let (store, door) = generations_under(elector.fence());
    assert!(!door.source_writes(), "it has not led yet");
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
    assert_eq!(standing(&store), (0, None));
    elector.tick(Instant::now()).expect("a take");
    assert!(door.source_writes());
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
    // Its lead ends: a look that was in flight applies nothing.
    assert!(elector.release().expect("a release"), "it held the lease");
    assert_eq!(yaml(&door, LEAST_LOAD), Ok(ApplyOutcome::Waits(1)));
    assert_eq!(standing(&store).0, 1);
    assert_eq!(door.apply_counts(), applied(1, 0));
    // One that does not run waits all the same: it is not read, so it
    // is not refused.
    let unrun = LEAST_LOAD.replace("healthy", "no-such-stage");
    assert_eq!(yaml(&door, unrun), Ok(ApplyOutcome::Waits(1)));
    assert_eq!(door.apply_counts(), applied(1, 0));
    let (_, leader) = generations();
    yaml(&leader, LEAST_LOAD).expect("a valid document");
    yaml(&leader, bound_to(1)).expect("a changed document");
    followed(&door, &sent(&leader)).expect("above its own");
    assert_eq!(standing(&store).0, 2);
    // A replica alone over its lease has no fence here: its source
    // writes always.
    let (_, alone) = generations();
    assert!(alone.source_writes());
}

/// A lead that ends while a document is passed applies nothing: the
/// source's entry asks who writes again behind the pass, as a decision
/// reads its fence again where its answer is handed over.
#[test]
fn a_lead_that_ends_under_a_documents_pass_applies_nothing() {
    thread_local! {
        static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
    }
    fn ending(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
        assert!(
            matches!(left, Some(Ok(true))),
            "it held the lease: {left:?}"
        );
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let mut elector = election();
    let fence = Some(elector.fence());
    let door = Generations::validating(Arc::clone(&store), stock(), ending, fence);
    elector.tick(Instant::now()).expect("a take");
    assert!(door.source_writes());
    LEAD.set(Some(elector));
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
    assert_eq!(standing(&store), (0, None));
    assert_eq!(door.apply_counts(), applied(0, 0));
}

/// The lock a case holds while an entry of the door's own source waits
/// behind it.
enum Locked {
    /// The door's, of the held bytes: the entry has passed both of the
    /// door's reads of the fence and waits in the door.
    Held,
    /// The store's last, of the loads, as a membership write holds it:
    /// the entry waits inside the store.
    Loads,
}

/// An entry under a lead whose fence closes by the clock while the
/// entry waits behind `locked`: nothing is published, the standing
/// generation stays and no bytes are kept.
fn a_fence_closes_behind(locked: Locked) {
    let deadline = Duration::from_millis(800);
    let mut elector = election_under(deadline);
    let (store, door) = generations_under(elector.fence());
    elector.tick(Instant::now()).expect("a take");
    assert!(door.source_writes());
    let (held, loads) = match locked {
        Locked::Held => (Some(door.held_lock().lock()), None),
        Locked::Loads => (None, Some(store.holding_loads())),
    };
    let outcome = thread::scope(|scope| {
        let entering = scope.spawn(|| yaml(&door, RANDOM));
        thread::sleep(deadline + Duration::from_millis(400));
        assert!(!door.source_writes(), "the fence closed by the clock");
        drop((held, loads));
        entering.join().expect("the entry's thread")
    });
    assert_eq!(outcome, Ok(ApplyOutcome::Waits(0)));
    assert_eq!(standing(&store), (0, None));
    assert_eq!(door.apply_counts(), applied(0, 0));
    assert!(!door.holds(), "no bytes are kept");
}

/// A fence that closes while the entry waits for the door's lock of
/// the held bytes publishes nothing: the store asks its last word
/// behind the locks its entry waits for.
#[test]
fn a_fence_that_closes_while_the_entry_waits_for_a_lock_publishes_nothing() {
    a_fence_closes_behind(Locked::Held);
}

/// The store asks its last word behind every lock its entry waits for.
/// An entry that waits for the loads' lock, the last one, while the
/// fence closes publishes nothing.
#[test]
fn a_fence_that_closes_while_the_store_waits_for_its_last_lock_publishes_nothing() {
    a_fence_closes_behind(Locked::Loads);
}

/// The standing document passed again under a lead that ends changes
/// nothing and keeps no bytes: the fence is read behind a pass ahead of
/// both of its ends.
#[test]
fn the_standing_document_passed_under_a_lead_that_ends_waits() {
    thread_local! {
        static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
        static PASSES: Cell<u32> = const { Cell::new(0) };
    }
    // Under its second pass the replica gives its lease up.
    fn ending(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 1 {
            let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
            assert!(
                matches!(left, Some(Ok(true))),
                "it held the lease: {left:?}"
            );
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let mut elector = election();
    let fence = Some(elector.fence());
    let door = Generations::validating(Arc::clone(&store), stock(), ending, fence);
    elector.tick(Instant::now()).expect("a take");
    LEAD.set(Some(elector));
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(1)));
    assert_eq!(door.apply_counts(), applied(1, 0));
}

/// Bytes read under one term are not applied under the next: an entry
/// is admitted under the lead its source took ahead of the read, and
/// the source reads again.
#[test]
fn bytes_read_under_one_term_are_not_applied_under_the_next() {
    thread_local! {
        static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
        static PASSES: Cell<u32> = const { Cell::new(0) };
    }
    // Under its first pass the replica is deposed and takes again.
    fn retaking(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 0 {
            LEAD.with_borrow_mut(|lead| {
                let lead = lead.as_mut().expect("an elector");
                assert!(matches!(lead.release(), Ok(true)), "it held the lease");
                lead.tick(Instant::now()).expect("a second take");
            });
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let mut elector = election();
    let fence = Some(elector.fence());
    let door = Generations::validating(Arc::clone(&store), stock(), retaking, fence);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(door.lead(), Some(Leading::Term(1)));
    LEAD.set(Some(elector));
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
    assert_eq!(standing(&store), (0, None));
    assert_eq!(door.lead(), Some(Leading::Term(2)), "it leads again");
    // Read again under the term that shows now, the bytes are applied.
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
    assert_eq!(door.apply_counts(), applied(1, 0));
}
