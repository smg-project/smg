//! A decision's claim on a key: the claim of record, the commit that
//! confirms or withdraws it, the claim a decision moves, the word of a lease
//! that reaches it, the tie of a decode call's claim to its lease's first
//! claim, and the life of a key a dated write goes into.

use super::*;

/// A retry's claim survives the commit of an older request on the owner:
/// only the claiming decision settles its claim, so the retries after it
/// read the claim as the pin of record instead of claiming elsewhere, and
/// the claim is confirmed when its own commit lands.
#[test]
fn an_older_requests_commit_on_the_owner_leaves_a_retrys_claim_standing() {
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 0, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(1), 0),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1), 0);
    // An older request decided on the owner is still in flight when a
    // retry that cannot use the owner claims B.
    let seen = table.lookup(s(1), 10, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(2)]),
        "the older request's commit on the owner leaves the claim standing"
    );
    // The next retry reads B as the pin of record; the claiming retry's
    // own commit confirms it.
    assert_eq!(
        table.lookup(s(1), 20, &ORIGINAL),
        [WorkerId(1), WorkerId(2)]
    );
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
}

#[test]
fn the_first_claim_stands_for_a_fresh_key_and_for_a_key_whose_pins_are_down() {
    let table = SessionTable::new(RULE.sweep_micros);
    let none = [WorkerId(0); 2];
    let seen = table.lookup(s(5), 10, &ORIGINAL);
    assert_eq!(
        table.claim(s(5), seen, WorkerId(3), 0),
        Claim::Made(WorkerId(3)),
        "the first"
    );
    assert_eq!(
        table.claim(s(5), none, WorkerId(4), 0),
        Claim::Standing(WorkerId(3), [WorkerId(3), WorkerId(0)]),
        "follows it"
    );
    assert_eq!(table.pins(s(5)), Some([WorkerId(3), WorkerId(0)]));
    table.dispatched(s(5), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3), 0);
    table.dispatched(s(5), SessionVerdict::Pinned, WorkerId(4), 0);
    assert_eq!(
        table.pins(s(5)),
        Some([WorkerId(3), WorkerId(4)]),
        "a loser dispatched elsewhere is the failover"
    );
    // Both pins down: two decisions read [3, 4]; the first claim appends
    // its worker by the recovery (the oldest slot goes), the second
    // follows the newcomer.
    let seen = table.lookup(s(5), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(5), seen, WorkerId(7), 0),
        Claim::Made(WorkerId(7))
    );
    assert_eq!(table.pins(s(5)), Some([WorkerId(4), WorkerId(7)]));
    assert_eq!(
        table.claim(s(5), seen, WorkerId(8), 0),
        Claim::Standing(WorkerId(7), [WorkerId(4), WorkerId(7)]),
        "the newcomer"
    );
    // Under the latest-first recovery the claim leads at once.
    let seen = table.lookup(s(6), 30, &RULE);
    assert_eq!(
        table.claim(s(6), seen, WorkerId(1), 0),
        Claim::Made(WorkerId(1))
    );
    assert_eq!(table.pins(s(6)), Some([WorkerId(1), WorkerId(0)]));
    assert_eq!(
        table.claim(s(0), none, WorkerId(9), 0),
        Claim::Standing(WorkerId(9), none),
        "no key, no claim"
    );
    assert_eq!(table.len(), 2);
}

/// Another request's respill turns the pins behind a decision's read:
/// the claim is not taken and no pin came in. The answer is the
/// caller's own worker with the pins the key held, and nothing is
/// recorded. Another decision that reads the same pins gets its own
/// worker: the answer is no pin to follow.
#[test]
fn a_claim_that_finds_no_new_pin_names_the_callers_own_worker() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1), 0);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2), 0);
    let seen = table.lookup(s(3), 20, &RULE);
    assert_eq!(seen, [WorkerId(2), WorkerId(1)]);
    table.dispatched(s(3), SessionVerdict::Respill, WorkerId(1), 0);
    let held = [WorkerId(1), WorkerId(2)];
    assert_eq!(
        table.claim(s(3), seen, WorkerId(7), 0),
        Claim::Standing(WorkerId(7), held)
    );
    assert_eq!(table.pins(s(3)), Some(held), "nothing is recorded");
    assert_eq!(
        Claim::standing(&held, &seen, WorkerId(8)),
        Some(WorkerId(8))
    );
    assert_eq!(Claim::standing(&seen, &seen, WorkerId(8)), None);
}

/// The owner is down, the decision claims B, the commit refuses B and
/// dispatches the alternate C: the claim is withdrawn and the owner keeps
/// its slot (B served nothing); a dispatch to the claimed worker confirms
/// the claim instead, and the normal rule follows after it.
#[test]
fn a_claim_whose_commit_landed_elsewhere_is_withdrawn_before_the_pins_move() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let seen = table.lookup(s(1), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3), 0);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(3)]),
        "the owner keeps the lead, the alternate is the failover"
    );
    // Under the latest-first recovery the claim led; the alternate leads
    // instead and the owner stays the failover.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(2), 10, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1), 0);
    let seen = table.lookup(s(2), 20, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(3), WorkerId(1)]));
    // A dispatch to the claimed worker confirms the claim.
    let seen = table.lookup(s(2), 30, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(4), 0),
        Claim::Made(WorkerId(4))
    );
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(4)), WorkerId(4), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(4), WorkerId(3)]));
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(5), 0);
    assert_eq!(
        table.pins(s(2)),
        Some([WorkerId(5), WorkerId(4)]),
        "a confirmed claim served: the normal rule moves it"
    );
}

#[test]
fn a_pin_another_request_already_used_survives_the_claimants_commit_elsewhere() {
    // X claims A for a fresh key; Y finds A pinned and commits there; A
    // goes unroutable and X's own commit lands on B. A was served, so it
    // stays: [A, B], and the recovered A keeps the key.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 0, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(1), 0),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(2)]),
        "a pin another request already used survives the claimant's commit elsewhere"
    );
    assert_eq!(
        table.lookup(s(1), 10, &ORIGINAL),
        [WorkerId(1), WorkerId(2)]
    );
    // Under the latest-first recovery the alternate leads and A stays the
    // failover instead of vanishing.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(2), 0, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(1), 0),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1), 0);
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
}

#[test]
fn a_commit_elsewhere_by_another_request_leaves_the_claim_unconfirmed() {
    // The owner is pinned; a retry claims C; an older request commits on
    // the owner (not on C): the claim stands unconfirmed, so the retry's
    // own commit on D still withdraws C.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let seen = table.lookup(s(1), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3), 0);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(3)]),
        "a commit elsewhere by another request confirms nothing"
    );
}

/// A decision that does not go to the worker it claimed moves the
/// claim while it stands unconfirmed: the worker claimed leaves the
/// pins, and the worker the request goes to is the claim in its place,
/// by the key's recovery. With no worker to go to, the claim is
/// withdrawn alone and the key is claimed afresh. A claim a dispatch
/// confirmed stands, and a worker that is a pin already takes no claim.
#[test]
fn a_claim_moves_to_the_worker_its_decision_goes_to_while_it_stands_unconfirmed() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let cases = [
        (&RULE, [2, 1], [3, 1], [4, 1]),
        (&ORIGINAL, [1, 2], [1, 3], [1, 4]),
    ];
    for (rule, claimed, moved, committed) in cases {
        // The owner is down: the decision claims 2, then goes to 3.
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, rule);
        table.dispatched(s(1), pinned, id(1), 0);
        let seen = table.lookup(s(1), 20, rule);
        assert_eq!(table.claim(s(1), seen, id(2), 0), Claim::Made(id(2)));
        // A decision whose claim is not the one of record moves none.
        assert_eq!(table.move_claim(s(1), id(9), id(3), 0), pinned);
        assert_eq!(table.pins(s(1)), Some(claimed.map(id)));
        let moved_to = SessionVerdict::Claimed(id(3));
        assert_eq!(table.move_claim(s(1), id(2), id(3), 0), moved_to);
        assert_eq!(table.pins(s(1)), Some(moved.map(id)));
        // The worker it moved to is the claim: the decision's commit
        // elsewhere withdraws that one.
        table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(4), 0);
        assert_eq!(table.pins(s(1)), Some(committed.map(id)));
        // No worker to go to: the claim is withdrawn alone, and the
        // key's next decision claims afresh.
        let seen = table.lookup(s(2), 10, rule);
        assert_eq!(table.claim(s(2), seen, id(2), 0), Claim::Made(id(2)));
        assert_eq!(table.move_claim(s(2), id(2), id(0), 0), pinned);
        assert_eq!(table.pins(s(2)), Some([id(0); 2]));
        assert_eq!(table.claim(s(2), seen, id(5), 0), Claim::Made(id(5)));
        // Another request was served on the claim: it stands.
        table.dispatched(s(2), pinned, id(5), 0);
        assert_eq!(table.move_claim(s(2), id(5), id(6), 0), pinned);
        assert_eq!(table.pins(s(2)), Some([id(5), id(0)]));
        // A key the table does not hold is not written.
        assert_eq!(table.move_claim(s(3), id(1), id(2), 0), pinned);
        assert_eq!(table.len(), 2);
    }
    // Another request's commit made 3 a pin behind the claim on 2: the
    // claim leaves, and 3 stands as the pin it is, marked no claim. A
    // claimant's commit elsewhere then withdraws nothing.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 0), Claim::Made(id(2)));
    table.dispatched(s(1), pinned, id(3), 0);
    assert_eq!(table.pins(s(1)), Some([id(3), id(2)]));
    assert_eq!(table.move_claim(s(1), id(2), id(3), 0), pinned);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]));
    table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(9), 0);
    assert_eq!(table.pins(s(1)), Some([id(9), id(3)]));
    // Two other requests' commits pushed the claim on 2 out of the
    // pins: withdrawn to no worker, it leaves them as they stand.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &ORIGINAL);
    assert_eq!(table.claim(s(1), seen, id(2), 0), Claim::Made(id(2)));
    table.dispatched(s(1), pinned, id(3), 0);
    table.dispatched(s(1), pinned, id(4), 0);
    assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 0), pinned);
    assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
}

/// A claim is known by its worker and by the instant its lease is
/// issued under. A decision's claim goes, and a later decision claims
/// the same worker. A word of the older lease then reaches no claim: its
/// refusal told late and its dispatch elsewhere told late both leave the
/// newer claim. The newer lease's own refusal takes that claim back.
///
/// The later decision comes inside the key's life, at 20, or behind its
/// idle time, at 1,200. Inside one life the older lease's dispatch pins
/// the worker it landed on. Behind the idle time it is an older life's
/// and pins nothing.
#[test]
fn a_word_of_an_older_lease_leaves_a_newer_claim_on_the_same_worker() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    for (rule, later) in [&RULE, &ORIGINAL]
        .into_iter()
        .flat_map(|rule| [(rule, 20), (rule, 1_200)])
    {
        let table = SessionTable::new(RULE.sweep_micros);
        // The older decision claims 2 at 10. Its commit refuses every
        // target, and the claim goes.
        let seen = table.lookup(s(1), 10, rule);
        assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
        assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
        // A later decision claims 2 again.
        let seen = table.lookup(s(1), later, rule);
        assert_eq!(seen, [id(0); 2]);
        assert_eq!(table.claim(s(1), seen, id(2), later), Claim::Made(id(2)));
        // The older lease's refusal, told late: the newer claim stands.
        assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
        assert_eq!(table.pins(s(1)), Some([id(2), id(0)]));
        // The older lease's dispatch elsewhere, told late, withdraws
        // nothing. In the life it read, its worker joins the pins by the
        // key's recovery. In a later life it pins nothing.
        table.dispatched(s(1), SessionVerdict::Claimed(id(2)), id(3), 10);
        let (joined, left) = match (later, rule.recovery) {
            (20, Recovery::Latest) => ([id(3), id(2)], [id(3), id(0)]),
            (20, Recovery::Original) => ([id(2), id(3)], [id(3), id(0)]),
            _ => ([id(2), id(0)], [id(0); 2]),
        };
        assert_eq!(table.pins(s(1)), Some(joined), "{later}");
        // The newer lease's own refusal takes its claim back.
        assert_eq!(table.move_claim(s(1), id(2), id(0), later), pinned);
        assert_eq!(table.pins(s(1)), Some(left), "{later}");
    }
}

/// A word that states no instant reaches a claim by its worker alone:
/// a lease whose host dates none is told so. A claim whose decision
/// stated no instant is left by a word dated otherwise.
#[test]
fn a_word_that_states_no_instant_reaches_a_claim_by_its_worker() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    // A dated claim, and the refusal of a lease that states no date.
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 0), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]));
    // The same for such a lease's dispatch elsewhere.
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    table.dispatched(s(1), SessionVerdict::Claimed(id(2)), id(3), 0);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]));
    // A claim that states no date: a word dated otherwise leaves it,
    // and a word that states none takes it back.
    let seen = table.lookup(s(2), 10, &RULE);
    assert_eq!(table.claim(s(2), seen, id(2), 0), Claim::Made(id(2)));
    assert_eq!(table.move_claim(s(2), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(2)), Some([id(2), id(0)]));
    assert_eq!(table.move_claim(s(2), id(2), id(0), 0), pinned);
    assert_eq!(table.pins(s(2)), Some([id(0); 2]));
}

/// A claim its decision moves keeps the instant of that decision's
/// lease: another lease's refusal leaves the moved claim, and its own
/// takes it back.
#[test]
fn a_moved_claim_keeps_the_instant_of_its_decisions_lease() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let moved = SessionVerdict::Claimed(id(3));
    assert_eq!(table.move_claim(s(1), id(2), id(3), 10), moved);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]));
    assert_eq!(table.move_claim(s(1), id(3), id(0), 20), pinned);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "another's");
    assert_eq!(table.move_claim(s(1), id(3), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]), "its own");
}

/// A withdrawn claim leaves no mark for its lease's next word. A commit
/// every target refused withdraws its decision's claim. Another decision
/// is served elsewhere, and a third meets that pin at its cap and moves
/// to the worker the claim was on. The refusal told again finds no claim
/// of record, and the move's pin stands.
#[test]
fn a_withdrawn_claims_lease_told_again_moves_nothing() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]));
    table.lookup(s(1), 20, &RULE);
    table.dispatched(s(1), pinned, id(3), 20);
    table.lookup(s(1), 30, &RULE);
    table.respilled(s(1), id(2), 30);
    assert_eq!(table.pins(s(1)), Some([id(2), id(3)]));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(2), id(3)]), "the move's pin");
}

/// The claim's date is the low 32 bits of its lease's instant, so the
/// map's slot keeps its 72 bytes: eight for the key and 64 for its entry,
/// whose fields take 61. Two claims of one key on one worker a whole 2^32
/// microseconds apart read as one: the older lease's word reaches the
/// newer claim, the answer by the worker alone.
#[test]
fn a_claims_date_is_32_bits_and_the_maps_slot_keeps_its_size() {
    assert_eq!(size_of::<(u64, Entry)>(), 72);
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let later = 10 + (1_u64 << 32);
    let seen = table.lookup(s(1), later, &RULE);
    assert_eq!(seen, [id(0); 2], "the key started over");
    assert_eq!(table.claim(s(1), seen, id(2), later), Claim::Made(id(2)));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]));
}

/// A decode call claims at its own instant, under the lease its pair's
/// first decision issued. The claim is dated again by that lease, and
/// the lease's refusal then takes it back. A claim on another worker
/// keeps its date. So does a claim another decision made in between:
/// the call's second write leaves it, and the lease's refusal with it.
/// A lease that states no instant dates nothing.
#[test]
fn a_decode_calls_claim_takes_the_date_of_its_pairs_lease() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    // The pair's lease is of 10, and the call claims 2 at 40.
    let seen = table.lookup(s(1), 40, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 40), Claim::Made(id(2)));
    // A lease that states no instant dates nothing: a word of the
    // call's own date still reaches the claim.
    table.told_under(s(1), Some(id(2)), 40, 0, id(0));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 40), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]), "the call's date");
    assert_eq!(table.claim(s(1), seen, id(2), 40), Claim::Made(id(2)));
    // A write that names another worker dates no claim.
    table.told_under(s(1), Some(id(3)), 40, 10, id(0));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(2), id(0)]), "dated by the call");
    // The call's own claim takes the lease's date.
    table.told_under(s(1), Some(id(2)), 40, 10, id(0));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 40), pinned);
    assert_eq!(
        table.pins(s(1)),
        Some([id(2), id(0)]),
        "by the call no more"
    );
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]), "the lease's refusal");
    // Another decision claims 2 at 50, ahead of a call's second write:
    // that claim keeps its own date.
    assert_eq!(table.claim(s(1), seen, id(2), 50), Claim::Made(id(2)));
    table.told_under(s(1), Some(id(2)), 40, 10, id(0));
    assert_eq!(table.move_claim(s(1), id(2), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(2), id(0)]), "another's claim");
    assert_eq!(table.move_claim(s(1), id(2), id(0), 50), pinned);
    assert_eq!(table.pins(s(1)), Some([id(0); 2]), "its own lease's");
    // A key the table does not hold is not written.
    table.told_under(s(9), Some(id(2)), 40, 10, id(0));
    assert_eq!(table.len(), 1);
}

/// A key that started over between a decision and its dispatch keeps its
/// new life's pins alone, under either recovery. The dispatch is dated
/// before the decision that began the new life. It pins nothing, whatever
/// its verdict and wherever it landed. It confirms no claim of the new
/// life: that claim's own decision still takes it back.
///
/// A gateway's sticky map pins where it selects a worker, and a send
/// writes nothing there (`ManualPolicy::select_by_routing_id` and
/// `ManualPolicy::pin_front`, `model_gateway/src/policies/manual.rs:269-368`).
/// A send that comes late leaves a key's new entry as its own selections
/// made it.
#[test]
fn a_dispatch_of_an_older_life_writes_nothing_into_the_keys_new_one() {
    let id = WorkerId;
    let verdicts = [
        SessionVerdict::Pinned,
        SessionVerdict::Respill,
        SessionVerdict::Claimed(id(1)),
        SessionVerdict::Claimed(id(2)),
    ];
    for rule in [RULE, ORIGINAL] {
        for verdict in verdicts {
            let table = SessionTable::new(rule.sweep_micros);
            // The older life: a decision of 10 claims worker 1.
            let seen = table.lookup(s(7), 10, &rule);
            assert_eq!(table.claim(s(7), seen, id(1), 10), Claim::Made(id(1)));
            // Its idle time runs out. A decision of 5,000 starts the key
            // over and claims worker 2.
            let seen = table.lookup(s(7), 5_000, &rule);
            assert_eq!(seen, [id(0); 2], "the key started over");
            assert_eq!(table.claim(s(7), seen, id(2), 5_000), Claim::Made(id(2)));
            // The older decision's send comes now: on its own worker, on
            // the new life's claim, and on a third.
            for target in [id(1), id(2), id(3)] {
                table.dispatched(s(7), verdict, target, 10);
                let pins = table.pins(s(7));
                assert_eq!(pins, Some([id(2), id(0)]), "{verdict:?} {target:?}");
            }
            // The new life's claim stands unconfirmed: its own decision's
            // commit elsewhere takes it back.
            table.dispatched(s(7), SessionVerdict::Claimed(id(2)), id(4), 5_000);
            assert_eq!(table.pins(s(7)), Some([id(4), id(0)]), "{verdict:?}");
        }
    }
}

/// The decision that begins a key's life tells its dispatch under the
/// instant the life began at, and that word is written. A word dated one
/// microsecond before it is an older life's. A word that states no
/// instant is written in any life, as a host that dates no lease tells it.
///
/// A gateway's sticky map pins a new key where its first selection picks
/// (`model_gateway/src/policies/manual.rs:355-366`).
#[test]
fn a_dispatch_dated_at_its_lifes_first_instant_is_written() {
    let id = WorkerId;
    for (rule, behind) in [(RULE, [id(4), id(3)]), (ORIGINAL, [id(3), id(4)])] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(7), 10, &rule);
        let seen = table.lookup(s(7), 5_000, &rule);
        assert_eq!(table.claim(s(7), seen, id(2), 5_000), Claim::Made(id(2)));
        // The first reader's commit elsewhere: its claim goes, and the
        // worker it landed on is pinned.
        table.dispatched(s(7), SessionVerdict::Claimed(id(2)), id(3), 5_000);
        assert_eq!(table.pins(s(7)), Some([id(3), id(0)]));
        table.dispatched(s(7), SessionVerdict::Pinned, id(5), 4_999);
        assert_eq!(table.pins(s(7)), Some([id(3), id(0)]), "one before");
        table.dispatched(s(7), SessionVerdict::Pinned, id(4), 0);
        assert_eq!(table.pins(s(7)), Some(behind), "no instant");
    }
    // The compare is of whole instants. A word 71 minutes older, whose
    // low 32 bits stand above the life's, is an older life's all the same.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(8), (1 << 32) + 5_000, &RULE);
    table.dispatched(s(8), SessionVerdict::Pinned, id(6), 10_000);
    assert_eq!(table.pins(s(8)), Some([id(0); 2]), "a whole instant");
}

/// A decision reads its clock ahead of its read of the table, so one of
/// an older instant can read a life that a later one began. It tells its
/// words under its own instant, and they are written: by the read that
/// renews a key, and by the read of a pin it follows. A word older than
/// every reader of the life is still an older life's.
///
/// A gateway's sticky map writes each selection's pin where it selects,
/// in the order its selections reach the map
/// (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn an_older_readers_dispatch_is_written_into_the_life_it_read() {
    let id = WorkerId;
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(7), 10, &RULE);
    // A decision of 5,200 begins the new life, and one of 5,100 reads it
    // behind that one.
    assert_eq!(table.lookup(s(7), 5_200, &RULE), [id(0); 2]);
    let seen = table.lookup(s(7), 5_100, &RULE);
    assert_eq!(table.claim(s(7), seen, id(2), 5_100), Claim::Made(id(2)));
    table.dispatched(s(7), SessionVerdict::Claimed(id(2)), id(3), 5_100);
    assert_eq!(table.pins(s(7)), Some([id(3), id(0)]), "the older reader");
    // A decision of 5,050 follows the pin, and its send lands elsewhere.
    assert_eq!(table.pins_renewed(s(7), 5_050, &RULE), [id(3), id(0)]);
    table.dispatched(s(7), SessionVerdict::Pinned, id(4), 5_050);
    assert_eq!(table.pins(s(7)), Some([id(4), id(3)]), "a followed pin");
    // A reader between the oldest and the newest leaves the date at the
    // oldest reader's: that reader's later word is still written.
    table.lookup(s(7), 5_150, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, id(5), 5_050);
    assert_eq!(table.pins(s(7)), Some([id(5), id(4)]), "a least");
    table.dispatched(s(7), SessionVerdict::Pinned, id(1), 5_049);
    assert_eq!(table.pins(s(7)), Some([id(5), id(4)]), "older than each");
    table.dispatched(s(7), SessionVerdict::Pinned, id(1), 10);
    assert_eq!(table.pins(s(7)), Some([id(5), id(4)]), "the older life");
}

/// A decode call reads its key at its own instant and tells its words
/// under its pair's lease, which is dated at the pair's first decision.
/// The key's life takes the lease's date as a reader's, so the call's
/// dispatch is written. It is written where the call began the life and
/// claimed. It is written too where the call followed a pin of a life
/// another request began. A lease that states no instant lowers nothing,
/// a newer lease raises nothing, and a call with no key writes nothing.
///
/// A gateway's sticky map pins a decode where the decode's selection
/// picks it, at that selection's time
/// (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn a_decode_calls_dispatch_is_written_under_its_pairs_lease() {
    let id = WorkerId;
    // The pair's lease is of 10. The call begins the key's life at 5,000
    // and claims worker 2.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(7), 10, &RULE);
    let seen = table.lookup(s(7), 5_000, &RULE);
    assert_eq!(seen, [id(0); 2], "the key started over");
    assert_eq!(table.claim(s(7), seen, id(2), 5_000), Claim::Made(id(2)));
    table.told_under(s(7), Some(id(2)), 5_000, 10, id(0));
    table.dispatched(s(7), SessionVerdict::Claimed(id(2)), id(3), 10);
    assert_eq!(table.pins(s(7)), Some([id(3), id(0)]), "with a claim");

    // Another request began the life at 4,000 and was served on worker
    // 2. The call follows that pin at 4,500 and its send lands elsewhere.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(7), 10, &RULE);
    table.lookup(s(7), 4_000, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, id(2), 4_000);
    assert_eq!(table.pins_renewed(s(7), 4_500, &RULE), [id(2), id(0)]);
    // A lease that states no instant is no reader's date, and neither is
    // a call's with no key.
    table.told_under(s(7), None, 4_500, 0, id(0));
    table.told_under(s(0), None, 4_500, 10, id(0));
    table.dispatched(s(7), SessionVerdict::Pinned, id(3), 10);
    assert_eq!(table.pins(s(7)), Some([id(2), id(0)]), "not told yet");
    table.told_under(s(7), None, 4_500, 10, id(0));
    table.dispatched(s(7), SessionVerdict::Pinned, id(3), 10);
    assert_eq!(table.pins(s(7)), Some([id(3), id(2)]), "with no claim");
    // The date fell to the lease's and no lower: a word one microsecond
    // older is an older life's.
    table.dispatched(s(7), SessionVerdict::Pinned, id(5), 9);
    assert_eq!(table.pins(s(7)), Some([id(3), id(2)]), "one before");
    // A later call under a newer lease leaves the oldest reader's date
    // where it stands: a word of the first call is still written.
    table.told_under(s(7), None, 4_600, 3_000, id(0));
    table.dispatched(s(7), SessionVerdict::Pinned, id(4), 10);
    assert_eq!(table.pins(s(7)), Some([id(4), id(3)]), "never rises");
    assert_eq!(table.len(), 1);
}

/// A decision reads its key and claims in two calls, and the key can
/// start over between them. Another decision, whose clock stands a whole
/// idle time ahead, reads the key first. The claim call reads the new
/// life's pins and answers by them, so that life takes the calling
/// decision's instant as a reader's. Its dispatch is then written.
///
/// On another worker the dispatch takes its claim back and pins that
/// worker. On a claim of the new life that the call followed, it
/// confirms that claim. A word dated before the call's instant is still
/// an older life's.
///
/// A gateway's sticky map reads and pins a key in one step where it
/// selects (`model_gateway/src/policies/manual.rs:322-368`). A selection
/// that reaches a new entry is that entry's own.
#[test]
fn a_claim_call_reads_the_life_it_finds_and_its_dispatch_is_written() {
    let id = WorkerId;
    for (rule, behind) in [(RULE, [id(3), id(2)]), (ORIGINAL, [id(2), id(3)])] {
        // A decision of 10 reads the key: no pin. One of 5,000 starts
        // the key over ahead of that decision's claim.
        let table = SessionTable::new(rule.sweep_micros);
        let seen = table.lookup(s(7), 10, &rule);
        assert_eq!(table.lookup(s(7), 5_000, &rule), [id(0); 2]);
        // The claim finds the pins as its decision read them.
        assert_eq!(table.claim(s(7), seen, id(1), 10), Claim::Made(id(1)));
        table.dispatched(s(7), SessionVerdict::Pinned, id(4), 9);
        assert_eq!(table.pins(s(7)), Some([id(1), id(0)]), "one before");
        // The decision's send lands on another worker.
        table.dispatched(s(7), SessionVerdict::Claimed(id(1)), id(2), 10);
        assert_eq!(table.pins(s(7)), Some([id(2), id(0)]), "its claim goes");

        // The decision of 5,000 claimed worker 2 first. The older
        // decision's claim call finds that claim and follows it.
        let table = SessionTable::new(rule.sweep_micros);
        let seen = table.lookup(s(7), 10, &rule);
        let newer = table.lookup(s(7), 5_000, &rule);
        assert_eq!(table.claim(s(7), newer, id(2), 5_000), Claim::Made(id(2)));
        let followed = Claim::Standing(id(2), [id(2), id(0)]);
        assert_eq!(table.claim(s(7), seen, id(1), 10), followed);
        // Its send there confirms the claim: the key was served on it.
        // The claiming decision's commit elsewhere then leaves that pin.
        table.dispatched(s(7), SessionVerdict::Pinned, id(2), 10);
        table.dispatched(s(7), SessionVerdict::Claimed(id(2)), id(3), 5_000);
        assert_eq!(table.pins(s(7)), Some(behind), "a claim it followed");

        // A claim that states no instant lowers nothing: a word of the
        // key's older life is still dropped behind it.
        let table = SessionTable::new(rule.sweep_micros);
        let seen = table.lookup(s(7), 10, &rule);
        assert_eq!(table.lookup(s(7), 5_000, &rule), [id(0); 2]);
        assert_eq!(table.claim(s(7), seen, id(1), 0), Claim::Made(id(1)));
        table.dispatched(s(7), SessionVerdict::Pinned, id(4), 10);
        assert_eq!(table.pins(s(7)), Some([id(1), id(0)]), "no instant");
    }
}

/// A decision that met its key's pin at the cap moves the pin where it
/// is decided, and the key can start over between its read and that
/// move. The move reads no pin: it would front its worker over pins its
/// decision never read. So a move dated before every reader of the key's
/// life writes nothing, and neither does the dispatch behind it.
///
/// A mover that read the life moves the pin, at the date the life began
/// at too. So does one that states no instant.
///
/// A gateway's sticky map moves a pin where it selects the worker, in
/// the entry as that selection reads it (`ManualPolicy::pin_front`,
/// `model_gateway/src/policies/manual.rs:269-300`).
#[test]
fn a_move_at_the_cap_writes_nothing_into_a_life_its_decision_did_not_read() {
    let id = WorkerId;
    for rule in [RULE, ORIGINAL] {
        // The key was served on worker 1. A decision of 20 reads that
        // pin at its cap. One of 5,000 starts the key over and claims
        // worker 5 ahead of the move.
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(7), 10, &rule);
        table.dispatched(s(7), SessionVerdict::Pinned, id(1), 10);
        assert_eq!(table.pins_renewed(s(7), 20, &rule), [id(1), id(0)]);
        let seen = table.lookup(s(7), 5_000, &rule);
        assert_eq!(table.claim(s(7), seen, id(5), 5_000), Claim::Made(id(5)));
        let stands = Some([id(5), id(0)]);
        table.respilled(s(7), id(2), 20);
        assert_eq!(table.pins(s(7)), stands, "no move");
        table.dispatched(s(7), SessionVerdict::Respill, id(3), 20);
        assert_eq!(table.pins(s(7)), stands, "no dispatch behind it");
        // A mover that read this life, and one that states no instant.
        table.respilled(s(7), id(2), 5_000);
        assert_eq!(table.pins(s(7)), Some([id(2), id(5)]), "its reader's");
        table.respilled(s(7), id(4), 0);
        assert_eq!(table.pins(s(7)), Some([id(4), id(2)]), "no instant");
    }
}

/// An entry a claim makes again, behind a sweep of its key, is one no
/// read began: it has no reader's date. So the dispatch of a request
/// that read the key's entry before is written there, though it is dated
/// before the claim.
///
/// Three requests read an empty key. The first claims worker 1, and the
/// key's end sweeps its entry. The second claims worker 1 on the entry
/// its claim makes. The first's dispatch confirms that claim: the key was
/// served there. The second's commit is refused then and takes back
/// nothing, and the third follows worker 1.
///
/// A gateway's sticky map answers so where it assigns the worker itself
/// and the third request may take the worker. A selection that comes
/// behind follows an entry made since its miss, by the first of the
/// entry's workers the request may take
/// (`ManualPolicy::select_by_routing_id` in
/// `model_gateway/src/policies/manual.rs`). A refused send takes no pin
/// back there.
#[test]
fn an_entry_a_claim_makes_again_takes_an_older_requests_dispatch() {
    let id = WorkerId;
    let table = SessionTable::new(RULE.sweep_micros);
    let first = table.lookup(s(7), 1_000, &RULE);
    let second = table.lookup(s(7), 1_010, &RULE);
    let third = table.lookup(s(7), 1_020, &RULE);
    assert_eq!(table.claim(s(7), first, id(1), 1_000), Claim::Made(id(1)));
    // Another key's read past this key's end sweeps the entry, and the
    // first claim with it.
    table.lookup(s(8), 5_000, &RULE);
    assert_eq!(table.pins(s(7)), None, "swept");
    assert_eq!(table.claim(s(7), second, id(1), 1_010), Claim::Made(id(1)));
    let unread = Some(([id(1), id(0)], SessionExpiry::Unread));
    assert_eq!(table.held(s(7)), unread, "no read began the entry");
    // The first request is sent to worker 1, under its own date. Every
    // target then refuses the second's commit.
    table.dispatched(s(7), SessionVerdict::Claimed(id(1)), id(1), 1_000);
    table.move_claim(s(7), id(1), id(0), 1_010);
    assert_eq!(table.pins(s(7)), Some([id(1), id(0)]), "nothing taken back");
    let followed = Claim::Standing(id(1), [id(1), id(0)]);
    assert_eq!(table.claim(s(7), third, id(2), 1_020), followed);
}

/// A move at the cap is written into an entry a claim made again, though
/// its decision read a pin the key holds no more. That entry has no
/// reader's date, so the move is no older life's there.
///
/// The key was served on worker 1, and a decision reads that pin at its
/// cap. The key's end sweeps it, two requests read it empty, and its end
/// sweeps it again. The first of the two claims worker 2 on the entry its
/// claim makes. The move onto worker 3 comes then: it fronts worker 3,
/// and the second request follows it.
///
/// A gateway's sticky map moves a pin where the capped request's
/// selection picks another worker under the cap, and tests no date of
/// that write (`ManualPolicy::pin_front`, called by `select_sticky` in
/// `model_gateway/src/policies/registry.rs`). A move that reaches the
/// map behind a new entry fronts its worker there.
#[test]
fn an_entry_a_claim_makes_again_takes_an_older_move_at_the_cap() {
    let id = WorkerId;
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(7), 1_000, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, id(1), 1_000);
    // The mover reads the pin at its cap.
    assert_eq!(table.pins_renewed(s(7), 1_001, &RULE), [id(1), id(0)]);
    table.lookup(s(8), 3_000, &RULE);
    assert_eq!(table.pins(s(7)), None, "the first sweep");
    let first = table.lookup(s(7), 3_100, &RULE);
    let second = table.lookup(s(7), 3_200, &RULE);
    table.lookup(s(8), 6_000, &RULE);
    assert_eq!(table.pins(s(7)), None, "the second sweep");
    assert_eq!(table.claim(s(7), first, id(2), 3_100), Claim::Made(id(2)));
    // The mover's write comes now, under its own date.
    table.respilled(s(7), id(3), 1_001);
    let followed = Claim::Standing(id(3), [id(3), id(2)]);
    assert_eq!(table.claim(s(7), second, id(9), 3_200), followed);
}

/// A decode call's date is written into the life the call read, and into
/// no other. Where the key started over behind the call's read, the new
/// life keeps its date, and the call's dispatch is an older life's
/// there. A call that read the new life dates it.
///
/// A gateway's sticky map pins a decode where that role's selection
/// picks it, in the entry that selection reads
/// (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn a_decode_calls_date_is_written_into_no_life_the_call_did_not_read() {
    let id = WorkerId;
    let table = SessionTable::new(RULE.sweep_micros);
    // The pair's lease is of 10. The call reads the key at 20, and a
    // decision of 5,000 starts the key over and claims worker 2.
    table.lookup(s(7), 10, &RULE);
    table.lookup(s(7), 20, &RULE);
    let seen = table.lookup(s(7), 5_000, &RULE);
    assert_eq!(table.claim(s(7), seen, id(2), 5_000), Claim::Made(id(2)));
    table.told_under(s(7), None, 20, 10, id(0));
    table.dispatched(s(7), SessionVerdict::Pinned, id(3), 10);
    assert_eq!(table.pins(s(7)), Some([id(2), id(0)]), "not its life");
    // A call of 5,000, the instant the new life began at, read it.
    table.told_under(s(7), None, 5_000, 10, id(0));
    table.dispatched(s(7), SessionVerdict::Pinned, id(3), 10);
    assert_eq!(table.pins(s(7)), Some([id(3), id(2)]), "its life");
}

/// A decode call that claims another worker than its pair's first decision
/// claimed takes the key with it at its dispatch: the lease's dispatch on
/// the call's worker withdraws the first claim, which no request was sent
/// on. Until then the key holds both. A first decision that claimed
/// nothing leaves its worker a pin, and so does another request sent
/// there ahead of the call's dispatch. The lease's dispatch on the first
/// claim's worker, behind a call taken back, withdraws the call's claim,
/// where another request sent there at the lease's instant withdraws
/// none, and the lease's refusal takes the call's claim back alone. A
/// decode sent on another target of the call's pick, and a pin moved
/// between the call's dating and the lease's dispatch, read as any claim
/// and pin do.
#[test]
fn a_decode_calls_dispatch_takes_the_keys_pin_with_it_from_its_leases_first_claim() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    // The pair's lease is of 10: its first decision claims 2, and its
    // decode call claims 3 at 40. The rule fronts the newer worker.
    let called = |first: WorkerId| {
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 10, &RULE);
        assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
        let seen = table.lookup(s(1), 40, &RULE);
        assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
        table.told_under(s(1), Some(id(3)), 40, 10, first);
        assert_eq!(table.pins(s(1)), Some([id(3), id(2)]), "behind the call");
        table
    };
    // The call's decode is dispatched: the key holds it alone.
    let table = called(id(2));
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "the call's worker");
    // The first decision followed a pin: its worker stays.
    let table = called(id(0));
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(2)]), "no first claim");
    // Another request is sent to the first worker ahead of the call's
    // dispatch: the key was served there.
    let table = called(id(2));
    table.dispatched(s(1), SessionVerdict::Pinned, id(2), 20);
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(
        table.pins(s(1)),
        Some([id(3), id(2)]),
        "served on the first"
    );
    // The call is taken back and the lease's first decode is sent: the
    // call's claim leaves.
    let table = called(id(2));
    table.dispatched(s(1), claimed(id(2)), id(2), 10);
    assert_eq!(table.pins(s(1)), Some([id(2), id(0)]), "the first worker");
    // Another request of the lease's instant is sent to the first worker:
    // it claimed nothing there, and the call's claim stands.
    let table = called(id(2));
    table.dispatched(s(1), SessionVerdict::Pinned, id(2), 10);
    assert_eq!(
        table.pins(s(1)),
        Some([id(3), id(2)]),
        "another request of the instant"
    );
    // The lease's decode is sent on another target of the call's pick: the
    // call's claim leaves, as a decision's does that goes elsewhere, and
    // the first claim stays a pin.
    let table = called(id(2));
    table.dispatched(s(1), claimed(id(3)), id(4), 10);
    assert_eq!(table.pins(s(1)), Some([id(4), id(2)]), "another target");
    // A move fronts a third worker between the call's dating and the
    // lease's dispatch: the tie ends there, and the dispatch on the call's
    // worker withdraws no pin.
    let table = called(id(2));
    table.respilled(s(1), id(4), 0);
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(4), id(3)]), "behind a move");
    // The lease's refusal takes the call's claim back alone.
    let table = called(id(2));
    assert_eq!(
        table.move_claim(s(1), id(3), id(0), 10),
        SessionVerdict::Pinned
    );
    assert_eq!(table.pins(s(1)), Some([id(2), id(0)]), "the refusal");
    // Another request is sent to the first worker between the call's
    // claim and its dating: the key was served there, and nothing is tied.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let seen = table.lookup(s(1), 40, &RULE);
    assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
    table.dispatched(s(1), SessionVerdict::Pinned, id(2), 20);
    table.told_under(s(1), Some(id(3)), 40, 10, id(2));
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(
        table.pins(s(1)),
        Some([id(3), id(2)]),
        "served ahead of the dating"
    );
}

/// A move at the cap onto the worker that leads a key's pins writes no pin,
/// and it ends the tie of a decode call's claim and its lease's first claim
/// all the same, under either recovery: the lease's dispatch behind it takes
/// back neither pin, on the call's worker or on the first one. So does a
/// move that comes between the call's claim and its dating.
#[test]
fn a_move_onto_the_leading_worker_keeps_both_pins_behind_the_leases_dispatch() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    // The pair's lease is of 10: its first decision claims 2, and its
    // decode call claims 3 at 40. The pins behind the call's claim, by
    // the rule: the newer worker fronted, or appended.
    for (rule, both) in [(&RULE, [id(3), id(2)]), (&ORIGINAL, [id(2), id(3)])] {
        let claims = || {
            let table = SessionTable::new(rule.sweep_micros);
            let seen = table.lookup(s(1), 10, rule);
            assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
            let seen = table.lookup(s(1), 40, rule);
            assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
            table
        };
        let called = || {
            let table = claims();
            table.told_under(s(1), Some(id(3)), 40, 10, id(2));
            assert_eq!(table.pins(s(1)), Some(both), "behind the call");
            table
        };
        // A request decided at 20 moves onto the worker that leads, and
        // the call's decode is dispatched.
        let table = called();
        table.respilled(s(1), both[0], 20);
        table.dispatched(s(1), claimed(id(3)), id(3), 10);
        assert_eq!(table.pins(s(1)), Some(both), "the call's worker");
        // The call is taken back and the lease's first decode is sent.
        let table = called();
        table.respilled(s(1), both[0], 20);
        table.dispatched(s(1), claimed(id(2)), id(2), 10);
        assert_eq!(table.pins(s(1)), Some(both), "the first worker");
        // The move comes between the call's claim and its dating.
        let table = claims();
        table.respilled(s(1), both[0], 20);
        table.told_under(s(1), Some(id(3)), 40, 10, id(2));
        table.dispatched(s(1), claimed(id(3)), id(3), 10);
        assert_eq!(table.pins(s(1)), Some(both), "ahead of the dating");
    }
}

/// A request that followed the pin of a lease's first claim and is sent
/// there behind that lease's dispatch on its decode call's worker puts the
/// pin back where it stood, under either recovery: the key was served on
/// both workers. A dispatch on a third worker between the two moves the
/// pins by the rule, and the request sent on the first worker then joins
/// them as any worker does. A respill on the call's worker between the two
/// fronts it over the pin, at its decision as at its dispatch. The pin
/// follows its worker's rename, and a forgotten worker's is not put back.
#[test]
fn a_decode_calls_dispatch_leaves_the_pin_of_a_request_that_followed_the_first_claim() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    let (pinned, respill) = (SessionVerdict::Pinned, SessionVerdict::Respill);
    // The pair's lease is of 10: its first decision claims 2, its decode
    // call claims 3 at 40, and the lease's decode is sent on 3.
    let sent = |rule: &SessionRule| {
        let table = SessionTable::new(rule.sweep_micros);
        let seen = table.lookup(s(1), 10, rule);
        assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
        let seen = table.lookup(s(1), 40, rule);
        assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
        table.told_under(s(1), Some(id(3)), 40, 10, id(2));
        table.dispatched(s(1), claimed(id(3)), id(3), 10);
        assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "the call's worker");
        table
    };
    // The pins behind the call's claim, by the rule: the newer worker
    // fronted, or appended.
    for (rule, both) in [(&RULE, [id(3), id(2)]), (&ORIGINAL, [id(2), id(3)])] {
        // The request that followed 2 at 20 is sent there.
        let table = sent(rule);
        table.dispatched(s(1), pinned, id(2), 20);
        assert_eq!(table.pins(s(1)), Some(both), "where it stood");
        // A request is sent on a third worker between the two.
        let table = sent(rule);
        table.dispatched(s(1), pinned, id(4), 30);
        table.dispatched(s(1), pinned, id(2), 20);
        let joined = match rule.recovery {
            Recovery::Latest => [id(2), id(4)],
            Recovery::Original => [id(4), id(2)],
        };
        assert_eq!(table.pins(s(1)), Some(joined), "behind a third worker");
        // A respill lands on the call's worker between the two.
        let table = sent(rule);
        table.dispatched(s(1), respill, id(3), 30);
        table.dispatched(s(1), pinned, id(2), 20);
        assert_eq!(
            table.pins(s(1)),
            Some([id(3), id(2)]),
            "behind a respill on the call's worker"
        );
        // A decision that met the first claim's pin at its cap moves the
        // key onto the call's worker behind the lease's dispatch, and its
        // request is never sent.
        let table = sent(rule);
        table.respilled(s(1), id(3), 0);
        table.dispatched(s(1), pinned, id(2), 20);
        assert_eq!(
            table.pins(s(1)),
            Some([id(3), id(2)]),
            "behind a move onto the call's worker"
        );
        // The first worker is registered again under 5 behind the lease's
        // dispatch: the request names the id it read.
        let table = sent(rule);
        table.rename_worker(id(2), id(5));
        table.dispatched(s(1), pinned, id(2), 20);
        let renamed = both.map(|pin| if pin == id(2) { id(5) } else { pin });
        assert_eq!(table.pins(s(1)), Some(renamed), "under the id that stands");
        // The first worker is forgotten: its request joins as any does.
        let table = sent(rule);
        table.forget_workers(&[id(2)]);
        table.dispatched(s(1), pinned, id(2), 20);
        let forgotten = match rule.recovery {
            Recovery::Latest => [id(2), id(3)],
            Recovery::Original => [id(3), id(2)],
        };
        assert_eq!(table.pins(s(1)), Some(forgotten), "behind a forget");
    }
}

/// The tie of a decode call's claim to its lease's first claim under a
/// lease that states no instant, across a rename and behind a forgotten
/// worker. A lease with no instant is known by its worker alone: its
/// dispatch on the call's worker withdraws the first claim, and its
/// dispatch on the first claim's worker withdraws nothing. A rename
/// between the call's dating and the lease's dispatch keeps the tie under
/// the id that stands. A first worker the table forgot leaves the call's
/// worker alone, and the lease's dispatch there withdraws nothing more.
#[test]
fn a_decode_calls_tie_to_its_leases_first_claim_reads_no_instant_a_rename_and_a_forget() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    // The pair's lease states no instant: its first decision claims 2, and
    // its decode call claims 3 at 40.
    let undated = || {
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 10, &RULE);
        assert_eq!(table.claim(s(1), seen, id(2), 0), Claim::Made(id(2)));
        let seen = table.lookup(s(1), 40, &RULE);
        assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
        table.told_under(s(1), Some(id(3)), 40, 0, id(2));
        assert_eq!(table.pins(s(1)), Some([id(3), id(2)]), "behind the call");
        table
    };
    let table = undated();
    table.dispatched(s(1), claimed(id(3)), id(3), 0);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "by the id alone");
    let table = undated();
    table.dispatched(s(1), claimed(id(2)), id(2), 0);
    assert_eq!(
        table.pins(s(1)),
        Some([id(3), id(2)]),
        "no instant, no take-back"
    );
    // The pair's lease is of 10: its first decision claims 2, and its
    // decode call claims 3 at 40.
    let called = || {
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 10, &RULE);
        assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
        let seen = table.lookup(s(1), 40, &RULE);
        assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
        table.told_under(s(1), Some(id(3)), 40, 10, id(2));
        table
    };
    // The first worker is registered again under 5 ahead of the dispatch.
    let table = called();
    table.rename_worker(id(2), id(5));
    assert_eq!(table.pins(s(1)), Some([id(3), id(5)]), "renamed");
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "the tie follows");
    // The first worker is forgotten ahead of the dispatch.
    let table = called();
    table.forget_workers(&[id(2)]);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "forgotten");
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]), "nothing more");
}

/// The id a tied dispatch asks by, in one order of calls. A pair's lease
/// of 10 claims worker 2 by its first decision and worker 3 by its decode
/// call. Worker 2 is registered again as 5, the call is taken back, and
/// the lease's first decode is dispatched on the id 2. The tie is asked
/// by that id, which the lease's own word names, and the pins are read by
/// the id that stands. So the call's claim is withdrawn with its mark,
/// and the key holds worker 5 alone.
#[test]
fn a_tied_dispatch_on_the_id_a_worker_had_withdraws_the_calls_claim() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    // The pair's lease is of 10: its first decision claims 2, and its
    // decode call claims 3 at 40.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let seen = table.lookup(s(1), 40, &RULE);
    assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
    table.told_under(s(1), Some(id(3)), 40, 10, id(2));
    // The first worker is registered again under 5. The call is taken
    // back, and the lease's first decode is sent on the id 2.
    table.rename_worker(id(2), id(5));
    assert_eq!(table.pins(s(1)), Some([id(3), id(5)]), "renamed");
    table.dispatched(s(1), claimed(id(2)), id(2), 10);
    assert_eq!(table.pins(s(1)), Some([id(5), id(0)]), "the first worker");
    // The call's claim left with its mark: a word of the lease that
    // would move it finds no claim, and writes nothing.
    let told = table.move_claim(s(1), id(3), id(7), 10);
    assert_eq!(told, SessionVerdict::Pinned, "no claim to move");
    assert_eq!(table.pins(s(1)), Some([id(5), id(0)]), "nothing moved");
}

/// A decode call's dispatch takes the key's pin with it from the first
/// claim of its own lease alone. A key that idled out behind that first
/// claim starts over, and a later decision's claim on the same worker is
/// no claim of the lease: the call's claim stands behind it, and the
/// lease's dispatch on the call's worker leaves both. A shard states the
/// claim its newest claim over a standing one replaced, so a second key
/// of the shard that writes such a claim between the call's claim and its
/// dating leaves the call's two claims as any pin and claim: both stand.
#[test]
fn a_decode_calls_dispatch_leaves_a_claim_that_is_not_its_leases_first() {
    let (id, claimed) = (WorkerId, SessionVerdict::Claimed);
    // The lease is of 10 and its first decision claims 2. The key idles
    // out, and a decision of 5000 claims 2 anew ahead of the call's.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let seen = table.lookup(s(1), 5_000, &RULE);
    assert_eq!(seen, [id(0); 2], "the key starts over");
    assert_eq!(table.claim(s(1), seen, id(2), 5_000), Claim::Made(id(2)));
    let seen = table.lookup(s(1), 5_040, &RULE);
    assert_eq!(table.claim(s(1), seen, id(3), 5_040), Claim::Made(id(3)));
    table.told_under(s(1), Some(id(3)), 5_040, 10, id(2));
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(2)]), "a later claim");
    // The call's claim replaces its lease's first, and a second key of
    // the shard (the two keys share their upper half) replaces a claim of
    // its own ahead of the call's dating.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2), 10), Claim::Made(id(2)));
    let seen = table.lookup(s(2), 20, &RULE);
    assert_eq!(table.claim(s(2), seen, id(5), 20), Claim::Made(id(5)));
    let seen = table.lookup(s(1), 40, &RULE);
    assert_eq!(table.claim(s(1), seen, id(3), 40), Claim::Made(id(3)));
    let seen = table.lookup(s(2), 41, &RULE);
    assert_eq!(table.claim(s(2), seen, id(6), 41), Claim::Made(id(6)));
    table.told_under(s(1), Some(id(3)), 40, 10, id(2));
    table.dispatched(s(1), claimed(id(3)), id(3), 10);
    assert_eq!(table.pins(s(1)), Some([id(3), id(2)]), "the word taken");
}

/// Pins that moved with no rename said are read as written, and
/// their first newcomer is the pin to follow, where the decision's
/// own pick is the second: two other requests committed, the later
/// one on the worker the decision picked. A worker that is a pin
/// already answers the decision's own pick only where the pins stand
/// as the decision read them.
#[test]
fn a_claim_on_pins_that_moved_follows_their_first_newcomer_past_its_own_pick() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        let read = [WorkerId(5), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), read);
        // The two commits, in the order that leaves worker 9 in the
        // lead and the decision's pick behind it.
        let commits = match rule.recovery {
            Recovery::Latest => [7, 9],
            Recovery::Original => [9, 7],
        };
        for worker in commits {
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(worker), 0);
        }
        let held = [WorkerId(9), WorkerId(7)];
        assert_eq!(table.pins(s(1)), Some(held));
        let claim = table.claim(s(1), read, WorkerId(7), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(9), held));
        assert_eq!(table.pins(s(1)), Some(held), "nothing is written");
    }
}

/// A decision that claims a worker it read as a pin of the key, on
/// pins that stand as it read them, makes no claim: the table writes
/// nothing, marks nothing and answers the decision's own pick, as for
/// a worker pinned under another id. The decision's commit on that
/// pick then leaves the pins as read, as a failover taken does, and
/// its commit elsewhere moves them as a pin's alternate does. The
/// keyed decision answers a pin it read ahead of its claim, and this
/// is the table's answer to a caller that does not. The gateway's
/// sticky map writes nothing where a key's request lands on a
/// candidate it holds
/// (`model_gateway/src/policies/registry.rs:794-797`).
#[test]
fn a_claim_of_a_pin_its_decision_read_is_no_claim() {
    for rule in [RULE, ORIGINAL] {
        // The key's table up to the claim, made once for each commit.
        let claimed = || {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
            let read = table.lookup(s(1), 20, &rule);
            // The decision picks the pin it read behind the lead.
            let pick = read[1];
            let claim = table.claim(s(1), read, pick, 0);
            assert_eq!(claim, Claim::Standing(pick, read));
            assert_eq!(table.pins(s(1)), Some(read), "nothing is written");
            (table, read, pick)
        };
        // The decision's commit lands on its pick: the failover taken.
        let (table, read, pick) = claimed();
        table.dispatched(s(1), SessionVerdict::Pinned, pick, 0);
        assert_eq!(table.pins(s(1)), Some(read), "the pins stay as read");
        // The decision's commit lands elsewhere, under a pin's verdict.
        let (table, read, pick) = claimed();
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(9), 0);
        let moved = match rule.recovery {
            Recovery::Latest => [WorkerId(9), read[0]],
            Recovery::Original => [pick, WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(moved), "as a pin's alternate");
    }
}

/// A key that holds one pin keeps it where the decision that read it
/// claims it and is sent elsewhere. The table marks no claim, so the
/// commit withdraws nothing, even one that names a claim: the worker
/// the key was served on stands behind the newcomer by the key's
/// recovery. A marked pin would go with that commit and the key would
/// forget its worker. The gateway's sticky map pins a new worker first
/// and keeps the one before it as its second candidate
/// (`model_gateway/src/policies/manual.rs:269-274`), and under its own
/// assignments the new worker joins behind (`manual.rs:125-130`).
#[test]
fn a_claim_of_a_keys_one_pin_leaves_it_behind_a_commit_elsewhere() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        let read = table.lookup(s(1), 20, &rule);
        assert_eq!(read, [WorkerId(5), WorkerId(0)]);
        let claim = table.claim(s(1), read, WorkerId(5), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(5), read));
        assert_eq!(table.pins(s(1)), Some(read), "nothing is written");
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(5)), WorkerId(9), 0);
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(5)],
            Recovery::Original => [WorkerId(5), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the served pin stays");
    }
}
