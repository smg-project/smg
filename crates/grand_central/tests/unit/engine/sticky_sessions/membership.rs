//! Workers that leave, join, are renamed or are forgotten: what stands of a
//! key's pins and of its claim, the step a rename takes for one key, and the
//! absences a table holds.

use super::*;

#[test]
fn a_forgotten_worker_leaves_every_pin_and_a_key_with_no_pin_left_leaves_the_table() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, first, second) in [(1, 1, 2), (2, 2, 0), (3, 2, 1), (4, 3, 0)] {
        table.lookup(s(key), 10, &ORIGINAL);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(first), 0);
        if second != 0 {
            table.lookup(s(key), 20, &ORIGINAL);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(second), 0);
        }
    }
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(1)]));
    table.forget_workers(&[WorkerId(2)]);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(0)]),
        "the failover slot went"
    );
    assert_eq!(table.pins(s(2)), None, "its only pin went");
    assert_eq!(
        table.pins(s(3)),
        Some([WorkerId(1), WorkerId(0)]),
        "the failover is the owner"
    );
    assert_eq!(
        table.pins(s(4)),
        Some([WorkerId(3), WorkerId(0)]),
        "untouched"
    );
    assert_eq!(table.len(), 3);
    table.forget_workers(&[WorkerId(0)]);
    assert_eq!(table.len(), 3, "the empty id is never a pin");
}

/// A walk that forgets a worker takes a key with no pin left out of
/// the table, and leaves the key a decision read and has not claimed.
#[test]
fn a_forgotten_workers_walk_keeps_a_key_read_and_not_yet_claimed() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(4), 0);
    table.forget_workers(&[WorkerId(0)]);
    table.forget_workers(&[WorkerId(4)]);
    assert_eq!(table.pins(s(1)), Some([WorkerId(0); 2]));
    assert_eq!(table.pins(s(2)), None);
}

/// A claim follows its worker to its new id and keeps the id its lease
/// names. The claiming decision's own dispatch, which names the old
/// id, confirms the renamed claim. Its commit elsewhere withdraws it,
/// and the worker that served the key is pinned alone.
#[test]
fn a_claim_renamed_with_its_worker_is_confirmed_or_withdrawn_by_its_lease() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    for key in [7, 8] {
        assert_eq!(table.lookup(s(key), 10, &ORIGINAL), [WorkerId(0); 2]);
        let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(1), 0);
        assert_eq!(claim, Claim::Made(WorkerId(1)));
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    let renamed = Some([WorkerId(11), WorkerId(0)]);
    assert_eq!((table.pins(s(7)), table.pins(s(8))), (renamed, renamed));
    // Its own dispatch names the id the worker had: confirmed.
    table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1), 0);
    assert_eq!(table.pins(s(7)), renamed);
    // A pin a dispatch confirmed is withdrawn by no later commit.
    table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(11), WorkerId(2)]));
    // The other key's commit lands elsewhere: withdrawn.
    table.dispatched(s(8), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    assert_eq!(table.pins(s(8)), Some([WorkerId(2), WorkerId(0)]));
}

/// A dispatch that lands behind its worker's rename names the id the
/// worker had. The pins move under the id that stands, through a
/// second rename too, and no pin names an id no row holds.
#[test]
fn a_dispatch_on_the_id_a_worker_had_moves_the_pins_under_the_id_that_stands() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    // The key's own late dispatch: its pin stands as renamed.
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
    // Another key's respill landed there: the standing id leads.
    table.dispatched(s(2), SessionVerdict::Respill, WorkerId(1), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(11), WorkerId(2)]));
    table.rename_worker(WorkerId(11), WorkerId(21));
    table.dispatched(s(1), SessionVerdict::Respill, WorkerId(1), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(21), WorkerId(0)]));
    // Back under the id it had first: that id stands again.
    table.rename_worker(WorkerId(21), WorkerId(1));
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(1), WorkerId(2)]));
    assert_eq!(table.renamed(), 2, "the two ids it left behind");
}

/// A decision reads its rows ahead of its move, and a worker may be
/// registered again between the two. The move fronts the id that
/// stands where the rename is over and where its walk has not reached
/// the key. A pin the walk has yet to rename leads under the id it
/// had, and the walk renames it there, as it renames a move made
/// ahead of the whole rename. The dispatch on the id the worker had
/// finds the pins in place, and no pin names an id no row holds.
#[test]
fn a_respills_move_on_the_id_a_worker_had_fronts_the_id_that_stands() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in [1, 2, 4] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2), 0);
    }
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4), 0);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(4)]));
    // The whole rename is ahead of the move.
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.respilled(s(1), WorkerId(1), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
    // The rename's first half: remembered, the shards not yet walked.
    table.remember(WorkerId(3), WorkerId(13));
    table.respilled(s(2), WorkerId(3), 0);
    assert_eq!(table.pins(s(2)), Some([WorkerId(13), WorkerId(2)]));
    // The same half where the worker is the key's failover: its pin
    // leads as it stands, and the walk renames it.
    table.remember(WorkerId(4), WorkerId(14));
    table.respilled(s(3), WorkerId(4), 0);
    assert_eq!(table.pins(s(3)), Some([WorkerId(4), WorkerId(2)]));
    // The move is ahead of the whole rename.
    table.respilled(s(4), WorkerId(5), 0);
    assert_eq!(table.pins(s(4)), Some([WorkerId(5), WorkerId(2)]));
    for (had, stands) in [(3, 13), (4, 14), (5, 15)] {
        table.rename_worker(WorkerId(had), WorkerId(stands));
    }
    for (key, had, stands) in [(1, 1, 11), (2, 3, 13), (3, 4, 14), (4, 5, 15)] {
        let led = Some([WorkerId(stands), WorkerId(2)]);
        assert_eq!(table.pins(s(key)), led, "key {key} behind the renames");
        table.dispatched(s(key), SessionVerdict::Respill, WorkerId(had), 0);
        assert_eq!(table.pins(s(key)), led, "key {key} behind its dispatch");
    }
}

/// A decision reads its rows ahead of its claim, and its worker may be
/// registered again between the two. The pin and the claim of record
/// take the id that stands, behind the whole rename and behind its
/// first half, and the answer keeps the id the decision read. The
/// lease names that id: the claim's own dispatch on it confirms the
/// claim, a later commit elsewhere withdraws no confirmed pin, and the
/// claiming decision's commit elsewhere withdraws its claim.
#[test]
fn a_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Worker 1 is registered again whole, worker 3 by the rename's
        // first half: remembered, the shards not yet walked.
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.remember(WorkerId(3), WorkerId(13));
        let claims = [(1, 1, 11), (2, 1, 11), (3, 3, 13), (4, 3, 13)];
        for (key, had, stands) in claims {
            assert_eq!(table.lookup(s(key), 10, &rule), [WorkerId(0); 2]);
            let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(had), 0);
            assert_eq!(
                claim,
                Claim::Made(WorkerId(had)),
                "the id the decision read"
            );
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(
                table.pins(s(key)),
                led,
                "key {key} claims the id that stands"
            );
        }
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, had, stands) in claims {
            let claimed = SessionVerdict::Claimed(WorkerId(had));
            if key % 2 == 1 {
                // Its own dispatch names the id the decision read.
                table.dispatched(s(key), claimed, WorkerId(had), 0);
                let led = Some([WorkerId(stands), WorkerId(0)]);
                assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
                // A pin a dispatch confirmed is withdrawn by no commit.
                table.dispatched(s(key), claimed, WorkerId(2), 0);
                let both = match rule.recovery {
                    Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                    Recovery::Original => [WorkerId(stands), WorkerId(2)],
                };
                assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
            } else {
                // The claiming decision's commit lands elsewhere.
                table.dispatched(s(key), claimed, WorkerId(2), 0);
                let alone = Some([WorkerId(2), WorkerId(0)]);
                assert_eq!(table.pins(s(key)), alone, "key {key} is withdrawn");
            }
        }
    }
}

/// A key pinned on a worker is claimed by a decision that read the
/// worker's row under the id it had: the key's pin, which names the id
/// that stands, read as not routable there, and the decision's pick is
/// the same worker. The table writes nothing and answers the
/// decision's own pick. A pin the key was served on is marked no
/// claim, so the request's commit on an alternate withdraws nothing.
#[test]
fn a_claim_on_a_worker_pinned_under_the_id_that_stands_writes_nothing() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        table.rename_worker(WorkerId(1), WorkerId(11));
        let pins = [WorkerId(11), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), pins);
        let claim = table.claim(s(1), pins, WorkerId(1), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(1), pins));
        assert_eq!(table.pins(s(1)), Some(pins), "nothing is written");
        // The request's dispatch on its pick is one on the pin.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        assert_eq!(table.pins(s(1)), Some(pins));
        // Its commit on an alternate moves the pins as a pin's does.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(11)],
            Recovery::Original => [WorkerId(11), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both));
    }
}

/// A pair's decision moves its claim to the prefill its pair formed
/// on, which it read ahead of that write: a prefill registered again
/// between the two is claimed under the id that stands, behind the
/// whole rename and behind its first half, and the answer keeps the
/// id the decision read. Its dispatch on that id confirms the claim,
/// and its commit elsewhere withdraws it. A prefill that is a pin of
/// the key under the id that stands is no claim.
#[test]
fn a_moved_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        for key in [1, 2, 3] {
            table.lookup(s(key), 10, &rule);
            let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(5), 0);
            assert_eq!(claim, Claim::Made(WorkerId(5)));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.remember(WorkerId(3), WorkerId(13));
        for (key, had, stands) in [(1, 1, 11), (2, 1, 11), (3, 3, 13)] {
            let verdict = table.move_claim(s(key), WorkerId(5), WorkerId(had), 0);
            let claimed = SessionVerdict::Claimed(WorkerId(had));
            assert_eq!(verdict, claimed, "the id the decision read");
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(
                table.pins(s(key)),
                led,
                "key {key} claims the id that stands"
            );
        }
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, had, stands) in [(1, 1, 11), (3, 3, 13)] {
            table.dispatched(
                s(key),
                SessionVerdict::Claimed(WorkerId(had)),
                WorkerId(had),
                0,
            );
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
        }
        table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
        assert_eq!(
            table.pins(s(2)),
            Some([WorkerId(2), WorkerId(0)]),
            "withdrawn"
        );
        // The key was served on the prefill under the id that stands.
        table.lookup(s(4), 10, &rule);
        table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(11), 0);
        let pins = [WorkerId(11), WorkerId(0)];
        assert_eq!(
            table.claim(s(4), pins, WorkerId(5), 0),
            Claim::Made(WorkerId(5))
        );
        let verdict = table.move_claim(s(4), WorkerId(5), WorkerId(1), 0);
        assert_eq!(verdict, SessionVerdict::Pinned);
        assert_eq!(
            table.pins(s(4)),
            Some(pins),
            "the pin stands alone, and no claim"
        );
        table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(1), 0);
        assert_eq!(table.pins(s(4)), Some(pins));
    }
}

/// The race the table leaves as it stands: the key's own pin is
/// renamed between a decision's read of the key and its claim. The
/// pins are not the ones the decision read, and the renamed id answers
/// as their newcomer. No row of that decision holds it, so the request
/// keeps its own pick, and its commit moves the pins by the key's
/// recovery beside the renamed pin.
#[test]
fn a_claim_behind_the_rename_of_the_keys_own_pin_answers_the_renamed_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        // The decision read the pin, found no row for it and chose 2.
        let seen = table.lookup(s(1), 20, &rule);
        assert_eq!(seen, [WorkerId(5), WorkerId(0)]);
        table.rename_worker(WorkerId(5), WorkerId(15));
        let renamed = [WorkerId(15), WorkerId(0)];
        let claim = table.claim(s(1), seen, WorkerId(2), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(15), renamed));
        assert_eq!(table.pins(s(1)), Some(renamed), "nothing is claimed");
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
        let moved = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(15)],
            Recovery::Original => [WorkerId(15), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(moved));
    }
}

/// A rename says the id that stands ahead of its walk of the shards,
/// so a pin the walk has yet to reach holds an id its worker had. A
/// claim of that worker takes the walk's step for the key first: the
/// pin is read by the id that stands, the claim finds the worker
/// pinned and answers the decision's own pick, and no pin is written
/// beside it. In both orders: a worker registered twice and claimed
/// under its first id while the second rename is half walked, and a
/// worker registered once and claimed under its new id while that
/// rename is half walked. Written beside the pin, the claim would be
/// made one with it by the walk, and the claiming decision's commit
/// on an alternate would withdraw a pin the key was served on.
#[test]
fn a_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Key 1 is served on a worker registered once since, key 2 on
        // a worker under its first id.
        for (key, worker) in [(1, 1), (2, 3)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        // Each worker's next rename by its first half: remembered,
        // the shards not yet walked.
        table.remember(WorkerId(11), WorkerId(21));
        table.remember(WorkerId(3), WorkerId(13));
        // Key 1's decision read the worker's first id, key 2's the
        // rows that stand: (key, the pin as held, the id read, the id
        // that stands).
        let orders = [(1, 11, 1, 21), (2, 3, 13, 13)];
        for (key, held, read, stands) in orders {
            let pins = [WorkerId(held), WorkerId(0)];
            assert_eq!(table.lookup(s(key), 20, &rule), pins);
            let claim = table.claim(s(key), pins, WorkerId(read), 0);
            let led = [WorkerId(stands), WorkerId(0)];
            assert_eq!(claim, Claim::Standing(WorkerId(read), led), "key {key}");
            assert_eq!(table.pins(s(key)), Some(led), "key {key}: the step alone");
        }
        table.rename_worker(WorkerId(11), WorkerId(21));
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, _, read, stands) in orders {
            let led = [WorkerId(stands), WorkerId(0)];
            assert_eq!(table.pins(s(key)), Some(led), "key {key} behind the walk");
            // The decision's dispatch on its pick is one on the pin.
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(read), 0);
            assert_eq!(table.pins(s(key)), Some(led), "key {key} on its pick");
            // Its commit on an alternate withdraws nothing.
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2), 0);
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                Recovery::Original => [WorkerId(stands), WorkerId(2)],
            };
            assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
        }
    }
}

/// A pair's moved claim beside such a pin takes the walk's step too:
/// the prefill its pair formed on is the key's pin, under an id the
/// second rename's walk has yet to reach, and the decision read the
/// first. The pin takes the id that stands, and nothing is claimed.
#[test]
fn a_moved_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        table.rename_worker(WorkerId(1), WorkerId(11));
        let pins = [WorkerId(11), WorkerId(0)];
        // The decision claims another prefill, then pairs on the
        // worker under its first id, the second rename half walked.
        let claim = table.claim(s(1), pins, WorkerId(5), 0);
        assert_eq!(claim, Claim::Made(WorkerId(5)));
        table.remember(WorkerId(11), WorkerId(21));
        let verdict = table.move_claim(s(1), WorkerId(5), WorkerId(1), 0);
        assert_eq!(verdict, SessionVerdict::Pinned);
        let led = [WorkerId(21), WorkerId(0)];
        assert_eq!(table.pins(s(1)), Some(led), "the pin alone, no claim");
        table.rename_worker(WorkerId(11), WorkerId(21));
        // The pair's commit on an alternate withdraws nothing.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(21)],
            Recovery::Original => [WorkerId(21), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both));
    }
}

/// A refusal on a key with a slot free takes no rename's step. It names
/// no worker, and the empty id reads as the pin of a free slot, so the
/// table asks no rename for it. The key's pin stands as written until
/// the rename's walk reaches it.
#[test]
fn a_refusal_on_a_key_with_a_slot_free_takes_no_renames_step() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let table = SessionTable::new(RULE.sweep_micros);
    // Every target refused the commit of the decision that claimed 9.
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(9), 10), Claim::Made(id(9)));
    assert_eq!(table.move_claim(s(1), id(9), id(0), 10), pinned);
    table.lookup(s(1), 20, &RULE);
    table.dispatched(s(1), pinned, id(1), 20);
    // The rename's first half: remembered, the shard not yet walked.
    table.remember(id(1), id(11));
    // The refusal told again.
    assert_eq!(table.move_claim(s(1), id(9), id(0), 10), pinned);
    assert_eq!(table.pins(s(1)), Some([id(1), id(0)]), "as written");
    table.rename_worker(id(1), id(11));
    assert_eq!(table.pins(s(1)), Some([id(11), id(0)]));
}

/// A commit and a respill's move on an id a worker had meet the
/// worker's own pin where a rename's walk has yet to reach it: the
/// pin takes the id that stands, the worker is pinned once, and the
/// key's other pin stays.
#[test]
fn a_commit_and_a_move_beside_a_pin_the_walk_has_yet_to_reach_pin_the_worker_once() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in [1, 2] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1), 0);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(4), 0);
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.remember(WorkerId(11), WorkerId(21));
    assert_eq!(table.pins(s(1)), Some([WorkerId(4), WorkerId(11)]));
    // A commit on the worker's first id is the failover taken.
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let taken = Some([WorkerId(4), WorkerId(21)]);
    assert_eq!(table.pins(s(1)), taken, "the pins keep their order");
    // A respill's move there fronts the worker's own pin.
    table.respilled(s(2), WorkerId(1), 0);
    let fronted = Some([WorkerId(21), WorkerId(4)]);
    assert_eq!(table.pins(s(2)), fronted);
    table.rename_worker(WorkerId(11), WorkerId(21));
    assert_eq!(table.pins(s(1)), taken);
    assert_eq!(table.pins(s(2)), fronted);
    // A commit on a worker's new id inside its first rename's walk:
    // the key's leading pin is the worker's, and the failover stays.
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4), 0);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(5), 0);
    table.remember(WorkerId(5), WorkerId(15));
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(15), 0);
    let pins = Some([WorkerId(15), WorkerId(4)]);
    assert_eq!(table.pins(s(3)), pins, "the failover stays");
    table.rename_worker(WorkerId(5), WorkerId(15));
    assert_eq!(table.pins(s(3)), pins);
}

/// A key that holds one worker under two ids, half walked: its pin
/// under the id the worker had, and a claim written under the new id
/// ahead of the table's word of the rename. Inside that rename's walk
/// a dispatch on an id the worker had is one on the claimed worker,
/// and a respill fronts another worker. The first write that asks
/// takes the walk's step for the key: the two pins are one, and the
/// claim's mark is gone. The claiming decision's commit on the
/// respill's worker then withdraws nothing, with the dispatch between
/// the two and without it, as behind the whole walk.
#[test]
fn a_key_that_holds_one_worker_twice_is_walked_by_its_first_write_that_asks() {
    for rule in [RULE, ORIGINAL] {
        for dispatch in [true, false] {
            let said = (rule.recovery, dispatch);
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
            table.rename_worker(WorkerId(1), WorkerId(11));
            let served = [WorkerId(11), WorkerId(0)];
            // The decision read the row that replaced the worker's,
            // ahead of the table's word of that rename.
            let claim = table.claim(s(1), served, WorkerId(21), 0);
            assert_eq!(claim, Claim::Made(WorkerId(21)), "{said:?}");
            // The rename's first half: remembered, the shards not
            // yet walked.
            table.remember(WorkerId(11), WorkerId(21));
            if dispatch {
                table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
                let one = [WorkerId(21), WorkerId(0)];
                assert_eq!(table.pins(s(1)), Some(one), "{said:?}: one pin");
            }
            table.respilled(s(1), WorkerId(2), 0);
            let moved = [WorkerId(2), WorkerId(21)];
            assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the move");
            table.rename_worker(WorkerId(11), WorkerId(21));
            let claimed = SessionVerdict::Claimed(WorkerId(21));
            table.dispatched(s(1), claimed, WorkerId(2), 0);
            assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the pin stands");
        }
    }
}

/// A claim's mark outlives its pin where a respill drops the pin: the
/// mark then names an id no pin of the key holds. Under `original` a
/// claim written ahead of the table's word of a rename stands behind
/// the key's served pin, and a respill ahead of that word drops it.
/// The rename brings the served pin under the mark's id, and the mark
/// goes there: that pin is no claim, and the claiming decision's
/// commit on the respill's worker withdraws nothing.
#[test]
fn a_mark_whose_pin_a_respill_dropped_goes_where_a_rename_brings_a_pin_under_it() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11), 0);
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(1)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    let moved = [WorkerId(2), WorkerId(11)];
    assert_eq!(table.pins(s(1)), Some(moved));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some(moved), "the served pin stands");
}

/// The limit of a claim written ahead of the table's word of a
/// rename, held as it stands. Under `latest` the claim leads the
/// key's served pin, and a respill ahead of that word drops the
/// served pin as it drops any failover: nothing in the table says the
/// two ids are one worker yet. The claim stands unconfirmed behind
/// the rename, and the claiming decision's commit on the respill's
/// worker withdraws it: the key holds no pin on that worker, where a
/// respill behind the walk keeps one.
#[test]
fn a_respill_ahead_of_the_tables_word_drops_the_served_pin_under_latest() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11), 0);
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(0)]));
}

/// A pair's moved claim reads its claim on the walked entry: the key
/// holds its served pin under the id a prefill had and the decision's
/// claim under the new id, written ahead of the table's word of the
/// rename, and the pair forms on another prefill, or on none, inside
/// the rename's walk. The key takes the walk's step ahead of the
/// move's own lines: the two pins are one and the mark is gone, so
/// the claim is a pin the key was served on, nothing moves and
/// nothing is claimed, as behind the whole walk. The pair's dispatch
/// moves the pins as a pin's alternate does.
#[test]
fn a_moved_claim_of_a_key_that_holds_one_worker_twice_is_read_on_the_walked_entry() {
    for rule in [RULE, ORIGINAL] {
        // The pair forms on another prefill, then on none.
        for paired in [5, 0] {
            let said = (rule.recovery, paired);
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
            table.rename_worker(WorkerId(1), WorkerId(11));
            let served = [WorkerId(11), WorkerId(0)];
            let claim = table.claim(s(1), served, WorkerId(21), 0);
            assert_eq!(claim, Claim::Made(WorkerId(21)), "{said:?}");
            table.remember(WorkerId(11), WorkerId(21));
            let verdict = table.move_claim(s(1), WorkerId(21), WorkerId(paired), 0);
            assert_eq!(verdict, SessionVerdict::Pinned, "{said:?}");
            let one = [WorkerId(21), WorkerId(0)];
            assert_eq!(table.pins(s(1)), Some(one), "{said:?}: one pin");
            table.rename_worker(WorkerId(11), WorkerId(21));
            assert_eq!(table.pins(s(1)), Some(one), "{said:?}");
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(5), WorkerId(21)],
                Recovery::Original => [WorkerId(21), WorkerId(5)],
            };
            assert_eq!(table.pins(s(1)), Some(both), "{said:?}");
        }
    }
}

/// A dispatch on a claim's own lease asks by the claim of record.
/// Under `original` a respill ahead of the table's word of a rename
/// drops the claim's pin, and the key keeps its served pin under the
/// id the worker had. Inside the rename's walk the claiming
/// decision's dispatch takes the walk's step for the key: the served
/// pin is the claimed worker's, the dispatch is the failover taken,
/// and the respill's worker stays, as behind the whole walk. Asked by
/// nothing, the dispatch would write the worker again and drop that
/// pin.
#[test]
fn a_dispatch_on_a_claims_lease_meets_the_keys_pin_of_its_worker_inside_the_walk() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11), 0);
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(1)]));
    // The rename's first half: remembered, the shards not yet walked.
    table.remember(WorkerId(1), WorkerId(11));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(11), 0);
    let taken = [WorkerId(2), WorkerId(11)];
    assert_eq!(table.pins(s(1)), Some(taken), "the failover taken");
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some(taken));
    // The claim is confirmed: a later commit elsewhere withdraws none.
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some(taken));
}

/// A claim reads the key's pins once more behind a rename's step. The
/// decision read the pin under the id its worker had, the rename's
/// walk has yet to reach the key, and the decision's pick is another
/// worker. The claim's step renames the pin, so the pins are not the
/// ones the decision read: the renamed pin answers as their newcomer,
/// and nothing is claimed, as where the whole walk came between the
/// read and the claim.
#[test]
fn a_claim_whose_step_renames_a_pin_answers_the_renamed_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        // The rename's first half: remembered, the shards not yet
        // walked.
        table.remember(WorkerId(1), WorkerId(11));
        let read = [WorkerId(1), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), read);
        let claim = table.claim(s(1), read, WorkerId(2), 0);
        let led = [WorkerId(11), WorkerId(0)];
        assert_eq!(claim, Claim::Standing(WorkerId(11), led));
        assert_eq!(table.pins(s(1)), Some(led), "nothing is claimed");
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some(led));
    }
}

/// Where the claim's step makes two pins of one worker one, the pins
/// are not the ones the decision read and hold no newcomer: the
/// answer is the decision's own pick, and nothing is claimed, as
/// where the whole walk came between the read and the claim.
#[test]
fn a_claim_whose_step_makes_two_pins_one_answers_its_own_pick() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        // Ahead of the table's word: a claim on the row that
        // replaced the worker's.
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11), 0);
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(11)));
        table.remember(WorkerId(1), WorkerId(11));
        let claim = table.claim(s(1), read, WorkerId(2), 0);
        let one = [WorkerId(11), WorkerId(0)];
        assert_eq!(claim, Claim::Standing(WorkerId(2), one));
        assert_eq!(table.pins(s(1)), Some(one), "nothing is claimed");
    }
}

/// Pins that moved under a decision are read behind a pending
/// rename's step. A host replaces a worker's row before it tells the
/// table. A decision reads the key's pin on the new rows, finds it
/// not routable and picks worker 2. A request decided on the earlier
/// rows commits on the replaced worker, under the id it had. The
/// rename is then said, and its walk has yet to reach the key. The
/// decision's claim names the newcomer by the id that stands, the
/// one its rows hold, and nothing is claimed. So too for key 2,
/// where another request committed on the decision's own pick as
/// well: a pin as written, which a write asks nothing for.
#[test]
fn a_claim_on_pins_that_moved_names_the_newcomer_by_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let latest = rule.recovery == Recovery::Latest;
        let table = SessionTable::new(rule.sweep_micros);
        // (key, the commits between the decision's read and its
        // claim, the pins behind the step). Key 2's commits come in
        // the order that leaves the replaced worker in the lead.
        let keys: [(u64, &[u64], [u64; 2]); 2] = [
            (1, &[1], if latest { [11, 5] } else { [5, 11] }),
            (2, if latest { &[2, 1] } else { &[1, 2] }, [11, 2]),
        ];
        let read = [WorkerId(5), WorkerId(0)];
        for (key, commits, _) in keys {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(5), 0);
            assert_eq!(table.lookup(s(key), 20, &rule), read);
            for worker in commits {
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(*worker), 0);
            }
        }
        // The rename's first half: remembered, the shards not yet
        // walked.
        table.remember(WorkerId(1), WorkerId(11));
        for (key, _, walked) in keys {
            let walked = walked.map(WorkerId);
            assert_ne!(table.pins(s(key)), Some(walked), "key {key} is not walked");
            let claim = table.claim(s(key), read, WorkerId(2), 0);
            assert_eq!(claim, Claim::Standing(WorkerId(11), walked), "key {key}");
            assert_eq!(
                table.pins(s(key)),
                Some(walked),
                "key {key}: the step alone"
            );
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        for (key, _, walked) in keys {
            let walked = walked.map(WorkerId);
            assert_eq!(
                table.pins(s(key)),
                Some(walked),
                "key {key} behind the walk"
            );
        }
    }
}

/// Pins the step makes the ones the decision read again stand as
/// read. One worker is registered three times around one decision:
/// it was 3 where the decision read its rows and 1 where the
/// decision read the key's pins, a rename to 11 walked the key, and
/// a rename back to 1 is said and has yet to walk it. The pins as
/// written are not the ones the decision read, and the step makes
/// them so. The decision's pick is that worker under its first id, a
/// pin of the key already: the table writes nothing and answers the
/// decision's own pick, as where the pins stood as written. A claim
/// there would mark a pin the key was served on, and the decision's
/// commit on an alternate would withdraw it.
#[test]
fn a_claim_on_pins_the_step_makes_the_ones_read_again_meets_its_workers_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.rename_worker(WorkerId(3), WorkerId(1));
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        let read = table.lookup(s(1), 20, &rule);
        let served = match rule.recovery {
            Recovery::Latest => [WorkerId(1), WorkerId(5)],
            Recovery::Original => [WorkerId(5), WorkerId(1)],
        };
        assert_eq!(read, served);
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(3), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(3), read));
        assert_eq!(table.pins(s(1)), Some(read), "the step alone");
        // The request's commit on an alternate withdraws nothing.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both), "the key keeps its pin");
    }
}

/// Pins the step makes the ones the decision read again stand as
/// read for a pick that is in neither pin too: the claim is taken, as
/// where the pins stood as written. One worker is registered twice
/// around one decision: it was 1 where the decision read the key's
/// pins, a rename to 11 walked the key, and a rename back to 1 is
/// said and has yet to walk it. The step makes the pins the ones
/// read, and the decision's pick, another worker, is claimed by the
/// key's recovery. The claim is marked: the claiming decision's
/// commit elsewhere withdraws it, and the key's pin stays.
#[test]
fn a_claim_on_pins_the_step_makes_the_ones_read_again_is_taken_for_a_new_worker() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(5)));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(2), 0);
        assert_eq!(claim, Claim::Made(WorkerId(2)));
        let claimed = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(claimed), "on the walked pins");
        // The claiming decision's commit lands elsewhere.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(7), 0);
        let withdrawn = match rule.recovery {
            Recovery::Latest => [WorkerId(7), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(7)],
        };
        assert_eq!(table.pins(s(1)), Some(withdrawn), "the claim is withdrawn");
    }
}

/// A pin the decision read under an id its worker had is no claim
/// either, ahead of the rename's walk. The key was served on worker 1,
/// the decision read that pin and its rows, and a rename of 1 to 11
/// is said with its walk pending. The decision's pick is 1, a pin as
/// written: the table takes no step, writes nothing and answers the
/// decision's own pick. The walk then renames the pin, and a commit
/// elsewhere that names a claim leaves the worker's pin under the id
/// that stands. A marked pin would follow the rename and go with that
/// commit. The gateway's sticky map names a key's workers by address
/// (`model_gateway/src/policies/manual.rs:120`, read by `:471-473`),
/// so a worker registered again is the key's candidate still.
#[test]
fn a_claim_of_a_pin_read_under_an_id_its_worker_had_is_no_claim() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        let read = table.lookup(s(1), 20, &rule);
        assert_eq!(read, [WorkerId(1), WorkerId(0)]);
        // The rename by its first half: remembered, the shards not
        // yet walked.
        table.remember(WorkerId(1), WorkerId(11));
        let claim = table.claim(s(1), read, WorkerId(1), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(1), read));
        assert_eq!(table.pins(s(1)), Some(read), "no step and no write");
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(9), 0);
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(11)],
            Recovery::Original => [WorkerId(11), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the worker's pin stays");
    }
}

/// A pin the decision read is no claim on pins a rename's step makes
/// the ones read again either. One worker is registered twice around
/// one decision: it was 1 where the decision read the key's pins, a
/// rename to 11 walked the key, and a rename back to 1 is said and
/// has yet to walk it. The decision's pick is that worker, a pin it
/// read. The step makes the pins the ones read, and the table writes
/// nothing more and answers the decision's own pick. A commit
/// elsewhere that names a claim then keeps the worker's pin. A marked
/// pin would go with that commit. The gateway's sticky map names a
/// key's workers by address
/// (`model_gateway/src/policies/manual.rs:120`, read by `:471-473`),
/// so a worker registered again twice is the key's candidate still.
#[test]
fn a_claim_of_a_pin_its_decision_read_is_no_claim_behind_a_renames_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5), 0);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(5)));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(1), 0);
        assert_eq!(claim, Claim::Standing(WorkerId(1), read));
        assert_eq!(table.pins(s(1)), Some(read), "the step alone");
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(9), 0);
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the worker's pin stays");
    }
}

/// A claim written ahead of the table's word of a rename finds no
/// rename to ask: a host replaces a worker's row before it tells the
/// table, and a decision between the two reads a pin no pool holds
/// and is placed by the rule. The key then holds the worker twice, as
/// a claim and as a pin, in both orders: a claim on the new row beside
/// the pin under the id the worker had, and a commit on the new row
/// beside a claim under that id. The rename's walk makes the two pins
/// one and the claim's mark goes there: the key holds the worker as a
/// pin, and the claiming decision's commit on an alternate withdraws
/// nothing.
#[test]
fn a_claim_made_one_with_the_keys_pin_by_a_rename_is_confirmed() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Key 1 was served on the worker and is claimed on the row
        // that replaced it.
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11), 0);
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        // Key 2 is claimed on the worker and served on the row that
        // replaced it.
        table.lookup(s(2), 10, &rule);
        let claim = table.claim(s(2), [WorkerId(0); 2], WorkerId(1), 0);
        assert_eq!(claim, Claim::Made(WorkerId(1)));
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11), 0);
        table.rename_worker(WorkerId(1), WorkerId(11));
        // (key, the id its claim's lease names)
        for (key, leased) in [(1, 11), (2, 1)] {
            let led = [WorkerId(11), WorkerId(0)];
            assert_eq!(table.pins(s(key)), Some(led), "key {key} holds one pin");
            // The claiming decision's commit on an alternate.
            let claimed = SessionVerdict::Claimed(WorkerId(leased));
            table.dispatched(s(key), claimed, WorkerId(2), 0);
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(11)],
                Recovery::Original => [WorkerId(11), WorkerId(2)],
            };
            assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
        }
    }
}

/// A claim's commit elsewhere withdraws it inside a rename's walk too:
/// the renames already say the new id, the entry still holds the old
/// one, and the claim is matched by the id its lease names.
#[test]
fn a_claims_commit_elsewhere_inside_the_renames_walk_withdraws_it() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(9), 10, &ORIGINAL);
    let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1), 0);
    assert_eq!(claim, Claim::Made(WorkerId(1)));
    // The rename's first half: remembered, the shard not yet walked.
    table.remember(WorkerId(1), WorkerId(11));
    table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
}

/// A claim outlives the renames the table forgets: the id its lease
/// names is on its entry, so its commit elsewhere withdraws it behind
/// a list that started over.
#[test]
fn a_claim_renamed_ahead_of_the_bound_is_withdrawn_by_its_commit() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(9), 10, &ORIGINAL);
    let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1), 0);
    assert_eq!(claim, Claim::Made(WorkerId(1)));
    table.rename_worker(WorkerId(1), WorkerId(11));
    let bound = RENAMED_WORKERS as u64;
    for life in 0..bound {
        table.rename_worker(WorkerId(100_000 + life), WorkerId(200_000 + life));
    }
    assert_eq!(table.renamed(), 1, "the list started over");
    table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2), 0);
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
}

/// Its own dispatch confirms such a claim behind a list that started
/// over too. The dispatch names the id the claim's lease names, and no
/// rename the table remembers says which id stands for it. The claim of
/// record does: the key keeps its worker's pin under the id that stands,
/// a pin the key was served on.
#[test]
fn a_claim_renamed_ahead_of_the_bound_is_confirmed_by_its_dispatch() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(9), 10, &ORIGINAL);
    let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1), 0);
    assert_eq!(claim, Claim::Made(WorkerId(1)));
    table.rename_worker(WorkerId(1), WorkerId(11));
    let bound = RENAMED_WORKERS as u64;
    for life in 0..bound {
        table.rename_worker(WorkerId(100_000 + life), WorkerId(200_000 + life));
    }
    assert_eq!(table.renamed(), 1, "the list started over");
    table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1), 0);
    let led = [WorkerId(11), WorkerId(0)];
    assert_eq!(table.pins(s(9)), Some(led));
    // The claim is a pin now: its decision's refusal withdraws nothing.
    let verdict = table.move_claim(s(9), WorkerId(1), WorkerId(0), 0);
    assert_eq!(verdict, SessionVerdict::Pinned);
    assert_eq!(table.pins(s(9)), Some(led));
}

/// The table remembers a bounded count of renames: at the bound the
/// list starts over, and the rename that follows is remembered.
#[test]
fn the_renames_the_table_remembers_are_bounded() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    let bound = RENAMED_WORKERS as u64;
    for life in 1..=bound {
        table.rename_worker(WorkerId(life), WorkerId(life + 1));
    }
    assert_eq!(table.renamed(), RENAMED_WORKERS);
    table.rename_worker(WorkerId(bound + 1), WorkerId(bound + 2));
    assert_eq!(table.renamed(), 1);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(bound + 1), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(bound + 2), WorkerId(0)]));
}

/// A worker that left keeps its pins. Its key is served elsewhere
/// meanwhile, and the worker registered again under a new id leads the
/// key again under `original`.
#[test]
fn a_worker_that_left_keeps_its_pins_and_takes_them_back_under_its_new_id() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &ORIGINAL);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
    }
    table.left(WorkerId(1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.lookup(s(1), 20, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
}

/// Under `latest` the worker that served the key in the absence keeps
/// the lead, and the one registered again is its failover.
#[test]
fn under_latest_recovery_a_worker_registered_again_is_the_failover() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.left(WorkerId(1));
    table.lookup(s(1), 20, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
}

/// A worker that stays away is forgotten once the absence's time
/// passes on the decision clock, counted from the first decision
/// behind the leave. A table no rule sweeps counts it too.
#[test]
fn a_worker_that_stays_away_is_forgotten_at_the_absences_end() {
    for sweep in [RULE.sweep_micros, 0] {
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: sweep,
            ..ORIGINAL
        };
        let table = SessionTable::new(sweep).with_absence(500);
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
        }
        table.left(WorkerId(1));
        // The decision at 100 stamps the end: 600.
        table.lookup(s(2), 100, &rule);
        table.lookup(s(2), 599, &rule);
        let stands = Some([WorkerId(1), WorkerId(0)]);
        assert_eq!(table.pins(s(1)), stands, "inside its time, sweep {sweep}");
        table.lookup(s(2), 600, &rule);
        assert_eq!(table.pins(s(1)), None, "its only pin went, sweep {sweep}");
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    }
}

/// A worker registered again inside its time is not forgotten at its
/// end, under a new id or under the one it had.
#[test]
fn a_worker_registered_again_in_time_is_not_forgotten_at_the_absences_end() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    table.lookup(s(1), 10, &rule);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.left(WorkerId(1));
    table.lookup(s(1), 100, &rule);
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.lookup(s(1), 10_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
    table.left(WorkerId(11));
    table.rename_worker(WorkerId(11), WorkerId(11));
    table.lookup(s(1), 20_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
}

/// A join under the id that left ends its absence, by its own word or
/// by a snapshot that lists it: no end of the absence takes the pins of
/// the worker that is back.
#[test]
fn a_join_under_the_id_that_left_ends_its_absence() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
    }
    table.left(WorkerId(1));
    table.returned(WorkerId(1));
    assert_eq!(table.absent(), 0);
    table.lookup(s(2), 100, &rule);
    table.lookup(s(2), 10_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.left(WorkerId(1));
    table.left(WorkerId(2));
    table.returned_all(&[WorkerId(2)]);
    assert_eq!(table.absent(), 1, "the one the snapshot does not list");
    table.lookup(s(2), 20_000, &rule);
    table.lookup(s(2), 30_000, &rule);
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    assert_eq!(table.pins(s(1)), None, "the worker that stayed away");
}

/// A leave asks for a sweep, and that sweep walks no idle key ahead of
/// the pass over them: the pass keeps its own instant. The key past its
/// end stays until the pass is due, as with no leave.
#[test]
fn a_leave_walks_no_idle_key_before_their_pass_is_due() {
    let rule = SessionRule {
        ttl_micros: Some(100),
        sweep_micros: 1_000,
        ..ORIGINAL
    };
    let table = SessionTable::new(rule.sweep_micros).with_absence(5_000);
    // The first lookup runs the pass: the next is due at 1,010.
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
    }
    table.left(WorkerId(9));
    // Key 1 is past its end, the sweep at 1,000, and the leave's sweep
    // runs.
    table.lookup(s(2), 1_005, &rule);
    assert_eq!(table.absent(), 1, "the leave is stamped");
    assert!(table.pins(s(1)).is_some(), "no pass before 1,010");
    table.lookup(s(2), 1_010, &rule);
    assert_eq!(table.pins(s(1)), None, "the pass at its own instant");
}

/// A table that holds no key files no absence: no pin names the
/// worker, and no decision would end the entry.
#[test]
fn a_table_that_holds_no_key_files_no_absence() {
    let table = SessionTable::new(0).with_absence(500);
    table.left(WorkerId(1));
    assert_eq!(table.absent(), 0);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2), 0);
    table.left(WorkerId(1));
    assert_eq!(table.absent(), 1);
}

/// The absent workers are bounded: at the bound the one that left
/// first is forgotten at once, its pins with it.
#[test]
fn at_the_bound_of_absent_workers_the_first_to_leave_is_forgotten() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    for (key, worker) in [(1, 1), (2, 9_999)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker), 0);
    }
    let bound = ABSENT_WORKERS as u64;
    for worker in 1..=bound {
        table.left(WorkerId(worker));
    }
    assert_eq!(table.absent(), ABSENT_WORKERS);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.left(WorkerId(bound + 1));
    assert_eq!(table.absent(), ABSENT_WORKERS);
    assert_eq!(table.pins(s(1)), None, "the first to leave");
    assert_eq!(table.pins(s(2)), Some([WorkerId(9_999), WorkerId(0)]));
}
