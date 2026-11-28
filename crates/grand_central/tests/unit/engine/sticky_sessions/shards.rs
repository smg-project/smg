//! The table's size: its shards, the keys spread over them, and a full
//! table.

use super::*;

#[test]
fn a_table_of_any_size_keeps_every_key_and_is_sized_by_a_power_of_two() {
    for (asked, shards) in [(0, 1), (1, 1), (3, 4), (16, 16), (1_000, 1_024)] {
        let table = SessionTable::new(RULE.sweep_micros).with_shards(asked);
        assert_eq!(table.shards(), shards, "{asked} shards asked");
        for count in 1..=200 {
            let key = mixed(count);
            assert_eq!(table.lookup(key, 10, &RULE), [WorkerId(0); 2]);
            table.dispatched(key, SessionVerdict::Pinned, WorkerId(count), 0);
            assert_eq!(table.lookup(key, 11, &RULE)[0], WorkerId(count));
        }
        assert_eq!(table.len(), 200);
        // A walk of the table reaches every shard.
        table.forget_workers(&[WorkerId(200)]);
        assert_eq!(table.pins(mixed(200)), None);
        assert_eq!(table.len(), 199);
    }
    // A table no host sized holds the count it always held.
    assert_eq!(SessionTable::new(0).shards(), SHARDS);
    assert!(shards_for_this_host().is_power_of_two());
    assert!(shards_for_this_host() >= 4);
}

/// The keys of a sized table spread over its shards: no shard holds
/// many times its share, so two decisions of different keys seldom
/// take one mutex.
#[test]
fn a_sized_tables_keys_spread_over_its_shards() {
    let table = SessionTable::new(0).with_shards(64);
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..RULE
    };
    let keys = 64 * 100;
    for count in 1..=keys {
        table.lookup(mixed(count), 10, &rule);
    }
    let held: Vec<usize> = table.shards.iter().map(|s| s.0.lock().len()).collect();
    assert_eq!(held.iter().sum::<usize>(), keys as usize);
    let (least, most) = (held.iter().min(), held.iter().max());
    assert!(
        least >= Some(&50) && most <= Some(&200),
        "a share of 100 a shard: {least:?} to {most:?}"
    );
}

/// A full table records no further key, in whichever shard it falls:
/// such a key reads no pin, is written none and claims none, while the
/// keys the table holds are answered. Keys that leave at their end give
/// their room back, and their shard its spare storage.
#[test]
fn a_full_table_records_no_further_key_and_answers_the_keys_it_holds() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in 1..=8 {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1), 0);
    }
    // One key under the 4,194,304 a table holds: the count stands for
    // the keys no test inserts, and key 9 takes the last room.
    table.held.0.store(4_194_303, Ordering::Relaxed);
    table.lookup(s(9), 10, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS);
    // A key of another shard finds the table full.
    let fresh = s(1 << 32);
    let none = [WorkerId(0); 2];
    assert_eq!(table.lookup(fresh, 20, &RULE), none);
    assert_eq!(table.claim(fresh, none, WorkerId(2), 0), Claim::NoRoom);
    table.dispatched(fresh, SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(fresh), None);
    // A key the table holds is read and renewed as before.
    assert_eq!(table.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
    // The other eight end at 1 100 and leave at the pass.
    let room = table.shards[0].0.lock().capacity();
    table.sweep_due(1_200);
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 8);
    assert!(table.shards[0].0.lock().capacity() < room);
    assert_eq!(table.lookup(fresh, 1_200, &RULE), none);
    assert_eq!(
        table.claim(fresh, none, WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 7);
}

#[test]
fn a_shard_stands_on_cache_lines_of_its_own() {
    assert_eq!(align_of::<Shard>(), 128);
    assert_eq!(size_of::<Shard>(), 128);
}
