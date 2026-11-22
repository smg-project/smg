//! A host's hash ring as one list of its workers reads it: the cases.
//!
//! The reference in each is the host's own lookup
//! ([`HashRing::find_healthy_url`]), the one its classic policy asks.

use super::*;
use crate::Draw;

/// The addresses of `count` workers in an order `shape` draws: a list's
/// order is its host's, and no order of the ring's.
fn addresses(count: u32, shape: u64) -> Vec<String> {
    let mut draw = Draw::new(shape);
    let mut addresses: Vec<String> = (0..count)
        .map(|at| format!("http://worker-{at}.pool-{shape}:8000"))
        .collect();
    for at in (1..addresses.len()).rev() {
        addresses.swap(at, draw.below(at + 1));
    }
    addresses
}

/// The ids of the rows of a list whose workers stand at `addresses`.
fn ids(addresses: &[String]) -> Vec<WorkerId> {
    addresses
        .iter()
        .map(|at| WorkerId::from_address(at))
        .collect()
}

/// A list's rows by id, as its host keeps them: ids ascending, each with
/// its row's place.
fn by_id(ids: &[WorkerId]) -> Vec<(WorkerId, u32)> {
    let mut by_id: Vec<(WorkerId, u32)> = ids.iter().copied().zip(0u32..).collect();
    by_id.sort_unstable();
    by_id
}

/// The host's ring of `addresses`, built over them in an order of its own.
fn ring_of(addresses: &[String], shape: u64) -> Arc<HashRing> {
    let mut draw = Draw::new(shape ^ 0xbead);
    let mut order: Vec<&String> = addresses.iter().collect();
    for at in (1..order.len()).rev() {
        order.swap(at, draw.below(at + 1));
    }
    Arc::new(HashRing::new(order))
}

/// The workers the host's own lookup answers for `key`, in the order its
/// classic policy would try them: each is asked for with the ones before it
/// refused, among the workers `keep` keeps.
fn by_lookup(ring: &HashRing, key: &str, keep: impl Fn(&str) -> bool) -> Vec<String> {
    let mut owners: Vec<String> = Vec::new();
    while owners.len() < MAX_TARGETS {
        let next = ring.find_healthy_url(key, |address| {
            keep(address) && !owners.iter().any(|owner| owner == address)
        });
        match next {
            Some(next) => owners.push(next.to_owned()),
            None => break,
        }
    }
    owners
}

/// The workers a walk of the list's ring hands out for `key`, among the
/// rows `keep` keeps.
fn by_walk(
    ring: &ListRing<'_>,
    addresses: &[String],
    key: &str,
    keep: impl Fn(u32) -> bool,
) -> Vec<String> {
    let mut out = [0; MAX_TARGETS];
    let handed = ring.walk(HashRing::hash_position(key), keep, &mut out);
    out[..handed]
        .iter()
        .map(|&row| addresses[row as usize].clone())
        .collect()
}

/// One key has one owner by the host's own lookup and by a walk of the same
/// ring as a list reads it, and the workers behind the owner follow in one
/// order: over lists of every size a gateway holds, in an order of the
/// host's own, for four thousand keys a list. The ring is read through the
/// list's table and with none, and the two agree.
#[test]
fn a_walk_hands_a_key_the_workers_the_hosts_lookup_answers() {
    for (count, shape) in [(1, 1), (2, 2), (3, 3), (8, 4), (64, 5), (512, 6)] {
        let addresses = addresses(count, shape);
        let (ids, ring) = (ids(&addresses), ring_of(&addresses, shape));
        let places = RingPlaces::new(&ring, &ids);
        assert!(places.is_of(&ring));
        let kept = places.list_ring();
        assert_eq!(
            (kept.len(), kept.has_distinct_ids()),
            (count as usize, true)
        );
        let by_id = by_id(&ids);
        let sought = ListRing::by_search(&ring, ids.len(), &by_id);
        for key in 0..4_000 {
            let key = format!("session-{shape}-{key}");
            let owners = by_lookup(&ring, &key, |_| true);
            assert_eq!(
                by_walk(&kept, &addresses, &key, |_| true),
                owners,
                "{key} over {count} workers, by the table"
            );
            assert_eq!(
                by_walk(&sought, &addresses, &key, |_| true),
                owners,
                "{key} over {count} workers, with no table"
            );
        }
    }
}

/// A worker the request may not take is passed over by the walk as by the
/// host's lookup: the key goes to the next worker on the ring that it may.
#[test]
fn a_walk_passes_over_the_rows_refused_as_the_hosts_lookup_does() {
    let addresses = addresses(64, 7);
    let (ids, ring) = (ids(&addresses), ring_of(&addresses, 7));
    let places = RingPlaces::new(&ring, &ids);
    let kept = places.list_ring();
    // One worker in five is refused: by its row for the walk, by its
    // address for the lookup.
    let refused = |row: u32| row.is_multiple_of(5);
    let refused_at = |address: &str| {
        let row = addresses.iter().position(|at| at == address);
        refused(row.expect("an address of the list") as u32)
    };
    let mut passed = 0;
    for key in 0..4_000 {
        let key = format!("session-{key}");
        let owners = by_lookup(&ring, &key, |address| !refused_at(address));
        assert_eq!(
            by_walk(&kept, &addresses, &key, |row| !refused(row)),
            owners,
            "{key}"
        );
        passed += u32::from(by_lookup(&ring, &key, |_| true)[0] != owners[0]);
    }
    // A fifth of the keys had an owner that was refused.
    assert!((500..=1_100).contains(&passed), "{passed} of 4000 keys");
}

/// A ring of another membership than the list: a worker that left is passed
/// over, a worker that joined is met by no walk, and every other key keeps
/// its owner, as the host's lookup on that ring answers. The ring is read
/// through a table and with none.
#[test]
fn a_ring_of_another_membership_places_as_the_hosts_lookup_on_it_does() {
    // The ring's membership: sixteen workers. The list's: the first of
    // them left, and two joined.
    let earlier = addresses(16, 8);
    let ring = ring_of(&earlier, 8);
    let mut listed: Vec<String> = earlier[1..].to_vec();
    listed.push("http://joined-a.pool:8000".into());
    listed.push("http://joined-b.pool:8000".into());
    let ids = ids(&listed);
    let (places, by_id) = (RingPlaces::new(&ring, &ids), by_id(&ids));
    let rings = [
        places.list_ring(),
        ListRing::by_search(&ring, ids.len(), &by_id),
    ];
    let here = |address: &str| listed.iter().any(|at| at == address);
    let mut moved = 0;
    for key in 0..4_000 {
        let key = format!("session-{key}");
        // The lookup takes an address only where the list holds a worker
        // at it, as the host's classic policy asks it.
        let owners = by_lookup(&ring, &key, here);
        assert!(owners
            .iter()
            .all(|owner| !owner.starts_with("http://joined")));
        for ring in &rings {
            assert_eq!(by_walk(ring, &listed, &key, |_| true), owners, "{key}");
        }
        moved += u32::from(by_lookup(&ring, &key, |_| true)[0] == earlier[0]);
    }
    // The worker that left held a sixteenth of the keys.
    assert!((150..=350).contains(&moved), "{moved} of 4000 keys");
    // A list no worker of which the ring holds: no walk meets a row.
    let apart = self::ids(&addresses(4, 9));
    let (places, sorted) = (RingPlaces::new(&ring, &apart), self::by_id(&apart));
    let mut out = [0; MAX_TARGETS];
    for ring in [places.list_ring(), ListRing::by_search(&ring, 4, &sorted)] {
        assert_eq!(ring.walk(77, |_| true, &mut out), 0);
    }
}

/// A walk asks about each row once, hands out the first rows kept in the
/// ring's order, and says nothing once it met its bound of refused rows or
/// asked every row and kept none.
#[test]
fn a_walk_asks_each_row_once_and_stops_at_its_bounds() {
    let addresses = addresses(64, 10);
    let (ids, ring) = (ids(&addresses), ring_of(&addresses, 10));
    let places = RingPlaces::new(&ring, &ids);
    let ring = places.list_ring();
    let mut keys = Draw::new(11);
    for _ in 0..500 {
        let position = keys.next_u64();
        // One row in three refused: the rows handed are the first kept of
        // the ring's order, and no row is asked about twice.
        let refused = |row: u32| row.is_multiple_of(3);
        let mut asked: Vec<u32> = Vec::new();
        let mut out = [0; MAX_TARGETS];
        let handed = ring.walk(
            position,
            |row| {
                assert!(!asked.contains(&row), "row {row} asked twice");
                asked.push(row);
                !refused(row)
            },
            &mut out,
        );
        assert!(asked.len() < MAX_TARGETS + DRAW_BOUND as usize);
        if handed != 0 {
            assert_eq!(handed, MAX_TARGETS);
            let kept: Vec<u32> = asked.iter().copied().filter(|&row| !refused(row)).collect();
            assert_eq!(out[..handed], kept[..]);
        }
        // Every row refused: the walk stops at its bound.
        let mut count = 0;
        let none = ring.walk(
            position,
            |_| {
                count += 1;
                false
            },
            &mut out,
        );
        assert_eq!((none, count), (0, DRAW_BOUND));
    }
    // A small list is asked about whole, and its rows kept are handed.
    let few = self::addresses(3, 12);
    let (ids, small) = (self::ids(&few), ring_of(&few, 12));
    let places = RingPlaces::new(&small, &ids);
    let ring = places.list_ring();
    let mut out = [0; MAX_TARGETS];
    let mut count = 0;
    let handed = ring.walk(
        77,
        |row| {
            count += 1;
            row != 1
        },
        &mut out,
    );
    assert_eq!((handed, count), (2, 3));
    assert!(!out[..handed].contains(&1));
    assert_eq!(ring.walk(77, |_| false, &mut out), 0);
    // No row, no walk.
    let empty = Arc::new(HashRing::new(std::iter::empty::<&str>()));
    let places = RingPlaces::new(&empty, &[]);
    assert!(places.list_ring().is_empty());
    assert_eq!(places.list_ring().walk(77, |_| true, &mut out), 0);
}

/// A list where two rows carry one id is one member twice over: its table
/// says so, a walk of it hands out nothing, and the pass over its rows
/// finds both rows at the slot of the first.
#[test]
fn a_list_with_one_id_twice_is_not_walked() {
    let mut addresses = addresses(8, 13);
    addresses.push(addresses[3].clone());
    let (ids, ring) = (ids(&addresses), ring_of(&addresses, 13));
    let places = RingPlaces::new(&ring, &ids);
    let ring = places.list_ring();
    assert_eq!((ring.len(), ring.has_distinct_ids()), (9, false));
    let mut out = [0; MAX_TARGETS];
    assert_eq!(ring.walk(5, |_| true, &mut out), 0);
    // Every row kept: the last row of the id takes the slot of the first.
    let mut slots = Vec::new();
    ring.fill(&[0, 1, 2, 3, 4, 5, 6, 7, 8], &mut slots);
    assert_eq!(slots, [0, 1, 2, 8, 4, 5, 6, 7, FREE]);
    // The first of the two alone kept: the slot is its own.
    ring.fill(&[3, 5], &mut slots);
    assert_eq!(slots[3], 0);
    assert_eq!(slots[5], 1);
    assert_eq!(slots[8], FREE);
    // The ring holds the address once: its entries name the first row.
    assert!(ring.around(0).all(|row| row != 8));
}

/// A table is of the ring it was made for, by the ring's identity: a ring
/// of the same addresses built again is another ring.
#[test]
fn a_table_is_of_the_one_ring_it_was_made_for() {
    let addresses = addresses(8, 14);
    let (ids, ring) = (ids(&addresses), ring_of(&addresses, 14));
    let places = RingPlaces::new(&ring, &ids);
    assert!(places.is_of(&ring));
    assert!(places.is_of(&Arc::clone(&ring)));
    assert!(!places.is_of(&ring_of(&addresses, 14)));
}
