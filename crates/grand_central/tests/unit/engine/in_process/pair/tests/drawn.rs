//! A pair whose two sides pick of two is decided by each side's draw where
//! its request states nothing beside its two lists. The cases here decide
//! such a pair both ways and hold the two answers together, and state the
//! requests, the lists and the documents the draw is not asked of.

use std::sync::atomic::{AtomicU32, Ordering};

use super::*;

/// No standing veto of the host on either list.
const UNMASKED: [[&[u32]; 2]; 2] = [[&[], &[]], [&[], &[]]];

/// The pair presets under `prefill` and `decode`, validated.
fn presets(prefill: &str, decode: &str) -> AppliedSpec {
    let mut document = PolicySpec::pair_presets(prefill, decode).expect("the pair presets");
    document.generation = 3;
    AppliedSpec::validate(&document).expect("a valid document")
}

/// A host of `prefills` and `decodes` whose two sides pick of two.
fn drawing(prefills: usize, decodes: usize) -> Host {
    let spec = presets("power_of_two", "power_of_two");
    Host::under((spec, Arc::default()), prefills, decodes)
}

/// An admission that counts what it is asked: each prefill's capacity bit
/// and each decode's room bit. The prefill at `refused` has no capacity
/// and the decode at `full` no room.
#[derive(Debug, Default)]
struct Counted {
    places: AtomicU32,
    rooms: AtomicU32,
    refused: Option<usize>,
    full: Option<usize>,
}

impl Counted {
    /// The capacity bits and the room bits asked.
    fn asked(&self) -> (u32, u32) {
        (
            self.places.load(Ordering::Relaxed),
            self.rooms.load(Ordering::Relaxed),
        )
    }
}

impl PairAdmission for Counted {
    fn has_capacity(&self, prefill: usize) -> bool {
        self.places.fetch_add(1, Ordering::Relaxed);
        self.refused != Some(prefill)
    }

    fn has_room(&self, decode: usize) -> bool {
        self.rooms.fetch_add(1, Ordering::Relaxed);
        self.full != Some(decode)
    }
}

#[test]
fn a_read_of_the_gates_asks_every_place_once_and_says_whether_none_is_refused() {
    // `Gates` states its two reads alone: the prefill at place 2 has no
    // capacity and the decode at place 1 no room.
    let gates = Gates(|prefill| prefill != 2, |decode| decode != 1);
    let mut read = GatesRead::default();
    assert!(read.read(&gates, (0, 0)));
    assert!(read.read(&gates, (2, 1)));
    assert!(!read.read(&gates, (3, 1)), "the prefill at place 2 refuses");
    assert!(!read.read(&gates, (2, 2)), "the decode at place 1 refuses");
    assert!(!read.read(&gates, (8, 8)));
    assert!(read.read(&OPEN, (8, 8)));
    // A refused place does not end the read: the whole pass that follows
    // takes every answer from it.
    let counted = Counted {
        refused: Some(0),
        ..Counted::default()
    };
    assert!(!read.read(&counted, (8, 8)));
    assert_eq!(counted.asked(), (8, 8));
}

#[test]
fn every_row_pairs_and_is_admitted_under_one_class_a_side_alone() {
    let all = every(8, 8);
    assert!(one_pair(&all, (8, 8)));
    // A list longer than its classes holds a row of no class.
    assert!(!one_pair(&all, (9, 8)));
    assert!(!one_pair(&all, (8, 9)));
    // One prefill with no capacity, one decode with no room: the
    // admission says so by itself, whatever the classes.
    let mut read = GatesRead::default();
    assert!(read.read(&OPEN, (8, 8)));
    assert!(!read.read(&Gates(|at| at != 3, |_| true), (8, 8)));
    assert!(!read.read(&Gates(|_| true, |at| at != 5), (8, 8)));
    // The second prefill pairs with the first decode alone: two classes.
    let narrowed = PartnerClasses::number(2, [&[0, 1][..], &[0][..]]);
    assert!(!one_pair(&narrowed, (2, 2)));
}

#[test]
fn a_binding_says_at_one_byte_whether_its_pair_draws() {
    let spec = presets("power_of_two", "power_of_two");
    let engine = spec.engine();
    let (unsaid, plain) = (pool(&[]), Constraints::default());
    assert!(engine.pairs_drawn());
    assert!(engine
        .pair_drawn_for(&facts(), &unsaid, [8, 8], &plain)
        .is_some());
    // A list under the floor of a pair's pick of two, on either side.
    assert!(engine
        .pair_drawn_for(&facts(), &unsaid, [7, 8], &plain)
        .is_none());
    assert!(engine
        .pair_drawn_for(&facts(), &unsaid, [8, 7], &plain)
        .is_none());
    // A request with a session key.
    assert!(engine
        .pair_drawn_for(&keyed(), &unsaid, [8, 8], &plain)
        .is_none());
    // A side that does not pick of two: the binding's byte says no, and a
    // document with no drawing pair says so at the engine's one byte.
    for (prefill, decode) in [
        ("round_robin", "round_robin"),
        ("power_of_two", "round_robin"),
        ("round_robin", "power_of_two"),
    ] {
        let spec = presets(prefill, decode);
        assert!(!spec.engine().pairs_drawn(), "{prefill} and {decode}");
        assert!(spec
            .engine()
            .pair_drawn_for(&facts(), &unsaid, [8, 8], &plain)
            .is_none());
    }
}

/// The pair of each of 256 seeds over `host`'s two lists, decided by its
/// draws and decided whole, the two answers of a seed held together: the
/// places every seed's pair names.
fn both_ways(host: &Host, prefills: usize, decodes: usize) -> Vec<(usize, usize)> {
    let every_prefill: Vec<u32> = (0..prefills as u32).collect();
    let mut picks = Vec::new();
    for seed in 0..256 {
        // The draw: every place is asked once ahead of it, and the rows
        // it writes ask no place again.
        let counted = Counted::default();
        let drawn = host.decide_vetoed(
            &counted,
            (seed, None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        assert_eq!(
            counted.asked(),
            (prefills as u32, decodes as u32),
            "the draw asks each place once"
        );
        // A host that holds the request to every place of its prefill
        // list narrows nothing, and its pair is decided whole: every
        // prefill's capacity bit is read.
        let counted = Counted::default();
        let whole = host.decide_vetoed(
            &counted,
            (seed, None),
            Constraints::default(),
            [Some(&every_prefill), None],
            UNMASKED,
        );
        assert_eq!(
            counted.asked(),
            (prefills as u32, decodes as u32),
            "the whole pass asks each place once"
        );
        assert!(drawn.pick().is_some(), "seed {seed}: {drawn:?}");
        assert_eq!(
            format!("{drawn:?}"),
            format!("{whole:?}"),
            "seed {seed} over {prefills} and {decodes}"
        );
        picks.push(host.picked(&drawn));
    }
    picks
}

#[test]
fn a_pick_of_two_pair_is_decided_by_its_draws_as_the_whole_pass_decides_it() {
    for (prefills, decodes) in [(8_usize, 8_usize), (16, 9), (16, 16), (64, 64)] {
        // Every load equal: the two rows a side draws tie, and the
        // request's seed orders them.
        let level = both_ways(&drawing(prefills, decodes), prefills, decodes);
        // Loads that differ: the less loaded of two rows leads, and two
        // rows of one load still tie.
        let host = drawing(prefills, decodes);
        for (at, worker) in host.prefills.iter().chain(host.decodes.iter()).enumerate() {
            for _ in 0..at % 3 {
                worker.increment_load();
            }
        }
        let loaded = both_ways(&host, prefills, decodes);
        assert_ne!(
            level, loaded,
            "the loads move a pick over {prefills} and {decodes}"
        );
    }
}

#[test]
fn a_request_or_a_list_the_draw_is_not_asked_of_is_decided_whole() {
    let host = drawing(8, 8);
    // The whole pass asks a prefill's capacity bit once at most.
    let whole = |decision: Decision, counted: &Counted| {
        let (places, _) = counted.asked();
        assert!(decision.pick().is_some(), "{decision:?}");
        assert!(places >= 7, "{places} capacity bits read by the whole pass");
        places
    };
    // A retry: the host names a worker it just failed to reach.
    let counted = Counted::default();
    let failed = [worker_id(host.decodes[0].as_ref())];
    let decision = host.decide_excluding(&counted, 5, None, &failed);
    assert!(whole(decision, &counted) <= 8);
    // The host holds the request to places of its decode list.
    let counted = Counted::default();
    let every_prefill = [0, 1, 2, 3, 4, 5, 6, 7];
    let places = [Some(&every_prefill[..]), Some(&[1, 2, 3][..])];
    let decision = host.decide_among(&counted, 5, None, Constraints::default(), places);
    assert!(whole(decision, &counted) <= 8);
    // A prefill with no capacity: the draw's question of every place
    // says no, the pair picks among the other prefills, and its whole
    // pass asks no place again.
    let counted = Counted {
        refused: Some(3),
        ..Counted::default()
    };
    let decision = host.decide(&counted, 5, None);
    assert_ne!(host.picked(&decision).0, 3);
    assert_eq!(whole(decision, &counted), 8);
    assert_eq!(counted.asked(), (8, 8));
    // A prefill list under the floor of a pair's pick of two.
    let small = drawing(7, 8);
    let counted = Counted::default();
    let decision = small.decide(&counted, 5, None);
    assert_eq!(whole(decision, &counted), 7);
    // A document whose sides do not pick of two.
    let turns = Host::under(
        (presets("round_robin", "round_robin"), Arc::default()),
        8,
        8,
    );
    let counted = Counted::default();
    let decision = turns.decide(&counted, 5, None);
    assert_eq!(whole(decision, &counted), 8);
}

#[test]
fn a_retry_that_names_a_tried_worker_is_decided_whole() {
    let host = drawing(8, 8);
    let every_prefill = [0, 1, 2, 3, 4, 5, 6, 7];
    // A retry: the request names the prefill it tried. The whole pass
    // drops that row ahead of its two draws, which then stand over seven
    // rows: the draw of a side is not asked.
    let tried = [worker_id(host.prefills[3].as_ref())];
    for seed in 0..256 {
        let stated = || RequestFacts {
            seed,
            tried: &tried,
            ..facts()
        };
        let counted = Counted::default();
        let retried = host.decide_stated(
            &counted,
            (stated(), None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        let (places, _) = counted.asked();
        let whole = host.decide_stated(
            &Counted::default(),
            (stated(), None),
            Constraints::default(),
            [Some(&every_prefill[..]), None],
            UNMASKED,
        );
        assert_ne!(host.picked(&retried).0, 3, "seed {seed}: {retried:?}");
        assert_eq!(
            format!("{retried:?}"),
            format!("{whole:?}"),
            "seed {seed}: a retry names the pair of the whole pass"
        );
        assert!(
            (7..=8).contains(&places),
            "seed {seed}: {places} capacity bits read by the whole pass"
        );
    }
}

#[test]
fn a_pair_with_a_refused_place_is_decided_whole_and_asks_each_place_once() {
    let host = drawing(8, 8);
    let every_prefill: Vec<u32> = (0..8).collect();
    // The last decode has no room; then one prefill has no capacity.
    for (refused, full) in [(None, Some(7)), (Some(3), None)] {
        let gates = || Counted {
            refused,
            full,
            ..Counted::default()
        };
        for seed in 0..256 {
            // The draw is asked, and its question of every place says no:
            // the whole pass stamps its rows from the answers it read.
            let asked = gates();
            let decision = host.decide_vetoed(
                &asked,
                (seed, None),
                Constraints::default(),
                [None, None],
                UNMASKED,
            );
            assert_eq!(asked.asked(), (8, 8), "seed {seed}: a place is asked once");
            assert!(decision.pick().is_some(), "seed {seed}: {decision:?}");
            let (prefill, decode) = host.picked(&decision);
            assert_ne!(Some(prefill), refused, "seed {seed}: {decision:?}");
            assert_ne!(Some(decode), full, "seed {seed}: {decision:?}");
            // The request decided whole from its start names the same pair.
            let again = gates();
            let whole = host.decide_vetoed(
                &again,
                (seed, None),
                Constraints::default(),
                [Some(&every_prefill), None],
                UNMASKED,
            );
            assert_eq!(again.asked(), (8, 8), "the whole pass asks each place");
            assert_eq!(format!("{decision:?}"), format!("{whole:?}"), "seed {seed}");
        }
    }
}

/// An admission that refuses no place and lays one request on `worker` at
/// the first place it is asked of: a host whose answer moves a load.
#[derive(Debug)]
struct Loading {
    worker: Arc<dyn Worker>,
    asked: AtomicU32,
}

impl PairAdmission for Loading {
    fn has_capacity(&self, _: usize) -> bool {
        if self.asked.fetch_add(1, Ordering::Relaxed) == 0 {
            self.worker.increment_load();
        }
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

#[test]
fn a_pair_decided_by_its_draws_asks_its_places_ahead_of_the_rows_it_writes() {
    // The draw asks every place ahead of the two rows a side it writes;
    // the whole pass writes every row and asks behind them. So a load
    // the host lays at its first answer is read by the draw alone: the
    // prefill a seed picks at level loads loses its tie there. Decided
    // whole, no seed moves its pick.
    let host = drawing(8, 8);
    let mut moved = false;
    for seed in 0..256 {
        let level = host.decide_vetoed(
            &Counted::default(),
            (seed, None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        let (prefill, _) = host.picked(&level);
        let loading = Loading {
            worker: Arc::clone(&host.prefills[prefill]),
            asked: AtomicU32::new(0),
        };
        let drawn = host.decide_vetoed(
            &loading,
            (seed, None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        host.prefills[prefill].decrement_load();
        moved |= host.picked(&drawn).0 != prefill;
    }
    assert!(
        moved,
        "no seed moved off the prefill loaded as its gates were read"
    );
}
