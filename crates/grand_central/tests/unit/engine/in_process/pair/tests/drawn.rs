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

/// An admission that counts what it is asked: each prefill's capacity bit,
/// and the one question of every place. The prefill at `refused` has no
/// capacity.
#[derive(Debug, Default)]
struct Counted {
    places: AtomicU32,
    all: AtomicU32,
    refused: Option<usize>,
}

impl Counted {
    /// The capacity bits asked, and the questions of every place.
    fn asked(&self) -> (u32, u32) {
        (
            self.places.load(Ordering::Relaxed),
            self.all.load(Ordering::Relaxed),
        )
    }
}

impl PairAdmission for Counted {
    fn has_capacity(&self, prefill: usize) -> bool {
        self.places.fetch_add(1, Ordering::Relaxed);
        self.refused != Some(prefill)
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }

    fn admits_all(&self, _: usize, _: usize) -> bool {
        self.all.fetch_add(1, Ordering::Relaxed);
        self.refused.is_none()
    }
}

#[test]
fn an_admission_that_takes_the_default_answers_what_the_walk_answers() {
    // `Gates` states its two reads alone: the prefill at place 2 has no
    // capacity and the decode at place 1 no room.
    let gates = Gates(|prefill| prefill != 2, |decode| decode != 1);
    assert!(gates.admits_all(0, 0));
    assert!(gates.admits_all(2, 1));
    assert!(!gates.admits_all(3, 1), "the prefill at place 2 is refused");
    assert!(!gates.admits_all(2, 2), "the decode at place 1 is refused");
    assert!(!gates.admits_all(8, 8));
    assert!(OPEN.admits_all(8, 8));
}

#[test]
fn every_row_pairs_and_is_admitted_under_one_class_a_side_alone() {
    let all = every(8, 8);
    assert!(one_pair(&all, &OPEN, (8, 8)));
    // A list longer than its classes holds a row of no class.
    assert!(!one_pair(&all, &OPEN, (9, 8)));
    assert!(!one_pair(&all, &OPEN, (8, 9)));
    // One prefill with no capacity, one decode with no room.
    assert!(!one_pair(&all, &Gates(|at| at != 3, |_| true), (8, 8)));
    assert!(!one_pair(&all, &Gates(|_| true, |at| at != 5), (8, 8)));
    // The second prefill pairs with the first decode alone: two classes.
    let narrowed = PartnerClasses::number(2, [&[0, 1][..], &[0][..]]);
    assert!(!one_pair(&narrowed, &OPEN, (2, 2)));
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
        // The draw: the one question of every place, and the capacity
        // bit of the prefills it asks about alone.
        let counted = Counted::default();
        let drawn = host.decide_vetoed(
            &counted,
            (seed, None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        let (places, all) = counted.asked();
        assert_eq!(all, 1, "the draw is asked over {prefills} and {decodes}");
        assert!(places <= 2, "{places} capacity bits read under the draw");
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
        let (places, all) = counted.asked();
        assert_eq!(all, 0, "the whole pass asks each place by itself");
        assert!(places as usize >= prefills);
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
    let whole = |decision: Decision, counted: &Counted| {
        let (places, all) = counted.asked();
        assert!(decision.pick().is_some(), "{decision:?}");
        assert!(places >= 7, "{places} capacity bits read by the whole pass");
        all
    };
    // A retry: the host names a worker it just failed to reach.
    let counted = Counted::default();
    let failed = [worker_id(host.decodes[0].as_ref())];
    let decision = host.decide_excluding(&counted, 5, None, &failed);
    assert_eq!(whole(decision, &counted), 0);
    // The host holds the request to places of its decode list.
    let counted = Counted::default();
    let every_prefill = [0, 1, 2, 3, 4, 5, 6, 7];
    let places = [Some(&every_prefill[..]), Some(&[1, 2, 3][..])];
    let decision = host.decide_among(&counted, 5, None, Constraints::default(), places);
    assert_eq!(whole(decision, &counted), 0);
    // A prefill with no capacity: the question of every place says no,
    // and the pair picks among the other prefills.
    let counted = Counted {
        refused: Some(3),
        ..Counted::default()
    };
    let decision = host.decide(&counted, 5, None);
    assert_ne!(host.picked(&decision).0, 3);
    assert_eq!(whole(decision, &counted), 1);
    // A prefill list under the floor of a pair's pick of two.
    let small = drawing(7, 8);
    let counted = Counted::default();
    let decision = small.decide(&counted, 5, None);
    assert_eq!(whole(decision, &counted), 0);
    // A document whose sides do not pick of two.
    let turns = Host::under(
        (presets("round_robin", "round_robin"), Arc::default()),
        8,
        8,
    );
    let counted = Counted::default();
    let decision = turns.decide(&counted, 5, None);
    assert_eq!(whole(decision, &counted), 0);
}

/// An admission whose prefill filled behind its answer to the question of
/// every place: that question says yes, and the prefill's own bit says no.
#[derive(Debug)]
struct Filled(Counted);

impl PairAdmission for Filled {
    fn has_capacity(&self, prefill: usize) -> bool {
        self.0.has_capacity(prefill)
    }

    fn has_room(&self, decode: usize) -> bool {
        self.0.has_room(decode)
    }

    fn admits_all(&self, prefills: usize, decodes: usize) -> bool {
        self.0.admits_all(prefills, decodes);
        true
    }
}

#[test]
fn a_draw_that_lands_on_a_refused_row_is_decided_whole() {
    let host = drawing(8, 8);
    let every_prefill: Vec<u32> = (0..8).collect();
    let filled = || {
        Filled(Counted {
            refused: Some(3),
            ..Counted::default()
        })
    };
    let (mut drawn, mut refused) = (0, 0);
    for seed in 0..256 {
        // The draw is asked. Where it lands on the prefill with no
        // capacity the pair is decided whole: every capacity bit is read.
        let asked = filled();
        let decision = host.decide_vetoed(
            &asked,
            (seed, None),
            Constraints::default(),
            [None, None],
            UNMASKED,
        );
        let (places, all) = asked.0.asked();
        assert_eq!(all, 1, "seed {seed}: the draw is asked");
        assert!(decision.pick().is_some(), "seed {seed}: {decision:?}");
        assert_ne!(host.picked(&decision).0, 3, "seed {seed}: {decision:?}");
        if places <= 2 {
            drawn += 1;
            continue;
        }
        refused += 1;
        assert!(places >= 8, "seed {seed}: {places} capacity bits read");
        // The request decided whole from its start names the same pair.
        let again = filled();
        let whole = host.decide_vetoed(
            &again,
            (seed, None),
            Constraints::default(),
            [Some(&every_prefill), None],
            UNMASKED,
        );
        assert_eq!(again.0.asked().1, 0, "the whole pass asks each place");
        assert_eq!(format!("{decision:?}"), format!("{whole:?}"), "seed {seed}");
    }
    assert!(drawn > 0 && refused > 0, "{drawn} drawn, {refused} refused");
}
