use super::*;
use crate::{
    engine::view::Health,
    stages::scorers::tests::{candidate, kept},
};

#[test]
fn the_filters_read_their_bits() {
    let facts = RequestFacts::default();
    let cx = Constraints::default();
    let mut c = candidate(0, 0);
    assert!(Healthy.keep(&facts, &c, &cx));
    for down in [
        Health {
            healthy: false,
            ..c.health
        },
        Health {
            breaker_open: true,
            ..c.health
        },
        Health {
            stalled: true,
            ..c.health
        },
        Health {
            draining: true,
            ..c.health
        },
    ] {
        let unroutable = Candidate { health: down, ..c };
        assert!(!Healthy.keep(&facts, &unroutable, &cx));
    }
    assert!(NotOverloaded.keep(&facts, &c, &cx));
    c.health.overloaded = true;
    assert!(!NotOverloaded.keep(&facts, &c, &cx));
    let lifted = Constraints {
        overload_lifted: true,
        ..cx
    };
    assert!(NotOverloaded.keep(&facts, &c, &lifted));
    assert!(!HasCapacity.keep(&facts, &c, &cx));
    c.has_capacity = true;
    assert!(HasCapacity.keep(&facts, &c, &cx));
    assert!(!HasRoom.keep(&facts, &c, &cx));
    c.has_room = true;
    assert!(HasRoom.keep(&facts, &c, &cx));
    let tried = [c.id];
    let retry = RequestFacts {
        tried: &tried,
        ..facts
    };
    assert!(!NotTried.keep(&retry, &c, &cx));
    let excluded = Constraints {
        excluded: &tried,
        ..cx
    };
    assert!(!NotTried.keep(&facts, &c, &excluded));
    assert!(NotTried.soft() && HasRoom.soft() && !Healthy.soft() && !HasCapacity.soft());

    let pool = [c];
    let mut kept = vec![0];
    NotTried.retain(&facts, &pool, &cx, &mut kept);
    assert_eq!(kept, [0]);
    NotTried.retain(&facts, &pool, &excluded, &mut kept);
    assert!(kept.is_empty());
    kept.push(0);
    NotTried.retain(&retry, &pool, &cx, &mut kept);
    assert!(kept.is_empty());
}

#[test]
fn a_profiles_filters_as_one_row_test_answer_as_each_filter_asked() {
    // One row every filter keeps, then a row each refusal takes alone:
    // the four health bits, the overload bit, the capacity bit, the
    // room bit, a row the request tried, one the host excluded and one
    // the host's mask names.
    let mut pool: Vec<Candidate> = (0..11)
        .map(|position| Candidate {
            has_capacity: true,
            has_room: true,
            ..candidate(position, 0)
        })
        .collect();
    // The ready row holds the session hit: a flag of the row's word that
    // none of these filters reads.
    pool[0].session_hit = true;
    pool[1].health.healthy = false;
    pool[2].health.breaker_open = true;
    pool[3].health.stalled = true;
    pool[4].health.draining = true;
    pool[5].health.overloaded = true;
    pool[6].has_capacity = false;
    pool[7].has_room = false;
    let (tried, excluded, mask) = ([pool[8].id], [pool[9].id], [10u32]);
    let facts = RequestFacts {
        tried: &tried,
        ..RequestFacts::default()
    };
    let plain = Constraints {
        excluded: &excluded,
        ..Constraints::default()
    };
    let lifted = Constraints {
        overload_lifted: true,
        ..plain
    };
    let scopes = [
        plain,
        lifted,
        Constraints {
            stalled: &mask,
            ..plain
        },
        Constraints {
            stalled: &mask,
            ..lifted
        },
    ];
    let mut refused = [0u32; 11];
    for set in 0..32u8 {
        let filters = filters_of(set);
        let tests = RowTests::of(&filters).expect("the filters of this file fold");
        for cx in &scopes {
            let masked = tests.masked(cx);
            assert_eq!(
                masked,
                set & 1 != 0 && !cx.stalled.is_empty(),
                "set {set:#07b}"
            );
            for c in &pool {
                let each = filters.iter().all(|f| f.keep(&facts, c, cx));
                let folded = tests.keeps(&facts, c, cx) && !(masked && cx.masks(c.position));
                assert_eq!(folded, each, "set {set:#07b}, row {}, {cx:?}", c.position);
                refused[c.position as usize] += u32::from(!each);
            }
        }
    }
    // The ready row is kept under every set and scope, and each other
    // row is refused under some.
    assert_eq!(refused[0], 0);
    assert!(refused[1..].iter().all(|&times| times > 0), "{refused:?}");
}

#[test]
fn a_rows_eight_flags_are_one_word() {
    use std::{array, mem::offset_of};
    // Every flag of the health record is named: one it gains does not
    // compile here until `row_word`'s packing and the masks are looked
    // at again.
    let none = Health::default();
    let Health {
        healthy: _,
        breaker_open: _,
        stalled: _,
        draining: _,
        overloaded: _,
    } = none;
    // `row_word` is one load only while these eight bytes sit side by
    // side in the row, in the word's order.
    let at = offset_of!(Candidate, health);
    assert_eq!(
        [
            at + offset_of!(Health, healthy),
            at + offset_of!(Health, breaker_open),
            at + offset_of!(Health, stalled),
            at + offset_of!(Health, draining),
            at + offset_of!(Health, overloaded),
            offset_of!(Candidate, session_hit),
            offset_of!(Candidate, has_capacity),
            offset_of!(Candidate, has_room),
        ],
        array::from_fn::<usize, 8, _>(|i| at + i),
    );
    // And the word holds each flag in its own byte, in that order.
    let health = |health| Candidate {
        health,
        ..Candidate::default()
    };
    let alone = [
        health(Health {
            healthy: true,
            ..none
        }),
        health(Health {
            breaker_open: true,
            ..none
        }),
        health(Health {
            stalled: true,
            ..none
        }),
        health(Health {
            draining: true,
            ..none
        }),
        health(Health {
            overloaded: true,
            ..none
        }),
        Candidate {
            session_hit: true,
            ..Candidate::default()
        },
        Candidate {
            has_capacity: true,
            ..Candidate::default()
        },
        Candidate {
            has_room: true,
            ..Candidate::default()
        },
    ];
    for (place, row) in alone.iter().enumerate() {
        assert_eq!(row_word(row), 1 << (8 * place), "flag {place}");
    }
}

#[test]
fn a_filter_of_a_hosts_folds_its_profile_into_no_row_test() {
    /// A host's filter around the health filter: the same word on every
    /// row, and a type the engine does not hold.
    #[derive(Debug)]
    struct Hosted(Healthy);
    impl Filter for Hosted {
        fn name(&self) -> &'static str {
            self.0.name()
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
            self.0.keep(facts, c, cx)
        }
    }
    assert!(RowTests::of(&[]).is_some(), "no filter keeps every row");
    let mut filters = filters_of(0b10011);
    assert!(RowTests::of(&filters).is_some());
    filters.insert(1, Box::new(Hosted(Healthy)));
    assert_eq!(RowTests::of(&filters), None);
}

#[test]
fn the_healthy_filter_reads_the_hosts_mask_beside_the_rows_own_bits() {
    let facts = RequestFacts::default();
    let mut pool: Vec<Candidate> = (0..6).map(|position| candidate(position, 0)).collect();
    pool[4].health.breaker_open = true;
    let mask = [1u32, 3];
    let cx = Constraints {
        stalled: &mask,
        ..Constraints::default()
    };
    assert!(!cx.masks(0) && cx.masks(1) && cx.masks(3) && !cx.masks(4));
    // Row by row and over the whole pool, one answer: a row's own bits
    // refuse it, and so does the mask.
    let row_by_row: Vec<u32> = pool
        .iter()
        .filter(|row| Healthy.keep(&facts, row, &cx))
        .map(|row| row.position)
        .collect();
    let mut whole = kept(6);
    Healthy.retain(&facts, &pool, &cx, &mut whole);
    assert_eq!(row_by_row, [0, 2, 5]);
    assert_eq!(whole, row_by_row);
    // Under no veto the mask refuses nobody.
    let mut whole = kept(6);
    Healthy.retain(&facts, &pool, &Constraints::default(), &mut whole);
    assert_eq!(whole, [0, 1, 2, 3, 5]);
}
