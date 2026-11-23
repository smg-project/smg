//! The cases of a written worker's derive: the derivation the store's
//! writer keeps, moved by one worker at a time, answers as a whole
//! derivation of the two pools does and holds no more than the pools do.

use super::{fleet, worker, Worker, MODES};
use crate::{
    document::spec::Pairing,
    engine::view::WorkerId,
    store::pairing::{place, Derivation, Member, Numbering, PairClasses},
};

/// A pool's workers, each under its id, in id order.
type Workers = Vec<(u64, Worker)>;

fn members(pool: &Workers) -> Vec<Member<'_>> {
    pool.iter().map(|(id, worker)| worker.member(*id)).collect()
}

/// How many offers of `pool` differ.
fn offers(pool: &Workers) -> usize {
    let members = members(pool);
    let first = |&(at, member): &(usize, &Member<'_>)| {
        let differs = |before: &Member<'_>| before.offer() != member.offer();
        members[..at].iter().all(differs)
    };
    members.iter().enumerate().filter(first).count()
}

/// `worker` joins `pool` under `id`, in its place by id.
fn join(pool: &mut Workers, id: u64, worker: Worker) {
    let at = pool.partition_point(|&(held, _)| held < id);
    pool.insert(at, (id, worker));
}

/// What the kept derivation answers for one written worker, beside the
/// whole derivation of the pools as they stand: the classes, and whether
/// the last ones were carried.
fn written(
    kept: &mut Derivation,
    last: &mut PairClasses,
    pools: &[Workers; 2],
    mode: Pairing,
    worker: (Option<u64>, Option<u64>),
) -> bool {
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    let whole = PairClasses::derive(&prefills, &decodes, mode);
    let worker = (worker.0.map(WorkerId), worker.1.map(WorkerId));
    let answered = kept.written(last, prefills.as_slice(), decodes.as_slice(), mode, worker);
    let carried = answered.is_none();
    if let Some(classes) = answered {
        *last = classes;
    }
    assert_eq!(*last, whole, "carried {carried}, {mode:?}, {worker:?}");
    carried
}

#[test]
fn a_walk_of_written_workers_answers_as_the_whole_derivation_at_every_step() {
    let (mut carried, mut steps) = (0, 0);
    for mode in MODES {
        for seed in 0..30 {
            // A few offers, so that a pool's offers repeat, come and go.
            let drawn = fleet(seed, 5);
            let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let mut draw = |of: usize| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % of as u64) as usize
            };
            let mut pools: [Workers; 2] = Default::default();
            let mut kept = Derivation::default();
            let none: &[Member<'_>] = &[];
            let mut last = kept.whole(none, none, mode);
            let mut most = [0; 2];
            for _ in 0..200 {
                let (side, offer) = (draw(2), drawn[draw(5)].clone());
                // An id of its own: the next worker stands anywhere in
                // its pool.
                let free = |id: &u64| pools.iter().flatten().all(|(held, _)| held != id);
                let mut ids = (1..).map(|_| 1 + draw(400) as u64);
                let id = ids.find(free).expect("an id nobody holds");
                let held = (!pools[side].is_empty()).then(|| draw(pools[side].len()));
                let worker = match (draw(5), held) {
                    // A worker leaves.
                    (1, Some(at)) => (Some(pools[side].remove(at).0), None),
                    // A worker writes an offer again, where it stands.
                    (2, Some(at)) => {
                        pools[side][at].1 = offer;
                        (Some(pools[side][at].0), Some(pools[side][at].0))
                    }
                    // A worker crosses to the other pool.
                    (3, Some(at)) => {
                        let left = pools[side].remove(at).0;
                        join(&mut pools[1 - side], left, offer);
                        (Some(left), Some(left))
                    }
                    // A worker comes again under a new id, in either pool.
                    (4, Some(at)) => {
                        let left = pools[side].remove(at).0;
                        join(&mut pools[draw(2)], id, offer);
                        (Some(left), Some(id))
                    }
                    // A worker joins.
                    _ => {
                        join(&mut pools[side], id, offer);
                        (None, Some(id))
                    }
                };
                carried += usize::from(written(&mut kept, &mut last, &pools, mode, worker));
                steps += 1;
                // What is kept is bounded by the pools: a slot a distinct
                // offer, and a verdict a pair of slots.
                for side in 0..2 {
                    most[side] = most[side].max(offers(&pools[side]));
                    assert!(kept.sides[side].slots.len() <= most[side], "{most:?}");
                    assert_eq!(kept.sides[side].order.len(), offers(&pools[side]));
                }
                assert!(kept.verdicts.len() <= most[0], "{most:?}");
                let widest = kept.verdicts.iter().map(Vec::len).max();
                assert!(widest.unwrap_or(0) <= most[1], "{most:?}");
            }
        }
    }
    // The walk reaches both answers: the last classes carried, and new ones.
    assert!(carried > 0 && carried < steps, "{carried} of {steps}");
}

#[test]
fn a_write_that_names_the_wrong_worker_is_answered_by_the_whole_derivation() {
    let drawn = fleet(7, 4);
    let mode = Pairing::Lenient;
    let mut pools: [Workers; 2] = [
        vec![(1, drawn[0].clone()), (2, drawn[1].clone())],
        vec![(11, drawn[2].clone())],
    ];
    let mut kept = Derivation::default();
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    let mut last = kept.whole(prefills.as_slice(), decodes.as_slice(), mode);
    // A decode joins and the write names nobody; then a prefill leaves and
    // the write names the decode that stayed.
    join(&mut pools[1], 12, drawn[3].clone());
    assert!(!written(&mut kept, &mut last, &pools, mode, (None, None)));
    pools[0].remove(0);
    assert!(!written(
        &mut kept,
        &mut last,
        &pools,
        mode,
        (Some(12), Some(12))
    ));
    // What is kept was made again from the pools: the next write, named
    // as it is, answers for its worker alone.
    pools[1].remove(0);
    written(&mut kept, &mut last, &pools, mode, (Some(11), None));
    assert_eq!(kept.sides[1].of.len(), 1);
}

#[test]
fn classes_of_other_pools_handed_as_the_last_are_answered_by_the_whole_derivation() {
    let mode = Pairing::Lenient;
    let pools: [Workers; 2] = [
        vec![(1, worker("rt-a")), (2, worker("rt-b"))],
        vec![(11, worker("rt-a"))],
    ];
    let mut kept = Derivation::default();
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    kept.whole(prefills.as_slice(), decodes.as_slice(), mode);
    // The last classes are of one prefill more than what is kept holds,
    // and the write names nobody: the pools' counts fit what is kept and
    // the last lists' do not.
    let mut other = pools.clone();
    join(&mut other[0], 3, worker("rt-b"));
    let mut last = PairClasses::derive(&members(&other[0]), &members(&other[1]), mode);
    assert!(!written(&mut kept, &mut last, &pools, mode, (None, None)));
}

#[test]
fn a_worker_that_moves_between_two_offers_of_one_class_is_carried() {
    // Two prefill offers that differ in a version, which lenient does not
    // read: one class. The last prefill writes the other offer and its own
    // again: no offer comes or goes, no first worker changes, and it stands
    // in the class it had.
    let mode = Pairing::Lenient;
    let (one, other) = (worker("rt-a").version("1"), worker("rt-a").version("2"));
    let mut pools: [Workers; 2] = [
        vec![(1, one.clone()), (2, other.clone()), (3, one.clone())],
        vec![(11, worker("rt-a"))],
    ];
    let mut kept = Derivation::default();
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    let mut last = kept.whole(prefills.as_slice(), decodes.as_slice(), mode);
    assert_eq!(last.classes(), 1);
    for offer in [other, one] {
        pools[0][2].1 = offer;
        assert!(written(
            &mut kept,
            &mut last,
            &pools,
            mode,
            (Some(3), Some(3))
        ));
    }
}

#[test]
fn a_whole_derivation_forgets_the_slots_the_last_left_free() {
    let mode = Pairing::Lenient;
    let mut pools: [Workers; 2] = [
        vec![(1, worker("rt-a")), (2, worker("rt-b"))],
        vec![(11, worker("rt-a"))],
    ];
    let mut kept = Derivation::default();
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    let mut last = kept.whole(prefills.as_slice(), decodes.as_slice(), mode);
    // The one prefill of an offer leaves: its slot is free.
    pools[0].remove(1);
    written(&mut kept, &mut last, &pools, mode, (Some(2), None));
    assert_eq!(kept.sides[0].free.len(), 1);
    // A whole derivation numbers the offers anew and holds no free slot:
    // the next new offer takes a slot past the held ones.
    let (prefills, decodes) = (members(&pools[0]), members(&pools[1]));
    last = kept.whole(prefills.as_slice(), decodes.as_slice(), mode);
    assert!(kept.sides[0].free.is_empty());
    join(&mut pools[0], 3, worker("rt-c"));
    written(&mut kept, &mut last, &pools, mode, (None, Some(3)));
    assert_eq!(kept.sides[0].slots.len(), 2);
}

#[test]
fn a_workers_place_is_found_at_its_id_and_at_no_other() {
    let drawn = fleet(3, 1);
    for count in 0..9_u64 {
        // The ids 10, 20 and so on, in order.
        let pool: Workers = (1..=count).map(|n| (10 * n, drawn[0].clone())).collect();
        let members = members(&pool);
        for id in 0..10 * count + 10 {
            let stands =
                (id % 10 == 0 && id > 0 && id <= 10 * count).then(|| (id / 10 - 1) as usize);
            assert_eq!(
                place(members.as_slice(), WorkerId(id)),
                stands,
                "{count} workers, id {id}"
            );
        }
    }
}

#[test]
fn two_things_of_one_hash_take_two_numbers() {
    let mut numbering = Numbering::default();
    let mut known: Vec<&str> = Vec::new();
    let mut numbers = Vec::new();
    for thing in ["a", "b", "a", "c", "b", "c"] {
        // Every thing under one hash: the compare tells them apart.
        let same = |number: u32| known[number as usize] == thing;
        let (number, new) = numbering.number(7, known.len() as u32, same);
        if new {
            known.push(thing);
        }
        numbers.push(number);
    }
    assert_eq!(known, ["a", "b", "c"]);
    assert_eq!(numbers, [0, 1, 0, 2, 1, 2]);
    numbering.clear();
    assert_eq!(numbering.number(7, 0, |_| false), (0, true));
}
