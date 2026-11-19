//! The predictor's cases: what a read finds, what a write adds, what the
//! bound takes out and in which order, and what is left of a worker that
//! left. Two cases run drawn requests against a plain record of them.

use std::collections::{BTreeSet, HashMap};

use super::*;

/// The namespace of the cases' requests.
const SEED: u64 = 7;

fn worker(id: u64) -> WorkerId {
    WorkerId(id)
}

/// The request of `head`, then `tail`.
fn request(head: &[u64], tail: &[u64]) -> Vec<u64> {
    [head, tail].concat()
}

/// What a read finds: how deep, and the holders' ids ascending.
fn read(predictor: &mut Predictor, hashes: &[u64]) -> (u32, Vec<u64>) {
    let found = predictor.deepest(SEED, hashes);
    let mut holders: Vec<u64> = found.holders().map(|worker| worker.0).collect();
    holders.sort_unstable();
    (found.blocks(), holders)
}

impl Predictor {
    /// The rules every state keeps: the counts are the entries', a block's
    /// followers are the blocks behind it, a worker of the pool that was
    /// sent a block was sent the block before it, every block that none
    /// follows is on the list, and the list reads the same both ways.
    fn check(&self) {
        let mut followers: HashMap<u32, u32> = HashMap::new();
        let (mut placements, mut blocks, mut listed) = (0, 0, 0);
        for at in 0..self.span {
            let block = self.block(at);
            if block.after == FREE {
                assert_eq!((block.older, block.newer), (UNLISTED, UNLISTED), "{at}");
                continue;
            }
            blocks += 1;
            listed += usize::from(block.older != UNLISTED);
            assert_eq!(self.find(block.key), Some(at), "{at}");
            if block.group != NONE {
                assert_eq!(block.lone, NOBODY, "{at}");
                assert!(!self.groups[block.group as usize].is_empty(), "{at}");
            }
            let holders = self.holders(block);
            placements += holders.iter().filter(|&&place| place != NOBODY).count();
            if block.after == 0 {
                assert_ne!(block.older, UNLISTED, "an end off the list: {at}");
            }
            if block.before != NONE {
                let before = self.block(block.before);
                assert_ne!(before.after, FREE, "{at}");
                *followers.entry(block.before).or_default() += 1;
                for &place in holders {
                    if place != NOBODY && self.is_live(place) {
                        assert!(self.holders(before).contains(&place), "{at}");
                    }
                }
            }
        }
        for at in 0..self.span {
            let block = self.block(at);
            if block.after != FREE {
                assert_eq!(
                    block.after,
                    followers.get(&at).copied().unwrap_or(0),
                    "{at}"
                );
            }
        }
        assert_eq!((placements, blocks), (self.placements, self.blocks));
        assert_eq!(self.maps.iter().map(|map| map.len()).sum::<usize>(), blocks);
        let (mut at, mut older, mut walked) = (self.oldest, NONE, 0);
        while at != NONE {
            let block = self.block(at);
            assert_ne!(block.after, FREE, "{at}");
            assert_eq!(block.older, older, "{at}");
            (older, at, walked) = (at, block.newer, walked + 1);
        }
        assert_eq!((older, walked), (self.newest, listed));
    }
}

#[test]
fn a_request_sent_before_reads_whole_and_adds_nothing() {
    let mut predictor = Predictor::new(1_000);
    let hashes = [11, 12, 13, 14];
    assert_eq!(read(&mut predictor, &hashes), (0, vec![]));
    predictor.sent(SEED, &hashes, worker(1));
    assert_eq!(read(&mut predictor, &hashes), (4, vec![1]));
    assert_eq!((predictor.len(), predictor.blocks()), (4, 4));
    predictor.sent(SEED, &hashes, worker(1));
    assert_eq!(
        (predictor.len(), predictor.blocks(), predictor.dropped()),
        (4, 4, 0)
    );
    predictor.sent(SEED, &[], worker(1));
    assert_eq!(read(&mut predictor, &[]), (0, vec![]));
    predictor.check();
}

#[test]
fn a_request_grown_by_a_turn_reads_its_old_depth_and_adds_its_new_blocks() {
    let mut predictor = Predictor::new(1_000);
    let turn = [11, 12, 13, 14];
    let next = [11, 12, 13, 14, 15, 16];
    predictor.sent(SEED, &turn, worker(1));
    assert_eq!(read(&mut predictor, &next), (4, vec![1]));
    predictor.sent(SEED, &next, worker(1));
    assert_eq!(read(&mut predictor, &next), (6, vec![1]));
    assert_eq!((predictor.len(), predictor.blocks()), (6, 6));
    // A leading part of what was sent reads whole, and a request that
    // parts ways reads what it shares.
    assert_eq!(read(&mut predictor, &turn), (4, vec![1]));
    assert_eq!(read(&mut predictor, &[11, 12, 99, 14]), (2, vec![1]));
    assert_eq!(read(&mut predictor, &[99, 12, 13, 14]), (0, vec![]));
    // The same blocks at other places are another prefix.
    assert_eq!(read(&mut predictor, &[12, 11]), (0, vec![]));
    predictor.check();
}

#[test]
fn the_deepest_group_alone_is_read() {
    let mut predictor = Predictor::new(1_000);
    let long: Vec<u64> = (1..=10).collect();
    predictor.sent(SEED, &long, worker(1));
    predictor.sent(SEED, &long[..5], worker(2));
    assert_eq!(read(&mut predictor, &long), (10, vec![1]));
    assert_eq!(read(&mut predictor, &long[..5]), (5, vec![1, 2]));
    assert_eq!(
        read(&mut predictor, &request(&long[..5], &[77])),
        (5, vec![1, 2])
    );
    assert_eq!(
        read(&mut predictor, &request(&long[..7], &[77])),
        (7, vec![1])
    );
    assert_eq!((predictor.len(), predictor.blocks()), (15, 10));
    predictor.check();
}

#[test]
fn a_worker_is_added_past_what_it_was_sent_alone() {
    let mut predictor = Predictor::new(1_000);
    let long: Vec<u64> = (1..=10).collect();
    predictor.sent(SEED, &long, worker(1));
    predictor.sent(SEED, &long[..3], worker(2));
    assert_eq!((predictor.len(), predictor.blocks()), (13, 10));
    predictor.sent(SEED, &long[..8], worker(2));
    assert_eq!((predictor.len(), predictor.blocks()), (18, 10));
    let longer = request(&long, &[11, 12]);
    predictor.sent(SEED, &longer, worker(2));
    assert_eq!((predictor.len(), predictor.blocks()), (22, 12));
    assert_eq!(read(&mut predictor, &longer), (12, vec![2]));
    assert_eq!(read(&mut predictor, &long), (10, vec![1, 2]));
    predictor.check();
}

#[test]
fn a_namespace_keeps_the_same_blocks_apart() {
    let mut predictor = Predictor::new(1_000);
    let hashes = [11, 12, 13, 14];
    predictor.sent(1, &hashes, worker(1));
    assert_eq!(predictor.deepest(2, &hashes).blocks(), 0);
    assert_eq!(predictor.deepest(1, &hashes).blocks(), 4);
    predictor.sent(2, &hashes, worker(2));
    assert_eq!(predictor.blocks(), 8);
    let of_two: Vec<u64> = predictor
        .deepest(2, &hashes)
        .holders()
        .map(|w| w.0)
        .collect();
    assert_eq!(of_two, [2]);
    predictor.check();
}

/// Three paths behind one head, sent in their order to one worker: five
/// placements, then three, then three.
fn three_paths(bound: usize) -> (Predictor, [Vec<u64>; 3]) {
    let mut predictor = Predictor::new(bound);
    let paths = [31, 41, 51].map(|tail| vec![1, 2, tail, tail + 1, tail + 2]);
    for path in &paths {
        predictor.sent(SEED, path, worker(1));
    }
    assert_eq!((predictor.len(), predictor.dropped()), (11, 0));
    (predictor, paths)
}

#[test]
fn the_oldest_path_leaves_first_from_its_deep_end() {
    let (mut predictor, [first, second, third]) = three_paths(11);
    // A fourth path needs three placements: the first path's three blocks
    // behind the head leave, and the head stays for the paths behind it.
    let fourth = [1, 2, 61, 62, 63];
    predictor.sent(SEED, &fourth, worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (11, 0));
    assert_eq!(read(&mut predictor, &first), (2, vec![1]));
    for path in [&second[..], &third[..], &fourth[..]] {
        assert_eq!(read(&mut predictor, path), (5, vec![1]), "{path:?}");
    }
    // One placement more takes the next oldest path's deepest block alone.
    predictor.sent(SEED, &[1, 2, 71], worker(1));
    assert_eq!(read(&mut predictor, &second), (4, vec![1]));
    assert_eq!(read(&mut predictor, &third), (5, vec![1]));
    assert_eq!(predictor.len(), 11);
    predictor.check();
}

#[test]
fn a_path_sent_again_is_the_newest() {
    let (mut predictor, [first, second, third]) = three_paths(11);
    predictor.sent(SEED, &first, worker(1));
    predictor.sent(SEED, &[1, 2, 61, 62, 63], worker(1));
    assert_eq!(read(&mut predictor, &first), (5, vec![1]));
    assert_eq!(read(&mut predictor, &second), (2, vec![1]));
    assert_eq!(read(&mut predictor, &third), (5, vec![1]));
    predictor.check();
}

#[test]
fn a_block_a_later_request_ended_at_outlives_the_older_paths_behind_it() {
    let mut predictor = Predictor::new(6);
    let (first, second) = ([1, 2, 31, 32], [1, 2, 41, 42]);
    predictor.sent(SEED, &first, worker(1));
    predictor.sent(SEED, &second, worker(1));
    // A request of the head alone ends inside both paths, after them.
    predictor.sent(SEED, &[1, 2], worker(1));
    assert_eq!(predictor.len(), 6);
    // Four placements for a path of its own: both tails leave, the oldest
    // first, and the head stays whole, the newest of what is left.
    let apart = [91, 92, 93, 94];
    predictor.sent(SEED, &apart, worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (6, 0));
    assert_eq!(read(&mut predictor, &first), (2, vec![1]));
    assert_eq!(read(&mut predictor, &second), (2, vec![1]));
    assert_eq!(read(&mut predictor, &apart), (4, vec![1]));
    // The head is the oldest now: its deep end leaves next.
    predictor.sent(SEED, &[81], worker(1));
    assert_eq!(read(&mut predictor, &[1, 2]), (1, vec![1]));
    assert_eq!(read(&mut predictor, &apart), (4, vec![1]));
    predictor.check();
}

#[test]
fn a_write_past_the_bound_is_cut_at_its_deep_end_and_counted() {
    let mut predictor = Predictor::new(3);
    let hashes = [11, 12, 13, 14, 15];
    predictor.sent(SEED, &hashes, worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (3, 2));
    assert_eq!(read(&mut predictor, &hashes), (3, vec![1]));
    // Again: nothing but the request's own blocks could leave for it.
    predictor.sent(SEED, &hashes, worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (3, 4));
    assert_eq!(read(&mut predictor, &hashes), (3, vec![1]));
    predictor.check();

    let mut none = Predictor::new(0);
    none.sent(SEED, &hashes, worker(1));
    assert_eq!((none.len(), none.blocks(), none.dropped()), (0, 0, 5));
    none.check();
}

#[test]
fn a_request_takes_room_from_older_paths_and_never_from_itself() {
    let mut predictor = Predictor::new(6);
    let (old, turn) = ([9, 8], [1, 2, 3, 4]);
    // The request's own path was sent first: its next turn makes it the
    // newest, and the path sent after it is the one that leaves.
    predictor.sent(SEED, &turn, worker(1));
    predictor.sent(SEED, &old, worker(1));
    let next = [1, 2, 3, 4, 5, 6];
    predictor.sent(SEED, &next, worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (6, 0));
    assert_eq!(read(&mut predictor, &next), (6, vec![1]));
    assert_eq!(read(&mut predictor, &old), (0, vec![]));
    // Nothing older is left: the next turn's blocks are dropped, and the
    // request keeps what it had.
    predictor.sent(SEED, &request(&next, &[7, 8]), worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (6, 2));
    assert_eq!(read(&mut predictor, &request(&next, &[7, 8])), (6, vec![1]));
    predictor.check();
}

#[test]
fn a_worker_that_left_is_not_read_and_a_shallower_holder_is() {
    let mut predictor = Predictor::new(1_000);
    let long: Vec<u64> = (1..=10).collect();
    predictor.sent(SEED, &long, worker(1));
    predictor.sent(SEED, &long[..5], worker(2));
    predictor.forget(worker(1));
    predictor.forget(worker(9));
    // The read skips the worker at once; its placements wait for the sweep.
    assert_eq!(read(&mut predictor, &long), (5, vec![2]));
    assert_eq!((predictor.len(), predictor.blocks()), (15, 10));
    predictor.check();
    // Until a whole pass has run, its place is no other worker's: a new
    // worker is not read as sent what the worker that left was.
    predictor.sent(SEED, &[60], worker(4));
    assert_eq!(read(&mut predictor, &long), (5, vec![2]));
    predictor.forget(worker(4));
    assert_eq!(predictor.sweep(4), 4);
    assert_eq!(read(&mut predictor, &long), (5, vec![2]));
    // The first pass's six entries left, then a second pass over all
    // eleven, for the worker that left meanwhile.
    assert_eq!(predictor.sweep(usize::MAX), 6 + 11);
    assert_eq!((predictor.len(), predictor.blocks()), (5, 5));
    assert_eq!(predictor.sweep(usize::MAX), 0);
    predictor.check();
    // The places of the two workers that left are other workers' now, and
    // none of their placements came with them.
    predictor.sent(SEED, &[50, 51], worker(3));
    predictor.sent(SEED, &[52], worker(5));
    assert_eq!(
        (predictor.places[&worker(3)], predictor.places[&worker(5)]),
        (2, 0)
    );
    assert_eq!(read(&mut predictor, &long), (5, vec![2]));
    assert_eq!(read(&mut predictor, &[50, 51]), (2, vec![3]));
    predictor.check();
}

#[test]
fn a_worker_back_under_its_old_id_starts_with_nothing() {
    let mut predictor = Predictor::new(1_000);
    let long: Vec<u64> = (1..=10).collect();
    predictor.sent(SEED, &long, worker(1));
    predictor.forget(worker(1));
    predictor.sent(SEED, &long[..2], worker(1));
    assert_eq!(read(&mut predictor, &long), (2, vec![1]));
    predictor.check();
    predictor.sweep(usize::MAX);
    assert_eq!((predictor.len(), predictor.blocks()), (2, 2));
    assert_eq!(read(&mut predictor, &long), (2, vec![1]));
    predictor.check();
}

#[test]
fn a_block_holds_as_many_workers_as_were_sent_it() {
    let mut predictor = Predictor::new(1_000);
    let head = [1, 2, 3];
    for id in 1..=6 {
        predictor.sent(SEED, &head, worker(id));
    }
    assert_eq!(read(&mut predictor, &head), (3, vec![1, 2, 3, 4, 5, 6]));
    assert_eq!((predictor.len(), predictor.blocks()), (18, 3));
    predictor.forget(worker(2));
    predictor.forget(worker(5));
    assert_eq!(read(&mut predictor, &head), (3, vec![1, 3, 4, 6]));
    predictor.sweep(usize::MAX);
    assert_eq!((predictor.len(), predictor.blocks()), (12, 3));
    predictor.check();
    for id in [1, 3, 4, 6] {
        predictor.forget(worker(id));
    }
    assert_eq!(read(&mut predictor, &head), (0, vec![]));
    predictor.sweep(usize::MAX);
    assert_eq!((predictor.len(), predictor.blocks()), (0, 0));
    predictor.check();
    // What the sweep freed is used again: entries, groups and places.
    for id in 11..=13 {
        predictor.sent(SEED, &head, worker(id));
    }
    assert_eq!(read(&mut predictor, &head), (3, vec![11, 12, 13]));
    assert_eq!((predictor.span, predictor.groups.len()), (3, 3));
    assert!(predictor.workers.len() <= 6);
    predictor.check();
}

#[test]
fn a_lowered_bound_is_met_by_the_sweep_a_bounded_count_a_call() {
    let mut predictor = Predictor::new(1_000);
    let paths: Vec<Vec<u64>> = (1..=5u64)
        .map(|path| (0..4).map(|block| path * 10 + block).collect())
        .collect();
    for path in &paths {
        predictor.sent(SEED, path, worker(1));
    }
    predictor.set_bound(8);
    assert_eq!((predictor.bound(), predictor.len()), (8, 20));
    assert_eq!(predictor.sweep(3), 3);
    assert_eq!(predictor.len(), 17);
    assert_eq!(predictor.sweep(100), 9);
    assert_eq!(predictor.len(), 8);
    for (place, path) in paths.iter().enumerate() {
        let kept = if place < 3 { (0, vec![]) } else { (4, vec![1]) };
        assert_eq!(read(&mut predictor, path), kept, "{path:?}");
    }
    predictor.check();
}

#[test]
fn a_worker_past_the_places_a_block_can_name_is_dropped_and_counted() {
    let mut predictor = Predictor::new(usize::MAX);
    let most = u64::from(NOBODY);
    for id in 0..most {
        predictor.sent(SEED, &[id], worker(id));
    }
    assert_eq!((predictor.len() as u64, predictor.dropped()), (most, 0));
    predictor.sent(SEED, &[most, most], worker(most));
    assert_eq!((predictor.len() as u64, predictor.dropped()), (most, 2));
    assert_eq!(read(&mut predictor, &[most, most]), (0, vec![]));
    // A worker that leaves makes the place free once a pass has run.
    predictor.forget(worker(0));
    predictor.sweep(usize::MAX);
    predictor.sent(SEED, &[most, most], worker(most));
    assert_eq!(read(&mut predictor, &[most, most]), (2, vec![most]));
}

#[test]
fn a_block_that_gains_a_follower_is_an_end_no_longer() {
    let mut predictor = Predictor::new(1_000);
    let path: Vec<u64> = (1..=6).collect();
    for turn in [2, 4, 6] {
        predictor.sent(SEED, &path[..turn], worker(1));
    }
    // The two earlier turns' ends left the list as they were grown past:
    // the sweep takes a step a block, none to pass a block over.
    predictor.set_bound(0);
    assert_eq!(predictor.sweep(usize::MAX), 6);
    assert_eq!((predictor.len(), predictor.blocks()), (0, 0));
    predictor.check();
}

#[test]
fn a_block_no_worker_is_left_of_stays_while_a_write_stands_on_it() {
    let mut predictor = Predictor::new(1_000);
    predictor.sent(SEED, &[1, 2, 3], worker(1));
    predictor.sent(SEED, &[1, 2, 3, 4], worker(1));
    predictor.sent(SEED, &[1, 2], worker(2));
    // The first worker leaves, and the pass stops before its last block:
    // the third block has no worker, the fourth still the one that left.
    predictor.forget(worker(1));
    assert_eq!(predictor.sweep(3), 3);
    assert_eq!((predictor.len(), predictor.blocks()), (3, 4));
    // A write that stands on the third block takes the fourth for room.
    // The third stays, with the path behind it, and the write is made as
    // far as the room goes.
    predictor.set_bound(3);
    predictor.sent(SEED, &[1, 2, 3, 5, 6], worker(2));
    assert_eq!((predictor.len(), predictor.dropped()), (3, 2));
    assert_eq!(read(&mut predictor, &[1, 2, 3, 5, 6]), (3, vec![2]));
    predictor.check();
}

#[test]
fn a_write_takes_a_bounded_count_of_steps_at_the_old_end() {
    let mut predictor = Predictor::new(11);
    let path: Vec<u64> = (1..=11).collect();
    predictor.sent(SEED, &path, worker(1));
    // Ten requests end inside the path, then the path again: ten blocks
    // that others follow stand older on the list than the path's end.
    for blocks in 1..=10 {
        predictor.sent(SEED, &path[..blocks], worker(1));
    }
    predictor.sent(SEED, &path, worker(1));
    // A write of one block takes ten steps and no more: each passes one
    // of the ten over, none makes room, and the block is dropped.
    predictor.sent(SEED, &[99], worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (11, 1));
    assert_eq!(read(&mut predictor, &[99]), (0, vec![]));
    assert_eq!(predictor.oldest, 10, "the path's end, its eleventh entry");
    // The next write finds the path's end the oldest, and takes it.
    predictor.sent(SEED, &[98], worker(1));
    assert_eq!((predictor.len(), predictor.dropped()), (11, 1));
    assert_eq!(read(&mut predictor, &[98]), (1, vec![1]));
    assert_eq!(read(&mut predictor, &path), (10, vec![1]));
    predictor.check();
}

/// The cases' draws: the same requests on every run.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    /// A request of `most` blocks at most, each one of three: requests
    /// share their heads often, and a block's content stands at many
    /// places.
    fn request(&mut self, most: u64) -> Vec<u64> {
        let blocks = 1 + self.below(most);
        (0..blocks).map(|_| 1 + self.below(3)).collect()
    }
}

/// A plain record of what was sent: every request with its worker, a
/// worker's requests dropped where it left.
#[derive(Default)]
struct Record {
    sent: Vec<(Vec<u64>, u64)>,
}

impl Record {
    /// How many leading blocks of `request` each worker was sent.
    fn depths(&self, request: &[u64]) -> HashMap<u64, u32> {
        let mut depths: HashMap<u64, u32> = HashMap::new();
        for (sent, worker) in &self.sent {
            let shared = sent.iter().zip(request).take_while(|(a, b)| a == b).count() as u32;
            let depth = depths.entry(*worker).or_default();
            *depth = (*depth).max(shared);
        }
        depths
    }

    /// What a read of `request` finds where nothing was forgotten.
    fn read(&self, request: &[u64]) -> (u32, Vec<u64>) {
        let depths = self.depths(request);
        let deepest = depths.values().copied().max().unwrap_or(0);
        if deepest == 0 {
            return (0, vec![]);
        }
        let holders: BTreeSet<u64> = depths
            .iter()
            .filter(|&(_, &depth)| depth == deepest)
            .map(|(&worker, _)| worker)
            .collect();
        (deepest, holders.into_iter().collect())
    }
}

/// Drawn writes, leavers, sweeps and reads against the record. Under
/// `bound` the predictor forgets, so a read is checked for what it may
/// say: no deeper than the record, and only workers sent that deep.
fn drawn(seed: u64, bound: Option<usize>, steps: usize) {
    let mut draws = Draws(seed);
    let mut predictor = Predictor::new(bound.unwrap_or(usize::MAX));
    let mut record = Record::default();
    for step in 0..steps {
        match draws.below(10) {
            0 => {
                let worker_id = 1 + draws.below(6);
                predictor.forget(worker(worker_id));
                record.sent.retain(|(_, of)| *of != worker_id);
            }
            1 => {
                predictor.sweep(draws.below(12) as usize);
            }
            _ => {
                let (hashes, worker_id) = (draws.request(12), 1 + draws.below(6));
                predictor.sent(SEED, &hashes, worker(worker_id));
                record.sent.push((hashes, worker_id));
            }
        }
        let asked = draws.request(14);
        let found = read(&mut predictor, &asked);
        match bound {
            None => assert_eq!(found, record.read(&asked), "step {step}: {asked:?}"),
            Some(bound) => {
                assert!(predictor.len() <= bound, "step {step}");
                let depths = record.depths(&asked);
                for holder in &found.1 {
                    assert!(depths[holder] >= found.0, "step {step}: {asked:?}");
                }
                assert!(found.0 <= record.read(&asked).0, "step {step}: {asked:?}");
            }
        }
        if step % 16 == 0 {
            predictor.check();
        }
    }
    predictor.check();
}

#[test]
fn drawn_requests_read_as_a_plain_record_of_them_reads() {
    for seed in [1, 0x9E37_79B9, 0xDEAD_BEEF_CAFE] {
        drawn(seed, None, 3_000);
    }
}

#[test]
fn drawn_requests_under_a_small_bound_keep_every_rule() {
    for (seed, bound) in [(3, 40), (0x51_7C_C1_B7, 9), (0xABCD_EF01, 150)] {
        drawn(seed, Some(bound), 3_000);
    }
}
