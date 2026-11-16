//! The consistent-hash scorer and the ring it owns (design 4.4): the ring
//! owner of a decision's key scores first and the ring's next members after
//! it, so a commit that misses the owner falls to the member the key would
//! walk to. The ring is the engine's. Its members are the worker ids the
//! membership feed names (added, replaced, a snapshot) and, until every host
//! path feeds additions, the ids a decision's candidates carry: a candidate
//! the ring does not know joins at its first decision. Each member stands at
//! [`VIRTUAL_NODES`] positions hashed from the worker's stable key (the
//! view's [`Candidate::key`], the registration key's fingerprint, equal
//! across the handles of one worker and across gateway replicas) and is
//! indexed by the handle's id: a member that leaves takes its positions with
//! it and nothing else moves, a replaced handle rejoins from the view at its
//! predecessor's positions, and every replica places a key alike. The
//! lifecycle prunes (removed, replaced, a snapshot); positions come from the
//! view, which carries the key. A pruned id is fenced for a while, so a
//! decision over a stale view does not relearn it.
//!
//! The key is the decision's session key: an encode item's content hash under
//! the `epd` plan, the request's routing key elsewhere. One ring serves every
//! pool: a lookup walks past the members that are not candidates of the
//! decision, so the owner is the first candidate clockwise whatever else the
//! ring holds, which is what a ring per model answers.

use std::{
    cell::RefCell,
    collections::VecDeque,
    time::{Duration, Instant},
};

use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    decision::MAX_TARGETS,
    framework::{Scorer, TieBand},
    lease::{Lifecycle, Membership},
    spec::{BoundedLoadParams, Needs},
    view::{Candidate, RequestFacts, WorkerId},
    Draw,
};

/// Positions per member: an even share of the key space per member for the
/// fleets a gateway holds, as the host's ring had.
pub const VIRTUAL_NODES: u32 = 150;

/// The position of one of a member's virtual nodes.
fn position(key: u32, vnode: u32) -> u64 {
    Draw::new(u64::from(key) ^ u64::from(vnode).wrapping_mul(0x9e37_79b9_7f4a_7c15)).next_u64()
}

/// How long a pruned id stays fenced against relearning from a view older
/// than the pruning: longer than any gap between a request's capture of the
/// pool and its decision (the stage holds no await between the two), shorter
/// than a worker's re-registration, after which a new handle at a reused
/// address may carry the same id and joins from its view as any other. The
/// feed naming the id (added, replacing, in a snapshot) lifts it at once.
pub const FENCE: Duration = Duration::from_secs(10);

/// The position a key hashes to.
fn key_position(key: u64) -> u64 {
    Draw::new(key).next_u64()
}

/// What a walk's visitor answers for the member it is shown.
pub enum Probe<T> {
    /// Take the member: the walk ends with the value.
    Take(T),
    /// Pass the member over.
    Skip,
    /// Nothing left to find: the walk ends empty.
    Stop,
}

/// The ring: its members, each at [`VIRTUAL_NODES`] sorted positions hashed
/// from the member's stable key and indexed by the handle id; the ids pruned
/// within [`FENCE`], fenced against relearning from a stale view.
#[derive(Debug, Default)]
pub struct Ring {
    /// The members, by the index the entries name them by.
    members: Vec<WorkerId>,
    index: FxHashMap<WorkerId, u32>,
    /// `(position, member index)`, sorted by position.
    entries: Vec<(u64, u32)>,
    /// The ids pruned within [`FENCE`], oldest first, each with its
    /// retirement.
    retired: VecDeque<(WorkerId, Instant)>,
    retired_set: FxHashSet<WorkerId>,
}

impl Ring {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn contains(&self, id: WorkerId) -> bool {
        self.index.contains_key(&id)
    }

    /// The members, by the index a walk names them by.
    pub fn members(&self) -> &[WorkerId] {
        &self.members
    }

    /// The index a walk names a member by.
    pub fn member_index(&self, id: WorkerId) -> Option<u32> {
        self.index.get(&id).copied()
    }

    /// Add one member at its key's positions; `false` when it is a member.
    /// One of [`Ring::add_many`]'s merges, so a fleet registering one by one
    /// costs a pass per worker and no re-sort.
    pub fn add(&mut self, id: WorkerId, key: u32) -> bool {
        self.add_many([(id, key)]) == 1
    }

    /// Add every member not yet in the ring, in one merge: the fresh
    /// positions of all of them sorted once and merged once into the
    /// entries, so a pool the ring never saw costs one pass. Returns how
    /// many joined.
    pub fn add_many(&mut self, members: impl IntoIterator<Item = (WorkerId, u32)>) -> usize {
        let mut fresh: Vec<(u64, u32)> = Vec::new();
        let mut joined = 0;
        for (id, key) in members {
            if self.index.contains_key(&id) {
                continue;
            }
            let member = self.members.len() as u32;
            self.members.push(id);
            self.index.insert(id, member);
            fresh.extend((0..VIRTUAL_NODES).map(|vnode| (position(key, vnode), member)));
            joined += 1;
        }
        if joined == 0 {
            return 0;
        }
        fresh.sort_unstable();
        let mut merged = Vec::with_capacity(self.entries.len() + fresh.len());
        let (mut i, mut j) = (0, 0);
        while i < self.entries.len() && j < fresh.len() {
            if self.entries[i] <= fresh[j] {
                merged.push(self.entries[i]);
                i += 1;
            } else {
                merged.push(fresh[j]);
                j += 1;
            }
        }
        merged.extend_from_slice(&self.entries[i..]);
        merged.extend_from_slice(&fresh[j..]);
        self.entries = merged;
        joined
    }

    /// [`Ring::add_many`] for the members a decision's view carries at `now`:
    /// an id the lifecycle pruned within [`FENCE`] is left out, since the
    /// view may be older than the pruning. Returns how many joined.
    pub fn learn(
        &mut self,
        members: impl IntoIterator<Item = (WorkerId, u32)>,
        now: Instant,
    ) -> usize {
        self.expire(now);
        let retired = &self.retired_set;
        let fresh: Vec<(WorkerId, u32)> = members
            .into_iter()
            .filter(|(id, _)| !retired.contains(id))
            .collect();
        self.add_many(fresh)
    }

    /// Whether `id` was pruned within [`FENCE`] of `now` and is fenced
    /// against relearning.
    pub fn is_retired(&self, id: WorkerId, now: Instant) -> bool {
        self.retired
            .iter()
            .any(|&(retired, at)| retired == id && now.saturating_duration_since(at) < FENCE)
    }

    /// Lift the fence on `id`: the feed named it as a member again.
    pub fn admit(&mut self, id: WorkerId) {
        if self.retired_set.remove(&id) {
            self.retired.retain(|&(retired, _)| retired != id);
        }
    }

    /// Forget the retirements older than [`FENCE`] at `now`.
    fn expire(&mut self, now: Instant) {
        while let Some(&(old, at)) = self.retired.front() {
            if now.saturating_duration_since(at) < FENCE {
                break;
            }
            self.retired.pop_front();
            if !self.retired.iter().any(|&(retired, _)| retired == old) {
                self.retired_set.remove(&old);
            }
        }
    }

    fn retire(&mut self, id: WorkerId, now: Instant) {
        self.expire(now);
        self.retired_set.insert(id);
        self.retired.push_back((id, now));
    }

    /// Remove a member at `now`, fencing its id for [`FENCE`]; `false` when
    /// it was none.
    pub fn remove(&mut self, id: WorkerId, now: Instant) -> bool {
        self.retire(id, now);
        let Some(gone) = self.index.remove(&id) else {
            return false;
        };
        self.entries.retain(|&(_, member)| member != gone);
        let last = (self.members.len() - 1) as u32;
        self.members.swap_remove(gone as usize);
        if gone != last {
            let moved = self.members[gone as usize];
            self.index.insert(moved, gone);
            for entry in &mut self.entries {
                if entry.1 == last {
                    entry.1 = gone;
                }
            }
        }
        true
    }

    /// Keep the named members only (a snapshot after a resync): the others
    /// leave and are fenced, a named id that was fenced is admitted again,
    /// and a named id the ring does not hold joins from the next view, which
    /// carries its key.
    pub fn retain_only(&mut self, ids: &[WorkerId], now: Instant) {
        let keep: FxHashSet<WorkerId> = ids.iter().copied().collect();
        let gone: Vec<WorkerId> = self
            .members
            .iter()
            .copied()
            .filter(|member| !keep.contains(member))
            .collect();
        for id in gone {
            self.remove(id, now);
        }
        for &id in ids {
            self.admit(id);
        }
    }

    /// Walk clockwise from the key's position, showing `visit` each entry's
    /// member until it takes one or stops; the walk goes once around.
    pub fn walk<T>(&self, key: u64, mut visit: impl FnMut(u32) -> Probe<T>) -> Option<T> {
        let at = key_position(key);
        let start = self.entries.partition_point(|&(p, _)| p < at);
        let (before, from) = self.entries.split_at(start);
        for &(_, member) in from.iter().chain(before) {
            match visit(member) {
                Probe::Take(value) => return Some(value),
                Probe::Skip => {}
                Probe::Stop => return None,
            }
        }
        None
    }
}

thread_local! {
    /// The slots of the decision in progress on this thread: per ring
    /// member, the kept index of the candidate it is, with the marks of the
    /// walk; reused so a decision allocates nothing after the first.
    static SLOTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// A member that is no candidate of this decision.
const FREE: u32 = u32::MAX;
/// The member was passed over by the bounded-load rule.
const PASSED: u32 = 1 << 31;
/// The member was ranked already.
const TAKEN: u32 = 1 << 30;
const MARKS: u32 = PASSED | TAKEN;

/// The consistent-hash scorer: the ring owner of the decision's key first,
/// the ring's next members after it, every other survivor at zero; a keyless
/// decision scores every survivor equal, so the picker's draw places it. With
/// a bounded-load rule an owner whose in-flight count is past `factor` times
/// the survivors' mean plus `abs_margin` is passed over for the next member;
/// when every survivor is past it, the plain owner stands.
#[derive(Debug)]
pub struct ConsistentHash {
    /// `(factor, abs_margin)` of the bounded-load rule, when the spec writes
    /// one.
    bounded: Option<(f32, u32)>,
    ring: RwLock<Ring>,
}

impl ConsistentHash {
    pub fn new(bounded: Option<&BoundedLoadParams>) -> Self {
        Self {
            bounded: bounded
                .map(|rule| (rule.factor.unwrap_or(1.25), rule.abs_margin.unwrap_or(10))),
            ring: RwLock::new(Ring::new()),
        }
    }

    /// The ring, for a reader.
    pub fn ring(&self) -> RwLockReadGuard<'_, Ring> {
        self.ring.read()
    }

    /// Fill `slots` with the kept index of each candidate that is a member;
    /// `true` when every kept candidate is one.
    fn fill(ring: &Ring, candidates: &[Candidate], kept: &[u32], slots: &mut Vec<u32>) -> bool {
        slots.clear();
        slots.resize(ring.len(), FREE);
        let mut every = true;
        for (i, &k) in kept.iter().enumerate() {
            match ring.member_index(candidates[k as usize].id) {
                Some(member) => slots[member as usize] = i as u32,
                None => every = false,
            }
        }
        every
    }

    /// One walk from the key: the members that are survivors are ranked in
    /// ring order into `out`, the owner at `1.0` and each next member a step
    /// below, at most [`MAX_TARGETS`] of them; under `bound` a survivor past
    /// it is passed over. Returns how many were ranked.
    fn rank(
        ring: &Ring,
        key: u64,
        candidates: &[Candidate],
        kept: &[u32],
        slots: &mut [u32],
        out: &mut [f32],
        bound: Option<f32>,
    ) -> usize {
        let mut found = 0usize;
        let mut passed = 0usize;
        ring.walk(key, |member| {
            let slot = slots[member as usize];
            if slot == FREE || slot & TAKEN != 0 {
                return Probe::Skip;
            }
            let index = (slot & !MARKS) as usize;
            if let Some(bound) = bound {
                if slot & PASSED != 0 {
                    return Probe::Skip;
                }
                if candidates[kept[index] as usize].in_flight as f32 > bound {
                    slots[member as usize] = slot | PASSED;
                    passed += 1;
                    return if passed == kept.len() {
                        Probe::Stop
                    } else {
                        Probe::Skip
                    };
                }
            }
            slots[member as usize] = slot | TAKEN;
            out[index] = 1.0 - found as f32 / MAX_TARGETS as f32;
            found += 1;
            if found == MAX_TARGETS || found + passed == kept.len() {
                Probe::Take(())
            } else {
                Probe::Skip
            }
        });
        found
    }
}

impl Lifecycle for ConsistentHash {
    fn on_membership(&self, change: &Membership<'_>) {
        let now = Instant::now();
        let mut ring = self.ring.write();
        match *change {
            // An added or replacing handle joins from the view, which
            // carries its key: the feed names ids alone, and naming one
            // lifts the fence a retirement left on it.
            Membership::Added(id) => ring.admit(id),
            Membership::Removed(id) => {
                ring.remove(id, now);
            }
            Membership::Replaced { old, new } => {
                ring.remove(old, now);
                ring.admit(new);
            }
            Membership::Snapshot(ids) => ring.retain_only(ids, now),
            Membership::Relabelled(_) | Membership::Resized(_) => {}
        }
    }
}

impl Scorer for ConsistentHash {
    fn name(&self) -> &'static str {
        "consistent-hash"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        out.fill(0.0);
        let key = facts.session_key.hash;
        if key == 0 || kept.is_empty() {
            return TieBand::NORMALIZED;
        }
        SLOTS.with(|slots| {
            let mut slots = slots.borrow_mut();
            let mut ring = self.ring.read();
            if !Self::fill(&ring, candidates, kept, &mut slots) {
                // A survivor the feed never named joins at its first
                // decision; rare, and never again for that worker.
                drop(ring);
                {
                    let mut ring = self.ring.write();
                    ring.learn(
                        kept.iter().map(|&k| {
                            let c = &candidates[k as usize];
                            (c.id, c.key)
                        }),
                        Instant::now(),
                    );
                }
                ring = self.ring.read();
                Self::fill(&ring, candidates, kept, &mut slots);
            }
            let bound = self.bounded.map(|(factor, margin)| {
                let total: f32 = kept
                    .iter()
                    .map(|&k| candidates[k as usize].in_flight as f32)
                    .sum();
                factor * (total / kept.len() as f32) + margin as f32
            });
            if Self::rank(&ring, key, candidates, kept, &mut slots, out, bound) == 0 {
                // Every survivor is past the bound: the plain owner stands.
                for slot in slots.iter_mut() {
                    if *slot != FREE {
                        *slot &= !PASSED;
                    }
                }
                Self::rank(&ring, key, candidates, kept, &mut slots, out, None);
            }
        });
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            ring: true,
            ..Needs::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{Health, Identity};

    fn ids(n: u64) -> Vec<WorkerId> {
        (0..n).map(|i| WorkerId(1_000 + i)).collect()
    }

    /// The stable key of a test worker: its id's low bits (the candidates
    /// below carry the same, so a ring built here and a view agree).
    fn key_of(id: WorkerId) -> u32 {
        id.0 as u32
    }

    fn ring_of(n: u64) -> Ring {
        let mut ring = Ring::new();
        for id in ids(n) {
            assert!(ring.add(id, key_of(id)));
        }
        ring
    }

    /// The owner of a key: the first member clockwise from its position.
    fn owner_of(ring: &Ring, key: u64) -> Option<WorkerId> {
        ring.walk(key, |member| Probe::Take(ring.members()[member as usize]))
    }

    fn owner(ring: &Ring, key: u64) -> WorkerId {
        owner_of(ring, key).expect("a member owns every key")
    }

    #[test]
    fn a_member_that_leaves_moves_its_keys_alone_and_a_replacement_takes_new_positions() {
        let mut ring = ring_of(8);
        assert_eq!(ring.len(), 8);
        assert!(!ring.add(WorkerId(1_000), 1_000), "a member is added once");
        let keys: Vec<u64> = (1..=2_000).collect();
        let before: Vec<WorkerId> = keys.iter().map(|&k| owner(&ring, k)).collect();
        let gone = WorkerId(1_003);
        let owned = before.iter().filter(|&&o| o == gone).count();
        assert!(owned > 100 && owned < 400, "an even share: {owned} of 2000");
        let now = Instant::now();
        assert!(ring.remove(gone, now));
        assert!(!ring.remove(gone, now));
        assert_eq!(ring.len(), 7);
        assert!(!ring.contains(gone));
        let (mut moved, mut kept) = (0, 0);
        for (key, was) in keys.iter().zip(&before) {
            let now = owner(&ring, *key);
            if *was == gone {
                moved += 1;
                assert_ne!(now, gone);
            } else {
                kept += 1;
                assert_eq!(now, *was, "key {key} left a member that stayed");
            }
        }
        assert_eq!(moved, owned);
        assert_eq!(kept, 2_000 - owned);
        // A replacement is a new id: it takes positions of its own and owns
        // keys, and the keys of the members that stayed stay with them.
        let fresh = WorkerId(77);
        assert!(ring.add(fresh, key_of(fresh)));
        let taken = keys.iter().filter(|&&k| owner(&ring, k) == fresh).count();
        assert!(taken > 100 && taken < 400, "{taken} of 2000");
        for (key, was) in keys.iter().zip(&before) {
            if *was != gone {
                let now = owner(&ring, *key);
                assert!(now == *was || now == fresh, "key {key}: {was:?} -> {now:?}");
            }
        }
    }

    #[test]
    fn a_snapshot_keeps_its_ids_only_and_the_indices_stay_dense() {
        let mut ring = ring_of(5);
        let now = Instant::now();
        // A member pruned and named again by the snapshot is admitted.
        ring.remove(WorkerId(1_004), now);
        assert!(ring.is_retired(WorkerId(1_004), now));
        let snapshot = [WorkerId(1_001), WorkerId(1_004), WorkerId(42)];
        ring.retain_only(&snapshot, now);
        let mut members = ring.members().to_vec();
        members.sort();
        assert_eq!(
            members,
            [WorkerId(1_001)],
            "the unnamed leave; the named wait for the view, which carries their keys"
        );
        assert!(ring.is_retired(WorkerId(1_000), now) && ring.is_retired(WorkerId(1_003), now));
        assert!(
            !ring.is_retired(WorkerId(42), now),
            "a named id is not fenced"
        );
        assert!(
            !ring.is_retired(WorkerId(1_004), now),
            "naming a pruned id admits it"
        );
        assert_eq!(
            ring.learn(
                [
                    (WorkerId(42), 42),
                    (WorkerId(1_004), key_of(WorkerId(1_004)))
                ],
                now
            ),
            2,
            "the view brings both"
        );
        let mut members = ring.members().to_vec();
        members.sort();
        assert_eq!(members, [WorkerId(42), WorkerId(1_001), WorkerId(1_004)]);
        for (i, &member) in ring.members().iter().enumerate() {
            assert_eq!(ring.member_index(member), Some(i as u32));
        }
        assert!(ring.entries.iter().all(|&(_, m)| (m as usize) < ring.len()));
        assert_eq!(ring.entries.len(), 3 * VIRTUAL_NODES as usize);
        assert!(ring.entries.windows(2).all(|w| w[0] <= w[1]), "sorted");
    }

    #[test]
    fn an_empty_ring_owns_nothing() {
        let ring = Ring::new();
        assert!(ring.is_empty());
        assert_eq!(owner_of(&ring, 7), None);
    }

    fn candidate(position: u32, in_flight: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(1_000 + u64::from(position)),
            key: 1_000 + position,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        }
    }

    fn keyed(key: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            ..RequestFacts::default()
        }
    }

    fn scored(scorer: &ConsistentHash, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Vec<f32> {
        let kept: Vec<u32> = (0..pool.len() as u32).collect();
        let mut out = vec![0.0; pool.len()];
        scorer.score(facts, pool, &kept, &mut out);
        out
    }

    fn leader(scores: &[f32]) -> usize {
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("a score")
    }

    #[test]
    fn the_scorer_learns_the_survivors_and_ranks_the_owner_then_the_ring_order() {
        let scorer = ConsistentHash::new(None);
        let pool: Vec<Candidate> = (0..8).map(|i| candidate(i, 0)).collect();
        assert!(scorer.ring().is_empty());
        let scores = scored(&scorer, &keyed(11), &pool);
        assert_eq!(scorer.ring().len(), 8, "every survivor joined");
        let owner = leader(&scores);
        assert_eq!(scores[owner], 1.0);
        let mut ranked: Vec<f32> = scores.iter().copied().filter(|&s| s > 0.0).collect();
        ranked.sort_by(f32::total_cmp);
        assert_eq!(
            ranked,
            [0.25, 0.5, 0.75, 1.0],
            "the owner and three successors"
        );
        // The ring's own answer agrees, and the next decision reads the same.
        let expected = owner_of(&scorer.ring(), 11).expect("an owner");
        assert_eq!(pool[owner].id, expected);
        assert_eq!(leader(&scored(&scorer, &keyed(11), &pool)), owner);
        // A survivor set without the owner is owned by the owner's successor.
        let rest: Vec<Candidate> = pool
            .iter()
            .copied()
            .filter(|c| c.position != owner as u32)
            .collect();
        let successor = pool
            .iter()
            .position(|c| scores[c.position as usize] == 0.75)
            .expect("the successor");
        let kept: Vec<u32> = (0..rest.len() as u32).collect();
        let mut out = vec![0.0; rest.len()];
        scorer.score(&keyed(11), &rest, &kept, &mut out);
        assert_eq!(rest[leader(&out)].id, pool[successor].id);
    }

    #[test]
    fn keys_spread_over_the_members_and_a_keyless_decision_scores_them_equal() {
        let scorer = ConsistentHash::new(None);
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let mut owners = [0u32; 4];
        for key in 1..=400u64 {
            owners[leader(&scored(&scorer, &keyed(key), &pool))] += 1;
        }
        assert!(owners.iter().all(|&n| n > 40), "{owners:?}");
        let equal = scored(&scorer, &RequestFacts::default(), &pool);
        assert!(equal.iter().all(|&s| s == 0.0), "{equal:?}");
    }

    #[test]
    fn a_removed_member_passes_its_keys_to_its_successor_and_the_others_keep_theirs() {
        let scorer = ConsistentHash::new(None);
        let pool: Vec<Candidate> = (0..6).map(|i| candidate(i, 0)).collect();
        let before: Vec<usize> = (1..=300u64)
            .map(|key| leader(&scored(&scorer, &keyed(key), &pool)))
            .collect();
        let gone = 2usize;
        scorer.on_membership(&Membership::Removed(pool[gone].id));
        assert!(!scorer.ring().contains(pool[gone].id));
        // The removed worker is still handed in as a candidate (a stale
        // view): it rejoins, so the test removes it from the pool as the
        // host does.
        let rest: Vec<Candidate> = pool
            .iter()
            .copied()
            .filter(|c| c.position != gone as u32)
            .collect();
        for (key, was) in (1..=300u64).zip(&before) {
            let kept: Vec<u32> = (0..rest.len() as u32).collect();
            let mut out = vec![0.0; rest.len()];
            scorer.score(&keyed(key), &rest, &kept, &mut out);
            let now = rest[leader(&out)].position as usize;
            if *was != gone {
                assert_eq!(now, *was, "key {key}");
            }
        }
        // A replacement at the same address is a new handle with the same
        // key: the old id leaves, the new one joins from the view at the
        // predecessor's positions, so the replaced member's keys come back
        // to it and nothing else moves.
        let live = pool[4];
        let replacement = WorkerId(9_999);
        scorer.on_membership(&Membership::Replaced {
            old: live.id,
            new: replacement,
        });
        assert!(!scorer.ring().contains(live.id), "the replaced member left");
        assert!(
            !scorer.ring().contains(replacement),
            "the new handle joins from the view, which carries its key"
        );
        assert_eq!(scorer.ring().len(), 4);
        let mut renewed = rest.clone();
        let slot = renewed
            .iter()
            .position(|c| c.id == live.id)
            .expect("the replaced member is in the stale pool");
        renewed[slot].id = replacement;
        for (key, was) in (1..=300u64).zip(&before) {
            let kept: Vec<u32> = (0..renewed.len() as u32).collect();
            let mut out = vec![0.0; renewed.len()];
            scorer.score(&keyed(key), &renewed, &kept, &mut out);
            let now = renewed[leader(&out)].position as usize;
            if *was != gone {
                assert_eq!(
                    now, *was,
                    "key {key}: the replacement keeps its predecessor's keys"
                );
            }
        }
        assert!(scorer.ring().contains(replacement));
        assert_eq!(scorer.ring().len(), 5);
    }

    /// Two rings built from different handles of the same workers (the ids
    /// differ, the keys do not: two gateway replicas, or a restart) agree on
    /// the owner of every key.
    #[test]
    fn two_rings_over_the_same_keys_from_different_handles_agree_on_every_owner() {
        let mut a = Ring::new();
        let mut b = Ring::new();
        for i in 0..16u64 {
            let key = 7_000 + i as u32;
            assert!(a.add(WorkerId(1_000 + i), key));
            assert!(b.add(WorkerId(5_000 + i), key));
        }
        for key in 1..=500u64 {
            let owner_a = owner(&a, key);
            let owner_b = owner(&b, key);
            assert_eq!(
                key_of(owner_a) - 1_000,
                key_of(owner_b) - 5_000,
                "key {key}"
            );
        }
    }

    /// A pruned id is fenced: a decision over a view older than the pruning
    /// does not relearn it; the feed naming it (added, or in a snapshot)
    /// admits it at once, and it joins from the next view.
    #[test]
    fn a_pruned_id_is_fenced_from_a_stale_view_until_the_feed_names_it() {
        let scorer = ConsistentHash::new(None);
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let kept: Vec<u32> = (0..4).collect();
        let mut out = vec![0.0; 4];
        scorer.score(&keyed(3), &pool, &kept, &mut out);
        assert_eq!(scorer.ring().len(), 4);
        let gone = pool[1].id;
        scorer.on_membership(&Membership::Removed(gone));
        assert!(!scorer.ring().contains(gone));
        assert!(scorer.ring().is_retired(gone, Instant::now()));
        // The stale view still carries the pruned member.
        scorer.score(&keyed(3), &pool, &kept, &mut out);
        assert!(
            !scorer.ring().contains(gone),
            "a stale view relearns nothing"
        );
        assert_eq!(scorer.ring().len(), 3);
        scorer.on_membership(&Membership::Added(gone));
        assert!(
            !scorer.ring().is_retired(gone, Instant::now()),
            "the feed naming it lifts the fence"
        );
        scorer.score(&keyed(3), &pool, &kept, &mut out);
        assert!(
            scorer.ring().contains(gone),
            "and the next view brings it back"
        );
        assert_eq!(scorer.ring().len(), 4);
        // A replacing handle is named too: the old id leaves, the new one
        // is admitted even when it repeats the old (a reused address).
        scorer.on_membership(&Membership::Replaced {
            old: gone,
            new: gone,
        });
        assert!(!scorer.ring().contains(gone));
        assert!(!scorer.ring().is_retired(gone, Instant::now()));
        scorer.score(&keyed(3), &pool, &kept, &mut out);
        assert!(scorer.ring().contains(gone));
    }

    /// The fence lapses: a handle at a reused address carrying a pruned id
    /// joins from its view once [`FENCE`] has passed, and a stale view
    /// within the window relearns nothing.
    #[test]
    fn the_fence_lapses_after_its_window_and_a_reused_id_joins_from_the_view() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_002);
        assert!(ring.remove(id, t0));
        let soon = t0 + FENCE / 2;
        assert!(ring.is_retired(id, soon));
        assert_eq!(
            ring.learn([(id, key_of(id))], soon),
            0,
            "fenced within the window"
        );
        let later = t0 + FENCE;
        assert!(!ring.is_retired(id, later), "the fence has lapsed");
        assert_eq!(
            ring.learn([(id, key_of(id))], later),
            1,
            "the reused id joins"
        );
        assert!(ring.contains(id));
        // A second retirement fences it afresh, from its own instant.
        assert!(ring.remove(id, later));
        assert!(ring.is_retired(id, later + FENCE / 2));
        assert!(!ring.is_retired(id, later + FENCE));
    }

    /// Learning many members at once builds the ring one add at a time
    /// would, in one merge.
    #[test]
    fn learning_many_at_once_builds_the_ring_one_add_at_a_time_would() {
        let one_by_one = ring_of(12);
        let mut at_once = Ring::new();
        let now = Instant::now();
        let joined = at_once.learn(ids(12).into_iter().map(|id| (id, key_of(id))), now);
        assert_eq!(joined, 12);
        assert_eq!(at_once.members(), one_by_one.members());
        for key in 1..=200u64 {
            assert_eq!(owner(&at_once, key), owner(&one_by_one, key), "key {key}");
        }
        assert_eq!(
            at_once.learn(ids(12).into_iter().map(|id| (id, key_of(id))), now),
            0
        );
    }

    #[test]
    fn the_bounded_load_rule_passes_an_overloaded_owner_on_and_falls_back_when_every_survivor_is_over(
    ) {
        let bounded = ConsistentHash::new(Some(&BoundedLoadParams {
            factor: Some(1.0),
            abs_margin: Some(0),
        }));
        let plain = ConsistentHash::new(None);
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let owner = leader(&scored(&plain, &keyed(5), &pool));
        let successor = pool
            .iter()
            .position(|c| scored(&plain, &keyed(5), &pool)[c.position as usize] == 0.75)
            .expect("the successor");
        let mut loaded = pool.clone();
        loaded[owner].in_flight = 8;
        let scores = scored(&bounded, &keyed(5), &loaded);
        assert_eq!(leader(&scores), successor, "the owner is past the bound");
        assert_eq!(scores[owner], 0.0, "a passed owner ranks nowhere");
        // Every survivor strictly above the bound: with every candidate at
        // 8 the mean is 8, so a factor of 1 passes nobody; half the mean
        // (bound 4) passes everyone, and the plain order is restored.
        for c in &mut loaded {
            c.in_flight = 8;
        }
        let half = ConsistentHash::new(Some(&BoundedLoadParams {
            factor: Some(0.5),
            abs_margin: Some(0),
        }));
        let all_over = scored(&half, &keyed(5), &loaded);
        assert_eq!(
            leader(&all_over),
            owner,
            "every survivor is past the bound: the plain owner stands"
        );
        assert_eq!(
            all_over[owner], 1.0,
            "the plain order is restored, not an empty one"
        );
        assert_eq!(all_over[successor], 0.75, "and the successor ranks next");
    }

    #[test]
    fn the_scorer_declares_the_ring_need() {
        assert!(ConsistentHash::new(None).needs().ring);
        assert!(!ConsistentHash::new(None).needs().loads);
    }
}
