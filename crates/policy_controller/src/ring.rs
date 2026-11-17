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
//!
//! Under the prefix form ([`ConsistentHash::keyed_by_prefix`]) a request
//! with no routing key is keyed by its prefix: the hash of its last whole
//! block at or under the form's depth in tokens. A block hash names the
//! prefix that ends with its block, so the key is one read and requests that
//! share that much of a prompt share an owner. The depth counts in the
//! blocks the request was hashed by, text windows at four characters a token
//! under the text kind. A request that carries no whole block has no key,
//! and the tokens past a request's last whole block are not in its key.

use std::{
    cell::RefCell,
    collections::VecDeque,
    time::{Duration, Instant},
};

use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    affinity::Gates,
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
    /// The retirements within [`FENCE`], oldest first. Those of an id the
    /// feed named since stay until they lapse and lift nothing.
    retired: VecDeque<(WorkerId, Instant)>,
    /// Each fenced id with its last retirement, the one whose lapse lifts
    /// the fence.
    fenced: FxHashMap<WorkerId, Instant>,
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
        let fenced = &self.fenced;
        let fresh: Vec<(WorkerId, u32)> = members
            .into_iter()
            .filter(|(id, _)| !fenced.contains_key(id))
            .collect();
        self.add_many(fresh)
    }

    /// Whether `id` was pruned within [`FENCE`] of `now` and is fenced
    /// against relearning.
    pub fn is_retired(&self, id: WorkerId, now: Instant) -> bool {
        self.fenced
            .get(&id)
            .is_some_and(|&at| now.saturating_duration_since(at) < FENCE)
    }

    /// Lift the fence on `id`: the feed named it as a member again.
    pub fn admit(&mut self, id: WorkerId) {
        // Its retirements stay in the list: none is its last any more, so
        // their lapses lift nothing.
        self.fenced.remove(&id);
    }

    /// Forget the retirements older than [`FENCE`] at `now`.
    fn expire(&mut self, now: Instant) {
        while let Some(&(old, at)) = self.retired.front() {
            if now.saturating_duration_since(at) < FENCE {
                break;
            }
            self.retired.pop_front();
            // The fence lifts with the id's last retirement. The map says
            // whether this entry is it: no pass over the list for a lapse.
            if self.fenced.get(&old) == Some(&at) {
                self.fenced.remove(&old);
            }
        }
    }

    fn retire(&mut self, id: WorkerId, now: Instant) {
        self.expire(now);
        let last = self.fenced.entry(id).or_insert(now);
        *last = (*last).max(now);
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
/// The member was ranked already.
const TAKEN: u32 = 1 << 30;

/// The key of a request's prefix at a depth of `tokens`: the hash of its
/// last whole block at or under that depth, `0` where it carries no block
/// that shallow. A request shorter than the depth is keyed at its own last
/// whole block, and so is one whose hashes the host capped short of the
/// depth.
fn prefix_key(tokens: u32, gates: Gates, facts: &RequestFacts<'_>) -> u64 {
    let depth = gates.whole_blocks(tokens, facts) as usize;
    depth
        .min(facts.block_hashes.len())
        .checked_sub(1)
        .map_or(0, |last| facts.block_hashes[last])
}

/// The consistent-hash scorer: the ring owner of the decision's key first,
/// the ring's next members after it, every other survivor at zero; a keyless
/// decision scores every survivor equal, so the profile's next stage places
/// it, or the picker's draw. With a bounded-load rule an owner whose
/// in-flight count is past both `factor` times the survivors' mean and
/// `abs_margin` over it, the mean counting the request being placed, is
/// passed over: the decision is scored as a keyless one. The ring's next
/// members follow an owner that keeps its key, and they take the key of an
/// owner that is no survivor, a worker that is down or was tried.
///
/// The owner must be among the rows scored, so the scorer answers for the
/// pool whole and never row by row ([`Scorer::row_wise`]): a profile that
/// names it runs its whole pass, with no rows drawn first and no sample.
#[derive(Debug)]
pub struct ConsistentHash {
    /// `(factor, abs_margin)` of the bounded-load rule, when the spec writes
    /// one.
    bounded: Option<(f64, u32)>,
    /// The prefix form: the depth in tokens a request with no routing key
    /// is keyed at, and the gates that hold the view's block size.
    prefix: Option<(u32, Gates)>,
    ring: RwLock<Ring>,
}

impl ConsistentHash {
    pub fn new(bounded: Option<&BoundedLoadParams>) -> Self {
        Self {
            bounded: bounded
                .map(|rule| (rule.factor.unwrap_or(1.25), rule.abs_margin.unwrap_or(10))),
            prefix: None,
            ring: RwLock::new(Ring::new()),
        }
    }

    /// The prefix form: a request with no routing key is keyed by its
    /// prefix, the hash of its last whole block at or under `tokens`, at the
    /// block size `gates` reads for the request.
    #[must_use]
    pub fn keyed_by_prefix(mut self, tokens: u32, gates: Gates) -> Self {
        self.prefix = Some((tokens, gates));
        self
    }

    /// Whether the key's owner among the survivors holds more requests than
    /// `bound`. `slots` is as [`Self::fill`] left it.
    fn passed_over(
        ring: &Ring,
        key: u64,
        candidates: &[Candidate],
        kept: &[u32],
        slots: &[u32],
        bound: f64,
    ) -> bool {
        ring.walk(key, |member| match slots[member as usize] {
            FREE => Probe::Skip,
            slot => {
                let owner = &candidates[kept[slot as usize] as usize];
                Probe::Take(f64::from(owner.in_flight) > bound)
            }
        })
        .unwrap_or(false)
    }

    /// The decision's key, `0` for none: the routing key, and under the
    /// prefix form the request's prefix where it carries no routing key.
    fn key_of(&self, facts: &RequestFacts<'_>) -> u64 {
        match (facts.session_key.hash, self.prefix) {
            (0, Some((tokens, gates))) => prefix_key(tokens, gates, facts),
            (key, _) => key,
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
    /// below, at most [`MAX_TARGETS`] of them.
    fn rank(ring: &Ring, key: u64, survivors: usize, slots: &mut [u32], out: &mut [f32]) {
        let mut found = 0usize;
        ring.walk(key, |member| {
            let slot = slots[member as usize];
            if slot == FREE || slot & TAKEN != 0 {
                return Probe::Skip;
            }
            slots[member as usize] = slot | TAKEN;
            out[slot as usize] = 1.0 - found as f32 / MAX_TARGETS as f32;
            found += 1;
            if found == MAX_TARGETS || found == survivors {
                Probe::Take(())
            } else {
                Probe::Skip
            }
        });
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
        let key = self.key_of(facts);
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
            if let Some((factor, margin)) = self.bounded {
                // Both margins over the mean, the request being placed
                // counted in it: the larger of the two is the bound. The
                // product is taken as written, in 64-bit floats: an owner
                // at the bound's edge stands or falls by its last bit.
                let total: u64 = kept
                    .iter()
                    .map(|&k| u64::from(candidates[k as usize].in_flight))
                    .sum();
                let mean = (total as f64 + 1.0) / kept.len() as f64;
                let bound = (mean * factor).max(mean + f64::from(margin));
                if Self::passed_over(&ring, key, candidates, kept, &slots, bound) {
                    return;
                }
            }
            Self::rank(&ring, key, kept.len(), &mut slots, out);
        });
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        // The prefix form reads the request's block hashes, in the kind the
        // host hashes by.
        let hashes = self.prefix.is_some();
        Needs {
            ring: true,
            token_hashes: hashes,
            text_hashes: hashes,
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
        scored_among(scorer, facts, pool, &kept)
    }

    /// The scores of the rows `kept` names, as a profile's filters leave a
    /// pool: one score a kept row.
    fn scored_among(
        scorer: &ConsistentHash,
        facts: &RequestFacts<'_>,
        pool: &[Candidate],
        kept: &[u32],
    ) -> Vec<f32> {
        let mut out = vec![0.0; kept.len()];
        scorer.score(facts, pool, kept, &mut out);
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

    /// An id retired twice inside one window stays fenced until its last
    /// retirement lapses: the lapse of the first lifts nothing.
    #[test]
    fn an_id_retired_twice_stays_fenced_until_its_last_retirement_lapses() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_002);
        assert!(ring.remove(id, t0));
        // The removal of an id the ring no longer holds retires it again.
        let again = t0 + FENCE / 2;
        assert!(!ring.remove(id, again));
        let first_lapsed = t0 + FENCE;
        assert_eq!(
            ring.learn([(id, key_of(id))], first_lapsed),
            0,
            "the later retirement stands"
        );
        assert!(ring.is_retired(id, first_lapsed));
        let both_lapsed = again + FENCE;
        assert_eq!(ring.learn([(id, key_of(id))], both_lapsed), 1);
        assert!(ring.contains(id));
        // Many ids retired at one instant lapse together and each is
        // learned again.
        let many: Vec<WorkerId> = (2_000..2_512).map(WorkerId).collect();
        for &gone in &many {
            ring.remove(gone, both_lapsed);
        }
        let pool = many.iter().map(|&gone| (gone, key_of(gone)));
        assert_eq!(ring.learn(pool.clone(), both_lapsed + FENCE / 2), 0);
        assert_eq!(ring.learn(pool, both_lapsed + FENCE), many.len());
    }

    /// An id the feed names and the ring prunes again is fenced by the
    /// later retirement alone: the lapse of the first lifts nothing.
    #[test]
    fn an_id_named_and_pruned_again_stays_fenced_until_the_later_retirement_lapses() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_001);
        assert!(ring.remove(id, t0));
        ring.admit(id);
        assert!(!ring.is_retired(id, t0), "named again");
        assert_eq!(ring.learn([(id, key_of(id))], t0), 1);
        let again = t0 + FENCE / 2;
        assert!(ring.remove(id, again));
        let first_lapsed = t0 + FENCE;
        assert_eq!(ring.learn([(id, key_of(id))], first_lapsed), 0);
        assert!(ring.is_retired(id, first_lapsed));
        assert_eq!(ring.learn([(id, key_of(id))], again + FENCE), 1);
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
    fn the_scorer_declares_the_ring_need() {
        assert!(ConsistentHash::new(None).needs().ring);
        assert!(!ConsistentHash::new(None).needs().loads);
    }

    /// A request of these block hashes, sixteen tokens a block, with no
    /// routing key.
    fn hashed(hashes: &[u64]) -> RequestFacts<'_> {
        RequestFacts {
            model: "m",
            block_hashes: hashes,
            prompt_tokens: 16 * hashes.len() as u32,
            ..RequestFacts::default()
        }
    }

    fn prefixed(tokens: u32) -> ConsistentHash {
        ConsistentHash::new(None).keyed_by_prefix(tokens, Gates::default())
    }

    #[test]
    fn the_prefix_key_is_the_last_whole_block_at_or_under_the_depth() {
        let hashes = [11, 22, 33, 44, 55, 66];
        let gates = Gates::default();
        // Blocks of 16 tokens: a depth of 64 tokens ends with the fourth
        // block, and a depth inside a block keys on the block before it.
        assert_eq!(prefix_key(64, gates, &hashed(&hashes)), 44);
        assert_eq!(prefix_key(79, gates, &hashed(&hashes)), 44);
        assert_eq!(prefix_key(80, gates, &hashed(&hashes)), 55);
        // A request shorter than the depth keys on its own last block. A
        // request with no whole block has no key, and neither has a depth
        // under one block.
        assert_eq!(prefix_key(256, gates, &hashed(&hashes)), 66);
        assert_eq!(prefix_key(256, gates, &hashed(&[])), 0);
        assert_eq!(prefix_key(15, gates, &hashed(&hashes)), 0);
        // The block size the request was counted with leads the view's.
        let learned = RequestFacts {
            block_size: 32,
            ..hashed(&hashes)
        };
        assert_eq!(prefix_key(64, gates, &learned), 22);
        // Under the text kind the depth is in windows at four characters a
        // token: 64 tokens are 256 characters, sixteen windows of sixteen.
        let windows: Vec<u64> = (1..=20).collect();
        let text = RequestFacts {
            hash_kind: crate::view::HashKind::Text,
            ..hashed(&windows)
        };
        assert_eq!(prefix_key(64, gates, &text), 16);
    }

    #[test]
    fn under_the_prefix_form_requests_that_share_a_prefix_share_an_owner() {
        let scorer = prefixed(64);
        let pool: Vec<Candidate> = (0..8).map(|i| candidate(i, 0)).collect();
        // The same first four blocks under two tails.
        let one = [11, 22, 33, 44, 55];
        let other = [11, 22, 33, 44, 99, 77];
        let scores = scored(&scorer, &hashed(&one), &pool);
        let owner = leader(&scores);
        assert_eq!(scores[owner], 1.0);
        assert_eq!(leader(&scored(&scorer, &hashed(&other), &pool)), owner);
        // The key is the block's hash: a routing key of that value has the
        // same owner under the plain form.
        let plain = ConsistentHash::new(None);
        assert_eq!(leader(&scored(&plain, &keyed(44), &pool)), owner);
    }

    #[test]
    fn a_routing_key_leads_the_prefix_and_the_plain_form_reads_no_prefix() {
        let pool: Vec<Candidate> = (0..8).map(|i| candidate(i, 0)).collect();
        let hashes = [11, 22, 33, 44];
        let plain = ConsistentHash::new(None);
        let by_prefix = leader(&scored(&prefixed(64), &hashed(&hashes), &pool));
        // A routing key whose owner is another worker than the prefix's.
        let (key, by_key) = (1..)
            .map(|key| (key, leader(&scored(&plain, &keyed(key), &pool))))
            .find(|&(_, owner)| owner != by_prefix)
            .expect("a key with another owner");
        let both = RequestFacts {
            session_key: Identity::hashed(key),
            ..hashed(&hashes)
        };
        assert_eq!(leader(&scored(&prefixed(64), &both, &pool)), by_key);
        // Without the form a request with no routing key has no key, and
        // under it a request with no whole block has none: every survivor
        // scores equal.
        for scores in [
            scored(&plain, &hashed(&hashes), &pool),
            scored(&prefixed(64), &hashed(&[]), &pool),
        ] {
            assert!(scores.iter().all(|&score| score == 0.0), "{scores:?}");
        }
    }

    fn bounded(factor: f64, abs_margin: u32) -> ConsistentHash {
        ConsistentHash::new(Some(&BoundedLoadParams {
            factor: Some(factor),
            abs_margin: Some(abs_margin),
        }))
    }

    /// Whether the owner of key 5 keeps it under `rule` over a pool that
    /// holds `loads`, listed in the key's ring order, the owner first. An
    /// owner passed over leaves the decision scored as a keyless one, and
    /// an owner that keeps its key leaves it scored as with no rule.
    fn keeps(rule: &ConsistentHash, loads: &[u32]) -> bool {
        let plain = ConsistentHash::new(None);
        let mut pool: Vec<Candidate> = (0..loads.len() as u32).map(|i| candidate(i, 0)).collect();
        let ranks = scored(&plain, &keyed(5), &pool);
        let mut order: Vec<usize> = (0..pool.len()).collect();
        order.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));
        for (&place, &load) in order.iter().zip(loads) {
            pool[place].in_flight = load;
        }
        let scores = scored(rule, &keyed(5), &pool);
        let kept = scores[order[0]] == 1.0;
        assert!(
            if kept {
                scores == ranks
            } else {
                scores.iter().all(|&score| score == 0.0)
            },
            "{loads:?}: {scores:?}"
        );
        kept
    }

    #[test]
    fn an_owner_past_both_margins_of_the_bounded_load_rule_is_passed_over() {
        let rule = bounded(1.25, 10);
        // The bound is the larger margin: past one alone the owner keeps
        // its key, past both it is passed over.
        assert!(keeps(&rule, &[12, 1, 2, 0]), "the relative margin alone");
        assert!(
            keeps(&rule, &[130, 100, 110, 90]),
            "the absolute margin alone"
        );
        assert!(!keeps(&rule, &[140, 100, 110, 90]), "both margins");
        // At the bound the owner keeps its key; one request past it, not.
        assert!(keeps(&rule, &[14, 1, 0, 0]));
        assert!(!keeps(&rule, &[15, 1, 0, 2]));
        // The spec's two numbers move the bound, and their defaults are
        // the rule's own.
        assert!(!keeps(&bounded(2.0, 4), &[14, 3, 4, 5]));
        assert!(keeps(&rule, &[14, 3, 4, 5]));
        let defaults = ConsistentHash::new(Some(&BoundedLoadParams::default()));
        assert!(keeps(&defaults, &[14, 1, 0, 0]));
        assert!(!keeps(&defaults, &[15, 1, 0, 2]));
        // An idle pool passes no owner over, under the tightest rule.
        assert!(keeps(&bounded(1.0, 0), &[0, 0, 0, 0]));
    }

    #[test]
    fn the_bounded_load_rule_counts_the_request_being_placed_in_the_mean() {
        let rule = bounded(1.25, 10);
        // Two workers: with the request counted the mean of 21 and 0 is 11
        // and the bound 21, which the owner is at; without it the bound
        // would be 20.5 and the owner past it.
        assert!(keeps(&rule, &[21, 0]));
        assert!(!keeps(&rule, &[22, 0]));
    }

    #[test]
    fn the_bounded_load_rule_reads_its_factor_at_the_spec_s_width() {
        // No binary fraction is 1.3. Two workers hold 52 and 27: with the
        // request the mean is 40 and the bound 52, which the owner is at.
        // A factor read through a narrower float falls short of 1.3 and
        // passes this owner over.
        let rule = bounded(1.3, 10);
        assert!(keeps(&rule, &[52, 27]));
        assert!(!keeps(&rule, &[53, 27]));
        // And 1.2 the other way. Three workers hold 14, 9 and 11: a third
        // of 35 times 1.2 is 14 on paper and a hair under it as the product
        // is computed, so the owner is passed over. A narrower 1.2 lies
        // over 1.2 and keeps it.
        assert!(!keeps(&bounded(1.2, 2), &[14, 9, 11]));
    }

    /// A scorer under the default rule whose ring knows four workers, and
    /// the ring order of `key` over them, the owner first.
    fn taught(key: u64) -> (ConsistentHash, Vec<usize>) {
        let rule = bounded(1.25, 10);
        let idle: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let ranks = scored(&rule, &keyed(key), &idle);
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));
        (rule, order)
    }

    /// Whether `rule` passes over the owner of `key` among the workers a
    /// filter leaves when it takes `filtered` out of four that hold `loads`.
    fn passes_over(rule: &ConsistentHash, key: u64, loads: [u32; 4], filtered: usize) -> bool {
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, loads[i as usize])).collect();
        let kept: Vec<u32> = (0..4).filter(|&i| i as usize != filtered).collect();
        let scores = scored_among(rule, &keyed(key), &pool, &kept);
        scores.iter().all(|&score| score == 0.0)
    }

    #[test]
    fn the_bounded_load_rule_reads_the_owner_s_own_row_behind_a_filtered_worker() {
        // Worker 0 is filtered and stands ahead of the owner in the pool:
        // the owner's place among the survivors is not its row.
        let key = (1..)
            .find(|&key| taught(key).1[0] != 0)
            .expect("a key that worker 0 does not own");
        let (rule, order) = taught(key);
        let mut loads = [0; 4];
        loads[order[0]] = 100;
        assert!(passes_over(&rule, key, loads, 0));
    }

    #[test]
    fn the_bounded_load_rule_holds_the_owner_among_the_survivors_to_the_bound() {
        // The ring's first member for the key is filtered: the next one
        // owns the key, and it is past the bound.
        let (rule, order) = taught(5);
        let mut loads = [0; 4];
        loads[order[1]] = 100;
        assert!(passes_over(&rule, 5, loads, order[0]));
    }

    #[test]
    fn the_bounded_load_rule_takes_its_mean_over_the_survivors_alone() {
        let (rule, order) = taught(5);
        let last = order[3];
        // The thousand requests of a filtered worker stay out of the sum:
        // over the three survivors the bound is under the owner's 30.
        let mut loads = [0; 4];
        (loads[order[0]], loads[last]) = (30, 1_000);
        assert!(passes_over(&rule, 5, loads, last));
        // And the filtered worker is no part of the count: 16 over three
        // is a bound of 15 and a third, and over four it would be 14.
        loads[order[0]] = 15;
        assert!(!passes_over(&rule, 5, loads, last));
    }

    #[test]
    fn a_request_hashed_as_deep_as_its_generation_reads_keeps_its_key() {
        use crate::{
            spec::{AppliedSpec, PolicySpec},
            view::HashKind,
        };
        let mut document = PolicySpec::preset("prefix_hash").expect("a preset of this build");
        document.generation = 1;
        let applied = AppliedSpec::validate(&document).expect("the preset validates");
        let gates = Gates::default();
        let hashes: Vec<u64> = (1..=300).collect();
        // Blocks of the view's size, under it and over it, in both kinds.
        for (hash_kind, block_size) in [
            (HashKind::Token, 0),
            (HashKind::Token, 8),
            (HashKind::Token, 64),
            (HashKind::Text, 0),
            (HashKind::Text, 8),
            (HashKind::Text, 64),
        ] {
            let read = applied.block_hashes_read(hash_kind, block_size);
            let handed = |count: usize| RequestFacts {
                block_hashes: &hashes[..count],
                hash_kind,
                block_size,
                ..RequestFacts::default()
            };
            // The blocks a host is told to hash give the key the whole
            // list gives, and one block fewer gives another: the last one
            // told is the block the key reads.
            let key = prefix_key(256, gates, &handed(hashes.len()));
            let told = prefix_key(256, gates, &handed(read));
            let short = prefix_key(256, gates, &handed(read - 1));
            assert_eq!(told, key, "{hash_kind:?} by {block_size}");
            assert_ne!(short, key, "{hash_kind:?} by {block_size}");
        }
    }

    #[test]
    fn the_prefix_form_declares_the_request_s_hashes() {
        let needs = prefixed(256).needs();
        assert!(needs.ring && needs.token_hashes && needs.text_hashes);
        let plain = ConsistentHash::new(None).needs();
        assert!(plain.ring && !plain.token_hashes && !plain.text_hashes);
    }
}
