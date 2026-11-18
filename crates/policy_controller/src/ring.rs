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
//!
//! Under the view's boundaries the prefix is keyed at the deepest boundary
//! the request reaches, and the form's depth serves a request that reaches
//! none. A request reaches a boundary by carrying its last whole block. With
//! a bounded-load rule an owner passed over at one boundary hands the
//! request to the owner at the next boundary down, so a prompt that grew
//! past a loaded worker still lands where its head is held.

use std::{
    cell::RefCell,
    collections::VecDeque,
    mem,
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
    /// carries its key. A ring that holds no member reads the names for no
    /// leaver, and one that holds no fence admits none.
    pub fn retain_only(&mut self, ids: &[WorkerId], now: Instant) {
        // A fence a named id holds stood before the pruning, which fences
        // the ids the snapshot does not name.
        let fenced = !self.fenced.is_empty();
        if !self.members.is_empty() {
            let keep: FxHashSet<WorkerId> = ids.iter().copied().collect();
            self.prune(&keep, now);
        }
        if fenced {
            for &id in ids {
                self.admit(id);
            }
        }
    }

    /// Remove at `now` every member `keep` does not hold, fencing each. One
    /// leaver goes as [`Ring::remove`] takes it. Several go in one pass:
    /// each entry is read once, dropped with its member or renamed to its
    /// member's new index, where a removal apiece reads every entry twice
    /// for each leaver.
    fn prune(&mut self, keep: &FxHashSet<WorkerId>, now: Instant) {
        /// The mark of a member that leaves, in place of its new index.
        const LEFT: u32 = u32::MAX;
        let gone: Vec<WorkerId> = self
            .members
            .iter()
            .copied()
            .filter(|member| !keep.contains(member))
            .collect();
        match gone.as_slice() {
            [] => {}
            &[one] => {
                self.remove(one, now);
            }
            _ => {
                // Each member's index once the leavers are out, in the
                // order the members stand in.
                let mut next = 0;
                let to: Vec<u32> = self
                    .members
                    .iter()
                    .map(|member| {
                        if keep.contains(member) {
                            next += 1;
                            next - 1
                        } else {
                            LEFT
                        }
                    })
                    .collect();
                // Each kept entry is written once, at its new place and
                // under its member's new index: renamed where it stands
                // and then moved, the move would wait on the write.
                let mut kept = 0;
                for at in 0..self.entries.len() {
                    let (position, member) = self.entries[at];
                    let member = to[member as usize];
                    self.entries[kept] = (position, member);
                    kept += usize::from(member != LEFT);
                }
                self.entries.truncate(kept);
                self.index.retain(|_, at| {
                    *at = to[*at as usize];
                    *at != LEFT
                });
                let mut at = 0;
                self.members.retain(|_| {
                    at += 1;
                    to[at - 1] != LEFT
                });
                for &id in &gone {
                    self.retire(id, now);
                }
            }
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

/// The prefix form of the ring: how a request with no routing key is keyed.
#[derive(Debug)]
struct Prefix {
    /// The depth in tokens such a request is keyed at when it reaches no
    /// boundary.
    tokens: u32,
    /// The gates that hold the view's block size.
    gates: Gates,
    /// The view's boundaries in tokens, ascending.
    boundaries: Box<[u32]>,
}

impl Prefix {
    /// The request's prefix at `boundary`, where it reaches it: the hash of
    /// the boundary's last whole block. A request reaches a boundary by
    /// carrying that block.
    fn at(&self, boundary: u32, facts: &RequestFacts<'_>) -> Option<u64> {
        let blocks = self.gates.whole_blocks(boundary, facts) as usize;
        facts.block_hashes.get(blocks.checked_sub(1)?).copied()
    }
}

/// The keys a decision is tried at, in order: a routing key alone, or under
/// the prefix form the request's prefix at each boundary it reaches, the
/// deepest first, and at the form's depth when it reaches none. A hash of
/// zero is no key. One decision walks them, and only past a passed owner,
/// so the walk reads a boundary when its turn comes and holds no list.
struct Keys<'a> {
    facts: &'a RequestFacts<'a>,
    /// The routing key until it is handed out, `0` for none.
    routing: u64,
    /// The prefix form until its keys are out; none under a routing key.
    prefix: Option<&'a Prefix>,
    /// The form's boundaries not read yet, the deepest last.
    below: &'a [u32],
    /// Whether the request reached a boundary read so far.
    reached: bool,
}

impl Iterator for Keys<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.routing != 0 {
            return Some(mem::take(&mut self.routing));
        }
        let prefix = self.prefix?;
        while let Some((&boundary, below)) = self.below.split_last() {
            self.below = below;
            if let Some(key) = prefix.at(boundary, self.facts) {
                self.reached = true;
                if key != 0 {
                    return Some(key);
                }
            }
        }
        self.prefix = None;
        if self.reached {
            return None;
        }
        Some(prefix_key(prefix.tokens, prefix.gates, self.facts)).filter(|&key| key != 0)
    }
}

/// The consistent-hash scorer: the ring owner of the decision's key first,
/// the ring's next members after it, every other survivor at zero; a keyless
/// decision scores every survivor equal, so the profile's next stage places
/// it, or the picker's draw. With a bounded-load rule an owner whose
/// in-flight count is past both `factor` times the survivors' mean and
/// `abs_margin` over it, the mean counting the request being placed, is
/// passed over. A prefix keyed at a boundary is then tried at the next
/// boundary down, and once no owner keeps the key the decision is scored as
/// a keyless one. The ring's next members follow an owner that keeps its
/// key, and they take the key of an owner that is no survivor, a worker that
/// is down or was tried.
///
/// The owner must be among the rows scored, so the scorer answers for the
/// pool whole and never row by row ([`Scorer::row_wise`]): a profile that
/// names it runs its whole pass, with no rows drawn first and no sample.
#[derive(Debug)]
pub struct ConsistentHash {
    /// `(factor, abs_margin)` of the bounded-load rule, when the spec writes
    /// one.
    bounded: Option<(f64, u32)>,
    /// The prefix form, for a request with no routing key.
    prefix: Option<Prefix>,
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
    /// prefix, at the block size `gates` reads for the request. The key is
    /// the hash of its last whole block at the deepest of `boundaries` it
    /// reaches, and at or under `tokens` when it reaches none.
    #[must_use]
    pub fn keyed_by_prefix(mut self, tokens: u32, gates: Gates, boundaries: &[u32]) -> Self {
        self.prefix = Some(Prefix {
            tokens,
            gates,
            boundaries: boundaries.into(),
        });
        self
    }

    /// Whether the key's owner among the survivors holds more requests than
    /// `bound`. `slots` is as [`Self::fill`] left it.
    fn passed_over(
        facts: &RequestFacts<'_>,
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
                Probe::Take(f64::from(facts.in_flight(owner)) > bound)
            }
        })
        .unwrap_or(false)
    }

    /// The keys the decision is tried at, in order; none for a keyless one.
    /// A routing key stands alone, and under the prefix form a request that
    /// carries none is tried at its prefix keys.
    fn keys<'a>(&'a self, facts: &'a RequestFacts<'_>) -> Keys<'a> {
        let routing = facts.session_key.hash;
        let prefix = self.prefix.as_ref().filter(|_| routing == 0);
        Keys {
            facts,
            routing,
            prefix,
            below: prefix.map(|prefix| &*prefix.boundaries).unwrap_or_default(),
            reached: false,
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
        // A ring that names no prefix form has one key, the routing key: it
        // is read here, and the walk is built for the prefix form alone.
        let mut keys = self.prefix.is_some().then(|| self.keys(facts));
        let first = match &mut keys {
            Some(keys) => keys.next(),
            None => Some(facts.session_key.hash).filter(|&key| key != 0),
        };
        let first = match first {
            Some(key) if !kept.is_empty() => key,
            _ => return TieBand::NORMALIZED,
        };
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
                // Both margins over the mean, the request being placed
                // counted in it: the larger of the two is the bound. The
                // product is taken as written, in 64-bit floats: an owner
                // at the bound's edge stands or falls by its last bit.
                let total: u64 = kept
                    .iter()
                    .map(|&k| u64::from(facts.in_flight(&candidates[k as usize])))
                    .sum();
                let mean = (total as f64 + 1.0) / kept.len() as f64;
                (mean * factor).max(mean + f64::from(margin))
            });
            // The first key whose owner keeps it: with no rule, the first.
            let mut key = Some(first);
            if let Some(bound) = bound {
                while let Some(tried) = key {
                    if !Self::passed_over(facts, &ring, tried, candidates, kept, &slots, bound) {
                        break;
                    }
                    key = keys.as_mut().and_then(Iterator::next);
                }
            }
            if let Some(key) = key {
                Self::rank(&ring, key, kept.len(), &mut slots, out);
            }
        });
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        // The prefix form reads the request's block hashes in the kind the
        // host hashes by: over its token ids, or over windows of its text.
        let hashes = self.prefix.is_some();
        Needs {
            ring: true,
            token_hashes: hashes,
            text: hashes,
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

    /// Two retirements of one id told out of order: the later instant holds
    /// the fence, whichever the ring heard last.
    #[test]
    fn the_later_of_two_retirements_told_out_of_order_holds_the_fence() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_002);
        let later = t0 + FENCE / 2;
        assert!(ring.remove(id, later));
        // The earlier instant arrives second and takes nothing from the
        // fence.
        assert!(!ring.remove(id, t0));
        assert!(ring.is_retired(id, t0 + FENCE), "the earlier instant won");
        assert_eq!(ring.learn([(id, key_of(id))], t0 + FENCE), 0);
        assert_eq!(ring.learn([(id, key_of(id))], later + FENCE), 1);
    }

    /// A snapshot that drops several members leaves the ring a removal
    /// apiece leaves: every key's owner, each leaver fenced, the indices
    /// dense.
    #[test]
    fn a_snapshot_prunes_several_members_as_a_removal_apiece_would() {
        let now = Instant::now();
        let all = ids(12);
        let stays = |id: &WorkerId| id.0.is_multiple_of(3);
        let kept: Vec<WorkerId> = all.iter().copied().filter(stays).collect();
        let mut apiece = ring_of(12);
        for &id in all.iter().filter(|&id| !stays(id)) {
            assert!(apiece.remove(id, now));
        }
        let mut at_once = ring_of(12);
        at_once.retain_only(&kept, now);
        assert_eq!(at_once.members(), kept, "the members keep their order");
        for key in 0..2_000 {
            assert_eq!(owner(&at_once, key), owner(&apiece, key), "key {key}");
        }
        for (i, &member) in at_once.members().iter().enumerate() {
            assert_eq!(at_once.member_index(member), Some(i as u32));
        }
        assert_eq!(at_once.index.len(), kept.len());
        assert_eq!(at_once.entries, {
            let mut alone = Ring::new();
            alone.add_many(kept.iter().map(|&id| (id, key_of(id))));
            alone.entries
        });
        for &id in &all {
            assert_eq!(at_once.contains(id), stays(&id), "{id:?}");
            assert_eq!(at_once.is_retired(id, now), !stays(&id), "{id:?}");
        }
        // Each leaver's fence lapses as a removal's does.
        let pool = all.iter().map(|&id| (id, key_of(id)));
        assert_eq!(at_once.learn(pool.clone(), now + FENCE / 2), 0);
        assert_eq!(at_once.learn(pool, now + FENCE), 8);
        // One leaver goes as a removal takes it.
        let mut one = ring_of(3);
        one.retain_only(&[WorkerId(1_000), WorkerId(1_002)], now);
        assert_eq!(one.members(), [WorkerId(1_000), WorkerId(1_002)]);
        assert!(one.is_retired(WorkerId(1_001), now));
    }

    /// A snapshot told to a ring with no member fences nothing and adds
    /// nothing: the named join from the view. A ring a removal emptied
    /// still holds its fence, and naming the id lifts it.
    #[test]
    fn a_snapshot_told_to_a_ring_with_no_member_still_lifts_a_fence() {
        let now = Instant::now();
        let named = ids(4);
        let mut ring = Ring::new();
        ring.retain_only(&named, now);
        assert!(ring.is_empty());
        assert!(named.iter().all(|&id| !ring.is_retired(id, now)));
        assert_eq!(ring.learn(named.iter().map(|&id| (id, key_of(id))), now), 4);

        let mut ring = ring_of(1);
        assert!(ring.remove(WorkerId(1_000), now));
        assert!(ring.is_empty() && ring.is_retired(WorkerId(1_000), now));
        ring.retain_only(&named, now);
        assert!(!ring.is_retired(WorkerId(1_000), now), "named and fenced");
        assert_eq!(ring.learn([(WorkerId(1_000), 1_000)], now), 1);
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
        ConsistentHash::new(None).keyed_by_prefix(tokens, Gates::default(), &[])
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
        use crate::counts::tests::Lent;
        let plain = ConsistentHash::new(None);
        let mut pool: Vec<Candidate> = (0..loads.len() as u32).map(|i| candidate(i, 0)).collect();
        let ranks = scored(&plain, &keyed(5), &pool);
        let mut order: Vec<usize> = (0..pool.len()).collect();
        order.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));
        for (&place, &load) in order.iter().zip(loads) {
            pool[place].in_flight = load;
        }
        // The loads stand in a table lent beside the rows, each row reading
        // zero: the rule reads them through the request's facts.
        let lent = Lent::take(&mut pool);
        let scores = scored(rule, &lent.beside(&keyed(5)), &pool);
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
        assert!(needs.ring && needs.token_hashes && needs.text && needs.text_hashes);
        let plain = ConsistentHash::new(None).needs();
        assert!(plain.ring && !plain.token_hashes && !plain.text && !plain.text_hashes);
    }

    /// The prefix form at a depth of 128 tokens under the boundaries 64 and
    /// 256.
    fn levelled(rule: Option<&BoundedLoadParams>) -> ConsistentHash {
        ConsistentHash::new(rule).keyed_by_prefix(128, Gates::default(), &[64, 256])
    }

    fn keys_of(scorer: &ConsistentHash, facts: &RequestFacts<'_>) -> Vec<u64> {
        scorer.keys(facts).collect()
    }

    #[test]
    fn a_request_hashed_as_deep_as_its_generation_reads_keeps_its_boundary_keys() {
        use crate::{
            spec::{AppliedSpec, PolicySpec},
            view::HashKind,
        };
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","view":{"boundaries":[64,256]},"profiles":{"p":{"score":[{"consistent-hash":{"prefix_tokens":128}}],"pick":"best"}}}"#,
        )
        .expect("a well-formed document");
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        let scorer = levelled(None);
        let hashes: Vec<u64> = (1..=300).collect();
        for (hash_kind, block_size) in [
            (HashKind::Token, 0),
            (HashKind::Token, 8),
            (HashKind::Token, 64),
            (HashKind::Text, 0),
            (HashKind::Text, 64),
        ] {
            let read = applied.block_hashes_read(hash_kind, block_size);
            let handed = |count: usize| RequestFacts {
                block_hashes: &hashes[..count],
                hash_kind,
                block_size,
                ..RequestFacts::default()
            };
            // Both boundaries' keys, as the whole list gives them: the
            // blocks a host is told to hash give the same two, and one
            // block fewer loses the deepest.
            let keys = keys_of(&scorer, &handed(hashes.len()));
            assert_eq!(keys.len(), 2, "{hash_kind:?} by {block_size}");
            assert_eq!(
                keys_of(&scorer, &handed(read)),
                keys,
                "{hash_kind:?} by {block_size}"
            );
            assert_ne!(
                keys_of(&scorer, &handed(read - 1)),
                keys,
                "{hash_kind:?} by {block_size}"
            );
        }
    }

    #[test]
    fn under_boundaries_a_prefix_is_keyed_at_each_one_it_reaches_the_deepest_first() {
        let hashes: Vec<u64> = (1..=20).collect();
        let scorer = levelled(None);
        // Blocks of 16 tokens: the boundaries end with the 16th block and
        // with the 4th. A request that reaches both is tried at both, and
        // one that reaches the shallower alone is tried there.
        assert_eq!(keys_of(&scorer, &hashed(&hashes)), [16, 4]);
        assert_eq!(keys_of(&scorer, &hashed(&hashes[..15])), [4]);
        assert_eq!(keys_of(&scorer, &hashed(&hashes[..4])), [4]);
        // A request that reaches no boundary is keyed at the form's depth,
        // at its own last whole block when it is shorter. A request that
        // reaches a boundary is not keyed at the depth: 128 tokens end with
        // the 8th block.
        assert_eq!(keys_of(&scorer, &hashed(&hashes[..3])), [3]);
        assert!(keys_of(&scorer, &hashed(&[])).is_empty());
        // A boundary under one block is reached by no request.
        let thin = ConsistentHash::new(None).keyed_by_prefix(128, Gates::default(), &[8, 64]);
        assert_eq!(keys_of(&thin, &hashed(&hashes)), [4]);
        // A routing key stands alone, and without the form nothing else
        // keys a request.
        let routed = RequestFacts {
            block_hashes: &hashes,
            ..keyed(5)
        };
        assert_eq!(keys_of(&scorer, &routed), [routed.session_key.hash]);
        assert!(keys_of(&ConsistentHash::new(None), &hashed(&hashes)).is_empty());
        // Under the text kind a boundary counts in windows at four
        // characters a token: 64 tokens end with the 16th window, and 256
        // with the 64th, which this request does not carry.
        let text = RequestFacts {
            hash_kind: crate::view::HashKind::Text,
            ..hashed(&hashes)
        };
        assert_eq!(keys_of(&scorer, &text), [16]);
    }

    #[test]
    fn an_owner_passed_over_hands_the_key_to_the_next_boundary_down_and_then_to_nobody() {
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let plain = ConsistentHash::new(None);
        let ranks = |key: u64| scored(&plain, &keyed(key), &pool);
        // Two block hashes that two workers own: the request's 16th block
        // and its 4th.
        let deep = 7;
        let shallow = (8..)
            .find(|&key| leader(&ranks(key)) != leader(&ranks(deep)))
            .expect("a key another worker owns");
        let (deep_owner, shallow_owner) = (leader(&ranks(deep)), leader(&ranks(shallow)));
        let mut hashes = [100u64; 16];
        hashes[15] = deep;
        hashes[3] = shallow;
        let under = |scorer: &ConsistentHash, deep_load: u32, shallow_load: u32| {
            let mut pool = pool.clone();
            pool[deep_owner].in_flight = deep_load;
            pool[shallow_owner].in_flight = shallow_load;
            scored(scorer, &hashed(&hashes), &pool)
        };
        let rule = BoundedLoadParams {
            factor: Some(1.25),
            abs_margin: Some(10),
        };
        let bounded = levelled(Some(&rule));
        // Under the bound the deeper boundary's owner keeps the key, and the
        // ranks are that key's.
        assert_eq!(under(&bounded, 0, 0), ranks(deep));
        assert_eq!(under(&bounded, 13, 0), ranks(deep));
        // Past it the request goes to the shallower boundary's owner, and
        // the ranks are that key's.
        assert_eq!(under(&bounded, 14, 0), ranks(shallow));
        // Both owners past the bound: nobody is scored.
        let nobody = under(&bounded, 30, 30);
        assert!(nobody.iter().all(|&score| score == 0.0), "{nobody:?}");
        // With no rule the deeper boundary's owner keeps the key whatever
        // it holds.
        assert_eq!(under(&levelled(None), 30, 30), ranks(deep));
    }
}
