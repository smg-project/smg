//! The `consistent_hashing` policy: how it is composed, and its scorer with
//! the ring the scorer owns.
//!
//! The consistent-hash scorer and the ring it owns: the ring
//! owner of a decision's key scores first and the ring's next members after
//! it, so a commit that misses the owner falls to the member the key would
//! walk to. The ring is the engine's, unless the request's caller lends
//! its own ([`LentRing`]). Its members are the worker ids the
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
//! with no routing key is keyed by its prefix: the fold of its block
//! hashes, from the first through its last whole block at or under the
//! form's depth in tokens, started at its cache namespace
//! (`fold_block`). A block hash is of its own block's content, so the fold
//! is what names the prefix: requests that share that much of a prompt
//! share an owner, and two that differ anywhere in it do not. It is the key
//! the sent-prefix predictor keeps that prefix under. The depth counts in
//! the blocks the request was hashed by, text windows at four bytes a
//! token under the text kind. A request that carries no whole block has no
//! key, and the tokens past a request's last whole block are not in its key.
//! On a ring the request's caller lends no block is folded: the prompt's
//! heads are cut at the token ([`HeadRule::heads`]) and the lender says
//! where each stands ([`LentRing::head_position`]).
//!
//! Under the view's boundaries the prefix is keyed at the deepest boundary
//! the request reaches, and the form's depth serves a request that reaches
//! none. A request reaches a boundary by carrying its last whole block. With
//! a bounded-load rule an owner passed over at one boundary hands the
//! request to the owner at the next boundary down, so a prompt that grew
//! past a loaded worker still lands where its head is held.

use std::{cell::RefCell, cmp::Ordering, iter, mem, time::Instant};

use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    document::spec::{BoundedLoadParams, KeySourceSpec, Needs, ProfileSpec, MANUAL_SESSION_HEADER},
    engine::{
        decision::MAX_TARGETS,
        framework::{Scorer, TieBand, DRAW_BOUND},
        view::{Candidate, PoolFacts, RequestFacts, WorkerId},
    },
    policies::{
        cache_aware::Gates,
        prefix_hash::{prefix_key, HeadRule, Prefix},
        standard_filters,
    },
    stages::{lifecycle::Lifecycle, retired_workers::RetiredWorkers},
    tracking::{in_flight::with_counts, lease::Membership},
    Draw,
};

mod lent_ring;

#[cfg(test)]
pub(crate) use lent_ring::InOrder;
pub use lent_ring::{Head, LentRing};

/// How the `consistent_hashing` policy is composed: its filters, its scorers
/// and its picker.
pub(crate) fn profile() -> ProfileSpec {
    // The ring owner of the decision's key (an encode item's content
    // hash, a request's routing key), the ring's next members as the
    // alternates; a request with no session key is placed by its
    // client's key, and one with neither is the picker's draw.
    ProfileSpec {
        filter: standard_filters(),
        score: vec!["consistent-hash".into()],
        pick: "best".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The headers the `consistent_hashing` policy reads a request's routing key
/// from, in order, and behind them the header a request names its worker
/// under ([`target_source`]). `header` makes one source from a header's name
/// and whether the header tells one client from another.
pub(crate) fn key_sources(header: impl Fn(&str, bool) -> KeySourceSpec) -> Vec<KeySourceSpec> {
    vec![
        header(MANUAL_SESSION_HEADER, false),
        header("x-forwarded-for", true),
        header("cookie", true),
        target_source(),
    ]
}

/// The policy's target source: the header a request names its one worker
/// under, by a number that counts the workers of the request's list
/// ([`KeySourceSpec::target`]). The gateway's policy of this name alone
/// reads that header, so no other policy's document names it.
pub(crate) fn target_source() -> KeySourceSpec {
    KeySourceSpec {
        header: Some("x-smg-target-worker".to_string()),
        target: true,
        ..KeySourceSpec::default()
    }
}

/// Positions per member: an even share of the key space per member for the
/// fleets a gateway holds, as the host's ring had.
pub const VIRTUAL_NODES: u32 = 150;

/// The position of one of a member's virtual nodes.
fn position(key: u32, vnode: u32) -> u64 {
    Draw::new(u64::from(key) ^ u64::from(vnode).wrapping_mul(0x9e37_79b9_7f4a_7c15)).next_u64()
}

/// The position a key hashes to.
fn key_position(key: u64) -> u64 {
    Draw::new(key).next_u64()
}

/// What a walk's visitor answers for the member it is shown.
pub enum WalkStep<T> {
    /// Take the member: the walk ends with the value.
    Take(T),
    /// Pass the member over.
    Skip,
    /// Nothing left to find: the walk ends empty.
    Stop,
}

/// The ring: its members, each at [`VIRTUAL_NODES`] sorted positions hashed
/// from the member's stable key and indexed by the handle id; the ids pruned
/// within [`RETIRED_WORKER_TTL`], fenced against relearning from a stale view.
///
/// [`RETIRED_WORKER_TTL`]: crate::stages::RETIRED_WORKER_TTL
#[derive(Debug, Default)]
pub struct Ring {
    /// The members in the order of their ids, by the index the entries
    /// name them by. A pool's rows stand in that order too, so a ring of
    /// one pool names each member by its row's position, through joins
    /// and removals.
    members: Vec<WorkerId>,
    index: FxHashMap<WorkerId, u32>,
    /// `(position, member index)`, sorted by position.
    entries: Vec<(u64, u32)>,
    /// The ids pruned within [`RETIRED_WORKER_TTL`].
    ///
    /// [`RETIRED_WORKER_TTL`]: crate::stages::RETIRED_WORKER_TTL
    fence: RetiredWorkers,
    /// The pools shown whole ([`Ring::show`]) under `shown_epoch`, by their
    /// host's number in rising order, each with whether it may be walked:
    /// its rows stood in the order of their ids, and each was a member
    /// behind the showing.
    shown: Vec<(u32, bool)>,
    /// The newest membership epoch a pool was shown under.
    shown_epoch: u64,
    /// For a member, by its index: the pool that showed it as a row a walk
    /// may take, by the pool's number, [`MANY_POOLS`] where several did
    /// and [`NO_POOL`] where none did. A member whose mark is another
    /// pool's is no row of the pool a walk is taken for. Dropped with the
    /// notes, and never longer than the members.
    shown_by: Vec<u32>,
    /// Whether a member carries the mark of several pools.
    shown_many: bool,
}

/// The mark of a member no pool showed since the notes were last dropped.
const NO_POOL: u32 = u32::MAX;

/// The mark of a member several pools showed: a row of each.
const MANY_POOLS: u32 = u32::MAX - 1;

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
    /// entries, so a pool the ring never saw costs one pass. The joiners
    /// take their places among the members by id, and the members behind
    /// a joiner are renamed in the same passes. Returns how many joined.
    pub fn add_many(&mut self, members: impl IntoIterator<Item = (WorkerId, u32)>) -> usize {
        let mut joiners: Vec<(WorkerId, u32)> = members
            .into_iter()
            .filter(|(id, _)| !self.index.contains_key(id))
            .collect();
        if joiners.is_empty() {
            return 0;
        }
        // An id named twice joins once, under the key named first.
        joiners.sort_by_key(|&(id, _)| id);
        joiners.dedup_by_key(|&mut (id, _)| id);
        let stood = mem::take(&mut self.members);
        self.members.reserve(stood.len() + joiners.len());
        // Each standing member's new index, in the order they stood in.
        let mut to = Vec::with_capacity(stood.len());
        let mut fresh: Vec<(u64, u32)> = Vec::with_capacity(joiners.len() * VIRTUAL_NODES as usize);
        let mut waiting = joiners.iter().peekable();
        for id in stood {
            while let Some(&(joiner, key)) = waiting.next_if(|&&(joiner, _)| joiner < id) {
                self.join(joiner, key, &mut fresh);
            }
            to.push(self.members.len() as u32);
            self.members.push(id);
        }
        for &(joiner, key) in waiting {
            self.join(joiner, key, &mut fresh);
        }
        // A member ahead of the first joiner keeps its index.
        let ahead = to
            .iter()
            .zip(0..)
            .take_while(|&(&at, stood)| at == stood)
            .count();
        for (at, &id) in self.members.iter().enumerate().skip(ahead) {
            self.index.insert(id, at as u32);
        }
        // A mark follows its member.
        if !self.shown_by.is_empty() {
            let mut marks = vec![NO_POOL; self.members.len()];
            for (&mark, &at) in self.shown_by.iter().zip(&to) {
                marks[at as usize] = mark;
            }
            self.shown_by = marks;
        }
        fresh.sort_unstable();
        // The renaming keeps the standing entries' order: a member's new
        // index rises with its old one.
        let renamed = |&(position, member): &(u64, u32)| (position, to[member as usize]);
        let mut merged = Vec::with_capacity(self.entries.len() + fresh.len());
        let (mut i, mut j) = (0, 0);
        while i < self.entries.len() && j < fresh.len() {
            let entry = renamed(&self.entries[i]);
            if entry <= fresh[j] {
                merged.push(entry);
                i += 1;
            } else {
                merged.push(fresh[j]);
                j += 1;
            }
        }
        merged.extend(self.entries[i..].iter().map(renamed));
        merged.extend_from_slice(&fresh[j..]);
        self.entries = merged;
        joiners.len()
    }

    /// One joiner takes the next index and its key's positions.
    fn join(&mut self, id: WorkerId, key: u32, fresh: &mut Vec<(u64, u32)>) {
        let member = self.members.len() as u32;
        self.members.push(id);
        fresh.extend((0..VIRTUAL_NODES).map(|vnode| (position(key, vnode), member)));
    }

    /// [`Ring::add_many`] for the members a decision's view carries at `now`:
    /// an id the lifecycle pruned within [`RETIRED_WORKER_TTL`] is left out,
    /// since the view may be older than the pruning. Returns how many joined.
    ///
    /// [`RETIRED_WORKER_TTL`]: crate::stages::RETIRED_WORKER_TTL
    pub fn learn(
        &mut self,
        members: impl IntoIterator<Item = (WorkerId, u32)>,
        now: Instant,
    ) -> usize {
        self.fence.expire(now);
        let fence = &self.fence;
        let fresh: Vec<(WorkerId, u32)> = members
            .into_iter()
            .filter(|&(id, _)| !fence.holds(id))
            .collect();
        self.add_many(fresh)
    }

    /// [`Ring::learn`] for every row of the pool its host numbers `pool`
    /// under the membership epoch `epoch`, and the note that the pool was
    /// shown, with whether a walk may find its rows: they stood in the
    /// order of their ids (`ordered`) and each is a member now. A row the
    /// fence left out is the whole pass's to place, so its pool is not
    /// walked until it is shown again: at the next epoch, or after the
    /// feed names an id ([`Ring::admit`], [`Ring::remove`],
    /// [`Ring::retain_only`]), which drops every note. The pools of an
    /// earlier epoch are dropped at the first of a later one, and a pool
    /// of an epoch behind the one held is not noted. Each row of a pool a
    /// walk may take is marked with the pool's number, or with the mark of
    /// several pools where another showed it too: the marks stand and
    /// fall with the notes.
    pub fn show(
        &mut self,
        epoch: u64,
        pool: u32,
        ordered: bool,
        members: impl Iterator<Item = (WorkerId, u32)> + Clone,
        now: Instant,
    ) {
        self.learn(members.clone(), now);
        let rows = members.clone();
        let mut members = members;
        let ordered = ordered && members.all(|(id, _)| self.contains(id));
        if epoch < self.shown_epoch {
            return;
        }
        if epoch > self.shown_epoch {
            self.forget_shown();
            self.shown_epoch = epoch;
        }
        match self
            .shown
            .binary_search_by_key(&pool, |&(number, _)| number)
        {
            Ok(at) => self.shown[at].1 = ordered,
            Err(at) => self.shown.insert(at, (pool, ordered)),
        }
        if !ordered {
            return;
        }
        // Each row of a pool a walk may take is marked with the pool, so
        // the walk of another pool passes it by one read.
        self.shown_by.resize(self.members.len(), NO_POOL);
        let mark = pool.min(MANY_POOLS);
        for (id, _) in rows {
            if let Some(member) = self.member_index(id) {
                let by = &mut self.shown_by[member as usize];
                *by = if *by == NO_POOL || *by == mark {
                    mark
                } else {
                    MANY_POOLS
                };
                self.shown_many |= *by == MANY_POOLS;
            }
        }
    }

    /// Drops what the ring was shown: every pool's note and every
    /// member's mark.
    fn forget_shown(&mut self) {
        self.shown.clear();
        self.shown_by.clear();
        self.shown_many = false;
    }

    /// The members' marks by member index, no longer than the members,
    /// and whether one of them is the mark of several pools: where none
    /// is, a member is a pool's by that pool's number alone.
    fn marks(&self) -> (&[u32], bool) {
        (&self.shown_by, self.shown_many)
    }

    /// What [`Ring::show`] noted of the pool numbered `pool` under `epoch`:
    /// whether a walk may find its rows. `None` for a pool not shown yet,
    /// or not since the feed last named an id, and `Some(false)` under an
    /// epoch behind the one held, whose pools are shown no more.
    pub fn shown(&self, epoch: u64, pool: u32) -> Option<bool> {
        match epoch.cmp(&self.shown_epoch) {
            Ordering::Less => Some(false),
            Ordering::Greater => None,
            Ordering::Equal => {
                let at = self
                    .shown
                    .binary_search_by_key(&pool, |&(number, _)| number);
                at.ok().map(|at| self.shown[at].1)
            }
        }
    }

    /// Whether `id` was pruned within [`RETIRED_WORKER_TTL`] of `now` and is
    /// fenced against relearning.
    ///
    /// [`RETIRED_WORKER_TTL`]: crate::stages::RETIRED_WORKER_TTL
    pub fn is_retired(&self, id: WorkerId, now: Instant) -> bool {
        self.fence.is_retired(id, now)
    }

    /// Lift the fence on `id`: the feed named it as a member again.
    pub fn admit(&mut self, id: WorkerId) {
        self.fence.admit(id);
        self.forget_shown();
    }

    /// Remove a member at `now`, fencing its id for [`RETIRED_WORKER_TTL`];
    /// `false` when it was none.
    ///
    /// [`RETIRED_WORKER_TTL`]: crate::stages::RETIRED_WORKER_TTL
    pub fn remove(&mut self, id: WorkerId, now: Instant) -> bool {
        self.fence.retire(id, now);
        self.forget_shown();
        let Some(gone) = self.index.remove(&id) else {
            return false;
        };
        // The members behind the leaver close the gap in their order.
        // Each entry is read once: dropped with the leaver, or renamed
        // where its member stood behind it.
        self.members.remove(gone as usize);
        let mut kept = 0;
        for at in 0..self.entries.len() {
            let (position, member) = self.entries[at];
            if member != gone {
                self.entries[kept] = (position, member - u32::from(member > gone));
                kept += 1;
            }
        }
        self.entries.truncate(kept);
        for at in self.index.values_mut() {
            *at -= u32::from(*at > gone);
        }
        true
    }

    /// Keep the named members only (a snapshot after a resync): the others
    /// leave and are fenced, a named id that was fenced is admitted again,
    /// and a named id the ring does not hold joins from the next view, which
    /// carries its key. A ring that holds no member reads the names for no
    /// leaver, and one that holds no fence admits none.
    pub fn retain_only(&mut self, ids: &[WorkerId], now: Instant) {
        self.forget_shown();
        // A fence a named id holds stood before the pruning, which fences
        // the ids the snapshot does not name.
        let fenced = !self.fence.is_empty();
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
    /// member's new index, where a removal apiece reads every entry once
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
                    self.fence.retire(id, now);
                }
            }
        }
    }

    /// The members at the ring's positions clockwise from the key's, once
    /// around: the one source of a walk's order.
    fn around(&self, key: u64) -> impl Iterator<Item = u32> + '_ {
        let at = key_position(key);
        let start = self.entries.partition_point(|&(p, _)| p < at);
        let (before, from) = self.entries.split_at(start);
        from.iter().chain(before).map(|&(_, member)| member)
    }

    /// Walk clockwise from the key's position, showing `visit` each entry's
    /// member until it takes one or stops; the walk goes once around.
    pub fn walk<T>(&self, key: u64, visit: impl FnMut(u32) -> WalkStep<T>) -> Option<T> {
        walk(self.around(key), visit)
    }
}

/// The walk of a ring: `visit` is shown each of `members` in turn until it
/// takes one or stops.
#[inline(always)]
fn walk<T>(
    members: impl Iterator<Item = u32>,
    mut visit: impl FnMut(u32) -> WalkStep<T>,
) -> Option<T> {
    for member in members {
        match visit(member) {
            WalkStep::Take(value) => return Some(value),
            WalkStep::Skip => {}
            WalkStep::Stop => return None,
        }
    }
    None
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

/// The keys a decision is tried at, in order: a routing key alone, or under
/// the prefix form the request's prefix at each boundary it reaches, the
/// deepest first, and at the form's depth when it reaches none. A fold of
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
/// names it has no rows drawn first and no sample. Its whole pass runs but
/// for a keyed request the plain ring places by itself, which is decided
/// by the workers its key reaches ([`Scorer::own`],
/// [`Scorer::own_by_place`]).
///
/// The ring is the scorer's own, behind a lock its decisions share, unless
/// the request's caller lent the ring it keeps and says where the request
/// stands on it ([`RequestFacts::ring`], [`LentRing`]). The scorer then
/// asks that ring alone: it takes no lock, and its own ring holds no member
/// for the decision and learns none. Under the bounded-load rule the
/// request's places on the lent ring are tried in the order stated, as the
/// keys of a prefix are on the scorer's own.
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

    /// Shows the ring every row of the pool its host numbers `pool` under
    /// `epoch` ([`Ring::show`]), as a whole pass shows it the survivors the
    /// feed never named, with whether the rows stand in the order of their
    /// ids: a walk finds a member's row by that order, and the pool of a
    /// host that hands another is left to the whole pass. Once a pool and
    /// membership epoch: out of line, a decision that finds its pool shown
    /// walks the ring under the one shared lock.
    #[cold]
    #[inline(never)]
    fn learn(&self, epoch: u64, pool: u32, candidates: &[Candidate]) {
        let ordered = candidates.windows(2).all(|pair| pair[0].id < pair[1].id);
        let rows = candidates.iter().map(|row| (row.id, row.key));
        let now = Instant::now();
        self.ring.write().show(epoch, pool, ordered, rows, now);
    }

    /// The prefix form: a request with no routing key is keyed by its
    /// prefix, at the block size `gates` reads for the request. The key is
    /// the fold of its block hashes through its last whole block at the
    /// deepest of `boundaries` it reaches, and at or under `tokens` when
    /// it reaches none.
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
        walk(ring.around(key), |member| match slots[member as usize] {
            FREE => WalkStep::Skip,
            slot => {
                let owner = &candidates[kept[slot as usize] as usize];
                WalkStep::Take(f64::from(facts.in_flight(owner)) > bound)
            }
        })
        .unwrap_or(false)
    }

    /// The keys the decision is tried at under the prefix form, in order;
    /// none for a keyless one. A routing key that leads the prompt stands
    /// alone ([`Self::leading_key`]), and a request that carries none is
    /// tried at its prefix keys, as is one whose key its own id handed.
    /// The prefix form alone asks: the plain ring's one key is read where
    /// it scores.
    fn keys<'a>(&'a self, facts: &'a RequestFacts<'_>) -> Keys<'a> {
        let routing = Self::leading_key(facts);
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
        let members = ring.members();
        // When the survivors stand in member order, each member's slot is
        // its own position.
        if kept.len() == members.len()
            && kept
                .iter()
                .zip(members)
                .all(|(&row, &id)| candidates[row as usize].id == id)
        {
            slots.extend((0..kept.len()).map(|slot| slot as u32));
            return true;
        }
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
        walk(ring.around(key), |member| {
            let slot = slots[member as usize];
            if slot == FREE || slot & TAKEN != 0 {
                return WalkStep::Skip;
            }
            slots[member as usize] = slot | TAKEN;
            out[slot as usize] = 1.0 - found as f32 / MAX_TARGETS as f32;
            found += 1;
            if found == MAX_TARGETS || found == survivors {
                WalkStep::Take(())
            } else {
                WalkStep::Skip
            }
        });
    }
}

impl ConsistentHash {
    /// The scores of a decision whose `slots` are filled, by `ring`: the
    /// first of the decision's keys whose owner keeps it under the
    /// bounded-load rule, with no rule the first, and that key's walk
    /// ranked ([`Self::rank`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "the scorer's inputs, with the ring, the slots and the keys still to try"
    )]
    #[inline(always)]
    fn rank_by(
        &self,
        facts: &RequestFacts<'_>,
        ring: &Ring,
        candidates: &[Candidate],
        kept: &[u32],
        slots: &mut [u32],
        first: u64,
        keys: &mut Option<Keys<'_>>,
        out: &mut [f32],
    ) {
        let bound = self.bound(facts, candidates, kept);
        // The first key whose owner keeps it: with no rule, the first.
        let mut key = Some(first);
        if let Some(bound) = bound {
            while let Some(tried) = key {
                if !Self::passed_over(facts, ring, tried, candidates, kept, slots, bound) {
                    break;
                }
                key = keys.as_mut().and_then(Iterator::next);
            }
        }
        if let Some(key) = key {
            Self::rank(ring, key, kept.len(), slots, out);
        }
    }

    /// The bound of the bounded-load rule over the survivors, where the
    /// spec writes one: the larger of the rule's two margins over the mean
    /// of the survivors' requests in flight.
    fn bound(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
    ) -> Option<f64> {
        self.bounded.map(|(factor, margin)| {
            // Both margins over the mean, the request being placed
            // counted in it: the larger of the two is the bound. The
            // product is taken as written, in 64-bit floats: an owner
            // at the bound's edge stands or falls by its last bit.
            let total: u64 = with_counts!(facts, |counts| kept
                .iter()
                .map(|&k| u64::from(counts(&candidates[k as usize])))
                .sum());
            let mean = (total as f64 + 1.0) / kept.len() as f64;
            (mean * factor).max(mean + f64::from(margin))
        })
    }

    /// The ring the request's caller lent, with where the request stands
    /// on it ([`RequestFacts::ring_positions`]): the first place stated,
    /// and the ones behind it, which the bounded-load rule tries in order.
    /// `None` where none is lent or no place is stated: the scorer's own
    /// ring places.
    fn lent<'a>(facts: &RequestFacts<'a>) -> Option<Lent<'a>> {
        let (&first, behind) = facts.ring_positions.split_first()?;
        Some((facts.ring?, first, behind))
    }

    /// The scores of a request placed on the ring its caller lent, by the
    /// first of its places whose owner keeps it: the first place with no
    /// bounded-load rule, and under one the first whose owner holds no more
    /// requests than the bound ([`Self::bound`]). Where every owner is past
    /// the bound nothing is ranked, as on the scorer's own ring.
    ///
    /// A place is walked once. The survivors its walk reaches are ranked in
    /// the ring's order, the first at `1.0` and each next one a step below,
    /// at most [`MAX_TARGETS`] of them, and the first of them is the
    /// place's owner: the worker the bound tests is the one that leads the
    /// ranking. A worker the ring names is the survivor that carries its
    /// address's id ([`survivor_at`]).
    ///
    /// A ring that reaches no survivor holds no worker yet, or is of
    /// another membership than the rows. Under the plain ring a place's
    /// owner is then the survivor at the position modulo the survivors, in
    /// their list's order, and no other is ranked, so a key keeps one
    /// worker there too: the bound holds that owner as any other. Under
    /// the prefix form such a place has no owner, and where no place has
    /// one nothing is ranked: the stages behind the ring place the
    /// request.
    fn rank_lent(
        &self,
        facts: &RequestFacts<'_>,
        (ring, first, behind): Lent<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) {
        // With no rule the first place is the one read, and no count is.
        let (bound, behind) = match self.bounded {
            Some(_) => (self.bound(facts, candidates, kept), behind),
            None => (None, &[][..]),
        };
        let most = MAX_TARGETS.min(kept.len());
        for position in iter::once(first).chain(behind.iter().copied()) {
            let mut ranked = [usize::MAX; MAX_TARGETS];
            let mut found = 0usize;
            ring.walk(position, &mut |address| {
                if found == most {
                    return true;
                }
                match survivor_at(address, candidates, kept) {
                    // A ring asks about a worker once: one it names again
                    // keeps the rank it has.
                    Some(slot) if !ranked[..found].contains(&slot) => {
                        ranked[found] = slot;
                        found += 1;
                        found == most
                    }
                    _ => false,
                }
            });
            if found == 0 && self.prefix.is_none() {
                if let Some(at) = position.checked_rem(kept.len() as u64) {
                    ranked[0] = at as usize;
                    found = 1;
                }
            }
            let Some(&owner) = ranked[..found].first() else {
                continue;
            };
            let past = bound.is_some_and(|bound| {
                f64::from(facts.in_flight(&candidates[kept[owner] as usize])) > bound
            });
            if past {
                continue;
            }
            for (rank, &slot) in ranked[..found].iter().enumerate() {
                out[slot] = 1.0 - rank as f32 / MAX_TARGETS as f32;
            }
            return;
        }
    }

    /// The key the scorer places a request by where it places it by itself
    /// ([`Scorer::own`]), `0` for none. A client's key is the plain ring's:
    /// the prefix form places a request with no routing key by its prefix,
    /// and one whose key its own id handed too ([`Self::leading_key`]).
    fn own_key(&self, facts: &RequestFacts<'_>) -> u64 {
        match &self.prefix {
            Some(_) => Self::leading_key(facts),
            None => facts.ring_key(),
        }
    }

    /// The routing key that leads a request's prompt under the prefix
    /// form, `0` for none: its session key, where a header or its caller
    /// handed it. A key the request's own id handed
    /// ([`RequestFacts::session_key_from_rid`]) names the session alone,
    /// and the ring places that request by its prompt: the gateway's own
    /// policy of this form hashes a header's key and never the id's
    /// (`model_gateway/src/policies/prefix_hash.rs`, `select_worker_impl`).
    /// The plain ring reads no such word: it places by the id's key as by
    /// any other.
    fn leading_key(facts: &RequestFacts<'_>) -> u64 {
        if facts.session_key_from_rid {
            0
        } else {
            facts.session_key.hash
        }
    }
}

/// What a caller's lent ring places a request by: the ring, the first place
/// the request states on it, and the places behind it
/// ([`ConsistentHash::lent`]).
type Lent<'a> = (&'a dyn LentRing, u64, &'a [u64]);

/// The survivor a lent ring names by `address`, by its place among `kept`:
/// the last of them whose row carries the address's id
/// ([`WorkerId::from_address`]), `None` where none does.
fn survivor_at(address: &str, candidates: &[Candidate], kept: &[u32]) -> Option<usize> {
    let id = WorkerId::from_address(address);
    kept.iter()
        .rposition(|&row| candidates[row as usize].id == id)
}

/// A ring of the pool's rows alone: [`walk_pool`] reads no mark.
const ALONE: u8 = 0;
/// No member is a row of several pools: a member is the pool's by its
/// number alone, in one compare.
const APART: u8 = 1;
/// A member may be a row of several pools, under their one mark.
const SHARED: u8 = 2;

/// The walk of [`ConsistentHash::own`] over a ring whose members `marks`
/// tells apart as `BY` says: the rows met for the pool `number` are asked
/// about and the first of them kept are handed out in the ring's order.
/// One body, built once for each form, so a walk past another pool's
/// members costs each the compares its ring needs. `false` where the walk
/// stopped at its bound or came round short of the rows handed.
fn walk_pool<const BY: u8>(
    ring: &Ring,
    marks: &[u32],
    key: u64,
    number: u32,
    candidates: &[Candidate],
    keep: &mut dyn FnMut(&Candidate) -> bool,
    kept: &mut Vec<u32>,
) -> bool {
    // The members met, kept or refused: a walk meets a member at each
    // of its positions and asks about it once.
    let mut met = [FREE; MAX_TARGETS + DRAW_BOUND as usize];
    let (mut asked, mut refused) = (0, 0);
    let mut members = ring.around(key);
    loop {
        let next = if BY == ALONE {
            members.next()
        } else {
            passed::<BY>(&mut members, marks, number)
        };
        // A walk that came round the ring short of the rows handed met no
        // member for one of them, which the whole pass places.
        let Some(member) = next else {
            return false;
        };
        if met[..asked].contains(&member) {
            continue;
        }
        let id = ring.members().get(member as usize);
        // A ring of one pool names a member by its row's position: the
        // members and the rows both stand in the order of their ids. The
        // row there is read first, its id compared.
        let row = id.and_then(|id| match candidates.get(member as usize) {
            Some(row) if row.id == *id => Some(member as usize),
            _ => candidates.binary_search_by_key(id, |row| row.id).ok(),
        });
        let (Some(row), Some(slot)) = (row, met.get_mut(asked)) else {
            continue;
        };
        *slot = member;
        asked += 1;
        if keep(&candidates[row]) {
            kept.push(row as u32);
            if kept.len() == MAX_TARGETS {
                return true;
            }
        } else {
            refused += 1;
            if refused == DRAW_BOUND {
                return false;
            }
        }
        // Every row handed was asked: no position further on holds
        // another.
        if asked == candidates.len() {
            return !kept.is_empty();
        }
    }
}

/// The next of `members` that may be a row of the pool `number`: its mark
/// is the pool's, or the mark of several pools where `BY` says a ring
/// holds one. A member no pool showed, or another did, is passed. Out of
/// line, a loop of its own: it holds nothing of the walk across a member
/// it passes, so each costs one read and one compare.
#[inline(never)]
fn passed<const BY: u8>(
    members: &mut impl Iterator<Item = u32>,
    marks: &[u32],
    number: u32,
) -> Option<u32> {
    members.find(|&member| {
        let by = marks.get(member as usize);
        by.is_some_and(|&by| by == number || (BY == SHARED && by == MANY_POOLS))
    })
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
            Membership::Relabelled(_) | Membership::Resized(_) | Membership::Left { .. } => {}
        }
    }
}

impl Scorer for ConsistentHash {
    fn name(&self) -> &'static str {
        "consistent-hash"
    }

    // Places by the ring. An owner over its bound of requests in flight is passed over.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    // The ring keyed by no prefix is asked where the routing value's own
    // text stands. The prefix form is asked where a head stands.
    fn places_by_key_text(&self) -> bool {
        self.prefix.is_none()
    }

    fn places_by_heads(&self) -> Option<HeadRule<'_>> {
        let prefix = self.prefix.as_ref()?;
        Some(HeadRule {
            depth: prefix.tokens,
            boundaries: &prefix.boundaries,
        })
    }

    /// The ring places a request that carries a routing key by its walk
    /// alone: under the prefix form too, where a routing key leads the
    /// prefix (`leading_key`). The bounded-load rule reads every
    /// survivor's count, and a request keyed by its prefix is tried at
    /// more keys than one: both read the pool whole.
    fn owns(&self) -> bool {
        self.bounded.is_none()
    }

    /// The walk a whole pass ranks, with no pool read: from the request's
    /// routing key over the ring's members, each member's row found by its
    /// id among the rows, the filters asked about the rows met and no
    /// others, and the first [`MAX_TARGETS`] of them kept handed out in the
    /// ring's order. The walk ends there, or once every row handed was
    /// asked. The targets are the whole pass's, row for row.
    ///
    /// The pool's list of engines is the host's word for what the walk
    /// takes on trust: the rows are a pool's own under a membership epoch,
    /// in the order of their ids, and the pool is numbered. `false`, with
    /// what `kept` holds to be dropped, where the host does not say so,
    /// where the request has no key the ring places by, where the pool was shown
    /// with a row the ring does not hold, once [`DRAW_BOUND`] rows were
    /// refused, which says nothing of the rows behind them, where the walk
    /// came round the ring short of the rows handed, one of which the ring
    /// does not hold: the whole pass decides.
    ///
    /// One ring serves every pool of its profile. The walk passes a member
    /// that another pool showed by one read of its mark, as the whole pass
    /// passes it by one slot, so a pool is walked whatever its share of
    /// the ring.
    ///
    /// `false` too where the stage places by the ring the request's caller
    /// lent ([`RequestFacts::ring`]): the whole pass places by that ring,
    /// and this stage's ring is not read.
    fn own(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
    ) -> bool {
        let key = self.own_key(facts);
        if key == 0 || !self.owns() {
            return false;
        }
        if Self::lent(facts).is_some() {
            return false;
        }
        if pool.membership_epoch == 0 || pool.engines.is_empty() {
            return false;
        }
        let (epoch, number) = (pool.membership_epoch, pool.engines.pool());
        let mut ring = self.ring.read();
        let ordered = ring.shown(epoch, number).or_else(|| {
            // No lock is held across the two.
            RwLockReadGuard::unlocked(&mut ring, || self.learn(epoch, number, candidates));
            ring.shown(epoch, number)
        });
        if ordered != Some(true) {
            return false;
        }
        // One ring serves every pool of its profile, and a walk passes the
        // other pools' members on its way, each by one read of its mark. A
        // ring of this pool's rows alone holds no other to pass.
        let (marks, many) = ring.marks();
        if candidates.len() == ring.len() {
            walk_pool::<ALONE>(&ring, marks, key, number, candidates, keep, kept)
        } else if many {
            walk_pool::<SHARED>(&ring, marks, key, number, candidates, keep, kept)
        } else {
            walk_pool::<APART>(&ring, marks, key, number, candidates, keep, kept)
        }
    }

    /// The walk of the ring the request's caller lent
    /// ([`RequestFacts::ring`]), for a host that reads a row when the walk
    /// asks about it. Each worker the ring names is found by its address's
    /// id, asked of `keep` once, and the first [`MAX_TARGETS`] kept are
    /// handed out in the ring's order: the rows the pass over the whole
    /// pool ranks on that ring, in its order.
    ///
    /// The walk is the plain ring's, from the first place the request
    /// states. The bounded-load rule reads every survivor's count, and the
    /// prefix form states a place for each of its heads: the pass over the
    /// whole pool places under both.
    ///
    /// `0` where the walk decides nothing and the pass over the whole pool
    /// must: under those two forms, where no ring is lent or no place is
    /// stated, and where a row of the pool is left not asked once
    /// [`DRAW_BOUND`] rows were refused, which says nothing of the rows
    /// behind them, or once the ring ended short of [`MAX_TARGETS`] kept.
    /// The walk named no place for that row, and the pass over the whole
    /// pool says what stands behind the ring's workers. A walk that asked
    /// about every row of the pool decides by the rows it kept, whatever
    /// it refused. Eleven rows are asked about at most: three kept and
    /// seven refused ahead of the one that ends the walk.
    fn own_by_place(
        &self,
        facts: &RequestFacts<'_>,
        rows: usize,
        place_of: &dyn Fn(WorkerId) -> Option<u32>,
        keep: &mut dyn FnMut(u32) -> bool,
    ) -> usize {
        if self.bounded.is_some() || self.prefix.is_some() {
            return 0;
        }
        let Some((ring, position, _)) = Self::lent(facts) else {
            return 0;
        };
        // The rows met, kept or refused: a row the ring names again is not
        // asked again.
        let mut met = [FREE; MAX_TARGETS + DRAW_BOUND as usize];
        let (mut asked, mut kept, mut refused) = (0usize, 0usize, 0u32);
        // Whether the walk ended, and whether it ended decided.
        let (mut ended, mut decided) = (false, false);
        ring.walk(position, &mut |address| {
            if ended {
                return true;
            }
            let place = place_of(WorkerId::from_address(address));
            let Some(place) = place.filter(|place| !met[..asked].contains(place)) else {
                return false;
            };
            if let Some(slot) = met.get_mut(asked) {
                *slot = place;
            }
            asked += 1;
            if keep(place) {
                kept += 1;
            } else {
                refused += 1;
            }
            // Every row was asked: the ring names no other.
            decided = kept == MAX_TARGETS || asked == rows;
            ended = decided || refused == DRAW_BOUND;
            ended
        });
        if decided {
            kept
        } else {
            0
        }
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        out.fill(0.0);
        if let Some(lent) = Self::lent(facts) {
            self.rank_lent(facts, lent, candidates, kept, out);
            return TieBand::NORMALIZED;
        }
        // A ring that names no prefix form has one key, the session's or
        // else the client's: it is read here, and the walk is built for the
        // prefix form alone.
        let mut keys = self.prefix.is_some().then(|| self.keys(facts));
        let first = match &mut keys {
            Some(keys) => keys.next(),
            None => Some(facts.ring_key()).filter(|&key| key != 0),
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
            self.rank_by(
                facts, &ring, candidates, kept, &mut slots, first, &mut keys, out,
            );
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
            text_hashes: hashes,
            ..Needs::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{self, AtomicU32};

    use super::*;
    use crate::{
        engine::{
            view,
            view::{fold_block, Health, Identity},
        },
        stages::retired_workers::RETIRED_WORKER_TTL,
        tracking::prefix_predictor,
    };

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
        ring.walk(key, |member| {
            WalkStep::Take(ring.members()[member as usize])
        })
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

    /// The ring holds `standing` in the order of their ids, each under its
    /// index, and owns every key as a ring that learned them at once does.
    fn stands_in_order(ring: &Ring, standing: &[u64]) {
        let mut ids: Vec<WorkerId> = standing.iter().map(|&id| WorkerId(id)).collect();
        ids.sort();
        assert_eq!(ring.members(), ids);
        for (at, &id) in ids.iter().enumerate() {
            assert_eq!(ring.member_index(id), Some(at as u32), "{id:?}");
        }
        assert_eq!(ring.entries.len(), ids.len() * VIRTUAL_NODES as usize);
        assert!(ring.entries.windows(2).all(|pair| pair[0] <= pair[1]));
        let mut at_once = Ring::new();
        at_once.add_many(ids.iter().map(|&id| (id, key_of(id))));
        for key in 1..=500 {
            assert_eq!(owner_of(ring, key), owner_of(&at_once, key), "key {key}");
        }
    }

    #[test]
    fn the_members_stand_in_the_order_of_their_ids_through_joins_and_removals() {
        let now = Instant::now();
        let mut ring = Ring::new();
        let joiners = |ids: &[u64]| -> Vec<(WorkerId, u32)> {
            ids.iter()
                .map(|&id| (WorkerId(id), key_of(WorkerId(id))))
                .collect()
        };
        assert_eq!(ring.add_many(joiners(&[1_004, 1_002, 1_006, 1_002])), 3);
        stands_in_order(&ring, &[1_002, 1_004, 1_006]);
        // One joiner between two members, then one ahead of them all and
        // one behind.
        assert!(ring.add(WorkerId(1_003), 1_003));
        stands_in_order(&ring, &[1_002, 1_003, 1_004, 1_006]);
        assert_eq!(ring.add_many(joiners(&[1_007, 1_001])), 2);
        stands_in_order(&ring, &[1_001, 1_002, 1_003, 1_004, 1_006, 1_007]);
        // A leaver from the middle, the front and the end, then two at once.
        for (gone, left) in [
            (1_004, &[1_001, 1_002, 1_003, 1_006, 1_007][..]),
            (1_001, &[1_002, 1_003, 1_006, 1_007][..]),
            (1_007, &[1_002, 1_003, 1_006][..]),
        ] {
            assert!(ring.remove(WorkerId(gone), now));
            stands_in_order(&ring, left);
        }
        ring.retain_only(&[WorkerId(1_003)], now);
        stands_in_order(&ring, &[1_003]);
    }

    #[test]
    fn an_id_named_twice_joins_under_the_key_named_first() {
        let mut twice = Ring::new();
        assert_eq!(twice.add_many([(WorkerId(9), 1), (WorkerId(9), 2)]), 1);
        let mut once = Ring::new();
        assert_eq!(once.add_many([(WorkerId(9), 1)]), 1);
        assert_eq!(twice.entries, once.entries);
    }

    #[test]
    fn a_mark_follows_its_member_when_a_joiner_takes_a_place_ahead_of_it() {
        let now = Instant::now();
        let mut ring = Ring::new();
        let rows = |ids: &[u64]| -> Vec<(WorkerId, u32)> {
            ids.iter()
                .map(|&id| (WorkerId(id), key_of(WorkerId(id))))
                .collect()
        };
        // The second pool's rows join ahead of and between the first's.
        let pools = [rows(&[1_002, 1_004]), rows(&[1_001, 1_003])];
        for (number, pool) in pools.iter().enumerate() {
            ring.show(1, number as u32, true, pool.iter().copied(), now);
        }
        let (marks, many) = ring.marks();
        assert!(!many, "no member is a row of both pools");
        for (number, pool) in pools.iter().enumerate() {
            for &(id, _) in pool {
                let member = ring.member_index(id).expect("a member");
                assert_eq!(marks[member as usize], number as u32, "{id:?}");
            }
        }
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
        let reversed: Vec<Candidate> = pool.iter().rev().copied().collect();
        assert_eq!(
            scored(&scorer, &keyed(11), &reversed),
            scores.iter().rev().copied().collect::<Vec<_>>()
        );
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
    /// joins from its view once [`RETIRED_WORKER_TTL`] has passed, and a stale
    /// view within the window relearns nothing.
    #[test]
    fn the_fence_lapses_after_its_window_and_a_reused_id_joins_from_the_view() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_002);
        assert!(ring.remove(id, t0));
        let soon = t0 + RETIRED_WORKER_TTL / 2;
        assert!(ring.is_retired(id, soon));
        assert_eq!(
            ring.learn([(id, key_of(id))], soon),
            0,
            "fenced within the window"
        );
        let later = t0 + RETIRED_WORKER_TTL;
        assert!(!ring.is_retired(id, later), "the fence has lapsed");
        assert_eq!(
            ring.learn([(id, key_of(id))], later),
            1,
            "the reused id joins"
        );
        assert!(ring.contains(id));
        // A second retirement fences it afresh, from its own instant.
        assert!(ring.remove(id, later));
        assert!(ring.is_retired(id, later + RETIRED_WORKER_TTL / 2));
        assert!(!ring.is_retired(id, later + RETIRED_WORKER_TTL));
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
        let again = t0 + RETIRED_WORKER_TTL / 2;
        assert!(!ring.remove(id, again));
        let first_lapsed = t0 + RETIRED_WORKER_TTL;
        assert_eq!(
            ring.learn([(id, key_of(id))], first_lapsed),
            0,
            "the later retirement stands"
        );
        assert!(ring.is_retired(id, first_lapsed));
        let both_lapsed = again + RETIRED_WORKER_TTL;
        assert_eq!(ring.learn([(id, key_of(id))], both_lapsed), 1);
        assert!(ring.contains(id));
        // Many ids retired at one instant lapse together and each is
        // learned again.
        let many: Vec<WorkerId> = (2_000..2_512).map(WorkerId).collect();
        for &gone in &many {
            ring.remove(gone, both_lapsed);
        }
        let pool = many.iter().map(|&gone| (gone, key_of(gone)));
        assert_eq!(
            ring.learn(pool.clone(), both_lapsed + RETIRED_WORKER_TTL / 2),
            0
        );
        assert_eq!(
            ring.learn(pool, both_lapsed + RETIRED_WORKER_TTL),
            many.len()
        );
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
        let again = t0 + RETIRED_WORKER_TTL / 2;
        assert!(ring.remove(id, again));
        let first_lapsed = t0 + RETIRED_WORKER_TTL;
        assert_eq!(ring.learn([(id, key_of(id))], first_lapsed), 0);
        assert!(ring.is_retired(id, first_lapsed));
        assert_eq!(
            ring.learn([(id, key_of(id))], again + RETIRED_WORKER_TTL),
            1
        );
    }

    /// Two retirements of one id told out of order: the later instant holds
    /// the fence, whichever the ring heard last.
    #[test]
    fn the_later_of_two_retirements_told_out_of_order_holds_the_fence() {
        let mut ring = ring_of(4);
        let t0 = Instant::now();
        let id = WorkerId(1_002);
        let later = t0 + RETIRED_WORKER_TTL / 2;
        assert!(ring.remove(id, later));
        // The earlier instant arrives second and takes nothing from the
        // fence.
        assert!(!ring.remove(id, t0));
        assert!(
            ring.is_retired(id, t0 + RETIRED_WORKER_TTL),
            "the earlier instant won"
        );
        assert_eq!(ring.learn([(id, key_of(id))], t0 + RETIRED_WORKER_TTL), 0);
        assert_eq!(
            ring.learn([(id, key_of(id))], later + RETIRED_WORKER_TTL),
            1
        );
    }

    /// Two ids retired out of their clocks' order: each lapses at its own
    /// instant, and a listing's read of the fence agrees with the clock's
    /// on both.
    #[test]
    fn a_retirement_told_behind_a_later_one_lapses_at_its_own_instant() {
        let t0 = Instant::now();
        let (late, early) = (WorkerId(1), WorkerId(2));
        let mut fence = RetiredWorkers::default();
        fence.retire(late, t0 + RETIRED_WORKER_TTL / 2);
        fence.retire(early, t0);
        for now in [
            t0 + RETIRED_WORKER_TTL / 2,
            t0 + RETIRED_WORKER_TTL,
            t0 + RETIRED_WORKER_TTL + RETIRED_WORKER_TTL / 4,
            t0 + RETIRED_WORKER_TTL + RETIRED_WORKER_TTL / 2,
        ] {
            fence.expire(now);
            for id in [late, early] {
                assert_eq!(
                    fence.holds(id),
                    fence.is_retired(id, now),
                    "{id:?} {:?} after the first clock read",
                    now - t0
                );
            }
        }
        assert!(fence.is_empty());
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
        assert_eq!(at_once.learn(pool.clone(), now + RETIRED_WORKER_TTL / 2), 0);
        assert_eq!(at_once.learn(pool, now + RETIRED_WORKER_TTL), 8);
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

    /// The key of a prefix of these blocks under no cache namespace, as
    /// the ring folds it.
    fn through(hashes: &[u64]) -> u64 {
        hashes.iter().fold(0, |key, &hash| fold_block(key, hash))
    }

    fn prefixed(tokens: u32) -> ConsistentHash {
        ConsistentHash::new(None).keyed_by_prefix(tokens, Gates::default(), &[])
    }

    #[test]
    fn the_prefix_key_folds_the_blocks_through_the_last_whole_one_under_the_depth() {
        let hashes = [11, 22, 33, 44, 55, 66];
        let gates = Gates::default();
        let key = |tokens: u32| prefix_key(tokens, gates, &hashed(&hashes));
        // Blocks of 16 tokens: a depth of 64 tokens ends with the fourth
        // block, and a depth inside a block ends with the block before it.
        assert_eq!(key(64), through(&hashes[..4]));
        assert_eq!(key(79), through(&hashes[..4]));
        assert_eq!(key(80), through(&hashes[..5]));
        // A request shorter than the depth is keyed through its own last
        // block. A request with no whole block has no key, and neither
        // has a depth under one block.
        assert_eq!(key(256), through(&hashes));
        assert_eq!(prefix_key(256, gates, &hashed(&[])), 0);
        assert_eq!(prefix_key(15, gates, &hashed(&hashes)), 0);
        // The block size the request was counted with leads the view's.
        let learned = RequestFacts {
            block_size: 32,
            ..hashed(&hashes)
        };
        assert_eq!(prefix_key(64, gates, &learned), through(&hashes[..2]));
        // Under the text kind the depth is in windows at four bytes a
        // token: 64 tokens are 256 bytes, sixteen windows of sixteen.
        let windows: Vec<u64> = (1..=20).collect();
        let text = RequestFacts {
            hash_kind: view::HashKind::Text,
            ..hashed(&windows)
        };
        assert_eq!(prefix_key(64, gates, &text), through(&windows[..16]));
    }

    /// A key names its whole prefix. Two requests that differ in their
    /// first block alone and share every block behind it take two keys, at
    /// the depth and at a boundary: nothing one prompt left in a cache
    /// serves the other. Equal prefixes take one key under any tail, and
    /// the same blocks in another cache namespace are another prefix.
    #[test]
    fn requests_that_differ_in_their_first_block_alone_take_two_keys() {
        let one = [11, 22, 33, 44, 55];
        let other = [12, 22, 33, 44, 55];
        let tailed = [11, 22, 33, 44, 99, 77];
        let gates = Gates::default();
        let at_depth = |hashes: &[u64]| prefix_key(64, gates, &hashed(hashes));
        assert_ne!(at_depth(&one), at_depth(&other));
        assert_eq!(at_depth(&one), at_depth(&tailed));
        // The boundary at 64 tokens ends with the fourth block.
        let scorer = levelled(None);
        let at_boundary = |hashes: &[u64]| keys_of(&scorer, &hashed(hashes));
        assert_ne!(at_boundary(&one), at_boundary(&other));
        assert_eq!(at_boundary(&one), at_boundary(&tailed));
        let apart = RequestFacts {
            namespace_seed: 9,
            ..hashed(&one)
        };
        assert_ne!(prefix_key(64, gates, &apart), at_depth(&one));
        assert_ne!(keys_of(&scorer, &apart), at_boundary(&one));
    }

    /// The ring's key of a prefix is the key the sent-prefix predictor
    /// keeps that prefix under: one fold, one number.
    #[test]
    fn a_prefix_has_one_key_on_the_ring_and_in_the_predictor() {
        let hashes = [11, 22, 33, 44, 55, 66];
        let facts = RequestFacts {
            namespace_seed: 9,
            ..hashed(&hashes)
        };
        let mut keys = Vec::new();
        prefix_predictor::PrefixPredictor::fold_into(&mut keys, facts.namespace_seed, &hashes);
        assert_eq!(prefix_key(64, Gates::default(), &facts), keys[3]);
        assert_eq!(keys_of(&levelled(None), &facts), [keys[3]]);
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
        // The key is the fold of the four blocks: a routing key of that
        // value has the same owner under the plain form.
        let plain = ConsistentHash::new(None);
        let key = through(&one[..4]);
        assert_eq!(leader(&scored(&plain, &keyed(key), &pool)), owner);
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

    /// The gateway's own policy of the prefix form hashes a key a header
    /// handed (`model_gateway/src/policies/prefix_hash.rs:358`) and places
    /// every other request by its prompt (`:360-372`): it never reads the
    /// key the request's own id hands. Its plain ring reads that key
    /// itself (`model_gateway/src/policies/consistent_hashing.rs:171`).
    #[test]
    fn a_key_the_requests_own_id_handed_leads_no_prefix_and_the_plain_form_places_by_it() {
        let pool: Vec<Candidate> = (0..8).map(|i| candidate(i, 0)).collect();
        let hashes = [11, 22, 33, 44];
        let plain = ConsistentHash::new(None);
        let unkeyed = scored(&prefixed(64), &hashed(&hashes), &pool);
        let by_prefix = leader(&unkeyed);
        // A routing key whose owner is another worker than the prefix's.
        let (key, by_key) = (1..)
            .map(|key| (key, leader(&scored(&plain, &keyed(key), &pool))))
            .find(|&(_, owner)| owner != by_prefix)
            .expect("a key with another owner");
        // A header handed the key: it leads the prefix.
        let by_header = RequestFacts {
            session_key: Identity::hashed(key),
            ..hashed(&hashes)
        };
        assert_eq!(leader(&scored(&prefixed(64), &by_header, &pool)), by_key);
        assert_eq!(keys_of(&levelled(None), &by_header), [key]);
        assert_eq!(prefixed(64).own_key(&by_header), key);
        // The request's own id handed it: the request scores as the same
        // prompt with no key, is tried at that prompt's keys, and the
        // stage places it by no walk of its own.
        let by_id = RequestFacts {
            session_key_from_rid: true,
            ..by_header
        };
        assert_eq!(scored(&prefixed(64), &by_id, &pool), unkeyed);
        assert_eq!(
            keys_of(&levelled(None), &by_id),
            keys_of(&levelled(None), &hashed(&hashes))
        );
        assert_eq!(prefixed(64).own_key(&by_id), 0);
        // The plain form reads no such word: the id's key has its owner.
        assert_eq!(leader(&scored(&plain, &by_id, &pool)), by_key);
        assert_eq!(plain.own_key(&by_id), key);
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
        use crate::tracking::in_flight::tests::Lent;
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
            document::spec::{AppliedSpec, PolicySpec},
            engine::view::HashKind,
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
            // told is the block the key ends with.
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
        assert!(needs.ring && needs.token_hashes && needs.text_hashes && !needs.text);
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
            document::spec::{AppliedSpec, PolicySpec},
            engine::view::HashKind,
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
        let (deep, shallow) = (through(&hashes[..16]), through(&hashes[..4]));
        assert_eq!(keys_of(&scorer, &hashed(&hashes)), [deep, shallow]);
        assert_eq!(keys_of(&scorer, &hashed(&hashes[..15])), [shallow]);
        assert_eq!(keys_of(&scorer, &hashed(&hashes[..4])), [shallow]);
        // A request that reaches no boundary is keyed at the form's depth,
        // at its own last whole block when it is shorter. A request that
        // reaches a boundary is not keyed at the depth: 128 tokens end with
        // the 8th block.
        assert_eq!(
            keys_of(&scorer, &hashed(&hashes[..3])),
            [through(&hashes[..3])]
        );
        assert!(keys_of(&scorer, &hashed(&[])).is_empty());
        // A boundary under one block is reached by no request.
        let thin = ConsistentHash::new(None).keyed_by_prefix(128, Gates::default(), &[8, 64]);
        assert_eq!(keys_of(&thin, &hashed(&hashes)), [shallow]);
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
            hash_kind: view::HashKind::Text,
            ..hashed(&hashes)
        };
        assert_eq!(keys_of(&scorer, &text), [deep]);
    }

    #[test]
    fn an_owner_passed_over_hands_the_key_to_the_next_boundary_down_and_then_to_nobody() {
        let pool: Vec<Candidate> = (0..4).map(|i| candidate(i, 0)).collect();
        let plain = ConsistentHash::new(None);
        let ranks = |key: u64| scored(&plain, &keyed(key), &pool);
        // A request whose two boundaries' keys two workers own: the fold
        // through its 16th block, and the fold through its 4th.
        let blocks = |last: u64| {
            let mut hashes = [100u64; 16];
            hashes[15] = last;
            hashes
        };
        let shallow = through(&blocks(0)[..4]);
        let hashes = (1..)
            .map(blocks)
            .find(|hashes| leader(&ranks(through(hashes))) != leader(&ranks(shallow)))
            .expect("a sixteenth block that hands the deep key to another worker");
        let deep = through(&hashes);
        let (deep_owner, shallow_owner) = (leader(&ranks(deep)), leader(&ranks(shallow)));
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

    #[test]
    fn a_ring_keeps_the_pools_it_was_shown_by_number_while_their_epoch_stands() {
        let mut ring = Ring::new();
        let now = Instant::now();
        assert_eq!(ring.shown(3, 0), None);
        for pool in 0..9 {
            let member = (WorkerId(u64::from(pool) + 1), pool);
            ring.show(3, pool, pool != 4, [member].into_iter(), now);
        }
        // Nine pools under one epoch: none forgotten, each with its word.
        for pool in 0..9 {
            assert_eq!(ring.shown(3, pool), Some(pool != 4), "pool {pool}");
        }
        assert_eq!(ring.shown(3, 9), None, "a pool not shown");
        assert_eq!(ring.len(), 9);
        // A later epoch: its pools are shown anew, and the earlier one's
        // are shown no more.
        assert_eq!(ring.shown(4, 0), None);
        ring.show(4, 0, true, [].into_iter(), now);
        assert_eq!(ring.shown(4, 0), Some(true));
        assert_eq!(ring.shown(4, 1), None);
        assert_eq!(ring.shown(3, 1), Some(false), "an epoch behind");
        // A pool of a passed epoch is learned and not noted.
        ring.show(3, 7, true, [(WorkerId(77), 77)].into_iter(), now);
        assert!(ring.contains(WorkerId(77)));
        assert_eq!(ring.shown(3, 7), Some(false));
        assert_eq!(ring.shown(4, 7), None);
        // The feed names an id: every note is dropped. A pool shown while
        // the fence leaves a row of it out is not walked, until the feed
        // names the row again and the pool is shown with it.
        let row = [(WorkerId(1), 0)];
        ring.remove(WorkerId(1), now);
        assert_eq!(ring.shown(4, 0), None, "a tell drops the notes");
        ring.show(4, 0, true, row.into_iter(), now);
        assert!(!ring.contains(WorkerId(1)));
        assert_eq!(ring.shown(4, 0), Some(false), "a row left out");
        ring.admit(WorkerId(1));
        assert_eq!(ring.shown(4, 0), None);
        ring.show(4, 0, true, row.into_iter(), now);
        assert_eq!(ring.shown(4, 0), Some(true));
        ring.retain_only(&[WorkerId(1)], now);
        assert_eq!(ring.shown(4, 0), None, "a snapshot drops the notes");
    }

    #[test]
    fn a_ring_marks_each_member_with_the_pool_that_showed_it_while_its_notes_stand() {
        let now = Instant::now();
        let rows = |ids: std::ops::Range<u64>| ids.map(|n| (WorkerId(n), n as u32));
        let by = |ring: &Ring, n: u64| {
            let member = ring.member_index(WorkerId(n)).expect("a member");
            let mark = ring.marks().0.get(member as usize);
            mark.copied().unwrap_or(NO_POOL)
        };
        let mut ring = Ring::new();
        ring.show(1, 0, true, rows(0..4), now);
        assert!(!ring.marks().1, "no member of two pools yet");
        ring.show(1, 3, true, rows(2..6), now);
        assert!(ring.marks().1, "two members of two pools");
        assert_eq!([by(&ring, 0), by(&ring, 1)], [0, 0]);
        assert_eq!([by(&ring, 2), by(&ring, 3)], [MANY_POOLS; 2], "of both");
        assert_eq!([by(&ring, 4), by(&ring, 5)], [3, 3]);
        // A pool no walk may take marks nothing, and one shown again
        // keeps its mark.
        ring.show(1, 7, false, rows(6..8), now);
        ring.show(1, 0, true, rows(0..4), now);
        assert_eq!([by(&ring, 6), by(&ring, 0)], [NO_POOL, 0]);
        // Each word that drops the notes drops the marks: a later epoch,
        // and the three tells of the feed that name an id.
        let drops: [fn(&mut Ring, Instant); 4] = [
            |ring, now| ring.show(2, 9, true, [(WorkerId(8), 8)].into_iter(), now),
            |ring, _| ring.admit(WorkerId(100)),
            |ring, now| {
                ring.remove(WorkerId(5), now);
            },
            |ring, now| ring.retain_only(&[WorkerId(0), WorkerId(8)], now),
        ];
        for (epoch, dropped) in (1..).zip(drops) {
            ring.show(epoch, 0, true, rows(0..1), now);
            assert_eq!(by(&ring, 0), 0);
            dropped(&mut ring, now);
            assert_eq!(by(&ring, 0), NO_POOL, "drop {epoch}");
            assert!(!ring.marks().1, "drop {epoch}");
        }
        assert_eq!(by(&ring, 8), NO_POOL, "a member the snapshot kept");
    }

    /// A caller's ring that reaches the workers at `places` of a list in
    /// that order: the worker at place `n` is registered at `w<n>`.
    fn reaches(places: &[u32]) -> InOrder {
        InOrder(places.iter().map(|place| format!("w{place}")).collect())
    }

    /// The rows of a caller's own list: each carries its address's id.
    fn addressed(count: u32) -> Vec<Candidate> {
        (0..count)
            .map(|place| Candidate {
                id: WorkerId::from_address(&format!("w{place}")),
                ..candidate(place, 0)
            })
            .collect()
    }

    /// A keyed request whose caller lends `ring` and says where it stands
    /// on it.
    fn lent<'a>(ring: &'a InOrder, positions: &'a [u64]) -> RequestFacts<'a> {
        RequestFacts {
            ring: Some(ring),
            ring_positions: positions,
            ..keyed(7)
        }
    }

    /// The rows a lent ring ranked among `kept` for a request that stands
    /// at its position `at`, best first.
    fn ranked(ring: &InOrder, at: u64, rows: &[Candidate], kept: &[u32]) -> Vec<u32> {
        let scorer = ConsistentHash::new(None);
        let scores = scored_among(&scorer, &lent(ring, &[at]), rows, kept);
        assert!(scorer.ring().is_empty(), "its own ring learned a row");
        let mut ranked: Vec<(f32, u32)> = scores.into_iter().zip(kept.iter().copied()).collect();
        ranked.retain(|&(score, _)| score > 0.0);
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        ranked.into_iter().map(|(_, row)| row).collect()
    }

    /// The plain ring places a key on the ring its caller lent: the first
    /// four survivors the ring's walk reaches from the position its caller
    /// states, in the ring's order. A worker that is no row or no survivor
    /// is passed. The first of them is the one worker the gateway's own
    /// policy takes, the first eligible worker its ring reaches
    /// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to 115
    /// at 882be6f4f). Of two rows at one address the last kept takes the
    /// ring's place, as that policy finds a ring's worker from its list's
    /// end (line 103). A client's key is placed alike. The scorer's own ring
    /// is neither read nor taught, and its walk over written rows leaves
    /// the request to the whole pass.
    #[test]
    fn a_lent_ring_ranks_the_survivors_its_walk_reaches_in_the_rings_order() {
        let mut rows = addressed(8);
        let ring = reaches(&[5, 9, 2, 0, 7, 3, 1, 6, 4]);
        let all = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(ranked(&ring, 0, &rows, &all), [5, 2, 0, 7]);
        assert_eq!(ranked(&ring, 5, &rows, &all), [3, 1, 6, 4]);
        assert_eq!(ranked(&ring, 0, &rows, &[0, 1, 3, 4, 6, 7]), [0, 7, 3, 1]);
        assert_eq!(ranked(&ring, 0, &rows, &[4, 6]), [6, 4]);
        let scorer = ConsistentHash::new(None);
        let client = RequestFacts {
            session_key: Identity::NONE,
            client_key: 9,
            ..lent(&ring, &[0])
        };
        let scores = scored(&scorer, &client, &rows);
        let steps = (scores[5], scores[2], scores[0], scores[7]);
        assert_eq!(steps, (1.0, 0.75, 0.5, 0.25));
        let (places, ends): (Vec<u32>, Vec<u32>) = ((0..8).collect(), (1..=8).collect());
        let pool = PoolFacts {
            membership_epoch: 3,
            engines: view::Engines::new(1, &places, &ends),
            ..PoolFacts::default()
        };
        let keep = &mut |_: &Candidate| true;
        assert!(!scorer.own(&lent(&ring, &[3]), &pool, &rows, keep, &mut Vec::new()));
        assert!(scorer.ring().is_empty(), "no member learned");
        // The prefix form with no rule leaves such a request to the whole
        // pass alike.
        let unruled = ConsistentHash::new(None).keyed_by_prefix(64, Gates::default(), &[]);
        assert!(!unruled.own(&lent(&ring, &[3]), &pool, &rows, keep, &mut Vec::new()));
        assert!(unruled.ring().is_empty(), "no member learned");
        // With no place stated the scorer's own ring places, and learns
        // the rows. Under a bounded-load rule and under the prefix form a
        // lent ring places too, and theirs learns none.
        let own = ConsistentHash::new(None);
        scored(&own, &lent(&ring, &[]), &rows);
        assert_eq!(own.ring().len(), 8);
        for own in [bounded(1.25, 10), prefixed(64)] {
            scored(&own, &lent(&ring, &[3]), &rows);
            assert!(own.ring().is_empty(), "its own ring learned a row");
        }
        rows.push(Candidate {
            position: 8,
            ..rows[4]
        });
        assert_eq!(ranked(&ring, 0, &rows, &[4, 6, 8]), [6, 8]);
    }

    /// A lent ring that reaches no survivor places nothing by its walk: it
    /// holds no worker yet, or is of another membership than the rows, or
    /// its workers are not kept. The survivor at the key's position modulo
    /// the survivors, in their list's order, then takes the key, and no
    /// other is ranked. It is the gateway's own rule for a key no ring
    /// places (`model_gateway/src/policies/consistent_hashing.rs`, line 139
    /// at 882be6f4f: the eligible worker at `hash_position(key)` modulo
    /// their count), so a key keeps one worker there too.
    #[test]
    fn a_lent_ring_that_reaches_no_survivor_places_by_the_position_modulo_the_survivors() {
        let rows = addressed(8);
        let scorer = ConsistentHash::new(None);
        for ring in [reaches(&[]), reaches(&[20, 21]), reaches(&[3])] {
            for (position, kept) in [
                (13_u64, vec![0, 1, 2, 4, 5, 6, 7]),
                (13, vec![1, 4, 6]),
                (u64::MAX, vec![2, 5]),
            ] {
                let at = [position];
                let scores = scored_among(&scorer, &lent(&ring, &at), &rows, &kept);
                let owner = (position % kept.len() as u64) as usize;
                let alone: Vec<f32> = (0..kept.len())
                    .map(|slot| f32::from(u8::from(slot == owner)))
                    .collect();
                assert_eq!(scores, alone, "{ring:?} at {position} over {kept:?}");
            }
        }
        assert!(scorer.ring().is_empty(), "no member learned");
    }

    /// [`InOrder`], with a count of the workers its walks named.
    #[derive(Debug)]
    struct Counted(InOrder, AtomicU32);

    impl LentRing for Counted {
        fn position(&self, key: &str) -> u64 {
            self.0.position(key)
        }

        fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
            self.0.walk(position, &mut |address| {
                self.1.fetch_add(1, atomic::Ordering::Relaxed);
                take(address)
            });
        }
    }

    /// A lent ring's walk is asked no further than its answer needs. It
    /// ends at the last survivor it ranks, four at most, and over no
    /// survivor at the first worker the ring names. With either stop gone
    /// no score moves: the walk names one worker past its answer, or goes
    /// round the whole ring for nobody.
    #[test]
    fn a_lent_rings_walk_ends_where_its_answer_stands() {
        let rows = addressed(8);
        let ring = Counted(reaches(&[5, 9, 2, 0, 7, 3, 1, 6, 4]), AtomicU32::new(0));
        let named_under = |scorer: &ConsistentHash, kept: &[u32]| {
            ring.1.store(0, atomic::Ordering::Relaxed);
            let facts = RequestFacts {
                ring: Some(&ring),
                ring_positions: &[0],
                ..keyed(7)
            };
            scored_among(scorer, &facts, &rows, kept);
            ring.1.load(atomic::Ordering::Relaxed)
        };
        let named = |kept: &[u32]| named_under(&ConsistentHash::new(None), kept);
        // The fourth survivor is the ring's fifth worker: `w9` is no row.
        assert_eq!(named(&[0, 1, 2, 3, 4, 5, 6, 7]), 5);
        // Two survivors, the ring's first and third workers.
        assert_eq!(named(&[2, 5]), 3);
        // Two survivors the ring names last: the walk goes round once.
        assert_eq!(named(&[4, 6]), 9);
        assert_eq!(named(&[]), 1, "nobody to rank");
        // Under a bounded-load rule a place is walked once too: its owner
        // is the first survivor of the walk that ranks.
        let rule = bounded(1.25, 10);
        assert_eq!(named_under(&rule, &[0, 1, 2, 3, 4, 5, 6, 7]), 5);
    }

    /// A ring stage that has placed keys by its own ring asks a lent ring
    /// all the same: what it learned of the rows names no worker for a
    /// request whose caller lends its ring.
    #[test]
    fn a_stage_that_placed_by_its_own_ring_places_by_a_lent_one() {
        let rows = addressed(8);
        let scorer = ConsistentHash::new(None);
        scored(&scorer, &keyed(7), &rows);
        assert_eq!(scorer.ring().len(), 8, "its own ring holds the rows");
        let ring = reaches(&[5, 9, 2, 0, 7, 3, 1, 6, 4]);
        let scores = scored(&scorer, &lent(&ring, &[0]), &rows);
        let steps = (scores[5], scores[2], scores[0], scores[7]);
        assert_eq!(steps, (1.0, 0.75, 0.5, 0.25));
    }

    /// Under a bounded-load rule the places a request states on the lent
    /// ring are tried in the order stated: the first whose owner, the first
    /// survivor the ring's walk reaches from there, holds no more requests
    /// than the bound places the request, with the ring's next survivors
    /// behind that owner. The owner's requests are read through the
    /// request's facts. Where every owner is past the bound nothing is
    /// ranked, and the stages behind the ring place the request. With no
    /// rule the first place is the one read. A ring that reaches no
    /// survivor ranks nothing under the prefix form, as the gateway's own
    /// policy takes the fewest requests where no ring places a prompt
    /// (`model_gateway/src/policies/prefix_hash.rs`, lines 338 to 340 at
    /// 882be6f4f). Under the plain ring a place's owner is then the
    /// survivor at its position modulo their count, held to the bound as
    /// any owner.
    ///
    /// The gateway's own policy tries a prompt's heads in that order, each
    /// at its ring's first eligible worker, and takes the first under the
    /// same bound (the same file, lines 291 to 319 and 236 to 247). With
    /// every one past it as it counted them, it takes the worker with the
    /// fewest requests (lines 321 to 331), as the stage behind the ring
    /// does under the stock document.
    #[test]
    fn a_lent_rings_places_are_tried_in_order_under_the_bounded_load_rule() {
        // Rows 5 and 2 hold 40 and 30 requests and the others none: the
        // mean is 71 / 8 and the bound 18.875.
        let mut rows = addressed(8);
        rows[5].in_flight = 40;
        rows[2].in_flight = 30;
        let ring = reaches(&[5, 2, 0, 7, 3, 1, 6, 4]);
        let all = [0, 1, 2, 3, 4, 5, 6, 7];
        let tried_on = |ring: &InOrder, scorer: &ConsistentHash, at: &[u64], kept: &[u32]| {
            let scores = scored_among(scorer, &lent(ring, at), &rows, kept);
            assert!(scorer.ring().is_empty(), "its own ring learned a row");
            let mut ranked: Vec<(f32, u32)> =
                scores.into_iter().zip(kept.iter().copied()).collect();
            ranked.retain(|&(score, _)| score > 0.0);
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            ranked.into_iter().map(|(_, row)| row).collect::<Vec<u32>>()
        };
        let tried =
            |scorer: &ConsistentHash, at: &[u64], kept: &[u32]| tried_on(&ring, scorer, at, kept);
        let rule = bounded(1.25, 10);
        // The owners at the places 0 and 1 are rows 5 and 2, past the
        // bound. The owner at 2 is row 0.
        assert_eq!(tried(&rule, &[0, 1, 2], &all), [0, 7, 3, 1]);
        assert_eq!(tried(&rule, &[2, 0], &all), [0, 7, 3, 1]);
        assert!(tried(&rule, &[0, 1], &all).is_empty());
        // The places behind the first are tried in the order stated: the
        // owners at 2 and 3 are rows 0 and 7, both under the bound.
        assert_eq!(tried(&rule, &[0, 2, 3], &all), [0, 7, 3, 1]);
        assert_eq!(tried(&rule, &[0, 3, 2], &all), [7, 3, 1, 6]);
        // The owner's requests are the ones the request's facts state: in
        // a table lent beside rows that each read zero, row 5 holds 40.
        {
            use crate::tracking::in_flight::tests::Lent;
            let mut bare = rows.clone();
            let counts = Lent::take(&mut bare);
            let facts = lent(&ring, &[0, 2]);
            let scores = scored_among(&rule, &counts.beside(&facts), &bare, &all);
            assert_eq!((scores[5], scores[0]), (0.0, 1.0));
        }
        // Row 5 is no survivor: the owner at 0 is the ring's next, row 2.
        assert_eq!(tried(&rule, &[0, 3], &[0, 1, 2, 3, 4, 6, 7]), [7, 3, 1, 6]);
        // With no rule the first place is read, whatever its owner holds.
        assert_eq!(
            tried(&ConsistentHash::new(None), &[0, 1, 2], &all),
            [5, 2, 0, 7]
        );
        // The prefix form tries its places alike, and a ring that reaches
        // no survivor ranks nothing under it.
        let gates = Gates::default();
        let prefix = bounded(1.25, 10).keyed_by_prefix(64, gates, &[]);
        assert_eq!(tried(&prefix, &[0, 1, 2], &all), [0, 7, 3, 1]);
        assert_eq!(tried(&prefix, &[0], &[1, 3]), [3, 1]);
        // A ring that reaches no survivor. Under the prefix form no place
        // has an owner. Under the plain ring the owners at 13 and 14 are
        // rows 5 and 6, by the position modulo the eight: row 5 is past the
        // bound and passed over, and with no rule it takes the key.
        let elsewhere = reaches(&[20, 21]);
        assert!(tried_on(&elsewhere, &prefix, &[13, 14], &all).is_empty());
        assert!(tried_on(&elsewhere, &rule, &[13], &all).is_empty());
        assert_eq!(tried_on(&elsewhere, &rule, &[13, 14], &all), [6]);
        let plain = ConsistentHash::new(None);
        assert_eq!(tried_on(&elsewhere, &plain, &[13, 14], &all), [5]);
    }

    /// The walk by place of `facts`' lent ring over `rows` rows in their
    /// list's order, the worker `w<n>` at place `n`, where the filters
    /// refuse the places in `refused`: the places kept, in the order asked,
    /// and how many the walk says it hands out.
    fn walked(facts: &RequestFacts<'_>, rows: u32, refused: &[u32]) -> (Vec<u32>, usize) {
        let ids: Vec<WorkerId> = addressed(rows).iter().map(|row| row.id).collect();
        let place_of = |id: WorkerId| ids.iter().position(|&own| own == id).map(|at| at as u32);
        let (mut kept, mut asked) = (Vec::new(), Vec::new());
        let handed =
            ConsistentHash::new(None).own_by_place(facts, rows as usize, &place_of, &mut |place| {
                assert!(!asked.contains(&place), "place {place} asked twice");
                asked.push(place);
                let keep = !refused.contains(&place);
                if keep {
                    kept.push(place);
                }
                keep
            });
        (kept, handed)
    }

    /// A lent ring walked by place hands out the rows the pass over the
    /// whole pool ranks, in its order, and asks about no row behind them:
    /// four kept end it, and so does the pool's last row. It decides
    /// nothing, and the whole pass must, once [`DRAW_BOUND`] rows were
    /// refused, where the ring ends with a row of the pool not asked,
    /// where no ring is lent or no place is stated, and under a
    /// bounded-load rule or the prefix form, which ask about no row.
    #[test]
    fn a_lent_ring_walked_by_place_hands_out_the_rows_the_whole_pass_ranks() {
        let ring = reaches(&[5, 9, 2, 0, 7, 3, 1, 1, 6, 4]);
        let facts = lent(&ring, &[0]);
        let rows = addressed(8);
        for refused in [&[][..], &[5, 2], &[5, 2, 0, 7, 3], &[6, 4, 1]] {
            let kept: Vec<u32> = (0..8).filter(|row| !refused.contains(row)).collect();
            let (places, handed) = walked(&facts, 8, refused);
            assert_eq!(places, ranked(&ring, 0, &rows, &kept), "{refused:?}");
            assert_eq!(handed, places.len(), "{refused:?}");
        }
        // Eight refused say nothing of the rows behind them. Seven do.
        let long = reaches(&(0..20).collect::<Vec<u32>>());
        let facts = lent(&long, &[0]);
        assert_eq!(walked(&facts, 20, &[0, 1, 2, 3, 4, 5, 6, 7]).1, 0);
        let kept = (vec![7, 8, 9, 10], MAX_TARGETS);
        assert_eq!(walked(&facts, 20, &[0, 1, 2, 3, 4, 5, 6]), kept);
        // Eight refused that are the pool's last rows leave none not
        // asked: the one row kept is the pool's answer.
        let nine = reaches(&(0..9).collect::<Vec<u32>>());
        let refused = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(walked(&lent(&nine, &[0]), 9, &refused), (vec![0], 1));
        // A refused row the ring names again is not asked again, and is
        // counted once.
        let again = reaches(&[0, 1, 1, 2]);
        assert_eq!(walked(&lent(&again, &[0]), 3, &[1]), (vec![0, 2], 2));
        // A ring of two of three rows names no third: the whole pass says
        // what stands behind its two.
        let short = reaches(&[2, 0]);
        assert_eq!(walked(&lent(&short, &[0]), 3, &[]), (vec![2, 0], 0));
        assert_eq!(walked(&lent(&short, &[0]), 3, &[1]), (vec![2, 0], 0));
        // No place stated, no ring lent: no row is asked.
        for facts in [lent(&ring, &[]), keyed(7)] {
            assert_eq!(walked(&facts, 8, &[]), (vec![], 0));
        }
        let asks: &mut dyn FnMut(u32) -> bool = &mut |_| panic!("a row was asked");
        for scorer in [bounded(1.25, 10), prefixed(64)] {
            assert_eq!(
                scorer.own_by_place(&lent(&ring, &[0]), 8, &|_| Some(0), asks),
                0
            );
        }
    }
}
