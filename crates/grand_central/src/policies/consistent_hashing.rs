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
    tracking::lease::Membership,
    Draw,
};

mod lent_ring;

#[cfg(test)]
pub(crate) use lent_ring::InOrder;
pub use lent_ring::LentRing;

pub use crate::policies::prefix_hash::Head;

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
#[non_exhaustive]
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
            let total: u64 = kept
                .iter()
                .map(|&k| u64::from(facts.in_flight(&candidates[k as usize])))
                .sum();
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
            ring.walk_ids(position, &mut |id| {
                if found == most {
                    return true;
                }
                match survivor_at(id, candidates, kept) {
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

/// The survivor a lent ring names by its address's `id`
/// ([`LentRing::walk_ids`]), by its place among `kept`: the last of them
/// whose row carries the id, `None` where none does.
fn survivor_at(id: WorkerId, candidates: &[Candidate], kept: &[u32]) -> Option<usize> {
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
        ring.walk_ids(position, &mut |id| {
            if ended {
                return true;
            }
            let place = place_of(id);
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
#[path = "../../tests/unit/policies/consistent_hashing/tests.rs"]
mod tests;
