//! The `bucket` policy: how it is composed, and its scorer: prefill placement
//! by request size.
//!
//! For each model the scorer keeps a partition of the request sizes among
//! the pool's workers. Each listed worker owns a range of sizes, in the
//! order the workers were listed, and the last range is open-ended. A
//! prefill decision scores every row of the worker that owns its prompt's
//! size first, so requests of like size meet on one worker. When the owner
//! is no survivor of the decision, the nearest listed survivor below it in
//! the partition takes the request, else the nearest above.
//!
//! The partition is learned from a window of bookings. Every issued
//! decision books its tokens on the worker it names
//! ([`Lifecycle::on_decision_issued`]), and a first dispatch on another
//! worker books them there too ([`Lifecycle::on_dispatched`]): both
//! bookings stand until each ages. A booking leaves the window once it is
//! older than `adjust_interval_s`. A cut falls due at each interval
//! and is made where a worker's share of the booked tokens moved by more
//! than a factor of two since the last cut, or the listing changed: the
//! ranges are cut again from the sizes in the window, so that each worker's
//! range carries an equal share of the booked tokens. The cut reads the
//! decision's clock ([`RequestFacts::now_micros`]): a model's first prefill
//! decision starts the cadence, and the first decision at or past an
//! interval's end makes the cut before it is placed. No thread and no timer
//! runs.
//!
//! The balance rule spills. When the most and the least booked survivors
//! differ by more than `abs_threshold` tokens and the most booked holds more
//! than `rel_threshold` times the least, the decision scores the least
//! booked survivor first, whatever its size.
//!
//! The listing follows the view: a survivor the model's partition does not
//! list joins at its first decision, with every other row of that view, and
//! a changed listing splits [`SPAN`] evenly among all of it until the next
//! cut. The lifecycle prunes: a removed worker's range goes to the range
//! below it, and a replaced handle keeps its predecessor's place. A new
//! incarnation that left a model is unlisted there by a second word: until
//! it the new handle is listed under that model, no view shows it there,
//! its range goes to the seated owner below and no request is placed by
//! it. A worker a registration took off a model is unlisted under that
//! model alone. A handle a view listed before the word of its replacement
//! keeps its own place and takes its predecessor's load and bookings, and
//! the predecessor's place leaves as a removed worker's does. A pruned
//! worker is fenced for a while
//! ([`RETIRED_WORKER_TTL`](crate::stages::retired_workers::RETIRED_WORKER_TTL)):
//! a view captured before its leave still shows it, and a decision over such a
//! view lists it nowhere, seats it nowhere and places the others. The feed
//! naming the worker again lifts the fence at once. A worker that left a model
//! is not fenced: it serves on, and the fence names no model, so a view older
//! than the word may list it there again. All of it is soft state, which a new
//! process learns again within one window, and so does a newly applied
//! document, whose generation builds its own scorer.
//!
//! A model's partition stands behind its own guard. A decision finds its
//! model's by the name's hash, in a map replaced whole at a model's first
//! prefill decision, and takes that guard alone where its view matches the
//! listing; a booking takes the guards of the partitions that list its
//! worker. Deciders of two models share no lock there. A decision that
//! lists holds the stage's one fence for that pass over its view's rows
//! and lets it go before it seats. A lease names no model, so a worker
//! two models list is booked in both partitions. A model whose last worker
//! left takes its partition with it: the table keeps no partition that lost
//! its listing.
//!
//! Any other phase scores every survivor equal, and so does a prefill
//! decision whose size no listed survivor owns a range for: the picker's
//! draw places those.

use std::{cmp::Ordering, collections::VecDeque, sync::Arc, time::Instant};

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    document::spec::{BucketParams, Needs, ProfileSpec},
    engine::{
        decision::{Decision, Target},
        framework::{Scorer, TieBand},
        view::{Candidate, Phase, RequestFacts, WorkerId},
    },
    policies::standard_filters,
    stages::{lifecycle::Lifecycle, retired_workers::RetiredWorkers},
    tracking::lease::{EventKind, EventSet, Lease, Membership, Side},
};

/// How the `bucket` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    // Prefill placement by request size: the worker whose range of
    // sizes holds the prompt, the least booked survivor when the
    // balance rule spills, each at the stage's defaults; any other
    // phase is the picker's draw.
    ProfileSpec {
        filter: standard_filters(),
        score: vec!["bucket".into()],
        pick: "best".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The span of sizes a changed listing splits evenly among its workers, in
/// tokens. The last range is open-ended whatever the span, and the next cut
/// replaces the split with ranges learned from the sizes seen.
pub const SPAN: u32 = 4096;

/// No member: a booking whose worker left the listing.
const NONE: u32 = u32::MAX;

/// The most bookings a window holds. The host's clock bounds the window; a
/// clock that stands still would keep every booking, so past this many the
/// oldest leaves as if its interval had passed.
const WINDOW_CAP: usize = 1 << 20;

/// The size the policy places and books a request by: the count of its
/// token ids, and where it states none the length of its text in
/// characters. The classic `bucket` sizes a request so
/// (`model_gateway/src/policies/bucket.rs:106-110`).
const fn sized_by(tokens: u32, chars: u32) -> u32 {
    if tokens == 0 {
        chars
    } else {
        tokens
    }
}

/// The absolute part of the balance rule, in booked tokens, when the
/// document names none.
pub const DEFAULT_ABS_THRESHOLD: u32 = 64;

/// The relative part of the balance rule when the document names none.
pub const DEFAULT_REL_THRESHOLD: f32 = 1.5;

/// The bookings' window and the cut's cadence, in seconds, when the document
/// names none.
pub const DEFAULT_ADJUST_INTERVAL_S: u32 = 5;

/// The bucket scorer: the owner of the request's size first and every other
/// survivor at zero, or the least booked survivor first when the balance
/// rule spills. See the module's notes for the partition and its window.
#[derive(Debug)]
pub struct Bucket {
    /// The absolute part of the balance rule, in booked tokens.
    abs_threshold: u64,
    /// The relative part of the balance rule.
    rel_threshold: f32,
    /// The window's length and the cut's cadence, in microseconds.
    interval_micros: u64,
    /// One partition a model, each behind its own guard, by the model's
    /// name. The map is replaced whole at a model's first prefill decision.
    partitions: ArcSwap<FxHashMap<Box<str>, Partition>>,
    /// The partitions that list a worker, by its id: where a booking finds
    /// them. Replaced whole when a partition's listing changes.
    listed: ArcSwap<FxHashMap<WorkerId, Vec<Partition>>>,
    /// One writer of `listed` at a time. It is taken under a partition's
    /// guard and takes none itself.
    relisting: Mutex<()>,
    /// The workers the lifecycle pruned a moment ago, fenced against a
    /// listing from a view older than the pruning. One fence serves the
    /// stage: the feed names an id and not its pool, and a partition made
    /// after a leave must see it. It is taken under a partition's guard or
    /// under none, and no partition's guard is taken under it. A decision
    /// holds it for the one pass that lists its view's rows and for nothing
    /// after.
    fence: Mutex<RetiredWorkers>,
}

/// One model's partition behind its guard, shared with the workers' index.
type Partition = Arc<Mutex<Pool>>;

/// One model's partition and its window.
#[derive(Debug)]
struct Pool {
    /// The listed workers, in the order they were listed.
    members: Vec<Member>,
    /// The ranges in ascending order, contiguous from zero while any stand.
    ranges: Vec<Range>,
    /// The bookings of the window, oldest first.
    window: VecDeque<Booking>,
    /// The tokens the window holds.
    total: u64,
    /// A worker's share of the window at the last cut. Zero after the
    /// listing changed, so the next cut is made whatever the share.
    cut_for: u64,
    /// The instant of the next cut on the host's clock; zero before the
    /// model's first prefill decision.
    next_cut_micros: u64,
    /// Each listed worker's place in `members`, by its id: a row that does
    /// not stand at its member's place is found here, never searched for.
    by_id: FxHashMap<WorkerId, u32>,
    /// The partition left the table, with its model's last worker or as an
    /// entry a decision made and listed nobody in: the decision that still
    /// holds it takes the model's next one.
    left: bool,
    /// Whether each member has a surviving row in the decision in progress.
    /// Reused from decision to decision.
    seated: Vec<bool>,
    /// The rows of the view in the decision in progress that the partition
    /// does not list and the fence names, each by its place in the view and
    /// its worker's id, in the view's order. Written where the decision
    /// lists and reused from decision to decision.
    fenced: Vec<(u32, WorkerId)>,
    /// The window's sizes in ascending order, for the cut. Reused.
    sizes: Vec<u32>,
}

#[derive(Debug)]
struct Member {
    id: WorkerId,
    /// The tokens of the window booked on this worker.
    load: u64,
}

/// The sizes one member owns, both ends included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    member: u32,
    lower: u32,
    upper: u32,
}

#[derive(Clone, Copy, Debug)]
struct Booking {
    tokens: u32,
    /// The member booked, or [`NONE`] once its worker left the listing.
    member: u32,
    at_micros: u64,
}

/// Whose clock dates a booking.
#[derive(Clone, Copy, Debug)]
enum Clock {
    /// The decider's own, at a decision's issue. It can go back.
    Issue,
    /// A gateway's word of a dispatch, told after the fact. A date behind
    /// the window's is late, and no clock that went back.
    Dispatch,
}

/// The booked tokens of a decision's surviving members: the most, the least,
/// and the first member that holds the least, by its place in the listing.
#[derive(Clone, Copy, Debug)]
struct Loads {
    most: u64,
    least: u64,
    lightest: usize,
}

impl Bucket {
    pub fn new(params: &BucketParams) -> Self {
        Self {
            abs_threshold: u64::from(params.abs_threshold.unwrap_or(DEFAULT_ABS_THRESHOLD)),
            rel_threshold: params.rel_threshold.unwrap_or(DEFAULT_REL_THRESHOLD),
            interval_micros: u64::from(
                params
                    .adjust_interval_s
                    .unwrap_or(DEFAULT_ADJUST_INTERVAL_S),
            ) * 1_000_000,
            partitions: ArcSwap::from_pointee(FxHashMap::default()),
            listed: ArcSwap::from_pointee(FxHashMap::default()),
            relisting: Mutex::new(()),
            fence: Mutex::new(RetiredWorkers::default()),
        }
    }

    /// The partition of `model`, made empty at the model's first prefill
    /// decision. Where two deciders bring one model at once, both end on
    /// the entry that stood first.
    fn partition(&self, model: &str) -> Partition {
        loop {
            if let Some(partition) = self.partitions.load().get(model) {
                return Arc::clone(partition);
            }
            self.partitions.rcu(|partitions| {
                let mut next = FxHashMap::clone(partitions);
                next.entry(model.into())
                    .or_insert_with(|| Arc::new(Mutex::new(Pool::new())));
                next
            });
        }
    }

    /// Take `partition` out of the table where it still stands under
    /// `model`: an entry made since under the same name stays. The caller
    /// holds the partition's guard and has marked it as one that left.
    fn forget(&self, model: &str, partition: &Partition) {
        self.partitions.rcu(|partitions| {
            let mut next = FxHashMap::clone(partitions);
            let stands = next
                .get(model)
                .is_some_and(|stands| Arc::ptr_eq(stands, partition));
            if stands {
                next.remove(model);
            }
            next
        });
    }

    /// Write a partition's listing into the workers' index: the partition
    /// under each worker `members` names and under no other. The caller
    /// holds the partition's guard, so no booking falls between a listing
    /// and its entry here.
    fn relist(&self, partition: &Partition, members: &[Member]) {
        let _one = self.relisting.lock();
        let mut listed = FxHashMap::clone(&self.listed.load());
        listed.retain(|_, partitions| {
            partitions.retain(|other| !Arc::ptr_eq(other, partition));
            !partitions.is_empty()
        });
        for member in members {
            listed
                .entry(member.id)
                .or_default()
                .push(Arc::clone(partition));
        }
        self.listed.store(Arc::new(listed));
    }
}

impl Pool {
    fn new() -> Self {
        Self {
            members: Vec::new(),
            ranges: Vec::new(),
            window: VecDeque::new(),
            total: 0,
            cut_for: 0,
            next_cut_micros: 0,
            by_id: FxHashMap::default(),
            left: false,
            seated: Vec::new(),
            fenced: Vec::new(),
            sizes: Vec::new(),
        }
    }

    /// The member a worker is: the one at its row's position where the
    /// listing and the view stand in one order, else the one its id names.
    /// No member is searched for: a view in another order, or with two
    /// rows a worker, costs a lookup a row.
    fn member_of(&self, position: u32, id: WorkerId) -> Option<u32> {
        match self.members.get(position as usize) {
            Some(member) if member.id == id => Some(position),
            _ => self.by_id.get(&id).copied(),
        }
    }

    /// Write every member's place under its id again, after the places
    /// moved.
    fn place_members(&mut self) {
        self.by_id.clear();
        for (at, member) in self.members.iter().enumerate() {
            self.by_id.insert(member.id, at as u32);
        }
    }

    /// Seat the decision's survivors, each at its member, and read their
    /// loads in the kept order. A survivor that is not listed ends the pass
    /// with `None`. `LISTED` is the pass behind a listing ([`Self::list`]):
    /// a survivor the fence names is seated nowhere, and `None` then says
    /// that no survivor is seated. A row behind fenced rows of the view is
    /// looked for that many places nearer the listing's head, where it
    /// stands when the view is older than their leave.
    fn seat<const LISTED: bool>(
        &mut self,
        candidates: &[Candidate],
        kept: &[u32],
    ) -> Option<Loads> {
        self.seated.clear();
        self.seated.resize(self.members.len(), false);
        let mut loads: Option<Loads> = None;
        // How many of the view's fenced rows stand before the row at hand.
        let mut behind = 0;
        for &row in kept {
            let id = candidates[row as usize].id;
            if LISTED {
                while self.fenced.get(behind).is_some_and(|&(at, _)| at < row) {
                    behind += 1;
                }
            }
            let Some(member) = self.member_of(row.wrapping_sub(behind as u32), id) else {
                if LISTED && self.fenced.iter().any(|&(_, fenced)| fenced == id) {
                    continue;
                }
                return None;
            };
            let member = member as usize;
            if self.seated[member] {
                continue;
            }
            self.seated[member] = true;
            let load = self.members[member].load;
            loads = Some(match loads {
                None => Loads {
                    most: load,
                    least: load,
                    lightest: member,
                },
                Some(so_far) => Loads {
                    most: so_far.most.max(load),
                    least: so_far.least.min(load),
                    lightest: if load < so_far.least {
                        member
                    } else {
                        so_far.lightest
                    },
                },
            });
        }
        loads
    }

    /// List every row of the view the partition does not list, in the
    /// view's order, but for a worker `fenced` names: the lifecycle pruned
    /// it a moment ago, and a view that still shows it is older than its
    /// leave. A changed listing keeps the loads of the workers it held and
    /// splits the span evenly among all of it. The rows `fenced` names are
    /// kept for the seating, each with its place in the view: a row behind
    /// them is looked for that many places nearer the listing's head.
    fn list(&mut self, candidates: &[Candidate], fenced: impl Fn(WorkerId) -> bool) {
        let listed = self.members.len();
        self.fenced.clear();
        for candidate in candidates {
            let behind = self.fenced.len() as u32;
            let place = candidate.position.wrapping_sub(behind);
            if self.member_of(place, candidate.id).is_some() {
                continue;
            }
            if fenced(candidate.id) {
                self.fenced.push((candidate.position, candidate.id));
            } else {
                self.by_id.insert(candidate.id, self.members.len() as u32);
                self.members.push(Member {
                    id: candidate.id,
                    load: 0,
                });
            }
        }
        if self.members.len() != listed {
            self.split_evenly();
        }
    }

    /// [`SPAN`] in equal ranges, one a member in the listing's order, the
    /// last open-ended. The split is provisional: the next cut is made
    /// whatever the share, from the sizes the window holds by then.
    fn split_evenly(&mut self) {
        let workers = self.members.len() as u32;
        self.ranges.clear();
        if let Some(gap) = SPAN.checked_div(workers) {
            let gap = gap.max(1);
            for member in 0..workers {
                let upper = if member + 1 == workers {
                    u32::MAX
                } else {
                    (member + 1) * gap - 1
                };
                self.ranges.push(Range {
                    member,
                    lower: member * gap,
                    upper,
                });
            }
        }
        self.cut_for = 0;
    }

    /// Forget a worker: its place, its load and its range, which the range
    /// below takes over (the one above when it was the first). Its bookings
    /// stay in the window: they count toward the next cut and toward no
    /// worker's load. The next cut is made whatever the share.
    fn remove(&mut self, id: WorkerId) {
        let Some(at) = self.by_id.get(&id).map(|&at| at as usize) else {
            return;
        };
        self.members.remove(at);
        self.place_members();
        let gone = at as u32;
        for booking in &mut self.window {
            if booking.member == gone {
                booking.member = NONE;
            } else if booking.member != NONE && booking.member > gone {
                booking.member -= 1;
            }
        }
        if let Some(at) = self.ranges.iter().position(|range| range.member == gone) {
            let range = self.ranges.remove(at);
            if let Some(below) = at.checked_sub(1) {
                self.ranges[below].upper = range.upper;
            } else if let Some(above) = self.ranges.first_mut() {
                above.lower = range.lower;
            }
        }
        for range in &mut self.ranges {
            if range.member > gone {
                range.member -= 1;
            }
        }
        self.cut_for = 0;
    }

    /// The same worker under a new handle keeps its place, range and load.
    /// Where a view listed the new handle before this word came, the worker
    /// stands once, at the place that view gave it and with the load booked
    /// under both handles: the old handle's bookings move first, then its
    /// member leaves as a removed worker's does. Whether a member took the
    /// new handle.
    fn rename(&mut self, old: WorkerId, new: WorkerId) -> bool {
        if old != new && self.by_id.contains_key(&new) {
            self.rebook(old, new);
            self.remove(old);
            return false;
        }
        let Some(at) = self.by_id.remove(&old) else {
            return false;
        };
        self.members[at as usize].id = new;
        self.by_id.insert(new, at);
        true
    }

    /// Move the window's bookings of `from`'s member, and their tokens,
    /// onto `to`'s: one worker's load, under the handle that stays. Each
    /// booking then ages off that handle. `from`'s member keeps its count
    /// of the load: the caller removes the member next, and the count with
    /// it.
    fn rebook(&mut self, from: WorkerId, to: WorkerId) {
        let (Some(&from), Some(&to)) = (self.by_id.get(&from), self.by_id.get(&to)) else {
            return;
        };
        let load = self.members[from as usize].load;
        let stays = &mut self.members[to as usize];
        stays.load = stays.load.saturating_add(load);
        for booking in &mut self.window {
            if booking.member == from {
                booking.member = to;
            }
        }
    }

    /// The listed workers `stays`, a snapshot's names, leaves out.
    fn unnamed(&self, stays: &FxHashSet<WorkerId>) -> Vec<WorkerId> {
        self.members
            .iter()
            .map(|member| member.id)
            .filter(|id| !stays.contains(id))
            .collect()
    }

    /// The instant a dispatch's booking takes: `since_micros` after the
    /// decision of `issued_micros`, as its gateway dates it, and no later
    /// than the newest booking on file, or than that decision where the
    /// partition booked nothing since. The date is a gateway's word, and a
    /// booking dated past every other would age the whole window. With no
    /// booking on file it stands as said.
    ///
    /// `None` where that instant is more than `window_micros` before the
    /// booking at the window's front: the dispatch aged before it was
    /// told, and its booking would stand behind a window that passed it.
    /// `None` too where the window is full and that instant is as far
    /// before the booking behind the front: a full window loses its front
    /// to the cap, and that one stands at the front next.
    fn dispatched_at(
        &self,
        issued_micros: u64,
        since_micros: u32,
        window_micros: u64,
    ) -> Option<u64> {
        let said = issued_micros.saturating_add(u64::from(since_micros));
        let (Some(oldest), Some(newest)) = (self.window.front(), self.window.back()) else {
            return Some(said);
        };
        let at_micros = said.min(newest.at_micros.max(issued_micros));
        // The newest date a booking can bear and not be aged by this one.
        let reach = at_micros.saturating_add(window_micros);
        let behind = if self.window.len() < WINDOW_CAP {
            None
        } else {
            self.window.get(1)
        };
        (reach >= oldest.at_micros && behind.is_none_or(|next| reach >= next.at_micros))
            .then_some(at_micros)
    }

    /// Book `tokens` on a member at `at_micros`. The bookings older than the
    /// window leave first, each with its tokens off its worker's load. At
    /// an issue so does one dated more than the window after this one: a
    /// clock that went back that far ends what it passed. A dispatch's date
    /// is late, and no clock that went back: no booking leaves for being
    /// newer than it. A booking dated a little before the window's oldest
    /// arrived out of its clock's order, as the bookings of two deciders
    /// can, and nothing leaves on its account. Inlined: the issue's booking
    /// ran it in line while it was its one caller, and a call there costs
    /// every decision.
    #[inline]
    fn book(&mut self, member: u32, tokens: u32, at_micros: u64, window_micros: u64, clock: Clock) {
        if let Some(booked) = self.members.get_mut(member as usize) {
            booked.load = booked.load.saturating_add(u64::from(tokens));
        }
        while let Some(oldest) = self.window.front() {
            let passed = match clock {
                Clock::Issue => at_micros.abs_diff(oldest.at_micros),
                Clock::Dispatch => at_micros.saturating_sub(oldest.at_micros),
            };
            let aged = passed > window_micros;
            if !aged && self.window.len() < WINDOW_CAP {
                break;
            }
            let left = u64::from(oldest.tokens);
            // A booking whose worker left names no member: `NONE` is past
            // every index.
            if let Some(member) = self.members.get_mut(oldest.member as usize) {
                member.load = member.load.saturating_sub(left);
            }
            self.total = self.total.saturating_sub(left);
            self.window.pop_front();
        }
        self.window.push_back(Booking {
            tokens,
            member,
            at_micros,
        });
        self.total = self.total.saturating_add(u64::from(tokens));
    }

    /// Make the cut where the decision's clock has reached it. The model's
    /// first prefill decision starts the cadence one interval ahead; a
    /// decision several intervals behind makes one cut and keeps the
    /// cadence's phase.
    fn cut_if_due(&mut self, now_micros: u64, interval_micros: u64) {
        if self.next_cut_micros == 0 {
            self.next_cut_micros = now_micros.saturating_add(interval_micros);
            return;
        }
        if now_micros < self.next_cut_micros {
            return;
        }
        self.cut();
        let behind = (now_micros - self.next_cut_micros) / interval_micros.max(1);
        self.next_cut_micros = self
            .next_cut_micros
            .saturating_add(behind.saturating_add(1).saturating_mul(interval_micros));
    }

    /// Cut the ranges again from the window, once a worker's share of the
    /// booked tokens moved by more than a factor of two since the last cut
    /// or the listing changed. The walk is over the sizes in ascending
    /// order: each worker but the last takes whole size classes until its
    /// share is reached, its range ends at the last class it took, and the
    /// last worker takes the open end. A class is never split between two
    /// workers. A worker stops short of its share when the classes left are
    /// no more than the workers after it, so each of them owns a seen size
    /// while classes remain; with fewer classes than workers a worker owns
    /// no range until the next cut, and the balance rule evens the load
    /// meanwhile.
    fn cut(&mut self) {
        let workers = self.members.len();
        if self.window.is_empty() || workers == 0 {
            return;
        }
        let share = self.total / workers as u64;
        let before = self.cut_for;
        if before != 0 && share <= before.saturating_mul(2) && before <= share.saturating_mul(2) {
            return;
        }
        self.cut_for = share;
        self.sizes.clear();
        self.sizes
            .extend(self.window.iter().map(|booking| booking.tokens));
        self.sizes.sort_unstable();
        // A class is the bookings of one size, with the tokens they add up to.
        let mut classes_left = self.sizes.chunk_by(|a, b| a == b).count();
        let mut classes = self.sizes.chunk_by(|a, b| a == b).filter_map(|class| {
            Some((
                *class.first()?,
                class.iter().map(|&size| u64::from(size)).sum::<u64>(),
            ))
        });
        self.ranges.clear();
        let mut lower = 0u32;
        for member in 0..workers {
            if member + 1 == workers {
                self.ranges.push(Range {
                    member: member as u32,
                    lower,
                    upper: u32::MAX,
                });
                break;
            }
            let after = workers - member - 1;
            let mut taken = 0u64;
            let mut upper = None;
            for (size, tokens) in classes.by_ref() {
                taken = taken.saturating_add(tokens);
                upper = Some(size);
                classes_left -= 1;
                if taken >= share || classes_left <= after {
                    break;
                }
            }
            if let Some(size) = upper {
                self.ranges.push(Range {
                    member: member as u32,
                    lower,
                    upper: size,
                });
                lower = size.saturating_add(1);
            }
        }
    }

    /// The member a size goes to among the seated: the owner of the size's
    /// range, else the nearest seated owner below it in the partition, else
    /// the nearest above; by its place in the listing.
    fn owner(&self, tokens: u32) -> Option<usize> {
        let at = self
            .ranges
            .binary_search_by(|range| {
                if tokens < range.lower {
                    Ordering::Greater
                } else if tokens > range.upper {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .ok()?;
        (0..=at)
            .rev()
            .chain(at + 1..self.ranges.len())
            .map(|at| self.ranges[at].member as usize)
            .find(|&member| self.seated[member])
    }
}

impl Lifecycle for Bucket {
    /// Books the lease's tokens on the worker the lease names, in every
    /// partition that lists it, at the lease's instant: a decode-side lease
    /// names its pair's decode, any other the target itself.
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        let Some(target) = decision
            .pick()
            .and_then(|pick| pick.targets.as_slice().get(usize::from(lease.target_index)))
        else {
            return;
        };
        let (position, id) = match (lease.side, target.pair) {
            (Side::Decode, Some(pair)) => (pair.decode_position, pair.decode_id),
            _ => (target.position, target.id),
        };
        let listed = self.listed.load();
        for partition in listed.get(&id).into_iter().flatten() {
            let mut pool = partition.lock();
            if let Some(member) = pool.member_of(position, id) {
                pool.book(
                    member,
                    sized_by(lease.tokens, lease.chars),
                    lease.issued_micros,
                    self.interval_micros,
                    Clock::Issue,
                );
            }
        }
    }

    /// A stage that books at the issue books the worker that serves. A
    /// lease is issued on its pick's first target, and the lease a host
    /// tells at its first dispatch names the target that dispatch took and
    /// the worker its issue booked ([`Lease::booked`]). A pair's decode
    /// that no call issued is issued at its dispatch, on the target that
    /// dispatch took, and its lease names that worker as booked. Where the
    /// dispatch took the first target, or another target of the worker the
    /// issue booked (another rank of it, or the one prefill of a pair
    /// pick), the booking of the issue stands and the call returns at
    /// once. Under a host that names no such worker, the first target
    /// alone is read as the issue's. A dispatch on another worker books
    /// the lease's tokens there too, at the dispatch's instant, unless it
    /// aged before it was told. The booking of the issue ages off its
    /// worker with its interval: nothing is un-booked. A host tells a
    /// lease's first dispatch alone.
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        _side: Side,
        since_decision_micros: u32,
    ) {
        if lease.target_index == 0 {
            return;
        }
        // The worker and its partitions, as the issue reads them.
        let (position, id) = match (lease.side, target.pair) {
            (Side::Decode, Some(pair)) => (pair.decode_position, pair.decode_id),
            _ => (target.position, target.id),
        };
        // Another target of the worker the issue booked.
        if id == lease.booked {
            return;
        }
        let listed = self.listed.load();
        for partition in listed.get(&id).into_iter().flatten() {
            let mut pool = partition.lock();
            let Some(member) = pool.member_of(position, id) else {
                continue;
            };
            let dated = pool.dispatched_at(
                lease.issued_micros,
                since_decision_micros,
                self.interval_micros,
            );
            if let Some(at_micros) = dated {
                pool.book(
                    member,
                    sized_by(lease.tokens, lease.chars),
                    at_micros,
                    self.interval_micros,
                    Clock::Dispatch,
                );
            }
        }
    }

    fn on_membership(&self, change: &Membership<'_>) {
        // The fence takes the word first. A leaver is fenced before any
        // partition forgets it, so no decision finds it unlisted and
        // unfenced, and a worker the feed names is admitted at once.
        let now = Instant::now();
        // A snapshot's names as a set, built once: its word's pass and each
        // partition's ask it of every worker they list.
        let stays: FxHashSet<WorkerId> = match *change {
            Membership::Snapshot(ids) => ids.iter().copied().collect(),
            _ => FxHashSet::default(),
        };
        match *change {
            Membership::Added(id) => self.fence.lock().admit(id),
            Membership::Removed(id) => self.fence.lock().retire(id, now),
            Membership::Replaced { old, new } => {
                let mut fence = self.fence.lock();
                fence.retire(old, now);
                fence.admit(new);
            }
            Membership::Snapshot(ids) => {
                // A snapshot names who stays. The workers the index lists
                // and it leaves out are its leavers, fenced here, before
                // the walk reads the table: a partition a decision makes
                // behind that read lists none of them, and neither does
                // one the walk has passed. The leavers are read before the
                // fence is taken: every decision that lists waits on it.
                let leavers: Vec<WorkerId> = self
                    .listed
                    .load()
                    .keys()
                    .copied()
                    .filter(|id| !stays.contains(id))
                    .collect();
                let mut fence = self.fence.lock();
                if !fence.is_empty() {
                    for &id in ids {
                        fence.admit(id);
                    }
                }
                for id in leavers {
                    fence.retire(id, now);
                }
            }
            // A relabel and a resize prune nobody. Nor does a worker that
            // left a model: it serves on, and the fence names no model, so
            // a view older than the word may list it under that model again.
            Membership::Relabelled(_) | Membership::Resized(_) | Membership::Left { .. } => {}
        }
        for (model, partition) in self.partitions.load().iter() {
            let mut pool = partition.lock();
            let listed = pool.members.len();
            let renamed = match *change {
                Membership::Removed(id) => {
                    pool.remove(id);
                    false
                }
                Membership::Replaced { old, new } => pool.rename(old, new),
                Membership::Snapshot(_) => {
                    // The workers the snapshot does not name leave. The
                    // fence holds those the index listed at the word; one
                    // a decision was listing then is fenced here, before
                    // its partition forgets it.
                    let gone = pool.unnamed(&stays);
                    if !gone.is_empty() {
                        let mut fence = self.fence.lock();
                        for &id in &gone {
                            fence.retire(id, now);
                        }
                    }
                    for id in gone {
                        pool.remove(id);
                    }
                    false
                }
                // A worker a registration took off this partition's model
                // leaves its listing. The partitions of the models the word
                // does not name keep its place, its range and its load.
                Membership::Left { id, models } => {
                    if models.iter().any(|left| **left == **model) {
                        pool.remove(id);
                    }
                    false
                }
                // A worker joins from a view of its pool: the feed names an
                // id and not the pool it serves.
                Membership::Added(_) | Membership::Relabelled(_) | Membership::Resized(_) => false,
            };
            // The index follows a listing that changed, under its guard.
            if renamed || pool.members.len() != listed {
                self.relist(partition, &pool.members);
            }
            // A model whose last worker left takes its partition with it.
            if listed != 0 && pool.members.is_empty() {
                pool.left = true;
                self.forget(model, partition);
            }
        }
    }

    /// The bookings and the listing are the stage's whole state, and these
    /// calls alone keep them: under a host that makes none, every size is
    /// placed by an even split, nothing spills and a worker that left stays
    /// listed.
    fn needs_lifecycle(&self) -> bool {
        true
    }

    /// The issue books and the dispatch books again where the request went
    /// elsewhere. A booking ages off: neither end is taken.
    fn hears(&self) -> EventSet {
        EventSet::NONE
            .with(EventKind::Issued)
            .with(EventKind::Dispatched)
    }
}

impl Scorer for Bucket {
    fn name(&self) -> &'static str {
        "bucket"
    }

    // Places by its book of issues and dispatches: a late one moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    // A prefill is placed by the prompt's size, so a host that can send a
    // request on unread reads it under this stage: a size of zero would
    // send every request to the owner of the smallest sizes.
    fn needs(&self) -> Needs {
        Needs {
            prompt_size: true,
            ..Needs::default()
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
        if facts.phase != Phase::Prefill || kept.is_empty() {
            return TieBand::NORMALIZED;
        }
        loop {
            let partition = self.partition(facts.model);
            let mut pool = partition.lock();
            // A partition that left the table between the two reads takes no
            // decision: the model's next one does.
            if pool.left {
                continue;
            }
            let loads = match pool.seat::<false>(candidates, kept) {
                Some(loads) => loads,
                None => {
                    // A survivor the partition does not list: the view's
                    // rows join, and the listing is split again before the
                    // cut. A worker the lifecycle pruned a moment ago joins
                    // nowhere: the view is older than its leave, so it is
                    // seated nowhere and scores zero, and the others are
                    // placed as if the view did not show it.
                    // The fence is held for the listing's one pass, which
                    // asks it of each row the partition does not list, and
                    // let go before the seating; an empty fence is let go
                    // before the pass. The partition's guard, which the
                    // feed's walk needs, is what keeps a leave behind this
                    // decision.
                    let listed = pool.members.len();
                    {
                        let mut fence = self.fence.lock();
                        fence.expire(Instant::now());
                        if fence.is_empty() {
                            drop(fence);
                            pool.list(candidates, |_| false);
                        } else {
                            pool.list(candidates, |id| fence.holds(id));
                        }
                    }
                    if pool.members.len() != listed {
                        self.relist(&partition, &pool.members);
                    }
                    match pool.seat::<true>(candidates, kept) {
                        Some(loads) => loads,
                        None => {
                            // A view of fenced rows alone lists nobody. The
                            // partition this decision made for it leaves
                            // the table again: the feed's walk takes only
                            // an entry whose last worker left.
                            if pool.members.is_empty() {
                                pool.left = true;
                                self.forget(facts.model, &partition);
                            }
                            return TieBand::NORMALIZED;
                        }
                    }
                }
            };
            pool.cut_if_due(facts.now_micros, self.interval_micros);
            let spills = loads.most - loads.least > self.abs_threshold
                && loads.most as f64 > f64::from(self.rel_threshold) * loads.least as f64;
            let leads = if spills {
                Some(loads.lightest)
            } else {
                pool.owner(sized_by(facts.prompt_tokens, facts.prompt_chars))
            };
            if let Some(id) =
                leads.and_then(|member| pool.members.get(member).map(|member| member.id))
            {
                for (&row, score) in kept.iter().zip(out) {
                    if candidates[row as usize].id == id {
                        *score = 1.0;
                    }
                }
            }
            return TieBand::NORMALIZED;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/bucket/tests.rs"]
mod tests;
