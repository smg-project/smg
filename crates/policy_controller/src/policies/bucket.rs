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
mod tests {
    use super::*;
    use crate::{
        document::{PolicySpec, SpecError},
        engine::{
            decision::{Pair, Pick, Target, Targets},
            view::{self, Health},
            Constraints, PoolFacts, Scratch,
        },
        stages::{retired_workers::RETIRED_WORKER_TTL, StageRegistry},
    };

    const SECOND: u64 = 1_000_000;
    /// An absolute threshold no booking of a case reaches: the balance rule
    /// never spills under it.
    const NEVER: u32 = u32::MAX;

    fn scorer(abs_threshold: u32, rel_threshold: f32, interval_s: u32) -> Bucket {
        Bucket::new(&BucketParams {
            abs_threshold: Some(abs_threshold),
            rel_threshold: Some(rel_threshold),
            adjust_interval_s: Some(interval_s),
            ..BucketParams::default()
        })
    }

    fn pool(workers: u32) -> Vec<Candidate> {
        (0..workers)
            .map(|row| Candidate {
                position: row,
                id: WorkerId(100 + u64::from(row)),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                ..Candidate::default()
            })
            .collect()
    }

    fn all(pool: &[Candidate]) -> Vec<u32> {
        (0..pool.len() as u32).collect()
    }

    fn prefill(model: &str, tokens: u32, now_micros: u64) -> RequestFacts<'_> {
        RequestFacts {
            model,
            phase: Phase::Prefill,
            prompt_tokens: tokens,
            now_micros,
            ..RequestFacts::default()
        }
    }

    /// The first row of the worker the scorer leads with among `kept`; `None`
    /// when every survivor scores equal. Every row of one worker at most
    /// leads, at the top of the scale, and every other stands at zero.
    fn led_by(
        scorer: &Bucket,
        facts: &RequestFacts<'_>,
        pool: &[Candidate],
        kept: &[u32],
    ) -> Option<u32> {
        let mut out = vec![0.5; kept.len()];
        assert_eq!(
            scorer.score(facts, pool, kept, &mut out),
            TieBand::NORMALIZED
        );
        let leads: Vec<u32> = kept
            .iter()
            .zip(&out)
            .filter(|&(_, &score)| score > 0.5)
            .map(|(&row, _)| row)
            .collect();
        if let Some(&first) = leads.first() {
            assert!(
                leads
                    .iter()
                    .all(|&row| pool[row as usize].id == pool[first as usize].id),
                "{out:?}"
            );
        }
        assert!(
            out.iter().all(
                |&score| score.to_bits() == 0f32.to_bits() || score.to_bits() == 1f32.to_bits()
            ),
            "{out:?}"
        );
        leads.first().copied()
    }

    /// The row a prefill request of `tokens` leads with over the whole pool.
    fn led(scorer: &Bucket, pool: &[Candidate], tokens: u32, now_micros: u64) -> Option<u32> {
        led_by(scorer, &prefill("m", tokens, now_micros), pool, &all(pool))
    }

    /// Book `tokens` on the worker of `row` at `at_micros`, as an issued
    /// decision that names it first.
    fn book(scorer: &Bucket, pool: &[Candidate], row: u32, tokens: u32, at_micros: u64) {
        let lease = Lease {
            tokens,
            issued_micros: at_micros,
            ..Lease::default()
        };
        issue(scorer, pool, row, &lease);
    }

    /// Tell the scorer the issue of `lease` on the worker of `row`, as a
    /// decision that names it first.
    fn issue(scorer: &Bucket, pool: &[Candidate], row: u32, lease: &Lease) {
        let mut targets = Targets::new();
        targets.push(Target {
            position: row,
            id: pool[row as usize].id,
            dp_rank: 0,
            pair: None,
        });
        let decision = Decision::Pick(Pick {
            targets,
            ..Pick::default()
        });
        scorer.on_decision_issued(lease, &decision);
    }

    /// The partition of `model`, which a decision listed.
    fn of(scorer: &Bucket, model: &str) -> Partition {
        Arc::clone(
            scorer
                .partitions
                .load()
                .get(model)
                .expect("the model's partition"),
        )
    }

    /// The ranges of model `m`: each as its member, its lower and its upper
    /// end.
    fn ranges(scorer: &Bucket) -> Vec<(u32, u32, u32)> {
        of(scorer, "m")
            .lock()
            .ranges
            .iter()
            .map(|range| (range.member, range.lower, range.upper))
            .collect()
    }

    /// Each member with its load, each range, and the window's total.
    type Listing = (Vec<(WorkerId, u64)>, Vec<(u32, u32, u32)>, u64);

    /// The listing of `model`'s partition.
    fn listing_of(scorer: &Bucket, model: &str) -> Listing {
        let partition = of(scorer, model);
        let held = partition.lock();
        let members = held
            .members
            .iter()
            .map(|member| (member.id, member.load))
            .collect();
        let ranges = held
            .ranges
            .iter()
            .map(|range| (range.member, range.lower, range.upper))
            .collect();
        (members, ranges, held.total)
    }

    /// The booked tokens of model `m`'s members, in the listing's order, and
    /// the window's total.
    fn loads(scorer: &Bucket) -> (Vec<u64>, u64) {
        let partition = of(scorer, "m");
        let pool = partition.lock();
        (
            pool.members.iter().map(|member| member.load).collect(),
            pool.total,
        )
    }

    /// A scorer over a listed pool with `tokens[row]` booked on each row at
    /// the listing's instant, one second on the clock.
    fn booked(abs_threshold: u32, rel_threshold: f32, tokens: &[u32]) -> (Bucket, Vec<Candidate>) {
        let scorer = scorer(abs_threshold, rel_threshold, 5);
        let pool = pool(tokens.len() as u32);
        led(&scorer, &pool, 0, SECOND);
        for (row, &tokens) in tokens.iter().enumerate() {
            if tokens > 0 {
                book(&scorer, &pool, row as u32, tokens, SECOND);
            }
        }
        (scorer, pool)
    }

    #[test]
    fn a_fresh_listing_splits_the_span_evenly_and_a_size_goes_to_its_range_s_owner() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(4);
        let placed: Vec<Option<u32>> =
            [0, 1_023, 1_024, 2_047, 2_048, 3_071, 3_072, 4_096, u32::MAX]
                .iter()
                .map(|&tokens| led(&scorer, &pool, tokens, SECOND))
                .collect();
        assert_eq!(
            placed,
            [0, 0, 1, 1, 2, 2, 3, 3, 3].map(Some),
            "a quarter of the span each, the last range open-ended"
        );
        assert_eq!(
            ranges(&scorer),
            [
                (0, 0, 1_023),
                (1, 1_024, 2_047),
                (2, 2_048, 3_071),
                (3, 3_072, u32::MAX)
            ]
        );
        // A span that does not divide leaves the rest to the open end.
        let thirds = self::scorer(NEVER, 1.5, 5);
        let three = self::pool(3);
        led(&thirds, &three, 0, SECOND);
        assert_eq!(
            ranges(&thirds),
            [(0, 0, 1_364), (1, 1_365, 2_729), (2, 2_730, u32::MAX)]
        );
    }

    /// A request of text alone is placed and booked by its text's length in
    /// characters, as the classic `bucket` sizes one
    /// (`model_gateway/src/policies/bucket.rs:104-110`) and books it
    /// (`:178-181`). A request that states token ids is placed and booked
    /// by their count, whatever length stands beside it. A dispatch on
    /// another worker books the same size there.
    #[test]
    fn a_request_of_text_alone_is_placed_and_booked_by_its_characters() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(4);
        let led = |tokens, chars| {
            let facts = RequestFacts {
                prompt_chars: chars,
                ..prefill("m", tokens, SECOND)
            };
            led_by(&scorer, &facts, &pool, &all(&pool))
        };
        // 2,500 characters and no token id: the third quarter's owner.
        assert_eq!(led(0, 2_500), Some(2));
        // Token ids win: 100 of them go to the first quarter's owner.
        assert_eq!(led(100, 2_500), Some(0));
        let sized = |tokens, chars, index| Lease {
            tokens,
            chars,
            target_index: index,
            issued_micros: SECOND,
            ..Lease::default()
        };
        issue(&scorer, &pool, 2, &sized(0, 2_500, 0));
        issue(&scorer, &pool, 0, &sized(100, 2_500, 0));
        assert_eq!(loads(&scorer), (vec![100, 0, 2_500, 0], 2_600));
        // The text's request is sent to its pick's second target, the last
        // row: that worker is booked its characters too.
        let target = Target {
            id: pool[3].id,
            ..Target::default()
        };
        scorer.on_dispatched(&sized(0, 2_500, 1), &target, Side::Single, 250);
        assert_eq!(loads(&scorer), (vec![100, 0, 2_500, 2_500], 5_100));
    }

    #[test]
    fn an_owner_that_is_no_survivor_hands_its_size_down_then_up() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(4);
        let over =
            |tokens, kept: &[u32]| led_by(&scorer, &prefill("m", tokens, SECOND), &pool, kept);
        // The first decision lists every row of its view, survivor or not:
        // row 1 owns its range from the start.
        assert_eq!(over(1_500, &[0, 2, 3]), Some(0), "the nearest below");
        assert_eq!(ranges(&scorer).len(), 4);
        assert_eq!(
            over(1_500, &[2, 3]),
            Some(2),
            "none below: the nearest above"
        );
        assert_eq!(
            over(3_500, &[0, 1]),
            Some(1),
            "the nearest below the last range"
        );
        assert_eq!(
            over(100, &[2, 3]),
            Some(2),
            "the nearest above the first range"
        );
        assert_eq!(over(100, &[3]), Some(3));
        assert_eq!(
            over(1_500, &[0, 1, 2, 3]),
            Some(1),
            "the owner when it survives"
        );
    }

    #[test]
    fn only_a_prefill_decision_is_placed() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(4);
        for phase in [Phase::Single, Phase::Decode, Phase::Encode] {
            let facts = RequestFacts {
                phase,
                ..prefill("m", 1_500, SECOND)
            };
            assert_eq!(
                led_by(&scorer, &facts, &pool, &all(&pool)),
                None,
                "{phase:?}"
            );
        }
        assert!(
            scorer.partitions.load().is_empty(),
            "another phase lists nothing and starts no cadence"
        );
        assert_eq!(
            led_by(&scorer, &prefill("m", 1_500, SECOND), &pool, &[]),
            None
        );
        assert!(
            scorer.partitions.load().is_empty(),
            "a decision with no survivor"
        );
    }

    #[test]
    fn a_booking_counts_for_its_worker_until_it_is_older_than_the_interval() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        // A booking before any listing names no member of any partition.
        book(&scorer, &pool, 0, 9, SECOND);
        led(&scorer, &pool, 0, SECOND);
        assert_eq!(loads(&scorer), (vec![0, 0], 0));
        book(&scorer, &pool, 0, 100, SECOND);
        assert_eq!(loads(&scorer), (vec![100, 0], 100));
        // Exactly one interval old is not older than the interval.
        book(&scorer, &pool, 1, 40, 6 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 40], 140));
        book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
        assert_eq!(loads(&scorer), (vec![0, 50], 50), "the first booking left");
        // A booking dated a little before the window's oldest arrived out
        // of its clock's order: both stand.
        book(&scorer, &pool, 0, 7, 6 * SECOND - 5);
        assert_eq!(loads(&scorer), (vec![7, 50], 57));
        // One dated more than an interval before it is a clock that went
        // back: it ends what it passed.
        book(&scorer, &pool, 0, 3, SECOND / 2);
        assert_eq!(loads(&scorer), (vec![3, 0], 3));
        // A worker no partition lists takes no booking.
        let stranger = self::pool(3);
        book(&scorer, &stranger, 2, 1_000, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![3, 0], 3));
    }

    #[test]
    fn two_bookings_that_arrive_out_of_their_clocks_order_both_count() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        // Two deciders read the clock five microseconds apart and book in
        // the reverse order, into a window that holds nothing older.
        book(&scorer, &pool, 0, 100, SECOND + 5);
        book(&scorer, &pool, 1, 40, SECOND);
        assert_eq!(loads(&scorer), (vec![100, 40], 140));
        // The older one waits behind the newer and leaves with it.
        book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
        assert_eq!(loads(&scorer), (vec![100, 50], 150));
        book(&scorer, &pool, 1, 10, 6 * SECOND + 6);
        assert_eq!(loads(&scorer), (vec![0, 20], 20), "both left");
    }

    #[test]
    fn a_stage_that_names_no_interval_keeps_a_booking_five_seconds() {
        let scorer = Bucket::new(&BucketParams::default());
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        book(&scorer, &pool, 0, 100, SECOND);
        // Exactly five seconds old is not older than the default interval.
        book(&scorer, &pool, 1, 40, 6 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 40], 140));
        book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
        assert_eq!(loads(&scorer), (vec![0, 50], 50), "the first booking left");
    }

    #[test]
    fn the_balance_rule_spills_to_the_least_booked_survivor_when_both_its_parts_hold() {
        // Size 0 is the first row's in an even split of three.
        let spilled = |abs, rel, tokens: &[u32]| {
            let (scorer, pool) = booked(abs, rel, tokens);
            led(&scorer, &pool, 0, SECOND)
        };
        assert_eq!(
            spilled(32, 1.5, &[100, 60, 80]),
            Some(1),
            "40 apart, 100 over 90"
        );
        assert_eq!(
            spilled(32, 1.5, &[100, 70, 80]),
            Some(0),
            "30 apart: the owner"
        );
        assert_eq!(
            spilled(32, 1.5, &[92, 60, 80]),
            Some(0),
            "32 apart is not more than 32"
        );
        assert_eq!(
            spilled(32, 1.5, &[100, 67, 80]),
            Some(0),
            "100 is not over 100.5"
        );
        assert_eq!(spilled(32, 1.5, &[100, 66, 80]), Some(1), "100 over 99");
        assert_eq!(spilled(0, 1.5, &[90, 60, 80]), Some(0), "90 is not over 90");
        assert_eq!(spilled(32, 1.5, &[50, 0, 10]), Some(1), "an idle worker");
        assert_eq!(
            spilled(32, 1.5, &[100, 60, 60]),
            Some(1),
            "the first of two least"
        );
        assert_eq!(spilled(32, 1.5, &[0, 0, 0]), Some(0), "nothing booked");
        // The rule weighs the survivors alone: the least booked worker of the
        // listing draws no spill while it is no survivor.
        let (scorer, pool) = booked(32, 1.5, &[100, 10, 80]);
        let facts = prefill("m", 0, SECOND);
        assert_eq!(led_by(&scorer, &facts, &pool, &[0, 2]), Some(0), "20 apart");
        let (scorer, pool) = booked(32, 1.5, &[100, 10, 60]);
        assert_eq!(led_by(&scorer, &facts, &pool, &[0, 2]), Some(2));
    }

    #[test]
    fn the_balance_rule_uses_the_stored_relative_threshold_at_an_integer_boundary() {
        let (scorer, pool) = booked(64, 1.3, &[286, 220]);
        assert_eq!(led(&scorer, &pool, 0, SECOND), Some(1));
    }

    #[test]
    fn large_exact_loads_still_cross_the_balance_gate() {
        let scorer = scorer(64, 1.0, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        for row in 0..2 {
            for _ in 0..256 {
                book(&scorer, &pool, row, u32::MAX, SECOND);
            }
        }
        book(&scorer, &pool, 0, 321, SECOND);
        book(&scorer, &pool, 1, 256, SECOND);

        assert_eq!(loads(&scorer).0, vec![(1 << 40) + 65, 1 << 40]);
        assert_eq!(led(&scorer, &pool, 0, SECOND), Some(1));
    }

    /// Three workers listed at ten seconds, then ten bookings of 100 tokens,
    /// five of 400 and two of 1,500 at eleven: 6,000 tokens, 2,000 a worker.
    fn three_classes() -> (Bucket, Vec<Candidate>) {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, 10 * SECOND);
        for (row, tokens, count) in [(0, 100, 10), (1, 400, 5), (2, 1_500, 2)] {
            for _ in 0..count {
                book(&scorer, &pool, row, tokens, 11 * SECOND);
            }
        }
        (scorer, pool)
    }

    #[test]
    fn the_first_decision_at_an_interval_s_end_cuts_the_ranges_before_it_is_placed() {
        let (scorer, pool) = three_classes();
        assert_eq!(
            led(&scorer, &pool, 400, 15 * SECOND - 1),
            Some(0),
            "the even split"
        );
        assert_eq!(ranges(&scorer)[0], (0, 0, 1_364));
        // The model's first decision came at ten seconds: the cut is due at
        // fifteen, and the decision that reaches it is placed by the new
        // ranges. Each worker but the last takes whole classes: the first
        // stops at one class, since the two left are owed to the two workers
        // after it, and the second reaches its share with the next.
        assert_eq!(led(&scorer, &pool, 400, 15 * SECOND), Some(1));
        assert_eq!(
            ranges(&scorer),
            [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)]
        );
        assert_eq!(led(&scorer, &pool, 100, 15 * SECOND), Some(0));
        assert_eq!(led(&scorer, &pool, 101, 15 * SECOND), Some(1));
        assert_eq!(led(&scorer, &pool, 401, 15 * SECOND), Some(2));
        let partition = of(&scorer, "m");
        let held = partition.lock();
        assert_eq!((held.cut_for, held.next_cut_micros), (2_000, 20 * SECOND));
    }

    #[test]
    fn a_cut_waits_until_a_worker_s_share_moved_by_more_than_a_factor_of_two() {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        let cut = [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
        assert_eq!(ranges(&scorer), cut);
        // A new class and 1,000 tokens more: 7,000 in the window, 2,333 a
        // worker against 2,000 at the last cut. The ranges stand.
        for _ in 0..20 {
            book(&scorer, &pool, 0, 50, 16 * SECOND);
        }
        assert_eq!(loads(&scorer).1, 7_000);
        led(&scorer, &pool, 0, 20 * SECOND);
        assert_eq!(ranges(&scorer), cut);
        // The bookings of eleven seconds leave and 18,000 tokens join: 6,333
        // a worker is more than twice 2,000, and the next cut is made.
        book(&scorer, &pool, 2, 9_000, 20 * SECOND + SECOND / 2);
        book(&scorer, &pool, 2, 9_000, 20 * SECOND + SECOND / 2);
        assert_eq!(loads(&scorer), (vec![1_000, 0, 18_000], 19_000));
        led(&scorer, &pool, 0, 25 * SECOND - 1);
        assert_eq!(ranges(&scorer), cut, "not before the interval's end");
        led(&scorer, &pool, 0, 25 * SECOND);
        assert_eq!(
            ranges(&scorer),
            [(0, 0, 50), (1, 51, 9_000), (2, 9_001, u32::MAX)]
        );
        // A share that fell by more than half is cut for too.
        book(&scorer, &pool, 0, 30, 31 * SECOND);
        book(&scorer, &pool, 1, 60, 31 * SECOND);
        book(&scorer, &pool, 2, 90, 31 * SECOND);
        assert_eq!(loads(&scorer), (vec![30, 60, 90], 180));
        led(&scorer, &pool, 0, 35 * SECOND);
        assert_eq!(
            ranges(&scorer),
            [(0, 0, 30), (1, 31, 60), (2, 61, u32::MAX)]
        );
    }

    #[test]
    fn fewer_size_classes_than_workers_leave_a_worker_without_a_range() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        for _ in 0..3 {
            book(&scorer, &pool, 0, 200, SECOND);
        }
        led(&scorer, &pool, 0, 6 * SECOND);
        assert_eq!(ranges(&scorer), [(0, 0, 200), (2, 201, u32::MAX)]);
        let over =
            |tokens, kept: &[u32]| led_by(&scorer, &prefill("m", tokens, 6 * SECOND), &pool, kept);
        assert_eq!(over(150, &[0, 1, 2]), Some(0));
        assert_eq!(
            over(150, &[1, 2]),
            Some(2),
            "the middle worker owns no range"
        );
        assert_eq!(over(5_000, &[0, 1]), Some(0), "the nearest owner below");
        assert_eq!(
            over(5_000, &[1]),
            None,
            "no owner survives: the picker's draw"
        );
    }

    #[test]
    fn a_decision_several_intervals_late_makes_one_cut_and_keeps_the_cadence() {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 32 * SECOND + 7);
        assert_eq!(ranges(&scorer)[0], (0, 0, 100));
        assert_eq!(
            of(&scorer, "m").lock().next_cut_micros,
            35 * SECOND,
            "the cadence still counts from the model's first decision"
        );
    }

    #[test]
    fn a_row_the_partition_never_listed_splits_the_span_again_and_keeps_the_loads() {
        let (scorer, two) = booked(NEVER, 1.5, &[100, 40]);
        led(&scorer, &two, 0, 6 * SECOND);
        assert_eq!(ranges(&scorer), [(0, 0, 40), (1, 41, u32::MAX)], "learned");
        // A third worker in the view: the span in thirds, the loads kept,
        // and the next cut made whatever the share.
        let three = pool(3);
        assert_eq!(led(&scorer, &three, 3_000, 7 * SECOND), Some(2));
        assert_eq!(
            ranges(&scorer),
            [(0, 0, 1_364), (1, 1_365, 2_729), (2, 2_730, u32::MAX)]
        );
        assert_eq!(loads(&scorer), (vec![100, 40, 0], 140));
        assert_eq!(of(&scorer, "m").lock().cut_for, 0);
        // A view that shows fewer rows forgets nobody: the lifecycle prunes.
        assert_eq!(
            led(&scorer, &two, 3_000, 8 * SECOND),
            Some(1),
            "the nearest below"
        );
        assert_eq!(ranges(&scorer).len(), 3);
        // A new row and a cut due at one decision: the listing first, so the
        // cut is over all three.
        let (scorer, _) = booked(NEVER, 1.5, &[100, 40]);
        assert_eq!(led(&scorer, &three, 3_000, 6 * SECOND), Some(2));
        assert_eq!(
            ranges(&scorer),
            [(0, 0, 40), (1, 41, 100), (2, 101, u32::MAX)]
        );
    }

    #[test]
    fn a_removed_worker_s_range_goes_to_the_range_below_and_the_first_s_to_the_one_above() {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        scorer.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
        // Its bookings stay in the window and count toward nobody's load.
        assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_000));
        assert_eq!(of(&scorer, "m").lock().cut_for, 0, "the next cut is made");
        // A view of the two that stay is placed by the merged ranges.
        let left = [
            pool[0],
            Candidate {
                position: 1,
                ..pool[2]
            },
        ];
        assert_eq!(
            led_by(&scorer, &prefill("m", 300, 16 * SECOND), &left, &[0, 1]),
            Some(0)
        );
        // The bookings of the removed worker leave with their interval.
        book(&scorer, &left, 1, 5, 17 * SECOND);
        assert_eq!(loads(&scorer), (vec![0, 5], 5));

        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        scorer.on_membership(&Membership::Removed(pool[0].id));
        assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
        assert_eq!(loads(&scorer), (vec![2_000, 3_000], 6_000));
        scorer.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(
            ranges(&scorer),
            [(0, 0, u32::MAX)],
            "the last owns every size"
        );
        // An id nobody lists changes nothing.
        scorer.on_membership(&Membership::Removed(WorkerId(7)));
        assert_eq!(loads(&scorer), (vec![3_000], 6_000));
    }

    #[test]
    fn a_model_whose_last_worker_left_takes_its_partition_with_it() {
        let (scorer, pool) = three_classes();
        let held = of(&scorer, "m");
        // A snapshot that names one of the three keeps the partition.
        scorer.on_membership(&Membership::Snapshot(&[pool[2].id]));
        assert_eq!(loads(&scorer), (vec![3_000], 6_000));
        assert!(!held.lock().left);
        // The last worker takes it along, window and all: the table holds
        // no model that lists nobody, and no worker's index names it.
        scorer.on_membership(&Membership::Removed(pool[2].id));
        assert!(scorer.partitions.load().is_empty());
        assert!(scorer.listed.load().is_empty());
        assert!(held.lock().left, "a decision that holds it takes the next");
        // The three are fenced: a view that still shows them lists none,
        // and the partition its decision made leaves the table again.
        assert_eq!(led(&scorer, &pool, 3_000, 16 * SECOND), None);
        assert!(scorer.partitions.load().is_empty());
        // The feed names them again, and the model's next decision lists
        // them from an even split.
        for worker in &pool {
            scorer.on_membership(&Membership::Added(worker.id));
        }
        assert_eq!(led(&scorer, &pool, 3_000, 17 * SECOND), Some(2));
        assert_eq!(loads(&scorer), (vec![0, 0, 0], 0));
    }

    /// A worker a registration took off a model is unlisted under that
    /// model alone and keeps its place, its range and its load under the
    /// models the word does not name, where its next booking alone lands.
    /// The words of a row's facts unlist nobody, and a model whose last
    /// worker left this way takes its partition with it. The word fences
    /// nobody: a model that never listed the two lists them from its first
    /// view.
    #[test]
    fn a_worker_a_registration_took_off_a_model_is_unlisted_under_that_model_alone() {
        let scorer = scorer(NEVER, 1.5, 5);
        let (pool_a, pool_b) = (pool(2), pool(4));
        for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
            led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
        }
        let moved = pool_a[1].id;
        book(&scorer, &pool_a, 1, 30, SECOND);
        let listing = |model: &str| listing_of(&scorer, model);
        let (a, b) = (listing("a"), listing("b"));
        assert_eq!(a.0, [(pool_a[0].id, 0), (moved, 30)]);
        // A row's facts changed: nobody is unlisted.
        for word in [
            Membership::Added(moved),
            Membership::Relabelled(moved),
            Membership::Resized(moved),
        ] {
            scorer.on_membership(&word);
        }
        assert_eq!((listing("a"), listing("b")), (a, b.clone()));
        // Off model a, and off a model no decision listed.
        scorer.on_membership(&Membership::Left {
            id: moved,
            models: &["a".into(), "c".into()],
        });
        assert_eq!(listing("a").0, [(pool_a[0].id, 0)]);
        assert_eq!(listing("b"), b, "a model the word does not name moved");
        // Its next booking lands under the model it serves, and there alone.
        book(&scorer, &pool_b, 1, 5, SECOND);
        assert_eq!(listing("a").2, 30, "booked under the model it left");
        assert_eq!(listing("b").0[1], (moved, 35));
        // The model's last worker leaves the same way, and the partition
        // with it.
        let held = of(&scorer, "a");
        scorer.on_membership(&Membership::Left {
            id: pool_a[0].id,
            models: &["a".into()],
        });
        assert!(scorer.partitions.load().get("a").is_none());
        assert!(held.lock().left, "a decision that holds it takes the next");
        assert_eq!(listing("b").0.len(), 4);
        // Both workers left a model a moment ago, and another lists them.
        led_by(
            &scorer,
            &prefill("d", 3_000, SECOND),
            &pool_a,
            &all(&pool_a),
        );
        assert_eq!(listing("d").0.len(), 2);
    }

    /// A new incarnation that comes back with a model less is unlisted
    /// under that model, by the word said for it behind its replacement's,
    /// and takes its predecessor's place, range and load under the rest. A
    /// decision of that model over an older view, between the two words
    /// and behind them, lists the old handle nowhere.
    #[test]
    fn a_new_incarnation_off_a_model_takes_its_predecessor_s_place_under_the_rest() {
        let scorer = scorer(NEVER, 1.5, 5);
        let (pool_a, pool_b) = (pool(2), pool(4));
        for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
            led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
        }
        let (old, new) = (pool_a[1].id, WorkerId(900));
        book(&scorer, &pool_a, 1, 30, SECOND);
        let b = listing_of(&scorer, "b");
        assert_eq!(b.0[1], (old, 30));
        // The replacement's word, then the word for the new handle. The
        // older view shows the old handle, which is fenced by the first.
        let older = |at| led_by(&scorer, &prefill("a", 3_000, at), &pool_a, &all(&pool_a));
        scorer.on_membership(&Membership::Replaced { old, new });
        older(2 * SECOND);
        let between = listing_of(&scorer, "a").0;
        assert!(between.iter().all(|&(id, _)| id != old), "{between:?}");
        scorer.on_membership(&Membership::Left {
            id: new,
            models: &["a".into()],
        });
        older(3 * SECOND);
        assert_eq!(listing_of(&scorer, "a").0, [(pool_a[0].id, 0)]);
        let after = listing_of(&scorer, "b");
        assert_eq!(after.0[1], (new, 30), "its predecessor's place and load");
        assert_eq!((after.0.len(), after.1, after.2), (4, b.1, b.2));
    }

    /// A new incarnation under another role left every model of its
    /// predecessor's list. The word names them all, so the replacement has
    /// no place to carry: the new handle is listed nowhere until a view of
    /// its own pool names it, and no place passes from one role to another.
    #[test]
    fn a_new_incarnation_under_another_role_takes_no_place_of_its_predecessor() {
        let scorer = scorer(NEVER, 1.5, 5);
        let (pool_a, pool_b) = (pool(2), pool(4));
        for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
            led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
        }
        let (old, new) = (pool_a[1].id, WorkerId(900));
        book(&scorer, &pool_a, 1, 30, SECOND);
        // The replacement's word, then the word for the new handle with
        // both models.
        scorer.on_membership(&Membership::Replaced { old, new });
        scorer.on_membership(&Membership::Left {
            id: new,
            models: &["a".into(), "b".into()],
        });
        let (a, b) = (listing_of(&scorer, "a").0, listing_of(&scorer, "b").0);
        assert_eq!((a.len(), b.len()), (1, 3));
        for (id, _) in a.iter().chain(&b) {
            assert!(*id != old && *id != new, "neither handle is listed");
        }
    }

    #[test]
    fn a_row_off_its_member_s_place_is_found_by_its_worker_s_id() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        // The same three workers in another order of the view.
        let turned: Vec<Candidate> = [2, 0, 1]
            .iter()
            .enumerate()
            .map(|(position, &row)| Candidate {
                position: position as u32,
                ..pool[row]
            })
            .collect();
        assert_eq!(led(&scorer, &turned, 3_000, SECOND), Some(0));
        assert_eq!(led(&scorer, &turned, 100, SECOND), Some(1));
        book(&scorer, &turned, 0, 50, SECOND);
        assert_eq!(loads(&scorer), (vec![0, 0, 50], 50));
        assert_eq!(ranges(&scorer).len(), 3, "no newcomer, no split");
        // Two rows of one worker are one member.
        let scorer = self::scorer(NEVER, 1.5, 5);
        let twice = [
            pool[0],
            Candidate {
                position: 1,
                dp_rank: 1,
                ..pool[0]
            },
            Candidate {
                position: 2,
                ..pool[1]
            },
        ];
        led(&scorer, &twice, 0, SECOND);
        assert_eq!(loads(&scorer), (vec![0, 0], 0));
    }

    #[test]
    fn all_rows_of_the_bucket_owner_score_alike() {
        let pool = pool(2);
        let mut candidates = [
            pool[0],
            Candidate {
                position: 1,
                dp_rank: 1,
                ..pool[0]
            },
            Candidate {
                position: 2,
                ..pool[1]
            },
        ];
        let scores = |candidates: &[Candidate]| {
            let scorer = scorer(NEVER, 1.5, 5);
            let mut out = vec![0.5; candidates.len()];
            assert_eq!(
                scorer.score(
                    &prefill("m", 0, SECOND),
                    candidates,
                    &all(candidates),
                    &mut out,
                ),
                TieBand::NORMALIZED
            );
            let score = |id, rank| {
                candidates
                    .iter()
                    .zip(&out)
                    .find(|(candidate, _)| candidate.id == id && candidate.dp_rank == rank)
                    .map(|(_, &score)| score)
            };
            [
                score(pool[0].id, 0),
                score(pool[0].id, 1),
                score(pool[1].id, 0),
            ]
        };
        assert_eq!(scores(&candidates), [Some(1.0), Some(1.0), Some(0.0)]);
        candidates.swap(0, 1);
        for (position, candidate) in candidates.iter_mut().enumerate() {
            candidate.position = position as u32;
        }
        assert_eq!(scores(&candidates), [Some(1.0), Some(1.0), Some(0.0)]);
    }

    #[test]
    fn the_balance_rule_spills_to_every_row_of_the_least_booked_worker() {
        let pool = pool(2);
        let candidates = [
            pool[0],
            Candidate {
                position: 1,
                dp_rank: 1,
                ..pool[0]
            },
            Candidate {
                position: 2,
                ..pool[1]
            },
        ];
        let scorer = scorer(32, 1.5, 5);
        let mut out = vec![0.5; candidates.len()];
        scorer.score(
            &prefill("m", 0, SECOND),
            &candidates,
            &all(&candidates),
            &mut out,
        );
        book(&scorer, &candidates, 2, 100, SECOND);

        // 3,000 is worker 101's size in the even split of two: worker 100's
        // rows lead only because 100 booked against 0 spills.
        scorer.score(
            &prefill("m", 3_000, SECOND),
            &candidates,
            &all(&candidates),
            &mut out,
        );
        assert_eq!(out, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn a_replaced_handle_keeps_its_place_and_a_snapshot_keeps_the_workers_it_names() {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        let cut = [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
        let fresh = WorkerId(900);
        scorer.on_membership(&Membership::Replaced {
            old: pool[1].id,
            new: fresh,
        });
        assert_eq!(ranges(&scorer), cut);
        assert_eq!(loads(&scorer), (vec![1_000, 2_000, 3_000], 6_000));
        // The view's row under the new handle is the member it replaced.
        let mut view = pool.clone();
        view[1].id = fresh;
        assert_eq!(led(&scorer, &view, 300, 16 * SECOND), Some(1));
        assert_eq!(ranges(&scorer), cut, "no newcomer, no split");
        // A booking finds the member under its new handle, and none under
        // the handle it replaced.
        book(&scorer, &view, 1, 5, 16 * SECOND);
        book(&scorer, &pool, 1, 7, 16 * SECOND);
        assert_eq!(loads(&scorer), (vec![1_000, 2_005, 3_000], 6_005));
        // An added, a relabelled or a resized worker changes nothing here.
        for change in [
            Membership::Added(WorkerId(901)),
            Membership::Relabelled(pool[0].id),
            Membership::Resized(pool[0].id),
        ] {
            scorer.on_membership(&change);
        }
        assert_eq!(ranges(&scorer), cut);
        // A snapshot names every worker of every pool: the listed ones it
        // leaves out are forgotten, the others it names are no members.
        scorer.on_membership(&Membership::Snapshot(&[
            pool[0].id,
            pool[2].id,
            WorkerId(77),
        ]));
        assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
        assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_005));
    }

    /// A lease's first dispatch as a served host tells it: the lease names
    /// the target the dispatch took by its `index` in the pick, and the
    /// target names its worker and no position.
    fn dispatch(
        scorer: &Bucket,
        worker: WorkerId,
        index: u8,
        tokens: u32,
        issued_micros: u64,
        since_micros: u32,
    ) {
        let lease = Lease {
            target_index: index,
            tokens,
            issued_micros,
            ..Lease::default()
        };
        let target = Target {
            id: worker,
            ..Target::default()
        };
        scorer.on_dispatched(&lease, &target, Side::Single, since_micros);
    }

    /// The date of the booking model `m` took last.
    fn last_booked_at(scorer: &Bucket) -> Option<u64> {
        let partition = of(scorer, "m");
        let pool = partition.lock();
        pool.window.back().map(|booking| booking.at_micros)
    }

    #[test]
    fn a_first_dispatch_on_another_target_books_the_worker_that_serves() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        // A lease is issued on its pick's first target, the first row here.
        book(&scorer, &pool, 0, 300, SECOND);
        // A dispatch on that target finds its booking made.
        dispatch(&scorer, pool[0].id, 0, 300, SECOND, 250);
        assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
        // Another decision books meanwhile, and the lease's gateway
        // dispatches on the pick's second target, the third row: that
        // worker serves and is booked, at the dispatch's instant. The
        // booking of the issue stands: nothing is un-booked.
        book(&scorer, &pool, 1, 10, SECOND + 400);
        dispatch(&scorer, pool[2].id, 1, 300, SECOND, 250);
        assert_eq!(loads(&scorer), (vec![300, 10, 300], 610));
        assert_eq!(last_booked_at(&scorer), Some(SECOND + 250));
        // The booking of the issue ages off the first target with its
        // interval, and the dispatch's stands on.
        book(&scorer, &pool, 1, 1, 6 * SECOND + 1);
        assert_eq!(loads(&scorer), (vec![0, 11, 300], 311));
        // A worker no partition lists takes no booking.
        dispatch(&scorer, WorkerId(999), 1, 300, 6 * SECOND, 250);
        assert_eq!(loads(&scorer), (vec![0, 11, 300], 311));
    }

    #[test]
    fn a_dispatch_s_booking_is_dated_no_later_than_the_newest_on_file() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        // A partition with no booking on file takes the gateway's date.
        dispatch(&scorer, pool[1].id, 1, 50, SECOND, 250);
        assert_eq!(last_booked_at(&scorer), Some(SECOND + 250));
        // A date far past the newest booking is a gateway's word: taken
        // whole it would age every booking on file. The booking is dated
        // at the newest.
        book(&scorer, &pool, 0, 40, 2 * SECOND);
        dispatch(&scorer, pool[1].id, 1, 7, 2 * SECOND, u32::MAX);
        assert_eq!(loads(&scorer), (vec![40, 57], 97));
        assert_eq!(last_booked_at(&scorer), Some(2 * SECOND));
        // And no earlier than its lease's issue, where the partition
        // booked nothing since.
        dispatch(&scorer, pool[1].id, 1, 3, 3 * SECOND, 250);
        assert_eq!(loads(&scorer), (vec![40, 60], 100));
        assert_eq!(last_booked_at(&scorer), Some(3 * SECOND));
    }

    #[test]
    fn a_dispatch_told_past_its_interval_books_nothing() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        // A lease is issued on the first row. Its booking has aged when
        // another decision books the second row two seconds later.
        book(&scorer, &pool, 0, 300, SECOND);
        book(&scorer, &pool, 1, 10, 3 * SECOND);
        assert_eq!(loads(&scorer), (vec![0, 10, 0], 10));
        // The first lease's dispatch on the third row is told then, dated
        // 250 microseconds behind its decision: more than an interval
        // before the booking on file. It aged before it was told, and books
        // nothing.
        dispatch(&scorer, pool[2].id, 1, 300, SECOND, 250);
        assert_eq!(loads(&scorer), (vec![0, 10, 0], 10));
        assert_eq!(last_booked_at(&scorer), Some(3 * SECOND));
    }

    #[test]
    fn a_dispatch_s_booking_ends_no_booking_newer_than_itself() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        let tenth = SECOND / 10;
        led(&scorer, &pool, 0, 10 * SECOND);
        for (tokens, at_micros) in [(1, 100 * tenth), (2, 105 * tenth), (4, 107 * tenth)] {
            book(&scorer, &pool, 0, tokens, at_micros);
        }
        // A dispatch dated one interval before the oldest booking, to the
        // microsecond, arrived out of its clock's order: it is booked,
        // behind the three.
        dispatch(&scorer, pool[1].id, 1, 8, 90 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![7, 8, 0], 15));
        assert_eq!(last_booked_at(&scorer), Some(90 * tenth));
        // A dispatch dated more than an interval before the oldest booking
        // is held against that one, not against the last one made: it books
        // nothing, and the three newer bookings stand.
        dispatch(&scorer, pool[2].id, 1, 16, 89 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![7, 8, 0], 15));
    }

    #[test]
    fn a_late_dispatch_into_a_full_window_ends_no_newer_booking() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        let tenth = SECOND / 10;
        led(&scorer, &pool, 0, 10 * SECOND);
        book(&scorer, &pool, 0, 1, 100 * tenth);
        for _ in 2..WINDOW_CAP {
            book(&scorer, &pool, 1, 1, 105 * tenth);
        }
        let full = WINDOW_CAP as u64;
        assert_eq!(loads(&scorer), (vec![1, full - 2, 0], full - 1));
        // One booking short of its cap the window loses no front unaged,
        // and the front alone holds the date: a dispatch dated within an
        // interval of it is booked, and fills the window.
        dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![1, full - 2, 8], full + 7));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
        // A full window loses its front to the cap, whatever its age, and
        // the booking behind it is the first one aged. The same dispatch
        // now books nothing: more than an interval before that booking, it
        // aged before it was told.
        dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![1, full - 2, 8], full + 7));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
        // One dated exactly an interval before that booking is booked: the
        // guard admits a date exactly an interval away, and the cap ends
        // the front alone.
        dispatch(&scorer, pool[2].id, 1, 8, 95 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![0, full - 2, 16], full + 14));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    }

    #[test]
    fn a_late_dispatch_is_held_against_the_front_of_a_full_window_too() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        let tenth = SECOND / 10;
        led(&scorer, &pool, 0, 10 * SECOND);
        // The second booking arrived out of its clock's order, an interval
        // before the first, and the window fills behind the two.
        book(&scorer, &pool, 0, 1, 100 * tenth);
        book(&scorer, &pool, 1, 1, 90 * tenth);
        for _ in 2..WINDOW_CAP {
            book(&scorer, &pool, 1, 1, 105 * tenth);
        }
        let full = WINDOW_CAP as u64;
        assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
        // A dispatch dated within an interval of the booking behind the
        // front, and more than an interval before the front, aged before it
        // was told, as it did in a window short of its cap: it books
        // nothing.
        dispatch(&scorer, pool[2].id, 1, 8, 85 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
        // One dated within an interval of both is booked, and the cap ends
        // the front alone: the booking step stops at the one behind it,
        // and the newer bookings behind that one stand.
        dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![0, full - 1, 8], full + 7));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    }

    #[test]
    fn a_late_dispatch_ends_no_booking_newer_than_its_date() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        let tenth = SECOND / 10;
        led(&scorer, &pool, 0, 10 * SECOND);
        // Three bookings arrive out of their clocks' order, and the window
        // fills behind them. The first of the newer ones ends the first of
        // the three as aged: the full window stands on a front, an older
        // booking behind it, and newer ones behind that.
        book(&scorer, &pool, 0, 1, 96 * tenth);
        book(&scorer, &pool, 0, 1, 105 * tenth);
        book(&scorer, &pool, 1, 1, 90 * tenth);
        for _ in 2..WINDOW_CAP {
            book(&scorer, &pool, 1, 1, 112 * tenth);
        }
        let full = WINDOW_CAP as u64;
        assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
        // A dispatch dated no more than an interval before the front, and
        // no more than an interval before the booking behind it, is booked.
        // The cap ends the front, and the booking behind it leaves as aged,
        // more than an interval older than the dispatch. The newer ones
        // stand: the dispatch's date is late, and no clock that went back.
        dispatch(&scorer, pool[2].id, 1, 8, 101 * tenth, 0);
        assert_eq!(loads(&scorer), (vec![0, full - 2, 8], full + 6));
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP - 1);
    }

    #[test]
    fn a_late_dispatch_ends_no_newer_booking_short_of_the_cap_either() {
        let scorer = scorer(NEVER, 1.5, 1);
        let pool = pool(3);
        let hundredth = SECOND / 100;
        led(&scorer, &pool, 0, 100 * SECOND);
        // Two issues an interval apart, and a dispatch dated before both,
        // within an interval of the front.
        book(&scorer, &pool, 0, 1, 10_020 * hundredth);
        book(&scorer, &pool, 0, 2, 10_120 * hundredth);
        dispatch(&scorer, pool[2].id, 1, 4, 9_920 * hundredth, 100_000);
        assert_eq!(loads(&scorer), (vec![3, 0, 4], 7));
        // An issue at 101.9 s ends the front as aged. One at 100.15 s, a
        // clock that stepped back, ends the next front. The dispatch's
        // booking stands at the front, the booking of 101.9 s behind it.
        book(&scorer, &pool, 1, 8, 10_190 * hundredth);
        book(&scorer, &pool, 1, 16, 10_015 * hundredth);
        assert_eq!(loads(&scorer), (vec![0, 24, 4], 28));
        // A dispatch dated 100.5 s: more than an interval after the front,
        // and more than an interval before the booking behind it. The front
        // leaves as aged. The newer booking stands, in a window far short
        // of its cap.
        dispatch(&scorer, pool[2].id, 1, 32, 10_050 * hundredth, 0);
        assert_eq!(loads(&scorer), (vec![0, 24, 32], 56));
        assert_eq!(of(&scorer, "m").lock().window.len(), 3);
    }

    #[test]
    fn a_dispatch_books_each_partition_that_lists_its_worker() {
        let scorer = scorer(NEVER, 1.5, 5);
        let (pool_a, pool_b) = (pool(2), pool(4));
        for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
            led_by(&scorer, &prefill(model, 0, SECOND), pool, &all(pool));
        }
        // Both models list the second row, and one alone the fourth.
        dispatch(&scorer, pool_a[1].id, 1, 30, SECOND, 250);
        dispatch(&scorer, pool_b[3].id, 1, 9, SECOND, 250);
        let booked = |model: &str| {
            let partition = of(&scorer, model);
            let held = partition.lock();
            let loads: Vec<u64> = held.members.iter().map(|member| member.load).collect();
            (loads, held.total)
        };
        assert_eq!(booked("a"), (vec![0, 30], 30));
        assert_eq!(booked("b"), (vec![0, 30, 0, 9], 39));
    }

    #[test]
    fn a_dispatch_on_another_target_of_the_issue_s_worker_books_nothing() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        book(&scorer, &pool, 0, 300, SECOND);
        // The lease its host tells names the worker its issue booked.
        let told = |index: u8, target: Target| {
            let lease = Lease {
                target_index: index,
                booked: pool[0].id,
                tokens: 300,
                issued_micros: SECOND,
                ..Lease::default()
            };
            scorer.on_dispatched(&lease, &target, Side::Single, 250);
        };
        // The pick's second target is another rank of that worker: the
        // booking of the issue is the serving worker's own.
        told(
            1,
            Target {
                id: pool[0].id,
                dp_rank: 1,
                ..Target::default()
            },
        );
        assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
        // Its third target is another worker, which is booked.
        told(
            2,
            Target {
                id: pool[2].id,
                ..Target::default()
            },
        );
        assert_eq!(loads(&scorer), (vec![300, 0, 300], 600));
    }

    #[test]
    fn a_pair_pick_s_second_target_books_its_decode_and_not_the_prefill_again() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        // Each target of a pair pick names the one prefill, the first row,
        // with another decode: the second row, then the third.
        let target = |decode: u32| Target {
            position: 0,
            id: pool[0].id,
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: decode,
                decode_id: pool[decode as usize].id,
                ..Pair::default()
            }),
        };
        let mut targets = Targets::new();
        targets.push(target(1));
        targets.push(target(2));
        let decision = Decision::Pick(Pick {
            targets,
            ..Pick::default()
        });
        let lease = |side: Side, booked: WorkerId, index: u8| Lease {
            side,
            target_index: index,
            booked,
            tokens: 300,
            issued_micros: SECOND,
            ..Lease::default()
        };
        // The prefill side: its issue books the prefill, and a dispatch on
        // the second target names that prefill again.
        let prefill = pool[0].id;
        scorer.on_decision_issued(&lease(Side::Prefill, prefill, 0), &decision);
        scorer.on_dispatched(
            &lease(Side::Prefill, prefill, 1),
            &target(2),
            Side::Prefill,
            250,
        );
        assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
        // The decode side: its issue books the first target's decode, and
        // a dispatch on the second target books that target's own.
        let decode = pool[1].id;
        scorer.on_decision_issued(&lease(Side::Decode, decode, 0), &decision);
        assert_eq!(loads(&scorer), (vec![300, 300, 0], 600));
        scorer.on_dispatched(
            &lease(Side::Decode, decode, 1),
            &target(2),
            Side::Decode,
            250,
        );
        assert_eq!(loads(&scorer), (vec![300, 300, 300], 900));
    }

    /// A pair's decode that goes out with no decode call is told issued at
    /// its dispatch, on the target the dispatch took, and then dispatched
    /// as booked there, as a store tells both words: no position, the
    /// issue dated when the store is told. One booking stands on the
    /// worker that serves, whichever target was taken, and a booking made
    /// since the dispatch stands with it. Dated as the gateway dates the
    /// dispatch, more than a window behind that booking, the same issue
    /// would end it: a clock that went back.
    #[test]
    fn a_decode_issued_at_its_dispatch_books_once_and_ends_no_newer_booking() {
        // The pair was decided at one second and its decode sent 250
        // microseconds later. The store is told at three seconds, behind
        // a booking of the fourth row made then, under a window of one.
        let datings = [(3 * SECOND, 1_000_u64), (SECOND + 250, 0)];
        for taken in [0_u8, 1] {
            for (issued_micros, stands) in datings {
                let scorer = scorer(NEVER, 1.5, 1);
                let pool = pool(4);
                led(&scorer, &pool, 0, SECOND);
                book(&scorer, &pool, 3, 1_000, 3 * SECOND);
                // Each target names the one prefill, the first row, with
                // another decode: the second row, then the third.
                let target = |decode: usize| Target {
                    id: pool[0].id,
                    pair: Some(Pair {
                        decode_id: pool[decode].id,
                        ..Pair::default()
                    }),
                    ..Target::default()
                };
                let mut targets = Targets::new();
                targets.push(target(1));
                targets.push(target(2));
                let decision = Decision::Pick(Pick {
                    targets,
                    ..Pick::default()
                });
                let serves = 1 + usize::from(taken);
                let lease = |issued_micros: u64| Lease {
                    side: Side::Decode,
                    target_index: taken,
                    booked: pool[serves].id,
                    tokens: 300,
                    issued_micros,
                    ..Lease::default()
                };
                scorer.on_decision_issued(&lease(issued_micros), &decision);
                let sent = Target {
                    id: pool[serves].id,
                    ..Target::default()
                };
                scorer.on_dispatched(&lease(SECOND), &sent, Side::Decode, 250);
                let mut booked = vec![0, 0, 0, stands];
                booked[serves] = 300;
                let read = loads(&scorer);
                assert_eq!(
                    read,
                    (booked, 300 + stands),
                    "target {taken} at {issued_micros}"
                );
            }
        }
    }

    /// A pair's decode side as the store files it. The lease the store
    /// derives for the side names the side's own worker as the one its
    /// issue booked: a dispatch on another rank of that worker leaves the
    /// booking of the issue as it stands.
    #[test]
    fn a_decode_side_s_dispatch_on_another_rank_of_its_worker_books_nothing() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        // Both targets of the pick name the one prefill, the first row,
        // and two ranks of one decode, the second row.
        let mut targets = Targets::new();
        for rank in [0, 1] {
            targets.push(Target {
                position: 0,
                id: pool[0].id,
                dp_rank: 0,
                pair: Some(Pair {
                    decode_position: 1,
                    decode_id: pool[1].id,
                    decode_dp_rank: rank,
                    ..Pair::default()
                }),
            });
        }
        let picked = crate::store::PickRecord::of_decodes(&targets);
        let decision = Decision::Pick(Pick {
            targets,
            ..Pick::default()
        });
        // The record's own lease as its decider issues it: it books the
        // pick's first target, the prefill.
        let prefill = Lease {
            side: Side::Prefill,
            target_index: 0,
            booked: pool[0].id,
            tokens: 300,
            issued_micros: SECOND,
            ..Lease::default()
        };
        let side = crate::store::DecodeSide {
            target: pool[1].id,
            picked,
            session: view::Session::NONE,
            state: crate::store::LeaseState::ISSUED,
            tokens: 300,
            target_index: 0,
        };
        let decode = side.lease(&prefill);
        scorer.on_decision_issued(&decode, &decision);
        assert_eq!(loads(&scorer), (vec![0, 300, 0], 300));
        // The side's first dispatch, on the second target, as the store
        // tells it: the side's lease with the frame's index, on the side's
        // own worker.
        let told = Lease {
            target_index: 1,
            ..decode
        };
        let on = Target {
            id: pool[1].id,
            dp_rank: 1,
            ..Target::default()
        };
        scorer.on_dispatched(&told, &on, Side::Decode, 250);
        assert_eq!(loads(&scorer), (vec![0, 300, 0], 300), "booked once");
    }

    #[test]
    fn a_decode_side_lease_books_its_pair_s_decode_and_not_the_prefill() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        let paired = |decode_position: u32, decode_id: WorkerId| {
            let mut targets = Targets::new();
            targets.push(Target {
                position: 0,
                id: pool[0].id,
                dp_rank: 0,
                pair: Some(Pair {
                    decode_position,
                    decode_id,
                    ..Pair::default()
                }),
            });
            Decision::Pick(Pick {
                targets,
                ..Pick::default()
            })
        };
        let lease = |side: Side| Lease {
            side,
            tokens: 300,
            issued_micros: SECOND,
            ..Lease::default()
        };
        scorer.on_decision_issued(&lease(Side::Decode), &paired(7, WorkerId(907)));
        assert_eq!(
            loads(&scorer),
            (vec![0, 0], 0),
            "a decode this partition does not list books nothing, and not on the pair's prefill"
        );
        scorer.on_decision_issued(&lease(Side::Prefill), &paired(7, WorkerId(907)));
        assert_eq!(
            loads(&scorer),
            (vec![300, 0], 300),
            "the prefill side books the prefill"
        );
        scorer.on_decision_issued(&lease(Side::Decode), &paired(1, pool[1].id));
        assert_eq!(
            loads(&scorer),
            (vec![300, 300], 600),
            "a decode the partition lists is booked as itself"
        );
    }

    #[test]
    fn each_model_has_a_partition_of_its_own() {
        let scorer = scorer(NEVER, 1.5, 5);
        let (pool_a, pool_b) = (pool(2), pool(4));
        let over = |model, tokens, pool: &[Candidate]| {
            led_by(&scorer, &prefill(model, tokens, SECOND), pool, &all(pool))
        };
        assert_eq!(over("a", 3_000, &pool_a), Some(1));
        assert_eq!(over("b", 3_000, &pool_b), Some(2));
        assert_eq!(
            over("a", 3_000, &pool_a),
            Some(1),
            "the other model listed nothing here"
        );
        // A worker two models list is booked in each of their partitions,
        // and a worker one model lists in that one alone.
        book(&scorer, &pool_a, 1, 30, SECOND);
        book(&scorer, &pool_b, 3, 9, SECOND);
        let booked = |model: &str| {
            let partition = of(&scorer, model);
            let held = partition.lock();
            let loads: Vec<u64> = held.members.iter().map(|member| member.load).collect();
            (loads, held.total)
        };
        assert_eq!(booked("a"), (vec![0, 30], 30));
        assert_eq!(booked("b"), (vec![0, 30, 0, 9], 39));
        let partitions = scorer.partitions.load();
        let mut listed: Vec<(&str, usize)> = partitions
            .iter()
            .map(|(model, partition)| (&**model, partition.lock().members.len()))
            .collect();
        listed.sort_unstable();
        assert_eq!(listed, [("a", 2), ("b", 4)]);
    }

    #[test]
    fn a_window_past_its_cap_drops_its_oldest_booking() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        book(&scorer, &pool, 1, 1_000, SECOND);
        for _ in 0..WINDOW_CAP {
            book(&scorer, &pool, 0, 3, SECOND);
        }
        // The clock stood still and the first booking left all the same.
        assert_eq!(
            loads(&scorer),
            (vec![3 * WINDOW_CAP as u64, 0], 3 * WINDOW_CAP as u64)
        );
        assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    }

    #[test]
    fn the_preset_places_a_prefill_request_by_its_size_and_books_what_is_issued() {
        let engine = PolicySpec::preset("bucket")
            .expect("the preset's form")
            .validate()
            .expect("the preset validates");
        let view = pool(4);
        let facts = PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        };
        let mut scratch = Scratch::new();
        let mut placed = Vec::new();
        for seed in 1..=3 {
            let request = RequestFacts {
                seed,
                ..prefill("m", 3_000, SECOND)
            };
            let decision = engine.decide(
                &request,
                &facts,
                &view,
                &Constraints::default(),
                &mut scratch,
            );
            let lease = Lease {
                tokens: request.prompt_tokens,
                issued_micros: request.now_micros,
                ..Lease::default()
            };
            engine.on_decision_issued(&lease, &decision);
            let pick = decision.pick().expect("a pick");
            placed.push(pick.targets.first().expect("a target").position);
        }
        // The owner of 3,000 tokens in an even split of four, then the
        // preset's balance rule (64 tokens and 1.5): the first idle worker,
        // then the next.
        assert_eq!(placed, [2, 0, 1]);
    }

    #[test]
    fn a_host_that_makes_no_lifecycle_call_refuses_the_stage_and_the_preset() {
        let served = StageRegistry::builtin();
        let unserved = StageRegistry::builtin().without_lifecycle();
        let named: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["bucket"],"pick":"best"}}}"#,
        )
        .expect("a well-formed document");
        let preset = PolicySpec::preset("bucket").expect("the preset's form");
        for (spec, path) in [
            (&named, "profiles.p.score[0].bucket"),
            (&preset, "profiles.bucket.score[0].bucket"),
        ] {
            let engine = spec
                .validate_in(&served, &[])
                .expect("the host makes the calls");
            assert!(engine.needs_lifecycle(), "{path}");
            assert_eq!(
                spec.validate_in(&unserved, &[]).err(),
                Some(SpecError::NotInBuild {
                    path: path.to_string(),
                    what: "the lifecycle this stage needs",
                }),
                "{path}"
            );
        }
    }

    #[test]
    fn a_share_exactly_twice_or_half_the_last_cut_s_is_not_cut_again() {
        // The ranges at twenty seconds, the window's tokens and the share
        // the ranges were cut for, after the cut for 2,000 a worker at
        // fifteen and `sizes` booked at `at_micros`.
        let second_cut = |at_micros: u64, sizes: &[u32]| {
            let (scorer, pool) = three_classes();
            led(&scorer, &pool, 0, 15 * SECOND);
            for &tokens in sizes {
                book(&scorer, &pool, 0, tokens, at_micros);
            }
            led(&scorer, &pool, 0, 20 * SECOND);
            let cut_for = of(&scorer, "m").lock().cut_for;
            (ranges(&scorer), loads(&scorer).1, cut_for)
        };
        let first = vec![(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
        // At sixteen seconds the bookings of eleven still count: 6,000 more
        // make 12,000, and 4,000 a worker is exactly twice 2,000.
        assert_eq!(
            second_cut(16 * SECOND, &[1_000; 6]),
            (first.clone(), 12_000, 2_000)
        );
        // Three tokens more make it 4,001.
        assert_eq!(
            second_cut(16 * SECOND, &[1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 3]),
            (
                vec![(0, 0, 400), (1, 401, 1_000), (2, 1_001, u32::MAX)],
                12_003,
                4_001
            )
        );
        // A microsecond later the bookings of eleven are gone: 3,000 in the
        // window, and 1,000 a worker is exactly half of 2,000.
        assert_eq!(
            second_cut(16 * SECOND + 1, &[1_000; 3]),
            (first, 3_000, 2_000)
        );
        // Three tokens fewer make it 999.
        assert_eq!(
            second_cut(16 * SECOND + 1, &[1_000, 1_000, 997]),
            (
                vec![(0, 0, 997), (1, 998, 1_000), (2, 1_001, u32::MAX)],
                2_997,
                999
            )
        );
    }

    #[test]
    fn a_worker_whose_classes_add_up_to_exactly_its_share_takes_no_class_more() {
        // Three workers' ranges at the first cut over `sizes`, each one
        // booking, and the share they were cut for.
        let first_cut = |sizes: &[u32]| {
            let scorer = scorer(NEVER, 1.5, 5);
            let pool = pool(3);
            led(&scorer, &pool, 0, SECOND);
            for &tokens in sizes {
                book(&scorer, &pool, 0, tokens, SECOND);
            }
            led(&scorer, &pool, 0, 6 * SECOND);
            let cut_for = of(&scorer, "m").lock().cut_for;
            (ranges(&scorer), cut_for)
        };
        // Ten bookings of 100 tokens are the first class: 1,000 of the
        // window's 3,000, exactly a worker's share, and the first range ends
        // there. The second worker's two classes add up to its share too.
        let mut sizes = vec![100; 10];
        sizes.extend([200, 300, 500, 1_000]);
        assert_eq!(
            first_cut(&sizes),
            (vec![(0, 0, 100), (1, 101, 500), (2, 501, u32::MAX)], 1_000)
        );
        // Three tokens more in the window make the share 1,001: the first
        // class is one token short of it, and its worker takes the next.
        sizes[13] = 1_003;
        assert_eq!(
            first_cut(&sizes),
            (vec![(0, 0, 200), (1, 201, 500), (2, 501, u32::MAX)], 1_001)
        );
    }

    #[test]
    fn a_handle_a_view_listed_before_the_word_of_its_replacement_stands_once() {
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        // A view shows the second worker under its new handle before the
        // word of the replacement: the row joins as a third member.
        let fresh = WorkerId(900);
        let mut view = pool.clone();
        view[1].id = fresh;
        led(&scorer, &view, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 40, 0], 140));
        // The word comes: the worker stands once, at the place the view gave
        // it, with the tokens booked under the handle it replaced, whose
        // member leaves as a removed worker's does.
        scorer.on_membership(&Membership::Replaced {
            old: pool[1].id,
            new: fresh,
        });
        assert_eq!(loads(&scorer), (vec![100, 40], 140));
        assert_eq!(ranges(&scorer), [(0, 0, 2_729), (1, 2_730, u32::MAX)]);
        // One lease of the worker is one booking.
        book(&scorer, &view, 1, 70, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 110], 210));
        // A replacement by the same handle changes nothing.
        scorer.on_membership(&Membership::Replaced {
            old: fresh,
            new: fresh,
        });
        assert_eq!(loads(&scorer), (vec![100, 110], 210));
        assert_eq!(ranges(&scorer), [(0, 0, 2_729), (1, 2_730, u32::MAX)]);
        // The bookings that moved age off the handle that stays: one booked
        // past the window ends them all.
        book(&scorer, &view, 0, 1, 20 * SECOND);
        assert_eq!(loads(&scorer), (vec![1, 0], 1));
    }

    #[test]
    fn a_view_older_than_a_leave_lists_nobody_splits_nothing_and_places_the_others() {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        scorer.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
        // A decision whose view was captured before the leave still shows
        // the worker. It joins nowhere, the ranges stand, and each size
        // goes to the worker that owns it, at the row the view shows it in.
        assert_eq!(led(&scorer, &pool, 300, 16 * SECOND), Some(0));
        assert_eq!(led(&scorer, &pool, 2_000, 16 * SECOND), Some(2));
        assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
        assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_000));
        assert_eq!(scorer.listed.load().len(), 2, "the index names the two");
        // The leaver alone among the survivors: nobody is seated, and the
        // picker's draw places the request.
        assert_eq!(
            led_by(&scorer, &prefill("m", 300, 16 * SECOND), &pool, &[1]),
            None
        );
        assert_eq!(ranges(&scorer).len(), 2);
    }

    #[test]
    fn a_partition_first_seen_through_a_view_older_than_a_leave_lists_the_others() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        // The worker left before the model's first prefill decision, whose
        // view was captured before the leave.
        scorer.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(led(&scorer, &pool, 3_000, SECOND), Some(2));
        assert_eq!(ranges(&scorer), [(0, 0, 2_047), (1, 2_048, u32::MAX)]);
        assert_eq!(loads(&scorer), (vec![0, 0], 0));
    }

    #[test]
    fn a_view_of_fenced_rows_alone_leaves_no_partition_behind() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(2);
        led(&scorer, &pool, 0, SECOND);
        // The model's two workers leave, and its partition with the last.
        scorer.on_membership(&Membership::Removed(pool[0].id));
        scorer.on_membership(&Membership::Removed(pool[1].id));
        assert!(scorer.partitions.load().is_empty());
        // A decision whose view was captured before the leaves shows the
        // two alone. Nobody is listed, and the partition the decision made
        // leaves the table again: no later word of the feed would take it.
        assert_eq!(led(&scorer, &pool, 300, 2 * SECOND), None);
        assert!(scorer.partitions.load().is_empty(), "an empty entry stands");
        scorer.on_membership(&Membership::Snapshot(&[]));
        assert!(scorer.partitions.load().is_empty());
        assert!(scorer.listed.load().is_empty());
        // The feed names one of the two again, and the model's next view
        // lists it in a partition of its own.
        scorer.on_membership(&Membership::Added(pool[0].id));
        assert_eq!(led(&scorer, &pool, 300, 3 * SECOND), Some(0));
        assert_eq!(ranges(&scorer), [(0, 0, u32::MAX)]);

        // A decider that took the entry before it left finds it marked,
        // and takes the model's next one.
        let scorer = self::scorer(NEVER, 1.5, 5);
        scorer.on_membership(&Membership::Removed(pool[0].id));
        scorer.on_membership(&Membership::Removed(pool[1].id));
        let held = scorer.partition("m");
        assert_eq!(led(&scorer, &pool, 300, SECOND), None);
        assert!(held.lock().left, "a decision that holds it takes the next");
        assert!(scorer.partitions.load().is_empty());
    }

    /// The removal takes a partition out only where it still stands under
    /// its model: an entry made since under the same name stays.
    #[test]
    fn forgetting_a_partition_leaves_an_entry_made_since_under_its_model() {
        let scorer = scorer(NEVER, 1.5, 5);
        let left = scorer.partition("m");
        scorer.forget("m", &left);
        assert!(scorer.partitions.load().is_empty());
        let made = scorer.partition("m");
        scorer.forget("m", &left);
        assert!(Arc::ptr_eq(&of(&scorer, "m"), &made));
        scorer.forget("m", &made);
        assert!(scorer.partitions.load().is_empty());
    }

    /// The fence takes a leave's word before the leaver's partition forgets
    /// it, and holds no lock across the walk.
    #[test]
    fn a_leaver_is_fenced_while_its_partition_still_lists_it() {
        let scorer = Arc::new(scorer(NEVER, 1.5, 5));
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        let gone = pool[1].id;
        let partition = of(&scorer, "m");
        // A decision holds the leaver's partition while the feed's call
        // runs: the call fences the leaver and waits for the partition.
        let held = partition.lock();
        let feed = {
            let scorer = Arc::clone(&scorer);
            std::thread::spawn(move || scorer.on_membership(&Membership::Removed(gone)))
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let mut fenced = false;
        while !fenced && Instant::now() < deadline {
            fenced = scorer
                .fence
                .try_lock_for(std::time::Duration::from_millis(100))
                .is_some_and(|fence| fence.holds(gone));
            std::thread::yield_now();
        }
        assert!(
            held.by_id.contains_key(&gone),
            "the held partition lists it"
        );
        assert!(fenced, "no fence on the leaver while the walk waited");
        // The call holds no fence while it waits for the partition: the
        // fence stays free for a decision that lists.
        let free = (0..10).all(|_| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            scorer
                .fence
                .try_lock_for(std::time::Duration::from_millis(50))
                .is_some()
        });
        assert!(free, "the fence is held across the walk");
        drop(held);
        feed.join().expect("the feed's call");
        assert!(!partition.lock().by_id.contains_key(&gone));
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer).0.len(), 2, "an older view lists it nowhere");
    }

    /// A snapshot fences its leavers before its walk reads the table: a
    /// partition a decision makes while the walk waits lists none of them.
    #[test]
    fn a_snapshot_s_leaver_is_fenced_before_a_partition_made_during_its_walk() {
        let scorer = Arc::new(scorer(NEVER, 1.5, 5));
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        let gone = pool[1].id;
        let stay = [pool[0].id, pool[2].id];
        let partition = of(&scorer, "m");
        // A decision holds the leaver's partition while the feed's call
        // runs: the walk has read the table and waits for the partition.
        let held = partition.lock();
        let feed = {
            let scorer = Arc::clone(&scorer);
            std::thread::spawn(move || scorer.on_membership(&Membership::Snapshot(&stay)))
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let mut fenced = false;
        while !fenced && Instant::now() < deadline {
            fenced = scorer
                .fence
                .try_lock_for(std::time::Duration::from_millis(100))
                .is_some_and(|fence| fence.holds(gone));
            std::thread::yield_now();
        }
        // Another model's first decision, over a view older than the
        // snapshot, makes its partition behind the walk's read of the
        // table: the walk never visits it.
        let _ = led_by(&scorer, &prefill("n", 0, SECOND), &pool, &all(&pool));
        drop(held);
        feed.join().expect("the feed's call");
        let listed = |model: &str| -> Vec<WorkerId> {
            of(&scorer, model)
                .lock()
                .members
                .iter()
                .map(|member| member.id)
                .collect()
        };
        assert_eq!(listed("n"), stay, "the partition made during the walk");
        assert_eq!(listed("m"), stay, "the partition the walk waited for");
        assert!(fenced, "no fence on the leaver while the walk waited");
    }

    /// A worker a decision was listing when the snapshot's word came is in
    /// no index yet: the walk fences it before its partition forgets it.
    #[test]
    fn a_snapshot_fences_a_leaver_its_partition_lists_and_the_index_does_not() {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool[..2], 0, SECOND);
        // The third row joins under the partition's guard, as at a
        // decision that has not written the index yet.
        of(&scorer, "m").lock().list(&pool, |_| false);
        let gone = pool[2].id;
        assert!(!scorer.listed.load().contains_key(&gone));
        scorer.on_membership(&Membership::Snapshot(&[pool[0].id, pool[1].id]));
        assert!(scorer.fence.lock().holds(gone), "the walk fenced nobody");
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer).0.len(), 2, "an older view lists it nowhere");
    }

    /// A decision asks the fence of every row of its view the partition does
    /// not list, the filters' dropped rows too: a fenced row the filters
    /// dropped joins nowhere when another row of the view joins.
    #[test]
    fn a_fenced_row_the_filters_dropped_joins_nowhere_beside_a_new_row() {
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        let gone = pool[1].id;
        scorer.on_membership(&Membership::Removed(gone));
        let wider = self::pool(3);
        let _ = led_by(&scorer, &prefill("m", 0, 2 * SECOND), &wider, &[0, 2]);
        let listed: Vec<WorkerId> = of(&scorer, "m")
            .lock()
            .members
            .iter()
            .map(|member| member.id)
            .collect();
        assert_eq!(listed, [wider[0].id, wider[2].id]);
    }

    /// A view older than two leaves places the rows behind the leavers as
    /// a view taken after the leaves does, whether the filters kept a
    /// leaver's row or dropped it.
    #[test]
    fn a_view_older_than_two_leaves_places_the_rows_behind_them_as_a_new_view_does() {
        let (scorer, pool) = booked(NEVER, 1.5, &[10, 20, 30, 40, 50, 60]);
        for row in [1, 3] {
            scorer.on_membership(&Membership::Removed(pool[row].id));
        }
        let mut new: Vec<Candidate> = [0, 2, 4, 5].map(|row: usize| pool[row]).to_vec();
        for (position, row) in new.iter_mut().enumerate() {
            row.position = position as u32;
        }
        // The old view's first leaver passed the filters and its second did
        // not.
        let kept = [0, 1, 2, 4, 5];
        for tokens in [0, 1_000, 2_000, 3_000, 4_000] {
            let facts = prefill("m", tokens, 2 * SECOND);
            let by_new = led(&scorer, &new, tokens, 2 * SECOND).map(|row| new[row as usize].id);
            let by_old = led_by(&scorer, &facts, &pool, &kept).map(|row| pool[row as usize].id);
            assert_eq!(by_old, by_new, "{tokens} tokens");
            assert!(by_old.is_some(), "{tokens} tokens have an owner");
        }
        // The leavers' bookings stay in the window and on no worker.
        assert_eq!(loads(&scorer), (vec![10, 30, 50, 60], 210));
    }

    /// The fenced rows a partition keeps are its last listing's alone: the
    /// next listing starts from none, so the list never outgrows a view.
    #[test]
    fn the_fenced_rows_a_partition_keeps_are_its_last_listings_alone() {
        let (scorer, pool) = booked(NEVER, 1.5, &[10, 20, 30]);
        let gone = pool[1].id;
        scorer.on_membership(&Membership::Removed(gone));
        for _ in 0..3 {
            led(&scorer, &pool, 0, 2 * SECOND);
            assert_eq!(of(&scorer, "m").lock().fenced, [(1, gone)]);
        }
        // The feed names the worker again: the old view's next decision
        // lists it, and that listing keeps no fenced row.
        scorer.on_membership(&Membership::Added(gone));
        led(&scorer, &pool, 0, 2 * SECOND);
        assert!(of(&scorer, "m").lock().fenced.is_empty());
        assert_eq!(loads(&scorer).0.len(), 3);
    }

    #[test]
    fn the_feed_naming_a_worker_again_lifts_its_fence() {
        // Added: the worker joins from the next view that shows it.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        let gone = pool[1].id;
        scorer.on_membership(&Membership::Removed(gone));
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100], 140), "fenced");
        scorer.on_membership(&Membership::Added(gone));
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 0], 140), "named again");

        // A replacing handle: the id comes back as the new handle of a
        // worker this model never listed.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        scorer.on_membership(&Membership::Removed(gone));
        scorer.on_membership(&Membership::Replaced {
            old: WorkerId(7),
            new: gone,
        });
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 0], 140));

        // A snapshot that names it.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        scorer.on_membership(&Membership::Removed(gone));
        scorer.on_membership(&Membership::Snapshot(&[pool[0].id, gone]));
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 0], 140));
    }

    #[test]
    fn a_replaced_handle_and_a_snapshot_s_leavers_are_fenced_as_a_removed_worker_is() {
        // The old handle of a replacement: a view older than the word still
        // shows it, and it joins as nobody.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        let fresh = WorkerId(900);
        scorer.on_membership(&Membership::Replaced {
            old: pool[1].id,
            new: fresh,
        });
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 40], 140), "it stands once");
        // A replacement by the same handle fences nothing.
        scorer.on_membership(&Membership::Replaced {
            old: fresh,
            new: fresh,
        });
        assert!(!scorer.fence.lock().is_retired(fresh, Instant::now()));

        // The workers a snapshot does not name.
        let (scorer, pool) = three_classes();
        scorer.on_membership(&Membership::Snapshot(&[pool[2].id]));
        assert_eq!(led(&scorer, &pool, 300, 12 * SECOND), Some(2));
        assert_eq!(loads(&scorer), (vec![3_000], 6_000));
    }

    #[test]
    fn the_fence_lapses_after_its_window_and_the_worker_joins_from_a_view_again() {
        let now = Instant::now();
        // A leave as the membership call makes it, dated half a fence back.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        let gone = pool[1].id;
        let half = now
            .checked_sub(RETIRED_WORKER_TTL / 2)
            .expect("half a fence back");
        scorer.fence.lock().retire(gone, half);
        of(&scorer, "m").lock().remove(gone);
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100], 140), "still fenced");

        // The same leave a whole fence back: the id may be a new worker's.
        let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
        let whole = now
            .checked_sub(RETIRED_WORKER_TTL)
            .expect("a whole fence back");
        scorer.fence.lock().retire(gone, whole);
        of(&scorer, "m").lock().remove(gone);
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(loads(&scorer), (vec![100, 0], 140), "lapsed");
        assert!(
            scorer.fence.lock().is_empty(),
            "the decision that listed forgot the lapsed retirement"
        );
    }
}
