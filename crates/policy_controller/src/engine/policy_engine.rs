//! The policy engine in a gateway's own process.
//!
//! The gateway keeps its workers where it keeps them with the classic
//! policies: its registry publishes one shared list of worker objects a
//! model at each membership change, and a worker's health and its vetoes
//! live on the worker object. The gateway's classic placement pass holds
//! the rules of a placement: which workers a request may take, what a retry
//! skips and what becomes of a pool no worker of which passes. This front
//! gives that pass its pick:
//!
//! - **Once a membership.** [`PolicyEngine::pool_members`] makes the
//!   engine's value of a pool's shared list: the list's handle and the
//!   pool's own word. The gateway keeps the value beside the list it was
//!   made from, for as long as that list stands. No worker is read for it
//!   and nothing of a worker is copied.
//! - **Once a request.** [`PolicyEngine::pick`] asks for one pick over the
//!   pool, under a profile the engine picks by its picker alone
//!   ([`PolicyEngine::live_profile`]). The pick turns the pool's own word
//!   and reads the worker the turn lands on, as that worker tests itself
//!   for routing: no row, no table, no request fact, no lock, and nothing
//!   allocated. A list nothing is kept of (a part of a pool, as the pass
//!   narrowed it) is passed over once ([`PolicyEngine::pick_among`]).
//!
//! The engine keeps no worker list, no health and no count of requests in
//! flight. What it keeps is its stages' own: a rotation.

use std::{
    cell::RefCell,
    hash::Hasher,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use rustc_hash::FxHasher;
use smg_worker::{identity::basic::Worker, runtime::RoutingState};

use super::framework::{LivePool, Picker};
use crate::document::spec::AppliedSpec;

/// The engine of one policy document, for a gateway that decides in its own
/// process.
///
/// Built once, at the gateway's start, from the document in force. It holds
/// the document's stages and their state, and no worker.
#[derive(Debug)]
pub struct PolicyEngine {
    spec: AppliedSpec,
    /// The pools made so far: each draws the place its turns start at from
    /// its number.
    epochs: AtomicU64,
}

/// One pool of one model as the engine keeps it: the gateway's shared list
/// of worker objects and the pool's own word.
///
/// Made by [`PolicyEngine::pool_members`] from a list the gateway's
/// registry published, and kept by the gateway beside that list. It holds
/// the list itself, not a copy: a worker's live facts are read on the
/// list's objects.
#[derive(Debug)]
pub struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>,
    /// The pool's own word, for the picker that places over the pool by
    /// itself ([`PolicyEngine::pick`]): a rotation counts its turns here.
    rotation: OwnWord,
}

/// A word every pick over a pool writes, on cache lines of its own: what
/// stands beside it, in this value and in the gateway's snapshot that keeps
/// the value, is read by every request and written by none.
#[derive(Debug)]
#[repr(align(128))]
struct OwnWord(AtomicU64);

/// A profile of the document one pick of whose picker is the profile's
/// whole decision ([`PolicyEngine::live_profile`]), by its place in the
/// document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveProfile(u16);

impl PoolMembers {
    /// The workers of the pool: the list the value was made from.
    #[must_use]
    pub fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.workers
    }
}

impl PolicyEngine {
    /// The engine of `spec`, the document in force.
    #[must_use]
    pub fn new(spec: AppliedSpec) -> Self {
        Self {
            spec,
            epochs: AtomicU64::new(0),
        }
    }

    /// The document the engine decides under.
    #[must_use]
    pub fn spec(&self) -> &AppliedSpec {
        &self.spec
    }

    /// The engine's value of one pool: `pool` is the shared list the
    /// gateway's registry published for it.
    ///
    /// Called once a list, by the gateway, which keeps the value for as
    /// long as the list stands. It reads no worker, so its cost does not
    /// grow with the list.
    #[must_use]
    pub fn pool_members(&self, pool: &Arc<[Arc<dyn Worker>]>) -> PoolMembers {
        let epoch = self.epochs.fetch_add(1, Ordering::Relaxed) + 1;
        PoolMembers {
            workers: Arc::clone(pool),
            // A pool starts its turns at a drawn worker, so a fleet whose
            // membership keeps changing does not lean on its first one.
            rotation: OwnWord(AtomicU64::new(mixed(epoch) >> 32)),
        }
    }

    /// The profile named `name` in the document, where one pick of its
    /// picker over a pool's workers is the profile's whole decision
    /// ([`Engine::picks_live`]). `None` where the document holds no profile
    /// of the name, or one whose decision is more than that pick.
    ///
    /// [`Engine::picks_live`]: super::framework::Engine::picks_live
    #[must_use]
    pub fn live_profile(&self, name: &str) -> Option<LiveProfile> {
        let engine = self.spec.engine();
        let at = engine
            .profiles()
            .iter()
            .position(|profile| profile.name == name)?;
        let index = u16::try_from(at).ok()?;
        engine.picks_live(index).then_some(LiveProfile(index))
    }

    /// One pick under `profile` over `members`, the value this engine made
    /// of a pool ([`Self::pool_members`]): the place in the pool's list of
    /// the worker picked. `None` where no worker of the pool passes.
    ///
    /// The entry for a caller that holds every other rule of a placement
    /// itself, as a gateway's classic placement pass does: which workers a
    /// request may take, what a retry skips, and what becomes of a pool no
    /// worker of which passes. The picker turns the pool's own word, kept
    /// in `members`, and the worker a turn lands on is read once and tested
    /// as the worker tests itself for routing: every veto, or every veto
    /// but the overload one where the caller lifted it (`overload_lifted`).
    /// No row is read, no table looked up, no lock taken and nothing
    /// allocated. Where the picker's turns all land on refused workers the
    /// pool is passed over once, as [`Self::pick_among`] passes over a list.
    #[must_use]
    pub fn pick(
        &self,
        profile: LiveProfile,
        members: &PoolMembers,
        overload_lifted: bool,
    ) -> Option<usize> {
        let picker = self.picker_of(profile)?;
        let workers = &*members.workers;
        let word = &members.rotation.0;
        if overload_lifted {
            pick_by(
                picker,
                workers,
                word,
                RoutingState::eligible_ignoring_overload,
            )
        } else {
            pick_by(picker, workers, word, RoutingState::eligible)
        }
    }

    /// One pick under `profile` among `workers`, a list this engine keeps
    /// nothing of (a part of a pool, as a caller narrowed it): the place in
    /// `workers` of the worker picked. `None` where no worker passes.
    ///
    /// The list is passed over once, a read and the worker's own test each,
    /// and the picker picks among the workers that pass: a rotation turns
    /// the one their set keeps, named by the workers' handles. The list of
    /// them is the calling thread's own: nothing is allocated after a
    /// thread's first picks.
    #[must_use]
    pub fn pick_among(
        &self,
        profile: LiveProfile,
        workers: &[Arc<dyn Worker>],
        overload_lifted: bool,
    ) -> Option<usize> {
        let picker = self.picker_of(profile)?;
        if overload_lifted {
            among(picker, workers, RoutingState::eligible_ignoring_overload)
        } else {
            among(picker, workers, RoutingState::eligible)
        }
    }

    /// The picker of `profile`.
    #[inline]
    fn picker_of(&self, profile: LiveProfile) -> Option<&dyn Picker> {
        let profiles = self.spec.engine().profiles();
        Some(profiles.get(usize::from(profile.0))?.picker.as_ref())
    }
}

/// [`PolicyEngine::pick`] by one test of a worker's read, compiled once a
/// test: the picker's own pick over the pool's word, then the pass.
#[inline(always)]
fn pick_by(
    picker: &dyn Picker,
    workers: &[Arc<dyn Worker>],
    word: &AtomicU64,
    admits: impl Fn(RoutingState) -> bool + Copy,
) -> Option<usize> {
    let pool = LivePool {
        rows: workers.len(),
        word,
    };
    let picked = picker.pick_live(&pool, &mut |at| {
        workers
            .get(at)
            .is_some_and(|worker| admits(worker.routing_state()))
    });
    if picked.is_some() {
        return picked;
    }
    among(picker, workers, admits)
}

/// [`PolicyEngine::pick_among`] by one test of a worker's read: the workers
/// of `workers` that pass, listed in the thread's list, their set named by
/// their handles' addresses in the list's order (one word hashed a worker,
/// and nothing of a worker read for it), and the picker's pick among them.
#[inline(always)]
fn among(
    picker: &dyn Picker,
    workers: &[Arc<dyn Worker>],
    admits: impl Fn(RoutingState) -> bool,
) -> Option<usize> {
    let pass = |passed: &mut Vec<u32>| {
        passed.clear();
        let mut set = FxHasher::default();
        for (worker, at) in workers.iter().zip(0u32..) {
            if admits(worker.routing_state()) {
                passed.push(at);
                set.write_usize(Arc::as_ptr(worker).addr());
            }
        }
        if passed.is_empty() {
            return None;
        }
        let at = picker.pick_among(set.finish(), passed.len());
        passed.get(at).map(|&position| position as usize)
    };
    PASSED.with(|passed| match passed.try_borrow_mut() {
        Ok(mut passed) => pass(&mut passed),
        // A picker asked for a pick from inside its own, on this thread:
        // that pick takes a list of its own.
        Err(_) => pass(&mut Vec::new()),
    })
}

thread_local! {
    /// The workers of a list that passed a pass, by their place in it: the
    /// calling thread's own list, so a pass allocates nothing after a
    /// thread's first picks.
    static PASSED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 finalizer: every bit of `z` moves every bit of the word.
const fn finalized(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A word drawn from a count: counts one apart give words far apart.
const fn mixed(count: u64) -> u64 {
    finalized(count.wrapping_mul(GOLDEN))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{runtime::StallReason, test_support::ready_worker};

    use super::*;
    use crate::document::spec::PolicySpec;

    const MODEL: &str = "model-a";

    fn engine_of(document: &PolicySpec) -> PolicyEngine {
        PolicyEngine::new(AppliedSpec::validate(document).expect("the document validates"))
    }

    fn preset(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        engine_of(&document)
    }

    fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    /// The value made of a pool holds the shared list itself, and making it
    /// reads no worker: a pool of workers that are not ready is made as any
    /// other, and each pool made draws a start of its own.
    #[test]
    fn the_members_hold_the_shared_list() {
        let engine = preset("round_robin");
        let list = pool(5);
        for worker in list.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
        let members = engine.pool_members(&list);
        assert!(Arc::ptr_eq(members.workers(), &list));
        let again = engine.pool_members(&list);
        assert!(Arc::ptr_eq(again.workers(), &list));
        assert_ne!(
            members.rotation.0.load(Ordering::Relaxed),
            again.rotation.0.load(Ordering::Relaxed),
            "each pool made draws the place its turns start at"
        );
    }

    /// A side of a pair holds a test the pick would drop: the prefill side's
    /// admission, the decode side's room. Neither side's profile is one a
    /// pick decides, whatever its picker.
    #[test]
    fn a_profile_with_a_test_beyond_health_is_not_picked_alone() {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
        document.generation = 1;
        let engine = engine_of(&document);
        assert!(engine.live_profile("prefill").is_none());
        assert!(engine.live_profile("decode").is_none());
        assert!(engine.live_profile("round_robin").is_none(), "not held");
    }

    /// The profiles one pick decides: the picker places by itself, no stage
    /// scores, and the filters hold both tests of a worker's health. Round
    /// robin's preset is one. A profile the document does not hold is none,
    /// and no other preset is one yet.
    #[test]
    fn a_live_profile_is_one_its_picker_decides_alone() {
        let engine = preset("round_robin");
        assert!(engine.live_profile("round_robin").is_some());
        assert!(engine.live_profile("least_load").is_none(), "not held");
        for name in [
            "random",
            "passthrough",
            "power_of_two",
            "least_load",
            "consistent_hashing",
            "prefix_hash",
            "manual",
            "cache_aware",
            "bucket",
        ] {
            assert!(preset(name).live_profile(name).is_none(), "{name}");
        }
    }

    /// With every worker ready a pick is the pool's own turn: one strict
    /// rotation at every pool size, from the position the pool drew.
    #[test]
    fn a_pick_turns_the_pools_own_rotation() {
        for workers in [1, 2, 3, 8, 16, 64] {
            let engine = preset("round_robin");
            let profile = engine.live_profile("round_robin").expect("picked alone");
            let members = engine.pool_members(&pool(workers));
            let first = engine.pick(profile, &members, false).expect("a worker");
            for step in 1..2 * workers {
                let at = engine.pick(profile, &members, false);
                assert_eq!(at, Some((first + step) % workers), "{workers} workers");
            }
        }
    }

    /// The rotation is the pool's own word: two pools of one engine turn a
    /// rotation each, and a turn of one moves no turn of the other.
    #[test]
    fn each_pool_turns_its_own_word() {
        let engine = preset("round_robin");
        let profile = engine.live_profile("round_robin").expect("picked alone");
        let (one, other) = (pool(5), pool(5));
        let (one, other) = (engine.pool_members(&one), engine.pool_members(&other));
        let first = engine.pick(profile, &one, false).expect("a worker");
        let apart = engine.pick(profile, &other, false).expect("a worker");
        for step in 1..=10 {
            assert_eq!(engine.pick(profile, &one, false), Some((first + step) % 5));
            assert_eq!(
                engine.pick(profile, &other, false),
                Some((apart + step) % 5)
            );
        }
    }

    /// Workers that are not ready take no request, and the ready ones take
    /// exact shares. In a small pool a turn on such a worker goes to the
    /// pass at once, and the ready workers' own rotation places the
    /// request. In a pool that earns turns the pool's next turn does. Past
    /// a run of such workers longer than those turns both rotations place
    /// requests, and both are exact.
    #[test]
    fn a_pick_keeps_exact_shares_past_workers_not_ready() {
        let engine = preset("round_robin");
        let profile = engine.live_profile("round_robin").expect("picked alone");
        let shares = |members: &PoolMembers, picks: usize| {
            let mut times = vec![0usize; members.workers().len()];
            for _ in 0..picks {
                times[engine.pick(profile, members, false).expect("a worker")] += 1;
            }
            times
        };
        let leave = |list: &[Arc<dyn Worker>], ready: &dyn Fn(usize) -> bool| {
            for (at, worker) in list.iter().enumerate() {
                if !ready(at) {
                    worker.set_status(WorkerStatus::NotReady);
                }
            }
        };

        // Eight workers, one not ready: seven turns of eight place by the
        // pool's turn and one by the seven ready workers' own rotation.
        let list = pool(8);
        leave(&list, &|at| at != 5);
        let members = engine.pool_members(&list);
        let times = shares(&members, 8 * 7 * 3);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at == 5 { 0 } else { 24 }, "worker {at} of 8");
        }

        // Sixty-four workers, one in four not ready, none beside another:
        // the pool's next turn places each request, and no pass runs.
        let list = pool(64);
        leave(&list, &|at| at % 4 != 1);
        let members = engine.pool_members(&list);
        let times = shares(&members, 48 * 5);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at % 4 == 1 { 0 } else { 5 }, "worker {at} of 64");
        }

        // Sixty-four workers, sixteen in a row not ready, three turns a
        // pick. From the pool's turn on the first of the run, each turn of
        // the pool places 53 requests: 48 by the pool's turn and five by
        // the ready workers' own rotation, which comes round once in 48
        // such picks. Over 48 turns of the pool each ready worker takes 53.
        let list = pool(64);
        leave(&list, &|at| !(20..36).contains(&at));
        let members = engine.pool_members(&list);
        members.rotation.0.store(20, Ordering::Relaxed);
        let times = shares(&members, 48 * 53);
        for (at, &times) in times.iter().enumerate() {
            let expected = if (20..36).contains(&at) { 0 } else { 53 };
            assert_eq!(times, expected, "worker {at} of 64, a run not ready");
        }
    }

    /// A worker over its overload thresholds takes a request only where the
    /// caller lifted that veto, and a worker that is not ready, doubted or
    /// behind an open breaker under neither word. A pool no worker of which
    /// passes names nobody: its caller decides such a pool.
    #[test]
    fn a_pick_tests_a_worker_as_the_worker_tests_itself() {
        let engine = preset("round_robin");
        let profile = engine.live_profile("round_robin").expect("picked alone");
        let list = pool(4);
        list[0].set_status(WorkerStatus::NotReady);
        list[1].set_overloaded(true);
        list[2].set_stall(Some(StallReason::Wedged));
        let members = engine.pool_members(&list);
        for _ in 0..8 {
            assert_eq!(engine.pick(profile, &members, false), Some(3));
        }
        let lifted: BTreeSet<usize> = (0..8)
            .filter_map(|_| engine.pick(profile, &members, true))
            .collect();
        assert_eq!(lifted, BTreeSet::from([1, 3]));

        list[3].set_overloaded(true);
        assert_eq!(engine.pick(profile, &members, false), None);
        assert_eq!(engine.pick_among(profile, &list, false), None);
        assert!(engine.pick(profile, &members, true).is_some());
        // A pool that holds no worker.
        let empty = engine.pool_members(&pool(0));
        assert_eq!(engine.pick(profile, &empty, false), None);
        assert_eq!(engine.pick_among(profile, &[], false), None);
    }

    /// A list the engine keeps nothing of is passed over once, and the
    /// workers that pass take turns by their set's own rotation: exact
    /// shares, named by their place in the list handed.
    #[test]
    fn a_pick_among_a_list_turns_the_rotation_of_the_workers_that_pass() {
        let engine = preset("round_robin");
        let profile = engine.live_profile("round_robin").expect("picked alone");
        let list = pool(9);
        // A part of the pool, as a caller narrows it: every third worker
        // left out, and one of the others not ready.
        let part: Vec<Arc<dyn Worker>> = (0..9)
            .filter(|at| at % 3 != 0)
            .map(|at| Arc::clone(&list[at]))
            .collect();
        part[2].set_status(WorkerStatus::NotReady);
        let mut times = [0usize; 6];
        for _ in 0..5 * 7 {
            times[engine.pick_among(profile, &part, false).expect("a worker")] += 1;
        }
        assert_eq!(times, [7, 7, 0, 7, 7, 7]);
    }
}
