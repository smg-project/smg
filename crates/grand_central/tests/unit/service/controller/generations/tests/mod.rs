mod dry_run;
mod enforce;
mod follow;
mod lead;
mod overload_switch;
mod read;
mod rollback;
mod start;

use std::{
    cell::Cell,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use super::*;
pub(crate) use crate::document::operator::test_support::{
    bound_to, faulting, long_text_under_aliases, padded, with_line, LEAST_LOAD, RANDOM,
};
use crate::{
    document::operator::{
        test_support::{stock, Validate},
        ApplyCounts,
    },
    service::elector::{ElectionSettings, Elector, MemoryLease},
};

impl Generations {
    /// A door with a controller's pass that states nothing on a holder, for
    /// a case.
    pub(in crate::service::controller) fn new(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
    ) -> Self {
        Self::over(
            DocumentManager::validating(store, catalog, Vec::new(), one_switch),
            None,
        )
    }

    /// [`Generations::new`] with the validator as its argument, and with
    /// the fence its door is built under, as [`Generations::stating`]
    /// takes it: a case hands in a validator that faults, or one that
    /// stalls, and the fence of an election of its own where its replica
    /// stands under a shared lease.
    pub(in crate::service::controller) fn validating(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
        validate: Validate,
        fence: Option<Arc<Fence>>,
    ) -> Self {
        let door = DocumentManager::validating(store, catalog, Vec::new(), validate);
        Self::over(door, fence)
    }

    /// Whether the source writes now: always for a replica alone over its
    /// lease, and while the fence shows a term for one under a shared lease
    /// ([`Generations::stating`]). A case's question: the source takes its
    /// lead ahead of a read ([`Generations::lead`]).
    pub(in crate::service::controller) fn source_writes(&self) -> bool {
        self.lead().is_some()
    }
}

thread_local! {
    static REPLICA_NOTICES: Cell<Option<u32>> = const { Cell::new(None) };
}

pub(super) fn count_replica_notice(door: &Generations, holds_document: bool) {
    REPLICA_NOTICES.with(|notices| {
        let Some(count) = notices.get() else {
            return;
        };
        assert!(door.applying_lock().try_lock().is_none());
        assert_eq!(door.held_lock().try_lock().is_none(), holds_document);
        assert_eq!(door.generation(), u64::from(holds_document));
        notices.set(Some(count + 1));
    });
}

fn generations() -> (Arc<MemoryStore>, Generations) {
    let store = Arc::new(MemoryStore::new());
    (Arc::clone(&store), Generations::new(store, stock()))
}

/// A store and its door built under `fence`, as the door of a replica
/// under a shared lease is.
fn generations_under(fence: Arc<Fence>) -> (Arc<MemoryStore>, Generations) {
    let store = Arc::new(MemoryStore::new());
    let door = Generations::validating(Arc::clone(&store), stock(), one_switch, Some(fence));
    (store, door)
}

/// `document` applied as YAML.
fn yaml(
    generations: &Generations,
    document: impl Into<Vec<u8>>,
) -> Result<ApplyOutcome, SpecRejected> {
    generations.apply(document.into(), Format::Yaml)
}

/// An election no case outlasts: a replica's candidate alone over its
/// lease, with an hour's renew deadline. Its fence shows a term from
/// its first step to its release.
pub(crate) fn election() -> Elector {
    election_under(Duration::from_secs(3_600))
}

/// A replica's candidate alone over its lease, with `deadline` as its
/// renew deadline. Under a short one a case closes the fence by the
/// clock, between two renewals of one term.
pub(crate) fn election_under(deadline: Duration) -> Elector {
    let settings = ElectionSettings {
        duration: 2 * deadline,
        renew_period: Duration::from_secs(1).min(deadline / 4),
        renew_deadline: deadline,
    };
    Elector::alone("controller-0@127.0.0.1:1", settings).expect("the settings are in order")
}

/// A candidate over a lease that is handed in, with `deadline` as its
/// renew deadline: its take has no holder to wait for, so its lead
/// opens one renew deadline behind the take.
pub(crate) fn candidate_under(deadline: Duration) -> Elector {
    let settings = ElectionSettings {
        duration: 2 * deadline,
        renew_period: Duration::from_secs(1).min(deadline / 4),
        renew_deadline: deadline,
    };
    Elector::new(
        Box::new(MemoryLease::new()),
        "controller-0@127.0.0.1:1",
        settings,
    )
    .expect("the settings are in order")
}

/// A validator's gate: once armed, the next pass to enter stands at it
/// until the case releases it, and the gate counts the passes that
/// entered since.
pub(crate) struct Gate {
    entered: AtomicU64,
    armed: AtomicBool,
    released: AtomicBool,
}

impl Gate {
    pub(crate) const fn new() -> Self {
        Self {
            entered: AtomicU64::new(0),
            armed: AtomicBool::new(false),
            released: AtomicBool::new(false),
        }
    }

    pub(crate) fn arm(&self) {
        self.entered.store(0, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }

    pub(crate) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
    }

    pub(crate) fn entered(&self) -> u64 {
        self.entered.load(Ordering::SeqCst)
    }

    /// The passes that entered a pace after the first did: the first
    /// stands at the gate, and one beside it had the pace to enter.
    pub(crate) async fn entered_a_pace_on(&self) -> u64 {
        let started = Instant::now();
        while self.entered() == 0 {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "no pass reached the validator"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        self.entered()
    }
}

/// A case's own gate. A validator is a plain function, so a case names
/// its gate by a type of its own ([`gate`]).
pub(crate) trait Gated {
    fn gate() -> &'static Gate;
}

/// Declares `$name`, a type whose gate is the case's own.
macro_rules! gate {
    ($name:ident) => {
        struct $name;
        impl Gated for $name {
            fn gate() -> &'static Gate {
                static GATE: Gate = Gate::new();
                &GATE
            }
        }
    };
}
pub(crate) use gate;

/// The stock validator behind `G`'s gate: a pass that finds the gate
/// armed stands there until it is released, ten seconds at the most.
fn stalling<G: Gated>(
    document: &PolicySpec,
    digest: Digest,
    catalog: &StageRegistry,
    denied_headers: &[&str],
) -> Result<AppliedSpec, SpecError> {
    let gate = G::gate();
    gate.entered.fetch_add(1, Ordering::SeqCst);
    if gate.armed.swap(false, Ordering::SeqCst) {
        for _ in 0..10_000 {
            if gate.released.load(Ordering::SeqCst) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
    AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
}

/// A door over a store of its own whose validator stands at `G`'s gate.
pub(crate) fn gated<G: Gated>() -> Arc<Generations> {
    let store = Arc::new(MemoryStore::new());
    Arc::new(Generations::validating(store, stock(), stalling::<G>, None))
}

/// Binds the directory at `dir` as the door's source, as a start does.
fn bind(generations: &Generations, dir: &Path) {
    let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
    generations.bind_store(opened).expect("the store binds");
}

/// A door with a spec store bound over a new, empty directory.
fn stored() -> (tempfile::TempDir, Arc<MemoryStore>, Arc<Generations>) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    (dir, store, Arc::new(generations))
}

/// What an enforce of `document` with no guard, as YAML, ran.
fn enforce(
    generations: &Generations,
    document: impl Into<Vec<u8>>,
) -> Result<(u64, Digest), SpecRejected> {
    generations
        .enforce("", document.into(), Format::Yaml)
        .expect("no guard")
        .ran
}

fn applied(applied: u64, rejected: u64) -> ApplyCounts {
    ApplyCounts { applied, rejected }
}

/// What a rollback to generation `wanted`, one on file, ran.
fn roll_back(generations: &Generations, wanted: u64) -> Result<(u64, Digest), SpecRejected> {
    generations.roll_back(wanted).expect("on file").ran
}

/// What a leader's door sends of its standing generation.
fn sent(leader: &Generations) -> proto::GenerationEvent {
    leader.standing_sent().expect("a held document")
}

/// `told` followed at no place, as a restore lands one.
fn followed(standby: &Generations, told: &proto::GenerationEvent) -> Result<(), Unfollowed> {
    standby.following(told)?.land(None)
}

/// The words a controller refuses bindings that differ on the overload
/// switch in: here the second binding says on, by the block's default,
/// where the first says off.
const DIFFERS: &str = "bindings[1].overload.enabled = true is not the false bindings[0] says, by its block or the block's default: a controller builds its workers with one default set of overload thresholds, so the bindings of its document say one word";

/// A document of two bindings over one profile that picks by `pick`,
/// the first for the chat models and the second for the rest, each
/// with the word of its `overload` block or with no block.
fn switched_by(pick: &str, first: Option<bool>, second: Option<bool>) -> String {
    let block = |word: Option<bool>| {
        word.map_or_else(String::new, |enabled| {
            format!("    overload:\n      enabled: {enabled}\n")
        })
    };
    let head = format!("apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: {pick}\nbindings:\n");
    let (first, second) = (block(first), block(second));
    format!(
        "{head}  - match:\n      model: chat/*\n    profile: p\n{first}  - profile: p\n{second}"
    )
}

/// [`switched_by`] the random pick.
pub(crate) fn switched(first: Option<bool>, second: Option<bool>) -> String {
    switched_by("random", first, second)
}
