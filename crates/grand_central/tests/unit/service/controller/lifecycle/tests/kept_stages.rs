//! What a served controller's stages keep across a document it applies,
//! which of them hear of a lease issued under an earlier one, and which
//! document a word of a lease is told by.
//!
//! A controller validates every document of its life over one stage table,
//! so a bed here validates its documents over one. Every case goes through
//! the store's own apply and the service's frames. A stage numbers each
//! object its factory builds and keeps every word it hears under that
//! number, so a case reads which object heard a word: the one a document
//! kept, or one it built new.

use std::{
    mem,
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    super::{completed, dispatched, released, told},
    a_completed, a_dispatched, a_released, counted, issued, issued_on, keyed, open, worker, A, KEY,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Pair, Target, Targets},
        framework::{Picker, TieBand},
        sticky_sessions::{Claim, SessionVerdict},
        view::{Candidate, RequestFacts, Session, WorkerId},
    },
    policies::random::Random,
    service::proto::{self, released::Reason},
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        memory::seam, DecodeSide, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord,
        Position, StoreSnapshot,
    },
    tracking::lease::{Dispatch, Lease, LeaseId, Membership, Outcome, ReleaseReason, Said, Side},
};

/// The filter a document gives the profile `moved` to build it anew.
const FILTERED: &str = r#""healthy""#;

/// One word a stage heard.
#[derive(Clone, Debug, PartialEq)]
enum Word {
    /// The list of the members that stand.
    Members(Vec<WorkerId>),
    /// One member's join.
    Joined(WorkerId),
    /// A side's dispatch: the lease and the worker the side went to.
    Dispatched(u64, WorkerId),
    /// A side's completion.
    Completed(u64),
    /// A side's release.
    Released(u64, Side),
}

impl Word {
    /// Whether the word is of a lease, and of no member.
    fn of_a_lease(&self) -> bool {
        !matches!(self, Self::Members(_) | Self::Joined(_))
    }
}

/// What the objects built under one picker's name heard, each word under
/// the number of the object that heard it.
type Log = Arc<Mutex<Vec<(u32, Word)>>>;

/// The `random` picker under a name of its own, keeping every word it
/// hears under its object's number.
#[derive(Debug)]
struct Notes {
    name: &'static str,
    made: u32,
    log: Log,
}

impl Notes {
    fn note(&self, word: Word) {
        self.log.lock().push((self.made, word));
    }
}

impl Lifecycle for Notes {
    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        self.note(Word::Dispatched(lease.id.0, target.id));
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        self.note(Word::Completed(lease.id.0));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.note(Word::Released(lease.id.0, lease.side));
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            Membership::Snapshot(members) => self.note(Word::Members(members.to_vec())),
            Membership::Added(joined) => self.note(Word::Joined(*joined)),
            _ => {}
        }
    }
}

impl Picker for Notes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A store, the one stage table its documents are validated over, and what
/// the stages of its documents' two profiles heard.
struct Bed {
    store: Arc<MemoryStore>,
    catalog: StageRegistry,
    /// What the objects of the profile `kept` heard.
    kept: Log,
    /// What the objects of the profile `moved` heard.
    moved: Log,
    now: Instant,
}

impl Bed {
    fn new() -> Self {
        let mut catalog = StageRegistry::builtin();
        let (kept, moved) = (Log::default(), Log::default());
        for (name, log) in [("keeper", &kept), ("mover", &moved)] {
            let (log, built) = (Arc::clone(log), AtomicU32::new(0));
            let build = move || -> Box<dyn Picker> {
                let made = built.fetch_add(1, Ordering::Relaxed);
                let log = Arc::clone(&log);
                Box::new(Notes { name, made, log })
            };
            catalog.picker(name, build).expect("a free name");
        }
        Self {
            store: Arc::new(MemoryStore::new()),
            catalog,
            kept,
            moved,
            now: Instant::now(),
        }
    }

    /// `generation` of the bed's document, validated over the bed's table:
    /// the profile `kept` for every model and the profile `moved` for the
    /// model `m`, each with a picker that keeps what it hears. `filters`
    /// are the filters of `moved`, and `view` the document's view block: a
    /// document that states other filters builds `moved` anew, and one
    /// that states another view builds both.
    fn spec_under(&self, generation: u64, filters: &str, view: &str) -> AppliedSpec {
        let text = format!(
            r#"{{
              "apiVersion": "smg.policy/v1",
              "view": {{ {view} }},
              "profiles": {{
                "kept": {{ "pick": "keeper" }},
                "moved": {{ "filter": [{filters}], "pick": "mover" }}
              }},
              "bindings": [
                {{ "match": {{ "model": "m" }}, "profile": "moved" }},
                {{ "match": {{ "model": "*" }}, "profile": "kept" }}
              ]
            }}"#
        );
        let mut document: PolicySpec = serde_json::from_str(&text).expect("a document");
        document.generation = generation;
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &self.catalog, &[])
            .expect("the table holds both stages")
    }

    /// [`Self::spec_under`] with the view every document states by default.
    fn spec(&self, generation: u64, filters: &str) -> AppliedSpec {
        self.spec_under(generation, filters, "")
    }

    /// Applies generation 1, where `moved` has no filter, then generation
    /// 2, which states `kept` as it stood and gives `moved` a filter.
    /// `between` runs behind the first.
    fn applied_twice(&self, between: impl FnOnce(&Self)) {
        self.store.apply(self.spec(1, "")).expect("the first");
        between(self);
        let next = self.spec(2, FILTERED);
        self.store.apply(next).expect("a later generation");
    }

    /// A pair's lease `id` as `generation` issued it: its prefill on `p`
    /// and its decode side on `d`.
    fn pair(&self, id: u64, generation: u64, (p, d): (WorkerId, WorkerId)) {
        let pair = Pair {
            decode_id: d,
            ..Pair::default()
        };
        let mut targets = Targets::new();
        targets.push(Target {
            id: p,
            pair: Some(pair),
            ..Target::default()
        });
        let side = DecodeSide {
            target: d,
            picked: PickRecord::of_decodes(&targets),
            session: Session::NONE,
            session_verdict: SessionVerdict::None,
            state: LeaseState::ISSUED,
            tokens: 0,
            target_index: 0,
        };
        let lease = Lease {
            id: LeaseId(id),
            side: Side::Prefill,
            generation,
            ..Lease::default()
        };
        let record = LeaseRecord {
            lease,
            gateway: A,
            request: id,
            target: p,
            picked: PickRecord::of(&targets),
            decode: None,
            untold: false,
            term: 0,
            issued_at: self.now,
            dispatch_deadline: self.now + Duration::from_secs(5),
            state: LeaseState::ISSUED,
        };
        LeasesStore::issue_pair(&*self.store, record, side).expect("the key is free");
    }

    /// A `Dispatched` frame of the lease `id` on `side`, to its first
    /// target.
    fn dispatched(&self, id: u64, side: proto::Side) {
        let frame = proto::Dispatched {
            side: side.into(),
            ..a_dispatched(id)
        };
        dispatched(&self.store, &open(&self.store, A), &frame, self.now).expect("the frame reads");
    }

    /// A `Completed` frame of the lease `id` on `side`, served by `served`.
    fn completed(&self, id: u64, side: proto::Side, served: WorkerId) {
        let frame = proto::Completed {
            side: side.into(),
            ..a_completed(id, served)
        };
        completed(&self.store, &open(&self.store, A), &frame).expect("the frame reads");
    }

    /// A `Released` frame of the lease `id` that names no side.
    fn released(&self, id: u64) {
        released(&self.store, &open(&self.store, A), &a_released(id)).expect("the frame reads");
    }
}

/// The numbers of the objects that heard a word, each once, in the order
/// they first heard one.
fn objects(log: &Log) -> Vec<u32> {
    let mut made = Vec::new();
    for (object, _) in log.lock().iter() {
        if !made.contains(object) {
            made.push(*object);
        }
    }
    made
}

/// Every word the object `made` heard.
fn heard(log: &Log, made: u32) -> Vec<Word> {
    let words = log.lock();
    let own = words.iter().filter(|(object, _)| *object == made);
    own.map(|(_, word)| word.clone()).collect()
}

/// The words of a lease the object `made` heard.
fn lease_words(log: &Log, made: u32) -> Vec<Word> {
    let mut words = heard(log, made);
    words.retain(Word::of_a_lease);
    words
}

/// The object of the profile `kept`, and of `moved` the one a second
/// document replaced and the one it built new.
fn kept_replaced_built(bed: &Bed) -> (u32, u32, u32) {
    let (kept, moved) = (objects(&bed.kept), objects(&bed.moved));
    assert_eq!((kept.len(), moved.len()), (1, 2), "{kept:?} and {moved:?}");
    (kept[0], moved[0], moved[1])
}

/// An apply keeps the stage object of a profile that did not change, with
/// what it heard, and tells it no second list of the members: it heard
/// each as it came. A profile that changed is a new object, told the
/// members that stand once. A join behind the apply is told to the one
/// object each profile has then.
#[test]
fn an_apply_keeps_an_unchanged_profiles_stage_and_lists_the_members_to_a_stage_built_new() {
    let bed = Bed::new();
    let first = worker(&bed.store, "http://a:8000");
    let mut both = Vec::new();
    bed.applied_twice(|bed| both = vec![first, worker(&bed.store, "http://b:8000")]);
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let before = [Word::Members(vec![first]), Word::Joined(both[1])];
    assert_eq!(heard(&bed.kept, kept), before);
    assert_eq!(heard(&bed.moved, replaced), before);
    both.sort_unstable();
    assert_eq!(heard(&bed.moved, built), [Word::Members(both)]);
    let third = worker(&bed.store, "http://c:8000");
    assert_eq!(heard(&bed.kept, kept)[2..], [Word::Joined(third)]);
    assert_eq!(heard(&bed.moved, built)[1..], [Word::Joined(third)]);
    assert_eq!(heard(&bed.moved, replaced), before, "it hears no more");
}

/// A single's lease issued under an earlier document is told to the stage
/// the standing document kept: its dispatch, its end and its release. A
/// stage built new hears none of them, and hears the lease its own
/// document issued.
#[test]
fn a_singles_lease_of_an_earlier_document_is_told_to_the_stage_kept_and_to_none_built_new() {
    let bed = Bed::new();
    let served = worker(&bed.store, "http://a:8000");
    bed.applied_twice(|bed| {
        for id in 1..=2 {
            issued(&bed.store, id, 1, bed.now);
        }
    });
    issued(&bed.store, 3, 2, bed.now);
    bed.dispatched(1, proto::Side::Single);
    bed.completed(1, proto::Side::Single, served);
    bed.released(2);
    bed.dispatched(3, proto::Side::Single);
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let words = [
        Word::Dispatched(1, WorkerId(7)),
        Word::Completed(1),
        Word::Released(2, Side::Single),
        Word::Dispatched(3, WorkerId(7)),
    ];
    assert_eq!(lease_words(&bed.kept, kept), words);
    assert_eq!(lease_words(&bed.moved, built), words[3..]);
    assert!(lease_words(&bed.moved, replaced).is_empty());
}

/// A keyed lease of an earlier document is told to the stage kept and to
/// the session table. The table is the store's engine's, one for every
/// document, so the claim the lease names was made in it: the claim stands
/// across the apply. The earlier lease's commit that refused every target
/// takes its claim back, and an earlier lease's dispatch on another worker
/// than it claimed pins the key where the request went, as in a gateway's
/// own process.
#[test]
fn a_keyed_lease_of_an_earlier_document_is_told_to_the_stage_kept_and_to_the_session_table() {
    let bed = Bed::new();
    let claimed = SessionVerdict::Claimed(WorkerId(7));
    let pins = || {
        bed.store
            .with_snapshot(|world| world.spec().engine().sessions().pins(KEY))
    };
    bed.applied_twice(|bed| {
        // The first lease's decision claims the key, under the earlier
        // document.
        let made = bed.store.with_snapshot(|world| {
            let table = world.spec().engine().sessions();
            table.claim(KEY, [WorkerId(0); 2], WorkerId(7), 0)
        });
        assert_eq!(made, Claim::Made(WorkerId(7)));
        keyed(&bed.store, 1, 7, claimed, bed.now);
        // The second lease's commit goes to another worker than it claimed.
        keyed(&bed.store, 2, 8, claimed, bed.now);
    });
    let claim = Some([WorkerId(7), WorkerId(0)]);
    assert_eq!(pins(), claim, "the claim stands across the apply");
    let refused = proto::Released {
        reason: Reason::CommitFailed.into(),
        ..a_released(1)
    };
    released(&bed.store, &open(&bed.store, A), &refused).expect("the frame reads");
    let none = Some([WorkerId(0), WorkerId(0)]);
    assert_eq!(pins(), none, "the refusal took the claim back");
    bed.dispatched(2, proto::Side::Single);
    let served = Some([WorkerId(8), WorkerId(0)]);
    assert_eq!(pins(), served, "the dispatch pinned the key where it went");
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let words = [
        Word::Released(1, Side::Single),
        Word::Dispatched(2, WorkerId(8)),
    ];
    assert_eq!(lease_words(&bed.kept, kept), words);
    assert!(lease_words(&bed.moved, built).is_empty());
    assert!(lease_words(&bed.moved, replaced).is_empty());
}

/// A pair's lease issued under an earlier document is told to the stage
/// kept a side at a time, as under its own document, and to no stage built
/// new.
#[test]
fn a_pairs_lease_of_an_earlier_document_is_told_by_side_to_the_stage_kept() {
    let bed = Bed::new();
    let sides = ["http://p:8000", "http://d:8000"].map(|key| worker(&bed.store, key));
    let (p, d) = (sides[0], sides[1]);
    bed.applied_twice(|bed| {
        for id in 1..=2 {
            bed.pair(id, 1, (p, d));
        }
    });
    bed.dispatched(1, proto::Side::Prefill);
    bed.dispatched(1, proto::Side::Decode);
    bed.completed(1, proto::Side::Prefill, p);
    // A release that names no side ends what stands: the first pair's
    // decode, and both sides of the second.
    bed.released(1);
    bed.released(2);
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let words = [
        Word::Dispatched(1, p),
        Word::Dispatched(1, d),
        Word::Completed(1),
        Word::Released(1, Side::Decode),
        Word::Released(2, Side::Prefill),
        Word::Released(2, Side::Decode),
    ];
    assert_eq!(lease_words(&bed.kept, kept), words);
    assert!(lease_words(&bed.moved, built).is_empty());
    assert!(lease_words(&bed.moved, replaced).is_empty());
}

/// A document that states another view builds every profile anew, as an
/// engine does that takes it itself: no stage is kept, and a lease of the
/// earlier document is told to no stage.
#[test]
fn a_document_of_another_view_keeps_no_stage() {
    let bed = Bed::new();
    bed.store.apply(bed.spec(1, "")).expect("the first");
    issued(&bed.store, 1, 1, bed.now);
    let other = bed.spec_under(2, "", r#""block_size": 32"#);
    bed.store.apply(other).expect("a later generation");
    bed.dispatched(1, proto::Side::Single);
    bed.released(1);
    for log in [&bed.kept, &bed.moved] {
        let made = objects(log);
        assert_eq!(made.len(), 2, "one object a document");
        assert_eq!(heard(log, made[1]), [Word::Members(Vec::new())]);
        assert!(lease_words(log, made[0]).is_empty());
    }
}

/// A standby keeps as a leader does: a generation it follows keeps the
/// stage of an unchanged profile, and that stage hears a lease of the
/// generation before.
#[test]
fn a_generation_a_standby_follows_keeps_as_an_apply_does() {
    let bed = Bed::new();
    bed.store.apply(bed.spec(1, "")).expect("the first");
    issued(&bed.store, 1, 1, bed.now);
    let next = bed.spec(2, FILTERED);
    let followed = bed.store.follow_generation(Position::default(), next);
    followed.expect("a later generation");
    bed.released(1);
    let (kept, replaced, built) = kept_replaced_built(&bed);
    assert_eq!(
        lease_words(&bed.kept, kept),
        [Word::Released(1, Side::Single)]
    );
    assert!(lease_words(&bed.moved, built).is_empty());
    assert!(lease_words(&bed.moved, replaced).is_empty());
}

/// A document the store refuses keeps nothing and changes nothing: one of
/// a number that is not above the standing one, and one its source no
/// longer admits. One that is both is answered as its source's word says,
/// the last one asked. The standing stages hear the next join, and a lease
/// of their document, as before.
#[test]
fn a_refused_document_leaves_the_standing_stages_as_they_were() {
    let bed = Bed::new();
    bed.store.apply(bed.spec(1, "")).expect("the first");
    issued(&bed.store, 1, 1, bed.now);
    let stale = bed.store.apply(bed.spec(1, FILTERED));
    assert!(
        stale.is_err(),
        "a number that is not above the standing one"
    );
    let admitted = bed.store.apply_admitted(bed.spec(2, FILTERED), || false);
    assert_eq!(admitted.ok(), Some(false));
    let both = bed.store.apply_admitted(bed.spec(1, FILTERED), || false);
    assert_eq!(both.ok(), Some(false), "its source has the last word");
    let joined = worker(&bed.store, "http://a:8000");
    bed.released(1);
    let told = [Word::Members(Vec::new()), Word::Joined(joined)];
    for log in [&bed.kept, &bed.moved] {
        let made = objects(log);
        assert_eq!(
            made.len(),
            1,
            "no object of a refused document heard a word"
        );
        assert_eq!(heard(log, made[0])[..2], told);
        assert_eq!(lease_words(log, made[0]), [Word::Released(1, Side::Single)]);
    }
}

/// Two applies at once keep in their order. The second waits for the lock
/// the first holds and keeps from the document the first put in place, not
/// from the one that stood when it was called: a profile the first changed
/// left with it. A document built ahead of that lock would keep the stage
/// that left, and the next join would be told to it.
#[test]
fn an_apply_that_waited_for_another_keeps_from_the_document_that_one_applied() {
    let bed = Bed::new();
    bed.store.apply(bed.spec(1, "")).expect("the first");
    // Generation 3 states `moved` as generation 1 did, and generation 2
    // states it another way between them.
    let (second, third) = (bed.spec(2, FILTERED), bed.spec(3, ""));
    let (holds, held) = mpsc::channel();
    let (go, gone) = mpsc::channel::<()>();
    let store = &bed.store;
    thread::scope(|scope| {
        scope.spawn(move || {
            // The last word of the second document's apply, asked under the
            // store's locks: it stays there while the third's starts.
            let admit = || {
                holds.send(()).expect("the case waits for it");
                gone.recv().is_ok()
            };
            let applied = store.apply_admitted(second, admit);
            assert_eq!(applied.ok(), Some(true));
        });
        held.recv().expect("the second apply holds the locks");
        let later = scope.spawn(move || store.apply(third));
        // The one wait a clock decides: the third document's apply reaches
        // the lock. The case holds whatever the wait.
        thread::sleep(Duration::from_millis(50));
        go.send(()).expect("the second apply waits for it");
        let applied = later.join().expect("the third apply's thread");
        applied.expect("a later generation");
    });
    let joined = worker(&bed.store, "http://a:8000");
    let told = [Word::Members(Vec::new()), Word::Joined(joined)];
    let (kept, moved) = (objects(&bed.kept), objects(&bed.moved));
    assert_eq!((kept.len(), moved.len()), (1, 3), "{kept:?} and {moved:?}");
    assert_eq!(heard(&bed.kept, kept[0]), told, "kept by both");
    assert_eq!(heard(&bed.moved, moved[2]), told, "built new at the third");
    assert_eq!(
        heard(&bed.moved, moved[0]),
        told[..1],
        "it left at the second"
    );
}

/// One single's dispatch, to its first target.
fn a_dispatch(at: Instant) -> Said {
    let dispatch = Dispatch {
        side: Side::Single,
        target_index: 0,
        since_decision_micros: 0,
    };
    Said::Dispatched(dispatch, at)
}

/// A word is told by the generation in force at the word, whatever world
/// its teller is handed. A run of words loads its world once, so a word of
/// that run behind an apply is handed the world of the generation before.
/// It is told by the generation that stands all the same: a lease issued
/// behind the apply reaches the stage that generation built new and the
/// stage it kept, and not the stage that left with the generation before.
#[test]
fn a_word_handed_a_world_of_the_generation_before_is_told_by_the_generation_in_force() {
    let bed = Bed::new();
    bed.store.apply(bed.spec(1, "")).expect("the first");
    let older = bed.store.snapshot();
    let next = bed.spec(2, FILTERED);
    bed.store.apply(next).expect("a later generation");
    issued(&bed.store, 1, 2, bed.now);
    let found = LeasesStore::sides(&*bed.store, A, LeaseId(1)).expect("a lease on file");
    told(
        &bed.store,
        &older,
        &found,
        a_dispatch(bed.now),
        None,
        |_, _| true,
    );
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let word = [Word::Dispatched(1, WorkerId(7))];
    assert_eq!(lease_words(&bed.kept, kept), word);
    assert_eq!(lease_words(&bed.moved, built), word);
    assert!(lease_words(&bed.moved, replaced).is_empty());
}

/// Inside an apply, between the engine's turn and the publication of the
/// generation's world, the generation in force is still the one the
/// standing world holds: no lease of the new one exists before its world
/// is published. A dispatch said there is told by the generation before,
/// and the lease's end, said behind the publication, by the new one. The
/// stage both generations hold hears each word once. The stage that leaves
/// hears the dispatch, and the stage built new hears neither word: the
/// lease is of the generation before its own.
#[test]
fn a_word_said_inside_an_apply_is_told_by_the_generation_the_standing_world_holds() {
    let bed = Bed::new();
    let served = worker(&bed.store, "http://a:8000");
    bed.store.apply(bed.spec(1, "")).expect("the first");
    issued(&bed.store, 1, 1, bed.now);
    let (store, at) = (Arc::clone(&bed.store), bed.now);
    seam::on_turn(move || {
        let turned = store.deciding(|world| !world.front().still_in_force());
        assert!(
            turned,
            "the engine holds the generation, the world does not"
        );
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
    });
    let next = bed.spec(2, FILTERED);
    bed.store.apply(next).expect("a later generation");
    bed.completed(1, proto::Side::Single, served);
    let (kept, replaced, built) = kept_replaced_built(&bed);
    let words = [Word::Dispatched(1, WorkerId(7)), Word::Completed(1)];
    assert_eq!(lease_words(&bed.kept, kept), words);
    assert_eq!(lease_words(&bed.moved, replaced), words[..1]);
    assert!(lease_words(&bed.moved, built).is_empty());
}

/// What a case reads of the stage objects one factory built: the thread
/// each was dropped on, and a gate the first of them to hear a dispatch
/// waits at.
#[derive(Debug, Default)]
struct Left {
    /// The thread each object was dropped on, under the object's number.
    dropped: Mutex<Vec<(u32, ThreadId)>>,
    /// Taken by the first dispatch a stage hears: the stage says its
    /// number from inside its call, then waits for the case's word.
    gate: Mutex<Option<(mpsc::Sender<u32>, mpsc::Receiver<()>)>>,
}

impl Left {
    /// The thread the object `made` was dropped on, where it was.
    fn dropped_on(&self, made: u32) -> Option<ThreadId> {
        let dropped = self.dropped.lock();
        let own = dropped.iter().find(|(object, _)| *object == made);
        own.map(|&(_, thread)| thread)
    }
}

/// The `random` picker under a number of its own, whose drop names its
/// thread and whose first dispatch waits at the case's gate.
#[derive(Debug)]
struct Leaves {
    made: u32,
    left: Arc<Left>,
}

impl Drop for Leaves {
    fn drop(&mut self) {
        let thread = thread::current().id();
        self.left.dropped.lock().push((self.made, thread));
    }
}

impl Lifecycle for Leaves {
    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        let gate = self.left.gate.lock().take();
        if let Some((inside, go)) = gate {
            inside.send(self.made).expect("the case waits for it");
            // A case that failed lets the gate go by dropping its end.
            let _gone = go.recv();
        }
    }
}

impl Picker for Leaves {
    fn name(&self) -> &'static str {
        "leaves"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A telling lets no stage go on its own thread, however many generations
/// start while it runs. The word here is handed a world of the generation
/// before, so its teller loads the standing world itself, and a stage of
/// that world's generation holds the telling inside its call while two
/// more generations start. The second of them hands back the generation
/// the word is told by, and the writer still keeps the world the telling
/// loaded: the telling's end frees nothing. The next publication finds no
/// reader on that world and lets the stage go, on the thread that writes.
#[test]
fn a_telling_that_outlasts_two_applies_lets_no_stage_go_on_its_own_thread() {
    let left = Arc::new(Left::default());
    let mut catalog = StageRegistry::builtin();
    let (built, kept) = (AtomicU32::new(0), Arc::clone(&left));
    let build = move || -> Box<dyn Picker> {
        let made = built.fetch_add(1, Ordering::Relaxed);
        let left = Arc::clone(&kept);
        Box::new(Leaves { made, left })
    };
    catalog.picker("leaves", build).expect("a free name");
    // Each generation states its one profile another way than the one
    // before, so each builds its stage anew.
    let spec = |generation: u64| {
        let filters = if generation.is_multiple_of(2) {
            FILTERED
        } else {
            ""
        };
        let text = format!(
            r#"{{
              "apiVersion": "smg.policy/v1",
              "profiles": {{ "only": {{ "filter": [{filters}], "pick": "leaves" }} }},
              "bindings": [ {{ "match": {{ "model": "*" }}, "profile": "only" }} ]
            }}"#
        );
        let mut document: PolicySpec = serde_json::from_str(&text).expect("a document");
        document.generation = generation;
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the table holds the stage")
    };
    let (store, now) = (MemoryStore::new(), Instant::now());
    store.apply(spec(1)).expect("the first");
    let older = store.snapshot();
    store.apply(spec(2)).expect("a later generation");
    issued(&store, 1, 2, now);
    let found = LeasesStore::sides(&store, A, LeaseId(1)).expect("a lease on file");
    let (inside, entered) = mpsc::channel();
    let (go, gone) = mpsc::channel();
    *left.gate.lock() = Some((inside, gone));
    let (third, fourth) = (spec(3), spec(4));
    // The stage the telling stood in, and the telling's thread.
    let (stage, telling) = thread::scope(|scope| {
        // Dropped where this closure unwinds: a stage at the gate goes on.
        let go = go;
        let telling = scope.spawn(|| {
            told(&store, &older, &found, a_dispatch(now), None, |_, _| true);
            thread::current().id()
        });
        let entered = entered.recv_timeout(Duration::from_secs(30));
        let stage = entered.expect("a stage of the standing generation hears the word");
        store.apply(third).expect("a later generation");
        store.apply(fourth).expect("a later generation");
        go.send(()).expect("the stage waits for it");
        (stage, telling.join().expect("the telling's thread"))
    });
    let freed = left.dropped_on(stage);
    assert_eq!(freed, None, "the telling's end freed no stage");
    store.apply(spec(5)).expect("a later generation");
    assert_eq!(left.dropped_on(stage), Some(thread::current().id()));
    let dropped = left.dropped.lock();
    let on_the_telling = dropped.iter().filter(|(_, thread)| *thread == telling);
    assert_eq!(on_the_telling.count(), 0, "{dropped:?}");
}

/// What a word of a lease costs its teller in each of its three states,
/// measured: handed a world that holds the generation in force, handed a
/// world of the generation before, and said inside an apply. The word is a
/// single's dispatch under the least-load preset, told with no file's turn
/// around it, and each span holds two reads of the clock. The first state
/// loads nothing. The other two load the store's standing world for the
/// word: what they cost more is that load.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn a_words_cost_in_its_three_states_measured() {
    use std::io::Write as _;

    const WORDS: u64 = 2_000;
    /// One dispatch of `lease` told under `world`, in nanoseconds.
    fn timed(store: &MemoryStore, world: &StoreSnapshot, lease: u64, at: Instant) -> u128 {
        let found = LeasesStore::sides(store, A, LeaseId(lease)).expect("a lease on file");
        let began = Instant::now();
        told(store, world, &found, a_dispatch(at), None, |_, _| true);
        began.elapsed().as_nanos()
    }
    let (store, now) = (Arc::new(MemoryStore::new()), Instant::now());
    let served = worker(&store, "http://a:8000");
    store.apply(counted(1)).expect("the first");
    for lease in 1..=3 * WORDS {
        issued_on(&store, lease, 1, served, now);
    }
    let current = store.snapshot();
    let in_force = (1..=WORDS).map(|lease| timed(&store, &current, lease, now));
    let in_force: Vec<u128> = in_force.collect();
    let window = Arc::new(Mutex::new(Vec::new()));
    let (inside, kept) = (Arc::clone(&store), Arc::clone(&window));
    seam::on_turn(move || {
        let standing = inside.snapshot();
        let leases = WORDS + 1..=2 * WORDS;
        let spans = leases.map(|lease| timed(&inside, &standing, lease, now));
        kept.lock().extend(spans);
    });
    store.apply(counted(2)).expect("a later generation");
    let behind = (2 * WORDS + 1..=3 * WORDS).map(|lease| timed(&store, &current, lease, now));
    let behind: Vec<u128> = behind.collect();
    let window = mem::take(&mut *window.lock());
    let states = [
        (
            "handed a world that holds the generation in force",
            in_force,
        ),
        ("handed a world of the generation before", behind),
        ("said inside an apply", window),
    ];
    let mut out = std::io::stdout();
    for (state, mut spans) in states {
        assert_eq!(spans.len() as u64, WORDS, "{state}");
        spans.sort_unstable();
        let (middle, ninth) = (spans[spans.len() / 2], spans[spans.len() * 9 / 10]);
        writeln!(
            out,
            "a word {state}, {WORDS} dispatches, ns (middle, ninth tenth): {middle} {ninth}"
        )
        .expect("stdout");
    }
}
