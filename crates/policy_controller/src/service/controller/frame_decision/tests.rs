//! The cases of the frame decision: rules a decision holds on the served
//! path. A single request's ask [`decide`] as a deciding thread asks it, in
//! the world of one load of a store, over the worker objects the store's
//! holder lays. One asks the engine's entry that [`decide`] asks, over the
//! same objects, for a request whose host states what the workers hold of
//! its prompt, and one asks which documents are given the load snapshot.
//! A pair's ask [`decide_pair`] for a pair at its prefill, and the two
//! that file its leases ask [`issue_pair`] behind it. A decode call's ask
//! [`decide_decode`] behind the pair its lease holds, and the two that
//! issue its side ask [`issue_decode`] behind it. One asks
//! [`decide_alone`] for a prefill and for a decode frame that stands
//! alone.

use std::{
    collections::BTreeSet,
    mem,
    panic::resume_unwind,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::{one_rank_load, LoadSnapshot};

use super::{
    decide, decide_alone, decide_decode, decide_pair, hands_reports, issue_decode, issue_pair,
    FrameBuffers,
};
use crate::{
    document::spec::{AppliedSpec, Assignment, PolicySpec},
    engine::{
        decision::{Decision, ShedReason, Target, Targets, SUBSET_HINT, WIRES},
        decision_engine::{Request, RequestMaker, WorkerLists},
        decision_slot::DecisionSlot,
        framework::{DecisionScope, Filter, Picker, TieBand},
        view::{
            AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Phase, PoolFacts,
            RequestFacts, Role, Session, WireModes, WorkerId,
        },
    },
    policies::random::Random,
    service::{
        controller::{
            decision_slots::{
                tests::{
                    a_frame_under_the_second_files_watch, fleet, frame_for, holder, preset, ready,
                    taken_at, BUDGET, GATEWAY, MODEL,
                },
                Part,
            },
            frame_request::FrameRequest,
            lifecycle,
            wire::{Checked, Lists},
        },
        proto,
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        pairing::PairingDescriptor, AtWork, DecodeIssue, DecodeMark, GatewayId, Hand, HealthSink,
        LeaseRecord, LeaseState, LeasesStore, LookOutcome, MemoryStore, PickRecord, Reachability,
        WorkerFacts, WorkerStatus, WorkersStore,
    },
    tracking::lease::{Dispatch, Lease, LeaseId, Outcome, ReleaseReason, Side, DISPATCH_TTL},
    worker_host::{Members, Objects},
};

/// A gateway that is not the cases' own.
const OTHER: GatewayId = GatewayId(12);

/// The model of the cases that file a fleet's leases themselves.
const BOOKED: &str = "model-http";

/// The decision of `frame` as `gateway`'s, received and taken at the two
/// instants, with the decider's `place` at the files of leases where the
/// case lends one: what a deciding thread asks for a single request. The
/// holder first lays the objects of the rows written since the last
/// decision, as a decider does under a case: no task follows the store
/// here.
fn served(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    taken: (Instant, Instant),
    place: Option<&AtWork>,
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, taken, place);
    objects.follow(store);
    let mut frames = FrameBuffers::default();
    let own = Part::default();
    // The load monitor published nothing: the request reads no report.
    let loads = Arc::new(LoadSnapshot::default());
    let decided = store.deciding(|world| decide(world, objects, &taken, &mut frames, &loads, &own));
    decided.expect("the engine decides").0
}

/// The `i`th ready member of [`BOOKED`] in `store`, on the HTTP wire.
fn member(store: &MemoryStore, i: usize) -> WorkerId {
    let name = format!("h{i}");
    ready(store, &name, BOOKED, Role::Single, WireModes::HTTP)
}

/// The worker a decision names first.
fn first(decision: &Decision) -> Option<WorkerId> {
    let pick = decision.pick()?;
    pick.targets.first().map(|target| target.id)
}

/// `document` as its store applies it, with the stages of `catalog`.
fn staged(document: &PolicySpec, catalog: &StageRegistry) -> AppliedSpec {
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(document, digest, catalog, &[])
        .expect("the catalog holds every stage the document names")
}

/// `frame` with a session key that hashes to `key`.
fn keyed(mut frame: proto::DecideRequest, key: u64) -> proto::DecideRequest {
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(key)),
    });
    frame
}

/// Files one issued lease of [`GATEWAY`] on `target`, under `id`.
fn file(store: &MemoryStore, id: u64, target: WorkerId, now: Instant) {
    file_of(store, id, target, Session::NONE, now);
}

/// [`file`] for a lease of `session`, which counts for its key.
fn file_of(store: &MemoryStore, id: u64, target: WorkerId, session: Session, now: Instant) {
    let leases: &dyn LeasesStore = store;
    let filed = leases.issue(LeaseRecord {
        lease: Lease {
            id: LeaseId(1 << 63 | id),
            session,
            ..Lease::default()
        },
        gateway: GATEWAY,
        request: id,
        target,
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: now + DISPATCH_TTL,
        state: LeaseState::ISSUED,
    });
    assert_eq!(filed, Ok(()));
}

/// A store under `spec` with one ready member of [`BOOKED`] a count of
/// `held`, each booked by its count of leases on file, and the holder of
/// the store's worker objects: the store, the holder, the members in the
/// counts' order and the instant the leases were issued at.
fn booked(
    spec: AppliedSpec,
    held: &[u64],
) -> (
    Arc<MemoryStore>,
    Arc<Objects<Members>>,
    Vec<WorkerId>,
    Instant,
) {
    let store = Arc::new(MemoryStore::new());
    store.apply(spec).expect("above the unbound start");
    let members: Vec<WorkerId> = (0..held.len()).map(|i| member(&store, i)).collect();
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let targets = members.iter().zip(held);
    let targets = targets.flat_map(|(&member, &held)| (0..held).map(move |_| member));
    for (target, id) in targets.zip(1..) {
        file(&store, id, target, now);
    }
    let objects = holder(&store);
    (store, objects, members, now)
}

/// `held` by the worker of `members` each count is of, by worker id.
fn by_id(members: &[WorkerId], held: &[u32]) -> Counts {
    let mut held: Counts = members.iter().copied().zip(held.to_vec()).collect();
    held.sort_unstable();
    held
}

/// A decision whose stages refuse by no report reads no place, though its
/// decider lent one: a frame left under another file's watch stays there
/// for that file's thread, and the decision takes no turn at it.
#[test]
fn a_served_decision_whose_stages_refuse_by_no_report_applies_nothing_at_the_other_files() {
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    store
        .apply(preset("random", 1))
        .expect("above the unbound start");
    let member = member(&store, 0);
    LeasesStore::set_term(&*store, 1);
    let mut watch = a_frame_under_the_second_files_watch(&store, member);
    // The first file's decision, its place lent, under the preset's stages:
    // each says it refuses by no report.
    let _acting = DecisionSlot::new(0).enter();
    let place = store.at_work();
    let objects = holder(&store);
    let now = Instant::now();
    let frame = frame_for(BOOKED, 1, 1);
    let decision = served(&store, &objects, frame, GATEWAY, (now, now), Some(&place));
    assert_eq!(first(&decision), Some(member), "its one worker is picked");
    assert_eq!(
        watch.look(),
        LookOutcome::Fed,
        "the frame waited for its file's look"
    );
}

/// A key under a sessions rule with an in-flight cap keeps an owner over
/// its thresholds in a decision this service makes, as under a host that
/// counts its requests. The leases this service holds count the key's
/// requests on each worker, and the request its maker makes says so: the
/// cap can hand the key on, so the veto is lifted for the key. With the
/// cap's two requests of the key on file on its owner, the key goes to
/// another worker.
#[test]
fn a_served_key_under_a_capped_rule_keeps_an_owner_over_its_thresholds_below_its_cap() {
    let mut document = PolicySpec::preset("consistent_hashing").expect("a preset of this build");
    let manual = PolicySpec::preset("manual").expect("a preset of this build");
    let rule = manual.bindings.first();
    let rule = rule.and_then(|binding| binding.sessions.clone());
    let mut rule = rule.expect("the manual preset's rule");
    rule.assignment = Assignment::Delegate;
    rule.inflight_cap = 2;
    document.generation = 1;
    document.view.headers = manual.view.headers.clone();
    let binding = document.bindings.first_mut().expect("the preset's binding");
    // The rule names the key's header, so the ring's own sources go: a
    // binding names them once.
    binding.routing_key = None;
    binding.sessions = Some(rule);
    let store = Arc::new(MemoryStore::with_publish_window(Duration::ZERO));
    let (singles, _) = fleet(&store);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("above the unbound start");
    let objects = holder(&store);
    let now = Instant::now();
    // The worker a request of the one key is sent to.
    let decided = |id: u8| {
        let frame = keyed(frame_for(MODEL, id, u64::from(id)), 17);
        first(&served(&store, &objects, frame, GATEWAY, (now, now), None))
    };
    let owner = decided(1).expect("a pick over a free pool");
    assert_eq!(decided(2), Some(owner), "the key has one owner");
    let named = singles.iter().position(|&single| single == owner);
    let named = named.expect("a worker of the fleet's singles");
    // The owner's object reads over its thresholds, as a load monitor's
    // verdict leaves it.
    let url = format!("http://w{named}:8000");
    let object = objects.registry().get_by_url(&url).expect("its object");
    object.set_overloaded(true);
    assert!(object.is_overloaded(), "the owner reads overloaded");
    assert_eq!(
        decided(3),
        Some(owner),
        "the cap's count stands, so the key keeps its owner"
    );
    // The cap's two requests of the key on file on its owner: the frame's
    // session is the number its key hashes to.
    LeasesStore::set_term(&*store, 1);
    for id in 1..=2 {
        file_of(&store, id, owner, Session::from_raw(17), now);
    }
    let next = decided(4).expect("a pick beside two workers under their thresholds");
    assert_ne!(next, owner, "the key's owner is at its cap");
}

/// The requests in flight a decision read of each candidate, by worker.
type Counts = Vec<(WorkerId, u32)>;

/// A filter that keeps every worker. As a decision opens it reads each
/// candidate's requests in flight, files the case's leases, and reads the
/// candidates again.
#[derive(Debug)]
struct Late {
    store: Weak<MemoryStore>,
    /// The workers a lease is filed on at the next decision's opening.
    late: Arc<Mutex<Vec<WorkerId>>>,
    /// What each decision read: at its opening, and behind the filing.
    read: Arc<Mutex<Vec<[Counts; 2]>>>,
}

impl Lifecycle for Late {}

impl DecisionScope for Late {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], _: &PoolFacts<'_>) {
        let read = || -> Counts {
            let counts = candidates.iter();
            let mut counts: Counts = counts.map(|row| (row.id, facts.in_flight(row))).collect();
            counts.sort_unstable();
            counts
        };
        let opening = read();
        let store = self.store.upgrade().expect("the case's store");
        let late = mem::take(&mut *self.late.lock());
        let now = Instant::now();
        for (target, id) in late.into_iter().zip(1_000..) {
            file(&store, id, target, now);
        }
        self.read.lock().push([opening, read()]);
    }

    fn close(&self) {}
}

impl Filter for Late {
    fn name(&self) -> &'static str {
        "late"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

/// A decision reads one count a worker: the count its records took is
/// kept, and a lease filed after it moves nothing the decision's stages
/// read. Two leases are filed on the worker that held the fewest, after a
/// stage's first reads and before the `count-first` picker's pick: the
/// picker still names that worker. The next decision reads the two.
#[test]
fn a_served_lease_filed_after_a_records_first_read_moves_nothing_the_decision_reads() {
    let store = Arc::new(MemoryStore::new());
    let late = Arc::new(Mutex::new(Vec::new()));
    let read = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let (kept, filed, own) = (Arc::clone(&read), Arc::clone(&late), Arc::downgrade(&store));
    let stage = move || -> Box<dyn Filter> {
        Box::new(Late {
            store: Weak::clone(&own),
            late: Arc::clone(&filed),
            read: Arc::clone(&kept),
        })
    };
    catalog.filter("late", stage).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    let profile = profile.expect("the preset's profile");
    profile.pick = "count-first".into();
    profile.filter.push("late".into());
    store
        .apply(staged(&document, &catalog))
        .expect("above the unbound start");
    let members: Vec<WorkerId> = (0..3).map(|i| member(&store, i)).collect();
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    for (target, id) in [members[0], members[2], members[2]].into_iter().zip(1..) {
        file(&store, id, target, now);
    }
    let objects = holder(&store);
    let fewest = members[1];
    *late.lock() = vec![fewest, fewest];
    let frame = frame_for(BOOKED, 1, 1);
    let decision = served(&store, &objects, frame, GATEWAY, (now, now), None);
    let before = by_id(&members, &[1, 0, 2]);
    assert_eq!(
        read.lock()[0],
        [before.clone(), before],
        "a decision reads one count a worker"
    );
    assert_eq!(first(&decision), Some(fewest), "the picker read it too");
    let frame = frame_for(BOOKED, 2, 2);
    let next = served(&store, &objects, frame, GATEWAY, (now, now), None);
    assert_eq!(read.lock()[1][0], by_id(&members, &[1, 2, 2]));
    assert_eq!(first(&next), Some(members[0]), "the next reads the two");
}

/// The engine's own placement by the fewest requests reads the leases on
/// file as a stage does, with nothing declared: a profile whose picker
/// leaves the placement to its rule, under no rule and under a rule that
/// delegates, and a sessions rule that assigns `min_load`. Each places a
/// request on the worker that holds the fewest leases, whatever the
/// request's seed.
#[test]
fn a_served_placement_by_the_fewest_requests_reads_the_leases_on_file() {
    let ruled = |assignment: Assignment, pick: &str| {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").assignment = assignment;
        let profile = document.profiles.get_mut("manual");
        profile.expect("the preset's profile").pick = pick.into();
        document
    };
    let unruled = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"manual"}}}"#;
    let mut unruled: PolicySpec = serde_json::from_str(unruled).expect("a document");
    unruled.generation = 1;
    let (delegate, min_load) = (Assignment::Delegate, Assignment::MinLoad);
    let cases = [
        ("no rule", unruled, false),
        ("a rule that delegates", ruled(delegate, "manual"), false),
        (
            "a rule that delegates, a key",
            ruled(delegate, "manual"),
            true,
        ),
        ("min_load", ruled(min_load, "random"), true),
        (
            "min_load, the placement its rule's",
            ruled(min_load, "manual"),
            true,
        ),
    ];
    for (name, document, has_key) in cases {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let (store, objects, members, now) = booked(spec, &[2, 0, 3]);
        for seed in 1..=16u8 {
            let mut frame = frame_for(BOOKED, seed, u64::from(seed));
            if has_key {
                frame = keyed(frame, u64::from(seed));
            }
            let decision = served(&store, &objects, frame, GATEWAY, (now, now), None);
            assert_eq!(first(&decision), Some(members[1]), "{name}, seed {seed}");
        }
    }
}

/// What the engine is handed for one request, as a stage reads it: the
/// decision's instant and what is left of the budget, the candidates, and
/// those of them its filters kept.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Handed {
    now_micros: u64,
    budget_micros: u32,
    candidates: Vec<WorkerId>,
    kept: Vec<WorkerId>,
}

/// The `random` picker, keeping what each decision is handed.
#[derive(Debug)]
struct Hands(Arc<Mutex<Vec<Handed>>>);

impl Lifecycle for Hands {}

impl Picker for Hands {
    fn name(&self) -> &'static str {
        "hands"
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
        let mut all: Vec<WorkerId> = candidates.iter().map(|row| row.id).collect();
        let passed = kept.iter().filter_map(|&at| candidates.get(at as usize));
        let mut passed: Vec<WorkerId> = passed.map(|row| row.id).collect();
        all.sort_unstable();
        passed.sort_unstable();
        self.0.lock().push(Handed {
            now_micros: facts.now_micros,
            budget_micros: facts.budget_micros,
            candidates: all,
            kept: passed,
        });
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The engine is handed the asking gateway's vetoes as they stand at the
/// decider's instant, that instant on its own clock, and what is left of
/// the request's budget. A veto is its gateway's alone, and it stands at
/// the decision's instant or it holds no worker out. A model no member
/// serves is decided over no worker: no stage is asked.
#[test]
fn a_served_decision_is_handed_its_gateways_vetoes_its_instant_and_the_budget_left() {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let kept = Arc::clone(&heard);
    let stage = move || -> Box<dyn Picker> { Box::new(Hands(Arc::clone(&kept))) };
    catalog.picker("hands", stage).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hands".into();
    let store = Arc::new(MemoryStore::new());
    let (mut singles, _) = fleet(&store);
    singles.sort_unstable();
    store
        .apply(staged(&document, &catalog))
        .expect("above the unbound start");
    let objects = holder(&store);
    let taken = Instant::now();
    let now = taken + Duration::from_secs(3);
    // Received so long before its taking that 1,234 microseconds of the
    // budget are left.
    let waited = BUDGET - Duration::from_micros(1_234);
    // One decision, and what its picker was handed where it was asked.
    let handed = |model: &str, gateway: GatewayId, at: Instant| {
        let frame = frame_for(model, 1, 1);
        let decision = served(&store, &objects, frame, gateway, (at - waited, at), None);
        let mut asked = mem::take(&mut *heard.lock());
        assert!(asked.len() <= 1, "a decision asks its picker once");
        (decision, asked.pop())
    };
    let micros = |at: Instant| store.deciding(|world| world.front().clock().micros(at));
    let whole = Handed {
        now_micros: micros(now),
        budget_micros: 1_234,
        candidates: singles.clone(),
        kept: singles.clone(),
    };
    assert_eq!(handed(MODEL, GATEWAY, now).1, Some(whole.clone()));

    let vetoed = singles[1];
    store.gateway_veto(GATEWAY, vetoed, Some(taken + Duration::from_secs(60)));
    let held_out = Handed {
        kept: vec![singles[0], singles[2]],
        ..whole.clone()
    };
    let standing = handed(MODEL, GATEWAY, now).1;
    assert_eq!(standing, Some(held_out), "a standing veto");
    let theirs = handed(MODEL, OTHER, now).1;
    assert_eq!(theirs, Some(whole.clone()), "the veto is its gateway's");
    let ended = taken + Duration::from_secs(61);
    let after = Handed {
        now_micros: micros(ended),
        ..whole
    };
    let past = handed(MODEL, GATEWAY, ended).1;
    assert_eq!(past, Some(after), "a veto that ended");

    let (nobody, asked) = handed("model-nobody-serves", GATEWAY, now);
    assert_eq!(first(&nobody), None, "no worker to name");
    assert_eq!(asked, None, "decided over no worker");
}

/// A frame's request with its host's own word of what the workers hold of
/// the prompt: the rows it names, or the column it fills.
struct Fed<'a> {
    frame: FrameRequest<'a>,
    named: Option<AffinityRows<'a>>,
    column: &'a [AffinityFacts],
}

impl RequestMaker for Fed<'_> {
    fn model(&self) -> &str {
        self.frame.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let mut request = self.frame.make(spec);
        request.facts.named = self.named;
        request.facts.affinity = self.column;
        request
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.frame.workers()
    }
}

/// One document, one list, one answer on both feeds. The entry a frame's
/// decision asks takes a host's own word of what the workers hold of a
/// prompt, as rows it names or as a column filled from them, and decides
/// alike by either. The profile takes no sample, so its rows are named
/// under no cap. Twenty of the list's twenty-eight workers hold the
/// request's first block and eight of those its second, and all twenty are
/// named. Each seed's pick is one of the deepest holders on either feed,
/// the same one, and the deepest tie.
#[test]
fn a_served_request_decides_by_the_rows_named_under_no_cap_as_by_the_column_filled_from_them() {
    let document: PolicySpec = serde_json::from_str(
        r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": { "p": { "score": ["prefix-affinity"], "pick": "best" } },
          "bindings": [{ "profile": "p" }]
        }"#,
    )
    .expect("a document of this build");
    let store = Arc::new(MemoryStore::new());
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let cap = spec.engine().affinity_rows_cap(BOOKED, Phase::Single);
    assert_eq!(cap, u32::MAX, "a profile that takes no sample");
    store.apply(spec).expect("above the unbound start");
    for i in 0..28 {
        member(&store, i);
    }
    let objects = holder(&store);
    objects.follow(&*store);
    // The rows by their place in the list: the first eight hold both of
    // the request's blocks, the next twelve its first.
    let depth = |row: u32| if row < 8 { 2 } else { 1 };
    let holder = |row: u32| AffinityHolder {
        row,
        overlap_blocks: depth(row),
        predicted_blocks: 0,
    };
    let holders: Vec<AffinityHolder> = (0..20).map(holder).collect();
    let named = AffinityRows {
        holders: &holders,
        ..AffinityRows::default()
    };
    // The column a host fills from the same holders.
    let mut column = vec![AffinityFacts::default(); 28];
    for holder in &holders {
        column[holder.row as usize].overlap_blocks = holder.overlap_blocks;
    }
    let now = Instant::now();
    let mut picked = BTreeSet::new();
    for seed in 1..=64 {
        let mut frame = frame_for(BOOKED, 1, seed);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.block_hashes = vec![1, 2];
        facts.hash_kind = proto::HashKind::Token.into();
        let request = Checked::new(frame).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        // The place of the first target in the list.
        let first = |named: Option<AffinityRows<'_>>, column: &[AffinityFacts]| -> u32 {
            store.deciding(|world| {
                let generation = objects.registry().model_generation(BOOKED);
                let generation = generation.as_deref().expect("the model's list");
                let lent = objects.lent(generation, Role::Single);
                let frame = FrameRequest::new(&request, &read, BUDGET, WorkerLists::Single(lent));
                let fed = Fed {
                    frame,
                    named,
                    column,
                };
                let decided = world.front().decide_without_lease(&fed, now);
                let (decision, _) = decided.expect("the engine decides");
                let pick = decision.pick().expect("a pick over a ready list");
                let first = pick.targets.first().expect("a pick names a target");
                first.position
            })
        };
        let to = first(Some(named), &[]);
        assert_eq!(to, first(None, &column), "seed {seed}");
        assert!(to < 8, "seed {seed} went to the worker at {to}");
        picked.insert(to);
    }
    assert!(picked.len() > 1, "the deepest holders tie: {picked:?}");
}

/// A frame's request is given the snapshot where its document reads load
/// reports and no stage of it keeps them: the three load-led presets. A
/// document that reads no report is given none, and neither is one that
/// holds the resident-work stage, which keeps the reports it is told.
#[test]
fn a_request_is_given_the_snapshot_where_its_document_reads_reports_and_keeps_none() {
    let applied = |document: &str| {
        let document: PolicySpec =
            serde_json::from_str(document).expect("a document of this build");
        AppliedSpec::validate(&document).expect("the document validates")
    };
    for name in ["least_load", "power_of_two", "cache_aware"] {
        assert!(hands_reports(&preset(name, 1)), "{name} reads load reports");
    }
    let unread = applied(
        r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": { "p": { "pick": "random" } },
          "bindings": [{ "profile": "p" }]
        }"#,
    );
    assert!(!hands_reports(&unread), "the document reads no report");
    let keeping = applied(
        r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": { "p": {
            "score": ["least-load", { "resident-work": { "request_tokens": 512 } }],
            "pick": "best"
          } },
          "bindings": [{ "profile": "p" }]
        }"#,
    );
    assert!(
        !hands_reports(&keeping),
        "a stage of the document keeps reports"
    );
}

/// The model of the pair's cases.
const PAIRED: &str = "model-paired";

/// The pair plan's document with the stages of `catalog`: a request of the
/// prefill or the decode phase under one pair binding, and a single request
/// beside it. Each side's profile keeps the workers `filter` keeps and
/// draws among them by its own picker, the prefills' first of `picks`.
fn pair_plan(catalog: &StageRegistry, filter: &str, picks: [&str; 2]) -> AppliedSpec {
    let [prefill, decode] = picks;
    let document = format!(
        r#"{{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": {{
            "prefills": {{ "filter": {filter}, "pick": "{prefill}" }},
            "decodes": {{ "filter": {filter}, "pick": "{decode}" }}
          }},
          "bindings": [
            {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
              "profiles": {{ "prefill": "prefills", "decode": "decodes" }} }},
            {{ "profile": "prefills" }}
          ]
        }}"#
    );
    let document: PolicySpec = serde_json::from_str(&document).expect("a document of this build");
    staged(&document, catalog)
}

/// The pair plan's document of this build's own stages: each side by a
/// seeded draw among the workers its health filter keeps.
fn stock_pair_plan() -> AppliedSpec {
    let draws = ["random", "random"];
    pair_plan(&StageRegistry::builtin(), r#"["healthy"]"#, draws)
}

/// A ready member `name` of [`PAIRED`] on `role`'s side, on the HTTP wire:
/// one `only` alone dials where it names a gateway.
fn side_member(store: &MemoryStore, name: &str, role: Role, only: Option<GatewayId>) -> WorkerId {
    let key = format!("http://{name}:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::HTTP,
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        reachable_from: only.map_or(Reachability::Any, Reachability::Gateway),
        ..WorkerFacts::default()
    };
    let id = store.upsert(facts).id();
    store.publish_now();
    let row = WorkersStore::get(store, id).expect("the row");
    let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
    ready.expect("the row stands");
    id
}

/// The `n`th frame for [`PAIRED`] at its prefill, with what its gateway
/// states of itself: its wires as the wire carries them, and its hint.
fn at_prefill(n: u8, wires: WireModes, hint: Option<u32>) -> proto::DecideRequest {
    let mut frame = frame_for(PAIRED, n, u64::from(n));
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.phase = proto::Phase::Prefill.into();
    frame.constraints = Some(proto::Constraints {
        wire_modes: u32::from(wires.bits()),
        subset_hint: hint,
        ..proto::Constraints::default()
    });
    frame
}

/// The pair's decision of `frame` as `gateway`'s at `now`: what a deciding
/// thread asks for a pair at its prefill, behind the holder's follow. The
/// load monitor published nothing: the request reads no report.
fn served_pair(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    now: Instant,
) -> Decision {
    let unread = Arc::new(LoadSnapshot::default());
    served_pair_reading(store, objects, frame, (gateway, now), &unread)
}

/// [`served_pair`] under `loads`, the snapshot the load monitor published.
fn served_pair_reading(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    loads: &Arc<LoadSnapshot>,
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), None);
    objects.follow(store);
    let mut frames = FrameBuffers::default();
    let own = Part::default();
    let decided =
        store.deciding(|world| decide_pair(world, objects, &taken, &mut frames, loads, &own));
    decided.expect("the engine decides").0
}

/// One frame of [`GATEWAY`]'s decided and its pick leased under term 1, as
/// a deciding thread does both under one load of the world: the decision,
/// the prefill side's lease, and the decode side's where it was issued.
fn leased_pair(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, Lease, Option<Lease>) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, GATEWAY, (now, now), None);
    objects.follow(store);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    store.deciding(|world| {
        let decided = decide_pair(world, objects, &taken, &mut frames, &loads, &own);
        let (decision, booked) = decided.expect("the engine decides");
        let filed = issue_pair(store, world, &taken, 1, &decision, booked);
        let (prefill, decode) = filed.expect("the store's own term").expect("a pick");
        (decision, prefill, decode)
    })
}

/// The prefill and the decode each target of a pick names, in its order.
fn pairs_of(decision: &Decision) -> Vec<(WorkerId, WorkerId)> {
    let targets = decision
        .pick()
        .into_iter()
        .flat_map(|pick| pick.targets.iter());
    let pairs = targets.filter_map(|target| Some((target.id, target.pair?.decode_id)));
    pairs.collect()
}

/// A shed's reason beside the stage it names; `None` for any other answer.
fn shed_of(decision: &Decision) -> Option<(ShedReason, &'static str)> {
    match decision {
        Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
        Decision::Pick(_) | Decision::Queue(_) => None,
    }
}

/// A pair at its prefill is decided over its model's two lists, and each
/// side's list is the asking gateway's to take by that side's own places
/// and vetoes. On each side another gateway alone dials one worker and the
/// asking gateway vetoes another, at places that differ between the sides.
/// Every frame is answered a pick, and every target of it names the one
/// prefill left with the one decode left: a place of one list says nothing
/// of the other. The other gateway dials all six and vetoes none: its
/// picks name every pair.
#[test]
fn a_pair_at_its_prefill_takes_each_side_by_that_sides_own_places_and_vetoes() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let member = |name: &str, role, only| side_member(&store, name, role, only);
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = [
        member("p0", prefill, None),
        member("p1", prefill, Some(OTHER)),
        member("p2", prefill, None),
    ];
    let decodes = [
        member("d0", decode, None),
        member("d1", decode, None),
        member("d2", decode, Some(OTHER)),
    ];
    let objects = holder(&store);
    let now = Instant::now();
    let until = Some(now + Duration::from_secs(60));
    store.gateway_veto(GATEWAY, prefills[2], until);
    store.gateway_veto(GATEWAY, decodes[0], until);
    let named = |gateway: GatewayId| -> BTreeSet<(WorkerId, WorkerId)> {
        let mut named = BTreeSet::new();
        for n in 1..=64 {
            let frame = at_prefill(n, WireModes::NONE, None);
            let answer = served_pair(&store, &objects, frame, gateway, now);
            let Decision::Pick(pick) = &answer else {
                panic!("frame {n} is answered no pick: {answer:?}");
            };
            let pairs = pairs_of(&answer);
            assert_eq!(pairs.len(), pick.targets.len(), "a decode a target");
            named.extend(pairs);
        }
        named
    };
    assert_eq!(named(GATEWAY), BTreeSet::from([(prefills[0], decodes[1])]));
    let every = prefills
        .iter()
        .flat_map(|&p| decodes.iter().map(move |&d| (p, d)));
    let every: BTreeSet<(WorkerId, WorkerId)> = every.collect();
    assert_eq!(named(OTHER), every);
}

/// The `n`th frame for [`PAIRED`] at `phase` that names no lease and
/// states nothing of its gateway.
fn alone_at(n: u8, phase: proto::Phase) -> proto::DecideRequest {
    let mut frame = frame_for(PAIRED, n, u64::from(n));
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.phase = phase.into();
    frame
}

/// The decision of `frame` as `gateway`'s at `now`: what a deciding
/// thread asks for a frame that stands alone at a side's phase, behind
/// the holder's follow. The load monitor published nothing. With it goes
/// the decider's place at the files of leases where the case lends one.
fn served_alone(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    place: Option<&AtWork>,
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), place);
    objects.follow(store);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let decided =
        store.deciding(|world| decide_alone(world, objects, &taken, &mut frames, &loads, &own));
    decided.expect("the engine decides").0
}

/// A prefill frame, and a decode frame that names no lease, stand alone
/// under a plan of one profile. Each is decided over the list of its own
/// phase's role, and that list is the asking gateway's to take by that
/// role's reach and standing vetoes. On each side another gateway alone
/// dials one worker and the asking gateway vetoes another, at places that
/// differ between the sides. Every frame is answered a pick whose every
/// target is the one worker left of its own side, with no decode beside
/// it: no worker of the other side and none of the single role is named.
/// The other gateway dials every worker and vetoes none: its picks name
/// the three of the frame's side.
#[test]
fn a_frame_that_stands_alone_takes_its_own_phases_list_by_that_roles_places_and_vetoes() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(preset("random", 1))
        .expect("above the unbound start");
    let member = |name: &str, role, only| side_member(&store, name, role, only);
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = [
        member("p0", prefill, None),
        member("p1", prefill, Some(OTHER)),
        member("p2", prefill, None),
    ];
    let decodes = [
        member("d0", decode, None),
        member("d1", decode, None),
        member("d2", decode, Some(OTHER)),
    ];
    member("s0", Role::Single, None);
    let objects = holder(&store);
    let now = Instant::now();
    let until = Some(now + Duration::from_secs(60));
    store.gateway_veto(GATEWAY, prefills[2], until);
    store.gateway_veto(GATEWAY, decodes[0], until);
    let named = |gateway: GatewayId, phase: proto::Phase| -> BTreeSet<WorkerId> {
        let mut named = BTreeSet::new();
        for n in 1..=64 {
            let asking = (gateway, now);
            let answer = served_alone(&store, &objects, alone_at(n, phase), asking, None);
            let Decision::Pick(pick) = &answer else {
                panic!("frame {n} at {phase:?} is answered no pick: {answer:?}");
            };
            let alone = pick.targets.iter().all(|target| target.pair.is_none());
            assert!(alone, "no decode beside a target: {answer:?}");
            named.extend(pick.targets.iter().map(|target| target.id));
        }
        named
    };
    let (ahead, behind) = (proto::Phase::Prefill, proto::Phase::Decode);
    assert_eq!(named(GATEWAY, ahead), BTreeSet::from([prefills[0]]));
    assert_eq!(named(GATEWAY, behind), BTreeSet::from([decodes[1]]));
    assert_eq!(named(OTHER, ahead), BTreeSet::from(prefills));
    assert_eq!(named(OTHER, behind), BTreeSet::from(decodes));
}

/// A frame that stands alone is decided behind an apply of what waits at
/// the other files where a stage of its OWN phase's binding may refuse by
/// a report of leases: a frame left under another file's watch is applied
/// by the decision, where its decider lent its place. Another phase's
/// binding says nothing for it: where the single requests' picker alone
/// may so refuse, the frame takes no turn there, and what waits is left
/// for its own file's look. A prefill frame and a decode frame alike.
#[test]
fn a_frame_that_stands_alone_applies_what_waits_where_its_own_phases_binding_reads_reports() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    // The picker of the two sides' phases, then of every other request.
    let cases = [
        (["by-reports", "random"], LookOutcome::Quiet),
        (["random", "by-reports"], LookOutcome::Fed),
    ];
    let sides = [
        (proto::Phase::Prefill, Role::Prefill),
        (proto::Phase::Decode, Role::Decode),
    ];
    for (picks, waited) in cases {
        for (phase, role) in sides {
            let [sided, rest] = picks;
            let document = format!(
                r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "profiles": {{
                    "sides": {{ "filter": [], "pick": "{sided}" }},
                    "rest": {{ "filter": [], "pick": "{rest}" }}
                  }},
                  "bindings": [
                    {{ "match": {{ "phase": ["prefill", "decode"] }}, "profile": "sides" }},
                    {{ "profile": "rest" }}
                  ]
                }}"#
            );
            let document: PolicySpec =
                serde_json::from_str(&document).expect("a document of this build");
            let store = Arc::new(MemoryStore::new().with_decision_slots(2));
            store
                .apply(staged(&document, &catalog))
                .expect("above the unbound start");
            let member = side_member(&store, "w0", role, None);
            LeasesStore::set_term(&*store, 1);
            let mut watch = a_frame_under_the_second_files_watch(&store, member);
            // The first file's decision, its place lent.
            let _acting = DecisionSlot::new(0).enter();
            let place = store.at_work();
            let objects = holder(&store);
            let asking = (GATEWAY, Instant::now());
            let decision = served_alone(&store, &objects, alone_at(1, phase), asking, Some(&place));
            assert_eq!(first(&decision), Some(member), "{phase:?}: {decision:?}");
            assert_eq!(watch.look(), waited, "{phase:?} under {picks:?}");
        }
    }
}

/// A pair's request is given the load monitor's snapshot as a single
/// request's is, and both sides read it: under a plan whose sides go by
/// the least load, every pick names first the prefill and the decode whose
/// reports show no queue. Under a snapshot that holds no report no worker
/// stands out, and the picks name more than one pair first.
#[test]
fn a_pairs_sides_read_the_reports_of_the_snapshot_their_request_is_given() {
    let document = r#"{
      "apiVersion": "smg.policy/v1",
      "generation": 1,
      "profiles": {
        "side": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
      },
      "bindings": [
        { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
          "profiles": { "prefill": "side", "decode": "side" } }
      ]
    }"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a document of this build");
    let store = Arc::new(MemoryStore::new());
    store
        .apply(staged(&document, &StageRegistry::builtin()))
        .expect("above the unbound start");
    let prefills = ["p0", "p1"].map(|name| side_member(&store, name, Role::Prefill, None));
    let decodes = ["d0", "d1"].map(|name| side_member(&store, name, Role::Decode, None));
    let objects = holder(&store);
    let now = Instant::now();
    // The second prefill and the first decode report no queue, and the
    // other two are deep in theirs.
    let reports = ["p0", "p1", "d0", "d1"].map(|name| {
        let load = if name == "p1" || name == "d0" {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    let published = LoadSnapshot::from_loads_for_test(reports.into_iter().collect());
    let unread = Arc::new(LoadSnapshot::default());
    let firsts = |loads: &Arc<LoadSnapshot>| -> BTreeSet<(WorkerId, WorkerId)> {
        let frames = (1..=32).map(|n| at_prefill(n, WireModes::NONE, None));
        let asked = (GATEWAY, now);
        let picks = frames.map(|frame| served_pair_reading(&store, &objects, frame, asked, loads));
        let firsts = picks.map(|pick| *pairs_of(&pick).first().expect("a pick"));
        firsts.collect()
    };
    let idle = (prefills[1], decodes[0]);
    assert_eq!(firsts(&published), BTreeSet::from([idle]));
    assert!(firsts(&unread).len() > 1, "no report, and one pair alone");
}

/// A hold of the asking gateway on a side's only worker sheds the pair's
/// frame under the health filter, on either side: no rule lifts a hold,
/// where a doubt of a side's every worker is lifted as the last resort.
/// Another gateway holds no veto, and the pair is its pick.
#[test]
fn a_hold_on_the_only_worker_of_either_side_sheds_a_pairs_frame_under_the_health_filter() {
    for held in [Role::Prefill, Role::Decode] {
        let store = Arc::new(MemoryStore::new());
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        let p = side_member(&store, "p0", Role::Prefill, None);
        let d = side_member(&store, "d0", Role::Decode, None);
        let objects = holder(&store);
        let now = Instant::now();
        let answered = |gateway| {
            let frame = at_prefill(1, WireModes::NONE, None);
            served_pair(&store, &objects, frame, gateway, now)
        };
        assert_eq!(pairs_of(&answered(GATEWAY)), [(p, d)]);
        let on = if held == Role::Prefill { p } else { d };
        store.gateway_veto(GATEWAY, on, Some(now + Duration::from_secs(60)));
        let shed = Some((ShedReason::NoEndpoints, "healthy"));
        assert_eq!(shed_of(&answered(GATEWAY)), shed, "a hold on the {held:?}");
        assert_eq!(pairs_of(&answered(OTHER)), [(p, d)]);
    }
}

/// The answers of a pair's frame come in one order. A side with no worker
/// at all is decided over no worker, whatever the gateway's wires take of
/// the other side: one side alone is no pair. Else a side the gateway dials
/// and can take nothing of sheds the frame under that stage, though the
/// gateway dials no worker of the other side. Else a side it dials no
/// worker of is decided over no worker. A model nobody serves is too.
#[test]
fn a_pairs_frame_answers_a_side_with_no_worker_then_one_emptied_then_one_not_dialed() {
    let now = Instant::now();
    // The answer of GATEWAY's frame over a fleet of `decodes`, beside one
    // prefill every gateway dials where `prefill` says so, with the
    // engine's own answer over no worker for the same frame.
    let answered = |prefill: bool, decodes: &[Option<GatewayId>], wires, hint: Option<u32>| {
        let store = Arc::new(MemoryStore::new());
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        if prefill {
            side_member(&store, "p0", Role::Prefill, None);
        }
        for (n, &only) in decodes.iter().enumerate() {
            side_member(&store, &format!("d{n}"), Role::Decode, only);
        }
        let objects = holder(&store);
        let frame = at_prefill(1, wires, hint);
        let request = Checked::new(frame.clone()).expect("a frame the record holds");
        let over_none = store.deciding(|world| {
            let mut lists = Lists::default();
            let read = lists.of(&request);
            let maker = FrameRequest::new(&request, &read, BUDGET, WorkerLists::None);
            let decided = world.front().decide_without_lease(&maker, now);
            shed_of(&decided.expect("the engine decides").0)
        });
        assert!(over_none.is_some(), "no worker is no pick");
        let answer = served_pair(&store, &objects, frame, GATEWAY, now);
        (answer, over_none)
    };
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    // A side with no worker at all: over no worker, though the gateway's
    // wires take every worker of the other side and a hint of 0 takes them
    // too.
    let lone: [(bool, &[Option<GatewayId>]); 2] = [(true, &[]), (false, &[None])];
    for (prefill, decodes) in lone {
        for (wires, hint) in [(WireModes::NONE, None), (grpc, None), (http, Some(0))] {
            let (answer, over_none) = answered(prefill, decodes, wires, hint);
            assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
        }
    }
    // A decode another gateway alone dials, beside a prefill the gateway's
    // wires or its hint take: the emptied side sheds.
    let theirs = [Some(OTHER)];
    let wires = Some((ShedReason::NoEndpoints, WIRES));
    assert_eq!(shed_of(&answered(true, &theirs, grpc, None).0), wires);
    let hinted = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    assert_eq!(shed_of(&answered(true, &theirs, http, Some(0)).0), hinted);
    // The same decode beside a prefill the gateway takes: over no worker.
    let (answer, over_none) = answered(true, &theirs, http, None);
    assert_eq!(shed_of(&answer), over_none);
    // And with a decode it dials, the pair is picked.
    let (answer, _) = answered(true, &[None], http, None);
    assert!(matches!(answer, Decision::Pick(_)), "{answer:?}");
    // A model nobody serves: over no worker.
    let (answer, over_none) = answered(false, &[], WireModes::NONE, None);
    assert_eq!(shed_of(&answer), over_none);
}

/// A pair's pick is filed a lease a side under one id: the prefill side's
/// on the pick's prefill, the decode side's on its decode, each under the
/// session the frame's own reader states of a request of that side's phase,
/// and each side's worker is booked for the decisions behind it. A frame
/// that states the later decode files its decode side chosen: the answer
/// carries the prefill's lease alone, and the decode's worker is booked no
/// more.
#[test]
fn a_pairs_pick_is_filed_a_lease_a_side_and_a_later_decode_stands_chosen() {
    let store = Arc::new(MemoryStore::new());
    let read = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let (kept, own) = (Arc::clone(&read), Arc::downgrade(&store));
    let stage = move || -> Box<dyn Filter> {
        Box::new(Late {
            store: Weak::clone(&own),
            late: Arc::default(),
            read: Arc::clone(&kept),
        })
    };
    catalog.filter("late", stage).expect("a free name");
    let late = r#"["healthy", "late"]"#;
    store
        .apply(pair_plan(&catalog, late, ["random", "random"]))
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let d = side_member(&store, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    let now = Instant::now();
    // The `n`th frame of one conversation, at `phase`: its key goes plain,
    // so each phase's reader makes the session of its own phase.
    let framed = |n: u8, phase: proto::Phase| {
        let mut frame = at_prefill(n, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        (facts.phase, facts.prompt_tokens) = (phase.into(), 300);
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        frame
    };
    let leased = |n: u8, later: bool| {
        let mut frame = framed(n, proto::Phase::Prefill);
        frame.decode_later = later;
        let (_, prefill, decode) = leased_pair(&store, &objects, frame, now);
        (prefill, decode)
    };
    // A side's session, as the frame's own reader states it of a request
    // of that phase.
    let session_at = |phase: proto::Phase| {
        let request = Checked::new(framed(9, phase)).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session()
    };
    // What the two sides of the newest decision read of their workers.
    let counts = || -> Counts {
        let read = read.lock();
        let sides = read.iter().rev().take(2);
        let mut counts: Counts = sides.flat_map(|side| side[0].clone()).collect();
        counts.sort_unstable();
        counts
    };
    let side_on_file = |lease: &Lease| {
        let sides = LeasesStore::sides(&*store, GATEWAY, lease.id).expect("on file");
        (sides.record.decode, sides.decode.map(|side| side.state))
    };

    let (prefill, decode) = leased(1, false);
    let decode = decode.expect("issued with its prefill");
    assert_eq!(
        (prefill.side, prefill.booked, prefill.tokens),
        (Side::Prefill, p, 300)
    );
    let sessions = [proto::Phase::Prefill, proto::Phase::Decode].map(session_at);
    assert_ne!(sessions[0], sessions[1], "a session a side");
    assert_eq!(prefill.session, sessions[0]);
    let beside = Lease {
        side: Side::Decode,
        booked: d,
        session: sessions[1],
        ..prefill
    };
    assert_eq!(decode, beside, "one id, and the prompt's tokens on both");
    let issued = (Some(DecodeMark), Some(LeaseState::ISSUED));
    assert_eq!(side_on_file(&prefill), issued);

    // The next pair reads both workers booked, and states the later decode.
    let (prefill, decode) = leased(2, true);
    assert_eq!(counts(), by_id(&[p, d], &[1, 1]));
    assert_eq!(decode, None, "the answer carries the prefill's lease alone");
    let chosen = (Some(DecodeMark), Some(LeaseState::Chosen));
    assert_eq!(side_on_file(&prefill), chosen);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 2);

    // A third reads the prefill booked twice and the decode once.
    leased(3, false);
    assert_eq!(counts(), by_id(&[p, d], &[2, 1]));
}

/// A pair's lease keeps each target of its pick for a dispatch that names
/// one by its index: the prefill side's record the target's prefill, and
/// the decode side's its decode.
#[test]
fn a_pairs_lease_keeps_each_targets_prefill_and_decode_for_a_dispatch_by_index() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    side_member(&store, "p0", Role::Prefill, None);
    for name in ["d0", "d1", "d2"] {
        side_member(&store, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    let frame = at_prefill(1, WireModes::NONE, None);
    let (decision, prefill, _) = leased_pair(&store, &objects, frame, Instant::now());
    let targets = &decision.pick().expect("a pick").targets;
    assert!(targets.len() > 1, "a target behind the first: {decision:?}");
    let on_file = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    for (index, target) in (0u8..).zip(targets.iter()) {
        let pair = target.pair.expect("a decode a target");
        let sides = [Side::Prefill, Side::Decode].map(|side| on_file.target_of(side, index));
        let named = [
            (target.id, target.dp_rank),
            (pair.decode_id, pair.decode_dp_rank),
        ];
        assert_eq!(sides, named.map(Some), "the target at {index}");
    }
    let past = u8::try_from(targets.len()).expect("a few targets");
    assert_eq!(on_file.target_of(Side::Decode, past), None);
}

/// The `random` picker, as a stage that says it may refuse by a report of
/// leases.
#[derive(Debug)]
struct ByReports;

impl Lifecycle for ByReports {}

impl Picker for ByReports {
    fn name(&self) -> &'static str {
        "by-reports"
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

    fn refuses_by_reports(&self) -> bool {
        true
    }
}

/// A pair either of whose sides holds a stage that may refuse by a report
/// of leases is decided behind an apply of what waits at the other files:
/// a frame left under another file's watch is applied by the decision,
/// where its decider lent its place. A pair whose stages refuse by none
/// takes no turn there, and the frame waits for its own file's look.
#[test]
fn a_pair_either_of_whose_sides_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    let cases = [
        (["by-reports", "random"], LookOutcome::Quiet),
        (["random", "by-reports"], LookOutcome::Quiet),
        (["random", "random"], LookOutcome::Fed),
    ];
    for (picks, waited) in cases {
        let store = Arc::new(MemoryStore::new().with_decision_slots(2));
        store
            .apply(pair_plan(&catalog, "[]", picks))
            .expect("above the unbound start");
        let p = side_member(&store, "p0", Role::Prefill, None);
        side_member(&store, "d0", Role::Decode, None);
        LeasesStore::set_term(&*store, 1);
        let mut watch = a_frame_under_the_second_files_watch(&store, p);
        // The first file's decision, its place lent.
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let objects = holder(&store);
        let now = Instant::now();
        let frame = at_prefill(1, WireModes::NONE, None);
        let request = Checked::new(frame).expect("a frame the record holds");
        let taken = taken_at(&request, GATEWAY, (now, now), Some(&place));
        objects.follow(&*store);
        let (mut frames, own) = (FrameBuffers::default(), Part::default());
        let loads = Arc::new(LoadSnapshot::default());
        let decided =
            store.deciding(|world| decide_pair(world, &objects, &taken, &mut frames, &loads, &own));
        let (decision, _) = decided.expect("the engine decides");
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert_eq!(watch.look(), waited, "{picks:?}");
    }
}

/// A decode call's frame for [`PAIRED`]: the `n`th of [`at_prefill`]'s that
/// names `lease`, which its reader takes at the decode phase.
fn at_call(n: u8, lease: LeaseId, wires: WireModes, hint: Option<u32>) -> proto::DecideRequest {
    let mut frame = at_prefill(n, wires, hint);
    frame.decode_of = Some(lease.0);
    frame
}

/// The decision of a decode call's `frame` as `gateway`'s at `now`, behind
/// the pair `held`: what a deciding thread asks for the call once the store
/// answered for its lease, behind the holder's follow. With it go the
/// snapshot the load monitor published, and the decider's place at the
/// files of leases where the case lends one.
fn served_call(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    held: (WorkerId, Option<(WorkerId, u16)>),
    (loads, place): (&Arc<LoadSnapshot>, Option<&AtWork>),
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), place);
    objects.follow(store);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let decided = store
        .deciding(|world| decide_decode(world, objects, &taken, held, &mut frames, loads, &own));
    decided.expect("the engine decides").0
}

/// One decode call of [`GATEWAY`]'s decided and its pick issued, as a
/// deciding thread does both under one load of the world: the store's
/// answer for the lease the frame names first, the decision behind it, and
/// for a pick the decode side's lease with its issue. The load monitor
/// published nothing.
fn leased_call(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, Option<(Lease, DecodeIssue)>) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let lease = request.decode_of().expect("a decode call's frame");
    let taken = taken_at(&request, GATEWAY, (now, now), None);
    objects.follow(store);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let held = store.held_pair(GATEWAY, lease);
    let held = held.expect("the lease takes a decode side");
    store.deciding(|world| {
        let decided = decide_decode(world, objects, &taken, held, &mut frames, &loads, &own);
        let (decision, booked) = decided.expect("the engine decides");
        let issued = issue_decode(store, world, &taken, lease, &decision, booked);
        (decision, issued.expect("the side stands chosen"))
    })
}

/// The `n`th pair of [`GATEWAY`]'s leased with its decode asked for later,
/// and its prefill dispatched to the pick's target at `index`: the pair's
/// decision and the prefill side's lease.
fn held_pair_at(
    store: &MemoryStore,
    objects: &Objects<Members>,
    (n, index): (u8, u8),
    now: Instant,
) -> (Decision, Lease) {
    let mut frame = at_prefill(n, WireModes::NONE, None);
    frame.decode_later = true;
    let (decision, prefill, decode) = leased_pair(store, objects, frame, now);
    assert_eq!(decode, None, "the decode side stands chosen");
    let sent = Dispatch {
        side: Side::Prefill,
        target_index: index,
        since_decision_micros: 0,
    };
    assert!(LeasesStore::dispatched(
        store, GATEWAY, prefill.id, sent, now
    ));
    (decision, prefill)
}

/// A decode call is decided over its model's two lists, and the decode
/// side's list alone is the asking gateway's to take, by that side's reach
/// and standing vetoes. Another gateway alone dials one decode and the
/// asking gateway holds another: every call's pick names the one decode
/// left. The held prefill was sent already and is found wherever it stands:
/// a hold of the asking gateway on it sheds a pair at its prefill and takes
/// nothing from the call, whose every target names it.
#[test]
fn a_decode_call_takes_the_decode_side_alone_and_finds_its_prefill_under_a_hold() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let decodes = [
        side_member(&store, "d0", Role::Decode, None),
        side_member(&store, "d1", Role::Decode, None),
        side_member(&store, "d2", Role::Decode, Some(OTHER)),
    ];
    let objects = holder(&store);
    let now = Instant::now();
    let until = Some(now + Duration::from_secs(60));
    store.gateway_veto(GATEWAY, p, until);
    store.gateway_veto(GATEWAY, decodes[0], until);
    let frame = at_prefill(1, WireModes::NONE, None);
    let pair = served_pair(&store, &objects, frame, GATEWAY, now);
    let at_health = Some((ShedReason::NoEndpoints, "healthy"));
    assert_eq!(shed_of(&pair), at_health, "a pair at its prefill");
    let unread = Arc::new(LoadSnapshot::default());
    let named = |gateway: GatewayId| -> BTreeSet<(WorkerId, WorkerId)> {
        let mut named = BTreeSet::new();
        for n in 1..=32 {
            let frame = at_call(n, LeaseId(1), WireModes::NONE, None);
            let (asked, given) = ((gateway, now), (&unread, None));
            let answer = served_call(&store, &objects, frame, asked, (p, None), given);
            let Decision::Pick(pick) = &answer else {
                panic!("call {n} is answered no pick: {answer:?}");
            };
            let pairs = pairs_of(&answer);
            assert_eq!(pairs.len(), pick.targets.len(), "a decode a target");
            named.extend(pairs);
        }
        named
    };
    assert_eq!(named(GATEWAY), BTreeSet::from([(p, decodes[1])]));
    let every: BTreeSet<(WorkerId, WorkerId)> = decodes.iter().map(|&d| (p, d)).collect();
    assert_eq!(named(OTHER), every);
}

/// The answers of a decode call come in a pair's order, and the decode side
/// alone states them. A model with no decode and a model nobody serves are
/// decided over no worker, whatever the gateway states. Else decodes the
/// gateway dials, whose every worker its wires or a hint of 0 take, shed
/// the call under that stage. Else decodes it dials none of are decided
/// over no worker. What the gateway dials of the prefills says nothing: the
/// held prefill was sent.
#[test]
fn a_decode_call_answers_no_decode_then_the_decodes_emptied_then_none_dialed() {
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    // GATEWAY's call over a fleet of one prefill, which `prefill` alone
    // dials, and of `decodes`, with the engine's own answer over no worker
    // for the same call.
    type Fleet<'a> = (Option<GatewayId>, &'a [Option<GatewayId>]);
    let answered = |(prefill, decodes): Fleet<'_>, wires, hint: Option<u32>| {
        let store = Arc::new(MemoryStore::new());
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        let p = side_member(&store, "p0", Role::Prefill, prefill);
        for (n, &only) in decodes.iter().enumerate() {
            side_member(&store, &format!("d{n}"), Role::Decode, only);
        }
        let objects = holder(&store);
        let frame = at_call(1, LeaseId(1), wires, hint);
        let request = Checked::new(frame.clone()).expect("a frame the record holds");
        let (over_none, traced) = store.deciding(|world| {
            let mut lists = Lists::default();
            let read = lists.of(&request);
            let maker = FrameRequest::new(&request, &read, BUDGET, WorkerLists::None);
            let held = Target {
                id: p,
                ..Target::default()
            };
            let decided = world
                .front()
                .decide_decode_without_lease(&maker, &held, now);
            let decided = decided.expect("the engine decides");
            let trace = decided.trace();
            (shed_of(&decided), (trace.seed, trace.generation))
        });
        assert!(over_none.is_some(), "no worker is no pick");
        assert_eq!(traced.0, 1, "the frame's own seed");
        let (asked, given) = ((GATEWAY, now), (&unread, None));
        let answer = served_call(&store, &objects, frame, asked, (p, None), given);
        // A shed the thread makes itself states what the engine's states:
        // the frame's seed and the document's generation.
        if shed_of(&answer).is_some() {
            let trace = answer.trace();
            assert_eq!((trace.seed, trace.generation), traced);
        }
        (answer, over_none)
    };
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    let stated = [(WireModes::NONE, None), (grpc, None), (http, Some(0))];
    // No decode at all: over no worker, whatever the gateway states.
    for (wires, hint) in stated {
        let (answer, over_none) = answered((None, &[]), wires, hint);
        assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
    }
    // A decode the gateway dials and its wires or its hint take: shed under
    // that stage, whoever dials the prefill.
    let wires = Some((ShedReason::NoEndpoints, WIRES));
    let hinted = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    for prefill in [None, Some(OTHER)] {
        let fleet: Fleet<'_> = (prefill, &[None]);
        assert_eq!(shed_of(&answered(fleet, grpc, None).0), wires);
        assert_eq!(shed_of(&answered(fleet, http, Some(0)).0), hinted);
    }
    // A decode another gateway alone dials: over no worker, whatever the
    // gateway states.
    for (wires, hint) in stated {
        let (answer, over_none) = answered((None, &[Some(OTHER)]), wires, hint);
        assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
    }
    // A decode the gateway dials: a pick behind the held prefill, though
    // another gateway alone dials that prefill.
    for prefill in [None, Some(OTHER)] {
        let (answer, _) = answered((prefill, &[None]), http, None);
        assert!(matches!(answer, Decision::Pick(_)), "{answer:?}");
    }
    // A model nobody serves: over no worker.
    let store = Arc::new(MemoryStore::new());
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let objects = holder(&store);
    let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
    let (asked, given) = ((GATEWAY, now), (&unread, None));
    let held = (WorkerId(7), None);
    let answer = served_call(&store, &objects, frame, asked, held, given);
    assert!(shed_of(&answer).is_some(), "{answer:?}");
}

/// The pair plan's document whose sides go by the least load, the best
/// first, among the workers the health filter keeps.
fn least_load_pair_plan() -> AppliedSpec {
    let document = r#"{
      "apiVersion": "smg.policy/v1",
      "generation": 1,
      "profiles": {
        "side": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
      },
      "bindings": [
        { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
          "profiles": { "prefill": "side", "decode": "side" } }
      ]
    }"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a document of this build");
    staged(&document, &StageRegistry::builtin())
}

/// A decode call's request is given the load monitor's snapshot as its
/// pair's first decision is: under a plan whose sides go by the least load,
/// every call's pick names first the decode whose report shows no queue.
/// Under a snapshot that holds no report no decode stands out, and the
/// calls name more than one first.
#[test]
fn a_decode_calls_request_reads_the_reports_of_the_snapshot_it_is_given() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(least_load_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let decodes = ["d0", "d1", "d2"].map(|name| side_member(&store, name, Role::Decode, None));
    let objects = holder(&store);
    let now = Instant::now();
    // The second decode reports no queue, and every other worker is deep
    // in its own.
    let reports = ["p0", "d0", "d1", "d2"].map(|name| {
        let load = if name == "d1" {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    let published = LoadSnapshot::from_loads_for_test(reports.into_iter().collect());
    let unread = Arc::new(LoadSnapshot::default());
    let firsts = |loads: &Arc<LoadSnapshot>| -> BTreeSet<WorkerId> {
        let calls = (1..=32).map(|n| at_call(n, LeaseId(1), WireModes::NONE, None));
        let picks = calls.map(|frame| {
            let (asked, given) = ((GATEWAY, now), (loads, None));
            served_call(&store, &objects, frame, asked, (p, None), given)
        });
        let firsts = picks.map(|pick| pairs_of(&pick).first().expect("a pick").1);
        firsts.collect()
    };
    assert_eq!(firsts(&published), BTreeSet::from([decodes[1]]));
    assert!(firsts(&unread).len() > 1, "no report, and one decode alone");
}

/// A decode call whose decode profile holds a stage that may refuse by a
/// report of leases is decided behind an apply of what waits at the other
/// files, where its decider lent its place. The prefill profile's stages
/// say nothing of it: the call runs none of them, and a frame left under
/// another file's watch waits for that file's look.
#[test]
fn a_decode_call_whose_decode_side_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    let cases = [
        (["random", "by-reports"], LookOutcome::Quiet),
        (["by-reports", "random"], LookOutcome::Fed),
        (["random", "random"], LookOutcome::Fed),
    ];
    for (picks, waited) in cases {
        let store = Arc::new(MemoryStore::new().with_decision_slots(2));
        store
            .apply(pair_plan(&catalog, "[]", picks))
            .expect("above the unbound start");
        let p = side_member(&store, "p0", Role::Prefill, None);
        side_member(&store, "d0", Role::Decode, None);
        LeasesStore::set_term(&*store, 1);
        let mut watch = a_frame_under_the_second_files_watch(&store, p);
        // The first file's decision, its place lent.
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let objects = holder(&store);
        let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
        let unread = Arc::new(LoadSnapshot::default());
        let (asked, given) = ((GATEWAY, Instant::now()), (&unread, Some(&place)));
        let decision = served_call(&store, &objects, frame, asked, (p, None), given);
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert_eq!(watch.look(), waited, "{picks:?}");
    }
}

/// A decode call starts from the decode its pair's first decision chose for
/// the target the prefill's dispatch took: the store's answer for the lease
/// reaches the engine. Under a decode profile that takes turns, and so
/// keeps its pick, the call answers that decode again, the pick's first or
/// the one the dispatch named, where a new turn names another. The side is
/// issued on it under the pair's own id.
#[test]
fn a_decode_call_starts_from_the_decode_of_the_target_its_prefills_dispatch_took() {
    let store = Arc::new(MemoryStore::new());
    let turns = ["random", "round-robin"];
    store
        .apply(pair_plan(&StageRegistry::builtin(), "[]", turns))
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    for name in ["d0", "d1", "d2"] {
        side_member(&store, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    let now = Instant::now();
    for (n, index) in [(1, 0), (2, 1)] {
        let (first, prefill) = held_pair_at(&store, &objects, (n, index), now);
        let chosen = pairs_of(&first);
        assert!(chosen.len() > 1 && chosen[0].1 != chosen[1].1, "{first:?}");
        let took = chosen[usize::from(index)].1;
        let held = store.held_pair(GATEWAY, prefill.id);
        assert_eq!(
            held.map(|(prefill, decode)| (prefill, decode.map(|at| at.0))),
            Ok((p, Some(took)))
        );
        let frame = at_call(n + 10, prefill.id, WireModes::NONE, None);
        let (again, issued) = leased_call(&store, &objects, frame, now);
        let named = pairs_of(&again);
        assert_eq!(named.first(), Some(&(p, took)), "the target at {index}");
        let (side, _) = issued.expect("a pick's issue");
        let lease = (side.id, side.side, side.booked);
        assert_eq!(lease, (prefill.id, Side::Decode, took));
        // The side keeps the call's own pick, for a dispatch by its index.
        let on_file = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let targets = &again.pick().expect("a pick").targets;
        assert!(targets.len() > 1, "a target behind the first: {again:?}");
        for (at, target) in (0u8..).zip(targets.iter()) {
            let pair = target.pair.expect("a decode a target");
            let kept = on_file.target_of(Side::Decode, at);
            let named = (pair.decode_id, pair.decode_dp_rank);
            assert_eq!(kept, Some(named), "the call's target at {at}");
        }
    }
}

/// A hold of the asking gateway on the only decode sheds a decode call
/// under the health filter: no rule lifts a hold, where a doubt of every
/// decode is lifted as the last resort. Another gateway holds no veto, and
/// the call is its pick.
#[test]
fn a_hold_on_the_only_decode_sheds_a_decode_call_under_the_health_filter() {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let d = side_member(&store, "d0", Role::Decode, None);
    let objects = holder(&store);
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    let answered = |gateway| {
        let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
        let (asked, given) = ((gateway, now), (&unread, None));
        served_call(&store, &objects, frame, asked, (p, None), given)
    };
    assert_eq!(pairs_of(&answered(GATEWAY)), [(p, d)]);
    store.gateway_veto(GATEWAY, d, Some(now + Duration::from_secs(60)));
    let shed = Some((ShedReason::NoEndpoints, "healthy"));
    assert_eq!(shed_of(&answered(GATEWAY)), shed);
    assert_eq!(pairs_of(&answered(OTHER)), [(p, d)]);
}

/// The `random` picker, keeping each lease its engine tells it the issue
/// of.
#[derive(Debug)]
struct Told(Arc<Mutex<Vec<Lease>>>);

impl Lifecycle for Told {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(*lease);
    }
}

impl Picker for Told {
    fn name(&self) -> &'static str {
        "told"
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

/// A decode call's pick is issued under its pair's own lease: the decode
/// side on the pick's first decode, under the session the frame's reader
/// states of a request of the decode phase and with the call's count of
/// tokens, and that decode's worker is booked. The stages hear the issue
/// dated at the call, three seconds behind the pair's own, and the
/// lease's dispatch deadline counts from the call's instant. A call that
/// sheds files nothing. The issue's undo puts the side and the deadline
/// back as the call found them and frees the booking, and the prefill's
/// stands throughout.
#[test]
fn a_decode_calls_pick_is_issued_under_its_pairs_lease_and_told_dated_at_the_call() {
    let store = Arc::new(MemoryStore::new());
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    let stage = move || -> Box<dyn Picker> { Box::new(Told(Arc::clone(&kept))) };
    catalog.picker("told", stage).expect("a free name");
    store
        .apply(pair_plan(&catalog, r#"["healthy"]"#, ["random", "told"]))
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let d = side_member(&store, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    let now = Instant::now();
    let (_, prefill) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(*told.lock(), [prefill], "the prefill side's issue alone");
    // What the leases on file book on the prefill and on the decode.
    let booked = || {
        let load = |name: &str| {
            let url = format!("http://{name}:8000");
            let object = objects.registry().get_by_url(&url).expect("its object");
            object.load()
        };
        [load("p0"), load("d0")]
    };
    let side_on_file = || {
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        sides.decode.map(|side| (side.state, side.target))
    };
    let chosen = Some((LeaseState::Chosen, d));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    // The dispatch deadline of the lease on file.
    let deadline = || {
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        sides.record.dispatch_deadline
    };
    let pairs = deadline();
    // The call's frame of one conversation, at `phase`: its key goes plain,
    // so its reader makes the session of the phase it takes the frame at.
    let framed = |phase: proto::Phase| {
        let mut frame = at_call(2, prefill.id, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        (facts.phase, facts.prompt_tokens) = (phase.into(), 300);
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        frame
    };
    // The session of a request of the decode phase, as the reader of a
    // frame that states that phase and names no lease makes it.
    let session = {
        let mut stated = framed(proto::Phase::Decode);
        stated.decode_of = None;
        let request = Checked::new(stated).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session()
    };
    let call = now + Duration::from_secs(3);

    // A call that sheds files nothing, and no stage hears of it.
    let hinted = at_call(3, prefill.id, WireModes::HTTP, Some(0));
    let (decision, issued) = leased_call(&store, &objects, hinted, call);
    let at_hint = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    assert_eq!((shed_of(&decision), issued.is_none()), (at_hint, true));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    assert_eq!(told.lock().len(), 1);
    assert_eq!(deadline(), pairs, "a shed starts no deadline");

    // The call's frame states the prefill phase: a frame that names a
    // lease is read at the decode phase whatever it states.
    let (decision, issued) = leased_call(&store, &objects, framed(proto::Phase::Prefill), call);
    assert_eq!(pairs_of(&decision), [(p, d)]);
    let (side, issue) = issued.expect("a pick's issue");
    let beside = Lease {
        side: Side::Decode,
        booked: d,
        session,
        tokens: 300,
        ..prefill
    };
    assert_eq!(side, beside, "the pair's own id, generation and instant");
    let dated = Lease {
        issued_micros: prefill.issued_micros + 3_000_000,
        ..side
    };
    assert_eq!(*told.lock(), [prefill, dated], "dated at the call");
    let issued = Some((LeaseState::ISSUED, d));
    assert_eq!((side_on_file(), booked()), (issued, [1, 1]));
    // The side's five seconds count from the call, not from the pair's
    // first decision three seconds earlier.
    assert_eq!(deadline(), call + DISPATCH_TTL);
    assert_ne!(deadline(), pairs);

    assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    assert_eq!(deadline(), pairs, "the deadline the call found");
}

/// The `random` picker, whose issue call panics for a decode side that
/// carries [`FAULTY_TOKENS`], with a payload whose own drop panics.
#[derive(Debug)]
struct Faulty;

/// A panic's payload whose own drop panics.
struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        panic!("the case's own panic, at a payload's drop");
    }
}

/// The count of tokens of the call whose issue [`Faulty`] panics at.
const FAULTY_TOKENS: u32 = 666;

impl Lifecycle for Faulty {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        if lease.side == Side::Decode && lease.tokens == FAULTY_TOKENS {
            resume_unwind(Box::new(Bomb));
        }
    }
}

impl Picker for Faulty {
    fn name(&self) -> &'static str {
        "faulty"
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

/// A stage that panics at the issue a decode call tells it is held and
/// counted, as under a report: the side stands issued on file and the call
/// still gets its lease. The panic's payload panics again at its own drop,
/// and that is held too. An issue whose telling returns ends the streak of
/// faults: the next fault is the first of a streak again.
#[test]
fn a_stage_that_panics_at_a_decode_calls_issue_is_held_and_the_side_stands_issued() {
    let store = Arc::new(MemoryStore::new());
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("faulty", || Box::new(Faulty))
        .expect("a free name");
    store
        .apply(pair_plan(&catalog, "[]", ["random", "faulty"]))
        .expect("above the unbound start");
    side_member(&store, "p0", Role::Prefill, None);
    let d = side_member(&store, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    let now = Instant::now();
    let (_, prefill) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(store.lifecycle_faults(), 0);
    // One call of the lease, whose frame states `tokens`.
    let called = |n: u8, tokens: u32| {
        let mut frame = at_call(n, prefill.id, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.prompt_tokens = tokens;
        let (_, issued) = leased_call(&store, &objects, frame, now);
        issued.expect("the pick the file holds")
    };
    let (side, issue) = called(2, FAULTY_TOKENS);
    assert_eq!((side.side, side.booked), (Side::Decode, d));
    assert_eq!(store.lifecycle_faults(), 1);
    let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    let on_file = sides.decode.map(|side| (side.state, side.target));
    assert_eq!(on_file, Some((LeaseState::ISSUED, d)));
    // The next call's issue is told and returns: the streak ends.
    assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
    called(3, 1);
    assert_eq!(store.lifecycle_faults(), 1);
    assert!(store.call_faulted(), "the first fault of a new streak");
}

/// A ready member `name` of [`PAIRED`] on `role`'s side, as [`side_member`]
/// lays one, that offers a partner the KV transport `transport`.
fn transport_member(store: &MemoryStore, name: &str, role: Role, transport: &str) -> WorkerId {
    let key = format!("http://{name}:8000");
    let descriptor = PairingDescriptor {
        transport: Some(transport.to_owned()),
        ..PairingDescriptor::default()
    };
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::HTTP,
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        pairing: Some(Arc::new(descriptor)),
        ..WorkerFacts::default()
    };
    let id = store.upsert(facts).id();
    store.publish_now();
    let row = WorkersStore::get(store, id).expect("the row");
    let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
    ready.expect("the row stands");
    id
}

/// A pair's two sides pair by the classes of the document's own pairing, at
/// the pair's first decision and at its decode call. Under the pairing of
/// the stock plan a decode on another KV transport than the prefill's is no
/// partner: no pick and no call names it. Under a plan that pairs on
/// nothing every decode is a partner.
#[test]
fn a_pair_and_its_decode_call_pair_by_the_classes_of_the_documents_pairing() {
    let plan = |pairing: Option<&str>| {
        let plans = pairing.map_or_else(String::new, |mode| {
            format!(r#""plans": {{ "pd-pair": {{ "pairing": "{mode}" }} }},"#)
        });
        let document = format!(
            r#"{{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              {plans}
              "profiles": {{ "side": {{ "pick": "random" }} }},
              "bindings": [
                {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                  "profiles": {{ "prefill": "side", "decode": "side" }} }}
              ]
            }}"#
        );
        let document: PolicySpec =
            serde_json::from_str(&document).expect("a document of this build");
        staged(&document, &StageRegistry::builtin())
    };
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    // The decodes the pair's first decisions name, and those its decode
    // calls name behind the one prefill.
    let named = |pairing: Option<&str>| {
        let store = Arc::new(MemoryStore::new());
        store.apply(plan(pairing)).expect("above the unbound start");
        let p = transport_member(&store, "p0", Role::Prefill, "a");
        let same = transport_member(&store, "d0", Role::Decode, "a");
        let other = transport_member(&store, "d1", Role::Decode, "b");
        let objects = holder(&store);
        let mut firsts = BTreeSet::new();
        let mut calls = BTreeSet::new();
        for n in 1..=32 {
            let frame = at_prefill(n, WireModes::NONE, None);
            let first = served_pair(&store, &objects, frame, GATEWAY, now);
            firsts.extend(pairs_of(&first).into_iter().map(|pair| pair.1));
            let frame = at_call(n, LeaseId(1), WireModes::NONE, None);
            let (asked, given) = ((GATEWAY, now), (&unread, None));
            let call = served_call(&store, &objects, frame, asked, (p, None), given);
            calls.extend(pairs_of(&call).into_iter().map(|pair| pair.1));
        }
        ([firsts, calls], same, other)
    };
    let (sides, same, _) = named(None);
    assert_eq!(sides, [BTreeSet::from([same]), BTreeSet::from([same])]);
    let (sides, same, other) = named(Some("off"));
    let both = BTreeSet::from([same, other]);
    assert_eq!(sides, [both.clone(), both]);
}

/// The `random` picker, keeping each word its engine tells it of a lease's
/// side: the call, the side, and the worker the call names.
#[derive(Debug)]
struct SideWords(Arc<Mutex<Vec<SideWord>>>);

/// One call of a stage about a lease's side: the worker is the one the
/// lease books at an issue or at an end, and the target of a dispatch.
type SideWord = (&'static str, Side, WorkerId);

impl Lifecycle for SideWords {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(("issued", lease.side, lease.booked));
    }

    fn on_dispatched(&self, _: &Lease, target: &Target, side: Side, _: u32) {
        self.0.lock().push(("dispatched", side, target.id));
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        self.0.lock().push(("completed", lease.side, lease.booked));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.0.lock().push(("released", lease.side, lease.booked));
    }
}

impl Picker for SideWords {
    fn name(&self) -> &'static str {
        "side-words"
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

/// A `Dispatched` frame at an index the pick holds no target at names no
/// worker. On a decode side that stands chosen it moves nothing, and no
/// stage hears of it: no issue, no dispatch and, where the gateway then
/// reports the side's end or gives the lease up, no end of that side. A
/// decode call behind the frame is decided as behind no frame, and its
/// pick is issued under the pair's lease. A frame behind it that names a
/// target is told as the side's issue and then its dispatch, once each.
#[test]
fn a_dispatched_frame_at_an_index_its_pick_does_not_hold_leaves_a_chosen_side_as_it_stood() {
    let store = Arc::new(MemoryStore::new());
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    let stage = move || -> Box<dyn Picker> { Box::new(SideWords(Arc::clone(&kept))) };
    catalog.picker("side-words", stage).expect("a free name");
    store
        .apply(pair_plan(
            &catalog,
            r#"["healthy"]"#,
            ["random", "side-words"],
        ))
        .expect("above the unbound start");
    let p = side_member(&store, "p0", Role::Prefill, None);
    let d = side_member(&store, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let objects = holder(&store);
    // The word to the engine of a frame applied to a lease, as a replica
    // gives it at the start of its deciders, and a session's hand at the
    // files: no decider is at work here, so the hand applies each frame.
    lifecycle::tells(&store, &objects);
    let hand = Hand::new(GATEWAY);
    let now = Instant::now();
    // A `Dispatched` frame of the decode side of `lease`, at `index`.
    let sent = |lease: LeaseId, index: u32| {
        let frame = proto::Dispatched {
            lease_id: lease.0,
            target_index: index,
            side: proto::Side::Decode.into(),
            since_decision_micros: 0,
        };
        lifecycle::dispatched(&store, &hand, &frame, now).expect("the frame reads");
    };
    // What the decode side's stage heard since the last read.
    let heard = || mem::take(&mut *told.lock());
    // What the leases on file book on the prefill and on the decode.
    let booked = || {
        let load = |name: &str| {
            let url = format!("http://{name}:8000");
            let object = objects.registry().get_by_url(&url).expect("its object");
            object.load()
        };
        [load("p0"), load("d0")]
    };
    let side_on_file = |lease: LeaseId| {
        let sides = LeasesStore::sides(&*store, GATEWAY, lease).expect("on file");
        sides.decode.map(|side| (side.state, side.target))
    };
    let chosen = Some((LeaseState::Chosen, d));

    // The pick holds one target. A frame past it, at the next index and at
    // the last a byte names, moves nothing and reaches no stage.
    let (_, first) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(heard(), [("issued", Side::Prefill, p)]);
    for index in [1, 255] {
        sent(first.id, index);
    }
    assert_eq!((side_on_file(first.id), booked()), (chosen, [1, 0]));
    assert!(
        heard().is_empty(),
        "no stage hears a frame that names no worker"
    );
    // A decode call behind it finds the side chosen: its pick is issued
    // under the pair's lease, and the stages hear that issue.
    let call = at_call(2, first.id, WireModes::NONE, None);
    let (decision, issued) = leased_call(&store, &objects, call, now);
    assert_eq!(pairs_of(&decision), [(p, d)]);
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(
        (side.id, side.side, side.booked),
        (first.id, Side::Decode, d)
    );
    assert_eq!(heard(), [("issued", Side::Decode, d)]);
    let issued = Some((LeaseState::ISSUED, d));
    assert_eq!((side_on_file(first.id), booked()), (issued, [1, 1]));

    // A second pair, whose decode is reported complete behind such a
    // frame and whose lease is then given up whole: the prefill's release
    // is told, and no end of a side no stage heard issued.
    let (_, second) = held_pair_at(&store, &objects, (3, 0), now);
    told.lock().clear();
    sent(second.id, 1);
    let ended = proto::Completed {
        lease_id: second.id.0,
        side: proto::Side::Decode.into(),
        served: d.0,
        status: 200,
        class: proto::completed::Class::Ok.into(),
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    lifecycle::completed(&store, &hand, &ended, now).expect("the frame reads");
    assert!(heard().is_empty(), "a side no stage heard issued");
    let complete = Some((LeaseState::Completed, d));
    assert_eq!((side_on_file(second.id), booked()), (complete, [2, 1]));
    let gone = proto::Released {
        what: Some(proto::released::What::LeaseId(second.id.0)),
        reason: proto::released::Reason::ClientGone.into(),
        side: proto::Side::Unspecified.into(),
    };
    lifecycle::released(&store, &hand, &gone).expect("the frame reads");
    assert_eq!(heard(), [("released", Side::Prefill, p)]);
    assert_eq!(booked(), [1, 1]);

    // A third: a frame behind it that names the pick's target is the
    // side's issue and then its dispatch, and the same frame again tells
    // nothing.
    let (_, third) = held_pair_at(&store, &objects, (4, 0), now);
    told.lock().clear();
    for index in [1, 0] {
        sent(third.id, index);
    }
    let words = [("issued", Side::Decode, d), ("dispatched", Side::Decode, d)];
    assert_eq!(heard(), words);
    sent(third.id, 0);
    assert!(heard().is_empty(), "the same frame again");
    let sent_to = Some((LeaseState::Dispatched { at: now }, d));
    assert_eq!((side_on_file(third.id), booked()), (sent_to, [2, 2]));
}
