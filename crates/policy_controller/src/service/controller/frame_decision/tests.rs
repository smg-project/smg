//! The cases of the frame decision: rules a decision holds on the served
//! path. A single request's ask [`decide`] as a deciding thread asks it, in
//! the world of one load of a store, over the worker objects the store's
//! holder lays. One asks the engine's entry that [`decide`] asks, over the
//! same objects, for a request whose host states what the workers hold of
//! its prompt, and one asks which documents are given the load snapshot.
//! A pair's ask [`decide_pair`] for a pair at its prefill, and the two
//! that file its leases ask [`issue_pair`] behind it.

use std::{
    collections::BTreeSet,
    mem,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::{one_rank_load, LoadSnapshot};

use super::{decide, decide_pair, hands_reports, issue_pair, FrameBuffers};
use crate::{
    document::spec::{self, AppliedSpec, Assignment, PolicySpec},
    engine::{
        decision::{Decision, ShedReason, Targets, SUBSET_HINT, WIRES},
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
            wire::{Checked, Lists},
        },
        proto,
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        AtWork, DecodeMark, GatewayId, HealthSink, LeaseRecord, LeaseState, LeasesStore,
        LookOutcome, MemoryStore, PickRecord, Reachability, WorkerFacts, WorkerStatus,
        WorkersStore,
    },
    tracking::lease::{Lease, LeaseId, Side, DISPATCH_TTL},
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
    binding.overload = Some(spec::OverloadSpec {
        waiting_requests: Some(1),
        ..spec::OverloadSpec::default()
    });
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
