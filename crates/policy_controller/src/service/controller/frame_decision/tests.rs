//! The cases of the frame decision: rules a single request's decision
//! holds on the served path. Each asks [`decide`] as a deciding thread asks
//! it, in the world of one load of a store, over the worker objects the
//! store's holder lays. The last asks the engine's entry that [`decide`]
//! asks, over the same objects, for a request whose host states what the
//! workers hold of its prompt.

use std::{
    collections::BTreeSet,
    mem,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{decide, FrameBuffers};
use crate::{
    document::spec::{self, AppliedSpec, Assignment, PolicySpec},
    engine::{
        decision::{Decision, Targets},
        decision_engine::{Request, RequestMaker, WorkerLists},
        decision_slot::DecisionSlot,
        framework::{DecisionScope, Filter, Picker, TieBand},
        store_backed_engine::Part,
        view::{
            AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Phase, PoolFacts,
            RequestFacts, Role, WireModes, WorkerId,
        },
    },
    policies::random::Random,
    service::{
        controller::{
            decision_slots::tests::{
                a_frame_under_the_second_files_watch, fleet, frame_for, holder, preset, ready,
                taken_at, BUDGET, GATEWAY, MODEL,
            },
            frame_request::FrameRequest,
            wire::{Checked, Lists},
        },
        proto,
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        AtWork, GatewayId, HealthSink, LeaseRecord, LeaseState, LeasesStore, LookOutcome,
        MemoryStore, PickRecord,
    },
    tracking::lease::{Lease, LeaseId, DISPATCH_TTL},
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
    let decided = store.deciding(|world| decide(world, objects, &taken, &mut frames, &own));
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
    let leases: &dyn LeasesStore = store;
    let filed = leases.issue(LeaseRecord {
        lease: Lease {
            id: LeaseId(1 << 63 | id),
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

/// A key under a sessions rule with an in-flight cap leaves an owner over
/// its thresholds in a decision this service makes. A frame states no
/// count of the key's requests, so the cap could not hand the key on: the
/// veto stands for the key, and it goes to another worker.
#[test]
fn a_served_key_under_a_capped_rule_leaves_an_owner_over_its_thresholds() {
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
    let next = decided(3).expect("a pick beside two workers under their thresholds");
    assert_ne!(
        next, owner,
        "the cap's count is not stated, so the veto stands for the key"
    );
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
