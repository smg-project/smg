//! The cases of the decode side of a pair's lease: the decode call, its
//! repeat, the give-up of its answer, and a decode frame that names no
//! lease.

use super::*;

/// A scorer that ranks every worker alike, keeping what the engine
/// tells it of each side of a lease.
#[derive(Debug)]
struct HeardSides(Arc<Mutex<SideCalls>>);

impl Lifecycle for HeardSides {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push((lease.side, lease.booked, None));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push((lease.side, lease.booked, Some(reason)));
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued, EventKind::Released])
    }
}

impl Scorer for HeardSides {
    fn name(&self) -> &'static str {
        "heard-sides"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// The bed of the decode call's cases: a pair plan whose sides take the
/// worker with the fewest requests in flight, over the fleet's prefill
/// and its decode beside one more decode, behind one decider. With the
/// bed: what the engine told a stage of the decode side's profile, and
/// the prefill.
fn later_bed() -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
    later_bed_of(1)
}

/// [`later_bed`] behind `deciders` deciders.
fn later_bed_of(deciders: usize) -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
    let told = Arc::new(Mutex::new(Vec::new()));
    let bed = bed(Some(later_spec(1, &told)), true, (deciders, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    (bed, told, prefill)
}

/// The document of [`later_bed`] as `generation`, whose decode side
/// keeps what its engine tells it in `told`.
fn later_spec(generation: u64, told: &Arc<Mutex<SideCalls>>) -> AppliedSpec {
    later_spec_with(generation, told, |_| {})
}

/// [`later_spec`] with `edit` made to its document ahead of its check.
fn later_spec_with(
    generation: u64,
    told: &Arc<Mutex<SideCalls>>,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    later_spec_in(StageRegistry::builtin(), generation, told, edit)
}

/// [`later_spec_with`] over `catalog`: this build's stages and a case's
/// own beside them.
fn later_spec_in(
    mut catalog: StageRegistry,
    generation: u64,
    told: &Arc<Mutex<SideCalls>>,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    hearing_sides(&mut catalog, told);
    later_spec_over(&catalog, generation, edit)
}

/// Adds to `catalog` the stage of [`later_bed`]'s decode side, which
/// keeps what its engine tells it in `told`.
fn hearing_sides(catalog: &mut StageRegistry, told: &Arc<Mutex<SideCalls>>) {
    let kept = Arc::clone(told);
    catalog
        .scorer("heard-sides", move || {
            Box::new(HeardSides(Arc::clone(&kept)))
        })
        .expect("a free name");
}

/// The document of [`later_bed`] as `generation`, validated over
/// `catalog`, which holds its decode side's stage ([`hearing_sides`]).
/// Two documents validated over one catalog are a controller's: the
/// later one keeps the stages of a profile both state alike.
fn later_spec_over(
    catalog: &StageRegistry,
    generation: u64,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    let mut document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {
                "side": { "pick": "count-first" },
                "told": { "score": ["heard-sides"], "pick": "count-first" }
              },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "side", "decode": "told" } },
                { "profile": "side" }
              ]
            }"#,
    )
    .expect("a document of this build");
    document.generation = generation;
    edit(&mut document);
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, catalog, &[])
        .expect("the catalog holds the stage")
}

/// A pair asked with its decode later, and its prefill dispatched: the
/// lease's id and the first pick's two decodes, the chosen one first.
/// A second pair asked whole then books the chosen decode, so the
/// fewest in flight sends a call to the other: that pair's lease comes
/// last.
async fn held(bed: &mut Bed) -> (u64, [WorkerId; 2], u64) {
    let answer = bed.ask(pair_frame(1, true)).await;
    let (targets, sides) = pair_of(&answer);
    let id = sides.first().expect("the prefill's lease").0;
    let decodes = [0, 1].map(|at| WorkerId(targets[at].1.expect("a pair's target")));
    // One seed, the same counts on the decodes: the whole pair takes
    // the same.
    let whole = bed.ask(pair_frame(2, false)).await;
    let (targets, sides) = pair_of(&whole);
    assert_eq!(targets[0].1, Some(decodes[0].0));
    let whole = sides.first().expect("the whole pair's lease").0;
    let at = bed.script.taken;
    let prefill = sent(Side::Prefill);
    assert!(bed.store.dispatched(GATEWAY, LeaseId(id), prefill, at));
    (id, decodes, whole)
}

/// Files one issued lease of [`GATEWAY`] on `target`, under `id`.
fn file(store: &MemoryStore, id: u64, target: WorkerId, now: Instant) {
    let leases: &dyn LeasesStore = store;
    let filed = leases.issue(crate::store::LeaseRecord {
        lease: Lease {
            id: LeaseId(1 << 63 | id),
            ..Lease::default()
        },
        gateway: GATEWAY,
        request: id,
        target,
        picked: crate::store::PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: now + DISPATCH_TTL,
        state: LeaseState::ISSUED,
    });
    assert_eq!(filed, Ok(()));
}

/// The decode call answers a decode pick under the pair's own lease:
/// decided on the counts of its instant, its targets name the held
/// prefill, and the new pick's decode is booked in the first's place.
#[tokio::test]
async fn a_served_decode_call_answers_a_decode_pick_under_the_same_lease() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let heard = told.lock().len();
    let answer = bed.ask(call(1, id)).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    // The chosen decode took a lease since: the call picks the other.
    assert_eq!(targets[0], (prefill.0, Some(other.0)));
    assert!(targets.iter().all(|target| target.0 == prefill.0));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(told.lock()[heard..], [(Side::Decode, other, None)]);
    assert_eq!(asked_again(&bed), [0, 0]);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
}

/// A decode call's answer says no word of a term's mark. Its lease's
/// id is its pair's, minted under the term before and at that term's
/// mark: the term that answers the call minted no id yet, and a word
/// there would hand a fresh term over.
#[tokio::test]
async fn a_decode_call_on_a_lease_at_an_earlier_terms_mark_says_no_word() {
    let (mut bed, _, _) = later_bed();
    let (words, go) = heard(&bed);
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    // The pair's lease takes the id at its term's mark, and its answer
    // says that term's word.
    bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
    let (id, _, _) = held(&mut bed).await;
    assert_eq!(id, 1 << 32 | u64::from(TERM_MARK));
    words.recv_timeout(PATIENCE).expect("the first term's word");
    // The bed's candidate, alone over its lease, leaves and takes again
    // at one instant, as its election would. The pair's lease stays on
    // its file under the new term.
    let taken = bed.script.taken;
    let left = bed.elector.step_down(None, taken);
    assert!(left.expect("a planned hand-over"));
    let took = bed.elector.tick(taken).expect("the take");
    assert_eq!(took, TickOutcome::Acquired { term: 2 });
    bed.store.set_term(2);
    let answer = bed.ask(call(3, id)).await;
    assert_eq!(answer.term, 2);
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    // The answer after the call's shows its decider said what it had
    // to say: nothing. That answer's own id is the new term's first.
    let next = bed.ask(pair_frame(4, false)).await;
    assert_eq!(sides_of(&next)[0].0, 2 << 32 | 1);
    assert!(words.try_recv().is_err(), "a call mints no id");
}

/// A decode call's frame is counted as a pair's: the decodes its
/// gateway's wires take, and a frame with more block hashes than its
/// document reads. The fleet's decodes state no wire, so a gateway
/// that states one takes both, and the call is shed under the wires'
/// stage with its side still chosen.
#[tokio::test]
async fn a_decode_calls_frame_counts_the_decodes_its_wires_take_and_its_cut_hashes() {
    let (mut bed, _, _) = later_bed();
    let (id, [chosen, _], _) = held(&mut bed).await;
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (0, 0));
    let mut wired = constrained(call(1, id), 0b010, None);
    let facts = wired.facts.as_mut().expect("the case's frame has facts");
    facts.block_hashes = vec![1, 2];
    let answer = bed.ask(wired).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (2, 1));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
}

/// A decode call can pick a decode the store's world does not list yet,
/// as a single request's decision can pick such a worker: the worker
/// objects follow a registration as its write lands. The side's issue
/// books the worker's count all the same, by the slot the registration
/// filed.
#[tokio::test]
async fn a_decode_calls_pick_inside_its_workers_registration_window_books_its_count() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    // One more decode, written ready and never published: the build
    // that lists it is still owed. A lease filed on the other
    // decode leaves the new one alone with nothing in flight.
    let key = String::from("http://late:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role: Role::Decode,
        models: vec![MODEL.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let late = rewritten(&bed, facts);
    let row = WorkersStore::get(&*bed.store, late).expect("the row");
    bed.reads
        .objects
        .connect_row(&bed.store, late, row.revision);
    let listed = bed.store.snapshot().booked_of(late);
    assert_eq!(listed, None, "no world lists the worker yet");
    file(&bed.store, 1, other, bed.script.taken);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(pair_of(&answer).0[0], (prefill.0, Some(late.0)));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, late));
    assert_eq!(booked(&bed, [prefill, chosen, other, late]), [2, 1, 1, 1]);
}

/// What [`Ends`] ends: the prefill side of a lease of [`GATEWAY`]'s on
/// file in a store, once.
type Armed = Arc<Mutex<Option<(Weak<MemoryStore>, LeaseId)>>>;

/// A filter that keeps every worker and, as a decision opens it, ends
/// what its case armed it with: what another hand does to a lease
/// between a decode call's first test of it and the call's issue.
#[derive(Debug)]
struct Ends(Armed);

impl Lifecycle for Ends {}

impl DecisionScope for Ends {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        let Some((store, lease)) = self.0.lock().take() else {
            return;
        };
        let store = store.upgrade().expect("the case's store");
        let (prefill, gone) = (Some(Side::Prefill), ReleaseReason::ClientGone);
        assert!(store.released(GATEWAY, lease, prefill, gone));
    }

    fn close(&self) {}
}

impl Filter for Ends {
    fn name(&self) -> &'static str {
        "ends"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

/// A lease that moves between a decode call's first test and its issue
/// refuses the issue, and the call leaves as the `Busy` of that
/// refusal: here the lease's prefill ends while the call's stages run,
/// and takes the lease off the file with its chosen side. Nothing is
/// issued, no stage hears an issue, and no count is booked.
#[tokio::test]
async fn a_lease_that_ends_under_a_decode_calls_decision_refuses_its_issue_by_name() {
    let armed = Armed::default();
    let cell = Arc::clone(&armed);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("ends", move || Box::new(Ends(Arc::clone(&cell))))
        .expect("a free name");
    let told = Arc::new(Mutex::new(Vec::new()));
    let spec = later_spec_in(catalog, 1, &told, |document| {
        let profile = document.profiles.get_mut("told");
        let profile = profile.expect("the decode side's profile");
        profile.filter.push("ends".into());
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    let (id, [chosen, other], _) = held(&mut bed).await;
    *armed.lock() = Some((Arc::downgrade(&bed.store), LeaseId(id)));
    let heard = told.lock().len();
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::DecodeNoLease);
    assert!(armed.lock().is_none(), "the call's stages ran");
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), None);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [1, 1, 0]);
    assert_eq!(told.lock().len(), heard, "no stage heard an issue");
}

/// A decode frame that names no lease has no decision under a plan that
/// pairs the decode phase: `Busy` with no reason, counted with the
/// sides, and nothing of a pair's lease on file moves.
#[tokio::test]
async fn a_decode_frame_that_names_no_lease_has_no_decision_and_is_counted() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let (filed, heard) = (bed.store.sides(GATEWAY, LeaseId(id)), told.lock().len());
    let alone = phased(frame(1, 77), proto::Phase::Decode);
    assert_eq!(busy(&bed.ask(alone).await), Reason::Unspecified);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(told.lock().len(), heard);
}

/// A served decode call's kept record names the pair binding the call
/// decided under, whatever phase its frame states. A binding of the
/// decode phase stands ahead of the pair's here: it decides a decode
/// asked alone, and no pair's.
#[tokio::test]
async fn a_served_decode_calls_record_names_its_pairs_binding() {
    use crate::{
        document::spec::{BindingSpec, Match},
        engine::view::Phase,
    };

    let (mut bed, told, _) = later_bed();
    let sampled = later_spec_with(2, &told, |document| {
        let ahead = BindingSpec {
            matcher: Match {
                phase: vec![Phase::Decode],
                ..Match::default()
            },
            profile: Some("side".into()),
            ..BindingSpec::default()
        };
        document.bindings.insert(0, ahead);
        document.explain.sample = 1;
    });
    bed.store.apply(sampled).expect("a later generation");
    let (id, _, _) = held(&mut bed).await;
    // The call is asked under an id of its own: the pair's two first
    // frames left their records under the same binding, and a record
    // looked up by either's id would be that frame's.
    let no_record = bed.deciders.explain(1, GATEWAY, &[3]);
    assert!(no_record.is_none(), "no frame was asked under the id yet");
    let answer = bed.ask(phased(call(3, id), proto::Phase::Decode)).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    let record = bed.deciders.explain(answer.term, GATEWAY, &[3]);
    // The pair's binding is the document's second.
    assert_eq!(record.expect("the sampled call").binding, Some(1));
}

/// Each refusal of the call leaves as a `Busy` of its own reason,
/// never as a decision, and moves nothing: no lease under the id, a
/// pair filed whole, a prefill not yet sent, and a side a call issued.
#[tokio::test]
async fn each_refusal_of_the_call_is_a_busy_of_its_own_reason_and_moves_nothing() {
    let (mut bed, told, prefill) = later_bed();
    let lease = |answer: &proto::DecideAnswer| sides_of(answer)[0].0;
    let later = lease(&bed.ask(pair_frame(1, true)).await);
    let whole = lease(&bed.ask(pair_frame(2, false)).await);
    let store = Arc::clone(&bed.store);
    let decodes = ["d0", "d1"].map(|name| {
        let row = WorkersStore::by_key(&*store, &format!("http://{name}:8000"));
        row.expect("a decode of the fleet").id
    });
    let file = |bed: &Bed| {
        let sides = [later, whole].map(|id| store.sides(GATEWAY, LeaseId(id)));
        let counts = booked(bed, [prefill, decodes[0], decodes[1]]);
        (sides, counts, told.lock().len())
    };
    let rows = [
        (later + 100, Reason::DecodeNoLease),
        (whole, Reason::DecodeNotChosen),
        (later, Reason::DecodePrefillNotHeld),
    ];
    for (id, reason) in rows {
        let before = file(&bed);
        assert_eq!(busy(&bed.ask(call(1, id)).await), reason);
        assert_eq!(file(&bed), before, "{reason:?}");
    }
    // The fourth: a side a first call issued stands.
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(later), sent(Side::Prefill), at));
    let issued = bed.ask(call(1, later)).await;
    assert!(matches!(outcome(&issued), Outcome::Pick(_)));
    let before = file(&bed);
    assert_eq!(busy(&bed.ask(call(1, later)).await), Reason::DecodeStands);
    assert_eq!(file(&bed), before);
    // Each is counted at its own reason, once.
    let counted = |reason: Reason| bed.counts.busy[reason as usize].load(Ordering::Relaxed);
    let reasons = [
        Reason::DecodeNoLease,
        Reason::DecodeNotChosen,
        Reason::DecodePrefillNotHeld,
        Reason::DecodeStands,
    ];
    assert_eq!(reasons.map(counted), [1; 4]);
}

/// A call that sheds is a decision, as any shed is: nothing is issued
/// and the side stays chosen with the first pick's targets. A dispatch
/// then books the first pick's decode.
#[tokio::test]
async fn a_shed_at_the_call_is_a_decision_and_leaves_the_side_chosen() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    let answer = bed.ask(constrained(call(1, id), 0, Some(0))).await;
    let Outcome::Shed(shed) = outcome(&answer) else {
        panic!("a shed: {answer:?}");
    };
    assert_eq!(shed.emptied_by, "subset-hint");
    assert!(sides_of(&answer).is_empty());
    assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(told.lock().len(), heard);
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode issued under a term that ended returns to chosen. The
/// answer leaves as `Busy`, the lease stands as the call found it with
/// the first pick's targets, the new pick's booking is freed and the
/// engine hears that issue's release once. A dispatch then books the
/// first pick's decode.
#[tokio::test]
async fn a_decode_issued_under_a_term_that_ended_returns_to_chosen() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    // The call's first read finds the fence open; its read at the
    // hand-over is past the renew deadline.
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = Some(ReleaseReason::Abandoned);
    let issue = [
        (Side::Decode, other, None),
        (Side::Decode, other, abandoned),
    ];
    assert_eq!(told.lock()[heard..], issue);
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode call whose decision's last read of the clock shows its
/// budget gone gives its issue back under the late reason, and is
/// counted as a decision that outlasted its budget.
#[tokio::test]
async fn a_decode_call_that_outlasts_its_budget_gives_its_issue_back() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let before = bed.store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    let steps = [Step::At(Duration::ZERO), Step::At(BUDGET)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let late = Some(ReleaseReason::LateDecision);
    let issue = [(Side::Decode, other, None), (Side::Decode, other, late)];
    assert_eq!(told.lock()[heard..], issue);
}

/// A repeat of the call frees the lost issue and issues the side anew
/// under the same lease id. The word takes back the first call's
/// issue, ahead of the repeat's own decision, and the engine hears
/// that issue's release as abandoned.
#[tokio::test]
async fn a_repeat_of_the_call_frees_the_lost_issue_and_issues_the_side_anew() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let first = bed.ask(call(1, id)).await;
    assert_eq!(pair_of(&first).0[0].1, Some(other.0));
    // The loads move while the answer is lost: two more requests on
    // the decode the first call issued.
    for extra in [1, 2] {
        file(&store, extra, other, bed.script.taken);
    }
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 3]);
    let heard = told.lock().len();
    let repeat = bed.ask(again(call(1, id))).await;
    let (targets, sides) = pair_of(&repeat);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    assert_eq!(targets[0], (prefill.0, Some(chosen.0)));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, chosen));
    // The first call's decode gave its booking back, and the repeat's
    // took one.
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 2]);
    let abandoned = Some(ReleaseReason::Abandoned);
    let words = [
        (Side::Decode, other, abandoned),
        (Side::Decode, chosen, None),
    ];
    assert_eq!(told.lock()[heard..], words);
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A repeat that sheds leaves the side chosen, as its first decision
/// filed it: the word took the lost issue back ahead of the repeat's
/// decision, whatever its answer. The gateway then sends the decode its
/// first pick named, and that dispatch books the side.
#[tokio::test]
async fn a_repeat_that_sheds_leaves_the_side_chosen_for_the_first_picks_dispatch() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    bed.ask(call(1, id)).await;
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_ne!(store.sides(GATEWAY, lease), filed);
    let repeat = again(constrained(call(1, id), 0, Some(0)));
    let answer = bed.ask(repeat).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    // The lease stands as its first decision filed it, both its parts:
    // the first pick's targets and the lease's dispatch deadline.
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = Some(ReleaseReason::Abandoned);
    assert_eq!(told.lock().last(), Some(&(Side::Decode, other, abandoned)));
    assert_eq!(asked_again(&bed), [0, 1]);
    // The first pick's decode goes out: its dispatch books the side.
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, lease, sent(Side::Decode), at));
    let sent_to = (LeaseState::Dispatched { at }, chosen);
    assert_eq!(decode_on_file(&bed, id), sent_to);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A repeat on a side still chosen is a first call: no call issued the
/// side, and the word ends nothing.
#[tokio::test]
async fn a_repeat_on_a_chosen_side_is_a_first_call() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let heard = told.lock().len();
    let answer = bed.ask(again(call(1, id))).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    assert_eq!(targets[0].1, Some(other.0));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(told.lock()[heard..], [(Side::Decode, other, None)]);
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A repeat on a side its gateway named a dispatch for is refused as
/// standing: the word ends no side that was sent.
#[tokio::test]
async fn a_repeat_on_a_dispatched_side_is_refused_as_standing() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    bed.ask(call(1, id)).await;
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    let file = || (store.sides(GATEWAY, LeaseId(id)), told.lock().len());
    let before = file();
    let answer = bed.ask(again(call(1, id))).await;
    assert_eq!(busy(&answer), Reason::DecodeStands);
    assert_eq!(file(), before);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A decode frame runs no search by its id's fingerprint, with the
/// word or without: another request of the gateway under the same id
/// keeps the lease it waits under.
#[tokio::test]
async fn a_decode_frame_runs_no_search_by_its_ids_fingerprint() {
    let (mut bed, _, _) = later_bed();
    let (id, _, _) = held(&mut bed).await;
    // A single of the gateway under the id the call's frame carries:
    // its lease waits for a dispatch.
    let single = bed.ask(frame(9, 77)).await;
    let waits = LeaseId(lease_of(&single).id);
    let answer = bed.ask(again(call(9, id))).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    let leases: &dyn LeasesStore = bed.store.as_ref();
    assert!(leases.get(GATEWAY, waits).is_some(), "the single's lease");
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// The issue that stands of `lease`'s decode side, taken back by the
/// case's own hand once the bed's decider left its work: a thread acts
/// at a file where nobody is at work, and beside nobody.
async fn taken_back_by_hand(store: &MemoryStore, lease: LeaseId) {
    let abandoned = ReleaseReason::Abandoned;
    settled(|| store.take_back_decode(GATEWAY, lease, abandoned)).await;
}

/// One decode call's order by the case's own hand, at the bed's first
/// instant: the issue its decider would undo.
fn issued_by_hand(bed: &Bed, frame: proto::DecideRequest) -> DecodeIssue {
    let request = checked(frame);
    let now = bed.script.taken;
    let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
    let (mut frames, part, mut caught) = (Default::default(), Part::default(), false);
    let lease = request.decode_of().expect("a decode call's frame");
    let Reads { store, objects, .. } = &bed.reads;
    let loads = Arc::new(LoadSnapshot::default());
    let decided = store.deciding(|world| {
        let theirs = (&mut frames, &loads, &part);
        decode_call(store, world, objects, &taken, lease, theirs, &mut caught)
    });
    match decided {
        Ok((_, Ok(Some(_)), FiledBeside::Issue(issue))) => issue,
        _ => panic!("a decode pick and its issue"),
    }
}

/// A decode call's answer that its stream does not take gives its one
/// issue back, and no more of the lease: the prefill that runs under
/// the same id keeps its booking. The side stands chosen with the
/// first pick's targets, the engine hears the issue's release as
/// abandoned, and the first pick's dispatch then books the side.
#[tokio::test]
async fn a_decode_calls_answer_its_stream_does_not_take_gives_its_issue_back() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    // A stream that ended takes no answer.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(call(1, id)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    // One decider decides a model's requests in order: it is past the
    // call, and past its answer's give-up, when it answers the next.
    bed.ask(frame(9, 9)).await;
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = (Side::Decode, other, Some(ReleaseReason::Abandoned));
    assert!(told.lock().contains(&abandoned), "{:?}", told.lock());
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, lease, sent(Side::Decode), at));
    let sent_to = (LeaseState::Dispatched { at }, chosen);
    assert_eq!(decode_on_file(&bed, id), sent_to);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode side's answer of `lease` as its stream's queue holds it,
/// handed over with `beside` at `at`.
fn queued(lease: LeaseId, beside: wire::Beside, at: Instant) -> wire::Leased {
    wire::Leased {
        lease: Lease {
            id: lease,
            side: Side::Decode,
            ..Lease::default()
        },
        beside,
        left: DISPATCH_TTL,
        due: at + DISPATCH_TTL,
    }
}

/// A decode call's answer given up while a decider is at work at the
/// lease's file moves nothing and tells nothing on the thread that
/// gives it up: the word waits in the file's ring. The decider applies
/// it at its next take, ahead of its own decision: the issue goes back
/// and the engine hears its release there, so no stage hears a word of
/// a lease beside its decider at work.
#[tokio::test]
async fn an_answer_given_up_beside_a_decider_at_work_is_left_to_that_decider() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    let issue = issued_by_hand(&bed, call(1, id));
    let answer = queued(lease, wire::Beside::Issue(issue), bed.script.taken);
    let heard = told.lock().len();
    // The next decision is held at its first read of the clock: its
    // decider has the job at hand, at the lease's file.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(9, 9)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    let late = ReleaseReason::LateDecision;
    // A take-back in place is refused beside the decider at work.
    assert!(!store.take_back_decode(GATEWAY, lease, late));
    answer.given_up(&store, &hand, late);
    assert_eq!(told.lock().len(), heard, "left to the decider at work");
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let released = (Side::Decode, other, Some(late));
    assert_eq!(told.lock().get(heard), Some(&released));
    store.hand_back(hand);
}

/// A decode call is decided at its lease's file, whatever model it
/// names: its job goes to the decider that issued the lease, not to
/// the decider of the model the call names. A repeat under such a
/// name takes the call's issue back there, and the first pick's decode
/// stands chosen where the named model has no worker.
#[tokio::test]
async fn a_decode_call_is_decided_by_its_leases_decider_whatever_model_it_names() {
    let (mut bed, _, prefill) = later_bed_of(4);
    let (id, [chosen, other], _) = held(&mut bed).await;
    let home = format!("decide-{}", bed.deciders.of(MODEL));
    let elsewhere = (0..64)
        .map(|index| format!("model-{index}"))
        .find(|model| bed.deciders.of(model) != bed.deciders.of(MODEL))
        .expect("some model of sixty-four has another decider");
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
    bed.script.readers.lock().clear();
    let mut repeat = again(call(1, id));
    let facts = repeat.facts.as_mut().expect("the case's frame has facts");
    facts.model = elsewhere;
    bed.ask(repeat).await;
    let readers = mem::take(&mut *bed.script.readers.lock());
    assert!(
        !readers.is_empty() && readers.iter().all(|reader| *reader == home),
        "{readers:?}"
    );
    assert_eq!(asked_again(&bed), [0, 1]);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// What an answer that does not leave gives up, by what its lease is
/// handed over with. A decode call's answer takes back its own issue
/// alone, by its count: behind a repeat's issue it moves nothing. Any
/// other answer's lease is released whole, both its sides.
#[tokio::test]
async fn an_answer_given_up_takes_back_its_own_issue_or_releases_its_lease_whole() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    let file = |bed: &Bed| {
        let counts = booked(bed, [prefill, chosen, other]);
        (store.sides(GATEWAY, lease), counts)
    };
    let at = bed.script.taken;
    let answer = |beside| queued(lease, beside, at);
    let hand = Hand::new(GATEWAY);
    let late = ReleaseReason::LateDecision;
    // The first call's answer, given up behind a repeat's issue.
    let first = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
    taken_back_by_hand(&store, lease).await;
    let repeat = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
    let issued = file(&bed);
    assert_eq!(issued.1, [2, 1, 1]);
    first.given_up(&store, &hand, late);
    assert_eq!(file(&bed), issued);
    // The repeat's answer, given up: its own issue goes back. Its word
    // is applied by the hand, or by a decider that leaves its work.
    repeat.given_up(&store, &hand, late);
    settled(|| file(&bed) == (filed, [2, 1, 0])).await;
    assert_eq!(told.lock().last(), Some(&(Side::Decode, other, Some(late))));
    // A pair's first answer, given up: the lease leaves whole.
    let whole = answer(wire::Beside::Decode(view::Session::NONE));
    whole.given_up(&store, &hand, late);
    store.hand_back(hand);
    assert_eq!(file(&bed), (None, [1, 1, 0]));
}

/// A give-up that names a chosen decode side ends it: a lost call that
/// runs late behind the give-up is refused as standing, with the word
/// or without, and books nobody.
#[tokio::test]
async fn a_give_up_that_names_a_chosen_side_ends_it() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let abandoned = ReleaseReason::Abandoned;
    assert!(store.released(GATEWAY, LeaseId(id), Some(Side::Decode), abandoned));
    let ended = (LeaseState::Released(abandoned), chosen);
    assert_eq!(decode_on_file(&bed, id), ended);
    for late in [call(1, id), again(call(1, id))] {
        assert_eq!(busy(&bed.ask(late).await), Reason::DecodeStands);
    }
    assert_eq!(decode_on_file(&bed, id), ended);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// A decode call whose model a reload bound to no pair plan is a
/// side's request still: it is not decided as a single. Its decision
/// sheds, nothing is issued and the side stays chosen.
#[tokio::test]
async fn a_decode_call_behind_a_reload_to_no_pair_plan_is_no_single() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let reloaded = store.apply(preset("random", 2));
    reloaded.expect("a later generation");
    let live = LeasesStore::live(&*store, GATEWAY);
    let answer = bed.ask(call(1, id)).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    assert!(sides_of(&answer).is_empty());
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(LeasesStore::live(&*store, GATEWAY), live);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// A call's issue behind a reload is told to no engine, and neither is
/// its take-back. The engine of the call's instant heard no issue of
/// the lease, whose generation is the first decision's: it hears no
/// issue of the side and no release of it.
#[tokio::test]
async fn a_decode_issued_and_taken_back_behind_a_reload_is_told_to_no_engine() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = told.lock().len();
    let reloaded = Arc::new(Mutex::new(Vec::new()));
    let applied = store.apply(later_spec(2, &reloaded));
    applied.expect("a later generation");
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(*reloaded.lock(), []);
    assert_eq!(told.lock().len(), before);
}

/// A call's issue behind a reload that states the decode side's
/// profile as it stood is told to that profile's stage, which the
/// reload kept, and so is its take-back: the stage stood when the lease
/// was issued. Both documents are validated over one catalog, as a
/// controller validates its documents.
#[tokio::test]
async fn a_decode_issued_and_taken_back_behind_a_reload_is_told_to_the_stage_it_kept() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    hearing_sides(&mut catalog, &told);
    let first = later_spec_over(&catalog, 1, |_| {});
    let mut bed = bed(Some(first), true, (1, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = told.lock().len();
    let applied = store.apply(later_spec_over(&catalog, 2, |_| {}));
    applied.expect("a later generation");
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    // The side's issue on the decode the call picked, then its release.
    let heard = told.lock()[before..].to_vec();
    let [(Side::Decode, issued, None), (Side::Decode, released, Some(_))] = heard[..] else {
        panic!("the stage kept heard {heard:?}");
    };
    assert_eq!((issued, released), (other, other));
}
