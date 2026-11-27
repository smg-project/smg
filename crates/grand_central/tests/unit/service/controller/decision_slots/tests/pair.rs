//! The cases of a pair at its prefill, of a prefill or a decode frame that
//! stands alone, and of the frames a pair plan has no decision for.

use super::*;

/// The pair plan's document of the cases, as generation 1: a request of
/// the prefill or the decode phase under one pair binding, each side by
/// `pick`, and a single request under the same profile.
fn pair_plan(catalog: &StageRegistry, pick: &str) -> AppliedSpec {
    let document = format!(
        r#"{{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {{ "p": {{ "pick": "{pick}" }} }},
              "bindings": [
                {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                  "profiles": {{ "prefill": "p", "decode": "p" }} }},
                {{ "profile": "p" }}
              ]
            }}"#
    );
    let document: PolicySpec = serde_json::from_str(&document).expect("a document of this build");
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, catalog, &[])
        .expect("the catalog holds the stage")
}

/// A controller decides a single request, a pair at its prefill and a
/// decode call. Under a pair plan a request of the decode phase that
/// names no lease, and one of the encode phase that does not state
/// `DecideRequest.encode`, have no decision: `Busy` with no reason,
/// counted by its cause, never one side's decision and never a shed,
/// which is a policy's word. Its gateway decides it. A prompt carried
/// as text is never decided as a request with no prefix.
#[tokio::test]
async fn a_request_of_a_decode_or_an_encode_phase_and_a_prompt_as_text_have_no_decision() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let sides = [proto::Phase::Decode, proto::Phase::Encode];
    for (id, phase) in (1..).zip(sides) {
        let answer = bed.ask(phased(frame(id, 77), phase)).await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::Unspecified, &[id][..], 1),
            "{phase:?}"
        );
    }
    // Text the controller is to hash: never decided as a request with
    // no prefix.
    let mut text = frame(4, 77);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    let answer = bed.ask(text).await;
    assert_eq!(
        (busy(&answer), answer.request_id.as_slice(), answer.term),
        (Reason::Unspecified, &[4][..], 1)
    );
    let counted = |count: &AtomicU64| count.load(Ordering::Relaxed);
    let DecideStats {
        busy: busies,
        faults,
        pair_sides,
        texts,
        ..
    } = &*bed.counts;
    assert_eq!(
        (
            counted(faults),
            counted(pair_sides),
            counted(texts),
            bed.counts.hashes_cut()
        ),
        (0, 2, 1, 0)
    );
    // Each of the three is a `Busy` with no reason, and counted as one.
    assert_eq!(busies.each_ref().map(counted), [3, 0, 0, 0, 0, 0, 0, 0, 0]);
    // The same document's single request is decided, and so is its
    // pair at the prefill.
    for phase in [proto::Phase::Single, proto::Phase::Prefill] {
        let answer = bed.ask(phased(frame(5, 77), phase)).await;
        assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{answer:?}");
    }
    assert_eq!(counted(&bed.counts.pair_sides), 2);
}

/// The leases a decision's answer carries, as the wire states each: its
/// id, its side and its session key.
fn leases_of(answer: &proto::DecideAnswer) -> Vec<(u64, proto::Side, u64)> {
    match &answer.answer {
        Some(Answer::Decision(decision)) => {
            let leases = decision.leases.iter();
            let stated = leases.map(|lease| (lease.id, lease.side(), lease.session_key));
            stated.collect()
        }
        other => panic!("a decision: {other:?}"),
    }
}

/// A pair at its prefill is decided once, for both sides, over its
/// model's prefills and decodes, and answered a lease a side under one
/// id, the prefill side's first, each under the session key of a
/// request of its side's phase: one record on file with its decode
/// side issued. A frame that states the later decode is answered the
/// prefill side's lease alone, and its decode side stands chosen.
#[tokio::test]
async fn a_pair_at_its_prefill_is_answered_a_lease_a_side_under_one_id() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let (prefill, decode) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
    let id = |count: u64| 1 << 32 | count;
    let on_file = |bed: &Bed, lease: u64| {
        let store: &dyn LeasesStore = bed.store.as_ref();
        let sides = store.sides(GATEWAY, LeaseId(lease)).expect("on file");
        (sides.record.decode, sides.decode.map(|side| side.state))
    };
    // The case's frame at `phase` under one plain key, and the session
    // key the frame's own reader states of it for that phase.
    let keyed = |phase: proto::Phase| {
        let mut keyed = phased(frame(1, 1), phase);
        let facts = keyed.facts.as_mut().expect("the case's frame has facts");
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        keyed
    };
    let key_at = |phase: proto::Phase| {
        let request = checked(keyed(phase));
        let mut lists = wire::Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session().get()
    };

    let answer = bed.ask(keyed(proto::Phase::Prefill)).await;
    let Outcome::Pick(pick) = outcome(&answer) else {
        panic!("a pick: {answer:?}");
    };
    let first = pick.targets.first().expect("a target");
    let pair = first
        .pair
        .as_ref()
        .expect("a pair's target names its decode");
    assert_eq!(
        (WorkerId(first.id), WorkerId(pair.decode_id)),
        (prefill, decode)
    );
    let keys = [proto::Phase::Prefill, proto::Phase::Decode].map(key_at);
    assert!(
        keys[0] != 0 && keys[1] != 0 && keys[0] != keys[1],
        "{keys:?}"
    );
    let sides = [
        (id(1), proto::Side::Prefill, keys[0]),
        (id(1), proto::Side::Decode, keys[1]),
    ];
    assert_eq!(leases_of(&answer), sides, "the prefill side's first");
    assert_eq!(
        on_file(&bed, id(1)),
        (Some(DecodeMark), Some(LeaseState::ISSUED))
    );
    assert_eq!(bed.store.live(GATEWAY), 1, "one record for both sides");

    let mut later = phased(frame(2, 2), proto::Phase::Prefill);
    later.decode_later = true;
    let answer = bed.ask(later).await;
    assert_eq!(leases_of(&answer), [(id(2), proto::Side::Prefill, 0)]);
    assert_eq!(
        on_file(&bed, id(2)),
        (Some(DecodeMark), Some(LeaseState::Chosen))
    );
}

/// A pair's frame is counted as a single's: the workers its gateway's
/// wires take, on both sides, and a frame with more block hashes than
/// its document reads. The fleet's two sides state no wire, so a
/// gateway that states one takes both workers, and the frame is shed
/// under the wires' stage.
#[tokio::test]
async fn a_pairs_frame_counts_the_workers_its_wires_take_on_both_sides_and_its_cut_hashes() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let mut wired = constrained(phased(frame(1, 1), proto::Phase::Prefill), 0b010, None);
    let facts = wired.facts.as_mut().expect("the case's frame has facts");
    facts.block_hashes = vec![1, 2];
    let answer = bed.ask(wired).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (2, 1));
}

/// A prefill frame, and a decode frame that names no lease, stand alone
/// under a plan that pairs no side. Each is decided over the list of its
/// own phase's role, here the fleet's one worker of that role and none
/// of its three single workers. Its pick is filed a single's lease: one
/// lease of the single side on the wire, one record with no decode side
/// on file, and its worker booked. Neither frame is counted with the
/// sides. An encode frame that does not state `DecideRequest.encode`
/// has no decision under this plan either: `Busy` with no reason,
/// counted, and nothing filed. A model that has no worker of the
/// frame's role is shed with no endpoint, as a single request of a
/// model nobody serves is.
#[tokio::test]
async fn a_prefill_or_a_decode_frame_that_stands_alone_is_decided_over_its_own_phases_list() {
    let mut bed = leading([]);
    let id = |count: u64| 1 << 32 | count;
    let sides = [
        (proto::Phase::Prefill, Role::Prefill),
        (proto::Phase::Decode, Role::Decode),
    ];
    for (n, (phase, role)) in (1..).zip(sides) {
        let worker = side_of(&bed, role);
        let answer = bed.ask(phased(frame(n, 1), phase)).await;
        assert_eq!(picked(&answer), worker, "{phase:?}");
        let lease = id(u64::from(n));
        assert_eq!(sides_of(&answer), [(lease, proto::Side::Single, 0)]);
        let store: &dyn LeasesStore = bed.store.as_ref();
        let filed = store.sides(GATEWAY, LeaseId(lease)).expect("on file");
        let no_decode = filed.record.decode.is_none() && filed.decode.is_none();
        assert_eq!(
            (filed.record.target, no_decode),
            (worker, true),
            "{phase:?}: a single's record"
        );
        assert_eq!(booked(&bed, [worker]), [1], "{phase:?}");
    }
    assert_eq!(bed.store.live(GATEWAY), 2);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    let encode = phased(frame(3, 1), proto::Phase::Encode);
    assert_eq!(busy(&bed.ask(encode).await), Reason::Unspecified);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    // A model served by a single worker alone: no prefill and no decode
    // of it stands in the registry's lists.
    serving(
        &bed.store,
        &bed.reads.objects,
        "s9",
        &["model-of-singles"],
        Role::Single,
        WireModes::NONE,
    );
    let none = declined(proto::ShedReason::NoEndpoints, "none");
    for (id, phase) in (9..).zip([proto::Phase::Prefill, proto::Phase::Decode]) {
        let lone = phased(frame_for("model-of-singles", id, 9), phase);
        assert_eq!(*outcome(&bed.ask(lone).await), none, "{phase:?}");
    }
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 2);
}

/// The decode phase's own lookup says whether a decode frame that names
/// no lease stands alone. Under the pair plan it is a pair's side with
/// no lease to decide behind: `Busy` with no reason, counted with the
/// sides. With a binding of one profile listed ahead of the pair's for
/// the decode phase, the same frame is decided alone over the decodes
/// and filed a single's lease, and the model's prefill frame is still
/// its pair's one request.
#[tokio::test]
async fn a_binding_of_one_profile_ahead_of_the_pairs_decides_a_decode_frame_alone() {
    let ahead = |alone: &str| {
        let document = format!(
            r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "profiles": {{ "p": {{ "pick": "random" }} }},
                  "bindings": [
                    {alone}
                    {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                      "profiles": {{ "prefill": "p", "decode": "p" }} }},
                    {{ "profile": "p" }}
                  ]
                }}"#
        );
        let document: PolicySpec =
            serde_json::from_str(&document).expect("a document of this build");
        applied(&document)
    };
    let decode = phased(frame(1, 1), proto::Phase::Decode);
    let mut paired = bed(Some(ahead("")), true, (2, 8), []);
    assert_eq!(busy(&paired.ask(decode.clone()).await), Reason::Unspecified);
    assert_eq!(paired.counts.pair_sides.load(Ordering::Relaxed), 1);

    let alone = r#"{ "match": { "phase": ["decode"] }, "profile": "p" },"#;
    let mut bed = bed(Some(ahead(alone)), true, (2, 8), []);
    let (prefill, decoder) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
    let answer = bed.ask(decode).await;
    assert_eq!(picked(&answer), decoder);
    let id = |count: u64| 1 << 32 | count;
    assert_eq!(sides_of(&answer), [(id(1), proto::Side::Single, 0)]);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    // The prefill lookup resolves to the pair still: both sides run
    // under the pair binding, whatever binding the decode phase names
    // first.
    let answer = bed.ask(phased(frame(2, 2), proto::Phase::Prefill)).await;
    let pair = [
        (id(2), proto::Side::Prefill, 0),
        (id(2), proto::Side::Decode, 0),
    ];
    assert_eq!(sides_of(&answer), pair);
    assert_eq!(booked(&bed, [prefill, decoder]), [1, 2]);
}

/// A prefill frame, and a decode frame that names no lease, of a model
/// that no binding matches stand alone too: the document pairs the
/// sides of another model, and no plan pairs a side of this one. The
/// model has workers of each role here, and each frame is declined as
/// the model's single request is: by the policy, at the stage
/// `binding`. Nothing is filed, and neither frame is counted with the
/// sides.
#[tokio::test]
async fn a_prefill_or_a_decode_frame_of_a_model_no_binding_matches_is_declined_as_its_single_is() {
    let document = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": { "p": { "pick": "random" } },
          "bindings": [
            { "match": { "model": "model-of-another", "phase": ["prefill", "decode"] },
              "plan": "pd-pair", "profiles": { "prefill": "p", "decode": "p" } },
            { "match": { "model": "model-of-another" }, "profile": "p" }
          ]
        }"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a document of this build");
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    let unbound = declined(proto::ShedReason::PolicyDeclined, "binding");
    let phases = [
        proto::Phase::Single,
        proto::Phase::Prefill,
        proto::Phase::Decode,
    ];
    for (n, phase) in (1..).zip(phases) {
        let answer = bed.ask(phased(frame(n, 1), phase)).await;
        assert_eq!(*outcome(&answer), unbound, "{phase:?}");
    }
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
}

/// The `random` picker, keeping what the engine tells it of each side.
#[derive(Debug)]
struct Sided(Arc<Mutex<SideCalls>>);

impl Lifecycle for Sided {
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

impl Picker for Sided {
    fn name(&self) -> &'static str {
        "sided"
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

/// A pair's answer that does not leave gives its lease up whole: the
/// record leaves the file with its decode side, and the engine that
/// heard each side's issue hears each side's release. A decode side
/// its frame asked for later was told no issue, and is told no
/// release. Here the decision's last read of the clock shows its
/// budget gone.
#[tokio::test]
async fn a_pairs_answer_that_does_not_leave_gives_up_both_sides_of_its_lease() {
    let late = Some(ReleaseReason::LateDecision);
    for later in [false, true] {
        let told = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("sided", move || Box::new(Sided(Arc::clone(&kept))))
            .expect("a free name");
        let steps = [Duration::ZERO, BUDGET].map(Step::At);
        let mut bed = bed(Some(pair_plan(&catalog, "sided")), true, (2, 8), steps);
        let (prefill, decode) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
        let mut ask = phased(frame(1, 1), proto::Phase::Prefill);
        ask.decode_later = later;
        let answer = bed.ask(ask).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        assert_eq!(bed.store.live(GATEWAY), 0);
        let both = [
            (Side::Prefill, prefill, None),
            (Side::Decode, decode, None),
            (Side::Prefill, prefill, late),
            (Side::Decode, decode, late),
        ];
        let heard = both
            .into_iter()
            .filter(|(side, ..)| !later || *side == Side::Prefill);
        let heard: SideCalls = heard.collect();
        assert_eq!(*told.lock(), heard, "a later decode: {later}");
    }
}
