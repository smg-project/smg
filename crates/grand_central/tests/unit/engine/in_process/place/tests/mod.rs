mod admission;
mod encode;
mod lent_ring;
mod pair;
mod prompt_entry;
mod request;
mod target;

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashMap},
    ptr,
    sync::{
        atomic::{self, AtomicU32},
        OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use http::{HeaderMap, HeaderName, HeaderValue};
use openai_protocol::worker::WorkerStatus;
use parking_lot::Mutex;
use smg_worker::{
    load::state::{one_rank_load, TestFeed},
    test_support::ready_worker,
};
use tokio::time::sleep;

use super::*;
use crate::{
    document::spec::{
        Assignment, BoundedLoadParams, ConsistentHashParams, DecodePick, FilterSpec, KeySourceSpec,
        Needs, OverloadSpec, PolicySpec, Recovery, RoutingKeySpec, ScoreStage, SessionsSpec,
        ShadowSpec, MANUAL_SESSION_HEADER,
    },
    engine::{
        decision::{Queue, Shed, Target, Targets, Trace},
        decision_engine::{Decided, DecisionEngine},
        framework::{Filter, Picker, Place, PoolWord, Scorer, TieBand, DRAW_FROM},
        in_process::single::tests::ruled_ahead,
        list_cell::PublishedList,
        records::worker_id,
        view::{fingerprint, Candidate, FixedFacts, HashKind},
        CacheIndex, CacheIndexes, EncodeLists, PairAdmission, PairLists, PartnerClasses,
    },
    policies::{
        consistent_hashing::{tests::InOrder as Lent, Head},
        prefix_hash::HeadRule,
    },
    stages::{pickers::push_drawn, registry::StageRegistry, Lifecycle},
    tracking::lease::{EventSet, Lease, Outcome, Side},
};

const MODEL: &str = "model-a";

fn engine_of(document: &PolicySpec) -> InProcess {
    InProcess::new(AppliedSpec::validate(document).expect("the document validates"))
}

/// An engine as its host holds it: for the life of the process.
fn hosted(engine: InProcess) -> &'static InProcess {
    Box::leak(Box::new(engine))
}

/// What a stage heard of a lease, in order: the word with the index its
/// lease names, and the worker the word names (the worker the lease books,
/// and the target taken at a dispatch).
type Heard = Arc<Mutex<Vec<(String, WorkerId)>>>;

/// The budget each decision of a picker was given, in order.
type Budgets = Arc<Mutex<Vec<u32>>>;

/// A picker that names every worker the filters keep, in the list's order,
/// takes every word of a lease, and notes the budget of each decision.
/// `panics` makes its pick panic.
#[derive(Debug)]
struct InOrder {
    heard: Heard,
    budgets: Budgets,
    panics: bool,
}

impl Lifecycle for InOrder {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        let heard = format!("issued at {}", lease.target_index);
        self.heard.lock().push((heard, lease.booked));
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        let heard = format!("dispatched at {}", lease.target_index);
        self.heard.lock().push((heard, target.id));
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        let heard = format!("completed {:?}", outcome.class);
        self.heard.lock().push((heard, lease.booked));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        let heard = format!("released {reason:?}");
        self.heard.lock().push((heard, lease.booked));
    }
}

impl Picker for InOrder {
    fn name(&self) -> &'static str {
        "in-order"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        assert!(!self.panics, "this picker panics");
        self.budgets.lock().push(facts.budget_micros);
        push_drawn(candidates, kept, out);
    }
}

/// The `random` preset with [`InOrder`] in its pick's place and without
/// its `healthy` filter, so that a worker that is not ready stays a
/// candidate of the decision, and what the picker heard.
fn in_order(panics: bool) -> (&'static InProcess, Heard) {
    let (engine, heard, _) = in_order_budgets(panics);
    (engine, heard)
}

/// [`in_order`], and the budget each of its decisions was given.
fn in_order_budgets(panics: bool) -> (&'static InProcess, Heard, Budgets) {
    in_order_edited(panics, |_| {})
}

/// [`in_order_budgets`] under its document as `edit` leaves it.
fn in_order_edited(
    panics: bool,
    edit: impl FnOnce(&mut PolicySpec),
) -> (&'static InProcess, Heard, Budgets) {
    let (heard, budgets) = (Heard::default(), Budgets::default());
    let (kept, told) = (Arc::clone(&heard), Arc::clone(&budgets));
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("in-order", move || {
        Box::new(InOrder {
            heard: Arc::clone(&kept),
            budgets: Arc::clone(&told),
            panics,
        })
    });
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    let profile = profile.expect("the preset's profile");
    profile.pick = "in-order".into();
    profile
        .filter
        .retain(|filter| *filter != FilterSpec::Name("healthy".to_owned()));
    edit(&mut document);
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = InProcess::new(spec.expect("the stages hold the picker"));
    (hosted(engine), heard, budgets)
}

/// A pair's host that admits every worker.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A pair's two lists, both `members`, under `classes`: every worker is
/// admitted and neither side is held to places.
fn pair_of<'a>(members: &'a PublishedList, classes: &'a PartnerClasses) -> PairLists<'a> {
    PairLists {
        prefill: members.lent(),
        decode: members.lent(),
        classes,
        admission: &Open,
        prefill_places: None,
        decode_places: None,
        prefill_stalled: &[],
        prefill_held: &[],
        decode_stalled: &[],
        decode_held: &[],
    }
}

/// An encode request's three lists: `pair` and its prefills as the
/// encoders, with nothing said of them.
fn three_of(pair: PairLists<'_>) -> WorkerLists<'_> {
    WorkerLists::Encode(EncodeLists {
        pair,
        encoders: pair.prefill,
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: &[],
        prefill_classes: &[],
    })
}

fn preset(name: &str) -> InProcess {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    engine_of(&document)
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// Opens `worker`'s circuit breaker: the worker stays healthy and cannot be
/// sent to.
fn trip(worker: &Arc<dyn Worker>) {
    for _ in 0..64 {
        if !worker.circuit_breaker_can_execute() {
            break;
        }
        worker.record_circuit_breaker_outcome(false);
    }
    assert!(worker.routing_state().healthy && !ready(worker.as_ref()));
}

/// A document whose binding reads a routing value from a header, then from
/// the request's own id, and a client's value from a second header.
fn keyed() -> InProcess {
    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    let header = |name: &str, client| KeySourceSpec {
        header: Some(name.to_owned()),
        client,
        ..KeySourceSpec::default()
    };
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![
            header("x-conversation", false),
            rid,
            header("x-client", true),
        ],
        strip_lineage: false,
    });
    engine_of(&document)
}

/// A whole decision's pick is handed on. A shed hands on its reason, the
/// stage that left nobody, its retry hint and the side of a pair it names,
/// and a queue ticket is answered as queued. The side costs the answer no
/// byte: it stands in one the shed's answer had spare.
#[test]
fn a_decision_that_placed_nothing_says_why() {
    assert!(picked(&Decision::Pick(Pick::default())).is_ok());
    for side in [None, Some(PairSide::Prefill), Some(PairSide::Decode)] {
        let shed = Shed {
            reason: ShedReason::Saturated,
            emptied_by: "has-capacity",
            retry_after_secs: 7,
            wait_allowed: false,
            side,
            trace: Trace::default(),
        };
        let said = Unplaced::Shed {
            reason: ShedReason::Saturated,
            emptied_by: "has-capacity",
            retry_after_secs: 7,
            side,
        };
        assert_eq!(picked(&Decision::Shed(shed)).err(), Some(said));
    }
    assert_eq!(size_of::<Unplaced>(), 24);
    let queue = Queue {
        ticket: 1,
        max_wait_micros: 0,
        trace: Trace::default(),
    };
    assert_eq!(
        picked(&Decision::Queue(queue)).err(),
        Some(Unplaced::Queued)
    );
}

/// A plain request is placed on the worker at the place the engine picked,
/// on the pick's own read of it: a worker that is not ready, which the pick
/// passes over, is never the answer. Nothing is owed of such a placement,
/// no routing value is derived for it, and it names the profile that
/// picked by its label.
#[test]
fn a_plain_request_is_placed_on_the_worker_at_its_place() {
    let engine = hosted(preset("round_robin"));
    let alone = matches!(
        engine.spec().place_for(MODEL, Phase::Single),
        Place::Alone(_)
    );
    assert!(alone, "one pick decides the place");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let headers = [("x-conversation", "talk-7")];
    let request = PlaceRequest {
        headers: &headers,
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let mut turns = [0_u32; 4];
    for _ in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        let at = list
            .iter()
            .position(|worker| Arc::ptr_eq(worker, &placement.worker));
        turns[at.expect("a worker of the list")] += 1;
        assert!(!placement.owes_events() && placement.decode.is_none());
        assert_eq!(
            (placement.profile, placement.routing_key),
            ("round_robin", None)
        );
    }
    assert_eq!(turns, [2; 4], "every worker takes its turn");

    list[1].set_status(WorkerStatus::NotReady);
    for turn in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        assert!(!Arc::ptr_eq(&placement.worker, &list[1]), "turn {turn}");
    }
}

/// A preset's document with a spare profile of another name ahead of the
/// bound one, so that the bound profile is the document's second.
fn spared(name: &str) -> &'static InProcess {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let spare = document.profiles[name].clone();
    document.profiles.insert("a_spare".to_owned(), spare);
    hosted(engine_of(&document))
}

/// One request of `engine` placed over a list of four workers, as the
/// request keeps it.
fn placed(engine: &'static InProcess) -> Placement<'static> {
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    engine.place(&request).expect("a placement").kept()
}

/// A placement names the profile that picked by its label: the profile's
/// name as its document writes it, one string a name for the life of the
/// process. A place's placement holds it, and a whole decision's. Two
/// engines of one document name the same string. An engine of another
/// document, whose profile at the same position has another name, names
/// its own.
#[test]
fn a_placement_names_the_profile_that_picked_by_its_label() {
    // Where one pick decides the place, the position of the profile that
    // picks: the second in each document, behind the spare one.
    let alone_at = |engine: &InProcess| match engine.spec().place_for(MODEL, Phase::Single) {
        Place::Alone(profile) => Some(profile.0),
        _ => None,
    };
    let (turns, again) = (spared("round_robin"), spared("round_robin"));
    let (passes, draws) = (spared("passthrough"), spared("power_of_two"));
    let at = [alone_at(turns), alone_at(passes), alone_at(draws)];
    assert_eq!(at, [Some(1), Some(1), None]);

    let first = placed(turns);
    assert_eq!(first.profile, "round_robin");
    let same = ptr::eq(first.profile, placed(again).profile);
    assert!(same, "one string a name");
    // The other document's second profile has another name.
    assert_eq!(placed(passes).profile, "passthrough");
    // A whole decision's placement names its profile the same way.
    assert_eq!(placed(draws).profile, "power_of_two");
}

/// Placing a request asks the process's table of names for nothing: the
/// label a placement holds is read from the engine's own list, which the
/// engine made at its start. A place's placement, and a whole decision's.
/// The control: a call of `intern` on this thread raises the count by one.
#[test]
fn placing_a_request_asks_the_table_of_names_for_nothing() {
    use crate::names::{intern, tests::ASKED};

    let asked = || ASKED.with(Cell::get);
    let before = asked();
    assert_eq!(intern("round_robin"), "round_robin");
    assert_eq!(asked(), before + 1, "a call is counted");
    for name in ["round_robin", "power_of_two"] {
        let engine = hosted(preset(name));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
        let before = asked();
        for _ in 0..8 {
            let placement = engine.place(&request).expect("a placement");
            assert_eq!(placement.profile, name);
        }
        assert_eq!(asked(), before, "{name}");
    }
}

/// A placement states what its pick states of the worker the request goes
/// to ([`Placement::overlap`]): a host observes its cache series from it.
/// A prompt's first request reads a miss of its blocks. Its next request
/// reads every block held, by the record of sent prefixes, and a kept
/// placement says the same. A pick states its first target's worker alone:
/// a single request its route's rule sends to the target behind goes to
/// another worker and is stated nothing, though one decision placed it.
/// Neither is a place one pick decides. A pair's statement stands with a
/// pair's cases (`a_pairs_placement_states_what_its_prefill_holds`).
///
/// The classic policy observes the share of the worker it chose
/// (`model_gateway/src/policies/cache_aware.rs:2676-2688`).
#[test]
fn a_placement_states_what_its_pick_states_of_the_worker_taken() {
    use crate::engine::decision::{Overlap, OverlapSource};

    let engine = hosted(preset("cache_aware"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let prompt = Prompt {
        tokens: Some(&tokens),
        ..Prompt::NONE
    };
    let mut request = PlaceRequest {
        prompt,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let first = engine.place(&request).expect("a placement");
    let of = first.overlap.of;
    assert_ne!(of, 0, "a miss is stated of the request's blocks");
    let miss = Overlap {
        of,
        ..Overlap::NONE
    };
    assert_eq!(first.overlap, miss);

    let again = engine.place(&request).expect("a placement");
    let held = Overlap {
        blocks: of,
        source: OverlapSource::Predictor,
        ..miss
    };
    assert_eq!((again.id(), again.overlap), (first.id(), held));
    assert_eq!(again.kept().overlap, held, "a kept placement states it too");

    // The rule refuses the prompt's worker: the request goes to the target
    // behind it, of which the pick states nothing.
    let holder = first.id();
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        asked.set(asked.get() + 1);
        worker_id(worker) != holder
    };
    request.admits = Some(&admits);
    let moved = engine.place(&request).expect("a placement");
    assert_ne!(moved.id(), holder);
    assert_eq!(
        asked.get(),
        2,
        "one decision: its first target, then the next"
    );
    assert_eq!(moved.overlap, Overlap::NONE);

    // A place one pick decides reads no row.
    request.admits = None;
    let plain = hosted(preset("round_robin"));
    let placed = plain.place(&request).expect("a placement");
    assert_eq!(placed.overlap, Overlap::NONE);
}

/// A decision kept behind a placement, which hands out no trace, reads no
/// best row for it where no stage of the document takes its issue and the
/// document keeps no sample: the debug text of the placement names none,
/// under a profile that draws first and beside a shadow profile too. A
/// caller that is handed a decision reads them, right behind such a
/// placement too, and so do a stage that takes the issue and the record of
/// a document that keeps a sample.
#[test]
fn a_trace_names_its_best_rows_for_whoever_can_read_them() {
    // The smallest pool a profile that draws first draws over.
    let list = pool(DRAW_FROM);
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest {
        explain_id: Some("host-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    // The debug text of a placement prints the decision its lease keeps.
    // Each preset here takes a word of a lease: `least_load` and
    // `power_of_two` the dispatch alone, `bucket` the issue too.
    let kept =
        |engine: &'static InProcess| format!("{:?}", engine.place(&request).expect("a placement"));
    // The best rows of a decision its caller is handed.
    let handed = |engine: &'static InProcess| {
        let handed = engine.decide_at(&Made::new(engine, &request, &[]), Instant::now);
        let handed = handed
            .ok()
            .and_then(Decided::whole)
            .expect("a whole decision");
        handed.0.decision.pick().expect("a pick").trace.top_len
    };
    let scores = hosted(preset("least_load"));
    assert!(kept(scores).contains("top_len: 0"), "nobody reads them");
    assert_eq!(handed(scores), 3, "a caller reads them");
    let draws = hosted(preset("power_of_two"));
    assert!(kept(draws).contains("top_len: 0"), "nor behind a draw");
    // Beside a shadow profile the front of the two decisions states that
    // nobody reads them, for that placement alone.
    let mut document = PolicySpec::preset("least_load").expect("a built-in preset");
    let twin = document.profiles.values().next().cloned();
    document
        .profiles
        .insert("twin".into(), twin.expect("the preset's profile"));
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    document.generation = 1;
    let shadowed = hosted(engine_of(&document));
    assert!(kept(shadowed).contains("top_len: 0"), "nor beside a shadow");
    assert_eq!(handed(shadowed), 3, "a caller reads them behind it");
    let books = hosted(preset("bucket"));
    assert!(kept(books).contains("top_len: 3"), "a stage is handed them");
    // A document that keeps a sample keeps a placement's decision as it
    // stands: the record names them.
    let mut document = PolicySpec::preset("least_load").expect("a built-in preset");
    document.generation = 1;
    document.explain.sample = 1;
    let sampled = hosted(engine_of(&document));
    assert!(
        kept(sampled).contains("top_len: 3"),
        "a sample is kept of them"
    );
    let record = sampled.explained(b"host-1").expect("the decision is kept");
    let pick = record.decision.pick().expect("a pick");
    assert_eq!(pick.trace.top_len, 3, "the record names them");
}

/// A program that asks `place` states no worker id of its own: a placement
/// of a place names its worker by the fingerprint of its address, and a
/// retry states that id as tried. A program whose list states ids of its
/// own is told so in a debug build, at its first request placed alone.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "states no worker id of its own")]
fn a_caller_that_states_worker_ids_is_told_in_a_debug_build() {
    let engine = hosted(preset("round_robin"));
    let list = pool(2);
    let mut stated = 6;
    let members = PublishedList::with_facts(Arc::clone(&list), |_| {
        stated += 1;
        FixedFacts {
            id: Some(WorkerId(stated)),
            ..FixedFacts::default()
        }
    });
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let _ = engine.place(&request);
}

/// A caller lends no load snapshot: the decision reads the newest one the
/// engine's own task keeps, of the channel its host handed. Under a
/// document that scores by load the one worker with an empty queue takes
/// every request.
///
/// The classic router's `least_load` reads its load monitor's snapshot at
/// each decision too (`model_gateway/src/policies/least_load.rs:918`).
#[tokio::test(start_paused = true)]
async fn a_decision_reads_the_load_snapshot_the_engines_task_keeps() {
    let engine = hosted(preset("least_load"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let reports: HashMap<_, _> = (0..4)
        .map(|at| {
            let queued = if at == 2 { 0 } else { 90_000 };
            let url = format!("http://worker-{at}:8000");
            (url, one_rank_load(queued, 0.1, 1_000.0))
        })
        .collect();
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    feed.publish(&reports);
    engine.set_loads(channel.expect("the feed's channel"));
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(engine.run());
    // The task's first step takes the channel and keeps its snapshot.
    sleep(Duration::from_millis(1)).await;
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    for turn in 0..16 {
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[2]), "turn {turn}");
    }
    task.abort();
}

/// What a picker hears of a decision none of whose targets can be sent to:
/// its issue and its release by name, on the decision's first target.
fn given_up(first: WorkerId) -> [(String, WorkerId); 2] {
    [
        ("issued at 0".to_owned(), first),
        ("released CommitFailed".to_owned(), first),
    ]
}

/// A request with no worker is answered why. A decision's shed keeps its
/// reason and the stage that left nobody. Encode items over other lists
/// than an encode request's three are declined by name. A demand no worker
/// of the list meets is shed under the demand's name. A decision that
/// panics is refused, and costs its request alone.
#[test]
fn a_request_with_no_worker_is_answered_why() {
    let engine = hosted(preset("round_robin"));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let lists = WorkerLists::Single(members.lent());
    let shed = |reason, emptied_by| {
        Some(Unplaced::Shed {
            reason,
            emptied_by,
            retry_after_secs: 1,
            side: None,
        })
    };
    let none = PlaceRequest::of(MODEL, WorkerLists::None);
    assert_eq!(
        engine.place(&none).err(),
        shed(ShedReason::NoEndpoints, "none")
    );
    let demands = Demands {
        vision: true,
        ..Demands::default()
    };
    let demanding = PlaceRequest {
        demands,
        ..PlaceRequest::of(MODEL, lists)
    };
    assert_eq!(
        engine.place(&demanding).err(),
        shed(ShedReason::NoEndpoints, "vision")
    );
    // A request this call declines asks its route's rule of nobody.
    let asked = Cell::new(0);
    let admits = |_: &dyn Worker| {
        asked.set(asked.get() + 1);
        true
    };
    let encoding = PlaceRequest {
        encode: Some(EncodeItems::default()),
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, lists)
    };
    assert_eq!(
        engine.place(&encoding).err(),
        shed(ShedReason::PolicyDeclined, "encode-items")
    );
    assert_eq!(asked.get(), 0, "the rule of a declined request");
    // No worker of the list is ready: the decision sheds, and names the
    // filter that left nobody.
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let plain = PlaceRequest::of(MODEL, lists);
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&plain).err()
    else {
        panic!("a shed");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "healthy"));

    let (faulty, _) = in_order(true);
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    assert_eq!(
        faulty.place(&request).err(),
        Some(Unplaced::Refused(DecisionRefused::Faulted))
    );
    assert_eq!(faulty.faults(), 1);
}

/// A placement names the request's routing value where its document
/// derived one that leaves in the clear: the request's own id here. One
/// value is placed on one worker: the document's ring stage places it,
/// and its caller builds nothing for that. A header's value stays a
/// fingerprint, which no stage is shown, and the placement names none.
/// What a request keeps to its end borrows nothing of it.
#[test]
fn a_placement_names_the_routing_value_its_document_derived() {
    let engine = hosted(keyed());
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let by_id = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&by_id).expect("a placement");
    assert_eq!(placement.routing_key, Some("request-1"));
    let again = engine.place(&by_id).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &again.worker));

    let headers = [("x-conversation", "talk-7")];
    let by_header = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&by_header).expect("a placement");
    assert_eq!(placement.routing_key, None);
    let kept: Placement<'static> = placement.kept();
    assert!(kept.routing_key.is_none());
}

/// A shadow profile is lent the request's ring only where it reads a ring
/// by the live profile's kind of place. Beside a live `consistent_hashing`
/// profile the place stated is the routing value's own text, and the
/// client's for a request its client alone keys: no place of `prefix_hash`,
/// which has nothing to hash for such a request and takes the worker with
/// the fewest requests (`PrefixHashPolicy::select_worker_impl` in
/// `model_gateway/src/policies/prefix_hash.rs`). So the shadow answers the
/// idle worker, and is counted as another target than the ring's owner,
/// which holds two requests.
#[test]
fn a_shadow_of_another_ring_form_is_lent_no_ring() {
    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    let prefix = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let twin = prefix.profiles.into_values().next();
    document
        .profiles
        .insert("twin".into(), twin.expect("the preset's profile"));
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let ring = Lent(vec![list[0].url().to_owned()]);
    list[0].increment_load();
    list[0].increment_load();
    let headers = [("x-forwarded-for", "10.0.0.1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the ring's owner");
    // Agreed, another target, another outcome, skipped.
    let counted = engine.shadow_kinds(0);
    assert_eq!(counted, [0, 1, 0, 0]);
}

/// [`Lent`], with a count of its walks.
#[derive(Debug)]
struct Walked(Lent, AtomicU32);

impl LentRing for Walked {
    fn position(&self, key: &str) -> u64 {
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.0.head_position(head)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.walk(position, take);
    }
}

/// A shadow profile that places as its live profile does is lent the
/// request's ring and walks it as the live profile does: a shadow of the
/// plain ring beside the plain ring, and a shadow of the prefix form that
/// cuts a request by the live profile's rule. The two then agree on the
/// ring's owner. A shadow of the prefix form that cuts by another depth is
/// lent none: the places stated are the live profile's heads', so the
/// lent ring is walked for the live profile alone, as beside a shadow of
/// another ring form.
#[test]
fn a_shadow_is_lent_the_ring_where_it_places_as_its_live_profile_does() {
    // The lent ring's walks under one request, and the shadow's counts.
    let walks = |live: &str, twin: &str, depth: Option<u32>| {
        let mut document = PolicySpec::preset(live).expect("a built-in preset");
        let twin = PolicySpec::preset(twin).expect("a built-in preset");
        let mut twin = twin.profiles.into_values().next().expect("its profile");
        if let Some(depth) = depth {
            twin.score[0] = ScoreStage::ConsistentHash(ConsistentHashParams {
                prefix_tokens: Some(depth),
                ..ConsistentHashParams::default()
            })
            .into();
        }
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        document.generation = 1;
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let ring = Walked(Lent(vec![list[0].url().to_owned()]), AtomicU32::new(0));
        let ids: Vec<u32> = (0..64).collect();
        let headers = [("x-smg-routing-key", "conv-1")];
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers: &headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the ring's owner");
        // Agreed, another target, another outcome, skipped.
        let counted = engine.shadow_kinds(0);
        (ring.1.load(atomic::Ordering::Relaxed), counted)
    };
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    for (live, other) in [(plain, prefix), (prefix, plain)] {
        let (alone, _) = walks(live, other, None);
        assert!(alone > 0, "{live}: the live profile walks the lent ring");
        let (both, counted) = walks(live, live, None);
        assert_eq!(both, 2 * alone, "{live}: its twin walks it too");
        assert_eq!(counted, [1, 0, 0, 0], "{live}: the two agree");
    }
    let (alone, _) = walks(prefix, plain, None);
    let (another_depth, _) = walks(prefix, prefix, Some(128));
    assert_eq!(another_depth, alone, "a shadow of another depth");
}

/// A shadow profile that places by a ring is lent the request's ring
/// behind a live profile that reads none, and walks it as it would live:
/// the live profile takes the idle worker and the shadow the ring's
/// owner. The lender is asked there for the shadow alone, so by the
/// shadow's rules: a request that keeps its budget for its answer asks it
/// nothing, and a lender that panics costs its request nothing.
#[test]
fn a_shadow_alone_walks_the_lent_ring() {
    // The ring preset's binding keeps its routing key: the preset's
    // profile is its shadow, and the profile of `least_load` decides.
    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    let twin = document.profiles.keys().next().cloned();
    let live = PolicySpec::preset("least_load").expect("a built-in preset");
    let live = live.profiles.into_values().next();
    document
        .profiles
        .insert("live".into(), live.expect("the preset's profile"));
    document.bindings[0].profile = Some("live".into());
    document.bindings[0].shadow = twin.map(ShadowSpec::Profile);
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    // The key stands at the second address: the ring's owner is the
    // busy worker, and the live profile takes the idle one.
    assert_eq!(ring.position("conv-1") % 2, 1, "the key's owner");
    for _ in 0..8 {
        list[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    // Agreed, another target, another outcome, skipped.
    let counted = || engine.shadow_kinds(0);
    let placed = |ring: &dyn LentRing, budget_micros| {
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(ring),
            budget_micros,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
        (counted(), engine.faults())
    };
    let said = placed(&ring, 0);
    let walks = ring.1.load(atomic::Ordering::Relaxed);
    assert_eq!((walks, said), (1, ([0, 1, 0, 0], 0)), "the lent ring");
    // Under the shadow's floor the lender is asked nothing.
    assert_eq!(placed(&Unasked, 500), ([0, 1, 0, 1], 0), "a short budget");
    // A lender that panics is its shadow's fault: counted, and nothing
    // is compared.
    assert_eq!(placed(&Unasked, 0), ([0, 1, 0, 1], 1), "a lender's panic");
    // A request no worker serves is refused ahead of its shadow.
    let unserved = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let refused = |emptied_by| {
        Some(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            retry_after_secs: 1,
            side: None,
        })
    };
    assert_eq!(engine.place(&unserved).err(), refused("vision"));
    assert_eq!((counted(), engine.faults()), ([0, 1, 0, 1], 1), "a refusal");
    // Over a list of nobody no row stands to be placed: the lender is
    // asked nothing, and the two refusals are counted as agreed.
    let nobody = PublishedList::new(pool(0));
    let empty = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(nobody.lent()))
    };
    assert_eq!(engine.place(&empty).err(), refused("none"));
    assert_eq!((counted(), engine.faults()), ([1, 1, 0, 1], 1), "nobody");
}

/// The `least_load` pair with the ring pair of `prefill` and `decode` as
/// its sides' shadows: the ring pair's binding keeps its routing key, its
/// two profiles are the sides' shadows, and the profiles of the
/// `least_load` pair decide. `after` picks the decode once the prefill is
/// sent.
fn ring_shadowed(prefill: &str, decode: &str, after: bool) -> PolicySpec {
    let document = PolicySpec::pair_presets(prefill, decode);
    let mut document = document.expect("the pair presets");
    let live = PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    let profiles = &mut document.profiles;
    let sides = live.profiles.into_iter().map(|(side, live)| {
        let twin = profiles.insert(side.clone(), live);
        let name = format!("{side}-next");
        profiles.insert(name.clone(), twin.expect("the side's ring profile"));
        (side, name)
    });
    let sides = ShadowSpec::Sides(sides.collect());
    document.bindings[0].shadow = Some(sides);
    if after {
        document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    }
    document.generation = 1;
    document
}

/// What each side's shadow said under `engine`: agreed, another target,
/// another outcome, skipped; the prefill's shadow, the decode's, and the
/// decode's beside a decode decided again.
fn sides_said(engine: &InProcess) -> [[u64; 4]; 3] {
    use crate::engine::shadow::ShadowSide;

    let mut kinds = [[0; 4]; 3];
    let _ = engine.shadow_counts(|count| {
        let side = match count.side {
            ShadowSide::Prefill => 0,
            ShadowSide::Decode => 1,
            ShadowSide::DecodeAgain => 2,
            _ => return,
        };
        kinds[side][count.kind.index()] = count.count;
    });
    kinds
}

/// A pair side's shadow profile that places by a ring is lent the
/// request's ring behind two live sides that read none, and walks it as it
/// would live: each live side takes the idle worker and each side's shadow
/// the ring's owner. A side's shadow of the plain ring beside the other
/// side's of the prefix form is stated its own place too: the key's text
/// for the one, the key as the request's one head for the other.
#[test]
fn a_sides_shadow_alone_walks_the_lent_ring() {
    // The lent ring's walks under one request, what each side's shadow
    // said and the engine's faults, under the pair of `ring_shadowed`.
    let walks = |prefill: &str, decode: &str, lifted: bool| {
        let document = ring_shadowed(prefill, decode, false);
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let addresses = list.iter().map(|worker| worker.url().to_owned());
        let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
        // The key stands at the second address: the ring's owner is the
        // busy worker, and each live side takes the idle one.
        assert_eq!(ring.position("conv-1") % 2, 1, "the key's owner");
        for _ in 0..8 {
            list[1].increment_load();
        }
        // Every worker overloaded: no prefill stands but for the veto, and
        // the pair is made past it, which the prefill's buffer marks. The
        // ring lent then panics where it is asked.
        if lifted {
            for worker in list.iter() {
                worker.set_overloaded(true);
            }
        }
        let headers = [("x-smg-routing-key", "conv-1")];
        let lender: &dyn LentRing = if lifted { &Unasked } else { &ring };
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(lender),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let placement = engine.place(&request).expect("a pair");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
        (
            ring.1.load(atomic::Ordering::Relaxed),
            sides_said(engine),
            engine.faults(),
        )
    };
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    let (agreed, another) = ([1, 0, 0, 0], [0, 1, 0, 0]);
    assert_eq!(
        walks(plain, plain, false),
        (2, [another, another, [0; 4]], 0),
        "the ring's owner"
    );
    // The key as the prefix form's one head stands at the idle worker on
    // the lent ring: that side's shadow agrees with its live side, and
    // the plain ring's beside it names the key's owner.
    assert_eq!(
        walks(plain, prefix, false),
        (2, [another, agreed, [0; 4]], 0),
        "the decode's prefix form"
    );
    assert_eq!(
        walks(prefix, plain, false),
        (2, [agreed, another, [0; 4]], 0),
        "the prefill's prefix form"
    );
    // Beside a pair made past a veto that left the prefills nobody, each
    // side's shadow follows the pair's default and agrees. Neither is
    // stated a place: the filters empty the prefill shadow's own set, and
    // the default asks for none. So the lender is not asked, and a ring
    // that panics where it is asked counts no fault.
    assert_eq!(
        walks(plain, plain, true),
        (0, [agreed, agreed, [0; 4]], 0),
        "the veto's mark"
    );
}

/// [`ring_shadowed`] with the prefix form as both sides' shadows, the
/// decode's `depth` tokens deep where one is named: the prefill's cuts a
/// request by the stock depth, 256 tokens. The stock weight keeps the
/// ring's owner above the least loaded worker.
fn prefix_shadowed(depth: Option<u32>, after: bool) -> PolicySpec {
    let mut document = ring_shadowed("prefix_hash", "prefix_hash", after);
    if let Some(depth) = depth {
        let decode = document.profiles.get_mut("decode-next");
        decode.expect("the decode's shadow").score[0] =
            ScoreStage::ConsistentHash(ConsistentHashParams {
                weight: 8.0,
                prefix_tokens: Some(depth),
                ..ConsistentHashParams::default()
            })
            .into();
    }
    document
}

/// A pair's two sides' shadows of the prefix form walk the lent ring once
/// each, of one depth and of two: a routing value that leads the prompt
/// is the request's one head under both rules.
#[test]
fn both_sides_shadows_of_the_prefix_form_walk_the_lent_ring() {
    let walks = |depth: Option<u32>| {
        let document = prefix_shadowed(depth, false);
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let addresses = list.iter().map(|worker| worker.url().to_owned());
        let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
        let headers = [("x-smg-routing-key", "conv-1")];
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        engine.place(&request).expect("a pair");
        (ring.1.load(atomic::Ordering::Relaxed), engine.faults())
    };
    assert_eq!(walks(None), (2, 0), "one depth");
    assert_eq!(walks(Some(128)), (2, 0), "another depth");
}

/// A decode's shadow that cuts a request by another depth than the
/// prefill's shadow is stated the place of its own head on the lent ring,
/// at the pair's decision and at the decode decided again. The prefill's
/// shadow walks the ring from the place of the prompt's 256 tokens, where
/// the walk reaches the idle worker both live sides take: it agrees. The
/// decode's walks it from the place of the first sixteen, where the walk
/// reaches the busy worker: it names another target, both times.
#[test]
fn a_decodes_shadow_of_another_depth_walks_the_lent_ring_from_its_own_head() {
    let engine = hosted(engine_of(&prefix_shadowed(Some(16), true)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    for _ in 0..8 {
        list[1].increment_load();
    }
    let ring = ByDepth {
        shallowest: list[1].url().to_owned(),
        deeper: list[0].url().to_owned(),
    };
    let ids: Vec<u32> = (0..256).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let (agreed, another) = ([1, 0, 0, 0], [0, 1, 0, 0]);
    let mut placement = engine.place(&request).expect("a pair");
    assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
    let first = sides_said(engine);
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(
        (first, sides_said(engine)),
        ([agreed, another, [0; 4]], [agreed, another, another]),
        "the pair's decision, then the later call"
    );
    assert_eq!(engine.faults(), 0, "no fault");
}

/// A decode's shadow of another depth moves no live side, and a maker is
/// asked once for each side's shadow. Under the document of the test
/// above, a maker that hands on the engine's own is asked for two shadow
/// profiles in the pair's decision, and the request made again on it
/// states no ring and no place: what a shadow's ask kept reaches no live
/// statement. The live pair is the one the same document takes with no
/// shadow, at the pair's decision and at the decode decided again.
#[test]
fn a_decodes_shadow_of_another_depth_asks_its_own_profile_and_moves_no_live_side() {
    /// A maker that hands on the engine's own and keeps the shadow
    /// profiles it was asked for.
    struct Recorded<'r, 'a>(Made<'r, 'a>, RefCell<Vec<u16>>);

    impl RequestMaker for Recorded<'_, '_> {
        fn model(&self) -> &str {
            self.0.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.0.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.0.workers()
        }

        fn shadow_ring<'w>(
            &'w self,
            spec: &'w AppliedSpec,
            shadow: u16,
        ) -> Option<(&'w dyn LentRing, &'w [u64])> {
            self.1.borrow_mut().push(shadow);
            self.0.shadow_ring(spec, shadow)
        }
    }

    let live = |shadowed: bool| {
        let mut document = prefix_shadowed(Some(16), true);
        if !shadowed {
            document.bindings[0].shadow = None;
        }
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        for _ in 0..8 {
            list[1].increment_load();
        }
        let ring = ByDepth {
            shallowest: list[1].url().to_owned(),
            deeper: list[0].url().to_owned(),
        };
        let ids: Vec<u32> = (0..256).collect();
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(&ring),
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let at = |placed: &_| list.iter().position(|member| Arc::ptr_eq(member, placed));
        let mut placement = engine.place(&request).expect("a pair");
        let first = (
            at(&placement.worker),
            placement.decode.as_ref().and_then(at),
        );
        placement.sent(Side::Prefill);
        let decided = placement.decode_again(&request);
        assert!(decided.is_ok(), "the decode decided again");
        let again = placement.decode.as_ref().and_then(at);
        let sides = (first, again, engine.faults());
        // The pair's decision once more, through a maker that keeps what
        // it was asked: then the request made again on that maker.
        let maker = Recorded(Made::new(engine, &request, &[]), RefCell::default());
        let decided = engine.decide_at(&maker, Instant::now);
        assert!(decided.is_ok(), "the pair's decision");
        let facts = maker.0.make(engine.standing_spec()).facts;
        let unstated = facts.ring.is_none()
            && facts.ring_positions.is_empty()
            && facts.decode_ring_positions.is_none();
        (sides, unstated, maker.1.take())
    };
    let (shadowed, plain) = (live(true), live(false));
    assert_eq!(shadowed.0, plain.0, "the live pair, shadowed or not");
    let pair = plain.0;
    assert!(
        matches!(pair, ((Some(0), _), Some(_), 0)),
        "the idle prefill, a decode, no fault: {pair:?}"
    );
    assert!(
        shadowed.1 && plain.1,
        "the request made again states no place"
    );
    let (asked, none) = (shadowed.2, plain.2);
    assert_eq!(none, [0_u16; 0], "no shadow, no ask");
    assert_eq!(asked.len(), 2, "one ask a side's shadow: {asked:?}");
    assert_ne!(asked[0], asked[1], "each side's own profile");
}

/// A lender that places no head is asked about one head for a pair's two
/// sides' shadows of two depths: it declined the prefill's shadow's, and
/// is asked about none for the decode's. Neither shadow walks the ring.
#[test]
fn a_lender_that_declined_one_sides_shadow_is_asked_nothing_for_the_other() {
    let engine = hosted(engine_of(&prefix_shadowed(Some(16), false)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let ring = Scripted(&[None], AtomicU32::new(0));
    let ids: Vec<u32> = (0..256).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    engine.place(&request).expect("a pair");
    let asked = ring.1.load(atomic::Ordering::Relaxed);
    assert_eq!((asked, engine.faults()), (1, 0), "one head, no walk");
}

/// A lender that panics when it is asked a head's place faults each
/// side's shadow that asks it, of one depth and of two: two faults a
/// pair's decision, and neither shadow says a word. The pair is placed
/// all the same.
#[test]
fn a_lender_that_panics_at_a_head_faults_both_sides_shadows() {
    let reading = |depth: Option<u32>| {
        let engine = hosted(engine_of(&prefix_shadowed(depth, false)));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let ids: Vec<u32> = (0..256).collect();
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(&Unasked),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let placed = engine.place(&request).is_ok();
        (placed, engine.faults(), sides_said(engine))
    };
    let both = (true, 2, [[0; 4]; 3]);
    assert_eq!(reading(None), both, "one depth");
    assert_eq!(reading(Some(16)), both, "two depths");
}

/// A pair's decode decided again runs the decode's shadow once more, and
/// that shadow is stated its own place on the lent ring once more: it
/// walks the lent ring a further time, and names the ring's owner beside
/// the idle worker its call takes.
#[test]
fn a_decode_decided_again_walks_the_lent_ring_once_more() {
    let plain = "consistent_hashing";
    let engine = hosted(engine_of(&ring_shadowed(plain, plain, true)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    // The key's owner is the busy worker, as in the test above.
    for _ in 0..8 {
        list[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let seen = || (ring.1.load(atomic::Ordering::Relaxed), sides_said(engine));
    let another = [0, 1, 0, 0];
    let mut placement = engine.place(&request).expect("a pair");
    assert_eq!(
        seen(),
        (2, [another, another, [0; 4]]),
        "the first decision"
    );
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(seen(), (3, [another; 3]), "one more walk");
    assert_eq!(engine.faults(), 0, "no fault");
}

/// A decode's shadow the all-vetoed rule steers is stated no place: every
/// partner of the prefill stands over its thresholds, the rule picks among
/// them for the shadow as it does for the live decode, and no stage reads
/// a place. The lender is not asked, at the pair's decision or at the
/// decode decided again: a ring that panics where it is asked counts no
/// fault, and the shadow agrees both times.
#[test]
fn a_steered_decodes_shadow_asks_the_lender_nothing() {
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, true);
    // The decode's shadow alone: the prefill's would ask for its place.
    if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
        sides.retain(|side, _| side.contains("decode"));
    }
    let engine = hosted(engine_of(&document));
    let members = PublishedList::new(pool(1));
    let decodes: Arc<[Arc<dyn Worker>]> = (0..2)
        .map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL))
        .collect();
    let decode_members = PublishedList::new(Arc::clone(&decodes));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..]]);
    for worker in decodes.iter() {
        worker.set_overloaded(true);
    }
    // One decode the busier: the rule's pick is the other, for both.
    for _ in 0..8 {
        decodes[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let pair = PairLists {
        decode: decode_members.lent(),
        ..pair_of(&members, &classes)
    };
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let seen = || (sides_said(engine), engine.faults());
    let agreed = [1, 0, 0, 0];
    let mut placement = engine.place(&request).expect("a pair");
    assert_eq!(seen(), ([[0; 4], agreed, [0; 4]], 0), "the pair's decision");
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(seen(), ([[0; 4], agreed, agreed], 0), "the later call");
}

/// The prefix form of the ring as a shadow behind a live profile that
/// reads none: the shadow is stated the places of its request's heads, by
/// its own rule, and walks the lent ring. A lender that places no head
/// lends the shadow no ring: it is asked nothing more, and the shadow
/// places by its own.
#[test]
fn a_prefix_shadow_alone_walks_the_lent_ring() {
    let mut document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let twin = document.profiles.keys().next().cloned();
    let live = PolicySpec::preset("least_load").expect("a built-in preset");
    let live = live.profiles.into_values().next();
    document
        .profiles
        .insert("live".into(), live.expect("the preset's profile"));
    document.bindings[0].profile = Some("live".into());
    document.bindings[0].shadow = twin.map(ShadowSpec::Profile);
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    let ids: Vec<u32> = (0..64).collect();
    // The shadow's skips and the faults, behind one more request.
    let placed = |ring: &dyn LentRing| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).expect("a placement");
        let [.., skipped] = engine.shadow_kinds(0);
        (skipped, engine.faults())
    };
    assert_eq!(placed(&ring), (0, 0), "the shadow ran");
    assert_eq!(ring.1.load(atomic::Ordering::Relaxed), 1, "the lent ring");
    // A lender that places no head panics where it is asked for a key's
    // place or walked: the shadow ran, and no fault is counted.
    assert_eq!(placed(&Headless), (0, 0), "a lender of no head");
}

/// A ring no decision may ask.
#[derive(Debug)]
struct Unasked;

impl LentRing for Unasked {
    fn position(&self, key: &str) -> u64 {
        panic!("the ring was asked where {key} stands");
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        panic!("the ring was asked where the head {head:?} stands");
    }

    fn walk(&self, position: u64, _: &mut dyn FnMut(&str) -> bool) {
        panic!("the ring was walked from {position}");
    }
}

/// A ring whose lender places no request by a head: it answers nothing for
/// one, as the trait does for a lender that says nothing, and may be asked
/// nothing else.
#[derive(Debug)]
struct Headless;

impl LentRing for Headless {
    fn position(&self, key: &str) -> u64 {
        panic!("the ring was asked where {key} stands");
    }

    fn walk(&self, position: u64, _: &mut dyn FnMut(&str) -> bool) {
        panic!("the ring was walked from {position}");
    }
}

/// The `prefix_hash` preset over a view of `boundaries`, whose binding
/// reads a client's value too, from the header `x-client`.
fn prefixed(boundaries: &[u32]) -> InProcess {
    let mut document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    document.view.boundaries = boundaries.to_vec();
    let header = |name: &str, client| KeySourceSpec {
        header: Some(name.to_owned()),
        client,
        ..KeySourceSpec::default()
    };
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![header("x-smg-routing-key", false), header("x-client", true)],
        strip_lineage: false,
    });
    engine_of(&document)
}

/// A ring that places a head of token ids at its length in tokens. Its walk
/// reaches one worker from the place of a head of sixteen tokens and
/// another from every other place.
#[derive(Debug)]
struct ByDepth {
    shallowest: String,
    deeper: String,
}

impl LentRing for ByDepth {
    fn position(&self, key: &str) -> u64 {
        panic!("the ring was asked where {key} stands");
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        match head {
            Head::Tokens(ids) => Some(ids.len() as u64),
            Head::Text(_) => None,
        }
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let from_sixteen = position == 16;
        let _ = take(if from_sixteen {
            &self.shallowest
        } else {
            &self.deeper
        });
    }
}

/// A lender that answers a script: for each head it is asked about, in the
/// order of its asks, the place the script holds, and `None` where the
/// script says so. It counts its asks, and is asked nothing else.
#[derive(Debug)]
struct Scripted(&'static [Option<u64>], AtomicU32);

impl LentRing for Scripted {
    fn position(&self, key: &str) -> u64 {
        panic!("the ring was asked where {key} stands");
    }

    fn head_position(&self, _: Head<'_>) -> Option<u64> {
        let at = self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.get(at as usize).copied().flatten()
    }

    fn walk(&self, position: u64, _: &mut dyn FnMut(&str) -> bool) {
        panic!("the ring was walked from {position}");
    }
}

/// A dispatch settles its decision's claim across a new document: the
/// session table is the engine's own, and a document put in force between
/// a decision and its dispatch reads the pin that dispatch wrote. A fresh
/// key whose claimed worker the route's rule refuses is sent to the
/// decision's next worker, and the key's request with no rule lands there:
/// a single request, and a pair's decode.
#[test]
fn a_dispatch_across_a_new_document_settles_its_decisions_claim() {
    let stated = |pair, generation| {
        let mut document = PolicySpec::preset("manual").expect("a built-in preset");
        let mut rule = document.bindings[0]
            .sessions
            .take()
            .expect("the stock rule");
        rule.assignment = Assignment::MinLoad;
        if pair {
            document = PolicySpec::pair_presets("manual", "manual").expect("the pair presets");
        }
        document.generation = generation;
        document.bindings[0].sessions = Some(rule);
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let members = PublishedList::new(pool(4));
    let decodes = (0..4).map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL));
    let decodes = PublishedList::new(decodes.collect());
    let classes = PartnerClasses::number(4, [&[0_u32, 1, 2, 3][..]; 4]);
    let pair = PairLists {
        decode: decodes.lent(),
        ..pair_of(&members, &classes)
    };
    for (lists, sides) in [
        (WorkerLists::Single(members.lent()), &[Side::Single][..]),
        (WorkerLists::Pair(pair), &[Side::Prefill, Side::Decode][..]),
    ] {
        let paired = sides.len() == 2;
        let engine = hosted(InProcess::new(stated(paired, 1)));
        // The rule refuses the worker it is asked of first, a pair's decode
        // first: the one the decision claimed. A second document is in
        // force from that call.
        let refused = Cell::new(None);
        let admits = |worker: &dyn Worker| {
            let id = worker_id(worker);
            if refused.get().is_none() && worker.url().contains("decode") == paired {
                refused.set(Some(id));
                let applied = engine.apply(stated(paired, 2));
                applied.expect("a number above the standing one");
            }
            refused.get() != Some(id)
        };
        let headers = [(MANUAL_SESSION_HEADER, "talk-across")];
        let keyed = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, lists)
        };
        let ruled = PlaceRequest {
            admits: Some(&admits),
            ..keyed
        };
        let mut placement = engine.place(&ruled).expect("a placement");
        for side in sides {
            placement.sent(*side);
        }
        let sent = (placement.id(), placement.decode_id());
        assert!(refused
            .get()
            .is_some_and(|id| id != sent.0 && Some(id) != sent.1));
        let next = engine.place(&keyed).expect("a placement");
        assert_eq!(
            (next.id(), next.decode_id()),
            sent,
            "the pins are the workers the request was sent to"
        );
    }
}
