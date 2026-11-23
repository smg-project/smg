mod pair;

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashMap},
    ptr,
    sync::atomic::{self, AtomicU32},
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
        BoundedLoadParams, ConsistentHashParams, FilterSpec, KeySourceSpec, OverloadSpec,
        PolicySpec, RoutingKeySpec, ScoreStage, ShadowSpec, MANUAL_SESSION_HEADER,
    },
    engine::{
        decision::{Queue, ShadowKind, Shed, Target, Targets, Trace},
        decision_engine::Decided,
        framework::{Filter, Picker, Place, TieBand},
        records::worker_id,
        view::{fingerprint, Candidate, FixedFacts, HashKind},
        EncodeLists, PairAdmission, PairLists, PartnerClasses,
    },
    policies::consistent_hashing::InOrder as Lent,
    stages::{pickers::push_drawn, registry::StageRegistry, Lifecycle},
    tracking::lease::{EventSet, Lease, Outcome, Side},
};

const MODEL: &str = "model-a";

fn engine_of(document: &PolicySpec) -> PolicyEngine {
    PolicyEngine::new(AppliedSpec::validate(document).expect("the document validates"))
}

/// An engine as its host holds it: for the life of the process.
fn hosted(engine: PolicyEngine) -> &'static PolicyEngine {
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
fn in_order(panics: bool) -> (&'static PolicyEngine, Heard) {
    let (engine, heard, _) = in_order_budgets(panics);
    (engine, heard)
}

/// [`in_order`], and the budget each of its decisions was given.
fn in_order_budgets(panics: bool) -> (&'static PolicyEngine, Heard, Budgets) {
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
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = PolicyEngine::new(spec.expect("the stages hold the picker"));
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
fn pair_of<'a>(members: &'a PoolMembers, classes: &'a PartnerClasses) -> PairLists<'a> {
    PairLists {
        prefill: members,
        decode: members,
        classes,
        admission: &Open,
        prefill_places: None,
        decode_places: None,
    }
}

/// An encode request's three lists: `pair` and its prefills as the
/// encoders, with nothing said of them.
fn three_of(pair: PairLists<'_>) -> WorkerLists<'_> {
    WorkerLists::Encode(EncodeLists {
        pair,
        encoders: pair.prefill,
        facts: &[],
        prefill_classes: &[],
    })
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

/// A document whose binding reads a routing value from a header, then from
/// the request's own id, and a client's value from a second header.
fn keyed() -> PolicyEngine {
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

/// The request's routing value is the first source of its binding that
/// holds a value: the header ahead of the request's own id. A header the
/// document shows no stage leaves as its fingerprint alone, and the
/// request's id in the clear. The client's value is read only where no
/// source holds a routing value. Under a document that names no source a
/// request carries neither, whatever its headers.
#[test]
fn the_routing_value_and_the_clients_value_come_from_the_documents_sources() {
    let engine = keyed();
    let keys = |headers: &dyn HeaderRead, request_id| {
        let request = PlaceRequest {
            headers,
            request_id,
            ..PlaceRequest::of(MODEL, WorkerLists::None)
        };
        let made = Made::new(&engine, &request, &[]);
        let facts = made.make(engine.spec()).facts;
        let clear = facts.session_key.plain.is_some();
        assert_eq!(
            facts.session_counted, clear,
            "a key in the clear is counted"
        );
        (facts.session_key.hash, clear, facts.client_key)
    };
    let both = [("x-conversation", "talk-7"), ("x-client", "10.0.0.7")];
    assert_eq!(
        keys(&both, Some("request-1")),
        (fingerprint(b"talk-7"), false, 0),
        "the header is the first source"
    );
    let client = [("x-client", "10.0.0.7")];
    assert_eq!(
        keys(&client, Some("request-1")),
        (fingerprint(b"request-1"), true, 0),
        "then the request's own id"
    );
    assert_eq!(
        keys(&client, None),
        (0, false, fingerprint(b"10.0.0.7")),
        "the client's value where no source holds a routing value"
    );
    assert_eq!(keys(&(), None), (0, false, 0));

    let unkeyed = preset("round_robin");
    let request = PlaceRequest {
        headers: &both,
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&unkeyed, &request, &[]);
    let facts = made.make(unkeyed.spec()).facts;
    assert_eq!((facts.session_key, facts.client_key), (Identity::NONE, 0));
}

/// What a caller states is the request's fact as stated: the attempt's
/// number beside the workers tried, which need not count alike, its own
/// id, and each plain field. The targets this call's earlier decisions
/// missed are left out of the next one. A request with a pair's two lists
/// is made at its prefill. A prompt's `Debug` form shows sizes alone.
#[test]
fn a_request_is_made_of_what_its_caller_states() {
    let engine = preset("round_robin");
    let (tried, missed) = ([WorkerId(5), WorkerId(9)], [WorkerId(3)]);
    let media = MediaCounts {
        images: 2,
        ..MediaCounts::NONE
    };
    let request = PlaceRequest {
        request_id: Some("request-1"),
        tenant: Some("tenant-a"),
        priority: 3,
        streaming: true,
        max_output_tokens: 256,
        media,
        attempt: 1,
        tried: &tried,
        budget_micros: 5_000,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &missed);
    let Request {
        id,
        facts,
        constraints,
    } = made.make(engine.spec());
    assert_eq!(id, b"request-1");
    assert_eq!((facts.model, facts.phase), (MODEL, Phase::Single));
    assert_eq!((facts.attempt, facts.tried), (1, &tried[..]));
    assert_eq!(facts.tenant, Identity::plain("tenant-a"));
    assert_eq!((facts.priority, facts.streaming), (3, true));
    assert_eq!((facts.max_output_tokens, facts.budget_micros), (256, 5_000));
    assert_eq!(facts.media, media);
    assert_eq!(constraints.excluded, &missed[..]);
    assert!(constraints.demands.is_none() && !constraints.decode_later);
    // A request with no id bears an empty one.
    let bare = PlaceRequest::of(MODEL, WorkerLists::None);
    assert!(Made::new(&engine, &bare, &[])
        .make(engine.spec())
        .id
        .is_empty());
    // A request with a pair's two lists is made, and says it is plain, at
    // its prefill.
    let list = pool(2);
    let members = engine.pool_members(&list);
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let paired = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let made = Made::new(&engine, &paired, &[]);
    assert_eq!(made.make(engine.spec()).facts.phase, Phase::Prefill);
    assert_eq!(made.plain().map(|plain| plain.phase), Some(Phase::Prefill));
    // And so is a request with an encode request's three.
    let three = PlaceRequest::of(MODEL, three_of(pair));
    let made = Made::new(&engine, &three, &[]);
    assert_eq!(made.make(engine.spec()).facts.phase, Phase::Prefill);
    // A client's tokens and text reach no log line.
    let prompt = Prompt {
        tokens: Some(&[7, 8, 9]),
        text: Some("secret words"),
        seeds: HashSeeds::PLAIN,
    };
    let shown = format!("{prompt:?}");
    assert!(shown.contains("tokens: Some(3)"), "{shown}");
    assert!(shown.contains("text: Some(12)"), "{shown}");
    assert!(!shown.contains("secret"), "{shown}");
}

/// Each making draws a seed no other making of the thread drew, and a
/// second thread draws others.
#[test]
fn each_making_draws_a_seed_of_its_own() {
    let draw = || {
        let engine = preset("random");
        let request = PlaceRequest::of(MODEL, WorkerLists::None);
        let made = Made::new(&engine, &request, &[]);
        (0..64)
            .map(|_| made.make(engine.spec()).facts.seed)
            .collect::<BTreeSet<u64>>()
    };
    let here = draw();
    assert_eq!(here.len(), 64, "no seed twice");
    let there = thread::spawn(draw).join().expect("the thread ends");
    assert_eq!(there.len(), 64);
    assert!(here.is_disjoint(&there), "two threads draw apart");
}

/// The prompt's blocks are the ones the engine hashes of it under the
/// seeds its caller states, token ids ahead of a text, and they are hashed
/// once: a second making borrows the first one's blocks. Token ids alone
/// and a text alone are hashed too. A request with no prompt states no
/// block and no size.
///
/// A text alone states its size as its bytes over four. The classic
/// router prices a request with no token ids at its mean prefill
/// (`model_gateway/src/policies/least_load.rs:810-816`).
#[test]
fn the_prompts_blocks_are_hashed_under_its_partition_once() {
    let engine = preset("cache_aware");
    let (tokens, text) = ((0..600).collect::<Vec<u32>>(), "t".repeat(4_000));
    let seeds = HashSeeds {
        event: 11,
        partition: 13,
    };
    let stated = |tokens, text| PlaceRequest {
        prompt: Prompt {
            tokens,
            text,
            seeds,
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };

    let request = stated(Some(&tokens[..]), Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let (first, again) = (
        made.make(engine.spec()).facts,
        made.make(engine.spec()).facts,
    );
    let mut expected = RequestFacts::default();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, seeds);
    hashes.write_to(&mut expected);
    assert!(!first.block_hashes.is_empty());
    assert_eq!(first.block_hashes, expected.block_hashes);
    assert_eq!(
        (first.hash_kind, first.block_size),
        (HashKind::Token, expected.block_size)
    );
    assert_eq!((first.namespace_seed, first.prompt_tokens), (13, 600));
    assert!(
        ptr::eq(first.block_hashes, again.block_hashes),
        "hashed once"
    );

    // Token ids alone, as a tokenized request states its prompt.
    let request = stated(Some(&tokens[..]), None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.spec()).facts;
    assert_eq!(facts.block_hashes, expected.block_hashes);
    assert_eq!(
        (facts.hash_kind, facts.prompt_tokens),
        (HashKind::Token, 600)
    );

    let request = stated(None, Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.spec()).facts;
    assert_eq!(
        (facts.hash_kind, facts.prompt_tokens),
        (HashKind::Text, 1_000)
    );
    assert!(!facts.block_hashes.is_empty());

    let request = stated(None, None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.spec()).facts;
    assert!(facts.block_hashes.is_empty());
    assert_eq!(facts.prompt_tokens, 0);
}

/// A first attempt that names nothing tried or missed, demands nothing,
/// carries no encode item and asks no decode again says it is plain, with
/// the size its request would state. The engine then answers it a place where one pick decides its
/// place: no clock is read for it. Each fact that asks more than a worker
/// of the list takes the request off that path. A decode asked again
/// does not: over one list the word asks nothing.
#[test]
fn a_plain_request_says_so_and_is_answered_a_place() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let tokens = [7_u32; 40];
    let plain = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let said = Made::new(&engine, &plain, &[]).plain();
    assert_eq!(
        said,
        Some(Plain {
            phase: Phase::Single,
            wires: WireModes::NONE,
            prompt_tokens: 40,
        })
    );
    // A text alone, as a request that is not tokenized states its prompt:
    // the size it says is the text's bytes over four.
    let text = "t".repeat(400);
    let worded = PlaceRequest {
        prompt: Prompt {
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, plain.lists)
    };
    let short = Made::new(&engine, &worded, &[]).plain();
    assert_eq!(short.expect("a plain request").prompt_tokens, 100);
    let read = Cell::new(0);
    let clock = || {
        read.set(read.get() + 1);
        Instant::now()
    };
    let decided = engine.decide_at(&Made::new(&engine, &plain, &[]), clock);
    assert!(matches!(decided, Ok(Decided::Alone(pick)) if pick.place < 4));
    assert_eq!(read.get(), 0, "a place dates nothing");

    let tried = [WorkerId(1)];
    let demands = Demands {
        vision: true,
        ..Demands::default()
    };
    let more = [
        PlaceRequest {
            attempt: 1,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            tried: &tried,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            demands,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            encode: Some(EncodeItems::default()),
            ..PlaceRequest::of(MODEL, plain.lists)
        },
    ];
    for request in &more {
        assert_eq!(Made::new(&engine, request, &[]).plain(), None);
    }
    assert_eq!(
        Made::new(&engine, &plain, &tried).plain(),
        None,
        "a missed target"
    );
    // A decode asked again is a pair's word: over one list it asks nothing.
    let again = PlaceRequest {
        decode_again: true,
        ..plain
    };
    assert_eq!(Made::new(&engine, &again, &[]).plain(), said);
}

/// A request states the later decode where its caller asks the decode
/// again and the document picks the decode after the prefill. Under a
/// document that picks both sides at once it states none, and so does a
/// caller that asks no decode again.
#[test]
fn the_later_decode_is_stated_where_the_document_picks_it_so() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    let at_once = engine_of(&document);
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let after = engine_of(&document);
    let later = |engine: &PolicyEngine, decode_again| {
        let request = PlaceRequest {
            decode_again,
            ..PlaceRequest::of(MODEL, WorkerLists::None)
        };
        let made = Made::new(engine, &request, &[]);
        made.make(engine.spec()).constraints.decode_later
    };
    assert!(later(&after, true));
    assert!(!later(&after, false));
    assert!(!later(&at_once, true));
}

/// A worker can be sent to while it is ready and its circuit breaker lets
/// requests through: either one alone holds a send back. The overload veto
/// does not: the decision that picked the worker read it.
#[test]
fn a_worker_is_ready_by_its_status_and_its_breaker() {
    let list = pool(2);
    let (worker, tripped) = (&list[0], &list[1]);
    assert!(ready(worker.as_ref()));
    worker.set_overloaded(true);
    assert!(
        ready(worker.as_ref()),
        "the overload veto is the decision's"
    );
    worker.set_status(WorkerStatus::NotReady);
    assert!(!ready(worker.as_ref()));

    for _ in 0..64 {
        if !tripped.circuit_breaker_can_execute() {
            break;
        }
        tripped.record_circuit_breaker_outcome(false);
    }
    assert!(tripped.routing_state().healthy, "still ready by its status");
    assert!(
        !ready(tripped.as_ref()),
        "an open breaker holds a send back"
    );
}

/// A whole decision's pick is handed on. A shed hands on its reason, the
/// stage that left nobody and its retry hint, and a queue ticket is answered
/// as queued.
#[test]
fn a_decision_that_placed_nothing_says_why() {
    assert!(picked(&Decision::Pick(Pick::default())).is_ok());
    let shed = Shed {
        reason: ShedReason::Saturated,
        emptied_by: "has-capacity",
        retry_after_secs: 7,
        wait_allowed: false,
        trace: Trace::default(),
    };
    let said = Unplaced::Shed {
        reason: ShedReason::Saturated,
        emptied_by: "has-capacity",
        retry_after_secs: 7,
    };
    assert_eq!(picked(&Decision::Shed(shed)).err(), Some(said));
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

/// An HTTP server's header map is read as it stands. The map keeps its
/// names lowercase, as a document writes them, whatever case the client
/// sent. A value that is not text is handed as its bytes, and the source
/// that reads it passes it over: the next source holds the routing value.
#[test]
fn a_header_map_hands_the_documents_sources_their_values() {
    let engine = keyed();
    let keys = |headers: &HeaderMap| {
        let request = PlaceRequest {
            headers,
            request_id: Some("request-1"),
            ..PlaceRequest::of(MODEL, WorkerLists::None)
        };
        let made = Made::new(&engine, &request, &[]);
        let key = made.make(engine.spec()).facts.session_key;
        (key.hash, key.plain.is_some())
    };
    let name = HeaderName::from_bytes(b"X-Conversation").expect("a header's name");
    let mut headers = HeaderMap::new();
    headers.insert(name.clone(), HeaderValue::from_static("talk-7"));
    assert_eq!(headers.header("x-conversation"), Some(&b"talk-7"[..]));
    assert_eq!(headers.header("x-client"), None);
    assert_eq!(keys(&headers), (fingerprint(b"talk-7"), false));

    let bytes = HeaderValue::from_bytes(&[0xFF, 0xFE]).expect("a header's value");
    headers.insert(name, bytes);
    assert_eq!(headers.header("x-conversation"), Some(&[0xFF, 0xFE][..]));
    assert_eq!(keys(&headers), (fingerprint(b"request-1"), true));
}

/// A plain request is placed on the worker at the place the engine picked,
/// on the pick's own read of it: a worker that is not ready, which the pick
/// passes over, is never the answer. Nothing is owed of such a placement,
/// no routing value is derived for it, and it names the profile that
/// picked by its label.
#[test]
fn a_plain_request_is_placed_on_the_worker_at_its_place() {
    let engine = hosted(preset("round_robin"));
    let alone = matches!(engine.place_for(MODEL, Phase::Single), Place::Alone(_));
    assert!(alone, "one pick decides the place");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let headers = [("x-conversation", "talk-7")];
    let request = PlaceRequest {
        headers: &headers,
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
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
fn spared(name: &str) -> &'static PolicyEngine {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let spare = document.profiles[name].clone();
    document.profiles.insert("a_spare".to_owned(), spare);
    hosted(engine_of(&document))
}

/// One request of `engine` placed over a list of four workers, as the
/// request keeps it.
fn placed(engine: &'static PolicyEngine) -> Placement<'static> {
    let list = pool(4);
    let members = engine.pool_members(&list);
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
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
    let alone_at = |engine: &PolicyEngine| match engine.place_for(MODEL, Phase::Single) {
        Place::Alone(profile) => Some(profile.0),
        _ => None,
    };
    let (turns, again) = (spared("round_robin"), spared("round_robin"));
    let (passes, draws) = (spared("passthrough"), spared("random"));
    let at = [alone_at(turns), alone_at(passes), alone_at(draws)];
    assert_eq!(at, [Some(1), Some(1), None]);

    let first = placed(turns);
    assert_eq!(first.profile, "round_robin");
    let same = ptr::eq(first.profile, placed(again).profile);
    assert!(same, "one string a name");
    // The other document's second profile has another name.
    assert_eq!(placed(passes).profile, "passthrough");
    // A whole decision's placement names its profile the same way.
    assert_eq!(placed(draws).profile, "random");
}

/// Placing a request asks the process's table of names for nothing: the
/// label a placement holds is read from the engine's own list, which the
/// engine made at its start. A place's placement, and a whole decision's.
#[test]
fn placing_a_request_asks_the_table_of_names_for_nothing() {
    let asked = || crate::names::ASKED.with(Cell::get);
    for name in ["round_robin", "random"] {
        let engine = hosted(preset(name));
        let list = pool(4);
        let members = engine.pool_members(&list);
        let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
        let before = asked();
        for _ in 0..8 {
            let placement = engine.place(&request).expect("a placement");
            assert_eq!(placement.profile, name);
        }
        assert_eq!(asked(), before, "{name}");
    }
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
    let members = engine.pool_members_with_fixed_facts(&list, |_| {
        stated += 1;
        FixedFacts {
            id: Some(WorkerId(stated)),
            ..FixedFacts::default()
        }
    });
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
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
    let members = engine.pool_members(&list);
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
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    for turn in 0..16 {
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[2]), "turn {turn}");
    }
    task.abort();
}

/// A retry states the worker it tried: it is decided whole and placed on
/// another worker of its list.
///
/// The engine leaves out the one worker a retry names, by its id. The
/// classic router leaves out every rank at that worker's base address,
/// which the ranks of a data-parallel worker share
/// (`model_gateway/src/routers/common/placement.rs:1174-1179`, used at
/// `:568`). Over workers of one rank each, as here, the two agree.
#[test]
fn a_retry_is_placed_past_the_worker_it_tried() {
    let engine = hosted(preset("round_robin"));
    let list = pool(2);
    let members = engine.pool_members(&list);
    let tried = [worker_id(list[0].as_ref())];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    for _ in 0..4 {
        let placement = engine.place(&retry).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[1]));
        assert_eq!(placement.id(), worker_id(list[1].as_ref()));
    }
}

/// A whole decision's targets are asked best first, and the first that can
/// be sent to is taken: ready, and let through by its circuit breaker. The
/// lease issued on the pick's first target says its dispatch on the target
/// taken, under that target's index.
///
/// The classic router sends to no other worker either. Its selection keeps
/// a worker that is available
/// (`model_gateway/src/routers/common/placement.rs:536-539`, over
/// `crates/worker/src/identity/basic.rs:438-443`), and where every worker
/// of a pool is vetoed it still takes one that is ready with its circuit
/// closed (`model_gateway/src/routers/common/overload.rs:59-61`, `:145`).
#[test]
fn the_first_target_that_can_be_sent_to_is_taken() {
    let (engine, heard) = in_order(false);
    let list = pool(4);
    let members = engine.pool_members(&list);
    for worker in &list[..2] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    let mut placement = engine.place(&request).expect("a placement");
    let (first, taken) = (worker_id(list[0].as_ref()), worker_id(list[2].as_ref()));
    assert!(Arc::ptr_eq(&placement.worker, &list[2]));
    assert_eq!(placement.id(), taken);
    assert!(placement.owes_events());
    placement.sent(Side::Single);
    assert_eq!(
        heard.lock().as_slice(),
        [
            ("issued at 0".to_owned(), first),
            ("dispatched at 2".to_owned(), taken)
        ]
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Where none of a decision's targets can be sent to, the lease its pick
/// was issued is given up by name at once, and the call decides again.
/// Over a list that holds no other worker the next decision names the same
/// workers, and after three decisions, each given up by name, the request
/// has no worker. Under a document none of whose stages takes a word, no
/// lease was issued and nothing is said.
///
/// The classic router answers no worker there too: its selection misses
/// where no candidate is available
/// (`model_gateway/src/routers/common/placement.rs:471-472`).
#[test]
fn a_decision_no_target_of_which_can_be_sent_to_is_given_up_by_name() {
    let (engine, heard) = in_order(false);
    let list = pool(3);
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = engine.pool_members(&list);
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    let first = worker_id(list[0].as_ref());
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(first), given_up(first), given_up(first)].concat()
    );

    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let profile = document.profiles.get_mut("round_robin");
    let filters = &mut profile.expect("the preset's profile").filter;
    filters.retain(|filter| *filter != FilterSpec::Name("healthy".to_owned()));
    let quiet = hosted(engine_of(&document));
    let members = quiet.pool_members(&list);
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    assert_eq!(quiet.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!((quiet.faults(), quiet.unheard()), (0, 0));
}

/// What a picker hears of a decision none of whose targets can be sent to:
/// its issue and its release by name, on the decision's first target.
fn given_up(first: WorkerId) -> [(String, WorkerId); 2] {
    [
        ("issued at 0".to_owned(), first),
        ("released CommitFailed".to_owned(), first),
    ]
}

/// Where no target of a decision can be sent to, the call decides again and
/// leaves those targets out: the second decision names the workers behind
/// the first one's, the third the workers behind the second's, and the
/// first target that can be sent to is taken. A decision that takes no
/// target is given up by name before the next is made. A worker the
/// request's failed attempt was sent to is left out beside them.
///
/// The classic router has no pick to take back: it narrows a request's
/// workers to the available ones ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:536-539`, `:549-553`),
/// and runs its selection once more where a worker came free (`:457-482`).
#[test]
fn a_call_decides_again_past_the_targets_it_could_not_send_to() {
    let (engine, heard) = in_order(false);
    let list = pool(14);
    for worker in &list[..8] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    let mut placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[8]));
    placement.sent(Side::Single);
    let mut said = [given_up(id(0)), given_up(id(4))].concat();
    said.push(("issued at 0".to_owned(), id(8)));
    said.push(("dispatched at 0".to_owned(), id(8)));
    assert_eq!(heard.lock().as_slice(), said);
    drop(placement);
    heard.lock().clear();

    // A retry that tried the ninth worker is placed on the tenth.
    let tried = [id(8)];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[9]));
    assert_eq!(
        heard.lock()[..4],
        [given_up(id(0)), given_up(id(4))].concat()
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A call makes three decisions, and a fourth only where its third found
/// the workers a route's rule takes. Over a list none of whose first
/// twelve workers can be sent to, the three decisions name them four by
/// four, each is given up by name, and the request has no worker. No fourth
/// decision names the workers behind them.
///
/// The classic router bounds its selection too: one pass, and one more
/// where a worker came free
/// (`model_gateway/src/routers/common/placement.rs:457-482`).
#[test]
fn a_call_makes_three_decisions_at_most() {
    let (engine, heard) = in_order(false);
    let list = pool(16);
    for worker in &list[..12] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(id(0)), given_up(id(4)), given_up(id(8))].concat()
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The route's own rule is asked of the worker picked: a request whose
/// first pick it admits pays one call of it, on the worker the request is
/// placed on, and the list is not asked.
///
/// The classic router asks a request's rule of the candidates ahead of its
/// policy (`model_gateway/src/routers/common/placement.rs:504-509`, over
/// `:361-370`). Both send to no worker the rule refuses.
#[test]
fn the_routes_rule_is_asked_of_the_pick_alone_while_it_admits_it() {
    let engine = hosted(preset("round_robin"));
    let list = pool(8);
    let members = engine.pool_members(&list);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        true
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    for turn in 1..=8 {
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(asked.borrow().len(), turn, "one call a request");
        assert_eq!(asked.borrow().last(), Some(&placement.id()));
    }
}

/// No request is placed on a worker the route's rule refuses. Behind the
/// first pick it refuses, the rule is asked of every worker of the list
/// once, in the list's order, and the call's next decision is held to the
/// workers it takes. That holds for a place's worker, and for a whole
/// decision's targets, whose walk asks the rule of a worker that is ready
/// and of no other. The worker the request is placed on is asked once
/// more, as the pick it is.
///
/// The classic router sends to no worker its request's rule refuses either:
/// it narrows the candidates by the rule ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:504-509`).
#[test]
fn a_worker_the_routes_rule_refuses_is_passed_over() {
    // A place: the list's first worker is the pick of every plain request,
    // and the rule takes the sixth alone. The rule is asked of the place,
    // of the six workers of the list, and of the sixth as the pick of the
    // call's second decision, which is held to it.
    let engine = hosted(preset("passthrough"));
    let list = pool(6);
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[5]));
    let calls: Vec<_> = [0].into_iter().chain(0..6).chain([5]).map(id).collect();
    assert_eq!(*asked.borrow(), calls);

    // A whole decision: the first worker is not ready, and the rule takes
    // the sixth alone.
    let (engine, heard) = in_order(false);
    let list = pool(8);
    list[0].set_status(WorkerStatus::NotReady);
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let mut placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[5]));
    let calls: Vec<_> = (1..4).chain(0..8).chain([5]).map(id).collect();
    assert_eq!(*asked.borrow(), calls);
    placement.sent(Side::Single);
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), id(5)));
    said.push(("dispatched at 0".to_owned(), id(5)));
    assert_eq!(heard.lock().as_slice(), said);
}

/// The one worker of a long list the route's rule takes is found by the
/// call's second decision: the first names four others, the rule refuses
/// them, the rule is asked of the sixteen workers of the list, and the
/// second decision is held to the worker it takes.
///
/// The classic router places such a request too: it narrows the list by
/// the rule ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:504-509`).
#[test]
fn the_one_worker_a_rule_takes_in_a_long_list_is_found_by_the_second_decision() {
    let (engine, heard) = in_order(false);
    let list = pool(16);
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(14);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[14]));
    assert_eq!(calls.get(), 4 + 16 + 1, "four picks, the list, the pick");
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), far));
    assert_eq!(heard.lock().as_slice(), said);
}

/// The call's later decisions are made among the workers the rule takes,
/// so the rule's one worker is placed where a decision over the whole list
/// never names it. A retry that tried it is handed it again: the filter of
/// tried workers is soft. Over its overload thresholds, it is answered by
/// the binding's rule for a list wholly over them: taken where the binding
/// falls back, shed as saturated where the binding sheds.
///
/// The classic router answers the same three ways: it narrows by the rule
/// first (`model_gateway/src/routers/common/placement.rs:504-509`), steps
/// the tried workers aside while another is left (`:556-576`), and falls
/// back over a pool that is all overloaded (`:577-579`).
#[test]
fn the_rules_only_worker_is_placed_though_tried_or_overloaded() {
    let engine = hosted(preset("passthrough"));
    let list = pool(5);
    let members = engine.pool_members(&list);
    let only = worker_id(list[4].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == only;
    let tried = [only];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[4]), "tried, and alone");
    drop(placement);

    list[4].set_overloaded(true);
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&first).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[4]), "the fallback");
    drop(placement);

    let mut document = PolicySpec::preset("passthrough").expect("a built-in preset");
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let members = shedding.pool_members(&list);
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let Some(Unplaced::Shed { reason, .. }) = shedding.place(&first).err() else {
        panic!("a shed");
    };
    assert_eq!(reason, ShedReason::Saturated);
}

/// A request whose route's rule takes no worker of its list is shed with no
/// endpoint under the rule's name, behind one decision: the rule is asked
/// of that decision's picks, then once of every worker of the list, ready
/// or not, and no second decision is made.
///
/// The classic gRPC route answers a refusal of its own where no worker can
/// take a request's images, by what the workers can do and not by their
/// health
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:1465-1472`).
#[test]
fn a_rule_that_takes_no_worker_is_shed_under_its_name() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    list[5].set_status(WorkerStatus::NotReady);
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let calls: Vec<_> = (0..4).chain(0..6).map(id).collect();
    assert_eq!(*asked.borrow(), calls);
    assert_eq!(heard.lock().as_slice(), given_up(id(0)), "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A rule that takes workers none of which is ready is the decision's to
/// answer: held to them, it sheds by its own filter and names it. The rule
/// is asked of a worker that is not ready too, which tells this request
/// from one whose rule takes no worker.
#[test]
fn a_rule_whose_workers_are_not_ready_is_answered_by_the_decision() {
    let engine = hosted(preset("round_robin"));
    let list = pool(4);
    list[3].set_status(WorkerStatus::NotReady);
    let members = engine.pool_members(&list);
    let last = worker_id(list[3].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == last;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "healthy"));
}

/// The places a call keeps are the list's places whose worker the rule
/// takes, ascending, each once, ready or not. Under a profile that draws,
/// sixty-four requests all land on workers the rule takes, and on more
/// than one of them.
#[test]
fn the_places_a_rule_takes_are_ascending_and_hold_every_later_pick() {
    let engine = hosted(preset("random"));
    let list = pool(64);
    list[3].set_status(WorkerStatus::NotReady);
    let members = engine.pool_members(&list);
    let thirds = || (0..64).step_by(3);
    let taken: BTreeSet<WorkerId> = thirds().map(|at| worker_id(list[at].as_ref())).collect();
    let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let mut call = Call::default();
    call.missed.push(worker_id(list[1].as_ref()));
    hold_to_rule(&request, Some(&members), &mut call).expect("the rule takes workers");
    let places: Vec<u32> = thirds().map(|at| at as u32).collect();
    assert_eq!(call.admitted.as_deref(), Some(&places[..]));

    let mut placed = BTreeSet::new();
    for turn in 0..64 {
        let placement = engine.place(&request).expect("a placement");
        assert!(taken.contains(&placement.id()), "turn {turn}");
        placed.insert(placement.id());
    }
    assert!(placed.len() > 1, "more than one of the rule's workers");
}

/// A keyed request whose owner the rule refuses is placed among the
/// workers the rule takes, and the key's next request under the same rule
/// lands on the same worker. A key that held one worker keeps it: its
/// request with no rule lands there as before. Under the sessions rule the
/// first decision names the key's pin alone, so every such request asks
/// the rule of the list: one call for the pin, sixteen for the list, one
/// for the pick. Under the hash ring the first decision names the owner
/// and three workers behind it, and the list is asked where the rule
/// refuses all four.
#[test]
fn a_keyed_request_whose_owner_the_rule_refuses_stays_on_one_worker() {
    for policy in ["manual", "consistent_hashing"] {
        let engine = hosted(preset(policy));
        let list = pool(16);
        let members = engine.pool_members(&list);
        let mut by_list = 0;
        for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
            let headers = [(MANUAL_SESSION_HEADER, key)];
            let keyed = PlaceRequest {
                headers: &headers,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
            };
            let owner = engine.place(&keyed).expect("a placement").id();
            let calls = Cell::new(0);
            let admits = |worker: &dyn Worker| {
                calls.set(calls.get() + 1);
                let id = worker_id(worker);
                id != owner && id.0 % 2 == owner.0 % 2
            };
            let ruled = PlaceRequest {
                admits: Some(&admits),
                ..keyed
            };
            let first = engine.place(&ruled).expect("a placement").id();
            assert!(first != owner && first.0 % 2 == owner.0 % 2, "{policy}");
            by_list += usize::from(calls.get() > 16);
            if policy == "manual" {
                assert_eq!(calls.get(), 1 + 16 + 1, "{key}");
            }
            let again = engine.place(&ruled).expect("a placement").id();
            assert_eq!(again, first, "{policy}, {key}");
            let unruled = engine.place(&keyed).expect("a placement").id();
            assert_eq!(unruled, owner, "{policy}, {key}: the key keeps its owner");
        }
        assert!(by_list > 0, "{policy}: no key asked the rule of the list");
    }
}

/// A call's first decision is given the request's whole budget, and a later
/// one what the call has left of it. A request none of whose budget is left
/// is shed under `budget`, with no further decision. A request that states
/// no budget is given none, and no instant is kept for it.
#[test]
fn a_later_decision_is_given_what_the_call_has_left_of_the_budget() {
    let slow = |list: &Arc<[Arc<dyn Worker>]>| {
        let far = worker_id(list[6].as_ref());
        move |worker: &dyn Worker| {
            thread::sleep(Duration::from_millis(2));
            worker_id(worker) == far
        }
    };
    let (engine, heard, budgets) = in_order_budgets(false);
    let list = pool(8);
    let members = engine.pool_members(&list);
    let admits = slow(&list);
    let request = PlaceRequest {
        budget_micros: 10_000_000,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[6]));
    let given = budgets.lock().clone();
    assert_eq!((given.len(), given[0]), (2, 10_000_000));
    // Less by the rule's calls at least, and not a tenth of the budget
    // gone: what is left of it, and no other number.
    let left = 9_000_000..=10_000_000 - 2_000;
    assert!(left.contains(&given[1]), "{given:?}");
    drop(placement);
    heard.lock().clear();

    let short = PlaceRequest {
        budget_micros: 1_000,
        ..request
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::Deadline,
        emptied_by: "budget",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&short).err(), Some(shed));
    let first = worker_id(list[0].as_ref());
    assert_eq!(heard.lock().as_slice(), given_up(first), "one decision");

    let unbounded = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
    let mut call = Call::default();
    assert_eq!(call.left(&unbounded), Ok(0));
    assert!(
        call.began.is_none(),
        "no clock for a request with no budget"
    );
}

/// A rule that takes a worker when it is asked of the list and refuses it
/// as a pick places no request on it: the pick's answer stands. The call
/// asks the list no second time, names the worker again while no other is
/// left, and ends with no worker.
#[test]
fn a_rule_that_changes_its_answer_places_no_request_on_a_refused_pick() {
    let (engine, heard) = in_order(false);
    let list = pool(8);
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let calls = Cell::new(0);
    // Calls 5 to 12 are the list's walk: the sixth worker is taken there,
    // and nowhere else.
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        (5..=12).contains(&calls.get()) && worker_id(worker) == id(5)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 4 + 8 + 2, "the list is asked once");
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(id(0)), given_up(id(5)), given_up(id(5))].concat()
    );
}

/// A call whose third decision finds the workers its route's rule takes
/// makes a fourth among them, and no fifth. Two decisions name the eight
/// workers of the list that are not ready, the third names four ready ones
/// the rule refuses, the rule is asked of the list, and the fourth decision
/// is held to the one worker it takes. With three decisions a call, such a
/// request had no worker though its list held a ready one its rule takes.
///
/// The classic router places it: it narrows the list by the rule and to
/// the available workers ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:504-509`, `:536-553`).
#[test]
fn a_call_whose_third_decision_finds_the_rules_workers_makes_a_fourth() {
    let (engine, heard) = in_order(false);
    let list = pool(13);
    for worker in &list[..8] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = engine.pool_members(&list);
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(12);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[12]));
    assert_eq!(calls.get(), 4 + 13 + 1, "four picks, the list, the pick");
    let mut said = [given_up(id(0)), given_up(id(4)), given_up(id(8))].concat();
    said.push(("issued at 0".to_owned(), far));
    assert_eq!(heard.lock().as_slice(), said);
    drop(placement);
    heard.lock().clear();

    // A rule that takes that worker of the list and refuses it as a pick:
    // the fourth decision is the call's last.
    calls.set(0);
    let fickle = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        (5..=17).contains(&calls.get()) && worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&fickle),
        ..request
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 4 + 13 + 1, "the list is asked once");
    let four = [0, 4, 8, 12].map(|first| given_up(id(first)));
    assert_eq!(heard.lock().as_slice(), four.concat());
}

/// The rule is asked of the list while the call holds no decision: where it
/// panics there, the decision it refused was given up by name already, and
/// no lease is left behind.
#[test]
fn a_rule_that_panics_at_the_list_leaves_no_lease_behind() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    let members = engine.pool_members(&list);
    let first = worker_id(list[0].as_ref());
    let calls = Cell::new(0);
    // The four picks are refused, and the list's first worker panics.
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        assert!(calls.get() <= 4, "this rule panics");
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the rule's panic reaches the caller");
    assert_eq!(heard.lock().as_slice(), given_up(first));
    assert_eq!(engine.unheard(), 0);
}

/// An ask is made of what its call's earlier asks handed on: the targets
/// they could not send to are left out of its decision, so it is not plain,
/// and the prompt's blocks are the ones the call's first making hashed.
#[test]
fn an_ask_is_made_of_what_its_call_handed_on() {
    let engine = preset("cache_aware");
    let tokens = (0..600).collect::<Vec<u32>>();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let mut call = Call::default();
    let blocks = {
        let first = Made::asked(&engine, &request, &call);
        assert!(first.plain().is_some());
        let facts = first.make(engine.spec()).facts;
        assert!(!facts.block_hashes.is_empty());
        facts.block_hashes.as_ptr()
    };
    assert!(call.hashes.get().is_some(), "hashed in the call's cell");
    call.missed.push(WorkerId(3));
    let again = Made::asked(&engine, &request, &call);
    assert_eq!(again.plain(), None, "a missed target");
    let made = again.make(engine.spec());
    assert_eq!(made.constraints.excluded, [WorkerId(3)]);
    assert!(
        ptr::eq(made.facts.block_hashes.as_ptr(), blocks),
        "hashed once a call"
    );
}

/// A request with no worker is answered why. A decision's shed keeps its
/// reason and the stage that left nobody. What this call does not place
/// yet is declined by name: encode items. A demand no worker of the list
/// meets is shed under the demand's name. A decision that panics is
/// refused, and costs its request alone.
#[test]
fn a_request_with_no_worker_is_answered_why() {
    let engine = hosted(preset("round_robin"));
    let list = pool(2);
    let members = engine.pool_members(&list);
    let lists = WorkerLists::Single(&members);
    let shed = |reason, emptied_by| {
        Some(Unplaced::Shed {
            reason,
            emptied_by,
            retry_after_secs: 1,
        })
    };
    let none = PlaceRequest::of(MODEL, WorkerLists::None);
    assert_eq!(
        engine.place(&none).err(),
        shed(ShedReason::NoEndpoints, "none")
    );
    // An encode request's three lists, with no item stated, are declined
    // by name.
    let partners = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let three = PlaceRequest::of(MODEL, three_of(pair_of(&members, &partners)));
    assert_eq!(
        engine.place(&three).err(),
        shed(ShedReason::PolicyDeclined, "encode-lists")
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
    let members = faulty.pool_members(&list);
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(&members));
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
    let members = engine.pool_members(&list);
    let by_id = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&by_id).expect("a placement");
    assert_eq!(placement.routing_key, Some("request-1"));
    let again = engine.place(&by_id).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &again.worker));

    let headers = [("x-conversation", "talk-7")];
    let by_header = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&by_header).expect("a placement");
    assert_eq!(placement.routing_key, None);
    let kept: Placement<'static> = placement.kept();
    assert!(kept.routing_key.is_none());
}

/// A request whose caller lends its ring goes to the worker that ring's
/// walk reaches from the place of the value its document's sources derive:
/// the routing key, and with none the client's address. The gateway's own
/// policy takes the two in that order and places each by its own text
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 189 to 241 at
/// 882be6f4f). The caller states the ring and nothing else: the one call
/// reads the value's place with the request's keys. A request with neither
/// value states no place, and one that lends no ring is placed by the
/// stage's own ring. Under a bounded-load rule the ring stage reads the
/// lent ring alike: every worker here is idle, so the place's owner is
/// within the bound. A decision held to the workers a route's rule takes
/// walks the ring from the same place: the walk passes every other worker
/// over, and the first of the rule's workers it reaches is the place's
/// owner, whichever of them stands first in the list.
#[test]
fn a_request_that_lends_its_ring_goes_where_that_rings_walk_reaches() {
    lent_ring_places(hosted(preset("consistent_hashing")));
    let mut document = PolicySpec::preset("consistent_hashing").expect("a preset's name");
    for profile in document.profiles.values_mut() {
        profile.score[0] = ScoreStage::ConsistentHash(ConsistentHashParams {
            bounded_load: Some(BoundedLoadParams::default()),
            ..ConsistentHashParams::default()
        })
        .into();
    }
    lent_ring_places(hosted(engine_of(&document)));
}

/// The requests of
/// [`a_request_that_lends_its_ring_goes_where_that_rings_walk_reaches`]
/// under `engine`.
fn lent_ring_places(engine: &'static PolicyEngine) {
    let list = pool(8);
    let members = engine.pool_members(&list);
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let ring = Lent(order.iter().map(|&at| list[at].url().to_owned()).collect());
    // The worker the ring's walk reaches first from where `text` stands.
    let reached = |text: &str| order[(ring.position(text) % 8) as usize];
    let placed = |headers: &dyn HeaderRead, ring: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
        };
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        list.iter().position(worker).expect("a worker of the list")
    };
    // What the request's facts state: whether a ring, and the places.
    let stated = |headers: &dyn HeaderRead, ring: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.spec()).facts;
        (facts.ring.is_some(), facts.ring_positions.to_vec())
    };
    let keyed = [
        ("x-forwarded-for", "10.0.0.1"),
        ("x-smg-routing-key", "conv-1"),
    ];
    let client = [("x-forwarded-for", "10.0.0.1")];
    assert_eq!(placed(&keyed, Some(&ring)), reached("conv-1"));
    assert_eq!(placed(&client, Some(&ring)), reached("10.0.0.1"));
    assert_eq!(
        stated(&keyed, Some(&ring)),
        (true, vec![ring.position("conv-1")])
    );
    assert_eq!(stated(&(), Some(&ring)), (true, Vec::new()));
    let _ = placed(&(), Some(&ring));
    // No ring lent: the stage's own ring keeps the value on one worker.
    assert_eq!(stated(&keyed, None), (false, Vec::new()));
    assert_eq!(placed(&keyed, None), placed(&keyed, None));
    // A route's rule that takes two workers alone: the sixth and the eighth
    // the walk reaches from the key's place. It refuses the first decision's
    // four targets, so the next one is held to the rule's two places.
    let mut eighth_first = 0;
    for key in ["conv-1", "conv-2", "conv-3", "conv-4", "conv-5", "conv-6"] {
        let from = ring.position(key) % 8;
        let along = |steps: u64| order[((from + steps) % 8) as usize];
        let taken = [along(5), along(7)].map(|at| worker_id(list[at].as_ref()));
        let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
        let headers = [("x-smg-routing-key", key)];
        let ruled = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
        };
        let placement = engine.place(&ruled).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[along(5)]), "{key}");
        eighth_first += usize::from(along(7) < along(5));
    }
    assert!(eighth_first > 0, "no key has the eighth ahead in the list");
}

/// [`Lent`], with a count of the keys it was asked the place of.
#[derive(Debug)]
struct Asked(Lent, AtomicU32);

impl LentRing for Asked {
    fn position(&self, key: &str) -> u64 {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.position(key)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.0.walk(position, take);
    }
}

/// A request's key is placed on the lent ring once: a second making of the
/// request states the place the first took, and the ring is asked nothing.
#[test]
fn a_requests_key_is_asked_of_the_lent_ring_once() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(2);
    let members = engine.pool_members(&list);
    let order = vec![list[1].url().to_owned(), list[0].url().to_owned()];
    let ring = Asked(Lent(order), AtomicU32::new(0));
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let made = Made::new(engine, &request, &[]);
    let first = made.make(engine.spec()).facts.ring_positions.to_vec();
    let again = made.make(engine.spec()).facts.ring_positions.to_vec();
    assert_eq!(first, [ring.0.position("conv-1")]);
    assert_eq!(again, first);
    assert_eq!(ring.1.load(atomic::Ordering::Relaxed), 1);
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
    let members = engine.pool_members(&list);
    let ring = Lent(vec![list[0].url().to_owned()]);
    list[0].increment_load();
    list[0].increment_load();
    let headers = [("x-forwarded-for", "10.0.0.1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the ring's owner");
    // Agreed, another target, another outcome, skipped.
    let counted = ShadowKind::ALL.map(|kind| engine.shadowed(0, kind));
    assert_eq!(counted, [0, 1, 0, 0]);
}

/// A ring no decision may ask.
#[derive(Debug)]
struct Unasked;

impl LentRing for Unasked {
    fn position(&self, key: &str) -> u64 {
        panic!("the ring was asked where {key} stands");
    }

    fn walk(&self, position: u64, _: &mut dyn FnMut(&str) -> bool) {
        panic!("the ring was walked from {position}");
    }
}

/// Under `prefix_hash` a request through the one call is lent no ring. The
/// classic policy asks its ring for another text than the routing value's
/// own (`model_gateway/src/policies/prefix_hash.rs`, lines 223 to 229 and
/// 291 to 293 at 882be6f4f), and no place is read here by that rule. So
/// the stage's own ring places the request on the worker it has with no
/// ring lent, and the caller's ring is asked nothing. A document with no
/// ring stage asks it nothing either.
#[test]
fn a_request_under_the_prefix_form_is_lent_no_ring() {
    let tokens: Vec<u32> = (0..64).collect();
    let headers = [("x-smg-routing-key", "conv-1")];
    for name in ["prefix_hash", "random"] {
        let engine = hosted(preset(name));
        let list = pool(8);
        let members = engine.pool_members(&list);
        let request = |ring: Option<&'static dyn LentRing>| PlaceRequest {
            prompt: Prompt {
                tokens: Some(&tokens),
                ..Prompt::NONE
            },
            headers: &headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
        };
        let placed = |request: &PlaceRequest<'_>| {
            let placement = engine.place(request).expect("a placement");
            let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
            list.iter().position(worker).expect("a worker of the list")
        };
        let lent = request(Some(&Unasked));
        let made = Made::new(engine, &lent, &[]);
        let facts = made.make(engine.spec()).facts;
        assert!(facts.ring.is_none(), "{name}");
        assert!(facts.ring_positions.is_empty(), "{name}");
        let with = placed(&lent);
        if name == "prefix_hash" {
            assert_eq!(with, placed(&request(None)), "the stage's own ring");
        }
    }
}

/// A pair's request is lent the ring where a side's ring stage places a
/// routing value by its own text, the prefill's or the decode's: both
/// sides are handed the one ring, and each takes it by its own stage
/// ([`RequestFacts::ring`]). A pair with no such stage is lent none, and
/// neither is a pair with a side of the prefix form, which would take the
/// one place stated for its own: the caller's ring is asked nothing.
#[test]
fn a_pairs_request_is_lent_the_ring_a_side_of_it_places_by() {
    let list = pool(2);
    let headers = [("x-smg-routing-key", "conv-1")];
    let ring = Lent(vec![list[1].url().to_owned(), list[0].url().to_owned()]);
    let stated = |prefill: &str, decode: &str, ring: &dyn LentRing| {
        let document = PolicySpec::pair_presets(prefill, decode);
        let engine = engine_of(&document.expect("the pair presets"));
        let members = engine.pool_members(&list);
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let pair = pair_of(&members, &classes);
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        let made = Made::new(&engine, &request, &[]);
        let facts = made.make(engine.spec()).facts;
        (facts.ring.is_some(), facts.ring_positions.to_vec())
    };
    let lent = (true, vec![ring.position("conv-1")]);
    let ringed = "consistent_hashing";
    assert_eq!(stated(ringed, ringed, &ring), lent);
    assert_eq!(stated("round_robin", ringed, &ring), lent);
    assert_eq!(stated(ringed, "round_robin", &ring), lent);
    let none = (false, Vec::new());
    assert_eq!(stated("round_robin", "round_robin", &Unasked), none);
    assert_eq!(stated("prefix_hash", "prefix_hash", &Unasked), none);
    assert_eq!(stated("prefix_hash", ringed, &Unasked), none);
    assert_eq!(stated(ringed, "prefix_hash", &Unasked), none);
}

/// A filter of the host's own. It keeps a worker for a request that is
/// shown `blue` in the second header of the document's view and nothing
/// else, and for no other request.
#[derive(Debug)]
struct BlueRoute;

impl Filter for BlueRoute {
    fn name(&self) -> &'static str {
        "blue-route"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        let blue = Header {
            name: 1,
            value: "blue",
        };
        facts.headers == [blue]
    }
}

/// A stage is shown the headers of the document's view that the request
/// carries, each named by its place in that list: a filter that asks for
/// one of them keeps its workers for a request that carries it. A request
/// that carries another value, the value under another name, no header or
/// a value that is not text is left no worker by that filter.
#[test]
fn a_stage_is_shown_the_headers_the_documents_view_lists() {
    let mut stages = StageRegistry::builtin();
    let added = stages.filter("blue-route", || Box::new(BlueRoute));
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    document.view.headers = vec!["x-tier".to_owned(), "x-route".to_owned()];
    let profile = document.profiles.get_mut("random");
    let profile = profile.expect("the preset's profile");
    profile
        .filter
        .push(FilterSpec::Name("blue-route".to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(PolicyEngine::new(spec.expect("the stages hold the filter")));
    let list = pool(2);
    let members = engine.pool_members(&list);
    // The stage that left the request no worker; none where it is placed.
    let emptied_by = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(&members))
        };
        match engine.place(&request) {
            Ok(_) => None,
            Err(Unplaced::Shed { emptied_by, .. }) => Some(emptied_by),
            Err(other) => panic!("{other:?}"),
        }
    };
    assert_eq!(emptied_by(&[("x-route", "blue")]), None);
    assert_eq!(emptied_by(&[("x-route", "green")]), Some("blue-route"));
    assert_eq!(emptied_by(&[("x-tier", "blue")]), Some("blue-route"));
    assert_eq!(emptied_by(&()), Some("blue-route"));
    let mut headers = HeaderMap::new();
    let bytes = HeaderValue::from_bytes(b"blue\xFF").expect("a header's value");
    headers.insert(HeaderName::from_static("x-route"), bytes);
    assert_eq!(emptied_by(&headers), Some("blue-route"));
}

/// A view lists 256 headers at most, and the last one's place is 255: a
/// request that carries it is shown it under that place.
#[test]
fn the_last_of_256_view_headers_is_named_by_its_place() {
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.view.headers = (0..256).map(|at| format!("x-header-{at}")).collect();
    let engine = engine_of(&document);
    let headers = [("x-header-255", "last")];
    let request = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    let last = Header {
        name: 255,
        value: "last",
    };
    assert_eq!(made.make(engine.spec()).facts.headers, [last]);
}
