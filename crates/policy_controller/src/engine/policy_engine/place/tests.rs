use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap},
    ptr,
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
    document::spec::{FilterSpec, KeySourceSpec, PolicySpec, RoutingKeySpec},
    engine::{
        decision::{Queue, Shed, Target, Targets, Trace},
        decision_engine::Decided,
        framework::{Filter, Picker, Place, TieBand},
        records::worker_id,
        view::{fingerprint, Candidate, FixedFacts, HashKind},
        PairAdmission, PairLists, PartnerClasses,
    },
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

/// A picker that names every worker the filters keep, in the list's order,
/// and takes every word of a lease. `panics` makes its pick panic.
#[derive(Debug)]
struct InOrder {
    heard: Heard,
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
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        assert!(!self.panics, "this picker panics");
        push_drawn(candidates, kept, out);
    }
}

/// The `random` preset with [`InOrder`] in its pick's place and without
/// its `healthy` filter, so that a worker that is not ready stays a
/// candidate of the decision, and what the picker heard.
fn in_order(panics: bool) -> (&'static PolicyEngine, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("in-order", move || {
        Box::new(InOrder {
            heard: Arc::clone(&kept),
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
    (hosted(engine), heard)
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
/// of the list takes the request off that path.
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
            decode_again: true,
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
/// picked.
#[test]
fn a_plain_request_is_placed_on_the_worker_at_its_place() {
    let engine = hosted(preset("round_robin"));
    let Place::Alone(profile) = engine.place_for(MODEL, Phase::Single) else {
        panic!("one pick decides the place");
    };
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
            (profile.0, None)
        );
    }
    assert_eq!(turns, [2; 4], "every worker takes its turn");

    list[1].set_status(WorkerStatus::NotReady);
    for turn in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        assert!(!Arc::ptr_eq(&placement.worker, &list[1]), "turn {turn}");
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

/// Where none of the decision's targets can be sent to, the request has no
/// worker, and the lease its pick was issued is given up by name at once.
/// Under a document none of whose stages takes a word, no lease was issued
/// and nothing is said.
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
        [
            ("issued at 0".to_owned(), first),
            ("released CommitFailed".to_owned(), first)
        ]
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

/// A request with no worker is answered why. A decision's shed keeps its
/// reason and the stage that left nobody. What this call does not place
/// yet is declined by name: a pair's two lists and encode items. A demand
/// no worker of the list meets is shed under the demand's name. A decision
/// that panics is refused, and costs its request alone.
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
    let encoding = PlaceRequest {
        encode: Some(EncodeItems::default()),
        ..PlaceRequest::of(MODEL, lists)
    };
    assert_eq!(
        engine.place(&encoding).err(),
        shed(ShedReason::PolicyDeclined, "encode-items")
    );
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let paired = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(
        engine.place(&paired).err(),
        shed(ShedReason::PolicyDeclined, "pair-lists")
    );
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
