mod encode;
mod pair;
mod prompt_entry;
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
        Assignment, BoundedLoadParams, ConsistentHashParams, FilterSpec, KeySourceSpec, Needs,
        OverloadSpec, PolicySpec, Recovery, RoutingKeySpec, ScoreStage, SessionsSpec, ShadowSpec,
        MANUAL_SESSION_HEADER,
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
        consistent_hashing::{Head, InOrder as Lent},
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
        let facts = made.make(engine.standing_spec()).facts;
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
    let facts = made.make(unkeyed.standing_spec()).facts;
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
    } = made.make(engine.standing_spec());
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
        .make(engine.standing_spec())
        .id
        .is_empty());
    // A request with a pair's two lists is made, and says it is plain, at
    // its prefill.
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let paired = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let made = Made::new(&engine, &paired, &[]);
    assert_eq!(
        made.make(engine.standing_spec()).facts.phase,
        Phase::Prefill
    );
    assert_eq!(made.plain().map(|plain| plain.phase), Some(Phase::Prefill));
    // And so is a request with an encode request's three.
    let three = PlaceRequest::of(MODEL, three_of(pair));
    let made = Made::new(&engine, &three, &[]);
    assert_eq!(
        made.make(engine.standing_spec()).facts.phase,
        Phase::Prefill
    );
    // A client's tokens and text reach no log line.
    let prompt = Prompt {
        tokens: Some(&[7, 8, 9]),
        text: Some("secret words"),
        ..Prompt::NONE
    };
    let shown = format!("{prompt:?}");
    assert!(shown.contains("tokens: Some(3)"), "{shown}");
    assert!(shown.contains("text: Some(12)"), "{shown}");
    assert!(!shown.contains("secret"), "{shown}");
}

/// A request states its id with no making. An engine that keeps a sample of
/// its decisions asks it of a plain request, which is never made. A request
/// with no id states an empty one.
#[test]
fn a_request_states_its_id_with_no_making() {
    let engine = preset("round_robin");
    let request = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    assert_eq!(made.request_id(engine.standing_spec()), b"request-1");
    assert!(made.headers.get().is_none(), "the request was not made");
    let _ = made.make(engine.standing_spec());
    assert!(made.headers.get().is_some(), "a making fills its headers");
    let bare = PlaceRequest::of(MODEL, WorkerLists::None);
    let made = Made::new(&engine, &bare, &[]);
    assert!(made.request_id(engine.standing_spec()).is_empty());
}

/// The id a host states for a kept decision alone is no id of the request
/// made. Under a document that keys by the request's id, a request that
/// states one is made with that id and its key. One that states an explain
/// id alone is made as a request that states neither: an empty id and no
/// key. The maker answers the explain id beside the request's own.
#[test]
fn an_explain_id_is_no_id_of_the_request_made() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        }],
        strip_lineage: false,
    });
    let engine = engine_of(&document);
    let spec = engine.standing_spec();
    let stated = |request_id, explain_id| PlaceRequest {
        request_id,
        explain_id,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let keys = |request: &Request<'_>| {
        let facts = &request.facts;
        let keyed = facts.session_key != Identity::NONE;
        (keyed, facts.session_key_from_rid, facts.client_key)
    };

    let both = stated(Some("request-1"), Some("explain-1"));
    let made = Made::new(&engine, &both, &[]);
    let request = made.make(spec);
    assert_eq!(request.id, b"request-1");
    assert!(keys(&request).0, "keyed by its own id");
    let ids = (made.request_id(spec), made.explain_id());
    assert_eq!(ids, (&b"request-1"[..], &b"explain-1"[..]));

    let named = stated(None, Some("explain-1"));
    let bare = stated(None, None);
    let named = Made::new(&engine, &named, &[]);
    let bare = Made::new(&engine, &bare, &[]);
    let (request, unnamed) = (named.make(spec), bare.make(spec));
    assert!(request.id.is_empty(), "the explain id is not the request's");
    assert!(!keys(&request).0, "and it is no key");
    assert_eq!(keys(&request), keys(&unnamed));
    assert_eq!(named.explain_id(), b"explain-1");
    assert!(bare.explain_id().is_empty());
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
            .map(|_| made.make(engine.standing_spec()).facts.seed)
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
/// block and no size, and the seed of its partition like any other.
///
/// A text alone states no count of tokens, so a stage that credits a send
/// of no stated size credits it at its own mean. The classic router's
/// `least_load` prices a request with no token ids at its mean prefill
/// (`model_gateway/src/policies/least_load.rs:812-816`). The text states
/// its two lengths. Its bytes are at hand, and a reader that sizes a
/// request of text reads them over four. Its characters are counted
/// because a stage of this document reads a request's size. Token ids
/// win: beside them no length is stated.
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
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };

    let request = stated(Some(&tokens[..]), Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let (first, again) = (
        made.make(engine.standing_spec()).facts,
        made.make(engine.standing_spec()).facts,
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
    assert_eq!(first.prompt_chars, 0, "the text is not passed over");
    assert!(
        ptr::eq(first.block_hashes, again.block_hashes),
        "hashed once"
    );

    // Token ids alone, as a tokenized request states its prompt.
    let request = stated(Some(&tokens[..]), None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert_eq!(facts.block_hashes, expected.block_hashes);
    assert_eq!(
        (facts.hash_kind, facts.prompt_tokens),
        (HashKind::Token, 600)
    );

    let request = stated(None, Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert_eq!(facts.hash_kind, HashKind::Text);
    // The bytes are stated under any document, and the size is theirs.
    assert_eq!(
        (facts.prompt_tokens, facts.prompt_bytes, facts.prompt_size()),
        (0, 4_000, 1_000)
    );
    assert!(!facts.block_hashes.is_empty());

    let request = stated(None, None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert!(facts.block_hashes.is_empty());
    assert_eq!(facts.prompt_tokens, 0);
    assert_eq!(facts.namespace_seed, 13, "its partition all the same");
}

/// A request that is lent a worker is hashed by the block size the host
/// knows under the id the first worker of its list registered under, the
/// id the list's table finds its index by, whatever size the host knows
/// under the id the request states. A pair is hashed by its prefill list.
#[test]
fn a_prompt_is_hashed_by_the_block_size_of_its_workers_id() {
    struct Learned;
    impl CacheIndexes for Learned {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            None
        }

        fn block_size(&self, model: &str) -> Option<u32> {
            Some(if model == MODEL { 4 } else { 8 })
        }
    }
    let engine = preset("cache_aware");
    engine.affinity.set_indexes(Arc::new(Learned));
    let members = PublishedList::new(pool(2));
    // The pair's decode registered under the id the request states.
    let decodes = PublishedList::new([ready_worker("http://decode-0:8000", "another-id")].into());
    let classes = PartnerClasses::number(1, [&[0_u32][..], &[0][..]]);
    let pair = PairLists {
        decode: decodes.lent(),
        ..pair_of(&members, &classes)
    };
    let tokens: Vec<u32> = (0..600).collect();
    for lists in [
        WorkerLists::Single(members.lent()),
        WorkerLists::Pair(pair),
        three_of(pair),
    ] {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&tokens),
                ..Prompt::NONE
            },
            ..PlaceRequest::of("another-id", lists)
        };
        let made = Made::new(&engine, &request, &[]);
        assert_eq!(made.make(engine.standing_spec()).facts.block_size, 4);
    }
}

/// A prompt is hashed under the document its decision hands the maker,
/// whatever document the engine holds in force by then: a decision that
/// began under one document reads no word of the next. The first document
/// reads a prompt's blocks, and the one put in force behind it reads none.
#[test]
fn a_prompt_is_hashed_under_the_document_its_decision_runs_under() {
    let engine = preset("cache_aware");
    let began_under = engine.spec();
    let mut next = PolicySpec::preset("random").expect("a built-in preset");
    next.generation = 2;
    let next = AppliedSpec::validate(&next).expect("the document validates");
    engine.apply(next).expect("a number above the standing one");

    let tokens = (0..600).collect::<Vec<u32>>();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    let hashed = made.make(&began_under).facts.block_hashes.len();
    assert!(hashed > 0, "the decision's document reads the blocks");
    let made = Made::new(&engine, &request, &[]);
    let in_force = engine.spec();
    assert!(made.make(&in_force).facts.block_hashes.is_empty());
}

/// A plain pick's lease sizes the prompt under the document its decision
/// runs under, whatever document the engine holds in force by then. The
/// first document's picker places by itself, reads a request's size and
/// takes a lease's issue. Its pick puts the `random` preset in force,
/// none of whose stages reads a size, ahead of the lease.
#[test]
fn a_plain_picks_lease_sizes_its_prompt_under_its_decisions_document() {
    const NAME: &str = "sizes-alone";
    type Host = Arc<OnceLock<&'static InProcess>>;

    /// Keeps the characters each lease it hears issued states.
    #[derive(Debug)]
    struct SizesAlone(Arc<Mutex<Vec<u32>>>, Host);

    impl Lifecycle for SizesAlone {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.lock().push(lease.chars);
        }
    }

    impl Picker for SizesAlone {
        fn name(&self) -> &'static str {
            NAME
        }

        fn needs(&self) -> Needs {
            Needs {
                prompt_size: true,
                ..Needs::default()
            }
        }

        fn refuses_by_reports(&self) -> bool {
            false
        }

        fn picks_alone(&self) -> bool {
            true
        }

        // The pool's first worker, behind the next document's apply.
        fn pick_alone(
            &self,
            _: &PoolWord<'_>,
            keep: &mut dyn FnMut(usize) -> bool,
        ) -> Option<usize> {
            let engine = self.1.get().expect("the engine is hosted");
            let mut next = PolicySpec::preset("random").expect("a built-in preset");
            next.generation = 2;
            let next = AppliedSpec::validate(&next).expect("the document validates");
            engine.apply(next).expect("a number above the standing one");
            keep(0).then_some(0)
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
            push_drawn(candidates, kept, out);
        }
    }

    let chars = Arc::new(Mutex::new(Vec::new()));
    let host = Host::default();
    let (kept, hosted_at) = (Arc::clone(&chars), Arc::clone(&host));
    let mut stages = StageRegistry::builtin();
    stages
        .picker(NAME, move || {
            Box::new(SizesAlone(Arc::clone(&kept), Arc::clone(&hosted_at)))
        })
        .expect("a free picker name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    profile.expect("the random profile").pick = NAME.into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("a valid document");
    let engine = hosted(InProcess::new(spec));
    let began_under = engine.spec();
    assert!(began_under.needs().prompt_size, "a stage reads the size");
    let place = began_under.place_for(MODEL, Phase::Single);
    assert!(matches!(place, Place::Alone(_)), "{place:?}");
    host.set(engine).expect("hosted once");

    // A text of 200 two-byte characters, held as text alone.
    let text = "\u{e9}".repeat(200);
    let members = PublishedList::new(pool(4));
    let request = PlaceRequest {
        prompt: Prompt {
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(placement.owes_events(), "the pick has a lease");
    let in_force = engine.spec();
    assert_eq!(in_force.generation(), 2, "the pick put the next in force");
    assert!(!in_force.needs().prompt_size, "no stage of it reads a size");
    assert_eq!(*chars.lock(), [200], "sized under the decision's document");
}

/// A first attempt that names nothing tried or missed, demands nothing,
/// carries no encode item and asks no decode again says it is plain, with
/// the size its request would state. The engine then answers it a place where one pick decides its
/// place: no clock is read for it. Each fact that asks more than a worker
/// of the list takes the request off that path. A decode asked again
/// does not: over one list the word asks nothing. Neither does a list
/// whose host states facts of its workers: a request that demands nothing
/// reads none of them.
#[test]
fn a_plain_request_says_so_and_is_answered_a_place() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens = [7_u32; 40];
    let plain = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    // A text alone, as a request that is not tokenized states its prompt,
    // says no count of tokens. Asked its text's lengths, it states the
    // bytes under any document, and counts the characters under a
    // document a stage of which reads a request's size, as `bucket` does,
    // and none under this one. The text is of 200 two-byte characters.
    let text = "\u{e9}".repeat(200);
    let worded = PlaceRequest {
        prompt: Prompt {
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, plain.lists)
    };
    let sizes = |engine: &InProcess, request: &PlaceRequest<'_>| {
        let (document, call) = (engine.in_force(), Call::default());
        let made = Made::asked(engine.under(&document), request, &call);
        let said = made.plain().expect("a plain request");
        (said.prompt_tokens, made.prompt_chars(), made.prompt_bytes())
    };
    assert_eq!(sizes(&engine, &worded), (0, 0, 400));
    assert_eq!(sizes(&preset("bucket"), &worded), (0, 200, 400));
    // Token ids win: a request that holds its ids and a text states the
    // count of the ids, no length of the text and no pass over it.
    let both = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, plain.lists)
    };
    assert_eq!(sizes(&preset("bucket"), &both), (40, 0, 0));
    let read = Cell::new(0);
    let clock = || {
        read.set(read.get() + 1);
        Instant::now()
    };
    let decided = engine.decide_at(&Made::new(&engine, &plain, &[]), clock);
    assert!(matches!(decided, Ok(Decided::Alone(pick)) if pick.place < 4));
    assert_eq!(read.get(), 0, "a place dates nothing");
    // A list whose host states a fact of each worker, as a host that dials
    // several runtimes does: the fill of the list's cell takes the facts,
    // and the plain request is answered a place as over the silent list.
    let stating = PublishedList::with_facts(Arc::clone(&list), |_| {
        let mut facts = FixedFacts::default();
        facts.demands.runtime = 7;
        facts
    });
    let stated = PlaceRequest {
        prompt: plain.prompt,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(stating.lent()))
    };
    assert_eq!(Made::new(&engine, &stated, &[]).plain(), said);
    let decided = engine.decide_at(&Made::new(&engine, &stated, &[]), clock);
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
///
/// The word is a pair's, over its two lists or an encode request's three.
/// Over one list, and over none, a request that carries it states no
/// later decode: the request made is as plain as its plain test says.
#[test]
fn the_later_decode_is_stated_where_the_document_picks_it_so() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    let at_once = engine_of(&document);
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let after = engine_of(&document);
    let members = PublishedList::new(pool(2));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let later = |engine: &InProcess, lists: WorkerLists<'_>, decode_again| {
        let request = PlaceRequest {
            decode_again,
            ..PlaceRequest::of(MODEL, lists)
        };
        let made = Made::new(engine, &request, &[]);
        made.make(engine.standing_spec()).constraints.decode_later
    };
    for lists in [WorkerLists::Pair(pair), three_of(pair)] {
        assert!(later(&after, lists, true));
        assert!(!later(&after, lists, false));
        assert!(!later(&at_once, lists, true));
    }
    for lists in [WorkerLists::Single(members.lent()), WorkerLists::None] {
        let request = PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, lists)
        };
        let made = Made::new(&after, &request, &[]);
        let stated = made.make(after.standing_spec());
        assert!(!stated.constraints.decode_later);
        assert!(made.plain().is_some());
        assert_eq!(stated.plain(), made.plain());
    }
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
        let key = made.make(engine.standing_spec()).facts.session_key;
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

/// A picker that places the worker at place 0 of its list by itself, takes
/// no word of a lease, and marks that worker not ready behind the read that
/// found it eligible.
#[derive(Debug)]
struct Turns(Arc<dyn Worker>);

impl Lifecycle for Turns {}

impl Picker for Turns {
    fn name(&self) -> &'static str {
        "turns"
    }

    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn picks_alone(&self) -> bool {
        true
    }

    fn pick_alone(&self, _: &PoolWord<'_>, keep: &mut dyn FnMut(usize) -> bool) -> Option<usize> {
        let kept = keep(0);
        if kept {
            self.0.set_status(WorkerStatus::NotReady);
        }
        kept.then_some(0)
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
        push_drawn(candidates, kept, out);
    }
}

/// Under a document whose session table alone takes a word on a lease, the
/// worker at a plain request's place is read a second time, as a noted
/// pick's target is at its walk: where its picker turned it, the pick is
/// noted, and the call's next decision places the request on a worker that
/// is ready.
#[test]
fn a_place_under_a_table_that_alone_takes_a_word_is_read_ready_a_second_time() {
    let list = pool(2);
    let first = Arc::clone(&list[0]);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("turns", move || Box::new(Turns(Arc::clone(&first))));
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "turns".into();
    let spec = AppliedSpec::validate_in(&ruled_ahead(document), &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the picker")));
    let hears = engine.in_force().hears.intersects(EventSet::LEASE);
    let beside = engine.in_force().hears_beside_table;
    assert!(
        hears && !beside.intersects(EventSet::LEASE),
        "the session table alone takes a word"
    );
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let placement = engine.place(&request).expect("a placement");
    assert!(!ready(list[0].as_ref()), "the pick turned its worker");
    assert!(
        Arc::ptr_eq(&placement.worker, &list[1]),
        "the worker that is ready"
    );
}

/// A room rule that names the request's output limit reserves room for the
/// answer its caller states through the one call: a worker whose requests
/// in flight leave no room for the answer is passed over while another
/// has room. No classic policy reads an output limit: the rule is a
/// document's own.
#[test]
fn the_output_limit_a_caller_states_reserves_room_under_the_documents_rule() {
    let document = r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{
        "filter":["healthy",{"has-room":{
            "used":{"candidate.in_flight":100,"request.max_output_tokens":1},
            "capacity":1000}}],
        "pick":"passthrough"}}}"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a well-formed document");
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    // Five requests in flight on the list's first worker: half its capacity.
    for _ in 0..5 {
        list[0].increment_load();
    }
    let placed_at = |max_output_tokens| {
        let request = PlaceRequest {
            max_output_tokens,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        list.iter()
            .position(|worker| Arc::ptr_eq(worker, &placement.worker))
    };
    // An answer of 500 tokens fits beside the five requests, so the first
    // worker of the list takes the request. A token more does not fit.
    assert_eq!(placed_at(500), Some(0));
    assert_eq!(placed_at(501), Some(1));
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
#[test]
fn placing_a_request_asks_the_table_of_names_for_nothing() {
    let asked = || crate::names::ASKED.with(Cell::get);
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
    let members = PublishedList::new(Arc::clone(&list));
    let tried = [worker_id(list[0].as_ref())];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    for _ in 0..4 {
        let placement = engine.place(&retry).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[1]));
        assert_eq!(placement.id(), worker_id(list[1].as_ref()));
    }
}

/// A retry whose tried worker has left its list names no worker of it and
/// states its number alone. It takes no turn of the list: the first
/// attempts between such retries take the list's workers one after the
/// other, and the retries go round every worker of the list, at every
/// size.
///
/// The classic router gives a request that tried a worker an untried list
/// of its own, whether or not that worker is still listed
/// (`model_gateway/src/routers/common/placement.rs:1256-1269`), and its
/// policy takes a list's turn for a list named whole alone (`:1277`,
/// `model_gateway/src/policies/round_robin.rs:145`). The gateway states the
/// number as the count of the workers a request tried and names the ones
/// still listed (`model_gateway/src/routers/common/place_request.rs:33-44`,
/// `:94`).
#[test]
fn a_retry_whose_tried_worker_left_the_list_takes_no_turn_of_it() {
    for workers in [4, DRAW_FROM, 64] {
        let engine = hosted(preset("round_robin"));
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        let first = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
        let retry = PlaceRequest {
            attempt: 1,
            ..PlaceRequest::of(MODEL, first.lists)
        };
        let at = |request: &PlaceRequest<'_>| {
            let placement = engine.place(request).expect("a placement");
            let taken = |worker: &Arc<dyn Worker>| Arc::ptr_eq(worker, &placement.worker);
            list.iter().position(taken).expect("a worker of the list")
        };
        // As many first attempts as turn the list once for each worker but
        // one, and a retry behind each: the retries go a whole number of
        // times round the list.
        let requests = workers * (workers - 1);
        let (mut firsts, mut retries) = (Vec::new(), vec![0_usize; workers]);
        for _ in 0..requests {
            firsts.push(at(&first));
            retries[at(&retry)] += 1;
        }
        let in_turn: Vec<usize> = (0..requests)
            .map(|turn| (firsts[0] + turn) % workers)
            .collect();
        assert_eq!(firsts, in_turn, "{workers} workers: the first attempts");
        let shares = vec![workers - 1; workers];
        assert_eq!(retries, shares, "{workers} workers: the retries");
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
    let members = PublishedList::new(Arc::clone(&list));
    for worker in &list[..2] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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

/// A call makes three decisions at most, and those of a request that
/// states no rule are held to no worker. Over a list none of whose first
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
    let members = PublishedList::new(Arc::clone(&list));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        true
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(14);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let only = worker_id(list[4].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == only;
    let tried = [only];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[4]), "tried, and alone");
    drop(placement);

    list[4].set_overloaded(true);
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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

/// A rule that takes no worker sheds the request so where no pick of the
/// first decision is ready too. No pick is asked of the rule: it is asked
/// once of every worker of the list, and no second decision is made.
#[test]
fn a_rule_that_takes_no_worker_is_shed_where_no_pick_is_ready() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let calls: Vec<_> = (0..6).map(id).collect();
    assert_eq!(*asked.borrow(), calls, "no pick, and the list once");
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
    let members = PublishedList::new(Arc::clone(&list));
    let last = worker_id(list[3].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == last;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let members = PublishedList::new(Arc::clone(&list));
    let thirds = || (0..64).step_by(3);
    let taken: BTreeSet<WorkerId> = thirds().map(|at| worker_id(list[at].as_ref())).collect();
    let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let mut call = Call::default();
    call.missed.push(worker_id(list[1].as_ref()));
    // A request that states no rule keeps no places.
    let unruled = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    hold_to_rule(&unruled, Some(members.lent()), None, &mut call.admitted).expect("no rule");
    assert_eq!(call.admitted, None);
    hold_to_rule(&request, Some(members.lent()), None, &mut call.admitted)
        .expect("the rule takes workers");
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
        let members = PublishedList::new(Arc::clone(&list));
        let mut by_list = 0;
        for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
            let headers = [(MANUAL_SESSION_HEADER, key)];
            let keyed = PlaceRequest {
                headers: &headers,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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

/// A fresh key whose first decision names workers the rule refuses keeps no
/// pin of them: the commit refused every target of that decision, and the
/// claim it made goes with it. The request is sent to the rule's one
/// worker, and that worker is the key's pin: the key's request with no rule
/// lands there.
///
/// The classic router pins no worker its rule refuses: it narrows the list
/// by the rule ahead of its policy (`Candidates::of`,
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_fresh_key_keeps_no_pin_of_a_worker_its_rule_refused() {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    rule.expect("the stock rule").assignment = Assignment::MinLoad;
    let engine = hosted(engine_of(&document));
    // Four idle workers and a busier one: the first decision names the four.
    let list = pool(5);
    list[4].increment_load();
    let members = PublishedList::new(Arc::clone(&list));
    let busy = worker_id(list[4].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == busy;
    let headers = [(MANUAL_SESSION_HEADER, "talk-fresh")];
    let keyed = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let ruled = PlaceRequest {
        admits: Some(&admits),
        ..keyed
    };
    let mut placement = engine.place(&ruled).expect("a placement");
    placement.sent(Side::Single);
    assert_eq!(placement.id(), busy);
    let next = engine.place(&keyed).expect("a placement");
    assert_eq!(next.id(), busy, "the pin is the worker that served the key");
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
    let members = PublishedList::new(Arc::clone(&list));
    let admits = slow(&list);
    let request = PlaceRequest {
        budget_micros: 10_000_000,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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

    let unbounded = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
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
    let members = PublishedList::new(Arc::clone(&list));
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
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 4 + 8 + 2, "the list is asked once");
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(id(0)), given_up(id(5)), given_up(id(5))].concat()
    );
}

/// A call whose first decision names workers that are not ready has its
/// route's rule asked of the list behind that decision, though the rule
/// refused nobody: a pick that is not ready is not asked of it, so such a
/// decision learns nothing of the rule. Over a list whose first twelve
/// workers are not ready, three decisions over the whole list name them
/// four by four, and the request had no worker though the thirteenth is
/// ready and the rule takes it. Held to that worker, the second decision
/// names it.
///
/// The classic router places it: it narrows the list by the rule
/// (`accepted_by`, `model_gateway/src/routers/common/placement.rs:474-483`
/// on main `77d6a42d3`) and picks among the workers that are available
/// (`:1338-1341` there).
#[test]
fn a_call_whose_first_picks_are_not_ready_is_held_to_the_rules_workers() {
    let (engine, heard) = in_order(false);
    let list = pool(13);
    for worker in &list[..12] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(12);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[12]));
    assert_eq!(calls.get(), 13 + 1, "the list, then the pick");
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), far));
    assert_eq!(heard.lock().as_slice(), said);
    drop(placement);
    heard.lock().clear();

    // A rule that takes that worker of the list and refuses it as a pick:
    // the list is asked once, and the third decision is the call's last.
    calls.set(0);
    let fickle = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        calls.get() <= 13 && worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&fickle),
        ..request
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 13 + 2, "the list is asked once");
    let three = [0, 12, 12].map(|first| given_up(id(first)));
    assert_eq!(heard.lock().as_slice(), three.concat());
}

/// The rule is asked of the list while the call holds no decision: where it
/// panics there, the decision it refused was given up by name already, and
/// no lease is left behind.
#[test]
fn a_rule_that_panics_at_the_list_leaves_no_lease_behind() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
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
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    let document = engine.in_force();
    let blocks = {
        let first = Made::asked(engine.under(&document), &request, &call);
        assert!(first.plain().is_some());
        let facts = first.make(engine.standing_spec()).facts;
        assert!(!facts.block_hashes.is_empty());
        facts.block_hashes.as_ptr()
    };
    assert!(call.hashes.get().is_some(), "hashed in the call's cell");
    call.missed.push(WorkerId(3));
    let again = Made::asked(engine.under(&document), &request, &call);
    assert_eq!(again.plain(), None, "a missed target");
    let made = again.make(engine.standing_spec());
    assert_eq!(made.constraints.excluded, [WorkerId(3)]);
    assert!(
        ptr::eq(made.facts.block_hashes.as_ptr(), blocks),
        "hashed once a call"
    );
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
fn lent_ring_places(engine: &'static InProcess) {
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let ring = Lent(order.iter().map(|&at| list[at].url().to_owned()).collect());
    // The worker the ring's walk reaches first from where `text` stands.
    let reached = |text: &str| order[(ring.position(text) % 8) as usize];
    let placed = |headers: &dyn HeaderRead, ring: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
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
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&ruled).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[along(5)]), "{key}");
        eighth_first += usize::from(along(7) < along(5));
    }
    assert!(eighth_first > 0, "no key has the eighth ahead in the list");
}

/// [`Lent`], with a count of the keys and the heads it was asked the place
/// of.
#[derive(Debug)]
struct Asked(Lent, AtomicU32);

impl LentRing for Asked {
    fn position(&self, key: &str) -> u64 {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.head_position(head)
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
    let members = PublishedList::new(Arc::clone(&list));
    let order = vec![list[1].url().to_owned(), list[0].url().to_owned()];
    let ring = Asked(Lent(order), AtomicU32::new(0));
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let first = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    let again = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
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

/// Under `prefix_hash` a lender that places no head lends no ring: the
/// request states no place, and the stage's own ring places it on the
/// worker it has with no ring lent. The lender is not asked where a key's
/// own text stands, which is no place of this form: a routing value and a
/// prompt are answered alike. A document with no ring stage asks a ring
/// nothing at all.
#[test]
fn a_lender_that_places_no_head_lends_no_ring_under_the_prefix_form() {
    let tokens: Vec<u32> = (0..64).collect();
    let keyed = [("x-smg-routing-key", "conv-1")];
    let rings: [(&str, &'static dyn LentRing); 2] =
        [("prefix_hash", &Headless), ("random", &Unasked)];
    for (name, ring) in rings {
        let engine = hosted(preset(name));
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        for headers in [&keyed as &dyn HeaderRead, &()] {
            let request = |ring: Option<&'static dyn LentRing>| PlaceRequest {
                prompt: Prompt {
                    tokens: Some(&tokens),
                    ..Prompt::NONE
                },
                headers,
                ring,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
            };
            let placed = |request: &PlaceRequest<'_>| {
                let placement = engine.place(request).expect("a placement");
                let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
                list.iter().position(worker).expect("a worker of the list")
            };
            let lent = request(Some(ring));
            let made = Made::new(engine, &lent, &[]);
            let facts = made.make(engine.standing_spec()).facts;
            assert!(facts.ring.is_none(), "{name}");
            assert!(facts.ring_positions.is_empty(), "{name}");
            let with = placed(&lent);
            if name == "prefix_hash" {
                assert_eq!(with, placed(&request(None)), "the stage's own ring");
            }
        }
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

/// Under the prefix form a request through the one call states where its
/// lender puts each of its heads, the deepest first: a head at each of the
/// view's boundaries the prompt reaches, and for a prompt that reaches
/// none the one head of the form's depth, which a shorter prompt is whole.
/// The gateway's own policy cuts a prompt so
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 361 to 386 at
/// 882be6f4f). A routing value leads the prompt: its text is the one head,
/// placed as a head is and never as a key's own text. A client's value
/// places nothing. The request goes where the ring's walk reaches from its
/// first place. The lender is asked once for each head: a second making
/// states the places the first took.
#[test]
fn a_request_under_the_prefix_form_states_where_its_heads_stand() {
    let engine = hosted(prefixed(&[64, 256]));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let urls = order.iter().map(|&at| list[at].url().to_owned());
    let ring = Asked(Lent(urls.collect()), AtomicU32::new(0));
    let asks = || ring.1.load(atomic::Ordering::Relaxed);
    let at = |head: Head<'_>| ring.0.head_position(head).expect("a lender of heads");
    // The request's places, and the worker of the list it went to.
    let stated = |tokens: Option<&[u32]>, text: Option<&str>, headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens,
                text,
                ..Prompt::NONE
            },
            headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let before = asks();
        let facts = made.make(engine.standing_spec()).facts;
        let places = facts.ring_positions.to_vec();
        assert_eq!(
            facts.ring.is_some(),
            !places.is_empty(),
            "lent with a place"
        );
        assert_eq!(asks() - before, places.len() as u32, "one ask a head");
        assert_eq!(
            made.make(engine.standing_spec()).facts.ring_positions,
            places
        );
        assert_eq!(asks() - before, places.len() as u32, "a second making");
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        let went = list.iter().position(worker).expect("a worker of the list");
        if let Some(&first) = places.first() {
            assert_eq!(went, order[(first % 8) as usize], "the walk's first");
        }
        places
    };
    let ids: Vec<u32> = (500..800).collect();
    let both = [at(Head::Tokens(&ids[..256])), at(Head::Tokens(&ids[..64]))];
    assert_eq!(stated(Some(&ids), None, &()), both);
    assert_eq!(
        stated(Some(&ids[..100]), None, &()),
        [at(Head::Tokens(&ids[..64]))]
    );
    assert_eq!(
        stated(Some(&ids[..40]), None, &()),
        [at(Head::Tokens(&ids[..40]))]
    );
    // A text is cut at four characters a token.
    let text: String = (0..300).map(|at| format!("{at:04}")).collect();
    let cut = [at(Head::Text(&text[..1024])), at(Head::Text(&text[..256]))];
    assert_eq!(stated(None, Some(&text), &()), cut);
    assert_eq!(stated(Some(&[]), Some(&text), &()), cut, "no token id");
    let keyed = [("x-smg-routing-key", "conv-1")];
    let led = stated(Some(&ids), None, &keyed);
    assert_eq!(led, [at(Head::Text("conv-1"))]);
    assert_ne!(led, [ring.0.position("conv-1")], "the key's own text");
    let client = [("x-client", "10.0.0.1")];
    assert_eq!(stated(Some(&ids), None, &client), both);
    assert_eq!(stated(None, None, &client), [0_u64; 0]);
}

/// The `prefix_hash` preset over a view of `boundaries`, whose binding
/// reads a routing value from the header `x-smg-routing-key` and then from
/// the request's own id, the turn and retry suffixes off: the sources a
/// gateway's sticky override names. The view lists the header, so its
/// value leaves in the clear as the id's does. `rule` is the binding's
/// sessions rule.
fn prefixed_by_id(boundaries: &[u32], rule: Option<SessionsSpec>) -> InProcess {
    let mut document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    document.view.boundaries = boundaries.to_vec();
    document.view.headers.push("x-smg-routing-key".to_owned());
    let header = KeySourceSpec {
        header: Some("x-smg-routing-key".to_owned()),
        ..KeySourceSpec::default()
    };
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    let binding = &mut document.bindings[0];
    binding.routing_key = Some(RoutingKeySpec {
        sources: vec![header, rid],
        strip_lineage: true,
    });
    binding.sessions = rule;
    engine_of(&document)
}

/// Under the prefix form a routing value leads the prompt only where a
/// header handed it. The gateway's own policy hashes a header's value
/// (`model_gateway/src/policies/prefix_hash.rs:358`) and places every other
/// request by its prompt (`:360-372`), one keyed by its own id among them.
/// So a request whose value its own id handed states its prompt's heads,
/// and lands where the same prompt lands with no value, on a lent ring and
/// on the stage's own. The value is still the request's session key, and
/// the one call says where it came from.
#[test]
fn a_value_the_requests_own_id_handed_leads_no_prompt_under_the_prefix_form() {
    let engine = hosted(prefixed_by_id(&[64, 256], None));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let urls = order.iter().map(|&at| list[at].url().to_owned());
    let ring = Lent(urls.collect());
    let at = |head: Head<'_>| ring.head_position(head).expect("a lender of heads");
    let ids: Vec<u32> = (500..800).collect();
    // The request's places, whether its own id handed its value, the value
    // in the clear, and the worker of the list it went to.
    let stated = |headers: &dyn HeaderRead, id: Option<&str>, lent: bool| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers,
            request_id: id,
            ring: lent.then_some(&ring as &dyn LentRing),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        let (places, by_id) = (facts.ring_positions.to_vec(), facts.session_key_from_rid);
        let key = facts.session_key.plain.map(str::to_owned);
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        let went = list.iter().position(worker).expect("a worker of the list");
        (places, by_id, key, went)
    };
    let heads = vec![at(Head::Tokens(&ids[..256])), at(Head::Tokens(&ids[..64]))];
    let keyed = [("x-smg-routing-key", "conv-1")];
    for lent in [true, false] {
        let places = |heads: &[u64]| if lent { heads.to_vec() } else { Vec::new() };
        let unkeyed = stated(&(), None, lent);
        assert_eq!(unkeyed, (places(&heads), false, None, unkeyed.3));
        // The id's value, its turn and retry suffixes off: the prompt's
        // heads and the prompt's worker, under two ids.
        for id in ["conv-1_t2_r1", "conv-2"] {
            let key = id.split('_').next().map(str::to_owned);
            let by_id = stated(&(), Some(id), lent);
            assert_eq!(by_id, (places(&heads), true, key, unkeyed.3), "{id}");
        }
        // A header's value leads: its text is the one head, whatever the
        // request's id.
        let led = stated(&keyed, Some("conv-9"), lent);
        let head = [at(Head::Text("conv-1"))];
        assert_eq!(led.0, places(&head));
        assert_eq!((led.1, led.2.as_deref()), (false, Some("conv-1")));
    }
    assert_ne!(heads[0], at(Head::Text("conv-1")), "two places");
}

/// A sessions rule pins the value the request's own id handed, as it pins
/// a header's: the key's next request follows the pin, whatever its
/// prompt. The ring placed the key's first request by its prompt, where a
/// request of that prompt with no value lands, and another id's request of
/// the second prompt lands on that prompt's worker.
#[test]
fn a_sessions_rule_pins_the_value_the_requests_own_id_handed() {
    let rule = SessionsSpec {
        sticky_header: String::new(),
        assignment: Assignment::Delegate,
        recovery: Recovery::Latest,
        ..PolicySpec::preset("manual")
            .expect("a built-in preset")
            .bindings
            .remove(0)
            .sessions
            .expect("the manual preset's rule")
    };
    let engine = hosted(prefixed_by_id(&[], Some(rule)));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let went = |tokens: &[u32], id: Option<&str>| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(tokens),
                ..Prompt::NONE
            },
            request_id: id,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        list.iter().position(worker).expect("a worker of the list")
    };
    let prompt = |from: u32| (from..from + 256).collect::<Vec<u32>>();
    let first = prompt(500);
    let home = went(&first, None);
    // A second prompt whose worker is another.
    let (second, other) = (1..)
        .map(|n| prompt(n * 1000))
        .map(|tokens| {
            let worker = went(&tokens, None);
            (tokens, worker)
        })
        .find(|&(_, worker)| worker != home)
        .expect("a prompt with another worker");
    assert_eq!(went(&first, Some("conv-1_t1")), home, "by its prompt");
    assert_eq!(went(&second, Some("conv-1_t2")), home, "by its pin");
    assert_eq!(went(&second, Some("conv-2_t1")), other, "another id");
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

/// A prompt that reaches more boundaries than its list of places holds in
/// itself states every one of them, the deepest first, and is tried at
/// each in turn: the gateway's own policy lists every boundary a prompt
/// reaches and tries each until one's owner is under the bound
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 172 to 180 and 291
/// to 319 at 882be6f4f). Here nine. The owner of the eight deepest is past
/// the bound, and the ninth's takes the request, though two workers hold
/// fewer requests. A prompt that reaches a tenth boundary states ten
/// places: none is lost and none stated twice.
#[test]
fn a_prompt_that_reaches_nine_boundaries_is_tried_at_the_ninth() {
    let boundaries: Vec<u32> = (1..=10).map(|at| 16 * at).collect();
    let engine = hosted(prefixed(&boundaries));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    (0..30).for_each(|_| list[0].increment_load());
    (0..5).for_each(|_| list[1].increment_load());
    let ring = ByDepth {
        shallowest: list[1].url().to_owned(),
        deeper: list[0].url().to_owned(),
    };
    let ids: Vec<u32> = (0..150).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let places = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    assert_eq!(places, [144, 128, 112, 96, 80, 64, 48, 32, 16]);
    let placement = engine.place(&request).expect("a placement");
    assert!(
        Arc::ptr_eq(&placement.worker, &list[1]),
        "the ninth's owner"
    );
    let ids: Vec<u32> = (0..160).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let places = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    assert_eq!(places, [160, 144, 128, 112, 96, 80, 64, 48, 32, 16]);
}

/// A scorer that scores every worker alike, and counts how often it is
/// asked what it needs and how it places by heads.
#[derive(Debug)]
struct CountsAsks(Arc<AtomicU32>);

impl Lifecycle for CountsAsks {}

impl Scorer for CountsAsks {
    fn name(&self) -> &'static str {
        "counts-asks"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        self.0.fetch_add(1, atomic::Ordering::Relaxed);
        Needs::default()
    }

    fn places_by_heads(&self) -> Option<HeadRule<'_>> {
        self.0.fetch_add(1, atomic::Ordering::Relaxed);
        None
    }
}

/// A stage says how it places by heads, and what it needs, where its
/// generation is built. No making of a request and no decision asks it
/// again, under a lent ring or under none: the engine answers from what
/// the build kept.
#[test]
fn a_stage_is_asked_how_it_places_where_its_generation_is_built() {
    let asks = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&asks);
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("counts-asks", move || {
        Box::new(CountsAsks(Arc::clone(&counted)))
    });
    added.expect("a free name");
    // The prefix form's document, the counting stage beside its ring stage.
    let document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let mut document = serde_json::to_value(document).expect("the document as JSON");
    let profiles = document["profiles"].as_object_mut();
    for profile in profiles.expect("the profiles").values_mut() {
        let score = profile["score"].as_array_mut();
        score
            .expect("the profile's scorers")
            .push("counts-asks".into());
    }
    let mut document: PolicySpec = serde_json::from_value(document).expect("a document");
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the scorer")));
    let built = asks.load(atomic::Ordering::Relaxed);
    assert!(built > 0, "the build asks the stage");

    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let lent = Lent(vec![list[0].url().to_owned()]);
    let ids: Vec<u32> = (0..64).collect();
    for ring in [Some(&lent as &dyn LentRing), None] {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        assert_eq!(facts.ring.is_some(), ring.is_some());
        engine.place(&request).expect("a placement");
    }
    let asked = asks.load(atomic::Ordering::Relaxed);
    assert_eq!(asked, built, "no decision asks the stage");
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

/// A lender that answers nothing for a head lends no ring, whichever head
/// it declines: the request states no place, a place answered ahead of the
/// `None` among them, and the lender is asked about no head behind that
/// one. A routing value is the request's one head: where the lender
/// declines it, the prompt behind it is not asked about. A second making
/// asks nothing again.
#[test]
fn a_lender_that_declines_a_head_is_asked_nothing_more() {
    let engine = hosted(prefixed(&[16, 32, 48]));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let ids: Vec<u32> = (0..48).collect();
    let keyed = [("x-smg-routing-key", "conv-1")];
    // The lender's script, the request's headers, the asks the lender
    // counts.
    let rows: [(&'static [Option<u64>], &dyn HeaderRead, u32); 4] = [
        (&[None, Some(1), Some(2)], &(), 1),
        (&[Some(1), None, Some(2)], &(), 2),
        (&[Some(1), Some(2), None], &(), 3),
        (&[None, Some(1), Some(2)], &keyed, 1),
    ];
    for (script, headers, asks) in rows {
        let ring = Scripted(script, AtomicU32::new(0));
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        for making in 0..2 {
            let facts = made.make(engine.standing_spec()).facts;
            assert!(facts.ring.is_none(), "{script:?}, making {making}");
            assert!(facts.ring_positions.is_empty(), "{script:?}");
            let asked = ring.1.load(atomic::Ordering::Relaxed);
            assert_eq!(asked, asks, "{script:?}, making {making}");
        }
    }
}

/// A pair's request is lent the ring where a side's ring stage places a
/// routing value by its own text, the prefill's or the decode's: both
/// sides are handed the one ring, and each takes it by its own stage
/// ([`RequestFacts::ring`]). The prefix form is lent it by the same rule,
/// and states the lender's place of the routing value as a head. A side
/// that reads no ring holds nothing back. A pair with no ring stage is
/// lent none: the caller's ring is asked nothing. A pair of two ring forms
/// or of two depths has no one list of places, and the engine's two
/// questions answer nothing for it: each of its sides is stated its own
/// (`pair::ring_words`).
#[test]
fn a_pairs_request_is_lent_the_ring_a_side_of_it_places_by() {
    let list = pool(2);
    let headers = [("x-smg-routing-key", "conv-1")];
    let ring = Lent(vec![list[1].url().to_owned(), list[0].url().to_owned()]);
    let stated = |document: PolicySpec, ring: &dyn LentRing| {
        let engine = engine_of(&document);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let pair = pair_of(&members, &classes);
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        let made = Made::new(&engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        (facts.ring.is_some(), facts.ring_positions.to_vec())
    };
    let pair = |prefill: &str, decode: &str| {
        PolicySpec::pair_presets(prefill, decode).expect("the pair presets")
    };
    let (ringed, prefixed, plain) = ("consistent_hashing", "prefix_hash", "round_robin");
    let lent = (true, vec![ring.position("conv-1")]);
    assert_eq!(stated(pair(ringed, ringed), &ring), lent);
    assert_eq!(stated(pair(plain, ringed), &ring), lent);
    assert_eq!(stated(pair(ringed, plain), &ring), lent);
    let head = ring.head_position(Head::Text("conv-1"));
    let lent = (true, head.into_iter().collect::<Vec<_>>());
    assert_eq!(lent.1.len(), 1);
    assert_eq!(stated(pair(prefixed, prefixed), &ring), lent);
    assert_eq!(stated(pair(plain, prefixed), &ring), lent);
    assert_eq!(stated(pair(prefixed, plain), &ring), lent);
    let none = (false, Vec::new());
    assert_eq!(stated(pair(plain, plain), &Unasked), none);
    // The engine's own word, phase by phase: a pair asked at its prefill
    // is answered for both its profiles, and a decode asked alone for its
    // own.
    let depths = |document: PolicySpec| {
        let engine = engine_of(&document);
        let spec = engine.standing_spec();
        let rule = |phase| spec.engine().states_head_places(MODEL, phase);
        [Phase::Prefill, Phase::Decode].map(|phase| rule(phase).map(|rule| rule.depth))
    };
    assert_eq!(depths(pair(prefixed, plain)), [Some(256), None]);
    assert_eq!(depths(pair(plain, prefixed)), [Some(256), Some(256)]);
    assert_eq!(depths(pair(prefixed, ringed)), [None, None]);
    assert_eq!(depths(pair(ringed, prefixed)), [None, Some(256)]);
}

/// An `epd` binding pairs its prefill and its decode as a `pd-pair`
/// binding does: asked at its prefill, the engine answers the lent ring's
/// two questions for the decode profile too. Here the decode profile alone
/// reads the ring, by a routing value's own text and then by heads.
#[test]
fn an_epd_models_prefill_is_answered_for_its_decode_profile_on_a_lent_ring() {
    let epd = |prefill: &str, decode: &str| {
        PolicySpec::epd_presets("round_robin", prefill, decode).expect("the epd presets")
    };
    let (ringed, prefixed, plain) = ("consistent_hashing", "prefix_hash", "round_robin");
    let phases = [Phase::Prefill, Phase::Decode];
    let by_text = |document: PolicySpec| {
        let engine = engine_of(&document);
        phases.map(|phase| engine.spec().engine().states_key_place(MODEL, phase))
    };
    assert_eq!(by_text(epd(plain, ringed)), [true, true]);
    assert_eq!(by_text(epd(ringed, plain)), [true, false]);
    assert_eq!(by_text(epd(plain, plain)), [false, false]);
    let depths = |document: PolicySpec| {
        let engine = engine_of(&document);
        let spec = engine.standing_spec();
        let rule = |phase| spec.engine().states_head_places(MODEL, phase);
        phases.map(|phase| rule(phase).map(|rule| rule.depth))
    };
    assert_eq!(depths(epd(plain, prefixed)), [Some(256), Some(256)]);
    assert_eq!(depths(epd(prefixed, plain)), [Some(256), None]);
    assert_eq!(depths(epd(plain, plain)), [None, None]);
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
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    // The stage that left the request no worker; none where it is placed.
    let emptied_by = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
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
    assert_eq!(made.make(engine.standing_spec()).facts.headers, [last]);
}

/// A later document whose view differs in its header list alone keeps the
/// profile's stages, however the list changed. A kept stage that reads a
/// header by its place reads that place of the later list. A stage built
/// anew from the later document reads the same.
#[test]
fn a_stage_kept_across_a_new_header_list_reads_its_place_of_that_list() {
    /// Draws among the workers the filters keep, and notes the value the
    /// request is shown at the view's first place: the empty text for none.
    #[derive(Debug)]
    struct ReadsFirst(Arc<Mutex<Vec<String>>>);

    impl Lifecycle for ReadsFirst {}

    impl Picker for ReadsFirst {
        fn name(&self) -> &'static str {
            "reads-first"
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
            let first = facts.headers.iter().find(|header| header.name == 0);
            let value = first.map_or("", |header| header.value);
            self.0.lock().push(value.to_owned());
            push_drawn(candidates, kept, out);
        }
    }

    let read = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&read);
    let mut stages = StageRegistry::builtin();
    let make = move || -> Box<dyn Picker> { Box::new(ReadsFirst(Arc::clone(&noted))) };
    stages.picker("reads-first", make).expect("a free name");
    let listing = |generation: u64, headers: &[&str]| {
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "reads-first".into();
        document.view.headers = headers.iter().map(|name| (*name).to_owned()).collect();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        spec.expect("the stages hold the picker")
    };
    let members = PublishedList::new(pool(2));
    // What the picker in force notes of a request that carries two headers.
    let asked = |engine: &'static InProcess| {
        let headers = [("x-a", "one"), ("x-b", "two")];
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        assert!(engine.place(&request).is_ok(), "a placement");
        std::mem::take(&mut *read.lock())
    };
    let engine = hosted(InProcess::new(listing(1, &["x-a"])));
    assert_eq!(asked(engine), ["one"]);
    // The later list, and the value at its first place. The list grew at
    // its end, shrank from its end, took another header first, and took a
    // first header the request does not carry.
    let later: [(&[&str], &str); 4] = [
        (&["x-a", "x-b"], "one"),
        (&["x-a"], "one"),
        (&["x-b", "x-a"], "two"),
        (&["x-c", "x-a"], ""),
    ];
    for (generation, (headers, value)) in (2..).zip(later) {
        let stood = engine.standing_spec().engine();
        let applied = engine.apply(listing(generation, headers));
        applied.expect("a number above the standing one");
        let stands = engine.standing_spec().engine();
        let kept = Arc::ptr_eq(&stands.profiles()[0], &stood.profiles()[0]);
        assert!(kept, "{headers:?}: the profile stands");
        assert_eq!(asked(engine), [value], "{headers:?}: the stage kept");
        let anew = hosted(InProcess::new(listing(generation, headers)));
        assert_eq!(asked(anew), [value], "{headers:?}: a stage built anew");
    }
}

/// A later document that lists another header ahead of the sessions rule's
/// keeps the profile's stages, and the rule reads its key from its own
/// header at its new place. A pinned key's request follows its pin, though
/// the header now first carries a text no key holds. A request that
/// carries a pinned key's text under that first header, and a new key
/// under the rule's own, is a new key's.
#[test]
fn a_sessions_rule_reads_its_header_behind_a_new_header_list() {
    let listing = |generation: u64, headers: &[&str]| {
        let mut document = PolicySpec::preset("manual").expect("a built-in preset");
        document.generation = generation;
        document.view.headers = headers.iter().map(|name| (*name).to_owned()).collect();
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let engine = hosted(InProcess::new(listing(1, &[MANUAL_SESSION_HEADER])));
    let members = PublishedList::new(pool(16));
    let went = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let mut placement = engine.place(&request).expect("a placement");
        placement.sent(Side::Single);
        placement.id()
    };
    let pins = || engine.standing_spec().engine().sessions().len();
    let keys = ["talk-1", "talk-2", "talk-3", "talk-4"];
    let homes = keys.map(|key| went(&[(MANUAL_SESSION_HEADER, key)]));
    assert_eq!(pins(), keys.len());

    let stood = engine.standing_spec().engine();
    let applied = engine.apply(listing(2, &["x-first", MANUAL_SESSION_HEADER]));
    applied.expect("a number above the standing one");
    let stands = engine.standing_spec().engine();
    let kept = Arc::ptr_eq(&stands.profiles()[0], &stood.profiles()[0]);
    assert!(kept, "the profile stands");
    for (key, home) in keys.into_iter().zip(homes) {
        let headers = [("x-first", "talk-none"), (MANUAL_SESSION_HEADER, key)];
        assert_eq!(went(&headers), home, "{key} follows its pin");
    }
    assert_eq!(pins(), keys.len(), "no new key");
    let headers = [("x-first", "talk-1"), (MANUAL_SESSION_HEADER, "talk-new")];
    went(&headers);
    assert_eq!(pins(), keys.len() + 1, "a new key");
}

/// The profile's picker receives the request's stated budget in microseconds.
/// The request gets a leased pick.
#[test]
fn the_picker_is_told_the_budget_a_request_states() {
    const NAME: &str = "record-budget";

    #[derive(Debug)]
    struct BudgetPicker(Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for BudgetPicker {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }
    }

    impl Picker for BudgetPicker {
        fn name(&self) -> &'static str {
            NAME
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
            self.0.lock().push(facts.budget_micros);
            push_drawn(candidates, kept, out);
        }
    }

    let budgets = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&budgets);
    let mut stages = StageRegistry::builtin();
    stages
        .picker(NAME, move || Box::new(BudgetPicker(Arc::clone(&observed))))
        .expect("a free picker name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    let profile = document
        .profiles
        .get_mut("random")
        .expect("the random profile");
    profile.pick = NAME.into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("a valid document");
    let engine = hosted(InProcess::new(spec));

    let members = PublishedList::new(pool(3));
    let request = PlaceRequest {
        budget_micros: 5_000,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(placement.owes_events(), "the pick has a lease");
    assert_eq!(budgets.lock().as_slice(), [5_000]);
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
