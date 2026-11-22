//! The cases of a frame's request maker: what the engine over worker lists
//! is handed of a frame, and what it answers.
//!
//! Each case names the place the controller on main answers by today, the
//! engine over stored records and its read of a frame.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use super::FrameRequest;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, ShedReason, MAX_TARGETS, SUBSET_HINT},
        decision_engine::{Decided, Plain, Request, RequestMaker, WorkerLists},
        policy_engine::{PolicyEngine, PoolMembers},
        view::{
            fingerprint, Constraints, HashKind, Header, Identity, MediaCounts, Phase, RequestFacts,
            WireModes, WorkerId,
        },
    },
    service::{
        controller::wire::{tests::full, Checked, Lists},
        proto,
    },
};

const MODEL: &str = "model-a";

/// What every case's frame has left of its budget at its decision.
const LEFT: Duration = Duration::from_micros(1_500);

fn applied(document: &PolicySpec) -> AppliedSpec {
    AppliedSpec::validate(document).expect("the document validates")
}

fn preset(name: &str) -> PolicySpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    document
}

fn engine(name: &str) -> PolicyEngine {
    PolicyEngine::new(applied(&preset(name)))
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// A frame that states its model and its seed, and what `edit` writes.
fn frame_of(seed: u64, edit: impl FnOnce(&mut proto::DecideRequest)) -> Checked {
    let mut frame = proto::DecideRequest {
        request_id: vec![1],
        facts: Some(proto::RequestFacts {
            model: MODEL.to_owned(),
            seed,
            ..proto::RequestFacts::default()
        }),
        ..proto::DecideRequest::default()
    };
    edit(&mut frame);
    Checked::new(frame).expect("a frame the record holds")
}

/// A request stated ahead with the list its caller hands for it, decided
/// whole: what a frame's maker is held against.
struct Stated<'a> {
    request: Request<'a>,
    members: &'a PoolMembers,
}

impl RequestMaker for Stated<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members)
    }
}

/// A frame's maker that says its request plain, as no controller's does.
struct SaysPlain<'a>(FrameRequest<'a>);

impl RequestMaker for SaysPlain<'_> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        Some(Plain {
            phase: Phase::Single,
            wires: WireModes::NONE,
            prompt_tokens: 0,
        })
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.0.workers()
    }
}

/// The engine's answer to `maker` at `now`.
fn decided(engine: &PolicyEngine, maker: &impl RequestMaker, now: Instant) -> Decided {
    engine.decide_at(maker, || now).expect("an answer")
}

/// The places an answer names, best first: a place alone, or a whole
/// pick's targets. None for an answer that picks nobody.
fn places(decided: &Decided) -> Vec<u32> {
    match decided {
        Decided::Alone(pick) => vec![pick.place],
        Decided::Whole(whole, _) => match &whole.decision {
            Decision::Pick(pick) => pick.targets.iter().map(|target| target.position).collect(),
            Decision::Shed(_) | Decision::Queue(_) => Vec::new(),
        },
    }
}

/// Every field of a frame reaches the request its maker makes, as it
/// reaches the request of the engine over stored records
/// (`Checked::read`, and its case over the same frame in the wire's own
/// cases). Four things are a controller's own and not the frame's: the
/// budget is what is left, the wires and the subset hint are not stated,
/// and the places and held-back positions are the ones the maker was
/// handed.
#[test]
fn every_field_of_a_frame_reaches_the_request_its_maker_makes() {
    // A ring keyed by the prompt reads a text's hashes: the frame's three
    // are inside its depth.
    let spec = applied(&preset("prefix_hash"));
    let frame = Checked::new(full()).expect("a frame the record holds");
    let mut buffers = Lists::default();
    let lists = buffers.of(&frame);
    let (may_take, stalled, held) = ([0_u32, 1, 3], [1_u32, 3], [3_u32]);
    let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None).held_to(
        Some(&may_take),
        &stalled,
        &held,
    );
    assert_eq!(maker.model(), "model-a");
    assert!(matches!(maker.workers(), WorkerLists::None));
    assert!(maker.loads().is_none(), "the engine holds the snapshot");

    let Request {
        id,
        facts,
        constraints,
    } = maker.make(&spec);
    assert_eq!(id, [7, 8, 9]);
    assert_eq!(facts.model, "model-a");
    assert_eq!(facts.namespace_seed, 11);
    assert_eq!(facts.tenant, Identity::plain("tenant-a"));
    assert_eq!(facts.tenant.hash, fingerprint(b"tenant-a"));
    assert_eq!(facts.priority, -100);
    assert!(facts.streaming);
    assert_eq!((facts.max_output_tokens, facts.prompt_tokens), (12, 13));
    assert_eq!(facts.block_hashes, [14, 15, 16]);
    assert_eq!(facts.hash_kind, HashKind::Text);
    // The hash form is the session's value, as it was sent.
    assert_eq!(facts.session().get(), 17);
    assert_eq!(facts.session_key.plain, None);
    assert_eq!(facts.phase, Phase::Decode);
    assert_eq!(facts.attempt, 2);
    assert_eq!(facts.tried, [WorkerId(18), WorkerId(19)]);
    let headers = [(0, "first"), (255, "last")].map(|(name, value)| Header { name, value });
    assert_eq!(facts.headers, headers);
    let media = MediaCounts {
        images: 20,
        audio: 21,
        video: 22,
        bytes: 23,
    };
    assert_eq!(facts.media, media);
    assert_eq!((facts.seed, facts.block_size), (24, 25));
    assert_eq!(facts.client_key, 29);
    // The frame's budget of 2,000 is not what the engine is told is left,
    // and the decision's instant is the engine's to write.
    assert_eq!((facts.budget_micros, facts.now_micros), (1_500, 0));
    assert!(facts.affinity.is_empty() && facts.named.is_none());
    assert!(facts.lent_in_flight.is_none() && facts.signal_cells.is_none());

    assert_eq!(
        constraints.excluded,
        [WorkerId(26), WorkerId(27), WorkerId(28)]
    );
    assert!(constraints.overload_lifted);
    assert!(!constraints.decode_later);
    // The frame states the wires 0b110 and a hint of 0: neither is said
    // to the engine.
    assert_eq!(constraints.wire_modes, WireModes::NONE);
    assert_eq!(constraints.subset_hint, None);
    assert_eq!(constraints.places, Some(&may_take[..]));
    assert_eq!(
        (constraints.stalled, constraints.held),
        (&stalled[..], &held[..])
    );
    assert!(constraints.pins.is_none() && constraints.demands.is_none());

    // Before it is held to anything, a request may take every place.
    let unheld = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
    let free = unheld.make(&spec).constraints;
    assert_eq!(free.places, None);
    assert!(free.stalled.is_empty() && free.held.is_empty());
    // More left than the field holds is told as its widest, as the read for
    // the engine over stored records tells it.
    let long = FrameRequest::new(
        &frame,
        &lists,
        Duration::from_secs(7_200),
        WorkerLists::None,
    );
    assert_eq!(long.make(&spec).facts.budget_micros, u32::MAX);
}

/// A decider reads frame after frame onto the same buffers: a later frame's
/// request holds its own lists and nothing of an earlier frame's, as the
/// read for the engine over stored records hands them
/// (`a_decider_reads_frame_after_frame_on_the_same_buffers` in the wire's
/// own cases).
#[test]
fn a_later_frames_request_holds_nothing_of_an_earlier_frames_lists() {
    let spec = applied(&preset("random"));
    let mut buffers = Lists::default();
    let first = Checked::new(full()).expect("a frame the record holds");
    {
        let lists = buffers.of(&first);
        let maker = FrameRequest::new(&first, &lists, LEFT, WorkerLists::None);
        let Request {
            facts, constraints, ..
        } = maker.make(&spec);
        assert_eq!(
            (
                facts.tried.len(),
                facts.headers.len(),
                constraints.excluded.len()
            ),
            (2, 2, 3)
        );
    }
    // A frame with one worker excluded, none tried and no header.
    let second = frame_of(7, |frame| {
        frame.constraints = Some(proto::Constraints {
            excluded: vec![5],
            ..proto::Constraints::default()
        });
    });
    let lists = buffers.of(&second);
    let maker = FrameRequest::new(&second, &lists, LEFT, WorkerLists::None);
    let Request {
        facts, constraints, ..
    } = maker.make(&spec);
    assert!(facts.tried.is_empty() && facts.headers.is_empty());
    assert_eq!(constraints.excluded, [WorkerId(5)]);
}

/// A frame's decision is a whole one and names its alternates, up to four
/// (`MAX_TARGETS`; a rotation's whole pick, `policies/round_robin.rs`), as
/// the engine over stored records answers every frame: a gateway's
/// dispatch names a target of the answer by its index. So the maker never
/// says its request plain. Said plain, the same frame is answered the one
/// place a caller in process takes.
#[test]
fn a_frames_decision_names_its_alternates() {
    let engine = engine("round_robin");
    let list = pool(6);
    let members = engine.pool_members(&list);
    let frame = frame_of(7, |_| {});
    let mut buffers = Lists::default();
    let lists = buffers.of(&frame);
    let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(&members));
    assert_eq!(maker.plain(), None);

    let now = Instant::now();
    let named = places(&decided(&engine, &maker, now));
    assert_eq!(named.len(), MAX_TARGETS, "{named:?}");
    let mut apart = named.clone();
    apart.sort_unstable();
    apart.dedup();
    assert_eq!(apart.len(), MAX_TARGETS, "four workers: {named:?}");

    let alone = places(&decided(&engine, &SaysPlain(maker), now));
    assert_eq!(alone.len(), 1, "{alone:?}");
}

/// The hashes past what the deciding document reads are cut where the
/// request is made, and the first ones kept, as the read for the engine
/// over stored records cuts them (`Viewed::max_block_hashes`): a prefix
/// cut short is still a prefix.
#[test]
fn the_hashes_past_what_the_document_reads_are_cut_and_the_first_kept() {
    let frame = frame_of(7, |frame| {
        let facts = frame.facts.as_mut().expect("the frame has facts");
        facts.block_hashes = (1..=12).collect();
    });
    let mut buffers = Lists::default();
    let lists = buffers.of(&frame);
    let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
    let read = |document: &PolicySpec| {
        let spec = applied(document);
        maker.make(&spec).facts.block_hashes.to_vec()
    };
    // A prefix key 256 tokens deep, in blocks of the view's 32 tokens.
    let mut keyed = preset("prefix_hash");
    keyed.view.block_size = Some(32);
    assert_eq!(read(&keyed), (1..=8).collect::<Vec<u64>>());
    // A stage with no depth of its own reads to the view's cap.
    assert_eq!(read(&preset("cache_aware")), (1..=12).collect::<Vec<u64>>());
    // No stage of this document reads a hash.
    assert_eq!(read(&preset("random")), Vec::<u64>::new());
}

/// A positive subset hint reaches no stage, as on main, where the
/// deciding threads say so of the engine over stored records: the frame
/// decides as one that states none. The engine over worker lists declines
/// a request that states a hint, so the maker leaves it out.
#[test]
fn a_positive_subset_hint_reaches_no_stage() {
    let engine = engine("random");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let now = Instant::now();
    let asked = |seed: u64, hint: Option<u32>| {
        let frame = frame_of(seed, |frame| {
            frame.constraints = Some(proto::Constraints {
                subset_hint: hint,
                ..proto::Constraints::default()
            });
        });
        let mut buffers = Lists::default();
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(&members));
        places(&decided(&engine, &maker, now))
    };
    for seed in 0..16 {
        let free = asked(seed, None);
        assert!(!free.is_empty(), "a pick at seed {seed}");
        assert_eq!(asked(seed, Some(2)), free, "seed {seed}");
    }

    let stated = Stated {
        request: Request {
            id: &[1],
            facts: RequestFacts {
                model: MODEL,
                ..RequestFacts::default()
            },
            constraints: Constraints {
                subset_hint: Some(2),
                ..Constraints::default()
            },
        },
        members: &members,
    };
    let Decided::Whole(whole, _) = decided(&engine, &stated, now) else {
        panic!("a whole decision");
    };
    match whole.decision {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, SUBSET_HINT)
        ),
        other => panic!("a shed, not {other:?}"),
    }
}

/// One frame and the equal request stated ahead decide alike over one list
/// of worker objects. Two engines of one document are each asked the same
/// requests in the same order: a rotation's turns are an engine's own.
#[test]
fn a_frame_and_the_equal_request_stated_ahead_decide_alike() {
    for name in ["round_robin", "random", "power_of_two"] {
        let (by_frame, by_request) = (engine(name), engine(name));
        let list = pool(8);
        let (framed, stated) = (by_frame.pool_members(&list), by_request.pool_members(&list));
        let now = Instant::now();
        for seed in 0..64 {
            let frame = frame_of(seed, |_| {});
            let mut buffers = Lists::default();
            let lists = buffers.of(&frame);
            let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(&framed));
            let request = Stated {
                request: Request {
                    id: &[1],
                    facts: RequestFacts {
                        model: MODEL,
                        seed,
                        budget_micros: 1_500,
                        ..RequestFacts::default()
                    },
                    constraints: Constraints::default(),
                },
                members: &stated,
            };
            let named = places(&decided(&by_frame, &maker, now));
            assert!(!named.is_empty(), "{name} picks at seed {seed}");
            assert_eq!(
                named,
                places(&decided(&by_request, &request, now)),
                "{name} at seed {seed}"
            );
        }
    }
}
