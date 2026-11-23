//! The cases of a frame's request maker: what the engine over worker lists
//! is handed of a frame, and what it answers.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::{one_rank_load, LoadSnapshot},
    test_support::{no_health_check, ready_worker},
    worker::{WorkerSpec, WorkerStatus, WorkerType},
};

use super::FrameRequest;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, ShedReason, MAX_TARGETS, SUBSET_HINT},
        decision_engine::{Decided, Plain, Request, RequestMaker, WorkerLists},
        list_cell::tests::PublishedList,
        pair_classes::PartnerClasses,
        policy_engine::PolicyEngine,
        view::{
            fingerprint, Constraints, FixedFacts, HashKind, Header, Identity, MediaCounts,
            ModelFingerprint, Phase, RequestFacts, SessionCountedUnder, WireModes, WorkerId,
        },
        PairAdmission, PairLists,
    },
    service::{
        controller::wire::{
            tests::{full, kept, made},
            Checked, Lists, Refusal,
        },
        proto::{self, identity::Form},
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

/// A ready worker of `worker_type`, a side of a pair.
fn typed(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
    let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
        worker_type,
        ..WorkerSpec::new(url)
    })
    .health_config(no_health_check())
    .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A host's admission with every prefill free and every decode's room open.
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

/// The frame whose every field holds a value of its own ([`full`]), with
/// its facts edited.
fn edited(edit: impl FnOnce(&mut proto::RequestFacts)) -> proto::DecideRequest {
    let mut frame = full();
    edit(frame.facts.as_mut().expect("the full frame has facts"));
    frame
}

/// The request a controller's maker makes of `frame`, a frame the record
/// holds, handed to `check` as its facts and its constraints ([`made`]).
fn making<R>(
    frame: proto::DecideRequest,
    check: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
) -> R {
    let frame = Checked::new(frame).expect("a frame the record holds");
    made(&frame, &mut Lists::default(), LEFT, check)
}

/// A request stated ahead with the list its caller hands for it, decided
/// whole: what a frame's maker is held against.
struct Stated<'a> {
    request: Request<'a>,
    members: &'a PublishedList,
}

impl RequestMaker for Stated<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members.lent())
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

/// The reason and the stage of an answer that sheds its request.
fn shed_of(decided: Decided) -> (ShedReason, &'static str) {
    let Decided::Whole(whole, _) = decided else {
        panic!("a whole decision");
    };
    match whole.decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("a shed, not {other:?}"),
    }
}

/// Every field of a frame reaches the request its maker makes.
/// Four things are a controller's own and not the frame's: the
/// budget is what is left, the wires are not stated, the places and
/// held-back positions are the ones the maker was handed, and a key's
/// requests are said counted
/// (`a_keyed_frames_request_says_its_keys_requests_are_counted`). Of the
/// subset hint a 0 alone is stated. The frame's whole budget and the wires
/// its gateway states are read on the frame, where it was checked: no
/// request states them. The frame gives its id whole for its answer.
#[test]
fn every_field_of_a_frame_reaches_the_request_its_maker_makes() {
    // A ring keyed by the prompt reads a text's hashes: the frame's three
    // are inside its depth.
    let spec = applied(&preset("prefix_hash"));
    let frame = Checked::new(full()).expect("a frame the record holds");
    assert_eq!(
        (frame.model(), frame.phase(), frame.seed()),
        ("model-a", Phase::Decode, 24)
    );
    assert_eq!(frame.budget().as_micros(), 2_000);
    assert_eq!(frame.wires(), Some(WireModes::GRPC.union(WireModes::ZMQ)));
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
    // The frame carries its model's name and no fingerprint of it: each
    // read of the request's session fingerprints the name.
    assert_eq!(facts.model_fingerprint, ModelFingerprint::NONE);

    assert_eq!(
        constraints.excluded,
        [WorkerId(26), WorkerId(27), WorkerId(28)]
    );
    assert!(constraints.overload_lifted);
    assert!(!constraints.decode_later);
    // The frame states the wires 0b110 and a hint of 0: the wires are not
    // said to the engine, and the 0 is.
    assert_eq!(constraints.wire_modes, WireModes::NONE);
    assert_eq!(constraints.subset_hint, Some(0));
    assert_eq!(constraints.places, Some(&may_take[..]));
    assert_eq!(
        (constraints.stalled, constraints.held),
        (&stalled[..], &held[..])
    );
    assert!(constraints.demands.is_none());

    // Before it is held to anything, a request may take every place.
    let unheld = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
    let free = unheld.make(&spec).constraints;
    assert_eq!(free.places, None);
    assert!(free.stalled.is_empty() && free.held.is_empty());
    // More left than the field holds is told as its widest.
    let long = FrameRequest::new(
        &frame,
        &lists,
        Duration::from_secs(7_200),
        WorkerLists::None,
    );
    assert_eq!(long.make(&spec).facts.budget_micros, u32::MAX);
    // A part of a microsecond left is not told: the whole ones are.
    let part = FrameRequest::new(
        &frame,
        &lists,
        Duration::from_nanos(1_999),
        WorkerLists::None,
    );
    assert_eq!(part.make(&spec).facts.budget_micros, 1);
    // The id goes back whole on the frame's answer.
    let answered = Checked::new(full()).expect("a frame the record holds");
    assert_eq!(answered.into_id(), [7, 8, 9]);
}

/// A controller serves no request: the decisions it holds on file count a
/// key's requests on each worker. So the request its maker makes of a keyed
/// frame says that the count stands, the key in either form, and the maker
/// says under what name: the session's number. A frame with no key states
/// no count. The frame's one reader states none of this: the maker does.
#[test]
fn a_keyed_frames_request_says_its_keys_requests_are_counted() {
    let spec = applied(&preset("random"));
    let counted = |key: Option<Form>| {
        let frame = frame_of(1, |frame| {
            let facts = frame.facts.as_mut().expect("the frame has facts");
            facts.session_key = key.map(|form| proto::Identity { form: Some(form) });
        });
        let mut buffers = Lists::default();
        let lists = buffers.of(&frame);
        assert!(!frame.facts(&lists, 0, 0).session_counted);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
        let under = maker.session_counted_under();
        assert_eq!(under, SessionCountedUnder::SessionNumber);
        maker.make(&spec).facts.session_counted
    };
    assert!(counted(Some(Form::Hash(17))));
    let clear = Form::Plain("talk-1".to_owned());
    assert!(counted(Some(clear)));
    assert!(!counted(None));
}

/// A decider reads frame after frame onto the same buffers: a later frame's
/// request holds its own lists and nothing of an earlier frame's.
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

/// A decider makes request after request on the same buffers. A frame's
/// headers end with its request, and the buffer they leave is the one the
/// next frame's take, emptied: a frame with shorter lists makes its own,
/// nothing of the last, and gives no buffer up
/// (`a_decider_reads_frame_after_frame_on_the_same_buffers` in the wire's
/// own cases).
#[test]
fn a_decider_makes_request_after_request_on_the_same_buffers() {
    let spec = applied(&preset("random"));
    let mut buffers = Lists::default();
    // The lists of `frame`'s request, made on a decider's buffers.
    let lists_of = |buffers: &mut Lists, frame: proto::DecideRequest| {
        let frame = Checked::new(frame).expect("a frame the record holds");
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
        let Request {
            facts, constraints, ..
        } = maker.make(&spec);
        let headers = facts.headers.iter();
        let headers: Vec<(u8, String)> = headers
            .map(|header| (header.name, header.value.to_owned()))
            .collect();
        (facts.tried.to_vec(), headers, constraints.excluded.to_vec())
    };
    // Eight headers: a buffer made anew for a shorter list holds fewer.
    let long = edited(|facts| {
        facts.headers = (0..8)
            .map(|name| proto::Header {
                name,
                value: "long".to_owned(),
            })
            .collect();
    });
    let (tried, headers, excluded) = lists_of(&mut buffers, long);
    assert_eq!((tried.len(), headers.len(), excluded.len()), (2, 8, 3));
    let (room, standing) = kept(&buffers);
    assert!(room[0] >= 2 && room[1] >= 3 && room[2] >= 8, "{room:?}");
    assert_eq!(standing, 0, "a frame's headers end with its request");
    // A frame with shorter lists makes its own, and nothing of the last.
    let shorter = edited(|facts| {
        facts.tried = vec![1];
        facts.headers.truncate(1);
    });
    let (tried, headers, _) = lists_of(&mut buffers, shorter);
    assert_eq!(tried, [WorkerId(1)]);
    assert_eq!(headers, [(0, "first".to_owned())]);
    assert_eq!(kept(&buffers), (room, 0), "the buffers are kept");
    // A frame with no list at all leaves each buffer its room too.
    let bare = proto::DecideRequest {
        request_id: vec![1],
        ..proto::DecideRequest::default()
    };
    let (tried, headers, excluded) = lists_of(&mut buffers, bare);
    assert!(tried.is_empty() && headers.is_empty() && excluded.is_empty());
    assert_eq!(kept(&buffers), (room, 0), "the buffers are kept");
}

/// A frame with nothing set but its id makes a request of the record's
/// defaults: no model, a single request of token hashes, no key, no count
/// and no list, and neither a wire nor a hint
/// (`a_frame_with_nothing_set_reads_as_the_records_defaults` in the wire's
/// own cases).
#[test]
fn a_frame_with_nothing_set_makes_a_request_of_the_records_defaults() {
    // Nothing but its id, which the contract's range asks of every frame.
    let bare = proto::DecideRequest {
        request_id: vec![1],
        ..proto::DecideRequest::default()
    };
    let bare = Checked::new(bare).expect("a frame of its id alone");
    assert_eq!(
        (bare.model(), bare.phase(), bare.seed()),
        ("", Phase::Single, 0)
    );
    assert!(bare.budget().is_zero());
    assert_eq!((bare.wires(), bare.subset()), (None, None));
    made(&bare, &mut Lists::default(), LEFT, |facts, constraints| {
        assert_eq!(facts.model, "");
        assert_eq!(
            (facts.phase, facts.hash_kind),
            (Phase::Single, HashKind::Token)
        );
        assert_eq!(
            (facts.tenant, facts.session_key),
            (Identity::NONE, Identity::NONE)
        );
        assert_eq!(facts.media, MediaCounts::default());
        assert_eq!((facts.seed, facts.block_size), (0, 0));
        assert!(facts.headers.is_empty() && facts.tried.is_empty());
        assert!(facts.block_hashes.is_empty());
        assert!(constraints.excluded.is_empty());
        assert_eq!(constraints.wire_modes, WireModes::NONE);
        assert_eq!(constraints.subset_hint, None);
    });
}

/// An enum reaches the request by its number, and a number that names no
/// value is refused by its field where the frame is checked, so no request
/// is made of it
/// (`an_enum_reads_by_its_number_and_a_number_that_names_no_value_is_refused_by_its_field`
/// in the wire's own cases).
#[test]
fn an_enum_is_made_by_its_number_and_a_number_that_names_no_value_is_refused_by_its_field() {
    for (wire, record) in [
        (proto::Phase::Unspecified, Phase::Single),
        (proto::Phase::Single, Phase::Single),
        (proto::Phase::Prefill, Phase::Prefill),
        (proto::Phase::Decode, Phase::Decode),
        (proto::Phase::Encode, Phase::Encode),
    ] {
        let frame = edited(|facts| facts.phase = wire.into());
        let read = making(frame, |facts, _| facts.phase);
        assert_eq!(read, record, "{wire:?}");
    }
    for (wire, record) in [
        (proto::HashKind::Unspecified, HashKind::Token),
        (proto::HashKind::Token, HashKind::Token),
        (proto::HashKind::Text, HashKind::Text),
    ] {
        let frame = edited(|facts| facts.hash_kind = wire.into());
        let read = making(frame, |facts, _| facts.hash_kind);
        assert_eq!(read, record, "{wire:?}");
    }
    let refused = |frame| Checked::new(frame).expect_err("refused");
    assert_eq!(
        refused(edited(|facts| facts.phase = 5)),
        Refusal::OutOfRange("facts.phase")
    );
    assert_eq!(
        refused(edited(|facts| facts.hash_kind = 3)),
        Refusal::OutOfRange("facts.hash_kind")
    );
    assert_eq!(
        refused(edited(|facts| facts.phase = 5)).to_string(),
        "facts.phase is out of its range in this protocol version"
    );
}

/// A text frame states its window, in bytes. One that carries hashes and
/// states none is refused by its field where it is checked: the view's
/// block size is in tokens, and read as a window it would count the
/// request's blocks at another size than its host hashed them. A text
/// frame with no hash prices nothing, and its request is made. A token
/// frame that states no size makes a request that states none: its hashes
/// are cut in the view's
/// (`a_text_request_that_states_no_window_is_refused_by_its_field` in the
/// wire's own cases).
#[test]
fn a_text_frame_that_states_no_window_is_refused_by_its_field() {
    let refused = |frame| Checked::new(frame).expect_err("refused");
    assert_eq!(
        refused(edited(|facts| facts.block_size = 0)),
        Refusal::OutOfRange("facts.block_size")
    );
    let bare = edited(|facts| {
        facts.block_size = 0;
        facts.block_hashes.clear();
    });
    let read = making(bare, |facts, _| {
        (facts.hash_kind, facts.block_size, facts.block_hashes.len())
    });
    assert_eq!(read, (HashKind::Text, 0, 0), "no hash to price");
    for kind in [proto::HashKind::Token, proto::HashKind::Unspecified] {
        let frame = edited(|facts| {
            facts.hash_kind = kind.into();
            facts.block_size = 0;
        });
        let read = making(frame, |facts, _| (facts.hash_kind, facts.block_size));
        assert_eq!(read, (HashKind::Token, 0), "{kind:?}");
    }
    let stated = edited(|facts| facts.block_size = 64);
    let read = making(stated, |facts, _| (facts.hash_kind, facts.block_size));
    assert_eq!(read, (HashKind::Text, 64));
}

/// A tenant reaches the request in the form its frame carries it: a
/// fingerprint as it was sent, and a plain name with the fingerprint made
/// of it at the read. No form, a fingerprint of zero and an empty name are
/// no key, and a frame that sets no session key makes a request with none
/// (`a_key_is_read_in_the_form_its_frame_carries` in the wire's own cases).
#[test]
fn a_key_is_made_in_the_form_its_frame_carries() {
    let tenant = |form: Option<Form>| {
        let frame = edited(|facts| facts.tenant = Some(proto::Identity { form }));
        making(frame, |facts, _| {
            (facts.tenant.hash, facts.tenant.plain.map(str::to_owned))
        })
    };
    assert_eq!(tenant(Some(Form::Hash(5))), (5, None));
    assert_eq!(
        tenant(Some(Form::Plain("a".to_owned()))),
        (fingerprint(b"a"), Some("a".to_owned()))
    );
    // No form, a fingerprint of zero and an empty name are no key.
    for absent in [None, Some(Form::Hash(0)), Some(Form::Plain(String::new()))] {
        assert_eq!(tenant(absent.clone()), (0, None), "{absent:?}");
    }
    let unset = edited(|facts| facts.session_key = None);
    assert_eq!(making(unset, |facts, _| facts.session_key.hash), 0);
}

/// The contract's own numbers: the key "conversation-7" under the model
/// "model-a" names one session a phase, and names it in either form. The
/// hash form is the session's value: it is never read as a key's
/// fingerprint and scoped again
/// (`a_session_key_names_the_contracts_session_in_its_plain_form_and_in_its_hash_form`
/// in the wire's own cases).
#[test]
fn a_frames_session_key_names_the_contracts_session_in_either_form() {
    let sessions: [(proto::Phase, u64); 4] = [
        (proto::Phase::Single, 0x6e88_ff71_cc98_2a42),
        (proto::Phase::Prefill, 0x619f_dbb0_c262_9a55),
        (proto::Phase::Decode, 0x3e92_fbbd_183d_f4b6),
        (proto::Phase::Encode, 0x3780_ba54_336a_57d0),
    ];
    let named = |phase: proto::Phase, form: Option<Form>| {
        let frame = edited(|facts| {
            facts.phase = phase.into();
            facts.session_key = Some(proto::Identity { form });
        });
        making(frame, |facts, _| {
            let key = facts.session_key;
            (
                facts.session().get(),
                key.hash,
                key.plain.map(str::to_owned),
            )
        })
    };
    let key = fingerprint(b"conversation-7");
    for (phase, session) in sessions {
        let plain = "conversation-7".to_owned();
        assert_eq!(
            named(phase, Some(Form::Plain(plain.clone()))),
            (session, key, Some(plain)),
            "{phase:?}"
        );
        // The same session, and the same key for a stage that reads it.
        assert_eq!(
            named(phase, Some(Form::Hash(session))),
            (session, key, None),
            "{phase:?}"
        );
    }
    // No form, a value of zero and an empty key are no session.
    for absent in [None, Some(Form::Hash(0)), Some(Form::Plain(String::new()))] {
        let read = named(proto::Phase::Single, absent.clone());
        assert_eq!(read, (0, 0, None), "{absent:?}");
    }
    // The one value that would give the fingerprint no key has is no key's
    // session: its frame is refused where it is checked.
    let keyless = edited(|facts| {
        facts.phase = proto::Phase::Single.into();
        facts.session_key = Some(proto::Identity {
            form: Some(Form::Hash(0xc771_96d2_b9cf_9497)),
        });
    });
    let refused = Checked::new(keyless).expect_err("no key's session");
    assert_eq!(refused, Refusal::OutOfRange("facts.session_key.hash"));
}

/// A count past its record's width is made at the record's widest. A
/// frame's headers are held to the 256 names a view has: a name past them
/// and a 257th header are refused where the frame is checked. A wire bit
/// this build does not know is dropped where the frame's wires are read,
/// and the request says no wire, whatever its frame states: its maker's
/// caller states the gateway's wires as places
/// (`a_count_past_its_records_width_saturates_and_a_header_past_256_names_is_refused`
/// in the wire's own cases).
#[test]
fn a_count_past_its_records_width_is_made_at_its_widest_and_a_header_past_256_names_is_refused() {
    let frame = edited(|facts| {
        facts.attempt = 256;
        facts.media = Some(proto::MediaCounts {
            images: 65_536,
            audio: u32::MAX,
            video: 65_535,
            bytes: u32::MAX,
        });
    });
    let (attempt, media) = making(frame, |facts, _| (facts.attempt, facts.media));
    assert_eq!(attempt, u8::MAX);
    let widest = MediaCounts {
        images: u16::MAX,
        audio: u16::MAX,
        video: 65_535,
        bytes: u32::MAX,
    };
    assert_eq!(media, widest);
    let frame = edited(|facts| facts.headers[1].name = 256);
    assert_eq!(
        Checked::new(frame).expect_err("a name past the record's"),
        Refusal::OutOfRange("facts.headers.name")
    );
    // A frame's headers by their count: 256 reach the request, each a
    // record of its own, and one more is refused by the list's path. An
    // empty header is two bytes as encoded.
    let headers = |count| edited(|facts| facts.headers = vec![proto::Header::default(); count]);
    let read = making(headers(256), |facts, _| facts.headers.len());
    assert_eq!(read, 256);
    assert_eq!(
        Checked::new(headers(257)).expect_err("a header past a view's names"),
        Refusal::OutOfRange("facts.headers")
    );
    let size = |count| {
        let facts = headers(count).facts.expect("the full frame has facts");
        prost::Message::encoded_len(&facts)
    };
    assert_eq!(size(257) - size(0), 2 * 257);
    // A wire bit this build does not know is dropped, wherever in the
    // wire's 32 it stands. A gateway that states such bits alone still
    // states wires: none this build knows. The request says none in each
    // case.
    let stating = |bits: u32| {
        let mut frame = full();
        frame
            .constraints
            .as_mut()
            .expect("the full frame has constraints")
            .wire_modes = bits;
        let frame = Checked::new(frame).expect("a frame the record holds");
        let said = made(&frame, &mut Lists::default(), LEFT, |_, constraints| {
            constraints.wire_modes
        });
        (frame.wires(), said)
    };
    assert_eq!(
        stating(0b1_0000_1001),
        (Some(WireModes::HTTP), WireModes::NONE)
    );
    assert_eq!(
        stating(0b1_0000_0000),
        (Some(WireModes::NONE), WireModes::NONE)
    );
    assert_eq!(stating(0), (None, WireModes::NONE));
}

/// A frame's decision is a whole one and names its alternates, up to four
/// (`MAX_TARGETS`; a rotation's whole pick, `policies/round_robin.rs`): a
/// gateway's dispatch names a target of the answer by its index. So the
/// maker never says its request plain. Said plain, the same frame is
/// answered the one place a caller in process takes.
#[test]
fn a_frames_decision_names_its_alternates() {
    let engine = engine("round_robin");
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let frame = frame_of(7, |_| {});
    let mut buffers = Lists::default();
    let lists = buffers.of(&frame);
    let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(members.lent()));
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

/// A request told the load snapshot its decision reads is decided by that
/// snapshot's reports: under the least-load preset every pick names the
/// worker that reports no queue. The same frames told no snapshot read the
/// one their engine keeps, which holds no report here, and the worker with
/// the queue is a pick among them.
#[test]
fn a_request_told_a_load_snapshot_is_decided_by_its_reports() {
    let engine = engine("least_load");
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = LoadSnapshot::from_loads_for_test(vec![
        (
            "http://worker-0:8000".to_owned(),
            one_rank_load(900_000, 0.9, 10.0),
        ),
        (
            "http://worker-1:8000".to_owned(),
            one_rank_load(0, 0.1, 10.0),
        ),
    ]);
    let now = Instant::now();
    let mut buffers = Lists::default();
    let (mut told, mut untold) = (Vec::new(), Vec::new());
    for seed in 0..32 {
        let frame = frame_of(seed, |_| {});
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(members.lent()));
        assert!(maker.loads().is_none());
        untold.push(places(&decided(&engine, &maker, now))[0]);
        let maker = maker.reading(&loads);
        assert!(maker.loads().is_some_and(|read| Arc::ptr_eq(read, &loads)));
        told.push(places(&decided(&engine, &maker, now))[0]);
    }
    assert!(told.iter().all(|place| *place == 1), "{told:?}");
    assert!(untold.contains(&0), "{untold:?}");
}

/// The hashes past what the deciding document reads are cut where the
/// request is made, and the first ones kept: a prefix cut short is
/// still a prefix.
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

/// The depth is counted in the frame's own blocks and in its own kind of
/// hash (`AppliedSpec::block_hashes_read`): a document that reads 256
/// tokens of a prompt reads sixteen blocks of 16 tokens and four of 64,
/// eight of the view's 32 where the frame states no size, and of a text
/// the bytes of as many tokens.
#[test]
fn the_hashes_are_cut_in_blocks_of_the_frames_own_size_and_kind() {
    let mut keyed = preset("prefix_hash");
    keyed.view.block_size = Some(32);
    let spec = applied(&keyed);
    let read = |kind: proto::HashKind, block_size: u32| {
        let frame = frame_of(7, |frame| {
            let facts = frame.facts.as_mut().expect("the frame has facts");
            facts.block_hashes = (1..=40).collect();
            facts.hash_kind = kind.into();
            facts.block_size = block_size;
        });
        let mut buffers = Lists::default();
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::None);
        maker.make(&spec).facts.block_hashes.len()
    };
    let tokens = |block_size: u32| read(proto::HashKind::Token, block_size);
    assert_eq!((tokens(0), tokens(16), tokens(64)), (8, 16, 4));
    // A text's blocks are sized in bytes, four to a token, and a frame that
    // carries a text's hashes states their size.
    let text = |block_size: u32| read(proto::HashKind::Text, block_size);
    assert_eq!((text(32), text(64)), (32, 16));
}

/// A hint of 0 is a decision over no candidates. The controller sheds such
/// a frame `NoEndpoints` under `subset-hint`. The maker keeps the 0, so the
/// answer cannot be lost: the frame's caller gives it before it asks the
/// engine (`GatewayPlaces::taken`), and the engine declines the single
/// request of a caller that did not, where it would have picked a worker.
#[test]
fn a_hint_of_0_reaches_the_engine_which_refuses_a_single_request_not_answered_for_it() {
    let engine = engine("random");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let frame = frame_of(7, |frame| {
        frame.constraints = Some(proto::Constraints {
            subset_hint: Some(0),
            ..proto::Constraints::default()
        });
    });
    let mut buffers = Lists::default();
    let lists = buffers.of(&frame);
    let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(members.lent()));
    let spec = applied(&preset("random"));
    assert_eq!(maker.make(&spec).constraints.subset_hint, Some(0));
    assert_eq!(
        shed_of(decided(&engine, &maker, Instant::now())),
        (ShedReason::PolicyDeclined, SUBSET_HINT)
    );
}

/// A pair's frame is decided over the two lists its controller hands for
/// it: its answer names a prefill and that prefill's decode. One that
/// hints a subset of 0 is shed `NoEndpoints` under `subset-hint`, as main
/// sheds it, and a positive hint decides as none.
#[test]
fn a_pairs_frame_is_decided_over_its_two_lists_and_shed_where_it_hints_0() {
    let mut document =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
    document.generation = 1;
    let engine = PolicyEngine::new(applied(&document));
    let side = |name: &str, worker_type: WorkerType, count: usize| -> Arc<[Arc<dyn Worker>]> {
        (0..count)
            .map(|at| typed(format!("grpc://{name}-{at}:9000"), worker_type))
            .collect()
    };
    let prefills = side("prefill", WorkerType::Prefill, 2);
    let decodes = side("decode", WorkerType::Decode, 3);
    let (prefill, decode) = (
        PublishedList::new(Arc::clone(&prefills)),
        PublishedList::new(Arc::clone(&decodes)),
    );
    // Every prefill pairs with every decode.
    let all = [0_u32, 1, 2];
    let classes = PartnerClasses::number(3, [&all[..], &all[..]]);
    let pair = PairLists {
        prefill: prefill.lent(),
        decode: decode.lent(),
        classes: &classes,
        admission: &Open,
        prefill_places: None,
        decode_places: None,
        prefill_stalled: &[],
        prefill_held: &[],
        decode_stalled: &[],
        decode_held: &[],
    };
    let now = Instant::now();
    let asked = |hint: Option<u32>| {
        let frame = frame_of(7, |frame| {
            let facts = frame.facts.as_mut().expect("the frame has facts");
            facts.phase = proto::Phase::Prefill.into();
            frame.constraints = Some(proto::Constraints {
                subset_hint: hint,
                ..proto::Constraints::default()
            });
        });
        let mut buffers = Lists::default();
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Pair(pair));
        decided(&engine, &maker, now)
    };
    for hint in [None, Some(2)] {
        let Decided::Whole(whole, _) = asked(hint) else {
            panic!("a whole decision");
        };
        let pick = whole.decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target");
        assert!(first.pair.is_some(), "a prefill with its decode: {first:?}");
    }
    assert_eq!(
        shed_of(asked(Some(0))),
        (ShedReason::NoEndpoints, SUBSET_HINT)
    );
}

/// A frame names the workers it tried and the ones it excludes by the
/// store's ids (`Lists::of`). Over a list whose records state those ids
/// (`FixedFacts::id`) an excluded worker is no target and each target
/// bears its worker's id from the store. Over records that carry an
/// address's fingerprint the same frame excludes nobody.
#[test]
fn a_frame_names_a_worker_by_the_stores_id_over_a_list_that_states_it() {
    let engine = engine("random");
    let list = pool(4);
    // The store's id of the worker at each place of the list.
    let stored = |place: usize| WorkerId(1_000 + place as u64);
    let mut place = 0;
    let stating = PublishedList::with_facts(Arc::clone(&list), |_| {
        place += 1;
        FixedFacts {
            id: Some(stored(place - 1)),
            ..FixedFacts::default()
        }
    });
    let silent = PublishedList::new(Arc::clone(&list));
    let now = Instant::now();
    let asked = |members: &PublishedList, seed: u64| {
        let frame = frame_of(seed, |frame| {
            frame.constraints = Some(proto::Constraints {
                excluded: vec![stored(1).0, stored(2).0],
                ..proto::Constraints::default()
            });
        });
        let mut buffers = Lists::default();
        let lists = buffers.of(&frame);
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(members.lent()));
        let Decided::Whole(whole, _) = decided(&engine, &maker, now) else {
            panic!("a whole decision");
        };
        let pick = whole.decision.pick().expect("a pick");
        pick.targets.iter().copied().collect::<Vec<_>>()
    };
    let mut unstated = Vec::new();
    for seed in 0..32 {
        for target in asked(&stating, seed) {
            assert!([0, 3].contains(&target.position), "seed {seed}: {target:?}");
            assert_eq!(target.id, stored(target.position as usize), "seed {seed}");
        }
        unstated.extend(asked(&silent, seed).iter().map(|target| target.position));
    }
    assert!(
        unstated.contains(&1) && unstated.contains(&2),
        "{unstated:?}"
    );
}

/// A positive subset hint reaches no stage: the frame decides as one that
/// states none. The engine over worker lists declines a request that
/// states a hint, so the maker leaves it out.
#[test]
fn a_positive_subset_hint_reaches_no_stage() {
    let engine = engine("random");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
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
        let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(members.lent()));
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
    assert_eq!(
        shed_of(decided(&engine, &stated, now)),
        (ShedReason::PolicyDeclined, SUBSET_HINT)
    );
}

/// One frame and the equal request stated ahead decide alike over one list
/// of worker objects. Two engines of one document are each asked the same
/// requests in the same order: a rotation's turns are an engine's own.
#[test]
fn a_frame_and_the_equal_request_stated_ahead_decide_alike() {
    for name in ["round_robin", "random", "power_of_two"] {
        let (by_frame, by_request) = (engine(name), engine(name));
        let list = pool(8);
        let (framed, stated) = (
            PublishedList::new(Arc::clone(&list)),
            PublishedList::new(Arc::clone(&list)),
        );
        let now = Instant::now();
        for seed in 0..64 {
            let frame = frame_of(seed, |_| {});
            let mut buffers = Lists::default();
            let lists = buffers.of(&frame);
            let maker = FrameRequest::new(&frame, &lists, LEFT, WorkerLists::Single(framed.lent()));
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
