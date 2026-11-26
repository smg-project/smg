use std::time::{Duration, Instant};

use super::{decided, encode_request, frame, placed, request};
use crate::{
    engine::{
        decision,
        decision::{
            Decision, LiftedVeto, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target, Targets,
            Trace,
        },
        decision_engine::Request,
        demands::{Demands, References},
        framework::{EncodeItems, EncodePlaced},
        view,
        view::{
            Constraints, HashKind, Header, Identity, MediaCounts, Phase, RequestFacts, WorkerId,
        },
    },
    service::{
        client::MAX_HASHES,
        controller::wire::{tests::made, Beside, Checked, Decided, Leased, Lists},
        proto::{
            self, decide_answer::Answer, decision::Outcome, gateway_frame::Frame, identity::Form,
        },
    },
    tracking::{
        lease,
        lease::{Lease, LeaseId, ReleaseReason, Said, Side, TransportClass},
    },
};

/// What a controller has left of a request's budget where it makes the
/// request of a frame.
const LEFT: Duration = Duration::from_micros(1_500);

/// Every field a host states of a request crosses the wire, and the
/// controller makes its request of the frame. Four things of that
/// request are the controller's own. Its budget is what the controller
/// has left. It says no wire: the wires the host states are read on
/// the frame, where it is checked. Its block hashes are cut at the
/// depth the deciding document reads, and of a subset hint it states
/// a 0 alone. The case's three hashes and its hint of 0 are inside
/// both, and the frame is read for each as it crossed.
#[test]
fn the_controller_reads_every_request_field_as_the_host_stated_it() {
    let mut asked = Request {
        id: b"a request",
        facts: RequestFacts {
            model: "model-a",
            namespace_seed: 11,
            tenant: Identity::plain("tenant-a"),
            priority: -100,
            streaming: true,
            max_output_tokens: 12,
            prompt_tokens: 13,
            block_hashes: &[14, 15, 16],
            hash_kind: HashKind::Text,
            session_key: Identity::hashed(17),
            phase: Phase::Decode,
            attempt: 2,
            tried: &[WorkerId(18), WorkerId(19)],
            budget_micros: 2_000,
            headers: &[
                Header {
                    name: 255,
                    value: "last",
                },
                Header {
                    name: 0,
                    value: "first",
                },
            ],
            media: MediaCounts {
                images: 20,
                audio: 21,
                video: 22,
                bytes: 23,
            },
            seed: 24,
            block_size: 25,
            client_key: 29,
            ..RequestFacts::default()
        },
        constraints: Constraints {
            excluded: &[WorkerId(26), WorkerId(27)],
            wire_modes: crate::engine::WireModes::HTTP.union(crate::engine::WireModes::GRPC),
            overload_lifted: true,
            subset_hint: Some(0),
            ..Constraints::default()
        },
    };
    for (tenant, session, phase, hash_kind) in [
        (
            Identity::plain("tenant-a"),
            Identity::hashed(17),
            Phase::Decode,
            HashKind::Text,
        ),
        (
            Identity::hashed(28),
            Identity::plain("session-a"),
            Phase::Single,
            HashKind::Token,
        ),
        (
            Identity::NONE,
            Identity::NONE,
            Phase::Prefill,
            HashKind::Token,
        ),
        (
            Identity::NONE,
            Identity::NONE,
            Phase::Encode,
            HashKind::Text,
        ),
    ] {
        asked.facts.tenant = tenant;
        asked.facts.session_key = session;
        // The request's own id handed the first turn's key, and no
        // other turn's.
        asked.facts.session_key_from_rid = session == Identity::hashed(17);
        asked.facts.phase = phase;
        asked.facts.hash_kind = hash_kind;
        let want = &asked.facts;
        let (sent, left_out) = request(&asked, None);
        assert_eq!(left_out, 0);
        let facts = sent.facts.as_ref().expect("request facts");
        assert_eq!(facts.client_key, want.client_key);
        assert_eq!(facts.session_key_from_rid, want.session_key_from_rid);
        if tenant.is_none() {
            assert!(facts.tenant.is_none());
        }
        if session.is_none() {
            assert!(facts.session_key.is_none());
        }
        if session.plain.is_none() && !session.is_none() {
            let key = facts.session_key.as_ref();
            assert_eq!(
                key.and_then(|key| key.form.as_ref()),
                Some(&Form::Hash(want.session().get()))
            );
        }
        let checked = Checked::new(sent).expect("the controller reads the request");
        assert_eq!(checked.id(), asked.id);
        assert_eq!(checked.budget().as_micros(), u128::from(want.budget_micros));
        assert!(!checked.has_text());
        assert!(!checked.again());
        assert_eq!(checked.wires(), Some(asked.constraints.wire_modes));
        assert_eq!(checked.block_hashes(), want.block_hashes);
        assert_eq!(checked.subset(), asked.constraints.subset_hint);
        made(
            &checked,
            &mut Lists::default(),
            LEFT,
            |facts, constraints| {
                assert_eq!(facts.model, want.model);
                assert_eq!(facts.namespace_seed, want.namespace_seed);
                assert_eq!(facts.tenant, want.tenant);
                assert_eq!(facts.priority, want.priority);
                assert_eq!(facts.streaming, want.streaming);
                assert_eq!(facts.max_output_tokens, want.max_output_tokens);
                assert_eq!(facts.prompt_tokens, want.prompt_tokens);
                assert_eq!(facts.block_hashes, want.block_hashes);
                assert_eq!(facts.hash_kind, want.hash_kind);
                assert_eq!(facts.session_key, want.session_key);
                assert_eq!(facts.phase, want.phase);
                assert_eq!(facts.attempt, want.attempt);
                assert_eq!(facts.tried, want.tried);
                assert_eq!(u128::from(facts.budget_micros), LEFT.as_micros());
                assert_eq!(facts.headers, want.headers);
                assert_eq!(facts.media, want.media);
                assert_eq!(facts.seed, want.seed);
                assert_eq!(facts.block_size, want.block_size);
                assert_eq!(facts.client_key, want.client_key);
                assert_eq!(facts.session_key_from_rid, want.session_key_from_rid);
                assert_eq!(constraints.excluded, asked.constraints.excluded);
                assert_eq!(constraints.wire_modes, crate::engine::WireModes::NONE);
                assert_eq!(
                    constraints.overload_lifted,
                    asked.constraints.overload_lifted
                );
                assert_eq!(constraints.subset_hint, asked.constraints.subset_hint);
            },
        );
    }
}

/// A frame states one count. A request of text alone states its text's
/// length and no count of tokens: it is sent as its bytes over four,
/// the count a frame stated before a request told its sizes apart. A
/// text of two-byte characters is sent by its bytes, and so is a text
/// whose characters nothing counted. A stated count of tokens is sent
/// as stated, whatever length stands beside it.
#[test]
fn a_frame_states_the_length_of_a_request_of_text_as_its_count() {
    let sent = |prompt_tokens, (prompt_chars, prompt_bytes): (u32, u32)| {
        let asked = Request {
            id: b"a request",
            facts: RequestFacts {
                model: "model-a",
                prompt_tokens,
                prompt_chars,
                prompt_bytes,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let facts = request(&asked, None).0.facts;
        let facts = facts.expect("the frame's facts");
        facts.prompt_tokens
    };
    // 52 one-byte characters, 26 two-byte characters, a text of 52
    // bytes under a document that counts no characters, a count of
    // tokens beside a text, and nothing.
    let texts = [sent(0, (52, 52)), sent(0, (26, 52)), sent(0, (0, 52))];
    assert_eq!(texts, [13, 13, 13]);
    assert_eq!((sent(13, (400, 400)), sent(0, (0, 0))), (13, 0));
}

/// A list longer than a frame holds is copied no further than a frame's
/// worth, and the rest is counted for the link's cut.
#[test]
fn a_request_copies_no_more_hashes_than_a_frame_holds() {
    let blocks = vec![7; MAX_HASHES + 1_000];
    let asked = Request {
        id: b"a request",
        facts: RequestFacts {
            model: "model-a",
            block_hashes: &blocks,
            budget_micros: 2_000,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let (sent, left_out) = request(&asked, None);
    let copied = sent.facts.expect("request facts").block_hashes;
    assert_eq!((copied.len(), copied.capacity()), (MAX_HASHES, MAX_HASHES));
    assert_eq!(left_out, 1_000);
}

/// An empty text is a key at the door in process, and the controller
/// reads an empty text as no key. The empty key crosses as its hash, so
/// the controller decides over the same tenant and session.
#[test]
fn an_empty_plain_key_reads_over_the_link_as_the_key_it_is() {
    let asked = Request {
        id: b"a request",
        facts: RequestFacts {
            model: "model-a",
            tenant: Identity::plain(""),
            session_key: Identity::plain(""),
            budget_micros: 2_000,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let want = &asked.facts;
    assert!(!want.tenant.is_none() && !want.session_key.is_none());
    let (sent, _) = request(&asked, None);
    let checked = Checked::new(sent).expect("the controller reads the request");
    made(&checked, &mut Lists::default(), LEFT, |facts, _| {
        assert_eq!(facts.tenant.hash, want.tenant.hash);
        assert_eq!(facts.session().get(), want.session().get());
    });
}

/// A request at the prefill phase, as the engine's entry for an encode
/// request takes it, under `session`.
fn at_prefill(session: Identity<'_>) -> Request<'_> {
    Request {
        id: b"a request",
        facts: RequestFacts {
            model: "model-a",
            session_key: session,
            phase: Phase::Prefill,
            budget_micros: 2_000,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// An encode request's frame states the encode phase and the items,
/// and the controller reads it at the prefill phase with the items its
/// host stated, in their order: a plain key, an empty text as the key
/// it is, a key that is none at its place, and a fingerprint. The
/// items' one key crosses as stated too: one that is none is stated
/// with no form and read as a one key, and a host that states none is
/// read as stating none. A request that states no item at all is an
/// encode request all the same.
#[test]
fn the_controller_reads_an_encode_requests_items_as_the_host_stated_them() {
    let keys = [
        Identity::plain("image-a"),
        Identity::plain(""),
        Identity::NONE,
        Identity::hashed(31),
    ];
    let items = EncodeItems {
        keys: &keys,
        one: Some(Identity::plain("digest")),
        references: true,
        decode_images: true,
        ..EncodeItems::default()
    };
    let asked = at_prefill(Identity::NONE);
    let (sent, left_out) = encode_request(&asked, items);
    assert_eq!(left_out, 0);
    let stated = sent.facts.as_ref().map(|facts| facts.phase);
    assert_eq!(stated, Some(proto::Phase::Encode.into()));
    let checked = Checked::new(sent).expect("the controller reads the request");
    assert!(checked.carries_items());
    assert_eq!(checked.phase(), Phase::Prefill);
    let read = checked.item_keys(Vec::new());
    let hashes = |keys: &[Identity<'_>]| keys.iter().map(|key| key.hash).collect::<Vec<_>>();
    assert_eq!(hashes(&read), hashes(&keys));
    assert!(read[2].is_none() && !read[1].is_none());
    let read = checked.items(&read);
    assert_eq!(read.one.map(|one| one.hash), items.one.map(|one| one.hash));
    assert!(read.references && read.decode_images);

    let formless = EncodeItems {
        one: Some(Identity::NONE),
        ..items
    };
    let (sent, _) = encode_request(&asked, formless);
    let one = sent.encode.as_ref().and_then(|stated| stated.one.as_ref());
    assert_eq!(one.map(|one| one.form.is_none()), Some(true));
    let checked = Checked::new(sent).expect("the controller reads the request");
    let read = checked.item_keys(Vec::new());
    assert!(checked.items(&read).one.is_some_and(|one| one.is_none()));
    let (sent, _) = encode_request(&asked, EncodeItems { one: None, ..items });
    assert!(sent
        .encode
        .as_ref()
        .is_some_and(|stated| stated.one.is_none()));
    let checked = Checked::new(sent).expect("the controller reads the request");
    let read = checked.item_keys(Vec::new());
    assert!(checked.items(&read).one.is_none());

    let (sent, _) = encode_request(&asked, EncodeItems::default());
    let checked = Checked::new(sent).expect("the controller reads the request");
    assert!(checked.carries_items() && checked.item_keys(Vec::new()).is_empty());
    let read = checked.items(&[]);
    assert!(read.one.is_none() && !read.references && !read.decode_images);
}

/// The frame of an encode request states the encode phase, and the
/// controller decides it at the prefill phase. A session key in its
/// hash form crosses as the session's value under the phase the frame
/// states, so the controller reads back the key its host holds: the
/// encode frame and the plain pair's frame of one request name one
/// session to the engine, under each form of the key.
#[test]
fn an_encode_frame_and_a_pairs_frame_of_one_key_name_one_session() {
    for session in [
        Identity::hashed(17),
        Identity::plain("session-a"),
        Identity::plain(""),
    ] {
        let asked = at_prefill(session);
        let want = asked.facts.session();
        assert!(!want.is_none());
        let read = |sent: proto::DecideRequest| {
            let checked = Checked::new(sent).expect("the controller reads the request");
            assert_eq!(checked.phase(), Phase::Prefill);
            made(&checked, &mut Lists::default(), LEFT, |facts, _| {
                (facts.session(), facts.session_key.hash)
            })
        };
        let (encode, _) = encode_request(&asked, EncodeItems::default());
        let stated = encode.facts.as_ref().and_then(|facts| {
            let key = facts.session_key.as_ref();
            key.and_then(|key| key.form.clone())
        });
        let pair = read(request(&asked, None).0);
        assert_eq!(pair, (want, session.hash));
        assert_eq!(read(encode), pair, "{session:?}");
        // The hash form on the encode frame is the session's value
        // under the encode phase, which is not the pair's session.
        let at_encode = RequestFacts {
            phase: Phase::Encode,
            ..asked.facts
        };
        assert_ne!(at_encode.session(), want);
        if session.plain.is_none_or(str::is_empty) {
            assert_eq!(stated, Some(Form::Hash(at_encode.session().get())));
        }
    }
}

/// A request that names its worker states the number its target source
/// answered for it, and the controller reads that number off the frame.
/// A text that is no number crosses as the widest number, which reaches
/// no worker of any list. A request that names no worker states no
/// number, as the frame of a host older than the field.
#[test]
fn a_request_that_names_its_worker_states_its_number_on_the_frame() {
    let asked = Request {
        id: b"a request",
        facts: RequestFacts {
            model: "model-a",
            budget_micros: 2_000,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let read = |named: Option<Option<u32>>| {
        let (sent, _) = request(&asked, named);
        let stated = sent.named_worker;
        let checked = Checked::new(sent).expect("the controller reads the request");
        (stated, checked.named_worker())
    };
    assert_eq!(read(Some(Some(0))), (Some(0), Some(0)));
    assert_eq!(read(Some(Some(7))), (Some(7), Some(7)));
    let widest = Some(u32::MAX);
    assert_eq!(read(Some(None)), (widest, widest), "no number");
    assert_eq!(read(None), (None, None), "names no worker");
}

/// A request that states a hard demand states each part on its frame
/// (`DecideRequest.demands`), and the controller makes its request with
/// the same demand. A request that demands nothing states no message,
/// as the frame of a host older than the field.
#[test]
fn a_requests_demands_cross_the_wire_part_by_part() {
    let read = |demands: Demands| {
        let asked = Request {
            id: b"a request",
            facts: RequestFacts {
                model: "model-a",
                budget_micros: 2_000,
                ..RequestFacts::default()
            },
            constraints: Constraints {
                demands,
                ..Constraints::default()
            },
        };
        let (sent, _) = request(&asked, None);
        let stated = sent.demands;
        let checked = Checked::new(sent).expect("the controller reads the request");
        let made = made(&checked, &mut Lists::default(), LEFT, |_, constraints| {
            constraints.demands
        });
        (stated, made)
    };
    assert_eq!(read(Demands::default()), (None, Demands::default()));
    let whole = Demands {
        runtime: 7,
        vision: true,
        references: References::Caller,
        input_window: 4_096,
        output_window: 512,
        decoding: 255,
    };
    let stated = proto::Demands {
        runtime: 7,
        vision: true,
        references: proto::demands::References::Caller.into(),
        input_window: 4_096,
        output_window: 512,
        decoding: 255,
    };
    assert_eq!(read(whole), (Some(stated), whole));
    // One part alone is a demand: it is stated, and comes back alone.
    let none = Demands::default();
    let parts = [
        Demands { runtime: 1, ..none },
        Demands {
            vision: true,
            ..none
        },
        Demands {
            references: References::Any,
            ..none
        },
        Demands {
            references: References::Worker,
            ..none
        },
        Demands {
            input_window: 1,
            ..none
        },
        Demands {
            output_window: 1,
            ..none
        },
        Demands {
            decoding: 1,
            ..none
        },
    ];
    for part in parts {
        let (stated, made) = read(part);
        assert!(stated.is_some(), "{part:?}");
        assert_eq!(made, part);
    }
}

fn trace(survivors: Survivors) -> Trace {
    Trace {
        profile: 7,
        survivors,
        top: [
            (WorkerId(58), 0.8),
            (WorkerId(42), 0.3),
            (WorkerId(17), 0.1),
        ],
        top_len: 3,
        seed: 27,
        generation: 13,
        lifted: LiftedVeto::Overload,
    }
}

fn pick(count: usize, paired: bool, survivors: Survivors) -> Decision {
    let mut targets = Targets::new();
    for (id, rank) in [(58, 7), (42, 3), (17, 1), (20, 5)].into_iter().take(count) {
        targets.push(Target {
            id: WorkerId(id),
            dp_rank: rank,
            pair: paired.then_some(Pair {
                decode_id: WorkerId(id + 100),
                decode_dp_rank: rank + 10,
                ..Pair::default()
            }),
            ..Target::default()
        });
    }
    Decision::Pick(Pick {
        targets,
        trace: trace(survivors),
        ..Pick::default()
    })
}

fn told(decision: Decision, lease: Option<Leased>) -> proto::Decision {
    let answer = Decided::Decision {
        id: vec![7],
        term: 3,
        decision,
        lease,
        encode: None,
    }
    .frame();
    let Some(Answer::Decision(told)) = answer.answer else {
        panic!("expected a decision, got {answer:?}");
    };
    told
}

#[test]
fn picks_keep_their_target_order_and_book_the_target_the_lease_names() {
    let now = Instant::now();
    for (count, paired, index, booked, survivors) in [
        (1, false, 0, 58, Survivors::NOT_COUNTED),
        (decision::MAX_TARGETS, false, 2, 17, Survivors::counted(4)),
        (2, true, 1, 42, Survivors::counted(2)),
    ] {
        let decision = pick(count, paired, survivors);
        let lease = Lease {
            id: LeaseId(17),
            side: if paired { Side::Prefill } else { Side::Single },
            target_index: index,
            booked: WorkerId(booked),
            generation: 13,
            session: RequestFacts {
                model: "model-a",
                session_key: Identity::hashed(17),
                ..RequestFacts::default()
            }
            .session(),
            tokens: 165,
            ..Lease::default()
        };
        let told = told(
            decision,
            Some(Leased {
                lease,
                beside: if paired {
                    Beside::Decode(lease.session)
                } else {
                    Beside::Alone
                },
                left: Duration::from_millis(17),
                due: now + Duration::from_secs(1),
            }),
        );
        assert_eq!(told.leases.len(), if paired { 2 } else { 1 });
        let read = decided(&told, now).expect("the wire's pick fits the record");
        assert_eq!(read.decision, decision);
        let leased = read.lease.expect("a lease was told");
        assert_eq!(leased.lease, lease);
        assert_eq!(leased.due, now + Duration::from_millis(17));
        if paired {
            let targets = &read.decision.pick().expect("a pair pick").targets;
            let decode = super::leased(&told.leases[1], Side::Decode, targets, now)
                .expect("the decode lease");
            assert_eq!(decode.lease.booked, WorkerId(booked + 100));
        }
    }
}

#[test]
fn a_pick_keeps_the_lifted_liveness_veto() {
    let Decision::Pick(mut pick) = pick(1, false, Survivors::counted(1)) else {
        panic!("a pick");
    };
    pick.trace.lifted = LiftedVeto::Liveness;
    let decision = Decision::Pick(pick);
    let read = decided(&told(decision, None), Instant::now()).expect("a pick");
    assert_eq!(read.decision, decision);
    assert!(read.lease.is_none());
}

#[test]
fn a_shed_keeps_its_reason_and_pacing_and_a_queue_keeps_its_ticket() {
    let now = Instant::now();
    let shed = Shed {
        reason: ShedReason::Saturated,
        emptied_by: "not-overloaded",
        retry_after_secs: 3,
        wait_allowed: true,
        side: None,
        trace: trace(Survivors::counted(0)),
    };
    for reason in [
        ShedReason::Saturated,
        ShedReason::FlowLimit,
        ShedReason::NoEndpoints,
        ShedReason::PolicyDeclined,
        ShedReason::Deadline,
    ] {
        let shed = Shed { reason, ..shed };
        let read = decided(&told(Decision::Shed(shed), None), now).expect("a shed");
        assert_eq!(
            read.decision,
            Decision::Shed(Shed {
                emptied_by: "controller",
                ..shed
            })
        );
        assert!(read.lease.is_none());
    }
    let queue = Decision::Queue(Queue {
        ticket: 75,
        max_wait_micros: 15_000,
        trace: Trace {
            lifted: LiftedVeto::None,
            ..shed.trace
        },
    });
    let read = decided(&told(queue, None), now).expect("a queue");
    assert_eq!(read.decision, queue);
    assert!(read.lease.is_none());
}

/// A shed keeps the stage a demand sheds its pool under, each by its name:
/// a host answers by it, as it does in process. Any other stage is the
/// controller's document's own and reads `controller`, and so does a text
/// that only looks like a demand's stage.
#[test]
fn a_shed_keeps_a_demands_stage_and_reads_any_other_as_the_controllers() {
    use crate::engine::demands::UnmetDemand;

    let now = Instant::now();
    let shed = Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "",
        retry_after_secs: 0,
        wait_allowed: false,
        side: None,
        trace: trace(Survivors::counted(0)),
    };
    let read = |emptied_by| {
        let wire = told(Decision::Shed(Shed { emptied_by, ..shed }), None);
        match decided(&wire, now).expect("a shed").decision {
            Decision::Shed(shed) => shed.emptied_by,
            other => panic!("a shed, got {other:?}"),
        }
    };
    for demand in [
        UnmetDemand::Runtime,
        UnmetDemand::Vision,
        UnmetDemand::References,
        UnmetDemand::Pipeline,
        UnmetDemand::Window,
        UnmetDemand::Decoding,
    ] {
        assert_eq!(read(demand.stage()), demand.stage(), "{demand:?}");
    }
    assert_eq!(read("context-window"), "context-window");
    for other in [
        "not-overloaded",
        "target-worker",
        "none",
        "",
        "Context-Window",
        "context-window ",
    ] {
        assert_eq!(read(other), "controller", "{other:?}");
    }
}

/// A shed made ahead of every profile goes out with the mark over a
/// zero position. Its record names no profile, and a decision under the
/// first profile still names that one.
#[test]
fn a_shed_made_ahead_of_every_profile_names_none_over_a_link() {
    let now = Instant::now();
    let ahead = decision::shed(ShedReason::NoEndpoints, "wires", 0, 6);
    let wire = told(ahead, None);
    let sent = wire.trace.as_ref().expect("a trace");
    assert!(sent.no_profile);
    assert_eq!(sent.profile, 0);
    let read = decided(&wire, now).expect("a shed");
    assert_eq!(read.decision.trace().made_under(), None);

    let first = Decision::Shed(Shed {
        reason: ShedReason::Saturated,
        emptied_by: "not-overloaded",
        retry_after_secs: 3,
        wait_allowed: true,
        side: None,
        trace: Trace {
            profile: 0,
            ..trace(Survivors::counted(0))
        },
    });
    let read = decided(&told(first, None), now).expect("a shed");
    assert_eq!(read.decision.trace().made_under(), Some(0));
}

#[test]
fn a_lease_keeps_the_kind_it_carries() {
    use crate::tracking::lease::LeaseKind;

    let now = Instant::now();
    let mut told = told(pick(1, false, Survivors::counted(1)), None);
    told.leases = vec![proto::Lease {
        id: 73,
        side: proto::Side::Single.into(),
        kind: proto::LeaseKind::Reservation.into(),
        ..proto::Lease::default()
    }];
    let leased = decided(&told, now).expect("a pick").lease.expect("a lease");
    assert_eq!(leased.lease.kind, LeaseKind::Reservation);
}

#[test]
fn a_pair_requires_a_prefill_and_decode_under_one_id_and_span() {
    let now = Instant::now();
    let good = told(
        pick(2, true, Survivors::counted(2)),
        Some(Leased {
            lease: Lease {
                id: LeaseId(17),
                side: Side::Prefill,
                ..Lease::default()
            },
            beside: Beside::Decode(view::Session::NONE),
            left: Duration::from_secs(1),
            due: now + Duration::from_secs(1),
        }),
    );
    let checks: [(_, fn(&mut proto::Decision)); 7] = [
        ("leases", |bad| {
            bad.leases.pop();
        }),
        ("leases", |bad| bad.leases.push(bad.leases[0])),
        ("leases.side", |bad| {
            bad.leases[0].side = proto::Side::Encode.into();
        }),
        ("leases.side", |bad| {
            bad.leases[1].side = proto::Side::Single.into();
        }),
        ("leases.id", |bad| bad.leases[1].id += 1),
        ("leases.dispatch_ttl_ms", |bad| {
            bad.leases[1].dispatch_ttl_ms += 1;
        }),
        ("pick.targets.pair", |bad| {
            let Some(Outcome::Pick(pick)) = &mut bad.outcome else {
                panic!("a pick");
            };
            pick.targets[1].pair = None;
        }),
    ];
    for (path, edit) in checks {
        let mut bad = good.clone();
        edit(&mut bad);
        let error = decided(&bad, now).expect_err("an unreadable pair");
        assert_eq!(error.path, path);
    }
}

#[test]
fn a_dispatch_deadline_outside_the_hosts_clock_is_unreadable() {
    let now = Instant::now();
    let mut last = now;
    for bit in (0..64).rev() {
        if let Some(further) = last.checked_add(Duration::from_secs(1 << bit)) {
            last = further;
        }
    }
    assert!(last.checked_add(Duration::from_secs(1)).is_none());
    let told = told(
        pick(1, false, Survivors::counted(1)),
        Some(Leased {
            lease: Lease {
                id: LeaseId(17),
                ..Lease::default()
            },
            beside: Beside::Alone,
            left: Duration::from_secs(1),
            due: now + Duration::from_secs(1),
        }),
    );
    let error = decided(&told, last).expect_err("the span runs past the clock's range");
    assert_eq!(error.path, "leases.dispatch_ttl_ms");
}

/// An encoder as an answer names it.
fn encoder(id: u64, dp_rank: u32) -> proto::Target {
    proto::Target {
        id,
        dp_rank,
        ..proto::Target::default()
    }
}

/// The encoders an answer names are read onto the host's room in the
/// keys' order, each by its worker and its rank, at position 0. The
/// one key's encoder is read alone, and the room says so, also where
/// the request's one key has no form. A decision
/// that picks nothing places nothing, whatever stands beside it, and a
/// pick of a request with no key wants no encoder. Each read empties
/// what the room held.
#[test]
fn an_answers_encoders_are_read_in_the_keys_order() {
    let keys = [Identity::hashed(1), Identity::NONE, Identity::hashed(3)];
    let items = EncodeItems {
        keys: &keys,
        one: Some(Identity::hashed(5)),
        ..EncodeItems::default()
    };
    let picked = pick(1, true, Survivors::counted(2));
    let named = |id, dp_rank| Target {
        id: WorkerId(id),
        dp_rank,
        ..Target::default()
    };
    let mut room = EncodePlaced::new();
    let each = proto::EncodePlaced {
        encoders: vec![encoder(9, 2), encoder(4, 0), encoder(9, 1)],
        one: false,
    };
    placed(Some(&each), &picked, &items, &mut room).expect("an encoder a key");
    assert_eq!(room.encoders(), [named(9, 2), named(4, 0), named(9, 1)]);
    assert!(!room.one());

    let one = proto::EncodePlaced {
        encoders: vec![encoder(4, 6)],
        one: true,
    };
    placed(Some(&one), &picked, &items, &mut room).expect("the one key's encoder");
    assert_eq!(room.encoders(), [named(4, 6)]);
    assert!(room.one());
    let formless = EncodeItems {
        one: Some(Identity::NONE),
        ..items
    };
    placed(Some(&each), &picked, &formless, &mut room).expect("an encoder a key");
    placed(Some(&one), &picked, &formless, &mut room).expect("the one key's encoder");
    assert_eq!(room.encoders(), [named(4, 6)]);
    assert!(room.one());

    let shed = Decision::Shed(Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "controller",
        retry_after_secs: 0,
        wait_allowed: false,
        side: None,
        trace: trace(Survivors::counted(0)),
    });
    placed(Some(&one), &shed, &items, &mut room).expect("a shed places nothing");
    assert!(room.encoders().is_empty() && !room.one());

    placed(Some(&one), &picked, &items, &mut room).expect("the one key's encoder");
    let keyless = EncodeItems::default();
    placed(None, &picked, &keyless, &mut room).expect("no key, no encoder");
    assert!(room.encoders().is_empty() && !room.one());
}

/// By the contract the encoders are one where the answer says so, else
/// one a key, and an encoder names no pair. An answer that says
/// otherwise is named by its field and leaves the room empty, also
/// where the room held an earlier answer's encoders. So are a pick
/// that names no encoder for its keys, and an answer that says the one
/// key was placed where the request stated none, or stated no item for
/// it to stand for: no engine writes that.
#[test]
fn an_answer_whose_encoders_do_not_fit_its_request_is_unreadable() {
    let keys = [Identity::hashed(1), Identity::hashed(2)];
    let items = EncodeItems {
        keys: &keys,
        one: Some(Identity::hashed(5)),
        ..EncodeItems::default()
    };
    let each_alone = EncodeItems { one: None, ..items };
    let picked = pick(1, true, Survivors::counted(2));
    let told = |ids: &[u64], one| proto::EncodePlaced {
        encoders: ids.iter().map(|&id| encoder(id, 0)).collect(),
        one,
    };
    let paired = proto::EncodePlaced {
        encoders: vec![
            encoder(9, 0),
            proto::Target {
                pair: Some(proto::Pair::default()),
                ..encoder(4, 0)
            },
        ],
        one: false,
    };
    let ranked = proto::EncodePlaced {
        encoders: vec![encoder(9, 0), encoder(4, 65_536)],
        one: false,
    };
    let cases = [
        (None, items, "encode"),
        (Some(told(&[9], false)), items, "encode.encoders"),
        (Some(told(&[9, 4, 7], false)), items, "encode.encoders"),
        (Some(told(&[], false)), items, "encode.encoders"),
        (Some(told(&[9, 4], true)), items, "encode.encoders"),
        (Some(told(&[9], true)), each_alone, "encode.one"),
        (Some(paired), items, "encode.encoders.pair"),
        (Some(ranked), items, "encode.encoders.dp_rank"),
    ];
    let mut room = EncodePlaced::new();
    for (told, items, path) in cases {
        let whole = proto::EncodePlaced {
            encoders: vec![encoder(9, 0), encoder(4, 0)],
            one: false,
        };
        placed(Some(&whole), &picked, &items, &mut room).expect("an encoder a key");
        let error = placed(told.as_ref(), &picked, &items, &mut room)
            .expect_err("the answer does not fit its request");
        assert_eq!(error.path, path, "{told:?}");
        assert!(!error.why.is_empty());
        assert!(room.encoders().is_empty() && !room.one(), "{path}");
    }
    let keyless = EncodeItems { keys: &[], ..items };
    placed(Some(&told(&[9, 4], false)), &picked, &items, &mut room).expect("an encoder a key");
    let error = placed(Some(&told(&[9], true)), &picked, &keyless, &mut room)
        .expect_err("the one key stands for no item");
    assert_eq!(error.path, "encode.one");
    assert!(room.encoders().is_empty() && !room.one());
}

#[test]
fn unreadable_fields_are_named_by_path() {
    let now = Instant::now();
    let good = told(
        pick(2, false, Survivors::counted(2)),
        Some(Leased {
            lease: Lease {
                id: LeaseId(17),
                ..Lease::default()
            },
            beside: Beside::Alone,
            left: Duration::from_secs(1),
            due: now + Duration::from_secs(1),
        }),
    );
    let refuses = |bad: proto::Decision, path| {
        let error = decided(&bad, now).expect_err("the field does not fit the record");
        assert_eq!(error.path, path);
        assert!(!error.why.is_empty());
    };
    let mut bad = good.clone();
    bad.leases[0].target_index = 256;
    refuses(bad, "leases.target_index");
    let mut bad = good.clone();
    bad.leases[0].target_index = 2;
    refuses(bad, "leases.target_index");
    let mut bad = good.clone();
    bad.leases[0].kind = 0;
    refuses(bad, "leases.kind");
    let mut bad = good.clone();
    bad.leases[0].side = 0;
    refuses(bad, "leases.side");
    let mut bad = good.clone();
    bad.outcome = None;
    refuses(bad, "outcome");
    let mut bad = good.clone();
    bad.outcome = Some(Outcome::Queue(proto::Queue::default()));
    refuses(bad, "leases");

    let checks: [(_, fn(&mut proto::Pick)); 4] = [
        ("pick.targets", |pick| {
            pick.targets.resize(5, proto::Target::default());
        }),
        ("pick.targets", |pick| pick.targets.clear()),
        ("pick.targets.dp_rank", |pick| {
            pick.targets[0].dp_rank = 65_536;
        }),
        ("pick.targets.pair.decode_dp_rank", |pick| {
            pick.targets[0].pair = Some(proto::Pair {
                decode_dp_rank: 65_536,
                ..proto::Pair::default()
            });
        }),
    ];
    for (path, edit) in checks {
        let mut bad = good.clone();
        if let Some(Outcome::Pick(pick)) = bad.outcome.as_mut() {
            edit(pick);
        }
        refuses(bad, path);
    }
    let checks: [(_, fn(&mut proto::Trace)); 6] = [
        ("trace.profile", |trace| trace.profile = 65_536),
        // The mark's own number is no position, and the mark stands
        // over no position but zero.
        ("trace.profile", |trace| trace.profile = 65_535),
        ("trace.profile", |trace| {
            trace.no_profile = true;
            trace.profile = 1;
        }),
        ("trace.top", |trace| {
            trace.top.resize(4, proto::Scored::default());
        }),
        ("trace.survivors", |trace| trace.survivors = Some(u32::MAX)),
        ("trace.lifted", |trace| trace.lifted = 100),
    ];
    for (path, edit) in checks {
        let mut bad = good.clone();
        edit(bad.trace.as_mut().expect("a trace"));
        refuses(bad, path);
    }
    refuses(
        proto::Decision {
            outcome: Some(Outcome::Shed(proto::Shed::default())),
            ..proto::Decision::default()
        },
        "shed.reason",
    );
    refuses(
        proto::Decision {
            outcome: Some(Outcome::Shed(proto::Shed {
                reason: proto::ShedReason::Saturated.into(),
                retry_after_secs: 65_536,
                ..proto::Shed::default()
            })),
            ..proto::Decision::default()
        },
        "shed.retry_after_secs",
    );
}

#[test]
fn reports_preserve_each_side_class_and_release_reason() {
    use proto::{completed::Class, released::Reason};

    let lease = LeaseId(17);
    for (side, wire) in [
        (Side::Single, proto::Side::Single),
        (Side::Prefill, proto::Side::Prefill),
        (Side::Decode, proto::Side::Decode),
        (Side::Encode, proto::Side::Encode),
    ] {
        let dispatch = Said::Dispatched(
            lease::Dispatch {
                target_index: 2,
                side,
                since_decision_micros: 250,
            },
            Instant::now(),
        );
        let report = frame(lease, Side::Single, dispatch);
        assert_eq!(
            report,
            Frame::Dispatched(proto::Dispatched {
                lease_id: 17,
                target_index: 2,
                side: wire.into(),
                since_decision_micros: 250
            })
        );
    }
    for (class, wire) in [
        (TransportClass::Ok, Class::Ok),
        (TransportClass::ClientError, Class::ClientError),
        (TransportClass::ServerError, Class::ServerError),
        (TransportClass::ConnectFailed, Class::ConnectFailed),
        (TransportClass::Timeout, Class::Timeout),
        (TransportClass::Cancelled, Class::Cancelled),
    ] {
        let outcome = lease::Outcome {
            side: Side::Decode,
            served: WorkerId(42),
            status: 503,
            class,
            ttft_micros: 15,
            tokens_in: 8,
            tokens_out: 9,
            duration_micros: 38,
        };
        let report = frame(lease, Side::Prefill, Said::Completed(outcome));
        assert_eq!(
            report,
            Frame::Completed(proto::Completed {
                lease_id: 17,
                side: proto::Side::Decode.into(),
                served: 42,
                status: 503,
                class: wire.into(),
                ttft_micros: 15,
                tokens_in: 8,
                tokens_out: 9,
                duration_micros: 38
            })
        );
    }
    for (reason, wire) in [
        (ReleaseReason::Abandoned, Reason::Abandoned),
        (ReleaseReason::LateDecision, Reason::LateDecision),
        (ReleaseReason::ClientGone, Reason::ClientGone),
        (ReleaseReason::CommitFailed, Reason::CommitFailed),
        (ReleaseReason::Expired, Reason::Expired),
        (ReleaseReason::WorkerRemoved, Reason::WorkerRemoved),
    ] {
        for side in [Some(Side::Prefill), Some(Side::Decode), None] {
            let report = frame(lease, Side::Prefill, Said::Released(side, reason));
            let side = side.map_or(proto::Side::Unspecified, proto::Side::from);
            assert_eq!(
                report,
                Frame::Released(proto::Released {
                    what: Some(proto::released::What::LeaseId(17)),
                    reason: wire.into(),
                    side: side.into()
                })
            );
        }
    }
}
