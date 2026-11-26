//! The types that grow, written as a crate outside this one writes them:
//! each enum the crate marks non-exhaustive is matched with every case it
//! has today and an arm for the rest, and the marked records are made by
//! their calls or started from their defaults.
//!
//! A test target is a program outside the library. The arm for the rest is
//! what a mark asks of such a program, and the arm holds the mark in turn:
//! with a mark taken off an enum, its arm can match nothing, and this file
//! refuses a pattern that can match nothing. A case that makes a marked
//! record by its call holds the call and no mark: a record's mark is held
//! where the record is declared, by the crate's lint, by an example in its
//! own doc where a literal of it must fail to build, or by review alone
//! where its module still expects the lint.

#![deny(unreachable_patterns)]

#[cfg(feature = "in-process")]
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "in-process")]
use smg_grand_central::engine::{TextKey, TokenKey};
use smg_grand_central::{
    document::Needs,
    engine::{
        DrawUnit, EncodeMember, Health, LoadRow, Phase, Plain, PoolWord, Profile, ScoreFront,
        ScoreSpread, Shortlist, WeightedScorer, WireModes, WorkerId,
    },
    load::SignalCell,
    policies::prefix_hash::HeadRule,
    stages::{Best, LeastRequests},
    store::{pairing::PairingDescriptor, EncodeHandoff, EncodeOffer, Position},
    tracking::{EventKind, Membership, Outcome, ReleaseReason, Side, Signal, TransportClass},
};

/// What a match outside the crate answers for a case the crate adds later.
const ANOTHER: &str = "another";

/// Each of the five events a stage hears is matched with the cases it has
/// today and an arm for the rest, and a case with named fields with `..`.
/// The cases with named fields are made by their calls.
#[test]
fn a_match_on_an_event_names_todays_cases_and_keeps_an_arm_for_the_rest() {
    let class = |class: TransportClass| match class {
        TransportClass::Ok => "ok",
        TransportClass::ClientError => "client-error",
        TransportClass::ServerError => "server-error",
        TransportClass::ConnectFailed => "connect-failed",
        TransportClass::Timeout => "timeout",
        TransportClass::Cancelled => "cancelled",
        _ => ANOTHER,
    };
    assert_eq!(class(TransportClass::Timeout), "timeout");

    let reason = |reason: ReleaseReason| match reason {
        ReleaseReason::Abandoned => "abandoned",
        ReleaseReason::LateDecision => "late-decision",
        ReleaseReason::ClientGone => "client-gone",
        ReleaseReason::CommitFailed => "commit-failed",
        ReleaseReason::Expired => "expired",
        ReleaseReason::WorkerRemoved => "worker-removed",
        _ => ANOTHER,
    };
    assert_eq!(reason(ReleaseReason::Expired), "expired");

    let kind = |kind: EventKind| match kind {
        EventKind::Issued => "issued",
        EventKind::Dispatched => "dispatched",
        EventKind::Completed => "completed",
        EventKind::Released => "released",
        EventKind::Membership => "membership",
        EventKind::Signal => "signal",
        _ => ANOTHER,
    };
    // The list of every kind is a slice, so a further kind changes no type
    // this file names.
    let kinds: &[EventKind] = EventKind::ALL;
    let named = kinds.iter().all(|each| kind(*each) != ANOTHER);
    assert!(named, "a kind this file does not name");

    let change = |change: &Membership<'_>| match change {
        Membership::Added(_) => "added",
        Membership::Removed(_) => "removed",
        Membership::Replaced { .. } => "replaced",
        Membership::Relabelled(_) => "relabelled",
        Membership::Resized(_) => "resized",
        Membership::Snapshot(_) => "snapshot",
        Membership::Left { .. } => "left",
        _ => ANOTHER,
    };
    let models = ["a-model".to_owned()];
    assert_eq!(
        change(&Membership::replaced(WorkerId(1), WorkerId(2))),
        "replaced"
    );
    assert_eq!(change(&Membership::left(WorkerId(2), &models)), "left");
    assert_eq!(change(&Membership::Added(WorkerId(3))), "added");

    let signal = |signal: &Signal<'_>| match signal {
        Signal::Load { .. } => "load",
        Signal::Health { .. } => "health",
        Signal::GatewayVeto { .. } => "gateway-veto",
        _ => ANOTHER,
    };
    let told = [
        Signal::load(WorkerId(1), 0, LoadRow::default(), 0),
        Signal::health(WorkerId(1), Health::default()),
        Signal::gateway_veto(WorkerId(1), 500_000),
    ];
    assert_eq!(
        told.each_ref().map(signal),
        ["load", "health", "gateway-veto"]
    );
}

/// The three answers a stage gives as an enum: a stage names the case it
/// gives, and a match outside the crate keeps an arm for the rest.
#[test]
fn a_match_on_a_stages_answer_keeps_an_arm_for_the_rest() {
    let spread = |spread: ScoreSpread| match spread {
        ScoreSpread::Any => "any",
        ScoreSpread::Levels(_) => "levels",
        ScoreSpread::Alike => "alike",
        _ => ANOTHER,
    };
    assert_eq!(spread(ScoreSpread::Levels(0.25)), "levels");

    let front = |front: ScoreFront| match front {
        ScoreFront::Unsaid => "unsaid",
        ScoreFront::Alike => "alike",
        ScoreFront::At(_) => "at",
        ScoreFront::Whole => "whole",
        _ => ANOTHER,
    };
    assert_eq!(front(ScoreFront::At(3)), "at");

    let unit = |unit: DrawUnit| match unit {
        DrawUnit::Row => "row",
        DrawUnit::Engine => "engine",
        _ => ANOTHER,
    };
    assert_eq!(unit(DrawUnit::default()), "row");
}

/// The two records a stage answers with are made by their calls, and a
/// pattern of each ends with `..`.
#[test]
fn an_answer_record_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let Shortlist {
        compared,
        alternates,
        ..
    } = Shortlist::new(2, 1);
    assert_eq!((compared, alternates), (2, 1));
    assert_eq!(Shortlist::all(5), Shortlist::new(5, 0));

    let boundaries = [64, 128];
    let HeadRule {
        depth,
        boundaries: cut,
        ..
    } = HeadRule::new(256, &boundaries);
    assert_eq!((depth, cut), (256, &boundaries[..]));
}

/// A stage writes its needs as one chain from none, a call a flag. Each
/// call declares its own flag and no other, and the sample's call states
/// its number. None is the record's default.
#[test]
fn needs_are_written_as_one_chain_a_call_a_flag() {
    assert_eq!(Needs::NONE, Needs::default());
    type Call = fn(Needs) -> Needs;
    type Set = fn(&mut Needs);
    let flags: [(Call, Set); 14] = [
        (Needs::with_text, |needs| needs.text = true),
        (Needs::with_token_hashes, |needs| needs.token_hashes = true),
        (Needs::with_text_hashes, |needs| needs.text_hashes = true),
        (Needs::with_prompt_size, |needs| needs.prompt_size = true),
        (Needs::with_loads, |needs| needs.loads = true),
        (Needs::with_events, |needs| needs.events = true),
        (Needs::with_sent_prefixes, |needs| {
            needs.sent_prefixes = true;
        }),
        (Needs::with_ring, |needs| needs.ring = true),
        (Needs::with_headers, |needs| needs.headers = true),
        (Needs::with_media, |needs| needs.media = true),
        (Needs::with_active_sessions, |needs| {
            needs.active_sessions = true;
        }),
        (Needs::with_session_in_flight, |needs| {
            needs.session_in_flight = true;
        }),
        (Needs::with_affinity, |needs| needs.affinity = true),
        (Needs::with_signal_cells, |needs| needs.signal_cells = true),
    ];
    let (mut chained, mut every) = (Needs::NONE, Needs::NONE);
    for (at, (call, set)) in flags.into_iter().enumerate() {
        let mut alone = Needs::NONE;
        set(&mut alone);
        assert_ne!(alone, Needs::NONE, "flag {at}");
        assert_eq!(call(Needs::NONE), alone, "flag {at}: its call");
        chained = call(chained);
        set(&mut every);
    }
    assert_eq!(chained, every);
    let mut sampled = every;
    sampled.sample = 8;
    assert_eq!(every.with_sample(8), sampled);
}

/// A host makes an outcome by the end's status and class, states the side
/// and the worker by one call, and sets what it measured a field each. A
/// pattern of the record ends with `..`.
#[test]
fn an_outcome_is_made_by_its_calls_and_read_by_a_pattern_that_ends_open() {
    let reported = Outcome::reported(503, TransportClass::ServerError);
    assert_eq!(
        (reported.side, reported.served),
        (Side::Single, WorkerId(0))
    );
    let mut outcome = reported.with_served(Side::Decode, WorkerId(7));
    outcome.tokens_out = 12;
    let Outcome {
        side,
        served,
        status,
        class,
        tokens_out,
        ..
    } = outcome;
    assert_eq!(
        (side, served, status, class, tokens_out),
        (
            Side::Decode,
            WorkerId(7),
            503,
            TransportClass::ServerError,
            12
        )
    );
}

/// A plain request's facts are made by their call, which takes every field
/// the record has today, and read by a pattern that ends open.
#[test]
fn a_plain_request_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let Plain {
        phase,
        wires,
        prompt_tokens,
        ..
    } = Plain::of(Phase::Decode, WireModes::GRPC, 7);
    assert_eq!(
        (phase, wires, prompt_tokens),
        (Phase::Decode, WireModes::GRPC, 7)
    );
}

/// A member of the encode derivation is made by its call and read by a
/// pattern that ends open.
#[test]
fn an_encode_member_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let offer = EncodeOffer::default();
    let EncodeMember {
        runtime,
        offer: read,
        ..
    } = EncodeMember::of("a runtime", &offer);
    assert_eq!(runtime, "a runtime");
    assert!(std::ptr::eq(read, &offer));
}

/// A worker's encode offer starts from its default, which states nothing,
/// and takes a field by assignment. It is read by a pattern that ends open.
#[test]
fn an_encode_offer_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut offer = EncodeOffer::default();
    offer.handoff = EncodeHandoff::Push;
    offer.producer = true;
    offer.connector = Some("a connector".to_owned());
    let EncodeOffer {
        handoff,
        producer,
        consumer,
        connector,
        references,
        ..
    } = offer;
    assert_eq!(
        (
            handoff,
            producer,
            consumer,
            connector.as_deref(),
            references
        ),
        (EncodeHandoff::Push, true, false, Some("a connector"), false)
    );
}

/// A worker's pairing descriptor starts from its default, which reports
/// nothing, and takes a field by assignment. It is read by a pattern that
/// ends open.
#[test]
fn a_pairing_descriptor_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut descriptor = PairingDescriptor::default();
    descriptor.transport = Some("a transport".to_owned());
    let PairingDescriptor {
        protocol,
        transport,
        version,
        kv_layout,
        ..
    } = descriptor;
    assert_eq!(
        (protocol, transport.as_deref(), version),
        (None, Some("a transport"), None)
    );
    assert!(kv_layout.iter().all(Option::is_none));
}

/// A place in a store's history starts from its default, the place before
/// every write, and takes a field by assignment. It is read by a pattern
/// that ends open.
#[test]
fn a_position_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut position = Position::default();
    position.seq = 9;
    position.term = 2;
    let Position { seq, term, .. } = position;
    assert_eq!((seq, term), (9, 2));
}

/// A signal's cell is made by its call, which takes both of its fields, and
/// read by a pattern that ends open. The cell of no value is its constant.
#[test]
fn a_signal_cell_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let SignalCell {
        value,
        sampled_at_micros,
        ..
    } = SignalCell::of(2.5, 40);
    assert_eq!(
        (value.to_bits(), sampled_at_micros),
        (2.5_f64.to_bits(), 40)
    );
    assert!(SignalCell::NONE.is_none());
}

/// A pool's word is made by its call, as a program that implements a
/// picker makes one to call it, and read by a pattern that ends open.
#[test]
fn a_pools_word_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let word = AtomicU64::new(3);
    let PoolWord {
        rows, word: read, ..
    } = PoolWord::of(8, &word);
    assert_eq!((rows, read.load(Ordering::Relaxed)), (8, 3));
}

/// A prompt's entry in a host's token tree is made by its call and read by
/// a pattern that ends open.
#[cfg(feature = "in-process")]
#[test]
fn a_token_key_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let TokenKey { tokens, marker, .. } = TokenKey::of(Cow::Borrowed(&[5, 6, 7]), 1);
    assert_eq!((&tokens[..], marker), (&[5, 6, 7][..], 1));
}

/// A prompt's entry in a host's string tree is made by its call and read
/// by a pattern that ends open.
#[cfg(feature = "in-process")]
#[test]
fn a_text_key_is_made_by_its_call_and_read_by_a_pattern_that_ends_open() {
    let TextKey { text, marker, .. } = TextKey::of(Cow::Borrowed("a prompt"), 2);
    assert_eq!((&*text, marker), ("a prompt", 2));
}

/// A weighted scorer is made by its call with the weight it is given, and
/// read by a pattern that ends open.
#[test]
fn a_weighted_scorer_reads_back_the_weight_it_was_started_with() {
    let WeightedScorer { weight, .. } = WeightedScorer::of(Box::new(LeastRequests), 2.5);
    assert_eq!(weight.to_bits(), 2.5_f32.to_bits());
}

/// A profile is made by its call under the name it is given, with no filter
/// and no scorer, and read by a pattern that ends open.
#[test]
fn a_profile_reads_back_the_name_it_was_started_with() {
    let Profile {
        name,
        filters,
        scorers,
        ..
    } = Profile::of("a profile".into(), Box::new(Best));
    assert_eq!(name, "a profile");
    assert!(filters.is_empty() && scorers.is_empty());
}
