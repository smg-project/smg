use prost::Message;

use super::{
    membership,
    proto::{self, decline::Reason as DeclineReason},
    since, since_of, Answer, Invalid, FILTER, PICK, SCORE,
};
use crate::{
    engine::{decision, view::WorkerId},
    tracking::lease::Membership,
};

/// The stages' rules, under a plugin that declared no counter.
fn read(d: proto::PluginDecision, asked: u32, count: u32) -> Result<Answer, Invalid> {
    super::read(d, asked, count, 0)
}

fn decision(
    survivors: Option<Vec<u32>>,
    scores: Vec<f32>,
    pick: Option<u32>,
    decline: Option<(u32, i32)>,
) -> proto::PluginDecision {
    proto::PluginDecision {
        decision_id: 7,
        survivors: survivors.map(|positions| proto::Survivors { positions }),
        scores,
        pick,
        decline: decline.map(|(stage, reason)| proto::Decline {
            stage,
            reason,
            detail: String::new(),
        }),
        counters: Vec::new(),
        tie_band: None,
    }
}

#[test]
fn each_asked_stage_answers_its_own_field_and_an_unanswered_one_is_declined() {
    let a = read(
        decision(Some(vec![2, 0]), vec![0.5, 2.0, -1.0], Some(2), None),
        FILTER | SCORE | PICK,
        3,
    )
    .unwrap();
    assert!(
        a.mask_set && super::bit(&a.mask, 0) && !super::bit(&a.mask, 1) && super::bit(&a.mask, 2)
    );
    assert_eq!(a.scores, vec![0.5, 1.0, 0.0], "clamped by the host");
    assert_eq!(a.pick, Some(2));
    assert_eq!(a.declined, 0);

    let a = read(
        decision(None, Vec::new(), None, None),
        FILTER | SCORE | PICK,
        3,
    )
    .unwrap();
    assert_eq!(
        a.declined,
        FILTER | SCORE | PICK,
        "nothing answered: every asked stage is declined"
    );
    assert!(!a.busy);

    let a = read(
        decision(Some(Vec::new()), Vec::new(), None, None),
        FILTER,
        3,
    )
    .unwrap();
    assert!(
        a.mask_set && a.mask.iter().all(|w| *w == 0),
        "present and empty sheds, never a fallback"
    );
}

#[test]
fn a_decline_names_its_stages_and_a_budget_decline_is_busy() {
    let a = read(
        decision(
            None,
            Vec::new(),
            None,
            Some((SCORE | PICK, proto::decline::Reason::Budget as i32)),
        ),
        FILTER | SCORE | PICK,
        2,
    )
    .unwrap();
    assert_eq!(
        a.declined,
        FILTER | SCORE | PICK,
        "the filter asked and unanswered is declined too"
    );
    assert!(a.busy);
    let a = read(
        decision(
            Some(vec![1]),
            Vec::new(),
            None,
            Some((PICK, proto::decline::Reason::NotApplicable as i32)),
        ),
        FILTER | PICK,
        2,
    )
    .unwrap();
    assert_eq!(a.declined, PICK);
    assert!(a.mask_set);
}

#[test]
fn the_validity_rules_refuse_the_answer_whole() {
    assert_eq!(
        read(decision(Some(vec![3]), Vec::new(), None, None), FILTER, 3).unwrap_err(),
        Invalid::Survivor
    );
    assert_eq!(
        read(
            decision(Some(vec![1, 1]), Vec::new(), None, None),
            FILTER,
            3
        )
        .unwrap_err(),
        Invalid::Survivor
    );
    assert_eq!(
        read(decision(None, vec![0.1], None, None), SCORE, 3).unwrap_err(),
        Invalid::Scores
    );
    assert_eq!(
        read(
            decision(None, vec![0.1, f32::NAN, 0.3], None, None),
            SCORE,
            3
        )
        .unwrap_err(),
        Invalid::Scores
    );
    assert_eq!(
        read(
            decision(Some(vec![0]), Vec::new(), Some(1), None),
            FILTER | PICK,
            3
        )
        .unwrap_err(),
        Invalid::Pick
    );
    assert_eq!(
        read(decision(None, Vec::new(), Some(3), None), PICK, 3).unwrap_err(),
        Invalid::Pick
    );
    assert_eq!(
        read(
            decision(Some(vec![0]), Vec::new(), None, Some((FILTER, 1))),
            FILTER,
            3
        )
        .unwrap_err(),
        Invalid::DeclinedAndAnswered
    );
    assert_eq!(
        read(
            decision(None, Vec::new(), Some(0), Some((PICK, 1))),
            PICK,
            3
        )
        .unwrap_err(),
        Invalid::DeclinedAndAnswered
    );
}

#[test]
fn a_band_beside_asked_scores_is_read() {
    for band in [Some(0.375), Some(0.0), Some(1.0), None] {
        let d = proto::PluginDecision {
            tie_band: band,
            ..decision(None, vec![0.625], None, None)
        };
        let answer = read(d, SCORE, 1).expect("scores with an optional band");
        assert_eq!(answer.band, band);
    }
}

#[test]
fn a_band_that_is_no_value_from_0_to_1_is_refused() {
    for band in [f32::NAN, -0.125, 1.125, f32::INFINITY, f32::NEG_INFINITY] {
        let d = proto::PluginDecision {
            tie_band: Some(band),
            ..decision(None, vec![0.875], None, None)
        };
        assert_eq!(
            read(d, SCORE, 1).expect_err("a band outside the range"),
            Invalid::Band,
            "{band}"
        );
    }
    assert_eq!(
        Invalid::Band.to_string(),
        "the tie band is not a finite value from 0 to 1 beside its scores"
    );
}

#[test]
fn a_band_beside_a_declined_scorer_or_beside_no_scores_is_refused() {
    let band = Some(0.1875);
    for (decline, invalid) in [
        (
            Some((SCORE, DeclineReason::NotApplicable as i32)),
            Invalid::DeclinedAndAnswered,
        ),
        (None, Invalid::Band),
    ] {
        let d = proto::PluginDecision {
            tie_band: band,
            ..decision(None, Vec::new(), None, decline)
        };
        assert_eq!(
            read(d, SCORE, 1).expect_err("a band without scores"),
            invalid
        );
    }
}

#[test]
fn a_band_where_the_scorer_was_not_asked_is_not_read() {
    for band in [0.6875, f32::NAN] {
        let d = proto::PluginDecision {
            tie_band: Some(band),
            ..decision(None, vec![0.4375], Some(0), None)
        };
        let answer = read(d, PICK, 1).expect("the pick answers the only stage asked");
        assert_eq!((answer.pick, answer.band), (Some(0), None));
        assert!(answer.scores.is_empty());
    }
}

#[test]
fn a_decision_with_no_band_encodes_as_before_and_tag_7_reads_as_the_band() {
    let d = proto::PluginDecision {
        decision_id: 29,
        ..decision(None, vec![0.625], None, None)
    };
    let before = [
        0x09, 0x1d, 0, 0, 0, 0, 0, 0, 0, // decision id
        0x1a, 0x04, 0, 0, 0x20, 0x3f, // scores
    ];
    assert_eq!(d.encode_to_vec(), before);

    let mut with_band = before.to_vec();
    with_band.extend([0x3d, 0, 0, 0, 0x3e]);
    let decoded = proto::PluginDecision::decode(with_band.as_slice()).expect("a decision");
    assert_eq!(
        decoded,
        proto::PluginDecision {
            tie_band: Some(0.125),
            ..d
        }
    );
}

#[test]
fn more_increments_than_declared_counters_break_the_contract() {
    let counted = |increments: usize| proto::PluginDecision {
        counters: vec![proto::CounterDelta { id: 0, by: 1 }; increments],
        ..decision(None, Vec::new(), Some(2), None)
    };
    let within = super::read(counted(2), PICK, 3, 2).unwrap();
    assert_eq!(within.counters.len(), 2, "an id named twice is no fault");
    assert_eq!(
        super::read(counted(3), PICK, 3, 2).unwrap_err(),
        Invalid::Counters
    );
    assert_eq!(
        super::read(counted(1), PICK, 3, 0).unwrap_err(),
        Invalid::Counters,
        "a plugin that declared none sends none"
    );
}

#[test]
fn a_pick_without_a_filter_answer_is_judged_against_the_candidate_count() {
    let a = read(decision(None, Vec::new(), Some(2), None), PICK, 3).unwrap();
    assert_eq!(a.pick, Some(2));
}

/// A frame carries the request's block hashes for a row that declares
/// them in either form, the first of a list past the contract's range,
/// and none for a row that declares neither, whatever else it declares.
#[test]
fn a_frame_carries_the_block_hashes_its_row_declares() {
    use crate::{
        document::spec::{Needs, MAX_BLOCK_HASHES},
        engine::view::{PoolFacts, RequestFacts},
        plugins::shared::config::Exposure,
    };

    let most = usize::try_from(MAX_BLOCK_HASHES).expect("the range fits a length");
    let held: Vec<u64> = (1..).take(most + 2).collect();
    let sent = |list: &[u64], needs: Needs| {
        let facts = RequestFacts {
            model: "m",
            block_hashes: list,
            ..RequestFacts::default()
        };
        super::decide(
            1,
            &facts,
            &[],
            &PoolFacts::default(),
            &Exposure::default(),
            needs,
        )
        .request
        .expect("a frame carries its request")
        .block_hashes
    };
    let tokens = Needs {
        token_hashes: true,
        ..Needs::default()
    };
    let text = Needs {
        text_hashes: true,
        ..Needs::default()
    };
    // Every other need, and neither form of the hashes.
    let neither = Needs {
        text: true,
        token_hashes: false,
        text_hashes: false,
        prompt_size: true,
        loads: true,
        events: true,
        sent_prefixes: true,
        ring: true,
        headers: true,
        media: true,
        active_sessions: true,
        session_in_flight: true,
        affinity: true,
        sample: 8,
        signal_cells: true,
    };
    assert_eq!(sent(&held[..3], tokens), [1, 2, 3]);
    assert_eq!(sent(&held[..3], text), [1, 2, 3]);
    assert!(sent(&held[..3], Needs::default()).is_empty());
    assert!(sent(&held, neither).is_empty());
    assert_eq!(
        sent(&held[..most], tokens),
        held[..most],
        "a list at the range, whole"
    );
    assert_eq!(
        sent(&held, tokens),
        held[..most],
        "the first of a longer list"
    );
}

/// A frame carries the request's size for a row that declares it, and
/// zero for a row that does not, whatever else it declares. A plugin
/// that reads the size undeclared reads zero at once: of a request its
/// host read whole as of one its host sent on unread.
#[test]
fn a_frame_carries_the_requests_size_its_row_declares() {
    use crate::{
        document::spec::Needs,
        engine::view::{PoolFacts, RequestFacts},
        plugins::shared::config::Exposure,
    };

    let sent = |needs: Needs| {
        let facts = RequestFacts {
            model: "m",
            prompt_tokens: 512,
            ..RequestFacts::default()
        };
        super::decide(
            1,
            &facts,
            &[],
            &PoolFacts::default(),
            &Exposure::default(),
            needs,
        )
        .request
        .expect("a frame carries its request")
        .prompt_tokens
    };
    let sized = Needs {
        prompt_size: true,
        ..Needs::default()
    };
    let other = Needs {
        token_hashes: true,
        text_hashes: true,
        loads: true,
        ..Needs::default()
    };
    assert_eq!(sent(sized), 512);
    assert_eq!(sent(other), 0);
    assert_eq!(sent(Needs::default()), 0);
}

/// A frame's lease states the size its request's frame stated: the
/// count of the request's token ids, and for a request of text alone
/// its text's bytes over four, whatever its characters. A text of 400
/// two-byte characters reads 200 in the request's frame and in the
/// lease's: a plugin that books by the one and frees by the other
/// keeps its books.
#[test]
fn a_frames_lease_states_the_size_its_requests_frame_stated() {
    use crate::{
        document::spec::Needs,
        engine::view::{PoolFacts, RequestFacts},
        plugins::shared::config::Exposure,
        tracking::lease::Lease,
    };

    let size = |tokens, chars, bytes| {
        let told = Lease {
            tokens,
            chars,
            bytes,
            ..Lease::default()
        };
        super::lease(&told, &Exposure::default()).tokens
    };
    let sizes = (size(7, 400, 800), size(0, 400, 800), size(0, 0, 0));
    assert_eq!(sizes, (7, 200, 0));
    let worded = RequestFacts {
        model: "m",
        prompt_chars: 400,
        prompt_bytes: 800,
        ..RequestFacts::default()
    };
    let sized = Needs {
        prompt_size: true,
        ..Needs::default()
    };
    let (pool, shown) = (PoolFacts::default(), Exposure::default());
    let sent = super::decide(1, &worded, &[], &pool, &shown, sized).request;
    let sent = sent.expect("a frame carries its request").prompt_tokens;
    assert_eq!(
        (sent, size(0, worded.prompt_chars, worded.prompt_bytes)),
        (200, 200)
    );
}

/// The frame carries each candidate's requests in flight as the
/// request's facts read them: the row's own count.
#[test]
fn the_frame_gives_a_row_the_count_its_request_reads() {
    use crate::{
        document::spec::Needs,
        engine::view::{Candidate, PoolFacts, RequestFacts},
        plugins::shared::config::Exposure,
    };

    let rows: Vec<Candidate> = (0..)
        .zip([4, 0, 9])
        .map(|(position, in_flight)| Candidate {
            position,
            in_flight,
            ..Candidate::default()
        })
        .collect();
    let facts = RequestFacts::default();
    let frame = super::decide(
        1,
        &facts,
        &rows,
        &PoolFacts::default(),
        &Exposure::default(),
        Needs::default(),
    );
    let read: Vec<u32> = frame.candidates.iter().map(|c| c.in_flight).collect();
    assert_eq!(read, [4, 0, 9]);
}

/// A plugin's frame carries no client key, whatever the plugin is shown
/// of the tenant and the session.
#[test]
fn the_frame_carries_no_client_key() {
    use crate::{
        document::spec::Needs,
        engine::view::{PoolFacts, RequestFacts},
        plugins::shared::config::{Exposure, Reveal},
    };

    let facts = RequestFacts {
        client_key: 29,
        ..RequestFacts::default()
    };
    let exposure = Exposure {
        tenant: Reveal::Plain,
        session_key: Reveal::Plain,
        ..Exposure::default()
    };
    let frame = super::decide(
        1,
        &facts,
        &[],
        &PoolFacts::default(),
        &exposure,
        Needs::default(),
    );
    assert_eq!(frame.request.expect("the frame's request").client_key, 0);
}

/// A plugin's frame says nothing of who handed its request's session
/// key, though the plugin is shown the key: the word is a host's to its
/// controller.
#[test]
fn the_frame_states_no_source_of_its_session_key() {
    use crate::{
        document::spec::Needs,
        engine::view::{Identity, PoolFacts, RequestFacts},
        plugins::shared::config::{Exposure, Reveal},
    };

    let facts = RequestFacts {
        session_key: Identity::plain("req-1"),
        session_key_from_rid: true,
        ..RequestFacts::default()
    };
    let exposure = Exposure {
        session_key: Reveal::Plain,
        ..Exposure::default()
    };
    let frame = super::decide(
        1,
        &facts,
        &[],
        &PoolFacts::default(),
        &exposure,
        Needs::default(),
    );
    let request = frame.request.expect("the frame's request");
    assert!(request.session_key.is_some(), "the plugin is shown the key");
    assert!(!request.session_key_from_rid);
}

/// The frame carries each candidate's count of the request's session
/// key, the row's own: a sessions rule's cap reads it there.
#[test]
fn the_frame_gives_a_row_the_keys_count_its_request_reads() {
    use crate::{
        document::spec::Needs,
        engine::view::{Candidate, PoolFacts, RequestFacts},
        plugins::shared::config::Exposure,
    };

    let rows: Vec<Candidate> = (0..)
        .zip([5, 0, 3])
        .map(|(position, session_in_flight)| Candidate {
            position,
            session_in_flight,
            ..Candidate::default()
        })
        .collect();
    let frame = super::decide(
        1,
        &RequestFacts::default(),
        &rows,
        &PoolFacts::default(),
        &Exposure::default(),
        Needs::default(),
    );
    let read = frame.candidates.iter().map(|c| c.session_in_flight);
    assert_eq!(read.collect::<Vec<u32>>(), [5, 0, 3]);
}

/// The view's trace is the contract's pair. A counted trace leaves with
/// its count and no word of a draw; a drawn one leaves with the count
/// unset, which is no zero, and `drew_first` set.
#[test]
fn the_views_trace_says_its_count_or_that_the_decision_drew() {
    use crate::engine::decision::{Survivors, Trace};
    let counted = Trace {
        profile: 3,
        survivors: Survivors::counted(5),
        seed: 41,
        generation: 6,
        ..Trace::default()
    };
    let wire = counted.wire();
    assert_eq!((wire.survivors, wire.drew_first), (Some(5), false));
    assert_eq!((wire.profile, wire.seed, wire.generation), (3, 41, 6));
    let drawn = Trace {
        survivors: Survivors::NOT_COUNTED,
        ..counted
    };
    let wire = drawn.wire();
    assert_eq!((wire.survivors, wire.drew_first), (None, true));
}

/// The trace a plugin is shown carries the veto the all-vetoed rule
/// lifted, and none for a pick no rule made.
#[test]
fn the_views_trace_carries_the_lifted_veto() {
    use crate::engine::decision::{LiftedVeto, Trace};
    let steered = Trace {
        lifted: LiftedVeto::Overload,
        ..Trace::default()
    };
    assert_eq!(steered.wire().lifted, proto::LiftedVeto::Overload as i32);
    assert_eq!(
        Trace::default().wire().lifted,
        proto::LiftedVeto::None as i32
    );
}

#[test]
fn the_plugin_decision_carries_the_no_profile_mark() {
    use prost::Message as _;

    use crate::{
        engine::decision::ShedReason, plugins::shared::config::Exposure, tracking::lease::Lease,
    };

    let shed = decision::shed(ShedReason::NoEndpoints, "wires", 0, 0);
    let decision = super::decision(&shed, &Lease::default(), &Exposure::default());
    assert_eq!(
        decision.trace.expect("a trace").encode_to_vec(),
        [0x10, 0, 0x70, 1]
    );
}

/// A measurement for the Numbers table, not a check: run in release
/// with `--ignored measure --nocapture`. What one frame costs before
/// the stream takes it, with a load row on every candidate: the message
/// built from the view (the deciding thread's share), its encoding (the
/// I/O thread's) and its size on the wire. The row declares the
/// request's hashes, and the last two frames carry a prompt's: the
/// view's default of 256 and the contract's most.
#[test]
#[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
#[expect(
    clippy::print_stdout,
    reason = "the measurement reports its rows on stdout under --nocapture"
)]
fn measure_the_frame_built_and_encoded() {
    use std::{hint::black_box, time::Instant};

    use prost::Message;

    use crate::{
        document::spec::{Needs, MAX_BLOCK_HASHES},
        engine::view::{Candidate, LoadRow, PoolFacts, RequestFacts, WorkerId},
        plugins::shared::config::Exposure,
    };

    let facts = RequestFacts {
        model: "m",
        prompt_tokens: 512,
        budget_micros: 25_000,
        seed: 9,
        ..RequestFacts::default()
    };
    let exposure = Exposure::default();
    let needs = Needs {
        token_hashes: true,
        ..Needs::default()
    };
    let prompt: Vec<u64> = (1..=u64::from(MAX_BLOCK_HASHES)).collect();
    let frames = [
        (8u32, 0),
        (64, 0),
        (128, 0),
        (512, 0),
        (64, 256),
        (64, prompt.len()),
    ];
    for (n, hashes) in frames {
        let facts = RequestFacts {
            block_hashes: &prompt[..hashes],
            ..facts
        };
        let candidates: Vec<Candidate> = (0..n)
            .map(|i| Candidate {
                position: i,
                id: WorkerId(u64::from(i) + 100),
                in_flight: n - i,
                load: Some(LoadRow {
                    running: i,
                    queued: 1,
                    ..LoadRow::default()
                }),
                ..Candidate::default()
            })
            .collect();
        let pool = PoolFacts {
            pool_size: n,
            generation: 1,
            ..PoolFacts::default()
        };
        let rounds = 20_000u32;
        let started = Instant::now();
        for id in 0..rounds {
            black_box(super::decide(
                u64::from(id),
                &facts,
                black_box(&candidates),
                &pool,
                &exposure,
                needs,
            ));
        }
        let build = started.elapsed().as_nanos() / u128::from(rounds);
        let request = proto::PluginRequest {
            request: Some(proto::plugin_request::Request::Decide(super::decide(
                1,
                &facts,
                &candidates,
                &pool,
                &exposure,
                needs,
            ))),
        };
        let mut wire = Vec::with_capacity(request.encoded_len());
        let started = Instant::now();
        for _ in 0..rounds {
            wire.clear();
            black_box(&request)
                .encode(&mut wire)
                .expect("the buffer grows");
        }
        let encode = started.elapsed().as_nanos() / u128::from(rounds);
        println!(
                "a frame of {n} candidates and {hashes} block hashes: built in {build} ns, encoded in {encode} ns, {} bytes",
                wire.len()
            );
    }
}

/// The contract carries the models a worker left as a kind of protocol
/// 2: the mapper hands a plugin the word with its models, and the
/// write's own word as a kind every version carries.
#[test]
fn a_plugin_is_told_the_models_a_worker_left_as_a_kind_of_protocol_2() {
    let (id, models) = (WorkerId(7), ["m".to_string()]);
    let left = Membership::Left {
        id,
        models: &models,
    };
    let word = membership(&left);
    assert_eq!(since(&word), 2);
    let proto::lifecycle_event::Event::Membership(change) = word else {
        panic!("a membership word");
    };
    let kind = proto::membership_change::Kind::Left as i32;
    assert_eq!((change.kind, change.worker_id), (kind, 7));
    assert_eq!(change.models, models);
    assert_eq!(since(&membership(&Membership::Relabelled(id))), 1);
}

/// The version that first carries a word reads the same before the word
/// is made and after, for every kind the feed tells.
#[test]
fn a_words_first_version_is_read_alike_before_and_after_it_is_made() {
    let (id, models) = (WorkerId(7), ["m".to_string()]);
    let members = [id];
    let words = [
        Membership::Snapshot(&members),
        Membership::Added(id),
        Membership::Removed(id),
        Membership::Replaced { old: id, new: id },
        Membership::Relabelled(id),
        Membership::Resized(id),
        Membership::Left {
            id,
            models: &models,
        },
    ];
    for word in &words {
        assert_eq!(since_of(word), since(&membership(word)));
    }
}
