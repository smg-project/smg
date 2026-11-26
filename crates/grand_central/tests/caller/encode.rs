//! An encode request, served by the program over its three lists.
//!
//! The program routes to pairs behind encoders ([`Plan::Encode`]): a request
//! lends the three lists its generation holds, one that carries items states
//! their keys, and the one call names each item's encoder beside the pair.
//! Each case serves requests as a route would ([`Program::serve`]) and reads
//! what the client was answered and what the workers saw. Where the classic
//! router has a case or a function of its own for the answer, the case here
//! names it and its lines. `worker_selection.rs` is
//! `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`.
//!
//! [`Plan::Encode`]: crate::program::Plan::Encode
//! [`Program::serve`]: crate::program::Program::serve

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use http::{HeaderMap, HeaderValue};
use smg_grand_central::{
    document::{
        AppliedSpec, Assignment, DecodePick, LoadMetricSetting, ModeSettings, PolicySettings,
        PolicySpec, RoutingSettings, ScoreSpec,
    },
    engine::{Identity, ShedReason, Unplaced},
    stages::StageRegistry,
};
use smg_worker::{
    identity::basic::Worker,
    worker::{WorkerStatus, WorkerType},
};
use tokio::time::sleep;

use crate::{
    pair::{self, decode, prefill, sends, typed, PLAIN},
    program::{encode_settings, pair_settings, settings, Answer, Asked, Plan},
    served,
    single::{taken, Heard, Hears},
    started_as, under, under_encode, under_pairs, url, workers, MODEL,
};

/// The address of the encoder at `at`.
fn encoder(at: usize) -> String {
    format!("http://encode-{at}:8000")
}

/// `prefills` prefill workers, `decodes` decode workers and then `encoders`
/// encoders, each kind in the order of its addresses.
fn fleet(prefills: usize, decodes: usize, encoders: usize) -> Vec<Arc<dyn Worker>> {
    let encoders = (0..encoders).map(|at| typed(&encoder(at), WorkerType::Encode));
    let pairs = pair::fleet(prefills, decodes);
    pairs.into_iter().chain(encoders).collect()
}

/// A request for [`MODEL`] that carries the items of `keys` and no header.
fn carrying<'a>(keys: &'a [Identity<'a>]) -> Asked<'a> {
    Asked {
        items: keys,
        ..PLAIN
    }
}

/// An encode request is answered by one call that names each item's encoder
/// beside its pair: an item goes out first, then the prefill, then the
/// decode. An item is placed by its content, so one content keeps one
/// encoder while the pair's sides take their turns behind it, and many
/// contents spread over the encoders. A request of two items sends each to
/// its own encoder, in the items' order.
///
/// The classic router places each item by its content's hash under the
/// encode policy (`assign_encode_workers`, `worker_selection.rs:2780-2821`).
#[test]
fn an_item_goes_to_its_encoder_ahead_of_its_pair() {
    let fleet = fleet(2, 2, 3);
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
    let keys = [Identity::plain("item-a")];
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&carrying(&keys))).collect();
    let kept = answers[0].sent[0].clone();
    for answer in &answers {
        assert_eq!((answer.status, answer.sent.len()), (200, 3), "{answer:?}");
        assert_eq!(answer.sent[0], kept, "one content keeps one encoder");
        assert!(kept.contains("encode"), "{answer:?}");
        assert!(answer.sent[1].contains("prefill"), "{answer:?}");
        assert!(answer.sent[2].contains("decode"), "{answer:?}");
    }
    let times = sends(&answers);
    let sides = [prefill(0), prefill(1), decode(0), decode(1)];
    let mut expected = BTreeMap::from([(kept.as_str(), 4)]);
    expected.extend(sides.iter().map(|side| (side.as_str(), 2)));
    assert_eq!(times, expected, "each side of the pair takes its turn");

    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let answer = program.serve(&carrying(&keys));
    assert_eq!((answer.status, answer.sent.len()), (200, 4), "{answer:?}");
    assert_eq!(answer.sent[0], kept, "the first item's encoder");
    assert!(answer.sent[1].contains("encode"), "{answer:?}");

    let spread: BTreeSet<String> = (0..16)
        .map(|content| {
            let key = format!("content-{content}");
            let keys = [Identity::plain(&key)];
            program.serve(&carrying(&keys)).sent.swap_remove(0)
        })
        .collect();
    assert!(spread.len() > 1, "sixteen contents on one encoder");
}

/// A request that carries no item is placed as its pair: two sends, and no
/// encoder is read. So it is over a model that has no encoder at all.
///
/// The classic router assigns no encoder to a request with no item
/// (`assign_encode_workers`, `worker_selection.rs:2787-2789`). Where the
/// gateway asks the engine, a request with no item is placed on the engine's
/// pair and on no encoder: its case
/// `the_engine_places_an_encode_requests_pair_and_refuses_an_item_by_name`
/// (`worker_selection.rs:5410-5539`).
#[test]
fn a_request_with_no_item_is_placed_as_its_pair() {
    for encoders in [2, 0] {
        let fleet = fleet(1, 1, encoders);
        let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
        let answer = program.serve(&PLAIN);
        assert_eq!(
            answer,
            Answer {
                sent: vec![prefill(0), decode(0)],
                status: 200,
                unplaced: None,
            },
            "over {encoders} encoders"
        );
    }
}

/// The header a keyed request of the cases below states its key under.
const KEY_HEADER: &str = "x-smg-routing-key";

/// The headers of a request that states `key`.
fn keyed(key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(key).expect("a header's value");
    headers.insert(KEY_HEADER, value);
    headers
}

/// A document of three sides whose binding keeps a sessions rule: the
/// encode side places an item on the plain ring, and the rule pins a key
/// by its header and hands the key on where `cap` of its requests run on
/// the pinned worker. Each side of the pair takes its workers in turn, or,
/// with `fewest`, the worker that holds the fewest requests. The rule is
/// the stock `manual` preset's, and the view lists the key's header.
fn pinned(cap: u16, fewest: bool) -> AppliedSpec {
    let sides = if fewest {
        "power_of_two"
    } else {
        "round_robin"
    };
    let document = PolicySpec::epd_presets("consistent_hashing", sides, sides);
    let mut document = document.expect("the presets");
    if fewest {
        for side in ["prefill", "decode"] {
            let profile = document.profiles.get_mut(side).expect("the side's profile");
            profile.pick = "count-first".into();
        }
    }
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let ruled = rule.as_mut().expect("the stock rule");
    (ruled.inflight_cap, ruled.assignment) = (cap, Assignment::MinLoad);
    document.bindings[0].sessions = rule;
    document.view.headers.push(KEY_HEADER.to_owned());
    AppliedSpec::validate(&document).expect("the document validates")
}

/// Under a sessions rule the pair of an encode request is its key's: the
/// key's requests keep the prefill and the decode its first request took,
/// with an item and with none, while each side of the pair would else take
/// its workers in turn. A request with no key takes those turns.
///
/// The classic gateway's sticky map pins a key within its model and its
/// role, prefill or decode, and its three-sided selection places the pair
/// through that same call, with items and without
/// (`model_gateway/src/policies/registry.rs:783`, `:801-806`,
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2532`).
/// The lines are main's at `9b72a221d`.
#[test]
fn a_key_keeps_its_pair_with_an_item_and_with_none() {
    let fleet = fleet(3, 3, 2);
    let spec = pinned(u16::MAX, false);
    let (program, _loads, _task) = started_as(Plan::Encode, spec, &fleet, served);
    let headers = keyed("session-7");
    let keys = [Identity::plain("item-a")];
    let with_item = Asked {
        headers: Some(&headers),
        ..carrying(&keys)
    };
    let bare = Asked {
        headers: Some(&headers),
        ..PLAIN
    };
    let first = program.serve(&with_item);
    assert_eq!((first.status, first.sent.len()), (200, 3), "{first:?}");
    let pair = &first.sent[1..];
    for _ in 0..4 {
        assert_eq!(&program.serve(&with_item).sent[1..], pair, "with an item");
        assert_eq!(program.serve(&bare).sent, pair, "with no item");
    }
    let turns: BTreeSet<Vec<String>> = (0..6).map(|_| program.serve(&PLAIN).sent).collect();
    assert!(turns.len() > 1, "a request with no key takes its turn");
}

/// Under a cap of two the third request of a key leaves the pinned worker:
/// with two of the key's requests running on its prefill, the key's next
/// request takes another prefill, and its decode, which runs none, stays.
/// Each side takes the worker that holds the fewest requests, so the
/// request handed on never lands on the pinned prefill again.
///
/// The classic gateway's sticky map hands a key on where two of its
/// requests run on the pinned worker, counted under the key's own text
/// (`model_gateway/src/policies/registry.rs:394`, `:810`).
#[test]
fn the_third_request_of_a_key_leaves_its_pinned_prefill_under_a_cap_of_two() {
    const KEY: &str = "session-7";
    let fleet = fleet(3, 3, 2);
    let (program, _loads, _task) = started_as(Plan::Encode, pinned(2, true), &fleet, served);
    let headers = keyed(KEY);
    let keys = [Identity::plain("item-a")];
    let asked = Asked {
        headers: Some(&headers),
        ..carrying(&keys)
    };
    let first = program.serve(&asked);
    assert_eq!((first.status, first.sent.len()), (200, 3), "{first:?}");
    assert_eq!(program.serve(&asked).sent, first.sent, "under the cap");
    let held = fleet.iter().find(|worker| worker.url() == first.sent[1]);
    let held = held.expect("the key's prefill");
    for _ in 0..2 {
        held.increment_load();
        held.increment_routing_key_load(KEY);
    }
    let third = program.serve(&asked);
    assert_eq!((third.status, third.sent.len()), (200, 3), "{third:?}");
    assert_ne!(third.sent[1], first.sent[1], "another prefill");
    assert_eq!(third.sent[2], first.sent[2], "the key's decode");
}

/// An item goes by its content whatever the request's key: one content
/// keeps one encoder under each key and under none, and the contents of
/// one key spread over the encoders.
///
/// The classic gateway asks the encode policy for an item under the hash
/// of its content alone, and its sticky map never sees an item
/// (`assign_encode_workers`, `worker_selection.rs:2780-2796`, `:2823-2829`).
/// The lines are main's at `9b72a221d`.
#[test]
fn an_item_goes_by_its_content_whatever_the_requests_key() {
    let fleet = fleet(2, 2, 3);
    let (program, _loads, _task) = started_as(Plan::Encode, pinned(2, false), &fleet, served);
    let encoder_of = |headers: Option<&HeaderMap>, content: &str| {
        let keys = [Identity::plain(content)];
        let answer = program.serve(&Asked {
            headers,
            ..carrying(&keys)
        });
        assert_eq!((answer.status, answer.sent.len()), (200, 3), "{answer:?}");
        answer.sent[0].clone()
    };
    let kept = encoder_of(None, "item-a");
    for key in ["session-7", "session-8", "session-9"] {
        assert_eq!(encoder_of(Some(&keyed(key)), "item-a"), kept, "{key}");
    }
    let session = keyed("session-7");
    let spread: BTreeSet<String> = (0..16)
        .map(|content| encoder_of(Some(&session), &format!("content-{content}")))
        .collect();
    assert!(spread.len() > 1, "sixteen contents on one encoder");
}

/// The manual policy as a gateway's settings say it: a key it has not seen
/// goes to the worker that holds the fewest requests.
fn manual() -> PolicySettings {
    PolicySettings::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: Assignment::MinLoad,
    }
}

/// The manual policy runs on the pair's two sides under the encode mode, as
/// a gateway's settings say it. A key's requests keep the prefill and the
/// decode its first request took, with an item and with none, while lighter
/// workers stand beside them. Another key's first request and a request
/// with no key go to a lighter worker on each side. An item goes to its
/// encoder by its content, whatever the request's key.
///
/// The classic router's `manual` policy keeps a key's worker under the role
/// it is asked for, and places a key it has not seen, and a request with no
/// key, by the configured assignment
/// (`model_gateway/src/policies/manual.rs:381-402`, `:228-236`). Its
/// three-sided selection asks those two policies for the pair
/// (`worker_selection.rs:2660`), and places each item under the encode
/// policy by its content's hash, with no key of the request's
/// (`:2785-2808`). Its own check starts the mode under that policy on the
/// prefill and the decode side, and on no encode side
/// (`model_gateway/src/config/validation.rs:1001-1010`, `:319-328`). The
/// lines are main's at `13f5cbe29`.
#[test]
fn the_manual_policy_places_an_encode_requests_pair_by_its_key() {
    let fleet = fleet(3, 3, 3);
    let (program, _loads, _task) = under_encode(manual(), &fleet, served);
    let (session, other) = (keyed("session-7"), keyed("session-8"));
    let keys = [Identity::plain("item-a")];
    let with_item = Asked {
        headers: Some(&session),
        ..carrying(&keys)
    };
    let bare = Asked {
        headers: Some(&session),
        ..PLAIN
    };
    let first = program.serve(&with_item);
    assert_eq!((first.status, first.sent.len()), (200, 3), "{first:?}");
    let pair = &first.sent[1..];
    // A key not seen yet, and a request with no key, go to lighter workers.
    for worker in &fleet {
        if pair.iter().any(|sent| worker.url() == *sent) {
            worker.increment_load();
        }
    }
    for _ in 0..4 {
        assert_eq!(program.serve(&with_item).sent, first.sent, "with an item");
        assert_eq!(program.serve(&bare).sent, pair, "with no item");
    }
    let stranger = Asked {
        headers: Some(&other),
        ..carrying(&keys)
    };
    for (asked, who) in [(stranger, "another key"), (carrying(&keys), "no key")] {
        let answer = program.serve(&asked);
        assert_eq!((answer.status, answer.sent.len()), (200, 3), "{who}");
        assert_eq!(answer.sent[0], first.sent[0], "{who}: one content");
        assert_ne!(answer.sent[1], pair[0], "{who}: a lighter prefill");
        assert_ne!(answer.sent[2], pair[1], "{who}: a lighter decode");
    }
}

/// A gateway of many routers mounts its three-sided router in every mode,
/// so its settings plan an encode side under the regular mode and under a
/// pair mode as well. An item then goes to its encoder ahead of its pair,
/// as it does under the encode mode, and a request with no item is placed
/// as its pair. So it is under the manual policy on the pair's two sides.
///
/// Where the plan of three sides has no document the settings say the
/// pair's plan, as where the pair's sides and the main policy name two
/// block sizes. The engine then names no worker for a request that carries
/// an item, under the binding's name.
///
/// The classic gateway builds that router in every mode of many routers,
/// its encode side under `consistent_hashing` where the mode names no
/// encode policy (`model_gateway/src/routers/factory.rs:207-218`, `:246`,
/// `:289-292`).
#[test]
fn many_routers_place_an_item_in_every_mode_that_plans_three_sides() {
    let keys = [Identity::plain("item-a")];
    let many_routers = |one_router: RoutingSettings| RoutingSettings {
        enable_igw: true,
        ..one_router
    };
    // The program routes to pairs behind encoders whatever mode the
    // settings name, as such a gateway's three-sided router does.
    let started = |said: &RoutingSettings, fleet: &[Arc<dyn Worker>]| {
        let document = said.policy_document();
        let document = document.expect("the settings have a document");
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        started_as(Plan::Encode, spec, fleet, served)
    };
    let manual_sides = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(manual()),
            decode_policy: Some(manual()),
        },
        ..settings(PolicySettings::RoundRobin)
    };
    for one_router in [
        settings(PolicySettings::RoundRobin),
        pair_settings(PolicySettings::RoundRobin),
        manual_sides,
    ] {
        let mode = format!("{:?}", one_router.mode);
        let fleet = fleet(1, 1, 1);
        let (program, _loads, _task) = started(&many_routers(one_router), &fleet);
        assert_eq!(
            program.serve(&carrying(&keys)),
            Answer {
                sent: vec![encoder(0), prefill(0), decode(0)],
                status: 200,
                unplaced: None,
            },
            "{mode}"
        );
        assert_eq!(
            program.serve(&PLAIN),
            Answer {
                sent: vec![prefill(0), decode(0)],
                status: 200,
                unplaced: None,
            },
            "{mode}"
        );
    }

    // One view holds one block size: the sides name 16 and the main policy
    // 32, so the settings plan the pair alone.
    let over = |blocks: usize| {
        let mut policy = PolicySettings::of_label("cache_aware").expect("a label's policy");
        if let PolicySettings::CacheAware { block_size, .. } = &mut policy {
            *block_size = blocks;
        }
        policy
    };
    let fallen = many_routers(RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(over(16)),
            decode_policy: Some(over(16)),
        },
        ..settings(over(32))
    });
    let fleet = fleet(1, 1, 1);
    let (program, _loads, _task) = started(&fallen, &fleet);
    let answer = program.serve(&carrying(&keys));
    assert!(answer.sent.is_empty(), "{answer:?}");
    assert!(
        matches!(
            answer.unplaced,
            Some(Unplaced::Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: "binding",
                ..
            })
        ),
        "{answer:?}"
    );
}

/// An encoder that cannot be sent to takes no item: the item goes to another
/// encoder. Where none can be sent to, the engine names no worker under the
/// encode side's own name, ahead of the pair, and the program answers a 503.
/// So it does over a model that has no encoder at all. A request with no
/// item is still placed as its pair.
///
/// The classic router keeps the encoders that are available
/// (`worker_selection.rs:2570`), and answers a request that needs one with
/// none of them left as unavailable, ahead of its pair (`:2587-2590`).
#[test]
fn an_encoder_that_cannot_be_sent_to_takes_no_item() {
    let fleet = fleet(1, 1, 3);
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
    let keys = [Identity::plain("item-a")];
    let first = program.serve(&carrying(&keys));
    assert_eq!((first.status, first.sent.len()), (200, 3), "{first:?}");
    let down = fleet.iter().find(|worker| worker.url() == first.sent[0]);
    down.expect("the item's encoder")
        .set_status(WorkerStatus::NotReady);
    let moved = program.serve(&carrying(&keys));
    assert_eq!((moved.status, moved.sent.len()), (200, 3), "{moved:?}");
    assert!(moved.sent[0].contains("encode"), "{moved:?}");
    assert_ne!(moved.sent[0], first.sent[0], "another encoder");

    for encoder in &fleet[2..] {
        encoder.set_status(WorkerStatus::NotReady);
    }
    // The refusal of a request no encoder takes: no send, and the encode
    // side's own name.
    let refused = || Answer {
        sent: Vec::new(),
        status: 503,
        unplaced: Some(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "encoders",
            retry_after_secs: 1,
            side: None,
        }),
    };
    assert_eq!(program.serve(&carrying(&keys)), refused());
    assert_eq!(program.serve(&PLAIN).status, 200, "no item, no encoder");

    let none = self::fleet(1, 1, 0);
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &none, served);
    assert_eq!(program.serve(&carrying(&keys)), refused(), "no encoder");
}

/// An encoder that answers a server error is the request's answer: neither
/// side of the pair is sent, and no attempt follows. An item goes out once
/// a request, so its encoder is not asked again, however many encoders
/// stand.
///
/// The classic router sends an item once a request: the first dispatch
/// takes its encode outputs, and a replay gets none
/// (`model_gateway/src/routers/grpc/common/stages/request_execution.rs:295-299`).
/// Where its encoder answers ahead of the pair (`:966-967`), a dispatch
/// that failed is marked as one that is not asked again (`:461-466`).
/// Where its items launch beside the pair, a failed encode dispatch is
/// answered a 502 that is final for the retry layer
/// (`encode_dispatch_failure_response`, `:2705-2720`).
#[test]
fn an_encoder_that_fails_is_the_requests_answer_and_no_side_is_sent() {
    let fleet = fleet(2, 2, 3);
    let sends = |worker: &dyn Worker| match worker.worker_type() {
        WorkerType::Encode => 502,
        _ => 200,
    };
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, sends);
    let keys = [Identity::plain("item-a")];
    let answer = program.serve(&carrying(&keys));
    let kept = answer.sent[0].clone();
    assert!(kept.contains("encode"), "{answer:?}");
    assert_eq!(
        answer,
        Answer {
            sent: vec![kept],
            status: 502,
            unplaced: None,
        },
        "one send: the item's encoder, and no attempt behind its failure"
    );
    assert_eq!(program.serve(&PLAIN).status, 200, "no item, no encoder");

    // Two items on two encoders, of which the first item's fails: the first
    // send that failed is the request's answer, whatever a later item's
    // encoder answered.
    let fleet = self::fleet(1, 1, 3);
    let failing = Arc::new(Mutex::new(String::new()));
    let fails = Arc::clone(&failing);
    let sends = move |worker: &dyn Worker| {
        let failing = fails.lock().expect("no case panicked");
        if worker.url() == *failing {
            502
        } else {
            200
        }
    };
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, sends);
    let first = program.serve(&carrying(&keys)).sent.swap_remove(0);
    let elsewhere = (0..16).map(|content| format!("content-{content}"));
    let elsewhere = elsewhere.into_iter().find(|key| {
        let keys = [Identity::plain(key)];
        program.serve(&carrying(&keys)).sent[0] != first
    });
    let other = elsewhere.expect("a content of another encoder");
    let second = program.serve(&carrying(&[Identity::plain(&other)]));
    let second = second.sent[0].clone();
    failing.lock().expect("no case panicked").clone_from(&first);
    let two = [Identity::plain("item-a"), Identity::plain(&other)];
    let answer = program.serve(&carrying(&two));
    assert_eq!(
        answer,
        Answer {
            sent: vec![first, second],
            status: 502,
            unplaced: None,
        },
        "both items go out once, and the first one's failure is the answer"
    );
}

/// A retry behind a failed side of the pair states no item and sends none:
/// the item went out with the first attempt, and the retry is placed as the
/// pair alone. It states the pair's two workers as tried and no encoder, so
/// the decode that failed steps aside and the one prefill stays.
///
/// The classic router's retry selects the pair alone again, since the first
/// dispatch launched the encode jobs (the doc of its re-selection,
/// `worker_selection.rs:781-783`): a replay launches none
/// (`model_gateway/src/routers/grpc/common/stages/request_execution.rs:1013-1016`).
/// It names the addresses of the prefill and of the decode as tried, and no
/// encoder's (`model_gateway/src/routers/grpc/context.rs:1717-1721`, read at
/// `model_gateway/src/routers/grpc/pipeline.rs:849-861`).
#[test]
fn a_retry_behind_a_failed_side_states_no_item_and_sends_none() {
    let fleet = fleet(1, 2, 2);
    let failing = decode(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 };
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, sends);
    let keys = [Identity::plain("item-a")];
    // The decodes take their turns, so one of two requests meets the one
    // that fails.
    let answers: Vec<Answer> = (0..2).map(|_| program.serve(&carrying(&keys))).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let kept = answers[0].sent[0].clone();
    assert!(kept.contains("encode"), "{answers:?}");
    let asked_again: Vec<&Answer> = answers
        .iter()
        .filter(|answer| answer.sent.len() > 3)
        .collect();
    let [answer] = asked_again[..] else {
        panic!("one of the two requests meets the failing decode: {answers:?}");
    };
    assert_eq!(
        answer.sent,
        [kept, prefill(0), decode(0), prefill(0), decode(1)],
        "the item goes out once, and the retry is its pair alone"
    );
}

/// An item is counted on its encoder while its own send runs, and under no
/// routing key: the key is the pair's. The prefill and the decode hold the
/// request among their requests in flight from the placement to the last
/// send, and under the request's key. The encoder holds it at the item's
/// send alone, and none under the key: at the pair's two sends it holds
/// nothing.
///
/// The classic router's encode stage counts an item on its encoder until
/// the item's send ends, under no key (`PreparedEncodeJob::dispatch`,
/// `model_gateway/src/routers/grpc/common/stages/encode.rs:190-197`). It
/// keys an item by its content's hash and by no conversation key
/// (`assign_encode_workers`, `worker_selection.rs:2800-2803`).
#[test]
fn an_items_encoder_is_counted_while_its_send_runs_and_under_no_key() {
    const KEY: &str = "session-7";
    // The prefill, the decode, the encoder.
    let fleet = fleet(1, 1, 1);
    // What each worker counts while a send runs: its requests in flight, and
    // those of them under the key.
    let counted = Arc::new(Mutex::new(Vec::new()));
    let (counts, read) = (Arc::clone(&counted), fleet.clone());
    let sends = move |_: &dyn Worker| {
        let running: Vec<(usize, usize)> = read
            .iter()
            .map(|worker| (worker.load(), worker.routing_key_inflight(KEY)))
            .collect();
        counts.lock().expect("no case panicked").push(running);
        200
    };
    // The pair's two sides are under a policy that reads a routing key, and
    // the document's view lists the key's header: a stage may read it, so
    // the placement names the key.
    let settings = encode_settings(PolicySettings::ConsistentHashing);
    let document = settings.policy_document();
    let mut document = document.expect("the settings have a document");
    document.view.headers.push("x-smg-routing-key".to_owned());
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let (program, _loads, _task) = started_as(Plan::Encode, spec, &fleet, sends);
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(KEY).expect("a header's value");
    headers.insert("x-smg-routing-key", value);
    let keys = [Identity::plain("item-a")];
    let asked = Asked {
        headers: Some(&headers),
        ..carrying(&keys)
    };
    let answer = program.serve(&asked);
    assert_eq!(
        answer,
        Answer {
            sent: vec![encoder(0), prefill(0), decode(0)],
            status: 200,
            unplaced: None,
        }
    );
    let counted = counted.lock().expect("no case panicked");
    let (encoding, behind) = ((1, 0), (0, 0));
    let pair = [(1, 1), (1, 1)];
    assert_eq!(
        counted.as_slice(),
        [
            vec![pair[0], pair[1], encoding],
            vec![pair[0], pair[1], behind],
            vec![pair[0], pair[1], behind],
        ],
        "at the item's send, the prefill's and the decode's"
    );
    let after: Vec<usize> = fleet.iter().map(|worker| worker.load()).collect();
    assert_eq!(after, [0, 0, 0], "every guard ended with the request");
}

/// An item stated to a program that routes to no encoder is read by no one:
/// under a pair's plan the request is served as a pair, and under single
/// workers as a single request.
///
/// The classic router reads a request's items in its encode mode's own
/// selection, and in no other (`placed_encode_prefill_decode_workers`,
/// `worker_selection.rs:2554`).
#[test]
fn an_item_stated_to_a_program_that_routes_to_no_encoder_is_read_by_no_one() {
    let keys = [Identity::plain("item-a")];
    let pairs = pair::fleet(1, 1);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &pairs, served);
    let answer = program.serve(&carrying(&keys));
    assert_eq!(
        (answer.status, &answer.sent),
        (200, &vec![prefill(0), decode(0)])
    );

    let single = workers(1);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &single, served);
    let answer = program.serve(&carrying(&keys));
    assert_eq!((answer.status, &answer.sent), (200, &vec![url(0)]));
}

/// An item for a model that has no worker is refused as a request with no
/// item is: the program lends no list and states no item, and the engine
/// names no worker with no endpoint. So it is where no worker ever joined,
/// and where every worker left.
///
/// The library takes items with an encode request's three lists and with no
/// other (`PlaceRequest::encode`): items stated over no list are a caller's
/// fault, which the engine declines by its own name.
#[test]
fn an_item_for_a_model_with_no_worker_is_refused_as_a_request_with_none_is() {
    let keys = [Identity::plain("item-a")];
    let no_worker = || Answer {
        sent: Vec::new(),
        status: 503,
        unplaced: Some(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "none",
            retry_after_secs: 1,
            side: None,
        }),
    };
    // No worker ever joined: the registry holds no generation of the model.
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &[], served);
    assert_eq!(program.serve(&PLAIN), no_worker());
    assert_eq!(program.serve(&carrying(&keys)), no_worker());

    // Every worker left: the registry holds the model's generation no more.
    let fleet = fleet(1, 1, 1);
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
    assert_eq!(program.serve(&carrying(&keys)).status, 200);
    for worker in &fleet {
        assert!(program.leave(worker.url()), "a worker that joined leaves");
    }
    assert!(program.published(MODEL).is_none(), "no generation is left");
    assert_eq!(program.serve(&carrying(&keys)), no_worker());
}

/// The engine hears the pair of an encode request and no word of an encoder:
/// the pair is issued, the item goes out, and each side is dispatched and
/// completed as a pair's is. Where the item's encoder fails, the pair that
/// was issued is given up at the request's end, a side at a time, and the
/// stages hear no more of the request.
///
/// The classic router has no case of its own here: the gateway asks the
/// engine for no item yet.
#[tokio::test(start_paused = true)]
async fn the_engine_hears_the_pair_of_an_encode_request_and_no_encoder() {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept))));
    added.expect("a free name");
    let settings = encode_settings(PolicySettings::RoundRobin);
    let document = settings.policy_document();
    let mut document = document.expect("the settings have a document");
    // One stage of the document hears: the decode side's profile names it.
    let profile = document.profiles.get_mut("decode");
    let profile = profile.expect("the decode side's profile");
    profile.score.push(ScoreSpec::from("hears"));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let spec = spec.expect("the stages hold the scorer");
    let fails = Arc::new(AtomicBool::new(false));
    let (next, went) = (Arc::clone(&fails), Arc::clone(&heard));
    let sends = move |worker: &dyn Worker| {
        let mut words = went.lock().expect("no case panicked");
        words.push("went out".to_owned());
        let encoder = *worker.worker_type() == WorkerType::Encode;
        if encoder && next.swap(false, Ordering::Relaxed) {
            502
        } else {
            200
        }
    };
    let (program, _loads, task) = started_as(Plan::Encode, spec, &fleet(1, 1, 1), sends);
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    // The task's first step reads the registry whole.
    sleep(Duration::from_millis(1)).await;
    let told = taken(&heard);
    assert_eq!(told.len(), 3, "each worker's join: {told:?}");

    let keys = [Identity::plain("item-a")];
    assert_eq!(program.serve(&carrying(&keys)).status, 200);
    let served = [
        "issued Prefill",
        "issued Decode",
        "went out",
        "dispatched Prefill",
        "dispatched Decode",
        "went out",
        "completed Prefill 200 Ok",
        "went out",
        "completed Decode 200 Ok",
    ];
    assert_eq!(taken(&heard), served);

    fails.store(true, Ordering::Relaxed);
    let answer = program.serve(&carrying(&keys));
    assert_eq!((answer.status, answer.sent.len()), (502, 1));
    let given_up = [
        "issued Prefill",
        "issued Decode",
        "went out",
        "released Prefill Abandoned",
        "released Decode Abandoned",
    ];
    assert_eq!(taken(&heard), given_up);
    task.abort();
}

/// An encoder that left takes no later item, and one that joined takes
/// items: each membership change publishes the next generation, whose
/// encoders' list the program cuts anew with the encode side's facts.
#[test]
fn an_encode_request_is_placed_over_the_generation_that_stands() {
    let fleet = fleet(1, 1, 2);
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
    let before = program.published(MODEL).expect("the model's generation");
    let keys = [Identity::plain("item-a")];
    let first = program.serve(&carrying(&keys));
    assert_eq!((first.status, first.sent.len()), (200, 3), "{first:?}");

    assert!(
        program.leave(&first.sent[0]),
        "an encoder that joined leaves"
    );
    let after = program.published(MODEL).expect("the model's generation");
    assert!(!Arc::ptr_eq(&before, &after), "the next generation");
    let left: BTreeSet<String> = (0..8)
        .map(|content| {
            let key = format!("content-{content}");
            let keys = [Identity::plain(&key)];
            program.serve(&carrying(&keys)).sent.swap_remove(0)
        })
        .collect();
    assert_eq!(left.len(), 1, "the one encoder that is left: {left:?}");
    assert!(!left.contains(&first.sent[0]), "{left:?}");

    let joined = encoder(2);
    assert!(program.join(typed(&joined, WorkerType::Encode)));
    let both: BTreeSet<String> = (0..16)
        .map(|content| {
            let key = format!("content-{content}");
            let keys = [Identity::plain(&key)];
            program.serve(&carrying(&keys)).sent.swap_remove(0)
        })
        .collect();
    assert!(both.contains(&joined), "the encoder that joined: {both:?}");
}

/// An encode request served in turn: its item goes out first, then the
/// prefill, and behind the prefill's answer the program asks the placement
/// for the decode again. The second ask states the request as the first
/// call stated it, with its item, over the three lists, and the engine
/// reads the pair's two of them. The first decision names the decode that
/// holds no request. While the prefill runs that decode takes three
/// requests, and the second ask names the other one: the item is sent once.
///
/// The classic router has no case of its own here: the gateway asks the
/// engine for no item yet.
#[test]
fn an_encode_requests_decode_is_asked_again_behind_its_prefills_answer() {
    let fewest = PolicySettings::PowerOfTwo {
        load_metric: LoadMetricSetting::LeastRequests,
    };
    let settings = RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: None,
            prefill_policy: None,
            decode_policy: Some(fewest),
        },
        ..settings(PolicySettings::RoundRobin)
    };
    let document = settings.policy_document();
    let mut document = document.expect("the settings have a document");
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    // The prefill, the two decodes, the encoder.
    let fleet = fleet(1, 2, 1);
    let (idle, busy) = (Arc::clone(&fleet[1]), &fleet[2]);
    busy.increment_load();
    let sends = move |worker: &dyn Worker| {
        if *worker.worker_type() == WorkerType::Prefill {
            (0..3).for_each(|_| idle.increment_load());
        }
        200
    };
    let (program, _loads, _task) = started_as(Plan::Encode, spec, &fleet, sends);
    let program = program.in_turn();
    let keys = [Identity::plain("item-a")];
    let answer = program.serve(&carrying(&keys));
    assert_eq!(
        answer,
        Answer {
            sent: vec![encoder(0), prefill(0), decode(1)],
            status: 200,
            unplaced: None,
        },
        "the item once, the prefill, and the decode the second ask names"
    );
}
