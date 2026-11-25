//! The rows of a pair's request under a document that sheds a list wholly
//! over its thresholds ([`ROWS`]), held for an encode request that carries
//! an item: its pair goes the pair's way, by the pair's own routine
//! ([`again`]). One encoder stands open and serves every prefill, so the
//! pair's decision is handed the two lists whole.
//!
//! The gateway's own selection for such a request makes its pair by its
//! pair selection, under one filter of its own, which takes the workers
//! an available encoder serves and asks no route's rule
//! (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2663`,
//! on main `56773d28d`). With the one encoder here serving every prefill
//! that filter takes every worker. So no state of that selection stands
//! behind a row whose rule refuses a worker: such a row is held for a
//! host's own rule, by the pair's routine, and answers as the pair's row
//! does. What the cases of a part of a list say of that selection was
//! read from those lines and not run.
//!
//! [`again`]: super::super::super::pair::again
//! [`ROWS`]: super::super::pair::rows::ROWS

use super::{
    super::pair::rows::{
        asked_inside, failing, holds, masked, partners, shed_of, stated, Answer, Kind, Row, ADMITS,
        SATURATED,
    },
    *,
};

/// One item, under a key its host holds as a fingerprint.
const KEYS: [Identity<'static>; 1] = [Identity::hashed(1)];

/// The encode document of the `power_of_two` preset on each of the three
/// sides, under bindings that shed a list wholly over its thresholds.
fn shedding() -> &'static InProcess {
    let mut document = fewest(DecodePick::default());
    for binding in &mut document.bindings {
        binding.overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    hearing(document).0
}

/// What a case states of an encode request beside a row's two lists.
#[derive(Clone, Copy)]
struct Beside<'a> {
    /// The encode class each prefill hands to: the one encoder is of class
    /// `1`.
    encode_classes: &'a [u32],
    /// Where the request's decode must take images, the workers that do,
    /// by their list and their place in it. `None`: no such demand.
    sees: Option<fn(usize, usize) -> bool>,
}

/// The shed `engine` answers an encode request of one item over a row's
/// two lists and one open encoder, and the times its rule was asked.
fn asked_of(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
    beside: Beside<'_>,
) -> (Answer, usize) {
    asked_among(engine, prefills, decodes, ruled, beside, (&[PLAIN], None))
}

/// [`asked_of`] over the encoders a case states (`encoders`): what each
/// says of itself, and the places of them the request's host states.
fn asked_among(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
    beside: Beside<'_>,
    encoders: (&[EncoderFacts], Option<&[u32]>),
) -> (Answer, usize) {
    let (prefills, decodes) = (stated(prefills), stated(decodes));
    let counts = [prefills.len(), decodes.len(), encoders.0.len()];
    let mut three = Three::of(counts, beside.sees.unwrap_or(|_, _| false));
    three.partners = partners(&prefills, &decodes);
    let mut taken = Vec::new();
    for (list, workers) in [&prefills, &decodes].into_iter().enumerate() {
        for (at, worker) in workers.iter().enumerate() {
            worker.set(three.workers[list][at].as_ref());
            if worker.taken {
                taken.push(three.id(list, at));
            }
        }
    }
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        asked.set(asked.get() + 1);
        taken.contains(&worker_id(worker))
    };
    let items = EncodeItems {
        keys: &KEYS,
        decode_images: beside.sees.is_some(),
        ..EncodeItems::default()
    };
    let (prefill_stalled, decode_stalled) = (masked(&prefills), masked(&decodes));
    let (prefill_held, decode_held) = (holds(&prefills), holds(&decodes));
    let mut lists = three.lists((encoders.0, beside.encode_classes));
    lists.encoders_places = encoders.1;
    lists.pair.prefill_stalled = &prefill_stalled;
    lists.pair.prefill_held = &prefill_held;
    lists.pair.decode_stalled = &decode_stalled;
    lists.pair.decode_held = &decode_held;
    let request = PlaceRequest {
        admits: if ruled { Some(&admits) } else { None },
        ..asking(lists, items)
    };
    let answer = shed_of(engine.place(&request));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    (answer, asked.get())
}

/// Every prefill of `prefills` hands to the one encoder, and the request's
/// decode takes whatever worker.
fn served(prefills: &str) -> Vec<u32> {
    vec![1; stated(prefills).len()]
}

/// Holds every row of `kind` for an encode request that carries an item:
/// one engine answers them all, each over lists of its own.
fn held(kind: Kind) {
    let engine = shedding();
    let asked = |row: &Row, ruled| {
        let beside = Beside {
            encode_classes: &served(row.prefills),
            sees: None,
        };
        asked_of(engine, row.prefills, row.decodes, ruled, beside)
    };
    let failed = failing(kind, asked);
    assert!(failed.is_empty(), "{failed:#?}");
}

/// An encode request's first decision that sheds it as saturated over a
/// list of its pair stands where the rule takes no worker of a list, in
/// the pair's two cases: the rule takes no prefill and the shed was made
/// over the prefills, or the rule takes a prefill and no decode and its
/// prefills let the shed stand.
///
/// The gateway's own selection answers its overload shed in each of these
/// rows, as for a pair: it judges the prefill list whole where the rule
/// takes none of it, the rule's prefills where none of them can take the
/// request, and else the decode list
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`,
/// `:2909-2936`, `admitted_pool` at `:1831-1852`, `failure_from` at
/// `:1710-1722`, on main `56773d28d`).
#[test]
fn an_encode_requests_saturated_shed_stands_where_the_rule_takes_no_worker_of_a_list() {
    held(Kind::Stands);
}

/// A rule that takes no worker of a list of an encode request's pair is
/// told by name in every other row, as for a pair.
#[test]
fn an_encode_requests_rule_that_takes_no_worker_of_a_list_is_told_where_no_shed_stands() {
    held(Kind::Told);
}

/// A rule that takes a worker of each list of an encode request's pair is
/// answered by the call's next decision, made among them.
#[test]
fn an_encode_requests_rule_that_takes_a_worker_of_each_list_is_answered_by_the_next_decision() {
    held(Kind::Among);
}

/// An encode request's rule is not asked behind a first decision that
/// sheds the request with no endpoint, as a pair's is not.
#[test]
fn an_encode_requests_rule_is_not_asked_behind_a_first_decision_that_sheds_with_no_endpoint() {
    held(Kind::NotAsked);
}

/// An encode request that states no rule keeps its first decision's shed.
#[test]
fn an_encode_request_with_no_rule_keeps_the_shed_of_its_first_decision() {
    held(Kind::Unruled);
}

/// A pair's decision that was handed a part of the prefill list says no
/// list of its shed, and the rule's shed is the answer as before. The
/// first prefill is over its thresholds. The second is under them and
/// hands to no encoder, so the pair's decision is handed the first alone
/// and sheds over it: under no rule that shed is the answer. The rule
/// takes no prefill: the request is shed under the rule's name, and with
/// both prefills served and over their thresholds it keeps the decision's
/// shed.
///
/// The gateway's own selection hands its pair selection one filter for
/// such a request, which refuses a prefill no available encoder pairs
/// with, and no rule of a route beside it
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2654`,
/// `:2663`, on main `56773d28d`). Under that filter alone it judges the
/// first prefill, which the filter takes, and answers its overload shed,
/// as the decision here does under no rule. A filter that takes neither
/// prefill, as that filter beside this rule takes neither, has the prefill
/// list judged whole
/// (`model_gateway/src/routers/common/placement.rs:1831-1852`): the second
/// prefill is under its thresholds, and the answer is that there is no
/// worker.
#[test]
fn a_part_of_a_list_says_no_list_of_its_shed() {
    let engine = shedding();
    let beside = |encode_classes| Beside {
        encode_classes,
        sees: None,
    };
    let part = beside(&[1, 2]);
    assert_eq!(asked_of(engine, "v+ o+", "o+", false, part), (SATURATED, 0));
    assert_eq!(asked_of(engine, "v- o-", "o+", true, part), (ADMITS, 2));
    let whole = beside(&[1, 1]);
    assert_eq!(asked_of(engine, "v- v-", "o+", true, whole), (SATURATED, 2));
}

/// So it is for a part of the decode list. The request's decode must take
/// images. The first decode takes them and is over its thresholds, and the
/// second takes none and is under them, so the pair's decision is handed
/// the first alone and sheds over it: under no rule that shed is the
/// answer. The rule takes the prefill and no decode: the request is shed
/// under the rule's name. With both decodes taking images and over their
/// thresholds it keeps the decision's shed.
///
/// The gateway's own selection hands its pair selection one filter for
/// such a request, which takes a decode that takes images where the
/// request's decode must, and no rule of a route beside it
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2654`,
/// `:2663`, on main `56773d28d`). Under that filter alone it judges the first
/// decode, which the filter takes, and answers its overload shed
/// (`model_gateway/src/routers/common/placement.rs:2909-2936`,
/// `:1831-1852`). A filter that takes no decode has the decode list judged
/// whole: the second decode is under its thresholds, and the answer is
/// that there is no worker.
#[test]
fn a_part_of_the_decode_list_says_no_list_of_its_shed() {
    let engine = shedding();
    let beside = |sees: fn(usize, usize) -> bool| Beside {
        encode_classes: &[1],
        sees: Some(sees),
    };
    let part = beside(|list, place| (list, place) == (1, 0));
    assert_eq!(asked_of(engine, "o+", "v+ o+", false, part), (SATURATED, 0));
    assert_eq!(asked_of(engine, "o+", "v- o-", true, part), (ADMITS, 3));
    let whole = beside(|list, _| list == 1);
    assert_eq!(asked_of(engine, "o+", "v- v-", true, whole), (SATURATED, 3));
}

/// A decision that was handed a part of one list says no list of the
/// other either. The second prefill hands to no encoder, so the pair's
/// decision is handed the first alone, beside the one decode, which is
/// over its thresholds: it sheds over the decodes. The rule takes the
/// second prefill and no decode. That prefill is ready and under its
/// thresholds, but no encoder takes the request's item for it, and the
/// decision made its shed for the first prefill alone: its word is taken
/// back, and the request is shed under the rule's name. So it is where
/// the rule takes no prefill: no word of the prefills stands in its
/// place. With both prefills served the word stands, and the shed with
/// it.
///
/// The gateway's own selection would hold such a request to the first
/// prefill by its filter, and a filter that takes no prefill, as that
/// filter beside this rule takes none, has the prefill list judged whole
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`,
/// `:1831-1852`, on main `56773d28d`): both prefills are under their
/// thresholds, and the answer is that there is no worker. That is read
/// from its lines and not run.
#[test]
fn a_part_of_the_prefill_list_takes_back_the_word_of_the_decodes_too() {
    let engine = shedding();
    let beside = |encode_classes| Beside {
        encode_classes,
        sees: None,
    };
    let part = beside(&[1, 2]);
    assert_eq!(asked_of(engine, "o- o+", "v-", true, part), (ADMITS, 3));
    assert_eq!(asked_of(engine, "o- o-", "v-", true, part), (ADMITS, 2));
    let whole = beside(&[1, 1]);
    assert_eq!(asked_of(engine, "o- o+", "v-", true, whole), (SATURATED, 3));
}

/// Places of the encoders alone ([`EncodeLists::encoders_places`]) hold an
/// encode request to no part of its pair's two lists. Two encoders of one
/// class, and the request held to the first. The one prefill is over its
/// thresholds and the rule takes no prefill: the pair's decision was
/// handed both lists whole, its word stands, and so does its shed.
///
/// Where the encoders at the places leave a prefill out, the pass hands
/// the decision a part of the prefill list and takes its word back, as
/// behind any cut. The second prefill hands to the second encoder alone,
/// which stands outside the places, and both prefills are over their
/// thresholds: the request is shed under the rule's name. With no places
/// stated both prefills are served, and the shed stands.
///
/// The gateway's own selection states no places, and asks no route's
/// rule of such a request
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2663`,
/// on main `56773d28d`): the state is a host's own.
#[test]
fn places_of_the_encoders_alone_leave_the_pairs_lists_whole() {
    let engine = shedding();
    let first: &[u32] = &[0];
    // The second encoder is of the class `second`, and the request is held
    // to the first encoder where `held`.
    let asked = |prefills: &str, hands: &[u32], second: u32, held: bool| {
        let encoders = [
            PLAIN,
            EncoderFacts {
                class: second,
                ..PLAIN
            },
        ];
        let beside = Beside {
            encode_classes: hands,
            sees: None,
        };
        let among = (&encoders[..], held.then_some(first));
        asked_among(engine, prefills, "o+", true, beside, among)
    };
    assert_eq!(asked("v-", &[1], 1, true), (SATURATED, 1));
    assert_eq!(asked("v- v-", &[1, 2], 2, true), (ADMITS, 2));
    assert_eq!(asked("v- v-", &[1, 2], 2, false), (SATURATED, 2));
}

/// So it is the other way round. The request's decode must take images
/// and the second decode takes none, so the pair's decision is handed the
/// first decode alone, beside the one prefill, which is over its
/// thresholds: it sheds over the prefills. The rule takes no prefill. The
/// decision was handed a part of the decode list, so its word is taken
/// back, and the request is shed under the rule's name. With both decodes
/// taking images the word stands, and the shed with it.
#[test]
fn a_part_of_the_decode_list_takes_back_the_word_of_the_prefills_too() {
    let engine = shedding();
    let beside = |sees: fn(usize, usize) -> bool| Beside {
        encode_classes: &[1],
        sees: Some(sees),
    };
    let part = beside(|list, place| (list, place) == (1, 0));
    assert_eq!(asked_of(engine, "v-", "o+ o+", true, part), (ADMITS, 1));
    let whole = beside(|list, _| list == 1);
    assert_eq!(asked_of(engine, "v-", "o+ o+", true, whole), (SATURATED, 1));
}

/// An encode request's decision made from inside another of its thread
/// keeps its word of a shed for its call, as a pair's does
/// ([`asked_inside`]): its prefills are over their thresholds under
/// bindings that shed, its rule takes none of them, and it keeps its
/// decision's shed though the thread's buffers were lent when it was made.
#[test]
fn an_encode_decision_inside_another_keeps_its_word_of_a_shed() {
    let engine = shedding();
    let beside = Beside {
        encode_classes: &[1, 1],
        sees: None,
    };
    for (lent, answered) in asked_inside(|| asked_of(engine, "v- v-", "o+", true, beside)) {
        assert_eq!(answered, (SATURATED, 2), "inside a decision: {lent}");
    }
    // The decodes' word is kept the same way.
    let one = Beside {
        encode_classes: &[1],
        sees: None,
    };
    for (lent, answered) in asked_inside(|| asked_of(engine, "o+", "v-", true, one)) {
        assert_eq!(answered, (SATURATED, 2), "the decodes; inside: {lent}");
    }
}
