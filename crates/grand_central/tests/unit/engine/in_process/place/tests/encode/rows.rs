//! The rows of a pair's request under a document that sheds a list wholly
//! over its thresholds ([`ROWS`]), held for an encode request. One encoder
//! stands open and serves every prefill.
//!
//! A request that carries no item and holds its decode to no images is its
//! pair's alone: the encode side lays no pass for it, and it answers each
//! row as the pair does, by the pair's own routine ([`again`]).
//!
//! A request that carries an item has the pass laid. Its pair's decision
//! is told so (`PairPools::part`) and asks no question of its shed, its
//! rule is asked behind a shed as saturated alone, and no decision more
//! is held to its rule's prefills. So it answers each row as the row
//! answered before the question stood ([`BEFORE`]).
//!
//! The gateway's own selection for such a request makes its pair by its
//! pair selection, under one filter of its own, which takes the workers
//! an available encoder is fit for and asks no route's rule
//! (`placed_encode_prefill_decode_workers` in
//! `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`).
//! With the one encoder here serving every prefill that filter takes every
//! worker. So no state of that selection stands behind a row whose rule
//! refuses a worker: such a row is held for a host's own rule. What the
//! cases of a part of a list say of that selection was read from those
//! lines and not run.
//!
//! [`again`]: super::super::super::pair::again
//! [`ROWS`]: super::super::pair::rows::ROWS

use super::{
    super::pair::rows::{
        asked_inside, failing, holds, masked, partners, stated, Answer, Kind, Row, ADMITS, HEALTHY,
        PARTNERED, ROWS, SATURATED,
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
    shed_said(answered_among(
        engine, prefills, decodes, ruled, beside, encoders,
    ))
}

/// The shed of what a call answered, beside the times its rule was asked.
fn shed_said(answered: (Placed, usize)) -> (Answer, usize) {
    match answered {
        (
            Err(Unplaced::Shed {
                reason, emptied_by, ..
            }),
            asked,
        ) => ((reason, emptied_by), asked),
        other => panic!("a shed, and not {other:?}"),
    }
}

/// What a call answers an encode request: the places of the workers it is
/// placed on, of the prefills, of the decodes and of the encoders, or the
/// call's whole refusal.
type Placed = Result<(usize, usize, Vec<usize>), Unplaced>;

/// What `engine` answers an encode request of one item over a row's two
/// lists and the encoders a case states, and the times its rule was asked.
fn answered_among(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
    beside: Beside<'_>,
    encoders: (&[EncoderFacts], Option<&[u32]>),
) -> (Placed, usize) {
    answered_with(engine, (prefills, decodes), ruled, beside, encoders, &KEYS)
}

/// [`answered_among`] for a request that carries the items of `keys`: the
/// one item, or none.
fn answered_with(
    engine: &'static InProcess,
    lists: (&str, &str),
    ruled: bool,
    beside: Beside<'_>,
    encoders: (&[EncoderFacts], Option<&[u32]>),
    keys: &[Identity<'static>],
) -> (Placed, usize) {
    let items = EncodeItems {
        keys,
        decode_images: beside.sees.is_some(),
        ..EncodeItems::default()
    };
    let stating = (items, MediaCounts::NONE);
    answered_stating(engine, lists, ruled, beside, encoders, stating)
}

/// [`answered_with`] for a request that states `items` beside the media
/// it holds.
fn answered_stating(
    engine: &'static InProcess,
    (prefills, decodes): (&str, &str),
    ruled: bool,
    beside: Beside<'_>,
    encoders: (&[EncoderFacts], Option<&[u32]>),
    (items, media): (EncodeItems<'_>, MediaCounts),
) -> (Placed, usize) {
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
        media,
        ..asking(lists, items)
    };
    let placed = engine.place(&request);
    let answer = placed.map(|placement| three.places(&placement));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    (answer, asked.get())
}

/// Every prefill of `prefills` hands to the one encoder, and the request's
/// decode takes whatever worker.
fn served(prefills: &str) -> Vec<u32> {
    vec![1; stated(prefills).len()]
}

/// The shed `engine` answers an encode request over a row's two lists and
/// the one open encoder, and the times its rule was asked: a request that
/// carries the items of `keys`, the one item or none.
fn asked_in(
    engine: &'static InProcess,
    row: &Row,
    ruled: bool,
    keys: &[Identity<'static>],
) -> (Answer, usize) {
    let beside = Beside {
        encode_classes: &served(row.prefills),
        sees: None,
    };
    let lists = (row.prefills, row.decodes);
    shed_said(answered_with(
        engine,
        lists,
        ruled,
        beside,
        (&[PLAIN], None),
        keys,
    ))
}

/// Holds every row of `kind` for an encode request that carries no item,
/// which is its pair's alone: one engine answers them all, each over lists
/// of its own.
fn held(kind: Kind) {
    let engine = shedding();
    let failed = failing(kind, |row, ruled| asked_in(engine, row, ruled, &[]));
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The rows an encode request that carries an item answers otherwise than
/// the pair's row says, each by its two lists and by whether the request
/// states a rule: the shed it is answered, and the times its rule is
/// asked. Every other row it answers as the row says. Each line is a run
/// of the engine over the row.
///
/// The decision of such a request asks no question: the stage that ended
/// its pass names its shed, as before the question stood. Its rule is not
/// asked behind a shed with no endpoint. Behind a shed as saturated the
/// rule is told by name where the decision said no list. No decision more
/// is held to its rule's prefills: a rule that takes only a prefill over
/// its thresholds and no decode is told by name, in the lines of the two
/// rows such a decision judges for a pair ([`Kind::Judged`]). Behind a
/// shed made over the prefills, a rule that takes prefills none of which
/// pairs with a decode is asked of the decodes too: the table's last two
/// lines, which part from their rows by that count alone.
///
/// The gateway's own selection answers the rows that state no rule as the
/// pair's row says: its filter for such a request takes every worker
/// here. The eight lines of them part from it. They stand until the
/// engine is told which workers that filter leaves out: where it leaves
/// one out, the answer before the question is that selection's
/// ([`an_items_shed_is_named_by_its_pass_beside_a_decode_that_pairs_with_nobody`]).
const BEFORE: [(&str, &str, bool, Answer, usize); 25] = [
    ("o+", "v+ o+x", false, SATURATED, 0),
    ("o+", "v- o-x", true, ADMITS, 3),
    ("o+", "v- s-x", true, ADMITS, 3),
    ("o+", "v- m-x", true, ADMITS, 3),
    ("v- s-", "u+", true, ADMITS, 2),
    ("u- v-x", "o+", true, ADMITS, 2),
    ("o+", "v+ s+x", false, SATURATED, 0),
    ("v+ s+", "u+", false, SATURATED, 0),
    ("u+ v+x", "o+", false, SATURATED, 0),
    ("v+", "o+x", false, SATURATED, 0),
    ("o+", "w+", false, HEALTHY, 0),
    ("w+", "o+", false, HEALTHY, 0),
    ("v+ o+x", "o+", false, PARTNERED, 0),
    ("o- v+", "v- o-x", true, ADMITS, 4),
    ("o- v+", "o-", true, ADMITS, 5),
    ("o+", "v+ o+x", true, SATURATED, 3),
    ("v+ s+", "u+", true, SATURATED, 3),
    ("o+", "o- v+x", true, PARTNERED, 5),
    ("o+", "u+", true, HEALTHY, 0),
    ("o- v+", "u+", true, HEALTHY, 0),
    ("w-", "o+", true, HEALTHY, 0),
    ("o+", "w-", true, HEALTHY, 0),
    ("v- o-x", "o+", true, PARTNERED, 0),
    ("v- v+x", "o-", true, SATURATED, 3),
    ("v- v+x", "o+", true, SATURATED, 3),
];

/// Whether a line of [`BEFORE`] is of `row`: its two lists, and whether
/// its request states a rule.
fn is_of(line: &(&str, &str, bool, Answer, usize), row: &Row) -> bool {
    let ruled = row.kind != Kind::Unruled;
    (line.0, line.1, line.2) == (row.prefills, row.decodes, ruled)
}

/// An encode request that carries an item answers each row as the row
/// answered before a pair's decision asked its one question: by the
/// row's line of [`BEFORE`], and as the pair's row says where the table
/// holds no line of it. Each line of the table is of one row, and parts
/// from it in the shed or in the times the rule is asked.
///
/// So the two requests part in these rows alone. One with no item is its
/// pair's and answers by the question. One with an item has the pass
/// laid, and keeps the answer it had.
#[test]
fn an_encode_request_with_an_item_answers_each_row_as_before_the_question() {
    let engine = shedding();
    for line in &BEFORE {
        let rows: Vec<&Row> = ROWS.iter().filter(|row| is_of(line, row)).collect();
        assert_eq!(rows.len(), 1, "one row of {line:?}");
        let said = (rows[0].answer, rows[0].asked);
        assert_ne!(said, (line.3, line.4), "{}", rows[0].name);
    }
    let failed = ROWS.iter().filter_map(|row| {
        let line = BEFORE.iter().find(|line| is_of(line, row));
        let said = line.map_or((row.answer, row.asked), |line| (line.3, line.4));
        let answered = asked_in(engine, row, row.kind != Kind::Unruled, &KEYS);
        (answered != said).then(|| format!("{}: {answered:?}, and not {said:?}", row.name))
    });
    let failed: Vec<String> = failed.collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

/// A rule that takes prefills none of which pairs with a decode, and a
/// decode, behind a first decision that shed the request over its
/// prefills. The first prefill is over its thresholds, pairs with the
/// decode, and is refused by the rule. The second is not ready, pairs
/// with nobody, and is the one prefill the rule takes.
///
/// A request with no item is its pair's. Such a rule is answered as one
/// that takes no prefill: the first decision's shed stands, and the
/// decodes are not asked. A request the pass is laid for is answered as
/// before the question stood: by an item, or by a decode held to the
/// ones that take images, which its demand does and references beside
/// an image do. The decodes are asked too, and the call's next
/// decision, made among the rule's workers, sheds it with no endpoint
/// under the health stage. No row of the table holds this state. In the
/// two rows of such a rule the rule's one prefill is over its
/// thresholds: the first shed stands where the rule takes no decode,
/// for the prefills' side, and where it takes one the next decision
/// sheds as the first did.
///
/// The gateway's own selection holds no prefill that pairs with no decode
/// (`PdPairIndex::build` in
/// `model_gateway/src/routers/common/pair_index.rs`). Under a filter that
/// takes the same workers it would judge its prefill list whole, the
/// first prefill alone, and answer its overload shed (`admitted_pool` and
/// `pair_pass` in `model_gateway/src/routers/common/placement.rs`). So
/// the pair's answer is of that selection's kind here, and the laid
/// request's is not. That is read from those lines and not run.
#[test]
fn a_laid_requests_rule_whose_prefills_pair_with_nobody_is_asked_of_the_decodes() {
    let engine = shedding();
    let lists = ("v- u+x", "o+");
    let asked = |keys: &[Identity<'static>], sees: Option<fn(usize, usize) -> bool>| {
        let beside = Beside {
            encode_classes: &served(lists.0),
            sees,
        };
        shed_said(answered_with(
            engine,
            lists,
            true,
            beside,
            (&[PLAIN], None),
            keys,
        ))
    };
    assert_eq!(asked(&[], None), (SATURATED, 2));
    assert_eq!(asked(&KEYS, None), (HEALTHY, 3));
    assert_eq!(asked(&[], Some(|_, _| true)), (HEALTHY, 3));
    // References beside an image lay the pass too, with no item and no
    // demand of the decode.
    let classes = served(lists.0);
    let beside = Beside {
        encode_classes: &classes,
        sees: None,
    };
    let by_references = (
        EncodeItems {
            references: true,
            ..EncodeItems::default()
        },
        MediaCounts {
            images: 1,
            ..MediaCounts::NONE
        },
    );
    let answered = answered_stating(engine, lists, true, beside, (&[PLAIN], None), by_references);
    assert_eq!(shed_said(answered), (HEALTHY, 3));
    // Where the rule takes no decode the first shed stands, for the
    // prefills' side.
    let stood = answered_with(
        engine,
        ("v- v+x", "o-"),
        true,
        beside,
        (&[PLAIN], None),
        &KEYS,
    );
    assert!(
        matches!(
            stood.0,
            Err(Unplaced::Shed {
                reason: ShedReason::Saturated,
                side: Some(PairSide::Prefill),
                ..
            })
        ),
        "{stood:?}"
    );
}

/// The first decision of an encode request with no item that sheds it as
/// saturated over a list of its pair stands where the rule takes no worker
/// of a list, in the pair's two cases: the rule takes no prefill and the
/// shed was made over the prefills, or the rule takes a prefill and no
/// decode and its prefills let the shed stand.
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
/// told by name where no shed of the first decision is the answer, as for
/// a pair: a request with no item.
#[test]
fn an_encode_requests_rule_that_takes_no_worker_of_a_list_is_told_where_no_shed_stands() {
    held(Kind::Told);
}

/// A rule that takes a worker of each list of an encode request's pair is
/// answered by the call's next decision, made among them: a request with
/// no item.
#[test]
fn an_encode_requests_rule_that_takes_a_worker_of_each_list_is_answered_by_the_next_decision() {
    held(Kind::Among);
}

/// The first decision of an encode request with no item that sheds it
/// with no endpoint stands where the rule takes no worker of a list of
/// its pair, as a pair's does.
#[test]
fn an_encode_requests_shed_with_no_endpoint_stands_where_the_rule_takes_no_worker_of_a_list() {
    held(Kind::Keeps);
}

/// An encode request with no item is its pair's. Its rule that takes only
/// prefills over their thresholds and no decode is judged by one decision
/// more, as a pair's is: held to the rule's prefills with no decode lent,
/// it sheds the request as saturated over them, and that shed is the
/// answer, for the prefill side.
#[test]
fn a_rule_of_prefills_over_their_thresholds_alone_is_judged_for_a_request_with_no_item() {
    held(Kind::Judged);
    let engine = shedding();
    for row in ROWS.iter().filter(|row| row.kind == Kind::Judged) {
        let beside = Beside {
            encode_classes: &served(row.prefills),
            sees: None,
        };
        let lists = (row.prefills, row.decodes);
        let lone = (&[PLAIN][..], None);
        let (placed, _) = answered_with(engine, lists, true, beside, lone, &[]);
        assert_eq!(sided(&placed).1, Some(PairSide::Prefill), "{}", row.name);
    }
}

/// A request the encode pass is laid for is asked no such decision: it
/// keeps the answer it had. The first prefill is open and the second over
/// its thresholds; the rule takes the second alone, and no decode. With no
/// item the request is its pair's, and one decision more sheds it as
/// saturated over the rule's prefill. With an item it is shed under the
/// rule's name as before, where the encoder serves every prefill and
/// where a third prefill hands to no encoder.
#[test]
fn a_laid_request_is_asked_no_decision_more_behind_a_rule_of_prefills_alone() {
    let engine = shedding();
    let asked = |prefills: &str, encode_classes: &[u32], keys: &[Identity<'static>]| {
        let beside = Beside {
            encode_classes,
            sees: None,
        };
        let (lists, lone) = ((prefills, "o-"), (&[PLAIN][..], None));
        shed_said(answered_with(engine, lists, true, beside, lone, keys)).0
    };
    assert_eq!(asked("o- v+", &[1, 1], &[]), SATURATED);
    assert_eq!(asked("o- v+", &[1, 1], &KEYS), ADMITS);
    assert_eq!(asked("o- v+ o-", &[1, 1, 2], &KEYS), ADMITS);
}

/// The rule of an encode request with no item is asked behind a first
/// decision that sheds it with no endpoint, as a pair's is, and the call's
/// next decision is made among the rule's workers. The one partner is
/// under a liveness doubt beside an idle decode that pairs with no
/// prefill, and the rule takes the prefill and the partner: among those
/// alone the pair's default crosses the doubt, and the request is placed
/// on the two. Under no rule the first decision's shed is the answer.
///
/// A request with an item keeps that shed under the rule too. Its pass is
/// laid, so its shed with no endpoint goes to no rule, and the rule is
/// not asked: the answer it had before the question stood.
#[test]
fn an_encode_requests_rule_asked_behind_a_shed_with_no_endpoint_pairs_across_a_doubt() {
    let engine = shedding();
    let beside = Beside {
        encode_classes: &[1],
        sees: None,
    };
    let (lists, one) = (("o+", "s+ o-x"), (&[PLAIN][..], None));
    let placed = answered_with(engine, lists, true, beside, one, &[]);
    assert_eq!(placed, (Ok((0, 0, vec![])), 5));
    let unruled = shed_said(answered_with(engine, lists, false, beside, one, &[]));
    assert_eq!(unruled, (HEALTHY, 0));
    let with_an_item = asked_of(engine, "o+", "s+ o-x", true, beside);
    assert_eq!(with_an_item, (HEALTHY, 0));
}

/// An encode request with no item that states no rule keeps its first
/// decision's shed.
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

/// Behind the encode pass's cut of the decodes the pair's decision asks
/// no question of its shed, as for every request the pass is laid for.
/// The request's item needs a decode that takes images, and one of the
/// two decodes takes none: the pass hands the pair the other alone. The
/// first prefill pairs with the decode that was left out, so it pairs
/// with nobody there. The second pairs with the decode that takes images
/// and is over its thresholds.
///
/// The request is shed with no endpoint under `partnered`: the reason and
/// the stage it had before the question stood. Asked over that part, the
/// question would count the one prefill that pairs there and shed the
/// request as saturated.
///
/// The shed names the prefill side. A prefill of the list pairs with a
/// decode the pass handed, and the one that stands pairs with none of
/// them.
///
/// The gateway's own selection pairs by its pool's index
/// (`PdPairIndex::build` in
/// `model_gateway/src/routers/common/pair_index.rs`). It takes the
/// decodes that take the images in its filter
/// (`placed_encode_prefill_decode_workers` in
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`).
/// The first prefill is eligible there with no decode open for it, and
/// that selection answers that it has no worker, for the decode's role:
/// by a run of it over this state, which this crate holds no case of. The
/// two answers meet where the pass hands the pair the decodes whole.
#[test]
fn an_encode_requests_shed_behind_the_cut_of_the_decodes_asks_no_question() {
    let engine = shedding();
    // The second decode alone takes images.
    let mut three = Three::of([2, 2, 1], |list, place| (list, place) == (1, 1));
    // Two pairs apart: each prefill pairs with the decode at its place.
    three.partners = PartnerClasses::number(2, [&[0][..], &[1][..]]);
    three.workers[0][1].set_overloaded(true);
    let items = EncodeItems {
        keys: &KEYS,
        decode_images: true,
        ..EncodeItems::default()
    };
    let lists = three.lists((&[PLAIN], &[1, 1]));
    let placed = engine.place(&asking(lists, items));
    let placed = placed.map(|placement| three.places(&placement));
    assert!(
        matches!(
            placed,
            Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "partnered",
                side: Some(PairSide::Prefill),
                ..
            })
        ),
        "{placed:?}"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request that states a hard demand is answered as its pair
/// is: as it was before a pair's decision asked its one question. The
/// demand hands the pair's decision a part of each list, the workers that
/// meet it. Every worker of the two states meets the demand, the one
/// encoder too, so the same lists answer by the question where the
/// request states no demand and carries no item.
///
/// The decision asks no question. The prefill's one partner is not ready
/// beside a decode over its thresholds that pairs with nobody. By the
/// question every ready decode is over its thresholds, and the shed is
/// saturated. Behind the demand the health stage names it.
///
/// A pass the overload veto ended keeps its word. The partner is over its
/// thresholds beside a decode with room that pairs with nobody and stands
/// under a liveness doubt. By the question the request has no endpoint,
/// under the health stage. Behind the demand the shed stays saturated.
///
/// A request that carries an item is answered so under no demand too: its
/// pass is laid, and its decision asks no question either.
#[test]
fn an_encode_request_that_states_a_hard_demand_is_answered_as_before_the_question() {
    let engine = shedding();
    let wire = Demands {
        runtime: 2,
        ..Demands::default()
    };
    let shed = |decodes: &str, demands: Demands, keys: &[Identity<'static>]| -> Answer {
        let (prefills, decodes) = (stated("o+"), stated(decodes));
        let mut three = Three::of([prefills.len(), decodes.len(), 1], |_, _| false);
        // Every worker of the three lists states the demand's runtime.
        three.published = [0, 1, 2].map(|list| {
            PublishedList::with_facts(Arc::clone(&three.workers[list]), |_| {
                let mut facts = FixedFacts::default();
                facts.demands.runtime = 2;
                facts
            })
        });
        three.partners = partners(&prefills, &decodes);
        for (at, worker) in decodes.iter().enumerate() {
            worker.set(three.workers[1][at].as_ref());
        }
        let items = EncodeItems {
            keys,
            ..EncodeItems::default()
        };
        let request = PlaceRequest {
            demands,
            ..asking(three.lists((&[PLAIN], &[1])), items)
        };
        let placed = engine.place(&request);
        match placed.map(|placement| three.places(&placement)) {
            Err(Unplaced::Shed {
                reason, emptied_by, ..
            }) => (reason, emptied_by),
            other => panic!("a shed, and not {other:?}"),
        }
    };
    let (none, one): (&[Identity<'static>], &[Identity<'static>]) = (&[], &KEYS);
    // No demand and no item: the pair's alone, answered by the question.
    assert_eq!(shed("u+ v+x", Demands::default(), none), SATURATED);
    assert_eq!(shed("v+ s+x", Demands::default(), none), HEALTHY);
    // Behind the demand, and with an item under either: as before it.
    for (demands, keys) in [(wire, none), (wire, one), (Demands::default(), one)] {
        assert_eq!(shed("u+ v+x", demands, keys), HEALTHY);
        assert_eq!(shed("v+ s+x", demands, keys), SATURATED);
    }
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The side of a call's shed, beside its reason and the stage it names.
fn sided(placed: &Placed) -> (Answer, Option<PairSide>) {
    match placed {
        Err(Unplaced::Shed {
            reason,
            emptied_by,
            side,
            ..
        }) => ((*reason, *emptied_by), *side),
        other => panic!("a shed, and not {other:?}"),
    }
}

/// A request with an item keeps the stage that ended its pass beside a
/// decode that pairs with nobody. The prefill's one partner is over its
/// thresholds beside an idle decode that pairs with no prefill: the
/// overload veto ended the pass, and the request is shed as saturated. A
/// request with no item is asked the question there. The idle decode
/// keeps the decode list from being wholly over its thresholds, and that
/// request has no endpoint.
///
/// The other way round, the one partner is not ready beside a decode over
/// its thresholds that pairs with no prefill. The health stage ended the
/// pass, and the request has no endpoint. By the question it is shed as
/// saturated.
///
/// The gateway's own selection answers an item's request as here in a
/// fleet of two runtimes, where the decode apart is of a runtime no
/// available encoder is of. Its filter for such a request leaves that
/// decode out (`placed_encode_prefill_decode_workers` in
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`),
/// and it judges the decodes its filter takes (`admitted_pool` in
/// `model_gateway/src/routers/common/placement.rs`): the one partner.
/// Where the decode apart is of the encoder's runtime the filter takes
/// it, and that selection answers as the question does. The lists the
/// engine is handed say one state for the two. That is read from those
/// lines and not run.
#[test]
fn an_items_shed_is_named_by_its_pass_beside_a_decode_that_pairs_with_nobody() {
    let engine = shedding();
    let beside = Beside {
        encode_classes: &[1],
        sees: None,
    };
    let one = (&[PLAIN][..], None);
    let alone = |lists| shed_said(answered_with(engine, lists, false, beside, one, &[]));
    assert_eq!(asked_of(engine, "o+", "v+ o+x", false, beside).0, SATURATED);
    assert_eq!(alone(("o+", "v+ o+x")).0, PARTNERED);
    assert_eq!(asked_of(engine, "o+", "u+ v+x", false, beside).0, HEALTHY);
    assert_eq!(alone(("o+", "u+ v+x")).0, SATURATED);
}

/// A request with an item keeps its shed as saturated behind the encode
/// pass's cut of the prefills. Both prefills are over their thresholds.
/// The first hands to the open encoder and pairs with no decode. The
/// second pairs with the one decode and hands to an encoder that is
/// registered and cannot be sent to: the pass hands the pair the first
/// alone. The overload veto ended the pass, and the request is shed as
/// saturated, as before the question stood. Asked over that part, the
/// question finds no prefill that pairs, and sheds the request with no
/// endpoint.
///
/// The gateway's own selection answers its overload shed there, for the
/// prefill's role. Its filter for such a request takes no prefill of its
/// pool. The pool does not hold the first, which pairs with no decode
/// (`PdPairIndex::build` in
/// `model_gateway/src/routers/common/pair_index.rs`), and no available
/// encoder is fit for the second (`placed_encode_prefill_decode_workers`
/// in `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`).
/// It then judges its prefill list whole (`admitted_pool` in
/// `model_gateway/src/routers/common/placement.rs`): the second prefill,
/// over its thresholds. That is read from those lines and not run.
///
/// The shed here names the decode side, where that selection refuses for
/// the prefill's role. The side reads the lists the decision was handed,
/// and no prefill of them pairs with a decode. With both encoders open
/// the pass hands the pair both prefills, and the shed names the prefill
/// side.
#[test]
fn an_items_shed_behind_the_cut_of_the_prefills_stays_saturated() {
    let engine = shedding();
    // Whether the second encoder can be sent to.
    let shed = |second_open: bool| {
        let mut three = Three::of([2, 1, 2], |_, _| false);
        // The second prefill alone pairs with the decode.
        three.partners = PartnerClasses::number(1, [&[][..], &[0][..]]);
        three.workers[0][0].set_overloaded(true);
        three.workers[0][1].set_overloaded(true);
        if !second_open {
            three.workers[2][1].set_status(WorkerStatus::NotReady);
        }
        let encoders = [PLAIN, EncoderFacts { class: 2, ..PLAIN }];
        let items = EncodeItems {
            keys: &KEYS,
            ..EncodeItems::default()
        };
        let placed = engine.place(&asking(three.lists((&encoders, &[1, 2])), items));
        sided(&placed.map(|placement| three.places(&placement)))
    };
    assert_eq!(shed(false), (SATURATED, Some(PairSide::Decode)));
    assert_eq!(shed(true), (SATURATED, Some(PairSide::Prefill)));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule of a request the encode pass is laid for is not asked behind
/// a shed with no endpoint, behind the pass's cut of the decodes as
/// elsewhere. The request's decode must take images, and the third decode
/// takes none: the pass hands the pair the first two. The prefill's one
/// partner is under a liveness doubt, beside an idle decode that pairs
/// with no prefill. The request has no endpoint, under the health stage
/// and for the decode side. The rule takes the prefill and the partner
/// and is asked of neither: the shed is the call's answer. A decision
/// among the rule's two workers would cross the doubt and place the
/// request. So it is with an item and with none: the image need alone
/// lays the pass.
///
/// The gateway's own selection asks no route's rule of such a request:
/// its own filter stands in the rule's place
/// (`placed_encode_prefill_decode_workers` in
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs`).
/// That filter takes the idle decode, so the decodes it judges are not
/// all under a veto, and its default crosses no doubt (`relaxed_pair` in
/// `model_gateway/src/routers/common/placement.rs`). It answers that it
/// has no worker, for the decode's role. That is read from those lines
/// and not run.
#[test]
fn a_laid_requests_rule_is_not_asked_behind_a_shed_with_no_endpoint_behind_the_cut_of_the_decodes()
{
    let engine = shedding();
    let beside = Beside {
        encode_classes: &[1],
        sees: Some(|list, place| list == 1 && place < 2),
    };
    let (lists, one) = (("o+", "s+ o-x o-x"), (&[PLAIN][..], None));
    for keys in [&KEYS[..], &[]] {
        for ruled in [true, false] {
            let (placed, asked) = answered_with(engine, lists, ruled, beside, one, keys);
            let said = (sided(&placed), asked);
            let items = keys.len();
            assert_eq!(
                said,
                ((HEALTHY, Some(PairSide::Decode)), 0),
                "{items} item(s), a rule: {ruled}"
            );
        }
    }
}
