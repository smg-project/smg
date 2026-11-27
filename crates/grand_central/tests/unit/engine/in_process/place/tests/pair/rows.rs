//! The rows of a pair's request under a document that sheds a list wholly
//! over its thresholds: what the call answers where its first decision
//! sheds the request, as saturated or with no endpoint, or picks workers
//! the rule refuses, and the route's rule takes all, some or none of a
//! list ([`again`]).
//!
//! A row is a prefill list and a decode list. A worker is written by its
//! state and by the rule's word of it. `o` is ready and under its
//! thresholds, `v` is ready and over them, and `u` is not ready. `s` and
//! `m` are ready, under their thresholds and under a liveness doubt: the
//! worker's own for `s`, and a standing veto of its host beside the list
//! for `m`. `w` is ready, over its thresholds and under a doubt of its
//! own. `h` is ready and under its thresholds, under a hold of its host
//! beside the list, which no rule lifts, and `k` is such a worker over its
//! thresholds. `+` says that the rule takes the worker, and `-` that it
//! refuses it. A worker marked `x` pairs with nobody, and every other
//! prefill pairs with every other decode.
//!
//! The gateway's own pair selection judges one list where it finds no pair
//! (`pair_pass`, `model_gateway/src/routers/common/placement.rs:2763`, on
//! main `56773d28d`): the prefills where no prefill the rule takes can take
//! the request (`:2866-2884`), and else the decodes (`:2909-2936`). It
//! judges that list over the rule's workers of it, and whole where the rule
//! takes none of it (`admitted_pool`, `:1831-1852`). Under its shed option
//! it answers its overload shed where the ready workers it judged are all
//! over their thresholds, and that it has no worker else (`failure_from`,
//! `:1710-1722`). A worker under a liveness doubt is ready in that count,
//! and cannot take a request
//! (`model_gateway/src/routers/common/overload.rs:58-97`,
//! `crates/worker/src/identity/basic.rs:444-449`). One behind an open
//! breaker is not ready there, as one under a hold is not here. It holds
//! the prefills that pair with a decode and no other
//! (`model_gateway/src/routers/common/pair_index.rs:158-161`).
//! Each row's answer here is that selection's by its kind: a shed as
//! saturated for its overload shed, and a shed with no endpoint for no
//! worker.
//!
//! [`again`]: super::super::super::pair::again

use smg_worker::runtime::StallReason;

use super::*;
use crate::engine::in_process::pair::BUFFERS;

/// A shed's reason and the stage it names.
pub(in super::super) type Answer = (ShedReason, &'static str);

/// A list wholly over its thresholds: the first decision's shed, or the
/// shed of a decision held to the rule's workers.
pub(in super::super) const SATURATED: Answer = (ShedReason::Saturated, "not-overloaded");
/// The rule takes no worker of a list.
pub(in super::super) const ADMITS: Answer = (ShedReason::NoEndpoints, "admits");
/// The workers a decision's pass was left are not ready, or the worker
/// with room of the list it judged is under a liveness doubt that nothing
/// lifts for it.
pub(in super::super) const HEALTHY: Answer = (ShedReason::NoEndpoints, "healthy");
/// No prefill that stands pairs with a decode at the request's places, or
/// the worker with room of the list a decision judged pairs with nobody
/// that stands.
pub(in super::super) const PARTNERED: Answer = (ShedReason::NoEndpoints, "partnered");

/// What a row shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Kind {
    /// The rule takes no worker of a list, and the first decision's shed
    /// as saturated is the call's answer.
    Stands,
    /// The rule takes no worker of a list, behind a first decision that
    /// picked or whose shed as saturated does not stand: the request is
    /// shed under the rule's name.
    Told,
    /// The rule takes no worker of a list, and the first decision's shed
    /// with no endpoint is the call's answer.
    Keeps,
    /// The rule takes a worker of each list, and a decision among them
    /// answers.
    Among,
    /// The rule takes only prefills over their thresholds and no decode,
    /// behind a first decision that picked or that shed the request with
    /// no endpoint: one decision more, held to the rule's prefills, sheds
    /// the request as saturated.
    Judged,
    /// The request states no rule.
    Unruled,
}

/// One row: its two lists, whether the request states a rule, the call's
/// answer, and the times the rule is asked.
#[derive(Debug)]
pub(in super::super) struct Row {
    pub(in super::super) name: &'static str,
    pub(in super::super) prefills: &'static str,
    pub(in super::super) decodes: &'static str,
    pub(in super::super) kind: Kind,
    pub(in super::super) answer: Answer,
    pub(in super::super) asked: usize,
}

/// A row: its answer is the gateway's own selection's by its kind.
const fn row(
    name: &'static str,
    prefills: &'static str,
    decodes: &'static str,
    kind: Kind,
    answer: Answer,
    asked: usize,
) -> Row {
    Row {
        name,
        prefills,
        decodes,
        kind,
        answer,
        asked,
    }
}

/// The rows. The gateway's own selection answers by the row's kind in
/// each. Each of its answers was run, by a run of that selection over the
/// row's state which this crate holds no case of: a worker `m` is an `s`
/// there, since that selection has one liveness doubt, the worker's own.
/// It was read and not run for the four rows that state a hold, which
/// say so.
pub(in super::super) const ROWS: [Row; 58] = [
    // The first decision sheds over the prefills.
    row(
        "the prefills over; one of each taken",
        "v+ v-",
        "o+",
        Kind::Among,
        SATURATED,
        3,
    ),
    row(
        "the prefills over; no prefill taken",
        "v- v-",
        "o+",
        Kind::Stands,
        SATURATED,
        2,
    ),
    row(
        "the prefills over; a prefill, no decode",
        "v+ v-",
        "o-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the prefills over; nobody taken",
        "v- v-",
        "o-",
        Kind::Stands,
        SATURATED,
        2,
    ),
    row(
        "the prefills over; a prefill that is not ready alone, no decode",
        "u+ v-",
        "o-",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the prefills over; a prefill that is not ready alone, and a decode",
        "u+ v-",
        "o+",
        Kind::Among,
        HEALTHY,
        3,
    ),
    // The first decision sheds over the decodes.
    row(
        "the decodes over; one of each taken",
        "o+",
        "v+ v-",
        Kind::Among,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; no prefill taken",
        "o-",
        "v+",
        Kind::Told,
        ADMITS,
        1,
    ),
    row(
        "the decodes over; an open prefill, no decode",
        "o+",
        "v- v-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; an over prefill alone, no decode",
        "o- v+",
        "v-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; a prefill that is not ready alone, no decode",
        "o- u+",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    // The first decision picks, and the rule refuses its picks. Where the
    // rule takes only a prefill over its thresholds and no decode, no
    // decision judged the rule's prefills: one more does.
    row(
        "a pick refused; an over prefill alone, no decode",
        "o- v+",
        "o-",
        Kind::Judged,
        SATURATED,
        5,
    ),
    row(
        "a pick refused; an over prefill alone, and a decode",
        "o- v+",
        "o+",
        Kind::Among,
        SATURATED,
        5,
    ),
    row(
        "a pick refused; an open prefill, no decode",
        "o+",
        "o-",
        Kind::Told,
        ADMITS,
        4,
    ),
    row(
        "a pick refused; no prefill taken",
        "o-",
        "o+",
        Kind::Told,
        ADMITS,
        3,
    ),
    // Both lists are over their thresholds: the shed is the prefills'.
    row(
        "both over; no prefill taken",
        "v-",
        "v+",
        Kind::Stands,
        SATURATED,
        1,
    ),
    row(
        "both over; a prefill, no decode",
        "v+",
        "v-",
        Kind::Stands,
        SATURATED,
        2,
    ),
    row(
        "the decodes over; no rule",
        "o+",
        "v+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    row(
        "the prefills over; no rule",
        "v+",
        "o+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    // The partners of the one prefill are over their thresholds beside an
    // idle decode that pairs with no prefill: the decode list is not wholly
    // over its thresholds, and the request has no endpoint.
    row(
        "the partners over beside an idle decode apart; no rule",
        "o+",
        "v+ o+x",
        Kind::Unruled,
        PARTNERED,
        0,
    ),
    row(
        "the partners over beside an idle decode apart; the prefill, no decode",
        "o+",
        "v- o-x",
        Kind::Keeps,
        PARTNERED,
        3,
    ),
    row(
        "the partners over beside an idle decode apart; the prefill and the partner",
        "o+",
        "v+ o-x",
        Kind::Among,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; an open prefill and a decode that is not ready alone",
        "o+",
        "v- u+",
        Kind::Among,
        HEALTHY,
        3,
    ),
    row(
        "the prefills over; an over prefill and a decode that is not ready alone",
        "v+",
        "o- u+",
        Kind::Among,
        SATURATED,
        3,
    ),
    row(
        "both over; one of each taken",
        "v+",
        "v+",
        Kind::Among,
        SATURATED,
        2,
    ),
    row(
        "both over; no rule",
        "v+",
        "v+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    // A worker under a liveness doubt is under its thresholds: its list is
    // not wholly over them, and the request has no endpoint, under the
    // health stage.
    row(
        "the partners over beside a decode under a doubt apart; the prefill, no decode",
        "o+",
        "v- s-x",
        Kind::Keeps,
        HEALTHY,
        3,
    ),
    row(
        "the partners over beside a decode under its host's veto apart; the prefill, no decode",
        "o+",
        "v- m-x",
        Kind::Keeps,
        HEALTHY,
        3,
    ),
    row(
        "the prefills over beside one under a doubt; no prefill taken",
        "v- s-",
        "u+",
        Kind::Keeps,
        HEALTHY,
        2,
    ),
    // A prefill that pairs with no decode is no part of the prefills'
    // count, and no prefill of the rule's: a rule that takes only such
    // prefills is answered as one that takes none, and behind a shed over
    // the prefills its decodes are not asked.
    row(
        "the one prefill over pairs with no decode; no prefill taken",
        "u- v-x",
        "o+",
        Kind::Keeps,
        HEALTHY,
        2,
    ),
    row(
        "the decodes over; the rule's one prefill pairs with no decode",
        "o- o+x",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the prefills over; the rule's one prefill pairs with no decode",
        "v- v+x",
        "o-",
        Kind::Stands,
        SATURATED,
        2,
    ),
    row(
        "the prefills over; the rule's one prefill pairs with no decode, and a decode",
        "v- v+x",
        "o+",
        Kind::Stands,
        SATURATED,
        2,
    ),
    // The rule's prefills, where it takes no decode.
    row(
        "the decodes over; the rule's one prefill is under a doubt",
        "o- s+",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the decodes over; the rule's one prefill is under its host's veto",
        "o- m+",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the prefills over; the rule's second prefill is the ready one, no decode",
        "u+ v+",
        "o-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; one of the rule's two prefills is under a doubt",
        "o+ s+",
        "v-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; the rule's one prefill is over its thresholds and under a doubt",
        "o- w+",
        "v-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    // A worker under a hold of its host is not ready: no part of a
    // list's count, no prefill of the rule's that could let a shed
    // stand, and none that keeps the rule's prefills from being wholly
    // over their thresholds. The gateway's own selection has no hold,
    // and a gateway hands the engine none. Its answer for these four
    // rows was read and not run: what it answers where that worker is
    // not ready, as one behind an open breaker is not.
    row(
        "the partners over beside a decode under its host's hold apart; the prefill, no decode",
        "o+",
        "v- h-x",
        Kind::Stands,
        SATURATED,
        3,
    ),
    row(
        "the decodes over; the rule's one prefill is under its host's hold",
        "o- h+",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    // Over its thresholds too, a held prefill is not ready: the hold is
    // read ahead of the thresholds, so the rule's prefills hold no ready
    // worker, and no shed stands by them.
    row(
        "the decodes over; the rule's one prefill is under its host's hold and over its thresholds",
        "o- k+",
        "v-",
        Kind::Told,
        ADMITS,
        3,
    ),
    // The held prefill is under its thresholds and the rule's other one
    // is over them: the hold leaves the second as the rule's one ready
    // prefill, so the rule's prefills are wholly over their thresholds.
    row(
        "the prefills over; the rule's first prefill is under its host's hold, no decode",
        "h+ v+",
        "o-",
        Kind::Stands,
        SATURATED,
        3,
    ),
    // Under no rule the first decision's shed is the answer. A worker
    // under its thresholds keeps the list judged from being wholly over
    // them, one under a doubt and one that is not judged for a pair too.
    row(
        "the partners over beside a decode under a doubt apart; no rule",
        "o+",
        "v+ s+x",
        Kind::Unruled,
        HEALTHY,
        0,
    ),
    row(
        "the prefills over beside one under a doubt; no rule",
        "v+ s+",
        "u+",
        Kind::Unruled,
        HEALTHY,
        0,
    ),
    row(
        "the one prefill over pairs with no decode; no rule",
        "u+ v+x",
        "o+",
        Kind::Unruled,
        HEALTHY,
        0,
    ),
    row(
        "no prefill pairs with a decode, and the prefill is over; no rule",
        "v+",
        "o+x",
        Kind::Unruled,
        PARTNERED,
        0,
    ),
    // A worker over its thresholds and under a doubt is ready in the
    // count, and a prefill that pairs with no decode is no part of it: the
    // list judged is wholly over its thresholds.
    row(
        "the one decode is over and under a doubt; no rule",
        "o+",
        "w+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    row(
        "the one prefill is over and under a doubt; no rule",
        "w+",
        "o+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    row(
        "the paired prefill over beside an open one that pairs with no decode; no rule",
        "v+ o+x",
        "o+",
        Kind::Unruled,
        SATURATED,
        0,
    ),
    // Behind a first decision that shed with no endpoint, as behind one
    // that picked: no decision judged the rule's prefills, and one more
    // does.
    row(
        "the partners over beside an idle decode apart; an over prefill alone, no decode",
        "o- v+",
        "v- o-x",
        Kind::Judged,
        SATURATED,
        4,
    ),
    // A decision among the rule's workers of each list asks the one
    // question a first decision asks, over those workers.
    row(
        "the partners over beside an idle decode apart; the prefill and both decodes",
        "o+",
        "v+ o+x",
        Kind::Among,
        PARTNERED,
        3,
    ),
    row(
        "the prefills over beside one under a doubt; one of each taken",
        "v+ s+",
        "u+",
        Kind::Among,
        HEALTHY,
        3,
    ),
    row(
        "a pick refused; the rule's one decode is over and pairs with no prefill",
        "o+",
        "o- v+x",
        Kind::Among,
        SATURATED,
        5,
    ),
    // A first decision that sheds with no endpoint has the rule asked:
    // the rule's workers may be wholly over their thresholds where the
    // whole lists are not.
    row(
        "the one decode is not ready; one of each taken",
        "o+",
        "u+",
        Kind::Among,
        HEALTHY,
        2,
    ),
    row(
        "the one decode is not ready; an over prefill alone, and the decode",
        "o- v+",
        "u+",
        Kind::Among,
        SATURATED,
        3,
    ),
    // A first decision's shed over a list whose one ready worker is under
    // a doubt too, or beside a prefill that pairs with no decode, stands
    // as any.
    row(
        "the one prefill is over and under a doubt; no prefill taken",
        "w-",
        "o+",
        Kind::Stands,
        SATURATED,
        1,
    ),
    row(
        "the one decode is over and under a doubt; the prefill, no decode",
        "o+",
        "w-",
        Kind::Stands,
        SATURATED,
        2,
    ),
    row(
        "the paired prefill over beside an open one that pairs with no decode; no prefill taken",
        "v- o-x",
        "o+",
        Kind::Stands,
        SATURATED,
        2,
    ),
];

/// One worker of a row.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Stated {
    /// `o`, `v`, `u`, `s`, `m`, `w`, `h` or `k`.
    state: char,
    /// Whether the route's rule takes the worker.
    pub(in super::super) taken: bool,
    /// Whether the worker pairs with nobody.
    pub(in super::super) apart: bool,
}

impl Stated {
    /// Puts `worker` in the row's state for it. A standing veto of its
    /// host is no state of the worker, and nor is its hold: they stand
    /// beside the list ([`masked`], [`holds`]).
    pub(in super::super) fn set(self, worker: &dyn Worker) {
        if matches!(self.state, 'v' | 'w' | 'k') {
            worker.set_overloaded(true);
        }
        if self.state == 'u' {
            worker.set_status(WorkerStatus::NotReady);
        }
        if matches!(self.state, 's' | 'w') {
            worker.set_stall(Some(StallReason::Wedged));
        }
    }
}

/// The workers one list of a row states, in the list's order.
pub(in super::super) fn stated(list: &str) -> Vec<Stated> {
    let worker = |token: &str| {
        let mut chars = token.chars();
        let (state, rule, apart) = (chars.next(), chars.next(), chars.next());
        assert!(
            matches!(state, Some('o' | 'v' | 'u' | 's' | 'm' | 'w' | 'h' | 'k'))
                && matches!(rule, Some('+' | '-'))
                && matches!(apart, None | Some('x'))
                && chars.next().is_none(),
            "a worker of a row: {token}"
        );
        Stated {
            state: state.unwrap_or('o'),
            taken: rule == Some('+'),
            apart: apart.is_some(),
        }
    };
    list.split_whitespace().map(worker).collect()
}

/// The pair classes of a row's two lists: every prefill that pairs pairs
/// with every decode that pairs, and a worker apart pairs with nobody.
pub(in super::super) fn partners(prefills: &[Stated], decodes: &[Stated]) -> PartnerClasses {
    let paired = (0..).zip(decodes).filter(|(_, decode)| !decode.apart);
    let paired: Vec<u32> = paired.map(|(place, _)| place).collect();
    let of = |prefill: &Stated| if prefill.apart { &[][..] } else { &paired[..] };
    PartnerClasses::number(decodes.len(), prefills.iter().map(of))
}

/// The places of a list's workers that a standing veto of their host
/// masks, as the host states them beside the list: a row its hold stands
/// on is a row of the mask too.
pub(in super::super) fn masked(list: &[Stated]) -> Vec<u32> {
    let masked = (0..)
        .zip(list)
        .filter(|(_, worker)| matches!(worker.state, 'm' | 'h' | 'k'));
    masked.map(|(place, _)| place).collect()
}

/// The places of a list's workers that a hold of their host stands on, as
/// the host states them beside the list.
pub(in super::super) fn holds(list: &[Stated]) -> Vec<u32> {
    let held = (0..)
        .zip(list)
        .filter(|(_, worker)| matches!(worker.state, 'h' | 'k'));
    held.map(|(place, _)| place).collect()
}

/// The stock pair document of `round_robin` on both sides, under a binding
/// that sheds a list wholly over its thresholds where `sheds`, and steers
/// to it else.
fn round_robin(sheds: bool) -> &'static InProcess {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: sheds,
        ..OverloadSpec::default()
    });
    hosted(engine_of(&document))
}

/// The shed `engine` answers a pair's request over a row's two lists, and
/// the times the request's rule was asked.
fn asked_under(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
) -> (Answer, usize) {
    let (unplaced, asked) = unplaced_under(engine, prefills, decodes, ruled);
    (shed_of(Err(unplaced)), asked)
}

/// [`asked_under`] with the call's whole answer, which says the shed's
/// side too.
fn unplaced_under(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
) -> (Unplaced, usize) {
    let (answer, asked) = answered_under(engine, prefills, decodes, ruled);
    (answer.expect_err("no placement"), asked)
}

/// What `engine` answers a pair's request over a row's two lists, and the
/// times the request's rule was asked: the places of the pair it is
/// placed on, or the call's whole answer.
fn answered_under(
    engine: &'static InProcess,
    prefills: &str,
    decodes: &str,
    ruled: bool,
) -> (Result<(usize, usize), Unplaced>, usize) {
    let (prefills, decodes) = (stated(prefills), stated(decodes));
    let lists = Lists::of(prefills.len(), decodes.len());
    let classes = partners(&prefills, &decodes);
    let (prefill_stalled, decode_stalled) = (masked(&prefills), masked(&decodes));
    let (prefill_held, decode_held) = (holds(&prefills), holds(&decodes));
    let pair = PairLists {
        prefill_stalled: &prefill_stalled,
        prefill_held: &prefill_held,
        decode_stalled: &decode_stalled,
        decode_held: &decode_held,
        ..lists.under(&classes, &Open)
    };
    let mut taken = Vec::new();
    for (at, worker) in prefills.iter().enumerate() {
        worker.set(lists.prefills[at].as_ref());
        if worker.taken {
            taken.push(lists.prefill_id(at));
        }
    }
    for (at, worker) in decodes.iter().enumerate() {
        worker.set(lists.decodes[at].as_ref());
        if worker.taken {
            taken.push(lists.decode_id(at));
        }
    }
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        asked.set(asked.get() + 1);
        taken.contains(&worker_id(worker))
    };
    let request = PlaceRequest {
        admits: if ruled { Some(&admits) } else { None },
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placed = engine.place(&request);
    let answer = placed.map(|placement| lists.places(&placement));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    (answer, asked.get())
}

/// The shed a call answered.
pub(in super::super) fn shed_of(placed: Result<Placement<'_>, Unplaced>) -> Answer {
    match placed.err() {
        Some(Unplaced::Shed {
            reason, emptied_by, ..
        }) => (reason, emptied_by),
        other => panic!("a shed, and not {other:?}"),
    }
}

/// The rows of `kind` that `asked` answers otherwise than the row says,
/// each with what it answered: none, where the rows hold.
pub(in super::super) fn failing(
    kind: Kind,
    asked: impl Fn(&Row, bool) -> (Answer, usize),
) -> Vec<String> {
    let mut rows = ROWS.iter().filter(|row| row.kind == kind).peekable();
    assert!(rows.peek().is_some(), "a row of {kind:?}");
    let failed = rows.filter_map(|row| {
        let answered = asked(row, row.kind != Kind::Unruled);
        let said = (row.answer, row.asked);
        (answered != said).then(|| format!("{}: {answered:?}, and not {said:?}", row.name))
    });
    failed.collect()
}

/// Holds every row of `kind` as a pair's request under a binding that
/// sheds: one engine answers them all, each over lists of its own.
fn held(kind: Kind) {
    let engine = round_robin(true);
    let asked = |row: &Row, ruled| asked_under(engine, row.prefills, row.decodes, ruled);
    let failed = failing(kind, asked);
    assert!(failed.is_empty(), "{failed:#?}");
}

/// A first decision's shed as saturated stands where the rule takes no
/// worker of a list, in two cases, and in both that decision said a list.
/// The rule takes no prefill that pairs with a decode, and the shed was
/// made over the prefills. Or the rule takes such a prefill and no decode,
/// and its prefills let the shed stand: one of them can take the request
/// and the shed was made over the decodes, or none can, one is ready and
/// every ready one is over its thresholds. No decision among the rule's
/// workers can be made there. The rule is asked of the prefills alone in
/// the first case, and of both lists in the second.
///
/// The gateway's own selection answers its overload shed in each of these
/// rows (`model_gateway/src/routers/common/placement.rs`, on main
/// `56773d28d`). Where the rule takes no prefill that pairs with a decode
/// it judges the prefill list whole (`:2866-2884`, `admitted_pool` at
/// `:1831-1852`), and the list is wholly over its thresholds. Where no
/// prefill the rule takes can take the request it judges the rule's
/// prefills (`:2866-2884`), one under a liveness doubt among them. Where
/// one can and the rule takes no decode it judges the decode list whole
/// (`:2909-2936`), and that list is wholly over its thresholds. The shed
/// is its answer under its shed option (`failure_from`, `:1710-1722`;
/// `model_gateway/src/routers/common/overload.rs:58-97`).
#[test]
fn a_saturated_shed_stands_where_the_rule_takes_no_worker_of_a_list() {
    held(Kind::Stands);
}

/// A shed that stands names the side of the list the gateway's own
/// selection judges in its row, which is the role it refuses the pair for
/// (`pair_pass` in `model_gateway/src/routers/common/placement.rs`): the
/// prefill's where the rule takes no prefill that pairs with a decode and
/// where no prefill it takes can take the request, and the decode's where
/// one can and the rule takes no decode. The four rows named here stand
/// for the decodes, and every other for the prefills.
///
/// In two of those the decision itself shed over the decodes: the rule's
/// one prefill is over its thresholds beside an idle prefill it refuses.
/// The shed stands for the prefills there, whichever list the decision
/// said ([`stands`](super::super::super::pair::stands)).
#[test]
fn a_shed_that_stands_names_the_side_of_the_list_it_stands_for() {
    const FOR_THE_DECODES: [&str; 4] = [
        "the decodes over; an open prefill, no decode",
        "the decodes over; one of the rule's two prefills is under a doubt",
        "the partners over beside a decode under its host's hold apart; the prefill, no decode",
        "the one decode is over and under a doubt; the prefill, no decode",
    ];
    const TURNED: [&str; 2] = [
        "the decodes over; an over prefill alone, no decode",
        "the decodes over; the rule's one prefill is over its thresholds and under a doubt",
    ];
    let engine = round_robin(true);
    let side_of = |row: &Row, ruled| match unplaced_under(engine, row.prefills, row.decodes, ruled)
    {
        (Unplaced::Shed { side, .. }, _) => side,
        other => panic!("{}: a shed, and not {other:?}", row.name),
    };
    let mut named = 0;
    for row in ROWS.iter().filter(|row| row.kind == Kind::Stands) {
        let decodes = FOR_THE_DECODES.contains(&row.name);
        named += usize::from(decodes) + usize::from(TURNED.contains(&row.name));
        let held = if decodes {
            PairSide::Decode
        } else {
            PairSide::Prefill
        };
        assert_eq!(side_of(row, true), Some(held), "{}", row.name);
        // With no rule the decision's own shed is the answer, with the
        // side the decision names.
        let own = if TURNED.contains(&row.name) {
            PairSide::Decode
        } else {
            held
        };
        assert_eq!(side_of(row, false), Some(own), "{}: no rule", row.name);
    }
    assert_eq!(named, 6, "each row named here is a row that stands");
}

/// A rule that takes no worker of a list is told by name where no shed of
/// the first decision is the answer: the request is shed with no endpoint,
/// under `admits`. So it is behind a first decision that picked. So it is
/// behind a shed as saturated that does not stand: one made over the
/// decodes where the rule takes no prefill that pairs, and one of either
/// list where the rule takes no decode and its prefills do not let it
/// stand, since none of them is ready, or one is under its thresholds and
/// under a liveness doubt.
///
/// The gateway's own selection answers that it has no worker in these
/// rows.
#[test]
fn a_rule_that_takes_no_worker_of_a_list_is_told_where_no_shed_stands() {
    held(Kind::Told);
}

/// A first decision's shed with no endpoint stands where the rule takes
/// no worker of a list: the rule is asked, of the prefills alone where it
/// takes none of them and of both lists else, and no decision can be made
/// among its workers. The shed names the stage that decision named: the
/// health stage where the worker with room of the list it judged is under
/// a liveness doubt or is not ready, and `partnered` where that worker is
/// an idle decode that pairs with no prefill.
///
/// The gateway's own selection answers that it has no worker in these
/// rows.
#[test]
fn a_shed_with_no_endpoint_stands_where_the_rule_takes_no_worker_of_a_list() {
    held(Kind::Keeps);
}

/// A first decision's shed with no endpoint made where no prefill of the
/// lists pairs with a decode answers the call as it is: the rule is not
/// asked, whatever it would take, and the shed keeps its stage and its
/// side, as under no rule. No rule changes that the lists pair nowhere,
/// and no decision among a rule's workers could pair. So it is for a
/// list that holds no worker, and under either word of the binding.
///
/// The gateway's own pair selection asks three tests of its lists whole,
/// in this order, ahead of any judgment its filter narrows: a prefill
/// list with no worker is refused for the prefill's role, a decode list
/// with no worker for the decode's, and lists where no prefill pairs
/// with a decode for the decode's too (`pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_shed_made_where_no_prefill_pairs_answers_ahead_of_the_rule() {
    const LISTS: [(&str, &str, PairSide); 8] = [
        ("o+", "o+x", PairSide::Decode),
        ("o-", "o-x", PairSide::Decode),
        ("v+ s-", "u+x o-x", PairSide::Decode),
        ("o+x u-x", "o+ v-", PairSide::Decode),
        ("o+", "", PairSide::Decode),
        ("o-", "", PairSide::Decode),
        ("", "o+", PairSide::Prefill),
        ("", "o-", PairSide::Prefill),
    ];
    for engine in [round_robin(true), round_robin(false)] {
        for (prefills, decodes, side) in LISTS {
            let (unruled, _) = unplaced_under(engine, prefills, decodes, false);
            let ruled = unplaced_under(engine, prefills, decodes, true);
            assert_eq!(ruled, (unruled, 0), "{prefills} | {decodes}");
            let named = match unruled {
                Unplaced::Shed {
                    reason: ShedReason::NoEndpoints,
                    side,
                    ..
                } => side,
                other => panic!("a shed with no endpoint, and not {other:?}"),
            };
            assert_eq!(named, Some(side), "{prefills} | {decodes}");
        }
    }
}

/// A decode list that holds no worker answers ahead of the rule by the
/// list's own test, whatever the classes say. The host numbered its
/// classes for a decode list of one worker, which the prefill pairs with,
/// and lends a decode list of none. The prefill's class is then of one
/// that pairs, and nothing pairs among the lists lent. The shed keeps
/// the decode's side, and the rule is not asked, under either word of
/// the binding.
#[test]
fn an_empty_decode_list_answers_ahead_of_the_rule_whatever_its_classes_say() {
    for engine in [round_robin(true), round_robin(false)] {
        let lists = Lists::of(1, 0);
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let asked = Cell::new(0);
        let admits = |_: &dyn Worker| {
            asked.set(asked.get() + 1);
            true
        };
        let request = PlaceRequest {
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let placed = engine.place(&request);
        assert!(
            matches!(
                placed,
                Err(Unplaced::Shed {
                    reason: ShedReason::NoEndpoints,
                    side: Some(PairSide::Decode),
                    ..
                })
            ),
            "{placed:?}"
        );
        assert_eq!(asked.get(), 0, "the rule is not asked");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }
}

/// A rule that takes only prefills over their thresholds and no decode
/// is judged by one decision more, where no decision judged the rule's
/// prefills: the first picked workers the rule refuses, or shed the
/// request with no endpoint over the lists whole. The call holds one
/// decision to the rule's prefills, with no decode lent: under a binding
/// that sheds it sheds the request as saturated over the prefills, and
/// that shed is the answer, for the prefill side. The rule is asked no
/// more for it.
///
/// The gateway's own selection judges the rule's prefills first where
/// none of them can take the request, and answers its overload shed for
/// the prefill's role (`pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`, where no prefill is
/// eligible).
#[test]
fn a_rule_of_prefills_over_their_thresholds_alone_is_judged_by_one_decision_more() {
    held(Kind::Judged);
    let engine = round_robin(true);
    for row in ROWS.iter().filter(|row| row.kind == Kind::Judged) {
        let (unplaced, _) = unplaced_under(engine, row.prefills, row.decodes, true);
        let side = match unplaced {
            Unplaced::Shed { side, .. } => side,
            other => panic!("{}: a shed, and not {other:?}", row.name),
        };
        assert_eq!(side, Some(PairSide::Prefill), "{}", row.name);
    }
}

/// Under a binding that steers, the decision held to the rule's prefills
/// sheds no request as saturated, and the answer stays what it was: the
/// shed under the rule's name behind a first decision that picked, and
/// the first decision's own shed behind one that shed the request with no
/// endpoint. The gateway's own selection sheds for overload under its shed
/// option alone.
#[test]
fn a_rule_of_prefills_over_their_thresholds_alone_keeps_its_answer_under_a_binding_that_steers() {
    let engine = round_robin(false);
    assert_eq!(asked_under(engine, "o- v+", "o-", true), (ADMITS, 5));
    assert_eq!(asked_under(engine, "o- v+", "v- o-x", true), (PARTNERED, 4));
}

/// A rule that takes a worker of each list is answered by the call's next
/// decision, made among them, by the one question a first decision asks:
/// its shed as saturated where the list judged among them is wholly over
/// its thresholds, and its shed with no endpoint else. So it is behind a
/// first decision that shed the request with no endpoint too: the rule's
/// workers may be wholly over their thresholds where the whole lists are
/// not.
#[test]
fn a_rule_that_takes_a_worker_of_each_list_is_answered_by_the_next_decision() {
    held(Kind::Among);
}

/// A pair that states no rule keeps its first decision's shed: as
/// saturated where the one list judged is wholly over its thresholds, and
/// with no endpoint else.
#[test]
fn a_pair_with_no_rule_keeps_the_shed_of_its_first_decision() {
    held(Kind::Unruled);
}

/// A first decision that sheds the request with no endpoint has the rule
/// asked, and the call's next decision is made among the rule's workers.
/// The one partner is under a liveness doubt beside an idle decode that
/// pairs with no prefill. Over the whole lists the decode list is not
/// refused on every worker, so the pair's default crosses no doubt, and
/// the first decision sheds the request under the health stage: under no
/// rule that shed is the answer. The rule takes the prefill and the
/// partner. Among those alone the default crosses the doubt, and the
/// request is placed on the two.
///
/// The gateway's own pair selection narrows its lists by the route's
/// filter ahead of every judgment of a worker, and its fallback pairs the two
/// (`pair_pass` and `relaxed_pair` in
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_rule_asked_behind_a_shed_with_no_endpoint_pairs_across_a_doubt() {
    let engine = round_robin(true);
    let placed = answered_under(engine, "o+", "s+ o-x", true);
    assert_eq!(placed, (Ok((0, 0)), 5));
    assert_eq!(asked_under(engine, "o+", "s+ o-x", false), (HEALTHY, 0));
}

/// The decision made among the rule's workers behind such a shed may
/// answer that the prefills are full, on which a host waits for room.
/// The lists are the ones above, and the one prefill is at its host's
/// admission limit. Under no rule the first decision's shed is the
/// answer, for the decode side. The rule takes the prefill and its
/// partner. Among those alone the default crosses the doubt, the pair is
/// the prefill's at its limit, and the request waits.
///
/// The gateway's own pair selection waits for such a prefill too. It
/// narrows its lists by the route's filter first, its fallback pairs the
/// two, and it asks the prefill's limit behind the pair (`pair_pass` and
/// `relaxed_pair` in `model_gateway/src/routers/common/placement.rs`).
/// That is read from those lines and not run.
#[test]
fn a_rule_asked_behind_a_shed_with_no_endpoint_may_find_its_prefills_full() {
    let engine = round_robin(true);
    let answered = |ruled: bool| {
        let (prefills, decodes) = (stated("o+"), stated("s+ o-x"));
        let lists = Lists::of(prefills.len(), decodes.len());
        let classes = partners(&prefills, &decodes);
        for (at, worker) in decodes.iter().enumerate() {
            worker.set(lists.decodes[at].as_ref());
        }
        let taken = [lists.prefill_id(0), lists.decode_id(0)];
        let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
        let full = Gates(|_| false, |_| true);
        let request = PlaceRequest {
            admits: if ruled { Some(&admits) } else { None },
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &full)))
        };
        engine.place(&request).err()
    };
    assert_eq!(answered(true), Some(Unplaced::PrefillsFull));
    let unruled = answered(false);
    assert!(
        matches!(
            unruled,
            Some(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "healthy",
                side: Some(PairSide::Decode),
                ..
            })
        ),
        "{unruled:?}"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Under a binding that steers to a list wholly over its thresholds, no
/// shed of a pair is saturated, and a decision says no list. The one
/// prefill is over its thresholds and the one decode is not ready, so the
/// binding has no pair to steer to: the first decision sheds the request
/// with no endpoint, under `partnered`, since the prefill has room to be
/// steered to and pairs with nobody that stands. The rule takes no
/// prefill, and that shed is the answer. Under a binding that sheds the
/// same request is shed as saturated, and that shed stands.
///
/// The gateway's own selection sheds for overload under its shed option
/// alone (`failure_from`,
/// `model_gateway/src/routers/common/placement.rs:1710-1722`, on main
/// `56773d28d`).
#[test]
fn a_binding_that_steers_sheds_no_pair_as_saturated() {
    assert_eq!(
        asked_under(round_robin(false), "v-", "u+", true),
        (PARTNERED, 1)
    );
    assert_eq!(
        asked_under(round_robin(true), "v-", "u+", true),
        (SATURATED, 1)
    );
}

/// A decision's word of the list it shed over is that decision's alone. A
/// first call's decision sheds over the prefills, and that shed stands. A
/// second call on the same thread sheds over its decodes, and its rule
/// takes no prefill: the request is shed under the rule's name. With the
/// word of the call before, which named the prefills, it would keep its
/// shed.
#[test]
fn a_decisions_word_of_its_shed_is_not_read_by_the_next_call() {
    let engine = round_robin(true);
    assert_eq!(asked_under(engine, "v- v-", "o+", true), (SATURATED, 2));
    assert_eq!(asked_under(engine, "o-", "v+", true), (ADMITS, 1));
}

/// The call reads its decision's word of the shed ahead of the rule. The
/// rule here places another pair on the calling thread while it is asked
/// of the first prefill, and that pair's decision picks: it says no list.
/// The first call's shed over its prefills stands all the same.
#[test]
fn the_decisions_word_is_read_ahead_of_the_rule() {
    let engine = round_robin(true);
    let lists = Lists::of(2, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    for prefill in lists.prefills.iter() {
        prefill.set_overloaded(true);
    }
    let other = Lists::of(1, 1);
    let other_classes = other.every();
    let placed = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        if placed.get() == 0 {
            let pair = other.under(&other_classes, &Open);
            let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
            let placement = engine.place(&request).expect("an open pair");
            assert_eq!(other.places(&placement), (0, 0));
        }
        placed.set(placed.get() + 1);
        worker_id(worker) == lists.decode_id(0)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(shed_of(engine.place(&request)), SATURATED);
    assert_eq!(placed.get(), 2, "the two prefills, and no decode");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The routine reads the rule's prefills behind the decision, and a prefill
/// that came under its thresholds in between keeps no shed over the
/// prefills. The rule takes the one prefill and no decode, and frees that
/// prefill while it is asked of it. The decision shed over the prefills,
/// and the prefill can take the request by the routine's read: the
/// gateway's own selection would then judge the decodes
/// (`model_gateway/src/routers/common/placement.rs:2909-2936`, on main
/// `56773d28d`), which that decision did not count. The request is shed
/// under the rule's name. A rule that leaves the prefill over its
/// thresholds keeps the decision's shed.
#[test]
fn a_prefill_that_comes_free_behind_the_decision_keeps_no_shed_over_the_prefills() {
    let engine = round_robin(true);
    let asked = |frees: bool| {
        let lists = Lists::of(1, 1);
        let classes = lists.every();
        let pair = lists.under(&classes, &Open);
        lists.prefills[0].set_overloaded(true);
        let admits = |worker: &dyn Worker| {
            let prefill = worker_id(worker) == lists.prefill_id(0);
            if prefill && frees {
                lists.prefills[0].set_overloaded(false);
            }
            prefill
        };
        let request = PlaceRequest {
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        shed_of(engine.place(&request))
    };
    assert_eq!(asked(true), ADMITS);
    assert_eq!(asked(false), SATURATED);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The decision's count of a list is not read again behind the rule. The
/// rule takes no prefill, and frees the one prefill while it is asked of
/// it: the decision shed over the prefills as it counted them, and that
/// shed stands. The rule's own prefills are the one read that is fresh
/// ([`a_prefill_that_comes_free_behind_the_decision_keeps_no_shed_over_the_prefills`]).
///
/// The gateway's own selection asks its filter first and judges the list
/// behind it
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`, on main
/// `56773d28d`): over the prefill as the rule left it, it would answer
/// that it has no worker. That is read from its lines and not run. Over
/// one state of the lists the two answer alike ([`ROWS`]).
#[test]
fn the_decisions_count_of_a_list_is_not_read_again_behind_the_rule() {
    let engine = round_robin(true);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    lists.prefills[0].set_overloaded(true);
    let admits = |worker: &dyn Worker| {
        if worker_id(worker) == lists.prefill_id(0) {
            lists.prefills[0].set_overloaded(false);
        }
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(shed_of(engine.place(&request)), SATURATED);
    assert!(!lists.prefills[0].routing_state().overloaded);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// `list` published with what each worker states for a demand: the
/// runtime 2 and a window of two tokens, but for the worker at `other`,
/// which states the runtime 1 and a window of one token.
fn stating(list: &Arc<[Arc<dyn Worker>]>, other: Option<usize>) -> PublishedList {
    let other = other.map(|at| Arc::clone(&list[at]));
    PublishedList::with_facts(Arc::clone(list), |worker| {
        let apart = other
            .as_ref()
            .is_some_and(|other| Arc::ptr_eq(worker, other));
        let mut facts = FixedFacts::default();
        facts.demands.runtime = if apart { 1 } else { 2 };
        facts.demands.window = if apart { 1 } else { 2 };
        facts
    })
}

/// A request that states a hard demand keeps the rule's name where its
/// rule takes no worker of a list: a demand hands the decision a part of
/// each list, the workers that meet it, and the call acts on no word of
/// the prefills or the decodes for such a request. Two states, each
/// answered as before a shed stood behind a rule.
///
/// A pinned runtime. The first prefill and the decode state it, and the
/// second prefill states another. The decode is over its thresholds, and
/// the rule takes the second prefill alone. The decision counted the
/// decode list and said it, and the rule's one prefill is no worker of
/// the request's. The gateway's own selection finds no prefill that its
/// wire and its filter both take, judges the prefills on the wire, and
/// answers that it has no worker
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`,
/// `admitted_pool` at `:1831-1852`, on main `56773d28d`).
///
/// An input window of two tokens. The first prefill holds it and is over
/// its thresholds, the second holds one token and is open, and the rule
/// takes no prefill. The decision counted the first prefill alone and
/// said the prefills. The gateway's own selection holds a window in its
/// filter
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2492-2526`,
/// on main `9bfcf9bd2`): where the filter takes nobody it judges both
/// prefills, and answers that it has no worker.
///
/// What that selection answers in the two states was read from those
/// lines and not run. With no demand the first state's shed stands, and
/// the second's lists are answered by a first decision that picks.
#[test]
fn a_request_that_states_a_hard_demand_keeps_the_rules_name() {
    let engine = round_robin(true);
    let none = Demands::default();
    // A pinned runtime: the second prefill is off it, and the rule takes
    // that prefill alone.
    let wire = |demands: Demands| {
        let mut lists = Lists::of(2, 1);
        lists.published = [
            stating(&lists.prefills, Some(1)),
            stating(&lists.decodes, None),
        ];
        let classes = lists.every();
        lists.decodes[0].set_overloaded(true);
        let admits = |worker: &dyn Worker| worker_id(worker) == lists.prefill_id(1);
        let request = PlaceRequest {
            admits: Some(&admits),
            demands,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        shed_of(engine.place(&request))
    };
    assert_eq!(wire(Demands { runtime: 2, ..none }), ADMITS);
    assert_eq!(wire(none), SATURATED, "no demand: the shed stands");
    // An input window: the second prefill does not hold it, and the rule
    // takes no prefill.
    let window = |demands: Demands| {
        let mut lists = Lists::of(2, 1);
        lists.published = [
            stating(&lists.prefills, Some(1)),
            stating(&lists.decodes, None),
        ];
        let classes = lists.every();
        lists.prefills[0].set_overloaded(true);
        let admits = |worker: &dyn Worker| worker_id(worker) == lists.decode_id(0);
        let request = PlaceRequest {
            admits: Some(&admits),
            demands,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        shed_of(engine.place(&request))
    };
    let two_tokens = Demands {
        input_window: 2,
        ..none
    };
    assert_eq!(window(two_tokens), ADMITS);
    assert_eq!(
        window(none),
        ADMITS,
        "no demand: a first decision that picks"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A request that states a hard demand is answered as it was before a
/// pair's decision asked its one question. The demand hands the decision
/// a part of each list, the workers that meet it, and the gateway's own
/// selection judges a wider list there (`admitted_pool` in
/// `model_gateway/src/routers/common/placement.rs`). Every worker of the
/// three states meets the demand, so the same lists answer by the
/// question where the request states none.
///
/// The decision asks no question. The prefill's one partner is not ready
/// beside a decode over its thresholds that pairs with nobody. With no
/// demand every ready decode is over its thresholds, and the shed is
/// saturated. Behind the demand the health stage names it, which left
/// the pass nobody.
///
/// A pass the overload veto ended keeps its word. The partner is over
/// its thresholds beside a decode with room that pairs with nobody and
/// stands under a liveness doubt. With no demand the request has no
/// endpoint, under the health stage. Behind the demand the shed stays
/// saturated.
///
/// The rule is asked behind a shed with no endpoint for a request that
/// states no demand alone. The first decision sheds the request for the
/// decode side: the prefill's one partner stands under a liveness doubt.
/// The rule takes the prefill and that decode. With no demand the next
/// decision pairs the two across the doubt. Behind the demand the shed
/// answers the call with its side, and the rule is not asked.
///
/// A shed with no endpoint that stands behind the rule names no side. Of
/// two prefills one is over its thresholds and one under a liveness doubt,
/// the decode is not ready, and the rule takes no prefill.
///
/// The call acts on no word of a list for a request that states a
/// demand: the decision's own, or one the call before left on its
/// thread. That call shed over its prefills, and its shed stood. The
/// request that states the demand is shed as saturated by its first
/// decision, and its rule takes no prefill: it is shed under the rule's
/// name. The framework's own case of a part holds the word such a
/// decision says.
#[test]
fn a_request_that_states_a_hard_demand_is_answered_as_before_the_question() {
    let engine = round_robin(true);
    let wire = Demands {
        runtime: 2,
        ..Demands::default()
    };
    let answered = |prefills: &str, decodes: &str, ruled: bool, demands: Demands| {
        let (prefills, decodes) = (stated(prefills), stated(decodes));
        let mut lists = Lists::of(prefills.len(), decodes.len());
        lists.published = [
            stating(&lists.prefills, None),
            stating(&lists.decodes, None),
        ];
        let classes = partners(&prefills, &decodes);
        let mut taken = Vec::new();
        for (at, worker) in prefills.iter().enumerate() {
            worker.set(lists.prefills[at].as_ref());
            if worker.taken {
                taken.push(lists.prefill_id(at));
            }
        }
        for (at, worker) in decodes.iter().enumerate() {
            worker.set(lists.decodes[at].as_ref());
            if worker.taken {
                taken.push(lists.decode_id(at));
            }
        }
        let asked = Cell::new(0);
        let admits = |worker: &dyn Worker| {
            asked.set(asked.get() + 1);
            taken.contains(&worker_id(worker))
        };
        let request = PlaceRequest {
            admits: if ruled { Some(&admits) } else { None },
            demands,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let placed = engine.place(&request);
        (
            placed.map(|placement| lists.places(&placement)),
            asked.get(),
        )
    };
    let shed = |prefills: &str, decodes: &str, ruled: bool, demands: Demands| -> Answer {
        match answered(prefills, decodes, ruled, demands).0 {
            Err(Unplaced::Shed {
                reason, emptied_by, ..
            }) => (reason, emptied_by),
            other => panic!("a shed, and not {other:?}"),
        }
    };
    let none = Demands::default();
    // No question: the count makes no shed saturated behind the demand.
    assert_eq!(shed("o+", "u+ v+x", false, none), SATURATED);
    assert_eq!(shed("o+", "u+ v+x", false, wire), HEALTHY);
    // The pass's word: the overload veto ended it, and it stays saturated.
    assert_eq!(shed("o+", "v+ s+x", false, none), HEALTHY);
    assert_eq!(shed("o+", "v+ s+x", false, wire), SATURATED);
    // The call's arm: the rule is asked behind a shed with no endpoint
    // for a request that states no demand alone.
    let (placed, asked) = answered("o+", "o-x s+", true, none);
    assert_eq!(placed.ok(), Some((0, 1)), "the rule's pair across a doubt");
    assert!(asked > 0, "the rule is asked");
    let (placed, asked) = answered("o+", "o-x s+", true, wire);
    assert!(
        matches!(
            placed,
            Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "healthy",
                side: Some(PairSide::Decode),
                ..
            })
        ),
        "{placed:?}"
    );
    assert_eq!(asked, 0, "the rule is not asked behind the demand");
    // Behind the rule that shed names no side. The rule takes no prefill
    // here, and it narrowed the lists the first decision judged.
    let (placed, asked) = answered("v- s-", "u+", true, none);
    assert!(
        matches!(
            placed,
            Err(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "healthy",
                side: None,
                ..
            })
        ),
        "{placed:?}"
    );
    assert!(asked > 0, "the rule is asked");
    // The call acts on no word behind a demand. A first call sheds over
    // its prefills, and that shed stands behind a rule that takes no
    // prefill. The next call on this thread states the demand: its
    // decision sheds as saturated, and its rule takes no prefill. It is
    // shed under the rule's name, whichever word stands.
    assert_eq!(shed("v- v-", "o+", true, none), SATURATED);
    assert_eq!(shed("v-", "o+", true, wire), ADMITS);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A request whose host holds it to places of the prefills or of the
/// decodes keeps the rule's name where its rule takes no worker of a
/// list, as one that states a hard demand does: the places hand the
/// decision a part of that list, the workers at them, and the call acts
/// on no word of the prefills or the decodes for such a request. Two
/// states, each answered as before a shed stood behind a rule.
///
/// The prefills' places. The first prefill is over its thresholds and is
/// the one place stated, and the rule takes no prefill. The decision
/// counts that prefill alone and says the prefills: the framework's own
/// case of a count at the request's places holds that word. The request
/// is shed under the rule's name with the second prefill under its
/// thresholds, and with it over them too. With no places stated and both
/// prefills over their thresholds the shed stands.
///
/// The decodes' places. The first decode is over its thresholds and is the
/// one place stated, and the rule takes the prefill and no decode. The
/// decision counts that decode alone and says the decodes, and the rule's
/// prefill can take the request. The three answers are as the prefills'.
///
/// The gateway's own selection states no places
/// (`model_gateway/src/routers/common/placement.rs:2511-2515`, on main
/// `56773d28d`). Held to such a part by its filter, it would judge the
/// list whole where the filter takes nobody (`admitted_pool`,
/// `:1831-1852`): it has no worker where the worker outside the places is
/// under its thresholds, and sheds for overload where that one is over
/// them too. That is read from its lines and not run.
#[test]
fn a_request_held_to_places_of_a_list_keeps_the_rules_name() {
    let engine = round_robin(true);
    // The one place a list's host states: its first.
    let first: &[u32] = &[0];
    // The prefills: the first is over its thresholds, and the rule takes
    // no prefill.
    let prefills = |held: bool, other_over: bool| {
        let lists = Lists::of(2, 1);
        let classes = lists.every();
        lists.prefills[0].set_overloaded(true);
        lists.prefills[1].set_overloaded(other_over);
        let admits = |worker: &dyn Worker| worker_id(worker) == lists.decode_id(0);
        let pair = PairLists {
            prefill_places: held.then_some(first),
            ..lists.under(&classes, &Open)
        };
        let request = PlaceRequest {
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        shed_of(engine.place(&request))
    };
    assert_eq!(prefills(true, false), ADMITS);
    assert_eq!(prefills(true, true), ADMITS, "the list wholly over");
    assert_eq!(prefills(false, true), SATURATED, "no places: the shed");
    // The decodes: the first is over its thresholds, and the rule takes
    // the prefill and no decode.
    let decodes = |held: bool, other_over: bool| {
        let lists = Lists::of(1, 2);
        let classes = lists.every();
        lists.decodes[0].set_overloaded(true);
        lists.decodes[1].set_overloaded(other_over);
        let admits = |worker: &dyn Worker| worker_id(worker) == lists.prefill_id(0);
        let pair = PairLists {
            decode_places: held.then_some(first),
            ..lists.under(&classes, &Open)
        };
        let request = PlaceRequest {
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        shed_of(engine.place(&request))
    };
    assert_eq!(decodes(true, false), ADMITS);
    assert_eq!(decodes(true, true), ADMITS, "the list wholly over");
    assert_eq!(decodes(false, true), SATURATED, "no places: the shed");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A request held to a part of a list is asked no decision more for its
/// rule's prefills: the answer stays under the rule's name. The first
/// prefill is open and the second over its thresholds; the rule takes the
/// second alone, and no decode. The first decision picks the first
/// prefill, which the rule refuses. Where the host states places of the
/// prefills, every one of them here, the call judges no list for such a
/// request, as behind a first shed as saturated. With no places stated,
/// one decision more is held to the rule's prefill and sheds the request
/// as saturated.
///
/// The gateway's own selection states no places: it holds such a need in
/// its filter.
#[test]
fn a_request_held_to_a_part_of_a_list_is_asked_no_decision_more() {
    let engine = round_robin(true);
    // The places a list's host states: every one.
    let every: &[u32] = &[0, 1];
    let asked = |held: bool| {
        let lists = Lists::of(2, 1);
        let classes = lists.every();
        lists.prefills[1].set_overloaded(true);
        let admits = |worker: &dyn Worker| worker_id(worker) == lists.prefill_id(1);
        let pair = PairLists {
            prefill_places: held.then_some(every),
            ..lists.under(&classes, &Open)
        };
        let request = PlaceRequest {
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        shed_of(engine.place(&request))
    };
    assert_eq!(asked(true), ADMITS);
    assert_eq!(asked(false), SATURATED, "no places: one decision more");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule's prefills are read to the first that can take the request:
/// here three that are not ready stand ahead of one that is open, beside
/// a decode over its thresholds that the rule refuses. The decision shed
/// over the decodes, and the shed stands by the fourth prefill. Where the
/// rule refuses that one, no prefill of the rule's is ready, and the
/// request is shed under the rule's name.
#[test]
fn the_rules_fourth_prefill_lets_the_shed_stand() {
    let engine = round_robin(true);
    let asked = |prefills| asked_under(engine, prefills, "v-", true);
    assert_eq!(asked("u+ u+ u+ o+"), (SATURATED, 5));
    assert_eq!(asked("u+ u+ u+ o-"), (ADMITS, 5));
}

/// Asks `inner` from inside a pair's decision of this thread, and answers
/// what it said each time, with whether the thread's buffers were lent
/// then. Under the `cache_aware` presets a decision that stores where its
/// prompt went asks the rule itself: the rule here is `inner`. One of the
/// times at least is from inside that decision.
pub(in super::super) fn asked_inside<T>(inner: impl Fn() -> T) -> Vec<(bool, T)> {
    let mut document = PolicySpec::pair_presets("cache_aware", "cache_aware").expect("presets");
    document.generation = 1;
    let outer = hosted(engine_of(&document));
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let tokens: Vec<u32> = (1_000..5_096).collect();
    let inside = RefCell::new(Vec::new());
    let admits = |_: &dyn Worker| {
        // Whether the rule is asked from inside a decision of this thread:
        // the thread's buffers are then lent.
        let lent = BUFFERS.with(|buffers| buffers.try_borrow_mut().is_err());
        inside.borrow_mut().push((lent, inner()));
        true
    };
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = outer.place(&request).expect("the open pair");
    assert_eq!(lists.places(&placement), (0, 0));
    drop(placement);
    assert_eq!((outer.faults(), outer.unheard()), (0, 0));
    let inside = inside.into_inner();
    assert!(
        inside.iter().any(|(lent, _)| *lent),
        "the rule is asked from inside the decision"
    );
    inside
}

/// A decision made from inside another of its thread takes buffers of its
/// own, and its word of a shed is kept for its call. The rule of a pair's
/// request places a second pair from inside that pair's decision
/// ([`asked_inside`]). The second pair's prefills are over their
/// thresholds under a binding that sheds and its rule takes none of them:
/// that request keeps its decision's shed, though the thread's buffers
/// were lent when it was made. So does one whose decodes are over their
/// thresholds and whose rule takes its open prefill and no decode: the
/// word kept is the decision's own, of whichever list.
#[test]
fn a_decision_inside_another_keeps_its_word_of_a_shed() {
    let inner = round_robin(true);
    for (lent, answered) in asked_inside(|| asked_under(inner, "v- v-", "o+", true)) {
        assert_eq!(answered, (SATURATED, 2), "inside a decision: {lent}");
    }
    for (lent, answered) in asked_inside(|| asked_under(inner, "o+", "v-", true)) {
        assert_eq!(answered, (SATURATED, 2), "the decodes; inside: {lent}");
    }
}
