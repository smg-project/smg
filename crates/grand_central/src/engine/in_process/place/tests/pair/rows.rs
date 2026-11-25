//! The rows of a pair's request under a document that sheds a list wholly
//! over its thresholds: what the call answers where its first decision
//! sheds the request as saturated, or picks workers the rule refuses, and
//! the route's rule takes all, some or none of a list ([`again`]).
//!
//! A row is a prefill list and a decode list. A worker is written by its
//! state and by the rule's word of it. `o` is ready and under its
//! thresholds, `v` is ready and over them, and `u` is not ready. `s` and
//! `m` are ready, under their thresholds and under a liveness doubt: the
//! worker's own for `s`, and a standing veto of its host beside the list
//! for `m`. `w` is ready, over its thresholds and under a doubt of its
//! own. `h` is ready and under its thresholds, under a hold of its host
//! beside the list, which no rule lifts. `+` says that the rule takes the
//! worker, and `-` that it refuses it. A worker marked `x` pairs with
//! nobody, and every other prefill pairs with every other decode.
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
//! Each row's answer here is that selection's by its kind, a shed as
//! saturated for its overload shed and a shed with no endpoint for no
//! worker, but for the rows that say what it answers instead and which
//! open answer they stand under ([`parted`], [`Parted`]).
//!
//! [`again`]: super::super::super::pair::again

use smg_worker::runtime::StallReason;

use super::*;
use crate::engine::in_process::pair::BUFFERS;

/// A shed's reason and the stage it names.
pub(in super::super) type Answer = (ShedReason, &'static str);

/// The first decision's shed, or the shed of a decision held to the rule's
/// workers.
pub(in super::super) const SATURATED: Answer = (ShedReason::Saturated, "not-overloaded");
/// The rule takes no worker of a list.
pub(in super::super) const ADMITS: Answer = (ShedReason::NoEndpoints, "admits");
/// The workers a decision's pass was left are not ready, or are under a
/// liveness doubt that nothing lifts for them.
pub(in super::super) const HEALTHY: Answer = (ShedReason::NoEndpoints, "healthy");
/// No prefill that stands pairs with a decode at the request's places.
pub(in super::super) const PARTNERED: Answer = (ShedReason::NoEndpoints, "partnered");

/// What a row shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Kind {
    /// The rule takes no worker of a list, and the first decision's shed
    /// as saturated is the call's answer.
    Stands,
    /// The rule takes no worker of a list, and the request is shed under
    /// the rule's name.
    Told,
    /// The rule takes a worker of each list, and a decision among them
    /// answers.
    Among,
    /// The first decision sheds the request with no endpoint. That shed
    /// is the call's answer, and the rule is not asked.
    NotAsked,
    /// The request states no rule.
    Unruled,
}

/// What the gateway's own pair selection answers for a row's state, by its
/// kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Gateway {
    /// Its overload shed.
    Sheds,
    /// That it has no worker for the request, or that no prefill pairs
    /// with a decode.
    NoWorker,
    /// A pair, which its fallback makes over a list whose every ready
    /// worker is vetoed.
    Pairs,
}

/// The answers of a pair that stand open: each is the engine's answer
/// where the gateway's own selection answers by another kind. A row that
/// parts from that selection says which it stands under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Parted {
    /// No decision said a list the rule's prefills could be judged by.
    /// The rule takes only prefills over their thresholds and no decode,
    /// behind a first decision that picked or that said no list: the
    /// request is shed under the rule's name. That selection judges the
    /// rule's prefills and answers its overload shed
    /// (`model_gateway/src/routers/common/placement.rs:2866-2884`).
    NoWord,
    /// A decision's own shed. It is saturated where the overload veto's
    /// stage left its pass nobody, and names another stage else. That
    /// selection sheds for overload by its count over the list it judges
    /// (`model_gateway/src/routers/common/overload.rs:58-74`), which holds
    /// a worker under a liveness doubt, every decode of the list, and no
    /// prefill that pairs with no decode
    /// (`model_gateway/src/routers/common/pair_index.rs:158-161`). The two
    /// part both ways: the decision sheds as saturated where that count is
    /// not wholly over, and names the health stage or `partnered` where it
    /// is. So it is under no rule, among a rule's workers of each list,
    /// and ahead of a rule that is not asked.
    OwnShed,
    /// A rule that is not asked. Behind a first decision that sheds with
    /// no endpoint the rule narrows nothing. Where it takes workers under
    /// a doubt alone of a list, that selection's fallback crosses the
    /// doubt and pairs
    /// (`model_gateway/src/routers/common/placement.rs:2129-2192`).
    NotAsked,
}

/// One row: its two lists, whether the request states a rule, the call's
/// answer, and the times the rule is asked. A row whose answer is not the
/// gateway's own selection's by its kind says what that selection answers,
/// and the open answer it stands under.
#[derive(Debug)]
pub(in super::super) struct Row {
    pub(in super::super) name: &'static str,
    pub(in super::super) prefills: &'static str,
    pub(in super::super) decodes: &'static str,
    pub(in super::super) kind: Kind,
    pub(in super::super) answer: Answer,
    pub(in super::super) asked: usize,
    pub(in super::super) parted: Option<(Gateway, Parted)>,
}

/// A row whose answer is the gateway's own selection's by its kind.
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
        parted: None,
    }
}

/// `row` where the gateway's own selection answers `gateway`, of another
/// kind than the row's answer: the row stands under the open answer
/// `under`. A row that names the kind of its own answer does not build.
const fn parted(row: Row, gateway: Gateway, under: Parted) -> Row {
    let sheds = matches!(row.answer.0, ShedReason::Saturated);
    let same = match gateway {
        Gateway::Sheds => sheds,
        Gateway::NoWorker => !sheds,
        Gateway::Pairs => false,
    };
    assert!(!same, "the gateway's answer is of the row's kind");
    Row {
        parted: Some((gateway, under)),
        ..row
    }
}

/// The rows. The gateway's own selection answers by the row's kind in
/// each but the ones made by [`parted`], where it answers what the row
/// says. Each of its answers was run, by a run of that selection over the
/// row's state which this crate holds no case of: a worker `m` is an `s`
/// there, since that selection has one liveness doubt, the worker's own.
/// It was read and not run for the three rows that state a hold, which
/// say so.
pub(in super::super) const ROWS: [Row; 56] = [
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
    // The first decision picks, and the rule refuses its picks.
    parted(
        row(
            "a pick refused; an over prefill alone, no decode",
            "o- v+",
            "o-",
            Kind::Told,
            ADMITS,
            5,
        ),
        Gateway::Sheds,
        Parted::NoWord,
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
    // idle decode that pairs with no prefill.
    parted(
        row(
            "the partners over beside an idle decode apart; no rule",
            "o+",
            "v+ o+x",
            Kind::Unruled,
            SATURATED,
            0,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    row(
        "the partners over beside an idle decode apart; the prefill, no decode",
        "o+",
        "v- o-x",
        Kind::Told,
        ADMITS,
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
    // not wholly over them, and the decision says no list.
    row(
        "the partners over beside a decode under a doubt apart; the prefill, no decode",
        "o+",
        "v- s-x",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the partners over beside a decode under its host's veto apart; the prefill, no decode",
        "o+",
        "v- m-x",
        Kind::Told,
        ADMITS,
        3,
    ),
    row(
        "the prefills over beside one under a doubt; no prefill taken",
        "v- s-",
        "u+",
        Kind::Told,
        ADMITS,
        2,
    ),
    // A prefill that pairs with no decode is no part of the prefills'
    // count, and no prefill of the rule's.
    row(
        "the one prefill over pairs with no decode; no prefill taken",
        "u- v-x",
        "o+",
        Kind::Told,
        ADMITS,
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
        3,
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
    // and a gateway hands the engine none. Its answer for these three
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
    // Under no rule the first decision's shed is its own count's.
    parted(
        row(
            "the partners over beside a decode under a doubt apart; no rule",
            "o+",
            "v+ s+x",
            Kind::Unruled,
            SATURATED,
            0,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the prefills over beside one under a doubt; no rule",
            "v+ s+",
            "u+",
            Kind::Unruled,
            SATURATED,
            0,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the one prefill over pairs with no decode; no rule",
            "u+ v+x",
            "o+",
            Kind::Unruled,
            SATURATED,
            0,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    parted(
        row(
            "no prefill pairs with a decode, and the prefill is over; no rule",
            "v+",
            "o+x",
            Kind::Unruled,
            SATURATED,
            0,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    // A first decision's shed that names another stage than the overload
    // veto's, where the list the gateway judges is wholly over.
    parted(
        row(
            "the one decode is over and under a doubt; no rule",
            "o+",
            "w+",
            Kind::Unruled,
            HEALTHY,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the one prefill is over and under a doubt; no rule",
            "w+",
            "o+",
            Kind::Unruled,
            HEALTHY,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the paired prefill over beside an open one that pairs with no decode; no rule",
            "v+ o+x",
            "o+",
            Kind::Unruled,
            PARTNERED,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    // Behind a first decision that said no list, as behind one that
    // picked.
    parted(
        row(
            "the partners over beside an idle decode apart; an over prefill alone, no decode",
            "o- v+",
            "v- o-x",
            Kind::Told,
            ADMITS,
            4,
        ),
        Gateway::Sheds,
        Parted::NoWord,
    ),
    // A decision among the rule's workers of each list sheds by its own
    // stages, as a first decision does.
    parted(
        row(
            "the partners over beside an idle decode apart; the prefill and both decodes",
            "o+",
            "v+ o+x",
            Kind::Among,
            SATURATED,
            3,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the prefills over beside one under a doubt; one of each taken",
            "v+ s+",
            "u+",
            Kind::Among,
            SATURATED,
            3,
        ),
        Gateway::NoWorker,
        Parted::OwnShed,
    ),
    parted(
        row(
            "a pick refused; the rule's one decode is over and pairs with no prefill",
            "o+",
            "o- v+x",
            Kind::Among,
            PARTNERED,
            5,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    // A first decision that sheds with no endpoint: the rule is not asked.
    row(
        "the one decode is not ready; the rule is not asked",
        "o+",
        "u+",
        Kind::NotAsked,
        HEALTHY,
        0,
    ),
    parted(
        row(
            "the one prefill is over and under a doubt; no prefill taken",
            "w-",
            "o+",
            Kind::NotAsked,
            HEALTHY,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the one decode is over and under a doubt; the prefill, no decode",
            "o+",
            "w-",
            Kind::NotAsked,
            HEALTHY,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the paired prefill over beside an open one that pairs with no decode; no prefill taken",
            "v- o-x",
            "o+",
            Kind::NotAsked,
            PARTNERED,
            0,
        ),
        Gateway::Sheds,
        Parted::OwnShed,
    ),
    parted(
        row(
            "the partner under a doubt beside an idle decode apart; the prefill and the partner",
            "o+",
            "s+ o-x",
            Kind::NotAsked,
            HEALTHY,
            0,
        ),
        Gateway::Pairs,
        Parted::NotAsked,
    ),
];

/// One worker of a row.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Stated {
    /// `o`, `v`, `u`, `s`, `m`, `w` or `h`.
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
        if matches!(self.state, 'v' | 'w') {
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
            matches!(state, Some('o' | 'v' | 'u' | 's' | 'm' | 'w' | 'h'))
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
        .filter(|(_, worker)| matches!(worker.state, 'm' | 'h'));
    masked.map(|(place, _)| place).collect()
}

/// The places of a list's workers that a hold of their host stands on, as
/// the host states them beside the list.
pub(in super::super) fn holds(list: &[Stated]) -> Vec<u32> {
    let held = (0..).zip(list).filter(|(_, worker)| worker.state == 'h');
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
    let answer = shed_of(engine.place(&request));
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
/// The rule takes no prefill, and the shed was made over the prefills. Or
/// the rule takes a prefill and no decode, and its prefills let the shed
/// stand: one of them can take the request and the shed was made over the
/// decodes, or none can, one is ready and every ready one is over its
/// thresholds. No decision among the rule's workers can be made there.
/// The rule is asked as before: of the prefills alone where it takes none
/// of them, and of both lists else.
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

/// A rule that takes no worker of a list is told by name in every other
/// row: the request is shed with no endpoint, under `admits`. So it is
/// where the shed was made over the decodes and the rule takes no prefill,
/// where no prefill the rule takes is ready, where an idle decode that
/// pairs with no prefill keeps the decode list from being wholly over its
/// thresholds, and behind a first decision that picked.
///
/// The gateway's own selection answers that it has no worker in these
/// rows, but for two. The rule takes only prefills over their thresholds
/// and no decode, behind a first decision that picked or that said no
/// list: that selection answers its overload shed ([`Parted::NoWord`]).
#[test]
fn a_rule_that_takes_no_worker_of_a_list_is_told_where_no_shed_stands() {
    held(Kind::Told);
}

/// A rule that takes a worker of each list is answered by the call's next
/// decision, made among them: its shed as saturated where they are over
/// their thresholds, and its shed by the health filter where they are not
/// ready. That decision sheds by its own stages, as a first decision does
/// ([`Parted::OwnShed`]).
#[test]
fn a_rule_that_takes_a_worker_of_each_list_is_answered_by_the_next_decision() {
    held(Kind::Among);
}

/// A first decision that sheds the request with no endpoint is the call's
/// answer, and the rule is not asked: a decision none of whose sides is
/// ready learns nothing of the rule. So it is where that decision names
/// the health stage or `partnered` over a list the gateway's own selection
/// reads as wholly over its thresholds ([`Parted::OwnShed`]), and where
/// the rule takes workers under a doubt alone of a list
/// ([`Parted::NotAsked`]).
#[test]
fn a_rule_is_not_asked_behind_a_first_decision_that_sheds_with_no_endpoint() {
    held(Kind::NotAsked);
}

/// A pair that states no rule keeps its first decision's shed, whatever
/// stage it names ([`Parted::OwnShed`]).
#[test]
fn a_pair_with_no_rule_keeps_the_shed_of_its_first_decision() {
    held(Kind::Unruled);
}

/// The table's word of a row that parts from the gateway's own selection
/// is held to the row's kind. No row where the shed stands says that it
/// parts. A row told under the rule's name parts where that selection
/// sheds for overload and no decision said a list ([`Parted::NoWord`]). A
/// row whose rule is not asked parts where that selection's fallback
/// pairs ([`Parted::NotAsked`]). Every other row that parts does so by a
/// decision's own shed ([`Parted::OwnShed`]). What that selection answers
/// is the table's word, run outside this crate ([`ROWS`]): a row that
/// says nothing of it is not held here.
#[test]
fn a_row_parts_from_the_gateway_under_the_open_answer_of_its_kind() {
    let fits = |row: &Row| match (row.kind, row.parted) {
        (_, None) => true,
        (Kind::Stands, Some(_)) => false,
        (Kind::Told, Some(said)) => said == (Gateway::Sheds, Parted::NoWord),
        (Kind::NotAsked, Some((Gateway::Pairs, under))) => under == Parted::NotAsked,
        (Kind::Among | Kind::NotAsked | Kind::Unruled, Some((_, under))) => {
            under == Parted::OwnShed
        }
    };
    let misplaced: Vec<_> = ROWS.iter().filter(|row| !fits(row)).collect();
    assert!(misplaced.is_empty(), "{misplaced:#?}");
}

/// Under a binding that steers to a list wholly over its thresholds, a
/// decision says no list of a shed, and a rule that takes no prefill is
/// told by name. The one prefill is over its thresholds and the one decode
/// is not ready, so the binding has no pair to steer to, and the first
/// decision sheds the request as saturated. Under a binding that sheds the
/// same request keeps that shed.
///
/// The gateway's own selection sheds for overload under its shed option
/// alone (`failure_from`,
/// `model_gateway/src/routers/common/placement.rs:1710-1722`, on main
/// `56773d28d`).
#[test]
fn a_binding_that_steers_says_no_list_of_its_shed() {
    assert_eq!(
        asked_under(round_robin(false), "v-", "u+", true),
        (ADMITS, 1)
    );
    assert_eq!(
        asked_under(round_robin(true), "v-", "u+", true),
        (SATURATED, 1)
    );
}

/// A decision's word of the list it shed over is that decision's alone. A
/// first call's decision sheds over the prefills, and that shed stands. A
/// second call on the same thread sheds as saturated and says no list: one
/// of its prefills is under its thresholds and under a liveness doubt, and
/// no decode is ready. Its rule takes no prefill, and the request is shed
/// under the rule's name. With the word of the call before, which named
/// the prefills, it would keep its shed.
#[test]
fn a_decisions_word_of_its_shed_is_not_read_by_the_next_call() {
    let engine = round_robin(true);
    assert_eq!(asked_under(engine, "v- v-", "o+", true), (SATURATED, 2));
    assert_eq!(asked_under(engine, "v- s-", "u+", true), (ADMITS, 2));
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
