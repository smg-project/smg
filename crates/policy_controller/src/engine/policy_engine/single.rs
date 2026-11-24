//! A single request's whole decision in a gateway's own process.
//!
//! **The short path, in front** ([`alone`]). Where the document decides the
//! request's place by one pick of its picker
//! ([`Place::Alone`](crate::engine::Place::Alone)), the decision is that
//! pick: the picker's own rule over the list, with one read of each worker
//! the rule asks. A rotation reads the worker its turn lands on, and a scan
//! reads the list in order up to the first eligible worker. It writes no
//! record and takes no lock. It answers a plain request ([`Plain`]) over a
//! list where the rule finds an eligible worker, and it reads no fact of a
//! request: the engine asks it before the request is made, so such a
//! request is never made.
//!
//! **A key's rows, between** ([`keys::decide`]). A keyed request that is
//! made is decided by the rows its key names: under a sessions rule its
//! key's pin, which is one worker's record, and under a ring the workers
//! its key reaches on the ring its caller lent. It answers where those rows
//! say what the whole pass would, and nothing where they do not.
//!
//! **The whole pass, behind them** ([`decide`]). Every other request is made
//! and put to the engine's whole decision over records written for it from
//! the gateway's own data ([`records::decide`]): where no key's rows
//! answered it, a retry that names the workers it tried, a list the
//! picker's turns find no eligible worker of, and a place whose profile
//! scores. The binding's rule for a pool every worker of which is vetoed,
//! and the test of a tried worker, so have one home.
//!
//! **A hard demand** ([`Demands`](crate::engine::Demands)). A request that
//! states one is held to the places of its list whose worker meets it, by
//! what the list's host said of each worker when the list's records were
//! made and by the rank the worker's object registered
//! ([`PoolRecords::meeting`]), among the places its host holds it to.
//! It is then decided as a request held to places is, so a key's pin on
//! a worker that does not meet the demand is not followed. A list the
//! demands leave nobody is shed with no endpoint, under the stage of the
//! demand that took its last worker. A pair that states a demand is
//! held to it a side at a time
//! ([`PairLists::decide`](super::pair::PairLists::decide)).
//!
//! **A retry** names the workers it tried ([`RequestFacts::tried`]). Over
//! a list where an engine holds several workers, the ranks of a
//! data-parallel engine, it is decided with every worker of a tried
//! worker's engine named as tried ([`PoolRecords::tried_engines`]): a
//! failed attempt reached the engine its ranks share. So a stage that
//! passes a tried worker over passes its engine over, and hands the
//! engine again where no other worker is left. The keyed decision and the
//! whole pass read the same list. A tried worker the list no longer holds
//! names itself alone.
//!
//! **What is refused by name.** The list a host hands is its request's own:
//! every worker of it is dialed on a wire the request states. A list that is
//! not, and a subset hint, are declined under the stage that would have run
//! them.
//!
//! **A request held to places of its list**
//! ([`Constraints::places`](crate::engine::view::Constraints::places)) is
//! made and decided over the list's own records: it is not plain, the
//! keyed decision leaves it to the whole pass, and that pass opens with
//! the places. A pair whose request states places is declined ([`PLACES`]).

use std::{
    cell::Cell,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, runtime::RoutingState};

use super::{keys, overlap, Loaded, PoolMembers};
use crate::{
    engine::{
        decision::{Decision, Shed, ShedReason, Trace, SUBSET_HINT, WIRES},
        decision_engine::{AlonePick, DecideOutcome, Leased, Plain, Request, RequestMaker},
        demands::UnmetDemand,
        framework::Engine,
        lease_events,
        records::{self, PoolRecords},
        sticky_sessions::SessionVerdict,
        view::{Constraints, Phase, RequestFacts, Session, WorkerId},
    },
    stages::Lifecycle,
    tracking::lease::{EventKind, EventSet, Lease, LeaseId, LeaseKind, Side, DISPATCH_TTL},
};

thread_local! {
    /// The calling thread's own list of the places that meet a request's
    /// demands, taken for one decision and put back behind it: such a
    /// request allocates nothing once the list has grown to its pool.
    static MEETING: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };

    /// The calling thread's own list of the workers a retry is decided to
    /// have tried, where an engine of its list holds several
    /// ([`retried`]): taken for one decision and put back behind it, as
    /// [`MEETING`] is. A pair's retry takes it the same way.
    pub(super) static TRIED: Cell<Vec<WorkerId>> = const { Cell::new(Vec::new()) };
}

/// The stage a pair whose request states places is declined under
/// ([`Constraints::places`](crate::engine::view::Constraints::places)):
/// one set of places cannot hold a pair's two lists. An encode request
/// that states places of its own is declined under it too.
pub(super) use crate::engine::decision::PLACES;

/// The crate's word, in a shed's `emptied_by`, for a pool no stage emptied.
const NO_STAGE: &str = "none";

/// The short path: the place in `members`, the engine's value of the list
/// its host lent, of the worker picked for a plain single request of
/// `model`, and the profile whose picker picked it ([`AlonePick`]). `None`
/// where the request takes the whole pass ([`decide`]): its place is not
/// decided by one pick of a picker, a worker of the list is not dialed on a
/// wire it states, or the picker's rule finds no eligible worker. Its
/// caller asks no list's value for a request of another phase.
///
/// It ends in the place picked. It reads no fact of a request, so none is
/// made ahead of it, and nothing a host is handed of a pick is built here.
/// Under a document that takes no word on a lease the place is the whole
/// answer. Under any other the pick is noted ([`noted`]).
pub(super) fn alone(
    own: Loaded<'_, '_>,
    model: &str,
    plain: Plain,
    members: &PoolMembers,
) -> Option<AlonePick> {
    debug_assert_eq!(plain.phase, Phase::Single, "a single request's");
    // The whole pass declines such a list by name.
    if !plain.wires.is_empty() && !plain.wires.contains(members.wires()) {
        return None;
    }
    let profile = members.alone(&own.document.spec, model)?;
    let position = own.pick_over(profile, members, |state: RoutingState| state.eligible())?;
    Some(AlonePick {
        place: u32::try_from(position).ok()?,
        profile: profile.0,
        generation: own.document.spec.generation(),
    })
}

/// The short path's pick under a document that takes a word on a lease,
/// as its host takes it: the decision that names the worker at the place
/// picked, alone ([`AlonePick::decision`]), and its lease, issued here as a
/// whole pass's is ([`issued`]). The lease states the size `plain` states,
/// the two lengths of its `text`, characters and bytes, and no session, and
/// is dated at `now` on the engine's clock. The issue is told where a
/// stage takes it, under a catch of its own as a whole pass's is. `None`
/// where the list holds no worker at the place.
pub(super) fn noted(
    own: Loaded<'_, '_>,
    pick: AlonePick,
    members: &PoolMembers,
    plain: Plain,
    text: (u32, u32),
    now: Instant,
) -> Option<DecideOutcome> {
    let position = pick.place as usize;
    let worker = members.workers().get(position)?;
    // The list's records name each worker once a list: a pick reads the id
    // there and hashes no address.
    let document = own.document;
    let records = members.records(&document.spec, &own.clock);
    let id = *records.ids().get(position)?;
    let dp_rank = records::worker_rank(worker.as_ref());
    let lease = Lease {
        id: LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        booked: id,
        generation: pick.generation,
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        issued_micros: own.clock.micros(now),
        tokens: plain.prompt_tokens,
        chars: text.0,
        bytes: text.1,
    };
    let decided = DecideOutcome {
        decision: pick.decision(id, dp_rank),
        lease: Some(Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: Session::NONE,
    };
    if document.hears.contains(EventKind::Issued) {
        let stages = document.spec.engine();
        own.word_caught(|| stages.on_decision_issued(&lease, &decided.decision));
    }
    Some(decided)
}

/// How a made request's whole decision runs over its pool's records. The
/// caller of [`decide`] states the form where it is written, so [`decide`]
/// holds no test of it. One form runs the binding's live profile alone
/// ([`WithoutShadow`]): one call of [`records::decide`] and nothing else.
/// The other runs the binding's shadow profile behind it
/// ([`super::with_shadow`]).
pub(super) trait WholeDecision {
    /// The engine's whole decision over `records`, the records of
    /// `workers`, for the request `facts` state. It is run by reference:
    /// the lookup that names a request's holders takes a decision it may
    /// ask again ([`overlap::decide`]), and asks it once.
    fn decide(
        &mut self,
        engine: &Engine,
        records: &PoolRecords,
        workers: &[Arc<dyn Worker>],
        live: &records::Live<'_>,
        facts: &mut RequestFacts<'_>,
        cx: &Constraints<'_>,
    ) -> Decision;
}

/// The whole decision by the live profile alone.
pub(super) struct WithoutShadow;

impl WholeDecision for WithoutShadow {
    #[inline(always)]
    fn decide(
        &mut self,
        engine: &Engine,
        records: &PoolRecords,
        workers: &[Arc<dyn Worker>],
        live: &records::Live<'_>,
        facts: &mut RequestFacts<'_>,
        cx: &Constraints<'_>,
    ) -> Decision {
        records::decide(engine, records, workers, live, facts, cx)
    }
}

/// One made request's decision over `members`, the list its host handed,
/// at `now`: by the rows its key names where they answer
/// ([`keys::decide`]), and else the engine's whole decision over records
/// written for the request from the list's workers, run as `whole` states
/// ([`WholeDecision`]). Each worker's report is read from `loads`, the
/// host's load snapshot for this decision. A request that states what no
/// stage here runs is declined under that stage's name.
///
/// `maker` is asked for the request's entry in its host's token tree of
/// sent prefixes ([`RequestMaker::token_key`]) where the host lends the
/// request's place one ([`overlap::decide`]). It is asked of the maker's
/// own type and no argument carries the entry: a maker that names none
/// costs a decision nothing. The decision is compiled once for each maker
/// type, as the engine's entry that calls it is, and once more for the
/// trait object through which a copy that carries signal cells asks
/// ([`super::signals::decide`]).
///
/// A request that states a hard demand is decided under its constraints
/// held to the workers that meet it ([`held`]). A list the demands leave
/// nobody is shed with no endpoint under the stage of the demand that took
/// its last worker ([`UnmetDemand::stage`]), as a pool its request's
/// demands emptied is shed over a store's rows.
///
/// A retry over a list where an engine holds several workers is decided
/// with every worker of the engines it tried named as tried ([`retried`]).
/// A first attempt pays one test for it, and a retry over any other list
/// one more.
pub(super) fn decide(
    own: Loaded<'_, '_>,
    request: &mut Request<'_>,
    members: &PoolMembers,
    maker: &(impl RequestMaker + ?Sized),
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    whole: impl WholeDecision,
) -> Decision {
    let spec = &own.document.spec;
    let cx = request.constraints;
    if cx.subset_hint.is_some() {
        return declined(own, request, SUBSET_HINT);
    }
    if !cx.wire_modes.is_empty() && !cx.wire_modes.contains(members.wires()) {
        return declined(own, request, WIRES);
    }
    let records = members.records(spec, &own.clock);
    // The calling thread's list of places, where the request states a
    // demand: lent for this decision and given back behind it. The rest of
    // the decision stands once and takes its constraints by reference.
    let mut meeting = None;
    let narrowed;
    let cx = if cx.demands.is_none() {
        &cx
    } else {
        match held(records, cx, request.facts.phase, &mut meeting) {
            Ok(held) => {
                narrowed = held;
                &narrowed
            }
            Err(unmet) => {
                MEETING.set(meeting.unwrap_or_default());
                return shed(own, request, ShedReason::NoEndpoints, unmet.stage());
            }
        }
    };
    let facts = &mut request.facts;
    // One value for every record of the decision: its instant, the host's
    // snapshot as the host holds it, the name the key's requests are
    // counted under on a worker, as the request's maker states it, for a
    // stage that reads that count, and the list's own word, the one a
    // pick alone turns ([`PolicyEngine::pick`]): a draw of this decision
    // takes the list's next turn. The maker is asked under a document
    // that reads the count and under no other.
    let mut name = [0; Session::TEXT_LEN];
    let key = if spec.engine().needs().session_in_flight {
        maker.session_counted_under().name(facts, &mut name)
    } else {
        None
    };
    let live =
        records::Live::at(spec.engine(), &own.clock, now, loads, key).turning(&members.rotation.0);
    // A retry leaves every worker of the engines it tried: over a list
    // where an engine holds several, its facts name them all.
    let decision = if facts.tried.is_empty() || !records.has_ranks() {
        by_facts::<false>(own, members, records, &live, facts, cx, maker, whole)
    } else {
        retried(own, members, records, &live, facts, cx, maker, whole)
    };
    if let Some(meeting) = meeting {
        MEETING.set(meeting);
    }
    decision
}

/// [`decide`]'s decision by the facts handed, which are the request's own
/// or a retry's ([`retried`]): by the rows the key names where they answer,
/// and else the whole decision, run as `whole` states.
///
/// A pin the keyed entry followed is the answer either way. Where its rule
/// keeps it from the profile ([`lease_events::stages_hear`]) it is returned
/// as it stands. Where it is the profile's own it is handed through the
/// lookup that names a request's holders ([`handed_on`]), so its prompt is
/// handed on as behind any pick, and by the rule's word a followed pin is
/// recorded the same on both its paths.
///
/// In line at both its callers: a first attempt calls nothing for it.
/// `RETRY` says which caller's instance this is, [`retried`]'s or
/// [`decide`]'s. Each instance so has one caller, and the compiler keeps
/// what a first attempt runs in line there. With one instance for both
/// callers a first attempt's whole decision was called, 30 instructions a
/// decision more.
///
/// `maker` is the request's own, a retry's too: the lookup that names a
/// request's holders asks it for the request's entry in a lent token tree,
/// and which target of its pick the request's caller sends to.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "a decision's inputs, the request's maker and the whole decision"
)]
fn by_facts<const RETRY: bool>(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    live: &records::Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
    maker: &(impl RequestMaker + ?Sized),
    mut whole: impl WholeDecision,
) -> Decision {
    let spec = &own.document.spec;
    let (engine, workers) = (spec.engine(), members.workers());
    // A keyed request is decided by the rows its key names, where they say
    // what the whole pass would: its key's pin, or its ring's walk.
    match keys::decide(engine, records, workers, live, facts, cx) {
        // A pin its rule keeps from the profile: no holder is named for it
        // and nothing is kept of where its prompt went.
        Some(keyed) if !lease_events::stages_hear(&keyed) => keyed,
        // A pin that is the profile's own goes through the same lookup with
        // its answer made.
        Some(keyed) => handed_on(own, members, records, facts, maker, keyed),
        // A profile whose stages read who holds the prompt is named its
        // rows from the gateway's own data, at the decision's instant the
        // keyed entry wrote, and its pick is recorded behind it; any other
        // passes straight to the whole pass.
        None => overlap::decide(
            &own.affinity,
            members.affinity(),
            spec,
            workers,
            records.ids(),
            facts,
            |page| maker.token_key(page),
            |target| maker.sends_to(target),
            |facts| whole.decide(engine, records, workers, live, facts, cx),
        ),
    }
}

/// A pin the keyed entry followed that is the profile's own
/// ([`lease_events::stages_hear`]), handed through the lookup that names a
/// request's holders ([`overlap::decide`]) with `keyed`, its answer, in the
/// whole decision's place. The lookup returns that answer as it was handed:
/// no stage is asked again and no row is written for a decision's pass.
/// Where the profile's stages read who holds the prompt, the lookup does
/// behind this answer what it does behind any pick. It states in the pick
/// what the pin's worker holds of the request, from the rows it names. It
/// hands the prompt to the record of sent prefixes, or to the host's token
/// tree by the entry and the target its maker names. It may read its list's
/// table again, as any decision that finds the table expired, and a tree's
/// match marks the path it walked as used. Where no binding of the phase
/// reads affinity, the lookup is one test.
///
/// Out of line: its code stays off the path of a request with no pin, and
/// of a pin its rule keeps from the profile.
#[inline(never)]
fn handed_on(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    facts: &mut RequestFacts<'_>,
    maker: &(impl RequestMaker + ?Sized),
    keyed: Decision,
) -> Decision {
    overlap::decide(
        &own.affinity,
        members.affinity(),
        &own.document.spec,
        members.workers(),
        records.ids(),
        facts,
        |page| maker.token_key(page),
        |target| maker.sends_to(target),
        |_| keyed,
    )
}

/// A retry's decision over a list where an engine holds several workers.
/// The request is decided ([`by_facts`]) by a copy of its facts whose tried
/// workers are its own and every other worker of their engines
/// ([`PoolRecords::tried_engines`]). The key's rows, the whole pass and a
/// binding's shadow profile so read one list, and the request's own list
/// stays as its host made it. The decision's instant is written back into
/// `facts`: the request's lease is dated by it. The request's maker goes
/// with the copy, so a retry is decided inside a lent token tree as a
/// first attempt is.
///
/// The workers stand in the calling thread's own list ([`TRIED`]), so a
/// retry allocates nothing once the list has grown. Out of line and cold:
/// a first attempt's path, and a retry's over a list with no such engine,
/// hold one test of this and nothing else.
#[cold]
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "a decision's inputs, the request's maker and the whole decision"
)]
fn retried(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    live: &records::Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
    maker: &(impl RequestMaker + ?Sized),
    whole: impl WholeDecision,
) -> Decision {
    let mut tried = TRIED.take();
    records.tried_engines(facts.tried, &mut tried);
    let mut widened = RequestFacts {
        tried: &tried,
        ..*facts
    };
    let decision = by_facts::<true>(own, members, records, live, &mut widened, cx, maker, whole);
    facts.now_micros = widened.now_micros;
    TRIED.set(tried);
    decision
}

/// The constraints a request that states a hard demand is decided under
/// ([`held_in`]). The places stand in `meeting`, the calling thread's own
/// list, taken here: its caller puts it back ([`MEETING`]). Out of line:
/// its code stays off the path of a request that states no demand.
///
/// A request stands alone here, so its list is held to what a worker of
/// its `phase`'s role holds of the demands ([`Demands::held_by`]). A
/// prefill worker holds the input alone, with no decode beside it to hold
/// the rest: a host states both bounds whatever the phase.
///
/// [`Demands::held_by`]: crate::engine::demands::Demands::held_by
#[inline(never)]
fn held<'a>(
    records: &PoolRecords,
    cx: Constraints<'a>,
    phase: Phase,
    meeting: &'a mut Option<Vec<u32>>,
) -> Result<Constraints<'a>, UnmetDemand> {
    let demands = cx.demands.held_by(phase.role());
    let cx = Constraints { demands, ..cx };
    // Where the role holds none of them, no row is judged.
    if demands.is_none() {
        return Ok(cx);
    }
    held_in(records, cx, meeting.insert(MEETING.take()))
}

/// The constraints a request that states a hard demand is decided under:
/// its own, held to the places of the list whose worker meets its demands,
/// among the places its host holds it to ([`PoolRecords::meeting`]). They
/// are the decision's places ([`Constraints::places`]), so the request is
/// answered no other worker. Where every worker the request may take meets
/// the demands, it is held to no more than its host said. Where none does,
/// the answer is the demand that took the last worker.
///
/// The places are written into `kept`, a list its caller keeps: a single
/// request's is its thread's ([`held`]), and each side of a pair has its
/// own, judged by what that side's role is held to ([`super::pair`]).
pub(super) fn held_in<'a>(
    records: &PoolRecords,
    cx: Constraints<'a>,
    kept: &'a mut Vec<u32>,
) -> Result<Constraints<'a>, UnmetDemand> {
    match records.meeting(cx.demands, cx.places, kept) {
        None => Ok(cx),
        Some(unmet) if kept.is_empty() => Err(unmet),
        Some(_) => Ok(Constraints {
            places: Some(kept),
            ..cx
        }),
    }
}

/// The answer for a request whose host handed no list: a pool of no worker,
/// which no stage emptied.
pub(super) fn over_none(own: Loaded<'_, '_>, request: &Request<'_>) -> Decision {
    shed(own, request, ShedReason::NoEndpoints, NO_STAGE)
}

/// The answer for a request that states what no stage of this engine runs,
/// under the name of the stage that would have run it. Cold: [`decide`]'s
/// tests are then predicted to pass. Without it, one test more in front
/// of the whole pass kept the pass from being inlined, 43 instructions a
/// decision.
#[cold]
pub(super) fn declined(
    own: Loaded<'_, '_>,
    request: &Request<'_>,
    stage: &'static str,
) -> Decision {
    shed(own, request, ShedReason::PolicyDeclined, stage)
}

fn shed(
    own: Loaded<'_, '_>,
    request: &Request<'_>,
    reason: ShedReason,
    emptied_by: &'static str,
) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace: Trace {
            seed: request.facts.seed,
            generation: own.document.spec.generation(),
            ..Trace::default()
        },
    })
}

/// The decision as its host takes it. Under a document that takes a word
/// on a lease a pick is issued before the answer leaves: its lease takes
/// the next id, books the pick's first target and is dated at the
/// decision's instant, and the stages that take the issue hear of it here,
/// on the calling thread and under a catch of its own
/// ([`super::PolicyEngine::word_caught`]). A stage that panics at the issue
/// is held and counted, the stages behind it do not hear it, and the pick
/// and its lease are answered all the same: the guard tells the lease's
/// dispatch and its end as any lease's. From the answer the lease is its
/// guard's: nothing of it is filed.
///
/// A pick the binding's sessions rule keeps from the profile is issued
/// too, and no stage hears its issue ([`lease_events::stages_hear`]).
///
/// Under a document that takes no word a pick has no lease: no id is
/// taken, and its guard owes the engine nothing.
pub(super) fn issued(
    own: Loaded<'_, '_>,
    decision: Decision,
    request: &Request<'_>,
    now: Instant,
) -> DecideOutcome {
    let facts = &request.facts;
    let document = own.document;
    let leased = document.hears.intersects(EventSet::LEASE);
    let noted = decision.pick().filter(|_| leased);
    let lease = noted.and_then(|pick| {
        let first = pick.targets.first()?;
        Some(Lease {
            id: LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1),
            kind: LeaseKind::Credit,
            side: Side::Single,
            target_index: 0,
            booked: first.id,
            generation: pick.trace.generation,
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: facts.now_micros,
            tokens: facts.prompt_tokens,
            chars: facts.prompt_chars,
            bytes: facts.prompt_bytes,
        })
    });
    let issue = document.hears.contains(EventKind::Issued);
    if let Some(lease) = lease.as_ref().filter(|_| issue) {
        // Asked behind the document's own word: a document whose stages
        // take no issue reads nothing of the pick here.
        if lease_events::stages_hear(&decision) {
            own.word_caught(|| document.spec.engine().on_decision_issued(lease, &decision));
        }
    }
    DecideOutcome {
        decision,
        lease: lease.map(|lease| Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: Session::NONE,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{borrow::Cow, cell::Cell, collections::BTreeSet, sync::Arc, time::Duration};

    use kv_index::{TenantId, TokenTree};
    use parking_lot::Mutex;
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        load::state::one_rank_load,
        model_card::ModelCard,
        test_support::{no_health_check, ready_worker},
        worker::WorkerStatus,
    };

    use super::{super::PolicyEngine, *};
    use crate::{
        document::spec::{
            AppliedSpec, Assignment, PolicySpec, RulePlaced, ScoreStage, ShadowSpec,
            WarmupSliceParams,
        },
        engine::{
            decision::{LiftedVeto, Overlap, PlacedBy, Target, Targets, MAX_TARGETS},
            decision_engine::{Decided, DecisionEngine, DecisionRefused, TokenKey, WorkerLists},
            demands::{DemandFacts, Demands},
            framework::{Picker, Place, PoolWord, TieBand, DRAW_FROM},
            list_cell::tests::PublishedList,
            prompt_hashes::HashSeeds,
            records::worker_id,
            view::{
                AffinityFacts, Candidate, Constraints, FixedFacts, Identity, RequestFacts,
                SessionCountedUnder, WireModes, WorkerId,
            },
        },
        policies::random::Random,
        stages::registry::StageRegistry,
        store::GatewayId,
        tracking::{
            lease::{Membership, Outcome, ReleaseReason, TransportClass},
            DecisionGuard,
        },
    };

    pub(crate) const MODEL: &str = "model-a";

    fn preset(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
    }

    /// The preset `name` with no filter in any of its profiles: a valid
    /// document that weighs no worker's health.
    fn unfiltered(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
    }

    pub(crate) fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    /// A host that keeps a token tree of the prompts it sent to its workers
    /// and lends it. It keeps no index of cache events.
    struct Lends(Arc<TokenTree>);

    impl overlap::CacheIndexes for Lends {
        fn index(&self, _: &str) -> Option<Arc<dyn overlap::CacheIndex>> {
            None
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            None
        }

        fn token_tree(&self, _: &str, _: Phase) -> Option<Arc<TokenTree>> {
            Some(Arc::clone(&self.0))
        }
    }

    /// What a lending host's document holds beyond its preset.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Also {
        /// Nothing.
        Nothing,
        /// The binding names a copy of its profile as its shadow.
        Shadow,
        /// The profile costs each worker by a named signal too. No report
        /// states it, so every worker costs the same.
        Signal,
    }

    /// The engine of a preset under a host that lends its token tree, a
    /// page of sixteen tokens, and the host's tree. `also` is what the
    /// document holds beyond the preset.
    pub(crate) fn lent(name: &str, also: Also) -> (PolicyEngine, Arc<TokenTree>) {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        let live = document.bindings[0].profile.clone();
        let live = live.expect("a binding of one profile");
        match also {
            Also::Nothing => {}
            Also::Shadow => {
                let twin = document.profiles[&live].clone();
                document.profiles.insert("twin".into(), twin);
                document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
            }
            Also::Signal => {
                let tokens = serde_json::from_str(r#"{"unit":"tokens"}"#).expect("a signal");
                document.signals.insert("held".into(), tokens);
                let cost =
                    r#"{"least-cost":{"terms":{"signal.held":1},"unreported":"missing-as-zero"}}"#;
                let profile = document.profiles.get_mut(&live);
                let profile = profile.expect("the binding's profile");
                profile
                    .score
                    .push(serde_json::from_str(cost).expect("a scorer"));
            }
        }
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        lending(spec)
    }

    /// The engine of `spec` under a host that lends its token tree, a page
    /// of sixteen tokens, and the host's tree: for a case that makes its
    /// own document.
    pub(crate) fn lending(spec: AppliedSpec) -> (PolicyEngine, Arc<TokenTree>) {
        let engine = PolicyEngine::new(spec);
        let tree = Arc::new(TokenTree::with_config(16, Default::default()));
        engine
            .affinity()
            .set_indexes(Arc::new(Lends(Arc::clone(&tree))));
        (engine, tree)
    }

    pub(crate) fn request<'a>() -> Request<'a> {
        Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: 7,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    /// A request stated ahead, with the lists its host hands for it, and how
    /// many times the engine made it.
    pub(crate) struct Asks<'a> {
        pub(crate) request: Request<'a>,
        pub(crate) members: WorkerLists<'a>,
        pub(crate) loads: Option<Arc<LoadSnapshot>>,
        pub(crate) made: Cell<u32>,
        /// The token ids the host names as the request's entry in a token
        /// tree, and how many times the engine asked for the entry.
        pub(crate) key: Option<&'a [u32]>,
        pub(crate) keyed: Cell<u32>,
    }

    impl<'a> Asks<'a> {
        pub(crate) fn of(request: Request<'a>, members: Option<&'a PublishedList>) -> Self {
            let lists = match members {
                Some(list) => WorkerLists::Single(list.lent()),
                None => WorkerLists::None,
            };
            Self::over(request, lists)
        }

        fn over(request: Request<'a>, members: WorkerLists<'a>) -> Self {
            Self {
                request,
                members,
                loads: None,
                made: Cell::new(0),
                key: None,
                keyed: Cell::new(0),
            }
        }
    }

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            self.request.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.made.set(self.made.get() + 1);
            self.request
        }

        fn plain(&self) -> Option<Plain> {
            self.request.plain()
        }

        fn prompt_chars(&self) -> u32 {
            self.request.prompt_chars()
        }

        fn prompt_bytes(&self) -> u32 {
            self.request.prompt_bytes()
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.members
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.loads.as_ref()
        }

        fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
            self.keyed.set(self.keyed.get() + 1);
            let tokens = Cow::Borrowed(self.key?);
            Some(TokenKey { tokens, marker: 0 })
        }
    }

    /// What the one entry answers a request stated ahead.
    pub(crate) fn answered(engine: &PolicyEngine, asks: &Asks<'_>) -> Decided {
        let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
        decided.expect("an answer")
    }

    /// The whole decision the entry answers: no place.
    pub(crate) fn decided(engine: &PolicyEngine, asks: &Asks<'_>) -> DecideOutcome {
        answered(engine, asks).whole().expect("a whole decision").0
    }

    /// The place the entry answers: no whole decision.
    fn placed(engine: &PolicyEngine, asks: &Asks<'_>) -> AlonePick {
        match answered(engine, asks) {
            Decided::Alone(pick) => pick,
            Decided::Whole(decided, _) => panic!("a place, not {decided:?}"),
        }
    }

    /// Round robin's plain single request is one turn of the list's
    /// rotation, answered as the place in the list the turn lands on: no
    /// pass over the pool ran, no request was made, and no lease took an
    /// id, since no stage of the document takes a word of one. The decision
    /// a host writes of the place names that worker alone.
    #[test]
    fn round_robin_is_one_turn_of_the_list_answered_as_a_place() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let mut sized = request();
        sized.facts.prompt_tokens = 321;
        sized.facts.session_key = Identity::plain("a-key");
        let asks = Asks::of(sized, Some(&members));
        let mut seen = [0_u32; 4];
        for _ in 0..8 {
            let pick = placed(&engine, &asks);
            assert_eq!(pick.generation, 1);
            seen[pick.place as usize] += 1;
            let id = worker_id(list[pick.place as usize].as_ref());
            let decision = pick.decision(id, 3);
            let named = decision.pick().expect("a pick");
            assert_eq!(named.targets.len(), 1);
            assert!(named.trace.drew_first(), "no pass over the pool ran");
            assert_eq!(named.trace.seed, 0, "no seed was drawn");
            let target = named.targets.first().expect("the target");
            assert_eq!(
                (target.position, target.id, target.dp_rank),
                (pick.place, id, 3)
            );
        }
        assert_eq!(seen, [2; 4], "every worker takes its turn");
        assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    }

    /// Passthrough's plain single request is answered as a place: the first
    /// worker of the list its host's test admits. That is the gateway's
    /// classic rule, the position of the first worker whose routing state
    /// the selection admits (`model_gateway/src/policies/passthrough.rs:48`
    /// at main `882be6f4f`). No pass over the pool ran, no request was made
    /// and no lease took an id. A list that admits no worker is the whole
    /// decision's: the request is made once.
    #[test]
    fn passthrough_is_the_first_admitted_worker_answered_as_a_place() {
        let engine = preset("passthrough");
        for workers in [1, 2, 8, 64] {
            let list = pool(workers);
            let members = PublishedList::new(Arc::clone(&list));
            let asks = Asks::of(request(), Some(&members));
            for refused in 0..workers.min(12) {
                // Classic's rule, asked of the same worker objects.
                let classic = list.iter().position(|w| w.routing_state().eligible());
                assert_eq!(classic, Some(refused));
                let place = placed(&engine, &asks).place;
                assert_eq!(place as usize, refused, "{workers} workers");
                // The first is refused by its status alone, the others by overload.
                list[refused].set_overloaded(refused > 0);
                list[0].set_status(WorkerStatus::NotReady);
            }
            assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
            if workers <= 12 {
                // The whole decision's rule, not the first worker: a worker
                // that is not ready is never named, and the overload veto
                // is lifted off the workers it alone refuses.
                match decided(&engine, &asks).decision {
                    Decision::Shed(shed) => assert_eq!((workers, shed.emptied_by), (1, "healthy")),
                    Decision::Pick(pick) => {
                        assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
                        assert!(pick.targets.iter().all(|target| target.position > 0));
                    }
                    Decision::Queue(_) => panic!("no queue is planned"),
                }
                assert_eq!(asks.made.get(), 1, "no worker is admitted: made once");
            }
        }
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    }

    /// Under a document a stage of which takes a word on a lease, a plain
    /// request of a place one pick decides is still never made, and its
    /// pick is noted: a whole decision of one target under a lease that
    /// takes the next id, states the size the plain word states and no
    /// session, and is told to the stage that takes the issue.
    #[test]
    fn a_place_picked_under_a_document_that_takes_a_word_is_noted() {
        let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
        let place = engine.place_for(MODEL, Phase::Single);
        assert!(matches!(place, Place::Alone(_)), "{place:?}");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let mut sized = request();
        (sized.facts.prompt_tokens, sized.facts.prompt_chars) = (321, 77);
        sized.facts.prompt_bytes = 154;
        sized.facts.session_key = Identity::plain("a-key");
        let asks = Asks::of(sized, Some(&members));
        let (mut seen, mut last) = ([0_u32; 4], 0);
        for _ in 0..8 {
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            assert_eq!(pick.targets.len(), 1);
            assert!(pick.trace.drew_first(), "no pass over the pool ran");
            assert_eq!(pick.trace.seed, 0, "no seed was drawn");
            let target = pick.targets.first().expect("the target");
            let worker = &list[target.position as usize];
            assert_eq!(target.id, worker_id(worker.as_ref()));
            seen[target.position as usize] += 1;
            let lease = decided.lease.expect("a lease").lease;
            assert_eq!(
                (lease.side, lease.booked, lease.target_index),
                (Side::Single, target.id, 0)
            );
            assert_eq!((lease.tokens, lease.chars, lease.bytes), (321, 77, 154));
            assert_eq!(lease.session, Session::NONE);
            assert_eq!(lease.session_verdict, SessionVerdict::None);
            assert!(lease.id.0 > last, "each pick takes the next id");
            last = lease.id.0;
            assert_eq!(decided.decode_session, Session::NONE);
            assert_eq!(heard.lock().last(), Some(&("issued", target.id, 0)));
        }
        assert_eq!(seen, [2; 4], "every worker takes its turn");
        assert_eq!(heard.lock().len(), 8, "one issue a pick");
        assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
    }

    /// A request that is not plain, and a plain one the turn finds no
    /// worker for, is made once and put to the whole pass. Under a document
    /// that takes no word its pick has no lease either.
    #[test]
    fn a_request_the_one_pick_does_not_decide_is_made_once() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));

        // A retry is not plain: it is made, once a decision.
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let whole = decided(&engine, &asks);
        assert!(whole.decision.pick().is_some());
        assert!(whole.lease.is_none(), "no stage takes a word of it");
        assert_eq!(asks.made.get(), 1);
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

        // A plain request whose turns land on no eligible worker is made
        // for the whole pass, which answers by the binding's own rule.
        for worker in list.iter() {
            worker.set_overloaded(true);
        }
        let asks = Asks::of(request(), Some(&members));
        let _ = decided(&engine, &asks);
        assert_eq!(asks.made.get(), 1);
    }

    /// A program that keeps ids of its own states each worker's with its
    /// list's fixed facts. The list's records carry those ids whole, and
    /// a record's key stays the low half of its address's fingerprint. A
    /// whole decision names every target by the stated id, and passes
    /// over the worker a retry names by it. A list handed with nothing
    /// said names each worker by its address's fingerprint.
    #[test]
    fn a_list_names_each_worker_by_the_id_its_host_states() {
        use crate::engine::view::FixedFacts;

        // Ids past 32 bits: no half of one stands for it.
        let wide = |at: u64| WorkerId((1 << 40) + at);
        let engine = preset("round_robin");
        let list = pool(4);
        let place = |worker: &Arc<dyn Worker>| list.iter().position(|one| Arc::ptr_eq(one, worker));
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            id: place(worker).map(|at| wide(at as u64)),
            ..FixedFacts::default()
        });
        let records = engine
            .members_of(members.lent())
            .records(engine.spec(), &engine.clock);
        let stated: Vec<WorkerId> = (0..4).map(wide).collect();
        assert_eq!(records.ids(), stated);
        for (row, worker) in records.fixed().iter().zip(list.iter()) {
            assert_eq!(row.key, worker_id(worker.as_ref()).0 as u32);
        }

        let tried = [stated[1]];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let mut seen = [false; 4];
        for _ in 0..12 {
            let whole = decided(&engine, &asks);
            let pick = whole.decision.pick().expect("a pick");
            for target in pick.targets.iter() {
                assert_eq!(target.id, stated[target.position as usize]);
                assert_ne!(target.id, tried[0], "the tried one is no alternate");
            }
            let target = pick.targets.first().expect("a target");
            seen[target.position as usize] = true;
        }
        assert_eq!(seen, [true, false, true, true], "the tried one is passed");

        let unsaid = PublishedList::new(Arc::clone(&list));
        let prints: Vec<WorkerId> = list
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .collect();
        let unsaid = engine.members_of(unsaid.lent());
        let records = unsaid.records(engine.spec(), &engine.clock);
        assert_eq!(records.ids(), prints);
    }

    /// Zero is no worker's id: it reads as absent wherever an id travels,
    /// and a session's table keeps no pin on it. A program that states
    /// zero for a worker states none: the list's records name the worker
    /// by its address's fingerprint, and a session placed on that worker
    /// keeps its pin.
    #[test]
    fn a_list_names_a_worker_by_its_address_where_its_host_states_zero() {
        use crate::engine::view::FixedFacts;

        let engine = preset("manual");
        let list = pool(4);
        let members = PublishedList::with_facts(Arc::clone(&list), |_| FixedFacts {
            id: Some(WorkerId(0)),
            ..FixedFacts::default()
        });
        let prints: Vec<WorkerId> = list
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .collect();
        let records = engine
            .members_of(members.lent())
            .records(engine.spec(), &engine.clock);
        assert_eq!(records.ids(), prints);

        // The stock sessions rule: a key's first request is placed and
        // its dispatch pins the key, and the key's next request follows
        // the pin.
        let mut keyed = request();
        keyed.facts.session_key = Identity::hashed(41);
        let asks = Asks::of(keyed, Some(&members));
        let placed = served(&engine, &asks, Instant::now());
        assert_eq!(placed.id, prints[placed.position as usize]);
        let followed = decided(&engine, &asks).decision;
        let pick = followed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        let again = pick.targets.first().expect("a target");
        assert_eq!(again.id, placed.id);
    }

    /// The count a capped sessions rule reads stands on a worker's object
    /// under the name the request's maker states. A maker that says
    /// nothing is of a host whose own guard counts, and the count is read
    /// under the key's text. A maker of a host that counts by the
    /// decisions it holds on file says the session's number, and the count
    /// is read under that number's text, though the request carries its
    /// key in the clear. A count under the other name is no count of the
    /// key: its next request stays on its worker.
    #[test]
    fn a_keys_count_is_read_under_the_name_its_requests_maker_states() {
        /// The maker of a host that counts by the decisions it holds.
        struct ByNumber<'a>(Asks<'a>);

        impl RequestMaker for ByNumber<'_> {
            fn model(&self) -> &str {
                self.0.model()
            }

            fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
                self.0.make(spec)
            }

            fn workers(&self) -> WorkerLists<'_> {
                self.0.workers()
            }

            fn session_counted_under(&self) -> SessionCountedUnder {
                SessionCountedUnder::SessionNumber
            }
        }

        /// The key's first request pins it on a worker, two requests of
        /// the key are counted there under `name`, and the key's next
        /// request is decided: what its rule said of the pin, and whether
        /// the request left the pin's worker.
        fn next(
            engine: &PolicyEngine,
            list: &[Arc<dyn Worker>],
            maker: &impl RequestMaker,
            name: &str,
        ) -> (SessionVerdict, bool) {
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(engine, GatewayId::LOCAL, maker, now)
                .expect("a decision");
            let owner = guard.take(0).expect("the first target");
            guard.dispatched(now);
            for _ in 0..2 {
                list[owner.position as usize].increment_load();
                list[owner.position as usize].increment_routing_key_load(name);
            }
            let decided = engine.decide_under(GatewayId::LOCAL, maker, now);
            let (decided, _) = decided.expect("an answer").whole().expect("a whole one");
            let pick = *decided.decision.pick().expect("a pick");
            let went = pick.targets.first().expect("a target");
            (pick.session_verdict, went.id != owner.id)
        }

        let keyed = || {
            let mut keyed = request();
            keyed.facts.session_key = Identity::plain("talk-1");
            keyed
        };
        let mut digits = [0; Session::TEXT_LEN];
        let number = keyed().facts.session().text(&mut digits).to_owned();
        let decided = |by_number: bool, name: &str| {
            // The stock rule with an in-flight cap of two, which places a
            // key on the worker with the fewest requests.
            let mut document = PolicySpec::preset("manual").expect("a built-in preset");
            let rule = document.bindings[0].sessions.as_mut();
            let rule = rule.expect("the stock rule");
            (rule.inflight_cap, rule.assignment) = (2, Assignment::MinLoad);
            document.generation = 1;
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let (engine, list) = (PolicyEngine::new(spec), pool(4));
            let members = PublishedList::new(Arc::clone(&list));
            let asks = Asks::of(keyed(), Some(&members));
            if by_number {
                next(&engine, &list, &ByNumber(asks), name)
            } else {
                next(&engine, &list, &asks, name)
            }
        };
        let (handed_on, kept) = (
            (SessionVerdict::Respill, true),
            (SessionVerdict::Pinned, false),
        );
        assert_eq!(decided(false, "talk-1"), handed_on, "a guard's count");
        assert_eq!(decided(false, &number), kept, "no count of a guard's");
        assert_eq!(
            decided(true, &number),
            handed_on,
            "the held decisions' count"
        );
        assert_eq!(decided(true, "talk-1"), kept, "no count of a held decision");
    }

    /// The entry reads its host's clock where a decision is dated and
    /// nowhere else: a place reads none, and a retry's whole pass, a profile
    /// that scores and a place noted for a stage that takes a word read it
    /// once each.
    #[test]
    fn a_place_reads_no_clock_and_a_dated_decision_reads_it_once() {
        let list = pool(4);
        let tried = [worker_id(list[0].as_ref())];
        let read = |engine: &PolicyEngine, retried: bool| {
            let members = PublishedList::new(Arc::clone(&list));
            let mut stated = request();
            if retried {
                stated.facts.tried = &tried;
            }
            let asks = Asks::of(stated, Some(&members));
            let reads = Cell::new(0_u32);
            let decided = engine.decide_at(&asks, || {
                reads.set(reads.get() + 1);
                Instant::now()
            });
            let place = matches!(decided.expect("an answer"), Decided::Alone(_));
            (place, reads.get())
        };
        assert_eq!(read(&preset("round_robin"), false), (true, 0));
        assert_eq!(read(&preset("round_robin"), true), (false, 1));
        assert_eq!(read(&preset("least_load"), false), (false, 1));
        let (noting, _) = hearing_as(false, true, EventSet::LEASE);
        assert_eq!(read(&noting, false), (false, 1));
    }

    /// A place whose profile scores is never picked ahead of its request:
    /// the request is made for every decision, plain or not.
    #[test]
    fn a_place_that_scores_makes_its_request() {
        let engine = preset("least_load");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        for made in 1..=3 {
            assert!(decided(&engine, &asks).decision.pick().is_some());
            assert_eq!(asks.made.get(), made);
        }
    }

    /// A place that scores by load reads each worker's report in the
    /// snapshot its host hands for the decision: the worker with the
    /// shortest queue is picked at every seed. Handed no snapshot, the same
    /// engine reads no report, and its picks spread over the list.
    #[test]
    fn a_place_that_scores_by_load_reads_the_reports_its_host_hands() {
        let engine = preset("least_load");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let reports = (0..4)
            .map(|at| {
                let queued = if at == 2 { 0 } else { 90_000 };
                let url = format!("http://worker-{at}:8000");
                (url, one_rank_load(queued, 0.1, 1_000.0))
            })
            .collect();
        let picked_at = |loads: Option<Arc<LoadSnapshot>>, seed: u64| {
            let mut request = request();
            request.facts.seed = seed;
            let asks = Asks {
                loads,
                ..Asks::of(request, Some(&members))
            };
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let snapshot = LoadSnapshot::from_loads_for_test(reports);
        for seed in 0..32 {
            assert_eq!(
                picked_at(Some(Arc::clone(&snapshot)), seed),
                2,
                "seed {seed}"
            );
        }
        let mut unread = [0_u32; 4];
        for seed in 0..64 {
            unread[picked_at(None, seed) as usize] += 1;
        }
        assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");
    }

    /// A request stated ahead is plain where it is a first attempt that
    /// asks nothing beyond a worker of its list, and says what a pick with
    /// no request made needs: its phase, its wires and its size. Each thing
    /// it states beyond that makes it a request the whole pass reads. Its
    /// text's two lengths are asked apart, where such a pick's lease is
    /// issued.
    #[test]
    fn a_request_stated_ahead_says_whether_it_is_plain() {
        let mut stated = request();
        stated.facts.prompt_tokens = 9;
        stated.facts.prompt_chars = 31;
        stated.facts.prompt_bytes = 62;
        stated.constraints.wire_modes = WireModes::GRPC;
        assert_eq!(
            stated.plain(),
            Some(Plain {
                phase: Phase::Single,
                wires: WireModes::GRPC,
                prompt_tokens: 9,
            })
        );
        assert_eq!((stated.prompt_chars(), stated.prompt_bytes()), (31, 62));
        let tried = [WorkerId(1)];
        let rows = [0_u32];
        let mut retried = request();
        retried.facts.tried = &tried;
        let mut missed = request();
        missed.constraints.excluded = &tried;
        let mut hinted = request();
        hinted.constraints.subset_hint = Some(2);
        let mut lifted = request();
        lifted.constraints.overload_lifted = true;
        let mut later = request();
        later.constraints.decode_later = true;
        let mut masked = request();
        masked.constraints.stalled = &rows;
        let mut held = request();
        held.constraints.places = Some(&rows);
        for request in [retried, missed, hinted, lifted, later, masked, held] {
            assert_eq!(request.plain(), None);
        }
    }

    /// What the one turn does not decide goes to the whole pass over
    /// records: a retry never meets the worker it tried, and a pool whose
    /// every worker is overloaded is answered by the binding's own rule.
    #[test]
    fn a_retry_and_a_pool_no_turn_admits_are_put_to_the_whole_pass() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        retry.facts.attempt = 1;
        let asks = Asks::of(retry, Some(&members));
        for _ in 0..8 {
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            assert!(pick.targets.iter().all(|target| target.id != tried[0]));
        }
        for worker in list.iter() {
            worker.set_overloaded(true);
        }
        let asks = Asks::of(request(), Some(&members));
        match decided(&engine, &asks).decision {
            Decision::Pick(pick) => assert_eq!(pick.trace.lifted, LiftedVeto::Overload),
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "not-overloaded"),
            Decision::Queue(_) => panic!("no queue is planned"),
        }
    }

    /// Round robin under a sessions rule that leaves a new session to the
    /// profile (`delegate`) turns one rotation for a request with no routing
    /// key and for a new session: in turn they take the list's workers one
    /// after the other, under the draw's floor, at it and above it. The
    /// gateway's classic override asks its one policy for both
    /// (`model_gateway/src/policies/round_robin.rs:142-167`): for a request
    /// with no key at `model_gateway/src/policies/registry.rs:471` and for a
    /// new session at `:581`.
    #[test]
    fn a_plain_request_and_a_new_session_share_one_rotation_under_delegate() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
        rule.assignment = Assignment::Delegate;
        document.bindings[0].sessions = Some(rule);
        let keys = ["key-a", "key-b", "key-c", "key-d", "key-e", "key-f"];
        for workers in [DRAW_FROM - 1, DRAW_FROM, 64] {
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(spec);
            let list = pool(workers);
            let members = PublishedList::new(Arc::clone(&list));
            let mut places = Vec::new();
            for turn in 0..12_usize {
                let mut ask = request();
                if turn % 2 == 1 {
                    ask.facts.session_key = Identity::plain(keys[turn / 2]);
                }
                let asks = Asks::of(ask, Some(&members));
                let decided = decided(&engine, &asks);
                let pick = decided.decision.pick().expect("a pick");
                places.push(pick.targets.first().expect("the target").position as usize);
            }
            let in_turn: Vec<usize> = (0..12).map(|turn| (places[0] + turn) % workers).collect();
            assert_eq!(places, in_turn, "{workers} workers");
        }
    }

    /// A whole decision turns the list's own word, the one a pick alone
    /// turns, at every size: under the draw's floor, at it and above it.
    /// Two lists of one model and one size so turn a rotation each: asked
    /// in turn, a request with no routing key and a new session at each,
    /// they take each list's workers one after the other. And over one
    /// list a pick alone and a new session's whole decision share one
    /// rotation: the pick is asked for a model the same profile is bound
    /// for with no rule, since a place under a rule is not one pick's.
    /// The gateway's classic policy keeps its rotation by the list it is
    /// handed whole, whatever its size
    /// (`model_gateway/src/policies/round_robin.rs:95-99`, `:151`).
    #[test]
    fn each_list_turns_its_own_rotation_for_every_request() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
        rule.assignment = Assignment::Delegate;
        let mut unruled = document.bindings[0].clone();
        unruled.matcher.model = "unruled-*".to_owned();
        document.bindings.insert(0, unruled);
        document.bindings[1].sessions = Some(rule);
        let keys: Vec<String> = (0..48).map(|key| format!("key-{key}")).collect();
        let mut keys = keys.iter();
        for workers in [3, DRAW_FROM - 1, DRAW_FROM, 64] {
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(spec);
            let Place::Alone(profile) = engine.place_for("unruled-1", Phase::Single) else {
                panic!("one pick of round robin decides the place");
            };
            let (one, other) = (pool(workers), pool(workers));
            let lists = [
                PublishedList::new(Arc::clone(&one)),
                PublishedList::new(Arc::clone(&other)),
            ];
            let mut whole = |members: &PublishedList, keyed: bool| {
                let mut ask = request();
                if keyed {
                    ask.facts.session_key = Identity::plain(keys.next().expect("a key"));
                }
                let asks = Asks::of(ask, Some(members));
                let decided = decided(&engine, &asks);
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.first().expect("the target").position as usize
            };
            let mut places = [Vec::new(), Vec::new()];
            for turn in 0..8 {
                for (list, members) in lists.iter().enumerate() {
                    places[list].push(whole(members, turn % 2 == 1));
                }
            }
            for places in &places {
                let in_turn: Vec<usize> = (0..8).map(|turn| (places[0] + turn) % workers).collect();
                assert_eq!(*places, in_turn, "{workers} workers, two lists");
            }
            // The first list's rotation goes on, a pick alone and a new
            // session's whole decision in turn.
            let mut shared = Vec::new();
            for turn in 0..8 {
                shared.push(if turn % 2 == 0 {
                    let alone = engine.pick(profile, lists[0].lent(), RoutingState::eligible);
                    alone.expect("a worker")
                } else {
                    whole(&lists[0], true)
                });
            }
            let in_turn: Vec<usize> = (8..16)
                .map(|turn| (places[0][0] + turn) % workers)
                .collect();
            assert_eq!(shared, in_turn, "{workers} workers, one list");
        }
    }

    /// A turn that lands on a refused worker falls to the rotation of the
    /// workers the filters kept, in every whole decision and at every size:
    /// the ready workers take exact shares, a request with no routing key
    /// and a new session in turn, and the list's word moves once a request.
    /// The gateway's classic policy asks the list's turn by place and lets
    /// a refused place fall to its set's rotation
    /// (`model_gateway/src/policies/round_robin.rs:151-153`, `:164-167`),
    /// and its own case holds these shares over four workers (`:430-441`).
    #[test]
    fn a_refused_turn_falls_to_the_kept_workers_rotation_at_every_size() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
        rule.assignment = Assignment::Delegate;
        document.bindings[0].sessions = Some(rule);
        for workers in [4, DRAW_FROM, 64] {
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(spec);
            let list = pool(workers);
            list[1].set_status(WorkerStatus::NotReady);
            let members = PublishedList::new(Arc::clone(&list));
            let word = &engine.members_of(members.lent()).rotation.0;
            let before = word.load(Ordering::Relaxed);
            // One request a turn of the list goes to the kept workers'
            // rotation, which comes round once in as many turns of the list
            // as workers are kept.
            let requests = workers * (workers - 1);
            let keys: Vec<String> = (0..requests).map(|key| format!("key-{key}")).collect();
            let mut times = vec![0_usize; workers];
            for (turn, key) in keys.iter().enumerate() {
                let mut ask = request();
                if turn % 2 == 1 {
                    ask.facts.session_key = Identity::plain(key);
                }
                let asks = Asks::of(ask, Some(&members));
                let decided = decided(&engine, &asks);
                let pick = decided.decision.pick().expect("a pick");
                times[pick.targets.first().expect("the target").position as usize] += 1;
            }
            let mut shares = vec![workers; workers];
            shares[1] = 0;
            assert_eq!(times, shares, "{workers} workers");
            let turned = word.load(Ordering::Relaxed) - before;
            assert_eq!(
                turned, requests as u64,
                "{workers} workers: a turn a request"
            );
        }
    }

    /// A retry takes no turn of the list: it turns the rotation of the
    /// workers it has not tried, at every size. So the first attempts
    /// between the retries take the list's workers one after the other,
    /// and each worker not tried takes an exact share of the retries. The
    /// gateway's classic placement hands its policy a retry's untried
    /// workers as a list of their own, which nobody names
    /// (`model_gateway/src/routers/common/placement.rs:1256-1269`,
    /// `:1277`), and its policy takes the list's turn for a list named
    /// whole alone (`model_gateway/src/policies/round_robin.rs:145`), a set's
    /// rotation for any other (`:158-167`).
    ///
    /// A request that names a worker it tried is a retry whatever number it
    /// states: a host that states none is answered as one that does.
    #[test]
    fn a_retry_turns_the_untried_workers_rotation_and_no_turn_of_the_list() {
        let sizes = [4, DRAW_FROM, 64];
        for (workers, attempt) in sizes.into_iter().flat_map(|n| [(n, 1), (n, 0)]) {
            let engine = preset("round_robin");
            let list = pool(workers);
            let members = PublishedList::new(Arc::clone(&list));
            let tried = [worker_id(list[0].as_ref())];
            let mut retry = request();
            retry.facts.tried = &tried;
            retry.facts.attempt = attempt;
            let (first, retry) = (
                Asks::of(request(), Some(&members)),
                Asks::of(retry, Some(&members)),
            );
            // As many first attempts as turn the list once for each worker
            // not tried, and a retry behind each: the retries go once round
            // the untried workers for each worker of the list.
            let requests = workers * (workers - 1);
            let (mut firsts, mut retries) = (Vec::new(), vec![0_usize; workers]);
            for _ in 0..requests {
                firsts.push(placed(&engine, &first).place as usize);
                let decided = decided(&engine, &retry);
                let pick = decided.decision.pick().expect("a pick");
                retries[pick.targets.first().expect("the target").position as usize] += 1;
            }
            let in_turn: Vec<usize> = (0..requests)
                .map(|turn| (firsts[0] + turn) % workers)
                .collect();
            let said = format!("{workers} workers, a retry that states attempt {attempt}");
            assert_eq!(firsts, in_turn, "{said}: the first attempts");
            let mut shares = vec![workers; workers];
            shares[0] = 0;
            assert_eq!(retries, shares, "{said}: the retries");
        }
    }

    /// Two callers of one engine ask in strict turn over one list, and each
    /// is held to the places of the list it may take
    /// (`Constraints::places`): every record but two. Each caller is held
    /// away from a record at each parity, so whichever parity its turns
    /// land on, a record it may not take stands there. Each caller's 112
    /// requests hand each of its fourteen records eight and its two held
    /// records none: a record one caller is held away from keeps the other
    /// caller's share. The picks are counted a caller.
    #[test]
    fn two_callers_in_strict_turn_leave_each_others_held_records_their_share() {
        let away: [[u32; 2]; 2] = [[0, 1], [2, 3]];
        let places = away.map(|held| {
            (0..16)
                .filter(|at| !held.contains(at))
                .collect::<Vec<u32>>()
        });
        let engine = preset("round_robin");
        let list = pool(16);
        let members = PublishedList::new(Arc::clone(&list));
        let mut picked = [[0_u32; 16]; 2];
        for turn in 0..224_usize {
            let mut ask = request();
            ask.constraints.places = Some(&places[turn % 2]);
            let asks = Asks::of(ask, Some(&members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a ready record");
            picked[turn % 2][pick.targets.first().expect("a target").position as usize] += 1;
        }
        for (caller, held) in away.iter().enumerate() {
            let mut want = [8_u32; 16];
            for &at in held {
                want[at as usize] = 0;
            }
            assert_eq!(picked[caller], want, "caller {caller}");
        }
    }

    /// A list of ready workers engine after engine, one engine a number
    /// of `shape`: that many ranks of one data-parallel engine in their
    /// order, or a worker that registered no rank for a `1`. Beside it,
    /// each worker's engine by its place in the list.
    fn engines(shape: &[usize]) -> (Arc<[Arc<dyn Worker>]>, Vec<usize>) {
        let address = |engine: usize| format!("http://engine-{engine}:8000");
        let rank = |engine: usize, rank: usize, size: usize| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(address(engine))
                .model(ModelCard::new(MODEL))
                .dp_config(rank, size)
                .health_config(no_health_check())
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker)
        };
        let (mut list, mut engine_of) = (Vec::new(), Vec::new());
        for (engine, &size) in shape.iter().enumerate() {
            for at in 0..size {
                list.push(match size {
                    1 => ready_worker(&address(engine), MODEL),
                    _ => rank(engine, at, size),
                });
                engine_of.push(engine);
            }
        }
        (list.into(), engine_of)
    }

    /// A retry leaves every rank of the engine it tried while its list
    /// holds another worker: no target it is answered, first or
    /// alternate, is a rank of that engine, under each built-in policy,
    /// over a short list and over one long enough for the passes that
    /// draw. Named the tried rank alone, the stages kept the engine's
    /// other ranks. That is the gateway's classic answer: its tried list
    /// names an engine by the base address its ranks share, and a tried
    /// engine steps aside for the retry while another candidate is
    /// available (`model_gateway/src/routers/common/placement.rs:556-576`
    /// and `:1174-1179` at main `ef096118a`).
    #[test]
    fn a_retry_leaves_every_rank_of_the_engine_it_tried() {
        // Four ranks, a worker alone, two ranks and a worker alone; then
        // three engines of eight ranks.
        for shape in [&[4, 1, 2, 1][..], &[8, 8, 8]] {
            let (list, engine_of) = engines(shape);
            // The first engine's last rank was tried.
            let tried = [worker_id(list[shape[0] - 1].as_ref())];
            for name in [
                "round_robin",
                "random",
                "passthrough",
                "power_of_two",
                "least_load",
                "bucket",
                "cache_aware",
                "consistent_hashing",
                "prefix_hash",
                "manual",
            ] {
                let engine = preset(name);
                let members = PublishedList::new(Arc::clone(&list));
                let mut landed = BTreeSet::new();
                for seed in 0..64 {
                    let mut retry = request();
                    retry.facts.seed = seed;
                    retry.facts.attempt = 1;
                    retry.facts.tried = &tried;
                    // A ring and a sessions rule place a request by its key.
                    retry.facts.session_key = Identity::hashed(seed + 1);
                    let asks = Asks::of(retry, Some(&members));
                    let outcome = decided(&engine, &asks);
                    let pick = outcome.decision.pick().expect("a pick");
                    landed.extend(pick.targets.iter().map(|target| target.position));
                }
                let tried_engine: Vec<_> = landed
                    .iter()
                    .filter(|&&place| engine_of[place as usize] == 0)
                    .collect();
                assert!(
                    tried_engine.is_empty(),
                    "{name} over {shape:?}: places {tried_engine:?} are ranks of the tried engine"
                );
                assert!(!landed.is_empty(), "{name} over {shape:?}");
            }
        }
    }

    /// Over a list that is one engine, a retry is handed the engine again:
    /// no other worker is left, so the stage that passes tried workers over
    /// steps aside for the whole list, the tried rank with the others. That
    /// is the gateway's classic answer: where no untried candidate is
    /// available, its pool is used whole
    /// (`model_gateway/src/routers/common/placement.rs:571-573` at main
    /// `ef096118a`). Named the tried rank alone, the stage kept the other
    /// ranks and never handed that one.
    #[test]
    fn a_retry_over_one_engine_is_handed_its_ranks_again() {
        let engine = preset("random");
        let (list, _) = engines(&[4]);
        let members = PublishedList::new(Arc::clone(&list));
        let tried = [worker_id(list[1].as_ref())];
        let mut firsts = BTreeSet::new();
        for seed in 0..64 {
            let mut retry = request();
            retry.facts.seed = seed;
            retry.facts.tried = &tried;
            let asks = Asks::of(retry, Some(&members));
            let outcome = decided(&engine, &asks);
            let pick = outcome.decision.pick().expect("a pick");
            firsts.insert(pick.targets.first().expect("a target").position);
        }
        assert_eq!(firsts, BTreeSet::from([0, 1, 2, 3]), "the engine whole");
    }

    /// A key pinned on one rank of an engine leaves that engine on a retry
    /// that tried another of its ranks: the key's rows are read under the
    /// retry's whole list of tried workers, so the pin is not followed, and
    /// the whole pass places the request on another engine. The gateway's
    /// classic sticky policy is handed its candidates with the tried
    /// engines out, and picks another worker where none of its key's is
    /// among them (`model_gateway/src/routers/common/placement.rs:556-576`
    /// and `model_gateway/src/policies/manual.rs:303-308` at main
    /// `ef096118a`).
    #[test]
    fn a_keyed_retry_leaves_the_engine_its_key_is_pinned_on() {
        // The stock sessions rule: its binding pins a key.
        let engine = preset("manual");
        let (list, engine_of) = engines(&[4, 4]);
        let members = PublishedList::new(Arc::clone(&list));
        // One request of `key` that tried `tried`, served by its first
        // target: that target's place.
        let sent = |key: u64, tried: &[WorkerId]| {
            let mut keyed = request();
            keyed.facts.seed = key;
            keyed.facts.session_key = Identity::hashed(key);
            keyed.facts.tried = tried;
            let asks = Asks::of(keyed, Some(&members));
            served(&engine, &asks, Instant::now()).position as usize
        };
        for key in 1..=16 {
            let pin = sent(key, &[]);
            assert_eq!(sent(key, &[]), pin, "key {key}: its pin");
            // Another rank of the pin's engine was tried.
            let other = (pin & !3) | ((pin + 1) & 3);
            let tried = [worker_id(list[other].as_ref())];
            let went = sent(key, &tried);
            assert_ne!(
                engine_of[went], engine_of[pin],
                "key {key}: place {went} is a rank of the tried engine"
            );
        }
    }

    /// The workers a retry is decided to have tried stand in the calling
    /// thread's own list, which only a retry over ranks takes: a first
    /// attempt leaves it as it was, and so does a retry over workers that
    /// each stand alone. It grows at a thread's first such retry and is
    /// kept: the retries behind it allocate nothing for it.
    #[test]
    fn only_a_retry_over_ranks_takes_the_threads_list_and_it_grows_once() {
        let engine = preset("power_of_two");
        let (ranks, _) = engines(&[4, 4]);
        let ranked = PublishedList::new(Arc::clone(&ranks));
        let alone = pool(8);
        let unranked = PublishedList::new(Arc::clone(&alone));
        // Where the thread's list stands and what it holds room for.
        let kept = || {
            let list = TRIED.take();
            let stands = (list.as_ptr(), list.capacity());
            TRIED.set(list);
            stands
        };
        let asked = |members: &PublishedList, tried: &[WorkerId]| {
            let mut asked = request();
            asked.facts.tried = tried;
            let asks = Asks::of(asked, Some(members));
            assert!(decided(&engine, &asks).decision.pick().is_some());
        };
        asked(&ranked, &[]);
        assert_eq!(kept().1, 0, "a first attempt takes no list");
        asked(&unranked, &[worker_id(alone[0].as_ref())]);
        assert_eq!(kept().1, 0, "nor a retry over workers that stand alone");
        let tried = [worker_id(ranks[1].as_ref())];
        asked(&ranked, &tried);
        let grown = kept();
        assert!(grown.1 >= 4, "the engine's four ranks stand in it");
        asked(&ranked, &tried);
        assert_eq!(kept(), grown, "the list is kept as it grew");
    }

    /// A whole pass's lease states the three sizes its request states: the
    /// count of its token ids and its text's two lengths. A reader that
    /// sizes a held decision of text reads the bytes over four off the
    /// lease, the number its request's facts read: here a text of 400
    /// two-byte characters.
    #[test]
    fn a_whole_passs_lease_states_the_requests_three_sizes() {
        // The stock sessions rule takes a word on a lease.
        let engine = preset("manual");
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let mut keyed = request();
        keyed.facts.session_key = Identity::hashed(9);
        (keyed.facts.prompt_chars, keyed.facts.prompt_bytes) = (400, 800);
        let size = keyed.facts.prompt_size();
        let asks = Asks::of(keyed, Some(&members));
        let lease = decided(&engine, &asks).lease.expect("a lease").lease;
        assert_eq!((lease.tokens, lease.chars, lease.bytes), (0, 400, 800));
        assert_eq!((lease.prompt_size(), size), (200, 200));
    }

    /// A retry decided under its whole list of tried workers is leased as
    /// a first attempt is: its lease is dated at the decision's instant on
    /// the engine's clock, which the decision writes into the request.
    #[test]
    fn a_retry_over_ranks_is_leased_at_its_decisions_instant() {
        // The stock sessions rule takes a word on a lease.
        let engine = preset("manual");
        let (list, _) = engines(&[4, 4]);
        let members = PublishedList::new(Arc::clone(&list));
        let now = Instant::now() + Duration::from_secs(2);
        let dated = |tried: &[WorkerId]| {
            let mut keyed = request();
            keyed.facts.session_key = Identity::hashed(9);
            keyed.facts.tried = tried;
            let asks = Asks::of(keyed, Some(&members));
            let decided = engine.decide_under(GatewayId::LOCAL, &asks, now);
            let whole = decided.expect("an answer").whole().expect("whole").0;
            whole.lease.expect("a lease").lease.issued_micros
        };
        let first = dated(&[]);
        assert!(first >= 2_000_000, "the engine's clock at `now`: {first}");
        let tried = [worker_id(list[1].as_ref())];
        assert_eq!(dated(&tried), first, "the retry's lease is dated alike");
    }

    /// A tried worker its list no longer holds names itself alone: the
    /// list says nothing of its engine, so the engine's other ranks stay
    /// candidates. A list holds some ranks of an engine and not the others
    /// while its registry removes them, one after another. The gateway's
    /// classic passes them over there: its tried list names the engine's
    /// base address, whichever of its ranks the list still holds
    /// (`model_gateway/src/routers/common/placement.rs:1174-1179` at main
    /// `ef096118a`).
    #[test]
    fn a_tried_worker_its_list_no_longer_holds_names_itself_alone() {
        let engine = preset("random");
        let (whole, engine_of) = engines(&[4, 4]);
        // The first engine's first rank was tried, and has left the list.
        let tried = [worker_id(whole[0].as_ref())];
        let list: Arc<[Arc<dyn Worker>]> = whole[1..].iter().cloned().collect();
        let members = PublishedList::new(Arc::clone(&list));
        let mut firsts = BTreeSet::new();
        for seed in 0..64 {
            let mut retry = request();
            retry.facts.seed = seed;
            retry.facts.tried = &tried;
            let asks = Asks::of(retry, Some(&members));
            let outcome = decided(&engine, &asks);
            let pick = outcome.decision.pick().expect("a pick");
            firsts.insert(pick.targets.first().expect("a target").position);
        }
        // A place of the shorter list is one less than the worker's place
        // in the whole one.
        let stayed = firsts
            .iter()
            .any(|&place| engine_of[place as usize + 1] == 0);
        assert!(
            stayed,
            "{firsts:?}: the tried engine's ranks stay candidates"
        );
    }

    /// A request that hands no list is decided over no worker. One that
    /// states what no stage here runs is declined under that stage's name.
    #[test]
    fn what_no_stage_runs_is_answered_by_name() {
        let engine = preset("round_robin");
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let shed = |request: Request<'_>, members: Option<&PublishedList>| {
            let asks = Asks::of(request, members);
            match decided(&engine, &asks).decision {
                Decision::Shed(shed) => (shed.reason, shed.emptied_by),
                other => panic!("a shed, not {other:?}"),
            }
        };
        assert_eq!(shed(request(), None), (ShedReason::NoEndpoints, NO_STAGE));
        let mut hinted = request();
        hinted.constraints.subset_hint = Some(1);
        assert_eq!(
            shed(hinted, Some(&members)),
            (ShedReason::PolicyDeclined, SUBSET_HINT)
        );
        // The list's workers are dialed over HTTP.
        let mut dialed = request();
        dialed.constraints.wire_modes = WireModes::GRPC;
        assert_eq!(
            shed(dialed, Some(&members)),
            (ShedReason::PolicyDeclined, WIRES)
        );
        dialed.constraints.wire_modes = WireModes::HTTP.union(WireModes::GRPC);
        let asks = Asks::of(dialed, Some(&members));
        assert!(matches!(answered(&engine, &asks), Decided::Alone(_)));
    }

    /// A list's cell is asked where the engine decides over the list, and
    /// nowhere else. A request the engine refuses by its phase, a pair's
    /// decode sent on its own, asks no
    /// cell, plain or not, at either entry: its list's cell stays empty. A
    /// maker that names its list while it makes the request is decided over
    /// that list. So is a plain maker that lends one list ahead of its
    /// making and another behind it: the pick ahead of the making found no
    /// ready worker in the first, and the value it asked there is not the
    /// one the request is decided over.
    #[test]
    fn a_cell_is_asked_where_its_list_is_decided_over() {
        /// A maker that names its list while it makes the request.
        struct Late<'a>(Request<'a>, &'a PublishedList, Cell<bool>);

        impl RequestMaker for Late<'_> {
            fn model(&self) -> &str {
                self.0.facts.model
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                self.2.set(true);
                self.0
            }

            fn workers(&self) -> WorkerLists<'_> {
                if self.2.get() {
                    WorkerLists::Single(self.1.lent())
                } else {
                    WorkerLists::None
                }
            }
        }

        /// A plain maker that lends its first list ahead of its making
        /// and its second behind it.
        struct Moved<'a>(Request<'a>, [&'a PublishedList; 2], Cell<bool>);

        impl RequestMaker for Moved<'_> {
            fn model(&self) -> &str {
                self.0.facts.model
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                self.2.set(true);
                self.0
            }

            fn plain(&self) -> Option<Plain> {
                self.0.plain()
            }

            fn workers(&self) -> WorkerLists<'_> {
                WorkerLists::Single(self.1[usize::from(self.2.get())].lent())
            }
        }

        let engine = preset("round_robin");
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let now = Instant::now();
        // Under a pair plan a decode is its pair's side: sent on its own
        // over one list it is refused, and the list's cell is not asked.
        let document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        let paired = PolicyEngine::new(AppliedSpec::validate(&document).expect("a valid document"));
        let mut side = request();
        side.facts.phase = Phase::Decode;
        // A plain request, and the same one from a maker that does not say
        // so.
        let plain = Asks::of(side, Some(&members));
        assert!(plain.plain().is_some());
        let leased = paired.decide_under(GatewayId::LOCAL, &plain, now);
        assert_eq!(leased.err(), Some(DecisionRefused::PairSide));
        let whole = Whole(Asks::of(side, Some(&members)));
        let leased = paired.decide_under(GatewayId::LOCAL, &whole, now);
        assert_eq!(leased.err(), Some(DecisionRefused::PairSide));
        let bare = paired.decide_without_lease(&whole, now);
        assert_eq!(bare.err(), Some(DecisionRefused::PairSide));
        assert_eq!(paired.values_made(), 0, "no cell was asked");
        assert!(!members.filled(), "the list's cell stays empty");

        let late = Late(request(), &members, Cell::new(false));
        let leased = engine.decide_under(GatewayId::LOCAL, &late, now);
        let (whole, _) = leased.expect("an answer").whole().expect("no place");
        assert!(whole.decision.pick().is_some(), "over the list it named");
        let late = Late(request(), &members, Cell::new(false));
        let (bare, _) = engine.decide_without_lease(&late, now).expect("an answer");
        assert!(bare.pick().is_some(), "over the list it named");
        assert_eq!(engine.values_made(), 1, "one value of the one list");

        let idle = pool(2);
        for worker in idle.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
        let ahead = PublishedList::new(Arc::clone(&idle));
        let moved = Moved(request(), [&ahead, &members], Cell::new(false));
        assert!(moved.plain().is_some());
        let leased = engine.decide_under(GatewayId::LOCAL, &moved, now);
        let (whole, _) = leased.expect("an answer").whole().expect("no place");
        assert!(whole.decision.pick().is_some(), "over the list lent behind");
        assert!(moved.2.get() && ahead.filled(), "the pick asked the first");
    }

    /// A request its host holds to places of its list is made and decided
    /// over the list's own records: every target stands at one of the
    /// places, and each of them takes requests. Under a rotation, two draws
    /// and a profile that scores, each as it ships and with no filter, over
    /// a list under the draw's floor and one over it.
    #[test]
    fn a_request_held_to_places_is_handed_no_other_place_under_any_document() {
        for name in ["round_robin", "random", "power_of_two", "least_load"] {
            for bare in [false, true] {
                let engine = if bare { unfiltered(name) } else { preset(name) };
                for workers in [8, 32] {
                    let list = pool(workers);
                    let members = PublishedList::new(Arc::clone(&list));
                    let mut asks = Asks::of(request(), Some(&members));
                    asks.request.constraints.places = Some(&[3, 6]);
                    let mut first = [0_u32; 8];
                    for seed in 0..32 {
                        asks.request.facts.seed = seed;
                        let decided = decided(&engine, &asks);
                        let pick = decided.decision.pick().expect("a pick");
                        let mut places = pick.targets.iter().map(|target| target.position);
                        assert!(places.all(|place| place == 3 || place == 6), "{name}");
                        let lead = pick.targets.first().expect("a target").position;
                        first[lead as usize] += 1;
                    }
                    assert!(first[3] > 0 && first[6] > 0, "{name}: {first:?}");
                    assert_eq!(first[3] + first[6], 32, "{name}: {first:?}");
                    assert_eq!(asks.made.get(), 32, "{name}: such a request is made");
                }
            }
        }
    }

    /// A keyed request held to places is handed the next members of the
    /// stock ring. Held to every member but its key's owner, it is handed
    /// the members a request held to nothing is named behind the owner, in
    /// that order: as many targets, and never the owner.
    #[test]
    fn a_keyed_request_held_to_places_takes_the_rings_next_members() {
        for engine in [
            preset("consistent_hashing"),
            unfiltered("consistent_hashing"),
        ] {
            let list = pool(16);
            let members = PublishedList::new(Arc::clone(&list));
            let places = |decided: &DecideOutcome| -> Vec<u32> {
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.iter().map(|target| target.position).collect()
            };
            for key in 1..=40 {
                let mut keyed = request();
                keyed.facts.seed = key;
                keyed.facts.session_key = Identity::hashed(key);
                let whole = places(&decided(&engine, &Asks::of(keyed, Some(&members))));
                assert_eq!(whole.len(), MAX_TARGETS, "key {key}");
                let others: Vec<u32> = (0..16).filter(|&place| place != whole[0]).collect();
                let mut held = Asks::of(keyed, Some(&members));
                held.request.constraints.places = Some(&others);
                let part = places(&decided(&engine, &held));
                assert_eq!(part.len(), MAX_TARGETS, "key {key}");
                assert_eq!(part[..MAX_TARGETS - 1], whole[1..], "key {key}");
                assert!(!part.contains(&whole[0]), "key {key}: the owner is handed");
            }
        }
    }

    /// The default policy holds a request to its places ahead of its
    /// prompt's holder. A prompt is sent to a worker, and the same prompt
    /// held to places that worker is outside of is handed none but them,
    /// as a request with no prompt is: over a list under the draw's floor
    /// and one over it.
    #[test]
    fn a_prompts_holder_outside_the_places_is_not_handed() {
        for workers in [8_u32, 32] {
            let engine = preset("cache_aware");
            let list = pool(workers as usize);
            let members = PublishedList::new(Arc::clone(&list));
            let tokens: Vec<u32> = (0..1_024).collect();
            let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
            let sent = |places: Option<&[u32]>, seed: u64, prompted: bool| {
                let mut asked = request();
                asked.facts.seed = seed;
                asked.constraints.places = places;
                if prompted {
                    hashes.write_to(&mut asked.facts);
                }
                let decided = decided(&engine, &Asks::of(asked, Some(&members)));
                let pick = decided.decision.pick().expect("a pick");
                let places = pick.targets.iter().map(|target| target.position);
                places.collect::<Vec<u32>>()
            };
            let holder = sent(None, 0, true)[0];
            let others: Vec<u32> = (0..workers)
                .filter(|&place| place != holder)
                .take(2)
                .collect();
            for seed in 1..32 {
                for prompted in [true, false] {
                    let named = sent(Some(&others), seed, prompted);
                    assert!(
                        named.iter().all(|place| others.contains(place)),
                        "{workers} workers, seed {seed}: {named:?}"
                    );
                }
            }
        }
    }

    /// The rule for a list whose every worker is refused steers a held
    /// request among its places. Held to a worker over its thresholds, it
    /// is handed that worker with the veto lifted, beside free workers
    /// outside its places.
    #[test]
    fn a_held_request_over_its_thresholds_everywhere_is_steered_among_its_places() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        list[1].set_overloaded(true);
        let mut held = request();
        held.constraints.places = Some(&[1]);
        let decided = decided(&engine, &Asks::of(held, Some(&members)));
        let pick = decided.decision.pick().expect("a pick");
        assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
        assert!(pick.targets.iter().all(|target| target.position == 1));
    }

    /// The default policy's single request is named the worker its prompt
    /// was sent to. The whole pass reads the engine's record of sent
    /// prefixes, which the pick before it wrote: a prompt goes back to its
    /// worker whatever the request's seed, other prompts are not held to
    /// that worker, and a worker told gone holds no prompt.
    #[test]
    fn the_default_policy_sends_a_prompt_back_to_the_worker_it_was_sent_to() {
        let engine = preset("cache_aware");
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        let sent = |tokens: &[u32], seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let asked = prompt(0);
        let first = sent(&asked, 0);
        for seed in 1..32 {
            assert_eq!(sent(&asked, seed), first, "seed {seed}");
        }
        // Sixteen other prompts are placed by their seeds, over idle workers.
        let others: Vec<u32> = (1..=16)
            .map(|at| sent(&prompt(at * 100_000), u64::from(at)))
            .collect();
        assert!(others.iter().any(|&position| position != others[0]));
        // With every worker told gone before a request, its seed places it.
        let moved = (0..32).any(|seed| {
            for worker in list.iter() {
                engine.forget(worker.url());
            }
            sent(&asked, seed) != first
        });
        assert!(moved, "a forgotten worker holds no prompt");
        assert_eq!(engine.affinity().left_out(), 0);
    }

    /// Round robin under a sessions rule sends a key back to the worker its
    /// first request was dispatched to. Every request of the binding takes
    /// the whole decision, where the rule is read ahead of the profile: one
    /// pick alone would turn the rotation past the pin.
    #[test]
    fn a_pinned_key_under_round_robin_goes_back_to_its_worker() {
        let engine = {
            let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
            let manual = PolicySpec::preset("manual").expect("a built-in preset");
            document.view.headers.clone_from(&manual.view.headers);
            let rule = manual.bindings[0].sessions.clone();
            document.bindings[0].sessions = Some(rule.expect("its rule"));
            PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
        };
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        // One request under the key, sent to its first target and served
        // there: the place it went to, and whether the engine made it.
        let sent = || {
            let mut asked = request();
            asked.facts.session_key = Identity::plain("key-a");
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (target.position, asks.made.get() > 0)
        };
        let (pin, made) = sent();
        assert!(made, "the key's first request takes the whole decision");
        for turn in 0..6 {
            assert_eq!(sent(), (pin, true), "turn {turn}");
        }
    }

    /// The document of the default policy's profile under the stock
    /// sessions rule, a new key placed by the profile's own stages, and
    /// `kept`, the rule's word of a request it places alone.
    fn default_under_sessions(kept: RulePlaced) -> AppliedSpec {
        let mut document = PolicySpec::preset("manual").expect("a built-in preset");
        let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
        document.profiles = default.profiles;
        let binding = &mut document.bindings[0];
        binding.profile = Some("cache_aware".into());
        let rule = binding.sessions.as_mut().expect("the stock rule");
        rule.assignment = Assignment::Delegate;
        rule.rule_placed = kept;
        document.generation = 1;
        AppliedSpec::validate(&document).expect("the document validates")
    }

    /// What `engine`'s record of sent prefixes holds of the prompt `tokens`
    /// for each worker of `list`, in blocks: what a decision of the same
    /// lookup is named. The lookup names the deepest holders alone, so a
    /// shorter copy at another worker reads as none. The read's own answer
    /// picks nobody, so it writes nothing.
    fn recorded(engine: &PolicyEngine, list: &[Arc<dyn Worker>], tokens: &[u32]) -> Vec<u32> {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.now_micros = engine.clock.micros(Instant::now());
        hashes.write_to(&mut asked.facts);
        let nobody = over_none(engine.loaded(), &asked);
        let ids: Vec<WorkerId> = list
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .collect();
        let mut held = vec![0; list.len()];
        overlap::decide(
            engine.affinity(),
            &overlap::PoolAffinity::default(),
            engine.spec(),
            list,
            &ids,
            &mut asked.facts,
            |_| None,
            |_| true,
            |facts| {
                match facts.named {
                    Some(rows) => {
                        for holder in rows.holders {
                            held[holder.row as usize] = holder.predicted_blocks;
                        }
                    }
                    None => {
                        for (row, worker) in facts.affinity.iter().enumerate() {
                            held[row] = worker.predicted_blocks;
                        }
                    }
                }
                nobody
            },
        );
        held
    }

    /// A prompt of 64 blocks held by the worker at `place` of eight alone,
    /// as [`recorded`] reads it.
    fn held_at(place: u32) -> Vec<u32> {
        let mut held = vec![0; 8];
        held[place as usize] = 64;
        held
    }

    /// The default policy under a sessions rule. A key's first request is
    /// placed by the whole decision, and its dispatch makes the key's pin.
    /// The key's next request follows that pin by one worker's record under
    /// either word of the rule: the whole decision does not run, so no
    /// stage is asked and no row is written for its pass. What is kept of
    /// that request's prompt is by the rule's word.
    ///
    /// Where the rule keeps the request from the profile the record takes
    /// nothing of its prompt, as a gateway's own sticky map keeps a pinned
    /// request from its policy. Where the request is the profile's own, the
    /// rule's default, its pick goes through the lookup that names a
    /// request's holders with the answer made. The record then holds the
    /// prompt for the pin's worker alone, and the prompt goes back there
    /// with no key. The pick is the one made under the other word but for
    /// the word itself and for what it states of the pin's worker: a prompt
    /// nobody was sent is a miss of its blocks, and a pick kept from the
    /// profile states nothing.
    ///
    /// A pinned worker that is not ready gives the key back to the whole
    /// decision.
    #[test]
    fn the_default_policy_follows_a_sessions_pin_by_one_record() {
        // The pinned requests' picks under the rule's word `kept`, a key
        // each.
        let followed = |kept: RulePlaced| {
            let engine = PolicyEngine::new(default_under_sessions(kept));
            let list = pool(8);
            let members = PublishedList::new(Arc::clone(&list));
            let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
            // One request of `tokens` under the session key `key`, with no
            // key where it is `0`, sent to its first target and served
            // there: the place it went to and its pick.
            let sent = |tokens: &[u32], key: u64, seed: u64| {
                let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
                let mut asked = request();
                asked.facts.seed = seed;
                asked.facts.session_key = Identity::hashed(key);
                hashes.write_to(&mut asked.facts);
                let asks = Asks::of(asked, Some(&members));
                let now = Instant::now();
                let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                    .expect("a decision");
                let pick = *guard.decision().pick().expect("a pick");
                let target = guard.take(0).expect("the first target");
                guard.dispatched(now);
                guard.completed(
                    &Outcome {
                        side: Side::Single,
                        served: target.id,
                        status: 200,
                        class: TransportClass::Ok,
                        ttft_micros: 0,
                        tokens_in: 0,
                        tokens_out: 0,
                        duration_micros: 0,
                    },
                    now,
                );
                (target.position, pick)
            };
            let fresh = |key: u32| prompt(key * 100_000 + 50_000);
            let (mut pins, mut picks) = (Vec::new(), Vec::new());
            for key in 1..=6_u32 {
                // The key's first request: the whole decision places it by
                // its seed, over idle workers, and claims it.
                let seed = u64::from(key);
                let (pin, pick) = sent(&prompt(key * 100_000), seed, seed);
                let claimed = matches!(pick.session_verdict, SessionVerdict::Claimed(_));
                assert!(claimed, "{kept:?}: key {key}");
                let whole = !pick.trace.drew_first();
                assert!(
                    whole,
                    "{kept:?}: key {key}: a new key is the whole decision's"
                );
                // Its next request, of a prompt nobody was sent: the pin,
                // followed by its one worker's record.
                let (went, pick) = sent(&fresh(key), seed, 1);
                let answered = (went, pick.session_verdict, pick.placed_by());
                let pinned = (pin, SessionVerdict::Pinned, PlacedBy::Rule);
                assert_eq!(answered, pinned, "{kept:?}: key {key}");
                let whole = !pick.trace.drew_first();
                assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
                // What the record holds of the pinned request's prompt.
                let held = recorded(&engine, &list, &fresh(key));
                match kept {
                    RulePlaced::Untold => assert_eq!(held, [0; 8], "key {key}: kept"),
                    RulePlaced::Told => assert_eq!(held, held_at(pin), "key {key}: told"),
                }
                pins.push(pin);
                picks.push(pick);
            }
            assert!(
                pins.iter().any(|pin| *pin != pins[0]),
                "{kept:?}: the keys are pinned to more workers than one: {pins:?}"
            );
            if kept == RulePlaced::Told {
                // A recorded prompt goes back to its pin's worker with no
                // key.
                for key in 1..=6_u32 {
                    let (went, _) = sent(&fresh(key), 0, u64::from(key) + 40);
                    assert_eq!(went, pins[key as usize - 1], "key {key}: no key");
                }
            }
            // A pin whose worker is not ready is the whole decision's to
            // place anew.
            list[pins[0] as usize].set_status(WorkerStatus::NotReady);
            let (moved, pick) = sent(&prompt(100_000), 1, 3);
            let whole = !pick.trace.drew_first();
            assert!(whole, "{kept:?}: the whole decision places the key anew");
            assert_ne!(moved, pins[0], "{kept:?}");
            picks
        };
        let (kept, own) = (followed(RulePlaced::Untold), followed(RulePlaced::Told));
        // The hand-over returns the keyed entry's answer. The first key's
        // pinned request met both engines alike: the pick that went through
        // the lookup and the one that did not differ by the rule's word,
        // and by what the lookup states behind the told one. Nobody was
        // sent that prompt: a miss of its 64 blocks.
        assert_eq!(kept[0].overlap(), Overlap::NONE, "kept from the profile");
        let mut twin = kept[0];
        twin.targets.by_rule(RulePlaced::Told);
        twin.targets
            .state(Overlap::held(AffinityFacts::default(), 64));
        assert_eq!(twin, own[0]);
        assert_ne!(kept[0], own[0], "the rule's word is on the pick");
    }

    /// A retry over a list where an engine holds several workers is its
    /// own instance of the decision ([`retried`]), and it follows its key's
    /// pin by one worker's record as a first attempt does: the pin's worker
    /// is no rank of the engine the request tried. What is kept of the
    /// retry's prompt is by the rule's word there too. Under `told` the
    /// record holds it for the pin's worker alone, and under `untold` for
    /// nobody.
    #[test]
    fn a_retrys_followed_pin_is_recorded_by_the_rules_word() {
        for kept in [RulePlaced::Untold, RulePlaced::Told] {
            let engine = PolicyEngine::new(default_under_sessions(kept));
            // Two engines of two ranks, then two workers alone.
            let (list, engine_of) = engines(&[2, 2, 1, 1]);
            let members = PublishedList::new(Arc::clone(&list));
            let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
            // One request of `tokens` under the session key `key` that
            // tried `tried`, sent to its first target and served there: the
            // place it went to and its pick.
            let sent = |tokens: &[u32], key: u64, tried: &[WorkerId]| {
                let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
                let mut asked = request();
                asked.facts.seed = key;
                asked.facts.session_key = Identity::hashed(key);
                if !tried.is_empty() {
                    asked.facts.attempt = 1;
                }
                asked.facts.tried = tried;
                hashes.write_to(&mut asked.facts);
                let asks = Asks::of(asked, Some(&members));
                let now = Instant::now();
                let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                    .expect("a decision");
                let pick = *guard.decision().pick().expect("a pick");
                let target = guard.take(0).expect("the first target");
                guard.dispatched(now);
                guard.completed(
                    &Outcome {
                        side: Side::Single,
                        served: target.id,
                        status: 200,
                        class: TransportClass::Ok,
                        ttft_micros: 0,
                        tokens_in: 0,
                        tokens_out: 0,
                        duration_micros: 0,
                    },
                    now,
                );
                (target.position as usize, pick)
            };
            for key in 1..=6_u32 {
                let (own, session) = (key * 100_000, u64::from(key));
                let (pin, _) = sent(&prompt(own), session, &[]);
                // The key's next request tried a worker of another engine
                // than its pin's, and comes again.
                let other = engine_of.iter().position(|&of| of != engine_of[pin]);
                let tried = [worker_id(list[other.expect("another engine")].as_ref())];
                let fresh = prompt(own + 50_000);
                let (went, pick) = sent(&fresh, session, &tried);
                let answered = (went, pick.session_verdict, pick.placed_by());
                let pinned = (pin, SessionVerdict::Pinned, PlacedBy::Rule);
                assert_eq!(answered, pinned, "{kept:?}: key {key}");
                let whole = !pick.trace.drew_first();
                assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
                let mut held = vec![0; list.len()];
                if kept == RulePlaced::Told {
                    held[pin] = 64;
                }
                let read = recorded(&engine, &list, &fresh);
                assert_eq!(read, held, "{kept:?}: key {key}: the retry's prompt");
            }
        }
    }

    /// A request stated ahead by a maker that states no plain request
    /// ([`RequestMaker::plain`]'s own answer), as a host that hands a
    /// decision's targets to another process states none: the leasing
    /// entry then makes the request and decides it whole.
    struct Whole<'a>(Asks<'a>);

    impl RequestMaker for Whole<'_> {
        fn model(&self) -> &str {
            self.0.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.0.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.0.workers()
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.0.loads()
        }
    }

    /// The entry for a host that files leases itself answers the decision
    /// alone. Under a document whose stages take every word of a lease and
    /// whose picker places a plain request by itself, the entry still makes
    /// each request once and decides it whole: its pick names its
    /// alternates, where the leasing entry answers the one place. No id is
    /// taken and no stage hears an issue: the host tells the stages, under
    /// the lease it files.
    #[test]
    fn a_decision_without_a_lease_takes_no_id_and_tells_no_stage() {
        let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
        let place = engine.place_for(MODEL, Phase::Single);
        assert!(matches!(place, Place::Alone(_)), "{place:?}");
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        assert!(asks.plain().is_some(), "a plain request");
        let mut last = None;
        for _ in 0..8 {
            let decision = engine.decide_without_lease(&asks, Instant::now());
            let (decision, _) = decision.expect("a decision");
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.targets.len(), 4, "the pick and its alternates");
            for target in pick.targets.iter() {
                let worker = &list[target.position as usize];
                assert_eq!(target.id, worker_id(worker.as_ref()));
            }
            last = Some(decision);
        }
        assert_eq!(asks.made.get(), 8, "each request is made once");
        assert!(heard.lock().is_empty(), "no stage heard an issue");
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

        // The host files the pick's lease and tells the stages under it.
        let decision = last.expect("a decision");
        let pick = decision.pick().expect("a pick");
        let first = *pick.targets.first().expect("the target");
        let lease = Lease {
            id: LeaseId(77),
            kind: LeaseKind::Credit,
            side: Side::Single,
            target_index: 0,
            booked: first.id,
            generation: pick.trace.generation,
            session: Session::NONE,
            session_verdict: SessionVerdict::None,
            issued_micros: 0,
            tokens: 0,
            chars: 0,
            bytes: 0,
        };
        engine.spec().engine().on_decision_issued(&lease, &decision);
        assert_eq!(heard.lock().as_slice(), [("issued", first.id, 0)]);
    }

    /// The entry makes the leasing entry's whole decision of the same
    /// request, under every stock policy: two engines made alike, the
    /// leasing one asked by a maker that states no plain request, name the
    /// same workers in the same order, request after request, by the
    /// request's seed and by a rotation's turns. With a load snapshot lent
    /// both read it: a policy that scores by load hands the worker with the
    /// shortest queue at every seed.
    #[test]
    fn a_decision_without_a_lease_is_the_leasing_entrys_whole_decision() {
        use crate::document::spec::PRESETS;

        let places = |decision: &Decision| -> Vec<u32> {
            let targets = decision.pick().map(|pick| pick.targets.iter());
            targets.map_or_else(Vec::new, |targets| {
                targets.map(|target| target.position).collect()
            })
        };
        let reports = (0..8)
            .map(|at| {
                let queued = if at == 5 { 0 } else { 90_000 };
                let url = format!("http://worker-{at}:8000");
                (url, one_rank_load(queued, 0.1, 1_000.0))
            })
            .collect();
        let snapshot = LoadSnapshot::from_loads_for_test(reports);
        for policy in PRESETS {
            for loads in [None, Some(Arc::clone(&snapshot))] {
                let (leasing, bare) = (preset(policy), preset(policy));
                let list = pool(8);
                let (theirs, ours) = (
                    PublishedList::new(Arc::clone(&list)),
                    PublishedList::new(Arc::clone(&list)),
                );
                for seed in 0..16 {
                    let asked = |members| {
                        let mut seeded = request();
                        seeded.facts.seed = seed;
                        Asks {
                            loads: loads.clone(),
                            ..Asks::of(seeded, Some(members))
                        }
                    };
                    let whole = Whole(asked(&theirs));
                    let decided = leasing.decide_under(GatewayId::LOCAL, &whole, Instant::now());
                    let decided = decided.expect("an answer").whole();
                    let whole = places(&decided.expect("a whole decision").0.decision);
                    assert!(!whole.is_empty(), "{policy}, seed {seed}: a pick");
                    let alone = bare.decide_without_lease(&asked(&ours), Instant::now());
                    let alone = places(&alone.expect("a decision").0);
                    assert_eq!(alone, whole, "{policy}, seed {seed}");
                    if policy == "least_load" && loads.is_some() {
                        assert_eq!(alone[0], 5, "seed {seed}: the shortest queue");
                    }
                }
            }
        }
    }

    /// A request that hands no list of its kind is decided over no worker,
    /// as the one entry decides it: a single with no list, a pair's prefill
    /// with none, and a pair's prefill over a single's list. A single over a
    /// pair's two lists or an encode request's three is shed too. So is a
    /// decode or an encode request that stands alone and hands no list. A
    /// pair's decode sent on its own is refused.
    #[test]
    fn a_decision_without_a_lease_over_no_list_names_no_worker() {
        use smg_worker::{test_support::worker_with_model_aliases, worker::WorkerType};

        use crate::engine::{EncodeLists, EncoderFacts, PairAdmission, PairLists, PartnerClasses};

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

        let engine = preset("round_robin");
        let document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        let paired = PolicyEngine::new(AppliedSpec::validate(&document).expect("a valid document"));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let side = |worker_type| {
            let url = format!("http://{worker_type}:8000");
            let worker = worker_with_model_aliases(&url, MODEL, &[], worker_type);
            worker.set_status(WorkerStatus::Ready);
            PublishedList::new(Arc::from([worker]))
        };
        let [prefill, decode, encoders] =
            [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode].map(side);
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
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
        let encode = EncodeLists {
            pair,
            encoders: encoders.lent(),
            encoders_places: None,
            encoders_stalled: &[],
            encoders_held: &[],
            facts: &[EncoderFacts {
                class: 1,
                ..EncoderFacts::default()
            }],
            prefill_classes: &[1],
        };
        let stating = |phase, members| {
            let mut stated = request();
            stated.facts.phase = phase;
            Asks::over(stated, members)
        };
        let none = (ShedReason::NoEndpoints, "none");
        let single = WorkerLists::Single(members.lent());
        let (both, three) = (WorkerLists::Pair(pair), WorkerLists::Encode(encode));
        for (phase, over, members, engine) in [
            (Phase::Single, "no list", WorkerLists::None, &engine),
            (Phase::Prefill, "no list", WorkerLists::None, &engine),
            (Phase::Decode, "no list", WorkerLists::None, &engine),
            (Phase::Encode, "no list", WorkerLists::None, &engine),
            (Phase::Prefill, "a single's list", single, &paired),
            (Phase::Single, "a pair's lists", both, &paired),
            (Phase::Single, "an encode request's lists", three, &paired),
        ] {
            let shed = engine.decide_without_lease(&stating(phase, members), Instant::now());
            let Ok((Decision::Shed(shed), None)) = shed else {
                panic!("{phase:?} over {over}: a shed, not {shed:?}");
            };
            assert_eq!(
                (shed.reason, shed.emptied_by),
                none,
                "{phase:?} over {over}"
            );
            let whole = Whole(stating(phase, members));
            let leased = engine.decide_under(GatewayId::LOCAL, &whole, Instant::now());
            let leased = leased.expect("an answer").whole().expect("no place").0;
            let Decision::Shed(theirs) = leased.decision else {
                panic!("{phase:?} over {over}: the one entry picks");
            };
            assert_eq!((theirs.reason, theirs.emptied_by), none);
        }
        let asks = stating(Phase::Decode, WorkerLists::None);
        let refused = paired.decide_without_lease(&asks, Instant::now());
        assert_eq!(refused.err(), Some(DecisionRefused::PairSide));
    }

    /// A request its sessions rule places alone says so on its pick, and its
    /// prompt is recorded by the rule's own word of it, in four forms. A pin
    /// followed by its one worker's record. A pin the whole decision
    /// follows: for a request held to places of its list, and under a
    /// binding with a shadow. And a key's first request under a rule whose
    /// own assignment places it behind the profile's filters, where no
    /// scorer and no picker of the profile is asked.
    ///
    /// Where the rule keeps what it places from the profile no form leaves
    /// a block of the prompt in the record: a gateway's sticky map hands
    /// none of them to its policy, so the policy's record never sees the
    /// prompt (`model_gateway/src/policies/registry.rs:736-739` for a pin
    /// under its cap, `:782-783` for the map's own assignment). Where the
    /// request is the profile's own, the rule's default, every form leaves
    /// the whole prompt for the session's worker and for no other, the pin
    /// by one record as the pin the whole decision follows. That prompt
    /// then goes back to the session's worker with no key.
    ///
    /// The forms under a shadow run again under a profile that reads the
    /// column whole, where the record's other write stands. No recorded
    /// sheet runs this.
    #[test]
    fn a_request_its_sessions_rule_places_alone_is_recorded_by_the_rules_word() {
        let forms = [
            (
                "a pin by one record",
                false,
                false,
                Assignment::Delegate,
                false,
            ),
            (
                "a pin held to places",
                false,
                true,
                Assignment::Delegate,
                false,
            ),
            ("a shadow", true, false, Assignment::Delegate, false),
            (
                "the rule's assignment",
                false,
                false,
                Assignment::Random,
                false,
            ),
            (
                "a shadow, the column",
                true,
                false,
                Assignment::Delegate,
                true,
            ),
            (
                "the rule's, the column",
                true,
                false,
                Assignment::Random,
                true,
            ),
        ];
        let words = [RulePlaced::Untold, RulePlaced::Told];
        for (kept, form) in words.into_iter().flat_map(|w| forms.map(|f| (w, f))) {
            let (form, shadowed, held, assignment, whole) = form;
            let case = format!("{kept:?}, {form}");
            let engine = {
                let mut document = PolicySpec::preset("manual").expect("a built-in preset");
                let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
                document.profiles = default.profiles;
                let binding = &mut document.bindings[0];
                binding.profile = Some("cache_aware".into());
                let rule = binding.sessions.as_mut().expect("the stock rule");
                rule.assignment = assignment;
                rule.rule_placed = kept;
                if shadowed {
                    let twin = document.profiles["cache_aware"].clone();
                    document.profiles.insert("twin".into(), twin);
                    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
                }
                if whole {
                    // A second warm-up stage that rules by another window:
                    // the live profile is named no row and reads the column
                    // whole, and its list keeps the record for the shadow.
                    let profile = document.profiles.get_mut("cache_aware");
                    profile.expect("the preset's profile").score.push(
                        ScoreStage::WarmupSlice(WarmupSliceParams {
                            weight: 1.0,
                            secs: Some(7),
                            ..WarmupSliceParams::default()
                        })
                        .into(),
                    );
                }
                document.generation = 1;
                let applied = AppliedSpec::validate(&document).expect("the document validates");
                let cap = applied.engine().affinity_rows_cap(MODEL, Phase::Single);
                assert_eq!(cap == 0, whole, "{case}: the form of the record's write");
                let record = applied.engine().pool_record_for(MODEL, Phase::Single);
                assert!(record.is_some(), "{case}: the list keeps a record");
                PolicyEngine::new(applied)
            };
            let list = pool(8);
            let members = PublishedList::new(Arc::clone(&list));
            let every: Vec<u32> = (0..8).collect();
            let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
            // One request sent to its first target and served there, held
            // to every place of its list where `held`: the place it went
            // to, who placed it, and whether the whole decision ran.
            let sent = |tokens: &[u32], key: u64, seed: u64, held: bool| {
                let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
                let mut asked = request();
                asked.facts.seed = seed;
                asked.facts.session_key = Identity::hashed(key);
                if held {
                    asked.constraints.places = Some(&every);
                }
                hashes.write_to(&mut asked.facts);
                let asks = Asks::of(asked, Some(&members));
                let now = Instant::now();
                let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                    .expect("a decision");
                let pick = *guard.decision().pick().expect("a pick");
                let target = guard.take(0).expect("the first target");
                guard.dispatched(now);
                guard.completed(
                    &Outcome {
                        side: Side::Single,
                        served: target.id,
                        status: 200,
                        class: TransportClass::Ok,
                        ttft_micros: 0,
                        tokens_in: 0,
                        tokens_out: 0,
                        duration_micros: 0,
                    },
                    now,
                );
                (target.position, pick.placed_by(), !pick.trace.drew_first())
            };
            for key in 1..=6_u32 {
                let seed = u64::from(key);
                // The key's first request, then one of a prompt nobody was
                // sent: the rule alone places the one this form is about.
                let (first, by, _) = sent(&prompt(key * 100_000), seed, seed, false);
                let fresh = prompt(key * 100_000 + 50_000);
                let asked = if assignment == Assignment::Delegate {
                    assert_eq!(by, PlacedBy::Stages, "{case}: the stages place a new key");
                    let (went, by, whole) = sent(&fresh, seed, 1, held);
                    assert_eq!((went, by), (first, PlacedBy::Rule), "{case}: key {key}");
                    // The whole decision follows the pin of a request held
                    // to places, and every pin under a shadow.
                    assert_eq!(whole, shadowed || held, "{case}: key {key}");
                    fresh
                } else {
                    assert_eq!(by, PlacedBy::Rule, "{case}: key {key}");
                    prompt(key * 100_000)
                };
                // What the record holds of the prompt the rule placed.
                let kept_of = recorded(&engine, &list, &asked);
                match kept {
                    RulePlaced::Untold => assert_eq!(kept_of, [0; 8], "{case}: key {key}"),
                    RulePlaced::Told => assert_eq!(kept_of, held_at(first), "{case}: key {key}"),
                }
                // The same prompt with no key: the policy places it, on the
                // session's worker where the record holds it there.
                let (went, by, _) = sent(&asked, 0, seed + 40, false);
                assert_eq!(by, PlacedBy::Stages, "{case}: no key");
                if kept == RulePlaced::Told {
                    assert_eq!(went, first, "{case}: key {key}: no key");
                }
            }
        }
    }

    /// [`Asks`] of a host that names the request's entry in its token tree
    /// and says whether it sends to a target of the pick: `to` for each.
    /// `page` is the tree's page it was last asked the entry by, zero while
    /// it was not asked.
    struct Sends<'a> {
        asks: Asks<'a>,
        to: bool,
        page: Cell<usize>,
    }

    impl RequestMaker for Sends<'_> {
        fn model(&self) -> &str {
            self.asks.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.asks.make(spec)
        }

        fn plain(&self) -> Option<Plain> {
            self.asks.plain()
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.asks.workers()
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.asks.loads()
        }

        fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
            self.page.set(page);
            self.asks.token_key(page)
        }

        fn sends_to(&self, _: &Target) -> bool {
            self.to
        }
    }

    /// A pin followed under a host that lends its token tree and names each
    /// request's entry there. Where the rule keeps the request from the
    /// profile its pin returns ahead of the lookup: the host is not asked
    /// for the entry and the tree holds no token of the prompt. Where the
    /// request is the profile's own the host is asked once, by the tree's
    /// own page, and the whole decision does not run. The tree then holds
    /// the whole prompt for the pin's worker alone, and none of a prompt
    /// whose host says it does not send to that worker: the hand-over asks
    /// the request's own maker which target its caller takes.
    #[test]
    fn a_followed_pin_is_stored_in_a_lent_tree_by_the_rules_word() {
        for kept in [RulePlaced::Untold, RulePlaced::Told] {
            let (engine, tree) = lending(default_under_sessions(kept));
            let list = pool(8);
            let members = PublishedList::new(Arc::clone(&list));
            let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
            // One request of `tokens`, which its host names as its entry,
            // sent to its first target and served there. Its host sends to
            // a target of its pick where `to`. The place it went to, its
            // pick, how often its host was asked for the entry, and by
            // which page.
            let sent = |tokens: &[u32], key: u64, seed: u64, to: bool| {
                let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
                let mut asked = request();
                asked.facts.seed = seed;
                asked.facts.session_key = Identity::hashed(key);
                hashes.write_to(&mut asked.facts);
                let asks = Asks {
                    key: Some(tokens),
                    ..Asks::of(asked, Some(&members))
                };
                let host = Sends {
                    asks,
                    to,
                    page: Cell::new(0),
                };
                let now = Instant::now();
                let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &host, now)
                    .expect("a decision");
                let pick = *guard.decision().pick().expect("a pick");
                let target = guard.take(0).expect("the first target");
                guard.dispatched(now);
                guard.completed(
                    &Outcome {
                        side: Side::Single,
                        served: target.id,
                        status: 200,
                        class: TransportClass::Ok,
                        ttft_micros: 0,
                        tokens_in: 0,
                        tokens_out: 0,
                        duration_micros: 0,
                    },
                    now,
                );
                (
                    target.position,
                    pick,
                    (host.asks.keyed.get(), host.page.get()),
                )
            };
            // The tokens of a prompt the tree holds, and the workers that
            // hold them.
            let stored = |tokens: &[u32]| {
                let found = tree.match_prefix_with_counts(tokens);
                (found.matched_token_count, found.matched_tenants)
            };
            let own = kept == RulePlaced::Told;
            for key in 1..=6_u32 {
                let seed = u64::from(key);
                let (pin, _, entry) = sent(&prompt(key * 100_000), seed, seed, true);
                assert_eq!(entry, (1, 16), "{kept:?}: key {key}: a new key's entry");
                let at = vec![TenantId::from(list[pin as usize].url())];
                // The key's pin, of a prompt nobody was sent.
                let fresh = prompt(key * 100_000 + 50_000);
                let (went, pick, entry) = sent(&fresh, seed, 1, true);
                assert_eq!(
                    (went, pick.placed_by()),
                    (pin, PlacedBy::Rule),
                    "{kept:?}: key {key}"
                );
                let whole = !pick.trace.drew_first();
                assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
                if own {
                    assert_eq!(
                        entry,
                        (1, 16),
                        "key {key}: the pin's entry, by the tree's page"
                    );
                    assert_eq!(stored(&fresh), (1_024, at), "key {key}: the pin's prompt");
                } else {
                    assert_eq!(entry, (0, 0), "key {key}: a kept pin's entry");
                    assert_eq!(stored(&fresh), (0, Vec::new()), "key {key}: a kept prompt");
                }
                // The key's pin again, where the host answers that its caller
                // sends to no target of the pick: the tree takes none of the
                // prompt. The helper still dispatches on the pin's worker, so
                // this holds what that answer does to the tree and no more.
                let refused = prompt(key * 100_000 + 70_000);
                let (went, _, entry) = sent(&refused, seed, 2, false);
                assert_eq!(went, pin, "{kept:?}: key {key}");
                assert_eq!(entry.0, u32::from(own), "{kept:?}: key {key}: the entry");
                assert_eq!(stored(&refused), (0, Vec::new()), "{kept:?}: key {key}");
            }
        }
    }

    /// A request with no key is never its sessions rule's, also where the
    /// profile's picker leaves the placement to the binding: the binding's
    /// placing assignment places such a request, under a rule and under
    /// none, and its pick keeps the stages' word. The rule's word is for
    /// its key alone, which the same assignment places.
    #[test]
    fn a_request_with_no_key_is_never_its_sessions_rules() {
        for ruled in [true, false] {
            let engine = {
                let mut document = PolicySpec::preset("manual").expect("a built-in preset");
                if !ruled {
                    document.bindings[0].sessions = None;
                }
                document.generation = 1;
                let applied = AppliedSpec::validate(&document).expect("the document validates");
                PolicyEngine::new(applied)
            };
            let list = pool(8);
            let members = PublishedList::new(Arc::clone(&list));
            let placed_by = |key: u64| {
                let mut asked = request();
                asked.facts.seed = 7;
                asked.facts.session_key = Identity::hashed(key);
                let asks = Asks::of(asked, Some(&members));
                let now = Instant::now();
                let guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                    .expect("a decision");
                guard.decision().pick().expect("a pick").placed_by()
            };
            assert_eq!(placed_by(0), PlacedBy::Stages, "no key, ruled: {ruled}");
            // A key is the rule's where a rule stands, and no one's else.
            let keyed = if ruled {
                PlacedBy::Rule
            } else {
                PlacedBy::Stages
            };
            assert_eq!(placed_by(9), keyed, "a key, ruled: {ruled}");
        }
    }

    /// A conversation whose host lends its token tree and names its entry
    /// there goes back to its worker however long it grows, whatever a
    /// turn's seed: the tree matches the whole prompt. The record of hashed
    /// prefixes holds a prompt's first 256 blocks, under the share a match
    /// needs once the prompt is past 13,653 tokens. The host is asked for
    /// the entry once a decision, and a worker its host took out of the tree
    /// holds no prompt. A binding that names a shadow is answered the same:
    /// its live decision is made inside the tree as well. So is a request
    /// whose profile reads a named signal: the copy that carries its list's
    /// signal cells is decided inside the tree too.
    #[test]
    fn a_long_conversation_goes_back_to_its_worker_by_the_token_tree() {
        for also in [Also::Nothing, Also::Shadow, Also::Signal] {
            let (engine, tree) = lent("cache_aware", also);
            let list = pool(8);
            let members = PublishedList::new(Arc::clone(&list));
            let sent = |tokens: &[u32], seed: u64| {
                let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
                let mut asked = request();
                asked.facts.seed = seed;
                hashes.write_to(&mut asked.facts);
                let asks = Asks {
                    key: Some(tokens),
                    ..Asks::of(asked, Some(&members))
                };
                let decided = decided(&engine, &asks);
                assert_eq!(asks.keyed.get(), 1, "the entry is asked for once: {also:?}");
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.first().expect("a target").position
            };
            let mut said: Vec<u32> = (0..16_000).collect();
            let first = sent(&said, 0);
            for turn in 1..32 {
                said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
                assert_eq!(sent(&said, u64::from(turn)), first, "turn {turn}: {also:?}");
            }
            // The tree is the host's to keep. With every worker taken out
            // of it before a request, the request's seed places it.
            let moved = (0..32).any(|seed| {
                for worker in list.iter() {
                    tree.remove_tenant_all(&TenantId::from(worker.url()));
                }
                sent(&said, seed) != first
            });
            assert!(moved, "a worker out of the tree holds no prompt: {also:?}");
        }
    }

    /// The entry is asked for where the host lends a tree of sent prefixes
    /// and the profile reads one, and nowhere else. A place one pick decides
    /// is answered ahead of the whole pass and never asks. A whole pass
    /// whose profile reads no record of sent prefixes never asks either, and
    /// neither does a decision whose host lends no tree.
    #[test]
    fn a_decision_that_is_lent_no_tree_asks_for_no_entry() {
        let tokens: Vec<u32> = (0..256).collect();
        let asked_of = |name: &str, lends: bool| {
            let engine = if lends {
                lent(name, Also::Nothing).0
            } else {
                preset(name)
            };
            let list = pool(4);
            let members = PublishedList::new(Arc::clone(&list));
            let asks = Asks {
                key: Some(&tokens),
                ..Asks::of(request(), Some(&members))
            };
            let alone = matches!(answered(&engine, &asks), Decided::Alone(_));
            (alone, asks.keyed.get())
        };
        assert_eq!(asked_of("round_robin", true), (true, 0));
        assert_eq!(asked_of("least_load", true), (false, 0));
        assert_eq!(asked_of("cache_aware", true), (false, 1));
        assert_eq!(asked_of("cache_aware", false), (false, 0));
    }

    /// A retry over a list where an engine holds several workers is decided
    /// inside the lent tree as a first attempt is. Its facts are a copy
    /// that names every worker of the engines it tried, and the copy goes
    /// to the same lookup with the request's maker: the host is asked for
    /// the entry once. The tree names the worker the first attempt was
    /// sent to, and the retry leaves that worker's engine.
    #[test]
    fn a_retry_over_ranks_is_decided_inside_the_lent_tree() {
        let (engine, _tree) = lent("cache_aware", Also::Nothing);
        let (list, engine_of) = engines(&[2, 2]);
        let members = PublishedList::new(Arc::clone(&list));
        let tokens: Vec<u32> = (0..256).collect();
        let sent = |tried: &[WorkerId]| {
            let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            hashes.write_to(&mut asked.facts);
            asked.facts.tried = tried;
            let asks = Asks {
                key: Some(&tokens),
                ..Asks::of(asked, Some(&members))
            };
            let decided = decided(&engine, &asks);
            assert_eq!(asks.keyed.get(), 1, "the entry is asked for once");
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position as usize
        };
        let first = sent(&[]);
        let again = sent(&[worker_id(list[first].as_ref())]);
        assert_ne!(engine_of[again], engine_of[first]);
    }

    /// What a stage heard of a lease: the word, the worker the lease told
    /// books (the target's own at a dispatch), and the lease's index.
    pub(in crate::engine::policy_engine) type Heard = Arc<Mutex<Vec<(&'static str, WorkerId, u8)>>>;

    /// The `random` picker, keeping what it hears of the words it names
    /// (`takes`). `panics` makes its pick panic. `alone` makes it place a
    /// request over a pool by itself, by the pool's own turn.
    #[derive(Debug)]
    struct Hears {
        heard: Heard,
        panics: bool,
        alone: bool,
        takes: EventSet,
    }

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            self.takes
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            let heard = ("issued", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
            let heard = ("dispatched", target.id, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_completed(&self, lease: &Lease, _: &Outcome) {
            let heard = ("completed", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_released(&self, lease: &Lease, _: ReleaseReason) {
            let heard = ("released", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_membership(&self, change: &Membership<'_>) {
            assert!(!self.panics, "this picker panics");
            let heard = match *change {
                Membership::Added(id) => ("joined", id, 0),
                Membership::Removed(id) => ("left", id, 0),
                // The ids that stand, one word each.
                Membership::Snapshot(ids) => {
                    let standing = ids.iter().map(|&id| ("standing", id, 0));
                    self.heard.lock().extend(standing);
                    return;
                }
                _ => return,
            };
            self.heard.lock().push(heard);
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        // A picker that places by itself refuses by no report.
        fn refuses_by_reports(&self) -> bool {
            !self.alone
        }

        fn picks_alone(&self) -> bool {
            self.alone
        }

        fn pick_alone(
            &self,
            pool: &PoolWord<'_>,
            keep: &mut dyn FnMut(usize) -> bool,
        ) -> Option<usize> {
            let turn = pool.word.fetch_add(1, Ordering::Relaxed) as usize % pool.rows.max(1);
            keep(turn).then_some(turn)
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            assert!(!self.panics, "this picker panics");
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The engine of the `random` preset with [`Hears`] in its pick's place,
    /// taking every word of a lease, and what it heard.
    pub(in crate::engine::policy_engine) fn hearing(panics: bool) -> (PolicyEngine, Heard) {
        hearing_as(panics, false, EventSet::LEASE)
    }

    /// The document of [`hearing`]'s engine, for a case that has its engine
    /// by another call, and what its picker hears.
    pub(in crate::engine::policy_engine) fn hearing_spec() -> (AppliedSpec, Heard) {
        hearing_spec_as(false, false, EventSet::LEASE)
    }

    /// [`hearing`] with a picker that places by itself where `alone`, and
    /// that names the words `takes`.
    fn hearing_as(panics: bool, alone: bool, takes: EventSet) -> (PolicyEngine, Heard) {
        let (spec, heard) = hearing_spec_as(panics, alone, takes);
        (PolicyEngine::new(spec), heard)
    }

    /// The document of [`hearing_as`]'s engine, and what its picker hears.
    fn hearing_spec_as(panics: bool, alone: bool, takes: EventSet) -> (AppliedSpec, Heard) {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let spec = spec_picked_by("hears", move || {
            Box::new(Hears {
                heard: Arc::clone(&kept),
                panics,
                alone,
                takes,
            })
        });
        (spec, heard)
    }

    /// The engine of the `random` preset with the picker `make` builds, a
    /// stage of the host's own build under `name`, in its pick's place.
    fn picked_by(
        name: &'static str,
        make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    ) -> PolicyEngine {
        PolicyEngine::new(spec_picked_by(name, make))
    }

    /// The document of [`picked_by`]'s engine.
    fn spec_picked_by(
        name: &'static str,
        make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    ) -> AppliedSpec {
        let mut stages = StageRegistry::builtin();
        let added = stages.picker(name, make);
        added.expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = 1;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = name.into();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        spec.expect("the stages hold the picker")
    }

    /// [`Hears`] as a stage of a host's own build that says it needs the
    /// lifecycle's calls and names no word of its own.
    #[derive(Debug)]
    struct Declares(Hears);

    impl Lifecycle for Declares {
        fn needs_lifecycle(&self) -> bool {
            true
        }

        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.0.on_decision_issued(lease, decision);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
            self.0.on_dispatched(lease, target, side, since);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.0.on_completed(lease, outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.on_released(lease, reason);
        }
    }

    impl Picker for Declares {
        fn name(&self) -> &'static str {
            "declares"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The words a stage of `engine` put into `heard` of two requests: one
    /// sent to its first target and completed, and one given up unsent. A
    /// stage panicked at `unheard` of the words, and no decision faulted.
    fn told(engine: &PolicyEngine, heard: &Heard, unheard: u64) -> Vec<&'static str> {
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let mut handle =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let first = handle.take(0).expect("the first target");
        handle.dispatched(now);
        handle.completed(
            &Outcome {
                side: Side::Single,
                served: first.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        drop(handle);
        let unsent =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        drop(unsent);
        assert_eq!((engine.faults(), engine.unheard()), (0, unheard));
        let words = heard.lock().iter().map(|heard| heard.0).collect();
        words
    }

    /// A stage hears a single's words from what the request's guard lends,
    /// with no lease on file: the issue on the pick's first target, the
    /// dispatch on the target taken under the index the word names, the
    /// end under the lease moved there, and a release under the lease as
    /// issued.
    #[test]
    fn the_stages_hear_a_singles_words_from_what_its_guard_lends() {
        let (engine, heard) = hearing(false);
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let mut handle =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let first = handle.take(0).expect("the first target");
        let taken = handle.take(1).expect("an alternate");
        handle.dispatched(now);
        handle.completed(
            &Outcome {
                side: Side::Single,
                served: taken.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        drop(handle);
        let unsent =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let given_up = unsent
            .decision()
            .pick()
            .and_then(|pick| pick.targets.first().copied())
            .expect("the first target");
        drop(unsent);
        assert_eq!(
            *heard.lock(),
            [
                ("issued", first.id, 0),
                ("dispatched", taken.id, 1),
                ("completed", taken.id, 1),
                ("issued", given_up.id, 0),
                ("released", given_up.id, 0),
            ]
        );
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A placement says a single's words as a guard driven by hand says
    /// them, and names the list's own worker at the place of the target
    /// taken. Four requests, each once through a placement and once
    /// through a guard: one sent to an alternate and ended, one sent and
    /// dropped, one dropped unsent, and one whose pick has no target at the
    /// index taken, which makes no placement and gives its lease up.
    #[test]
    fn a_placement_says_a_singles_words_as_its_guard_says_them() {
        let (engine, heard) = hearing(false);
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let ended = |side: Side, served: WorkerId| Outcome {
            side,
            served,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        let whole = || {
            let answer = answered(engine, &asks).whole();
            let (decided, later) = answer.expect("a whole decision");
            (decided, later, Instant::now())
        };
        // One request as a placement with the target at `index` taken, and
        // that target.
        let taken_at = |index: usize| {
            let (decided, later, now) = whole();
            let targets = decided.decision.pick().expect("a pick").targets;
            let taken = targets.as_slice().get(index).copied();
            let worker = Arc::clone(&list[taken.map_or(0, |taken| taken.position as usize)]);
            let placement = crate::tracking::Placement::taken(
                engine.loaded(),
                decided,
                later,
                now,
                index,
                worker,
                None,
            );
            (placement, taken)
        };
        let by_hand = || {
            let (decided, later, now) = whole();
            (
                DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now),
                now,
            )
        };
        // What the stage heard since the last call: each word with its
        // worker, and each word with its lease's index.
        let words = || {
            let heard = std::mem::take(&mut *heard.lock());
            let told: Vec<_> = heard.iter().map(|word| (word.0, word.2)).collect();
            (heard, told)
        };

        // Sent to an alternate and ended. Its host wrote another side and no
        // worker into the outcome.
        let (placement, alternate) = taken_at(1);
        let mut placement = placement.expect("a placement");
        let alternate = alternate.expect("a target");
        let at = alternate.position as usize;
        assert!(Arc::ptr_eq(&placement.worker, &list[at]));
        assert_eq!(
            (placement.id(), placement.decode_id()),
            (alternate.id, None)
        );
        assert!(placement.decode.is_none() && placement.routing_key.is_none());
        placement.routing_key = Some("a-key");
        let mut placement = placement.kept();
        assert_eq!(
            placement.routing_key, None,
            "a kept placement borrows nothing"
        );
        assert!(placement.owes_events());
        placement.sent(Side::Single);
        placement.sent(Side::Single);
        assert!(placement.owes_events(), "sent, and not ended");
        placement.ended(Side::Single, ended(Side::Decode, WorkerId(0)));
        assert!(!placement.owes_events());
        drop(placement);
        // Sent and dropped: a cancelled end.
        let mut placement = taken_at(0).0.expect("a placement");
        placement.sent(Side::Single);
        drop(placement);
        // Dropped unsent: a release.
        drop(taken_at(0).0.expect("a placement"));
        // No target at the index taken: no placement, and the lease is given
        // up.
        assert!(taken_at(9).0.is_none());
        let (heard_of_placements, through_placements) = words();

        let (mut guard, now) = by_hand();
        let taken = guard.take(1).expect("a target");
        guard.dispatched(now);
        guard.completed(&ended(Side::Single, taken.id), now);
        drop(guard);
        let (mut guard, now) = by_hand();
        guard.dispatched(now);
        drop(guard);
        drop(by_hand());
        let (mut guard, _) = by_hand();
        assert!(guard.take(9).is_none());
        drop(guard);
        assert_eq!(through_placements, words().1);
        assert_eq!(
            through_placements,
            [
                ("issued", 0),
                ("dispatched", 1),
                ("completed", 1),
                ("issued", 0),
                ("dispatched", 0),
                ("completed", 0),
                ("issued", 0),
                ("released", 0),
                ("issued", 0),
                ("released", 0),
            ]
        );
        // The dispatch names the target taken, and the end is told under
        // the lease moved there.
        let on_the_alternate = [heard_of_placements[1].1, heard_of_placements[2].1];
        assert_eq!(on_the_alternate, [alternate.id; 2]);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// [`Hears`] as a stage that also keeps the outcome of each end it is
    /// told.
    #[derive(Debug)]
    struct Ends(Hears, Arc<Mutex<Vec<Outcome>>>);

    impl Lifecycle for Ends {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.0.on_decision_issued(lease, decision);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
            self.0.on_dispatched(lease, target, side, since);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.0.on_completed(lease, outcome);
            self.1.lock().push(*outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.on_released(lease, reason);
        }
    }

    impl Picker for Ends {
        fn name(&self) -> &'static str {
            "ends"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// An engine whose one stage is [`Ends`], with what the stage hears
    /// and the outcomes it keeps.
    fn hearing_ends() -> (&'static PolicyEngine, Heard, Arc<Mutex<Vec<Outcome>>>) {
        let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
        let kept = (Arc::clone(&heard), Arc::clone(&ends));
        let engine = picked_by("ends", move || {
            let hears = Hears {
                heard: Arc::clone(&kept.0),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Ends(hears, Arc::clone(&kept.1)))
        });
        (Box::leak(Box::new(engine)), heard, ends)
    }

    /// A single request's placement goes out under any side its host names
    /// but an encode side, as it ends under any. A host that names its one
    /// call a decode side or a prefill side says the request's dispatch:
    /// the stage counts the request as sent while it runs, and a drop
    /// behind the send is an end of class cancelled, not a release.
    #[test]
    fn a_single_requests_placement_is_sent_under_the_side_its_host_names() {
        let (engine, heard, ends) = hearing_ends();
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let words = || std::mem::take(&mut *heard.lock());
        for side in [Side::Decode, Side::Prefill, Side::Single] {
            let asks = Asks::of(request(), Some(&members));
            let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
            let first = decided
                .decision
                .pick()
                .and_then(|pick| pick.targets.first().copied());
            let first = first.expect("a target");
            let worker = Arc::clone(&list[first.position as usize]);
            let now = Instant::now();
            let placement = crate::tracking::Placement::taken(
                engine.loaded(),
                decided,
                later,
                now,
                0,
                worker,
                None,
            );
            let mut placement = placement.expect("a placement");
            assert_eq!(words(), [("issued", first.id, 0)]);
            placement.sent(side);
            assert_eq!(words(), [("dispatched", first.id, 0)], "{side:?}");
            drop(placement);
            assert_eq!(words(), [("completed", first.id, 0)], "{side:?}");
            let cancelled = std::mem::take(&mut *ends.lock());
            let classes: Vec<TransportClass> = cancelled.iter().map(|end| end.class).collect();
            assert_eq!(classes, [TransportClass::Cancelled], "{side:?}");
        }
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// An encode side's end says nothing through a placement, as its send
    /// says nothing: no lease stands for such a side. A single request's
    /// placement is told one before the request is sent and one behind its
    /// send. No stage hears a word of either, and the request is still
    /// owed its end. Its own end behind them is said once, under the
    /// request's side and the worker of the target taken, whatever its
    /// host wrote there.
    #[test]
    fn an_encode_sides_end_through_a_placement_says_nothing() {
        let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
        let kept = (Arc::clone(&heard), Arc::clone(&ends));
        let engine = picked_by("ends", move || {
            let hears = Hears {
                heard: Arc::clone(&kept.0),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Ends(hears, Arc::clone(&kept.1)))
        });
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first().copied());
        let first = first.expect("a target");
        let worker = Arc::clone(&list[first.position as usize]);
        let now = Instant::now();
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            now,
            0,
            worker,
            None,
        );
        let mut placement = placement.expect("a placement");
        let ended = |side: Side| Outcome {
            side,
            served: WorkerId(77),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        let words = || std::mem::take(&mut *heard.lock());
        assert_eq!(words(), [("issued", first.id, 0)]);
        // Unsent: no dispatch is said for it, and no end.
        placement.ended(Side::Encode, ended(Side::Encode));
        assert!(placement.owes_events(), "the request has not ended");
        assert!(words().is_empty(), "an encode side's end is no word");
        placement.sent(Side::Single);
        assert_eq!(words(), [("dispatched", first.id, 0)]);
        // Sent: the request's lease stands.
        placement.ended(Side::Encode, ended(Side::Encode));
        assert!(placement.owes_events(), "the request has not ended");
        assert!(words().is_empty(), "an encode side's end is no word");
        assert!(ends.lock().is_empty());
        // The request's own end. Its host wrote another side there.
        placement.ended(Side::Single, ended(Side::Decode));
        assert!(!placement.owes_events());
        assert_eq!(words(), [("completed", first.id, 0)]);
        let told = Outcome {
            side: Side::Single,
            served: first.id,
            ..ended(Side::Decode)
        };
        assert_eq!(*ends.lock(), [told]);
        drop(placement);
        assert!(words().is_empty(), "the drop says nothing more");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A placement under no lease owes nothing and says nothing: a place's,
    /// and a whole decision's under a document no stage of which takes a
    /// word. It names the list's own worker, names the profile that picked
    /// by its label, and takes its worker's id when it is asked, and no
    /// lease's. The profile that picks is the document's second: a spare
    /// one of another name stands ahead of it.
    #[test]
    fn a_placement_under_no_lease_owes_nothing_and_says_nothing() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        document.generation = 1;
        let spare = document.profiles["round_robin"].clone();
        document.profiles.insert("a_spare".into(), spare);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let ended = Outcome {
            side: Side::Single,
            served: WorkerId(0),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };

        let pick = placed(engine, &Asks::of(request(), Some(&members)));
        assert_eq!(pick.profile, 1, "the document's second profile picked");
        let worker = &list[pick.place as usize];
        let label = engine.loaded().label_at(pick.profile);
        let mut placement = crate::tracking::Placement::alone(Arc::clone(worker), label);
        assert!(!placement.owes_events());
        placement.sent(Side::Single);
        placement.ended(Side::Single, ended);
        assert!(Arc::ptr_eq(&placement.worker, worker));
        assert_eq!(placement.id(), worker_id(worker.as_ref()));
        assert_eq!(
            (placement.decode_id(), placement.profile),
            (None, "round_robin")
        );
        assert!(placement.decode.is_none() && placement.routing_key.is_none());

        // A retry is decided whole, and under this document with no lease.
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let picked = decided.decision.pick().expect("a pick");
        let first = picked.targets.first().expect("a target");
        let (profile, target) = (picked.trace.profile, *first);
        assert_eq!(profile, 1, "the document's second profile picked");
        let worker = &list[target.position as usize];
        let now = Instant::now();
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            now,
            0,
            Arc::clone(worker),
            None,
        );
        let mut placement = placement.expect("a placement");
        assert!(!placement.owes_events());
        placement.sent(Side::Single);
        placement.ended(Side::Single, ended);
        assert!(Arc::ptr_eq(&placement.worker, worker));
        let named = (placement.id(), placement.profile);
        assert_eq!(named, (target.id, "round_robin"));
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// [`Hears`] as a stage that keeps the span each dispatch states since
    /// its decision.
    #[derive(Debug)]
    struct Spans(Hears, Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Spans {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, since: u32) {
            self.1.lock().push(since);
        }
    }

    impl Picker for Spans {
        fn name(&self) -> &'static str {
            "spans"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A placement dates its lease at the instant its host hands, the
    /// decision's, and reads the clock itself at a send: the span the
    /// dispatch states counts from the one to the other.
    #[test]
    fn a_placements_dispatch_counts_from_the_decisions_instant() {
        let spans = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&spans);
        let engine = picked_by("spans", move || {
            let hears = Hears {
                heard: Heard::default(),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Spans(hears, Arc::clone(&kept)))
        });
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first());
        let worker = Arc::clone(&list[first.expect("a target").position as usize]);
        // The decision stood a second before its request went out.
        let second = Duration::from_secs(1);
        let decided_at = Instant::now().checked_sub(second);
        let decided_at = decided_at.expect("an earlier instant");
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            decided_at,
            0,
            worker,
            None,
        );
        let mut placement = placement.expect("a placement");
        placement.sent(Side::Single);
        drop(placement);
        let stated = spans.lock().clone();
        assert_eq!(stated.len(), 1, "one dispatch");
        assert!(stated[0] >= 1_000_000, "{stated:?}");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A caller ends its request with an outcome that names no side and no
    /// worker ([`Outcome::reported`]), and adds what it measured. The
    /// placement writes the worker of the target it took, and what the
    /// caller measured stands.
    #[test]
    fn a_placement_ends_under_an_outcome_that_names_no_worker() {
        let (engine, _, ends) = hearing_ends();
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let picked = decided.decision.pick().expect("a pick");
        let first = *picked.targets.first().expect("a target");
        let worker = Arc::clone(&list[first.position as usize]);
        let now = Instant::now();
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            now,
            0,
            worker,
            None,
        );
        let mut placement = placement.expect("a placement");
        let measured = Outcome {
            tokens_out: 34,
            duration_micros: 5_000,
            ..Outcome::reported(200, TransportClass::Ok)
        };
        assert_eq!(measured.served, WorkerId(0), "no worker named");
        placement.ended(Side::Single, measured);
        let told = Outcome {
            served: first.id,
            ..measured
        };
        assert_eq!(*ends.lock(), [told]);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A word of a lease is told where a stage takes it and to nobody
    /// where none does. One request is sent and completed and one is given
    /// up unsent: a document whose one stage takes one kind of word hears
    /// that kind of the two leases and no other. A completion and a release
    /// are told apart.
    #[test]
    fn a_word_no_stage_takes_is_told_to_nobody() {
        let told_of = |kind: EventKind| {
            let (engine, heard) = hearing_as(false, false, EventSet::NONE.with(kind));
            told(&engine, &heard, 0)
        };
        assert_eq!(told_of(EventKind::Issued), ["issued", "issued"]);
        assert_eq!(told_of(EventKind::Dispatched), ["dispatched"]);
        assert_eq!(told_of(EventKind::Completed), ["completed"]);
        assert_eq!(told_of(EventKind::Released), ["released"]);
    }

    /// A stage of a host's own build that says it needs the lifecycle's
    /// calls and names no word of its own is told every word on a lease:
    /// the issue, the dispatch and the completion of a request sent, and
    /// the issue and the release of one given up.
    #[test]
    fn a_stage_that_needs_the_calls_and_names_no_word_is_told_each() {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let engine = picked_by("declares", move || {
            Box::new(Declares(Hears {
                heard: Arc::clone(&kept),
                panics: false,
                alone: false,
                takes: EventSet::NONE,
            }))
        });
        assert_eq!(engine.document.hears, EventSet::LEASE);
        assert_eq!(
            told(&engine, &heard, 0),
            ["issued", "dispatched", "completed", "issued", "released"]
        );
    }

    /// [`Hears`] as a stage that panics at one word of a lease, the kind it
    /// names, and keeps each other word.
    #[derive(Debug)]
    struct StopsAt(Hears, EventKind);

    impl StopsAt {
        /// Panics where `word` is the stage's own.
        fn stop(&self, word: EventKind) {
            assert!(self.1 != word, "this stage panics at that word");
        }
    }

    impl Lifecycle for StopsAt {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.stop(EventKind::Issued);
            self.0.on_decision_issued(lease, decision);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
            self.stop(EventKind::Dispatched);
            self.0.on_dispatched(lease, target, side, since);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.stop(EventKind::Completed);
            self.0.on_completed(lease, outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.stop(EventKind::Released);
            self.0.on_released(lease, reason);
        }
    }

    impl Picker for StopsAt {
        fn name(&self) -> &'static str {
            "stops-at"
        }

        fn refuses_by_reports(&self) -> bool {
            self.0.refuses_by_reports()
        }

        fn picks_alone(&self) -> bool {
            self.0.picks_alone()
        }

        fn pick_alone(
            &self,
            pool: &PoolWord<'_>,
            keep: &mut dyn FnMut(usize) -> bool,
        ) -> Option<usize> {
            self.0.pick_alone(pool, keep)
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// What [`told`] reads of an engine whose stage panics at `word`, with
    /// `unheard` held panics counted and no fault. The stage's picker
    /// places by itself where `alone`.
    fn told_past(word: EventKind, alone: bool, unheard: u64) -> Vec<&'static str> {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let engine = picked_by("stops-at", move || {
            let hears = Hears {
                heard: Arc::clone(&kept),
                panics: false,
                alone,
                takes: EventSet::LEASE,
            };
            Box::new(StopsAt(hears, word))
        });
        told(&engine, &heard, unheard)
    }

    /// A stage that panics at a first decision's issue is held and counted
    /// with the words a stage did not hear, on the whole pass and on the
    /// short path of a picker that places by itself. The decision answers
    /// its pick and its lease, so the guard tells the lease's dispatch and
    /// its end as any lease's: of a request sent and one given up, the
    /// stage hears every word but the two issues. Before, each request was
    /// answered a fault.
    ///
    /// The classic pass tells a policy nothing where a pick is issued: it
    /// asks the policy for one pick
    /// (`model_gateway/src/routers/common/placement.rs:1391`).
    #[test]
    fn a_stage_that_panics_at_a_first_issue_costs_the_request_nothing() {
        for alone in [false, true] {
            // The two held panics alone are counted.
            assert_eq!(
                told_past(EventKind::Issued, alone, 2),
                ["dispatched", "completed", "released"],
                "a picker that places by itself: {alone}"
            );
        }
    }

    /// A stage that panics at a later word of a lease is held under the
    /// same catch and counted the same way, as before this catch was the
    /// issue's too: the request goes on, and the stage hears every other
    /// word of the request sent and of the one given up.
    #[test]
    fn a_stage_that_panics_at_a_later_word_of_a_lease_is_held_and_counted() {
        let past = |word| told_past(word, false, 1);
        assert_eq!(
            past(EventKind::Dispatched),
            ["issued", "completed", "issued", "released"]
        );
        assert_eq!(
            past(EventKind::Completed),
            ["issued", "dispatched", "issued", "released"]
        );
        assert_eq!(
            past(EventKind::Released),
            ["issued", "dispatched", "completed", "issued"]
        );
    }

    /// A decision that panics costs its request alone and is counted.
    #[test]
    fn a_decision_that_panics_is_refused_and_counted() {
        let (engine, heard) = hearing(true);
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        for faults in 1..=2 {
            assert_eq!(
                engine
                    .decide_under(GatewayId::LOCAL, &asks, Instant::now())
                    .err(),
                Some(DecisionRefused::Faulted)
            );
            assert_eq!(engine.faults(), faults);
        }
        assert!(heard.lock().is_empty());
    }

    /// A worker that joins or leaves is told to the document's stages under
    /// the id a decision names it by, in the order its host says the two. A
    /// stage that panics at the word is held and counted: the call returns.
    #[test]
    fn a_worker_that_joins_or_leaves_is_told_to_the_documents_stages() {
        let list = pool(2);
        let id = |at: usize| worker_id(list[at].as_ref());
        let (engine, heard) = hearing(false);
        engine.joined(list[0].url());
        engine.forget(list[1].url());
        assert_eq!(*heard.lock(), [("joined", id(0), 0), ("left", id(1), 0)]);
        assert_eq!(engine.unheard(), 0);
        let (engine, heard) = hearing(true);
        engine.joined(list[0].url());
        engine.forget(list[1].url());
        assert!(heard.lock().is_empty());
        assert_eq!(engine.unheard(), 2);
    }

    /// One request of `asks` decided at `now`, sent to its first target and
    /// served there: that target.
    fn served(engine: &PolicyEngine, asks: &Asks<'_>, now: Instant) -> Target {
        let mut guard =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, asks, now).expect("a decision");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        let outcome = Outcome {
            side: Side::Single,
            served: target.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        guard.completed(&outcome, now);
        target
    }

    /// A worker said gone and back at its address serves its sessions
    /// again, with no join said: the session table is not told a leave, as
    /// the classic policies' map is not. A key is pinned on a worker that
    /// leaves, and is served by the worker that stayed for twice the time
    /// the table holds an absent worker's pins, a quarter of that time
    /// apart, by the engine's own number. Back in the list, its first
    /// worker serves it again.
    #[test]
    fn a_worker_said_gone_and_back_at_its_address_serves_its_sessions_again() {
        let (engine, both) = (preset("manual"), pool(2));
        let start = Instant::now();
        let absence = Duration::from_micros(engine.spec().engine().pins_absence_micros());
        // The key's request over `list` in quarter `turn`: its worker.
        let sent = |list: &Arc<[Arc<dyn Worker>]>, turn: u32| {
            let members = PublishedList::new(Arc::clone(list));
            let mut keyed = request();
            keyed.facts.session_key = Identity::hashed(1);
            let asks = Asks::of(keyed, Some(&members));
            served(&engine, &asks, start + absence / 4 * turn).id
        };
        let first = sent(&both, 0);
        let at = usize::from(worker_id(both[1].as_ref()) == first);
        let (gone, other) = (&both[at], Arc::from([Arc::clone(&both[1 - at])]));
        engine.forget(gone.url());
        for turn in 1..=8 {
            assert_ne!(sent(&other, turn), first);
        }
        assert_eq!(sent(&both, 9), first, "it has its key again");
    }

    /// The places sixteen decisions of one request that states `demands`
    /// name over `members`, each as a place or as a whole decision's first
    /// target, or what the request is shed for.
    fn places_named(
        engine: &PolicyEngine,
        members: &PublishedList,
        demands: Demands,
    ) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
        places_named_among(engine, members, demands, None)
    }

    /// [`places_named`], for a request its host holds to `among`.
    fn places_named_among(
        engine: &PolicyEngine,
        members: &PublishedList,
        demands: Demands,
        among: Option<&[u32]>,
    ) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
        let mut stated = request();
        stated.constraints.demands = demands;
        stated.constraints.places = among;
        let asks = Asks::of(stated, Some(members));
        let mut places = BTreeSet::new();
        for _ in 0..16 {
            let place = match answered(engine, &asks) {
                Decided::Alone(pick) => pick.place,
                Decided::Whole(decided, _) => match decided.decision {
                    Decision::Pick(pick) => pick.targets.first().expect("a target").position,
                    Decision::Shed(shed) => return Err((shed.reason, shed.emptied_by)),
                    Decision::Queue(_) => panic!("a queue ticket"),
                },
            };
            places.insert(place);
        }
        Ok(places)
    }

    /// A request that states a hard demand is decided among the workers of
    /// its list that meet it, by what the list's host said of each worker
    /// when it published the list:
    /// every pick lands on one of them, and each of them takes its turn. A
    /// demand no worker meets sheds the list with no endpoint, under that
    /// demand's stage. Of a list its host said nothing of, no worker takes
    /// images and each holds a request of any size.
    #[test]
    fn a_request_that_states_a_demand_is_decided_among_the_workers_that_meet_it() {
        let engine = preset("round_robin");
        let list = pool(4);
        // The second and the fourth worker take images; the fourth states
        // a window, each of the others a runtime.
        let (sees, bounded) = (Arc::clone(&list[1]), Arc::clone(&list[3]));
        let said = PublishedList::with_facts(Arc::clone(&list), |worker| {
            let bounded = Arc::ptr_eq(worker, &bounded);
            // Each fact behind the rest of the type: a further fact of a
            // worker adds its field.
            let seeing = FixedFacts {
                vision: bounded || Arc::ptr_eq(worker, &sees),
                ..FixedFacts::default()
            };
            FixedFacts {
                demands: DemandFacts {
                    window: if bounded { 4_096 } else { 0 },
                    runtime: if bounded { 0 } else { 7 },
                    ..DemandFacts::default()
                },
                ..seeing
            }
        });
        let silent = PublishedList::new(Arc::clone(&list));
        let asked = |members, demands| places_named(&engine, members, demands);
        let pixels = Demands {
            vision: true,
            ..Demands::default()
        };
        let long = Demands {
            input_window: 5_000,
            ..Demands::default()
        };
        assert_eq!(asked(&said, pixels), Ok(BTreeSet::from([1, 3])));
        assert_eq!(asked(&said, long), Ok(BTreeSet::from([0, 1, 2])));
        let both = Demands {
            vision: true,
            ..long
        };
        assert_eq!(asked(&said, both), Ok(BTreeSet::from([1])));
        let stays = Demands {
            runtime: 7,
            ..Demands::default()
        };
        assert_eq!(asked(&said, stays), Ok(BTreeSet::from([0, 1, 2])));
        // Held to places by its host too: the places that meet the demand,
        // every place where each meets it, and the demand's stage where
        // none does. The classic router's tests of a retry are one test of
        // a candidate too
        // (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:779-782`
        // at main `f55a0a6bf`).
        let among = |places| places_named_among(&engine, &said, pixels, Some(places));
        assert_eq!(among(&[0, 1, 2]), Ok(BTreeSet::from([1])));
        assert_eq!(among(&[1, 3]), Ok(BTreeSet::from([1, 3])));
        let blind = Err((ShedReason::NoEndpoints, "vision"));
        assert_eq!(among(&[0, 2]), blind);
        // Nobody meets both: the second worker alone takes images and is
        // too small, and the others take none. The list is shed under the
        // later of the two demands in a host's order, which is neither
        // the first worker's failure nor the last one's.
        let narrow = PublishedList::with_facts(Arc::clone(&list), |worker| {
            let small = Arc::ptr_eq(worker, &sees);
            let seeing = FixedFacts {
                vision: small,
                ..FixedFacts::default()
            };
            FixedFacts {
                demands: DemandFacts {
                    window: if small { 4_096 } else { 0 },
                    ..DemandFacts::default()
                },
                ..seeing
            }
        });
        let too_small = Err((ShedReason::NoEndpoints, "context-window"));
        assert_eq!(asked(&narrow, both), too_small);
        // Nothing said of the list: no worker takes images. The classic
        // router reads a worker with no label as taking them
        // (`model_gateway/src/routers/grpc/multimodal/capability.rs:98-105`
        // at main `c22acc3b9`): a host that follows it says so of each such
        // worker when it makes the list's value.
        assert_eq!(asked(&silent, long), Ok(BTreeSet::from([0, 1, 2, 3])));
        assert_eq!(asked(&silent, pixels), blind);
    }

    /// Over a list long enough for the passes that draw, a request that
    /// states a demand is answered a worker that meets it at every
    /// decision, whatever the picker: those passes step aside for a request
    /// held to places.
    #[test]
    fn a_demand_holds_under_a_pick_that_draws_over_a_long_list() {
        for name in [
            "random",
            "power_of_two",
            "round_robin",
            "consistent_hashing",
        ] {
            let engine = preset(name);
            let list = pool(32);
            // Every fourth worker takes images.
            let sees: Vec<_> = list.iter().step_by(4).cloned().collect();
            let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
                vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
                ..FixedFacts::default()
            });
            let (mut landed, mut firsts) = (BTreeSet::new(), BTreeSet::new());
            for seed in 0..64 {
                let mut stated = request();
                stated.facts.seed = seed;
                // The ring places a request by its key.
                if name == "consistent_hashing" {
                    stated.facts.session_key = Identity::hashed(seed + 1);
                }
                stated.constraints.demands = Demands {
                    vision: true,
                    ..Demands::default()
                };
                let asks = Asks::of(stated, Some(&members));
                let outcome = decided(&engine, &asks);
                let pick = outcome.decision.pick().expect("a pick");
                firsts.insert(pick.targets.first().expect("a target").position);
                let places = pick.targets.iter().map(|target| target.position);
                landed.extend(places);
            }
            // Every target takes images, the alternates too, and the first
            // targets are more than one worker.
            let seeing = landed.iter().all(|place| place % 4 == 0);
            assert!(seeing, "{name}: {landed:?} take images");
            assert!(firsts.len() > 1, "{name}: {firsts:?}");
        }
    }

    /// Where a binding of the request's phase pins a key, a request that
    /// states a hard demand is decided among the workers that meet it, as
    /// a request held to places is: a pin on a worker that meets the
    /// demand is followed, and a pin on one that does not is not. That is
    /// the gateway's classic answer: it narrows its candidates by the
    /// request's own test before its policy runs
    /// (`model_gateway/src/routers/common/placement.rs:508-509` at main
    /// `f55a0a6bf`), and its sticky policy looks a key's pinned workers up
    /// among the workers it is handed and picks another of them where none
    /// is there (`model_gateway/src/policies/manual.rs:303-308`).
    #[test]
    fn a_demand_under_a_pinned_key_is_decided_among_the_workers_that_meet_it() {
        // The stock sessions rule: its binding pins a key.
        let engine = preset("manual");
        let list = pool(8);
        // Every other worker takes images.
        let sees: Vec<_> = list.iter().step_by(2).cloned().collect();
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
            ..FixedFacts::default()
        });
        // One request of `key` that states `demands`, served by its first
        // target: that target's place.
        let sent = |key: u64, demands: Demands| {
            let mut keyed = request();
            keyed.facts.seed = key;
            keyed.facts.session_key = Identity::hashed(key);
            keyed.constraints.demands = demands;
            let asks = Asks::of(keyed, Some(&members));
            served(&engine, &asks, Instant::now()).position
        };
        let pixels = Demands {
            vision: true,
            ..Demands::default()
        };
        let (mut followed, mut left) = (0, 0);
        for key in 1..=16 {
            // The key's first request states nothing and is pinned.
            let pin = sent(key, Demands::default());
            assert_eq!(sent(key, Demands::default()), pin, "key {key}: its pin");
            let went = sent(key, pixels);
            assert_eq!(went % 2, 0, "key {key}: place {went} takes images");
            if pin % 2 == 0 {
                assert_eq!(went, pin, "key {key}: its pin meets the demand");
                followed += 1;
            } else {
                left += 1;
            }
        }
        assert!(
            followed > 0 && left > 0,
            "{followed} pins followed, {left} left"
        );
    }

    /// One worker states runtime 7 and the other states no runtime.
    /// A request for runtime 9 is shed under `wires` with no endpoint.
    /// The shed takes no lease ID and sends no issue event.
    #[test]
    fn a_runtime_demand_no_member_meets_is_shed_without_a_lease() {
        let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
        let list = pool(2);
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            demands: DemandFacts {
                runtime: if Arc::ptr_eq(worker, &list[0]) { 7 } else { 0 },
                ..DemandFacts::default()
            },
            ..FixedFacts::default()
        });
        let mut asked = request();
        asked.constraints.demands.runtime = 9;

        let outcome = decided(&engine, &Asks::of(asked, Some(&members)));
        let Decision::Shed(shed) = outcome.decision else {
            panic!("expected a shed for runtime 9, got {:?}", outcome.decision);
        };
        assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::NoEndpoints, "wires")
        );
        assert!(outcome.lease.is_none());
        assert_eq!(
            engine.leases.0.load(Ordering::Relaxed),
            0,
            "no lease ID taken"
        );
        assert!(heard.lock().is_empty(), "no lease issue reaches the picker");
    }

    /// The workers state decoding values 5 and 0.
    /// A request for 7 is shed under `decoding`.
    /// A request for 5 picks the worker that states 5.
    #[test]
    fn a_constraint_no_member_decides_alike_is_shed_under_its_own_stage() {
        let engine = preset("random");
        let list = pool(2);
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            demands: DemandFacts {
                decoding: if Arc::ptr_eq(worker, &list[0]) { 5 } else { 0 },
                ..DemandFacts::default()
            },
            ..FixedFacts::default()
        });
        let mut asks = Asks::of(request(), Some(&members));
        asks.request.constraints.demands.decoding = 7;
        let missed = decided(&engine, &asks).decision;
        let Decision::Shed(shed) = missed else {
            panic!("expected a shed for decoding 7, got {missed:?}");
        };
        assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::NoEndpoints, "decoding")
        );

        asks.request.constraints.demands.decoding = 5;
        let matched = decided(&engine, &asks).decision;
        let pick = matched.pick().expect("a worker states decoding 5");
        let first = pick.targets.first().expect("a first target");
        assert_eq!(first.id, worker_id(list[0].as_ref()));
    }

    /// Vision and reference demands select the workers that meet them.
    /// Each excluded worker fails at its first unmet demand.
    /// The latest of those demands names the shed when the pool empties.
    #[test]
    fn a_media_demand_keeps_the_members_that_take_it_and_names_its_shed() {
        use crate::engine::References;

        let engine = preset("random");
        let members_with = |states: &[(bool, bool, bool)]| {
            let list = pool(states.len());
            let mut states = states.iter().copied();
            PublishedList::with_facts(Arc::clone(&list), |_| {
                let (vision, references, host_pipeline) =
                    states.next().expect("media facts for each worker");
                FixedFacts {
                    vision,
                    demands: DemandFacts {
                        runtime: 7,
                        references,
                        host_pipeline,
                        ..DemandFacts::default()
                    },
                    ..FixedFacts::default()
                }
            })
        };
        let mixed = members_with(&[
            (true, false, false),
            (true, true, true),
            (false, true, false),
        ]);
        let id = |at: usize| worker_id(mixed.workers()[at].as_ref());
        let (plain, caller, own) = (id(0), id(1), id(2));
        let media = |vision, references| Demands {
            vision,
            references,
            ..Demands::default()
        };

        for (vision, references, expected) in [
            (true, References::None, BTreeSet::from([plain, caller])),
            (false, References::Any, BTreeSet::from([caller, own])),
            (false, References::Caller, BTreeSet::from([caller])),
            (false, References::Worker, BTreeSet::from([own])),
        ] {
            let mut asks = Asks::of(request(), Some(&mixed));
            asks.request.constraints.demands = media(vision, references);
            let mut firsts = BTreeSet::new();
            for seed in 0..48_u64 {
                asks.request.facts.seed = seed;
                let outcome = decided(&engine, &asks);
                let pick = outcome.decision.pick().expect("a matching worker");
                firsts.insert(pick.targets.first().expect("a first target").id);
            }
            assert_eq!(
                firsts, expected,
                "vision {vision}, references {references:?}"
            );
        }

        let other_runtime = Demands {
            runtime: 9,
            ..media(true, References::Caller)
        };
        let blind = members_with(&[(false, true, true), (false, false, false)]);
        for (members, demands, stage) in [
            (&mixed, media(true, References::Worker), "pipeline"),
            (&mixed, other_runtime, "wires"),
            (&blind, media(true, References::Any), "vision"),
        ] {
            assert_eq!(
                places_named(&engine, members, demands),
                Err((ShedReason::NoEndpoints, stage)),
                "{demands:?}"
            );
        }
    }

    /// One model has a worker that takes images.
    /// A vision request for a model with no list is shed under `none`.
    #[test]
    fn a_demand_for_a_model_no_member_serves_is_shed_under_no_stage() {
        let engine = preset("random");
        let members = PublishedList::with_facts(pool(1), |_| FixedFacts {
            vision: true,
            ..FixedFacts::default()
        });
        let mut asked = request();
        asked.constraints.demands.vision = true;
        let listed = decided(&engine, &Asks::of(asked, Some(&members)));
        assert!(
            listed.decision.pick().is_some(),
            "the listed model takes images"
        );

        asked.facts.model = "a-model-nobody-serves";
        let absent = decided(&engine, &Asks::of(asked, None)).decision;
        let Decision::Shed(shed) = absent else {
            panic!("expected a shed for the absent model, got {absent:?}");
        };
        assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::NoEndpoints, "none")
        );
    }

    /// The first pick records the block for the blind worker.
    /// An unrestricted repeat picks that worker again.
    /// A vision request for the same block picks the sighted worker.
    #[test]
    fn a_vision_demand_skips_the_blind_holder_of_the_first_block() {
        let engine = preset("cache_aware");
        let list = pool(2);
        let (blind, sighted) = (&list[0], &list[1]);
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            vision: Arc::ptr_eq(worker, sighted),
            ..FixedFacts::default()
        });

        let mut prompt = request();
        prompt.facts.block_hashes = &[17];
        let mut to_blind = prompt;
        to_blind.constraints.places = Some(&[0]);
        let mut vision = prompt;
        vision.constraints.demands.vision = true;

        for (asked, expected, step) in [
            (to_blind, blind, "first placement"),
            (prompt, blind, "unrestricted repeat"),
            (vision, sighted, "vision request"),
        ] {
            let outcome = decided(&engine, &Asks::of(asked, Some(&members)));
            let pick = outcome.decision.pick().expect("a pick");
            let first = pick.targets.first().expect("a target");
            assert_eq!(first.id, worker_id(expected.as_ref()), "{step}");
        }
    }

    /// The picker declares its needs while the document is built.
    /// Two single requests get picks without another call to `needs` or a fault.
    #[test]
    fn a_stage_is_asked_what_it_needs_at_the_build_alone() {
        use std::sync::atomic::AtomicBool;

        use crate::{
            document::spec::Needs,
            engine::{framework::DecisionScope, view::PoolFacts},
        };

        #[derive(Debug)]
        struct BuildNeeds(Arc<AtomicBool>);

        impl Lifecycle for BuildNeeds {}

        impl DecisionScope for BuildNeeds {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

            fn close(&self) {}
        }

        impl Picker for BuildNeeds {
            fn name(&self) -> &'static str {
                "build-needs"
            }

            fn needs(&self) -> Needs {
                assert!(!self.0.load(Ordering::Relaxed), "needs ran after the build");
                Needs {
                    active_sessions: true,
                    ..Needs::default()
                }
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }

            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                Random.pick(facts, candidates, kept, scores, band, out);
            }
        }

        let armed = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&armed);
        let engine = picked_by("build-needs", move || {
            Box::new(BuildNeeds(Arc::clone(&flag)))
        });
        armed.store(true, Ordering::Relaxed);

        let members = PublishedList::new(pool(3));
        for id in [b"r-1", b"r-2"] {
            let asks = Asks::of(Request { id, ..request() }, Some(&members));
            assert!(decided(&engine, &asks).decision.pick().is_some());
        }
        assert_eq!(engine.faults(), 0);
        assert!(
            engine.spec().needs().active_sessions,
            "the build recorded the picker's declared need"
        );
    }
}
