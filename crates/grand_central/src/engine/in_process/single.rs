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
//! request: the engine asks it before the request is made, so no request
//! is made for such a decision. Under a document that keeps a sample of its
//! decisions the request's maker is asked its id behind the pick
//! ([`RequestMaker::request_id`]).
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

use super::{keys, overlap, place::Made, Loaded, PoolMembers};
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

    /// The calling thread's own run of lease ids ([`lease_id`]): the number
    /// of the document its engine decides under, the run's next id and how
    /// many ids the run has left.
    static IDS: Cell<(u64, u64, u64)> = const { Cell::new((0, 0, 0)) };
}

/// The ids a thread takes of its engine's counter at once.
pub(super) const ID_RUN: u64 = 64;

/// The id of the next lease the calling thread issues a single request: no
/// other lease of the engine has it. A thread takes [`ID_RUN`] ids of the
/// engine's counter at once and issues them one by one, in rising order: the
/// word every deciding thread shares is written once a run, where a write a
/// lease made each caller wait for the others' line. The ids a thread leaves
/// of its run name no lease. A pair's lease takes its own step of the
/// counter, so its id may read above a single's the thread issues after it.
#[inline]
fn lease_id(own: Loaded<'_, '_>) -> LeaseId {
    let engine = own.document.applied;
    let (of, next, left) = IDS.get();
    if left != 0 && of == engine {
        IDS.set((of, next + 1, left - 1));
        return LeaseId(next);
    }
    let first = own.leases.0.fetch_add(ID_RUN, Ordering::Relaxed) + 1;
    IDS.set((engine, first + 1, ID_RUN - 1));
    LeaseId(first)
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
/// answer. Under one whose session table alone takes a word it is the whole
/// answer where its worker reads ready a second time ([`still_ready`]): the
/// table has no word for a pick made under a binding that states no rule,
/// as a whole pass's lease has it ([`issued`]). Any other pick is noted
/// ([`noted`]).
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

/// Whether the worker at the place `pick` names reads ready now
/// ([`super::place::ready`]): the second read of it, behind the one that
/// found it eligible, as a noted pick's target has at its walk. Under a
/// document whose session table alone takes a word on a lease a place is
/// the whole answer where this holds. A pick whose worker turned not
/// ready between the two reads is noted ([`noted`]) and takes its lease,
/// and the walk of its target reads that worker once more.
///
/// Out of line: no decision of a document that takes no word runs it.
#[cold]
#[inline(never)]
pub(super) fn still_ready(pick: AlonePick, members: &PoolMembers) -> bool {
    members
        .workers()
        .get(pick.place as usize)
        .is_some_and(|worker| super::place::ready(worker.as_ref()))
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
        id: lease_id(own),
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
/// sent prefixes ([`RequestMaker::token_key`]), or in its string tree
/// ([`RequestMaker::text_key`]), where the host lends the request's place
/// one ([`overlap::decide`]). It is asked of the maker's own type and no
/// argument carries the entry: a maker that names none costs a decision
/// nothing. The decision is compiled once for each maker type, as the
/// engine's entry that calls it is, and once more for the trait object
/// through which a copy that carries signal cells asks
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
///
/// `late` is the request as the engine's own call made it, where that call
/// asks ([`Loaded::decide_into`]): the lookup asks it for the prompt's
/// block hashes on a road that reads them ([`Made::hashed_whole`]).
///
/// A request its own call made (`late`) is one whose decision is kept
/// behind a placement, which hands out no trace. Where no stage of the
/// document takes the issue too and the document keeps no sample, nothing
/// reads the best rows of its trace, and the run reads none
/// ([`records::Live::unread`]).
#[expect(
    clippy::too_many_arguments,
    reason = "a made request, its list, its maker and the request its call made, the loads, the instant and the whole decision"
)]
pub(super) fn decide(
    own: Loaded<'_, '_>,
    request: &mut Request<'_>,
    members: &PoolMembers,
    maker: &(impl RequestMaker + ?Sized),
    late: Option<&Made<'_, '_>>,
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
    // pick alone turns ([`InProcess::pick`]): a draw of this decision
    // takes the list's next turn. The maker is asked under a document
    // that reads the count and under no other.
    let mut name = [0; Session::TEXT_LEN];
    let key = if spec.engine().needs().session_in_flight {
        maker.session_counted_under().name(facts, &mut name)
    } else {
        None
    };
    // The engine's own call made the request (`late`): its decision is kept
    // behind a placement, where a kept sample and a stage that takes the
    // issue are the two readers of the best rows of its trace.
    let document = own.document;
    let read = document.samples || document.hears.contains(EventKind::Issued);
    let live = records::Live::at(spec.engine(), &own.clock, now, loads, key)
        .turning(&members.rotation.0)
        .unread(late.is_some() && !read);
    // A retry leaves every worker of the engines it tried: over a list
    // where an engine holds several, its facts name them all.
    let decision = if facts.tried.is_empty() || !records.has_ranks() {
        by_facts::<false>(own, members, records, &live, facts, cx, maker, late, whole)
    } else {
        retried(own, members, records, &live, facts, cx, maker, late, whole)
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
/// request's holders asks it for the request's entry in a lent tree,
/// and which target of its pick the request's caller sends to.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "a decision's inputs, the request's maker, the request its call made and the whole decision"
)]
fn by_facts<const RETRY: bool>(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    live: &records::Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
    maker: &(impl RequestMaker + ?Sized),
    late: Option<&Made<'_, '_>>,
    mut whole: impl WholeDecision,
) -> Decision {
    let spec = &own.document.spec;
    let (engine, workers) = (spec.engine(), members.workers());
    // A keyed request is decided by the rows its key names, where they say
    // what the whole pass would: its key's pin, or its ring's walk.
    // Read ahead of the keyed entry: a leave told behind a pin's choice is
    // seen where the pin is handed on.
    let leaves = own.affinity.leaves();
    match keys::decide(engine, records, workers, live, facts, cx) {
        // A pin its rule keeps from the profile: no holder is named for it
        // and nothing is kept of where its prompt went.
        Some(keyed) if !lease_events::stages_hear(&keyed, Side::Single) => keyed,
        // A pin that is the profile's own goes through the same lookup with
        // its answer made.
        Some(keyed) => handed_on(own, members, records, facts, maker, late, keyed, leaves),
        // A profile whose stages read who holds the prompt is named its
        // rows from the gateway's own data, at the decision's instant the
        // keyed entry wrote, and its pick is recorded behind it; any other
        // passes straight to the whole pass.
        None => overlap::decide_hashed(
            &own.affinity,
            members.affinity(),
            spec,
            workers,
            records.ids(),
            facts,
            cx.places,
            |page| maker.token_key(page),
            || maker.text_key(),
            |target| maker.sends_to(target),
            || late?.hashed_whole(spec),
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
/// hands the prompt to the record of sent prefixes, or to a tree its host
/// lends by the entry and the target its maker names. It may read its list's
/// table again, as any decision that finds the table expired, and a tree's
/// match marks the path it walked as used. Where no binding of the phase
/// reads affinity, the lookup is one test.
///
/// Out of line: its code stays off the path of a request with no pin, and
/// of a pin its rule keeps from the profile.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the list and its records, the request's facts, its maker and the request its call made, the answer and the leaves read ahead of it"
)]
fn handed_on(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    facts: &mut RequestFacts<'_>,
    maker: &(impl RequestMaker + ?Sized),
    late: Option<&Made<'_, '_>>,
    keyed: Decision,
    leaves: u64,
) -> Decision {
    let spec = &own.document.spec;
    overlap::decide_behind(
        &own.affinity,
        members.affinity(),
        spec,
        members.workers(),
        records.ids(),
        facts,
        |page| maker.token_key(page),
        || maker.text_key(),
        |target| maker.sends_to(target),
        || late?.hashed_whole(spec),
        leaves,
        keyed,
    )
}

/// A retry's decision over a list where an engine holds several workers.
/// The request is decided ([`by_facts`]) by a copy of its facts whose tried
/// workers are its own and every other worker of their engines, which the
/// list's records name in one operation ([`PoolRecords::tried_engines`]).
/// The key's rows, the whole pass and a binding's shadow profile so read
/// one list, and the request's own list stays as its host made it. The
/// decision's instant is written back into `facts`: the request's lease
/// is dated by it. The request's maker goes with the copy, so a retry is
/// decided inside a lent tree as a first attempt is.
///
/// The workers stand in the calling thread's own list ([`TRIED`]), so a
/// retry allocates nothing once the list has grown. Out of line and cold:
/// a first attempt's path, and a retry's over a list with no such engine,
/// hold one test of this and nothing else. The records' own answer is read
/// here all the same: over a list with no such engine they hand no list
/// back, and the copy names the request's own.
#[cold]
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "a decision's inputs, the request's maker, the request its call made and the whole decision"
)]
fn retried(
    own: Loaded<'_, '_>,
    members: &PoolMembers,
    records: &PoolRecords,
    live: &records::Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
    maker: &(impl RequestMaker + ?Sized),
    late: Option<&Made<'_, '_>>,
    whole: impl WholeDecision,
) -> Decision {
    let tried = PoolRecords::tried_engines([records], facts.tried, || TRIED.take());
    let mut widened = RequestFacts {
        tried: tried.as_deref().unwrap_or(facts.tried),
        ..*facts
    };
    let decision = by_facts::<true>(
        own,
        members,
        records,
        live,
        &mut widened,
        cx,
        maker,
        late,
        whole,
    );
    facts.now_micros = widened.now_micros;
    if let Some(tried) = tried {
        TRIED.set(tried);
    }
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
        side: None,
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
/// ([`super::InProcess::word_caught`]). A stage that panics at the issue
/// is held and counted, the stages behind it do not hear it, and the pick
/// and its lease are answered all the same: the guard tells the lease's
/// dispatch and its end as any lease's. From the answer the lease is its
/// guard's: nothing of it is filed.
///
/// A pick the binding's sessions rule keeps from the profile is issued
/// too, and no stage hears its issue ([`lease_events::stages_hear`]).
///
/// Under a document that takes no word a pick has no lease: no id is
/// taken, and its guard owes the engine nothing. Nor has a pick the
/// session table has no word for ([`SessionVerdict::None`]: a request
/// with no key or under a binding that states no rule, and any other pick
/// the table is told nothing of) under a document whose session table
/// alone takes one: the table passes such a dispatch by. A stage or a
/// count of sends that takes a word keeps every pick's lease.
pub(super) fn issued(
    own: Loaded<'_, '_>,
    decision: Decision,
    request: &Request<'_>,
    now: Instant,
) -> DecideOutcome {
    let facts = &request.facts;
    let document = own.document;
    let noted = decision.pick().filter(|pick| {
        let hears = match pick.session_verdict {
            SessionVerdict::None => document.hears_beside_table,
            _ => document.hears,
        };
        hears.intersects(EventSet::LEASE)
    });
    let lease = noted.and_then(|pick| {
        let first = pick.targets.first()?;
        Some(Lease {
            id: lease_id(own),
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
        if lease_events::stages_hear(&decision, Side::Single) {
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
#[path = "../../../tests/unit/engine/in_process/single/tests.rs"]
pub(crate) mod tests;
