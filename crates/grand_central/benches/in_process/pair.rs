//! A pair's cells: two lists of worker objects under a pair document.
//!
//! A cell named `pair_...` is run here ([`runs`]), by the bench's own
//! measuring. Its host hands the engine two lists of `--workers` workers
//! each, the prefills and the decodes, every pair open, and a caller asks
//! the one entry a host asks for a request ([`InProcess::decide_at`]). A
//! pair's decision is leased: its guard is told both dispatches and each
//! side's end. A request that states the later decode has it decided again
//! behind the prefill's dispatch (`pair_round_robin_later`). The cell
//! `pair_round_robin_three_sided` places the same pair under a document of
//! three sides, whose encode side places nothing of a pair.
//!
//! `pair_sessions_later` states the later decode under a sessions rule,
//! each turn for a session of its own ([`sessions`], [`session_later`]).
//! Its decode is asked for once the session's key has ended: a key ends at
//! its rule's first sweep past its idle time. So the call claims its decode
//! anew and dates that claim by the pair's lease. The result's
//! `call_claims=` counts the calls that claimed.
//!
//! A session met a second time claims nothing, and the list's first
//! requests take the seeds of the first caller's first turns: eight a
//! worker, and 4,096 at the least. So count the cell by two values of
//! `--decisions` above that count. Where each counted call claims,
//! `call_claims=` rises by their difference. The rule sweeps nothing
//! within a run: the table keeps two keys a turn.
//!
//! `--entry place` asks the one call a host makes for a request instead
//! ([`InProcess::place`]): the model, the two lists and a prompt of the
//! plain request's size, with both dispatches and each side's end said of
//! the placement. Such a request states no later decode, and its run
//! takes every `--refused` but `limit`. `--entry place_ruled` and
//! `--entry place_held` state a route's rule with it (the bench's `ruled`
//! file). `--entry place_named` asks the one call with a pair that names
//! its worker ([`NAMED`]). `--entry place_alone` states a rule that takes
//! every prefill behind the first and no decode, over a cell whose every
//! such prefill is over its thresholds under a binding that sheds a list
//! wholly over them: each request is shed, and a turn counts the shed.
//!
//! Every worker has a load report, and a prefill states its rate as its
//! measured input rate too: the one drain a prefill's record reads. Under
//! a document that declares the signal of the `signal_cost` cells, each
//! decode's report states a value of it (`pair_signal_cost`).
//! `--overloaded 1` puts a pair's decodes alone over their thresholds: the
//! rule for such a list steers a pair's decode side. A pair whose prefills
//! are all over theirs goes to the pair's default: the cell `pair_default`
//! puts them there ([`DEFAULT`]).
//! `--refused` states what refuses every pair ([`Refused`]): the engine
//! then names the refusal of each request, and a turn counts where it was
//! refused the way the run states. A cell that leaves that path counts no
//! turn. `--refused overload` puts every decode over its thresholds under
//! a binding that sheds a list wholly over them: each request is shed as
//! saturated, by the engine's entry and by the one call alike. `--refused
//! outage` takes every decode out of service under such a binding, and
//! `--refused partners` under the cell's own: each request is shed with no
//! endpoint, by either entry too. The result's `conversions=` counts the
//! two lists together.

use std::{
    cell::OnceCell,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use smg_grand_central::{
    document::{AppliedSpec, Assignment, PolicySpec},
    engine::{
        prefills_full, Decided, Decision, DecodeVerdict, Identity, InProcess, LiftedVeto,
        PairAdmission, PairLists, PartnerClasses, Phase, PlaceRequest, Plain, Prompt,
        PublishedList, Request, RequestMaker, ShedReason, Unplaced, WorkerLists,
    },
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Side, TransportClass},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::TestFeed,
    test_support::no_health_check,
    worker::{WorkerSpec, WorkerStatus, WorkerType},
};
use tokio::runtime::Builder;

use super::{
    encode, measure, place, plain, preset, published, reports, ruled, signals, Cell, Host, Run,
    Shadowed, Stated, MODEL,
};

/// The word of `--entry` that asks the one call with a pair's request
/// that names its worker: the number 0 under the header a hash ring's
/// document reads for it. What the hold costs a request is this entry's
/// count over the same cell's under a build that does not read the header
/// for a pair.
pub(super) const NAMED: &str = "place_named";

/// The header such a request carries: the first worker a side.
const NUMBER: [(&str, &str); 1] = [("x-smg-target-worker", "0")];

/// The cell whose pairs the pair's default makes. Every prefill is over its
/// thresholds beside open decodes, so the prefill side's run sheds. A turn
/// counts where the pair's trace names the lifted overload veto.
pub(super) const DEFAULT: &str = "pair_default";

/// Whether this run's pairs name their worker: what the result line's word
/// says.
static NAMES: AtomicBool = AtomicBool::new(false);

/// The result line's word for a run whose pairs name their worker.
pub(super) fn word() -> Option<&'static str> {
    NAMES
        .load(Ordering::Relaxed)
        .then_some(" entry=place_named")
}

/// The pair document of the preset `name` on both sides.
pub(super) fn of(name: &str) -> PolicySpec {
    PolicySpec::pair_presets(name, name).expect("a preset's name")
}

/// A pair's request of the plain request's size, stated for the prefill
/// phase.
pub(super) fn request(seed: u64) -> Request<'static> {
    let mut request = plain(seed);
    request.facts.phase = Phase::Prefill;
    request
}

/// The same pair's request, stating that its decode is decided again behind
/// the prefill's dispatch.
pub(super) fn later(seed: u64) -> Request<'static> {
    let mut request = request(seed);
    request.constraints.decode_later = true;
    request
}

/// The decode calls of this run that claimed their decode so far.
static CALL_CLAIMS: AtomicU64 = AtomicU64::new(0);

/// The result line's word of `pair_sessions_later`: how many of its decode
/// calls claimed their decode.
pub(super) fn call_claims() -> String {
    format!(" call_claims={}", CALL_CLAIMS.load(Ordering::Relaxed))
}

/// The round-robin pair document under the stock sessions rule, a new
/// session placed by each side's own stages. The rule's sweep waits as
/// long as its idle time, so no key leaves the table within a run.
pub(super) fn sessions() -> PolicySpec {
    let stock = preset("manual");
    let mut document = of("round_robin");
    let rule = stock.bindings.into_iter().next();
    let rule = rule.and_then(|binding| binding.sessions);
    let mut rule = rule.expect("the stock rule");
    rule.assignment = Assignment::Delegate;
    rule.sweep_s = rule.ttl_s;
    document.view = stock.view;
    document.bindings[0].sessions = Some(rule);
    document
}

/// The later-decode request of a session of its own: a turn's seed is its
/// session's key.
pub(super) fn session_later(seed: u64) -> Request<'static> {
    let mut request = later(seed);
    request.facts.session_key = Identity::hashed(seed.max(1));
    request
}

/// A ready worker of `worker_type` at `url`, with no health checker behind it.
pub(super) fn typed(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
    let spec = WorkerSpec {
        worker_type,
        ..WorkerSpec::new(url)
    };
    let worker = <BasicWorkerBuilder>::from_spec(spec)
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// Puts every worker of `side` over its thresholds and gives each requests
/// in flight: its place plus one, the middle worker none. The rule for
/// such a list then has loads to steer by.
fn over(side: &[Arc<dyn Worker>]) {
    for (at, worker) in side.iter().enumerate() {
        worker.set_overloaded(true);
        if at != side.len() / 2 {
            for _ in 0..=at {
                worker.increment_load();
            }
        }
    }
}

/// A pair's host with no limit of its own: every prefill has capacity and
/// every decode a room.
#[derive(Debug)]
pub(super) struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A pair's host every prefill of which holds its limit: none has capacity.
#[derive(Debug)]
struct Full;

impl PairAdmission for Full {
    fn has_capacity(&self, _: usize) -> bool {
        false
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// What a run states refuses every pair of its cell (`--refused`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Refused {
    /// Nothing: every pair is open.
    No,
    /// The host's admission limit: every prefill holds it, beside open
    /// partners.
    Limit,
    /// The decode side: no decode is ready, and no prefill has a partner
    /// to go to.
    Partners,
    /// The overload veto, under a binding that sheds a list wholly over its
    /// thresholds: every decode is over them, and each request is shed as
    /// saturated.
    Overload,
    /// The decode side under such a binding: no decode is ready, and each
    /// request is shed with no endpoint.
    Outage,
}

impl Refused {
    /// What `--refused` names.
    pub(super) fn named(name: &str) -> Self {
        match name {
            "limit" => Self::Limit,
            "partners" => Self::Partners,
            "overload" => Self::Overload,
            "outage" => Self::Outage,
            other => panic!("nothing refuses a pair by {other}"),
        }
    }
}

/// A pair's decodes as its host keeps them beside its prefills (the host's
/// own list), with every worker's pair class, the host's admission and
/// what the run states refuses every pair.
struct Decodes {
    list: PublishedList,
    classes: PartnerClasses,
    admission: &'static (dyn PairAdmission + Sync),
    refused: Refused,
    /// Under a sessions rule, how far behind a pair's decision its later
    /// decode is asked for, sent and ended. It is a second more than the
    /// rule's idle time and its sweep interval together. The pair's key has
    /// ended by then, at the rule's first sweep past its idle time.
    idle_out: Option<Duration>,
    /// Whether the run's rule takes the prefills behind the first alone
    /// ([`ruled::ALONE`]): each request is then shed behind the rule.
    alone: bool,
}

/// One turn's pair request as its host states it: the request of
/// [`Stated`], over the host's two lists.
struct StatedPair<'a> {
    stated: Stated<'a>,
    decodes: &'a Decodes,
}

impl RequestMaker for StatedPair<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.stated.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        self.stated.plain()
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.decodes.lists(self.stated.host)
    }
}

/// The engine as a pair's guard holds it. The type makes that guard an
/// instance of [`DecisionGuard`] apart from a single request's, whose calls
/// then have the one caller they have with no pair in the binary: a single
/// request's cell counts the same with a pair's cells beside it.
struct PairsEngine<'a>(&'a InProcess);

impl Deref for PairsEngine<'_> {
    type Target = InProcess;

    fn deref(&self) -> &InProcess {
        self.0
    }
}

impl Decodes {
    /// The pair's two lists as `host` hands them for one request: its own
    /// list as the prefills, and these decodes.
    fn lists<'a>(&'a self, host: &'a Host) -> WorkerLists<'a> {
        WorkerLists::Pair(PairLists {
            prefill: host.list.lent(),
            decode: self.list.lent(),
            classes: &self.classes,
            admission: self.admission,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        })
    }

    /// One pair request of `host` by the one call: the model, the two
    /// lists and a prompt of the plain request's size, with both sides'
    /// dispatch and end said of the placement. With `rule` the request
    /// states the run's rule of its route ([`ruled::admits`]), and with
    /// `named` the worker it names ([`NAMED`]). The call answers workers
    /// and no place of a list, so the place this says is the first.
    fn called(&self, host: &Host, rule: bool, named: bool) -> Option<u32> {
        let mut request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&place::PROMPT),
                ..Prompt::NONE
            },
            admits: ruled::admits(rule),
            ..PlaceRequest::of(MODEL, self.lists(host))
        };
        if named {
            request.headers = &NUMBER;
        }
        let mut answer = host.engine.place(&request);
        // A rule that takes prefills over their thresholds alone: the turn
        // is a shed, as saturated behind the one decision more that is held
        // to them, and with no endpoint where no such decision is made.
        if self.alone {
            return matches!(answer, Err(Unplaced::Shed { .. })).then_some(0);
        }
        // A run whose every decode is over its thresholds counts the turns
        // shed as saturated, and one whose every decode is out of service
        // the turns shed with no endpoint.
        let counted = match self.refused {
            Refused::Overload => Some(ShedReason::Saturated),
            Refused::Partners | Refused::Outage => Some(ShedReason::NoEndpoints),
            Refused::Limit | Refused::No => None,
        };
        if let Some(counted) = counted {
            let reason = match &answer {
                Err(Unplaced::Shed { reason, .. }) => Some(*reason),
                _ => None,
            };
            return (reason == Some(counted)).then_some(0);
        }
        let placement = answer.as_mut().ok()?;
        if placement.owes_events() {
            encode::said(placement);
        }
        Some(0)
    }

    /// One pair request of `host` by the engine's one entry: the prefill's
    /// place in its list, with both dispatches and each side's end told
    /// where the decision is leased. Under `DEFAULTED` a turn counts where
    /// the pair's default made the pair ([`DEFAULT`]). The test is that
    /// cell's alone: every other cell's turn is compiled without it.
    fn place<const DEFAULTED: bool>(&self, host: &Host, seed: u64) -> Option<u32> {
        let stated = StatedPair {
            stated: Stated {
                host,
                seed,
                prompt: None,
            },
            decodes: self,
        };
        let dated = OnceCell::new();
        let clock = || *dated.get_or_init(Instant::now);
        let Decided::Whole(decided, later) = host.engine.decide_at(&stated, clock).ok()? else {
            return None;
        };
        // A run that refuses every pair counts the turns refused the way it
        // states: by full prefills under the limit, by a shed as saturated
        // under the overload veto, by a shed with no endpoint under an
        // outage, by any other answer where the decode side refuses.
        if self.refused != Refused::No {
            let decision = &decided.decision;
            let full = prefills_full(decision);
            let reason = match decision {
                Decision::Shed(shed) => Some(shed.reason),
                _ => None,
            };
            let stated = match self.refused {
                Refused::Limit => full,
                Refused::Overload => reason == Some(ShedReason::Saturated) && !full,
                Refused::Outage => reason == Some(ShedReason::NoEndpoints) && !full,
                Refused::Partners | Refused::No => !full,
            };
            return (decision.pick().is_none() && stated).then_some(0);
        }
        let pick = decided.decision.pick()?;
        // The default's pair says the overload veto it lifted.
        if DEFAULTED && pick.trace.lifted != LiftedVeto::Overload {
            return None;
        }
        let target = *pick.targets.first()?;
        let decode = target.pair?.decode_id;
        if decided.lease.is_some() {
            let now = *dated.get()?;
            let engine = PairsEngine(host.engine);
            let mut held = DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now);
            // The decode goes out with its prefill, and each side's end is
            // told under its own lease.
            held.dispatched(now);
            // A request that stated the later decode asks it again here. A
            // call the engine sheds answers the first pick and leaves the
            // decode chosen: no decode was decided again, and the turn
            // places nothing. Under a sessions rule the call comes once the
            // pair's key has ended, and the decode is sent and ended at the
            // call's instant.
            let (decode, now) = if later {
                let call = self.idle_out.map_or(now, |idle| now + idle);
                let again = held.decide_decode_under(&stated, call).ok()?;
                let again = again.pick()?.targets.first()?.pair?;
                if held.decode_chosen() {
                    return None;
                }
                if again.decode_verdict == DecodeVerdict::CLAIMED {
                    CALL_CLAIMS.fetch_add(1, Ordering::Relaxed);
                }
                (again.decode_id, call)
            } else {
                (decode, now)
            };
            held.dispatched_decode(now);
            let end = |side, served| {
                let mut end = Outcome::reported(200, TransportClass::Ok).with_served(side, served);
                end.tokens_in = 1_024;
                end
            };
            held.completed(&end(Side::Prefill, target.id), now);
            held.completed(&end(Side::Decode, decode), now);
        }
        Some(target.position)
    }
}

/// Runs the cell of `run` where it is a pair's, one named `pair_...`, and
/// says whether it did. The cell's request reads nothing the cell makes
/// behind its engine, and none is made ahead of that.
pub(super) fn runs(run: &Run) -> bool {
    let Cell {
        behind,
        document,
        name,
        request,
        ..
    } = run.cell;
    if !name.starts_with("pair_") {
        return false;
    }
    // A shadow profile decides beside a single request's decision.
    assert!(run.shadow == Shadowed::No, "a pair's cell runs no shadow");
    assert!(run.other_models == 0, "a pair's cell states no other model");
    let side = |side: &str, worker_type| -> Arc<[Arc<dyn Worker>]> {
        let url = |at: usize| format!("http://{side}-{at:04}:8000");
        (0..run.workers)
            .map(|at| typed(url(at), worker_type))
            .collect()
    };
    let prefills = side("prefill", WorkerType::Prefill);
    let decodes = side("decode", WorkerType::Decode);
    // The rule of `place_alone` takes every prefill behind the first: the
    // cell puts those over their thresholds beside the open first one.
    let alone = run.entry == ruled::ALONE;
    if alone {
        let own = run.refused == Refused::No && !run.overloaded && run.workers > 1;
        assert!(
            own,
            "place_alone states its own lists, of two workers or more"
        );
        for worker in prefills.iter().skip(1) {
            worker.set_overloaded(true);
        }
    }
    if run.overloaded {
        // A pair's prefill side is put to no rule: the cell puts its decodes
        // over their thresholds and leaves its prefills open.
        over(&decodes);
    }
    if name == DEFAULT {
        // The prefill side's run sheds over such prefills, and the pair's
        // default makes the pair of the least load.
        over(&prefills);
    }
    if matches!(run.refused, Refused::Partners | Refused::Outage) {
        for worker in decodes.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let mut document = document();
    if run.refused == Refused::Overload {
        for worker in decodes.iter() {
            worker.set_overloaded(true);
        }
    }
    if alone || matches!(run.refused, Refused::Overload | Refused::Outage) {
        // Each binding sheds a list wholly over its thresholds.
        for binding in &mut document.bindings {
            let mut overload = binding.overload.take().unwrap_or_default();
            overload.shed_when_all_vetoed = true;
            binding.overload = Some(overload);
        }
    }
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(spec)));
    behind(engine);
    // Every prefill pairs with every decode.
    let partners: Vec<u32> = (0..decodes.len() as u32).collect();
    let every = (0..prefills.len()).map(|_| partners.as_slice());
    // The sessions rule of the pair's binding, where its document states one.
    let rule = document.bindings.first();
    let rule = rule.and_then(|binding| binding.sessions.as_ref());
    let decode_side = Decodes {
        classes: PartnerClasses::number(decodes.len(), every),
        list: published(&decodes),
        admission: if run.refused == Refused::Limit {
            &Full
        } else {
            &Open
        },
        refused: run.refused,
        idle_out: rule.map(|rule| {
            let ends = u64::from(rule.ttl_s) + u64::from(rule.sweep_s);
            Duration::from_secs(ends + 1)
        }),
        alone,
    };
    let mut receiver = None;
    let feed = TestFeed::attached_to(|loads| receiver = loads);
    let mut loads = reports(&prefills);
    for report in loads.values_mut() {
        let rank = &mut report.loads[0];
        if rank.gen_throughput > 0.0 {
            rank.prefill_throughput = Some(rank.gen_throughput);
        }
    }
    loads.extend(reports(&decodes));
    let loads = signals::stated(&document, &decodes, loads);
    feed.publish(&loads);
    // What a host does once, as for one list: it hands its load monitor's
    // channel and spawns the engine's task.
    engine.set_loads(receiver.expect("the feed's channel"));
    let upkeep = Builder::new_current_thread().enable_time().build();
    let upkeep = upkeep.expect("a runtime");
    upkeep.spawn(engine.run());
    let host = Host {
        engine,
        list: published(&prefills),
        request,
        plain: request(0).plain(),
    };
    // A pair's reports are converted a list: its two lists' counts together.
    let converted = || host.conversions() + engine.conversions(decode_side.list.lent());
    // The one call a host makes for a request, where the run asks it.
    if place::asked_with_lists(run, true) {
        let answered = run.refused != Refused::Limit;
        assert!(
            answered,
            "the one call's pair is open, or shed by its decodes"
        );
        // The one call answers workers and no trace: nothing there says
        // which pairs the default made.
        assert!(
            name != DEFAULT,
            "the default's cell asks the engine's entry"
        );
        let rule = ruled::stated(run);
        let named = run.entry == NAMED;
        NAMES.store(named, Ordering::Relaxed);
        let publish = || feed.publish(&loads);
        measure(run, &host, upkeep, publish, converted, |_| {
            decode_side.called(&host, rule, named)
        });
        return true;
    }
    let publish = || feed.publish(&loads);
    if name == DEFAULT {
        let decide = |seed| decode_side.place::<true>(&host, seed);
        measure(run, &host, upkeep, publish, converted, decide);
        return true;
    }
    let decide = |seed| decode_side.place::<false>(&host, seed);
    measure(run, &host, upkeep, publish, converted, decide);
    true
}
