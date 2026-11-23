//! A pair's cells: two lists of worker objects under a pair document.
//!
//! A cell named `pair_...` is run here ([`runs`]), by the bench's own
//! measuring. Its host hands the engine two lists of `--workers` workers
//! each, the prefills and the decodes, every pair open, and a caller asks
//! the one entry a host asks for a request ([`PolicyEngine::decide_at`]). A
//! pair's decision is leased: its guard is told both dispatches and each
//! side's end. A request that states the later decode has it decided again
//! behind the prefill's dispatch (`pair_round_robin_later`).
//!
//! Every worker has a load report, and a prefill states its rate as its
//! measured input rate too: the one drain a prefill's record reads. Under
//! a document that declares the signal of the `signal_cost` cells, each
//! decode's report states a value of it (`pair_signal_cost`).
//! `--overloaded 1` puts a pair's decodes alone over their thresholds: the
//! rule for such a list steers a pair's decode side. A pair whose prefills
//! are all over theirs goes to the pair's default, which no cell runs.
//! `--refused` states what refuses every pair ([`Refused`]): the engine
//! then names the refusal of each request, and a turn counts where it was
//! refused the way the run states. A cell that leaves that path counts no
//! turn. The result's `conversions=` counts the two lists together.

use std::{cell::OnceCell, ops::Deref, sync::Arc, time::Instant};

use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        prefills_full, Decided, PairAdmission, PairLists, PartnerClasses, Phase, Plain,
        PolicyEngine, Request, RequestMaker, WorkerLists,
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
    measure, plain, reports, signals, Cell, Host, Published, Run, Shadowed, Stated, MODEL,
};

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
}

impl Refused {
    /// What `--refused` names.
    pub(super) fn named(name: &str) -> Self {
        match name {
            "limit" => Self::Limit,
            "partners" => Self::Partners,
            other => panic!("nothing refuses a pair by {other}"),
        }
    }
}

/// A pair's decodes as its host keeps them beside its prefills (the host's
/// own list), with every worker's pair class, the host's admission and
/// what the run states refuses every pair.
struct Decodes {
    list: Published,
    classes: PartnerClasses,
    admission: &'static (dyn PairAdmission + Sync),
    refused: Refused,
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
        WorkerLists::Pair(PairLists {
            prefill: self.stated.host.list.lent(),
            decode: self.decodes.list.lent(),
            classes: &self.decodes.classes,
            admission: self.decodes.admission,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        })
    }
}

/// The engine as a pair's guard holds it. The type makes that guard an
/// instance of [`DecisionGuard`] apart from a single request's, whose calls
/// then have the one caller they have with no pair in the binary: a single
/// request's cell counts the same with a pair's cells beside it.
struct PairsEngine<'a>(&'a PolicyEngine);

impl Deref for PairsEngine<'_> {
    type Target = PolicyEngine;

    fn deref(&self) -> &PolicyEngine {
        self.0
    }
}

impl Decodes {
    /// One pair request of `host` by the engine's one entry: the prefill's
    /// place in its list, with both dispatches and each side's end told
    /// where the decision is leased.
    fn place(&self, host: &Host, seed: u64) -> Option<u32> {
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
        // states: by full prefills under the limit, by any other shed where
        // the decode side refuses.
        if self.refused != Refused::No {
            let full = prefills_full(&decided.decision);
            let stated = full == (self.refused == Refused::Limit);
            return (decided.decision.pick().is_none() && stated).then_some(0);
        }
        let target = *decided.decision.pick()?.targets.first()?;
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
            // places nothing.
            let decode = if later {
                let again = held.decide_decode_under(&stated, now).ok()?;
                let decode = again.pick()?.targets.first()?.pair?.decode_id;
                if held.decode_chosen() {
                    return None;
                }
                decode
            } else {
                decode
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
    let side = |side: &str, worker_type| -> Arc<[Arc<dyn Worker>]> {
        let url = |at: usize| format!("http://{side}-{at:04}:8000");
        (0..run.workers)
            .map(|at| typed(url(at), worker_type))
            .collect()
    };
    let prefills = side("prefill", WorkerType::Prefill);
    let decodes = side("decode", WorkerType::Decode);
    if run.overloaded {
        // A pair's prefill side is put to no rule: the cell puts its decodes
        // over their thresholds and leaves its prefills open.
        for (at, worker) in decodes.iter().enumerate() {
            worker.set_overloaded(true);
            if at != decodes.len() / 2 {
                for _ in 0..=at {
                    worker.increment_load();
                }
            }
        }
    }
    if run.refused == Refused::Partners {
        for worker in decodes.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let document = document();
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
    behind(engine);
    // Every prefill pairs with every decode.
    let partners: Vec<u32> = (0..decodes.len() as u32).collect();
    let every = (0..prefills.len()).map(|_| partners.as_slice());
    let decode_side = Decodes {
        classes: PartnerClasses::number(decodes.len(), every),
        list: Published::of(&decodes),
        admission: if run.refused == Refused::Limit {
            &Full
        } else {
            &Open
        },
        refused: run.refused,
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
        list: Published::of(&prefills),
        request,
        plain: request(0).plain(),
    };
    // A pair's reports are converted a list: its two lists' counts together.
    let converted = || host.conversions() + engine.conversions(decode_side.list.lent());
    measure(
        run,
        &host,
        upkeep,
        || feed.publish(&loads),
        converted,
        |seed| decode_side.place(&host, seed),
    );
    true
}
