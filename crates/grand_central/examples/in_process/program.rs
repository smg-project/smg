//! The program: what it holds, how it starts the engine, and the one way it
//! serves a request.
//!
//! It holds what a program that routes requests holds already, in the
//! worker crate's own types: the worker registry, with the program's value
//! of each membership generation ([`Lists`]). That value keeps the members
//! the registry published. At its first lending it cuts them into one list
//! for each kind of worker the program routes to, each with an empty cell
//! of its own ([`PublishedList`]): the single workers, and the prefills and
//! the decodes of a pair.
//!
//! The engine takes the registry and the load monitor's channel once, at
//! its start ([`InProcess::start`]), and reads both in place: the program
//! tells it nothing of a worker that joins or leaves. The program hands its
//! lists by model beside the start ([`PublishedLists`]), and the engine's
//! task fills the cells of a model's newest lists behind each membership
//! change. Its sends are a function its builder gives ([`Program::start`]):
//! handed the worker a request goes to, it answers the status that worker
//! answered.
//!
//! The document is the one its routing settings mean
//! ([`RoutingSettings::policy_document`]): the program writes settings as a
//! configuration file would, and no document. The settings' mode says the
//! kinds of worker the program routes to as well ([`Plan`]): single workers
//! ([`settings`]), or pairs of a prefill worker and a decode worker
//! ([`pair_settings`]).
//!
//! Of the engine the program asks what [`Engine`] lists, and nothing else.
//! It keeps no circuit breaker's count and chooses no rank inside a worker:
//! both are a route's own, behind the engine's answer. It keeps no limit of
//! its own on a pair's workers either ([`NoLimit`]).

use std::{
    future::Future,
    sync::{Arc, OnceLock},
};

use http::HeaderMap;
use smg_grand_central::{
    document::{AppliedSpec, ModeSettings, PolicySettings, RoutingSettings},
    engine::{
        HeaderRead, InProcess, LentList, ModelLists, PairAdmission, PairLists, PartnerClasses,
        PlaceRequest, PublishedList, Unplaced, WorkerData, WorkerId, WorkerLists,
    },
    tracking::{Outcome, Placement, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::{inflight::WorkerLoadGuard, state::LoadReceiver},
    registry::{Generation, WorkerRegistry},
    worker::WorkerType,
};

/// The attempts one client request gets at most: the first and two more.
pub const ATTEMPTS: usize = 3;

/// The status the program answers a request the engine named no worker
/// for, whatever the reason.
pub const NO_WORKER: u16 = 503;

/// The statuses a send is asked again for: the set a gateway's retry layer
/// asks again for.
pub const ASKED_AGAIN: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// The routing settings of a gateway that sets `policy` and nothing else:
/// each other setting at the value a configuration that leaves it out gets
/// ([`RoutingSettings::default`]). Such a gateway routes to single workers.
pub fn settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        policy,
        ..RoutingSettings::default()
    }
}

/// The routing settings of a gateway that routes to pairs of a prefill
/// worker and a decode worker, both sides under `policy`, and sets nothing
/// else ([`settings`]).
pub fn pair_settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: None,
        },
        ..settings(policy)
    }
}

/// The kinds of worker the program routes to: what each of its requests
/// lends the engine. It is the settings' word, as a gateway's mode is, and
/// is not read off a model's workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Single workers: a request lends their one list.
    Single,
    /// Pairs of a prefill worker and a decode worker: a request lends the
    /// two lists.
    Pair,
}

impl Plan {
    /// The plan `settings` mean by their mode. `None` for a mode this
    /// program does not serve.
    pub fn of(settings: &RoutingSettings) -> Option<Self> {
        match settings.mode {
            ModeSettings::Regular => Some(Self::Single),
            ModeSettings::PrefillDecode { .. } => Some(Self::Pair),
            _ => None,
        }
    }

    /// The workers one attempt under the plan is sent to.
    const fn sends(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Pair => 2,
        }
    }
}

/// The program's value of one membership generation of a model: the
/// members the registry published and, cut from them at the value's first
/// lending, the list of each kind of worker with the cell the program keeps
/// beside it ([`Cut`]). The program knows nothing of a worker beside its
/// object, so it states no fact of one.
#[derive(Debug)]
pub struct Lists {
    members: Arc<[Arc<dyn Worker>]>,
    cut: OnceLock<Cut>,
}

/// A generation's members by the kind of worker each is: each kind's list
/// in the registry's order with an empty cell beside it, and the pair
/// classes of the prefills and the decodes. Which workers pair is the
/// program's own word: here every prefill pairs with every decode.
#[derive(Debug)]
struct Cut {
    single: PublishedList,
    prefill: PublishedList,
    decode: PublishedList,
    classes: PartnerClasses,
}

impl Generation for Lists {
    /// Made under the registry's guard: no worker is read and no lock
    /// taken. The lists are cut behind it ([`Lists::cut`]).
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            members,
            cut: OnceLock::new(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.members
    }
}

impl Lists {
    /// The lists, cut at the first ask: each member is read once, for the
    /// kind of worker it is. Asks that meet there wait for the one that
    /// cuts. The engine's task asks behind each membership change, so a
    /// request cuts only where it comes ahead of the task.
    fn cut(&self) -> &Cut {
        self.cut.get_or_init(|| {
            let of = |kind: WorkerType| -> Arc<[Arc<dyn Worker>]> {
                let members = self.members.iter();
                let of_kind = members.filter(|worker| *worker.worker_type() == kind);
                of_kind.cloned().collect()
            };
            let (prefills, decodes) = (of(WorkerType::Prefill), of(WorkerType::Decode));
            // Each prefill states every place of the decode list.
            let every: Vec<u32> = (0u32..).take(decodes.len()).collect();
            let partners = prefills.iter().map(|_| every.as_slice());
            Cut {
                classes: PartnerClasses::number(decodes.len(), partners),
                single: PublishedList::new(of(WorkerType::Regular)),
                prefill: PublishedList::new(prefills),
                decode: PublishedList::new(decodes),
            }
        })
    }

    /// The single workers' list lent with its cell, and with no fact.
    fn lent(&self) -> LentList<'_> {
        self.cut().single.lent()
    }

    /// A pair's two lists lent, each with its cell, and their pair classes:
    /// every place of each list, under no veto of the program's own and no
    /// limit ([`NoLimit`]).
    fn pair(&self) -> PairLists<'_> {
        let cut = self.cut();
        PairLists {
            prefill: cut.prefill.lent(),
            decode: cut.decode.lent(),
            classes: &cut.classes,
            admission: &NoLimit,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        }
    }
}

/// The program's reads of its own admission for a pair. It keeps no limit
/// on a prefill and no rooms on a decode, so every prefill has capacity and
/// every decode a room. A gateway answers from its prefill queue and its
/// decode rooms here.
#[derive(Debug)]
struct NoLimit;

impl PairAdmission for NoLimit {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// The program's lists by model, as it hands them to the engine beside its
/// start: the generation the registry holds of a model now, each list of
/// it that holds a worker lent as a request lends it. The engine's task
/// asks them behind each membership word of the registry and fills a cell
/// it finds empty, so a list's first request reads a filled cell where the
/// task came first.
#[derive(Debug)]
struct PublishedLists(Arc<WorkerRegistry<Lists>>);

impl ModelLists for PublishedLists {
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>)) {
        // The generation is taken out of the registry first: the engine
        // reads each worker of a list it fills, and no guard of the
        // registry stands meanwhile.
        let Some(lists) = self.0.model_generation(model) else {
            return;
        };
        let cut = lists.cut();
        for list in [&cut.single, &cut.prefill, &cut.decode] {
            if !list.workers().is_empty() {
                hand(list.lent());
            }
        }
    }
}

/// Every call the program makes of the engine once it is started: one.
///
/// [`Engine::place`] is the host contract's one call for a request, a
/// single one and a pair alike. The start ([`Program::start`]) is the one
/// other place that names the engine.
struct Engine(&'static InProcess);

impl Engine {
    /// The one call for a request.
    fn place<'a>(&self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced> {
        self.0.place(request)
    }
}

/// What a client's request states.
#[derive(Clone, Copy, Debug)]
pub struct Asked<'a> {
    /// The model the request is for.
    pub model: &'a str,
    /// The request's headers, where it carries any.
    pub headers: Option<&'a HeaderMap>,
}

/// One attempt of a request on its way: the placement, and the load guard
/// of each worker it names, a pair's decode as well as its prefill. A route
/// keeps them to the end of the response, streamed or not.
pub struct InFlight {
    /// The placement, kept: it owns all it holds.
    pub placement: Placement<'static>,
    _guard: WorkerLoadGuard,
    _decode: Option<WorkerLoadGuard>,
}

impl InFlight {
    /// The sides of the attempt, in the order they are sent: the one side
    /// of a single request, or a pair's prefill and then its decode.
    pub fn sides(&self) -> &'static [Side] {
        if self.placement.decode.is_some() {
            &[Side::Prefill, Side::Decode]
        } else {
            &[Side::Single]
        }
    }

    /// The worker `side` of the attempt goes to.
    pub fn worker(&self, side: Side) -> &Arc<dyn Worker> {
        match (side, &self.placement.decode) {
            (Side::Decode, Some(decode)) => decode,
            _ => &self.placement.worker,
        }
    }

    /// Says the attempt sent: its one side, or both sides of a pair in one
    /// word, ahead of the first send.
    pub fn sent(&mut self) {
        for &side in self.sides() {
            self.placement.sent(side);
        }
    }

    /// `side` of the attempt ended with `status`. A 408 and a 504 say the
    /// worker ran out of time, whichever side of 500 they stand on.
    pub fn ended(&mut self, side: Side, status: u16) {
        let class = match status {
            ..400 => TransportClass::Ok,
            408 | 504 => TransportClass::Timeout,
            400..500 => TransportClass::ClientError,
            _ => TransportClass::ServerError,
        };
        let outcome = Outcome::reported(status, class);
        self.placement.ended(side, outcome);
    }
}

/// How the program answered one client request.
#[derive(Debug, PartialEq, Eq)]
pub struct Answer {
    /// The address of the worker each send went to, in order. A pair's
    /// attempt is two sends: its prefill, then its decode.
    pub sent: Vec<String>,
    /// The status the client reads: the last attempt's, or [`NO_WORKER`]
    /// where the engine named no worker.
    pub status: u16,
    /// Why the engine named no worker, where it named none.
    pub unplaced: Option<Unplaced>,
}

/// A send: handed the worker a request goes to, it answers the status the
/// worker answered.
type Sends = Box<dyn Fn(&dyn Worker) -> u16 + Send + Sync>;

/// The program.
pub struct Program {
    /// The program's registry. The engine holds it too, since its start.
    registry: Arc<WorkerRegistry<Lists>>,
    engine: Engine,
    plan: Plan,
    sends: Sends,
}

impl Program {
    /// The program's start: the engine's one call
    /// ([`InProcess::start`]), made over the program's own data. The
    /// engine is built from `spec` for the life of the process and takes
    /// the program's registry, and its load monitor's channel where the
    /// program has a monitor. This program keeps no index of cache events.
    /// Beside the start it hands the engine its lists by model
    /// ([`PublishedLists`]). `plan` is the kinds of worker the settings
    /// behind `spec` route to ([`Plan::of`]).
    ///
    /// It answers the program with the engine's own task, which a program
    /// spawns on its runtime ahead of its first request. The task keeps the
    /// newest load snapshot at hand, runs the engine's upkeep, and follows
    /// the registry: it tells the document's stages a worker that joined or
    /// left, and fills the cells of the model's lists that stand then.
    /// `sends` are the program's sends.
    pub fn start(
        plan: Plan,
        spec: AppliedSpec,
        loads: Option<LoadReceiver>,
        sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
    ) -> (Self, impl Future<Output = ()> + Send + 'static) {
        let registry = Arc::new(WorkerRegistry::new());
        let handed: Arc<WorkerRegistry<Lists>> = Arc::clone(&registry);
        let data = WorkerData {
            registry: handed,
            loads,
            cache_indexes: None,
        };
        let (engine, task) = InProcess::start(spec, data);
        engine.set_lists(Arc::new(PublishedLists(Arc::clone(&registry))));
        let program = Self {
            registry,
            engine: Engine(engine),
            plan,
            sends: Box::new(sends),
        };
        (program, task)
    }

    /// A worker joins: the registry publishes the next generation of its
    /// model, with empty cells. `false` where the registry holds a worker
    /// at its address already. The engine's task hears the join from the
    /// registry, and fills the cells of the model's lists that stand then.
    pub fn join(&self, worker: Arc<dyn Worker>) -> bool {
        self.registry.register(worker).is_some()
    }

    /// The worker at `address` leaves. `false` where the registry holds
    /// none there. The registry finds a worker by an address written
    /// without its scheme too. The engine's task hears the removal with
    /// the worker's own object, so the stages are told the worker under
    /// the address it states.
    pub fn leave(&self, address: &str) -> bool {
        self.registry.remove_by_url(address).is_some()
    }

    /// The model's generation as the registry holds it now.
    pub fn published(&self, model: &str) -> Option<Arc<Lists>> {
        self.registry.model_generation(model)
    }

    /// One attempt placed: the engine asked once over the generation that
    /// stands, for a request that already tried `tried`. It lends what the
    /// program's plan says: the single workers' list, or a pair's two. The
    /// answer is the placement kept with the load guard of each worker it
    /// names, taken under the routing key the placement names.
    ///
    /// # Errors
    ///
    /// Why the engine named no worker.
    pub fn place(&self, asked: &Asked<'_>, tried: &[WorkerId]) -> Result<InFlight, Unplaced> {
        let lists = self.published(asked.model);
        let lent = match (lists.as_deref(), self.plan) {
            (None, _) => WorkerLists::None,
            (Some(lists), Plan::Single) => WorkerLists::Single(lists.lent()),
            (Some(lists), Plan::Pair) => WorkerLists::Pair(lists.pair()),
        };
        let headers: &dyn HeaderRead = match asked.headers {
            Some(headers) => headers,
            None => &(),
        };
        // A failed attempt states each worker it was sent to as tried, a
        // pair's two: the attempt's number is counted in attempts.
        let attempt = tried.len() / self.plan.sends();
        let request = PlaceRequest {
            headers,
            attempt: u8::try_from(attempt).unwrap_or(u8::MAX),
            tried,
            ..PlaceRequest::of(asked.model, lent)
        };
        let placement = self.engine.place(&request)?;
        let key = placement.routing_key;
        let guard = WorkerLoadGuard::with_key(Arc::clone(&placement.worker), key);
        let decode = placement.decode.as_ref();
        let decode = decode.map(|decode| WorkerLoadGuard::with_key(Arc::clone(decode), key));
        Ok(InFlight {
            placement: placement.kept(),
            _guard: guard,
            _decode: decode,
        })
    }

    /// Serves one client request: places an attempt, says it sent, sends
    /// each side of it, says each side's end, and where the attempt was
    /// answered a status a gateway asks again for ([`ASKED_AGAIN`]) places
    /// the next attempt past the workers tried, up to [`ATTEMPTS`]
    /// attempts. Both sides of a pair are sent, whatever the first answered:
    /// the attempt's status is its prefill's where that one failed, and its
    /// decode's otherwise. A route does more than this: it waits between two
    /// attempts, and a response can say that it is not to be asked again.
    pub fn serve(&self, asked: &Asked<'_>) -> Answer {
        let (mut tried, mut sent, mut attempts) = (Vec::new(), Vec::new(), 0);
        loop {
            attempts += 1;
            let mut attempt = match self.place(asked, &tried) {
                Ok(attempt) => attempt,
                Err(unplaced) => {
                    return Answer {
                        sent,
                        status: NO_WORKER,
                        unplaced: Some(unplaced),
                    };
                }
            };
            attempt.sent();
            let mut status = 0;
            for &side in attempt.sides() {
                let worker = Arc::clone(attempt.worker(side));
                let answered = (self.sends)(worker.as_ref());
                attempt.ended(side, answered);
                sent.push(worker.url().to_owned());
                // The first side that failed is the attempt's answer.
                if status < 400 {
                    status = answered;
                }
            }
            if !ASKED_AGAIN.contains(&status) || attempts == ATTEMPTS {
                return Answer {
                    sent,
                    status,
                    unplaced: None,
                };
            }
            tried.push(attempt.placement.id());
            tried.extend(attempt.placement.decode_id());
        }
    }
}
