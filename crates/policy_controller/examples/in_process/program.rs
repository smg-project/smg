//! The program: what it holds, how it starts the engine, and the one way it
//! serves a request.
//!
//! It holds what a program that routes requests holds already, in the
//! worker crate's own types: the worker registry, with the program's value
//! of each membership generation ([`Lists`]), which is the list the registry
//! published and the empty cell the program keeps beside it. The engine
//! takes the registry and the load monitor's channel once, at its start
//! ([`PolicyEngine::start`]), and reads both in place: the program tells it
//! nothing of a worker that joins or leaves. The program hands its lists
//! by model beside the start ([`PublishedLists`]), and the engine's task
//! fills the cell of a model's newest list behind each membership change.
//! Its sends are a function its
//! builder gives ([`Program::start`]): handed the worker a request goes to,
//! it answers the status that worker answered.
//!
//! The document is the one its routing settings mean
//! ([`RoutingSettings::policy_document`]): the program writes settings as a
//! configuration file would, and no document.
//!
//! Of the engine the program asks what [`Engine`] lists, and nothing else.
//! It keeps no circuit breaker's count and chooses no rank inside a worker:
//! both are a route's own, behind the engine's answer.

use std::{future::Future, sync::Arc};

use http::HeaderMap;
use smg_policy_controller::{
    document::{AppliedSpec, PolicySettings, RoutingSettings},
    engine::{
        HeaderRead, LentList, ListCell, ModelLists, PlaceRequest, PolicyEngine, Unplaced,
        WorkerData, WorkerId, WorkerLists,
    },
    tracking::{Outcome, Placement, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::{inflight::WorkerLoadGuard, state::LoadReceiver},
    registry::{Generation, WorkerRegistry},
};

/// The sends one client request gets at most: the first and two more.
pub const ATTEMPTS: usize = 3;

/// The status the program answers a request the engine named no worker
/// for, whatever the reason.
pub const NO_WORKER: u16 = 503;

/// The statuses a send is asked again for: the set a gateway's retry layer
/// asks again for.
pub const ASKED_AGAIN: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// The routing settings of a gateway that sets `policy` and nothing else:
/// each other setting at the value a configuration that leaves it out gets
/// ([`RoutingSettings::default`]).
pub fn settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        policy,
        ..RoutingSettings::default()
    }
}

/// The program's value of one membership generation of a model: the list
/// the registry published, and the cell the program keeps beside it, empty
/// at the publish.
#[derive(Debug)]
pub struct Lists {
    all: Arc<[Arc<dyn Worker>]>,
    cell: ListCell,
}

impl Generation for Lists {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            all: members,
            cell: ListCell::default(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.all
    }
}

impl Lists {
    /// The list lent with its cell. The program knows nothing of a worker
    /// beside its object, so it lends no fact.
    fn lent(&self) -> LentList<'_> {
        LentList {
            workers: &self.all,
            cell: &self.cell,
            facts: None,
        }
    }
}

/// The program's lists by model, as it hands them to the engine beside its
/// start: the generation the registry holds of a model now, lent as a
/// request lends it. The engine's task asks them behind each membership
/// word of the registry and fills a cell it finds empty, so a list's first
/// request reads a filled cell where the task came first.
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
        hand(lists.lent());
    }
}

/// Every call the program makes of the engine once it is started: one.
///
/// [`Engine::place`] is the host contract's one call for a request. The
/// start ([`Program::start`]) is the one other place that names the engine.
struct Engine(&'static PolicyEngine);

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
/// of the worker it names. A route keeps both to the end of the response,
/// streamed or not.
pub struct InFlight {
    /// The placement, kept: it owns all it holds.
    pub placement: Placement<'static>,
    _guard: WorkerLoadGuard,
}

impl InFlight {
    /// The attempt went out to its worker.
    pub fn sent(&mut self) {
        self.placement.sent(Side::Single);
    }

    /// The attempt ended with `status`. A 408 and a 504 say the worker
    /// ran out of time, whichever side of 500 they stand on.
    pub fn ended(&mut self, status: u16) {
        let class = match status {
            ..400 => TransportClass::Ok,
            408 | 504 => TransportClass::Timeout,
            400..500 => TransportClass::ClientError,
            _ => TransportClass::ServerError,
        };
        let outcome = Outcome::reported(status, class);
        self.placement.ended(Side::Single, outcome);
    }
}

/// How the program answered one client request.
#[derive(Debug, PartialEq, Eq)]
pub struct Answer {
    /// The address of the worker each attempt was sent to, in order.
    pub sent: Vec<String>,
    /// The status the client reads: the last send's, or [`NO_WORKER`]
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
    sends: Sends,
}

impl Program {
    /// The program's start: the engine's one call
    /// ([`PolicyEngine::start`]), made over the program's own data. The
    /// engine is built from `spec` for the life of the process and takes
    /// the program's registry, and its load monitor's channel where the
    /// program has a monitor. This program keeps no index of cache events.
    /// Beside the start it hands the engine its lists by model
    /// ([`PublishedLists`]).
    ///
    /// It answers the program with the engine's own task, which a program
    /// spawns on its runtime ahead of its first request. The task keeps the
    /// newest load snapshot at hand, runs the engine's upkeep, and follows
    /// the registry: it tells the document's stages a worker that joined or
    /// left, and fills the cell of the model's list that stands then.
    /// `sends` are the program's sends.
    pub fn start(
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
        let (engine, task) = PolicyEngine::start(spec, data);
        engine.set_lists(Arc::new(PublishedLists(Arc::clone(&registry))));
        let program = Self {
            registry,
            engine: Engine(engine),
            sends: Box::new(sends),
        };
        (program, task)
    }

    /// A worker joins: the registry publishes the next generation of its
    /// model, with an empty cell. `false` where the registry holds a worker
    /// at its address already. The engine's task hears the join from the
    /// registry, and fills the cell of the model's list that stands then.
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
    /// stands, for a request that already tried `tried`. The answer is the
    /// placement kept with the worker's load guard, taken under the
    /// routing key the placement names.
    ///
    /// # Errors
    ///
    /// Why the engine named no worker.
    pub fn place(&self, asked: &Asked<'_>, tried: &[WorkerId]) -> Result<InFlight, Unplaced> {
        let lists = self.published(asked.model);
        let lent = lists.as_deref().map(Lists::lent);
        let headers: &dyn HeaderRead = match asked.headers {
            Some(headers) => headers,
            None => &(),
        };
        let request = PlaceRequest {
            headers,
            attempt: u8::try_from(tried.len()).unwrap_or(u8::MAX),
            tried,
            ..PlaceRequest::of(
                asked.model,
                lent.map_or(WorkerLists::None, WorkerLists::Single),
            )
        };
        let placement = self.engine.place(&request)?;
        let guard = WorkerLoadGuard::with_key(Arc::clone(&placement.worker), placement.routing_key);
        Ok(InFlight {
            placement: placement.kept(),
            _guard: guard,
        })
    }

    /// Serves one client request: places an attempt, says it sent, sends
    /// it, says its end, and where the send was answered a status a gateway
    /// asks again for ([`ASKED_AGAIN`]) places the next attempt past the
    /// workers tried, up to [`ATTEMPTS`] sends. A route does more than
    /// this: it waits between two sends, and a response can say that it is
    /// not to be asked again.
    pub fn serve(&self, asked: &Asked<'_>) -> Answer {
        let (mut tried, mut sent) = (Vec::new(), Vec::new());
        loop {
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
            let status = (self.sends)(attempt.placement.worker.as_ref());
            attempt.ended(status);
            sent.push(attempt.placement.worker.url().to_owned());
            if !ASKED_AGAIN.contains(&status) || sent.len() == ATTEMPTS {
                return Answer {
                    sent,
                    status,
                    unplaced: None,
                };
            }
            tried.push(attempt.placement.id());
        }
    }
}
