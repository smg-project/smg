//! The program: what it holds, how it starts the engine, and the one way it
//! serves a request.
//!
//! It holds what a program that routes requests holds already, in the
//! worker crate's own types: the worker registry, with the program's value
//! of each membership generation ([`Lists`]), which is the list the registry
//! published and the empty cell the program keeps beside it. Its load
//! monitor's channel goes to the engine once, at the start. Its sends are a
//! function its builder gives ([`Program::start`]): handed the worker a
//! request goes to, it answers the status that worker answered.
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
    document::{
        AppliedSpec, Assignment, ModeSettings, Pairing, PolicySettings, RoutingKeyOverrideSettings,
        RoutingSettings,
    },
    engine::{
        HeaderRead, LentList, ListCell, PlaceRequest, PolicyEngine, PoolMembers, Unplaced,
        WorkerId, WorkerLists,
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

/// The routing settings of a gateway that sets `policy` and nothing else,
/// each other setting at the value a configuration that leaves it out gets.
pub fn settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        mode: ModeSettings::Regular,
        policy,
        enable_igw: false,
        policy_spec: None,
        grouped_dp_sticky: None,
        dp_minimum_tokens_scheduler: false,
        pd_pairing_mode: Pairing::Lenient,
        worker_warmup_secs: 60,
        worker_warmup_share: 0.25,
        worker_warmup_blocks: 1024,
        worker_warmup_thin_ratio: 0.5,
        worker_warmup_divert_every: 8,
        routing_key_override: RoutingKeyOverrideSettings {
            enabled: false,
            headers: vec!["x-smg-routing-key".to_owned()],
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode: Assignment::Delegate,
        },
        worker_overload_protection: true,
        worker_overload_waiting_requests: Some(8),
        worker_overload_token_usage: Some(0.8),
        worker_overload_shed: false,
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

/// Every call the program makes of the engine once it is started.
///
/// [`Engine::place`] is the host contract's one call for a request. Each
/// other method is a call the contract still owes a program, and leaves
/// this list with the change that pays it. The start ([`Program::start`])
/// is the one other place that names the engine.
struct Engine(&'static PolicyEngine);

impl Engine {
    /// The one call for a request.
    fn place<'a>(&self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced> {
        self.0.place(request)
    }

    /// The engine's value of a lent list. Owed: a request lends each list
    /// with its cell, and the call for a request asks the cell.
    fn members_of<'a>(&self, lent: LentList<'a>) -> &'a PoolMembers {
        self.0.members_of(lent)
    }

    /// A worker left. Owed: the engine's own task follows the registry's
    /// events.
    fn forget(&self, address: &str) {
        self.0.forget(address);
    }

    /// A worker joined. Owed with [`Engine::forget`].
    fn joined(&self, address: &str) {
        self.0.joined(address);
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

    /// The attempt ended with `status`.
    pub fn ended(&mut self, status: u16) {
        let class = match status {
            ..400 => TransportClass::Ok,
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
    registry: WorkerRegistry<Lists>,
    engine: Engine,
    sends: Sends,
}

impl Program {
    /// The program's start: the engine built from `spec` for the life of
    /// the process, and the load monitor's channel handed once, where the
    /// program has a monitor. It answers the program with the engine's own
    /// task, which a program spawns on its runtime: the task keeps the
    /// newest load snapshot at hand and runs the engine's upkeep. `sends`
    /// are the program's sends.
    pub fn start(
        spec: AppliedSpec,
        loads: Option<LoadReceiver>,
        sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
    ) -> (Self, impl Future<Output = ()> + Send + 'static) {
        let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
        if let Some(loads) = loads {
            engine.set_loads(loads);
        }
        let program = Self {
            registry: WorkerRegistry::new(),
            engine: Engine(engine),
            sends: Box::new(sends),
        };
        (program, engine.run())
    }

    /// A worker joins: the registry publishes the next generation of its
    /// model, with an empty cell. `false` where the registry holds a worker
    /// at its address already.
    pub fn join(&self, worker: Arc<dyn Worker>) -> bool {
        let address = worker.url().to_owned();
        let joined = self.registry.register(worker).is_some();
        if joined {
            self.engine.joined(&address);
        }
        joined
    }

    /// The worker at `address` leaves. `false` where the registry holds
    /// none there.
    pub fn leave(&self, address: &str) -> bool {
        let left = self.registry.remove_by_url(address).is_some();
        if left {
            self.engine.forget(address);
        }
        left
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
        let members = lists
            .as_deref()
            .map(|lists| self.engine.members_of(lists.lent()));
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
                members.map_or(WorkerLists::None, WorkerLists::Single),
            )
        };
        let placement = self.engine.place(&request)?;
        let guard = WorkerLoadGuard::with_key(Arc::clone(&placement.worker), placement.routing_key);
        Ok(InFlight {
            placement: placement.kept(),
            _guard: guard,
        })
    }

    /// Serves one client request as a route does: places an attempt, says
    /// it sent, sends it, says its end, and where the send was answered a
    /// server error places the next attempt past the workers tried, up to
    /// [`ATTEMPTS`] sends.
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
            if status < 500 || sent.len() == ATTEMPTS {
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
