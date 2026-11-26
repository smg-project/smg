//! The controller's gRPC service (design 5.2): the one implementation of
//! `smg.policy.v1.GrandCentral`, mounted on every listener beside the
//! health service, one instance a listener, holding that listener's scopes.
//!
//! `Leader` is served, on every listener and by every replica, to a caller
//! with a principal or with none. `Session` and `Decide` are served to a
//! gateway ([`session`], [`decide`]); the leader serves the control scope's
//! three registration calls, the watch of the worker rows
//! ([`watch_workers`]) and the operators' snapshot of those rows and of
//! the spec's status ([`snapshot`]), and Explain reads its deciders' recent
//! samples. A controller given a spec source serves
//! `GetPolicySpec` on the control scope, and `ApplyPolicySpec` and
//! `RollbackPolicySpec` beside it where it leads. Every other call is
//! refused:
//! `PERMISSION_DENIED` where the listener mounts none of the call's scopes
//! (the contract: the scope does not cover the RPC), `UNAUTHENTICATED`
//! where it does and the listener's layer placed no principal on the call
//! ([`principal`]), `UNIMPLEMENTED` where one stands and
//! no step serves the call yet. The step that serves a call replaces that
//! call's one line here, behind [`Service::admit`], acts for the principal
//! it hands back and for no other peer, and adds what it reads to
//! [`Shared`], which [`Shared::new`] alone builds.
//!
//! A spec source that is a file takes no write, since no call writes a
//! file: `ApplyPolicySpec` answers a dry run's verdict with nothing applied
//! and an enforce `read_only`, and `RollbackPolicySpec` answers
//! `read_only`. A spec store takes an enforce: the document is persisted,
//! then applied as the next generation. It takes a rollback to a generation
//! on file, whose document runs again as the next generation, and
//! `GetPolicySpec` answers a generation on file by its number. One dry run
//! runs at a time and one write, and a call past the ones that wait their
//! turn is told to ask again.

use std::{
    io,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime},
};

use prost::Message as _;
use smg_worker::load::state::LoadReceiver;
use tokio::sync::watch;
use tonic::{
    codegen::http,
    server::NamedService,
    service::{interceptor::InterceptedService, Routes},
    Code, Request, Response, Status, Streaming,
};

use super::{
    config::Scope,
    decide::{self, Streams},
    decision_slots::{self, DecisionSlots, Reads},
    election::{self, split_identity, ElectionObservation, LeasePlace, Running},
    floor::Floors,
    gateways::Gateways,
    generations::{
        Format, Generations, SpecRejected, Unanswered, DRY_RUNS_WAITING, WRITES_WAITING,
    },
    lifecycle,
    membership::MembershipCalls,
    principal::{self, Layer, Principal},
    replication::{Batches, Replication},
    session,
    snapshot::{self, SnapshotQuery},
    standby::Lag,
    watch_workers::{self, Feed, Readers, Source, Start, Watching},
    wire, PROTOCOL_FLOOR, PROTOCOL_VERSION,
};
use crate::{
    document::{
        operator::{DryRun, EnforceOutcome, EnforceRefused, NeedsRow},
        spec::Digest,
    },
    engine::{explain::DecisionRecord, framework::Engine},
    service::{
        elector::{Elector, Fence},
        proto::{
            self,
            grand_central_server::{GrandCentral, GrandCentralServer},
        },
        replication::NotLeading,
    },
    stages::registry::StageRegistry,
    store::MemoryStore,
    worker_host::{Members, Objects},
};

/// A leader's refusal in the instant between its take of the lease and its
/// store's entry into the term.
const NOT_IN_TERM: &str = "this replica leads a term its store has not entered yet";

/// A leader's refusal where its take started no term in its store's log:
/// the log followed a term at or above the one taken. The election leaves
/// that lease in the take's step, or keeps it where no term is left to take.
const NO_TERM_LED: &str =
    "this replica holds the lease and its store leads no term: this call is not served until a later take";

/// A leader's refusal of a call admitted under a take its store has since
/// passed: the fence was read before the later take, the log after it. A
/// moment: the next call reads the later term.
const TAKE_OVER: &str =
    "this replica's store leads a later term than this call was admitted under: retry";

/// A watch's refusal by a replica that does not lead.
const FOLLOWS_NO_LOG: &str = "this replica does not lead and follows no leader's log";

/// The path the transport routes the `Decide` call by.
const DECIDE_PATH: &str = "/smg.policy.v1.GrandCentral/Decide";

/// The contract's service behind two bounds on a message as received: a
/// `Decide` frame's own on that call ([`wire::MAX_FRAME`]), the
/// transport's on every other.
#[derive(Clone)]
struct ByCall<S> {
    decide: S,
    rest: S,
}

impl<S, B> tonic::codegen::Service<http::Request<B>> for ByCall<S>
where
    S: tonic::codegen::Service<http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // Either side may take the next request: both are asked.
        match self.decide.poll_ready(cx) {
            Poll::Ready(Ok(())) => self.rest.poll_ready(cx),
            other => other,
        }
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        if request.uri().path() == DECIDE_PATH {
            self.decide.call(request)
        } else {
            self.rest.call(request)
        }
    }
}

impl<S: NamedService> NamedService for ByCall<S> {
    const NAME: &'static str = S::NAME;
}

/// What every listener's instance of the service reads. A step that serves
/// a call adds its state here.
pub(crate) struct Shared {
    /// This controller's id.
    pub(crate) controller_id: String,
    /// Its dialable address, `host:port`; empty when no client could be
    /// told one.
    pub(crate) address: String,
    /// The holder identity it elects under.
    pub(crate) identity: String,
    /// The controller's one fence: the judge of "leader" at every answer.
    pub(crate) fence: Arc<Fence>,
    /// The election's last sight of the lease.
    pub(crate) standing: watch::Receiver<ElectionObservation>,
    /// The controller's store: what the calls feed and read, and the run
    /// loop's tasks with them.
    pub(crate) store: Arc<MemoryStore>,
    /// Who is present: the gateways a `Session` welcomed, over this fence
    /// and this store.
    pub(crate) gateways: Arc<Gateways>,
    /// The spec's generations over that store: the door a spec source
    /// applies through, and the document the spec calls read.
    pub(crate) generations: Arc<Generations>,
    /// How long a `Session` stream may stand with no `Hello`.
    pub(crate) hello_deadline: Duration,
    /// The membership door over that store: the control scope's
    /// registration calls, and the writes of the replica's own discovery.
    pub(crate) door: Arc<MembershipCalls>,
    /// The replica's worker objects: what a standby's applier lays its
    /// leader's rows as, and what the door decides a row's write over.
    pub(crate) objects: Arc<Objects<Members>>,
    /// The replication scope's calls over that store and its log.
    pub(crate) replication: Replication,
    /// The threads the decisions run on, a model's on one.
    pub(crate) deciders: DecisionSlots,
    /// The decide streams each gateway holds open.
    pub(crate) streams: Arc<Streams>,
    /// The run loop's word that the listeners close: `true` from the end
    /// of the drain's settle window on. An open decide stream ends at it.
    /// The drain alone says it: a listener holds this receiver.
    pub(crate) stop: watch::Receiver<bool>,
    /// The log that store journals to and the wake of its watched writes:
    /// what a watch of the rows reads.
    pub(crate) feed: Feed,
    /// The term this replica leads, if it leads one, told at a change alone
    /// by the run loop's one task: how an idle watch learns its replica
    /// stopped leading, and a task of the lead that another term began.
    /// That task alone tells it: a listener holds this receiver, whose
    /// clone has seen no word, and marks its own seen where it hears from
    /// its start on.
    pub(crate) role: watch::Receiver<Option<u64>>,
    /// Who reads the rows of that feed and can share its kept cut: the
    /// watches, within the bounds of a watch's open, and the operators'
    /// snapshots in their turn, one taken at a time off the runtime's
    /// workers.
    pub(crate) readers: Arc<Readers>,
    /// A gateway's term floor above this replica's term, on its way to the
    /// election.
    pub(crate) floors: Floors,
    /// How far behind its leader this replica is while it follows one: the
    /// standby loop writes it. It holds the bound a standby reads synced
    /// under, the first of the three counts of design 5.4: the bind makes
    /// it once, where the configuration is known.
    pub(crate) lag: Arc<Lag>,
}

/// Who leads, as this replica knows it at one read of its fence.
pub(crate) struct Leader {
    /// The leader's id; empty when nobody is named.
    pub(crate) controller_id: String,
    /// Its dialable address; empty when nobody is named or it has none.
    pub(crate) address: String,
    /// The highest term this replica knows.
    pub(crate) term: u64,
    /// Whether the leader is this replica.
    pub(crate) is_leader: bool,
}

impl Shared {
    /// The one place a `Shared` is built: a step that serves a call adds
    /// its state as a field and fills it here, from these arguments where
    /// it can, so the other builders do not change. The identity and the
    /// fence are `elector`'s, so no instance answers under an identity
    /// another fence guards. A spec document names the stages `catalog`
    /// holds, and its door is built under that fence where `place`, the
    /// lease's, is one another replica can take
    /// ([`Generations::stating`]): no later step puts it there.
    /// `deciders` is how many threads decide. The store comes with
    /// the feed of the log it journals to, the one value one place builds
    /// ([`journaled`](super::replication::journaled)), and with the holder
    /// of the replica's worker objects, whose rows stand in that store: the
    /// deciders decide over their lists. The run loop's side comes as a
    /// listener holds it: the receivers of the election's sight of the
    /// lease, of the listeners' close and of the role's word, each word's
    /// sender with its one writer, and the standby's lag, made once from
    /// the configured interval.
    ///
    /// # Errors
    ///
    /// A deciding thread could not be started.
    pub(crate) fn new(
        controller_id: String,
        address: String,
        elector: &Elector,
        (standing, stop, role, lag): (
            watch::Receiver<ElectionObservation>,
            watch::Receiver<bool>,
            watch::Receiver<Option<u64>>,
            Arc<Lag>,
        ),
        (store, feed, objects, loads): (
            Arc<MemoryStore>,
            Feed,
            Arc<Objects<Members>>,
            LoadReceiver,
        ),
        (catalog, place): (Arc<StageRegistry>, &LeasePlace),
        deciders: usize,
    ) -> io::Result<Self> {
        let gateways = Arc::new(Gateways::new(
            session::GRACE_PERIOD,
            session::PRESENT_MAX,
            elector.fence(),
            Arc::clone(&store),
        ));
        // The store's word to the engine asks this replica's worker objects.
        // It is given here, ahead of the deciders' start and of any hand: a
        // word the store was given before fails a debug build.
        let kept = lifecycle::tells(&store, &objects);
        debug_assert!(
            kept,
            "the store held a word to the engine before its replica's"
        );
        let deciders = DecisionSlots::start(
            deciders,
            decision_slots::QUEUE_DEPTH,
            &Reads {
                store: Arc::clone(&store),
                objects: Arc::clone(&objects),
                loads,
                fence: elector.fence(),
                standing: standing.clone(),
                clock: Arc::new(Instant::now),
                counts: Arc::default(),
            },
        )?;
        // The door states each generation's overload switch on the worker
        // objects the deciding threads read. Under a lease another replica
        // can take it stands under this replica's fence from its first
        // instant.
        let generations = Arc::new(Generations::stating(
            Arc::clone(&store),
            catalog,
            Arc::clone(&objects),
            place.is_shared().then(|| elector.fence()),
        ));
        Ok(Self {
            controller_id,
            address,
            identity: elector.identity().to_owned(),
            fence: elector.fence(),
            standing,
            generations: Arc::clone(&generations),
            objects: Arc::clone(&objects),
            // The door decides a row's write over the worker objects its
            // deciding threads read.
            door: Arc::new(MembershipCalls::new(
                Arc::clone(&store),
                Arc::clone(&feed.log),
                Arc::clone(&objects),
            )),
            // A standby's first copy is cut from the same objects.
            replication: Replication::new(
                Arc::clone(&store),
                Arc::clone(&feed.log),
                elector.fence(),
                objects,
            )
            .telling(Arc::clone(&generations)),
            lag,
            store,
            gateways,
            hello_deadline: session::HELLO_DEADLINE,
            deciders,
            streams: Arc::new(Streams::new(decide::MAX_STREAMS)),
            stop,
            // The feed's readers are built over it: as many principals
            // watch as may be present, and the snapshots wait their turn.
            readers: Arc::new(Readers::of(
                feed.clone(),
                session::PRESENT_MAX,
                snapshot::SNAPSHOTS_WAITING,
            )),
            feed,
            role,
            floors: Floors::default(),
        })
    }

    /// Starts the election thread over `elector` for this state: the term
    /// goes to its store at every take, each step's sight of the lease to
    /// `standing`, and the election's door to the two that say a word
    /// through it: the floors, a term a gateway saw, and the deciders, the
    /// planned hand-over at a term's mark.
    ///
    /// # Errors
    ///
    /// The election's thread could not be started.
    pub(crate) fn elect(
        &self,
        mut elector: Elector,
        standing: watch::Sender<ElectionObservation>,
    ) -> io::Result<Running> {
        // At a take the door is its source's again, and a replica that
        // deferred its start's document applies it at its first one, behind
        // the stamp and under the fence: a take that never leads publishes
        // nothing.
        let documents = Arc::clone(&self.generations);
        elector.at_take(move || documents.at_take());
        let election = election::start(elector, Arc::clone(&self.store), standing)?;
        let door = election.door();
        self.floors.open(door.clone());
        self.deciders
            .leave_through(move |term| door.hand_over(term));
        Ok(election)
    }

    /// Who leads, given `leading`, one read of the fence: the term this
    /// replica leads under, or `None`. A replica whose fence is open names
    /// itself under the fence's term. Any other names the holder its last
    /// look at the lease saw, or nobody (an empty address: no leader known)
    /// when the lease is free, when that look failed, when it read a term
    /// under the highest this replica saw, when the record still names this
    /// replica behind a closed fence, and when this replica's own take is
    /// newer than that look: a holder is named under a term one write of
    /// the lease stored for it. The term is the highest this replica knows
    /// either way, so no answer of its falls in term.
    pub(crate) fn leader(&self, leading: Option<u64>) -> Leader {
        let (controller_id, address, term, is_leader) = match leading {
            Some(term) => (self.controller_id.clone(), self.address.clone(), term, true),
            None => {
                let standing = self.standing.borrow();
                let fenced = self.fence.term();
                let term = standing.term.max(fenced);
                if standing.holder.is_empty()
                    || standing.holder == self.identity
                    || fenced > standing.term
                {
                    (String::new(), String::new(), term, false)
                } else {
                    let (id, address) = split_identity(&standing.holder);
                    (id.to_owned(), address.to_owned(), term, false)
                }
            }
        };
        Leader {
            controller_id,
            address,
            term,
            is_leader,
        }
    }

    /// Opens a watch of the worker rows for `by`, admitted under `term`,
    /// from `start` and for `models`. The stream reads this state's store,
    /// worker objects, feed and fence, and takes its place among the feed's
    /// readers ([`watch_workers::open`]).
    ///
    /// # Errors
    ///
    /// The readers' refusal of a watch past one of their ceilings.
    pub(crate) fn watch(
        &self,
        term: u64,
        by: &Principal,
        start: Start,
        models: &[String],
    ) -> Result<Watching, Status> {
        // The role's word as it stands at the open is seen: the watch
        // ends at a word told from here on, whatever the word before it.
        let mut role = self.role.clone();
        role.mark_unchanged();
        let source = Source {
            store: Arc::clone(&self.store),
            objects: Arc::clone(&self.objects),
            feed: self.feed.clone(),
            fence: Arc::clone(&self.fence),
            role,
            term,
        };
        watch_workers::open(source, &self.readers, by, start, models.to_vec())
    }

    /// Takes the snapshot `asked` for, as this replica holds it under
    /// `term` ([`snapshot::taken`]). The term is the one its caller read:
    /// the act reads no lead, before the take or behind it. One snapshot is
    /// taken at a time, off the runtime's workers, in the turn of the feed's
    /// readers ([`snapshot::in_turn`]), and its rows come by one of the
    /// roads of [`snapshot::read`]. The readers let the feed's kept cut go
    /// at the last call out, where no watch is open to share it.
    ///
    /// # Errors
    ///
    /// As many calls wait their turn as may, the take did not end, or the
    /// take's own refusal.
    pub(crate) async fn snapshot(
        self: &Arc<Self>,
        asked: SnapshotQuery,
        term: u64,
    ) -> Result<proto::Snapshot, Status> {
        let shared = Arc::clone(self);
        let take = move || {
            snapshot::taken(
                &asked,
                term,
                &shared.controller_id,
                || snapshot::read(&asked, &shared.readers, (&shared.objects, &shared.store)),
                &shared.generations,
                Instant::now(),
            )
        };
        snapshot::in_turn(&self.readers, take).await?
    }
}

/// A call of the service a scope owns. `Leader` is not one: every scope
/// makes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Call {
    Session,
    Decide,
    WatchWorkers,
    RegisterWorker,
    PatchWorker,
    DeregisterWorker,
    GetSnapshot,
    Explain,
    ApplyPolicySpec,
    GetPolicySpec,
    RollbackPolicySpec,
    StepDown,
    WatchRegime,
    Replicate,
}

impl Call {
    /// The call's name in the contract.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Call::Session => "Session",
            Call::Decide => "Decide",
            Call::WatchWorkers => "WatchWorkers",
            Call::RegisterWorker => "RegisterWorker",
            Call::PatchWorker => "PatchWorker",
            Call::DeregisterWorker => "DeregisterWorker",
            Call::GetSnapshot => "GetSnapshot",
            Call::Explain => "Explain",
            Call::ApplyPolicySpec => "ApplyPolicySpec",
            Call::GetPolicySpec => "GetPolicySpec",
            Call::RollbackPolicySpec => "RollbackPolicySpec",
            Call::StepDown => "StepDown",
            Call::WatchRegime => "WatchRegime",
            Call::Replicate => "Replicate",
        }
    }

    /// The scopes that make the call, by the contract's header.
    pub(crate) const fn scopes(self) -> &'static [Scope] {
        match self {
            Call::Session | Call::Decide | Call::WatchWorkers => &[Scope::Gateway],
            Call::GetSnapshot
            | Call::RegisterWorker
            | Call::PatchWorker
            | Call::DeregisterWorker
            | Call::Explain
            | Call::ApplyPolicySpec
            | Call::GetPolicySpec
            | Call::RollbackPolicySpec
            | Call::StepDown
            | Call::WatchRegime => &[Scope::Control],
            Call::Replicate => &[Scope::Replication],
        }
    }
}

/// The response stream of a streaming call no step serves yet. It is never
/// built: the call is refused before a stream exists, and the step that
/// serves the call names its own stream.
type Unserved<T> = Streaming<T>;

/// `google.rpc.Status`: what every gRPC client decodes a status's details
/// as. It says the code and the message again and holds each detail as an
/// [`Any`].
#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<Any>,
}

/// `google.protobuf.Any`: a message under its type's URL.
#[derive(Clone, PartialEq, prost::Message)]
struct Any {
    #[prost(string, tag = "1")]
    type_url: String,
    #[prost(bytes = "vec", tag = "2")]
    value: Vec<u8>,
}

/// The refusal of a replica that does not lead, under `code` and `message`.
/// Its details are a `google.rpc.Status`, the form every gRPC client decodes
/// them as: it says both again and holds a `NotLeader` that names the
/// holder's address; an empty one says no leader is known.
pub(crate) fn not_leader(code: Code, message: impl Into<String>, leader_address: String) -> Status {
    let message = message.into();
    let named = proto::NotLeader {
        leader_address,
        successor_id: String::new(),
    };
    let details = RpcStatus {
        code: code as i32,
        message: message.clone(),
        details: vec![Any {
            type_url: "type.googleapis.com/smg.policy.v1.NotLeader".to_owned(),
            value: named.encode_to_vec(),
        }],
    };
    Status::with_details(code, message, details.encode_to_vec().into())
}

/// One listener's instance of the service.
pub(crate) struct Service {
    /// The scopes this listener mounts.
    scopes: Vec<Scope>,
    shared: Arc<Shared>,
}

impl Service {
    pub(crate) fn new(scopes: Vec<Scope>, shared: Arc<Shared>) -> Self {
        Self { scopes, shared }
    }

    /// This instance as its listener mounts it: behind the listener's one
    /// layer, or behind none.
    pub(crate) fn routes(self, layer: Layer) -> Routes {
        let service = Arc::new(self);
        // A `Decide` frame is held to its bound as received: the transport
        // refuses a longer one at its length. Every other message is under
        // the transport's own limit.
        let decide = GrandCentralServer::from_arc(Arc::clone(&service))
            .max_decoding_message_size(wire::MAX_FRAME);
        let rest = GrandCentralServer::from_arc(service);
        match layer {
            Layer::Off => Routes::new(ByCall { decide, rest }),
            Layer::AsTold => {
                let told = |request: Request<()>| -> Result<Request<()>, Status> {
                    Ok(principal::as_told(request))
                };
                Routes::new(ByCall {
                    decide: InterceptedService::new(decide, told),
                    rest: InterceptedService::new(rest, told),
                })
            }
        }
    }

    /// Admits `call` on this listener and hands back the peer it is from,
    /// or refuses it. A listener answers the calls of the scopes it mounts
    /// (`PERMISSION_DENIED` off them, whoever asks), to a caller its layer
    /// placed a principal on (`UNAUTHENTICATED` with none). The first line
    /// of every served call's body and the one place a body learns its
    /// peer: a body holds no check of its own.
    pub(crate) fn admit<'a, T>(
        &self,
        call: Call,
        request: &'a Request<T>,
    ) -> Result<&'a Principal, Status> {
        if !call
            .scopes()
            .iter()
            .any(|scope| self.scopes.contains(scope))
        {
            return Err(Status::permission_denied(format!(
                "{} is not a call of this listener's scopes",
                call.name()
            )));
        }
        principal::principal(request)
    }

    /// The answer to a call no step serves yet, behind the listener's
    /// admission.
    fn unserved<T, R>(&self, call: Call, request: &Request<T>) -> Result<R, Status> {
        self.admit(call, request)?;
        Err(Status::unimplemented(format!(
            "{} is not served by this build",
            call.name()
        )))
    }

    /// The answer to a spec call on a controller that was given no spec
    /// source, behind the listener's admission: this build serves the call,
    /// and this controller's start named no source for it.
    fn sourceless<T, R>(&self, call: Call, request: &Request<T>) -> Result<R, Status> {
        self.admit(call, request)?;
        Err(Status::unimplemented(format!(
            "{} is not served by a controller started with no spec source",
            call.name()
        )))
    }

    /// The term this replica leads under at `now`, once its store stands in
    /// it: what a write is admitted under. A replica that does not lead
    /// refuses the write: its status details are a `google.rpc.Status`, the
    /// form every gRPC client decodes them as, holding a `NotLeader` that
    /// names the holder as `Leader` does (no address: no leader known). That
    /// address is the holder's gateway listener; a control client finds the
    /// leader's control listener by the health service's word. The lease is
    /// taken before the store is told its term: in between the answer is
    /// `UNAVAILABLE`. A take at or below a term the log followed starts no
    /// term. The answer is `UNAVAILABLE` while that lead lasts: the election
    /// leaves it in the take's own step, or keeps it where no term is left
    /// to take. A call admitted under a take the store has since passed is
    /// told to retry.
    /// So a call that reads this guard is not served under a term the store
    /// does not lead: a membership write, a spec write, its dry run and a
    /// rollback, a standby's stream and its snapshot here, a watch's open
    /// and the operators' snapshot through [`Service::watched`]. A session
    /// and a `Decide` read the fence alone.
    pub(crate) fn leading(&self, now: Instant) -> Result<u64, Status> {
        self.led(now, Code::FailedPrecondition, "this replica does not lead")
    }

    /// The term a read of the rows is served under at `now`: a watch's
    /// open, and the operators' snapshot. A standby serves one once it
    /// follows the leader's log; none does in this build, so a replica that
    /// does not lead is a standby not yet synced: `UNAVAILABLE`, with the
    /// `NotLeader` a write's refusal carries, in the same form.
    fn watched(&self, now: Instant) -> Result<u64, Status> {
        self.led(now, Code::Unavailable, FOLLOWS_NO_LOG)
    }

    /// The term this replica leads under at `now`, once its store leads it:
    /// one read of the log ([`Log::leads`](crate::service::replication::Log::leads)),
    /// whose refusal says whether it is a moment or the whole lead. A
    /// replica that does not lead refuses under `code` with `message`: the
    /// details are the `google.rpc.Status` that says both again and holds
    /// the `NotLeader`.
    fn led(&self, now: Instant, code: Code, message: &'static str) -> Result<u64, Status> {
        if let Some(term) = self.shared.fence.leader_term(now) {
            return match self.shared.feed.log.leads(term) {
                Ok(()) => Ok(term),
                Err(NotLeading::NotYet) => Err(Status::unavailable(NOT_IN_TERM)),
                Err(NotLeading::Follows(_)) => Err(Status::unavailable(NO_TERM_LED)),
                Err(NotLeading::Past(_)) => Err(Status::unavailable(TAKE_OVER)),
            };
        }
        let holder = self.leader_info(now).address;
        Err(not_leader(code, message, holder))
    }

    /// The `Leader` call's answer at `now`: [`Shared::leader`] at one read
    /// of the fence.
    fn leader_info(&self, now: Instant) -> proto::LeaderInfo {
        let shared = &self.shared;
        let Leader {
            controller_id,
            address,
            term,
            is_leader,
        } = shared.leader(shared.fence.leader_term(now));
        // A replica that follows says how far behind it is, by the age of
        // the last frame that left it at its leader's head; a leader follows
        // nobody.
        // `synced` is the same read of that lag, and reads it alone: the
        // live ranks and the generation narrow the same bit with their
        // steps.
        let read = shared.lag.read(now).filter(|_| !is_leader);
        proto::LeaderInfo {
            controller_id,
            address,
            term,
            is_leader,
            synced: read.is_some_and(|(_, lapse)| lapse.is_some()),
            sync: read.map(|(lag, _)| proto::SyncState {
                replication_lag_ms: u32::try_from(lag.as_millis()).unwrap_or(u32::MAX),
                ..proto::SyncState::default()
            }),
            // One lease covers every model until a lease per model group
            // exists.
            groups: Vec::new(),
            protocol_min: PROTOCOL_FLOOR,
            protocol_max: PROTOCOL_VERSION,
        }
    }
}

/// The sampled decision as the control call carries it. Positions are of
/// the decision's generation. Only that generation can supply names.
fn explained(
    record: DecisionRecord,
    engine: &Engine,
    request_id: Vec<u8>,
    gateway_id: String,
) -> proto::DecisionRecord {
    let trace = record.decision.trace();
    let age = Instant::now().saturating_duration_since(record.at);
    let at_unix_micros = SystemTime::now()
        .checked_sub(age)
        .map_or(0, wire::unix_micros);
    let mut answer = proto::DecisionRecord {
        request_id,
        gateway_id,
        generation: trace.generation,
        binding: record.binding.unwrap_or_default(),
        decision: Some(wire::decision(&record.decision)),
        at_unix_micros,
        ..Default::default()
    };
    if let Some(names) = record.names(engine) {
        names.plan.clone_into(&mut answer.plan_name);
        answer.profile_name = names.profile;
        answer.picker_name = names.picker;
        answer.filter_names = names.filters;
        answer.scorer_names = names.scorers;
    }
    answer
}

/// The answer of a write call its source cannot take: the spec's source is
/// a file, which no call writes. Nothing was validated; the number and the
/// digest are the standing generation's.
fn read_only(generations: &Generations) -> proto::ApplyResult {
    let (generation, digest) = generations.standing();
    proto::ApplyResult {
        accepted: false,
        generation,
        digest: digest.map(|digest| digest.to_string()).unwrap_or_default(),
        read_only: true,
        ..proto::ApplyResult::default()
    }
}

/// A document's verdict as a write call answers it, a dry run's or an
/// enforce's: accepted, with the number the document runs under (a dry
/// run's: would run under now) and its digest, or refused, with the
/// standing number and digest and one issue by its field. `read_only` is
/// whether the source takes no write: a file takes none.
fn verdict(
    (standing, standing_digest): (u64, Option<Digest>),
    outcome: Result<(u64, Digest), SpecRejected>,
    needs: Option<(Vec<NeedsRow>, Vec<NeedsRow>)>,
    read_only: bool,
) -> proto::ApplyResult {
    match outcome {
        Ok((generation, digest)) => proto::ApplyResult {
            accepted: true,
            generation,
            digest: digest.to_string(),
            needs: needs.map(wire::needs_diff),
            read_only,
            ..proto::ApplyResult::default()
        },
        Err(rejected) => proto::ApplyResult {
            accepted: false,
            generation: standing,
            digest: standing_digest
                .map(|digest| digest.to_string())
                .unwrap_or_default(),
            issues: vec![proto::SpecIssue {
                field_path: rejected.field_path(),
                message: rejected.to_string(),
                plugin: String::new(),
            }],
            read_only,
            ..proto::ApplyResult::default()
        },
    }
}

/// The status of a spec call that gave no answer of its own: a guard's by
/// the standing generation, a full wait's as one to ask again, a generation
/// no document is held for as not found, and a call's thread that did not
/// end or a file that was not read as internal.
fn unanswered(unanswered: Unanswered) -> Status {
    match unanswered {
        Unanswered::Manager(door @ EnforceRefused::Guarded(..)) => {
            Status::failed_precondition(door.to_string())
        }
        Unanswered::Full => Status::resource_exhausted(format!(
            "{DRY_RUNS_WAITING} dry runs wait their turn: ask again later"
        )),
        Unanswered::Unended(join) => Status::internal(format!("the dry run did not end: {join}")),
        Unanswered::Waiting => Status::resource_exhausted(format!(
            "{WRITES_WAITING} applies wait their turn: ask again later"
        )),
        Unanswered::Unwritten(join) => Status::internal(format!("the apply did not end: {join}")),
        Unanswered::Manager(EnforceRefused::NotHeld(not_held)) => {
            Status::not_found(not_held.to_string())
        }
        Unanswered::Unread { generation, why } => Status::internal(format!(
            "generation {generation} was not read from its file: {why}"
        )),
    }
}

#[tonic::async_trait]
impl GrandCentral for Service {
    type SessionStream = session::Frames;
    type DecideStream = decide::Answers;
    type WatchWorkersStream = Watching;
    type WatchRegimeStream = Unserved<proto::Regime>;
    type ReplicateStream = Batches;

    async fn session(
        &self,
        request: Request<Streaming<proto::GatewayFrame>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let principal = self.admit(Call::Session, &request)?.clone();
        let frames = session::serve(Arc::clone(&self.shared), principal, request.into_inner());
        Ok(Response::new(frames))
    }

    async fn decide(
        &self,
        request: Request<Streaming<proto::DecideRequest>>,
    ) -> Result<Response<Self::DecideStream>, Status> {
        let principal = self.admit(Call::Decide, &request)?.clone();
        let answers = decide::serve(Arc::clone(&self.shared), &principal, request.into_inner())?;
        Ok(Response::new(answers))
    }

    async fn watch_workers(
        &self,
        request: Request<proto::WatchWorkersRequest>,
    ) -> Result<Response<Self::WatchWorkersStream>, Status> {
        let by = self.admit(Call::WatchWorkers, &request)?;
        let term = self.watched(Instant::now())?;
        let asked = request.get_ref();
        let start = watch_workers::start(asked)?;
        let watching = self.shared.watch(term, by, start, &asked.models)?;
        Ok(Response::new(watching))
    }

    async fn register_worker(
        &self,
        request: Request<proto::RegisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        let by = self.admit(Call::RegisterWorker, &request)?;
        self.shared
            .door
            .register(by, || self.leading(Instant::now()), request.get_ref())
            .map(Response::new)
    }

    async fn patch_worker(
        &self,
        request: Request<proto::PatchWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        let by = self.admit(Call::PatchWorker, &request)?;
        self.shared
            .door
            .patch(by, || self.leading(Instant::now()), request.get_ref())
            .map(Response::new)
    }

    async fn deregister_worker(
        &self,
        request: Request<proto::DeregisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        let by = self.admit(Call::DeregisterWorker, &request)?;
        self.shared
            .door
            .deregister(by, || self.leading(Instant::now()), request.get_ref())
            .map(Response::new)
    }

    /// The operators' read of what this replica holds ([`snapshot`]): the
    /// slices this build serves, redacted, under the term this replica
    /// leads. A replica that does not lead refuses it as it refuses a
    /// watch's open. The request's own rules come behind the lead, each
    /// field refused by its name, the spec slice of a controller with no
    /// spec source among them, and a refused request takes no turn. One
    /// snapshot is taken at a time, off the runtime's workers
    /// ([`snapshot::in_turn`]), and its rows come by one of the roads of
    /// [`snapshot::read`]. The lead is read again behind that wait: the
    /// answer is told while this replica still leads the term the call was
    /// admitted under. A standby is restored on its `Replicate` stream,
    /// never from this call.
    async fn get_snapshot(
        &self,
        request: Request<proto::SnapshotRequest>,
    ) -> Result<Response<proto::Snapshot>, Status> {
        self.admit(Call::GetSnapshot, &request)?;
        let term = self.watched(Instant::now())?;
        let asked = snapshot::asked(request.into_inner())?;
        asked.sourced(self.shared.generations.bound())?;
        let taken = self.shared.snapshot(asked, term).await?;
        if self.watched(Instant::now())? != term {
            return Err(Status::unavailable(TAKE_OVER));
        }
        Ok(Response::new(taken))
    }

    /// Reads a redacted decision from this replica's current lead.
    async fn explain(
        &self,
        request: Request<proto::ExplainRequest>,
    ) -> Result<Response<proto::DecisionRecord>, Status> {
        use proto::explain_request::Key;

        self.admit(Call::Explain, &request)?;
        let term = self.leading(Instant::now())?;
        let proto::ExplainRequest { key, gateway_id } = request.into_inner();
        let request_id = match key {
            Some(Key::DecisionId(_)) => {
                return Err(Status::invalid_argument(
                    "decision_id: use request_id and gateway_id",
                ));
            }
            Some(Key::RequestId(id)) if (1..=wire::MAX_REQUEST_ID).contains(&id.len()) => id,
            Some(Key::RequestId(_)) | None => {
                return Err(Status::invalid_argument(
                    "request_id: expected 1 to 256 bytes",
                ));
            }
        };
        if gateway_id.is_empty() {
            return Err(Status::invalid_argument(
                "gateway_id: required with request_id",
            ));
        }
        let record =
            self.shared
                .deciders
                .explain(term, principal::gateway_id(&gateway_id), &request_id);
        let answer = record.map(|record| {
            self.shared
                .store
                .deciding(|world| explained(record, world.spec().engine(), request_id, gateway_id))
        });
        if self.leading(Instant::now())? != term {
            return Err(Status::unavailable(TAKE_OVER));
        }
        answer
            .map(Response::new)
            .ok_or_else(|| Status::not_found("no sampled decision for gateway_id and request_id"))
    }

    /// `ApplyPolicySpec` by a controller with a spec source, its checks in
    /// one order: the scope, the lead, the mode, the source, the guard, the
    /// document. A replica that does not lead refuses by the leader's name.
    /// A write states its mode: none is refused by its field, and audit is
    /// not served by this build. Under a file, which no call writes, an
    /// enforce answers `read_only` with the standing number and digest,
    /// whatever its guard and its document. Under a spec store an enforce
    /// whose `expected_digest` is not the standing one is refused: another
    /// apply landed first. Any other is persisted, then applied as the next
    /// generation, one apply at a time ([`Generations::enforce_in_turn`]),
    /// and answers the number it runs under, or the one issue it was
    /// refused by. A dry run is guarded alike and answers its document's
    /// verdict with nothing applied, one dry run at a time
    /// ([`Generations::dry_run_in_turn`]).
    async fn apply_policy_spec(
        &self,
        request: Request<proto::ApplyRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        let generations = &self.shared.generations;
        if !generations.bound() {
            return self.sourceless(Call::ApplyPolicySpec, &request);
        }
        self.admit(Call::ApplyPolicySpec, &request)?;
        self.leading(Instant::now())?;
        let apply = request.into_inner();
        let enforce = match proto::ApplyMode::try_from(apply.mode) {
            Ok(proto::ApplyMode::DryRun) => false,
            Ok(proto::ApplyMode::Enforce) if generations.writable() => true,
            Ok(proto::ApplyMode::Enforce) => return Ok(Response::new(read_only(generations))),
            Ok(proto::ApplyMode::Audit) => {
                return Err(Status::unimplemented(
                    "ApplyPolicySpec in audit mode is not served by this build",
                ));
            }
            Ok(proto::ApplyMode::Unspecified) | Err(_) => {
                return Err(Status::invalid_argument(
                    "mode: a write states its mode, enforce, audit or dry-run",
                ));
            }
        };
        let Some(format) = Format::named(&apply.format) else {
            // A form no one names takes no pass and no turn: the guard,
            // then the form.
            generations
                .guarded(&apply.expected_digest)
                .map_err(|refused| unanswered(refused.into()))?;
            return Err(Status::invalid_argument("format: name yaml or json"));
        };
        // The guard is the door's to read, behind the turn its pass waits.
        let (expected, document) = (apply.expected_digest, apply.document);
        let answer = if enforce {
            let EnforceOutcome {
                standing,
                ran,
                needs,
            } = generations
                .enforce_in_turn(expected, document, format)
                .await
                .map_err(unanswered)?;
            verdict(standing, ran, needs, false)
        } else {
            let DryRun {
                standing,
                would_run,
                needs,
            } = generations
                .dry_run_in_turn(expected, document, format)
                .await
                .map_err(unanswered)?;
            verdict(standing, would_run, needs, !generations.writable())
        };
        Ok(Response::new(answer))
    }

    /// The standing generation's document as its source wrote it, with the
    /// number and the digest beside it: every replica answers from its own
    /// holder. Under a spec store a generation on file is answered by its
    /// number, from its file ([`Generations::read`]). A controller that was
    /// given no spec source serves no spec call.
    async fn get_policy_spec(
        &self,
        request: Request<proto::GetPolicySpecRequest>,
    ) -> Result<Response<proto::PolicySpecDocument>, Status> {
        let generations = &self.shared.generations;
        if !generations.bound() {
            return self.sourceless(Call::GetPolicySpec, &request);
        }
        self.admit(Call::GetPolicySpec, &request)?;
        generations
            .read(request.get_ref().generation)
            .await
            .map(Response::new)
            .map_err(unanswered)
    }

    /// `RollbackPolicySpec` by a controller with a spec source: the scope,
    /// the lead, then the source. Under a file, which no call writes, the
    /// answer is `read_only`, with the standing number and digest. Under a
    /// spec store the document of the generation the call names, one on
    /// file, is persisted and applied again as the next generation, one
    /// write at a time ([`Generations::roll_back_in_turn`]), and the call
    /// answers as an enforce does. A call that names no generation is
    /// refused by the field, and one that names a generation not on file is
    /// answered `NOT_FOUND`.
    async fn rollback_policy_spec(
        &self,
        request: Request<proto::RollbackRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        let generations = &self.shared.generations;
        if !generations.bound() {
            return self.sourceless(Call::RollbackPolicySpec, &request);
        }
        self.admit(Call::RollbackPolicySpec, &request)?;
        self.leading(Instant::now())?;
        if !generations.writable() {
            return Ok(Response::new(read_only(generations)));
        }
        let wanted = request.get_ref().generation;
        if wanted == 0 {
            return Err(Status::invalid_argument(
                "generation: a rollback names a generation on file",
            ));
        }
        let EnforceOutcome {
            standing,
            ran,
            needs,
        } = generations
            .roll_back_in_turn(wanted)
            .await
            .map_err(unanswered)?;
        Ok(Response::new(verdict(standing, ran, needs, false)))
    }

    /// Every scope, every replica, any caller: no principal is read, since
    /// the answer tells a caller where to go before it holds a session. One
    /// lease covers every model today, so a named model reads the
    /// controller's own lease.
    async fn leader(
        &self,
        _request: Request<proto::LeaderRequest>,
    ) -> Result<Response<proto::LeaderInfo>, Status> {
        Ok(Response::new(self.leader_info(Instant::now())))
    }

    async fn step_down(
        &self,
        request: Request<proto::StepDownRequest>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::StepDown, &request)
    }

    async fn watch_regime(
        &self,
        request: Request<proto::RegimeRequest>,
    ) -> Result<Response<Self::WatchRegimeStream>, Status> {
        self.unserved(Call::WatchRegime, &request)
    }

    async fn replicate(
        &self,
        request: Request<Streaming<proto::ReplicaFrame>>,
    ) -> Result<Response<Self::ReplicateStream>, Status> {
        let standby = self.admit(Call::Replicate, &request)?.subject.clone();
        let term = self.leading(Instant::now())?;
        self.shared
            .replication
            .replicate(&self.shared.floors, term, &standby, request.into_inner())
            .await
            .map(Response::new)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/service/tests/mod.rs"]
mod tests;
