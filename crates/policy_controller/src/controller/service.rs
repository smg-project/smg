//! The controller's gRPC service (design 5.2): the one implementation of
//! `smg.policy.v1.PolicyController`, mounted on every listener beside the
//! health service, one instance a listener, holding that listener's scopes.
//!
//! `Leader` is served, on every listener and by every replica, to a caller
//! with a principal or with none. `Session` is served to a gateway
//! ([`session`](super::session)). `GetPolicySpec` is served on the control
//! scope by a controller given a spec source. Every other call is refused:
//! `PERMISSION_DENIED` where the listener mounts none of the call's scopes
//! (the contract: the scope does not cover the RPC), `UNAUTHENTICATED`
//! where it does and the listener's layer placed no principal on the call
//! ([`principal`](super::principal)), `UNIMPLEMENTED` where one stands and
//! no step serves the call yet. The step that serves a call replaces that
//! call's one line here, behind [`Service::admit`], acts for the principal
//! it hands back and for no other peer, and adds what it reads to
//! [`Shared`], which [`Shared::new`] alone builds.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::watch;
use tonic::{service::Routes, Request, Response, Status, Streaming};

use super::{
    config::Scope,
    election::{split_identity, Standing},
    gateways::Gateways,
    generations::Generations,
    principal::{self, Layer, Principal},
    session, PROTOCOL_VERSION,
};
use crate::{
    catalog::Catalog,
    elector::{Elector, Fence},
    proto::{
        self,
        policy_controller_server::{PolicyController, PolicyControllerServer},
    },
    store::MemoryStore,
};

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
    pub(crate) standing: watch::Receiver<Standing>,
    /// The controller's store: what the calls feed and read, and the run
    /// loop's drivers with them.
    pub(crate) store: Arc<MemoryStore>,
    /// Who is present: the gateways a `Session` welcomed, over this fence
    /// and this store.
    pub(crate) gateways: Arc<Gateways>,
    /// The spec's generations over that store: the door a spec source
    /// applies through, and the document the spec calls read.
    pub(crate) generations: Arc<Generations>,
    /// How long a `Session` stream may stand with no `Hello`.
    pub(crate) hello_deadline: Duration,
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
    /// holds.
    pub(crate) fn new(
        controller_id: String,
        address: String,
        elector: &Elector,
        standing: watch::Receiver<Standing>,
        store: Arc<MemoryStore>,
        catalog: Arc<Catalog>,
    ) -> Self {
        let gateways = Arc::new(Gateways::new(
            session::GRACE_PERIOD,
            session::PRESENT_MAX,
            elector.fence(),
            Arc::clone(&store),
        ));
        Self {
            controller_id,
            address,
            identity: elector.identity().to_owned(),
            fence: elector.fence(),
            standing,
            generations: Arc::new(Generations::new(Arc::clone(&store), catalog)),
            store,
            gateways,
            hello_deadline: session::HELLO_DEADLINE,
        }
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
}

/// A call of the service a scope owns. `Leader` is not one: every scope
/// makes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Call {
    Session,
    Decide,
    WatchWorkers,
    PushLoad,
    PushKvEvents,
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
    /// Every call, in the contract's order.
    #[cfg(test)]
    pub(crate) const ALL: [Call; 16] = [
        Call::Session,
        Call::Decide,
        Call::WatchWorkers,
        Call::PushLoad,
        Call::PushKvEvents,
        Call::RegisterWorker,
        Call::PatchWorker,
        Call::DeregisterWorker,
        Call::GetSnapshot,
        Call::Explain,
        Call::ApplyPolicySpec,
        Call::GetPolicySpec,
        Call::RollbackPolicySpec,
        Call::StepDown,
        Call::WatchRegime,
        Call::Replicate,
    ];

    /// The call's name in the contract.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Call::Session => "Session",
            Call::Decide => "Decide",
            Call::WatchWorkers => "WatchWorkers",
            Call::PushLoad => "PushLoad",
            Call::PushKvEvents => "PushKvEvents",
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

    /// The scopes that make the call, by the contract's header: the
    /// snapshot is the operators' read and a standby's re-sync base.
    pub(crate) const fn scopes(self) -> &'static [Scope] {
        match self {
            Call::Session | Call::Decide | Call::WatchWorkers => &[Scope::Gateway],
            Call::PushLoad | Call::PushKvEvents => &[Scope::Ingestion],
            Call::GetSnapshot => &[Scope::Control, Scope::Replication],
            Call::RegisterWorker
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
        match layer {
            Layer::Off => Routes::new(PolicyControllerServer::new(self)),
            Layer::AsTold => Routes::new(PolicyControllerServer::with_interceptor(
                self,
                |request: Request<()>| Ok(principal::as_told(request)),
            )),
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
            "{} is not served by a controller started with no spec file",
            call.name()
        )))
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
        proto::LeaderInfo {
            controller_id,
            address,
            term,
            is_leader,
            // The sync state is a standby's and arrives with it; one lease
            // covers every model until a lease per model group exists.
            synced: false,
            sync: None,
            groups: Vec::new(),
            protocol_min: PROTOCOL_VERSION,
            protocol_max: PROTOCOL_VERSION,
        }
    }
}

#[tonic::async_trait]
impl PolicyController for Service {
    type SessionStream = session::Frames;
    type DecideStream = Unserved<proto::DecideAnswer>;
    type WatchWorkersStream = Unserved<proto::WorkerEvent>;
    type WatchRegimeStream = Unserved<proto::Regime>;
    type ReplicateStream = Unserved<proto::ReplicationBatch>;

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
        self.unserved(Call::Decide, &request)
    }

    async fn watch_workers(
        &self,
        request: Request<proto::WatchWorkersRequest>,
    ) -> Result<Response<Self::WatchWorkersStream>, Status> {
        self.unserved(Call::WatchWorkers, &request)
    }

    async fn push_load(
        &self,
        request: Request<Streaming<proto::LoadReport>>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::PushLoad, &request)
    }

    async fn push_kv_events(
        &self,
        request: Request<Streaming<proto::RelayKvFrame>>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::PushKvEvents, &request)
    }

    async fn register_worker(
        &self,
        request: Request<proto::RegisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::RegisterWorker, &request)
    }

    async fn patch_worker(
        &self,
        request: Request<proto::PatchWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::PatchWorker, &request)
    }

    async fn deregister_worker(
        &self,
        request: Request<proto::DeregisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::DeregisterWorker, &request)
    }

    async fn get_snapshot(
        &self,
        request: Request<proto::SnapshotRequest>,
    ) -> Result<Response<proto::Snapshot>, Status> {
        self.unserved(Call::GetSnapshot, &request)
    }

    async fn explain(
        &self,
        request: Request<proto::ExplainRequest>,
    ) -> Result<Response<proto::DecisionRecord>, Status> {
        self.unserved(Call::Explain, &request)
    }

    async fn apply_policy_spec(
        &self,
        request: Request<proto::ApplyRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        self.unserved(Call::ApplyPolicySpec, &request)
    }

    /// The standing generation's document as its source wrote it, with the
    /// number and the digest beside it: every replica answers from its own
    /// holder. A controller that was given no spec source serves no spec
    /// call.
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
            .document(request.get_ref().generation)
            .map(Response::new)
            .map_err(|not_held| Status::not_found(not_held.to_string()))
    }

    async fn rollback_policy_spec(
        &self,
        request: Request<proto::RollbackRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        self.unserved(Call::RollbackPolicySpec, &request)
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
        self.unserved(Call::Replicate, &request)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use http::uri::PathAndQuery;
    use tokio::{net::TcpListener, sync::watch, task::JoinSet};
    use tonic::{
        client::Grpc,
        transport::{server::TcpIncoming, Channel, Server},
        Code, Request,
    };
    use tonic_prost::ProstCodec;

    use super::{Call, Service, Shared};
    use crate::{
        catalog::Catalog,
        controller::{
            config::Scope,
            election::{self, Standing},
            generations::{tests::LEAST_LOAD, Format},
            principal::{
                tests::{stating, told, NO_PRINCIPAL},
                Layer,
            },
            PROTOCOL_VERSION,
        },
        elector::{
            ElectError, ElectionSettings, Elector, FileLease, LeaseBackend, LeaseRecord,
            MemoryLease, Tick, Written,
        },
        proto::{self, policy_controller_server::PolicyController},
        store::MemoryStore,
    };

    const ID: &str = "controller-0";
    const ADDRESS: &str = "10.0.0.7:30100";
    const IDENTITY: &str = "controller-0@10.0.0.7:30100";

    /// A candidate over a lease of its own, the election's writer and one
    /// listener's service over that candidate's fence.
    fn service(scopes: &[Scope]) -> (Elector, watch::Sender<Standing>, Service) {
        let elector = Elector::new(
            Box::new(MemoryLease::new()),
            IDENTITY,
            ElectionSettings::default(),
        )
        .expect("the default settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let shared = Arc::new(Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            stood,
            Arc::new(MemoryStore::new()),
            Arc::new(Catalog::builtin()),
        ));
        (elector, standing, Service::new(scopes.to_vec(), shared))
    }

    /// Polls the answer until `met`, bounded: every answer's term is at or
    /// above the one before it, `last` carried from poll to poll.
    fn answers_until(
        service: &Service,
        last: &mut u64,
        what: &str,
        met: impl Fn(&proto::LeaderInfo) -> bool,
    ) -> proto::LeaderInfo {
        let started = Instant::now();
        loop {
            let info = service.leader_info(Instant::now());
            assert!(
                info.term >= *last,
                "{what}: an answer under term {} after one under {last}",
                info.term
            );
            *last = info.term;
            if met(&info) {
                return info;
            }
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "{what} did not happen in 20 s: {info:?}"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// The promise at the answer, over the election's own thread: the lease
    /// file goes back to this run's first record while the run leads under a
    /// later term. Nobody is named under that later term while the record
    /// stands still for its duration, then this run leads above it: no
    /// answer's term is under the one before it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_store_put_back_to_this_runs_older_record_never_lowers_an_answers_term() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let elector = Elector::new(Box::new(FileLease::new(&path)), IDENTITY, settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let store = Arc::new(MemoryStore::new());
        let shared = Arc::new(Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            stood,
            Arc::clone(&store),
            Arc::new(Catalog::builtin()),
        ));
        let service = Service::new(vec![Scope::Gateway], shared);
        let mut running = election::start(elector, store, standing).expect("a thread");
        let mut last = 0;

        let first = answers_until(&service, &mut last, "the first take", |info| info.is_leader);
        assert_eq!(first, answer(ID, ADDRESS, 1, true));
        let kept = dir.path().join("kept");
        std::fs::copy(&path, &kept).expect("the copy");
        // The case moves the file under the store's own lock, as a writer of
        // the store does: no renewal stands between its compare and its
        // rename at a move, so none puts back what the move took.
        let lock = dir.path().join("lease.lock");
        let locked = |moved: &dyn Fn()| {
            let held = std::fs::File::create(&lock).expect("the lock file");
            held.lock().expect("the store's lock");
            moved();
        };

        // The record is gone, by no release: this run waits its own lost
        // record's duration out, as a candidate that saw it does, and takes
        // again under the next term.
        let removed = Instant::now();
        locked(&|| std::fs::remove_file(&path).expect("the removal"));
        answers_until(&service, &mut last, "the take after the removal", |info| {
            info.is_leader && info.term == 2
        });
        assert!(
            removed.elapsed() >= settings.duration,
            "the take came {:?} after the removal, sooner than the lost record's duration",
            removed.elapsed()
        );

        // The store goes back to the first record, whole, as a restore puts
        // it. The run's own record under an older term: nobody leads under
        // term two, and the take that follows is above it.
        locked(&|| std::fs::rename(&kept, &path).expect("the restore"));
        let fenced = answers_until(
            &service,
            &mut last,
            "the look at the restored record",
            |info| !info.is_leader,
        );
        assert_eq!(fenced, answer("", "", 2, false));
        let third = answers_until(&service, &mut last, "the take above", |info| info.is_leader);
        assert_eq!(third, answer(ID, ADDRESS, 3, true));

        running.stop();
        // Bounded like every wait here: a loop that ignored its stop fails
        // this case by name.
        tokio::time::timeout(Duration::from_secs(20), running.ended())
            .await
            .expect("the election thread ends within 20 s of its stop");
    }

    fn saw(holder: &str, term: u64) -> Standing {
        Standing {
            holder: holder.to_owned(),
            term,
            looked: true,
        }
    }

    /// An answer that names `controller_id` at `address` under `term`.
    fn answer(controller_id: &str, address: &str, term: u64, is_leader: bool) -> proto::LeaderInfo {
        proto::LeaderInfo {
            controller_id: controller_id.to_owned(),
            address: address.to_owned(),
            term,
            is_leader,
            synced: false,
            sync: None,
            groups: Vec::new(),
            protocol_min: PROTOCOL_VERSION,
            protocol_max: PROTOCOL_VERSION,
        }
    }

    /// The table a second time, copied by hand from the contract's header:
    /// a change of a call's scopes is made in both.
    #[test]
    fn every_call_keeps_the_scopes_copied_here_from_the_contracts_header() {
        let expected: [(&str, &[Scope]); 16] = [
            ("Session", &[Scope::Gateway]),
            ("Decide", &[Scope::Gateway]),
            ("WatchWorkers", &[Scope::Gateway]),
            ("PushLoad", &[Scope::Ingestion]),
            ("PushKvEvents", &[Scope::Ingestion]),
            ("RegisterWorker", &[Scope::Control]),
            ("PatchWorker", &[Scope::Control]),
            ("DeregisterWorker", &[Scope::Control]),
            ("GetSnapshot", &[Scope::Control, Scope::Replication]),
            ("Explain", &[Scope::Control]),
            ("ApplyPolicySpec", &[Scope::Control]),
            ("GetPolicySpec", &[Scope::Control]),
            ("RollbackPolicySpec", &[Scope::Control]),
            ("StepDown", &[Scope::Control]),
            ("WatchRegime", &[Scope::Control]),
            ("Replicate", &[Scope::Replication]),
        ];
        assert_eq!(Call::ALL.map(|call| (call.name(), call.scopes())), expected);
    }

    /// `message` as a call the listener's layer placed `subject` on.
    fn from<T>(subject: &str, message: T) -> Request<T> {
        let mut request = Request::new(message);
        request.extensions_mut().insert(told(subject));
        request
    }

    /// Admission hands back the principal the layer placed, on the scopes
    /// the listener mounts. Off them the call is denied whoever asks; on
    /// them a call with no principal is refused as unauthenticated.
    #[test]
    fn a_listener_admits_the_calls_of_its_scopes_from_a_caller_with_a_principal() {
        let listeners: [&[Scope]; 4] = [
            &[Scope::Gateway, Scope::Ingestion],
            &[Scope::Control],
            &[Scope::Replication],
            &Scope::ALL,
        ];
        for scopes in listeners {
            let (_elector, _standing, service) = service(scopes);
            for call in Call::ALL {
                let mounted = call.scopes().iter().any(|scope| scopes.contains(scope));
                let placed = from("gateway-a", ());
                match service.admit(call, &placed) {
                    Ok(principal) => {
                        assert!(mounted, "{} admitted on {scopes:?}", call.name());
                        assert_eq!(principal, &told("gateway-a"));
                    }
                    Err(status) => {
                        assert!(!mounted, "{} refused on {scopes:?}", call.name());
                        assert_eq!(status.code(), Code::PermissionDenied);
                        assert_eq!(
                            status.message(),
                            format!("{} is not a call of this listener's scopes", call.name())
                        );
                    }
                }
                let status = service
                    .admit(call, &Request::new(()))
                    .expect_err("no principal stands on the call");
                let code = if mounted {
                    Code::Unauthenticated
                } else {
                    Code::PermissionDenied
                };
                assert_eq!(status.code(), code, "{} on {scopes:?}", call.name());
                if mounted {
                    assert_eq!(status.message(), NO_PRINCIPAL);
                }
            }
        }
    }

    #[tokio::test]
    async fn an_unserved_call_is_denied_off_its_scope_and_unimplemented_to_a_principal_on_it() {
        let (_elector, _standing, gateway) = service(&[Scope::Gateway, Scope::Ingestion]);
        let (_elector, _standing, control) = service(&[Scope::Control]);
        let (_elector, _standing, replication) = service(&[Scope::Replication]);

        let read = || from("operator-a", proto::GetPolicySpecRequest::default());
        let denied = gateway.get_policy_spec(read()).await.expect_err("refused");
        assert_eq!(denied.code(), Code::PermissionDenied);
        let unserved = control.get_policy_spec(read()).await.expect_err("refused");
        assert_eq!(unserved.code(), Code::Unimplemented);
        assert_eq!(
            unserved.message(),
            "GetPolicySpec is not served by a controller started with no spec file"
        );
        // A caller no layer placed is refused before the call is named
        // unserved: no call body is reached without a principal.
        let unplaced = control
            .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
            .await
            .expect_err("refused");
        assert_eq!(unplaced.code(), Code::Unauthenticated);
        assert_eq!(unplaced.message(), NO_PRINCIPAL);

        // The snapshot is the operators' read and a standby's re-sync base.
        let snapshot = || from("standby-a", proto::SnapshotRequest::default());
        for (listener, code) in [
            (&gateway, Code::PermissionDenied),
            (&control, Code::Unimplemented),
            (&replication, Code::Unimplemented),
        ] {
            let status = listener
                .get_snapshot(snapshot())
                .await
                .expect_err("refused");
            assert_eq!(status.code(), code);
        }
    }

    /// The methods of the contract's service, by the contract's own text.
    fn methods() -> Vec<&'static str> {
        const CONTRACT: &str = include_str!("../../contracts/policy_controller.proto");
        CONTRACT
            .lines()
            .skip_while(|line| !line.starts_with("service PolicyController"))
            .take_while(|line| !line.starts_with('}'))
            .filter_map(|line| line.trim_start().strip_prefix("rpc "))
            .filter_map(|rest| rest.split('(').next())
            .collect()
    }

    /// Every method of the contract, called by its own name over a socket
    /// on a listener of each scope set, three ways: a name stated to a
    /// listener with no layer, a name stated to one that takes identities
    /// as told, and no name stated to that one. `Leader` is served every
    /// way; any other answers in its own name: denied off the listener's
    /// scopes, and on them unauthenticated to a caller the layer did not
    /// place. To the caller it placed, a call no step serves is
    /// unimplemented, and `Session` is reached: the empty message is no
    /// `Hello`. The names are read from the contract, so a method added
    /// there joins the case, and fails it until the table holds its call.
    #[tokio::test]
    async fn every_method_of_the_contract_answers_by_its_calls_scopes_and_its_callers_principal() {
        let methods = methods();
        assert_eq!(
            methods.len(),
            Call::ALL.len() + 1,
            "one call a method, and Leader: {methods:?}"
        );
        let listeners = [
            &[Scope::Gateway, Scope::Ingestion][..],
            &[Scope::Control],
            &[Scope::Replication],
        ];
        let ways = [
            (Layer::Off, true),
            (Layer::AsTold, true),
            (Layer::AsTold, false),
        ];
        for (scopes, (layer, stated)) in listeners
            .into_iter()
            .flat_map(|scopes| ways.map(|way| (scopes, way)))
        {
            let (_elector, _standing, service) = service(scopes);
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("a loopback port");
            let addr = listener.local_addr().expect("the bound address");
            // Aborted at its drop, at the end of this listener's round.
            let mut serving = JoinSet::new();
            serving.spawn(
                Server::builder()
                    .add_routes(service.routes(layer))
                    .serve_with_incoming(TcpIncoming::from(listener)),
            );
            let channel = Channel::from_shared(format!("http://{addr}"))
                .expect("a uri")
                .connect()
                .await
                .expect("the listener accepts");
            let mut client = Grpc::new(channel);
            for method in &methods {
                let path =
                    PathAndQuery::try_from(format!("/smg.policy.v1.PolicyController/{method}"))
                        .expect("a path");
                client.ready().await.expect("the channel is ready");
                // An empty message is every request's default.
                let request = if stated {
                    stating("gateway-a")
                } else {
                    Request::new(())
                };
                let answered = client
                    .unary(request, path, ProstCodec::<(), ()>::default())
                    .await;
                let round = format!("{method} on {scopes:?}, {layer:?}, a name stated: {stated}");
                if *method == "Leader" {
                    assert!(answered.is_ok(), "{round}: {answered:?}");
                    continue;
                }
                let call = Call::ALL.into_iter().find(|call| call.name() == *method);
                assert!(call.is_some(), "{method} is no call of the table");
                let mounted = call
                    .into_iter()
                    .flat_map(Call::scopes)
                    .any(|scope| scopes.contains(scope));
                let expected = if !mounted {
                    (
                        Code::PermissionDenied,
                        format!("{method} is not a call of this listener's scopes"),
                    )
                } else if layer == Layer::AsTold && stated && *method == "Session" {
                    (
                        Code::FailedPrecondition,
                        "no session stands on this stream: Hello is the first frame, a frame of no kind came first"
                            .to_owned(),
                    )
                } else if layer == Layer::AsTold && stated && *method == "GetPolicySpec" {
                    (
                        Code::Unimplemented,
                        format!("{method} is not served by a controller started with no spec file"),
                    )
                } else if layer == Layer::AsTold && stated {
                    (
                        Code::Unimplemented,
                        format!("{method} is not served by this build"),
                    )
                } else {
                    (Code::Unauthenticated, NO_PRINCIPAL.to_owned())
                };
                let status = answered.expect_err("no step serves the call");
                assert_eq!(
                    (status.code(), status.message().to_owned()),
                    expected,
                    "{round}"
                );
            }
        }
    }

    /// A lease whose every write takes `slow`.
    struct WritesSlowly {
        lease: MemoryLease,
        slow: Duration,
    }

    impl LeaseBackend for WritesSlowly {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
            thread::sleep(self.slow);
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    /// A take the election thread has not published yet: the fence holds
    /// the new term while the last look still names the holder before it.
    #[test]
    fn a_take_newer_than_the_last_look_names_nobody() {
        const OTHER: &str = "controller-1@10.0.0.8:30100";
        let settings = ElectionSettings {
            duration: Duration::from_millis(60),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(40),
        };
        let lease = MemoryLease::new();
        let mut other = Elector::new(Box::new(lease.clone()), OTHER, settings)
            .expect("the settings are in order");
        // This replica's writes outlive its renew deadline.
        let mut elector = Elector::new(
            Box::new(WritesSlowly {
                lease: lease.clone(),
                slow: settings.renew_deadline + Duration::from_millis(20),
            }),
            IDENTITY,
            settings,
        )
        .expect("the settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let service = Service::new(
            vec![Scope::Gateway],
            Arc::new(Shared::new(
                ID.to_owned(),
                ADDRESS.to_owned(),
                &elector,
                stood,
                Arc::new(MemoryStore::new()),
                Arc::new(Catalog::builtin()),
            )),
        );

        // The other replica holds the lease under term 1; this one's look
        // saw it and is published.
        assert_eq!(
            other.tick(Instant::now()).expect("the take"),
            Tick::Acquired { term: 1 }
        );
        assert_eq!(
            elector.tick(Instant::now()).expect("a look"),
            Tick::Following {
                holder: OTHER.to_owned(),
                term: 1
            }
        );
        standing.send_replace(saw(OTHER, 1));
        assert_eq!(
            service.leader_info(Instant::now()),
            answer("controller-1", "10.0.0.8:30100", 1, false)
        );

        // The other renews no more. This replica's take lands later than
        // the renew deadline after its step began: the fence holds term 2,
        // closed at birth. Until the look is published the answer names
        // nobody: the other never held term 2.
        thread::sleep(settings.duration + Duration::from_millis(20));
        assert_eq!(
            elector.tick(Instant::now()).expect("the take"),
            Tick::Acquired { term: 2 }
        );
        assert_eq!(
            service.leader_info(Instant::now()),
            answer("", "", 2, false)
        );

        // Published, the look names this replica behind its closed fence:
        // nobody still, under the same term.
        standing.send_replace(saw(IDENTITY, 2));
        assert_eq!(
            service.leader_info(Instant::now()),
            answer("", "", 2, false)
        );
    }

    #[test]
    fn a_replica_whose_fence_is_open_names_itself_under_the_fences_term() {
        let (mut elector, standing, service) = service(&[Scope::Gateway]);
        // Before any look: nobody, at no term.
        assert_eq!(
            service.leader_info(Instant::now()),
            answer("", "", 0, false)
        );

        // The lease held no record: the take opens no fence, and the answer
        // names nobody under the take's term until the first renewal a renew
        // deadline later. The instants are ahead of the clock: the wait is
        // dated by the later of the two.
        let took = Instant::now() + ElectionSettings::default().renew_period;
        assert_eq!(
            elector.tick(took).expect("the take"),
            Tick::Acquired { term: 1 }
        );
        standing.send_replace(saw(IDENTITY, 1));
        assert_eq!(service.leader_info(took), answer("", "", 1, false));
        let taken = took + ElectionSettings::default().renew_deadline;
        assert_eq!(
            elector.tick(taken).expect("the renewal"),
            Tick::Renewed { term: 1 }
        );
        assert_eq!(service.leader_info(taken), answer(ID, ADDRESS, 1, true));

        // Past the renew deadline the fence is closed: the record still
        // names this replica, the answer names nobody and keeps the term.
        let late = taken + ElectionSettings::default().renew_deadline + Duration::from_millis(1);
        assert_eq!(service.leader_info(late), answer("", "", 1, false));

        // The answer reads the fence, never the standing: a standing that
        // lags the take still answers as the leader.
        standing.send_replace(Standing::default());
        assert_eq!(service.leader_info(taken), answer(ID, ADDRESS, 1, true));
    }

    #[test]
    fn a_follower_names_the_holder_its_last_look_saw() {
        let (_elector, standing, service) = service(&[Scope::Control]);
        let now = Instant::now();

        standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
        assert_eq!(
            service.leader_info(now),
            answer("controller-1", "10.0.0.8:30100", 4, false)
        );

        // A holder that advertises no address is named by its id alone.
        standing.send_replace(saw("controller-9", 5));
        assert_eq!(
            service.leader_info(now),
            answer("controller-9", "", 5, false)
        );

        // A free lease, and a look that failed: nobody, under the highest
        // term seen.
        standing.send_replace(saw("", 5));
        assert_eq!(service.leader_info(now), answer("", "", 5, false));
        standing.send_replace(Standing {
            holder: String::new(),
            term: 5,
            looked: false,
        });
        assert_eq!(service.leader_info(now), answer("", "", 5, false));
    }

    #[tokio::test]
    async fn leader_answers_on_every_listener_and_for_every_model() {
        for scopes in [
            &[Scope::Gateway, Scope::Ingestion][..],
            &[Scope::Control],
            &[Scope::Replication],
        ] {
            let (_elector, standing, service) = service(scopes);
            standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
            for model in ["", "model-a"] {
                let asked = Request::new(proto::LeaderRequest {
                    model: model.to_owned(),
                });
                let info = service.leader(asked).await.expect("served").into_inner();
                assert_eq!(
                    info,
                    answer("controller-1", "10.0.0.8:30100", 4, false),
                    "{scopes:?} {model:?}"
                );
            }
        }
    }

    /// A controller with a spec source serves the read call on the control
    /// scope: the source's bytes under the standing number, to a caller with
    /// a principal. A number that does not stand is not found, by name, and
    /// the scope's rules stand as for every call.
    #[tokio::test]
    async fn the_read_call_answers_the_standing_document_once_a_source_applied_one() {
        let (_elector, _standing, control) = service(&[Scope::Control]);
        let (_elector, _standing, gateway) = service(&[Scope::Gateway, Scope::Ingestion]);
        for service in [&control, &gateway] {
            service
                .shared
                .generations
                .apply(LEAST_LOAD.into(), Format::Yaml)
                .expect("a valid document");
        }
        let read = |generation| from("operator-a", proto::GetPolicySpecRequest { generation });

        let answer = control
            .get_policy_spec(read(0))
            .await
            .expect("the standing document")
            .into_inner();
        assert_eq!(answer.generation, 1);
        assert_eq!(answer.document, LEAST_LOAD.as_bytes());
        assert_eq!(answer.format, "yaml");
        assert!(answer.digest.starts_with("sha256:"), "{}", answer.digest);

        let older = control.get_policy_spec(read(4)).await.expect_err("refused");
        assert_eq!(older.code(), Code::NotFound);
        assert_eq!(
            older.message(),
            "no document is held for generation 4: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
        );

        let unplaced = control
            .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
            .await
            .expect_err("refused");
        assert_eq!(unplaced.code(), Code::Unauthenticated);
        let denied = gateway.get_policy_spec(read(0)).await.expect_err("refused");
        assert_eq!(denied.code(), Code::PermissionDenied);
    }
}
