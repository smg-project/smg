//! The controller's gRPC service (design 5.2): the one implementation of
//! `smg.policy.v1.PolicyController`, mounted on every listener beside the
//! health service, one instance a listener, holding that listener's scopes.
//!
//! `Leader` is served, on every listener and by every replica. Every other
//! call is refused: `PERMISSION_DENIED` where the listener mounts none of
//! the call's scopes (the contract: the scope does not cover the RPC),
//! `UNIMPLEMENTED` where it does and no step serves the call yet. The step
//! that serves a call replaces that call's one line here, behind
//! [`Service::admit`], and adds what it reads to [`Shared`], which
//! [`Shared::new`] alone builds.

use std::{sync::Arc, time::Instant};

use tokio::sync::watch;
use tonic::{Request, Response, Status, Streaming};

use super::{
    config::Scope,
    election::{split_identity, Standing},
    PROTOCOL_VERSION,
};
use crate::{
    elector::{Elector, Fence},
    proto::{self, policy_controller_server::PolicyController},
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
}

impl Shared {
    /// The one place a `Shared` is built: a step that serves a call adds
    /// its state as a field and fills it here, from these arguments where
    /// it can, so the other builders do not change. The identity and the
    /// fence are `elector`'s, so no instance answers under an identity
    /// another fence guards.
    pub(crate) fn new(
        controller_id: String,
        address: String,
        elector: &Elector,
        standing: watch::Receiver<Standing>,
        store: Arc<MemoryStore>,
    ) -> Self {
        Self {
            controller_id,
            address,
            identity: elector.identity().to_owned(),
            fence: elector.fence(),
            standing,
            store,
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

    /// Admits `call` on this listener, or refuses it: a listener answers
    /// the calls of the scopes it mounts. The first line of every served
    /// call's body.
    pub(crate) fn admit(&self, call: Call) -> Result<(), Status> {
        if call
            .scopes()
            .iter()
            .any(|scope| self.scopes.contains(scope))
        {
            Ok(())
        } else {
            Err(Status::permission_denied(format!(
                "{} is not a call of this listener's scopes",
                call.name()
            )))
        }
    }

    /// The answer to a call no step serves yet, behind the listener's scope
    /// check.
    fn unserved<T>(&self, call: Call) -> Result<T, Status> {
        self.admit(call)?;
        Err(Status::unimplemented(format!(
            "{} is not served by this build",
            call.name()
        )))
    }

    /// The `Leader` call's answer at `now`. A replica whose fence is open
    /// names itself under the fence's term. Any other names the holder its
    /// last look at the lease saw, or nobody (an empty address: no leader
    /// known) when the lease is free, when that look failed, when it read a
    /// term under the highest this replica saw, when the record still names
    /// this replica behind a closed fence, and when this replica's own take
    /// is newer than that look: a holder is named under a term one write of
    /// the lease stored for it. The term is the highest this replica knows
    /// either way, so no answer of its falls in term.
    fn leader_info(&self, now: Instant) -> proto::LeaderInfo {
        let shared = &self.shared;
        let (controller_id, address, term, is_leader) = match shared.fence.leader_term(now) {
            Some(term) => (
                shared.controller_id.clone(),
                shared.address.clone(),
                term,
                true,
            ),
            None => {
                let standing = shared.standing.borrow();
                let fenced = shared.fence.term();
                let term = standing.term.max(fenced);
                if standing.holder.is_empty()
                    || standing.holder == shared.identity
                    || fenced > standing.term
                {
                    (String::new(), String::new(), term, false)
                } else {
                    let (id, address) = split_identity(&standing.holder);
                    (id.to_owned(), address.to_owned(), term, false)
                }
            }
        };
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
    type SessionStream = Unserved<proto::ControllerFrame>;
    type DecideStream = Unserved<proto::DecideAnswer>;
    type WatchWorkersStream = Unserved<proto::WorkerEvent>;
    type WatchRegimeStream = Unserved<proto::Regime>;
    type ReplicateStream = Unserved<proto::ReplicationBatch>;

    async fn session(
        &self,
        _request: Request<Streaming<proto::GatewayFrame>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        self.unserved(Call::Session)
    }

    async fn decide(
        &self,
        _request: Request<Streaming<proto::DecideRequest>>,
    ) -> Result<Response<Self::DecideStream>, Status> {
        self.unserved(Call::Decide)
    }

    async fn watch_workers(
        &self,
        _request: Request<proto::WatchWorkersRequest>,
    ) -> Result<Response<Self::WatchWorkersStream>, Status> {
        self.unserved(Call::WatchWorkers)
    }

    async fn push_load(
        &self,
        _request: Request<Streaming<proto::LoadReport>>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::PushLoad)
    }

    async fn push_kv_events(
        &self,
        _request: Request<Streaming<proto::RelayKvFrame>>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::PushKvEvents)
    }

    async fn register_worker(
        &self,
        _request: Request<proto::RegisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::RegisterWorker)
    }

    async fn patch_worker(
        &self,
        _request: Request<proto::PatchWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::PatchWorker)
    }

    async fn deregister_worker(
        &self,
        _request: Request<proto::DeregisterWorkerRequest>,
    ) -> Result<Response<proto::WorkerEvent>, Status> {
        self.unserved(Call::DeregisterWorker)
    }

    async fn get_snapshot(
        &self,
        _request: Request<proto::SnapshotRequest>,
    ) -> Result<Response<proto::Snapshot>, Status> {
        self.unserved(Call::GetSnapshot)
    }

    async fn explain(
        &self,
        _request: Request<proto::ExplainRequest>,
    ) -> Result<Response<proto::DecisionRecord>, Status> {
        self.unserved(Call::Explain)
    }

    async fn apply_policy_spec(
        &self,
        _request: Request<proto::ApplyRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        self.unserved(Call::ApplyPolicySpec)
    }

    async fn get_policy_spec(
        &self,
        _request: Request<proto::GetPolicySpecRequest>,
    ) -> Result<Response<proto::PolicySpecDocument>, Status> {
        self.unserved(Call::GetPolicySpec)
    }

    async fn rollback_policy_spec(
        &self,
        _request: Request<proto::RollbackRequest>,
    ) -> Result<Response<proto::ApplyResult>, Status> {
        self.unserved(Call::RollbackPolicySpec)
    }

    /// Every scope, every replica. One lease covers every model today, so a
    /// named model reads the controller's own lease.
    async fn leader(
        &self,
        _request: Request<proto::LeaderRequest>,
    ) -> Result<Response<proto::LeaderInfo>, Status> {
        Ok(Response::new(self.leader_info(Instant::now())))
    }

    async fn step_down(
        &self,
        _request: Request<proto::StepDownRequest>,
    ) -> Result<Response<proto::Ack>, Status> {
        self.unserved(Call::StepDown)
    }

    async fn watch_regime(
        &self,
        _request: Request<proto::RegimeRequest>,
    ) -> Result<Response<Self::WatchRegimeStream>, Status> {
        self.unserved(Call::WatchRegime)
    }

    async fn replicate(
        &self,
        _request: Request<Streaming<proto::ReplicaFrame>>,
    ) -> Result<Response<Self::ReplicateStream>, Status> {
        self.unserved(Call::Replicate)
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
        controller::{
            config::Scope,
            election::{self, Standing},
            PROTOCOL_VERSION,
        },
        elector::{
            ElectError, ElectionSettings, Elector, FileLease, LeaseBackend, LeaseRecord,
            MemoryLease, Tick, Written,
        },
        proto::{
            self,
            policy_controller_server::{PolicyController, PolicyControllerServer},
        },
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

        // The record is gone: this run takes again, under the next term.
        locked(&|| std::fs::remove_file(&path).expect("the removal"));
        answers_until(&service, &mut last, "the take after the removal", |info| {
            info.is_leader && info.term == 2
        });

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

    #[test]
    fn a_listener_admits_the_calls_of_the_scopes_it_mounts() {
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
                match service.admit(call) {
                    Ok(()) => assert!(mounted, "{} admitted on {scopes:?}", call.name()),
                    Err(status) => {
                        assert!(!mounted, "{} refused on {scopes:?}", call.name());
                        assert_eq!(status.code(), Code::PermissionDenied);
                        assert_eq!(
                            status.message(),
                            format!("{} is not a call of this listener's scopes", call.name())
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn an_unserved_call_is_denied_off_its_scope_and_unimplemented_on_it() {
        let (_elector, _standing, gateway) = service(&[Scope::Gateway, Scope::Ingestion]);
        let (_elector, _standing, control) = service(&[Scope::Control]);
        let (_elector, _standing, replication) = service(&[Scope::Replication]);

        let read = || Request::new(proto::GetPolicySpecRequest::default());
        let denied = gateway.get_policy_spec(read()).await.expect_err("refused");
        assert_eq!(denied.code(), Code::PermissionDenied);
        let unserved = control.get_policy_spec(read()).await.expect_err("refused");
        assert_eq!(unserved.code(), Code::Unimplemented);
        assert_eq!(
            unserved.message(),
            "GetPolicySpec is not served by this build"
        );

        // The snapshot is the operators' read and a standby's re-sync base.
        let snapshot = || Request::new(proto::SnapshotRequest::default());
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
    /// on a listener of each scope set. `Leader` is served; any other
    /// answers in its own name, denied off the listener's scopes and
    /// unimplemented on them. The names are read from the contract, so a
    /// method added there joins the case, and fails it until the table
    /// holds its call.
    #[tokio::test]
    async fn every_method_of_the_contract_answers_by_its_own_calls_scopes() {
        let methods = methods();
        assert_eq!(
            methods.len(),
            Call::ALL.len() + 1,
            "one call a method, and Leader: {methods:?}"
        );
        for scopes in [
            &[Scope::Gateway, Scope::Ingestion][..],
            &[Scope::Control],
            &[Scope::Replication],
        ] {
            let (_elector, _standing, service) = service(scopes);
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("a loopback port");
            let addr = listener.local_addr().expect("the bound address");
            // Aborted at its drop, at the end of this listener's round.
            let mut serving = JoinSet::new();
            serving.spawn(
                Server::builder()
                    .add_service(PolicyControllerServer::new(service))
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
                let answered = client
                    .unary(Request::new(()), path, ProstCodec::<(), ()>::default())
                    .await;
                if *method == "Leader" {
                    assert!(answered.is_ok(), "Leader on {scopes:?}: {answered:?}");
                    continue;
                }
                let call = Call::ALL.into_iter().find(|call| call.name() == *method);
                assert!(call.is_some(), "{method} is no call of the table");
                let mounted = call
                    .into_iter()
                    .flat_map(Call::scopes)
                    .any(|scope| scopes.contains(scope));
                let expected = if mounted {
                    (
                        Code::Unimplemented,
                        format!("{method} is not served by this build"),
                    )
                } else {
                    (
                        Code::PermissionDenied,
                        format!("{method} is not a call of this listener's scopes"),
                    )
                };
                let status = answered.expect_err("no step serves the call");
                assert_eq!(
                    (status.code(), status.message().to_owned()),
                    expected,
                    "{method} on {scopes:?}"
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

        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            Tick::Acquired { term: 1 }
        );
        standing.send_replace(saw(IDENTITY, 1));
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
}
