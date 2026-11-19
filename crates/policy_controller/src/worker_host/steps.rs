//! Registration and removal steps for a host that keeps a membership store.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use smg_worker::{
    identity::endpoint::Endpoint,
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    settings::{RegistrationSettings, UnhealthyRemovalScope},
    workflow::{
        pin, require_host, single_attempt, worker_registration_workflow, worker_removal_workflow,
        Pin, RegistrationGates, RemovalOrder, WorkerList, WorkerRegistrationWorkflowData,
        WorkerRemovalWorkflowData, WorkflowHost, CREATE_LOCAL_WORKER_STEP,
        FIND_WORKERS_TO_REMOVE_STEP, SETTLE_DRAINED_WORKERS_STEP,
    },
};
use tracing::{debug, warn};
use wfaas::{
    FailureAction, StepDefinition, StepExecutor, StepId, StepResult, WorkflowContext,
    WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::{facts, Membership, Objects, Refused, Remover, Wrote};
use crate::store::{Maker, Unmade};

/// The provider name used by the host's own discovery source.
pub const DISCOVERY: &str = "discovery";

/// Why a job's asker cannot make a registration.
#[derive(Debug, thiserror::Error)]
pub enum Unasked {
    /// Health checks ask for removals.
    #[error("health checks do not register workers")]
    Health,
    /// The store refused the maker's name or key.
    #[error(transparent)]
    Unmade(#[from] Unmade),
}

/// The maker a job registers as, using the address as written in its spec.
pub fn maker_of(asker: &WorkerJobAsker, address: &str) -> Result<Maker, Unasked> {
    match asker {
        WorkerJobAsker::Api | WorkerJobAsker::Startup => Ok(Maker::Control),
        WorkerJobAsker::Discovery => Ok(Maker::provider(DISCOVERY, address)?),
        WorkerJobAsker::Health => Err(Unasked::Health),
    }
}

/// Who asks for a job's removal at the door.
pub fn remover_of(asker: &WorkerJobAsker, scope: UnhealthyRemovalScope) -> Remover<'static> {
    match asker {
        WorkerJobAsker::Api | WorkerJobAsker::Startup => Remover::Maker(&Maker::Control),
        WorkerJobAsker::Discovery => Remover::Provider(DISCOVERY),
        WorkerJobAsker::Health => Remover::Health(scope),
    }
}

/// The step that writes the registration's created workers at the door.
pub const WRITE_WORKERS_AT_DOOR_STEP: &str = "write_workers_at_door";
/// The step that marks the selected workers' rows as draining.
pub const DRAIN_WORKERS_AT_DOOR_STEP: &str = "drain_workers_at_door";
/// The step that removes the drained workers' rows after they settle.
pub const REMOVE_WORKERS_AT_DOOR_STEP: &str = "remove_workers_at_door";

/// Write the workers the preceding registration steps created.
pub struct WriteWorkersAtDoor {
    door: Arc<dyn Membership>,
}

/// Mark the rows that still stand for the workers this removal names.
pub struct DrainWorkersAtDoor<G> {
    door: Arc<dyn Membership>,
    objects: Arc<Objects<G>>,
    scope: UnhealthyRemovalScope,
}

/// Remove the rows of the workers that drained and are still held.
pub struct RemoveWorkersAtDoor<G> {
    door: Arc<dyn Membership>,
    objects: Arc<Objects<G>>,
    scope: UnhealthyRemovalScope,
}

/// Build the worker registration using the host's settings and write the
/// created workers at the door. With an empty `behind`, the write follows
/// creation. Otherwise it waits for every step in `behind`.
///
/// The write is the only next step after a step in `behind`. A step writes
/// back the run's data whole, so parallel steps can overwrite each other.
pub fn registration_at_door<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn Membership>,
    behind: &[&'static str],
) -> WorkflowDefinition<WorkerRegistrationWorkflowData<H>> {
    let dependencies = if behind.is_empty() {
        &[CREATE_LOCAL_WORKER_STEP]
    } else {
        behind
    };
    worker_registration_workflow(settings, gates).add_step(
        StepDefinition::new(
            WRITE_WORKERS_AT_DOOR_STEP,
            "Write workers at the membership door",
            Arc::new(WriteWorkersAtDoor { door }),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(dependencies),
    )
}

/// Build a removal that selects workers, marks their rows draining, waits
/// for them to settle and removes their rows. The holder drops the objects
/// when it follows the store. The drain marks the row: routing reads the
/// row's status, and the holder mirrors it onto the held object where it
/// follows the store.
pub fn removal_at_door<H: WorkflowHost>(
    door: Arc<dyn Membership>,
    objects: Arc<Objects<H::Generation>>,
    scope: UnhealthyRemovalScope,
) -> WorkflowDefinition<WorkerRemovalWorkflowData<H>> {
    worker_removal_workflow(RemovalOrder::Door {
        drain: DRAIN_WORKERS_AT_DOOR_STEP,
    })
    .add_step(
        StepDefinition::new(
            DRAIN_WORKERS_AT_DOOR_STEP,
            "Drain workers at the membership door",
            Arc::new(DrainWorkersAtDoor {
                door: Arc::clone(&door),
                objects: Arc::clone(&objects),
                scope,
            }),
        )
        .with_timeout(Duration::from_secs(10))
        .with_retry(single_attempt())
        .depends_on(&[FIND_WORKERS_TO_REMOVE_STEP]),
    )
    .add_step(
        StepDefinition::new(
            REMOVE_WORKERS_AT_DOOR_STEP,
            "Remove workers at the membership door",
            Arc::new(RemoveWorkersAtDoor {
                door,
                objects,
                scope,
            }),
        )
        .with_timeout(Duration::from_secs(10))
        .with_retry(single_attempt())
        .depends_on(&[SETTLE_DRAINED_WORKERS_STEP]),
    )
}

const NOT_LEADING: &str = "the membership door does not lead";
const MOVED: &str = "the membership row moved";
const ABSENT: &str = "no row is registered under the URL";

fn failed(step: &str, message: impl Into<String>) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new(step),
        message: message.into(),
    }
}

fn registration_refusal(refusal: Refused) -> WorkflowError {
    let message = match refusal {
        Refused::NotMaker(why) => why,
        Refused::Past(past) => format!("{}: {}", past.fact, past.why),
        Refused::Absent => ABSENT.to_owned(),
        Refused::NotLeading(_) => NOT_LEADING.to_owned(),
        Refused::Moved => MOVED.to_owned(),
        Refused::OutOfScope => Refused::OUT_OF_SCOPE.to_owned(),
    };
    failed(WRITE_WORKERS_AT_DOOR_STEP, message)
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for WriteWorkersAtDoor {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        let data = &mut context.data;
        let workers = data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_owned()))?;
        let unsupported = match data.registration_mode {
            WorkerRegistrationMode::Upsert => None,
            WorkerRegistrationMode::CreateOnly => Some("CreateOnly"),
            WorkerRegistrationMode::ReplaceById { .. } => Some("ReplaceById"),
        };
        if let Some(mode) = unsupported {
            return Err(failed(
                WRITE_WORKERS_AT_DOOR_STEP,
                format!("{mode} is not supported at the membership door"),
            ));
        }
        let maker = maker_of(&data.asker, &data.config.url)
            .map_err(|why| failed(WRITE_WORKERS_AT_DOOR_STEP, why.to_string()))?;
        for worker in workers {
            self.door
                .found(&maker, worker.url(), &|standing| {
                    facts(worker.as_ref(), standing)
                })
                .map_err(registration_refusal)?;
        }
        data.workers = Some(WorkerList::from_workers(workers));
        Ok(StepResult::Success)
    }

    /// A step's error carries text: the two fixed messages name the refusals
    /// worth a second attempt.
    fn is_retryable(&self, error: &WorkflowError) -> bool {
        matches!(error, WorkflowError::StepFailed { message, .. } if message == NOT_LEADING || message == MOVED)
    }
}

fn skipped(url: &str, why: &str) {
    debug!(worker_url = %Endpoint::redacted_str(url), why, "Worker left alone");
}

fn removal_answer(
    step: &str,
    url: &str,
    answer: Result<Wrote, Refused>,
) -> WorkflowResult<Option<Wrote>> {
    match answer {
        Ok(wrote) => return Ok(Some(wrote)),
        Err(Refused::NotLeading(_)) => return Err(failed(step, NOT_LEADING)),
        Err(Refused::Moved) => skipped(url, MOVED),
        Err(Refused::Absent) => skipped(url, ABSENT),
        Err(Refused::NotMaker(why)) => {
            warn!(worker_url = %Endpoint::redacted_str(url), why, "Worker left alone");
        }
        Err(Refused::OutOfScope) => {
            warn!(worker_url = %Endpoint::redacted_str(url), why = Refused::OUT_OF_SCOPE, "Worker left alone");
        }
        Err(Refused::Past(past)) => {
            return Err(failed(step, format!("{}: {}", past.fact, past.why)))
        }
    }
    Ok(None)
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRemovalWorkflowData<H>>
    for DrainWorkersAtDoor<H::Generation>
{
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        let data = &mut context.data;
        let registry = require_host(data.host.as_ref())?.worker_registry();
        let workers = data
            .actual_workers_to_remove
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_owned()))?;
        let pins = data.config.expected_revisions.as_ref();
        let by = remover_of(&data.asker, self.scope);
        data.drained_workers.clear();
        for worker in workers {
            let url = worker.url();
            let pinned = pin(registry, pins, url);
            let (id, revision) = match pinned {
                Pin::Held { id, revision } => (id, revision),
                Pin::Absent => {
                    skipped(url, "no worker stands in the registry");
                    continue;
                }
                Pin::LeftAlone => {
                    skipped(url, "the removal did not name this worker");
                    continue;
                }
            };
            let Some(current) = registry
                .get(&id)
                .filter(|current| current.revision() == revision)
            else {
                skipped(url, "the worker moved after selection");
                continue;
            };
            let Some(seen) = self.objects.standing_of(&id, &current) else {
                skipped(url, "no row stands for this object");
                continue;
            };
            let answer = removal_answer(
                DRAIN_WORKERS_AT_DOOR_STEP,
                url,
                self.door.drained(by, url, seen),
            )?;
            match answer {
                Some(Wrote::Draining(_) | Wrote::Unmoved(_)) => data.drained_workers.push(current),
                None => {}
                Some(_) => {
                    return Err(failed(
                        DRAIN_WORKERS_AT_DOOR_STEP,
                        "the door did not answer a drain",
                    ))
                }
            }
        }
        Ok(StepResult::Success)
    }

    /// One attempt a removal write, as the registry's: no refusal is tried
    /// again here.
    fn is_retryable(&self, _: &WorkflowError) -> bool {
        false
    }
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRemovalWorkflowData<H>>
    for RemoveWorkersAtDoor<H::Generation>
{
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        let data = &mut context.data;
        let registry = require_host(data.host.as_ref())?.worker_registry();
        let by = remover_of(&data.asker, self.scope);
        data.removed_workers.clear();
        for worker in &data.drained_workers {
            let url = worker.url();
            let Some(id) = registry.get_id_by_url(url) else {
                skipped(url, "no worker stands in the registry");
                continue;
            };
            let Some(current) = registry
                .get(&id)
                .filter(|current| Arc::ptr_eq(current, worker))
            else {
                skipped(url, "the worker moved while draining");
                continue;
            };
            let Some(seen) = self.objects.standing_of(&id, &current) else {
                skipped(url, "no row stands for this object");
                continue;
            };
            let answer = removal_answer(
                REMOVE_WORKERS_AT_DOOR_STEP,
                url,
                self.door.gone(by, url, seen),
            )?;
            match answer {
                Some(Wrote::Removed(_)) => data.removed_workers.push(current),
                None => {}
                Some(_) => {
                    return Err(failed(
                        REMOVE_WORKERS_AT_DOOR_STEP,
                        "the door did not answer a removal",
                    ))
                }
            }
        }
        Ok(StepResult::Success)
    }

    /// One attempt a removal write, as the registry's: no refusal is tried
    /// again here.
    fn is_retryable(&self, _: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use async_trait::async_trait;
    use smg_worker::{
        connection::{
            engine::{EngineConnection, EngineConnector, ZmqLink},
            http_client::{WorkerHttpClientCache, WorkerHttpClientSettings},
        },
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        jobs::{WorkerJobAsker, WorkerRegistrationMode},
        model_card::ModelCard,
        registry::WorkerRegistry,
        settings::{RegistrationSettings, UnhealthyRemovalScope},
        worker::{RuntimeType, WorkerSpec},
        workflow::{
            RegistrationGates, WorkerRegistrationWorkflowData, WorkerRemovalRequest,
            WorkerRemovalWorkflowData, WorkflowHost,
        },
    };
    use tokio::time::{self, Instant};
    use wfaas::{
        BackoffStrategy, FailureAction, StepExecutor, StepId, StepResult, WorkflowContext,
        WorkflowEngine, WorkflowError, WorkflowId, WorkflowInstanceId, WorkflowStatus,
    };

    use super::*;
    use crate::{
        store::{Maker, MemoryStore, Unmade, WorkerFacts, WorkerStatus, WorkersStore},
        worker_host::{
            facts, Members, Membership, Objects, Refused, Remover, Standing, StoreDoor, Wrote,
        },
    };

    const URL: &str = "http://worker:8000";
    const SCOPE: UnhealthyRemovalScope = UnhealthyRemovalScope::DiscoveryOwned;
    type Registration = WorkerRegistrationWorkflowData<Host>;
    type Removal = WorkerRemovalWorkflowData<Host>;

    struct Host {
        registry: Arc<WorkerRegistry<Members>>,
        settings: RegistrationSettings,
        clients: WorkerHttpClientCache,
    }

    impl WorkflowHost for Host {
        type Generation = Members;
        type RegistrationNotes = ();

        fn worker_registry(&self) -> &Arc<WorkerRegistry<Members>> {
            &self.registry
        }

        fn worker_client_cache(&self) -> &WorkerHttpClientCache {
            &self.clients
        }

        fn registration_settings(&self) -> &RegistrationSettings {
            &self.settings
        }

        fn engine_connector(&self) -> Arc<dyn EngineConnector> {
            Arc::new(NoConnection)
        }
    }

    #[derive(Debug)]
    struct NoConnection;

    #[async_trait]
    impl EngineConnector for NoConnection {
        async fn connect_grpc(
            &self,
            _: &str,
            _: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            Err("no connection".to_owned())
        }

        async fn connect_zmq(&self, _: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            Err("no connection".to_owned())
        }

        async fn dial_grpc(&self, _: &str, _: Duration) -> Result<(), String> {
            Err("no connection".to_owned())
        }
    }

    struct Fixture {
        host: Arc<Host>,
        store: Arc<MemoryStore>,
        door: Arc<dyn Membership>,
        objects: Arc<Objects<Members>>,
    }

    impl Fixture {
        fn new() -> Self {
            let registry = Arc::new(WorkerRegistry::new());
            let objects = Arc::new(Objects::new(Arc::clone(&registry)));
            let store = Arc::new(MemoryStore::default());
            let door = Arc::new(StoreDoor::new(Arc::clone(&store)));
            let host = Arc::new(Host {
                registry,
                settings: RegistrationSettings {
                    upstream_http2: false,
                    health_check_timeout_secs: 5,
                    startup_timeout_secs: 67,
                    startup_delay_secs: 11,
                    startup_check_interval_secs: 7,
                    dp_aware: false,
                    model_path: None,
                    model_aliases: Default::default(),
                    retry: Default::default(),
                    retries_enabled: true,
                    circuit_breaker: Default::default(),
                    circuit_breaker_enabled: true,
                    overload_defaults: Default::default(),
                    health: Default::default(),
                    health_endpoint: "/health".to_owned(),
                    zmq_startup_timeout_secs: 3,
                    request_timeout_secs: 30,
                },
                clients: WorkerHttpClientCache::new(WorkerHttpClientSettings {
                    client_identity: None,
                    ca_certificates: Vec::new(),
                    upstream_http2: false,
                    request_timeout_secs: 30,
                    pool_idle_timeout_secs: 0,
                }),
            });
            Self {
                host,
                store,
                door,
                objects,
            }
        }

        fn registration(
            &self,
            asker: WorkerJobAsker,
            workers: Vec<Arc<dyn Worker>>,
        ) -> WorkflowContext<Registration> {
            let mut data = Registration::new(
                WorkerSpec::new(URL.to_owned()),
                WorkerRegistrationMode::Upsert,
                Arc::clone(&self.host),
                asker,
            );
            data.actual_workers = Some(workers);
            WorkflowContext::new(WorkflowInstanceId::new(), data)
        }

        fn removal(
            &self,
            asker: WorkerJobAsker,
            selected: Vec<Arc<dyn Worker>>,
        ) -> WorkflowContext<Removal> {
            let mut data = Removal {
                asker,
                config: WorkerRemovalRequest {
                    url: URL.to_owned(),
                    expected_revisions: None,
                },
                host: Some(Arc::clone(&self.host)),
                workers_to_remove: None,
                worker_urls: Vec::new(),
                affected_models: Default::default(),
                actual_workers_to_remove: None,
                drained_workers: Vec::new(),
                removed_workers: Vec::new(),
            };
            data.keep_only(selected);
            WorkflowContext::new(WorkflowInstanceId::new(), data)
        }

        fn seed(&self, maker: &Maker, worker: &dyn Worker) -> Result<Wrote, Refused> {
            let answer = self
                .door
                .found(maker, worker.url(), &|old| facts(worker, old))?;
            self.objects.follow(&*self.store);
            Ok(answer)
        }

        fn write(&self) -> WriteWorkersAtDoor {
            WriteWorkersAtDoor {
                door: Arc::clone(&self.door),
            }
        }

        fn drain(&self) -> DrainWorkersAtDoor<Members> {
            DrainWorkersAtDoor {
                door: Arc::clone(&self.door),
                objects: Arc::clone(&self.objects),
                scope: SCOPE,
            }
        }

        fn remove(&self) -> RemoveWorkersAtDoor<Members> {
            RemoveWorkersAtDoor {
                door: Arc::clone(&self.door),
                objects: Arc::clone(&self.objects),
                scope: SCOPE,
            }
        }
    }

    fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            models: vec![ModelCard::new(model)].into(),
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..WorkerSpec::new(url.to_owned())
        };
        Arc::new(<BasicWorkerBuilder>::from_spec(spec).build())
    }

    #[test]
    fn an_asker_registers_as_its_maker() {
        use WorkerJobAsker::{Api, Discovery, Health, Startup};

        for asker in [Api, Startup] {
            assert_eq!(
                maker_of(&asker, URL).expect("a control maker"),
                Maker::Control
            );
            assert!(matches!(
                remover_of(&asker, SCOPE),
                Remover::Maker(Maker::Control)
            ));
        }
        assert_eq!(DISCOVERY, "discovery");
        let discovered = Maker::provider("discovery", URL).expect("a provider and key");
        assert_eq!(
            maker_of(&Discovery, URL).expect("the discovery maker"),
            discovered
        );
        assert!(matches!(
            remover_of(&Discovery, SCOPE),
            Remover::Provider("discovery")
        ));
        assert!(matches!(maker_of(&Health, URL), Err(Unasked::Health)));
        assert!(matches!(
            maker_of(&Discovery, ""),
            Err(Unasked::Unmade(Unmade::Key(_)))
        ));
        for scope in [
            UnhealthyRemovalScope::Disabled,
            SCOPE,
            UnhealthyRemovalScope::AllLocal,
        ] {
            assert!(matches!(remover_of(&Health, scope), Remover::Health(found) if found == scope));
        }
    }

    #[tokio::test]
    async fn a_registration_writes_one_row_a_worker_under_its_askers_maker() {
        let f = Fixture::new();
        let urls = ["http://worker:8000@0", "http://worker:8000@1"];
        let workers = urls.map(|url| worker(url, "model"));
        let mut cx = f.registration(WorkerJobAsker::Discovery, workers.to_vec());
        assert_eq!(
            f.write().execute(&mut cx).await.expect("written"),
            StepResult::Success
        );

        let maker = Maker::provider("discovery", URL).expect("the spec's address is the key");
        for (url, worker) in urls.into_iter().zip(workers) {
            let row = f.store.by_key(url).expect("the row for this worker");
            let expected = WorkerFacts {
                incarnation: row.facts.incarnation,
                ..facts(worker.as_ref(), None)
            };
            assert_eq!(row.facts, expected);
            assert_eq!(row.maker, maker);
            assert!(f.host.registry.get_by_url(url).is_none());
        }
        assert_eq!(f.store.epoch(), 2);
        assert_eq!(cx.data.workers.expect("the written URLs").worker_urls, urls);
    }

    #[tokio::test]
    async fn a_registration_of_equal_facts_writes_nothing_and_succeeds() {
        let f = Fixture::new();
        let worker = worker(URL, "model");
        f.seed(&Maker::Control, worker.as_ref())
            .expect("registered");
        let standing = f.store.by_key(URL);
        let epoch = f.store.epoch();

        let mut cx = f.registration(WorkerJobAsker::Startup, vec![worker]);
        assert_eq!(
            f.write().execute(&mut cx).await.expect("unchanged"),
            StepResult::Success
        );
        assert_eq!(f.store.by_key(URL), standing);
        assert_eq!(f.store.epoch(), epoch);
        assert_eq!(
            cx.data.workers.expect("the written URLs").worker_urls,
            [URL]
        );
    }

    #[tokio::test]
    async fn a_row_already_draining_counts_as_drained_once_and_is_taken() {
        let f = Fixture::new();
        let worker = worker(URL, "model");
        f.seed(&Maker::Control, worker.as_ref())
            .expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);
        f.drain().execute(&mut removal).await.expect("first drain");
        let epoch = f.store.epoch();

        f.drain()
            .execute(&mut removal)
            .await
            .expect("already draining");
        assert_eq!(removal.data.drained_workers.len(), 1);
        assert!(Arc::ptr_eq(&removal.data.drained_workers[0], &live));
        assert_eq!(f.store.epoch(), epoch);
        f.remove().execute(&mut removal).await.expect("removed");
        assert!(f.store.by_key(URL).is_none());
    }

    /// A source that says gone and then found inside the settle window: the
    /// registration is another one, and the removal leaves its row.
    #[tokio::test]
    async fn a_worker_found_again_while_its_removal_settles_is_registered_anew_and_stays() {
        let f = Fixture::new();
        let worker = worker(URL, "model");
        f.seed(&Maker::Control, worker.as_ref())
            .expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let mut removal = f.removal(WorkerJobAsker::Api, vec![live]);
        f.drain().execute(&mut removal).await.expect("drained");
        let draining = f.store.by_key(URL).expect("the draining row");
        assert_eq!(draining.status, WorkerStatus::Draining);

        // Equal facts, found again: the URL's next life, out of the drain.
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![worker]);
        assert_eq!(
            f.write().execute(&mut cx).await.expect("registered anew"),
            StepResult::Success
        );
        let anew = f.store.by_key(URL).expect("the new row");
        assert_ne!(anew.id, draining.id);
        assert_ne!(anew.status, WorkerStatus::Draining);
        assert_eq!(
            cx.data.workers.expect("the written URLs").worker_urls,
            [URL]
        );

        // The removal's guard names the row that left: the new one stays.
        f.remove().execute(&mut removal).await.expect("skipped");
        assert!(removal.data.removed_workers.is_empty());
        assert_eq!(f.store.by_key(URL), Some(anew));
    }

    #[tokio::test]
    async fn another_makers_row_fails_a_registration_with_the_stores_sentence() {
        let f = Fixture::new();
        let occupied = worker(URL, "model");
        f.seed(&Maker::Control, occupied.as_ref())
            .expect("registered");
        let before = f.store.by_key(URL);
        let available = "http://available:8000";
        let mut cx = f.registration(
            WorkerJobAsker::Discovery,
            vec![worker(available, "model"), occupied],
        );
        let write = f.write();
        let error = write.execute(&mut cx).await.expect_err("another maker");
        assert_eq!(
            error.to_string(),
            "Step failed: write_workers_at_door - the row was made by a control call: its maker alone changes it"
        );
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(f.store.by_key(URL), before);
        assert!(f.store.by_key(available).is_some());
        assert!(cx.data.workers.is_none());
    }

    #[tokio::test]
    async fn a_fact_past_its_bound_fails_a_registration_by_its_name() {
        let f = Fixture::new();
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![worker(URL, &"m".repeat(257))]);
        let write = f.write();
        let error = write
            .execute(&mut cx)
            .await
            .expect_err("model id is too long");
        assert!(
            matches!(&error, WorkflowError::StepFailed { step_id, message }
            if step_id == &StepId::new("write_workers_at_door") && message.starts_with("models"))
        );
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(f.store.epoch(), 0);
    }

    struct Shut(Refused);

    impl Membership for Shut {
        fn found(
            &self,
            _: &Maker,
            _: &str,
            _: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }

        fn drained(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }

        fn gone(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }
    }

    #[tokio::test]
    async fn a_door_that_does_not_lead_fails_a_removal_step_with_no_second_attempt() {
        let f = Fixture::new();
        let built = worker(URL, "model");
        f.seed(&Maker::Control, built.as_ref()).expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);

        let door: Arc<dyn Membership> = Arc::new(Shut(Refused::NotLeading("follower".to_owned())));
        let drain = DrainWorkersAtDoor {
            door: Arc::clone(&door),
            ..f.drain()
        };
        let error = drain.execute(&mut removal).await.expect_err("a follower");
        assert!(!StepExecutor::<Removal>::is_retryable(&drain, &error));
        assert!(removal.data.drained_workers.is_empty());

        removal.data.drained_workers.push(live);
        let remove = RemoveWorkersAtDoor { door, ..f.remove() };
        let error = remove.execute(&mut removal).await.expect_err("a follower");
        assert!(!StepExecutor::<Removal>::is_retryable(&remove, &error));
        assert!(removal.data.removed_workers.is_empty());
    }

    #[tokio::test]
    async fn a_registration_retries_only_when_the_door_or_row_moved() {
        for (refusal, retryable) in [
            (Refused::NotLeading("follower".to_owned()), true),
            (Refused::Moved, true),
            (Refused::Absent, false),
            (Refused::OutOfScope, false),
        ] {
            let f = Fixture::new();
            let mut cx = f.registration(WorkerJobAsker::Api, vec![worker(URL, "model")]);
            let write = WriteWorkersAtDoor {
                door: Arc::new(Shut(refusal)),
            };
            let error = write.execute(&mut cx).await.expect_err("refused");
            assert!(matches!(&error, WorkflowError::StepFailed { step_id, .. }
                if step_id == &StepId::new("write_workers_at_door")));
            assert_eq!(
                StepExecutor::<Registration>::is_retryable(&write, &error),
                retryable
            );
        }
    }

    #[tokio::test]
    async fn a_replacement_by_id_is_refused_by_name() {
        let f = Fixture::new();
        let mut cx = f.registration(WorkerJobAsker::Api, vec![worker(URL, "model")]);
        let write = f.write();
        for (mode, name) in [
            (
                WorkerRegistrationMode::ReplaceById {
                    worker_id: "id".to_owned(),
                    expected_revision: 1,
                },
                "ReplaceById",
            ),
            (WorkerRegistrationMode::CreateOnly, "CreateOnly"),
        ] {
            cx.data.registration_mode = mode;
            let error = write
                .execute(&mut cx)
                .await
                .expect_err("only Upsert is supported");
            assert!(
                matches!(&error, WorkflowError::StepFailed { step_id, message }
                if step_id == &StepId::new("write_workers_at_door") && message.contains(name))
            );
            assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        }
        assert_eq!(f.store.epoch(), 0);
    }

    #[tokio::test]
    async fn a_registration_needs_created_workers_and_an_asker_that_registers() {
        let f = Fixture::new();
        let mut cx = f.registration(WorkerJobAsker::Health, vec![worker(URL, "model")]);
        let write = f.write();
        let error = write
            .execute(&mut cx)
            .await
            .expect_err("health cannot register");
        assert!(matches!(&error, WorkflowError::StepFailed { step_id, .. }
            if step_id == &StepId::new("write_workers_at_door")));
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        cx.data.actual_workers = None;
        let error = write
            .execute(&mut cx)
            .await
            .expect_err("nothing was created");
        assert!(matches!(&error, WorkflowError::ContextValueNotFound(name) if name == "workers"));
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(f.store.epoch(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_removal_marks_its_rows_draining_and_takes_them_behind_the_settle() {
        let f = Fixture::new();
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let workflow = removal_at_door::<Host>(Arc::clone(&f.door), Arc::clone(&f.objects), SCOPE);
        let engine = WorkflowEngine::new();
        engine
            .register_workflow(workflow)
            .expect("the removal definition");
        let data = f.removal(WorkerJobAsker::Api, Vec::new()).data;

        let at = Instant::now();
        let run = engine
            .start_workflow(WorkflowId::new("worker_removal"), data)
            .await
            .expect("started");
        time::sleep(Duration::from_secs(4)).await;
        assert_eq!(
            f.store.by_key(URL).expect("drain is settling").status,
            WorkerStatus::Draining
        );
        let timeout = Duration::from_secs(4);
        let state = time::timeout(
            timeout,
            engine.wait_for_terminal_state(run, "removal", timeout),
        )
        .await
        .expect("the settle window ended")
        .expect("the final removal state");
        assert_eq!(state.status, WorkflowStatus::Completed);
        assert!(at.elapsed() >= Duration::from_secs(5));
        let data = state.context.data;
        for workers in [&data.drained_workers, &data.removed_workers] {
            assert_eq!(workers.len(), 1);
            assert!(Arc::ptr_eq(&workers[0], &live));
        }
        assert!(f.store.by_key(URL).is_none());
        assert!(f.host.registry.get_by_url(URL).is_some());
        f.objects.follow(&*f.store);
        assert!(f.host.registry.get_by_url(URL).is_none());
    }

    #[tokio::test]
    async fn a_removal_leaves_a_worker_its_pins_do_not_name() {
        let f = Fixture::new();
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let row = f.store.by_key(URL);
        let mut cx = f.removal(WorkerJobAsker::Api, vec![live]);
        cx.data.config.expected_revisions = Some(HashMap::new());
        assert_eq!(
            f.drain().execute(&mut cx).await.expect("skipped"),
            StepResult::Success
        );
        assert!(cx.data.drained_workers.is_empty());
        assert_eq!(f.store.by_key(URL), row);
    }

    #[tokio::test]
    async fn a_removal_pinned_at_another_revision_of_the_object_leaves_the_row() {
        let f = Fixture::new();
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        let id = f.host.registry.get_id_by_url(URL).expect("the object id");
        let live = f.host.registry.get(&id).expect("the held worker");
        let old_pin = Some(live.revision() + 1);
        let row = f.store.by_key(URL);
        let mut cx = f.removal(WorkerJobAsker::Api, vec![live]);
        cx.data.config.expected_revisions = Some([(id.as_str().to_owned(), old_pin)].into());
        assert_eq!(
            f.drain().execute(&mut cx).await.expect("skipped"),
            StepResult::Success
        );
        assert!(cx.data.drained_workers.is_empty());
        assert_eq!(f.store.by_key(URL), row);
    }

    #[tokio::test]
    async fn a_removal_leaves_workers_without_a_row() {
        let f = Fixture::new();
        let live = worker(URL, "model");
        f.host
            .registry
            .register(Arc::clone(&live))
            .expect("an object without a row");
        let door: Arc<dyn Membership> = Arc::new(Shut(Refused::NotLeading("follower".to_owned())));
        let drain = DrainWorkersAtDoor {
            door: Arc::clone(&door),
            ..f.drain()
        };
        let remove = RemoveWorkersAtDoor { door, ..f.remove() };
        let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);
        assert_eq!(
            drain.execute(&mut removal).await.expect("no standing"),
            StepResult::Success
        );
        assert!(removal.data.drained_workers.is_empty());
        removal.data.drained_workers.push(live);
        assert_eq!(
            remove.execute(&mut removal).await.expect("no standing"),
            StepResult::Success
        );
        assert!(removal.data.removed_workers.is_empty());
        assert_eq!(f.store.epoch(), 0);
    }

    #[cfg(feature = "controller")]
    #[test]
    fn a_removal_skip_logs_one_reason_and_masks_the_address() {
        use tracing::{
            subscriber::{self, NoSubscriber},
            Dispatch, Level,
        };

        let log = tempfile::NamedTempFile::new().expect("a file for this case");
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_writer(log.reopen().expect("the log writer"))
            .finish();
        let _beside = Dispatch::new(NoSubscriber::default());
        let _kept = subscriber::set_default(subscriber);

        let url = "http://user:secret@worker:8000/p?token=hidden";
        let answers = [
            (Refused::Moved, "DEBUG", "the membership row moved"),
            (
                Refused::Absent,
                "DEBUG",
                "no row is registered under the URL",
            ),
            (
                Refused::NotMaker("another maker's row".to_owned()),
                "WARN",
                "another maker's row",
            ),
            (
                Refused::OutOfScope,
                "WARN",
                "the row's maker is outside the health removal scope",
            ),
        ];
        for (refusal, _, _) in &answers {
            assert_eq!(
                removal_answer("drain_workers_at_door", url, Err(refusal.clone()))
                    .expect("skipped"),
                None
            );
        }
        let log = std::fs::read_to_string(log.path()).expect("the log text");
        assert_eq!(log.lines().count(), answers.len(), "{log}");
        for (line, (_, level, reason)) in log.lines().zip(answers) {
            assert!(line.contains(level) && line.contains(reason), "{line}");
            assert!(line.contains("http://***@worker:8000/p"), "{line}");
        }
        for unmasked in ["user", "secret", "hidden"] {
            assert!(!log.contains(unmasked), "{log}");
        }
    }

    /// A removal leaves a row its guard does not name: one that moved under
    /// the held object, and another maker's.
    #[tokio::test]
    async fn a_removal_leaves_a_row_its_guard_does_not_name() {
        type Setup = fn(&Fixture) -> Arc<dyn Worker>;
        let moved: Setup = |f| {
            let built = worker(URL, "model");
            f.seed(&Maker::Control, built.as_ref()).expect("registered");
            let live = f.host.registry.get_by_url(URL).expect("the held worker");
            f.door
                .found(&Maker::Control, URL, &|old| WorkerFacts {
                    priority: 5,
                    ..facts(built.as_ref(), old)
                })
                .expect("the row moved in place");
            live
        };
        let anothers: Setup = |f| {
            let maker = Maker::provider("discovery", URL).expect("a provider's maker");
            f.seed(&maker, worker(URL, "model").as_ref())
                .expect("registered");
            f.host.registry.get_by_url(URL).expect("the held worker")
        };
        for setup in [moved, anothers] {
            let f = Fixture::new();
            let live = setup(&f);
            let row = f.store.by_key(URL);
            let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);
            assert_eq!(
                f.drain().execute(&mut removal).await.expect("skipped"),
                StepResult::Success
            );
            assert!(removal.data.drained_workers.is_empty());
            removal.data.drained_workers.push(live);
            assert_eq!(
                f.remove().execute(&mut removal).await.expect("skipped"),
                StepResult::Success
            );
            assert!(removal.data.removed_workers.is_empty());
            assert_eq!(f.store.by_key(URL), row);
        }
    }

    #[tokio::test]
    async fn a_worker_registered_again_while_it_drains_is_not_taken() {
        let f = Fixture::new();
        let built = worker(URL, "model");
        f.seed(&Maker::Control, built.as_ref()).expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        let mut removal = f.removal(WorkerJobAsker::Api, vec![live]);
        f.drain().execute(&mut removal).await.expect("drained");
        assert_eq!(removal.data.drained_workers.len(), 1);

        f.door
            .found(&Maker::Control, URL, &|old| WorkerFacts {
                priority: 5,
                ..facts(built.as_ref(), old)
            })
            .expect("registered again");
        f.objects.follow(&*f.store);
        let replacement = f
            .host
            .registry
            .get_by_url(URL)
            .expect("the replacement at the URL");
        assert!(!Arc::ptr_eq(&replacement, &removal.data.drained_workers[0]));
        let row = f.store.by_key(URL);
        assert_eq!(
            f.remove().execute(&mut removal).await.expect("skipped"),
            StepResult::Success
        );
        assert!(removal.data.removed_workers.is_empty());
        assert_eq!(f.store.by_key(URL), row);
        // Other facts, found again: the row is out of the drain as well.
        let status = row.map(|row| row.status);
        assert_ne!(status, Some(WorkerStatus::Draining));
    }

    #[test]
    fn the_lines_at_the_door_order_their_steps() {
        let f = Fixture::new();
        for (behind, dependencies) in [
            (&[][..], &["create_local_worker"][..]),
            (
                &["check_local_worker", "create_external_workers"][..],
                &["check_local_worker", "create_external_workers"][..],
            ),
        ] {
            let workflow = registration_at_door::<Host>(
                f.host.registration_settings(),
                RegistrationGates::default(),
                Arc::clone(&f.door),
                behind,
            );
            assert_eq!(workflow.id, WorkflowId::new("worker_registration"));
            let worker_steps = [
                ("detect_connection_mode", None),
                ("detect_backend", Some("detect_connection_mode")),
                ("discover_metadata", Some("detect_backend")),
                ("discover_dp_info", Some("discover_metadata")),
                ("create_local_worker", Some("discover_dp_info")),
            ];
            assert_eq!(workflow.steps.len(), worker_steps.len() + 1);
            for (step, (id, dependency)) in workflow.steps.iter().zip(worker_steps) {
                assert_eq!(step.id, StepId::new(id));
                assert_eq!(
                    step.depends_on,
                    dependency.map(StepId::new).into_iter().collect::<Vec<_>>()
                );
                assert!(step.depends_on_any.is_empty());
            }
            let detection = &workflow.steps[0];
            assert_eq!(detection.delay, Some(Duration::from_secs(11)));
            assert_eq!(detection.timeout, Some(Duration::from_secs(67)));
            assert!(
                matches!(&detection.retry_policy, Some(policy) if policy.max_attempts == 9
                && matches!(policy.backoff, BackoffStrategy::Fixed(value) if value == Duration::from_secs(7)))
            );
            let write = workflow.steps.last().expect("the write step");
            assert_eq!(write.id, StepId::new("write_workers_at_door"));
            assert_eq!(
                write.depends_on,
                dependencies
                    .iter()
                    .map(|id| StepId::new(*id))
                    .collect::<Vec<_>>()
            );
            assert!(write.depends_on_any.is_empty());
            assert_eq!(write.timeout, Some(Duration::from_secs(5)));
            assert!(matches!(write.on_failure, FailureAction::FailWorkflow));
            assert!(write.retry_policy.is_none());
            let retry = workflow.default_retry_policy;
            assert_eq!(retry.max_attempts, 3);
            assert!(
                matches!(retry.backoff, BackoffStrategy::Exponential { base, max }
                if base == Duration::from_secs(1) && max == Duration::from_secs(30))
            );
        }

        let gated = registration_at_door::<Host>(
            f.host.registration_settings(),
            RegistrationGates {
                detection: Some("classify_worker"),
                creation: Some("prepare_worker"),
            },
            Arc::clone(&f.door),
            &[],
        );
        assert_eq!(gated.steps[0].depends_on, [StepId::new("classify_worker")]);
        assert_eq!(gated.steps[0].delay, None);
        assert_eq!(gated.steps[4].depends_on, [StepId::new("prepare_worker")]);

        let mut workflow = removal_at_door::<Host>(f.door, f.objects, SCOPE);
        assert_eq!(workflow.id, WorkflowId::new("worker_removal"));
        assert!(workflow.validate().is_ok());
        let edges = [
            ("find_workers_to_remove", None),
            ("settle_drained_workers", Some("drain_workers_at_door")),
            ("drain_workers_at_door", Some("find_workers_to_remove")),
            ("remove_workers_at_door", Some("settle_drained_workers")),
        ];
        assert_eq!(workflow.steps.len(), edges.len());
        for (step, (id, dependency)) in workflow.steps.iter().zip(edges) {
            assert_eq!(step.id, StepId::new(id));
            assert_eq!(
                step.depends_on,
                dependency.map(StepId::new).into_iter().collect::<Vec<_>>()
            );
            assert!(step.depends_on_any.is_empty());
            // The two writes at the door are this module's: ten seconds and
            // one attempt each. The other two steps are the worker crate's.
            if id.ends_with("_at_door") {
                assert!(matches!(step.on_failure, FailureAction::FailWorkflow));
                assert_eq!(step.timeout, Some(Duration::from_secs(10)));
                let retry = step.retry_policy.as_ref().expect("one attempt");
                assert_eq!(retry.max_attempts, 1);
                assert!(matches!(retry.backoff, BackoffStrategy::Fixed(value) if value.is_zero()));
            }
        }
    }
}
