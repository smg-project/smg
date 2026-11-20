//! Registration and removal steps for a host that keeps a membership store.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use smg_worker::{
    identity::{basic::Worker, endpoint::Endpoint},
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
    settings::{RegistrationSettings, UnhealthyRemovalScope},
    workflow::{
        pin, require_host, single_attempt, worker_registration_workflow, worker_removal_workflow,
        Pin, RegistrationGates, RemovalOrder, WorkerKind, WorkerList,
        WorkerRegistrationWorkflowData, WorkerRemovalWorkflowData, WorkflowHost,
        CREATE_LOCAL_WORKER_STEP, FIND_WORKERS_TO_REMOVE_STEP, SETTLE_DRAINED_WORKERS_STEP,
    },
};
use tracing::{debug, warn};
use wfaas::{
    FailureAction, StepDefinition, StepExecutor, StepId, StepResult, WorkflowContext,
    WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::{facts, Expect, Membership, Objects, Refused, Remover, Standing, Wrote};
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
pub struct WriteWorkersAtDoor<G> {
    door: Arc<dyn Membership>,
    objects: Arc<Objects<G>>,
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
///
/// The write is a step of the local kind, as the creation is: it skips a
/// registration of another kind, whose workers the host's own steps write.
///
/// A job's mode is what the write expects under each URL. An upsert
/// expects anything and a create-only job expects nothing. A replacement
/// by id expects the row behind the registry's worker of that id, read
/// through `objects` at the revision the job names.
pub fn registration_at_door<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn Membership>,
    objects: Arc<Objects<H::Generation>>,
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
            Arc::new(WriteWorkersAtDoor { door, objects }),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(dependencies),
    )
}

/// Build a removal that selects workers, marks their rows draining, waits
/// for them to settle and removes their rows. The holder drops the objects
/// when it follows the store. The drain marks the row: routing reads the
/// row's status, and a holder built with [`Objects::with_status_followed`]
/// mirrors it onto the held object.
///
/// A step drops from the job's lists a worker it leaves standing, as the
/// registry's steps do: a host step behind reads the workers the removal
/// took or found gone, and an object no row stands for. Where the door
/// refuses the asker and the job drained nothing, the drain step fails with
/// the door's sentence.
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

fn registration_refusal(url: &str, refusal: Refused) -> WorkflowError {
    let message = match refusal {
        Refused::NotMaker(why) => why,
        Refused::Past(past) => format!("{}: {}", past.fact, past.why),
        Refused::Absent => ABSENT.to_owned(),
        Refused::NotLeading(_) => NOT_LEADING.to_owned(),
        Refused::Moved => MOVED.to_owned(),
        Refused::Stands => format!("Worker {} already exists", Endpoint::redacted_str(url)),
        Refused::OutOfScope => Refused::OUT_OF_SCOPE.to_owned(),
    };
    failed(WRITE_WORKERS_AT_DOOR_STEP, message)
}

/// The row behind the worker a replacement by id names: the registry's
/// worker of that id, where it still stands at the job's revision under the
/// one built worker's URL, as the holder built it. The door checks the row
/// again in its turn.
fn replaced<G: Generation>(
    registry: &WorkerRegistry<G>,
    objects: &Objects<G>,
    workers: &[Arc<dyn Worker>],
    worker_id: &str,
    expected_revision: u64,
) -> WorkflowResult<Standing> {
    let [worker] = workers else {
        return Err(failed(
            WRITE_WORKERS_AT_DOOR_STEP,
            format!(
                "Replacing worker {worker_id} needs exactly one worker, got {}",
                workers.len()
            ),
        ));
    };
    let id = ObjectId::from_string(worker_id.to_owned());
    registry
        .get(&id)
        .filter(|held| held.revision() == expected_revision && held.url() == worker.url())
        .and_then(|held| objects.standing_of(&id, &held))
        .ok_or_else(|| {
            failed(
                WRITE_WORKERS_AT_DOOR_STEP,
                format!(
                    "Worker {worker_id} at {} is no longer at revision {expected_revision}, \
                     so the replacement was dropped",
                    Endpoint::redacted_str(worker.url())
                ),
            )
        })
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>>
    for WriteWorkersAtDoor<H::Generation>
{
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        let data = &mut context.data;
        if data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }
        let workers = data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_owned()))?;
        let maker = maker_of(&data.asker, &data.config.url)
            .map_err(|why| failed(WRITE_WORKERS_AT_DOOR_STEP, why.to_string()))?;
        let expect = match &data.registration_mode {
            WorkerRegistrationMode::Upsert => Expect::Anything,
            // The door takes one row a call. An add refused at a later rank
            // would leave the earlier ones, and its retry would be refused at
            // the rank it wrote: no rank is written until the door lays a
            // worker's ranks in one turn.
            WorkerRegistrationMode::CreateOnly if workers.len() > 1 => {
                return Err(failed(
                    WRITE_WORKERS_AT_DOOR_STEP,
                    format!(
                        "Adding a worker of {} ranks needs them written in one turn, \
                         which this door does not take yet",
                        workers.len()
                    ),
                ));
            }
            WorkerRegistrationMode::CreateOnly => Expect::Nothing,
            WorkerRegistrationMode::ReplaceById {
                worker_id,
                expected_revision,
            } => {
                let registry = require_host(data.host.as_ref())?.worker_registry();
                Expect::Seen(replaced(
                    registry,
                    &self.objects,
                    workers,
                    worker_id,
                    *expected_revision,
                )?)
            }
        };
        for worker in workers {
            let url = worker.url();
            self.door
                .found(&maker, expect, url, &|standing| {
                    facts(worker.as_ref(), standing)
                })
                .map_err(|refusal| registration_refusal(url, refusal))?;
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

/// What a removal write left of a worker.
#[derive(Debug, PartialEq, Eq)]
enum Left {
    /// The door wrote.
    Wrote(Wrote),
    /// No row stands under the URL: the worker is gone and stays the job's.
    Gone,
    /// The row moved under the asker: another registration stands there,
    /// and the job leaves it.
    Standing,
    /// The door refused the asker, with its sentence: the worker stays.
    Refused(String),
}

fn removal_answer(step: &str, url: &str, answer: Result<Wrote, Refused>) -> WorkflowResult<Left> {
    let refused = |why: String| {
        warn!(worker_url = %Endpoint::redacted_str(url), why, "Worker left alone");
        Left::Refused(why)
    };
    Ok(match answer {
        Ok(wrote) => Left::Wrote(wrote),
        Err(Refused::Moved) => {
            skipped(url, MOVED);
            Left::Standing
        }
        Err(Refused::Absent) => {
            skipped(url, ABSENT);
            Left::Gone
        }
        Err(Refused::NotMaker(why)) => refused(why),
        Err(Refused::OutOfScope) => refused(Refused::OUT_OF_SCOPE.to_owned()),
        Err(Refused::NotLeading(_)) => return Err(failed(step, NOT_LEADING)),
        Err(Refused::Stands) => return Err(failed(step, "the door answered a removal as an add")),
        Err(Refused::Past(past)) => {
            return Err(failed(step, format!("{}: {}", past.fact, past.why)))
        }
    })
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
        let mut kept = Vec::with_capacity(workers.len());
        let mut refusal = None;
        for worker in workers {
            let url = worker.url();
            let (id, revision) = match pin(registry, pins, url) {
                Pin::Held { id, revision } => (id, revision),
                // A worker gone altogether stays the job's, as the
                // registry's drain keeps it.
                Pin::Absent => {
                    skipped(url, "no worker stands in the registry");
                    kept.push(Arc::clone(worker));
                    continue;
                }
                Pin::LeftAlone => {
                    skipped(url, "the removal did not name this worker");
                    continue;
                }
            };
            // Gone between the two reads: the job's, as a worker gone
            // ahead of them.
            let Some(current) = registry.get(&id) else {
                skipped(url, "no worker stands in the registry");
                kept.push(Arc::clone(worker));
                continue;
            };
            if current.revision() != revision {
                skipped(url, "the worker moved after selection");
                continue;
            }
            // An object no row stands for is not the store's: the job keeps
            // it for the host's own steps.
            let Some(seen) = self.objects.standing_of(&id, &current) else {
                skipped(url, "no row stands for this object");
                kept.push(Arc::clone(worker));
                continue;
            };
            let answer = removal_answer(
                DRAIN_WORKERS_AT_DOOR_STEP,
                url,
                self.door.drained(by, url, seen),
            )?;
            match answer {
                Left::Wrote(Wrote::Draining(_) | Wrote::Unmoved(_)) => {
                    data.drained_workers.push(current);
                    kept.push(Arc::clone(worker));
                }
                Left::Wrote(_) => {
                    return Err(failed(
                        DRAIN_WORKERS_AT_DOOR_STEP,
                        "the door did not answer a drain",
                    ))
                }
                Left::Gone => kept.push(Arc::clone(worker)),
                Left::Standing => {}
                Left::Refused(why) => {
                    refusal.get_or_insert(why);
                }
            }
        }
        // A refusal of the request itself reaches its asker, as a
        // registration's does, where the job drained nothing.
        if let (true, Some(why)) = (data.drained_workers.is_empty(), refusal) {
            return Err(failed(DRAIN_WORKERS_AT_DOOR_STEP, why));
        }
        if kept.len() != workers.len() {
            data.keep_only(kept);
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
        let mut standing = Vec::new();
        for worker in &data.drained_workers {
            let url = worker.url();
            let Some(id) = registry.get_id_by_url(url) else {
                skipped(url, "no worker stands in the registry");
                continue;
            };
            let Some(current) = registry.get(&id) else {
                skipped(url, "no worker stands in the registry");
                continue;
            };
            if !Arc::ptr_eq(&current, worker) {
                skipped(url, "the worker moved while draining");
                standing.push(url.to_owned());
                continue;
            }
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
                Left::Wrote(Wrote::Removed(_)) => data.removed_workers.push(current),
                Left::Wrote(_) => {
                    return Err(failed(
                        REMOVE_WORKERS_AT_DOOR_STEP,
                        "the door did not answer a removal",
                    ))
                }
                Left::Gone => {}
                Left::Standing | Left::Refused(_) => standing.push(url.to_owned()),
            }
        }
        // A worker left standing under its URL leaves the job, as at the
        // registry's step: the steps behind touch nothing of it.
        if !standing.is_empty() {
            let kept = data
                .actual_workers_to_remove
                .iter()
                .flatten()
                .filter(|worker| !standing.iter().any(|url| url == worker.url()))
                .map(Arc::clone)
                .collect();
            data.keep_only(kept);
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
        worker::{HealthCheckConfig, RuntimeType, WorkerSpec},
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
            facts, Expect, Members, Membership, Objects, Refused, Remover, Standing, StoreDoor,
            Wrote,
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
            data.worker_kind = Some(WorkerKind::Local);
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
                .found(maker, Expect::Anything, worker.url(), &|old| {
                    facts(worker, old)
                })?;
            self.objects.follow(&*self.store);
            Ok(answer)
        }

        fn write(&self) -> WriteWorkersAtDoor<Members> {
            WriteWorkersAtDoor {
                door: Arc::clone(&self.door),
                objects: Arc::clone(&self.objects),
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
    async fn a_registration_that_is_not_local_is_skipped_and_writes_no_row() {
        let f = Fixture::new();
        for kind in [None, Some(WorkerKind::External)] {
            let mut cx = f.registration(WorkerJobAsker::Api, vec![worker(URL, "model")]);
            cx.data.worker_kind = kind;
            assert_eq!(
                f.write().execute(&mut cx).await.expect("skipped"),
                StepResult::Skip
            );
            assert!(cx.data.workers.is_none());
            // A run the host's steps have built nothing for yet is skipped as well.
            cx.data.actual_workers = None;
            assert_eq!(
                f.write().execute(&mut cx).await.expect("skipped"),
                StepResult::Skip
            );
        }
        assert_eq!(f.store.epoch(), 0);
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
        assert!(removal.data.worker_urls.is_empty());
        assert_eq!(f.store.by_key(URL), Some(anew));
    }

    /// The door answers that the row left, under an object the holder has
    /// not dropped yet: the worker stays the job's at either step.
    #[tokio::test]
    async fn a_removal_keeps_a_worker_whose_row_left_before_the_holder_followed() {
        let f = Fixture::new();
        let built = worker(URL, "model");
        let maker = Maker::Control;
        let by = Remover::Maker(&maker);
        for at_the_drain in [true, false] {
            f.seed(&maker, built.as_ref()).expect("registered");
            let live = f.host.registry.get_by_url(URL).expect("the held worker");
            let id = f.host.registry.get_id_by_url(URL).expect("its id");
            let seen = f.objects.standing_of(&id, &live).expect("its row");
            let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);
            if !at_the_drain {
                f.drain().execute(&mut removal).await.expect("drained");
                assert_eq!(removal.data.drained_workers.len(), 1);
            }
            f.door.gone(by, URL, seen).expect("the row left");
            if at_the_drain {
                f.drain().execute(&mut removal).await.expect("skipped");
                assert!(removal.data.drained_workers.is_empty());
            } else {
                f.remove().execute(&mut removal).await.expect("skipped");
                assert!(removal.data.removed_workers.is_empty());
            }
            assert_eq!(removal.data.worker_urls, [URL]);
            f.objects.follow(&*f.store);
        }
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
            _: Expect,
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
                ..f.write()
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
    async fn a_create_only_job_adds_a_row_and_fails_where_one_stands() {
        let f = Fixture::new();
        let mut cx = f.registration(WorkerJobAsker::Api, vec![worker(URL, "model")]);
        cx.data.registration_mode = WorkerRegistrationMode::CreateOnly;
        let write = f.write();
        assert_eq!(
            write.execute(&mut cx).await.expect("added"),
            StepResult::Success
        );
        let row = f.store.by_key(URL).expect("the added row");
        assert_eq!(row.maker, Maker::Control);

        let epoch = f.store.epoch();
        let error = write.execute(&mut cx).await.expect_err("the URL stands");
        assert_eq!(
            error.to_string(),
            "Step failed: write_workers_at_door - Worker http://worker:8000 already exists"
        );
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(f.store.epoch(), epoch);

        // Two ranks, the second standing: one door call a rank would write
        // the first, and the add's retry would be refused there. None is.
        let ranks =
            ["http://worker:8000@0", "http://worker:8000@1"].map(|url| worker(url, "model"));
        cx.data.actual_workers = Some(vec![Arc::clone(&ranks[1])]);
        write.execute(&mut cx).await.expect("the second rank");
        let epoch = f.store.epoch();
        cx.data.actual_workers = Some(ranks.to_vec());
        let error = write.execute(&mut cx).await.expect_err("two ranks");
        assert!(error.to_string().contains("2 ranks"), "{error}");
        assert_eq!(f.store.epoch(), epoch);
    }

    #[tokio::test]
    async fn a_replacement_by_id_is_laid_over_the_row_behind_that_worker() {
        let f = Fixture::new();
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        let id = f.host.registry.get_id_by_url(URL).expect("the worker's id");
        let revision = f.host.registry.get(&id).expect("the worker").revision();
        let before = f.store.by_key(URL).expect("the row");
        let by_id = |worker_id: &str, expected_revision| WorkerRegistrationMode::ReplaceById {
            worker_id: worker_id.to_owned(),
            expected_revision,
        };
        let relabelled = || -> Arc<dyn Worker> {
            let spec = WorkerSpec {
                models: vec![ModelCard::new("model")].into(),
                labels: [("tier".to_owned(), "silver".to_owned())].into(),
                ..WorkerSpec::new(URL.to_owned())
            };
            Arc::new(<BasicWorkerBuilder>::from_spec(spec).build())
        };
        let write = f.write();

        // Another revision than the job's, an id the registry does not
        // hold, a built worker at another URL, or two built workers: the
        // door is not asked.
        for (mode, workers, said) in [
            (
                by_id(id.as_str(), revision + 1),
                vec![relabelled()],
                "is no longer at revision",
            ),
            (
                by_id("no-such-id", revision),
                vec![relabelled()],
                "is no longer at revision",
            ),
            (
                by_id(id.as_str(), revision),
                vec![worker("http://other:8000", "model")],
                "is no longer at revision",
            ),
            (
                by_id(id.as_str(), revision),
                vec![relabelled(), relabelled()],
                "needs exactly one worker, got 2",
            ),
        ] {
            let mut cx = f.registration(WorkerJobAsker::Api, workers);
            cx.data.registration_mode = mode;
            let error = write.execute(&mut cx).await.expect_err("dropped");
            assert!(error.to_string().contains(said), "{error}");
            assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        }
        assert_eq!(f.store.by_key(URL).as_ref(), Some(&before));

        // The worker of that id at that revision: its row takes the new
        // labels in place, and the worker keeps its id in the registry.
        let mut cx = f.registration(WorkerJobAsker::Api, vec![relabelled()]);
        cx.data.registration_mode = by_id(id.as_str(), revision);
        assert_eq!(
            write.execute(&mut cx).await.expect("replaced"),
            StepResult::Success
        );
        let row = f.store.by_key(URL).expect("the row");
        assert_eq!(row.id, before.id);
        assert_eq!(
            row.facts.labels.get("tier").map(String::as_str),
            Some("silver")
        );
        f.objects.follow(&*f.store);
        assert_eq!(f.host.registry.get_id_by_url(URL), Some(id));
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
    async fn a_removal_marks_its_rows_draining_and_takes_them_behind_the_workers_own_window() {
        let f = Fixture::new();
        // The registration states a drain window of thirty seconds: the row
        // carries it, and the object the holder builds settles by it.
        let checks = HealthCheckConfig {
            drain_settle_secs: 30,
            ..HealthCheckConfig::default()
        };
        let built = <BasicWorkerBuilder>::from_spec(WorkerSpec::new(URL.to_owned()))
            .health_config(checks)
            .build();
        let mut cx = f.registration(WorkerJobAsker::Api, vec![Arc::new(built)]);
        f.write().execute(&mut cx).await.expect("registered");
        f.objects.follow(&*f.store);
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
        time::sleep(Duration::from_secs(29)).await;
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
        assert!(at.elapsed() >= Duration::from_secs(30));
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
        assert!(cx.data.worker_urls.is_empty());
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
        assert!(cx.data.worker_urls.is_empty());
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
        // Not the store's worker: the job keeps it for the host's own steps.
        assert_eq!(removal.data.worker_urls, [URL]);

        // A worker gone from the registry stays the job's as well.
        let mut gone = f.removal(
            WorkerJobAsker::Api,
            vec![worker("http://gone:8000", "model")],
        );
        f.drain()
            .execute(&mut gone)
            .await
            .expect("nothing to drain");
        assert_eq!(gone.data.worker_urls, ["http://gone:8000"]);
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
            (
                Refused::Moved,
                "DEBUG",
                "the membership row moved",
                Left::Standing,
            ),
            (
                Refused::Absent,
                "DEBUG",
                "no row is registered under the URL",
                Left::Gone,
            ),
            (
                Refused::NotMaker("another maker's row".to_owned()),
                "WARN",
                "another maker's row",
                Left::Refused("another maker's row".to_owned()),
            ),
            (
                Refused::OutOfScope,
                "WARN",
                "the row's maker is outside the health removal scope",
                Left::Refused("the row's maker is outside the health removal scope".to_owned()),
            ),
        ];
        for (refusal, _, _, left) in &answers {
            let answer = removal_answer("drain_workers_at_door", url, Err(refusal.clone()));
            assert_eq!(&answer.expect("left alone"), left);
        }
        let log = std::fs::read_to_string(log.path()).expect("the log text");
        assert_eq!(log.lines().count(), answers.len(), "{log}");
        for (line, (_, level, reason, _)) in log.lines().zip(answers) {
            assert!(line.contains(level) && line.contains(reason), "{line}");
            assert!(line.contains("http://***@worker:8000/p"), "{line}");
        }
        for unmasked in ["user", "secret", "hidden"] {
            assert!(!log.contains(unmasked), "{log}");
        }
    }

    /// A row that moved under the held object is left, and its worker leaves
    /// the job at either step.
    #[tokio::test]
    async fn a_removal_leaves_a_row_that_moved_and_drops_its_worker_from_the_job() {
        let f = Fixture::new();
        let built = worker(URL, "model");
        f.seed(&Maker::Control, built.as_ref()).expect("registered");
        let live = f.host.registry.get_by_url(URL).expect("the held worker");
        f.door
            .found(&Maker::Control, Expect::Anything, URL, &|old| WorkerFacts {
                priority: 5,
                ..facts(built.as_ref(), old)
            })
            .expect("the row moved in place");
        let row = f.store.by_key(URL);
        for at_the_drain in [true, false] {
            let mut removal = f.removal(WorkerJobAsker::Api, vec![Arc::clone(&live)]);
            if at_the_drain {
                f.drain().execute(&mut removal).await.expect("skipped");
                assert!(removal.data.drained_workers.is_empty());
            } else {
                removal.data.drained_workers.push(Arc::clone(&live));
                f.remove().execute(&mut removal).await.expect("skipped");
                assert!(removal.data.removed_workers.is_empty());
            }
            assert!(removal.data.worker_urls.is_empty());
            assert_eq!(f.store.by_key(URL), row);
        }
    }

    /// A removal the door refuses its asker fails the drain with the door's
    /// sentence where the job drained nothing. Beside a worker the job
    /// drained, the refused one leaves the job.
    #[tokio::test]
    async fn a_removal_the_door_refuses_its_asker_fails_with_the_doors_sentence() {
        let provider = Maker::provider("discovery", URL).expect("a provider's maker");
        for (maker, asker, said) in [
            (
                provider.clone(),
                WorkerJobAsker::Api,
                "the row is provider discovery's: its worker is changed at the provider's source",
            ),
            (
                Maker::Control,
                WorkerJobAsker::Health,
                "the row's maker is outside the health removal scope",
            ),
        ] {
            let f = Fixture::new();
            f.seed(&maker, worker(URL, "model").as_ref())
                .expect("registered");
            let live = f.host.registry.get_by_url(URL).expect("the held worker");
            let row = f.store.by_key(URL);
            let mut removal = f.removal(asker, vec![live]);
            let drain = f.drain();
            let error = drain.execute(&mut removal).await.expect_err("refused");
            assert_eq!(
                error.to_string(),
                format!("Step failed: drain_workers_at_door - {said}")
            );
            assert!(!StepExecutor::<Removal>::is_retryable(&drain, &error));
            assert_eq!(f.store.by_key(URL), row);
        }

        let f = Fixture::new();
        let other = "http://other:8000";
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        f.seed(&provider, worker(other, "model").as_ref())
            .expect("registered");
        let both = [URL, other].map(|url| f.host.registry.get_by_url(url).expect("held"));
        let mut removal = f.removal(WorkerJobAsker::Api, both.to_vec());
        f.drain().execute(&mut removal).await.expect("one drained");
        assert_eq!(removal.data.drained_workers.len(), 1);
        assert_eq!(removal.data.worker_urls, [URL]);
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
            .found(&Maker::Control, Expect::Anything, URL, &|old| WorkerFacts {
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
        assert!(removal.data.worker_urls.is_empty());
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
                Arc::clone(&f.objects),
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
            Arc::clone(&f.objects),
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
