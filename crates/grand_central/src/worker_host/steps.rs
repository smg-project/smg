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

use super::{
    facts, found::engine_of, MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter,
    Objects, Over, Precondition, Remover, RowRevision,
};
use crate::store::{ProviderKeyError, Registrant, WorkersStore};

/// The provider name used by the host's own discovery source.
pub const DISCOVERY: &str = "discovery";

/// Why a job's asker cannot make a registration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RegistrantError {
    /// Health checks ask for removals.
    #[error("health checks do not register workers")]
    Health,
    /// The store refused the maker's name or key.
    #[error(transparent)]
    Unmade(#[from] ProviderKeyError),
}

/// Who a job registers its workers as, using the address written in its spec.
pub fn registrant_of(asker: &WorkerJobAsker, address: &str) -> Result<Registrant, RegistrantError> {
    match asker {
        WorkerJobAsker::Api | WorkerJobAsker::Startup => Ok(Registrant::Control),
        WorkerJobAsker::Discovery => Ok(Registrant::provider(DISCOVERY, address)?),
        WorkerJobAsker::Health => Err(RegistrantError::Health),
    }
}

/// Who asks for a job's removal at the door.
pub fn remover_of(asker: &WorkerJobAsker, scope: UnhealthyRemovalScope) -> Remover<'static> {
    match asker {
        WorkerJobAsker::Api | WorkerJobAsker::Startup => Remover::Registrant(&Registrant::Control),
        WorkerJobAsker::Discovery => Remover::Provider(DISCOVERY),
        WorkerJobAsker::Health => Remover::Health(scope),
    }
}

/// The step that writes the registration's created workers into the store.
pub const WRITE_WORKERS_TO_STORE_STEP: &str = "write_workers_at_door";
/// The step that marks the selected workers' rows as draining.
pub const DRAIN_WORKERS_IN_STORE_STEP: &str = "drain_workers_at_door";
/// The step that removes the drained workers' rows after they settle.
pub const REMOVE_WORKERS_FROM_STORE_STEP: &str = "remove_workers_at_door";

/// Write the workers the preceding steps created in one registration.
pub struct WriteWorkersToStore<G> {
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<G>>,
    /// The store the write's door writes, where the write keeps. A row
    /// holds no credential: the holder keeps what each registration states
    /// ahead of its rows, in the keeper it reads at an object's build, and
    /// answers that keeper with the door's answer
    /// ([`Objects::register_keeping`]). Where the keeper's answer says so,
    /// the holder lays the address's objects again, and a lay hands each
    /// row's count to this store. No row is read from it.
    keeping: Option<Arc<dyn WorkersStore>>,
}

/// A registration that keeps credentials was asked over worker objects built
/// to keep none ([`Objects::with_credentials`]). The objects built there
/// would hold nothing of what the registration kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the holder was built with no keeper, so its registration line can keep no credential")]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct NoCredentials;

/// Mark the rows that still stand for the workers this removal names.
pub struct DrainWorkersInStore<G> {
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<G>>,
    scope: UnhealthyRemovalScope,
}

/// Remove the rows of the workers that drained and are still held.
pub struct RemoveWorkersFromStore<G> {
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<G>>,
    scope: UnhealthyRemovalScope,
}

/// Build the worker registration using the caller's settings and write the
/// created workers into the store. With an empty `behind`, the write follows
/// creation. Otherwise it waits for every step in `behind`.
///
/// The write is the only next step after a step in `behind`. A step writes
/// back the run's data whole, so parallel steps can overwrite each other.
///
/// The write is a step of the local kind, as the creation is: it skips a
/// registration of another kind, whose workers the host's own steps write.
///
/// The write lays a job's workers, the ranks of one engine, in one door
/// turn, and the job's mode is what it expects to stand. An upsert expects
/// anything. A create-only job expects nothing of the engine: tried again
/// behind a lost lead, it lays the ranks the lead left unwritten
/// ([`Precondition::Absent`]). A replacement by id expects the row
/// behind the registry's worker of that id, read through `objects` at the
/// revision the job names.
///
/// The write keeps no credential: the object the holder builds from a row
/// carries none. [`registration_into_store_keeping_credentials`] is the line of
/// a host that keeps its workers' credentials.
pub fn registration_into_store<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<H::Generation>>,
    behind: &[&'static str],
) -> WorkflowDefinition<WorkerRegistrationWorkflowData<H>> {
    registration_line(settings, gates, door, objects, None, behind)
}

/// [`registration_into_store`] for a caller that keeps its workers'
/// credentials. The write keeps the credential each worker states ahead of
/// the row, in the keeper `objects` was built with
/// ([`Objects::with_credentials`]), and the holder builds the row's object
/// with it.
///
/// `door` writes through the holder's write calls ([`Objects::upsert`],
/// [`Objects::patch`], [`Objects::remove`]), so a row's object stands when
/// the door answers: the line follows no store. A row written around the
/// holder has no object when its registration is answered, and the keeper
/// forgets it there.
///
/// `store` is the store that door writes: where the keeper's answer says
/// so, the write has the holder lay the address's objects again, and a lay
/// hands each row's count to it. No row is read from it.
///
/// The write runs in the holder's turn, from its keep to that lay-again. A
/// registration that states another credential alone moves no row, and the
/// holder lays another object while the row's revision stands. So a
/// replacement by id is refused as moved, ahead of its keep, where another
/// object stands under its URL by then.
///
/// A holder built with no keeper is refused ([`NoCredentials`]).
pub fn registration_into_store_keeping_credentials<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<H::Generation>>,
    store: Arc<dyn WorkersStore>,
    behind: &[&'static str],
) -> Result<WorkflowDefinition<WorkerRegistrationWorkflowData<H>>, NoCredentials> {
    if !objects.keeps_credentials() {
        return Err(NoCredentials);
    }
    Ok(registration_line(
        settings,
        gates,
        door,
        objects,
        Some(store),
        behind,
    ))
}

/// The registration line of both doors above.
fn registration_line<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<H::Generation>>,
    keeping: Option<Arc<dyn WorkersStore>>,
    behind: &[&'static str],
) -> WorkflowDefinition<WorkerRegistrationWorkflowData<H>> {
    let dependencies = if behind.is_empty() {
        &[CREATE_LOCAL_WORKER_STEP]
    } else {
        behind
    };
    worker_registration_workflow(settings, gates).add_step(
        StepDefinition::new(
            WRITE_WORKERS_TO_STORE_STEP,
            "Write workers at the membership door",
            Arc::new(WriteWorkersToStore {
                door,
                objects,
                keeping,
            }),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(dependencies),
    )
}

/// Build a removal that selects workers, marks their rows draining, waits
/// for them to settle and removes their rows. A door that writes through
/// the holder drops each object with its row.
/// The drain is the door's write. A door with
/// a holder marks the worker object draining and tells its row, so a
/// decision that reads the object's status picks it no more. Where a door
/// marks the row alone, the held object's status does not move.
///
/// A step drops from the job's lists a worker it leaves standing, as the
/// registry's steps do: a host step behind reads the workers the removal
/// took or found gone, and an object no row stands for. Where the door
/// refuses the asker and the job drained nothing, the drain step fails with
/// the door's sentence.
pub fn removal_from_store<H: WorkflowHost>(
    door: Arc<dyn MembershipWriter>,
    objects: Arc<Objects<H::Generation>>,
    scope: UnhealthyRemovalScope,
) -> WorkflowDefinition<WorkerRemovalWorkflowData<H>> {
    worker_removal_workflow(RemovalOrder::Door {
        drain: DRAIN_WORKERS_IN_STORE_STEP,
    })
    .add_step(
        StepDefinition::new(
            DRAIN_WORKERS_IN_STORE_STEP,
            "Drain workers at the membership door",
            Arc::new(DrainWorkersInStore {
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
            REMOVE_WORKERS_FROM_STORE_STEP,
            "Remove workers at the membership door",
            Arc::new(RemoveWorkersFromStore {
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
const IN_FLIGHT: &str = "another registration is writing at the worker's address";
const ABSENT: &str = "no row is registered under the URL";

fn failed(step: &str, message: impl Into<String>) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new(step),
        message: message.into(),
    }
}

fn registration_refusal(engine: &str, refusal: MembershipWriteRefused) -> WorkflowError {
    let message = match refusal {
        MembershipWriteRefused::NotRegistrant(why) => why,
        MembershipWriteRefused::Past(past) => format!("{}: {}", past.fact, past.why),
        MembershipWriteRefused::Absent => ABSENT.to_owned(),
        MembershipWriteRefused::NotLeading(_) => NOT_LEADING.to_owned(),
        MembershipWriteRefused::Moved => MOVED.to_owned(),
        MembershipWriteRefused::Stands => {
            format!("Worker {} already exists", Endpoint::redacted_str(engine))
        }
        MembershipWriteRefused::Draining => format!(
            "Worker {} is draining, so the replacement was dropped",
            Endpoint::redacted_str(engine)
        ),
        MembershipWriteRefused::OutOfScope => MembershipWriteRefused::OUT_OF_SCOPE.to_owned(),
        MembershipWriteRefused::NotLabelsAlone => {
            MembershipWriteRefused::NOT_LABELS_ALONE.to_owned()
        }
    };
    failed(WRITE_WORKERS_TO_STORE_STEP, message)
}

/// The row behind the worker a replacement by id names: the registry's
/// worker of that id, where it still stands at the job's revision under the
/// one built worker's URL, as the holder laid it, and that worker. The door
/// checks the row again in its turn, and the holder the worker.
fn replaced<G: Generation>(
    registry: &WorkerRegistry<G>,
    objects: &Objects<G>,
    workers: &[Arc<dyn Worker>],
    worker_id: &str,
    expected_revision: u64,
) -> WorkflowResult<(RowRevision, Arc<dyn Worker>)> {
    let [worker] = workers else {
        return Err(failed(
            WRITE_WORKERS_TO_STORE_STEP,
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
        .and_then(|held| Some((objects.standing_of(&id, &held)?, held)))
        .ok_or_else(|| {
            failed(
                WRITE_WORKERS_TO_STORE_STEP,
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
    for WriteWorkersToStore<H::Generation>
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
        let maker = registrant_of(&data.asker, &data.config.url)
            .map_err(|why| failed(WRITE_WORKERS_TO_STORE_STEP, why.to_string()))?;
        // A replacement by id names the worker it replaces beside the row.
        let (expect, over) = match &data.registration_mode {
            WorkerRegistrationMode::Upsert => (Precondition::Any, None),
            WorkerRegistrationMode::CreateOnly => (Precondition::Absent, None),
            WorkerRegistrationMode::ReplaceById {
                worker_id,
                expected_revision,
            } => {
                let registry = require_host(data.host.as_ref())?.worker_registry();
                let (standing, held) = replaced(
                    registry,
                    &self.objects,
                    workers,
                    worker_id,
                    *expected_revision,
                )?;
                (Precondition::AtRevision(standing), Some(held))
            }
        };
        self.write_at_door(&maker, expect, workers, over.as_ref().map(Over::Replaced))?;
        data.workers = Some(WorkerList::from_workers(workers));
        Ok(StepResult::Success)
    }

    /// A step's error carries text: the three fixed messages name the
    /// refusals worth a second attempt.
    fn is_retryable(&self, error: &WorkflowError) -> bool {
        matches!(
            error,
            WorkflowError::StepFailed { message, .. }
                if message == NOT_LEADING || message == MOVED || message == IN_FLIGHT
        )
    }
}

impl<G: Generation> WriteWorkersToStore<G> {
    /// Write a registration's workers at the door, in one call. A line
    /// that keeps has its holder run the whole of it in the holder's turn
    /// ([`Objects::register_keeping`]): the keep, the door's write, the
    /// keeper's answer and the lay-again that answer asks for. A guarded
    /// write is refused there as moved, ahead of its keep, where another
    /// object stands under the URL by then. The plain line takes no turn
    /// here.
    ///
    /// The workers are the ranks of one engine and dial one address, where
    /// one registration is in flight. So the keep is the registration's:
    /// it holds the credential the first worker states, and its acceptance
    /// binds every row the door answered. A second registration in flight
    /// there fails before the door, and is tried again.
    fn write_at_door(
        &self,
        maker: &Registrant,
        expect: Precondition,
        workers: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> WorkflowResult<Vec<MembershipWriteOutcome>> {
        let first = workers.first();
        // The refusal names the engine as the rows' facts state it.
        let engine = first.map_or("", |worker| engine_of(worker.as_ref()));
        let write = || self.door.found_rows(maker, expect, workers, over);
        let answer = match &self.keeping {
            None => write(),
            Some(store) => {
                let url = first.map_or("", |worker| worker.url());
                let stating = first.map(|worker| facts(worker.as_ref(), None));
                let stated = first
                    .zip(stating.as_ref())
                    .map(|(worker, facts)| (facts, worker.api_key().cloned()));
                self.objects
                    .register_keeping(store.as_ref(), stated, (url, over.map(Over::object)), write)
                    .map_err(|_| failed(WRITE_WORKERS_TO_STORE_STEP, IN_FLIGHT))?
            }
        };
        answer.map_err(|refusal| registration_refusal(engine, refusal))
    }
}

/// What the registry holds for a removal's worker at a step's second read.
enum Held {
    /// The object under the id.
    Is(Arc<dyn Worker>),
    /// No live object under the id or the URL: the worker left, or a
    /// replacement is still being laid.
    Gone,
    /// No object under the id and another under the URL: the worker was
    /// registered anew between the two reads.
    Anew,
}

fn held<G: Generation>(registry: &WorkerRegistry<G>, id: &ObjectId, url: &str) -> Held {
    match registry.get(id) {
        Some(worker) => Held::Is(worker),
        None if registry.get_id_by_url(url).is_some() => Held::Anew,
        None => Held::Gone,
    }
}

fn skipped(url: &str, why: &str) {
    debug!(worker_url = %Endpoint::redacted_str(url), why, "Worker left alone");
}

/// What a removal write left of a worker.
#[derive(Debug, PartialEq, Eq)]
enum Left {
    /// The door wrote.
    Wrote(MembershipWriteOutcome),
    /// No row stands under the URL: the worker is gone and stays the job's.
    Gone,
    /// The row moved under the asker: another registration stands there,
    /// and the job leaves it.
    Standing,
    /// The door refused the asker, with its sentence: the worker stays.
    Refused(String),
}

fn removal_answer(
    step: &str,
    url: &str,
    answer: Result<MembershipWriteOutcome, MembershipWriteRefused>,
) -> WorkflowResult<Left> {
    let refused = |why: String| {
        warn!(worker_url = %Endpoint::redacted_str(url), why, "Worker left alone");
        Left::Refused(why)
    };
    Ok(match answer {
        Ok(wrote) => Left::Wrote(wrote),
        Err(MembershipWriteRefused::Moved) => {
            skipped(url, MOVED);
            Left::Standing
        }
        Err(MembershipWriteRefused::Absent) => {
            skipped(url, ABSENT);
            Left::Gone
        }
        Err(MembershipWriteRefused::NotRegistrant(why)) => refused(why),
        Err(MembershipWriteRefused::OutOfScope) => {
            refused(MembershipWriteRefused::OUT_OF_SCOPE.to_owned())
        }
        Err(MembershipWriteRefused::NotLeading(_)) => return Err(failed(step, NOT_LEADING)),
        Err(MembershipWriteRefused::Stands) => {
            return Err(failed(step, "the door answered a removal as an add"))
        }
        Err(MembershipWriteRefused::Draining) => {
            return Err(failed(step, "the door answered a removal as a replacement"))
        }
        Err(MembershipWriteRefused::NotLabelsAlone) => {
            return Err(failed(step, "the door answered a removal as a refresh"))
        }
        Err(MembershipWriteRefused::Past(past)) => {
            return Err(failed(step, format!("{}: {}", past.fact, past.why)))
        }
    })
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRemovalWorkflowData<H>>
    for DrainWorkersInStore<H::Generation>
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
                // registry's drain keeps it. One registered anew between
                // the pin's own two reads stands under the URL again, and
                // leaves the job.
                Pin::Absent => {
                    if registry.get_id_by_url(url).is_some() {
                        skipped(url, "the worker moved after selection");
                        continue;
                    }
                    skipped(url, "no worker stands in the registry");
                    kept.push(Arc::clone(worker));
                    continue;
                }
                Pin::LeftAlone => {
                    skipped(url, "the removal did not name this worker");
                    continue;
                }
            };
            let current = match held(registry, &id, url) {
                Held::Is(current) if current.revision() == revision => current,
                // Gone between the two reads: the job's, as a worker gone
                // ahead of them.
                Held::Gone => {
                    skipped(url, "no worker stands in the registry");
                    kept.push(Arc::clone(worker));
                    continue;
                }
                Held::Is(_) | Held::Anew => {
                    skipped(url, "the worker moved after selection");
                    continue;
                }
            };
            // An object no row stands for is not the store's: the job keeps
            // it for the host's own steps.
            let Some(seen) = self.objects.standing_of(&id, &current) else {
                skipped(url, "no row stands for this object");
                kept.push(Arc::clone(worker));
                continue;
            };
            let answer = removal_answer(
                DRAIN_WORKERS_IN_STORE_STEP,
                url,
                self.door.drained(by, url, seen),
            )?;
            match answer {
                Left::Wrote(
                    MembershipWriteOutcome::Draining(_) | MembershipWriteOutcome::Unmoved(_),
                ) => {
                    data.drained_workers.push(current);
                    kept.push(Arc::clone(worker));
                }
                Left::Wrote(_) => {
                    return Err(failed(
                        DRAIN_WORKERS_IN_STORE_STEP,
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
            return Err(failed(DRAIN_WORKERS_IN_STORE_STEP, why));
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
    for RemoveWorkersFromStore<H::Generation>
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
            let current = match held(registry, &id, url) {
                Held::Is(current) if Arc::ptr_eq(&current, worker) => current,
                Held::Gone => {
                    skipped(url, "no worker stands in the registry");
                    continue;
                }
                Held::Is(_) | Held::Anew => {
                    skipped(url, "the worker moved while draining");
                    standing.push(url.to_owned());
                    continue;
                }
            };
            let Some(seen) = self.objects.standing_of(&id, &current) else {
                skipped(url, "no row stands for this object");
                continue;
            };
            let answer = removal_answer(
                REMOVE_WORKERS_FROM_STORE_STEP,
                url,
                self.door.gone(by, url, seen),
            )?;
            match answer {
                Left::Wrote(MembershipWriteOutcome::Removed(_)) => {
                    data.removed_workers.push(current);
                }
                Left::Wrote(_) => {
                    return Err(failed(
                        REMOVE_WORKERS_FROM_STORE_STEP,
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
#[path = "../../tests/unit/worker_host/steps/tests.rs"]
mod tests;
