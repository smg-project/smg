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
    facts, Credentials, Expect, KeptCredential, Membership, Objects, Over, Refused, Remover,
    Standing, Wrote,
};
use crate::{
    store::{Maker, Unmade, WorkersStore},
    view::WorkerId,
};

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
    keeping: Option<Keeping>,
}

/// What a write that keeps holds. A row holds no credential: the write
/// keeps what each worker states ahead of its row, and answers the keeper
/// with the door's answer.
struct Keeping {
    /// The keeper the write's holder reads at an object's build, read from
    /// that holder where the line is built ([`Objects::credentials`]). So
    /// the write and the build read one keeper.
    credentials: Arc<Credentials>,
    /// The store that holder follows. Where the keeper's answer says so,
    /// the write has the holder lay the address's objects again from it.
    store: Arc<dyn WorkersStore>,
}

/// A registration line that keeps was asked over a holder built with no
/// keeper ([`Objects::with_credentials`]). The objects such a holder
/// builds would hold nothing of what the line kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the holder was built with no keeper, so its registration line can keep no credential")]
pub struct NoKeeper;

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
///
/// The write keeps no credential: the object the holder builds from a row
/// carries none. [`registration_at_door_keeping`] is the line of a host
/// that keeps its workers' credentials.
pub fn registration_at_door<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn Membership>,
    objects: Arc<Objects<H::Generation>>,
    behind: &[&'static str],
) -> WorkflowDefinition<WorkerRegistrationWorkflowData<H>> {
    registration_line(settings, gates, door, objects, None, behind)
}

/// [`registration_at_door`] for a host that keeps its workers'
/// credentials. The write keeps the credential each worker states ahead of
/// the row, in the keeper `objects` was built with
/// ([`Objects::with_credentials`]), and the holder builds the row's object
/// with it.
///
/// `store` is the store the holder follows: where the keeper's answer says
/// so, the write has the holder lay the address's objects again from it. A
/// holder is built with no store, since each of its reads is handed one.
///
/// The write runs in the holder's turn, from its keep to that lay-again. A
/// registration that states another credential alone moves no row, and the
/// holder lays another object while the row's revision stands. So a
/// replacement by id is refused as moved, ahead of its keep, where another
/// object stands under its URL by then.
///
/// A holder built with no keeper is refused ([`NoKeeper`]). A holder that
/// keeps built objects ([`Objects::with_built_kept`]) is one: a built
/// object holds its own credential.
pub fn registration_at_door_keeping<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn Membership>,
    objects: Arc<Objects<H::Generation>>,
    store: Arc<dyn WorkersStore>,
    behind: &[&'static str],
) -> Result<WorkflowDefinition<WorkerRegistrationWorkflowData<H>>, NoKeeper> {
    let credentials = Arc::clone(objects.credentials().ok_or(NoKeeper)?);
    let keeping = Keeping { credentials, store };
    Ok(registration_line(
        settings,
        gates,
        door,
        objects,
        Some(keeping),
        behind,
    ))
}

/// The registration line of both doors above.
fn registration_line<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
    door: Arc<dyn Membership>,
    objects: Arc<Objects<H::Generation>>,
    keeping: Option<Keeping>,
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
            Arc::new(WriteWorkersAtDoor {
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
const IN_FLIGHT: &str = "another registration is writing at the worker's address";
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
        Refused::Draining => format!(
            "Worker {} is draining, so the replacement was dropped",
            Endpoint::redacted_str(url)
        ),
        Refused::OutOfScope => Refused::OUT_OF_SCOPE.to_owned(),
        Refused::NotLabelsAlone => Refused::NOT_LABELS_ALONE.to_owned(),
    };
    failed(WRITE_WORKERS_AT_DOOR_STEP, message)
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
) -> WorkflowResult<(Standing, Arc<dyn Worker>)> {
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
        .and_then(|held| Some((objects.standing_of(&id, &held)?, held)))
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
        // A replacement by id names the worker it replaces beside the row.
        let (expect, over) = match &data.registration_mode {
            WorkerRegistrationMode::Upsert => (Expect::Anything, None),
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
            WorkerRegistrationMode::CreateOnly => (Expect::Nothing, None),
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
                (Expect::Seen(standing), Some(held))
            }
        };
        for worker in workers {
            self.write_at_door(&maker, expect, worker, over.as_ref().map(Over::Replaced))?;
        }
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

impl<G: Generation> WriteWorkersAtDoor<G> {
    /// Write one worker at the door. A line that keeps runs the whole of
    /// it in its holder's turn ([`Objects::keeping`]): the keep, the door's
    /// write, the keeper's answer and the lay-again that answer asks for.
    /// A guarded write is refused there as moved, ahead of its keep, where
    /// another object stands under the URL by then. The plain line takes
    /// no turn here: where its holder keeps built objects, its door hands
    /// in under that turn.
    fn write_at_door(
        &self,
        maker: &Maker,
        expect: Expect,
        worker: &Arc<dyn Worker>,
        over: Option<Over<'_>>,
    ) -> WorkflowResult<Wrote> {
        let url = worker.url();
        let write = || -> WorkflowResult<Wrote> {
            // Kept ahead of the write: the holder may build the row's
            // object inside the door's turn.
            let kept = self.keep(worker.as_ref())?;
            let answer = self.door.found_built(maker, expect, worker, over);
            self.answer_kept(worker.as_ref(), kept, answer.as_ref().ok());
            answer.map_err(|refusal| registration_refusal(url, refusal))
        };
        if self.keeping.is_none() {
            return write();
        }
        self.objects
            .keeping(url, over.map(Over::object), write)
            .map_err(|refusal| registration_refusal(url, refusal))?
    }

    /// Keep what `worker` states for its engine, ahead of its row's write.
    /// One registration is in flight at an address: a second fails here,
    /// before the door, and is tried again.
    fn keep(&self, worker: &dyn Worker) -> WorkflowResult<Option<KeptCredential>> {
        let Some(keeping) = &self.keeping else {
            return Ok(None);
        };
        keeping
            .credentials
            .keep(&facts(worker, None), worker.api_key().cloned())
            .map(Some)
            .map_err(|_| failed(WRITE_WORKERS_AT_DOOR_STEP, IN_FLIGHT))
    }

    /// Answer the keeper at the door's answer, with no await between the
    /// two: the row the door wrote is accepted, and a refusal puts back
    /// what stood. Where the keeper says an object may hold another
    /// credential than its row takes, the holder lays the address again.
    fn answer_kept(
        &self,
        worker: &dyn Worker,
        kept: Option<KeptCredential>,
        wrote: Option<&Wrote>,
    ) {
        let (Some(keeping), Some(kept)) = (&self.keeping, kept) else {
            return;
        };
        let again = match wrote.and_then(written_row) {
            Some(row) => kept.accepted(row),
            None => kept.put_back(),
        };
        if again {
            self.objects
                .lay_again(keeping.store.as_ref(), &facts(worker, None));
        }
    }
}

/// The row a registration's write stands on: the new one of a replacement.
/// A removal's answers name no such row.
fn written_row(wrote: &Wrote) -> Option<WorkerId> {
    match *wrote {
        Wrote::Registered(row) | Wrote::InPlace(row) | Wrote::Unmoved(row) => Some(row),
        Wrote::Another { new, .. } => Some(new),
        Wrote::Draining(_) | Wrote::Removed(_) => None,
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
        Err(Refused::Draining) => {
            return Err(failed(step, "the door answered a removal as a replacement"))
        }
        Err(Refused::NotLabelsAlone) => {
            return Err(failed(step, "the door answered a removal as a refresh"))
        }
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
        store::{Maker, MemoryStore, Overload, Unmade, WorkerFacts, WorkerStatus, WorkersStore},
        worker_host::{
            facts, Expect, Members, Membership, Objects, Over, Refused, Remover, Standing,
            StoreDoor, Wrote,
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
                keeping: None,
            }
        }

        /// The fixture with a keeper that its holder reads.
        fn keeping() -> (Self, Arc<Credentials>) {
            let mut fixture = Self::new();
            let credentials = Arc::new(Credentials::new());
            let registry = Arc::clone(&fixture.host.registry);
            fixture.objects =
                Arc::new(Objects::new(registry).with_credentials(Arc::clone(&credentials)));
            (fixture, credentials)
        }

        /// The write that keeps in `credentials`.
        fn write_keeping(&self, credentials: &Arc<Credentials>) -> WriteWorkersAtDoor<Members> {
            let store: Arc<dyn WorkersStore> = self.store.clone();
            WriteWorkersAtDoor {
                keeping: Some(Keeping {
                    credentials: Arc::clone(credentials),
                    store,
                }),
                ..self.write()
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

    /// A worker that states `key` for its engine.
    fn keyed(url: &str, key: &str) -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            models: vec![ModelCard::new("model")].into(),
            api_key: Some(key.to_owned()),
            ..WorkerSpec::new(url.to_owned())
        };
        Arc::new(<BasicWorkerBuilder>::from_spec(spec).build())
    }

    /// The credential the row's object at `url` holds, once the holder has
    /// followed the store.
    fn held_key(f: &Fixture, url: &str) -> Option<String> {
        f.objects.follow(&*f.store);
        let object = f.host.registry.get_by_url(url).expect("the row's object");
        object.api_key().cloned()
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

        fn found_built(
            &self,
            _: &Maker,
            _: Expect,
            _: &Arc<dyn Worker>,
            _: Option<Over<'_>>,
        ) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }

        fn drained(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }

        fn gone(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(self.0.clone())
        }

        fn refreshed(
            &self,
            _: Standing,
            _: &Arc<dyn Worker>,
            _: &Arc<dyn Worker>,
        ) -> Result<Wrote, Refused> {
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

    /// A row holds no credential. A write that keeps hands what each worker
    /// states to the host's keeper, and the holder builds the row's object
    /// with it. A worker that states none is built with none.
    #[tokio::test]
    async fn a_keeping_write_builds_each_rows_object_with_what_its_worker_states() {
        let (f, credentials) = Fixture::keeping();
        let plain = "http://plain:8000";
        let workers = vec![keyed(URL, "key-a"), worker(plain, "model")];
        let mut cx = f.registration(WorkerJobAsker::Startup, workers);
        let write = f.write_keeping(&credentials);
        assert_eq!(
            write.execute(&mut cx).await.expect("written"),
            StepResult::Success
        );
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-a"));
        assert_eq!(held_key(&f, plain), None);
    }

    /// A worker registered again with another credential moves no row, so
    /// no change tells the holder. The write has the holder lay the
    /// address's object again, and that object holds the new credential.
    #[tokio::test]
    async fn a_keeping_write_of_another_credential_lays_the_object_again() {
        let (f, credentials) = Fixture::keeping();
        let write = f.write_keeping(&credentials);
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-a")]);
        write.execute(&mut cx).await.expect("written");
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-a"));
        let epoch = f.store.epoch();

        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-b")]);
        write.execute(&mut cx).await.expect("unchanged");
        assert_eq!(f.store.epoch(), epoch, "the row did not move");
        let object = f.host.registry.get_by_url(URL).expect("the row's object");
        assert_eq!(object.api_key().map(String::as_str), Some("key-b"));
    }

    /// A write the door refuses puts its credential back: nothing is kept
    /// for a row that was not written.
    #[tokio::test]
    async fn a_keeping_write_the_door_refuses_keeps_nothing() {
        let (f, credentials) = Fixture::keeping();
        let discovered = Maker::provider(DISCOVERY, URL).expect("a provider");
        f.seed(&discovered, worker(URL, "model").as_ref())
            .expect("registered");

        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-a")]);
        let write = f.write_keeping(&credentials);
        write
            .execute(&mut cx)
            .await
            .expect_err("another maker's row");
        assert!(credentials.is_empty());
        assert_eq!(held_key(&f, URL), None);
    }

    /// One registration is in flight at an address. A second write there
    /// fails before the door, and is tried again.
    #[tokio::test]
    async fn a_keeping_write_at_an_address_in_flight_fails_and_is_tried_again() {
        let (f, credentials) = Fixture::keeping();
        let first = keyed(URL, "key-a");
        let token = credentials
            .keep(&facts(first.as_ref(), None), first.api_key().cloned())
            .expect("no registration in flight");

        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-b")]);
        let write = f.write_keeping(&credentials);
        let error = write.execute(&mut cx).await.expect_err("in flight");
        assert!(StepExecutor::<Registration>::is_retryable(&write, &error));
        assert!(f.store.by_key(URL).is_none(), "the door was not asked");
        assert!(!token.put_back());
    }

    /// The line that keeps reads the keeper from its holder, where the line
    /// is built. A holder built with none is refused. A holder takes a
    /// keeper or kept objects, never both, and the later call stands: so a
    /// holder that keeps built objects has no keeper, and is refused too.
    #[test]
    fn the_line_that_keeps_is_not_built_over_a_holder_with_no_keeper() {
        let f = Fixture::new();
        let store: Arc<dyn WorkersStore> = f.store.clone();
        let line = |objects: Objects<Members>| {
            registration_at_door_keeping::<Host>(
                f.host.registration_settings(),
                RegistrationGates::default(),
                Arc::clone(&f.door),
                Arc::new(objects),
                Arc::clone(&store),
                &[],
            )
        };
        let registry = || Arc::clone(&f.host.registry);
        let keeper = Arc::new(Credentials::new());
        assert!(matches!(line(Objects::new(registry())), Err(NoKeeper)));

        let built_kept = Objects::new(registry())
            .with_credentials(Arc::clone(&keeper))
            .with_built_kept();
        assert!(built_kept.credentials().is_none());
        assert!(matches!(line(built_kept), Err(NoKeeper)));

        let keeping = Objects::new(registry())
            .with_built_kept()
            .with_credentials(keeper);
        assert!(keeping.credentials().is_some());
        assert!(line(keeping).is_ok());
    }

    /// A door another writer moved the row under: it writes the row another
    /// way, has the holder follow, and then refuses. So the holder builds
    /// the row's object while the registration is in flight.
    struct MovedUnder {
        door: Arc<dyn Membership>,
        objects: Arc<Objects<Members>>,
        store: Arc<MemoryStore>,
    }

    impl Membership for MovedUnder {
        fn found(
            &self,
            _: &Maker,
            _: Expect,
            _: &str,
            _: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<Wrote, Refused> {
            Err(Refused::Moved)
        }

        fn found_built(
            &self,
            maker: &Maker,
            _: Expect,
            built: &Arc<dyn Worker>,
            _: Option<Over<'_>>,
        ) -> Result<Wrote, Refused> {
            self.door
                .found(maker, Expect::Anything, built.url(), &|standing| {
                    let mut moved = standing.cloned().expect("the row stands");
                    moved.labels.insert("moved".to_owned(), "under".to_owned());
                    moved
                })?;
            self.objects.follow(&*self.store);
            Err(Refused::Moved)
        }

        fn drained(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(Refused::Moved)
        }

        fn gone(&self, _: Remover<'_>, _: &str, _: Standing) -> Result<Wrote, Refused> {
            Err(Refused::Moved)
        }

        fn refreshed(
            &self,
            _: Standing,
            _: &Arc<dyn Worker>,
            _: &Arc<dyn Worker>,
        ) -> Result<Wrote, Refused> {
            Err(Refused::Moved)
        }
    }

    /// A refused write puts back what stood. Where the holder built the
    /// row's object while the registration was in flight, that object holds
    /// what the registration stated. The write has the holder lay it again,
    /// with the credential that stands.
    #[tokio::test]
    async fn a_refused_keeping_write_lays_again_an_object_built_in_its_flight() {
        let (f, credentials) = Fixture::keeping();
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-a")]);
        f.write_keeping(&credentials)
            .execute(&mut cx)
            .await
            .expect("written");
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-a"));

        let write = WriteWorkersAtDoor {
            door: Arc::new(MovedUnder {
                door: Arc::clone(&f.door),
                objects: Arc::clone(&f.objects),
                store: Arc::clone(&f.store),
            }),
            ..f.write_keeping(&credentials)
        };
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-b")]);
        let error = write.execute(&mut cx).await.expect_err("moved");
        assert!(StepExecutor::<Registration>::is_retryable(&write, &error));
        let object = f.host.registry.get_by_url(URL).expect("the row's object");
        assert_eq!(object.api_key().map(String::as_str), Some("key-a"));
    }

    /// A registration that states other models takes the row's place with a
    /// new row, and the keeper is answered with that new row. So an object
    /// built from it later, at its next revision, still holds what the
    /// registration stated.
    #[tokio::test]
    async fn a_keeping_write_that_replaces_a_row_binds_the_new_row() {
        let (f, credentials) = Fixture::keeping();
        let write = f.write_keeping(&credentials);
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-a")]);
        write.execute(&mut cx).await.expect("written");
        let old = f.store.by_key(URL).expect("the row").id;

        let replacing = |tier: &str, key: Option<&str>| -> Arc<dyn Worker> {
            let spec = WorkerSpec {
                models: vec![ModelCard::new("model"), ModelCard::new("model-b")].into(),
                labels: [("tier".to_owned(), tier.to_owned())].into(),
                api_key: key.map(str::to_owned),
                ..WorkerSpec::new(URL.to_owned())
            };
            Arc::new(<BasicWorkerBuilder>::from_spec(spec).build())
        };
        let mut cx = f.registration(
            WorkerJobAsker::Startup,
            vec![replacing("gold", Some("key-b"))],
        );
        write.execute(&mut cx).await.expect("replaced");
        let new = f.store.by_key(URL).expect("the new row").id;
        assert_ne!(new, old, "another row stands under the URL");
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-b"));

        // The row moves in place through the line that keeps nothing: its
        // object is built anew, with what the keeper binds to the new row.
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![replacing("silver", None)]);
        f.write().execute(&mut cx).await.expect("relabelled");
        assert_eq!(f.store.by_key(URL).expect("the row").id, new);
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-b"));
    }

    /// A credential is no fact of a row. A registration that states another
    /// credential alone moves no row, and the holder lays another object
    /// while the row's revision stands. A replacement by id that read its
    /// worker ahead of that registration is refused as moved in the
    /// holder's turn, ahead of its keep: the door is not asked, and the
    /// object holds what the later registration stated. Tried again, the
    /// replacement is dropped: the object laid again is at another
    /// revision.
    #[tokio::test]
    async fn a_guarded_keeping_write_is_refused_once_another_object_stands() {
        let (f, credentials) = Fixture::keeping();
        let write = f.write_keeping(&credentials);
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-a")]);
        write.execute(&mut cx).await.expect("written");
        let id = f.host.registry.get_id_by_url(URL).expect("the worker's id");
        let revision = f.host.registry.get(&id).expect("the worker").revision();

        // The replacement reads the worker it replaces: the step's first
        // read.
        let stale = vec![keyed(URL, "key-old")];
        let (seen, over) = replaced(&f.host.registry, &f.objects, &stale, id.as_str(), revision)
            .expect("the worker stands at the job's revision");

        // A registration of another credential alone runs whole meanwhile.
        let mut cx = f.registration(WorkerJobAsker::Startup, vec![keyed(URL, "key-b")]);
        write.execute(&mut cx).await.expect("unchanged");
        let epoch = f.store.epoch();
        let row = f.store.by_key(URL).expect("the row");
        assert_eq!((row.id, row.revision), (seen.row, seen.revision));

        let error = write
            .write_at_door(
                &Maker::Control,
                Expect::Seen(seen),
                &stale[0],
                Some(Over::Replaced(&over)),
            )
            .expect_err("another object stands under the URL");
        assert!(StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(f.store.epoch(), epoch, "the door was not asked");
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-b"));
        let kept = facts(stale[0].as_ref(), None);
        assert_eq!(f.objects.lay_again(&*f.store, &kept), 0);

        let mut cx = f.registration(WorkerJobAsker::Api, stale);
        cx.data.registration_mode = WorkerRegistrationMode::ReplaceById {
            worker_id: id.as_str().to_owned(),
            expected_revision: revision,
        };
        let error = write.execute(&mut cx).await.expect_err("dropped");
        assert!(
            error.to_string().contains("is no longer at revision"),
            "{error}"
        );
        assert!(!StepExecutor::<Registration>::is_retryable(&write, &error));
        assert_eq!(held_key(&f, URL).as_deref(), Some("key-b"));
    }

    /// The line that keeps is the plain line, step for step. The write is
    /// the one step that differs, in what it holds.
    #[test]
    fn the_line_that_keeps_orders_its_steps_as_the_plain_line() {
        let (f, _) = Fixture::keeping();
        let store: Arc<dyn WorkersStore> = f.store.clone();
        let plain = registration_at_door::<Host>(
            f.host.registration_settings(),
            RegistrationGates::default(),
            Arc::clone(&f.door),
            Arc::clone(&f.objects),
            &[],
        );
        let keeping = registration_at_door_keeping::<Host>(
            f.host.registration_settings(),
            RegistrationGates::default(),
            Arc::clone(&f.door),
            Arc::clone(&f.objects),
            store,
            &[],
        )
        .expect("a holder with a keeper");
        let order = |line: &WorkflowDefinition<Registration>| {
            line.steps
                .iter()
                .map(|step| (step.id.clone(), step.depends_on.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(order(&plain), order(&keeping));
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
        assert_eq!(f.host.registry.get_id_by_url(URL), Some(id.clone()));

        // The worker drains: its removal wins, and the replacement is told.
        let live = f.host.registry.get(&id).expect("the worker");
        let seen = f.objects.standing_of(&id, &live).expect("its row");
        f.door
            .drained(Remover::Maker(&Maker::Control), URL, seen)
            .expect("draining");
        let mut cx = f.registration(WorkerJobAsker::Api, vec![relabelled()]);
        cx.data.registration_mode = by_id(id.as_str(), live.revision());
        let error = write.execute(&mut cx).await.expect_err("draining");
        assert_eq!(
            error.to_string(),
            "Step failed: write_workers_at_door - Worker http://worker:8000 is draining, \
             so the replacement was dropped"
        );
    }

    /// A replacement by id is its maker's whole statement of what a
    /// registration can state: a threshold it leaves out is cleared. An add
    /// states its own. An upsert and an update in place keep the standing
    /// row's.
    #[tokio::test]
    async fn a_replacement_by_id_clears_a_threshold_it_leaves_out_and_the_other_verbs_keep_it() {
        let f = Fixture::new();
        let limited = |waiting| -> Arc<dyn Worker> {
            let mut spec = WorkerSpec {
                models: vec![ModelCard::new("model")].into(),
                ..WorkerSpec::new(URL.to_owned())
            };
            spec.overload.waiting_requests = waiting;
            Arc::new(<BasicWorkerBuilder>::from_spec(spec).build())
        };
        let pair = || f.store.by_key(URL).expect("the row").facts.overload;
        let eight = Overload {
            waiting_requests: Some(8),
            ..Overload::default()
        };
        let standing = || {
            f.objects.follow(&*f.store);
            let id = f.host.registry.get_id_by_url(URL).expect("the worker's id");
            let live = f.host.registry.get(&id).expect("the worker");
            (id, live)
        };
        let write = f.write();

        // An add states its own pair.
        let mut cx = f.registration(WorkerJobAsker::Api, vec![limited(Some(8))]);
        cx.data.registration_mode = WorkerRegistrationMode::CreateOnly;
        write.execute(&mut cx).await.expect("added");
        assert_eq!(pair(), eight);

        // An upsert that leaves the threshold out keeps the row's.
        let mut cx = f.registration(WorkerJobAsker::Api, vec![limited(None)]);
        cx.data.registration_mode = WorkerRegistrationMode::Upsert;
        write.execute(&mut cx).await.expect("upserted");
        assert_eq!(pair(), eight);

        // So does an update in place.
        let (id, live) = standing();
        let seen = f.objects.standing_of(&id, &live).expect("its row");
        let over = Some(Over::Updated(&live));
        f.door
            .found_built(&Maker::Control, Expect::Seen(seen), &limited(None), over)
            .expect("updated");
        assert_eq!(pair(), eight);

        // A replacement by id that leaves it out clears it.
        let (id, live) = standing();
        let by_id = |live: &Arc<dyn Worker>| WorkerRegistrationMode::ReplaceById {
            worker_id: id.as_str().to_owned(),
            expected_revision: live.revision(),
        };
        let mut cx = f.registration(WorkerJobAsker::Api, vec![limited(None)]);
        cx.data.registration_mode = by_id(&live);
        write.execute(&mut cx).await.expect("replaced");
        assert_eq!(pair(), Overload::default());
        let (_, live) = standing();

        // One that states a threshold sets it.
        let mut cx = f.registration(WorkerJobAsker::Api, vec![limited(Some(3))]);
        cx.data.registration_mode = by_id(&live);
        write.execute(&mut cx).await.expect("replaced again");
        assert_eq!(pair().waiting_requests, Some(3));
    }

    /// At a step's second read a worker registered anew under the URL is
    /// told from one that left.
    #[tokio::test]
    async fn a_worker_registered_anew_between_a_steps_reads_is_told_from_one_gone() {
        let f = Fixture::new();
        f.seed(&Maker::Control, worker(URL, "model").as_ref())
            .expect("registered");
        let first = f.host.registry.get_id_by_url(URL).expect("its id");
        assert!(matches!(held(&f.host.registry, &first, URL), Held::Is(_)));

        // Other models: the URL's next life, under a new id.
        f.seed(&Maker::Control, worker(URL, "other").as_ref())
            .expect("registered anew");
        let second = f.host.registry.get_id_by_url(URL).expect("the new id");
        assert_ne!(first, second);
        assert!(matches!(held(&f.host.registry, &first, URL), Held::Anew));

        let live = f.host.registry.get(&second).expect("the worker");
        let seen = f.objects.standing_of(&second, &live).expect("its row");
        f.door
            .gone(Remover::Maker(&Maker::Control), URL, seen)
            .expect("removed");
        f.objects.follow(&*f.store);
        assert!(matches!(held(&f.host.registry, &second, URL), Held::Gone));
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
