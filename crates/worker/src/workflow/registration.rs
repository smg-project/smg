//! The registration of a worker as a workflow: the steps of it that stand in
//! this crate, in this crate's order.

use std::{sync::Arc, time::Duration};

use wfaas::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition};

use super::{
    create_worker::{CreateLocalWorkerStep, CREATE_LOCAL_WORKER_STEP},
    data::WorkerRegistrationWorkflowData,
    detect_connection::DetectConnectionModeStep,
    detect_runtime::DetectRuntimeStep,
    discover_dp::DiscoverDpInfoStep,
    discover_metadata::DiscoverMetadataStep,
    host::WorkflowHost,
};
use crate::settings::RegistrationSettings;

/// The id of the registration workflow.
pub const WORKER_REGISTRATION_WORKFLOW: &str = "worker_registration";
/// The id of the step that settles how a worker is spoken to: HTTP, and in
/// which version, or gRPC.
pub const DETECT_CONNECTION_MODE_STEP: &str = "detect_connection_mode";
/// The id of the step that tells which runtime a worker runs.
pub const DETECT_RUNTIME_STEP: &str = "detect_backend";
/// The id of the step that reads what a worker's engine reports about
/// itself.
pub const DISCOVER_METADATA_STEP: &str = "discover_metadata";
/// The id of the step that reads a worker's data-parallel (DP) shape.
pub const DISCOVER_DP_STEP: &str = "discover_dp_info";

/// The host's steps that this crate's steps of a registration wait on, one
/// field a gate. A gate names a step the host adds to the workflow; the
/// engine refuses a definition whose gate names a step nobody added.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegistrationGates {
    /// The host's step that ends before the connection detection starts, for
    /// a host that settles something about the worker first. With none, the
    /// detection opens the workflow.
    pub detection: Option<&'static str>,
    /// The host's step that ends before creation starts. With none, creation
    /// waits on the DP read. The named step must itself wait on the DP read,
    /// directly or through a chain of the host's own steps: creation reads
    /// the DP shape, and a step beside the DP read races it for the run's data.
    pub creation: Option<&'static str>,
}

/// How many times a step that polls a starting worker tries.
///
/// Startup detection polls the starting worker every
/// `startup_check_interval_secs` until `startup_timeout_secs` elapses. The
/// attempt budget is derived from that cadence (reserving 10% of the timeout
/// for workflow overhead like step transitions) so the two knobs stay
/// consistent: total wait ≈ attempts × check interval ≈ startup timeout. A
/// floor keeps a very short timeout retrying a few times.
pub fn startup_attempts(settings: &RegistrationSettings) -> u32 {
    const EFFECTIVE_TIMEOUT_FACTOR: f64 = 0.9;
    const MIN_ATTEMPTS: u32 = 3;

    let interval_secs = (settings.startup_check_interval_secs as f64).max(1.0);
    let effective_timeout = settings.startup_timeout_secs as f64 * EFFECTIVE_TIMEOUT_FACTOR;
    ((effective_timeout / interval_secs).ceil() as u32).max(MIN_ATTEMPTS)
}

/// The worker side of a registration: the connection detection, polled at
/// the startup cadence of `settings`, the runtime detection behind it with
/// the same budget of attempts, the metadata read, the DP read, and then
/// creation. The DP read asks the worker nothing: it takes the shape from
/// the labels the metadata read left and from the spec, on every
/// registration. Creation waits on the DP read or the host's creation gate.
///
/// The startup delay, the time a worker is left alone before its first
/// probe, stands on the step that opens the workflow. With
/// `gates.detection` named that step is the host's, and the host sets the
/// delay there; the detection then waits on it and carries none. With no
/// gate the detection opens the workflow and carries the delay.
///
/// The engine runs a step on a copy of the run's data and writes the whole
/// copy back, unless the step skipped. Two steps that wait on one step can
/// run at once, and the copy written last then replaces what the other
/// step wrote. So a host hangs each step of its own that writes the data
/// in this one chain, never beside a step of it: ahead of creation as its
/// gate, behind creation, or behind another step of its own.
pub fn worker_registration_workflow<H: WorkflowHost>(
    settings: &RegistrationSettings,
    gates: RegistrationGates,
) -> WorkflowDefinition<WorkerRegistrationWorkflowData<H>> {
    let mut detect_connection_mode = StepDefinition::new(
        DETECT_CONNECTION_MODE_STEP,
        "Detect Connection Mode",
        Arc::new(DetectConnectionModeStep),
    )
    .with_retry(RetryPolicy {
        max_attempts: startup_attempts(settings),
        // Poll the starting worker at the configured startup cadence.
        backoff: BackoffStrategy::Fixed(Duration::from_secs(settings.startup_check_interval_secs)),
    })
    .with_timeout(Duration::from_secs(settings.startup_timeout_secs))
    .with_failure_action(FailureAction::FailWorkflow);
    match gates.detection {
        Some(gate) => detect_connection_mode = detect_connection_mode.depends_on(&[gate]),
        None => {
            let startup_delay = Duration::from_secs(settings.startup_delay_secs);
            if !startup_delay.is_zero() {
                detect_connection_mode = detect_connection_mode.with_delay(startup_delay);
            }
        }
    }

    WorkflowDefinition::new(WORKER_REGISTRATION_WORKFLOW, "Worker Registration")
        .add_step(detect_connection_mode)
        .add_step(
            StepDefinition::new(
                DETECT_RUNTIME_STEP,
                "Detect Backend",
                Arc::new(DetectRuntimeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: startup_attempts(settings),
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&[DETECT_CONNECTION_MODE_STEP]),
        )
        .add_step(
            StepDefinition::new(
                DISCOVER_METADATA_STEP,
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&[DETECT_RUNTIME_STEP]),
        )
        .add_step(
            StepDefinition::new(
                DISCOVER_DP_STEP,
                "Discover DP Info",
                Arc::new(DiscoverDpInfoStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&[DISCOVER_METADATA_STEP]),
        )
        .add_step(
            StepDefinition::new(
                CREATE_LOCAL_WORKER_STEP,
                "Create Local Worker",
                Arc::new(CreateLocalWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&[gates.creation.unwrap_or(DISCOVER_DP_STEP)]),
        )
}

#[cfg(test)]
mod tests;
