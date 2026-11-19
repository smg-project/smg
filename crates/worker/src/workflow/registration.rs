//! The registration of a worker as a workflow: the steps of it that stand in
//! this crate, in this crate's order.

use std::{sync::Arc, time::Duration};

use wfaas::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition};

use super::{
    data::WorkerRegistrationWorkflowData, detect_connection::DetectConnectionModeStep,
    host::WorkflowHost,
};
use crate::settings::RegistrationSettings;

/// The id of the registration workflow.
pub const WORKER_REGISTRATION_WORKFLOW: &str = "worker_registration";
/// The id of the step that settles how a worker is spoken to: HTTP, and in
/// which version, or gRPC.
pub const DETECT_CONNECTION_MODE_STEP: &str = "detect_connection_mode";

/// The host's steps that this crate's steps of a registration wait on, one
/// field a gate. A gate names a step the host adds to the workflow; the
/// engine refuses a definition whose gate names a step nobody added.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegistrationGates {
    /// The host's step that ends before the connection detection starts, for
    /// a host that settles something about the worker first. With none, the
    /// detection opens the workflow.
    pub detection: Option<&'static str>,
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
/// the startup cadence of `settings`.
///
/// The startup delay, the time a worker is left alone before its first
/// probe, stands on the step that opens the workflow. With
/// `gates.detection` named that step is the host's, and the host sets the
/// delay there; the detection then waits on it and carries none. With no
/// gate the detection opens the workflow and carries the delay.
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
}

#[cfg(test)]
mod tests;
