//! The registration order's cases.

use std::time::Duration;

use wfaas::{
    BackoffStrategy, StepDefinition, StepId, ValidationError, WorkflowDefinition, WorkflowEngine,
};

use super::*;
use crate::workflow::host::testing::{registration_settings, RegistryHost};

type Data = WorkerRegistrationWorkflowData<RegistryHost>;

const HOST_GATE_STEP: &str = "host_gate";

fn settings(timeout_secs: u64, delay_secs: u64, interval_secs: u64) -> RegistrationSettings {
    RegistrationSettings {
        startup_timeout_secs: timeout_secs,
        startup_delay_secs: delay_secs,
        startup_check_interval_secs: interval_secs,
        ..registration_settings()
    }
}

fn detection(workflow: &WorkflowDefinition<Data>) -> &StepDefinition<Data> {
    workflow
        .steps
        .iter()
        .find(|step| step.id == StepId::new(DETECT_CONNECTION_MODE_STEP))
        .unwrap()
}

/// With a gate named, the detection waits on the host's step and carries no
/// delay: the delay stands on the step that opens the workflow, the host's.
/// The engine refuses the definition until the host has added that step.
#[test]
fn a_named_gate_holds_the_detection_behind_the_hosts_step() {
    let mut workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 7, 5),
        RegistrationGates {
            detection: Some(HOST_GATE_STEP),
            ..RegistrationGates::default()
        },
    );

    let step = detection(&workflow);
    assert_eq!(step.depends_on, vec![StepId::new(HOST_GATE_STEP)]);
    assert_eq!(step.delay, None);
    assert!(workflow.validate().is_err());
}

/// With no gate the detection opens the workflow and carries the startup
/// delay. A delay of zero is none.
#[test]
fn with_no_gate_the_detection_opens_the_workflow_and_carries_the_delay() {
    let mut workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 7, 5),
        RegistrationGates::default(),
    );

    let step = detection(&workflow);
    assert!(step.depends_on.is_empty());
    assert_eq!(step.delay, Some(Duration::from_secs(7)));
    assert!(workflow.validate().is_ok());

    let immediate = worker_registration_workflow::<RegistryHost>(
        &settings(300, 0, 5),
        RegistrationGates::default(),
    );
    assert_eq!(detection(&immediate).delay, None);
}

/// The detection polls at the startup cadence: one probe a check interval,
/// for the budget of [`startup_attempts`], inside the startup timeout.
#[test]
fn the_detection_polls_at_the_startup_cadence() {
    let workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 0, 5),
        RegistrationGates::default(),
    );

    let step = detection(&workflow);
    let retry = step.retry_policy.as_ref().unwrap();
    assert_eq!(retry.max_attempts, 54);
    assert!(
        matches!(retry.backoff, BackoffStrategy::Fixed(pause) if pause == Duration::from_secs(5))
    );
    assert_eq!(step.timeout, Some(Duration::from_secs(300)));
}

/// The budget of attempts is nine tenths of the startup timeout over the
/// check interval, rounded up, and never under three. An interval of zero
/// counts as one second.
#[test]
fn the_budget_of_attempts_follows_the_startup_cadence() {
    for (timeout_secs, interval_secs, attempts) in [
        (300, 5, 54),
        (301, 5, 55),
        (10, 5, 3),
        (0, 5, 3),
        (100, 0, 90),
    ] {
        assert_eq!(
            startup_attempts(&settings(timeout_secs, 0, interval_secs)),
            attempts,
            "{timeout_secs} s, a probe every {interval_secs} s"
        );
    }
}

/// The runtime detection waits on the connection detection and polls with
/// the same budget of attempts.
#[test]
fn the_runtime_detection_waits_on_the_connection_detection() {
    let workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 7, 5),
        RegistrationGates::default(),
    );

    let step = workflow
        .steps
        .iter()
        .find(|step| step.id == StepId::new(DETECT_RUNTIME_STEP))
        .unwrap();
    assert_eq!(
        step.depends_on,
        vec![StepId::new(DETECT_CONNECTION_MODE_STEP)]
    );
    assert_eq!(step.delay, None);
    assert_eq!(step.retry_policy.as_ref().unwrap().max_attempts, 54);
    assert_eq!(step.timeout, Some(Duration::from_secs(10)));
}

/// The metadata read waits on the runtime detection, with three attempts of
/// ten seconds each.
#[test]
fn the_metadata_read_waits_on_the_runtime_detection() {
    let workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 7, 5),
        RegistrationGates::default(),
    );

    let step = workflow
        .steps
        .iter()
        .find(|step| step.id == StepId::new(DISCOVER_METADATA_STEP))
        .unwrap();
    assert_eq!(step.depends_on, vec![StepId::new(DETECT_RUNTIME_STEP)]);
    assert_eq!(step.retry_policy.as_ref().unwrap().max_attempts, 3);
    assert_eq!(step.timeout, Some(Duration::from_secs(10)));
}

/// The DP read waits on the metadata read alone, whatever gate the host
/// names. It fails the workflow when it fails, after three attempts of ten
/// seconds each.
#[test]
fn the_dp_read_waits_on_the_metadata_read() {
    for gates in [
        RegistrationGates::default(),
        RegistrationGates {
            detection: Some(HOST_GATE_STEP),
            ..RegistrationGates::default()
        },
    ] {
        let workflow = worker_registration_workflow::<RegistryHost>(&settings(300, 7, 5), gates);

        let step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new(DISCOVER_DP_STEP))
            .unwrap();
        assert_eq!(step.depends_on, vec![StepId::new(DISCOVER_METADATA_STEP)]);
        assert_eq!(step.delay, None);
        assert_eq!(step.retry_policy.as_ref().unwrap().max_attempts, 3);
        assert_eq!(step.timeout, Some(Duration::from_secs(10)));
        assert!(matches!(step.on_failure, FailureAction::FailWorkflow));
    }
}

#[test]
fn the_creation_waits_on_the_dp_read_or_on_the_hosts_gate() {
    for (creation, dependency) in [
        (None, DISCOVER_DP_STEP),
        (Some(HOST_GATE_STEP), HOST_GATE_STEP),
    ] {
        let workflow = worker_registration_workflow::<RegistryHost>(
            &settings(300, 7, 5),
            RegistrationGates {
                creation,
                ..RegistrationGates::default()
            },
        );
        let step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("create_local_worker"))
            .expect("the creation step is in the workflow");

        assert_eq!(step.depends_on, [StepId::new(dependency)]);
        assert!(step.depends_on_any.is_empty());
        assert_eq!(step.name, "Create Local Worker");
        assert_eq!(step.timeout, Some(Duration::from_secs(5)));
        assert!(step.retry_policy.is_none());
        assert!(matches!(step.on_failure, FailureAction::FailWorkflow));
    }
}

#[test]
fn a_creation_gate_must_name_a_step_the_host_added() {
    let workflow = worker_registration_workflow::<RegistryHost>(
        &settings(300, 7, 5),
        RegistrationGates {
            creation: Some(HOST_GATE_STEP),
            ..RegistrationGates::default()
        },
    );

    let error = WorkflowEngine::new()
        .register_workflow(workflow)
        .unwrap_err();

    assert!(matches!(
        error,
        ValidationError::MissingDependency { step, dependency }
            if step == StepId::new("create_local_worker") && dependency == StepId::new(HOST_GATE_STEP)
    ));
}
