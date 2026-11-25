//! The registration and removal data's cases.

use std::{collections::HashMap, sync::Arc};

use wfaas::WorkflowError;

use super::{
    DpInfo, WorkerKind, WorkerList, WorkerRegistrationWorkflowData, WorkerRemovalRequest,
    WorkerRemovalWorkflowData,
};
use crate::{
    identity::basic::ConnectionMode,
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    model_card::ModelCard,
    test_support::WorkerRegistry,
    worker::WorkerSpec,
    workflow::host::{testing::RegistryHost, WorkflowHost},
};

/// A registration with every saved field filled, one entry a map so that its
/// saved text is stable.
fn registration() -> WorkerRegistrationWorkflowData<RegistryHost> {
    WorkerRegistrationWorkflowData {
        config: WorkerSpec::new("http://worker-a:8000"),
        registration_mode: WorkerRegistrationMode::ReplaceById {
            worker_id: "w-1".to_string(),
            expected_revision: 7,
        },
        asker: WorkerJobAsker::Api,
        worker_kind: Some(WorkerKind::Local),
        connection_mode: Some(ConnectionMode::Grpc),
        http2: Some(true),
        http_client_handle: None,
        detected_runtime_type: Some("runtime-a".to_string()),
        discovered_labels: HashMap::from([("model_id".to_string(), "model-a".to_string())]),
        dp_info: Some(DpInfo {
            dp_size: 2,
            model_id: "model-a".to_string(),
        }),
        host_notes: (),
        model_cards: vec![ModelCard::new("model-a")],
        workers: Some(WorkerList {
            worker_urls: vec!["http://worker-a:8000".to_string()],
        }),
        final_labels: HashMap::from([("tier".to_string(), "a".to_string())]),
        host: None,
        actual_workers: None,
    }
}

fn host() -> Arc<RegistryHost> {
    Arc::new(RegistryHost {
        worker_registry: Arc::new(WorkerRegistry::new()),
    })
}

#[test]
fn the_held_client_is_handed_before_the_host_is_asked() {
    let mut data = registration();
    let held = host()
        .worker_client_cache()
        .get(&data.config.http_pool, true)
        .unwrap();
    data.http_client_handle = Some(Arc::clone(&held));

    // No host stands on the data: the held client answers by itself.
    let client = data.http_client("a_step").unwrap();

    assert!(Arc::ptr_eq(&client, &held));
}

#[test]
fn a_registration_read_back_takes_its_client_from_the_hosts_cache() {
    let mut data = registration();
    data.host = Some(host());

    let first = data.http_client("a_step").unwrap();
    let second = data.http_client("a_step").unwrap();

    // One cache, one pool configuration, one HTTP version: one client.
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn a_registration_with_no_host_and_no_held_client_fails_naming_the_host() {
    let data = registration();

    let error = data.http_client("a_step").unwrap_err();

    assert!(matches!(error, WorkflowError::ContextValueNotFound(name) if name == "host"));
}

const SAVED_REGISTRATION: &str = concat!(
    r#"{"config":{"url":"http://worker-a:8000","worker_type":"regular","connection_mode":"http","runtime_type":"unspecified","priority":50,"cost":1.0,"max_connection_attempts":20}"#,
    r#","registration_mode":{"ReplaceById":{"worker_id":"w-1","expected_revision":7}}"#,
    r#","asker":"Api""#,
    r#","worker_kind":"Local""#,
    r#","connection_mode":"grpc""#,
    r#","http2":true"#,
    r#","detected_runtime_type":"runtime-a""#,
    r#","discovered_labels":{"model_id":"model-a"}"#,
    r#","dp_info":{"dp_size":2,"model_id":"model-a"}"#,
    r#","model_cards":[{"id":"model-a","model_type":["chat","completions","responses","tools"]}]"#,
    r#","workers":{"worker_urls":["http://worker-a:8000"]}"#,
    r#","final_labels":{"tier":"a"}}"#,
);

#[test]
fn a_saved_registration_reads_back_with_its_asker() {
    assert_eq!(
        serde_json::to_string(&registration()).unwrap(),
        SAVED_REGISTRATION
    );

    // What was saved reads back whole, and what is never saved comes back
    // empty.
    let read: WorkerRegistrationWorkflowData<RegistryHost> =
        serde_json::from_str(SAVED_REGISTRATION).unwrap();
    assert_eq!(serde_json::to_string(&read).unwrap(), SAVED_REGISTRATION);
    assert_eq!(read.asker, WorkerJobAsker::Api);
    assert_eq!(read.clone().asker, WorkerJobAsker::Api);
    assert!(read.host.is_none());
    assert!(read.actual_workers.is_none());
    assert!(read.http_client_handle.is_none());
}

const SAVED_REMOVAL: &str = concat!(
    r#"{"config":{"url":"http://worker-a:8000","expected_revisions":{"w-1":7}},"asker":"Health","#,
    r#""workers_to_remove":null,"worker_urls":[],"affected_models":[]}"#,
);

#[test]
fn a_saved_removal_reads_back_with_its_asker() {
    let data: WorkerRemovalWorkflowData<RegistryHost> = WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest {
            url: "http://worker-a:8000".to_string(),
            expected_revisions: Some(HashMap::from([("w-1".to_string(), Some(7))])),
        },
        asker: WorkerJobAsker::Health,
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: Default::default(),
        host: Some(host()),
        actual_workers_to_remove: None,
        drained_workers: Vec::new(),
        removed_workers: Vec::new(),
    };

    assert_eq!(serde_json::to_string(&data).unwrap(), SAVED_REMOVAL);
    let read: WorkerRemovalWorkflowData<RegistryHost> =
        serde_json::from_str(SAVED_REMOVAL).unwrap();
    assert_eq!(serde_json::to_string(&read).unwrap(), SAVED_REMOVAL);
    assert_eq!(read.asker, WorkerJobAsker::Health);
    assert_eq!(read.clone().asker, WorkerJobAsker::Health);
    assert!(read.host.is_none());
}

#[test]
fn saved_worker_data_without_an_asker_is_refused() {
    let registration = SAVED_REGISTRATION.replace(r#","asker":"Api""#, "");
    let removal = SAVED_REMOVAL.replace(r#","asker":"Health""#, "");
    assert!(
        serde_json::from_str::<WorkerRegistrationWorkflowData<RegistryHost>>(&registration)
            .unwrap_err()
            .to_string()
            .contains("missing field `asker`")
    );
    assert!(
        serde_json::from_str::<WorkerRemovalWorkflowData<RegistryHost>>(&removal)
            .unwrap_err()
            .to_string()
            .contains("missing field `asker`")
    );
}

/// A registration enters the workflow with nothing found out about its worker.
#[test]
fn a_new_registration_holds_its_spec_mode_host_and_asker() {
    let host = host();

    let data = WorkerRegistrationWorkflowData::new(
        WorkerSpec::new("http://worker-a:8000"),
        WorkerRegistrationMode::Upsert,
        Arc::clone(&host),
        WorkerJobAsker::Startup,
    );

    assert_eq!(data.config.url, "http://worker-a:8000");
    assert!(matches!(
        data.registration_mode,
        WorkerRegistrationMode::Upsert
    ));
    assert!(Arc::ptr_eq(data.host.as_ref().unwrap(), &host));
    assert_eq!(data.asker, WorkerJobAsker::Startup);
    assert!(data.worker_kind.is_none());
    assert!(data.connection_mode.is_none());
    assert!(data.http2.is_none());
    assert!(data.http_client_handle.is_none());
    assert!(data.detected_runtime_type.is_none());
    assert!(data.discovered_labels.is_empty());
    assert!(data.dp_info.is_none());
    assert!(data.model_cards.is_empty());
    assert!(data.workers.is_none());
    assert!(data.final_labels.is_empty());
    assert!(data.actual_workers.is_none());
}
