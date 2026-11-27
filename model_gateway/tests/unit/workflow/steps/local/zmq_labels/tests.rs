use smg_worker::jobs::WorkerJobAsker;

use super::*;

#[tokio::test]
async fn zmq_labels_carry_the_local_generation_config_sampling_defaults() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        r#"{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05, "max_new_tokens": 2048, "do_sample": true}"#,
    )
    .unwrap();
    let labels = zmq_model_labels(Some(dir.path().to_str().unwrap())).await;
    let defaults: serde_json::Value =
        serde_json::from_str(&labels[DEFAULT_SAMPLING_PARAMS_LABEL]).unwrap();
    assert_eq!(defaults["top_k"], 20);
    assert_eq!(defaults["temperature"], 0.7);
    assert!(
        defaults.get("max_new_tokens").is_none(),
        "length limits are not sampling defaults"
    );
    // Only a worker serving the gateway's model takes its defaults.
    assert_eq!(
        zmq_model(Some("/m/a".into()), UNKNOWN_MODEL_ID).as_deref(),
        Some("/m/a")
    );
    assert_eq!(
        zmq_model(Some("/m/a".into()), "/m/a").as_deref(),
        Some("/m/a")
    );
    assert_eq!(zmq_model(Some("/m/a".into()), "org/other"), None);
    assert_eq!(zmq_model(None, UNKNOWN_MODEL_ID), None);
    // No model, or a directory without the file: no label.
    assert!(zmq_model_labels(None).await.is_empty());
    let bare = tempfile::tempdir().unwrap();
    assert!(zmq_model_labels(Some(bare.path().to_str().unwrap()))
        .await
        .is_empty());
    // A local file, or a path that is no repo id and does not exist here
    // (an engine-side mount), never goes to the Hub: no label.
    let file = tempfile::NamedTempFile::new().unwrap();
    assert!(zmq_model_labels(Some(file.path().to_str().unwrap()))
        .await
        .is_empty());
    assert!(zmq_model_labels(Some("/models/not/mounted/here"))
        .await
        .is_empty());
    assert!(is_hub_repo_id("Qwen/Qwen3-0.6B"));
    assert!(is_hub_repo_id("gpt2"));
    for not_id in [
        "/models/qwen",
        "./qwen",
        "~/qwen",
        "a/b/c",
        "models--Qwen--Qwen3-0.6B/snapshots/x",
        "",
        "org/",
    ] {
        assert!(!is_hub_repo_id(not_id), "{not_id}");
    }
}

/// A ZMQ worker's labels say its grammars start at the first generated
/// token (the gateway's own link holds them there), with or without the
/// model's sampling defaults, in the value the grammar-start reader names.
#[tokio::test]
async fn zmq_labels_say_the_links_grammar_start() {
    let labels = zmq_worker_labels(None).await;
    assert_eq!(labels.len(), 1, "{labels:?}");
    assert_eq!(
        GrammarStart::from_label(labels.get(GrammarStart::LABEL).map(String::as_str)),
        GrammarStart::FirstToken
    );
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        r#"{"temperature": 0.7, "top_p": 0.8}"#,
    )
    .unwrap();
    let labels = zmq_worker_labels(Some(dir.path().to_str().unwrap())).await;
    assert!(labels.contains_key(DEFAULT_SAMPLING_PARAMS_LABEL));
    assert_eq!(
        labels.get(GrammarStart::LABEL).map(String::as_str),
        Some(GrammarStart::FIRST_TOKEN)
    );
}

/// A ZMQ worker that declared no runtime takes the default in this step,
/// and one whose runtime is settled keeps it.
#[tokio::test]
async fn a_zmq_worker_with_no_runtime_takes_the_default_here() {
    use openai_protocol::worker::WorkerSpec;
    use smg_worker::jobs::WorkerRegistrationMode;
    use wfaas::WorkflowInstanceId;

    use crate::{
        app_context::testing::create_test_app_context, workflow::steps::create_worker_workflow_data,
    };

    async fn runtime_after(settled: Option<&str>) -> Option<String> {
        let mut data = create_worker_workflow_data(
            WorkerSpec::new("ipc:///tmp/engine-a.sock"),
            WorkerRegistrationMode::Upsert,
            create_test_app_context(),
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Zmq);
        data.detected_runtime_type = settled.map(str::to_string);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        DiscoverZmqLabelsStep.execute(&mut context).await.unwrap();
        context.data.detected_runtime_type
    }

    assert_eq!(runtime_after(None).await.as_deref(), Some("vllm"));
    assert_eq!(
        runtime_after(Some("tokenspeed")).await.as_deref(),
        Some("tokenspeed")
    );
}
