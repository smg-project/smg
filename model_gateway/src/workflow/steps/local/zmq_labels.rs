//! What the gateway knows of a ZMQ worker ahead of its handshake: the runtime
//! it defaults to, and the labels of the model the gateway was started with.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use openai_protocol::{worker::GrammarStart, UNKNOWN_MODEL_ID};
use smg_worker::{
    identity::{
        basic::ConnectionMode,
        endpoint::Endpoint,
        sampling_defaults::{SamplingDefaults, DEFAULT_SAMPLING_PARAMS_LABEL},
    },
    workflow::{resolve_model_id, WorkerKind},
};
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowResult};

use crate::workflow::data::WorkerWorkflowData;

/// The id of the step that writes a ZMQ worker's runtime default and labels.
pub(crate) const ZMQ_LABELS_STEP: &str = "discover_zmq_labels";

/// The gateway's model (a directory or a Hub id), when this worker serves it: a
/// worker with no identity of its own takes the gateway's (as registration
/// does), one declaring a different model keeps its own defaults.
fn zmq_model(router_model: Option<String>, declared_model: &str) -> Option<String> {
    router_model.filter(|path| declared_model == UNKNOWN_MODEL_ID || declared_model == path)
}

/// The labels of a ZMQ worker before its handshake: the model's sampling
/// defaults ([`zmq_model_labels`]) and where its grammars start. The gateway's
/// own link fronts an engine it did not launch, whose handshake does not say
/// whether a reasoning parser gates its grammars, so the worker says the start
/// the gateway's tags are shaped for: the first generated token, the thought
/// closed ahead of the calls or the schema (the `constrained_decoding_start`
/// label a servicer's worker carries for its engine's real gate). An engine
/// that starts there applies such a tag at once; one that gates behind a
/// parser is told by the request that this grammar starts at the first token
/// (`GenerateRequestBuildOptions::first_token_grammar`), and a grammar the
/// gateway did not shape starts where the engine does, on either kind.
async fn zmq_worker_labels(model: Option<&str>) -> HashMap<String, String> {
    let mut labels = zmq_model_labels(model).await;
    labels.insert(
        GrammarStart::LABEL.to_string(),
        GrammarStart::FIRST_TOKEN.to_string(),
    );
    labels
}

/// Labels for a ZMQ worker from the gateway's model: the sampling defaults in
/// its `generation_config.json`. A local directory is read in place; a Hub
/// repo id is served from the HF cache the gateway's tokenizer loads from,
/// fetched into it when absent, within a bound that keeps registration short
/// on an offline gateway. A local file, a path that does not exist on this
/// host, or a model without the file yields none.
async fn zmq_model_labels(model: Option<&str>) -> HashMap<String, String> {
    let Some(model) = model else {
        return HashMap::new();
    };
    let local = std::path::Path::new(model);
    let path = if local.is_dir() {
        local.join("generation_config.json")
    } else if local.exists() || !is_hub_repo_id(model) {
        return HashMap::new();
    } else {
        let fetch = llm_tokenizer::hub::fetch_file(model, "generation_config.json");
        match tokio::time::timeout(HUB_FETCH_TIMEOUT, fetch).await {
            Ok(Ok(path)) => path,
            Ok(Err(e)) => {
                debug!("No generation_config.json for {model} via the Hub cache: {e}");
                return HashMap::new();
            }
            Err(_) => {
                warn!("Hub lookup of generation_config.json for {model} timed out; no sampling defaults");
                return HashMap::new();
            }
        }
    };
    sampling_defaults_label(&path).await
}

/// Bound on fetching `generation_config.json` from the Hub, inside the step's
/// own timeout so an unreachable Hub costs one wait, not the step's retries.
const HUB_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// `owner/name`, or a bare legacy `name`: no leading `/`, `.` or `~` and at
/// most one separator, so a path meant for another host is never sent to the
/// Hub as a repo id.
fn is_hub_repo_id(model: &str) -> bool {
    !model.starts_with(['/', '.', '~'])
        && model.matches('/').count() <= 1
        && model.split('/').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

/// The sampling-defaults label from one `generation_config.json`, or none.
async fn sampling_defaults_label(path: &std::path::Path) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    match tokio::fs::read_to_string(path).await {
        Ok(raw) => match SamplingDefaults::canonical_json_from_str(&raw) {
            Ok(Some(canonical)) => {
                labels.insert(DEFAULT_SAMPLING_PARAMS_LABEL.to_string(), canonical);
            }
            Ok(None) => {}
            Err(e) => warn!("Ignoring {}: {}", path.display(), e),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!("Could not read {}: {}", path.display(), e),
    }
    labels
}

/// Writes what the gateway knows of a ZMQ worker ahead of its handshake. It
/// runs behind the worker crate's detection, whose metadata read reads
/// nothing for a ZMQ worker, and is skipped for every other worker. The
/// worker's creation waits on it.
///
/// A ZMQ engine reports its geometry at the handshake, which runs after the
/// registration's detection; the facts worth having earlier are the model's
/// own sampling defaults, read from the model the gateway was started with
/// (a local directory, or a Hub id through the HF cache), for a worker that
/// serves that model (one registered for another model, by spec or label,
/// must not inherit them), and where the gateway's own link starts a
/// grammar, which is the link's rule and needs no engine.
pub struct DiscoverZmqLabelsStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverZmqLabelsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local)
            || context.data.connection_mode != Some(ConnectionMode::Zmq)
        {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        // A ZMQ EngineCore handshake is shared across engine runtimes, so the
        // runtime cannot be probed. A declared `runtime_type` (vllm, sglang,
        // tokenspeed, ...) is settled by the runtime detection; only a worker
        // that left it unspecified reaches this default of vLLM. The default
        // is written ahead of the label read, so it stands where that read
        // does not end: the engine keeps what an attempt wrote.
        if context.data.detected_runtime_type.is_none() {
            warn!(
                worker = %Endpoint::redacted_str(&config.url),
                "runtime_type unspecified for ZMQ worker; defaulting to vLLM \
                 EngineCore. A TokenSpeed or SGLang engine must declare its \
                 runtime to speak the correct wire protocol: `--backend \
                 tokenspeed` / `--backend sglang` for startup --worker-urls \
                 workers, or an explicit runtime_type on the worker API spec \
                 / YAML worker config."
            );
            context.data.detected_runtime_type = Some("vllm".to_string());
        }

        let router_model = context
            .data
            .host
            .as_ref()
            .and_then(|app| app.router_config.model_path.clone());
        let model = zmq_model(router_model, resolve_model_id(config, &config.labels));
        let labels = zmq_worker_labels(model.as_deref()).await;

        debug!(
            "Discovered {} labels for {}",
            labels.len(),
            Endpoint::redacted_str(&config.url)
        );
        context.data.discovered_labels = labels;
        Ok(StepResult::Success)
    }
}

#[cfg(test)]
mod tests {
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
            app_context::testing::create_test_app_context,
            workflow::steps::create_worker_workflow_data,
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
}
