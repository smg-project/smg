use std::sync::atomic::{AtomicUsize, Ordering};

use openai_protocol::{model_card::ModelCard, worker::WorkerType};
use smg_worker::test_support::no_health_check;
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    app_context::testing::app_context_for_tests,
    config::RouterConfig,
    routers::grpc::{backend_client::BasicWorkerBuilder, multimodal::HubEnv},
};

#[test]
fn tokenizer_sources_prefer_ready_and_skip_unhealthy_workers() {
    use WorkerStatus::{Draining, Failed, NotReady, Pending, Ready};

    let registry = WorkerRegistry::new();
    for status in [Failed, Pending, Draining, NotReady, Ready] {
        let worker = BasicWorkerBuilder::new(format!("grpc://{status}:30000"))
            .model(ModelCard::new("m"))
            .connection_mode(ConnectionMode::Grpc)
            .status(status)
            .build();
        registry.register(Arc::new(worker)).unwrap();
    }
    let workers = tokenizer_workers(&registry, "m");
    let statuses: Vec<_> = workers.iter().map(|worker| worker.status()).collect();
    assert_eq!(statuses, [Ready, Pending]);
}

/// A bundle as a worker serves it: a zip archive of `files` and its
/// SHA-256.
fn bundle_of<const N: usize>(files: [(&str, String); N]) -> StreamBundle {
    use std::io::{Cursor, Write};

    use sha2::{Digest, Sha256};
    use zip::{write::SimpleFileOptions, ZipWriter};

    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data) in files {
        archive
            .start_file(name, SimpleFileOptions::default())
            .unwrap();
        archive.write_all(data.as_bytes()).unwrap();
    }
    let compressed_data = archive.finish().unwrap().into_inner();
    StreamBundle {
        sha256: Sha256::digest(&compressed_data)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        compressed_data,
    }
}

#[test]
fn worker_tokenizer_bundle_preserves_configured_template() {
    let bundle = bundle_of([
        (
            "tokenizer.json",
            serde_json::json!({
                "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null, "decoder":null,
                "model":{"type":"BPE", "vocab":{"hello":0}, "merges":[]}
            })
            .to_string(),
        ),
        (
            "tokenizer_config.json",
            serde_json::json!({"chat_template":"worker_default"}).to_string(),
        ),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let template = dir.path().join("override.jinja");
    std::fs::write(&template, "router_override").unwrap();
    let messages = [serde_json::json!({"role":"user", "content":"hello"})];
    let (default, _) = load_tokenizer_from_bundle(&bundle, None).unwrap();
    assert_eq!(
        default
            .apply_chat_template(&messages, Default::default())
            .unwrap(),
        "worker_default"
    );
    let (overridden, _) = load_tokenizer_from_bundle(&bundle, template.to_str()).unwrap();
    assert_eq!(
        overridden
            .apply_chat_template(&messages, Default::default())
            .unwrap(),
        "router_override"
    );
    let missing = dir.path().join("missing.jinja");
    assert!(
        load_tokenizer_from_bundle(&bundle, missing.to_str()).is_err(),
        "an invalid configured override must not silently use the worker default"
    );
}

#[test]
fn test_tokenizer_config_request_serialization() {
    let config = TokenizerConfigRequest {
        id: "test-uuid-1234".to_string(),
        name: "test-model".to_string(),
        source: "meta-llama/Llama-2-7b-hf".to_string(),
        chat_template_path: None,
        cache_config: None,
        fail_on_duplicate: false,
        for_worker: false,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, "test-uuid-1234");
    assert_eq!(parsed.name, "test-model");
    assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
    assert!(parsed.chat_template_path.is_none());
    assert!(parsed.cache_config.is_none());
    assert!(!parsed.fail_on_duplicate);
}

#[test]
fn test_tokenizer_config_request_fail_on_duplicate_defaults_to_false() {
    // Test that fail_on_duplicate defaults to false when not specified in JSON
    let json = r#"{
            "id": "test-uuid",
            "name": "test-model",
            "source": "/path/to/tokenizer"
        }"#;
    let parsed: TokenizerConfigRequest = serde_json::from_str(json).unwrap();
    assert!(!parsed.fail_on_duplicate);
}

#[test]
fn test_tokenizer_config_request_fail_on_duplicate_true() {
    let config = TokenizerConfigRequest {
        id: "test-uuid-1234".to_string(),
        name: "test-model".to_string(),
        source: "meta-llama/Llama-2-7b-hf".to_string(),
        chat_template_path: None,
        cache_config: None,
        fail_on_duplicate: true,
        for_worker: false,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();
    assert!(parsed.fail_on_duplicate);
}

#[test]
fn test_tokenizer_config_request_with_cache() {
    let config = TokenizerConfigRequest {
        id: "test-uuid-1234".to_string(),
        name: "test-model".to_string(),
        source: "meta-llama/Llama-2-7b-hf".to_string(),
        chat_template_path: None,
        cache_config: Some(TokenizerCacheConfig {
            enable_l0: true,
            l0_max_entries: 1000,
            l0_max_memory: 256 * 1024 * 1024,
            enable_l1: false,
            l1_max_memory: 0,
        }),
        fail_on_duplicate: false,
        for_worker: false,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

    assert!(parsed.cache_config.is_some());
    let cache = parsed.cache_config.unwrap();
    assert!(cache.enable_l0);
    assert_eq!(cache.l0_max_entries, 1000);
    assert!(!cache.enable_l1);
}

#[test]
fn test_workflow_creation() {
    let mut workflow = create_tokenizer_registration_workflow();
    assert_eq!(workflow.id.to_string(), "tokenizer_registration");
    // Validate the workflow DAG
    workflow
        .validate()
        .expect("Workflow validation should pass");
}

/// A worker registry serving `model`.
fn serving(model: &str) -> WorkerRegistry {
    let workers = WorkerRegistry::new();
    workers
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://worker:30000")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new(model))
                .health_config(no_health_check())
                .build(),
        ))
        .expect("register worker");
    workers
}

/// A failed load is recorded once per model and entry (the log line's
/// allowance), a different reason is a new entry, and the next load,
/// accepted or succeeding, clears the record.
#[test]
fn a_load_failure_is_recorded_once_per_entry_and_cleared_by_the_next_load() {
    let failures = TokenizerLoadFailures::new();
    let registry = TokenizerRegistry::new();
    let workers = serving("m");
    let error = "Workflow failed at step load_tokenizer: boom".to_string();
    assert_eq!(failures.get("m"), None);

    failures.load_started("m");
    assert_eq!(failures.get("m"), None);
    note_tokenizer_load_outcome(&failures, &registry, &workers, "m", &Err(error.clone()));
    assert_eq!(
        failures.get("m"),
        Some(TokenizerLoadFailure::Failed(error.clone()))
    );
    assert!(!failures.record("m", TokenizerLoadFailure::Failed(error.clone())));
    assert!(failures.record("m", TokenizerLoadFailure::NoSource));
    assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::NoSource));
    assert_eq!(failures.get("other"), None);

    failures.load_started("m");
    assert_eq!(failures.get("m"), None, "a load accepted: pending again");
    note_tokenizer_load_outcome(
        &failures,
        &registry,
        &workers,
        "m",
        &Ok("m completed".to_string()),
    );
    assert_eq!(failures.get("m"), None);
    assert!(!failures.clear("m"));
}

/// While another load of the model is in flight, neither a registration
/// without a path nor an earlier load's failure is recorded: the last
/// load decides.
#[test]
fn a_failure_is_not_recorded_while_another_load_of_the_model_is_in_flight() {
    let failures = TokenizerLoadFailures::new();
    let registry = TokenizerRegistry::new();
    let workers = serving("m");

    failures.load_started("m");
    assert!(!failures.record("m", TokenizerLoadFailure::NoSource));
    assert_eq!(failures.get("m"), None);

    failures.load_started("m");
    note_tokenizer_load_outcome(
        &failures,
        &registry,
        &workers,
        "m",
        &Err("first".to_string()),
    );
    assert_eq!(failures.get("m"), None, "a load is still in flight");
    note_tokenizer_load_outcome(
        &failures,
        &registry,
        &workers,
        "m",
        &Err("second".to_string()),
    );
    assert_eq!(
        failures.get("m"),
        Some(TokenizerLoadFailure::Failed("second".to_string()))
    );
}

/// A load that ended without a count (none is taken outside the queue's
/// hand-over) leaves no state behind, and the model records as before.
#[test]
fn a_load_that_ended_without_a_count_leaves_nothing_behind() {
    let failures = TokenizerLoadFailures::new();
    assert!(failures.load_ended("m"));
    assert_eq!(failures.get("m"), None);
    assert!(failures.record("m", TokenizerLoadFailure::NoSource));
    assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::NoSource));
    failures.load_started("m");
    assert_eq!(
        failures.get("m"),
        None,
        "pending while the load is in flight"
    );
    assert!(failures.load_ended("m"));
    assert_eq!(
        failures.get("m"),
        None,
        "the load that ended left no record"
    );
}

/// A failed load of a name no served model carries (the start-up load,
/// named by its source; an operator's registration of another name) is
/// nobody's request: nothing is recorded.
#[test]
fn a_failed_load_of_a_name_nobody_serves_is_not_recorded() {
    let failures = TokenizerLoadFailures::new();
    let registry = TokenizerRegistry::new();
    let workers = WorkerRegistry::new();
    failures.load_started("/models/x");
    note_tokenizer_load_outcome(
        &failures,
        &registry,
        &workers,
        "/models/x",
        &Err("boom".to_string()),
    );
    assert_eq!(failures.get("/models/x"), None);
}

/// A registration that fails for a model whose tokenizer the registry
/// holds (a duplicate) takes nothing away and is not recorded.
#[tokio::test]
async fn a_failed_duplicate_registration_of_a_registered_tokenizer_is_no_failure() {
    let failures = TokenizerLoadFailures::new();
    let registry = TokenizerRegistry::new();
    let workers = serving("m");
    registry
        .load("id", "m", "mock", || async {
            factory::create_tokenizer_with_chat_template("mock", None).map_err(|e| e.to_string())
        })
        .await
        .expect("the mock tokenizer loads");
    failures.load_started("m");
    note_tokenizer_load_outcome(
        &failures,
        &registry,
        &workers,
        "m",
        &Err("Tokenizer 'm' already exists (id: id)".to_string()),
    );
    assert_eq!(failures.get("m"), None);
}

/// A tokenizer removed while its model is served is recorded, once; one
/// of a name nobody serves is not.
#[test]
fn a_removed_tokenizer_is_recorded_only_for_a_served_model() {
    let failures = TokenizerLoadFailures::new();
    let workers = serving("m");
    note_tokenizer_removed(&failures, &workers, "m");
    assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::Removed));
    assert!(!failures.record("m", TokenizerLoadFailure::Removed));
    note_tokenizer_removed(&failures, &workers, "other");
    assert_eq!(failures.get("other"), None);
}

/// The application context the step runs on, its worker registry given.
fn app_context(workers: WorkerRegistry) -> Arc<AppContext> {
    app_context_for_tests(
        RouterConfig::builder()
            .round_robin_policy()
            .build_unchecked(),
        Arc::new(workers),
        None,
    )
}

/// The registration step run for the tokenizer of `source` under `id`
/// and `name`.
async fn load(app_context: &Arc<AppContext>, id: &str, name: &str, source: &str, for_worker: bool) {
    let request = TokenizerConfigRequest {
        id: id.to_string(),
        name: name.to_string(),
        source: source.to_string(),
        chat_template_path: None,
        cache_config: None,
        fail_on_duplicate: false,
        for_worker,
    };
    let mut context = WorkflowContext::new(
        WorkflowInstanceId::new(),
        create_tokenizer_workflow_data(request, Arc::clone(app_context)),
    );
    let result = LoadTokenizerStep
        .execute(&mut context)
        .await
        .expect("the step runs");
    assert_eq!(result, StepResult::Success);
}

/// The record follows a worker's fresh load alone: the entry loaded
/// afresh for a worker is recorded, one the registry already held is
/// not, and a startup or operator load records nothing and stays
/// whether or not a worker serves its model.
#[tokio::test]
async fn the_record_follows_a_workers_fresh_load_alone() {
    let dir = test_support::tokenizer_dir();
    let source = dir.path().to_str().unwrap();
    let app_context = app_context(serving("m"));
    load(&app_context, "t1", "m", source, true).await;
    assert!(app_context.worker_loaded_tokenizers.contains("t1"));

    load(&app_context, "t2", "m", source, true).await;
    assert_eq!(
        app_context.tokenizer_registry.get_by_name("m").unwrap().id,
        "t1",
        "the entry the registry held stays"
    );
    assert!(!app_context.worker_loaded_tokenizers.contains("t2"));
    assert!(app_context.worker_loaded_tokenizers.contains("t1"));

    load(&app_context, "s1", "n", source, false).await;
    assert!(
        app_context.tokenizer_registry.get_by_name("n").is_some(),
        "a startup or operator entry stays for a model no worker serves"
    );
    assert!(!app_context.worker_loaded_tokenizers.contains("s1"));
}

/// A worker's load that publishes after the model's last worker left
/// (that removal's release found no entry to take) releases its entry
/// at once, record and all.
#[tokio::test]
async fn a_workers_load_that_publishes_after_its_last_worker_left_releases_the_entry() {
    let dir = test_support::tokenizer_dir();
    let app_context = app_context(WorkerRegistry::new());
    load(&app_context, "t1", "m", dir.path().to_str().unwrap(), true).await;
    assert!(app_context.tokenizer_registry.get_by_name("m").is_none());
    assert_eq!(app_context.tokenizer_registry.len(), 0);
    assert!(!app_context.worker_loaded_tokenizers.contains("t1"));
}

/// A failed job keeps the fetch error: it names the first failure, not
/// one for every worker tried.
#[tokio::test]
async fn a_failed_fetch_names_its_first_failure_alone() {
    let workers = WorkerRegistry::new();
    // The ready worker is tried first, though it registers second.
    for (url, status) in [
        ("grpc://127.0.0.1:9", WorkerStatus::Pending),
        ("grpc://127.0.0.1:11", WorkerStatus::Ready),
    ] {
        workers
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .model(ModelCard::new("m"))
                    .health_config(no_health_check())
                    .status(status)
                    .build(),
            ))
            .expect("register worker");
    }
    let error = fetch_tokenizer_from_worker(&app_context(workers), "t", "m", None)
        .await
        .err()
        .expect("no worker serves a tokenizer");
    assert!(
        error.contains("from 2 ready or pending gRPC worker(s), first failure: "),
        "{error}"
    );
    assert!(error.contains("grpc://127.0.0.1:11 (runtime"), "{error}");
    assert!(!error.contains("grpc://127.0.0.1:9 (runtime"), "{error}");
}

/// A request for the tokenizer of `source` under `name`, nothing set
/// beside it.
fn request(name: &str, source: &std::path::Path) -> TokenizerConfigRequest {
    TokenizerConfigRequest {
        id: TokenizerRegistry::generate_id(),
        name: name.to_string(),
        source: source.to_str().unwrap().to_string(),
        chat_template_path: None,
        cache_config: None,
        fail_on_duplicate: false,
        for_worker: false,
    }
}

/// The registration step run for `request`; the entry it left.
async fn registered(
    app_context: &Arc<AppContext>,
    request: TokenizerConfigRequest,
) -> llm_tokenizer::registry::TokenizerEntry {
    let name = request.name.clone();
    let mut context = WorkflowContext::new(
        WorkflowInstanceId::new(),
        create_tokenizer_workflow_data(request, Arc::clone(app_context)),
    );
    let result = LoadTokenizerStep
        .execute(&mut context)
        .await
        .expect("the step runs");
    assert_eq!(result, StepResult::Success);
    app_context.tokenizer_registry.get_by_name(&name).unwrap()
}

/// What the tokenizer of `entry` renders of one user message, `hi`.
fn rendered(entry: &llm_tokenizer::registry::TokenizerEntry) -> String {
    use llm_tokenizer::chat_template::ChatTemplateParams;

    let messages = [serde_json::json!({"role": "user", "content": "hi"})];
    entry
        .tokenizer
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap()
}

/// Two names over one tokenizer directory hold one tokenizer, each
/// under its own entry and id.
#[tokio::test]
async fn two_names_over_one_tokenizer_directory_hold_one_tokenizer() {
    let dir = test_support::tokenizer_dir();
    let app_context = app_context(WorkerRegistry::new());

    let first = registered(&app_context, request("first", dir.path())).await;
    let second = registered(&app_context, request("second", dir.path())).await;

    assert!(Arc::ptr_eq(&first.tokenizer, &second.tokenizer));
    assert_ne!(first.id, second.id);
    assert_eq!(app_context.tokenizer_registry.len(), 2);
}

/// A count of the factory's log lines written on this thread from here
/// on, with the guard that keeps the count running: a test runs on one
/// runtime thread, and a load that parsed on it would write them there.
fn factory_lines_on_this_thread() -> (Arc<AtomicUsize>, tracing::subscriber::DefaultGuard) {
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    struct FactoryLines(Arc<AtomicUsize>);
    impl<S: tracing::Subscriber> Layer<S> for FactoryLines {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            if event.metadata().target().starts_with("llm_tokenizer") {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    let lines = Arc::new(AtomicUsize::new(0));
    let on_this_thread = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(FactoryLines(Arc::clone(&lines))),
    );
    (lines, on_this_thread)
}

#[tokio::test]
async fn a_load_from_files_parses_off_the_runtime_thread() {
    let dir = test_support::tokenizer_dir();
    let source = dir.path().to_str().unwrap();
    let (lines, _on_this_thread) = factory_lines_on_this_thread();

    // The factory logs as it loads a directory: the zero below is a
    // count of something.
    factory::create_tokenizer_with_chat_template(source, None).unwrap();
    assert!(lines.swap(0, Ordering::SeqCst) > 0);

    load_from_source(source, None).await.unwrap();
    assert_eq!(lines.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_load_from_a_workers_bundle_parses_off_the_runtime_thread() {
    let dir = test_support::tokenizer_dir();
    let tokenizer = std::fs::read_to_string(dir.path().join("tokenizer.json")).unwrap();
    let bundle = bundle_of([("tokenizer.json", tokenizer)]);
    let app_context = app_context(WorkerRegistry::new());
    let (lines, _on_this_thread) = factory_lines_on_this_thread();

    // As above: the factory logs as it loads an unpacked bundle.
    load_tokenizer_from_bundle(&bundle, None).unwrap();
    assert!(lines.swap(0, Ordering::SeqCst) > 0);

    install_bundle(&app_context, "t", "m", bundle, None)
        .await
        .unwrap();
    assert_eq!(lines.load(Ordering::SeqCst), 0);
}

/// A load that panics on the pool unwinds the step's task, as it did when
/// it ran there: it is not turned into a failed load the step retries.
#[tokio::test]
#[should_panic(expected = "the load panicked")]
async fn a_load_that_panics_on_the_pool_unwinds_its_step() {
    on_the_pool(|| panic!("the load panicked")).await;
}

/// A path that is not on this host fails in the factory's words, as it
/// did before: the hub is not asked for it.
#[tokio::test]
async fn a_path_that_is_not_there_fails_as_it_did_on_the_runtime() {
    let error = load_from_source("/no/such/tokenizer", None)
        .await
        .err()
        .expect("nothing stands at the path");
    assert_eq!(error.to_string(), "File not found: /no/such/tokenizer");
}

/// A download that never ends stops with its step: the engine drops the
/// step's future when the step times out, and the request the load
/// holds at the Hub is closed with it, so nothing goes on downloading
/// behind the step. No network: the Hub is a loopback listener that
/// takes the request and never answers.
#[tokio::test]
async fn a_download_that_never_ends_stops_with_its_step() {
    use tokio::io::AsyncReadExt;

    const REPO: &str = "cell/never-answers";
    let home = tempfile::tempdir().unwrap();
    let hub = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", hub.local_addr().unwrap());
    let _env = HubEnv::set(home.path(), &endpoint).await;

    let app_context = app_context(WorkerRegistry::new());
    let mut context = WorkflowContext::new(
        WorkflowInstanceId::new(),
        create_tokenizer_workflow_data(
            request("never-answers", std::path::Path::new(REPO)),
            Arc::clone(&app_context),
        ),
    );
    let mut step = Box::pin(LoadTokenizerStep.execute(&mut context));

    // The step runs until the Hub holds the download's request. The
    // Hub's variables are process-wide, so a request of another test may
    // come first: it is let go.
    let request_for_repo = async {
        loop {
            let (mut socket, _) = hub.accept().await.unwrap();
            let mut head = [0; 1024];
            let read = socket.read(&mut head).await.unwrap_or(0);
            if String::from_utf8_lossy(&head[..read]).contains(REPO) {
                break socket;
            }
        }
    };
    let mut asked = tokio::select! {
        socket = request_for_repo => socket,
        ended = &mut step => panic!("the step ended before the Hub was asked: {ended:?}"),
    };

    // The step times out: the engine drops its future.
    drop(step);

    let mut rest = Vec::new();
    let closed = tokio::time::timeout(Duration::from_secs(5), asked.read_to_end(&mut rest)).await;
    assert!(
        closed.is_ok(),
        "the download holds its request at the Hub behind the step that timed out"
    );
}

/// Two spellings of one directory are two sources: each name holds its
/// own tokenizer.
#[tokio::test]
async fn two_spellings_of_one_directory_hold_two_tokenizers() {
    let dir = test_support::tokenizer_dir();
    let app_context = app_context(WorkerRegistry::new());

    let first = registered(&app_context, request("first", dir.path())).await;
    let second = registered(&app_context, request("second", &dir.path().join("."))).await;

    assert!(!Arc::ptr_eq(&first.tokenizer, &second.tokenizer));
}

/// Two names registered at once over one directory end with one
/// tokenizer: the second load waits for the first and takes its
/// tokenizer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_names_loading_one_directory_at_once_hold_one_tokenizer() {
    for _ in 0..8 {
        let dir = test_support::tokenizer_dir();
        let app_context = app_context(WorkerRegistry::new());
        let loads = ["first", "second", "third"].map(|name| {
            let (app_context, request) = (app_context.clone(), request(name, dir.path()));
            #[expect(
                clippy::disallowed_methods,
                reason = "the test awaits the task it spawns"
            )]
            tokio::spawn(async move { registered(&app_context, request).await })
        });

        let mut entries = Vec::new();
        for load in loads {
            entries.push(load.await.unwrap());
        }
        assert!(entries
            .iter()
            .all(|entry| Arc::ptr_eq(&entry.tokenizer, &entries[0].tokenizer)));
    }
}

/// A tokenizer file rewritten between two loads is read anew: the later
/// name gets the new vocabulary, the earlier one keeps what it loaded.
#[tokio::test]
async fn a_tokenizer_file_rewritten_between_two_loads_is_read_anew() {
    let dir = test_support::tokenizer_dir();
    let app_context = app_context(WorkerRegistry::new());
    let first = registered(&app_context, request("first", dir.path())).await;

    let file = dir.path().join("tokenizer.json");
    let rewritten = std::fs::read_to_string(&file)
        .unwrap()
        .replace(r#""<unk>": 1}"#, r#""<unk>": 1, "world": 2}"#);
    std::fs::write(&file, rewritten).unwrap();
    let second = registered(&app_context, request("second", dir.path())).await;

    assert!(!Arc::ptr_eq(&first.tokenizer, &second.tokenizer));
    assert_eq!(first.tokenizer.token_to_id("world"), None);
    assert_eq!(second.tokenizer.token_to_id("world"), Some(2));
}

/// A load reads below the tokenizer's directory too: a file one
/// directory down rewritten between two loads is read anew, and the
/// names behind it share again.
#[tokio::test]
async fn a_file_one_directory_down_rewritten_between_two_loads_is_read_anew() {
    let dir = test_support::tokenizer_dir();
    let below = dir.path().join("below");
    std::fs::create_dir(&below).unwrap();
    std::fs::write(below.join("table"), "one").unwrap();
    let app_context = app_context(WorkerRegistry::new());
    let first = registered(&app_context, request("first", dir.path())).await;

    std::fs::write(below.join("table"), "two").unwrap();
    let second = registered(&app_context, request("second", dir.path())).await;
    let third = registered(&app_context, request("third", dir.path())).await;

    assert!(!Arc::ptr_eq(&first.tokenizer, &second.tokenizer));
    assert!(Arc::ptr_eq(&second.tokenizer, &third.tokenizer));
}

/// Two links to one tokenizer file, each beside a `tokenizer_config.json`
/// of its own, are two tokenizers: a load reads beside the path it is
/// pointed at, so each name renders the template of its own directory.
#[cfg(unix)]
#[tokio::test]
async fn two_links_to_one_tokenizer_file_each_render_the_template_beside_them() {
    use std::os::unix::fs::symlink;

    let shared = test_support::tokenizer_dir();
    let linked = |template: &str| {
        let dir = tempfile::tempdir().unwrap();
        symlink(
            shared.path().join("tokenizer.json"),
            dir.path().join("tokenizer.json"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": template }).to_string(),
        )
        .unwrap();
        dir
    };
    let formal = linked("formal: {{ messages[0]['content'] }}");
    let casual = linked("casual: {{ messages[0]['content'] }}");
    let app_context = app_context(WorkerRegistry::new());

    let first = formal.path().join("tokenizer.json");
    let first = registered(&app_context, request("first", &first)).await;
    let second = casual.path().join("tokenizer.json");
    let second = registered(&app_context, request("second", &second)).await;

    assert_eq!(rendered(&first), "formal: hi");
    assert_eq!(rendered(&second), "casual: hi");
}

/// A name removed leaves the tokenizer to the name that shares it, and
/// the tokenizer goes with its last name: nothing else keeps it.
#[tokio::test]
async fn a_name_removed_leaves_the_shared_tokenizer_to_the_other_name() {
    let dir = test_support::tokenizer_dir();
    let app_context = app_context(WorkerRegistry::new());
    let registry = &app_context.tokenizer_registry;
    registered(&app_context, request("first", dir.path())).await;
    registered(&app_context, request("second", dir.path())).await;

    drop(registry.remove("first"));
    let second = registry.get("second").unwrap();
    assert_eq!(second.token_to_id("hello"), Some(0));

    let tokenizer = Arc::downgrade(&second);
    drop(second);
    drop(registry.remove("second"));
    assert!(tokenizer.upgrade().is_none());

    // A name registered after the last holder went loads afresh.
    let third = registered(&app_context, request("third", dir.path())).await;
    assert_eq!(third.tokenizer.token_to_id("hello"), Some(0));
}

/// What a name sets beside the vocabulary stays its own: a chat
/// template file makes a tokenizer of its own for the names that hand
/// it in, and a cache wraps the shared tokenizer for its name alone.
#[tokio::test]
async fn a_names_chat_template_and_cache_stay_its_own() {
    let dir = test_support::tokenizer_dir();
    let templates = tempfile::tempdir().unwrap();
    let template = |name: &str, text: &str| {
        let path = templates.path().join(name);
        std::fs::write(&path, text).unwrap();
        Some(path.to_str().unwrap().to_string())
    };
    let formal = template("formal.jinja", "formal: {{ messages[0]['content'] }}");
    let casual = template("casual.jinja", "casual: {{ messages[0]['content'] }}");
    let app_context = app_context(WorkerRegistry::new());
    let with = |name: &str, chat_template_path: &Option<String>| TokenizerConfigRequest {
        chat_template_path: chat_template_path.clone(),
        ..request(name, dir.path())
    };

    let plain = registered(&app_context, with("plain", &None)).await;
    let first = registered(&app_context, with("first", &formal)).await;
    let second = registered(&app_context, with("second", &casual)).await;
    let third = registered(&app_context, with("third", &formal)).await;

    assert_eq!(rendered(&first), "formal: hi");
    assert_eq!(rendered(&second), "casual: hi");
    assert_eq!(rendered(&third), "formal: hi");
    assert!(Arc::ptr_eq(&first.tokenizer, &third.tokenizer));
    assert!(!Arc::ptr_eq(&first.tokenizer, &second.tokenizer));
    assert!(!Arc::ptr_eq(&first.tokenizer, &plain.tokenizer));

    // A name with a cache holds its own wrapper around the tokenizer
    // the plain name holds.
    let cached = TokenizerConfigRequest {
        cache_config: Some(TokenizerCacheConfig {
            enable_l0: true,
            l0_max_entries: 16,
            l0_max_memory: 1 << 20,
            enable_l1: false,
            l1_max_memory: 0,
        }),
        ..request("cached", dir.path())
    };
    let cached = registered(&app_context, cached).await;
    let wrapper = cached
        .tokenizer
        .as_any()
        .downcast_ref::<CachedTokenizer>()
        .expect("the name asked for a cache");
    assert!(Arc::ptr_eq(wrapper.inner(), &plain.tokenizer));
    assert!(plain
        .tokenizer
        .as_any()
        .downcast_ref::<CachedTokenizer>()
        .is_none());
}
