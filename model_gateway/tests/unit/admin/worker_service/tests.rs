use openai_protocol::model_card::ModelCard;
use smg_worker::{identity::basic::WorkerType, jobs::queue::testing::QueueTest};

use super::*;
use crate::{
    app_context::testing::app_context_for_tests,
    routers::grpc::backend_client::BasicWorkerBuilder,
    workflow::{job_queue::recording_for_tests, JobQueueConfig},
};

fn make_service(registry: Arc<WorkerRegistry>) -> WorkerService {
    WorkerService::new(
        registry,
        Arc::new(std::sync::OnceLock::new()),
        RouterConfig::default(),
    )
}

fn register_worker(registry: &WorkerRegistry, url: &str, model: Option<&str>) {
    let mut builder = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);
    if let Some(model) = model {
        builder = builder.model(ModelCard::new(model));
    }
    registry.register(Arc::new(builder.build())).unwrap();
}

#[test]
fn test_list_workers_unfiltered_returns_all() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "http://kimi:30000", Some("moonshotai/Kimi-K2.5"));
    register_worker(
        &registry,
        "http://llama:30000",
        Some("meta-llama/Llama-3.1-8B"),
    );
    let service = make_service(registry);

    let result = service.list_workers(None, None);
    assert_eq!(result.workers.len(), 2);
    assert_eq!(result.total, 2);
    assert_eq!(result.regular_count, 2);
}

#[test]
fn test_list_workers_filters_by_model() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "http://kimi:30000", Some("moonshotai/Kimi-K2.5"));
    register_worker(
        &registry,
        "http://llama:30000",
        Some("meta-llama/Llama-3.1-8B"),
    );
    let service = make_service(registry);

    let result = service.list_workers(Some("moonshotai/Kimi-K2.5"), None);
    assert_eq!(result.workers.len(), 1);
    assert_eq!(
        result.workers[0].model_id.as_deref(),
        Some("moonshotai/Kimi-K2.5")
    );
    assert_eq!(result.total, 1);
    assert_eq!(result.regular_count, 1);

    let result = service.list_workers(Some("no-such-model"), None);
    assert!(result.workers.is_empty());
    assert_eq!(result.total, 0);
    assert_eq!(result.regular_count, 0);
}

#[test]
fn test_list_workers_wildcard_worker_matches_any_model() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "http://wildcard:30000", None);
    let service = make_service(registry);

    let result = service.list_workers(Some("moonshotai/Kimi-K2.5"), None);
    assert_eq!(result.workers.len(), 1);
    assert_eq!(result.total, 1);
}

fn worker_spec(url: &str) -> WorkerSpec {
    serde_json::from_value(json!({ "url": url })).expect("worker spec")
}

#[tokio::test]
async fn the_worker_api_asks_as_api() {
    use crate::app_context::testing::create_test_app_context;

    let context = create_test_app_context();
    let (queue, mut next_job) = recording_for_tests(&context);
    context.worker_job_queue.set(queue).unwrap();
    let registry = &context.worker_registry;
    let registered = |url| {
        register_worker(registry, url, Some("served"));
        registry.get_id_by_url(url).unwrap()
    };
    let replace_url = "http://replace:8000";
    let replace = registered(replace_url);
    let update = registered("http://update:8000");
    let delete = registered("http://delete:8000");
    let service = &context.worker_service;

    service
        .create_worker(worker_spec("http://create:8000"))
        .await
        .unwrap();
    assert!(matches!(
        next_job(),
        Job::AddWorker {
            registration_mode: WorkerRegistrationMode::CreateOnly,
            asker: WorkerJobAsker::Api,
            ..
        }
    ));
    service
        .replace_worker(replace.as_str(), worker_spec(replace_url))
        .await
        .unwrap();
    assert!(matches!(
        next_job(),
        Job::AddWorker {
            registration_mode: WorkerRegistrationMode::ReplaceById { .. },
            asker: WorkerJobAsker::Api,
            ..
        }
    ));
    let request = serde_json::from_value(json!({"priority": 2})).unwrap();
    service
        .update_worker(update.as_str(), request)
        .await
        .unwrap();
    assert!(matches!(
        next_job(),
        Job::UpdateWorker {
            asker: WorkerJobAsker::Api,
            ..
        }
    ));
    service.delete_worker(delete.as_str()).await.unwrap();
    assert!(matches!(
        next_job(),
        Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            ..
        }
    ));
}

#[tokio::test]
async fn create_worker_rejects_schemeless_url_before_reserving() {
    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(registry);

    let err = service
        .create_worker(worker_spec("10.0.0.5:8000"))
        .await
        .expect_err("schemeless URL must be rejected at the boundary");

    assert!(matches!(err, WorkerServiceError::BadRequest { .. }));
    assert_eq!(err.error_code(), "BAD_REQUEST");
    assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn create_worker_rejects_mixed_case_scheme() {
    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(registry);

    let err = service
        .create_worker(worker_spec("HTTP://10.0.0.5:8000"))
        .await
        .expect_err("mixed-case scheme must be rejected at the boundary");

    assert!(matches!(err, WorkerServiceError::BadRequest { .. }));
}

/// A worker URL that ends in `@` and digits is refused with a 400 and
/// its reason: every line would name a rank of the host before the
/// digits while the probe dialled the digits as the host. The answer
/// shows the scheme of the address and no other part: the last two are
/// a token alone, and a user with a password of digits, which the
/// gateway's own rendering reads as a host and writes whole.
#[tokio::test]
async fn create_worker_refuses_an_address_read_two_ways() {
    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(registry);
    for url in [
        "http://127.0.0.1:35099@2130706433",
        "http://user:s3cret@127.0.0.1:52605@2130706433",
        "grpc://127.0.0.1:61845@2130706433",
        "http://s3cret-token@8080",
        "http://s3cret-user:8080@2130706433",
    ] {
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err(url);
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST, "{url}");
        let WorkerServiceError::BadRequest { message } = err else {
            panic!("expected a bad request, got {err}");
        };
        assert!(
            message.contains("cannot end in '@' and digits"),
            "{url}: {message}"
        );
        assert!(!message.contains("s3cret"), "{url}: {message}");
    }
}

#[tokio::test]
async fn create_worker_accepts_schemed_url() {
    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(registry);

    // The harness leaves the job queue uninitialized, so a valid URL
    // passes boundary validation and stops at queue submission —
    // proving validation did not reject it.
    let err = service
        .create_worker(worker_spec("grpc://10.0.0.5:8000"))
        .await
        .expect_err("queue is uninitialized in the test harness");

    assert!(matches!(err, WorkerServiceError::QueueNotInitialized));
}

#[tokio::test]
async fn create_worker_releases_reservation_when_submission_fails() {
    let registry = Arc::new(WorkerRegistry::new());
    // A queue whose AppContext is gone rejects every submission.
    let queue = Arc::new(std::sync::OnceLock::new());
    queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            std::sync::Weak::new(),
        ))
        .ok();
    let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
    // Idempotent, so this is the id create_worker will hand out.
    let reserved = registry.reserve_id_for_url("grpc://10.0.0.5:8000");

    let err = service
        .create_worker(worker_spec("grpc://10.0.0.5:8000"))
        .await
        .expect_err("submission must fail without an AppContext");
    assert!(matches!(err, WorkerServiceError::QueueSubmitFailed { .. }));

    // The 202 was never sent, so nothing may keep the URL reserved (#1533).
    assert!(registry.get_url_by_id(&reserved).is_none());
}

/// A registered data-parallel group stands under its rank URLs, none
/// under the id its `POST` answered. The read of that id answers from
/// the ranks, the least ready one, where it answered 404.
#[tokio::test]
async fn get_worker_reads_a_registered_data_parallel_group_by_the_id_of_its_post() {
    let registry = Arc::new(WorkerRegistry::new());
    let queue = Arc::new(std::sync::OnceLock::new());
    queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            std::sync::Weak::new(),
        ))
        .ok();
    let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
    let url = "grpc://10.0.0.7:8000";
    let id = registry.reserve_id_for_url(url);
    for rank in 0..2 {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .dp_config(rank, 2)
            .build();
        worker.set_status(WorkerStatus::Ready);
        registry
            .register(Arc::new(worker))
            .expect("a rank registers");
    }

    let GetWorkerResponse(group) = service
        .get_worker(id.as_str())
        .expect("the ranks answer for the id of the POST");
    assert_eq!(group.id, id.as_str());
    assert!(group.is_healthy, "every rank is ready");

    // The lookup's second rank, so that a read of the first one found
    // would still say healthy.
    let not_ready = find_workers_by_url(&registry, url)[1].clone();
    not_ready.set_status(WorkerStatus::NotReady);
    let GetWorkerResponse(group) = service
        .get_worker(id.as_str())
        .expect("the ranks answer for the id of the POST");
    assert!(!group.is_healthy, "one rank is not ready");
    assert_eq!(
        group.spec.url,
        not_ready.url(),
        "the view is the not-ready rank's"
    );
}

/// The job a `POST` of `url` claims the queue with (the claim reads the
/// job's type and URL, not its registration mode).
fn registration(url: &str) -> Job {
    Job::AddWorker {
        config: Box::new(worker_spec(url)),
        registration_mode: WorkerRegistrationMode::CreateOnly,
        asker: WorkerJobAsker::Api,
    }
}

/// A registration in flight holds the queue's claim for its URL, and a
/// POST's has reserved the id its 202 named: a POST refused by that claim
/// is told to poll that id and leaves the mapping and the queue alone (a
/// queue with no context behind it fails any submission, so reaching it
/// would show).
#[tokio::test]
async fn a_post_under_a_posts_registration_in_flight_is_told_to_poll_its_id_and_leaves_it() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "grpc://10.0.0.5:8000";
    let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
    // The first POST's claim, then its reservation: create_worker's order.
    assert!(queue.claim_if_idle(&registration(url)));
    let reserved = registry.reserve_id_for_url(url);
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(queue).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("another POST's registration is in flight");
    let WorkerServiceError::CreateInProgress { worker_id, .. } = err else {
        panic!("expected CreateInProgress, got {err:?}");
    };
    assert_eq!(
        worker_id,
        Some(reserved.clone()),
        "the id in flight is the one to poll"
    );
    assert_eq!(
        registry.get_id_by_url_or_reservation(url),
        Some(reserved),
        "the refused POST released a mapping that is not its own"
    );
}

/// A POST refused by the queue's claim reserves nothing: the registration
/// in flight here is a startup or discovery one, which reserves no id, so
/// there is none to poll, and none outlives the refusal for a DELETE to
/// find.
#[tokio::test]
async fn a_post_refused_for_a_registration_of_another_kind_leaves_no_reservation() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "grpc://10.0.0.5:8000";
    let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
    // The discovery registration's claim, with no reservation behind it.
    assert!(queue.claim_if_idle(&registration(url)));
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(queue).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("the discovery registration is in flight");
    let WorkerServiceError::CreateInProgress { worker_id, .. } = err else {
        panic!("expected CreateInProgress, got {err:?}");
    };
    assert_eq!(worker_id, None, "no id of a POST's is in flight to poll");
    assert_eq!(
        registry.get_id_by_url_or_reservation(url),
        None,
        "the refused POST's reservation outlived the refusal"
    );
}

/// A mapping with no job behind it is a reservation left over (a
/// data-parallel registration leaves its base URL's once its ranks are
/// registered under their own URLs, and nothing releases it): the next
/// POST adopts it and is admitted under that id, as it was before the
/// queue's claim moved in front of the reservation.
#[tokio::test]
async fn a_leftover_reservation_with_no_job_in_flight_is_admitted_under_its_id() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "grpc://10.0.0.5:8000";
    let reserved = registry.reserve_id_for_url(url);
    let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
    // A queue that accepts a job and never runs it: no slot to run one in.
    let queue = JobQueue::new(
        JobQueueConfig {
            max_concurrent_jobs: 0,
            ..JobQueueConfig::default()
        },
        Arc::downgrade(&context),
    );
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(queue).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
    let admitted = service
        .create_worker(worker_spec(url))
        .await
        .expect("a leftover reservation is no registration in flight");
    assert_eq!(admitted.worker_id, reserved, "the 202 names the adopted id");
    assert_eq!(registry.get_id_by_url_or_reservation(url), Some(reserved));
}

/// An id with no worker behind it is a registration in flight: a PATCH
/// of it is told to poll the id and submits nothing, so the registration's
/// entry in the job queue, and with it the claim that keeps a second POST
/// off its id, stay its own. Before this, the update job's status took
/// the entry and the second POST was admitted under the same id; a DELETE
/// then cancelled the first registration alone, whose end released the
/// shared reservation, and the second registered under a fresh id.
#[tokio::test]
async fn a_patch_of_a_registration_in_flight_is_told_to_poll_it_and_leaves_its_claim() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "grpc://10.0.0.5:8000";
    let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
    // A queue that accepts jobs and never runs one: no slot to run one in.
    let queue = JobQueue::new(
        JobQueueConfig {
            max_concurrent_jobs: 0,
            ..JobQueueConfig::default()
        },
        Arc::downgrade(&context),
    );
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(queue).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
    let id = service
        .create_worker(worker_spec(url))
        .await
        .expect("the first POST is admitted")
        .worker_id;
    let patch: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"tier": "a"}})).expect("an update");
    let err = service
        .update_worker(id.as_str(), patch)
        .await
        .expect_err("the registration is in flight");
    assert!(
        matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
        "{err:?}"
    );
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("the registration still holds the URL");
    assert!(
        matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
        "a second POST was admitted beside the registration: {err:?}"
    );
}

/// A successful data-parallel POST leaves its id mapped to the base URL,
/// the ranks registered under the rank URLs and no job in flight: a PATCH
/// by that id is admitted and reaches every rank, as before the in-flight
/// guard, which read the empty base as a registration in flight and
/// refused the group's PATCH on every try.
#[tokio::test]
async fn a_patch_of_a_completed_data_parallel_group_by_its_post_id_reaches_every_rank() {
    use crate::workflow::WorkflowEngines;

    let registry = Arc::new(WorkerRegistry::new());
    let url = "http://10.0.0.6:8000";
    let router_config = RouterConfig::default();
    let context = app_context_for_tests(router_config.clone(), Arc::clone(&registry), None);
    assert!(
        context
            .workflow_engines
            .set(WorkflowEngines::new(&router_config))
            .is_ok(),
        "the engines are set once"
    );
    let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
    assert!(
        context.worker_job_queue.set(Arc::clone(&queue)).is_ok(),
        "the queue is set once"
    );
    // The group as its POST leaves it: the id reserved for the base URL,
    // the ranks registered under the rank URLs, the job over.
    let id = registry.reserve_id_for_url(url);
    for rank in 0..2 {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("served"))
            .dp_config(rank, 2)
            .build();
        registry
            .register(Arc::new(worker))
            .expect("a rank registers");
    }
    assert!(
        registry.get_by_url(url).is_none(),
        "nothing stands under the base URL itself"
    );
    assert_eq!(find_workers_by_url(&registry, url).len(), 2);

    let patch: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"tier": "a"}})).expect("an update");
    let updated = context
        .worker_service
        .update_worker(id.as_str(), patch)
        .await
        .expect("the group's PATCH is admitted");
    assert_eq!(updated.worker_id, id);

    // The update job reaches both ranks.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let ranks = find_workers_by_url(&registry, url);
        if ranks.len() == 2
            && ranks.iter().all(|worker| {
                worker
                    .metadata()
                    .spec
                    .labels
                    .get("tier")
                    .map(String::as_str)
                    == Some("a")
            })
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the update did not reach every rank: {:?}",
            queue.get_status(url)
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// A DELETE behind a removal in flight for its URL keeps its removal for
/// that removal's end (`run_kept`), where before it was answered by the
/// removal in flight alone: a guarded cleanup (the entries a
/// registration's end saw) skips a worker replaced since its snapshot,
/// and a DELETE answered by it removed nothing.
#[tokio::test]
async fn a_delete_behind_a_removal_in_flight_keeps_its_removal() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "http://10.0.0.7:8000";
    let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
    // A queue that accepts jobs and never runs one: no slot to run one in.
    let queue = JobQueue::new(
        JobQueueConfig {
            max_concurrent_jobs: 0,
            ..JobQueueConfig::default()
        },
        Arc::downgrade(&context),
    );
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(Arc::clone(&queue)).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new("served"))
        .build();
    let id = registry
        .register(Arc::new(worker))
        .expect("the worker registers");
    // The cleanup removal of a registration's end, pinned to the entry it
    // saw, holds the URL.
    let cleanup = Job::RemoveWorker {
        url: url.to_string(),
        expected_revisions: Some(HashMap::from([(id.as_str().to_string(), Some(1))])),
        asker: WorkerJobAsker::Api,
    };
    assert!(queue.claim_if_idle(&cleanup), "the cleanup holds the URL");
    let deleted = service
        .delete_worker(id.as_str())
        .await
        .expect("the DELETE is admitted");
    assert_eq!(deleted.worker_id, id);
    assert!(
        QueueTest::kept_for(&queue, url),
        "the DELETE's removal is kept for the removal in flight to run on its heels"
    );
}

/// The status, the body and the Retry-After header of an error's answer.
async fn answer(err: WorkerServiceError) -> (StatusCode, serde_json::Value, Option<String>) {
    let response = err.into_response();
    let status = response.status();
    let retry_after = response
        .headers()
        .get(http::header::RETRY_AFTER)
        .map(|value| value.to_str().expect("a visible header").to_string());
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body");
    (
        status,
        serde_json::from_slice(&bytes).expect("a JSON body"),
        retry_after,
    )
}

/// A registration of a URL whose worker is being removed is refused for
/// the drain window under the code it had, with an answer that says what
/// holds: the removal in progress, the drain, when to retry, and a
/// Retry-After of the drain. A live worker's refusal keeps its hint to
/// replace or update it.
#[tokio::test]
async fn create_worker_names_the_removal_in_progress_of_a_draining_worker() {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    let registry = Arc::new(WorkerRegistry::new());
    let queue = Arc::new(std::sync::OnceLock::new());
    queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            std::sync::Weak::new(),
        ))
        .ok();
    let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
    let url = "grpc://10.0.0.8:8000";
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                drain_settle_secs: 7,
                ..Default::default()
            })
            .build(),
    );
    registry.register(Arc::clone(&worker)).unwrap();
    let id = registry.reserve_id_for_url(url);

    let (status, body, retry_after) = answer(
        service
            .create_worker(worker_spec(url))
            .await
            .expect_err("a live worker holds the URL"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "WORKER_ALREADY_EXISTS", "{body}");
    let message = body["error"].as_str().expect("a message");
    assert!(
        message.contains(&format!("Use PUT /workers/{}", id.as_str())),
        "a live worker's refusal hints at the replacement: {message}"
    );
    assert_eq!(retry_after, None, "a live worker's refusal has no retry");

    worker.set_status(WorkerStatus::Draining);
    let (status, body, retry_after) = answer(
        service
            .create_worker(worker_spec(url))
            .await
            .expect_err("a draining worker holds the URL for its drain"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "WORKER_ALREADY_EXISTS", "{body}");
    let message = body["error"].as_str().expect("a message");
    assert!(
        message.contains("is being removed")
            && message.contains("for 7s, the worker's configured drain"),
        "the refusal names the removal and its drain: {message}"
    );
    assert!(
        message.contains(&format!("GET /workers/{} answers 404", id.as_str())),
        "the refusal says when to retry: {message}"
    );
    assert!(
        !message.contains("PUT"),
        "no replacement of a worker that is leaving: {message}"
    );
    assert_eq!(
        retry_after.as_deref(),
        Some("7"),
        "Retry-After carries the drain window"
    );
}

/// A PUT re-runs one registration under one id, which a rank of a
/// data-parallel engine cannot be: the PUT of a rank worker is refused by
/// name, on a gateway given no setting for it. The PUT of a plain worker
/// passes that test: here it stops at the next thing it needs, the job
/// queue this service was given none of.
#[tokio::test]
async fn a_put_is_refused_for_a_rank_worker_and_for_no_other() {
    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(Arc::clone(&registry));
    let plain_url = "http://10.0.0.9:8000";
    register_worker(&registry, plain_url, Some("m"));
    let plain = registry.get_id_by_url(plain_url).expect("the plain worker");
    let engine = "http://10.0.0.10:8000";
    let rank = registry
        .register(Arc::new(
            BasicWorkerBuilder::new(engine)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .dp_config(1, 2)
                .build(),
        ))
        .expect("a rank registers");

    let refused = service
        .replace_worker(rank.as_str(), worker_spec(engine))
        .await;
    assert!(
        matches!(
            &refused,
            Err(WorkerServiceError::BadRequest { message })
                if message.contains("belongs to a data-parallel group")
        ),
        "a rank worker's PUT was not refused as one: {refused:?}"
    );
    let passed = service
        .replace_worker(plain.as_str(), worker_spec(plain_url))
        .await;
    assert!(
        matches!(&passed, Err(WorkerServiceError::QueueNotInitialized)),
        "a plain worker's PUT stopped short of the queue: {passed:?}"
    );
}

/// A worker that a removal is draining is leaving, and a PUT does not
/// bring it back to serve inside the drain.
#[tokio::test]
async fn a_put_does_not_bring_a_draining_worker_back() {
    use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

    use crate::workflow::{create_worker_workflow_data, ActivateWorkersStep, RegisterWorkersStep};

    let registry = Arc::new(WorkerRegistry::new());
    let service = make_service(Arc::clone(&registry));
    let url = "http://10.0.0.8:8000";
    let build = || -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .build(),
        )
    };
    let worker = build();
    let id = registry.register(Arc::clone(&worker)).unwrap();
    // The run of a PUT answered now, bound to the revision it read.
    let run_of_a_put = |replacement: &Arc<dyn Worker>| {
        let mut data = create_worker_workflow_data(
            worker_spec(url),
            WorkerRegistrationMode::ReplaceById {
                worker_id: id.as_str().to_string(),
                expected_revision: worker.revision(),
            },
            app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None),
            WorkerJobAsker::Api,
        );
        data.actual_workers = Some(vec![Arc::clone(replacement)]);
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    };

    let written = build();
    let mut run = run_of_a_put(&written);
    RegisterWorkersStep
        .execute(&mut run)
        .await
        .expect("the PUT writes its replacement");
    // A probe promotes the worker and a removal marks it after the PUT's
    // activation chose Ready, before it takes the worker's lock.
    let lost = registry.with_another_write_before_the_lock(
        |_| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime for the step")
                .block_on(ActivateWorkersStep.execute(&mut run))
        },
        |registry| {
            registry.transition_status(&id, WorkerStatus::Ready);
            registry.transition_status(&id, WorkerStatus::Draining);
        },
    );
    assert!(
        matches!(&lost, Err(error) if error.to_string().contains("was not activated"))
            && written.status() == WorkerStatus::Draining,
        "a removal's mark behind the write of a PUT stands and fails its activation"
    );

    assert!(
        matches!(
            service.replace_worker(id.as_str(), worker_spec(url)).await,
            Err(WorkerServiceError::RemovalInProgress { .. })
        ),
        "a PUT of a draining worker is refused as a POST of its URL is"
    );

    let dropped = RegisterWorkersStep
        .execute(&mut run_of_a_put(&build()))
        .await;
    assert!(
        matches!(&dropped, Err(error) if error.to_string().contains("is being removed")),
        "the registry step of a PUT answered before the removal drops its replacement"
    );

    // The same revision is written without the Draining test: the refusal
    // above was the removal's, not a stale revision's.
    let rebuilt = build();
    assert!(registry.replace_claiming_local(&id, Arc::clone(&rebuilt), worker.revision()));
    ActivateWorkersStep
        .execute(&mut run)
        .await
        .expect("a superseded PUT has nothing to activate");
    assert_eq!(
        rebuilt.status(),
        WorkerStatus::Draining,
        "the activation of a PUT superseded since leaves its successor's status alone"
    );
}

/// A registered data-parallel group stands under its rank URLs, none
/// under the id its `POST` answered. A second `POST` of its address is
/// refused with that id while a rank stays, and as a removal with the
/// longest drain once every rank is leaving. A registration of the
/// address in flight is still the answer to poll.
#[tokio::test]
async fn create_worker_refuses_the_address_of_a_registered_data_parallel_group() {
    use openai_protocol::worker::HealthCheckConfig;

    let registry = Arc::new(WorkerRegistry::new());
    let queue = Arc::new(std::sync::OnceLock::new());
    queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            std::sync::Weak::new(),
        ))
        .ok();
    let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
    let url = "grpc://10.0.0.9:8000";
    let id = registry.reserve_id_for_url(url);
    for (rank, drain_settle_secs) in [3, 9].into_iter().enumerate() {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                drain_settle_secs,
                ..Default::default()
            })
            .dp_config(rank, 2)
            .build();
        registry
            .register(Arc::new(worker))
            .expect("a rank registers");
    }
    let ranks = find_workers_by_url(&registry, url);

    ranks[0].set_status(WorkerStatus::Draining);
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("the rank that stays holds the address");
    assert!(
        matches!(&err, WorkerServiceError::Conflict { worker_id, .. } if *worker_id == id),
        "the refusal of a registered worker, with the id of the first POST: {err}"
    );

    ranks[1].set_status(WorkerStatus::Draining);
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("the draining ranks hold the address");
    assert!(
        matches!(
            &err,
            WorkerServiceError::RemovalInProgress { worker_id, drain_secs: 9, .. }
                if *worker_id == id
        ),
        "a group whose every rank is leaving is being removed, for its longest drain: {err}"
    );

    assert!(service
        .get_job_queue()
        .expect("the queue is set")
        .claim_if_idle(&registration(url)));
    let err = service
        .create_worker(worker_spec(url))
        .await
        .expect_err("a registration in flight holds the address");
    assert!(
        matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
        "a registration in flight answers with its id to poll, not for the ranks: {err}"
    );
}

/// A control character in a string the log writes as it came is refused
/// at the door, on POST and PUT: the URL, a model's id, a parser's name.
#[tokio::test]
async fn a_control_character_in_a_string_the_log_writes_is_refused_on_post_and_put() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
    let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
    let service = make_service(Arc::clone(&registry));
    let url = "grpc://10.0.0.9:8000";
    let refused = [
        (
            "url",
            json!({"url": "grpc://10.0.0.9:8000/\nINJECTED level=ERROR forged"}),
        ),
        ("url", json!({"url": "grpc://10.0.0.9:8000/\rINJECTED"})),
        (
            "models[].id",
            json!({"url": url, "models": [{"id": "m\nINJECTED"}]}),
        ),
        (
            "models[].reasoning_parser",
            json!({"url": url, "models": [{"id": "m", "reasoning_parser": "p\nINJECTED"}]}),
        ),
        (
            "models[].tool_parser",
            json!({"url": url, "models": [{"id": "m", "tool_parser": "p\nINJECTED"}]}),
        ),
    ];
    for (field, body) in refused {
        let wanted = format!("{field} must not contain control characters");
        let spec = || serde_json::from_value::<WorkerSpec>(body.clone()).expect("a spec");
        let post = service.create_worker(spec()).await.expect_err("POST");
        let put = service
            .replace_worker(id.as_str(), spec())
            .await
            .expect_err("PUT");
        for (verb, err) in [("POST", post), ("PUT", put)] {
            assert!(
                matches!(&err, WorkerServiceError::BadRequest { message } if *message == wanted),
                "{verb} {field}: {err}"
            );
        }
    }
    let taken: WorkerSpec = serde_json::from_value(json!({
        "url": url,
        "models": [{"id": "m", "reasoning_parser": "p", "tool_parser": "p"}]
    }))
    .expect("a spec");
    assert!(matches!(
        service
            .create_worker(taken)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// A registration whose labels or fields the routers could not read is
/// refused at the boundary with a 400 naming the field, before any job
/// is queued; one they can read passes the boundary (and stops at the
/// uninitialized queue of this harness).
#[tokio::test]
async fn create_worker_refuses_labels_and_fields_the_routers_cannot_read() {
    let service = make_service(Arc::new(WorkerRegistry::new()));
    let url = "grpc://10.0.0.5:8000";
    let thousand: serde_json::Map<String, serde_json::Value> =
        (0..1000).map(|i| (format!("k{i}"), json!("v"))).collect();
    let refused = [
        (
            json!({"url": url, "labels": {"max_running_requests": "abc"}}),
            "max_running_requests",
        ),
        (
            json!({"url": url, "labels": {"max_running_requests": "0"}}),
            "max_running_requests",
        ),
        (
            json!({"url": url, "labels": {"dp_size": "zero"}}),
            "dp_size",
        ),
        (
            json!({"url": url, "labels": {"supports_vision": "maybe"}}),
            "supports_vision",
        ),
        (
            json!({"url": url, "labels": {"note": "x".repeat(4097)}}),
            "note",
        ),
        (json!({"url": url, "labels": thousand}), "labels"),
        (
            json!({"url": url, "labels": {"key with space\nnewline": "v"}}),
            "label key",
        ),
        (json!({"url": url, "cost": 0}), "cost"),
        (json!({"url": url, "cost": -1.0}), "cost"),
        (json!({"url": url, "kv_block_size": 0}), "kv_block_size"),
        (json!({"url": url, "dp_rank": 1}), "dp_rank"),
        (json!({"url": url, "dp_rank": 2, "dp_size": 2}), "dp_rank"),
        (json!({"url": url, "bootstrap_port": 0}), "bootstrap_port"),
        (
            json!({"url": url, "resilience": {"cb_failure_threshold": 0}}),
            "'resilience.cb_failure_threshold': 0 - Must be >= 1 (set to u32::MAX to effectively disable CB)",
        ),
        (
            json!({"url": url, "resilience": {"cb_success_threshold": 0}}),
            "'resilience.cb_success_threshold': 0 - Must be >= 1",
        ),
        (
            json!({"url": url, "resilience": {"cb_timeout_secs": 0}}),
            "'resilience.cb_timeout_secs': 0 - Must be > 0",
        ),
        (
            json!({"url": url, "resilience": {"cb_window_secs": 0}}),
            "'resilience.cb_window_secs': 0 - Must be > 0",
        ),
        (
            json!({"url": format!("grpc://{}:1", "a".repeat(2049))}),
            "url",
        ),
    ];
    for (body, field) in refused {
        let spec: WorkerSpec = serde_json::from_value(body.clone()).expect("a spec the type takes");
        let err = service
            .create_worker(spec)
            .await
            .expect_err(&body.to_string());
        assert!(
            matches!(err, WorkerServiceError::BadRequest { .. }),
            "{body}: {err}"
        );
        assert!(err.to_string().contains(field), "{body}: {err}");
    }
    // A window wider than 65535 passes the boundary: the routers saturate
    // it. A window set here is the worker's only window, kept as set on
    // every dp-aware rank worker; only a discovered one is divided per rank.
    let spec: WorkerSpec = serde_json::from_value(json!({
        "url": url, "cost": 2.5, "kv_block_size": 16, "dp_rank": 1, "dp_size": 2,
        "resilience": {"cb_failure_threshold": 1, "cb_success_threshold": 1, "cb_timeout_secs": 1, "cb_window_secs": 1},
        "labels": {
            "max_running_requests": "256", "max_num_seqs": "70000",
            "supports_vision": "true", "served_model_name": "m"
        }
    }))
    .expect("a valid spec");
    assert!(matches!(
        service
            .create_worker(spec)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// An update is held to the same rules for the fields it may change.
#[tokio::test]
async fn update_worker_refuses_a_cost_or_labels_the_routers_cannot_read() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "grpc://10.0.0.5:8000", Some("m"));
    let id = registry.reserve_id_for_url("grpc://10.0.0.5:8000");
    let service = make_service(Arc::clone(&registry));
    for body in [
        json!({"cost": 0}),
        json!({"cost": -2.0}),
        json!({"labels": {"supports_vision": "yes"}}),
        json!({"labels": {"max_running_requests": "many"}}),
    ] {
        let update: WorkerUpdateRequest =
            serde_json::from_value(body.clone()).expect("an update the type takes");
        let err = service
            .update_worker(id.as_str(), update)
            .await
            .expect_err(&body.to_string());
        assert!(
            matches!(err, WorkerServiceError::BadRequest { .. }),
            "{body}: {err}"
        );
    }
    let update: WorkerUpdateRequest =
        serde_json::from_value(json!({"cost": 1.5, "labels": {"max_running_requests": "64"}}))
            .expect("a valid update");
    assert!(matches!(
        service
            .update_worker(id.as_str(), update)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// The early check counts the labels the merge leaves: a window set by
/// the update replaces a discovered one under the other name, so swapping
/// the window's name on a full worker passes (as the step would take it),
/// while a new key on a full worker is refused.
#[test]
fn the_early_cap_check_counts_the_labels_the_merge_leaves() {
    let mut full: HashMap<String, String> = (1..MAX_WORKER_LABELS)
        .map(|i| (format!("k{i}"), "v".to_string()))
        .collect();
    full.insert("max_num_seqs".to_string(), "256".to_string());
    assert_eq!(full.len(), MAX_WORKER_LABELS);
    let swap = HashMap::from([("max_running_requests".to_string(), "512".to_string())]);
    assert!(validate_merged_labels(&full, &swap).is_ok());
    assert_eq!(merge_labels(&full, &swap).len(), MAX_WORKER_LABELS);
    let new_key = HashMap::from([("k_new".to_string(), "v".to_string())]);
    let message = validate_merged_labels(&full, &new_key)
        .expect_err("a 129th label")
        .to_string();
    assert!(
        message.contains("129 labels on the worker after this update (128 now, 1 new)"),
        "{message}"
    );
    assert!(
        message.contains("new); at most 128 are accepted"),
        "{message}"
    );
}

/// The label cap holds for the worker, not the request: a PATCH merges
/// its labels into the existing ones, so one that would leave more than
/// the cap on the worker is refused, while one that only rewrites keys
/// the worker has is taken.
#[tokio::test]
async fn update_worker_holds_the_label_cap_over_the_merged_labels() {
    let registry = Arc::new(WorkerRegistry::new());
    let full: HashMap<String, String> = (0..MAX_WORKER_LABELS)
        .map(|i| (format!("k{i}"), "v".to_string()))
        .collect();
    let worker = BasicWorkerBuilder::new("grpc://10.0.0.6:8000")
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new("m"))
        .labels(full)
        .build();
    registry.register(Arc::new(worker)).unwrap();
    let id = registry.reserve_id_for_url("grpc://10.0.0.6:8000");
    let service = make_service(Arc::clone(&registry));
    let one_more: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"k_new": "v"}})).expect("an update");
    let err = service
        .update_worker(id.as_str(), one_more)
        .await
        .expect_err("a 129th label");
    assert!(
        matches!(&err, WorkerServiceError::BadRequest { message } if message.contains("129 labels on the worker")),
        "{err}"
    );
    let rewrite: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"k0": "w", "k1": "w"}})).expect("an update");
    assert!(matches!(
        service
            .update_worker(id.as_str(), rewrite)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// A worker over the cap through the labels discovered at registration
/// still takes an update that adds no key through the endpoint: a rewrite
/// of keys it has reaches the queue, a new key is refused by name.
#[tokio::test]
async fn update_worker_over_the_cap_takes_a_rewrite_through_the_endpoint() {
    let registry = Arc::new(WorkerRegistry::new());
    let over: HashMap<String, String> = (0..MAX_WORKER_LABELS + 20)
        .map(|i| (format!("k{i}"), "v".to_string()))
        .collect();
    let worker = BasicWorkerBuilder::new("grpc://10.0.0.7:8000")
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new("m"))
        .labels(over)
        .build();
    registry.register(Arc::new(worker)).unwrap();
    let id = registry.reserve_id_for_url("grpc://10.0.0.7:8000");
    let service = make_service(Arc::clone(&registry));
    let rewrite: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"k3": "w"}})).expect("an update");
    assert!(matches!(
        service
            .update_worker(id.as_str(), rewrite)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
    let adds: WorkerUpdateRequest =
        serde_json::from_value(json!({"labels": {"k_new": "v"}})).expect("an update");
    let err = service
        .update_worker(id.as_str(), adds)
        .await
        .expect_err("a new key on a worker over the cap");
    assert!(
        matches!(&err, WorkerServiceError::BadRequest { message } if message.contains("149 labels on the worker") && message.contains("1 new")),
        "{err}"
    );
}

/// No request can write the labels discovery reads ownership from: a
/// key under the reserved prefix is a 400 naming it on a POST, on a PUT
/// and on a PATCH of a manual worker, before any job is queued; a key
/// that only resembles the prefix passes.
#[tokio::test]
async fn a_label_under_the_discovery_prefix_is_refused_on_post_put_and_patch() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
    let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
    let service = make_service(Arc::clone(&registry));
    let refused = |err: WorkerServiceError, key: &str| {
        assert!(
            matches!(&err, WorkerServiceError::BadRequest { message }
                if message.contains(&format!("'{key}'"))
                    && message.contains("reserved for worker discovery")),
            "{key}: {err}"
        );
    };
    for key in [
        "smg.ai/discovery-provider",
        "smg.ai/discovery-id",
        "smg.ai/discovery-spec-hash",
        "smg.ai/discovery-anything",
    ] {
        let labels = json!({ key: "file" });
        let post: WorkerSpec =
            serde_json::from_value(json!({"url": "grpc://10.0.0.9:8000", "labels": labels}))
                .expect("a spec");
        refused(service.create_worker(post).await.expect_err("POST"), key);
        let put: WorkerSpec =
            serde_json::from_value(json!({"url": "grpc://10.0.0.8:8000", "labels": labels}))
                .expect("a spec");
        refused(
            service
                .replace_worker(id.as_str(), put)
                .await
                .expect_err("PUT"),
            key,
        );
        let patch: WorkerUpdateRequest =
            serde_json::from_value(json!({ "labels": labels })).expect("an update");
        refused(
            service
                .update_worker(id.as_str(), patch)
                .await
                .expect_err("PATCH"),
            key,
        );
    }
    let lookalikes = json!({
        "smg.ai/discovery": "v",
        "smg.ai/discoveryid": "v",
        "x/smg.ai/discovery-id": "v",
    });
    let post: WorkerSpec =
        serde_json::from_value(json!({"url": "grpc://10.0.0.9:8000", "labels": lookalikes}))
            .expect("a spec");
    assert!(matches!(
        service
            .create_worker(post)
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// A probe interval or timeout the health schedule cannot hold is refused
/// at the door, on POST, PUT and PATCH; the longest it can hold is taken.
#[tokio::test]
async fn a_probe_delay_past_the_schedules_bound_is_refused_on_post_put_and_patch() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
    let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
    let service = make_service(Arc::clone(&registry));
    for field in ["check_interval_secs", "timeout_secs"] {
        let refused = |err: WorkerServiceError, verb: &str| {
            assert!(
                matches!(&err, WorkerServiceError::BadRequest { message }
                    if message.contains(&format!("health.{field} is"))),
                "{verb} {field}: {err}"
            );
        };
        for secs in [946_080_001, u64::MAX] {
            let health = json!({ field: secs });
            let post: WorkerSpec =
                serde_json::from_value(json!({"url": "grpc://10.0.0.9:8000", "health": health}))
                    .expect("a spec");
            refused(service.create_worker(post).await.expect_err("POST"), "POST");
            let put: WorkerSpec =
                serde_json::from_value(json!({"url": "grpc://10.0.0.8:8000", "health": health}))
                    .expect("a spec");
            refused(
                service
                    .replace_worker(id.as_str(), put)
                    .await
                    .expect_err("PUT"),
                "PUT",
            );
            let patch: WorkerUpdateRequest =
                serde_json::from_value(json!({ "health": health })).expect("an update");
            refused(
                service
                    .update_worker(id.as_str(), patch)
                    .await
                    .expect_err("PATCH"),
                "PATCH",
            );
        }
        let longest: WorkerSpec = serde_json::from_value(json!({
            "url": "grpc://10.0.0.9:8000",
            "health": { field: 946_080_000_u64 }
        }))
        .expect("a spec");
        assert!(matches!(
            service
                .create_worker(longest)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
    }
}

/// A discovered worker's spec and labels are changed at its discovery
/// source: a PUT of it is a 409 naming its provider, whatever the body —
/// the worker's own spec sent back included — and so is a PATCH that
/// carries labels, reserved ones included. A PATCH of what no provider
/// publishes, and a DELETE, are taken (the provider adds a deleted
/// worker back while it still publishes it).
#[tokio::test]
async fn put_and_a_label_patch_of_a_discovered_worker_are_refused_naming_its_provider() {
    let registry = Arc::new(WorkerRegistry::new());
    let url = "grpc://10.0.0.10:8000";
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new("m"))
        .label(DISCOVERY_PROVIDER_LABEL, "consul")
        .label("smg.ai/discovery-id", "node-1:8000")
        .build();
    registry.register(Arc::new(worker)).unwrap();
    let id = registry.reserve_id_for_url(url);
    let service = make_service(Arc::clone(&registry));
    let owned = |err: WorkerServiceError| {
        assert_eq!(err.status_code(), StatusCode::CONFLICT, "{err}");
        assert_eq!(err.error_code(), "WORKER_DISCOVERY_OWNED");
        let message = err.to_string();
        assert!(message.contains("'consul' discovery provider"), "{message}");
        assert!(message.contains("discovery source"), "{message}");
    };
    let own_spec = (*registry.get(&id).expect("registered").metadata().spec).clone();
    for put in [worker_spec(url), own_spec] {
        owned(
            service
                .replace_worker(id.as_str(), put)
                .await
                .expect_err("PUT"),
        );
    }
    for body in [
        json!({"labels": {"env": "prod"}}),
        json!({"priority": 5, "labels": {"env": "prod"}}),
        json!({"labels": {DISCOVERY_PROVIDER_LABEL: "file"}}),
    ] {
        let patch: WorkerUpdateRequest = serde_json::from_value(body).expect("an update");
        owned(
            service
                .update_worker(id.as_str(), patch)
                .await
                .expect_err("PATCH"),
        );
    }
    for body in [
        json!({"priority": 5}),
        json!({"cost": 2.0}),
        json!({"api_key": "rotated"}),
        json!({"health": {"timeout_secs": 9}}),
    ] {
        let patch: WorkerUpdateRequest = serde_json::from_value(body.clone()).expect("an update");
        assert!(
            matches!(
                service.update_worker(id.as_str(), patch).await,
                Err(WorkerServiceError::QueueNotInitialized)
            ),
            "{body} reaches the queue"
        );
    }
    assert!(matches!(
        service
            .delete_worker(id.as_str())
            .await
            .expect_err("queue is uninitialized"),
        WorkerServiceError::QueueNotInitialized
    ));
}

/// A rejected key is not echoed whole: a key the size of the payload
/// limit comes back as its first characters and its length.
#[test]
fn a_rejected_label_key_is_shown_truncated() {
    let long = "k".repeat(1 << 20);
    let labels: HashMap<String, String> = [(long.clone(), "v".to_string())].into();
    let err = validate_labels(&labels).expect_err("a key over the cap");
    let WorkerServiceError::BadRequest { message } = err else {
        panic!("expected a bad request, got {err}");
    };
    assert!(message.len() < 512, "{} bytes", message.len());
    assert!(
        message.contains(&format!("({} bytes)", long.len())),
        "{message}"
    );
    let short: HashMap<String, String> = [("has space".to_string(), "v".to_string())].into();
    let err = validate_labels(&short).expect_err("a key with a space");
    assert!(err.to_string().contains("\"has space\""), "{err}");
}

#[test]
fn list_workers_by_origin_tells_local_from_discovered() {
    let registry = Arc::new(WorkerRegistry::new());
    register_worker(&registry, "http://local:30000", None);
    let discovered: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://discovered:30000")
            .worker_type(WorkerType::Regular)
            .label(DISCOVERY_PROVIDER_LABEL, "file")
            .build(),
    );
    registry.register(discovered).unwrap();
    let service = make_service(registry);
    let urls = |result: ListWorkersResult| -> Vec<String> {
        result.workers.iter().map(|w| w.spec.url.clone()).collect()
    };

    assert_eq!(
        urls(service.list_workers(None, Some(WorkerListOrigin::Local))),
        ["http://local:30000"],
        "local: the worker the API registered"
    );
    assert_eq!(
        urls(service.list_workers(None, Some(WorkerListOrigin::Discovery))),
        ["http://discovered:30000"],
        "discovery: the worker a provider owns"
    );
    assert_eq!(service.list_workers(None, None).total, 2, "no origin: both");
}

#[tokio::test]
async fn the_reads_of_a_worker_show_an_address_that_is_not_plain_in_the_log_form() {
    let registry = Arc::new(WorkerRegistry::new());
    let live = "http://name:made-up@10.0.0.5:8000";
    register_worker(&registry, live, None);
    let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
    assert!(queue.claim_if_idle(&registration(live)));
    let held = Arc::new(std::sync::OnceLock::new());
    held.set(queue).ok();
    let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());

    let listed = service.list_workers(None, None);
    let id = listed.workers[0].id.clone();
    let read = service.get_worker(&id).expect("a read");
    let refused = service
        .create_worker(worker_spec(live))
        .await
        .expect_err("the address is registered")
        .to_string();
    let put = service
        .replace_worker(&id, worker_spec("http://10.0.0.6:8000"))
        .await
        .expect_err("a PUT with another address is refused")
        .to_string();

    assert_eq!(
        (
            listed.workers[0].spec.url.as_str(),
            read.0.spec.url.as_str()
        ),
        (live, live),
        "the load and the export read the registered address"
    );
    let mut written = Vec::new();
    for response in [listed.into_response(), read.into_response()] {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("a JSON body");
        let row = body.get("workers").map_or(&body, |rows| &rows[0]);
        written.push(row["url"].clone());
        written.extend(row["job_status"].get("worker_url").cloned());
    }
    let shown = worker_spec(live).redacted_url();
    assert_eq!(
        written,
        [json!(shown), json!(shown), json!(shown)],
        "url of the list's row, then url and job_status.worker_url of the read"
    );
    assert!(
        refused.contains(&shown) && !refused.contains("made-up"),
        "the refusal of a second registration: {refused}"
    );
    assert!(
        put.contains("URL mismatch") && !put.contains("10.0.0."),
        "the refusal of a PUT with another address names neither: {put}"
    );
}
