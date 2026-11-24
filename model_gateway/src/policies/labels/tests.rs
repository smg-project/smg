//! The cases of a worker's `policy` label under Grand Central: who
//! decides a model's policy, what stands in the spec in force behind each
//! registration and each leave, and what a refused label leaves standing.

use std::{sync::Arc, thread};

use smg_grand_central::document::PolicySpec;
use tracing_test::traced_test;

use crate::{
    app_context::AppContextBuilder,
    config::RouterConfig,
    policies::PolicyRegistry,
    routers::common::snapshot::{WorkerMonitor, WorkerRegistry},
};

const MODEL: &str = "llama";

/// A registry whose Grand Central a test's start started over `config`:
/// over the stock stages, with the maker of the specs a label builds.
fn started(config: &RouterConfig) -> Arc<PolicyRegistry> {
    started_over(config).0
}

/// [`started`], with the worker registry whose lists that engine is lent.
fn started_over(config: &RouterConfig) -> (Arc<PolicyRegistry>, Arc<WorkerRegistry>) {
    let policies = Arc::new(PolicyRegistry::of_config(config));
    let workers = Arc::new(WorkerRegistry::new());
    let monitor = WorkerMonitor::new(Arc::clone(&workers), Arc::clone(&policies), 1, false, true);
    AppContextBuilder::start_engine_by_hand(config, None, &policies, &workers, &monitor)
        .expect("the configuration's own spec");
    (policies, workers)
}

/// The settings of a gateway whose main policy is `round_robin`, with
/// Grand Central on.
fn round_robin() -> RouterConfig {
    let mut config = RouterConfig::builder()
        .round_robin_policy()
        .build_unchecked();
    config.grand_central = true;
    config
}

/// The number of the spec in force.
fn generation(policies: &PolicyRegistry) -> u64 {
    policies.engine().expect("an engine").spec().generation()
}

/// The name of the profile that places `model`'s single requests.
fn placing(policies: &PolicyRegistry, model: &str) -> Option<String> {
    policies.placing_profile(model)
}

/// The model's first registration decides its policy by its label, and the
/// spec that names it is put in force: the profile of that policy places
/// the model's single requests, and every other model stays under the
/// settings' own. A later registration's label changes nothing and is told
/// so. The decision goes with the model's last worker: the spec is built
/// without the label, and the model's next first registration names its
/// own. No policy object is built at any step.
#[tokio::test]
async fn the_first_registration_decides_a_models_policy_until_its_last_worker_leaves() {
    let policies = started(&round_robin());
    let start = generation(&policies);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));

    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("cache_aware"));
    assert_eq!(placing(&policies, "qwen").as_deref(), Some("round_robin"));
    assert_eq!(generation(&policies), start + 1);
    // A label is read in the spellings a classic registry reads it in.
    for spelling in ["cache_aware", "CacheAware", "CACHE_AWARE"] {
        assert_eq!(policies.label_places(MODEL, spelling), Some(true));
    }

    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("cache_aware"));
    assert_eq!(generation(&policies), start + 1);
    assert_eq!(policies.label_places(MODEL, "random"), Some(false));

    policies.on_model_unserved(MODEL);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    assert_eq!(generation(&policies), start + 2);

    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("random"));
    assert_eq!(generation(&policies), start + 3);
    assert_eq!(policies.classic_made_or_asked(), 0);
}

/// A registration that carries no label decides its model too: a later
/// one's label does not take the model over. Neither does a label of the
/// settings' own policy build a spec, in any spelling: that policy places
/// the model already. A name that is no policy's builds none either, and
/// is told that it places nothing.
#[tokio::test]
async fn a_registration_decides_with_no_label_and_with_one_that_builds_nothing() {
    let policies = started(&round_robin());
    let start = generation(&policies);

    policies.on_worker_added(MODEL, None);
    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    assert_eq!(policies.label_places(MODEL, "cache_aware"), Some(false));

    policies.on_worker_added("qwen", Some("RoundRobin"));
    assert_eq!(policies.label_places("qwen", "RoundRobin"), Some(true));

    policies.on_worker_added("phi", Some("no-such-policy"));
    assert_eq!(policies.label_places("phi", "no-such-policy"), Some(false));
    policies.on_worker_added("phi", Some("random"));
    assert_eq!(placing(&policies, "phi").as_deref(), Some("round_robin"));

    // No leave of such a model builds a spec either.
    for model in [MODEL, "qwen", "phi"] {
        policies.on_model_unserved(model);
    }
    assert_eq!(generation(&policies), start, "no spec was put in force");
}

/// A label the spec has no binding for fails alone: a model whose id holds
/// a `*` names no binding. The spec in force stays, one warning names the
/// reason, and the registration goes on. The model stays decided, so its
/// next registration changes nothing. Another model's label builds beside
/// it, and its leave builds again without it.
#[traced_test]
#[tokio::test]
async fn a_refused_label_fails_alone_and_leaves_the_spec_in_force() {
    let policies = started(&round_robin());
    let start = generation(&policies);

    // The lines that name a refusal of the model's label, whatever policy
    // the label names.
    let refusals = |wanted: usize| {
        logs_assert(|lines: &[&str]| {
            let refused = "of model llama* is not applied";
            let told = lines.iter().filter(|line| line.contains(refused));
            match told.filter(|line| line.contains("WARN")).count() {
                count if count == wanted => Ok(()),
                count => Err(format!("the lines that name the refusal: {count}")),
            }
        });
    };
    policies.on_worker_added("llama*", Some("cache_aware"));
    assert_eq!(generation(&policies), start);
    refusals(1);
    logs_assert(|lines: &[&str]| {
        let refused = "Worker label policy = cache_aware of model llama* is not applied";
        match lines.iter().filter(|line| line.contains(refused)).count() {
            1 => Ok(()),
            count => Err(format!("the lines that name the label's policy: {count}")),
        }
    });

    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("random"));
    assert_eq!(generation(&policies), start + 1);

    // The refused model is decided: its next registration builds nothing,
    // and is refused nothing.
    policies.on_worker_added("llama*", Some("random"));
    refusals(1);
    policies.on_model_unserved("llama*");
    assert_eq!(generation(&policies), start + 1);

    policies.on_model_unserved(MODEL);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    assert_eq!(generation(&policies), start + 2);
}

/// The registry keeps two answers of the spec in force, and stores both
/// again behind each spec a label puts in force. So a request is asked for
/// its text while a label's policy reads the prompt, and neither before the
/// label nor behind the model's last worker.
#[tokio::test]
async fn the_text_questions_follow_each_spec_a_label_puts_in_force() {
    let policies = started(&round_robin());
    assert!(!policies.any_policy_needs_request_text(None));

    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert!(policies.any_policy_needs_request_text(None));

    policies.on_model_unserved(MODEL);
    assert!(!policies.any_policy_needs_request_text(None));
}

/// Registrations of one model that arrive at once decide in turn: one of
/// them decides, and one spec is put in force.
#[tokio::test]
async fn registrations_of_one_model_at_once_put_one_spec_in_force() {
    let labels = ["cache_aware", "random", "power_of_two", "least_load"];
    let policies = started(&round_robin());
    let start = generation(&policies);
    thread::scope(|scope| {
        for label in labels {
            let policies = &policies;
            scope.spawn(move || {
                policies.on_worker_added(MODEL, Some(label));
            });
        }
    });
    assert_eq!(generation(&policies), start + 1);
    let decided = placing(&policies, MODEL).expect("a profile places the model");
    assert!(labels.contains(&decided.as_str()), "{decided}");
    let stand = labels.map(|label| policies.label_places(MODEL, label));
    assert_eq!(
        stand.iter().filter(|stands| **stands == Some(true)).count(),
        1
    );
}

/// Where the settings' spec binds no single request, a label binds nothing:
/// a pair mode of one router over gRPC plans its pairs alone. The label
/// builds the spec that stands, so none is put in force, and the label is
/// told that it places nothing.
#[tokio::test]
async fn a_label_puts_no_spec_in_force_where_it_binds_nothing() {
    let mut config = RouterConfig::builder()
        .prefill_decode_mode(
            vec![("grpc://prefill:1".to_owned(), None)],
            vec!["grpc://decode:1".to_owned()],
        )
        .grpc_connection()
        .build_unchecked();
    config.grand_central = true;
    let policies = started(&config);
    let start = generation(&policies);
    assert_eq!(placing(&policies, MODEL), None);

    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(generation(&policies), start);
    assert_eq!(placing(&policies, MODEL), None);
    assert_eq!(policies.label_places(MODEL, "cache_aware"), Some(false));
    policies.on_model_unserved(MODEL);
    assert_eq!(generation(&policies), start);
}

/// Beside a supplied spec no label is read: that spec decides, and a
/// registration puts none in force. A label is told whether the supplied
/// spec's profile for the model bears the policy's name.
#[tokio::test]
async fn beside_a_supplied_spec_no_label_is_read() {
    let supplied = PolicySpec::preset("least_load").expect("a stock preset");
    let config = RouterConfig {
        grand_central: true,
        policy_spec: Some(supplied),
        ..RouterConfig::default()
    };
    let policies = started(&config);
    let start = generation(&policies);

    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(generation(&policies), start);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("least_load"));
    assert_eq!(policies.label_places(MODEL, "cache_aware"), Some(false));
    assert_eq!(policies.label_places(MODEL, "least_load"), Some(true));
    policies.on_model_unserved(MODEL);
    assert_eq!(generation(&policies), start);
}

/// Under the classic policies the label builds a policy object, and the
/// registry answers nothing of a spec: no engine stands.
#[tokio::test]
async fn under_the_classic_policies_a_label_builds_its_policy_object() {
    let config = RouterConfig::builder()
        .round_robin_policy()
        .build_unchecked();
    let policies = PolicyRegistry::of_config(&config);
    let policy = policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(policy.name(), "cache_aware");
    assert_eq!(policies.label_places(MODEL, "cache_aware"), None);
    assert_eq!(placing(&policies, MODEL), None);
}

/// Each spec a label puts in force publishes the worker lists again, so a
/// model whose workers stay is read from a list that stands beside an empty
/// cell. Its cell holds one value of the engine however many labels came
/// and went, and the request that asks first behind a label's spec makes
/// the list's value, as behind a membership change. A registration that
/// puts no spec in force publishes no list.
///
/// The registry's list of no workers is built once and is not published
/// again. A request for a model no worker serves is placed over it with a
/// cell of its own, so the cell kept beside that list takes no value.
#[tokio::test]
async fn a_spec_a_label_puts_in_force_publishes_the_worker_lists_again() {
    use openai_protocol::worker::RuntimeType;
    use smg_worker::registry::Generation;

    use crate::routers::common::{
        placement::{select_single, PlacementInputs, Stated},
        snapshot::{test_support::grpc_worker, RoutingPool, WorkerRegistryExt},
    };

    const STAYS: &str = "qwen";
    const UNSERVED: &str = "phi";
    let (policies, workers) = started_over(&round_robin());
    let worker = grpc_worker("grpc://stays:9000", STAYS, RuntimeType::Vllm, &[]);
    workers.register(worker).expect("a worker");
    let pool = RoutingPool::GrpcPipelineRegular;
    // One request of each model: the engine is asked over the model's list.
    let place = || {
        let asked = |model| {
            let inputs = PlacementInputs::default();
            let nothing = Stated::nothing;
            select_single(&workers, &policies, model, pool, None, inputs, nothing)
        };
        assert!(asked(STAYS).is_some(), "a worker");
        assert!(asked(UNSERVED).is_none(), "no worker");
    };
    // What the engine keeps in the cell beside a model's list, as the cell
    // prints it: the count of its values, then each one's spec.
    let cell = |model| format!("{:?}", workers.get_routing_snapshot(model).lent(pool).cell);

    place();
    let published = workers.get_routing_snapshot(STAYS);
    assert!(cell(STAYS).contains("values: 1,"), "{}", cell(STAYS));
    policies.on_worker_added(STAYS, Some("RoundRobin"));
    let stands = workers.get_routing_snapshot(STAYS);
    assert!(Arc::ptr_eq(&published, &stands), "no spec, no publish");

    for _ in 0..2 {
        policies.on_worker_added(MODEL, Some("cache_aware"));
        assert!(cell(STAYS).contains("values: 0,"), "{}", cell(STAYS));
        place();
        policies.on_model_unserved(MODEL);
        place();
    }
    let now = workers.get_routing_snapshot(STAYS);
    assert!(!Arc::ptr_eq(&published, &now), "a new value");
    let same_list = Arc::ptr_eq(published.members(), now.members());
    assert!(same_list, "over the list it held");
    assert!(cell(STAYS).contains("values: 1,"), "{}", cell(STAYS));
    // Five specs, and requests of the unserved model under each: the cell
    // kept beside the list of no workers took no value.
    assert!(cell(UNSERVED).contains("values: 0,"), "{}", cell(UNSERVED));
}

/// A request for a model no worker serves is answered under Grand
/// Central as a classic gateway answers it, a 404 `model_not_found`, before
/// a label's spec and behind one. The cell kept beside the registry's list
/// of no workers takes no value of the engine at either.
#[tokio::test]
async fn a_model_no_worker_serves_is_not_found_before_and_behind_a_labels_spec() {
    use axum::http::StatusCode;
    use openai_protocol::chat::ChatCompletionRequest;

    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        middleware::{TenantKey, TenantRequestMeta},
        routers::{
            common::snapshot::{RoutingPool, WorkerRegistryExt},
            router::Router,
            RouterTrait,
        },
    };

    const UNSERVED: &str = "phi";
    let config = round_robin();
    let (policies, workers) = started_over(&config);
    let made = app_context_for_tests(config, Arc::clone(&workers), None);
    let mut context = AppContext::clone(&made);
    context.policy_registry = Arc::clone(&policies);
    let router = Router::new(&Arc::new(context)).await.expect("a router");
    let tenant = TenantRequestMeta::new(TenantKey::new("a-tenant"));
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": UNSERVED,
        "messages": [{"role": "user", "content": "hello"}],
    }))
    .expect("a chat request");
    let request = Arc::new(request);

    for labelled in [false, true] {
        if labelled {
            let start = generation(&policies);
            policies.on_worker_added(MODEL, Some("cache_aware"));
            assert_eq!(generation(&policies), start + 1, "a label's spec");
        }
        let asked = Arc::clone(&request);
        let response = router.route_chat(None, &tenant, asked, UNSERVED).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX);
        let body = body.await.expect("the answer's body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    }
    let unserved = workers.get_routing_snapshot(UNSERVED);
    let kept = format!("{:?}", unserved.lent(RoutingPool::HttpRegular).cell);
    assert!(kept.contains("values: 0,"), "{kept}");
}
