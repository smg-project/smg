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

/// The name of the profile a label of `policy` gives `model`, as the crate
/// names it.
fn labelled(policy: &str, model: &str) -> String {
    smg_grand_central::document::RoutingSettings::label_profile(policy, model)
}

/// The model's first registration decides its policy by its label, and the
/// spec that names it is put in force. A profile of the model's own, named
/// for that policy, places the model's single requests. Every other model
/// stays under the settings' own.
///
/// A later registration's label changes nothing and is told so. The
/// decision goes with the model's last worker: the spec is built without
/// the label, and the model's next first registration names its own. No
/// policy object is built at any step.
#[tokio::test]
async fn the_first_registration_decides_a_models_policy_until_its_last_worker_leaves() {
    let policies = started(&round_robin());
    let start = generation(&policies);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));

    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(
        placing(&policies, MODEL),
        Some(labelled("cache_aware", MODEL))
    );
    assert_eq!(placing(&policies, "qwen").as_deref(), Some("round_robin"));
    assert_eq!(generation(&policies), start + 1);
    // A label is read in the spellings a classic registry reads it in.
    for spelling in ["cache_aware", "CacheAware", "CACHE_AWARE"] {
        assert_eq!(policies.label_places(MODEL, spelling), Some(true));
    }

    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(
        placing(&policies, MODEL),
        Some(labelled("cache_aware", MODEL))
    );
    assert_eq!(generation(&policies), start + 1);
    assert_eq!(policies.label_places(MODEL, "random"), Some(false));

    policies.on_model_unserved(MODEL);
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    assert_eq!(generation(&policies), start + 2);

    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(placing(&policies, MODEL), Some(labelled("random", MODEL)));
    assert_eq!(generation(&policies), start + 3);
    assert_eq!(policies.classic_made_or_asked(), 0);
}

/// A registration that carries no label decides its model too: a later
/// one's label does not take the model over. A name that is no policy's
/// builds no spec either, and is told that it places nothing.
#[tokio::test]
async fn a_registration_decides_with_no_label_and_with_one_that_builds_nothing() {
    let policies = started(&round_robin());
    let start = generation(&policies);

    policies.on_worker_added(MODEL, None);
    policies.on_worker_added(MODEL, Some("cache_aware"));
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    assert_eq!(policies.label_places(MODEL, "cache_aware"), Some(false));
    // The settings' own policy places a model that has no label.
    assert_eq!(policies.label_places(MODEL, "RoundRobin"), Some(true));

    policies.on_worker_added("phi", Some("no-such-policy"));
    assert_eq!(policies.label_places("phi", "no-such-policy"), Some(false));
    policies.on_worker_added("phi", Some("random"));
    assert_eq!(placing(&policies, "phi").as_deref(), Some("round_robin"));

    // No leave of such a model builds a spec either.
    for model in [MODEL, "phi"] {
        policies.on_model_unserved(model);
    }
    assert_eq!(generation(&policies), start, "no spec was put in force");
}

/// A label of the settings' own policy is a label, in any spelling. Its
/// model runs a profile of its own, as a classic registry builds it an
/// object of its own. A model with no label stays under the settings'
/// shared profile. The model's leave builds the spec again without it.
///
/// Where no binding names the model's id alone, that label stands and
/// builds nothing. The main profile places the model, under the policy
/// the label names.
#[tokio::test]
async fn a_label_of_the_settings_own_policy_gives_its_model_a_profile_of_its_own() {
    let policies = started(&round_robin());
    let start = generation(&policies);

    policies.on_worker_added("qwen", Some("RoundRobin"));
    assert_eq!(generation(&policies), start + 1);
    assert_eq!(
        placing(&policies, "qwen"),
        Some(labelled("round_robin", "qwen"))
    );
    assert_eq!(placing(&policies, MODEL).as_deref(), Some("round_robin"));
    for spelling in ["round_robin", "RoundRobin"] {
        assert_eq!(policies.label_places("qwen", spelling), Some(true));
    }
    assert_eq!(policies.label_places("qwen", "random"), Some(false));

    policies.on_model_unserved("qwen");
    assert_eq!(generation(&policies), start + 2);
    assert_eq!(placing(&policies, "qwen").as_deref(), Some("round_robin"));

    policies.on_worker_added("qw*n", Some("round_robin"));
    assert_eq!(generation(&policies), start + 2, "no spec was put in force");
    assert_eq!(placing(&policies, "qw*n").as_deref(), Some("round_robin"));
    assert_eq!(policies.label_places("qw*n", "RoundRobin"), Some(true));
    assert_eq!(policies.label_places("qw*n", "random"), Some(false));
    assert_eq!(policies.classic_made_or_asked(), 0);
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
    assert_eq!(placing(&policies, MODEL), Some(labelled("random", MODEL)));
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

/// Under the sticky override each binding of the settings' spec takes a
/// routing key from the request's own id. The id stands in the body, so a
/// route reads the body of a request that carries no token ids. That
/// answer stands behind each spec a label puts in force, also where the
/// label's policy reads no text.
///
/// With the override off the question follows the spec's text alone. The
/// body is read while a label's policy reads the prompt, and neither
/// before the label nor behind the model's last worker.
#[tokio::test]
async fn the_body_question_follows_each_spec_a_label_puts_in_force() {
    use crate::config::RoutingKeyOverrideConfig;

    let mut keyed = round_robin();
    keyed.routing_key_override = RoutingKeyOverrideConfig {
        enabled: true,
        ..Default::default()
    };
    // The settings, the label, and the answer before the label, behind its
    // apply and behind the model's last leave.
    for (config, label, body) in [
        (keyed, "random", [true, true, true]),
        (round_robin(), "cache_aware", [false, true, false]),
    ] {
        let policies = started(&config);
        let start = generation(&policies);
        let asked = || policies.any_policy_needs_request_body(None);
        assert_eq!(asked(), body[0], "{label}: before the label");

        policies.on_worker_added(MODEL, Some(label));
        assert_eq!(generation(&policies), start + 1, "{label}");
        assert_eq!(asked(), body[1], "{label}: behind its apply");

        policies.on_model_unserved(MODEL);
        assert_eq!(generation(&policies), start + 2, "{label}");
        assert_eq!(asked(), body[2], "{label}: behind the last leave");
    }
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
    let decided = placing(&policies, MODEL);
    let named = labels
        .iter()
        .filter(|label| Some(labelled(label, MODEL)) == decided);
    assert_eq!(named.count(), 1, "{decided:?}");
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
    policies.on_worker_added(STAYS, None);
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

/// The start's spec reads no sent prefix under `round_robin`, so the start
/// lends the engine no tree. Neither does a label whose policy reads none.
/// A worker's label that makes its model `cache_aware` puts the first spec
/// in force that reads them, and the trees are lent there. A classic
/// registry builds that model's trees at its registration.
///
/// The engine reads the trees through the object the start handed it. A
/// conversation's next turns go back to the worker its first turn was sent
/// to, and the model's lent tree holds the conversation for that worker
/// alone. The trees are sized by the spec the label put in force.
#[tokio::test]
async fn a_label_whose_spec_first_reads_sent_prefixes_lends_the_trees() {
    use openai_protocol::worker::RuntimeType;

    use crate::routers::common::{
        placement::{select_single, PlacementInputs, Stated},
        snapshot::{test_support::grpc_worker, RoutingPool},
    };

    let (policies, workers) = started_over(&round_robin());
    assert!(policies.lent_trees().is_none(), "the start's spec");
    policies.on_worker_added("qwen", Some("random"));
    assert!(policies.lent_trees().is_none(), "a label that reads none");

    for at in 0..4 {
        let url = format!("grpc://w{at}:9000");
        let worker = grpc_worker(&url, MODEL, RuntimeType::Vllm, &[]);
        workers.register(worker).expect("a worker");
    }
    policies.on_worker_added(MODEL, Some("cache_aware"));
    let trees = policies.lent_trees().expect("the label's spec lent them");
    assert_eq!(trees.follows(), generation(&policies));
    let turn = |said: &[u32]| {
        let inputs = PlacementInputs {
            tokens: Some(said),
            ..PlacementInputs::default()
        };
        let pool = RoutingPool::GrpcPipelineRegular;
        let placed = select_single(
            &workers,
            &policies,
            MODEL,
            pool,
            None,
            inputs,
            Stated::nothing,
        );
        placed.expect("a worker").into_worker().url().to_owned()
    };
    let mut said: Vec<u32> = (0..256).collect();
    let first = turn(&said);
    for next in 1..8_u32 {
        said.extend(next * 1_000_000..next * 1_000_000 + 16);
        assert_eq!(turn(&said), first, "turn {next}");
    }
    let tree = trees.token_tree(MODEL);
    let held: Vec<String> = tree.get_tenant_token_counts().into_keys().collect();
    assert_eq!(held, [first.as_str()]);
}

/// A start whose spec reads sent prefixes lends the trees itself, and a
/// label's spec makes no second owner. The trees follow each spec a label
/// puts in force, at its arrival and at its leave. A model whose `index`
/// entry stands as it stood keeps the tree it had.
#[tokio::test]
async fn trees_the_start_lent_follow_each_spec_a_label_puts_in_force() {
    use crate::config::types::PolicyConfig;

    let mut config = RouterConfig {
        policy: PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        ..RouterConfig::default()
    };
    config.grand_central = true;
    let policies = started(&config);
    let start = generation(&policies);
    let trees = policies.lent_trees().expect("the start lent them");
    assert_eq!(trees.follows(), start);
    let tree = trees.token_tree(MODEL);

    policies.on_worker_added("qwen", Some("random"));
    assert_eq!(generation(&policies), start + 1);
    let lent = policies.lent_trees().expect("the trees stand");
    assert!(Arc::ptr_eq(&lent, &trees), "one owner");
    assert_eq!(trees.follows(), start + 1, "the label's spec");
    assert!(
        Arc::ptr_eq(&trees.token_tree(MODEL), &tree),
        "its entry stands"
    );

    policies.on_model_unserved("qwen");
    assert_eq!(generation(&policies), start + 2);
    assert_eq!(trees.follows(), start + 2, "the spec without the label");
    assert!(Arc::ptr_eq(&trees.token_tree(MODEL), &tree));
}

/// The trees are put under a label's document ahead of its apply. Where the
/// engine refuses that apply, they are put back under the document they
/// followed: the spec in force still sizes them. The engine refuses a spec
/// whose number is not above the standing one, and behind a standing spec
/// of the largest number no next number is.
#[tokio::test]
async fn a_refused_apply_puts_the_trees_back_under_the_document_they_followed() {
    use std::collections::BTreeMap;

    use smg_grand_central::stages::StageRegistry;

    use crate::config::types::PolicyConfig;

    let mut config = RouterConfig {
        policy: PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        ..RouterConfig::default()
    };
    config.grand_central = true;
    let policies = started(&config);
    let start = generation(&policies);
    let trees = policies.lent_trees().expect("the start lent them");
    assert_eq!(trees.follows(), start);

    // The start's own spec again, under the largest number: applied by hand,
    // so the trees still follow the start's document.
    let last = config
        .hosted_policy_with_labels(&BTreeMap::new(), u64::MAX, &StageRegistry::builtin())
        .expect("the settings' own spec");
    let engine = policies.engine().expect("an engine");
    assert!(engine.apply(last).is_ok(), "a higher number");
    assert_eq!(generation(&policies), u64::MAX);

    policies.on_worker_added("qwen", Some("random"));
    assert_eq!(
        generation(&policies),
        u64::MAX,
        "the label's spec is refused"
    );
    assert_eq!(placing(&policies, "qwen").as_deref(), Some("cache_aware"));
    assert_eq!(
        trees.follows(),
        start,
        "back under the document they followed"
    );
}

/// A labelled model's single request is counted under its label's policy,
/// and its traced row names that policy: what a classic selection writes.
/// The profile that places the model bears the model's id too, and that
/// name reaches neither. So a label adds no value to the selection
/// metric's `policy` label. A model with no label is counted under the
/// settings' own policy.
#[tokio::test]
async fn a_labelled_models_request_is_counted_and_traced_under_its_labels_policy() {
    use openai_protocol::worker::RuntimeType;

    use crate::{
        observability::{cache_trace, metrics::test_support::render_with_recorder},
        routers::common::{
            placement::{select_single, PlacementInputs, Stated},
            snapshot::{test_support::grpc_worker, RoutingPool},
        },
    };

    const OTHER: &str = "qwen";
    let (policies, workers) = started_over(&round_robin());
    for (url, model) in [
        ("grpc://labelled:9000", MODEL),
        ("grpc://other:9000", OTHER),
    ] {
        let worker = grpc_worker(url, model, RuntimeType::Vllm, &[]);
        workers.register(worker).expect("a worker");
    }
    policies.on_worker_added(MODEL, Some("random"));
    assert_eq!(placing(&policies, MODEL), Some(labelled("random", MODEL)));

    let pool = RoutingPool::GrpcPipelineRegular;
    let (rendered, rows) = cache_trace::scope(true, async {
        let rendered = render_with_recorder(|| {
            for model in [MODEL, OTHER] {
                let inputs = PlacementInputs::default();
                let nothing = Stated::nothing;
                let placed = select_single(&workers, &policies, model, pool, None, inputs, nothing);
                assert!(placed.is_some(), "{model}: a worker");
            }
        });
        (rendered, cache_trace::captured_selections())
    })
    .await;
    let named: Vec<_> = rows.iter().map(|row| row["policy"].as_str()).collect();
    assert_eq!(named, [Some("random"), Some("round_robin")]);
    for (model, policy) in [(MODEL, "random"), (OTHER, "round_robin")] {
        let series = format!("model=\"{model}\",policy=\"{policy}\"}} 1");
        assert!(rendered.contains(&series), "{series}:\n{rendered}");
    }
    assert!(!rendered.contains("random:"), "{rendered}");
}
