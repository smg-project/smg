use super::*;
use crate::{routers::grpc::backend_client::BackendConnector, server::http_client};

fn parts<'a>(
    cards: &'a [ModelCard],
    labels: &'a HashMap<String, String>,
    client: &Arc<reqwest::Client>,
) -> ExternalWorkerParts<'a> {
    ExternalWorkerParts {
        url: "https://api.example.com",
        engine_connector: BackendConnector::shared(),
        model_cards: cards,
        api_key: Some("key"),
        labels,
        priority: 3,
        cost: 1.5,
        http_client: Arc::clone(client),
        http2: true,
        resilience: ResolvedResilience::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        health: HealthCheckConfig::default(),
        health_endpoint: "/health",
    }
}

/// The wildcard worker and the worker with discovered models are the same
/// worker but for the models: one build, one set of settings.
#[test]
fn the_worker_is_built_the_same_way_with_and_without_model_cards() {
    let client = Arc::new(http_client());
    let labels = HashMap::from([("tier".to_string(), "gold".to_string())]);
    let cards = [ModelCard::new("m1"), ModelCard::new("m2")];
    let wildcard = build_external_worker(parts(&[], &labels, &client));
    let listed = build_external_worker(parts(&cards, &labels, &client));

    assert!(wildcard.metadata().is_wildcard());
    assert_eq!(wildcard.metadata().model_ids().count(), 0);
    assert!(!listed.metadata().is_wildcard());
    assert_eq!(
        listed.metadata().model_ids().collect::<Vec<_>>(),
        ["m1", "m2"]
    );

    for worker in [&wildcard, &listed] {
        let spec = &worker.metadata().spec;
        assert_eq!(worker.url(), "https://api.example.com");
        assert_eq!(spec.worker_type, WorkerType::Regular);
        assert_eq!(spec.connection_mode, ConnectionMode::Http);
        assert_eq!(spec.runtime_type, RuntimeType::External);
        assert_eq!(worker.api_key().map(String::as_str), Some("key"));
        assert_eq!(worker.priority(), 3);
        assert_eq!(worker.cost(), 1.5);
        assert_eq!(spec.labels, labels);
        assert_eq!(worker.metadata().health_endpoint, "/health");
    }
}
