use serde_json::json;

use super::*;

fn spec(value: serde_json::Value) -> WorkerSpec {
    serde_json::from_value(value).expect("worker spec")
}

fn with_compiled(backends: &[&str]) -> Vec<ExternalRouterSpec> {
    known::all()
        .into_iter()
        .map(|mut spec| {
            spec.compiled = backends.contains(&spec.backend);
            spec
        })
        .collect()
}

fn anthropic_only() -> Vec<ExternalRouterSpec> {
    with_compiled(&["anthropic"])
}

fn everything() -> Vec<ExternalRouterSpec> {
    with_compiled(&["openai", "anthropic", "gemini"])
}

#[test]
fn self_hosted_engines_need_no_provider_router() {
    let local = spec(json!({"url": "http://10.0.0.5:8000"}));
    assert_eq!(missing_router_among(&local, &with_compiled(&[])), None);
    let sglang = spec(json!({"url": "grpc://10.0.0.5:8000", "runtime_type": "sglang"}));
    assert_eq!(missing_router_among(&sglang, &with_compiled(&[])), None);
}

#[test]
fn a_provider_target_is_matched_to_the_router_it_would_reach() {
    let anthropic = spec(json!({"url": "https://api.anthropic.com"}));
    assert_eq!(missing_router_among(&anthropic, &anthropic_only()), None);
    assert_eq!(
        missing_router_among(&anthropic, &with_compiled(&[])).map(|m| m.feature),
        Some("provider-anthropic")
    );

    // xAI, an explicit external runtime with an unknown host, and any
    // custom provider all ride the OpenAI-compatible router.
    for value in [
        json!({"url": "https://api.x.ai"}),
        json!({"url": "https://llm.internal:8443", "runtime_type": "external"}),
        json!({"url": "https://llm.internal:8443", "provider": "together"}),
    ] {
        let worker = spec(value);
        assert_eq!(
            missing_router_among(&worker, &anthropic_only()).map(|m| m.feature),
            Some("provider-openai")
        );
        assert_eq!(missing_router_among(&worker, &everything()), None);
    }

    let gemini = spec(json!({"url": "https://generativelanguage.googleapis.com"}));
    assert_eq!(
        missing_router_among(&gemini, &anthropic_only()).map(|m| m.label),
        Some("Gemini")
    );
}

#[test]
fn a_model_that_names_its_own_provider_is_judged_by_it() {
    // A proxy on a private host: the worker says nothing about its
    // provider, the model card does. Dispatch keys on the card.
    let proxied = spec(json!({
        "url": "https://llm.internal:8443",
        "runtime_type": "external",
        "models": [{"id": "claude-3-5-sonnet", "provider": "anthropic"}]
    }));
    assert_eq!(missing_router_among(&proxied, &anthropic_only()), None);
    let openai_only = with_compiled(&["openai"]);
    assert_eq!(
        missing_router_among(&proxied, &openai_only).map(|m| m.feature),
        Some("provider-anthropic")
    );

    // A worker serving models of two providers needs both routers.
    let mixed = spec(json!({
        "url": "https://llm.internal:8443",
        "runtime_type": "external",
        "models": [
            {"id": "gpt-4o"},
            {"id": "gemini-2.5-pro", "provider": "gemini"}
        ]
    }));
    assert_eq!(
        missing_router_among(&mixed, &openai_only).map(|m| m.label),
        Some("Gemini")
    );
    assert_eq!(missing_router_among(&mixed, &everything()), None);
}

#[test]
fn discovered_cards_are_kept_only_when_their_router_is_compiled_in() {
    // A proxy that lists models of two providers on a build carrying
    // only the OpenAI-compatible router keeps the routable models and
    // names the feature for the rest.
    let worker = spec(json!({"url": "https://llm.internal:8443", "runtime_type": "external"}));
    let cards = vec![
        ModelCard::new("gpt-4o"),
        ModelCard::new("claude-3-5-sonnet").with_provider(ProviderType::Anthropic),
    ];
    let (routable, unroutable) =
        partition_routable_among(&worker, cards, &with_compiled(&["openai"]));
    assert_eq!(
        routable.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        ["gpt-4o"]
    );
    assert_eq!(unroutable.len(), 1);
    assert_eq!(unroutable[0].0.id, "claude-3-5-sonnet");
    assert_eq!(unroutable[0].1.feature, "provider-anthropic");
}
