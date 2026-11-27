use openai_protocol::model_card::ModelCard;
use smg_worker::identity::basic::WorkerType;

use super::*;
use crate::routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder};

fn registry_with_card(card: ModelCard) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = BasicWorkerBuilder::new("http://w1:8000")
        .model(card)
        .worker_type(WorkerType::Regular)
        .build();
    registry.register(Arc::new(worker));
    registry
}

/// A request's card names, resolved once in preparation, are what the later reads take: the
/// names as resolved then with the legacy flags behind them, and no second walk over the
/// model's workers (a card registered after the resolution does not move them); a read with
/// nothing resolved walks the cards as before.
#[test]
fn names_resolved_once_are_read_in_place_of_the_cards() {
    let registry = registry_with_card(ModelCard::new("m").with_tool_parser("json"));
    let resolver =
        ParserResolver::with_flags(Arc::clone(&registry), None, Some("qwen3".to_string()));
    let resolved = resolver.names("m").labels;
    assert_eq!(resolved.tool.as_deref(), Some("json"));
    let later = BasicWorkerBuilder::new("http://w2:8000")
        .model(ModelCard::new("m").with_tool_parser("alpha"))
        .worker_type(WorkerType::Regular)
        .build();
    registry.register(Arc::new(later));
    let read = resolver.names_for(Some(&resolved), "m");
    assert_eq!(read.labels, resolved);
    assert_eq!(read.flags, &resolver.flags);
    assert_eq!(
        resolver.tool_parser_for(Some(&resolved), "m").as_deref(),
        Some("json")
    );
    assert_eq!(
        resolver
            .reasoning_parser_for(Some(&resolved), "m")
            .as_deref(),
        Some("qwen3")
    );
    assert_eq!(
        resolver.names_for(None, "m").labels.tool.as_deref(),
        Some("alpha")
    );
    assert_eq!(
        resolver.tool_parser_for(None, "m").as_deref(),
        Some("alpha")
    );
    assert_eq!(
        resolver.reasoning_parser_for(None, "m").as_deref(),
        Some("qwen3")
    );
}

#[test]
fn card_override_wins_over_configured() {
    let registry = registry_with_card(
        ModelCard::new("m")
            .with_tool_parser("json")
            .with_reasoning_parser("basic"),
    );
    let resolver = ParserResolver::with_flags(
        registry,
        Some("mistral".to_string()),
        Some("deepseek_r1".to_string()),
    );
    assert_eq!(resolver.tool_parser("m").as_deref(), Some("json"));
    assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("basic"));
}

#[test]
fn falls_back_to_configured_without_card_override() {
    let registry = registry_with_card(ModelCard::new("m"));
    let resolver = ParserResolver::with_flags(
        registry,
        Some("mistral".to_string()),
        Some("deepseek_r1".to_string()),
    );
    assert_eq!(resolver.tool_parser("m").as_deref(), Some("mistral"));
    assert_eq!(
        resolver.reasoning_parser("m").as_deref(),
        Some("deepseek_r1")
    );
    // Unknown model: no card, same configured fallback.
    assert_eq!(resolver.tool_parser("other").as_deref(), Some("mistral"));
}

#[test]
fn no_override_and_no_configured_resolves_none() {
    let registry = registry_with_card(ModelCard::new("m"));
    let resolver = ParserResolver::with_flags(registry, None, None);
    assert_eq!(resolver.tool_parser("m"), None);
    assert_eq!(resolver.reasoning_parser("m"), None);
}

#[test]
fn disabled_resolver_never_resolves() {
    let resolver = ParserResolver::disabled();
    assert_eq!(resolver.tool_parser("m"), None);
    assert_eq!(resolver.reasoning_parser("m"), None);
}

/// The card names read among a model's workers are the names a registry walk resolves, each
/// kind chosen on its own in the one pass (the smallest where the cards disagree, a card for
/// another model skipped), and a disabled resolver takes none of them.
#[test]
fn the_names_among_a_models_workers_are_the_registry_walks() {
    let registry = Arc::new(WorkerRegistry::new());
    let cards = [
        ModelCard::new("m")
            .with_tool_parser("zebra")
            .with_reasoning_parser("qwen3"),
        ModelCard::new("m").with_tool_parser("alpha"),
        ModelCard::new("other")
            .with_tool_parser("aardvark")
            .with_reasoning_parser("basic"),
    ];
    for (i, card) in cards.into_iter().enumerate() {
        let worker = BasicWorkerBuilder::new(format!("http://w{i}:8000"))
            .model(card)
            .worker_type(WorkerType::Regular)
            .build();
        registry.register(Arc::new(worker));
    }
    let resolver =
        ParserResolver::with_flags(Arc::clone(&registry), None, Some("flag".to_string()));
    let workers = registry.get_by_model("m");
    let read = ParserResolver::card_names_among(&workers, "m");
    let among = resolver.names_of(read.clone());
    assert_eq!(among, resolver.names("m"));
    assert_eq!(
        among.labels,
        Names {
            tool: Some("alpha".to_string()),
            reasoning: Some("qwen3".to_string()),
        }
    );
    assert_eq!(among.flags, &resolver.flags);
    assert_eq!(
        ParserResolver::card_names_among(&workers, "other"),
        Names::default()
    );
    assert_eq!(
        ParserResolver::disabled().names_of(read).labels,
        Names::default()
    );
    // The slice is the stage's snapshot: a worker registered after it was
    // taken moves a registry walk's names, not the names read among it.
    let later = BasicWorkerBuilder::new("http://w3:8000")
        .model(ModelCard::new("m").with_tool_parser("aardvark"))
        .worker_type(WorkerType::Regular)
        .build();
    registry.register(Arc::new(later));
    assert_eq!(
        ParserResolver::card_names_among(&workers, "m")
            .tool
            .as_deref(),
        Some("alpha")
    );
    assert_eq!(resolver.names("m").labels.tool.as_deref(), Some("aardvark"));
}

#[test]
fn conflicting_overrides_resolve_deterministically() {
    // Two same-model workers with different overrides: resolution must
    // not depend on registration/iteration order — the lexicographically
    // smallest name wins either way.
    for (first, second) in [("zebra", "alpha"), ("alpha", "zebra")] {
        let registry = Arc::new(WorkerRegistry::new());
        for (i, name) in [first, second].iter().enumerate() {
            let worker = BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                .model(ModelCard::new("m").with_tool_parser(*name))
                .worker_type(WorkerType::Regular)
                .build();
            registry.register(Arc::new(worker));
        }
        let resolver = ParserResolver::with_flags(registry, None, None);
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("alpha"));
    }
}

/// The legacy parsers take from the flags only the names they read: under `--parser symphony`
/// a flag may name a Symphony table or the response template, which the labels' choice of the
/// legacy parsers for one model (`passthrough` alone, a legacy tool-call name) must not hand
/// them, else the tool side parses nothing (its availability check asks for the exact name)
/// and the reasoning side warns on every request. Symphony's view of the flags
/// ([`ParserResolver::names`]) keeps them whole.
#[test]
fn the_legacy_parsers_take_from_the_flags_only_the_names_they_read() {
    let tools = ToolParserFactory::new();
    let reasoning = ReasoningParserFactory::new();
    assert!(!tools.has_parser("mistral_v13") && !reasoning.registry().has_parser("ling"));
    let passthrough = registry_with_card(ModelCard::new("m").with_reasoning_parser("passthrough"));
    let resolver = ParserResolver::new(
        Arc::clone(&passthrough),
        Some("mistral_v13".to_string()),
        Some("ling".to_string()),
        &tools,
        &reasoning,
    );
    // The flags reach the legacy parsers for no model: their own detection reads it.
    assert_eq!(resolver.tool_parser("m"), None);
    assert_eq!(resolver.tool_parser("other"), None);
    assert_eq!(resolver.reasoning_parser("other"), None);
    // The labels' names reach them as before.
    assert_eq!(
        resolver.reasoning_parser("m").as_deref(),
        Some("passthrough")
    );
    let names = resolver.names("m");
    assert_eq!(resolver.legacy_tool(&names), None);
    // Symphony sees the flags as set, beside the labels.
    assert_eq!(names.flags.tool.as_deref(), Some("mistral_v13"));
    assert_eq!(names.flags.reasoning.as_deref(), Some("ling"));
    assert_eq!(names.labels.reasoning.as_deref(), Some("passthrough"));
    // A flag the legacy parsers read passes whole; a legacy name among the labels first.
    let resolver = ParserResolver::new(
        registry_with_card(ModelCard::new("m").with_tool_parser("llama")),
        Some("qwen".to_string()),
        Some("qwen3".to_string()),
        &tools,
        &reasoning,
    );
    assert_eq!(resolver.tool_parser("m").as_deref(), Some("llama"));
    assert_eq!(resolver.tool_parser("other").as_deref(), Some("qwen"));
    assert_eq!(resolver.reasoning_parser("other").as_deref(), Some("qwen3"));
    assert_eq!(resolver.legacy_tool(&resolver.names("m")), Some("llama"));
    assert_eq!(resolver.legacy_tool(&resolver.names("other")), Some("qwen"));
    // The tests' constructor passes the flags whole, the registries unasked.
    let plain = ParserResolver::with_flags(passthrough, Some("mistral_v13".to_string()), None);
    assert_eq!(plain.tool_parser("other").as_deref(), Some("mistral_v13"));
}
