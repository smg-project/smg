//! The registry remembers what it resolved per model id and forgets it when a mapping is added.
use tool_parser::factory::ParserRegistry;

#[test]
fn a_models_parser_is_resolved_once_and_again_after_a_new_mapping() {
    let registry = ParserRegistry::new();
    registry.map_model("acme-*", "json");
    assert_eq!(
        registry.resolve_model_to_parser("org/Acme-Chat"),
        Some("json".to_string())
    );
    assert_eq!(
        registry.resolve_model_to_parser("org/Acme-Chat"),
        Some("json".to_string()),
        "the remembered answer is the same answer"
    );
    assert_eq!(registry.resolve_model_to_parser("other"), None);
    registry.map_model("other", "pythonic");
    assert_eq!(
        registry.resolve_model_to_parser("other"),
        Some("pythonic".to_string()),
        "a new mapping is seen: the remembered answers are dropped with it"
    );
    registry.map_model("org/acme-chat*", "mistral");
    assert_eq!(
        registry.resolve_model_to_parser("org/Acme-Chat"),
        Some("mistral".to_string()),
        "the longest pattern wins, remembered or not"
    );
}
