//! Qwen3.6 / Qwen3.8 family resolution tests.
//!
//! The dotted Qwen3.x generations from 3.5 onward render tool calls in the
//! XML-parameter format (verified against each family's own published chat
//! template), so their ids must resolve to the `qwen_xml` parser instead of
//! falling through the generic `qwen*` glob to the JSON-style parser. These
//! tests pin the mapping; the grammar itself is covered by the qwen_xml suite.

use tool_parser::ParserFactory;

#[tokio::test]
async fn qwen36_and_qwen38_resolve_to_qwen_xml() {
    let factory = ParserFactory::new();
    for model in [
        "Qwen/Qwen3.8-2.4T-A95B",
        "qwen3.8-2.4t-a95b",
        "Qwen/Qwen3.6-35B-A3B",
        "qwen3.6-35b-a3b",
    ] {
        assert_eq!(
            factory.registry().resolve_model_to_parser(model).as_deref(),
            Some("qwen_xml"),
            "{model} must resolve to the XML tool parser"
        );
    }
}

#[tokio::test]
async fn earlier_qwen_families_keep_their_parsers() {
    let factory = ParserFactory::new();
    // The dotted 3.5 mapping is unchanged.
    assert_eq!(
        factory
            .registry()
            .resolve_model_to_parser("Qwen/Qwen3.5-122B-A10B")
            .as_deref(),
        Some("qwen_xml")
    );
    // Pre-3.5 families keep the JSON-style parser.
    for model in ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen3-32B", "qwen-max"] {
        assert_eq!(
            factory.registry().resolve_model_to_parser(model).as_deref(),
            Some("qwen"),
            "{model} must keep the JSON-style qwen parser"
        );
    }
}

#[tokio::test]
async fn qwen38_parses_xml_tool_call_end_to_end() {
    let factory = ParserFactory::new();
    let parser = factory.get_pooled("Qwen/Qwen3.8-2.4T-A95B");
    let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n</parameter>\n</function>\n</tool_call>";
    let (_normal, tools) = parser.lock().await.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Santa Clara");
}
