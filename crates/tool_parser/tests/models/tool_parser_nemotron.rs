//! Nemotron-3 family resolution tests.
//!
//! The Nemotron-3 generation (Nano/Super/Ultra/3.5-Lightning) renders tool
//! calls in the same XML-parameter format as Qwen3-Coder, so the family maps
//! to the `qwen_xml` parser; `nemotron` is also a named alias for explicit
//! selection. These tests pin the mapping and prove an end-to-end parse for a
//! representative model ID; the grammar itself is covered by the qwen_xml
//! suite.

use tool_parser::ParserFactory;

const LIGHTNING_ID: &str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";

#[tokio::test]
async fn nemotron_3_family_resolves_and_parses() {
    let factory = ParserFactory::new();

    for model in [
        LIGHTNING_ID,
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "nemotron-3-nano-omni",
        "nemotron-3-ultra",
    ] {
        assert!(
            factory.registry().has_parser_for_model(model),
            "{model} must resolve to a tool parser"
        );
    }

    let parser = factory.get_pooled(LIGHTNING_ID);
    let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n</parameter>\n<parameter=units>\ncelsius\n</parameter>\n</function>\n</tool_call>";
    let (_normal_text, tools) = parser.lock().await.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Santa Clara");
}

#[tokio::test]
async fn nemotron_named_alias_is_registered() {
    let factory = ParserFactory::new();
    assert!(factory.registry().has_parser("nemotron"));
}

#[tokio::test]
async fn earlier_nemotron_families_do_not_match_the_generation_glob() {
    let factory = ParserFactory::new();

    // Pre-3 families use different formats; the `nemotron-3` stem must not
    // capture them. (They may resolve via unrelated globs — llama-*, etc. —
    // but never to qwen_xml through the nemotron mapping.)
    for model in ["nemotron-4-340b-instruct", "nemotron-mini-4b"] {
        let matched = factory
            .registry()
            .resolve_model_to_parser(model)
            .unwrap_or_default();
        assert_ne!(
            matched, "qwen_xml",
            "{model} must not match the nemotron-3 mapping"
        );
    }
}
