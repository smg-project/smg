use super::*;

#[test]
fn test_factory_creates_deepseek_r1() {
    let factory = ParserFactory::new();
    let parser = factory.create("deepseek-r1-distill");
    assert_eq!(parser.model_type(), "deepseek_r1");
}

#[test]
fn test_factory_auto_registers_deepseek_v4_models() {
    let factory = ParserFactory::new();
    assert_eq!(
        factory
            .create("deepseek-ai/DeepSeek-V4-Flash-0731")
            .model_type(),
        "deepseek_v4"
    );
    assert_eq!(
        factory.create("DEEPSEEK_V4_LOCAL").model_type(),
        "deepseek_v4"
    );
    assert_eq!(
        factory.create("deepseek-ai/DeepSeek-V4-Pro").model_type(),
        "deepseek_v4"
    );

    let mut parser = factory.create("deepseek-ai/DeepSeek-V4-Flash");
    parser.mark_reasoning_started();
    let parsed = parser
        .detect_and_parse_reasoning("analysis</think>answer")
        .unwrap();
    assert_eq!(parsed.reasoning_text, "analysis");
    assert_eq!(parsed.normal_text, "answer");
}

#[test]
fn test_factory_creates_qwen3() {
    let factory = ParserFactory::new();
    let parser = factory.create("qwen3-7b");
    assert_eq!(parser.model_type(), "qwen3");
}

#[test]
fn test_factory_routes_dotted_qwen3_generations_to_qwen3() {
    let factory = ParserFactory::new();
    // The dotted generations keep <think> reasoning; the qwen3 substring
    // pattern must keep matching their ids.
    assert_eq!(
        factory.create("Qwen/Qwen3.8-2.4T-A95B").model_type(),
        "qwen3"
    );
    assert_eq!(factory.create("Qwen/Qwen3.6-35B-A3B").model_type(), "qwen3");
}

#[test]
fn test_factory_creates_kimi() {
    let factory = ParserFactory::new();
    let parser = factory.create("kimi-chat");
    assert_eq!(parser.model_type(), "kimi");
}

#[test]
fn test_factory_creates_kimi_k3() {
    let factory = ParserFactory::new();
    // Kimi-K3 ids must resolve to the dedicated kimi_k3 parser, not the
    // legacy "kimi" parser (whose pattern is a substring of "kimi-k3").
    assert_eq!(factory.create("moonshotai/Kimi-K3").model_type(), "kimi_k3");
    assert_eq!(factory.create("Kimi_K3").model_type(), "kimi_k3");
    // The legacy kimi pattern still resolves plain Kimi-K2 ids.
    assert_eq!(factory.create("kimi-chat").model_type(), "kimi");
}

#[test]
fn test_factory_creates_inkling() {
    let factory = ParserFactory::new();
    assert_eq!(factory.create("inkling-chat").model_type(), "inkling");
}

#[test]
fn test_factory_fallback_to_passthrough() {
    let factory = ParserFactory::new();
    let parser = factory.create("unknown-model");
    assert_eq!(parser.model_type(), "passthrough");
}

#[test]
fn test_case_insensitive_matching() {
    let factory = ParserFactory::new();
    let parser1 = factory.create("DeepSeek-R1");
    let parser2 = factory.create("QWEN3");
    let parser3 = factory.create("Kimi");

    assert_eq!(parser1.model_type(), "deepseek_r1");
    assert_eq!(parser2.model_type(), "qwen3");
    assert_eq!(parser3.model_type(), "kimi");
}

#[test]
fn test_step3_model() {
    let factory = ParserFactory::new();
    let step3 = factory.create("step3-model");
    assert_eq!(step3.model_type(), "step3");
}

#[test]
fn test_glm45_model() {
    let factory = ParserFactory::new();
    let glm45 = factory.create("glm45-v2");
    assert_eq!(glm45.model_type(), "glm45");
    // GLM-5.x reuse the glm45 reasoning parser.
    assert_eq!(factory.create("glm-5.2-fp8").model_type(), "glm45");
}

#[test]
fn test_minimax_model() {
    let factory = ParserFactory::new();
    let minimax = factory.create("minimax-m2");
    assert_eq!(minimax.model_type(), "minimax");

    // Also test alternate patterns
    let mm = factory.create("mm-m2-chat");
    assert_eq!(mm.model_type(), "minimax");
}

#[test]
fn test_minimax_m3_model() {
    let factory = ParserFactory::new();

    // M3 IDs route to the M3 parser, not the broad `minimax` (M2) pattern.
    let m3 = factory.create("MiniMaxAI/MiniMax-M3");
    assert_eq!(m3.model_type(), "minimax_m3");

    let m3_quant = factory.create("nvidia/MiniMax-M3-NVFP4");
    assert_eq!(m3_quant.model_type(), "minimax_m3");

    let m3_alt = factory.create("mm-m3-chat");
    assert_eq!(m3_alt.model_type(), "minimax_m3");

    // M2 IDs still resolve to the M2 (`minimax`) parser.
    let m2 = factory.create("MiniMax-M2");
    assert_eq!(m2.model_type(), "minimax");
}

#[test]
fn test_nano_v3_model() {
    let factory = ParserFactory::new();

    let nano = factory.create("nano-v3-chat");
    assert_eq!(nano.model_type(), "nano_v3");

    let nemotron_nano = factory.create("nemotron-nano-4b");
    assert_eq!(nemotron_nano.model_type(), "nano_v3");

    // Generation-qualified names: the -3/-3.5 infix breaks the
    // nemotron-nano/nemotron-super substrings, so they match via the
    // generation prefix instead.
    let lightning = factory.create("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4");
    assert_eq!(lightning.model_type(), "nano_v3");
    let super_3 = factory.create("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8");
    assert_eq!(super_3.model_type(), "nano_v3");

    let nemotron_super = factory.create("NVIDIA-Nemotron/nemotron-super");
    assert_eq!(nemotron_super.model_type(), "nano_v3");
}

#[test]
fn test_cohere_cmd_model() {
    let factory = ParserFactory::new();

    // Test various Cohere model patterns
    let command_r = factory.create("command-r-plus");
    assert_eq!(command_r.model_type(), "cohere_cmd");

    let command_a = factory.create("command-a-03-2025");
    assert_eq!(command_a.model_type(), "cohere_cmd");

    let c4ai = factory.create("c4ai-command-r-v01");
    assert_eq!(c4ai.model_type(), "cohere_cmd");

    let cohere = factory.create("cohere-embed");
    assert_eq!(cohere.model_type(), "cohere_cmd");
}
