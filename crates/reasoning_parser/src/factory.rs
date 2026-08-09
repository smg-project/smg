// Factory and registry for creating model-specific reasoning parsers.

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;

use crate::{
    parsers::{
        BaseReasoningParser, CohereCmdParser, DeepSeekR1Parser, Glm45Parser, InklingParser,
        KimiK3Parser, KimiParser, MiniMaxParser, NanoV3Parser, PassthroughParser, Qwen3Parser,
        QwenThinkingParser, Step3Parser,
    },
    traits::{ParserConfig, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ReasoningParser> + Send + Sync>;

fn conditional_deepseek_parser(model_type: &str) -> Box<dyn ReasoningParser> {
    let config = ParserConfig {
        think_start_token: "<think>".to_string(),
        think_end_token: "</think>".to_string(),
        stream_reasoning: true,
        max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        always_in_reasoning: false,
    };
    Box::new(BaseReasoningParser::new(config).with_model_type(model_type.to_string()))
}

/// Registry for model-specific parsers.
#[derive(Clone)]
pub struct ParserRegistry {
    /// Creator functions for parsers
    creators: Arc<RwLock<HashMap<String, ParserCreator>>>,
    /// Model pattern to parser name mappings
    patterns: Arc<RwLock<Vec<(String, String)>>>, // (pattern, parser_name)
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            creators: Arc::new(RwLock::new(HashMap::new())),
            patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a parser creator for a given parser type.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ReasoningParser> + Send + Sync + 'static,
    {
        let mut creators = self.creators.write();
        creators.insert(name.to_string(), Arc::new(creator));
    }

    /// Register a model pattern to parser mapping.
    /// Patterns are checked in order, first match wins.
    pub fn register_pattern(&self, pattern: &str, parser_name: &str) {
        let mut patterns = self.patterns.write();
        patterns.push((pattern.to_string(), parser_name.to_string()));
    }

    /// Check if a parser with the given name is registered.
    pub fn has_parser(&self, name: &str) -> bool {
        let creators = self.creators.read();
        creators.contains_key(name)
    }

    /// Create a fresh parser instance by exact name (not pooled).
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_parser(&self, name: &str) -> Option<Box<dyn ReasoningParser>> {
        let creators = self.creators.read();
        creators.get(name).map(|creator| creator())
    }

    /// Check if a parser can be created for a specific model without actually creating it.
    /// Returns true if a parser is available (registered) for this model.
    pub fn has_parser_for_model(&self, model_id: &str) -> bool {
        let patterns = self.patterns.read();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                let creators = self.creators.read();
                return creators.contains_key(parser_name);
            }
        }
        false
    }

    /// Create a fresh parser instance for a given model ID by pattern matching (not pooled).
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_for_model(&self, model_id: &str) -> Option<Box<dyn ReasoningParser>> {
        let patterns = self.patterns.read();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                return self.create_parser(parser_name);
            }
        }
        None
    }

    /// List all registered parser names in sorted order.
    pub fn list_parsers(&self) -> Vec<String> {
        let mut parsers: Vec<_> = self.creators.read().keys().cloned().collect();
        parsers.sort_unstable();
        parsers
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating reasoning parsers based on model type.
#[derive(Clone)]
pub struct ParserFactory {
    registry: ParserRegistry,
}

impl ParserFactory {
    /// Create a new factory with default parsers registered.
    pub fn new() -> Self {
        let registry = ParserRegistry::new();

        registry.register_parser("base", || {
            Box::new(BaseReasoningParser::new(ParserConfig::default()))
        });

        // Passthrough: explicit `--reasoning-parser passthrough` and unknown-model fallback.
        registry.register_parser("passthrough", || Box::new(PassthroughParser::new()));

        // starts with in_reasoning=true
        registry.register_parser("deepseek_r1", || Box::new(DeepSeekR1Parser::new()));

        // starts with in_reasoning=false
        registry.register_parser("qwen3", || Box::new(Qwen3Parser::new()));

        // starts with in_reasoning=true
        registry.register_parser("qwen3_thinking", || Box::new(QwenThinkingParser::new()));

        // Unicode tokens, starts with in_reasoning=false
        registry.register_parser("kimi", || Box::new(KimiParser::new()));

        // glm45/step3 mirror qwen3/deepseek_r1 respectively; kept separate for debugging.
        registry.register_parser("glm45", || Box::new(Glm45Parser::new()));
        registry.register_parser("step3", || Box::new(Step3Parser::new()));

        // appends <think> token at the beginning
        registry.register_parser("minimax", || Box::new(MiniMaxParser::new()));

        // uses <|START_THINKING|> / <|END_THINKING|>
        registry.register_parser("cohere_cmd", || Box::new(CohereCmdParser::new()));

        registry.register_parser("nano_v3", || Box::new(NanoV3Parser::new()));

        registry.register_parser("inkling", || Box::new(InklingParser::new()));

        // standard think tokens, always_in_reasoning=false
        registry.register_parser("deepseek_v31", || {
            conditional_deepseek_parser("deepseek_v31")
        });

        // DeepSeek V4 uses the same delimiters, but has its own name so CLI
        // configuration and model-based selection match the served model.
        registry.register_parser("deepseek_v4", || conditional_deepseek_parser("deepseek_v4"));

        registry.register_parser("kimi_k25", || {
            let config = ParserConfig {
                think_start_token: "<think>".to_string(),
                think_end_token: "</think>".to_string(),
                stream_reasoning: true,
                max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
                always_in_reasoning: false,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("kimi_k25".to_string()))
        });

        registry.register_parser("kimi_thinking", || {
            let config = ParserConfig {
                think_start_token: "<think>".to_string(),
                think_end_token: "</think>".to_string(),
                stream_reasoning: true,
                max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
                always_in_reasoning: true,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("kimi_thinking".to_string()))
        });

        // Kimi K3 XTML think channel (structural <|open|>/<|close|>/<|sep|> tokens).
        registry.register_parser("kimi_k3", || Box::new(KimiK3Parser::new()));

        registry.register_pattern("deepseek-r1", "deepseek_r1");
        registry.register_pattern("deepseek-v3.1", "deepseek_v31");
        registry.register_pattern("deepseek-v3-1", "deepseek_v31");
        registry.register_pattern("deepseek-v4", "deepseek_v4");
        registry.register_pattern("qwen3-thinking", "qwen3_thinking");
        registry.register_pattern("qwen-thinking", "qwen3_thinking");
        registry.register_pattern("qwen3", "qwen3");
        registry.register_pattern("qwen", "qwen3");
        registry.register_pattern("glm45", "glm45");
        registry.register_pattern("glm47", "glm45"); // glm47 uses same reasoning format as glm45
        registry.register_pattern("glm-5", "glm45"); // GLM-5.x reuse glm45 reasoning format
        registry.register_pattern("kimi-k2-thinking", "kimi_thinking");
        registry.register_pattern("kimi-k2.5", "kimi_k25");
        // K3 patterns must precede the generic "kimi" pattern below, since
        // "kimi-k3".contains("kimi") and first-substring-match wins.
        registry.register_pattern("kimi-k3", "kimi_k3");
        registry.register_pattern("kimi_k3", "kimi_k3");
        registry.register_pattern("kimi", "kimi"); // legacy: Kimi-K2-Instruct with unicode tokens
        registry.register_pattern("step3", "step3");
        registry.register_pattern("minimax", "minimax");
        registry.register_pattern("minimax-m2", "minimax");
        registry.register_pattern("mm-m2", "minimax");

        // Cohere Command models use <|START_THINKING|> / <|END_THINKING|>
        registry.register_pattern("command-r", "cohere_cmd");
        registry.register_pattern("command-a", "cohere_cmd");
        registry.register_pattern("c4ai-command", "cohere_cmd");
        registry.register_pattern("cohere", "cohere_cmd");

        // Nano V3 / Nemotron (always_in_reasoning=false, uses enable_thinking toggle)
        registry.register_pattern("nemotron-nano", "nano_v3");
        registry.register_pattern("nemotron-super", "nano_v3");
        registry.register_pattern("nano-v3", "nano_v3");

        // Inkling checkpoints use the model-family name in their ID or config.
        registry.register_pattern("inkling", "inkling");

        Self { registry }
    }

    /// Create a new parser instance for the given model ID.
    /// Returns a fresh instance (not pooled).
    /// Use this when you need an isolated parser instance.
    #[expect(
        clippy::expect_used,
        reason = "passthrough parser is registered eagerly in new(); None indicates a bug in registration logic"
    )]
    pub fn create(&self, model_id: &str) -> Box<dyn ReasoningParser> {
        // First try to find by pattern
        if let Some(parser) = self.registry.create_for_model(model_id) {
            return parser;
        }

        // Fall back to passthrough
        self.registry
            .create_parser("passthrough")
            .expect("passthrough parser is registered in new()")
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }

    /// List all registered parser names.
    pub fn list_parsers(&self) -> Vec<String> {
        self.registry.list_parsers()
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creates_deepseek_r1() {
        let factory = ParserFactory::new();
        let parser = factory.create("deepseek-r1-distill");
        assert_eq!(parser.model_type(), "deepseek_r1");
    }

    #[test]
    fn test_factory_creates_deepseek_v4() {
        let factory = ParserFactory::new();
        let mut parser = factory.create("deepseek-v4-pro");
        assert_eq!(parser.model_type(), "deepseek_v4");

        // The V4 prompt consumes `<think>` before generation starts.
        parser.mark_reasoning_started();
        let result = parser
            .detect_and_parse_reasoning("reasoning</think>answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_factory_creates_qwen3() {
        let factory = ParserFactory::new();
        let parser = factory.create("qwen3-7b");
        assert_eq!(parser.model_type(), "qwen3");
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
    fn test_nano_v3_model() {
        let factory = ParserFactory::new();

        let nano = factory.create("nano-v3-chat");
        assert_eq!(nano.model_type(), "nano_v3");

        let nemotron_nano = factory.create("nemotron-nano-4b");
        assert_eq!(nemotron_nano.model_type(), "nano_v3");

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
}
