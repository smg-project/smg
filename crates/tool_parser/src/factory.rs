// Factory and pool for creating model-specific tool parsers with pooling support.

use std::{collections::HashMap, sync::Arc};

use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
use parking_lot::RwLock;
use serde_json::json;
use tokio::sync::Mutex;

use crate::{
    parsers::{
        CohereParser, DeepSeek31Parser, DeepSeekDsmlParser, DeepSeekParser, Glm4MoeParser,
        InklingParser, JsonParser, KimiK2Parser, KimiK3Parser, LlamaParser, MinimaxM2Parser,
        MistralParser, PassthroughParser, PythonicParser, QwenParser, QwenXmlParser,
        SarashinaParser, Step3Parser,
    },
    traits::ToolParser,
};

/// Type alias for pooled parser instances.
pub type PooledParser = Arc<Mutex<Box<dyn ToolParser>>>;

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ToolParser> + Send + Sync>;

/// Function that builds the complete structural tag JSON for a set of tools.
/// Takes (tools, at_least_one) and returns the full xgrammar structural tag value.
type BuildStructuralTagFn = Arc<dyn Fn(&[Tool], bool) -> serde_json::Value + Send + Sync>;

/// Constraint type returned by [`ParserRegistry::generate_tool_constraint`].
#[derive(Debug, Clone)]
pub enum ToolConstraint {
    /// JSON schema constraint — output is pure JSON from token 0.
    /// The model-specific parser is NOT used; the generic "json" parser handles it.
    JsonSchema(String),
    /// Structural tag constraint — output includes model-native framing tokens.
    /// The model-specific parser IS used to parse the response.
    StructuralTag(String),
}

impl ToolConstraint {
    /// Convert to (type, value) tuple for gRPC constraint building.
    pub fn to_tuple(&self) -> (String, String) {
        match self {
            ToolConstraint::JsonSchema(s) => ("json_schema".to_string(), s.clone()),
            ToolConstraint::StructuralTag(s) => ("structural_tag".to_string(), s.clone()),
        }
    }

    /// Returns true if this is a JSON schema constraint.
    pub fn is_json_schema(&self) -> bool {
        matches!(self, ToolConstraint::JsonSchema(_))
    }
}

/// Registration entry for a parser: creator + optional structural tag builder.
///
/// The structural tag builder is queried at preparation time without instantiating the parser.
/// The creator is called at response processing time.
struct ParserEntry {
    creator: ParserCreator,
    build_structural_tag: Option<BuildStructuralTagFn>,
}

/// Registry for model-specific tool parsers with pooling support.
#[derive(Clone)]
pub struct ParserRegistry {
    /// Parser entries (creator + metadata)
    entries: Arc<RwLock<HashMap<String, Arc<ParserEntry>>>>,
    /// Pooled parser instances for reuse
    pool: Arc<RwLock<HashMap<String, PooledParser>>>,
    /// Model pattern to parser name mappings
    model_mapping: Arc<RwLock<HashMap<String, String>>>,
    /// Default parser name
    default_parser: Arc<RwLock<String>>,
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())),
            model_mapping: Arc::new(RwLock::new(HashMap::new())),
            default_parser: Arc::new(RwLock::new("passthrough".to_string())),
        }
    }

    /// Register a parser without structural tag support.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ToolParser> + Send + Sync + 'static,
    {
        let mut entries = self.entries.write();
        entries.insert(
            name.to_string(),
            Arc::new(ParserEntry {
                creator: Arc::new(creator),
                build_structural_tag: None,
            }),
        );
    }

    /// Register a parser with a structural tag builder.
    ///
    /// The `build_structural_tag` function takes `(&[Tool], at_least_one)` and returns
    /// the full xgrammar structural tag JSON for this parser's native tool-call format.
    pub fn register_parser_with_structural_tag<F>(
        &self,
        name: &str,
        creator: F,
        build_structural_tag: fn(&[Tool], bool) -> serde_json::Value,
    ) where
        F: Fn() -> Box<dyn ToolParser> + Send + Sync + 'static,
    {
        let mut entries = self.entries.write();
        entries.insert(
            name.to_string(),
            Arc::new(ParserEntry {
                creator: Arc::new(creator),
                build_structural_tag: Some(Arc::new(build_structural_tag)),
            }),
        );
    }

    /// Map a model name/pattern to a parser
    pub fn map_model(&self, model: impl Into<String>, parser: impl Into<String>) {
        let mut mapping = self.model_mapping.write();
        mapping.insert(model.into(), parser.into());
    }

    /// Get a pooled parser by exact name.
    /// Returns a shared parser instance from the pool, creating one if needed.
    pub fn get_pooled_parser(&self, name: &str) -> Option<PooledParser> {
        // First check if we have a pooled instance
        {
            let pool = self.pool.read();
            if let Some(parser) = pool.get(name) {
                return Some(Arc::clone(parser));
            }
        }

        // If not in pool, create one and add to pool
        let entries = self.entries.read();
        if let Some(entry) = entries.get(name) {
            let parser = Arc::new(Mutex::new((entry.creator)()));

            // Add to pool for future use
            let mut pool = self.pool.write();
            pool.insert(name.to_string(), Arc::clone(&parser));

            Some(parser)
        } else {
            None
        }
    }

    /// Check if a parser with the given name is registered.
    pub fn has_parser(&self, name: &str) -> bool {
        let entries = self.entries.read();
        entries.contains_key(name)
    }

    /// Create a fresh (non-pooled) parser instance by exact name.
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_parser(&self, name: &str) -> Option<Box<dyn ToolParser>> {
        let entries = self.entries.read();
        entries.get(name).map(|entry| (entry.creator)())
    }

    /// Check if the parser supports structural tag constraints.
    pub fn has_structural_tag(&self, name: &str) -> bool {
        let entries = self.entries.read();
        entries
            .get(name)
            .is_some_and(|e| e.build_structural_tag.is_some())
    }

    /// Check if the configured parser supports structural tags.
    /// Returns false if no parser is configured (structural tags require explicit opt-in
    /// via `--tool-call-parser`).
    pub fn has_structural_tag_for_parser(&self, configured: Option<&str>) -> bool {
        configured.is_some_and(|p| self.has_structural_tag(p))
    }

    /// Generate tool call constraint.
    ///
    /// If `configured_parser` supports structural tags → `StructuralTag(json)`.
    /// Otherwise → `JsonSchema(schema)` for required/function tool_choice.
    /// Returns `Ok(None)` for auto/none tool_choice.
    pub fn generate_tool_constraint(
        &self,
        configured_parser: Option<&str>,
        tools: &[Tool],
        tool_choice: &ToolChoice,
    ) -> Result<Option<ToolConstraint>, String> {
        if tools.is_empty() {
            return Ok(None);
        }

        let at_least_one = match tool_choice {
            ToolChoice::Value(ToolChoiceValue::Required) => true,
            ToolChoice::Function { .. } => true,
            ToolChoice::AllowedTools { mode, .. } if mode == "required" => true,
            _ => return Ok(None),
        };

        // Try structural tag from configured parser
        if let Some(name) = configured_parser {
            let entries = self.entries.read();
            if let Some(entry) = entries.get(name) {
                if let Some(build_fn) = entry.build_structural_tag.as_ref() {
                    let tag = build_fn(tools, at_least_one);
                    let json_str = serde_json::to_string(&tag)
                        .map_err(|e| format!("Failed to serialize structural tag: {e}"))?;
                    return Ok(Some(ToolConstraint::StructuralTag(json_str)));
                }
            }
        }

        // Fall back to generic JSON schema
        match tool_choice {
            ToolChoice::Function { .. } => {
                let params_schema = serde_json::to_string(&tools[0].function.parameters)
                    .map_err(|e| format!("Failed to serialize tool parameters: {e}"))?;
                Ok(Some(ToolConstraint::JsonSchema(params_schema)))
            }
            _ => {
                let schema = build_required_array_schema(tools)?;
                Ok(Some(ToolConstraint::JsonSchema(schema)))
            }
        }
    }

    /// Resolve model name to parser name via model mappings.
    /// Returns None if no mapping matches (caller decides fallback).
    pub fn resolve_model_to_parser(&self, model: &str) -> Option<String> {
        let mapping = self.model_mapping.read();
        // Try exact match
        if let Some(parser_name) = mapping.get(model) {
            return Some(parser_name.clone());
        }
        // Case-insensitive substring matching (longest pattern wins) so namespaced
        // and differently-cased ids (e.g. "org/Inkling-Chat") still resolve.
        let model_lower = model.to_lowercase();
        mapping
            .iter()
            .filter_map(|(pattern, parser_name)| {
                let stem = pattern.strip_suffix('*')?;
                model_lower
                    .contains(&stem.to_lowercase())
                    .then_some((stem, parser_name))
            })
            .max_by_key(|(stem, _)| stem.len())
            .map(|(_, parser_name)| parser_name.clone())
    }

    /// Check if a parser can be created for a specific model without actually creating it.
    pub fn has_parser_for_model(&self, model: &str) -> bool {
        self.resolve_model_to_parser(model)
            .is_some_and(|name| self.has_parser(&name))
    }

    /// Create a fresh (non-pooled) parser instance for a specific model.
    pub fn create_for_model(&self, model: &str) -> Option<Box<dyn ToolParser>> {
        let parser_name = self
            .resolve_model_to_parser(model)
            .unwrap_or_else(|| self.default_parser.read().clone());
        self.create_parser(&parser_name)
    }

    /// Get parser for a specific model
    pub fn get_pooled_for_model(&self, model: &str) -> Option<PooledParser> {
        let parser_name = self
            .resolve_model_to_parser(model)
            .unwrap_or_else(|| self.default_parser.read().clone());
        self.get_pooled_parser(&parser_name)
    }

    /// Clear the parser pool, forcing new instances to be created.
    pub fn clear_pool(&self) {
        let mut pool = self.pool.write();
        pool.clear();
    }

    /// Set the default parser
    pub fn set_default_parser(&self, name: impl Into<String>) {
        let mut default = self.default_parser.write();
        *default = name.into();
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating tool parsers based on model type.
#[derive(Clone)]
pub struct ParserFactory {
    registry: ParserRegistry,
}

impl ParserFactory {
    /// Create a new factory with default parsers registered.
    pub fn new() -> Self {
        let registry = ParserRegistry::new();

        // Register default parsers
        registry.register_parser("passthrough", || Box::new(PassthroughParser::new()));
        registry.register_parser("json", || Box::new(JsonParser::new()));
        registry.register_parser_with_structural_tag(
            "mistral",
            || Box::new(MistralParser::new()),
            MistralParser::build_structural_tag,
        );
        registry.register_parser("qwen", || Box::new(QwenParser::new()));
        registry.register_parser("qwen_xml", || Box::new(QwenXmlParser::new()));
        registry.register_parser("qwen_coder", || Box::new(QwenXmlParser::new()));
        // Nemotron-3 family emits the same XML-parameter format as Qwen3-Coder;
        // named alias so operators can select it explicitly per model.
        registry.register_parser("nemotron", || Box::new(QwenXmlParser::new()));
        registry.register_parser("pythonic", || Box::new(PythonicParser::new()));
        registry.register_parser("llama", || Box::new(LlamaParser::new()));
        registry.register_parser("deepseek", || Box::new(DeepSeekParser::new()));
        registry.register_parser("deepseek31", || Box::new(DeepSeek31Parser::new()));
        registry.register_parser("deepseek32", || Box::new(DeepSeekDsmlParser::v32()));
        registry.register_parser("deepseek_v4", || Box::new(DeepSeekDsmlParser::v4()));
        registry.register_parser("glm45_moe", || Box::new(Glm4MoeParser::glm45()));
        registry.register_parser("glm47_moe", || Box::new(Glm4MoeParser::glm47()));
        registry.register_parser("step3", || Box::new(Step3Parser::new()));
        registry.register_parser("sarashina", || Box::new(SarashinaParser::new()));
        registry.register_parser_with_structural_tag(
            "kimik2",
            || Box::new(KimiK2Parser::new()),
            KimiK2Parser::build_structural_tag,
        );
        registry.register_parser_with_structural_tag(
            "kimi_k3",
            || Box::new(KimiK3Parser::new()),
            KimiK3Parser::build_structural_tag,
        );
        registry.register_parser_with_structural_tag(
            "inkling",
            || Box::new(InklingParser::new()),
            InklingParser::build_structural_tag,
        );
        registry.register_parser("minimax_m2", || Box::new(MinimaxM2Parser::new()));
        registry.register_parser("cohere", || Box::new(CohereParser::new()));

        // Register default model mappings
        Self::register_default_mappings(&registry);

        Self { registry }
    }

    fn register_default_mappings(registry: &ParserRegistry) {
        // OpenAI models
        registry.map_model("gpt-4*", "json");
        registry.map_model("gpt-3.5*", "json");
        registry.map_model("gpt-4o*", "json");

        // Anthropic models
        registry.map_model("claude-*", "json");

        // Mistral models
        registry.map_model("mistral-*", "mistral");
        registry.map_model("mixtral-*", "mistral");

        // Qwen models (more specific patterns first - longer patterns take precedence)
        // Qwen3.5+ and Qwen3-Coder use XML format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        registry.map_model("Qwen/Qwen3.5*", "qwen_xml");
        registry.map_model("Qwen3.5*", "qwen_xml");
        registry.map_model("qwen3.5*", "qwen_xml");
        registry.map_model("qwen/qwen3.5*", "qwen_xml");
        registry.map_model("Qwen/Qwen3-Coder*", "qwen_xml");
        registry.map_model("Qwen3-Coder*", "qwen_xml");
        registry.map_model("qwen3-coder*", "qwen_xml");
        registry.map_model("qwen/qwen3-coder*", "qwen_xml");
        // Qwen3 and earlier (including Qwen2.5-Coder) use JSON format
        registry.map_model("qwen*", "qwen");
        registry.map_model("Qwen*", "qwen");

        // Nemotron-3 generation (Nano/Super/Ultra/3.5-Lightning): the chat
        // template renders tool calls in the Qwen3-Coder XML-parameter format
        // (`<tool_call>\n<function=name>\n<parameter=key>value</parameter>...`)
        // and tool results as `<tool_response>` blocks. Scoped to `nemotron-3`
        // so earlier Nemotron families with different formats never match
        // (matching is case-insensitive substring, so one spelling suffices).
        registry.map_model("nemotron-3*", "qwen_xml");

        // Llama models
        registry.map_model("llama-4*", "pythonic");
        registry.map_model("meta-llama-4*", "pythonic");
        registry.map_model("llama-3.2*", "llama");
        registry.map_model("meta-llama-3.2*", "llama");
        registry.map_model("llama-*", "json");
        registry.map_model("meta-llama-*", "json");

        // DeepSeek models
        registry.map_model("deepseek-v3*", "deepseek");
        registry.map_model("deepseek-ai/DeepSeek-V3*", "deepseek");
        registry.map_model("deepseek-v3.1*", "deepseek31");
        registry.map_model("deepseek-ai/DeepSeek-V3.1*", "deepseek31");
        // V3.2-Exp uses V3.1 format (longer patterns take precedence)
        registry.map_model("deepseek-v3.2-exp*", "deepseek31");
        registry.map_model("deepseek-ai/DeepSeek-V3.2-Exp*", "deepseek31");
        // V3.2 DSML format (outer block: function_calls)
        registry.map_model("deepseek-v3.2*", "deepseek32");
        registry.map_model("deepseek-ai/DeepSeek-V3.2*", "deepseek32");
        // V4 DSML format (outer block: tool_calls — same parser as V3.2, different block name)
        registry.map_model("deepseek-v4*", "deepseek_v4");
        registry.map_model("deepseek-ai/DeepSeek-V4*", "deepseek_v4");
        registry.map_model("deepseek-*", "pythonic");

        // GLM models
        registry.map_model("glm-4.5*", "glm45_moe");
        registry.map_model("glm-4.6*", "glm45_moe");
        registry.map_model("glm-4.7*", "glm47_moe");
        registry.map_model("glm-5*", "glm47_moe");
        registry.map_model("glm-*", "json");

        // Sarashina models
        registry.map_model("sarashina*", "sarashina");
        registry.map_model("Sarashina*", "sarashina");

        // Step3 models
        registry.map_model("step3*", "step3");
        registry.map_model("Step-3*", "step3");

        // Kimi models
        registry.map_model("kimi-k2*", "kimik2");
        registry.map_model("Kimi-K2*", "kimik2");
        registry.map_model("moonshot*/Kimi-K2*", "kimik2");
        registry.map_model("kimi-k3*", "kimi_k3");
        registry.map_model("Kimi-K3*", "kimi_k3");
        registry.map_model("moonshot*/Kimi-K3*", "kimi_k3");
        // Underscore spellings (e.g. `kimi_k3`, `Kimi_K3`) as used by some
        // checkpoint ids; mirrors the reasoning-parser factory, whose glob
        // matching is case-insensitive but still enumerates both separators.
        registry.map_model("kimi_k3*", "kimi_k3");
        registry.map_model("Kimi_K3*", "kimi_k3");
        registry.map_model("moonshot*/Kimi_K3*", "kimi_k3");

        // Inkling models use TML JSON tool calls.
        registry.map_model("inkling*", "inkling");
        registry.map_model("Inkling*", "inkling");

        // MiniMax models
        registry.map_model("minimax*", "minimax_m2");
        registry.map_model("MiniMax*", "minimax_m2");

        // Cohere models
        registry.map_model("command-r*", "cohere");
        registry.map_model("command-r-plus*", "cohere");
        registry.map_model("command-a*", "cohere");
        registry.map_model("c4ai-command*", "cohere");
        registry.map_model("cohere*", "cohere");
        registry.map_model("CohereForAI*", "cohere");

        // Other models
        registry.map_model("gemini-*", "json");
        registry.map_model("palm-*", "json");
        registry.map_model("gemma-*", "json");
    }

    /// Get a pooled parser for the given model ID.
    /// Returns a shared instance that can be used concurrently.
    /// Falls back to passthrough parser if model is not recognized.
    #[expect(
        clippy::expect_used,
        reason = "passthrough parser is registered in new(); None indicates a bug in registration logic"
    )]
    pub fn get_pooled(&self, model_id: &str) -> PooledParser {
        self.registry
            .get_pooled_for_model(model_id)
            .unwrap_or_else(|| {
                // Fallback to passthrough parser (no-op, returns text unchanged)
                self.registry
                    .get_pooled_parser("passthrough")
                    .expect("Passthrough parser should always be registered")
            })
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }

    /// Clear the parser pool.
    pub fn clear_pool(&self) {
        self.registry.clear_pool();
    }

    /// Get a non-pooled parser for the given model ID (creates a fresh instance each time).
    /// This is useful for benchmarks and testing where you want independent parser instances.
    pub fn get_parser(&self, model_id: &str) -> Option<Arc<dyn ToolParser>> {
        let parser_type = self
            .registry
            .resolve_model_to_parser(model_id)
            .unwrap_or_else(|| self.registry.default_parser.read().clone());
        let entries = self.registry.entries.read();
        entries.get(&parser_type).map(|entry| {
            let boxed_parser = (entry.creator)();
            Arc::from(boxed_parser)
        })
    }

    /// Check if a parser with the given name is registered.
    pub fn has_parser(&self, name: &str) -> bool {
        self.registry.entries.read().contains_key(name)
    }

    /// List all registered parsers (for compatibility with old API).
    pub fn list_parsers(&self) -> Vec<String> {
        self.registry.entries.read().keys().cloned().collect()
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Build JSON schema for required tool calls (array with minItems: 1).
fn build_required_array_schema(tools: &[Tool]) -> Result<String, String> {
    let mut any_of_schemas = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    // Consolidate $defs from all tools
    let mut all_defs: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for tool in tools {
        if let serde_json::Value::Object(params) = &tool.function.parameters {
            if let Some(serde_json::Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        if existing != def_schema {
                            return Err(format!(
                                "Tool definition '{def_name}' has multiple conflicting schemas"
                            ));
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    if !all_defs.is_empty() {
        if let serde_json::Value::Object(ref mut obj) = array_schema {
            obj.insert("$defs".to_string(), serde_json::Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {e}"))
}
