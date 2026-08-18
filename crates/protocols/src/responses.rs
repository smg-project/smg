// OpenAI Responses API types
// https://platform.openai.com/docs/api-reference/responses

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::{Validate, ValidationError};

use super::{
    common::{
        default_true, validate_stop, ChatLogProbs, ContextManagementEntry, ConversationRef, Detail,
        Function, FunctionChoice, GenerationRequest, PromptCacheRetention, PromptTokenUsageInfo,
        ResponsePrompt, StreamOptions, StringOrArray, ToolChoice as ChatToolChoice,
        ToolChoiceValue as ChatToolChoiceValue, ToolReference, UsageInfo,
    },
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::{builders::ResponsesResponseBuilder, validated::Normalizable};

// ============================================================================
// Responses API Tool Choice
// ============================================================================

/// Simple tool-choice option strings supported by the Responses API.
///
/// Spec: `tool_choice` may be the bare string `"none"`, `"auto"`, or `"required"`.
/// Shared chat/responses semantics but the Responses type owns its own enum so
/// chat-path validators cannot accept unknown Responses variants by accident.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceOptions {
    None,
    Auto,
    Required,
}

/// Single-value tag used to force the `"type": "function"` discriminator on the
/// flat function-selection variant so the untagged outer enum can distinguish
/// `Function` from `AllowedTools` / `Mcp` / `Custom` / `Types`.
///
/// Responses spec: `{"type": "function", "name": "..."}` — note the **flat**
/// shape (no nested `function` object). This differs from Chat Completions
/// which wraps the name in `{"function": {"name": "..."}}`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum FunctionToolChoiceTag {
    #[serde(rename = "function")]
    Function,
}

/// Tag enum forcing the `"type": "allowed_tools"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum AllowedToolsToolChoiceTag {
    #[serde(rename = "allowed_tools")]
    AllowedTools,
}

/// Tag enum forcing the `"type": "mcp"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum McpToolChoiceTag {
    #[serde(rename = "mcp")]
    Mcp,
}

/// Tag enum forcing the `"type": "custom"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum CustomToolChoiceTag {
    #[serde(rename = "custom")]
    Custom,
}

/// Tag enum forcing the `"type": "apply_patch"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum ApplyPatchToolChoiceTag {
    #[serde(rename = "apply_patch")]
    ApplyPatch,
}

/// Tag enum forcing the `"type": "shell"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum ShellToolChoiceTag {
    #[serde(rename = "shell")]
    Shell,
}

/// Built-in (hosted) tool types that can be referenced directly by
/// `tool_choice: {"type": "..."}` in the Responses API.
///
/// Each variant is a distinct string per the spec — we keep multiple
/// `web_search_preview*` forms because the spec enumerates them separately
/// and older clients may still send the versioned aliases.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInToolChoiceType {
    FileSearch,
    WebSearch,
    WebSearchPreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311,
    ImageGeneration,
    ComputerUsePreview,
    CodeInterpreter,
}

/// Canonical payload for the Responses API `Function` tool-choice variant.
///
/// Serializes as the spec-flat shape `{"type": "function", "name": "..."}`.
///
/// Deserialization accepts **both** wire shapes for backward compatibility
/// with smg clients written against the pre-split shared `ToolChoice` type
/// (which used the Chat-style nested `{"function": {"name": "..."}}` layout
/// on the Responses endpoint):
///
/// * Canonical flat: `{"type": "function", "name": "..."}`
/// * Legacy nested:  `{"type": "function", "function": {"name": "..."}}`
///
/// Either shape normalizes to `name: String` at deserialize time so the rest
/// of the gateway only ever sees the canonical form.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ResponsesFunctionToolChoice {
    #[serde(rename = "type")]
    pub tool_type: FunctionToolChoiceTag,
    pub name: String,
}

impl<'de> Deserialize<'de> for ResponsesFunctionToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept either the spec-flat `{type, name}` or the legacy
        // Chat-nested `{type, function: {name}}` shape. The helper lets
        // both fields be absent so serde can bind whichever wire form
        // the caller sent; we then pick one and fail loudly if neither
        // provides a function name.
        #[derive(Deserialize)]
        struct Helper {
            #[serde(rename = "type")]
            tool_type: FunctionToolChoiceTag,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            function: Option<FunctionChoice>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let name = helper
            .name
            .or_else(|| helper.function.map(|f| f.name))
            .ok_or_else(|| {
                serde::de::Error::custom(
                    "tool_choice function requires a `name` field or a `function.name` field",
                )
            })?;
        Ok(Self {
            tool_type: helper.tool_type,
            name,
        })
    }
}

/// `tool_choice` accepted on the Responses API (`POST /v1/responses`).
///
/// The Responses spec enumerates eight concrete wire shapes, each with a
/// distinct discriminator — see `ResponsesToolChoice` variants below.
/// Deserialised via `#[serde(untagged)]` because the outermost JSON is
/// either a bare string (`Options`) or an object whose `"type"` picks
/// the variant.
///
/// Each object variant pins the discriminator through a single-value tag
/// enum (`FunctionToolChoiceTag`, etc.) so serde cannot match a payload
/// whose `type` does not belong to that variant. Without the tag pinning,
/// the `#[serde(untagged)]` enum would accept any object shape that
/// happened to fit the field set of an earlier variant.
///
/// This type deliberately does NOT live in `common.rs`: Chat Completions
/// has its own `ToolChoice` with a different `Function` wire shape
/// (nested `{"function": {"name": ...}}`) and does not accept the
/// `Types` / `Mcp` / `Custom` / `ApplyPatch` / `Shell` variants at all.
/// Sharing one enum across both APIs would silently accept spec-invalid
/// payloads on `/v1/chat/completions`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// `"none"` | `"auto"` | `"required"`.
    Options(ToolChoiceOptions),

    /// `{"type": "file_search" | "web_search" | "web_search_preview" |
    /// "web_search_preview_2025_03_11" | "image_generation" |
    /// "computer_use_preview" | "code_interpreter"}` — select a built-in
    /// hosted tool by type alone (no additional payload).
    Types {
        #[serde(rename = "type")]
        tool_type: BuiltInToolChoiceType,
    },

    /// `{"type": "function", "name": "..."}` — Responses spec flat shape.
    ///
    /// Accepts both the spec-canonical flat wire shape and the legacy
    /// Chat-style nested shape (`{"type": "function", "function": {"name": "..."}}`)
    /// on deserialize to preserve backward compatibility with smg clients
    /// written against the pre-split shared `ToolChoice` type. Always
    /// serializes as the canonical flat shape per the OpenAI Responses spec
    /// — Postel's law: liberal on input, conservative on output.
    ///
    /// The nested legacy shape is gated behind a custom `Deserialize` impl on
    /// `ResponsesFunctionToolChoice`; the untagged outer enum still pins the
    /// `"type": "function"` discriminator via `FunctionToolChoiceTag` so
    /// payloads without that tag cannot reach this variant.
    Function(ResponsesFunctionToolChoice),

    /// `{"type": "allowed_tools", "mode": "auto"|"required", "tools": [...]}`.
    ///
    /// `tools` is an array of `ToolReference` items — the same type reused
    /// from Chat's Allowed Tools payload because the Responses spec also
    /// allows function / mcp / file_search / web_search_preview /
    /// computer_use_preview / code_interpreter / image_generation entries.
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: AllowedToolsToolChoiceTag,
        /// `"auto"` or `"required"`. Validated at request-normalisation time
        /// (see `validate_tool_choice_with_tools`).
        mode: String,
        tools: Vec<ToolReference>,
    },

    /// `{"type": "mcp", "server_label": "...", "name"?: "..."}` — force
    /// routing to a specific MCP server, optionally pinning a tool name.
    Mcp {
        #[serde(rename = "type")]
        tool_type: McpToolChoiceTag,
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// `{"type": "custom", "name": "..."}` — pin a user-registered
    /// custom tool by name.
    Custom {
        #[serde(rename = "type")]
        tool_type: CustomToolChoiceTag,
        name: String,
    },

    /// `{"type": "apply_patch"}` — force the built-in `apply_patch` tool.
    ApplyPatch {
        #[serde(rename = "type")]
        tool_type: ApplyPatchToolChoiceTag,
    },

    /// `{"type": "shell"}` — force the built-in `shell` tool.
    Shell {
        #[serde(rename = "type")]
        tool_type: ShellToolChoiceTag,
    },
}

impl Default for ResponsesToolChoice {
    fn default() -> Self {
        Self::Options(ToolChoiceOptions::Auto)
    }
}

impl ResponsesToolChoice {
    /// Serialize tool_choice to string for ResponsesResponse payloads.
    ///
    /// Returns the JSON-serialized tool_choice or `"auto"` as default.
    pub fn serialize_to_string(tool_choice: Option<&ResponsesToolChoice>) -> String {
        tool_choice
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }

    /// Return the pinned function name for the `Function` variant, regardless
    /// of which wire shape (spec-flat `name` or legacy nested `function.name`)
    /// was used at deserialize time. `None` for any non-`Function` variant.
    ///
    /// Consumers that need to project / validate the function name should go
    /// through this accessor rather than pattern-matching so future wire
    /// shapes can be added without touching call sites.
    pub fn function_name(&self) -> Option<&str> {
        match self {
            Self::Function(payload) => Some(payload.name.as_str()),
            _ => None,
        }
    }

    /// Project the Responses-level tool_choice onto a Chat Completions
    /// tool_choice when a Responses request is being routed through the
    /// Chat Completions gRPC pipeline.
    ///
    /// Mapping rules:
    /// - `Options(None|Auto|Required)` → `ChatToolChoice::Value(...)` — shared
    ///   semantics.
    /// - `Function { name }` → `ChatToolChoice::Function { nested name }` —
    ///   shape translation (flat Responses → nested Chat wire form).
    /// - `AllowedTools { mode, tools }` → `ChatToolChoice::AllowedTools {...}`.
    /// - Hosted / custom / apply_patch / shell / mcp — Chat Completions has no
    ///   equivalent spec variant; fall back to `Auto` so the downstream chat
    ///   backend still runs with tool-calling enabled.
    pub fn to_chat_tool_choice(&self) -> ChatToolChoice {
        match self {
            Self::Options(ToolChoiceOptions::None) => {
                ChatToolChoice::Value(ChatToolChoiceValue::None)
            }
            Self::Options(ToolChoiceOptions::Auto) => {
                ChatToolChoice::Value(ChatToolChoiceValue::Auto)
            }
            Self::Options(ToolChoiceOptions::Required) => {
                ChatToolChoice::Value(ChatToolChoiceValue::Required)
            }
            Self::Function(payload) => ChatToolChoice::Function {
                tool_type: "function".to_string(),
                function: FunctionChoice {
                    name: payload.name.clone(),
                },
            },
            Self::AllowedTools { mode, tools, .. } => ChatToolChoice::AllowedTools {
                tool_type: "allowed_tools".to_string(),
                mode: mode.clone(),
                tools: tools.clone(),
            },
            // No matching Chat spec variant — fall through to `auto` so
            // downstream Chat backends still see tool-calling enabled.
            Self::Types { .. }
            | Self::Mcp { .. }
            | Self::Custom { .. }
            | Self::ApplyPatch { .. }
            | Self::Shell { .. } => ChatToolChoice::Value(ChatToolChoiceValue::Auto),
        }
    }
}

// ============================================================================
// Response Tools (MCP and others)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseTool {
    /// Function tool.
    #[serde(rename = "function")]
    Function(FunctionTool),

    /// Built-in tool.
    #[serde(rename = "web_search_preview")]
    WebSearchPreview(WebSearchPreviewTool),

    /// Built-in non-preview hosted web search tool.
    ///
    /// Spec: `{ type: "web_search" | "web_search_2025_08_26", filters? { allowed_domains? },
    /// search_context_size?: "low"|"medium"|"high", user_location? }`. Distinct from
    /// `web_search_preview` — non-preview adds `filters.allowed_domains` and constrains
    /// `search_context_size` to a typed enum.
    #[serde(rename = "web_search", alias = "web_search_2025_08_26")]
    WebSearch(WebSearchTool),

    /// Built-in tool.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter(CodeInterpreterTool),

    /// MCP server tool.
    #[serde(rename = "mcp")]
    Mcp(McpTool),

    /// Built-in file search tool over vector stores.
    #[serde(rename = "file_search")]
    FileSearch(FileSearchTool),

    /// Built-in image generation tool. Spec:
    /// `{ type: "image_generation", action?, background?, input_fidelity?,
    ///    input_image_mask?, model?, moderation?, output_compression?,
    ///    output_format?, partial_images?, quality?, size? }`.
    #[serde(rename = "image_generation")]
    ImageGeneration(ImageGenerationTool),

    /// Generic computer tool — `{ type: "computer" }`.
    ///
    /// Spec (openai-responses-api-spec.md §tools): `Computer { type: "computer" }`.
    /// Carries no payload; the model is told that a computer-control surface is
    /// available without committing to display dimensions or environment.
    #[serde(rename = "computer")]
    Computer,

    /// Computer-use preview tool — `{ type: "computer_use_preview",
    /// display_height, display_width, environment }`.
    ///
    /// Spec (openai-responses-api-spec.md §tools): `ComputerUsePreview
    /// { display_height, display_width, environment: "browser"|"ubuntu"|
    /// "windows"|"mac", type: "computer_use_preview" }`.
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview(ComputerUsePreviewTool),

    /// User-defined custom tool with optional grammar-constrained input.
    ///
    /// Spec: `{ name, type: "custom", defer_loading?, description?, format? }`.
    /// `format` constrains the model's free-form `input` payload — `Text` is
    /// unconstrained, `Grammar` enforces a Lark or regex production at decode
    /// time. The model returns a `custom_tool_call` output item carrying the
    /// raw `input` string back to the client; the client owns execution.
    #[serde(rename = "custom")]
    Custom(CustomTool),

    /// Grouping of `Function` / `Custom` tools under a shared namespace.
    ///
    /// Spec (openai-responses-api-spec.md §tools L475):
    /// `Namespace { description, name, tools: array of Function | Custom,
    /// type: "namespace" }` — inner elements share the top-level shape but
    /// are restricted to `Function` or `Custom`. Nested namespaces and
    /// hosted/built-in tools are explicitly not permitted as elements.
    #[serde(rename = "namespace")]
    Namespace(NamespaceToolDef),

    /// Containerized `shell` tool. Distinct from `local_shell` — the tool
    /// definition itself may carry an optional [`ShellEnvironment`]
    /// (container-auto, local, or existing container reference) that the
    /// platform resolves into a concrete execution target. Emitted
    /// `shell_call` items use the narrower call-side unions
    /// [`ShellCallEnvironment`] (input path) and
    /// [`ResponseShellCallEnvironment`] (response path), which drop the
    /// `container_auto` variant.
    ///
    /// Spec (openai-responses-api-spec.md §tools, L463-470):
    /// `Shell { type: "shell", environment? }`.
    #[serde(rename = "shell")]
    Shell(ShellTool),

    /// Built-in `apply_patch` tool — `{ type: "apply_patch" }`.
    ///
    /// Spec (openai-responses-api-spec.md §tools, L478): `ApplyPatch
    /// { type: "apply_patch" }`. Unit variant with no payload — the model is
    /// simply told the apply_patch surface is available and subsequently emits
    /// `apply_patch_call` items carrying file-edit operations (see
    /// [`ResponseInputOutputItem::ApplyPatchCall`]). Pairs with
    /// [`ResponsesToolChoice::ApplyPatch`] when callers want to force usage.
    #[serde(rename = "apply_patch")]
    ApplyPatch,

    /// Built-in host-execute shell tool — `{ type: "local_shell" }`.
    ///
    /// Spec (openai-responses-api-spec.md §tools L462): `LocalShell { type:
    /// "local_shell" }` — carries no payload. Distinct from `shell`,
    /// which carries a containerized `environment`. The model emits
    /// `local_shell_call` output items carrying a `LocalShellExec` action;
    /// the client executes the command on the host and replies with a
    /// matching `local_shell_call_output` item.
    #[serde(rename = "local_shell")]
    LocalShell,
}

impl ResponseTool {
    /// Wire `type` tag for this variant — matches the variant's
    /// `#[serde(rename = ...)]` attribute. Stable across serde
    /// roundtrips and safe to use as a discriminator string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseTool::Function(_) => "function",
            ResponseTool::WebSearchPreview(_) => "web_search_preview",
            ResponseTool::WebSearch(_) => "web_search",
            ResponseTool::CodeInterpreter(_) => "code_interpreter",
            ResponseTool::Mcp(_) => "mcp",
            ResponseTool::FileSearch(_) => "file_search",
            ResponseTool::ImageGeneration(_) => "image_generation",
            ResponseTool::Computer => "computer",
            ResponseTool::ComputerUsePreview(_) => "computer_use_preview",
            ResponseTool::Custom(_) => "custom",
            ResponseTool::Namespace(_) => "namespace",
            ResponseTool::Shell(_) => "shell",
            ResponseTool::ApplyPatch => "apply_patch",
            ResponseTool::LocalShell => "local_shell",
        }
    }
}

/// Payload carried by [`ResponseTool::Namespace`].
///
/// Using a dedicated struct (rather than inline struct-variant fields) lets
/// us apply `#[serde(deny_unknown_fields)]`, matching sibling variants like
/// [`CustomTool`] and [`FunctionTool`] so unrecognized namespace-level keys
/// are rejected instead of silently swallowed.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NamespaceToolDef {
    /// Human-readable description surfaced to the model alongside the group.
    pub description: String,
    /// Stable identifier the model uses to address the namespace in
    /// `function_call` / `custom_tool_call` items (via the `namespace` field).
    pub name: String,
    /// Tools in this namespace. Spec restricts elements to `Function` or
    /// `Custom`; the dedicated [`NamespaceTool`] enum prevents nested
    /// namespaces and hosted-tool leakage that the parent `ResponseTool`
    /// enum would otherwise allow.
    pub tools: Vec<NamespaceTool>,
}

/// Element type accepted inside a [`NamespaceToolDef`]'s `tools` array.
///
/// Spec (openai-responses-api-spec.md §tools L475): namespace elements must be
/// either a `Function` or a `Custom` tool. Using a dedicated enum rather than
/// `ResponseTool` prevents recursive nesting (`Namespace` inside `Namespace`)
/// and hosted/built-in tool leakage that the spec forbids.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum NamespaceTool {
    /// Function tool — same shape as [`ResponseTool::Function`].
    #[serde(rename = "function")]
    Function(FunctionTool),

    /// Custom tool — same shape as [`ResponseTool::Custom`].
    #[serde(rename = "custom")]
    Custom(CustomTool),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionTool {
    /// Flatten to match Responses API tool JSON shape.
    #[serde(flatten)]
    pub function: Function,
}

/// File search tool configuration.
///
/// Spec: `{ type: "file_search", vector_store_ids, filters?, max_num_results?, ranking_options? }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileSearchTool {
    /// Vector store IDs to search over.
    pub vector_store_ids: Vec<String>,
    /// Optional filter applied to candidate documents.
    pub filters: Option<FileSearchFilter>,
    /// Maximum number of results to return.
    pub max_num_results: Option<u32>,
    /// Ranking options for the search.
    pub ranking_options: Option<FileSearchRankingOptions>,
}

/// Filter expression for file search.
///
/// Either a single comparison or a boolean compound (`and` / `or`) over nested filters.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum FileSearchFilter {
    #[serde(rename = "eq")]
    Eq(ComparisonFilter),
    #[serde(rename = "ne")]
    Ne(ComparisonFilter),
    #[serde(rename = "gt")]
    Gt(ComparisonFilter),
    #[serde(rename = "gte")]
    Gte(ComparisonFilter),
    #[serde(rename = "lt")]
    Lt(ComparisonFilter),
    #[serde(rename = "lte")]
    Lte(ComparisonFilter),
    #[serde(rename = "in")]
    In(ComparisonFilter),
    #[serde(rename = "nin")]
    Nin(ComparisonFilter),
    #[serde(rename = "and")]
    And(CompoundFilter),
    #[serde(rename = "or")]
    Or(CompoundFilter),
}

/// Key/value comparison used by the `eq`/`ne`/`gt`/`gte`/`lt`/`lte`/`in`/`nin` filter variants.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComparisonFilter {
    pub key: String,
    /// Spec allows `string | number | boolean | array of string | number`.
    pub value: Value,
}

/// Boolean composition over nested filters (used by `and` / `or` variants).
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompoundFilter {
    pub filters: Vec<FileSearchFilter>,
}

/// Ranking options for file search results.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileSearchRankingOptions {
    pub hybrid_search: Option<HybridSearchOptions>,
    pub ranker: Option<FileSearchRanker>,
    pub score_threshold: Option<f64>,
}

/// Weights combining embedding-based and text-based similarity.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HybridSearchOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_weight: Option<f64>,
}

/// Ranker selection for file search.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
pub enum FileSearchRanker {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default-2024-11-15")]
    Default20241115,
}

/// User-defined custom tool definition.
///
/// Spec: `{ name, type: "custom", defer_loading?, description?, format? }`.
/// The discriminator (`type: "custom"`) is enforced by the parent
/// [`ResponseTool`] enum; this struct only carries the payload fields so the
/// `flatten`-style wire shape survives a round-trip.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CustomTool {
    /// Stable identifier the model uses to address this tool in
    /// `custom_tool_call` items.
    pub name: String,
    /// Optional human-readable description supplied to the model.
    pub description: Option<String>,
    /// When `true`, the tool definition is deferred to a later
    /// `tool_search`-style fetch instead of being loaded inline.
    pub defer_loading: Option<bool>,
    /// Optional input-format constraint — `text` (unconstrained) or
    /// `grammar` (Lark / regex production).
    pub format: Option<CustomToolInputFormat>,
}

/// Input-format constraint applied to a [`CustomTool`].
///
/// Spec: `CustomToolInputFormat = Text { type: "text" } |
/// Grammar { definition, syntax: "lark" | "regex", type: "grammar" }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum CustomToolInputFormat {
    /// Unconstrained free-form text input.
    #[serde(rename = "text")]
    Text,
    /// Grammar-constrained input. The model's free-form output must match
    /// `definition` interpreted under `syntax`.
    #[serde(rename = "grammar")]
    Grammar(CustomToolGrammar),
}

/// Grammar payload carried by [`CustomToolInputFormat::Grammar`].
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CustomToolGrammar {
    /// Grammar source text. Interpretation depends on `syntax`.
    pub definition: String,
    /// `"lark"` or `"regex"` per spec.
    pub syntax: CustomToolGrammarSyntax,
}

/// Grammar dialect for [`CustomToolGrammar::syntax`].
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CustomToolGrammarSyntax {
    /// Lark grammar (https://lark-parser.readthedocs.io/).
    Lark,
    /// Regular expression.
    Regex,
}

/// Input-only content part accepted inside a `custom_tool_call_output` array.
///
/// Spec (openai-responses-api-spec.md L269): the array form of
/// `custom_tool_call_output.output` permits only `ResponseInputText |
/// ResponseInputImage | ResponseInputFile` — assistant-facing shapes such as
/// `output_text` and `refusal` are explicitly not allowed here. This enum
/// mirrors the three input variants of [`ResponseContentPart`] with identical
/// field shapes so spec-compliant payloads round-trip unchanged, while
/// deserialization of `output_text`/`refusal` fails loudly instead of being
/// silently coerced.
///
/// Other call sites that legitimately carry mixed input/output content parts
/// continue to use [`ResponseContentPart`] (Postel-of-liberality preserved for
/// cross-tool reuse).
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
// Variant names intentionally mirror the spec's `input_*` tag set; the shared
// prefix communicates the input-only restriction that justifies this enum's
// existence (see type-level docs above).
#[expect(
    clippy::enum_variant_names,
    reason = "variant names mirror spec `input_*` tags by design"
)]
pub enum CustomToolInputContentPart {
    /// `type: "input_text"` — inline textual input.
    #[serde(rename = "input_text")]
    InputText { text: String },
    /// `type: "input_image"` — reference to an image supplied by the client.
    /// Exactly one of `file_id` / `image_url` is typically set; both may be
    /// absent when only `detail` is being conveyed.
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<Detail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    /// `type: "input_file"` — reference to an attached file. `file_data` is a
    /// base64 blob; `file_url` / `file_id` reference external/uploaded files.
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<FileDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Output payload variant accepted by `custom_tool_call_output`.
///
/// Spec: `output: string or array of ResponseInputText | ResponseInputImage |
/// ResponseInputFile`. The array variant uses the restricted
/// [`CustomToolInputContentPart`] type so assistant-only shapes
/// (`output_text`, `refusal`) are rejected at the type boundary rather than
/// accepted and silently reinterpreted.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum CustomToolCallOutputContent {
    /// Plain string output.
    Text(String),
    /// Array of input-typed content parts (`input_text` / `input_image` /
    /// `input_file`) per the spec's `output` array shape.
    Parts(Vec<CustomToolInputContentPart>),
}

/// Containerized `shell` tool. Spec
/// (openai-responses-api-spec.md §tools, L463-470):
/// `Shell { type: "shell", environment? }`.
///
/// The discriminator (`type: "shell"`) is enforced by the parent
/// [`ResponseTool`] enum; this struct carries only the optional environment
/// payload so the flatten-style wire shape survives a round-trip.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellTool {
    /// Optional environment scope for the shell tool. When omitted, the
    /// model runs commands inside the platform-default environment.
    pub environment: Option<ShellEnvironment>,
}

/// Tool-side environment union carried on [`ShellTool::environment`].
///
/// Spec (openai-responses-api-spec.md §tools, L464-470):
/// `environment: ContainerAuto | LocalEnvironment | ContainerReference`.
///
/// Distinct from the call-side environment unions
/// [`ShellCallEnvironment`] (input-side call form, reuses
/// [`LocalShellEnvironment`]) and
/// [`ResponseShellCallEnvironment`] (response-side call form, carries the
/// narrower [`ResponseLocalShellEnvironment`]): the tool
/// form permits the `container_auto` variant, which asks the platform to
/// provision a new container; both call forms only carry the resolved
/// `local` / `container_reference` shape that the model echoes back.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ShellEnvironment {
    /// `type: "container_auto"` — spec L465-468. Requests a
    /// platform-provisioned container with optional `file_ids`,
    /// `memory_limit`, and `network_policy`.
    #[serde(rename = "container_auto")]
    ContainerAuto(ContainerAutoEnvironment),
    /// `type: "local"` — spec L469. Runs in the caller-owned environment.
    #[serde(rename = "local")]
    Local(LocalShellEnvironment),
    /// `type: "container_reference"` — spec L470. Pins execution to an
    /// existing container by id.
    #[serde(rename = "container_reference")]
    ContainerReference(ContainerReferenceEnvironment),
}

/// Payload for [`ShellEnvironment::ContainerAuto`]. All fields are optional
/// per spec.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContainerAutoEnvironment {
    /// Files pre-mounted into the container by id.
    pub file_ids: Option<Vec<String>>,
    /// Memory budget. Spec (CodeInterpreter §447): `"1g"|"4g"|"16g"|"64g"`
    /// — kept `String`-typed here for forward compatibility with future tiers.
    pub memory_limit: Option<String>,
    /// Network isolation policy.
    pub network_policy: Option<ContainerNetworkPolicy>,
}

/// Payload for [`ShellEnvironment::Local`]. Carries no fields.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LocalShellEnvironment {}

/// Payload for [`ShellEnvironment::ContainerReference`]. Pins the tool to
/// an existing container id.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContainerReferenceEnvironment {
    /// Existing container id.
    pub container_id: String,
}

/// Network isolation policy for [`ContainerAutoEnvironment::network_policy`].
///
/// Spec (openai-responses-api-spec.md §tools, L448):
/// `ContainerNetworkPolicyDisabled { type: "disabled" } |
/// ContainerNetworkPolicyAllowlist { allowed_domains, type: "allowlist",
/// domain_secrets? }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ContainerNetworkPolicy {
    /// `type: "disabled"` — no outbound network access.
    #[serde(rename = "disabled")]
    Disabled,
    /// `type: "allowlist"` — only the listed domains are reachable.
    #[serde(rename = "allowlist")]
    Allowlist(ContainerNetworkAllowlist),
}

/// Payload for [`ContainerNetworkPolicy::Allowlist`].
///
/// Spec (openai-responses-api-spec.md §tools, L448-449):
/// `{ allowed_domains, type: "allowlist", domain_secrets? }` where
/// `domain_secrets: array of { domain, name, value }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContainerNetworkAllowlist {
    /// Outbound-reachable hostnames.
    pub allowed_domains: Vec<String>,
    /// Optional per-domain secret bindings.
    pub domain_secrets: Option<Vec<ContainerDomainSecret>>,
}

/// Per-domain secret binding carried by
/// [`ContainerNetworkAllowlist::domain_secrets`].
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContainerDomainSecret {
    /// Target domain the secret applies to.
    pub domain: String,
    /// Secret identifier / lookup name.
    pub name: String,
    /// Secret value.
    pub value: String,
}

/// Input-side environment union carried on
/// [`ResponseInputOutputItem::ShellCall`].
///
/// Spec (openai-responses-api-spec.md §ShellCall, L228-230): the input-side
/// call form carries `environment: optional LocalEnvironment { type: "local"
/// } | ContainerReference { container_id, type: "container_reference"
/// }`. `container_auto` is rejected here — it is a request-side *tool*
/// hint (§tools L465) that the platform resolves into `local` /
/// `container_reference` before a call is surfaced, not a call-form value.
///
/// The tool-side [`LocalShellEnvironment`] is intentionally reused so
/// spec-compliant replay flows that echo back input call items round-trip
/// losslessly. Response-side emissions use the narrower
/// [`ResponseShellCallEnvironment`] instead.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ShellCallEnvironment {
    /// `type: "local"` — input-side local environment. Reuses the tool-form
    /// [`LocalShellEnvironment`] verbatim.
    #[serde(rename = "local")]
    Local(LocalShellEnvironment),
    /// `type: "container_reference"` — resolved container binding.
    #[serde(rename = "container_reference")]
    ContainerReference(ContainerReferenceEnvironment),
}

/// Response-side environment union carried on
/// [`ResponseOutputItem::ShellCall`].
///
/// Spec (openai-responses-api-spec.md §returns L512-513): the ShellCall
/// response form has `environment: ResponseLocalEnvironment { type: "local"
/// } | ResponseContainerReference { container_id, type: "container_reference"
/// }`. The response-side local arm is
/// `ResponseLocalEnvironment { type: "local" }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ResponseShellCallEnvironment {
    /// `type: "local"` — resolved local environment on the response-side
    /// call form. Per spec L513 this is `ResponseLocalEnvironment { type:
    /// "local" }`.
    #[serde(rename = "local")]
    Local(ResponseLocalShellEnvironment),
    /// `type: "container_reference"` — resolved container binding.
    /// Structurally identical to the input-side variant; reused directly.
    #[serde(rename = "container_reference")]
    ContainerReference(ContainerReferenceEnvironment),
}

/// Payload for [`ResponseShellCallEnvironment::Local`].
///
/// Spec (openai-responses-api-spec.md §returns L513): response-side local
/// environment is `ResponseLocalEnvironment { type: "local" }` — the
/// discriminator is the only field the model echoes back.
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseLocalShellEnvironment {}

/// Action payload for [`ResponseInputOutputItem::ShellCall`] /
/// [`ResponseOutputItem::ShellCall`].
///
/// Spec (openai-responses-api-spec.md §ShellCall, L229):
/// `action: { commands, max_output_length?, timeout_ms? }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellCallAction {
    /// Command line to run inside the environment, as a positional-argv
    /// array (equivalent to `argv` of `execve`).
    pub commands: Vec<String>,
    /// Optional cap on captured stdout / stderr bytes.
    pub max_output_length: Option<u64>,
    /// Optional per-command timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Status for [`ResponseInputOutputItem::ShellCall`] /
/// [`ResponseOutputItem::ShellCall`] and their `*_output` siblings.
///
/// Spec (openai-responses-api-spec.md §ShellCall, L221+L238):
/// `status: "in_progress" | "completed" | "incomplete"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShellCallStatus {
    /// `in_progress` — call is still executing.
    InProgress,
    /// `completed` — call finished and `output` chunks are final.
    Completed,
    /// `incomplete` — call aborted or never reached completion.
    Incomplete,
}

/// One entry of a `shell_call_output.output` array.
///
/// Spec (openai-responses-api-spec.md §ShellCallOutput, L234-238):
/// `{ outcome, stderr, stdout }` plus the optional `created_by` marker
/// mirroring the same tag on `FunctionCallOutput`/`ComputerCallOutput` for
/// provenance.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellOutputChunk {
    /// Call outcome — timeout or numeric exit.
    pub outcome: ShellOutcome,
    /// Captured stderr bytes (UTF-8 where possible).
    pub stderr: String,
    /// Captured stdout bytes (UTF-8 where possible).
    pub stdout: String,
    /// Optional provenance marker — `"system"`, `"user"`, or similar per
    /// spec. Kept `Option<String>` for forward-compatibility with future
    /// `created_by` tags.
    pub created_by: Option<String>,
}

/// Outcome of a shell call. Spec (openai-responses-api-spec.md
/// §ShellCallOutput, L235):
/// `outcome: Timeout { type: "timeout" } | Exit { exit_code, type: "exit" }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ShellOutcome {
    /// `type: "timeout"` — the call exceeded `action.timeout_ms` and was
    /// killed by the environment.
    #[serde(rename = "timeout")]
    Timeout,
    /// `type: "exit"` — the process exited, carrying the numeric exit code.
    #[serde(rename = "exit")]
    Exit(ShellExit),
}

/// Exit payload for [`ShellOutcome::Exit`]. Spec: `{ exit_code, type: "exit" }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellExit {
    /// Process exit code. Signed to preserve negative error codes on
    /// platforms that report them.
    pub exit_code: i32,
}

/// File-edit operation payload carried by
/// [`ResponseInputOutputItem::ApplyPatchCall`] /
/// [`ResponseOutputItem::ApplyPatchCall`].
///
/// Spec (openai-responses-api-spec.md §ApplyPatchCall L240-L246): the
/// `operation` field is a `type`-tagged union of three shapes —
/// `CreateFile { diff, path, type: "create_file" }`,
/// `DeleteFile { path, type: "delete_file" }`, and
/// `UpdateFile { diff, path, type: "update_file" }`. `DeleteFile` carries no
/// diff because the whole file is removed; the other two carry a unified
/// diff payload describing the edit.
///
/// `deny_unknown_fields` is applied so variants fail fast on foreign keys —
/// e.g. `{"type":"delete_file","path":"x","diff":"..."}` is rejected rather
/// than silently dropping the stray `diff`, matching the P5 fail-fast
/// contract applied elsewhere on protocol-surface structs.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
// Variant names intentionally mirror the spec's `*_file` tag set; the shared
// `File` postfix tracks the spec verbatim and keeps the type discriminator
// symmetric with the JSON wire shape.
#[expect(
    clippy::enum_variant_names,
    reason = "variant names mirror spec `create_file` / `delete_file` / `update_file` tags by design"
)]
pub enum ApplyPatchOperation {
    /// `{ type: "create_file", diff, path }` — create a new file whose
    /// contents are described by `diff`.
    CreateFile { diff: String, path: String },
    /// `{ type: "delete_file", path }` — remove an existing file.
    DeleteFile { path: String },
    /// `{ type: "update_file", diff, path }` — apply a unified diff to an
    /// existing file at `path`.
    UpdateFile { diff: String, path: String },
}

/// Status for a [`ResponseInputOutputItem::ApplyPatchCall`] /
/// [`ResponseOutputItem::ApplyPatchCall`] item.
///
/// Spec (openai-responses-api-spec.md §ApplyPatchCall L245):
/// `status: "in_progress" | "completed"`. Distinct from
/// [`ApplyPatchCallOutputStatus`] which adds `"failed"` for the output item.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApplyPatchCallStatus {
    InProgress,
    Completed,
}

/// Status for a [`ResponseInputOutputItem::ApplyPatchCallOutput`] /
/// [`ResponseOutputItem::ApplyPatchCallOutput`] item.
///
/// Spec (openai-responses-api-spec.md §ApplyPatchCallOutput L249):
/// `status: "completed" | "failed"`. The output-side status intentionally
/// drops `"in_progress"` because the output only materialises once the
/// apply_patch attempt has terminated — either the edit applied cleanly or
/// it failed — so an in-progress output would be spec-invalid.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApplyPatchCallOutputStatus {
    Completed,
    Failed,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct McpTool {
    pub server_url: Option<String>,
    pub authorization: Option<String>,
    /// Custom headers to send to MCP server (from request payload, not HTTP headers)
    pub headers: Option<HashMap<String, String>>,
    pub server_label: String,
    pub server_description: Option<String>,
    /// Approval requirement configuration for MCP tools.
    pub require_approval: Option<RequireApproval>,
    /// List of allowed tool names, or a filter object with `read_only` /
    /// `tool_names` fields. Spec (openai-responses-api-spec.md L442):
    /// `allowed_tools: McpAllowedTools = array of string | McpToolFilter`.
    /// Backward-compat: legacy `["a","b"]` wire shape still deserializes via
    /// the untagged `List` variant.
    pub allowed_tools: Option<McpAllowedTools>,
    /// Identifier for service connectors (e.g. Dropbox, Gmail). One of
    /// `server_url` or `connector_id` must be provided per spec
    /// (openai-responses-api-spec.md L441-445).
    pub connector_id: Option<McpConnectorId>,
    /// When `true`, the MCP server's tool list is fetched lazily on first
    /// use rather than eagerly at request time. Spec
    /// (openai-responses-api-spec.md L441): `defer_loading?: bool`.
    /// Not yet present in OpenAI Python SDK 2.8.1 `types/responses/tool.py`;
    /// included here to track the documented Responses API surface.
    pub defer_loading: Option<bool>,
}

/// Allowed-tools filter for an MCP tool.
///
/// Spec (openai-responses-api-spec.md L442): `array of string | McpToolFilter`.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `List` first
/// (JSON array) so a bare `["foo","bar"]` wire shape keeps deserializing into
/// `List(vec!["foo","bar"])`. A JSON object falls through to `Filter`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(untagged)]
pub enum McpAllowedTools {
    /// Flat allow-list of tool names.
    List(Vec<String>),
    /// Object filter selecting tools by read-only flag and/or explicit names.
    Filter(McpToolFilter),
}

/// Object filter selecting MCP tools by read-only flag and/or explicit names.
///
/// Spec (openai-responses-api-spec.md L442): `McpToolFilter { read_only?, tool_names? }`.
///
/// `deny_unknown_fields` is applied so typoed keys (e.g. `"tool_namse"`) fail
/// fast at deserialize time rather than silently collapsing to an empty filter
/// — an empty filter would be projected by the router to "no name constraint",
/// unexpectedly broadening MCP tool exposure for payloads that meant to scope
/// it down.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct McpToolFilter {
    /// Match tools flagged read-only via MCP `readOnlyHint` annotation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    /// Explicit list of allowed tool names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
}

/// Service-connector identifier for hosted MCP tools.
///
/// Spec (openai-responses-api-spec.md L443): exactly one of `server_url` or
/// `connector_id` is required on `McpTool`. The wire values are literal
/// snake-case strings such as `"connector_dropbox"`.
///
/// Enum values mirror OpenAI Python SDK 2.8.1
/// `types/responses/tool.py::Mcp.connector_id`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
pub enum McpConnectorId {
    #[serde(rename = "connector_dropbox")]
    Dropbox,
    #[serde(rename = "connector_gmail")]
    Gmail,
    #[serde(rename = "connector_googlecalendar")]
    GoogleCalendar,
    #[serde(rename = "connector_googledrive")]
    GoogleDrive,
    #[serde(rename = "connector_microsoftteams")]
    MicrosoftTeams,
    #[serde(rename = "connector_outlookcalendar")]
    OutlookCalendar,
    #[serde(rename = "connector_outlookemail")]
    OutlookEmail,
    #[serde(rename = "connector_sharepoint")]
    SharePoint,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WebSearchPreviewTool {
    pub search_context_size: Option<String>,
    pub user_location: Option<Value>,
}

/// Non-preview hosted web search tool configuration.
///
/// Spec: `{ type: "web_search" | "web_search_2025_08_26", filters? { allowed_domains? },
/// return_token_budget?: "default"|"unlimited", search_context_size?: "low"|"medium"|"high",
/// user_location? }`.
///
/// Distinct from `WebSearchPreviewTool`: adds `filters.allowed_domains` (domain
/// allowlist), `return_token_budget`, and pins `search_context_size` to the spec-listed enum.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WebSearchTool {
    /// Optional domain allowlist applied to candidate sources.
    pub filters: Option<WebSearchFilters>,
    /// Search-result context token budget. Spec enum: `"default" | "unlimited"`.
    pub return_token_budget: Option<WebSearchReturnTokenBudget>,
    /// Search context budget. Spec enum: `"low" | "medium" | "high"`.
    pub search_context_size: Option<WebSearchContextSize>,
    /// Approximate user location used to bias results.
    pub user_location: Option<WebSearchUserLocation>,
}

/// Filters for the non-preview `web_search` tool.
///
/// Spec: `filters? { allowed_domains? }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WebSearchFilters {
    /// Optional list of domains to restrict search results to.
    pub allowed_domains: Option<Vec<String>>,
}

/// Search context budget for the non-preview `web_search` tool.
///
/// Spec: `search_context_size?: "low" | "medium" | "high"`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchContextSize {
    Low,
    Medium,
    High,
}

/// Search-result token budget for the non-preview `web_search` tool.
///
/// Spec: `return_token_budget?: "default" | "unlimited"`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchReturnTokenBudget {
    Default,
    Unlimited,
}

/// Approximate user location for the non-preview `web_search` tool.
///
/// Spec: `user_location: { city?, country?: ISO2, region?, timezone?: IANA, type?: "approximate" }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WebSearchUserLocation {
    /// City name.
    pub city: Option<String>,
    /// ISO-3166-1 alpha-2 country code (e.g. `"US"`).
    pub country: Option<String>,
    /// Region / state / province name.
    pub region: Option<String>,
    /// IANA timezone identifier (e.g. `"America/Los_Angeles"`).
    pub timezone: Option<String>,
    /// Discriminator. Spec only enumerates `"approximate"`.
    #[serde(rename = "type")]
    pub location_type: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CodeInterpreterTool {
    pub container: Option<Value>,
    pub environment: Option<ResponseToolEnvironment>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseToolEnvironment {}

/// Configuration payload for the `image_generation` built-in tool.
///
/// Spec: `{ type: "image_generation", action?, background?, input_fidelity?,
/// input_image_mask?, model?, moderation?, output_compression?, output_format?,
/// partial_images?, quality?, size? }`. All inner fields are optional; the
/// model picks defaults documented in the spec.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationTool {
    /// `"generate" | "edit" | "auto"` (default `auto`). Free-form string here so
    /// future actions added by OpenAI deserialize without a wire break.
    pub action: Option<String>,
    /// `"transparent" | "opaque" | "auto"` (default `auto`).
    pub background: Option<String>,
    /// `"high" | "low"` — gpt-image-1 / gpt-image-1.5 only (not 1-mini).
    pub input_fidelity: Option<String>,
    /// Reference image used by `edit` action; mask either by uploaded file or URL.
    pub input_image_mask: Option<ImageInputMask>,
    /// `string | "gpt-image-1" | "gpt-image-1-mini" | "gpt-image-1.5"` — kept as
    /// `String` so unknown model identifiers passed through unchanged.
    pub model: Option<String>,
    /// `"auto" | "low"`.
    pub moderation: Option<String>,
    /// Output compression level. Spec default `100` when omitted; we keep
    /// `Option` so an unset field round-trips as absent (not `null`, via
    /// `#[serde_with::skip_serializing_none]`) rather than forcing 100.
    pub output_compression: Option<u32>,
    /// `"png" | "webp" | "jpeg"`.
    pub output_format: Option<String>,
    /// `0..3` — number of partial images to stream.
    pub partial_images: Option<u32>,
    /// `"low" | "medium" | "high" | "auto"`.
    pub quality: Option<String>,
    /// `"1024x1024" | "1024x1536" | "1536x1024" | "auto"`.
    pub size: Option<String>,
}

/// Mask reference for image-generation `edit` calls. Spec: `{ file_id?, image_url? }`.
/// Reuses the same upload conventions as P1 `InputImage`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageInputMask {
    pub file_id: Option<String>,
    pub image_url: Option<String>,
}

/// Status values for an `image_generation_call` output item.
///
/// Spec: `"in_progress" | "completed" | "generating" | "failed"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationCallStatus {
    InProgress,
    Completed,
    Generating,
    Failed,
}

/// `require_approval` values.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(untagged)]
pub enum RequireApproval {
    Mode(RequireApprovalMode),
    Rules(RequireApprovalRules),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RequireApprovalMode {
    Always,
    Never,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequireApprovalRules {
    pub always: Option<RequireApprovalFilter>,
    pub never: Option<RequireApprovalFilter>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequireApprovalFilter {
    pub tool_names: Option<Vec<String>>,
    pub read_only: Option<bool>,
}

// ============================================================================
// Computer Tool
// ============================================================================

/// Computer-use preview tool payload.
///
/// Spec (openai-responses-api-spec.md §tools): `ComputerUsePreview
/// { display_height, display_width, environment, type: "computer_use_preview" }`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComputerUsePreviewTool {
    /// Height of the simulated display in pixels.
    pub display_height: u32,
    /// Width of the simulated display in pixels.
    pub display_width: u32,
    /// Operating environment the model should target.
    pub environment: ComputerEnvironment,
}

/// Environment selector for [`ComputerUsePreviewTool`].
///
/// Spec (openai-responses-api-spec.md §tools): `environment:
/// "windows"|"mac"|"linux"|"ubuntu"|"browser"`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComputerEnvironment {
    Windows,
    Mac,
    Linux,
    Ubuntu,
    Browser,
}

/// Mouse button used by the `Click` action.
///
/// Spec (openai-responses-api-spec.md §ComputerAction): `button:
/// "left"|"right"|"wheel"|"back"|"forward"`. Only the `Click` action carries a
/// `button` field; `Scroll` uses `scroll_x`/`scroll_y` offsets.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    Left,
    Right,
    Wheel,
    Back,
    Forward,
}

/// `(x, y)` coordinate pair used in `Drag.path` and `Move/Click/Scroll` actions.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComputerCoordinate {
    pub x: i32,
    pub y: i32,
}

/// Discriminated union of computer-use actions the model may emit.
///
/// Spec (openai-responses-api-spec.md §ComputerAction):
/// `Click | DoubleClick | Drag | Keypress | Move | Screenshot | Scroll | Type | Wait`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerAction {
    /// `{ type: "click", button, x, y, keys? }`.
    Click {
        button: MouseButton,
        x: i32,
        y: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    /// `{ type: "double_click", x, y, keys? }`.
    DoubleClick {
        x: i32,
        y: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    /// `{ type: "drag", path, keys? }`.
    Drag {
        path: Vec<ComputerCoordinate>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    /// `{ type: "keypress", keys }`.
    Keypress { keys: Vec<String> },
    /// `{ type: "move", x, y, keys? }`.
    Move {
        x: i32,
        y: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    /// `{ type: "screenshot" }`.
    Screenshot,
    /// `{ type: "scroll", scroll_x, scroll_y, x, y, keys? }`.
    Scroll {
        scroll_x: i32,
        scroll_y: i32,
        x: i32,
        y: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    /// `{ type: "type", text }`.
    Type { text: String },
    /// `{ type: "wait" }`.
    Wait,
}

/// Status for a [`ResponseInputOutputItem::ComputerCall`] /
/// [`ResponseOutputItem::ComputerCall`] item.
///
/// Spec (openai-responses-api-spec.md §ComputerCall): `status: "in_progress"
/// | "completed" | "incomplete"`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComputerCallStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// One pending or acknowledged safety check attached to a computer-use call.
///
/// Spec (openai-responses-api-spec.md §ComputerCall): `pending_safety_checks:
/// array of { id, code, message }`. Per SDK v2.8.1
/// (`types/responses/response_computer_tool_call.py::PendingSafetyCheck`),
/// `code` and `message` are `Optional[str] = None`, so we mirror that here to
/// round-trip payloads that omit either field.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComputerSafetyCheck {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Output payload of a [`ResponseInputOutputItem::ComputerCallOutput`] item.
///
/// Spec (openai-responses-api-spec.md §ComputerCallOutput.output):
/// `ResponseComputerToolCallOutputScreenshot { type: "computer_screenshot",
/// file_id?, image_url? }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerCallOutputContent {
    /// `{ type: "computer_screenshot", file_id?, image_url? }`.
    ComputerScreenshot {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
}

// ============================================================================
// Reasoning Parameters
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ResponseReasoningParam {
    #[serde(default = "default_reasoning_effort")]
    pub effort: Option<ReasoningEffort>,
    pub summary: Option<ReasoningSummary>,
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<T>"
)]
fn default_reasoning_effort() -> Option<ReasoningEffort> {
    Some(ReasoningEffort::Medium)
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

// ============================================================================
// Input/Output Items
// ============================================================================

/// Content can be either a simple string or array of content parts (for SimpleInputMessage)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum StringOrContentParts {
    String(String),
    Array(Vec<ResponseContentPart>),
}

/// Phase label for assistant messages in the Responses API.
///
/// For gpt-5.3-codex+ multi-turn conversations, preserving and resending the
/// original `phase` value avoids quality / latency degradation — the model
/// relies on it to disambiguate commentary-style reasoning from the final
/// answer. Opaque to SMG otherwise; preserved verbatim through store+retrieve,
/// SSE output, and upstream passthrough.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessagePhase {
    Commentary,
    FinalAnswer,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseInputOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Optional phase label, preserved from previous assistant output so
        /// gpt-5.3-codex+ multi-turn does not degrade (spec: ResponseOutputMessage.phase).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    #[serde(rename = "reasoning")]
    #[non_exhaustive]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryTextContent>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        #[serde(default)]
        content: Vec<ResponseReasoningContent>,
        /// Encrypted reasoning payload for gpt-5 / o-series round-trip via
        /// `previous_response_id`. Opaque to SMG; preserved verbatim.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "mcp_approval_response")]
    McpApprovalResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        approval_request_id: String,
        approve: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// `type: "image_generation_call"` — round-trip form for an image generated
    /// in a prior turn. Spec (OpenAI Responses API, multi-turn image-edit
    /// flow): clients may resubmit only `{ type, id }` to reference a prior
    /// generation by identifier, so `result` and `status` are accepted as
    /// absent on the input side. The full shape is
    /// `{ id, action?, background?, output_format?, quality?, result?: base64,
    /// revised_prompt?, size?, status?, type }`.
    ///
    /// This mirrors the OpenAI Python SDK 2.8.x
    /// `response_input_item_param.ImageGenerationCall` TypedDict: while the
    /// TypedDict types those fields as `Required[Optional[...]]`, the HTTP
    /// API itself documents the id-only multi-turn reference form (see the
    /// image-generation tool guide), and `skip_serializing_if` keeps the
    /// serialized form spec-compatible when a full item is round-tripped.
    /// The server-side `ResponseOutputItem::ImageGenerationCall` variant
    /// carries the same metadata so real OpenAI responses
    /// (`action`/`background`/`output_format`/`quality`/`size`) survive
    /// cloud-passthrough and persistence round-trips.
    ///
    /// The metadata fields (`action`, `background`, `output_format`,
    /// `quality`, `size`) are typed as `Option<String>` rather than
    /// narrow enums so unknown or future-added values pass through
    /// unchanged — this mirrors `ImageGenerationTool` on the input-tool
    /// side.
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        id: String,
        /// `"generate" | "edit" | "auto"` — which image-generation action the
        /// prior turn dispatched. Preserved free-form so future actions pass
        /// through without a wire break.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<String>,
        /// `"transparent" | "opaque" | "auto"`. Matches the
        /// `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<String>,
        /// `"png" | "webp" | "jpeg"`. Matches the `image_generation` tool
        /// input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_format: Option<String>,
        /// `"auto" | "low" | "medium" | "high" | "standard" | "hd"`. Matches
        /// the `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quality: Option<String>,
        /// Base64-encoded image bytes. Omitted on id-only references.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        /// Prompt text the mainline model rewrote before dispatching the
        /// image-generation call. Preserved so downstream turns/storage do
        /// not drop it on replay.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revised_prompt: Option<String>,
        /// `"auto" | "1024x1024" | "1024x1536" | "1536x1024"`. Matches the
        /// `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size: Option<String>,
        /// Generation status. Omitted on id-only references.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<ImageGenerationCallStatus>,
    },
    /// `type: "compaction"` — opaque compacted-history payload generated by
    /// the `/v1/responses/compact` API. Spec
    /// (openai-responses-api-spec.md §InputItemList L203-205):
    /// `Compaction { encrypted_content, type, id }`. `id` is optional on the
    /// input wire so newly-minted client-side compactions can omit it; it is
    /// always present on items round-tripped from a previous response.
    #[serde(rename = "compaction")]
    Compaction {
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// `{ type: "computer_call", id, call_id, action?, actions?, status,
    /// pending_safety_checks }`.
    ///
    /// Spec (openai-responses-api-spec.md §ComputerCall): single-action
    /// `action` is the legacy shape; `actions` carries the flattened batch
    /// for `computer_use`. Both fields are optional independently, so callers
    /// can roundtrip either form.
    #[serde(rename = "computer_call")]
    ComputerCall {
        id: String,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<ComputerAction>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actions: Option<Vec<ComputerAction>>,
        status: ComputerCallStatus,
        /// Always serialized (including an empty `[]`). The official OpenAI
        /// Python SDK (`openai==2.8.1`,
        /// `types/responses/response_computer_tool_call.py`) declares this as a
        /// non-`Optional` `List[PendingSafetyCheck]`, so the field must always
        /// appear on the wire — an empty array is semantically distinct from
        /// omitting the field.
        #[serde(default)]
        pending_safety_checks: Vec<ComputerSafetyCheck>,
    },
    /// `{ type: "computer_call_output", id?, call_id, output,
    /// acknowledged_safety_checks?, status? }`.
    ///
    /// Spec (openai-responses-api-spec.md §ComputerCallOutput): `output` is the
    /// [`ComputerCallOutputContent::ComputerScreenshot`] payload;
    /// `acknowledged_safety_checks` and `status` are both optional per spec.
    #[serde(rename = "computer_call_output")]
    ComputerCallOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: ComputerCallOutputContent,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        acknowledged_safety_checks: Vec<ComputerSafetyCheck>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<ComputerCallStatus>,
    },
    /// `type: "custom_tool_call"` — assistant's call into a registered
    /// custom tool. Spec: `{ call_id, input, name, type, id?, namespace? }`.
    /// `id` / `namespace` are modelled as `Option<String>` so newly-minted
    /// client-side calls can omit them; they are populated on items
    /// round-tripped from a previous response. `input` is the model's
    /// free-form payload (constrained by the tool's `format` if grammar is
    /// set); the client owns execution and replies with a matching
    /// [`Self::CustomToolCallOutput`].
    #[serde(rename = "custom_tool_call")]
    CustomToolCall {
        call_id: String,
        input: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
    /// `type: "custom_tool_call_output"` — client's response to a
    /// `custom_tool_call`. Spec: `{ call_id, output, type, id? }` (no
    /// `status` field per spec). `id` is
    /// `Option<String>` for the same reason as `CustomToolCall.id` above.
    /// `output` is either a plain string or an array of input-typed content
    /// parts (`input_text` / `input_image` / `input_file`).
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput {
        call_id: String,
        output: CustomToolCallOutputContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// `type: "shell_call"` — assistant's call into the containerized
    /// [`ResponseTool::Shell`] tool.
    ///
    /// Spec (openai-responses-api-spec.md §ShellCall, L228-231) +
    /// OpenAI SDK v2.8.1 `ResponseFunctionShellToolCall`:
    /// `{ action, call_id, type, id, environment, status, created_by? }`.
    /// `id` is `Option<String>` so newly-minted client-side calls can omit
    /// it; the model populates it on items round-tripped from a previous
    /// response. `created_by` carries provenance metadata the SDK types
    /// as `Optional[str]` — present when the item was emitted by the
    /// platform, absent on client-authored calls.
    #[serde(rename = "shell_call")]
    ShellCall {
        action: ShellCallAction,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Resolved execution environment. Spec constrains this to
        /// `local` or `container_reference` on the call form (see
        /// [`ShellCallEnvironment`] docs).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        environment: Option<ShellCallEnvironment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<ShellCallStatus>,
        /// Provenance tag mirroring the SDK's `created_by: Optional[str]`
        /// on `ResponseFunctionShellToolCall`. Dropped at serialize time
        /// when absent so client-authored calls do not carry a null
        /// placeholder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    /// `type: "shell_call_output"` — client's reply to a `shell_call`.
    ///
    /// Spec (openai-responses-api-spec.md §ShellCallOutput, L233-238) +
    /// OpenAI SDK v2.8.1 `ResponseFunctionShellToolCallOutput`:
    /// `{ call_id, output, type, id, max_output_length?, status, created_by? }`.
    /// `id`, `max_output_length`, and `created_by` are modelled as
    /// `Option` per the SDK's `Optional[...]` typing; the server populates
    /// them on items round-tripped from a previous response.
    #[serde(rename = "shell_call_output")]
    ShellCallOutput {
        call_id: String,
        output: Vec<ShellOutputChunk>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_output_length: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<ShellCallStatus>,
        /// Provenance tag mirroring the SDK's `created_by: Optional[str]`
        /// on `ResponseFunctionShellToolCallOutput`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    /// `type: "apply_patch_call"` — model-issued file-edit request. Spec
    /// (openai-responses-api-spec.md §ApplyPatchCall L240-L246):
    /// `{ call_id, operation, status, type, id }`. `id` is `Option<String>`
    /// so newly-minted client-side calls can omit it; it is always present on
    /// items round-tripped from a previous response. The `operation` union is
    /// `CreateFile | DeleteFile | UpdateFile` per
    /// [`ApplyPatchOperation`]; the client owns execution (apply the diff on
    /// disk) and replies with a matching [`Self::ApplyPatchCallOutput`].
    #[serde(rename = "apply_patch_call")]
    ApplyPatchCall {
        call_id: String,
        operation: ApplyPatchOperation,
        status: ApplyPatchCallStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// `type: "apply_patch_call_output"` — client's response to an
    /// `apply_patch_call`. Spec (openai-responses-api-spec.md
    /// §ApplyPatchCallOutput L248-L251): `{ call_id, status, type, id,
    /// output }` where `output` is optional log text. `id` is
    /// `Option<String>` for the same reason as `ApplyPatchCall.id` above;
    /// `output` uses `skip_serializing_if` so a no-log success round-trips
    /// without emitting an explicit `null`.
    #[serde(rename = "apply_patch_call_output")]
    ApplyPatchCallOutput {
        call_id: String,
        status: ApplyPatchCallOutputStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
    },
    /// `type: "local_shell_call"` — assistant's call into the
    /// `local_shell` built-in tool. Spec
    /// (openai-responses-api-spec.md §LocalShellCall L219-222):
    /// `{ id, action, call_id, status, type }` where `action` is a
    /// [`LocalShellExec`] payload describing the command to run on the
    /// host. The client executes the command and replies with a
    /// matching [`Self::LocalShellCallOutput`].
    #[serde(rename = "local_shell_call")]
    LocalShellCall {
        id: String,
        call_id: String,
        action: LocalShellExec,
        status: LocalShellCallStatus,
    },
    /// `type: "local_shell_call_output"` — client's response to a
    /// `local_shell_call`. Spec
    /// (openai-responses-api-spec.md §LocalShellCallOutput L224-226):
    /// `{ id, output, type, status }`. `output` is a single string
    /// carrying the command's serialized JSON output; `status` is
    /// optional per SDK v2.8.1 (`openai==2.8.1`,
    /// `types/responses/response_input_item_param.py`
    /// `LocalShellCallOutput` — `Optional` on `status`).
    #[serde(rename = "local_shell_call_output")]
    LocalShellCallOutput {
        id: String,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<LocalShellCallStatus>,
    },
    /// `type: "mcp_call"` — assistant-emitted hosted-MCP tool call replayed
    /// as an input item for stateless multi-turn (`store=false`) flows.
    ///
    /// Spec (openai-responses-api-spec.md §McpCall L264-266):
    /// `{ id, arguments, name, server_label, type, approval_request_id?, error?, output?, status? }`.
    /// Shape mirrors [`ResponseOutputItem::McpCall`] but `approval_request_id`,
    /// `error`, `output`, and `status` are optional on the input side so
    /// replay of an abridged or in-flight call (no output yet) stays
    /// lossless.
    #[serde(rename = "mcp_call")]
    McpCall {
        id: String,
        arguments: String,
        name: String,
        server_label: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        approval_request_id: Option<String>,
        #[serde(
            default,
            deserialize_with = "mcp_call_error_compat",
            skip_serializing_if = "Option::is_none"
        )]
        error: Option<McpToolCallError>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// `type: "mcp_list_tools"` — hosted-MCP server's tool listing replayed
    /// as an input item.
    ///
    /// Spec (openai-responses-api-spec.md §McpListTools L253-255):
    /// `{ id, server_label, tools, type, error? }` where each `tools` entry
    /// is `{ input_schema, name, annotations?, description? }`. Shape
    /// mirrors [`ResponseOutputItem::McpListTools`] with `error` optional
    /// per SDK v2.8.1
    /// `types/responses/response_input_item.py::McpListTools`.
    #[serde(rename = "mcp_list_tools")]
    McpListTools {
        id: String,
        server_label: String,
        tools: Vec<McpToolInfo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(untagged)]
    SimpleInputMessage {
        content: StringOrContentParts,
        role: String,
        /// Spec: `EasyInputMessage.type` is `optional "message"`. Constrained
        /// to a single-value tag enum so payloads with an unknown `type`
        /// (e.g. `"input_file"`, `"totally_made_up"`) do not silently land
        /// in this untagged catch-all variant — P5 fail-fast contract.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[serde(rename = "type")]
        r#type: Option<SimpleInputMessageTypeTag>,
        /// Optional phase label (spec: EasyInputMessage.phase).
        ///
        /// Preserved through conversation storage so gpt-5.3-codex+ does not
        /// lose the commentary/final_answer distinction across turns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    /// `type: "item_reference"` — pointer to a previously-stored item in the
    /// active conversation. Spec (openai-responses-api-spec.md §InputItemList
    /// L275-276): `ItemReference { id, type }` where `type` is
    /// `optional "item_reference"`. The variant is declared as
    /// `#[serde(untagged)]` because the `type` discriminator is optional on
    /// the wire; `r#type` is pinned to [`ItemReferenceTypeTag`] so payloads
    /// whose `type` is not `"item_reference"` (e.g. `"totally_made_up"`) do
    /// not silently land in this catch-all variant — P5 fail-fast contract.
    ///
    /// Declared AFTER [`Self::SimpleInputMessage`] so a `{id, role, content}`
    /// payload (the id-carrying shape of `SimpleInputMessage`) still lands in
    /// `SimpleInputMessage` first; only `{id}` / `{id, type: "item_reference"}`
    /// payloads — which fail `SimpleInputMessage`'s required-field check —
    /// fall through to this arm.
    ///
    /// Backend resolution (router looks up `id` from conversation history and
    /// substitutes the referenced item inline) is deferred to a future R
    /// task; this variant only adds the schema surface.
    #[serde(untagged)]
    ItemReference {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[serde(rename = "type")]
        r#type: Option<ItemReferenceTypeTag>,
    },
}

/// Single-value tag enum pinning [`ResponseInputOutputItem::ItemReference`]'s
/// optional `type` discriminator to the spec's only permitted value,
/// `"item_reference"`. Used because the outer enum is `type`-tagged and the
/// `ItemReference` variant is declared `#[serde(untagged)]` to accept payloads
/// that omit `type` entirely — without this pin the catch-all would silently
/// swallow payloads whose `type` discriminator is an unknown string.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ItemReferenceTypeTag {
    ItemReference,
}

/// Single-value tag enum pinning `EasyInputMessage.type` to the spec's only
/// permitted value, `"message"`. Used to keep [`ResponseInputOutputItem::SimpleInputMessage`]
/// — which is the `#[serde(untagged)]` fallback in the outer `type`-tagged enum
/// — from silently swallowing payloads whose `type` discriminator is unknown
/// (P5 fail-fast contract).
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SimpleInputMessageTypeTag {
    Message,
}

/// Detail level for [`ResponseContentPart::InputFile`]. Spec restricts this
/// to `"low" | "high"` (defaults to `low`); it is narrower than [`Detail`]
/// used for images which also admits `auto` / `original`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileDetail {
    #[default]
    Low,
    High,
}

/// Typed annotation attached to [`ResponseContentPart::OutputText`]. Matches
/// the OpenAI Responses API `Annotation` union; a `type` discriminator selects
/// the variant on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// `type: "file_citation"` — points at a file previously uploaded.
    FileCitation {
        file_id: String,
        filename: String,
        index: u32,
    },
    /// `type: "url_citation"` — citation back to a URL in a web-search result.
    UrlCitation {
        url: String,
        title: String,
        start_index: u32,
        end_index: u32,
    },
    /// `type: "container_file_citation"` — citation to a file inside a
    /// code-interpreter / computer-use container.
    ContainerFileCitation {
        container_id: String,
        file_id: String,
        filename: String,
        start_index: u32,
        end_index: u32,
    },
    /// `type: "file_path"` — reference to a generated file path.
    FilePath { file_id: String, index: u32 },
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<ChatLogProbs>,
    },
    #[serde(rename = "input_text")]
    InputText { text: String },
    /// `type: "input_image"` — reference to an image supplied by the client.
    /// Exactly one of `file_id` / `image_url` is typically set; both may be
    /// absent when only `detail` is being conveyed.
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<Detail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    /// `type: "input_file"` — reference to an attached file. `file_data` is a
    /// base64 blob; `file_url` / `file_id` reference external/uploaded files.
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<FileDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// `type: "refusal"` — model refusal surfaced as a content part (spec's
    /// `ResponseOutputRefusal`).
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningContent {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

/// Tagged content element carried in `Reasoning.summary`.
///
/// OpenAI spec: `summary: array of SummaryTextContent { text, type: "summary_text" }`.
/// Replaces the prior `Vec<String>` wire-type that broke bidirectional
/// interoperability with spec-compliant clients.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SummaryTextContent {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

/// MCP Tool information for the mcp_list_tools output item
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub annotations: Option<Value>,
}

/// Structured `mcp_call.error`: a `type`-tagged union of protocol,
/// tool-execution, and HTTP failures. OpenAI removed the legacy plain-string
/// shape from the wire; stored/replayed string errors still deserialize via
/// [`mcp_call_error_compat`].
#[expect(
    clippy::enum_variant_names,
    reason = "variant names mirror the spec's `*_error` tag set"
)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum McpToolCallError {
    #[serde(rename = "mcp_protocol_error")]
    ProtocolError { code: i64, message: String },
    #[serde(rename = "mcp_tool_execution_error")]
    ToolExecutionError { content: Value },
    #[serde(rename = "http_error")]
    HttpError { code: i64, message: String },
}

impl McpToolCallError {
    /// Wrap a bare failure message in the tool-execution variant.
    pub fn execution(message: impl Into<String>) -> Self {
        Self::ToolExecutionError {
            content: Value::String(message.into()),
        }
    }
}

/// Accept both the structured union and the legacy plain-string error shape.
fn mcp_call_error_compat<'de, D>(deserializer: D) -> Result<Option<McpToolCallError>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Compat {
        Structured(McpToolCallError),
        Legacy(String),
    }
    Ok(
        Option::<Compat>::deserialize(deserializer)?.map(|error| match error {
            Compat::Structured(error) => error,
            Compat::Legacy(message) => McpToolCallError::execution(message),
        }),
    )
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
        /// Optional phase label (spec: ResponseOutputMessage.phase).
        ///
        /// Labels assistant messages; for gpt-5.3-codex+ we must preserve and
        /// resend this on subsequent turns to avoid perf degradation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    #[serde(rename = "reasoning")]
    #[non_exhaustive]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        /// Encrypted reasoning payload for gpt-5 / o-series round-trip.
        /// Opaque to SMG; preserved verbatim.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        encrypted_content: Option<String>,
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        status: String,
    },
    #[serde(rename = "mcp_list_tools")]
    McpListTools {
        id: String,
        server_label: String,
        tools: Vec<McpToolInfo>,
        /// Spec (openai-responses-api-spec.md §McpListTools L253-255):
        /// `error?: string`. Preserves the failure message when the MCP
        /// server could not list tools; symmetric with the matching field
        /// on `ResponseInputOutputItem::McpListTools` so emit↔replay is
        /// lossless.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(rename = "mcp_call")]
    McpCall {
        id: String,
        status: String,
        approval_request_id: Option<String>,
        arguments: String,
        #[serde(default, deserialize_with = "mcp_call_error_compat")]
        error: Option<McpToolCallError>,
        name: String,
        output: String,
        server_label: String,
    },
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        id: String,
        status: WebSearchCallStatus,
        action: WebSearchAction,
        /// Search hits surfaced when callers request `web_search_call.results`
        /// via the top-level `include[]` array. Mirrors the `file_search_call.results`
        /// shape — array of typed entries when populated, omitted otherwise so the
        /// default wire shape (`{id, action, status, type}`) stays spec-byte-identical.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        results: Option<Vec<WebSearchResult>>,
    },
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        id: String,
        status: CodeInterpreterCallStatus,
        container_id: String,
        code: Option<String>,
        outputs: Option<Vec<CodeInterpreterOutput>>,
    },
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        id: String,
        status: FileSearchCallStatus,
        queries: Vec<String>,
        results: Option<Vec<FileSearchResult>>,
    },
    /// `type: "image_generation_call"` — output item carrying a base64 image
    /// produced by the `image_generation` built-in tool. Spec:
    /// `{ id, action?, background?, output_format?, quality?, result: base64,
    /// revised_prompt?, size?, status, type }`.
    ///
    /// Real OpenAI production responses include the five metadata fields
    /// (`action`, `background`, `output_format`, `quality`, `size`) even
    /// though the OpenAI Rust SDK v2.8.1 omits them. We carry them as
    /// `Option<String>` so cloud passthrough and persistence round-trips
    /// preserve them verbatim — and so downstream consumers can read them
    /// without a second round-trip to the provider.
    ///
    /// The metadata fields are typed as `Option<String>` rather than narrow
    /// enums so unknown or future-added values pass through unchanged;
    /// this mirrors `ImageGenerationTool` on the input-tool side.
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        id: String,
        /// `"generate" | "edit" | "auto"` — which image-generation action
        /// this call dispatched. Preserved free-form so future actions pass
        /// through without a wire break.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<String>,
        /// `"transparent" | "opaque" | "auto"`. Mirrors the
        /// `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<String>,
        /// `"png" | "webp" | "jpeg"`. Mirrors the `image_generation` tool
        /// input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_format: Option<String>,
        /// `"auto" | "low" | "medium" | "high" | "standard" | "hd"`. Mirrors
        /// the `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quality: Option<String>,
        /// Base64-encoded image bytes.
        result: String,
        /// Prompt text the mainline model rewrote before dispatching the
        /// image-generation call. Preserved so downstream turns/storage do
        /// not drop it on replay.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revised_prompt: Option<String>,
        /// `"auto" | "1024x1024" | "1024x1536" | "1536x1024"`. Mirrors the
        /// `image_generation` tool input knob of the same name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size: Option<String>,
        status: ImageGenerationCallStatus,
    },
    /// `type: "compaction"` — server-emitted item carrying an opaque
    /// compacted-history payload. Spec
    /// (openai-responses-api-spec.md §InputItemList L203-205,
    /// §`output: array of ResponseOutputItem`): `{ encrypted_content, type, id }`.
    /// `id` is required on the output wire because the server always assigns
    /// one when emitting the compaction item.
    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        encrypted_content: String,
    },
    /// `{ type: "computer_call", id, call_id, action?, actions?, status,
    /// pending_safety_checks }`.
    ///
    /// Spec (openai-responses-api-spec.md §ComputerCall): output-side mirror of
    /// the input variant — emitted when the model issues a computer-use action.
    /// See [`ComputerAction`].
    #[serde(rename = "computer_call")]
    ComputerCall {
        id: String,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<ComputerAction>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actions: Option<Vec<ComputerAction>>,
        status: ComputerCallStatus,
        /// Always serialized (including an empty `[]`). The official OpenAI
        /// Python SDK (`openai==2.8.1`,
        /// `types/responses/response_computer_tool_call.py`) declares this as a
        /// non-`Optional` `List[PendingSafetyCheck]`, so the field must always
        /// appear on the wire — an empty array is semantically distinct from
        /// omitting the field.
        #[serde(default)]
        pending_safety_checks: Vec<ComputerSafetyCheck>,
    },
    /// `{ type: "computer_call_output", id?, call_id, output,
    /// acknowledged_safety_checks?, status? }`.
    ///
    /// Spec (openai-responses-api-spec.md §ComputerCallOutput).
    #[serde(rename = "computer_call_output")]
    ComputerCallOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: ComputerCallOutputContent,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        acknowledged_safety_checks: Vec<ComputerSafetyCheck>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<ComputerCallStatus>,
    },
    /// `type: "shell_call"` — output-side mirror of the input variant.
    ///
    /// Spec (openai-responses-api-spec.md §ShellCall, L228-231 and §returns
    /// L512-513) + OpenAI SDK v2.8.1 `ResponseFunctionShellToolCall`:
    /// emitted when the model issues a containerized shell action. The
    /// environment echoed back is restricted to `local` /
    /// `container_reference` via [`ResponseShellCallEnvironment`] — per spec
    /// L513 the response form uses `ResponseLocalEnvironment { type: "local" }`
    /// only.
    ///
    /// `id` and `status` are required on the output wire — the SDK types
    /// them as non-`Optional` on `ResponseFunctionShellToolCall`, mirroring
    /// the `ComputerCall` treatment above. `created_by` is the SDK's
    /// `Optional[str]` provenance tag, populated when the platform stamps
    /// the item.
    #[serde(rename = "shell_call")]
    ShellCall {
        id: String,
        call_id: String,
        action: ShellCallAction,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        environment: Option<ResponseShellCallEnvironment>,
        status: ShellCallStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    /// `type: "shell_call_output"` — output-side mirror of the input
    /// variant.
    ///
    /// Spec (openai-responses-api-spec.md §ShellCallOutput, L233-238) +
    /// OpenAI SDK v2.8.1 `ResponseFunctionShellToolCallOutput`:
    /// `{ call_id, output, type, id, max_output_length?, status, created_by? }`.
    /// Emitted when the platform surfaces captured stdout/stderr plus an
    /// [`ShellOutcome`] for a prior shell call.
    ///
    /// `id` and `status` are required per the SDK's non-`Optional` typing.
    /// `max_output_length` is `Optional[int]` in the SDK (the platform may
    /// emit shell outputs when the originating `shell_call.action` did not
    /// specify a cap) and `created_by` is `Optional[str]` — both dropped at
    /// serialize time when absent so downstream consumers do not see null
    /// placeholders.
    #[serde(rename = "shell_call_output")]
    ShellCallOutput {
        id: String,
        call_id: String,
        output: Vec<ShellOutputChunk>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_output_length: Option<u64>,
        status: ShellCallStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        created_by: Option<String>,
    },
    /// `type: "apply_patch_call"` — server-emitted mirror of the input
    /// variant. Spec (openai-responses-api-spec.md §ApplyPatchCall L240-L246):
    /// `{ call_id, operation, status, type, id }`. `id` is required on the
    /// output wire because the server always assigns one when emitting the
    /// apply_patch call item.
    #[serde(rename = "apply_patch_call")]
    ApplyPatchCall {
        id: String,
        call_id: String,
        operation: ApplyPatchOperation,
        status: ApplyPatchCallStatus,
    },
    /// `type: "apply_patch_call_output"` — server-emitted mirror of the
    /// input variant. Spec (openai-responses-api-spec.md
    /// §ApplyPatchCallOutput L248-L251): `{ call_id, status, type, id,
    /// output }`. `id` is required on the output wire; `output` is optional
    /// log text surfaced by the upstream apply_patch executor.
    #[serde(rename = "apply_patch_call_output")]
    ApplyPatchCallOutput {
        id: String,
        call_id: String,
        status: ApplyPatchCallOutputStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
    },
    /// `type: "local_shell_call"` — output-side mirror of the input
    /// variant — emitted when the model issues a `local_shell` tool call.
    /// Spec (openai-responses-api-spec.md §LocalShellCall L219-222):
    /// `{ id, action, call_id, status, type }` with `action` as a
    /// [`LocalShellExec`] payload. See [`ResponseInputOutputItem::LocalShellCall`].
    #[serde(rename = "local_shell_call")]
    LocalShellCall {
        id: String,
        call_id: String,
        action: LocalShellExec,
        status: LocalShellCallStatus,
    },
    /// `type: "local_shell_call_output"` — output-side mirror of the
    /// input variant. Spec
    /// (openai-responses-api-spec.md §LocalShellCallOutput L224-226):
    /// `{ id, output, type, status }` with `status` optional per SDK
    /// v2.8.1. See [`ResponseInputOutputItem::LocalShellCallOutput`].
    #[serde(rename = "local_shell_call_output")]
    LocalShellCallOutput {
        id: String,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<LocalShellCallStatus>,
    },
}

// ============================================================================
// Built-in Tool Call Types
// ============================================================================

/// Status for web search tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchCallStatus {
    InProgress,
    Searching,
    Completed,
    Failed,
}

/// Action performed during a web search.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchAction {
    Search {
        #[serde(skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        queries: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sources: Vec<WebSearchSource>,
    },
    OpenPage {
        url: String,
    },
    Find {
        url: String,
        pattern: String,
    },
}

/// A source returned from web search.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct WebSearchSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub url: String,
}

/// A single search result attached to a `WebSearchCall` when the caller
/// requested `web_search_call.results` via the top-level `include[]` array.
///
/// Optional fields mirror the `FileSearchResult` shape — only `url` is
/// guaranteed; titles, snippets, and scores ride along when the upstream
/// search backend supplies them.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct WebSearchResult {
    /// Canonical URL of the result.
    pub url: String,
    /// Page or document title, when surfaced by the search backend.
    pub title: Option<String>,
    /// Short text snippet excerpted from the result.
    pub snippet: Option<String>,
    /// Relevance score in `[0, 1]`, when the backend supplies one.
    pub score: Option<f32>,
}

/// Status for code interpreter tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CodeInterpreterCallStatus {
    InProgress,
    Completed,
    Incomplete,
    Interpreting,
    Failed,
}

/// Output from code interpreter execution.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterOutput {
    Logs { logs: String },
    Image { url: String },
}

/// Status for file search tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileSearchCallStatus {
    InProgress,
    Searching,
    Completed,
    Incomplete,
    Failed,
}

/// A result from file search.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FileSearchResult {
    pub file_id: String,
    pub filename: String,
    pub text: Option<String>,
    pub score: Option<f32>,
    pub attributes: Option<Value>,
}

/// Status for `local_shell` tool calls.
///
/// Spec (openai-responses-api-spec.md §LocalShellCall L221): `"in_progress"
/// | "completed" | "incomplete"`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LocalShellCallStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// `action` payload carried by a [`ResponseInputOutputItem::LocalShellCall`] /
/// [`ResponseOutputItem::LocalShellCall`] item.
///
/// Spec (openai-responses-api-spec.md §LocalShellCall L220):
/// `{ command: array of string, env: map[string], type: "exec",
///   timeout_ms?, user?, working_directory? }`. `env` is always present
/// (an empty object is semantically distinct from omitting the field),
/// matching the OpenAI Python SDK (`openai==2.8.1`,
/// `types/responses/response_input_item_param.py` `LocalShellCallAction`
/// — non-`Optional` `Dict[str, str]`).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LocalShellExec {
    /// `type: "exec"` — the only action kind defined by the spec today.
    #[serde(rename = "exec")]
    Exec {
        /// Argv of the command to run on the host.
        command: Vec<String>,
        /// Environment variables overlaid on the host process env.
        /// Always serialized (possibly empty) to match SDK shape.
        env: std::collections::BTreeMap<String, String>,
        /// Hard timeout in milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        /// User to run the command as.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        /// Working directory for the command.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        working_directory: Option<String>,
    },
}

// ============================================================================
// Configuration Enums
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "ResponsesServiceTier")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    Auto,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

/// Why a response stopped before producing complete output.
///
/// Mirrors OpenAI's `incomplete_details.reason`: reserved strictly for the two
/// truncation semantics. Any other stop condition (wall-clock timeout,
/// `max_tool_calls` exhaustion, provider errors) is surfaced as a `failed`
/// status with an `error` payload instead.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum IncompleteReason {
    /// Output was truncated because it hit `max_output_tokens`.
    MaxOutputTokens,
    /// Output was truncated by the content filter.
    ContentFilter,
}

/// Structured detail attached to a response whose status is `incomplete`.
///
/// Wire shape: `{ "reason": "max_output_tokens" | "content_filter" }`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct IncompleteDetails {
    /// The reason the response is incomplete.
    pub reason: IncompleteReason,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ReasoningInfo {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

// ============================================================================
// Text Format (structured outputs)
// ============================================================================

/// Text configuration for structured output requests
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

/// Text format: text (default), json_object (legacy), or json_schema (recommended)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum TextFormat {
    #[serde(rename = "text")]
    Text,

    #[serde(rename = "json_object")]
    JsonObject,

    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: Value,
        description: Option<String>,
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncludeField {
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    #[serde(rename = "web_search_call.results")]
    WebSearchCallResults,
}

// ============================================================================
// Usage Types (Responses API format)
// ============================================================================

/// OpenAI Responses API usage format (different from standard UsageInfo)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponsesUsage {
    Classic(UsageInfo),
    Modern(ResponseUsage),
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

impl From<&PromptTokenUsageInfo> for InputTokensDetails {
    fn from(d: &PromptTokenUsageInfo) -> Self {
        Self {
            cached_tokens: d.cached_tokens,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl UsageInfo {
    /// Convert to OpenAI Responses API format
    pub fn to_response_usage(&self) -> ResponseUsage {
        ResponseUsage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            input_tokens_details: self
                .prompt_tokens_details
                .as_ref()
                .map(InputTokensDetails::from),
            output_tokens_details: self.reasoning_tokens.map(|tokens| OutputTokensDetails {
                reasoning_tokens: tokens,
            }),
        }
    }
}

impl From<UsageInfo> for ResponseUsage {
    fn from(usage: UsageInfo) -> Self {
        usage.to_response_usage()
    }
}

impl ResponseUsage {
    /// Convert back to standard UsageInfo format
    pub fn to_usage_info(&self) -> UsageInfo {
        UsageInfo {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.total_tokens,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            prompt_tokens_details: self.input_tokens_details.as_ref().map(|details| {
                PromptTokenUsageInfo {
                    cached_tokens: details.cached_tokens,
                }
            }),
        }
    }
}

impl ResponsesUsage {
    pub fn to_response_usage(&self) -> ResponseUsage {
        match self {
            ResponsesUsage::Classic(usage) => usage.to_response_usage(),
            ResponsesUsage::Modern(usage) => usage.clone(),
        }
    }

    pub fn to_usage_info(&self) -> UsageInfo {
        match self {
            ResponsesUsage::Classic(usage) => usage.clone(),
            ResponsesUsage::Modern(usage) => usage.to_usage_info(),
        }
    }
}

// ============================================================================
// Helper Functions for Defaults
// ============================================================================

fn default_top_k() -> i32 {
    -1
}

fn default_repetition_penalty() -> f32 {
    1.0
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<T>"
)]
fn default_temperature() -> Option<f32> {
    Some(1.0)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_responses_cross_parameters"))]
pub struct ResponsesRequest {
    /// Fields to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeField>>,

    /// Input content - can be string or structured items
    #[validate(custom(function = "validate_response_input"))]
    pub input: ResponseInput,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_tool_calls: Option<u32>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,

    /// Model to use
    pub model: String,

    /// Optional conversation reference to persist input/output as items.
    ///
    /// Spec: `conversation` accepts either a bare ID string or
    /// `ResponseConversationParam { id }`. Both wire shapes deserialize into
    /// [`ConversationRef`]; downstream code reads the id via
    /// [`ConversationRef::as_id`].
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_conversation_id"))]
    pub conversation: Option<ConversationRef>,

    /// Whether to enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// ID of previous response to continue from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningParam>,

    /// Service tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Whether to store the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,

    /// Temperature for sampling
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "Option::is_none"
    )]
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// Tool choice behavior (Responses-spec enum — see `ResponsesToolChoice`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,

    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_response_tools"))]
    pub tools: Option<Vec<ResponseTool>>,

    /// Number of top logprobs to return
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Text format for structured outputs (text, json_object, json_schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_text_format"))]
    pub text: Option<TextConfig>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Reference to a prompt template and its variables.
    /// Spec: body param `prompt` (ResponsePrompt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,

    /// Stable cache key used by upstream to share prompt-prefix caches across
    /// requests. Spec: body param `prompt_cache_key` (replaces `user`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Retention policy for prompt-cache entries.
    /// Spec: body param `prompt_cache_retention` (`"in-memory"` | `"24h"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,

    /// Stable user identifier for policy/abuse detection (max 64 chars on the
    /// spec, but we do not enforce length here — routers may pass through).
    /// Spec: body param `safety_identifier` (replaces `user` on request side).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Streaming-only options. Spec: body param `stream_options`.
    /// On the Responses API the only documented field is `include_obfuscation`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Per-request context-management configuration.
    /// Spec: body param `context_management` — array of entries describing how
    /// the upstream should compact context for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagementEntry>>,

    /// Top-k sampling parameter (SGLang extension)
    #[serde(default = "default_top_k")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: i32,

    /// Min-p sampling parameter (SGLang extension)
    #[serde(default)]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: f32,

    /// Repetition penalty (SGLang extension)
    #[serde(default = "default_repetition_penalty")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponseInput {
    Items(Vec<ResponseInputOutputItem>),
    Text(String),
}

impl Default for ResponsesRequest {
    fn default() -> Self {
        Self {
            include: None,
            input: ResponseInput::Text(String::new()),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: String::new(),
            conversation: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            store: None,
            stream: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            text: None,
            user: None,
            request_id: None,
            priority: 0,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            safety_identifier: None,
            stream_options: None,
            context_management: None,
            top_k: default_top_k(),
            min_p: 0.0,
            repetition_penalty: default_repetition_penalty(),
        }
    }
}

impl Normalizable for ResponsesRequest {
    /// Normalize the request by applying defaults:
    /// 1. Apply tool_choice defaults based on tools presence
    /// 2. Apply parallel_tool_calls defaults
    /// 3. Apply store field defaults
    fn normalize(&mut self) {
        // 1. Apply tool_choice defaults
        if self.tool_choice.is_none() {
            if let Some(tools) = &self.tools {
                let choice_value = if tools.is_empty() {
                    ToolChoiceOptions::None
                } else {
                    ToolChoiceOptions::Auto
                };
                self.tool_choice = Some(ResponsesToolChoice::Options(choice_value));
            }
            // If tools is None, leave tool_choice as None (don't set it)
        }

        // 2. Apply default for parallel_tool_calls if tools are present
        if self.parallel_tool_calls.is_none() && self.tools.is_some() {
            self.parallel_tool_calls = Some(true);
        }

        // 3. Ensure store defaults to true if not specified
        if self.store.is_none() {
            self.store = Some(true);
        }
    }
}

impl GenerationRequest for ResponsesRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(items) => {
                let mut result = String::with_capacity(256);
                let mut has_parts = false;

                let mut append_text = |text: &str| {
                    if has_parts {
                        result.push(' ');
                    }
                    has_parts = true;
                    result.push_str(text);
                };

                for item in items {
                    match item {
                        ResponseInputOutputItem::Message { content, .. } => {
                            for part in content {
                                let text = match part {
                                    ResponseContentPart::OutputText { text, .. } => {
                                        Some(text.as_str())
                                    }
                                    ResponseContentPart::InputText { text } => Some(text.as_str()),
                                    // Non-text parts (images, files, refusals) contribute no
                                    // prompt text; skip without appending.
                                    ResponseContentPart::InputImage { .. }
                                    | ResponseContentPart::InputFile { .. }
                                    | ResponseContentPart::Refusal { .. } => None,
                                };
                                if let Some(t) = text {
                                    append_text(t);
                                }
                            }
                        }
                        ResponseInputOutputItem::SimpleInputMessage { content, .. } => {
                            match content {
                                StringOrContentParts::String(s) => {
                                    append_text(s.as_str());
                                }
                                StringOrContentParts::Array(parts) => {
                                    for part in parts {
                                        let text = match part {
                                            ResponseContentPart::OutputText { text, .. } => {
                                                Some(text.as_str())
                                            }
                                            ResponseContentPart::InputText { text } => {
                                                Some(text.as_str())
                                            }
                                            ResponseContentPart::InputImage { .. }
                                            | ResponseContentPart::InputFile { .. }
                                            | ResponseContentPart::Refusal { .. } => None,
                                        };
                                        if let Some(t) = text {
                                            append_text(t);
                                        }
                                    }
                                }
                            }
                        }
                        ResponseInputOutputItem::Reasoning { content, .. } => {
                            for part in content {
                                match part {
                                    ResponseReasoningContent::ReasoningText { text } => {
                                        append_text(text.as_str());
                                    }
                                }
                            }
                        }
                        ResponseInputOutputItem::FunctionToolCall { .. }
                        | ResponseInputOutputItem::FunctionCallOutput { .. }
                        | ResponseInputOutputItem::McpApprovalRequest { .. }
                        | ResponseInputOutputItem::McpApprovalResponse { .. }
                        | ResponseInputOutputItem::ImageGenerationCall { .. }
                        | ResponseInputOutputItem::Compaction { .. }
                        | ResponseInputOutputItem::ComputerCall { .. }
                        | ResponseInputOutputItem::ComputerCallOutput { .. }
                        | ResponseInputOutputItem::CustomToolCall { .. }
                        | ResponseInputOutputItem::CustomToolCallOutput { .. }
                        | ResponseInputOutputItem::ShellCall { .. }
                        | ResponseInputOutputItem::ShellCallOutput { .. }
                        | ResponseInputOutputItem::ItemReference { .. }
                        | ResponseInputOutputItem::ApplyPatchCall { .. }
                        | ResponseInputOutputItem::ApplyPatchCallOutput { .. }
                        | ResponseInputOutputItem::LocalShellCall { .. }
                        | ResponseInputOutputItem::LocalShellCallOutput { .. }
                        | ResponseInputOutputItem::McpCall { .. }
                        | ResponseInputOutputItem::McpListTools { .. } => {}
                    }
                }

                result
            }
        }
    }
}

/// Validate the conversation reference's ID format.
///
/// The validator crate auto-unwraps `Option<ConversationRef>` for the
/// `#[validate(custom(...))]` attribute, so this function only runs when
/// the field is present. Both wire shapes (bare string or `{ id }` object)
/// are validated against the same rule by extracting the underlying id via
/// [`ConversationRef::as_id`].
pub fn validate_conversation_id(conv: &ConversationRef) -> Result<(), ValidationError> {
    let conv_id = conv.as_id();
    if !conv_id.starts_with("conv_") {
        let mut error = ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{conv_id}'. Expected an ID that begins with 'conv_'."
        )));
        return Err(error);
    }

    // Check if the conversation ID contains only valid characters
    let is_valid = conv_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-');

    if !is_valid {
        let mut error = ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{conv_id}'. Expected an ID that contains letters, numbers, underscores, or dashes, but this value contained additional characters."
        )));
        return Err(error);
    }
    Ok(())
}

/// Validates tool_choice requires tools and references exist
fn validate_tool_choice_with_tools(request: &ResponsesRequest) -> Result<(), ValidationError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
    let is_some_choice = !matches!(
        tool_choice,
        ResponsesToolChoice::Options(ToolChoiceOptions::None)
    );

    // Check if tool_choice requires tools but none are provided
    if is_some_choice && !has_tools {
        let mut e = ValidationError::new("tool_choice_requires_tools");
        e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
        return Err(e);
    }

    // Validate tool references exist when tools are present
    if !has_tools {
        return Ok(());
    }

    // Extract function tool names from ResponseTools
    // INVARIANT: has_tools is true here, so tools is Some and non-empty
    let Some(tools) = request.tools.as_ref() else {
        return Ok(());
    };
    let function_tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| match t {
            ResponseTool::Function(ft) => Some(ft.function.name.as_str()),
            _ => None,
        })
        .collect();

    // Validate tool references exist
    match tool_choice {
        ResponsesToolChoice::Function(_) => {
            // Accessor goes through `function_name()` so we stay agnostic to
            // the underlying wire shape (flat vs. legacy nested) — both are
            // normalized at deserialize time.
            if let Some(name) = tool_choice.function_name() {
                if !function_tool_names.contains(&name) {
                    let mut e = ValidationError::new("tool_choice_function_not_found");
                    e.message = Some(
                        format!(
                            "Invalid value for 'tool_choice': function '{name}' not found in 'tools'.",
                        )
                        .into(),
                    );
                    return Err(e);
                }
            }
        }
        ResponsesToolChoice::AllowedTools {
            mode,
            tools: allowed_tools,
            ..
        } => {
            // Validate mode is "auto" or "required"
            if mode != "auto" && mode != "required" {
                let mut e = ValidationError::new("tool_choice_invalid_mode");
                e.message = Some(
                    format!(
                        "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{mode}'."
                    )
                    .into(),
                );
                return Err(e);
            }

            // Validate that all function tool references exist
            for tool_ref in allowed_tools {
                if let ToolReference::Function { name } = tool_ref {
                    if !function_tool_names.contains(&name.as_str()) {
                        let mut e = ValidationError::new("tool_choice_tool_not_found");
                        e.message = Some(
                            format!(
                                "Invalid value for 'tool_choice.tools': tool '{name}' not found in 'tools'."
                            )
                            .into(),
                        );
                        return Err(e);
                    }
                }
                // Note: MCP and hosted tools don't need existence validation here
                // as they are resolved dynamically at runtime
            }
        }
        // Remaining variants have no cross-field existence constraints —
        // hosted built-ins, MCP server selection, custom tool names, and
        // `apply_patch` / `shell` are resolved at routing time.
        ResponsesToolChoice::Options(_)
        | ResponsesToolChoice::Types { .. }
        | ResponsesToolChoice::Mcp { .. }
        | ResponsesToolChoice::Custom { .. }
        | ResponsesToolChoice::ApplyPatch { .. }
        | ResponsesToolChoice::Shell { .. } => {}
    }

    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_responses_cross_parameters(request: &ResponsesRequest) -> Result<(), ValidationError> {
    // 1. Validate tool_choice requires tools (enhanced)
    validate_tool_choice_with_tools(request)?;

    // 2. Validate top_logprobs requires include field
    if request.top_logprobs.is_some() {
        let has_logprobs_include = request
            .include
            .as_ref()
            .is_some_and(|inc| inc.contains(&IncludeField::MessageOutputTextLogprobs));

        if !has_logprobs_include {
            let mut e = ValidationError::new("top_logprobs_requires_include");
            e.message = Some(
                "top_logprobs requires include field with 'message.output_text.logprobs'".into(),
            );
            return Err(e);
        }
    }

    // 3. Validate conversation and previous_response_id are mutually exclusive
    if request.conversation.is_some() && request.previous_response_id.is_some() {
        let mut e = ValidationError::new("mutually_exclusive_parameters");
        e.message = Some("Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.".into());
        return Err(e);
    }

    // 4. Validate input items structure
    if let ResponseInput::Items(items) = &request.input {
        // Check for at least one valid input message
        let has_valid_input = items.iter().any(|item| {
            matches!(
                item,
                ResponseInputOutputItem::Message { .. }
                    | ResponseInputOutputItem::SimpleInputMessage { .. }
            )
        });

        if !has_valid_input {
            let mut e = ValidationError::new("input_missing_user_message");
            e.message = Some("Input items must contain at least one message".into());
            return Err(e);
        }
    }

    // 5. Validate text format conflicts (for future structured output constraints)
    // Currently, Responses API doesn't have regex/ebnf like Chat API,
    // but this is here for completeness and future-proofing

    Ok(())
}

// ============================================================================
// Field-Level Validation Functions
// ============================================================================

/// Validates response input is not empty and has valid content
fn validate_response_input(input: &ResponseInput) -> Result<(), ValidationError> {
    match input {
        ResponseInput::Text(text) => {
            if text.is_empty() {
                let mut e = ValidationError::new("input_text_empty");
                e.message = Some("Input text cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInput::Items(items) => {
            if items.is_empty() {
                let mut e = ValidationError::new("input_items_empty");
                e.message = Some("Input items cannot be empty".into());
                return Err(e);
            }
            // Validate each item has valid content
            for item in items {
                validate_input_item(item)?;
            }
        }
    }
    Ok(())
}

/// Validates individual input items have valid content
fn validate_input_item(item: &ResponseInputOutputItem) -> Result<(), ValidationError> {
    match item {
        ResponseInputOutputItem::Message { content, .. } => {
            if content.is_empty() {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInputOutputItem::SimpleInputMessage { content, .. } => match content {
            StringOrContentParts::String(s) if s.is_empty() => {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
            StringOrContentParts::Array(parts) if parts.is_empty() => {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content parts cannot be empty".into());
                return Err(e);
            }
            _ => {}
        },
        ResponseInputOutputItem::Reasoning { .. } => {
            // Reasoning content can be empty - no validation needed
        }
        ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
            if output.is_empty() {
                let mut e = ValidationError::new("function_output_empty");
                e.message = Some("Function call output cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInputOutputItem::FunctionToolCall { .. } => {}
        ResponseInputOutputItem::McpApprovalRequest { .. } => {}
        ResponseInputOutputItem::McpApprovalResponse { .. } => {}
        ResponseInputOutputItem::ImageGenerationCall { .. } => {}
        ResponseInputOutputItem::Compaction { .. } => {}
        ResponseInputOutputItem::ComputerCall { .. } => {}
        ResponseInputOutputItem::ComputerCallOutput { .. } => {}
        // CustomToolCall is model-generated and echoed back on multi-turn
        // replay; matches the FunctionToolCall arm above with no content
        // validation so a parameterless custom tool with empty input can
        // round-trip cleanly.
        ResponseInputOutputItem::CustomToolCall { .. } => {}
        ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
            CustomToolCallOutputContent::Text(s) if s.is_empty() => {
                let mut e = ValidationError::new("custom_tool_call_output_empty");
                e.message = Some("Custom tool call output cannot be empty".into());
                return Err(e);
            }
            CustomToolCallOutputContent::Parts(parts) if parts.is_empty() => {
                let mut e = ValidationError::new("custom_tool_call_output_empty");
                e.message = Some("Custom tool call output parts cannot be empty".into());
                return Err(e);
            }
            _ => {}
        },
        // ShellCall is model-generated and echoed back on multi-turn replay;
        // mirrors FunctionToolCall above with no content validation so a
        // parameterless shell call can round-trip cleanly.
        ResponseInputOutputItem::ShellCall { .. } => {}
        ResponseInputOutputItem::ShellCallOutput { .. } => {
            // The router returns 501 for shell calls, so SMG never synthesises
            // a ShellCallOutput itself. Skip content validation here so
            // round-tripping a previously-recorded response (even with an
            // empty chunk list) stays lossless — the cross-turn replay
            // contract is the motivating use case for keeping this arm
            // content-agnostic.
        }
        // A bare reference to a prior item; no content to validate.
        ResponseInputOutputItem::ItemReference { .. } => {}
        // ApplyPatchCall is model-generated and echoed back on multi-turn
        // replay; matches the FunctionToolCall / CustomToolCall arms with no
        // diff/path validation — the operation payload is structurally
        // enforced by the `ApplyPatchOperation` enum, and accepting empty
        // diffs for `create_file` / `update_file` preserves round-trip
        // fidelity with items emitted by upstream providers.
        ResponseInputOutputItem::ApplyPatchCall { .. } => {}
        // ApplyPatchCallOutput.output is optional log text per spec
        // (openai-responses-api-spec.md §ApplyPatchCallOutput L251); an
        // absent or empty log is spec-legal (a clean `completed` with no
        // output, or a `failed` where the executor had nothing to log) so
        // no emptiness check applies here.
        ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {}
        // Validation mirrors `ComputerCall` / `ImageGenerationCall` above
        // (no payload-level content checks).
        ResponseInputOutputItem::LocalShellCall { .. } => {}
        ResponseInputOutputItem::LocalShellCallOutput { .. } => {}
        // MCP call/list-tools input items replayed for stateless multi-turn.
        // Matches `McpApprovalRequest` above with no content validation so an
        // abridged or in-flight call (output / error absent) can round-trip
        // cleanly.
        ResponseInputOutputItem::McpCall { .. } => {}
        ResponseInputOutputItem::McpListTools { .. } => {}
    }
    Ok(())
}

/// Validates ResponseTool structure based on tool type
fn validate_response_tools(tools: &[ResponseTool]) -> Result<(), ValidationError> {
    // MCP server_label must be present and unique (case-insensitive).
    let mut seen_mcp_labels: HashSet<String> = HashSet::new();

    for (idx, tool) in tools.iter().enumerate() {
        if let ResponseTool::Mcp(mcp) = tool {
            let raw_label = mcp.server_label.as_str();
            if raw_label.is_empty() {
                let mut e = ValidationError::new("missing_required_parameter");
                e.message = Some(
                    format!("Missing required parameter: 'tools[{idx}].server_label'.").into(),
                );
                return Err(e);
            }

            // OpenAI spec-compatible validation: require a non-empty label that starts with a
            // letter and contains only letters, digits, '-' and '_'.
            let valid = raw_label.starts_with(|c: char| c.is_ascii_alphabetic())
                && raw_label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !valid {
                let mut e = ValidationError::new("invalid_server_label");
                e.message = Some(
                    format!(
                        "Invalid input {raw_label}: 'server_label' must start with a letter and consist of only letters, digits, '-' and '_'"
                    )
                    .into(),
                );
                return Err(e);
            }

            let normalized = raw_label.to_lowercase();
            if !seen_mcp_labels.insert(normalized) {
                let mut e = ValidationError::new("mcp_tool_duplicate_server_label");
                e.message = Some(
                    format!("Duplicate MCP server_label '{raw_label}' found in 'tools' parameter.")
                        .into(),
                );
                return Err(e);
            }

            // One of `server_url` or `connector_id` is required, and the two
            // are mutually exclusive. Reject payloads that set both so
            // downstream target resolution is unambiguous.
            if mcp.server_url.is_some() && mcp.connector_id.is_some() {
                let mut e = ValidationError::new("mcp_tool_conflicting_targets");
                e.message = Some(
                    format!(
                        "MCP tool with server_label '{raw_label}' sets both 'server_url' and 'connector_id'; exactly one is required."
                    )
                    .into(),
                );
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Validates text format configuration (JSON schema name cannot be empty)
fn validate_text_format(text: &TextConfig) -> Result<(), ValidationError> {
    if let Some(TextFormat::JsonSchema { name, .. }) = &text.format {
        if name.is_empty() {
            let mut e = ValidationError::new("json_schema_name_empty");
            e.message = Some("JSON schema name cannot be empty".into());
            return Err(e);
        }
    }
    Ok(())
}

/// Normalize a SimpleInputMessage to a proper Message item
///
/// This helper converts SimpleInputMessage (which can have flexible content)
/// into a fully-structured Message item with a generated ID, role, and content array.
///
/// SimpleInputMessage items are converted to Message items with IDs generated using
/// the centralized ID generation pattern with "msg_" prefix for consistency.
///
/// # Arguments
/// * `item` - The input item to normalize
///
/// # Returns
/// A normalized ResponseInputOutputItem (either Message if converted, or original if not SimpleInputMessage)
pub fn normalize_input_item(item: &ResponseInputOutputItem) -> ResponseInputOutputItem {
    match item {
        ResponseInputOutputItem::SimpleInputMessage {
            content,
            role,
            phase,
            ..
        } => {
            let content_vec = match content {
                StringOrContentParts::String(s) => {
                    vec![ResponseContentPart::InputText { text: s.clone() }]
                }
                StringOrContentParts::Array(parts) => parts.clone(),
            };

            ResponseInputOutputItem::Message {
                id: generate_id("msg"),
                role: role.clone(),
                content: content_vec,
                status: Some("completed".to_string()),
                phase: *phase,
            }
        }
        _ => item.clone(),
    }
}

pub fn generate_id(prefix: &str) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    // Generate exactly 50 hex characters (25 bytes) for the part after the underscore
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let hex_string: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{prefix}_{hex_string}")
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[non_exhaustive]
pub struct ResponsesResponse {
    /// Response ID
    pub id: String,

    /// Object type
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Creation timestamp (unix seconds)
    pub created_at: i64,

    /// Completion timestamp (unix seconds). `None` until the response reaches
    /// a terminal state (`completed`, `incomplete`, `failed`, `cancelled`).
    #[serde(default)]
    pub completed_at: Option<i64>,

    /// Whether the response was created in background mode.
    #[serde(default)]
    pub background: Option<bool>,

    /// Conversation this response is linked to, if any.
    #[serde(default)]
    pub conversation: Option<String>,

    /// Response status
    pub status: ResponseStatus,

    /// Error information if status is failed
    pub error: Option<Value>,

    /// Incomplete details if the response was truncated (`incomplete` status).
    pub incomplete_details: Option<IncompleteDetails>,

    /// System instructions used
    pub instructions: Option<String>,

    /// Max output tokens setting
    pub max_output_tokens: Option<u32>,

    /// Model name
    pub model: String,

    /// Output items
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,

    /// Whether parallel tool calls are enabled
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Previous response ID if this is a continuation
    pub previous_response_id: Option<String>,

    /// Reasoning information
    pub reasoning: Option<ReasoningInfo>,

    /// Whether the response is stored
    #[serde(default = "default_true")]
    pub store: bool,

    /// Temperature setting used
    pub temperature: Option<f32>,

    /// Text format settings
    pub text: Option<TextConfig>,

    /// Tool choice setting
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,

    /// Top-p setting used
    pub top_p: Option<f32>,

    /// Truncation strategy used
    pub truncation: Option<String>,

    /// Usage information
    pub usage: Option<ResponsesUsage>,

    /// User identifier
    pub user: Option<String>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

fn default_object_type() -> String {
    "response".to_string()
}

fn default_tool_choice() -> String {
    "auto".to_string()
}

impl ResponsesResponse {
    /// Create a builder for constructing a ResponsesResponse
    pub fn builder(id: impl Into<String>, model: impl Into<String>) -> ResponsesResponseBuilder {
        ResponsesResponseBuilder::new(id, model)
    }

    /// Check if the response is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, ResponseStatus::Completed)
    }

    /// Check if the response is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, ResponseStatus::InProgress)
    }

    /// Check if the response failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ResponseStatus::Failed)
    }

    /// Check if the response terminated as incomplete (max_output_tokens / content_filter)
    pub fn is_incomplete(&self) -> bool {
        matches!(self.status, ResponseStatus::Incomplete)
    }
}

impl ResponseInputOutputItem {
    /// Create a new reasoning input/output item.
    ///
    /// `encrypted_content` defaults to `None`; use
    /// [`Self::new_reasoning_encrypted`] when round-tripping gpt-5 /
    /// o-series encrypted reasoning.
    pub fn new_reasoning(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: None,
            status,
        }
    }

    /// Create a new reasoning input/output item carrying an encrypted
    /// reasoning payload. The `encrypted_content` must be the opaque
    /// ciphertext.
    pub fn new_reasoning_encrypted(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        encrypted_content: String,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: Some(encrypted_content),
            status,
        }
    }
}

impl ResponseOutputItem {
    /// Create a new message output item (no phase).
    pub fn new_message(
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    ) -> Self {
        Self::Message {
            id,
            role,
            content,
            status,
            phase: None,
        }
    }

    /// Create a new reasoning output item.
    ///
    /// `encrypted_content` defaults to `None`; use
    /// [`Self::new_reasoning_encrypted`] when carrying gpt-5 / o-series
    /// encrypted reasoning.
    pub fn new_reasoning(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: None,
            status,
        }
    }

    /// Create a new reasoning output item carrying an encrypted reasoning payload.
    ///
    /// The `encrypted_content` must be the opaque ciphertext; a `None` value
    /// would defeat the purpose of the `_encrypted` constructor — callers
    /// without ciphertext should use [`Self::new_reasoning`] instead.
    pub fn new_reasoning_encrypted(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        encrypted_content: String,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: Some(encrypted_content),
            status,
        }
    }

    /// Create a new function tool call output item
    pub fn new_function_tool_call(
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    ) -> Self {
        Self::FunctionToolCall {
            id: Some(id),
            call_id,
            name,
            arguments,
            output,
            status,
        }
    }
}

impl ResponseContentPart {
    /// Create a new `output_text` content part.
    pub fn new_text(
        text: String,
        annotations: Vec<Annotation>,
        logprobs: Option<ChatLogProbs>,
    ) -> Self {
        Self::OutputText {
            text,
            annotations,
            logprobs,
        }
    }
}

impl ResponseReasoningContent {
    /// Create a new reasoning text content
    pub fn new_reasoning_text(text: String) -> Self {
        Self::ReasoningText { text }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock `as_str()` to the canonical serde tag for the unit variants
    /// — drift between the two would produce inconsistent wire labels
    /// across dispatch paths and serializers.
    #[test]
    fn response_tool_as_str_matches_serde_tag_for_unit_variants() {
        for tool in [
            ResponseTool::Computer,
            ResponseTool::ApplyPatch,
            ResponseTool::LocalShell,
        ] {
            let serialized = serde_json::to_value(&tool).unwrap();
            let serde_tag = serialized.get("type").and_then(|v| v.as_str()).unwrap();
            assert_eq!(tool.as_str(), serde_tag);
        }
    }
}
