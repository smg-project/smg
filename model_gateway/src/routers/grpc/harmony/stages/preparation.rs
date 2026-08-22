//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    responses::ResponsesRequest,
};
use serde_json::json;
use tracing::error;

use super::super::HarmonyBuilder;
use crate::routers::{
    error,
    grpc::{
        common::{responses::utils::extract_tools_from_response_tools, stages::PipelineStage},
        context::{PreparationOutput, RequestContext, RequestType},
        utils,
    },
};

/// Harmony Preparation stage: Encode requests using Harmony protocol
///
/// Replaces the regular PreparationStage for Harmony models.
/// Converts chat/generate requests to Harmony-encoded token_ids and extraction_text.
pub(crate) struct HarmonyPreparationStage {
    builder: HarmonyBuilder,
}

impl HarmonyPreparationStage {
    /// Create a new Harmony preparation stage
    pub fn new() -> Self {
        Self {
            builder: HarmonyBuilder::new(),
        }
    }
}

impl Default for HarmonyPreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Clone Arc before match to avoid borrow checker issues
        // Arc clone is cheap (8 bytes) - avoids full request clone (15KB-200KB)
        let is_chat = matches!(&ctx.input.request_type, RequestType::Chat(_));
        let is_responses = matches!(&ctx.input.request_type, RequestType::Responses(_));

        if is_chat {
            let request_arc = ctx.chat_request_arc();
            // Reject ignore_eos for Harmony models: Harmony requires EOS-based stop tokens
            // to produce well-formed output. When ignore_eos is true, some backends skip all
            // stop token checks, causing the Harmony parser to receive malformed token sequences.
            if request_arc.ignore_eos {
                return Err(error::bad_request(
                    "ignore_eos_not_supported",
                    "ignore_eos is not supported for Harmony models",
                ));
            }
            self.prepare_chat(ctx, &request_arc)?;
        } else if is_responses {
            let request_arc = ctx.responses_request_arc();
            self.prepare_responses(ctx, &request_arc)?;
        } else {
            error!(
                function = "HarmonyPreparationStage::execute",
                "Unsupported request type for Harmony pipeline"
            );
            return Err(error::bad_request(
                "harmony_request_type_invalid",
                "Only Chat and Responses requests supported in Harmony pipeline".to_string(),
            ));
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyPreparation"
    }
}

impl HarmonyPreparationStage {
    /// Prepare a chat completion request using Harmony encoding
    fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 1: Filter tools if needed
        let mut body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Build structural tag constraint
        let tool_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            Self::generate_tool_call_constraint(tools, body_ref.tool_choice.as_ref())
                .map_err(|e| *e)?
        } else {
            None
        };
        let response_format_constraint =
            Self::generate_response_format_constraint(body_ref.response_format.as_ref())
                .map_err(|e| *e)?;

        // Reject requests that specify both tool call and response_format constraints
        if tool_constraint.is_some() && response_format_constraint.is_some() {
            return Err(error::bad_request(
                "invalid_request_parameters",
                "Constrained decoding (response_format) is not compatible with tool calls",
            ));
        }

        let has_response_format_constraint = response_format_constraint.is_some();
        let constraint = tool_constraint.or(response_format_constraint);

        // If response_format was converted to a structural tag, clear it from the request
        // so the backend builder doesn't also try to add a json_schema constraint from it.
        if has_response_format_constraint {
            let mut owned = body_ref.into_owned();
            owned.response_format = None;
            body_ref = Cow::Owned(owned);
        }

        // Step 3: Build via Harmony
        let build_output = self.builder.build_from_chat(&body_ref).map_err(|e| {
            error!(
                function = "prepare_chat",
                error = %e,
                "Harmony build failed for chat request"
            );
            error::bad_request("harmony_build_failed", format!("Harmony build failed: {e}"))
        })?;

        // Step 4: Store results
        ctx.state.preparation = Some(PreparationOutput::Harmony {
            token_ids: build_output.input_ids,
            selection_text: build_output.selection_text,
            tool_constraints: constraint,
            modified_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(Box::new(body_ref.into_owned()))
            } else {
                None
            },
            harmony_messages: build_output.harmony_messages,
            harmony_stop_ids: build_output.stop_token_ids,
        });

        Ok(None)
    }

    /// Prepare a responses API request using Harmony encoding
    ///
    /// For responses API, we build from conversation history using the same Harmony
    /// encoding that the builder provides. This handles the MCP loop integration.
    pub fn prepare_responses(
        &self,
        ctx: &mut RequestContext,
        request: &ResponsesRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 1: Extract function tools with schemas from ResponseTools
        let mut function_tools = extract_tools_from_response_tools(request.tools.as_deref());

        let chat_tool_choice = request
            .tool_choice
            .as_ref()
            .map(|tc| tc.to_chat_tool_choice());

        // Step 2: Filter tools based on tool_choice (AllowedTools or Function)
        // Note: Tool existence is already validated in ResponsesRequest::validate()
        if let Some(filtered) =
            utils::filter_tools_by_tool_choice(&function_tools, chat_tool_choice.as_ref())
        {
            function_tools = filtered;
        }

        // Step 3: Generate Harmony structural tags
        let tool_constraint = if function_tools.is_empty() {
            None
        } else {
            Self::generate_tool_call_constraint(&function_tools, chat_tool_choice.as_ref())
                .map_err(|e| *e)?
        };

        let text_constraint = if let Some(text_config) = &request.text {
            Self::generate_text_format_constraint(text_config).map_err(|e| *e)?
        } else {
            None
        };

        if tool_constraint.is_some() && text_constraint.is_some() {
            error!(
                function = "prepare_responses",
                "Conflicting constraints: both tool_choice and text format specified"
            );
            return Err(error::bad_request(
                "conflicting_constraints",
                "Cannot use both tool_choice (required/function) and text format (json_object/json_schema) simultaneously".to_string(),
            ));
        }

        let constraint = tool_constraint.or(text_constraint);

        // Step 3: Build via Harmony from responses API request
        let build_output = self.builder.build_from_responses(request).map_err(|e| {
            error!(
                function = "prepare_responses",
                error = %e,
                "Harmony build failed for responses request"
            );
            error::bad_request("harmony_build_failed", format!("Harmony build failed: {e}"))
        })?;

        // Step 4: Store results with constraint
        ctx.state.preparation = Some(PreparationOutput::Harmony {
            token_ids: build_output.input_ids,
            selection_text: build_output.selection_text,
            tool_constraints: constraint,
            modified_request: None,
            harmony_messages: build_output.harmony_messages,
            harmony_stop_ids: build_output.stop_token_ids,
        });

        Ok(None)
    }

    /// Generate Harmony structural tag for structured output (text field)
    ///
    /// Converts text.format to structural tag that constrains the final channel.
    /// Returns None if text.format is not specified or is "text".
    fn generate_text_format_constraint(
        text_config: &openai_protocol::responses::TextConfig,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        use openai_protocol::responses::TextFormat;

        let Some(format) = &text_config.format else {
            return Ok(None);
        };

        match format {
            TextFormat::Text => Ok(None),
            TextFormat::JsonObject => {
                let tag = build_text_format_structural_tag(&serde_json::json!({"type": "object"}))
                    .map_err(|e| {
                        error!(
                            function = "generate_text_format_constraint",
                            error = %e,
                            "Failed to build text format structural tag for JsonObject"
                        );
                        Box::new(error::internal_error("build_text_format_tag_failed", e))
                    })?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            TextFormat::JsonSchema { schema, .. } => {
                let tag = build_text_format_structural_tag(schema).map_err(|e| {
                    error!(
                        function = "generate_text_format_constraint",
                        error = %e,
                        "Failed to build text format structural tag for JsonSchema"
                    );
                    Box::new(error::internal_error("build_text_format_tag_failed", e))
                })?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
        }
    }

    /// Generate Harmony structural tag for Chat Completions response_format
    ///
    /// Converts response_format (json_object, json_schema) to structural tag that constrains
    /// the final channel. Uses the same `build_text_format_structural_tag` as the Responses API.
    /// Returns None if response_format is not specified or is "text".
    fn generate_response_format_constraint(
        response_format: Option<&ResponseFormat>,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        let Some(format) = response_format else {
            return Ok(None);
        };

        let schema = match format {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject => Cow::Owned(serde_json::json!({"type": "object"})),
            ResponseFormat::JsonSchema { json_schema } => Cow::Borrowed(&json_schema.schema),
        };

        let tag = build_text_format_structural_tag(&schema).map_err(|e| {
            error!(
                function = "generate_response_format_constraint",
                error = %e,
                "Failed to build structural tag for response_format"
            );
            Box::new(error::internal_error("build_response_format_tag_failed", e))
        })?;
        Ok(Some(("structural_tag".to_string(), tag)))
    }

    /// Generate Harmony structural tag for tool constraints
    ///
    /// Uses structural tags with `triggered_tags` format to force Harmony format output.
    /// This ensures the model outputs in Harmony format (with channels) even when constrained.
    fn generate_tool_call_constraint(
        tools: &[Tool],
        tool_choice: Option<&ToolChoice>,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        let Some(choice) = tool_choice else {
            return Ok(None);
        };

        match choice {
            ToolChoice::Function { function, .. } => {
                let tag = Self::build_tool_call_structural_tag(tools, Some(&function.name))?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::Value(ToolChoiceValue::Required) => {
                let tag = Self::build_tool_call_structural_tag(tools, None)?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::AllowedTools { mode, .. } => {
                if mode == "required" {
                    let tag = Self::build_tool_call_structural_tag(tools, None)?;
                    Ok(Some(("structural_tag".to_string(), tag)))
                } else {
                    Ok(None)
                }
            }
            ToolChoice::Value(_) => Ok(None),
        }
    }

    /// Build Harmony structural tag for tool calling constraints
    ///
    /// Supports both reasoning-enabled and reasoning-disabled modes:
    /// - With reasoning: triggers on `<|start|>assistant<|channel|>commentary` (waits for analysis)
    /// - Without reasoning: triggers on `<|channel|>commentary` (goes directly to commentary)
    fn build_tool_call_structural_tag(
        tools: &[Tool],
        specific_function: Option<&str>,
    ) -> Result<String, Box<Response>> {
        let mut tags = Vec::new();

        // Filter tools if specific function requested
        let tools_to_use: Vec<&Tool> = if let Some(func_name) = specific_function {
            tools
                .iter()
                .filter(|t| t.function.name == func_name)
                .collect()
        } else {
            tools.iter().collect()
        };

        // Validate specific function exists
        match specific_function {
            Some(tool_name) if tools_to_use.is_empty() => {
                error!(
                    function = "generate_tool_call_constraint",
                    tool_name = %tool_name,
                    "Specified tool not found in tools list"
                );
                return Err(Box::new(error::bad_request(
                    "tool_not_found",
                    format!("Tool '{tool_name}' not found in tools list"),
                )));
            }
            _ => {}
        }

        // Build tags for each tool - need two patterns per tool for reasoning on/off
        for tool in tools_to_use {
            let tool_name = &tool.function.name;
            let params_schema = &tool.function.parameters;

            // Pattern 1: For reasoning-enabled mode (with analysis channel before commentary)
            tags.push(json!({
                "begin": format!("<|start|>assistant<|channel|>commentary to=functions.{}<|constrain|>json<|message|>", tool_name),
                "content": {
                    "type": "json_schema",
                    "json_schema": params_schema
                },
                "end": "" // `end` is empty because <|call|> comes naturally from Harmony stop tokens
            }));

            // Pattern 2: For reasoning-disabled mode (goes directly to commentary channel)
            tags.push(json!({
                "begin": format!("<|channel|>commentary to=functions.{}<|constrain|>json<|message|>", tool_name),
                "content": {
                    "type": "json_schema",
                    "json_schema": params_schema
                },
                "end": ""
            }));
        }

        let stop_after_first = specific_function.is_some();

        let structural_tag = json!({
            "format": {
                "type": "triggered_tags",
                "triggers": ["<|start|>assistant<|channel|>commentary", "<|channel|>commentary"],
                "tags": tags,
                "at_least_one": true,
                "stop_after_first": stop_after_first
            }
        });

        serde_json::to_string(&structural_tag).map_err(|e| {
            error!(
                function = "generate_tool_call_constraint",
                error = %e,
                "Failed to serialize structural tag"
            );
            Box::new(error::internal_error(
                "serialize_structural_tag_failed",
                format!("Failed to serialize structural tag: {e}"),
            ))
        })
    }
}

/// Build Harmony structural tag for structured output (JSON schema constraint)
///
/// Creates a structural tag that handles the full Harmony channel flow:
/// 1. `<|channel|>analysis` trigger → reasoning content (any_text) until `<|end|>`
/// 2. Free text (allows `<|start|>assistant` between messages)
/// 3. `<|channel|>final` trigger → JSON content constrained to schema
///
/// Both triggers are needed so xgrammar's `at_least_one` mode allows the analysis
/// channel tokens (otherwise xgrammar blocks all non-trigger-prefix tokens).
///
/// This is used for the Responses API text.format field (json_object or json_schema).
pub(crate) fn build_text_format_structural_tag(
    schema: &serde_json::Value,
) -> Result<String, String> {
    let structural_tag = json!({
        "format": {
            "type": "triggered_tags",
            "triggers": ["<|channel|>analysis", "<|channel|>final"],
            "tags": [
                {
                    // Analysis (reasoning) channel: any text until <|end|>
                    "begin": "<|channel|>analysis<|message|>",
                    "content": { "type": "any_text" },
                    "end": "<|end|>"
                },
                {
                    // Final channel: JSON content constrained to schema
                    "begin": "<|channel|>final<|constrain|>json<|message|>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema
                    },
                    "end": ""
                }
            ],
            "at_least_one": true,
            "stop_after_first": false
        }
    });

    serde_json::to_string(&structural_tag)
        .map_err(|e| format!("Failed to serialize structural tag for structured output: {e}"))
}
