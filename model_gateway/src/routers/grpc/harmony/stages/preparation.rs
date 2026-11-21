//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice},
    responses::ResponsesRequest,
};
use serde_json::{json, Value};
use tracing::{error, warn};

use super::super::HarmonyBuilder;
use crate::routers::{
    error,
    grpc::{
        common::{responses::utils::extract_tools_from_response_tools, stages::PipelineStage},
        context::{HarmonyConstraint, PreparationOutput, RequestContext, RequestType},
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
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
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

        Ok(())
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
    ) -> Result<(), Response> {
        // Step 1: Filter tools if needed
        let mut body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Build structural tag constraint over the tools the choice may
        // force, wherever the request declares them (`callable_tools`; the
        // request declares some when `declares_tools` says so).
        let tool_constraint = if request.declares_tools() {
            Self::generate_tool_call_constraint(
                &request.callable_tools(),
                body_ref.tool_choice.as_ref(),
            )
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
            harmony_stop_ids: build_output.stop_token_ids,
        });

        Ok(())
    }

    /// Prepare a responses API request using Harmony encoding
    ///
    /// For responses API, we build from conversation history using the same Harmony
    /// encoding that the builder provides. This handles the MCP loop integration.
    pub fn prepare_responses(
        &self,
        ctx: &mut RequestContext,
        request: &ResponsesRequest,
    ) -> Result<(), Response> {
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

        let text_constraint = request
            .text
            .as_ref()
            .and_then(Self::generate_text_format_constraint);

        if tool_constraint.is_some() && text_constraint.is_some() {
            warn!(
                function = "prepare_responses",
                "Conflicting constraints: both tool_choice and text format specified"
            );
            return Err(utils::reported(error::bad_request(
                "conflicting_constraints",
                "Cannot use both tool_choice (required/function) and text format (json_object/json_schema) simultaneously".to_string(),
            )));
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
            harmony_stop_ids: build_output.stop_token_ids,
        });

        Ok(())
    }

    /// Generate Harmony structural tag for structured output (text field)
    ///
    /// Converts text.format to structural tag that constrains the final channel.
    /// Returns None if text.format is not specified or is "text".
    fn generate_text_format_constraint(
        text_config: &openai_protocol::responses::TextConfig,
    ) -> Option<HarmonyConstraint> {
        use openai_protocol::responses::TextFormat;

        let schema = match text_config.format.as_ref()? {
            TextFormat::Text => return None,
            TextFormat::JsonObject => Cow::Owned(json!({"type": "object"})),
            TextFormat::JsonSchema { schema, .. } => Cow::Borrowed(schema),
        };
        Some(build_json_constraint(&schema))
    }

    /// Generate Harmony structural tag for Chat Completions response_format
    ///
    /// Converts response_format (json_object, json_schema) to structural tag that constrains
    /// the final channel. Uses the same Harmony framing as the Responses API.
    /// A `structural_tag` response_format is the request's own tag, passed on as it arrived.
    /// Returns None if response_format is not specified or is "text".
    fn generate_response_format_constraint(
        response_format: Option<&ResponseFormat>,
    ) -> Result<Option<HarmonyConstraint>, Box<Response>> {
        let Some(format) = response_format else {
            return Ok(None);
        };

        let schema = match format {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject => Cow::Owned(serde_json::json!({"type": "object"})),
            ResponseFormat::JsonSchema { json_schema } => Cow::Borrowed(&json_schema.schema),
            // The request's own tag goes to the engine as it arrived, as the
            // engine's own server sends it; no channel framing is added.
            ResponseFormat::StructuralTag { .. } => {
                let tag = serde_json::to_string(format).map_err(|e| {
                    Box::new(error::internal_error(
                        "serialize_response_format_failed",
                        e.to_string(),
                    ))
                })?;
                return Ok(Some(HarmonyConstraint {
                    tag,
                    vllm_first_token_tag: None,
                }));
            }
        };

        Ok(Some(build_json_constraint(&schema)))
    }

    /// Generate Harmony structural tag for tool constraints
    ///
    /// Uses structural tags with `triggered_tags` format to force Harmony format output.
    /// This ensures the model outputs in Harmony format (with channels) even when constrained.
    fn generate_tool_call_constraint(
        tools: &[Tool],
        tool_choice: Option<&ToolChoice>,
    ) -> Result<Option<HarmonyConstraint>, Box<Response>> {
        let Some(choice) = tool_choice.filter(|choice| choice.forces_call()) else {
            return Ok(None);
        };
        let function = match choice {
            ToolChoice::Function { function, .. } => Some(function.name.as_str()),
            _ => None,
        };
        Self::build_tool_call_constraint(tools, function).map(Some)
    }

    /// Build Harmony structural tag for tool calling constraints
    ///
    /// vLLM receives a whole-turn grammar with optional reasoning before a
    /// mandatory tool call. Other backends keep the existing triggered tags.
    fn build_tool_call_constraint(
        tools: &[Tool],
        specific_function: Option<&str>,
    ) -> Result<HarmonyConstraint, Box<Response>> {
        let mut tags = Vec::new();
        let mut calls = Vec::new();

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
            let params_schema = tool.function.parameters_schema();
            for constrain in ["", "<|constrain|>json"] {
                calls.push(json!({
                    "type": "tag",
                    "begin": format!("<|channel|>commentary to=functions.{tool_name}{constrain}<|message|>"),
                    "content": {"type": "json_schema", "json_schema": params_schema},
                    // Some vLLM tokenizers handle <|call|> as a stop, so it
                    // must also be possible to complete the grammar first.
                    "end": ["<|call|>", ""],
                }));
            }

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

        Ok(HarmonyConstraint {
            tag: structural_tag.to_string(),
            vllm_first_token_tag: Some(vllm_harmony_tag(json!({
                "type": "or",
                "elements": calls,
            }))),
        })
    }
}

/// The initial assistant role is supplied by the Harmony prompt. After any
/// analysis or preamble, require a fresh assistant header before the result.
fn vllm_harmony_tag(output: Value) -> String {
    let preamble = |begin: &str| {
        // The empty alternative is supported by older XGrammar workers too.
        json!({
            "type": "or",
            "elements": [
                {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": begin,
                            "content": {"type": "any_text"},
                            "end": "<|end|>",
                        },
                        {"type": "const_string", "value": "<|start|>assistant"},
                    ],
                },
                {"type": "const_string", "value": ""},
            ],
        })
    };
    json!({
        "format": {
            "type": "sequence",
            "elements": [
                preamble("<|channel|>analysis<|message|>"),
                preamble("<|channel|>commentary<|message|>"),
                output,
            ],
        },
    })
    .to_string()
}

fn build_json_constraint(schema: &Value) -> HarmonyConstraint {
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

    HarmonyConstraint {
        tag: structural_tag.to_string(),
        vllm_first_token_tag: Some(vllm_harmony_tag(json!({
            "type": "tag",
            "begin": "<|channel|>final<|message|>",
            "content": {"type": "json_schema", "json_schema": schema},
            // The final return token is a normal engine stop. Let the grammar
            // complete after the JSON so the matcher can accept that stop.
            "end": "",
        }))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use serde_json::Value;
    use tracing::Level;

    use super::*;
    use crate::{
        middleware::ReportedRefusal,
        observability::logging::test_support::{test_subscriber, LinesAtWarnOrAbove},
        routers::grpc::regular::stages::testing::components,
    };

    #[test]
    fn vllm_harmony_forced_tool_keeps_reasoning_and_limits_the_recipient() {
        let tools: Vec<Tool> = serde_json::from_value(json!([
            {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}},
        ]))
        .expect("valid tools");
        let constraint = HarmonyPreparationStage::build_tool_call_constraint(&tools, Some("b"))
            .expect("valid tool constraint");
        let grammar: Value =
            serde_json::from_str(&constraint.vllm_first_token_tag.expect("vLLM tag"))
                .expect("valid grammar JSON");
        let parts = grammar["format"]["elements"].as_array().expect("sequence");
        assert_eq!(
            parts[0]["elements"][0]["elements"][0]["begin"],
            "<|channel|>analysis<|message|>"
        );
        let calls = parts[2]["elements"].as_array().expect("call formats");
        assert_eq!(calls.len(), 2);
        for call in calls {
            assert!(call["begin"]
                .as_str()
                .expect("call header")
                .starts_with("<|channel|>commentary to=functions.b<|"));
            assert_eq!(call["end"], json!(["<|call|>", ""]));
        }
    }

    /// A `structural_tag` response format is the engine's constraint as the
    /// client sent it, key for key, with no channel framing around it; a JSON
    /// format gets Harmony framing, and
    /// `text` constrains nothing.
    #[test]
    fn a_structural_tag_response_format_is_the_constraint_as_sent() {
        let constraint = |format: &ResponseFormat| {
            HarmonyPreparationStage::generate_response_format_constraint(Some(format))
                .expect("accepted")
        };
        let sent = json!({
            "type": "structural_tag",
            "format": {"type": "json_schema", "json_schema": {"type": "object"}}
        });
        let format: ResponseFormat = serde_json::from_value(sent.clone()).expect("a tag");
        let raw = constraint(&format).expect("a constraint");
        assert_eq!(
            serde_json::from_str::<Value>(&raw.tag).expect("the tag"),
            sent
        );
        assert!(raw.vllm_first_token_tag.is_none());
        let generated = constraint(&ResponseFormat::JsonObject).expect("a constraint");
        let framed: Value = serde_json::from_str(&generated.tag).expect("the tag");
        assert_eq!(
            framed["format"]["tags"][1]["begin"],
            "<|channel|>final<|constrain|>json<|message|>"
        );
        assert_eq!(
            framed["format"]["tags"][1]["content"]["json_schema"],
            json!({"type": "object"})
        );
        let vllm: Value = serde_json::from_str(&generated.vllm_first_token_tag.expect("vLLM tag"))
            .expect("the tag");
        assert_eq!(vllm["format"]["elements"][0]["elements"][1]["value"], "");
        let final_format = &vllm["format"]["elements"][2];
        assert_eq!(final_format["begin"], "<|channel|>final<|message|>");
        assert_eq!(
            final_format["content"]["json_schema"],
            json!({"type": "object"})
        );
        assert_eq!(final_format["end"], "");
        assert!(constraint(&ResponseFormat::Text).is_none());
    }

    /// A Responses request that forces a tool call and also asks for a JSON text format brings
    /// the conflict itself: one WARN with the cause, and the answer is marked reported. The
    /// check answers before the Harmony build and before the context is touched.
    #[test]
    fn a_tool_choice_with_a_text_format_is_one_warn_and_marked() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "weather as json",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "text": {"format": {"type": "json_object"}}
        }))
        .expect("a Responses request");
        let request = Arc::new(request);
        let mut ctx = RequestContext::for_responses(
            Arc::clone(&request),
            None,
            "m".to_string(),
            components(),
        );
        let lines = LinesAtWarnOrAbove::default();
        let _subscriber = test_subscriber(lines.clone());

        let response = HarmonyPreparationStage::new()
            .prepare_responses(&mut ctx, &request)
            .expect_err("the two constraints conflict");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "conflicting_constraints"
        );
        assert!(response.extensions().get::<ReportedRefusal>().is_some());
        assert_eq!(
            lines.lines(),
            [(
                Level::WARN,
                "Conflicting constraints: both tool_choice and text format specified".to_string()
            )]
        );
    }

    /// A forced call that names a tool the request does not declare cannot come from a request:
    /// validation refuses it at the door, and the Responses preparation narrows the tools to the
    /// chosen one first. If the stage meets it, the two have drifted apart: the line stays ERROR
    /// and the answer is not marked.
    #[test]
    fn a_forced_call_to_an_undeclared_tool_keeps_the_error_line() {
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "function", "function": {"name": "get_weather"}
        }))
        .expect("a forced call");
        let lines = LinesAtWarnOrAbove::default();
        let _subscriber = test_subscriber(lines.clone());

        let response = HarmonyPreparationStage::generate_tool_call_constraint(&[], Some(&choice))
            .expect_err("the tool is not declared");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "tool_not_found"
        );
        assert!(response.extensions().get::<ReportedRefusal>().is_none());
        assert_eq!(
            lines.lines(),
            [(
                Level::ERROR,
                "Specified tool not found in tools list".to_string()
            )]
        );
    }
}
