//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Function, ResponseFormat, Tool, ToolChoice, ToolChoiceValue, ToolReference},
    responses::{ResponsesRequest, ResponsesToolChoice, ToolChoiceOptions},
};
use serde_json::{json, Value};
use tracing::{error, warn};

use super::super::{
    builder::response_tool_descriptions, grammar::VllmHarmonyGrammar, HarmonyBuilder,
};
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
        let callable_tools = request.callable_tools();
        let tool_choice = body_ref.tool_choice.as_ref();
        let tool_constraint = if request.declares_tools() {
            Self::generate_tool_call_constraint(&callable_tools, tool_choice).map_err(|e| *e)?
        } else {
            None
        };
        let response_format_constraint = Self::generate_response_format_constraint(
            body_ref.response_format.as_ref(),
            optional_tools(&callable_tools, tool_choice),
        )
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

        let text_constraint = Self::generate_text_format_constraint(request);

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
    fn generate_text_format_constraint(request: &ResponsesRequest) -> Option<HarmonyConstraint> {
        use openai_protocol::responses::TextFormat;

        let schema = match request.text.as_ref()?.format.as_ref()? {
            TextFormat::Text => return None,
            TextFormat::JsonObject => Cow::Owned(json!({"type": "object"})),
            TextFormat::JsonSchema { schema, .. } => Cow::Borrowed(schema),
        };
        Some(build_json_constraint(
            &schema,
            &optional_response_tools(request),
        ))
    }

    /// Generate Harmony structural tag for Chat Completions response_format
    ///
    /// Converts response_format (json_object, json_schema) to structural tag that constrains
    /// the final channel. Uses the same Harmony framing as the Responses API.
    /// A `structural_tag` response_format is the request's own tag, passed on as it arrived.
    /// Returns None if response_format is not specified or is "text".
    fn generate_response_format_constraint(
        response_format: Option<&ResponseFormat>,
        auto_tools: &[Tool],
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
                    vllm_grammar: None,
                }));
            }
        };

        Ok(Some(build_json_constraint(&schema, auto_tools)))
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
        let vllm_grammar = VllmHarmonyGrammar::new(tools_to_use.iter().copied(), None);
        for tool in tools_to_use {
            let tool_name = &tool.function.name;
            let params_schema = tool.function.parameters_schema();
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
            vllm_grammar: Some(vllm_grammar),
        })
    }
}

/// A JSON answer may follow a lookup when the request leaves tool use optional.
fn optional_tools<'a>(tools: &'a [Tool], choice: Option<&ToolChoice>) -> &'a [Tool] {
    match choice {
        None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => tools,
        Some(ToolChoice::AllowedTools { mode, .. }) if mode == "auto" => tools,
        _ => &[],
    }
}

/// Hosted tools dispatched by MCP are exposed as functions. An image generation
/// declaration without an MCP replacement has its own synthesized function.
fn optional_response_tools(request: &ResponsesRequest) -> Vec<Tool> {
    let allowlist = match request.tool_choice.as_ref() {
        None | Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto)) => None,
        Some(ResponsesToolChoice::AllowedTools { mode, tools, .. }) if mode == "auto" => {
            Some(tools)
        }
        _ => return Vec::new(),
    };
    response_tool_descriptions(request.tools.as_deref().unwrap_or_default())
        .into_iter()
        .filter(|tool| {
            allowlist.is_none_or(|allowed| {
                allowed.iter().any(|reference| {
                    reference.function_name() == Some(tool.name.as_str())
                        || (matches!(reference, ToolReference::ImageGeneration)
                            && tool.name == "image_generation")
                })
            })
        })
        .map(|tool| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: tool.name,
                description: Some(tool.description),
                parameters: tool.parameters,
                strict: None,
                extra: Default::default(),
            },
        })
        .collect()
}

fn build_json_constraint(schema: &Value, auto_tools: &[Tool]) -> HarmonyConstraint {
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
        vllm_grammar: Some(VllmHarmonyGrammar::new(auto_tools, Some(schema))),
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
        let grammar = constraint.vllm_grammar.expect("vLLM grammar");
        for token_nodes in [false, true] {
            let tag = grammar.render(token_nodes);
            let output = &tag["format"]["elements"][2]["elements"];
            let calls = output.as_array().expect("call formats");
            assert_eq!(calls.len(), 1);
            assert_eq!(
                calls[0]["elements"][1]["value"],
                "commentary to=functions.b"
            );
            let channel = &calls[0]["elements"][0];
            assert_eq!(
                channel["type"],
                if token_nodes { "token" } else { "const_string" }
            );
            assert_eq!(
                channel[if token_nodes { "token" } else { "value" }],
                "<|channel|>"
            );
            assert_eq!(
                calls[0]["elements"][4]["json_schema"],
                json!({"type": "object"})
            );
        }
    }

    #[test]
    fn chat_json_answers_keep_only_optional_tool_recipients() {
        let declared = json!([
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}},
        ]);
        for (choice, expected) in [
            (None, vec!["lookup", "other"]),
            (Some(json!("auto")), vec!["lookup", "other"]),
            (
                Some(json!({"type": "allowed_tools", "mode": "auto",
                "tools": [{"type": "function", "name": "lookup"}]})),
                vec!["lookup"],
            ),
            (Some(json!("none")), vec![]),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "gpt-oss", "messages": [{"role": "user", "content": "Lookup"}],
                "tools": declared, "tool_choice": choice, "response_format": {"type": "json_object"}
            }))
            .expect("valid chat request");
            let callable = request.callable_tools();
            let auto = optional_tools(&callable, request.tool_choice.as_ref());
            let constraint = HarmonyPreparationStage::generate_response_format_constraint(
                request.response_format.as_ref(),
                auto,
            )
            .expect("valid response format")
            .expect("JSON constraint");
            let tag = constraint
                .vllm_grammar
                .expect("vLLM grammar")
                .render(true)
                .to_string();
            for name in ["lookup", "other"] {
                assert_eq!(
                    tag.contains(&format!("functions.{name}")),
                    expected.contains(&name)
                );
            }
            assert!(tag.contains("final"));
            assert!(
                !constraint.tag.contains("functions."),
                "only vLLM gets the alternatives"
            );
        }
    }

    #[test]
    fn responses_json_keeps_advertised_and_allowlisted_hosted_functions() {
        let mut request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-oss", "input": "Lookup", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "image_generation"}
            ],
            "text": {"format": {"type": "json_schema", "name":"answer",
                "schema":{"type":"object","properties":{"answer":{"type":"string"}}}}}
        }))
        .expect("valid Responses request");
        for (choice, lookup, image) in [
            (None, true, true),
            (Some(json!("auto")), true, true),
            (
                Some(json!({"type": "allowed_tools", "mode": "auto",
                "tools": [{"type": "image_generation"}]})),
                false,
                true,
            ),
            (
                Some(json!({"type": "allowed_tools", "mode": "auto",
                "tools": [{"type": "function", "name": "lookup"}]})),
                true,
                false,
            ),
            (Some(json!("none")), false, false),
        ] {
            request.tool_choice =
                choice.map(|value| serde_json::from_value(value).expect("valid tool choice"));
            let constraint = HarmonyPreparationStage::generate_text_format_constraint(&request)
                .expect("JSON constraint");
            let tag = constraint
                .vllm_grammar
                .expect("vLLM grammar")
                .render(true)
                .to_string();
            assert_eq!(tag.contains("functions.lookup"), lookup);
            assert_eq!(tag.contains("functions.image_generation"), image);
            assert!(tag.contains("final") && tag.contains("answer"));
            assert!(!constraint.tag.contains("functions."));
        }
    }

    #[test]
    fn responses_json_accepts_a_custom_tool_lifted_from_additional_tools() {
        use crate::routers::common::responses_compat::lower_responses_request_for_local_backend;

        let mut request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-oss",
            "input": [
                {"type": "additional_tools", "tools": [{
                    "type": "namespace", "name": "functions", "description": "Editor",
                    "tools": [{"type": "custom", "name": "edit", "description": "Edit a file"}]
                }]},
                {"role": "user", "content": "Edit the file"}
            ],
            "text": {"format": {"type": "json_object"}}
        }))
        .expect("valid Responses request");
        // The history loader lowers the request before preparing it for Harmony.
        lower_responses_request_for_local_backend(&mut request).expect("inline tool is lifted");
        let tag = HarmonyPreparationStage::generate_text_format_constraint(&request)
            .expect("JSON constraint")
            .vllm_grammar
            .expect("vLLM grammar")
            .render(true);
        let outputs = tag["format"]["elements"][2]["elements"]
            .as_array()
            .expect("tool and final alternatives");
        assert_eq!(outputs.len(), 2);
        let call = &outputs[0]["elements"];
        assert_eq!(call[1]["value"], "commentary to=functions.edit");
        assert_eq!(call[4]["json_schema"]["required"], json!(["input"]));
        assert_eq!(outputs[1]["elements"][1]["value"], "final");
    }

    /// A `structural_tag` response format is the engine's constraint as the
    /// client sent it, key for key, with no channel framing around it; a JSON
    /// format gets Harmony framing, and
    /// `text` constrains nothing.
    #[test]
    fn a_structural_tag_response_format_is_the_constraint_as_sent() {
        let constraint = |format: &ResponseFormat| {
            HarmonyPreparationStage::generate_response_format_constraint(Some(format), &[])
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
        assert!(raw.vllm_grammar.is_none());
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
        let vllm = generated.vllm_grammar.expect("vLLM grammar").render(true);
        let answer = &vllm["format"]["elements"][2]["elements"][0]["elements"];
        assert_eq!(answer[0], json!({"type":"token","token":"<|channel|>"}));
        assert_eq!(answer[1]["value"], "final");
        assert_eq!(answer[2], json!({"type":"token","token":"<|message|>"}));
        assert_eq!(answer[3]["json_schema"], json!({"type":"object"}));
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
