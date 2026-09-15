//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ToolChoice, ToolChoiceValue},
};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        multimodal, utils, ProcessedMessages,
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage;

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        let (token_ids, processed_messages, tool_constraints) =
            prepare_chat_like(ctx, request).await?;
        ctx.state.preparation = Some(PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        });
        Ok(())
    }
}

/// The chat request → prepared inputs pipeline, shared by the chat endpoint
/// and the transcription endpoint (whose backend request is chat-shaped).
///
/// Applies the model's chat template + multimodal expansion, tokenizes,
/// derives `skip_special_tokens`, and builds the stop decoder — writing the
/// intermediate/decoder/skip-special onto `ctx.state`. Returns the pieces the
/// caller stores in its own `PreparationOutput` variant. Does NOT set
/// `ctx.state.preparation`.
pub(crate) async fn prepare_chat_like(
    ctx: &mut RequestContext,
    request: &ChatCompletionRequest,
) -> Result<(Vec<u32>, ProcessedMessages, Option<(String, String)>), Response> {
    utils::validate_chat_content_parts(&request.messages)
        .map_err(|e| error::bad_request("unsupported_content_part", e))?;
    {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Resolve media-part ordering from the model registry so it stays owned
        // by the per-model spec. Falls back to vLLM-compatible media-first when
        // the model has no multimodal components or matches no spec.
        let model_id = ctx.input.model_id.as_str();
        let tokenizer_entry = ctx
            .components
            .tokenizer_registry
            .get_by_name(model_id)
            .or_else(|| ctx.components.tokenizer_registry.get_by_id(model_id));
        let rendering = match (ctx.components.multimodal.as_ref(), tokenizer_entry.as_ref()) {
            (Some(mm_components), Some(entry)) => {
                multimodal::resolve_media_rendering(
                    model_id,
                    &*tokenizer,
                    mm_components,
                    &entry.id,
                    &entry.source,
                )
                .await
            }
            _ => multimodal::MediaRenderingContract::default(),
        };
        let content_format = if rendering.requires_structured_chat_content {
            llm_tokenizer::chat_template::ChatTemplateContentFormat::OpenAI
        } else {
            tokenizer.chat_template_content_format()
        };

        // Normalize media once. The same plan drives placeholder resolution,
        // rendering, fetching, preprocessing, and final count validation.
        let media_plan =
            multimodal::media_plan_chat(&request.messages, rendering.tool_result_order);
        if rendering.requires_structured_chat_content {
            multimodal::validate_marker_backing(
                multimodal::renderable_image_marker_count_chat(&request.messages),
                &media_plan,
            )
            .map_err(|error| {
                error::bad_request("multimodal_prompt_contract_mismatch", error.to_string())
            })?;
        }
        let (placeholder_tokens, mm_context) = if media_plan.is_empty() {
            (None, None)
        } else if let Some(mm_components) = ctx.components.multimodal.as_ref() {
            let model_id = ctx.input.model_id.as_str();
            let (tokenizer_id, tokenizer_source) = match tokenizer_entry {
                Some(e) => (e.id, e.source),
                None => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        model = %model_id,
                        "Tokenizer entry not found for multimodal processing"
                    );
                    return Err(error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer not found for model: {model_id}"),
                    ));
                }
            };

            let placeholders = multimodal::prepare_placeholder_tokens(
                &media_plan,
                model_id,
                &*tokenizer,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            .map_err(|e| {
                error!(
                    function = "ChatPreparationStage::execute",
                    model = %model_id,
                    error = %e,
                    "Failed to prepare multimodal prompt plan"
                );
                error::bad_request(
                    "invalid_multimodal_request",
                    format!("Invalid multimodal request: {e}"),
                )
            })?;

            (
                Some(placeholders),
                Some((
                    mm_components,
                    model_id,
                    tokenizer_id,
                    tokenizer_source,
                    media_plan,
                )),
            )
        } else {
            error!(
                function = "ChatPreparationStage::execute",
                "Multimodal content detected but multimodal components not initialized"
            );
            return Err(error::bad_request(
                "multimodal_not_supported",
                "Multimodal content detected but multimodal processing is not available",
            ));
        };

        // Step 2: Process messages and apply chat template
        let (processed_messages, prompt_encoding) =
            match utils::process_chat_messages_with_placeholders(
                &body_ref,
                &*tokenizer,
                placeholder_tokens.as_ref(),
                rendering.media_part_order,
                content_format,
            ) {
                Ok(msgs) => msgs,
                Err(e) => {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                    return Err(error::bad_request("process_messages_failed", e));
                }
            };

        // Step 3: Tokenize the prompt the way its renderer said to (a flat
        // encode of the text, or the encode the renderer prepared)
        let encoding = match utils::encode_prompt_blocking(
            tokenizer.clone(),
            &processed_messages.text,
            prompt_encoding,
        )
        .await
        {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {e}"),
                ));
            }
        };

        let mut token_ids = encoding.token_ids().to_vec();

        if let (Some(placeholders), Some((_, _, _, _, media_plan))) =
            (placeholder_tokens.as_ref(), mm_context.as_ref())
        {
            multimodal::validate_rendered_media_anchors(
                media_plan,
                placeholders,
                &*tokenizer,
                &token_ids,
            )
            .map_err(|error| {
                error!(
                    function = "ChatPreparationStage::execute",
                    %error,
                    "Rendered multimodal anchors do not match request media"
                );
                error::bad_request("multimodal_prompt_contract_mismatch", error.to_string())
            })?;
        }

        // Step 4: Full multimodal processing (fetch + preprocess + expand tokens + hash)
        let mut multimodal_intermediate = None;
        if let Some((mm_components, model_id, tokenizer_id, tokenizer_source, media_plan)) =
            mm_context
        {
            match multimodal::process_multimodal_plan(
                media_plan,
                model_id,
                &*tokenizer,
                token_ids,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = "ChatPreparationStage::execute",
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
                }
                Err(e) => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        error = %e,
                        "Multimodal processing failed"
                    );
                    return Err(error::bad_request(
                        "multimodal_processing_failed",
                        format!("Multimodal processing failed: {e}"),
                    ));
                }
            }
        }

        // Step 4: Build tool constraints if needed
        let enable_thinking = utils::resolve_user_thinking(
            request.chat_template_kwargs.as_ref(),
            request.reasoning_effort.as_deref(),
            tokenizer.as_ref(),
        )
        .unwrap_or(true);
        let tool_call_constraint = ctx
            .components
            .tool_parser_factory
            .registry()
            .generate_chat_constraint(
                ctx.components
                    .parser_resolver
                    .tool_parser(&request.model)
                    .as_deref(),
                body_ref.tools.as_deref().unwrap_or_default(),
                request
                    .tool_choice
                    .as_ref()
                    .unwrap_or(&ToolChoice::Value(ToolChoiceValue::Auto)),
                enable_thinking,
            )
            .map_err(|e| {
                error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                error::bad_request(
                    "invalid_tool_configuration",
                    format!("Invalid tool configuration: {e}"),
                )
            })?;

        let preserve_reasoning_special_tokens = request.separate_reasoning
            && utils::reasoning_parser_requires_special_tokens(
                &ctx.components.reasoning_parser_factory,
                ctx.components
                    .parser_resolver
                    .reasoning_parser(&request.model)
                    .as_deref(),
                &request.model,
            );

        // Derive skip_special_tokens from parser and constraint type:
        // - typed reasoning parsers need their control tokens preserved
        // - json_schema: backend forces JSON, no trigger tokens to preserve
        // - structural_tag or no constraint (auto): parser needs trigger tokens
        let skip_special_tokens = if preserve_reasoning_special_tokens {
            false
        } else {
            match &tool_call_constraint {
                Some(c) if c.is_json_schema() => request.skip_special_tokens,
                _ if request.tools.is_some()
                    && !matches!(
                        request.tool_choice,
                        Some(ToolChoice::Value(ToolChoiceValue::None))
                    ) =>
                {
                    false
                }
                _ => request.skip_special_tokens,
            }
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        // Store the intermediate + decoder + derived skip_special_tokens on
        // ctx (PreparationOutput is consumed by request_building before
        // response_processing runs); hand the rest to the caller.
        ctx.state.multimodal_intermediate = multimodal_intermediate;
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok((
            token_ids,
            processed_messages,
            tool_call_constraint.map(|c| c.to_tuple()),
        ))
    }
}

#[cfg(test)]
mod deepseek_v4_vision_preparation_tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use std::{
        io::Cursor,
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use anyhow::Result;
    use axum::http::StatusCode;
    use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use llm_multimodal::{Modality, PreProcessorConfig};
    use llm_tokenizer::{
        chat_template::{
            ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
        },
        encoders::deepseek_v4::{encode_messages, EncodeParams, ThinkingMode},
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
        HuggingFaceTokenizer, TokenizerRegistry,
    };
    use openai_protocol::{chat::ChatCompletionRequest, messages::CreateMessageRequest};
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::{json, Value};
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::{ChatPreparationStage, PipelineStage};
    use crate::{
        routers::grpc::{
            context::{PreparationOutput, RequestContext, SharedComponents},
            multimodal::{
                MediaBatch, MultimodalComponents, MultimodalConfigRegistry, MultimodalModelConfig,
            },
            regular::stages::messages::MessagePreparationStage,
            utils::ParserResolver,
        },
        worker::WorkerRegistry,
    };

    const MODEL: &str = "deepseek-v4-vision-preparation";
    const REFERENCE_ROOT_ENV: &str = "DSV4_VISION_REFERENCE_ROOT";
    const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    struct RecordingTokenizer {
        inner: HuggingFaceTokenizer,
        inputs: Arc<Mutex<Vec<Vec<Value>>>>,
    }

    impl Encoder for RecordingTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            self.inner.encode(input, add_special_tokens)
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            self.inner.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for RecordingTokenizer {
        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for RecordingTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn apply_chat_template(
            &self,
            messages: &[Value],
            _params: ChatTemplateParams,
        ) -> Result<String> {
            self.inputs.lock().unwrap().push(messages.to_vec());
            encode_messages(messages, ThinkingMode::Chat, &EncodeParams::default())
                .map_err(Into::into)
        }

        fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
            self.inner.chat_template_content_format()
        }

        fn thinking_toggle(&self) -> ThinkingToggle {
            ThinkingToggle::DefaultOff
        }

        fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
            Some(ThinkingKeyName::Thinking)
        }

        fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
            &["high", "max"]
        }

        fn think_in_prefill(&self) -> bool {
            true
        }

        fn eos_token_ids(&self) -> &[u32] {
            self.inner.eos_token_ids()
        }
    }

    struct Harness {
        components: Arc<SharedComponents>,
        recorder: Arc<RecordingTokenizer>,
        multimodal: Arc<MultimodalComponents>,
        config_registry: Arc<MultimodalConfigRegistry>,
    }

    #[expect(
        clippy::print_stderr,
        reason = "portable golden tests report explicit skips"
    )]
    fn external_tokenizer_root(test_name: &str) -> Option<PathBuf> {
        let Some(root) = std::env::var_os(REFERENCE_ROOT_ENV) else {
            eprintln!(
                "skipping {test_name}: set {REFERENCE_ROOT_ENV} to the DeepSeek reference artifact root"
            );
            return None;
        };
        Some(PathBuf::from(root).join("dsv4vision-mp8"))
    }

    fn png_base64(width: u32, height: u32, color: [u8; 3]) -> String {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        BASE64_STANDARD.encode(bytes.into_inner())
    }

    async fn harness(test_name: &str) -> Option<Harness> {
        let tokenizer_dir = external_tokenizer_root(test_name)?;
        let tokenizer_json = tokenizer_dir.join("tokenizer.json");
        let inputs = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(RecordingTokenizer {
            inner: HuggingFaceTokenizer::from_file(tokenizer_json.to_str().unwrap()).unwrap(),
            inputs,
        });
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let registered: Arc<dyn Tokenizer> = recorder.clone();
        let outcome = tokenizer_registry
            .load(
                "dsv4-preparation-tokenizer",
                MODEL,
                tokenizer_dir.to_str().unwrap(),
                || async move { Ok(registered) },
            )
            .await
            .unwrap();

        let config_registry = Arc::new(MultimodalConfigRegistry::new());
        config_registry.insert(
            outcome.id().to_string(),
            Arc::new(MultimodalModelConfig {
                config: json!({"model_type": "deepseek_v4", "vision_n_layers": 32}),
                preprocessor_config: PreProcessorConfig::default(),
                video_preprocessor_config: None,
            }),
        );
        let multimodal =
            Arc::new(MultimodalComponents::new(config_registry.clone(), None).unwrap());
        let components = Arc::new(SharedComponents {
            tokenizer_registry,
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: ParserResolver::disabled(),
            multimodal: Some(multimodal.clone()),
        });
        Some(Harness {
            components,
            recorder,
            multimodal,
            config_registry,
        })
    }

    fn source_less_marker_count(value: &Value) -> usize {
        match value {
            Value::Array(values) => values.iter().map(source_less_marker_count).sum(),
            Value::Object(object) => {
                let here = usize::from(object.get("type").and_then(Value::as_str) == Some("image"));
                if here == 1 {
                    assert_eq!(object.len(), 1, "encoder image marker must be source-less");
                }
                here + object.values().map(source_less_marker_count).sum::<usize>()
            }
            _ => 0,
        }
    }

    fn chat_request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    fn messages_request(value: Value) -> CreateMessageRequest {
        serde_json::from_value(value).unwrap()
    }

    fn assert_lookup_counts(harness: &Harness, expected: usize) {
        assert_eq!(
            harness.config_registry.get_or_load_call_count(),
            expected,
            "config lookups moved from the frozen HEAD request profile"
        );
        assert_eq!(
            harness.multimodal.model_lookup_call_count(),
            expected,
            "model-spec lookups must stay paired with config lookups"
        );
    }

    #[tokio::test]
    async fn deepseek_v4_vision_preparation_chat_prompt_tokens_media_and_guards() {
        let Some(harness) =
            harness("deepseek_v4_vision_preparation_chat_prompt_tokens_media_and_guards").await
        else {
            return;
        };
        let data_url = format!("data:image/png;base64,{PNG_BASE64}");
        let request = chat_request(json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }]
        }));
        let mut ctx = RequestContext::for_chat(
            Arc::new(request),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        if let Err(response) = ChatPreparationStage.execute(&mut ctx).await {
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            panic!(
                "chat preparation failed with {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        assert_lookup_counts(&harness, 3);

        let golden: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/multimodal/tests/fixtures/deepseek_v4_vision/prompts.json"
        )))
        .unwrap();
        let expected = &golden["cases"]["image-f1"]["chat-high"];
        let expected_ids = expected["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let PreparationOutput::Chat {
            token_ids,
            processed_messages,
            ..
        } = ctx.state.preparation.as_ref().unwrap()
        else {
            panic!("chat preparation output missing");
        };
        assert_eq!(processed_messages.text, expected["prompt"]);
        assert_eq!(
            harness
                .recorder
                .encode(&processed_messages.text, false)
                .unwrap()
                .token_ids(),
            expected_ids
        );
        let intermediate = ctx.state.multimodal_intermediate.as_ref().unwrap();
        let [batch] = intermediate.batches() else {
            panic!("expected one image batch");
        };
        assert_eq!(batch.media.modality(), Modality::Image);
        assert_eq!(batch.media.len(), 1);
        let [binding] = batch.bindings.as_slice() else {
            panic!("expected one image binding");
        };
        assert_eq!(binding.item_index, 0);
        assert_eq!(binding.prompt_ordinal, 0);
        assert_eq!(binding.structural.offset, 6);
        assert_eq!(
            &token_ids[..binding.structural.offset],
            &expected_ids[..binding.structural.offset]
        );
        assert_eq!(
            &token_ids[binding.structural.offset + binding.structural.length..],
            &expected_ids[binding.structural.offset + 1..]
        );
        {
            let captured = harness.recorder.inputs.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                source_less_marker_count(&Value::Array(captured[0].clone())),
                1
            );
        }

        let text_only = chat_request(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let mut text_ctx = RequestContext::for_chat(
            Arc::new(text_only),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        ChatPreparationStage.execute(&mut text_ctx).await.unwrap();
        assert!(text_ctx.state.multimodal_intermediate.is_none());
        assert_lookup_counts(&harness, 4);

        let assistant_only = chat_request(json!({
            "model": MODEL,
            "messages": [{
                "role": "assistant",
                "content": [{"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG_BASE64}")}}]
            }]
        }));
        let mut assistant_ctx = RequestContext::for_chat(
            Arc::new(assistant_only),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        let response = ChatPreparationStage
            .execute(&mut assistant_ctx)
            .await
            .unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_lookup_counts(&harness, 5);
        assert_eq!(harness.recorder.inputs.lock().unwrap().len(), 2);

        let injected = chat_request(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "literal <｜deepseek_image｜> marker"}]
        }));
        let mut injected_ctx = RequestContext::for_chat(
            Arc::new(injected),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        let response = ChatPreparationStage
            .execute(&mut injected_ctx)
            .await
            .unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_lookup_counts(&harness, 6);

        let no_multimodal_components = Arc::new(SharedComponents {
            tokenizer_registry: harness.components.tokenizer_registry.clone(),
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: ParserResolver::disabled(),
            multimodal: None,
        });
        let no_components_request = chat_request(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let mut no_components_ctx = RequestContext::for_chat(
            Arc::new(no_components_request),
            None,
            MODEL.to_string(),
            no_multimodal_components,
        );
        ChatPreparationStage
            .execute(&mut no_components_ctx)
            .await
            .unwrap();
        assert_lookup_counts(&harness, 6);
    }

    #[tokio::test]
    async fn chat_reordered_tool_results_bind_pixels_to_rendered_placeholders() {
        let Some(harness) =
            harness("chat_reordered_tool_results_bind_pixels_to_rendered_placeholders").await
        else {
            return;
        };
        let call_1_image = png_base64(32, 64, [220, 10, 20]);
        let call_2_image = png_base64(80, 20, [10, 20, 220]);
        let request = chat_request(json!({
            "model": MODEL,
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "first", "arguments": "{}"}
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-2",
                    "content": [
                        {"type": "text", "text": "MARK-call-2"},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{call_2_image}")}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": [
                        {"type": "text", "text": "MARK-call-1"},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{call_1_image}")}}
                    ]
                }
            ]
        }));
        let mut ctx = RequestContext::for_chat(
            Arc::new(request),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        if let Err(response) = ChatPreparationStage.execute(&mut ctx).await {
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            panic!(
                "chat preparation failed with {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        assert_lookup_counts(&harness, 3);

        let PreparationOutput::Chat {
            processed_messages, ..
        } = ctx.state.preparation.as_ref().unwrap()
        else {
            panic!("chat preparation output missing");
        };
        let call_1_marker = processed_messages.text.find("MARK-call-1").unwrap();
        let call_2_marker = processed_messages.text.find("MARK-call-2").unwrap();
        assert!(
            call_1_marker < call_2_marker,
            "DeepSeek rendering must preserve assistant tool-call order"
        );

        let intermediate = ctx.state.multimodal_intermediate.as_ref().unwrap();
        let [batch] = intermediate.batches() else {
            panic!("expected one image batch");
        };
        let MediaBatch::Images(images) = &batch.media else {
            panic!("expected image batch");
        };
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].size(), llm_multimodal::ImageSize::new(32, 64));
        assert_eq!(images[1].size(), llm_multimodal::ImageSize::new(80, 20));
        assert_eq!(&images[0].data().to_rgb8().as_raw()[..3], &[220, 10, 20]);
        assert_eq!(&images[1].data().to_rgb8().as_raw()[..3], &[10, 20, 220]);
        assert_eq!(batch.bindings.len(), 2);
        assert_eq!(batch.bindings[0].item_index, 0);
        assert_eq!(batch.bindings[0].prompt_ordinal, 0);
        assert_eq!(batch.bindings[1].item_index, 1);
        assert_eq!(batch.bindings[1].prompt_ordinal, 1);
    }

    #[tokio::test]
    async fn deepseek_v4_vision_preparation_messages_tool_result_binding_order() {
        let Some(harness) =
            harness("deepseek_v4_vision_preparation_messages_tool_result_binding_order").await
        else {
            return;
        };
        let nested_first = png_base64(800, 100, [220, 10, 20]);
        let top_level_second = png_base64(64, 64, [10, 20, 220]);
        let request = messages_request(json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [
                            {"type": "text", "text": "nested"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": nested_first}}
                        ]
                    },
                    {"type": "text", "text": "top-level"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": top_level_second}}
                ]
            }]
        }));
        let mut ctx = RequestContext::for_messages(
            Arc::new(request),
            None,
            MODEL.to_string(),
            harness.components.clone(),
        );
        if let Err(response) = MessagePreparationStage.execute(&mut ctx).await {
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            panic!(
                "messages preparation failed with {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        assert_lookup_counts(&harness, 3);

        let PreparationOutput::Messages {
            token_ids,
            processed_messages,
            ..
        } = ctx.state.preparation.as_ref().unwrap()
        else {
            panic!("messages preparation output missing");
        };
        assert!(processed_messages.text.contains("top-level"));
        assert!(processed_messages.text.contains("nested"));
        let placeholder_id = harness
            .recorder
            .token_to_id("<｜deepseek_image｜>")
            .unwrap();
        assert!(token_ids.iter().filter(|&&id| id == placeholder_id).count() > 2);

        let intermediate = ctx.state.multimodal_intermediate.as_ref().unwrap();
        let [batch] = intermediate.batches() else {
            panic!("expected one image batch");
        };
        assert_eq!(batch.media.len(), 2);
        let MediaBatch::Images(images) = &batch.media else {
            panic!("expected image batch");
        };
        assert_eq!(images[0].size(), llm_multimodal::ImageSize::new(64, 64));
        assert_eq!(images[1].size(), llm_multimodal::ImageSize::new(800, 100));
        assert_eq!(batch.preprocessed.item_sizes, [(64, 64), (100, 800)]);
        assert_eq!(batch.bindings.len(), 2);
        assert_eq!(batch.bindings[0].item_index, 0);
        assert_eq!(batch.bindings[0].prompt_ordinal, 0);
        assert_eq!(batch.bindings[1].item_index, 1);
        assert_eq!(batch.bindings[1].prompt_ordinal, 1);

        let captured = harness.recorder.inputs.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            source_less_marker_count(&Value::Array(captured[0].clone())),
            2
        );
    }

    #[test]
    fn marker_backing_guard_precedes_empty_plan_return_in_both_stages() {
        for source in [
            include_str!("preparation.rs"),
            include_str!("../messages/preparation.rs"),
        ] {
            let guard = source.find("validate_marker_backing(").unwrap();
            let empty_plan = source.find("media_plan.is_empty()").unwrap();
            assert!(
                guard < empty_plan,
                "marker guard must run before empty-plan return"
            );
        }
    }
}
