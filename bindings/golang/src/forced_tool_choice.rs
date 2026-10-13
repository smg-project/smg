//! The tool parser of a request whose tool choice forced a call.
//!
//! The client constrains `tool_choice: required` and a named function through
//! the generic JSON schema (see `generate_tool_constraint` in `client.rs`: no
//! parser is configured there), so the engine's output is not in the model's
//! own tool-call syntax: `required` yields `[{"name": .., "parameters": ..}]`,
//! which the generic JSON parser reads, and a named function yields that
//! function's arguments object alone, read as the arguments of that one call.
//! The gateway's streaming path makes the same switch; the model's parser
//! serves every other request.

use async_trait::async_trait;
use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
use tool_parser::{
    errors::ParserResult, types::ToolCallItem, FunctionCall, StreamingParseResult, ToolCall,
    ToolParser,
};

use super::runtime::PARSER_FACTORY;

/// The parser that reads this request's output: the generic JSON parser when
/// the request forced a call through the generic schema, a reader of the
/// named function's arguments when it named one, else the model's parser.
pub(crate) fn tool_parser_for(
    model: &str,
    tool_choice: Option<&ToolChoice>,
) -> Option<Box<dyn ToolParser>> {
    let registry = PARSER_FACTORY.registry();
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            Some(Box::new(NamedFunctionParser::new(function.name.clone())))
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required)) => registry.create_parser("json"),
        Some(ToolChoice::AllowedTools { mode, .. }) if mode.as_str() == "required" => {
            registry.create_parser("json")
        }
        _ => registry.create_for_model(model),
    }
}

/// Reads the output of a request whose tool choice named one function: the
/// engine, constrained to that function's parameter schema, emits the
/// arguments object alone, so every piece of text is that call's arguments.
/// The function's name goes out with the first piece.
pub(crate) struct NamedFunctionParser {
    name: String,
    announced: bool,
}

impl NamedFunctionParser {
    pub(crate) fn new(name: String) -> Self {
        Self {
            name,
            announced: false,
        }
    }
}

#[async_trait]
impl ToolParser for NamedFunctionParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok((
            String::new(),
            vec![ToolCall {
                function: FunctionCall {
                    name: self.name.clone(),
                    arguments: output.to_string(),
                },
            }],
        ))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        if chunk.is_empty() {
            return Ok(StreamingParseResult::default());
        }
        let name = (!self.announced).then(|| self.name.clone());
        self.announced = true;
        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls: vec![ToolCallItem {
                tool_index: 0,
                name,
                parameters: chunk.to_string(),
            }],
        })
    }

    fn has_tool_markers(&self, _text: &str) -> bool {
        true
    }

    fn reset(&mut self) {
        self.announced = false;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        ffi::{CStr, CString},
        os::raw::c_char,
        ptr,
        sync::Arc,
    };

    use llm_tokenizer::traits::Tokenizer;
    use serde_json::Value;
    use smg_grpc_client::sglang_proto as proto;

    use super::*;
    use crate::{
        grpc_converter::{
            convert_proto_chunk_to_openai, sgl_grpc_response_converter_create,
            sgl_grpc_response_converter_free, GrpcResponseConverterHandle,
        },
        tokenizer::TokenizerHandle,
    };

    const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#;
    const NAMED: &str = r#"{"type":"function","function":{"name":"get_weather"}}"#;

    #[tokio::test]
    async fn a_forced_choice_picks_the_generic_readers_and_auto_keeps_the_models_parser() {
        let tools: Vec<Tool> = serde_json::from_str(TOOLS).unwrap();
        let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();
        let named: ToolChoice = serde_json::from_str(NAMED).unwrap();
        let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        let allowed: ToolChoice = serde_json::from_str(
            r#"{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"get_weather"}]}"#,
        )
        .unwrap();
        let output = r#"[{"name":"get_weather","parameters":{"city":"Paris"}}]"#;
        let arguments = r#"{"city":"Paris"}"#;

        // `required` and an allowed-tools `required` mode: the generic JSON
        // reader, which turns the array the schema forces into a call.
        for choice in [&required, &allowed] {
            let (_, calls) = tool_parser_for("unmapped-model", Some(choice))
                .unwrap()
                .parse_complete_with_tools(output, &tools)
                .await
                .unwrap();
            assert_eq!(calls.len(), 1, "{choice:?}");
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                serde_json::json!({"city": "Paris"})
            );
        }

        // A named function: the arguments object alone is that call.
        let mut parser = tool_parser_for("unmapped-model", Some(&named)).unwrap();
        let (text, calls) = parser.parse_complete(arguments).await.unwrap();
        assert!(text.is_empty());
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, arguments);
        let first = parser
            .parse_incremental(r#"{"city":"#, &tools)
            .await
            .unwrap();
        let second = parser
            .parse_incremental(r#""Paris"}"#, &tools)
            .await
            .unwrap();
        assert!(first.normal_text.is_empty() && second.normal_text.is_empty());
        assert_eq!(first.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(first.calls[0].parameters, r#"{"city":"#);
        assert_eq!(second.calls[0].name, None);
        assert_eq!(second.calls[0].parameters, r#""Paris"}"#);
        assert_eq!(
            (first.calls[0].tool_index, second.calls[0].tool_index),
            (0, 0)
        );

        // `auto` (and no choice): the model's parser, here the default one,
        // which passes text through and announces no call.
        for choice in [Some(&auto), None] {
            let mut parser = tool_parser_for("unmapped-model", choice).unwrap();
            let released = parser.parse_incremental(output, &tools).await.unwrap();
            assert_eq!(released.normal_text, output, "{choice:?}");
            assert!(released.calls.is_empty());
        }
    }

    /// A tokenizer whose ids are bytes (id = byte + 1), so a test spells the
    /// engine's output as token ids; the decoder releases text byte by byte.
    fn byte_tokenizer() -> Arc<dyn Tokenizer> {
        let path =
            std::env::temp_dir().join(format!("smg-forced-choice-{}.json", uuid::Uuid::now_v7()));
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".into(), serde_json::json!(0));
        for byte in 0_u32..=255 {
            vocab.insert(format!("<0x{byte:02X}>"), serde_json::json!(byte + 1));
        }
        let definition = serde_json::json!({
            "version":"1.0", "added_tokens":[],
            "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":vocab},
            "decoder":{"type":"Sequence", "decoders":[
                {"type":"ByteFallback"}, {"type":"Fuse"}
            ]}
        });
        std::fs::write(&path, serde_json::to_vec(&definition).unwrap()).unwrap();
        let tokenizer =
            llm_tokenizer::HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        Arc::new(tokenizer)
    }

    fn byte_ids(text: &str) -> Vec<u32> {
        text.bytes().map(|byte| u32::from(byte) + 1).collect()
    }

    fn chunk_at(index: u32, text: &str) -> proto::GenerateResponse {
        proto::GenerateResponse {
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: byte_ids(text),
                    index,
                    ..Default::default()
                },
            )),
        }
    }

    fn complete_at(index: u32, finish_reason: &str) -> proto::GenerateResponse {
        proto::GenerateResponse {
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    finish_reason: finish_reason.into(),
                    index,
                    ..Default::default()
                },
            )),
        }
    }

    /// What a streamed choice came to: its content pieces, its tool-call
    /// deltas and its finish reason, with the converter's handle created
    /// the way the client creates it (the tool choice as JSON).
    struct Streamed {
        content: Vec<String>,
        names: Vec<String>,
        arguments: String,
        finish_reason: Option<String>,
    }

    async fn stream_through_converter(tool_choice_json: Option<&str>, chunks: &[&str]) -> Streamed {
        let steps: Vec<(u32, &str)> = chunks.iter().map(|text| (0, *text)).collect();
        stream_choices_through_converter(tool_choice_json, &steps)
            .await
            .remove(&0)
            .expect("choice 0 streamed")
    }

    /// The choices of one request streamed through one converter handle,
    /// their chunks interleaved as given (choice index, text); every choice
    /// ends with a Complete, in index order.
    async fn stream_choices_through_converter(
        tool_choice_json: Option<&str>,
        steps: &[(u32, &str)],
    ) -> BTreeMap<u32, Streamed> {
        let tokenizer = byte_tokenizer();
        let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
            tokenizer: tokenizer.clone(),
        }));
        let model = CString::new("unmapped-model").unwrap();
        let request_id = CString::new("forced-choice").unwrap();
        let tools = CString::new(TOOLS).unwrap();
        let tool_choice = tool_choice_json.map(|json| CString::new(json).unwrap());
        let mut error_out: *mut c_char = ptr::null_mut();
        // SAFETY: every pointer is a live C string or null where the function allows null;
        // the tokenizer handle is a live Box released below.
        let handle = unsafe {
            sgl_grpc_response_converter_create(
                tokenizer_handle,
                model.as_ptr(),
                request_id.as_ptr(),
                tools.as_ptr(),
                tool_choice
                    .as_ref()
                    .map_or(ptr::null(), |choice| choice.as_ptr()),
                ptr::null(),
                ptr::null(),
                1,
                &mut error_out,
            )
        };
        if handle.is_null() {
            // SAFETY: a non-null error_out is a C string the create function allocated.
            let error = unsafe { CStr::from_ptr(error_out) }
                .to_string_lossy()
                .into_owned();
            panic!("converter creation failed: {error}");
        }
        let mut streamed: BTreeMap<u32, Streamed> = BTreeMap::new();
        {
            // SAFETY: the handle was just created and is freed once, below.
            let handle: &mut GrpcResponseConverterHandle = unsafe { &mut *handle };
            let mut responses = Vec::new();
            for (index, text) in steps {
                responses.push(
                    convert_proto_chunk_to_openai(chunk_at(*index, text), handle, &tokenizer).await,
                );
            }
            let indexes: BTreeMap<u32, ()> = steps.iter().map(|(index, _)| (*index, ())).collect();
            for index in indexes.keys() {
                responses.push(
                    convert_proto_chunk_to_openai(complete_at(*index, "stop"), handle, &tokenizer)
                        .await,
                );
            }
            for response in responses.into_iter().flat_map(|result| result.unwrap()) {
                let choice = &response.choices[0];
                let streamed = streamed.entry(choice.index).or_insert_with(|| Streamed {
                    content: Vec::new(),
                    names: Vec::new(),
                    arguments: String::new(),
                    finish_reason: None,
                });
                if let Some(content) = &choice.delta.content {
                    streamed.content.push(content.clone());
                }
                for call in choice.delta.tool_calls.iter().flatten() {
                    if let Some(function) = &call.function {
                        if let Some(name) = &function.name {
                            streamed.names.push(name.clone());
                        }
                        if let Some(arguments) = &function.arguments {
                            streamed.arguments.push_str(arguments);
                        }
                    }
                }
                if let Some(reason) = &choice.finish_reason {
                    streamed.finish_reason = Some(reason.clone());
                }
            }
        }
        // SAFETY: both handles were created above and are released exactly once.
        unsafe {
            sgl_grpc_response_converter_free(handle);
            drop(Box::from_raw(tokenizer_handle));
        }
        streamed
    }

    /// `tool_choice: required` on a model whose own parser is marker-based
    /// (here the default parser): the engine's schema-forced array streams as
    /// a call named in it, no content, and the stream ends with `tool_calls`.
    #[tokio::test]
    async fn a_required_choice_streams_the_forced_array_as_a_tool_call() {
        let streamed = stream_through_converter(
            Some(r#""required""#),
            &[
                r#"[{"name":"get_weather","parameters":{"city":"Pa"#,
                r#"ris"}}]"#,
            ],
        )
        .await;
        assert!(streamed.content.is_empty(), "{:?}", streamed.content);
        assert_eq!(streamed.names, ["get_weather"]);
        assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// A named function: the engine emits that function's arguments object
    /// alone; it streams as that call's arguments, the name with the first
    /// delta, and the stream ends with `tool_calls`.
    #[tokio::test]
    async fn a_named_function_streams_its_arguments_as_that_call() {
        let streamed = stream_through_converter(Some(NAMED), &[r#"{"city":"#, r#""Paris"}"#]).await;
        assert!(streamed.content.is_empty(), "{:?}", streamed.content);
        assert_eq!(streamed.names, ["get_weather"]);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed.arguments).unwrap(),
            serde_json::json!({"city": "Paris"})
        );
        assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// `auto` keeps the model's parser: text that is not in its tool-call
    /// syntax stays content and the stream ends as the engine said.
    #[tokio::test]
    async fn an_auto_choice_keeps_the_models_parser() {
        let streamed = stream_through_converter(Some(r#""auto""#), &["hello ", "there"]).await;
        assert_eq!(streamed.content.concat(), "hello there");
        assert!(streamed.names.is_empty() && streamed.arguments.is_empty());
        assert_eq!(streamed.finish_reason.as_deref(), Some("stop"));
    }

    /// Two choices of one request, their chunks interleaved, under a named
    /// function: each choice has its own parser, so each announces the call's
    /// name with its first delta and keeps its own arguments.
    #[tokio::test]
    async fn each_choice_of_a_named_function_request_announces_its_own_call() {
        let streamed = stream_choices_through_converter(
            Some(NAMED),
            &[
                (0, r#"{"city":"#),
                (1, r#"{"city":"#),
                (0, r#""Paris"}"#),
                (1, r#""Rome"}"#),
            ],
        )
        .await;
        for (index, city) in [(0, "Paris"), (1, "Rome")] {
            let choice = &streamed[&index];
            assert_eq!(choice.names, ["get_weather"], "choice {index}");
            assert_eq!(
                serde_json::from_str::<Value>(&choice.arguments).unwrap(),
                serde_json::json!({"city": city}),
                "choice {index}"
            );
            assert_eq!(
                choice.finish_reason.as_deref(),
                Some("tool_calls"),
                "choice {index}"
            );
        }
    }

    /// Two choices under `required`, read by the buffering generic JSON
    /// parser: each choice's call carries its own arguments and no content,
    /// the other choice's text never enters its buffer.
    #[tokio::test]
    async fn each_choice_of_a_required_request_keeps_its_own_arguments() {
        let streamed = stream_choices_through_converter(
            Some(r#""required""#),
            &[
                (0, r#"[{"name":"get_weather","parameters":{"city":"Pa"#),
                (1, r#"[{"name":"get_weather","parameters":{"city":"Ro"#),
                (0, r#"ris"}}]"#),
                (1, r#"me"}}]"#),
            ],
        )
        .await;
        for (index, city) in [(0, "Paris"), (1, "Rome")] {
            let choice = &streamed[&index];
            assert!(
                choice.content.is_empty(),
                "choice {index}: {:?}",
                choice.content
            );
            assert_eq!(choice.names, ["get_weather"], "choice {index}");
            assert_eq!(
                serde_json::from_str::<Value>(&choice.arguments).unwrap(),
                serde_json::json!({"city": city}),
                "choice {index}"
            );
            assert_eq!(
                choice.finish_reason.as_deref(),
                Some("tool_calls"),
                "choice {index}"
            );
        }
    }
}
