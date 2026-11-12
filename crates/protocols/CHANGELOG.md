# Changelog

All notable changes to `openai-protocol` are recorded here, newest first.

## Unreleased

### Added

- `ValidatedQuery<T>` and `ValidatedPath<T>`, behind the `axum` feature: a route's query string
  and path segments read as the typed routes' body extractor reads a body. A query value the type
  does not take, one that does not parse or a required one that is missing, and a path segment
  that is not UTF-8 or does not parse as its type answer the gateway's JSON error envelope
  (`400`, `error.code` `unknown_parameter`, `invalid_type`, `invalid_value` or
  `missing_required_parameter`, `error.param` the parameter's or the segment's name) where the
  framework answers text/plain; a rejection of the route's own making (a path parameter the
  route does not declare, a shape the extractor cannot fill) keeps the framework's answer. The
  gateway's model, Responses id, realtime and Conversations list and id routes read them.
- `ChatCompletionRequest::dynamic_tools_for`, `declares_tools_for`, `effective_tool_list_for`,
  `shared_tool_list_for` and `callable_tools_for`: the tool-set readers under a provider profile
  the caller derived already, beside `template_thinking_for`. The readers without the suffix
  derive the profile from the model id as before; a serving path that derives it once per
  request no longer derives it again at each of them.

### Changed

- Breaking: `ResponseFormat::JsonSchema` carries `Box<JsonSchemaFormat>` and
  `ResponseFormat::StructuralTag` carries `Box<serde_json::Map<String, Value>>`. A format had
  grown from 104 to 128 bytes when the schema kept its description and the structural tag
  arrived, and the Gateway's chat route task, which holds a request's format three times over
  its awaits, moved up one allocator size class on every chat request. Behind a pointer each, a
  format is 16 bytes, a chat request 1,040 (1,152 before), and the task is back in its class;
  the wire shape and the schema are unchanged, and a text or json_object format allocates
  nothing.
- Breaking: on `ChatMessage::Assistant`, `function_call` is `Option<Box<FunctionCallResponse>>`,
  and the provider extension's `tools` is `Option<Box<serde_json::Value>>`. The assistant turn is
  the enum's largest variant, so the two rarely-present values had grown every chat message to
  272 bytes (176 before the refusal, the audio reference and the function call were kept), and
  a history of four or eight turns moved up one allocator size class on every chat request.
  Behind a pointer each, a message is 168 bytes and the history is back in its class; the wire
  shape and the schema are unchanged.

### Removed

- Breaking: the fifty-two builder constructors nothing in this repository uses are removed, with
  the imports only they used; the change before this one deprecated them. They are: the
  `ClientEvent` constructors `conversation_item_delete`, `conversation_item_retrieve`,
  `conversation_item_truncate`, `input_audio_buffer_append`, `input_audio_buffer_clear`,
  `input_audio_buffer_commit`, `output_audio_buffer_clear`, `response_cancel`; the `ServerEvent`
  constructors `session_updated`, `conversation_created`, `conversation_item_created`,
  `conversation_item_added`, `conversation_item_done`, `conversation_item_deleted`,
  `conversation_item_retrieved`, `conversation_item_truncated`,
  `input_audio_transcription_completed`, `input_audio_transcription_completed_with_logprobs`,
  `input_audio_transcription_delta`, `input_audio_transcription_failed`,
  `input_audio_transcription_segment`, `input_audio_buffer_cleared`, `input_audio_buffer_committed`,
  `input_audio_buffer_speech_started`, `input_audio_buffer_speech_stopped`,
  `input_audio_buffer_timeout_triggered`, `dtmf_event_received`, `mcp_list_tools_in_progress`,
  `mcp_list_tools_completed`, `mcp_list_tools_failed`, `mcp_call_in_progress`, `mcp_call_completed`,
  `mcp_call_failed`, `rate_limits_updated`, `response_created`, `response_done`;
  `ResponseEventBuilder::output_audio_buffer_stopped`,
  `ResponseEventBuilder::output_audio_buffer_cleared`; `ItemEventBuilder::output_item_added`,
  `ItemEventBuilder::function_call_arguments_delta`,
  `ItemEventBuilder::function_call_arguments_done`, `ItemEventBuilder::mcp_call_arguments_delta`,
  `ItemEventBuilder::mcp_call_arguments_done`; `ContentEventBuilder::content_part_added`,
  `ContentEventBuilder::content_part_done`, `ContentEventBuilder::output_text_done`,
  `ContentEventBuilder::output_audio_delta`, `ContentEventBuilder::output_audio_done`,
  `ContentEventBuilder::output_audio_transcript_delta`,
  `ContentEventBuilder::output_audio_transcript_done`; and
  `ChatCompletionStreamResponseBuilder::add_choice_tool_args`,
  `ChatCompletionStreamResponseBuilder::add_choice_reasoning`.

### Fixed

- Under the MiniMax profile a `reasoning_effort` of `none` switches thinking off
  (`ProviderProfile::thinking_from_reasoning_effort`, so the `template_thinking` of a chat or
  Messages request), as the vendor's contract for the family reads it: the request answers as one
  with `thinking.type: disabled` does, no reasoning and `reasoning_tokens` 0. The profile had read
  every effort value as thinking on, so a `none` was rendered into the template's effort tag and
  the model reasoned anyway. `minimal` keeps thinking on under the profile, as the contract's
  lowest level. An explicit `thinking.type` wins over the effort as before. Adds the profile's
  second reading of the field, `ProviderProfile::template_reasoning_effort`, with
  `ChatCompletionRequest::template_reasoning_effort_for`,
  `CreateMessageRequest::template_reasoning_effort_for` and
  `CountMessageTokensRequest::template_reasoning_effort_for`: the `reasoning_effort` a prompt is
  rendered with, as the profile hands it to the template given the thinking preference the prompt
  renders with. Every other profile hands the value over as sent; MiniMax hands over the levels
  the contract names (`low`, `medium`, `high`, `xhigh`, `max`), `minimal` as `low`, and nothing
  for any other value or while thinking is off, whether the typed toggle, the effort or, where the
  template reads that key, its own `thinking_mode` switch sent in `chat_template_kwargs` says so
  (the gateway applies the switch under the template's detected thinking key, as it does for the
  mode itself; `ProviderProfile::thinking_from_template_kwargs` is the reading).
- A completion choice without log probabilities is written with `"logprobs": null`, in a buffered
  body and in each chunk of a stream, as the public API sends it and a chat choice is written. The
  key was left out, so a client generated from the public schema, which marks it required, read a
  choice it does not expect. A choice with log probabilities is written as before.
- A completion request with `"echo": null` is served, as the public API allows. The crate refused
  it as a 400 `invalid_type`. The engine flags `no_stop_trim`, `ignore_eos`, `skip_special_tokens`
  and `return_hidden_states` read a `null` as their default too, as the chat request's flags do.
- A `json_schema` response format keeps its `description` when the request is written back for an
  upstream, as the HTTP router and the provider routers write it. The crate dropped the key while
  reading the format. The gRPC path's Responses-to-Chat conversion carries a Responses
  `text.format` description into the Chat request it builds, but a gRPC engine still gets only the
  format's schema, as before. Breaking: `JsonSchemaFormat` gains `description: Option<String>`.
- A usage object whose `prompt_tokens_details` leaves out `cached_tokens` parses, with no tokens
  cached, as the public API defaults the key. The crate refused the whole usage, and the
  completion around it, with ``missing field `cached_tokens` ``. The newer counters an upstream
  reports are kept: `text_tokens`, `image_tokens` and `cache_write_tokens` in the prompt details,
  `text_tokens` in the completion details. A usage the gateway builds is written as before.
  Breaking: `PromptTokenUsageInfo` gains `text_tokens`, `image_tokens` and `cache_write_tokens`,
  and `CompletionTokensDetails` gains `text_tokens`, all `Option<u32>`.
- `GenerateRequest` runs the rules of its `sampling_params`, with the chat route's lower bound on
  `n`, and refuses an empty `input_ids` as the chat route refuses an empty `messages`, so the
  gateway's `/generate` reads its body through the validated extractor and answers a negative
  temperature, a `top_p` above one, an `n` of zero or an empty `input_ids` in the error envelope
  (`400`, with `error.code` and `error.param`) where it sent them to the engine. The rule over the whole
  request, live for the first time, refuses only a body with no prompt at all (none of `text`,
  `input_ids`, `input_embeds`), as `missing_required_parameter`; a body with both `text` and
  `input_ids` and an `input_embeds`-only body stay with the routers as today. The cross-field sampling rules (`min_new_tokens` above
  `max_new_tokens`, more than one structured-output constraint) name their field in `error.param`.
- An assistant message in the chat history keeps its `refusal`, `audio` and `function_call` when
  the request is written back for an upstream, as the HTTP router and the provider routers write
  it. The crate dropped all three while reading the message, so a replayed refused turn reached a
  provider as a bare `{"role": "assistant"}`. The gRPC chat template still gets the turn without
  them, as the engines' own chat utils build it. Breaking: `ChatMessage::Assistant` gains the
  fields `refusal: Option<String>`, `audio: Option<ChatAssistantAudio>` and
  `function_call: Option<FunctionCallResponse>`.
- A GLM-5.3 request with a `minimal`, `medium` or `xhigh` effort is served. The profile reads the
  name as `low`, `high` or `max`, as z.ai does, in `reasoning_effort` and in `thinking.effort`.
  Since 1.11.0 the profile refused those names with `reasoning_effort_not_allowed`, and `medium`
  is the default most clients send (#5181). `none` is still refused, because the series cannot
  switch thinking off.
- An assistant message with its reasoning under both `reasoning_content` and `reasoning` (the key
  of vLLM and OpenRouter, which agent clients send too) is accepted when the two texts are the
  same. Since 1.12.0 the serde alias between the two keys refused such a message as a duplicate
  field (#5092). Two different texts are refused as a value (`invalid_value`; `param` names the
  message). Breaking: the `ChatMessage::Assistant` field `reasoning_content: Option<String>` is now
  `reasoning: AssistantReasoning`, whose `text` is that option.
- `EmbeddingRequest` and `ClassifyRequest` keep the fields they do not model in a flattened
  `other` map, as `ChatCompletionRequest` and `CompletionRequest` do, so the HTTP relay forwards an
  embeddings or classify body as the client wrote it (`truncate_prompt_tokens`,
  `add_special_tokens`, `normalize`, `priority`, `extra_body`, ...). The relay re-serializes the
  typed body and dropped every key the types did not declare. Breaking: the two structs gain the
  field `other: serde_json::Map<String, Value>`.
