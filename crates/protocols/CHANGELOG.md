# Changelog

All notable changes to `openai-protocol` are recorded here, newest first.

## Unreleased

### Changed

- A `CompletionRequest` whose `best_of` equals `n` validates: the public API gives both fields
  the default 1, so a request that states both defaults is the plain one and is no longer refused
  under `best_of_less_than_n`. A `best_of` under `n` is still refused with that code (its sentence
  now reads `best_of must be greater than or equal to n`), and on a stream only a `best_of` above
  `n` is refused under `best_of_not_supported_with_stream`, whose sentence now reads
  `best_of greater than n is not supported when stream is enabled`.

- `EmbeddingObject.embedding` is an `Embedding`: the float array, or the base64 string of the
  vector's little-endian f32 bytes when the request asked for `encoding_format: "base64"`; the
  type is untagged, so an answer reads either shape. The gateway's gRPC embeddings path answered
  the float array for every value of `encoding_format`; it now encodes for `base64`. Breaking:
  `EmbeddingObject.embedding` is `Embedding`, not `Vec<f32>`.

- A Messages request's `tool_use` block whose `input` is not an object is still refused under
  `tool_use_input_not_an_object` with the public API's sentence, and the sentence now names the
  block: `messages.1.content.0.tool_use.input: Input should be a valid dictionary (the tool_use
  block toolu_... for calc carries an array)`, so a caller who replays a streamed answer sees which
  block and what it carries.

### Added

- `responses::validate_metadata_pairs` (the public API's bounds on a `metadata` object, for any
  map of its pairs), `responses::validate_metadata_pair_count` (its 16-pair bound alone, for a
  count read on its own), `responses::validate_input_message_role` (an input message's role in
  the public set) and `common::json_type_name` are public: a route that reads a metadata object or
  stores message items of its own (the gateway's Conversations routes) validates them with the
  rules the Responses request already validates with, in the same words and under the same codes.
- `into_chat_tools` and the by-value `From<CustomTool> for Tool` and `From<InputSchema> for Value`:
  a caller that owns a Messages request's tools converts them into chat tools without copying the
  schemas (the properties and the additional fields move in); the by-reference forms stay for a
  caller that keeps its tools and copy once, as before.
- `MessagesBetas`, beside `CreateMessageRequest`: the two betas the request rules read, kept as
  flags filled in one pass over the `anthropic-beta` header's values as the public API names them
  (the dated form `name-YYYY-MM-DD`; a header that repeats or lists several; a longer name that
  begins with a known one is another beta; no allocation per request), with `values_name` for
  another name read from the raw values; and
  `validate_with_betas`, the request rules that read a beta beside the body
  (`thinking.budget_tokens` under `max_tokens` unless the interleaved-thinking beta is named;
  `thread` only with the message-threads beta), refused with the codes and messages the gateway's
  Messages route answered before. That route reads both through a typed header extractor, and the
  external router's beta reading shares the grammar.
- `ValueBody`, behind the `axum` feature: the body of a route that reads JSON it checks itself, a
  newtype over `serde_json::Value` with nothing to validate or normalize before the handler sees
  it, so such a route rides `ValidatedJson` and refuses a body that does not parse, or one under
  another or no content type, in the gateway's JSON error envelope (`400`, `error.code`
  `invalid_json` or `json_parse_error`) as the typed routes do. A repeated key is refused as a
  typed body's is (`json_parse_error`, serde's `duplicate field` words, the place as `param`), at
  the body's own keys and at any depth below them; the keys of `metadata` keep the last value, as
  that map's do in a typed body. The gateway's Conversations routes read it.
- `Usage::with_image_tokens`: the prompt's image token count reported as
  `prompt_tokens_details.image_tokens`, kept within `prompt_tokens`; `None` leaves the details as
  they are. The gateway's chat route reports the tokens its own multimodal expansion wrote into
  the prompt for the request's images, on the non-streamed answer and the final usage chunk
  alike; a request without such an image answers as before.
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
- `ModelCard` derives `PartialEq`: a caller that holds a worker's cards tells a refresh that
  repeats them from one that changes them (the gateway's model refresh publishes nothing for the
  first).

### Changed

- A `metadata` value that is not a string is refused as `400 invalid_type`, naming the key and the
  type sent (`Invalid type for 'metadata.k': expected a string, but got an object instead.`), as
  the public API refuses one; before, a request's `metadata` passed validation with an object, a
  number, a boolean or a null value under a key.
- A rerank worker's answer is read as `RerankWorkerResponse { results, usage, id }`: its `usage`
  (a rerank server's `prompt_tokens` and `total_tokens`, read as the crate's `UsageInfo`, whose
  `completion_tokens` defaults; a usage in another spelling is left out, as before) and its `id`
  ride into the gateway's response, and
  `RerankResponse::new` sorts the results by score, highest first, before `top_n` cuts them,
  whatever order the worker sent. The crate read the results alone and kept the worker's order,
  so a worker that did not sort lost its best documents to the cut and the response carried none
  of the worker's usage. A `RerankRequest` without a `user` writes no `user` key, as one without a
  `rid` writes none. Breaking: `RerankWorkerResponse` is a struct (the enum and its `into_results`
  are gone) and `RerankResponse::new` takes `(results, model, usage, id)` where it took
  `(results, model, request_id)`.
- Breaking: `normalize_input_item` takes the item by value. A bare `{role, content}` message moves
  its content and role into the message item built from it, and every other item comes back as it
  was passed, so a caller that owns its input no longer copies each item; a caller holding a
  reference passes `item.clone()`, which costs what the old signature did.
- `ContentPart`: a part whose `type` names a kind the gateway knows (`text`, `image_url`,
  `audio_url`, `input_audio`, `video_url`, `refusal`) but whose body does not fit it is refused with
  serde's reason for the field (`400`, `error.code` `invalid_type`, `integer_below_min_value` or
  `missing_required_parameter`, `error.param` the message's path, extended by the missing field's
  path when a field is the one missing) instead of reading as an unknown part, which the gRPC
  routes refused as an unsupported part type and the HTTP relay passed through.
- `MessageContent`: a part's refusal reaches the request's error under the part's place
  (`` missing field `content[1].image_url.url` ``, `` ..., in `content[1].image_url` ``) instead of
  the untagged read's generic words (`data did not match any variant of untagged enum
  MessageContent`), so `error.code` and `error.param` are the field's.
- `ValidatedJson` refuses a body under a `Content-Encoding` the gateway does not decode (any but
  `identity`, in whichever field of a repeated header or position of a comma-separated list it
  stands) before it reads the body, as a `415` in the gateway's JSON error envelope (`error.code`
  `json_parse_error`, the encoding named in the message, at most 64 characters of it, a longer
  one cut and marked with `...`) with `Accept-Encoding: identity`; it parsed the compressed bytes
  as JSON and refused them as a `400` with a syntax error. `undecodable_content_encoding`,
  `content_encoding_message` and `with_accept_encoding_identity`, behind the `axum` feature, are
  the header's reading, the message and the answer's header, shared with the gateway's
  control-plane JSON extractor, which answers the same.
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

- A function declared as a member of a Responses namespace is held to the two rules a top-level
  function is held to. `parameters` that are not a JSON object are refused (`400`, `error.code`
  `invalid_type`, `Invalid type for 'tools[0].tools[1].parameters': expected an object, but got a
  string instead.`, `error.param` `tools`). For one of OpenAI's own models a strict member whose
  schema does not pin `additionalProperties: false` on every object is refused
  (`invalid_function_parameters`, the message naming the member's function), as a top-level strict
  function is. Such a member passed validation, and the model was handed a function whose schema
  was a string, an array, a number or a boolean. A member with an object, with `null` or with no
  `parameters` is accepted as before, and a self-hosted model's strict member is passed through as
  before.
- A tool the model calls by name is refused when that name is the empty string (`400`,
  `error.code` `empty_string`, in the public API's words: `Invalid 'tools[1].function.name': empty
  string. Expected a string with minimum length 1, but got an empty string instead.`). On Chat
  Completions that is a function (`error.param` the field; a function of the deprecated
  `functions` list at the place its migration gives it), and the rule stands ahead of the
  `tool_choice` rule, so a choice that rule refuses does not hide the unnamed tool. On Responses it
  is a function, a custom tool, a namespace and a namespace's member (`tools[1].name`,
  `tools[1].tools[0].name` in the message, `error.param` `tools`), and on Interactions a function.
  The place counts every tool. Such a request passed validation, and the model could be made to
  call a tool no client can dispatch.
- A Messages request's Anthropic-defined tools keep their configuration when the request is
  written back for an upstream: a web search or web fetch tool's domain filters, use cap and
  location, a computer tool's display size, a text editor's `max_characters`, and any tool's
  `defer_loading`, `strict` or `input_examples`. The untagged `Tool` reads every tool with a
  `type` and a `name` and no `input_schema` as `ToolSearch`, which kept only those two keys,
  `allowed_callers` and `cache_control`, so the provider route dropped the rest. Breaking:
  `ToolSearchTool` gains `extra: Map<String, Value>`.
- A reasoning output item without `content`, as the public API writes it (`id`, `summary` and
  `encrypted_content`), parses, with no content. The crate refused the item, and any Response
  carrying it, with ``missing field `content` ``. An item the gateway writes is unchanged.
- An embeddings answer whose usage has only `prompt_tokens` and `total_tokens`, as the public API
  writes it, parses, with no completion tokens. The crate refused the whole answer with
  ``missing field `completion_tokens` ``. The rerank and classify answers share `UsageInfo` and
  read the same way; a usage the gateway writes is unchanged.
- A Messages document's content source takes its content as a string, and the request's
  `container` takes the id of a container as a string, as the public API takes both beside their
  other forms. Each string was refused with the whole request. On the gRPC router a string
  content source renders as the document's text, as a text source does. Breaking:
  `DocumentSource::Content` holds the new `DocumentContent`, and `CreateMessageRequest.container`
  is the new `MessagesContainer`.
- `PromptCacheRetention::InMemory` reads and writes `in_memory`, as the public API spells it
  (`PromptCacheRetentionEnum`), and still reads `in-memory`. The crate refused a Responses
  request carrying `in_memory` as an unknown variant, and wrote `in-memory`, which the public
  API does not list, in the requests and responses it serialized.
- `StringOrContentParts::to_text_only` names the part it cannot carry: a function output with an
  `input_image`, an `input_file` or a `refusal` part is refused as `Unsupported content part type
  "input_image" in a function output for text-only backends: only text parts are carried as a tool
  result`, where every such part answered the one fixed message `Multimodal function output is not
  supported by this text-only backend`. The gateway's `400 unsupported_content_part` for a function
  output now names the part, as the same route's refusal of a message part does; the code and the
  status are unchanged.
  The message reaches a client on both routes that read it: the chat-family Responses conversion
  (`400 unsupported_content_part`) and the Harmony builder's tool message
  (`400 harmony_build_failed`).
- A Response linked to a conversation writes it as the public API does, `{"id": "conv_..."}`
  (`ResponseConversation`), where the crate wrote the bare id, and reads either shape: the crate
  refused a Response carrying the object with ``invalid type: map, expected a string``. The field
  keeps the id alone (`Option<String>`), so code that reads it is unchanged, and its schema takes
  both shapes, so a client generated from it also reads a response stored with the bare id.
- The generate route's request rule refuses an empty `text` under `prompt_empty` only when the body
  has no `input_ids` and no media field carries an item (beside ids the engine serves the ids without
  reading the text; a media-only prompt is sent with an empty `text`, as the engine reads it);
  a body with two prompts (two of `text`, `input_ids` and `input_embeds`) is served, the engine
  taking the ids over the text (and the embeds only without a text), and the gateway's gRPC
  generate path now reads the ids before the text too, where it read the text first and dropped
  the ids without a word. The
  request's media read (`GenerateRequest::media_fields`, `carries_media`) is one definition here,
  read by the gRPC path's media refusal as well.
- A built-in `range` or `length` rule that carries no message of its own answers a sentence in the
  routes' form (`Invalid 'dimensions': integer below minimum value. Expected an integer >= 1, but got 0
  instead.`) instead of the validation library's rendering of the rule's name and parameters
  (`Validation error: range [{"min": Number(1), "value": Number(0)}]`); the code and param are
  unchanged. Every such rule of the crate (`dimensions`, the sampling parameters, ...) answers this way.
- A Responses input file part with `detail: "auto"`, the public API's default (`FileInputDetail`),
  parses, and a prompt's file variable keeps its `detail` when the request is written back for an
  upstream. The crate refused the request with ``unknown variant `auto`, expected `low` or
  `high` `` and dropped the variable's detail. Breaking: `FileDetail` gains `Auto`, now its default,
  and `PromptVariableTyped::ResponseInputFile` gains `detail: Option<FileDetail>`.
- Realtime events read and write as the public API does. A `session.update`, `session.created`
  or `session.updated` event reads its session into the variant its `type` names, where every
  one the API writes failed with "missing field `type`", and is written with that `type` once,
  where it was written twice. A transcription's `completed` event keeps its detected `languages`
  (an empty list means none was detected), a transcription segment its `start` and `end` at the
  precision written, and a failed response's error its `message`. Breaking:
  `ServerEvent::InputAudioTranscriptionCompleted` gains `languages`, the segment's `start` and
  `end` are `f64`, and `ResponseStatusError` gains `message`.
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
- A web search call parses as the public API writes it: a `find_in_page` action (the crate named
  it `find`, which still reads and is now written `find_in_page`), an `open_page` action without
  a URL or with a null one, and the `incomplete` status. The crate refused each as an unknown
  variant or a missing field. Breaking: `WebSearchCallStatus` gains `Incomplete`, and
  `WebSearchAction::OpenPage.url` is `Option<String>`.
- A completion choice without log probabilities is written with `"logprobs": null`, in a buffered
  body and in each chunk of a stream, as the public API sends it and a chat choice is written. The
  key was left out, so a client generated from the public schema, which marks it required, read a
  choice it does not expect. A choice with log probabilities is written as before.
- A Responses `tool_choice` of type `computer` or `computer_use`, as the public API lists them
  (`ToolChoiceTypes`), parses. The crate refused the request whole, as matching no variant of
  `ResponsesToolChoice`. Breaking: `BuiltInToolChoiceType` gains `Computer` and `ComputerUse`.
- A Messages web search result is written with its `type` (`web_search_result`) and its null
  `page_age`, and the web search error `request_too_large` and a tool search's error result
  (`tool_search_tool_result_error`) read, as the public API writes them. A result had been written
  without the `type` that the public API requires of a replayed one, and each of the two errors
  failed a whole answer or request that held it. A tool reference in a replayed tool search result
  keeps its `cache_control`, which was dropped. Breaking: `WebSearchResultBlock` gains
  `block_type`, `WebSearchToolResultErrorCode` gains `RequestTooLarge`, a tool search result's
  `content` is the new `ToolSearchToolResultContent`, and `ToolReferenceBlock` gains
  `cache_control`.
- A completion request with `"echo": null` is served, as the public API allows. The crate refused
  it as a 400 `invalid_type`. The engine flags `no_stop_trim`, `ignore_eos`, `skip_special_tokens`
  and `return_hidden_states` read a `null` as their default too, as the chat request's flags do.
- A replayed `computer_call_output` input item with `acknowledged_safety_checks: null`, which the
  public API allows (`ComputerCallOutputItemParam`), parses as none acknowledged. The crate refused
  the request whole, as matching no variant of `ResponseInput`.
- `EmbeddingRequest`, `ClassifyRequest`, `TokenizeRequest`, `DetokenizeRequest` and
  `V1RerankReqInput` implement `Validate` and `Normalizable`, so the gateway reads
  `/v1/embeddings`, `/v1/classify`, `/v1/tokenize`, `/v1/detokenize` and `/v1/rerank` through its
  validated extractor: a body that does not parse, a body of the wrong type and a failed rule
  answer the gateway's JSON error envelope (`400`, with `error.code` and `error.param`) where
  axum's own text/plain `422`, `415` and `400` answered before. `V1RerankReqInput` refuses an
  empty `query` and an empty `documents` list, the rules of the native route's `RerankRequest`;
  its `top_n` keeps reading `0` as all documents. `TokenizeRequest` and `DetokenizeRequest`
  declare no rule; the rules `EmbeddingRequest` and `ClassifyRequest` declare follow.
- The `/v1/audio/transcriptions` relay forwards the form's text fields the gateway's types do not
  model: the public API's `include[]`, `chunking_strategy` (and the SDK's bracketed
  `chunking_strategy[...]` keys), `known_speaker_names[]`, `known_speaker_references[]` and the
  next field the API adds. The extractor drained them and the HTTP router rebuilt the worker's
  form from the typed fields alone, so a transcription answered without the logprobs, speaker
  names or chunking the client asked for. `TranscriptionRequest.other` keeps them in order under
  the names they came with, and the router appends them to the worker's form; a file part under
  another name is still drained. The gRPC pipeline builds its request from the typed fields and
  does not read them. Breaking: `TranscriptionRequest` gains `other`.
- The preview web search tool takes the public API's dated type, `web_search_preview_2025_03_11`,
  written back under the canonical `web_search_preview`, and its `search_content_types`. The crate
  refused a request declaring either, as an unknown tool type or field. Breaking:
  `WebSearchPreviewTool` gains `search_content_types: Option<Vec<String>>`.
- An Interactions request declaring a `computer_use`, `google_maps` or `retrieval` tool is
  accepted, an `mcp_server` tool takes its `allowed_tools` as the list the public API declares, and
  `google_search` keeps its `search_types`. The three tools were unknown variants and the list
  failed to parse, each refusing the whole request, and `search_types` was dropped. Breaking:
  `InteractionsTool` gains `ComputerUse`, `GoogleMaps` and `Retrieval`, `GoogleSearch` gains
  `search_types`, `McpServer.allowed_tools` is a `Vec<AllowedTools>`, and `ToolChoiceType`, which
  the SDK keeps open as it does the tools' other sets, gains `Other` and is `#[non_exhaustive]`.
- Interactions content keeps the fields the public API added: an audio part's `channels` and
  `sample_rate` (and an audio delta's), a video part's `processing` and `name`, and a text
  annotation's `type` and that kind's fields (a URL citation's `url` and `title`, a file
  citation's `file_name` and `page_number`, ...). Each was dropped on the way to the upstream.
  Breaking: `Content::Audio`, `Delta::Audio` and `Content::Video` gain the fields, and
  `Annotation` gains `other`.
- `EmbeddingRequest` and `ClassifyRequest` run the public API's shape rule on `input` (a string,
  an array of strings, an array of token ids or an array of token id arrays; nothing empty; a
  member of another kind than the first is refused by its index), and `EmbeddingRequest` the
  API's bound on `dimensions` (at least one), so the gateway refuses the shapes the API refuses
  in its error envelope with the field in `error.param` (`empty_array`, `empty_input`,
  `invalid_type`, `integer_below_min_value`) before a worker sees them, on the gRPC path and the
  HTTP relay alike. `validate_input` is the classify route's rule and the base of the embeddings
  route's (`validate_embedding_input`), which also takes the serving engine's array of objects, a
  structured member each, forwarded with no field of them read. The rule reads shape only: a
  batch's length, the `encoding_format` and the objects' fields are the serving engine's to bound.
- A `ResponsesResponse` whose `metadata` is `null`, or whose `incomplete_details.reason` is
  `max_messages` or `steered`, as the public API writes them, parses; it refused all three.
  `null` metadata reads as the empty map (`deserialize_null_as_empty_map`, a map twin of
  `deserialize_null_as_empty_vec`), and `IncompleteReason`, `#[non_exhaustive]`, gains
  `MaxMessages` and `Steered`.
- A Messages search result's text items are written with their `type`, which the public API
  requires of them, in a message and in a tool result alike. They were read into untagged text
  blocks, so a forwarded search result lost the `type` of each item. An item read without one
  still reads, as text, and an item of another type is refused. Breaking:
  `SearchResultBlock.content` holds the new `SearchResultTextBlock`, whose type is
  `SearchResultTextType`.
- A `json_schema` response format keeps its `description` when the request is written back for an
  upstream, as the HTTP router and the provider routers write it. The crate dropped the key while
  reading the format. The gRPC path's Responses-to-Chat conversion carries a Responses
  `text.format` description into the Chat request it builds, but a gRPC engine still gets only the
  format's schema, as before. Breaking: `JsonSchemaFormat` gains `description: Option<String>`.
- A Responses request's reasoning config keeps `mode`, `context` and the deprecated
  `generate_summary`, as the public API's `Reasoning` takes them, when the request is written back
  for an upstream, and a Response's `reasoning` echoes the ones the request set. The crate dropped
  all three, so a `pro`-mode request reached the upstream in its default mode. Breaking:
  `ResponseReasoningParam` gains `mode: Option<String>`, `context: Option<ReasoningContext>` and
  `generate_summary: Option<ReasoningSummary>`, `ReasoningInfo` gains the three as
  `Option<String>`, and the crate gains `ReasoningContext`.
- A usage object whose `prompt_tokens_details` leaves out `cached_tokens` parses, with no tokens
  cached, as the public API defaults the key. The crate refused the whole usage, and the
  completion around it, with ``missing field `cached_tokens` ``. The newer counters an upstream
  reports are kept: `text_tokens`, `image_tokens` and `cache_write_tokens` in the prompt details,
  `text_tokens` in the completion details. A usage the gateway builds is written as before.
  Breaking: `PromptTokenUsageInfo` gains `text_tokens`, `image_tokens` and `cache_write_tokens`,
  and `CompletionTokensDetails` gains `text_tokens`, all `Option<u32>`.
- A Responses function tool declared with `async`, `defer_loading`, `output_schema` or
  `allowed_callers`, and a custom tool declared with `async` or `allowed_callers`, as the public
  API's tools take them, parse and are forwarded as given. The crate refused each as an unknown
  field. The function tool's four keys are validated for shape as the custom tool's typed fields
  are (`invalid_type` otherwise). `FunctionTool::chat_function` leaves the four keys out of the
  copy a chat template renders, and `FunctionTool::allows_direct_calls` and
  `CustomTool::allows_direct_calls` say whether the model may call a tool itself. Breaking:
  `CustomTool` gains `r#async: Option<bool>` and `allowed_callers: Option<Vec<String>>`.
- An Interactions request keeps the fields the gateway does not model (`continuation_token`,
  `safety_settings`, `service_tier`, `webhook_config`, `environment`, `labels`, `cached_content`,
  and any the API adds), forwarded to the upstream as written, as the Messages request's `other`
  keeps its. The Gemini router rebuilds the upstream body from the typed request, which dropped
  them: the safety settings were lost, and a `store: false` client's continuation hop reached the
  upstream without its token, so each hop started the answer over. A modeled field also reads its
  camelCase spelling (`previousInteractionId`, `systemInstruction`, `generationConfig`, ...) into
  the typed field, where the gateway's checks see it, and a body with both spellings is refused.
  So does a key of the generation and agent configs (`maxOutputTokens`, `stopSequences`,
  `thinkingSummaries`, ...), which goes upstream in snake_case with its value.
  Breaking: `InteractionsRequest` gains `other: Map<String, Value>`.
- A Messages image or document whose source is a file uploaded through the Files API
  (`{"type": "file", "file_id": ...}`) is accepted and forwarded as written, as the public API
  takes it. The source was an unknown variant, which refused the whole request. The gRPC router,
  which has no Files API to read the file from, refuses such an image naming the block wherever it
  appears, and such a document at a message's top level; a document inside a tool result is left
  out of its prompt, as every document there is today. Breaking: `ImageSource` and
  `DocumentSource` gain `File`.
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
- A Realtime `response.create`'s input may reference an item already in the session,
  `{"type": "item_reference", "id": ...}`, as the public API takes it: such an input was refused as
  an unknown conversation item variant. `RealtimeResponseCreateParams.input` holds the new
  `RealtimeResponseInputItem`, a conversation item or a `RealtimeItemReference`; a reference is
  taken there only, as the API takes it, not as a new conversation item. Breaking: `input` changes
  type.
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
- A Messages answer with a content block of a kind this build does not read (a web fetch, code
  execution, bash or text editor result, a container upload, or any block the public API adds)
  parses, the block kept as written in `ContentBlock::Other`, as `ToolUseCaller::Other` keeps an
  unknown caller. A whole answer with such a block failed: the provider route answered a 502, and
  a streamed thread turn was left unstored. Breaking: `ContentBlock` gains `Other(Value)`.
- `EmbeddingRequest` and `ClassifyRequest` keep the fields they do not model in a flattened
  `other` map, as `ChatCompletionRequest` and `CompletionRequest` do, so the HTTP relay forwards an
  embeddings or classify body as the client wrote it (`truncate_prompt_tokens`,
  `add_special_tokens`, `normalize`, `priority`, `extra_body`, ...). The relay re-serializes the
  typed body and dropped every key the types did not declare. Breaking: the two structs gain the
  field `other: serde_json::Map<String, Value>`.
- A Messages conversation that replays the public API's server tool results (`web_fetch_tool_result`,
  `code_execution_tool_result`, `bash_code_execution_tool_result`,
  `text_editor_code_execution_tool_result`) or sends a `container_upload` block is accepted, and the
  blocks are forwarded as written. Each was an unknown variant of `InputContentBlock`, which refused
  the whole request, a token count included. Breaking: `InputContentBlock` gains the five variants,
  and the crate gains `ServerToolResultBlock` and `ContainerUploadParam`.
