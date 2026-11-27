use serde_json::json;

use super::*;

fn usage(input: u32, output: u32) -> UsageSettlement {
    UsageSettlement {
        actual_input_tokens: input,
        completion_tokens: output,
        ..UsageSettlement::default()
    }
}

fn parse(route: &str, body: Value) -> Option<UsageSettlement> {
    usage_from_body(route, 1, body.to_string().as_bytes())
}

fn parse_generate(samples_per_prompt: u32, body: Value) -> Option<UsageSettlement> {
    usage_from_body("/generate", samples_per_prompt, body.to_string().as_bytes())
}

#[test]
fn each_routes_usage_shape_is_read() {
    let openai = json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}});
    for route in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/classify",
    ] {
        assert_eq!(parse(route, openai.clone()), Some(usage(7, 3)), "{route}");
    }
    let anthropic = json!({"usage": {"input_tokens": 5, "output_tokens": 2}});
    for route in ["/v1/messages", "/v1/responses"] {
        assert_eq!(
            parse(route, anthropic.clone()),
            Some(usage(5, 2)),
            "{route}"
        );
    }
    let interactions = json!({"usage": {
        "total_input_tokens": 6, "total_output_tokens": 1, "total_tokens": 7,
    }});
    assert_eq!(parse("/v1/interactions", interactions), Some(usage(6, 1)));
    // A thinking model's thought tokens are generated too: charged with
    // the output and kept as the reasoning share; its tool-use prompts
    // are input. An answer that is all thought leaves the output count
    // out: zero, not unreadable.
    let thinking = json!({"usage": {
        "total_input_tokens": 6, "total_output_tokens": 1, "total_thought_tokens": 40,
        "total_tool_use_tokens": 3, "total_tokens": 50,
    }});
    assert_eq!(
        parse("/v1/interactions", thinking),
        Some(UsageSettlement {
            reasoning_tokens: 40,
            ..usage(9, 41)
        })
    );
    let all_thought = json!({"usage": {
        "total_input_tokens": 6, "total_thought_tokens": 40, "total_tokens": 46,
    }});
    assert_eq!(
        parse("/v1/interactions", all_thought),
        Some(UsageSettlement {
            reasoning_tokens: 40,
            ..usage(6, 40)
        })
    );
    assert_eq!(
        parse(
            "/generate",
            json!({"text": "x", "meta_info": {"prompt_tokens": 4, "completion_tokens": 9}})
        ),
        Some(usage(4, 9))
    );
}

/// The usage embeddings and classify report generates nothing: their
/// `prompt_tokens` alone settles, where a generating route's would not.
/// Anthropic's `input_tokens` leave out the cached prefix, read or
/// written, which the prompt adds back; Responses' include theirs.
#[test]
fn a_messages_prompt_counts_its_cache_reads_and_writes() {
    let cached = json!({"usage": {
        "input_tokens": 50,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 50,
        "output_tokens": 7,
    }});
    assert_eq!(parse("/v1/messages", cached), Some(usage(1000, 7)));

    let read_only = json!({"usage": {
        "input_tokens": 50,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": null,
        "output_tokens": 7,
    }});
    assert_eq!(parse("/v1/messages", read_only), Some(usage(950, 7)));

    let responses = json!({"usage": {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 900},
        "output_tokens": 7,
        "total_tokens": 1007,
    }});
    assert_eq!(parse("/v1/responses", responses), Some(usage(1000, 7)));
}

#[test]
fn embeddings_and_classify_settle_their_prompt_tokens_alone() {
    for usage_object in [
        json!({"prompt_tokens": 8, "total_tokens": 8}),
        json!({"prompt_tokens": 8, "completion_tokens": null, "total_tokens": 8}),
    ] {
        let body = json!({"usage": usage_object});
        for route in ["/v1/embeddings", "/v1/classify"] {
            assert_eq!(parse(route, body.clone()), Some(usage(8, 0)), "{route}");
        }
        assert_eq!(parse("/v1/chat/completions", body), None);
    }
}

/// Every entry of an `n=1` batch is a prompt of its own: the prompts sum
/// with the completions.
#[test]
fn a_generate_batch_sums_its_prompts_and_its_completions() {
    let batch = json!([
        {"meta_info": {"prompt_tokens": 2, "completion_tokens": 9}},
        {"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}},
    ]);
    assert_eq!(parse_generate(1, batch), Some(usage(6, 10)));
}

/// The `n` samples of one prompt repeat its count: it is counted once,
/// per prompt of a batch, and every sample's completion sums.
#[test]
fn the_samples_of_a_prompt_count_it_once() {
    let one_prompt = json!([
        {"meta_info": {"prompt_tokens": 4, "completion_tokens": 9}},
        {"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}},
    ]);
    assert_eq!(parse_generate(2, one_prompt), Some(usage(4, 10)));

    let two_prompts = json!([
        {"meta_info": {"prompt_tokens": 2, "completion_tokens": 1}},
        {"meta_info": {"prompt_tokens": 2, "completion_tokens": 2}},
        {"meta_info": {"prompt_tokens": 5, "completion_tokens": 3}},
        {"meta_info": {"prompt_tokens": 5, "completion_tokens": 4}},
    ]);
    assert_eq!(parse_generate(2, two_prompts), Some(usage(7, 10)));
}

/// An answer that does not hold `n` entries to each prompt is not one
/// this request asked for.
#[test]
fn a_generate_answer_short_of_its_samples_is_none() {
    let entry = json!({"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}});
    assert_eq!(parse_generate(2, entry.clone()), None);
    assert_eq!(
        parse_generate(2, json!([entry.clone(), entry.clone(), entry])),
        None
    );
}

#[test]
fn usage_that_cannot_be_read_whole_is_none() {
    for (route, body) in [
        ("/v1/chat/completions", json!({"choices": []})),
        ("/v1/chat/completions", json!({"usage": null})),
        (
            "/v1/chat/completions",
            json!({"usage": {"prompt_tokens": 7}}),
        ),
        (
            "/v1/chat/completions",
            json!({"usage": {"prompt_tokens": -1, "completion_tokens": 3}}),
        ),
        (
            "/v1/messages",
            json!({"usage": {"input_tokens": "5", "output_tokens": 2}}),
        ),
        (
            "/v1/messages",
            json!({"usage": {"input_tokens": 5, "cache_read_input_tokens": "9", "output_tokens": 2}}),
        ),
        (
            "/v1/messages",
            json!({"usage": {"input_tokens": u32::MAX, "cache_creation_input_tokens": 1, "output_tokens": 2}}),
        ),
        (
            "/v1/completions",
            json!({"usage": {"prompt_tokens": u64::from(u32::MAX) + 1, "completion_tokens": 0}}),
        ),
        ("/v1/embeddings", json!({"usage": {"total_tokens": 8}})),
        (
            "/v1/embeddings",
            json!({"usage": {"prompt_tokens": 8, "completion_tokens": "0"}}),
        ),
        ("/v1/classify", json!({"usage": {"prompt_tokens": -8}})),
        (
            "/generate",
            json!({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
        ),
        ("/generate", json!([])),
        (
            "/generate",
            json!([{"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}}, {}]),
        ),
        (
            "/generate",
            json!([
                {"meta_info": {"prompt_tokens": 1, "completion_tokens": u32::MAX}},
                {"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}},
            ]),
        ),
        (
            "/generate",
            json!([
                {"meta_info": {"prompt_tokens": u32::MAX, "completion_tokens": 1}},
                {"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}},
            ]),
        ),
    ] {
        assert_eq!(parse(route, body.clone()), None, "{route}: {body}");
    }
    assert_eq!(
        usage_from_body("/v1/chat/completions", 1, b"not json"),
        None
    );
}

const TEN_WORDS: &str = "one two three four five six seven eight nine ten";

fn request<T: serde::de::DeserializeOwned>(body: Value) -> T {
    serde_json::from_value(body).unwrap()
}

/// Text is estimated at 1.3 tokens per word, rounded up; a pre-tokenized
/// request at its own ids, whatever text routing saw.
#[test]
fn the_estimate_is_the_requests_own_ids_or_its_words() {
    let text: GenerateRequest =
        request(json!({"model": "m", "text": "one two three four five six seven"}));
    assert_eq!(text.estimated_input_tokens(None), 10);

    let ids: GenerateRequest = request(json!({"model": "m", "input_ids": [1, 2, 3, 4, 5]}));
    assert_eq!(ids.estimated_input_tokens(None), 5);

    let chat: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "one two three four five six seven"}],
    }));
    assert_eq!(chat.estimated_input_tokens(Some("one two three")), 4);
    assert_eq!(chat.estimated_input_tokens(Some("")), 0);
}

/// A completion prompt of ids reserves its count and a batch the sum of
/// its lists, whatever text routing saw; a text prompt its words.
#[test]
fn a_completion_token_prompt_reserves_its_ids() {
    let ids: CompletionRequest = request(json!({"model": "m", "prompt": [1, 2, 3, 4, 5]}));
    assert_eq!(ids.estimated_input_tokens(Some("1 2 3 4 5")), 5);
    assert_eq!(ids.estimated_input_tokens(None), 5);

    let batch: CompletionRequest = request(json!({"model": "m", "prompt": [[1, 2], [3, 4, 5, 6]]}));
    assert_eq!(batch.estimated_input_tokens(None), 6);

    let text: CompletionRequest =
        request(json!({"model": "m", "prompt": "one two three four five six seven"}));
    assert_eq!(text.estimated_input_tokens(None), 10);
    let texts: CompletionRequest = request(json!({"model": "m", "prompt": ["one two", "three"]}));
    assert_eq!(texts.estimated_input_tokens(Some("one two three")), 4);
}

/// Routing keys on the first sequence of a batch; the estimate counts
/// them all.
#[test]
fn a_generate_batch_reserves_every_sequence() {
    let batch: GenerateRequest =
        request(json!({"model": "m", "input_ids": [[1, 2], [3, 4, 5, 6]]}));
    assert_eq!(batch.routing_tokens().map(<[i32]>::len), Some(2));
    assert_eq!(batch.estimated_input_tokens(None), 6);

    let leading_empty: GenerateRequest =
        request(json!({"model": "m", "input_ids": [[], [3, 4, 5]]}));
    assert_eq!(leading_empty.estimated_input_tokens(Some("3 4 5")), 3);
}

#[test]
fn a_generate_request_has_its_n_samples_per_prompt() {
    for (sampling_params, samples) in [
        (json!(null), 1),
        (json!({}), 1),
        (json!({"n": 3}), 3),
        (json!({"n": 0}), 1),
    ] {
        let generate: GenerateRequest = request(
            json!({"model": "m", "text": "hi", "sampling_params": sampling_params.clone()}),
        );
        assert_eq!(generate.samples_per_prompt(), samples, "{sampling_params}");
    }
    let chat: ChatCompletionRequest = request(json!({
        "model": "m", "n": 3, "messages": [{"role": "user", "content": "hi"}],
    }));
    assert_eq!(chat.samples_per_prompt(), 1);
}

/// Embeddings and classify take text or token ids, one input or several:
/// each id is a token, each word 1.3.
#[test]
fn embeddings_and_classify_reserve_each_input_shape() {
    for (input, tokens) in [
        (json!("one two three four five six seven"), 10),
        (json!(["one two", "three"]), 4),
        (json!([1, 2, 3, 4]), 4),
        (json!([[1, 2], [3]]), 3),
        (json!([]), 0),
    ] {
        let embedding: EmbeddingRequest = request(json!({"model": "m", "input": input}));
        assert_eq!(embedding.estimated_input_tokens(None), tokens, "{input}");
        let classify: ClassifyRequest = request(json!({"model": "m", "input": input}));
        assert_eq!(classify.estimated_input_tokens(None), tokens, "{input}");
    }
}

/// The instructions and the function calls and their outputs count with
/// the messages routing reads.
#[test]
fn a_responses_request_reserves_its_instructions_and_function_calls() {
    let instructed: ResponsesRequest = request(json!({
        "model": "m",
        "instructions": TEN_WORDS,
        "input": "hello world",
    }));
    assert_eq!(instructed.estimated_input_tokens(Some("hello world")), 16);

    let tool_turn: ResponsesRequest = request(json!({
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "weather in Paris"},
            {"type": "function_call", "call_id": "c1", "name": "weather",
             "arguments": "{\"city\": \"Paris\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny and warm"},
        ],
    }));
    assert_eq!(tool_turn.extract_text_for_routing(), "weather in Paris");
    assert_eq!(tool_turn.estimated_input_tokens(None), 11);
}

#[test]
fn an_interactions_request_reserves_its_system_instruction() {
    let instructed: InteractionsRequest = request(json!({
        "model": "m",
        "system_instruction": TEN_WORDS,
        "input": "hello world",
    }));
    assert_eq!(instructed.extract_text_for_routing(), "hello world");
    assert_eq!(instructed.estimated_input_tokens(None), 16);
}

/// A custom tool's input and its output, a string or text parts.
#[test]
fn a_responses_request_reserves_its_custom_tool_calls() {
    for output in [
        json!(TEN_WORDS),
        json!([{"type": "input_text", "text": TEN_WORDS}]),
    ] {
        let custom: ResponsesRequest = request(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hello world"},
                {"type": "custom_tool_call", "call_id": "c1", "name": "grep", "input": TEN_WORDS},
                {"type": "custom_tool_call_output", "call_id": "c1", "output": output},
            ],
        }));
        assert_eq!(custom.extract_text_for_routing(), "hello world");
        // 2 + 10 + 10 words: ceil(22 * 1.3).
        assert_eq!(custom.estimated_input_tokens(None), 29);
    }
}

/// The arguments of the assistant's tool calls, whatever text routing saw.
#[test]
fn a_chat_history_reserves_its_tool_call_arguments() {
    let chat: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": "{\"query\": \"one two three four five six seven eight nine\"}",
                },
            }]},
        ],
    }));
    assert_eq!(chat.extract_text_for_routing(), "weather in Paris");
    // 3 words of text and 10 of arguments: ceil(13 * 1.3).
    assert_eq!(chat.estimated_input_tokens(None), 17);
    assert_eq!(chat.estimated_input_tokens(Some("weather in Paris")), 17);
}

/// A refused turn's refusal and the arguments of a deprecated function
/// call, which the upstream reads with the rest of the history.
#[test]
fn a_chat_history_reserves_its_refusals_and_function_call_arguments() {
    let chat: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": null, "refusal": TEN_WORDS},
            {"role": "assistant", "content": null, "function_call": {
                "name": "weather",
                "arguments": "{\"query\": \"one two three four five six seven eight nine\"}",
            }},
        ],
    }));
    assert_eq!(chat.extract_text_for_routing(), "weather in Paris");
    // 3 words of text, 10 of refusal and 10 of arguments: ceil(23 * 1.3).
    assert_eq!(chat.estimated_input_tokens(None), 30);
    assert_eq!(chat.estimated_input_tokens(Some("weather in Paris")), 30);
}

/// A call's arguments count by the words of their JSON, as a tool use's
/// input does: written without spaces, as clients send them and as an
/// object-form `arguments` is kept, they are not one word. Arguments that
/// are not JSON, a call cut off mid-way among them, count the runs between
/// spaces and JSON punctuation.
#[test]
fn a_call_s_compact_json_arguments_reserve_their_words() {
    let compact = "{\"city\":\"Paris\",\"days\":3,\"units\":\"metric\"}";
    let chat: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "weather", "arguments": compact},
            }]},
            {"role": "assistant", "content": null, "function_call": {
                "name": "weather",
                "arguments": {"city": "Paris", "days": 3, "units": "metric"},
            }},
        ],
    }));
    assert_eq!(chat.extract_text_for_routing(), "weather in Paris");
    // 3 words of text and 6 of each call's arguments (three keys, two
    // words and a number): ceil(15 * 1.3).
    assert_eq!(chat.estimated_input_tokens(None), 20);

    let responses: ResponsesRequest = request(json!({
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "weather in Paris"},
            {"type": "function_call", "call_id": "c1", "name": "weather",
             "arguments": compact},
        ],
    }));
    assert_eq!(responses.extract_text_for_routing(), "weather in Paris");
    // 3 words of text and 6 of arguments: ceil(9 * 1.3).
    assert_eq!(responses.estimated_input_tokens(None), 12);

    let text: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "weather", "arguments": "city Paris, three days"},
            }]},
        ],
    }));
    // Not JSON: 3 words of text and 4 of arguments: ceil(7 * 1.3).
    assert_eq!(text.estimated_input_tokens(None), 10);

    let cut_off: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Paris\",\"days\":3,\"uni"},
            }]},
        ],
    }));
    // Cut off mid-way, so not JSON: 3 words of text and 5 of arguments
    // (`city`, `Paris`, `days`, `3`, `uni`): ceil(8 * 1.3).
    assert_eq!(cut_off.estimated_input_tokens(None), 11);
}

/// A tool result's text, a string or text blocks, and a tool use's input.
#[test]
fn a_messages_history_reserves_its_tool_uses_and_results() {
    for content in [
        json!(TEN_WORDS),
        json!([{"type": "text", "text": TEN_WORDS}]),
    ] {
        let result: CreateMessageRequest = request(json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello world"},
                {"type": "tool_result", "tool_use_id": "t1", "content": content},
            ]}],
        }));
        assert_eq!(result.extract_text_for_routing(), "hello world");
        // 2 + 10 words: ceil(12 * 1.3).
        assert_eq!(result.estimated_input_tokens(None), 16);
    }

    let tool_use: CreateMessageRequest = request(json!({
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "weather in Paris"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "weather",
                 "input": {"city": "Paris", "days": 3}},
            ]},
        ],
    }));
    assert_eq!(tool_use.extract_text_for_routing(), "weather in Paris");
    // 3 words of text; two keys, a word and a number of input: ceil(7 * 1.3).
    assert_eq!(tool_use.estimated_input_tokens(None), 10);
}

/// Routing on the client's tokens hint extracts no text; the estimate
/// reads the body's own, not the hint's length.
#[test]
fn a_routing_hint_does_not_set_the_estimate() {
    let chat: ChatCompletionRequest = request(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "one two three four five six seven"}],
    }));
    assert_eq!(chat.estimated_input_tokens(None), 10);
}

/// The estimate of a chat request counts the words of its prompt alone,
/// whichever text routing hands it. The routing text carries a marker at
/// the place of a part that is no text, and counted as it stands it
/// would charge the marker as a word: no kind of such a part does, in a
/// user's content or in an assistant's.
#[test]
fn a_chat_estimate_charges_no_marker_of_a_part_that_is_no_text() {
    let text = json!({"type": "text", "text": "what does it show"});
    for (part, marked) in [
        (
            json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
            true,
        ),
        (
            json!({"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}),
            true,
        ),
        (
            json!({"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}),
            true,
        ),
        (
            json!({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}),
            true,
        ),
        (json!({"type": "file", "file": {"file_id": "file-1"}}), true),
        (json!({"type": "refusal", "refusal": "no"}), false),
    ] {
        for messages in [
            json!([{"role": "user", "content": [text, part, text]}]),
            json!([
                {"role": "user", "content": "what does it show"},
                {"role": "assistant", "content": [text, part]},
            ]),
        ] {
            let chat: ChatCompletionRequest = request(json!({"model": "m", "messages": messages}));
            let routed = chat.routing_text();
            let own = chat.estimated_input_tokens(None);
            assert_eq!(
                routed != chat.extract_text_for_routing(),
                marked,
                "{routed}"
            );
            assert_eq!(
                tokens_for_words(word_count(&routed)) != own,
                marked,
                "{routed}"
            );
            assert_eq!(chat.estimated_input_tokens(Some(&routed)), own, "{routed}");
        }
    }
}
