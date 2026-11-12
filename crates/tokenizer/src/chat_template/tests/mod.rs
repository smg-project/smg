mod hy_v4;
mod render_context;

use super::*;

/// A history call's `arguments` reach a template as the text the request
/// sent only when the template writes them that way: one that concatenates
/// them cannot render the parsed object; one that serializes or indexes
/// the object, or branches on its type, and one without a tool-call branch
/// keep the object.
#[test]
fn a_template_that_concatenates_tool_call_arguments_takes_them_as_text() {
    let state = ChatTemplateState::new(Some(CONCATENATING_CALLS.to_string())).unwrap();
    assert!(state.raw_tool_call_arguments());

    let indexing = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                    <call>{{ t.function.name }}:{% for k, v in t.function.arguments | items %}\
                    {{ k }}={{ v }}{% endfor %}</call>{% endfor %}{% else %}{{ m.content }}\
                    {% endif %}{% endfor %}";
    let branching = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                     <call>{{ t.function.name }}:{% if t.function.arguments is string %}\
                     {{ t.function.arguments }}{% else %}{{ t.function.arguments | tojson }}\
                     {% endif %}</call>{% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
    let without_calls = "{% for m in messages %}{{ m.content }}{% endfor %}";
    for template in [SERIALIZING_CALLS, indexing, branching, without_calls] {
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert!(!state.raw_tool_call_arguments(), "{template}");
    }
    assert!(!ChatTemplateState::empty().raw_tool_call_arguments());
    assert!(!ChatTemplateState::new(None)
        .unwrap()
        .raw_tool_call_arguments());
}

/// A replacement template answers for itself.
#[test]
fn a_replacement_template_refreshes_the_tool_call_arguments_answer() {
    let mut state = ChatTemplateState::new(Some(SERIALIZING_CALLS.to_string())).unwrap();
    assert!(!state.raw_tool_call_arguments());
    state.set(CONCATENATING_CALLS.to_string()).unwrap();
    assert!(state.raw_tool_call_arguments());
    state.set(SERIALIZING_CALLS.to_string()).unwrap();
    assert!(!state.raw_tool_call_arguments());
}

/// A template that writes a history call by string concatenation, as the
/// DeepSeek V3 line's templates do.
const CONCATENATING_CALLS: &str =
    "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
     {{ '<call>' + t.function.name + ':' + t.function.arguments + '</call>' }}\
     {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
/// A template that serializes the arguments object, as most do.
const SERIALIZING_CALLS: &str =
    "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
     <call>{{ t.function.name }}:{{ t.function.arguments | tojson }}</call>\
     {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";

/// Transformers' `{% generation %}` block (the assistant-token mask marker)
/// renders its body; the engine here has no such statement, so the block
/// is rewritten to a plain `if` before parsing, and the content-format
/// detection parses such a template as well.
#[test]
fn generation_blocks_render_their_body() {
    let template = "{%- for m in messages -%}\n{%- if m.role == 'assistant' -%}\n\
                    {% generation %}\n<a>{{ m.content }}</a>\n{%- endgeneration -%}\n\
                    {%- else -%}\n<u>{{ m.content }}</u>\n{%- endif -%}\n{%- endfor -%}";
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages = [
        serde_json::json!({"role": "user", "content": "hi"}),
        serde_json::json!({"role": "assistant", "content": "yo"}),
    ];
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered, "<u>hi</u><a>yo</a>");

    let parts_template = "{% for message in messages %}{% if message['role'] == 'user' %}USER: \
                          {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}\
                          {% generation %}{{ content['text'] + ' ' }}{% endgeneration %}{% endfor %}\
                          {% endif %}{% endfor %}";
    assert_eq!(
        detect_chat_template_content_format(parts_template),
        ChatTemplateContentFormat::OpenAI
    );
}

/// A template that keeps the messages it loops over in a namespace
/// attribute (`ns.rest = messages[1:]`) and renders each message's
/// content through a macro that walks its parts is of the "openai"
/// format: its parts reach the template, which writes the media anchors
/// itself. Any other non-name assignment target still falls back to the
/// "string" format.
#[test]
fn namespace_message_lists_are_followed_to_their_content_loop() {
    let template = "{%- macro render_content(content) -%}\
                    {%- if content is string -%}{{ content }}\
                    {%- else -%}{%- for part in content -%}\
                    {%- if part.get('type', '') in ['image', 'image_url'] -%}<|patch|>\
                    {%- elif part.get('type', '') == 'text' -%}{{ part.get('text', '') }}\
                    {%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}\
                    {%- set ns = namespace(rest=messages) -%}\
                    {%- if messages and messages[0]['role'] == 'system' -%}\
                    {%- set ns.rest = messages[1:] -%}{%- endif -%}\
                    {%- for message in ns.rest -%}\
                    <|start|>{{ message['role'] }}<|message|>{{ render_content(message['content']) }}<|eot|>\
                    {%- endfor -%}";
    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages = [serde_json::json!({"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "what is this?"}
    ]})];
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<|start|>user<|message|><|patch|>what is this?<|eot|>"
    );

    // A plain string-content template with the same namespace shape stays string-format.
    let plain = "{%- set ns = namespace(rest=messages) -%}{%- for message in ns.rest -%}\
                 {{ message['content'] }}{%- endfor -%}";
    assert_eq!(
        detect_chat_template_content_format(plain),
        ChatTemplateContentFormat::String
    );
}

/// `{{ x }}` of a list or a dict is Python's repr, the bytes the engine's
/// prompt holds when a template prints a content value its chat utils
/// handed over as parts; a float alone spells as Python's `str` does; a
/// string, an integer and a flag are written as before.
#[test]
fn a_sequence_or_a_map_is_written_as_python_s_repr() {
    let state = ChatTemplateState::new(Some("{{ messages[0].content }}".to_string())).unwrap();
    let render = |content: serde_json::Value| {
        state
            .apply(
                &[serde_json::json!({"role": "user", "content": content})],
                ChatTemplateParams::default(),
            )
            .unwrap()
    };
    assert_eq!(
        render(serde_json::json!([{"type": "text", "text": "Welcome! How can I help?"}])),
        "[{'type': 'text', 'text': 'Welcome! How can I help?'}]"
    );
    // Python's quote choice and escapes (`python3 -c 'print(repr(...))'`).
    assert_eq!(
        render(serde_json::json!([
            {"type": "text", "text": "it's"},
            {"type": "text", "text": "say \"hi\""},
            {"t": "both ' and \""}
        ])),
        r#"[{'type': 'text', 'text': "it's"}, {'type': 'text', 'text': 'say "hi"'}, {'t': 'both \' and "'}]"#
    );
    assert_eq!(
        render(serde_json::json!({
            "a": null, "b": true, "c": 1.0, "d": 1e16, "e": 0.0001, "f": 1e-5,
            "g": "tab\tnl\nbs\\ nbsp\u{a0} zw\u{200b} eé 😀", "h": [1, 2.5, -3, 100.0, 0.001]
        })),
        "{'a': None, 'b': True, 'c': 1.0, 'd': 1e+16, 'e': 0.0001, 'f': 1e-05, \
         'g': 'tab\\tnl\\nbs\\\\ nbsp\\xa0 zw\\u200b eé 😀', 'h': [1, 2.5, -3, 100.0, 0.001]}"
    );
    assert_eq!(render(serde_json::json!([])), "[]");
    assert_eq!(render(serde_json::json!({})), "{}");
    assert_eq!(render(serde_json::json!("plain")), "plain");
    assert_eq!(render(serde_json::json!(2.0)), "2.0");
    // A float printed alone spells as Python's `str` does, like one in a
    // list; an integer as before.
    assert_eq!(render(serde_json::json!(1e-5)), "1e-05");
    assert_eq!(render(serde_json::json!(1e16)), "1e+16");
    assert_eq!(render(serde_json::json!(100.0)), "100.0");
    assert_eq!(render(serde_json::json!(-0.001)), "-0.001");
    assert_eq!(render(serde_json::json!(12)), "12");
    assert_eq!(render(serde_json::json!(false)), "False");
    assert_eq!(render(serde_json::json!(null)), "None");

    // The `string` and `trim` filters start with the same `str`.
    let filtered = ChatTemplateState::new(Some(
        "{{ messages[0].content | string }}|{{ messages[0].content | trim }}|\
         {{ ' x ' | trim }}|{{ 'ab' | trim('a') }}|{{ 7 | string }}|{{ 0.00001 | string }}"
            .to_string(),
    ))
    .unwrap();
    let rendered = filtered
        .apply(
            &[serde_json::json!({"role": "user", "content": [{"type": "text", "text": "Hi"}]})],
            ChatTemplateParams::default(),
        )
        .unwrap();
    assert_eq!(
        rendered,
        "[{'type': 'text', 'text': 'Hi'}]|[{'type': 'text', 'text': 'Hi'}]|x|b|7|1e-05"
    );
}

/// The string filters Jinja2 starts with `str(value)` write a list or a
/// dict as Python does, like `string` and `trim`: `join` of every item
/// (or of an attribute of each), `replace` (with a count), `upper`,
/// `lower`, `capitalize`, `title` (a word-final sigma and a `\x1c`
/// separator as Python has them); a string and an integer are untouched.
#[test]
fn the_string_filters_start_with_python_s_str() {
    let state = ChatTemplateState::new(Some(
        "{{ messages[0].content | join(' + ') }}|\
         {{ messages[0].content | replace('text', 'T') }}|\
         {{ messages[0].content | upper }}|{{ messages[0].content[0] | lower }}|\
         {{ ['a', 'b'] | join(', ') }}|{{ 'Ab' | upper }}|{{ 7 | lower }}|\
         {{ 1.5 | replace('.', ',') }}|{{ 'x.y' | replace('.', '') }}|\
         {{ messages[0].content | capitalize }}|{{ messages[0].content | title }}|\
         {{ \"get_weather don't\" | title }}|\
         {{ messages[1].content | title }}|{{ messages[1].content | capitalize }}|\
         {{ 'ΑΣ' | title }}{{ 'ΑΣ' | capitalize }}|\
         {{ 'x.y.z' | replace('.', '-', 1) }}|{{ messages[0].content | join(', ', attribute='text') }}"
            .to_string(),
    ))
    .unwrap();
    let rendered = state
        .apply(
            &[
                serde_json::json!({"role": "user", "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "text", "text": "Yo"}
                ]}),
                serde_json::json!({"role": "user", "content": "a\u{1c}b ΟΔΟΣ ΚΑΙ"}),
            ],
            ChatTemplateParams::default(),
        )
        .unwrap();
    assert_eq!(
        rendered,
        "{'type': 'text', 'text': 'Hi'} + {'type': 'text', 'text': 'Yo'}|\
         [{'type': 'T', 'T': 'Hi'}, {'type': 'T', 'T': 'Yo'}]|\
         [{'TYPE': 'TEXT', 'TEXT': 'HI'}, {'TYPE': 'TEXT', 'TEXT': 'YO'}]|\
         {'type': 'text', 'text': 'hi'}|a, b|AB|7|1,5|xy|\
         [{'type': 'text', 'text': 'hi'}, {'type': 'text', 'text': 'yo'}]|\
         [{'type': 'text', 'text': 'hi'}, {'type': 'text', 'text': 'yo'}]|\
         Get_weather Don't|A\u{1c}B Οδος Και|A\u{1c}b οδος και|ΑσΑς|x-y.z|Hi, Yo"
    );
}

/// A filter call without keyword arguments takes its defaults without
/// reading any (the engine records every read in a set it allocates per
/// call); a call with a keyword the filter does not know still raises
/// the engine's own error, and so does a splat with a non-string key
/// (the key `args()` does not list), for `tojson` and for `join`.
#[test]
fn a_filter_call_with_an_unknown_keyword_still_raises_the_engine_s_error() {
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = vec![serde_json::json!({"type": "function", "function": {"name": "f"}})];
    for (template, expected) in [
        (
            "{{ tools | tojson(bogus=1) }}",
            "unknown keyword argument 'bogus'",
        ),
        (
            "{{ ['a', 'b'] | join(', ', bogus=1) }}",
            "unknown keyword argument 'bogus'",
        ),
        (
            "{{ tools | tojson(**{1: 2}) }}",
            "non string keys passed to kwargs",
        ),
        (
            "{{ ['a', 'b'] | join(', ', **{1: 2}) }}",
            "non string keys passed to kwargs",
        ),
    ] {
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let error = state.apply(&messages, params).unwrap_err().to_string();
        assert!(error.contains(expected), "{template}: {error}");
    }
    let state = ChatTemplateState::new(Some(
        "{{ tools | tojson }}|{{ ['a', 'b'] | join(', ') }}".to_string(),
    ))
    .unwrap();
    let params = ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    assert_eq!(
        state.apply(&messages, params).unwrap(),
        "[{\"type\": \"function\", \"function\": {\"name\": \"f\"}}]|a, b"
    );
}

/// The case of the record: a template that loops over content somewhere
/// (the openai content format) and prints `message.content` elsewhere
/// gets a string content as a one-item part list, and prints it the way
/// the engine's prompt holds it.
#[test]
fn a_parts_list_a_template_prints_comes_out_as_the_engine_writes_it() {
    let template = "{%- for m in messages %}\n{%- if m.role == 'system' %}<|system|>\n{{ m.content }}\
                    {%- else %}<|{{ m.role }}|>\n{%- for part in m.content %}{{ part.text }}{%- endfor %}\
                    {%- endif %}{%- endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    assert_eq!(state.content_format(), ChatTemplateContentFormat::OpenAI);
    let rendered = state
        .apply(
            &[
                serde_json::json!({"role": "system", "content": "You are a terse assistant."}),
                serde_json::json!({"role": "user", "content": "Hello."}),
            ],
            ChatTemplateParams::default(),
        )
        .unwrap();
    assert_eq!(
        rendered,
        "<|system|>\n[{'type': 'text', 'text': 'You are a terse assistant.'}]<|user|>Hello."
    );
}

/// A configuration that ships its templates as a named list is read as
/// transformers reads it: `default` renders a plain request, `tool_use`
/// one that declares tools; a string stays the one template; a replacement
/// template drops the pair.
#[test]
fn a_named_template_list_renders_tool_use_for_a_request_with_tools() {
    let config = serde_json::json!([
        {"name": "default", "template": "default:{{ messages | length }}"},
        {"name": "tool_use", "template": "tool_use:{{ tools | length }}"}
    ]);
    let templates = ConfigChatTemplates::read(Some(&config));
    assert_eq!(
        templates,
        ConfigChatTemplates {
            default: Some("default:{{ messages | length }}".to_string()),
            tool_use: Some("tool_use:{{ tools | length }}".to_string()),
        }
    );
    let mut state = ChatTemplateState::from_config(templates).unwrap();
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let with_tools = ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    assert_eq!(
        state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "default:1"
    );
    assert_eq!(state.apply(&messages, with_tools).unwrap(), "tool_use:1");

    state.set("alone:{{ tools is none }}".to_string()).unwrap();
    let with_tools = ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    assert_eq!(state.apply(&messages, with_tools).unwrap(), "alone:False");

    assert_eq!(
        ConfigChatTemplates::read(Some(&serde_json::json!("one"))),
        ConfigChatTemplates {
            default: Some("one".to_string()),
            tool_use: None,
        }
    );
    assert!(!ConfigChatTemplates::read(Some(&serde_json::json!({"name": "default"}))).is_some());
    assert!(!ConfigChatTemplates::read(None).is_some());
}

/// A tool set converted once renders as the values it was made of, with
/// and without a continued final message; one made of other values is not
/// read, and the render converts the values it was given.
#[test]
fn prepared_tools_render_as_the_values_they_were_made_of() {
    let state = ChatTemplateState::new(Some(
        "{% for tool in tools %}{{ tool | tojson }};{% endfor %}{{ tools | length }}:\
         {% for message in messages %}{{ message['content'] }}{% endfor %}"
            .to_string(),
    ))
    .unwrap();
    let messages = [
        serde_json::json!({"role": "user", "content": "Hi"}),
        serde_json::json!({"role": "assistant", "content": "The answer is"}),
    ];
    let tools = vec![
        serde_json::json!({"type": "function", "function": {"name": "f", "description": "first",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}),
        serde_json::json!({"type": "function", "function": {"name": "g", "parameters": {}}}),
    ];
    let prepared = PreparedTools::new(tools.clone());
    assert_eq!(prepared.tools(), tools.as_slice());
    fn params<'a>(
        tools: &'a [serde_json::Value],
        prepared_tools: Option<&'a PreparedTools>,
        continue_final_message: bool,
    ) -> ChatTemplateParams<'a> {
        ChatTemplateParams {
            tools: Some(tools),
            prepared_tools,
            add_generation_prompt: !continue_final_message,
            continue_final_message,
            ..Default::default()
        }
    }
    for continued in [false, true] {
        let plain = state
            .apply(&messages, params(&tools, None, continued))
            .unwrap();
        assert!(plain.contains("\"name\": \"f\""), "{plain}");
        assert_eq!(
            state
                .apply(
                    &messages,
                    params(prepared.tools(), Some(&prepared), continued)
                )
                .unwrap(),
            plain,
            "the prepared value renders as the values it was made of"
        );
        let others = [serde_json::json!({"type": "function", "function": {"name": "h"}})];
        let mismatched = state
            .apply(&messages, params(&others, Some(&prepared), continued))
            .unwrap();
        assert_eq!(
            mismatched,
            state
                .apply(&messages, params(&others, None, continued))
                .unwrap(),
            "a prepared value made of other tools is not read"
        );
        assert_ne!(mismatched, plain);
    }
}

/// Every way a template reads a tool of a prepared value answers as the
/// plain map does, on the first render (the texts are written and kept)
/// and on the second (the texts are read); the kept texts are one per
/// argument set of the filter, none for `sort_keys` or a sub-value.
#[test]
fn a_prepared_tool_reads_as_its_map_and_keeps_its_tojson_text_per_argument_set() {
    let state = ChatTemplateState::new(Some(
        "{% for tool in tools %}{{ tool | tojson }}|{{ tool | tojson(indent=4) }}|\
         {{ tool | tojson(ensure_ascii=True) }}|{{ tool | tojson(sort_keys=True) }}|\
         {{ tool | tojson(separators=(',', ':')) }}|{{ tool.function | tojson }}|\
         {{ tool.function.parameters.properties | tojson(indent=2) }}|{{ tool | length }}|\
         {{ tool is mapping }}|{% for k, v in tool | items %}{{ k }}={{ v }};{% endfor %}|\
         {% for k in tool %}{{ k }},{% endfor %}|{{ tool.get('type') }}|{{ tool.keys() | list }}|\
         {{ tool.items() | list }}|{{ 'function' in tool }}|{{ tool == tools[0] }}|\
         {{ tool != tools[1] }}|{{ tool }}|{{ tool | string }}|{{ tool | pprint }}|\
         {{ tool | dictsort | list }}|{% if tool %}truthy{% endif %}|{{ tool['type'] }}\n\
         {% endfor %}{{ tools | tojson }}|{{ tools | tojson(indent=2) }}|{{ tools | length }}|\
         {{ tools[1] | tojson }}|{{ tools | map(attribute='function.name') | list }}|\
         {{ tools | selectattr('function.name', 'eq', 'g') | list | length }}|\
         {{ tools is sequence }}|{{ tools[0] in tools }}"
            .to_string(),
    ))
    .unwrap();
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = vec![
        serde_json::json!({"type": "function", "function": {"name": "f",
            "description": "naïve résumé 日本 \"quoted\" <tag> & more",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string", "enum": ["a", "b"]},
                "n": {"type": "integer", "minimum": 1, "maximum": 100},
                "nested": {"type": "object", "properties": {"deep": {"type": "array", "items": {"type": "string"}}}}},
                "required": ["q"], "additionalProperties": false}}}),
        serde_json::json!({"type": "function", "function": {"name": "g", "parameters": {}}}),
    ];
    let params = |tools: &'static [serde_json::Value], prepared: Option<&'static PreparedTools>| {
        ChatTemplateParams {
            tools: Some(tools),
            prepared_tools: prepared,
            add_generation_prompt: true,
            ..Default::default()
        }
    };
    let tools: &'static [serde_json::Value] = Box::leak(tools.into_boxed_slice());
    let prepared: &'static PreparedTools = Box::leak(Box::new(PreparedTools::new(tools.to_vec())));
    let plain = state.apply(&messages, params(tools, None)).unwrap();
    assert!(
        plain.contains("naïve résumé 日本") && plain.contains("\\u65e5\\u672c"),
        "{plain}"
    );
    assert_eq!(prepared.kept_tojson_texts(), vec![0, 0]);
    let first = state
        .apply(&messages, params(prepared.tools(), Some(prepared)))
        .unwrap();
    assert_eq!(
        first, plain,
        "the first render writes the texts and keeps them"
    );
    assert_eq!(
        prepared.kept_tojson_texts(),
        vec![4, 4],
        "one text per argument set: default, indent, ensure_ascii, separators; not sort_keys, not a sub-value"
    );
    let second = state
        .apply(&messages, params(prepared.tools(), Some(prepared)))
        .unwrap();
    assert_eq!(second, plain, "the second render reads the kept texts");
    assert_eq!(prepared.kept_tojson_texts(), vec![4, 4]);
}

/// A tool keeps texts for the first few argument sets only; a form beyond
/// them is written as any value is, and renders the same.
#[test]
fn a_prepared_tool_keeps_a_bounded_number_of_tojson_texts() {
    let state = ChatTemplateState::new(Some(
        "{% for tool in tools %}{{ tool | tojson(indent=1) }}{{ tool | tojson(indent=2) }}\
         {{ tool | tojson(indent=3) }}{{ tool | tojson(indent=5) }}{{ tool | tojson(indent=6) }}\
         {{ tool | tojson(indent=1) }}{{ tool | tojson }}{% endfor %}"
            .to_string(),
    ))
    .unwrap();
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = vec![
        serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}),
    ];
    let prepared = PreparedTools::new(tools.clone());
    let plain = state
        .apply(
            &messages,
            ChatTemplateParams {
                tools: Some(&tools),
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..2 {
        let kept = state
            .apply(
                &messages,
                ChatTemplateParams {
                    tools: Some(prepared.tools()),
                    prepared_tools: Some(&prepared),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(kept, plain);
        assert_eq!(
            prepared.kept_tojson_texts(),
            vec![KEPT_TOJSON_TEXTS_PER_TOOL]
        );
    }
}

/// A tool that is not a map (nothing to keep a text for) is the plain
/// value, and a prepared set holding one renders as the values do.
#[test]
fn a_prepared_tool_that_is_not_a_map_is_the_plain_value() {
    let state = ChatTemplateState::new(Some(
        "{% for tool in tools %}{{ tool | tojson }}|{{ tool }}|{% endfor %}{{ tools | tojson }}"
            .to_string(),
    ))
    .unwrap();
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = vec![
        serde_json::json!("not a map"),
        serde_json::json!(["a", {"b": 1}]),
        serde_json::json!({"type": "function", "function": {"name": "f"}}),
    ];
    let prepared = PreparedTools::new(tools.clone());
    let plain = state
        .apply(
            &messages,
            ChatTemplateParams {
                tools: Some(&tools),
                ..Default::default()
            },
        )
        .unwrap();
    let kept = state
        .apply(
            &messages,
            ChatTemplateParams {
                tools: Some(prepared.tools()),
                prepared_tools: Some(&prepared),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(kept, plain);
    assert_eq!(prepared.kept_tojson_texts(), vec![0, 0, 1]);
}

/// The key-sorted form of a prepared set is the tools with every map's
/// keys sorted at every depth (lists keep their order, a tool that is not
/// a map is as it was), made once and kept; a render with it is the
/// render of the sorted values, with their own kept `tojson` texts.
#[test]
fn a_prepared_set_keeps_its_key_sorted_form_once() {
    let state = ChatTemplateState::new(Some(
        "[{% for tool in tools %}{{ tool | tojson(separators=(',', ':')) }}\
         {% if not loop.last %},{% endif %}{% endfor %}]"
            .to_string(),
    ))
    .unwrap();
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = vec![
        serde_json::json!({"type": "function", "function": {"name": "f", "description": "d",
            "parameters": {"type": "object", "properties": {
                "b": {"type": "string"}, "a": {"enum": ["z", "a"], "type": "string"}},
                "required": ["b", "a"]}}}),
        serde_json::json!("not a map"),
    ];
    let prepared = PreparedTools::new(tools.clone());
    let sorted = prepared.key_sorted();
    assert!(std::ptr::eq(sorted, prepared.key_sorted()), "made once");
    let sorted_text = concat!(
        r#"[{"function":{"description":"d","name":"f","parameters":{"properties":"#,
        r#"{"a":{"enum":["z","a"],"type":"string"},"b":{"type":"string"}},"#,
        r#""required":["b","a"],"type":"object"}},"type":"function"},"not a map"]"#
    );
    assert_eq!(serde_json::to_string(sorted.tools()).unwrap(), sorted_text);
    assert_eq!(
        serde_json::to_string(PreparedTools::key_sorted_of(&tools).tools()).unwrap(),
        sorted_text
    );
    let render = |tools: &[serde_json::Value], prepared: Option<&PreparedTools>| {
        state
            .apply(
                &messages,
                ChatTemplateParams {
                    tools: Some(tools),
                    prepared_tools: prepared,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    assert_eq!(render(sorted.tools(), Some(sorted)), sorted_text);
    assert_eq!(sorted.kept_tojson_texts(), vec![1, 0]);
    assert_eq!(render(sorted.tools(), Some(sorted)), sorted_text);
    assert_eq!(render(sorted.tools(), None), sorted_text);
    assert!(
        render(prepared.tools(), Some(&prepared)).starts_with(r#"[{"type":"function","#),
        "the plain form is as it was"
    );
    assert_eq!(prepared.kept_tojson_texts(), vec![1, 0]);
}

/// A checkpoint's processor template renders the requests that declare no
/// tools and the configuration's templates the ones that do, as the
/// engine's server resolves them; the content format answers for the
/// template selected. One equal to the default template, or one without
/// a default template to differ from, is not kept; a replacement template
/// drops it.
#[test]
fn a_checkpoint_s_processor_template_renders_a_request_without_tools() {
    let default = "default:{{ messages[0]['content'] }}";
    let processor = "processor:{% for message in messages %}\
         {% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endfor %}";
    let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
    let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let with_tools = || ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    let templates = || ConfigChatTemplates {
        default: Some(default.to_string()),
        tool_use: Some("tool_use:{{ tools | length }}".to_string()),
    };

    let mut state =
        ChatTemplateState::from_checkpoint(templates(), Some(processor.to_string())).unwrap();
    assert!(format!("{state:?}").contains("has_processor: true"));
    assert_eq!(
        state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "processor:Hi"
    );
    assert_eq!(state.apply(&messages, with_tools()).unwrap(), "tool_use:1");
    assert_eq!(
        state.content_format_for_tools(false),
        ChatTemplateContentFormat::OpenAI
    );
    assert_eq!(
        state.content_format_for_tools(true),
        ChatTemplateContentFormat::String
    );
    assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
    state.set("alone:{{ tools is none }}".to_string()).unwrap();
    assert!(format!("{state:?}").contains("has_processor: false"));
    assert_eq!(
        state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "alone:True"
    );

    let equal = ChatTemplateState::from_checkpoint(templates(), Some(default.to_string())).unwrap();
    assert!(format!("{equal:?}").contains("has_processor: false"));
    assert_eq!(
        equal
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "default:Hi"
    );

    let without_default = ChatTemplateState::from_checkpoint(
        ConfigChatTemplates::default(),
        Some(processor.to_string()),
    )
    .unwrap();
    assert!(format!("{without_default:?}").contains("has_processor: false"));
    assert!(without_default
        .apply(&messages, ChatTemplateParams::default())
        .is_err());

    // A processor template that disagrees with the default one on how a
    // request's thinking is prepared (here: a toggle the default has
    // none of) is not used: the render would not match the parse.
    let thinking = "{% if enable_thinking %}<think>{% endif %}{{ messages[0]['content'] }}";
    let disagreeing =
        ChatTemplateState::from_checkpoint(templates(), Some(thinking.to_string())).unwrap();
    assert!(format!("{disagreeing:?}").contains("has_processor: false"));
    assert_eq!(
        disagreeing
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "default:Hi"
    );
    let agreeing = ChatTemplateState::from_checkpoint(
        ConfigChatTemplates {
            default: Some(thinking.to_string()),
            tool_use: None,
        },
        Some(
            "processor:{{ messages[0]['content'] }}{% if enable_thinking %}<think>{% endif %}"
                .to_string(),
        ),
    )
    .unwrap();
    assert!(format!("{agreeing:?}").contains("has_processor: true"));

    // The raw-arguments answer compared is the one the gateway reads: in
    // a named pair the `tool_use` template's, not the default's.
    let concatenating = "{% for m in messages %}{% for c in m.tool_calls or [] %}\
                         {{ 'args=' + c.function.arguments }}{% endfor %}{% endfor %}";
    let named_pair = || ConfigChatTemplates {
        default: Some("{{ messages | length }}".to_string()),
        tool_use: Some(concatenating.to_string()),
    };
    let kept = ChatTemplateState::from_checkpoint(
        named_pair(),
        Some(format!("processor:{concatenating}")),
    )
    .unwrap();
    assert!(kept.raw_tool_call_arguments());
    assert!(format!("{kept:?}").contains("has_processor: true"));
    let dropped = ChatTemplateState::from_checkpoint(
        named_pair(),
        Some(
            "{% for m in messages %}{% for c in m.tool_calls or [] %}\
             {{ c.function.arguments | tojson }}{% endfor %}{% endfor %}"
                .to_string(),
        ),
    )
    .unwrap();
    assert!(format!("{dropped:?}").contains("has_processor: false"));

    // A processor template that does not compile leaves the tokenizer's
    // templates serving every request; the configuration's own still
    // fails the load, as before.
    let broken = ChatTemplateState::from_checkpoint(templates(), Some("{{".to_string())).unwrap();
    assert!(format!("{broken:?}").contains("has_processor: false"));
    assert_eq!(
        broken
            .apply(&messages, ChatTemplateParams::default())
            .unwrap(),
        "default:Hi"
    );
    assert!(ChatTemplateState::from_checkpoint(
        ConfigChatTemplates {
            default: Some("{{".to_string()),
            tool_use: None,
        },
        None,
    )
    .is_err());
}

/// The answer to how a history tool call's arguments are written comes
/// from the `tool_use` template of a named pair: the default one has no
/// tool-call branch to ask.
#[test]
fn a_named_pair_answers_tool_call_arguments_from_the_tool_use_template() {
    let concatenating = "{% for m in messages %}{% for c in m.tool_calls or [] %}\
                         {{ 'args=' + c.function.arguments }}{% endfor %}{% endfor %}";
    let alone = ChatTemplateState::new(Some(concatenating.to_string())).unwrap();
    assert!(alone.raw_tool_call_arguments());
    let pair = ChatTemplateState::from_config(ConfigChatTemplates {
        default: Some("{{ messages | length }}".to_string()),
        tool_use: Some(concatenating.to_string()),
    })
    .unwrap();
    assert!(pair.raw_tool_call_arguments());
    let plain = ChatTemplateState::from_config(ConfigChatTemplates {
        default: Some("{{ messages | length }}".to_string()),
        tool_use: Some("{{ tools | tojson }}".to_string()),
    })
    .unwrap();
    assert!(!plain.raw_tool_call_arguments());
}

/// A template file is read as written: a newline before its first block
/// is the template's and reaches the prompt, as it does through the
/// engine's chat utils, which read the file whole; exactly one trailing
/// newline of the source is dropped, as Jinja2 drops it.
#[test]
fn a_template_file_keeps_the_newline_it_opens_with() {
    let dir = tempfile::tempdir().unwrap();
    let render = |source: &str| {
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, source).unwrap();
        let template = load_chat_template_from_file(path.to_str().unwrap())
            .unwrap()
            .unwrap();
        let state = ChatTemplateState::new(Some(template)).unwrap();
        state
            .apply(
                &[serde_json::json!({"role": "user", "content": "Hi"})],
                ChatTemplateParams::default(),
            )
            .unwrap()
    };
    // Text around the blocks, as the templates of the record open (a `{%-`
    // tag strips the whitespace before it, and `trim_blocks` the newline
    // after a tag, in both engines alike).
    let body = "<s>{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}</s>";
    assert_eq!(render(&format!("\n{body}")), "\n<s><|user|>Hi</s>");
    assert_eq!(render(&format!("\n{body}\n")), "\n<s><|user|>Hi</s>");
    assert_eq!(render(&format!("\n{body}\n\n")), "\n<s><|user|>Hi</s>\n");
    assert_eq!(render(&format!("  {body}  ")), "  <s><|user|>Hi</s>  ");
}

/// The load warning counts a backslash-n in the template's text, not one
/// inside a tag (a string literal the template engine's escape turns into a
/// newline), and a real newline is no such pair.
#[test]
fn a_backslash_n_counts_only_outside_the_template_s_tags() {
    assert!(text_outside_tags_holds_backslash_n(
        "<|im_start|>system<|im_end|>\\n{{ messages[0].content }}"
    ));
    assert!(text_outside_tags_holds_backslash_n(
        "{%- for m in messages %}{{ m.content }}{%- endfor %}\\n"
    ));
    assert!(!text_outside_tags_holds_backslash_n(
        "{{ '\\n' }}{% set nl = \"\\n\" %}{# a \\n in a note #}plain text"
    ));
    assert!(!text_outside_tags_holds_backslash_n("line one\nline two\n"));
}

/// A backslash followed by an `n` in a template file's text is the two
/// characters it is, as the engine's chat utils read the file (a file
/// that is one escaped line renders as that line); inside a Jinja string
/// literal the template engine's own escape makes a newline of it, in
/// both engines alike.
#[test]
fn a_template_file_s_backslash_n_stays_the_two_characters_it_is() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat_template.jinja");
    fs::write(
        &path,
        "<|im_start|>system<|im_end|>\\n{% for m in messages %}{{ m.content }}{{ '\\n' }}{% endfor %}",
    )
    .unwrap();
    let template = load_chat_template_from_file(path.to_str().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(
        template,
        "<|im_start|>system<|im_end|>\\n{% for m in messages %}{{ m.content }}{{ '\\n' }}{% endfor %}"
    );
    let state = ChatTemplateState::new(Some(template)).unwrap();
    let rendered = state
        .apply(
            &[serde_json::json!({"role": "user", "content": "Hi"})],
            ChatTemplateParams::default(),
        )
        .unwrap();
    assert_eq!(rendered, "<|im_start|>system<|im_end|>\\nHi\n");
}

/// The rewrite keeps each tag's whitespace modifiers and leaves a template
/// without the block untouched.
#[test]
fn generation_tags_are_rewritten_with_their_modifiers() {
    assert_eq!(
        plain_generation_blocks(
            "{%- generation -%}x{% endgeneration %}y{%+ generation %}z{%-endgeneration-%}"
        ),
        "{%- if true -%}x{% endif %}y{%+ if true %}z{%- endif -%}"
    );
    assert!(matches!(
        plain_generation_blocks("{% if add_generation_prompt %}a{% endif %}"),
        Cow::Borrowed(_)
    ));
}

/// transformers' `strftime_now(format)`: the render's instant written with
/// the format; a template may guard it with `is defined`; a render that
/// names no instant writes today; an invalid format is an error.
#[test]
fn strftime_now_writes_the_render_instant() {
    let template = "{%- if strftime_now is defined -%}Current date: \
                    {{ strftime_now('%Y-%m-%d') }}. {{ strftime_now('%H:%M') }}{%- endif -%}";
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages: [serde_json::Value; 0] = [];
    let now = DateTime::parse_from_rfc3339("2026-10-07T09:30:00+02:00").unwrap();
    let params = ChatTemplateParams {
        now: Some(now),
        ..Default::default()
    };
    let rendered = processor.apply_chat_template(&messages, params).unwrap();
    assert_eq!(rendered, "Current date: 2026-10-07. 09:30");

    let today = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        today.len(),
        "Current date: 2026-10-07. 09:30".len(),
        "{today}"
    );

    let invalid = ChatTemplateProcessor::new("{{ strftime_now('%Q') }}".to_string()).unwrap();
    let error = invalid
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("strftime_now"), "{error}");
}

/// A render that names no instant reads the clock when the template first
/// writes it, once: every `strftime_now` of the render writes the same
/// instant, down to the nanosecond.
#[test]
fn strftime_now_reads_one_instant_per_render() {
    let template = "{{ strftime_now('%s.%f') }}|{{ strftime_now('%s.%f') }}";
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages: [serde_json::Value; 0] = [];
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    let (first, second) = rendered.split_once('|').unwrap();
    assert_eq!(first, second, "{rendered}");
    assert!(first.contains('.'), "{rendered}");
}

/// Jinja2's `dict()` is Python's: it builds a map from nothing, from a
/// mapping or from an iterable of key/value pairs, keyword arguments are
/// written over the result, and a repeated key keeps its first position
/// with its last value. A template merges a schema's `$defs` entry into
/// the property that references it as `dict((a | items | list) + (b |
/// items | list))`; the engine's own `dict` takes a mapping only.
#[test]
fn dict_builds_a_map_from_pairs_as_python_does() {
    let template = "{{ dict([('a', 1), ('b', 2), ('a', 3)]) | tojson }}|\
                    {{ dict({'x': 1}, y=2) | tojson }}|{{ dict() | tojson }}|\
                    {{ dict((messages[0].defs | items | list) + \
                    (messages[0].spec | items | rejectattr('0', 'equalto', '$ref') | list)) | tojson }}";
    let messages = [serde_json::json!({
        "role": "user",
        "defs": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "spec": {"$ref": "#/$defs/airport", "description": "IATA code"},
    })];
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        r#"{"a": 3, "b": 2}|{"x": 1, "y": 2}|{}|{"type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA code"}"#
    );

    // An element that is not a key/value pair is an error, as in Python.
    let processor = ChatTemplateProcessor::new("{{ dict([['a']]) }}".to_string()).unwrap();
    let error = processor
        .apply_chat_template(&[], ChatTemplateParams::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("2 is required"), "{error}");
}

/// `tojson` writes the template value straight into its text; the text is
/// Python's `json.dumps` of the value under each option, and the same as
/// a `serde_json::Value` built from the template value would serialize
/// as (the path the sorted form still takes).
#[test]
fn tojson_writes_the_value_as_python_and_as_its_tree_would() {
    let value = serde_json::json!({
        "name": "Zoë", "n": 1, "big": 18446744073709551615u64, "neg": -5, "f": 1.5,
        "whole": 2.0, "t": true, "none": null,
        "list": [1, "two", [3.25, {"k": "v"}]],
        "nested": {"b": 1, "a": {"c": [], "d": {}}},
        "esc": "quote\" back\\ tab\t nl\n ctl\u{1} emoji😀",
    });
    let template = "{{ messages[0].v | tojson }}\n\
                    {{ messages[0].v | tojson(ensure_ascii=True) }}\n\
                    {{ messages[0].v | tojson(indent=2) }}\n\
                    {{ messages[0].v | tojson(separators=(',', ':')) }}\n\
                    {{ messages[0].v | tojson(sort_keys=True) }}";
    let messages = [serde_json::json!({"role": "user", "v": value})];
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    // Each line is `json.dumps(v, ensure_ascii=False, ...)` with that option.
    let want = [
        r#"{"name": "Zoë", "n": 1, "big": 18446744073709551615, "neg": -5, "f": 1.5, "whole": 2.0, "t": true, "none": null, "list": [1, "two", [3.25, {"k": "v"}]], "nested": {"b": 1, "a": {"c": [], "d": {}}}, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji😀"}"#,
        r#"{"name": "Zo\u00eb", "n": 1, "big": 18446744073709551615, "neg": -5, "f": 1.5, "whole": 2.0, "t": true, "none": null, "list": [1, "two", [3.25, {"k": "v"}]], "nested": {"b": 1, "a": {"c": [], "d": {}}}, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji\ud83d\ude00"}"#,
        "{\n  \"name\": \"Zoë\",\n  \"n\": 1,\n  \"big\": 18446744073709551615,\n  \"neg\": -5,\n  \"f\": 1.5,\n  \"whole\": 2.0,\n  \"t\": true,\n  \"none\": null,\n  \"list\": [\n    1,\n    \"two\",\n    [\n      3.25,\n      {\n        \"k\": \"v\"\n      }\n    ]\n  ],\n  \"nested\": {\n    \"b\": 1,\n    \"a\": {\n      \"c\": [],\n      \"d\": {}\n    }\n  },\n  \"esc\": \"quote\\\" back\\\\ tab\\t nl\\n ctl\\u0001 emoji😀\"\n}",
        r#"{"name":"Zoë","n":1,"big":18446744073709551615,"neg":-5,"f":1.5,"whole":2.0,"t":true,"none":null,"list":[1,"two",[3.25,{"k":"v"}]],"nested":{"b":1,"a":{"c":[],"d":{}}},"esc":"quote\" back\\ tab\t nl\n ctl\u0001 emoji😀"}"#,
        r#"{"big": 18446744073709551615, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji😀", "f": 1.5, "list": [1, "two", [3.25, {"k": "v"}]], "n": 1, "name": "Zoë", "neg": -5, "nested": {"a": {"c": [], "d": {}}, "b": 1}, "none": null, "t": true, "whole": 2.0}"#,
    ];
    assert_eq!(rendered, want.join("\n"));

    // The direct write and the write of the tree agree byte for byte.
    let given = Value::from_serialize(&messages[0]["v"]);
    for (indent, ensure_ascii) in [
        (None, false),
        (None, true),
        (Some(&b"  "[..]), false),
        (Some(&b""[..]), true),
    ] {
        let direct = serialize_with_python_json(
            &given,
            indent,
            JsonSeparators::python_default(indent.is_some()),
            ensure_ascii,
        )
        .unwrap();
        let tree: serde_json::Value = serde_json::to_value(&given).unwrap();
        let through_tree = serialize_with_python_json(
            &tree,
            indent,
            JsonSeparators::python_default(indent.is_some()),
            ensure_ascii,
        )
        .unwrap();
        assert_eq!(
            direct, through_tree,
            "indent {indent:?} ensure_ascii {ensure_ascii}"
        );
    }
}

/// `tojson(indent=...)` renders as Python's `json.dumps(indent=...)` for
/// every kind of indent a template may write: an integer is that many
/// spaces, zero and a negative integer break the lines and indent nothing,
/// a bool is the integer it is in Python, a string is the literal indent
/// (a tab, two spaces, any text, the empty string), and the item separator
/// is the pretty one (`,`) whenever an indent was given, empty or not; a
/// float is refused, as Python refuses it. The expectations are Python
/// 3.12's `json.dumps(value, indent=indent, ensure_ascii=False)`.
#[test]
fn tojson_indent_renders_as_python_for_every_kind_of_indent() {
    let value = serde_json::json!({"a": [1, {"b": 2}], "c": {}, "d": [], "e": "x"});
    let newlines_only =
        "{\n\"a\": [\n1,\n{\n\"b\": 2\n}\n],\n\"c\": {},\n\"d\": [],\n\"e\": \"x\"\n}";
    let one_space = "{\n \"a\": [\n  1,\n  {\n   \"b\": 2\n  }\n ],\n \"c\": {},\n \"d\": [],\n \"e\": \"x\"\n}";
    let two_spaces = "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ],\n  \"c\": {},\n  \"d\": [],\n  \"e\": \"x\"\n}";
    let oracle: [(serde_json::Value, &str); 13] = [
        (serde_json::Value::Null, r#"{"a": [1, {"b": 2}], "c": {}, "d": [], "e": "x"}"#),
        (serde_json::json!(0), newlines_only),
        (serde_json::json!(-1), newlines_only),
        (serde_json::json!(-5), newlines_only),
        (serde_json::json!(1), one_space),
        (serde_json::json!(2), two_spaces),
        (serde_json::json!(true), one_space),
        (serde_json::json!(false), newlines_only),
        (serde_json::json!(""), newlines_only),
        (serde_json::json!(" "), one_space),
        (serde_json::json!("  "), two_spaces),
        (
            serde_json::json!("\t"),
            "{\n\t\"a\": [\n\t\t1,\n\t\t{\n\t\t\t\"b\": 2\n\t\t}\n\t],\n\t\"c\": {},\n\t\"d\": [],\n\t\"e\": \"x\"\n}",
        ),
        (
            serde_json::json!("ab"),
            "{\nab\"a\": [\nabab1,\nabab{\nababab\"b\": 2\nabab}\nab],\nab\"c\": {},\nab\"d\": [],\nab\"e\": \"x\"\n}",
        ),
    ];
    let processor = ChatTemplateProcessor::new(
        "{{ messages[0].v | tojson(indent=messages[0].i) }}".to_string(),
    )
    .unwrap();
    for (indent, want) in &oracle {
        let messages = [serde_json::json!({"role": "user", "v": value, "i": indent})];
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(&rendered, want, "indent {indent}");
    }
    // Keys sorted under a string indent, and empty containers on one line.
    let processor = ChatTemplateProcessor::new(
        "{{ messages[0].v | tojson(indent=messages[0].i, sort_keys=True) }}".to_string(),
    )
    .unwrap();
    let messages = [serde_json::json!({
        "role": "user", "v": {"b": 1, "a": {"d": [], "c": 2}}, "i": "\t"
    })];
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "{\n\t\"a\": {\n\t\t\"c\": 2,\n\t\t\"d\": []\n\t},\n\t\"b\": 1\n}"
    );
    // A float is no indent, in Python either.
    let processor = ChatTemplateProcessor::new(
        "{{ messages[0].v | tojson(indent=messages[0].i) }}".to_string(),
    )
    .unwrap();
    let messages = [serde_json::json!({"role": "user", "v": value, "i": 2.0})];
    let error = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("indent must be an integer or a string"),
        "{error}"
    );
}

/// transformers renders with `tools=None` and `documents=None` when the
/// request carries none, so a template that tests `tools is none` (Olmo 3)
/// takes its no-tools branch, and one that guards with `is defined and
/// tools` does too.
#[test]
fn absent_tools_and_documents_are_none_as_transformers_passes_them() {
    let template = "{% if tools is none %}no tools{% else %}{{ tools | tojson }}{% endif %}|\
                    {% if tools is defined and tools %}has tools{% else %}none{% endif %}|\
                    {% if tools is iterable and tools | length > 0 %}iterable{% else %}not iterable{% endif %}|\
                    {% if documents is none %}no documents{% endif %}";
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages: [serde_json::Value; 0] = [];
    let rendered = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered, "no tools|none|not iterable|no documents");

    let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let params = ChatTemplateParams {
        tools: Some(&tools),
        ..Default::default()
    };
    let rendered = processor.apply_chat_template(&messages, params).unwrap();
    assert_eq!(
        rendered,
        "[{\"type\": \"function\", \"function\": {\"name\": \"f\"}}]|has tools|iterable|no documents"
    );
}

#[test]
fn test_chat_template_state_no_template() {
    let state = ChatTemplateState::new(None).unwrap();
    assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
    let result = state.apply(&[], ChatTemplateParams::default());
    assert!(result.is_err());
}

#[test]
fn test_chat_template_state_set() {
    let mut state = ChatTemplateState::new(None).unwrap();
    state.set("{{ messages }}".to_string()).unwrap();
    assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
}

#[test]
fn developer_message_renders_as_system_when_the_template_has_no_developer_branch() {
    // The template names system/user/assistant only: a developer message
    // would render nothing. The engine's renderer turns it into a system message.
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}\
                    {% if messages[0].tools is defined %}TOOLS{% endif %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "developer", "content": "Be terse.", "tools": [{"name": "t"}]}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered, "<sys>Be terse.</sys><usr>hi</usr>");
}

/// With a system message of its own in the request, the rewritten
/// developer message is no longer the first message: the two merge into
/// one system turn at the front, as the engine's renderer merges them.
#[test]
fn a_developer_message_after_a_system_message_merges_into_one_system_turn() {
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "system", "content": "You are terse."}),
        serde_json::json!({"role": "developer", "content": "Answer in French."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<sys>You are terse.\n\nAnswer in French.</sys><usr>hi</usr>"
    );
}

/// A developer message in the middle of the conversation becomes the
/// system turn at the front; the other messages keep their order, and a
/// parts list contributes its text parts.
#[test]
fn a_developer_message_after_an_assistant_turn_moves_to_the_front() {
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>\
                    {% elif m.role == 'assistant' %}<ast>{{ m.content }}</ast>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "user", "content": "hi"}),
        serde_json::json!({"role": "assistant", "content": "hello"}),
        serde_json::json!({"role": "developer", "content": [
            {"type": "text", "text": "Be brief."},
            {"type": "text", "text": "No lists."},
        ]}),
        serde_json::json!({"role": "user", "content": "go on"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<sys>Be brief.\nNo lists.</sys><usr>hi</usr><ast>hello</ast><usr>go on</usr>"
    );
}

/// A template that never names the `root` role (the MiniMax templates
/// before M3 read `messages[0].role == "system"` only) would drop a root
/// message: it gets the message as the leading system message instead.
#[test]
fn a_root_message_renders_as_the_system_message_when_the_template_has_no_root_branch() {
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "root", "content": "Answer in French."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered, "<sys>Answer in French.</sys><usr>hi</usr>");
}

/// A system message after the root is not lost on such a template: the
/// two merge into one system turn, the root's text first.
#[test]
fn a_system_message_after_a_root_merges_into_one_system_turn_without_a_root_branch() {
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "root", "content": "Answer in French."}),
        serde_json::json!({"role": "system", "content": "You are terse."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<sys>Answer in French.\n\nYou are terse.</sys><usr>hi</usr>"
    );
}

/// A root followed by a developer message on a template with neither
/// branch: both become the one system turn, the root's text first.
#[test]
fn a_root_and_a_developer_message_merge_root_first_without_either_branch() {
    let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "root", "content": "Answer in French."}),
        serde_json::json!({"role": "developer", "content": "Be brief."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<sys>Answer in French.\n\nBe brief.</sys><usr>hi</usr>"
    );
}

#[test]
fn a_root_message_reaches_a_template_that_handles_the_role() {
    let template = "{% for m in messages %}{% if m.role == 'root' %}<root>{{ m.content }}</root>\
                    {% elif m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "root", "content": "Answer in French."}),
        serde_json::json!({"role": "system", "content": "You are terse."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(
        rendered,
        "<root>Answer in French.</root><sys>You are terse.</sys><usr>hi</usr>"
    );
}

#[test]
fn developer_message_reaches_a_template_that_handles_the_role() {
    let template =
        "{% for m in messages %}{% if m.role == 'developer' %}<dev>{{ m.content }}</dev>\
                    {% elif m.role == 'system' %}<sys>{{ m.content }}</sys>\
                    {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    let messages = vec![
        serde_json::json!({"role": "developer", "content": "Be terse."}),
        serde_json::json!({"role": "user", "content": "hi"}),
    ];
    let rendered = state
        .apply(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered, "<dev>Be terse.</dev><usr>hi</usr>");
}

/// A non-thinking template without a switch whose generation prompt
/// writes an empty, closed think block: `<think>` is in the prefill, but
/// the completion starts in content, so the parser must not be armed.
#[test]
fn a_closed_think_block_in_the_generation_prompt_does_not_open_reasoning() {
    let closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                  {%- if add_generation_prompt -%}<|im_start|>assistant\n<think>\n\n</think>\n\n{%- endif -%}";
    let state = ChatTemplateState::new(Some(closed.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::None);
    assert!(
        state.think_in_prefill(),
        "the loose flag still sees the tag"
    );
    assert!(!state.prefill_opens_think_block());

    // Opened in one branch, closed in another: the open branch counts.
    let branched = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|>\
                    {%- if clear_thinking is defined and clear_thinking -%}<think></think>\
                    {%- else -%}<think>{%- endif -%}{%- endif -%}";
    let state = ChatTemplateState::new(Some(branched.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());

    // Both branches open a block and the prompt closes it after them.
    let closed_after_branches = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|>\
                    {%- if clear_thinking is defined and clear_thinking -%}<think>\
                    {%- else -%}<think>\n{%- endif -%}</think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(closed_after_branches.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());

    // A block opened only when there is no generation prompt is not the
    // prompt's; the else branch of that condition is.
    let not_prompt = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if not add_generation_prompt -%}<think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(not_prompt.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());
    let else_prompt = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if not add_generation_prompt -%}done{%- else -%}<|assistant|><think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(else_prompt.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());

    // A comparison is not a positive reference either: this true body
    // runs without a generation prompt.
    let compared = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt == false -%}<think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(compared.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());
    let conjunction = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt and messages -%}<|assistant|><think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(conjunction.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());

    // The prompt written in two generation-prompt blocks: the second may
    // close what the first opened, or open what the first left closed.
    let open_then_closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}\
                    {%- if add_generation_prompt -%}\n\n</think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(open_then_closed.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());
    let closed_then_open = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|><think></think>{%- endif -%}\
                    {%- if add_generation_prompt -%}<think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(closed_then_open.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());

    // Two blocks that run on different requests (a conjunction each):
    // either may leave the block open, so the prompt counts as opening
    // whichever order they come in; a block that always runs still closes
    // what an earlier one opened.
    let exclusive = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt and thinking -%}<|assistant|><think>{%- endif -%}\
                    {%- if add_generation_prompt and not thinking -%}<|assistant|><think></think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(exclusive.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());
    let maybe_then_closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt and thinking -%}<|assistant|><think>{%- endif -%}\
                    {%- if add_generation_prompt -%}</think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(maybe_then_closed.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());

    // Conditions the variable alone decides: `not (add_generation_prompt
    // == false)` (the parse of the unparenthesised form) and
    // `add_generation_prompt == true` run their true body for a
    // generation prompt; `add_generation_prompt or x` always runs too, so
    // it closes what an earlier block opened; `not (add_generation_prompt
    // or x)` always runs its false body.
    let negated_comparison = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if not add_generation_prompt == false -%}<think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(negated_comparison.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());
    let eq_true = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt == true -%}<|assistant|><think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(eq_true.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());
    let or_closes = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}\
                    {%- if add_generation_prompt or continue_final_message -%}</think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(or_closes.to_string())).unwrap();
    assert!(!state.prefill_opens_think_block());
    let not_or = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if not (add_generation_prompt or continue_final_message) -%}done\
                    {%- else -%}<|assistant|><think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(not_or.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());

    // A test on the variable inside a condition something else also
    // decides: the block may run, so what it opens counts.
    let is_true_and = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt is true and not tools -%}<|assistant|><think>{%- endif -%}";
    let state = ChatTemplateState::new(Some(is_true_and.to_string())).unwrap();
    assert!(state.prefill_opens_think_block());
}

#[test]
fn clear_thinking_is_not_a_thinking_switch() {
    // GLM-5.3: the template reads `clear_thinking` and `reasoning_effort`
    // and always opens the think block; it has no switch.
    let template =
        "{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}\
                    {%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                    {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}";
    assert_eq!(
        detect_thinking_toggle(template),
        (ThinkingToggle::None, None)
    );
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::None);
    assert!(state.think_in_prefill());
    assert!(state.prefill_opens_think_block());

    // A real `thinking` variable is still a switch, in each spelling.
    for template in [
        "{% if thinking is defined and not thinking %}off{% endif %}",
        "{% if thinking == false %}off{% endif %}",
        "{% if thinking %}on{% endif %}",
        "{% set thinking = thinking %}",
    ] {
        assert_eq!(
            detect_thinking_toggle(template),
            (ThinkingToggle::DefaultOn, Some(ThinkingKeyName::Thinking)),
            "{template}"
        );
    }
}

/// The word `thinking` in what a template prints is not a switch: the
/// MiniMax M3.1 template's system prose ("When thinking is enabled, wrap your
/// reasoning in ...") stands in a string literal, and the template branches
/// on no thinking variable, where the M3 template branched on
/// `thinking_mode`. A comment or the raw text outside the blocks is not a
/// switch either; the same words as code are.
#[test]
fn thinking_in_prose_or_a_comment_is_not_a_thinking_switch() {
    let m31_shape = "{{- 'You have a thinking capability that allows you to reason step by step \
                     before responding. When thinking is enabled, wrap your reasoning in ' ~ \
                     think_begin_token ~ ' and ' ~ think_end_token }}\
                     {%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                     {%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";
    for template in [
        m31_shape,
        "{{ \"When thinking is enabled, think.\" }}{% for m in messages %}{{ m.content }}{% endfor %}",
        "{% set note = 'if thinking == true' %}{{ note }}",
        "{# if thinking is defined #}{{ messages | length }}",
        "When thinking is enabled, think.{% for m in messages %}{{ m.content }}{% endfor %}",
        // A raw block prints its text as it is, switch-looking words included.
        "{% raw %}{% if thinking %}on{% endif %}{% endraw %}{{ messages | length }}",
    ] {
        assert_eq!(
            detect_thinking_toggle(template),
            (ThinkingToggle::None, None),
            "{template}"
        );
    }
    let state = ChatTemplateState::new(Some(m31_shape.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::None);
    assert_eq!(state.thinking_key_name(), None);
    assert!(!state.prefill_opens_think_block());

    // The same words as code are the switch, after a literal and after a
    // literal with an escaped quote in it.
    for template in [
        "{% if thinking is defined and thinking %}<think>{% endif %}",
        "{{ 'When thinking is enabled' }}{% if thinking %}<think>{% endif %}",
        "{{ 'it\\'s thinking' }}{% if thinking == true %}<think>{% endif %}",
        "{{ \"a 'quoted' word\" }}{% set thinking = thinking %}",
        // A nested dict literal's `}}` does not end the block before the switch.
        "{{ {'a': {'b': '<think>'}}['a']['b'] if thinking else '' }}",
    ] {
        assert_eq!(
            detect_thinking_toggle(template),
            (ThinkingToggle::DefaultOn, Some(ThinkingKeyName::Thinking)),
            "{template}"
        );
    }
    // The M3 shape keeps its tri-state switch.
    assert_eq!(
        detect_thinking_toggle(
            "{{- 'When thinking is enabled, wrap your reasoning.' }}\
             {%- if thinking_mode is defined and thinking_mode == \"enabled\" -%}<think>{%- endif -%}"
        ),
        (
            ThinkingToggle::DefaultOff,
            Some(ThinkingKeyName::ThinkingMode)
        )
    );
}

#[test]
fn test_chat_template_state_invalid_template() {
    let result = ChatTemplateState::new(Some("{% invalid".to_string()));
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Failed to add template"),
        "Error should explain parse failure, got: {err}"
    );
}

#[test]
fn test_chat_template_processor_invalid_template() {
    let result = ChatTemplateProcessor::new("{% invalid".to_string());
    assert!(result.is_err());
}

#[test]
fn test_raise_exception_surfaces_template_validation_message() {
    let state = ChatTemplateState::new(Some(
        "{{ raise_exception('reasoning_effort is invalid') }}".to_string(),
    ))
    .unwrap();

    let error = state
        .apply(&[], ChatTemplateParams::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("reasoning_effort is invalid"), "{error}");
}

#[test]
fn test_special_tokens_injected_into_context() {
    let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

    let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
    let special_tokens = crate::traits::SpecialTokens {
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        ..Default::default()
    };

    let result = state
        .apply(
            &messages,
            ChatTemplateParams {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(result, "<s>hello</s>");
}

#[test]
fn test_special_tokens_undefined_when_not_provided() {
    let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

    let result = state.apply(&[], ChatTemplateParams::default()).unwrap();
    assert_eq!(result, "hello");
}

#[test]
fn test_special_tokens_partial() {
    let template = "{{ bos_token }}hello{% if eos_token is defined %}{{ eos_token }}{% endif %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

    let special_tokens = crate::traits::SpecialTokens {
        bos_token: Some("<s>".to_string()),
        eos_token: None,
        ..Default::default()
    };

    let result = state
        .apply(
            &[],
            ChatTemplateParams {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(result, "<s>hello");
}

#[test]
fn thinking_param_sets_template_key_and_explicit_wins() {
    use std::collections::HashMap;

    // Template echoes the enable_thinking value so we can observe what was set.
    let state = ChatTemplateState::new(Some("{{ enable_thinking }}".to_string())).unwrap();
    assert_eq!(
        state.thinking_key_name(),
        Some(ThinkingKeyName::EnableThinking)
    );

    // thinking = Some(false) injects enable_thinking=false under the model's key.
    // Rendered Python-style: transformers evaluates these templates under
    // Jinja2, where `{{ False }}` is "False", so a template that echoes the
    // flag into the prompt must produce the same bytes we would there.
    let out = state
        .apply(
            &[],
            ChatTemplateParams {
                thinking: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out, "False");

    // An explicit template_kwargs entry overrides the injected default.
    let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
    kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
    let out = state
        .apply(
            &[],
            ChatTemplateParams {
                thinking: Some(false),
                template_kwargs: Some(&kwargs),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out, "True");
}

#[test]
fn gemma4_filter_default_keeps_thinking_off() {
    let template = "{%- set enable_thinking = enable_thinking | default(false) -%}{% if enable_thinking %}<|channel>thought\n{% endif %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOff);
    assert_eq!(
        state.thinking_key_name(),
        Some(ThinkingKeyName::EnableThinking)
    );
    for (thinking, expected) in [
        (None, ""),
        (Some(false), ""),
        (Some(true), "<|channel>thought\n"),
    ] {
        assert_eq!(
            state
                .apply(
                    &[],
                    ChatTemplateParams {
                        thinking,
                        ..Default::default()
                    }
                )
                .unwrap(),
            expected
        );
    }
}

#[test]
fn thinking_filter_on_other_values_does_not_change_the_switch_default() {
    for template in [
        "{{ message.thinking | default(false) }}{% if enable_thinking %}<think>{% endif %}",
        "{{ other_thinking | default(false) }}{% if thinking %}<think>{% endif %}",
    ] {
        assert_eq!(
            detect_thinking_toggle(template).0,
            ThinkingToggle::DefaultOn
        );
    }
    assert_eq!(
        detect_thinking_toggle(
            "{% set thinking = thinking | default(false) %}{% if thinking %}<think>{% endif %}"
        ),
        (ThinkingToggle::DefaultOff, Some(ThinkingKeyName::Thinking))
    );
}

#[test]
fn thinking_mode_detects_tristate_and_injects_strings() {
    // MiniMax M3-style tri-state toggle.
    let template = r#"{%- if thinking_mode is defined and thinking_mode == "enabled" -%}<mm:think>{%- endif -%}"#;
    assert_eq!(
        detect_thinking_toggle(template),
        (
            ThinkingToggle::DefaultOff,
            Some(ThinkingKeyName::ThinkingMode)
        )
    );

    // Mentioning the variable without branching on it must not trigger
    // tri-state detection.
    assert_eq!(
        detect_thinking_toggle("{{ enable_thinking }}{# thinking_mode note #}"),
        (
            ThinkingToggle::DefaultOn,
            Some(ThinkingKeyName::EnableThinking)
        )
    );

    // The resolved boolean preference injects the template's string values.
    let state = ChatTemplateState::new(Some(
        "{% if thinking_mode is defined %}{{ thinking_mode }}{% endif %}".to_string(),
    ))
    .unwrap();
    assert_eq!(
        state.thinking_key_name(),
        Some(ThinkingKeyName::ThinkingMode)
    );
    for (thinking, rendered) in [(true, "enabled"), (false, "disabled")] {
        let out = state
            .apply(
                &[],
                ChatTemplateParams {
                    thinking: Some(thinking),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out, rendered);
    }
}

/// Regression: a conditional expression used as a keyword-argument value.
/// Jinja2 accepts it, so published chat templates use it — Muse-Glimmer's
/// does, in a `namespace(...)` call — but minijinja rejected it before
/// 2.24, which made the whole template uncompilable and left the model
/// unservable rather than merely mis-rendered.
#[test]
fn conditional_expression_in_keyword_argument_compiles() {
    let template = "{%- set r = namespace(name=x if x else '') -%}{{ r.name }}";
    let state = ChatTemplateState::new(Some(template.to_string()))
        .expect("conditional kwargs must compile");
    let out = state.apply(&[], ChatTemplateParams::default()).unwrap();
    assert_eq!(out, "");
}

/// None renders Python-style too, for the same Jinja2-parity reason.
#[test]
fn none_renders_python_style() {
    let state = ChatTemplateState::new(Some("{{ undefined_value }}".to_string())).unwrap();
    let out = state.apply(&[], ChatTemplateParams::default()).unwrap();
    assert_eq!(out, "");
}
