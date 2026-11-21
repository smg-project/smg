mod hy_v4;
mod random_filter;
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

/// A render without a template says what the model lacks and that no request
/// can supply it: a render call carries no template of its own. The failure is
/// not the template's refusal, so a caller does not read it as its input's.
#[test]
fn a_render_without_a_template_says_that_the_model_has_none() {
    let messages = [serde_json::json!({"role": "user", "content": "hi"})];
    for state in [
        ChatTemplateState::empty(),
        ChatTemplateState::new(None).unwrap(),
    ] {
        let error = state
            .apply(&messages, ChatTemplateParams::default())
            .expect_err("a render without a template");
        assert_eq!(
            error.to_string(),
            "The model has no chat template. One must be configured for the model; \
             a request cannot supply it."
        );
        assert!(!TemplateRefusal::is(&error), "{error:#}");
    }
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

/// A state reads the names its templates read: a bare name anywhere in an
/// expression of the default template, or of an alternate a request may
/// render through. The gateway hands a request control to the template
/// under its name only when the template reads it.
#[test]
fn a_state_reads_the_names_its_templates_read() {
    let reading = "{% macro format(f) %}{{ f | tojson }}{% endmacro %}{% for m in messages %}\
                   {{ m.content }}{% endfor %}{% if response_format is defined %}\
                   {{ format(response_format) }}{% endif %}";
    let state = ChatTemplateState::new(Some(reading.to_string())).unwrap();
    assert!(state.reads("response_format"));
    assert!(state.reads("messages"));
    assert!(!state.reads("tool_choice"));

    let plain = "{% for m in messages %}{{ m.content }}{% endfor %}";
    let state = ChatTemplateState::new(Some(plain.to_string())).unwrap();
    assert!(!state.reads("response_format"));
    assert!(!state.reads("tool_choice"));

    // An alternate's reads count: the `tool_use` template a request with
    // tools renders through.
    let state = ChatTemplateState::from_config(ConfigChatTemplates {
        default: Some(plain.to_string()),
        tool_use: Some(format!("{plain}{{{{ tool_choice }}}}")),
    })
    .unwrap();
    assert!(state.reads("tool_choice"));
    assert!(!state.reads("response_format"));

    // A name read only as a call block's parameter default counts: the
    // block's parameters are expressions of the template, as a macro's.
    let call_block = "{% macro wrap() %}{{ caller() }}{% endmacro %}{% for m in messages %}\
                      {{ m.content }}{% endfor %}{% call(fmt=response_format) wrap() %}\
                      {{ fmt | tojson }}{% endcall %}";
    let state = ChatTemplateState::new(Some(call_block.to_string())).unwrap();
    assert!(state.reads("response_format"));
    assert!(!state.reads("tool_choice"));

    // The processor alternate's reads count too: a checkpoint's processor
    // template, which a request without tools renders through.
    let state = ChatTemplateState::from_checkpoint(
        ConfigChatTemplates {
            default: Some(plain.to_string()),
            tool_use: None,
        },
        Some(format!("processor:{plain}{{{{ response_format }}}}")),
    )
    .unwrap();
    assert!(format!("{state:?}").contains("has_processor: true"));
    assert!(state.reads("response_format"));
    assert!(!state.reads("tool_choice"));

    // A replacement template answers for itself.
    let mut state = ChatTemplateState::new(Some(plain.to_string())).unwrap();
    state.set(reading.to_string()).unwrap();
    assert!(state.reads("response_format"));

    // No template: nothing to leave a name out of.
    assert!(ChatTemplateState::empty().reads("response_format"));
    assert!(ChatTemplateState::new(None).unwrap().reads("tool_choice"));

    // A template the detector could not parse leaves the names unknown,
    // and an unknown set reads every name rather than none.
    assert!(detect_all("{% if %}").names_read.is_none());
    let mut state = ChatTemplateState::new(Some(plain.to_string())).unwrap();
    state.names_read = None;
    assert!(state.reads("response_format"));
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

/// A template's own `raise_exception` is typed as the template's refusal, from
/// inside a macro too, under the text every render failure reads; an operation
/// the template could not run is a render failure and no refusal.
#[test]
fn a_templates_own_refusal_is_typed_and_another_render_failure_is_not() {
    let failure = |template: &str| {
        ChatTemplateState::new(Some(template.to_string()))
            .unwrap()
            .apply(
                &[serde_json::json!({"role": "user", "content": []})],
                ChatTemplateParams::default(),
            )
            .unwrap_err()
    };

    let refused = failure("{{ raise_exception('the roles must alternate') }}");
    assert!(TemplateRefusal::is(&refused), "{refused:#}");
    assert!(
        refused
            .to_string()
            .starts_with("Failed to render template: invalid operation: the roles must alternate"),
        "{refused}"
    );
    let in_a_macro =
        failure("{% macro check() %}{{ raise_exception('no') }}{% endmacro %}{{ check() }}");
    assert!(TemplateRefusal::is(&in_a_macro), "{in_a_macro:#}");

    let failed = failure("{{ 'user: ' + messages[0]['content'] }}");
    assert!(!TemplateRefusal::is(&failed), "{failed:#}");
    assert!(
        failed
            .to_string()
            .starts_with("Failed to render template: invalid operation: "),
        "{failed}"
    );

    // A continued message whose reasoning the fallback would lose says so
    // after the failure of its render, which is typed as it was: the
    // template's refusal a refusal, a render that stops on a value neither.
    let thinking = [
        serde_json::json!({"role": "user", "content": "hi"}),
        serde_json::json!({
            "role": "assistant",
            "content": "Partial",
            "reasoning_content": "still thinking"
        }),
    ];
    let continued = |template: &str| {
        let failed = render_failure(template, &thinking, serde_json::json!({}), true);
        assert!(
            failed
                .to_string()
                .ends_with("its reasoning_content would be lost"),
            "{template}: {failed}"
        );
        failed
    };
    let refused = continued("{{ raise_exception('the roles must alternate') }}");
    assert!(TemplateRefusal::is(&refused), "{refused:#}");
    assert!(
        refused
            .to_string()
            .starts_with("Failed to render template: invalid operation: the roles must alternate"),
        "{refused}"
    );
    let failed = continued("{{ messages[2]['content'] }}");
    assert!(!TemplateRefusal::is(&failed), "{failed:#}");
    assert!(!TemplateFailure::is(&failed), "{failed:#}");
    assert!(
        failed
            .to_string()
            .starts_with("Failed to render template: undefined value"),
        "{failed}"
    );
}

/// The classes of the engine's error kinds, a kind at a time: the template's
/// own where the template's source alone raises the kind, a plain failure
/// where a value can. Three kinds are the template's own on a condition.
/// Another template that is not there, where the source spells every name it
/// asks for. A called name nothing defines, where the error names it and the
/// source binds it nowhere. A filter or a test the engine does not have,
/// where the source names it: looked up from an argument's value, the error
/// has no detail.
#[test]
fn each_error_kind_is_classed_by_whether_a_value_can_raise_it() {
    let environment = |source: &str| build_environment(source.to_string()).unwrap();
    let spells = environment("{% include 'tools.jinja' %}{% macro helper() %}{% endmacro %}");
    let reads = environment("{% include 'tools.jinja' %}{% include fragments %}");
    let own = |environment: &Environment<'_>, error: MinijinjaError| {
        is_the_templates_own(&error, &environment.get_template("chat").unwrap())
    };
    for kind in [
        ErrorKind::SyntaxError,
        ErrorKind::BadEscape,
        ErrorKind::TemplateNotFound,
        ErrorKind::UnknownBlock,
        ErrorKind::UnknownFilter,
        ErrorKind::UnknownTest,
        ErrorKind::WriteFailure,
    ] {
        assert!(
            own(&spells, MinijinjaError::new(kind, "a detail")),
            "{kind:?}"
        );
    }
    for kind in [
        ErrorKind::InvalidOperation,
        ErrorKind::UnknownMethod,
        ErrorKind::UndefinedError,
        ErrorKind::TooManyArguments,
        ErrorKind::MissingArgument,
        ErrorKind::CannotUnpack,
        ErrorKind::BadSerialization,
        ErrorKind::CannotDeserialize,
        ErrorKind::NonPrimitive,
        ErrorKind::NonKey,
        ErrorKind::BadInclude,
        ErrorKind::EvalBlock,
    ] {
        assert!(
            !own(&spells, MinijinjaError::new(kind, "a detail")),
            "{kind:?}"
        );
    }

    assert!(
        !own(
            &reads,
            MinijinjaError::new(ErrorKind::TemplateNotFound, "a detail")
        ),
        "a template name read from a value"
    );
    let unknown = |detail: &'static str| MinijinjaError::new(ErrorKind::UnknownFunction, detail);
    assert!(own(&spells, unknown("late is unknown")));
    for (error, why) in [
        (unknown("helper is unknown"), "a name the source binds"),
        (unknown("loop is unknown"), "a name the engine binds"),
        (unknown("caller is unknown"), "a name the engine binds"),
        (unknown("macro not found"), "a detail that names nothing"),
        (ErrorKind::UnknownFunction.into(), "no detail"),
        (ErrorKind::UnknownFilter.into(), "no detail"),
        (ErrorKind::UnknownTest.into(), "no detail"),
    ] {
        assert!(!own(&spells, error), "{why}");
    }
}

/// `conversation` rendered through `template` with `kwargs` as the caller's
/// template arguments.
fn rendered_with(
    template: &str,
    conversation: &[serde_json::Value],
    kwargs: serde_json::Value,
    continue_final_message: bool,
) -> Result<String> {
    let kwargs: HashMap<String, serde_json::Value> = serde_json::from_value(kwargs).unwrap();
    ChatTemplateState::new(Some(template.to_string()))
        .unwrap()
        .apply(
            conversation,
            ChatTemplateParams {
                template_kwargs: Some(&kwargs),
                continue_final_message,
                ..Default::default()
            },
        )
}

/// The failure of that render ([`rendered_with`]).
fn render_failure(
    template: &str,
    conversation: &[serde_json::Value],
    kwargs: serde_json::Value,
    continue_final_message: bool,
) -> anyhow::Error {
    rendered_with(template, conversation, kwargs, continue_final_message).unwrap_err()
}

/// A render that stops on something the template's source alone decides is
/// typed as the template's failure, under the text every render failure
/// reads, a kind of the engine's at a time: another template it asks for,
/// and a filter, a test, a function and a block the engine does not have. So
/// is a template that is not there to look up. A continued render types it
/// the same, and so does one of a message whose reasoning the fallback would
/// lose, which says so after the failure.
#[test]
fn a_render_failure_the_templates_source_alone_decides_is_typed_as_the_templates() {
    let user = serde_json::json!({"role": "user", "content": "hi"});
    let partial = serde_json::json!({"role": "assistant", "content": "Partial"});
    let thinking = serde_json::json!({
        "role": "assistant",
        "content": "Partial",
        "reasoning_content": "still thinking"
    });
    let conversation = [user.clone(), partial.clone()];
    for (template, kind) in [
        ("{% include 'tools.jinja' %}", "template not found"),
        ("{% extends 'base.jinja' %}", "template not found"),
        (
            "{% import 'macros.jinja' as macros %}",
            "template not found",
        ),
        (
            "{% from 'macros.jinja' import helper %}",
            "template not found",
        ),
        (
            "{% include ['tools.jinja', 'functions.jinja'] %}",
            "template not found",
        ),
        ("{{ messages | no_such_filter }}", "unknown filter"),
        (
            "{% if messages is no_such_test %}x{% endif %}",
            "unknown test",
        ),
        ("{{ no_such_function() }}", "unknown function"),
        ("{{ self.no_such_block() }}", "unknown block"),
    ] {
        for (last, continued) in [(&partial, false), (&partial, true), (&thinking, true)] {
            let conversation = [user.clone(), last.clone()];
            let failed = render_failure(template, &conversation, serde_json::json!({}), continued);
            assert!(TemplateFailure::is(&failed), "{template}: {failed:#}");
            assert!(!TemplateRefusal::is(&failed), "{template}: {failed:#}");
            assert!(
                failed
                    .to_string()
                    .starts_with(&format!("Failed to render template: {kind}: ")),
                "{template}: {failed}"
            );
            assert_eq!(
                failed.to_string().ends_with(
                    ": only its text could follow the generation prompt, and its \
                     reasoning_content would be lost"
                ),
                std::ptr::eq(last, &thinking),
                "{template}: {failed}"
            );
        }
    }

    // A block is the template's wherever its statement stands, under a
    // branch that does not run too: no value makes a block the source has
    // unknown.
    let block_under_a_branch = "{% if messages | length > 2 %}{% block late %}here{% endblock %}\
                                {% endif %}{{ self.late() }}";
    assert_eq!(
        rendered_with(
            block_under_a_branch,
            &conversation,
            serde_json::json!({}),
            false
        )
        .unwrap(),
        "here"
    );

    let unnamed = render_messages(
        &Environment::new(),
        conversation.iter(),
        ChatTemplateParams::default(),
        &RenderedLength::default(),
    )
    .unwrap_err();
    assert!(TemplateFailure::is(&unnamed), "{unnamed:#}");
    assert!(
        unnamed
            .to_string()
            .starts_with("Failed to get template: template not found: "),
        "{unnamed}"
    );

    // A template whose source does not compile is refused when it loads,
    // under the compiler's two kinds: no render meets either.
    for (template, kind) in [
        ("{% if %}", "syntax error"),
        (r#"{{ "\uzzzz" }}"#, "bad string escape"),
    ] {
        let refused = ChatTemplateState::new(Some(template.to_string()))
            .expect_err("a template that does not compile");
        assert!(
            refused
                .to_string()
                .starts_with(&format!("Failed to add template: {kind}")),
            "{template}: {refused}"
        );
    }
}

/// A render that stops on a value stays the plain failure it was, neither
/// the template's failure nor its refusal, a kind of the engine's at a time.
/// A caller's template argument can break a sound template this way: one of
/// the wrong kind, or one that takes the name of a function the template
/// calls, its own `raise_exception` among them. A template that breaks on
/// every conversation (it iterates over a number) fails with the same kind,
/// which the render cannot tell from a caller's value, and stays plain too.
/// So does a function the engine does not have where a macro calls it: the
/// engine fails that call as an operation on an undefined value.
#[test]
fn a_render_failure_a_value_can_bring_about_stays_plain() {
    let one_turn = [serde_json::json!({"role": "user", "content": "hi"})];
    let parts = [serde_json::json!({"role": "user", "content": []})];
    let none = || serde_json::json!({});
    let calls_namespace = "{% set ns = namespace(found=false) %}{{ ns.found }}";
    let refuses = "{{ raise_exception('the roles must alternate') }}";
    let writes_a_date = "{{ 'Today Date: ' + date_string }}";
    let cases = [
        // A caller's template argument under the name of a function.
        (
            calls_namespace,
            &one_turn[..],
            serde_json::json!({"namespace": 0}),
            "invalid operation",
        ),
        (
            calls_namespace,
            &one_turn[..],
            serde_json::json!({"namespace": null}),
            "invalid operation",
        ),
        (
            refuses,
            &one_turn[..],
            serde_json::json!({"raise_exception": 0}),
            "invalid operation",
        ),
        // A caller's template argument of the wrong kind.
        (
            writes_a_date,
            &one_turn[..],
            serde_json::json!({"date_string": 20_261_120}),
            "invalid operation",
        ),
        // Content of a shape the template does not take.
        (
            "{{ 'user: ' + messages[0]['content'] }}",
            &parts[..],
            none(),
            "invalid operation",
        ),
        // The template's own bug, on every conversation.
        (
            "{% for part in 42 %}{{ part }}{% endfor %}",
            &one_turn[..],
            none(),
            "invalid operation",
        ),
        (
            "{{ messages[0]['content'].no_such_method() }}",
            &one_turn[..],
            none(),
            "unknown method",
        ),
        // A second message the conversation does not have.
        (
            "{{ messages[1]['content'] }}",
            &one_turn[..],
            none(),
            "undefined value",
        ),
        (
            "{{ raise_exception() }}",
            &one_turn[..],
            none(),
            "missing argument",
        ),
        (
            "{{ raise_exception('a', 'b') }}",
            &one_turn[..],
            none(),
            "too many arguments",
        ),
        (
            "{% for a, b, c in messages %}{% endfor %}",
            &one_turn[..],
            none(),
            "cannot unpack",
        ),
        // The template includes itself: the wrapper around whatever stopped
        // the included render.
        (
            "{% include 'chat' %}",
            &one_turn[..],
            none(),
            "could not render include",
        ),
        // A function the engine does not have, called in a macro: the engine
        // hands a macro each name it reads, this one as an undefined value.
        (
            "{% macro helper() %}{{ no_such_function() }}{% endmacro %}{{ helper() }}",
            &one_turn[..],
            none(),
            "invalid operation",
        ),
        // A filter and a test named by an argument's value.
        (
            "{{ messages | map('no_such_filter') | list }}",
            &one_turn[..],
            none(),
            "unknown filter",
        ),
        (
            "{{ messages | select('no_such_test') | list }}",
            &one_turn[..],
            none(),
            "unknown test",
        ),
    ];
    for (template, conversation, kwargs, kind) in cases {
        let failed = render_failure(template, conversation, kwargs.clone(), false);
        assert!(
            !TemplateFailure::is(&failed),
            "{template} {kwargs}: {failed:#}"
        );
        assert!(
            !TemplateRefusal::is(&failed),
            "{template} {kwargs}: {failed:#}"
        );
        assert!(
            failed
                .to_string()
                .starts_with(&format!("Failed to render template: {kind}")),
            "{template} {kwargs}: {failed}"
        );
    }

    // Without the argument that broke them, the templates above are sound:
    // two render, and the third refuses in its own words.
    let sound = |template: &str, kwargs: serde_json::Value| {
        rendered_with(template, &one_turn, kwargs, false)
    };
    assert_eq!(sound(calls_namespace, none()).unwrap(), "False");
    assert_eq!(
        sound(
            writes_a_date,
            serde_json::json!({"date_string": "20 Nov 2026"})
        )
        .unwrap(),
        "Today Date: 20 Nov 2026"
    );
    assert!(TemplateRefusal::is(&sound(refuses, none()).unwrap_err()));
}

/// Another template that is not there is the template's failure only where
/// the source spells every name it asks for. A name read from a value is the
/// caller's to get wrong, and an empty list of them renders. The error does
/// not say which statement raised it, so one statement that reads its name
/// keeps the failure plain: a value can hand it a name the source spells in
/// another statement, one that did not run.
#[test]
fn a_missing_template_a_value_can_name_stays_plain() {
    let one_turn = [serde_json::json!({"role": "user", "content": "hi"})];
    let includes_fragments = "{% include fragments %}";
    for (template, kwargs) in [
        (
            includes_fragments,
            serde_json::json!({"fragments": ["missing.jinja"]}),
        ),
        (
            includes_fragments,
            serde_json::json!({"fragments": "missing.jinja"}),
        ),
        (
            "{% extends parent %}",
            serde_json::json!({"parent": "base.jinja"}),
        ),
        (
            "{% import helpers as helper %}",
            serde_json::json!({"helpers": "macros.jinja"}),
        ),
        (
            "{% from helpers import helper %}",
            serde_json::json!({"helpers": "macros.jinja"}),
        ),
        (
            "{% if messages | length > 1 %}{% include 'tools.jinja' %}{% endif %}\
             {% include fragments %}",
            serde_json::json!({"fragments": ["tools.jinja"]}),
        ),
    ] {
        let failed = render_failure(template, &one_turn, kwargs.clone(), false);
        assert!(
            !TemplateFailure::is(&failed),
            "{template} {kwargs}: {failed:#}"
        );
        assert!(
            failed
                .to_string()
                .starts_with("Failed to render template: template not found: "),
            "{template} {kwargs}: {failed}"
        );
    }
    let no_fragments = serde_json::json!({"fragments": []});
    assert_eq!(
        rendered_with(includes_fragments, &one_turn, no_fragments, false).unwrap(),
        ""
    );
}

/// A call of a name nothing defines is the template's failure only where the
/// source binds the name on no branch. Where a macro, a `set`, a loop, a
/// `with`, an import or a parameter binds it somewhere, a value decides
/// whether the binding ran before the call, and the failure stays plain: the
/// first template renders a conversation that opens with `hello` and fails
/// any other. `loop` and `caller` are the engine's to bind and stay plain too.
#[test]
fn a_call_of_a_name_the_source_binds_on_some_branch_stays_plain() {
    let greets_hello = "{% if messages[0].content == 'hello' %}{% macro greet() %}hi there\
                        {% endmacro %}{% endif %}{{ greet() }}";
    let one_turn = |content: &str| [serde_json::json!({"role": "user", "content": content})];
    let none = || serde_json::json!({});
    assert_eq!(
        rendered_with(greets_hello, &one_turn("hello"), none(), false).unwrap(),
        "hi there"
    );
    let long = "messages | length > 1";
    let bound_somewhere = [
        greets_hello.to_string(),
        format!("{{% if {long} %}}{{% set late = namespace %}}{{% endif %}}{{{{ late() }}}}"),
        format!("{{% if {long} %}}{{% set late %}}x{{% endset %}}{{% endif %}}{{{{ late() }}}}"),
        format!("{{% if {long} %}}{{% set late, other = 1, 2 %}}{{% endif %}}{{{{ late() }}}}"),
        "{% for late in messages %}{% endfor %}{{ late() }}".to_string(),
        "{% for role, late in [] %}{% endfor %}{{ late() }}".to_string(),
        "{% with late = namespace %}{% endwith %}{{ late() }}".to_string(),
        "{% macro helper(late) %}{% endmacro %}{{ late() }}".to_string(),
        "{% macro helper() %}{{ caller(1) }}{% endmacro %}\
         {% call(late) helper() %}{% endcall %}{{ late() }}"
            .to_string(),
        format!("{{% if {long} %}}{{% import 'chat' as late %}}{{% endif %}}{{{{ late() }}}}"),
        format!("{{% if {long} %}}{{% from 'chat' import late %}}{{% endif %}}{{{{ late() }}}}"),
        "{% block body %}{% set late = namespace %}{% endblock %}{{ late() }}".to_string(),
        "{{ loop(messages) }}".to_string(),
        "{{ caller() }}".to_string(),
    ];
    for template in &bound_somewhere {
        let failed = render_failure(template, &one_turn("hi"), none(), false);
        assert!(!TemplateFailure::is(&failed), "{template}: {failed:#}");
        assert!(
            failed
                .to_string()
                .starts_with("Failed to render template: unknown function: "),
            "{template}: {failed}"
        );
    }

    // A name the source binds nowhere is the template's own to call, beside
    // a binding of another name and under a branch a value picks.
    for template in [
        format!("{{% if {long} %}}{{% set other = namespace %}}{{% endif %}}{{{{ late() }}}}"),
        "{% if messages[0].content != 'hello' %}{{ late() }}{% endif %}".to_string(),
    ] {
        let failed = render_failure(&template, &one_turn("hi"), none(), false);
        assert!(TemplateFailure::is(&failed), "{template}: {failed:#}");
        assert!(
            failed
                .to_string()
                .starts_with("Failed to render template: unknown function: late is unknown"),
            "{template}: {failed}"
        );
    }
}

/// The render's own checks of the request are not the template's failure,
/// and not its refusal: a continuation asked for beside a generation prompt,
/// one with no message to continue, and one of a message whose reasoning
/// the fallback would lose, because the message has no text or because the
/// template never writes it. A caller answers each as a request it could not
/// render.
#[test]
fn the_renders_checks_of_the_request_are_not_typed_as_the_templates_failure() {
    let state = |template: &str| ChatTemplateState::new(Some(template.to_string())).unwrap();
    let every_turn = state("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}");
    let user_turns = state(
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}\n{% endif %}{% endfor %}",
    );
    let continuing = || ChatTemplateParams {
        continue_final_message: true,
        ..Default::default()
    };
    let user = serde_json::json!({"role": "user", "content": "hi"});
    let thinking = |content: serde_json::Value| {
        serde_json::json!({
            "role": "assistant",
            "content": content,
            "reasoning_content": "still thinking"
        })
    };
    let lost = "only its text could follow the generation prompt, and its reasoning_content \
                would be lost";
    let cases = [
        (
            &every_turn,
            vec![user.clone(), thinking("Partial".into())],
            ChatTemplateParams {
                add_generation_prompt: true,
                ..continuing()
            },
            "continue_final_message and add_generation_prompt are not compatible".to_string(),
        ),
        (
            &every_turn,
            vec![],
            continuing(),
            "no final message to continue".to_string(),
        ),
        (
            &every_turn,
            vec![user.clone(), thinking(serde_json::Value::Null)],
            continuing(),
            format!("the final message has no text to continue: {lost}"),
        ),
        (
            &user_turns,
            vec![user.clone(), thinking("Partial".into())],
            continuing(),
            format!("the final message does not appear in the rendered chat: {lost}"),
        ),
    ];
    for (state, messages, params, expected) in cases {
        let error = state.apply(&messages, params).unwrap_err();
        assert_eq!(error.to_string(), expected);
        assert!(!TemplateFailure::is(&error), "{expected}");
        assert!(!TemplateRefusal::is(&error), "{expected}");
    }
}

#[test]
fn test_special_tokens_injected_into_context() {
    let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

    let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
    let special_tokens = SpecialTokens {
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

    let special_tokens = SpecialTokens {
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

/// A template's thinking default is what it renders with the switch absent,
/// whatever its source spells. A template that opens the thought, or asks
/// for it, only where the switch is true defaults off: an undefined switch
/// is not true. The text matcher knows two spellings of a default off and
/// calls each of these three on.
#[test]
fn the_thinking_default_is_what_the_template_renders_with_the_switch_absent() {
    let defined_and_true = "{%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                            {%- if add_generation_prompt -%}<|assistant|>\
                            {%- if enable_thinking is defined and enable_thinking is true -%}\
                            {{- '<think>\\n' -}}{%- else -%}{{- '<think>\\n\\n</think>\\n\\n' -}}\
                            {%- endif -%}{%- endif -%}";
    let truthy = "{%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                  {%- if add_generation_prompt -%}<|assistant|>\
                  {%- if enable_thinking -%}<think>{%- endif -%}{%- endif -%}";
    let in_the_system_prompt = "{%- if thinking -%}<|system|>Think before you answer.\
                                {%- else -%}<|system|>Answer.{%- endif -%}\
                                {%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                                {%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";
    for (template, key) in [
        (defined_and_true, ThinkingKeyName::EnableThinking),
        (truthy, ThinkingKeyName::EnableThinking),
        (in_the_system_prompt, ThinkingKeyName::Thinking),
    ] {
        assert_eq!(
            detect_thinking_toggle(template),
            (ThinkingToggle::DefaultOn, Some(key)),
            "the text matcher: {template}"
        );
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(
            state.thinking_toggle(),
            ThinkingToggle::DefaultOff,
            "{template}"
        );
        assert_eq!(state.thinking_key_name(), Some(key));
    }

    // A template that closes the thought only where the switch is false
    // defaults on, and one the matcher reads as off renders off: both as
    // before.
    let defined_and_false = "{%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                             {%- if add_generation_prompt -%}<|assistant|>\
                             {%- if enable_thinking is defined and enable_thinking is false -%}\
                             {{- '<think>\\n\\n</think>\\n\\n' -}}{%- else -%}{{- '<think>\\n' -}}\
                             {%- endif -%}{%- endif -%}";
    let set_false = "{%- if enable_thinking is not defined -%}\
                     {%- set enable_thinking = false -%}{%- endif -%}\
                     {%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                     {%- if add_generation_prompt -%}<|assistant|>\
                     {%- if enable_thinking -%}<think>{%- endif -%}{%- endif -%}";
    for (template, default) in [
        (defined_and_false, ThinkingToggle::DefaultOn),
        (set_false, ThinkingToggle::DefaultOff),
    ] {
        assert_eq!(detect_thinking_toggle(template).0, default, "{template}");
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), default, "{template}");
    }

    // A replacement template answers for itself.
    let mut state = ChatTemplateState::new(Some(defined_and_false.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOn);
    state.set(truthy.to_string()).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOff);
}

/// The three probe renders share one instant. A template that writes the
/// time would otherwise separate them by the clock alone, and its default
/// would fall back to the text matcher's answer: here the time is written to
/// the nanosecond, so three reads of the clock never agree.
#[test]
fn the_probe_renders_share_one_instant() {
    let writes_the_time = "{{- strftime_now('%H:%M:%S.%f') -}}\
                           {%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                           {%- if add_generation_prompt -%}<|assistant|>\
                           {%- if enable_thinking -%}<think>{%- endif -%}{%- endif -%}";
    assert_eq!(
        detect_thinking_toggle(writes_the_time),
        (
            ThinkingToggle::DefaultOn,
            Some(ThinkingKeyName::EnableThinking)
        ),
        "the text matcher"
    );
    let state = ChatTemplateState::new(Some(writes_the_time.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOff);
}

/// Where the renders do not say, the text matcher's answer stands: a bare
/// turn that does not render, a switch that moves nothing on a bare turn,
/// and an absent render that equals neither the on nor the off render.
#[test]
fn the_text_matcher_answers_where_the_renders_do_not_say() {
    // The template refuses a conversation with no system message.
    let refuses = "{%- if messages[0].role != 'system' -%}\
                   {{- raise_exception('a system message is required') -}}{%- endif -%}\
                   {%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                   {%- if add_generation_prompt -%}<|assistant|>\
                   {%- if enable_thinking -%}<think>{%- endif -%}{%- endif -%}";
    // The switch only shapes an assistant message of the history.
    let history_only = "{%- for m in messages -%}\
                        {%- if m.role == 'assistant' and enable_thinking -%}\
                        <think>{{ m.reasoning_content }}</think>{%- endif -%}\
                        {{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";
    // No thought marker with the switch absent; true opens the thought and
    // false closes it.
    let third = "{%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}\
                 {%- if add_generation_prompt -%}<|assistant|>\
                 {%- if enable_thinking is defined and enable_thinking is true -%}\
                 {{- '<think>\\n' -}}\
                 {%- elif enable_thinking is defined and enable_thinking is false -%}\
                 {{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}{%- endif -%}";
    for template in [refuses, history_only, third] {
        let matched = detect_thinking_toggle(template);
        assert_eq!(
            matched,
            (
                ThinkingToggle::DefaultOn,
                Some(ThinkingKeyName::EnableThinking)
            ),
            "{template}"
        );
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), matched.0, "{template}");
    }
    let bare = [serde_json::json!({"role": "user", "content": "probe"})];
    let params = || ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };
    let state = ChatTemplateState::new(Some(refuses.to_string())).unwrap();
    assert!(state.apply(&bare, params()).is_err(), "the bare turn");
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
