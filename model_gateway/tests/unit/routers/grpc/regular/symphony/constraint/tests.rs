use openai_protocol::worker::GrammarStart;

use super::*;
use crate::routers::grpc::regular::symphony::testing::{assign, constraint, *};

/// A tag the worker's grammar backend has no spelling of counts as a tag the table does not
/// have: a registry table's request goes out as before, whatever it asked (the older registry
/// decides a forced call or an answer, `auto` asks for no constraint); the response template's
/// table refuses by the backend's name, as it refuses a tag it lacks. Every worker spells for
/// xgrammar today, so the rule is pinned on the helper that takes the spelling's refusal.
#[test]
fn an_unspelled_tag_counts_as_a_tag_the_table_does_not_have() {
    for backend in [GrammarBackend::Llguidance, GrammarBackend::Outlines] {
        let unspelled = Unspelled { backend };
        assert!(
            matches!(
                unspelled_tag(unspelled, false, "m", "a forced call"),
                Ok(None)
            ),
            "a registry table leaves the request as it was"
        );
        let refused = unspelled_tag(unspelled, true, "m", "a forced call")
            .expect_err("the response template's table refuses");
        let words = refused.to_string();
        assert!(words.contains(backend.name()), "{words}");
        assert!(words.contains("model 'm'"), "{words}");
        assert!(words.contains("a forced call"), "{words}");
    }
}

/// A Messages request's untabled schema is its format's, except under a choice that forces a
/// call to a declared tool: the turn has no text answer to format, whether or not the call has a
/// constraint; with no callable tool, or a choice naming a tool outside the callable ones (a
/// tool of another kind, which the model never sees), the choice forces nothing and the answer
/// is formatted.
#[test]
fn a_forced_call_leaves_the_untabled_schema_out() {
    let schema =
        json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
    let request = |choice: Option<Value>| -> CreateMessageRequest {
        let mut body = json!({
            "model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        });
        if let Some(choice) = choice {
            body["tool_choice"] = choice;
        }
        serde_json::from_value(body).expect("a Messages request")
    };
    let tools: Vec<Tool> = serde_json::from_value(json!([
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    ]))
    .expect("a tool");
    let asks = |request: &CreateMessageRequest| {
        Asks::for_messages(request, false)
            .untabled_answer_schema(&tools)
            .map(Cow::into_owned)
    };
    assert_eq!(asks(&request(None)), Some(schema.clone()));
    assert_eq!(
        asks(&request(Some(json!({"type": "auto"})))),
        Some(schema.clone())
    );
    assert_eq!(asks(&request(Some(json!({"type": "any"})))), None);
    assert_eq!(
        asks(&request(Some(json!({"type": "tool", "name": "lookup"})))),
        None
    );
    assert_eq!(
        asks(&request(Some(
            json!({"type": "tool", "name": "web_search"})
        ))),
        Some(schema.clone()),
        "a choice naming a tool outside the callable ones forces no call to them"
    );
    let forced = request(Some(json!({"type": "any"})));
    assert_eq!(
        Asks::for_messages(&forced, false)
            .untabled_answer_schema(&[])
            .map(Cow::into_owned),
        Some(schema.clone()),
        "no callable tool: the choice forces nothing"
    );
}

/// A Messages request's `output_config.format` is the answer's schema as a chat request's
/// `response_format` is: the table derives the same tag for both, and hands a worker the
/// schema itself where it has no tag for it.
#[test]
fn a_messages_output_format_is_the_answers_schema_as_a_chat_response_format_is() {
    let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let chat = request(json!({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "city", "schema": schema},
    }}));
    let messages: CreateMessageRequest = serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B", "max_tokens": 16, "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }))
    .expect("a Messages request");
    let answer_of = |asks| {
        let prepared = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            asks,
            &[],
            true,
            false,
            symphony::GrammarStart::Either,
            TagVocabulary::Current,
            &templates(),
        );
        let Ok(Prepared::Answer(answer)) = prepared else {
            panic!("the answer's tag, not {prepared:?}");
        };
        answer
    };
    let for_chat = answer_of(Asks::Chat(&chat));
    let for_messages = answer_of(Asks::for_messages(&messages, false));
    assert_eq!(for_messages.schema(), Some(&schema));
    for start in [
        symphony::GrammarStart::FirstToken,
        symphony::GrammarStart::Unknown,
    ] {
        let tag = |answer: &Arc<AnswerTag>| {
            answer
                .for_start(start, TagVocabulary::Current)
                .expect("no hf")
                .map(|constraint| constraint.to_tuple())
        };
        assert_eq!(tag(&for_messages), tag(&for_chat), "{start:?}");
    }
}

/// Where a worker's engine starts the grammar shapes what precedes a table's answer or its
/// calls when the prompt leaves the thought open: nothing after the reasoning parser's gate
/// (a JSON answer then needs no tag; the request's own schema goes to the engine as it is),
/// the thought closed first from the first token, the schema alone (the request's own, no
/// tag) when the worker does not say, and for the calls the free-text prefix, marked as
/// such, when the model's workers name different starts or none: free text owes the call
/// nothing, so preparation gives the call's JSON schema in the tag's place. The
/// answer's tag derives for the selected worker, whatever start the pool names.
#[test]
fn the_workers_grammar_start_shapes_what_precedes_a_tables_answer_and_calls() {
    let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
    let asked = request(json!({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "city", "schema": {"type": "object"}},
    }}));
    // The pool names either kind (a first-token worker beside another): the answer's tag is
    // not built for it, but for the worker each attempt selects.
    let prepared = super::constraint(
        ParserChoice::Symphony,
        &deepseek,
        Asks::Chat(&asked),
        &[],
        true,
        false,
        symphony::GrammarStart::Either,
        TagVocabulary::Current,
        &templates(),
    );
    let Ok(Prepared::Answer(answer)) = prepared else {
        panic!("the answer's tag, derived per worker, not {prepared:?}");
    };
    let tag_of = |start| {
        answer
            .for_start(start, TagVocabulary::Current)
            .expect("no hf")
            .map(|constraint| match constraint {
                ToolConstraint::StructuralTag(tag) => {
                    serde_json::from_str::<Value>(&tag).expect("JSON")
                }
                ToolConstraint::JsonSchema(schema) => {
                    panic!("a structural tag, not the JSON schema {schema}")
                }
            })
    };
    // Where the worker does not say, the answer takes no prefix either: free text before a
    // schema constrains nothing, so the schema stands alone and goes to the engine as the
    // request's own, as it does after the gate.
    assert_eq!(
        tag_of(symphony::GrammarStart::Unknown),
        None,
        "the schema stands alone where the worker does not say"
    );
    assert_eq!(
        tag_of(symphony::GrammarStart::AfterReasoning),
        None,
        "after the gate the schema stands alone, and needs no tag"
    );
    let first = tag_of(symphony::GrammarStart::FirstToken).expect("a tag");
    assert_eq!(first["format"]["elements"][0]["type"], "tag");
    assert_eq!(first["format"]["elements"][0]["begin"], "");
    assert_eq!(first["format"]["elements"][0]["end"], "</think>");
    assert_eq!(first["format"]["elements"][1]["type"], "json_schema");
    let forced = request(json!({"tool_choice": "required"}));
    let call_of = |start| {
        let Ok(Prepared::Constraint {
            constraint: ToolConstraint::StructuralTag(tag),
            first_token,
            ..
        }) = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(&forced),
            &tools(),
            true,
            false,
            start,
            TagVocabulary::Current,
            &templates(),
        )
        else {
            panic!("a forced tag");
        };
        assert_eq!(
            first_token,
            start == symphony::GrammarStart::FirstToken,
            "{start:?}: built for the first token only there"
        );
        serde_json::from_str::<Value>(&tag).expect("JSON")
    };
    for start in [
        symphony::GrammarStart::Unknown,
        symphony::GrammarStart::Either,
    ] {
        let Ok(Prepared::OpensWithFreeText {
            constraint: ToolConstraint::StructuralTag(tag),
            first_token,
        }) = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(&forced),
            &tools(),
            true,
            false,
            start,
            TagVocabulary::Current,
            &templates(),
        )
        else {
            panic!("{start:?}: a tag marked as opening with free text");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert_eq!(
            tag["format"]["elements"][0]["type"], "any_text",
            "{start:?}: {tag}"
        );
        assert!(!first_token, "{start:?}: not built for the first token");
    }
    let bare = call_of(symphony::GrammarStart::AfterReasoning);
    assert_eq!(
        bare["format"]["elements"][0]["type"], "const_string",
        "the block's opener comes first at the gate: {bare}"
    );
    let first = call_of(symphony::GrammarStart::FirstToken);
    assert_eq!(first["format"]["elements"][0]["end"], "</think>");
    assert_eq!(first["format"]["elements"][1], bare["format"]);
}

/// No tag of a registry table that opens with free text goes unmarked. Over every table,
/// wherever the workers start a grammar, the thought never opened, closed by the prompt or
/// open, one call and many: the table's tag opens with free text only from an open thought
/// on workers that name no one start, the tag is marked exactly there, and every other tag
/// stands unmarked.
#[test]
fn a_forced_calls_tag_is_marked_only_where_it_opens_with_free_text() {
    let tools = tools();
    let mut marked = 0;
    for table in symphony::formats::Table::ALL {
        let format = table.format();
        for (reasoning_open, thought_closed) in [(false, false), (false, true), (true, false)] {
            for start in [
                symphony::GrammarStart::Unknown,
                symphony::GrammarStart::Either,
                symphony::GrammarStart::AfterReasoning,
                symphony::GrammarStart::FirstToken,
            ] {
                for one_call in [true, false] {
                    let cell = format!(
                        "{}, open {reasoning_open}, closed {thought_closed}, {start:?}, one \
                             call {one_call}",
                        table.name()
                    );
                    let built = format.grammar_for(
                        &tools,
                        Options {
                            at_least_one: true,
                            reasoning_open,
                            thought_closed,
                            one_call,
                            start,
                        },
                    );
                    let free = built.as_ref().is_some_and(opens_with_free_text);
                    let found = calls_tag(
                        &format,
                        false,
                        "m",
                        &tools,
                        one_call,
                        reasoning_open,
                        thought_closed,
                        start,
                        TagVocabulary::Current,
                        false,
                    )
                    .expect("a registry table refuses nothing");
                    if free {
                        assert!(
                            reasoning_open
                                && matches!(
                                    start,
                                    symphony::GrammarStart::Unknown
                                        | symphony::GrammarStart::Either
                                ),
                            "{cell}: free text only from an open thought on workers that \
                                 name no one start"
                        );
                        assert!(
                            matches!(found, Prepared::OpensWithFreeText { .. }),
                            "{cell}: the tag is marked, not {found:?}"
                        );
                        marked += 1;
                    } else {
                        let stands = match &found {
                            Prepared::Constraint { table: true, .. } => true,
                            Prepared::None => false,
                            other => panic!("{cell}: {other:?}"),
                        };
                        assert_eq!(stands, built.is_some(), "{cell}: every other tag stands");
                    }
                }
            }
        }
    }
    assert!(
        marked > 0,
        "the sweep meets tables whose tag opens with free text"
    );
}

/// A forced call's JSON schema keeps a reference when its target there is its target in the
/// tool's own parameters: any local reference of a named function, whose parameters are the
/// schema; a `$defs` entry under the one schema for a list of tools, which lifts those and
/// nothing else. A reference to an entry of `definitions` or to the tool's own root under
/// that list, one that resolves nowhere, one that is no local pointer, and one whose target
/// the schema carries with another value are not kept.
/// A member named like a reference inside data (`const`, `enum`, `default`, `examples`) is
/// none; a property named like one of those keywords is still read.
#[test]
fn a_schema_keeps_a_reference_whose_target_it_carries() {
    let tool = |parameters: &Value| -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": "lookup", "parameters": parameters},
        }))
        .expect("a tool")
    };
    let referring = |reference: &str| {
        json!({
            "type": "object",
            "properties": {"place": {"$ref": reference}},
            "$defs": {"place": {"type": "string"}},
            "definitions": {"place": {"type": "string"}},
        })
    };
    // A named function's parameters are the schema as they are.
    let own = |parameters: &Value| ToolConstraint::JsonSchema(parameters.to_string());
    // A list's one schema nests the parameters and carries `defs` at its root.
    let listed = |parameters: &Value, defs: Value| {
        ToolConstraint::JsonSchema(
            json!({
                "type": "array",
                "items": {"anyOf": [{"properties": {"parameters": parameters}}]},
                "$defs": defs,
            })
            .to_string(),
        )
    };
    let place = || json!({"place": {"type": "string"}});
    for (reference, named, list) in [
        ("#/$defs/place", true, true),
        ("#/definitions/place", true, false),
        ("#", true, false),
        ("#/$defs/missing", false, false),
        ("https://example.com/place.json", false, false),
    ] {
        let parameters = referring(reference);
        let tools = [tool(&parameters)];
        assert_eq!(
            keeps_references(&own(&parameters), &tools),
            named,
            "{reference}: a named function"
        );
        assert_eq!(
            keeps_references(&listed(&parameters, place()), &tools),
            list,
            "{reference}: a list of tools"
        );
    }
    let parameters = referring("#/$defs/place");
    let other = json!({"place": {"type": "integer"}});
    assert!(
        !keeps_references(&listed(&parameters, other), &[tool(&parameters)]),
        "another value under the name"
    );
    for (parameters, list) in [
        (
            json!({"type": "object", "examples": [{"$ref": "literal"}]}),
            true,
        ),
        (
            json!({"type": "object", "default": {"$ref": "literal"}}),
            true,
        ),
        (
            json!({"type": "object", "properties": {"kind": {"const": {"$ref": "literal"}}}}),
            true,
        ),
        (
            json!({"type": "object", "properties": {"kind": {"enum": [{"$ref": "literal"}]}}}),
            true,
        ),
        (
            json!({
                "type": "object",
                "properties": {"default": {"$ref": "#/definitions/place"}},
                "definitions": {"place": {"type": "string"}},
            }),
            false,
        ),
        (
            json!({
                "type": "object",
                "properties": {"examples": {"$ref": "#/definitions/place"}},
                "definitions": {"place": {"type": "string"}},
            }),
            false,
        ),
    ] {
        let tools = [tool(&parameters)];
        assert!(
            keeps_references(&own(&parameters), &tools),
            "{parameters}: a named function"
        );
        assert_eq!(
            keeps_references(&listed(&parameters, json!({})), &tools),
            list,
            "{parameters}: a list of tools"
        );
    }
    let plain = json!({"type": "object"});
    assert!(
        keeps_references(
            &ToolConstraint::JsonSchema("not JSON".to_string()),
            &[tool(&plain)]
        ),
        "no reference: nothing to keep"
    );
}

/// A forced or required call on a table whose calls are given as data with JSON arguments
/// takes the table's tag and the table reads the answer: the request is not handed to the
/// older path. Under the older registry's JSON-schema constraint the table reads the answer
/// too, with its forced call.
#[test]
fn a_forced_call_on_the_kimi_k2_table_takes_its_tag_and_the_table_reads_the_answer() {
    let kimi = names(Some("kimi_k2"), Some("kimi_k2"));
    let prompt = "<|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|>";
    let named = request(json!({
        "model": "moonshotai/Kimi-K2-Thinking",
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }));
    let required = request(json!({
        "model": "moonshotai/Kimi-K2-Thinking",
        "tool_choice": "required",
    }));
    for (asked, one_call) in [(&named, true), (&required, false)] {
        let Ok(Prepared::Constraint {
            constraint: ToolConstraint::StructuralTag(tag),
            first_token,
            ..
        }) = super::constraint(
            ParserChoice::Symphony,
            &kimi,
            Asks::Chat(asked),
            &tools(),
            false,
            false,
            symphony::GrammarStart::AfterReasoning,
            TagVocabulary::Current,
            &templates(),
        )
        else {
            panic!("the table's tag, not the older registry's schema");
        };
        assert!(
            !first_token,
            "built for the gate behind the reasoning parser"
        );
        let format = serde_json::from_str::<Value>(&tag).expect("JSON");
        let format = &format["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(
            format["triggers"],
            json!(["<|tool_calls_section_begin|>", "</think>"])
        );
        assert_eq!(format["at_least_one"], true);
        assert_eq!(
            format["stop_after_first"],
            if one_call { json!(true) } else { Value::Null }
        );
        let call = if one_call {
            &format["tags"][0]["content"]["elements"][0]
        } else {
            &format["tags"][0]["content"]["content"]["elements"][0]
        };
        assert_eq!(
            call["elements"][0]["value"],
            "<|tool_call_begin|>functions.lookup:"
        );
        assert_eq!(format["tags"][0]["end"], "<|tool_calls_section_end|>");
        // The constraint is a structural tag, so the table is assigned to the response.
        let structural = ("structural_tag".to_string(), tag);
        let assigned = assign(
            ParserChoice::Symphony,
            &kimi,
            asked,
            Some(&structural),
            prompt,
        )
        .expect("assigned");
        assert_eq!(assigned.format.name(), "kimi_k2");
    }
    // The older registry's forced call (a JSON schema) is the table's to read too, with its
    // forced call: the thought as the table reads one, then the arguments alone.
    let schema = ("json_schema".to_string(), "{}".to_string());
    let forced = assign(ParserChoice::Symphony, &kimi, &named, Some(&schema), prompt)
        .expect("the table reads the forced call");
    assert_eq!(forced.format.name(), "kimi_k2");
    assert_eq!(forced.forcing.as_deref(), Some("lookup"));
}

/// A forced or required call on a table whose calls are named arguments inside a block takes
/// the table's tag and the table reads the answer: the request is not handed to the older
/// path. Under the older registry's JSON-schema constraint the table reads the answer too,
/// with its forced call.
#[test]
fn a_forced_call_on_the_deepseek_v3_1_table_takes_its_tag_and_the_table_reads_the_answer() {
    let deepseek = names(Some("deepseek31"), Some("deepseek_v31"));
    let prompt = "<｜User｜>hi<｜Assistant｜></think>";
    let named = request(json!({
        "model": "deepseek-ai/DeepSeek-V3.1",
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }));
    let required = request(json!({
        "model": "deepseek-ai/DeepSeek-V3.1",
        "tool_choice": "required",
    }));
    for (asked, one_call) in [(&named, true), (&required, false)] {
        let Ok(Prepared::Constraint {
            constraint: ToolConstraint::StructuralTag(tag),
            first_token,
            ..
        }) = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(asked),
            &tools(),
            false,
            false,
            symphony::GrammarStart::AfterReasoning,
            TagVocabulary::Current,
            &templates(),
        )
        else {
            panic!("the table's tag, not the older registry's schema");
        };
        assert!(
            !first_token,
            "built for the gate behind the reasoning parser"
        );
        let format = serde_json::from_str::<Value>(&tag).expect("JSON");
        let format = &format["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(
            format["triggers"],
            json!(["<｜tool▁calls▁begin｜>", "</think>"])
        );
        assert_eq!(format["at_least_one"], true);
        assert_eq!(
            format["stop_after_first"],
            if one_call { json!(true) } else { Value::Null }
        );
        let call = if one_call {
            &format["tags"][0]["content"]["elements"][0]
        } else {
            &format["tags"][0]["content"]["content"]["elements"][0]
        };
        assert_eq!(
            call["elements"][0]["value"],
            "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>"
        );
        assert_eq!(call["elements"][2]["value"], "<｜tool▁call▁end｜>");
        assert_eq!(format["tags"][0]["end"], "<｜tool▁calls▁end｜>");
        // The constraint is a structural tag, so the table is assigned to the response.
        let structural = ("structural_tag".to_string(), tag);
        let assigned = assign(
            ParserChoice::Symphony,
            &deepseek,
            asked,
            Some(&structural),
            prompt,
        )
        .expect("assigned");
        assert_eq!(assigned.format.name(), "deepseek_v3.1");
    }
    // The older registry's forced call (a JSON schema) is the table's to read too, with its
    // forced call: the thought as the table reads one, then the arguments alone.
    let schema = ("json_schema".to_string(), "{}".to_string());
    let forced = assign(
        ParserChoice::Symphony,
        &deepseek,
        &named,
        Some(&schema),
        prompt,
    )
    .expect("the table reads the forced call");
    assert_eq!(forced.format.name(), "deepseek_v3.1");
    assert_eq!(forced.forcing.as_deref(), Some("lookup"));
}

/// The model's workers name the start when they all carry the same label; none or a missing
/// label leaves it unknown, as does a pool of workers that start after their reasoning parser
/// beside ones that do not say, and a first-token worker beside another kind names either
/// kind. A worker's own label maps one to one.
#[test]
fn the_models_workers_name_the_grammar_start_when_they_agree() {
    let worker = |start: Option<&str>| -> Arc<dyn Worker> {
        let mut builder =
            BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
        if let Some(start) = start {
            builder = builder.label(WorkerGrammarStart::LABEL, start);
        }
        Arc::new(builder.build())
    };
    assert_eq!(grammar_start_of(&[]), symphony::GrammarStart::Unknown);
    assert_eq!(
        grammar_start_of(&[worker(None)]),
        symphony::GrammarStart::Unknown
    );
    assert_eq!(
        grammar_start_of(&[
            worker(Some("after_reasoning")),
            worker(Some("after_reasoning"))
        ]),
        symphony::GrammarStart::AfterReasoning
    );
    assert_eq!(
        grammar_start_of(&[worker(Some("first_token"))]),
        symphony::GrammarStart::FirstToken
    );
    assert_eq!(
        grammar_start_of(&[worker(Some("after_reasoning")), worker(Some("first_token"))]),
        symphony::GrammarStart::Either
    );
    // After the reasoning parser beside not saying: the unknown start's tags fit both (a
    // forced call's prefix, an answer's schema alone), so the pool needs no prefix.
    assert_eq!(
        grammar_start_of(&[worker(Some("after_reasoning")), worker(None)]),
        symphony::GrammarStart::Unknown
    );
    assert_eq!(
        grammar_start_of(&[worker(Some("first_token")), worker(None)]),
        symphony::GrammarStart::Either
    );
    assert_eq!(
        grammar_start_of(&[
            worker(Some("after_reasoning")),
            worker(None),
            worker(Some("first_token"))
        ]),
        symphony::GrammarStart::Either
    );
    // A prefill worker generates no token a grammar constrains: its label does not count.
    let prefill = |start: Option<&str>| -> Arc<dyn Worker> {
        let mut builder =
            BasicWorkerBuilder::new("grpc://prefill:50051").worker_type(WorkerType::Prefill);
        if let Some(start) = start {
            builder = builder.label(WorkerGrammarStart::LABEL, start);
        }
        Arc::new(builder.build())
    };
    assert_eq!(
        grammar_start_of(&[
            worker(Some("after_reasoning")),
            prefill(Some("first_token"))
        ]),
        symphony::GrammarStart::AfterReasoning
    );
    // An encode worker (EPD) generates none either: its label does not count.
    let encode = |start: Option<&str>| -> Arc<dyn Worker> {
        let mut builder =
            BasicWorkerBuilder::new("grpc://encode:50051").worker_type(WorkerType::Encode);
        if let Some(start) = start {
            builder = builder.label(WorkerGrammarStart::LABEL, start);
        }
        Arc::new(builder.build())
    };
    assert_eq!(
        grammar_start_of(&[worker(Some("after_reasoning")), encode(Some("first_token"))]),
        symphony::GrammarStart::AfterReasoning
    );
    assert_eq!(
        grammar_start_of(&[encode(Some("first_token"))]),
        symphony::GrammarStart::Unknown
    );
    assert_eq!(
        grammar_start_of(&[worker(Some("after_reasoning")), prefill(None)]),
        symphony::GrammarStart::AfterReasoning
    );
    assert_eq!(
        grammar_start_of(&[prefill(Some("first_token"))]),
        symphony::GrammarStart::Unknown
    );
    assert_eq!(
        grammar_start_from(WorkerGrammarStart::AfterReasoning),
        symphony::GrammarStart::AfterReasoning
    );
    assert_eq!(
        grammar_start_from(WorkerGrammarStart::FirstToken),
        symphony::GrammarStart::FirstToken
    );
    assert_eq!(
        grammar_start_from(WorkerGrammarStart::Unknown),
        symphony::GrammarStart::Unknown
    );
}

/// A prompt with no turn opener after a system message (as the checkpoint's own encoder file
/// writes one, or the template one a latest reminder follows) starts the model inside its
/// thought on the DeepSeek V4 table, and the
/// grammar built for the request reads it as the engine does, whatever the thinking switch
/// said: the answer tag of a JSON `response_format` takes the closed thought first where the
/// engine starts the grammar at the first token, and is the engine's own schema where it does
/// not say or starts after its reasoning parser's gate, as on a prompt that opened the
/// thought. A prompt that opened and closed the thought (thinking off) leaves the model in
/// the content, and the legacy choice asks the table nothing.
#[test]
fn a_turn_the_prompt_leaves_unopened_is_inside_the_thought_for_the_grammar_too() {
    let model = "deepseek-ai/DeepSeek-V4-Flash";
    let unopened = concat!(
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>Be brief.",
        "<\u{ff5c}User\u{ff5c}>Fix the failing test.# Environment\n - Platform: linux"
    );
    let thinking = "<\u{ff5c}User\u{ff5c}>q<\u{ff5c}Assistant\u{ff5c}><think>";
    let closed = "<\u{ff5c}User\u{ff5c}>q<\u{ff5c}Assistant\u{ff5c}></think>";
    let open =
        |parser, prompt| reasoning_open(parser, &names(None, None), model, &templates(), prompt);
    let is_open = |parser, prompt| open(parser, prompt).map(|thought| thought.open);
    assert_eq!(is_open(ParserChoice::Symphony, unopened), Some(true));
    assert_eq!(is_open(ParserChoice::Symphony, thinking), Some(true));
    assert_eq!(is_open(ParserChoice::Symphony, closed), Some(false));
    assert_eq!(is_open(ParserChoice::Legacy, unopened), None);
    // The JSON answer's tag on the unopened prompt is the thinking prompt's: the closed
    // thought first where the engine starts at the first token, and otherwise the schema
    // alone, the engine's own constraint and no tag; with the thought closed the schema alone
    // on every start.
    let request = request(json!({
        "model": model, "response_format": {"type": "json_object"}
    }));
    let answer = |prompt, start| {
        super::constraint(
            ParserChoice::Symphony,
            &names(None, None),
            Asks::Chat(&request),
            &[],
            is_open(ParserChoice::Symphony, prompt) == Some(true),
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            &templates(),
        )
        .and_then(|prepared| prepared.for_start(start, TagVocabulary::Current))
    };
    let first = tag_of(answer(unopened, symphony::GrammarStart::FirstToken));
    assert_eq!(
        first,
        tag_of(answer(thinking, symphony::GrammarStart::FirstToken))
    );
    assert_eq!(first["format"]["elements"][0]["end"], "</think>", "{first}");
    for start in [
        symphony::GrammarStart::Unknown,
        symphony::GrammarStart::AfterReasoning,
    ] {
        assert!(matches!(answer(unopened, start), Ok(None)), "{start:?}");
        assert!(matches!(answer(thinking, start), Ok(None)), "{start:?}");
        assert!(matches!(answer(closed, start), Ok(None)), "{start:?}");
    }
    assert!(matches!(
        answer(closed, symphony::GrammarStart::FirstToken),
        Ok(None)
    ));
}

/// The four places the workers of a model may start a grammar at.
const STARTS: [symphony::GrammarStart; 4] = [
    symphony::GrammarStart::Unknown,
    symphony::GrammarStart::Either,
    symphony::GrammarStart::AfterReasoning,
    symphony::GrammarStart::FirstToken,
];

/// The generation prompt of a table whose template leaves the thought to the model (MiniMax
/// M3's) with no marker, and the one that ends with the thought's close: the template closed
/// the thought in the model's place, as it does with thinking switched off.
const TURN_WITH_NO_MARKER: &str = "]~b]user\nhi[e~[\n]~b]ai\n";
const TURN_THAT_CLOSED_THE_THOUGHT: &str = "]~b]user\nhi[e~[\n]~b]ai\n</mm:think>";

/// What a request (`ask`, over the `lookup` tool unless it asks for a JSON answer) takes on
/// `prompt` from the table that reads `MiniMaxAI/MiniMax-M3`, for workers that start the
/// grammar at `start`, the request's thinking switch saying `switch`: the table's replay of
/// the prompt decides with it as preparation does (`PromptThought::or_switch`). An ask with
/// an `output_config` is a Messages request's, every other a chat request's.
fn found_where_the_thought_is_the_models(
    ask: &Value,
    prompt: &str,
    switch: bool,
    start: symphony::GrammarStart,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    let model = "MiniMaxAI/MiniMax-M3";
    let mut body = ask.clone();
    body["model"] = json!(model);
    let of_messages = !ask["output_config"].is_null();
    let (chat, messages);
    let asks = if of_messages {
        body["max_tokens"] = json!(8);
        body["messages"] = json!([{"role": "user", "content": "hi"}]);
        messages =
            serde_json::from_value::<CreateMessageRequest>(body).expect("a Messages request");
        Asks::for_messages(&messages, false)
    } else {
        chat = request(body);
        Asks::Chat(&chat)
    };
    let thought = reasoning_open(
        ParserChoice::Symphony,
        &names(None, None),
        model,
        &templates(),
        prompt,
    )
    .expect("a table reads");
    let (open, closed) = (thought.or_switch(|| switch), thought.closed);
    let tools = if ask["response_format"].is_null() && !of_messages {
        tools()
    } else {
        Vec::new()
    };
    if ask["tool_choice"] == "auto" {
        auto_constraint(
            ParserChoice::Symphony,
            &names(None, None),
            asks,
            &tools,
            open,
            closed,
            start,
            TagVocabulary::Current,
            &templates(),
        )
    } else {
        super::constraint(
            ParserChoice::Symphony,
            &names(None, None),
            asks,
            &tools,
            open,
            closed,
            start,
            TagVocabulary::Current,
            &templates(),
        )
        .and_then(|prepared| prepared.for_start(start, TagVocabulary::Current))
    }
}

/// The structural tag [`found_where_the_thought_is_the_models`] finds.
fn tag_where_the_thought_is_the_models(
    ask: &Value,
    prompt: &str,
    switch: bool,
    start: symphony::GrammarStart,
) -> Value {
    tag_of(found_where_the_thought_is_the_models(
        ask, prompt, switch, start,
    ))
}

/// A forced call by name and by `required`, the `auto` choice, and a JSON answer by
/// `json_object` and by `json_schema`: every ask that takes a tag from the table.
fn asks_that_take_a_tag() -> [Value; 5] {
    [
        json!({"tool_choice": {"type": "function", "function": {"name": "lookup"}}}),
        json!({"tool_choice": "required"}),
        json!({"tool_choice": "auto"}),
        json!({"response_format": {"type": "json_object"}}),
        json!({"response_format": {"type": "json_schema", "json_schema": {
            "name": "city",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }}}),
    ]
}

/// A prompt that ends with the thought's close, of a table whose template leaves the thought
/// to the model, is past the thought whatever the request's thinking switch says, and a
/// forced call, named or required, takes the block alone wherever the workers start the
/// grammar: no thought and no second close stand before it. A tag that offered the thought
/// again could hold the model there until its tokens ran out.
#[test]
fn a_forced_call_takes_the_block_alone_on_a_prompt_that_closed_the_thought() {
    let thought = reasoning_open(
        ParserChoice::Symphony,
        &names(None, None),
        "MiniMaxAI/MiniMax-M3",
        &templates(),
        TURN_THAT_CLOSED_THE_THOUGHT,
    )
    .expect("a table reads");
    assert!(thought.closed && !thought.open && !thought.alone);
    assert!(!thought.or_switch(|| true));
    assert!(!thought.or_switch(|| false));
    let named = json!({"tool_choice": {"type": "function", "function": {"name": "lookup"}}});
    let required = json!({"tool_choice": "required"});
    for start in STARTS {
        for ask in [&named, &required] {
            let unmarked =
                tag_where_the_thought_is_the_models(ask, TURN_WITH_NO_MARKER, false, start);
            assert_eq!(
                unmarked["format"]["elements"][0]["type"], "optional",
                "{start:?}: {unmarked}"
            );
            for switch in [false, true] {
                let closed = tag_where_the_thought_is_the_models(
                    ask,
                    TURN_THAT_CLOSED_THE_THOUGHT,
                    switch,
                    start,
                );
                assert_eq!(
                    closed["format"], unmarked["format"]["elements"][1],
                    "{start:?}, switch {switch}"
                );
            }
        }
    }
}

/// The bound of the rule above, which holds before it too. A prompt with no marker hears the
/// request's thinking switch and admits the thought the model may write before a forced
/// call, a JSON answer (`json_object` and `json_schema`) and the calls of the `auto` tag.
/// For a request that switched thinking off, the `auto` tag is the same on a prompt that
/// closed the thought as on one with no marker, wherever the workers start the grammar: the
/// forced call and the JSON answer read the closed thought, the `auto` tag does not.
#[test]
fn the_auto_tag_keeps_its_thought_on_a_prompt_that_closed_it() {
    let unmarked = reasoning_open(
        ParserChoice::Symphony,
        &names(None, None),
        "MiniMaxAI/MiniMax-M3",
        &templates(),
        TURN_WITH_NO_MARKER,
    )
    .expect("a table reads");
    assert!(!unmarked.closed && !unmarked.open && !unmarked.alone);
    assert!(unmarked.or_switch(|| true));
    assert!(!unmarked.or_switch(|| false));
    let the_models_thought = json!({
        "type": "optional",
        "content": {"type": "or", "elements": [
            {
                "type": "tag",
                "begin": "<mm:think>",
                "content": {"type": "any_text", "excludes": [
                    "<mm:think>", "</mm:think>", "<tool_call>", "</tool_call>",
                    "<invoke name=\"", "</invoke>", "]<]minimax[>["
                ]},
                "end": "</mm:think>"
            },
            {"type": "const_string", "value": "</mm:think>"}
        ]}
    });
    let asks = asks_that_take_a_tag();
    for start in STARTS {
        for ask in &asks {
            let unmarked =
                tag_where_the_thought_is_the_models(ask, TURN_WITH_NO_MARKER, false, start);
            assert_eq!(
                unmarked["format"]["elements"][0], the_models_thought,
                "{start:?}: {ask}"
            );
            if ask["tool_choice"] == "auto" {
                assert_eq!(
                    tag_where_the_thought_is_the_models(
                        ask,
                        TURN_THAT_CLOSED_THE_THOUGHT,
                        false,
                        start
                    ),
                    unmarked,
                    "{start:?}: {ask}"
                );
            }
        }
    }
}

/// A JSON answer on a prompt that closed the thought: the thought is over before the answer
/// starts, so the answer's tag (a chat request's `json_object` and `json_schema`, a Messages
/// request's `output_config.format`) owes the schema alone,
/// wherever the workers start the grammar and whatever the request's thinking switch says.
/// No thought and no second close stand before it, as before a forced call: the reader is
/// past the thought on such a prompt, and a thought the tag let through would come back as
/// content ahead of the JSON.
#[test]
fn a_json_answer_is_the_schema_alone_on_a_prompt_that_closed_the_thought() {
    let of_messages = json!({"output_config": {"format": {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }}}});
    let mut kept = Vec::new();
    for start in STARTS {
        for switch in [false, true] {
            for ask in asks_that_take_a_tag()
                .iter()
                .filter(|ask| !ask["response_format"].is_null())
                .chain([&of_messages])
            {
                // The schema alone is the engine's own constraint: the table sends no tag.
                let found = found_where_the_thought_is_the_models(
                    ask,
                    TURN_THAT_CLOSED_THE_THOUGHT,
                    switch,
                    start,
                );
                if !matches!(found, Ok(None)) {
                    kept.push(format!("{start:?}, the switch {switch}, {ask}: {found:?}"));
                }
            }
        }
    }
    assert!(
        kept.is_empty(),
        "JSON answers that take a tag on a prompt that closed the thought:\n{}",
        kept.join("\n")
    );
}

/// The corner of the two rules above: the request's thinking switch says the model thinks
/// while the prompt ends with the thought's close (a template closes the thought for a
/// request whose word the switch does not hear). The prompt decides, as it does for every
/// table whose template writes text after the close: a forced call, the `auto` tag and a
/// JSON answer take what the same prompt takes with the switch off, wherever the workers
/// start the grammar. Before, the switch decided and each was built as inside an open
/// thought.
#[test]
fn a_prompt_that_closed_the_thought_takes_the_tags_of_the_switch_off_under_any_switch() {
    let asks = asks_that_take_a_tag();
    let mut turned = Vec::new();
    for start in STARTS {
        for ask in &asks {
            let [off, on] =
                [false, true].map(|switch| {
                    match found_where_the_thought_is_the_models(
                        ask,
                        TURN_THAT_CLOSED_THE_THOUGHT,
                        switch,
                        start,
                    ) {
                        Ok(Some(ToolConstraint::StructuralTag(tag))) => tag,
                        other => format!("{other:?}"),
                    }
                });
            if on != off {
                turned.push(format!(
                    "{start:?}: {ask}\n  the switch off: {off}\n  the switch on: {on}"
                ));
            }
        }
    }
    assert!(
        turned.is_empty(),
        "{} of {} differ:\n{}",
        turned.len(),
        STARTS.len() * asks.len(),
        turned.join("\n")
    );
}

/// The reach of the rule over the whole registry. Every table is asked on prompts that end
/// with each of its markers and what its layout writes after the thought, and those that
/// read one closed are asked for a JSON answer both ways, with the closed reading and
/// without, wherever the workers start the grammar. The answer turns for the one table
/// that leaves the thought to the model and writes nothing after the close; a table whose
/// template writes text after the close reads the prompt closed too and keeps its tag. The
/// rule is by the table's rows and layout; the table is named here as a gate on purpose:
/// whoever registers a table the rule reaches names it here.
#[test]
fn a_closed_prompt_turns_the_json_answer_of_the_table_that_leaves_the_thought_to_the_model() {
    use ::symphony::formats::registry::Table;
    let mut closed = Vec::new();
    let mut reached = Vec::new();
    for table in Table::ALL {
        let name = table.name();
        let format = table.shared();
        let after = format
            .layout()
            .and_then(|layout| layout.thought.as_ref())
            .map_or("", |thought| &*thought.after);
        // A prompt that ends with a marker of the table and what its layout writes after
        // the thought: the table says which marker closed the thought.
        let reads_closed = format.terminal_texts().any(|marker| {
            format.thought_closed(&format!("the turn the prompt opened\n{marker}{after}"))
        });
        if !reads_closed {
            continue;
        }
        closed.push(name);
        let read = Reading::Table(Arc::clone(&format), false);
        for start in STARTS {
            for ask in asks_that_take_a_tag()
                .iter()
                .filter(|ask| !ask["response_format"].is_null())
            {
                let asked = request(ask.clone());
                let [without, with] = [false, true].map(|thought_closed| {
                    let found = constraint_for(
                        &read,
                        Asks::Chat(&asked),
                        &[],
                        false,
                        thought_closed,
                        start,
                        TagVocabulary::Current,
                    )
                    .and_then(|prepared| prepared.for_start(start, TagVocabulary::Current));
                    format!("{found:?}")
                });
                if without != with && !reached.contains(&name) {
                    reached.push(name);
                }
            }
        }
    }
    assert_eq!(
        reached,
        ["minimax_m3"],
        "of the tables that read the prompt closed: {closed:?}"
    );
    assert!(
        closed.len() > reached.len(),
        "no table whose template writes text after the close was asked: {closed:?}"
    );
}

#[test]
fn the_legacy_choice_assigns_nothing_and_asks_for_no_constraint() {
    assert!(assign(
        ParserChoice::Legacy,
        &ModelNames::default(),
        &request(json!({})),
        None,
        PROMPT
    )
    .is_none());
    let required = request(json!({"tool_choice": "required"}));
    assert!(constraint(
        ParserChoice::Legacy,
        &ModelNames::default(),
        &required,
        &tools(),
        false
    )
    .is_none());
}

#[test]
fn a_forced_call_takes_the_tables_structural_tag_and_auto_takes_none() {
    let required = request(json!({"tool_choice": "required"}));
    let Some(ToolConstraint::StructuralTag(tag)) = constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        &required,
        &tools(),
        false,
    ) else {
        panic!("a structural tag");
    };
    let tag: Value = serde_json::from_str(&tag).expect("JSON");
    assert!(tag["format"].is_object(), "{tag}");
    let auto = request(json!({"tool_choice": "auto"}));
    assert!(constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        &auto,
        &tools(),
        false
    )
    .is_none());
    let unknown = request(json!({"model": "unknown", "tool_choice": "required"}));
    assert!(constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        &unknown,
        &tools(),
        false
    )
    .is_none());
}

/// A JSON `response_format` takes the table's answer tag whatever names select the table,
/// unless the tag is the schema alone, the engine's own constraint, which the engine keeps as
/// before. A table whose answer is written inside a message (Muse Spark's, Kimi K3's,
/// Inkling's) takes its tag: the engine's schema writes bare JSON from the turn's own state,
/// which such a table drops as framing, so the content came back empty; where the thought is
/// a message of its own (Inkling's), the tag admits it first and leaves the way into the
/// answer to the engine unless it says it starts at the first token. A table whose output
/// starts in content takes the tag with the thought's close before the schema when the prompt
/// leaves the thought open, and with the thought the model may write first where its table
/// admits one (MiniMax M3's); a table with no answer state and a prompt that continues the
/// assistant's message keep the engine's schema.
#[test]
fn a_json_response_format_takes_the_tables_answer_tag_unless_it_is_the_schema_alone() {
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["city", "country", "population"],
        "additionalProperties": false
    });
    let muse = names(Some("muse_spark"), Some("muse_spark"));
    let any_object = json!({"model": "muse-spark", "response_format": {"type": "json_object"}});
    let answer = |names: &ModelNames<'_>, extra: Value, reasoning_open: bool| {
        super::constraint(
            ParserChoice::Symphony,
            names,
            Asks::Chat(&request(extra)),
            &[],
            reasoning_open,
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            &templates(),
        )
        .and_then(|prepared| {
            prepared.for_start(symphony::GrammarStart::Unknown, TagVocabulary::Current)
        })
    };
    // The named table, for a model the registry does not know: `json_object` asks for any
    // object, written in the message to the user.
    let tag = tag_of(answer(&muse, any_object.clone(), false));
    assert_eq!(tag["format"]["type"], "tag", "{tag}");
    assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
    assert_eq!(
        tag["format"]["content"],
        json!({"type": "json_schema", "json_schema": {"type": "object"}}),
        "{tag}"
    );
    // `json_schema` asks for the request's schema, strict or not.
    for strict in [true, false] {
        let tag = tag_of(answer(
            &muse,
            json!({
                "model": "muse-spark",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "capital", "schema": schema, "strict": strict}
                }
            }),
            false,
        ));
        assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
        assert_eq!(
            tag["format"]["content"]["json_schema"], schema,
            "strict {strict}: {tag}"
        );
    }
    // With the prompt inside the thought, the tag closes it before the message.
    let open = tag_of(answer(&muse, any_object.clone(), true));
    assert_ne!(open, tag, "{open}");
    let begin = open["format"]["begin"].as_str().unwrap_or_default();
    assert!(
        begin.ends_with(" to=user<|message|>") && begin.len() > " to=user<|message|>".len(),
        "{open}"
    );
    // The registry's tables whose answer is a message take theirs too: Kimi K3's is the
    // answer's tag alone.
    let kimi = tag_of(answer(
        &ModelNames::default(),
        json!({"model": "moonshotai/Kimi-K3", "response_format": {"type": "json_object"}}),
        false,
    ));
    assert_eq!(kimi["format"]["type"], "tag", "{kimi}");
    assert_eq!(kimi["format"]["content"]["type"], "json_schema", "{kimi}");
    // Inkling's thought is a message of its own: the tag admits it first, closed and the
    // next message opened with the text marker owed after it, or the marker alone, or
    // neither where the engine does not say it starts at the first token (an engine that
    // starts after its reasoning parser has consumed the model's own), then the schema and
    // the message's close.
    let inkling = tag_of(answer(
        &ModelNames::default(),
        json!({"model": "thinkingmachines/Inkling", "response_format": {"type": "json_object"}}),
        false,
    ));
    assert_eq!(inkling["format"]["type"], "sequence", "{inkling}");
    let elements = inkling["format"]["elements"]
        .as_array()
        .expect("a sequence");
    assert_eq!(elements.len(), 3, "{inkling}");
    assert_eq!(elements[0]["type"], "optional", "{inkling}");
    let ways = &elements[0]["content"];
    assert_eq!(ways["type"], "or", "{inkling}");
    let opener = json!({"type": "const_string", "value": "<|content_text|>"});
    assert_eq!(
        ways["elements"][0]["elements"][0]["begin"], "<|content_thinking|>",
        "{inkling}"
    );
    assert_eq!(ways["elements"][0]["elements"][1], opener, "{inkling}");
    assert_eq!(ways["elements"][1], opener, "{inkling}");
    assert_eq!(
        elements[1],
        json!({"type": "json_schema", "json_schema": {"type": "object"}}),
        "{inkling}"
    );
    assert_eq!(
        elements[2],
        json!({"type": "const_string", "value": "<|end_message|>"}),
        "{inkling}"
    );
    // A table whose output starts in content (the registry's Qwen3 table) has the schema
    // alone for its tag with the thought closed: the engine's own constraint, which stands.
    let content_first = json!({"response_format": {"type": "json_object"}});
    assert!(answer(&ModelNames::default(), content_first.clone(), false)
        .expect("read")
        .is_none());
    // With the prompt inside the thought and the engine silent on where it starts the
    // grammar, the schema stands alone as well: a free-text prefix would let the answer run
    // on unconstrained, and the engine's own constraint applies where the engine starts.
    assert!(answer(&ModelNames::default(), content_first, true)
        .expect("read")
        .is_none());
    // A table that admits a thought before its answer (MiniMax M3's) keeps that tag with the
    // thought closed: the thought the model may write first, then the schema.
    let thought_first = tag_of(answer(
        &ModelNames::default(),
        json!({"model": "MiniMaxAI/MiniMax-M3", "response_format": {"type": "json_object"}}),
        false,
    ));
    assert_eq!(
        thought_first["format"]["type"], "sequence",
        "{thought_first}"
    );
    assert_eq!(
        thought_first["format"]["elements"][0]["type"], "optional",
        "{thought_first}"
    );
    assert_eq!(
        thought_first["format"]["elements"][1],
        json!({"type": "json_schema", "json_schema": {"type": "object"}}),
        "{thought_first}"
    );
    // No answer state (xLAM's list), a continued assistant message, or no JSON asked: the
    // engine's schema, or nothing.
    assert!(answer(
        &names(Some("xlam"), None),
        json!({"response_format": {"type": "json_object"}}),
        false
    )
    .expect("read")
    .is_none());
    let continued = json!({
        "model": "muse-spark",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{"}],
        "continue_final_message": true, "add_generation_prompt": false,
    });
    assert!(request(continued.clone()).continues_final_assistant());
    assert!(answer(&muse, continued, false).expect("read").is_none());
    assert!(answer(
        &muse,
        json!({"model": "muse-spark", "response_format": {"type": "text"}}),
        false
    )
    .expect("read")
    .is_none());
}

/// For a worker on `always`, the calls a request may make under `auto` take the table's tag
/// with no call required. Nothing is refused: a request with no tools, `tool_choice: none`,
/// a constraint of its own (a JSON `response_format`, a `regex`), or a prompt that continues
/// the assistant's message takes none, and so does a forced Messages choice.
#[test]
fn an_always_worker_takes_the_tables_tag_for_the_calls_auto_allows() {
    let auto = request(json!({"tool_choice": "auto"}));
    let tag = auto_constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::Chat(&auto),
        &tools(),
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates(),
    )
    .expect("no hf");
    let Some(ToolConstraint::StructuralTag(tag)) = tag else {
        panic!("a structural tag");
    };
    let tag: Value = serde_json::from_str(&tag).expect("JSON");
    assert!(tag["format"].is_object(), "{tag}");
    let Some(ToolConstraint::StructuralTag(forced)) = constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        &request(json!({"tool_choice": "required"})),
        &tools(),
        false,
    ) else {
        panic!("a forced tag");
    };
    let forced: Value = serde_json::from_str(&forced).expect("JSON");
    assert_ne!(tag, forced, "the auto tag requires no call");
    assert!(auto_constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::Chat(&auto),
        &[],
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates()
    )
    .expect("no hf")
    .is_none());
    assert!(auto_constraint(
        ParserChoice::Legacy,
        &ModelNames::default(),
        Asks::Chat(&auto),
        &tools(),
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates()
    )
    .expect("legacy")
    .is_none());
    assert!(auto_constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::Chat(&request(json!({"tool_choice": "none"}))),
        &tools(),
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates()
    )
    .expect("no hf")
    .is_none());
    for own in [
        json!({"tool_choice": "auto", "response_format": {"type": "json_object"}}),
        json!({"tool_choice": "auto", "regex": "[a-z]+"}),
        json!({
            "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "I was saying"}
            ],
            "continue_final_message": true, "add_generation_prompt": false
        }),
    ] {
        assert!(
            auto_constraint(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Chat(&request(own.clone())),
                &tools(),
                false,
                false,
                symphony::GrammarStart::Unknown,
                TagVocabulary::Current,
                &templates()
            )
            .expect("no hf")
            .is_none(),
            "{own}"
        );
    }
    let messages_auto = message_request(json!({
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
    }));
    let tools: Vec<Tool> = serde_json::from_value(json!([{
        "type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}
    }]))
    .unwrap();
    assert!(matches!(
        auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::for_messages(&messages_auto, false),
            &tools,
            false,
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            &templates()
        )
        .expect("no hf"),
        Some(ToolConstraint::StructuralTag(_))
    ));
    assert!(auto_constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::for_messages(&messages_auto, true),
        &tools,
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates()
    )
    .expect("no hf")
    .is_none());
    let messages_any = message_request(json!({
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"}
    }));
    assert!(auto_constraint(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::for_messages(&messages_any, false),
        &tools,
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates()
    )
    .expect("no hf")
    .is_none());
}

#[tokio::test]
async fn the_templates_table_gives_a_forced_call_and_a_json_answer_their_tags_or_refuses() {
    let model = "Qwen/Qwen3-8B";
    let with_layout =
        template::fixtures::templates_with(model, template::fixtures::qwen3_5_tokenizer()).await;
    let without_layout =
        template::fixtures::templates_with(model, template::fixtures::unrendered_tokenizer()).await;
    let hf = names(Some("hf"), Some("hf"));
    let constrain = |templates: &Templates, extra: Value| {
        super::constraint(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&request(extra)),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            templates,
        )
        .and_then(|prepared| {
            prepared.for_start(symphony::GrammarStart::Unknown, TagVocabulary::Current)
        })
    };
    // A forced call, one call for a named function, and a JSON answer take the table's tags.
    let required = tag_of(constrain(&with_layout, json!({"tool_choice": "required"})));
    assert!(required["format"].is_object(), "{required}");
    let named = tag_of(constrain(
        &with_layout,
        json!({"tool_choice": {"type": "function", "function": {"name": "lookup"}}}),
    ));
    assert_ne!(named, required, "one call ends where the turn does");
    let answer = tag_of(constrain(
        &with_layout,
        json!({"response_format": {"type": "json_object"}}),
    ));
    assert!(answer["format"].is_object(), "{answer}");
    // Nothing asked, nothing constrained.
    assert!(constrain(&with_layout, json!({"tool_choice": "auto"}))
        .expect("read")
        .is_none());
    // Without a layout the table has no tag for a call: the template's table refuses, where
    // a registry table would leave the older registry to decide. Its content opens with no
    // marker, so a JSON answer is written from the first byte, layout or not.
    let why = constrain(&without_layout, json!({"tool_choice": "required"})).expect_err("no tag");
    assert!(
        matches!(
            &why,
            SymphonyError::Constraint(text)
                if text.contains("no structural tag for a forced tool choice")
        ),
        "{why}"
    );
    let answer = tag_of(constrain(
        &without_layout,
        json!({"response_format": {"type": "json_object"}}),
    ));
    assert_eq!(answer["format"]["type"], "json_schema", "{answer}");
    let plain = names(None, None);
    assert!(
        super::constraint(
            ParserChoice::Symphony,
            &plain,
            Asks::Chat(&request(
                json!({"response_format": {"type": "json_object"}})
            )),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            &templates(),
        )
        .and_then(
            |prepared| prepared.for_start(symphony::GrammarStart::Unknown, TagVocabulary::Current)
        )
        .expect("read")
        .is_none(),
        "the Qwen3 table's answer tag is the schema alone, the engine's own constraint, so the \
             older registry's JSON schema stands"
    );
    // A regex or ebnf constraint would apply outside the template's format.
    let why = constrain(
        &with_layout,
        json!({"tool_choice": "required", "regex": "a+"}),
    )
    .expect_err("refused");
    assert!(
        matches!(&why, SymphonyError::Constraint(text) if text.contains("regex or ebnf")),
        "{why}"
    );
    // The tag writes a turn from its start: a continued assistant message is refused.
    let continued = json!({
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "I"}],
        "continue_final_message": true, "add_generation_prompt": false,
    });
    assert!(
        request(continued.clone()).continues_final_assistant(),
        "the request continues"
    );
    let why = constrain(&with_layout, continued).expect_err("refused");
    assert!(
        matches!(
            &why,
            SymphonyError::Constraint(text)
                if text.contains("cannot continue the assistant's message")
        ),
        "{why}"
    );
}

#[tokio::test]
async fn messages_are_read_by_the_same_tables_and_their_tool_choice_forces_calls() {
    let symphony = ParserChoice::Symphony;
    let plain = ModelNames::default();
    let read = reads(symphony, &plain, "Qwen/Qwen3-8B", &templates())
        .expect("read")
        .expect("the registry's table");
    assert_eq!(read.name(), "qwen3");
    assert!(
        reads(ParserChoice::Legacy, &plain, "Qwen/Qwen3-8B", &templates())
            .expect("read")
            .is_none()
    );
    // A tool_choice of none over no tools takes the table without its calls, as every
    // request without tools does; over declared tools it keeps them, for the reader to drop.
    let no_calls = super::super::assign(
        symphony,
        &plain,
        Asks::for_messages(
            &message_request(json!({"tool_choice": {"type": "none"}})),
            false,
        ),
        None,
        PROMPT,
        &templates(),
    )
    .expect("read")
    .expect("assigned");
    assert_eq!(no_calls.format.name(), "qwen3");
    assert_eq!(no_calls.format.call_syntax(), None);
    assert_eq!(no_calls.forbids_calls, Forbids::Nothing);
    let forbidden = super::super::assign(
        symphony,
        &plain,
        Asks::for_messages(
            &message_request(json!({
                "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
                "tool_choice": {"type": "none"}
            })),
            false,
        ),
        None,
        PROMPT,
        &templates(),
    )
    .expect("read")
    .expect("assigned");
    assert_eq!(forbidden.format.name(), "qwen3");
    assert!(
        forbidden.format.call_syntax().is_some(),
        "the table keeps its calls under none over declared tools"
    );
    assert_eq!(forbidden.forbids_calls, Forbids::Every);
    // Past the route the request's own `tools` reads `None` (the route took them out while it
    // owned the request): the declaration the spec recorded is what the table reads.
    let taken = super::super::assign(
        symphony,
        &plain,
        Asks::Messages {
            request: &message_request(json!({"tool_choice": {"type": "none"}})),
            continues: false,
            has_tools: true,
        },
        None,
        PROMPT,
        &templates(),
    )
    .expect("read")
    .expect("assigned");
    assert!(
        taken.format.call_syntax().is_some(),
        "the declaration is read from the ask, not from the request's emptied field"
    );
    assert_eq!(taken.forbids_calls, Forbids::Every);
    let constrain = |extra: Value| {
        super::constraint(
            symphony,
            &plain,
            Asks::for_messages(&message_request(extra), false),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            TagVocabulary::Current,
            &templates(),
        )
        .and_then(|prepared| {
            prepared.for_start(symphony::GrammarStart::Unknown, TagVocabulary::Current)
        })
    };
    let any = tag_of(constrain(json!({"tool_choice": {"type": "any"}})));
    let one = tag_of(constrain(
        json!({"tool_choice": {"type": "any", "disable_parallel_tool_use": true}}),
    ));
    let tool = tag_of(constrain(
        json!({"tool_choice": {"type": "tool", "name": "lookup"}}),
    ));
    assert_ne!(any, one, "one call ends where the turn does");
    assert_eq!(tool, one, "a named tool is one call");
    assert!(constrain(json!({"tool_choice": {"type": "auto"}}))
        .expect("read")
        .is_none());
    assert!(constrain(json!({})).expect("read").is_none());
    let assigned = super::super::assign(
        symphony,
        &plain,
        Asks::for_messages(&message_request(json!({})), false),
        None,
        PROMPT,
        &templates(),
    )
    .expect("read")
    .expect("assigned");
    assert_eq!(assigned.format.name(), "qwen3");
    assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
    // The template's table reads messages too, and refuses what it cannot tag.
    let hf = names(Some("hf"), None);
    let unrendered = template::fixtures::templates_with(
        "Qwen/Qwen3-8B",
        template::fixtures::unrendered_tokenizer(),
    )
    .await;
    let why = super::constraint(
        symphony,
        &hf,
        Asks::for_messages(
            &message_request(json!({"tool_choice": {"type": "any"}})),
            false,
        ),
        &tools(),
        false,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &unrendered,
    )
    .expect_err("no tag");
    assert!(
        matches!(&why, SymphonyError::Constraint(text) if text.contains("no structural tag")),
        "{why}"
    );
}

/// The decision says when its constraint is shaped for an engine that starts the grammar at
/// the first token: the `auto` tag derived for that start, and the whole turn's grammar; a tag
/// derived for another start, or no tag, is not.
#[test]
fn the_decision_says_when_its_constraint_starts_at_the_first_token() {
    use ConstrainedDecoding::{Always, Requested};
    use GrammarStart::{AfterReasoning, FirstToken, Unknown};

    let templates = Arc::new(templates());
    let auto_tag = || AutoTag {
        parser: ParserChoice::Symphony,
        labels: pair(Some("deepseek_v4"), Some("deepseek_v4")),
        flags: Names::default(),
        by_alias: None,
        request: AutoRequest::Chat(Arc::new(request(json!({"tools": tools()})))),
        reasoning_open: true,
        thought_closed: false,
        templates: Arc::clone(&templates),
    };
    let decided = |policy, start, turn: Option<ToolConstraint>| {
        worker_constraint(
            WorkerDecoding {
                policy,
                start,
                vocabulary: TagVocabulary::Current,
            },
            Ok(Prepared::None),
            || Ok(turn),
            auto_tag,
        )
        .expect("decided")
    };
    for start in [FirstToken, AfterReasoning, Unknown] {
        let auto = decided(Always, start, None);
        assert!(auto.constraint.is_some(), "{start:?}: the auto tag");
        assert_eq!(auto.first_token, start == FirstToken, "{start:?}");
        assert!(auto.table, "{start:?}: the table built the auto tag");
        let none = decided(Requested, start, None);
        assert!(
            none.constraint.is_none() && !none.first_token && !none.table,
            "{start:?}: nothing shaped"
        );
    }
    let turn = decided(
        Always,
        FirstToken,
        Some(ToolConstraint::StructuralTag("turn".to_string())),
    );
    assert!(turn.whole_turn && turn.first_token && turn.table);
    // A prepared constraint speaks for itself, whatever the worker's start: the table's
    // forced-call tag built for a pool that starts at the first token says so; the older
    // registry's says nothing here, since request building shapes it and speaks for it.
    for start in [FirstToken, AfterReasoning, Unknown] {
        for built_for_first_token in [true, false] {
            let prepared = worker_constraint(
                WorkerDecoding {
                    policy: Requested,
                    start,
                    vocabulary: TagVocabulary::Current,
                },
                Ok(Prepared::Constraint {
                    constraint: ToolConstraint::StructuralTag("prepared".to_string()),
                    first_token: built_for_first_token,
                    table: true,
                }),
                || Ok(None),
                auto_tag,
            )
            .expect("decided");
            assert!(
                prepared.constraint.is_some() && prepared.table,
                "{start:?}: kept as it is, the table's"
            );
            assert_eq!(
                prepared.first_token, built_for_first_token,
                "{start:?}: as the table built it, not as the worker starts"
            );
        }
    }
}

#[test]
fn the_selected_workers_policy_decides_once() {
    use ConstrainedDecoding::{Always, Requested};
    use GrammarStart::{AfterReasoning, FirstToken, Unknown};

    let tag = |text: &str| ToolConstraint::StructuralTag(text.to_string());
    let refusal = |why: &str| SymphonyError::Constraint(why.to_string());
    // The request's `auto` tag: DeepSeek V4's table over one tool, the thought left open, so
    // the tag takes a shape for each start.
    let templates = Arc::new(templates());
    let auto_tag = || AutoTag {
        parser: ParserChoice::Symphony,
        labels: pair(Some("deepseek_v4"), Some("deepseek_v4")),
        flags: Names::default(),
        by_alias: None,
        request: AutoRequest::Chat(Arc::new(request(json!({"tools": tools()})))),
        reasoning_open: true,
        thought_closed: false,
        templates: Arc::clone(&templates),
    };
    // What one decision gives (the tag's text, whether it is the whole turn's, whether the
    // policy chose it, and whether the `auto` tag goes with the plan), and which of the whole
    // turn's grammar and the `auto` tag it built.
    let decide = |policy,
                  start,
                  prepared: PreparedConstraint,
                  turn: Result<Option<ToolConstraint>, SymphonyError>| {
        let built = std::cell::RefCell::new(Vec::new());
        let decided = worker_constraint(
            WorkerDecoding {
                policy,
                start,
                vocabulary: TagVocabulary::Current,
            },
            prepared,
            || {
                built.borrow_mut().push("turn");
                turn
            },
            || {
                built.borrow_mut().push("auto");
                auto_tag()
            },
        )
        .map(|decided| {
            let text = decided.constraint.map(|constraint| constraint.to_tuple().1);
            (
                text,
                decided.whole_turn,
                decided.by_policy,
                decided.table_tag.is_some(),
            )
        });
        (decided, built.into_inner())
    };
    // What preparation found is the request's choice.
    let found = |text: &str| Ok((Some(text.to_string()), false, false, false));
    // The `auto` tag is the policy's, in the shape the worker's start takes, and goes with the
    // plan for a retry's worker.
    let auto = |start| {
        let text = auto_tag()
            .for_start(grammar_start_from(start), TagVocabulary::Current)
            .expect("a table")
            .map(|constraint| constraint.to_tuple().1);
        assert!(text.is_some(), "{start:?}");
        Ok((text, false, true, true))
    };

    // `requested`: what preparation found, its refusal included; nothing else is built.
    for start in [FirstToken, AfterReasoning, Unknown] {
        assert_eq!(
            decide(
                Requested,
                start,
                Ok(Prepared::Constraint {
                    constraint: tag("prepared"),
                    first_token: false,
                    table: true,
                }),
                Ok(Some(tag("turn")))
            ),
            (found("prepared"), vec![])
        );
        assert_eq!(
            decide(
                Requested,
                start,
                Err(refusal("no tag")),
                Ok(Some(tag("turn")))
            ),
            (Err(SymphonyError::Constraint("no tag".to_string())), vec![])
        );
        assert_eq!(
            decide(Requested, start, Ok(Prepared::None), Ok(Some(tag("turn")))),
            (Ok((None, false, false, false)), vec![])
        );
    }
    // `always` from the first token: the whole turn's grammar stands over what preparation
    // found or refused, and its refusal refuses the request.
    assert_eq!(
        decide(
            Always,
            FirstToken,
            Err(refusal("no tag")),
            Ok(Some(tag("turn")))
        ),
        (
            Ok((Some("turn".to_string()), true, true, false)),
            vec!["turn"]
        )
    );
    assert_eq!(
        decide(
            Always,
            FirstToken,
            Ok(Prepared::None),
            Err(refusal("cannot hold"))
        ),
        (
            Err(SymphonyError::Constraint("cannot hold".to_string())),
            vec!["turn"]
        )
    );
    // Where the whole turn's grammar does not start, and on an engine that starts the grammar
    // later or does not say: what preparation found or refused, else the `auto` tag.
    assert_eq!(
        decide(
            Always,
            FirstToken,
            Ok(Prepared::Constraint {
                constraint: tag("prepared"),
                first_token: false,
                table: true,
            }),
            Ok(None)
        ),
        (found("prepared"), vec!["turn"])
    );
    assert_eq!(
        decide(Always, FirstToken, Err(refusal("no tag")), Ok(None)),
        (
            Err(SymphonyError::Constraint("no tag".to_string())),
            vec!["turn"]
        )
    );
    assert_eq!(
        decide(Always, FirstToken, Ok(Prepared::None), Ok(None)),
        (auto(FirstToken), vec!["turn", "auto"])
    );
    for start in [AfterReasoning, Unknown] {
        assert_eq!(
            decide(Always, start, Err(refusal("no tag")), Ok(Some(tag("turn")))),
            (Err(SymphonyError::Constraint("no tag".to_string())), vec![])
        );
        assert_eq!(
            decide(
                Always,
                start,
                Ok(Prepared::Constraint {
                    constraint: tag("prepared"),
                    first_token: false,
                    table: true,
                }),
                Ok(Some(tag("turn")))
            ),
            (found("prepared"), vec![])
        );
        assert_eq!(
            decide(Always, start, Ok(Prepared::None), Ok(Some(tag("turn")))),
            (auto(start), vec!["auto"])
        );
    }
    // Where the table gives no `auto` tag (no tool to call), the policy chose nothing.
    let nothing = worker_constraint(
        WorkerDecoding {
            policy: Always,
            start: AfterReasoning,
            vocabulary: TagVocabulary::Current,
        },
        Ok(Prepared::None),
        || Ok(None),
        || AutoTag {
            parser: ParserChoice::Symphony,
            labels: pair(Some("deepseek_v4"), Some("deepseek_v4")),
            flags: Names::default(),
            by_alias: None,
            request: AutoRequest::Chat(Arc::new(request(json!({})))),
            reasoning_open: true,
            thought_closed: false,
            templates: Arc::clone(&templates),
        },
    )
    .unwrap();
    assert!(nothing.constraint.is_none() && !nothing.by_policy && nothing.table_tag.is_none());
    assert_eq!(
        decide(Requested, Unknown, Ok(Prepared::None), Ok(None)).0,
        Ok((None, false, false, false))
    );
}

/// A JSON answer's tag derives for the selected worker's own start, not for the pool's: from
/// one mixed pool (a first-token worker beside one that does not say, which names either
/// kind), a worker that starts after its reasoning parser or does not say gets the schema
/// alone, the engine's own constraint, and the first-token worker the closed thought then the
/// schema. The derivation goes with the plan, and the worker a retry takes gets its own shape
/// from it, the request's own schema standing in where the table gives it no tag. A forced
/// call's tag keeps the pool's start, and its prefix for a pool of either kind.
#[test]
fn a_json_answers_tag_derives_for_the_selected_worker_and_the_worker_a_retry_takes() {
    use GrammarStart::{AfterReasoning, FirstToken, Unknown};
    let worker = |start: Option<&str>| -> Arc<dyn Worker> {
        let mut builder =
            BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
        if let Some(start) = start {
            builder = builder.label(WorkerGrammarStart::LABEL, start);
        }
        Arc::new(builder.build())
    };
    let pool = [worker(Some("first_token")), worker(None)];
    assert_eq!(grammar_start_of(&pool), symphony::GrammarStart::Either);
    let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let asked = request(json!({"response_format": {
        "type": "json_schema", "json_schema": {"name": "city", "schema": schema},
    }}));
    let prepared = super::constraint(
        ParserChoice::Symphony,
        &deepseek,
        Asks::Chat(&asked),
        &[],
        true,
        false,
        grammar_start_of(&pool),
        vocabulary_of(&pool),
        &templates(),
    );
    let Ok(Prepared::Answer(answer)) = prepared else {
        panic!("the answer's tag, derived per worker, not {prepared:?}");
    };
    // Decided on the selected worker: each kind its own shape.
    let decided = |start| {
        worker_constraint(
            WorkerDecoding {
                policy: ConstrainedDecoding::Requested,
                start,
                vocabulary: TagVocabulary::Current,
            },
            Ok(Prepared::Answer(Arc::clone(&answer))),
            || panic!("requested: the whole turn is not asked"),
            || panic!("requested: the auto tag is not asked"),
        )
        .expect("decided")
    };
    for start in [AfterReasoning, Unknown] {
        let decided = decided(start);
        assert!(
            decided.constraint.is_none(),
            "{start:?}: the schema alone, the engine's own constraint, not {:?}",
            decided.constraint
        );
        assert!(!decided.by_policy && !decided.whole_turn, "{start:?}");
        assert!(
            !decided.table,
            "{start:?}: no tag, so nothing the table built"
        );
        let stamp = decided
            .table_tag
            .expect("the derivation goes with the plan");
        assert_eq!(
            stamp
                .tag_for(start, TagVocabulary::Current)
                .expect("a table"),
            None,
            "{start:?}"
        );
        assert_eq!(stamp.schema(), Some(&schema), "{start:?}");
    }
    let first = decided(FirstToken);
    let Some(ToolConstraint::StructuralTag(text)) = first.constraint else {
        panic!("the closed thought then the schema: {:?}", first.constraint);
    };
    let tag: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(tag["format"]["elements"][0]["end"], "</think>", "{tag}");
    assert_eq!(tag["format"]["elements"][1]["type"], "json_schema", "{tag}");
    assert_eq!(tag["format"]["elements"][1]["json_schema"], schema, "{tag}");
    assert!(!first.by_policy && !first.whole_turn);
    assert!(first.table, "the tag is the table's");
    // The worker a retry takes, of the other kind, and back.
    let stamp = first.table_tag.expect("the derivation goes with the plan");
    assert_eq!(
        stamp
            .tag_for(AfterReasoning, TagVocabulary::Current)
            .expect("a table"),
        None
    );
    assert_eq!(
        stamp
            .tag_for(Unknown, TagVocabulary::Current)
            .expect("a table"),
        None
    );
    assert_eq!(
        stamp
            .tag_for(FirstToken, TagVocabulary::Current)
            .expect("a table")
            .as_deref(),
        Some(text.as_str())
    );
    assert_eq!(stamp.schema(), Some(&schema));
    // The forced call's tag keeps the pool's start: the prefix that fits either kind, which
    // is free text, so the tag comes marked.
    let forced = request(json!({"tool_choice": "required"}));
    let Ok(Prepared::OpensWithFreeText {
        constraint: ToolConstraint::StructuralTag(call),
        first_token,
    }) = super::constraint(
        ParserChoice::Symphony,
        &deepseek,
        Asks::Chat(&forced),
        &tools(),
        true,
        false,
        grammar_start_of(&pool),
        vocabulary_of(&pool),
        &templates(),
    )
    else {
        panic!("a forced tag, marked");
    };
    let call: Value = serde_json::from_str(&call).expect("JSON");
    assert_eq!(call["format"]["elements"][0]["type"], "any_text", "{call}");
    assert!(
        !first_token,
        "the pool's start is either: not built for the first token"
    );
}

/// The request's `auto` tag derives again for where another worker's engine starts the
/// grammar, as a retry needs it: the free-text prefix before the calls for an engine that does
/// not say, the bare block for one that starts after its reasoning parser's gate, the closed
/// thought then the block for one that starts at the first token; each the tag request
/// building gives a worker of that start. For a chat and a Messages request alike.
#[test]
fn the_auto_tag_derives_again_for_the_start_of_the_worker_a_retry_takes() {
    use WorkerGrammarStart::{AfterReasoning, FirstToken, Unknown};
    let templates = Arc::new(templates());
    let names = names(Some("deepseek_v4"), Some("deepseek_v4"));
    let chat = request(json!({"tools": tools()}));
    let messages = message_request(json!({
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
    }));
    let tags = [
        AutoTag {
            parser: ParserChoice::Symphony,
            labels: names.labels.clone(),
            flags: Names::default(),
            by_alias: None,
            request: AutoRequest::Chat(Arc::new(chat.clone())),
            reasoning_open: true,
            thought_closed: false,
            templates: Arc::clone(&templates),
        },
        AutoTag {
            parser: ParserChoice::Symphony,
            labels: names.labels.clone(),
            flags: Names::default(),
            by_alias: None,
            request: AutoRequest::Messages {
                request: Arc::new(messages.clone()),
                tools: tools().into(),
                continues: false,
                has_tools: messages.tools.is_some(),
            },
            reasoning_open: true,
            thought_closed: false,
            templates: Arc::clone(&templates),
        },
    ];
    for auto_tag in &tags {
        let built = |start| {
            let constraint = match &auto_tag.request {
                AutoRequest::Chat(_) => auto_constraint(
                    ParserChoice::Symphony,
                    &names,
                    Asks::Chat(&chat),
                    &chat.callable_tools(),
                    true,
                    false,
                    start,
                    TagVocabulary::Current,
                    &templates,
                ),
                AutoRequest::Messages { .. } => auto_constraint(
                    ParserChoice::Symphony,
                    &names,
                    Asks::for_messages(&messages, false),
                    &tools(),
                    true,
                    false,
                    start,
                    TagVocabulary::Current,
                    &templates,
                ),
            };
            tag_of(constraint)
        };
        let derived = |start| {
            let tag = TagForStart::tag_for(auto_tag, start, TagVocabulary::Current)
                .expect("a table")
                .expect("a tag");
            serde_json::from_str::<Value>(&tag).expect("a tag as JSON")
        };
        let unknown = derived(Unknown);
        assert_eq!(
            unknown,
            built(symphony::GrammarStart::Unknown),
            "{auto_tag:?}"
        );
        assert_eq!(unknown["format"]["elements"][0]["type"], "any_text");
        let bare = derived(AfterReasoning);
        assert_eq!(bare, built(symphony::GrammarStart::AfterReasoning));
        assert_ne!(bare["format"]["elements"][0]["type"], "any_text", "{bare}");
        let first = derived(FirstToken);
        assert_eq!(first, built(symphony::GrammarStart::FirstToken));
        assert_eq!(first["format"]["elements"][0]["end"], "</think>", "{first}");
        assert_eq!(first["format"]["elements"][1], bare["format"]);
    }
    // A tag the table gives no engine (no tool to call) derives as none for every start.
    let none = AutoTag {
        parser: ParserChoice::Symphony,
        labels: names.labels,
        flags: Names::default(),
        by_alias: None,
        request: AutoRequest::Chat(Arc::new(request(json!({})))),
        reasoning_open: true,
        thought_closed: false,
        templates,
    };
    for start in [Unknown, AfterReasoning, FirstToken] {
        assert_eq!(
            TagForStart::tag_for(&none, start, TagVocabulary::Current),
            Ok(None)
        );
    }
    assert!(format!("{none:?}").contains("Qwen/Qwen3-8B"));
}

#[test]
fn the_whole_turn_holds_the_answer_and_the_calls() {
    let free = whole(json!({}), None);
    assert_eq!(
        after_thought(&free)["elements"][1],
        json!({"type": "any_text", "excludes": ["<|done|>", "<|stop|>"]})
    );
    assert_eq!(
        after_thought(&whole(json!({"tools": weather()}), None))["type"],
        "or"
    );
    let required = whole(json!({"tools": weather(), "tool_choice": "required"}), None);
    assert_eq!(after_thought(&required)["type"], "sequence");
    let json_answer = whole(json!({"response_format": {"type": "json_object"}}), None);
    assert_eq!(
        after_thought(&json_answer)["elements"][1],
        json!({"type": "json_schema", "json_schema": {"type": "object"}})
    );
    let regex = whole(json!({"regex": "(yes|no)"}), None);
    assert_eq!(
        after_thought(&regex)["elements"][1],
        json!({"type": "regex", "pattern": "(yes|no)"})
    );
    let ebnf = whole(json!({"ebnf": "root ::= \"y\""}), None);
    assert_eq!(after_thought(&ebnf)["elements"][1]["type"], "grammar");
    // A forced call writes no answer, so it leaves the answer's format out, Lark's included.
    assert_eq!(
        turn(
            json!({"tools": weather(), "tool_choice": "required", "ebnf": "start: \"y\""}),
            None
        ),
        Ok(Some(required))
    );
}

#[test]
fn the_prompt_says_where_the_turn_starts() {
    let thinking = whole(json!({}), Some("<|via|>notes<|body|>Let me"));
    assert_eq!(thinking["elements"][0]["begin"], "");
    let after = whole(
        json!({}),
        Some("<|via|>notes<|body|>Plan.<|stop|><|head|>assistant"),
    );
    assert_eq!(after["elements"][0]["value"], "<|via|>reply<|body|>");

    // Inside an answer: the rest of a free one; a forced call or an answer's format starts
    // no grammar there, and the table's tags decide (`worker_constraint`).
    let answering = Some("<|via|>reply<|body|>The answer");
    assert_eq!(
        whole(json!({}), answering)["elements"][0]["type"],
        "any_text"
    );
    for extra in [
        json!({"tools": weather(), "tool_choice": "required"}),
        json!({"tools": weather(), "tool_choice": {"type": "allowed_tools",
            "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}}),
        json!({"response_format": {"type": "json_object"}}),
    ] {
        assert_eq!(turn(extra, answering), Ok(None));
    }
    // Past text no field holds, no grammar starts.
    assert_eq!(turn(json!({"tools": weather()}), Some("Hello")), Ok(None));
    // Inside a call: the rest of it is free.
    let calling = " to=get_weather<|via|>notes <|as|>xml<|body|><arg name=\"city\">\nPa";
    assert_eq!(
        whole(json!({"tools": weather()}), Some(calling))["elements"][0]["type"],
        "any_text"
    );
}

#[test]
fn the_whole_turn_refuses_what_it_cannot_hold() {
    let refused = |extra: Value| turn(extra, None).is_err();
    let served = |extra: Value| turn(extra, None).expect("held").is_some();
    let named = |name: &str| {
        json!([{"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}])
    };
    // A name a call cannot carry, whatever the choice.
    for choice in [json!("auto"), json!("none"), json!("required")] {
        assert!(refused(
            json!({"tools": named("get weather"), "tool_choice": choice})
        ));
    }
    let key = json!([{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"say \"hi\"": {"type": "string"}}}}}]);
    assert!(refused(json!({"tools": key})));

    // A strict tool whose calls the grammar would hold more loosely, while it is callable.
    let tuple = |strict: bool| {
        json!([{"type": "function", "function": {"name": "f", "strict": strict,
            "parameters": {"type": "object", "properties": {"p": {"type": "tuple"}},
                "required": ["p"]}}}])
    };
    assert!(refused(json!({"tools": tuple(true)})));
    assert!(served(json!({"tools": tuple(false)})));
    assert!(served(json!({"tools": tuple(true), "tool_choice": "none"})));

    // Under `auto`, a strict tool or one call at most beside the request's own grammar; and
    // an ebnf in Lark's syntax.
    let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    for extra in [
        json!({"tools": strict, "regex": "(a|b)"}),
        json!({"tools": strict, "ebnf": "root ::= \"y\""}),
        json!({"tools": weather(), "parallel_tool_calls": false, "regex": "(a|b)"}),
        json!({"ebnf": "start: \"y\""}),
        json!({"tools": weather(), "ebnf": "start: \"y\""}),
    ] {
        assert!(refused(extra.clone()), "{extra}");
    }
    for extra in [
        json!({"tools": strict}),
        json!({"tools": strict, "response_format": {"type": "json_object"}}),
        json!({"tools": strict, "tool_choice": "required", "regex": "(a|b)"}),
        json!({"tools": weather(), "regex": "(a|b)"}),
        json!({"tools": weather(), "parallel_tool_calls": false}),
        json!({"regex": "(a|b)", "parallel_tool_calls": false}),
    ] {
        assert!(served(extra.clone()), "{extra}");
    }
    let why = turn(json!({"ebnf": "start: \"y\""}), None).expect_err("refused");
    assert!(why.contains("Lark"), "{why}");
}

#[test]
fn over_its_limits_the_whole_turn_leaves_the_tags_to_decide() {
    let properties: serde_json::Map<String, Value> = (0
        ..=symphony::OutputSpelling::MAX_TOOL_PARAMETERS)
        .map(|i| (format!("p{i}"), json!({"type": "string"})))
        .collect();
    let wide = json!([{"type": "function", "function": {"name": "wide",
        "parameters": {"type": "object", "properties": properties}}}]);
    assert_eq!(turn(json!({"tools": wide}), None), Ok(None));

    let huge = "x".repeat(symphony::OutputSpelling::MAX_TAG_BYTES);
    let large = json!([{"type": "function", "function": {"name": "large",
    "parameters": {"type": "object", "properties": {
        "p": {"type": "string", "enum": [huge]}
    }}}}]);
    assert_eq!(turn(json!({"tools": large}), None), Ok(None));
}

#[test]
fn a_messages_choice_takes_the_turn_its_chat_twin_takes() {
    let tools = json!([{"name": "get_weather", "input_schema": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
    let prompt = "user<|head|>assistant";
    for (choice, chat) in [
        (json!({"type": "auto"}), json!({})),
        (json!({"type": "any"}), json!({"tool_choice": "required"})),
        (
            json!({"type": "any", "disable_parallel_tool_use": true}),
            json!({"tool_choice": "required", "parallel_tool_calls": false}),
        ),
        (
            json!({"type": "tool", "name": "get_weather"}),
            json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
        ),
        (json!({"type": "none"}), json!({"tool_choice": "none"})),
    ] {
        let messages = message_request(json!({"tools": tools, "tool_choice": choice}));
        let mut chat = chat;
        chat["tools"] = weather();
        let tag = |constraint: Result<Option<ToolConstraint>, SymphonyError>| {
            tag_of(constraint)["format"].clone()
        };
        let chat = request(chat);
        assert_eq!(
            tag(turn_of(
                &turn_table(),
                Asks::for_messages(&messages, false),
                &messages.chat_tools(),
                prompt,
                TagVocabulary::Current,
            )),
            tag(turn_of(
                &turn_table(),
                Asks::Chat(&chat),
                &chat.effective_tool_list(),
                prompt,
                TagVocabulary::Current,
            )),
            "{choice}"
        );
    }
}

#[test]
fn only_the_response_templates_table_spells_a_whole_turn() {
    // Any other name, and the legacy parsers, leave the turn to the table's tags without
    // looking for a template.
    let request = request(json!({"tools": weather(), "tool_choice": "required"}));
    for (parser, names) in [
        (ParserChoice::Symphony, names(Some("qwen"), None)),
        (ParserChoice::Symphony, names(None, None)),
        (ParserChoice::Legacy, names(Some("hf"), None)),
    ] {
        assert!(matches!(
            turn_constraint(
                parser,
                &names,
                Asks::Chat(&request),
                &request.effective_tool_list(),
                PROMPT,
                TagVocabulary::Current,
                &templates()
            ),
            Ok(None)
        ));
    }
    // The template's names ask for the model's own template, and a model without one is
    // refused, as the table's tags refuse it.
    let why = turn_constraint(
        ParserChoice::Symphony,
        &names(Some("hf"), None),
        Asks::Chat(&request),
        &request.effective_tool_list(),
        PROMPT,
        TagVocabulary::Current,
        &templates(),
    )
    .expect_err("refused");
    assert_eq!(
        why,
        SymphonyError::Template("no tokenizer is registered for model 'Qwen/Qwen3-8B'".to_string())
    );
}

/// The model's workers name the vocabulary of structural-tag nodes a forced call's tag is
/// spelled in together: the current one unless a worker that generates tokens advertises the
/// older one, which every compiler takes, so one such worker beside others takes it for all;
/// a prefill or encode worker's label does not count.
#[test]
fn the_models_workers_name_the_vocabulary_when_one_takes_the_older() {
    use WorkerType::{Decode, Encode, Prefill, Regular};
    let worker = |kind: WorkerType, vocabulary: Option<&str>| -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new("grpc://engine:50051").worker_type(kind);
        if let Some(vocabulary) = vocabulary {
            builder = builder.label(TagVocabulary::LABEL, vocabulary);
        }
        Arc::new(builder.build())
    };
    assert_eq!(vocabulary_of(&[]), TagVocabulary::Current);
    assert_eq!(
        vocabulary_of(&[worker(Regular, None)]),
        TagVocabulary::Current
    );
    assert_eq!(
        vocabulary_of(&[worker(Regular, Some("current"))]),
        TagVocabulary::Current
    );
    assert_eq!(
        vocabulary_of(&[worker(Regular, Some("older"))]),
        TagVocabulary::Older
    );
    assert_eq!(
        vocabulary_of(&[worker(Regular, None), worker(Decode, Some("older"))]),
        TagVocabulary::Older
    );
    assert_eq!(
        vocabulary_of(&[
            worker(Regular, None),
            worker(Prefill, Some("older")),
            worker(Encode, Some("older"))
        ]),
        TagVocabulary::Current
    );
}

/// A worker that advertises the older vocabulary gets its tags spelled without the
/// `optional`, `plus` and `star` nodes its compiler refuses (a forced call's, the `auto`
/// tag's, the whole turn's), the same language in the nodes it takes
/// ([`Grammar::in_older_vocabulary`]: a tag without those nodes may still lose the excludes
/// its tag end makes redundant); a worker that does not advertise it gets every tag as the
/// table built it (the tests above read those).
#[test]
fn a_worker_of_the_older_vocabulary_gets_its_tags_without_the_newer_nodes() {
    fn newer_nodes(value: &Value) -> usize {
        match value {
            Value::Object(map) => {
                let own = usize::from(matches!(
                    map.get("type").and_then(Value::as_str),
                    Some("optional" | "plus" | "star")
                ));
                own + map.values().map(newer_nodes).sum::<usize>()
            }
            Value::Array(items) => items.iter().map(newer_nodes).sum(),
            _ => 0,
        }
    }
    let parsed = |constraint: Result<Option<ToolConstraint>, SymphonyError>| -> Value {
        let Ok(Some(ToolConstraint::StructuralTag(text))) = constraint else {
            panic!("a structural tag, not {constraint:?}");
        };
        serde_json::from_str(&text).expect("JSON")
    };
    let qwen = ModelNames::default();
    let kimi = names(Some("kimi_k2"), Some("kimi_k2"));
    let forced = request(json!({"tool_choice": "required"}));
    let one = request(json!({"tool_choice": "required", "parallel_tool_calls": false}));
    let auto = request(json!({"tool_choice": "auto"}));
    let turn = request(json!({"tools": weather(), "tool_choice": "required"}));
    let tags = |vocabulary: TagVocabulary| -> Vec<(String, Value)> {
        let mut tags = Vec::new();
        for (table, names) in [("qwen3", &qwen), ("kimi_k2", &kimi)] {
            for (ask, request) in [("required", &forced), ("one call", &one)] {
                for reasoning_open in [false, true] {
                    for (start_name, start) in [
                        ("unknown", symphony::GrammarStart::Unknown),
                        ("first token", symphony::GrammarStart::FirstToken),
                    ] {
                        let calls = super::constraint(
                            ParserChoice::Symphony,
                            names,
                            Asks::Chat(request),
                            &tools(),
                            reasoning_open,
                            false,
                            start,
                            vocabulary,
                            &templates(),
                        )
                        .and_then(|prepared| prepared.for_start(start, vocabulary));
                        tags.push((
                            format!("{table}, {ask}, thought open {reasoning_open}, {start_name}"),
                            parsed(calls),
                        ));
                    }
                }
            }
            let allowed = auto_constraint(
                ParserChoice::Symphony,
                names,
                Asks::Chat(&auto),
                &tools(),
                false,
                false,
                symphony::GrammarStart::Unknown,
                vocabulary,
                &templates(),
            );
            tags.push((format!("{table}, auto"), parsed(allowed)));
        }
        let whole = turn_of(
            &turn_table(),
            Asks::Chat(&turn),
            &turn.effective_tool_list(),
            "user<|head|>assistant",
            vocabulary,
        );
        tags.push(("the whole turn".to_string(), parsed(whole)));
        tags
    };
    let current = tags(TagVocabulary::Current);
    let older = tags(TagVocabulary::Older);
    let mut spelled = 0;
    for ((name, built), (_, older)) in current.iter().zip(&older) {
        assert_eq!(newer_nodes(older), 0, "{name}: {older}");
        if newer_nodes(built) > 0 {
            assert_ne!(older, built, "{name}");
            spelled += 1;
        }
    }
    assert!(spelled > 0, "no tag carried a newer node");
}

/// The older registry's constraint, which preparation takes where the table gives a forced
/// call no tag, is not the table's ([`Prepared::from`]): the decision says so, whatever the
/// constraint's kind, and Symphony's assignment then reads it by its kind alone.
#[test]
fn the_older_registrys_constraint_is_not_the_tables() {
    use ConstrainedDecoding::Requested;
    use WorkerGrammarStart::Unknown;
    for constraint in [
        ToolConstraint::JsonSchema("{}".to_string()),
        ToolConstraint::StructuralTag("{}".to_string()),
    ] {
        let decided = worker_constraint(
            WorkerDecoding {
                policy: Requested,
                start: Unknown,
                vocabulary: TagVocabulary::Current,
            },
            Ok(Prepared::from(Some(constraint.clone()))),
            || panic!("requested: the whole turn is not asked"),
            || panic!("requested: the auto tag is not asked"),
        )
        .expect("decided");
        assert_eq!(
            decided.constraint.map(|decided| decided.to_tuple()),
            Some(constraint.to_tuple())
        );
        assert!(!decided.table && !decided.first_token && !decided.by_policy);
    }
}

/// A choice forbids exactly the calls outside the tools it leaves callable ([`callable`]),
/// for every form of the choice; `auto` narrows nothing and forbids nothing.
#[test]
fn a_choice_forbids_exactly_the_calls_outside_its_callable_tools() {
    let tools = json!([
        {"type": "function", "function": {"name": "save_note", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "delete_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
    ]);
    let choices = [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type": "function", "function": {"name": "save_note"}}),
        json!({"type": "allowed_tools", "mode": "auto", "tools": [
            {"type": "function", "name": "save_note"}, {"type": "function", "name": "lookup"}]}),
        json!({"type": "allowed_tools", "mode": "required", "tools": [
            {"type": "function", "name": "delete_file"}]}),
    ];
    for choice in choices {
        let request = request(json!({"tools": tools.clone(), "tool_choice": choice.clone()}));
        let declared = request.tools.as_deref().expect("the declared tools");
        let forbids = Asks::Chat(&request).forbids_for(ProviderProfile::for_model(&request.model));
        let callable = callable(request.tool_choice.as_ref(), declared);
        for function in ["save_note", "delete_file", "lookup", "undeclared"] {
            let is = |tools: &[Tool]| tools.iter().any(|tool| tool.function.name == function);
            let forbidden = forbids != Forbids::Nothing && !is(&callable);
            assert_eq!(
                forbids.call_to(function, is(declared)),
                forbidden,
                "{choice}: {function}"
            );
        }
        assert_eq!(
            forbids.over(declared),
            forbids,
            "{choice}: over declared tools"
        );
        let over_none = match forbids {
            Forbids::Nothing => Forbids::Nothing,
            _ => Forbids::Every,
        };
        assert_eq!(forbids.over(&[]), over_none, "{choice}: over no tool");
    }
}

/// A Messages choice forbids as the chat choice it converts to: `tool` all but the tool it
/// names, `any` the tools the request did not declare, `none` every call, `auto` none.
#[test]
fn a_messages_choice_forbids_as_its_chat_form() {
    let forbids = |choice: Value| {
        let request = message_request(json!({
            "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
            "tool_choice": choice
        }));
        Asks::for_messages(&request, false).forbids_for(ProviderProfile::for_model(&request.model))
    };
    assert_eq!(
        forbids(json!({"type": "tool", "name": "lookup"})),
        Forbids::AllBut(["lookup".to_string()].into())
    );
    assert_eq!(forbids(json!({"type": "any"})), Forbids::Undeclared);
    assert_eq!(forbids(json!({"type": "none"})), Forbids::Every);
    assert_eq!(forbids(json!({"type": "auto"})), Forbids::Nothing);
}
