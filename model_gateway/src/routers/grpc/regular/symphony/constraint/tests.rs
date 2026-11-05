use tool_parser::GrammarStart;

use super::*;
use crate::routers::grpc::regular::symphony::testing::{assign, constraint, *};

/// Where a worker's engine starts the grammar shapes what precedes a table's answer or its
/// calls when the prompt leaves the thought open: nothing after the reasoning parser's gate
/// (a JSON answer then needs no tag; the request's own schema goes to the engine as it is),
/// the thought closed first from the first token, the schema alone (the request's own, no
/// tag) when the worker does not say, and for the calls the free-text prefix when the
/// model's workers name different starts. The answer's tag derives for the selected worker,
/// whatever start the pool names.
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
    assert_eq!(
        call_of(symphony::GrammarStart::Unknown)["format"]["elements"][0]["type"],
        "any_text"
    );
    let bare = call_of(symphony::GrammarStart::AfterReasoning);
    assert_eq!(
        bare["format"]["elements"][0]["type"], "const_string",
        "the block's opener comes first at the gate: {bare}"
    );
    let first = call_of(symphony::GrammarStart::FirstToken);
    assert_eq!(first["format"]["elements"][0]["end"], "</think>");
    assert_eq!(first["format"]["elements"][1], bare["format"]);
}

/// A forced or required call on a table whose calls are given as data with JSON arguments
/// takes the table's tag and the table reads the answer: the request is not handed to the
/// older path. The older path's JSON-schema constraint still leaves the table out, as before.
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
    // The older path's forced call (a JSON schema) is still read by the older parsers.
    let schema = ("json_schema".to_string(), "{}".to_string());
    assert!(assign(ParserChoice::Symphony, &kimi, &named, Some(&schema), prompt).is_none());
}

/// A forced or required call on a table whose calls are named arguments inside a block takes
/// the table's tag and the table reads the answer: the request is not handed to the older
/// path. The older path's JSON-schema constraint still leaves the table out, as before.
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
    // The older path's forced call (a JSON schema) is still read by the older parsers.
    let schema = ("json_schema".to_string(), "{}".to_string());
    assert!(assign(
        ParserChoice::Symphony,
        &deepseek,
        &named,
        Some(&schema),
        prompt
    )
    .is_none());
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
            Asks::Messages {
                request: &messages_auto,
                continues: false
            },
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
        Asks::Messages {
            request: &messages_auto,
            continues: true
        },
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
        Asks::Messages {
            request: &messages_any,
            continues: false
        },
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
        Asks::Messages {
            request: &message_request(json!({"tool_choice": {"type": "none"}})),
            continues: false,
        },
        None,
        PROMPT,
        &templates(),
    )
    .expect("read")
    .expect("assigned");
    assert_eq!(no_calls.format.name(), "qwen3");
    assert_eq!(no_calls.format.call_syntax(), None);
    assert!(!no_calls.forbids_calls);
    let forbidden = super::super::assign(
        symphony,
        &plain,
        Asks::Messages {
            request: &message_request(json!({
                "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
                "tool_choice": {"type": "none"}
            })),
            continues: false,
        },
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
    assert!(forbidden.forbids_calls);
    let constrain = |extra: Value| {
        super::constraint(
            symphony,
            &plain,
            Asks::Messages {
                request: &message_request(extra),
                continues: false,
            },
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
        Asks::Messages {
            request: &message_request(json!({})),
            continues: false,
        },
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
        Asks::Messages {
            request: &message_request(json!({"tool_choice": {"type": "any"}})),
            continues: false,
        },
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
        let none = decided(Requested, start, None);
        assert!(
            none.constraint.is_none() && !none.first_token,
            "{start:?}: nothing shaped"
        );
    }
    let turn = decided(
        Always,
        FirstToken,
        Some(ToolConstraint::StructuralTag("turn".to_string())),
    );
    assert!(turn.whole_turn && turn.first_token);
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
                }),
                || Ok(None),
                auto_tag,
            )
            .expect("decided");
            assert!(prepared.constraint.is_some(), "{start:?}: kept as it is");
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
    // The forced call's tag keeps the pool's start: the prefix that fits either kind.
    let forced = request(json!({"tool_choice": "required"}));
    let Ok(Prepared::Constraint {
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
        panic!("a forced tag");
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
            request: AutoRequest::Chat(Arc::new(chat.clone())),
            reasoning_open: true,
            thought_closed: false,
            templates: Arc::clone(&templates),
        },
        AutoTag {
            parser: ParserChoice::Symphony,
            labels: names.labels.clone(),
            flags: Names::default(),
            request: AutoRequest::Messages {
                request: Arc::new(messages.clone()),
                tools: tools().into(),
                continues: false,
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
                    Asks::Messages {
                        request: &messages,
                        continues: false,
                    },
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
                Asks::Messages {
                    request: &messages,
                    continues: false
                },
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
