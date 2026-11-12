//! What the chat and Messages stages decide alike about how the response is
//! read (see `regular::symphony`): in preparation, the constraint the request
//! asks for before its worker is known and whether the decoder keeps the
//! special tokens the reader needs ([`prepare`]); in request building, whether
//! the rendered prompt leaves the model inside its thought
//! ([`reasoning_decision`]), what constrains the output on the selected worker
//! and which table reads the response ([`decide`]). The two stages spell what
//! their request asks ([`Asks`]) and decide through these functions; each
//! keeps what differs between the APIs, the tokenizer-driven reasoning switch
//! and the request the `auto` tag derives for.

use std::sync::Arc;

use ::symphony::formats::{forced, json};
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    common::{Tool, ToolChoice},
    profile::ProviderProfile,
    worker::ToolConstraint,
};
use serde_json::Value;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::helpers::TagForStart,
        context::RequestContext,
        regular::symphony::{
            self, Asks, Assignment, AutoRequest, PreparedConstraint, PromptReplay, Reading,
        },
        utils::{self, Names},
    },
};

/// What preparation found for the response.
pub(crate) struct Prepared {
    /// The constraint for what the request asks, before its worker is known:
    /// the table's tag for a forced call, or the JSON schema the engine writes
    /// the call under where the table has no tag for it ([`forced_call_schema`]);
    /// a JSON answer's tag as it derives for a worker's start, or its schema
    /// itself where the table gives a Messages request's answer no tag; or why
    /// the table cannot constrain the request. Request building decides on the
    /// selected worker (`regular::symphony::worker_constraint`).
    pub constraint: PreparedConstraint,
    /// The constraint is a JSON schema: the output is the JSON alone, with no
    /// marker, which the `forced`, `json` or `plain` table reads
    /// (`regular::symphony::assign_for`).
    pub json_schema: bool,
    /// The decoder keeps the special tokens the table's markers need.
    pub keeps_special_tokens: bool,
    /// The parser names the model's cards set, resolved here once for the request: the stage
    /// stores them (`ProcessingState::parser_names`) and request building reads them in
    /// place of a second resolution.
    pub names: Names,
    /// The table's replay of the rendered prompt, when preparation replayed it for the forced
    /// call's tag: the stage stores it (`ProcessingState::prompt_thought`) and request building's
    /// reasoning decision reads it in place of a second replay of the same prompt. `None` when
    /// the request's thinking switch armed the reasoning and no replay was needed.
    pub prompt_thought: Option<PromptReplay>,
    /// The table decision for the request (`regular::symphony::reading`: which table reads the
    /// response), read here once for the replay, the constraint and the decoder's special
    /// tokens: the stage stores it (`ProcessingState::reading`) and request building's assignment
    /// reads it in place of a fourth decision over the same names; the whole turn's grammar and
    /// the `auto` tag still decide for the selected worker. `None` when the names refused (the
    /// constraint carries the refusal).
    pub reading: Option<Reading>,
}

/// The constraint preparation finds for what the request asks, and whether
/// the decoder keeps the special tokens the reader needs. The table that reads
/// the response gives a forced call, or a JSON `response_format`, its tag, so
/// the output comes back in the syntax the table parses
/// (`regular::symphony::constraint_for`, over the table decision the reader is
/// assigned by; the answer's tag derives for the selected worker at request
/// building). Where the table has no tag for the call the choice forces, the
/// engine writes the call under a JSON schema instead ([`forced_call_schema`])
/// and the `forced` or `json` table reads the output
/// (`regular::symphony::assign_for`). On a thinking prompt a table with a
/// reasoning prefix gets its tag wrapped so a forced call follows the
/// reasoning instead of preempting it: `switch` is the API's tokenizer-driven
/// answer, and the table's replay of the rendered prompt decides beside it (a
/// turn the prompt leaves unopened starts inside the thought whatever the
/// switch says: `regular::symphony::reasoning_open`), so the tag reads the
/// prompt as the engine will. The table's refusal waits for request building,
/// which decides on the selected worker; a schema two tools define one name
/// differently under refuses the request here (a 400, logged under `stage`).
/// A table whose markers the decoder would drop as special tokens needs them
/// kept whatever the request asks of the channels: without them the table
/// reads neither the thought nor the calls
/// (`regular::symphony::keeps_special_tokens_for`); under a JSON schema the
/// output has no marker, and the decoder keeps none.
pub(crate) fn prepare(
    ctx: &RequestContext,
    asks: Asks<'_>,
    tools: &[Tool],
    processed_text: &str,
    switch: bool,
    tokenizer: &Arc<dyn Tokenizer>,
    stage: &'static str,
) -> Result<Prepared, Response> {
    let model = asks.model();
    // The model's workers, fetched once for the request: the cards' parser
    // names are read among them here, and a forced call's tag takes the
    // start they name together; a JSON answer's tag and the `auto` tag take
    // the selected worker's own start at request building.
    let workers = ctx
        .components
        .worker_registry
        .get_by_model(&ctx.input.model_id);
    let names = ctx.components.parser_resolver.names_among(&workers, model);
    // Which table reads the response, decided once for the request: the
    // replay, the constraint and the decoder's special tokens read it here,
    // the assignment at request building after them. A refusal (the names
    // ask for a response template the model cannot compile) is the
    // constraint's to carry.
    let reading = symphony::reading(&names, model, &ctx.components.templates);
    let prompt_thought = (!switch).then(|| {
        reading
            .as_ref()
            .ok()
            .map(|read| symphony::reasoning_open_for(read, processed_text))
    });
    let replay = prompt_thought.flatten();
    let reasoning = switch || replay.is_some_and(|thought| thought.open);
    // The prompt closed the thought itself and wrote the template's text
    // after it (thinking switched off): the calls' tag owes none of it.
    let thought_closed = replay.is_some_and(|thought| thought.closed);
    let from_the_table = reading.as_ref().map_err(Clone::clone).and_then(|read| {
        symphony::constraint_for(
            read,
            asks,
            tools,
            reasoning,
            thought_closed,
            symphony::grammar_start_of(&workers),
            symphony::vocabulary_of(&workers),
        )
    });
    // The table has no tag for the call the choice forces: the engine writes
    // the call under the JSON schema instead, and the `forced` or `json`
    // table reads the output.
    let constraint = match from_the_table {
        Ok(symphony::Prepared::None) => {
            let schema = forced_call_schema(tools, asks.choice().as_deref()).map_err(|why| {
                error!(function = stage, error = %why, "Invalid tool configuration");
                error::bad_request(
                    "invalid_tool_configuration",
                    format!("Invalid tool configuration: {why}"),
                )
            })?;
            Ok(symphony::Prepared::from(schema.map(|schema| {
                ToolConstraint::JsonSchema(schema.to_string())
            })))
        }
        prepared => prepared,
    };
    // Where nothing above constrains the answer, a Messages request's answer
    // schema is the engine's own JSON-schema constraint, as a chat request's
    // `response_format` is through its request builder; a table that reads
    // the response gave the schema its tag above (`Prepared::Answer`), and a
    // request whose choice forces a call takes none (the turn has no text
    // answer to format, whether or not the call has a constraint).
    let constraint = match constraint {
        Ok(symphony::Prepared::None) => Ok(symphony::Prepared::from(
            asks.untabled_answer_schema(tools)
                .map(|schema| ToolConstraint::JsonSchema(schema.to_string())),
        )),
        prepared => prepared,
    };
    let json_schema = matches!(
        &constraint,
        Ok(symphony::Prepared::Constraint { constraint, .. }) if constraint.is_json_schema()
    );
    let keeps_special_tokens = !json_schema
        && reading.as_ref().ok().is_some_and(|read| {
            symphony::keeps_special_tokens_for(read, model, &ctx.components.templates, tokenizer)
        });
    Ok(Prepared {
        constraint,
        json_schema,
        keeps_special_tokens,
        names: names.labels,
        prompt_thought,
        reading: reading.ok(),
    })
}

/// The JSON schema the engine writes the call `choice` forces under, where the
/// table that reads the response has no tag for it: the named function's
/// `parameters` (`symphony::formats::forced::arguments_schema`; the `forced`
/// table reads the output), or for `required`, and for allowed tools in
/// `required` mode, a list of at least one call to one of `tools`, the ones the
/// choice lets the model call (`symphony::formats::json::required_calls_schema`;
/// the `json` table reads it). `None` for a choice that forces no call, and for
/// a named function `tools` does not declare (a tool of another kind, which the
/// model never sees). An error when two tools define one `$defs` name
/// differently, which one list cannot hold.
fn forced_call_schema(
    tools: &[Tool],
    choice: Option<&ToolChoice>,
) -> Result<Option<Value>, String> {
    match choice {
        Some(ToolChoice::Function { function, .. }) => {
            Ok(forced::arguments_schema(tools, &function.name))
        }
        Some(choice) if choice.forces_call() && !tools.is_empty() => {
            json::required_calls_schema(tools).map(Some)
        }
        _ => Ok(None),
    }
}

/// Whether the rendered prompt leaves the model inside its thought, as the
/// request's grammar, the plan's reasoning state and the response spec all
/// read it: the table's own replay of the prompt (a turn the prompt leaves
/// unopened starts inside the thought: `regular::symphony::reasoning_open`)
/// with the tokenizer-driven `switch` (the API's:
/// `utils::chat_reasoning_starts_in_rendered_prefill`,
/// `utils::messages_reasoning_starts_in_prefill`, over `model`'s tokenizer),
/// or alone where the family's rule sets the switch aside (Gemma 4 prefills
/// an open channel only after a tool response); the switch alone where the
/// names refuse the model. With it, whether the prompt closed the thought
/// itself and wrote the template's text after it (`PromptThought::closed`),
/// which the `auto` tag then owes nothing of.
pub(crate) fn reasoning_decision(
    ctx: &RequestContext,
    model: &str,
    processed_text: &str,
    switch: impl FnOnce(&dyn Tokenizer) -> bool,
) -> (bool, bool) {
    let switch = || {
        ctx.tokenizer_arc()
            .is_some_and(|tokenizer| switch(tokenizer.as_ref()))
    };
    // Preparation's replay of the same prompt stands: the names are the
    // request's (`parser_names`), so the table is the same. Another
    // endpoint's preparation kept none, and the table replays here.
    let replay = match ctx.state.prompt_thought {
        Some(replayed) => replayed,
        None => symphony::reasoning_open(
            &ctx.components
                .parser_resolver
                .names_for(ctx.state.parser_names.as_ref(), model),
            model,
            &ctx.components.templates,
            processed_text,
        ),
    };
    match replay {
        Some(thought) => (thought.or_switch(switch), thought.closed),
        None => (switch(), false),
    }
}

/// What the table's `auto` tag derives for, when the worker's policy asks for
/// the tag ([`symphony::AutoTag`]): the request, with the tools the tag is
/// over, and whether the prompt leaves the model inside its thought, or
/// closed it itself and wrote the template's text after it.
pub(crate) struct Auto {
    pub request: AutoRequest,
    pub reasoning_open: bool,
    pub thought_closed: bool,
}

/// What [`decide`] decided for the response.
pub(crate) struct Decided {
    /// What derives the table's tag for a worker's start, where the table
    /// shapes it by the start (the `auto` tag the worker's policy chose, a
    /// JSON answer's): the attempt stamp carries it, so a retry gives the
    /// worker it selects the tag shaped for where its own engine starts the
    /// grammar (`helpers::ToolConstraintStamp`).
    pub table_tag: Option<Arc<dyn TagForStart>>,
    /// The constraint as the request carries it, before the plan builder
    /// shapes it for the worker.
    pub tool_constraints: Option<(String, String)>,
    /// The constraint is the whole turn's grammar, which holds the request's
    /// own format: the engine's request carries none beside it.
    pub whole_turn: bool,
    /// The constraint was shaped for an engine that starts the grammar at the
    /// first generated token (the thought, closed, ahead of the answer or the
    /// calls; the whole turn's grammar): the engine's request says so where
    /// its dialect has a switch for the gate behind a reasoning parser
    /// (`WorkerConstraint::first_token`).
    pub first_token: bool,
    /// Symphony's assignment, for the response spec: the table that reads the
    /// response, and the part of the rendered prompt it replays.
    pub assignment: Assignment,
}

/// What constrains the output, decided on the worker selected for the
/// request (`regular::symphony::worker_constraint`): the whole turn's grammar
/// on a worker whose engine takes one on every turn from its first token;
/// else what preparation found (`prepared`; a JSON answer's tag derived for
/// the worker's start), and on a worker that takes a grammar on every turn
/// the table's tag for the calls the request allows under `auto` (`auto`,
/// derived for the worker's start). When the worker's policy chose the
/// constraint, the state records the worker's decoding, so a retry re-selects
/// only workers that decide alike (`utils::WorkerDecoding::decides_alike`);
/// what the request asked for suits any worker. Then Symphony's assignment,
/// taken while the rendered prompt is still here
/// (`regular::symphony::assign_for`): it reads the constraint's kind alone,
/// before any shaping for the worker, and a JSON schema puts the `forced`,
/// `json` or `plain` table in the model's table's place. `declared` is the
/// tools the template sees (a chat request's effective tool list, a Messages
/// request's chat tools) and `processed_text` the rendered prompt. Either
/// refusal is the 400 the request gets.
pub(crate) fn decide(
    ctx: &mut RequestContext,
    asks: Asks<'_>,
    declared: &[Tool],
    processed_text: &str,
    prepared: PreparedConstraint,
    auto: impl FnOnce(&RequestContext) -> Auto,
) -> Result<Decided, Response> {
    let model = asks.model();
    let names = ctx
        .components
        .parser_resolver
        .names_for(ctx.state.parser_names.as_ref(), model);
    let decoding = utils::WorkerDecoding::of(ctx.state.workers.as_ref());
    let decided = symphony::worker_constraint(
        decoding,
        prepared,
        || {
            symphony::turn_constraint(
                &names,
                asks,
                declared,
                processed_text,
                decoding.vocabulary,
                &ctx.components.templates,
            )
        },
        || {
            let Auto {
                request,
                reasoning_open,
                thought_closed,
            } = auto(ctx);
            symphony::AutoTag {
                labels: names.labels.clone(),
                flags: names.flags.clone(),
                request,
                reasoning_open,
                thought_closed,
                templates: Arc::clone(&ctx.components.templates),
            }
        },
    )
    .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
    if decided.by_policy {
        ctx.state.constraint_decided_on = Some(decoding);
    }
    // The table's constraint, as Symphony's assignment reads it below: a
    // `json_schema` pair there is the call the table had no tag for, or the
    // answer it gave none, which the `forced`, `json` or `plain` table reads.
    let tabled = decided.constraint.map(|constraint| constraint.to_tuple());
    let provider = ctx
        .state
        .provider
        .unwrap_or_else(|| ProviderProfile::for_model(model));
    // The table decision preparation read for the request stands: the names
    // are the request's. Where none was kept (another endpoint's preparation;
    // a refusal the constraint carried), the assignment decides itself.
    let read = match ctx.state.reading.clone() {
        Some(read) => read,
        None => symphony::reading(&names, model, &ctx.components.templates)
            .map_err(|why| error::bad_request("invalid_parser_configuration", why))?,
    };
    let assignment = symphony::assign_for(read, asks, tabled.as_ref(), processed_text, provider);
    // A Messages answer the table gave this worker no tag for (a registry
    // table leaves the schema alone to the engine, as it does for a chat
    // `response_format`, whose request builder then carries the schema) is
    // the request's own schema, the engine's JSON-schema constraint, on the
    // pair the engine receives alone: the assignment above read the table's,
    // so the model's table still reads the response, and the tag stays on the
    // attempt stamp, from which a retry's worker derives its own
    // (`helpers::ToolConstraintStamp::capture_table`).
    let tool_constraints = tabled.or_else(|| {
        asks.untabled_answer_schema(declared)
            .map(|schema| ToolConstraint::JsonSchema(schema.to_string()).to_tuple())
    });
    Ok(Decided {
        table_tag: decided.table_tag,
        tool_constraints,
        whole_turn: decided.whole_turn,
        first_token: decided.first_token,
        assignment,
    })
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        chat::ChatCompletionRequest, common::ToolChoiceValue, messages::CreateMessageRequest,
    };
    use serde_json::json;

    use super::*;
    use crate::routers::grpc::{regular::stages::testing::components, utils::message_utils};

    /// A model the registry has a table for, named by no label or flag.
    const MODEL: &str = "Qwen/Qwen3-8B";
    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    /// A model the registry has no table for, with no flag naming one: `plain` reads it.
    const NO_TABLE_MODEL: &str = "acme/plain-7b";

    /// What [`decide`] said: the table assigned with the prompt it replays, and the constraint.
    #[derive(Debug, PartialEq)]
    struct Outcome {
        assigned: (String, String),
        constraint: Option<(String, String)>,
    }

    /// What [`decide`] says for `asks` on `ctx` under `prepared`, where the worker's policy is
    /// `requested`; no `auto` tag is derived.
    fn decided_under(
        ctx: &mut RequestContext,
        asks: Asks<'_>,
        prepared: PreparedConstraint,
    ) -> Outcome {
        let decided = decide(ctx, asks, &[], PROMPT, prepared, |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert!(!decided.whole_turn && decided.table_tag.is_none());
        Outcome {
            assigned: (
                decided.assignment.format.name().to_string(),
                decided.assignment.prompt,
            ),
            constraint: decided.tool_constraints,
        }
    }

    /// [`decided_under`] where preparation found nothing.
    fn decided(ctx: &mut RequestContext, asks: Asks<'_>) -> Outcome {
        decided_under(ctx, asks, Ok(symphony::Prepared::None))
    }

    fn a_tool(name: &str) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"$ref": "#/$defs/City"}},
                    "$defs": {"City": {"type": "string"}}
                }
            }
        }))
        .expect("a tool")
    }

    /// The schema the engine writes a forced call under where the table has no tag for it: the
    /// named function's parameters; for `required`, and for allowed tools in `required` mode, a
    /// list of at least one call naming one of the tools, their `$defs` gathered at the root; a
    /// schema under which two tools define one name differently refuses the request here. A
    /// choice that forces no call, and a named function the tools do not declare, give none.
    #[test]
    fn the_forced_calls_schema_is_the_functions_parameters_or_the_required_list() {
        let tools = [a_tool("lookup"), a_tool("book")];
        let named: ToolChoice =
            serde_json::from_value(json!({"type": "function", "function": {"name": "book"}}))
                .expect("a named choice");
        assert_eq!(
            forced_call_schema(&tools, Some(&named)),
            Ok(Some(tools[1].function.parameters_schema().clone()))
        );
        let required = ToolChoice::Value(ToolChoiceValue::Required);
        let list = forced_call_schema(&tools, Some(&required))
            .expect("one definition of City")
            .expect("a list");
        assert_eq!(list["type"], "array");
        assert_eq!(list["minItems"], 1);
        assert_eq!(list["$defs"]["City"], json!({"type": "string"}));
        let names: Vec<&str> = list["items"]["anyOf"]
            .as_array()
            .expect("the calls")
            .iter()
            .map(|call| call["properties"]["name"]["enum"][0].as_str().unwrap())
            .collect();
        assert_eq!(names, ["lookup", "book"]);
        let allowed: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "book"}}]
            }
        }))
        .expect("allowed tools");
        assert!(matches!(
            forced_call_schema(&tools[1..], Some(&allowed)),
            Ok(Some(_))
        ));

        let mut other = a_tool("other");
        other.function.parameters = Some(json!({
            "type": "object", "$defs": {"City": {"type": "integer"}}
        }));
        assert!(
            forced_call_schema(&[tools[0].clone(), other], Some(&required))
                .expect_err("two definitions of City")
                .contains("City")
        );

        let auto = ToolChoice::Value(ToolChoiceValue::Auto);
        assert_eq!(forced_call_schema(&tools, Some(&auto)), Ok(None));
        assert_eq!(forced_call_schema(&tools, None), Ok(None));
        let stranger: ToolChoice =
            serde_json::from_value(json!({"type": "function", "function": {"name": "web_search"}}))
                .expect("a named choice");
        assert_eq!(forced_call_schema(&tools, Some(&stranger)), Ok(None));
        assert_eq!(forced_call_schema(&[], Some(&required)), Ok(None));
    }

    /// Under the JSON schema the gateway built for a call the table had no tag for, the
    /// assignment reads with the table the request names, not the model's: `forced` for the
    /// function the choice names, which reads its arguments alone, `json` for a required call,
    /// which reads the list; the constraint reaches the engine as the schema. Where the table
    /// gave the call its tag, the model's own table reads.
    #[test]
    fn a_json_schema_constraint_assigns_the_forced_or_json_table() {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let under_schema = || {
            Ok(symphony::Prepared::from(Some(ToolConstraint::JsonSchema(
                schema.to_string(),
            ))))
        };
        let tools =
            json!([{"type": "function", "function": {"name": "book", "parameters": schema}}]);
        let naming: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "book"}}
            }))
            .expect("chat request"),
        );
        let mut ctx =
            RequestContext::for_chat(Arc::clone(&naming), None, MODEL.into(), components());
        let outcome = decided_under(&mut ctx, Asks::Chat(&naming), under_schema());
        assert_eq!(outcome.assigned.0, "forced");
        assert_eq!(
            outcome.constraint,
            Some(("json_schema".to_string(), schema.to_string()))
        );

        let requiring: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "tools": tools, "tool_choice": "required"
            }))
            .expect("chat request"),
        );
        let mut ctx =
            RequestContext::for_chat(Arc::clone(&requiring), None, MODEL.into(), components());
        let outcome = decided_under(&mut ctx, Asks::Chat(&requiring), under_schema());
        assert_eq!(outcome.assigned.0, "json");

        let tagged = Ok(symphony::Prepared::from(Some(
            ToolConstraint::StructuralTag("{}".to_string()),
        )));
        let mut ctx =
            RequestContext::for_chat(Arc::clone(&requiring), None, MODEL.into(), components());
        let outcome = decided_under(&mut ctx, Asks::Chat(&requiring), tagged);
        assert_eq!(outcome.assigned.0, "qwen3");
    }

    /// A Messages request's answer schema reaches the engine where the table gives the worker no
    /// tag for it: a registry table leaves the schema alone to the engine, as it does for a chat
    /// `response_format` (whose request builder carries the schema), so the decision hands the
    /// engine the request's own JSON-schema constraint, and the table's tag stays for the stamp;
    /// Symphony's assignment read the table's constraint, not the schema, so the model's table
    /// still reads the response.
    #[test]
    fn a_messages_answer_schema_the_table_leaves_alone_is_the_engines_constraint() {
        let schema = json!({
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]
        });
        let request: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }))
            .expect("a Messages request"),
        );
        let mut ctx =
            RequestContext::for_messages(Arc::clone(&request), None, MODEL.into(), components());
        let asks = Asks::Messages {
            request: &request,
            continues: false,
        };
        let names = ctx.components.parser_resolver.names_for(None, MODEL);
        let read = symphony::reading(&names, MODEL, &ctx.components.templates)
            .expect("a table reads the model");
        let prepared = symphony::constraint_for(
            &read,
            asks,
            &[],
            true,
            false,
            symphony::grammar_start_of(&[]),
            symphony::vocabulary_of(&[]),
        );
        assert!(
            matches!(prepared, Ok(symphony::Prepared::Answer(_))),
            "{prepared:?}"
        );
        let decided = decide(&mut ctx, asks, &[], PROMPT, prepared, |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert_eq!(
            decided.tool_constraints,
            Some(("json_schema".to_string(), schema.to_string()))
        );
        assert!(decided.table_tag.is_some() && !decided.whole_turn);
        assert_eq!(
            decided.assignment.format.name(),
            "qwen3",
            "the model's table reads the response"
        );
    }

    /// The decision follows preparation on whether the choice forces a call, over different tool
    /// lists: preparation reads the tools the template renders, narrowed by `tool_choice`
    /// (`message_utils::template_tools`), the decision the request's declared chat tools. A choice
    /// naming a tool of another kind (a server tool, which the custom tools never hold) narrows
    /// preparation's list to nothing and names no declared chat tool at the decision, so both
    /// leave the answer formatted and the engine gets the schema.
    #[test]
    fn a_choice_naming_a_tool_the_model_never_sees_leaves_the_answer_formatted_at_the_decision() {
        let schema = json!({
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]
        });
        let request: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "lookup", "input_schema": {"type": "object", "properties": {}}},
                    {"type": "web_search_20250305", "name": "web_search"}
                ],
                "tool_choice": {"type": "tool", "name": "web_search"},
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }))
            .expect("a Messages request"),
        );
        let declared = request.chat_tools();
        let narrowed = message_utils::template_tools(&declared, request.tool_choice.as_ref());
        assert_eq!(
            (declared.len(), narrowed.len()),
            (1, 0),
            "the custom tool alone is declared, and the choice narrows it away"
        );
        let chat_choice = request.tool_choice.as_ref().map(ToolChoice::from);
        assert_eq!(
            forced_call_schema(&narrowed, chat_choice.as_ref()),
            Ok(None),
            "no call to force over the narrowed list"
        );
        let mut ctx =
            RequestContext::for_messages(Arc::clone(&request), None, MODEL.into(), components());
        let asks = Asks::Messages {
            request: &request,
            continues: false,
        };
        let names = ctx.components.parser_resolver.names_for(None, MODEL);
        let read = symphony::reading(&names, MODEL, &ctx.components.templates)
            .expect("a table reads the model");
        let prepared = symphony::constraint_for(
            &read,
            asks,
            &narrowed,
            true,
            false,
            symphony::grammar_start_of(&[]),
            symphony::vocabulary_of(&[]),
        );
        assert!(
            matches!(prepared, Ok(symphony::Prepared::Answer(_))),
            "{prepared:?}"
        );
        let decided = decide(&mut ctx, asks, &declared, PROMPT, prepared, |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert_eq!(
            decided.tool_constraints,
            Some(("json_schema".to_string(), schema.to_string())),
            "the decision names no forced call over the declared tools either"
        );
        assert_eq!(decided.assignment.format.name(), "qwen3");
    }

    /// Request building's reasoning decision reads the replay preparation kept where it replayed
    /// the prompt (the same prompt, replayed once per request), and replays the prompt itself
    /// where none was kept. A model no registry table reads replays through `plain`, which
    /// leaves the thought to the request's switch, kept or fresh alike.
    #[test]
    fn the_reasoning_decision_reads_the_replay_preparation_kept() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        // Nothing kept: the decision replays the prompt, which leaves the thought open.
        assert_eq!(
            reasoning_decision(&ctx, MODEL, PROMPT, |_| false),
            (true, false)
        );
        // Preparation's replay stands in place of a second one: its answer, not the prompt's.
        ctx.state.prompt_thought = Some(symphony::reasoning_open(
            &ctx.components.parser_resolver.names(MODEL),
            MODEL,
            &ctx.components.templates,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ));
        assert_eq!(
            reasoning_decision(&ctx, MODEL, PROMPT, |_| false),
            (false, true)
        );
        // A model no registry table reads: `plain` replays nothing of the prompt, so the switch
        // decides, whether the replay is fresh or kept (the switch reads the request's tokenizer,
        // which preparation leaves on the context).
        ctx.state.prompt_thought = None;
        ctx.state.tokenizer = Some(Arc::new(llm_tokenizer::MockTokenizer::new()));
        assert_eq!(
            reasoning_decision(&ctx, NO_TABLE_MODEL, PROMPT, |_| false),
            (false, false)
        );
        assert_eq!(
            reasoning_decision(&ctx, NO_TABLE_MODEL, PROMPT, |_| true),
            (true, false)
        );
        ctx.state.prompt_thought = Some(symphony::reasoning_open(
            &ctx.components.parser_resolver.names(NO_TABLE_MODEL),
            NO_TABLE_MODEL,
            &ctx.components.templates,
            PROMPT,
        ));
        assert!(
            ctx.state.prompt_thought.flatten().is_some(),
            "plain replays"
        );
        assert_eq!(
            reasoning_decision(&ctx, NO_TABLE_MODEL, PROMPT, |_| true),
            (true, false)
        );
    }

    /// A chat request and a Messages request that ask the same of the table decide the same,
    /// through the one function: the registry's table reads both, replaying the same part of the
    /// prompt, and nothing constrains either.
    #[test]
    fn a_chat_and_a_messages_request_that_ask_the_same_decide_alike() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        let for_chat = decided(&mut ctx, Asks::Chat(&chat));

        let messages: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .expect("messages request"),
        );
        let mut ctx =
            RequestContext::for_messages(Arc::clone(&messages), None, MODEL.into(), components());
        let for_messages = decided(
            &mut ctx,
            Asks::Messages {
                request: &messages,
                continues: false,
            },
        );

        assert_eq!(for_chat, for_messages);
        assert_eq!(for_chat.assigned.0, "qwen3");
        assert!(for_chat.constraint.is_none());
    }

    /// Request building reads the parser names preparation resolved for the request
    /// (`ProcessingState::parser_names`), not the registry again: the names stored name the
    /// Kimi K2 table, so that table reads the response, though no label of the model's workers
    /// and no flag names it (the registry's own table for the model is Qwen3's).
    #[test]
    fn the_decision_reads_the_names_preparation_resolved() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        ctx.state.parser_names = Some(Names {
            tool: Some("kimi_k2".to_string()),
            reasoning: None,
        });
        let outcome = decided(&mut ctx, Asks::Chat(&chat));
        assert_eq!(outcome.assigned.0, "kimi_k2");
    }

    /// Request building's assignment reads the table decision preparation kept for the request
    /// (`ProcessingState::reading`), not the names again: the decision kept names the Kimi K2
    /// table, so that table reads the response, though the names and the registry say Qwen3's; a
    /// kept `plain` decision reads the response as content; with nothing kept the assignment
    /// decides itself.
    #[test]
    fn the_assignment_reads_the_decision_preparation_kept() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        let kept = |ctx: &RequestContext, tool: &str| {
            let names = Names {
                tool: Some(tool.to_string()),
                reasoning: None,
            };
            symphony::reading(
                &ctx.components
                    .parser_resolver
                    .names_for(Some(&names), MODEL),
                MODEL,
                &ctx.components.templates,
            )
            .expect("a table the names select")
        };
        // Nothing kept: the assignment decides itself, the registry's table for the model.
        assert_eq!(decided(&mut ctx, Asks::Chat(&chat)).assigned.0, "qwen3");
        // The kept decision stands in place of a fourth one over the same names.
        ctx.state.reading = Some(kept(&ctx, "kimi_k2"));
        assert_eq!(decided(&mut ctx, Asks::Chat(&chat)).assigned.0, "kimi_k2");
        // A kept `plain` decision: the response is content.
        ctx.state.reading = Some(kept(&ctx, "plain"));
        assert_eq!(decided(&mut ctx, Asks::Chat(&chat)).assigned.0, "plain");
    }
}
