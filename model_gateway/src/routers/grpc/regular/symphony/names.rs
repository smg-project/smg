//! Name resolution: which table reads a model's responses, from the parser names set for it and
//! the registry, and the checks on those names at startup and at a worker's registration.

use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use symphony::{
    formats::{for_model_shared, table_for, Table},
    Format,
};

use super::{telemetry, template::Templates, SymphonyError};
use crate::{
    config::{ParserChoice, TEMPLATE_PARSER},
    routers::grpc::utils::{answering_to, gives_way, ModelNames, Names},
};

/// The legacy parser that leaves its channel as written: an operator's choice no table reads
/// for, so it keeps the legacy parsers, alone or beside another name.
const PASSTHROUGH: &str = "passthrough";

fn is_named(name: Option<&str>, what: &str) -> bool {
    name.is_some_and(|name| name.trim().eq_ignore_ascii_case(what))
}

fn is_passthrough(name: Option<&str>) -> bool {
    is_named(name, PASSTHROUGH)
}

/// Whether the parser names select the response template's table: either name is
/// [`TEMPLATE_PARSER`]; [`check_names`] binds the other to it or to nothing.
pub(crate) fn names_template(names: &Names) -> bool {
    is_named(names.tool.as_deref(), TEMPLATE_PARSER)
        || is_named(names.reasoning.as_deref(), TEMPLATE_PARSER)
}

/// The table the parser names set for a model select, when they do: the tool-call name's, else
/// the reasoning name's when no tool-call name is set. A name is one a table answers to
/// ([`answering_to`]): its own, a legacy parser's that reads its format, or a launcher's
/// for a family whose table has no older parser; any other name selects none, `passthrough` and
/// [`TEMPLATE_PARSER`] among them, and so do no names at all.
pub(crate) fn named_table(names: &Names) -> Option<Table> {
    names
        .tool
        .as_deref()
        .or(names.reasoning.as_deref())
        .and_then(answering_to)
}

/// What one source's parser names select: the response template's table ([`TEMPLATE_PARSER`] in
/// either name); a table ([`named_table`]); the legacy parsers, by a tool-call name no table
/// answers to, which is the way back for one model when its table misbehaves, or by `passthrough`
/// alone, the choice to leave the thought as written, which no table does; or nothing, by no name
/// at all, or a reasoning name alone that no table answers to, since every table reads the thought
/// channel, or one that gives way ([`gives_way`]).
enum Selection {
    Template,
    Table(Table),
    Legacy,
    Nothing,
}

fn selection(names: &Names) -> Selection {
    if names_template(names) {
        return Selection::Template;
    }
    match named_table(names) {
        // A reasoning name alone that gives way: [`decision`] reads with its table where
        // neither source selects anything.
        Some(_) if names.tool.is_none() && names.reasoning.as_deref().is_some_and(gives_way) => {
            Selection::Nothing
        }
        Some(table) => Selection::Table(table),
        None if names.tool.is_none() && !is_passthrough(names.reasoning.as_deref()) => {
            Selection::Nothing
        }
        None => Selection::Legacy,
    }
}

/// Whether `names`, one source's, select nothing (no name at all, or a reasoning name alone that no
/// table answers to, `passthrough` excepted, or that gives way), so that [`decision`] passes them
/// over for the next source: a worker labeled so is read by the registry's table, as an unlabeled
/// one is.
pub(crate) fn labels_select_nothing(names: &Names) -> bool {
    matches!(selection(names), Selection::Nothing)
}

/// Why the legacy parsers read a request Symphony was configured for: the precedence among the
/// names and the registry decides it ([`decision`]), and nothing else does. A request so read is
/// counted once under its reason ([`telemetry::record_fallback`]); its response's header says
/// `legacy`, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fallback {
    /// No table reads the model and no name is set for it: the registry does not know the
    /// model's served name, no `--model-alias` it is served under reads it ([`read_by_alias`];
    /// an alias declared on a worker's own card is not consulted), and neither its labels nor
    /// the flags name anything.
    NoTable,
    /// A reasoning name alone that no table answers to (`deepseek_r1`, `qwen3_thinking`), in the
    /// labels or the flags, for a model the registry does not know: every table reads the thought
    /// channel, so the name selects nothing, and nothing else selects a table.
    ReasoningNameOnly,
    /// A legacy tool-call parser's name no table answers to (`pythonic`): the way back
    /// to the legacy parsers for one model (the labels) or for every model (the flags), the
    /// registry's included.
    LegacyToolName,
    /// `passthrough`, as the tool-call name or alone as the reasoning name: the choice to leave a
    /// channel as written, which no table does.
    Passthrough,
}

/// What reads a model's responses under Symphony ([`decision`]).
pub(super) enum Decision {
    /// The response template's table, compiled when the request comes ([`Templates`]).
    Template,
    /// A table, shared with every request and engine that reads with it: the registry's, or
    /// the one the names select.
    Table(Arc<Format>),
    /// The legacy parsers, for the reason the precedence names.
    Legacy(Fallback),
}

/// Why one source's names select the legacy parsers ([`Selection::Legacy`]): `passthrough` as
/// the tool-call name, or alone as the reasoning name; else a tool-call name no table answers to.
fn legacy_reason(names: &Names) -> Fallback {
    if is_passthrough(names.tool.as_deref())
        || (names.tool.is_none() && is_passthrough(names.reasoning.as_deref()))
    {
        Fallback::Passthrough
    } else {
        Fallback::LegacyToolName
    }
}

/// Why nothing reads a model the registry does not know when both sources select nothing
/// ([`Selection::Nothing`]): a reasoning name alone that no table answers to, in either, or no
/// name at all.
fn nothing_reason(names: &ModelNames<'_>) -> Fallback {
    if names.labels.reasoning.is_some() || names.flags.reasoning.is_some() {
        Fallback::ReasoningNameOnly
    } else {
        Fallback::NoTable
    }
}

/// The reader of `model`'s responses under Symphony, by the one precedence among the parser names
/// set for the model ([`ModelNames`]) and the registry: the names of the model's card (its
/// workers' labels), when they select anything; else the gateway's `--tool-call-parser` and
/// `--reasoning-parser`, when they select anything; else the registry's table, for a model it
/// knows (`symphony::formats::for_model`); else, where no name is set at all, the registry's table
/// for a configured alias the model is served under (`--model-alias`), the first the registry
/// knows in the aliases' sorted order, for a served name the legacy parsers' own detection reads
/// nothing by ([`read_by_alias`]); else the legacy parsers, with their own detection, the
/// reason named ([`Fallback`]). The flags' legacy tool-call name or `passthrough` keeps the legacy
/// parsers for a model the registry knows too. A reasoning name alone in the flags does not come
/// before the registry ([`flags_read_first`]): a table that answers to it reads only the models
/// the registry does not know, and one no table answers to selects nothing. A table the labels or
/// the flags select reads a model the registry gives the same table with the model's own turn
/// opener; another model, or one the registry does not know, replays from the table's own. A
/// reasoning name alone that gives way ([`gives_way`]: `kimi`) is passed over in either source,
/// as before its table came, and its table reads the model where the legacy parsers would read
/// with the name: neither source selects anything, and it is the reasoning name in force, the
/// labels' before the flags'. The registry is asked once per path, as before the flags came
/// apart: the path that reaches the aliases reads what the start recorded of them.
pub(super) fn decision(names: &ModelNames<'_>, model: &str) -> Decision {
    if let Some(decided) = decided_by(&names.labels, model) {
        return decided;
    }
    if flags_read_first(names.flags) {
        if let Some(decided) = decided_by(names.flags, model) {
            return decided;
        }
    }
    match for_model_shared(model) {
        Some(format) => Decision::Table(format),
        // The registry does not know the model: a table a reasoning name alone in the flags
        // selects reads it, from the table's own opener.
        None => match selection(names.flags) {
            Selection::Template => Decision::Template,
            Selection::Table(table) => Decision::Table(table.shared()),
            Selection::Legacy => Decision::Legacy(legacy_reason(names.flags)),
            // The reasoning name in force is the labels' before the flags', as the legacy
            // parsers take it: its table reads when it is one that gives way.
            Selection::Nothing => match names
                .labels
                .reasoning
                .as_deref()
                .or(names.flags.reasoning.as_deref())
                .filter(|name| gives_way(name))
                .and_then(answering_to)
            {
                Some(table) => Decision::Table(table.shared()),
                None => match nothing_reason(names) {
                    // No name is set, the registry does not know the served name and the
                    // legacy parsers detect nothing by it: the registry's table for an
                    // alias the model is served under reads the model.
                    Fallback::NoTable => by_alias(names, model)
                        .map_or(Decision::Legacy(Fallback::NoTable), |(_, format, _)| {
                            Decision::Table(Arc::clone(format))
                        }),
                    why => Decision::Legacy(why),
                },
            },
        },
    }
}

/// What one source's names decide for `model` ahead of the registry ([`decision`]): the response
/// template, the legacy parsers with the reason, or a table, the registry's own format of it where
/// the registry gives `model` the same table, so the model keeps its own turn opener; `None` when
/// they select nothing, which leaves the decision to the next source.
fn decided_by(names: &Names, model: &str) -> Option<Decision> {
    Some(match selection(names) {
        Selection::Nothing => return None,
        Selection::Template => Decision::Template,
        Selection::Legacy => Decision::Legacy(legacy_reason(names)),
        Selection::Table(table) => Decision::Table(match for_model_shared(model) {
            Some(format) if table_for(model) == Some(table) => format,
            _ => table.shared(),
        }),
    })
}

/// Whether the gateway's flags decide ahead of the registry ([`decision`]): they do unless they
/// select a table through a reasoning name alone. That name is the usual engine flag of a whole
/// family (`--reasoning-parser qwen3` for every Qwen 3 model), and its table need not read the call
/// syntax the registry reads the model with (the JSON calls of `qwen3` for a model the registry
/// reads with the tagged calls of `qwen3_tagged`), so the registry reads a model it knows, as it
/// does where a reasoning name alone that no table answers to selects nothing. A worker's labels
/// are not asked this: a reasoning name alone in them names the table for their one model.
pub(crate) fn flags_read_first(flags: &Names) -> bool {
    flags.tool.is_some() || !matches!(selection(flags), Selection::Table(_))
}

/// What reads a model's responses under a parser ([`reading`]). Preparation reads it once per
/// request and keeps it (`ProcessingState::reading`) for the readers after it: the replay, the
/// constraint, the decoder's special tokens, the assignment.
#[derive(Clone)]
pub(crate) enum Reading {
    /// A table, and whether it is the response template's, compiled for the model: a constraint
    /// is compiled against it, and a worker may take its whole turn's grammar.
    Table(Arc<Format>, bool),
    /// The legacy parsers, by the gateway's choice: `--parser legacy`, or the request's header.
    Legacy,
    /// The legacy parsers under Symphony: no table reads the request, for the reason the
    /// precedence names.
    Fallback(Fallback),
}

/// What reads `model`'s responses under `parser`: the legacy parsers under `legacy`; else what
/// [`decision`] gives, the response template's table compiled for the model (an error when the
/// model has none that compiles, which refuses the request, since the names ask for what the
/// model cannot give), another table, or the legacy parsers with the reason.
pub(crate) fn reading(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
) -> Result<Reading, SymphonyError> {
    if parser != ParserChoice::Symphony {
        return Ok(Reading::Legacy);
    }
    Ok(match decision(names, model) {
        Decision::Template => Reading::Table(
            templates.format(model).map_err(SymphonyError::Template)?,
            true,
        ),
        Decision::Table(format) => Reading::Table(format, false),
        Decision::Legacy(why) => Reading::Fallback(why),
    })
}

/// The table that reads `model`'s responses under `parser`, if one does ([`reading`]), and whether
/// it is the response template's. `None` under `legacy`, and when the legacy parsers read the
/// response.
pub(super) fn table_reading(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
) -> Result<Option<(Arc<Format>, bool)>, SymphonyError> {
    Ok(match reading(parser, names, model, templates)? {
        Reading::Table(format, compiled) => Some((format, compiled)),
        Reading::Legacy | Reading::Fallback(_) => None,
    })
}

/// The table that reads `model`'s responses under `parser`, if one does: the response template's,
/// the registry's, or the one the parser names select, by the precedence [`decision`] states; `None`
/// under `legacy`, and when the legacy parsers read the response. What a request asks of the
/// channels takes a variant of the table ([`assign_for`](super::reader::assign_for)). Preparation
/// decides through [`reading`] once and carries the decision; the `auto` tag decides with this
/// predicate for the selected worker. The table comes shared, built once for every request and
/// engine that reads with it; a variant is a copy.
pub(crate) fn reads(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
) -> Result<Option<Arc<Format>>, SymphonyError> {
    table_reading(parser, names, model, templates).map(|read| read.map(|(format, _)| format))
}

/// The `table` label under which Symphony counts `model`'s outputs ([`telemetry`]), when a table
/// reads them: the table [`decision`] gives for the names set for the model, or the response
/// template's name for a model read through its template. `None` when the legacy parsers read the
/// model: its requests count under `none`, whose series exist from start-up. Asked at the model's
/// first registration, so the table's series exist at zero before its first output
/// (`workflow::steps::local::check_local_worker`).
pub(crate) fn table_label_read(names: &ModelNames<'_>, model: &str) -> Option<&'static str> {
    match decision(names, model) {
        Decision::Template => Some(telemetry::TEMPLATE),
        Decision::Table(format) => Some(telemetry::table_label(format.name())),
        Decision::Legacy(_) => None,
    }
}

/// What the gateway's flags select for `model`, a model the registry knows, that the precedence of
/// [`decision`] sets aside for a worker labeling no parser: the registry's table, which reads the
/// model, and the table a reasoning name alone in the flags selects, spelled for a log line (`the
/// table 'qwen3'`). Only such a name is set aside ([`flags_read_first`]); `None` for flags that
/// read first, flags that select nothing or the registry's own table, and a model the registry
/// does not know. Registration warns with it once per model
/// (`workflow::steps::local::check_local_worker`).
pub(crate) fn set_aside_by_the_registry(
    flags: &Names,
    model: &str,
) -> Option<(Table, Cow<'static, str>)> {
    if flags_read_first(flags) {
        return None;
    }
    let table = table_for(model)?;
    match selection(flags) {
        Selection::Table(named) if named != table => {
            Some((table, Cow::Owned(format!("the table '{}'", named.name()))))
        }
        _ => None,
    }
}

/// A served model an alias reads ([`read_by_alias`]): its served name, the table that reads it,
/// and each `--model-alias` of it the registry knows with its table, in the aliases' sorted order.
pub(crate) type ByAlias = (String, Arc<Format>, Vec<(String, Table)>);

/// The served models an alias reads, sorted by served name, read once at the start: for each
/// served name among `aliases` (`--model-alias`, the alias and the name it is served under) that
/// the registry does not know and neither legacy registry detects a parser for, the registry's
/// table for the first of its aliases the registry knows, in their sorted order. A served name a
/// legacy registry detects a parser for keeps the legacy parsers, as before the aliases were
/// read, whatever table an alias of it names; an absent registry detects nothing. The context
/// that asked keeps the record and hands it on with the names of a model ([`ModelNames`]), and
/// [`decision`] reads it where no name is set for the model: a request asks neither the aliases
/// nor the legacy registries, and one context reads nothing another recorded.
pub(crate) fn read_by_alias<'a>(
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
    tools: Option<&tool_parser::ParserFactory>,
    reasoning: Option<&reasoning_parser::ParserFactory>,
) -> Vec<ByAlias> {
    let mut of_served: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (alias, served) in aliases {
        of_served.entry(served).or_default().push(alias);
    }
    // The map hands out the served names in their order, so the record is sorted by them.
    let mut by_alias = Vec::new();
    for (served, mut aliases) in of_served {
        let read_by_its_name = table_for(served).is_some()
            || tools.is_some_and(|tools| tools.registry().has_parser_for_model(served))
            || reasoning.is_some_and(|reasoning| reasoning.registry().has_parser_for_model(served));
        if read_by_its_name {
            continue;
        }
        aliases.sort_unstable();
        if let Some(format) = aliases.iter().find_map(|alias| for_model_shared(alias)) {
            let tables = aliases
                .iter()
                .filter_map(|alias| Some(((*alias).to_string(), table_for(alias)?)))
                .collect();
            by_alias.push((served.to_string(), format, tables));
        }
    }
    by_alias
}

/// The record of the served model `model` among those the context of `names` reads by an alias
/// ([`read_by_alias`]), where one reads it. No lock is taken and no shared count written; where
/// no alias reads any served model the search ends on the empty record.
fn by_alias<'a>(names: &ModelNames<'a>, model: &str) -> Option<&'a ByAlias> {
    let at = names
        .by_alias
        .binary_search_by(|(served, ..)| served.as_str().cmp(model))
        .ok()?;
    names.by_alias.get(at)
}

/// The `--model-alias` names of `model` the registry knows, each with its table, in the aliases'
/// sorted order, where [`decision`] reads `model` by the first of them: no name is set for the
/// model and an alias reads it ([`read_by_alias`]). Empty where the decision does not reach the
/// aliases or none reads the model. Registration says the first once per model, and warns where
/// a later one names another table (`workflow::steps::local::check_local_worker`).
pub(crate) fn tables_by_alias(names: &ModelNames<'_>, model: &str) -> Vec<(String, Table)> {
    if names.labels != Names::default() || *names.flags != Names::default() {
        return Vec::new();
    }
    by_alias(names, model).map_or_else(Vec::new, |(_, _, tables)| tables.clone())
}

/// Whether the parser names set for a model (the flags, or one worker's labels) are names the
/// gateway reads with under `parser`: a legacy parser's, or, with Symphony on, a table's or
/// [`TEMPLATE_PARSER`]. `tools` and `reasoning` are the legacy registries; a name whose registry
/// is absent is not checked (its parsers are unused in that configuration). A name no legacy
/// parser of its own kind has that a table answers to (the table's own name, a launcher's name
/// for a family whose table has no older parser, or a legacy parser's name of the other kind),
/// and [`TEMPLATE_PARSER`], bind the other name to the same table or to nothing: the two names
/// are one selection, and where the other name selects the legacy parsers ([`decision`]) those
/// read by the names they are given and have no parser of that name. A legacy parser's name a
/// table answers to is the legacy name it is in its own kind, and selects its table in the
/// other. The error says what is wrong with which name, for the startup or registration
/// failure.
pub(crate) fn check_names(
    parser: ParserChoice,
    names: &Names,
    tools: Option<&tool_parser::ParserFactory>,
    reasoning: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), SymphonyError> {
    checked_names(parser, names, tools, reasoning).map_err(SymphonyError::Name)
}

/// [`check_names`], with what is wrong with which name as the text.
fn checked_names(
    parser: ParserChoice,
    names: &Names,
    tools: Option<&tool_parser::ParserFactory>,
    reasoning: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), String> {
    let symphony = parser == ParserChoice::Symphony;
    // Whether a kind's legacy registry has the name; an absent registry checks nothing, so
    // every name is its kind's there.
    let legacy_tool = |name: &str| tools.is_none_or(|factory| factory.has_parser(name));
    let legacy_reasoning =
        |name: &str| reasoning.is_none_or(|factory| factory.registry().has_parser(name));
    let kinds = [
        (
            "tool-call",
            names.tool.as_deref(),
            &legacy_tool as &dyn Fn(&str) -> bool,
            names.reasoning.as_deref(),
        ),
        (
            "reasoning",
            names.reasoning.as_deref(),
            &legacy_reasoning as &dyn Fn(&str) -> bool,
            names.tool.as_deref(),
        ),
    ];
    for (kind, name, legacy, other) in kinds {
        let Some(name) = name else {
            continue;
        };
        if is_named(Some(name), TEMPLATE_PARSER) {
            if !symphony {
                return Err(format!(
                    "unknown {kind} parser '{name}': the model's response template, which \
                     --parser symphony reads with"
                ));
            }
            if let Some(other) = other.filter(|other| !is_named(Some(other), TEMPLATE_PARSER)) {
                return Err(format!(
                    "{kind} parser '{name}' reads with the model's response template, so the \
                     other parser must name '{TEMPLATE_PARSER}' or nothing, not '{other}'"
                ));
            }
            continue;
        }
        if legacy(name) {
            continue;
        }
        // The table a name selects when no legacy parser of its kind has it: the table named
        // so, else the one that answers to it, a legacy parser's name of the other kind among
        // them, unless its own kind's registry has the name spelled its way (a table answers
        // in any case, the registries in theirs).
        let table = Table::named(name).or_else(|| {
            let spelled = name.trim().to_ascii_lowercase();
            (!legacy(&spelled)).then(|| answering_to(name)).flatten()
        });
        match (symphony, table) {
            (true, Some(selected)) => {
                if let Some(other) = other {
                    if answering_to(other) != Some(selected) {
                        return Err(format!(
                            "{kind} parser '{name}' selects the Symphony table '{}' and is no \
                             legacy {kind} parser's name, so the other parser must name the \
                             same table or nothing, not '{other}'",
                            selected.name()
                        ));
                    }
                }
            }
            (true, None) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': neither a legacy {kind} parser nor a \
                     Symphony table"
                ));
            }
            (false, Some(selected)) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': the Symphony table '{}', which --parser \
                     symphony reads with",
                    selected.name()
                ));
            }
            (false, None) => return Err(format!("unknown {kind} parser '{name}'")),
        }
    }
    Ok(())
}

/// Whether the rendered `prompt` leaves the model inside its thought as the table reads it: the
/// table's replay of the prompt ([`Format::reasoning_open`]), which starts a turn the prompt leaves
/// unopened inside the thought where the template writes no opener (DeepSeek's after a system
/// message a latest reminder follows, or the checkpoint's own encoder file's after any system
/// message). The grammar built for the request takes it OR'd with the request's
/// thinking switch, so the tag and the engine read the prompt the same way: a JSON answer the
/// switch would place at the first token follows the thought's close instead of streaming as
/// reasoning. A prompt that closed the thought itself overrules the switch
/// ([`PromptThought::or_switch`]). `None` when no table reads the response, leaving the switch's
/// answer as it was; where a table reads, no legacy parser is built to be asked about the prompt:
/// the table's replay answers, alone for the family whose legacy rule set the switch aside
/// ([`PromptThought::alone`]).
pub(crate) fn reasoning_open(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    prompt: &str,
) -> Option<PromptThought> {
    match reading(parser, names, model, templates) {
        Ok(read) => reasoning_open_for(&read, prompt),
        Err(_) => None,
    }
}

/// [`reasoning_open`] over the table decision already read for the request (`read`): the
/// thought's state in `prompt` where a table reads the response, `None` where none does.
pub(crate) fn reasoning_open_for(read: &Reading, prompt: &str) -> Option<PromptThought> {
    match read {
        // The registry's Gemma 4 table carries the rule; a Gemma 4 model read through its own
        // response template (a table named for the template) is left to the switch beside the
        // replay, as every other table is: telling that model apart would take the legacy
        // registry's pattern match, which it offers only by building the parser, and no Gemma 4
        // tokenizer ships a response template today.
        Reading::Table(format, _) => Some(PromptThought {
            open: format.reasoning_open(prompt),
            closed: format.thought_closed(prompt),
            alone: format.name() == Table::Gemma4.name(),
        }),
        Reading::Legacy | Reading::Fallback(_) => None,
    }
}

/// The thought's state in the rendered prompt as the table that reads the response sees it
/// ([`reasoning_open`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptThought {
    /// The prompt leaves the model inside its thought.
    pub open: bool,
    /// The prompt closed the thought itself and wrote what the template writes after the close
    /// (thinking switched off), so the tag owes none of it ([`Format::thought_closed`]) and the
    /// model starts past the thought, whatever the request's thinking switch says.
    pub closed: bool,
    /// The replay decides alone, without the request's thinking switch: Gemma 4 prefills an open
    /// thought channel only after a tool response, so thinking enabled on a new turn says nothing
    /// about the prompt. The rule its legacy reasoning parser applied in place of the switch,
    /// kept here without building one.
    pub alone: bool,
}

/// A table's replay of a rendered prompt ([`reasoning_open`]'s answer): the thought's state where
/// a table reads the response, `None` where none does. Preparation keeps its replay for request
/// building (`ProcessingState::prompt_thought`), which decides over the same prompt.
pub(crate) type PromptReplay = Option<PromptThought>;

impl PromptThought {
    /// Whether the prompt leaves the model inside its thought, as request building decides it:
    /// the replay's answer, or the request's thinking `switch` where the replay does not decide
    /// alone. A prompt that closed the thought is past it whatever the switch says: a template
    /// may close the thought for a request that names no switch while the switch's default says
    /// thinking.
    pub fn or_switch(self, switch: impl FnOnce() -> bool) -> bool {
        !self.closed && (self.open || (!self.alone && switch()))
    }
}

/// [`keeps_special_tokens_for`] over the table decision read here ([`reading`]; `None` when it
/// errs), as the tests ask it.
#[cfg(test)]
pub(crate) fn keeps_special_tokens(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Option<bool> {
    match reading(parser, names, model, templates) {
        Ok(read) => keeps_special_tokens_for(&read, model, templates, tokenizer),
        Err(_) => None,
    }
}

/// Whether the table that reads `model`'s responses (`read`, the table decision, which preparation
/// reads once for the request, `regular::stages::reading::prepare`) needs the vocabulary's special
/// tokens kept in the decoded output: one of its markers is, or holds, a token the stop decoder
/// would drop with `skip_special_tokens` (Muse Spark's `<|message|>` and `<|eom|>`, Kimi K3's
/// `<|open|>` and `<|sep|>`). A legacy reasoning parser says so of itself
/// (`requires_special_tokens`); a table says nothing, so the gateway asks the tokenizer: a marker
/// whose tokens decode to its text with the special tokens kept and to something else with them
/// dropped is such a marker. `false` when every marker survives the decoder either way (Qwen3's
/// `<think>` and `<tool_call>` are added tokens the decoder keeps), so a model the older rule
/// already serves is read as before. Asking the tokenizer encodes and decodes every marker, and the
/// answer depends on the tokenizer and the markers alone, so it is decided once per model and kept
/// with them ([`Templates::markers_hidden`]). `None` when no table reads the response: the legacy
/// parsers answer for themselves then, and where a table reads they are not built to be asked,
/// since the table's markers alone decide what its reader needs. The tests ask through
/// `keeps_special_tokens`, which reads the decision first.
pub(crate) fn keeps_special_tokens_for(
    read: &Reading,
    model: &str,
    templates: &Templates,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Option<bool> {
    match read {
        Reading::Table(format, _) => {
            Some(templates.markers_hidden(model, format, tokenizer, || {
                markers_hidden_by_the_decoder(format, tokenizer.as_ref())
            }))
        }
        Reading::Legacy | Reading::Fallback(_) => None,
    }
}

/// Whether one of `format`'s markers loses text when its tokens are decoded with the special
/// tokens dropped: the marker as the tokenizer encodes it decodes back to itself with them kept.
pub(super) fn markers_hidden_by_the_decoder(format: &Format, tokenizer: &dyn Tokenizer) -> bool {
    format.terminal_texts().any(|marker| {
        let Ok(encoding) = tokenizer.encode(marker, false) else {
            return false;
        };
        let ids = encoding.token_ids();
        let kept = tokenizer.decode(ids, false);
        let dropped = tokenizer.decode(ids, true);
        matches!((kept, dropped), (Ok(kept), Ok(dropped)) if kept == marker && dropped != marker)
    })
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/symphony/names/tests.rs"]
mod tests;
