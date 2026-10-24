//! Name resolution: which table reads a model's responses, from the parser names set for it and
//! the registry, and the checks on those names at startup and at a worker's registration.

use std::{borrow::Cow, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use symphony::{
    formats::{for_model_shared, table_for, Table},
    Format,
};

use super::{telemetry, template::Templates, SymphonyError};
use crate::{
    config::{ParserChoice, TEMPLATE_PARSER},
    routers::grpc::utils::{answering_to, ModelNames, Names},
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
fn names_template(names: &Names) -> bool {
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
/// channel.
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
        Some(table) => Selection::Table(table),
        None if names.tool.is_none() && !is_passthrough(names.reasoning.as_deref()) => {
            Selection::Nothing
        }
        None => Selection::Legacy,
    }
}

/// Whether `names`, one source's, select nothing (no name at all, or a reasoning name alone that no
/// table answers to, `passthrough` excepted), so that [`decision`] passes them over for the next
/// source: a worker labeled so is read by the registry's table, as an unlabeled one is.
pub(crate) fn labels_select_nothing(names: &Names) -> bool {
    matches!(selection(names), Selection::Nothing)
}

/// Why the legacy parsers read a request Symphony was configured for, by the site that decided
/// it: the precedence among the names and the registry ([`decision`]), or the table's missing
/// tag ([`assign`](super::assign)). A request so read is counted once under its reason
/// ([`telemetry::record_fallback`]); its response's header
/// says `legacy`, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fallback {
    /// No table reads the model and no name is set for it: the registry does not know the model,
    /// and neither its labels nor the flags name anything.
    NoTable,
    /// A reasoning name alone that no table answers to (`deepseek_r1`, `qwen3_thinking`), in the
    /// labels or the flags, for a model the registry does not know: every table reads the thought
    /// channel, so the name selects nothing, and nothing else selects a table.
    ReasoningNameOnly,
    /// A legacy tool-call parser's name no table answers to (`pythonic`): the way back
    /// to the legacy parsers for one model (the labels) or for the models the registry does not
    /// know (the flags).
    LegacyToolName,
    /// `passthrough`, as the tool-call name or alone as the reasoning name: the choice to leave a
    /// channel as written, which no table does.
    Passthrough,
    /// A table reads the model and has no tag for the request's forced or required call: the older
    /// registry's JSON schema constrains the output, which then has no markers for the table to
    /// read, so the legacy parsers read it ([`assign`](super::assign)).
    ForcedCallWithoutTag,
}

/// What reads a model's responses under Symphony ([`decision`]).
pub(super) enum Decision {
    /// The response template's table, compiled when the request comes ([`Templates`]).
    Template,
    /// A table, shared with every request and engine that reads with it: the registry's, or
    /// the one the names select.
    Table(Arc<Format>),
    /// The legacy parsers, for the reason the precedence names: never
    /// [`Fallback::ForcedCallWithoutTag`], which the assignment decides.
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
/// workers' labels), when they select anything; else the registry's table, for a model it knows
/// (`symphony::formats::for_model`); else the gateway's `--tool-call-parser` and
/// `--reasoning-parser`, for a model the registry does not know; else the legacy parsers, with
/// their own detection, the reason named ([`Fallback`]). So a fleet-wide name reads the models
/// the registry has no table for and never a family it has one for, and a worker's labels read
/// their model whatever the registry has. A table the labels select reads a model the registry
/// gives the same table with the model's own turn opener; another model, or one the registry
/// does not know, replays from the table's own. The registry is asked once per path, as before
/// the flags came apart.
pub(super) fn decision(names: &ModelNames<'_>, model: &str) -> Decision {
    match selection(&names.labels) {
        Selection::Template => Decision::Template,
        Selection::Legacy => Decision::Legacy(legacy_reason(&names.labels)),
        Selection::Table(table) => Decision::Table(match for_model_shared(model) {
            Some(format) if table_for(model) == Some(table) => format,
            _ => table.shared(),
        }),
        Selection::Nothing => match for_model_shared(model) {
            Some(format) => Decision::Table(format),
            // The registry does not know the model: the flags read it, a table they name from
            // its own opener.
            None => match selection(names.flags) {
                Selection::Template => Decision::Template,
                Selection::Table(table) => Decision::Table(table.shared()),
                Selection::Legacy => Decision::Legacy(legacy_reason(names.flags)),
                Selection::Nothing => Decision::Legacy(nothing_reason(names)),
            },
        },
    }
}

/// What reads a model's responses under a parser ([`reading`]).
pub(super) enum Reading {
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
pub(super) fn reading(
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
/// channels takes a variant of the table ([`assign`](super::assign)). Preparation and request building both
/// decide with this one predicate. The table comes shared, built once for every request and
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
/// (`workflow::steps::local::create_worker`).
pub(crate) fn table_label_read(names: &ModelNames<'_>, model: &str) -> Option<&'static str> {
    match decision(names, model) {
        Decision::Template => Some(telemetry::TEMPLATE),
        Decision::Table(format) => Some(telemetry::table_label(format.name())),
        Decision::Legacy(_) => None,
    }
}

/// What the gateway's flags select for `model`, a model the registry knows, that the precedence of
/// [`decision`] sets aside for a worker labeling no parser: the registry's table, which reads the
/// model, and the flags' choice spelled for a log line (`the table 'mistral_v13'`, `the model's
/// response template`, `the legacy parsers`). `None` when the flags select nothing or the
/// registry's own table, or when the registry does not know the model. Registration warns with it
/// once per model (`workflow::steps::local::create_worker`).
pub(crate) fn set_aside_by_the_registry(
    flags: &Names,
    model: &str,
) -> Option<(Table, Cow<'static, str>)> {
    let table = table_for(model)?;
    let spelled = match selection(flags) {
        Selection::Nothing => return None,
        Selection::Table(named) if named == table => return None,
        Selection::Table(named) => Cow::Owned(format!("the table '{}'", named.name())),
        Selection::Template => Cow::Borrowed("the model's response template"),
        Selection::Legacy => Cow::Borrowed("the legacy parsers"),
    };
    Some((table, spelled))
}

/// Whether the parser names set for a model (the flags, or one worker's labels) are names the
/// gateway reads with under `parser`: a legacy parser's, or, with Symphony on, a table's or
/// [`TEMPLATE_PARSER`]. `tools` and `reasoning` are the legacy registries; a name whose registry
/// is absent is not checked (its parsers are unused in that configuration). A name no legacy
/// parser has that a table answers to (the table's own name, or a launcher's name for a family
/// whose table has no older parser), and [`TEMPLATE_PARSER`], bind the other name to the same or
/// to nothing: the legacy parsers read a forced call the table has no tag for by the names they
/// are given, and have no parser of that name. A legacy parser's name a table answers to is
/// checked as the legacy name it is, in its own kind. The error says what is wrong with which
/// name, for the startup or registration failure.
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
    let legacy_tool = |name: &str| tools.is_some_and(|factory| factory.has_parser(name));
    let legacy_reasoning =
        |name: &str| reasoning.is_some_and(|factory| factory.registry().has_parser(name));
    // The table a name selects when no legacy parser of its kind has it: the table named so,
    // else the one that answers to it when no legacy parser of either kind has the name as the
    // registries spell it (a table answers in any case, the registries in theirs); a legacy
    // parser's name of the other kind selects none, whatever table answers to it.
    let table = |name: &str| {
        Table::named(name).or_else(|| {
            let spelled = name.trim().to_ascii_lowercase();
            (!legacy_tool(&spelled) && !legacy_reasoning(&spelled))
                .then(|| answering_to(name))
                .flatten()
        })
    };
    let kinds = [
        (
            "tool-call",
            names.tool.as_deref(),
            tools.is_none() || names.tool.as_deref().is_some_and(legacy_tool),
            names.reasoning.as_deref(),
        ),
        (
            "reasoning",
            names.reasoning.as_deref(),
            reasoning.is_none() || names.reasoning.as_deref().is_some_and(legacy_reasoning),
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
        if legacy {
            continue;
        }
        match (symphony, table(name)) {
            (true, Some(selected)) => {
                if let Some(other) = other {
                    if table(other) != Some(selected) {
                        return Err(format!(
                            "{kind} parser '{name}' selects the Symphony table '{}' and is no \
                             legacy parser's name, so the other parser must name the same \
                             table or nothing, not '{other}'",
                            selected.name()
                        ));
                    }
                }
            }
            (true, None) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': neither a legacy parser nor a Symphony table"
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
/// message rendered in place). The grammar built for the request takes it OR'd with the request's
/// thinking switch, so the tag and the engine read the prompt the same way: a JSON answer the
/// switch would place at the first token follows the thought's close instead of streaming as
/// reasoning. `None` when no table reads the response, leaving the switch's answer as it was;
/// where a table reads, no legacy parser is built to be asked about the prompt: the table's
/// replay answers, alone for the family whose legacy rule set the switch aside
/// ([`PromptThought::alone`]).
pub(crate) fn reasoning_open(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    prompt: &str,
) -> Option<PromptThought> {
    match reads(parser, names, model, templates) {
        // The registry's Gemma 4 table carries the rule; a Gemma 4 model read through its own
        // response template (a table named for the template) is left to the switch beside the
        // replay, as every other table is: telling that model apart would take the legacy
        // registry's pattern match, which it offers only by building the parser, and no Gemma 4
        // tokenizer ships a response template today.
        Ok(Some(format)) => Some(PromptThought {
            open: format.reasoning_open(prompt),
            closed: format.thought_closed(prompt),
            alone: format.name() == Table::Gemma4.name(),
        }),
        _ => None,
    }
}

/// The thought's state in the rendered prompt as the table that reads the response sees it
/// ([`reasoning_open`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptThought {
    /// The prompt leaves the model inside its thought.
    pub open: bool,
    /// The prompt closed the thought itself and wrote what the template writes after the close
    /// (thinking switched off), so the tag owes none of it ([`Format::thought_closed`]).
    pub closed: bool,
    /// The replay decides alone, without the request's thinking switch: Gemma 4 prefills an open
    /// thought channel only after a tool response, so thinking enabled on a new turn says nothing
    /// about the prompt. The rule its legacy reasoning parser applied in place of the switch,
    /// kept here without building one.
    pub alone: bool,
}

impl PromptThought {
    /// Whether the prompt leaves the model inside its thought, as request building decides it:
    /// the replay's answer, or the request's thinking `switch` where the replay does not decide
    /// alone.
    pub fn or_switch(self, switch: impl FnOnce() -> bool) -> bool {
        self.open || (!self.alone && switch())
    }
}

/// Whether the table that reads `model`'s responses under `parser` ([`reads`]) needs the
/// vocabulary's special tokens kept in the decoded output: one of its markers is, or holds, a
/// token the stop decoder would drop with `skip_special_tokens` (Muse Spark's `<|message|>` and
/// `<|eom|>`, Kimi K3's `<|open|>` and `<|sep|>`). A legacy reasoning parser says so of itself
/// (`requires_special_tokens`); a table says nothing, so the gateway asks the tokenizer: a marker
/// whose tokens decode to its text with the special tokens kept and to something else with them
/// dropped is such a marker. `false` when no table reads the response, or when every marker
/// survives the decoder either way (Qwen3's `<think>` and `<tool_call>` are added tokens the
/// decoder keeps), so a model the older rule already serves is read as before. Asking the
/// tokenizer encodes and decodes every marker, and the answer depends on the tokenizer and the
/// markers alone, so it is decided once per model and kept with them
/// ([`Templates::markers_hidden`]). `None` when no table reads the response: the legacy parsers
/// answer for themselves then, and where a table reads they are not built to be asked, since the
/// table's markers alone decide what its reader needs.
pub(crate) fn keeps_special_tokens(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Option<bool> {
    match reads(parser, names, model, templates) {
        Ok(Some(format)) => Some(templates.markers_hidden(model, &format, tokenizer, || {
            markers_hidden_by_the_decoder(&format, tokenizer.as_ref())
        })),
        _ => None,
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
mod tests {
    use super::*;
    use crate::routers::grpc::{
        regular::symphony::testing::*,
        utils::{ALIASES, UNANSWERED},
    };

    #[test]
    fn the_tool_call_name_selects_the_table_that_answers_to_it() {
        assert_eq!(named_table(&Names::default()), None);
        assert_eq!(
            named_table(&pair(Some("kimi_k3"), Some("kimi_k3"))),
            Some(Table::KimiK3)
        );
        // The Kimi K2 line's older parsers: the tool-call name, and its thought parsers alone;
        // the `kimi` thought parser reads another family's markers and selects none.
        assert_eq!(
            named_table(&pair(Some("kimik2"), Some("kimi_thinking"))),
            Some(Table::KimiK2)
        );
        assert_eq!(
            named_table(&pair(None, Some("kimi_k25"))),
            Some(Table::KimiK2)
        );
        assert_eq!(named_table(&pair(None, Some("kimi"))), None);
        assert_eq!(named_table(&pair(None, Some("Ling"))), Some(Table::Ling));
        assert_eq!(
            named_table(&pair(Some("qwen3_tagged"), None)),
            Some(Table::Qwen3Tagged)
        );
        // A legacy name reads with the table of the format it reads.
        assert_eq!(named_table(&pair(Some("qwen"), None)), Some(Table::Qwen3));
        assert_eq!(
            named_table(&pair(Some("qwen"), Some("qwen3"))),
            Some(Table::Qwen3)
        );
        assert_eq!(
            named_table(&pair(Some(" QWEN_XML "), Some("qwen3"))),
            Some(Table::Qwen3Tagged)
        );
        assert_eq!(
            named_table(&pair(Some("glm45_moe"), Some("glm45"))),
            Some(Table::Glm4)
        );
        assert_eq!(
            named_table(&pair(Some("glm47_moe"), Some("glm45"))),
            Some(Table::Glm)
        );
        // The older DSML parser for the V3.2 checkpoints reads with their table.
        assert_eq!(
            named_table(&names(Some("deepseek32"), None).labels),
            Some(Table::DeepSeekV3_2)
        );
        // The reasoning name counts only without a tool-call name.
        assert_eq!(named_table(&pair(None, Some("hy_v4"))), Some(Table::Hy4));
        assert_eq!(
            named_table(&pair(None, Some("minimax"))),
            Some(Table::MinimaxM2)
        );
        assert_eq!(named_table(&pair(None, Some("deepseek_r1"))), None);
        // A launcher's name for a family whose table has no older parser reads with the table:
        // in both flags, in the reasoning flag alone, or as the tool-call name beside another.
        assert_eq!(
            named_table(&pair(Some("muse"), Some("muse"))),
            Some(Table::MuseSpark)
        );
        assert_eq!(
            named_table(&pair(None, Some("muse"))),
            Some(Table::MuseSpark)
        );
        assert_eq!(
            named_table(&pair(Some("muse"), Some("passthrough"))),
            Some(Table::MuseSpark)
        );
        // The tool-call name decides; a legacy name with no alias selects none.
        assert_eq!(
            named_table(&pair(Some("kimi_k3"), Some("qwen3"))),
            Some(Table::KimiK3)
        );
        // The older `pythonic` parser's name serves more than one family, so it keeps the
        // legacy parsers until every one of them has a table; the Llama 4 table answers to
        // its exact alias.
        assert_eq!(named_table(&pair(Some("pythonic"), None)), None);
        assert_eq!(named_table(&pair(Some("pythonic"), Some("qwen3"))), None);
        assert_eq!(
            named_table(&pair(Some("llama4_pythonic"), None)),
            Some(Table::Llama4)
        );
        // The older `llama` parser's format is the json table's.
        assert_eq!(named_table(&pair(Some("llama"), None)), Some(Table::Json));
        // The older `mistral` parser's list has a table of that name; the v13 spelling keeps its
        // own.
        assert_eq!(
            named_table(&pair(Some("mistral"), None)),
            Some(Table::Mistral)
        );
        assert_eq!(
            named_table(&pair(Some("mistral_v13"), None)),
            Some(Table::MistralV13)
        );
        // The older DeepSeek pairings read with the tables, the thought included: the R1 line
        // behind `deepseek` + `deepseek_r1` writes V3's calls after a thought, and the V3 table
        // reads it, whether the prompt opens the thought (R1, R1-Zero and the distills: the
        // original R1's tail, below) or the model writes `<think>` itself (R1-0528,
        // R1-0528-Qwen3-8B and the bf16 mirror).
        assert_eq!(
            named_table(&pair(Some("deepseek"), Some("deepseek_r1"))),
            Some(Table::DeepSeekV3)
        );
        assert!(Table::DeepSeekV3
            .format()
            .reasoning_open("<｜Assistant｜><think>\n"));
        assert_eq!(
            named_table(&pair(Some("deepseek31"), Some("deepseek_v31"))),
            Some(Table::DeepSeekV3_1)
        );
        // `passthrough` leaves a channel as written, which no table reads for: the legacy
        // parsers keep it, alone or beside another name.
        assert_eq!(
            named_table(&pair(Some("passthrough"), Some(" Passthrough "))),
            None
        );
        assert_eq!(named_table(&pair(Some("passthrough"), None)), None);
        assert_eq!(named_table(&pair(None, Some("passthrough"))), None);
        assert_eq!(named_table(&pair(Some("passthrough"), Some("qwen3"))), None);
    }

    #[test]
    fn every_name_a_table_answers_to_besides_its_own_is_a_legacy_parsers_or_a_launchers() {
        let tools = tool_parser::ParserFactory::new();
        let reasoning = reasoning_parser::ParserFactory::new();
        let symphony = ParserChoice::Symphony;
        let mut launchers = Vec::new();
        for (alias, table) in ALIASES {
            if tools.has_parser(alias) || reasoning.registry().has_parser(alias) {
                // A legacy parser's name is checked as the legacy name it is
                // (`names_are_checked_against_both_registries_by_the_parser_in_force`).
                continue;
            }
            // A name no legacy registry has (a launcher's for a family whose table has no older
            // parser, an engine's exact name for a table) passes under Symphony in both flags, in
            // either alone, and beside the table's own name.
            launchers.push(alias);
            let doubled = names(Some(alias), Some(alias));
            assert_eq!(named_table(&doubled.labels), Some(table), "{alias}");
            assert_eq!(check(symphony, &doubled), Ok(()), "{alias}");
            assert_eq!(
                check(symphony, &names(Some(alias), None)),
                Ok(()),
                "{alias}"
            );
            assert_eq!(
                check(symphony, &names(None, Some(alias))),
                Ok(()),
                "{alias}"
            );
            let own = table.name();
            assert_eq!(
                check(symphony, &names(Some(alias), Some(own))),
                Ok(()),
                "{alias}"
            );
            assert_eq!(
                check(symphony, &names(Some(own), Some(alias))),
                Ok(()),
                "{alias}"
            );
        }
        assert_eq!(launchers, ["llama4_pythonic", "muse"]);
    }

    /// Every name of the two legacy registries answers to the table the flip maps it to (its
    /// own, or the one whose format it reads), so that a deployment's existing flags keep a
    /// working parser under Symphony, or is in `UNANSWERED` with its reason; never both,
    /// never neither, so a new legacy parser's name cannot appear unmapped, and a table that
    /// takes a listed name removes its row. In its own kind every legacy name passes the startup
    /// check under Symphony; a tool-call name decides beside any reasoning name, and a reasoning
    /// name alone selects its table or none.
    #[test]
    fn every_legacy_parser_name_answers_to_its_table_or_is_listed_as_unanswered() {
        let tools = tool_parser::ParserFactory::new();
        let reasoning = reasoning_parser::ParserFactory::new();
        let symphony = ParserChoice::Symphony;
        let mapped = [
            ("json", "json"),
            ("llama", "json"),
            ("gemma4", "gemma4"),
            ("qwen", "qwen3"),
            ("qwen3", "qwen3"),
            ("qwen_xml", "qwen3_tagged"),
            ("qwen_coder", "qwen3_tagged"),
            ("nemotron", "qwen3_tagged"),
            ("deepseek_v4", "deepseek_v4"),
            ("deepseek_v41", "deepseek_v4.1"),
            ("deepseek32", "deepseek_v3.2"),
            ("glm45_moe", "glm4"),
            ("glm47_moe", "glm"),
            ("hy_v4", "hy4"),
            ("kimi_k3", "kimi_k3"),
            ("kimik2", "kimi_k2"),
            ("kimi_thinking", "kimi_k2"),
            ("kimi_k25", "kimi_k2"),
            ("inkling", "inkling"),
            ("minimax", "minimax_m2"),
            ("minimax_m2", "minimax_m2"),
            ("minimax_m3", "minimax_m3"),
            ("deepseek", "deepseek_v3"),
            ("deepseek31", "deepseek_v3.1"),
            ("deepseek_v31", "deepseek_v3.1"),
            ("mistral", "mistral"),
        ];
        let mut legacy: Vec<(String, &str)> = tools
            .list_parsers()
            .into_iter()
            .map(|name| (name, "tool-call"))
            .chain(
                reasoning
                    .list_parsers()
                    .into_iter()
                    .map(|name| (name, "reasoning")),
            )
            .collect();
        legacy.sort();
        for (name, kind) in &legacy {
            let as_given = match *kind {
                "tool-call" => names(Some(name), Some("qwen3")),
                _ => names(None, Some(name)),
            };
            assert_eq!(check(symphony, &as_given), Ok(()), "{kind} parser {name}");
            let expected = mapped
                .iter()
                .find(|(own, _)| own == name)
                .map(|(_, table)| *table);
            let listed = UNANSWERED.iter().find(|(own, _)| own == name);
            match (answering_to(name), listed) {
                (Some(table), None) => {
                    assert_eq!(
                        Some(table.name()),
                        expected,
                        "{kind} parser {name} answers to {}",
                        table.name()
                    );
                    assert_eq!(
                        named_table(&as_given.labels),
                        Some(table),
                        "{kind} parser {name}"
                    );
                }
                (None, Some((_, reason))) => {
                    assert_eq!(expected, None, "{kind} parser {name} is listed: {reason}");
                    assert_eq!(named_table(&as_given.labels), None, "{kind} parser {name}");
                }
                (Some(table), Some(_)) => panic!(
                    "{kind} parser {name} answers to {} and is listed as unanswered: remove \
                     its row from UNANSWERED",
                    table.name()
                ),
                (None, None) => panic!(
                    "{kind} parser {name} answers to no table and is not listed: add it to \
                     ALIASES or to UNANSWERED with the reason"
                ),
            }
        }
        // Neither list outlives the registries.
        let is_legacy = |name: &str| legacy.iter().any(|(own, _)| own == name);
        for (name, _) in mapped {
            assert!(is_legacy(name), "{name} is no legacy parser's name");
        }
        for (name, _) in UNANSWERED {
            assert!(
                is_legacy(name),
                "{name} is no legacy parser's name: remove its row"
            );
        }
    }

    #[test]
    fn names_are_checked_against_both_registries_by_the_parser_in_force() {
        let legacy = ParserChoice::Legacy;
        let symphony = ParserChoice::Symphony;
        assert_eq!(check(legacy, &ModelNames::default()), Ok(()));
        assert_eq!(check(legacy, &names(Some("qwen"), Some("qwen3"))), Ok(()));
        let why = check(legacy, &names(Some("nonsense"), None));
        assert_eq!(
            why.expect_err("unknown"),
            "unknown tool-call parser 'nonsense'"
        );
        let why = check(legacy, &names(None, Some("ling")));
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'ling': the Symphony table 'ling', which --parser symphony \
             reads with"
        );
        // Under Symphony a table's name passes alone or doubled, and binds the other name.
        assert_eq!(check(symphony, &names(Some("ling"), None)), Ok(()));
        assert_eq!(check(symphony, &names(Some("ling"), Some("LING"))), Ok(()));
        // Legacy names pass as the legacy names they are, one a table's own name as well.
        assert_eq!(
            check(symphony, &names(Some("qwen"), Some("kimi_k3"))),
            Ok(())
        );
        let why = check(symphony, &names(Some("ling"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'ling' selects the Symphony table 'ling' and is no legacy parser's \
             name, so the other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(None, Some("nonsense")));
        assert_eq!(
            why.expect_err("unknown"),
            "unknown reasoning parser 'nonsense': neither a legacy parser nor a Symphony table"
        );
        // A legacy name a table answers to is checked as the legacy name it is: a tool parser's
        // given as the reasoning parser is unknown there, under either parser.
        let why = check(symphony, &names(None, Some("qwen_xml")));
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml': neither a legacy parser nor a Symphony table"
        );
        let why = check(legacy, &names(None, Some("qwen_xml")));
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml'"
        );
        // The legacy registries spell their names one way: a legacy parser's name in another
        // case or with spaces around it is no legacy parser's, and selects no table either
        // (a table's own name, which reads in any case, excepted).
        let why = check(symphony, &names(Some("QWEN_XML"), None));
        assert_eq!(
            why.expect_err("not as spelled"),
            "unknown tool-call parser 'QWEN_XML': neither a legacy parser nor a Symphony table"
        );
        let why = check(symphony, &names(None, Some(" hy_v4 ")));
        assert_eq!(
            why.expect_err("not as spelled"),
            "unknown reasoning parser ' hy_v4 ': neither a legacy parser nor a Symphony table"
        );
        // A launcher's name for a family whose table has no older parser (`muse`) passes in
        // either flag and binds the other name as the table's own name does; under `legacy` it
        // is unknown, as the table's own name is.
        assert_eq!(check(symphony, &names(Some("muse"), Some("muse"))), Ok(()));
        assert_eq!(
            check(symphony, &names(Some("MUSE"), Some("muse_spark"))),
            Ok(())
        );
        assert_eq!(
            check(symphony, &names(Some("muse_spark"), Some("muse"))),
            Ok(())
        );
        let why = check(symphony, &names(Some("muse"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             parser's name, so the other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(Some("muse"), Some("passthrough")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             parser's name, so the other parser must name the same table or nothing, not \
             'passthrough'"
        );
        let why = check(legacy, &names(None, Some("muse")));
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'muse': the Symphony table 'muse_spark', which --parser \
             symphony reads with"
        );
        // A registry that is absent checks nothing, as before.
        assert_eq!(
            check_names(symphony, &pair(Some("nonsense"), Some("muse")), None, None),
            Ok(())
        );
        assert_eq!(
            check_names(legacy, &pair(Some("muse"), Some("nonsense")), None, None),
            Ok(())
        );
    }

    #[test]
    fn the_header_names_the_parser_over_the_flag_and_an_unknown_name_is_refused() {
        assert_eq!(choice(ParserChoice::Legacy, None), Ok(ParserChoice::Legacy));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&HeaderMap::new())),
            Ok(ParserChoice::Symphony)
        );
        let mut headers = HeaderMap::new();
        headers.insert(HEADER, "symphony".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Legacy, Some(&headers)),
            Ok(ParserChoice::Symphony)
        );
        headers.insert(HEADER, " Legacy ".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&headers)),
            Ok(ParserChoice::Legacy)
        );
        headers.insert(HEADER, "fast".parse().expect("a header value"));
        let why = choice(ParserChoice::Legacy, Some(&headers)).expect_err("refused");
        assert_eq!(
            why,
            SymphonyError::Header(
                "x-smg-parser must be legacy or symphony, not \"fast\"".to_string()
            )
        );
    }

    #[test]
    fn a_table_whose_marker_the_decoder_drops_keeps_the_special_tokens() {
        let tokenizer = marker_tokenizer();
        // A marker the decoder drops with the special tokens: the table needs them kept.
        assert!(markers_hidden_by_the_decoder(
            &table_with_markers("<bos>", "<eos>"),
            &tokenizer
        ));
        // Markers the decoder keeps either way, and a marker the tokenizer does not know,
        // which decodes to nothing with or without them: read as before.
        assert!(!markers_hidden_by_the_decoder(
            &table_with_markers("<think>", "</think>"),
            &tokenizer
        ));
        assert!(!markers_hidden_by_the_decoder(
            &table_with_markers("<unknown>", "</unknown>"),
            &tokenizer
        ));
        // Through the names: Qwen3's table over this vocabulary needs nothing, and no table
        // (the legacy parsers, or `--parser legacy`) answers nothing: the legacy parsers decide
        // for themselves then.
        let shared: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
        assert_eq!(
            keeps_special_tokens(
                ParserChoice::Symphony,
                &names(Some("qwen3"), None),
                "Qwen/Qwen3-8B",
                &templates(),
                &shared,
            ),
            Some(false)
        );
        assert_eq!(
            keeps_special_tokens(
                ParserChoice::Symphony,
                &names(Some("passthrough"), None),
                "Qwen/Qwen3-8B",
                &templates(),
                &shared,
            ),
            None
        );
        assert_eq!(
            keeps_special_tokens(
                ParserChoice::Legacy,
                &names(Some("qwen3"), None),
                "Qwen/Qwen3-8B",
                &templates(),
                &shared,
            ),
            None
        );
    }

    #[test]
    fn a_table_whose_markers_the_decoder_hides_answers_true_and_the_legacy_names_answer_nothing() {
        // The Kimi K3 table's markers are special tokens of this vocabulary (`<bos>` stands for
        // them here): the table that reads the response says so itself, whatever a legacy parser
        // of the same name would require; under the legacy parsers there is no table to ask.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
        let hidden = table_with_markers("<bos>", "<eos>");
        assert!(markers_hidden_by_the_decoder(&hidden, tokenizer.as_ref()));
        assert_eq!(
            keeps_special_tokens(
                ParserChoice::Legacy,
                &names(Some("kimi_k3"), Some("kimi_k3")),
                "m",
                &templates(),
                &tokenizer,
            ),
            None
        );
        assert_eq!(
            reasoning_open(
                ParserChoice::Legacy,
                &names(Some("kimi_k3"), Some("kimi_k3")),
                "m",
                &templates(),
                "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
            ),
            None
        );
        // Where the table reads, its replay of the prompt answers: Kimi K3's prompt opens the
        // thought, a prompt that writes no such opener does not; the request's thinking switch
        // is heard beside the replay.
        let read = |table: &str, prompt: &str| {
            reasoning_open(
                ParserChoice::Symphony,
                &names(Some(table), Some(table)),
                "m",
                &templates(),
                prompt,
            )
            .expect("a table reads")
        };
        let opened = read(
            "kimi_k3",
            "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
        );
        assert!(opened.open && !opened.alone);
        let unopened = read("kimi_k3", "<|open|>message role=\"assistant\"<|sep|>");
        assert!(!unopened.open && !unopened.alone);
        assert!(unopened.or_switch(|| true));
        assert!(!unopened.or_switch(|| false));
        // Gemma 4 prefills an open thought channel only after a tool response: its table's
        // replay decides alone, and thinking enabled on a new turn does not open the thought.
        let channel = read("gemma4", "<|turn>model\n<|channel>thought\n");
        assert!(channel.open && channel.alone);
        let turn = read("gemma4", "<|turn>model\n");
        assert!(!turn.open && turn.alone);
        assert!(!turn.or_switch(|| true));
        assert!(channel.or_switch(|| false));
    }

    #[test]
    fn a_decided_true_is_answered_as_true_without_asking_again() {
        let templates = templates();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
        let asked = std::cell::Cell::new(0);
        let decide = |format: &Format| {
            templates.markers_hidden("m", format, &tokenizer, || {
                asked.set(asked.get() + 1);
                markers_hidden_by_the_decoder(format, tokenizer.as_ref())
            })
        };
        // Markers this vocabulary hides: decided true once, answered true after.
        let hidden = table_with_markers("<bos>", "<eos>");
        assert!(decide(&hidden));
        assert!(decide(&hidden));
        assert_eq!(asked.get(), 1);
        // Markers it keeps, for the same model: decided again, false, once.
        let kept = table_with_markers("<think>", "</think>");
        assert!(!decide(&kept));
        assert!(!decide(&kept));
        assert_eq!(asked.get(), 2);
    }

    #[test]
    fn the_decoders_answer_is_decided_once_per_model_tokenizer_and_markers() {
        let templates = templates();
        let (counting, tokenizer) = Counting::shared();
        let qwen3 = names(Some("qwen3"), None);
        let decide = |names: &ModelNames<'_>, tokenizer: &Arc<dyn Tokenizer>| {
            keeps_special_tokens(
                ParserChoice::Symphony,
                names,
                "Qwen/Qwen3-8B",
                &templates,
                tokenizer,
            )
        };
        // The first request asks the tokenizer about every marker of the table; the next ones
        // for the same model, tokenizer and markers ask nothing.
        assert_eq!(decide(&qwen3, &tokenizer), Some(false));
        let asked = counting.encodes();
        assert_eq!(asked, Table::Qwen3.format().terminal_texts().count());
        assert_eq!(decide(&qwen3, &tokenizer), Some(false));
        assert_eq!(decide(&qwen3, &tokenizer), Some(false));
        assert_eq!(counting.encodes(), asked);
        // A table with the same markers is decided: the tagged syntax over the same table, and
        // GLM's table, whose markers are Qwen3's (its tags are the call syntax's).
        assert_eq!(
            decide(&names(Some("qwen3_tagged"), None), &tokenizer),
            Some(false)
        );
        assert_eq!(decide(&names(Some("glm"), None), &tokenizer), Some(false));
        assert_eq!(counting.encodes(), asked);
        // The same model and markers under a replaced tokenizer are asked about again: the
        // entry holds Qwen3's markers still, so the tokenizer alone decides that.
        let (replaced, other) = Counting::shared();
        assert_eq!(decide(&qwen3, &other), Some(false));
        assert_eq!(replaced.encodes(), asked);
        assert_eq!(counting.encodes(), asked);
        // A table with other markers is asked about again, on the tokenizer in place.
        assert_eq!(
            decide(&names(Some("mistral_v13"), None), &other),
            Some(false)
        );
        let mistral = Table::MistralV13.format().terminal_texts().count();
        assert_eq!(replaced.encodes(), asked + mistral);
        // Another model asks for itself.
        assert_eq!(
            keeps_special_tokens(
                ParserChoice::Symphony,
                &qwen3,
                "Qwen/Qwen3-4B",
                &templates,
                &other,
            ),
            Some(false)
        );
        assert_eq!(replaced.encodes(), 2 * asked + mistral);
    }

    #[test]
    fn hf_is_a_name_under_symphony_alone_and_binds_the_other_name() {
        let symphony = ParserChoice::Symphony;
        assert_eq!(check(symphony, &names(Some("hf"), None)), Ok(()));
        assert_eq!(check(symphony, &names(None, Some("hf"))), Ok(()));
        assert_eq!(check(symphony, &names(Some("HF"), Some(" hf "))), Ok(()));
        let why = check(symphony, &names(Some("hf"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound"),
            "tool-call parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(Some("qwen"), Some("hf")));
        assert_eq!(
            why.expect_err("bound"),
            "reasoning parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen'"
        );
        let why = check(ParserChoice::Legacy, &names(Some("hf"), None));
        assert_eq!(
            why.expect_err("legacy"),
            "unknown tool-call parser 'hf': the model's response template, which --parser \
             symphony reads with"
        );
    }

    /// The precedence among the parser names set for a model, which [`super::reads`] states
    /// once: the labels of the model's workers read over the registry; the registry's table
    /// reads over the gateway's flags, for a model it knows; the flags read a model it does not
    /// know; and nothing names a table for the rest. Each rule in turn.
    #[test]
    fn the_labels_read_over_the_registry_which_reads_over_the_flags_for_the_models_it_knows() {
        let known = "Qwen/Qwen3-8B";
        let unknown = "acme/ministral-3-14b-instruct-w4a16";
        assert_eq!(table_for(known), Some(Table::Qwen3));
        assert_eq!(table_for(unknown), None);
        let read = |names: &ModelNames<'_>, model: &str| {
            reads(ParserChoice::Symphony, names, model, &templates())
                .map(|format| format.map(|format| format.name().to_string()))
        };
        let table = |name: &str| Ok(Some(name.to_string()));
        let v13 = pair(Some("mistral_v13"), None);
        let no_flags = Names::default();
        fn labeled<'a>(labels: Names, flags: &'a Names) -> ModelNames<'a> {
            ModelNames { labels, flags }
        }

        // The labels over the registry: a table they name reads a model the registry reads with
        // another; a legacy tool-call name, or `passthrough` alone, is their way back to the
        // legacy parsers for the one model; a reasoning name alone that a table answers to
        // selects it over a tool-call name the flags set.
        assert_eq!(
            read(&labeled(v13.clone(), &no_flags), known),
            table("mistral_v13")
        );
        assert_eq!(
            read(&labeled(pair(Some("pythonic"), None), &v13), known),
            Ok(None)
        );
        assert_eq!(
            read(&labeled(pair(None, Some("passthrough")), &v13), known),
            Ok(None)
        );
        assert_eq!(
            read(&labeled(pair(None, Some("qwen3")), &v13), unknown),
            table("qwen3")
        );

        // The registry over the flags, for a model it knows, whatever they name: a table, the
        // response template (no refusal for a model without one), a legacy parser.
        for flags in [
            v13.clone(),
            pair(Some(TEMPLATE_PARSER), Some(TEMPLATE_PARSER)),
            pair(Some("cohere"), None),
            pair(Some("passthrough"), None),
        ] {
            assert_eq!(
                read(&ModelNames::flagged(&flags), known),
                table("qwen3"),
                "{flags:?}"
            );
        }

        // The flags for a model the registry does not know: a table's name reads it; `hf` asks
        // its response template, refused when it has none; a legacy name keeps the legacy
        // parsers; a reasoning name alone that a table answers to selects it.
        assert_eq!(
            read(&ModelNames::flagged(&v13), unknown),
            table("mistral_v13")
        );
        assert!(read(
            &ModelNames::flagged(&pair(Some(TEMPLATE_PARSER), None)),
            unknown
        )
        .is_err());
        assert_eq!(
            read(&ModelNames::flagged(&pair(Some("cohere"), None)), unknown),
            Ok(None)
        );
        assert_eq!(
            read(&ModelNames::flagged(&pair(None, Some("qwen3"))), unknown),
            table("qwen3")
        );

        // Labels that select nothing (a reasoning name alone that no table answers to) leave the
        // decision to the registry, then to the flags.
        let reasoning_only = pair(None, Some("deepseek_r1"));
        assert_eq!(
            read(&labeled(reasoning_only.clone(), &v13), known),
            table("qwen3")
        );
        assert_eq!(
            read(&labeled(reasoning_only, &v13), unknown),
            table("mistral_v13")
        );

        // Nothing named: the registry's table for a model it knows, the legacy parsers for one
        // it does not.
        assert_eq!(read(&ModelNames::default(), known), table("qwen3"));
        assert_eq!(read(&ModelNames::default(), unknown), Ok(None));
    }

    /// The legacy parsers read for a reason the decision names: one source's way back
    /// (`passthrough`, or a tool-call name no table answers to, the tool-call name deciding), or
    /// nothing anywhere for a model the registry does not know (a reasoning name alone that no
    /// table answers to, in either source, else no name at all). A table's reading names none,
    /// and the gateway's own choice of `legacy` is no fallback.
    #[test]
    fn the_legacy_parsers_read_for_a_reason_the_decision_names() {
        let known = "Qwen/Qwen3-8B";
        let unknown = "acme/ministral-3-14b-instruct-w4a16";
        let why = |names: &ModelNames<'_>, model: &str| match decision(names, model) {
            Decision::Legacy(why) => Some(why),
            Decision::Template | Decision::Table(_) => None,
        };
        fn labeled<'a>(labels: Names, flags: &'a Names) -> ModelNames<'a> {
            ModelNames { labels, flags }
        }

        // The labels' way back, for a model the registry knows as for one it does not.
        for model in [known, unknown] {
            assert_eq!(
                why(&names(Some("pythonic"), None), model),
                Some(Fallback::LegacyToolName)
            );
            assert_eq!(
                why(&names(Some("pythonic"), Some("passthrough")), model),
                Some(Fallback::LegacyToolName)
            );
            assert_eq!(
                why(&names(Some("passthrough"), None), model),
                Some(Fallback::Passthrough)
            );
            assert_eq!(
                why(&names(Some("passthrough"), Some("qwen3")), model),
                Some(Fallback::Passthrough)
            );
            assert_eq!(
                why(&names(None, Some("passthrough")), model),
                Some(Fallback::Passthrough)
            );
        }

        // The flags' way back, for a model the registry does not know; the registry's table
        // reads one it knows whatever they name. The tool-call name is one no table answers
        // to (`pythonic`): a name a table answers to (`mistral`) names that table instead.
        let pythonic = pair(Some("pythonic"), None);
        let passthrough = pair(None, Some("passthrough"));
        assert_eq!(
            why(&ModelNames::flagged(&pythonic), unknown),
            Some(Fallback::LegacyToolName)
        );
        assert_eq!(
            why(&ModelNames::flagged(&passthrough), unknown),
            Some(Fallback::Passthrough)
        );
        assert_eq!(why(&ModelNames::flagged(&pythonic), known), None);
        assert_eq!(why(&ModelNames::flagged(&passthrough), known), None);

        // Nothing anywhere: a reasoning name alone that no table answers to, in the labels or
        // in the flags, else no name at all; the flags' legacy name decides after labels that
        // select nothing.
        let r1 = pair(None, Some("deepseek_r1"));
        assert_eq!(
            why(&ModelNames::default(), unknown),
            Some(Fallback::NoTable)
        );
        assert_eq!(
            why(&ModelNames::flagged(&r1), unknown),
            Some(Fallback::ReasoningNameOnly)
        );
        assert_eq!(
            why(&names(None, Some("deepseek_r1")), unknown),
            Some(Fallback::ReasoningNameOnly)
        );
        assert_eq!(
            why(&labeled(r1, &pythonic), unknown),
            Some(Fallback::LegacyToolName)
        );

        // A table reads: the registry's for a model it knows, or the one a reasoning name alone
        // answers to.
        assert_eq!(why(&ModelNames::default(), known), None);
        assert_eq!(why(&names(None, Some("deepseek_r1")), known), None);
        assert_eq!(why(&names(None, Some("qwen3")), unknown), None);
        assert!(matches!(
            reading(
                ParserChoice::Symphony,
                &ModelNames::default(),
                unknown,
                &templates()
            ),
            Ok(Reading::Fallback(Fallback::NoTable))
        ));
        assert!(matches!(
            reading(
                ParserChoice::Legacy,
                &ModelNames::default(),
                unknown,
                &templates()
            ),
            Ok(Reading::Legacy)
        ));
    }

    /// What the gateway's flags select for a model the registry knows, set aside for a worker
    /// that labels no parser, is spelled for the registration warning beside the registry's
    /// table: a table, the response template, or the legacy parsers; nothing when the flags
    /// select nothing or the registry's own table, or for a model the registry does not know.
    #[test]
    fn the_flags_the_registry_sets_aside_are_spelled_for_the_warning() {
        let known = "Qwen/Qwen3-8B";
        let spelled = |flags: Names, model: &str| {
            set_aside_by_the_registry(&flags, model)
                .map(|(table, flags)| (table, flags.into_owned()))
        };
        let qwen3 = |flags: &str| Some((Table::Qwen3, flags.to_string()));
        assert_eq!(
            spelled(pair(Some("mistral_v13"), None), known),
            qwen3("the table 'mistral_v13'")
        );
        assert_eq!(
            spelled(pair(None, Some("ling")), known),
            qwen3("the table 'ling'")
        );
        assert_eq!(
            spelled(pair(Some(TEMPLATE_PARSER), None), known),
            qwen3("the model's response template")
        );
        assert_eq!(
            spelled(pair(Some("cohere"), None), known),
            qwen3("the legacy parsers")
        );
        assert_eq!(
            spelled(pair(None, Some("passthrough")), known),
            qwen3("the legacy parsers")
        );
        assert_eq!(spelled(pair(Some("qwen"), Some("qwen3")), known), None);
        assert_eq!(spelled(pair(None, Some("deepseek_r1")), known), None);
        assert_eq!(spelled(Names::default(), known), None);
        assert_eq!(
            spelled(
                pair(Some("mistral_v13"), None),
                "acme/ministral-3-14b-instruct-w4a16"
            ),
            None
        );
    }

    /// A failure converts to the text the gateway answered before its classes were named: a
    /// `String` where the drivers log it, a stream failure where a stream ends on it; a name's
    /// failure is a name's.
    #[test]
    fn a_failure_of_the_seam_converts_to_its_text_and_says_its_class() {
        let read = SymphonyError::Read("the stop decoder refused token 7".to_string());
        assert_eq!(read.to_string(), "the stop decoder refused token 7");
        assert_eq!(
            String::from(read.clone()),
            "the stop decoder refused token 7"
        );
        assert!(matches!(
            StreamFailure::from(read),
            StreamFailure::Internal(text) if text == "the stop decoder refused token 7"
        ));
        // The refused input converts to its bare text like the others; the non-streaming Messages
        // path alone matches on the class.
        let refused = SymphonyError::InputRefused(
            "the tool_use input of call 0 (f) cannot carry the arguments the model wrote: number out of range"
                .to_string(),
        );
        assert_eq!(
            String::from(refused.clone()),
            "the tool_use input of call 0 (f) cannot carry the arguments the model wrote: number out of range"
        );
        assert!(matches!(
            StreamFailure::from(refused),
            StreamFailure::Internal(text) if text.starts_with("the tool_use input of call 0 (f)")
        ));
        let tools = tool_parser::ParserFactory::new();
        assert!(matches!(
            check_names(
                ParserChoice::Legacy,
                &pair(Some("nonsense"), None),
                Some(&tools),
                None
            ),
            Err(SymphonyError::Name(text)) if text == "unknown tool-call parser 'nonsense'"
        ));
    }

    /// The label a model's outputs count under is known from its names at registration: the
    /// registry's table for a model it knows, the labels' table over it, the flags' table for a
    /// model the registry does not know, the template's name, and none for a model the legacy
    /// parsers read.
    #[test]
    fn the_table_label_a_model_counts_under_is_known_from_its_names() {
        let no_flags = Names::default();
        let flags = pair(Some("mistral_v13"), None);
        let read = |labels: Names, flags: &Names, model: &str| {
            table_label_read(&ModelNames { labels, flags }, model)
        };
        assert_eq!(
            read(Names::default(), &no_flags, "Qwen/Qwen3-8B"),
            Some("qwen3")
        );
        assert_eq!(
            read(pair(Some("kimi_k2"), None), &no_flags, "Qwen/Qwen3-8B"),
            Some("kimi_k2")
        );
        assert_eq!(
            read(Names::default(), &flags, "acme/unknown-model"),
            Some("mistral_v13")
        );
        assert_eq!(
            read(
                pair(Some(TEMPLATE_PARSER), None),
                &no_flags,
                "Qwen/Qwen3-8B"
            ),
            Some("response_template")
        );
        assert_eq!(
            read(Names::default(), &no_flags, "acme/unknown-model"),
            None
        );
        assert_eq!(
            read(pair(Some("passthrough"), None), &no_flags, "Qwen/Qwen3-8B"),
            None
        );
    }
}
