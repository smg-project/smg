//! Name resolution: which table reads a model's responses, from the parser names set for it and
//! the registry, and the checks on those names at startup and at a worker's registration.

use std::{borrow::Cow, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use symphony::{
    formats::{for_model_shared, table_for, Table},
    Format,
};

use super::{reader::asked, telemetry, template::Templates, SymphonyError};
use crate::{
    config::TEMPLATE_PARSER,
    routers::grpc::utils::{answering_to, ModelNames, Names, ALIASES, NOTED, UNANSWERED},
};

fn is_named(name: Option<&str>, what: &str) -> bool {
    name.is_some_and(|name| name.trim().eq_ignore_ascii_case(what))
}

/// Whether the parser names select the response template's table: either name is
/// [`TEMPLATE_PARSER`]; [`check_names`] binds the other to it or to nothing.
fn names_template(names: &Names) -> bool {
    is_named(names.tool.as_deref(), TEMPLATE_PARSER)
        || is_named(names.reasoning.as_deref(), TEMPLATE_PARSER)
}

/// The table the parser names set for a model select, when they do: the tool-call name's, else
/// the reasoning name's when no tool-call name is set; the reasoning name's again when the
/// tool-call name is the `plain` table's (`passthrough`, the older parser that left its channel
/// as written) and the reasoning name is another table's, which then reads without its calls
/// ([`no_calls`]), as the older pair read. A name is one a table answers to ([`answering_to`]):
/// its own, an older parser's that reads its format, or a launcher's for a family whose table
/// has no older parser; any other name selects none, [`TEMPLATE_PARSER`] among them, and so do
/// no names at all.
pub(crate) fn named_table(names: &Names) -> Option<Table> {
    let Some(tool) = names.tool.as_deref() else {
        return names.reasoning.as_deref().and_then(answering_to);
    };
    let table = answering_to(tool)?;
    if table == Table::Plain {
        if let Some(other) = names.reasoning.as_deref().and_then(answering_to) {
            return Some(other);
        }
    }
    Some(table)
}

/// Whether the table the names select ([`named_table`]) reads without its calls: the tool-call
/// name is the `plain` table's and the reasoning name is another table's, so the pair reads as
/// the older one did, the thought split out and every call left as content.
pub(crate) fn no_calls(names: &Names) -> bool {
    names.tool.as_deref().and_then(answering_to) == Some(Table::Plain)
        && names
            .reasoning
            .as_deref()
            .and_then(answering_to)
            .is_some_and(|table| table != Table::Plain)
}

/// What one source's parser names select: the response template's table ([`TEMPLATE_PARSER`] in
/// either name); a table ([`named_table`]), with or without its calls ([`no_calls`]); or nothing,
/// by no name at all, or by a reasoning name alone that no table answers to, since every table
/// reads the thought channel (a tool-call name no table answers to never reaches a request:
/// [`check_names`] refuses it).
enum Selection {
    Template,
    Table { table: Table, no_calls: bool },
    Nothing,
}

fn selection(names: &Names) -> Selection {
    if names_template(names) {
        return Selection::Template;
    }
    match named_table(names) {
        Some(table) => Selection::Table {
            table,
            no_calls: no_calls(names),
        },
        None => Selection::Nothing,
    }
}

/// Whether `names`, one source's, select nothing (no name at all, or a reasoning name alone that no
/// table answers to), so that [`decision`] passes them over for the next source: a worker labeled
/// so is read by the registry's table, as an unlabeled one is.
pub(crate) fn labels_select_nothing(names: &Names) -> bool {
    matches!(selection(names), Selection::Nothing)
}

/// What reads a model's responses ([`decision`]).
pub(super) enum Decision {
    /// The response template's table, compiled when the request comes ([`Templates`]).
    Template,
    /// A table, shared with every request and engine that reads with it: the registry's, the
    /// one the names select, or `plain` for a model nothing else reads.
    Table(Arc<Format>),
}

/// The reader of `model`'s responses, by the one precedence among the parser names set for the
/// model ([`ModelNames`]) and the registry: the names of the model's card (its workers' labels),
/// when they select anything; else the registry's table, for a model it knows
/// (`symphony::formats::for_model`); else the gateway's `--tool-call-parser` and
/// `--reasoning-parser`, for a model the registry does not know; else the `plain` table, which
/// reads the whole output as content. No id is matched by pattern: a quantized or renamed
/// checkpoint the registry does not list is read as content, calls and thought included, until
/// a label or a flag names its table, where the older parsers matched such an id by name;
/// the registration of its first worker says so once ([`nothing_names_a_table`]). So a
/// fleet-wide name reads the models the registry has no table for and never a family it has
/// one for, and a worker's labels read their model whatever the registry has. A table the labels
/// select reads a model the registry gives the same table with the model's own turn opener;
/// another model, or one the registry does not know, replays from the table's own. A table the
/// names select without its calls is read through the variant every such request shares
/// ([`asked`]). The registry is asked once per path.
pub(super) fn decision(names: &ModelNames<'_>, model: &str) -> Decision {
    match selection(&names.labels) {
        Selection::Template => Decision::Template,
        Selection::Table { table, no_calls } => Decision::Table(asked(
            match for_model_shared(model) {
                Some(format) if table_for(model) == Some(table) => format,
                _ => table.shared(),
            },
            no_calls,
            false,
        )),
        Selection::Nothing => match for_model_shared(model) {
            Some(format) => Decision::Table(format),
            // The registry does not know the model: the flags read it, a table they name from
            // its own opener; with no name, `plain`.
            None => match selection(names.flags) {
                Selection::Template => Decision::Template,
                Selection::Table { table, no_calls } => {
                    Decision::Table(asked(table.shared(), no_calls, false))
                }
                Selection::Nothing => Decision::Table(Table::Plain.shared()),
            },
        },
    }
}

/// The table that reads a model's responses ([`reading`]). Preparation reads it once per request
/// and keeps it (`ProcessingState::reading`) for the readers after it: the replay, the
/// constraint, the decoder's special tokens, the assignment.
#[derive(Clone)]
pub(crate) struct Reading {
    /// The table, shared with every request and engine that reads with it.
    pub format: Arc<Format>,
    /// The table is the response template's, compiled for the model: a constraint is compiled
    /// against it, and a worker may take its whole turn's grammar.
    pub compiled: bool,
}

/// What reads `model`'s responses: what [`decision`] gives, the response template's table
/// compiled for the model (an error when the model has none that compiles, which refuses the
/// request, since the names ask for what the model cannot give), or another table.
pub(crate) fn reading(
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
) -> Result<Reading, SymphonyError> {
    Ok(match decision(names, model) {
        Decision::Template => Reading {
            format: templates.format(model).map_err(SymphonyError::Template)?,
            compiled: true,
        },
        Decision::Table(format) => Reading {
            format,
            compiled: false,
        },
    })
}

/// The table that reads `model`'s responses ([`reading`]): the response template's, the
/// registry's, the one the parser names select, or `plain`, by the precedence [`decision`]
/// states. What a request asks of the channels takes a variant of the table
/// ([`assign_for`](super::reader::assign_for)). Preparation decides through [`reading`] once and
/// carries the decision; the `auto` tag decides with this predicate for the selected worker. The
/// table comes shared, built once for every request and engine that reads with it; a variant is a
/// copy.
pub(crate) fn reads(
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
) -> Result<Arc<Format>, SymphonyError> {
    reading(names, model, templates).map(|read| read.format)
}

/// The `table` label under which Symphony counts `model`'s outputs ([`telemetry`]): the table
/// [`decision`] gives for the names set for the model, or the response template's name for a
/// model read through its template. Asked at the model's first registration, so the table's
/// series exist at zero before its first output (`workflow::steps::local::create_worker`).
pub(crate) fn table_label_read(names: &ModelNames<'_>, model: &str) -> &'static str {
    match decision(names, model) {
        Decision::Template => telemetry::TEMPLATE,
        Decision::Table(format) => telemetry::table_label(format.name()),
    }
}

/// What the gateway's flags select for `model`, a model the registry knows, that the precedence of
/// [`decision`] sets aside for a worker labeling no parser: the registry's table, which reads the
/// model, and the flags' choice spelled for a log line (`the table 'mistral_v13'`, `the model's
/// response template`). `None` when the flags select nothing or the registry's own table, or when
/// the registry does not know the model. Registration warns with it once per model
/// (`workflow::steps::local::create_worker`).
pub(crate) fn set_aside_by_the_registry(
    flags: &Names,
    model: &str,
) -> Option<(Table, Cow<'static, str>)> {
    let table = table_for(model)?;
    let spelled = match selection(flags) {
        Selection::Nothing => return None,
        Selection::Table {
            table: named,
            no_calls: false,
        } if named == table => return None,
        Selection::Table {
            table: named,
            no_calls,
        } => Cow::Owned(format!(
            "the table '{}'{}",
            named.name(),
            if no_calls { " without its calls" } else { "" }
        )),
        Selection::Template => Cow::Borrowed("the model's response template"),
    };
    Some((table, spelled))
}

/// Whether nothing names a table for `model`, so that [`decision`] reads it with `plain`: its
/// worker's labels select none, the registry does not list the id, and the gateway's flags select
/// none. The older parsers matched such an id by name (a quantized or renamed checkpoint of a
/// family they read), so its calls and its thought now stay in the content until a label or a
/// flag names its table. Registration warns with it once per model
/// (`workflow::steps::local::create_worker`).
pub(crate) fn nothing_names_a_table(names: &ModelNames<'_>, model: &str) -> bool {
    matches!(selection(&names.labels), Selection::Nothing)
        && table_for(model).is_none()
        && matches!(selection(names.flags), Selection::Nothing)
}

/// Whether the parser names set for a model (the flags, or one worker's labels) are names the
/// gateway reads with: a table's own name, one of its aliases ([`answering_to`]: an older
/// parser's name whose format the table reads, a launcher's name for a family whose table has no
/// older parser), or [`TEMPLATE_PARSER`], which binds the other name to itself or to nothing,
/// since the response template's table reads both channels. Any other tool-call name is refused
/// with the tables an operator may name, and with the reason the older parser of that name is not
/// read with when it was one ([`UNANSWERED`]). An older reasoning name no table answers to
/// passes and selects nothing ([`UNANSWERED`] again: its thought is a family's the tool-call name
/// or the registry decides), so a deployment's flag pair keeps starting; startup and a worker's
/// registration say so ([`unanswered_reasoning`]). A reasoning name that is none of these is
/// refused. When both names select a table, the tool-call name's reads both channels
/// ([`named_table`]); [`other_table`] says when the reasoning name's differs. The error says what
/// is wrong with which name, for the startup or registration failure.
pub(crate) fn check_names(names: &Names) -> Result<(), SymphonyError> {
    checked_names(names).map_err(SymphonyError::Name)
}

/// [`check_names`], with what is wrong with which name as the text.
fn checked_names(names: &Names) -> Result<(), String> {
    let kinds = [
        (
            "tool-call",
            names.tool.as_deref(),
            names.reasoning.as_deref(),
        ),
        (
            "reasoning",
            names.reasoning.as_deref(),
            names.tool.as_deref(),
        ),
    ];
    for (kind, name, other) in kinds {
        let Some(name) = name else {
            continue;
        };
        if is_named(Some(name), TEMPLATE_PARSER) {
            if let Some(other) = other.filter(|other| !is_named(Some(other), TEMPLATE_PARSER)) {
                return Err(format!(
                    "{kind} parser '{name}' reads with the model's response template, so the \
                     other parser must name '{TEMPLATE_PARSER}' or nothing, not '{other}'"
                ));
            }
            continue;
        }
        if answering_to(name).is_none() && !(kind == "reasoning" && is_unanswered(name)) {
            return Err(unknown_name(kind, name));
        }
    }
    Ok(())
}

/// Whether `name` is an older parser's that no table answers to ([`UNANSWERED`]).
fn is_unanswered(name: &str) -> bool {
    let spelled = name.trim().to_ascii_lowercase();
    UNANSWERED.iter().any(|(older, _)| *older == spelled)
}

/// The reasoning name set for a model that is an older parser's no table answers to
/// ([`UNANSWERED`]), as given, with the reason: it passed [`check_names`] and selects nothing, so
/// startup and a worker's registration say what reads the model instead. `None` when the
/// reasoning name is a table's, an alias, the template's, or not set.
pub(crate) fn unanswered_reasoning(names: &Names) -> Option<(&str, &'static str)> {
    let name = names.reasoning.as_deref()?;
    let spelled = name.trim().to_ascii_lowercase();
    let (_, reason) = UNANSWERED.iter().find(|(older, _)| *older == spelled)?;
    Some((name, *reason))
}

/// The refusal of a `kind` parser name no table answers to: the reason the older parser of that
/// name is not read with ([`UNANSWERED`]) when it was one, then the names an operator may give.
fn unknown_name(kind: &str, name: &str) -> String {
    let spelled = name.trim().to_ascii_lowercase();
    let older = UNANSWERED
        .iter()
        .find(|(older, _)| *older == spelled)
        .map_or_else(String::new, |(_, reason)| format!(" ({reason})"));
    let tables: Vec<&str> = Table::ALL.iter().map(|table| table.name()).collect();
    let aliases: Vec<&str> = ALIASES.iter().map(|(alias, _)| *alias).collect();
    format!(
        "unknown {kind} parser '{name}'{older}: a Symphony table's name ({}), one of the names \
         a table answers to ({}), or '{TEMPLATE_PARSER}' for the model's response template",
        tables.join(", "),
        aliases.join(", ")
    )
}

/// The table the reasoning name selects when the tool-call name selects another: the tool-call
/// name's table reads both channels ([`named_table`]), and the other name reads nothing, which
/// startup says once. `None` when the names agree, when either selects no table, or when the
/// tool-call name is the `plain` table's, since the reasoning name's table reads then
/// ([`no_calls`]).
pub(crate) fn other_table(names: &Names) -> Option<(Table, Table)> {
    let reads = names.tool.as_deref().and_then(answering_to)?;
    let other = names.reasoning.as_deref().and_then(answering_to)?;
    (reads != Table::Plain && other != reads).then_some((reads, other))
}

/// The word startup says for an older name a table answers to with a note ([`NOTED`]): the name
/// as given, the table that reads it, and the note; the tool-call name first, as it decides.
/// `None` when neither name is noted.
pub(crate) fn noted_name(names: &Names) -> Option<(&str, Table, &'static str)> {
    [names.tool.as_deref(), names.reasoning.as_deref()]
        .into_iter()
        .flatten()
        .find_map(|name| {
            let spelled = name.trim().to_ascii_lowercase();
            let (_, note) = NOTED.iter().find(|(noted, _)| *noted == spelled)?;
            Some((name, answering_to(name)?, *note))
        })
}

/// Whether the rendered `prompt` leaves the model inside its thought as the table reads it: the
/// table's replay of the prompt ([`Format::reasoning_open`]), which starts a turn the prompt leaves
/// unopened inside the thought where the template writes no opener (DeepSeek's after a system
/// message a latest reminder follows, or the checkpoint's own encoder file's after any system
/// message). The grammar built for the request takes it OR'd with the request's
/// thinking switch, so the tag and the engine read the prompt the same way: a JSON answer the
/// switch would place at the first token follows the thought's close instead of streaming as
/// reasoning. `None` when the names refuse the model (the response template it has none of),
/// leaving the switch's answer as it was. The table's replay answers alone for the family whose
/// older rule set the switch aside ([`PromptThought::alone`]).
pub(crate) fn reasoning_open(
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    prompt: &str,
) -> Option<PromptThought> {
    reading(names, model, templates)
        .ok()
        .map(|read| reasoning_open_for(&read, prompt))
}

/// [`reasoning_open`] over the table decision already read for the request (`read`): the
/// thought's state in `prompt` as the table replays it.
pub(crate) fn reasoning_open_for(read: &Reading, prompt: &str) -> PromptThought {
    // The registry's Gemma 4 table carries the rule; a Gemma 4 model read through its own
    // response template (a table named for the template) is left to the switch beside the
    // replay, as every other table is: no Gemma 4 tokenizer ships a response template today.
    PromptThought {
        open: read.format.reasoning_open(prompt),
        closed: read.format.thought_closed(prompt),
        alone: read.format.name() == Table::Gemma4.name(),
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
    /// about the prompt. The rule the family's older reasoning parser applied in place of the
    /// switch, kept here.
    pub alone: bool,
}

/// A table's replay of a rendered prompt ([`reasoning_open`]'s answer): the thought's state where
/// a table reads the response, `None` where the names refused the model. Preparation keeps its
/// replay for request building (`ProcessingState::prompt_thought`), which decides over the same
/// prompt.
pub(crate) type PromptReplay = Option<PromptThought>;

impl PromptThought {
    /// Whether the prompt leaves the model inside its thought, as request building decides it:
    /// the replay's answer, or the request's thinking `switch` where the replay does not decide
    /// alone.
    pub fn or_switch(self, switch: impl FnOnce() -> bool) -> bool {
        self.open || (!self.alone && switch())
    }
}

/// [`keeps_special_tokens_for`] over the table decision read here ([`reading`]; `None` when it
/// errs), as the tests ask it.
#[cfg(test)]
pub(crate) fn keeps_special_tokens(
    names: &ModelNames<'_>,
    model: &str,
    templates: &Templates,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Option<bool> {
    reading(names, model, templates)
        .ok()
        .map(|read| keeps_special_tokens_for(&read, model, templates, tokenizer))
}

/// Whether the table that reads `model`'s responses (`read`, the table decision, which preparation
/// reads once for the request, `regular::stages::reading::prepare`) needs the vocabulary's special
/// tokens kept in the decoded output: one of its markers is, or holds, a token the stop decoder
/// would drop with `skip_special_tokens` (Muse Spark's `<|message|>` and `<|eom|>`, Kimi K3's
/// `<|open|>` and `<|sep|>`). A table says nothing of itself, so the gateway asks the tokenizer: a
/// marker whose tokens decode to its text with the special tokens kept and to something else with
/// them dropped is such a marker. `false` when every marker survives the decoder either way
/// (Qwen3's `<think>` and `<tool_call>` are added tokens the decoder keeps). Asking the tokenizer
/// encodes and decodes every marker, and the answer depends on the tokenizer and the markers
/// alone, so it is decided once per model and kept with them ([`Templates::markers_hidden`]). The
/// tests ask through `keeps_special_tokens`, which reads the decision first.
pub(crate) fn keeps_special_tokens_for(
    read: &Reading,
    model: &str,
    templates: &Templates,
    tokenizer: &Arc<dyn Tokenizer>,
) -> bool {
    templates.markers_hidden(model, &read.format, tokenizer, || {
        markers_hidden_by_the_decoder(&read.format, tokenizer.as_ref())
    })
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
        // An older parser's name reads with the table of the format it read.
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
        assert_eq!(named_table(&pair(None, Some("qwen3_thinking"))), None);
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
        // The tool-call name decides.
        assert_eq!(
            named_table(&pair(Some("kimi_k3"), Some("qwen3"))),
            Some(Table::KimiK3)
        );
        // The older `pythonic` parser's name serves more than one family, so no table answers
        // to it (the startup check refuses it with the reason); the Llama 4 table answers to its
        // exact alias.
        assert_eq!(named_table(&pair(Some("pythonic"), None)), None);
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
        assert_eq!(
            named_table(&pair(None, Some("deepseek_r1"))),
            Some(Table::DeepSeekV3)
        );
        assert!(Table::DeepSeekV3
            .format()
            .reasoning_open("<｜Assistant｜><think>\n"));
        assert_eq!(
            named_table(&pair(Some("deepseek31"), Some("deepseek_v31"))),
            Some(Table::DeepSeekV3_1)
        );
        // `passthrough`, the older parser that left its channel as written, names the `plain`
        // table: alone, doubled, or as the reasoning name; a tool-call name beside it decides.
        // As the tool-call name beside a reasoning name a table answers to, that table reads,
        // without its calls: the older pair split the thought out and left every call as content.
        assert_eq!(
            named_table(&pair(Some("passthrough"), Some(" Passthrough "))),
            Some(Table::Plain)
        );
        assert!(!no_calls(&pair(Some("passthrough"), Some(" Passthrough "))));
        assert_eq!(
            named_table(&pair(Some("passthrough"), None)),
            Some(Table::Plain)
        );
        assert!(!no_calls(&pair(Some("passthrough"), None)));
        assert_eq!(
            named_table(&pair(None, Some("passthrough"))),
            Some(Table::Plain)
        );
        assert_eq!(
            named_table(&pair(Some("passthrough"), Some("qwen3"))),
            Some(Table::Qwen3)
        );
        assert!(no_calls(&pair(Some("passthrough"), Some("qwen3"))));
        assert_eq!(
            named_table(&pair(Some("qwen"), Some("passthrough"))),
            Some(Table::Qwen3)
        );
        assert!(!no_calls(&pair(Some("qwen"), Some("passthrough"))));
        assert!(!no_calls(&pair(None, Some("qwen3"))));
        // A tool-call name no table answers to selects none, whatever the reasoning name.
        assert_eq!(named_table(&pair(Some("pythonic"), Some("qwen3"))), None);
    }

    /// Every name a table answers to besides its own ([`ALIASES`]) passes the startup check in
    /// both flags, in either alone, and beside the table's own name, as every table's own name
    /// does in any case; a name no table answers to is refused, with the reason where an older
    /// parser carried it ([`UNANSWERED`]).
    #[test]
    fn every_name_a_table_answers_to_passes_the_check_and_an_unanswered_name_is_refused() {
        for (alias, table) in ALIASES {
            let doubled = names(Some(alias), Some(alias));
            assert_eq!(named_table(&doubled.labels), Some(table), "{alias}");
            assert_eq!(check(&doubled), Ok(()), "{alias}");
            assert_eq!(check(&names(Some(alias), None)), Ok(()), "{alias}");
            assert_eq!(check(&names(None, Some(alias))), Ok(()), "{alias}");
            let own = table.name();
            assert_eq!(check(&names(Some(alias), Some(own))), Ok(()), "{alias}");
            assert_eq!(check(&names(Some(own), Some(alias))), Ok(()), "{alias}");
        }
        for table in Table::ALL {
            let own = table.name();
            assert_eq!(check(&names(Some(own), Some(own))), Ok(()), "{own}");
            assert_eq!(
                check(&names(Some(&own.to_ascii_uppercase()), None)),
                Ok(()),
                "{own}"
            );
        }
        for (name, reason) in UNANSWERED {
            let why = check(&names(Some(name), None)).expect_err(name);
            assert!(
                why.starts_with(&format!("unknown tool-call parser '{name}' ({reason})")),
                "{why}"
            );
            // An older reasoning name passes, alone or beside a tool-call name, and selects
            // nothing: startup says why.
            let alone = names(None, Some(name));
            assert_eq!(check(&alone), Ok(()), "{name}");
            assert_eq!(named_table(&alone.labels), None, "{name}");
            assert_eq!(
                unanswered_reasoning(&alone.labels),
                Some((name, reason)),
                "{name}"
            );
            assert_eq!(check(&names(Some("qwen"), Some(name))), Ok(()), "{name}");
        }
        // The pairings a launch script carries keep starting; the tool-call name reads.
        let glm = names(Some("glm47_moe"), Some("glm45"));
        assert_eq!(check(&glm), Ok(()));
        assert_eq!(named_table(&glm.labels), Some(Table::Glm));
        assert_eq!(
            unanswered_reasoning(&glm.labels),
            Some(("glm45", UNANSWERED[1].1))
        );
        assert_eq!(
            unanswered_reasoning(&names(None, Some("qwen3")).labels),
            None
        );
        assert_eq!(
            unanswered_reasoning(&names(Some("qwen"), None).labels),
            None
        );
        // A reasoning name that is nobody's is still refused.
        let why = check(&names(None, Some("nonsense"))).expect_err("nonsense");
        assert!(
            why.starts_with("unknown reasoning parser 'nonsense'"),
            "{why}"
        );
    }

    #[test]
    fn an_unknown_name_is_refused_with_the_names_an_operator_may_give() {
        assert_eq!(check(&ModelNames::default()), Ok(()));
        assert_eq!(check(&names(Some("qwen"), Some("qwen3"))), Ok(()));
        // Two names that select different tables are two valid names: the tool-call name's
        // table reads both channels, and startup says so once ([`other_table`]).
        assert_eq!(check(&names(Some("ling"), Some("qwen3"))), Ok(()));
        assert_eq!(
            other_table(&pair(Some("ling"), Some("qwen3"))),
            Some((Table::Ling, Table::Qwen3))
        );
        assert_eq!(other_table(&pair(Some("qwen"), Some("qwen3"))), None);
        assert_eq!(other_table(&pair(None, Some("qwen3"))), None);
        assert_eq!(other_table(&pair(Some("ling"), Some("nonsense"))), None);
        // The reasoning name's table reads beside `passthrough`: nothing to say.
        assert_eq!(other_table(&pair(Some("passthrough"), Some("qwen3"))), None);
        // An unknown name is refused with the tables, the other names a table answers to, and
        // the template's name.
        let why = check(&names(Some("nonsense"), None)).expect_err("unknown");
        assert!(
            why.starts_with("unknown tool-call parser 'nonsense': a Symphony table's name ("),
            "{why}"
        );
        assert!(
            why.contains("qwen3")
                && why.contains("), one of the names a table answers to (")
                && why.contains("qwen_xml")
                && why.ends_with("), or 'hf' for the model's response template"),
            "{why}"
        );
        let why = check(&names(None, Some("nonsense"))).expect_err("unknown");
        assert!(
            why.starts_with("unknown reasoning parser 'nonsense': "),
            "{why}"
        );
        // The names are one list for both flags: an older tool-call parser's name a table answers
        // to passes as the reasoning name too; a name reads in any case, trimmed.
        assert_eq!(check(&names(None, Some("qwen_xml"))), Ok(()));
        assert_eq!(check(&names(Some("QWEN_XML"), None)), Ok(()));
        assert_eq!(check(&names(None, Some(" hy_v4 "))), Ok(()));
        // A launcher's name for a family whose table has no older parser (`muse`) passes in
        // either flag, with the table's own name, or beside another table's name.
        assert_eq!(check(&names(Some("muse"), Some("muse"))), Ok(()));
        assert_eq!(check(&names(Some("MUSE"), Some("muse_spark"))), Ok(()));
        assert_eq!(check(&names(Some("muse_spark"), Some("muse"))), Ok(()));
        assert_eq!(check(&names(Some("muse"), Some("qwen3"))), Ok(()));
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
        // Through the names: Qwen3's table over this vocabulary needs nothing, and neither does
        // the `plain` table, which has no markers.
        let shared: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
        assert_eq!(
            keeps_special_tokens(
                &names(Some("qwen3"), None),
                "Qwen/Qwen3-8B",
                &templates(),
                &shared,
            ),
            Some(false)
        );
        assert_eq!(
            keeps_special_tokens(
                &names(Some("passthrough"), None),
                "Qwen/Qwen3-8B",
                &templates(),
                &shared,
            ),
            Some(false)
        );
    }

    #[test]
    fn a_table_whose_markers_the_decoder_hides_answers_true_and_its_replay_opens_the_thought() {
        // The Kimi K3 table's markers are special tokens of this vocabulary (`<bos>` stands for
        // them here): the table that reads the response says so itself.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(marker_tokenizer());
        let hidden = table_with_markers("<bos>", "<eos>");
        assert!(markers_hidden_by_the_decoder(&hidden, tokenizer.as_ref()));
        // Where the table reads, its replay of the prompt answers: Kimi K3's prompt opens the
        // thought, a prompt that writes no such opener does not; the request's thinking switch
        // is heard beside the replay.
        let read = |table: &str, prompt: &str| {
            reasoning_open(&names(Some(table), Some(table)), "m", &templates(), prompt)
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
            keeps_special_tokens(names, "Qwen/Qwen3-8B", &templates, tokenizer)
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
            keeps_special_tokens(&qwen3, "Qwen/Qwen3-4B", &templates, &other),
            Some(false)
        );
        assert_eq!(replaced.encodes(), 2 * asked + mistral);
    }

    #[test]
    fn hf_binds_the_other_name_to_itself_or_to_nothing() {
        assert_eq!(check(&names(Some("hf"), None)), Ok(()));
        assert_eq!(check(&names(None, Some("hf"))), Ok(()));
        assert_eq!(check(&names(Some("HF"), Some(" hf "))), Ok(()));
        let why = check(&names(Some("hf"), Some("qwen3"))).expect_err("bound");
        assert_eq!(
            why,
            "tool-call parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen3'"
        );
        let why = check(&names(Some("qwen"), Some("hf"))).expect_err("bound");
        assert_eq!(
            why,
            "reasoning parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen'"
        );
    }

    /// The precedence among the parser names set for a model, which [`super::reads`] states
    /// once: the labels of the model's workers read over the registry; the registry's table
    /// reads over the gateway's flags, for a model it knows; the flags read a model it does not
    /// know; and `plain` reads the rest. Each rule in turn.
    #[test]
    fn the_labels_read_over_the_registry_which_reads_over_the_flags_for_the_models_it_knows() {
        let known = "Qwen/Qwen3-8B";
        let unknown = "acme/ministral-3-14b-instruct-w4a16";
        assert_eq!(table_for(known), Some(Table::Qwen3));
        assert_eq!(table_for(unknown), None);
        let read = |names: &ModelNames<'_>, model: &str| {
            reads(names, model, &templates()).map(|format| format.name().to_string())
        };
        let table = |name: &str| Ok(name.to_string());
        let v13 = pair(Some("mistral_v13"), None);
        let no_flags = Names::default();
        fn labeled<'a>(labels: Names, flags: &'a Names) -> ModelNames<'a> {
            ModelNames { labels, flags }
        }

        // The labels over the registry: a table they name reads a model the registry reads with
        // another; `passthrough` alone names `plain` for the one model; a reasoning name alone
        // that a table answers to selects it over a tool-call name the flags set.
        assert_eq!(
            read(&labeled(v13.clone(), &no_flags), known),
            table("mistral_v13")
        );
        assert_eq!(
            read(&labeled(pair(None, Some("passthrough")), &v13), known),
            table("plain")
        );
        assert_eq!(
            read(&labeled(pair(None, Some("qwen3")), &v13), unknown),
            table("qwen3")
        );

        // The registry over the flags, for a model it knows, whatever they name: a table, the
        // response template (no refusal for a model without one), `plain`.
        for flags in [
            v13.clone(),
            pair(Some(TEMPLATE_PARSER), Some(TEMPLATE_PARSER)),
            pair(Some("passthrough"), None),
        ] {
            assert_eq!(
                read(&ModelNames::flagged(&flags), known),
                table("qwen3"),
                "{flags:?}"
            );
        }

        // The flags for a model the registry does not know: a table's name reads it; `hf` asks
        // its response template, refused when it has none; a reasoning name alone that a table
        // answers to selects it.
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
            read(&ModelNames::flagged(&pair(None, Some("qwen3"))), unknown),
            table("qwen3")
        );

        // Labels that select nothing leave the decision to the registry, then to the flags.
        assert!(labels_select_nothing(&Names::default()));
        assert!(!labels_select_nothing(&pair(None, Some("passthrough"))));
        assert_eq!(
            read(&labeled(Names::default(), &v13), known),
            table("qwen3")
        );
        assert_eq!(
            read(&labeled(Names::default(), &v13), unknown),
            table("mistral_v13")
        );

        // Nothing named: the registry's table for a model it knows, `plain` for one it does not.
        assert_eq!(read(&ModelNames::default(), known), table("qwen3"));
        assert_eq!(read(&ModelNames::default(), unknown), table("plain"));
    }

    /// `deepseek_r1`, the R1 line's older reasoning name, reads with DeepSeek V3's table and
    /// passes the startup check; startup says so, for the name as given, in either flag, the
    /// tool-call name first; a name no table notes says nothing.
    #[test]
    fn deepseek_r1_reads_with_the_v3_table_and_startup_says_so() {
        let (_, note) = NOTED[0];
        let r1 = pair(None, Some(" DeepSeek_R1 "));
        assert_eq!(named_table(&r1), Some(Table::DeepSeekV3));
        assert!(check_names(&r1).is_ok());
        assert_eq!(
            noted_name(&r1),
            Some((" DeepSeek_R1 ", Table::DeepSeekV3, note))
        );
        assert_eq!(
            noted_name(&pair(Some("deepseek"), Some("deepseek_r1"))),
            Some(("deepseek_r1", Table::DeepSeekV3, note))
        );
        assert_eq!(noted_name(&pair(Some("deepseek"), None)), None);
        assert_eq!(noted_name(&pair(None, Some("qwen3"))), None);
        assert_eq!(noted_name(&Names::default()), None);
    }

    /// `passthrough` as the tool-call name beside a reasoning name a table answers to reads with
    /// that table without its calls, from the labels or the flags, over the registry's format for
    /// a model it knows: the older pair's reading, the thought split out and every call content.
    /// The one kept variant serves every such request.
    #[test]
    fn passthrough_beside_a_reasoning_name_reads_its_table_without_calls() {
        let split = pair(Some("passthrough"), Some("qwen3"));
        let format = |names: &ModelNames<'_>, model: &str| match decision(names, model) {
            Decision::Table(format) => format,
            Decision::Template => panic!("a table"),
        };
        let unknown = "acme/ministral-3-14b-instruct-w4a16";
        assert_eq!(
            *format(&ModelNames::flagged(&split), unknown),
            Arc::unwrap_or_clone(Table::Qwen3.shared()).without_calls()
        );
        let known = "Qwen/Qwen3-8B";
        let no_flags = Names::default();
        let labeled = ModelNames {
            labels: split.clone(),
            flags: &no_flags,
        };
        assert_eq!(
            *format(&labeled, known),
            Arc::unwrap_or_clone(for_model_shared(known).expect("a known model")).without_calls()
        );
        assert!(Arc::ptr_eq(
            &format(&ModelNames::flagged(&split), unknown),
            &format(&ModelNames::flagged(&split), unknown)
        ));
    }

    /// The `plain` table reads what nothing else names: a model the registry does not know under
    /// no names, and any model whose labels say `passthrough`; the one shared table, counted
    /// under its name like the others, not a fallback.
    #[test]
    fn plain_reads_what_nothing_else_names() {
        let known = "Qwen/Qwen3-8B";
        let unknown = "acme/ministral-3-14b-instruct-w4a16";
        let table = |names: &ModelNames<'_>, model: &str| match decision(names, model) {
            Decision::Table(format) => format.name().to_string(),
            Decision::Template => telemetry::TEMPLATE.to_string(),
        };
        assert_eq!(table(&ModelNames::default(), unknown), "plain");
        assert_eq!(table(&ModelNames::default(), known), "qwen3");
        for model in [known, unknown] {
            assert_eq!(table(&names(Some("passthrough"), None), model), "plain");
            assert_eq!(table(&names(None, Some("passthrough")), model), "plain");
        }
        let passthrough = pair(None, Some("passthrough"));
        assert_eq!(table(&ModelNames::flagged(&passthrough), unknown), "plain");
        assert_eq!(table(&ModelNames::flagged(&passthrough), known), "qwen3");
        // The one `plain` table, shared by every request and model that reads with it.
        let by_nothing = reads(&ModelNames::default(), unknown, &templates()).expect("read");
        let by_name = reads(&names(Some("passthrough"), None), known, &templates()).expect("read");
        assert!(Arc::ptr_eq(&by_nothing, &by_name));
        assert_eq!(table_label_read(&ModelNames::default(), unknown), "plain");
    }

    /// Nothing names a table for a model exactly where [`decision`] reads it with `plain`
    /// unasked: the registry does not list the id (a quantized checkpoint, which the older
    /// parsers matched by name) and neither the labels nor the flags select a table, an older
    /// reasoning name no table answers to selecting none. A label or a flag that names a table,
    /// `passthrough` (plain by choice) or the response template names one, and so does the
    /// registry for a model it lists.
    #[test]
    fn nothing_names_a_table_for_an_unlisted_model_without_a_label_or_a_flag() {
        let listed = "Qwen/Qwen3-8B";
        let unlisted = "acme/ministral-3-14b-instruct-w4a16";
        let table = |names: &ModelNames<'_>, model: &str| match decision(names, model) {
            Decision::Table(format) => format.name().to_string(),
            Decision::Template => telemetry::TEMPLATE.to_string(),
        };
        assert!(nothing_names_a_table(&ModelNames::default(), unlisted));
        assert_eq!(table(&ModelNames::default(), unlisted), "plain");
        // An older reasoning name no table answers to selects nothing, as a label and as a flag.
        assert!(nothing_names_a_table(
            &names(None, Some("qwen3_thinking")),
            unlisted
        ));
        let idle = pair(None, Some("qwen3_thinking"));
        assert!(nothing_names_a_table(&ModelNames::flagged(&idle), unlisted));
        // A name selects a table: a label, a flag, plain by choice, the response template.
        assert!(!nothing_names_a_table(&names(Some("qwen"), None), unlisted));
        let qwen = pair(Some("qwen"), None);
        assert!(!nothing_names_a_table(
            &ModelNames::flagged(&qwen),
            unlisted
        ));
        assert_eq!(table(&ModelNames::flagged(&qwen), unlisted), "qwen3");
        let passthrough = pair(Some("passthrough"), None);
        assert!(!nothing_names_a_table(
            &ModelNames::flagged(&passthrough),
            unlisted
        ));
        assert!(!nothing_names_a_table(
            &names(Some(TEMPLATE_PARSER), None),
            unlisted
        ));
        // The registry lists the model: its table reads it under no names.
        assert!(!nothing_names_a_table(&ModelNames::default(), listed));
        assert_eq!(table(&ModelNames::default(), listed), "qwen3");
    }

    /// What the gateway's flags select for a model the registry knows, set aside for a worker
    /// that labels no parser, is spelled for the registration warning beside the registry's
    /// table: a table or the response template; nothing when the flags select nothing or the
    /// registry's own table, or for a model the registry does not know.
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
            spelled(pair(None, Some("passthrough")), known),
            qwen3("the table 'plain'")
        );
        assert_eq!(spelled(pair(Some("qwen"), Some("qwen3")), known), None);
        // The flags' variant without calls is set aside by the registry's table with them.
        assert_eq!(
            spelled(pair(Some("passthrough"), Some("qwen3")), known),
            qwen3("the table 'qwen3' without its calls")
        );
        assert_eq!(spelled(pair(None, Some("qwen3_thinking")), known), None);
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
        assert!(matches!(
            check_names(&pair(Some("nonsense"), None)),
            Err(SymphonyError::Name(text)) if text.starts_with("unknown tool-call parser 'nonsense'")
        ));
    }

    /// The label a model's outputs count under is known from its names at registration: the
    /// registry's table for a model it knows, the labels' table over it, the flags' table for a
    /// model the registry does not know, the template's name, and `plain` for a model nothing
    /// else names.
    #[test]
    fn the_table_label_a_model_counts_under_is_known_from_its_names() {
        let no_flags = Names::default();
        let flags = pair(Some("mistral_v13"), None);
        let read = |labels: Names, flags: &Names, model: &str| {
            table_label_read(&ModelNames { labels, flags }, model)
        };
        assert_eq!(read(Names::default(), &no_flags, "Qwen/Qwen3-8B"), "qwen3");
        assert_eq!(
            read(pair(Some("kimi_k2"), None), &no_flags, "Qwen/Qwen3-8B"),
            "kimi_k2"
        );
        assert_eq!(
            read(Names::default(), &flags, "acme/unknown-model"),
            "mistral_v13"
        );
        assert_eq!(
            read(
                pair(Some(TEMPLATE_PARSER), None),
                &no_flags,
                "Qwen/Qwen3-8B"
            ),
            "response_template"
        );
        assert_eq!(
            read(Names::default(), &no_flags, "acme/unknown-model"),
            "plain"
        );
        assert_eq!(
            read(pair(Some("passthrough"), None), &no_flags, "Qwen/Qwen3-8B"),
            "plain"
        );
    }
}
