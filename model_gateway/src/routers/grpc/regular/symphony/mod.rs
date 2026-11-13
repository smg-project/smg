//! Symphony reading a chat response or a message: one [`Engine`](::symphony::Engine) per choice
//! parses the model's output into typed events.
//!
//! One precedence decides what reads a model ([`reads`](names::reads)), among the parser names set
//! for it (`ParserResolver::names`: the `tool_parser` and `reasoning_parser` of the model's card,
//! its workers' labels, apart from the gateway's `--tool-call-parser` and `--reasoning-parser`
//! flags) and the registry: the labels, when they select anything; else the registry's table for a
//! model it knows (`symphony::formats::for_model`); else the flags, for a model the registry does
//! not know, so that a fleet-wide name never reads a family the registry has a table for; else the
//! `plain` table, which reads the whole output as content. Names select a table when it answers to
//! them (`utils::answering_to`: its own name, an older parser's that reads its format, `qwen` for
//! `qwen3`, `qwen_xml` for `qwen3_tagged`, `passthrough` for `plain`, or a launcher's name for a
//! family whose table has no older parser, `muse` for `muse_spark`), the tool-call name deciding,
//! and the table then reads every request for the model; [`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)
//! selects the table the model's tokenizer's `response_template` compiles into ([`template`]),
//! which refuses the request when the model has none; no name at all, or a reasoning name alone
//! that no table answers to, selects nothing, and the next source decides. The names are checked
//! at startup and at a worker's registration ([`check_names`]): an older tool-call parser's name
//! no table answers to is refused with the reason (`utils::UNANSWERED`) and the tables an operator
//! may name; an older reasoning parser's passes, selects nothing, and both say so with its reason;
//! the check also says, once per model, when the registry's table sets the flags' choice
//! aside ([`set_aside_by_the_registry`]). What a request asks of the channels takes a variant of
//! the table rather than another reader: where no tool calls are read (no tools declared, for a
//! provider that reads calls only with them), the table without its calls; with
//! `separate_reasoning: false`, the table without its thought ([`assign_for`]). A `tool_choice` of
//! `none` over declared tools keeps the table's calls: the prompt declares the tools as under
//! `auto`, as the engine's own server renders it, and the reader drops every call the model writes
//! anyway, counted per table (`smg_symphony_calls_forbidden_total`), so none is delivered as a call
//! or as content. An output the gateway constrained to a JSON schema (a forced or required call the
//! table has no tag for, a JSON answer the table gives no tag) has no markers for the model's table
//! to read: the format comes from the request then ([`assign_for`]), `forced` for the named
//! function's arguments, `json` for a required call's list, `plain` for an answer.
//!
//! Request building assigns Symphony to a request ([`assign_for`]) and stores
//! the [`Assignment`] in the response spec: the table, and the part of the rendered prompt the
//! engine replays to find the state the generation starts in. Preparation asks the table for the
//! structural tag of a forced or required call, or of a JSON `response_format`
//! ([`constraint_for`]); where the table has none, the gateway's JSON schema constrains the
//! output. Both decide with one predicate, so the output a table's tag constrains is read by that
//! table and never by another, which expects another syntax. Request building decides what constrains the output on the worker
//! selected for the request ([`worker_constraint`]): a worker whose engine takes a grammar on every
//! turn from its first token gets the whole turn's grammar where the response template's table
//! spells one ([`turn_constraint`]); a table's refusal in preparation waits for that decision. The
//! streaming and non-streaming drivers run one [`Reader`] per choice
//! over the stop decoder's output and render its events with `symphony::adapt::chat` or
//! `symphony::adapt::messages`, so the client sees the deltas and the message it saw before.

mod constraint;
mod names;
mod reader;
pub(crate) mod telemetry;
pub(crate) mod template;
#[cfg(test)]
pub(crate) mod testing;

pub(crate) use symphony::event::reasoning_text_tokens;

use crate::routers::grpc::utils::StreamFailure;

/// Why the seam refused a request or failed to read an output, as five classes a caller can match
/// on. Every caller but one takes the text, which is the one the gateway answered or logged before
/// the classes were named: the non-streaming Messages path answers [`Self::InputRefused`] under
/// its own code at warn level, since that one is the output's doing, and counts it under the
/// table by its reason (`smg_symphony_refused_total`, [`telemetry::record_refused`]). A status by
/// class is a change of its own.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SymphonyError {
    /// A parser name the flags or a worker's labels set is none the gateway reads with, or binds
    /// the other name wrongly (a startup or registration failure).
    #[error("{0}")]
    Name(String),
    /// The names ask for the model's response template and the model has none that compiles.
    #[error("{0}")]
    Template(String),
    /// The table cannot constrain what the request asks (a 400).
    #[error("{0}")]
    Constraint(String),
    /// The output could not be read: the stop decoder or the engine refused it.
    #[error("{0}")]
    Read(String),
    /// The output was read, and a `tool_use` block's `input` cannot carry the arguments the model
    /// wrote (JSON that no input value holds; the Messages fold names the call): the output's own
    /// doing, not the gateway's, so it is logged as a warning under its own code.
    #[error("{0}")]
    InputRefused(String),
}

impl From<SymphonyError> for String {
    fn from(error: SymphonyError) -> Self {
        error.to_string()
    }
}

impl From<SymphonyError> for StreamFailure {
    fn from(error: SymphonyError) -> Self {
        Self::Internal(error.to_string())
    }
}

#[cfg(test)]
pub(crate) use self::{constraint::constraint, reader::assign};
pub(crate) use self::{
    constraint::{
        constraint_for, grammar_start_of, turn_constraint, vocabulary_of, worker_constraint, Asks,
        AutoRequest, AutoTag, Prepared, PreparedConstraint,
    },
    names::{
        check_names, keeps_special_tokens_for, labels_select_nothing, noted_name,
        nothing_names_a_table, other_table, reading, reads, reasoning_open, reasoning_open_for,
        set_aside_by_the_registry, table_label_read, unanswered_reasoning, PromptReplay, Reading,
    },
    reader::{assign_for, engine_finish, finished, Assignment, Delta, Reader},
};
