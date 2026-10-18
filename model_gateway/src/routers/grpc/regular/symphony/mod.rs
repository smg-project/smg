//! Symphony reading a chat response or a message: the opt-in path that parses a model's output
//! with one [`Engine`] per choice in place of the tool and reasoning parser crates.
//!
//! The switch is the gateway's `--parser` flag (`RouterConfig::parser`, `legacy` by default),
//! which a request's `X-SMG-Parser` header overrides for that one request ([`choice`]). With
//! `symphony` on, one precedence decides what reads a model ([`reads`]), among the parser names
//! set for it (`ParserResolver::names`: the `tool_parser` and `reasoning_parser` of the model's
//! card, its workers' labels, apart from the gateway's `--tool-call-parser` and
//! `--reasoning-parser` flags) and the registry: the labels, when they select anything; else the
//! registry's table for a model it knows (`symphony::formats::for_model`); else the flags, for a
//! model the registry does not know, so that a fleet-wide name never reads a family the registry
//! has a table for; else the legacy parsers with their own detection. Names select a table when
//! it answers to them (`utils::answering_to`: its own name, a legacy parser's
//! that reads its format, `qwen` for `qwen3`, `qwen_xml` for `qwen3_tagged`, or a launcher's name
//! for a family whose table has no older parser, `muse` for `muse_spark`), the tool-call name
//! deciding, and the table then reads every request for the model; [`TEMPLATE_PARSER`] selects
//! the table the model's tokenizer's `response_template` compiles into ([`template`]), which
//! refuses the request when the model has none; any other legacy tool-call parser's name selects
//! the legacy parsers, which is the labels' way back for one model when its table misbehaves,
//! `passthrough` among them (the legacy parsers then take the labels' names, and from the flags
//! only names they read: a table's name or [`TEMPLATE_PARSER`] there is nothing to them,
//! `ParserResolver::new`), the rule being that every legacy name whose format a table reads is
//! the table's alias, so that a deployment's existing flags keep a working parser, and
//! `utils::UNANSWERED` lists each name no table answers to with the reason (the test over both
//! legacy registries below holds every name to one of the two); no name at all, or a reasoning
//! name alone that no table answers to, other than `passthrough`, selects nothing. The names are
//! checked at startup and at a worker's registration ([`check_names`]), which also says, once per
//! model, when the registry's table sets the flags' choice aside ([`set_aside_by_the_registry`]).
//! Every request read with `legacy` keeps the older parsers. The response names
//! the parser that read it in `x-smg-parser`, legacy when the choice was Symphony but no table
//! took the request, so a client comparing the two can tell which one read each answer. What a
//! request asks of the channels takes a variant of the table rather than another reader: where the
//! legacy path would run no tool parser (`tool_choice: none`; no tools declared, for a provider
//! that reads calls only with them), the table without its calls; with
//! `separate_reasoning: false`, the table without its thought ([`assign`]).
//!
//! Request building assigns Symphony to a request ([`assign`]) and stores
//! the [`Assignment`] in the response spec: the table, and the part of the rendered prompt the
//! engine replays to find the state the generation starts in. Preparation asks the table for the
//! structural tag of a forced or required call, or of a JSON `response_format` ([`constraint`])
//! before the older registry. Both decide with one predicate, so the
//! output a table's tag constrains is read by that table and never by the older parsers, which
//! expect another syntax. Request building decides what constrains the output on the worker
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
mod testing;

use axum::http::HeaderMap;
pub(crate) use symphony::{adapt::chat::content_spans, event::reasoning_text_tokens};

use crate::{config::ParserChoice, routers::grpc::utils::StreamFailure};

/// The request header that names the parser for one request, `legacy` or `symphony`, over the
/// gateway's `--parser` flag.
pub(crate) const HEADER: &str = "x-smg-parser";

/// Why the seam refused a request or failed to read an output, as six classes a caller can match
/// on. Every caller but one takes the text, which is the one the gateway answered or logged before
/// the classes were named: the non-streaming Messages path answers [`Self::InputRefused`] under
/// its own code at warn level, since that one is the output's doing. A count or a status by class
/// is a change of its own.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SymphonyError {
    /// The request's parser header names neither parser (a 400).
    #[error("{0}")]
    Header(String),
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

/// The parser that reads a request's response: the one its `X-SMG-Parser` header names, else
/// `default`, the gateway's `--parser` flag. The error names a header value that is neither
/// `legacy` nor `symphony`, for the 400 response.
pub(crate) fn choice(
    default: ParserChoice,
    headers: Option<&HeaderMap>,
) -> Result<ParserChoice, SymphonyError> {
    let Some(value) = headers.and_then(|headers| headers.get(HEADER)) else {
        return Ok(default);
    };
    let spelled = String::from_utf8_lossy(value.as_bytes());
    ParserChoice::parse(&spelled).ok_or_else(|| {
        SymphonyError::Header(format!(
            "{HEADER} must be legacy or symphony, not {spelled:?}"
        ))
    })
}

pub(crate) use self::{
    constraint::{
        constraint, grammar_start_of, turn_constraint, worker_constraint, Asks, AutoRequest,
        AutoTag, PreparedConstraint,
    },
    names::{
        check_names, keeps_special_tokens, labels_select_nothing, named_table, reasoning_open,
        set_aside_by_the_registry,
    },
    reader::{assign, engine_finish, finished, Assignment, Reader},
};
