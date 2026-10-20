//! What an engine's grammar backend enforces, as the servicer that launched
//! the engine read it off the engine's configuration, and the one refusal
//! every wire applies before a constraint reaches an engine that cannot
//! enforce it. The engines differ in what they do with such a request (one
//! answers off the constraint with a 200, one aborts it after admission with
//! a reason, one aborts it with no reason on its wire); behind this adapter
//! they answer alike: a refusal before the engine, the caller's error.

use std::borrow::Cow;

use engine_zmq_client::protocol::vllm::structured_outputs::{
    xgrammar_unsupported_feature_in_schema_text, xgrammar_unsupported_feature_in_structural_tag,
    StructuredOutputBackend, XgrammarUnsupportedFeature,
};

/// The engine's grammar backend, as it bears on what is refused before the
/// engine ([`refuse_unenforceable_constraint`]) and on the `_backend` the
/// wire that carries one stamps on structured requests. Such an engine keeps
/// the backend of its first structured request for its life, so the stamp is
/// a per-engine pin, not a per-request choice: under `Auto` (the engine's own
/// default, and the state of a connection nothing configured) every
/// constraint carries xgrammar, the one backend that compiles a structural
/// tag in the triggered-tags format; the engine's own frontend would send a
/// JSON schema with features xgrammar lacks to guidance instead, which on a
/// single-backend engine pins guidance when it comes first and fails every
/// structural tag after it; such a schema is refused here before it reaches
/// the engine, as that frontend refuses it under `backend=xgrammar`, because
/// xgrammar would compile it with the unsupported keywords dropped and answer
/// off the schema. An explicit backend is pinned on every request as
/// configured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StructuredOutputsBackendConfig {
    /// Nothing configured: the engine's own default, xgrammar on every
    /// constraint.
    #[default]
    Auto,
    /// An explicit backend, pinned on every request as configured.
    Pinned(StructuredOutputBackend),
    /// A backend this adapter does not know by name (an engine's plugin
    /// backend): nothing is judged ahead of the engine, which compiles and
    /// refuses on its own.
    Other,
    /// The engine runs no grammar backend (`--grammar-backend none`, or a
    /// start that leaves it none): no constraint can be enforced, so every
    /// one is refused before the engine.
    Disabled,
}

impl StructuredOutputsBackendConfig {
    /// The backend a structured request is stamped with under this config, on
    /// the wire that stamps one. `Other` and `Disabled` have none to stamp: a
    /// `Disabled` engine's constraints are refused before any stamp, and the
    /// stamping wire's servicer resolves its engine's names to `Auto` or
    /// `Pinned`; both read as the default.
    pub fn backend(self) -> StructuredOutputBackend {
        match self {
            Self::Pinned(backend) => backend,
            Self::Auto | Self::Other | Self::Disabled => StructuredOutputBackend::default(),
        }
    }

    /// Whether the constraints go to xgrammar, whose converter compiles a
    /// JSON schema with the features it lacks dropped.
    fn is_xgrammar(self) -> bool {
        matches!(
            self,
            Self::Auto | Self::Pinned(StructuredOutputBackend::Xgrammar)
        )
    }

    /// The backend's name, for a refusal that names it.
    fn name(self) -> &'static str {
        match self {
            Self::Auto | Self::Pinned(StructuredOutputBackend::Xgrammar) => "xgrammar",
            Self::Pinned(StructuredOutputBackend::Guidance) => "guidance",
            Self::Pinned(StructuredOutputBackend::Outlines) => "outlines",
            Self::Pinned(StructuredOutputBackend::LmFormatEnforcer) => "lm-format-enforcer",
            Self::Other => "its own",
            Self::Disabled => "none",
        }
    }
}

/// The constraint kinds a backend compiles, by the names the request's
/// `constraint` carries: a JSON schema, a regular expression, an EBNF
/// grammar, a structural tag. A kind outside the set is refused before the
/// engine, which would fail the request at grammar compile after admitting
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintKinds {
    pub json_schema: bool,
    pub regex: bool,
    pub ebnf: bool,
    pub structural_tag: bool,
}

impl ConstraintKinds {
    /// Every kind: the backends that compile all four, and a backend this
    /// adapter does not judge.
    pub const ALL: Self = Self {
        json_schema: true,
        regex: true,
        ebnf: true,
        structural_tag: true,
    };

    /// No kind: an engine with its grammar backend disabled.
    pub const NONE: Self = Self {
        json_schema: false,
        regex: false,
        ebnf: false,
        structural_tag: false,
    };

    /// Whether the backend compiles `constraint`'s kind. A bare JSON object
    /// is the backend's builtin JSON schema; a choice is the grammar xgrammar
    /// compiles it as.
    fn compiles(self, constraint: ConstraintRef<'_>) -> bool {
        match constraint {
            ConstraintRef::JsonSchema(_) | ConstraintRef::JsonObject => self.json_schema,
            ConstraintRef::Regex(_) => self.regex,
            ConstraintRef::Ebnf(_) | ConstraintRef::Choice => self.ebnf,
            ConstraintRef::StructuralTag(_) => self.structural_tag,
        }
    }

    /// The kinds compiled, named for a refusal: "JSON schemas and regular
    /// expressions".
    fn named(self) -> String {
        let names: Vec<&str> = [
            (self.json_schema, "JSON schemas"),
            (self.regex, "regular expressions"),
            (self.ebnf, "EBNF grammars"),
            (self.structural_tag, "structural tags"),
        ]
        .into_iter()
        .filter_map(|(compiles, name)| compiles.then_some(name))
        .collect();
        match names.split_last() {
            None => "nothing".to_string(),
            Some((last, [])) => (*last).to_string(),
            Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        }
    }
}

/// What the engine enforces: its grammar backend and the constraint kinds
/// that backend compiles, as the servicer that launched the engine read them
/// off its configuration and set them on the connection
/// (`ZmqEngineClient::set_structured_outputs_support`). `Default` is the
/// engine's `auto` with every kind, what the wire that stamps a backend on
/// each structured request ran before this support existed; a connection
/// nobody configured gets its own wire's default
/// (`client::default_structured_outputs_support`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredOutputsSupport {
    pub backend: StructuredOutputsBackendConfig,
    pub kinds: ConstraintKinds,
}

impl StructuredOutputsSupport {
    /// An engine with its grammar backend disabled: nothing is enforced.
    pub const DISABLED: Self = Self {
        backend: StructuredOutputsBackendConfig::Disabled,
        kinds: ConstraintKinds::NONE,
    };

    pub const fn new(backend: StructuredOutputsBackendConfig, kinds: ConstraintKinds) -> Self {
        Self { backend, kinds }
    }
}

impl Default for StructuredOutputsSupport {
    fn default() -> Self {
        StructuredOutputsBackendConfig::default().into()
    }
}

/// A backend alone: it compiles every kind, except a disabled one, which
/// compiles none.
impl From<StructuredOutputsBackendConfig> for StructuredOutputsSupport {
    fn from(backend: StructuredOutputsBackendConfig) -> Self {
        let kinds = if backend == StructuredOutputsBackendConfig::Disabled {
            ConstraintKinds::NONE
        } else {
            ConstraintKinds::ALL
        };
        Self { backend, kinds }
    }
}

/// A request's constraint as every wire's proto carries it (the `constraint`
/// one-of of its sampling params): the kind, with the text of the kinds that
/// carry one. Each wire's module builds the view from its own proto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintRef<'a> {
    /// A JSON schema: a `response_format` schema, or the tool parameters of a
    /// forced tool call on a parser without structural tags.
    JsonSchema(&'a str),
    /// A regular expression the output must match.
    Regex(&'a str),
    /// An EBNF grammar the output must conform to.
    Ebnf(&'a str),
    /// A structural tag (JSON text): the tool parameters of a constrained
    /// call, or a JSON answer's schema, between the template's tags.
    StructuralTag(&'a str),
    /// A bare JSON object (`json_object`): the backend's builtin JSON schema.
    JsonObject,
    /// A choice among literal strings.
    Choice,
}

impl ConstraintRef<'_> {
    /// The constraint as a refusal's subject, naming what builds one.
    fn source(self) -> &'static str {
        match self {
            Self::JsonSchema(_) => {
                "The JSON schema constraint (a response_format schema, or the tool parameters of a \
                 forced tool call)"
            }
            Self::Regex(_) => "The regex constraint",
            Self::Ebnf(_) => "The EBNF grammar constraint",
            Self::StructuralTag(_) => {
                "The structural tag constraint (the tool parameters of a constrained tool call: \
                 `tool_choice` required, a named function, or every turn on an always-constrained \
                 worker; or a JSON answer's schema)"
            }
            Self::JsonObject => "The JSON object constraint (a `json_object` response format)",
            Self::Choice => "The choice constraint",
        }
    }
}

/// Refuse, before the engine, a constraint its grammar backend cannot
/// enforce; `Ok` for a request without one, and for one the backend compiles
/// whole. Every wire applies it on its request path, with the support the
/// servicer set on the connection.
///
/// Three rules, in order. An engine whose grammar backend is disabled
/// enforces nothing, so every constraint is refused: such an engine takes the
/// request and aborts it after admission, and one wire reports that abort
/// without its reason, as a bare empty completion the caller reads as an
/// answer. A kind outside what the backend compiles ([`ConstraintKinds`]) is
/// refused: the engine would admit the request and fail it at grammar
/// compile. Under xgrammar (`Auto`, or pinned) a JSON schema with a feature
/// xgrammar's converter drops is refused, as the request's own schema or
/// inside a structural tag alike (its `json_schema` elements go through the
/// same converter): xgrammar compiles such a schema with the keyword dropped
/// and the engine answers off the schema with a 200, which is why an engine's
/// own frontend refuses it under an xgrammar backend (a 400); a backend that
/// enforces such a schema itself takes the constraint
/// as is, and so does one this adapter does not know. The message names the
/// constraint's source, the keyword, where it stands and, in a tag, the tags
/// around it by their `begin` (a call's opens with the tool's name).
pub fn refuse_unenforceable_constraint(
    constraint: Option<ConstraintRef<'_>>,
    support: StructuredOutputsSupport,
) -> Result<(), String> {
    let Some(constraint) = constraint else {
        return Ok(());
    };
    let source = constraint.source();
    if support.backend == StructuredOutputsBackendConfig::Disabled {
        return Err(format!(
            "{source} cannot be enforced: this engine runs with its grammar backend disabled \
             (`none`), so it serves no structured output (no JSON schema, regex, EBNF grammar or \
             structural tag)"
        ));
    }
    if !support.kinds.compiles(constraint) {
        return Err(format!(
            "{source} cannot be enforced: `{}`, the structured-output backend this engine runs, \
             compiles {} only",
            support.backend.name(),
            support.kinds.named()
        ));
    }
    if !support.backend.is_xgrammar() {
        return Ok(());
    }
    let feature = match constraint {
        ConstraintRef::JsonSchema(schema) => xgrammar_unsupported_feature_in_schema_text(schema),
        ConstraintRef::StructuralTag(tag) => xgrammar_unsupported_feature_in_structural_tag(tag),
        _ => None,
    };
    match feature {
        Some(feature) => Err(xgrammar_cannot_enforce(source, &feature)),
        None => Ok(()),
    }
}

/// The refusal of a schema feature xgrammar drops: the source, the keyword,
/// its place and the tags around it.
fn xgrammar_cannot_enforce(source: &str, feature: &XgrammarUnsupportedFeature) -> String {
    let place = if feature.pointer.is_empty() {
        Cow::Borrowed("its root")
    } else {
        Cow::Owned(format!("`{}`", feature.pointer))
    };
    let around = match feature.openers.as_slice() {
        [] => String::new(),
        [one] => format!(" (inside the tag that begins `{}`)", printable(one)),
        many => {
            let begins: Vec<String> = many
                .iter()
                .map(|begin| format!("`{}`", printable(begin)))
                .collect();
            format!(" (inside the tags that begin {})", begins.join(" then "))
        }
    };
    format!(
        "{source} uses `{}` at {place}{around}, a JSON-schema feature not supported by xgrammar, \
         the structured-output backend this engine runs: xgrammar would drop it and answer off \
         the schema",
        feature.keyword
    )
}

/// `text` for an error message: its control characters (a tag's `begin`
/// often ends in a newline) written as escapes, the rest as it is.
fn printable(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_control() {
            out.extend(ch.escape_default());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use StructuredOutputsBackendConfig::{Auto, Disabled, Other, Pinned};

    use super::*;

    const SCHEMA: &str = r#"{"type":"object","properties":{"city":{"type":"string"}}}"#;
    const MULTIPLE_OF: &str = r#"{"type":"integer","multipleOf":3}"#;

    fn every_kind() -> [ConstraintRef<'static>; 6] {
        [
            ConstraintRef::JsonSchema(SCHEMA),
            ConstraintRef::Regex("[0-9]+"),
            ConstraintRef::Ebnf("root ::= \"a\""),
            ConstraintRef::StructuralTag(r#"{"format":{}}"#),
            ConstraintRef::JsonObject,
            ConstraintRef::Choice,
        ]
    }

    /// No constraint, nothing to refuse, whatever the engine runs.
    #[test]
    fn a_request_without_a_constraint_passes_every_support() {
        for support in [
            StructuredOutputsSupport::default(),
            StructuredOutputsSupport::DISABLED,
            StructuredOutputsSupport::from(Other),
        ] {
            assert_eq!(refuse_unenforceable_constraint(None, support), Ok(()));
        }
    }

    /// An engine with its grammar backend disabled enforces nothing: every
    /// kind is refused, and the refusal says why in the engine's own words.
    #[test]
    fn a_disabled_backend_refuses_every_kind() {
        for constraint in every_kind() {
            let refused = refuse_unenforceable_constraint(
                Some(constraint),
                StructuredOutputsSupport::DISABLED,
            )
            .expect_err("refused");
            assert!(
                refused.contains("grammar backend disabled (`none`)"),
                "{constraint:?}: {refused}"
            );
        }
        // The subject names the request's own field.
        let refused = refuse_unenforceable_constraint(
            Some(ConstraintRef::Regex("[0-9]+")),
            StructuredOutputsSupport::DISABLED,
        )
        .expect_err("refused");
        assert!(
            refused.starts_with("The regex constraint cannot be enforced"),
            "{refused}"
        );
        assert_eq!(
            StructuredOutputsSupport::from(Disabled),
            StructuredOutputsSupport::DISABLED
        );
    }

    /// A kind the backend does not compile is refused, naming the backend
    /// and what it compiles; the kinds it compiles pass.
    #[test]
    fn a_kind_outside_the_backends_set_is_refused() {
        let outlines = StructuredOutputsSupport::new(
            Pinned(StructuredOutputBackend::Outlines),
            ConstraintKinds {
                json_schema: true,
                regex: true,
                ebnf: false,
                structural_tag: false,
            },
        );
        for constraint in [
            ConstraintRef::Ebnf("root ::= \"a\""),
            ConstraintRef::StructuralTag(r#"{"format":{}}"#),
            ConstraintRef::Choice,
        ] {
            let refused =
                refuse_unenforceable_constraint(Some(constraint), outlines).expect_err("refused");
            assert!(
                refused.contains(
                    "`outlines`, the structured-output backend this engine runs, compiles JSON \
                     schemas and regular expressions only"
                ),
                "{constraint:?}: {refused}"
            );
        }
        for constraint in [
            ConstraintRef::JsonSchema(MULTIPLE_OF),
            ConstraintRef::Regex("[0-9]+"),
            ConstraintRef::JsonObject,
        ] {
            assert_eq!(
                refuse_unenforceable_constraint(Some(constraint), outlines),
                Ok(()),
                "{constraint:?}"
            );
        }
        // One kind reads without a conjunction, three with the serial form.
        assert_eq!(
            ConstraintKinds {
                json_schema: true,
                ..ConstraintKinds::NONE
            }
            .named(),
            "JSON schemas"
        );
        assert_eq!(
            ConstraintKinds {
                structural_tag: false,
                ..ConstraintKinds::ALL
            }
            .named(),
            "JSON schemas, regular expressions and EBNF grammars"
        );
    }

    /// Under xgrammar, pinned or as the engine's default, a schema with a
    /// feature xgrammar drops is refused as the request's own schema and
    /// inside a structural tag alike; the other kinds pass, as does a schema
    /// xgrammar compiles whole; a backend that enforces such a schema itself,
    /// and one this adapter does not know, take it as it is.
    #[test]
    fn a_schema_xgrammar_cannot_enforce_is_refused_under_xgrammar_alone() {
        let tag = serde_json::json!({"format": {"type": "triggered_tags",
            "triggers": ["<tool_call>"], "tags": [
                {"begin": "<tool_call>{\"name\": \"tagged\", \"arguments\": ",
                 "content": {"type": "json_schema", "json_schema":
                    {"type": "object", "properties": {"tags": {"type": "array",
                        "items": {"type": "string"}, "uniqueItems": true}}}},
                 "end": "}</tool_call>"}],
            "at_least_one": true}})
        .to_string();
        for backend in [Auto, Pinned(StructuredOutputBackend::Xgrammar)] {
            let support = StructuredOutputsSupport::from(backend);
            let refused = refuse_unenforceable_constraint(
                Some(ConstraintRef::JsonSchema(MULTIPLE_OF)),
                support,
            )
            .expect_err("refused");
            assert!(
                refused.contains("tool parameters of a forced tool call")
                    && refused.contains("`multipleOf` at its root")
                    && refused.contains("not supported by xgrammar"),
                "{backend:?}: {refused}"
            );
            let refused =
                refuse_unenforceable_constraint(Some(ConstraintRef::StructuralTag(&tag)), support)
                    .expect_err("refused");
            assert!(
                refused.contains("structural tag constraint")
                    && refused.contains(
                        "`uniqueItems` at `/format/tags/0/content/json_schema/properties/tags` \
                         (inside the tag that begins `<tool_call>{\"name\": \"tagged\", \
                         \"arguments\": `)"
                    ),
                "{backend:?}: {refused}"
            );
            for constraint in [
                ConstraintRef::JsonSchema(SCHEMA),
                ConstraintRef::Regex("[0-9]+"),
                ConstraintRef::Ebnf("root ::= \"a\""),
                ConstraintRef::JsonObject,
                ConstraintRef::Choice,
            ] {
                assert_eq!(
                    refuse_unenforceable_constraint(Some(constraint), support),
                    Ok(()),
                    "{backend:?}: {constraint:?}"
                );
            }
        }
        for backend in [Pinned(StructuredOutputBackend::Guidance), Other] {
            let support = StructuredOutputsSupport::from(backend);
            for constraint in [
                ConstraintRef::JsonSchema(MULTIPLE_OF),
                ConstraintRef::StructuralTag(&tag),
            ] {
                assert_eq!(
                    refuse_unenforceable_constraint(Some(constraint), support),
                    Ok(()),
                    "{backend:?}: {constraint:?}"
                );
            }
        }
    }

    /// A connection nobody configured keeps what its wire ran before: the
    /// stamping wire the engine's `auto`, the others no judgement at all.
    #[test]
    fn an_unconfigured_connection_keeps_its_wires_old_behaviour() {
        use crate::{client::default_structured_outputs_support, ZmqDialect};
        assert_eq!(
            default_structured_outputs_support(ZmqDialect::Vllm),
            StructuredOutputsSupport::default()
        );
        for dialect in [ZmqDialect::Sglang, ZmqDialect::TokenSpeed] {
            let support = default_structured_outputs_support(dialect);
            assert_eq!(
                support,
                StructuredOutputsSupport::from(Other),
                "{dialect:?}"
            );
            assert_eq!(
                refuse_unenforceable_constraint(
                    Some(ConstraintRef::JsonSchema(MULTIPLE_OF)),
                    support
                ),
                Ok(()),
                "{dialect:?}"
            );
        }
    }

    /// The stamp a wire puts on a structured request: the pinned backend,
    /// else the default; a config that stamps nothing reads as the default.
    #[test]
    fn the_stamp_is_the_pinned_backend_or_the_default() {
        assert_eq!(
            Pinned(StructuredOutputBackend::Guidance).backend(),
            StructuredOutputBackend::Guidance
        );
        for config in [Auto, Other, Disabled] {
            assert_eq!(
                config.backend(),
                StructuredOutputBackend::Xgrammar,
                "{config:?}"
            );
        }
        assert_eq!(StructuredOutputsBackendConfig::default(), Auto);
        assert_eq!(
            StructuredOutputsSupport::default(),
            StructuredOutputsSupport::new(Auto, ConstraintKinds::ALL)
        );
    }

    /// A tag's `begin` with control characters reads as escapes in the
    /// refusal, and several tags around a schema are listed outermost first.
    #[test]
    fn the_refusal_prints_the_tags_around_a_schema() {
        let feature = XgrammarUnsupportedFeature {
            keyword: Cow::Borrowed("multipleOf"),
            pointer: "/properties/n".to_string(),
            openers: vec!["<outer>\n".to_string(), "<inner>".to_string()],
        };
        let message = xgrammar_cannot_enforce("The JSON schema constraint", &feature);
        assert!(
            message.contains(
                "`multipleOf` at `/properties/n` (inside the tags that begin `<outer>\\n` then \
                 `<inner>`)"
            ),
            "{message}"
        );
    }
}
