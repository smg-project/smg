//! Template loading and validation, ported from transformers'
//! `response_templates.py`. Loading also refuses, as `Unsupported`, what the
//! parser would not reproduce exactly.

use std::{cmp::Reverse, sync::Arc};

use serde_json::{Map, Value};

use crate::{
    content_parsers::{self, Content, ContentParser},
    error::{LoadError, Unsupported},
    py,
    pyre::{Pattern, Role},
};

/// An open or close delimiter; `literals` when it came from literal strings,
/// `can_extend` when one of them is a strict prefix of another.
#[derive(Debug)]
pub(crate) struct Delimiter {
    pub(crate) pattern: Pattern,
    pub(crate) literals: bool,
    pub(crate) can_extend: bool,
}

/// transformers: `ResponseTemplateField`.
#[derive(Debug)]
pub(crate) struct ResponseTemplateField {
    pub(crate) name: String,
    pub(crate) open: Option<Delimiter>,
    pub(crate) close: Option<Delimiter>,
    pub(crate) content: Content,
    pub(crate) repeats: bool,
    pub(crate) join: Option<String>,
    pub(crate) optional: bool,
    pub(crate) transform: Option<Value>,
    pub(crate) transform_each: bool,
}

#[derive(Debug)]
pub(crate) struct Template {
    pub(crate) defaults: Map<String, Value>,
    pub(crate) fields: Vec<ResponseTemplateField>,
    pub(crate) implicit: Option<usize>,
    start_anchor: Pattern,
}

/// transformers: `ResponseTemplate`, a loaded template. Cheap to clone.
#[derive(Debug, Clone)]
pub struct ResponseTemplate(pub(crate) Arc<Template>);

impl ResponseTemplate {
    /// transformers: `truncate_past_last_anchor`. The text after the last
    /// start anchor, or the whole text when there is none.
    pub fn truncate_past_last_anchor<'a>(&self, text: &'a str) -> &'a str {
        match self.0.start_anchor.finditer(text).last() {
            Some(m) => &text[m.end..],
            None => text,
        }
    }
}

fn invalid(scope: &str, message: impl Into<String>) -> LoadError {
    LoadError::Invalid {
        scope: scope.to_owned(),
        message: message.into(),
    }
}

fn unsupported(scope: &str, feature: Unsupported) -> LoadError {
    LoadError::Unsupported {
        scope: scope.to_owned(),
        feature,
    }
}

/// transformers: `_compile_anchor`.
fn compile_anchor(
    scope: &str,
    spec: &Map<String, Value>,
    literal_key: &str,
    pattern_key: &str,
    role: Role,
) -> Result<Option<Delimiter>, LoadError> {
    if spec.contains_key(literal_key) && spec.contains_key(pattern_key) {
        return Err(invalid(
            scope,
            format!("cannot specify both '{literal_key}' and '{pattern_key}'"),
        ));
    }
    if let Some(raw) = spec.get(literal_key) {
        let mut literals: Vec<String> = Vec::new();
        match raw {
            Value::String(s) => literals.push(s.clone()),
            Value::Array(items) if items.is_empty() => {
                return Err(invalid(
                    scope,
                    format!("'{literal_key}' list must contain at least one literal"),
                ));
            }
            Value::Array(items) => {
                for item in items {
                    let Value::String(s) = item else {
                        return Err(invalid(
                            scope,
                            format!("'{literal_key}' list must contain only strings"),
                        ));
                    };
                    if !literals.contains(s) {
                        literals.push(s.clone());
                    }
                }
            }
            other => {
                return Err(invalid(
                    scope,
                    format!(
                        "'{literal_key}' must be a string or list of strings, got {}",
                        py::type_name(other)
                    ),
                ))
            }
        }
        if literals.iter().any(String::is_empty) {
            return Err(invalid(
                scope,
                format!("'{literal_key}' literals cannot be empty strings"),
            ));
        }
        let can_extend = literals
            .iter()
            .any(|a| literals.iter().any(|b| a != b && a.starts_with(b.as_str())));
        // Longest first (in characters, stable), as transformers sorts them.
        literals.sort_by_key(|s| Reverse(s.chars().count()));
        return Ok(Some(Delimiter {
            pattern: Pattern::literals(&literals),
            literals: true,
            can_extend,
        }));
    }
    let Some(raw) = spec.get(pattern_key) else {
        return Ok(None);
    };
    let Value::String(source) = raw else {
        return Err(invalid(
            scope,
            format!("invalid {pattern_key} regex: not a string"),
        ));
    };
    let pattern = Pattern::new(source, role).map_err(|feature| unsupported(scope, feature))?;
    Ok(Some(Delimiter {
        pattern,
        literals: false,
        can_extend: false,
    }))
}

/// transformers: `_validate_template_shape`.
fn validate_template_shape(spec: &Value) -> Result<&Map<String, Value>, LoadError> {
    let scope = "response_template";
    let Value::Object(spec) = spec else {
        return Err(invalid(
            scope,
            format!(
                "response_template must be a dict, got {}",
                py::type_name(spec)
            ),
        ));
    };
    if spec.get("version").is_some_and(|v| !py::equals_one(v)) {
        return Err(invalid(scope, "Unsupported response_template version"));
    }
    let allowed = [
        "version",
        "defaults",
        "fields",
        "start_anchor",
        "start_anchor_pattern",
    ];
    if let Some(key) = spec.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(invalid(
            scope,
            format!("Unknown keys in response_template: {key:?}"),
        ));
    }
    if spec.get("defaults").is_some_and(|d| !d.is_object()) {
        return Err(invalid(scope, "response_template.defaults must be a dict"));
    }
    match spec.get("fields") {
        Some(Value::Object(fields)) if !fields.is_empty() => Ok(spec),
        _ => Err(invalid(
            scope,
            "response_template.fields must be a non-empty dict",
        )),
    }
}

const ALLOWED_FIELD_KEYS: [&str; 11] = [
    "open",
    "open_pattern",
    "close",
    "close_pattern",
    "content",
    "content_args",
    "repeats",
    "join",
    "optional",
    "transform",
    "transform_each",
];

/// transformers: `_build_field`.
fn build_field(name: &str, field: &Value) -> Result<ResponseTemplateField, LoadError> {
    let scope = format!("Field '{name}'");
    let Value::Object(field) = field else {
        return Err(invalid(&scope, format!("{scope} must be a dict")));
    };
    if let Some(key) = field
        .keys()
        .find(|k| !ALLOWED_FIELD_KEYS.contains(&k.as_str()))
    {
        return Err(invalid(&scope, format!("unknown keys [{key:?}]")));
    }
    let parser = match field.get("content") {
        None => ContentParser::Text,
        Some(Value::String(s)) => ContentParser::from_name(s)
            .ok_or_else(|| invalid(&scope, format!("unknown content parser '{s}'")))?,
        Some(other) => return Err(invalid(&scope, format!("unknown content parser {other}"))),
    };
    let open = compile_anchor(&scope, field, "open", "open_pattern", Role::Delimiter)?;
    let close = compile_anchor(&scope, field, "close", "close_pattern", Role::Delimiter)?;
    let join = match field.get("join") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            return Err(invalid(
                &scope,
                format!("'join' must be a string, got {}", py::type_name(other)),
            ))
        }
    };
    let repeats = field.get("repeats").is_some_and(py::truthy);
    if join.is_some() && !repeats {
        return Err(invalid(&scope, "'join' requires 'repeats': true"));
    }
    let transform = field.get("transform").filter(|t| !t.is_null()).cloned();
    let transform_each = match field.get("transform_each") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => {
            return Err(invalid(
                &scope,
                format!(
                    "transform_each must be a bool, got {}",
                    py::type_name(other)
                ),
            ))
        }
    };
    if transform_each && transform.is_none() {
        return Err(invalid(
            &scope,
            "transform_each is set but no transform was provided",
        ));
    }
    match &transform {
        Some(transform) => content_parsers::validate_transform_strings(&scope, transform)?,
        None => {
            // Named captures only reach the output through a transform.
            let delimiters = open.iter().chain(close.iter());
            if delimiters
                .flat_map(|d| d.pattern.group_names())
                .next()
                .is_some()
            {
                return Err(invalid(
                    &scope,
                    "open_pattern/close_pattern declares named group(s), but the field has no 'transform'",
                ));
            }
        }
    }
    let content = Content::new(parser, field.get("content_args"))
        .map_err(|feature| unsupported(&scope, feature))?;
    let nullable = |d: &Option<Delimiter>| d.as_ref().is_some_and(|d| d.pattern.nullable());
    if nullable(&open) && nullable(&close) {
        return Err(unsupported(&scope, Unsupported::EmptyMatch));
    }
    if transform
        .as_ref()
        .is_some_and(|t| content_parsers::shares_values(t, transform_each, parser))
    {
        return Err(unsupported(&scope, Unsupported::SharedValue));
    }
    Ok(ResponseTemplateField {
        name: name.to_owned(),
        open,
        close,
        content,
        repeats,
        join,
        optional: field.get("optional").is_none_or(py::truthy),
        transform,
        transform_each,
    })
}

/// transformers: `load_response_template`. Rejects what transformers rejects
/// (`LoadError::Invalid`), and what this crate would not parse exactly as it
/// does (`LoadError::Unsupported`, naming the construct).
pub fn load_response_template(spec: &Value) -> Result<ResponseTemplate, LoadError> {
    let spec = validate_template_shape(spec)?;
    let fields = spec
        .get("fields")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .map(|(name, raw)| build_field(name, raw))
        .collect::<Result<Vec<_>, _>>()?;
    let implicit: Vec<&str> = fields
        .iter()
        .filter(|f| f.open.is_none())
        .map(|f| f.name.as_str())
        .collect();
    if implicit.len() > 1 {
        return Err(invalid(
            "response_template",
            format!(
                "At most one field may omit 'open'/'open_pattern'. Found: {}",
                implicit.join(", ")
            ),
        ));
    }
    let scope = "response_template";
    let start_anchor = compile_anchor(
        scope,
        spec,
        "start_anchor",
        "start_anchor_pattern",
        Role::Finditer,
    )?
    .ok_or_else(|| {
        invalid(
            scope,
            "response_template must define 'start_anchor' or 'start_anchor_pattern'.",
        )
    })?
    .pattern;
    let defaults = spec
        .get("defaults")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    // transformers appends to a list default in place, so it would grow
    // across parses.
    if fields.iter().any(|f| {
        f.repeats && f.join.is_none() && defaults.get(&f.name).is_some_and(Value::is_array)
    }) {
        return Err(unsupported(scope, Unsupported::SharedValue));
    }
    let implicit = fields.iter().position(|f| f.open.is_none());
    Ok(ResponseTemplate(Arc::new(Template {
        defaults,
        fields,
        implicit,
        start_anchor,
    })))
}
