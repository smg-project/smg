//! Where an output grammar holds a tool's calls more loosely than the tool's schema. A tool that
//! asks for its calls to follow its schema (`"strict": true`) cannot have that kept where the
//! grammar takes a part of the schema as any value, drops a pattern, leaves a raw string's length
//! or format free, or lets a call leave out a required parameter. A JSON value's own parts are
//! xgrammar's to hold.

use serde_json::{Map, Value};

use super::{
    values::{self, Kind, Root, Source, ValueOption},
    Output, OutputChoice, OutputPosition, OutputSpelling, Syntax,
};

/// The keywords of a string schema that raw text does not hold a value to.
const RAW_FREE: [&str; 3] = ["minLength", "maxLength", "format"];

/// The keywords of a parameters schema that the arguments do not hold a call to.
const ARGUMENTS_FREE: [&str; 12] = [
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
];

impl OutputSpelling {
    /// Why the grammar of `request` cannot hold the calls of a callable tool that asks for its
    /// schema to be followed (`"strict": true`) to that schema, naming the tool and the first
    /// part of its schema the calls are held to more loosely; `None` where the grammar holds no
    /// call, or holds every such tool's calls to its schema.
    pub fn unenforced(&self, request: &Output<'_>) -> Option<String> {
        let holds_calls = match request.position {
            OutputPosition::Start | OutputPosition::InThought | OutputPosition::AfterThought => {
                true
            }
            OutputPosition::InCall => request.parallel && request.choice != OutputChoice::One,
            OutputPosition::InAnswer | OutputPosition::Elsewhere => false,
        };
        if !holds_calls {
            return None;
        }
        request
            .callable
            .iter()
            .filter(|tool| tool.function.strict == Some(true))
            .find_map(|tool| {
                let why = self.loose(&tool.function.parameters)?;
                Some(format!(
                    "tool '{}' is strict, and this model's calls cannot be held to {why}",
                    tool.function.name
                ))
            })
    }

    /// The first part of `schema`, a tool's parameters schema, its calls are held to more
    /// loosely than it says.
    fn loose(&self, schema: &Value) -> Option<String> {
        if values::drop_uncompiled_patterns(&mut schema.clone()) {
            return Some("a pattern xgrammar does not compile".to_string());
        }
        let root = Root(schema);
        let resolved = root.resolved();
        if std::ptr::eq(resolved, &values::ANY) {
            return Some("its parameters' $ref, which does not resolve".to_string());
        }
        let object = match resolved {
            Value::Object(object) => object,
            Value::Bool(_) => return None,
            _ => return Some("a parameters schema that is not an object".to_string()),
        };
        let typed_object = |name: &Value| name.as_str().and_then(Kind::parse) == Some(Kind::Object);
        let object_type = match object.get("type") {
            None => true,
            Some(Value::Array(names)) => names.iter().any(typed_object),
            Some(name) => typed_object(name),
        };
        if !object_type {
            return Some("a parameters schema that is not an object".to_string());
        }
        if let Some(keyword) = ARGUMENTS_FREE.iter().find(|key| object.contains_key(**key)) {
            return Some(format!("its parameters' {keyword}"));
        }
        let declared = object.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = object
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if let Some(key) = required
            .iter()
            .find(|key| !declared.is_some_and(|declared| declared.contains_key(**key)))
        {
            return Some(format!(
                "required parameter '{key}', which it does not declare"
            ));
        }
        for (key, schema) in declared.into_iter().flatten() {
            let what = format!("parameter '{key}'");
            if let Some(why) =
                self.loose_value(root, schema, &what, required.contains(&key.as_str()))
            {
                return Some(why);
            }
        }
        match object.get("additionalProperties") {
            None | Some(Value::Bool(false)) => None,
            Some(additional) => self.loose_value(root, additional, "a free parameter", false),
        }
    }

    /// The first part of `schema`, the schema of `what`'s value, its values are held to more
    /// loosely than it says, or a required parameter no value can be written for.
    fn loose_value(
        &self,
        root: Root<'_>,
        schema: &Value,
        what: &str,
        required: bool,
    ) -> Option<String> {
        let (options, loose) = root.options_loosely(schema);
        if let Some(why) = loose {
            return Some(format!("{what}: {why}"));
        }
        for option in &options {
            if let (Kind::String, Source::Schema(schema)) = (option.kind, &option.source) {
                if let Some(keyword) = raw_free(schema) {
                    return Some(format!("the {keyword} of {what}, written as raw text"));
                }
            }
        }
        let syntax = if self.next_key_closes {
            Syntax::Joined
        } else {
            Syntax::Closed
        };
        let raw = self.raw_excludes(syntax);
        if required && !options.iter().any(|option| writable(option, &raw)) {
            return Some(format!(
                "required {what}, which no value can be written for"
            ));
        }
        None
    }
}

/// The first keyword of a string schema raw text does not hold a value to.
fn raw_free(schema: &Map<String, Value>) -> Option<&'static str> {
    RAW_FREE.into_iter().find(|key| schema.contains_key(*key))
}

/// Whether `option` can be written: a string constant holding a word a raw value cannot hold is
/// left out of the grammar.
fn writable(option: &ValueOption<'_>, raw: &[String]) -> bool {
    match &option.source {
        Source::Literal(Value::String(value)) => {
            !raw.iter().any(|word| value.contains(word.as_str()))
        }
        _ => true,
    }
}
