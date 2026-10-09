//! Kimi-K2.5 tool-declaration encoder. See module-level docs.
//!
//! Ported from <https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tool_declaration_ts.py>.
//! Mirrors the upstream Python reference function-by-function. Output must be
//! byte-equal to the Python reference; gated by golden tests in
//! `tests/kimi_k25_renderer_detection.rs`.

use std::{collections::HashMap, fmt::Write};

use anyhow::Result;
use serde_json::Value;

use crate::chat_template::{ChatTemplateParams, ChatTemplateState};

const TS_INDENT: &str = "  ";
const TS_FIELD_DELIMITER: &str = ",\n";

// On overflow, emit `any` and continue — never panic on adversarial schemas.
const MAX_RECURSION_DEPTH: usize = 32;

pub fn encode_tools_to_typescript(tools: &[Value]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut function_strs: Vec<String> = Vec::new();
    for t in tools {
        // Skip unsupported tool types (e.g. "_plugin"), matching upstream.
        if t.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        // Upstream `if func_def:` skips a missing/empty function object.
        let func = match t.get("function") {
            Some(f) if f.as_object().is_some_and(|o| !o.is_empty()) => f,
            _ => continue,
        };
        match openai_function_to_typescript(func) {
            Some(s) => function_strs.push(s),
            // Upstream raises here; the caller catches and renders JSON tool
            // declarations instead. We return `None` so `tools_ts_str` stays
            // undefined and the chat template takes its JSON fallback branch.
            None => {
                tracing::warn!(
                    "Kimi-K2.5 tool schema unsupported by the TypeScript encoder; \
                     leaving tools_ts_str undefined so the template falls back to JSON \
                     tool declarations"
                );
                return None;
            }
        }
    }
    if function_strs.is_empty() {
        return None;
    }
    let mut out = String::from("# Tools\n\n## functions\nnamespace functions {\n");
    out.push_str(&function_strs.join("\n"));
    out.push('\n');
    out.push_str("}\n");
    Some(out)
}

/// Renderer for `Renderer::KimiK25Tools`. Computes `tools_ts_str` and merges
/// it into `template_kwargs`, then delegates to the standard minijinja path,
/// which also says whether the prompt continues the final message.
pub(crate) fn apply_kimi_k25_tools(
    chat_template: &ChatTemplateState,
    messages: &[Value],
    params: ChatTemplateParams,
) -> Result<(String, bool)> {
    let ts_str = params.tools.and_then(encode_tools_to_typescript);

    let owned: Option<HashMap<String, Value>> = match (params.template_kwargs, ts_str.as_ref()) {
        (Some(existing), Some(ts)) => {
            let mut m = existing.clone();
            m.insert("tools_ts_str".to_string(), Value::String(ts.clone()));
            Some(m)
        }
        (None, Some(ts)) => {
            let mut m = HashMap::with_capacity(1);
            m.insert("tools_ts_str".to_string(), Value::String(ts.clone()));
            Some(m)
        }
        _ => None, // No tools → leave tools_ts_str undefined
    };

    let new_params = ChatTemplateParams {
        template_kwargs: owned.as_ref().or(params.template_kwargs),
        ..params
    };
    chat_template.apply_with_continuation(messages, new_params)
}

// ---------------------------------------------------------------------------
// Function-level encoding
// ---------------------------------------------------------------------------

fn openai_function_to_typescript(function: &Value) -> Option<String> {
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let mut registry = SchemaRegistry::default();
    let parsed = ParameterTypeObject::parse(&parameters, &mut registry);

    let mut interfaces: Vec<String> = Vec::new();
    let mut root_interface_name: Option<&str> = None;

    if registry.has_self_ref {
        root_interface_name = Some("parameters");
        let body = parsed
            .properties
            .iter()
            .map(|p| p.to_typescript(TS_INDENT, &registry))
            .collect::<Vec<_>>()
            .join(TS_FIELD_DELIMITER);
        let body = if body.is_empty() {
            String::new()
        } else {
            format!("\n{body}\n")
        };
        interfaces.push(format!("interface parameters {{{body}}}"));
    }

    // Emit definitions in `$defs` insertion order to match upstream, which
    // iterates an insertion-ordered Python dict (do NOT sort).
    let defs_ordered: Vec<(String, Value)> = registry
        .order
        .iter()
        .filter_map(|name| {
            registry
                .definitions
                .get(name)
                .map(|v| (name.clone(), v.clone()))
        })
        .collect();
    for (name, schema) in defs_ordered {
        let obj_type = parse_parameter_type(&schema, &mut registry);
        let body = obj_type.to_typescript("", &registry);
        let mut def_str = String::new();
        if let Some(desc) = schema.get("description").and_then(Value::as_str) {
            if !desc.is_empty() {
                def_str.push_str(&format_description(desc, ""));
                def_str.push('\n');
            }
        }
        #[expect(
            clippy::unwrap_used,
            reason = "write!/writeln! into String cannot fail"
        )]
        {
            write!(def_str, "interface {name} {body}").unwrap();
        }
        interfaces.push(def_str);
    }

    let interface_str = interfaces.join("\n");
    let function_name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("function");
    let type_def = match root_interface_name {
        Some(n) => format!("type {function_name} = (_: {n}) => any;"),
        None => format!(
            "type {function_name} = (_: {}) => any;",
            parsed.to_typescript("", &registry)
        ),
    };
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    let desc_block = if description.is_empty() {
        String::new()
    } else {
        format_description(description, "")
    };

    // Upstream raises on unsupported schema constructs, aborting the whole TS
    // encoding; mirror that by signalling the caller to drop to JSON.
    if registry.unsupported {
        return None;
    }

    Some(
        [interface_str, desc_block, type_def]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

// ---------------------------------------------------------------------------
// Schema parsing
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SchemaRegistry {
    definitions: HashMap<String, Value>,
    /// Insertion order of `$defs` keys. Upstream iterates a Python `dict`,
    /// which is insertion-ordered, so we must preserve that order (not sort)
    /// to stay byte-equal for schemas whose keys aren't alphabetical.
    order: Vec<String>,
    has_self_ref: bool,
    depth: usize,
    /// Set when a schema construct is encountered that upstream would `raise`
    /// on (unresolvable `$ref`, unsupported reference format, or an
    /// unrecognized schema shape). The caller abandons the TypeScript
    /// namespace and leaves `tools_ts_str` undefined so the chat template
    /// falls back to JSON tool declarations — matching upstream semantics.
    unsupported: bool,
}

impl SchemaRegistry {
    fn register_definitions(&mut self, defs: &Value) {
        if let Some(map) = defs.as_object() {
            for (name, schema) in map {
                // A Python dict keeps a key's original position on re-assignment;
                // only record order the first time we see a key.
                if !self.definitions.contains_key(name) {
                    self.order.push(name.clone());
                }
                self.definitions.insert(name.clone(), schema.clone());
            }
        }
    }

    fn resolve_ref(&mut self, reference: &str) -> Option<Value> {
        if reference == "#" {
            self.has_self_ref = true;
            return Some(serde_json::json!({"$self_ref": true}));
        }
        if let Some(name) = reference.strip_prefix("#/$defs/") {
            match self.definitions.get(name).cloned() {
                Some(v) => return Some(v),
                // Upstream: `raise ValueError(f"Reference not found: {ref}")`.
                None => {
                    self.unsupported = true;
                    return None;
                }
            }
        }
        // Upstream: `raise ValueError(f"Unsupported reference format: {ref}")`.
        self.unsupported = true;
        None
    }
}

enum ParameterType {
    Scalar(ParameterTypeScalar),
    Object(ParameterTypeObject),
    Array(ParameterTypeArray),
    Enum(ParameterTypeEnum),
    AnyOf(ParameterTypeAnyOf),
    Union(ParameterTypeUnion),
    Ref(ParameterTypeRef),
}

impl ParameterType {
    fn format_docstring(&self, indent: &str) -> String {
        match self {
            ParameterType::Scalar(s) => s.base.format_docstring(indent),
            ParameterType::Object(o) => o.base.format_docstring(indent),
            ParameterType::Array(a) => a.base.format_docstring(indent),
            ParameterType::Enum(e) => e.base.format_docstring(indent),
            ParameterType::AnyOf(a) => a.base.format_docstring(indent),
            ParameterType::Union(u) => u.base.format_docstring(indent),
            ParameterType::Ref(r) => r.base.format_docstring(indent),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        match self {
            ParameterType::Scalar(s) => s.to_typescript(),
            ParameterType::Object(o) => o.to_typescript(indent, registry),
            ParameterType::Array(a) => a.to_typescript(indent, registry),
            ParameterType::Enum(e) => e.to_typescript(),
            ParameterType::AnyOf(a) => a.to_typescript(indent, registry),
            ParameterType::Union(u) => u.to_typescript(),
            ParameterType::Ref(r) => r.to_typescript(),
        }
    }
}

#[derive(Default)]
struct BaseType {
    description: String,
    constraints: Vec<(String, Value)>,
}

impl BaseType {
    fn from_extra_props(props: &Value, allowed_keys: &[&str]) -> Self {
        let description = props
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut constraints: Vec<(String, Value)> = props
            .as_object()
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| allowed_keys.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();
        constraints.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            description,
            constraints,
        }
    }

    fn format_docstring(&self, indent: &str) -> String {
        let mut out = String::new();
        if !self.description.is_empty() {
            out.push_str(&format_description(&self.description, indent));
            out.push('\n');
        }
        if !self.constraints.is_empty() {
            let parts: Vec<String> = self
                .constraints
                .iter()
                .map(|(k, v)| format!("{k}: {}", json_inline(v)))
                .collect();
            #[expect(
                clippy::unwrap_used,
                reason = "write!/writeln! into String cannot fail"
            )]
            {
                writeln!(out, "{indent}// {}", parts.join(", ")).unwrap();
            }
        }
        out
    }
}

struct ParameterTypeScalar {
    base: BaseType,
    typ: String,
}

impl ParameterTypeScalar {
    fn parse(typ: &str, props: &Value) -> Self {
        let allowed: &[&str] = match typ {
            "string" => &["maxLength", "minLength", "pattern"],
            "number" | "integer" => &["maximum", "minimum"],
            _ => &[],
        };
        Self {
            base: BaseType::from_extra_props(props, allowed),
            typ: typ.to_string(),
        }
    }

    fn any() -> Self {
        Self {
            base: BaseType::default(),
            typ: "any".to_string(),
        }
    }

    fn to_typescript(&self) -> String {
        match self.typ.as_str() {
            "integer" => "number".to_string(),
            other => other.to_string(),
        }
    }
}

struct Parameter {
    name: String,
    typ: ParameterType,
    optional: bool,
    default: Option<Value>,
}

impl Parameter {
    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut out = self.typ.format_docstring(indent);
        if let Some(d) = &self.default {
            let repr = match d {
                Value::Bool(true) => "True".to_string(),
                Value::Bool(false) => "False".to_string(),
                Value::Number(_) => d.to_string(),
                _ => serde_json::to_string(d).unwrap_or_else(|_| "null".to_string()),
            };
            #[expect(
                clippy::unwrap_used,
                reason = "write!/writeln! into String cannot fail"
            )]
            {
                writeln!(out, "{indent}// Default: {repr}").unwrap();
            }
        }
        let opt_marker = if self.optional { "?" } else { "" };
        #[expect(
            clippy::unwrap_used,
            reason = "write!/writeln! into String cannot fail"
        )]
        {
            write!(
                out,
                "{indent}{}{opt_marker}: {}",
                self.name,
                self.typ.to_typescript(indent, registry)
            )
            .unwrap();
        }
        out
    }
}

struct ParameterTypeObject {
    base: BaseType,
    properties: Vec<Parameter>,
    additional_properties: AdditionalProperties,
}

enum AdditionalProperties {
    None,
    True,
    False,
    Schema(Box<ParameterType>),
}

impl ParameterTypeObject {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let base = BaseType::from_extra_props(schema, &[]);
        if let Some(defs) = schema.get("$defs") {
            registry.register_definitions(defs);
        }

        let additional_properties = match schema.get("additionalProperties") {
            None => AdditionalProperties::None,
            Some(Value::Bool(true)) => AdditionalProperties::True,
            Some(Value::Bool(false)) => AdditionalProperties::False,
            Some(other) => {
                AdditionalProperties::Schema(Box::new(parse_parameter_type(other, registry)))
            }
        };

        let props_map = schema.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        let properties: Vec<Parameter> = props_map
            .map(|props| {
                props
                    .iter()
                    .map(|(name, prop_schema)| {
                        let optional = !required.contains(&name.as_str());
                        let default = prop_schema.get("default").cloned();
                        let typ = parse_parameter_type(prop_schema, registry);
                        Parameter {
                            name: name.clone(),
                            typ,
                            optional,
                            default,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            base,
            properties,
            additional_properties,
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut required: Vec<&Parameter> =
            self.properties.iter().filter(|p| !p.optional).collect();
        let mut optional: Vec<&Parameter> = self.properties.iter().filter(|p| p.optional).collect();
        required.sort_by(|a, b| a.name.cmp(&b.name));
        optional.sort_by(|a, b| a.name.cmp(&b.name));
        let inner_indent = format!("{indent}{TS_INDENT}");
        let mut parts: Vec<String> = required
            .into_iter()
            .chain(optional)
            .map(|p| p.to_typescript(&inner_indent, registry))
            .collect();

        match &self.additional_properties {
            AdditionalProperties::None => {}
            AdditionalProperties::True => parts.push(format!("{inner_indent}[k: string]: any")),
            AdditionalProperties::False => parts.push(format!("{inner_indent}[k: string]: never")),
            AdditionalProperties::Schema(inner) => {
                let ty = inner.to_typescript(&inner_indent, registry);
                parts.push(format!("{inner_indent}[k: string]: {ty}"));
            }
        }

        if parts.is_empty() {
            return "{}".to_string();
        }
        let body = parts.join(TS_FIELD_DELIMITER);
        let body = format!("\n{body}\n");
        format!("{{{body}{indent}}}")
    }
}

struct ParameterTypeArray {
    base: BaseType,
    item: Box<ParameterType>,
}

impl ParameterTypeArray {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let base = BaseType::from_extra_props(schema, &["minItems", "maxItems"]);
        let item = match schema.get("items") {
            Some(items) if !items.is_null() => parse_parameter_type(items, registry),
            _ => ParameterType::Scalar(ParameterTypeScalar::any()),
        };
        Self {
            base,
            item: Box::new(item),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let inner_indent = format!("{indent}{TS_INDENT}");
        let item_doc = self.item.format_docstring(&inner_indent);
        let item_ts = self.item.to_typescript(&inner_indent, registry);
        if item_doc.is_empty() {
            format!("Array<{item_ts}>")
        } else {
            format!("Array<\n{item_doc}{inner_indent}{item_ts}\n{indent}>")
        }
    }
}

struct ParameterTypeEnum {
    base: BaseType,
    values: Vec<Value>,
}

impl ParameterTypeEnum {
    fn parse(schema: &Value) -> Self {
        let values = schema
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Self {
            base: BaseType::from_extra_props(schema, &[]),
            values,
        }
    }

    fn to_typescript(&self) -> String {
        self.values
            .iter()
            .map(|v| match v {
                // Upstream emits `f'"{e}"'` with no escaping; match it byte-for-byte.
                Value::String(s) => format!("\"{s}\""),
                Value::Null => "None".to_string(),
                Value::Bool(true) => "True".to_string(),
                Value::Bool(false) => "False".to_string(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

struct ParameterTypeAnyOf {
    base: BaseType,
    branches: Vec<ParameterType>,
}

impl ParameterTypeAnyOf {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let branches = schema
            .get("anyOf")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|s| parse_parameter_type(s, registry))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            base: BaseType::from_extra_props(schema, &[]),
            branches,
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        self.branches
            .iter()
            .map(|b| b.to_typescript(indent, registry))
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

struct ParameterTypeUnion {
    base: BaseType,
    types: Vec<String>,
}

struct ParameterTypeRef {
    base: BaseType,
    ref_name: String,
}

impl ParameterTypeUnion {
    fn parse(schema: &Value) -> Self {
        let raw_types = schema
            .get("type")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let types = raw_types
            .into_iter()
            .map(|t| match t {
                "string" => "string".to_string(),
                "number" => "number".to_string(),
                "integer" => "number".to_string(),
                "boolean" => "boolean".to_string(),
                "null" => "null".to_string(),
                "object" => "{}".to_string(),
                "array" => "Array<any>".to_string(),
                other => other.to_string(),
            })
            .collect();
        Self {
            base: BaseType::from_extra_props(schema, &[]),
            types,
        }
    }

    fn to_typescript(&self) -> String {
        self.types.join(" | ")
    }
}

impl ParameterTypeRef {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let reference = schema.get("$ref").and_then(Value::as_str).unwrap_or("");
        let resolved = registry.resolve_ref(reference);
        let ref_name = match resolved {
            Some(ref v) if v.get("$self_ref").and_then(Value::as_bool) == Some(true) => {
                "parameters".to_string()
            }
            Some(_) => reference.rsplit('/').next().unwrap_or("").to_string(),
            None => "any".to_string(),
        };
        Self {
            base: BaseType::from_extra_props(schema, &[]),
            ref_name,
        }
    }

    fn to_typescript(&self) -> String {
        self.ref_name.clone()
    }
}

fn parse_parameter_type(schema: &Value, registry: &mut SchemaRegistry) -> ParameterType {
    if registry.depth >= MAX_RECURSION_DEPTH {
        return ParameterType::Scalar(ParameterTypeScalar::any());
    }
    registry.depth += 1;
    let result = parse_parameter_type_inner(schema, registry);
    registry.depth -= 1;
    result
}

fn parse_parameter_type_inner(schema: &Value, registry: &mut SchemaRegistry) -> ParameterType {
    if schema.is_boolean() {
        return ParameterType::Scalar(ParameterTypeScalar {
            base: BaseType::default(),
            typ: if schema.as_bool() == Some(true) {
                "any"
            } else {
                "null"
            }
            .to_string(),
        });
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        // A non-object, non-boolean schema is malformed; upstream raises.
        None => {
            registry.unsupported = true;
            return ParameterType::Scalar(ParameterTypeScalar::any());
        }
    };

    if obj.contains_key("$ref") {
        return ParameterType::Ref(ParameterTypeRef::parse(schema, registry));
    }

    if obj.contains_key("anyOf") {
        return ParameterType::AnyOf(ParameterTypeAnyOf::parse(schema, registry));
    }
    if obj.contains_key("enum") {
        return ParameterType::Enum(ParameterTypeEnum::parse(schema));
    }

    if let Some(typ_value) = obj.get("type") {
        if typ_value.is_array() {
            return ParameterType::Union(ParameterTypeUnion::parse(schema));
        }
        if let Some(typ) = typ_value.as_str() {
            return match typ {
                "object" => ParameterType::Object(ParameterTypeObject::parse(schema, registry)),
                "array" => ParameterType::Array(ParameterTypeArray::parse(schema, registry)),
                other => ParameterType::Scalar(ParameterTypeScalar::parse(other, schema)),
            };
        }
    }
    if obj.is_empty() {
        // Upstream maps the empty schema `{}` to `any`.
        return ParameterType::Scalar(ParameterTypeScalar::any());
    }
    // Fallthrough: a non-empty schema with no type/anyOf/enum/$ref. Upstream
    // raises `ValueError(f"Invalid JSON Schema object: ...")` here; flag it so
    // the caller drops the TS namespace and falls back to JSON.
    registry.unsupported = true;
    ParameterType::Scalar(ParameterTypeScalar::any())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_description(description: &str, indent: &str) -> String {
    description
        .split('\n')
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{indent}// {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn json_inline(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}
