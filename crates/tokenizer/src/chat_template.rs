//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use std::{collections::HashMap, fs, io};

use anyhow::{anyhow, Result};
use minijinja::{
    context,
    machinery::{
        ast::{Expr, Stmt},
        parse, WhitespaceConfig,
    },
    syntax::SyntaxConfig,
    value::Kwargs,
    Environment, Error as MinijinjaError, ErrorKind, Value,
};
use serde::Serialize;
use serde_json::{self, ser::Formatter, Value as JsonValue};

/// Chat template content format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format)
    OpenAI,
}

impl std::fmt::Display for ChatTemplateContentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::OpenAI => write!(f, "openai"),
        }
    }
}

/// Result of detecting the thinking/reasoning toggle in a chat template.
/// The variable name the template uses for the thinking toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingKeyName {
    /// Template uses `enable_thinking` (Qwen3, GLM, Nemotron)
    EnableThinking,
    /// Template uses `thinking` (DeepSeek V3.1, Kimi-K2.5)
    Thinking,
}

impl ThinkingKeyName {
    /// The template kwarg name this toggle uses.
    pub fn as_kwarg(self) -> &'static str {
        match self {
            ThinkingKeyName::EnableThinking => "enable_thinking",
            ThinkingKeyName::Thinking => "thinking",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingToggle {
    /// Template has no thinking toggle. The model either always reasons
    /// (e.g. DeepSeek R1) or never does — controlled by the parser's
    /// `always_in_reasoning` config.
    #[default]
    None,
    /// Template supports a thinking toggle that defaults to ON.
    /// If the user doesn't pass anything, thinking is enabled.
    /// (Qwen3, Qwen3.5, Nemotron, GLM-4.6, GLM-5, Kimi-K2.5)
    DefaultOn,
    /// Template supports a thinking toggle that defaults to OFF.
    /// Thinking only activates when the user explicitly passes `thinking=true`.
    /// (DeepSeek V3.1)
    DefaultOff,
}

/// Detect whether the chat template supports a thinking/reasoning toggle
/// and what its default value is.
pub fn detect_thinking_toggle(template: &str) -> (ThinkingToggle, Option<ThinkingKeyName>) {
    let has_enable_thinking = template.contains("enable_thinking");
    // Trailing space prevents matching "thinking_mode", "thinking_budget", etc.
    let has_thinking_var = template.contains("if thinking ")
        || template.contains("thinking is ")
        || template.contains("thinking ==")
        || template.contains("set thinking ");

    if !has_enable_thinking && !has_thinking_var {
        return (ThinkingToggle::None, None);
    }

    // At least one must be true — both false returned ThinkingToggle::None above.
    let key_name = if has_enable_thinking {
        ThinkingKeyName::EnableThinking
    } else {
        ThinkingKeyName::Thinking
    };

    // Check if the template explicitly defaults thinking to false/off.
    // DeepSeek V3.1 pattern: {% if not thinking is defined %}{% set thinking = false %}
    if template.contains("set thinking = false") || template.contains("set thinking=false") {
        return (ThinkingToggle::DefaultOff, Some(key_name));
    }
    if template.contains("set enable_thinking = false")
        || template.contains("set enable_thinking=false")
    {
        return (ThinkingToggle::DefaultOff, Some(key_name));
    }

    // All other models default to thinking ON
    (ThinkingToggle::DefaultOn, Some(key_name))
}

/// Detect the content format expected by a Jinja2 chat template
///
/// This implements the same detection logic as SGLang's detect_jinja_template_content_format
/// which uses AST parsing to look for content iteration patterns.
///
/// Returns:
/// - ChatTemplateContentFormat::OpenAI if template expects structured content (list of parts)
/// - ChatTemplateContentFormat::String if template expects simple string content
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    // Use AST-based detection (enabled by default)
    detect_all_with_ast(template).0
}

/// Flags tracking which OpenAI-style patterns we've seen
#[derive(Default, Debug, Clone, Copy)]
struct Flags {
    saw_iteration: bool,
    saw_structure: bool,
    saw_assignment: bool,
    saw_macro: bool,
}

impl Flags {
    fn any(self) -> bool {
        // `saw_assignment` alone (e.g. `set content = message.content`) is NOT sufficient
        // to classify as OpenAI format. Many string-format templates (Qwen3, etc.) use this
        // pattern to extract content into a local variable, then check `content is string`.
        // Without iteration or structural access, the template handles string content only.
        self.saw_iteration || self.saw_structure || self.saw_macro
    }
}

/// Single-pass AST detector with scope tracking
struct Detector<'a> {
    ast: &'a Stmt<'a>,
    /// Message loop vars currently in scope (e.g., `message`, `m`, `msg`)
    scope: std::collections::VecDeque<String>,
    scope_set: std::collections::HashSet<String>,
    flags: Flags,
    /// Whether `<think>` appears inside an `add_generation_prompt` if-block
    think_in_prefill: bool,
}

impl<'a> Detector<'a> {
    fn new(ast: &'a Stmt<'a>) -> Self {
        Self {
            ast,
            scope: std::collections::VecDeque::new(),
            scope_set: std::collections::HashSet::new(),
            flags: Flags::default(),
            think_in_prefill: false,
        }
    }

    fn run(mut self) -> (Flags, bool) {
        self.walk_stmt(self.ast);
        (self.flags, self.think_in_prefill)
    }

    fn push_scope(&mut self, var: String) {
        self.scope.push_back(var.clone());
        self.scope_set.insert(var);
    }

    fn pop_scope(&mut self) {
        if let Some(v) = self.scope.pop_back() {
            self.scope_set.remove(&v);
        }
    }

    fn is_var_access(expr: &Expr, varname: &str) -> bool {
        matches!(expr, Expr::Var(v) if v.id == varname)
    }

    fn is_const_str(expr: &Expr, value: &str) -> bool {
        matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
    }

    fn is_numeric_const(expr: &Expr) -> bool {
        matches!(expr, Expr::Const(c) if c.value.is_number())
    }

    /// Check if expr is varname.content or varname["content"]
    fn is_var_dot_content(expr: &Expr, varname: &str) -> bool {
        match expr {
            Expr::GetAttr(g) => Self::is_var_access(&g.expr, varname) && g.name == "content",
            Expr::GetItem(g) => {
                Self::is_var_access(&g.expr, varname)
                    && Self::is_const_str(&g.subscript_expr, "content")
            }
            // Unwrap filters/tests that just wrap the same expr
            Expr::Filter(f) => f
                .expr
                .as_ref()
                .is_some_and(|e| Self::is_var_dot_content(e, varname)),
            Expr::Test(t) => Self::is_var_dot_content(&t.expr, varname),
            _ => false,
        }
    }

    /// Check if expr accesses .content on any variable in our scope, or any descendant of it.
    fn is_any_scope_var_content(&self, expr: &Expr) -> bool {
        let mut current_expr = expr;
        loop {
            // Check if current level matches <scopeVar>.content
            if self
                .scope_set
                .iter()
                .any(|v| Self::is_var_dot_content(current_expr, v))
            {
                return true;
            }
            // Walk up the expression tree
            match current_expr {
                Expr::GetAttr(g) => current_expr = &g.expr,
                Expr::GetItem(g) => current_expr = &g.expr,
                _ => return false,
            }
        }
    }

    /// Check if an expression references a variable by name (walks through BinOp/UnaryOp).
    fn expr_references_var(expr: &Expr, name: &str) -> bool {
        match expr {
            Expr::Var(v) => v.id == name,
            Expr::BinOp(b) => {
                Self::expr_references_var(&b.left, name)
                    || Self::expr_references_var(&b.right, name)
            }
            Expr::UnaryOp(u) => Self::expr_references_var(&u.expr, name),
            _ => false,
        }
    }

    /// Check if a list of statements contains `<think>` in EmitRaw or string constants.
    fn body_has_think_tag(stmts: &[Stmt]) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::EmitRaw(raw) if raw.raw.contains("<think>") => return true,
                Stmt::EmitExpr(e) => {
                    if let Expr::Const(c) = &e.expr {
                        if c.value.as_str().is_some_and(|s| s.contains("<think>")) {
                            return true;
                        }
                    }
                }
                Stmt::IfCond(ic)
                    if Self::body_has_think_tag(&ic.true_body)
                        || Self::body_has_think_tag(&ic.false_body) =>
                {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Template(t) => {
                for ch in &t.children {
                    self.walk_stmt(ch);
                }
            }
            // {% for message in messages %}
            Stmt::ForLoop(fl) => {
                // Detect "for X in messages" → push X into scope
                if let Expr::Var(iter) = &fl.iter {
                    if iter.id == "messages" {
                        if let Expr::Var(target) = &fl.target {
                            self.push_scope(target.id.to_string());
                        }
                    }
                }

                // Also detect "for ... in message.content" or "for ... in content"
                // - Iterating directly over <scopeVar>.content => OpenAI style
                if self.is_any_scope_var_content(&fl.iter) {
                    self.flags.saw_iteration = true;
                }
                // - Iterating over a local var named "content"
                if matches!(&fl.iter, Expr::Var(v) if v.id == "content") {
                    self.flags.saw_iteration = true;
                }

                for b in &fl.body {
                    self.walk_stmt(b);
                }

                // Pop scope if we pushed it
                if let Expr::Var(iter) = &fl.iter {
                    if iter.id == "messages" && matches!(&fl.target, Expr::Var(_)) {
                        self.pop_scope();
                    }
                }
            }
            Stmt::IfCond(ic) => {
                self.inspect_expr_for_structure(&ic.expr);

                // Detect <think> inside {% if add_generation_prompt [and ...] %} body
                if !self.think_in_prefill
                    && Self::expr_references_var(&ic.expr, "add_generation_prompt")
                {
                    self.think_in_prefill = Self::body_has_think_tag(&ic.true_body);
                }

                for b in &ic.true_body {
                    self.walk_stmt(b);
                }
                for b in &ic.false_body {
                    self.walk_stmt(b);
                }
            }
            Stmt::EmitExpr(e) => {
                self.inspect_expr_for_structure(&e.expr);
            }
            // {% set content = message.content %}
            Stmt::Set(s)
                if Self::is_var_access(&s.target, "content")
                    && self.is_any_scope_var_content(&s.expr) =>
            {
                self.flags.saw_assignment = true;
            }
            Stmt::Macro(m) => {
                // Heuristic: macro that checks type (via `is` test) and also has any loop
                let mut has_type_check = false;
                let mut has_loop = false;
                Self::scan_macro_body(&m.body, &mut has_type_check, &mut has_loop);
                if has_type_check && has_loop {
                    self.flags.saw_macro = true;
                }
            }
            _ => {}
        }
    }

    fn inspect_expr_for_structure(&mut self, expr: &Expr) {
        if self.flags.saw_structure {
            return;
        }

        match expr {
            // content[0] or message.content[0]
            Expr::GetItem(gi)
                if (matches!(&gi.expr, Expr::Var(v) if v.id == "content")
                    || self.is_any_scope_var_content(&gi.expr))
                    && Self::is_numeric_const(&gi.subscript_expr) =>
            {
                self.flags.saw_structure = true;
            }
            // content|length or message.content|length
            Expr::Filter(f) => {
                if f.name == "length" {
                    if let Some(inner) = &f.expr {
                        // Box derefs automatically, so `&**inner` is `&Expr`
                        let inner_ref: &Expr = inner;
                        let is_content_var = matches!(inner_ref, Expr::Var(v) if v.id == "content");
                        if is_content_var || self.is_any_scope_var_content(inner_ref) {
                            self.flags.saw_structure = true;
                        }
                    }
                } else if let Some(inner) = &f.expr {
                    let inner_ref: &Expr = inner;
                    self.inspect_expr_for_structure(inner_ref);
                }
            }
            // Type tests like `content is iterable` or `message.content is string`
            // These are used for branching (e.g., Llama 3.1 uses them for tool output formatting),
            // not as indicators that the template expects structured content. Keep walking.
            Expr::Test(t) => self.inspect_expr_for_structure(&t.expr),
            Expr::GetAttr(g) => {
                // Keep walking; nested expressions can hide structure checks
                self.inspect_expr_for_structure(&g.expr);
            }
            // Handle binary operations like: if (message.content is string) and other_cond
            Expr::BinOp(op) => {
                self.inspect_expr_for_structure(&op.left);
                self.inspect_expr_for_structure(&op.right);
            }
            // Handle unary operations like: if not (message.content is string)
            Expr::UnaryOp(op) => {
                self.inspect_expr_for_structure(&op.expr);
            }
            _ => {}
        }
    }

    fn scan_macro_body(body: &[Stmt], has_type_check: &mut bool, has_loop: &mut bool) {
        for s in body {
            if *has_type_check && *has_loop {
                return;
            }

            match s {
                Stmt::IfCond(ic) => {
                    if matches!(&ic.expr, Expr::Test(_)) {
                        *has_type_check = true;
                    }
                    Self::scan_macro_body(&ic.true_body, has_type_check, has_loop);
                    Self::scan_macro_body(&ic.false_body, has_type_check, has_loop);
                }
                Stmt::ForLoop(fl) => {
                    *has_loop = true;
                    Self::scan_macro_body(&fl.body, has_type_check, has_loop);
                }
                Stmt::Template(t) => {
                    Self::scan_macro_body(&t.children, has_type_check, has_loop);
                }
                _ => {}
            }
        }
    }
}

/// Single-pass detection of content format, think-in-prefill, and thinking toggle.
fn detect_all(
    template: &str,
) -> (
    ChatTemplateContentFormat,
    bool,
    ThinkingToggle,
    Option<ThinkingKeyName>,
) {
    let (thinking_toggle, thinking_key_name) = detect_thinking_toggle(template);
    let (content_format, think_in_prefill) = detect_all_with_ast(template);
    (
        content_format,
        think_in_prefill,
        thinking_toggle,
        thinking_key_name,
    )
}

/// AST detection of content format and think-in-prefill.
fn detect_all_with_ast(template: &str) -> (ChatTemplateContentFormat, bool) {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return (ChatTemplateContentFormat::String, false),
    };

    let (flags, think_in_prefill) = Detector::new(&ast).run();
    let content_format = if flags.any() {
        ChatTemplateContentFormat::OpenAI
    } else {
        ChatTemplateContentFormat::String
    };
    (content_format, think_in_prefill)
}

/// Parameters for chat template application
#[derive(Default)]
pub struct ChatTemplateParams<'a> {
    pub add_generation_prompt: bool,
    pub tools: Option<&'a [serde_json::Value]>,
    pub documents: Option<&'a [serde_json::Value]>,
    pub template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
    /// Special tokens to inject into the template context.
    /// Many templates reference `{{ bos_token }}`, `{{ eos_token }}`, etc.
    pub special_tokens: Option<&'a crate::traits::SpecialTokens>,
    /// Resolved thinking preference. When `Some`, `apply` sets the template's
    /// own thinking-toggle key (`enable_thinking`/`thinking`, per detection) to
    /// this value as a default. An explicit `template_kwargs` entry for that
    /// key still wins.
    pub thinking: Option<bool>,
}

/// JSON separator pair passed through HuggingFace's `tojson` filter.
#[derive(Debug, Clone)]
struct JsonSeparators {
    item: Vec<u8>,
    key: Vec<u8>,
}

impl JsonSeparators {
    fn python_default(indent: Option<i64>) -> Self {
        // Python's json.dumps defaults to `(', ', ': ')` for compact output
        // and `(',', ': ')` when pretty indentation is enabled.
        let item = if indent.is_some() { "," } else { ", " };
        Self {
            item: item.as_bytes().to_vec(),
            key: b": ".to_vec(),
        }
    }
}

/// Formatter matching Python's `json.dumps` separator and ASCII escaping rules.
#[derive(Debug, Clone)]
struct PythonJsonFormatter {
    current_indent: usize,
    has_value: bool,
    indent: Option<Vec<u8>>,
    separators: JsonSeparators,
    ensure_ascii: bool,
}

impl PythonJsonFormatter {
    fn new(indent: Option<usize>, separators: JsonSeparators, ensure_ascii: bool) -> Self {
        Self {
            current_indent: 0,
            has_value: false,
            indent: indent.map(|spaces| vec![b' '; spaces]),
            separators,
            ensure_ascii,
        }
    }
}

fn write_indent<W>(writer: &mut W, count: usize, indent: &[u8]) -> io::Result<()>
where
    W: ?Sized + io::Write,
{
    for _ in 0..count {
        writer.write_all(indent)?;
    }
    Ok(())
}

fn write_u_escape<W>(writer: &mut W, code: u16) -> io::Result<()>
where
    W: ?Sized + io::Write,
{
    const HEX: &[u8; 16] = b"0123456789abcdef";
    writer.write_all(&[
        b'\\',
        b'u',
        HEX[((code >> 12) & 0xF) as usize],
        HEX[((code >> 8) & 0xF) as usize],
        HEX[((code >> 4) & 0xF) as usize],
        HEX[(code & 0xF) as usize],
    ])
}

impl Formatter for PythonJsonFormatter {
    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }

        for ch in fragment.chars() {
            if ch.is_ascii() {
                let mut buf = [0; 4];
                writer.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
                continue;
            }

            let code = ch as u32;
            if code <= 0xFFFF {
                write_u_escape(writer, code as u16)?;
            } else {
                let shifted = code - 0x1_0000;
                let high = 0xD800 + ((shifted >> 10) as u16);
                let low = 0xDC00 + ((shifted & 0x3FF) as u16);
                write_u_escape(writer, high)?;
                write_u_escape(writer, low)?;
            }
        }
        Ok(())
    }

    fn begin_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.indent.is_some() {
            self.current_indent += 1;
            self.has_value = false;
        }
        writer.write_all(b"[")
    }

    fn end_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = self.indent.as_deref() {
            self.current_indent -= 1;
            if self.has_value {
                writer.write_all(b"\n")?;
                write_indent(writer, self.current_indent, indent)?;
            }
        }
        writer.write_all(b"]")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = self.indent.as_deref() {
            if first {
                writer.write_all(b"\n")?;
            } else {
                writer.write_all(&self.separators.item)?;
                writer.write_all(b"\n")?;
            }
            write_indent(writer, self.current_indent, indent)
        } else if first {
            Ok(())
        } else {
            writer.write_all(&self.separators.item)
        }
    }

    fn end_array_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.indent.is_some() {
            self.current_indent += 1;
            self.has_value = false;
        }
        writer.write_all(b"{")
    }

    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = self.indent.as_deref() {
            self.current_indent -= 1;
            if self.has_value {
                writer.write_all(b"\n")?;
                write_indent(writer, self.current_indent, indent)?;
            }
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = self.indent.as_deref() {
            if first {
                writer.write_all(b"\n")?;
            } else {
                writer.write_all(&self.separators.item)?;
                writer.write_all(b"\n")?;
            }
            write_indent(writer, self.current_indent, indent)
        } else if first {
            Ok(())
        } else {
            writer.write_all(&self.separators.item)
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(&self.separators.key)
    }

    fn end_object_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }
}

fn invalid_tojson_option(message: impl Into<String>) -> MinijinjaError {
    MinijinjaError::new(ErrorKind::InvalidOperation, message.into())
}

fn parse_separators(
    separators: Option<Value>,
    indent: Option<i64>,
) -> std::result::Result<JsonSeparators, MinijinjaError> {
    let Some(separators) = separators else {
        return Ok(JsonSeparators::python_default(indent));
    };
    if separators.is_none() || separators.is_undefined() {
        return Ok(JsonSeparators::python_default(indent));
    }

    let parsed: serde_json::Value = serde_json::to_value(&separators).map_err(|e| {
        invalid_tojson_option(format!("Failed to convert separators to JSON value: {e}"))
    })?;
    let JsonValue::Array(values) = parsed else {
        return Err(invalid_tojson_option(
            "separators must be a two-item sequence",
        ));
    };
    if values.len() != 2 {
        return Err(invalid_tojson_option(
            "separators must be a two-item sequence",
        ));
    }

    let item = values[0]
        .as_str()
        .ok_or_else(|| invalid_tojson_option("item separator must be a string"))?;
    let key = values[1]
        .as_str()
        .ok_or_else(|| invalid_tojson_option("key separator must be a string"))?;

    Ok(JsonSeparators {
        item: item.as_bytes().to_vec(),
        key: key.as_bytes().to_vec(),
    })
}

fn serialize_with_python_json<T: Serialize>(
    value: &T,
    indent: Option<i64>,
    separators: JsonSeparators,
    ensure_ascii: bool,
) -> std::result::Result<String, MinijinjaError> {
    let indent = indent
        .map(|spaces| {
            if spaces < 0 {
                Err(invalid_tojson_option("indent cannot be negative"))
            } else {
                Ok(spaces as usize)
            }
        })
        .transpose()?;

    let formatter = PythonJsonFormatter::new(indent, separators, ensure_ascii);
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut serializer).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to serialize JSON: {e}"),
        )
    })?;
    String::from_utf8(buf).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Invalid UTF-8 in JSON output: {e}"),
        )
    })
}

/// Custom tojson filter compatible with HuggingFace transformers' implementation.
///
/// HuggingFace transformers registers a custom `tojson` filter that accepts additional
/// keyword arguments beyond what standard Jinja2 provides:
/// - `ensure_ascii` (bool): Whether to escape non-ASCII characters
/// - `indent` (int): Number of spaces for indentation (pretty-printing)
/// - `separators`: Custom item/key separators for JSON output
/// - `sort_keys` (bool): Whether to sort dictionary keys
///
/// This is necessary for compatibility with chat templates from HuggingFace Hub models.
/// See: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py
fn tojson_filter(value: Value, kwargs: Kwargs) -> std::result::Result<Value, MinijinjaError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<i64> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;

    // Ensure all kwargs are consumed to avoid "unknown keyword argument" errors
    kwargs.assert_all_used()?;

    let json_value: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to convert to JSON value: {e}"),
        )
    })?;

    // Serialize with options
    let json_str: std::result::Result<String, MinijinjaError> = {
        let sorted_json;
        let value_to_serialize = if sort_keys.unwrap_or(false) {
            sorted_json = sort_json_keys(&json_value);
            &sorted_json
        } else {
            &json_value
        };

        let separators = parse_separators(separators, indent)?;
        serialize_with_python_json(
            value_to_serialize,
            indent,
            separators,
            ensure_ascii.unwrap_or(false),
        )
    };

    json_str.map(Value::from_safe_string)
}

/// Recursively sort all object keys in a JSON value
fn sort_json_keys(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let mut sorted: serde_json::Map<String, JsonValue> = serde_json::Map::new();
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json_keys(&map[key]));
            }
            JsonValue::Object(sorted)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(sort_json_keys).collect()),
        _ => value.clone(),
    }
}

/// Hugging Face chat-template helper for surfacing model-authored validation
/// errors instead of a generic "unknown function" render failure.
fn raise_exception(message: String) -> std::result::Result<String, MinijinjaError> {
    Err(MinijinjaError::new(ErrorKind::InvalidOperation, message))
}

/// Build a pre-configured `Environment<'static>` with the given template string,
/// Python-compat method callback, and custom `tojson` filter already registered.
/// The template is stored under the name `"chat"` using owned storage so the
/// environment carries no borrows.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    // Match HuggingFace's Jinja2 defaults: trim_blocks and lstrip_blocks are
    // enabled in Python's transformers but default to false in minijinja.
    // Without these, templates like GLM-5's produce incorrect whitespace.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // Register the template with owned storage (no lifetime dependency on caller)
    env.add_template_owned("chat".to_owned(), template)
        .map_err(|e| anyhow!("Failed to add template: {e}"))?;

    // Enable Python method compatibility (e.g., str.startswith, str.endswith)
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Register custom tojson filter compatible with HuggingFace transformers
    // This overrides minijinja's built-in tojson to support additional kwargs
    // like ensure_ascii, separators, and sort_keys that HuggingFace templates use
    env.add_filter("tojson", tojson_filter);
    env.add_function("raise_exception", raise_exception);

    Ok(env)
}

/// Render the `"chat"` template in the given environment against messages and params.
/// Convert an optional token string to a minijinja Value.
/// Present tokens become strings; absent tokens become UNDEFINED
/// so templates can use `{% if bos_token is defined %}` guards.
fn special_token_value(token: Option<&str>) -> Value {
    token.map_or(Value::UNDEFINED, Value::from)
}

fn render_chat_template(
    env: &Environment<'_>,
    messages: &[serde_json::Value],
    params: ChatTemplateParams,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("Failed to get template: {e}"))?;

    // Convert messages to minijinja::Value (messages already processed by router)
    let minijinja_messages: Vec<Value> = messages.iter().map(Value::from_serialize).collect();

    // Use Value::UNDEFINED for missing optional params so they are truly "undefined"
    // in the template context, matching HuggingFace Python behavior. Many chat templates
    // use `{% if tools is defined %}` guards — passing null (none) instead of undefined
    // would bypass those guards since `none` IS defined, causing `tools | length` to fail.
    let tools_value = params.tools.map_or(Value::UNDEFINED, Value::from_serialize);
    let documents_value = params
        .documents
        .map_or(Value::UNDEFINED, Value::from_serialize);

    // Inject special tokens (bos_token, eos_token, etc.) into context.
    // Use UNDEFINED for missing tokens so `{% if bos_token is defined %}` works correctly.
    // This matches HuggingFace Python which passes self.special_tokens_map to the renderer.
    let bos_value =
        special_token_value(params.special_tokens.and_then(|st| st.bos_token.as_deref()));
    let eos_value =
        special_token_value(params.special_tokens.and_then(|st| st.eos_token.as_deref()));
    let unk_value =
        special_token_value(params.special_tokens.and_then(|st| st.unk_token.as_deref()));
    let pad_value =
        special_token_value(params.special_tokens.and_then(|st| st.pad_token.as_deref()));

    let base_context = context! {
        messages => &minijinja_messages,
        add_generation_prompt => params.add_generation_prompt,
        tools => tools_value,
        documents => documents_value,
        bos_token => bos_value,
        eos_token => eos_value,
        unk_token => unk_value,
        pad_token => pad_value,
    };

    // Merge with template_kwargs if provided (caller kwargs override special tokens)
    let ctx = if let Some(kwargs) = params.template_kwargs {
        context! {
            ..base_context,
            ..Value::from_serialize(kwargs)
        }
    } else {
        base_context
    };

    // Render the template
    let rendered = tmpl
        .render(&ctx)
        .map_err(|e| anyhow!("Failed to render template: {e}"))?;

    Ok(rendered)
}

/// Chat template processor using Jinja2 - simple wrapper like HuggingFace
pub struct ChatTemplateProcessor {
    env: Environment<'static>,
}

impl ChatTemplateProcessor {
    /// Create a new chat template processor.
    ///
    /// Returns an error if the template fails to parse, so callers get an
    /// actionable message immediately rather than a confusing "template not
    /// found" error on the first render.
    pub fn new(template: String) -> Result<Self> {
        let env = build_environment(template)?;
        Ok(ChatTemplateProcessor { env })
    }

    /// Apply the chat template to a list of messages
    ///
    /// This mimics the behavior of HuggingFace's apply_chat_template method
    /// but returns the formatted string instead of token IDs.
    /// Messages should be pre-processed into the format expected by the template.
    pub fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        render_chat_template(&self.env, messages, params)
    }
}

/// Load chat template from tokenizer config JSON
pub fn load_chat_template_from_config(config_path: &str) -> Result<Option<String>> {
    let content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;

    // Look for chat_template in the config
    if let Some(template) = config.get("chat_template") {
        if let Some(template_str) = template.as_str() {
            return Ok(Some(template_str.to_string()));
        }
    }

    Ok(None)
}

/// Load chat template from a file (.jinja or .json containing Jinja).
/// Shared between all tokenizer backends.
pub fn load_chat_template_from_file(template_path: &str) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path)
        .map_err(|e| anyhow!("Failed to read chat template file: {e}"))?;

    if template_path.ends_with(".json") {
        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse chat_template.json: {e}"))?;

        if let Some(template_str) = json_value.as_str() {
            return Ok(Some(template_str.to_string()));
        } else if let Some(obj) = json_value.as_object() {
            if let Some(template_value) = obj.get("chat_template") {
                if let Some(template_str) = template_value.as_str() {
                    return Ok(Some(template_str.to_string()));
                }
            }
        }

        return Err(anyhow!(
            "chat_template.json does not contain a valid template",
        ));
    }

    // Plain .jinja file
    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// Chat template state that can be embedded in any tokenizer struct.
/// Eliminates duplicated apply/set/format methods across tokenizer backends.
///
/// The compiled `minijinja::Environment` (with the template parsed, filters
/// registered, and Python-compat callback installed) is cached so that
/// `apply()` only performs rendering -- no parsing or environment setup.
/// The cache is rebuilt whenever `set()` is called.
///
/// `Environment<'static>` is both `Send` and `Sync`, so embedding this in
/// tokenizer structs shared across threads is safe.
pub struct ChatTemplateState {
    /// Cached, fully-configured environment. `None` when no template is set.
    env: Option<Environment<'static>>,
    content_format: ChatTemplateContentFormat,
    /// Thinking toggle support detected from the template.
    thinking_toggle: ThinkingToggle,
    /// The variable name used for the thinking toggle (if any).
    thinking_key_name: Option<ThinkingKeyName>,
    /// Whether the template injects `<think>` in the generation prompt.
    think_in_prefill: bool,
}

impl std::fmt::Debug for ChatTemplateState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplateState")
            .field("has_template", &self.env.is_some())
            .field("content_format", &self.content_format)
            .field("thinking_toggle", &self.thinking_toggle)
            .field("think_in_prefill", &self.think_in_prefill)
            .finish()
    }
}

impl ChatTemplateState {
    pub fn new(template: Option<String>) -> Result<Self> {
        let (content_format, think_in_prefill, thinking_toggle, thinking_key_name) =
            template.as_ref().map(|t| detect_all(t)).unwrap_or_default();
        let env = template.map(build_environment).transpose()?;
        Ok(Self {
            env,
            content_format,
            thinking_toggle,
            thinking_key_name,
            think_in_prefill,
        })
    }

    /// Create a `ChatTemplateState` with no template set.
    ///
    /// Unlike `new(None)`, this is infallible since there is no template to
    /// parse — useful in constructors that don't return `Result`.
    pub fn empty() -> Self {
        Self {
            env: None,
            content_format: ChatTemplateContentFormat::default(),
            thinking_toggle: ThinkingToggle::None,
            thinking_key_name: None,
            think_in_prefill: false,
        }
    }

    pub fn apply(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        let env = self.env.as_ref().ok_or_else(|| {
            anyhow!(
                "Cannot use chat template functions because tokenizer.chat_template is not set \
                 and no template argument was passed! For information about writing templates and \
                 setting the tokenizer.chat_template attribute, please see the documentation at \
                 https://huggingface.co/docs/transformers/main/en/chat_templating",
            )
        })?;

        // Apply the resolved thinking preference under the template's own toggle
        // key (`enable_thinking` vs `thinking`, per detection). Skip entirely
        // (no clone) when the caller already set that key explicitly — the
        // explicit value wins.
        if let (Some(thinking), Some(key)) = (params.thinking, self.thinking_key_name) {
            let kwarg_key = key.as_kwarg();
            if params
                .template_kwargs
                .is_none_or(|k| !k.contains_key(kwarg_key))
            {
                let mut kwargs = params.template_kwargs.cloned().unwrap_or_default();
                kwargs.insert(kwarg_key.to_string(), serde_json::Value::Bool(thinking));
                let params = ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    thinking: None,
                    ..params
                };
                return render_chat_template(env, messages, params);
            }
        }

        render_chat_template(env, messages, params)
    }

    pub fn set(&mut self, template: String) -> Result<()> {
        let (content_format, think_in_prefill, thinking_toggle, thinking_key_name) =
            detect_all(&template);
        let env = build_environment(template)?;
        self.content_format = content_format;
        self.thinking_toggle = thinking_toggle;
        self.thinking_key_name = thinking_key_name;
        self.think_in_prefill = think_in_prefill;
        self.env = Some(env);
        Ok(())
    }

    pub fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }

    pub fn thinking_toggle(&self) -> ThinkingToggle {
        self.thinking_toggle
    }

    pub fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.thinking_key_name
    }

    pub fn think_in_prefill(&self) -> bool {
        self.think_in_prefill
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_state_no_template() {
        let state = ChatTemplateState::new(None).unwrap();
        assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
        let result = state.apply(&[], ChatTemplateParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_chat_template_state_set() {
        let mut state = ChatTemplateState::new(None).unwrap();
        state.set("{{ messages }}".to_string()).unwrap();
        assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
    }

    #[test]
    fn test_chat_template_state_invalid_template() {
        let result = ChatTemplateState::new(Some("{% invalid".to_string()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Failed to add template"),
            "Error should explain parse failure, got: {err}"
        );
    }

    #[test]
    fn test_chat_template_processor_invalid_template() {
        let result = ChatTemplateProcessor::new("{% invalid".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_raise_exception_surfaces_template_validation_message() {
        let state = ChatTemplateState::new(Some(
            "{{ raise_exception('reasoning_effort is invalid') }}".to_string(),
        ))
        .unwrap();

        let error = state
            .apply(&[], ChatTemplateParams::default())
            .unwrap_err()
            .to_string();

        assert!(error.contains("reasoning_effort is invalid"), "{error}");
    }

    #[test]
    fn test_special_tokens_injected_into_context() {
        let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let special_tokens = crate::traits::SpecialTokens {
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            ..Default::default()
        };

        let result = state
            .apply(
                &messages,
                ChatTemplateParams {
                    special_tokens: Some(&special_tokens),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result, "<s>hello</s>");
    }

    #[test]
    fn test_special_tokens_undefined_when_not_provided() {
        let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

        let result = state.apply(&[], ChatTemplateParams::default()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_special_tokens_partial() {
        let template =
            "{{ bos_token }}hello{% if eos_token is defined %}{{ eos_token }}{% endif %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();

        let special_tokens = crate::traits::SpecialTokens {
            bos_token: Some("<s>".to_string()),
            eos_token: None,
            ..Default::default()
        };

        let result = state
            .apply(
                &[],
                ChatTemplateParams {
                    special_tokens: Some(&special_tokens),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result, "<s>hello");
    }

    #[test]
    fn thinking_param_sets_template_key_and_explicit_wins() {
        use std::collections::HashMap;

        // Template echoes the enable_thinking value so we can observe what was set.
        let state = ChatTemplateState::new(Some("{{ enable_thinking }}".to_string())).unwrap();
        assert_eq!(
            state.thinking_key_name(),
            Some(ThinkingKeyName::EnableThinking)
        );

        // thinking = Some(false) injects enable_thinking=false under the model's key.
        // Rendered Python-style: transformers evaluates these templates under
        // Jinja2, where `{{ False }}` is "False", so a template that echoes the
        // flag into the prompt must produce the same bytes we would there.
        let out = state
            .apply(
                &[],
                ChatTemplateParams {
                    thinking: Some(false),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out, "False");

        // An explicit template_kwargs entry overrides the injected default.
        let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        let out = state
            .apply(
                &[],
                ChatTemplateParams {
                    thinking: Some(false),
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out, "True");
    }

    /// Regression: a conditional expression used as a keyword-argument value.
    /// Jinja2 accepts it, so published chat templates use it — Muse-Glimmer's
    /// does, in a `namespace(...)` call — but minijinja rejected it before
    /// 2.24, which made the whole template uncompilable and left the model
    /// unservable rather than merely mis-rendered.
    #[test]
    fn conditional_expression_in_keyword_argument_compiles() {
        let template = "{%- set r = namespace(name=x if x else '') -%}{{ r.name }}";
        let state = ChatTemplateState::new(Some(template.to_string()))
            .expect("conditional kwargs must compile");
        let out = state.apply(&[], ChatTemplateParams::default()).unwrap();
        assert_eq!(out, "");
    }

    /// None renders Python-style too, for the same Jinja2-parity reason.
    #[test]
    fn none_renders_python_style() {
        let state = ChatTemplateState::new(Some("{{ undefined_value }}".to_string())).unwrap();
        let out = state.apply(&[], ChatTemplateParams::default()).unwrap();
        assert_eq!(out, "");
    }
}
