//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use std::{borrow::Cow, collections::HashMap, fs, io, sync::OnceLock};

use anyhow::{anyhow, Result};
use chrono::{
    format::{strftime::StrftimeItems, Item},
    DateTime, FixedOffset, Local, TimeZone,
};
use minijinja::{
    context,
    machinery::{
        ast::{Call, CallArg, Expr, ForLoop, Macro, Set, Stmt},
        parse, WhitespaceConfig,
    },
    syntax::SyntaxConfig,
    value::{Kwargs, ValueKind},
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
    /// Template uses the tri-state `thinking_mode` string (MiniMax M3):
    /// "enabled" prefills the think-start token, "disabled" prefills the
    /// think-end token, "adaptive"/absent adds no prefix.
    ThinkingMode,
    /// Template switches thinking through the `reasoning_effort` kwarg with
    /// its own on/off words (Hy4: `high` / `no_think`); see
    /// [`REASONING_EFFORT_ON_VALUES`] and [`REASONING_EFFORT_OFF_VALUES`].
    ReasoningEffort,
}

/// `reasoning_effort` values that switch a [`ThinkingKeyName::ReasoningEffort`]
/// template into thinking mode.
pub const REASONING_EFFORT_ON_VALUES: &[&str] = &["high"];
/// `reasoning_effort` values that switch it off.
pub const REASONING_EFFORT_OFF_VALUES: &[&str] = &["no_think"];

impl ThinkingKeyName {
    /// The template kwarg name this toggle uses.
    pub fn as_kwarg(self) -> &'static str {
        match self {
            ThinkingKeyName::EnableThinking => "enable_thinking",
            ThinkingKeyName::Thinking => "thinking",
            ThinkingKeyName::ThinkingMode => "thinking_mode",
            ThinkingKeyName::ReasoningEffort => "reasoning_effort",
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
    if template.contains("think_begin_token")
        && template.contains("no_think")
        && template.contains("reasoning_effort")
    {
        return (
            ThinkingToggle::DefaultOn,
            Some(ThinkingKeyName::ReasoningEffort),
        );
    }
    // Tri-state string toggle, detected only when the template actually
    // branches on the variable: only `thinking_mode == "enabled"` prefills
    // the think-start token, so the toggle defaults OFF.
    if template.contains("thinking_mode ==")
        || template.contains("thinking_mode is defined")
        || template.contains("if thinking_mode")
    {
        return (
            ThinkingToggle::DefaultOff,
            Some(ThinkingKeyName::ThinkingMode),
        );
    }

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
/// The rule is the serving engine's own content-format detection, so that
/// the gateway hands string content as a one-item text part list to the
/// same templates as the engine does (see [`ChatTemplateState::apply`]). A
/// template is of the "openai" format when it has a loop over a message's
/// content, a message being a loop variable over `messages` or over a
/// variable assigned from it: the loop runs over `message.content` (also
/// through a filter, a test or a slice), over a macro parameter the template
/// fills with a message's content, or, outside any macro, over a variable
/// named `content`. Any other template is of the "string" format, one that
/// does not parse included; a type test, a `length` filter or an index on the
/// content does not make the "openai" format, as it does not on the engine.
///
/// Returns:
/// - ChatTemplateContentFormat::OpenAI if template expects structured content (list of parts)
/// - ChatTemplateContentFormat::String if template expects simple string content
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    detect_all_with_ast(template).0
}

/// A loop or assignment target that is not a plain name where the engine's
/// walk asserts one; the engine then falls back to its default, the "string"
/// format.
struct NotAName;

/// The nodes of a template the engine's rule reads, in document order: each
/// loop with the macros it sits in (outermost first), the assignments, the
/// macros and the calls.
#[derive(Default)]
struct Nodes<'a> {
    loops: Vec<(&'a ForLoop<'a>, Vec<&'a Macro<'a>>)>,
    sets: Vec<&'a Set<'a>>,
    macros: Vec<&'a Macro<'a>>,
    calls: Vec<&'a Call<'a>>,
}

impl<'a> Nodes<'a> {
    fn of(ast: &'a Stmt<'a>) -> Self {
        let mut nodes = Self::default();
        nodes.stmt(ast, &mut Vec::new());
        nodes
    }

    fn stmts(&mut self, stmts: &'a [Stmt<'a>], macros: &mut Vec<&'a Macro<'a>>) {
        for stmt in stmts {
            self.stmt(stmt, macros);
        }
    }

    fn stmt(&mut self, stmt: &'a Stmt<'a>, macros: &mut Vec<&'a Macro<'a>>) {
        match stmt {
            Stmt::Template(t) => self.stmts(&t.children, macros),
            Stmt::EmitExpr(e) => self.expr(&e.expr),
            Stmt::EmitRaw(_) | Stmt::Continue(_) | Stmt::Break(_) => {}
            Stmt::ForLoop(fl) => {
                self.loops.push((fl, macros.clone()));
                self.expr(&fl.target);
                self.expr(&fl.iter);
                self.stmts(&fl.body, macros);
                self.stmts(&fl.else_body, macros);
                if let Some(filter) = &fl.filter_expr {
                    self.expr(filter);
                }
            }
            Stmt::IfCond(ic) => {
                self.expr(&ic.expr);
                self.stmts(&ic.true_body, macros);
                self.stmts(&ic.false_body, macros);
            }
            Stmt::WithBlock(w) => {
                for (target, value) in &w.assignments {
                    self.expr(target);
                    self.expr(value);
                }
                self.stmts(&w.body, macros);
            }
            Stmt::Set(s) => {
                self.sets.push(s);
                self.expr(&s.target);
                self.expr(&s.expr);
            }
            Stmt::SetBlock(s) => {
                self.expr(&s.target);
                if let Some(filter) = &s.filter {
                    self.expr(filter);
                }
                self.stmts(&s.body, macros);
            }
            Stmt::AutoEscape(a) => {
                self.expr(&a.enabled);
                self.stmts(&a.body, macros);
            }
            Stmt::FilterBlock(f) => {
                self.expr(&f.filter);
                self.stmts(&f.body, macros);
            }
            Stmt::Block(b) => self.stmts(&b.body, macros),
            Stmt::Import(i) => {
                self.expr(&i.expr);
                self.expr(&i.name);
            }
            Stmt::FromImport(f) => {
                self.expr(&f.expr);
                for (name, alias) in &f.names {
                    self.expr(name);
                    if let Some(alias) = alias {
                        self.expr(alias);
                    }
                }
            }
            Stmt::Extends(e) => self.expr(&e.name),
            Stmt::Include(i) => self.expr(&i.name),
            Stmt::Macro(m) => {
                self.macros.push(m);
                for default in &m.defaults {
                    self.expr(default);
                }
                macros.push(m);
                self.stmts(&m.body, macros);
                macros.pop();
            }
            // A `{% call %}` block is a call with a body, not a macro of the
            // template (jinja2 keeps the two apart).
            Stmt::CallBlock(cb) => {
                self.call(&cb.call);
                self.stmts(&cb.macro_decl.body, macros);
            }
            Stmt::Do(d) => self.call(&d.call),
        }
    }

    fn call(&mut self, call: &'a Call<'a>) {
        self.calls.push(call);
        self.expr(&call.expr);
        self.args(&call.args);
    }

    fn args(&mut self, args: &'a [CallArg<'a>]) {
        for arg in args {
            match arg {
                CallArg::Pos(e)
                | CallArg::Kwarg(_, e)
                | CallArg::PosSplat(e)
                | CallArg::KwargSplat(e) => {
                    self.expr(e);
                }
            }
        }
    }

    fn expr(&mut self, expr: &'a Expr<'a>) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Slice(s) => {
                self.expr(&s.expr);
                for bound in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                    self.expr(bound);
                }
            }
            Expr::UnaryOp(u) => self.expr(&u.expr),
            Expr::BinOp(b) => {
                self.expr(&b.left);
                self.expr(&b.right);
            }
            Expr::Compare(c) => {
                self.expr(&c.expr);
                for op in &c.ops {
                    self.expr(&op.expr);
                }
            }
            Expr::IfExpr(i) => {
                self.expr(&i.test_expr);
                self.expr(&i.true_expr);
                if let Some(e) = &i.false_expr {
                    self.expr(e);
                }
            }
            Expr::Filter(f) => {
                if let Some(e) = &f.expr {
                    self.expr(e);
                }
                self.args(&f.args);
            }
            Expr::Test(t) => {
                self.expr(&t.expr);
                self.args(&t.args);
            }
            Expr::GetAttr(g) => self.expr(&g.expr),
            Expr::GetItem(g) => {
                self.expr(&g.expr);
                self.expr(&g.subscript_expr);
            }
            Expr::Call(c) => self.call(c),
            Expr::List(l) => {
                for item in &l.items {
                    self.expr(item);
                }
            }
            Expr::Map(m) => {
                for e in m.keys.iter().chain(&m.values) {
                    self.expr(e);
                }
            }
        }
    }
}

/// Whether `expr` reads the variable `varname` itself (`key` None) or its
/// `key` entry (`varname.key`, `varname['key']`), through filters, tests and
/// slices: the engine's `_is_var_or_elems_access`.
fn is_var_or_elems_access(expr: &Expr<'_>, varname: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => f
            .expr
            .as_ref()
            .is_some_and(|e| is_var_or_elems_access(e, varname, key)),
        Expr::Test(t) => is_var_or_elems_access(&t.expr, varname, key),
        Expr::Slice(s) => is_var_or_elems_access(&s.expr, varname, key),
        Expr::Var(v) => key.is_none() && v.id == varname,
        // The base of the attribute is the name itself, as in the engine's
        // `_is_attr_access`; the wrappers above are allowed around the whole
        // access only.
        Expr::GetAttr(g) => key.is_some_and(|key| g.name == key && is_var(&g.expr, varname)),
        Expr::GetItem(g) => key.is_some_and(|key| {
            matches!(&g.subscript_expr, Expr::Const(c) if c.value.as_str() == Some(key))
                && is_var(&g.expr, varname)
        }),
        _ => false,
    }
}

/// Whether `expr` is the plain name `varname`: the engine's `_is_var_access`.
fn is_var(expr: &Expr<'_>, varname: &str) -> bool {
    matches!(expr, Expr::Var(v) if v.id == varname)
}

/// `root` and every name assigned from it or from such a name, through
/// filters, tests and slices: the engine's `_iter_nodes_assign_var_or_elems`.
fn names_assigned_from<'a>(sets: &[&'a Set<'a>], root: &'a str) -> Result<Vec<&'a str>, NotAName> {
    let mut names = vec![root];
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(related) = queue.pop_front() {
        for set in sets {
            if !is_var_or_elems_access(&set.expr, related, None) {
                continue;
            }
            let Expr::Var(target) = &set.target else {
                return Err(NotAName);
            };
            if !names.contains(&target.id) {
                names.push(target.id);
                queue.push_back(target.id);
            }
        }
    }
    Ok(names)
}

/// The loop variables of the loops over a messages name: the engine's
/// `_iter_nodes_assign_messages_item`.
fn message_loop_vars<'a>(
    loops: &[(&'a ForLoop<'a>, Vec<&'a Macro<'a>>)],
    messages_names: &[&str],
) -> Result<Vec<&'a str>, NotAName> {
    let mut vars = Vec::new();
    for (fl, _) in loops {
        if messages_names
            .iter()
            .any(|name| is_var_or_elems_access(&fl.iter, name, None))
        {
            let Expr::Var(target) = &fl.target else {
                return Err(NotAName);
            };
            vars.push(target.id);
        }
    }
    Ok(vars)
}

/// Whether the template has a loop over a message's content: the engine's
/// `_iter_nodes_assign_content_item`, which decides its content format.
fn iterates_message_content(ast: &Stmt<'_>) -> Result<bool, NotAName> {
    let nodes = Nodes::of(ast);
    let messages_names = names_assigned_from(&nodes.sets, "messages")?;
    let message_vars = message_loop_vars(&nodes.loops, &messages_names)?;
    let is_message_content = |expr: &Expr<'_>| {
        message_vars
            .iter()
            .any(|var| is_var_or_elems_access(expr, var, Some("content")))
    };

    // The parameters of each macro that a call fills with a message's content.
    let fed_params: Vec<(&Macro<'_>, std::collections::HashSet<&str>)> = nodes
        .macros
        .iter()
        .map(|m| {
            let param = |index: usize| match m.args.get(index) {
                Some(Expr::Var(v)) => Some(v.id),
                _ => None,
            };
            let mut fed = std::collections::HashSet::new();
            for call in nodes
                .calls
                .iter()
                .filter(|call| matches!(&call.expr, Expr::Var(callee) if callee.id == m.name))
            {
                let mut position = 0;
                for arg in &call.args {
                    match arg {
                        CallArg::Pos(value) => {
                            if let Some(name) =
                                param(position).filter(|_| is_message_content(value))
                            {
                                fed.insert(name);
                            }
                            position += 1;
                        }
                        CallArg::Kwarg(name, value) => {
                            if (0..m.args.len()).any(|i| param(i) == Some(name))
                                && is_message_content(value)
                            {
                                fed.insert(*name);
                            }
                        }
                        CallArg::PosSplat(_) | CallArg::KwargSplat(_) => {}
                    }
                }
            }
            (*m, fed)
        })
        .collect();

    for (fl, enclosing) in &nodes.loops {
        let matched = if is_message_content(&fl.iter) {
            true
        } else if let Expr::Var(iter) = &fl.iter {
            // Inside a macro, the innermost macro with fed parameters decides;
            // outside any macro, a variable named `content` does.
            let fed = enclosing.iter().rev().find_map(|m| {
                fed_params
                    .iter()
                    .find(|(fed_macro, fed)| std::ptr::eq(*fed_macro, *m) && !fed.is_empty())
                    .map(|(_, fed)| fed)
            });
            match fed {
                Some(fed) => fed.contains(iter.id),
                None => enclosing.is_empty() && iter.id == "content",
            }
        } else {
            false
        };
        if matched {
            if !matches!(fl.target, Expr::Var(_)) {
                return Err(NotAName);
            }
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether `<think>` appears inside an `add_generation_prompt` if-block.
struct ThinkDetector {
    think_in_prefill: bool,
}

impl ThinkDetector {
    fn run(ast: &Stmt<'_>) -> bool {
        let mut detector = Self {
            think_in_prefill: false,
        };
        detector.walk_stmt(ast);
        detector.think_in_prefill
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
            Stmt::ForLoop(fl) => {
                for b in &fl.body {
                    self.walk_stmt(b);
                }
            }
            Stmt::IfCond(ic) => {
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
            _ => {}
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
    let template = plain_generation_blocks(template);
    let ast = match parse(
        &template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return (ChatTemplateContentFormat::String, false),
    };

    let content_format = match iterates_message_content(&ast) {
        Ok(true) => ChatTemplateContentFormat::OpenAI,
        Ok(false) | Err(NotAName) => ChatTemplateContentFormat::String,
    };
    (content_format, ThinkDetector::run(&ast))
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
    /// The instant the template's `strftime_now` writes, for a render that
    /// must come out the same each time; `None` is the clock (see
    /// [`render_instant`]).
    pub now: Option<DateTime<FixedOffset>>,
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

/// Transformers' `{% generation %}` ... `{% endgeneration %}` block marks the
/// assistant tokens for a training mask and renders its body unchanged; the
/// engine here has no such statement and would refuse the whole template. The
/// block becomes `{% if true %}` ... `{% endif %}`: the same body, and the same
/// whitespace handling, as both are block tags (`trim_blocks`, `lstrip_blocks`,
/// and each tag's own `-` or `+` modifier, which is kept as written).
fn plain_generation_blocks(template: &str) -> Cow<'_, str> {
    if !template.contains("generation") {
        return Cow::Borrowed(template);
    }
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        match generation_tag(rest) {
            Some((len, replacement)) => {
                out.push_str(&replacement);
                rest = &rest[len..];
            }
            None => {
                out.push_str("{%");
                rest = &rest[2..];
            }
        }
    }
    out.push_str(rest);
    if out == template {
        Cow::Borrowed(template)
    } else {
        Cow::Owned(out)
    }
}

/// When `tag`, which starts with `{%`, is a `generation` or `endgeneration`
/// block tag: its length, and the `if true` or `endif` tag that stands in for
/// it with the same whitespace modifiers.
fn generation_tag(tag: &str) -> Option<(usize, String)> {
    fn modifier(body: &str) -> (&str, &str) {
        match body.as_bytes().first() {
            Some(b'-') => ("-", &body[1..]),
            Some(b'+') => ("+", &body[1..]),
            _ => ("", body),
        }
    }
    let (left, body) = modifier(tag.strip_prefix("{%")?);
    let body = body.trim_start();
    let (word, body) = ["endgeneration", "generation"]
        .iter()
        .find_map(|word| body.strip_prefix(word).map(|body| (*word, body)))?;
    let (right, body) = modifier(body.trim_start());
    let body = body.strip_prefix("%}")?;
    let statement = if word == "generation" {
        "if true"
    } else {
        "endif"
    };
    Some((
        tag.len() - body.len(),
        format!("{{%{left} {statement} {right}%}}"),
    ))
}

/// Jinja2's `dict()` is Python's: it builds a map from nothing, from a
/// mapping or from an iterable of key/value pairs, keyword arguments are
/// written over the result, and a repeated key keeps its first position with
/// its last value. Templates merge a schema's `$defs` entry into the property
/// that references it as `dict((defs | items | list) + (spec | items | list))`;
/// minijinja's own `dict` takes a mapping only and answers the pair list with
/// a bare "invalid operation".
fn dict_function(
    value: Option<Value>,
    kwargs: Kwargs,
) -> std::result::Result<Value, MinijinjaError> {
    let mut pairs: Vec<(Value, Value)> = Vec::new();
    match value {
        None => {}
        Some(value) if value.is_undefined() => {}
        Some(value) if value.kind() == ValueKind::Map => {
            if let Some(iter) = value.as_object().and_then(|object| object.try_iter_pairs()) {
                pairs.extend(iter);
            }
        }
        Some(value) => {
            let not_iterable = |kind: ValueKind| {
                MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("dict() takes a mapping or an iterable of key/value pairs, not {kind}"),
                )
            };
            if value.is_none() {
                return Err(not_iterable(value.kind()));
            }
            let items = value.try_iter().map_err(|_| not_iterable(value.kind()))?;
            for (index, item) in items.enumerate() {
                let pair: Vec<Value> = item
                    .try_iter()
                    .map_err(|_| {
                        MinijinjaError::new(
                            ErrorKind::InvalidOperation,
                            format!("cannot convert dictionary update sequence element #{index} to a sequence"),
                        )
                    })?
                    .collect();
                let [key, item_value] = <[Value; 2]>::try_from(pair).map_err(|pair| {
                    MinijinjaError::new(
                        ErrorKind::InvalidOperation,
                        format!(
                            "dictionary update sequence element #{index} has length {}; 2 is required",
                            pair.len()
                        ),
                    )
                })?;
                pairs.push((key, item_value));
            }
        }
    }
    for name in kwargs.args() {
        pairs.push((Value::from(name), kwargs.peek::<Value>(name)?));
    }
    Ok(Value::from_iter(pairs))
}

/// Build a pre-configured `Environment<'static>` with the given template string,
/// Python-compat method callback, and custom `tojson` filter already registered.
/// The template is stored under the name `"chat"` using owned storage so the
/// environment carries no borrows.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let template = match plain_generation_blocks(&template) {
        Cow::Borrowed(_) => template,
        Cow::Owned(rewritten) => rewritten,
    };
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
    // Jinja2's dict() also builds a map from key/value pairs; minijinja's takes a mapping only
    env.add_function("dict", dict_function);
    env.add_test("iterable", is_iterable);

    Ok(env)
}

/// Jinja2's `iterable` test: `iter(value)` succeeds. minijinja iterates
/// `none` as an empty sequence and would call it iterable; in Python
/// `iter(None)` raises, so `none is iterable` is false, which the templates
/// that guard with `tools is iterable and tools | length > 0` rely on when
/// the request carries no tools. An undefined value is iterable in both.
fn is_iterable(value: &Value) -> bool {
    !value.is_none() && value.try_iter().is_ok()
}

/// Render the `"chat"` template in the given environment against messages and params.
/// Convert an optional token string to a minijinja Value.
/// Present tokens become strings; absent tokens become UNDEFINED
/// so templates can use `{% if bos_token is defined %}` guards.
fn special_token_value(token: Option<&str>) -> Value {
    token.map_or(Value::UNDEFINED, Value::from)
}

/// transformers' `strftime_now(format)`: `now` written with the strftime
/// `format`, as `datetime.now().strftime(format)` writes the local time.
fn strftime(
    now: &DateTime<FixedOffset>,
    format: &str,
) -> std::result::Result<String, MinijinjaError> {
    let items: Vec<Item<'_>> = StrftimeItems::new(format).collect();
    if items.contains(&Item::Error) {
        return Err(MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("strftime_now: {format:?} is not a valid strftime format"),
        ));
    }
    Ok(now.format_with_items(items.iter()).to_string())
}

/// The instant `strftime_now` writes when the render names none: the local
/// time now, as transformers' `datetime.now()`; or, when the environment sets
/// `SOURCE_DATE_EPOCH` (seconds since the Unix epoch, the reproducible-builds
/// convention), that instant in local time, read once, so a prompt that writes
/// the date can be reproduced on another day.
fn render_instant() -> DateTime<FixedOffset> {
    static SOURCE_DATE_EPOCH: OnceLock<Option<DateTime<FixedOffset>>> = OnceLock::new();
    SOURCE_DATE_EPOCH
        .get_or_init(|| {
            let seconds = std::env::var("SOURCE_DATE_EPOCH")
                .ok()?
                .trim()
                .parse()
                .ok()?;
            Local
                .timestamp_opt(seconds, 0)
                .single()
                .map(|instant| instant.fixed_offset())
        })
        .unwrap_or_else(|| Local::now().fixed_offset())
}

/// Whether the template mentions the `developer` role at all (the engine's
/// `_detect_developer_role_support`): a template that never names it has no
/// branch for it, and the renderer rewrites developer messages to system ones.
fn detect_developer_role_support(template: &str) -> bool {
    template.contains("\"developer\"") || template.contains("'developer'")
}

/// The messages with every `developer` message rewritten as a `system`
/// message without its `tools` field (the engine's `_convert_developer_to_system`),
/// then the system messages merged into one at the front when a system
/// message is not the first message (the engine's `_consolidate_system_messages`,
/// which follows the rewrite in its renderer). `None` when no message has
/// the role, so the caller keeps its slice.
fn developer_messages_as_system(messages: &[serde_json::Value]) -> Option<Vec<serde_json::Value>> {
    let is_developer =
        |m: &serde_json::Value| m.get("role").and_then(|r| r.as_str()) == Some("developer");
    if !messages.iter().any(is_developer) {
        return None;
    }
    let rewritten = messages
        .iter()
        .map(|m| {
            if !is_developer(m) {
                return m.clone();
            }
            let mut rewritten = m.clone();
            if let Some(fields) = rewritten.as_object_mut() {
                fields.insert(
                    "role".to_string(),
                    serde_json::Value::String("system".to_string()),
                );
                fields.remove("tools");
            }
            rewritten
        })
        .collect();
    Some(consolidate_system_messages(rewritten))
}

/// The engine's `_consolidate_system_messages`, for the templates that want the
/// system message first: the messages unchanged when the only system message
/// is the first one; otherwise one system message at the front carrying the
/// non-empty system texts joined by blank lines (a parts list contributes its
/// text parts joined by newlines), then the other messages in their order.
fn consolidate_system_messages(messages: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let is_system =
        |m: &serde_json::Value| m.get("role").and_then(|r| r.as_str()) == Some("system");
    if !messages
        .iter()
        .enumerate()
        .any(|(index, m)| index > 0 && is_system(m))
    {
        return messages;
    }
    let mut system_texts: Vec<String> = Vec::new();
    let mut others = Vec::with_capacity(messages.len());
    for message in messages {
        if !is_system(&message) {
            others.push(message);
            continue;
        }
        let text = match message.get("content") {
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| match part {
                    serde_json::Value::String(text) => Some(text.as_str()),
                    serde_json::Value::Object(fields) => {
                        fields.get("text").and_then(serde_json::Value::as_str)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !text.is_empty() {
            system_texts.push(text);
        }
    }
    let merged = serde_json::json!({"role": "system", "content": system_texts.join("\n\n")});
    std::iter::once(merged).chain(others).collect()
}

fn render_chat_template(
    env: &Environment<'_>,
    messages: &[serde_json::Value],
    params: ChatTemplateParams,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("Failed to get template: {e}"))?;

    // transformers adds `strftime_now` to the template environment; here it is
    // a callable of the render context, bound to the render's instant.
    let now = params.now.unwrap_or_else(render_instant);
    let strftime_now = Value::from_function(move |format: String| strftime(&now, &format));

    // Convert messages to minijinja::Value (messages already processed by router)
    let minijinja_messages: Vec<Value> = messages.iter().map(Value::from_serialize).collect();

    // transformers renders with `tools=None` and `documents=None` when the
    // request carries none: the names are defined and hold none. A template
    // may test them either way (`tools is none`, `tools is defined and tools`,
    // `tools is iterable`); undefined would send a template that tests
    // `is none` down its tools branch with nothing to write.
    let tools_value = params
        .tools
        .map_or_else(|| Value::from(()), Value::from_serialize);
    let documents_value = params
        .documents
        .map_or_else(|| Value::from(()), Value::from_serialize);

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
        strftime_now => strftime_now,
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

/// The messages with every string `content` of a non-tool message replaced by
/// a one-item text part list, or `None` when no message has one.
fn string_content_as_text_parts(messages: &[serde_json::Value]) -> Option<Vec<serde_json::Value>> {
    fn needs_wrap(message: &serde_json::Value) -> bool {
        message
            .get("content")
            .is_some_and(serde_json::Value::is_string)
            && message.get("role").and_then(serde_json::Value::as_str) != Some("tool")
    }
    if !messages.iter().any(needs_wrap) {
        return None;
    }
    let wrapped = messages
        .iter()
        .map(|message| {
            if !needs_wrap(message) {
                return message.clone();
            }
            let mut message = message.clone();
            let text = message["content"].take();
            message["content"] = serde_json::json!([{ "type": "text", "text": text }]);
            message
        })
        .collect();
    Some(wrapped)
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
    /// Whether the template has a branch for the `developer` role. When it
    /// has none, `apply` renders developer messages as system messages, as
    /// the engine's HF renderer does, instead of letting the template drop them.
    developer_role_supported: bool,
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
        let developer_role_supported = template
            .as_deref()
            .is_none_or(detect_developer_role_support);
        let env = template.map(build_environment).transpose()?;
        Ok(Self {
            env,
            content_format,
            thinking_toggle,
            thinking_key_name,
            think_in_prefill,
            developer_role_supported,
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
            developer_role_supported: true,
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

        // The serving engine hands an "openai"-format template every message's string content
        // as a one-item text part list (`_parse_chat_message_content`: a `str`
        // becomes `[{"type": "text", "text": ...}]`, and in that format the
        // parts stay dicts), so such a template always takes its parts branch.
        // Render the same way, so that a template whose two branches differ (a
        // separator after every part, a truthiness check on the content)
        // produces the engine's prompt for string content too. A tool result
        // stays a string: the engine joins its text parts back into one.
        let wrapped = (self.content_format == ChatTemplateContentFormat::OpenAI)
            .then(|| string_content_as_text_parts(messages))
            .flatten();
        let messages = wrapped.as_deref().unwrap_or(messages);

        // A template without a `developer` branch renders nothing for a
        // developer message; the engine's renderer hands such a template the message
        // as a system message, and so does this one (`tools` on it dropped).
        let converted;
        let messages: &[serde_json::Value] = if self.developer_role_supported {
            messages
        } else {
            match developer_messages_as_system(messages) {
                Some(rewritten) => {
                    converted = rewritten;
                    &converted
                }
                None => messages,
            }
        };

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
                let value = match key {
                    ThinkingKeyName::EnableThinking | ThinkingKeyName::Thinking => {
                        serde_json::Value::Bool(thinking)
                    }
                    ThinkingKeyName::ReasoningEffort => serde_json::Value::String(
                        if thinking {
                            REASONING_EFFORT_ON_VALUES[0]
                        } else {
                            REASONING_EFFORT_OFF_VALUES[0]
                        }
                        .to_string(),
                    ),
                    // The tri-state key compares strings, not booleans.
                    ThinkingKeyName::ThinkingMode => serde_json::Value::String(
                        if thinking { "enabled" } else { "disabled" }.to_string(),
                    ),
                };
                kwargs.insert(kwarg_key.to_string(), value);
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
        let developer_role_supported = detect_developer_role_support(&template);
        let env = build_environment(template)?;
        self.developer_role_supported = developer_role_supported;
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

    /// `reasoning_effort` values that turn thinking on for this template.
    pub fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        match self.thinking_key_name {
            Some(ThinkingKeyName::ReasoningEffort) => REASONING_EFFORT_ON_VALUES,
            _ => &[],
        }
    }

    /// `reasoning_effort` values that turn thinking off for this template.
    pub fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        match self.thinking_key_name {
            Some(ThinkingKeyName::ReasoningEffort) => REASONING_EFFORT_OFF_VALUES,
            _ => &[],
        }
    }

    pub fn think_in_prefill(&self) -> bool {
        self.think_in_prefill
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transformers' `{% generation %}` block (the assistant-token mask marker)
    /// renders its body; the engine here has no such statement, so the block
    /// is rewritten to a plain `if` before parsing, and the content-format
    /// detection parses such a template as well.
    #[test]
    fn generation_blocks_render_their_body() {
        let template = "{%- for m in messages -%}\n{%- if m.role == 'assistant' -%}\n\
                        {% generation %}\n<a>{{ m.content }}</a>\n{%- endgeneration -%}\n\
                        {%- else -%}\n<u>{{ m.content }}</u>\n{%- endif -%}\n{%- endfor -%}";
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let messages = [
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "yo"}),
        ];
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "<u>hi</u><a>yo</a>");

        let parts_template = "{% for message in messages %}{% if message['role'] == 'user' %}USER: \
                              {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}\
                              {% generation %}{{ content['text'] + ' ' }}{% endgeneration %}{% endfor %}\
                              {% endif %}{% endfor %}";
        assert_eq!(
            detect_chat_template_content_format(parts_template),
            ChatTemplateContentFormat::OpenAI
        );
    }

    /// The rewrite keeps each tag's whitespace modifiers and leaves a template
    /// without the block untouched.
    #[test]
    fn generation_tags_are_rewritten_with_their_modifiers() {
        assert_eq!(
            plain_generation_blocks(
                "{%- generation -%}x{% endgeneration %}y{%+ generation %}z{%-endgeneration-%}"
            ),
            "{%- if true -%}x{% endif %}y{%+ if true %}z{%- endif -%}"
        );
        assert!(matches!(
            plain_generation_blocks("{% if add_generation_prompt %}a{% endif %}"),
            Cow::Borrowed(_)
        ));
    }

    /// transformers' `strftime_now(format)`: the render's instant written with
    /// the format; a template may guard it with `is defined`; a render that
    /// names no instant writes today; an invalid format is an error.
    #[test]
    fn strftime_now_writes_the_render_instant() {
        let template = "{%- if strftime_now is defined -%}Current date: \
                        {{ strftime_now('%Y-%m-%d') }}. {{ strftime_now('%H:%M') }}{%- endif -%}";
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let messages: [serde_json::Value; 0] = [];
        let now = DateTime::parse_from_rfc3339("2026-10-07T09:30:00+02:00").unwrap();
        let params = ChatTemplateParams {
            now: Some(now),
            ..Default::default()
        };
        let rendered = processor.apply_chat_template(&messages, params).unwrap();
        assert_eq!(rendered, "Current date: 2026-10-07. 09:30");

        let today = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            today.len(),
            "Current date: 2026-10-07. 09:30".len(),
            "{today}"
        );

        let invalid = ChatTemplateProcessor::new("{{ strftime_now('%Q') }}".to_string()).unwrap();
        let error = invalid
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("strftime_now"), "{error}");
    }

    /// Jinja2's `dict()` is Python's: it builds a map from nothing, from a
    /// mapping or from an iterable of key/value pairs, keyword arguments are
    /// written over the result, and a repeated key keeps its first position
    /// with its last value. A template merges a schema's `$defs` entry into
    /// the property that references it as `dict((a | items | list) + (b |
    /// items | list))`; the engine's own `dict` takes a mapping only.
    #[test]
    fn dict_builds_a_map_from_pairs_as_python_does() {
        let template = "{{ dict([('a', 1), ('b', 2), ('a', 3)]) | tojson }}|\
                        {{ dict({'x': 1}, y=2) | tojson }}|{{ dict() | tojson }}|\
                        {{ dict((messages[0].defs | items | list) + \
                        (messages[0].spec | items | rejectattr('0', 'equalto', '$ref') | list)) | tojson }}";
        let messages = [serde_json::json!({
            "role": "user",
            "defs": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "spec": {"$ref": "#/$defs/airport", "description": "IATA code"},
        })];
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            r#"{"a": 3, "b": 2}|{"x": 1, "y": 2}|{}|{"type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA code"}"#
        );

        // An element that is not a key/value pair is an error, as in Python.
        let processor = ChatTemplateProcessor::new("{{ dict([['a']]) }}".to_string()).unwrap();
        let error = processor
            .apply_chat_template(&[], ChatTemplateParams::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("2 is required"), "{error}");
    }

    /// transformers renders with `tools=None` and `documents=None` when the
    /// request carries none, so a template that tests `tools is none` (Olmo 3)
    /// takes its no-tools branch, and one that guards with `is defined and
    /// tools` does too.
    #[test]
    fn absent_tools_and_documents_are_none_as_transformers_passes_them() {
        let template = "{% if tools is none %}no tools{% else %}{{ tools | tojson }}{% endif %}|\
                        {% if tools is defined and tools %}has tools{% else %}none{% endif %}|\
                        {% if tools is iterable and tools | length > 0 %}iterable{% else %}not iterable{% endif %}|\
                        {% if documents is none %}no documents{% endif %}";
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let messages: [serde_json::Value; 0] = [];
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "no tools|none|not iterable|no documents");

        let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let rendered = processor.apply_chat_template(&messages, params).unwrap();
        assert_eq!(
            rendered,
            "[{\"type\": \"function\", \"function\": {\"name\": \"f\"}}]|has tools|iterable|no documents"
        );
    }

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
    fn developer_message_renders_as_system_when_the_template_has_no_developer_branch() {
        // The template names system/user/assistant only: a developer message
        // would render nothing. The engine's renderer turns it into a system message.
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}\
                        {% if messages[0].tools is defined %}TOOLS{% endif %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "developer", "content": "Be terse.", "tools": [{"name": "t"}]}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "<sys>Be terse.</sys><usr>hi</usr>");
    }

    /// With a system message of its own in the request, the rewritten
    /// developer message is no longer the first message: the two merge into
    /// one system turn at the front, as the engine's renderer merges them.
    #[test]
    fn a_developer_message_after_a_system_message_merges_into_one_system_turn() {
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are terse."}),
            serde_json::json!({"role": "developer", "content": "Answer in French."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<sys>You are terse.\n\nAnswer in French.</sys><usr>hi</usr>"
        );
    }

    /// A developer message in the middle of the conversation becomes the
    /// system turn at the front; the other messages keep their order, and a
    /// parts list contributes its text parts.
    #[test]
    fn a_developer_message_after_an_assistant_turn_moves_to_the_front() {
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>\
                        {% elif m.role == 'assistant' %}<ast>{{ m.content }}</ast>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "hello"}),
            serde_json::json!({"role": "developer", "content": [
                {"type": "text", "text": "Be brief."},
                {"type": "text", "text": "No lists."},
            ]}),
            serde_json::json!({"role": "user", "content": "go on"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<sys>Be brief.\nNo lists.</sys><usr>hi</usr><ast>hello</ast><usr>go on</usr>"
        );
    }

    #[test]
    fn developer_message_reaches_a_template_that_handles_the_role() {
        let template = "{% for m in messages %}{% if m.role == 'developer' %}<dev>{{ m.content }}</dev>\
                        {% elif m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "developer", "content": "Be terse."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "<dev>Be terse.</dev><usr>hi</usr>");
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

    #[test]
    fn thinking_mode_detects_tristate_and_injects_strings() {
        // MiniMax M3-style tri-state toggle.
        let template = r#"{%- if thinking_mode is defined and thinking_mode == "enabled" -%}<mm:think>{%- endif -%}"#;
        assert_eq!(
            detect_thinking_toggle(template),
            (
                ThinkingToggle::DefaultOff,
                Some(ThinkingKeyName::ThinkingMode)
            )
        );

        // Mentioning the variable without branching on it must not trigger
        // tri-state detection.
        assert_eq!(
            detect_thinking_toggle("{{ enable_thinking }}{# thinking_mode note #}"),
            (
                ThinkingToggle::DefaultOn,
                Some(ThinkingKeyName::EnableThinking)
            )
        );

        // The resolved boolean preference injects the template's string values.
        let state = ChatTemplateState::new(Some(
            "{% if thinking_mode is defined %}{{ thinking_mode }}{% endif %}".to_string(),
        ))
        .unwrap();
        assert_eq!(
            state.thinking_key_name(),
            Some(ThinkingKeyName::ThinkingMode)
        );
        for (thinking, rendered) in [(true, "enabled"), (false, "disabled")] {
            let out = state
                .apply(
                    &[],
                    ChatTemplateParams {
                        thinking: Some(thinking),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(out, rendered);
        }
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

#[cfg(test)]
mod hy_v4_tests {
    use super::*;
    #[test]
    fn hy4_effort_controls_prefill_and_explicit_kwarg_wins() {
        let template = "{% set think_begin_token = '<think:6124c78e>' %}{% if reasoning_effort is not defined %}{% set reasoning_effort = 'high' %}{% endif %}{{ think_begin_token }}{% if reasoning_effort == 'no_think' %}</think:6124c78e>{% endif %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOn);
        assert_eq!(
            state.thinking_key_name(),
            Some(ThinkingKeyName::ReasoningEffort)
        );
        assert_eq!(state.native_reasoning_effort_values(), &["high"]);
        assert_eq!(state.native_reasoning_effort_off_values(), &["no_think"]);
        for (thinking, expected) in [
            (None, "<think:6124c78e>"),
            (Some(true), "<think:6124c78e>"),
            (Some(false), "<think:6124c78e></think:6124c78e>"),
        ] {
            assert_eq!(
                state
                    .apply(
                        &[],
                        ChatTemplateParams {
                            thinking,
                            ..Default::default()
                        }
                    )
                    .unwrap(),
                expected
            );
        }
        let kwargs = HashMap::from([(
            "reasoning_effort".to_string(),
            serde_json::json!("no_think"),
        )]);
        assert_eq!(
            state
                .apply(
                    &[],
                    ChatTemplateParams {
                        thinking: Some(true),
                        template_kwargs: Some(&kwargs),
                        ..Default::default()
                    }
                )
                .unwrap(),
            "<think:6124c78e></think:6124c78e>"
        );
    }
}
