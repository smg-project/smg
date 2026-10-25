//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    fs, io, iter,
    sync::{Arc, OnceLock},
};

use anyhow::{anyhow, Result};
use chrono::{
    format::{strftime::StrftimeItems, Item},
    DateTime, FixedOffset, Local, TimeZone,
};
use minijinja::{
    context,
    machinery::{
        ast::{BinOpKind, Call, CallArg, Expr, ForLoop, IfCond, Macro, Set, Stmt, UnaryOpKind},
        parse, WhitespaceConfig,
    },
    syntax::SyntaxConfig,
    value::{DynObject, Enumerator, Kwargs, Object, ObjectRepr, ValueKind},
    Environment, Error as MinijinjaError, ErrorKind, Output, State, Value,
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
    let has_thinking_var = mentions_thinking_variable(template);

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

    // Gemma4 assigns its default through a Jinja filter. A filter on a
    // message field or another variable does not set the thinking switch.
    let compact: String = template.split_whitespace().collect();
    if compact.contains("setenable_thinking=enable_thinking|default(false)")
        || compact.contains("setthinking=thinking|default(false)")
    {
        return (ThinkingToggle::DefaultOff, Some(key_name));
    }

    // All other models default to thinking ON
    (ThinkingToggle::DefaultOn, Some(key_name))
}

/// Whether the template reads a variable named exactly `thinking` (`if
/// thinking ...`, `thinking is ...`, `thinking == ...`, `set thinking ...`).
/// The word must stand alone: `clear_thinking is defined` (GLM-5.3) or
/// `thinking_mode`/`thinking_budget` name other variables, and a template
/// that only reads those has no thinking switch.
fn mentions_thinking_variable(template: &str) -> bool {
    template.match_indices("thinking").any(|(start, matched)| {
        let before = &template[..start];
        let after = &template[start + matched.len()..];
        if before
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return false;
        }
        after.starts_with(" is ")
            || after.starts_with(" ==")
            || ((before.ends_with("if ") || before.ends_with("set ")) && after.starts_with(' '))
    })
}

/// Detect the content format expected by a Jinja2 chat template
///
/// The rule is the serving engine's own content-format detection, so that
/// the gateway hands string content as a one-item text part list to the
/// same templates as the engine does (see [`ChatTemplateState::apply`]). A
/// template is of the "openai" format when it has a loop over a message's
/// content, a message being a loop variable over `messages` or over a
/// variable assigned from it (a namespace attribute included, `{% set
/// ns.rest = messages[1:] %}`): the loop runs over `message.content` (also
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
        // access only. Without a key, a namespace attribute path is the name.
        Expr::GetAttr(g) => match key {
            Some(key) => g.name == key && is_var(&g.expr, varname),
            None => attr_path(expr).as_deref() == Some(varname),
        },
        Expr::GetItem(g) => key.is_some_and(|key| {
            matches!(&g.subscript_expr, Expr::Const(c) if c.value.as_str() == Some(key))
                && is_var(&g.expr, varname)
        }),
        _ => false,
    }
}

/// Whether `expr` is the plain name `varname`, or the namespace attribute
/// path `varname` spells (`ns.rest`): the engine's `_is_var_access`, plus
/// the namespace attributes [`names_assigned_from`] records.
fn is_var(expr: &Expr<'_>, varname: &str) -> bool {
    match expr {
        Expr::Var(v) => v.id == varname,
        Expr::GetAttr(_) => attr_path(expr).as_deref() == Some(varname),
        _ => false,
    }
}

/// The dotted path of a chain of attribute accesses on a plain name
/// (`ns.rest`), the way a namespace attribute is written; `None` for any
/// other expression.
fn attr_path(expr: &Expr<'_>) -> Option<String> {
    match expr {
        Expr::Var(v) => Some(v.id.to_string()),
        Expr::GetAttr(g) => attr_path(&g.expr).map(|base| format!("{base}.{}", g.name)),
        _ => None,
    }
}

/// `root` and every name assigned from it or from such a name, through
/// filters, tests and slices: the engine's `_iter_nodes_assign_var_or_elems`.
/// A namespace attribute assigned that way (`{% set ns.rest = messages[1:] %}`,
/// which a template loops over as `ns.rest`) is recorded by its dotted path,
/// where the engine's walk stops; any other target that is not a name stops
/// this one too.
fn names_assigned_from<'a>(sets: &[&'a Set<'a>], root: &'a str) -> Result<Vec<String>, NotAName> {
    let mut names = vec![root.to_string()];
    let mut queue = std::collections::VecDeque::from([root.to_string()]);
    while let Some(related) = queue.pop_front() {
        for set in sets {
            if !is_var_or_elems_access(&set.expr, &related, None) {
                continue;
            }
            let target = match &set.target {
                Expr::Var(target) => target.id.to_string(),
                Expr::GetAttr(_) => attr_path(&set.target).ok_or(NotAName)?,
                _ => return Err(NotAName),
            };
            if !names.contains(&target) {
                names.push(target.clone());
                queue.push_back(target);
            }
        }
    }
    Ok(names)
}

/// The loop variables of the loops over a messages name: the engine's
/// `_iter_nodes_assign_messages_item`.
fn message_loop_vars<'a>(
    loops: &[(&'a ForLoop<'a>, Vec<&'a Macro<'a>>)],
    messages_names: &[String],
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
    /// Whether the `add_generation_prompt` if-block opens a `<think>` block
    /// it does not close, so a completion starts inside reasoning.
    prefill_opens_think_block: bool,
}

impl ThinkDetector {
    fn run(ast: &Stmt<'_>) -> (bool, bool) {
        let mut detector = Self {
            think_in_prefill: false,
            prefill_opens_think_block: false,
        };
        detector.walk_stmt(ast);
        (
            detector.think_in_prefill,
            detector.prefill_opens_think_block,
        )
    }

    /// Check if an expression references a variable by name (walks through
    /// BinOp/UnaryOp/Test).
    fn expr_references_var(expr: &Expr, name: &str) -> bool {
        match expr {
            Expr::Var(v) => v.id == name,
            Expr::BinOp(b) => {
                Self::expr_references_var(&b.left, name)
                    || Self::expr_references_var(&b.right, name)
            }
            Expr::UnaryOp(u) => Self::expr_references_var(&u.expr, name),
            Expr::Test(t) => Self::expr_references_var(&t.expr, name),
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

    /// Whether a `<think>` block is open after the generation prompt an `if`
    /// writes, starting from `open` (the state the generation-prompt blocks
    /// before it left; a template may write the prompt in several of them).
    /// A block whose condition is decided by `add_generation_prompt` alone
    /// (see [`Self::generation_prompt_truth`]) carries the state through the
    /// body that runs for a generation prompt; a block whose condition also
    /// depends on something else may run either body, so the block is open
    /// after it when either body leaves it open; `None` for a condition that
    /// does not mention the variable.
    fn generation_prompt_open_after(ic: &IfCond, open: bool) -> Option<bool> {
        match Self::generation_prompt_truth(&ic.expr) {
            Some(true) => Some(Self::think_block_open_after(&ic.true_body, open)),
            Some(false) => Some(Self::think_block_open_after(&ic.false_body, open)),
            None if Self::expr_references_var(&ic.expr, "add_generation_prompt") => Some(
                Self::think_block_open_after(&ic.true_body, open)
                    || Self::think_block_open_after(&ic.false_body, open),
            ),
            None => None,
        }
    }

    /// The truth of a condition when `add_generation_prompt` is true: `Some`
    /// when the variable alone decides it (the variable itself, `not`, `and`,
    /// `or`, `==` / `!=` against a boolean, `is true` / `is false` /
    /// `is defined`), `None` when it depends on anything else.
    fn generation_prompt_truth(expr: &Expr) -> Option<bool> {
        match expr {
            Expr::Var(v) if v.id == "add_generation_prompt" => Some(true),
            Expr::UnaryOp(u) if matches!(u.op, UnaryOpKind::Not) => {
                Self::generation_prompt_truth(&u.expr).map(|truth| !truth)
            }
            Expr::BinOp(b) => match b.op {
                BinOpKind::ScAnd => match (
                    Self::generation_prompt_truth(&b.left),
                    Self::generation_prompt_truth(&b.right),
                ) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                BinOpKind::ScOr => match (
                    Self::generation_prompt_truth(&b.left),
                    Self::generation_prompt_truth(&b.right),
                ) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
                BinOpKind::Eq | BinOpKind::Ne => {
                    let (side, constant) = match (&b.left, &b.right) {
                        (side, Expr::Const(constant)) | (Expr::Const(constant), side) => {
                            (side, constant)
                        }
                        _ => return None,
                    };
                    if !matches!(constant.value.kind(), ValueKind::Bool) {
                        return None;
                    }
                    let equal = Self::generation_prompt_truth(side)? == constant.value.is_true();
                    Some(if matches!(b.op, BinOpKind::Eq) {
                        equal
                    } else {
                        !equal
                    })
                }
                _ => None,
            },
            Expr::Test(t) => {
                let truth = Self::generation_prompt_truth(&t.expr)?;
                match t.name {
                    "defined" => Some(true),
                    "undefined" | "none" => Some(false),
                    "true" => Some(truth),
                    "false" => Some(!truth),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Whether the literal text a list of statements writes leaves a
    /// `<think>` block open at its end, starting from `open`: the text is
    /// read in order, a `<think>` opens and a `</think>` closes; either
    /// branch of a nested `if` may run, so the block is open after it when
    /// either branch leaves it open, and text after the `if` still counts.
    /// A closed `<think>\n\n</think>` (a non-thinking template) opens nothing.
    fn think_block_open_after(stmts: &[Stmt], mut open: bool) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::EmitRaw(raw) => open = Self::think_block_open_after_text(raw.raw, open),
                Stmt::EmitExpr(e) => {
                    if let Expr::Const(c) = &e.expr {
                        if let Some(s) = c.value.as_str() {
                            open = Self::think_block_open_after_text(s, open);
                        }
                    }
                }
                Stmt::IfCond(ic) => {
                    open = Self::think_block_open_after(&ic.true_body, open)
                        || Self::think_block_open_after(&ic.false_body, open);
                }
                _ => {}
            }
        }
        open
    }

    /// `open` after `text`: the later of its last `<think>` and its last
    /// `</think>` decides; a text with neither leaves it as it was.
    fn think_block_open_after_text(text: &str, open: bool) -> bool {
        match (text.rfind("<think>"), text.rfind("</think>")) {
            (None, None) => open,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(start), Some(end)) => start > end,
        }
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
                if let Some(open) =
                    Self::generation_prompt_open_after(ic, self.prefill_opens_think_block)
                {
                    self.prefill_opens_think_block = open;
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
    bool,
) {
    let (thinking_toggle, thinking_key_name) = detect_thinking_toggle(template);
    let (content_format, think_in_prefill, prefill_opens_think_block) =
        detect_all_with_ast(template);
    (
        content_format,
        think_in_prefill,
        thinking_toggle,
        thinking_key_name,
        prefill_opens_think_block,
    )
}

/// AST detection of content format, think-in-prefill and whether the
/// generation prompt leaves a think block open.
fn detect_all_with_ast(template: &str) -> (ChatTemplateContentFormat, bool, bool) {
    let template = plain_generation_blocks(template);
    let ast = match parse(
        &template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return (ChatTemplateContentFormat::String, false, false),
    };

    let content_format = match iterates_message_content(&ast) {
        Ok(true) => ChatTemplateContentFormat::OpenAI,
        Ok(false) | Err(NotAName) => ChatTemplateContentFormat::String,
    };
    let (think_in_prefill, prefill_opens_think_block) = ThinkDetector::run(&ast);
    (content_format, think_in_prefill, prefill_opens_think_block)
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
    /// `render_instant`).
    pub now: Option<DateTime<FixedOffset>>,
    /// Leave the final message open for the model to continue, as
    /// transformers' `continue_final_message` does: the conversation is
    /// rendered with that message and cut right after its text. Not
    /// compatible with `add_generation_prompt`. Where the template cannot
    /// continue the message, its text follows the generation prompt instead
    /// (`ChatTemplateOutput::continued_final_message` says which happened),
    /// and a message that also has `reasoning_content` is refused.
    pub continue_final_message: bool,
    /// `tools` converted for the render once ([`PreparedTools::new`] of these
    /// same values): the render takes it instead of converting `tools` again,
    /// so a caller that keeps one per distinct tool set spares every request
    /// that carries the same tools the conversion. It must be made of
    /// `tools`: one made of other values is not read, and the render converts
    /// `tools` as without it.
    pub prepared_tools: Option<&'a PreparedTools>,
}

/// The tools of a request converted for the render once, so a caller can keep
/// the conversion per distinct tool set and hand it to every request that
/// carries the same tools instead of having the render convert the values
/// again ([`ChatTemplateParams::prepared_tools`]). The conversion is the one
/// the render makes of [`ChatTemplateParams::tools`], so a render with the
/// prepared value is the render without it, byte for byte.
pub struct PreparedTools {
    tools: Vec<JsonValue>,
    /// The render's value of `tools`: the serializer's sequence, each tool
    /// that is a map held as a [`KeptTool`].
    value: Value,
}

impl PreparedTools {
    /// Converts `tools` as a render converts [`ChatTemplateParams::tools`].
    pub fn new(tools: Vec<JsonValue>) -> Self {
        // The serializer's sequence of the slice, item by item: a tool that
        // is a map renders through a `KeptTool`, which answers every template
        // operation as the map itself and keeps the text `tojson` writes of
        // it; any other item is the serializer's value of it.
        let items: Vec<Value> = tools.iter().map(KeptTool::value_of_tool).collect();
        let value = Value::from_object(items);
        Self { tools, value }
    }

    /// The values this was made of: what a caller passes as
    /// [`ChatTemplateParams::tools`] beside it.
    pub fn tools(&self) -> &[JsonValue] {
        &self.tools
    }

    /// The render's value of `tools`, when they are the values this was made
    /// of (the same slice).
    fn value_of(&self, tools: &[JsonValue]) -> Option<Value> {
        (std::ptr::eq(self.tools.as_ptr(), tools.as_ptr()) && self.tools.len() == tools.len())
            .then(|| self.value.clone())
    }
}

/// A prepared tool as the render reads it: the map the serializer made of
/// the tool, which answers every template operation (the tests of `is
/// mapping`, `in` and truth, the lookups, the iteration and its order, the
/// methods, the comparisons, the text forms), and beside it the text
/// `tojson` wrote of the map, kept for the next request that renders the
/// same kept value: a tool set is an application's, and its `tojson` text
/// under the same arguments is the same bytes for every request. The two
/// forms the shipped templates use are kept, `tojson` with no argument and
/// `tojson(indent=...)` with one indent; any other call (`ensure_ascii`,
/// `separators`, `sort_keys`, a second indent) is written as for any value.
struct KeptTool {
    /// The tool's map as the serializer made it: the object every
    /// operation forwards to.
    object: DynObject,
    /// The text of `tojson` with no argument, once written.
    plain: OnceLock<Value>,
    /// The indent of the first `tojson(indent=...)` asked and its text,
    /// once written.
    indented: OnceLock<(Vec<u8>, Value)>,
}

impl KeptTool {
    /// The render's value of one prepared tool: the kept form when the tool
    /// is a map (a function tool is), else the serializer's value of it.
    fn value_of_tool(tool: &JsonValue) -> Value {
        let value = Value::from_serialize(tool);
        match value.as_object() {
            Some(object) if object.repr() == ObjectRepr::Map => Value::from_object(KeptTool {
                object: object.clone(),
                plain: OnceLock::new(),
                indented: OnceLock::new(),
            }),
            _ => value,
        }
    }

    /// The `tojson` text of this tool for `indent` (`None` for the call
    /// without an argument) as a safe string: the kept text when the form
    /// was written before, else `write`'s text, kept when the form is a
    /// kept one. The text is always the writer's own for this value and
    /// these arguments.
    fn tojson(
        &self,
        indent: Option<&[u8]>,
        write: impl FnOnce() -> std::result::Result<String, MinijinjaError>,
    ) -> std::result::Result<Value, MinijinjaError> {
        let Some(indent) = indent else {
            if let Some(text) = self.plain.get() {
                return Ok(text.clone());
            }
            let text = Value::from_safe_string(write()?);
            // Two requests may write the first text at once: the same bytes
            // either way, the slot keeps one.
            return Ok(self.plain.get_or_init(|| text).clone());
        };
        if let Some((kept_indent, text)) = self.indented.get() {
            return if kept_indent.as_slice() == indent {
                Ok(text.clone())
            } else {
                write().map(Value::from_safe_string)
            };
        }
        let text = Value::from_safe_string(write()?);
        // Two requests may race to keep two indents: the slot takes the
        // first; the other answers with the text it wrote.
        let (kept_indent, kept) = self
            .indented
            .get_or_init(|| (indent.to_vec(), text.clone()));
        Ok(if kept_indent.as_slice() == indent {
            kept.clone()
        } else {
            text
        })
    }
}

/// Every operation is the inner map's: the kept tool is that map to the
/// template, in every test, lookup, iteration, method, comparison and text
/// (`Value`'s `Debug` and `Display` of an object both go through `render`).
impl Object for KeptTool {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        self.object.repr()
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.object.get_value(key)
    }

    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
        self.object.get_value_by_str(key)
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.object.enumerate()
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        self.object.enumerator_len()
    }

    fn is_true(self: &Arc<Self>) -> bool {
        self.object.is_true()
    }

    fn call(
        self: &Arc<Self>,
        state: &State<'_, '_>,
        args: &[Value],
    ) -> std::result::Result<Value, MinijinjaError> {
        self.object.call(state, args)
    }

    fn call_method(
        self: &Arc<Self>,
        state: &State<'_, '_>,
        method: &str,
        args: &[Value],
    ) -> std::result::Result<Value, MinijinjaError> {
        self.object.call_method(state, method, args)
    }

    fn custom_cmp(self: &Arc<Self>, other: &DynObject) -> Option<Ordering> {
        self.object.custom_cmp(other)
    }

    // `render` stays the trait's own: for a map it is the debug map of the
    // pairs, which the inner map renders by the same default, and it keeps
    // the formatter's flags (`pprint` asks for the alternate form); a
    // forward through `Display` would write a fresh formatter and lose them.
}

/// As the map's own `Debug`: the pairs in their order.
impl std::fmt::Debug for KeptTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in self.object.try_iter_pairs().into_iter().flatten() {
            map.entry(&key, &value);
        }
        map.finish()
    }
}

/// The initial size of a `tojson` output buffer: room for a function schema
/// of the usual size without a reallocation.
const TOJSON_BUFFER_BYTES: usize = 1024;

/// JSON separator pair passed through HuggingFace's `tojson` filter. The
/// defaults are the static bytes; only a `separators` argument owns its own.
#[derive(Debug, Clone)]
struct JsonSeparators {
    item: Cow<'static, [u8]>,
    key: Cow<'static, [u8]>,
}

impl JsonSeparators {
    /// Python's defaults: `(', ', ': ')` for compact output and `(',', ': ')`
    /// when an indent was given (`pretty`), whatever the indent is: an empty
    /// one still puts every item on its own line.
    fn python_default(pretty: bool) -> Self {
        let item: &'static [u8] = if pretty { b"," } else { b", " };
        Self {
            item: Cow::Borrowed(item),
            key: Cow::Borrowed(b": "),
        }
    }
}

/// Formatter matching Python's `json.dumps` separator and ASCII escaping rules.
#[derive(Debug, Clone)]
struct PythonJsonFormatter<'a> {
    current_indent: usize,
    has_value: bool,
    indent: Option<&'a [u8]>,
    separators: JsonSeparators,
    ensure_ascii: bool,
}

impl<'a> PythonJsonFormatter<'a> {
    /// `indent` is the bytes written per level of nesting, borrowed from the
    /// filter call that owns them, `None` for compact output; an empty indent
    /// breaks the lines and indents nothing, as Python's `json.dumps(indent=0)`
    /// does.
    fn new(indent: Option<&'a [u8]>, separators: JsonSeparators, ensure_ascii: bool) -> Self {
        Self {
            current_indent: 0,
            has_value: false,
            indent,
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

impl Formatter for PythonJsonFormatter<'_> {
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
        if let Some(indent) = self.indent {
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
        if let Some(indent) = self.indent {
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
        if let Some(indent) = self.indent {
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
        if let Some(indent) = self.indent {
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

/// The indent `tojson(indent=...)` asked for, as Python's `json.dumps` reads
/// it: nothing (compact output); an integer, that many spaces, and none at
/// all when it is zero or negative (Python multiplies one space by it, so the
/// lines still break); a bool as the integer it is in Python (`True` is one
/// space); a string as the literal indent (`"\t"`, `""`); any other kind is
/// refused, as Python refuses a float.
fn parse_indent(indent: Option<Value>) -> std::result::Result<Option<Vec<u8>>, MinijinjaError> {
    let Some(indent) = indent else {
        return Ok(None);
    };
    match indent.kind() {
        ValueKind::None | ValueKind::Undefined => Ok(None),
        ValueKind::Bool => Ok(Some(if indent.is_true() {
            vec![b' ']
        } else {
            Vec::new()
        })),
        ValueKind::Number if indent.is_integer() => {
            let spaces = indent.as_i64().ok_or_else(|| {
                invalid_tojson_option("indent is too large to be a number of spaces")
            })?;
            Ok(Some(vec![b' '; usize::try_from(spaces).unwrap_or(0)]))
        }
        ValueKind::String => Ok(Some(
            indent.as_str().unwrap_or_default().as_bytes().to_vec(),
        )),
        other => Err(invalid_tojson_option(format!(
            "indent must be an integer or a string, not {other}"
        ))),
    }
}

fn parse_separators(
    separators: Option<Value>,
    pretty: bool,
) -> std::result::Result<JsonSeparators, MinijinjaError> {
    let Some(separators) = separators else {
        return Ok(JsonSeparators::python_default(pretty));
    };
    if separators.is_none() || separators.is_undefined() {
        return Ok(JsonSeparators::python_default(pretty));
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
        item: Cow::Owned(item.as_bytes().to_vec()),
        key: Cow::Owned(key.as_bytes().to_vec()),
    })
}

fn serialize_with_python_json<T: Serialize>(
    value: &T,
    indent: Option<&[u8]>,
    separators: JsonSeparators,
    ensure_ascii: bool,
) -> std::result::Result<String, MinijinjaError> {
    let formatter = PythonJsonFormatter::new(indent, separators, ensure_ascii);
    // Sized for a tool schema up front: a buffer grown from empty is
    // reallocated at every doubling on the way to the same size.
    let mut buf = Vec::with_capacity(TOJSON_BUFFER_BYTES);
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

/// Compact JSON spelling shared by the Jinja filter and native renderers.
pub(crate) fn compact_python_json(value: &JsonValue) -> Result<String> {
    serialize_with_python_json(value, None, JsonSeparators::python_default(false), false)
        .map_err(Into::into)
}

/// Custom tojson filter compatible with HuggingFace transformers' implementation.
///
/// HuggingFace transformers registers a custom `tojson` filter that accepts additional
/// keyword arguments beyond what standard Jinja2 provides:
/// - `ensure_ascii` (bool): Whether to escape non-ASCII characters
/// - `indent` (int or str): the indent per level of nesting, as Python reads it: a number of
///   spaces (zero or negative breaks the lines and indents nothing) or the literal string
/// - `separators`: Custom item/key separators for JSON output
/// - `sort_keys` (bool): Whether to sort dictionary keys
///
/// This is necessary for compatibility with chat templates from HuggingFace Hub models.
/// See: <https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py>
fn tojson_filter(value: Value, kwargs: Kwargs) -> std::result::Result<Value, MinijinjaError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;

    // Ensure all kwargs are consumed to avoid "unknown keyword argument" errors
    kwargs.assert_all_used()?;

    let separators_given = separators.is_some();
    let indent = parse_indent(indent)?;
    let separators = parse_separators(separators, indent.is_some())?;
    let ensure_ascii = ensure_ascii.unwrap_or(false);
    let sort_keys = sort_keys.unwrap_or(false);
    // A prepared tool's text under the forms the shipped templates use is
    // written once per kept tool and shared by every request that renders
    // the same kept value; every other form is written as for any value.
    if !sort_keys && !ensure_ascii && !separators_given {
        if let Some(kept) = value
            .as_object()
            .and_then(|object| object.downcast_ref::<KeptTool>())
        {
            let separators = separators.clone();
            return kept.tojson(indent.as_deref(), || {
                serialize_with_python_json(&value, indent.as_deref(), separators, ensure_ascii)
            });
        }
    }
    // The template value is written straight into its JSON text: its serde
    // form is what a `serde_json::Value` built from it would serialize as
    // (the same `Serialize` impl drives both, in the same key order), so
    // the tree is built only when the keys are to be sorted, which needs
    // the value as a whole.
    let json_str = if sort_keys {
        let json_value: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
            MinijinjaError::new(
                ErrorKind::InvalidOperation,
                format!("Failed to convert to JSON value: {e}"),
            )
        })?;
        serialize_with_python_json(
            &sort_json_keys(&json_value),
            indent.as_deref(),
            separators,
            ensure_ascii,
        )
    } else {
        serialize_with_python_json(&value, indent.as_deref(), separators, ensure_ascii)
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

    // Jinja2 writes `{{ x }}` as Python's `str(x)`, so a list or a dict comes
    // out as its repr; minijinja would write it in Rust's debug form (see
    // `python_formatter`).
    env.set_formatter(python_formatter);

    // Register the template with owned storage (no lifetime dependency on caller)
    env.add_template_owned("chat".to_owned(), template)
        .map_err(|e| anyhow!("Failed to add template: {e}"))?;

    // Enable Python method compatibility (e.g., str.startswith, str.endswith)
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Register custom tojson filter compatible with HuggingFace transformers
    // This overrides minijinja's built-in tojson to support additional kwargs
    // like ensure_ascii, separators, and sort_keys that HuggingFace templates use
    env.add_filter("tojson", tojson_filter);
    // Jinja2's `string` and `trim` filters start with `str(value)` too: a
    // value `spelled_by_python` (a sequence, a map, a float) is written as
    // Python spells it.
    env.add_filter("string", python_string_filter);
    env.add_filter("trim", python_trim_filter);
    // So do `join` (of every item), `replace`, `upper`, `lower`,
    // `capitalize` and `title`. Two remain with the engine: the `~` operator
    // concatenates inside it (minijinja's own spelling of a list or a dict)
    // and cannot be hooked from here; `format` reads its format string, a
    // string in any template that uses it, and writes its arguments inside
    // the engine too.
    env.add_filter("join", python_join_filter);
    env.add_filter("replace", python_replace_filter);
    env.add_filter("upper", python_upper_filter);
    env.add_filter("lower", python_lower_filter);
    env.add_filter("capitalize", python_capitalize_filter);
    env.add_filter("title", python_title_filter);
    env.add_function("raise_exception", raise_exception);
    // Jinja2's dict() also builds a map from key/value pairs; minijinja's takes a mapping only
    env.add_function("dict", dict_function);
    env.add_test("iterable", is_iterable);

    Ok(env)
}

/// `str(value)` as Jinja2's string filters apply it: a string is itself, a
/// value `spelled_by_python` (a sequence, a map, a float) is written as
/// Python spells it (see `python_formatter`), anything else keeps
/// minijinja's spelling.
fn python_str(value: &Value) -> Cow<'_, str> {
    match value.kind() {
        ValueKind::String => Cow::Borrowed(value.as_str().unwrap_or_default()),
        _ if spelled_by_python(value) => {
            let mut text = String::new();
            // Writing into a String cannot fail.
            let _ = write_python_repr(&mut text, value);
            Cow::Owned(text)
        }
        _ => Cow::Owned(value.to_string()),
    }
}

/// Whether `str(value)` differs from the template engine's own spelling: a
/// sequence or a map (Python's repr of it) and a float (`1e-05`, `1e+16`,
/// `nan`, where the engine writes `0.00001`, `10000000000000000.0`, `NaN`).
/// An integer, a string, `True` / `False` / `None` spell alike.
fn spelled_by_python(value: &Value) -> bool {
    match value.kind() {
        ValueKind::Seq | ValueKind::Map => true,
        ValueKind::Number => !value.is_integer(),
        _ => false,
    }
}

/// Jinja2's `string` filter, `str(value)`: a value `spelled_by_python` comes
/// out as Python spells it, as it does from `{{ x }}`.
fn python_string_filter(value: Value) -> Value {
    match value.kind() {
        ValueKind::String => value,
        _ => Value::from(python_str(&value).into_owned()),
    }
}

/// Jinja2's `trim` filter, `str(value).strip(chars)`: the same `str` as the
/// `string` filter, then minijinja's trim.
fn python_trim_filter(value: Value, chars: Option<Cow<'_, str>>) -> String {
    let text = python_str(&value);
    match chars {
        Some(chars) => {
            let chars: Vec<char> = chars.chars().collect();
            text.trim_matches(&chars[..]).to_string()
        }
        None => text.trim().to_string(),
    }
}

/// Jinja2's `join` filter, `joiner.join(str(item) for item in value)`: an
/// item that is not a string goes through the same `str`; with
/// `attribute=`, each item's attribute (a dotted path) is joined instead.
fn python_join_filter(
    value: Value,
    joiner: Option<Cow<'_, str>>,
    kwargs: Kwargs,
) -> std::result::Result<String, MinijinjaError> {
    let attribute: Option<Cow<'_, str>> = kwargs.get("attribute")?;
    kwargs.assert_all_used()?;
    let joiner = joiner.as_deref().unwrap_or_default();
    let mut text = String::new();
    for (index, item) in value.try_iter()?.enumerate() {
        if index > 0 {
            text.push_str(joiner);
        }
        let item =
            match &attribute {
                // Jinja2's attribute getter: a dotted path, each segment an index
                // when it is a number and an attribute otherwise.
                Some(attribute) => attribute.split('.').try_fold(item, |value, segment| {
                    match segment.parse::<usize>() {
                        Ok(index) => value.get_item(&Value::from(index)),
                        Err(_) => value.get_attr(segment),
                    }
                })?,
                None => item,
            };
        text.push_str(&python_str(&item));
    }
    Ok(text)
}

/// Jinja2's `replace` filter, `str(value).replace(old, new, count)`: a
/// non-negative `count` bounds the replacements, as Python's does.
fn python_replace_filter(
    value: Value,
    old: Cow<'_, str>,
    new: Cow<'_, str>,
    count: Option<i64>,
) -> String {
    let text = python_str(&value);
    match count.and_then(|count| usize::try_from(count).ok()) {
        Some(count) => text.replacen(&*old, &new, count),
        None => text.replace(&*old, &new),
    }
}

/// Jinja2's `upper` filter, `str(value).upper()`.
fn python_upper_filter(value: Value) -> String {
    python_str(&value).to_uppercase()
}

/// Jinja2's `lower` filter, `str(value).lower()`.
fn python_lower_filter(value: Value) -> String {
    python_str(&value).to_lowercase()
}

/// Jinja2's `capitalize` filter, `str(value).capitalize()`: the first
/// character upper-cased, the rest lower-cased in the whole string's context,
/// so a sigma right after the first character takes the final form Python
/// gives it (`'ΑΣ'` is `Ας` here and `Ασ` under `title`, which slices the
/// word first). Python 3.8+ title-cases the first character: the same as
/// upper-casing it but for `ß`, the digraphs, the ligatures, Georgian and
/// the Greek letters with an iota subscript.
fn python_capitalize_filter(value: Value) -> String {
    let text = python_str(&value);
    let Some(first) = text.chars().next() else {
        return String::new();
    };
    // The whole string lower-cased keeps the first character in the sigma
    // rule's context, as CPython's `capitalize` reads it; the first
    // character's own lower-case form is then swapped for the upper-case one.
    let lowered = text.to_lowercase();
    let first_lowered = first.to_lowercase().map(char::len_utf8).sum::<usize>();
    let mut out = String::with_capacity(lowered.len());
    out.extend(first.to_uppercase());
    out.push_str(&lowered[first_lowered..]);
    out
}

/// Jinja2's `title` filter (not Python's `str.title()`): each word's first
/// character upper-cased and the rest lower-cased as one string (a word-final
/// sigma takes the final form Python's `lower()` gives it), a word starting
/// after a run of `-`, `(`, `{`, `[`, `<` or Python's `\s` (Jinja2's
/// `_word_beginning_split_re`, `([-\s({\[<]+)`; see `python_isspace`), on
/// the Python `str`.
fn python_title_filter(value: Value) -> String {
    let is_separator = |c: char| python_isspace(c) || matches!(c, '-' | '(' | '{' | '[' | '<');
    let text = python_str(&value);
    let mut out = String::with_capacity(text.len());
    let mut rest: &str = &text;
    while let Some(start) = rest.find(|c: char| !is_separator(c)) {
        let (separators, word) = rest.split_at(start);
        out.push_str(separators);
        let end = word.find(is_separator).unwrap_or(word.len());
        let mut chars = word[..end].chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(&chars.as_str().to_lowercase());
        }
        rest = &word[end..];
    }
    out.push_str(rest);
    out
}

/// The output formatter of `{{ }}`: a sequence, a map or a float is written
/// as Python's `str` spells it (see `spelled_by_python`), everything else as
/// minijinja's default formatter writes it (a string as it is, integers,
/// `True` / `False` / `None` already spell as `str` does). The `~` operator
/// and the string filters not re-registered in `build_environment` go
/// through the engine's own `Display`, which this renderer cannot hook.
/// A template that prints a message's content where the engine's chat utils
/// handed it a list of parts writes `[{'type': 'text', 'text': 'Hi'}]` into
/// the prompt, and so must this renderer; minijinja's debug form
/// (`[{"type": "text", "text": "Hi"}]`) is different bytes.
fn python_formatter(
    out: &mut Output,
    state: &State,
    value: &Value,
) -> std::result::Result<(), MinijinjaError> {
    if spelled_by_python(value) {
        write_python_repr(out, value).map_err(Into::into)
    } else {
        minijinja::escape_formatter(out, state, value)
    }
}

/// `repr(value)` of a JSON value, as the template renderer writes a value
/// `spelled_by_python` it prints (a sequence, a map, a float; see
/// `write_python_repr`): for a native encoder that must write the bytes the
/// checkpoint's template would.
pub(crate) fn python_repr(value: &JsonValue) -> String {
    let mut text = String::new();
    // Writing into a String cannot fail.
    let _ = write_python_repr(&mut text, &Value::from_serialize(value));
    text
}

/// `repr(value)` as Python writes it: strings quoted with Python's quote
/// choice and escapes, `True` / `False` / `None`, Python's float spelling,
/// `[a, b]` for a sequence and `{k: v}` for a map in insertion order. A value
/// of another kind (bytes, an iterator, a plain object) keeps minijinja's
/// spelling.
fn write_python_repr(out: &mut dyn std::fmt::Write, value: &Value) -> std::fmt::Result {
    match value.kind() {
        ValueKind::None => out.write_str("None"),
        ValueKind::Bool => out.write_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number if value.is_integer() => write!(out, "{value}"),
        ValueKind::Number => match f64::try_from(value.clone()) {
            Ok(number) => write_python_float(out, number),
            Err(_) => write!(out, "{value}"),
        },
        ValueKind::String => write_python_str(out, value.as_str().unwrap_or_default()),
        ValueKind::Seq => match value.try_iter() {
            Ok(items) => {
                out.write_str("[")?;
                for (index, item) in items.enumerate() {
                    if index > 0 {
                        out.write_str(", ")?;
                    }
                    write_python_repr(out, &item)?;
                }
                out.write_str("]")
            }
            Err(_) => write!(out, "{value}"),
        },
        ValueKind::Map => match value.as_object().and_then(|map| map.try_iter_pairs()) {
            Some(pairs) => {
                out.write_str("{")?;
                for (index, (key, item)) in pairs.enumerate() {
                    if index > 0 {
                        out.write_str(", ")?;
                    }
                    write_python_repr(out, &key)?;
                    out.write_str(": ")?;
                    write_python_repr(out, &item)?;
                }
                out.write_str("}")
            }
            None => write!(out, "{value}"),
        },
        _ => write!(out, "{value}"),
    }
}

/// `repr(float)`: the shortest digits that read back to the number, written
/// positionally when the decimal point falls within 1e-4 .. 1e16 and in
/// exponent form with a sign and at least two exponent digits beyond
/// (`1e+16`, `1e-05`), `.0` on a whole number, `nan`, `inf` and `-inf`.
fn write_python_float(out: &mut dyn std::fmt::Write, number: f64) -> std::fmt::Result {
    if number.is_nan() {
        return out.write_str("nan");
    }
    if number.is_infinite() {
        return out.write_str(if number < 0.0 { "-inf" } else { "inf" });
    }
    if number.is_sign_negative() {
        out.write_str("-")?;
    }
    // Rust's `{:e}` is the shortest round-trip form: `d.ddde[-]x`.
    let scientific = format!("{:e}", number.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // The digits before the decimal point (Python's `decpt`).
    let point = exponent + 1;
    if !(-3..=16).contains(&point) {
        out.write_str(&digits[..1])?;
        if digits.len() > 1 {
            write!(out, ".{}", &digits[1..])?;
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        return write!(out, "e{sign}{:02}", exponent.abs());
    }
    if point <= 0 {
        write!(
            out,
            "0.{}{digits}",
            "0".repeat(point.unsigned_abs() as usize)
        )
    } else if point as usize >= digits.len() {
        write!(
            out,
            "{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        )
    } else {
        let (whole, fraction) = digits.split_at(point as usize);
        write!(out, "{whole}.{fraction}")
    }
}

/// `repr(str)`: single quotes, or double quotes when the text holds a single
/// quote and no double quote; `\\`, the quote, `\t`, `\n`, `\r` escaped, and
/// every other character `str.isprintable` rejects as `\xNN`, `\uNNNN` or
/// `\UNNNNNNNN`.
fn write_python_str(out: &mut dyn std::fmt::Write, text: &str) -> std::fmt::Result {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.write_char(quote)?;
    // The text is written in runs between the characters that need an
    // escape; printable ASCII, the common case, is one run.
    let mut plain = 0;
    for (at, c) in text.char_indices() {
        let code = c as u32;
        if (0x20..0x7f).contains(&code) && c != '\\' && c != quote {
            continue;
        }
        if code >= 0x80 && python_printable(c) {
            continue;
        }
        out.write_str(&text[plain..at])?;
        match c {
            '\\' => out.write_str("\\\\")?,
            '\t' => out.write_str("\\t")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            c if c == quote => {
                out.write_char('\\')?;
                out.write_char(c)?;
            }
            _ if code < 0x100 => write!(out, "\\x{code:02x}")?,
            _ if code < 0x10000 => write!(out, "\\u{code:04x}")?,
            _ => write!(out, "\\U{code:08x}")?,
        }
        plain = at + c.len_utf8();
    }
    out.write_str(&text[plain..])?;
    out.write_char(quote)
}

/// `str.isprintable` for one character, which decides what `repr` escapes:
/// everything but the control (Cc), format (Cf), private-use (Co), separator
/// (Zs other than the space, Zl, Zp) characters and the noncharacters. An
/// unassigned code point (Cn), which Python also escapes, is written as it is.
fn python_printable(c: char) -> bool {
    let code = c as u32;
    !(c.is_control()
        || matches!(
            code,
            0xa0 | 0x1680 | 0x2000..=0x200a | 0x202f | 0x205f | 0x3000
        )
        || matches!(code, 0x2028 | 0x2029)
        || matches!(
            code,
            0xad | 0x600..=0x605
                | 0x61c
                | 0x6dd
                | 0x70f
                | 0x890
                | 0x891
                | 0x8e2
                | 0x180e
                | 0x200b..=0x200f
                | 0x202a..=0x202e
                | 0x2060..=0x2064
                | 0x2066..=0x206f
                | 0xfeff
                | 0xfff9..=0xfffb
                | 0x110bd
                | 0x110cd
                | 0x13430..=0x1343f
                | 0x1bca0..=0x1bca3
                | 0x1d173..=0x1d17a
                | 0xe0001
                | 0xe0020..=0xe007f
        )
        || matches!(code, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)
        || matches!(code, 0xfdd0..=0xfdef)
        || (code & 0xfffe) == 0xfffe)
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

/// Whether the template mentions the `root` role at all (the MiniMax M3
/// template reads a leading `root` message into its high-priority system
/// slot; the vendor's earlier templates read `system` only): a template that
/// never names it has no branch for it and would drop the message, so the
/// renderer rewrites root messages to system ones.
fn detect_root_role_support(template: &str) -> bool {
    template.contains("\"root\"") || template.contains("'root'")
}

/// The messages with every `developer` message rewritten as a `system`
/// message without its `tools` field (the engine's `_convert_developer_to_system`),
/// then the system messages merged into one at the front when a system
/// message is not the first message (the engine's `_consolidate_system_messages`,
/// which follows the rewrite in its renderer). `None` when no message has
/// the role, so the caller keeps its slice.
pub(crate) fn developer_messages_as_system(
    messages: &[serde_json::Value],
) -> Option<Vec<serde_json::Value>> {
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

/// The messages with every `root` message rewritten as a `system` message,
/// then the system messages merged into one at the front when a system
/// message is not the first message (the rule of
/// [`developer_messages_as_system`]), so a template that reads one leading
/// system message keeps a system message that followed the root. `None`
/// when no message has the role, so the caller keeps its slice.
pub(crate) fn root_messages_as_system(
    messages: &[serde_json::Value],
) -> Option<Vec<serde_json::Value>> {
    let is_root = |m: &serde_json::Value| m.get("role").and_then(|r| r.as_str()) == Some("root");
    if !messages.iter().any(is_root) {
        return None;
    }
    let rewritten = messages
        .iter()
        .map(|m| {
            if !is_root(m) {
                return m.clone();
            }
            let mut rewritten = m.clone();
            if let Some(fields) = rewritten.as_object_mut() {
                fields.insert(
                    "role".to_string(),
                    serde_json::Value::String("system".to_string()),
                );
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
    iter::once(merged).chain(others).collect()
}

/// The marker transformers appends to the continued text before rendering;
/// the prompt is cut where it lands.
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Python's `str.isspace` (and `re`'s `\s` on a `str` pattern), which
/// transformers' `strip`/`rstrip`, the checkpoints' whitespace runs and
/// Jinja2's `title` go by: Unicode white space plus the separators
/// U+001C..=U+001F.
pub(crate) fn python_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The marker for one render: transformers' own, unless the request already
/// carries its words (in a message, a tool, a document or a template kwarg);
/// then the words followed by one more underscore than ever follows them
/// there. Either way only the final message carries the marker.
fn continuation_marker(messages: &[JsonValue], params: &ChatTemplateParams) -> Cow<'static, str> {
    let words = CONTINUE_FINAL_MESSAGE_TAG.trim_end();
    fn longest_run(text: &str, words: &str, longest: &mut Option<usize>) {
        for (at, _) in text.match_indices(words) {
            let run = text[at + words.len()..]
                .bytes()
                .take_while(|&b| b == b'_')
                .count();
            *longest = Some(longest.map_or(run, |n| n.max(run)));
        }
    }
    fn walk(value: &JsonValue, words: &str, longest: &mut Option<usize>) {
        match value {
            JsonValue::String(text) => longest_run(text, words, longest),
            JsonValue::Array(items) => items.iter().for_each(|item| walk(item, words, longest)),
            JsonValue::Object(fields) => fields.iter().for_each(|(key, item)| {
                longest_run(key, words, longest);
                walk(item, words, longest);
            }),
            _ => {}
        }
    }
    let mut longest = None;
    for value in messages
        .iter()
        .chain(params.tools.into_iter().flatten())
        .chain(params.documents.into_iter().flatten())
    {
        walk(value, words, &mut longest);
    }
    for (key, value) in params.template_kwargs.into_iter().flatten() {
        longest_run(key, words, &mut longest);
        walk(value, words, &mut longest);
    }
    match longest {
        Some(n) => Cow::Owned(format!("{words}{} ", "_".repeat(n + 1))),
        None => Cow::Borrowed(CONTINUE_FINAL_MESSAGE_TAG),
    }
}

/// The text the final message is continued from: its string content, or
/// the text of its last part that has one.
fn final_text(message: &JsonValue) -> Option<&str> {
    match message.get("content")? {
        JsonValue::Array(parts) => parts.iter().rev().find_map(|part| part.get("text")),
        content => Some(content),
    }?
    .as_str()
}

/// Append `marker` to the text the final message is continued from.
fn tag_final_message(message: &mut JsonValue, marker: &str) -> Result<()> {
    let text = match message.get_mut("content") {
        Some(JsonValue::Array(parts)) => parts.iter_mut().rev().find_map(|p| p.get_mut("text")),
        content => content,
    };
    let Some(JsonValue::String(text)) = text else {
        return Err(anyhow!("the final message has no text to continue"));
    };
    text.push_str(marker);
    Ok(())
}

/// Cut a rendering of the tagged conversation right after the continued
/// text: at `marker`, and also past the whitespace before it when the
/// template trimmed the marker's trailing space. The text must end right
/// there (whitespace aside), so the cut belongs to the final message;
/// transformers only checks that the text appears somewhere in the prompt.
fn cut_at_final_message(mut rendered: String, text: &str, marker: &str) -> Result<String> {
    let tag_at = rendered
        .rfind(marker.trim_end())
        .filter(|&at| {
            rendered[..at]
                .trim_end_matches(python_isspace)
                .ends_with(text.trim_matches(python_isspace))
        })
        .ok_or_else(|| anyhow!("the final message does not appear in the rendered chat"))?;
    let cut = if rendered[tag_at..].starts_with(marker) {
        tag_at
    } else {
        rendered[..tag_at].trim_end_matches(python_isspace).len()
    };
    rendered.truncate(cut);
    Ok(rendered)
}

/// Render with the final message left open, following transformers'
/// `continue_final_message`: mark the end of the message's text, render the
/// whole conversation without a generation prompt, and cut at the marker.
///
/// Where transformers raises instead (no text to continue, the template
/// fails on the message, or the text never reaches the prompt, as with a
/// template that renders no assistant turns), or would cut anywhere but
/// right after the final message's text, the message is left out and
/// its text appended after the generation prompt, which is how every
/// continued message rendered before. Speech-recognition prompts rely on
/// this: their forced-language prefill follows a generation prompt the
/// template opens but whose turn it never renders. Only the text survives
/// that, so a message that also has `reasoning_content` is refused there
/// instead. The flag says which happened: true when the message was
/// continued.
fn render_continuing_final_message(
    env: &Environment<'_>,
    messages: &[JsonValue],
    params: ChatTemplateParams,
) -> Result<(String, bool)> {
    continue_final_message(messages, params, |messages, params| {
        render_messages(env, messages.iter(), params).map(|(text, _)| text)
    })
}

/// Continue a message with the same marker, cut and fallback rules for both
/// Jinja and native renderers. The supplied renderer receives no continuation flag.
pub(crate) fn continue_final_message(
    messages: &[JsonValue],
    params: ChatTemplateParams,
    render: impl Fn(&[JsonValue], ChatTemplateParams) -> Result<String>,
) -> Result<(String, bool)> {
    if params.add_generation_prompt {
        return Err(anyhow!(
            "continue_final_message and add_generation_prompt are not compatible"
        ));
    }
    let Some((last, head)) = messages.split_last() else {
        return Err(anyhow!("no final message to continue"));
    };
    let marker = continuation_marker(messages, &params);
    let continued = (|| {
        let text =
            final_text(last).ok_or_else(|| anyhow!("the final message has no text to continue"))?;
        let mut tagged = last.clone();
        tag_final_message(&mut tagged, &marker)?;
        let params = ChatTemplateParams {
            continue_final_message: false,
            ..params
        };
        let mut tagged_messages = head.to_vec();
        tagged_messages.push(tagged);
        let rendered = render(&tagged_messages, params)?;
        cut_at_final_message(rendered, text, &marker)
    })();
    let reason = match continued {
        Ok(prompt) => return Ok((prompt, true)),
        Err(reason) => reason,
    };
    if last
        .get("reasoning_content")
        .and_then(JsonValue::as_str)
        .is_some_and(|reasoning| !reasoning.is_empty())
    {
        return Err(anyhow!(
            "{reason}: only its text could follow the generation prompt, and its \
             reasoning_content would be lost"
        ));
    }
    tracing::debug!(%reason, "appending the continued message after the generation prompt");
    let mut prompt = render(
        head,
        ChatTemplateParams {
            add_generation_prompt: true,
            continue_final_message: false,
            ..params
        },
    )?;
    match last.get("content") {
        Some(JsonValue::String(text)) => prompt.push_str(text),
        Some(JsonValue::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text")?.as_str())
            .for_each(|text| prompt.push_str(text)),
        _ => {}
    }
    Ok((prompt, false))
}

/// Render the chat, and say whether the prompt continues the final message
/// (see [`render_continuing_final_message`]).
fn render_chat_template(
    env: &Environment<'_>,
    messages: &[serde_json::Value],
    params: ChatTemplateParams,
) -> Result<(String, bool)> {
    if params.continue_final_message {
        return render_continuing_final_message(env, messages, params);
    }
    render_messages(env, messages.iter(), params)
}

/// Render `messages` through the template with the generation prompt the
/// params ask for; the flag says the prompt does not continue a message.
fn render_messages<'a>(
    env: &Environment<'_>,
    messages: impl Iterator<Item = &'a JsonValue>,
    params: ChatTemplateParams,
) -> Result<(String, bool)> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("Failed to get template: {e}"))?;

    // transformers adds `strftime_now` to the template environment; here it is
    // a callable of the render context, bound to the render's instant.
    let now = params.now.unwrap_or_else(render_instant);
    let strftime_now = Value::from_function(move |format: String| strftime(&now, &format));

    // Convert messages to minijinja::Value (messages already processed by router)
    let minijinja_messages: Vec<Value> = messages.map(Value::from_serialize).collect();

    // transformers renders with `tools=None` and `documents=None` when the
    // request carries none: the names are defined and hold none. A template
    // may test them either way (`tools is none`, `tools is defined and tools`,
    // `tools is iterable`); undefined would send a template that tests
    // `is none` down its tools branch with nothing to write. A caller that
    // converted the tools once hands the value in; else they are converted here.
    let tools_value = match params.tools {
        Some(tools) => params
            .prepared_tools
            .and_then(|prepared| prepared.value_of(tools))
            .unwrap_or_else(|| Value::from_serialize(tools)),
        None => Value::from(()),
    };
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

    Ok((rendered, false))
}

/// The messages with every string `content` of a non-tool message replaced by
/// a one-item text part list, or `None` when no message has one.
pub(crate) fn string_content_as_text_parts(
    messages: &[serde_json::Value],
) -> Option<Vec<serde_json::Value>> {
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
        render_chat_template(&self.env, messages, params).map(|(prompt, _)| prompt)
    }
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

    // A plain .jinja file is the template as written: transformers' tokenizer
    // reads a checkpoint's `chat_template.jinja` whole, and so does the
    // `--chat-template` loader of the engine the render fixtures are recorded
    // against, and neither translates its text: a newline the template opens
    // with reaches the prompt (`\n<|im_start|>system`), and a backslash
    // followed by an `n` in the file's text stays the two characters it is
    // (only inside a string literal does the template engine's own escape
    // make a newline of it, in Jinja2 and here alike), so a checkpoint whose
    // file is one escaped line renders as the engine renders it. An engine's
    // own HTTP server may load an operator-supplied file with a loader of its
    // own; this renderer follows transformers. The template engine drops
    // exactly one trailing newline of the source, as Jinja2 does. A file
    // written for a loader that unescaped the pair is logged once here,
    // since nothing else at runtime would say why its prompts carry it.
    if text_outside_tags_holds_backslash_n(&content) {
        tracing::warn!(
            path = %template_path,
            "chat template text contains a literal backslash-n outside its tags; it is rendered as written, as transformers does"
        );
    }
    Ok(Some(content))
}

/// Whether the template's text outside its `{{ }}`, `{% %}` and `{# #}` tags
/// holds a backslash followed by an `n`: the pair a loader that unescaped it
/// (an engine's own `--chat-template` loader) turned into a newline and this
/// loader renders as written. Inside a tag the template engine's own string
/// escape makes the newline, as in Jinja2, so the pair there is not counted.
fn text_outside_tags_holds_backslash_n(template: &str) -> bool {
    let bytes = template.as_bytes();
    let mut close: Option<&[u8]> = None;
    let mut i = 0;
    while i < bytes.len() {
        let rest = &bytes[i..];
        match close {
            Some(end) if rest.starts_with(end) => {
                close = None;
                i += end.len();
            }
            Some(_) => i += 1,
            None if rest.starts_with(b"{{") => {
                close = Some(b"}}");
                i += 2;
            }
            None if rest.starts_with(b"{%") => {
                close = Some(b"%}");
                i += 2;
            }
            None if rest.starts_with(b"{#") => {
                close = Some(b"#}");
                i += 2;
            }
            None if rest.starts_with(b"\\n") => return true,
            None => i += 1,
        }
    }
    false
}

/// The chat templates a `tokenizer_config.json` `chat_template` value holds:
/// a string is the one template; a list of named templates (`[{"name":
/// "default", "template": ...}, {"name": "tool_use", "template": ...}]`) is
/// read as transformers reads it into a dict, and the engine's chat utils
/// take `tool_use` when the request declares tools and `default` otherwise
/// (`get_chat_template`). A later entry of a name replaces an earlier one;
/// any other value is no template.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ConfigChatTemplates {
    pub default: Option<String>,
    pub tool_use: Option<String>,
}

impl ConfigChatTemplates {
    pub fn read(value: Option<&serde_json::Value>) -> Self {
        match value {
            Some(serde_json::Value::String(template)) => Self {
                default: Some(template.clone()),
                tool_use: None,
            },
            Some(serde_json::Value::Array(entries)) => {
                let mut templates = Self::default();
                for entry in entries {
                    let name = entry.get("name").and_then(serde_json::Value::as_str);
                    let template = entry.get("template").and_then(serde_json::Value::as_str);
                    match (name, template) {
                        (Some("default"), Some(template)) => {
                            templates.default = Some(template.to_string());
                        }
                        (Some("tool_use"), Some(template)) => {
                            templates.tool_use = Some(template.to_string());
                        }
                        _ => {}
                    }
                }
                templates
            }
            _ => Self::default(),
        }
    }

    /// Whether the value names a template at all.
    pub fn is_some(&self) -> bool {
        self.default.is_some() || self.tool_use.is_some()
    }

    /// Whether the value names no template.
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }
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
    /// Whether the generation prompt opens a `<think>` block it does not
    /// close, so a completion starts inside reasoning; a closed block (a
    /// non-thinking template) does not count.
    prefill_opens_think_block: bool,
    /// Whether the template has a branch for the `developer` role. When it
    /// has none, `apply` renders developer messages as system messages, as
    /// the engine's HF renderer does, instead of letting the template drop them.
    developer_role_supported: bool,
    /// Whether the template has a branch for the `root` role (the MiniMax M3
    /// template). When it has none, `apply` renders root messages as system
    /// messages, the way such a request was dispatched before the MiniMax
    /// profile kept the role, instead of letting the template drop them.
    root_role_supported: bool,
    /// Whether the template writes a history tool call's `arguments` as the
    /// text the request sent (see [`Self::raw_tool_call_arguments`]).
    raw_tool_call_arguments: bool,
    /// The `tool_use` template of a configuration that names its templates
    /// (see [`ConfigChatTemplates`]), rendered in place of the default one
    /// when the request declares tools, as the engine's chat utils render.
    /// The accessors answer for the default template.
    tool_use: Option<Box<ChatTemplateState>>,
    /// The processor's template of a checkpoint that ships one beside the
    /// tokenizer's and differs from it (see [`Self::from_checkpoint`]),
    /// rendered in place of the default one when the request declares no
    /// tools, as the engine's server resolves a request's template: the
    /// request's own first, the processor's second unless tools are
    /// declared, the tokenizer's third. The accessors answer for the default
    /// template, which the processor's agrees with on the thinking answers;
    /// [`Self::content_format_for_tools`] answers for the one selected.
    processor: Option<Box<ChatTemplateState>>,
}

impl std::fmt::Debug for ChatTemplateState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplateState")
            .field("has_template", &self.env.is_some())
            .field("content_format", &self.content_format)
            .field("thinking_toggle", &self.thinking_toggle)
            .field("think_in_prefill", &self.think_in_prefill)
            .field("has_tool_use", &self.tool_use.is_some())
            .field("has_processor", &self.processor.is_some())
            .finish()
    }
}

impl ChatTemplateState {
    pub fn new(template: Option<String>) -> Result<Self> {
        let (
            content_format,
            think_in_prefill,
            thinking_toggle,
            thinking_key_name,
            prefill_opens_think_block,
        ) = template.as_ref().map(|t| detect_all(t)).unwrap_or_default();
        let developer_role_supported = template
            .as_deref()
            .is_none_or(detect_developer_role_support);
        let root_role_supported = template.as_deref().is_none_or(detect_root_role_support);
        let env = template.map(build_environment).transpose()?;
        let mut state = Self {
            env,
            content_format,
            thinking_toggle,
            thinking_key_name,
            think_in_prefill,
            prefill_opens_think_block,
            developer_role_supported,
            root_role_supported,
            raw_tool_call_arguments: false,
            tool_use: None,
            processor: None,
        };
        state.raw_tool_call_arguments = state.writes_tool_call_arguments_as_text();
        Ok(state)
    }

    /// The templates of a configuration (see [`ConfigChatTemplates`]): the
    /// default one, and the `tool_use` one for requests that declare tools.
    pub fn from_config(templates: ConfigChatTemplates) -> Result<Self> {
        let mut state = Self::new(templates.default)?;
        if let Some(template) = templates.tool_use {
            let tool_use = Self::new(Some(template))?;
            // A request is prepared (its content shape, its thinking switch)
            // and its answer read (the prefill's think block) by the default
            // template's answers before the template is chosen; a pair that
            // disagrees on them is prepared for the wrong one when tools are
            // declared, so say so once at load, naming every answer tested.
            if tool_use.content_format != state.content_format
                || tool_use.thinking_toggle != state.thinking_toggle
                || tool_use.thinking_key_name != state.thinking_key_name
                || tool_use.think_in_prefill != state.think_in_prefill
                || tool_use.prefill_opens_think_block != state.prefill_opens_think_block
            {
                tracing::warn!(
                    default_content_format = ?state.content_format,
                    tool_use_content_format = ?tool_use.content_format,
                    default_thinking_toggle = ?state.thinking_toggle,
                    tool_use_thinking_toggle = ?tool_use.thinking_toggle,
                    default_thinking_key_name = ?state.thinking_key_name,
                    tool_use_thinking_key_name = ?tool_use.thinking_key_name,
                    default_think_in_prefill = state.think_in_prefill,
                    tool_use_think_in_prefill = tool_use.think_in_prefill,
                    default_prefill_opens_think_block = state.prefill_opens_think_block,
                    tool_use_prefill_opens_think_block = tool_use.prefill_opens_think_block,
                    "the named chat templates disagree on how a request is prepared; \
                     requests are prepared by the default template's answers"
                );
            }
            state.tool_use = Some(Box::new(tool_use));
        }
        Ok(state)
    }

    /// The templates of a checkpoint: the configuration's (see
    /// [`Self::from_config`]) and the processor's, the legacy
    /// `chat_template.json` a multimodal checkpoint may ship beside the
    /// tokenizer's template. The engine's server renders a request that
    /// declares no tools through the processor's template when the checkpoint
    /// has one, and one that declares tools through the tokenizer's, so the
    /// processor's is kept for the tool-less requests where it differs from
    /// the default template; one equal to it, or one without a default
    /// template to differ from (the loaders take the legacy file as the one
    /// template of such a configuration), changes nothing.
    ///
    /// The processor's template is kept only when it agrees with the default
    /// one on how a request's thinking is prepared and its answer read (the
    /// toggle and its key, the prefill's think block) and on how a history
    /// tool call's arguments are written: those answers are read from the
    /// default template before the request's template is chosen, so a
    /// processor template that disagrees would render a prompt the parser is
    /// not set up for. Such a template, like one that does not compile, is
    /// not used, every request renders through the tokenizer's templates as
    /// before, and the load says why. The content format may differ: it
    /// follows the selected template ([`Self::content_format_for_tools`]).
    pub fn from_checkpoint(
        templates: ConfigChatTemplates,
        processor: Option<String>,
    ) -> Result<Self> {
        let processor = processor.filter(|template| {
            templates
                .default
                .as_deref()
                .is_some_and(|default| default != template)
        });
        let mut state = Self::from_config(templates)?;
        if let Some(template) = processor {
            match Self::new(Some(template)) {
                Ok(processor) if state.prepares_thinking_as(&processor) => {
                    state.processor = Some(Box::new(processor));
                }
                Ok(processor) => tracing::warn!(
                    default_thinking_toggle = ?state.thinking_toggle,
                    processor_thinking_toggle = ?processor.thinking_toggle,
                    default_thinking_key_name = ?state.thinking_key_name,
                    processor_thinking_key_name = ?processor.thinking_key_name,
                    default_think_in_prefill = state.think_in_prefill,
                    processor_think_in_prefill = processor.think_in_prefill,
                    default_prefill_opens_think_block = state.prefill_opens_think_block,
                    processor_prefill_opens_think_block = processor.prefill_opens_think_block,
                    default_raw_tool_call_arguments = state.raw_tool_call_arguments(),
                    processor_raw_tool_call_arguments = processor.raw_tool_call_arguments(),
                    "the checkpoint's processor chat template disagrees with the tokenizer's on \
                     how a request's thinking is prepared and read or on how a history tool \
                     call's arguments are written, and is not used; requests render through the \
                     tokenizer's template"
                ),
                Err(e) => tracing::warn!(
                    "the checkpoint's processor chat template does not compile and is not used; \
                     requests render through the tokenizer's template: {e}"
                ),
            }
        }
        Ok(state)
    }

    /// Whether `other` prepares a request's thinking, reads its answer and
    /// writes a history tool call's arguments as this template does (see
    /// [`Self::from_checkpoint`]).
    fn prepares_thinking_as(&self, other: &Self) -> bool {
        other.thinking_toggle == self.thinking_toggle
            && other.thinking_key_name == self.thinking_key_name
            && other.think_in_prefill == self.think_in_prefill
            && other.prefill_opens_think_block == self.prefill_opens_think_block
            // The answer the gateway reads: a named pair's comes from its
            // `tool_use` template (see [`Self::raw_tool_call_arguments`]).
            && other.raw_tool_call_arguments == self.raw_tool_call_arguments()
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
            prefill_opens_think_block: false,
            developer_role_supported: true,
            root_role_supported: true,
            raw_tool_call_arguments: false,
            tool_use: None,
            processor: None,
        }
    }

    pub fn apply(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.apply_with_continuation(messages, params)
            .map(|(prompt, _)| prompt)
    }

    /// [`Self::apply`], also saying whether the prompt continues the final
    /// message: false unless `continue_final_message` was asked for and the
    /// template could continue the message instead of appending its text
    /// after the generation prompt (see
    /// [`ChatTemplateParams::continue_final_message`]).
    pub fn apply_with_continuation(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
        self.template_for(&params)
            .render_with_continuation(messages, params)
    }

    /// The template a request renders through (see
    /// [`Self::template_for_tools`]).
    fn template_for(&self, params: &ChatTemplateParams) -> &Self {
        self.template_for_tools(params.tools.is_some())
    }

    /// The template a request renders through: the `tool_use` one of a
    /// configuration that names it when the request declares tools, the
    /// processor's one of a checkpoint that ships it when the request
    /// declares none, else this one.
    fn template_for_tools(&self, has_tools: bool) -> &Self {
        let selected = if has_tools {
            &self.tool_use
        } else {
            &self.processor
        };
        selected.as_deref().unwrap_or(self)
    }

    /// The content format of the template a request renders through
    /// (`template_for_tools`): what the request's message content is
    /// shaped into before the render, as the engine's server detects it on
    /// the template it resolved for the request.
    pub fn content_format_for_tools(&self, has_tools: bool) -> ChatTemplateContentFormat {
        self.template_for_tools(has_tools).content_format
    }

    fn render_with_continuation(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
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

        // A template without a `root` branch (the MiniMax templates before
        // M3) has no branch that reads the role and would drop a root
        // message: it gets the message as the leading system message
        // instead, the way such a request was dispatched before the MiniMax
        // profile kept the role, and a system message after the root merges
        // into that one rather than being lost (the consolidation below).
        // Before the developer step, so a developer message merged next
        // comes after the root's text.
        let root_converted;
        let messages: &[serde_json::Value] = if self.root_role_supported {
            messages
        } else {
            match root_messages_as_system(messages) {
                Some(rewritten) => {
                    root_converted = rewritten;
                    &root_converted
                }
                None => messages,
            }
        };

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
        let (
            content_format,
            think_in_prefill,
            thinking_toggle,
            thinking_key_name,
            prefill_opens_think_block,
        ) = detect_all(&template);
        let developer_role_supported = detect_developer_role_support(&template);
        let root_role_supported = detect_root_role_support(&template);
        let env = build_environment(template)?;
        self.developer_role_supported = developer_role_supported;
        self.root_role_supported = root_role_supported;
        self.content_format = content_format;
        self.thinking_toggle = thinking_toggle;
        self.thinking_key_name = thinking_key_name;
        self.think_in_prefill = think_in_prefill;
        self.prefill_opens_think_block = prefill_opens_think_block;
        self.env = Some(env);
        // A replacement template stands alone, as a template file does.
        self.tool_use = None;
        self.processor = None;
        self.raw_tool_call_arguments = self.writes_tool_call_arguments_as_text();
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

    /// Whether the generation prompt opens a `<think>` block it does not
    /// close (see `Tokenizer::prefill_opens_think_block`).
    pub fn prefill_opens_think_block(&self) -> bool {
        self.prefill_opens_think_block
    }

    /// Whether the template writes a history tool call's `arguments` as the
    /// text the request sent: the Jinja renderer's answer to
    /// `RendererCapabilities::raw_tool_call_arguments`. The API carries the
    /// arguments as a JSON string and the gateway parses them into the object
    /// the Transformers docs say a template reads; a template that writes
    /// them by string concatenation of `'<sep>'`, `tool['function']['arguments']`
    /// and `'<end>'` cannot render that object, so the string must reach it as
    /// written. Resolved when the template is bound (see
    /// `Self::writes_tool_call_arguments_as_text`).
    pub fn raw_tool_call_arguments(&self) -> bool {
        // In a named pair the `tool_use` template is the one with the
        // tool-call branch, and every request that renders a tool call goes
        // through it; the default template (plain chat, as a rule) has no
        // answer of its own.
        self.tool_use
            .as_ref()
            .map_or(self.raw_tool_call_arguments, |tool_use| {
                tool_use.raw_tool_call_arguments
            })
    }

    /// Decide [`Self::raw_tool_call_arguments`] by the template's behaviour,
    /// not its source: one history call rendered with the arguments as the
    /// string and as the object. The template takes the text only when the
    /// object fails to render and the string renders, so a template that
    /// takes the object (`tojson`, indexing, an `is string` branch) and one
    /// that renders neither keep the parsed form the gateway hands every
    /// template today.
    fn writes_tool_call_arguments_as_text(&self) -> bool {
        if self.env.is_none() {
            return false;
        }
        let renders = |arguments: serde_json::Value| {
            let messages = [
                serde_json::json!({"role": "user", "content": "probe"}),
                serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "probe", "arguments": arguments},
                }]}),
                serde_json::json!({"role": "tool", "tool_call_id": "call_0", "content": "probe"}),
            ];
            self.apply(&messages, ChatTemplateParams::default()).is_ok()
        };
        renders(serde_json::json!("{\"probe\": 1}")) && !renders(serde_json::json!({"probe": 1}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A history call's `arguments` reach a template as the text the request
    /// sent only when the template writes them that way: one that concatenates
    /// them cannot render the parsed object; one that serializes or indexes
    /// the object, or branches on its type, and one without a tool-call branch
    /// keep the object.
    #[test]
    fn a_template_that_concatenates_tool_call_arguments_takes_them_as_text() {
        let state = ChatTemplateState::new(Some(CONCATENATING_CALLS.to_string())).unwrap();
        assert!(state.raw_tool_call_arguments());

        let indexing = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                        <call>{{ t.function.name }}:{% for k, v in t.function.arguments | items %}\
                        {{ k }}={{ v }}{% endfor %}</call>{% endfor %}{% else %}{{ m.content }}\
                        {% endif %}{% endfor %}";
        let branching = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                         <call>{{ t.function.name }}:{% if t.function.arguments is string %}\
                         {{ t.function.arguments }}{% else %}{{ t.function.arguments | tojson }}\
                         {% endif %}</call>{% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
        let without_calls = "{% for m in messages %}{{ m.content }}{% endfor %}";
        for template in [SERIALIZING_CALLS, indexing, branching, without_calls] {
            let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
            assert!(!state.raw_tool_call_arguments(), "{template}");
        }
        assert!(!ChatTemplateState::empty().raw_tool_call_arguments());
        assert!(!ChatTemplateState::new(None)
            .unwrap()
            .raw_tool_call_arguments());
    }

    /// A replacement template answers for itself.
    #[test]
    fn a_replacement_template_refreshes_the_tool_call_arguments_answer() {
        let mut state = ChatTemplateState::new(Some(SERIALIZING_CALLS.to_string())).unwrap();
        assert!(!state.raw_tool_call_arguments());
        state.set(CONCATENATING_CALLS.to_string()).unwrap();
        assert!(state.raw_tool_call_arguments());
        state.set(SERIALIZING_CALLS.to_string()).unwrap();
        assert!(!state.raw_tool_call_arguments());
    }

    /// A template that writes a history call by string concatenation, as the
    /// DeepSeek V3 line's templates do.
    const CONCATENATING_CALLS: &str =
        "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
         {{ '<call>' + t.function.name + ':' + t.function.arguments + '</call>' }}\
         {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
    /// A template that serializes the arguments object, as most do.
    const SERIALIZING_CALLS: &str =
        "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
         <call>{{ t.function.name }}:{{ t.function.arguments | tojson }}</call>\
         {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";

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

    /// A template that keeps the messages it loops over in a namespace
    /// attribute (`ns.rest = messages[1:]`) and renders each message's
    /// content through a macro that walks its parts is of the "openai"
    /// format: its parts reach the template, which writes the media anchors
    /// itself. Any other non-name assignment target still falls back to the
    /// "string" format.
    #[test]
    fn namespace_message_lists_are_followed_to_their_content_loop() {
        let template = "{%- macro render_content(content) -%}\
                        {%- if content is string -%}{{ content }}\
                        {%- else -%}{%- for part in content -%}\
                        {%- if part.get('type', '') in ['image', 'image_url'] -%}<|patch|>\
                        {%- elif part.get('type', '') == 'text' -%}{{ part.get('text', '') }}\
                        {%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}\
                        {%- set ns = namespace(rest=messages) -%}\
                        {%- if messages and messages[0]['role'] == 'system' -%}\
                        {%- set ns.rest = messages[1:] -%}{%- endif -%}\
                        {%- for message in ns.rest -%}\
                        <|start|>{{ message['role'] }}<|message|>{{ render_content(message['content']) }}<|eot|>\
                        {%- endfor -%}";
        assert_eq!(
            detect_chat_template_content_format(template),
            ChatTemplateContentFormat::OpenAI
        );
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let messages = [serde_json::json!({"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "what is this?"}
        ]})];
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<|start|>user<|message|><|patch|>what is this?<|eot|>"
        );

        // A plain string-content template with the same namespace shape stays string-format.
        let plain = "{%- set ns = namespace(rest=messages) -%}{%- for message in ns.rest -%}\
                     {{ message['content'] }}{%- endfor -%}";
        assert_eq!(
            detect_chat_template_content_format(plain),
            ChatTemplateContentFormat::String
        );
    }

    /// `{{ x }}` of a list or a dict is Python's repr, the bytes the engine's
    /// prompt holds when a template prints a content value its chat utils
    /// handed over as parts; a float alone spells as Python's `str` does; a
    /// string, an integer and a flag are written as before.
    #[test]
    fn a_sequence_or_a_map_is_written_as_python_s_repr() {
        let state = ChatTemplateState::new(Some("{{ messages[0].content }}".to_string())).unwrap();
        let render = |content: serde_json::Value| {
            state
                .apply(
                    &[serde_json::json!({"role": "user", "content": content})],
                    ChatTemplateParams::default(),
                )
                .unwrap()
        };
        assert_eq!(
            render(serde_json::json!([{"type": "text", "text": "Welcome! How can I help?"}])),
            "[{'type': 'text', 'text': 'Welcome! How can I help?'}]"
        );
        // Python's quote choice and escapes (`python3 -c 'print(repr(...))'`).
        assert_eq!(
            render(serde_json::json!([
                {"type": "text", "text": "it's"},
                {"type": "text", "text": "say \"hi\""},
                {"t": "both ' and \""}
            ])),
            r#"[{'type': 'text', 'text': "it's"}, {'type': 'text', 'text': 'say "hi"'}, {'t': 'both \' and "'}]"#
        );
        assert_eq!(
            render(serde_json::json!({
                "a": null, "b": true, "c": 1.0, "d": 1e16, "e": 0.0001, "f": 1e-5,
                "g": "tab\tnl\nbs\\ nbsp\u{a0} zw\u{200b} eé 😀", "h": [1, 2.5, -3, 100.0, 0.001]
            })),
            "{'a': None, 'b': True, 'c': 1.0, 'd': 1e+16, 'e': 0.0001, 'f': 1e-05, \
             'g': 'tab\\tnl\\nbs\\\\ nbsp\\xa0 zw\\u200b eé 😀', 'h': [1, 2.5, -3, 100.0, 0.001]}"
        );
        assert_eq!(render(serde_json::json!([])), "[]");
        assert_eq!(render(serde_json::json!({})), "{}");
        assert_eq!(render(serde_json::json!("plain")), "plain");
        assert_eq!(render(serde_json::json!(2.0)), "2.0");
        // A float printed alone spells as Python's `str` does, like one in a
        // list; an integer as before.
        assert_eq!(render(serde_json::json!(1e-5)), "1e-05");
        assert_eq!(render(serde_json::json!(1e16)), "1e+16");
        assert_eq!(render(serde_json::json!(100.0)), "100.0");
        assert_eq!(render(serde_json::json!(-0.001)), "-0.001");
        assert_eq!(render(serde_json::json!(12)), "12");
        assert_eq!(render(serde_json::json!(false)), "False");
        assert_eq!(render(serde_json::json!(null)), "None");

        // The `string` and `trim` filters start with the same `str`.
        let filtered = ChatTemplateState::new(Some(
            "{{ messages[0].content | string }}|{{ messages[0].content | trim }}|\
             {{ ' x ' | trim }}|{{ 'ab' | trim('a') }}|{{ 7 | string }}|{{ 0.00001 | string }}"
                .to_string(),
        ))
        .unwrap();
        let rendered = filtered
            .apply(
                &[serde_json::json!({"role": "user", "content": [{"type": "text", "text": "Hi"}]})],
                ChatTemplateParams::default(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            "[{'type': 'text', 'text': 'Hi'}]|[{'type': 'text', 'text': 'Hi'}]|x|b|7|1e-05"
        );
    }

    /// The string filters Jinja2 starts with `str(value)` write a list or a
    /// dict as Python does, like `string` and `trim`: `join` of every item
    /// (or of an attribute of each), `replace` (with a count), `upper`,
    /// `lower`, `capitalize`, `title` (a word-final sigma and a `\x1c`
    /// separator as Python has them); a string and an integer are untouched.
    #[test]
    fn the_string_filters_start_with_python_s_str() {
        let state = ChatTemplateState::new(Some(
            "{{ messages[0].content | join(' + ') }}|\
             {{ messages[0].content | replace('text', 'T') }}|\
             {{ messages[0].content | upper }}|{{ messages[0].content[0] | lower }}|\
             {{ ['a', 'b'] | join(', ') }}|{{ 'Ab' | upper }}|{{ 7 | lower }}|\
             {{ 1.5 | replace('.', ',') }}|{{ 'x.y' | replace('.', '') }}|\
             {{ messages[0].content | capitalize }}|{{ messages[0].content | title }}|\
             {{ \"get_weather don't\" | title }}|\
             {{ messages[1].content | title }}|{{ messages[1].content | capitalize }}|\
             {{ 'ΑΣ' | title }}{{ 'ΑΣ' | capitalize }}|\
             {{ 'x.y.z' | replace('.', '-', 1) }}|{{ messages[0].content | join(', ', attribute='text') }}"
                .to_string(),
        ))
        .unwrap();
        let rendered = state
            .apply(
                &[
                    serde_json::json!({"role": "user", "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "text", "text": "Yo"}
                    ]}),
                    serde_json::json!({"role": "user", "content": "a\u{1c}b ΟΔΟΣ ΚΑΙ"}),
                ],
                ChatTemplateParams::default(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            "{'type': 'text', 'text': 'Hi'} + {'type': 'text', 'text': 'Yo'}|\
             [{'type': 'T', 'T': 'Hi'}, {'type': 'T', 'T': 'Yo'}]|\
             [{'TYPE': 'TEXT', 'TEXT': 'HI'}, {'TYPE': 'TEXT', 'TEXT': 'YO'}]|\
             {'type': 'text', 'text': 'hi'}|a, b|AB|7|1,5|xy|\
             [{'type': 'text', 'text': 'hi'}, {'type': 'text', 'text': 'yo'}]|\
             [{'type': 'text', 'text': 'hi'}, {'type': 'text', 'text': 'yo'}]|\
             Get_weather Don't|A\u{1c}B Οδος Και|A\u{1c}b οδος και|ΑσΑς|x-y.z|Hi, Yo"
        );
    }

    /// The case of the record: a template that loops over content somewhere
    /// (the openai content format) and prints `message.content` elsewhere
    /// gets a string content as a one-item part list, and prints it the way
    /// the engine's prompt holds it.
    #[test]
    fn a_parts_list_a_template_prints_comes_out_as_the_engine_writes_it() {
        let template = "{%- for m in messages %}\n{%- if m.role == 'system' %}<|system|>\n{{ m.content }}\
                        {%- else %}<|{{ m.role }}|>\n{%- for part in m.content %}{{ part.text }}{%- endfor %}\
                        {%- endif %}{%- endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.content_format(), ChatTemplateContentFormat::OpenAI);
        let rendered = state
            .apply(
                &[
                    serde_json::json!({"role": "system", "content": "You are a terse assistant."}),
                    serde_json::json!({"role": "user", "content": "Hello."}),
                ],
                ChatTemplateParams::default(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|system|>\n[{'type': 'text', 'text': 'You are a terse assistant.'}]<|user|>Hello."
        );
    }

    /// A configuration that ships its templates as a named list is read as
    /// transformers reads it: `default` renders a plain request, `tool_use`
    /// one that declares tools; a string stays the one template; a replacement
    /// template drops the pair.
    #[test]
    fn a_named_template_list_renders_tool_use_for_a_request_with_tools() {
        let config = serde_json::json!([
            {"name": "default", "template": "default:{{ messages | length }}"},
            {"name": "tool_use", "template": "tool_use:{{ tools | length }}"}
        ]);
        let templates = ConfigChatTemplates::read(Some(&config));
        assert_eq!(
            templates,
            ConfigChatTemplates {
                default: Some("default:{{ messages | length }}".to_string()),
                tool_use: Some("tool_use:{{ tools | length }}".to_string()),
            }
        );
        let mut state = ChatTemplateState::from_config(templates).unwrap();
        let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
        let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
        let with_tools = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        assert_eq!(
            state
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "default:1"
        );
        assert_eq!(state.apply(&messages, with_tools).unwrap(), "tool_use:1");

        state.set("alone:{{ tools is none }}".to_string()).unwrap();
        let with_tools = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        assert_eq!(state.apply(&messages, with_tools).unwrap(), "alone:False");

        assert_eq!(
            ConfigChatTemplates::read(Some(&serde_json::json!("one"))),
            ConfigChatTemplates {
                default: Some("one".to_string()),
                tool_use: None,
            }
        );
        assert!(
            !ConfigChatTemplates::read(Some(&serde_json::json!({"name": "default"}))).is_some()
        );
        assert!(!ConfigChatTemplates::read(None).is_some());
    }

    /// A tool set converted once renders as the values it was made of, with
    /// and without a continued final message; one made of other values is not
    /// read, and the render converts the values it was given.
    #[test]
    fn prepared_tools_render_as_the_values_they_were_made_of() {
        let state = ChatTemplateState::new(Some(
            "{% for tool in tools %}{{ tool | tojson }};{% endfor %}{{ tools | length }}:\
             {% for message in messages %}{{ message['content'] }}{% endfor %}"
                .to_string(),
        ))
        .unwrap();
        let messages = [
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "The answer is"}),
        ];
        let tools = vec![
            serde_json::json!({"type": "function", "function": {"name": "f", "description": "first",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}),
            serde_json::json!({"type": "function", "function": {"name": "g", "parameters": {}}}),
        ];
        let prepared = PreparedTools::new(tools.clone());
        assert_eq!(prepared.tools(), tools.as_slice());
        fn params<'a>(
            tools: &'a [serde_json::Value],
            prepared_tools: Option<&'a PreparedTools>,
            continue_final_message: bool,
        ) -> ChatTemplateParams<'a> {
            ChatTemplateParams {
                tools: Some(tools),
                prepared_tools,
                add_generation_prompt: !continue_final_message,
                continue_final_message,
                ..Default::default()
            }
        }
        for continued in [false, true] {
            let plain = state
                .apply(&messages, params(&tools, None, continued))
                .unwrap();
            assert!(plain.contains("\"name\": \"f\""), "{plain}");
            assert_eq!(
                state
                    .apply(
                        &messages,
                        params(prepared.tools(), Some(&prepared), continued)
                    )
                    .unwrap(),
                plain,
                "the prepared value renders as the values it was made of"
            );
            let others = [serde_json::json!({"type": "function", "function": {"name": "h"}})];
            let mismatched = state
                .apply(&messages, params(&others, Some(&prepared), continued))
                .unwrap();
            assert_eq!(
                mismatched,
                state
                    .apply(&messages, params(&others, None, continued))
                    .unwrap(),
                "a prepared value made of other tools is not read"
            );
            assert_ne!(mismatched, plain);
        }
    }

    /// A prepared tool answers every template operation as the map it holds,
    /// and its `tojson` text is the writer's under every argument shape: the
    /// kept forms (no argument, the first indent) come from the kept text on
    /// the second render, the others are written as for any value.
    #[test]
    fn a_prepared_tool_renders_as_its_map_and_keeps_its_tojson_text() {
        let template = "{% for tool in tools %}\
            {{ tool | tojson }}|{{ tool | tojson(indent=2) }}|{{ tool | tojson(indent=4) }}|\
            {{ tool | tojson(ensure_ascii=True) }}|{{ tool | tojson(sort_keys=True) }}|\
            {{ tool | tojson(separators=(',', ':')) }}|{{ tool.function | tojson }}|\
            {{ tool is mapping }}|{{ tool is iterable }}|{{ tool | length }}|{{ 'function' in tool }}|\
            {{ 'nope' in tool }}|{{ tool.function.name }}|{{ tool['type'] }}|{{ tool.get('type') }}|\
            {{ tool.get('nope', 'd') }}|{{ tool.items() | list | length }}|{{ tool.keys() | list }}|\
            {{ tool.values() | list | length }}|{{ tool == tool }}|{{ tool != tools[0] }}|{{ tool }}|\
            {{ tool | string }}|{{ tool | pprint }}|{% for k in tool %}{{ k }},{% endfor %}|\
            {% for k, v in tool.items() %}{{ k }}={{ v }};{% endfor %}|{{ tool | list }}|\
            {{ tool if tool else 'empty' }}|{{ tool | dictsort | length }}\n\
            {% endfor %}{{ tools | tojson }}|{{ tools | length }}|{{ tools[1].function.name }}|\
            {{ tools | map(attribute='type') | list }}|{{ tools | selectattr('function.name', 'eq', 'g') | list | length }}";
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let tools = vec![
            serde_json::json!({"type": "function", "function": {"name": "zoë", "description": "first ✓",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}),
            serde_json::json!({"type": "function", "function": {"name": "g", "parameters": {}}}),
        ];
        let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
        let plain = processor
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    tools: Some(&tools),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(plain.contains("\"name\": \"zoë\""), "{plain}");
        assert!(plain.contains("\"name\": \"zo\\u00eb\""), "{plain}");
        let prepared = PreparedTools::new(tools.clone());
        let params = || ChatTemplateParams {
            tools: Some(prepared.tools()),
            prepared_tools: Some(&prepared),
            ..Default::default()
        };
        let first = processor.apply_chat_template(&messages, params()).unwrap();
        assert_eq!(first, plain, "the first render writes the kept texts");
        let second = processor.apply_chat_template(&messages, params()).unwrap();
        assert_eq!(second, plain, "the second render reads them");
    }

    /// The kept text is written once per kept tool and form: the second
    /// `tojson` of a kept tool answers with the string the first wrote; a
    /// second indent is not kept and is written every time.
    #[test]
    fn a_prepared_tools_tojson_text_is_written_once_per_kept_form() {
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "f"}})];
        let prepared = PreparedTools::new(tools);
        let tool = prepared.value.get_item(&Value::from(0)).unwrap();
        let kept = tool
            .as_object()
            .and_then(|object| object.downcast_ref::<KeptTool>())
            .expect("a prepared tool is kept");
        let mut writes = 0;
        let mut write = || {
            writes += 1;
            Ok(format!("text {writes}"))
        };
        let first = kept.tojson(None, &mut write).unwrap();
        let again = kept.tojson(None, &mut write).unwrap();
        assert_eq!(first.as_str(), Some("text 1"));
        assert_eq!(
            first.as_str().unwrap().as_ptr(),
            again.as_str().unwrap().as_ptr(),
            "one string, shared"
        );
        let two = kept.tojson(Some(b"  "), &mut write).unwrap();
        assert_eq!(two.as_str(), Some("text 2"));
        assert_eq!(
            kept.tojson(Some(b"  "), &mut write).unwrap().as_str(),
            Some("text 2")
        );
        assert_eq!(
            kept.tojson(Some(b"    "), &mut write).unwrap().as_str(),
            Some("text 3")
        );
        assert_eq!(
            kept.tojson(Some(b"    "), &mut write).unwrap().as_str(),
            Some("text 4")
        );
        assert_eq!(
            kept.tojson(None, &mut write).unwrap().as_str(),
            Some("text 1")
        );
        assert_eq!(writes, 4);
    }

    /// A checkpoint's processor template renders the requests that declare no
    /// tools and the configuration's templates the ones that do, as the
    /// engine's server resolves them; the content format answers for the
    /// template selected. One equal to the default template, or one without
    /// a default template to differ from, is not kept; a replacement template
    /// drops it.
    #[test]
    fn a_checkpoint_s_processor_template_renders_a_request_without_tools() {
        let default = "default:{{ messages[0]['content'] }}";
        let processor = "processor:{% for message in messages %}\
             {% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endfor %}";
        let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
        let tools = [serde_json::json!({"type": "function", "function": {"name": "f"}})];
        let with_tools = || ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let templates = || ConfigChatTemplates {
            default: Some(default.to_string()),
            tool_use: Some("tool_use:{{ tools | length }}".to_string()),
        };

        let mut state =
            ChatTemplateState::from_checkpoint(templates(), Some(processor.to_string())).unwrap();
        assert!(format!("{state:?}").contains("has_processor: true"));
        assert_eq!(
            state
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "processor:Hi"
        );
        assert_eq!(state.apply(&messages, with_tools()).unwrap(), "tool_use:1");
        assert_eq!(
            state.content_format_for_tools(false),
            ChatTemplateContentFormat::OpenAI
        );
        assert_eq!(
            state.content_format_for_tools(true),
            ChatTemplateContentFormat::String
        );
        assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
        state.set("alone:{{ tools is none }}".to_string()).unwrap();
        assert!(format!("{state:?}").contains("has_processor: false"));
        assert_eq!(
            state
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "alone:True"
        );

        let equal =
            ChatTemplateState::from_checkpoint(templates(), Some(default.to_string())).unwrap();
        assert!(format!("{equal:?}").contains("has_processor: false"));
        assert_eq!(
            equal
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "default:Hi"
        );

        let without_default = ChatTemplateState::from_checkpoint(
            ConfigChatTemplates::default(),
            Some(processor.to_string()),
        )
        .unwrap();
        assert!(format!("{without_default:?}").contains("has_processor: false"));
        assert!(without_default
            .apply(&messages, ChatTemplateParams::default())
            .is_err());

        // A processor template that disagrees with the default one on how a
        // request's thinking is prepared (here: a toggle the default has
        // none of) is not used: the render would not match the parse.
        let thinking = "{% if enable_thinking %}<think>{% endif %}{{ messages[0]['content'] }}";
        let disagreeing =
            ChatTemplateState::from_checkpoint(templates(), Some(thinking.to_string())).unwrap();
        assert!(format!("{disagreeing:?}").contains("has_processor: false"));
        assert_eq!(
            disagreeing
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "default:Hi"
        );
        let agreeing = ChatTemplateState::from_checkpoint(
            ConfigChatTemplates {
                default: Some(thinking.to_string()),
                tool_use: None,
            },
            Some(
                "processor:{{ messages[0]['content'] }}{% if enable_thinking %}<think>{% endif %}"
                    .to_string(),
            ),
        )
        .unwrap();
        assert!(format!("{agreeing:?}").contains("has_processor: true"));

        // The raw-arguments answer compared is the one the gateway reads: in
        // a named pair the `tool_use` template's, not the default's.
        let concatenating = "{% for m in messages %}{% for c in m.tool_calls or [] %}\
                             {{ 'args=' + c.function.arguments }}{% endfor %}{% endfor %}";
        let named_pair = || ConfigChatTemplates {
            default: Some("{{ messages | length }}".to_string()),
            tool_use: Some(concatenating.to_string()),
        };
        let kept = ChatTemplateState::from_checkpoint(
            named_pair(),
            Some(format!("processor:{concatenating}")),
        )
        .unwrap();
        assert!(kept.raw_tool_call_arguments());
        assert!(format!("{kept:?}").contains("has_processor: true"));
        let dropped = ChatTemplateState::from_checkpoint(
            named_pair(),
            Some(
                "{% for m in messages %}{% for c in m.tool_calls or [] %}\
                 {{ c.function.arguments | tojson }}{% endfor %}{% endfor %}"
                    .to_string(),
            ),
        )
        .unwrap();
        assert!(format!("{dropped:?}").contains("has_processor: false"));

        // A processor template that does not compile leaves the tokenizer's
        // templates serving every request; the configuration's own still
        // fails the load, as before.
        let broken =
            ChatTemplateState::from_checkpoint(templates(), Some("{{".to_string())).unwrap();
        assert!(format!("{broken:?}").contains("has_processor: false"));
        assert_eq!(
            broken
                .apply(&messages, ChatTemplateParams::default())
                .unwrap(),
            "default:Hi"
        );
        assert!(ChatTemplateState::from_checkpoint(
            ConfigChatTemplates {
                default: Some("{{".to_string()),
                tool_use: None,
            },
            None,
        )
        .is_err());
    }

    /// The answer to how a history tool call's arguments are written comes
    /// from the `tool_use` template of a named pair: the default one has no
    /// tool-call branch to ask.
    #[test]
    fn a_named_pair_answers_tool_call_arguments_from_the_tool_use_template() {
        let concatenating = "{% for m in messages %}{% for c in m.tool_calls or [] %}\
                             {{ 'args=' + c.function.arguments }}{% endfor %}{% endfor %}";
        let alone = ChatTemplateState::new(Some(concatenating.to_string())).unwrap();
        assert!(alone.raw_tool_call_arguments());
        let pair = ChatTemplateState::from_config(ConfigChatTemplates {
            default: Some("{{ messages | length }}".to_string()),
            tool_use: Some(concatenating.to_string()),
        })
        .unwrap();
        assert!(pair.raw_tool_call_arguments());
        let plain = ChatTemplateState::from_config(ConfigChatTemplates {
            default: Some("{{ messages | length }}".to_string()),
            tool_use: Some("{{ tools | tojson }}".to_string()),
        })
        .unwrap();
        assert!(!plain.raw_tool_call_arguments());
    }

    /// A template file is read as written: a newline before its first block
    /// is the template's and reaches the prompt, as it does through the
    /// engine's chat utils, which read the file whole; exactly one trailing
    /// newline of the source is dropped, as Jinja2 drops it.
    #[test]
    fn a_template_file_keeps_the_newline_it_opens_with() {
        let dir = tempfile::tempdir().unwrap();
        let render = |source: &str| {
            let path = dir.path().join("chat_template.jinja");
            fs::write(&path, source).unwrap();
            let template = load_chat_template_from_file(path.to_str().unwrap())
                .unwrap()
                .unwrap();
            let state = ChatTemplateState::new(Some(template)).unwrap();
            state
                .apply(
                    &[serde_json::json!({"role": "user", "content": "Hi"})],
                    ChatTemplateParams::default(),
                )
                .unwrap()
        };
        // Text around the blocks, as the templates of the record open (a `{%-`
        // tag strips the whitespace before it, and `trim_blocks` the newline
        // after a tag, in both engines alike).
        let body = "<s>{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}</s>";
        assert_eq!(render(&format!("\n{body}")), "\n<s><|user|>Hi</s>");
        assert_eq!(render(&format!("\n{body}\n")), "\n<s><|user|>Hi</s>");
        assert_eq!(render(&format!("\n{body}\n\n")), "\n<s><|user|>Hi</s>\n");
        assert_eq!(render(&format!("  {body}  ")), "  <s><|user|>Hi</s>  ");
    }

    /// The load warning counts a backslash-n in the template's text, not one
    /// inside a tag (a string literal the template engine's escape turns into a
    /// newline), and a real newline is no such pair.
    #[test]
    fn a_backslash_n_counts_only_outside_the_template_s_tags() {
        assert!(text_outside_tags_holds_backslash_n(
            "<|im_start|>system<|im_end|>\\n{{ messages[0].content }}"
        ));
        assert!(text_outside_tags_holds_backslash_n(
            "{%- for m in messages %}{{ m.content }}{%- endfor %}\\n"
        ));
        assert!(!text_outside_tags_holds_backslash_n(
            "{{ '\\n' }}{% set nl = \"\\n\" %}{# a \\n in a note #}plain text"
        ));
        assert!(!text_outside_tags_holds_backslash_n("line one\nline two\n"));
    }

    /// A backslash followed by an `n` in a template file's text is the two
    /// characters it is, as the engine's chat utils read the file (a file
    /// that is one escaped line renders as that line); inside a Jinja string
    /// literal the template engine's own escape makes a newline of it, in
    /// both engines alike.
    #[test]
    fn a_template_file_s_backslash_n_stays_the_two_characters_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(
            &path,
            "<|im_start|>system<|im_end|>\\n{% for m in messages %}{{ m.content }}{{ '\\n' }}{% endfor %}",
        )
        .unwrap();
        let template = load_chat_template_from_file(path.to_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            template,
            "<|im_start|>system<|im_end|>\\n{% for m in messages %}{{ m.content }}{{ '\\n' }}{% endfor %}"
        );
        let state = ChatTemplateState::new(Some(template)).unwrap();
        let rendered = state
            .apply(
                &[serde_json::json!({"role": "user", "content": "Hi"})],
                ChatTemplateParams::default(),
            )
            .unwrap();
        assert_eq!(rendered, "<|im_start|>system<|im_end|>\\nHi\n");
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

    /// `tojson` writes the template value straight into its text; the text is
    /// Python's `json.dumps` of the value under each option, and the same as
    /// a `serde_json::Value` built from the template value would serialize
    /// as (the path the sorted form still takes).
    #[test]
    fn tojson_writes_the_value_as_python_and_as_its_tree_would() {
        let value = serde_json::json!({
            "name": "Zoë", "n": 1, "big": 18446744073709551615u64, "neg": -5, "f": 1.5,
            "whole": 2.0, "t": true, "none": null,
            "list": [1, "two", [3.25, {"k": "v"}]],
            "nested": {"b": 1, "a": {"c": [], "d": {}}},
            "esc": "quote\" back\\ tab\t nl\n ctl\u{1} emoji😀",
        });
        let template = "{{ messages[0].v | tojson }}\n\
                        {{ messages[0].v | tojson(ensure_ascii=True) }}\n\
                        {{ messages[0].v | tojson(indent=2) }}\n\
                        {{ messages[0].v | tojson(separators=(',', ':')) }}\n\
                        {{ messages[0].v | tojson(sort_keys=True) }}";
        let messages = [serde_json::json!({"role": "user", "v": value})];
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        // Each line is `json.dumps(v, ensure_ascii=False, ...)` with that option.
        let want = [
            r#"{"name": "Zoë", "n": 1, "big": 18446744073709551615, "neg": -5, "f": 1.5, "whole": 2.0, "t": true, "none": null, "list": [1, "two", [3.25, {"k": "v"}]], "nested": {"b": 1, "a": {"c": [], "d": {}}}, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji😀"}"#,
            r#"{"name": "Zo\u00eb", "n": 1, "big": 18446744073709551615, "neg": -5, "f": 1.5, "whole": 2.0, "t": true, "none": null, "list": [1, "two", [3.25, {"k": "v"}]], "nested": {"b": 1, "a": {"c": [], "d": {}}}, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji\ud83d\ude00"}"#,
            "{\n  \"name\": \"Zoë\",\n  \"n\": 1,\n  \"big\": 18446744073709551615,\n  \"neg\": -5,\n  \"f\": 1.5,\n  \"whole\": 2.0,\n  \"t\": true,\n  \"none\": null,\n  \"list\": [\n    1,\n    \"two\",\n    [\n      3.25,\n      {\n        \"k\": \"v\"\n      }\n    ]\n  ],\n  \"nested\": {\n    \"b\": 1,\n    \"a\": {\n      \"c\": [],\n      \"d\": {}\n    }\n  },\n  \"esc\": \"quote\\\" back\\\\ tab\\t nl\\n ctl\\u0001 emoji😀\"\n}",
            r#"{"name":"Zoë","n":1,"big":18446744073709551615,"neg":-5,"f":1.5,"whole":2.0,"t":true,"none":null,"list":[1,"two",[3.25,{"k":"v"}]],"nested":{"b":1,"a":{"c":[],"d":{}}},"esc":"quote\" back\\ tab\t nl\n ctl\u0001 emoji😀"}"#,
            r#"{"big": 18446744073709551615, "esc": "quote\" back\\ tab\t nl\n ctl\u0001 emoji😀", "f": 1.5, "list": [1, "two", [3.25, {"k": "v"}]], "n": 1, "name": "Zoë", "neg": -5, "nested": {"a": {"c": [], "d": {}}, "b": 1}, "none": null, "t": true, "whole": 2.0}"#,
        ];
        assert_eq!(rendered, want.join("\n"));

        // The direct write and the write of the tree agree byte for byte.
        let given = Value::from_serialize(&messages[0]["v"]);
        for (indent, ensure_ascii) in [
            (None, false),
            (None, true),
            (Some(&b"  "[..]), false),
            (Some(&b""[..]), true),
        ] {
            let direct = serialize_with_python_json(
                &given,
                indent,
                JsonSeparators::python_default(indent.is_some()),
                ensure_ascii,
            )
            .unwrap();
            let tree: serde_json::Value = serde_json::to_value(&given).unwrap();
            let through_tree = serialize_with_python_json(
                &tree,
                indent,
                JsonSeparators::python_default(indent.is_some()),
                ensure_ascii,
            )
            .unwrap();
            assert_eq!(
                direct, through_tree,
                "indent {indent:?} ensure_ascii {ensure_ascii}"
            );
        }
    }

    /// `tojson(indent=...)` renders as Python's `json.dumps(indent=...)` for
    /// every kind of indent a template may write: an integer is that many
    /// spaces, zero and a negative integer break the lines and indent nothing,
    /// a bool is the integer it is in Python, a string is the literal indent
    /// (a tab, two spaces, any text, the empty string), and the item separator
    /// is the pretty one (`,`) whenever an indent was given, empty or not; a
    /// float is refused, as Python refuses it. The expectations are Python
    /// 3.12's `json.dumps(value, indent=indent, ensure_ascii=False)`.
    #[test]
    fn tojson_indent_renders_as_python_for_every_kind_of_indent() {
        let value = serde_json::json!({"a": [1, {"b": 2}], "c": {}, "d": [], "e": "x"});
        let newlines_only =
            "{\n\"a\": [\n1,\n{\n\"b\": 2\n}\n],\n\"c\": {},\n\"d\": [],\n\"e\": \"x\"\n}";
        let one_space = "{\n \"a\": [\n  1,\n  {\n   \"b\": 2\n  }\n ],\n \"c\": {},\n \"d\": [],\n \"e\": \"x\"\n}";
        let two_spaces = "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ],\n  \"c\": {},\n  \"d\": [],\n  \"e\": \"x\"\n}";
        let oracle: [(serde_json::Value, &str); 13] = [
            (serde_json::Value::Null, r#"{"a": [1, {"b": 2}], "c": {}, "d": [], "e": "x"}"#),
            (serde_json::json!(0), newlines_only),
            (serde_json::json!(-1), newlines_only),
            (serde_json::json!(-5), newlines_only),
            (serde_json::json!(1), one_space),
            (serde_json::json!(2), two_spaces),
            (serde_json::json!(true), one_space),
            (serde_json::json!(false), newlines_only),
            (serde_json::json!(""), newlines_only),
            (serde_json::json!(" "), one_space),
            (serde_json::json!("  "), two_spaces),
            (
                serde_json::json!("\t"),
                "{\n\t\"a\": [\n\t\t1,\n\t\t{\n\t\t\t\"b\": 2\n\t\t}\n\t],\n\t\"c\": {},\n\t\"d\": [],\n\t\"e\": \"x\"\n}",
            ),
            (
                serde_json::json!("ab"),
                "{\nab\"a\": [\nabab1,\nabab{\nababab\"b\": 2\nabab}\nab],\nab\"c\": {},\nab\"d\": [],\nab\"e\": \"x\"\n}",
            ),
        ];
        let processor = ChatTemplateProcessor::new(
            "{{ messages[0].v | tojson(indent=messages[0].i) }}".to_string(),
        )
        .unwrap();
        for (indent, want) in &oracle {
            let messages = [serde_json::json!({"role": "user", "v": value, "i": indent})];
            let rendered = processor
                .apply_chat_template(&messages, ChatTemplateParams::default())
                .unwrap();
            assert_eq!(&rendered, want, "indent {indent}");
        }
        // Keys sorted under a string indent, and empty containers on one line.
        let processor = ChatTemplateProcessor::new(
            "{{ messages[0].v | tojson(indent=messages[0].i, sort_keys=True) }}".to_string(),
        )
        .unwrap();
        let messages = [serde_json::json!({
            "role": "user", "v": {"b": 1, "a": {"d": [], "c": 2}}, "i": "\t"
        })];
        let rendered = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "{\n\t\"a\": {\n\t\t\"c\": 2,\n\t\t\"d\": []\n\t},\n\t\"b\": 1\n}"
        );
        // A float is no indent, in Python either.
        let processor = ChatTemplateProcessor::new(
            "{{ messages[0].v | tojson(indent=messages[0].i) }}".to_string(),
        )
        .unwrap();
        let messages = [serde_json::json!({"role": "user", "v": value, "i": 2.0})];
        let error = processor
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("indent must be an integer or a string"),
            "{error}"
        );
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

    /// A template that never names the `root` role (the MiniMax templates
    /// before M3 read `messages[0].role == "system"` only) would drop a root
    /// message: it gets the message as the leading system message instead.
    #[test]
    fn a_root_message_renders_as_the_system_message_when_the_template_has_no_root_branch() {
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "root", "content": "Answer in French."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "<sys>Answer in French.</sys><usr>hi</usr>");
    }

    /// A system message after the root is not lost on such a template: the
    /// two merge into one system turn, the root's text first.
    #[test]
    fn a_system_message_after_a_root_merges_into_one_system_turn_without_a_root_branch() {
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "root", "content": "Answer in French."}),
            serde_json::json!({"role": "system", "content": "You are terse."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<sys>Answer in French.\n\nYou are terse.</sys><usr>hi</usr>"
        );
    }

    /// A root followed by a developer message on a template with neither
    /// branch: both become the one system turn, the root's text first.
    #[test]
    fn a_root_and_a_developer_message_merge_root_first_without_either_branch() {
        let template = "{% for m in messages %}{% if m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "root", "content": "Answer in French."}),
            serde_json::json!({"role": "developer", "content": "Be brief."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<sys>Answer in French.\n\nBe brief.</sys><usr>hi</usr>"
        );
    }

    #[test]
    fn a_root_message_reaches_a_template_that_handles_the_role() {
        let template = "{% for m in messages %}{% if m.role == 'root' %}<root>{{ m.content }}</root>\
                        {% elif m.role == 'system' %}<sys>{{ m.content }}</sys>\
                        {% elif m.role == 'user' %}<usr>{{ m.content }}</usr>{% endif %}{% endfor %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        let messages = vec![
            serde_json::json!({"role": "root", "content": "Answer in French."}),
            serde_json::json!({"role": "system", "content": "You are terse."}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let rendered = state
            .apply(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(
            rendered,
            "<root>Answer in French.</root><sys>You are terse.</sys><usr>hi</usr>"
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

    /// A non-thinking template without a switch whose generation prompt
    /// writes an empty, closed think block: `<think>` is in the prefill, but
    /// the completion starts in content, so the parser must not be armed.
    #[test]
    fn a_closed_think_block_in_the_generation_prompt_does_not_open_reasoning() {
        let closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                      {%- if add_generation_prompt -%}<|im_start|>assistant\n<think>\n\n</think>\n\n{%- endif -%}";
        let state = ChatTemplateState::new(Some(closed.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), ThinkingToggle::None);
        assert!(
            state.think_in_prefill(),
            "the loose flag still sees the tag"
        );
        assert!(!state.prefill_opens_think_block());

        // Opened in one branch, closed in another: the open branch counts.
        let branched = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|>\
                        {%- if clear_thinking is defined and clear_thinking -%}<think></think>\
                        {%- else -%}<think>{%- endif -%}{%- endif -%}";
        let state = ChatTemplateState::new(Some(branched.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());

        // Both branches open a block and the prompt closes it after them.
        let closed_after_branches = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|>\
                        {%- if clear_thinking is defined and clear_thinking -%}<think>\
                        {%- else -%}<think>\n{%- endif -%}</think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(closed_after_branches.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());

        // A block opened only when there is no generation prompt is not the
        // prompt's; the else branch of that condition is.
        let not_prompt = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if not add_generation_prompt -%}<think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(not_prompt.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());
        let else_prompt = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if not add_generation_prompt -%}done{%- else -%}<|assistant|><think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(else_prompt.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());

        // A comparison is not a positive reference either: this true body
        // runs without a generation prompt.
        let compared = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt == false -%}<think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(compared.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());
        let conjunction = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt and messages -%}<|assistant|><think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(conjunction.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());

        // The prompt written in two generation-prompt blocks: the second may
        // close what the first opened, or open what the first left closed.
        let open_then_closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}\
                        {%- if add_generation_prompt -%}\n\n</think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(open_then_closed.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());
        let closed_then_open = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|><think></think>{%- endif -%}\
                        {%- if add_generation_prompt -%}<think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(closed_then_open.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());

        // Two blocks that run on different requests (a conjunction each):
        // either may leave the block open, so the prompt counts as opening
        // whichever order they come in; a block that always runs still closes
        // what an earlier one opened.
        let exclusive = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt and thinking -%}<|assistant|><think>{%- endif -%}\
                        {%- if add_generation_prompt and not thinking -%}<|assistant|><think></think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(exclusive.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());
        let maybe_then_closed = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt and thinking -%}<|assistant|><think>{%- endif -%}\
                        {%- if add_generation_prompt -%}</think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(maybe_then_closed.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());

        // Conditions the variable alone decides: `not (add_generation_prompt
        // == false)` (the parse of the unparenthesised form) and
        // `add_generation_prompt == true` run their true body for a
        // generation prompt; `add_generation_prompt or x` always runs too, so
        // it closes what an earlier block opened; `not (add_generation_prompt
        // or x)` always runs its false body.
        let negated_comparison = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if not add_generation_prompt == false -%}<think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(negated_comparison.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());
        let eq_true = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt == true -%}<|assistant|><think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(eq_true.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());
        let or_closes = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}\
                        {%- if add_generation_prompt or continue_final_message -%}</think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(or_closes.to_string())).unwrap();
        assert!(!state.prefill_opens_think_block());
        let not_or = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if not (add_generation_prompt or continue_final_message) -%}done\
                        {%- else -%}<|assistant|><think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(not_or.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());

        // A test on the variable inside a condition something else also
        // decides: the block may run, so what it opens counts.
        let is_true_and = "{%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt is true and not tools -%}<|assistant|><think>{%- endif -%}";
        let state = ChatTemplateState::new(Some(is_true_and.to_string())).unwrap();
        assert!(state.prefill_opens_think_block());
    }

    #[test]
    fn clear_thinking_is_not_a_thinking_switch() {
        // GLM-5.3: the template reads `clear_thinking` and `reasoning_effort`
        // and always opens the think block; it has no switch.
        let template =
            "{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}\
                        {%- for m in messages -%}{{ m.content }}{%- endfor -%}\
                        {%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}";
        assert_eq!(
            detect_thinking_toggle(template),
            (ThinkingToggle::None, None)
        );
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), ThinkingToggle::None);
        assert!(state.think_in_prefill());
        assert!(state.prefill_opens_think_block());

        // A real `thinking` variable is still a switch, in each spelling.
        for template in [
            "{% if thinking is defined and not thinking %}off{% endif %}",
            "{% if thinking == false %}off{% endif %}",
            "{% if thinking %}on{% endif %}",
            "{% set thinking = thinking %}",
        ] {
            assert_eq!(
                detect_thinking_toggle(template),
                (ThinkingToggle::DefaultOn, Some(ThinkingKeyName::Thinking)),
                "{template}"
            );
        }
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
    fn gemma4_filter_default_keeps_thinking_off() {
        let template = "{%- set enable_thinking = enable_thinking | default(false) -%}{% if enable_thinking %}<|channel>thought\n{% endif %}";
        let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
        assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOff);
        assert_eq!(
            state.thinking_key_name(),
            Some(ThinkingKeyName::EnableThinking)
        );
        for (thinking, expected) in [
            (None, ""),
            (Some(false), ""),
            (Some(true), "<|channel>thought\n"),
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
    }

    #[test]
    fn thinking_filter_on_other_values_does_not_change_the_switch_default() {
        for template in [
            "{{ message.thinking | default(false) }}{% if enable_thinking %}<think>{% endif %}",
            "{{ other_thinking | default(false) }}{% if thinking %}<think>{% endif %}",
        ] {
            assert_eq!(
                detect_thinking_toggle(template).0,
                ThinkingToggle::DefaultOn
            );
        }
        assert_eq!(
            detect_thinking_toggle(
                "{% set thinking = thinking | default(false) %}{% if thinking %}<think>{% endif %}"
            ),
            (ThinkingToggle::DefaultOff, Some(ThinkingKeyName::Thinking))
        );
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
