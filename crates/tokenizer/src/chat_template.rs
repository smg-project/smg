//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fs, io, iter,
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
};

use anyhow::{anyhow, Result};
use chrono::{
    format::{strftime::StrftimeItems, Item},
    DateTime, FixedOffset, Local, TimeZone,
};
use minijinja::{
    machinery::{
        ast::{
            BinOpKind, Call, CallArg, CallType, Expr, ForLoop, IfCond, Macro, Set, Stmt,
            UnaryOpKind,
        },
        get_compiled_template, parse, tokenize, Instruction, Token, WhitespaceConfig,
    },
    syntax::SyntaxConfig,
    value::{DynObject, Enumerator, Kwargs, Object, ObjectRepr, ValueKind},
    Environment, Error as MinijinjaError, ErrorKind, Output, State, Template, Value,
};
use rustc_hash::FxHashSet;
use serde::Serialize;
use serde_json::{self, ser::Formatter, Value as JsonValue};

use crate::{encoders::marks, native::AddedTokens, traits::SpecialTokens};

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

    /// The kwarg value that switches thinking on or off under this key: a
    /// boolean, or the key's own words where the template compares strings.
    fn value(self, thinking: bool) -> serde_json::Value {
        match self {
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
            ThinkingKeyName::ThinkingMode => {
                serde_json::Value::String(if thinking { "enabled" } else { "disabled" }.to_string())
            }
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
/// and what its default value is. A `thinking` variable counts where the
/// template's code reads it (`mentions_thinking_variable`), not where its
/// prose prints the word. The default is read from the template's text: a
/// state built from the template takes it only where the template's own
/// renders do not say (`ChatTemplateState::rendered_thinking_default`).
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
/// that only reads those has no thinking switch. Only the template's code
/// is read (`jinja_code`): the MiniMax M3.1 template prints the prose
/// "When thinking is enabled" from a string literal and branches on no
/// variable, so it has no switch either.
fn mentions_thinking_variable(template: &str) -> bool {
    let code = jinja_code(template);
    let template = code.as_str();
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

/// The template's code with everything else blanked to spaces: the text the
/// template prints (the data outside the blocks and inside a `{% raw %}`
/// block) and the string literals; the comments never reach the lexer's
/// tokens. Built from minijinja's own tokenizer, so a block ends where the
/// lexer ends it (a nested dict literal's `}}` does not close a `{{ }}`
/// block). A word of the result is a variable, a keyword or an operator the
/// template evaluates; the words a template prints are not in it. A template
/// the lexer refuses is read whole, as the detector read every template
/// before this view.
fn jinja_code(template: &str) -> String {
    let mut code = vec![b' '; template.len()];
    for item in tokenize(template, false, SyntaxConfig, WhitespaceConfig::default()) {
        match item {
            Ok((Token::TemplateData(_) | Token::Str(_) | Token::String(_), _)) => {}
            Ok((_, span)) => {
                let (start, end) = (span.start_offset as usize, span.end_offset as usize);
                if let (Some(from), Some(to)) = (
                    template.as_bytes().get(start..end),
                    code.get_mut(start..end),
                ) {
                    to.copy_from_slice(from);
                }
            }
            Err(_) => return template.to_string(),
        }
    }
    String::from_utf8(code).unwrap_or_else(|_| template.to_string())
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
/// macros and the calls; and every name the template reads.
#[derive(Default)]
struct Nodes<'a> {
    loops: Vec<(&'a ForLoop<'a>, Vec<&'a Macro<'a>>)>,
    sets: Vec<&'a Set<'a>>,
    macros: Vec<&'a Macro<'a>>,
    calls: Vec<&'a Call<'a>>,
    /// Every name the template reads: a bare name anywhere in an expression
    /// (see [`ChatTemplateState::reads`]).
    names: HashSet<&'a str>,
    /// The name of every other template a statement asks for (`include`,
    /// `extends`, `import`, `from`), as the expression that names it (see
    /// `spells_its_templates`).
    templates: Vec<&'a Expr<'a>>,
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
                self.template(&i.expr);
                self.expr(&i.name);
            }
            Stmt::FromImport(f) => {
                self.template(&f.expr);
                for (name, alias) in &f.names {
                    self.expr(name);
                    if let Some(alias) = alias {
                        self.expr(alias);
                    }
                }
            }
            Stmt::Extends(e) => self.template(&e.name),
            Stmt::Include(i) => self.template(&i.name),
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
            // template (jinja2 keeps the two apart); its parameter defaults
            // are expressions of the template, as a macro's are.
            Stmt::CallBlock(cb) => {
                self.call(&cb.call);
                for default in &cb.macro_decl.defaults {
                    self.expr(default);
                }
                self.stmts(&cb.macro_decl.body, macros);
            }
            Stmt::Do(d) => self.call(&d.call),
        }
    }

    fn template(&mut self, name: &'a Expr<'a>) {
        self.templates.push(name);
        self.expr(name);
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
            Expr::Var(v) => {
                self.names.insert(v.id);
            }
            Expr::Const(_) => {}
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
fn iterates_message_content(nodes: &Nodes<'_>) -> Result<bool, NotAName> {
    let messages_names = names_assigned_from(&nodes.sets, "messages")?;
    let message_vars = message_loop_vars(&nodes.loops, &messages_names)?;
    let is_message_content = |expr: &Expr<'_>| {
        message_vars
            .iter()
            .any(|var| is_var_or_elems_access(expr, var, Some("content")))
    };

    // The parameters of each macro that a call fills with a message's content.
    let fed_params: Vec<(&Macro<'_>, HashSet<&str>)> = nodes
        .macros
        .iter()
        .map(|m| {
            let param = |index: usize| match m.args.get(index) {
                Some(Expr::Var(v)) => Some(v.id),
                _ => None,
            };
            let mut fed = HashSet::new();
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

/// What a template's text says before it is compiled: the answers a
/// [`ChatTemplateState`] keeps per template.
#[derive(Default)]
struct Detected {
    content_format: ChatTemplateContentFormat,
    think_in_prefill: bool,
    thinking_toggle: ThinkingToggle,
    thinking_key_name: Option<ThinkingKeyName>,
    prefill_opens_think_block: bool,
    /// Every name the template reads (see [`ChatTemplateState::reads`]);
    /// `None` where the template did not parse, the names being unknown.
    names_read: Option<HashSet<String>>,
    /// Whether the root `set` statements build values and the template only
    /// writes them ([`built_values_read_plainly`]).
    writes_built_constants: bool,
}

/// Single-pass detection of content format, think-in-prefill, thinking toggle,
/// the names read and whether the template only writes the values its root
/// statements build.
fn detect_all(template: &str) -> Detected {
    let (thinking_toggle, thinking_key_name) = detect_thinking_toggle(template);
    let (
        content_format,
        think_in_prefill,
        prefill_opens_think_block,
        names_read,
        writes_built_constants,
    ) = detect_all_with_ast(template);
    Detected {
        content_format,
        think_in_prefill,
        thinking_toggle,
        thinking_key_name,
        prefill_opens_think_block,
        names_read,
        writes_built_constants,
    }
}

/// AST detection of content format, think-in-prefill, whether the generation
/// prompt leaves a think block open, the names the template reads (`None`
/// where it did not parse), and whether the root `set` statements build
/// values that the template only writes ([`built_values_read_plainly`]).
fn detect_all_with_ast(
    template: &str,
) -> (
    ChatTemplateContentFormat,
    bool,
    bool,
    Option<HashSet<String>>,
    bool,
) {
    let template = plain_generation_blocks(template);
    let ast = match parse(
        &template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return (ChatTemplateContentFormat::String, false, false, None, false),
    };

    let nodes = Nodes::of(&ast);
    let content_format = match iterates_message_content(&nodes) {
        Ok(true) => ChatTemplateContentFormat::OpenAI,
        Ok(false) | Err(NotAName) => ChatTemplateContentFormat::String,
    };
    let names_read = Some(nodes.names.iter().map(|name| (*name).to_string()).collect());
    let (think_in_prefill, prefill_opens_think_block) = ThinkDetector::run(&ast);
    let mut sets = Vec::new();
    root_set_statement(&ast, &template, false, &mut sets);
    let namespaces = namespaces(&sets);
    let writes_built_constants =
        built_values_read_plainly(&ast, &computed_statements(sets).0, &namespaces);
    (
        content_format,
        think_in_prefill,
        prefill_opens_think_block,
        names_read,
        writes_built_constants,
    )
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
    pub special_tokens: Option<&'a SpecialTokens>,
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
    value: Value,
    /// The same tools with the keys of every map in them sorted, prepared
    /// as these were, for a checkpoint whose own tokenizer class reads them
    /// so ([`Self::key_sorted`]); made on the first such render.
    key_sorted: OnceLock<Box<PreparedTools>>,
}

impl PreparedTools {
    /// How many `tojson` texts each tool keeps at most, one per distinct
    /// argument set of the filter, each about the tool's serialized bytes
    /// (more under `indent`): what a keeper of prepared tools charges for
    /// them on top of the values.
    pub const KEPT_TOJSON_TEXTS_PER_TOOL: usize = KEPT_TOJSON_TEXTS_PER_TOOL;

    /// Converts `tools` as a render converts [`ChatTemplateParams::tools`]:
    /// the same values, each tool read through a `PreparedTool` that keeps
    /// its `tojson` text once written.
    pub fn new(tools: Vec<JsonValue>) -> Self {
        let value = Value::from(
            tools
                .iter()
                .map(PreparedTool::value)
                .collect::<Vec<Value>>(),
        );
        Self {
            tools,
            value,
            key_sorted: OnceLock::new(),
        }
    }

    /// How many `tojson` texts the tools keep between them.
    #[cfg(test)]
    fn kept_tojson_texts(&self) -> Vec<usize> {
        self.value
            .try_iter()
            .into_iter()
            .flatten()
            .map(|tool| {
                tool.downcast_object_ref::<PreparedTool>()
                    .map_or(0, PreparedTool::kept_texts)
            })
            .collect()
    }

    /// The values this was made of: what a caller passes as
    /// [`ChatTemplateParams::tools`] beside it.
    pub fn tools(&self) -> &[JsonValue] {
        &self.tools
    }

    /// Whether this was made of `tools` (the same slice): the only tools it
    /// is read for.
    pub(crate) fn is_of(&self, tools: &[JsonValue]) -> bool {
        std::ptr::eq(self.tools.as_ptr(), tools.as_ptr()) && self.tools.len() == tools.len()
    }

    /// These tools with the keys of every map in them sorted, at every depth
    /// (a tool's `function` before its `type`, a schema's `properties` before
    /// its `type`), prepared as these were: what a checkpoint whose own
    /// tokenizer class sorts the tools before its template or tool encoder
    /// reads them renders. Made once, on the first call, and kept beside the
    /// plain form, so a caller that keeps a prepared set per distinct tool
    /// set spares the sort and the conversion per request as it spares them
    /// for the plain form.
    pub(crate) fn key_sorted(&self) -> &PreparedTools {
        self.key_sorted
            .get_or_init(|| Box::new(Self::key_sorted_of(&self.tools)))
    }

    /// `tools` with the keys of every map in them sorted, prepared
    /// ([`Self::new`] of the sorted values); see [`Self::key_sorted`].
    pub(crate) fn key_sorted_of(tools: &[JsonValue]) -> PreparedTools {
        Self::new(tools.iter().map(sort_json_keys).collect())
    }

    /// The render's value of `tools`, when they are the values this was made
    /// of (the same slice).
    fn value_of(&self, tools: &[JsonValue]) -> Option<Value> {
        self.is_of(tools).then(|| self.value.clone())
    }
}

/// One tool of a [`PreparedTools`] value: the converted map, which the render
/// reads as it reads any map (every read delegates to it), and that map's
/// `tojson` texts by argument set, kept from the first filter call. A kept
/// text is what the filter wrote for the map under those arguments, so
/// `tool | tojson` renders the same bytes kept or written; written, it is a
/// serialization of the schema for every request (one boxed iterator per map
/// or array in it), kept, it is a clone of the text.
struct PreparedTool {
    map: Value,
    texts: Mutex<Vec<(TojsonArgs, Value)>>,
}

/// The arguments a kept `tojson` text was written with. Texts under
/// `sort_keys` are not kept: they are written from the value's tree.
#[derive(Debug)]
struct TojsonArgs {
    indent: Option<Vec<u8>>,
    separators: JsonSeparators,
    ensure_ascii: bool,
}

impl TojsonArgs {
    fn matches(
        &self,
        indent: Option<&[u8]>,
        separators: &JsonSeparators,
        ensure_ascii: bool,
    ) -> bool {
        self.ensure_ascii == ensure_ascii
            && self.indent.as_deref() == indent
            && self.separators.item == separators.item
            && self.separators.key == separators.key
    }
}

/// How many argument sets a tool keeps a text for: a template applies one
/// form of the filter to a tool, two at most.
const KEPT_TOJSON_TEXTS_PER_TOOL: usize = 4;

impl PreparedTool {
    /// The render's value of one tool: a map read through this, or the plain
    /// value where the tool is not a map (nothing to keep a text for).
    fn value(tool: &JsonValue) -> Value {
        let map = Value::from_serialize(tool);
        if map.kind() != ValueKind::Map {
            return map;
        }
        Value::from_object(Self {
            map,
            texts: Mutex::new(Vec::new()),
        })
    }

    fn map(&self) -> Option<&DynObject> {
        self.map.as_object()
    }

    /// The map's `tojson` text under these arguments: the kept one, else
    /// written now and kept (up to [`KEPT_TOJSON_TEXTS_PER_TOOL`] sets).
    fn tojson_text(
        &self,
        indent: Option<&[u8]>,
        separators: &JsonSeparators,
        ensure_ascii: bool,
    ) -> std::result::Result<Value, MinijinjaError> {
        let kept = self
            .texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(args, _)| args.matches(indent, separators, ensure_ascii))
            .map(|(_, text)| text.clone());
        if let Some(text) = kept {
            return Ok(text);
        }
        let text = Value::from_safe_string(serialize_with_python_json(
            &self.map,
            indent,
            separators.clone(),
            ensure_ascii,
        )?);
        let mut texts = self.texts.lock().unwrap_or_else(PoisonError::into_inner);
        if texts.len() < KEPT_TOJSON_TEXTS_PER_TOOL
            && !texts
                .iter()
                .any(|(args, _)| args.matches(indent, separators, ensure_ascii))
        {
            texts.push((
                TojsonArgs {
                    indent: indent.map(<[u8]>::to_vec),
                    separators: separators.clone(),
                    ensure_ascii,
                },
                text.clone(),
            ));
        }
        Ok(text)
    }

    #[cfg(test)]
    fn kept_texts(&self) -> usize {
        self.texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Prints as the map it reads through (`pprint`, `debug`).
impl std::fmt::Debug for PreparedTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.map, f)
    }
}

/// Every read a template makes of a tool (`tool.function`, `tool | items`,
/// `tool | length`, `if tool`, `{{ tool }}`, equality, the slow `tojson`
/// paths through `Serialize`) goes to the converted map, so the value
/// behaves as the map did; only [`tojson_filter`] tells the two apart.
impl Object for PreparedTool {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
        self.map()?.get_value_by_str(key)
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.map()?.get_value(key)
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.map().map_or(Enumerator::Empty, DynObject::enumerate)
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        self.map()?.enumerator_len()
    }

    fn is_true(self: &Arc<Self>) -> bool {
        self.map().is_some_and(DynObject::is_true)
    }

    fn render(self: &Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    where
        Self: Sized + 'static,
    {
        match self.map() {
            Some(map) => map.render(f),
            None => f.write_str("{}"),
        }
    }
}

impl ChatTemplateParams<'_> {
    /// Whether a render under these parameters carries the generation header
    /// unless it continues the final message: the one `add_generation_prompt`
    /// asks for, or the one a continuation the renderer cannot honour falls
    /// back to, with the message's text after it. A render that continued
    /// the message carries none (`ChatTemplateOutput::generation_prompt`).
    pub fn generation_prompt_unless_continued(&self) -> bool {
        self.add_generation_prompt || self.continue_final_message
    }
}

/// The initial size of a `tojson` output buffer: room for a function schema
/// of the usual size without a reallocation.
const TOJSON_BUFFER_BYTES: usize = 1024;

/// JSON separator pair passed through HuggingFace's `tojson` filter. The
/// defaults are the static bytes; only a `separators` argument owns its own.
#[derive(Debug, Clone)]
pub(crate) struct JsonSeparators {
    item: Cow<'static, [u8]>,
    key: Cow<'static, [u8]>,
}

impl JsonSeparators {
    /// Python's defaults: `(', ', ': ')` for compact output and `(',', ': ')`
    /// when an indent was given (`pretty`), whatever the indent is: an empty
    /// one still puts every item on its own line.
    pub(crate) fn python_default(pretty: bool) -> Self {
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

    /// One more array item or object key: the item separator before every
    /// item but the first, then, with an indent, a line break and the
    /// indentation. Python's `json.dumps` begins the two the same way.
    fn begin_item<W>(&self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(&self.separators.item)?;
        }
        if let Some(indent) = self.indent {
            writer.write_all(b"\n")?;
            write_indent(writer, self.current_indent, indent)?;
        }
        Ok(())
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
        self.begin_item(writer, first)
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
        self.begin_item(writer, first)
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

/// Python's `json.dumps` of `value` through the formatter the `tojson` filter
/// and the prompt encoders share: `indent`, `separators` and `ensure_ascii` as
/// the filter reads them.
pub(crate) fn serialize_with_python_json<T: Serialize>(
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

/// Whether a filter call passed no keyword argument at all. `Kwargs::args`
/// lists the string keys only, so a splat with a non-string key would read
/// as empty; the engine's own check, run before any read, is `Ok` for an
/// empty map alone (a string key reads as unknown, a non-string key as its
/// own error), and costs nothing on an empty map.
fn passed_no_keyword(kwargs: &Kwargs) -> bool {
    kwargs.args().next().is_none() && kwargs.assert_all_used().is_ok()
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
    // A call without keyword arguments, the common `value | tojson`, takes the
    // defaults without reading the keywords: `Kwargs::get` records every key
    // it reads in a set the engine allocates per call, so the four reads cost
    // six allocations on each call, paid before a kept text is found. A call
    // with keywords reads them as before, and an unknown one still raises the
    // engine's own error.
    let (ensure_ascii, indent, separators, sort_keys): (
        Option<bool>,
        Option<Value>,
        Option<Value>,
        Option<bool>,
    ) = if passed_no_keyword(&kwargs) {
        (None, None, None, None)
    } else {
        let ensure_ascii = kwargs.get("ensure_ascii")?;
        let indent = kwargs.get("indent")?;
        let separators = kwargs.get("separators")?;
        let sort_keys = kwargs.get("sort_keys")?;
        kwargs.assert_all_used()?;
        (ensure_ascii, indent, separators, sort_keys)
    };

    let indent = parse_indent(indent)?;
    let separators = parse_separators(separators, indent.is_some())?;
    let ensure_ascii = ensure_ascii.unwrap_or(false);
    // A tool of a prepared tool set keeps the text written for it under these
    // arguments, so every later request with the same tools reads the text
    // instead of serializing the schema again; sorted keys are written from
    // the tree below, as for any value.
    if !sort_keys.unwrap_or(false) {
        if let Some(tool) = value.downcast_object_ref::<PreparedTool>() {
            return tool.tojson_text(indent.as_deref(), &separators, ensure_ascii);
        }
    }
    // The template value is written straight into its JSON text: its serde
    // form is what a `serde_json::Value` built from it would serialize as
    // (the same `Serialize` impl drives both, in the same key order), so
    // the tree is built only when the keys are to be sorted, which needs
    // the value as a whole.
    let json_str = if sort_keys.unwrap_or(false) {
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
    Err(MinijinjaError::new(ErrorKind::InvalidOperation, message).with_source(RaisedByTemplate))
}

/// The source of the error [`raise_exception`] returns. The template engine
/// reports an operation it could not run under the same kind, so the kind
/// alone does not say that the template refused its input on purpose; this
/// source does.
#[derive(Debug)]
struct RaisedByTemplate;

impl std::fmt::Display for RaisedByTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("raised by the template")
    }
}

impl std::error::Error for RaisedByTemplate {}

/// A render the template itself refused through `raise_exception`: the helper
/// a template's author calls to reject a conversation the template cannot
/// format (roles out of order, a field it needs). The caller's input is at
/// fault; any other render failure may be the template's own. The text is the
/// render failure's, as every render failure reads.
#[derive(Debug)]
pub struct TemplateRefusal(String);

impl TemplateRefusal {
    /// Whether `error` is this refusal, at any depth of its chain.
    pub fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }
}

impl std::fmt::Display for TemplateRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TemplateRefusal {}

/// A render that failed on the template's own account, whatever its caller
/// handed it: the compiled template is not there to render, or the engine
/// stopped on something the template's source alone decides
/// (`is_the_templates_own`: another template it asks for by a name it spells,
/// a filter, a test or a block the engine does not have, a function it calls
/// and binds nowhere). Output that is not UTF-8 is typed the same: the engine
/// writes text alone, so no value makes it. The model's template is the one
/// to fix, which its deployment can and no caller can.
///
/// A failure a value can bring about (an operation on a value of the wrong
/// kind, an undefined value, a bad argument) is not this one, though the
/// template may be at fault there too: the render cannot tell a caller's
/// value that broke a sound template from a template that breaks on some
/// values, and it does not put a caller's mistake on the deployment. The
/// text is the render failure's, as every render failure reads.
#[derive(Debug)]
pub struct TemplateFailure(String);

impl TemplateFailure {
    /// Whether `error` is this failure, at any depth of its chain.
    pub fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }
}

impl std::fmt::Display for TemplateFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TemplateFailure {}

/// Whether an error of rendering `template` is the template's own, whatever
/// its caller handed it ([`TemplateFailure`]).
///
/// The rule: an error is the template's own only where the template's source
/// alone decides that the statement fails, so that no value of the caller's
/// (a message, a tool, a template argument) makes that statement render. An
/// error a value can bring about stays the plain failure it was, and so does
/// one not shown to be free of them: a caller's mistake never reads as the
/// template's. The arms are read from the engine's sources at the release
/// this crate pins. The enum is open, so a kind another release adds, or one
/// behind an engine feature this crate does not enable, falls to the plain
/// side until it is classed here.
fn is_the_templates_own(error: &MinijinjaError, template: &Template<'_, '_>) -> bool {
    match error.kind() {
        // The compiler's alone, over a template's source text: no render
        // raises them, and a template that has them does not load.
        ErrorKind::SyntaxError | ErrorKind::BadEscape => true,
        // Another template asked for by name (`include`, `extends`,
        // `import`, `from`). The environment a chat template renders in holds
        // that one template and has no loader, so a name the source spells is
        // never another template. A name read from a value is the caller's
        // to get wrong: a list of fragments renders when it is empty. The
        // error does not say which statement raised it, and a value can hand
        // one statement the name another spells, so the kind is the
        // template's own only where every such statement spells its name.
        ErrorKind::TemplateNotFound => spells_its_templates(template.source()),
        // A called name that nothing defines. A caller supplies values and
        // never a callable, so no value makes the call run: one put under
        // the name fails it as an operation on that value (not callable),
        // another kind. The template's own statements bind names too, and a
        // value decides which of them run: a macro declared under an `if`
        // is there for one conversation and unknown for the next. So the
        // kind is the template's own only where its source binds the name
        // nowhere (`binds`). The detail names it: `name is unknown`.
        ErrorKind::UnknownFunction => error
            .detail()
            .and_then(|detail| detail.strip_suffix(" is unknown"))
            .is_some_and(|name| !binds(template, name)),
        // `self.name()` for a block the template does not have: the name is
        // in the source. The engine compiles a template's blocks wherever
        // they stand, one under an `if` that does not run among them, so no
        // value decides which blocks there are.
        ErrorKind::UnknownBlock => true,
        // A filter or a test the engine does not have. Named in the source
        // (`value | name`, `value is name`), the engine says which in the
        // error's detail, and Jinja2 refuses such a template when it compiles
        // it: the engine reads the name from the environment, which no
        // statement and no value adds to. Looked up from an argument's value
        // by `map`, `select` and their kin, the error has no detail, and the
        // value may be the caller's.
        ErrorKind::UnknownFilter | ErrorKind::UnknownTest => error.detail().is_some(),
        // A write failed, to the output or of a value as text: the engine
        // makes this kind of every `fmt::Error`, the ones `python_formatter`
        // passes up among them. A render writes into memory, where a write
        // does not fail, and the values of its context format without fail,
        // so no value brings it about.
        ErrorKind::WriteFailure => true,
        // An operation on values the engine or a function could not run: an
        // operator over the wrong kinds, a value that is not callable or not
        // iterable, a function's own refusal of its input.
        ErrorKind::InvalidOperation => false,
        // A method the value does not have: the value's kind decides.
        ErrorKind::UnknownMethod => false,
        // An undefined value was used: whether a name or an attribute is
        // defined is the caller's data.
        ErrorKind::UndefinedError => false,
        // A call with the wrong arguments. A call can spread a value into
        // its arguments, and a function can find one of them undefined.
        ErrorKind::TooManyArguments | ErrorKind::MissingArgument => false,
        // A value of the wrong shape was unpacked into names.
        ErrorKind::CannotUnpack => false,
        // A value did not serialize, or did not read as a function's
        // argument type.
        ErrorKind::BadSerialization | ErrorKind::CannotDeserialize => false,
        // About a value where a primitive or a map key was expected; the
        // engine at this release raises neither.
        ErrorKind::NonPrimitive | ErrorKind::NonKey => false,
        // Wrappers around the failure of an included template or a parent
        // block, which may be of any kind.
        ErrorKind::BadInclude | ErrorKind::EvalBlock => false,
        // A kind this build does not name.
        _ => false,
    }
}

/// Whether every statement of `source` that asks for another template
/// (`include`, `extends`, `import`, `from`) spells the name it asks for: a
/// literal, or what the compiler folds to one, which no value takes part in.
/// False for a source that does not parse.
fn spells_its_templates(source: &str) -> bool {
    parse(source, "chat", SyntaxConfig {}, WhitespaceConfig::default()).is_ok_and(|ast| {
        Nodes::of(&ast)
            .templates
            .iter()
            .all(|name| name.as_const().is_some())
    })
}

/// Whether the source of `template` binds `name` on any branch: a macro of
/// that name, or a `set`, a `for`, a `with`, an import, or a parameter of a
/// macro or of a `call` block that assigns it. The compiled template stores
/// each through one instruction (`StoreLocal`), wherever its statement
/// stands. The engine binds two names without it: `loop` in a `for`, and
/// `caller` in a macro that a `call` block invokes.
fn binds(template: &Template<'_, '_>, name: &str) -> bool {
    let compiled = get_compiled_template(template);
    matches!(name, "loop" | "caller")
        || iter::once(&compiled.instructions)
            .chain(compiled.blocks.values())
            .any(|code| {
                (0..)
                    .map_while(|at| code.get(at))
                    .any(|step| matches!(step, Instruction::StoreLocal(bound) if *bound == name))
            })
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

/// A root `set` statement of a chat template: one that binds a name at the
/// top level, or in a statement that shares the top level's names (a
/// top-level `if`, `filter`, `autoescape` or block `set`, the `else` of a
/// top-level `for`), where a template binds the names its whole render
/// reads.
struct RootSet<'a> {
    name: &'a str,
    /// The expression the name is bound to. `None` when nothing says what
    /// the value reads: the statement binds the name another way (a block's
    /// output, one of several names at once, a macro, an import), or its
    /// expression calls what the load cannot follow ([`names_read`]).
    expression: Option<Expression<'a>>,
    /// Whether the statement stands under a top-level `if` or in the `else`
    /// of a top-level `for`: it may not run.
    conditional: bool,
    /// Whether the expression is a call of `namespace`: the name is a
    /// namespace, whose attributes other statements bind.
    namespace: bool,
}

/// The expression of a root `set` statement.
struct Expression<'a> {
    /// Its byte span in the source.
    span: (usize, usize),
    /// The names it reads.
    names: Vec<&'a str>,
    /// A filter or a test takes part in it: its value may be another at the
    /// next render (`random`).
    draws: bool,
    /// The source spells its value already: it is a literal the source
    /// holds as it stands, or another name alone.
    spelled: bool,
}

/// The root `set` statements of `source`, a chat template, in the template's
/// order. Empty when the template does not parse.
fn root_sets(source: &str) -> Vec<RootSet<'_>> {
    let mut sets = Vec::new();
    if let Ok(ast) = parse(source, "chat", SyntaxConfig {}, WhitespaceConfig::default()) {
        root_set_statement(&ast, source, false, &mut sets);
    }
    sets
}

/// [`root_sets`] of `statements`, under a top-level `if` or not
/// (`conditional`).
fn root_sets_under<'a>(
    statements: &[Stmt<'a>],
    source: &'a str,
    conditional: bool,
    sets: &mut Vec<RootSet<'a>>,
) {
    for stmt in statements {
        root_set_statement(stmt, source, conditional, sets);
    }
}

/// [`root_sets`] of `stmt` and of the statements under it.
fn root_set_statement<'a>(
    stmt: &Stmt<'a>,
    source: &'a str,
    conditional: bool,
    sets: &mut Vec<RootSet<'a>>,
) {
    match stmt {
        Stmt::Template(template) => {
            root_sets_under(&template.children, source, conditional, sets);
        }
        // The walk follows the engine's scopes. It enters every list of
        // statements that runs among the names it stands in, where a `set`
        // binds a top-level name: the bodies of an `if`, a `filter`, an
        // `autoescape` and a block `set`, and the `else` of a `for`, which
        // runs once the loop's own names are gone. It enters none that runs
        // among names of its own, which end with it: the body of a `for`, a
        // `with`, a macro, a `call`, a `block`.
        Stmt::IfCond(cond) => {
            root_sets_under(&cond.true_body, source, true, sets);
            root_sets_under(&cond.false_body, source, true, sets);
        }
        Stmt::ForLoop(for_loop) => root_sets_under(&for_loop.else_body, source, true, sets),
        Stmt::FilterBlock(block) => root_sets_under(&block.body, source, conditional, sets),
        Stmt::AutoEscape(block) => root_sets_under(&block.body, source, conditional, sets),
        Stmt::SetBlock(block) => {
            root_sets_under(&block.body, source, conditional, sets);
            unread_bindings(&block.target, conditional, sets);
        }
        Stmt::Set(set) => {
            let Expr::Var(target) = &set.target else {
                unread_bindings(&set.target, conditional, sets);
                return;
            };
            // The statement's span runs from `set` to the end of its
            // expression, which starts after the `=` that follows the name.
            let (name_end, end) = (
                target.span().end_offset as usize,
                set.span().end_offset as usize,
            );
            let span = source.get(name_end..end).and_then(|rest| {
                // Found by its byte: a search by a `char` pattern here makes
                // the optimiser build the mark scans of `encoders::marks`
                // differently (33 instructions more on each natively marked
                // request).
                let assign = rest.bytes().position(|byte| byte == b'=')?;
                Some((end - rest[assign + 1..].trim_start().len(), end))
            });
            let spelled = match &set.expr {
                Expr::Const(literal) => literal
                    .value
                    .as_str()
                    .is_none_or(|text| source.contains(text)),
                Expr::Var(_) => true,
                _ => false,
            };
            let (mut names, mut draws) = (Vec::new(), false);
            let expression = span
                .filter(|_| names_read(&set.expr, &mut names, &mut draws))
                .map(|span| Expression {
                    span,
                    names,
                    draws,
                    spelled,
                });
            let namespace = calls_namespace(&set.expr);
            sets.push(RootSet {
                name: target.id,
                expression,
                conditional,
                namespace,
            });
        }
        Stmt::Macro(declared) => sets.push(RootSet {
            name: declared.name,
            expression: None,
            conditional,
            namespace: false,
        }),
        Stmt::Import(import) => unread_bindings(&import.name, conditional, sets),
        Stmt::FromImport(import) => {
            for (name, alias) in &import.names {
                unread_bindings(alias.as_ref().unwrap_or(name), conditional, sets);
            }
        }
        _ => {}
    }
}

/// Whether the load can compute `expr` from literals and names: by what the
/// syntax itself does (operators, subscripts, the methods of such values,
/// `'<{}>'.format(suffix)`), by filters and by tests. The names it reads go
/// to `names`. `draws` is set where a filter or a test takes part: one may
/// answer anew at each render (`random`), so such a value is computed and
/// never kept. Any other call is beyond the load: a function, a macro or a
/// block reads what it likes.
fn names_read<'a>(expr: &Expr<'a>, names: &mut Vec<&'a str>, draws: &mut bool) -> bool {
    match expr {
        Expr::Var(var) => {
            names.push(var.id);
            true
        }
        Expr::Const(_) => true,
        Expr::UnaryOp(op) => names_read(&op.expr, names, draws),
        Expr::BinOp(op) => {
            names_read(&op.left, names, draws) && names_read(&op.right, names, draws)
        }
        Expr::Compare(compare) => {
            names_read(&compare.expr, names, draws)
                && compare
                    .ops
                    .iter()
                    .all(|op| names_read(&op.expr, names, draws))
        }
        Expr::IfExpr(cond) => {
            names_read(&cond.test_expr, names, draws)
                && names_read(&cond.true_expr, names, draws)
                && cond
                    .false_expr
                    .as_ref()
                    .is_none_or(|expr| names_read(expr, names, draws))
        }
        Expr::GetAttr(attr) => names_read(&attr.expr, names, draws),
        Expr::GetItem(item) => {
            names_read(&item.expr, names, draws) && names_read(&item.subscript_expr, names, draws)
        }
        Expr::Slice(slice) => {
            names_read(&slice.expr, names, draws)
                && [&slice.start, &slice.stop, &slice.step]
                    .into_iter()
                    .flatten()
                    .all(|bound| names_read(bound, names, draws))
        }
        Expr::List(list) => list.items.iter().all(|item| names_read(item, names, draws)),
        Expr::Map(map) => map
            .keys
            .iter()
            .chain(&map.values)
            .all(|expr| names_read(expr, names, draws)),
        // A method of a value, as the engine itself reads the call: a call
        // through `self` runs a block of the template.
        Expr::Call(call) => {
            matches!(call.identify_call(), CallType::Method(..))
                && names_read(&call.expr, names, draws)
                && arguments_read(&call.args, names, draws)
        }
        Expr::Filter(filter) => {
            *draws = true;
            filter
                .expr
                .as_ref()
                .is_none_or(|expr| names_read(expr, names, draws))
                && arguments_read(&filter.args, names, draws)
        }
        Expr::Test(test) => {
            *draws = true;
            names_read(&test.expr, names, draws) && arguments_read(&test.args, names, draws)
        }
    }
}

/// [`names_read`] of `args`, the arguments of a call, a filter or a test.
fn arguments_read<'a>(args: &[CallArg<'a>], names: &mut Vec<&'a str>, draws: &mut bool) -> bool {
    args.iter().all(|arg| match arg {
        CallArg::Pos(expr)
        | CallArg::Kwarg(_, expr)
        | CallArg::PosSplat(expr)
        | CallArg::KwargSplat(expr) => names_read(expr, names, draws),
    })
}

/// The names `target` binds, in a block `set` or a `set` of several names at
/// once, as root sets whose value nothing reads ([`RootSet::expression`]).
fn unread_bindings<'a>(target: &Expr<'a>, conditional: bool, sets: &mut Vec<RootSet<'a>>) {
    match target {
        Expr::Var(var) => sets.push(RootSet {
            name: var.id,
            expression: None,
            conditional,
            namespace: false,
        }),
        Expr::List(list) => {
            for item in &list.items {
                unread_bindings(item, conditional, sets);
            }
        }
        _ => {}
    }
}

/// Stands before each value the load reads back from a render of a
/// template's own statements (see [`ChatTemplateState::control_bindings`]):
/// the unit separator, an ASCII control character no template writes.
const VALUE_SEPARATOR: u8 = 0x1f;

/// Stands in that render for a value that is no string: the record
/// separator, one byte that is no control spelling and is not empty.
const NOT_A_STRING: u8 = 0x1e;

/// A root `set` statement that binds control spellings for every render:
/// the byte span of its expression, its value, and where the control
/// spellings stand in the value. One spelling is the whole value, or the
/// value begins with one and ends with one ([`several_controls`]).
struct ControlBinding {
    span: (usize, usize),
    value: String,
    controls: Vec<Range<usize>>,
}

impl ControlBinding {
    /// The control spellings of the value, in its order.
    fn spellings(&self) -> impl Iterator<Item = &str> {
        self.controls
            .iter()
            .map(|control| &self.value[control.clone()])
    }
}

/// A root `set` statement whose value the load computes
/// ([`computed_statements`]).
struct Computed<'a> {
    name: &'a str,
    /// The byte span of its expression.
    span: (usize, usize),
    /// Its value is the same at every render: it may stand as a literal.
    fixed: bool,
    /// Its expression builds the value: the source does not spell it
    /// already ([`Expression::spelled`]).
    builds: bool,
    /// The names its expression reads.
    names: Vec<&'a str>,
}

/// The root `set` statements among `sets` ([`root_sets`]) whose value the
/// load computes, in the template's order; and the names that some
/// statement binds to no one value.
///
/// The load computes a statement whose expression reads nothing of a
/// request ([`names_read`]): literals, and names that such statements bound
/// before it. The value is the same at every render, `fixed`, when also no
/// filter or test takes part in it and each name it reads was bound at the
/// top level by a statement of that kind: a name bound under an `if`, by a
/// statement that draws, or in a way nothing reads has no one value from
/// there on. A value that a request argument may carry is never computed
/// (`{% set close = custom if custom is defined else '</s>' %}`), and its
/// name is the request's to set, whatever else binds it
/// (`{% if custom is defined %}{% set close = custom %}{% endif %}` after
/// `{% set close = '</s>' %}`).
fn computed_statements<'a>(sets: Vec<RootSet<'a>>) -> (Vec<Computed<'a>>, Vec<&'a str>) {
    // The names the load has a value for; `true` where every render sees
    // that value from here on.
    let mut known: HashMap<&str, bool> = HashMap::new();
    let mut unsettled = Vec::new();
    let mut computed = Vec::new();
    for set in sets {
        let expression = set
            .expression
            .filter(|expression| expression.names.iter().all(|name| known.contains_key(name)));
        let fixed = expression.as_ref().is_some_and(|expression| {
            !expression.draws
                && expression
                    .names
                    .iter()
                    .all(|name| known.get(name) == Some(&true))
        });
        match expression {
            Some(expression) => {
                known.insert(set.name, fixed && !set.conditional);
                computed.push(Computed {
                    name: set.name,
                    span: expression.span,
                    fixed,
                    builds: !expression.spelled,
                    names: expression.names,
                });
            }
            None => {
                known.remove(set.name);
            }
        }
        if !fixed {
            unsettled.push(set.name);
        }
    }
    (computed, unsettled)
}

/// Whether `expr` is a call of `namespace`.
fn calls_namespace(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Call(call) if matches!(&call.expr, Expr::Var(callee) if callee.id == "namespace")
    )
}

/// The names among `sets` ([`root_sets`]) that the top level binds by a call
/// of `namespace`, once, where it always runs, and in no other way.
fn namespaces<'a>(sets: &[RootSet<'a>]) -> Vec<&'a str> {
    sets.iter()
        .filter(|set| set.namespace && !set.conditional)
        .filter(|set| sets.iter().filter(|other| other.name == set.name).count() == 1)
        .map(|set| set.name)
        .collect()
}

/// How [`built_values_read_plainly`] watches a name.
#[derive(Clone, Copy, PartialEq)]
enum Watched {
    /// Its value may hold a control spelling between marks: a value a
    /// top-level statement builds, a special token, a name bound from a join
    /// that holds one of these, a macro that writes one.
    Carrier,
    /// A built value is made from it.
    Part,
    /// A namespace the top level binds once and nothing else binds
    /// ([`namespaces`]): read as a template likes, until a top-level
    /// statement binds one of its attributes from a carrier.
    Namespace,
    /// Such a namespace from then on: each of its attributes stands where
    /// a carrier may stand (`ns.attr`), the name by itself only under a
    /// truth test.
    Holder,
}

/// The names watched where the walk of [`built_values_read_plainly`] stands.
type Names<'a> = HashMap<&'a str, Watched>;

/// Whether the syntax of a template shows that a copy with its control
/// spellings between marks would write what the template writes, at every
/// render: `ast` is the template, `computed` its top-level statements the
/// load computes ([`computed_statements`]).
///
/// One of those statements that builds its value may bind a control
/// spelling, which the marked copy holds between marks, and a render gives
/// each of the four special tokens between marks too. Such a name is a
/// carrier; so is a name a plain `set` binds from a join that holds one, and
/// a macro that writes one. Whether a built value is a control spelling is
/// read later, with the vocabulary: one that is not empty and does not
/// begin and end with a control spelling leaves the template unmarked
/// there ([`ChatTemplateState::control_bindings`]). A carrier may stand in
/// three places:
///
/// - as a written value, alone or as a whole operand of a join: `~`, or
///   `+`, which joins two strings and fails on anything else, with the marks
///   as without them;
/// - the same way as the value of a plain `set` of one name;
/// - alone, or under `not`, `and` and `or`, in the condition of an `if`,
///   which asks only whether it is empty: the marks stand around a control
///   spelling, so a value is empty with them exactly when it is without.
///
/// A `set` of a namespace's attribute is no plain `set`: a statement
/// elsewhere reads what it bound. Where the top level binds a name once by
/// a call of `namespace` and a top-level statement binds one of its
/// attributes from a join that holds a carrier, the namespace is a holder:
/// each attribute stands in the three places, as the value of such a `set`
/// too, and in such a condition against the empty string, the same
/// question. An attribute bound from a carrier only inside a loop, a macro
/// or a block makes no holder, and fails as any other read; so does the
/// holder's name bound again to anything but a new namespace.
///
/// A name a built value is made from, a part, may stand in the statements
/// that build and in such a condition. Any other statement that reads it
/// could put together a control the load never computed.
///
/// Every other read fails, wherever it stands and whatever a render would
/// pass: a filter, a test, a comparison, a subscript, an argument, a list,
/// a loop's sequence, a block that keeps or changes what is written inside
/// it. So does a statement that builds a value the load cannot know as one
/// value ([`Computed::fixed`]). The names are the engine's: the body of a
/// loop, of a macro and of a `block` has names of its own. A loop's target
/// and a macro's parameter take the place of a watched name there; a `set`
/// inside takes it only from where it runs, so the name stays watched.
fn built_values_read_plainly<'a>(
    ast: &'a Stmt<'a>,
    computed: &[Computed<'a>],
    namespaces: &[&'a str],
) -> bool {
    let Stmt::Template(template) = ast else {
        return false;
    };
    let mut names = Names::new();
    let mut built = HashSet::new();
    for statement in computed.iter().filter(|statement| statement.builds) {
        if !statement.fixed {
            return false;
        }
        names.insert(statement.name, Watched::Carrier);
        built.insert(statement.span.1);
    }
    if built.is_empty() {
        return false;
    }
    for name in SPECIAL_TOKEN_NAMES {
        names.insert(name, Watched::Carrier);
    }
    // What a built value is made from, through every statement between.
    for statement in computed.iter().rev() {
        if statement.builds || names.get(statement.name) == Some(&Watched::Part) {
            for name in &statement.names {
                names.entry(name).or_insert(Watched::Part);
            }
        }
    }
    for name in namespaces {
        names.entry(name).or_insert(Watched::Namespace);
    }
    scope_reads_plainly(&template.children, names, &built)
}

/// [`built_values_read_plainly`] of `body`, the statements of one scope,
/// among `names`; `built` holds the end of each statement that builds.
fn scope_reads_plainly<'a>(
    body: &'a [Stmt<'a>],
    mut names: Names<'a>,
    built: &HashSet<usize>,
) -> bool {
    // A carrier may be bound from another, and by a statement that stands
    // behind the read: a loop's body runs again.
    // A namespace that becomes a holder keeps its entry: the watched names
    // are counted, not the entries.
    let watched = |names: &Names| {
        names
            .values()
            .filter(|kind| **kind != Watched::Namespace)
            .count()
    };
    loop {
        let known = watched(&names);
        bind_carriers(body, &mut names);
        if watched(&names) == known {
            break;
        }
    }
    statements_read_plainly(body, &names, built)
}

/// Adds to `names` the carriers the statements of one scope bind: the name
/// of a plain `set` whose value reads a carrier, and a macro whose body
/// reads one. A namespace becomes a holder where a `set` binds one of its
/// attributes from a value that reads a carrier.
fn bind_carriers<'a>(body: &'a [Stmt<'a>], names: &mut Names<'a>) {
    for stmt in body {
        match stmt {
            Stmt::Set(set) => match &set.target {
                Expr::Var(target) if expression_reads(&set.expr, names, false) => {
                    names.entry(target.id).or_insert(Watched::Carrier);
                }
                // A namespace's attribute bound from a carrier: it holds one.
                target => {
                    if let Some(namespace) = attribute_of(target, names, Watched::Namespace)
                        .filter(|_| expression_reads(&set.expr, names, false))
                    {
                        names.insert(namespace, Watched::Holder);
                    }
                }
            },
            Stmt::Macro(declared)
                if statements_read(&declared.body, &without(names, &declared.args), false) =>
            {
                names.entry(declared.name).or_insert(Watched::Carrier);
            }
            // The statements that run among this scope's names.
            Stmt::IfCond(cond) => {
                bind_carriers(&cond.true_body, names);
                bind_carriers(&cond.false_body, names);
            }
            Stmt::ForLoop(for_loop) => bind_carriers(&for_loop.else_body, names),
            _ => {}
        }
    }
}

/// Whether the statements of one scope read `names` plainly
/// ([`built_values_read_plainly`]).
fn statements_read_plainly<'a>(
    body: &'a [Stmt<'a>],
    names: &Names<'a>,
    built: &HashSet<usize>,
) -> bool {
    body.iter().all(|stmt| match stmt {
        Stmt::EmitRaw(_) => true,
        Stmt::EmitExpr(emit) => joins(&emit.expr, names, false),
        Stmt::Set(set) if rebinds_holder(set, names) => false,
        // A statement that builds reads its parts as it likes.
        Stmt::Set(set) if matches!(set.target, Expr::Var(_)) => joins(
            &set.expr,
            names,
            built.contains(&(set.span().end_offset as usize)),
        ),
        // A holder's attribute is bound as a name is.
        Stmt::Set(set) if attribute_of(&set.target, names, Watched::Holder).is_some() => {
            joins(&set.expr, names, false)
        }
        Stmt::IfCond(cond) => {
            asks_whether_empty(&cond.expr, names)
                && statements_read_plainly(&cond.true_body, names, built)
                && statements_read_plainly(&cond.false_body, names, built)
        }
        Stmt::ForLoop(for_loop) => {
            !expression_reads(&for_loop.iter, names, true)
                && for_loop
                    .filter_expr
                    .as_ref()
                    .is_none_or(|filter| !expression_reads(filter, names, true))
                && scope_reads_plainly(
                    &for_loop.body,
                    without(names, std::slice::from_ref(&for_loop.target)),
                    built,
                )
                && statements_read_plainly(&for_loop.else_body, names, built)
        }
        Stmt::Macro(declared) => {
            declared
                .defaults
                .iter()
                .all(|default| !expression_reads(default, names, true))
                && scope_reads_plainly(&declared.body, without(names, &declared.args), built)
        }
        Stmt::Block(block) => scope_reads_plainly(&block.body, without(names, &[]), built),
        _ => !statements_read(std::slice::from_ref(stmt), names, true),
    })
}

/// Whether `set` binds a holder's name to anything but a new namespace: the
/// name would then name a namespace that another name reads.
fn rebinds_holder(set: &Set, names: &Names) -> bool {
    matches!(&set.target, Expr::Var(target) if names.get(target.id) == Some(&Watched::Holder))
        && !calls_namespace(&set.expr)
}

/// The namespace whose attribute `expr` reads or binds (`ns.attr`), where
/// `names` watch it as `kind`.
fn attribute_of<'a>(expr: &Expr<'a>, names: &Names<'a>, kind: Watched) -> Option<&'a str> {
    match expr {
        Expr::GetAttr(attr) => match &attr.expr {
            Expr::Var(var) if names.get(var.id) == Some(&kind) => Some(var.id),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `expr`, a written value, the value of a plain `set` or the value
/// a `set` binds to a holder's attribute, reads `names` plainly: a carrier
/// or a holder's attribute alone or as a whole operand of `~` or `+`, or a
/// carrier called (a macro's text) with arguments that read none; a part
/// only in a statement that `builds`. A holder is read by its attributes,
/// never by its name alone.
fn joins<'a>(expr: &'a Expr<'a>, names: &Names<'a>, builds: bool) -> bool {
    match expr {
        Expr::Var(var) => match names.get(var.id) {
            Some(Watched::Part) => builds,
            // A holder is read by its attributes.
            Some(Watched::Holder) => false,
            _ => true,
        },
        Expr::GetAttr(_) if attribute_of(expr, names, Watched::Holder).is_some() => true,
        Expr::BinOp(op) if matches!(op.op, BinOpKind::Concat | BinOpKind::Add) => {
            joins(&op.left, names, builds) && joins(&op.right, names, builds)
        }
        Expr::Call(call) if matches!(&call.expr, Expr::Var(var) if names.get(var.id) == Some(&Watched::Carrier)) =>
        {
            let mut read = Nodes::default();
            read.args(&call.args);
            !any_watched(&read.names, names, true)
        }
        _ => !expression_reads(expr, names, !builds),
    }
}

/// Whether `expr`, the condition of an `if`, reads `names` only to ask
/// whether one is empty: a name or a holder's attribute alone, such an
/// attribute against the empty string (`ns.attr == ''` or `!= ''`), or
/// under `not`, `and` and `or`.
fn asks_whether_empty<'a>(expr: &'a Expr<'a>, names: &Names<'a>) -> bool {
    match expr {
        Expr::Var(_) => true,
        Expr::GetAttr(_) if attribute_of(expr, names, Watched::Holder).is_some() => true,
        // A holder's attribute against the empty string: the same question.
        Expr::BinOp(op)
            if matches!(op.op, BinOpKind::Eq | BinOpKind::Ne)
                && attribute_of(&op.left, names, Watched::Holder).is_some()
                && matches!(&op.right, Expr::Const(empty) if empty.value.as_str() == Some("")) =>
        {
            true
        }
        Expr::UnaryOp(op) if matches!(op.op, UnaryOpKind::Not) => {
            asks_whether_empty(&op.expr, names)
        }
        Expr::BinOp(op) if matches!(op.op, BinOpKind::ScAnd | BinOpKind::ScOr) => {
            asks_whether_empty(&op.left, names) && asks_whether_empty(&op.right, names)
        }
        _ => !expression_reads(expr, names, true),
    }
}

/// Whether `expr` reads one of `names` anywhere ([`any_watched`]).
fn expression_reads<'a>(expr: &'a Expr<'a>, names: &Names<'a>, parts: bool) -> bool {
    let mut read = Nodes::default();
    read.expr(expr);
    any_watched(&read.names, names, parts)
}

/// Whether `stmts` read one of `names` anywhere ([`any_watched`]); a name
/// they bind counts as read.
fn statements_read<'a>(stmts: &'a [Stmt<'a>], names: &Names<'a>, parts: bool) -> bool {
    let mut read = Nodes::default();
    read.stmts(stmts, &mut Vec::new());
    any_watched(&read.names, names, parts)
}

/// Whether one of `read`, the names a piece of a template reads, is watched
/// among `names`: as a carrier or a holder or, where `parts` count, as a
/// part.
fn any_watched(read: &HashSet<&str>, names: &Names, parts: bool) -> bool {
    read.iter().any(|name| match names.get(name) {
        Some(Watched::Carrier | Watched::Holder) => true,
        Some(Watched::Part) => parts,
        Some(Watched::Namespace) | None => false,
    })
}

/// `names` without those that `targets` name: a loop's targets and a macro's
/// parameters are the body's own names from its first statement.
fn without<'a>(names: &Names<'a>, targets: &'a [Expr<'a>]) -> Names<'a> {
    let mut bound = Nodes::default();
    for target in targets {
        bound.expr(target);
    }
    let mut inner = names.clone();
    // A namespace becomes a holder at the top level alone: inside, a `set` of
    // its attribute from a carrier fails as before.
    inner.retain(|name, kind| !bound.names.contains(name) && *kind != Watched::Namespace);
    inner
}

/// The quote of a plain string literal of a template's source that holds
/// `value`. `None` for a value none holds (a backslash, a control
/// character, both kinds of quote).
fn literal_quote(value: &str) -> Option<char> {
    let holds = |quote: u8| {
        !value
            .bytes()
            .any(|byte| byte == quote || byte == b'\\' || byte < b' ')
    };
    if holds(b'\'') {
        Some('\'')
    } else if holds(b'"') {
        Some('"')
    } else {
        None
    }
}

/// The value of `binding` as a string literal of a template's source, each
/// of its control spellings between the marks of `encoders::marks`. `None`
/// for a value no plain literal holds ([`literal_quote`]).
fn marked_literal(binding: &ControlBinding) -> Option<String> {
    let value = binding.value.as_str();
    let quote = literal_quote(value)?;
    let mut literal = String::from(quote);
    let mut cursor = 0;
    for control in &binding.controls {
        literal.push_str(&value[cursor..control.start]);
        literal.push_str(&marks::mark(&value[control.clone()]));
        cursor = control.end;
    }
    literal.push_str(&value[cursor..]);
    literal.push(quote);
    Some(literal)
}

/// Where the control spellings stand in `value`, a string a statement
/// builds that is no control spelling whole: the added tokens `added` finds
/// in it, each as it stands, leftmost and the longest at one place. `None`
/// unless the value begins with one and ends with one. Text at an edge may
/// be a piece of a control: the template would complete that control where
/// it writes the value, and no mark would stand around it. Text between two
/// controls is closed in by them. A token whose entry carries a flag is
/// found as its spelling, and the read of the vocabulary refuses it
/// ([`contested`]).
fn several_controls(value: &str, added: &AddedTokens) -> Option<Vec<Range<usize>>> {
    let controls: Vec<_> = added.ranges(value).collect();
    let at_both_edges = controls.first()?.start == 0 && controls.last()?.end == value.len();
    at_both_edges.then_some(controls)
}

/// `source`, a chat template, with the expression of each of `bindings`
/// ([`ChatTemplateState::control_bindings`], in the template's order) replaced by its
/// value, the control spellings between marks ([`marked_literal`]). `None`
/// when a value has no literal.
fn with_marked_literals(source: &str, bindings: &[ControlBinding]) -> Option<String> {
    let mut marked = String::with_capacity(source.len());
    let mut cursor = 0;
    for binding in bindings {
        let (start, end) = binding.span;
        marked.push_str(&source[cursor..start]);
        marked.push_str(&marked_literal(binding)?);
        cursor = end;
    }
    marked.push_str(&source[cursor..]);
    Some(marked)
}

/// Whether the flat encode may read one of `spellings`, where a text holds
/// it, otherwise than alone and as it stands. `added` is every added token
/// of the vocabulary, each with a flag: whether the caller's flat encode
/// reads that token otherwise by itself. Its entry says so (the token takes
/// the whitespace beside it, it matches only as a whole word, or it is
/// matched after normalization), or the caller's encoder does not take the
/// longer of two added tokens that begin at one place and the token begins
/// with another one.
///
/// Between marks a spelling is read as its own token. The flat encode takes
/// the added tokens leftmost, each under its flags: first the ones matched
/// as they stand, then, in what is left between them, the ones matched
/// after normalization. So a flagged spelling is more than its characters
/// there, or none of them, or it is read only in the second pass, behind
/// every token of the first and wherever the text normalizes to it, or it
/// is read as the token it begins with; and a token that holds a spelling,
/// or that ends inside one, takes the spelling's place where the text
/// beside it completes that token. A token that stands inside a spelling
/// without a flag, or that begins inside one, never does: the spelling
/// starts first, or is the longer and is taken.
fn contested<'a>(spellings: &[&str], mut added: impl Iterator<Item = (&'a str, bool)>) -> bool {
    // The bytes a spelling begins with: a token meets a spelling only there.
    let mut begins = [false; 256];
    for spelling in spellings {
        if let Some(&first) = spelling.as_bytes().first() {
            begins[usize::from(first)] = true;
        }
    }
    added.any(|(token, flagged)| {
        if flagged && spellings.contains(&token) {
            return true;
        }
        let token = token.as_bytes();
        token.iter().enumerate().any(|(at, &byte)| {
            begins[usize::from(byte)]
                && spellings.iter().any(|spelling| {
                    let (spelling, rest) = (spelling.as_bytes(), &token[at..]);
                    if at == 0 {
                        // The token begins with the spelling and goes on.
                        rest.len() > spelling.len() && rest.starts_with(spelling)
                    } else {
                        // The token holds the spelling further in, or ends
                        // inside it.
                        rest.starts_with(spelling) || spelling.starts_with(rest)
                    }
                })
        })
    })
}

/// What `source`, a chat template, writes by itself: its raw text and its
/// string literals, each piece a text of its own, then each run of two or
/// more pieces that stand side by side, joined, and last every piece in
/// the order of the source, glued with nothing between. Between two pieces
/// of a run nothing lies but a comment, the marks of an expression or a
/// join, so a render that writes one writes the next right after it, and
/// a token the pieces spell together is written whole. Between two pieces
/// that follow each other in the source a render may write nothing (a
/// parenthesis, a block tag, an expression that comes out empty): the
/// glued text holds a token such pieces spell together, and in it a piece
/// stands between the pieces before and after it in the source. Each text
/// is asked by itself: a token that matches only as a whole word is
/// found in the piece that holds it. The glued
/// text is left out where it would repeat a text: after one piece alone,
/// or one run of every piece. The lexer reads under the whitespace rules
/// of the render ([`build_environment`]).
/// A comment is not in the texts, and neither is a literal that `built`
/// names by its byte offset: one inside an expression the template builds a
/// value from, the pattern of that value, which it writes only as the
/// value. A template the lexer refuses is read whole.
fn written_text(source: &str, built: impl Fn(usize) -> bool) -> Vec<String> {
    let mut written = Vec::new();
    // The runs of pieces that stand side by side, and how many pieces the
    // open run holds so far.
    let mut runs: Vec<String> = Vec::new();
    let mut pieces = 0_usize;
    // Every piece in the order of the source, with nothing between.
    let mut glued = String::with_capacity(source.len());
    // A newline after a block or a comment that ends a line is not written,
    // nor the blanks before a block that begins one.
    let whitespace = WhitespaceConfig {
        trim_blocks: true,
        lstrip_blocks: true,
        ..WhitespaceConfig::default()
    };
    for item in tokenize(source, false, SyntaxConfig, whitespace) {
        let Ok((token, span)) = item else {
            return vec![source.to_owned()];
        };
        let text = match &token {
            Token::TemplateData(text) | Token::Str(text) => Some(*text),
            Token::String(text) => Some(&**text),
            Token::VariableStart | Token::VariableEnd | Token::Tilde | Token::Plus => continue,
            _ => None,
        };
        let Some(text) = text.filter(|_| !built(span.start_offset as usize)) else {
            // The run ends here; one piece alone is no run.
            if pieces == 1 {
                runs.pop();
            }
            pieces = 0;
            continue;
        };
        written.push(text.to_owned());
        glued.push_str(text);
        if pieces == 0 {
            runs.push(String::new());
        }
        if let Some(run) = runs.last_mut() {
            run.push_str(text);
        }
        pieces += 1;
    }
    if pieces == 1 {
        runs.pop();
    }
    // One piece alone is the glued text, and so is a run of every piece.
    let repeats = written.len() < 2 || runs.iter().any(|run| run.len() == glued.len());
    written.extend(runs);
    if !repeats {
        written.push(glued);
    }
    written
}

/// Whether the texts of [`written_text`] spell one of `added`, the added
/// tokens, between them: the token begins with the end of one text, goes on
/// over whole texts, any number of them, and ends with the start of another.
/// A render writes the pieces in an order of its own: it leaves one out (a
/// branch), writes one again or never (a loop), or writes one that stands
/// elsewhere in the source (a macro, a block bound to a name). So any piece
/// may come right after any other, itself among them, and a token they
/// spell together there is written whole. The joined texts are pieces in
/// one such order. Each token is read as it is spelled.
fn spelled_across(written: &[String], added: &[(&str, bool)]) -> bool {
    let longest = added
        .iter()
        .map(|(token, _)| token.len())
        .max()
        .unwrap_or(0);
    // The ends of the texts a token may begin with: each is shorter than
    // the longest token.
    let ends: FxHashSet<&[u8]> = written
        .iter()
        .map(String::as_bytes)
        .flat_map(|text| {
            let from = text.len().saturating_sub(longest.saturating_sub(1));
            (from..text.len()).map(move |at| &text[at..])
        })
        .collect();
    added.iter().any(|(token, _)| {
        let token = token.as_bytes();
        // Where the token may go on: behind the end of a text, and behind
        // each whole text that follows one.
        let mut begun = vec![false; token.len()];
        for at in 1..token.len() {
            let (head, rest) = token.split_at(at);
            if !begun[at] && !ends.contains(head) {
                continue;
            }
            for text in written.iter().map(String::as_bytes) {
                if text.starts_with(rest) {
                    return true;
                }
                if !text.is_empty() && rest.starts_with(text) {
                    begun[at + text.len()] = true;
                }
            }
        }
        false
    })
}

/// The names a render gives the four special tokens a template reads
/// ([`RENDER_CONTEXT_NAMES`]). No template reads any other special token of
/// the vocabulary.
const SPECIAL_TOKEN_NAMES: [&str; 4] = ["bos_token", "eos_token", "unk_token", "pad_token"];

/// A conversation a template that builds its control spellings is checked on
/// before its marks are put in force (see
/// [`ChatTemplateState::mark_built_controls`]).
struct Probe {
    messages: Vec<JsonValue>,
    tools: Option<Vec<JsonValue>>,
    documents: Option<Vec<JsonValue>>,
}

impl Probe {
    /// The parameters the probe renders under at the instant `now` (a
    /// template may write the time): with the generation prompt or without,
    /// the thinking switch as the gateway passes it or leaves it.
    fn params<'a>(
        &'a self,
        special_tokens: &'a SpecialTokens,
        now: DateTime<FixedOffset>,
        (add_generation_prompt, thinking): (bool, Option<bool>),
    ) -> ChatTemplateParams<'a> {
        ChatTemplateParams {
            add_generation_prompt,
            tools: self.tools.as_deref(),
            documents: self.documents.as_deref(),
            special_tokens: Some(special_tokens),
            thinking,
            now: Some(now),
            ..Default::default()
        }
    }
}

/// The probe conversations, the bare turn first: a system message and a
/// second turn, a history message with reasoning, a tool round with its
/// tools (the call's arguments as the template takes them), a document.
fn probes(raw_tool_call_arguments: bool) -> [Probe; 5] {
    let text = "probe";
    let message = |role: &str| serde_json::json!({"role": role, "content": text});
    let conversation = |messages: Vec<JsonValue>| Probe {
        messages,
        tools: None,
        documents: None,
    };
    let arguments = serde_json::json!({"probe": text});
    let arguments = if raw_tool_call_arguments {
        JsonValue::String(arguments.to_string())
    } else {
        arguments
    };
    [
        conversation(vec![message("user")]),
        conversation(vec![
            message("system"),
            message("user"),
            message("assistant"),
            message("user"),
        ]),
        conversation(vec![
            message("user"),
            serde_json::json!({
                "role": "assistant", "content": text, "reasoning_content": text,
            }),
            message("user"),
        ]),
        Probe {
            tools: Some(vec![serde_json::json!({"type": "function", "function": {
                "name": "probe",
                "description": text,
                "parameters": {"type": "object", "properties": {"probe": {"type": "string"}}},
            }})]),
            ..conversation(vec![
                message("user"),
                serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "probe", "arguments": arguments},
                }]}),
                serde_json::json!({"role": "tool", "tool_call_id": "call_0", "content": text}),
                message("user"),
            ])
        },
        Probe {
            documents: Some(vec![serde_json::json!({"title": text, "text": text})]),
            ..conversation(vec![message("user")])
        },
    ]
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
    // Jinja2's `random` filter draws one element of a sequence or one character
    // of a string, afresh on every render, as the reference renderer draws it: a
    // template that draws a nonce after every role renders here too, its bytes
    // its own on each render by design. minijinja-contrib's filter also seeds
    // from a `RAND_SEED` integer in the render context, which a request's
    // `chat_template_kwargs` can supply; Jinja2's has no such seed.
    env.add_filter("random", python_random_filter);
    env.add_function("raise_exception", raise_exception);
    // Jinja2's dict() also builds a map from key/value pairs; minijinja's takes a mapping only
    env.add_function("dict", dict_function);
    env.add_test("iterable", is_iterable);

    Ok(env)
}

/// Jinja2's `random` filter is `random.choice(seq)`, a draw from any sized
/// sequence; minijinja-contrib's draws from a sequence or a string, so an
/// iterable (minijinja's `range()`, capped at 100,000 items) is drawn from as
/// its items.
fn python_random_filter(state: &State, value: &Value) -> Result<Value, MinijinjaError> {
    if value.kind() == ValueKind::Iterable {
        let items: Vec<Value> = value.try_iter()?.collect();
        return minijinja_contrib::filters::random(state, &Value::from(items));
    }
    minijinja_contrib::filters::random(state, value)
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
    // As in `tojson_filter`: no keyword, no recorded read.
    let attribute: Option<Cow<'_, str>> = if passed_no_keyword(&kwargs) {
        None
    } else {
        let attribute = kwargs.get("attribute")?;
        kwargs.assert_all_used()?;
        attribute
    };
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
    output_len: &RenderedLength,
) -> Result<(String, bool)> {
    continue_final_message(messages, params, |messages, params| {
        render_messages(env, messages.iter(), params, output_len).map(|(text, _)| text)
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
        // The reason stays in the chain under the text, so a caller still
        // reads the type of a refusal or of the template's own failure.
        let lost = format!(
            "{reason}: only its text could follow the generation prompt, and its \
             reasoning_content would be lost"
        );
        return Err(reason.context(lost));
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
    output_len: &RenderedLength,
) -> Result<(String, bool)> {
    if params.continue_final_message {
        return render_continuing_final_message(env, messages, params, output_len);
    }
    render_messages(env, messages.iter(), params, output_len)
}

/// Render `messages` through the template with the generation prompt the
/// params ask for; the flag says the prompt does not continue a message.
fn render_messages<'a>(
    env: &Environment<'_>,
    messages: impl Iterator<Item = &'a JsonValue>,
    params: ChatTemplateParams,
    output_len: &RenderedLength,
) -> Result<(String, bool)> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TemplateFailure(format!("Failed to get template: {e}")))?;

    // transformers adds `strftime_now` to the template environment; here it is
    // a callable of the render context, bound to the render's instant. The
    // instant is read when the template first writes it: a template that never
    // calls `strftime_now` costs its render no wall-clock read, and one that
    // calls it twice writes one instant.
    let instant = params.now.map_or_else(OnceLock::new, OnceLock::from);
    let strftime_now = Value::from_function(move |format: String| {
        strftime(instant.get_or_init(render_instant), &format)
    });

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

    // The render's names as one object (no map built per render): its own
    // names are read first, a caller's kwargs for a name left undefined or
    // not set, as the merge of two maps read them.
    let ctx = Value::from_object(RenderContext {
        messages: Value::from_object(minijinja_messages),
        add_generation_prompt: Value::from(params.add_generation_prompt),
        strftime_now,
        tools: tools_value,
        documents: documents_value,
        bos_token: bos_value,
        eos_token: eos_value,
        unk_token: unk_value,
        pad_token: pad_value,
        kwargs: params.template_kwargs.map(Value::from_serialize),
    });

    // Render into a buffer sized by the last render through this template:
    // the text grows past no capacity on the way.
    let mut rendered = Vec::with_capacity(output_len.hint());
    tmpl.render_captured_to(&ctx, &mut rendered).map_err(|e| {
        let message = format!("Failed to render template: {e}");
        // The template's own refusal keeps its type, at any depth of the
        // error's sources (a macro or an include wraps it), and so does a
        // failure that is the template's own whatever the caller handed it.
        // Any other stays the plain failure it was.
        let raised = iter::successors(std::error::Error::source(&e), |cause| cause.source())
            .any(|cause| cause.is::<RaisedByTemplate>());
        if raised {
            anyhow::Error::new(TemplateRefusal(message))
        } else if is_the_templates_own(&e, &tmpl) {
            anyhow::Error::new(TemplateFailure(message))
        } else {
            anyhow!(message)
        }
    })?;
    // The engine writes its output a `str` at a time (the template's text,
    // and each value formatted to text), so no caller's value makes the
    // whole anything but UTF-8: a failure here is the render's own.
    let mut rendered = String::from_utf8(rendered).map_err(|e| {
        TemplateFailure(format!(
            "Failed to render template: the output is not UTF-8: {e}"
        ))
    })?;
    output_len.record(rendered.len());
    // The text travels with the request: a render much shorter than the hint
    // gives the slack back, or a short prompt after a long one would carry the
    // long one's buffer until its request drops.
    if rendered.capacity() > 2 * rendered.len() + RENDERED_SLACK {
        rendered.shrink_to_fit();
    }

    Ok((rendered, false))
}

/// The names a template reads, as the render's root context: the nine the
/// render sets, read first, then the caller's `template_kwargs` for a name
/// the render leaves undefined (an unset special token) or does not set, as
/// the merge of the two context maps read them (the render's map first, so a
/// kwarg never replaces the request's own fields). One object per render
/// instead of a map of pairs grown and hashed per render; a lookup is a match
/// on the name.
#[derive(Debug)]
struct RenderContext {
    messages: Value,
    add_generation_prompt: Value,
    strftime_now: Value,
    tools: Value,
    documents: Value,
    bos_token: Value,
    eos_token: Value,
    unk_token: Value,
    pad_token: Value,
    /// The caller's `template_kwargs` as one map value, when it gave any.
    kwargs: Option<Value>,
}

/// The names [`RenderContext`] sets, in the order a map built of them held.
const RENDER_CONTEXT_NAMES: &[&str] = &[
    "messages",
    "add_generation_prompt",
    "strftime_now",
    "tools",
    "documents",
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
];

impl RenderContext {
    fn base(&self, name: &str) -> Option<&Value> {
        Some(match name {
            "messages" => &self.messages,
            "add_generation_prompt" => &self.add_generation_prompt,
            "strftime_now" => &self.strftime_now,
            "tools" => &self.tools,
            "documents" => &self.documents,
            "bos_token" => &self.bos_token,
            "eos_token" => &self.eos_token,
            "unk_token" => &self.unk_token,
            "pad_token" => &self.pad_token,
            _ => return None,
        })
    }
}

impl Object for RenderContext {
    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
        let base = self.base(key);
        let Some(kwargs) = &self.kwargs else {
            return base.cloned();
        };
        if let Some(value) = base.filter(|value| !value.is_undefined()) {
            return Some(value.clone());
        }
        kwargs
            .get_attr(key)
            .ok()
            .filter(|value| !value.is_undefined())
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.get_value_by_str(key.as_str()?)
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        let Some(kwargs) = &self.kwargs else {
            return Enumerator::Str(RENDER_CONTEXT_NAMES);
        };
        let mut names: Vec<Value> = RENDER_CONTEXT_NAMES
            .iter()
            .copied()
            .map(Value::from)
            .collect();
        if let Ok(keys) = kwargs.try_iter() {
            names.extend(keys.filter(|key| {
                !key.as_str()
                    .is_some_and(|name| RENDER_CONTEXT_NAMES.contains(&name))
            }));
        }
        Enumerator::Values(names)
    }
}

/// The length of the last text a template rendered, the capacity the next
/// render starts from: a prompt is about as long as the previous one through
/// the same template, so the output grows past no capacity on the way. Capped,
/// so one very long prompt does not size every render after it; a render much
/// shorter than the hint gives the slack back (`RENDERED_SLACK`), so the text
/// that travels with a request is never much larger than itself.
#[derive(Debug, Default)]
pub(crate) struct RenderedLength(AtomicUsize);

/// The most a render pre-allocates from the hint, whatever the last length.
const RENDERED_LENGTH_CAP: usize = 64 * 1024;

/// The capacity a rendered text keeps beyond twice its length; more than that
/// is given back before the text leaves the render.
const RENDERED_SLACK: usize = 4 * 1024;

impl RenderedLength {
    /// The capacity the next render starts from.
    fn hint(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    /// Records a render's length for the next one.
    fn record(&self, len: usize) {
        self.0
            .store(len.min(RENDERED_LENGTH_CAP), Ordering::Relaxed);
    }
}

/// The messages with every string `content` of a non-tool message replaced by
/// a one-item text part list, or `None` when no message has one.
pub(crate) fn string_content_as_text_parts(
    messages: &[serde_json::Value],
) -> Option<Vec<serde_json::Value>> {
    if !messages.iter().any(needs_text_part_wrap) {
        return None;
    }
    let mut wrapped = messages.to_vec();
    wrap_string_content_in_place(&mut wrapped);
    Some(wrapped)
}

/// Whether the OpenAI-format render wraps `message`'s content: a string, on
/// any message but a tool result (the engine joins a result's parts back
/// into one string, so a string it stays).
fn needs_text_part_wrap(message: &serde_json::Value) -> bool {
    message
        .get("content")
        .is_some_and(serde_json::Value::is_string)
        && message.get("role").and_then(serde_json::Value::as_str) != Some("tool")
}

/// [`string_content_as_text_parts`] on messages the caller owns: every
/// string `content` of a non-tool message becomes a one-item text part list
/// where it stands, and no message is copied.
pub(crate) fn wrap_string_content_in_place(messages: &mut [serde_json::Value]) {
    for message in messages.iter_mut() {
        if !needs_text_part_wrap(message) {
            continue;
        }
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        let text = content.take();
        *content = serde_json::json!([{ "type": "text", "text": text }]);
    }
}

/// Chat template processor using Jinja2 - simple wrapper like HuggingFace
pub struct ChatTemplateProcessor {
    env: Environment<'static>,
    /// The last rendered length, the next render's starting capacity.
    output_len: RenderedLength,
}

impl ChatTemplateProcessor {
    /// Create a new chat template processor.
    ///
    /// Returns an error if the template fails to parse, so callers get an
    /// actionable message immediately rather than a confusing "template not
    /// found" error on the first render.
    pub fn new(template: String) -> Result<Self> {
        let env = build_environment(template)?;
        Ok(ChatTemplateProcessor {
            env,
            output_len: RenderedLength::default(),
        })
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
        render_chat_template(&self.env, messages, params, &self.output_len)
            .map(|(prompt, _)| prompt)
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
    /// The template's thinking switch and its default: what the template
    /// renders with the switch absent, or the text matcher's answer where
    /// the renders do not say (see `rendered_thinking_default`).
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
    /// The last rendered length, the next render's starting capacity.
    output_len: RenderedLength,
    /// The processor's template of a checkpoint that ships one beside the
    /// tokenizer's and differs from it (see [`Self::from_checkpoint`]),
    /// rendered in place of the default one when the request declares no
    /// tools, as the engine's server resolves a request's template: the
    /// request's own first, the processor's second unless tools are
    /// declared, the tokenizer's third. The accessors answer for the default
    /// template, which the processor's agrees with on the thinking answers;
    /// [`Self::content_format_for_tools`] answers for the one selected.
    processor: Option<Box<ChatTemplateState>>,
    /// The templates are compiled with their control spellings marked (see
    /// [`Self::mark_controls`]).
    marked: bool,
    /// Every name the default template reads (see [`Self::reads`]); `None`
    /// where the template did not parse, the names being unknown.
    names_read: Option<HashSet<String>>,
    /// Whether the root `set` statements of the default template build
    /// values and the template only writes them
    /// ([`built_values_read_plainly`]): only such a template may have its
    /// control spellings marked, and the load reads no other for one (see
    /// [`Self::control_bindings`]).
    writes_built_constants: bool,
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
        let Detected {
            content_format,
            think_in_prefill,
            thinking_toggle,
            thinking_key_name,
            prefill_opens_think_block,
            names_read,
            writes_built_constants,
        } = template.as_ref().map(|t| detect_all(t)).unwrap_or_default();
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
            output_len: RenderedLength::default(),
            processor: None,
            marked: false,
            names_read,
            writes_built_constants,
        };
        state.settle_by_behaviour();
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
            output_len: RenderedLength::default(),
            processor: None,
            marked: false,
            names_read: None,
            writes_built_constants: false,
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
        // A render call carries no template of its own, so the words name
        // what holds for every caller: the model has none, and only its
        // configuration can bring one.
        let env = self.env.as_ref().ok_or_else(|| {
            anyhow!(
                "The model has no chat template. One must be configured for the model; \
                 a request cannot supply it.",
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
                kwargs.insert(kwarg_key.to_string(), key.value(thinking));
                let params = ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    thinking: None,
                    ..params
                };
                return render_chat_template(env, messages, params, &self.output_len);
            }
        }

        render_chat_template(env, messages, params, &self.output_len)
    }

    /// Compiles the templates (this one, the `tool_use` one, the processor's)
    /// with every control spelling `controls` knows between the marks of
    /// `encoders::marks`, so a render tells the control tokens the template
    /// wrote from the same spellings in request text, which the renderer
    /// strips of marks first. For a checkpoint whose reference library writes
    /// its control tokens structurally and encodes every request string as
    /// plain text. A template that names a spelling in a comparison rather
    /// than in its output would compare against the marked spelling; the
    /// templates of such a family write their spellings only.
    pub(crate) fn mark_controls(&mut self, controls: &AddedTokens) -> Result<()> {
        if let Some(env) = &self.env {
            let source = env.get_template("chat")?.source().to_owned();
            if let Cow::Owned(marked) = marks::mark_controls(&source, controls) {
                self.env = Some(build_environment(marked)?);
            }
        }
        self.marked = true;
        for nested in [&mut self.tool_use, &mut self.processor]
            .into_iter()
            .flatten()
        {
            nested.mark_controls(controls)?;
        }
        Ok(())
    }

    /// Whether the templates are compiled marked (see [`Self::mark_controls`]).
    pub(crate) fn marked(&self) -> bool {
        self.marked
    }

    /// Marks the control spellings the templates build. A template may never
    /// spell a control token whole: it binds each to a name once, out of a
    /// constant of its own (`'<open{}>'.format(suffix)`), and writes the
    /// names. [`Self::mark_controls`] finds no spelling in such a source, so
    /// the load reads the bindings instead ([`Self::control_bindings`]): a
    /// root `set` statement whose value is the same at every render and is
    /// one added token (`is_control`) binds a control spelling, and a copy
    /// of the source takes that spelling, between the marks of
    /// `encoders::marks`, as a literal in the statement's place. A value
    /// such a statement builds that begins with an added token and ends
    /// with one holds several, and takes the marks around each
    /// ([`several_controls`]). A statement beside them that spells one
    /// control whole in a literal of its own binds it the same way, where
    /// its name is read as theirs. Compiled
    /// so, the template writes control tokens where it writes the names,
    /// alone or inside a longer value, and the same spellings in request
    /// text stay text, as under [`Self::mark_controls`]; a render builds no
    /// spelling again.
    ///
    /// The templates (this one, the `tool_use` one, the processor's) are
    /// marked together or not at all, and only where the syntax shows that
    /// the marks change nothing a render reads
    /// ([`built_values_read_plainly`]): each value the top-level statements
    /// build is only written, joined or asked whether it is empty, and what
    /// it is built from is read by nothing else. A template that reads one
    /// any other way (a filter, a comparison, a search, a slice), binds one
    /// from a request argument, builds a value that is not empty and does
    /// not begin and end with a control spelling, or writes an added token
    /// whole by itself (in its text or in a string literal, not in a
    /// comment; in one piece, or in pieces a render may write one right
    /// after another, see [`spelled_across`]) stays as it is, its prompts
    /// encoded from their text as before. So does one that writes a control
    /// outside its constants on a probe conversation
    /// ([`Self::marks_every_control`]). Every reading is
    /// made through copies of the templates ([`Self::twin`]), so the render
    /// of a request carries nothing for it. `controls`, the matcher over the
    /// added tokens, is asked for only once a template binds a control
    /// spelling it does not write whole by itself, or builds a value a
    /// plain literal holds that is neither a control spelling nor empty,
    /// so a checkpoint without either never builds it. `as_spelled` says
    /// whether the flat encode matches every added token as it is spelled.
    /// It is asked at such a built value, which is bound only then: the
    /// text between two controls of a bound value is read by its
    /// characters, and the flat encode reads it so only then. It is asked
    /// again once a template binds a control spelling: under a pipeline
    /// that matches an added token after normalization the flat encode
    /// reads as that token a text the normalizer turns into its spelling,
    /// and the templates stay as they are.
    ///
    /// Before that the vocabulary is read once: `added` is every added
    /// token, with whether its entry carries a flag. Where the flat encode
    /// may read a spelling that would stand between marks otherwise than
    /// alone and as it stands ([`contested`]), the templates stay as they
    /// are: a prompt with the marks would read that spelling as its own
    /// token everywhere.
    /// Returns the spellings the templates now hold between marks, the
    /// special tokens a render gives by name among them, so that the caller
    /// can tell which added tokens a request may still type as tokens;
    /// `None` when the templates stay as they are.
    pub(crate) fn mark_built_controls<'a, 'b, I>(
        &mut self,
        is_control: &dyn Fn(&str) -> bool,
        added: impl FnOnce() -> I,
        controls: impl Fn() -> Option<&'a AddedTokens>,
        as_spelled: &dyn Fn() -> bool,
        special_tokens: &SpecialTokens,
    ) -> Option<Vec<String>>
    where
        I: Iterator<Item = (&'b str, bool)>,
    {
        // A built value of several controls is read with the matcher only
        // where the flat encode matches every added token as it is spelled.
        let several = || as_spelled().then(&controls).flatten();
        let own = self.control_bindings(is_control, &several, special_tokens);
        if own.is_empty() {
            return None;
        }
        if !as_spelled() {
            tracing::debug!(
                "the chat template binds control spellings and stays unmarked: the flat encode \
                 matches an added token after normalization"
            );
            return None;
        }
        let nested: Vec<_> = [&self.tool_use, &self.processor]
            .into_iter()
            .flatten()
            .map(|state| state.control_bindings(is_control, &several, special_tokens))
            .collect();
        // Every spelling that would stand between marks: what the templates
        // bind, and the special tokens a render gives by name.
        let named = [
            &special_tokens.bos_token,
            &special_tokens.eos_token,
            &special_tokens.unk_token,
            &special_tokens.pad_token,
        ];
        let spellings: Vec<&str> = own
            .iter()
            .chain(nested.iter().flatten())
            .flat_map(ControlBinding::spellings)
            .chain(named.into_iter().flatten().map(String::as_str))
            .filter(|spelling| !spelling.is_empty())
            .collect();
        let added: Vec<_> = added().collect();
        if contested(&spellings, added.iter().copied()) {
            tracing::debug!(
                "the chat template binds control spellings and stays unmarked: the flat encode \
                 may read one of them otherwise than alone"
            );
            return None;
        }
        let controls = controls()?;
        let own = self.built_controls_marked(&own, controls, &added, special_tokens)?;
        let mut envs = Vec::new();
        for (state, bindings) in [&self.tool_use, &self.processor]
            .into_iter()
            .flatten()
            .zip(&nested)
        {
            envs.push(state.built_controls_marked(bindings, controls, &added, special_tokens)?);
        }
        self.env = Some(own);
        self.marked = true;
        for (state, env) in [&mut self.tool_use, &mut self.processor]
            .into_iter()
            .flatten()
            .zip(envs)
        {
            state.env = Some(env);
            state.marked = true;
        }
        tracing::info!(
            "the chat template builds its control spellings: the control tokens it writes are \
             marked, and the same spellings in a request are encoded as text"
        );
        Some(spellings.into_iter().map(str::to_owned).collect())
    }

    /// This template's state around `env`, without nested templates: a
    /// render through it takes `env` whatever the request declares, and
    /// prepares the request as this template's own renders do (the content
    /// shape, the roles, the thinking key, the form of a tool call's
    /// arguments).
    fn twin(&self, env: Environment<'static>) -> Self {
        Self {
            env: Some(env),
            content_format: self.content_format,
            thinking_toggle: self.thinking_toggle,
            thinking_key_name: self.thinking_key_name,
            think_in_prefill: self.think_in_prefill,
            prefill_opens_think_block: self.prefill_opens_think_block,
            developer_role_supported: self.developer_role_supported,
            root_role_supported: self.root_role_supported,
            raw_tool_call_arguments: self.raw_tool_call_arguments,
            tool_use: None,
            output_len: RenderedLength::default(),
            processor: None,
            marked: false,
            names_read: None,
            writes_built_constants: false,
        }
    }

    /// The root `set` statements of this template that bind control
    /// spellings for every render: those whose value is the same at every
    /// render ([`computed_statements`]) and is one control spelling
    /// (`is_control`), or a string the statement builds that begins with a
    /// control spelling and ends with one ([`several_controls`]), under a
    /// name no statement binds another way. The template engine says the
    /// values: the statements the load computes, alone, each followed by
    /// its value behind a separator ([`VALUE_SEPARATOR`]), are rendered
    /// once, the way a request's template is.
    ///
    /// None when one of those values holds a control spelling the load
    /// cannot write as a literal: its statement reads a name that has no
    /// one value, or another statement binds its name another way. The
    /// template would write that control without marks wherever no probe
    /// conversation goes, so it stays as it is. None when a statement
    /// builds any other value that is not empty, whether a string or not
    /// ([`NOT_A_STRING`]): it may be a piece of a control, which takes no
    /// marks, and the template would complete that control where it writes
    /// the piece. None as well when the template writes one of the
    /// spellings whole by itself, in its text or in a string literal
    /// ([`written_text`]): that template is of the other kind, and it says
    /// so here. A comment that names a spelling is nothing the template
    /// writes, and neither is the pattern a value is built from, nor the
    /// literal of a statement that spells one control whole. Such a
    /// statement binds its control like the ones that build theirs, where
    /// the syntax, read again with it counted among them, shows its name
    /// read as theirs ([`built_values_read_plainly`]); None where it does
    /// not. The first read of the syntax stands before that one and counts
    /// only the statements that build: a spelled control that a built value
    /// is made from is a part of that value there ([`Watched::Part`]), and a
    /// template that also writes it by its name stays as it is.
    ///
    /// `matcher`, the matcher over the added tokens, is asked for only at a
    /// value a statement builds that is neither a control spelling nor
    /// empty and that a plain literal holds ([`literal_quote`]): it says
    /// where the control spellings stand in that value. A default prompt
    /// with a line break in it, which the load takes for built, never asks.
    fn control_bindings<'a>(
        &self,
        is_control: &dyn Fn(&str) -> bool,
        matcher: &dyn Fn() -> Option<&'a AddedTokens>,
        special_tokens: &SpecialTokens,
    ) -> Vec<ControlBinding> {
        // Only a template that writes the values its root statements build
        // may be marked: nothing is read again for any other.
        let Some(plain) = self.env.as_ref().filter(|_| self.writes_built_constants) else {
            return Vec::new();
        };
        let Ok(template) = plain.get_template("chat") else {
            return Vec::new();
        };
        let source = template.source();
        let sets = root_sets(source);
        let namespaces = namespaces(&sets);
        let (mut computed, unsettled) = computed_statements(sets);
        let mut statements = String::new();
        for statement in &computed {
            let (start, end) = statement.span;
            statements.push_str("{% set ");
            statements.push_str(statement.name);
            statements.push_str(" = ");
            statements.push_str(&source[start..end]);
            statements.push_str(" %}");
            // A spelling is a string; a value of any other type is written
            // as the byte that stands for one.
            statements.push(char::from(VALUE_SEPARATOR));
            statements.push_str("{{ ");
            statements.push_str(statement.name);
            statements.push_str(" if ");
            statements.push_str(statement.name);
            statements.push_str(" is string else '");
            statements.push(char::from(NOT_A_STRING));
            statements.push_str("' }}");
        }
        // The statements stand in a copy of this template's environment, in
        // the template's place.
        let mut alone = plain.clone();
        if alone
            .add_template_owned("chat".to_owned(), statements)
            .is_err()
        {
            return Vec::new();
        }
        let messages = [serde_json::json!({"role": "user", "content": "probe"})];
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            special_tokens: Some(special_tokens),
            now: Some(render_instant()),
            ..Default::default()
        };
        let Ok((rendered, _)) = self.twin(alone).apply_with_continuation(&messages, params) else {
            return Vec::new();
        };
        // One value behind each separator, in the statements' order; before
        // the first the statements write nothing. A cut at an ASCII byte
        // leaves each piece the text it was.
        let values: Vec<&str> = rendered
            .split(char::from(VALUE_SEPARATOR))
            .skip(1)
            .collect();
        if values.len() != computed.len() {
            return Vec::new();
        }
        // What the template writes by itself, read once the source holds a
        // control spelling somewhere. The expression of a statement that
        // builds its value, or that spells one control whole, is no part of
        // it: the template writes that literal only as the value.
        let written_by_itself = || {
            let bound: Vec<_> = computed
                .iter()
                .zip(&values)
                .filter(|(statement, value)| statement.builds || is_control(value))
                .map(|(statement, _)| statement.span.0..statement.span.1)
                .collect();
            written_text(source, |at| bound.iter().any(|span| span.contains(&at)))
        };
        let mut written = None;
        let mut bindings = Vec::new();
        // The statements that spell one control whole in a literal of their
        // own.
        let mut spelled = Vec::new();
        for (index, (statement, &value)) in computed.iter().zip(&values).enumerate() {
            let whole = is_control(value);
            if !whole && (!statement.builds || value.is_empty()) {
                continue;
            }
            if !statement.fixed || unsettled.contains(&statement.name) {
                return Vec::new();
            }
            let controls = if whole {
                Some(iter::once(0..value.len()).collect())
            } else if literal_quote(value).is_none() {
                // No plain literal holds the value (one that is no string
                // among them): it could take no marks, and the matcher is
                // not asked.
                None
            } else {
                matcher().and_then(|added| several_controls(value, added))
            };
            // A built value that does not begin and end with a control may
            // be a piece of one: the template would complete that control
            // where it writes the piece, and no mark would stand around it.
            let Some(controls) = controls else {
                return Vec::new();
            };
            for control in &controls {
                // The source holds the spelling: the template may write it
                // whole by itself, in its text or in a string literal. In a
                // comment, or in the pattern a value is built from, it
                // writes nothing.
                let spelling = &value[control.clone()];
                if source.contains(spelling)
                    && written
                        .get_or_insert_with(written_by_itself)
                        .iter()
                        .any(|text| text.contains(spelling))
                {
                    return Vec::new();
                }
            }
            if !statement.builds {
                spelled.push(index);
            }
            bindings.push(ControlBinding {
                span: statement.span,
                value: value.to_owned(),
                controls,
            });
        }
        // A statement that spells one control whole binds it like the ones
        // that build theirs only where the template reads its name as it may
        // read theirs: the syntax is read again, with that statement counted
        // among the ones that build.
        if !spelled.is_empty() {
            for index in spelled {
                computed[index].builds = true;
            }
            let read_plainly = parse(source, "chat", SyntaxConfig {}, WhitespaceConfig::default())
                .is_ok_and(|ast| built_values_read_plainly(&ast, &computed, &namespaces));
            if !read_plainly {
                return Vec::new();
            }
        }
        bindings
    }

    /// This template compiled with the control spellings of `bindings`
    /// between marks, when the marks are complete (see
    /// [`Self::mark_built_controls`]).
    fn built_controls_marked(
        &self,
        bindings: &[ControlBinding],
        controls: &AddedTokens,
        added: &[(&str, bool)],
        special_tokens: &SpecialTokens,
    ) -> Option<Environment<'static>> {
        let plain = self.env.as_ref()?;
        let template = plain.get_template("chat").ok()?;
        let source = template.source();
        if bindings.is_empty() {
            return None;
        }
        let declined = |guard: &str| -> Option<Environment<'static>> {
            tracing::debug!(
                guard,
                "the chat template binds control spellings and stays unmarked"
            );
            None
        };
        // A template that writes an added token whole by itself is of the
        // other kind: its spellings would want the scan of `mark_controls`,
        // which is not made here. What the template writes is asked, not
        // its source: in a comment, or in the pattern a bound spelling is
        // built from, the template writes none, and pieces that stand side
        // by side write one whole, so they are asked joined and each by
        // itself, and every piece once more glued in the order of the
        // source: a render may write nothing between two that follow each
        // other. A render may also write one piece right after any other,
        // wherever the two stand in the source, so a token the pieces
        // spell between them in any order counts as written whole.
        let bound = |at| {
            bindings
                .iter()
                .any(|binding| (binding.span.0..binding.span.1).contains(&at))
        };
        let written = written_text(source, bound);
        if written
            .iter()
            .any(|text| controls.ranges(text).next().is_some())
            || spelled_across(&written, added)
        {
            return declined("it writes an added token whole by itself");
        }
        let Some(literals) = with_marked_literals(source, bindings) else {
            return declined("no string literal holds one of the spellings");
        };
        let Ok(marked) = build_environment(literals) else {
            return declined("the template does not compile with the spellings as literals");
        };
        let marked = self.twin(marked);
        if !self
            .twin(plain.clone())
            .marks_every_control(&marked, controls, special_tokens)
        {
            return declined(
                "the marked template does not stand for it on the probe conversations",
            );
        }
        marked.env
    }

    /// Whether `marked`, this template compiled with its control spellings
    /// between marks, stands for this one on the probe conversations; both
    /// are templates alone ([`Self::twin`]).
    ///
    /// The syntax already shows that `marked` writes this template's text at
    /// every render ([`built_values_read_plainly`]). What no syntax shows is
    /// a control the template puts together outside its constants, out of
    /// pieces of text side by side: on each conversation, with the
    /// generation prompt and without, under each value of the thinking
    /// switch the gateway passes by itself, `marked` must write the text
    /// this template writes, every added token in it between marks. Then no
    /// control the template writes there would be encoded as text, and a
    /// request that types no spelling is encoded as it was. The bare turn
    /// must render, and a control must be written.
    ///
    /// A conversation this template refuses is not judged; `marked` must
    /// refuse it as well.
    fn marks_every_control(
        &self,
        marked: &Self,
        controls: &AddedTokens,
        special_tokens: &SpecialTokens,
    ) -> bool {
        let marked_tokens = marks::marked_special_tokens(special_tokens);
        let now = render_instant();
        let with_prompt = (true, None);
        let switch: &[Option<bool>] = if self.thinking_key_name.is_some() {
            &[None, Some(true), Some(false)]
        } else {
            &[None]
        };
        let mut writes_a_control = false;
        for (index, probe) in probes(self.raw_tool_call_arguments).iter().enumerate() {
            for prompt in [true, false] {
                for thinking in switch {
                    let mode = (prompt, *thinking);
                    let Ok((expected, _)) = self.apply_with_continuation(
                        &probe.messages,
                        probe.params(special_tokens, now, mode),
                    ) else {
                        let refused = marked
                            .apply_with_continuation(
                                &probe.messages,
                                probe.params(&marked_tokens, now, mode),
                            )
                            .is_err();
                        if (index == 0 && mode == with_prompt) || !refused {
                            return false;
                        }
                        continue;
                    };
                    let Ok((rendered, _)) = marked.apply_with_continuation(
                        &probe.messages,
                        probe.params(&marked_tokens, now, mode),
                    ) else {
                        return false;
                    };
                    let Ok((text, written)) = marks::flatten(&rendered) else {
                        return false;
                    };
                    if text != expected || !controls.ranges(&text).eq(written.iter().cloned()) {
                        return false;
                    }
                    writes_a_control |= !written.is_empty();
                }
            }
        }
        writes_a_control
    }

    /// [`Self::set`] and [`Self::mark_controls`] as one step: the template
    /// is replaced only when it both builds and marks, so a template that
    /// fails to mark leaves the previous one in force, marks included (set
    /// first and marked after, a failed mark left the new template installed
    /// unmarked, and every later render read a typed control spelling as
    /// its token). The fresh state is what `set` would leave: the template's
    /// own answers, no nested templates, the length hint starting over.
    pub(crate) fn replace_marked(
        &mut self,
        template: String,
        controls: Option<&AddedTokens>,
    ) -> Result<()> {
        let mut fresh = Self::new(Some(template))?;
        if let Some(controls) = controls {
            fresh.mark_controls(controls)?;
        }
        *self = fresh;
        Ok(())
    }

    pub fn set(&mut self, template: String) -> Result<()> {
        let Detected {
            content_format,
            think_in_prefill,
            thinking_toggle,
            thinking_key_name,
            prefill_opens_think_block,
            names_read,
            writes_built_constants,
        } = detect_all(&template);
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
        self.names_read = names_read;
        self.writes_built_constants = writes_built_constants;
        self.env = Some(env);
        // A replacement template stands alone, as a template file does.
        self.tool_use = None;
        self.processor = None;
        self.marked = false;
        self.settle_by_behaviour();
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

    /// Whether a render through this state can read the name `name`: the
    /// default template reads it, or an alternate a request may render through
    /// does (the `tool_use` one, the processor's). The gateway hands the
    /// template a request control (`tool_choice`, `response_format`) under its
    /// name only when the template reads it: one no template reads cannot
    /// reach the prompt. Without a template every name reads as read, there
    /// being nothing to leave it out of; so does a template whose names are
    /// not known (one the detector could not parse).
    pub fn reads(&self, name: &str) -> bool {
        self.env.is_none()
            || self
                .names_read
                .as_ref()
                .is_none_or(|names| names.contains(name))
            || self
                .tool_use
                .as_ref()
                .is_some_and(|state| state.reads(name))
            || self
                .processor
                .as_ref()
                .is_some_and(|state| state.reads(name))
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

    /// Decide the thinking switch's default by the template's behaviour, not
    /// its source: one bare user turn with the generation prompt, rendered
    /// with the switch absent, on and off, at one instant (a template may
    /// write the time). The absent render is the default:
    /// on where it equals the on render alone, off where it equals the off
    /// render alone. `None` where the renders do not say, and the text
    /// matcher's answer stands ([`detect_thinking_toggle`]): a template with
    /// no switch, a bare turn that does not render, a switch that moves
    /// nothing on a bare turn, an absent render that is a prompt of its own.
    fn rendered_thinking_default(&self) -> Option<ThinkingToggle> {
        let key = self.thinking_key_name?;
        let messages = [serde_json::json!({"role": "user", "content": "probe"})];
        // One instant for the three renders: a template that writes the time
        // (`strftime_now`) must not tell them apart by the clock alone.
        let now = render_instant();
        let render = |thinking: Option<bool>| {
            let kwargs = thinking
                .map(|thinking| HashMap::from([(key.as_kwarg().to_string(), key.value(thinking))]));
            let params = ChatTemplateParams {
                add_generation_prompt: true,
                template_kwargs: kwargs.as_ref(),
                now: Some(now),
                ..Default::default()
            };
            self.apply(&messages, params).ok()
        };
        let (absent, on, off) = (render(None)?, render(Some(true))?, render(Some(false))?);
        match (absent == on, absent == off) {
            (true, false) => Some(ThinkingToggle::DefaultOn),
            (false, true) => Some(ThinkingToggle::DefaultOff),
            _ => None,
        }
    }

    /// The answers read from the template's behaviour once its environment
    /// is in place: how it writes a history tool call's arguments, and the
    /// thinking switch's default where its renders say it.
    fn settle_by_behaviour(&mut self) {
        self.raw_tool_call_arguments = self.writes_tool_call_arguments_as_text();
        if let Some(default) = self.rendered_thinking_default() {
            self.thinking_toggle = default;
        }
    }
}

#[cfg(test)]
mod tests;
