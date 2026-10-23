//! The recorded output slices the crate's bench rows and fuzz seeds stand on: one output per
//! table and family (plain text, one call, parallel calls, a thought and a call, a malformed
//! call), taken from bellwether's hand-written parse sets at bench and fuzz time.
//!
//! bellwether (smg-project/bellwether) records, for each checkpoint group, what the group's own
//! template renders for the hand-written cases every group records, the sets `common` and
//! `turns`, as `<slug>/parse/<set>.jsonl` of the tree `bellwether unpack` writes; CI pins the
//! bellwether commit in `.github/versions/bellwether.ref`. `BELLWETHER_FIXTURES` names that tree.
//! Without it, or for a group the tree has not recorded, the callers skip the rows and the seeds
//! and say so; nothing here is read from the repository. Each table that stands for a call syntax
//! names its group ([`GROUPS`]) and the prompt the group's template leaves before the output: the
//! turn's opener and, for a template with a thought, the thought open or closed as the case's
//! request asks, so the output is read where the model wrote it. The slice of a family is the
//! longest recorded output whose message has the family's shape, the first in the sets' order
//! among equals: content alone, with none of the table's markers in it (a probe that fences a
//! call in its text is not plain text); one call with arguments and nothing else; two or more
//! calls and nothing else; a thought and a call. The malformed slice is derived from the one call's: the
//! fewest trailing characters of the call's body before the terminal that closes the call (each
//! group names its table's, [`Group::call_close`]) are cut that make the table report the call
//! malformed, which the table itself is asked: one character for most (the `}` of a JSON object,
//! the `>` of a tag, the `]` of a list), more where the table forgives a cut at the close (Gemma 4
//! closes an open object at the call's end, so the string before it is cut open). A message's or
//! a list's close after the call's, and the separators a table ignores, are not the call's close.
//!
//! Included by path from `benches/throughput.rs` and from the fuzz crate: one definition for both.

use std::{
    fs,
    path::{Path, PathBuf},
};

use openai_protocol::common::{Function as ToolFunction, Tool};
use serde::Deserialize;
use serde_json::Value;
use symphony::{
    formats::Table, Declared, Engine, EngineFinish, Event, Events, Input, MalformedReason, Parser,
};

/// The variable that names the tree `bellwether unpack` writes.
pub const ENV: &str = "BELLWETHER_FIXTURES";

/// The families a table's slices come in, in the order the rows and seeds list them.
pub const FAMILIES: [&str; 5] = [
    "plain",
    "one_call",
    "parallel_calls",
    "thought_call",
    "malformed",
];

/// The hand-written sets every group records, read in this order where the tree holds them.
pub const HAND_WRITTEN_SETS: [&str; 2] = ["common", "turns"];

/// A table that stands for a call syntax, the bellwether group whose recorded outputs it reads,
/// the prompt its template leaves (the turn's opener, then the bytes that open or close the
/// thought when the request leaves thinking on, or turns it off (`enable_thinking: false`); a
/// template with no switch, or no thought, writes the same for both), and the terminal that
/// closes one call in the table, before which the malformed slice is cut.
pub struct Group {
    pub table: &'static str,
    pub slug: &'static str,
    pub opener: &'static str,
    pub thinking_on: &'static str,
    pub thinking_off: &'static str,
    pub call_close: &'static str,
}

/// The eight tables and their groups: the groups bellwether recorded first, one per call syntax,
/// whose prompts the corpus harness replays (`tests/bellwether_parse_fixtures.rs`).
pub const GROUPS: [Group; 8] = [
    // Qwen3: the model writes its own `<think>`; thinking off closes an empty thought in the prompt.
    Group {
        table: "qwen3",
        slug: "qwen3-8b",
        opener: "<|im_start|>assistant\n",
        thinking_on: "",
        thinking_off: "<think>\n\n</think>\n\n",
        call_close: "</tool_call>",
    },
    // Qwen 3.5: the prompt opens the thought, or closes an empty one when thinking is off.
    Group {
        table: "qwen3_tagged",
        slug: "qwen3.5-27b",
        opener: "<|im_start|>assistant\n",
        thinking_on: "<think>\n",
        thinking_off: "<think>\n\n</think>\n\n",
        call_close: "</tool_call>",
    },
    // GLM 5.3: the prompt always opens the thought.
    Group {
        table: "glm",
        slug: "glm-5.3-flash",
        opener: "<|assistant|>",
        thinking_on: "<think>\n",
        thinking_off: "<think>\n",
        call_close: "</tool_call>",
    },
    // DeepSeek V4.1: the prompt opens the thought, or closes an empty one when thinking is off.
    Group {
        table: "deepseek_v4.1",
        slug: "deepseek-v4.1-flash",
        opener: "<｜Assistant｜>",
        thinking_on: "<think>\n",
        thinking_off: "<think>\n\n</think>\n\n",
        call_close: "</｜DSML｜ invoke>",
    },
    // Kimi K3: the turn of messages opens with the thought, whatever the request says.
    Group {
        table: "kimi_k3",
        slug: "kimi-k3",
        opener: "<|open|>message role=\"assistant\"<|sep|>",
        thinking_on: "<|open|>think<|sep|>\n",
        thinking_off: "<|open|>think<|sep|>\n",
        call_close: "<|close|>call<|sep|>",
    },
    // MiniMax M3: the model writes `<mm:think>` itself, or closes an empty thought at once.
    Group {
        table: "minimax_m3",
        slug: "minimax-m3",
        opener: "]~b]ai\n",
        thinking_on: "",
        thinking_off: "",
        call_close: "</invoke>",
    },
    // Gemma 4: the model writes its thought channel itself.
    Group {
        table: "gemma4",
        slug: "gemma-4-e4b-it",
        opener: "<|turn>model\n",
        thinking_on: "",
        thinking_off: "",
        call_close: "<tool_call|>",
    },
    // LFM2.5: the model writes its own `<think>`.
    Group {
        table: "lfm2.5",
        slug: "lfm2.5-1.2b-instruct",
        opener: "<|im_start|>assistant\n",
        thinking_on: "",
        thinking_off: "",
        call_close: "<|tool_call_end|>",
    },
];

/// One recorded output slice of a table family: the prompt's tail that places it and the output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub family: String,
    pub prompt: String,
    pub output: String,
}

/// What the tree gave: the slices found, by table, and one notice per group the tree has not
/// recorded, which the caller prints and goes on without.
pub struct Recorded {
    pub slices: Vec<(Table, Slice)>,
    pub missing: Vec<String>,
}

/// The tree [`ENV`] names, or None when the variable is not set.
pub fn root() -> Option<PathBuf> {
    std::env::var_os(ENV).map(PathBuf::from)
}

/// The slices of every group recorded under `root`, in the order of [`GROUPS`] and [`FAMILIES`].
/// A group whose parse sets are not under `root` is a notice, not an error; a set that is there
/// and cannot be read is an error naming the file.
pub fn recorded(root: &Path) -> Result<Recorded, String> {
    let mut slices = Vec::new();
    let mut missing = Vec::new();
    for group in &GROUPS {
        let table = Table::named(group.table)
            .ok_or_else(|| format!("no table is named {}", group.table))?;
        let dir = root.join(group.slug).join("parse");
        if !dir.is_dir() {
            missing.push(format!(
                "{}: bellwether has not recorded {} under {}",
                group.table,
                group.slug,
                root.display()
            ));
            continue;
        }
        let mut cases = Vec::new();
        for set in HAND_WRITTEN_SETS {
            let path = dir.join(format!("{set}.jsonl"));
            if path.is_file() {
                cases.extend(read_cases(&path)?);
            }
        }
        if cases.is_empty() {
            missing.push(format!(
                "{}: {} has no hand-written parse set under {}",
                group.table,
                group.slug,
                root.display()
            ));
            continue;
        }
        let terminals: Vec<String> = table.format().terminal_texts().map(str::to_owned).collect();
        slices.extend(
            select(group, table, &cases, &terminals)
                .into_iter()
                .map(|slice| (table, slice)),
        );
    }
    Ok(Recorded { slices, missing })
}

/// One parse case as bellwether records it, in the fields read here.
#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    request: Request,
    reference: Reference,
}

#[derive(Deserialize, Default)]
struct Request {
    #[serde(default)]
    chat_template_kwargs: Option<Kwargs>,
}

#[derive(Deserialize, Default)]
struct Kwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct Reference {
    /// The output, as the model writes it after the prompt.
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    message: Option<Message>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<Call>>,
}

#[derive(Deserialize)]
struct Call {
    function: Function,
}

#[derive(Deserialize)]
struct Function {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
}

impl Case {
    /// The names of the tools the message's calls name, in order, each once.
    fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for call in self
            .reference
            .message
            .as_ref()
            .and_then(|message| message.tool_calls.as_deref())
            .unwrap_or(&[])
        {
            if !names.contains(&call.function.name) {
                names.push(call.function.name.clone());
            }
        }
        names
    }

    /// Whether the request turns thinking off, which places the prompt's tail.
    fn thinking_off(&self) -> bool {
        self.request
            .chat_template_kwargs
            .as_ref()
            .is_some_and(|kwargs| kwargs.enable_thinking == Some(false))
    }
}

impl Message {
    /// The family the message's shape belongs to, or None for a shape no family names: plain
    /// text is content with none of the table's `terminals` in it.
    fn family(&self, terminals: &[String]) -> Option<&'static str> {
        let calls = self.tool_calls.as_deref().unwrap_or(&[]);
        let reasoning = self
            .reasoning_content
            .as_deref()
            .is_some_and(|text| !text.is_empty());
        let content = self.content.as_ref().and_then(Value::as_str);
        let plain = content.is_some_and(|text| {
            !text.is_empty()
                && !terminals
                    .iter()
                    .any(|terminal| text.contains(terminal.as_str()))
        });
        let content = content.is_some_and(|text| !text.is_empty());
        match (reasoning, content, calls.len()) {
            (false, true, 0) if plain => Some("plain"),
            (false, false, 1) if calls[0].has_arguments() => Some("one_call"),
            (false, false, 2..) => Some("parallel_calls"),
            (true, _, 1..) => Some("thought_call"),
            _ => None,
        }
    }
}

impl Call {
    /// Whether the call carries an argument: its arguments, the JSON string the parser returns or
    /// the object it decodes to, are not empty.
    fn has_arguments(&self) -> bool {
        match &self.function.arguments {
            Some(Value::String(text)) => {
                let text = text.trim();
                !(text.is_empty() || text == "{}")
            }
            Some(Value::Object(fields)) => !fields.is_empty(),
            Some(Value::Null) | None => false,
            Some(_) => true,
        }
    }
}

/// The cases of one set file, in its order.
fn read_cases(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(number, line)| {
            serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), number + 1))
        })
        .collect()
}

/// The group's slices from its cases: for each shaped family the longest recorded output of that
/// shape, the first in order among equals, placed after the prompt its request asks for; then the
/// malformed slice derived from the one call's.
fn select(group: &Group, table: Table, cases: &[Case], terminals: &[String]) -> Vec<Slice> {
    let mut picked: Vec<Slice> = Vec::new();
    let mut one_call_tools: Vec<String> = Vec::new();
    for family in FAMILIES.iter().filter(|family| **family != "malformed") {
        let longest = cases
            .iter()
            .filter_map(|case| {
                let message = case.reference.message.as_ref()?;
                let text = case.reference.text.as_deref()?;
                (message.family(terminals)? == *family).then_some((text, case))
            })
            // `max_by_key` keeps the last of equals; the first is wanted.
            .fold(None, |best: Option<(&str, &Case)>, candidate| match best {
                Some((text, _)) if text.len() >= candidate.0.len() => best,
                _ => Some(candidate),
            });
        if let Some((output, case)) = longest {
            let tail = if case.thinking_off() {
                group.thinking_off
            } else {
                group.thinking_on
            };
            if *family == "one_call" {
                one_call_tools = case.tool_names();
            }
            picked.push(Slice {
                family: (*family).to_string(),
                prompt: format!("{}{tail}", group.opener),
                output: output.to_string(),
            });
        }
    }
    if let Some(one_call) = picked.iter().find(|slice| slice.family == "one_call") {
        if let Some(output) = malformed(table, one_call, &one_call_tools, group.call_close) {
            picked.push(Slice {
                family: "malformed".to_string(),
                prompt: one_call.prompt.clone(),
                output,
            });
        }
    }
    picked
}

/// How many trailing characters of a call's body a cut may take before the search gives up: a
/// table that forgives that much at its close has no malformed slice from this call.
const CUT_AT_MOST: usize = 8;

/// The one call's output with the fewest trailing characters of its call's body cut that make
/// `table` report the call malformed: the characters that are not whitespace before the last
/// `call_close` in it, the terminal that closes the call (not a message's or a list's close after
/// it, nor a separator the table ignores), taken away one more at a time and the output replayed
/// through the table after each. None when the output holds no call close, nothing before it, or
/// no cut of [`CUT_AT_MOST`] characters that the table reports.
fn malformed(table: Table, one_call: &Slice, tools: &[String], call_close: &str) -> Option<String> {
    let output = one_call.output.as_str();
    let close = output.rfind(call_close)?;
    let body: Vec<(usize, char)> = output[..close]
        .char_indices()
        .rev()
        .filter(|(_, c)| !c.is_whitespace())
        .take(CUT_AT_MOST)
        .collect();
    let mut cut = output.to_string();
    for (at, c) in body {
        cut.replace_range(at..at + c.len_utf8(), "");
        if reads_malformed(table, &one_call.prompt, &cut, tools) {
            return Some(cut);
        }
    }
    None
}

/// Whether `table` reports a malformed call in `output` after `prompt`: the output fed whole and
/// ended with a stop, any `Malformed` event but one for a tool the request did not declare (the
/// slice's tools are declared by name, their parameters an open object, so a reader typed by the
/// declaration reads the arguments as written).
fn reads_malformed(table: Table, prompt: &str, output: &str, tools: &[String]) -> bool {
    let declared: Vec<Tool> = tools
        .iter()
        .map(|name| Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: name.clone(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
                extra: Default::default(),
            },
        })
        .collect();
    let mut engine = Engine::new(table.format(), Declared::of(&declared));
    let mut out = Events::new();
    let fed = engine
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            &mut out,
        )
        .is_ok()
        && engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .is_ok()
        && engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .is_ok();
    fed && out.drain().iter().any(|event| {
        matches!(event, Event::Malformed { why, .. } if *why != MalformedReason::UnknownTool)
    })
}
