//! Parity with bellwether's parse fixtures.
//!
//! bellwether (smg-project/bellwether) records, per model, what a model's output must parse to: the
//! output text, its token ids, chunk plans, and the reference assistant message from the round trip
//! through the checkpoint's template. This test replays every Qwen3-8B parse case through the
//! Qwen3 table ([`qwen3()`]), folds the events into the assistant message with
//! [`adapt::chat::message`], and compares it with the reference: whole, at every byte split, byte
//! by byte, and on thirty seeded byte plans. Every replay also has to conserve the output's bytes
//! across its events and agree with every other replay of the same case.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory of a bellwether
//! checkout; without it the test prints a skip notice and passes. A second parity test replays
//! every Qwen checkpoint bellwether has recorded ([`MODELS`]): the Qwen3 table with the JSON call
//! syntax, the same table with the tagged syntax (Qwen 3.5 and later, Qwen3-Coder), typed by each
//! case's request tools, and the Qwen2.5 table, each after the prompt tail its template leaves; it
//! skips the slugs bellwether has not recorded yet, and says so. When a preview run sets
//! `SYMPHONY_CORPUS_ALLOWANCES=1`, that test also allows three classes of difference the scale-run
//! sets of 2026-10-06 have ([`Allowance`]): a reference argument whose type contradicts the one the
//! tool declares, which the template writes the same way as the string; reasoning in a reference
//! whose template writes no thought; and calls in a reference whose template writes content or
//! calls, not both. Each is refused at record time by bellwether now, each allowed case is counted
//! and printed, and without the switch the classes fail the run, so they cannot hide anything.
//!
//! Two policy questions stand between the parser and bitwise parity, and the test declares them
//! rather than hides them. Bellwether #17: the template's separator bytes (the newline after
//! `<think>`, the two after `</think>`) are in the parser's content and reasoning and not in the
//! reference's; a case that matches once the newlines at both ends are trimmed is counted as
//! "separators only", and any other whitespace still counts as a difference. Two of
//! the bitwise cases, the parallel calls, owe their count to the adapter rather than the parser:
//! there the parser's content is nothing but the separator between the calls, and the adapter's
//! rule that whitespace-only content is absent folds the difference away before the comparison.
//! Bellwether #16: two probe cases put marker strings inside text, and the parser reads them as
//! markers like every marker parser; they are listed in [`KNOWN_DIFFERENCES`] with their reason and
//! with the call count and finish reason the parser gives, so the list allows that difference and
//! no other. The run fails on any other difference, when a listed case starts matching, and when a
//! listed case is no longer among the fixtures, so the list cannot rot. Each replay ends with the
//! engine finish the reference implies. A second test replays the fixtures' token-level chunk
//! plans, each delta carrying its tokens' pieces as bellwether records them in `output_pieces`, and
//! checks that every token is counted once, in the event that carries its first byte.

mod common;

use std::{collections::BTreeMap, fs, path::PathBuf};

use common::{bytes_of, chunkings, prompt, replay, replay_after};
use openai_protocol::common::Tool;
use serde::Deserialize;
use symphony::{
    adapt,
    formats::{deepseek_v4_1, hy4, iquest, ling, qwen2_5, qwen3, seed_oss},
    CallSyntax, Declared, DropReason, Engine, EngineFinish, Event, Events, Input, ParseError,
    Parser, TokenSpan,
};

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const SLUG: &str = "qwen3-8b";
/// bellwether's slugs for the checkpoints the tables read, in its manifests' spelling, each with
/// the table that reads it and how its template ends the generation prompt. The fixtures carry
/// the request and the output, not the rendered prompt, so the prompt's tail is stated here until
/// bellwether records it. A slug bellwether has not recorded is skipped with a notice; `qwen3-8b`
/// is the one set bellwether's main always holds, and has its own test.
const MODELS: &[(&str, Family, GenerationPrompt)] = &[
    // Qwen3: the model writes its own `<think>`; thinking off closes it in the prompt.
    (
        "qwen3-8b",
        Family::Qwen3,
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "qwen3-0.6b",
        Family::Qwen3,
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "qwen3-235b-a22b",
        Family::Qwen3,
        GenerationPrompt::ModelWritesTheThought,
    ),
    // The 2507 line, Qwen3-Next and Qwen3-VL split into instruct, with no thought, and thinking,
    // whose template always opens the thought in the prompt.
    (
        "qwen3-4b-instruct-2507",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-30b-a3b-instruct-2507",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-235b-a22b-instruct-2507",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-4b-thinking-2507",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-235b-a22b-thinking-2507",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-next-80b-a3b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-next-80b-a3b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-2b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-4b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-8b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-30b-a3b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-32b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-235b-a22b-instruct",
        Family::Qwen3,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-vl-2b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-4b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-30b-a3b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-32b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-235b-a22b-thinking",
        Family::Qwen3,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    // Qwen2.5: no thought at all.
    (
        "qwen2.5-7b-instruct-1m",
        Family::Qwen2_5,
        GenerationPrompt::Plain,
    ),
    (
        "qwen2.5-14b-instruct-1m",
        Family::Qwen2_5,
        GenerationPrompt::Plain,
    ),
    (
        "qwen2.5-vl-3b-instruct",
        Family::Qwen2_5,
        GenerationPrompt::Plain,
    ),
    (
        "qwen2.5-vl-7b-instruct",
        Family::Qwen2_5,
        GenerationPrompt::Plain,
    ),
    (
        "qwen2.5-vl-32b-instruct",
        Family::Qwen2_5,
        GenerationPrompt::Plain,
    ),
    ("qwen2.5-omni-3b", Family::Qwen2_5, GenerationPrompt::Plain),
    // Qwen 3.5 and later write calls as tags and open the thought in the prompt; Qwen3-Coder
    // writes tags and has no thought.
    // Qwen3.5-0.8B's template has no thought.
    ("qwen3.5-0.8b", Family::Qwen3Tagged, GenerationPrompt::Plain),
    (
        "qwen3.5-4b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.5-9b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.5-27b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.5-35b-a3b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.5-122b-a10b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.6-27b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.6-35b-a3b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3.8-27b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "qwen3-coder-30b-a3b-instruct",
        Family::Qwen3Tagged,
        GenerationPrompt::Plain,
    ),
    (
        "qwen3-coder-next",
        Family::Qwen3Tagged,
        GenerationPrompt::Plain,
    ),
    // DeepSeek V4.1 writes DSML and opens the thought in the prompt (`<think>`, no newline).
    (
        "deepseek-v4.1-flash",
        Family::DeepSeekV4_1,
        GenerationPrompt::OpensTheThought,
    ),
    // Other families that write Qwen's syntaxes, each as its own template spells the thought.
    (
        "webworld-32b",
        Family::Qwen3,
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "webworld-14b",
        Family::Qwen3,
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "k-exaone-236b-a23b",
        Family::Qwen3,
        GenerationPrompt::OpensTheThought,
    ),
    ("hermes-4-14b", Family::Qwen2_5, GenerationPrompt::Plain),
    ("granite-4.1-3b", Family::Qwen2_5, GenerationPrompt::Plain),
    ("ai21-jamba2-3b", Family::Qwen2_5, GenerationPrompt::Plain),
    (
        "step-3.5-flash",
        Family::Qwen3Tagged,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "nanbeige4.2-3b",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "mimo-v2.5",
        Family::Qwen3Tagged,
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16",
        Family::Qwen3Tagged,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "seed-oss-36b-instruct",
        Family::SeedOss,
        GenerationPrompt::ModelWritesTheThought,
    ),
    // Keyed arguments, each family under its own markers; the prompt opens the thought, Hy4's
    // whatever the request says.
    (
        "hy4-preview",
        Family::Hy4,
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "ling-3.0-flash",
        Family::Ling,
        GenerationPrompt::OpensTheThought,
    ),
    (
        "iquest-q1",
        Family::IQuest,
        GenerationPrompt::OpensTheThought,
    ),
];

/// The turn opener a model's own chat template writes, for a model that reads another family's
/// table: the opener is the template's, not the table's, so the prompt's replay has to start at
/// the model's own. K-EXAONE writes Qwen3's markers under its own template, Granite 4.1 Qwen2.5's;
/// every other recorded model renders `<|im_start|>assistant` or its own table's opener. The
/// replay below feeds only the generation prompt's tail, so it does not check these spellings:
/// they were measured against the templates' rendered prompts (smg #2841, Alex's probes), and
/// `formats/qwen3.rs` holds the K-EXAONE shape as a test.
const OPENERS: &[(&str, &str)] = &[
    ("k-exaone-236b-a23b", "<|assistant|>"),
    (
        "granite-4.1-3b",
        "<|start_of_role|>assistant<|end_of_role|>",
    ),
];

/// The table that reads a checkpoint's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// [`qwen3`] with the JSON call syntax.
    Qwen3,
    /// [`qwen3`] with the tagged call syntax, typed by the case's request tools.
    Qwen3Tagged,
    /// [`qwen2_5`].
    Qwen2_5,
    /// [`deepseek_v4_1`]: DSML, whose parameter tags type their own values.
    DeepSeekV4_1,
    /// [`seed_oss`]: the tagged syntax under Seed-OSS's markers, typed by the request tools.
    SeedOss,
    /// [`hy4`], [`ling`], [`iquest`]: keyed arguments, typed by the request tools.
    Hy4,
    Ling,
    IQuest,
}

impl Family {
    /// How the family spells the thought's markers, for the prompt's tail and the reasoning
    /// allowance's guard.
    fn think_markers(self) -> (&'static str, &'static str) {
        match self {
            Self::SeedOss => ("<seed:think>", "</seed:think>"),
            Self::Hy4 => ("<think:opensource>", "</think:opensource>"),
            _ => ("<think>", "</think>"),
        }
    }

    /// The engine for one of the model's cases: the family's table, with the model's own turn
    /// opener when its template is not the table's ([`OPENERS`]), and the case's tools.
    fn engine(self, slug: &str, fixture: &Fixture) -> Engine {
        let mut format = self.format();
        if let Some((_, opener)) = OPENERS.iter().find(|(model, _)| *model == slug) {
            format = format.opens_turn(opener);
        }
        Engine::new(format, Declared::of(&fixture.request.tools))
    }

    fn format(self) -> symphony::Format {
        match self {
            Self::Qwen3 => qwen3(CallSyntax::Json),
            Self::Qwen3Tagged => qwen3(CallSyntax::Tagged),
            Self::Qwen2_5 => qwen2_5(),
            Self::DeepSeekV4_1 => deepseek_v4_1(),
            Self::SeedOss => seed_oss(),
            Self::Hy4 => hy4(),
            Self::Ling => ling(),
            Self::IQuest => iquest(),
        }
    }

    /// The probe cases known to differ under this syntax: the code fence holds the JSON syntax,
    /// which the JSON assembler reads as a call and the tagged one reports as text between a
    /// call's tags.
    fn known_differences(self, prompt: GenerationPrompt) -> Vec<KnownDifference> {
        let list = match self {
            Self::Qwen3 | Self::Qwen2_5 => KNOWN_DIFFERENCES,
            Self::Qwen3Tagged => KNOWN_TAGGED_DIFFERENCES,
            // Seed-OSS and Hy4 read neither of the probes' Qwen markers: `</think>` stays reasoning
            // text and the fenced `<tool_call>` block stays content, as the reference says.
            Self::SeedOss | Self::Hy4 => &[],
            // Ling reads `<tool_call>` and `</think>`, so its two probes are the tagged ones' (no
            // call comes of the fence); IQuest and DSML read `</think>` but not `<tool_call>`.
            Self::Ling => KNOWN_TAGGED_DIFFERENCES,
            Self::IQuest | Self::DeepSeekV4_1 => KNOWN_REASONING_PROBE,
        };
        // A template without a thought leaves the reasoning out, so the marker inside it is never
        // read; that case falls under the reasoning allowance instead of the list.
        list.iter()
            .copied()
            .filter(|known| prompt != GenerationPrompt::Plain || known.id != REASONING_PROBE.id)
            .collect()
    }

    /// The corpus classes this syntax meets in sets recorded before bellwether refused them,
    /// when the run opts in ([`CORPUS_ALLOWANCES_ENV`]): the tagged syntax cannot carry a type
    /// the text does not say, a template without a thought drops the reference's reasoning, and
    /// four Qwen2.5 templates write content or calls, not both.
    fn allowances(self, slug: &str, prompt: GenerationPrompt) -> Vec<Allowance> {
        if !corpus_allowances_on() {
            return Vec::new();
        }
        let mut allowed = Vec::new();
        if matches!(
            self,
            Self::Qwen3Tagged | Self::SeedOss | Self::Hy4 | Self::Ling | Self::IQuest
        ) {
            allowed.push(Allowance::DeclaredTypeConflict);
        }
        if prompt == GenerationPrompt::Plain {
            allowed.push(Allowance::ReasoningNotWritten);
        }
        if CONTENT_OR_CALLS.contains(&slug) {
            allowed.push(Allowance::CallsNotWritten);
        }
        allowed
    }
}

/// How a template ends the generation prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationPrompt {
    /// Qwen 3.5 and later: `<think>\n`, so the output starts inside the thought; when the request
    /// turns thinking off, `<think>\n\n</think>\n\n`, and the output starts in content.
    OpensTheThought,
    /// The Qwen3 thinking variants (the 2507 line, Qwen3-Next, Qwen3-VL): `<think>\n` whatever
    /// the request says; their templates have no switch, so the output always starts inside the
    /// thought.
    AlwaysOpensTheThought,
    /// Qwen3: nothing, and the model writes its own `<think>`; thinking off closes it in the
    /// prompt as above.
    ModelWritesTheThought,
    /// Qwen3-Coder, Qwen2.5 and the instruct variants: `<|im_start|>assistant\n` and nothing more;
    /// the template has no thought.
    Plain,
}

impl GenerationPrompt {
    /// The bytes after the assistant header for the case's request, in the family's spelling of
    /// the thought's markers.
    fn tail(self, fixture: &Fixture, family: Family) -> String {
        let (open, close) = family.think_markers();
        match (self, fixture.request.thinking_off()) {
            (Self::Plain, _) | (Self::ModelWritesTheThought, false) => String::new(),
            (Self::OpensTheThought | Self::ModelWritesTheThought, true) => {
                format!("{open}\n\n{close}\n\n")
            }
            (Self::OpensTheThought, false) | (Self::AlwaysOpensTheThought, _) => {
                format!("{open}\n")
            }
        }
    }
}

/// The variable a preview run sets to allow the corpus classes below; CI, on sets bellwether
/// records now, leaves it unset, so a class that comes back fails the run.
const CORPUS_ALLOWANCES_ENV: &str = "SYMPHONY_CORPUS_ALLOWANCES";

/// Whether the run opted in: the variable is set to anything but `0` or nothing.
fn corpus_allowances_on() -> bool {
    std::env::var(CORPUS_ALLOWANCES_ENV).is_ok_and(|value| !value.is_empty() && value != "0")
}

/// The slugs whose template writes a message's content or its calls, not both.
const CONTENT_OR_CALLS: &[&str] = &[
    "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-omni-3b",
];

/// A class of difference the scale-run sets of 2026-10-06 have, allowed for what the fixture
/// shows rather than by id, and only when the run opts in. Each is a case bellwether refuses at
/// record time now; the sets that hold them were recorded before the rule that refuses them, and
/// the classes go once those sets are recorded again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allowance {
    /// The reference's argument is a boolean, a number or null for a parameter the tool declares
    /// `string`, so the template writes it as it writes the string, and the parser gives the
    /// string back, as vLLM does (BFCL declares `smoking_allowed` an enum of `"True"`, `"False"`
    /// and `"dontcare"` and answers `false`). Bellwether #56 refuses such a case.
    DeclaredTypeConflict,
    /// The template writes no thought, so the reasoning the reference carries is not in the output
    /// (bellwether #52, refused by bellwether #68). Allowed only when the output holds neither of
    /// the family's thought markers, so a thought the parser lost never passes as one the
    /// template dropped.
    ReasoningNotWritten,
    /// The template writes a message's content or its calls, not both, so the calls the reference
    /// carries are not in the output: Qwen2.5-VL and Qwen2.5-Omni (bellwether #33's class, refused
    /// since it merged). Allowed only when the output holds no call marker at all, so a call the
    /// parser missed never passes as one the template dropped.
    CallsNotWritten,
}

/// A case known to differ from the reference beyond the separator bytes: why, and what the parser
/// says instead, its call count and finish reason, so that the list allows that difference and no
/// other. `id` is the case's id after the slug, so one list serves every slug it names.
#[derive(Clone, Copy)]
struct KnownDifference {
    id: &'static str,
    reason: &'static str,
    calls: usize,
    finish: &'static str,
}

const KNOWN_DIFFERENCES: &[KnownDifference] = &[
    REASONING_PROBE,
    KnownDifference {
        id: "parse/content-with-marker-in-code-fence",
        reason: "the content holds a complete `<tool_call>` block inside a code fence; the parser makes \
                 a call of it, as every marker parser does, and the reference keeps it as content \
                 (bellwether #16)",
        calls: 1,
        finish: "tool_calls",
    },
];

/// The reasoning probe: its reasoning holds a `</think>`, which every table with that marker
/// reads as the thought's end.
const REASONING_PROBE: KnownDifference = KnownDifference {
    id: "parse/reasoning-with-marker-text",
    reason: "the reasoning holds a `</think>`; the parser ends the reasoning there, as every \
             marker parser does, and the reference keeps the marker as reasoning text \
             (bellwether #16)",
    calls: 0,
    finish: "stop",
};

/// The code-fence probe: its content holds a complete `<tool_call>` block inside a code fence,
/// which every table with that marker reads as a block.
const FENCE_PROBE: KnownDifference = KnownDifference {
    id: "parse/content-with-marker-in-code-fence",
    reason: "the content holds a complete `<tool_call>` block inside a code fence; the parser \
             reads the block, as every marker parser does, and the reference keeps it as \
             content (bellwether #16)",
    calls: 0,
    finish: "stop",
};

/// Under the tagged syntax both probes differ.
const KNOWN_TAGGED_DIFFERENCES: &[KnownDifference] = &[REASONING_PROBE, FENCE_PROBE];

/// Under a table that reads `</think>` but not `<tool_call>` (DSML, IQuest), only the reasoning
/// probe differs: the fence holds Qwen's syntax, which the table never reads as a call, so it is
/// content, as the reference says.
const KNOWN_REASONING_PROBE: &[KnownDifference] = &[REASONING_PROBE];

/// The case's id after its slug: what [`KnownDifference::id`] names.
fn after_slug(id: &str) -> &str {
    id.split_once('/').map_or(id, |(_, rest)| rest)
}

#[derive(Deserialize)]
struct Fixture {
    id: String,
    #[serde(default)]
    request: Request,
    reference: Reference,
    #[serde(default)]
    output_ids: Vec<u32>,
    /// The text each output token contributes under the reference tokenizer's incremental decode;
    /// absent before bellwether 95f079b.
    output_pieces: Option<Vec<String>>,
    /// Chunk sizes in tokens by plan name; `whole` and `per_token` carry none and are derived.
    #[serde(default)]
    chunk_plans: BTreeMap<String, Option<Vec<usize>>>,
}

/// The parts of the case's request the replay needs: the tools, which type a tagged call's values,
/// and whether the request turned thinking off, which decides the prompt's tail.
#[derive(Default, Deserialize)]
struct Request {
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

impl Request {
    fn thinking_off(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.enable_thinking)
            == Some(false)
    }
}

#[derive(Deserialize)]
struct ChatTemplateKwargs {
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct Reference {
    text: String,
    message: ReferenceMessage,
    finish_reason: String,
}

#[derive(Deserialize)]
struct ReferenceMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ReferenceCall>,
}

#[derive(Deserialize)]
struct ReferenceCall {
    function: ReferenceFunction,
}

#[derive(Deserialize)]
struct ReferenceFunction {
    name: String,
    arguments: String,
}

/// What the parser said an output means, in the terms the reference uses.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Said {
    content: Option<String>,
    reasoning: Option<String>,
    calls: Vec<(String, String)>,
    finish: String,
}

impl Said {
    fn of(events: &[Event]) -> Self {
        let choice = adapt::chat::message(0, events);
        Self {
            content: choice.message.content,
            reasoning: choice.message.reasoning_content,
            calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    (
                        call.function.name,
                        call.function.arguments.unwrap_or_default(),
                    )
                })
                .collect(),
            finish: choice.finish_reason.unwrap_or_default(),
        }
    }

    fn of_reference(reference: &Reference) -> Self {
        Self {
            content: reference.message.content.clone().filter(|c| !c.is_empty()),
            reasoning: reference.message.reasoning_content.clone(),
            calls: reference
                .message
                .tool_calls
                .iter()
                .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
                .collect(),
            finish: reference.finish_reason.clone(),
        }
    }

    /// The same, with the separator bytes trimmed from content and reasoning (bellwether #17). The
    /// template's separators are newlines, so only newlines are trimmed: any other whitespace the
    /// parser kept or lost still counts as a difference.
    fn trimmed(&self) -> Self {
        let trim = |part: &Option<String>| {
            part.as_deref()
                .map(|text| text.trim_matches('\n'))
                .filter(|t| !t.is_empty())
                .map(str::to_string)
        };
        Self {
            content: trim(&self.content),
            reasoning: trim(&self.reasoning),
            calls: self.calls.clone(),
            finish: self.finish.clone(),
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn qwen3_parse_fixtures_match_the_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let fixtures = read_fixtures(&root.join(SLUG).join("parse")).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !fixtures.is_empty(),
        "no parse fixtures under {}",
        root.display()
    );
    let failures = parity(
        &fixtures,
        &|fixture| Family::Qwen3.engine(SLUG, fixture),
        &|_| String::new(),
        KNOWN_DIFFERENCES,
        &[],
        Family::Qwen3.think_markers(),
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn every_recorded_qwen_model_parses_like_its_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let mut failures = Vec::new();
    let mut recorded = 0;
    for &(slug, family, prompt) in MODELS {
        let dir = root.join(slug).join("parse");
        if !dir.is_dir() {
            eprintln!("skipping {slug}: bellwether has not recorded its parse sets yet");
            continue;
        }
        let fixtures = read_fixtures(&dir).unwrap_or_else(|e| panic!("{e}"));
        println!("{slug} ({family:?}, {prompt:?}):");
        failures.extend(parity(
            &fixtures,
            &|fixture| family.engine(slug, fixture),
            &|fixture| prompt.tail(fixture, family),
            &family.known_differences(prompt),
            &family.allowances(slug, prompt),
            family.think_markers(),
        ));
        recorded += 1;
    }
    if recorded == 0 {
        eprintln!(
            "skipping: none of the table's slugs is recorded under {}",
            root.display()
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Replays every fixture through a fresh parser from `new_parser`, after a prompt ending in the
/// case's `prompt_tail`, on every chunking, prints one line per case, and returns every difference
/// that neither `known_differences` nor `allowed` allows.
#[expect(
    clippy::print_stdout,
    clippy::panic,
    reason = "the per-case report is diagnostic output; a fixture that cannot be replayed ends \
              the test with its id"
)]
fn parity(
    fixtures: &[Fixture],
    new_parser: &dyn Fn(&Fixture) -> Engine,
    prompt_tail: &dyn Fn(&Fixture) -> String,
    known_differences: &[KnownDifference],
    allowed: &[Allowance],
    think_markers: (&str, &str),
) -> Vec<String> {
    let known = |id: &str| {
        known_differences
            .iter()
            .find(|known| known.id == after_slug(id))
    };
    let mut failures = Vec::new();
    let (mut bitwise, mut separators_only, mut listed, mut allowed_count) = (0, 0, 0, 0);
    for fixture in fixtures {
        let text = fixture.reference.text.as_str();
        let expected = Said::of_reference(&fixture.reference);
        let mut saids = Vec::new();
        let finish = engine_finish(&fixture.reference.finish_reason);
        for plan in chunkings(text) {
            let events = replay_after(
                &mut new_parser(fixture),
                &prompt_tail(fixture),
                text,
                &plan,
                &finish,
                false,
            )
            .unwrap_or_else(|e| panic!("{}: plan {plan:?}: {e}", fixture.id));
            let conserved: String = events.iter().map(bytes_of).collect();
            if conserved != text {
                failures.push(format!(
                    "{}: bytes not conserved on plan {plan:?}",
                    fixture.id
                ));
            }
            let mut index = 0;
            for event in &events {
                if let Event::ToolCallStart { index: i, id, .. } = event {
                    if *i != index || *id != format!("call_{index}") {
                        failures.push(format!(
                            "{}: call index {i} / id {id} on plan {plan:?}",
                            fixture.id
                        ));
                    }
                    index += 1;
                }
            }
            saids.push(Said::of(&events));
        }
        let said = saids[0].clone();
        if saids.iter().any(|s| *s != said) {
            failures.push(format!(
                "{}: the replays disagree with each other",
                fixture.id
            ));
        }
        let verdict = if said == expected {
            bitwise += 1;
            "bitwise"
        } else if said.trimmed() == expected.trimmed() {
            separators_only += 1;
            "separators only (bellwether #17)"
        } else if let Some(listed_case) = known(&fixture.id) {
            if said.calls.len() != listed_case.calls || said.finish != listed_case.finish {
                failures.push(format!(
                    "{}: listed in KNOWN_DIFFERENCES for {} call(s) and `{}`, but the parser says {} \
                     call(s) and `{}`",
                    fixture.id,
                    listed_case.calls,
                    listed_case.finish,
                    said.calls.len(),
                    said.finish
                ));
            }
            listed += 1;
            "listed (bellwether #16)"
        } else if let Some(allowance) =
            allowance_for(&said, &expected, text, allowed, think_markers)
        {
            allowed_count += 1;
            match allowance {
                Allowance::DeclaredTypeConflict => "allowed: declared-type conflict (corpus)",
                Allowance::ReasoningNotWritten => "allowed: reasoning not written (corpus)",
                Allowance::CallsNotWritten => "allowed: calls not written (corpus)",
            }
        } else {
            failures.push(format!(
                "{}: differs from the reference\n    said:      {said:?}\n    reference: {expected:?}",
                fixture.id
            ));
            "DIFFERS"
        };
        if known(&fixture.id).is_some() && said.trimmed() == expected.trimmed() {
            failures.push(format!(
                "{}: listed in KNOWN_DIFFERENCES but matching the reference now; remove it",
                fixture.id
            ));
        }
        println!("  {verdict:34} {}", fixture.id);
        if let Some(listed_case) = known(&fixture.id) {
            println!("  {:34} {}", "", listed_case.reason);
        }
    }
    for listed_case in known_differences {
        if !fixtures
            .iter()
            .any(|fixture| after_slug(&fixture.id) == listed_case.id)
        {
            failures.push(format!(
                "{}: listed in KNOWN_DIFFERENCES but not among the fixtures; remove it",
                listed_case.id
            ));
        }
    }
    println!(
        "{} cases: {bitwise} bitwise, {separators_only} separators only, {listed} listed, \
         {allowed_count} allowed for the corpus",
        fixtures.len()
    );
    failures
}

/// The allowance that covers the difference between `said` and `expected`, if one of `allowed`
/// does: the two agree once the separators are trimmed, except as the allowance says. `text` is
/// the output, for the allowance that asks what the template wrote.
fn allowance_for(
    said: &Said,
    expected: &Said,
    text: &str,
    allowed: &[Allowance],
    think_markers: (&str, &str),
) -> Option<Allowance> {
    let (said, expected) = (said.trimmed(), expected.trimmed());
    allowed.iter().copied().find(|allowance| match allowance {
        Allowance::ReasoningNotWritten => {
            said.reasoning.is_none()
                && expected.reasoning.is_some()
                && !text.contains(think_markers.0)
                && !text.contains(think_markers.1)
                && Said {
                    reasoning: None,
                    ..expected.clone()
                } == said
        }
        Allowance::CallsNotWritten => {
            said.calls.is_empty()
                && !expected.calls.is_empty()
                && !text.contains("<tool_call>")
                && Said {
                    calls: Vec::new(),
                    finish: "stop".to_string(),
                    ..expected.clone()
                } == said
        }
        Allowance::DeclaredTypeConflict => {
            said.content == expected.content
                && said.reasoning == expected.reasoning
                && said.finish == expected.finish
                && said.calls.len() == expected.calls.len()
                && said.calls.iter().zip(&expected.calls).all(
                    |((name, arguments), (expected_name, expected_arguments))| {
                        name == expected_name
                            && differs_at_most_in_declared_type(arguments, expected_arguments)
                    },
                )
        }
    })
}

/// Whether two arguments objects differ at most in members where `said` holds a string spelling
/// the value `expected` holds, in JSON or as Python writes it. The caller knows the two messages
/// differ somewhere; a call of theirs that is the same on both sides passes.
fn differs_at_most_in_declared_type(said: &str, expected: &str) -> bool {
    let (Ok(said), Ok(expected)) = (
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(said),
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(expected),
    ) else {
        return false;
    };
    if said.len() != expected.len() || said.keys().ne(expected.keys()) {
        return false;
    }
    for (key, value) in &said {
        let reference = &expected[key];
        if value == reference {
            continue;
        }
        let spelled_the_same = match (value, reference) {
            (serde_json::Value::String(text), reference) if !reference.is_string() => {
                [reference.to_string(), python_spelling(reference)].contains(text)
            }
            _ => false,
        };
        if !spelled_the_same {
            return false;
        }
    }
    true
}

/// How Qwen 3.5's and Qwen3-Coder's templates write a boolean or null: Python's spelling.
fn python_spelling(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

/// The engine's finish for a reference's finish reason. The adapter turns `stop` after a call into
/// `tool_calls`, so only a truncation needs its own engine reason.
fn engine_finish(reference: &str) -> EngineFinish {
    match reference {
        "length" => EngineFinish::Length,
        _ => EngineFinish::Stop,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the summary are test diagnostic output"
)]
fn qwen3_token_plans_count_every_token_where_its_first_byte_lands() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let fixtures = read_fixtures(&root.join(SLUG).join("parse")).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !fixtures.is_empty(),
        "no parse fixtures under {}",
        root.display()
    );
    let (mut plans_run, mut tokens_counted, mut reasoning_counted) = (0, 0, 0);
    for fixture in &fixtures {
        let pieces = fixture.output_pieces.as_deref().unwrap_or_else(|| {
            panic!(
                "{}: no output_pieces; these fixtures predate bellwether 95f079b",
                fixture.id
            )
        });
        let text = fixture.reference.text.as_str();
        assert_eq!(
            pieces.len(),
            fixture.output_ids.len(),
            "{}: one piece per id",
            fixture.id
        );
        assert_eq!(
            pieces.concat(),
            text,
            "{}: the pieces give back the text",
            fixture.id
        );
        let starts = token_starts(pieces);
        let expected_reasoning = reasoning_oracle(text, &starts);
        let finish = engine_finish(&fixture.reference.finish_reason);
        let by_text = Said::of(
            &replay(
                &mut Engine::new(qwen3(CallSyntax::Json), Declared::default()),
                text,
                &[],
                &finish,
                false,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.id)),
        );
        for (name, sizes) in token_plans(fixture) {
            let place = format!("{} plan {name}", fixture.id);
            let events = replay_tokens(&fixture.output_ids, pieces, &sizes, &finish)
                .unwrap_or_else(|e| panic!("{place}: {e}"));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, text, "{place}: bytes conserved");
            let mut at = 0;
            let mut total = 0;
            for event in &events {
                if matches!(
                    event,
                    Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. }
                ) {
                    continue;
                }
                let tokens = tokens_of(event)
                    .unwrap_or_else(|| panic!("{place}: uncounted {event:?}"))
                    as usize;
                let length = bytes_of(event).len();
                let expected = match event {
                    // Tokens left without bytes at the end are reported once, after the text.
                    Event::Dropped {
                        why: DropReason::ControlToken,
                        ..
                    } if length == 0 => starts.iter().filter(|&&start| start >= text.len()).count(),
                    _ => starts
                        .iter()
                        .filter(|&&start| start >= at && start < at + length)
                        .count(),
                };
                assert_eq!(tokens, expected, "{place}: {event:?} at byte {at}");
                at += length;
                total += tokens;
            }
            assert_eq!(
                total,
                fixture.output_ids.len(),
                "{place}: every token counted once"
            );
            let Some(Event::Finish {
                reasoning_tokens, ..
            }) = events.last()
            else {
                panic!("{place}: Finish is last");
            };
            assert_eq!(
                *reasoning_tokens as usize, expected_reasoning,
                "{place}: the reasoning tokens"
            );
            assert_eq!(
                Said::of(&events),
                by_text,
                "{place}: says what the text replay says"
            );
            plans_run += 1;
            tokens_counted += total;
            reasoning_counted += expected_reasoning;
        }
    }
    println!(
        "{} cases, {plans_run} token plans, {tokens_counted} tokens counted, \
         {reasoning_counted} of them reasoning",
        fixtures.len()
    );
}

/// Every token plan the fixture records, as chunk sizes in tokens: `whole` and `per_token` derived,
/// the others as recorded.
fn token_plans(fixture: &Fixture) -> Vec<(String, Vec<usize>)> {
    let count = fixture.output_ids.len();
    fixture
        .chunk_plans
        .iter()
        .map(|(name, sizes)| {
            let sizes = match (name.as_str(), sizes) {
                (_, Some(sizes)) => sizes.clone(),
                ("per_token", None) => vec![1; count],
                (_, None) => vec![count],
            };
            (name.clone(), sizes)
        })
        .collect()
}

/// Where each token's text starts in the output. A token whose piece is empty starts where the
/// next byte is, so it counts with the text that follows it.
fn token_starts(pieces: &[String]) -> Vec<usize> {
    let mut at = 0;
    pieces
        .iter()
        .map(|piece| {
            let start = at;
            at += piece.len();
            start
        })
        .collect()
}

/// How many tokens start inside the first reasoning region of `text`: after the first `<think>`
/// and before the `</think>` that follows it, or the end. Found by searching the text, not by the
/// parser, so it checks the parser's count.
fn reasoning_oracle(text: &str, starts: &[usize]) -> usize {
    let Some(open) = text.find("<think>") else {
        return 0;
    };
    let from = open + "<think>".len();
    let to = text[from..]
        .find("</think>")
        .map_or(text.len(), |at| from + at);
    starts
        .iter()
        .filter(|&&start| start >= from && start < to)
        .count()
}

/// Feed the output token by token as the plan `sizes` groups them, each delta carrying its tokens'
/// ids and the span of each token's piece, after the prompt, then the end with the engine's
/// `finish`.
fn replay_tokens(
    ids: &[u32],
    pieces: &[String],
    sizes: &[usize],
    finish: &EngineFinish,
) -> Result<Vec<Event>, ParseError> {
    let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
    let mut out = Events::new();
    parser.feed(prompt(), &mut out)?;
    let mut first = 0;
    for &size in sizes {
        let last = (first + size).min(ids.len());
        let mut text = String::new();
        let mut spans = Vec::with_capacity(last - first);
        for (offset, piece) in pieces[first..last].iter().enumerate() {
            spans.push(TokenSpan {
                token_id: ids[first + offset],
                start: text.len(),
                end: text.len() + piece.len(),
                continued: false,
            });
            text.push_str(piece);
        }
        parser.feed(
            Input::Delta {
                token_ids: &ids[first..last],
                text: &text,
                spans: &spans,
            },
            &mut out,
        )?;
        first = last;
    }
    parser.feed(
        Input::End {
            finish: finish.clone(),
        },
        &mut out,
    )?;
    Ok(out.drain())
}

fn tokens_of(event: &Event) -> Option<u32> {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.tokens,
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.tokens,
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.tokens,
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
    }
}

fn read_fixtures(dir: &std::path::Path) -> Result<Vec<Fixture>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    let mut fixtures = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture: Fixture = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}
