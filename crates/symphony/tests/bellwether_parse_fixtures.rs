//! Parity with bellwether's parse fixtures.
//!
//! bellwether (smg-project/bellwether) records, per model, what a model's output must parse to: the
//! output text, its token ids, chunk plans, and the reference assistant message from the round trip
//! through the checkpoint's template. This test replays every Qwen3-8B parse case through the
//! Qwen3 table ([`Table::Qwen3`]), folds the events into the assistant message with
//! [`adapt::chat::message`], and compares it with the reference: whole, character by character,
//! on thirty seeded byte plans, and on the token-level chunk plans bellwether recorded with the
//! case, each delta carrying its tokens' ids and pieces. Every replay also has to conserve the
//! output's bytes across its events and agree with the whole one, and a streamed replay's content
//! deltas have to carry the whole answer's content and nothing else: the template's spacing
//! before a call, shown as content before the call arrived, is what a client of the stream would
//! see and a client of the whole answer would not. An output of at most 256 bytes
//! also replays at every two-way split; past that the set is linear in the output's length, so
//! a case of any size replays forty-odd times.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory of a bellwether
//! checkout; without it the test prints a skip notice and passes. `SYMPHONY_SAMPLE_EVERY=n`
//! replays every n-th case of each recorded set, the same slice every run, for a CI job that
//! cannot afford a benchmark-scale set whole, and the hand-written sets (`common`, `turns`)
//! whole; a run that leaves it unset or empty replays every case. A second parity test replays
//! every Qwen checkpoint bellwether has recorded ([`MODELS`]): the Qwen3 table with the JSON call
//! syntax, the same table with the tagged syntax (Qwen 3.5 and later, Qwen3-Coder), typed by each
//! case's request tools, and the Qwen2.5 table, each after the prompt tail its template leaves,
//! on as many threads as the run has CPUs, each model's report written in the table's order as
//! soon as it and the models before it are done; it skips the slugs bellwether has not recorded
//! yet, and says so. When a preview run sets
//! `SYMPHONY_CORPUS_ALLOWANCES=1`, that test also allows the classes of difference the recorded
//! sets have ([`Allowance`]): a reference argument whose type contradicts the one the tool
//! declares, which the template writes the same way as the string; reasoning in a reference whose
//! template writes no thought; calls, or content, in a reference whose template writes content or
//! calls, not both; a null the template writes no element for; and whitespace at the content's
//! edges that the template trims. Each is refused at record time by bellwether now, or will be
//! once the issue named on it lands; each allowed case is counted and printed, and without the
//! switch the classes fail the run, so they cannot hide anything.
//!
//! Two policy questions stand between the parser and bitwise parity, and the test declares them
//! rather than hides them. Bellwether #17: the template's separator bytes (the newline after
//! `<think>`, the two after `</think>`) are in the parser's content and reasoning and not in the
//! reference's; a case that matches once the newlines at both ends are trimmed is counted as
//! "separators only", and any other whitespace still counts as a difference. Two of
//! the bitwise cases, the parallel calls, owe their count to the adapter rather than the parser:
//! there the parser's content is nothing but the separator between the calls, and the adapter's
//! rule that whitespace-only content is absent folds the difference away before the comparison.
//! A template that re-serializes the arguments (Inkling's `tojson(sort_keys=true,
//! separators=(",", ":"))`) has the model write the keys sorted and the spaces dropped, and the
//! parser hands on the model's bytes; a case whose arguments differ from the reference's only so,
//! equal as values, is counted as "respelled", and only for a family whose template does that.
//! Bellwether #16: two probe cases put marker strings inside text, and the parser reads them as
//! markers like every marker parser; they are listed in [`KNOWN_DIFFERENCES`] with their reason and
//! with the call count and finish reason the parser gives, so the list allows that difference and
//! no other. The run fails on any other difference, when a listed case starts matching, and when a
//! listed case is no longer among the fixtures, so the list cannot rot. Each replay ends with the
//! engine finish the reference implies. A second test replays the fixtures' token-level chunk
//! plans, each delta carrying its tokens' pieces as bellwether records them in `output_pieces`, and
//! checks that every token is counted once, in the event that carries its first byte.
//!
//! A fourth test corrupts the recorded outputs: three seeded corruptions of every sampled case of
//! every recorded model (a span deleted or doubled, one of the table's own markers or a cut one
//! inserted, the tail cut, a character swapped), each read through the model's table after its
//! prompt tail, whole, on two seeded plans and, up to 512 bytes, a character at a time. Nothing
//! may panic, every byte must land in exactly one event with `Finish` last, and the message the
//! chat adapter folds from the events must be the same however the output was cut; a panic is
//! caught per output, so one run reports every breach with the case, the round and the corrupted
//! output that replay it. The sample switch applies as above.
//!
//! What a replay prints, per model: the cases that differ from the reference and nothing allows
//! (`DIFFERS`, up to [`SHOWN`] of them) and the summary line of its sets. The per-case report, one
//! verdict line per case with the reason of a listed one, followed by the failures, goes to a file
//! per test under `BELLWETHER_REPORT_DIR`, or under the build's `target/tmp` when the variable is
//! not set, and the run says where. `BELLWETHER_VERBOSE=1` prints the per-case report on stdout
//! too, as the run did before it went to a file.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one keeps an id by its number, not its shape"
)]
mod common;

use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use common::{bytes_of, chunkings, prompt, replay, replay_after};
use openai_protocol::common::Tool;
use serde::Deserialize;
use symphony::{
    adapt,
    formats::{for_model, groups, table_for, Table},
    Declared, DropReason, Engine, EngineFinish, Event, Events, Input, ParseError, Parser,
    TokenSpan,
};

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
/// Replay every n-th case of each recorded set, counted from the set's first, when this is set
/// and not empty: a deterministic slice of a benchmark-scale set, so CI can replay a set of fifty
/// thousand cases in minutes, and a nightly run, which leaves it unset, replays them all. The
/// hand-written sets ([`HAND_WRITTEN_SETS`]) are replayed whole whatever the step. The run says
/// so.
const SAMPLE_ENV: &str = "SYMPHONY_SAMPLE_EVERY";
/// Set to anything but empty or `0`, the per-case report is printed on stdout as well.
const VERBOSE_ENV: &str = "BELLWETHER_VERBOSE";
/// The directory the per-case reports are written to; the build's `target/tmp` when it is not set.
const REPORT_DIR_ENV: &str = "BELLWETHER_REPORT_DIR";
/// How many differing cases of a model stdout shows, and how many failures the run's message
/// lists; the rest are in the per-case report.
const SHOWN: usize = 20;
/// The hand-written sets in every slug's directory, `common` (the probes, bellwether #16, and the
/// parallel calls) and `turns` (the turn shapes a table reads): a few cases that a sample must
/// not thin.
const HAND_WRITTEN_SETS: &[&str] = &["common", "turns"];
/// Up to this many bytes an output is also replayed at every two-way split, which costs as many
/// replays as it has characters; past it the sampled plans alone keep a case's cost linear in
/// its size (a 252 KB swebench case took 80 minutes and 13 GB under every split).
const EVERY_SPLIT_UP_TO: usize = 256;
const SLUG: &str = "qwen3-8b";
/// Slugs the replay over [`MODELS`] leaves out for now, each with the reason
/// and where it is tracked: a family whose parser is known to differ from the
/// reference on whole sets, which no per-case list could carry. The slug stays
/// in the rows so its prompt tail is kept; the issue brings it back.
const SKIPPED: &[(&str, &str)] = &[
    // bellwether #149 recorded its sets again without the common set (the 16 cases are gone from
    // main 25b7767791), so the probes its family lists have no fixture to meet; replayed again
    // once bellwether records them.
    (
        "step-3.5-flash",
        "bellwether #149 dropped its common set, so the listed probes have no fixture to meet",
    ),
];
/// bellwether's slugs for the checkpoints the tables read, in its manifests' spelling, each with
/// how its template ends the generation prompt; the table that reads it, and its turn opener, are
/// the registry's ([`for_model`]). The fixtures carry the request and the output, not the rendered
/// prompt, so the prompt's tail is stated here until bellwether records it. A slug bellwether has
/// not recorded is skipped with a notice; `qwen3-8b` is the one set bellwether's main always holds,
/// and has its own test.
const MODELS: &[(&str, GenerationPrompt)] = &[
    // Qwen3: the model writes its own `<think>`; thinking off closes it in the prompt.
    ("qwen3-8b", GenerationPrompt::ModelWritesTheThought),
    ("qwen3-4b-saferl", GenerationPrompt::ModelWritesTheThought),
    ("qwen3-0.6b", GenerationPrompt::ModelWritesTheThought),
    ("qwen3-235b-a22b", GenerationPrompt::ModelWritesTheThought),
    ("qwen3-30b-a3b", GenerationPrompt::ModelWritesTheThought),
    // The 2507 line, Qwen3-Next and Qwen3-VL split into instruct, with no thought, and thinking,
    // whose template always opens the thought in the prompt.
    ("qwen3-4b-instruct-2507", GenerationPrompt::Plain),
    ("qwen3-30b-a3b-instruct-2507", GenerationPrompt::Plain),
    ("qwen3-235b-a22b-instruct-2507", GenerationPrompt::Plain),
    (
        "qwen3-4b-thinking-2507",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-30b-a3b-thinking-2507",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-235b-a22b-thinking-2507",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    ("qwen3-next-80b-a3b-instruct", GenerationPrompt::Plain),
    (
        "qwen3-next-80b-a3b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    ("qwen3-vl-2b-instruct", GenerationPrompt::Plain),
    ("qwen3-vl-4b-instruct", GenerationPrompt::Plain),
    ("qwen3-vl-8b-instruct", GenerationPrompt::Plain),
    ("qwen3-vl-30b-a3b-instruct", GenerationPrompt::Plain),
    ("qwen3-vl-32b-instruct", GenerationPrompt::Plain),
    ("qwen3-vl-235b-a22b-instruct", GenerationPrompt::Plain),
    (
        "qwen3-vl-2b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-4b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-8b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-30b-a3b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-32b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "qwen3-vl-235b-a22b-thinking",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    // Qwen3-Omni: the Qwen3 template over audio and video content too, so the same syntax and
    // the same switch; both checkpoints leave the thought to the model.
    (
        "qwen3-omni-30b-a3b-instruct",
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "qwen3-omni-30b-a3b-thinking",
        GenerationPrompt::ModelWritesTheThought,
    ),
    // Qwen2.5: no thought at all.
    ("qwen2.5-0.5b-instruct", GenerationPrompt::Plain),
    ("qwen2.5-1.5b-instruct", GenerationPrompt::Plain),
    ("qwen2.5-7b-instruct-1m", GenerationPrompt::Plain),
    ("qwen2.5-14b-instruct-1m", GenerationPrompt::Plain),
    ("qwen2.5-vl-3b-instruct", GenerationPrompt::Plain),
    ("qwen2.5-vl-7b-instruct", GenerationPrompt::Plain),
    ("qwen2.5-vl-32b-instruct", GenerationPrompt::Plain),
    ("qwen2.5-omni-7b", GenerationPrompt::Plain),
    // Qwen 3.5 and later write calls as tags and open the thought in the prompt; Qwen3-Coder
    // writes tags and has no thought.
    // Qwen3.5-0.8B's template has no thought.
    ("qwen3.5-0.8b", GenerationPrompt::Plain),
    // Qwen3.5-2B's opens the thought only when the request asks for it.
    ("qwen3.5-2b", GenerationPrompt::OpensTheThoughtWhenAsked),
    ("qwen3.5-4b", GenerationPrompt::OpensTheThought),
    ("qwen3.5-9b", GenerationPrompt::OpensTheThought),
    ("qwen3.5-27b", GenerationPrompt::OpensTheThought),
    ("qwen3.5-35b-a3b", GenerationPrompt::OpensTheThought),
    ("qwen3.5-122b-a10b", GenerationPrompt::OpensTheThought),
    ("qwen3.6-27b", GenerationPrompt::OpensTheThought),
    ("qwen3.6-35b-a3b", GenerationPrompt::OpensTheThought),
    ("qwen3.8-27b", GenerationPrompt::OpensTheThought),
    ("qwen3.8-flash-next", GenerationPrompt::OpensTheThought),
    // Qwen3.8-2.4T-A95B's template opens the thought whatever the request says and raises on a
    // request that turns thinking off, so bellwether refused those cases ([`REFUSED_PROBES`]).
    ("qwen3.8-2.4t-a95b", GenerationPrompt::AlwaysOpensTheThought),
    ("qwen3-coder-30b-a3b-instruct", GenerationPrompt::Plain),
    ("qwen3-coder-next", GenerationPrompt::Plain),
    // Qwen-AgentWorld and Qwen-Drive write Qwen 3.5's template (their `chat_template.jinja`):
    // tagged calls, the thought opened in the prompt. Qwen-Drive's `tokenizer_config.json` carries
    // an older template without a thought; the recorded outputs follow the former.
    ("qwen-agentworld-35b-a3b", GenerationPrompt::OpensTheThought),
    ("qwen-drive-1.0-4b", GenerationPrompt::OpensTheThought),
    // DeepSeek V4.1 writes DSML and opens the thought in the prompt (`<think>`, no newline).
    ("deepseek-v4.1-flash", GenerationPrompt::OpensTheThought),
    // DeepSeek V3.2 and V3.2-Speciale write DSML; the encoder ends the prompt with `<think>` in
    // thinking mode and `</think>` in chat mode, right after `<｜Assistant｜>`. bellwether holds
    // their manifests and no set yet: each slug is skipped with the notice until the pin holds it.
    ("deepseek-v3.2", GenerationPrompt::OpensTheThought),
    ("deepseek-v3.2-speciale", GenerationPrompt::OpensTheThought),
    // The Kimi K2 line reads with one table, and each template ends the generation prompt its own
    // way: K2 Instruct and Instruct-0905 write no thought; K2 Thinking ends at the turn's opener
    // and the model writes its own `<think>`; K2.5, K2.6 and K2.7-Code open the thought in the
    // prompt (`<think>`, no newline: the shared tail's is prompt text inside the thought, as for
    // Kimi K3). K2.5 and K2.6 close it at once (`<think></think>`) when the request's `thinking`
    // is false, a switch bellwether's requests do not set (they carry `enable_thinking`), so the
    // recorded sets start inside the thought whatever the request says. bellwether holds the six
    // manifests and no set yet: each slug is skipped with the notice until the pin holds its sets.
    ("kimi-k2-instruct", GenerationPrompt::Plain),
    ("kimi-k2-instruct-0905", GenerationPrompt::Plain),
    ("kimi-k2-thinking", GenerationPrompt::ModelDecides),
    ("kimi-k2.5", GenerationPrompt::AlwaysOpensTheThought),
    ("kimi-k2.6", GenerationPrompt::AlwaysOpensTheThought),
    ("kimi-k2.7-code", GenerationPrompt::AlwaysOpensTheThought),
    // DeepSeek V3.1 and V3.2-Exp close the thought in the prompt (`</think>` right after the
    // opener) unless the request's `thinking` kwarg opens it; the recorded requests carry
    // `enable_thinking`, which these templates do not read, so every recorded prompt closes it
    // and the output starts in content. DeepSeek V3's template writes no thought. Their sets hold
    // content alone: the templates take a call's arguments as a string, and the recorder refused every
    // call case of the corpus, whose arguments are objects.
    ("deepseek-v3.1", GenerationPrompt::Plain),
    ("deepseek-v3.2-exp", GenerationPrompt::Plain),
    ("deepseek-v3-0324", GenerationPrompt::Plain),
    ("deepseek-v3-0324-bf16", GenerationPrompt::Plain),
    // The R1 line writes V3's call spelling from its history and a thought of its own. The
    // recorder rendered each reference through the template, which keeps an assistant message
    // from its `</think>` on and takes a call's arguments as a string, so the sets hold content
    // alone and the replay starts in content; the prompt's own tail (`<think>` after the
    // assistant header in R1's, R1-Zero's and the distills' templates, nothing in R1-0528's,
    // R1-0528-Qwen3-8B's and the bf16 mirror's) is held by the table's module tests.
    ("deepseek-r1-0528", GenerationPrompt::Plain),
    ("deepseek-r1-0528-qwen3-8b", GenerationPrompt::Plain),
    ("deepseek-r1-bf16", GenerationPrompt::Plain),
    ("deepseek-r1-distill-llama-8b", GenerationPrompt::Plain),
    ("deepseek-r1-distill-qwen-1.5b", GenerationPrompt::Plain),
    // Kimi K3 writes XTML and opens the thought in the prompt, whatever the request says: its
    // recorded requests carry no switch. The shared tail adds a newline the template does not
    // write; it is prompt text inside the thought and moves nothing.
    ("kimi-k3", GenerationPrompt::AlwaysOpensTheThought),
    // Muse Spark and Muse Glimmer: the generation prompt ends at the turn's opener, and the model
    // writes the thought's message itself. Glimmer's template writes the same messages (to self,
    // to the tool, to the user) and the same block; its recorded set reads with the Spark table.
    ("muse-glimmer-30b", GenerationPrompt::ModelDecides),
    ("muse-spark-open-816b-a42b", GenerationPrompt::ModelDecides),
    // Other families that write Qwen's syntaxes, each as its own template spells the thought.
    ("webworld-32b", GenerationPrompt::ModelWritesTheThought),
    ("webworld-14b", GenerationPrompt::ModelWritesTheThought),
    ("k-exaone-236b-a23b", GenerationPrompt::OpensTheThought),
    ("hermes-4-14b", GenerationPrompt::Plain),
    ("granite-4.1-3b", GenerationPrompt::Plain),
    ("ai21-jamba2-3b", GenerationPrompt::Plain),
    ("step-3.5-flash", GenerationPrompt::AlwaysOpensTheThought),
    ("nanbeige4.2-3b", GenerationPrompt::OpensTheThought),
    // MiMo writes Qwen 3.5's tags with the value between them directly, and its own `<think>`.
    ("mimo-v2.5", GenerationPrompt::ModelWritesTheThought),
    // The V2.6 line writes the same tags with no newline between them; the generation prompt
    // leaves the model to write `<think>`, as V2.5's does, and thinking off closes it in the prompt
    // as `<think></think>` (no newlines, where the shared tail writes them: the state after the
    // replay is content either way).
    (
        "mimo-v2.6-flash-rl",
        GenerationPrompt::ModelWritesTheThought,
    ),
    ("mimo-v2.6-pro-rl", GenerationPrompt::ModelWritesTheThought),
    (
        "mimo-v2.6-distill-qwen-9b",
        GenerationPrompt::ModelWritesTheThought,
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "seed-oss-36b-instruct",
        GenerationPrompt::ModelWritesTheThought,
    ),
    // Keyed arguments, each family under its own markers; the prompt opens the thought, Hy4's
    // whatever the request says.
    ("hy4-preview", GenerationPrompt::AlwaysOpensTheThought),
    ("ling-3.0-flash", GenerationPrompt::OpensTheThought),
    // GLM writes Ling's syntax with nothing between the tags; its prompt opens the thought
    // whatever the request says (the template has no thinking switch).
    ("glm-5.3-flash", GenerationPrompt::AlwaysOpensTheThought),
    // GLM 4.5 and 4.6 write Ling's syntax with a newline after each pair instead of before the
    // close, under GLM's turn opener, and the model writes its own
    // `<think>`; the template closes an empty thought in the prompt only when the request turns
    // thinking off (as `\n<think></think>`; the tail's newlines are not read, its markers are).
    ("glm-4.6", GenerationPrompt::ModelWritesTheThought),
    // K2-Horizon writes GLM 4.6's keyed spelling under `ifm|`-prefixed tags in a block, and its
    // prompt opens the thought whatever the request says (the template has no thinking switch;
    // `reasoning_effort` picks the tag, `<ifm|think>` by default). The template raises on an
    // assistant turn without a thinking field, so bellwether records five of its sixteen common
    // cases, the reasoning ones and the thinking-on call.
    ("k2-horizon-36b", GenerationPrompt::AlwaysOpensTheThought),
    ("iquest-q1", GenerationPrompt::OpensTheThought),
    // Content and nothing else: templates without a thought or a call syntax, whose parse sets
    // hold content alone.
    ("tinyllama-1.1b-chat-v1.0", GenerationPrompt::Plain),
    ("phi-4-mini-instruct", GenerationPrompt::Plain),
    ("phi-4-multimodal-instruct", GenerationPrompt::Plain),
    // Python calls: Olmo 3 has no thought; LFM2.5's template adds nothing to the turn opener in
    // either thinking setting (its render set), and the model writes `<think>` itself or not.
    ("olmo-3-7b-instruct", GenerationPrompt::Plain),
    ("lfm2.5-1.2b-instruct", GenerationPrompt::ModelDecides),
    // Llama 4's list stands bare in the turn after the content, every value between double
    // quotes; the template has no thought and ends the generation prompt with the opener and two
    // newlines, which the shared tail does not write and the replay does not need.
    ("llama-4-scout-17b-16e-instruct", GenerationPrompt::Plain),
    (
        "llama-4-maverick-17b-128e-instruct",
        GenerationPrompt::Plain,
    ),
    // A bare JSON list of calls, or content; no thought.
    ("llama-xlam-2-8b-fc-r", GenerationPrompt::Plain),
    // The Mistral list behind `[TOOL_CALLS]`; Mistral Small 3's template renders no tool call,
    // so its recorded sets hold content alone, and no thought.
    ("mistral-small-24b-instruct-2501", GenerationPrompt::Plain),
    // An XML tree of arguments behind a separator token; the model writes the thought or closes
    // an empty one, and the generation prompt adds nothing.
    ("minimax-m3", GenerationPrompt::ModelDecides),
    // MiniMax M2 and M2.7 write `<think>` blocks and `<minimax:tool_call>` blocks of invokes
    // whose parameter tags name their key in an attribute; their prompt opens the thought on its
    // own line whatever the request says (the template has no thinking switch).
    ("minimax-m2", GenerationPrompt::AlwaysOpensTheThought),
    ("minimax-m2.7", GenerationPrompt::AlwaysOpensTheThought),
    // Inkling: the generation prompt is the first message's opener and nothing more, whatever the
    // request says (a system message carries the reasoning effort); the model writes the thought
    // or not, and its markers are its own.
    ("inkling", GenerationPrompt::ModelDecides),
    // MiniCPM5: the model writes its own `<think>`, as Qwen3 does; thinking off closes it in the
    // prompt.
    ("minicpm5-2b", GenerationPrompt::ModelWritesTheThought),
    // dots3: the model writes its own `<think>`, as Qwen3 does; thinking off closes it in the
    // prompt.
    ("dots3-note-prev", GenerationPrompt::ModelWritesTheThought),
    // Laguna writes GLM 4.5's spelling (a newline after the name, each key and each value) under
    // its own turn opener; its prompt ends in the thought's opener or its closer alone.
    ("laguna-xs.2", GenerationPrompt::MarkerAlone),
    // Gemma 4: the prompt is the turn's opener alone, whatever the request says (the template opens
    // the thought channel only after a tool response); the model writes `<|channel>thought` or not.
    ("gemma-4-e4b-it", GenerationPrompt::ModelDecides),
    // Qwen3's JSON calls under other templates (bellwether main 5bcd7b0b0): each as its own
    // template writes the thought.
    ("agentcpm-explore", GenerationPrompt::ModelWritesTheThought),
    (
        "ai21-jamba-reasoning-3b",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    ("astabrief-8b", GenerationPrompt::ModelWritesTheThought),
    ("baichuan-m3-235b", GenerationPrompt::AlwaysOpensTheThought),
    ("caprl-qwen3vl-4b", GenerationPrompt::Plain),
    ("caprl-video-4b", GenerationPrompt::Plain),
    ("contextpilot-14b", GenerationPrompt::ModelWritesTheThought),
    ("falcon-h1r-7b", GenerationPrompt::ModelDecides),
    (
        "nemotron-cascade-14b-thinking",
        GenerationPrompt::ModelDecides,
    ),
    (
        "nemotron-orchestrator-8b",
        GenerationPrompt::ModelWritesTheThought,
    ),
    // Qwen2.5's JSON calls, no thought.
    ("falcon-h1-tiny-90m-instruct", GenerationPrompt::Plain),
    (
        "nemotron-research-goosereason-4b-instruct",
        GenerationPrompt::Plain,
    ),
    ("opencodereasoning-nemotron-7b", GenerationPrompt::Plain),
    // Qwen 3.5's tagged calls, the thought opened in the prompt and closed there when the
    // request turns thinking off.
    ("arex-2", GenerationPrompt::OpensTheThought),
    ("arex-base", GenerationPrompt::OpensTheThought),
    ("arex-turbo", GenerationPrompt::OpensTheThought),
    ("cosmos3-edge", GenerationPrompt::OpensTheThought),
    (
        "cosmos3-edge-policy-droid",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nemotron-cascade-2-30b-a3b",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-3-nano-4b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-3-super-120b-a12b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-3-ultra-550b-a55b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-3.5-lightning-30b-a3b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-labs-3-elastic-30b-a3b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    (
        "nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16",
        GenerationPrompt::OpensTheThought,
    ),
    // GLM 4.5's keyed spelling (a newline after the name, each key and each value); the model
    // writes its own `<think>`, closed in the prompt when thinking is off.
    ("glm-4.5v", GenerationPrompt::ModelWritesTheThought),
    ("glm-4.6v", GenerationPrompt::ModelWritesTheThought),
    ("glm-4.6v-flash", GenerationPrompt::ModelWritesTheThought),
    // GLM's compact keyed spelling; the prompt opens the thought, or closes it when the request
    // turns thinking off; GLM-5.3 opens it whatever the request says.
    ("atria-dawn-preview", GenerationPrompt::OpensTheThought),
    ("glm-4.7", GenerationPrompt::OpensTheThought),
    ("glm-4.7-flash", GenerationPrompt::OpensTheThought),
    ("glm-5", GenerationPrompt::OpensTheThought),
    ("glm-5.1", GenerationPrompt::OpensTheThought),
    ("glm-5.2", GenerationPrompt::OpensTheThought),
    ("glm-5.3", GenerationPrompt::AlwaysOpensTheThought),
    // MiniMax M2's blocks of invokes; the prompt opens the thought whatever the request says.
    ("minimax-m2.5", GenerationPrompt::AlwaysOpensTheThought),
    // LFM2.5's Python calls under d1's template, which adds nothing to the turn opener in either
    // thinking setting; the model writes `<think>` itself or not.
    ("d1-3b", GenerationPrompt::ModelDecides),
    // Mistral: `[THINK]` and `[/THINK]` the model's own, `[TOOL_CALLS]name[ARGS]{…}` per call; the
    // prompt ends at `[/INST]` whatever the request says. The instruct templates have no thought.
    ("devstral-2-123b-instruct-2512", GenerationPrompt::Plain),
    (
        "ministral-3-14b-instruct-2512-bf16",
        GenerationPrompt::Plain,
    ),
    (
        "ministral-3-14b-reasoning-2512",
        GenerationPrompt::ModelDecides,
    ),
    ("ministral-3-3b-instruct-2512", GenerationPrompt::Plain),
    ("ministral-3-3b-instruct-2512-bf16", GenerationPrompt::Plain),
    ("ministral-3-8b-instruct-2512-bf16", GenerationPrompt::Plain),
    ("mistral-medium-3.5-128b", GenerationPrompt::ModelDecides),
    ("mistral-small-4-119b-2603", GenerationPrompt::ModelDecides),
    // Llama Nemotron: one JSON list of calls between `<TOOLCALL>` and `</TOOLCALL>`. The Super
    // template has a thought the model writes itself (`/no_think` in the system prompt closes it
    // in the prompt; the request's switch changes nothing); the Nano and DLER templates have none.
    (
        "dler-llama-nemotron-8b-merge-research",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3-3-nemotron-super-49b-v1-5",
        GenerationPrompt::ModelDecides,
    ),
    ("llama-3.1-nemotron-nano-8b-v1", GenerationPrompt::Plain),
    // The Ultra 253B v1 template (not the CPT checkpoint's, which keeps Llama 3.1's own) renders
    // neither a call nor a thought and keeps an assistant message from its `</think>` on: the set
    // holds content alone, the reasoning probe was refused at record time, and the model writes
    // `<think>` itself under its "detailed thinking on" system prompt, as the Nano does.
    ("llama-3-1-nemotron-ultra-253b-v1", GenerationPrompt::Plain),
    // Qwen3's JSON calls under other templates (bellwether main 8029536af5): EXAONE 4.5 opens the
    // thought in the prompt under its own opener; AReaL-SEA and the tau2 SFT open it whatever the
    // request says; the boba-2 line and the tau2 user SFT leave it to the model.
    ("areal-boba-2-14b", GenerationPrompt::ModelWritesTheThought),
    ("areal-boba-2-32b", GenerationPrompt::ModelWritesTheThought),
    ("areal-boba-2-8b", GenerationPrompt::ModelWritesTheThought),
    (
        "areal-sea-235b-a22b",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "areal-tau2-telecom-sft-30b",
        GenerationPrompt::AlwaysOpensTheThought,
    ),
    (
        "areal-tau2-telecom-user-sft-30b",
        GenerationPrompt::ModelWritesTheThought,
    ),
    ("exaone-4.5-33b", GenerationPrompt::OpensTheThought),
    // Qwen2.5's JSON calls, no thought: A.X under its own opener (`<|im_start|><|assistant|>`,
    // the object without newlines), the Seed-OSS alternate tokenizer under Llama 3's header, the
    // rest under Qwen2.5's own template.
    ("a.x-3.1-light", GenerationPrompt::Plain),
    ("a.x-4.0", GenerationPrompt::Plain),
    ("a.x-4.0-light", GenerationPrompt::Plain),
    ("baichuanmed-ocr-7b", GenerationPrompt::Plain),
    (
        "bytedance-seed-oss-36b-alternate-tokenizer",
        GenerationPrompt::Plain,
    ),
    ("caller", GenerationPrompt::Plain),
    ("caprl-3b", GenerationPrompt::Plain),
    ("caprl-eval-3b", GenerationPrompt::Plain),
    ("cogalign-internvl2-5-mpo-4b", GenerationPrompt::Plain),
    ("crispers-7b-v1", GenerationPrompt::Plain),
    ("ernie-image-aes", GenerationPrompt::Plain),
    // Qwen 3.5's tagged calls, the thought opened in the prompt and closed there when the
    // request turns thinking off.
    ("fara1.5-27b", GenerationPrompt::OpensTheThought),
    ("fara1.5-9b", GenerationPrompt::OpensTheThought),
    ("h2o-lightning-4b", GenerationPrompt::OpensTheThought),
    ("huatuogpt-3-27b", GenerationPrompt::OpensTheThought),
    // Llama 3.1's own template under the Nemotron Ultra, UltraLong, Safety Guard and Reward
    // checkpoints: one call object alone in the turn, `parameters` naming the arguments, no
    // thought; the template refuses parallel calls and content beside a call.
    (
        "llama-3-1-nemotron-ultra-253b-cpt-v1",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-1m-instruct",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-2m-instruct",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-4m-instruct",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3.1-nemotron-safety-guard-8b-v3",
        GenerationPrompt::Plain,
    ),
    (
        "llama-3.3-nemotron-70b-reward-principle",
        GenerationPrompt::Plain,
    ),
    // Llama 3.2's own template: the same one call alone, `parameters` naming the arguments, no
    // thought; the model's `<|python_tag|>`, which the template never writes, is dropped.
    ("llama-3.2-1b-instruct", GenerationPrompt::Plain),
    ("llama-3.2-11b-vision-instruct", GenerationPrompt::Plain),
    // Llama 3.1 8B/70B/405B Instruct and Llama 3.3 70B Instruct: one template, byte for byte the
    // one the Nemotron Ultra CPT, UltraLong and Safety Guard rows above carry; bellwether has
    // recorded neither slug yet, so each is skipped with the notice until the pin holds its sets.
    ("llama-3.1-8b-instruct", GenerationPrompt::Plain),
    ("llama-3.3-70b-instruct", GenerationPrompt::Plain),
    // FunctionGemma: Gemma's calls between `<start_function_call>` and `<end_function_call>`,
    // `<escape>` around a string and `None` for a null; the template has no thought.
    ("functiongemma-270m-it", GenerationPrompt::Plain),
];

/// What the replay knows about a table beyond the registry: the spelling of its thought's
/// markers, whether its template respells a call's arguments, the probe cases known to differ
/// under its syntax, and the corpus classes it meets.
trait Recorded: Copy {
    fn think_markers(self) -> (&'static str, &'static str);
    fn respells_arguments(self) -> bool;
    fn quotes_values(self) -> bool;
    fn holds_a_list_after_content(self) -> bool;
    fn known_differences(self, slug: &str, prompt: GenerationPrompt) -> Vec<KnownDifference>;
    fn allowances(self, slug: &str, prompt: GenerationPrompt) -> Vec<Allowance>;
}

impl Recorded for Table {
    /// How the family spells the thought's markers, for the prompt's tail and the reasoning
    /// allowance's guard.
    fn think_markers(self) -> (&'static str, &'static str) {
        match self {
            Self::SeedOss => ("<seed:think>", "</seed:think>"),
            Self::Hy4 => ("<think:opensource>", "</think:opensource>"),
            Self::MinimaxM3 => ("<mm:think>", "</mm:think>"),
            Self::KimiK3 => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
            Self::MuseSpark => (" to=self<|message|>", "<|eom|>"),
            Self::Inkling => ("<|content_thinking|>", "<|end_message|>"),
            Self::Gemma4 => ("<|channel>thought\n", "<channel|>"),
            Self::K2Horizon => ("<ifm|think>", "</ifm|think>"),
            Self::MistralV13 => ("[THINK]", "[/THINK]"),
            _ => ("<think>", "</think>"),
        }
    }

    /// Whether the family's template re-serializes a call's arguments, so the model writes them
    /// with the keys sorted and the spaces dropped, as Inkling's `tojson(sort_keys=true,
    /// separators=(",", ":"))` does: a case whose arguments differ from the reference's only so,
    /// equal as values, is counted as respelled, not as a difference.
    fn respells_arguments(self) -> bool {
        matches!(self, Self::Inkling | Self::Gemma4 | Self::FunctionGemma)
    }

    /// Whether the table's template writes every value between double quotes without escaping
    /// a quote inside it (Llama 4's `"%s"`): what the template wrote for a string argument that
    /// holds a `"` does not say where the value ends, so the parser reads it back only when the
    /// quotes inside pair up with nothing between a pair that would end or reopen the value, and
    /// a case that differs and whose reference holds one is counted apart from the differences
    /// (`docs/tables.md`).
    fn quotes_values(self) -> bool {
        matches!(self, Self::Llama4)
    }

    /// Whether the table's list of calls stands bare after the content, so that the table reads
    /// one there only when it is whole, well formed and made of calls to the request's tools
    /// (Llama 4, `docs/tables.md`): a reference whose message holds content and then a call to a
    /// tool the request does not declare is read as content, as the family's own reader reads
    /// it, and such a case is counted apart from the differences.
    fn holds_a_list_after_content(self) -> bool {
        matches!(self, Self::Llama4)
    }

    /// The probe cases known to differ under this syntax: the code fence holds the JSON syntax,
    /// which the JSON assembler reads as a call and the tagged one reports as text between a
    /// call's tags.
    fn known_differences(self, slug: &str, prompt: GenerationPrompt) -> Vec<KnownDifference> {
        let list = match self {
            Self::Qwen3 | Self::Qwen2_5 => KNOWN_DIFFERENCES,
            Self::Qwen3Tagged | Self::Qwen3TaggedDirect => KNOWN_TAGGED_DIFFERENCES,
            // Seed-OSS, Hy4, Olmo 3, Llama 4, xLAM, the json table, Kimi K3, Muse Spark, K2-Horizon,
            // Inkling, Gemma 4 and the plain table read neither of the probes' Qwen markers:
            // `</think>` stays reasoning text and the fenced `<tool_call>` block stays content,
            // as the reference says.
            Self::SeedOss
            | Self::Hy4
            | Self::Olmo3
            | Self::Llama4
            | Self::Xlam
            | Self::Json
            | Self::KimiK3
            | Self::MuseSpark
            | Self::K2Horizon
            | Self::Inkling
            | Self::Gemma4
            | Self::FunctionGemma
            | Self::Mistral
            | Self::MistralV13
            | Self::Plain => &[],
            // Ling and GLM read `<tool_call>` and `</think>`, so their two probes are the
            // tagged ones' (no call comes of the fence); IQuest, the DSML tables, Kimi K2, LFM2.5,
            // MiniMax M2, MiniCPM5, dots3 and Llama Nemotron read `</think>` but not
            // `<tool_call>`.
            Self::Ling | Self::Glm | Self::Glm4 => KNOWN_TAGGED_DIFFERENCES,
            Self::IQuest
            | Self::DeepSeekV3_2
            | Self::DeepSeekV3
            | Self::DeepSeekV3_1
            | Self::DeepSeekV4
            | Self::DeepSeekV4_1
            | Self::KimiK2
            | Self::Lfm2_5
            | Self::MinimaxM2
            | Self::MiniCpm5
            | Self::Dots3
            | Self::LlamaNemotron => KNOWN_REASONING_PROBE,
            Self::MinimaxM3 => KNOWN_M3_DIFFERENCES,
        };
        // A template that writes no thought for the probe's request leaves the reasoning out, so
        // the marker inside it is never read; that case falls under the reasoning allowance, or
        // was refused at record time, instead of the list. A probe a template refused outright is
        // not among the slug's fixtures either.
        list.iter()
            .copied()
            .filter(|known| !prompt.writes_no_thought_unasked() || known.id != REASONING_PROBE.id)
            .filter(|known| !REFUSED_PROBES.contains(&(slug, known.id)))
            .collect()
    }

    /// The corpus classes this syntax meets in sets recorded before bellwether refused them,
    /// when the run opts in ([`CORPUS_ALLOWANCES_ENV`]): the tagged syntaxes cannot carry a type
    /// the text does not say, a template without a thought drops the reference's reasoning, four
    /// Qwen2.5 templates write content or calls, not both, xLAM's writes the calls and drops the
    /// content beside them, MiniMax M3's writes no element for a null, and Qwen 3.5's trims the
    /// content's edges.
    fn allowances(self, slug: &str, prompt: GenerationPrompt) -> Vec<Allowance> {
        if !corpus_allowances_on() {
            return Vec::new();
        }
        let mut allowed = Vec::new();
        if matches!(
            self,
            Self::Qwen3Tagged
                | Self::Qwen3TaggedDirect
                | Self::SeedOss
                | Self::Hy4
                | Self::Ling
                | Self::IQuest
                | Self::Glm
        ) {
            allowed.push(Allowance::DeclaredTypeConflict);
        }
        if prompt == GenerationPrompt::Plain {
            allowed.push(Allowance::ReasoningNotWritten);
        }
        if CONTENT_OR_CALLS.contains(&slug) {
            allowed.push(Allowance::CallsNotWritten);
        }
        if self == Self::Xlam {
            allowed.push(Allowance::ContentNotWritten);
        }
        if self == Self::MinimaxM3 {
            allowed.push(Allowance::DeclaredTypeConflict);
            allowed.push(Allowance::NullNotWritten);
        }
        if matches!(self, Self::Qwen3Tagged | Self::Qwen3TaggedDirect) {
            allowed.push(Allowance::ContentTrimmed);
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
    /// Qwen3.5-2B: `<think>\n` only when the request turns thinking on (`enable_thinking: true`);
    /// otherwise `<think>\n\n</think>\n\n`, and the output starts in content. The reverse of
    /// [`Self::OpensTheThought`]'s default, so a request without the switch gets no thought.
    OpensTheThoughtWhenAsked,
    /// Qwen3: nothing, and the model writes its own `<think>`; thinking off closes it in the
    /// prompt as above.
    ModelWritesTheThought,
    /// Qwen3-Coder, Qwen2.5 and the instruct variants: `<|im_start|>assistant\n` and nothing more;
    /// the template has no thought.
    Plain,
    /// MiniMax M3 and Kimi K2 Thinking: nothing, whatever the request says (M3's template reads
    /// `thinking_mode`, which the requests do not set; K2 Thinking's has no switch), and the
    /// template has a thought: the model writes `<mm:think>` or `<think>` itself, or (M3 only)
    /// closes an empty thought with `</mm:think>` at once; the Kimi K2 table reads a closer
    /// with no opener as content.
    ModelDecides,
    /// Laguna: the thought's opener alone, with no newline, when the request turns thinking on,
    /// so the output starts inside the thought; otherwise (the default) its closer alone, which
    /// closes a thought never opened and leaves the output in content.
    MarkerAlone,
}

impl GenerationPrompt {
    /// The bytes after the assistant header for the case's request, in the family's spelling of
    /// the thought's markers.
    fn tail(self, fixture: &Fixture, table: Table) -> String {
        self.tail_for(&fixture.request, table)
    }

    fn tail_for(self, request: &Request, table: Table) -> String {
        let (open, close) = table.think_markers();
        let opened = format!("{open}\n");
        let closed = format!("{open}\n\n{close}\n\n");
        match (self, request.thinking_off()) {
            (Self::Plain | Self::ModelDecides, _) | (Self::ModelWritesTheThought, false) => {
                String::new()
            }
            (Self::OpensTheThought | Self::ModelWritesTheThought, true) => closed,
            (Self::OpensTheThought, false) | (Self::AlwaysOpensTheThought, _) => opened,
            (Self::OpensTheThoughtWhenAsked, _) if request.thinking_on() => opened,
            (Self::OpensTheThoughtWhenAsked, _) => closed,
            (Self::MarkerAlone, _) if request.thinking_on() => open.to_string(),
            (Self::MarkerAlone, _) => close.to_string(),
        }
    }

    /// Whether the template writes no thought for a request that leaves the switch alone: the
    /// reasoning probe's does, so its reasoning is not in the output and the case is refused or
    /// allowed, not listed.
    fn writes_no_thought_unasked(self) -> bool {
        matches!(
            self,
            Self::Plain | Self::OpensTheThoughtWhenAsked | Self::MarkerAlone
        )
    }
}

/// Every recorded model is in the registry, so the replays below read each with the table
/// `formats::for_model` gives it, the one the gateway asks for; a correction to the registry
/// reaches the replay. The replay feeds the generation prompt's tail alone, so it says nothing
/// about a turn opener: [`MEASURED_OPENERS`] pins those.
#[test]
fn every_recorded_model_is_in_the_registry() {
    for (slug, _) in MODELS {
        let model = model_of(slug);
        assert!(for_model(&model).is_some(), "{slug}: not in the registry");
    }
}

/// The turn opener a recorded model's own chat template writes where it is not its table's,
/// measured against the templates' rendered prompts (smg #2841): K-EXAONE writes Qwen3's markers
/// under its own template, Granite 4.1 Qwen2.5's, and the rest render their own table's opener
/// or `<|im_start|>assistant`. An independent statement of what the registry's groups hold, so a
/// typo in a group's opener fails here rather than in a gateway that replays the whole prompt.
const MEASURED_OPENERS: &[(&str, &str)] = &[
    ("k-exaone-236b-a23b", "<|assistant|>"),
    (
        "granite-4.1-3b",
        "<|start_of_role|>assistant<|end_of_role|>",
    ),
    ("laguna-xs.2", "<assistant>"),
    ("a.x-3.1-light", "<|im_start|><|assistant|>"),
    ("a.x-4.0", "<|im_start|><|assistant|>"),
    ("a.x-4.0-light", "<|im_start|><|assistant|>"),
    (
        "bytedance-seed-oss-36b-alternate-tokenizer",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.2-1b-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.2-11b-vision-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.1-8b-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.3-70b-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    ("exaone-4.5-33b", "<|assistant|>"),
    (
        "llama-3-1-nemotron-ultra-253b-cpt-v1",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-1m-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-2m-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.1-nemotron-8b-ultralong-4m-instruct",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.1-nemotron-safety-guard-8b-v3",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
    (
        "llama-3.3-nemotron-70b-reward-principle",
        "<|start_header_id|>assistant<|end_header_id|>",
    ),
];

/// What a recorded model's own chat template writes after the thought's close before a call,
/// where it is not its table's, measured against the rendered prompts (bellwether's
/// `render/common.jsonl`, `tools-call-arguments-object`): the NVIDIA Nemotron 3 groups and Step
/// 3.5 Flash write `</think>\n<tool_call>` where the Qwen 3.5 template behind their table writes
/// two newlines.
/// The registry's groups carry it as the group's `after`; the test after the next reads the
/// rendered prompts themselves where the tree holds them.
const MEASURED_AFTER: &[(&str, &str)] = &[
    ("cosmos3-edge", "\n"),
    ("cosmos3-edge-policy-droid", "\n"),
    ("nemotron-cascade-2-30b-a3b", "\n"),
    ("nvidia-nemotron-3-nano-30b-a3b-bf16", "\n"),
    ("nvidia-nemotron-3-nano-4b-bf16", "\n"),
    ("nvidia-nemotron-3-super-120b-a12b-bf16", "\n"),
    ("nvidia-nemotron-3-ultra-550b-a55b-bf16", "\n"),
    ("nvidia-nemotron-3.5-lightning-30b-a3b-bf16", "\n"),
    ("nvidia-nemotron-labs-3-elastic-30b-a3b-bf16", "\n"),
    ("nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16", "\n"),
    ("step-3.5-flash", "\n"),
];

#[test]
fn every_measured_opener_and_text_after_the_close_is_the_registrys() {
    for (slug, _) in MEASURED_OPENERS.iter().chain(MEASURED_AFTER) {
        assert!(
            MODELS.iter().any(|(recorded, _)| recorded == slug),
            "{slug}: measured but not recorded"
        );
    }
    for (slug, _) in MODELS {
        let model = model_of(slug);
        let table = table_for(&model).unwrap_or_else(|| panic!("{slug}: not in the registry"));
        // The table as it is, with the measured opener and text after the close where the
        // model's template writes its own; the other recorded models render their table's.
        let mut expected = table.format();
        if let Some((_, opener)) = MEASURED_OPENERS
            .iter()
            .find(|(measured, _)| measured == slug)
        {
            expected = expected.opens_turn(opener);
        }
        if let Some((_, after)) = MEASURED_AFTER.iter().find(|(measured, _)| measured == slug) {
            expected = expected.with_thought_after(after);
        }
        assert_eq!(
            for_model(&model),
            Some(expected),
            "{slug}: the registry's opener and text after the thought's close"
        );
    }
}

/// The text after the thought's close before a call, read from each recorded model's rendered
/// `tools-call-arguments-object` (`render/common.jsonl`, the reference's `text`: the assistant
/// turn that made the call),
/// is the registry's for it: the table's layout, or the group's own `after`. Skipped where the
/// tree is not given; a model the tree has not recorded, a table without a thought layout or a
/// close, dots3, whose template writes the call's own lead before every block, and a turn whose
/// template wrote text of the message there, are not read; a tree under which nothing is read
/// fails, and so does a measured group ([`MEASURED_AFTER`]) the tree records but this test cannot
/// read.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and the count are test diagnostic output"
)]
fn every_recorded_models_text_after_the_thoughts_close_is_the_registrys() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let mut read = Vec::new();
    let mut wrong = Vec::new();
    for (slug, _) in MODELS {
        // A measured group the tree records must be read: a rendering that cannot be read as
        // this test reads it would leave its `after` unchecked without a word.
        let measured = MEASURED_AFTER.iter().any(|(measured, _)| measured == slug);
        let mut unread = |why: String| {
            if measured {
                wrong.push(format!("{slug}: {why}"));
            }
        };
        let format =
            for_model(&model_of(slug)).unwrap_or_else(|| panic!("{slug}: not in the registry"));
        let Some(thought) = format.layout().and_then(|layout| layout.thought.as_ref()) else {
            continue;
        };
        let Some(close) = thought.close.as_deref() else {
            continue;
        };
        if format.name() == "dots3" {
            continue;
        }
        let Ok(text) = fs::read_to_string(root.join(slug).join("render").join("common.jsonl"))
        else {
            continue;
        };
        let Some(rendered) = text
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|case| {
                case["id"]
                    .as_str()
                    .is_some_and(|id| id.ends_with("/render/tools-call-arguments-object"))
            })
            .and_then(|case| case["reference"]["text"].as_str().map(str::to_string))
        else {
            unread("no rendered tools-call-arguments-object with a reference text".to_string());
            continue;
        };
        let Some(at) = rendered.find(close) else {
            unread(format!("no {close:?} in the rendered call turn"));
            continue;
        };
        let rest = &rendered[at + close.len()..];
        let between = &rest[..rest.find('<').unwrap_or(rest.len())];
        // A turn whose template wrote text of the message between the close and the call (a
        // `None` for a content the request left out) says nothing about the layout.
        if !between.chars().all(|c| c == '\n' || c == ' ') {
            unread(format!(
                "{between:?} between {close:?} and the call is message text"
            ));
            continue;
        }
        if between != &*thought.after {
            wrong.push(format!(
                "{slug} ({}): the template writes {between:?} after {close:?} before the call, the registry {:?}",
                format.name(),
                &*thought.after
            ));
        }
        read.push(*slug);
    }
    eprintln!(
        "{} recorded models' rendered call turns read: {}",
        read.len(),
        read.join(", ")
    );
    assert!(
        !read.is_empty(),
        "no rendered call turn read under {}: is it the fixtures/ directory?",
        root.display()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The id the registry knows `slug` by. Bellwether's slug is the repository name in lower case
/// with `_` written `-`, which the registry's case-insensitive match reads as it is, except for a
/// name with an underscore (AstaBrief_8B): that one is found through the registry's own groups.
fn model_of(slug: &str) -> String {
    let slug_of = |id: &str| {
        id.rsplit('/')
            .next()
            .unwrap_or(id)
            .to_ascii_lowercase()
            .replace('_', "-")
    };
    groups()
        .flat_map(|(model, also, _)| std::iter::once(model).chain(also.iter().copied()))
        .find(|id| slug_of(id) == slug)
        .map_or_else(|| slug.to_string(), str::to_string)
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
    "qwen2.5-omni-7b",
];

/// A class of difference the recorded sets have, allowed for what the fixture shows rather than
/// by id, and only when the run opts in. Each is a case bellwether refuses at record time now, or
/// will once the issue named on it lands; the sets that hold them were recorded before the rule
/// that refuses them, and the classes go once those sets are recorded again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allowance {
    /// The reference's argument is a boolean, a number or null for a parameter the tool declares
    /// `string`, so the template writes it as it writes the string, and the parser gives the
    /// string back, as the engine's parser does (BFCL declares `smoking_allowed` an enum of `"True"`, `"False"`
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
    /// The other half of the same: the template writes the calls and drops the content beside
    /// them, so the content the reference carries is not in the output: xLAM (bellwether #68
    /// refuses the case now). Allowed only when the output is the list and nothing else.
    ContentNotWritten,
    /// The template writes no element for an argument whose value is null (`if v is not none`),
    /// so the null the reference carries is not in the output: MiniMax M3. Allowed only when the
    /// output holds no element of that name, so a null the parser lost never passes as one the
    /// template dropped; with the declared-type class allowed too, a member of either class
    /// passes.
    NullNotWritten,
    /// The template trims the content's edges (`content | trim` in Qwen 3.5's), so whitespace at
    /// the edges of the reference's content is not in the output: two Hermes cases of every Qwen
    /// 3.5 group end their content with a space (bellwether #81). Allowed only when the content the
    /// parser gives is the reference's trimmed, the output does not carry the untrimmed content,
    /// and nothing else differs, so a byte the parser lost, inside the content or at its edge,
    /// never passes as one the template trimmed.
    ContentTrimmed,
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

/// Probe cases bellwether refused for a slug, by the slug and the case's id after it: the listed
/// difference cannot be among that slug's fixtures, so the check that the list has not rotted
/// skips it there.
const REFUSED_PROBES: &[(&str, &str)] = &[
    // Its template refuses the reasoning probe's request (the recorded common set has the fence
    // probe and ten more, not it).
    ("laguna-xs.2", "parse/reasoning-with-marker-text"),
    // Its template raises on a request that turns thinking off, and the content probe's does.
    (
        "qwen3.8-2.4t-a95b",
        "parse/content-with-marker-in-code-fence",
    ),
    // bellwether's reference refuses the reasoning probe for its template: twelve of the sixteen
    // common cases are recorded, and that one is not among them.
    ("glm-4.6", "parse/reasoning-with-marker-text"),
    // Probes bellwether's reference refused for these templates (main 5bcd7b0b0): each slug's
    // common set holds its other cases and not these.
    (
        "ai21-jamba-reasoning-3b",
        "parse/content-with-marker-in-code-fence",
    ),
    ("glm-4.5v", "parse/reasoning-with-marker-text"),
    ("glm-4.6v", "parse/reasoning-with-marker-text"),
    ("glm-4.6v-flash", "parse/reasoning-with-marker-text"),
    ("glm-4.7", "parse/reasoning-with-marker-text"),
    ("glm-4.7-flash", "parse/reasoning-with-marker-text"),
    ("glm-5", "parse/reasoning-with-marker-text"),
    ("glm-5.1", "parse/reasoning-with-marker-text"),
    ("glm-5.2", "parse/reasoning-with-marker-text"),
    // Its template renders no `reasoning_content`, so bellwether refused the reasoning probe's
    // request (the recorded common set holds the fence probe and ten more, not it).
    (
        "llama-3-3-nemotron-super-49b-v1-5",
        "parse/reasoning-with-marker-text",
    ),
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

/// Under a table that reads `</think>` but not `<tool_call>`, only the reasoning probe differs:
/// the fence holds Qwen's syntax, which the table never reads as a call, so it is content, as the
/// reference says. The match on the family says which tables these are.
const KNOWN_REASONING_PROBE: &[KnownDifference] = &[REASONING_PROBE];

/// Under the MiniMax M3 table only the code-fence probe differs: the fence holds `<tool_call>`,
/// which this table reads as the block's opening, so the JSON inside is text between calls and the
/// markers are dropped; the reference keeps the fence as content. The reasoning probe's `</think>`
/// is not this family's marker, so it stays reasoning text, as the reference says.
const KNOWN_M3_DIFFERENCES: &[KnownDifference] = &[FENCE_PROBE];

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
        self.enable_thinking() == Some(false)
    }

    fn thinking_on(&self) -> bool {
        self.enable_thinking() == Some(true)
    }

    fn enable_thinking(&self) -> Option<bool> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.enable_thinking)
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

    /// Whether a call's arguments hold a string, at any depth, with a `"` in it, or a string
    /// inside a list or an object with a `'` in it, the keys of a nested object included: what
    /// a template that writes every value between double quotes without escaping cannot write
    /// readably. A nested value is Python's repr between the template's quotes, and repr quotes
    /// an element, or a key, that holds an apostrophe with `"`, so the apostrophe puts a `"`
    /// inside the value just the same; a top-level key is the argument's name, written bare. The
    /// recorder refuses every form by one rule, on the bytes the template writes for the argument.
    fn holds_a_quote_in_a_string(&self) -> bool {
        fn quoted(value: &serde_json::Value, nested: bool) -> bool {
            let unreadable =
                |text: &str, nested: bool| text.contains('"') || (nested && text.contains('\''));
            match value {
                serde_json::Value::String(text) => unreadable(text, nested),
                serde_json::Value::Array(items) => items.iter().any(|item| quoted(item, true)),
                // An object here is an argument's value or deeper, so its keys are nested.
                serde_json::Value::Object(members) => members
                    .iter()
                    .any(|(key, member)| unreadable(key, true) || quoted(member, true)),
                _ => false,
            }
        }
        self.calls.iter().any(|(_, arguments)| {
            serde_json::from_str::<serde_json::Value>(arguments).is_ok_and(|value| {
                value
                    .as_object()
                    .is_some_and(|arguments| arguments.values().any(|v| quoted(v, false)))
            })
        })
    }

    /// Whether `other` says the same, the calls' arguments compared as JSON values: a template that
    /// re-serializes the arguments (Inkling's `tojson(sort_keys=true, separators=(",", ":"))`)
    /// writes the keys sorted and the spaces dropped, and the parser hands on what the model
    /// wrote, so the bytes differ from the reference's while the values are the same.
    fn same_values(&self, other: &Self) -> bool {
        let value = |text: &str| serde_json::from_str::<serde_json::Value>(text).ok();
        self.content == other.content
            && self.reasoning == other.reasoning
            && self.finish == other.finish
            && self.calls.len() == other.calls.len()
            && self.calls.iter().zip(&other.calls).all(
                |((name, args), (other_name, other_args))| {
                    name == other_name && value(args).is_some() && value(args) == value(other_args)
                },
            )
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
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn qwen3_parse_fixtures_match_the_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let Cases { fixtures, ids } =
        read_fixtures(&root.join(SLUG).join("parse")).unwrap_or_else(|e| panic!("{e}"));
    let format = for_model(&model_of(SLUG)).expect("the registry knows the slug");
    assert!(
        !fixtures.is_empty(),
        "no parse fixtures under {}",
        root.display()
    );
    let Parity {
        report,
        summary,
        failures,
    } = parity(
        &fixtures,
        &ids,
        &|fixture| Engine::new(format.clone(), Declared::of(&fixture.request.tools)),
        &|_| String::new(),
        Rules {
            known_differences: KNOWN_DIFFERENCES,
            allowed: &[],
            think_markers: Table::Qwen3.think_markers(),
            respells_arguments: false,
            quotes_values: false,
            holds_a_list_after_content: false,
        },
    );
    let verbose = verbose();
    let mut file = ReportFile::create("bellwether_parse_fixtures-qwen3-8b.txt")
        .unwrap_or_else(|e| panic!("{e}"));
    file.write(&report);
    print!("{}", if verbose { &report } else { &summary });
    let path = file.finish(&failures).unwrap_or_else(|e| panic!("{e}"));
    println!("per-case report: {}", path.display());
    assert!(
        failures.is_empty(),
        "\n{}",
        message(&failures, verbose, &path)
    );
}

/// Whether [`VERBOSE_ENV`] asks for the per-case report on stdout: set to anything but empty or
/// `0`.
fn verbose() -> bool {
    std::env::var(VERBOSE_ENV).is_ok_and(|value| !value.is_empty() && value != "0")
}

/// A per-case report being written: `name` under [`REPORT_DIR_ENV`] or the build's `target/tmp`.
/// A write that fails is kept and reported when the file is finished, so that the replay's
/// closures need not handle it.
struct ReportFile {
    path: PathBuf,
    file: BufWriter<fs::File>,
    error: Option<std::io::Error>,
}

impl ReportFile {
    /// Create the file, and its directory.
    fn create(name: &str) -> Result<Self, String> {
        let dir = std::env::var_os(REPORT_DIR_ENV)
            .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
        fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let path = dir.join(name);
        let file = fs::File::create(&path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        Ok(Self {
            path,
            file: BufWriter::new(file),
            error: None,
        })
    }

    /// Append `text`.
    fn write(&mut self, text: &str) {
        if let Err(e) = self.file.write_all(text.as_bytes()) {
            self.error.get_or_insert(e);
        }
    }

    /// Append the failures and close the file; its path, or the first error met.
    fn finish(mut self, failures: &[String]) -> Result<PathBuf, String> {
        for failure in failures {
            self.write(failure);
            self.write("\n");
        }
        if let Err(e) = self.file.flush() {
            self.error.get_or_insert(e);
        }
        match self.error {
            Some(e) => Err(format!("cannot write {}: {e}", self.path.display())),
            None => Ok(self.path),
        }
    }
}

/// The run's failure message: every failure when the run is verbose, else [`capped`].
fn message(failures: &[String], verbose: bool, report: &std::path::Path) -> String {
    if verbose {
        failures.join("\n")
    } else {
        capped(failures, report)
    }
}

/// The first [`SHOWN`] of `failures`; the rest are counted, and the reader sent to the per-case
/// report at `report`.
fn capped(failures: &[String], report: &std::path::Path) -> String {
    if failures.len() <= SHOWN {
        return failures.join("\n");
    }
    format!(
        "{}\n... and {} more, every one in {}",
        failures[..SHOWN].join("\n"),
        failures.len() - SHOWN,
        report.display()
    )
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notices and the per-case reports are test diagnostic output"
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
    let verbose = verbose();
    let mut file = ReportFile::create("bellwether_parse_fixtures-models.txt")
        .unwrap_or_else(|e| panic!("{e}"));
    in_order(
        MODELS,
        |&(slug, prompt)| replay_model(&root, slug, prompt),
        |replayed| match replayed {
            Replayed::Skipped(notice) => eprintln!("{notice}"),
            Replayed::Compared {
                report,
                summary,
                failures: of_model,
            } => {
                file.write(&report);
                print!("{}", if verbose { &report } else { &summary });
                failures.extend(of_model);
                recorded += 1;
            }
        },
    );
    if recorded == 0 {
        eprintln!(
            "skipping: none of the table's slugs is recorded under {}",
            root.display()
        );
    }
    let path = file.finish(&failures).unwrap_or_else(|e| panic!("{e}"));
    println!("per-case report: {}", path.display());
    assert!(
        failures.is_empty(),
        "\n{}",
        message(&failures, verbose, &path)
    );
}

/// What replaying one of [`MODELS`] gave: the notice for a slug the run leaves out, or the
/// report a serial run would have printed for the model, what stdout shows of it, and the
/// differences nothing allows.
enum Replayed {
    Skipped(String),
    Compared {
        report: String,
        summary: String,
        failures: Vec<String>,
    },
}

/// Replay one of [`MODELS`]: its parse sets under `root`, read as [`read_fixtures`] reads them,
/// through its table after its prompt tail, with its known differences and allowances.
#[expect(
    clippy::panic,
    reason = "a set that cannot be read ends the test with its path"
)]
fn replay_model(root: &std::path::Path, slug: &str, prompt: GenerationPrompt) -> Replayed {
    let dir = root.join(slug).join("parse");
    if !dir.is_dir() {
        return Replayed::Skipped(format!(
            "skipping {slug}: bellwether has not recorded its parse sets yet"
        ));
    }
    if let Some((_, reason)) = SKIPPED.iter().find(|(skipped, _)| *skipped == slug) {
        return Replayed::Skipped(format!("skipping {slug}: {reason}"));
    }
    let Cases { fixtures, ids } = read_fixtures(&dir).unwrap_or_else(|e| panic!("{e}"));
    let model = model_of(slug);
    let table = table_for(&model).unwrap_or_else(|| panic!("{slug}: not in the registry"));
    let format = for_model(&model).unwrap_or_else(|| panic!("{slug}: not in the registry"));
    let known_differences = table.known_differences(slug, prompt);
    let allowed = table.allowances(slug, prompt);
    let Parity {
        report,
        summary,
        failures,
    } = parity(
        &fixtures,
        &ids,
        &|fixture| Engine::new(format.clone(), Declared::of(&fixture.request.tools)),
        &|fixture| prompt.tail(fixture, table),
        Rules {
            known_differences: &known_differences,
            allowed: &allowed,
            think_markers: table.think_markers(),
            respells_arguments: table.respells_arguments(),
            quotes_values: table.quotes_values(),
            holds_a_list_after_content: table.holds_a_list_after_content(),
        },
    );
    Replayed::Compared {
        report: format!("{slug} ({}, {prompt:?}):\n{report}", table.name()),
        summary: format!("{slug} ({}, {prompt:?}):\n{summary}", table.name()),
        failures,
    }
}

/// Run `each` on every item, on as many threads as the run has CPUs, each thread taking the
/// next item from a shared index, and hand what it gives to `then` in the items' order, each
/// as soon as it and the ones before it are done. A panic in `each` ends the run once the items
/// in flight are done, as it would on one thread: the thread that panics raises a flag as it
/// unwinds, and no thread takes an item once it is up.
fn in_order<I: Sync, T: Send>(items: &[I], each: impl Fn(&I) -> T + Sync, then: impl FnMut(T)) {
    let threads = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(items.len())
        .max(1);
    in_order_on(threads, &AtomicBool::new(false), items, each, then);
}

/// [`in_order`] on `threads` threads, with `stop` as the flag the thread that panics raises as it
/// unwinds; no thread takes an item once it is up.
fn in_order_on<I: Sync, T: Send>(
    threads: usize,
    stop: &AtomicBool,
    items: &[I],
    each: impl Fn(&I) -> T + Sync,
    mut then: impl FnMut(T),
) {
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, each) = (&next, &each);
            scope.spawn(move || {
                let _raised_on_panic = StopOnPanic(stop);
                while !stop.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    if tx.send((index, each(item))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut done = 0;
        for (index, outcome) in rx {
            pending.insert(index, outcome);
            while let Some(outcome) = pending.remove(&done) {
                then(outcome);
                done += 1;
            }
        }
    });
}

/// Raises the flag it holds when its thread unwinds from a panic, so that the other threads of
/// [`in_order_on`] take no further item.
struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

/// What [`parity`] gave for a set: the report, one line per case and the set's summary, as a
/// serial run would have printed it; what stdout shows of it, the first differing cases that
/// nothing allows and the set's summary; and every difference that nothing allows.
struct Parity {
    report: String,
    summary: String,
    failures: Vec<String>,
}

/// The first differing cases of a set, as stdout shows them.
#[derive(Default)]
struct Shown {
    lines: String,
    differing: usize,
}

impl Shown {
    /// Count one differing case, and keep its line while fewer than [`SHOWN`] are kept.
    fn differs(&mut self, line: &str) {
        self.differing += 1;
        if self.differing <= SHOWN {
            self.lines.push_str(line);
        }
    }

    /// The lines kept, and how many more the per-case report has.
    fn text(mut self) -> String {
        if self.differing > SHOWN {
            self.lines.push_str(&format!(
                "  ... and {} more differing cases, every one in the per-case report\n",
                self.differing - SHOWN
            ));
        }
        self.lines
    }
}

/// What a family declares to the judge: the cases whose differences it lists, the classes it
/// claims for the corpus, its thinking markers, and whether its template respells the arguments.
#[derive(Clone, Copy)]
struct Rules<'a> {
    known_differences: &'a [KnownDifference],
    allowed: &'a [Allowance],
    think_markers: (&'a str, &'a str),
    /// [`Recorded::respells_arguments`].
    respells_arguments: bool,
    /// [`Recorded::quotes_values`].
    quotes_values: bool,
    /// [`Recorded::holds_a_list_after_content`].
    holds_a_list_after_content: bool,
}

/// Replays every fixture through a fresh parser from `new_parser`, after a prompt ending in the
/// case's `prompt_tail`, on every chunking, reports one line per case and the set's summary, and
/// returns with them every difference the family's `rules` do not account for. `every_id` is the
/// id of every case the set files hold, before any sample thinned them: a listed difference no
/// longer among them is a failure too, so the list cannot rot.
#[expect(
    clippy::panic,
    reason = "a fixture that cannot be replayed ends the test with its id"
)]
fn parity(
    fixtures: &[Fixture],
    every_id: &[String],
    new_parser: &dyn Fn(&Fixture) -> Engine,
    prompt_tail: &dyn Fn(&Fixture) -> String,
    rules: Rules<'_>,
) -> Parity {
    let Rules {
        known_differences,
        allowed,
        think_markers,
        respells_arguments,
        quotes_values,
        holds_a_list_after_content,
    } = rules;
    let known = |id: &str| {
        known_differences
            .iter()
            .find(|known| known.id == after_slug(id))
    };
    let mut failures = Vec::new();
    let mut out = String::new();
    let mut shown = Shown::default();
    let (mut bitwise, mut separators_only, mut respelled, mut listed, mut allowed_count) =
        (0, 0, 0, 0, 0);
    let (mut unreadable, mut undeclared_after_content) = (0, 0);
    let (mut token_plans_run, mut without_pieces, mut content_off_whole) = (0, 0, 0);
    for fixture in fixtures {
        let text = &*fixture.reference.text;
        let expected = Said::of_reference(&fixture.reference);
        let finish = engine_finish(&fixture.reference.finish_reason);
        // The whole replay comes first; every other replay is checked against it as it is made,
        // so a case holds one `Said` at a time however many plans it has.
        let plans = chunkings(text, EVERY_SPLIT_UP_TO);
        let replay_text = |cuts: &[usize], plan: &str| {
            replay_after(
                &mut new_parser(fixture),
                &prompt_tail(fixture),
                text,
                cuts,
                &finish,
                false,
            )
            .unwrap_or_else(|e| panic!("{}: {plan} replay: {e}", fixture.id))
        };
        let said = checked(
            fixture,
            "whole",
            &replay_text(&plans[0].1, "whole"),
            &mut failures,
        );
        // What a client of the stream receives as content is every content delta, one per
        // content or malformed event; the whole answer's content is the message's, absent when
        // it is only whitespace. A streamed replay whose deltas carry other content than the
        // whole answer (the template's spacing before a call, shown before the call arrived)
        // is counted once per case, with the first plan that showed it.
        let whole_content = said.content.clone().unwrap_or_default();
        let mut off_whole: Option<String> = None;
        let mut streamed = |plan: &str, events: &[Event], failures: &mut Vec<String>| {
            if checked(fixture, plan, events, failures) != said {
                failures.push(format!(
                    "{}: the {plan} replay disagrees with the whole one",
                    fixture.id
                ));
            }
            if off_whole.is_none() && delivered_content(events) != whole_content {
                off_whole = Some(plan.to_string());
            }
        };
        for (plan, cuts) in plans.iter().skip(1) {
            streamed(plan, &replay_text(cuts, plan), &mut failures);
        }
        // The token plans need the pieces bellwether records with the case: one per id, giving
        // back the text. A case that lacks them is counted, and one whose pieces are wrong is a
        // failure of the fixture, not of the parser.
        let pieces = match fixture.output_pieces.as_deref() {
            None => {
                without_pieces += 1;
                None
            }
            Some(pieces) if pieces.len() != fixture.output_ids.len() => {
                failures.push(format!(
                    "{}: {} output_pieces for {} output_ids",
                    fixture.id,
                    pieces.len(),
                    fixture.output_ids.len()
                ));
                None
            }
            Some(pieces) if pieces.concat() != text => {
                failures.push(format!(
                    "{}: the output_pieces do not give back the text",
                    fixture.id
                ));
                None
            }
            Some(pieces) => Some(pieces),
        };
        if let Some(pieces) = pieces {
            for (name, sizes) in token_plans(fixture) {
                token_plans_run += 1;
                let plan = format!("token plan {name}");
                let events = replay_tokens(
                    &mut new_parser(fixture),
                    Input::Prompt {
                        token_ids: &[],
                        text: &prompt_tail(fixture),
                    },
                    &fixture.output_ids,
                    pieces,
                    &sizes,
                    &finish,
                )
                .unwrap_or_else(|e| panic!("{}: {plan} replay: {e}", fixture.id));
                streamed(&plan, &events, &mut failures);
            }
        }
        if let Some(plan) = off_whole {
            content_off_whole += 1;
            failures.push(format!(
                "{}: the {plan} replay's content deltas differ from the whole answer's content",
                fixture.id
            ));
        }
        let verdict = if said == expected {
            bitwise += 1;
            "bitwise"
        } else if said.trimmed() == expected.trimmed() {
            separators_only += 1;
            "separators only (bellwether #17)"
        } else if respells_arguments && said.trimmed().same_values(&expected.trimmed()) {
            respelled += 1;
            "respelled arguments (the template re-serializes them)"
        } else if quotes_values && expected.holds_a_quote_in_a_string() {
            // The template wrote the quote unescaped (the value's own, or repr's around a nested
            // element with an apostrophe), so where the string ends is anyone's guess and what
            // follows it may land anywhere: the case is counted apart as a whole.
            unreadable += 1;
            "unreadable: a quote inside a quoted value"
        } else if holds_a_list_after_content
            && said.calls.is_empty()
            && expected.content.is_some()
            && expected.calls.iter().any(|(name, _)| {
                !fixture
                    .request
                    .tools
                    .iter()
                    .any(|tool| tool.function.name == *name)
            })
        {
            undeclared_after_content += 1;
            "undeclared call after content (read as content)"
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
                Allowance::ContentNotWritten => "allowed: content not written (corpus)",
                Allowance::NullNotWritten => "allowed: null not written (corpus)",
                Allowance::ContentTrimmed => "allowed: content trimmed (corpus)",
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
        let line = format!("  {verdict:34} {}\n", fixture.id);
        if verdict == "DIFFERS" {
            shown.differs(&line);
        }
        out.push_str(&line);
        if let Some(listed_case) = known(&fixture.id) {
            out.push_str(&format!("  {:34} {}\n", "", listed_case.reason));
        }
    }
    // A listed case that is no longer among the fixtures would let the list rot. The check reads
    // every id the set files hold, before the sample thinned them, so it holds for a sampled run.
    for listed_case in known_differences {
        if !every_id.iter().any(|id| after_slug(id) == listed_case.id) {
            failures.push(format!(
                "{}: listed in KNOWN_DIFFERENCES but not among the fixtures; remove it",
                listed_case.id
            ));
        }
    }
    let sample = sample_note(sample_every());
    // The unreadable class is named only for a family whose template writes it, so every other
    // family's totals line reads as before.
    let unreadable = if quotes_values {
        format!(", {unreadable} unreadable (a quote inside a quoted value)")
    } else {
        String::new()
    };
    let undeclared = if holds_a_list_after_content {
        format!(
            ", {undeclared_after_content} undeclared calls after content (read as content, as the \
             family's reader does)"
        )
    } else {
        String::new()
    };
    let totals = format!(
        "{} cases{sample}: {bitwise} bitwise, {separators_only} separators only, {respelled} \
         respelled, {listed} listed, {allowed_count} allowed for the corpus{unreadable}{undeclared}; \
         {token_plans_run} token plans replayed, {without_pieces} cases without output_pieces, \
         {content_off_whole} cases whose streamed content differs from the whole answer's\n",
        fixtures.len()
    );
    out.push_str(&totals);
    Parity {
        report: out,
        summary: shown.text() + &totals,
        failures,
    }
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
        Allowance::ContentNotWritten => {
            said.content.is_none()
                && expected.content.is_some()
                && text.trim_start().starts_with('[')
                && Said {
                    content: None,
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
        Allowance::ContentTrimmed => {
            let trimmed = expected.content.as_deref().map(str::trim);
            // The output must not carry the untrimmed content either: then the parser lost the
            // whitespace, not the template.
            let template_trimmed = expected
                .content
                .as_deref()
                .is_some_and(|content| !text.contains(content));
            said.content != expected.content
                && said.content.as_deref() == trimmed
                && template_trimmed
                && Said {
                    content: said.content.clone(),
                    ..expected.clone()
                } == said
        }
        Allowance::NullNotWritten => {
            let types_too = allowed.contains(&Allowance::DeclaredTypeConflict);
            said.content == expected.content
                && said.reasoning == expected.reasoning
                && said.finish == expected.finish
                && said.calls.len() == expected.calls.len()
                && said.calls.iter().zip(&expected.calls).all(
                    |((name, arguments), (expected_name, expected_arguments))| {
                        name == expected_name
                            && differs_at_most_in_null_members(
                                arguments,
                                expected_arguments,
                                text,
                                types_too,
                            )
                    },
                )
        }
    })
}

#[test]
fn a_content_the_template_trimmed_is_allowed_and_nothing_else_is() {
    let markers = ("<think>", "</think>");
    let allowed = [Allowance::ContentTrimmed];
    let expected = Said {
        content: Some("Done. ".to_string()),
        reasoning: None,
        calls: Vec::new(),
        finish: "stop".to_string(),
    };
    let trimmed = Said {
        content: Some("Done.".to_string()),
        ..expected.clone()
    };
    assert!(
        allowance_for(&trimmed, &expected, "Done.", &allowed, markers)
            == Some(Allowance::ContentTrimmed)
    );
    // A byte lost inside the content is not trimming, and neither is a call beside it.
    let lost = Said {
        content: Some("Done".to_string()),
        ..expected.clone()
    };
    assert!(allowance_for(&lost, &expected, "Done", &allowed, markers).is_none());
    let with_call = Said {
        calls: vec![("f".to_string(), "{}".to_string())],
        ..trimmed.clone()
    };
    assert!(allowance_for(&with_call, &expected, "Done.", &allowed, markers).is_none());
    // Whitespace the output carries was the parser's to keep, not the template's to trim.
    assert!(allowance_for(&trimmed, &expected, "Done. ", &allowed, markers).is_none());
    // Without the allowance, the trimmed content is a difference.
    assert!(allowance_for(&trimmed, &expected, "Done.", &[], markers).is_none());
}

/// Whether `said` is `expected` less the members whose value is null, none of which the output
/// `text` writes an element for; with `types_too`, a member that differs only in its declared
/// type's spelling passes as well.
fn differs_at_most_in_null_members(
    said: &str,
    expected: &str,
    text: &str,
    types_too: bool,
) -> bool {
    let (Ok(said), Ok(expected)) = (
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(said),
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(expected),
    ) else {
        return false;
    };
    if !said.keys().all(|key| expected.contains_key(key)) {
        return false;
    }
    for (key, reference) in &expected {
        match said.get(key) {
            Some(value) if value == reference => {}
            Some(serde_json::Value::String(spelled)) if types_too && !reference.is_string() => {
                if ![reference.to_string(), python_spelling(reference)].contains(spelled) {
                    return false;
                }
            }
            Some(_) => return false,
            None => {
                if !reference.is_null() || text.contains(&format!("<{key}>")) {
                    return false;
                }
            }
        }
    }
    true
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
    let Cases { fixtures, .. } =
        read_fixtures(&root.join(SLUG).join("parse")).unwrap_or_else(|e| panic!("{e}"));
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
        let text = &*fixture.reference.text;
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
                &mut Engine::new(Table::Qwen3.format(), Declared::default()),
                text,
                &[],
                &finish,
                false,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.id)),
        );
        for (name, sizes) in token_plans(fixture) {
            let place = format!("{} plan {name}", fixture.id);
            let events = replay_tokens(
                &mut Engine::new(Table::Qwen3.format(), Declared::default()),
                prompt(),
                &fixture.output_ids,
                pieces,
                &sizes,
                &finish,
            )
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
        "{} cases{}, {plans_run} token plans, {tokens_counted} tokens counted, \
         {reasoning_counted} of them reasoning",
        fixtures.len(),
        sample_note(sample_every())
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

/// One replay's events checked: every output byte in exactly one event, and the calls indexed
/// and named in order; what it said, for the comparison with the whole replay.
fn checked(fixture: &Fixture, plan: &str, events: &[Event], failures: &mut Vec<String>) -> Said {
    let conserved: String = events.iter().map(bytes_of).collect();
    if conserved != fixture.reference.text {
        failures.push(format!(
            "{}: bytes not conserved on the {plan} replay",
            fixture.id
        ));
    }
    let mut index = 0;
    let mut first = None;
    for event in events {
        if let Event::ToolCallStart { index: i, id, .. } = event {
            // The ids count from a nonce of the engine's own, `call_` and 24 hex characters.
            let number = common::minted_id(id);
            let origin = *first.get_or_insert(number);
            if *i != index || number.is_none() || number != origin.map(|f| f + u128::from(index)) {
                failures.push(format!(
                    "{}: call index {i} / id {id} on the {plan} replay",
                    fixture.id
                ));
            }
            index += 1;
        }
    }
    Said::of(events)
}

/// The content a client of the stream receives: the text of every content and malformed event in
/// order, as the gateway writes one content delta per event.
fn delivered_content(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
            _ => None,
        })
        .collect()
}

/// Feed the output token by token as the plan `sizes` groups them, each delta carrying its tokens'
/// ids and the span of each token's piece, after `prompt`, then the end with the engine's
/// `finish`.
fn replay_tokens(
    parser: &mut dyn Parser,
    prompt: Input<'_>,
    ids: &[u32],
    pieces: &[String],
    sizes: &[usize],
    finish: &EngineFinish,
) -> Result<Vec<Event<'static>>, ParseError> {
    // Each step's text and spans are built first: a step's events borrow its text for as long as
    // the buffer they are read into lives, here the whole output.
    let mut steps: Vec<(usize, usize, String, Vec<TokenSpan>)> = Vec::with_capacity(sizes.len());
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
        steps.push((first, last, text, spans));
        first = last;
    }
    let mut out = Events::new();
    parser.feed(prompt, &mut out)?;
    for (first, last, text, spans) in &steps {
        parser.feed(
            Input::Delta {
                token_ids: &ids[*first..*last],
                text,
                spans,
            },
            &mut out,
        )?;
    }
    parser.feed(
        Input::End {
            finish: finish.clone(),
        },
        &mut out,
    )?;
    Ok(out.into_owned().drain())
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

/// The sample's step from [`SAMPLE_ENV`]: 1 when it is unset or empty, as the allowances switch
/// reads an empty value as off. A value that is not a positive integer is a mistake in the run's
/// setup, not a smaller sample.
fn sample_every() -> usize {
    sample_step(std::env::var(SAMPLE_ENV).ok().as_deref())
}

/// [`sample_every`]'s reading of the variable's value.
fn sample_step(value: Option<&str>) -> usize {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return 1;
    };
    let every: usize = value.parse().unwrap_or(0);
    assert!(
        every > 0,
        "{SAMPLE_ENV}={value:?} is not a positive integer"
    );
    every
}

/// What a run's summary says of its sample: nothing for a whole run.
fn sample_note(every: usize) -> String {
    match every {
        1 => String::new(),
        every => {
            format!(
                " (1 in {every} cases of each recorded set, the hand-written sets whole, {SAMPLE_ENV})"
            )
        }
    }
}

/// The step for one set's file: the hand-written sets ([`HAND_WRITTEN_SETS`]) are never thinned.
fn step_for(file: &std::path::Path, every: usize) -> usize {
    if file
        .file_stem()
        .is_some_and(|stem| HAND_WRITTEN_SETS.iter().any(|set| stem == *set))
    {
        1
    } else {
        every
    }
}

/// Whether the case at `index` of its set is in the sample: every `every`-th from the first, the
/// same slice of a set every run.
fn kept(index: usize, every: usize) -> bool {
    index.is_multiple_of(every.max(1))
}

/// The cases a slug's set files gave.
struct Cases {
    /// The cases to replay, in the files' order, each recorded set sampled by [`SAMPLE_ENV`] and
    /// the [`HAND_WRITTEN_SETS`] whole.
    fixtures: Vec<Fixture>,
    /// Every case's id, in the same order, before the sample thinned them: what the check that a
    /// listed difference is still among the cases reads, so that a sample never fails it.
    ids: Vec<String>,
}

/// Reads every `.jsonl` set under `dir`, in the files' order, into [`Cases`].
fn read_fixtures(dir: &std::path::Path) -> Result<Cases, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    let every = sample_every();
    let mut fixtures = Vec::new();
    let mut ids = Vec::new();
    for file in files {
        let step = step_for(&file, every);
        let reader = BufReader::new(
            fs::File::open(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?,
        );
        // The set is read a line at a time, every case parsed as before, and a case outside the
        // sample dropped as it is read, so the run holds the sampled cases and one line of a
        // set, however many models are in flight.
        let mut index = 0;
        for (number, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let fixture: Fixture = serde_json::from_str(&line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            ids.push(fixture.id.clone());
            if kept(index, step) {
                fixtures.push(fixture);
            }
            index += 1;
        }
    }
    Ok(Cases { fixtures, ids })
}

/// A seeded generator for the corruptions, the same sequence for a seed every run.
struct Scramble(u64);

impl Scramble {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A number below `n`; zero when `n` is.
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() as usize) % n
        }
    }
}

/// The seed of a case's corruption round: a hash of the case's id and the round, so that a case is
/// corrupted the same way every run, whatever the sample around it.
fn corruption_seed(id: &str, round: u64) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// The pieces a corruption puts into a recorded output: the table's own terminals and the tags
/// of its call syntax, each whole and cut short at either end and in the middle, and the
/// delimiters every syntax has a use for.
fn markers_of(format: &symphony::Format) -> Vec<String> {
    let mut markers: Vec<String> = format.terminal_texts().map(str::to_string).collect();
    if let Some(syntax) = format.call_syntax() {
        markers.extend(syntax.tag_texts().into_iter().map(str::to_string));
    }
    let mut pieces = markers.clone();
    for marker in &markers {
        let chars: Vec<char> = marker.chars().collect();
        if chars.len() >= 2 {
            pieces.push(chars[..chars.len() / 2].iter().collect());
            pieces.push(chars[..chars.len() - 1].iter().collect());
            pieces.push(chars[1..].iter().collect());
        }
    }
    pieces.extend(
        [
            "{", "}", "[", "]", "\"", "\\", ",", ":", "<", ">", "/", "\n", " ",
        ]
        .iter()
        .map(|piece| piece.to_string()),
    );
    pieces
}

/// `recorded` with one to three seeded corruptions, each on character boundaries: a span deleted,
/// a span doubled, a piece inserted, the tail cut, or a span swapped for one character. Every
/// corruption changes the text it is given and none gives the recorded text back (a draw that
/// would, an empty span, a cut at the end, a round that undoes the one before, is drawn again,
/// and an empty text takes an insertion), so a counted output is never the recorded one.
fn corrupted(random: &mut Scramble, recorded: &str, pieces: &[String]) -> String {
    let mut text = recorded.to_string();
    for _ in 0..1 + random.below(3) {
        let before = std::mem::take(&mut text);
        text = (0..8)
            .map(|_| corrupted_once(random, &before, pieces))
            .find(|after| *after != before && after != recorded)
            .unwrap_or_else(|| inserted(random, &before, 0, pieces));
    }
    if text == recorded {
        // The fallback insertion undid the round before it: one more
        // insertion (a piece is never empty) leaves the recorded text for good.
        text = inserted(random, &text, 0, pieces);
    }
    text
}

/// `text` with `piece`, drawn from `pieces`, inserted at the character boundary `at`.
fn inserted(random: &mut Scramble, text: &str, at: usize, pieces: &[String]) -> String {
    format!(
        "{}{}{}",
        &text[..at],
        pieces[random.below(pieces.len())],
        &text[at..]
    )
}

/// One seeded corruption of `text`; it may draw a span that changes nothing (see [`corrupted`]).
fn corrupted_once(random: &mut Scramble, text: &str, pieces: &[String]) -> String {
    if text.is_empty() {
        return inserted(random, text, 0, pieces);
    }
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(at, _)| at)
        .chain(std::iter::once(text.len()))
        .collect();
    let (a, b) = (
        bounds[random.below(bounds.len())],
        bounds[random.below(bounds.len())],
    );
    let (from, to) = (a.min(b), a.max(b));
    match random.below(5) {
        0 => format!("{}{}", &text[..from], &text[to..]),
        1 => format!("{}{}{}", &text[..to], &text[from..to], &text[to..]),
        2 => inserted(random, text, from, pieces),
        3 => text[..to].to_string(),
        _ => {
            let swapped = ['<', '>', '"', '\n', ' ', 'x', '}', ']', 'é', '🌧'];
            format!(
                "{}{}{}",
                &text[..from],
                swapped[random.below(swapped.len())],
                &text[to..]
            )
        }
    }
}

/// A seeded plan of cuts into `text`, each piece one to eight characters.
fn seeded_cuts(random: &mut Scramble, text: &str) -> Vec<usize> {
    let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    let mut cuts = Vec::new();
    let mut at = 0;
    while at < starts.len() {
        at += 1 + random.below(8);
        if at < starts.len() {
            cuts.push(starts[at]);
        }
    }
    cuts
}

/// Whether the events carry every byte of `text` in order, with exactly one `Finish`, last.
fn conserved(text: &str, events: &[Event]) -> Result<(), String> {
    let carried: String = events.iter().map(bytes_of).collect();
    if carried != text {
        return Err(format!("the events carry {carried:?}"));
    }
    let finishes = events
        .iter()
        .filter(|event| matches!(event, Event::Finish { .. }))
        .count();
    if finishes != 1 || !matches!(events.last(), Some(Event::Finish { .. })) {
        return Err(format!(
            "{finishes} Finish events, where one must come last"
        ));
    }
    Ok(())
}

/// One replay of `text` cut at `cuts`, with a panic inside the parser caught and named by the
/// plan, so a breach says which replay broke.
fn replay_plan(
    name: &str,
    new_engine: &dyn Fn() -> Engine,
    tail: &str,
    text: &str,
    cuts: &[usize],
    finish: &EngineFinish,
) -> Result<Vec<Event<'static>>, String> {
    let replayed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        replay_after(&mut new_engine(), tail, text, cuts, finish, false)
    }));
    match replayed {
        Ok(Ok(events)) => Ok(events),
        Ok(Err(e)) => Err(format!("{name}: {e}")),
        Err(payload) => Err(format!("{name}: PANIC: {}", panic_message(payload))),
    }
}

/// One corrupted output through a fresh engine from `new_engine`, after the prompt's `tail`:
/// whole, on two seeded plans and, up to 512 bytes, a character at a time; the first breach of
/// conservation, or of the message the chat adapter folds from the events, if any, named by its
/// plan.
fn check_corrupted(
    new_engine: &dyn Fn() -> Engine,
    tail: &str,
    text: &str,
    finish: &EngineFinish,
    random: &mut Scramble,
) -> Result<(), String> {
    let whole = replay_plan("whole", new_engine, tail, text, &[], finish)?;
    conserved(text, &whole).map_err(|e| format!("whole: {e}"))?;
    let expected = Said::of(&whole);
    let mut plans = vec![
        ("seeded plan 1", seeded_cuts(random, text)),
        ("seeded plan 2", seeded_cuts(random, text)),
    ];
    if text.len() <= 512 {
        plans.push((
            "character by character",
            text.char_indices().map(|(at, _)| at).skip(1).collect(),
        ));
    }
    for (name, cuts) in plans {
        let events = replay_plan(name, new_engine, tail, text, &cuts, finish)?;
        // The plan is named, not spelled out: the seeded plans are rebuilt from the case's id and
        // round, and the character plan from the text.
        conserved(text, &events).map_err(|e| format!("{name}: {e}"))?;
        let said = Said::of(&events);
        if said != expected {
            return Err(format!(
                "{name} says something else:\n    whole: {expected:?}\n    cut:   {said:?}"
            ));
        }
    }
    Ok(())
}

/// The first `chars` characters of `text`, with a note of how much was left out; the report file
/// has it whole.
fn window(text: &str, chars: usize) -> String {
    match text.char_indices().nth(chars) {
        Some((at, _)) => format!(
            "{}... ({} bytes in all; the report file has it whole)",
            &text[..at],
            text.len()
        ),
        None => text.to_string(),
    }
}

/// The message of a panic, as `panic!` formatted it.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic without a message".to_string())
}

/// What corrupting the recorded outputs of one of [`MODELS`] gave.
enum Corrupted {
    Skipped(String),
    Done {
        slug: &'static str,
        cases: usize,
        outputs: usize,
        failures: Vec<String>,
    },
}

/// Three seeded corruptions of every sampled output of `slug`'s parse sets, each replayed through
/// the model's table after its prompt tail ([`check_corrupted`]), a panic caught per output.
#[expect(
    clippy::panic,
    reason = "a set that cannot be read ends the test with its path"
)]
fn corrupt_model(
    root: &std::path::Path,
    slug: &'static str,
    prompt: GenerationPrompt,
) -> Corrupted {
    let dir = root.join(slug).join("parse");
    if !dir.is_dir() {
        return Corrupted::Skipped(format!(
            "skipping {slug}: bellwether has not recorded its parse sets yet"
        ));
    }
    let Cases { fixtures, .. } = read_fixtures(&dir).unwrap_or_else(|e| panic!("{e}"));
    let model = model_of(slug);
    let table = table_for(&model).unwrap_or_else(|| panic!("{slug}: not in the registry"));
    let format = for_model(&model).unwrap_or_else(|| panic!("{slug}: not in the registry"));
    let pieces = markers_of(&format);
    let mut failures = Vec::new();
    let mut outputs = 0;
    for fixture in &fixtures {
        let tail = prompt.tail(fixture, table);
        let finish = engine_finish(&fixture.reference.finish_reason);
        for round in 0..3u64 {
            let mut random = Scramble::new(corruption_seed(&fixture.id, round));
            let text = corrupted(&mut random, &fixture.reference.text, &pieces);
            outputs += 1;
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_corrupted(
                    &|| Engine::new(format.clone(), Declared::of(&fixture.request.tools)),
                    &tail,
                    &text,
                    &finish,
                    &mut random,
                )
            }));
            let failure = match outcome {
                Ok(Ok(())) => continue,
                Ok(Err(breach)) => breach,
                Err(payload) => format!("PANIC: {}", panic_message(payload)),
            };
            // The breach and its plan first, the output last: a window on the
            // message keeps what names the failure.
            failures.push(format!(
                "{} round {round} ({}, prompt tail {tail:?}, finish {finish:?}):\n  \
                 {failure}\n  corrupted output {text:?}",
                fixture.id,
                table.name()
            ));
        }
    }
    Corrupted::Done {
        slug,
        cases: fixtures.len(),
        outputs,
        failures,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notices and the per-model counts are test diagnostic output"
)]
fn corrupted_recorded_outputs_never_panic_and_keep_the_contract() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let (mut cases, mut outputs, mut recorded) = (0, 0, 0);
    let mut failures = Vec::new();
    let mut file = ReportFile::create("bellwether_parse_fixtures-corrupted.txt")
        .unwrap_or_else(|e| panic!("{e}"));
    in_order(
        MODELS,
        |&(slug, prompt)| corrupt_model(&root, slug, prompt),
        |outcome| match outcome {
            Corrupted::Skipped(notice) => eprintln!("{notice}"),
            Corrupted::Done {
                slug,
                cases: of_model,
                outputs: of_outputs,
                failures: of_model_failures,
            } => {
                let line = format!(
                    "{slug}: {of_outputs} corrupted outputs of {of_model} cases, {} breaches\n",
                    of_model_failures.len()
                );
                print!("{line}");
                file.write(&line);
                cases += of_model;
                outputs += of_outputs;
                recorded += 1;
                failures.extend(of_model_failures);
            }
        },
    );
    if recorded == 0 {
        eprintln!(
            "skipping: none of the table's slugs is recorded under {}",
            root.display()
        );
    }
    println!(
        "{outputs} corrupted outputs of {cases} recorded cases of {recorded} models replayed{}",
        sample_note(sample_every())
    );
    let path = file.finish(&failures).unwrap_or_else(|e| panic!("{e}"));
    println!("per-model report and every breach: {}", path.display());
    assert!(
        failures.is_empty(),
        "{} corrupted outputs broke the contract or panicked; the first {} (every one in {}):\n{}",
        failures.len(),
        failures.len().min(SHOWN),
        path.display(),
        failures
            .iter()
            .take(SHOWN)
            .map(|failure| window(failure, 2_048))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn a_template_that_opens_the_thought_only_when_asked_closes_it_otherwise() {
    let request = |kwargs: &str| -> Request {
        serde_json::from_str(&format!("{{\"chat_template_kwargs\": {kwargs}}}")).expect("a request")
    };
    let when_asked = GenerationPrompt::OpensTheThoughtWhenAsked;
    assert_eq!(
        when_asked.tail_for(&request("{\"enable_thinking\": true}"), Table::Qwen3Tagged),
        "<think>\n"
    );
    for unasked in ["{\"enable_thinking\": false}", "{}", "null"] {
        assert_eq!(
            when_asked.tail_for(&request(unasked), Table::Qwen3Tagged),
            "<think>\n\n</think>\n\n",
            "{unasked}"
        );
    }
    // The default-on variant reads the same requests the other way round.
    assert_eq!(
        GenerationPrompt::OpensTheThought.tail_for(&request("null"), Table::Qwen3Tagged),
        "<think>\n"
    );
    // A slug whose template refused a probe does not list it.
    let listed: Vec<&str> = Table::Qwen3Tagged
        .known_differences("qwen3.8-2.4t-a95b", GenerationPrompt::AlwaysOpensTheThought)
        .iter()
        .map(|known| known.id)
        .collect();
    assert_eq!(listed, [REASONING_PROBE.id]);
    assert!(Table::Qwen3Tagged
        .known_differences("qwen3.5-2b", when_asked)
        .iter()
        .all(|known| known.id != REASONING_PROBE.id));
}

/// Spins until `ready` holds, for at most ten seconds, so that a test waiting on another thread's
/// progress fails with `never` rather than hangs.
fn wait_until(ready: impl Fn() -> bool, never: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready() {
        assert!(Instant::now() < deadline, "{never}");
        std::thread::yield_now();
    }
}

#[test]
fn stdout_shows_the_first_differing_cases_and_counts_the_rest() {
    let mut shown = Shown::default();
    for i in 0..SHOWN + 3 {
        shown.differs(&format!("  DIFFERS  m/parse/common-{i}\n"));
    }
    let text = shown.text();
    assert_eq!(text.lines().count(), SHOWN + 1, "{text}");
    assert!(
        text.starts_with("  DIFFERS  m/parse/common-0\n")
            && text
                .ends_with("  ... and 3 more differing cases, every one in the per-case report\n"),
        "{text}"
    );
    let mut few = Shown::default();
    few.differs("  DIFFERS  m/parse/common-0\n");
    assert_eq!(few.text(), "  DIFFERS  m/parse/common-0\n");
    // The run's message lists the first failures the same way.
    let failures: Vec<String> = (0..SHOWN + 2)
        .map(|i| format!("m/parse/common-{i}: differs from the reference"))
        .collect();
    let report = std::path::Path::new("reports/parse.txt");
    let listed = capped(&failures, report);
    assert_eq!(listed.lines().count(), SHOWN + 1, "{listed}");
    assert!(
        listed.starts_with(&failures[0])
            && listed.ends_with("\n... and 2 more, every one in reports/parse.txt"),
        "{listed}"
    );
    assert_eq!(capped(&failures[..2], report), failures[..2].join("\n"));
    // A verbose run's message is whole.
    assert_eq!(message(&failures, true, report), failures.join("\n"));
    assert_eq!(message(&failures, false, report), listed);
}

#[test]
fn the_fan_out_hands_the_items_over_in_their_order_however_they_finish() {
    // Item 0 waits until item 1 is done, so on two threads item 1 finishes first and items 2
    // and 3 may follow it, all before item 0; `then` still sees them in the items' order.
    let item_1_done = AtomicBool::new(false);
    let items = [0, 1, 2, 3];
    let mut handed = Vec::new();
    in_order_on(
        2,
        &AtomicBool::new(false),
        &items,
        |&item| {
            if item == 0 {
                wait_until(
                    || item_1_done.load(Ordering::Relaxed),
                    "item 1 never finished",
                );
            }
            if item == 1 {
                item_1_done.store(true, Ordering::Relaxed);
            }
            item * 10
        },
        |outcome| handed.push(outcome),
    );
    assert_eq!(handed, [0, 10, 20, 30]);
}

#[test]
fn a_panic_in_the_fan_out_leaves_the_other_threads_their_item_in_flight_and_no_more() {
    // On two threads, item 0 panics once the other thread has an item in flight, and that item
    // ends once the panic has raised the flag: the other thread finishes the one item it had,
    // finds the flag up when it looks for the next, and takes none of the six left. Nothing
    // here waits a length of time; the item in flight waits for the flag itself, however long
    // the unwind takes on a loaded host.
    let stop = AtomicBool::new(false);
    let started = AtomicUsize::new(0);
    let items: Vec<usize> = (0..8).collect();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        in_order_on(
            2,
            &stop,
            &items,
            |&item| {
                if item == 0 {
                    wait_until(
                        || started.load(Ordering::Relaxed) >= 1,
                        "the other thread never started an item",
                    );
                    panic!("item 0");
                }
                started.fetch_add(1, Ordering::Relaxed);
                wait_until(
                    || stop.load(Ordering::Relaxed),
                    "item 0's panic never raised the flag",
                );
                item
            },
            |_| {},
        );
    }));
    assert!(outcome.is_err(), "the panic reaches the caller");
    assert!(stop.load(Ordering::Relaxed), "the panic raised the flag");
    assert_eq!(
        started.load(Ordering::Relaxed),
        1,
        "the other thread finishes the item it had in flight and takes no more"
    );
}

#[test]
fn a_sample_keeps_every_nth_case_of_a_set_from_its_first() {
    let sampled = |cases: usize, every| (0..cases).filter(|&i| kept(i, every)).collect::<Vec<_>>();
    assert_eq!(sampled(10, 3), [0, 3, 6, 9]);
    assert_eq!(sampled(4, 1), [0, 1, 2, 3]);
    assert_eq!(sampled(4, 10), [0]);
    assert!(sampled(0, 2).is_empty());
}

#[test]
fn the_sample_step_is_one_when_the_variable_is_unset_or_empty() {
    assert_eq!(sample_step(None), 1);
    assert_eq!(sample_step(Some("")), 1);
    assert_eq!(sample_step(Some("10")), 10);
}

#[test]
fn a_summary_names_its_sample_and_a_whole_run_says_nothing() {
    assert_eq!(sample_note(1), "");
    assert_eq!(
        sample_note(2),
        " (1 in 2 cases of each recorded set, the hand-written sets whole, SYMPHONY_SAMPLE_EVERY)"
    );
}

#[test]
#[should_panic(expected = "is not a positive integer")]
fn a_sample_step_that_is_not_a_positive_integer_is_refused() {
    sample_step(Some("0"));
}

#[test]
fn the_common_set_is_replayed_whole_whatever_the_step() {
    let parse = std::path::Path::new("fixtures/qwen3-8b/parse");
    assert_eq!(step_for(&parse.join("common.jsonl"), 10), 1);
    assert_eq!(step_for(&parse.join("gsm8k-test-content.jsonl"), 10), 10);
}

#[test]
fn a_reference_is_unreadable_between_quotes_by_a_quote_at_any_depth_or_a_nested_apostrophe() {
    let said = |arguments: &str| Said {
        content: None,
        reasoning: None,
        calls: vec![("save_note".to_string(), arguments.to_string())],
        finish: "tool_calls".to_string(),
    };
    // A `"` in a string, at the top or inside a list or an object.
    assert!(said(r#"{"body": "a \"quoted\" word"}"#).holds_a_quote_in_a_string());
    assert!(said(r#"{"tags": ["a", "the \"b\""]}"#).holds_a_quote_in_a_string());
    assert!(said(r#"{"meta": {"deep": {"word": "\""}}}"#).holds_a_quote_in_a_string());
    // An apostrophe inside a list or an object: repr writes `"O'Brien"` between the template's `"`.
    assert!(said(r#"{"tags": ["a", "O'Brien"]}"#).holds_a_quote_in_a_string());
    assert!(said(r#"{"meta": {"name": "O'Brien"}}"#).holds_a_quote_in_a_string());
    // A nested object's key is quoted by repr as its values are.
    assert!(said(r#"{"meta": {"O'Brien": 1}}"#).holds_a_quote_in_a_string());
    assert!(said(r#"{"meta": {"deep": {"a\"b": 1}}}"#).holds_a_quote_in_a_string());
    // A top-level string's apostrophe stands between the template's `"` as it is, and so does a
    // backslash at the value's end; a list without one is repr's own quotes alone.
    assert!(!said(r#"{"city": "O'Brien's"}"#).holds_a_quote_in_a_string());
    assert!(!said(r#"{"command": "dir C:\\"}"#).holds_a_quote_in_a_string());
    assert!(!said(r#"{"tags": ["a", "b"], "days": 3}"#).holds_a_quote_in_a_string());
    assert!(!said("not json").holds_a_quote_in_a_string());
}

#[test]
fn every_skipped_slug_is_in_the_table() {
    for (slug, _) in SKIPPED {
        assert!(
            MODELS.iter().any(|(listed, _)| listed == slug),
            "{slug} is skipped but not in MODELS"
        );
    }
}
