//! Render parity with the bellwether reference fixtures, through the gateway's
//! chat request path.
//!
//! bellwether (smg-project/bellwether) records, for each model at a pinned
//! Hugging Face revision, what the checkpoint's own chat template renders for
//! a corpus of chat requests: the prompt text and its token ids. This test
//! turns each recorded request into the `ChatCompletionRequest` the HTTP layer
//! would hand on (deserialized, normalized and validated as `ValidatedJson`
//! does, tools narrowed by `tool_choice` as the chat preparation stage does),
//! renders it with `process_chat_messages`, the entry point the gateway and the
//! bindings share (content format, typed tools, tool-call arguments parsed for
//! renderers that do not take them raw, template kwargs, the thinking toggle,
//! `continue_final_message`), encodes the text as the gateway's tokenize step
//! does, and compares text and ids with the reference byte for byte.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the tree `bellwether
//! unpack` writes (`fixtures-plain/` unless `--out` names another), where
//! every fixture set is plain JSON Lines beside each model's `manifest.toml`
//! and `sets.toml`; without it the test prints a skip notice and passes.
//! bellwether's own `fixtures/` directory keeps a benchmark set as
//! `<set>.jsonl.zst` in Git LFS, so a compressed set or a Git LFS pointer under
//! the root fails the run and says to unpack first. Each case is compared as
//! its line is read, and the cases read from each render set must number what
//! the model's `sets.toml` lists for it; a render set the table lists that the
//! tree lacks, or a render set file the table does not list, fails the run
//! too, so it cannot pass on part of the fixtures. The run prints each set
//! with its count, each difference, and a tally per model.
//!
//! Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`. A model whose tokenizer
//! does not load is listed when the run fails at the end, after every other
//! model has been compared.
//!
//! A difference is a finding, not something to hide: every known one is listed
//! in [`KNOWN_DIFFERENCES`] with its reason and where it is tracked, by id or
//! by a prefix when a cause covers a model or a set wholesale; the run fails
//! on any other, on a listed case that starts matching, and on a listed case
//! that the loaded fixtures no longer contain, so the list cannot rot. A model
//! whose tokenizer does not load for a known reason is listed in
//! [`KNOWN_UNLOADED`] the same way.
//!
//! What the test cannot see: the public entry point returns the rendered text
//! and not the deferred encode a segment-aware renderer prepares, so for such
//! a renderer the ids compared here are a flat encode of the text, which the
//! gateway itself warns may not reproduce its own ids. The two models recorded
//! so far render to text that is encoded flat.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{BufRead, BufReader, Read},
    num::NonZeroUsize,
    ops::Bound,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer, MockTokenizer};
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;
use smg::routers::grpc::utils::process_chat_messages;
use validator::Validate;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";
/// The fixture kinds this test reads, each a directory of sets beside a
/// model's manifest.
const KINDS: [&str; 1] = ["render"];
/// How a file Git LFS has not fetched begins: the pointer stands where the
/// content would be.
const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// Cases known to differ from the reference, each with the reason and where
/// it is tracked: a fixture id, `<slug>/render/<name>`, or a prefix ending in
/// `*`, which stands for every case whose id begins with what is before it
/// (`<slug>/render/*` for a model's whole render side,
/// `<slug>/render/bfcl-multi-turn-*` for a family of its sets). A prefix is
/// for a cause that covers a model or a set wholesale and is held to the rule
/// an id is: the run fails when no loaded case under it differs any more, and
/// when no loaded case is under it at all.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[
    (
        "apertus-8b-instruct-2509/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "hunyuan-a13b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "iquest-q1/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "iquest-q1/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "iquest-q1/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "k2-horizon-36b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "k2-horizon-36b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "k2-horizon-36b/render/tools-schema-with-defs",
        "the template merges a tool's `$defs` into its parameters with `dict()` over a list of \
         (key, value) pairs, which the gateway's template engine does not accept, so the \
         render fails (smg-project/smg-lab#119)",
    ),
    (
        "laguna-xs.2/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "laguna-xs.2/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "laguna-xs.2/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "laguna-xs.2/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "lfm2.5-1.2b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "lfm2.5-1.2b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "lfm2.5-1.2b-instruct/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "llava-1.5-7b-hf/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "llava-1.5-7b-hf/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "llava-1.5-7b-hf/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "muse-glimmer-30b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "muse-glimmer-30b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "muse-glimmer-30b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "muse-glimmer-30b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-8b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and appending its \
         text after the generation header, which drops the empty think block the template writes \
         before a continued turn (smg-project/smg#2779)",
    ),
    (
        "deepseek-r1/render/continue-final-message",
        "the gateway's generation header adds `<think>\\n` before the continued text; the template \
         writes none for a continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-8b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
    (
        "deepseek-r1/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
    (
        "deepseek-r1/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "deepseek-r1/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "qwen3-8b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the engines do; \
         the Qwen3 template accepts an object (smg-project/bellwether#12, needs:simo)",
    ),
    (
        "qwen3-omni-30b-a3b-instruct/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-omni-30b-a3b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-omni-30b-a3b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-omni-30b-a3b-thinking/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-omni-30b-a3b-thinking/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-omni-30b-a3b-thinking/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "glm-5.3-flash/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "glm-5.3-flash/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "glm-5.3-flash/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "glm-5.3-flash/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "hy4-preview/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "hy4-preview/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "hy4-preview/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "hy4-preview/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "minimax-m2.7/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "minimax-m2.7/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "minimax-m2.7/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "minimax-m2.7/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "minimax-m3/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "minimax-m3/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "minimax-m3/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "ai21-jamba2-3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "ai21-jamba2-3b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "deepseek-v3-0324/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3-0324/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3-0324/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "deepseek-v3-0324/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3-0324/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3-0324/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3-0324/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "deepseek-v3.1/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "deepseek-v3.1/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v3.1/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "deepseek-v4.1-flash/render/*",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779); and the gateway's default \
         reasoning effort for this family is not the template's own default \
         (smg-project/smg-lab#33); and add_generation_prompt is not a field of SMG's chat \
         request; the header is always appended (smg-project/smg#2780); and SMG's request \
         schema types tool-call arguments as a string, as the API and the engines do; the \
         template accepts an object (smg-project/bellwether#12); and the gateway parses \
         tool-call arguments into objects before rendering, which this template does not \
         render as the reference does (smg-project/smg#2783)",
    ),
    (
        "dots3-note-prev/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "dots3-note-prev/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "dots3-note-prev/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "ernie-4.5-21b-a3b-thinking/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "ernie-4.5-21b-a3b-thinking/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "ernie-4.5-21b-a3b-thinking/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "gemma-4-e4b-it/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "gemma-4-e4b-it/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "glm-4.6/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "glm-4.6/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "glm-4.6/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "glm-4.6/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "glm-4.7-flash/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "glm-4.7-flash/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "glm-4.7-flash/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "glm-4.7-flash/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "granite-4.1-3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "granite-4.1-3b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "granite-4.1-3b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "hermes-4-14b/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "hermes-4-14b/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "hermes-4-14b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "hermes-4-14b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "hermes-4-14b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "hermes-4-14b/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "hermes-4-14b/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "hermes-4-14b/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "hermes-4-14b/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "inkling/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "inkling/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "inkling/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "k-exaone-236b-a23b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "k-exaone-236b-a23b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "k-exaone-236b-a23b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "ling-3.0-flash/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "ling-3.0-flash/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "ling-3.0-flash/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "ling-3.0-flash/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "llama-xlam-2-8b-fc-r/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "mimo-v2.5/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "mimo-v2.5/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "mimo-v2.5/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "minicpm5-2b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "minicpm5-2b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "minicpm5-2b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "minicpm5-2b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "minimax-m2/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "minimax-m2/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "minimax-m2/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "minimax-m2/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "mistral-7b-instruct-v0.3/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "nanbeige4.2-3b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "nanbeige4.2-3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "nanbeige4.2-3b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "nanbeige4.2-3b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "olmo-3-7b-instruct/render/*",
        "the gateway renders continue_final_message by popping the assistant turn and appending \
         its text after the generation header, which does not reproduce this template's \
         continued turn (smg-project/smg#2779); and add_generation_prompt is not a field of \
         SMG's chat request; the header is always appended (smg-project/smg#2780); and the \
         gateway's system prompt for this template is not the reference's: the function-calling \
         preamble is rendered for a request without tools, where the template writes that no \
         functions are available, and a system message is placed differently \
         (smg-project/smg-lab#107); and SMG's request schema types tool-call arguments as a \
         string, as the API and the engines do; the template accepts an object \
         (smg-project/bellwether#12)",
    ),
    (
        "phi-4-mini-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "phi-4-mini-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "phi-4-multimodal-instruct/render/*",
        "the reference's ids follow transformers' GPT2Tokenizer pattern (contractions, digit \
         runs and punctuation split as GPT-2 does) where tokenizer.json carries an \
         o200k-style regex (smg-project/smg-lab#102); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen-agentworld-35b-a3b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen-drive-1.0-4b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-7b-instruct-1m/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "qwen2.5-omni-7b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen2.5-omni-7b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen2.5-vl-32b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen2.5-vl-32b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen2.5-vl-7b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen2.5-vl-7b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-30b-a3b-instruct-2507/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-30b-a3b-instruct-2507/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-30b-a3b-instruct-2507/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-30b-a3b-thinking-2507/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-30b-a3b-thinking-2507/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-30b-a3b-thinking-2507/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-30b-a3b-thinking-2507/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-30b-a3b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-30b-a3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-30b-a3b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-30b-a3b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-4b-instruct-2507/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-4b-instruct-2507/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-4b-instruct-2507/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-4b-saferl/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-4b-saferl/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-4b-saferl/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-4b-saferl/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-4b-thinking-2507/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-4b-thinking-2507/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-4b-thinking-2507/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-4b-thinking-2507/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-8b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-coder-30b-a3b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-coder-30b-a3b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-coder-30b-a3b-instruct/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-coder-next/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-coder-next/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-coder-next/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-next-80b-a3b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-next-80b-a3b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-next-80b-a3b-instruct/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-next-80b-a3b-thinking/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-next-80b-a3b-thinking/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-next-80b-a3b-thinking/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-next-80b-a3b-thinking/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3-vl-235b-a22b-thinking/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-vl-235b-a22b-thinking/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-vl-235b-a22b-thinking/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-vl-30b-a3b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-vl-30b-a3b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-vl-8b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-vl-8b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3-vl-8b-thinking/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-vl-8b-thinking/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3-vl-8b-thinking/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwen3.5-27b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen3.5-2b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen3.5-35b-a3b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3.5-35b-a3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3.5-35b-a3b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3.5-9b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen3.6-27b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3.6-27b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3.6-27b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3.6-35b-a3b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3.6-35b-a3b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3.6-35b-a3b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3.8-2.4t-a95b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen3.8-27b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3.8-27b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwen3.8-27b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "qwen3.8-flash-next/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "qwen3guard-gen-0.6b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3guard-gen-0.6b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwq-32b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "qwq-32b/render/hermes-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "qwq-32b/render/hermes-glaive-func-calling-*",
        "the gateway parses tool-call arguments into objects before rendering and this \
         template writes them as the API's string, so a call in the history renders \
         differently (smg-project/smg#2783)",
    ),
    (
        "qwq-32b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "qwq-32b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "qwq-32b/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering, which this \
         template does not render as the reference does (smg-project/smg#2783)",
    ),
    (
        "seed-oss-36b-instruct/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "seed-oss-36b-instruct/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "seed-oss-36b-instruct/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "step-3.5-flash/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "step-3.5-flash/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "step-3.5-flash/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "step-3.5-flash/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "step3/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "step3/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "step3/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "step3/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "tinyllama-1.1b-chat-v1.0/render/*",
        "the reference's ids follow transformers' LlamaTokenizer with legacy=false, which \
         prepends the dummy space once per input where tokenizer.json's normalizer prepends \
         it per segment (smg-project/smg-lab#104); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump removes this entry; the model's other \
         render differences are listed again once the pin moves",
    ),
    (
        "trinity-mini/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "trinity-mini/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
    (
        "trinity-mini/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "trinity-mini/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the \
         engines do; the template accepts an object (smg-project/bellwether#12)",
    ),
    (
        "webworld-32b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and \
         appending its text after the generation header, which does not reproduce this \
         template's continued turn (smg-project/smg#2779)",
    ),
    (
        "webworld-32b/render/text-developer-role",
        "the gateway renders a developer message as a system message when the template has no \
         developer branch, as the engine's renderer does (smg-project/smg#3022); the recorded \
         template run keeps or drops the role",
    ),
    (
        "webworld-32b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always \
         appended (smg-project/smg#2780)",
    ),
];
/// Models whose tokenizer is known not to load, with the reason and where it
/// is tracked, so the run compares the others: an unlisted model that does
/// not load fails the run, and so does a listed one whose tokenizer loads
/// now, so the list cannot rot.
const KNOWN_UNLOADED: &[(&str, &str)] = &[(
    "kimi-k3",
    "the checkpoint ships tiktoken.model and no tokenizer.json; this harness fetches \
         tokenizer.json only (the tokenizer harness reads the tiktoken file since \
         smg-project/smg#2923)",
)];

/// The reason `known` lists for `id`: the entry that is the id itself, else
/// the first prefix entry the id begins with; none when the id is not listed.
fn known_reason<'a>(known: &BTreeMap<&'a str, &'a str>, id: &str) -> Option<&'a str> {
    known.get(id).copied().or_else(|| {
        known
            .iter()
            .find(|(entry, _)| prefix_of(entry).is_some_and(|prefix| id.starts_with(prefix)))
            .map(|(_, reason)| *reason)
    })
}

/// What a prefix entry stands for: the part before its trailing `*`; none for
/// an entry that is one fixture id.
fn prefix_of(entry: &str) -> Option<&str> {
    entry.strip_suffix('*')
}

/// `fixtures/<slug>/manifest.toml`: the model and the revision its fixtures
/// were recorded at.
#[derive(Deserialize)]
struct Manifest {
    model: String,
    revision: String,
}

/// One line of `fixtures/<slug>/render/<set>.jsonl`, bellwether's
/// `case.schema.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    id: String,
    kind: String,
    model: String,
    request: Value,
    reference: Reference,
    #[serde(default)]
    witnesses: Option<Value>,
}

#[derive(Deserialize)]
struct Reference {
    source: String,
    input_ids: Vec<u32>,
    text: String,
    #[serde(default)]
    provenance: Value,
}

/// One `[<kind>.<set>]` table of `<slug>/sets.toml`, which `bellwether record`
/// writes beside the sets and `bellwether unpack` copies along: how many cases
/// the set holds. Its other fields (`form`, `rejected`, `plain_bytes`,
/// `plain_sha256`) are not read here.
#[derive(Deserialize)]
struct SetTable {
    cases: usize,
}

struct Rendered {
    text: String,
    ids: Vec<u32>,
}

/// What one model compared.
#[derive(Default)]
struct Tally {
    cases: usize,
    differ: usize,
    witnessed: usize,
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} cases compared, {} match, {} differ, {} carry engine witnesses",
            self.cases,
            self.cases - self.differ,
            self.differ,
            self.witnessed
        )
    }
}

/// What a run compared, and what it could not.
#[derive(Default)]
struct Report {
    /// Each model whose render sets were read.
    loaded_slugs: BTreeSet<String>,
    /// The id of every case compared.
    seen: BTreeSet<String>,
    /// Why each case that differs from the reference differs, by id.
    differences: BTreeMap<String, String>,
    /// How many of the cases compared carry engine witnesses.
    witnessed: usize,
    /// Each model whose tokenizer did not load, and why.
    unloaded: Vec<(String, String)>,
    /// Each render set whose cases read differ from what its model's
    /// `sets.toml` lists: `<slug>/render/<set>`, the cases read (none when the
    /// tree has no file for the set) and the cases listed (none when the table
    /// does not list it).
    mismatches: Vec<(String, Option<usize>, Option<usize>)>,
}

impl Report {
    /// Count one compared case into `tally`, and keep its difference if it
    /// has one, written to `lines` as it is found.
    fn record(
        &mut self,
        fixture: &Fixture,
        outcome: Result<(), String>,
        tally: &mut Tally,
        known: &BTreeMap<&str, &str>,
        lines: &mut String,
    ) {
        self.seen.insert(fixture.id.clone());
        tally.cases += 1;
        if fixture.witnesses.is_some() {
            tally.witnessed += 1;
            self.witnessed += 1;
        }
        if let Err(why) = outcome {
            tally.differ += 1;
            let listed = known_reason(known, &fixture.id).map_or(String::new(), |reason| {
                format!("\n          known: {reason}")
            });
            lines.push_str(&format!(
                "  differs {} (reference {}): {why}{listed}\n",
                fixture.id, fixture.reference.source
            ));
            self.differences.insert(fixture.id.clone(), why);
        }
    }

    /// Take in what one model's comparison reported.
    fn merge(&mut self, other: Report) {
        self.loaded_slugs.extend(other.loaded_slugs);
        self.seen.extend(other.seen);
        self.differences.extend(other.differences);
        self.witnessed += other.witnessed;
        self.unloaded.extend(other.unloaded);
        self.mismatches.extend(other.mismatches);
    }

    /// What fails the run; empty when it passes.
    fn failures(
        &self,
        root: &Path,
        known: &BTreeMap<&str, &str>,
        known_unloaded: &BTreeMap<&str, &str>,
    ) -> Vec<String> {
        let mut failures = Vec::new();
        let unlisted: Vec<String> = self
            .unloaded
            .iter()
            .filter(|(slug, _)| !known_unloaded.contains_key(slug.as_str()))
            .map(|(slug, why)| format!("{slug}: {why}"))
            .collect();
        if !unlisted.is_empty() {
            failures.push(format!(
                "models not compared, because their tokenizer did not load:\n{}",
                unlisted.join("\n")
            ));
        }
        let loads_now: Vec<&str> = known_unloaded
            .keys()
            .copied()
            .filter(|slug| self.loaded_slugs.contains(*slug))
            .collect();
        if !loads_now.is_empty() {
            failures.push(format!(
                "listed in KNOWN_UNLOADED but their tokenizer loads now; remove: {}",
                loads_now.join(", ")
            ));
        }
        if !self.mismatches.is_empty() {
            let sets: Vec<String> = self
                .mismatches
                .iter()
                .map(|(set, read, listed)| format!("{set}: {}", counts(*read, *listed)))
                .collect();
            failures.push(format!(
                "sets whose cases read are not the cases sets.toml lists:\n{}",
                sets.join("\n")
            ));
        }
        if self.seen.is_empty() && self.unloaded.is_empty() {
            failures.push(format!(
                "no render case under {}: every <slug>/render directory is missing or empty",
                root.display()
            ));
        }
        let unexpected: Vec<String> = self
            .differences
            .iter()
            .filter(|(id, _)| known_reason(known, id).is_none())
            .map(|(id, why)| format!("{id}: {why}"))
            .collect();
        if !unexpected.is_empty() {
            failures.push(format!(
                "differences not listed in KNOWN_DIFFERENCES:\n{}",
                unexpected.join("\n")
            ));
        }
        let healed: Vec<&str> = known
            .keys()
            .copied()
            .filter(|entry| match prefix_of(entry) {
                Some(prefix) => self.seen_under(prefix) && !self.differs_under(prefix),
                None => self.seen.contains(*entry) && !self.differences.contains_key(*entry),
            })
            .collect();
        if !healed.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but matching the reference now; remove: {}",
                healed.join(", ")
            ));
        }
        let gone: Vec<&str> = known
            .keys()
            .copied()
            .filter(|entry| {
                let slug = entry.split('/').next().unwrap_or_default();
                self.loaded_slugs.contains(slug)
                    && match prefix_of(entry) {
                        Some(prefix) => !self.seen_under(prefix),
                        None => !self.seen.contains(*entry),
                    }
            })
            .collect();
        if !gone.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but no longer among the loaded fixtures; remove or \
                 rename: {}",
                gone.join(", ")
            ));
        }
        failures
    }

    /// Whether a compared case's id begins with `prefix`.
    fn seen_under(&self, prefix: &str) -> bool {
        self.seen
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .is_some_and(|id| id.starts_with(prefix))
    }

    /// Whether a differing case's id begins with `prefix`.
    fn differs_under(&self, prefix: &str) -> bool {
        self.differences
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .is_some_and(|(id, _)| id.starts_with(prefix))
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn render_fixtures_match_the_reference_byte_for_byte() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; point it at the tree `bellwether unpack` writes"
        );
        return;
    };
    let manifests = read_manifests(&root).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !manifests.is_empty(),
        "no <slug>/manifest.toml under {}",
        root.display()
    );
    let packed = packed_sets(&root, &manifests).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        packed.is_empty(),
        "{FIXTURES_ENV}={} is not the tree `bellwether unpack` writes: {} fixture sets under it \
         are compressed or Git LFS pointers (the first is {}), and this test reads plain JSON \
         Lines only. Run `bellwether unpack --fixtures <bellwether checkout>/fixtures --out \
         <dir>` and point {FIXTURES_ENV} at <dir>",
        root.display(),
        packed.len(),
        packed[0].display()
    );

    let known: BTreeMap<&str, &str> = KNOWN_DIFFERENCES.iter().copied().collect();
    let known_unloaded: BTreeMap<&str, &str> = KNOWN_UNLOADED.iter().copied().collect();
    let report =
        compare(&root, &manifests, &known, load_tokenizer).unwrap_or_else(|e| panic!("{e}"));
    let failures = report.failures(&root, &known, &known_unloaded);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Render each model's render sets under `root` through the chat request path
/// with the tokenizer `load` gives for the model, each case as its line is
/// read, and compare the cases read from each set with the model's
/// `sets.toml`. The models are compared on as many threads as the run has
/// CPUs; what each printed is written in the manifests' order, each model as
/// soon as it and the ones before it are done, so the output reads as a
/// serial run's. A model whose tokenizer does not load is kept in the report
/// and the run goes on. Prints each set read, each difference, and a tally
/// per model.
#[expect(
    clippy::print_stdout,
    reason = "the per-set and per-model report is test diagnostic output"
)]
fn compare(
    root: &Path,
    manifests: &[(String, Manifest)],
    known: &BTreeMap<&str, &str>,
    load: impl Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String> + Sync,
) -> Result<Report, String> {
    let mut report = Report::default();
    let mut summaries = Vec::with_capacity(manifests.len());
    in_order(
        manifests,
        |slug, manifest| compare_model(root, slug, manifest, known, &load),
        |compared| {
            print!("{}", compared.lines);
            report.merge(compared.report);
            summaries.push(compared.summary);
        },
    )?;
    println!("summary:");
    for summary in &summaries {
        println!("  {summary}");
    }
    println!(
        "{} cases match, {} differ, {} carry engine witnesses",
        report.seen.len() - report.differences.len(),
        report.differences.len(),
        report.witnessed
    );
    Ok(report)
}

/// What comparing one model gave: the lines a serial run would have printed
/// for it, its line of the summary, and its part of the report.
struct Compared {
    lines: String,
    summary: String,
    report: Report,
}

/// Compare one model's render sets, as [`compare`] describes.
fn compare_model<F>(
    root: &Path,
    slug: &str,
    manifest: &Manifest,
    known: &BTreeMap<&str, &str>,
    load: &F,
) -> Result<Compared, String>
where
    F: Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String>,
{
    let mut report = Report::default();
    let mut lines = String::new();
    let model_dir = root.join(slug);
    let sets = set_files(&model_dir)?;
    let listed = read_set_table(&model_dir)?;
    if sets.is_empty() {
        // Nothing to compare; a set the table lists is missing.
        let listed = listed.unwrap_or_default();
        let mismatches = set_mismatches(slug, &BTreeMap::new(), &listed);
        report.mismatches.extend(mismatches);
        return Ok(Compared {
            lines,
            summary: format!("{slug}: no render sets"),
            report,
        });
    }
    let Some(listed) = listed else {
        return Err(format!(
            "{slug}: no sets.toml beside its manifest.toml to check its sets against; \
             {FIXTURES_ENV} must point at the tree `bellwether unpack` writes, which carries \
             each model's sets.toml"
        ));
    };
    let (tok, from) = match load(slug, manifest) {
        Ok(loaded) => loaded,
        Err(why) => {
            report.unloaded.push((slug.to_string(), why));
            return Ok(Compared {
                lines,
                summary: format!("{slug}: not compared, its tokenizer did not load"),
                report,
            });
        }
    };
    report.loaded_slugs.insert(slug.to_string());
    lines.push_str(&format!(
        "{slug}: {} at {} from {from}\n",
        manifest.model, manifest.revision
    ));
    let mut tally = Tally::default();
    let mut read = BTreeMap::new();
    for (_, set, path) in &sets {
        let mut transformers = None;
        let cases = for_each_case(path, |fixture: Fixture| {
            assert_eq!(fixture.kind, "render", "{}: not a render case", fixture.id);
            assert_eq!(
                fixture.model, manifest.model,
                "{}: model differs from the manifest",
                fixture.id
            );
            transformers.get_or_insert_with(|| recorded_with(&fixture.reference.provenance));
            let outcome = match render(tok.as_ref(), &manifest.model, &fixture.request) {
                Err(e) => Err(e),
                Ok(got)
                    if got.text == fixture.reference.text
                        && got.ids == fixture.reference.input_ids =>
                {
                    Ok(())
                }
                Ok(got) => Err(describe(&got, &fixture.reference)),
            };
            report.record(&fixture, outcome, &mut tally, known, &mut lines);
        })?;
        let transformers = transformers.map_or(String::new(), |version| {
            format!("; recorded with transformers {version}")
        });
        let counted = counts(Some(cases), listed.get(set).copied());
        lines.push_str(&format!("  {set}: {counted}{transformers}\n"));
        read.insert(set.clone(), cases);
    }
    report
        .mismatches
        .extend(set_mismatches(slug, &read, &listed));
    Ok(Compared {
        lines,
        summary: format!("{slug}: {tally}"),
        report,
    })
}

/// Run `each` on every model, on as many threads as the run has CPUs, and
/// hand what it gives to `then` in the models' order, each as soon as it and
/// the ones before it are done. An error from `each` ends the run once the
/// models in flight are done.
fn in_order<T: Send>(
    manifests: &[(String, Manifest)],
    each: impl Fn(&str, &Manifest) -> Result<T, String> + Sync,
    mut then: impl FnMut(T),
) -> Result<(), String> {
    let threads = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(manifests.len())
        .max(1);
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, each) = (&next, &each);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some((slug, manifest)) = manifests.get(index) else {
                    break;
                };
                if tx.send((index, each(slug, manifest))).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut done = 0;
        for (index, outcome) in rx {
            pending.insert(index, outcome);
            while let Some(outcome) = pending.remove(&done) {
                then(outcome?);
                done += 1;
            }
        }
        Ok(())
    })
}

/// Hand a corpus request to the gateway's own request processing. The request
/// becomes the `ChatCompletionRequest` the HTTP layer would hand on: the
/// manifest's model added, keys SMG does not know ignored the way the gateway
/// ignores them, then normalized and validated as `ValidatedJson` does with
/// every chat request (the provider profile's rewrites, deprecated-field
/// migration, the `tool_choice` default; a request the gateway would answer
/// with 400 is a difference, not a rendering), then the tools narrowed by
/// `tool_choice` as the chat preparation stage does before rendering.
/// `process_chat_messages` then renders it exactly as the gateway does before
/// tokenizing, and the ids are the flat encode of that text, which is the
/// gateway's tokenize step for a renderer that returns text to encode.
fn render(tok: &dyn Tokenizer, model: &str, request: &Value) -> Result<Rendered, String> {
    let mut body = request.clone();
    let object = body.as_object_mut().ok_or("the request is not an object")?;
    object.insert("model".to_string(), Value::String(model.to_string()));
    let mut request: ChatCompletionRequest = serde_json::from_value(body)
        .map_err(|e| format!("the gateway does not accept this request: {e}"))?;
    request.normalize();
    request
        .validate()
        .map_err(|e| format!("the gateway rejects this request with 400: {e}"))?;
    // `filter_chat_request_by_tool_choice`, crate-private, applies this rule.
    let narrowed = match (&request.tools, &request.tool_choice) {
        (Some(tools), Some(choice)) => choice.narrow_tools(tools),
        _ => None,
    };
    if let Some(tools) = narrowed {
        request.tools = Some(tools);
    }
    let processed = process_chat_messages(&request, tok, None)?;
    let ids = tok
        .encode(&processed.text, false)
        .map_err(|e| format!("encode failed: {e}"))?
        .token_ids()
        .to_vec();
    Ok(Rendered {
        text: processed.text,
        ids,
    })
}

/// Where the rendering and the reference part: the first differing token and
/// the text around the first differing byte.
fn describe(got: &Rendered, want: &Reference) -> String {
    let id_at = got
        .ids
        .iter()
        .zip(&want.input_ids)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.ids.len().min(want.input_ids.len()));
    let byte_at = got
        .text
        .bytes()
        .zip(want.text.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.text.len().min(want.text.len()));
    format!(
        "ids part at index {id_at} ({} rendered, {} in the reference); text parts at byte {byte_at}: rendered {:?}, reference {:?}",
        got.ids.len(),
        want.input_ids.len(),
        window(&got.text, byte_at),
        window(&want.text, byte_at)
    )
}

/// Up to 40 bytes before `at` and 60 after, cut on character boundaries.
fn window(text: &str, at: usize) -> &str {
    let mut start = at.saturating_sub(40);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (at + 60).min(text.len());
    while !text.is_char_boundary(end) {
        end += 1;
    }
    &text[start..end]
}

fn read_manifests(root: &Path) -> Result<Vec<(String, Manifest)>, String> {
    let entries = fs::read_dir(root).map_err(|e| format!("cannot read {}: {e}", root.display()))?;
    let mut manifests = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", root.display()))?;
        let path = entry.path().join("manifest.toml");
        if !path.is_file() {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let manifest: Manifest =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        manifests.push((entry.file_name().to_string_lossy().into_owned(), manifest));
    }
    manifests.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(manifests)
}

/// The set files under `root` that are not plain JSON Lines: a
/// `<set>.jsonl.zst`, the form bellwether keeps a benchmark set in, and a set
/// file holding the pointer a clone has until Git LFS fetches the set.
/// bellwether's `fixtures/` directory has them; the tree `bellwether unpack`
/// writes has none.
fn packed_sets(root: &Path, manifests: &[(String, Manifest)]) -> Result<Vec<PathBuf>, String> {
    let mut packed = Vec::new();
    for (slug, _) in manifests {
        for dir in entries(&root.join(slug))? {
            if !dir.is_dir() {
                continue;
            }
            for path in entries(&dir)? {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.ends_with(".jsonl.zst")
                    || (name.ends_with(".jsonl") && starts_with_lfs_pointer(&path)?)
                {
                    packed.push(path);
                }
            }
        }
    }
    Ok(packed)
}

/// Whether the file begins with a Git LFS pointer.
fn starts_with_lfs_pointer(path: &Path) -> Result<bool, String> {
    let mut start = Vec::with_capacity(LFS_POINTER.len());
    fs::File::open(path)
        .and_then(|file| file.take(LFS_POINTER.len() as u64).read_to_end(&mut start))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(start == LFS_POINTER)
}

/// The model's sets of the kinds this test reads, kind by kind and in name
/// order: the kind, `<kind>/<set>`, and the set's file.
fn set_files(model_dir: &Path) -> Result<Vec<(&'static str, String, PathBuf)>, String> {
    let mut sets = Vec::new();
    for kind in KINDS {
        let dir = model_dir.join(kind);
        if !dir.is_dir() {
            continue;
        }
        for path in entries(&dir)? {
            let name = path.file_name().and_then(|name| name.to_str());
            if let Some(set) = name.and_then(|name| name.strip_suffix(".jsonl")) {
                let set = format!("{kind}/{set}");
                sets.push((kind, set, path));
            }
        }
    }
    Ok(sets)
}

/// The cases `<slug>/sets.toml` lists for each set of the kinds this test
/// reads, by `<kind>/<set>`; none when the model has no `sets.toml`.
fn read_set_table(model_dir: &Path) -> Result<Option<BTreeMap<String, usize>>, String> {
    let path = model_dir.join("sets.toml");
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let tables: BTreeMap<String, BTreeMap<String, SetTable>> =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let listed = tables
        .into_iter()
        .filter(|(kind, _)| KINDS.contains(&kind.as_str()))
        .flat_map(|(kind, sets)| {
            sets.into_iter()
                .map(move |(set, table)| (format!("{kind}/{set}"), table.cases))
        })
        .collect();
    Ok(Some(listed))
}

/// Each of the model's sets whose cases read differ from the cases its
/// `sets.toml` lists: `<slug>/<kind>/<set>`, the cases read (none when the
/// tree has no file for the set) and the cases listed (none when the table
/// does not list it).
fn set_mismatches(
    slug: &str,
    read: &BTreeMap<String, usize>,
    listed: &BTreeMap<String, usize>,
) -> Vec<(String, Option<usize>, Option<usize>)> {
    let sets: BTreeSet<&String> = read.keys().chain(listed.keys()).collect();
    sets.into_iter()
        .map(|set| (set, read.get(set).copied(), listed.get(set).copied()))
        .filter(|(_, read, listed)| read != listed)
        .map(|(set, read, listed)| (format!("{slug}/{set}"), read, listed))
        .collect()
}

/// A set's cases read and the cases its `sets.toml` lists, as the report
/// gives them.
fn counts(read: Option<usize>, listed: Option<usize>) -> String {
    let read = read.map_or_else(|| "no file".to_string(), |n| format!("{n} read"));
    let listed = listed.map_or_else(
        || "not in sets.toml".to_string(),
        |n| format!("sets.toml lists {n}"),
    );
    format!("{read}, {listed}")
}

/// The paths in `dir`, sorted.
fn entries(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = fs::read_dir(dir)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    paths.sort();
    Ok(paths)
}

/// Read a set a line at a time, handing each case to `each` as its line is
/// read; the number of cases read.
fn for_each_case<T: DeserializeOwned>(
    path: &Path,
    mut each: impl FnMut(T),
) -> Result<usize, String> {
    let file = fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut cases = 0;
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let case = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), number + 1))?;
        each(case);
        cases += 1;
    }
    Ok(cases)
}

/// The model's tokenizer, from the files [`tokenizer_dir`] finds, and where
/// they are.
fn load_tokenizer(slug: &str, manifest: &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String> {
    let dir = tokenizer_dir(&manifest.model, &manifest.revision, slug)?;
    let dir = dir
        .to_str()
        .ok_or_else(|| format!("tokenizer path {} is not UTF-8", dir.display()))?
        .to_string();
    let tok =
        create_tokenizer(&dir).map_err(|e| format!("the tokenizer at {dir} does not load: {e}"))?;
    Ok((tok, dir))
}

/// The transformers version a fixture set was recorded with, from the
/// provenance of its first line.
fn recorded_with(provenance: &Value) -> String {
    provenance
        .get("transformers")
        .and_then(Value::as_str)
        .unwrap_or("an unrecorded version")
        .to_string()
}

/// The vocabulary files a checkpoint may ship, in the order they are tried:
/// a `tokenizers` file, or the `vocab.json` + `merges.txt` pair of the models
/// that have no `tokenizer.json` (Qwen3-Omni), which counts only whole.
const VOCABULARIES: [&[&str]; 2] = [&["tokenizer.json"], &["vocab.json", "merges.txt"]];

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download of the files
/// the tokenizer and the renderer read: `tokenizer_config.json`, the separate
/// chat template files a checkpoint may ship instead of a template inside the
/// config, `config.json` for renderer detection, and the first of
/// [`VOCABULARIES`] the checkpoint serves whole. A download is written beside
/// its name and renamed into place, so an interrupted write never leaves a
/// short file the next run would trust.
fn tokenizer_dir(model: &str, revision: &str, slug: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = hf_cache_snapshot(model, revision) {
        return Ok(snapshot);
    }
    let dir = PathBuf::from(CACHE_DIR).join(slug).join(revision);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let client = reqwest::blocking::Client::new();
    download(&client, model, revision, &dir, "tokenizer_config.json", 100)?
        .ok_or_else(|| format!("{model} at {revision} serves no tokenizer_config.json"))?;
    for (file, min_bytes) in [
        ("chat_template.jinja", 1),
        ("chat_template.json", 1),
        ("config.json", 50),
    ] {
        download(&client, model, revision, &dir, file, min_bytes)?;
    }
    let mut vocabulary = None;
    for files in VOCABULARIES {
        let mut whole = true;
        for file in files {
            whole &= download(&client, model, revision, &dir, file, 100_000)?.is_some();
        }
        if whole {
            vocabulary = Some(files);
            break;
        }
    }
    vocabulary.ok_or_else(|| format!("{model} at {revision} serves none of {}", vocabularies()))?;
    Ok(dir)
}

/// [`VOCABULARIES`] for a message: `tokenizer.json, vocab.json + merges.txt`.
fn vocabularies() -> String {
    VOCABULARIES
        .iter()
        .map(|files| files.join(" + "))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Fetches `file` into `dir` unless it is there already; `Ok(None)` when the
/// checkpoint does not serve it (HTTP 404), an error for any other failure
/// or a file shorter than `min_bytes`.
fn download(
    client: &reqwest::blocking::Client,
    model: &str,
    revision: &str,
    dir: &Path,
    file: &str,
    min_bytes: usize,
) -> Result<Option<PathBuf>, String> {
    let path = dir.join(file);
    if path.is_file() {
        return Ok(Some(path));
    }
    let url = format!("https://huggingface.co/{model}/resolve/{revision}/{file}");
    let response = client
        .get(&url)
        .send()
        .map_err(|e| format!("GET {url}: {e}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
    let bytes = response.bytes().map_err(|e| format!("GET {url}: {e}"))?;
    if bytes.len() < min_bytes {
        return Err(format!(
            "{url}: {} bytes, expected at least {min_bytes}",
            bytes.len()
        ));
    }
    let part = dir.join(format!("{file}.part"));
    fs::write(&part, &bytes).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
    fs::rename(&part, &path)
        .map_err(|e| format!("cannot rename {} into place: {e}", part.display()))?;
    Ok(Some(path))
}

/// `<hub cache>/models--<org>--<name>/snapshots/<revision>`, the layout
/// `huggingface_hub` keeps, under `HF_HUB_CACHE`, `HF_HOME/hub` or the default
/// `~/.cache/huggingface/hub`. A snapshot holds every file the checkpoint
/// ships, so a separate chat template file is there when one exists.
fn hf_cache_snapshot(model: &str, revision: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        })?;
    snapshot_under(&hub, model, revision)
}

/// The snapshot of `model` at `revision` under the hub cache `hub`, when it
/// holds `tokenizer_config.json` and one of [`VOCABULARIES`] whole.
fn snapshot_under(hub: &Path, model: &str, revision: &str) -> Option<PathBuf> {
    let dir = hub
        .join(format!("models--{}", model.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    (dir.join("tokenizer_config.json").is_file()
        && VOCABULARIES
            .iter()
            .any(|files| files.iter().all(|file| dir.join(file).is_file())))
    .then_some(dir)
}

// The run's own checks, on fixture trees the tests below write.

/// A fixture tree in a temporary directory: each file's path under the root
/// and its contents.
fn tree(files: &[(&str, &str)]) -> std::io::Result<tempfile::TempDir> {
    let root = tempfile::tempdir()?;
    for (path, contents) in files {
        let path = root.path().join(path);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, contents)?;
    }
    Ok(root)
}

fn manifest(model: &str) -> String {
    format!("model = \"{model}\"\nrevision = \"0123456789abcdef0123456789abcdef01234567\"\n")
}

/// A `sets.toml` as `bellwether record` writes it, listing each
/// `<kind>/<set>` with its number of cases.
fn sets_toml(sets: &[(&str, usize)]) -> String {
    let mut text = String::new();
    for (set, cases) in sets {
        let (kind, name) = set.split_once('/').unwrap_or((set, ""));
        let sha256 = "0".repeat(64);
        text += &format!(
            "[{kind}.{name}]\nform = \"zstd\"\ncases = {cases}\nrejected = 0\nplain_bytes = 100\n\
             plain_sha256 = \"{sha256}\"\n\n"
        );
    }
    text
}

/// One render line whose request the mock tokenizer renders and encodes to
/// its reference.
fn render_line(id: &str, model: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"kind\":\"render\",\"model\":\"{model}\",\
         \"request\":{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}},\
         \"reference\":{{\"source\":\"hf-template\",\"input_ids\":[1],\
         \"text\":\"user: Hello\\nassistant: \"}}}}\n"
    )
}

/// One render line whose reference ids are not what the mock tokenizer
/// encodes the rendered text to.
fn differing_line(id: &str, model: &str) -> String {
    render_line(id, model).replace("\"input_ids\":[1]", "\"input_ids\":[2]")
}

/// The tokenizer crate's mock tokenizer, and where a loader would say it
/// came from.
fn mock() -> (Arc<dyn Tokenizer>, String) {
    (
        Arc::new(MockTokenizer::new()),
        "the mock tokenizer".to_string(),
    )
}

#[test]
fn compressed_sets_and_lfs_pointers_are_found_under_the_root() {
    let pointer = "version https://git-lfs.github.com/spec/v1\n\
                   oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n\
                   size 12345\n";
    let root = tree(&[
        ("m/manifest.toml", &manifest("org/m")),
        ("m/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("m/render/common.jsonl", &render_line("m/render/a", "org/m")),
        ("m/render/bfcl-simple.jsonl.zst", "a zstd frame"),
        ("m/render/bfcl-live.jsonl.zst", pointer),
        ("m/parse/bfcl-multiple.jsonl", pointer),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let found = packed_sets(root.path(), &manifests).unwrap();
    let under = |path: &str| root.path().join(path);
    assert_eq!(
        found,
        [
            under("m/parse/bfcl-multiple.jsonl"),
            under("m/render/bfcl-live.jsonl.zst"),
            under("m/render/bfcl-simple.jsonl.zst"),
        ]
    );
}

#[test]
fn the_cases_read_must_be_the_cases_sets_toml_lists() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[
                ("parse/common", 1),
                ("render/bfcl-simple", 3),
                ("render/common", 2),
            ]),
        ),
        ("m/render/common.jsonl", &render_line("m/render/a", model)),
        ("m/render/extra.jsonl", &render_line("m/render/b", model)),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock())).unwrap();
    assert_eq!(
        report.mismatches,
        [
            ("m/render/bfcl-simple".to_string(), None, Some(3)),
            ("m/render/common".to_string(), Some(1), Some(2)),
            ("m/render/extra".to_string(), Some(1), None),
        ]
    );
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    for set in ["m/render/bfcl-simple", "m/render/common", "m/render/extra"] {
        assert!(failures[0].contains(set), "{set} is not in {failures:#?}");
    }
}

#[test]
fn sets_without_a_sets_toml_stop_the_run() {
    let root = tree(&[
        ("m/manifest.toml", &manifest("org/m")),
        ("m/render/common.jsonl", &render_line("m/render/a", "org/m")),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let error = compare(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock()))
        .err()
        .expect("sets with no sets.toml to check them against should stop the run");
    assert!(error.contains("sets.toml"), "{error}");
}

#[test]
fn a_cache_snapshot_with_any_whole_vocabulary_is_the_tokenizer_dir() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let snapshot = |model: &str, files: &[&str]| -> Vec<(String, String)> {
        files
            .iter()
            .map(|file| {
                (
                    format!(
                        "models--{}/snapshots/{revision}/{file}",
                        model.replace('/', "--")
                    ),
                    String::from("{}"),
                )
            })
            .collect()
    };
    let files = [
        snapshot("org/json", &["tokenizer_config.json", "tokenizer.json"]),
        snapshot(
            "org/pair",
            &["tokenizer_config.json", "vocab.json", "merges.txt"],
        ),
        snapshot("org/half-pair", &["tokenizer_config.json", "vocab.json"]),
        snapshot("org/no-config", &["vocab.json", "merges.txt"]),
    ]
    .concat();
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_str()))
        .collect();
    let hub = tree(&files).unwrap();
    for (model, found) in [
        ("org/json", true),
        ("org/pair", true),
        ("org/half-pair", false),
        ("org/no-config", false),
        ("org/absent", false),
    ] {
        let snapshot = snapshot_under(hub.path(), model, revision);
        assert_eq!(snapshot.is_some(), found, "{model}: {snapshot:?}");
    }
}

#[test]
fn a_model_whose_tokenizer_does_not_load_fails_the_run_after_the_others() {
    let root = tree(&[
        ("a/manifest.toml", &manifest("org/a")),
        ("a/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("a/render/common.jsonl", &render_line("a/render/x", "org/a")),
        ("b/manifest.toml", &manifest("org/b")),
        ("b/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("b/render/common.jsonl", &render_line("b/render/x", "org/b")),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare(
        root.path(),
        &manifests,
        &BTreeMap::new(),
        |slug, _| match slug {
            "a" => Err("no tokenizer.json".to_string()),
            _ => Ok(mock()),
        },
    )
    .unwrap();
    assert_eq!(
        report.unloaded,
        [("a".to_string(), "no tokenizer.json".to_string())]
    );
    assert_eq!(report.seen, BTreeSet::from(["b/render/x".to_string()]));
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("a: no tokenizer.json"),
        "{failures:#?}"
    );
    // Listed as known not to load, "a" no longer fails the run; a listed model
    // whose tokenizer loads must leave the list.
    let listed = BTreeMap::from([("a", "its template uses a statement the engine lacks")]);
    assert_eq!(
        report.failures(root.path(), &BTreeMap::new(), &listed),
        Vec::<String>::new()
    );
    let stale = BTreeMap::from([("a", "does not load"), ("b", "does not load")]);
    let failures = report.failures(root.path(), &BTreeMap::new(), &stale);
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("loads now") && failures[0].ends_with("remove: b"),
        "{failures:#?}"
    );
}

#[test]
fn a_prefix_entry_covers_the_cases_under_it_and_must_cover_one() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[("render/bfcl-multi-turn-base", 2), ("render/common", 1)]),
        ),
        (
            "m/render/bfcl-multi-turn-base.jsonl",
            &(differing_line("m/render/bfcl-multi-turn-base-0", model)
                + &differing_line("m/render/bfcl-multi-turn-base-1", model)),
        ),
        (
            "m/render/common.jsonl",
            &render_line("m/render/common-1", model),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let listed = BTreeMap::from([(
        "m/render/bfcl-multi-turn-*",
        "the template writes a field the typed tools drop",
    )]);
    let report = compare(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    // The prefix covers both differing cases, so nothing fails the run.
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // Without it the two are unlisted.
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("m/render/bfcl-multi-turn-base-0")
            && failures[0].contains("m/render/bfcl-multi-turn-base-1"),
        "{failures:#?}"
    );
    // A prefix under which every loaded case matches must go, and so must one
    // that no loaded case begins with.
    let stale = BTreeMap::from([
        (
            "m/render/bfcl-multi-turn-*",
            "the template writes a field the typed tools drop",
        ),
        ("m/render/common-*", "matches now"),
        ("m/render/gsm8k-*", "no such set"),
    ]);
    let failures = report.failures(root.path(), &stale, &BTreeMap::new());
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures.iter().any(|failure| {
            failure.contains("matching the reference now")
                && failure.contains("m/render/common-*")
                && !failure.contains("bfcl")
        }),
        "{failures:#?}"
    );
    assert!(
        failures.iter().any(|failure| {
            failure.contains("no longer among the loaded fixtures")
                && failure.contains("m/render/gsm8k-*")
                && !failure.contains("bfcl")
        }),
        "{failures:#?}"
    );
}
