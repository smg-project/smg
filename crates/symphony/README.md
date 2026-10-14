# smg-symphony

One parser for everything a model says. A model's output stream interleaves content, reasoning and
tool calls in a vendor-specific format; this crate turns it into typed events through one object
with one method, `Parser::feed`, whose lifecycle is expressed as input: the prompt the engine was
given, each decoded delta, the end of the stream. Every byte of output lands in exactly one event,
including bytes that were dropped or could not be parsed.

Status: the public types are the contract, the adapters render them for the Chat Completions,
Responses and Messages APIs, and the engine runs a format table; the tables under `formats` are
the families recorded so far, and the models table below says which, and how far each is.

## Architecture

Symphony is a library, not a service: a caller feeds it a model's output and takes typed events
back. SMG's gateway is its first caller, and the one crate of SMG's it depends on is
`openai-protocol`: the request's `tools` as the OpenAI API spells them, and the response shapes of
the three APIs the adapters render. Everything else here is the model families' own syntax.

### The pipeline

```text
prompt, output bytes --> Engine, running a Format table --> Events --> adapter --> API response
                         |                                   ^
                         +---- the call syntax's assembler --+   (json, tagged, pythonic, shape)

request tools, Format table --> Format::grammar --> Grammar --> the engine's structural tag
```

One object, one method. A `Parser` is fed `Input`s in lifecycle order: one `Prompt` (the prompt
the engine was given, so the parser knows where the model starts, inside its thought or not), any
number of `Delta`s (each engine chunk, decoded, with the spans that say which token wrote which
byte), one `End` (the engine's finish reason). Each call appends `Event`s to an `Events` list;
whole-output parsing is the same sequence with one `Delta`, so streaming and non-streaming share
one implementation.

Every byte of output lands in exactly one event: content, reasoning, a call's name, a fragment of
its arguments, bytes the format dropped as wrapping (`Dropped`, with the reason), or bytes a region
could not be parsed from (`Malformed`, returned whole with the reason). Nothing vanishes silently,
and a client that wants the raw text back can have it.

### The public types

| Type or function | What it is |
|---|---|
| `Parser` (`feed`) | The one trait: one instance per generated choice, fed `Input`s, filling `Events`. |
| `Input`, `TokenSpan`, `EngineFinish` | The lifecycle steps and the token-to-byte alignment the gateway's decoder gives. |
| `formats::for_model`, `formats::Table` | The registry: the table that reads a model id's output, with the model's own turn opener, for every checkpoint in the models table below with a table; `Table::named` the table an operator names (`Table::name`: the gateway's `--tool-call-parser` and `--reasoning-parser`, or a worker's labels, take one in place of a legacy parser's name). |
| `Event`, `Events`, `Text` | The output: `Content`, `ReasoningStart`/`Reasoning`/`ReasoningEnd`, `ToolCallStart`/`ToolCallArguments`/`ToolCallEnd`, `Dropped`, `Malformed`, `Finish`. `Text` carries the bytes and the engine tokens they cost. |
| `DropReason`, `MalformedReason`, `FinishReason` | Why bytes were dropped, why a region could not be parsed, and the refined finish (`tool_calls` when calls were made). |
| `ParseError` | Buffer overflow, lifecycle misuse, or a defect in a parser; what was pushed before the error is final. |
| `Format`, `Emits`, `CallSyntax` | A model family's table: terminals (the marker strings), states (what the text in them is: content, reasoning, a call's arguments, wrapping), rows (`state + terminal = state`), the call syntax inside the arguments state, and the turn opener. The call syntax is a family's, or a call given as data (`CallSyntax::Shaped`, a `shape::CallShape`) as a response template writes it. A table read from a response template has the rows its fields need as well: turn anchors, the state a turn starts in, framing whose text is dropped, stays trimmed as the template's fields are, a stop that closes a call, and the terminals that are stop tokens. Built with a small builder and checked by `Format::validate`. `Format::replay` cuts the prompt to the part the engine replays, and `Format::reasoning_open` says whether that replay leaves the model inside its thought; `Format::thinks_first` says the template writes the thought before every call and answer. |
| `Engine` | The one parser that runs any table: `Engine::new(format, declared)`. |
| `ValueRead`, `Inferred`, `Format::reads_values_with` | How a shaped call's values read: a tagged value's text by the template's parser (`shape::ValueParser`), typed by its parameter's schema; a member of JSON arguments held and typed where the reader says; a key written twice. A table names its reader; `Inferred`, the default, reads as the tagged tables do, and a crate that implements a template library's semantics gives its own. |
| `shape::Layout`, `Format::with_layout` | How the model's chat template writes a turn where a response template's patterns leave the text open: the whitespace a `\s*` stands for, the text outside every field, each part of a call, the values, a thought and an answer, and which close the template writes for a region it closes more than one way. Whoever compiles the table learns it from a rendering, reading the call's shape through `Format::call_syntax` and the markers through `Format::terminal_texts`, and gives it to the table; `Layout::check` says whether it fits. Parsing does not read it; the structural tags of a shaped call are written from it. |
| `Declared` | The parameter types the request's tools declare, which type a tagged call's values. |
| `formats::*` | One function per family, each returning its table: `qwen3(syntax)`, `qwen2_5()`, `deepseek_v4_1()`, `seed_oss()`, `hy4()`, `ling()`, `iquest()`, `glm()`, `glm4()`, `k2_horizon()`, `olmo3()`, `lfm2_5()`, `xlam()`, `json()`, `minimax_m2()`, `minimax_m3()`, `kimi_k3()`, `inkling()`, `minicpm5()`, `dots3()`, `gemma4()`, `function_gemma()`, `mistral_v13()`, `llama_nemotron()`, `forced()`, `plain()`. |
| `adapt::chat`, `adapt::responses`, `adapt::messages` | Pure functions from events to Chat Completions, Responses and Messages API shapes, streamed (`delta`, `Stream`) and whole (`message`, `output`); the driver adds only the envelope. |
| `Grammar`, `Tag`, `Format::grammar`, `Format::grammar_for`, `grammar::Options` | The structural tag an engine takes for a forced or required call, derived from the same table and the request's tools, or `None` for a table without calls: `to_json()` as the engine reads it, `payload()` as a request carries it. `grammar_for` takes the request's `Options`, one call among them (a named function, `parallel_tool_calls: false`); `Format::answer_grammar` is the tag for an answer that is JSON a schema accepts (`response_format`). A table given its vocabulary's added tokens (`Format::marker_tokens`) writes each marker that is one token as that token (`Grammar::Token`) inside its tags; `CallSyntax::tag_texts` lists each syntax's own tags. A call given as data gets its tag from the table's layout (`Format::with_layout`): what the model's chat template writes, and no tag without one. |
| `Scanner`, `Piece` | Marker matching across chunk boundaries, holding back a half-arrived marker. |
| `Ledger` | Token accounting: which tokens wrote which bytes, so every text event can say what it cost. |

### The modules

- `format`, `formats`: the table type and the tables, one per family. A table is data, read by
  the engine and by the grammar derivation alike; it says what the template writes and nothing
  about how to parse it.
- `engine`, `markers`, `tokens`: the engine, its scanner and its token ledger.
- `json`: the JSON family's assembler (`{"name": ..., "arguments": {...}}` as it arrives,
  streaming the model's own argument bytes), the JSON list (xLAM's bare, Llama Nemotron's between
  markers), the arguments object alone (Mistral's after its marker, the forced call's as the whole
  output), and the partial-JSON and prefix readers they rest on.
- `tagged`: the assemblers for arguments written as tags: Qwen 3.5's `<function=`/`<parameter=`,
  the keyed `<arg_key>`/`<arg_value>` pairs (GLM, Ling, Hy4, IQuest), the attributed
  `<parameter name="…">` tags in four spellings (MiniMax M2, dots3, MiniCPM5, DeepSeek's DSML,
  Kimi K3's XTML), MiniMax M3's XML tree; and the value reader that turns a value's text into the
  JSON the request's tools declare.
- `pythonic`: calls written as Python, `name(key=value)`.
- `shape`: a call's parts given as data, as a template's own patterns write them: linear patterns
  (literals, runs of a class of characters, `\b`, optional literals) and the ways a value ends,
  checked for what a stream can decide; a matcher that runs them over the output as it arrives
  and answers as Python's `regex` module does for the whole text, and a search that finds where
  the text before a closing marker starts; an interner that keeps a shape built at run time
  `'static`; and the call they make up, its header after the region's marker, its arguments (one
  JSON value, an object that names the call, or tagged parameters found as `finditer` finds them)
  and its trailer before the closing marker, with the assembler that reads it: a marker whose
  header does not match opens no call and is given back to the state it came in.
- `grammar`: the structural-tag derivation and its per-syntax spellings (`keyed`, `xtml`,
  `tagged`, `pythonic`, `attributed` for MiniMax M2's, dots3's, MiniCPM5's and DSML's, and `xml`
  for MiniMax M3's; the JSON, compact keyed, list, Inkling and Mistral spellings in the module
  itself), with
  the argument shapes a JSON schema pins (`schema`).
- `adapt`: the three API adapters, every response policy in one place.
- `event`, `input`, `parser`: the contract above.

### Adding a model

A model of a known syntax costs a table and evidence, no engine change:

1. Read the chat template and write the table under `formats/`: the markers as terminals, the
   states, the rows, the call syntax, the turn opener. Unit tests on the recorded shapes.
2. Add the model's parity row to `tests/bellwether_parse_fixtures.rs`, which replays bellwether's
   recorded sets through the table at every chunking and compares every byte against the reference.
3. Replay the whole set on a checkout; the models table below records the result.

A new call syntax costs an assembler under `json`, `tagged` or `pythonic`, a `CallSyntax` variant,
and a grammar module under `grammar`, each with the same evidence. A syntax a template writes as
patterns (a response template's) costs none of them: it is a `shape::CallShape`, data the shaped
syntax runs. Its structural tag comes from the layout its chat template renders, which no table
derives yet.

### Evidence

Symphony is judged against [bellwether](https://github.com/smg-project/bellwether), a corpus of
recorded fixtures: for every checkpoint group, the chat template's own renders and the model's
outputs, at benchmark scale, with the reference parse beside each. The parity test replays them
through the tables; a derived grammar is pinned in unit tests to the hand-written tag it replaces
where one existed, and was checked in review against the recorded calls where none did.

## Models

Symphony's target is every checkpoint group bellwether has a manifest for; a group is the
checkpoints that share a tokenizer and a template, recorded once by its primary. The table below
lists the groups bellwether has recorded, and the groups it listed before bellwether's manifests
outgrew it; it is written from `tests/readme.rs` and checked by `cargo test`, so it says what this
branch holds, and a stale table fails the build. It runs newest first: **Released** is the day the
primary's Hugging Face repository was created.

A group is **ready** when its table is on main, bellwether's `main` holds the group's
benchmark-scale set (the count in **bellwether set**), and a run of the fixture test
(`tests/bellwether_parse_fixtures.rs`) on main's code has replayed every case of that set at every
chunking with no difference. That run is local, and the pull request that marked the group ready
records it: SMG's CI runs the fixture test on every change to the crate against the bellwether
commit pinned in `.github/versions/bellwether.ref`, which holds the hand-written sets only, so CI
guards a ready group's set once the pin moves to a commit that holds it. **Replaying the set**
means the table is on main and bellwether's `main` holds the set, and that run has not finished.
**Awaiting fixtures** means the table is on main and bellwether has not recorded the group's set
yet; every such table was previewed against bellwether's scale run of 2026-10-06 with no
difference. **Pending** means no table yet.

**SMG today** is what the gateway takes for the group's primary before Symphony is wired in: the
`--tool-call-parser` and `--reasoning-parser` names its registries resolve from the model id, as
in `smg serve --model-path Qwen/Qwen3-8B --tool-call-parser qwen --reasoning-parser qwen3`; none
means passthrough, the output read as content. Symphony's tables carry one name each, the
**Table** column, and the Qwen3 table takes its call syntax as a second selector (JSON calls, or
the tagged calls of Qwen 3.5 and later and Qwen3-Coder). `formats::for_model` picks the table
from the model id by its exact repository name in any case, the group's primary or any checkpoint
under **Also**, so a model listed here with a table needs neither flag once the gateway reads it
with Symphony (a quantized or renamed checkpoint is not listed, and keeps the older parsers until
its group names it); whether the flags stay as aliases is an open question of the design.

<!-- models: begin -->
199 groups, 339 checkpoints, newest first.
12 ready, 128 replaying a recorded set, 0 on main awaiting fixtures, 59 pending.

| Released | Group | Model | Also | Table | bellwether set | Status | SMG today |
|---|---|---|---|---|---:|---|---|
| 2026-10-05 | [d1-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/d1-3b) | LiquidAI/d1-3B |  | `lfm2.5` | 589,657 | replaying the set | none |
| 2026-10-02 | [h2o-lightning-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/h2o-lightning-4b) | h2oai/h2o-lightning-4b |  | `qwen3`, tagged calls | 474,543 | replaying the set | none |
| 2026-09-29 | [arex-2](https://github.com/smg-project/bellwether/tree/main/fixtures/arex-2) | BAAI/AREX-2 |  | `qwen3`, tagged calls | 474,563 | replaying the set | none |
| 2026-09-28 | [iquest-q1](https://github.com/smg-project/bellwether/tree/main/fixtures/iquest-q1) | IQuestLab/IQuest-Q1 |  | `iquest` | 503,273 | replaying the set | none |
| 2026-09-24 | [huatuogpt-3-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/huatuogpt-3-27b) | FreedomIntelligence/HuatuoGPT-3-27B |  | `qwen3`, tagged calls | 474,563 | replaying the set | none |
| 2026-09-11 | [atria-dawn-preview](https://github.com/smg-project/bellwether/tree/main/fixtures/atria-dawn-preview) | internlm/Atria-Dawn-Preview | Atria-Dawn-Preview-Ascend-w8a8 | `glm` | 503,273 | replaying the set | none |
| 2026-09-10 | [deepseek-v4.1-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v4.1-flash) | deepseek-ai/DeepSeek-V4.1-Flash |  | `deepseek_v4.1` | 589,659 | replaying the set | `deepseek_v41`, `deepseek_v41` |
| 2026-09-06 | [minicpm5-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/minicpm5-2b) | openbmb/MiniCPM5-2B |  | `minicpm5` | 589,277 | replaying the set | none |
| 2026-09-01 | [k2-horizon-36b](https://github.com/smg-project/bellwether/tree/main/fixtures/k2-horizon-36b) | IFM/K2-Horizon-36B | K2-Horizon-32B, K2-Horizon-375B-A23B | `k2_horizon` | 12,814 | replaying the set | none |
| 2026-09-01 | [nvidia-nemotronlabs-ai-for-media-sports-tennis](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotronlabs-ai-for-media-sports-tennis) | nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis |  | — | 12,825 | pending | none |
| 2026-08-27 | [contextpilot-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/contextpilot-14b) | tencent/ContextPilot-14B | ByteDance-Seed/cudaLLM-8B, FreedomIntelligence/HuatuoGPT-3-32B, nvidia/Nemotron-Terminal-14B, nvidia/Nemotron-Terminal-32B, nvidia/Nemotron-Terminal-8B, OpenMOSS-Team/Qwen3-32B-ABC, ContextPilot-8B | `qwen3` | 583,086 | replaying the set | none |
| 2026-08-27 | [hy4-preview](https://github.com/smg-project/bellwether/tree/main/fixtures/hy4-preview) | tencent/Hy4-preview |  | `hy4` | 589,298 | replaying the set | `hy_v4`, `hy_v4` |
| 2026-08-27 | [qwen-drive-1.0-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen-drive-1.0-4b) | Qwen/Qwen-Drive-1.0-4B |  | `qwen3`, tagged calls | 474,543 | replaying the set | `qwen`, `qwen3` |
| 2026-08-25 | [glm-5.3](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5.3) | zai-org/GLM-5.3 | GLM-5.3-BF16 | `glm` | 503,273 | replaying the set | `glm47_moe`, `glm45` |
| 2026-08-25 | [glm-5.3-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5.3-flash) | zai-org/GLM-5.3-Flash | GLM-5.3-Flash-BF16 | `glm` | 503,273 | replaying the set | `glm47_moe`, `glm45` |
| 2026-08-24 | [qwen3.8-flash-next](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-flash-next) | Qwen/Qwen3.8-Flash-Next |  | `qwen3`, tagged calls | 474,563 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-08-09 | [dots3-note-prev](https://github.com/smg-project/bellwether/tree/main/fixtures/dots3-note-prev) | dots-studio/dots3-note-prev |  | `dots3` | 589,296 | replaying the set | none |
| 2026-08-09 | [muse-glimmer-30b](https://github.com/smg-project/bellwether/tree/main/fixtures/muse-glimmer-30b) | meta-models/Muse-Glimmer-30B |  | — | 494,673 | pending | none |
| 2026-08-08 | [qwen3.8-2.4t-a95b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-2.4t-a95b) | Qwen/Qwen3.8-2.4T-A95B |  | `qwen3`, tagged calls | 474,553 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-08-05 | [qwen3.8-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-27b) | Qwen/Qwen3.8-27B | JetBrains/Qwen3.8-3.6-27B-blend | `qwen3`, tagged calls | 474,563 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-08-02 | [ling-3.0-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/ling-3.0-flash) | inclusionAI/Ling-3.0-flash | Ling-3.0-flash-fp4 | `ling` | 589,298 | replaying the set | none |
| 2026-08-01 | [nvidia-nemotron-3.5-lightning-30b-a3b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3.5-lightning-30b-a3b-bf16) | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2026-07-23 | [arex-base](https://github.com/smg-project/bellwether/tree/main/fixtures/arex-base) | BAAI/AREX-Base |  | `qwen3`, tagged calls | 474,543 | replaying the set | none |
| 2026-07-23 | [arex-turbo](https://github.com/smg-project/bellwether/tree/main/fixtures/arex-turbo) | BAAI/AREX-Turbo |  | `qwen3`, tagged calls | 474,543 | replaying the set | none |
| 2026-07-21 | [nanbeige4.2-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/nanbeige4.2-3b) | Nanbeige/Nanbeige4.2-3B |  | `qwen3`, tagged calls | 589,298 | replaying the set | none |
| 2026-07-17 | [fara1.5-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/fara1.5-27b) | microsoft/Fara1.5-27B |  | `qwen3`, tagged calls | 474,543 | replaying the set | none |
| 2026-07-14 | [inkling](https://github.com/smg-project/bellwether/tree/main/fixtures/inkling) | thinkingmachines/Inkling | Inkling-Small | `inkling` | 589,659 | replaying the set | `inkling`, `inkling` |
| 2026-07-09 | [cosmos3-edge-policy-droid](https://github.com/smg-project/bellwether/tree/main/fixtures/cosmos3-edge-policy-droid) | nvidia/Cosmos3-Edge-Policy-DROID |  | `qwen3`, tagged calls | 12,829 | replaying the set | none |
| 2026-07-01 | [cosmos3-edge](https://github.com/smg-project/bellwether/tree/main/fixtures/cosmos3-edge) | nvidia/Cosmos3-Edge |  | `qwen3`, tagged calls | 12,829 | replaying the set | none |
| 2026-06-24 | [nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16) | nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | none |
| 2026-06-22 | [qwen-agentworld-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen-agentworld-35b-a3b) | Qwen/Qwen-AgentWorld-35B-A3B |  | `qwen3`, tagged calls | 474,563 | replaying the set | `qwen`, `qwen3` |
| 2026-06-16 | [glm-5.2](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5.2) | zai-org/GLM-5.2 |  | `glm` | 492,389 | replaying the set | `glm47_moe`, `glm45` |
| 2026-06-13 | [kimi-k3](https://github.com/smg-project/bellwether/tree/main/fixtures/kimi-k3) | moonshotai/Kimi-K3 |  | `kimi_k3` | 54,418 | ready | `kimi_k3`, `kimi_k3` |
| 2026-06-03 | [nvidia-nemotron-3-ultra-550b-a55b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3-ultra-550b-a55b-bf16) | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | Nemotron-3-Labs-Ultra-Math-RL, NVIDIA-Nemotron-Labs-Teacher-Chat, NVIDIA-Nemotron-Labs-Teacher-Competition-Coding, NVIDIA-Nemotron-Labs-Teacher-General-Reasoning, NVIDIA-Nemotron-Labs-Teacher-Instruction-Following, NVIDIA-Nemotron-Labs-Teacher-STEM | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2026-06-02 | [minimax-m3](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m3) | MiniMaxAI/MiniMax-M3 |  | `minimax_m3` | 589,302 | replaying the set | `minimax_m3`, `minimax_m3` |
| 2026-05-22 | [caprl-video-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/caprl-video-4b) | internlm/CapRL-Video-4B | inclusionAI/ArmorOCR | `qwen3` | 576,772 | replaying the set | none |
| 2026-05-18 | [bitcpm-cann-3b-unquantized](https://github.com/smg-project/bellwether/tree/main/fixtures/bitcpm-cann-3b-unquantized) | openbmb/BitCPM-CANN-3B-unquantized | BitCPM-CANN-1B, BitCPM-CANN-1B-unquantized, BitCPM-CANN-3B | — | 328,662 | pending | none |
| 2026-05-18 | [ernie-image-aes](https://github.com/smg-project/bellwether/tree/main/fixtures/ernie-image-aes) | baidu/ERNIE-Image-Aes |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2026-05-12 | [fara1.5-9b](https://github.com/smg-project/bellwether/tree/main/fixtures/fara1.5-9b) | microsoft/Fara1.5-9B | Fara1.5-4B | `qwen3`, tagged calls | 474,543 | replaying the set | none |
| 2026-05-01 | [apertus-v1.1-4b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/apertus-v1.1-4b-instruct) | swiss-ai/Apertus-v1.1-4B-Instruct | Apertus-v1.1-0.5B-Instruct, Apertus-v1.1-1.5B-Instruct | — | 184,972 | pending | none |
| 2026-04-27 | [mimo-v2.5](https://github.com/smg-project/bellwether/tree/main/fixtures/mimo-v2.5) | XiaomiMiMo/MiMo-V2.5 | MiMo-V2.5-Base | `qwen3`, tagged calls, the value between the tags | 589,278 | replaying the set | none |
| 2026-04-23 | [laguna-xs.2](https://github.com/smg-project/bellwether/tree/main/fixtures/laguna-xs.2) | poolside/Laguna-XS.2 |  | `glm4` | 490,452 | replaying the set | none |
| 2026-04-21 | [qwen3.6-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.6-27b) | Qwen/Qwen3.6-27B |  | `qwen3`, tagged calls | 474,563 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-04-19 | [bar-2x7b-base](https://github.com/smg-project/bellwether/tree/main/fixtures/bar-2x7b-base) | allenai/BAR-2x7B-Base |  | — | 328,662 | pending | none |
| 2026-04-19 | [bar-2x7b-code](https://github.com/smg-project/bellwether/tree/main/fixtures/bar-2x7b-code) | allenai/BAR-2x7B-Code | BAR-2x7B-Code-SFT, BAR-2x7B-Safety, BAR-2x7B-Tool-Use, BAR-5x7B | — | 328,662 | pending | none |
| 2026-04-19 | [bar-2x7b-math](https://github.com/smg-project/bellwether/tree/main/fixtures/bar-2x7b-math) | allenai/BAR-2x7B-Math | BAR-2x7B-Math-SFT | — | 328,662 | pending | none |
| 2026-04-19 | [bar-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/bar-7b) | allenai/BAR-7B |  | — | 328,662 | pending | none |
| 2026-04-15 | [qwen3.6-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.6-35b-a3b) | Qwen/Qwen3.6-35B-A3B |  | `qwen3`, tagged calls | 474,563 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-04-09 | [minimax-m2.7](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m2.7) | MiniMaxAI/MiniMax-M2.7 |  | `minimax_m2` | 12,821 | replaying the set | `minimax_m2`, `minimax` |
| 2026-04-06 | [granite-4.1-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-4.1-3b) | ibm-granite/granite-4.1-3b | granite-4.1-30b | `qwen2.5` | 576,772 | replaying the set | none |
| 2026-04-04 | [exaone-4.5-33b](https://github.com/smg-project/bellwether/tree/main/fixtures/exaone-4.5-33b) | LGAI-EXAONE/EXAONE-4.5-33B |  | `qwen3` | 503,634 | replaying the set | none |
| 2026-04-03 | [glm-5.1](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5.1) | zai-org/GLM-5.1 |  | `glm` | 163,746 | replaying the set | `glm47_moe`, `glm45` |
| 2026-04-01 | [nvidia-nemotron-labs-3-elastic-30b-a3b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-labs-3-elastic-30b-a3b-bf16) | nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | none |
| 2026-03-31 | [mistral-medium-3.5-128b](https://github.com/smg-project/bellwether/tree/main/fixtures/mistral-medium-3.5-128b) | mistralai/Mistral-Medium-3.5-128B |  | `mistral_v13` | 583,079 | replaying the set | `mistral`, none |
| 2026-03-19 | [dots.mocr](https://github.com/smg-project/bellwether/tree/main/fixtures/dots.mocr) | dots-studio/dots.mocr | dots.mocr-svg | — | 328,662 | pending | none |
| 2026-03-18 | [nemotron-cascade-2-30b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/nemotron-cascade-2-30b-a3b) | nvidia/Nemotron-Cascade-2-30B-A3B |  | `qwen3`, tagged calls | 12,829 | replaying the set | none |
| 2026-03-10 | [nvidia-nemotron-3-super-120b-a12b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3-super-120b-a12b-bf16) | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | Nemotron-3-Labs-Ultra-Math-SFT, NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2026-03-07 | [nvidia-nemotron-3-nano-4b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3-nano-4b-bf16) | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2026-03-05 | [areal-tau2-telecom-sft-30b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-tau2-telecom-sft-30b) | inclusionAI/AReaL-tau2-telecom-sft-30B | AReaL-tau2-airline-sft-30B, AReaL-tau2-retail-sft-30B | `qwen3` | 554,276 | replaying the set | none |
| 2026-03-05 | [areal-tau2-telecom-user-sft-30b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-tau2-telecom-user-sft-30b) | inclusionAI/AReaL-tau2-telecom-user-sft-30B |  | `qwen3` | 583,086 | replaying the set | none |
| 2026-03-02 | [gemma-4-e4b-it](https://github.com/smg-project/bellwether/tree/main/fixtures/gemma-4-e4b-it) | google/gemma-4-E4B-it | gemma-4-E2B-it | `gemma4` | 478,241 | replaying the set | `json`, none |
| 2026-03-01 | [areal-sea-235b-a22b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-sea-235b-a22b) | inclusionAI/AReaL-SEA-235B-A22B | AReaL-tau2-merge-sft-235B | `qwen3` | 554,276 | replaying the set | none |
| 2026-02-28 | [qwen3.5-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-2b) | Qwen/Qwen3.5-2B | ATH-MaaS/OvisOCR2, Qwen3.5-0.8B | `qwen3`, tagged calls | 461,722 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-02-27 | [qwen3.5-9b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-9b) | Qwen/Qwen3.5-9B | microsoft/FrogNano-4B-2609, Qwen3.5-4B | `qwen3`, tagged calls | 474,543 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-02-24 | [qwen3.5-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-27b) | Qwen/Qwen3.5-27B |  | `qwen3`, tagged calls | 54,179 | ready | `qwen_xml`, `qwen3` |
| 2026-02-24 | [qwen3.5-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-35b-a3b) | Qwen/Qwen3.5-35B-A3B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, xlangai/Qwen-CUA, xlangai/RoboFine-VLM-397B-A17B | `qwen3`, tagged calls | 474,543 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-02-13 | [webworld-32b](https://github.com/smg-project/bellwether/tree/main/fixtures/webworld-32b) | Qwen/WebWorld-32B | WebWorld-8B, WebWorld-14B | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2026-02-12 | [minimax-m2.5](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m2.5) | MiniMaxAI/MiniMax-M2.5 |  | `minimax_m2` | 12,821 | replaying the set | `minimax_m2`, `minimax` |
| 2026-02-11 | [glm-5](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5) | zai-org/GLM-5 |  | `glm` | 1,947 | replaying the set | `glm47_moe`, `glm45` |
| 2026-02-09 | [astabrief-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/astabrief-8b) | allenai/AstaBrief_8B | AstaBrief_8B_SFT | `qwen3` | 583,086 | replaying the set | none |
| 2026-02-01 | [step-3.5-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/step-3.5-flash) | stepfun-ai/Step-3.5-Flash |  | `qwen3`, tagged calls | 54,165 | replaying the set | `step3`, none |
| 2026-01-30 | [qwen3-coder-next](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-coder-next) | Qwen/Qwen3-Coder-Next | Qwen3-Coder-Next-Base | `qwen3`, tagged calls | 490,469 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-01-23 | [mistral-small-4-119b-2603](https://github.com/smg-project/bellwether/tree/main/fixtures/mistral-small-4-119b-2603) | mistralai/Mistral-Small-4-119B-2603 |  | `mistral_v13` | 583,079 | replaying the set | `mistral`, none |
| 2026-01-19 | [agentcpm-report](https://github.com/smg-project/bellwether/tree/main/fixtures/agentcpm-report) | openbmb/AgentCPM-Report |  | — | 328,660 | pending | none |
| 2026-01-19 | [glm-4.7-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.7-flash) | zai-org/GLM-4.7-Flash |  | `glm` | 1,947 | replaying the set | `glm47_moe`, none |
| 2026-01-14 | [nemotron-research-goosereason-4b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/nemotron-research-goosereason-4b-instruct) | nvidia/Nemotron-Research-GooseReason-4B-Instruct | OpenMOSS-Team/SciJudge-4B, OpenMOSS-Team/SciJudge-4B-2605 | `qwen2.5` | 576,772 | replaying the set | none |
| 2026-01-13 | [baichuan-m3-235b](https://github.com/smg-project/bellwether/tree/main/fixtures/baichuan-m3-235b) | baichuan-inc/Baichuan-M3-235B |  | `qwen3` | 554,276 | replaying the set | none |
| 2026-01-12 | [falcon-h1-tiny-90m-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/falcon-h1-tiny-90m-instruct) | tiiuae/Falcon-H1-Tiny-90M-Instruct | Falcon-H1-Tiny-90M-Instruct-Curriculum | `qwen2.5` | 429,360 | replaying the set | none |
| 2026-01-12 | [falcon-h1-tiny-multilingual-100m-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/falcon-h1-tiny-multilingual-100m-instruct) | tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct |  | — | 328,662 | pending | none |
| 2026-01-12 | [falcon-h1-tiny-r-0.6b](https://github.com/smg-project/bellwether/tree/main/fixtures/falcon-h1-tiny-r-0.6b) | tiiuae/Falcon-H1-Tiny-R-0.6B |  | — | 328,662 | pending | none |
| 2026-01-12 | [falcon-h1-tiny-r-90m](https://github.com/smg-project/bellwether/tree/main/fixtures/falcon-h1-tiny-r-90m) | tiiuae/Falcon-H1-Tiny-R-90M |  | — | 328,662 | pending | none |
| 2026-01-11 | [agentcpm-explore](https://github.com/smg-project/bellwether/tree/main/fixtures/agentcpm-explore) | openbmb/AgentCPM-Explore |  | `qwen3` | 583,086 | replaying the set | none |
| 2026-01-06 | [ai21-jamba2-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/ai21-jamba2-3b) | ai21labs/AI21-Jamba2-3B | AI21-Jamba2-Mini | `qwen2.5` | 576,772 | replaying the set | none |
| 2026-01-06 | [lfm2.5-1.2b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/lfm2.5-1.2b-instruct) | LiquidAI/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Thinking | `lfm2.5` | 589,657 | replaying the set | none |
| 2025-12-26 | [k-exaone-236b-a23b](https://github.com/smg-project/bellwether/tree/main/fixtures/k-exaone-236b-a23b) | LGAI-EXAONE/K-EXAONE-236B-A23B |  | `qwen3` | 503,634 | replaying the set | none |
| 2025-12-24 | [caprl-qwen3vl-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/caprl-qwen3vl-4b) | internlm/CapRL-Qwen3VL-4B | CapRL-Qwen3VL-2B | `qwen3` | 576,772 | replaying the set | `qwen`, `qwen3` |
| 2025-12-22 | [glm-4.7](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.7) | zai-org/GLM-4.7 |  | `glm` | 1,947 | replaying the set | `glm47_moe`, none |
| 2025-12-09 | [autoglm-phone-9b-multilingual](https://github.com/smg-project/bellwether/tree/main/fixtures/autoglm-phone-9b-multilingual) | zai-org/AutoGLM-Phone-9B-Multilingual |  | — | 328,664 | pending | `json`, none |
| 2025-12-08 | [autoglm-phone-9b](https://github.com/smg-project/bellwether/tree/main/fixtures/autoglm-phone-9b) | zai-org/AutoGLM-Phone-9B |  | — | 328,664 | pending | `json`, none |
| 2025-12-08 | [nemotron-cascade-14b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/nemotron-cascade-14b-thinking) | nvidia/Nemotron-Cascade-14B-Thinking | Nemotron-Cascade-8B, Nemotron-Cascade-8B-Thinking | `qwen3` | 583,086 | replaying the set | none |
| 2025-12-07 | [glm-4.6v](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.6v) | zai-org/GLM-4.6V |  | `glm4` | 492,389 | replaying the set | `glm45_moe`, none |
| 2025-12-07 | [glm-4.6v-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.6v-flash) | zai-org/GLM-4.6V-Flash |  | `glm4` | 492,389 | replaying the set | `glm45_moe`, none |
| 2025-12-04 | [nvidia-nemotron-3-nano-30b-a3b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3-nano-30b-a3b-bf16) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2025-12-01 | [trinity-mini](https://github.com/smg-project/bellwether/tree/main/fixtures/trinity-mini) | arcee-ai/Trinity-Mini |  | — | — | pending | none |
| 2025-11-28 | [devstral-2-123b-instruct-2512](https://github.com/smg-project/bellwether/tree/main/fixtures/devstral-2-123b-instruct-2512) | mistralai/Devstral-2-123B-Instruct-2512 |  | `mistral_v13` | 570,194 | replaying the set | none |
| 2025-11-25 | [nemotron-orchestrator-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/nemotron-orchestrator-8b) | nvidia/Nemotron-Orchestrator-8B | IQuestLab/Fleming-R1-32B | `qwen3` | 583,086 | replaying the set | none |
| 2025-11-19 | [olmo-3-7b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/olmo-3-7b-instruct) | allenai/Olmo-3-7B-Instruct |  | `olmo3` | 570,199 | replaying the set | none |
| 2025-10-31 | [ministral-3-14b-instruct-2512-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/ministral-3-14b-instruct-2512-bf16) | mistralai/Ministral-3-14B-Instruct-2512-BF16 |  | `mistral_v13` | 570,194 | replaying the set | none |
| 2025-10-31 | [ministral-3-14b-reasoning-2512](https://github.com/smg-project/bellwether/tree/main/fixtures/ministral-3-14b-reasoning-2512) | mistralai/Ministral-3-14B-Reasoning-2512 | Ministral-3-3B-Reasoning-2512, Ministral-3-8B-Reasoning-2512 | `mistral_v13` | 583,079 | replaying the set | none |
| 2025-10-31 | [ministral-3-3b-instruct-2512](https://github.com/smg-project/bellwether/tree/main/fixtures/ministral-3-3b-instruct-2512) | mistralai/Ministral-3-3B-Instruct-2512 |  | `mistral_v13` | 570,194 | replaying the set | none |
| 2025-10-31 | [ministral-3-3b-instruct-2512-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/ministral-3-3b-instruct-2512-bf16) | mistralai/Ministral-3-3B-Instruct-2512-BF16 |  | `mistral_v13` | 570,194 | replaying the set | none |
| 2025-10-31 | [ministral-3-8b-instruct-2512-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/ministral-3-8b-instruct-2512-bf16) | mistralai/Ministral-3-8B-Instruct-2512-BF16 |  | `mistral_v13` | 570,194 | replaying the set | none |
| 2025-10-29 | [falcon-h1r-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/falcon-h1r-7b) | tiiuae/Falcon-H1R-7B |  | `qwen3` | 442,119 | replaying the set | none |
| 2025-10-24 | [audio-flamingo-3-hf](https://github.com/smg-project/bellwether/tree/main/fixtures/audio-flamingo-3-hf) | nvidia/audio-flamingo-3-hf |  | — | 328,662 | pending | none |
| 2025-10-22 | [minimax-m2](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m2) | MiniMaxAI/MiniMax-M2 |  | `minimax_m2` | 12,821 | replaying the set | `minimax_m2`, `minimax` |
| 2025-10-16 | [caprl-internvl3.5-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/caprl-internvl3.5-8b) | internlm/CapRL-InternVL3.5-8B |  | — | 328,662 | pending | none |
| 2025-10-13 | [deep-ignorance-unfiltered-instruct-test](https://github.com/smg-project/bellwether/tree/main/fixtures/deep-ignorance-unfiltered-instruct-test) | EleutherAI/deep-ignorance-unfiltered-instruct-test | deep-ignorance-e2e-strong-filter-instruct-test, deep-ignorance-e2e-strong-filter-instruct-test-v2, deep-ignorance-strong-filter-pt-weak-filter-anneal-instruct-test, deep-ignorance-strong-filter-pt-weak-filter-anneal-instruct-test-v2 | — | 273,678 | pending | none |
| 2025-10-12 | [llama-3.3-nemotron-70b-reward-principle](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.3-nemotron-70b-reward-principle) | nvidia/Llama-3.3-Nemotron-70B-Reward-Principle | Llama-3.3-Nemotron-70B-Edit, Llama-3.3-Nemotron-70B-Feedback | `json` | 458,185 | replaying the set | `json`, none |
| 2025-10-11 | [qwen3-vl-8b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-8b-instruct) | Qwen/Qwen3-VL-8B-Instruct | BAAI/RoboBrain2.5-4B, lightonai/OriOn-Qwen, lightonai/OriOn-Qwen-SR1, nvidia/Cosmos-Reason2-32B, Qwen3-VL-2B-Instruct, Qwen3-VL-32B-Instruct, Qwen3-VL-4B-Instruct, tencent/WeVisDoc-2B, tencent/WeVisDoc-4B | `qwen3` | 576,772 | replaying the set | `qwen`, `qwen3` |
| 2025-10-11 | [qwen3-vl-8b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-8b-thinking) | Qwen/Qwen3-VL-8B-Thinking | Qwen3-VL-2B-Thinking, Qwen3-VL-32B-Thinking, Qwen3-VL-4B-Thinking | `qwen3` | 554,276 | replaying the set | `qwen`, `qwen3` |
| 2025-10-08 | [functiongemma-270m-it](https://github.com/smg-project/bellwether/tree/main/fixtures/functiongemma-270m-it) | google/functiongemma-270m-it |  | `function_gemma` | 246,750 | replaying the set | `json`, none |
| 2025-10-05 | [ai21-jamba-reasoning-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/ai21-jamba-reasoning-3b) | ai21labs/AI21-Jamba-Reasoning-3B |  | `qwen3` | 12,885 | replaying the set | none |
| 2025-10-01 | [llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad) | nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD |  | — | 328,662 | pending | `json`, `nano_v3` |
| 2025-09-30 | [qwen3-4b-saferl](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-saferl) | Qwen/Qwen3-4B-SafeRL |  | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-30 | [qwen3-vl-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-30b-a3b-instruct) | Qwen/Qwen3-VL-30B-A3B-Instruct | Qwen3-VL-235B-A22B-Instruct | `qwen3` | 576,772 | replaying the set | `qwen`, `qwen3` |
| 2025-09-29 | [deepseek-v3.2-exp](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3.2-exp) | deepseek-ai/DeepSeek-V3.2-Exp | DeepSeek-V3.2-Exp-Base | — | 184,972 | pending | `deepseek31`, none |
| 2025-09-29 | [glm-4.6](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.6) | zai-org/GLM-4.6 | GLM-4.5, GLM-4.5-Air | `glm4` | 492,389 | replaying the set | `glm45_moe`, none |
| 2025-09-25 | [caprl-eval-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/caprl-eval-3b) | internlm/CapRL-Eval-3B |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-09-24 | [caprl-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/caprl-3b) | internlm/CapRL-3B |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-09-23 | [qwen3guard-gen-0.6b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3guard-gen-0.6b) | Qwen/Qwen3Guard-Gen-0.6B | Qwen3Guard-Gen-4B, Qwen3Guard-Gen-8B | — | — | pending | `qwen`, `qwen3` |
| 2025-09-22 | [qwen3-vl-235b-a22b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-235b-a22b-thinking) | Qwen/Qwen3-VL-235B-A22B-Thinking | Qwen3-VL-30B-A3B-Thinking | `qwen3` | 554,276 | replaying the set | `qwen`, `qwen3` |
| 2025-09-20 | [qwen3-omni-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-omni-30b-a3b-instruct) | Qwen/Qwen3-Omni-30B-A3B-Instruct |  | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-15 | [qwen3-omni-30b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-omni-30b-a3b-thinking) | Qwen/Qwen3-Omni-30B-A3B-Thinking |  | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-10 | [bytedance-seed-oss-36b-alternate-tokenizer](https://github.com/smg-project/bellwether/tree/main/fixtures/bytedance-seed-oss-36b-alternate-tokenizer) | NousResearch/ByteDance-Seed-OSS-36B-Alternate-Tokenizer |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-09-09 | [qwen3-next-80b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-next-80b-a3b-instruct) | Qwen/Qwen3-Next-80B-A3B-Instruct |  | `qwen3` | 576,772 | replaying the set | `qwen`, `qwen3` |
| 2025-09-09 | [qwen3-next-80b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-next-80b-a3b-thinking) | Qwen/Qwen3-Next-80B-A3B-Thinking |  | `qwen3` | 554,276 | replaying the set | `qwen`, `qwen3` |
| 2025-09-08 | [ernie-4.5-21b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/ernie-4.5-21b-a3b-thinking) | baidu/ERNIE-4.5-21B-A3B-Thinking |  | — | — | pending | none |
| 2025-09-01 | [apertus-70b-instruct-2509](https://github.com/smg-project/bellwether/tree/main/fixtures/apertus-70b-instruct-2509) | swiss-ai/Apertus-70B-Instruct-2509 |  | — | 137,178 | pending | none |
| 2025-08-30 | [hermes-4-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/hermes-4-14b) | NousResearch/Hermes-4-14B |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-08-21 | [deepseek-v3.1](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3.1) | deepseek-ai/DeepSeek-V3.1 | DeepSeek-V3.1-Base, tencent/DeepSeek-V3.1-Terminus-W4AFP8 | — | 19,299 | pending | `deepseek31`, `deepseek_v31` |
| 2025-08-20 | [llama-3.1-nemotron-safety-guard-8b-v3](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-safety-guard-8b-v3) | nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 |  | `json` | 458,185 | replaying the set | `json`, none |
| 2025-08-20 | [seed-oss-36b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/seed-oss-36b-instruct) | ByteDance-Seed/Seed-OSS-36B-Instruct | Seed-OSS-36B-Base-woSyn | `seed_oss` | 495,977 | replaying the set | none |
| 2025-08-13 | [apertus-8b-instruct-2509](https://github.com/smg-project/bellwether/tree/main/fixtures/apertus-8b-instruct-2509) | swiss-ai/Apertus-8B-Instruct-2509 |  | — | 137,178 | pending | none |
| 2025-08-11 | [dler-llama-nemotron-8b-merge-research](https://github.com/smg-project/bellwether/tree/main/fixtures/dler-llama-nemotron-8b-merge-research) | nvidia/DLER-Llama-Nemotron-8B-Merge-Research |  | `llama_nemotron` | 416,850 | replaying the set | `json`, none |
| 2025-08-11 | [dler-r1-1.5b-research](https://github.com/smg-project/bellwether/tree/main/fixtures/dler-r1-1.5b-research) | nvidia/DLER-R1-1.5B-Research | DLER-R1-7B-Research | — | 94,830 | pending | none |
| 2025-08-10 | [glm-4.5v](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.5v) | zai-org/GLM-4.5V |  | `glm4` | 492,389 | replaying the set | `glm45_moe`, none |
| 2025-08-06 | [dogr](https://github.com/smg-project/bellwether/tree/main/fixtures/dogr) | tencent/DOGR |  | — | 328,662 | pending | none |
| 2025-08-05 | [qwen3-4b-instruct-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-instruct-2507) | Qwen/Qwen3-4B-Instruct-2507 |  | `qwen3` | 41,533 | ready | `qwen`, `qwen3` |
| 2025-08-05 | [qwen3-4b-thinking-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-thinking-2507) | Qwen/Qwen3-4B-Thinking-2507 |  | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-07-31 | [qwen3-coder-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-coder-30b-a3b-instruct) | Qwen/Qwen3-Coder-30B-A3B-Instruct | Qwen3-Coder-480B-A35B-Instruct | `qwen3`, tagged calls | 490,451 | replaying the set | `qwen_xml`, `qwen3` |
| 2025-07-29 | [afm-4.5b](https://github.com/smg-project/bellwether/tree/main/fixtures/afm-4.5b) | arcee-ai/AFM-4.5B |  | — | 328,662 | pending | none |
| 2025-07-29 | [qwen3-30b-a3b-thinking-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b-thinking-2507) | Qwen/Qwen3-30B-A3B-Thinking-2507 | NousResearch/nomos-1, Qwen3-235B-A22B-Thinking-2507 | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-07-28 | [qwen3-30b-a3b-instruct-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b-instruct-2507) | Qwen/Qwen3-30B-A3B-Instruct-2507 | Qwen3-235B-A22B-Instruct-2507 | `qwen3` | 41,533 | ready | `qwen`, `qwen3` |
| 2025-07-28 | [step3](https://github.com/smg-project/bellwether/tree/main/fixtures/step3) | stepfun-ai/step3 |  | — | — | pending | `step3`, `step3` |
| 2025-07-25 | [llama-3-3-nemotron-super-49b-v1-5](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3-3-nemotron-super-49b-v1-5) | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 |  | `llama_nemotron` | 576,772 | replaying the set | `json`, `nano_v3` |
| 2025-07-15 | [openreasoning-nemotron-1.5b](https://github.com/smg-project/bellwether/tree/main/fixtures/openreasoning-nemotron-1.5b) | nvidia/OpenReasoning-Nemotron-1.5B | OpenReasoning-Nemotron-32B, OpenReasoning-Nemotron-7B | — | 328,662 | pending | none |
| 2025-07-15 | [openreasoning-nemotron-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/openreasoning-nemotron-14b) | nvidia/OpenReasoning-Nemotron-14B |  | — | 328,662 | pending | none |
| 2025-07-10 | [a.x-3.1-light](https://github.com/smg-project/bellwether/tree/main/fixtures/a.x-3.1-light) | skt/A.X-3.1-Light |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-07-02 | [a.x-4.0](https://github.com/smg-project/bellwether/tree/main/fixtures/a.x-4.0) | skt/A.X-4.0 |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-07-02 | [a.x-4.0-light](https://github.com/smg-project/bellwether/tree/main/fixtures/a.x-4.0-light) | skt/A.X-4.0-Light |  | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-06-25 | [hunyuan-a13b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/hunyuan-a13b-instruct) | tencent/Hunyuan-A13B-Instruct |  | — | 328,662 | pending | none |
| 2025-06-09 | [caller](https://github.com/smg-project/bellwether/tree/main/fixtures/caller) | arcee-ai/Caller |  | `qwen2.5` | 324,160 | replaying the set | none |
| 2025-06-03 | [areal-boba-2-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-boba-2-14b) | inclusionAI/AReaL-boba-2-14B |  | `qwen3` | 583,086 | replaying the set | none |
| 2025-06-03 | [areal-boba-2-32b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-boba-2-32b) | inclusionAI/AReaL-boba-2-32B |  | `qwen3` | 583,086 | replaying the set | none |
| 2025-06-03 | [areal-boba-2-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/areal-boba-2-8b) | inclusionAI/AReaL-boba-2-8B | AReaL-boba-2-14B-Open, AReaL-boba-2-8B-Open | `qwen3` | 583,086 | replaying the set | none |
| 2025-06-03 | [granite-vision-3.3-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-vision-3.3-2b) | ibm-granite/granite-vision-3.3-2b |  | — | 328,662 | pending | none |
| 2025-05-22 | [acereason-nemotron-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/acereason-nemotron-7b) | nvidia/AceReason-Nemotron-7B | AceMath-RL-Nemotron-7B, AceReason-Nemotron-14B, Nemotron-Research-Reasoning-Qwen-1.5B | — | 94,830 | pending | none |
| 2025-05-19 | [e1-code-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/e1-code-14b) | Salesforce/E1-Code-14B |  | — | 94,830 | pending | none |
| 2025-05-03 | [llama-3.1-nemotron-nano-4b-v1.1](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-nano-4b-v1.1) | nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 |  | — | 576,772 | pending | `json`, `nano_v3` |
| 2025-04-28 | [deephermes-financial-fundamentals-prediction-specialist-atropos](https://github.com/smg-project/bellwether/tree/main/fixtures/deephermes-financial-fundamentals-prediction-specialist-atropos) | NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos | DeepHermes-AscensionMaze-RLAIF-8b-Atropos, DeepHermes-Egregore-v1-RLAIF-8b-Atropos, DeepHermes-Egregore-v2-RLAIF-8b-Atropos, DeepHermes-ToolCalling-Specialist-Atropos | — | 137,178 | pending | none |
| 2025-04-27 | [qwen3-30b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b) | Qwen/Qwen3-30B-A3B | Qwen3-235B-A22B | `qwen3` | 583,086 | replaying the set | `qwen`, `qwen3` |
| 2025-04-27 | [qwen3-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-8b) | Qwen/Qwen3-8B | NousResearch/NousCoder-14B, Qwen3-0.6B, Qwen3-1.7B, Qwen3-14B, Qwen3-32B, Qwen3-4B | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-04-24 | [crispers-7b-v1](https://github.com/smg-project/bellwether/tree/main/fixtures/crispers-7b-v1) | thu-coai/Crispers-7B-v1 | Crispers-14B-v1 | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-04-22 | [openmath-nemotron-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/openmath-nemotron-14b) | nvidia/OpenMath-Nemotron-14B | OpenMath-Nemotron-1.5B, OpenMath-Nemotron-32B, OpenMath-Nemotron-7B | — | 328,662 | pending | none |
| 2025-04-22 | [openmath-nemotron-14b-kaggle](https://github.com/smg-project/bellwether/tree/main/fixtures/openmath-nemotron-14b-kaggle) | nvidia/OpenMath-Nemotron-14B-Kaggle |  | — | 328,662 | pending | none |
| 2025-04-18 | [cosmos-reason1-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/cosmos-reason1-7b) | nvidia/Cosmos-Reason1-7B |  | — | 328,662 | pending | none |
| 2025-04-15 | [opencodereasoning-nemotron-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/opencodereasoning-nemotron-7b) | nvidia/OpenCodeReasoning-Nemotron-7B | OpenCodeReasoning-Nemotron-1.1-14B, OpenCodeReasoning-Nemotron-1.1-32B, OpenCodeReasoning-Nemotron-1.1-7B, OpenCodeReasoning-Nemotron-14B, OpenCodeReasoning-Nemotron-32B, OpenCodeReasoning-Nemotron-32B-IOI | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-04-09 | [academic-ds-9b](https://github.com/smg-project/bellwether/tree/main/fixtures/academic-ds-9b) | ByteDance-Seed/academic-ds-9B |  | — | 3,114 | pending | none |
| 2025-04-08 | [gemma-3-12b-it-qat-q4-0-unquantized](https://github.com/smg-project/bellwether/tree/main/fixtures/gemma-3-12b-it-qat-q4-0-unquantized) | google/gemma-3-12b-it-qat-q4_0-unquantized | gemma-3-27b-it-qat-q4_0-unquantized, gemma-3-4b-it-qat-q4_0-unquantized | — | 184,969 | pending | `json`, none |
| 2025-04-08 | [llama-3-1-nemotron-ultra-253b-cpt-v1](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3-1-nemotron-ultra-253b-cpt-v1) | nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1 |  | `json` | 458,185 | replaying the set | `json`, none |
| 2025-04-07 | [baichuanmed-ocr-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/baichuanmed-ocr-7b) | baichuan-inc/BaichuanMed-OCR-7B | BaichuanMed-OCR-72B | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-04-07 | [llama-3-1-nemotron-ultra-253b-v1](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3-1-nemotron-ultra-253b-v1) | nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 |  | — | 328,643 | pending | `json`, none |
| 2025-03-27 | [llama-xlam-2-8b-fc-r](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-xlam-2-8b-fc-r) | Salesforce/Llama-xLAM-2-8b-fc-r |  | `xlam` | 483,062 | replaying the set | `json`, none |
| 2025-03-26 | [granite-speech-3.2-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-speech-3.2-8b) | ibm-granite/granite-speech-3.2-8b |  | — | 328,662 | pending | none |
| 2025-03-24 | [deepseek-v3-0324](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3-0324) | deepseek-ai/DeepSeek-V3-0324 | DeepSeek-Prover-V2-671B | — | 19,299 | pending | `deepseek`, none |
| 2025-03-24 | [deepseek-v3-0324-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3-0324-bf16) | arcee-ai/DeepSeek-V3-0324-bf16 |  | — | 3,110 | pending | `deepseek`, none |
| 2025-03-22 | [qwen2.5-omni-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-omni-7b) | Qwen/Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | `qwen2.5` | 328,662 | replaying the set | `qwen`, `qwen3` |
| 2025-03-21 | [qwen2.5-vl-32b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-vl-32b-instruct) | Qwen/Qwen2.5-VL-32B-Instruct |  | `qwen2.5` | 328,662 | replaying the set | `qwen`, `qwen3` |
| 2025-03-16 | [llama-3.1-nemotron-nano-8b-v1](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-nano-8b-v1) | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 |  | `llama_nemotron` | 416,850 | replaying the set | `json`, `nano_v3` |
| 2025-03-10 | [gemma-3-1b-it](https://github.com/smg-project/bellwether/tree/main/fixtures/gemma-3-1b-it) | google/gemma-3-1b-it |  | — | 184,969 | pending | `json`, none |
| 2025-03-05 | [qwq-32b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwq-32b) | Qwen/QwQ-32B |  | — | — | pending | `qwen`, `qwen3` |
| 2025-03-04 | [llama-3.1-nemotron-8b-ultralong-1m-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-8b-ultralong-1m-instruct) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct |  | `json` | 458,185 | replaying the set | `json`, none |
| 2025-03-04 | [llama-3.1-nemotron-8b-ultralong-2m-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-8b-ultralong-2m-instruct) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct |  | `json` | 458,185 | replaying the set | `json`, none |
| 2025-03-04 | [llama-3.1-nemotron-8b-ultralong-4m-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-3.1-nemotron-8b-ultralong-4m-instruct) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct |  | `json` | 458,185 | replaying the set | `json`, none |
| 2025-02-24 | [phi-4-multimodal-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/phi-4-multimodal-instruct) | microsoft/Phi-4-multimodal-instruct |  | — | 328,662 | pending | none |
| 2025-02-20 | [gemma-3-4b-it](https://github.com/smg-project/bellwether/tree/main/fixtures/gemma-3-4b-it) | google/gemma-3-4b-it | gemma-3-12b-it, gemma-3-27b-it | — | 184,969 | pending | `json`, none |
| 2025-02-19 | [phi-4-mini-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/phi-4-mini-instruct) | microsoft/Phi-4-mini-instruct |  | `plain` | 328,662 | replaying the set | none |
| 2025-02-17 | [cogalign-internvl2-5-mpo-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/cogalign-internvl2-5-mpo-4b) | Salesforce/cogalign-internvl2_5-mpo-4b | cogalign-internvl2_5-mpo-1b | `qwen2.5` | 576,772 | replaying the set | none |
| 2025-02-17 | [cogalign-llava-ov-0-5b](https://github.com/smg-project/bellwether/tree/main/fixtures/cogalign-llava-ov-0-5b) | Salesforce/cogalign-llava-ov-0_5b |  | — | 328,662 | pending | none |
| 2025-02-17 | [granite-vision-3.2-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-vision-3.2-2b) | ibm-granite/granite-vision-3.2-2b |  | — | 328,662 | pending | none |
| 2025-01-28 | [mistral-small-24b-instruct-2501](https://github.com/smg-project/bellwether/tree/main/fixtures/mistral-small-24b-instruct-2501) | mistralai/Mistral-Small-24B-Instruct-2501 |  | — | 184,972 | pending | `mistral`, none |
| 2025-01-27 | [granite-vision-3.1-2b-preview](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-vision-3.1-2b-preview) | ibm-granite/granite-vision-3.1-2b-preview |  | — | 328,662 | pending | none |
| 2025-01-26 | [qwen2.5-vl-7b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-vl-7b-instruct) | Qwen/Qwen2.5-VL-7B-Instruct | allenai/olmOCR-7B-0725, allenai/olmOCR-7B-0825, Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-72B-Instruct | `qwen2.5` | 328,662 | replaying the set | `qwen`, `qwen3` |
| 2025-01-23 | [qwen2.5-7b-instruct-1m](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-7b-instruct-1m) | Qwen/Qwen2.5-7B-Instruct-1M | Qwen2.5-14B-Instruct-1M | `qwen2.5` | 576,772 | replaying the set | `qwen`, `qwen3` |
| 2025-01-20 | [deepseek-r1](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-r1) | deepseek-ai/DeepSeek-R1 | DeepSeek-R1-Zero | — | — | pending | `pythonic`, `deepseek_r1` |
| 2025-01-20 | [deepseek-r1-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-r1-bf16) | arcee-ai/DeepSeek-R1-bf16 |  | — | 3,110 | pending | `pythonic`, `deepseek_r1` |
| 2024-05-22 | [mistral-7b-instruct-v0.3](https://github.com/smg-project/bellwether/tree/main/fixtures/mistral-7b-instruct-v0.3) | mistralai/Mistral-7B-Instruct-v0.3 |  | — | — | pending | `mistral`, none |
| 2023-12-30 | [tinyllama-1.1b-chat-v1.0](https://github.com/smg-project/bellwether/tree/main/fixtures/tinyllama-1.1b-chat-v1.0) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |  | `plain` | 27,865 | ready | `json`, none |
| 2023-12-05 | [llava-1.5-7b-hf](https://github.com/smg-project/bellwether/tree/main/fixtures/llava-1.5-7b-hf) | llava-hf/llava-1.5-7b-hf |  | — | — | pending | none |
<!-- models: end -->

Why the crate has a name: the community has spent years on many parser implementations, and none
of them is right the way we want it, SMG's own two crates included. Not right as working code;
right as a made thing. This crate is the attempt to write the one that is. Code here is a craft,
and "this can be better" is always a good enough reason for a change.

Rules this crate works under:

- It changes nothing outside `crates/symphony/`. It depends on other crates but never edits them;
  wiring it into the gateway is a separate step taken once recorded fixtures show parity.
- Formats are data. A format definition yields both the parser and the guided-decoding grammar;
  hand-written parsers are the exception and say why.
- Every format ships with fixtures recorded from the vendor's reference and the engines, replayed
  at many chunkings, and five property tests: conservation of bytes, prefix-stable arguments,
  chunking invariance, committed output stays committed, and token identity for markers.
- Every commit carries `Co-authored-by: Chang Su`, whose design this is.
