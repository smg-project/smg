//! The seam's test support: the requests, names, templates and tables the tests of its modules
//! build, and the wrappers that read with no templates.

pub(super) use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub(super) use llm_tokenizer::{registry::TokenizerRegistry, traits::Tokenizer};
pub(super) use openai_protocol::{
    chat::ChatCompletionRequest, common::Tool, messages::CreateMessageRequest,
};
pub(super) use serde_json::{json, Value};
pub(super) use symphony::{formats::qwen3, CallSyntax, EngineFinish, Event, Events, Format};
pub(super) use tool_parser::ToolConstraint;

pub(super) use super::{
    constraint::*, names::*, reader::*, telemetry::Who, template::Templates, *,
};
pub(super) use crate::{
    routers::grpc::{
        spec::scans_for_tool_calls,
        utils::{ModelNames, Names, TagVocabulary},
    },
    worker::BasicWorkerBuilder,
};
pub(super) fn request(extra: Value) -> ChatCompletionRequest {
    let mut request = json!({"model": "Qwen/Qwen3-8B", "messages": [], "stream": true});
    for (key, value) in extra.as_object().into_iter().flatten() {
        request[key] = value.clone();
    }
    serde_json::from_value(request).expect("chat request")
}

pub(super) fn tools() -> Vec<Tool> {
    serde_json::from_value(json!([{
        "type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {}}
        }
    }]))
    .expect("tools")
}

/// The request the tests' readers report under.
pub(super) fn who() -> Who<'static> {
    Who {
        request_id: "test-request",
        model: "Qwen/Qwen3-8B",
        choice: 0,
    }
}

pub(super) const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// One source's names.
pub(super) fn pair(tool: Option<&str>, reasoning: Option<&str>) -> Names {
    Names {
        tool: tool.map(str::to_string),
        reasoning: reasoning.map(str::to_string),
    }
}

/// The names a model's workers label it with, and no flags.
pub(super) fn names(tool: Option<&str>, reasoning: Option<&str>) -> ModelNames<'static> {
    ModelNames::labeled(pair(tool, reasoning))
}

/// No tokenizer registered: the response template's table is never there.
pub(super) fn templates() -> Templates {
    Templates::new(Arc::new(TokenizerRegistry::new()))
}

/// [`super::check_names`] of the labels, against the legacy registries as the gateway builds
/// them.
pub(super) fn check(parser: ParserChoice, names: &ModelNames<'_>) -> Result<(), String> {
    let tools = tool_parser::ParserFactory::new();
    let reasoning = reasoning_parser::ParserFactory::new();
    check_names(parser, &names.labels, Some(&tools), Some(&reasoning)).map_err(String::from)
}

/// [`super::assign`] with no templates: the names never say `hf`, so nothing is refused.
pub(super) fn assign(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    request: &ChatCompletionRequest,
    constraint: Option<&(String, String)>,
    prompt: &str,
) -> Option<Assignment> {
    super::assign(
        parser,
        names,
        Asks::Chat(request),
        constraint,
        prompt,
        &templates(),
    )
    .expect("no hf")
}

/// [`super::constraint`](fn@super::constraint) with no templates, as [`assign`]; a JSON answer's
/// tag as a worker that does not say where it starts the grammar takes it.
pub(super) fn constraint(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    request: &ChatCompletionRequest,
    tools: &[Tool],
    reasoning_open: bool,
) -> Option<ToolConstraint> {
    super::constraint(
        parser,
        names,
        Asks::Chat(request),
        tools,
        reasoning_open,
        false,
        symphony::GrammarStart::Unknown,
        TagVocabulary::Current,
        &templates(),
    )
    .and_then(|prepared| {
        prepared.for_start(symphony::GrammarStart::Unknown, TagVocabulary::Current)
    })
    .expect("no hf")
}

/// The conversation of a retry: the user asked for `list_skills`, the assistant called it,
/// the tool was unavailable, the user asks again.
pub(super) fn retry_messages() -> Value {
    json!([
        {"role": "user", "content": "Call the `list_skills` tool once to see what we can do, \
                                     and reply with nothing else."},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "list_skills", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "call_0",
         "content": "`list_skills` is unavailable for now; call it again later."},
        {"role": "user", "content": "The cooldown is over. Try again."}
    ])
}

/// A tokenizer whose decoder drops `<eos>` and `<bos>` as special tokens and keeps every
/// other token, the markers of two tables among them.
pub(super) fn marker_tokenizer() -> llm_tokenizer::MockTokenizer {
    llm_tokenizer::MockTokenizer::new().with_tokens(&[
        ("<eos>", 1),
        ("<bos>", 2),
        ("<think>", 3),
        ("</think>", 4),
    ])
}

pub(super) fn table_with_markers(open: &str, close: &str) -> Format {
    Format::new("markers")
        .terminal("open", open)
        .terminal("close", close)
        .state("content", symphony::Emits::Content)
        .state("reasoning", symphony::Emits::Reasoning)
        .transition("content", "open", "reasoning")
        .transition("reasoning", "close", "content")
}

/// The marker tokenizer, counting the encodes asked of it.
pub(super) struct Counting {
    inner: llm_tokenizer::MockTokenizer,
    encodes: AtomicUsize,
}

impl Counting {
    pub(super) fn shared() -> (Arc<Self>, Arc<dyn Tokenizer>) {
        let counting = Arc::new(Self {
            inner: marker_tokenizer(),
            encodes: AtomicUsize::new(0),
        });
        let tokenizer: Arc<dyn Tokenizer> = counting.clone();
        (counting, tokenizer)
    }

    pub(super) fn encodes(&self) -> usize {
        self.encodes.load(Ordering::Relaxed)
    }
}

impl llm_tokenizer::traits::Encoder for Counting {
    fn encode(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> anyhow::Result<llm_tokenizer::traits::Encoding> {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        self.inner.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<llm_tokenizer::traits::Encoding>> {
        self.inner.encode_batch(inputs, add_special_tokens)
    }
}

impl llm_tokenizer::traits::Decoder for Counting {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for Counting {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) fn message_request(extra: Value) -> CreateMessageRequest {
    let mut request = json!({
        "model": "Qwen/Qwen3-8B", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    });
    for (key, value) in extra.as_object().into_iter().flatten() {
        request[key] = value.clone();
    }
    serde_json::from_value(request).expect("messages request")
}

/// The rendered scrape after `record` ran under a recorder of its own.
pub(super) fn rendered(record: impl FnOnce()) -> String {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, record);
    handle.render()
}

/// The sample line of `scrape` under `name` that carries every one of `labels`, if one does.
pub(super) fn line_with<'a>(scrape: &'a str, name: &str, labels: &[&str]) -> Option<&'a str> {
    scrape
        .lines()
        .find(|line| line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
}

pub(super) fn tag_of(constraint: Result<Option<ToolConstraint>, SymphonyError>) -> Value {
    match constraint {
        Ok(Some(ToolConstraint::StructuralTag(tag))) => {
            serde_json::from_str(&tag).expect("a tag as JSON")
        }
        other => panic!("a structural tag, not {other:?}"),
    }
}

/// A Qwen3 output: a thought, content, and one call of the declared `lookup` tool.
pub(super) const QWEN3_OUTPUT: &str = "<think>plan</think>\n\nHello <tool_call>\n\
                            {\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>";

pub(super) fn read_with(format: Format) -> Vec<Event<'static>> {
    let assignment = Assignment {
        format: Arc::new(format),
        prompt: "<|im_start|>assistant\n".to_string(),
        forbids_calls: false,
    };
    let mut events = Events::new();
    let mut reader = Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
    read_tail(&mut reader, QWEN3_OUTPUT)
}

/// The events of `output` read whole as the decoder's last text, then ended by a stop, owned.
pub(super) fn read_tail(reader: &mut Reader, output: &str) -> Vec<Event<'static>> {
    let mut delta = Delta::default();
    let mut events = Events::new();
    reader
        .tail(output.to_string(), &mut delta, &mut events)
        .expect("read");
    reader
        .end(EngineFinish::Stop, who(), &mut events)
        .expect("ended");
    events.into_owned().drain()
}

pub(super) fn content_of(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Content(text) => Some(&*text.text),
            _ => None,
        })
        .collect()
}

/// A table whose calls spell a whole turn: a call's header names the tool, its parameters
/// are tags around raw text, and a row out of a call goes on to the next or stops.
pub(super) fn turn_table() -> Format {
    let source = json!({
      "start_anchor_pattern": "<\\|head\\|>assistant",
      "fields": {
        "thinking": {"open": "<|via|>notes<|body|>", "close": "<|stop|><|head|>assistant",
                     "content": "text"},
        "content": {"open": "<|via|>reply<|body|>", "close": ["<|done|>", "<|stop|>", "<eos>"],
                    "content": "text"},
        "tool_calls": {
          "open_pattern": " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
          "close": ["<|stop|><|head|>assistant", "<eos>"], "repeats": true,
          "content": "xml-inline",
          "content_args": {
            "tag_pattern":
              "<arg name=\"(?P<key>[^\"]+)\">\\n(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))",
            "value_parser": {"name": "text", "args": {"strip": false}}
          },
          "transform": {"type": "function",
                        "function": {"name": "{name}", "arguments": "{content}"}}
        }
      }
    });
    let template = smg_response_template::load_response_template(&source).expect("template");
    smg_response_template::to_symphony::compile(&template, &["<|done|>", "<eos>"])
        .expect("compiled")
        .format
}

pub(super) fn weather() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

/// The format of the whole turn's grammar a chat request takes over [`turn_table`], the prompt
/// ending at the turn's opener, then at `continued` when the request continues an assistant
/// message; the error is the refusal.
pub(super) fn turn(mut extra: Value, continued: Option<&str>) -> Result<Option<Value>, String> {
    if continued.is_some() {
        extra["continue_final_message"] = json!(true);
        extra["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "partial"}
        ]);
    }
    let prompt = format!("user<|head|>assistant{}", continued.unwrap_or(""));
    let request = request(extra);
    let constraint = turn_of(
        &turn_table(),
        Asks::Chat(&request),
        &request.effective_tool_list(),
        &prompt,
        TagVocabulary::Current,
    )?;
    Ok(constraint.map(|constraint| tag_of(Ok(Some(constraint)))["format"].clone()))
}

pub(super) fn whole(extra: Value, continued: Option<&str>) -> Value {
    turn(extra, continued).expect("held").expect("a grammar")
}

/// The part of a turn's format after its thought: the answer, the calls, or either.
pub(super) fn after_thought(format: &Value) -> &Value {
    &format["elements"][1]
}

/// A character tokenizer (a token is a character's code point) whose decoder refuses where a
/// test says: the step that decodes the id no character carries ([`Self::REFUSED`]) fails, and
/// under [`Self::the_end`] the decoder's end fails, where the text it held would go out.
#[derive(Default)]
pub(crate) struct Refusing {
    at_the_end: bool,
    special_tokens: llm_tokenizer::traits::SpecialTokens,
}

impl Refusing {
    /// An id no character carries (a surrogate): the step that decodes it fails.
    pub(crate) const REFUSED: u32 = 0xD800;

    /// A tokenizer whose decoder's end fails.
    pub(crate) fn the_end() -> Self {
        Self {
            at_the_end: true,
            ..Self::default()
        }
    }
}

impl llm_tokenizer::traits::Encoder for Refusing {
    fn encode(
        &self,
        input: &str,
        _add_special_tokens: bool,
    ) -> anyhow::Result<llm_tokenizer::traits::Encoding> {
        Ok(llm_tokenizer::traits::Encoding::Plain(
            input.chars().map(u32::from).collect(),
        ))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<llm_tokenizer::traits::Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl llm_tokenizer::traits::Decoder for Refusing {
    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        token_ids
            .iter()
            .map(|id| {
                char::from_u32(*id)
                    .ok_or_else(|| anyhow::anyhow!("no character carries token {id}"))
            })
            .collect()
    }

    fn finish_decode(
        &self,
        _ids: &mut Vec<u32>,
        _prefix: &mut String,
        _prefix_index: &mut usize,
        _skip_special_tokens: bool,
    ) -> anyhow::Result<llm_tokenizer::traits::DecodeStep> {
        if self.at_the_end {
            anyhow::bail!("the decoder's end refused the text it held");
        }
        Ok(llm_tokenizer::traits::DecodeStep::default())
    }
}

impl Tokenizer for Refusing {
    fn vocab_size(&self) -> usize {
        128
    }

    fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.chars().next().map(u32::from)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
