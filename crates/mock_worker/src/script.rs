//! Scripted output: what the model says, stated ahead of the request, on
//! every wire the worker speaks (HTTP, gRPC, ZMQ). One store serves them all;
//! each wire renders the answer's frames in its own shape.
//!
//! Two sources:
//! - `--script PATH`: a JSONL of answers keyed by the prompt's token ids, for
//!   recorded corpora and bisects (one line per answer, see [`Answer::parse`]);
//! - `POST /admin/script/{worker}`: answers enqueued in order for the next
//!   requests of one worker, or of every worker (`all`), for a test that
//!   states the model's answer right before it sends the request; an answer
//!   given a `prompt_contains` text is filed for the first request whose
//!   prompt holds the text instead, whatever order the requests arrive in
//!   (concurrent requests, or a fleet whose worker the gateway picks).
//!
//! For a request a worker looks in this order: its own queue, the shared
//! queue, the answers filed under a text its prompt holds (its own, then the
//! shared ones, each served once), the keyed answers under its prompt's key.
//! Nothing found is a miss, counted per worker, and the request goes on to
//! the worker's canned or realistic engine as if no script existed. A
//! scripted answer is served by the wire, not by the engine simulator: it
//! adds no load, publishes no KV events and leaves no request record.
//!
//! The prompt's key is `<len>:<fold>` with `fold` the FNV-style fold of the
//! prompt's token ids: `h = (h * 1099511628211) ^ id` over a `u64` starting at
//! 0, wrapping. gRPC and ZMQ workers key on the ids the gateway sent; an HTTP
//! worker keys on its synthetic ids, one per whitespace word of the prompt
//! text, the ids its realistic engine runs on. The prompt's text, for the
//! answers filed under one, is the text an HTTP worker was sent, and the ids
//! decoded with the worker's tokenizer on the token wires: read only while
//! such answers are filed for the worker.
//!
//! A recorded line (`--record` writes them) states under `recorded.asked`
//! what its request asked beyond the prompt, and is filed under the prompt
//! AND that: `<len>:<fold>/<ask>`, with `ask` the same fold over the object's
//! text (`ask_fold`). The lines of one such key are served in turn, in file
//! order, and from the first again; a line without `recorded.asked` is filed
//! under its prompt alone, serves that prompt whatever is asked, and the last
//! such line of a key is the one served. Once a recorded line is in the
//! store, a gRPC worker fails a request that finds nothing with the key it
//! missed (`INTERNAL`, the code of a request an engine took and failed on)
//! instead of going on to its engine: a recording is replayed whole, or the
//! test learns what was not recorded. The other wires do not read what a
//! request asks, so no recorded line serves them. A recorded line names the
//! gRPC wire it was read off (`recorded.wire`): a gRPC worker of that wire
//! reports the completion count the line states, and one of another gRPC
//! wire never a count below the line's output ids.

use std::{
    cell::Cell,
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
    time::Duration,
};

use llm_tokenizer::traits::Tokenizer;
use serde_json::{json, Value};

use crate::config::{Config, SchedulerWire};

/// The queue every worker falls back to, and the `{worker}` that names it.
pub const ALL: &str = "all";

/// The stop an output matched, when it stopped on one: the stop token's id
/// or the stop text.
#[derive(Clone, Debug, PartialEq)]
pub enum MatchedStop {
    Token(u32),
    Text(String),
}

impl MatchedStop {
    /// The JSON an HTTP answer carries in its choice (`matched_stop`).
    pub fn to_json(&self) -> Value {
        match self {
            Self::Token(id) => json!(id),
            Self::Text(text) => json!(text),
        }
    }
}

/// One scripted answer: the output and how it is delivered.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    /// The output's token ids, in order (what the token wires stream).
    pub output_ids: Vec<u32>,
    /// The output's text, as pieces: one per output id when given as a list
    /// alongside `output_ids`, or the whole text as one piece (what the HTTP
    /// wire streams).
    pub output_text: Option<Vec<String>>,
    /// `stop`, `length`, `abort`, or whatever the wire's reader expects.
    pub finish_reason: String,
    pub matched_stop: Option<MatchedStop>,
    /// Prompt tokens reported as served from cache.
    pub cached_tokens: u32,
    /// Prompt tokens reported; the request's own count when unset.
    pub prompt_tokens: Option<u32>,
    /// Completion tokens reported (`completion_tokens`), as an engine counted
    /// them: one more than the output where it took a matched stop token
    /// off. Unset: the output's own count.
    pub completion_tokens: Option<u32>,
    /// The sampled token's logprob per output token, sent when the request
    /// asks for logprobs.
    pub logprobs: Option<Vec<f32>>,
    /// `"logprobs": "sampled"`: the logprobs are made up, one per output id,
    /// once the ids are known (the encode of a text answer), so a scripted
    /// text can answer a request that asks for logprobs on a token wire.
    pub sampled_logprobs: bool,
    /// The logprob of each token id the request asks a score for
    /// (`token_ids_logprob` on the SGLang wire), in the request's order: the
    /// answer of a prefill-only scoring request, whose output is empty. The
    /// request asks for logprobs too, as the engine computes the row only
    /// then; one that does not is refused.
    pub token_id_logprobs: Option<Vec<f32>>,
    /// The pause before every frame but the first.
    pub delay: Duration,
    /// Tokens per frame.
    pub chunk_tokens: usize,
    /// The tokens of each frame in turn, as an engine cut a recorded answer
    /// (`chunks`); they add up to the output's tokens, and a frame may hold
    /// none. Cuts that hold none of an output (an empty list too) say the
    /// engine sent it in its completion alone: a gRPC wire streams a frame
    /// of no tokens for each. Unset: `chunk_tokens` per frame.
    pub chunks: Option<Vec<usize>>,
    /// The status a recorded engine ended the call with (`status`): a gRPC
    /// wire ends the call with it, behind the frames the call's answers
    /// hold. Beside the finish reason `error` (what a line with a status
    /// reads when it states none) the engine left the choice open: its
    /// frames are sent and no completion.
    pub status: Option<Refusal>,
    /// The gRPC wire a recorded line was read off (`recorded.wire`): the
    /// count the line states is that wire's ([`Answer::counted_on`]).
    pub(crate) recorded_wire: Option<SchedulerWire>,
}

/// The status an engine refused a request with, as a recorded line keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The gRPC code.
    pub code: i32,
    /// The status's message.
    pub message: String,
    /// The trailers of the status that say what kind of failure the worker
    /// reports (`trailers`, name and value; a recorder keeps the ones named
    /// `smg-...`): a gateway answers its client by them.
    pub trailers: Vec<(String, String)>,
}

impl Refusal {
    /// The status a gRPC wire answers: the code, the message and the
    /// trailers.
    pub(crate) fn status(&self) -> tonic::Status {
        let mut status = tonic::Status::new(tonic::Code::from(self.code), self.message.clone());
        for (name, value) in &self.trailers {
            if let (Ok(name), Ok(value)) = (
                tonic::metadata::AsciiMetadataKey::from_bytes(name.as_bytes()),
                value.parse::<tonic::metadata::AsciiMetadataValue>(),
            ) {
                status.metadata_mut().insert(name, value);
            }
        }
        status
    }
}

/// One frame of an answer as a wire delivers it: the tokens of the cut, their
/// text pieces and logprobs where the answer has one per token, and whether
/// it is the last.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub ids: Vec<u32>,
    pub pieces: Vec<String>,
    pub logprobs: Vec<f32>,
    pub last: bool,
}

impl Frame {
    /// The frame's text: its pieces, joined.
    pub fn text(&self) -> String {
        self.pieces.concat()
    }
}

/// The top alternatives of one output position on a token wire: the sampled
/// id first, then the others, with their logprobs in the same order.
#[derive(Clone, Debug, PartialEq)]
pub struct TopAlternatives {
    pub ids: Vec<u32>,
    pub values: Vec<f32>,
}

/// Up to `k` alternatives for `sampled`: itself first at `logprob`, then the
/// pool's other distinct entries in order from its first occurrence on
/// (wrapping), each half a nat less likely than the one before. Keyed by
/// value, not by position: a repeated entry reads the same alternatives
/// wherever it stands, which keeps a streamed chunk's reading the whole
/// answer's.
fn top_k<T: Clone + PartialEq>(
    pool: &[T],
    sampled: &T,
    logprob: f32,
    k: usize,
) -> (Vec<T>, Vec<f32>) {
    let mut picked = vec![sampled.clone()];
    let start = pool
        .iter()
        .position(|entry| entry == sampled)
        .map_or(0, |at| at + 1);
    for other in pool.iter().cycle().skip(start).take(pool.len()) {
        if picked.len() == k {
            break;
        }
        if !picked.contains(other) {
            picked.push(other.clone());
        }
    }
    let values = (0..picked.len())
        .map(|rank| logprob - 0.5 * rank as f32)
        .collect();
    (picked, values)
}

/// An answer with what it is filed under, when it is: the prompt's key, or
/// a text the prompt must hold (never both).
pub struct Scripted {
    pub key: Option<String>,
    pub prompt_contains: Option<String>,
    pub answer: Answer,
}

impl Answer {
    /// Read one answer from its JSON object:
    ///
    /// ```json
    /// {"key": "<len>:<fold>" | "input_ids": [...] | "prompt_contains": "...",
    ///  "output_ids": [...], "output_text": "..." | ["piece", ...],
    ///  "finish_reason": "stop", "matched_stop": 2 | "</s>",
    ///  "cached_tokens": 0, "prompt_tokens": 12, "completion_tokens": 3,
    ///  "logprobs": [-0.1, ...],
    ///  "token_id_logprobs": [-0.2, ...], "delay_ms": 0, "chunk_tokens": 1,
    ///  "chunks": [2, 1], "status": {"code": 3, "message": "...", "trailers": {}}}
    /// ```
    ///
    /// Everything is optional but one of `output_ids` and `output_text`;
    /// `key` and `input_ids` file the answer under a prompt (a `--script`
    /// line needs one of them, an enqueued answer may carry one), and
    /// `prompt_contains` files an enqueued answer for the first request
    /// whose prompt holds the text (one of the three at most). An empty
    /// `output_text` without `output_ids` is an answer of no tokens: the
    /// completion alone, as an engine answers a request it may not generate
    /// for. Beside ids, an empty string is one whole-text piece like any other.
    /// `token_id_logprobs` scores the token ids a request asks for, in the
    /// request's order: the answer of a prefill-only scoring request.
    /// `chunks` states the tokens of each frame in turn (all of the output,
    /// or none of it where the completion alone carries it), and `status` the
    /// gRPC status the call ends with behind them: its `code` is the number
    /// of a failure (1 to 16), and a line whose call ends well states none.
    pub fn parse(value: &Value) -> Result<Scripted, String> {
        let object = value.as_object().ok_or("an answer is a JSON object")?;
        let known = [
            "key",
            "input_ids",
            "prompt_contains",
            "output_ids",
            "output_text",
            "finish_reason",
            "matched_stop",
            "cached_tokens",
            "prompt_tokens",
            "completion_tokens",
            "logprobs",
            "token_id_logprobs",
            "delay_ms",
            "chunk_tokens",
            "chunks",
            "status",
            // What a recorded line says beside its answer (the recorder's
            // facts of the exchange): `asked` joins the key, `wire` says
            // whose count the line states, the rest is for people and tools.
            "recorded",
        ];
        if let Some(unknown) = object.keys().find(|k| !known.contains(&k.as_str())) {
            return Err(format!("unknown field {unknown:?}"));
        }
        let ids = |name: &str| -> Result<Option<Vec<u32>>, String> {
            match object.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|item| {
                        item.as_u64()
                            .and_then(|id| u32::try_from(id).ok())
                            .ok_or_else(|| format!("{name}: {item} is not a token id"))
                    })
                    .collect::<Result<Vec<u32>, String>>()
                    .map(Some),
                Some(other) => Err(format!("{name}: expected an array, got {other}")),
            }
        };
        let key = match (object.get("key"), ids("input_ids")?) {
            (Some(_), Some(_)) => return Err("key and input_ids exclude each other".to_string()),
            (Some(Value::String(key)), None) => Some(key.clone()),
            (Some(other), None) => return Err(format!("key: expected a string, got {other}")),
            (None, Some(input_ids)) => Some(script_key(&input_ids)),
            (None, None) => None,
        };
        // A recorded line is filed under its prompt and what its request
        // asked beyond it.
        let key = match (key, object.get("recorded").and_then(|r| r.get("asked"))) {
            (Some(key), Some(asked)) if !asked.is_null() => Some(asked_key(&key, ask_fold(asked))),
            (key, _) => key,
        };
        let recorded_wire = match object.get("recorded").and_then(|r| r.get("wire")) {
            None | Some(Value::Null) => None,
            Some(Value::String(wire)) => Some(
                wire.parse::<SchedulerWire>()
                    .map_err(|_| format!("recorded.wire: {wire:?} is no gRPC wire"))?,
            ),
            Some(other) => return Err(format!("recorded.wire: expected a string, got {other}")),
        };
        let prompt_contains = match object.get("prompt_contains") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) if text.is_empty() => {
                return Err("prompt_contains: every prompt holds an empty text".to_string())
            }
            Some(Value::String(text)) => Some(text.clone()),
            Some(other) => return Err(format!("prompt_contains: expected a string, got {other}")),
        };
        if key.is_some() && prompt_contains.is_some() {
            return Err("prompt_contains and a prompt key exclude each other".to_string());
        }
        let output_ids = ids("output_ids")?.unwrap_or_default();
        let output_text = match object.get("output_text") {
            None | Some(Value::Null) => None,
            // No tokens: nothing said and no ids to say it with. An empty
            // string beside ids is the one whole-text piece it always was (a
            // recorded answer of one special token decodes to "").
            Some(Value::String(text)) if text.is_empty() && output_ids.is_empty() => {
                Some(Vec::new())
            }
            Some(Value::String(text)) => Some(vec![text.clone()]),
            Some(Value::Array(pieces)) => Some(
                pieces
                    .iter()
                    .map(|piece| {
                        piece
                            .as_str()
                            .map(str::to_string)
                            .ok_or_else(|| format!("output_text: {piece} is not a string"))
                    })
                    .collect::<Result<Vec<String>, String>>()?,
            ),
            Some(other) => {
                return Err(format!(
                    "output_text: expected a string or an array of strings, got {other}"
                ))
            }
        };
        if output_ids.is_empty() && output_text.is_none() {
            return Err("an answer needs output_ids or output_text".to_string());
        }
        if let Some(pieces) = &output_text {
            if !output_ids.is_empty() && pieces.len() != 1 && pieces.len() != output_ids.len() {
                return Err(format!(
                    "output_text has {} pieces for {} output ids: one per id, or one string",
                    pieces.len(),
                    output_ids.len()
                ));
            }
        }
        let finish_reason = match object.get("finish_reason") {
            // A status ends a choice the engine left open, unless the line
            // says how the choice completed ahead of it.
            None | Some(Value::Null) if object.get("status").is_some_and(|s| !s.is_null()) => {
                "error".to_string()
            }
            None | Some(Value::Null) => "stop".to_string(),
            Some(Value::String(reason)) => reason.clone(),
            Some(other) => return Err(format!("finish_reason: expected a string, got {other}")),
        };
        let matched_stop = match object.get("matched_stop") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(MatchedStop::Text(text.clone())),
            Some(Value::Number(id)) => Some(MatchedStop::Token(
                id.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| format!("matched_stop: {id} is not a token id"))?,
            )),
            Some(other) => {
                return Err(format!(
                    "matched_stop: expected a token id or a string, got {other}"
                ))
            }
        };
        let count = |name: &str| -> Result<Option<u64>, String> {
            match object.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => value
                    .as_u64()
                    .map(Some)
                    .ok_or_else(|| format!("{name}: expected a count, got {value}")),
            }
        };
        let to_u32 = |name: &str, value: u64| -> Result<u32, String> {
            u32::try_from(value).map_err(|_| format!("{name}: {value} is out of range"))
        };
        let cached_tokens = count("cached_tokens")?
            .map(|n| to_u32("cached_tokens", n))
            .transpose()?
            .unwrap_or(0);
        let prompt_tokens = count("prompt_tokens")?
            .map(|n| to_u32("prompt_tokens", n))
            .transpose()?;
        let completion_tokens = count("completion_tokens")?
            .map(|n| to_u32("completion_tokens", n))
            .transpose()?;
        let mut sampled_logprobs = false;
        let logprobs = match object.get("logprobs") {
            None | Some(Value::Null) => None,
            Some(Value::String(mode)) if mode == "sampled" => {
                sampled_logprobs = true;
                None
            }
            Some(Value::Array(values)) => {
                let logprobs = values
                    .iter()
                    .map(|value| {
                        logprob(value).ok_or_else(|| format!("logprobs: {value} is not a number"))
                    })
                    .collect::<Result<Vec<f32>, String>>()?;
                let units = if output_ids.is_empty() {
                    output_text.as_ref().map_or(0, Vec::len)
                } else {
                    output_ids.len()
                };
                if logprobs.len() != units {
                    return Err(format!(
                        "logprobs has {} values for {units} output tokens",
                        logprobs.len()
                    ));
                }
                Some(logprobs)
            }
            Some(other) => {
                return Err(format!(
                    "logprobs: expected an array or \"sampled\", got {other}"
                ))
            }
        };
        let token_id_logprobs = match object.get("token_id_logprobs") {
            None | Some(Value::Null) => None,
            Some(Value::Array(values)) => Some(
                values
                    .iter()
                    .map(|value| {
                        logprob(value)
                            .ok_or_else(|| format!("token_id_logprobs: {value} is not a number"))
                    })
                    .collect::<Result<Vec<f32>, String>>()?,
            ),
            Some(other) => {
                return Err(format!("token_id_logprobs: expected an array, got {other}"))
            }
        };
        let delay = Duration::from_millis(count("delay_ms")?.unwrap_or(0));
        let chunk_tokens = match count("chunk_tokens")? {
            None => 1,
            Some(0) => return Err("chunk_tokens must be at least 1".to_string()),
            Some(n) => {
                usize::try_from(n).map_err(|_| format!("chunk_tokens: {n} is out of range"))?
            }
        };
        let units = if output_ids.is_empty() {
            output_text.as_ref().map_or(0, Vec::len)
        } else {
            output_ids.len()
        };
        let chunks = match object.get("chunks") {
            None | Some(Value::Null) => None,
            Some(Value::Array(sizes)) => {
                let sizes = sizes
                    .iter()
                    .map(|size| {
                        size.as_u64()
                            .and_then(|size| usize::try_from(size).ok())
                            .ok_or_else(|| format!("chunks: {size} is not a count"))
                    })
                    .collect::<Result<Vec<usize>, String>>()?;
                // The cuts hold the whole output, or none of it: what an
                // engine sent in its completion alone.
                if total(&sizes) != Some(units) && total(&sizes) != Some(0) {
                    return Err(format!(
                        "chunks add up to {} for {units} output tokens",
                        total(&sizes).map_or_else(
                            || "more than a count holds".to_string(),
                            |n| n.to_string()
                        )
                    ));
                }
                Some(sizes)
            }
            Some(other) => return Err(format!("chunks: expected an array, got {other}")),
        };
        let status = match object.get("status") {
            None | Some(Value::Null) => None,
            Some(status) => Some(Refusal {
                // The code of a failure: 0 is OK, which ends no call ahead
                // of its answer, and a number past 16 names no code.
                code: status
                    .get("code")
                    .and_then(Value::as_i64)
                    .filter(|code| (1..=16).contains(code))
                    .and_then(|code| i32::try_from(code).ok())
                    .ok_or(
                        "status: expected {\"code\": <the gRPC code of a failure, 1 to 16>, \
                         \"message\": ...}",
                    )?,
                message: status
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                trailers: match status.get("trailers") {
                    None | Some(Value::Null) => Vec::new(),
                    Some(Value::Object(trailers)) => trailers
                        .iter()
                        .map(|(name, value)| {
                            let text = value.as_str().filter(|text| {
                                tonic::metadata::AsciiMetadataKey::from_bytes(name.as_bytes())
                                    .is_ok()
                                    && text.parse::<tonic::metadata::AsciiMetadataValue>().is_ok()
                            });
                            text.map(|text| (name.clone(), text.to_string()))
                                .ok_or_else(|| {
                                    format!("status: {name}: {value} is not a gRPC trailer")
                                })
                        })
                        .collect::<Result<Vec<(String, String)>, String>>()?,
                    Some(other) => {
                        return Err(format!("status: trailers: expected an object, got {other}"))
                    }
                },
            }),
        };
        Ok(Scripted {
            key,
            prompt_contains,
            answer: Answer {
                output_ids,
                output_text,
                finish_reason,
                matched_stop,
                cached_tokens,
                prompt_tokens,
                completion_tokens,
                logprobs,
                sampled_logprobs,
                token_id_logprobs,
                delay,
                chunk_tokens,
                chunks,
                status,
                recorded_wire,
            },
        })
    }

    /// Give an answer that has text but no ids the ids of its text, encoded
    /// with `tokenizer` (a token wire delivers ids): the whole text as one
    /// encode, so the pieces a test wrote are not forced to be tokens. An
    /// answer with ids, or without text, is returned as it is.
    pub fn encoded_with(mut self, tokenizer: &dyn Tokenizer) -> Result<Self, String> {
        if !self.output_ids.is_empty() {
            self.synthesize_logprobs();
            return Ok(self);
        }
        let Some(pieces) = &self.output_text else {
            return Ok(self);
        };
        let text = pieces.concat();
        let encoding = tokenizer
            .encode(&text, false)
            .map_err(|e| format!("encoding the scripted text: {e}"))?;
        self.output_ids = encoding.token_ids().to_vec();
        self.synthesize_logprobs();
        // Per-piece logprobs align with the ids only when the pieces were
        // tokens, which the encode does not promise: an answer that wants
        // logprobs on a token wire gives ids.
        if let Some(logprobs) = &self.logprobs {
            if logprobs.len() != self.output_ids.len() {
                return Err(format!(
                    "{} logprobs for {} encoded ids: give output_ids with logprobs on a token wire",
                    logprobs.len(),
                    self.output_ids.len()
                ));
            }
        }
        Ok(self)
    }

    /// The answer under the request's token limit, as an engine would end
    /// it: cut after `max_new_tokens` tokens with the `length` finish and no
    /// matched stop when it is longer, whole when it fits or when the request
    /// names no limit. A token is an output id, or a text piece on an answer
    /// without ids (the text wire's units): a whole text written as one
    /// piece beside ids is kept as the ids say.
    pub fn cut_to(mut self, max_new_tokens: Option<u32>) -> Self {
        let Some(limit) = max_new_tokens else {
            return self;
        };
        let limit = usize::try_from(limit).unwrap_or(usize::MAX);
        let units = if self.output_ids.is_empty() {
            self.output_text.as_ref().map_or(0, Vec::len)
        } else {
            self.output_ids.len()
        };
        if units <= limit {
            return self;
        }
        self.output_ids.truncate(limit);
        if let Some(pieces) = &mut self.output_text {
            if pieces.len() == units {
                pieces.truncate(limit);
            }
        }
        if let Some(logprobs) = &mut self.logprobs {
            if logprobs.len() == units {
                logprobs.truncate(limit);
            }
        }
        self.finish_reason = "length".to_string();
        self.matched_stop = None;
        self.chunks = None;
        self.completion_tokens = None;
        self
    }

    /// The answer as a gRPC worker of `wire` counts it. A recorded line
    /// states its completion tokens as the wire it was read off carried
    /// them, and a worker of that wire reports the count as it stands. A
    /// wire whose stream a gateway counts frame by frame may state the last
    /// frame's tokens alone, where the completion of the other two carries
    /// the total: on another wire the count is never below the output's ids.
    pub(crate) fn counted_on(mut self, wire: SchedulerWire) -> Self {
        if self.recorded_wire.is_some_and(|recorded| recorded != wire) {
            let carried = u32::try_from(self.output_ids.len()).unwrap_or(u32::MAX);
            self.completion_tokens = self.completion_tokens.map(|stated| stated.max(carried));
        }
        self
    }

    /// Whether the engine left this choice open: the call ended with a
    /// status ahead of the choice's completion.
    pub(crate) fn left_open(&self) -> bool {
        self.status.is_some() && self.finish_reason == "error"
    }

    /// The prompt tokens to report for a prompt of `prompt_len` ids.
    pub fn prompt_tokens(&self, prompt_len: usize) -> u32 {
        self.prompt_tokens
            .unwrap_or_else(|| u32::try_from(prompt_len).unwrap_or(u32::MAX))
    }

    /// The completion tokens to report: the count the answer states, else
    /// the output ids, or the text pieces when the answer has no ids.
    pub fn completion_tokens(&self) -> u32 {
        if let Some(stated) = self.completion_tokens {
            return stated;
        }
        let units = if self.output_ids.is_empty() {
            self.output_text.as_ref().map_or(0, Vec::len)
        } else {
            self.output_ids.len()
        };
        u32::try_from(units).unwrap_or(u32::MAX)
    }

    /// The made-up logprobs of a `"logprobs": "sampled"` answer, one per
    /// output id: eight values from -0.125 to -1.0, repeating, so a top-`k`
    /// reading finds alternatives among the answer's tokens.
    fn synthesize_logprobs(&mut self) {
        if self.sampled_logprobs && self.logprobs.is_none() {
            self.logprobs = Some(
                (0..self.output_ids.len())
                    .map(|i| -0.125 * ((i % 8) as f32 + 1.0))
                    .collect(),
            );
        }
    }

    /// The whole output text, for a non-streaming HTTP answer.
    pub fn text(&self) -> String {
        self.output_text
            .as_ref()
            .map_or_else(String::new, |pieces| pieces.concat())
    }

    /// The frames of a token wire: `chunk_tokens` output ids per frame, with
    /// their text and logprobs when the answer has one per id. No ids, no
    /// frames: the wire sends only its completion.
    pub fn token_frames(&self) -> Vec<Frame> {
        let units = self.output_ids.len();
        self.frames(
            units,
            |range| self.output_ids[range].to_vec(),
            |range| match &self.output_text {
                Some(pieces) if pieces.len() == units => pieces[range].to_vec(),
                _ => Vec::new(),
            },
        )
    }

    /// The frames a gRPC wire streams ahead of its completion: the token
    /// frames, or, where the cuts hold none of an output that has ids (an
    /// engine that sent its output in the completion alone), a frame of no
    /// tokens for each cut.
    pub(crate) fn chunk_frames(&self) -> Vec<Frame> {
        match &self.chunks {
            Some(cuts) if !self.output_ids.is_empty() && total(cuts) == Some(0) => {
                let count = cuts.len();
                (1..=count)
                    .map(|n| Frame {
                        ids: Vec::new(),
                        pieces: Vec::new(),
                        logprobs: Vec::new(),
                        last: n == count,
                    })
                    .collect()
            }
            _ => self.token_frames(),
        }
    }

    /// The frames of the text wire: `chunk_tokens` pieces per frame, with
    /// their ids and logprobs when the answer has one per piece. No text, no
    /// frames.
    pub fn text_frames(&self) -> Vec<Frame> {
        let Some(pieces) = &self.output_text else {
            return Vec::new();
        };
        let units = pieces.len();
        self.frames(
            units,
            |range| {
                if self.output_ids.len() == units {
                    self.output_ids[range].to_vec()
                } else {
                    Vec::new()
                }
            },
            |range| pieces[range].to_vec(),
        )
    }

    /// The top alternatives of some output tokens, as a request asking for
    /// `top_logprobs` reads them: at most `k` entries per position, the
    /// sampled token first with its own logprob, then the answer's other
    /// distinct tokens in output order from the token's first occurrence in
    /// the answer on (wrapping), each half a nat less likely than the one
    /// before; a token the answer repeats reads the same alternatives at each
    /// of its positions. The alternatives are the answer's own
    /// tokens, so they decode under the request's tokenizer; an answer with
    /// fewer distinct tokens than `k` reports fewer. Empty when `k` is 0 or
    /// the logprobs do not align with the tokens.
    pub fn top_logprobs(&self, ids: &[u32], logprobs: &[f32], k: usize) -> Vec<TopAlternatives> {
        if k == 0 || ids.len() != logprobs.len() {
            return Vec::new();
        }
        ids.iter()
            .zip(logprobs)
            .map(|(sampled, &logprob)| {
                let (ids, values) = top_k(&self.output_ids, sampled, logprob, k);
                TopAlternatives { ids, values }
            })
            .collect()
    }

    /// The top alternatives of some text pieces, the text wire's reading of
    /// [`Self::top_logprobs`]: the piece first, then the answer's other
    /// distinct pieces, up to `k` per position, each with its logprob.
    pub fn top_pieces(
        &self,
        pieces: &[String],
        logprobs: &[f32],
        k: usize,
    ) -> Vec<Vec<(String, f32)>> {
        if k == 0 || pieces.len() != logprobs.len() {
            return Vec::new();
        }
        let pool = self.output_text.as_deref().unwrap_or(&[]);
        pieces
            .iter()
            .zip(logprobs)
            .map(|(sampled, &logprob)| {
                let (pieces, values) = top_k(pool, sampled, logprob, k);
                pieces.into_iter().zip(values).collect()
            })
            .collect()
    }

    fn frames(
        &self,
        units: usize,
        ids: impl Fn(std::ops::Range<usize>) -> Vec<u32>,
        pieces: impl Fn(std::ops::Range<usize>) -> Vec<String>,
    ) -> Vec<Frame> {
        let step = self.chunk_tokens.max(1);
        // The recorded cuts where the answer has them, else even ones.
        let cuts: Vec<(usize, usize)> = match &self.chunks {
            Some(sizes) if total(sizes) == Some(units) => sizes
                .iter()
                .scan(0, |start, size| {
                    let cut = (*start, *start + size);
                    *start += size;
                    Some(cut)
                })
                .collect(),
            _ => (0..units)
                .step_by(step)
                .map(|start| (start, (start + step).min(units)))
                .collect(),
        };
        let count = cuts.len();
        cuts.into_iter()
            .enumerate()
            .map(|(n, (start, end))| {
                let range = start..end;
                let logprobs = match &self.logprobs {
                    Some(logprobs) if logprobs.len() == units => logprobs[range.clone()].to_vec(),
                    _ => Vec::new(),
                };
                Frame {
                    ids: ids(range.clone()),
                    pieces: pieces(range),
                    logprobs,
                    last: n + 1 == count,
                }
            })
            .collect()
    }
}

/// The script's key for a prompt: its length and the fold of its token ids.
pub fn script_key(ids: &[u32]) -> String {
    let mut h: u64 = 0;
    for &id in ids {
        h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(id);
    }
    format!("{}:{}", ids.len(), h)
}

/// What a request asks with the three fields the wires state differently
/// for one client request brought to one form, so that a line recorded on
/// one wire is found on the other two: a temperature left unstated reads 1.0
/// (two wires state that default, one leaves it out), a `top_k` that is off
/// reads 0 (one wire states 0, two state -1), and the seed is left out (one
/// wire carries none). The price: requests that differ only in their seed,
/// or in stating a temperature of 1.0, share a key and are served in turn.
fn wire_neutral(asked: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let mut asked = asked.clone();
    asked.remove("seed");
    if asked.get("temperature").is_some_and(Value::is_null) {
        asked.insert("temperature".to_string(), json!(1.0));
    }
    let off = |top_k: &Value| top_k.is_null() || top_k.as_i64().is_some_and(|k| k <= 0);
    if asked.get("top_k").is_some_and(off) {
        asked.insert("top_k".to_string(), json!(0));
    }
    asked
}

/// The fold of what a request asks beyond its prompt (a recorded line's
/// `recorded.asked`, and what a gRPC wire reads off its request): the script
/// key's fold over the object's text, written with every object's names in
/// order so that it reads the same whatever order a writer kept them in.
pub(crate) fn ask_fold(asked: &Value) -> u64 {
    fn write(value: &Value, text: &mut String) {
        match value {
            Value::Object(fields) => {
                let mut names: Vec<&String> = fields.keys().collect();
                names.sort();
                text.push('{');
                for name in names {
                    text.push_str(&Value::from(name.as_str()).to_string());
                    text.push(':');
                    write(&fields[name], text);
                    text.push(',');
                }
                text.push('}');
            }
            Value::Array(items) => {
                text.push('[');
                for item in items {
                    write(item, text);
                    text.push(',');
                }
                text.push(']');
            }
            other => text.push_str(&other.to_string()),
        }
    }
    let mut text = String::new();
    match asked {
        Value::Object(fields) => write(&Value::Object(wire_neutral(fields)), &mut text),
        other => write(other, &mut text),
    }
    fold_bytes(text.as_bytes())
}

/// The sum of an answer's cuts, `None` where it passes what a count holds:
/// cuts that wrap around to the output's own count are no cuts of it.
fn total(sizes: &[usize]) -> Option<usize> {
    sizes
        .iter()
        .try_fold(0_usize, |sum, size| sum.checked_add(*size))
}

/// The fold of the bytes of an identity this crate alone reads (what a
/// request asks, a media item's reference): the step of the
/// prompt key's fold, from the FNV offset basis, so that a zero byte moves
/// it wherever it stands. The prompt's key keeps its own start: its length
/// stands beside its fold.
pub(crate) fn fold_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, byte| {
        h.wrapping_mul(1_099_511_628_211) ^ u64::from(*byte)
    })
}

/// The key of a recorded line: its prompt's key and the fold of what its
/// request asked.
fn asked_key(key: &str, ask: u64) -> String {
    format!("{key}/{ask:016x}")
}

/// Whether a key names what its request asked ([`asked_key`]).
fn names_an_ask(key: &str) -> bool {
    key.contains('/')
}

/// A logprob as a line states it: a number, or one of the three values JSON
/// has no number for as the recorder writes them (`"NaN"`, `"inf"`, `"-inf"`).
fn logprob(value: &Value) -> Option<f32> {
    match value.as_str() {
        Some(text @ ("NaN" | "inf" | "-inf")) => text.parse().ok(),
        Some(_) => None,
        None => value.as_f64().map(|v| v as f32),
    }
}

/// The lines of an info file: the gRPC path of each call and the bytes of
/// its answer.
type InfoAnswers = Vec<(String, Vec<u8>)>;

/// The info answers a recorder kept beside the record file `path`: the file
/// `path` with its extension replaced by `info.jsonl`, one
/// `{"path", "answer_hex"}` line
/// a call (the call's gRPC path and the bytes of its answer). `None` where no
/// such file stands, as beside a hand-written script.
fn info_beside(path: &Path) -> Result<Option<InfoAnswers>, String> {
    let path = path.with_extension("info.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut answers = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let bad = |what: &str| format!("{}:{}: {what}", path.display(), n + 1);
        let value: Value = serde_json::from_str(line).map_err(|e| bad(&e.to_string()))?;
        let field = |name: &str| value.get(name).and_then(Value::as_str);
        let call = field("path").ok_or_else(|| bad("an info line needs path"))?;
        let hex = field("answer_hex").ok_or_else(|| bad("an info line needs answer_hex"))?;
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(hex.get(at..at + 2)?, 16).ok())
            .collect::<Option<Vec<u8>>>()
            .ok_or_else(|| bad("answer_hex is not hex"))?;
        answers.push((call.to_string(), bytes));
    }
    Ok(Some(answers))
}

#[derive(Default)]
struct Store {
    /// Answers by key, from `--script` lines and keyed enqueues, with how
    /// many of a key were served so far. Under a prompt's key: one answer,
    /// the last one stated. Under a recorded line's key ([`asked_key`]): the
    /// lines in file order, served in turn and from the first again (a
    /// recorded request sent many times).
    keyed: HashMap<String, (Vec<Answer>, usize)>,
    /// Whether a recording was ever loaded: a recorded line, or a record
    /// file with no line beside its recorder's info file. A gRPC worker
    /// fails a request that finds nothing from then on ([`not_recorded`]).
    recorded: bool,
    /// Answers in order, per worker name; [`ALL`] is the shared queue.
    queues: HashMap<String, VecDeque<Answer>>,
    /// Answers filed under a text the prompt must hold, in order, per worker
    /// name; [`ALL`]'s serve every worker. Each is served once.
    marked: HashMap<String, Vec<(String, Answer)>>,
    hits: HashMap<String, u64>,
    misses: HashMap<String, u64>,
    /// Whether anything was ever scripted: misses are counted from then on.
    active: bool,
    /// The script files loaded so far, each once, with their answer counts.
    loaded: HashMap<PathBuf, usize>,
    /// The info answers a recorder kept beside a record file, by the path of
    /// their call: the bytes of the first answer of each ([`info_beside`]).
    info: HashMap<String, Vec<u8>>,
    /// The workers that started in this process, by name: what the admin
    /// routes accept besides [`ALL`].
    workers: HashSet<String>,
}

impl Store {
    /// Load a script file once: a file loaded before is not read again, its
    /// count is reported.
    fn load(&mut self, path: &Path) -> Result<usize, String> {
        if let Some(count) = self.loaded.get(path) {
            return Ok(*count);
        }
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut answers = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
            let scripted =
                Answer::parse(&value).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
            let key = scripted.key.ok_or_else(|| {
                format!(
                    "{}:{}: a script line needs key or input_ids",
                    path.display(),
                    n + 1
                )
            })?;
            answers.push((key, scripted.answer));
        }
        // A recorder's info file stands beside a record file: one that holds
        // a recorded line, or none at all (a session of info calls alone, or
        // of no call). A file of hand-written answers is read alone, whatever
        // stands beside it.
        if answers.is_empty() || answers.iter().any(|(key, _)| names_an_ask(key)) {
            if let Some(info) = info_beside(path)? {
                for (call, answer) in info {
                    self.info.entry(call).or_insert(answer);
                }
                // A file with no line beside that info file is a recording
                // that holds nothing: every request misses it.
                if answers.is_empty() {
                    self.recorded = true;
                    RECORDED.store(true, Ordering::Release);
                }
            }
        }
        self.activate();
        let count = answers.len();
        self.loaded.insert(path.to_path_buf(), count);
        for (key, answer) in answers {
            if names_an_ask(&key) {
                self.keyed.entry(key).or_default().0.push(answer);
                self.recorded = true;
                RECORDED.store(true, Ordering::Release);
            } else {
                self.keyed.insert(key, (vec![answer], 0));
            }
        }
        Ok(count)
    }

    /// Something is scripted from now on: requests look in the store, and a
    /// request that finds nothing counts as a miss.
    fn activate(&mut self) {
        self.active = true;
        ACTIVE.store(true, Ordering::Release);
    }

    /// The answer for `worker`'s next request whose prompt has `key` and
    /// reads `prompt_text`.
    fn take(&mut self, worker: &str, key: &str, prompt_text: Option<&str>) -> Option<Answer> {
        // One choice never runs short: the queues hold it or the key does.
        self.take_n(worker, key, None, prompt_text, 1)
            .ok()
            .flatten()
            .and_then(|mut answers| answers.pop())
    }

    /// Whether answers filed under a text wait for `worker`'s requests (its
    /// own or every worker's): whether a request's prompt is worth reading.
    fn has_marked(&self, worker: &str) -> bool {
        [worker, ALL].iter().any(|scope| {
            self.marked
                .get(*scope)
                .is_some_and(|filed| !filed.is_empty())
        })
    }

    /// The first answer filed under a text `prompt_text` holds, for `worker`
    /// (its own first, then every worker's): where it is filed and its place
    /// there.
    fn marked_for(&self, worker: &str, prompt_text: &str) -> Option<(String, usize)> {
        [worker, ALL].iter().find_map(|scope| {
            self.marked
                .get(*scope)?
                .iter()
                .position(|(text, _)| prompt_text.contains(text.as_str()))
                .map(|at| ((*scope).to_string(), at))
        })
    }

    /// The answers for the `n` choices of `worker`'s next request whose
    /// prompt has `key`, asks `ask` beyond it (`None`: a wire that does not
    /// say) and reads `prompt_text` (`None`: unread, no answer filed under a
    /// text is looked at), taken under this one lock: the worker's queue,
    /// then the shared one, then the answer filed under a text the prompt
    /// holds (served once) or the keyed answers for the rest (the one under
    /// the prompt's key, else the recorded ones under the prompt and the
    /// ask, in turn). `None` when nothing is scripted for the first choice
    /// (the request takes the worker's other mode, whatever `n`); an `Err`
    /// when `n` is above [`CHOICES_LIMIT`], or naming the counts when the
    /// queues hold some of the choices but not all and nothing fills the
    /// rest, with the queues as they were.
    fn take_n(
        &mut self,
        worker: &str,
        key: &str,
        ask: Option<u64>,
        prompt_text: Option<&str>,
        n: usize,
    ) -> Result<Option<Vec<Answer>>, String> {
        let queued = [worker, ALL]
            .iter()
            .map(|queue| self.queues.get(*queue).map_or(0, VecDeque::len))
            .sum::<usize>();
        let marked = prompt_text.and_then(|text| self.marked_for(worker, text));
        let asked = ask.map(|ask| asked_key(key, ask));
        let keyed = if self.keyed.contains_key(key) {
            Some(key)
        } else {
            asked
                .as_deref()
                .filter(|asked| self.keyed.contains_key(*asked))
        };
        if queued == 0 && marked.is_none() && keyed.is_none() {
            if self.active {
                *self.misses.entry(worker.to_string()).or_insert(0) += 1;
            }
            return Ok(None);
        }
        if n > CHOICES_LIMIT {
            return Err(format!(
                "n = {n} choices is above the mock's limit of {CHOICES_LIMIT} per request"
            ));
        }
        if queued < n && marked.is_none() && keyed.is_none() {
            return Err(format!(
                "n = {n} choices need {n} scripted answers for this prompt, {queued} queued"
            ));
        }
        let mut answers = Vec::with_capacity(n);
        for queue in [worker, ALL] {
            if let Some(queue) = self.queues.get_mut(queue) {
                while answers.len() < n {
                    match queue.pop_front() {
                        Some(answer) => answers.push(answer),
                        None => break,
                    }
                }
            }
        }
        if answers.len() < n {
            let filed = marked.and_then(|(scope, at)| {
                let filed = self.marked.get_mut(&scope)?;
                (at < filed.len()).then(|| filed.remove(at).1)
            });
            if let Some(answer) = filed.as_ref() {
                while answers.len() < n {
                    answers.push(answer.clone());
                }
            } else if let Some((listed, served)) = keyed.and_then(|key| self.keyed.get_mut(key)) {
                if !listed.is_empty() {
                    // The turn moves by a whole call: where the queues
                    // answered the first choices, the rest read their own
                    // lines of this call, and the next call reads its own.
                    for at in answers.len()..n {
                        answers.push(listed[(*served + at) % listed.len()].clone());
                    }
                    *served += n;
                }
            }
        }
        *self.hits.entry(worker.to_string()).or_insert(0) +=
            u64::try_from(answers.len()).unwrap_or(u64::MAX);
        Ok(Some(answers))
    }

    /// Whether `worker` names a worker of the process, or every worker.
    fn knows(&self, worker: &str) -> bool {
        worker == ALL || self.workers.contains(worker)
    }

    fn enqueue(&mut self, worker: &str, answers: Vec<Scripted>) {
        self.activate();
        for scripted in answers {
            match (scripted.key, scripted.prompt_contains) {
                (Some(key), _) => {
                    if names_an_ask(&key) {
                        self.recorded = true;
                        RECORDED.store(true, Ordering::Release);
                    }
                    self.keyed.insert(key, (vec![scripted.answer], 0));
                }
                (None, Some(text)) => self
                    .marked
                    .entry(worker.to_string())
                    .or_default()
                    .push((text, scripted.answer)),
                (None, None) => self
                    .queues
                    .entry(worker.to_string())
                    .or_default()
                    .push_back(scripted.answer),
            }
        }
    }

    fn status(&self, worker: &str) -> Value {
        let queued = self.queues.get(worker).map_or(0, VecDeque::len);
        let marked = self.marked.get(worker).map_or(0, Vec::len);
        let (hits, misses) = if worker == ALL {
            (self.hits.values().sum(), self.misses.values().sum())
        } else {
            (
                self.hits.get(worker).copied().unwrap_or(0),
                self.misses.get(worker).copied().unwrap_or(0),
            )
        };
        json!({
            "worker": worker,
            "queued": queued,
            "marked": marked,
            "hits": hits,
            "misses": misses,
            "keyed": self.keyed.values().map(|(listed, _)| listed.len()).sum::<usize>(),
        })
    }

    fn clear(&mut self, worker: &str) {
        if worker == ALL {
            self.queues.clear();
            self.marked.clear();
        } else {
            self.queues.remove(worker);
            self.marked.remove(worker);
        }
    }

    /// A worker starts under `worker`: the name is known to the admin
    /// routes, with nothing queued or filed for it and its counters at
    /// zero, whatever a worker of that name left before.
    fn start(&mut self, worker: &str) {
        self.queues.remove(worker);
        self.marked.remove(worker);
        self.hits.remove(worker);
        self.misses.remove(worker);
        self.workers.insert(worker.to_string());
    }
}

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
/// Whether anything was ever scripted in the process, readable without the
/// lock: until then a request's path does not touch the store at all.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Whether a recording was ever loaded in the process, readable without
/// the lock: until then a gRPC wire does not read what a request asks.
static RECORDED: AtomicBool = AtomicBool::new(false);

/// The process-wide store every worker and the admin API share.
fn store() -> std::sync::MutexGuard<'static, Store> {
    STORE
        .get_or_init(|| Mutex::new(Store::default()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Whether anything was ever scripted in the process (a file loaded or an
/// answer enqueued). A wire with work of its own to do only for a script
/// (the HTTP worker's prompt ids in canned mode) skips it while this is false.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Whether a recording was ever loaded in the process (a recorded line, or
/// a record file with no line): a gRPC wire reads what a request asks beyond
/// its prompt ([`ask_fold`]) only then, and fails a request no line answers
/// ([`not_recorded`]).
pub(crate) fn recorded() -> bool {
    RECORDED.load(Ordering::Acquire)
}

/// The bytes a recorded engine answered the info call `call` with (its gRPC
/// path), where a record file with an info file beside it is loaded: a wire
/// answers that call with them, so that a gateway reads of a replay what it
/// read of the engine (the sampling defaults, the context length and the
/// rest that shape the requests it sends on).
pub(crate) fn recorded_info(call: &str) -> Option<Vec<u8>> {
    // Whatever the record file holds beside them: a session of info calls
    // alone leaves no recorded line.
    if !active() {
        return None;
    }
    store().info.get(call).cloned()
}

/// Why a request with `prompt_ids` that asks `ask` beyond them and found
/// nothing fails, once the process's store holds a recording: the key it
/// missed, and whether its prompt is recorded asked otherwise. `None` while
/// the store holds no recording: the request goes on to the engine.
#[cold]
pub(crate) fn not_recorded(prompt_ids: &[u32], ask: u64) -> Option<String> {
    let store = store();
    if !store.recorded {
        return None;
    }
    let key = script_key(prompt_ids);
    let prompt = store.keyed.keys().any(|held| {
        held.strip_prefix(key.as_str())
            .is_some_and(|ask| ask.starts_with('/'))
    });
    Some(format!(
        "mock-worker: no recorded answer under the key {}: {}",
        asked_key(&key, ask),
        if prompt {
            "this prompt is recorded, asked otherwise (the lines of the record file with these \
             input_ids say how under recorded.asked)"
        } else {
            "this prompt is not in the record file"
        }
    ))
}

/// A worker that started, by name (`grpc:<port>`, `http:<port>`,
/// `zmq:<index>`): the admin routes accept it from then on. The name starts
/// clean: a port a stopped worker of this process had, bound again, hands
/// its successor no queued or filed answers and no hits or misses. The name
/// must be one live worker's: a port is, as the bind precedes the start, and
/// a ZMQ rank's index is the caller's to keep unique, since a second live
/// rank of one index would start the first rank's entry clean while it
/// still serves.
pub fn register(worker: &str) {
    store().start(worker);
}

/// The workers that started in this process, by name, sorted: every listener
/// on every wire, canned or realistic. The admin fleet list walks them and
/// joins the engines a realistic worker spawned.
pub fn registered() -> Vec<String> {
    let mut names: Vec<String> = store().workers.iter().cloned().collect();
    names.sort();
    names
}

type LoadedTokenizer = Result<Arc<dyn Tokenizer>, String>;
static TOKENIZERS: OnceLock<Mutex<HashMap<String, LoadedTokenizer>>> = OnceLock::new();

/// The tokenizer the worker's `--tokenizer` names, loaded on first use and
/// kept per path: what a token wire encodes a scripted `output_text` with
/// when the answer has no ids. The gateway tokenizes prompts with the same
/// files, so the ids the mock sends back decode to the text the test wrote.
pub fn tokenizer(cfg: &Config) -> LoadedTokenizer {
    let mut loaded = TOKENIZERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    loaded
        .entry(cfg.tokenizer_path.clone())
        .or_insert_with(|| load_tokenizer(cfg))
        .clone()
}

fn load_tokenizer(cfg: &Config) -> LoadedTokenizer {
    // Local files only: the mock encodes with the files the gateway
    // was pointed at, and never fetches a tokenizer from a hub (the
    // hub loader blocks on a runtime of its own, which a worker's
    // request path must not do).
    let path = Path::new(&cfg.tokenizer_path);
    if !path.exists() {
        return Err(format!(
            "tokenizer {}: not a local file or directory",
            cfg.tokenizer_path
        ));
    }
    llm_tokenizer::factory::create_tokenizer_from_file(&cfg.tokenizer_path)
        .map_err(|e| format!("tokenizer {}: {e}", cfg.tokenizer_path))
}

type OutputTable = Option<Arc<[u32]>>;

/// What a realistic engine of the worker sends for each id of its synthetic
/// draw ([`crate::engine::output_table`]), built once per tokenizer path: an
/// id decodes where the worker's tokenizer gives text for it alone (special
/// tokens skipped; a part of a character is no text) and it is no
/// end-of-sequence id. `None` where the tokenizer does not load from local
/// files: the ids are then the ids drawn. A load that fails here is not
/// kept, so [`tokenizer`] still loads at its first use. Local files that give
/// no table are warned of where the table of their path is built.
pub(crate) fn output_table(cfg: &Config) -> OutputTable {
    static TABLES: OnceLock<Mutex<HashMap<String, OutputTable>>> = OnceLock::new();
    TABLES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(cfg.tokenizer_path.clone())
        .or_insert_with(|| {
            let (table, warning) = build_output_table(cfg);
            if let Some(warning) = warning {
                tracing::warn!("{warning}");
            }
            table
        })
        .clone()
}

/// The table of `cfg`'s tokenizer path, and the warning owed where local
/// files give none: files that do not load, and a tokenizer that decodes no
/// id of the draw. A path that is no local file (a model id of a hub) is
/// expected to give no table and owes no warning.
fn build_output_table(cfg: &Config) -> (OutputTable, Option<String>) {
    let mut loaded = TOKENIZERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let kept = match loaded.get(&cfg.tokenizer_path) {
        Some(kept) => kept.clone(),
        None => {
            let fresh = load_tokenizer(cfg);
            if let Ok(tokenizer) = &fresh {
                loaded.insert(cfg.tokenizer_path.clone(), Ok(Arc::clone(tokenizer)));
            }
            fresh
        }
    };
    drop(loaded);
    let tokenizer = match kept {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            let local = Path::new(&cfg.tokenizer_path).exists();
            let warning = local.then(|| format!("{e}: synthetic output keeps the IDs drawn"));
            return (None, warning);
        }
    };
    let (table, any_decodes) = output_table_of(tokenizer.as_ref());
    let warning = (!any_decodes).then(|| {
        format!(
            "tokenizer {}: decodes no ID of the synthetic output range: \
             synthetic output keeps the IDs drawn",
            cfg.tokenizer_path
        )
    });
    (table, warning)
}

/// The table of `tokenizer`, and whether any id of the draw decodes under it
/// (no table stands for both ends: every id decodes, or none does).
fn output_table_of(tokenizer: &dyn Tokenizer) -> (OutputTable, bool) {
    let eos = tokenizer.eos_token_ids();
    let any_decodes = Cell::new(false);
    let table = crate::engine::output_table(|id| {
        let decodes = !eos.contains(&id)
            && tokenizer
                .decode(&[id], true)
                .is_ok_and(|text| !text.is_empty() && !text.contains('\u{fffd}'));
        any_decodes.set(any_decodes.get() || decodes);
        decodes
    });
    (table, any_decodes.get())
}

/// The prompt's text for the answers filed under one: the rendered text the
/// request carries beside its ids (`rendered`, what the gateway sends on the
/// gRPC wire), else the ids decoded with the worker's tokenizer. Read only
/// while such answers are filed for the worker (a decode per request for
/// nothing otherwise). A prompt that cannot be read then (no tokenizer to
/// load, ids it refuses) is an `Err` the wire fails the request with, the
/// filed answers left where they are, so a test learns the reason instead of
/// reading a canned answer.
fn prompt_text_when_marked(
    cfg: &Config,
    worker: &str,
    prompt_ids: &[u32],
    rendered: &str,
) -> Result<Option<String>, String> {
    if !store().has_marked(worker) {
        return Ok(None);
    }
    if !rendered.is_empty() {
        return Ok(Some(rendered.to_string()));
    }
    tokenizer(cfg)
        .and_then(|tokenizer| {
            tokenizer
                .decode(prompt_ids, true)
                .map_err(|e| format!("decoding the prompt: {e}"))
        })
        .map(Some)
        .map_err(|e| {
            tracing::error!(
                "{worker}: the prompt cannot be read for the answers filed under a text: {e}"
            );
            format!("the prompt cannot be read for the answers filed under a text: {e}")
        })
}

/// [`take`] on a token wire that carries the prompt's ids alone, with the
/// prompt decoded for the answers filed under a text and a text-only answer
/// encoded for the wire: the answer with its ids. A prompt that cannot be
/// read while such answers wait, or an answer that cannot be encoded (no
/// tokenizer to load, or logprobs that do not align), is an `Err` the wire
/// fails the request with, so a test learns the reason instead of reading a
/// canned answer.
pub fn take_ids(cfg: &Config, worker: &str, prompt_ids: &[u32]) -> Result<Option<Answer>, String> {
    if !active() {
        return Ok(None);
    }
    let prompt_text = prompt_text_when_marked(cfg, worker, prompt_ids, "")?;
    let key = script_key(prompt_ids);
    let Some(answer) = store().take(worker, &key, prompt_text.as_deref()) else {
        return Ok(None);
    };
    encode_for_wire(cfg, worker, answer).map(Some)
}

/// `answer` with its ids for a token wire: a text-only answer encoded with
/// the worker's tokenizer, any other as it is (an answer of no tokens, a
/// scoring answer's, needs none), its sampled logprobs made up once the ids
/// are known either way.
fn encode_for_wire(cfg: &Config, worker: &str, mut answer: Answer) -> Result<Answer, String> {
    if !answer.output_ids.is_empty() || answer.output_text.as_ref().is_none_or(Vec::is_empty) {
        answer.synthesize_logprobs();
        return Ok(answer);
    }
    tokenizer(cfg)
        .and_then(|tokenizer| answer.encoded_with(tokenizer.as_ref()))
        .map_err(|e| {
            tracing::error!("{worker}: a scripted text cannot go on this wire: {e}");
            format!("the scripted answer cannot go on this wire: {e}")
        })
}

/// The most choices one request takes scripted answers for: a bound on what
/// a request's `n` makes the worker hold (a keyed answer is cloned once per
/// choice), reported like a short script; a request nothing is scripted for
/// is not bound by it.
pub const CHOICES_LIMIT: usize = 128;

/// [`take`] for every choice of a request that samples `n` times and whose
/// prompt reads `prompt_text` (the text wire's own; `None` leaves the answers
/// filed under a text alone), under the store's one lock: one scripted
/// answer per choice, in the order they were enqueued (the worker's queue,
/// then the shared one), the answer filed under a text the prompt holds or
/// keyed by the prompt for the rest. `None` when nothing is scripted for the
/// first choice: the request takes the worker's other mode, as a request for
/// one choice does, whatever `n`. An `Err` when `n` is above
/// [`CHOICES_LIMIT`], or when the queues hold some of the choices but not
/// all and nothing fills the rest, the queues left as they were either way:
/// a test learns what it stated instead of reading a canned choice.
pub fn take_per_choice(
    worker: &str,
    prompt_ids: &[u32],
    prompt_text: Option<&str>,
    n: usize,
) -> Result<Option<Vec<Answer>>, String> {
    if !active() {
        return Ok(None);
    }
    take_asked(worker, prompt_ids, None, prompt_text, n)
}

/// [`take_per_choice`] for a request that asks `ask` beyond its prompt.
fn take_asked(
    worker: &str,
    prompt_ids: &[u32],
    ask: Option<u64>,
    prompt_text: Option<&str>,
    n: usize,
) -> Result<Option<Vec<Answer>>, String> {
    let key = script_key(prompt_ids);
    store()
        .take_n(worker, &key, ask, prompt_text, n.max(1))
        .map_err(|why| {
            tracing::error!("{worker}: {why}");
            why
        })
}

/// [`take_per_choice`] on a token wire: the prompt read for the answers
/// filed under a text (`rendered`, the text the request carries beside its
/// ids, else the ids decoded), and the text-only answers encoded for the
/// wire (outside the lock), as [`take_ids`] reads and encodes. `ask` is the
/// fold of what the request asks beyond its prompt (`ask_fold`), for the
/// recorded lines; `None` on a wire that does not read it.
pub fn take_ids_per_choice(
    cfg: &Config,
    worker: &str,
    prompt_ids: &[u32],
    rendered: &str,
    n: usize,
    ask: Option<u64>,
) -> Result<Option<Vec<Answer>>, String> {
    if !active() {
        return Ok(None);
    }
    let prompt_text = prompt_text_when_marked(cfg, worker, prompt_ids, rendered)?;
    let Some(answers) = take_asked(worker, prompt_ids, ask, prompt_text.as_deref(), n)? else {
        return Ok(None);
    };
    answers
        .into_iter()
        .map(|answer| encode_for_wire(cfg, worker, answer))
        .collect::<Result<Vec<Answer>, String>>()
        .map(Some)
}

/// Load a `--script` file: one answer per line, each with a `key` or its
/// `input_ids`. A file is loaded once per process; loading it again is a
/// no-op that reports its answers. Returns how many answers the file holds.
pub fn load(path: &Path) -> Result<usize, String> {
    store().load(path)
}

/// Load the configuration's `--script` file if it has one and it is not
/// loaded yet (the workers call this as they start; the binary checks the
/// file once before any of them does).
pub fn ensure_loaded(cfg: &Config) -> Result<(), String> {
    match &cfg.replay.script {
        Some(path) => load(path).map(|_| ()),
        None => Ok(()),
    }
}

/// The answer for `worker`'s next request with `prompt_ids`, if one is
/// scripted: its queue, then the shared queue, then the prompt's key (the
/// answers filed under a text need the prompt read: [`take_per_choice`],
/// [`take_ids`]). A miss is counted once anything was ever scripted in the
/// process; until then nothing is computed and the store is not locked.
pub fn take(worker: &str, prompt_ids: &[u32]) -> Option<Answer> {
    if !active() {
        return None;
    }
    let key = script_key(prompt_ids);
    store().take(worker, &key, None)
}

/// Enqueue `answers` for `worker` ([`ALL`] for every worker), the keyed ones
/// under their key and the ones given a text under it instead; answers the
/// worker's state, or `None` for a name no worker of the process has.
pub fn enqueue(worker: &str, answers: Vec<Scripted>) -> Option<Value> {
    let mut store = store();
    if !store.knows(worker) {
        return None;
    }
    store.enqueue(worker, answers);
    Some(store.status(worker))
}

/// The worker's queue length, the answers filed under a text for it
/// (`marked`), its hits and misses, and the keyed answers of the process;
/// [`ALL`] reports the shared queue, the shared filed answers and the
/// totals. `None` for a name no worker of the process has.
pub fn status(worker: &str) -> Option<Value> {
    let store = store();
    store.knows(worker).then(|| store.status(worker))
}

/// Drop the worker's queued answers and the ones filed under a text
/// ([`ALL`]: every worker's); the keyed answers and the counters stay.
/// Answers the worker's state, or `None` for a name no worker of the process
/// has.
pub fn clear(worker: &str) -> Option<Value> {
    let mut store = store();
    if !store.knows(worker) {
        return None;
    }
    store.clear(worker);
    Some(store.status(worker))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Scripted, String> {
        Answer::parse(&serde_json::from_str(text).expect("test JSON"))
    }

    #[test]
    fn the_key_is_the_length_and_the_fold_of_the_ids() {
        assert_eq!(script_key(&[]), "0:0");
        assert_eq!(script_key(&[7]), "1:7");
        // (7 * 1099511628211) ^ 9
        assert_eq!(script_key(&[7, 9]), "2:7696581397484");
        assert_ne!(script_key(&[7, 9]), script_key(&[9, 7]), "order counts");
    }

    #[test]
    fn an_answer_reads_its_fields_and_defaults() {
        let scripted = parse(
            r#"{"input_ids": [1, 2], "output_ids": [10, 11, 12], "output_text": ["a", "b", "c"],
                "finish_reason": "length", "matched_stop": "</s>", "cached_tokens": 1,
                "prompt_tokens": 5, "logprobs": [-0.5, -1.0, -1.5], "delay_ms": 3, "chunk_tokens": 2}"#,
        )
        .expect("a full answer");
        assert_eq!(scripted.key.as_deref(), Some(script_key(&[1, 2]).as_str()));
        let answer = scripted.answer;
        assert_eq!(answer.output_ids, vec![10, 11, 12]);
        assert_eq!(answer.text(), "abc");
        assert_eq!(answer.finish_reason, "length");
        assert_eq!(
            answer.matched_stop,
            Some(MatchedStop::Text("</s>".to_string()))
        );
        assert_eq!((answer.cached_tokens, answer.prompt_tokens), (1, Some(5)));
        assert_eq!(answer.prompt_tokens(99), 5);
        assert_eq!(answer.completion_tokens(), 3);
        assert_eq!(answer.delay, Duration::from_millis(3));

        let minimal = parse(r#"{"output_text": "hi"}"#).expect("text only").answer;
        assert_eq!(minimal.finish_reason, "stop");
        assert_eq!(minimal.output_ids, Vec::<u32>::new());
        assert_eq!(minimal.completion_tokens(), 1);
        assert_eq!(minimal.prompt_tokens(4), 4);
        assert_eq!(minimal.chunk_tokens, 1);
        let token =
            parse(r#"{"key": "1:7", "output_ids": [3], "matched_stop": 3}"#).expect("keyed");
        assert_eq!(token.key.as_deref(), Some("1:7"));
        assert_eq!(token.answer.matched_stop, Some(MatchedStop::Token(3)));
    }

    #[test]
    fn an_empty_text_is_no_tokens_alone_and_one_piece_beside_ids() {
        let silent = parse(r#"{"output_text": ""}"#).expect("no tokens").answer;
        assert_eq!(silent.completion_tokens(), 0);
        assert_eq!(silent.text(), "");
        // A recorded answer of one special token decodes to an empty string:
        // the one whole-text piece for its ids, as any one string is.
        let eos = parse(r#"{"output_ids": [2], "output_text": ""}"#)
            .expect("ids beside an empty string")
            .answer;
        assert_eq!(eos.output_ids, vec![2]);
        assert_eq!(eos.completion_tokens(), 1);
        assert_eq!(eos.text(), "");
    }

    #[test]
    fn an_answer_refuses_what_it_cannot_mean() {
        for bad in [
            r#"{"finish_reason": "stop"}"#,
            r#"{"output_ids": [1, 2], "output_text": ["a", "b", "c"]}"#,
            r#"{"output_ids": [1, 2], "logprobs": [-1.0]}"#,
            r#"{"output_ids": [1], "chunk_tokens": 0}"#,
            r#"{"output_ids": [1], "key": "1:1", "input_ids": [1]}"#,
            r#"{"output_ids": [1], "matched_stop": true}"#,
            r#"{"output_ids": [1], "delay": 5}"#,
            r#"{"output_ids": ["x"]}"#,
            r#"{"output_ids": [1], "recorded": {"wire": "http"}}"#,
            r#"{"output_ids": [1], "recorded": {"wire": 1}}"#,
            r"[1]",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn frames_cut_the_output_by_chunk_tokens_and_carry_what_aligns() {
        let answer = parse(
            r#"{"output_ids": [10, 11, 12], "output_text": ["a", "b", "c"], "logprobs": [-1, -2, -3], "chunk_tokens": 2}"#,
        )
        .expect("answer")
        .answer;
        let frames = answer.token_frames();
        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[0],
            Frame {
                ids: vec![10, 11],
                pieces: vec!["a".to_string(), "b".to_string()],
                logprobs: vec![-1.0, -2.0],
                last: false
            }
        );
        assert_eq!(frames[0].text(), "ab");
        assert_eq!(
            frames[1],
            Frame {
                ids: vec![12],
                pieces: vec!["c".to_string()],
                logprobs: vec![-3.0],
                last: true
            }
        );
        assert_eq!(answer.text_frames(), frames, "pieces per id: the same cuts");

        // One string beside the ids: the token frames carry no text, the
        // text wire sends the string as one frame with every id.
        let whole = parse(r#"{"output_ids": [10, 11], "output_text": "ab", "logprobs": [-1, -2]}"#)
            .expect("answer")
            .answer;
        assert!(whole.token_frames().iter().all(|f| f.pieces.is_empty()));
        assert_eq!(
            whole.text_frames(),
            vec![Frame {
                ids: vec![],
                pieces: vec!["ab".to_string()],
                logprobs: vec![],
                last: true
            }]
        );
        let text_only = parse(r#"{"output_text": "ab"}"#).expect("answer").answer;
        assert!(
            text_only.token_frames().is_empty(),
            "no ids, no token frames"
        );
        assert_eq!(text_only.text_frames().len(), 1);
    }

    #[test]
    fn a_store_serves_queues_before_keys_and_counts_misses_once_active() {
        let mut store = Store::default();
        let worker = "grpc:19000";
        assert!(
            store.take(worker, &script_key(&[1, 2, 3]), None).is_none(),
            "nothing scripted"
        );
        assert_eq!(
            store.status(worker)["misses"],
            0,
            "no script yet: no miss counted"
        );

        assert!(
            !store.knows(worker) && store.knows(ALL),
            "the workers register their names"
        );

        store.workers.insert(worker.to_string());

        assert!(store.knows(worker));

        let keyed = parse(r#"{"input_ids": [1, 2, 3], "output_ids": [7]}"#).expect("keyed");
        let first = parse(r#"{"output_ids": [1]}"#).expect("queued");
        let second = parse(r#"{"output_ids": [2]}"#).expect("queued");
        let shared = parse(r#"{"output_ids": [3]}"#).expect("shared");
        store.enqueue(worker, vec![keyed, first, second]);
        assert_eq!(store.status(worker)["queued"], 2);
        store.enqueue(ALL, vec![shared]);

        let ids = |answer: Option<Answer>| answer.map(|a| a.output_ids);
        assert_eq!(
            ids(store.take(worker, &script_key(&[9]), None)),
            Some(vec![1])
        );
        assert_eq!(
            ids(store.take(worker, &script_key(&[9]), None)),
            Some(vec![2])
        );
        assert_eq!(
            ids(store.take(worker, &script_key(&[9]), None)),
            Some(vec![3]),
            "the shared queue"
        );
        assert_eq!(
            ids(store.take(worker, &script_key(&[1, 2, 3]), None)),
            Some(vec![7]),
            "the key"
        );
        assert_eq!(
            ids(store.take(worker, &script_key(&[1, 2, 3]), None)),
            Some(vec![7]),
            "a key stays"
        );
        assert!(store.take(worker, &script_key(&[9]), None).is_none());
        let state = store.status(worker);
        assert_eq!(
            (state["hits"].as_u64(), state["misses"].as_u64()),
            (Some(5), Some(1))
        );
        assert_eq!(store.status(ALL)["misses"], 1, "the totals");

        store.enqueue(
            worker,
            vec![parse(r#"{"output_ids": [4]}"#).expect("queued")],
        );
        store.clear(worker);
        assert_eq!(store.status(worker)["queued"], 0);
        assert!(
            store.take(worker, &script_key(&[9]), None).is_none(),
            "cleared"
        );
    }

    /// A worker's port, bound again by a later worker of the same process,
    /// names the same entry: the new worker starts it clean, so the counts
    /// it reports are its own (two filed answers served and one miss read as
    /// two hits and one miss, not the predecessor's hit on top).
    #[test]
    fn a_name_bound_again_starts_with_nothing_of_its_predecessor() {
        let mut store = Store::default();
        let worker = "grpc:19002";
        let marked = |text: &str, id: u32| {
            parse(&format!(
                r#"{{"prompt_contains": "{text}", "output_ids": [{id}]}}"#
            ))
            .expect("an answer filed under a text")
        };
        let key = script_key(&[9]);
        store.start(worker);
        store.enqueue(
            worker,
            vec![
                parse(r#"{"output_ids": [1]}"#).expect("queued"),
                marked("quince", 4),
            ],
        );
        assert!(
            store.take(worker, &key, None).is_some(),
            "the queued answer"
        );
        assert!(store.take(worker, &key, None).is_none(), "a miss");
        let before = store.status(worker);
        assert_eq!(
            (
                before["hits"].as_u64(),
                before["misses"].as_u64(),
                before["marked"].as_u64()
            ),
            (Some(1), Some(1), Some(1)),
            "{before}"
        );

        store.start(worker);
        let after = store.status(worker);
        assert!(store.knows(worker), "still a worker of the process");
        assert_eq!(
            (after["queued"].as_u64(), after["marked"].as_u64()),
            (Some(0), Some(0)),
            "{after}"
        );
        assert_eq!(
            (after["hits"].as_u64(), after["misses"].as_u64()),
            (Some(0), Some(0)),
            "{after}"
        );

        store.enqueue(worker, vec![marked("apricot", 20), marked("bramble", 30)]);
        let ids = |answer: Option<Answer>| answer.map(|a| a.output_ids);
        assert_eq!(
            ids(store.take(worker, &key, Some("say bramble"))),
            Some(vec![30])
        );
        assert_eq!(
            ids(store.take(worker, &key, Some("say apricot"))),
            Some(vec![20])
        );
        assert!(
            store.take(worker, &key, Some("say say")).is_none(),
            "a miss"
        );
        let state = store.status(worker);
        assert_eq!(
            (
                state["marked"].as_u64(),
                state["hits"].as_u64(),
                state["misses"].as_u64()
            ),
            (Some(0), Some(2), Some(1)),
            "{state}"
        );
    }

    /// The seam the workers use: `register` starts the name clean on the
    /// process-wide store, whatever a worker of that name left.
    #[test]
    fn registering_a_name_again_starts_it_clean_on_the_process_store() {
        let worker = "grpc:19003";
        register(worker);
        enqueue(
            worker,
            vec![parse(r#"{"output_ids": [1]}"#).expect("queued")],
        )
        .expect("a known worker");
        assert!(take(worker, &[9]).is_some(), "the queued answer");
        assert!(take(worker, &[9]).is_none(), "a miss");
        let before = status(worker).expect("a known worker");
        assert_eq!(
            (before["hits"].as_u64(), before["misses"].as_u64()),
            (Some(1), Some(1)),
            "{before}"
        );

        register(worker);
        let after = status(worker).expect("still a known worker");
        assert_eq!(
            (
                after["queued"].as_u64(),
                after["hits"].as_u64(),
                after["misses"].as_u64()
            ),
            (Some(0), Some(0), Some(0)),
            "{after}"
        );
    }

    #[test]
    fn an_answer_filed_under_a_text_serves_the_first_prompt_that_holds_it_once() {
        let mut store = Store::default();
        let worker = "grpc:19001";
        store.workers.insert(worker.to_string());
        let marked = |text: &str, id: u32| {
            parse(&format!(
                r#"{{"prompt_contains": "{text}", "output_ids": [{id}]}}"#
            ))
            .expect("an answer filed under a text")
        };
        store.enqueue(ALL, vec![marked("apricot", 1), marked("bramble", 2)]);
        store.enqueue(worker, vec![marked("bramble", 3)]);
        assert_eq!(store.status(ALL)["marked"], 2);
        assert_eq!(store.status(worker)["marked"], 1);
        let key = script_key(&[9]);
        let ids = |answer: Option<Answer>| answer.map(|a| a.output_ids);

        assert_eq!(
            ids(store.take(worker, &key, Some("say cobalt"))),
            None,
            "a prompt that holds no filed text is a miss"
        );
        assert_eq!(
            ids(store.take(worker, &key, None)),
            None,
            "an unread prompt matches nothing"
        );
        assert_eq!(
            ids(store.take(worker, &key, Some("say bramble"))),
            Some(vec![3]),
            "the worker's own answer before the shared one"
        );
        assert_eq!(
            ids(store.take(worker, &key, Some("say bramble"))),
            Some(vec![2]),
            "then the shared one"
        );
        assert_eq!(
            ids(store.take(worker, &key, Some("say bramble"))),
            None,
            "each is served once"
        );
        store.enqueue(
            worker,
            vec![parse(r#"{"output_ids": [4]}"#).expect("queued")],
        );
        assert_eq!(
            ids(store.take(worker, &key, Some("say apricot"))),
            Some(vec![4]),
            "the queues come first"
        );
        let two = store
            .take_n(worker, &key, None, Some("say apricot"), 2)
            .expect("filled")
            .expect("scripted");
        assert_eq!(
            two.iter().map(|a| a.output_ids.clone()).collect::<Vec<_>>(),
            vec![vec![1], vec![1]],
            "a filed answer fills every choice, like a keyed one"
        );
        assert_eq!(store.status(ALL)["marked"], 0);
        let state = store.status(worker);
        assert_eq!(
            (state["hits"].as_u64(), state["misses"].as_u64()),
            (Some(5), Some(3))
        );

        store.enqueue(ALL, vec![marked("cobalt", 5)]);
        store.enqueue(worker, vec![marked("dahlia", 6)]);
        store.clear(worker);
        assert_eq!(
            store.status(worker)["marked"],
            0,
            "a worker's clear drops its own filed answers"
        );
        assert_eq!(store.status(ALL)["marked"], 1, "and not the shared ones");
        store.clear(ALL);
        assert_eq!(
            store.status(ALL)["marked"],
            0,
            "the shared clear drops them all"
        );
    }

    #[test]
    fn a_text_to_file_an_answer_under_is_one_of_the_prompts_handles_and_not_empty() {
        let filed = parse(r#"{"prompt_contains": "apricot", "output_ids": [1]}"#)
            .expect("an answer filed under a text");
        assert_eq!(
            (filed.key, filed.prompt_contains.as_deref()),
            (None, Some("apricot"))
        );
        assert!(parse(r#"{"prompt_contains": "", "output_ids": [1]}"#)
            .err()
            .expect("an empty text is refused")
            .contains("every prompt holds an empty text"));
        assert!(parse(r#"{"prompt_contains": 7, "output_ids": [1]}"#).is_err());
        assert!(
            parse(r#"{"prompt_contains": "apricot", "input_ids": [1], "output_ids": [1]}"#)
                .err()
                .expect("a text and a key exclude each other")
                .contains("exclude each other")
        );
        assert!(
            parse(r#"{"prompt_contains": "apricot", "key": "1:1", "output_ids": [1]}"#).is_err()
        );

        let mut store = Store::default();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("filed.jsonl");
        std::fs::write(
            &path,
            "{\"prompt_contains\": \"apricot\", \"output_ids\": [1]}\n",
        )
        .expect("write");
        assert!(store
            .load(&path)
            .expect_err("a script line is keyed by its prompt, never by a text")
            .contains("needs key or input_ids"));
    }

    #[test]
    fn sampled_logprobs_are_made_up_once_the_ids_are_known() {
        use llm_tokenizer::MockTokenizer;
        let tokenizer = MockTokenizer::new();
        let answer = parse(r#"{"output_text": "hello world", "logprobs": "sampled"}"#)
            .expect("answer")
            .answer;
        assert_eq!(answer.logprobs, None, "nothing before the encode");
        let encoded = answer.encoded_with(&tokenizer).expect("encoded");
        let logprobs = encoded.logprobs.expect("one logprob per id");
        assert_eq!(logprobs.len(), encoded.output_ids.len());
        assert!(logprobs.iter().all(|lp| *lp < 0.0));
        let with_ids = parse(r#"{"output_ids": [4, 5], "logprobs": "sampled"}"#)
            .expect("answer")
            .answer
            .encoded_with(&tokenizer)
            .expect("encoded");
        assert_eq!(with_ids.logprobs, Some(vec![-0.125, -0.25]));
        assert_eq!(with_ids.token_frames().len(), 2, "one logprob per frame");
        assert!(parse(r#"{"output_text": "x", "logprobs": "random"}"#).is_err());
    }

    #[test]
    fn an_answer_scripted_with_ids_gets_its_sampled_logprobs_on_the_wire() {
        // Through `take_ids`, the token wires' path: an answer that carries
        // ids skips the encode, and its sampled logprobs are made up all the
        // same, so a request that asks for logprobs is answered with them.
        let worker = "zmq:sampled-logprobs";
        register(worker);
        let scripted = parse(
            r#"{"input_ids": [4101, 4102, 4103], "output_ids": [4, 5], "logprobs": "sampled"}"#,
        )
        .expect("answer");
        assert!(enqueue(worker, vec![scripted]).is_some());
        let answer = take_ids(&Config::default(), worker, &[4101, 4102, 4103])
            .expect("nothing to encode, so no tokenizer is needed")
            .expect("the scripted answer");
        assert_eq!(answer.output_ids, vec![4, 5]);
        assert_eq!(answer.logprobs, Some(vec![-0.125, -0.25]));
        assert_eq!(answer.token_frames().len(), 2, "one logprob per frame");
    }

    /// The table of a tokenizer keeps each id of the draw the tokenizer
    /// decodes and gives every other one an id that decodes: an id outside
    /// the vocabulary, a special token, and the end-of-sequence id whatever
    /// its text.
    #[test]
    fn the_output_table_holds_the_ids_the_tokenizer_decodes() {
        use llm_tokenizer::MockTokenizer;
        let words: Vec<(String, u32)> = (100..164).map(|id| (format!("w{id}"), id)).collect();
        let words: Vec<(&str, u32)> = words
            .iter()
            .map(|(word, id)| (word.as_str(), *id))
            .collect();
        // The mock tokenizer holds `<bos>` (1,000), which it skips as a
        // special token, and three plain tokens from 1,001; `</s>` takes the
        // end-of-sequence id (999) and decodes to its text.
        let tokenizer = MockTokenizer::new()
            .with_tokens(&words)
            .with_tokens(&[("\u{fffd}", 2_500)])
            .with_eos_token("</s>");
        let table = output_table_of(&tokenizer).0.expect("a table");
        assert_eq!(table.len(), 30_000);
        assert!(
            table
                .iter()
                .all(|id| (100..164).contains(id) || (1_001..1_004).contains(id)),
            "only ids that decode"
        );
        let kept: Vec<u32> = (100..164).collect();
        assert_eq!(&table[..64], &kept[..], "an id that decodes stays");
        assert_eq!(table[1_001 - 100], 1_001);
        assert_ne!(table[999 - 100], 999, "the end-of-sequence id");
        assert_ne!(table[1_000 - 100], 1_000, "a special token");
        assert_ne!(table[2_500 - 100], 2_500, "a part of a character");
    }

    /// A worker whose `--tokenizer` names local files takes the table of
    /// that tokenizer for its engine; one whose tokenizer is no local file
    /// takes none, and its engine sends the ids it draws.
    #[test]
    fn a_workers_engine_takes_the_table_of_its_local_tokenizer() {
        let dir = tempfile::tempdir().expect("temp dir");
        // A folder that holds no tokenizer yet gives no table, and the load
        // that failed is not kept: the files written later load.
        let late = Config {
            tokenizer_path: dir.path().join("late").display().to_string(),
            ..Config::default()
        };
        std::fs::create_dir(&late.tokenizer_path).expect("a folder");
        assert!(late.engine_for("grpc:19000").output_table.is_none());
        let vocab: Vec<String> = (0..300).map(|id| format!(r#""w{id}":{id}"#)).collect();
        let special = r#"{"id":300,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}"#;
        let model = format!(
            r#"{{"type":"WordLevel","vocab":{{{}}},"unk_token":"w0"}}"#,
            vocab.join(",")
        );
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{special}],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{model}}}"#
        );
        std::fs::write(dir.path().join("tokenizer.json"), &json).expect("tokenizer.json");
        std::fs::write(
            Path::new(&late.tokenizer_path).join("tokenizer.json"),
            &json,
        )
        .expect("tokenizer.json");
        assert!(tokenizer(&late).is_ok(), "the files written later load");
        let cfg = Config {
            tokenizer_path: dir.path().display().to_string(),
            ..Config::default()
        };
        let table = cfg.engine_for("grpc:19000").output_table.expect("a table");
        assert_eq!(table.len(), 30_000);
        assert!(
            table.iter().all(|id| (100..300).contains(id)),
            "ids of the vocabulary, its special token left out"
        );
        assert_eq!(table[0], 100, "an id that decodes stays");

        let unnamed = Config {
            tokenizer_path: "no-such-tokenizer".to_string(),
            ..Config::default()
        };
        assert!(unnamed.engine_for("grpc:19000").output_table.is_none());
    }

    /// Writes into the folder `dir` the files of a tokenizer whose
    /// vocabulary is the ids under `size`, one word an id.
    fn write_word_tokenizer(dir: &str, size: u32) {
        let vocab: Vec<String> = (0..size).map(|id| format!(r#""w{id}":{id}"#)).collect();
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"w0"}}}}"#,
            vocab.join(",")
        );
        std::fs::write(Path::new(dir).join("tokenizer.json"), json).expect("tokenizer.json");
    }

    /// Local tokenizer files that give the synthetic output no table owe a
    /// warning: files that do not load, and a tokenizer that decodes no id
    /// of the draw. Files that give a table owe none, and neither does a
    /// path that is no local file, which is expected to give no table.
    #[test]
    fn local_tokenizer_files_that_give_no_table_owe_a_warning() {
        let dir = tempfile::tempdir().expect("temp dir");
        let at = |name: &str| {
            let cfg = Config {
                tokenizer_path: dir.path().join(name).display().to_string(),
                ..Config::default()
            };
            std::fs::create_dir(&cfg.tokenizer_path).expect("a folder");
            cfg
        };

        let unloadable = at("unloadable");
        let (table, warning) = build_output_table(&unloadable);
        assert!(table.is_none());
        let warning = warning.expect("files that do not load");
        assert!(
            warning.starts_with(&format!("tokenizer {}: ", unloadable.tokenizer_path)),
            "{warning}"
        );
        assert!(
            warning.ends_with(": synthetic output keeps the IDs drawn"),
            "{warning}"
        );

        // A vocabulary that ends under the first id of the draw.
        let small = at("small");
        write_word_tokenizer(&small.tokenizer_path, 50);
        let (table, warning) = build_output_table(&small);
        assert!(table.is_none());
        assert_eq!(
            warning,
            Some(format!(
                "tokenizer {}: decodes no ID of the synthetic output range: \
                 synthetic output keeps the IDs drawn",
                small.tokenizer_path
            ))
        );

        let decoding = at("decoding");
        write_word_tokenizer(&decoding.tokenizer_path, 300);
        let (table, warning) = build_output_table(&decoding);
        assert!(table.is_some());
        assert_eq!(warning, None, "files that give a table");

        let hub = Config {
            tokenizer_path: "no-such-org/no-such-model".to_string(),
            ..Config::default()
        };
        assert_eq!(build_output_table(&hub), (None, None), "no local file");
    }

    /// An HTTP worker sends text and no ids: its engine takes no table and
    /// builds none, so nothing is kept for its tokenizer path.
    #[test]
    fn an_http_workers_engine_builds_no_table() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg = Config {
            tokenizer_path: dir.path().display().to_string(),
            ..Config::default()
        };
        // The folder holds no tokenizer yet: a table built now would be kept
        // as none for the path, and the files written next would give none.
        assert!(cfg.engine_for("http:18000").output_table.is_none());
        write_word_tokenizer(&cfg.tokenizer_path, 300);
        assert!(
            cfg.engine_for("grpc:19000").output_table.is_some(),
            "the HTTP worker built none"
        );
        assert!(
            cfg.engine_for("http:18000").output_table.is_none(),
            "the HTTP worker takes none"
        );
    }

    #[test]
    fn a_text_only_answer_takes_its_ids_from_the_tokenizer() {
        use llm_tokenizer::{traits::Encoder, MockTokenizer};
        let tokenizer = MockTokenizer::new();
        let expected: Vec<u32> = tokenizer
            .encode("hello world", false)
            .expect("the mock tokenizer encodes")
            .token_ids()
            .to_vec();
        assert!(!expected.is_empty());

        let text_only = parse(r#"{"output_text": "hello world"}"#)
            .expect("answer")
            .answer;
        let encoded = text_only.encoded_with(&tokenizer).expect("encoded");
        assert_eq!(encoded.output_ids, expected);
        assert_eq!(encoded.text(), "hello world", "the text stays");
        if expected.len() != 1 {
            let with_logprob = parse(r#"{"output_text": "hello world", "logprobs": [-0.5]}"#)
                .expect("answer")
                .answer;
            let refused = with_logprob
                .encoded_with(&tokenizer)
                .expect_err("per-piece logprobs that do not align with the ids are refused");
            assert!(refused.contains("give output_ids"), "{refused}");
        }

        let with_ids = parse(r#"{"output_ids": [4, 5], "output_text": "hello world"}"#)
            .expect("answer")
            .answer;
        let kept = with_ids.clone().encoded_with(&tokenizer).expect("kept");
        assert_eq!(kept, with_ids, "an answer with ids is not re-encoded");
        let ids_only = parse(r#"{"output_ids": [4, 5]}"#).expect("answer").answer;
        assert_eq!(
            ids_only.clone().encoded_with(&tokenizer).expect("kept"),
            ids_only
        );
    }

    #[test]
    fn a_script_file_loads_once_and_needs_a_key_per_line() {
        let mut store = Store::default();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("script.jsonl");
        std::fs::write(
            &path,
            "{\"input_ids\": [5, 6], \"output_ids\": [8, 9], \"finish_reason\": \"length\"}\n\n{\"key\": \"1:4\", \"output_text\": \"four\"}\n",
        )
        .expect("write");
        assert_eq!(store.load(&path).expect("loads"), 2);
        std::fs::remove_file(&path).expect("remove the script: a second load does not read it");
        assert_eq!(
            store.load(&path).expect("loads again"),
            2,
            "a second load is a no-op"
        );
        assert_eq!(store.keyed.len(), 2);
        let answer = store
            .take("http:9000", &script_key(&[5, 6]), None)
            .expect("the keyed answer");
        assert_eq!(
            (answer.output_ids, answer.finish_reason),
            (vec![8, 9], "length".to_string())
        );
        assert_eq!(
            store
                .take("zmq:0", &script_key(&[4]), None)
                .map(|a| a.text()),
            Some("four".to_string())
        );

        let unkeyed = dir.path().join("unkeyed.jsonl");
        std::fs::write(&unkeyed, "{\"output_ids\": [1]}\n").expect("write");
        let error = store
            .load(&unkeyed)
            .expect_err("a line without a key is refused");
        assert!(error.contains("needs key or input_ids"), "{error}");
        assert!(store.load(&dir.path().join("missing.jsonl")).is_err());
    }

    #[test]
    fn a_file_of_hand_written_answers_is_read_alone() {
        // A sibling under the name a recorder gives its info file is another
        // script here: nothing of it is read.
        let mut store = Store::default();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("answers.jsonl");
        let row = "{\"input_ids\": [5, 6], \"output_ids\": [8, 9]}\n";
        std::fs::write(&path, row).expect("write");
        std::fs::write(dir.path().join("answers.info.jsonl"), row).expect("write");
        assert_eq!(store.load(&path).expect("loads"), 1);
        assert!(store.info.is_empty());
    }

    #[test]
    fn a_file_with_no_line_is_a_recording_beside_the_info_file_of_a_recorder() {
        let dir = tempfile::tempdir().expect("temp dir");
        // What a recorder leaves of a session with no call: both files, no
        // line in either.
        let record = dir.path().join("record.jsonl");
        std::fs::write(&record, "").expect("write");
        std::fs::write(dir.path().join("record.info.jsonl"), "").expect("write");
        let mut store = Store::default();
        assert_eq!(store.load(&record).expect("loads"), 0);
        assert!(store.recorded, "every request misses this recording");
        // A script with no line and nothing beside it holds no answer, and
        // fails no request.
        let script = dir.path().join("script.jsonl");
        std::fs::write(&script, "").expect("write");
        let mut store = Store::default();
        assert_eq!(store.load(&script).expect("loads"), 0);
        assert!(!store.recorded);
    }

    #[test]
    fn a_recorded_line_is_filed_under_its_prompt_and_what_its_request_asked() {
        let streamed = json!({"stream": true, "max_new_tokens": 16});
        // The same ask as a writer that keeps another order of names puts it.
        let reordered: Value =
            serde_json::from_str(r#"{"max_new_tokens": 16, "stream": true}"#).expect("json");
        assert_eq!(ask_fold(&streamed), ask_fold(&reordered));
        let plain = json!({"stream": false, "max_new_tokens": 16});
        assert_ne!(ask_fold(&streamed), ask_fold(&plain));
        // One client request as two wires state it folds alike: a default
        // temperature stated or left out, `top_k` off as 0 or -1, a seed or
        // none. A temperature that is not the default is another ask.
        let on_one_wire = json!({"temperature": null, "top_k": 0, "seed": 7, "top_p": 0.95});
        let on_another = json!({"temperature": 1.0, "top_k": -1, "seed": null, "top_p": 0.95});
        let cooler = json!({"temperature": 0.5, "top_k": 0, "seed": 7, "top_p": 0.95});
        assert_eq!(ask_fold(&on_one_wire), ask_fold(&on_another));
        assert_ne!(ask_fold(&on_one_wire), ask_fold(&cooler));

        let mut store = Store::default();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("record.jsonl");
        let line = |output: u32, asked: &Value| {
            json!({"input_ids": [5, 6], "output_ids": [output], "recorded": {"asked": asked}})
                .to_string()
        };
        std::fs::write(
            &path,
            [
                line(1, &streamed),
                line(2, &plain),
                line(3, &reordered),
                // Lines written by hand: their prompt alone, the last one.
                r#"{"input_ids": [7], "output_ids": [8]}"#.to_string(),
                r#"{"input_ids": [7], "output_ids": [9]}"#.to_string(),
            ]
            .join("\n"),
        )
        .expect("write");
        // The info answers the recorder kept beside the file load with it.
        std::fs::write(
            dir.path().join("record.info.jsonl"),
            r#"{"path": "/a.Service/GetModelInfo", "answer_hex": "0a01ff"}"#,
        )
        .expect("write");
        assert_eq!(store.load(&path).expect("loads"), 5);
        assert!(store.recorded);
        assert_eq!(
            store.info.get("/a.Service/GetModelInfo"),
            Some(&vec![0x0a, 0x01, 0xff])
        );

        let worker = "grpc:9000";
        let key = script_key(&[5, 6]);
        let mut served = |ask: &Value| {
            store
                .take_n(worker, &key, Some(ask_fold(ask)), None, 1)
                .expect("one choice")
                .map(|answers| answers[0].output_ids.clone())
        };
        // Whatever order the requests arrive in, each reads a line recorded
        // for what it asks: the lines of one ask in turn, then from the
        // first again.
        assert_eq!(served(&plain), Some(vec![2]));
        assert_eq!(served(&streamed), Some(vec![1]));
        assert_eq!(served(&streamed), Some(vec![3]));
        assert_eq!(served(&streamed), Some(vec![1]));
        assert_eq!(served(&plain), Some(vec![2]));
        // The prompt asked otherwise, and the prompt on a wire that states
        // no ask: nothing is filed for either.
        assert_eq!(served(&json!({"stream": true, "max_new_tokens": 17})), None);
        assert!(store.take(worker, &key, None).is_none());
        assert_eq!(store.misses.get(worker), Some(&2));

        // A line written by hand serves its prompt whatever is asked.
        let by_hand = script_key(&[7]);
        for ask in [None, Some(ask_fold(&streamed))] {
            let answers = store
                .take_n(worker, &by_hand, ask, None, 1)
                .expect("one choice")
                .expect("the keyed answer");
            assert_eq!(answers[0].output_ids, vec![9]);
        }
    }

    /// Cuts whose sum wraps around to the output's own count are refused at
    /// the load, and an answer built with such cuts streams even frames: no
    /// frame is sliced by them.
    #[test]
    fn cuts_that_add_up_past_any_count_are_refused() {
        let wrapped = format!(
            r#"{{"output_ids": [1, 2, 3], "chunks": [{}, 4]}}"#,
            usize::MAX
        );
        assert!(parse(&wrapped).is_err());

        let mut built = parse(r#"{"output_ids": [1, 2, 3]}"#)
            .expect("an answer of three tokens")
            .answer;
        built.chunks = Some(vec![usize::MAX, 4]);
        let streamed: usize = (built.token_frames().iter())
            .map(|frame| frame.ids.len())
            .sum();
        assert_eq!(streamed, 3);
    }

    #[test]
    fn a_status_states_the_code_of_a_failure() {
        // OK ends no call ahead of its answer and a number past the last
        // code names none: a line that states either is refused.
        let stating = |code: i32| {
            parse(&format!(
                r#"{{"output_text": "", "status": {{"code": {code}}}}}"#
            ))
        };
        for code in [0, 17, -1] {
            assert!(
                stating(code).is_err_and(|refused| refused.contains("1 to 16")),
                "code {code} is no code of a failure"
            );
        }
        for code in [1, 16] {
            let status = stating(code).expect("the code of a failure").answer.status;
            assert_eq!(status.map(|status| status.code), Some(code));
        }
    }

    #[test]
    fn a_cut_of_no_tokens_is_a_frame_and_a_stated_count_is_the_one_reported() {
        // An engine's stream whose one token was the stop it trimmed: an
        // empty chunk opens the choice, and the completion counts the token.
        let opened = parse(r#"{"output_text": "", "chunks": [0], "completion_tokens": 1}"#)
            .expect("a cut of no tokens")
            .answer;
        let frames = opened.token_frames();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].ids.is_empty() && frames[0].last);
        assert_eq!(opened.completion_tokens(), 1);

        // The cuts still add up to the output's tokens, the empty ones among
        // them; with no count stated the output's own is reported.
        let cuts = parse(r#"{"output_ids": [1, 2, 3], "chunks": [0, 2, 0, 1]}"#)
            .expect("empty cuts among the others")
            .answer;
        let sizes: Vec<usize> = (cuts.token_frames().iter())
            .map(|frame| frame.ids.len())
            .collect();
        assert_eq!(sizes, [0, 2, 0, 1]);
        assert_eq!(cuts.completion_tokens(), 3);
        assert!(parse(r#"{"output_ids": [1, 2], "chunks": [1]}"#).is_err());

        // Cuts that hold none of an output: the engine sent it in its
        // completion alone. A gRPC wire streams a frame of no tokens for
        // each cut (none for an empty list); the frames of the other wires
        // are the output, so they deliver it in even cuts.
        for (stated, empty_frames) in [("[]", 0), ("[0]", 1), ("[0, 0]", 2)] {
            let alone = parse(&format!(
                r#"{{"output_ids": [7, 8, 9], "chunks": {stated}}}"#
            ))
            .expect("cuts that hold none of the output")
            .answer;
            let frames = alone.chunk_frames();
            assert_eq!(frames.len(), empty_frames, "{stated}");
            assert!(frames.iter().all(|frame| frame.ids.is_empty()), "{stated}");
            assert_eq!(
                frames.last().map(|frame| frame.last),
                (empty_frames > 0).then_some(true)
            );
            let sizes: Vec<usize> = (alone.token_frames().iter())
                .map(|frame| frame.ids.len())
                .collect();
            assert_eq!(sizes, [1, 1, 1], "{stated}");
            assert_eq!(alone.completion_tokens(), 3);
        }
        // Cuts that hold the output are the frames of every wire.
        assert_eq!(cuts.chunk_frames(), cuts.token_frames());
        // An output of text alone has no ids for a frame of a gRPC wire, so
        // cuts that hold none of it are its frames too, one of no tokens each.
        let text = parse(r#"{"output_text": "ab", "chunks": [0, 0]}"#)
            .expect("cuts that hold none of a text output")
            .answer;
        let sizes: Vec<usize> = (text.chunk_frames().iter())
            .map(|frame| frame.ids.len() + frame.pieces.len())
            .collect();
        assert_eq!(sizes, [0, 0]);

        // A request's limit cuts the answer short of the count it stated.
        let cut = parse(r#"{"output_ids": [1, 2, 3], "completion_tokens": 4}"#)
            .expect("a stated count")
            .answer
            .cut_to(Some(2));
        assert_eq!(cut.completion_tokens(), 2);
    }

    #[test]
    fn a_queued_answer_leaves_the_calls_of_a_key_whole() {
        // Two recorded calls of two choices under one key, and one answer
        // queued ahead of them.
        let mut store = Store::default();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("record.jsonl");
        let asked = json!({"n": 2});
        let lines: Vec<String> = (1..=4)
            .map(|output| {
                json!({"input_ids": [5, 6], "output_ids": [output], "recorded": {"asked": asked}})
                    .to_string()
            })
            .collect();
        std::fs::write(&path, lines.join("\n")).expect("write");
        assert_eq!(store.load(&path).expect("loads"), 4);
        let worker = "grpc:9000";
        let queued = parse(r#"{"output_ids": [9]}"#).expect("an answer");
        store.enqueue(worker, vec![queued]);
        let key = script_key(&[5, 6]);
        let mut call = || {
            let answers = store
                .take_n(worker, &key, Some(ask_fold(&asked)), None, 2)
                .expect("two choices")
                .expect("the recorded call");
            [answers[0].output_ids[0], answers[1].output_ids[0]]
        };
        // The queued answer stands for the first choice of its call; the
        // call behind it reads its own two lines, not one of each call.
        assert_eq!([call(), call(), call()], [[9, 2], [3, 4], [1, 2]]);
    }

    #[test]
    fn a_logprob_json_has_no_number_for_reads_back_as_written() {
        // What the recorder writes for a value JSON cannot state.
        let scored = parse(r#"{"output_ids": [1, 2], "logprobs": ["-inf", -0.5]}"#)
            .expect("a logprob of no number")
            .answer;
        assert_eq!(scored.logprobs, Some(vec![f32::NEG_INFINITY, -0.5]));
        assert!(parse(r#"{"output_ids": [1], "logprobs": ["-0.5"]}"#).is_err());
    }
}
