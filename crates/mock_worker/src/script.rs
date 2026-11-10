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

use std::{
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

use crate::config::Config;

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
    ///  "cached_tokens": 0, "prompt_tokens": 12, "logprobs": [-0.1, ...],
    ///  "token_id_logprobs": [-0.2, ...], "delay_ms": 0, "chunk_tokens": 1}
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
            "logprobs",
            "token_id_logprobs",
            "delay_ms",
            "chunk_tokens",
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
                        value
                            .as_f64()
                            .map(|v| v as f32)
                            .ok_or_else(|| format!("logprobs: {value} is not a number"))
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
                        value
                            .as_f64()
                            .map(|v| v as f32)
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
                logprobs,
                sampled_logprobs,
                token_id_logprobs,
                delay,
                chunk_tokens,
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
        self
    }

    /// The prompt tokens to report for a prompt of `prompt_len` ids.
    pub fn prompt_tokens(&self, prompt_len: usize) -> u32 {
        self.prompt_tokens
            .unwrap_or_else(|| u32::try_from(prompt_len).unwrap_or(u32::MAX))
    }

    /// The completion tokens to report: the output ids, or the text pieces
    /// when the answer has no ids.
    pub fn completion_tokens(&self) -> u32 {
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
        let cuts: Vec<usize> = (0..units).step_by(step).collect();
        let count = cuts.len();
        cuts.into_iter()
            .enumerate()
            .map(|(n, start)| {
                let range = start..(start + step).min(units);
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

#[derive(Default)]
struct Store {
    /// Answers by prompt key, from `--script` lines and keyed enqueues.
    keyed: HashMap<String, Answer>,
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
        self.activate();
        let count = answers.len();
        self.loaded.insert(path.to_path_buf(), count);
        self.keyed.extend(answers);
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
        self.take_n(worker, key, prompt_text, 1)
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
    /// prompt has `key` and reads `prompt_text` (`None`: unread, no answer
    /// filed under a text is looked at), taken under this one lock: the
    /// worker's queue, then the shared one, then the answer filed under a
    /// text the prompt holds (served once) or the keyed answer for the rest.
    /// `None` when nothing is scripted for the first choice (the request
    /// takes the worker's other mode, whatever `n`); an `Err` when `n` is
    /// above [`CHOICES_LIMIT`], or naming the counts when the queues hold
    /// some of the choices but not all and nothing fills the rest, with the
    /// queues as they were.
    fn take_n(
        &mut self,
        worker: &str,
        key: &str,
        prompt_text: Option<&str>,
        n: usize,
    ) -> Result<Option<Vec<Answer>>, String> {
        let queued = [worker, ALL]
            .iter()
            .map(|queue| self.queues.get(*queue).map_or(0, VecDeque::len))
            .sum::<usize>();
        let marked = prompt_text.and_then(|text| self.marked_for(worker, text));
        let keyed = self.keyed.get(key);
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
            if let Some(answer) = filed.as_ref().or(keyed) {
                while answers.len() < n {
                    answers.push(answer.clone());
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
                    self.keyed.insert(key, scripted.answer);
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
            "keyed": self.keyed.len(),
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
}

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
/// Whether anything was ever scripted in the process, readable without the
/// lock: until then a request's path does not touch the store at all.
static ACTIVE: AtomicBool = AtomicBool::new(false);

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

/// A worker that started, by name (`grpc:<port>`, `http:<port>`,
/// `zmq:<index>`): the admin routes accept it from then on.
pub fn register(worker: &str) {
    store().workers.insert(worker.to_string());
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
        .or_insert_with(|| {
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
        })
        .clone()
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
    let key = script_key(prompt_ids);
    store()
        .take_n(worker, &key, prompt_text, n.max(1))
        .map_err(|why| {
            tracing::error!("{worker}: {why}");
            why
        })
}

/// [`take_per_choice`] on a token wire: the prompt read for the answers
/// filed under a text (`rendered`, the text the request carries beside its
/// ids, else the ids decoded), and the text-only answers encoded for the
/// wire (outside the lock), as [`take_ids`] reads and encodes.
pub fn take_ids_per_choice(
    cfg: &Config,
    worker: &str,
    prompt_ids: &[u32],
    rendered: &str,
    n: usize,
) -> Result<Option<Vec<Answer>>, String> {
    if !active() {
        return Ok(None);
    }
    let prompt_text = prompt_text_when_marked(cfg, worker, prompt_ids, rendered)?;
    let Some(answers) = take_per_choice(worker, prompt_ids, prompt_text.as_deref(), n)? else {
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
            r#"[1]"#,
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
            .take_n(worker, &key, Some("say apricot"), 2)
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
}
