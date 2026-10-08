// SGLang per-step batched output on SMG's wire — `BatchTokenIDSlimOutput`,
// declared by the SMG plugin that runs inside the scheduler
// (`smg_grpc_servicer.sglang.zmq_msgpack`): a tagged positional msgpack array
// of parallel per-request columns. **Field order is the wire contract** — do
// not reorder; the plugin appends only.

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::OpaqueValue,
    error::{Error, Result},
    protocol::{
        positional::{drain_trailing, expect_tag, next_field},
        EngineOutput,
    },
};

/// The msgspec tag for [`BatchTokenIDSlimOutput`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_SLIM_OUTPUT_TAG: &str = "BatchTokenIDSlimOutput";
/// The msgspec tag for [`BatchEmbeddingSlimOutput`] (element 0 on the wire).
pub const BATCH_EMBEDDING_SLIM_OUTPUT_TAG: &str = "BatchEmbeddingSlimOutput";
/// The msgspec tag for [`ControlReplySlim`] (element 0 on the wire).
pub const CONTROL_REPLY_SLIM_TAG: &str = "ControlReplySlim";

/// What a request stopped on: the scheduler reports the matched stop token id
/// or stop string in its finish reason, and the plugin forwards it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MatchedStop {
    TokenId(u32),
    Text(String),
}

/// A batch of per-request token outputs from one scheduler step.
///
/// Every field is a column indexed in parallel by request: `rids[i]` owns
/// `output_ids[i]`, `finished_reasons[i]`, and so on. The logprob columns are
/// always present (length == `rids.len()`); the inner `Vec` is empty for a
/// request that did not ask for logprobs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchTokenIDSlimOutput {
    pub rids: Vec<String>,
    /// Newly generated token ids per request this step.
    pub output_ids: Vec<Vec<u32>>,
    /// `""` while generating, else `stop` | `length` | `abort`.
    pub finished_reasons: Vec<String>,
    /// The abort message, when the finish was an abort.
    pub finished_messages: Vec<Option<String>>,
    /// The matched stop, when the finish was a stop on a token or string.
    pub finished_matched: Vec<Option<MatchedStop>>,
    pub prompt_tokens: Vec<u32>,
    /// Completion token count so far, per request.
    pub completion_tokens: Vec<u32>,
    pub cached_tokens: Vec<u32>,
    /// Sampled-token logprob per newly decoded token, per request.
    pub output_token_logprobs_val: Vec<Vec<f64>>,
    /// The token id each logprob belongs to, parallel to the values.
    pub output_token_logprobs_idx: Vec<Vec<u32>>,
    /// Producing rank's engine index (the PULL socket carries no identity).
    pub engine_index: u32,
    /// Scheduler load sampled at send time, so the output batch is the one
    /// in-band load channel. `kv_total_tokens == 0` means no snapshot.
    pub num_running: u64,
    pub num_waiting: u64,
    pub kv_used_tokens: u64,
    pub kv_total_tokens: u64,
    /// HTTP status carried by an `abort` finish (SGLang stamps 400 on a
    /// request it refuses, as does the plugin's own validation); `None` for
    /// other finishes. Empty from a plugin that predates the column.
    pub finished_status: Vec<Option<u16>>,
    /// Ranked candidates per newly decoded token (`top_logprobs_num` of them)
    /// per request, parallel to the sampled-token logprob columns; empty when
    /// not requested or from a plugin that predates the columns.
    pub output_top_logprobs_val: Vec<Vec<Vec<f64>>>,
    pub output_top_logprobs_idx: Vec<Vec<Vec<u32>>>,
    /// Reasoning tokens counted so far (hybrid-reasoning models); appended.
    pub reasoning_tokens: Vec<u32>,
    /// Prompt logprobs when `logprob_start_len >= 0`: the sampled prompt
    /// token's logprob per position (`None` for the first, which has no
    /// predecessor), its id, and the ranked candidates per position when
    /// `top_logprobs_num > 0`; empty per request otherwise. Appended.
    pub input_token_logprobs_val: Vec<Vec<Option<f64>>>,
    pub input_token_logprobs_idx: Vec<Vec<u32>>,
    pub input_top_logprobs_val: Vec<Vec<Vec<f64>>>,
    pub input_top_logprobs_idx: Vec<Vec<Vec<u32>>>,
    /// Requested candidate logprobs per output position, per request. A
    /// prefill-only score carries one row even with no generated tokens.
    /// Appended; absent from older plugins.
    pub output_token_ids_logprobs_val: Vec<Vec<Vec<f64>>>,
    pub output_token_ids_logprobs_idx: Vec<Vec<Vec<u32>>>,
}

impl Serialize for BatchTokenIDSlimOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let has_selected = !self.output_token_ids_logprobs_val.is_empty()
            || !self.output_token_ids_logprobs_idx.is_empty();
        let mut tuple = serializer.serialize_tuple(if has_selected { 26 } else { 24 })?;
        tuple.serialize_element(BATCH_TOKEN_ID_SLIM_OUTPUT_TAG)?;
        tuple.serialize_element(&self.rids)?;
        tuple.serialize_element(&self.output_ids)?;
        tuple.serialize_element(&self.finished_reasons)?;
        tuple.serialize_element(&self.finished_messages)?;
        tuple.serialize_element(&self.finished_matched)?;
        tuple.serialize_element(&self.prompt_tokens)?;
        tuple.serialize_element(&self.completion_tokens)?;
        tuple.serialize_element(&self.cached_tokens)?;
        tuple.serialize_element(&self.output_token_logprobs_val)?;
        tuple.serialize_element(&self.output_token_logprobs_idx)?;
        tuple.serialize_element(&self.engine_index)?;
        tuple.serialize_element(&self.num_running)?;
        tuple.serialize_element(&self.num_waiting)?;
        tuple.serialize_element(&self.kv_used_tokens)?;
        tuple.serialize_element(&self.kv_total_tokens)?;
        tuple.serialize_element(&self.finished_status)?;
        tuple.serialize_element(&self.output_top_logprobs_val)?;
        tuple.serialize_element(&self.output_top_logprobs_idx)?;
        tuple.serialize_element(&self.reasoning_tokens)?;
        tuple.serialize_element(&self.input_token_logprobs_val)?;
        tuple.serialize_element(&self.input_token_logprobs_idx)?;
        tuple.serialize_element(&self.input_top_logprobs_val)?;
        tuple.serialize_element(&self.input_top_logprobs_idx)?;
        if has_selected {
            tuple.serialize_element(&self.output_token_ids_logprobs_val)?;
            tuple.serialize_element(&self.output_token_ids_logprobs_idx)?;
        }
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for BatchTokenIDSlimOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct BatchVisitor;

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = BatchTokenIDSlimOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged BatchTokenIDSlimOutput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, BATCH_TOKEN_ID_SLIM_OUTPUT_TAG)?;
                read_token_batch_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

/// An optional appended column: absent, or `None`, from an older plugin.
fn appended_column<'de, A: SeqAccess<'de>, T: Deserialize<'de>>(
    seq: &mut A,
) -> std::result::Result<Vec<T>, A::Error> {
    Ok(seq
        .next_element::<Option<Vec<T>>>()?
        .flatten()
        .unwrap_or_default())
}

/// The token batch's columns after the tag; shared by the struct's own
/// decoder and the wire enum's.
fn read_token_batch_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> std::result::Result<BatchTokenIDSlimOutput, A::Error> {
    let batch = BatchTokenIDSlimOutput {
        rids: next_field(seq, "rids")?,
        output_ids: next_field(seq, "output_ids")?,
        finished_reasons: next_field(seq, "finished_reasons")?,
        finished_messages: next_field(seq, "finished_messages")?,
        finished_matched: next_field(seq, "finished_matched")?,
        prompt_tokens: next_field(seq, "prompt_tokens")?,
        completion_tokens: next_field(seq, "completion_tokens")?,
        cached_tokens: next_field(seq, "cached_tokens")?,
        output_token_logprobs_val: next_field(seq, "output_token_logprobs_val")?,
        output_token_logprobs_idx: next_field(seq, "output_token_logprobs_idx")?,
        // The tail has msgspec defaults; a shorter array from an
        // older plugin decodes as "rank 0, no snapshot".
        engine_index: seq.next_element::<u32>()?.unwrap_or(0),
        num_running: seq.next_element::<u64>()?.unwrap_or(0),
        num_waiting: seq.next_element::<u64>()?.unwrap_or(0),
        kv_used_tokens: seq.next_element::<u64>()?.unwrap_or(0),
        kv_total_tokens: seq.next_element::<u64>()?.unwrap_or(0),
        finished_status: seq
            .next_element::<Option<Vec<Option<u16>>>>()?
            .flatten()
            .unwrap_or_default(),
        output_top_logprobs_val: seq
            .next_element::<Option<Vec<Vec<Vec<f64>>>>>()?
            .flatten()
            .unwrap_or_default(),
        output_top_logprobs_idx: seq
            .next_element::<Option<Vec<Vec<Vec<u32>>>>>()?
            .flatten()
            .unwrap_or_default(),
        reasoning_tokens: appended_column(seq)?,
        input_token_logprobs_val: appended_column(seq)?,
        input_token_logprobs_idx: appended_column(seq)?,
        input_top_logprobs_val: appended_column(seq)?,
        input_top_logprobs_idx: appended_column(seq)?,
        output_token_ids_logprobs_val: appended_column(seq)?,
        output_token_ids_logprobs_idx: appended_column(seq)?,
    };
    drain_trailing(seq)?;
    Ok(batch)
}

/// One request's slice of a [`BatchTokenIDSlimOutput`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SglangOutput {
    pub request_id: String,
    pub output_ids: Vec<u32>,
    /// `None` while the request is still generating.
    pub finish_reason: Option<String>,
    /// The abort message, when the finish was an abort.
    pub finish_message: Option<String>,
    /// The HTTP status an abort carried, when the scheduler reported one.
    pub finish_status: Option<u16>,
    pub matched_stop: Option<MatchedStop>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub output_logprobs_val: Vec<f64>,
    pub output_logprobs_idx: Vec<u32>,
    /// Ranked candidates per newly decoded token, when requested.
    pub output_top_logprobs_val: Vec<Vec<f64>>,
    pub output_top_logprobs_idx: Vec<Vec<u32>>,
    /// Reasoning tokens counted so far (0 when the model has no such phase).
    pub reasoning_tokens: u32,
    /// Prompt logprobs, delivered once with the prefill tick (empty otherwise).
    pub input_logprobs_val: Vec<Option<f64>>,
    pub input_logprobs_idx: Vec<u32>,
    pub input_top_logprobs_val: Vec<Vec<f64>>,
    pub input_top_logprobs_idx: Vec<Vec<u32>>,
    /// Raw full-vocabulary logprobs for the requested candidates. Independent
    /// of sampled-token/top-k output, including prefill-only scoring.
    pub output_token_ids_logprobs_val: Vec<Vec<f64>>,
    pub output_token_ids_logprobs_idx: Vec<Vec<u32>>,
    /// A selected-score shape error belongs to this request. Keeping it on
    /// the routed output prevents isolated malformed batches being dropped
    /// by the connector and leaving their requests waiting indefinitely.
    pub selected_logprobs_error: Option<String>,
    /// The pooled vector of an embedding request (its one, finished, output).
    pub embedding: Option<Vec<f32>>,
}

impl EngineOutput for SglangOutput {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

impl BatchTokenIDSlimOutput {
    /// Split the parallel columns into one [`SglangOutput`] per request.
    /// Errors if the columns are ragged (a length mismatch is a protocol bug).
    pub fn into_outputs(self) -> Result<Vec<SglangOutput>> {
        let n = self.rids.len();
        let columns = [
            ("output_ids", self.output_ids.len()),
            ("finished_reasons", self.finished_reasons.len()),
            ("finished_messages", self.finished_messages.len()),
            ("finished_matched", self.finished_matched.len()),
            ("prompt_tokens", self.prompt_tokens.len()),
            ("completion_tokens", self.completion_tokens.len()),
            ("cached_tokens", self.cached_tokens.len()),
            (
                "output_token_logprobs_val",
                self.output_token_logprobs_val.len(),
            ),
            (
                "output_token_logprobs_idx",
                self.output_token_logprobs_idx.len(),
            ),
        ];
        // Appended columns are optional as a whole (older plugins omit them).
        let optional = [
            ("finished_status", self.finished_status.len()),
            (
                "output_top_logprobs_val",
                self.output_top_logprobs_val.len(),
            ),
            (
                "output_top_logprobs_idx",
                self.output_top_logprobs_idx.len(),
            ),
            ("reasoning_tokens", self.reasoning_tokens.len()),
            (
                "input_token_logprobs_val",
                self.input_token_logprobs_val.len(),
            ),
            (
                "input_token_logprobs_idx",
                self.input_token_logprobs_idx.len(),
            ),
            ("input_top_logprobs_val", self.input_top_logprobs_val.len()),
            ("input_top_logprobs_idx", self.input_top_logprobs_idx.len()),
        ];
        if let Some((name, len)) = columns
            .iter()
            .copied()
            .chain(optional.into_iter().filter(|(_, len)| *len > 0))
            .find(|(_, len)| *len != n)
        {
            return Err(Error::Decode {
                target_type: "BatchTokenIDSlimOutput",
                message: format!("column `{name}` has {len} entries for {n} rids"),
            });
        }
        let selected_columns_valid = (self.output_token_ids_logprobs_val.is_empty()
            && self.output_token_ids_logprobs_idx.is_empty())
            || (self.output_token_ids_logprobs_val.len() == n
                && self.output_token_ids_logprobs_idx.len() == n);
        let mut finished_reasons = self.finished_reasons.into_iter();
        let mut finished_messages = self.finished_messages.into_iter();
        let mut finished_status = self.finished_status.into_iter();
        let mut top_val = self.output_top_logprobs_val.into_iter();
        let mut top_idx = self.output_top_logprobs_idx.into_iter();
        let mut finished_matched = self.finished_matched.into_iter();
        let mut prompt_tokens = self.prompt_tokens.into_iter();
        let mut completion_tokens = self.completion_tokens.into_iter();
        let mut cached_tokens = self.cached_tokens.into_iter();
        let mut logprobs_val = self.output_token_logprobs_val.into_iter();
        let mut logprobs_idx = self.output_token_logprobs_idx.into_iter();
        let mut reasoning_tokens = self.reasoning_tokens.into_iter();
        let mut input_val = self.input_token_logprobs_val.into_iter();
        let mut input_idx = self.input_token_logprobs_idx.into_iter();
        let mut input_top_val = self.input_top_logprobs_val.into_iter();
        let mut input_top_idx = self.input_top_logprobs_idx.into_iter();
        let mut selected_val = self.output_token_ids_logprobs_val.into_iter();
        let mut selected_idx = self.output_token_ids_logprobs_idx.into_iter();
        Ok(self
            .rids
            .into_iter()
            .zip(self.output_ids)
            .map(|(request_id, output_ids)| {
                let reason = finished_reasons.next().unwrap_or_default();
                let selected_values = selected_val.next().unwrap_or_default();
                let selected_ids = selected_idx.next().unwrap_or_default();
                let selected_logprobs_error = (!selected_columns_valid
                    || selected_values.len() != selected_ids.len()
                    || selected_values
                        .iter()
                        .zip(&selected_ids)
                        .any(|(values, ids)| values.len() != ids.len()))
                .then(|| "selected-token logprobs have mismatched values and IDs".to_string());
                SglangOutput {
                    request_id,
                    output_ids,
                    finish_reason: (!reason.is_empty()).then_some(reason),
                    finish_message: finished_messages.next().flatten(),
                    finish_status: finished_status.next().flatten(),
                    matched_stop: finished_matched.next().flatten(),
                    prompt_tokens: prompt_tokens.next().unwrap_or(0),
                    completion_tokens: completion_tokens.next().unwrap_or(0),
                    cached_tokens: cached_tokens.next().unwrap_or(0),
                    output_logprobs_val: logprobs_val.next().unwrap_or_default(),
                    output_logprobs_idx: logprobs_idx.next().unwrap_or_default(),
                    output_top_logprobs_val: top_val.next().unwrap_or_default(),
                    output_top_logprobs_idx: top_idx.next().unwrap_or_default(),
                    reasoning_tokens: reasoning_tokens.next().unwrap_or(0),
                    input_logprobs_val: input_val.next().unwrap_or_default(),
                    input_logprobs_idx: input_idx.next().unwrap_or_default(),
                    input_top_logprobs_val: input_top_val.next().unwrap_or_default(),
                    input_top_logprobs_idx: input_top_idx.next().unwrap_or_default(),
                    output_token_ids_logprobs_val: selected_values,
                    output_token_ids_logprobs_idx: selected_ids,
                    selected_logprobs_error,
                    embedding: None,
                }
            })
            .collect())
    }
}

/// The plugin's per-step batch for embedding requests: each request's pooled
/// vector with its counts and finish (`stop`, or `abort` with the message and
/// status when the scheduler could not serve it), plus the load tail.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchEmbeddingSlimOutput {
    pub rids: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: Vec<u32>,
    pub cached_tokens: Vec<u32>,
    pub finished_reasons: Vec<String>,
    pub finished_messages: Vec<Option<String>>,
    pub finished_status: Vec<Option<u16>>,
    pub engine_index: u32,
    pub num_running: u64,
    pub num_waiting: u64,
    pub kv_used_tokens: u64,
    pub kv_total_tokens: u64,
}

impl Serialize for BatchEmbeddingSlimOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(13)?;
        tuple.serialize_element(BATCH_EMBEDDING_SLIM_OUTPUT_TAG)?;
        tuple.serialize_element(&self.rids)?;
        tuple.serialize_element(&self.embeddings)?;
        tuple.serialize_element(&self.prompt_tokens)?;
        tuple.serialize_element(&self.cached_tokens)?;
        tuple.serialize_element(&self.finished_reasons)?;
        tuple.serialize_element(&self.finished_messages)?;
        tuple.serialize_element(&self.finished_status)?;
        tuple.serialize_element(&self.engine_index)?;
        tuple.serialize_element(&self.num_running)?;
        tuple.serialize_element(&self.num_waiting)?;
        tuple.serialize_element(&self.kv_used_tokens)?;
        tuple.serialize_element(&self.kv_total_tokens)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for BatchEmbeddingSlimOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct BatchVisitor;

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = BatchEmbeddingSlimOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged BatchEmbeddingSlimOutput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, BATCH_EMBEDDING_SLIM_OUTPUT_TAG)?;
                read_embedding_batch_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

fn read_embedding_batch_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> std::result::Result<BatchEmbeddingSlimOutput, A::Error> {
    let batch = BatchEmbeddingSlimOutput {
        rids: next_field(seq, "rids")?,
        embeddings: next_field(seq, "embeddings")?,
        prompt_tokens: next_field(seq, "prompt_tokens")?,
        cached_tokens: next_field(seq, "cached_tokens")?,
        finished_reasons: next_field(seq, "finished_reasons")?,
        finished_messages: next_field(seq, "finished_messages")?,
        finished_status: next_field(seq, "finished_status")?,
        engine_index: seq.next_element::<u32>()?.unwrap_or(0),
        num_running: seq.next_element::<u64>()?.unwrap_or(0),
        num_waiting: seq.next_element::<u64>()?.unwrap_or(0),
        kv_used_tokens: seq.next_element::<u64>()?.unwrap_or(0),
        kv_total_tokens: seq.next_element::<u64>()?.unwrap_or(0),
    };
    drain_trailing(seq)?;
    Ok(batch)
}

impl BatchEmbeddingSlimOutput {
    /// One finished [`SglangOutput`] per request; an `abort` carries no vector.
    pub fn into_outputs(self) -> Result<Vec<SglangOutput>> {
        let n = self.rids.len();
        let columns = [
            ("embeddings", self.embeddings.len()),
            ("prompt_tokens", self.prompt_tokens.len()),
            ("cached_tokens", self.cached_tokens.len()),
            ("finished_reasons", self.finished_reasons.len()),
            ("finished_messages", self.finished_messages.len()),
            ("finished_status", self.finished_status.len()),
        ];
        if let Some((name, len)) = columns.iter().copied().find(|(_, len)| *len != n) {
            return Err(Error::Decode {
                target_type: "BatchEmbeddingSlimOutput",
                message: format!("column `{name}` has {len} entries for {n} rids"),
            });
        }
        let mut embeddings = self.embeddings.into_iter();
        let mut prompt_tokens = self.prompt_tokens.into_iter();
        let mut cached_tokens = self.cached_tokens.into_iter();
        let mut finished_messages = self.finished_messages.into_iter();
        let mut finished_status = self.finished_status.into_iter();
        Ok(self
            .rids
            .into_iter()
            .zip(self.finished_reasons)
            .map(|(request_id, reason)| {
                let embedding = embeddings.next().unwrap_or_default();
                SglangOutput {
                    request_id,
                    // An embedding output is terminal by construction.
                    finish_reason: Some(if reason.is_empty() {
                        "stop".to_string()
                    } else {
                        reason.clone()
                    }),
                    finish_message: finished_messages.next().flatten(),
                    finish_status: finished_status.next().flatten(),
                    prompt_tokens: prompt_tokens.next().unwrap_or(0),
                    cached_tokens: cached_tokens.next().unwrap_or(0),
                    embedding: (reason != "abort").then_some(embedding),
                    ..SglangOutput::default()
                }
            })
            .collect())
    }
}

/// The plugin's answer to a control call (`ControlReplySlim`): the
/// scheduler's `success`/`message` for the call id SMG issued.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ControlReplySlim {
    pub call_id: i64,
    pub success: bool,
    pub message: Option<String>,
    pub engine_index: u32,
}

impl Serialize for ControlReplySlim {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(5)?;
        tuple.serialize_element(CONTROL_REPLY_SLIM_TAG)?;
        tuple.serialize_element(&self.call_id)?;
        tuple.serialize_element(&self.success)?;
        tuple.serialize_element(&self.message)?;
        tuple.serialize_element(&self.engine_index)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for ControlReplySlim {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct ReplyVisitor;

        impl<'de> Visitor<'de> for ReplyVisitor {
            type Value = ControlReplySlim;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged ControlReplySlim positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, CONTROL_REPLY_SLIM_TAG)?;
                read_control_reply_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(ReplyVisitor)
    }
}

fn read_control_reply_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> std::result::Result<ControlReplySlim, A::Error> {
    let reply = ControlReplySlim {
        call_id: next_field(seq, "call_id")?,
        success: next_field(seq, "success")?,
        message: next_field(seq, "message")?,
        engine_index: seq.next_element::<u32>()?.unwrap_or(0),
    };
    drain_trailing(seq)?;
    Ok(reply)
}

impl ControlReplySlim {
    /// The reply as a utility outcome: a map of `success` and `message`, so
    /// a refused flush (success false) is a reply, not a transport failure.
    pub fn into_outcome(self) -> OpaqueValue {
        OpaqueValue::Map(vec![
            (
                OpaqueValue::from("success"),
                OpaqueValue::Boolean(self.success),
            ),
            (
                OpaqueValue::from("message"),
                self.message.map_or(OpaqueValue::Nil, OpaqueValue::from),
            ),
        ])
    }
}

/// One output message on the wire, told apart by its tag. The token batch is
/// boxed: it is by far the largest and the common case.
#[derive(Debug, Clone, PartialEq)]
pub enum SglangWireOutput {
    Tokens(Box<BatchTokenIDSlimOutput>),
    Embeddings(BatchEmbeddingSlimOutput),
    ControlReply(ControlReplySlim),
}

impl<'de> Deserialize<'de> for SglangWireOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct WireVisitor;

        impl<'de> Visitor<'de> for WireVisitor {
            type Value = SglangWireOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged SGLang slim output positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let tag: String = next_field(&mut seq, "tag")?;
                match tag.as_str() {
                    BATCH_TOKEN_ID_SLIM_OUTPUT_TAG => read_token_batch_after_tag(&mut seq)
                        .map(|batch| SglangWireOutput::Tokens(Box::new(batch))),
                    BATCH_EMBEDDING_SLIM_OUTPUT_TAG => {
                        read_embedding_batch_after_tag(&mut seq).map(SglangWireOutput::Embeddings)
                    }
                    CONTROL_REPLY_SLIM_TAG => {
                        read_control_reply_after_tag(&mut seq).map(SglangWireOutput::ControlReply)
                    }
                    other => Err(serde::de::Error::custom(format!(
                        "unknown SGLang output tag `{other}`"
                    ))),
                }
            }
        }

        deserializer.deserialize_seq(WireVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_msgpack, encode_msgpack};

    /// `msgspec.msgpack.encode` of the plugin's slim batch for rids a/b with
    /// b finished on token 42 with logprobs and two ranked candidates on its
    /// last token, load 2/3/40/400 on engine 1, no abort status.
    const PYTHON_SLIM: &str = "dc0013b64261746368546f6b656e4944536c696d4f757470757492a161a16292910a92141592a0a473746f7092c0c092c02a920304920102920001929092cbbfe0000000000000cbbfd0000000000000929092141501020328cd019092c0c092909192cbbfe0000000000000cbbff0000000000000929091921516";
    /// The same batch from the earlier 24-element plugin: `b` with 4
    /// reasoning tokens and prompt logprobs for its three prompt tokens (no
    /// value for the first), one ranked candidate per later position.
    const PYTHON_SLIM24: &str = "dc0018b64261746368546f6b656e4944536c696d4f757470757492a161a16292910a92141592a0a473746f7092c0c092c02a920304920102920001929092cbbfe0000000000000cbbfd00000000000009290921415000000000092c0c092909192cbbfe0000000000000cbbff0000000000000929091921516920004929093c0cbbfe6666666666666cbbff199999999999a9290930102039290939091cbbfe666666666666691cbbff199999999999a9290939091029103";
    /// Python plugin's 26-element score-only result: no generated tokens,
    /// candidate IDs 42 and 5 with raw logprobs -0.25 and -12.0.
    const PYTHON_SCORE26: &str = "dc001ab64261746368546f6b656e4944536c696d4f757470757491a573636f7265919091a66c656e67746891c091c091039100910091909190000000000091c09190919091009190919091909190919192cbbfd0000000000000cbc0280000000000009191922a05";
    /// The plugin's embedding batch for rid `e1`: vector `[0.25, -0.5]`, 3
    /// prompt tokens, finished `stop`, engine 1, no load tail.
    const PYTHON_EMBED: &str = "9db84261746368456d62656464696e67536c696d4f757470757491a265319192cb3fd0000000000000cbbfe00000000000009103910091a473746f7091c091c00100000000";
    /// The plugin's control reply for call 7: success, no message, engine 1.
    const PYTHON_CONTROL: &str = "95b0436f6e74726f6c5265706c79536c696d07c3c001";
    /// The plugin's terminal abort for rid `x`: its `n=2` rejection, status 400.
    const PYTHON_ABORT: &str = "dc0013b64261746368546f6b656e4944536c696d4f757470757491a178919091a561626f727491d9396e3d32206973206e6f7420736572766564206f6e207468697320776972653b20534d472066616e73206f7574206e203e203120697473656c6691c091009100910091909190000000000091cd019091909190";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The 19-element pin predates the appended prompt-logprob and reasoning
    /// columns: it decodes with them empty, and re-encoding (24 elements)
    /// round-trips to the same value.
    #[test]
    fn python_pinned_batch_decodes_and_re_encodes_identically() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SLIM)).unwrap();
        assert!(batch.reasoning_tokens.is_empty() && batch.input_token_logprobs_val.is_empty());
        assert_eq!(batch.rids, vec!["a", "b"]);
        assert_eq!(batch.output_ids, vec![vec![10], vec![20, 21]]);
        assert_eq!(
            batch.finished_matched,
            vec![None, Some(MatchedStop::TokenId(42))]
        );
        assert_eq!(batch.output_token_logprobs_val[1], vec![-0.5, -0.25]);
        assert_eq!(
            (
                batch.engine_index,
                batch.num_running,
                batch.num_waiting,
                batch.kv_used_tokens,
                batch.kv_total_tokens
            ),
            (1, 2, 3, 40, 400)
        );
        let re_encoded = encode_msgpack(&batch).unwrap();
        assert_eq!(re_encoded.len(), hex(PYTHON_SLIM).len() + 5);
        assert_eq!(
            decode_msgpack::<BatchTokenIDSlimOutput>(&re_encoded).unwrap(),
            batch
        );
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs[0].finish_reason, None);
        assert_eq!(outputs[1].finish_reason.as_deref(), Some("stop"));
        assert_eq!(outputs[1].completion_tokens, 2);
        assert!(outputs[0].output_top_logprobs_val.is_empty());
        assert_eq!(outputs[1].output_top_logprobs_val, vec![vec![-0.5, -1.0]]);
        assert_eq!(outputs[1].output_top_logprobs_idx, vec![vec![21, 22]]);
    }

    #[test]
    fn python_pinned_batch_with_appended_columns_round_trips() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SLIM24)).unwrap();
        assert_eq!(batch.reasoning_tokens, vec![0, 4]);
        assert!(batch.output_token_ids_logprobs_val.is_empty());
        assert!(batch.output_token_ids_logprobs_idx.is_empty());
        assert_eq!(
            batch.input_token_logprobs_val,
            vec![vec![], vec![None, Some(-0.7), Some(-1.1)]]
        );
        assert_eq!(batch.input_token_logprobs_idx, vec![vec![], vec![1, 2, 3]]);
        assert_eq!(
            batch.input_top_logprobs_val,
            vec![vec![], vec![vec![], vec![-0.7], vec![-1.1]]]
        );
        assert_eq!(
            batch.input_top_logprobs_idx,
            vec![vec![], vec![vec![], vec![2], vec![3]]]
        );
        assert_eq!(encode_msgpack(&batch).unwrap(), hex(PYTHON_SLIM24));
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs[1].reasoning_tokens, 4);
        assert_eq!(outputs[1].input_logprobs_idx, vec![1, 2, 3]);
        assert_eq!(outputs[1].input_top_logprobs_idx[1], vec![2]);
        assert!(outputs[0].input_logprobs_val.is_empty() && outputs[0].embedding.is_none());
        // The wire enum reads the same bytes by tag.
        assert!(matches!(
            decode_msgpack::<SglangWireOutput>(&hex(PYTHON_SLIM24)).unwrap(),
            SglangWireOutput::Tokens(_)
        ));
    }

    #[test]
    fn python_score_only_batch_preserves_selected_candidates() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SCORE26)).unwrap();
        assert_eq!(encode_msgpack(&batch).unwrap(), hex(PYTHON_SCORE26));
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].output_ids.is_empty());
        assert!(outputs[0].output_logprobs_val.is_empty());
        assert_eq!(outputs[0].completion_tokens, 0);
        assert_eq!(
            outputs[0].output_token_ids_logprobs_val,
            vec![vec![-0.25, -12.0]]
        );
        assert_eq!(outputs[0].output_token_ids_logprobs_idx, vec![vec![42, 5]]);
    }

    #[test]
    fn selected_candidate_shape_mismatch_is_refused() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SCORE26)).unwrap();
        let mut missing_column = batch.clone();
        missing_column.output_token_ids_logprobs_idx.clear();
        let mut missing_row = batch.clone();
        missing_row.output_token_ids_logprobs_idx[0].clear();
        let mut missing_id = batch;
        missing_id.output_token_ids_logprobs_idx[0][0].pop();
        for malformed in [missing_column, missing_row, missing_id] {
            let outputs = malformed.into_outputs().unwrap();
            assert_eq!(outputs[0].request_id, "score");
            assert!(outputs[0]
                .selected_logprobs_error
                .as_deref()
                .is_some_and(|error| error.contains("selected-token logprobs")));
        }
    }

    #[test]
    fn python_pinned_embedding_batch_and_control_reply_decode_by_tag() {
        let SglangWireOutput::Embeddings(batch) = decode_msgpack(&hex(PYTHON_EMBED)).unwrap()
        else {
            panic!("expected an embedding batch");
        };
        assert_eq!(batch.rids, vec!["e1"]);
        assert_eq!(batch.embeddings, vec![vec![0.25, -0.5]]);
        assert_eq!((batch.prompt_tokens[0], batch.engine_index), (3, 1));
        // Python writes the vector as float64, this side keeps float32 (the
        // proto's `repeated float`): the value round-trips, not the bytes.
        assert_eq!(
            decode_msgpack::<BatchEmbeddingSlimOutput>(&encode_msgpack(&batch).unwrap()).unwrap(),
            batch
        );
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(outputs[0].embedding, Some(vec![0.25, -0.5]));
        assert!(outputs[0].finished());
        // An aborted embedding carries no vector.
        let aborted = BatchEmbeddingSlimOutput {
            rids: vec!["e2".into()],
            embeddings: vec![vec![]],
            prompt_tokens: vec![0],
            cached_tokens: vec![0],
            finished_reasons: vec!["abort".into()],
            finished_messages: vec![Some("too long".into())],
            finished_status: vec![Some(400)],
            ..Default::default()
        };
        let outputs = aborted.into_outputs().unwrap();
        assert_eq!(outputs[0].embedding, None);
        assert_eq!(outputs[0].finish_status, Some(400));

        let SglangWireOutput::ControlReply(reply) = decode_msgpack(&hex(PYTHON_CONTROL)).unwrap()
        else {
            panic!("expected a control reply");
        };
        assert_eq!(
            reply,
            ControlReplySlim {
                call_id: 7,
                success: true,
                message: None,
                engine_index: 1,
            }
        );
        assert_eq!(encode_msgpack(&reply).unwrap(), hex(PYTHON_CONTROL));
        let unknown = decode_msgpack::<SglangWireOutput>(&encode_msgpack(&("Mystery", 1)).unwrap());
        assert!(unknown.is_err());
    }

    #[test]
    fn python_pinned_abort_carries_the_message() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_ABORT)).unwrap();
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "x");
        assert_eq!(outputs[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(
            outputs[0].finish_message.as_deref(),
            Some("n=2 is not served on this wire; SMG fans out n > 1 itself")
        );
        assert_eq!(outputs[0].finish_status, Some(400));
        assert!(outputs[0].output_ids.is_empty());
    }

    #[test]
    fn a_string_stop_round_trips_as_text() {
        let batch = BatchTokenIDSlimOutput {
            rids: vec!["a".into()],
            output_ids: vec![vec![7]],
            finished_reasons: vec!["stop".into()],
            finished_messages: vec![None],
            finished_matched: vec![Some(MatchedStop::Text("###".into()))],
            prompt_tokens: vec![1],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            ..Default::default()
        };
        let decoded: BatchTokenIDSlimOutput =
            decode_msgpack(&encode_msgpack(&batch).unwrap()).unwrap();
        assert_eq!(decoded, batch);
        let outputs = decoded.into_outputs().unwrap();
        assert_eq!(
            outputs[0].matched_stop,
            Some(MatchedStop::Text("###".into()))
        );
        assert_eq!(outputs[0].finish_status, None);
    }

    #[test]
    fn ragged_columns_are_a_protocol_error() {
        let batch = BatchTokenIDSlimOutput {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![1]],
            finished_reasons: vec![String::new(); 2],
            finished_messages: vec![None; 2],
            finished_matched: vec![None; 2],
            prompt_tokens: vec![1; 2],
            completion_tokens: vec![1; 2],
            cached_tokens: vec![0; 2],
            output_token_logprobs_val: vec![vec![]; 2],
            output_token_logprobs_idx: vec![vec![]; 2],
            ..Default::default()
        };
        assert!(batch.into_outputs().is_err());
    }

    #[test]
    fn a_shorter_tail_from_an_older_plugin_decodes_with_defaults() {
        let full: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SLIM)).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&hex(PYTHON_SLIM)).unwrap();
        let mut arr = value.as_array().unwrap().clone();
        arr.truncate(11); // through the logprob columns
        let bytes = rmp_serde::to_vec(&rmpv::Value::Array(arr)).unwrap();
        let short: BatchTokenIDSlimOutput = decode_msgpack(&bytes).unwrap();
        assert_eq!(short.rids, full.rids);
        assert_eq!((short.engine_index, short.kv_total_tokens), (0, 0));
        assert!(short.finished_status.is_empty() && short.output_top_logprobs_val.is_empty());
    }
}
