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
    error::{Error, Result},
    protocol::{
        positional::{drain_trailing, expect_tag, next_field},
        EngineOutput,
    },
};

/// The msgspec tag for [`BatchTokenIDSlimOutput`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_SLIM_OUTPUT_TAG: &str = "BatchTokenIDSlimOutput";

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
}

impl Serialize for BatchTokenIDSlimOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(19)?;
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
                let batch = BatchTokenIDSlimOutput {
                    rids: next_field(&mut seq, "rids")?,
                    output_ids: next_field(&mut seq, "output_ids")?,
                    finished_reasons: next_field(&mut seq, "finished_reasons")?,
                    finished_messages: next_field(&mut seq, "finished_messages")?,
                    finished_matched: next_field(&mut seq, "finished_matched")?,
                    prompt_tokens: next_field(&mut seq, "prompt_tokens")?,
                    completion_tokens: next_field(&mut seq, "completion_tokens")?,
                    cached_tokens: next_field(&mut seq, "cached_tokens")?,
                    output_token_logprobs_val: next_field(&mut seq, "output_token_logprobs_val")?,
                    output_token_logprobs_idx: next_field(&mut seq, "output_token_logprobs_idx")?,
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
                };
                drain_trailing(&mut seq)?;
                Ok(batch)
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
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
        Ok(self
            .rids
            .into_iter()
            .zip(self.output_ids)
            .map(|(request_id, output_ids)| {
                let reason = finished_reasons.next().unwrap_or_default();
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
                }
            })
            .collect())
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
    /// The plugin's terminal abort for rid `x`: its `n=2` rejection, status 400.
    const PYTHON_ABORT: &str = "dc0013b64261746368546f6b656e4944536c696d4f757470757491a178919091a561626f727491d9396e3d32206973206e6f7420736572766564206f6e207468697320776972653b20534d472066616e73206f7574206e203e203120697473656c6691c091009100910091909190000000000091cd019091909190";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn python_pinned_batch_decodes_and_re_encodes_identically() {
        let batch: BatchTokenIDSlimOutput = decode_msgpack(&hex(PYTHON_SLIM)).unwrap();
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
        assert_eq!(encode_msgpack(&batch).unwrap(), hex(PYTHON_SLIM));
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs[0].finish_reason, None);
        assert_eq!(outputs[1].finish_reason.as_deref(), Some("stop"));
        assert_eq!(outputs[1].completion_tokens, 2);
        assert!(outputs[0].output_top_logprobs_val.is_empty());
        assert_eq!(outputs[1].output_top_logprobs_val, vec![vec![-0.5, -1.0]]);
        assert_eq!(outputs[1].output_top_logprobs_idx, vec![vec![21, 22]]);
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
