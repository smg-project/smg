// TokenSpeed per-step batched output — the native `BatchTokenIDOutSlim` from
// `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire it is
// a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder.

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    error::{Error, Result},
    protocol::{
        tokenspeed::{drain_trailing, expect_tag, next_field},
        EngineOutput,
    },
};

/// The msgspec tag for [`BatchTokenIDOutSlim`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_OUT_SLIM_TAG: &str = "BatchTokenIDOutSlim";

/// A batch of per-request token outputs from one engine step. Mirrors Python
/// `BatchTokenIDOutSlim`: the per-request slice of an engine step for a
/// frontend that detokenizes itself.
///
/// Every field is a column indexed in parallel by request: `rids[i]` owns
/// `output_ids[i]`, `finished_reasons[i]`, `prompt_tokens[i]`, etc. The two
/// logprob columns are always present (length == `rids.len()`); the inner `Vec`
/// is empty for a request that did not ask for logprobs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchTokenIDOutSlim {
    /// Request ids, one per column.
    pub rids: Vec<String>,
    /// Newly generated token ids per request this step.
    pub output_ids: Vec<Vec<u32>>,
    /// Finish reason per request (`"stop"`, `"length"`, `"abort"`); an empty
    /// string means "not finished".
    pub finished_reasons: Vec<String>,
    /// Prompt token count per request.
    pub prompt_tokens: Vec<u32>,
    /// Completion token count so far, per request.
    pub completion_tokens: Vec<u32>,
    /// Prefix-cache-hit token count per request.
    pub cached_tokens: Vec<u32>,
    /// Sampled-token logprob value per newly decoded token, per request. `f64`
    /// matches the Python float encoding. Empty inner `Vec` when logprobs were
    /// not requested.
    pub output_token_logprobs_val: Vec<Vec<f64>>,
    /// Token id each logprob in `output_token_logprobs_val` belongs to,
    /// parallel to it per request. Empty inner `Vec` when logprobs were not
    /// requested.
    pub output_token_logprobs_idx: Vec<Vec<u32>>,
    /// Producing DP rank's engine index. The output PULL socket carries no
    /// routing identity, so under DP the batch itself names its rank. Appended
    /// tail field: an older (pre-DP) sender emits 9 elements and this defaults
    /// to `0`, which is also the sole rank of a single-engine worker.
    pub engine_index: u32,
    /// Piggybacked scheduler-load snapshot, sampled by the producing rank at
    /// send time (the msgpack wire drops control replies, so the output batch
    /// is the only in-band load channel). Appended tail fields: all default
    /// to `0` from older senders, and `kv_total_pages == 0` means "no
    /// snapshot" — the decoder then reports no load at all rather than a
    /// fabricated zero.
    pub num_running: u64,
    /// Scheduler waiting-queue depth at send time.
    pub num_waiting: u64,
    /// KV pages held by running requests (the usage ratio's numerator).
    pub kv_active_pages: u64,
    /// Usable KV pages on the rank (the usage ratio's denominator).
    pub kv_total_pages: u64,
}

impl Serialize for BatchTokenIDOutSlim {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(14)?;
        tuple.serialize_element(BATCH_TOKEN_ID_OUT_SLIM_TAG)?;
        tuple.serialize_element(&self.rids)?;
        tuple.serialize_element(&self.output_ids)?;
        tuple.serialize_element(&self.finished_reasons)?;
        tuple.serialize_element(&self.prompt_tokens)?;
        tuple.serialize_element(&self.completion_tokens)?;
        tuple.serialize_element(&self.cached_tokens)?;
        tuple.serialize_element(&self.output_token_logprobs_val)?;
        tuple.serialize_element(&self.output_token_logprobs_idx)?;
        tuple.serialize_element(&self.engine_index)?;
        tuple.serialize_element(&self.num_running)?;
        tuple.serialize_element(&self.num_waiting)?;
        tuple.serialize_element(&self.kv_active_pages)?;
        tuple.serialize_element(&self.kv_total_pages)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for BatchTokenIDOutSlim {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct BatchVisitor;

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = BatchTokenIDOutSlim;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged BatchTokenIDOutSlim positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, BATCH_TOKEN_ID_OUT_SLIM_TAG)?;
                let batch = BatchTokenIDOutSlim {
                    rids: next_field(&mut seq, "rids")?,
                    output_ids: next_field(&mut seq, "output_ids")?,
                    finished_reasons: next_field(&mut seq, "finished_reasons")?,
                    prompt_tokens: next_field(&mut seq, "prompt_tokens")?,
                    completion_tokens: next_field(&mut seq, "completion_tokens")?,
                    cached_tokens: next_field(&mut seq, "cached_tokens")?,
                    output_token_logprobs_val: next_field(&mut seq, "output_token_logprobs_val")?,
                    output_token_logprobs_idx: next_field(&mut seq, "output_token_logprobs_idx")?,
                    // Appended by the DP wire revision: a 9-element batch from
                    // an older sender means rank 0 (the only rank it can be).
                    engine_index: seq.next_element::<u32>()?.unwrap_or(0),
                    // Appended by the load-piggyback revision; zeros from
                    // older senders decode as "no snapshot".
                    num_running: seq.next_element::<u64>()?.unwrap_or(0),
                    num_waiting: seq.next_element::<u64>()?.unwrap_or(0),
                    kv_active_pages: seq.next_element::<u64>()?.unwrap_or(0),
                    kv_total_pages: seq.next_element::<u64>()?.unwrap_or(0),
                };
                drain_trailing(&mut seq)?;
                Ok(batch)
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

/// One request's slice of a [`BatchTokenIDOutSlim`], in the engine-neutral
/// shape the connector routes to per-request streams.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenSpeedOutput {
    /// The request id this output belongs to.
    pub request_id: String,
    /// Newly generated token ids this step.
    pub output_ids: Vec<u32>,
    /// Finish reason; `None` while the request is still generating.
    pub finish_reason: Option<String>,
    /// Prompt token count.
    pub prompt_tokens: u32,
    /// Completion token count so far.
    pub completion_tokens: u32,
    /// Prefix-cache-hit token count.
    pub cached_tokens: u32,
    /// Sampled-token logprob value per decoded token this step. Empty when
    /// logprobs were not requested.
    pub output_logprobs_val: Vec<f64>,
    /// Token id each logprob in `output_logprobs_val` belongs to. Empty when
    /// logprobs were not requested.
    pub output_logprobs_idx: Vec<u32>,
}

impl EngineOutput for TokenSpeedOutput {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

impl BatchTokenIDOutSlim {
    /// Split the parallel columns into one [`TokenSpeedOutput`] per request.
    /// Errors if the columns are ragged (a length mismatch is a protocol bug).
    pub fn into_outputs(self) -> Result<Vec<TokenSpeedOutput>> {
        let n = self.rids.len();
        let ragged = self.output_ids.len() != n
            || self.finished_reasons.len() != n
            || self.prompt_tokens.len() != n
            || self.completion_tokens.len() != n
            || self.cached_tokens.len() != n
            || self.output_token_logprobs_val.len() != n
            || self.output_token_logprobs_idx.len() != n;
        if ragged {
            return Err(Error::Decode {
                target_type: "BatchTokenIDOutSlim",
                message: format!(
                    "ragged columns: rids={}, output_ids={}, finished_reasons={}, \
                     prompt_tokens={}, completion_tokens={}, cached_tokens={}, \
                     output_token_logprobs_val={}, output_token_logprobs_idx={}",
                    n,
                    self.output_ids.len(),
                    self.finished_reasons.len(),
                    self.prompt_tokens.len(),
                    self.completion_tokens.len(),
                    self.cached_tokens.len(),
                    self.output_token_logprobs_val.len(),
                    self.output_token_logprobs_idx.len(),
                ),
            });
        }

        let BatchTokenIDOutSlim {
            rids,
            output_ids,
            finished_reasons,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            // Batch-level fields, not per-request columns; the caller reads
            // them off the batch before splitting.
            engine_index: _,
            num_running: _,
            num_waiting: _,
            kv_active_pages: _,
            kv_total_pages: _,
        } = self;

        Ok(rids
            .into_iter()
            .zip(output_ids)
            .zip(finished_reasons)
            .zip(prompt_tokens)
            .zip(completion_tokens)
            .zip(cached_tokens)
            .zip(output_token_logprobs_val)
            .zip(output_token_logprobs_idx)
            .map(
                |(
                    ((((((rid, ids), reason), prompt), completion), cached), logprobs_val),
                    logprobs_idx,
                )| {
                    TokenSpeedOutput {
                        request_id: rid,
                        output_ids: ids,
                        // An empty finish-reason string means "still generating".
                        finish_reason: (!reason.is_empty()).then_some(reason),
                        prompt_tokens: prompt,
                        completion_tokens: completion,
                        cached_tokens: cached,
                        output_logprobs_val: logprobs_val,
                        output_logprobs_idx: logprobs_idx,
                    }
                },
            )
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack};

    /// A slim output batch captured from the Python msgspec encoder: rids
    /// ["vec-1"], output_ids [[10, 11]], finished_reasons ["length"], prompt 3
    /// / completion 2 / cached 1, logprobs [[-0.5, -0.25]] over tokens
    /// [[10, 11]], engine_index 1, load snapshot (2 running, 5 waiting,
    /// 100/400 KV pages).
    const PYTHON_OUTPUT_VECTOR: &str =
        "9eb34261746368546f6b656e49444f7574536c696d91a57665632d3191920a0b91a66c656e\
         6774689103910291019192cbbfe0000000000000cbbfd000000000000091920a0b01020564\
         cd0190";

    /// The same batch as encoded by the engine_index-era sender: 10 elements,
    /// no load snapshot.
    const PYTHON_OUTPUT_VECTOR_PRE_LOAD: &str =
        "9ab34261746368546f6b656e49444f7574536c696d91a57665632d3191920a0b91a66c656e\
         6774689103910291019192cbbfe0000000000000cbbfd000000000000091920a0b01";

    /// The same batch as encoded before the DP wire revision: 9 elements, no
    /// engine_index tail field.
    const PYTHON_OUTPUT_VECTOR_PRE_DP: &str =
        "99b34261746368546f6b656e49444f7574536c696d91a57665632d3191920a0b91a66c656e\
         6774689103910291019192cbbfe0000000000000cbbfd000000000000091920a0b";

    fn vector_bytes(hex_vector: &str) -> Vec<u8> {
        let hex: String = hex_vector.chars().filter(|c| !c.is_whitespace()).collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn python_output_bytes() -> Vec<u8> {
        vector_bytes(PYTHON_OUTPUT_VECTOR)
    }

    fn vector_batch() -> BatchTokenIDOutSlim {
        BatchTokenIDOutSlim {
            rids: vec!["vec-1".into()],
            output_ids: vec![vec![10, 11]],
            finished_reasons: vec!["length".into()],
            prompt_tokens: vec![3],
            completion_tokens: vec![2],
            cached_tokens: vec![1],
            output_token_logprobs_val: vec![vec![-0.5, -0.25]],
            output_token_logprobs_idx: vec![vec![10, 11]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 5,
            kv_active_pages: 100,
            kv_total_pages: 400,
        }
    }

    /// The pinned cross-language vector — the exact bytes the engine sends —
    /// decodes into the full 14-element (tag + 8 columns + rank + load) batch.
    #[test]
    fn python_output_vector_decodes() {
        let decoded: BatchTokenIDOutSlim = decode_msgpack(&python_output_bytes()).unwrap();
        assert_eq!(decoded, vector_batch());

        let outputs = decoded.into_outputs().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "vec-1");
        assert_eq!(outputs[0].output_ids, vec![10, 11]);
        assert_eq!(outputs[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(outputs[0].prompt_tokens, 3);
        assert_eq!(outputs[0].completion_tokens, 2);
        assert_eq!(outputs[0].cached_tokens, 1);
        assert_eq!(outputs[0].output_logprobs_val, vec![-0.5, -0.25]);
        assert_eq!(outputs[0].output_logprobs_idx, vec![10, 11]);
    }

    /// The Rust encoder mirrors the Python encoding byte-for-byte, so the mock
    /// engine emits exactly what a real engine would.
    #[test]
    fn encoder_matches_python_bytes() {
        let encoded = encode_msgpack(&vector_batch()).unwrap();
        assert_eq!(encoded, python_output_bytes());
    }

    #[test]
    fn batch_output_serializes_as_tagged_fourteen_element_array() {
        let encoded = encode_msgpack(&vector_batch()).unwrap();
        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array.len(), 14);
        assert_eq!(array[0], Value::from(BATCH_TOKEN_ID_OUT_SLIM_TAG));
        assert_eq!(array[1], Value::Array(vec![Value::from("vec-1")])); // rids
        assert_eq!(array[3], Value::Array(vec![Value::from("length")])); // finished_reasons
        assert_eq!(array[6], Value::Array(vec![Value::from(1)])); // cached_tokens
        assert_eq!(array[9], Value::from(1)); // engine_index
                                              // Load snapshot tail: running, waiting, active pages, total pages.
        assert_eq!(array[10], Value::from(2));
        assert_eq!(array[11], Value::from(5));
        assert_eq!(array[12], Value::from(100));
        assert_eq!(array[13], Value::from(400));
    }

    /// A pre-DP sender emits 9 elements; the missing tail decodes as rank 0
    /// (the only rank a single-engine worker can be) with no load snapshot.
    #[test]
    fn pre_dp_nine_element_batch_decodes_as_rank_zero() {
        let decoded: BatchTokenIDOutSlim =
            decode_msgpack(&vector_bytes(PYTHON_OUTPUT_VECTOR_PRE_DP)).unwrap();
        assert_eq!(
            decoded,
            BatchTokenIDOutSlim {
                engine_index: 0,
                num_running: 0,
                num_waiting: 0,
                kv_active_pages: 0,
                kv_total_pages: 0,
                ..vector_batch()
            }
        );
    }

    /// An engine_index-era sender emits 10 elements; the missing load tail
    /// decodes as the zero "no snapshot" defaults.
    #[test]
    fn pre_load_ten_element_batch_decodes_with_zero_snapshot() {
        let decoded: BatchTokenIDOutSlim =
            decode_msgpack(&vector_bytes(PYTHON_OUTPUT_VECTOR_PRE_LOAD)).unwrap();
        assert_eq!(
            decoded,
            BatchTokenIDOutSlim {
                num_running: 0,
                num_waiting: 0,
                kv_active_pages: 0,
                kv_total_pages: 0,
                ..vector_batch()
            }
        );
    }

    #[test]
    fn decode_rejects_wrong_tag() {
        // Re-tag the batch as some other message type: decode must fail loudly.
        let mut bytes = python_output_bytes();
        bytes[2] = b'X'; // "BatchTokenIDOutSlim" -> "XatchTokenIDOutSlim"
        let error = decode_msgpack::<BatchTokenIDOutSlim>(&bytes).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("wrong msgspec tag"), "{rendered}");
    }

    #[test]
    fn decode_tolerates_appended_trailing_columns() {
        // TokenSpeed appends new fields at the end; the decoder skips them.
        let mut with_extra = match decode_value(&python_output_bytes()).unwrap() {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };
        with_extra.push(Value::Array(vec![Value::from(99)]));
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Array(with_extra)).unwrap();
        let decoded: BatchTokenIDOutSlim = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded, vector_batch());
    }

    #[test]
    fn into_outputs_splits_columns_and_maps_finish() {
        let batch = BatchTokenIDOutSlim {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20, 21]],
            finished_reasons: vec![String::new(), "stop".into()],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 2],
            cached_tokens: vec![0, 1],
            output_token_logprobs_val: vec![vec![-0.5], vec![-1.0, -2.0]],
            output_token_logprobs_idx: vec![vec![10], vec![20, 21]],
            ..Default::default()
        };
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].request_id, "a");
        assert_eq!(outputs[0].output_ids, vec![10]);
        assert!(!outputs[0].finished()); // empty reason -> still generating
        assert_eq!(outputs[0].output_logprobs_val, vec![-0.5]);
        assert_eq!(outputs[0].output_logprobs_idx, vec![10]);
        assert_eq!(outputs[1].request_id, "b");
        assert_eq!(outputs[1].finish_reason.as_deref(), Some("stop"));
        assert!(outputs[1].finished());
        assert_eq!(outputs[1].completion_tokens, 2);
        assert_eq!(outputs[1].output_logprobs_val, vec![-1.0, -2.0]);
        assert_eq!(outputs[1].output_logprobs_idx, vec![20, 21]);
    }

    #[test]
    fn into_outputs_leaves_logprobs_empty_when_not_requested() {
        let batch = BatchTokenIDOutSlim {
            rids: vec!["a".into()],
            output_ids: vec![vec![10]],
            finished_reasons: vec![String::new()],
            prompt_tokens: vec![3],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            ..Default::default()
        };
        let outputs = batch.into_outputs().unwrap();
        assert!(outputs[0].output_logprobs_val.is_empty());
        assert!(outputs[0].output_logprobs_idx.is_empty());
    }

    #[test]
    fn into_outputs_rejects_ragged_columns() {
        let batch = BatchTokenIDOutSlim {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10]],
            finished_reasons: vec![String::new(), String::new()],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 2],
            cached_tokens: vec![0, 1],
            output_token_logprobs_val: vec![vec![], vec![]],
            output_token_logprobs_idx: vec![vec![], vec![]],
            ..Default::default()
        };
        assert!(batch.into_outputs().is_err());
    }

    #[test]
    fn into_outputs_rejects_ragged_logprob_columns() {
        let batch = BatchTokenIDOutSlim {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20]],
            finished_reasons: vec![String::new(), String::new()],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 1],
            cached_tokens: vec![0, 0],
            // Only one logprob column entry for two requests.
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![], vec![]],
            ..Default::default()
        };
        assert!(batch.into_outputs().is_err());
    }
}
