// TokenSpeed per-step batched output — the native `BatchTokenIDOutSlim` from
// `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire it is
// a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder.

use std::collections::BTreeMap;

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    error::{Error, Result},
    protocol::{
        control::{read_control_reply_after_tag, ControlReplySlim, CONTROL_REPLY_SLIM_TAG},
        positional::{appended_column, drain_trailing, expect_tag, next_field},
        EngineOutput,
    },
};

/// The msgspec tag for [`BatchTokenIDOutSlim`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_OUT_SLIM_TAG: &str = "BatchTokenIDOutSlim";

/// The columns a scheduler appends to [`BatchTokenIDOutSlim`] when its wire
/// carries the ranked candidates per decoded token (`top_logprobs_num`), in
/// their order on the wire. A scheduler that has them declares them in its
/// READY layout of this struct ([`wire_carries_top_logprobs`]); the decoder
/// reads them from any tuple long enough. The gRPC contract of this engine
/// has no output for prompt or selected-candidate logprobs, so no column
/// carries those.
pub const TOP_LOGPROB_COLUMNS: [&str; 2] = ["output_top_logprobs_val", "output_top_logprobs_idx"];

/// Where the declared layout must carry the two columns: the decoder reads
/// them at the two slots right after `kv_total_pages` (the 14th and 15th
/// wire elements; a layout lists fields without the tag, so indices 13 and
/// 14), whatever a newer scheduler appends behind them.
pub const TOP_LOGPROB_COLUMN_INDEX: usize = 13;

/// Whether the engine's handshake declared the ranked-candidate columns on
/// its slim output where the decoder reads them
/// (`positional_layouts["BatchTokenIDOutSlim"]` names [`TOP_LOGPROB_COLUMNS`]
/// at [`TOP_LOGPROB_COLUMN_INDEX`] and the slot after it): the adapter then
/// forwards a `top_logprobs_num` above one, which it refuses on a wire
/// without them. A layout that carries the columns elsewhere reads as a wire
/// without them (and fails the connect: [`check_slim_output_layout`]); a
/// scheduler that declares no layout at all (today's) has none.
pub fn wire_carries_top_logprobs(layouts: &BTreeMap<String, Vec<String>>) -> bool {
    layouts
        .get(BATCH_TOKEN_ID_OUT_SLIM_TAG)
        .is_some_and(|fields| {
            TOP_LOGPROB_COLUMNS
                .iter()
                .enumerate()
                .all(|(offset, column)| {
                    fields
                        .get(TOP_LOGPROB_COLUMN_INDEX + offset)
                        .map(String::as_str)
                        == Some(*column)
                })
        })
}

/// A declared slim-output layout that lists something else where the decoder
/// reads the ranked-candidate columns would break every batch (the decoder
/// reads those slots whether or not the request asked for candidates): refused
/// at connect, naming the slot. A layout that stops before the slots (today's
/// 13 fields) or lists the columns there passes.
pub fn check_slim_output_layout(layouts: &BTreeMap<String, Vec<String>>) -> Result<()> {
    let Some(fields) = layouts.get(BATCH_TOKEN_ID_OUT_SLIM_TAG) else {
        return Ok(());
    };
    for (offset, column) in TOP_LOGPROB_COLUMNS.iter().enumerate() {
        let index = TOP_LOGPROB_COLUMN_INDEX + offset;
        match fields.get(index) {
            None => return Ok(()),
            Some(theirs) if theirs == column => continue,
            Some(theirs) => {
                return Err(Error::WireLayoutMismatch {
                    message: format!(
                        "`{BATCH_TOKEN_ID_OUT_SLIM_TAG}` field {index} is `{theirs}` on the \
                         engine's side where this side reads `{column}` (the ranked-candidate \
                         columns must sit right after `kv_total_pages`)"
                    ),
                });
            }
        }
    }
    Ok(())
}

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
    /// Ranked candidates per newly decoded token (`top_logprobs_num` of
    /// them) per request, parallel to the sampled-token logprob columns.
    /// Appended by the ranked-candidates wire revision
    /// ([`TOP_LOGPROB_COLUMNS`]); empty from a scheduler without it, and
    /// empty per request when not asked.
    pub output_top_logprobs_val: Vec<Vec<Vec<f64>>>,
    pub output_top_logprobs_idx: Vec<Vec<Vec<u32>>>,
}

impl Serialize for BatchTokenIDOutSlim {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        // A batch of the ranked-candidates revision carries both columns.
        let with_top_logprobs =
            !self.output_top_logprobs_val.is_empty() || !self.output_top_logprobs_idx.is_empty();
        let len = if with_top_logprobs {
            14 + TOP_LOGPROB_COLUMNS.len()
        } else {
            14
        };
        let mut tuple = serializer.serialize_tuple(len)?;
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
        if with_top_logprobs {
            tuple.serialize_element(&self.output_top_logprobs_val)?;
            tuple.serialize_element(&self.output_top_logprobs_idx)?;
        }
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
                read_token_batch_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

/// The token batch's columns after the tag; shared by the struct's own
/// decoder and the wire enum's.
fn read_token_batch_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> std::result::Result<BatchTokenIDOutSlim, A::Error> {
    let batch = BatchTokenIDOutSlim {
        rids: next_field(seq, "rids")?,
        output_ids: next_field(seq, "output_ids")?,
        finished_reasons: next_field(seq, "finished_reasons")?,
        prompt_tokens: next_field(seq, "prompt_tokens")?,
        completion_tokens: next_field(seq, "completion_tokens")?,
        cached_tokens: next_field(seq, "cached_tokens")?,
        output_token_logprobs_val: next_field(seq, "output_token_logprobs_val")?,
        output_token_logprobs_idx: next_field(seq, "output_token_logprobs_idx")?,
        // Appended by the DP wire revision: a 9-element batch from
        // an older sender means rank 0 (the only rank it can be).
        engine_index: seq.next_element::<u32>()?.unwrap_or(0),
        // Appended by the load-piggyback revision; zeros from
        // older senders decode as "no snapshot".
        num_running: seq.next_element::<u64>()?.unwrap_or(0),
        num_waiting: seq.next_element::<u64>()?.unwrap_or(0),
        kv_active_pages: seq.next_element::<u64>()?.unwrap_or(0),
        kv_total_pages: seq.next_element::<u64>()?.unwrap_or(0),
        // Appended by the ranked-candidates revision; today's 14-element
        // batch decodes with them empty.
        output_top_logprobs_val: appended_column(seq)?,
        output_top_logprobs_idx: appended_column(seq)?,
    };
    drain_trailing(seq)?;
    Ok(batch)
}

/// One output message on the wire, told apart by its tag: the token batch
/// (boxed: by far the larger and the common case), or the control reply of a
/// scheduler of the control-wire revision.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenSpeedWireOutput {
    Tokens(Box<BatchTokenIDOutSlim>),
    ControlReply(ControlReplySlim),
}

impl<'de> Deserialize<'de> for TokenSpeedWireOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct WireVisitor;

        impl<'de> Visitor<'de> for WireVisitor {
            type Value = TokenSpeedWireOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged TokenSpeed output positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let tag: String = next_field(&mut seq, "tag")?;
                match tag.as_str() {
                    BATCH_TOKEN_ID_OUT_SLIM_TAG => read_token_batch_after_tag(&mut seq)
                        .map(|batch| TokenSpeedWireOutput::Tokens(Box::new(batch))),
                    CONTROL_REPLY_SLIM_TAG => read_control_reply_after_tag(&mut seq)
                        .map(TokenSpeedWireOutput::ControlReply),
                    other => Err(serde::de::Error::custom(format!(
                        "wrong msgspec tag: expected `{BATCH_TOKEN_ID_OUT_SLIM_TAG}` or \
                         `{CONTROL_REPLY_SLIM_TAG}`, got `{other}`"
                    ))),
                }
            }
        }

        deserializer.deserialize_seq(WireVisitor)
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
    /// Ranked candidates per newly decoded token this step, when asked and
    /// the wire carries them ([`TOP_LOGPROB_COLUMNS`]); empty otherwise.
    pub output_top_logprobs_val: Vec<Vec<f64>>,
    pub output_top_logprobs_idx: Vec<Vec<u32>>,
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

        // The appended columns are optional as a whole (a scheduler without
        // the revision sends none); one that is there is a column like any.
        for (name, len) in [
            (
                "output_top_logprobs_val",
                self.output_top_logprobs_val.len(),
            ),
            (
                "output_top_logprobs_idx",
                self.output_top_logprobs_idx.len(),
            ),
        ] {
            if len > 0 && len != n {
                return Err(Error::Decode {
                    target_type: "BatchTokenIDOutSlim",
                    message: format!("column `{name}` has {len} entries for {n} rids"),
                });
            }
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
            output_top_logprobs_val,
            output_top_logprobs_idx,
        } = self;
        let mut top_val = output_top_logprobs_val.into_iter();
        let mut top_idx = output_top_logprobs_idx.into_iter();

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
                        output_top_logprobs_val: top_val.next().unwrap_or_default(),
                        output_top_logprobs_idx: top_idx.next().unwrap_or_default(),
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
    use crate::{
        codec::{decode_msgpack, decode_value, encode_msgpack},
        protocol::positional::vector_bytes,
    };

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

    /// The same batch from a scheduler that carries the ranked candidates:
    /// the first token's three candidates (-0.5, -1.0, -2.0 for ids 10, 12,
    /// 13) and the second's one (-0.25 for id 11) in the two appended
    /// columns, captured from the Python msgspec encoder.
    const PYTHON_OUTPUT_VECTOR_WITH_TOP_LOGPROBS: &str =
        "dc0010b34261746368546f6b656e49444f7574536c696d91a57665632d3191920a0b91a66c65\
         6e6774689103910291019192cbbfe0000000000000cbbfd000000000000091920a0b01020564\
         cd0190919293cbbfe0000000000000cbbff0000000000000cbc00000000000000091cbbfd000\
         00000000009192930a0c0d910b";


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
            ..Default::default()
        }
    }

    /// The ranked-candidates revision appends two columns after the load
    /// tail: a batch carrying them encodes 16 elements and splits per
    /// request into the ranked rows (three candidates on the first token,
    /// one on the second); today's 14-element batch decodes with them
    /// empty and re-encodes byte for byte; a ragged appended column is a
    /// protocol bug; and the READY layout tells the two schedulers apart.
    #[test]
    fn top_logprob_columns_are_appended_optional_and_declared() {
        let batch = BatchTokenIDOutSlim {
            output_top_logprobs_val: vec![vec![vec![-0.5, -1.0, -2.0], vec![-0.25]]],
            output_top_logprobs_idx: vec![vec![vec![10, 12, 13], vec![11]]],
            ..vector_batch()
        };
        let encoded = encode_msgpack(&batch).unwrap();
        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array.len(), 14 + TOP_LOGPROB_COLUMNS.len());
        let decoded: BatchTokenIDOutSlim = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded, batch);
        // Byte for byte the 16-element batch the scheduler's own struct
        // emits for these rows (the `dc 00 10` header, the two columns last).
        assert_eq!(
            encoded,
            vector_bytes(PYTHON_OUTPUT_VECTOR_WITH_TOP_LOGPROBS)
        );
        let outputs = decoded.into_outputs().unwrap();
        assert_eq!(
            outputs[0].output_top_logprobs_val,
            vec![vec![-0.5, -1.0, -2.0], vec![-0.25]]
        );
        assert_eq!(
            outputs[0].output_top_logprobs_idx,
            vec![vec![10, 12, 13], vec![11]]
        );

        let today: BatchTokenIDOutSlim = decode_msgpack(&python_output_bytes()).unwrap();
        assert!(today.output_top_logprobs_val.is_empty());
        assert_eq!(encode_msgpack(&today).unwrap(), python_output_bytes());
        assert!(today.into_outputs().unwrap()[0]
            .output_top_logprobs_val
            .is_empty());

        let mut ragged = batch;
        ragged.output_top_logprobs_idx.push(vec![vec![9]]);
        let error = ragged.into_outputs().unwrap_err().to_string();
        assert!(error.contains("output_top_logprobs_idx"), "{error}");

        let mut layouts = BTreeMap::new();
        assert!(!wire_carries_top_logprobs(&layouts));
        let mut fields: Vec<String> = [
            "rids",
            "output_ids",
            "finished_reasons",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "engine_index",
            "num_running",
            "num_waiting",
            "kv_active_pages",
            "kv_total_pages",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        layouts.insert(BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(), fields.clone());
        assert!(!wire_carries_top_logprobs(&layouts), "today's layout");
        assert!(check_slim_output_layout(&layouts).is_ok());
        // The columns elsewhere (something else appended first) read as a wire
        // without them, and the connect refuses the layout by its slot.
        let mut elsewhere = fields.clone();
        elsewhere.push("something_else".to_string());
        elsewhere.extend(TOP_LOGPROB_COLUMNS.iter().map(ToString::to_string));
        layouts.insert(BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(), elsewhere);
        assert!(
            !wire_carries_top_logprobs(&layouts),
            "the columns elsewhere"
        );
        let error = check_slim_output_layout(&layouts).unwrap_err().to_string();
        assert!(error.contains("field 13 is `something_else`"), "{error}");
        fields.extend(TOP_LOGPROB_COLUMNS.iter().map(ToString::to_string));
        layouts.insert(BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(), fields.clone());
        assert!(wire_carries_top_logprobs(&layouts));
        assert!(check_slim_output_layout(&layouts).is_ok());
        // Columns a newer scheduler appends behind the two pass as before.
        fields.push("a_newer_column".to_string());
        layouts.insert(BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(), fields);
        assert!(wire_carries_top_logprobs(&layouts));
        assert!(check_slim_output_layout(&layouts).is_ok());
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
        // TokenSpeed appends new fields at the end; the decoder skips what it
        // does not model (past the ranked-candidate columns, nil here).
        let mut with_extra = match decode_value(&python_output_bytes()).unwrap() {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };
        with_extra.extend([Value::Nil, Value::Nil]);
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
