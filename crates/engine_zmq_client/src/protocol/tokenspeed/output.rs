// TokenSpeed per-step batched output — the native `BatchTokenIDOutSlim` from
// `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire it is
// a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder.

use std::collections::BTreeMap;

use serde::{
    de::{DeserializeSeed, IgnoredAny, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::decode_msgpack_seed,
    error::{Error, Result},
    protocol::{
        control::{read_control_reply_after_tag, ControlReplySlim, CONTROL_REPLY_SLIM_TAG},
        positional::{expect_tag, next_field},
        EngineOutput,
    },
};

/// The msgspec tag for [`BatchTokenIDOutSlim`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_OUT_SLIM_TAG: &str = "BatchTokenIDOutSlim";

/// The names of the two finish-detail columns an engine appends to its slim
/// batch and declares in its ready response's `positional_layouts`: the
/// HTTP-style status and the words of each request's finish reason (an abort
/// at admission carries its 400 and the reason), parallel to
/// `finished_reasons`.
pub const FINISHED_STATUS_CODES: &str = "finished_status_codes";
pub const FINISHED_MESSAGES: &str = "finished_messages";

/// The columns every engine emits, in wire order (the field indices of a
/// declared layout, the tag not counted). An optional column an engine
/// declares sits at or past their end.
pub const BASE_COLUMN_NAMES: [&str; 13] = [
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
];

/// Where a connected engine puts the optional columns of its slim batch,
/// read once off its declared layout: `None` for a column the engine does not
/// declare (an older engine), which the decoder then never reads. A column
/// is found by its name, never by a pinned position, so engine revisions that
/// append their columns in either order all decode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SlimOutputLayout {
    /// The field index of [`FINISHED_STATUS_CODES`].
    pub finish_status: Option<usize>,
    /// The field index of [`FINISHED_MESSAGES`].
    pub finish_message: Option<usize>,
}

impl SlimOutputLayout {
    /// The layout an engine declared (the `positional_layouts` of its ready
    /// response, keyed by the struct's tag). A name declared inside the base
    /// prefix is not the column this client knows and is left unread.
    pub fn declared(layouts: &BTreeMap<String, Vec<String>>) -> Self {
        let names = layouts.get(BATCH_TOKEN_ID_OUT_SLIM_TAG);
        let position = |name: &str| -> Option<usize> {
            names?
                .iter()
                .position(|field| field == name)
                .filter(|&index| index >= BASE_COLUMN_NAMES.len())
        };
        Self {
            finish_status: position(FINISHED_STATUS_CODES),
            finish_message: position(FINISHED_MESSAGES),
        }
    }
}

/// Decode one slim batch, reading the optional columns where `layout` puts
/// them and skipping every other appended column.
pub fn decode_slim_batch(bytes: &[u8], layout: &SlimOutputLayout) -> Result<BatchTokenIDOutSlim> {
    decode_msgpack_seed(bytes, layout)
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
    /// The HTTP-style status of each request's finish reason, parallel to
    /// `finished_reasons`: an abort at admission carries its 400; a finish
    /// without one is `None`. Empty from an engine that does not declare the
    /// column (an older engine).
    pub finished_status_codes: Vec<Option<u16>>,
    /// The words of each request's finish reason (an abort's message),
    /// parallel to `finished_reasons`; `None` for a finish without them.
    /// Empty from an engine that does not declare the column.
    pub finished_messages: Vec<Option<String>>,
}

impl Serialize for BatchTokenIDOutSlim {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        // The two finish-detail columns ride right behind the base columns
        // when set (the layout a test engine declares for them); a batch
        // without them is the 14-element form every engine emits.
        let detailed = !self.finished_status_codes.is_empty() || !self.finished_messages.is_empty();
        let mut tuple = serializer.serialize_tuple(if detailed { 16 } else { 14 })?;
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
        if detailed {
            tuple.serialize_element(&self.finished_status_codes)?;
            tuple.serialize_element(&self.finished_messages)?;
        }
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for BatchTokenIDOutSlim {
    /// The layout-less decode: the base columns, every appended column
    /// skipped (the view of an engine that declares none).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        DeserializeSeed::deserialize(&SlimOutputLayout::default(), deserializer)
    }
}

impl<'de> DeserializeSeed<'de> for &SlimOutputLayout {
    type Value = BatchTokenIDOutSlim;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        struct BatchVisitor<'a>(&'a SlimOutputLayout);

        impl<'de> Visitor<'de> for BatchVisitor<'_> {
            type Value = BatchTokenIDOutSlim;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged BatchTokenIDOutSlim positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, BATCH_TOKEN_ID_OUT_SLIM_TAG)?;
                read_token_batch_after_tag(&mut seq, self.0)
            }
        }

        deserializer.deserialize_seq(BatchVisitor(self))
    }
}

/// The token batch's columns after the tag; shared by the struct's own
/// decoder and the wire enum's. The columns past the base are read where
/// `layout` puts the ones this client knows and skipped otherwise (a column
/// of a newer engine, or one this client does not read).
fn read_token_batch_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
    layout: &SlimOutputLayout,
) -> std::result::Result<BatchTokenIDOutSlim, A::Error> {
    let mut batch = BatchTokenIDOutSlim {
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
        finished_status_codes: Vec::new(),
        finished_messages: Vec::new(),
    };
    let mut index = BASE_COLUMN_NAMES.len();
    loop {
        if Some(index) == layout.finish_status {
            let Some(column) = seq.next_element()? else {
                break;
            };
            batch.finished_status_codes = column;
        } else if Some(index) == layout.finish_message {
            let Some(column) = seq.next_element()? else {
                break;
            };
            batch.finished_messages = column;
        } else if seq.next_element::<IgnoredAny>()?.is_none() {
            break;
        }
        index += 1;
    }
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

/// The seed that decodes one wire output with a connection's layout
/// ([`decode_wire_output`]): a token batch reads its optional columns where
/// the layout puts them; a control reply has no layout to read.
pub struct WireOutputSeed<'a>(pub &'a SlimOutputLayout);

impl<'de> DeserializeSeed<'de> for WireOutputSeed<'_> {
    type Value = TokenSpeedWireOutput;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        struct WireVisitor<'a>(&'a SlimOutputLayout);

        impl<'de> Visitor<'de> for WireVisitor<'_> {
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
                    BATCH_TOKEN_ID_OUT_SLIM_TAG => read_token_batch_after_tag(&mut seq, self.0)
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

        deserializer.deserialize_seq(WireVisitor(self.0))
    }
}

impl<'de> Deserialize<'de> for TokenSpeedWireOutput {
    /// The layout-less decode: a token batch's base columns, every appended
    /// column skipped (the view of an engine that declares none).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        WireOutputSeed(&SlimOutputLayout::default()).deserialize(deserializer)
    }
}

/// Decode one wire output (a token batch or a control reply), reading the
/// batch's optional columns where `layout` puts them.
pub fn decode_wire_output(bytes: &[u8], layout: &SlimOutputLayout) -> Result<TokenSpeedWireOutput> {
    decode_msgpack_seed(bytes, WireOutputSeed(layout))
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
    /// The HTTP-style status the finish reason carries (an abort at admission
    /// carries its 400); `None` for a finish without one, and always `None`
    /// from an engine that does not declare the column.
    pub finish_status: Option<u16>,
    /// The words the finish reason carries (an abort's message); `None`
    /// without them, and always `None` from an engine that does not declare
    /// the column.
    pub finish_message: Option<String>,
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
        // An optional column is absent (empty) or parallel to the rest.
        let detail_ragged = |len: usize| len != 0 && len != n;
        let ragged = self.output_ids.len() != n
            || self.finished_reasons.len() != n
            || self.prompt_tokens.len() != n
            || self.completion_tokens.len() != n
            || self.cached_tokens.len() != n
            || self.output_token_logprobs_val.len() != n
            || self.output_token_logprobs_idx.len() != n
            || detail_ragged(self.finished_status_codes.len())
            || detail_ragged(self.finished_messages.len());
        if ragged {
            return Err(Error::Decode {
                target_type: "BatchTokenIDOutSlim",
                message: format!(
                    "ragged columns: rids={}, output_ids={}, finished_reasons={}, \
                     prompt_tokens={}, completion_tokens={}, cached_tokens={}, \
                     output_token_logprobs_val={}, output_token_logprobs_idx={}, \
                     finished_status_codes={}, finished_messages={}",
                    n,
                    self.output_ids.len(),
                    self.finished_reasons.len(),
                    self.prompt_tokens.len(),
                    self.completion_tokens.len(),
                    self.cached_tokens.len(),
                    self.output_token_logprobs_val.len(),
                    self.output_token_logprobs_idx.len(),
                    self.finished_status_codes.len(),
                    self.finished_messages.len(),
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
            finished_status_codes,
            finished_messages,
            // Batch-level fields, not per-request columns; the caller reads
            // them off the batch before splitting.
            engine_index: _,
            num_running: _,
            num_waiting: _,
            kv_active_pages: _,
            kv_total_pages: _,
        } = self;

        // Absent columns (an engine that declares none) read as `None` for
        // every request; present ones are parallel to `rids` (checked above).
        let mut statuses = finished_status_codes.into_iter();
        let mut messages = finished_messages.into_iter();
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
                        finish_status: statuses.next().flatten(),
                        finish_message: messages.next().flatten(),
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
            finished_status_codes: Vec::new(),
            finished_messages: Vec::new(),
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

    fn layout_of(names: &[&str]) -> SlimOutputLayout {
        let mut layouts = BTreeMap::new();
        layouts.insert(
            BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(),
            names.iter().map(|name| name.to_string()).collect(),
        );
        SlimOutputLayout::declared(&layouts)
    }

    fn names_with(appended: &[&str]) -> Vec<&'static str> {
        let mut names = BASE_COLUMN_NAMES.to_vec();
        for name in appended {
            names.push(match *name {
                FINISHED_STATUS_CODES => FINISHED_STATUS_CODES,
                FINISHED_MESSAGES => FINISHED_MESSAGES,
                "output_top_logprobs_val" => "output_top_logprobs_val",
                "output_top_logprobs_idx" => "output_top_logprobs_idx",
                other => panic!("unexpected test column {other}"),
            });
        }
        names
    }

    const REFUSAL: &str =
        "Invalid request: the input (8191 tokens) exceeds the engine's maximum input length \
         (8190 tokens)";

    fn detailed_batch() -> BatchTokenIDOutSlim {
        BatchTokenIDOutSlim {
            rids: vec!["edge".into(), "running".into()],
            output_ids: vec![vec![], vec![12]],
            finished_reasons: vec!["abort".into(), String::new()],
            prompt_tokens: vec![8191, 4],
            completion_tokens: vec![0, 1],
            cached_tokens: vec![0, 0],
            output_token_logprobs_val: vec![vec![], vec![]],
            output_token_logprobs_idx: vec![vec![], vec![]],
            finished_status_codes: vec![Some(400), None],
            finished_messages: vec![Some(REFUSAL.to_string()), None],
            ..Default::default()
        }
    }

    /// The layout names the columns by position; a column is found by its
    /// name wherever the engine put it, and an engine that declares neither
    /// (or declares a name inside the base prefix, which is not this column)
    /// leaves both unread.
    #[test]
    fn the_layout_finds_the_detail_columns_by_name() {
        assert_eq!(
            layout_of(&names_with(&[FINISHED_STATUS_CODES, FINISHED_MESSAGES])),
            SlimOutputLayout {
                finish_status: Some(13),
                finish_message: Some(14),
            }
        );
        assert_eq!(
            layout_of(&names_with(&[
                "output_top_logprobs_val",
                "output_top_logprobs_idx",
                FINISHED_STATUS_CODES,
                FINISHED_MESSAGES,
            ])),
            SlimOutputLayout {
                finish_status: Some(15),
                finish_message: Some(16),
            }
        );
        assert_eq!(
            layout_of(&names_with(&[FINISHED_MESSAGES])),
            SlimOutputLayout {
                finish_status: None,
                finish_message: Some(13),
            }
        );
        assert_eq!(layout_of(&names_with(&[])), SlimOutputLayout::default());
        assert_eq!(
            SlimOutputLayout::declared(&BTreeMap::new()),
            SlimOutputLayout::default()
        );
        let mut inside_the_prefix = BASE_COLUMN_NAMES.to_vec();
        inside_the_prefix[2] = FINISHED_STATUS_CODES;
        assert_eq!(layout_of(&inside_the_prefix), SlimOutputLayout::default());
    }

    /// A batch with the two columns encodes as 16 elements and decodes whole
    /// under the layout that declares them; the layout-less decode (an engine
    /// that declares none, or a client that reads none) skips them.
    #[test]
    fn the_detail_columns_round_trip_under_the_declared_layout_and_are_skipped_without() {
        let batch = detailed_batch();
        let bytes = encode_msgpack(&batch).unwrap();
        match decode_value(&bytes).unwrap() {
            Value::Array(array) => assert_eq!(array.len(), 16),
            other => panic!("expected array, got {other:?}"),
        }
        let layout = layout_of(&names_with(&[FINISHED_STATUS_CODES, FINISHED_MESSAGES]));
        assert_eq!(decode_slim_batch(&bytes, &layout).unwrap(), batch);
        let skipped = BatchTokenIDOutSlim {
            finished_status_codes: Vec::new(),
            finished_messages: Vec::new(),
            ..batch
        };
        assert_eq!(
            decode_msgpack::<BatchTokenIDOutSlim>(&bytes).unwrap(),
            skipped
        );
        assert_eq!(
            decode_slim_batch(&bytes, &SlimOutputLayout::default()).unwrap(),
            skipped
        );
    }

    /// The columns are read where the engine declares them, behind columns
    /// this client does not read (an engine that appended other columns
    /// first), and a batch of an engine that declares them but ends before
    /// them decodes with both empty.
    #[test]
    fn the_detail_columns_are_read_behind_columns_this_client_skips() {
        let mut array = match decode_value(&encode_msgpack(&vector_batch()).unwrap()).unwrap() {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };
        // Two appended columns of another revision, then the two of this one.
        array.push(Value::Array(vec![Value::Array(vec![Value::from(-0.1)])]));
        array.push(Value::Array(vec![Value::Array(vec![Value::from(10)])]));
        array.push(Value::Array(vec![Value::from(429)]));
        array.push(Value::Array(vec![Value::from("the queue is full")]));
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Array(array)).unwrap();
        let layout = layout_of(&names_with(&[
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            FINISHED_STATUS_CODES,
            FINISHED_MESSAGES,
        ]));
        let decoded = decode_slim_batch(&bytes, &layout).unwrap();
        assert_eq!(decoded.finished_status_codes, vec![Some(429)]);
        assert_eq!(
            decoded.finished_messages,
            vec![Some("the queue is full".to_string())]
        );
        assert_eq!(
            BatchTokenIDOutSlim {
                finished_status_codes: Vec::new(),
                finished_messages: Vec::new(),
                ..decoded
            },
            vector_batch()
        );
        // The same layout over a 14-element batch: nothing to read.
        let short = decode_slim_batch(&python_output_bytes(), &layout).unwrap();
        assert_eq!(short, vector_batch());
    }

    /// Per request, the finish details come out next to the finish reason;
    /// absent columns read as `None`, a column of the wrong length is a
    /// protocol bug.
    #[test]
    fn into_outputs_carries_the_finish_details_per_request() {
        let outputs = detailed_batch().into_outputs().unwrap();
        assert_eq!(outputs[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(outputs[0].finish_status, Some(400));
        assert_eq!(outputs[0].finish_message.as_deref(), Some(REFUSAL));
        assert_eq!(outputs[1].finish_reason, None);
        assert_eq!(outputs[1].finish_status, None);
        assert_eq!(outputs[1].finish_message, None);
        let plain = vector_batch().into_outputs().unwrap();
        assert_eq!(
            (plain[0].finish_status, plain[0].finish_message.clone()),
            (None, None)
        );
        let ragged = BatchTokenIDOutSlim {
            finished_status_codes: vec![Some(400)],
            ..detailed_batch()
        };
        let error = ragged.into_outputs().unwrap_err().to_string();
        assert!(error.contains("ragged columns"), "{error}");
        assert!(error.contains("finished_status_codes=1"), "{error}");
    }
}
