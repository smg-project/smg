// SGLang's tokenized generate request — the scheduler's own
// `TokenizedGenerateReqInput` from `io_struct.py`, a tagged
// `msgspec.Struct(array_like=True)`: on the wire a positional msgpack array
// with the class-name tag string as element 0. **Field order is the wire
// contract** — do not reorder.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{
    de::{IgnoredAny, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::tensor::{TensorFields, WireArrayData},
    protocol::{
        positional::{drain_trailing, expect_tag, next_field},
        sglang::{multimodal::SglangWireMmInputs, sampling::SamplingParams},
    },
};

/// The msgspec tag for [`TokenizedGenerateReqInput`] (element 0 on the wire).
pub const TOKENIZED_GENERATE_REQ_INPUT_TAG: &str = "TokenizedGenerateReqInput";
/// The msgspec tag for [`TokenizedEmbeddingReqInput`] (element 0 on the wire).
pub const TOKENIZED_EMBEDDING_REQ_INPUT_TAG: &str = "TokenizedEmbeddingReqInput";

/// The msgspec tag for [`EncoderDispatchErrorReq`] (element 0 on the wire).
pub const ENCODER_DISPATCH_ERROR_REQ_TAG: &str = "EncoderDispatchErrorReq";

/// The scheduler's field order of [`TokenizedGenerateReqInput`] this encoder
/// relies on, through `encoder_urls` (the tag, element 0, is not a field):
/// the slots the extended forms fill, which the handshake check compares
/// with the order the engine declares. The form through `require_reasoning`
/// (34 elements) serves every request but one whose media the encoders
/// serve; that one reaches `encoder_urls` (42 elements), the slots between
/// at the scheduler's defaults.
pub const GENERATE_WIRE_FIELDS: [&str; 41] = [
    "rid",
    "http_worker_ipc",
    "input_text",
    "input_ids",
    "input_embeds",
    "mm_inputs",
    "token_type_ids",
    "sampling_params",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "stream",
    "return_sampling_mask",
    "return_flat_raw_top_logprobs",
    "return_hidden_states",
    "return_routed_experts",
    "routed_experts_start_len",
    "return_indexer_topk",
    "session_id",
    "session_params",
    "lora_id",
    "custom_logit_processor",
    "positional_embed_overrides",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "bootstrap_pair_key",
    "decode_tp_size",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
    "routing_key",
    "require_reasoning",
    "priority",
    "extra_key",
    "no_logs",
    "return_bytes",
    "return_entropy",
    "need_wait_for_mm_inputs",
    "num_items_assigned",
    "encoder_urls",
];

/// The scheduler's field order of [`EncoderDispatchErrorReq`] (the tag not
/// counted): its base `rid` / `http_worker_ipc`, then the error.
pub const ENCODER_DISPATCH_ERROR_WIRE_FIELDS: [&str; 4] =
    ["rid", "http_worker_ipc", "error_msg", "error_code"];

/// The same for [`TokenizedEmbeddingReqInput`], through `dimensions`.
pub const EMBEDDING_WIRE_FIELDS: [&str; 12] = [
    "rid",
    "http_worker_ipc",
    "input_text",
    "input_ids",
    "mm_inputs",
    "token_type_ids",
    "sampling_params",
    "lora_id",
    "positional_embed_overrides",
    "routed_dp_rank",
    "priority",
    "dimensions",
];

/// Request types: a single raw byte sent as its own ZMQ frame ahead of the
/// msgpack payload, so the plugin dispatches without decoding first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SglangRequestType {
    Add = 0,
    Abort = 1,
    /// A control call for the plugin: `(call_id, method, args)`, answered by
    /// a `ControlReplySlim` under the same call id.
    Control = 2,
}

impl SglangRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        match frame {
            [0] => Some(Self::Add),
            [1] => Some(Self::Abort),
            [2] => Some(Self::Control),
            _ => None,
        }
    }

    /// Encode as the single-byte frame used on the engine input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
            Self::Control => b"\x02",
        })
    }
}

/// The request SMG sends an SGLang scheduler.
///
/// Models the Python class's prefix through `require_reasoning`: the base
/// `rid` / `http_worker_ipc`, the required `input_text`, `input_ids`,
/// `input_embeds`, `mm_inputs` (the scheduler's own processor output, see
/// [`SglangWireMmInputs`]), `token_type_ids`, `sampling_params`,
/// `return_logprob`, `logprob_start_len`, `top_logprobs_num`,
/// `token_ids_logprob`, `stream`, then the 19 defaulted fields up to
/// `require_reasoning`, of which this side fills `return_hidden_states`, the
/// LoRA id, the logit processor, the prefill/decode bootstrap triple and the
/// routed data-parallel rank (the session and routing-key slots stay at the
/// scheduler's defaults), so that flag lands in its slot. The encoder emits
/// the shortest valid prefix: 14 elements (tag + 13 fields) through
/// `stream`, or 34 when a later field is set; every field after the prefix
/// keeps its `msgspec` default. Token ids go as a plain integer array; the
/// plugin widens the scheduler's `array('q')` decode hook to accept it.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// Original prompt text. `None`: SMG sends token ids only.
    pub input_text: Option<String>,
    /// Pre-tokenized prompt token ids (SMG tokenizes upstream).
    pub input_ids: Vec<u32>,
    /// The request's multimodal payload, the scheduler's processor output:
    /// the preprocessed tensors and their placeholder offsets, as the Python
    /// servicer hands them over. `None` for a text request.
    pub mm_inputs: Option<SglangWireMmInputs>,
    /// Sampling parameters (nested untagged positional array).
    pub sampling_params: SamplingParams,
    /// Whether to return the sampled token's logprob per output token.
    pub return_logprob: bool,
    /// Prompt-logprob start offset; `-1` is the API default (none).
    pub logprob_start_len: i32,
    /// Output top-k logprob count; `0` means only the sampled token's.
    pub top_logprobs_num: u32,
    /// Token ids to report logprobs for; `None` when not requested.
    pub token_ids_logprob: Option<Vec<u32>>,
    /// Whether to stream outputs incrementally.
    pub stream: bool,
    /// Whether the scheduler captures the final hidden layer's vector of every
    /// token for this request (SGLang's `return_hidden_states=True`; the
    /// engine's `"last"` mode has no slot on the gRPC contract). The slim
    /// output carries the rows back.
    pub return_hidden_states: bool,
    /// LoRA adapter id, forwarded as the Python servicer forwards it (the
    /// scheduler resolves it against the adapters it loaded).
    pub lora_id: Option<String>,
    /// Serialized custom logit processor (`--enable-custom-logit-processor`).
    pub custom_logit_processor: Option<String>,
    /// A disaggregated request's rendezvous, in the scheduler's own slots:
    /// the prefill side's bootstrap server (`host`, `port`) and the room both
    /// sides meet in. A prefill scheduler registers the request's KV there
    /// under the room, a decode scheduler asks there for it. `None` leaves
    /// the scheduler's defaults (a request outside prefill/decode
    /// disaggregation, or a side the frontend hands no server for).
    pub bootstrap_host: Option<String>,
    pub bootstrap_port: Option<i32>,
    pub bootstrap_room: Option<i64>,
    /// The data-parallel rank the request is routed to, in the scheduler's
    /// own `routed_dp_rank` slot: what SGLang's frontend sets from the API's
    /// pin and its data-parallel controller dispatches on. On this wire the
    /// connector is that dispatcher: the request goes out on the socket of
    /// the engine whose index is this rank (each scheduler rank dials in with
    /// its rank as its identity), and the slot carries the same value so the
    /// scheduler's record of the request matches its frontend's. `None`
    /// leaves the placement to the least-loaded selection.
    pub routed_dp_rank: Option<u32>,
    /// Hybrid-reasoning request: the scheduler tracks the reasoning phase and
    /// accounts its tokens separately.
    pub require_reasoning: bool,
    /// A request whose media the encoders of an encode/prefill-disaggregated
    /// deployment serve, in the scheduler's own slots as the engine's
    /// frontend sets them when it dispatches the media: the scheduler is to
    /// wait for the embeddings (`need_wait_for_mm_inputs`), how many items
    /// each encoder was given per modality (`num_items_assigned`: the
    /// modality's enum value to the count per encoder, in the encoder list's
    /// order) and the encoder list that assignment indexes (`encoder_urls`,
    /// the frontend's snapshot at dispatch time). `None` on every other
    /// request: the scheduler's defaults.
    pub need_wait_for_mm_inputs: Option<bool>,
    pub num_items_assigned: Option<BTreeMap<u32, Vec<u32>>>,
    pub encoder_urls: Option<Vec<String>>,
    /// The caller's W3C trace context (`traceparent`, `tracestate`) for the
    /// engine's tracer. Not a slot of the scheduler's struct (its own
    /// `time_stats` slot is a pickled object no frontend outside Python can
    /// fill): it rides in the request's metadata frame after the positional
    /// array, see [`RequestMeta`]. Never serialized into the array.
    pub trace_headers: Option<BTreeMap<String, String>>,
}

/// What rides beside an ADD's positional request, as one more msgpack frame
/// (a map, every key optional): the request's metadata the scheduler's own
/// struct has no slot for. The plugin in the scheduler reads it at the END
/// of the message, so it is the last frame whenever anything follows the
/// payload: alone when there is something to carry and no tensor frame,
/// written empty when tensor frames come first (the plugin counts those
/// from the message length, this frame excluded); absent only from a
/// message of the payload alone. A scheduler without the plugin's reader
/// ignores it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RequestMeta {
    /// The caller's W3C trace context (`traceparent`, `tracestate`): the
    /// plugin starts the request's trace from it, as SGLang's own frontend
    /// starts it from an HTTP request's headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl RequestMeta {
    /// The metadata frame for `request`, or `None` when there is nothing to
    /// carry (the encoder still writes it empty behind tensor frames: see
    /// the type's doc).
    pub fn for_request(request: &TokenizedGenerateReqInput) -> Option<Self> {
        request.trace_headers.as_ref().map(|headers| Self {
            trace_headers: Some(headers.clone()),
        })
    }
}

/// The request's tensors: its multimodal payload's, in wire order.
impl TensorFields for TokenizedGenerateReqInput {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        self.mm_inputs.for_each_tensor(visit);
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        self.mm_inputs.for_each_tensor_mut(visit);
    }
}

impl TokenizedGenerateReqInput {
    /// Whether any field past `stream` is set, so the encoder must emit the
    /// defaulted slots up to the last set one.
    fn needs_extended_form(&self) -> bool {
        self.require_reasoning
            || self.return_hidden_states
            || self.lora_id.is_some()
            || self.custom_logit_processor.is_some()
            || self.bootstrap_host.is_some()
            || self.bootstrap_port.is_some()
            || self.bootstrap_room.is_some()
            || self.routed_dp_rank.is_some()
            || self.needs_encode_slots()
    }

    /// Whether the encode dispatch set its slots (38-40), so the encoder
    /// must emit the defaulted slots 33-37 before them.
    fn needs_encode_slots(&self) -> bool {
        self.need_wait_for_mm_inputs.is_some()
            || self.num_items_assigned.is_some()
            || self.encoder_urls.is_some()
    }
}

impl Default for TokenizedGenerateReqInput {
    fn default() -> Self {
        Self {
            rid: String::new(),
            input_text: None,
            input_ids: Vec::new(),
            mm_inputs: None,
            sampling_params: SamplingParams::default(),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
            return_hidden_states: false,
            lora_id: None,
            custom_logit_processor: None,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            routed_dp_rank: None,
            require_reasoning: false,
            need_wait_for_mm_inputs: None,
            num_items_assigned: None,
            encoder_urls: None,
            trace_headers: None,
        }
    }
}

/// The tag plus the fields through `require_reasoning`: the extended form
/// every request but an encode-dispatched one stops at.
const THROUGH_REQUIRE_REASONING: usize = 34;

impl Serialize for TokenizedGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        // The shortest valid prefix: through `stream` unless a later field
        // (`return_hidden_states`, `lora_id`, `custom_logit_processor`, the bootstrap triple,
        // `routed_dp_rank`, `require_reasoning`) has to reach its slot past the
        // defaulted ones; through `encoder_urls` when the encode dispatch set
        // its slots.
        let len = if self.needs_encode_slots() {
            GENERATE_WIRE_FIELDS.len() + 1
        } else if self.needs_extended_form() {
            THROUGH_REQUIRE_REASONING
        } else {
            14
        };
        let mut tuple = serializer.serialize_tuple(len)?;
        tuple.serialize_element(TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&NIL)?; // http_worker_ipc
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
        tuple.serialize_element(&NIL)?; // input_embeds
        tuple.serialize_element(&self.mm_inputs)?;
        tuple.serialize_element(&NIL)?; // token_type_ids
        tuple.serialize_element(&self.sampling_params)?;
        tuple.serialize_element(&self.return_logprob)?;
        tuple.serialize_element(&self.logprob_start_len)?;
        tuple.serialize_element(&self.top_logprobs_num)?;
        tuple.serialize_element(&self.token_ids_logprob)?;
        tuple.serialize_element(&self.stream)?;
        if self.needs_extended_form() {
            // The fields between `stream` and `require_reasoning`, at the
            // scheduler's own defaults except the seven this wire carries:
            // return_sampling_mask, return_flat_raw_top_logprobs,
            // return_hidden_states, return_routed_experts,
            // routed_experts_start_len, return_indexer_topk, session_id,
            // session_params, lora_id, custom_logit_processor,
            // positional_embed_overrides, then the bootstrap triple
            // (bootstrap_host, bootstrap_port, bootstrap_room), then
            // bootstrap_pair_key, decode_tp_size, routed_dp_rank,
            // disagg_prefill_dp_rank, routing_key.
            tuple.serialize_element(&false)?; // return_sampling_mask
            tuple.serialize_element(&false)?; // return_flat_raw_top_logprobs
            tuple.serialize_element(&self.return_hidden_states)?;
            tuple.serialize_element(&false)?; // return_routed_experts
            tuple.serialize_element(&0u32)?;
            tuple.serialize_element(&false)?;
            tuple.serialize_element(&NIL)?;
            tuple.serialize_element(&NIL)?;
            tuple.serialize_element(&self.lora_id)?;
            tuple.serialize_element(&self.custom_logit_processor)?;
            tuple.serialize_element(&NIL)?; // positional_embed_overrides
            tuple.serialize_element(&self.bootstrap_host)?;
            tuple.serialize_element(&self.bootstrap_port)?;
            tuple.serialize_element(&self.bootstrap_room)?;
            tuple.serialize_element(&NIL)?; // bootstrap_pair_key
            tuple.serialize_element(&NIL)?; // decode_tp_size
            tuple.serialize_element(&self.routed_dp_rank)?;
            tuple.serialize_element(&NIL)?; // disagg_prefill_dp_rank
            tuple.serialize_element(&NIL)?; // routing_key
            tuple.serialize_element(&self.require_reasoning)?;
            if self.needs_encode_slots() {
                // priority, extra_key, no_logs, return_bytes, return_entropy
                // at the scheduler's defaults, then the dispatch's three.
                tuple.serialize_element(&NIL)?;
                tuple.serialize_element(&NIL)?;
                for _ in 0..3 {
                    tuple.serialize_element(&false)?;
                }
                tuple.serialize_element(&self.need_wait_for_mm_inputs)?;
                tuple.serialize_element(&self.num_items_assigned)?;
                tuple.serialize_element(&self.encoder_urls)?;
            }
        }
        tuple.end()
    }
}

/// The defaulted tail of a positional array past its required prefix: reads
/// the next slot when the array still has one and remembers when it ran
/// out, so a shorter array (a frontend that stopped at an earlier field)
/// leaves every later field unset instead of failing the decode.
struct Tail<'a, A> {
    seq: &'a mut A,
    ended: bool,
}

impl<'de, A: SeqAccess<'de>> Tail<'_, A> {
    /// The next slot's value; `None` for a nil slot or an ended array.
    fn next<T: Deserialize<'de>>(&mut self) -> Result<Option<T>, A::Error> {
        if self.ended {
            return Ok(None);
        }
        match self.seq.next_element::<Option<T>>()? {
            Some(value) => Ok(value),
            None => {
                self.ended = true;
                Ok(None)
            }
        }
    }

    /// Skip `count` slots this side leaves at the scheduler's defaults.
    fn skip(&mut self, count: usize) -> Result<(), A::Error> {
        for _ in 0..count {
            self.next::<IgnoredAny>()?;
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TokenizedGenerateReqInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReqVisitor;

        impl<'de> Visitor<'de> for ReqVisitor {
            type Value = TokenizedGenerateReqInput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "a tagged SGLang TokenizedGenerateReqInput positional array"
                )
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                let _http_worker_ipc: IgnoredAny = next_field(&mut seq, "http_worker_ipc")?;
                let input_text = next_field(&mut seq, "input_text")?;
                let input_ids = next_field(&mut seq, "input_ids")?;
                let _input_embeds: IgnoredAny = next_field(&mut seq, "input_embeds")?;
                let mm_inputs = next_field(&mut seq, "mm_inputs")?;
                let _token_type_ids: IgnoredAny = next_field(&mut seq, "token_type_ids")?;
                let sampling_params = next_field(&mut seq, "sampling_params")?;
                let return_logprob = next_field(&mut seq, "return_logprob")?;
                let logprob_start_len = next_field(&mut seq, "logprob_start_len")?;
                let top_logprobs_num = next_field(&mut seq, "top_logprobs_num")?;
                let token_ids_logprob = next_field(&mut seq, "token_ids_logprob")?;
                let stream = next_field(&mut seq, "stream")?;
                // The slots after `stream`: two defaulted flags, then
                // `return_hidden_states`, five more defaulted slots, then
                // `lora_id` and `custom_logit_processor`,
                // `positional_embed_overrides`, the bootstrap triple, 2
                // defaulted ones, `routed_dp_rank`, 2 more defaulted ones, then
                // `require_reasoning`. A shorter array (a frontend that stopped
                // at `stream`) leaves them all unset.
                let mut tail = Tail {
                    seq: &mut seq,
                    ended: false,
                };
                tail.skip(2)?;
                let return_hidden_states = tail
                    .next::<HiddenStatesSlot>()?
                    .is_some_and(HiddenStatesSlot::asked);
                tail.skip(5)?;
                let lora_id = tail.next::<String>()?;
                let custom_logit_processor = tail.next::<String>()?;
                tail.skip(1)?;
                let bootstrap_host = tail.next::<String>()?;
                let bootstrap_port = tail.next::<i32>()?;
                let bootstrap_room = tail.next::<i64>()?;
                tail.skip(2)?;
                let routed_dp_rank = tail.next::<u32>()?;
                tail.skip(2)?;
                let require_reasoning = tail.next::<bool>()?.unwrap_or(false);
                // priority, extra_key, no_logs, return_bytes, return_entropy,
                // then the encode dispatch's three slots.
                tail.skip(5)?;
                let need_wait_for_mm_inputs = tail.next::<bool>()?;
                let num_items_assigned = tail.next::<BTreeMap<u32, Vec<u32>>>()?;
                let encoder_urls = tail.next::<Vec<String>>()?;
                drain_trailing(&mut seq)?;
                Ok(TokenizedGenerateReqInput {
                    rid,
                    input_text,
                    input_ids,
                    mm_inputs,
                    sampling_params,
                    return_logprob,
                    logprob_start_len,
                    top_logprobs_num,
                    token_ids_logprob,
                    stream,
                    return_hidden_states,
                    lora_id,
                    custom_logit_processor,
                    bootstrap_host,
                    bootstrap_port,
                    bootstrap_room,
                    routed_dp_rank,
                    require_reasoning,
                    need_wait_for_mm_inputs,
                    num_items_assigned,
                    encoder_urls,
                    trace_headers: None,
                })
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

/// The `return_hidden_states` slot as the scheduler declares it
/// (`ReturnHiddenStatesMode = bool | "last"`): `true` is the capture of every
/// token's vector, the one this wire asks for; the engine's own frontend's
/// `"last"` (never sent by SMG) and a nil read as not asked.
#[derive(Deserialize)]
#[serde(untagged)]
enum HiddenStatesSlot {
    Flag(bool),
    Other(IgnoredAny),
}

impl HiddenStatesSlot {
    fn asked(self) -> bool {
        matches!(self, Self::Flag(true))
    }
}

/// Skip up to `count` defaulted slots; how many the array still had tells a
/// shorter array (an older frontend's prefix) from one that reaches the
/// slot after them.
fn skip_slots<'de, A: SeqAccess<'de>>(seq: &mut A, count: usize) -> Result<usize, A::Error> {
    let mut skipped = 0;
    while skipped < count && seq.next_element::<IgnoredAny>()?.is_some() {
        skipped += 1;
    }
    Ok(skipped)
}

/// The scheduler's `TokenizedEmbeddingReqInput`, as the SMG plugin decodes
/// it on the same `ADD` frame (the tag tells the two requests apart). The
/// encoder emits the prefix through `dimensions` (13 elements); the fields
/// after it differ between SGLang releases and keep their defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedEmbeddingReqInput {
    pub rid: String,
    pub input_text: Option<String>,
    pub input_ids: Vec<u32>,
    /// Sampling params with `max_new_tokens = 0`, as the Python servicer
    /// sends them (the scheduler normalizes them, nothing samples).
    pub sampling_params: SamplingParams,
    /// Token type ids for cross-encoder inputs; empty for a plain embedding.
    pub token_type_ids: Vec<i32>,
    /// Requested output dimensionality (Matryoshka truncation); `None` keeps
    /// the model's.
    pub dimensions: Option<u32>,
}

impl Default for TokenizedEmbeddingReqInput {
    fn default() -> Self {
        Self {
            rid: String::new(),
            input_text: None,
            input_ids: Vec::new(),
            sampling_params: SamplingParams {
                max_new_tokens: Some(0),
                ..SamplingParams::default()
            },
            token_type_ids: Vec::new(),
            dimensions: None,
        }
    }
}

impl Serialize for TokenizedEmbeddingReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(EMBEDDING_WIRE_FIELDS.len() + 1)?;
        tuple.serialize_element(TOKENIZED_EMBEDDING_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&NIL)?; // http_worker_ipc
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
        tuple.serialize_element(&NIL)?; // mm_inputs
        tuple.serialize_element(
            &(!self.token_type_ids.is_empty()).then_some(&self.token_type_ids),
        )?;
        tuple.serialize_element(&self.sampling_params)?;
        tuple.serialize_element(&NIL)?; // lora_id
        tuple.serialize_element(&NIL)?; // positional_embed_overrides
        tuple.serialize_element(&NIL)?; // routed_dp_rank
        tuple.serialize_element(&NIL)?; // priority
        tuple.serialize_element(&self.dimensions)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TokenizedEmbeddingReqInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReqVisitor;

        impl<'de> Visitor<'de> for ReqVisitor {
            type Value = TokenizedEmbeddingReqInput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "a tagged SGLang TokenizedEmbeddingReqInput positional array"
                )
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, TOKENIZED_EMBEDDING_REQ_INPUT_TAG)?;
                read_embedding_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

/// The embedding request's fields after the tag; shared by the struct's own
/// decoder and the request enum's.
fn read_embedding_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> Result<TokenizedEmbeddingReqInput, A::Error> {
    let rid = next_field(seq, "rid")?;
    let _http_worker_ipc: IgnoredAny = next_field(seq, "http_worker_ipc")?;
    let input_text = next_field(seq, "input_text")?;
    let input_ids = next_field(seq, "input_ids")?;
    let _mm_inputs: IgnoredAny = next_field(seq, "mm_inputs")?;
    let token_type_ids: Option<Vec<i32>> = next_field(seq, "token_type_ids")?;
    let sampling_params = next_field(seq, "sampling_params")?;
    // lora_id, positional_embed_overrides, routed_dp_rank, priority: defaults.
    let dimensions = if skip_slots(seq, 4)? == 4 {
        seq.next_element::<Option<u32>>()?.flatten()
    } else {
        None
    };
    drain_trailing(seq)?;
    Ok(TokenizedEmbeddingReqInput {
        rid,
        input_text,
        input_ids,
        sampling_params,
        token_type_ids: token_type_ids.unwrap_or_default(),
        dimensions,
    })
}

/// The scheduler's own notice that a request's encode dispatch failed
/// (`EncoderDispatchErrorReq`, a tagged positional struct like the requests:
/// `[tag, rid, http_worker_ipc, error_msg, error_code]`). The engine's
/// frontend sends it after the request it names, on the same input socket,
/// when its dispatch to the encoders failed; the scheduler's receiver then
/// aborts that request with the message and the HTTP code instead of
/// waiting out the embeddings that will never come. It is a notice, not a
/// request: nothing answers it, so it rides [`crate::Client::notify`], never
/// a submit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderDispatchErrorReq {
    /// The request whose dispatch failed.
    pub rid: String,
    /// The encoder's (or the dispatcher's) message.
    pub error_msg: String,
    /// The HTTP status the scheduler aborts the request with (the engine's
    /// frontend sends 502 for a dispatch it could not deliver).
    pub error_code: u16,
}

impl Serialize for EncoderDispatchErrorReq {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(ENCODER_DISPATCH_ERROR_WIRE_FIELDS.len() + 1)?;
        tuple.serialize_element(ENCODER_DISPATCH_ERROR_REQ_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&NIL)?; // http_worker_ipc
        tuple.serialize_element(&self.error_msg)?;
        tuple.serialize_element(&self.error_code)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for EncoderDispatchErrorReq {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ErrorVisitor;

        impl<'de> Visitor<'de> for ErrorVisitor {
            type Value = EncoderDispatchErrorReq;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "a tagged SGLang EncoderDispatchErrorReq positional array"
                )
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, ENCODER_DISPATCH_ERROR_REQ_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                let _http_worker_ipc: IgnoredAny = next_field(&mut seq, "http_worker_ipc")?;
                let error_msg = next_field(&mut seq, "error_msg")?;
                let error_code = next_field(&mut seq, "error_code")?;
                drain_trailing(&mut seq)?;
                Ok(EncoderDispatchErrorReq {
                    rid,
                    error_msg,
                    error_code,
                })
            }
        }

        deserializer.deserialize_seq(ErrorVisitor)
    }
}

/// What rides the `ADD` frame: a generation or an embedding request, told
/// apart by the tag, both keyed by `rid`.
#[derive(Debug, Clone, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "each variant mirrors one of the scheduler's request structs slot for slot; a \
              request is built, encoded on submit and dropped, never queued"
)]
pub enum SglangRequest {
    Generate(TokenizedGenerateReqInput),
    Embed(TokenizedEmbeddingReqInput),
}

impl SglangRequest {
    pub fn rid(&self) -> &str {
        match self {
            Self::Generate(request) => &request.rid,
            Self::Embed(request) => &request.rid,
        }
    }
}

impl From<TokenizedGenerateReqInput> for SglangRequest {
    fn from(request: TokenizedGenerateReqInput) -> Self {
        Self::Generate(request)
    }
}

impl From<TokenizedEmbeddingReqInput> for SglangRequest {
    fn from(request: TokenizedEmbeddingReqInput) -> Self {
        Self::Embed(request)
    }
}

impl Serialize for SglangRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Generate(request) => request.serialize(serializer),
            Self::Embed(request) => request.serialize(serializer),
        }
    }
}

/// A generation's tensors; an embedding request carries none on this wire.
impl TensorFields for SglangRequest {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        if let Self::Generate(request) = self {
            request.for_each_tensor(visit);
        }
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        if let Self::Generate(request) = self {
            request.for_each_tensor_mut(visit);
        }
    }
}

/// The field order of the scheduler's request structs in the releases the
/// wire was recorded against (`__struct_fields__`, base classes first), for
/// the tests of this module and of the handshake check.
#[cfg(test)]
pub(crate) mod recorded {
    /// SGLang 0.5.21 and 0.5.22 declare the same generate request.
    pub(crate) const GENERATE_FIELDS_0_5_21: [&str; 48] = [
        "rid",
        "http_worker_ipc",
        "input_text",
        "input_ids",
        "input_embeds",
        "mm_inputs",
        "token_type_ids",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "stream",
        "return_sampling_mask",
        "return_flat_raw_top_logprobs",
        "return_hidden_states",
        "return_routed_experts",
        "routed_experts_start_len",
        "return_indexer_topk",
        "session_id",
        "session_params",
        "lora_id",
        "custom_logit_processor",
        "positional_embed_overrides",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "bootstrap_pair_key",
        "decode_tp_size",
        "routed_dp_rank",
        "disagg_prefill_dp_rank",
        "routing_key",
        "require_reasoning",
        "priority",
        "extra_key",
        "no_logs",
        "return_bytes",
        "return_entropy",
        "need_wait_for_mm_inputs",
        "num_items_assigned",
        "encoder_urls",
        "multi_item_delimiter_indices",
        "token_indices_to_pool",
        "time_stats",
        "cache_salt",
        "kv_hints",
        "pp_prefetch_ticketed",
        "sampling_logprobs_mode",
    ];
    pub(crate) const GENERATE_FIELDS_0_5_22: [&str; 48] = GENERATE_FIELDS_0_5_21;

    /// SGLang 0.5.21's embedding request.
    pub(crate) const EMBEDDING_FIELDS_0_5_21: [&str; 16] = [
        "rid",
        "http_worker_ipc",
        "input_text",
        "input_ids",
        "mm_inputs",
        "token_type_ids",
        "sampling_params",
        "lora_id",
        "positional_embed_overrides",
        "routed_dp_rank",
        "priority",
        "dimensions",
        "return_pooled_hidden_states",
        "multi_item_delimiter_indices",
        "token_indices_to_pool",
        "time_stats",
    ];
    /// SGLang 0.5.22 appended `decision_layout` before `time_stats`, past
    /// the prefix this side fills.
    pub(crate) const EMBEDDING_FIELDS_0_5_22: [&str; 17] = [
        "rid",
        "http_worker_ipc",
        "input_text",
        "input_ids",
        "mm_inputs",
        "token_type_ids",
        "sampling_params",
        "lora_id",
        "positional_embed_overrides",
        "routed_dp_rank",
        "priority",
        "dimensions",
        "return_pooled_hidden_states",
        "multi_item_delimiter_indices",
        "token_indices_to_pool",
        "decision_layout",
        "time_stats",
    ];
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::{recorded::*, *};
    use crate::{
        codec::{decode_msgpack, encode_msgpack},
        protocol::sglang::sampling::{
            recorded::PYTHON_PINNED_0_5_22, WIRE_FIELDS as SAMPLING_WIRE_FIELDS,
        },
    };

    /// `msgspec.msgpack.encode` of the positional request the SMG plugin's
    /// tests build: rid `r1`, ids `[1, 2, 3]`, `SamplingParams(max_new_tokens=8,
    /// temperature=0.0, top_k=-1, n=1)` (the engine's `__post_init__` already
    /// collapsed greedy to temperature 1.0 / top_k 1), `stream=True`,
    /// `require_reasoning=True`, the 19 defaulted fields between them at
    /// their defaults.
    const PYTHON_REQUEST: &str = "dc0022b9546f6b656e697a656447656e6572617465526571496e707574a27231c0c093010203c0c0c0dc002008c0c0c0cb3ff0000000000000cb3ff000000000000001cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2c2c2ff00c0c3c2c2c2c200c2c0c0c0c0c0c0c0c0c0c0c0c0c0c3";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn encodes_the_prefix_through_require_reasoning_with_the_tag_first() {
        let request = TokenizedGenerateReqInput {
            rid: "r1".into(),
            input_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                max_new_tokens: Some(8),
                ..SamplingParams::default()
            },
            stream: true,
            require_reasoning: true,
            ..TokenizedGenerateReqInput::default()
        };
        let value: Value = rmp_serde::from_slice(&encode_msgpack(&request).unwrap()).unwrap();
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 34);
        assert_eq!(arr[0].as_str(), Some(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[2].is_nil(), "http_worker_ipc at 2");
        assert!(arr[3].is_nil(), "input_text at 3");
        assert_eq!(arr[4].as_array().unwrap().len(), 3, "input_ids at 4");
        assert!(arr[5].is_nil() && arr[6].is_nil() && arr[7].is_nil());
        assert_eq!(
            arr[8].as_array().unwrap().len(),
            SAMPLING_WIRE_FIELDS.len(),
            "sampling_params at 8, their API prefix"
        );
        assert_eq!(arr[9].as_bool(), Some(false), "return_logprob at 9");
        assert_eq!(arr[10].as_i64(), Some(-1), "logprob_start_len at 10");
        assert_eq!(arr[11].as_u64(), Some(0), "top_logprobs_num at 11");
        assert!(arr[12].is_nil(), "token_ids_logprob at 12");
        assert_eq!(arr[13].as_bool(), Some(true), "stream at 13");
        for (i, slot) in arr[14..18].iter().enumerate() {
            assert_eq!(slot.as_bool(), Some(false), "return_* flag at {}", 14 + i);
        }
        assert_eq!(arr[18].as_u64(), Some(0), "routed_experts_start_len at 18");
        assert_eq!(arr[19].as_bool(), Some(false), "return_indexer_topk at 19");
        assert!(
            arr[20..33].iter().all(Value::is_nil),
            "session..routing_key nil"
        );
        assert_eq!(arr[33].as_bool(), Some(true), "require_reasoning at 33");
    }

    #[test]
    fn a_request_without_reasoning_stops_at_stream() {
        let request = TokenizedGenerateReqInput {
            rid: "r1".into(),
            input_ids: vec![1],
            stream: true,
            ..TokenizedGenerateReqInput::default()
        };
        let value: Value = rmp_serde::from_slice(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 14);
        let decoded: TokenizedGenerateReqInput =
            decode_msgpack(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn a_prefix_that_stops_at_stream_decodes_with_require_reasoning_off() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        arr.truncate(14);
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert!(decoded.stream && !decoded.require_reasoning);
    }

    #[test]
    fn python_pinned_request_round_trips() {
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&hex(PYTHON_REQUEST)).unwrap();
        assert_eq!(decoded.rid, "r1");
        assert_eq!(decoded.input_ids, vec![1, 2, 3]);
        assert_eq!(decoded.sampling_params.max_new_tokens, Some(8));
        assert_eq!(decoded.sampling_params.top_k, 1);
        assert!(decoded.stream && decoded.require_reasoning);
        // What SMG encodes is the Python struct's array with the sampling
        // params cut to their API prefix: the same request on either release.
        let encoded = encode_msgpack(&decoded).unwrap();
        let again: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(again, decoded);
        let python: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        let ours: Vec<Value> = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(ours.len(), python.len());
        for (index, (ours, python)) in ours.iter().zip(&python).enumerate() {
            if index == 8 {
                let prefix = &python.as_array().unwrap()[..SAMPLING_WIRE_FIELDS.len()];
                assert_eq!(ours.as_array().unwrap(), prefix, "sampling_params at 8");
            } else {
                assert_eq!(ours, python, "element {index}");
            }
        }
    }

    /// The newer release's sampling array (one field longer) nested in the
    /// same request decodes to the same request.
    #[test]
    fn a_request_with_the_newer_releases_sampling_array_decodes_the_same() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        arr[8] = rmp_serde::from_slice(&hex(PYTHON_PINNED_0_5_22)).unwrap();
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let newer: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        let pinned: TokenizedGenerateReqInput = decode_msgpack(&hex(PYTHON_REQUEST)).unwrap();
        assert_eq!(newer, pinned);
    }

    /// The wire field lists name the slots the encoders fill, and head the
    /// field order of both recorded releases.
    #[test]
    fn the_wire_field_lists_name_the_slots_the_encoders_fill() {
        assert_eq!(GENERATE_WIRE_FIELDS[7], "sampling_params");
        assert_eq!(GENERATE_WIRE_FIELDS[12], "stream");
        assert_eq!(GENERATE_WIRE_FIELDS[15], "return_hidden_states");
        assert_eq!(GENERATE_WIRE_FIELDS[21], "lora_id");
        assert_eq!(GENERATE_WIRE_FIELDS[22], "custom_logit_processor");
        assert_eq!(GENERATE_WIRE_FIELDS[32], "require_reasoning");
        assert_eq!(GENERATE_WIRE_FIELDS[38], "need_wait_for_mm_inputs");
        assert_eq!(GENERATE_WIRE_FIELDS[39], "num_items_assigned");
        assert_eq!(GENERATE_WIRE_FIELDS[40], "encoder_urls");
        assert_eq!(GENERATE_WIRE_FIELDS.len(), 41);
        assert_eq!(EMBEDDING_WIRE_FIELDS[6], "sampling_params");
        assert_eq!(EMBEDDING_WIRE_FIELDS[11], "dimensions");
        for release in [&GENERATE_FIELDS_0_5_21[..], &GENERATE_FIELDS_0_5_22[..]] {
            assert!(release.starts_with(&GENERATE_WIRE_FIELDS));
        }
        for release in [&EMBEDDING_FIELDS_0_5_21[..], &EMBEDDING_FIELDS_0_5_22[..]] {
            assert!(release.starts_with(&EMBEDDING_WIRE_FIELDS));
        }
    }

    #[test]
    fn decoder_tolerates_the_full_length_python_array() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        for _ in 0..35 {
            arr.push(Value::Nil); // the defaulted tail the scheduler declares
        }
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.rid, "r1");
    }

    #[test]
    fn embedding_request_encodes_the_prefix_through_dimensions() {
        let request = TokenizedEmbeddingReqInput {
            rid: "e1".into(),
            input_text: Some("hi".into()),
            input_ids: vec![1, 2, 3],
            ..Default::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(array.len(), 13);
        assert_eq!(array[0].as_str(), Some("TokenizedEmbeddingReqInput"));
        assert_eq!(array[1].as_str(), Some("e1"));
        assert_eq!(array[4], Value::Array(vec![1.into(), 2.into(), 3.into()]));
        assert_eq!(array[6], Value::Nil, "no token type ids");
        // sampling_params is the API prefix with max_new_tokens 0.
        assert_eq!(
            array[7].as_array().map(Vec::len),
            Some(SAMPLING_WIRE_FIELDS.len())
        );
        assert_eq!(array[7].as_array().unwrap()[0], Value::from(0));
        assert_eq!(array[12], Value::Nil, "dimensions unset");
        let back: TokenizedEmbeddingReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, request);
        // Token type ids ride their slot; a longer array (a later SGLang's
        // extra fields) still decodes.
        let typed = TokenizedEmbeddingReqInput {
            token_type_ids: vec![0, 0, 1],
            dimensions: Some(64),
            ..request
        };
        let mut array: Vec<Value> = decode_msgpack(&encode_msgpack(&typed).unwrap()).unwrap();
        assert_eq!(array[6], Value::Array(vec![0.into(), 0.into(), 1.into()]));
        array.extend([Value::Nil, Value::Nil, Value::Nil]);
        let back: TokenizedEmbeddingReqInput =
            decode_msgpack(&encode_msgpack(&array).unwrap()).unwrap();
        assert_eq!(back, typed);
    }

    #[test]
    fn request_enum_serializes_either_variant_under_its_rid() {
        let generate = SglangRequest::from(TokenizedGenerateReqInput {
            rid: "g".into(),
            ..Default::default()
        });
        let embed = SglangRequest::from(TokenizedEmbeddingReqInput {
            rid: "e".into(),
            ..Default::default()
        });
        assert_eq!((generate.rid(), embed.rid()), ("g", "e"));
        let tag = |request: &SglangRequest| -> String {
            let array: Vec<Value> = decode_msgpack(&encode_msgpack(request).unwrap()).unwrap();
            array[0].as_str().unwrap().to_string()
        };
        assert_eq!(tag(&generate), "TokenizedGenerateReqInput");
        assert_eq!(tag(&embed), "TokenizedEmbeddingReqInput");
    }

    #[test]
    fn lora_and_logit_processor_take_the_extended_form() {
        let request = TokenizedGenerateReqInput {
            rid: "r".into(),
            lora_id: Some("adapter".into()),
            custom_logit_processor: Some("proc".into()),
            ..Default::default()
        };
        let array: Vec<Value> = decode_msgpack(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(array.len(), 34);
        assert_eq!(array[22].as_str(), Some("adapter"));
        assert_eq!(array[23].as_str(), Some("proc"));
        assert_eq!(
            array[33],
            Value::Boolean(false),
            "require_reasoning stays off"
        );
        let back: TokenizedGenerateReqInput =
            decode_msgpack(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(back, request);
    }

    /// `return_hidden_states` rides its own slot of the extended form (the
    /// array's element 16), is reason enough for that form on its own, and
    /// round-trips. The pinned Python request carries `False` there; the
    /// engine frontend's `"last"` in the slot decodes as not asked.
    #[test]
    fn hidden_states_take_the_extended_form_at_their_slot() {
        let request = TokenizedGenerateReqInput {
            rid: "h".into(),
            input_ids: vec![1, 2],
            return_hidden_states: true,
            ..Default::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(array.len(), 34);
        assert_eq!(
            array[14],
            Value::Boolean(false),
            "return_sampling_mask at 14"
        );
        assert_eq!(
            array[15],
            Value::Boolean(false),
            "return_flat_raw_top_logprobs at 15"
        );
        assert_eq!(
            array[16],
            Value::Boolean(true),
            "return_hidden_states at 16"
        );
        assert_eq!(
            array[17],
            Value::Boolean(false),
            "return_routed_experts at 17"
        );
        assert_eq!(
            array[33],
            Value::Boolean(false),
            "require_reasoning stays off"
        );
        let back: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, request);
        let pinned: TokenizedGenerateReqInput = decode_msgpack(&hex(PYTHON_REQUEST)).unwrap();
        assert!(!pinned.return_hidden_states);
        let mut last: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        last[16] = Value::from("last");
        let bytes = rmp_serde::to_vec(&Value::Array(last)).unwrap();
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert!(!decoded.return_hidden_states && decoded.require_reasoning);
    }

    /// A prefill/decode request carries its rendezvous in the scheduler's own
    /// slots: the bootstrap triple takes the extended form, lands at the
    /// struct's indices, and round-trips; room 0 is a room (the fake-transfer
    /// probe's), and a partial triple leaves the other slots nil.
    #[test]
    fn a_disaggregated_request_fills_the_bootstrap_slots() {
        let request = TokenizedGenerateReqInput {
            rid: "pd".into(),
            input_ids: vec![1, 2],
            bootstrap_host: Some("10.0.0.7".into()),
            bootstrap_port: Some(8998),
            bootstrap_room: Some(41),
            ..Default::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(
            array.len(),
            34,
            "the extended form through require_reasoning"
        );
        assert!(
            array[22].is_nil() && array[23].is_nil(),
            "no LoRA, no processor"
        );
        assert!(array[24].is_nil(), "positional_embed_overrides at 24");
        assert_eq!(array[25].as_str(), Some("10.0.0.7"), "bootstrap_host at 25");
        assert_eq!(array[26].as_i64(), Some(8998), "bootstrap_port at 26");
        assert_eq!(array[27].as_i64(), Some(41), "bootstrap_room at 27");
        assert!(
            array[28..33].iter().all(Value::is_nil),
            "bootstrap_pair_key .. routing_key nil"
        );
        assert_eq!(array[33], Value::Boolean(false), "require_reasoning at 33");
        let back: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, request);

        let partial = TokenizedGenerateReqInput {
            rid: "probe".into(),
            bootstrap_room: Some(0),
            ..Default::default()
        };
        let encoded = encode_msgpack(&partial).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(array.len(), 34);
        assert!(array[25].is_nil() && array[26].is_nil());
        assert_eq!(array[27].as_i64(), Some(0), "room 0 is sent, not dropped");
        let back: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, partial);
        // The slots sit inside the prefix the handshake check compares.
        assert_eq!(
            &GENERATE_WIRE_FIELDS[24..27],
            ["bootstrap_host", "bootstrap_port", "bootstrap_room"]
        );
    }

    /// A request whose media the encoders serve reaches `encoder_urls`: the
    /// defaulted slots 33-37 at the scheduler's defaults, then the dispatch's
    /// three in 38-40 as the engine's frontend sets them; a request without
    /// a dispatch keeps the shorter forms.
    #[test]
    fn an_encode_dispatched_request_fills_the_three_slots() {
        let request = TokenizedGenerateReqInput {
            rid: "epd".into(),
            input_text: Some("<image> what is this".into()),
            input_ids: vec![1, 2, 3],
            need_wait_for_mm_inputs: Some(true),
            num_items_assigned: Some(BTreeMap::from([(1, vec![1, 0])])),
            encoder_urls: Some(vec![
                "grpc://10.0.0.8:30001".into(),
                "grpc://10.0.0.9:30001".into(),
            ]),
            ..Default::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(array.len(), 42, "the extended form through encoder_urls");
        assert_eq!(
            array[3].as_str(),
            Some("<image> what is this"),
            "input_text at 3"
        );
        assert_eq!(array[33], Value::Boolean(false), "require_reasoning at 33");
        assert!(
            array[34].is_nil() && array[35].is_nil(),
            "priority, extra_key nil"
        );
        for (i, slot) in array[36..39].iter().enumerate() {
            assert_eq!(
                slot,
                &Value::Boolean(false),
                "no_logs .. return_entropy at {}",
                36 + i
            );
        }
        assert_eq!(
            array[39],
            Value::Boolean(true),
            "need_wait_for_mm_inputs at 39"
        );
        let assigned = array[40]
            .as_map()
            .expect("num_items_assigned is a map at 40");
        assert_eq!(assigned.len(), 1);
        assert_eq!(
            assigned[0].0.as_u64(),
            Some(1),
            "the modality's enum value keys it"
        );
        assert_eq!(
            assigned[0]
                .1
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64())
                .collect::<Vec<_>>(),
            vec![Some(1), Some(0)]
        );
        let urls = array[41].as_array().expect("encoder_urls at 41");
        assert_eq!(urls.len(), 2);
        let back: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, request);
        // The slots sit inside the prefix the handshake check compares.
        assert_eq!(
            &GENERATE_WIRE_FIELDS[38..41],
            [
                "need_wait_for_mm_inputs",
                "num_items_assigned",
                "encoder_urls"
            ]
        );

        // No dispatch: the form stops where it did before.
        let plain = TokenizedGenerateReqInput {
            rid: "pd".into(),
            bootstrap_room: Some(0),
            ..Default::default()
        };
        let array: Vec<Value> = decode_msgpack(&encode_msgpack(&plain).unwrap()).unwrap();
        assert_eq!(array.len(), 34);
        let decoded: TokenizedGenerateReqInput =
            decode_msgpack(&encode_msgpack(&plain).unwrap()).unwrap();
        assert_eq!(
            (
                decoded.need_wait_for_mm_inputs,
                decoded.num_items_assigned,
                decoded.encoder_urls
            ),
            (None, None, None)
        );
    }

    /// The failure notice is the scheduler's own `EncoderDispatchErrorReq`,
    /// byte for byte what `msgspec` writes for it (the pinned bytes come from
    /// the engine's struct definition: tag, rid, `http_worker_ipc` nil, the
    /// message, the code).
    #[test]
    fn the_dispatch_error_notice_is_the_schedulers_own_struct() {
        const PYTHON_PINNED: &str = "95b7456e636f64657244697370617463684572726f72526571a56570642d31c0d925456e636f6465206661696c656420666f722072657175657374206570642d313a20626f6f6dcd01f6";
        let notice = EncoderDispatchErrorReq {
            rid: "epd-1".into(),
            error_msg: "Encode failed for request epd-1: boom".into(),
            error_code: 502,
        };
        let encoded = encode_msgpack(&notice).unwrap();
        assert_eq!(encoded, hex(PYTHON_PINNED));
        let back: EncoderDispatchErrorReq = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, notice);
        assert_eq!(
            ENCODER_DISPATCH_ERROR_WIRE_FIELDS,
            ["rid", "http_worker_ipc", "error_msg", "error_code"]
        );
    }

    /// A request routed to a data-parallel rank carries the rank in the
    /// scheduler's own `routed_dp_rank` slot: the extended form, the struct's
    /// index, a round trip, and the protocol's rank hook reads it (an
    /// embedding request has no pin on this wire). Unrouted, the slot is nil
    /// and the hook leaves the placement to the connector.
    #[test]
    fn a_routed_request_fills_the_routed_dp_rank_slot() {
        use crate::protocol::{sglang::SglangProtocol, EngineProtocol};

        let routed = TokenizedGenerateReqInput {
            rid: "dp".into(),
            input_ids: vec![1, 2],
            routed_dp_rank: Some(1),
            ..Default::default()
        };
        let encoded = encode_msgpack(&routed).unwrap();
        let array: Vec<Value> = decode_msgpack(&encoded).unwrap();
        assert_eq!(
            array.len(),
            34,
            "the extended form through require_reasoning"
        );
        assert!(
            array[25..30].iter().all(Value::is_nil),
            "bootstrap_host .. decode_tp_size nil"
        );
        assert_eq!(array[30].as_u64(), Some(1), "routed_dp_rank at 30");
        assert!(
            array[31].is_nil() && array[32].is_nil(),
            "disagg_prefill_dp_rank, routing_key nil"
        );
        assert_eq!(array[33], Value::Boolean(false), "require_reasoning at 33");
        let back: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, routed);
        assert_eq!(GENERATE_WIRE_FIELDS[29], "routed_dp_rank");
        assert_eq!(
            SglangProtocol::data_parallel_rank(&SglangRequest::Generate(routed)),
            Some(1)
        );

        let unrouted = TokenizedGenerateReqInput {
            rid: "plain".into(),
            input_ids: vec![1, 2],
            ..Default::default()
        };
        let array: Vec<Value> = decode_msgpack(&encode_msgpack(&unrouted).unwrap()).unwrap();
        assert_eq!(array.len(), 14, "nothing past stream: the short form");
        assert_eq!(
            SglangProtocol::data_parallel_rank(&SglangRequest::Generate(unrouted)),
            None
        );
        let embed = SglangRequest::Embed(TokenizedEmbeddingReqInput {
            rid: "e".into(),
            input_ids: vec![1],
            ..Default::default()
        });
        assert_eq!(SglangProtocol::data_parallel_rank(&embed), None);
    }

    #[test]
    fn trace_headers_ride_as_the_metadata_frame_not_the_positional_array() {
        use crate::protocol::{
            sglang::{SglangLayout, SglangProtocol},
            EngineProtocol,
        };

        let headers = BTreeMap::from([
            ("traceparent".to_string(), "00-aa-bb-01".to_string()),
            ("tracestate".to_string(), "vendor=a".to_string()),
        ]);
        let plain = TokenizedGenerateReqInput {
            rid: "t1".to_string(),
            input_ids: vec![1, 2],
            ..TokenizedGenerateReqInput::default()
        };
        let traced = TokenizedGenerateReqInput {
            trace_headers: Some(headers.clone()),
            ..plain.clone()
        };
        let layout = SglangLayout::default();
        let (plain_payload, plain_aux) =
            SglangProtocol::encode_add(&SglangRequest::Generate(plain), &layout).unwrap();
        let (payload, aux) =
            SglangProtocol::encode_add(&SglangRequest::Generate(traced), &layout).unwrap();
        // The scheduler's own array is byte-identical: the context never enters it.
        assert_eq!(payload, plain_payload);
        assert!(plain_aux.is_empty());
        assert_eq!(aux.len(), 1, "one metadata frame");
        let meta_frame = &aux[0];
        let meta: RequestMeta = decode_msgpack(meta_frame.as_ref()).unwrap();
        assert_eq!(meta.trace_headers, Some(headers));
        // On the wire the frame is a map keyed by name, as the plugin reads it.
        let raw: BTreeMap<String, BTreeMap<String, String>> =
            decode_msgpack(meta_frame.as_ref()).unwrap();
        assert_eq!(raw["trace_headers"]["traceparent"], "00-aa-bb-01");
        // A decoded request never carries one: the array has no slot for it.
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded.trace_headers, None);
        assert!(RequestMeta::for_request(&decoded).is_none());
    }

    /// The multimodal payload rides slot 6 (`mm_inputs`), nil for a text
    /// request, and round-trips; nothing else of the array moves.
    #[test]
    fn multimodal_inputs_ride_their_slot() {
        use crate::protocol::sglang::multimodal::{
            SglangWireMmItem, SglangWireModality, SglangWireTensor, PROCESSOR_OUTPUT_WIRE_FIELDS,
        };
        let text = TokenizedGenerateReqInput {
            rid: "m".into(),
            input_ids: vec![1, 7, 7, 7, 2],
            ..TokenizedGenerateReqInput::default()
        };
        let media = TokenizedGenerateReqInput {
            mm_inputs: Some(SglangWireMmInputs {
                mm_items: vec![SglangWireMmItem {
                    modality: SglangWireModality::Image,
                    hash: None,
                    offsets: Some(vec![(1, 3)]),
                    feature: Some(
                        SglangWireTensor::new("uint8", vec![3, 2], Bytes::from(vec![9u8; 6]))
                            .unwrap(),
                    ),
                    model_specific_data: BTreeMap::new(),
                }],
                im_token_id: Some(7),
                video_token_id: None,
            }),
            ..text.clone()
        };
        let text_array: Vec<Value> = decode_msgpack(&encode_msgpack(&text).unwrap()).unwrap();
        let media_array: Vec<Value> = decode_msgpack(&encode_msgpack(&media).unwrap()).unwrap();
        assert_eq!(text_array.len(), 14);
        assert_eq!(media_array.len(), 14);
        assert_eq!(text_array[6], Value::Nil);
        // The processor output rides as the prefix this side fills.
        let slots_sent = PROCESSOR_OUTPUT_WIRE_FIELDS.len();
        assert!(matches!(&media_array[6], Value::Array(slots) if slots.len() == slots_sent));
        for index in (0..14).filter(|index| *index != 6) {
            assert_eq!(text_array[index], media_array[index], "element {index}");
        }
        let back: TokenizedGenerateReqInput =
            decode_msgpack(&encode_msgpack(&media).unwrap()).unwrap();
        assert_eq!(back, media);
    }

    /// The `ADD` payload the plugin's own test decodes through the scheduler's
    /// real classes (`grpc_servicer/tests/test_sglang_zmq_msgpack.py`,
    /// `RUST_MM_REQUEST`): rid `mm`, ids `[1, 7, 7, 7, 2]`, the default
    /// sampling params, one image item whose feature is
    /// `torch.arange(8, dtype=torch.float32).reshape(2, 4) * 0.5` with
    /// `image_grid_thw = [[1, 2, 2]]` (int64), offsets `[(1, 3)]` and
    /// `im_token_id = 7`. The two pins are one constant: a change here is a
    /// change there.
    #[test]
    fn the_payload_the_plugins_test_decodes_is_pinned() {
        use crate::protocol::sglang::multimodal::{
            SglangWireMmItem, SglangWireModality, SglangWireTensor,
        };
        const PLUGIN_TEST_PAYLOAD: &str = "9eb9546f6b656e697a656447656e6572617465526571496e707574a26d6dc0c0950107070702c094919901c0c09192010301c734020000001093920204a7666c6f61743332a3637075000000000000003f0000803f0000c03f00000040000020400000404000006040c0c281ae696d6167655f677269645f746877c72a020000000e93920103a5696e743634a3637075010000000000000002000000000000000200000000000000c0c007c0dc001acc80c0c0c0cb3ff0000000000000cb3ff0000000000000ffcb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c2ff00c0c2";
        let pixels: Vec<u8> = (0..8)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        let grid: Vec<u8> = [1i64, 2, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut request = TokenizedGenerateReqInput {
            rid: "mm".into(),
            input_ids: vec![1, 7, 7, 7, 2],
            mm_inputs: Some(SglangWireMmInputs {
                mm_items: vec![SglangWireMmItem {
                    modality: SglangWireModality::Image,
                    hash: None,
                    offsets: Some(vec![(1, 3)]),
                    feature: Some(
                        SglangWireTensor::new("float32", vec![2, 4], Bytes::from(pixels)).unwrap(),
                    ),
                    model_specific_data: BTreeMap::from([(
                        "image_grid_thw".to_string(),
                        SglangWireTensor::new("int64", vec![1, 3], Bytes::from(grid)).unwrap(),
                    )]),
                }],
                im_token_id: Some(7),
                video_token_id: None,
            }),
            ..TokenizedGenerateReqInput::default()
        };
        assert_eq!(encode_msgpack(&request).unwrap(), hex(PLUGIN_TEST_PAYLOAD));
        let mm = request.mm_inputs.as_mut().unwrap();
        mm.video_token_id = Some(7);
        let item = &mut mm.mm_items[0];
        item.modality = SglangWireModality::Video;
        let grid = item.model_specific_data.remove("image_grid_thw").unwrap();
        item.model_specific_data
            .insert("video_grid_thw".into(), grid);
        // The Python regression decodes this exact payload and runs SGLang's padding.
        const VIDEO_PAYLOAD: &str = "9eb9546f6b656e697a656447656e6572617465526571496e707574a26d6dc0c0950107070702c099919902c0c09192010301c734020000001093920204a7666c6f61743332a3637075000000000000003f0000803f0000c03f00000040000020400000404000006040c0c281ae766964656f5f677269645f746877c72a020000000e93920103a5696e743634a3637075010000000000000002000000000000000200000000000000c0c007c0c0c0c007c0dc001acc80c0c0c0cb3ff0000000000000cb3ff0000000000000ffcb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c2ff00c0c2";
        assert_eq!(encode_msgpack(&request).unwrap(), hex(VIDEO_PAYLOAD));
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&hex(VIDEO_PAYLOAD)).unwrap();
        assert_eq!(decoded, request);
    }

    /// The `ADD` message the plugin's own test decodes with the pixel tensor
    /// as an auxiliary frame (`grpc_servicer/tests/test_sglang_zmq_msgpack.py`,
    /// `RUST_MM_FRAMED_REQUEST`): the request above with the feature
    /// `torch.arange(8192, dtype=torch.float32) * 0.5` (32 KiB, the frame
    /// bar) on a scheduler whose plugin resolves frames. The payload carries
    /// the frame's index in the feature's slot, the grid inline; the frames
    /// behind it are the pixels, then the empty metadata map. The two pins
    /// are one constant: a change here is a change there.
    #[test]
    fn the_framed_payload_the_plugins_test_decodes_is_pinned() {
        use crate::{
            codec::tensor::TENSOR_AUX_MIN_BYTES,
            protocol::{
                sglang::{
                    multimodal::{SglangWireMmItem, SglangWireModality, SglangWireTensor},
                    SglangLayout, SglangProtocol,
                },
                EngineProtocol,
            },
        };
        const PLUGIN_TEST_FRAMED_PAYLOAD: &str = "9eb9546f6b656e697a656447656e6572617465526571496e707574a26d6dc0c0950107070702c094919901c0c09192010301c70e649391cd2000a7666c6f6174333201c0c281ae696d6167655f677269645f746877c72a020000000e93920103a5696e743634a3637075010000000000000002000000000000000200000000000000c0c007c0dc001acc80c0c0c0cb3ff0000000000000cb3ff0000000000000ffcb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c2ff00c0c2";
        let pixels: Vec<u8> = (0..8192)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        assert_eq!(pixels.len(), TENSOR_AUX_MIN_BYTES);
        let grid: Vec<u8> = [1i64, 2, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let request = TokenizedGenerateReqInput {
            rid: "mm".into(),
            input_ids: vec![1, 7, 7, 7, 2],
            mm_inputs: Some(SglangWireMmInputs {
                mm_items: vec![SglangWireMmItem {
                    modality: SglangWireModality::Image,
                    hash: None,
                    offsets: Some(vec![(1, 3)]),
                    feature: Some(
                        SglangWireTensor::new("float32", vec![8192], Bytes::from(pixels.clone()))
                            .unwrap(),
                    ),
                    model_specific_data: BTreeMap::from([(
                        "image_grid_thw".to_string(),
                        SglangWireTensor::new("int64", vec![1, 3], Bytes::from(grid)).unwrap(),
                    )]),
                }],
                im_token_id: Some(7),
                video_token_id: None,
            }),
            ..TokenizedGenerateReqInput::default()
        };
        let (payload, aux) = SglangProtocol::encode_add(
            &SglangRequest::Generate(request),
            &SglangLayout {
                tensor_aux_frames: true,
            },
        )
        .unwrap();
        assert_eq!(
            aux.iter().map(|frame| frame.to_vec()).collect::<Vec<_>>(),
            vec![pixels, vec![0x80]],
            "the pixel frame, then the empty metadata map"
        );
        assert_eq!(payload, hex(PLUGIN_TEST_FRAMED_PAYLOAD));
        // The feature's slot is the frame reference extension (type 100) over
        // `[[8192], "float32", 1]`.
        let reference = hex("c70e649391cd2000a7666c6f6174333201");
        assert!(
            payload
                .windows(reference.len())
                .any(|window| window == reference),
            "the frame reference in the feature's slot"
        );
    }

    #[test]
    fn request_type_frames() {
        assert_eq!(
            SglangRequestType::from_frame(b"\x00"),
            Some(SglangRequestType::Add)
        );
        assert_eq!(
            SglangRequestType::from_frame(b"\x01"),
            Some(SglangRequestType::Abort)
        );
        assert_eq!(
            SglangRequestType::from_frame(b"\x02"),
            Some(SglangRequestType::Control)
        );
        assert_eq!(SglangRequestType::Control.to_frame().as_ref(), b"\x02");
        assert_eq!(SglangRequestType::from_frame(b"\x03"), None);
    }
}
