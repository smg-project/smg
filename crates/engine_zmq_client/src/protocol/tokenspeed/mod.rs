//! TokenSpeed wire protocol.
//!
//! Clean-room port of TokenSpeed's native engine-IPC messages
//! (`runtime/engine/io_struct.py`). Every data-plane message is a Python
//! `msgspec.Struct(tag=True, tag_field="_tag", kw_only=True, array_like=True)`:
//! it rides the wire as a positional msgpack array whose element 0 is the
//! class-name tag string, followed by the fields in declaration order. Field
//! order is the wire contract — append only, never reorder. Decoders here
//! validate the tag and skip trailing fields they do not model (TokenSpeed
//! appends fields over time); the encoder emits the shortest valid prefix,
//! since `msgspec` fills missing trailing fields from their defaults.
//!
//! Only the text-generation path is typed. Per TokenSpeed's `zmq_wire.py`, the
//! request-type frame (a single raw byte ahead of the payload, so the receiver
//! dispatches without decoding) is `ADD = 0x00` / `ABORT = 0x01`, and the
//! abort payload is a plain msgpack `list[str]` of request ids, not a tagged
//! struct. A scheduler of the control-wire revision also takes `CONTROL =
//! 0x02` (`(call_id, method, args)`, answered by a `ControlReplySlim` on the
//! output socket) and says so by declaring that reply in its handshake
//! ([`wire_carries_control_calls`]); SMG sends control calls to no other.
//!
//! That is the wire of the engine's Python control loop. The engine's Rust
//! control plane speaks the same messages in the engine's own published
//! schema (`tokenspeed-wire`, the `ts_wire` crate): the full request array,
//! tensors framed out from 256 bytes, a 22-column output batch, tagged
//! control structs. Which of the two a connection speaks is a fact the
//! launcher reads off the installed engine before the first frame
//! ([`WireRevision`], part of the connection's [`TokenSpeedLayout`]); the
//! [`wire`] module frames the latter with the crate's codec, this module
//! frames the former as it always has, byte for byte.

pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;
pub mod wire;

use std::collections::BTreeMap;

use bytes::Bytes;

use crate::{
    codec::{encode_msgpack, tensor::encode_with_aux_frames, OpaqueValue},
    error::Result,
    protocol::{
        control::{CONTROL_REPLY_SLIM_TAG, CONTROL_REPLY_WIRE_FIELDS},
        handshake::EngineCoreReadyResponse,
        positional::check_prefix,
        tokenspeed::{
            output::{
                decode_wire_output, SlimOutputLayout, TokenSpeedOutput, TokenSpeedWireOutput,
            },
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
        },
        EngineBatch, EngineLoad, EngineProtocol, UtilityReply,
    },
};

/// Whether the engine's handshake declared the control reply
/// (`positional_layouts["ControlReplySlim"]`): the scheduler then takes
/// control calls (`FlushCache`, profiling) on its input socket and answers
/// them on its output socket. A scheduler that declares no layout at all
/// (today's) carries generate and abort only, and the servicer says so.
pub fn wire_carries_control_calls(layouts: &BTreeMap<String, Vec<String>>) -> bool {
    layouts.contains_key(CONTROL_REPLY_SLIM_TAG)
}

/// Which revision of the scheduler's msgpack wire a connection speaks: the
/// one capability read of this wire, made by the launcher off the installed
/// engine before any scheduler starts and handed down with the connection;
/// nothing is sniffed off the frames, and an engine is never asked. The
/// default is the wire this module has always spoken, so a caller that
/// says nothing changes nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WireRevision {
    /// A scheduler of the engine's Python control loop: the request prefix,
    /// the 13 output columns plus the ones the engine declares, control
    /// calls as the `(call_id, method, args)` triple, all framed here.
    #[default]
    PythonLoop,
    /// A scheduler of the engine's Rust control plane: the engine's
    /// published schema, framed by its own crate ([`wire`]).
    RustControlPlane,
}

impl WireRevision {
    /// The revision by the launcher's word (`python_loop`,
    /// `rust_control_plane`); `None` for any other word.
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "python_loop" => Some(Self::PythonLoop),
            "rust_control_plane" => Some(Self::RustControlPlane),
            _ => None,
        }
    }

    /// The launcher's word for the revision.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PythonLoop => "python_loop",
            Self::RustControlPlane => "rust_control_plane",
        }
    }
}

/// A connection's layout on this wire: the revision the launcher read, and
/// where the engines declared the optional output columns this module knows
/// (a scheduler of the Python loop alone declares any; the Rust control
/// plane's batch carries its details at fixed columns).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenSpeedLayout {
    pub wire: WireRevision,
    pub columns: SlimOutputLayout,
}

impl TokenSpeedLayout {
    /// This layout on the given wire revision: what the caller read of the
    /// engine before the handshake, over what the handshake declared.
    pub fn with_wire(self, wire: WireRevision) -> Self {
        Self { wire, ..self }
    }
}

/// The TokenSpeed engine protocol: drives [`TokenizedGenerateReqInput`] over
/// the shared ZMQ transport and decodes [`BatchTokenIDOutSlim`](output::BatchTokenIDOutSlim) back.
pub struct TokenSpeedProtocol;

impl EngineProtocol for TokenSpeedProtocol {
    type Request = TokenizedGenerateReqInput;
    type Output = TokenSpeedOutput;
    type Layout = TokenSpeedLayout;

    fn add_frame() -> Bytes {
        TokenSpeedRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        TokenSpeedRequestType::Abort.to_frame()
    }

    /// A scheduler that declares the control reply declares its field order
    /// too (a tagged struct's tag, element 0, is not a field): a reordered
    /// or renamed reply fails here, naming the field, instead of decoding
    /// wrong and leaving the caller to wait out its deadline. A scheduler
    /// that declares nothing (today's) passes: it takes no control calls.
    fn check_layouts(ready: &EngineCoreReadyResponse) -> Result<()> {
        check_prefix(
            CONTROL_REPLY_SLIM_TAG,
            &CONTROL_REPLY_WIRE_FIELDS,
            &ready.positional_layouts,
        )
    }

    fn prompt_tokens(request: &Self::Request) -> u64 {
        request.input_ids.len() as u64
    }

    fn reported_every_step(request: &Self::Request) -> bool {
        // A request that does not stream is reported at its end (and at the
        // scheduler's forced interval), a streaming one every
        // `stream_interval` tokens (its own, else the server's, 1 unless
        // raised); only a streaming request at an interval of one is
        // reported every step.
        request.stream
            && request
                .sampling_params
                .stream_interval
                .is_none_or(|interval| interval <= 1)
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.rid
    }

    fn data_parallel_rank(request: &Self::Request) -> Option<u32> {
        // Rank routing is by ZMQ identity: the connector sends on the chosen
        // rank's socket. The sender decides the rank as the engine's own
        // dispatcher would (a disaggregated request by its room's residue, a
        // pinned one by its pin); `None` is the least-loaded selection.
        request.data_parallel_rank
    }

    fn validate(_request: &Self::Request) -> Result<()> {
        // The tokenized text path has no fields this client cannot represent.
        Ok(())
    }

    /// Encode the request with its large multimodal tensors as auxiliary
    /// frames ([`encode_with_aux_frames`]): the engine's `MsgpackDecoder`
    /// resolves a tensor's frame index against the message's frames, the
    /// payload being frame 0, as it resolves its own encoder's. A text
    /// request, or one whose tensors are all small, keeps its one frame.
    /// On the Rust control plane's wire the engine's own crate frames the
    /// request instead ([`wire::encode_add`]).
    fn encode_add(request: &Self::Request, layout: &Self::Layout) -> Result<(Vec<u8>, Vec<Bytes>)> {
        match layout.wire {
            WireRevision::PythonLoop => encode_with_aux_frames(request),
            WireRevision::RustControlPlane => wire::encode_add(request),
        }
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        // The abort payload is a positional msgpack array of request-id strings,
        // decoded by the engine's `decode_abort` (a `msgspec.msgpack.Decoder(
        // list[str])`) into abort requests. A single-element array aborts one rid.
        encode_msgpack(&[request_id])
    }

    fn encode_start_wave(_wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // TokenSpeed has no wave protocol: its scheduler never pauses a group
        // of ranks, so there is nothing to wake.
        Ok(None)
    }

    fn encode_utility(
        call_id: i64,
        method: &str,
        args: &[OpaqueValue],
    ) -> Result<Option<(Bytes, Vec<u8>)>> {
        // `[engine, CONTROL, (call_id, method, args)]`, the triple the SGLang
        // plugin takes: a scheduler of the control-wire revision builds its
        // own control request from it and answers under the call id. The
        // caller sends it only to a scheduler whose handshake declared the
        // reply ([`wire_carries_control_calls`]).
        Ok(Some((
            TokenSpeedRequestType::Control.to_frame(),
            encode_msgpack(&(call_id, method, args))?,
        )))
    }

    fn read_layout(ready: &EngineCoreReadyResponse) -> TokenSpeedLayout {
        // The finish-detail columns are read where the engine declares them
        // (`positional_layouts["BatchTokenIDOutSlim"]`), never at a pinned
        // position: an engine that declares neither is read as before. The
        // wire revision is not the handshake's to tell: the caller sets it
        // ([`TokenSpeedLayout::with_wire`]) from what the launcher read.
        TokenSpeedLayout {
            wire: WireRevision::default(),
            columns: SlimOutputLayout::declared(&ready.positional_layouts),
        }
    }

    fn decode_batch(
        frames: &[Bytes],
        layout: &TokenSpeedLayout,
    ) -> Result<EngineBatch<Self::Output>> {
        if layout.wire == WireRevision::RustControlPlane {
            return wire::decode_batch(frames);
        }
        let layout = &layout.columns;
        // Output messages are `[payload, aux...]`. The slim batch carries no
        // tensor fields today, so aux frames are unexpected but not fatal.
        if frames.len() > 1 {
            tracing::debug!(
                aux_frames = frames.len() - 1,
                "ignoring aux frames on a TokenSpeed output (BatchTokenIDOutSlim \
                 has no tensor fields)"
            );
        }
        let payload = frames.first().map(AsRef::as_ref).unwrap_or_default();
        let batch = match decode_wire_output(payload, layout)? {
            TokenSpeedWireOutput::Tokens(batch) => *batch,
            TokenSpeedWireOutput::ControlReply(reply) => {
                return Ok(EngineBatch {
                    engine_index: reply.engine_index,
                    utility: Some(UtilityReply {
                        call_id: reply.call_id,
                        outcome: Ok(reply.into_outcome()),
                    }),
                    ..EngineBatch::default()
                });
            }
        };
        let engine_index = batch.engine_index;
        // `kv_total_pages == 0` marks a pre-piggyback sender (or a snapshot
        // the engine could not take): report no load rather than fabricating
        // an empty-scheduler signal that least-loaded selection would trust.
        let load = (batch.kv_total_pages > 0).then(|| EngineLoad {
            num_running: batch.num_running,
            num_waiting: batch.num_waiting,
            kv_cache_usage: batch.kv_active_pages as f64 / batch.kv_total_pages as f64,
        });
        let outputs = batch.into_outputs()?;
        // TokenSpeed has no out-of-band finish channel: every finish is carried
        // by its own output. Reporting them here anyway keeps the batch shape
        // self-describing for consumers that only read the finished list.
        let finished_request_ids = outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .map(|output| output.request_id.clone())
            .collect();
        Ok(EngineBatch {
            engine_index,
            outputs,
            finished_request_ids,
            load,
            wave: None,
            utility: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rmpv::Value;

    use super::*;
    use crate::{
        codec::{
            decode_msgpack, decode_value,
            tensor::{
                resolve_array_bytes, TensorFields, WireArrayData, WireTensor, TENSOR_AUX_MIN_BYTES,
            },
        },
        protocol::tokenspeed::{
            multimodal::{
                mm_pad_value, TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality,
            },
            output::BatchTokenIDOutSlim,
        },
    };

    /// A multimodal item whose feature has `feature` bytes and whose one side
    /// tensor has `side` bytes.
    fn media_item(hash: u64, feature: usize, side: usize) -> TokenSpeedWireMmItem {
        TokenSpeedWireMmItem {
            modality: TokenSpeedWireModality::Image,
            hash,
            pad_value: mm_pad_value(TokenSpeedWireModality::Image, hash),
            offsets: vec![(1, 2)],
            feature: Some(WireTensor::from_raw(
                "uint8",
                vec![feature],
                vec![7; feature],
            )),
            model_specific_data: BTreeMap::from([(
                "grid".to_string(),
                WireTensor::from_raw("uint8", vec![side], vec![3; side]),
            )]),
            encode_handshake: None,
        }
    }

    /// A request with two items: a feature of the auxiliary size beside a
    /// small side tensor, then a small feature beside a side tensor of that
    /// size.
    fn media_request() -> TokenizedGenerateReqInput {
        let pad = mm_pad_value(TokenSpeedWireModality::Image, 1);
        TokenizedGenerateReqInput {
            rid: "media".into(),
            input_ids: vec![10, pad, pad, 50],
            input_ids_unpadded: Some(vec![10, 20, 30, 50]),
            multimodal_inputs: Some(TokenSpeedWireMmInputs {
                mm_items: vec![
                    media_item(1, 4 * TENSOR_AUX_MIN_BYTES, 255),
                    media_item(2, 256, TENSOR_AUX_MIN_BYTES),
                ],
                im_token_id: Some(9),
                video_token_id: None,
                ..Default::default()
            }),
            ..TokenizedGenerateReqInput::default()
        }
    }

    /// The `(dtype, shape, data)` tuple's data slot of an item's feature
    /// (slot 4) or of its one side tensor (slot 6, under its key).
    fn tensor_data(item: &Value, feature: bool) -> Value {
        let Value::Array(slots) = item else {
            panic!("an item is a positional array, got {item:?}");
        };
        let tensor = if feature {
            slots[4].clone()
        } else {
            let Value::Map(side) = &slots[6] else {
                panic!("model_specific_data is a map, got {:?}", slots[6]);
            };
            side[0].1.clone()
        };
        let Value::Array(tuple) = tensor else {
            panic!("a tensor is a tuple, got {tensor:?}");
        };
        tuple[2].clone()
    }

    /// The large tensors leave the payload as frames, in slot order, sharing
    /// the request's buffers; the small ones stay inline; the payload
    /// addresses the frames one-based and the request itself is untouched.
    #[test]
    fn large_media_tensors_ride_as_frames_the_payload_addresses() {
        let request = media_request();
        let inline = encode_msgpack(&request).unwrap();
        let (payload, aux) =
            TokenSpeedProtocol::encode_add(&request, &TokenSpeedLayout::default()).unwrap();
        assert_eq!(
            aux.iter().map(Bytes::len).collect::<Vec<_>>(),
            vec![4 * TENSOR_AUX_MIN_BYTES, TENSOR_AUX_MIN_BYTES]
        );
        assert!(payload.len() < 1024, "{} bytes", payload.len());
        assert_eq!(encode_msgpack(&request).unwrap(), inline);
        let mut pointers = Vec::new();
        request.for_each_tensor(&mut |data| {
            if let WireArrayData::RawView(bytes) = data {
                if bytes.len() >= TENSOR_AUX_MIN_BYTES {
                    pointers.push(bytes.as_ptr());
                }
            }
        });
        assert_eq!(
            pointers,
            aux.iter().map(|bytes| bytes.as_ptr()).collect::<Vec<_>>()
        );
        // The engine's view: the request array's multimodal slot, its items.
        let Value::Array(array) = decode_value(&payload).unwrap() else {
            panic!("a positional array");
        };
        let Value::Array(mm) = &array[22] else {
            panic!("multimodal_inputs at 22, got {:?}", array[22]);
        };
        let Value::Array(items) = &mm[0] else {
            panic!("mm_items at 0, got {:?}", mm[0]);
        };
        assert_eq!(tensor_data(&items[0], true), Value::from(1u64));
        assert!(matches!(tensor_data(&items[0], false), Value::Ext(3, _)));
        assert!(matches!(tensor_data(&items[1], true), Value::Ext(3, _)));
        assert_eq!(tensor_data(&items[1], false), Value::from(2u64));
        // The frames resolve to the tensors' bytes, and the text prefix of
        // the payload still decodes.
        let mut frames = vec![Bytes::from(payload.clone())];
        frames.extend(aux);
        let item = &request.multimodal_inputs.as_ref().unwrap().mm_items[1];
        assert_eq!(
            resolve_array_bytes(WireArrayData::AuxIndex(2), "grid", &frames).unwrap(),
            *item.model_specific_data["grid"].data.as_raw_view().unwrap()
        );
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded.rid, "media");
        assert_eq!(decoded.input_ids, request.input_ids);
    }

    /// Requests whose tensors are all below the auxiliary size, and text
    /// requests, are encoded as before: one frame, byte for byte.
    #[test]
    fn small_media_tensors_and_text_requests_keep_their_one_frame() {
        for size in [0, 255, 256, 4096, TENSOR_AUX_MIN_BYTES - 1] {
            let mut request = media_request();
            request.multimodal_inputs.as_mut().unwrap().mm_items =
                vec![media_item(1, size, size), media_item(2, size, 0)];
            let (payload, aux) =
                TokenSpeedProtocol::encode_add(&request, &TokenSpeedLayout::default()).unwrap();
            assert!(aux.is_empty(), "size={size}");
            assert_eq!(payload, encode_msgpack(&request).unwrap(), "size={size}");
        }
        let text = TokenizedGenerateReqInput {
            rid: "text".into(),
            input_ids: vec![1, 2, 3],
            ..TokenizedGenerateReqInput::default()
        };
        let (payload, aux) =
            TokenSpeedProtocol::encode_add(&text, &TokenSpeedLayout::default()).unwrap();
        assert!(aux.is_empty());
        assert_eq!(payload, encode_msgpack(&text).unwrap());
    }

    /// A payload that already addresses auxiliary frames (a caller's own
    /// split) is sent as it is: a second extraction would rebase its indices.
    #[test]
    fn a_payload_that_already_addresses_frames_is_sent_as_it_is() {
        let mut request = media_request();
        request.multimodal_inputs.as_mut().unwrap().mm_items[1]
            .feature
            .as_mut()
            .unwrap()
            .data = WireArrayData::AuxIndex(1);
        let (payload, aux) =
            TokenSpeedProtocol::encode_add(&request, &TokenSpeedLayout::default()).unwrap();
        assert!(aux.is_empty());
        assert_eq!(payload, encode_msgpack(&request).unwrap());
    }

    fn slim_batch() -> BatchTokenIDOutSlim {
        BatchTokenIDOutSlim {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20]],
            finished_reasons: vec![String::new(), "stop".into()],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 1],
            cached_tokens: vec![0, 0],
            output_token_logprobs_val: vec![vec![], vec![]],
            output_token_logprobs_idx: vec![vec![], vec![]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 5,
            kv_active_pages: 100,
            kv_total_pages: 400,
            finished_status_codes: Vec::new(),
            finished_messages: Vec::new(),
        }
    }

    /// Which requests the engine reports every step, for the silence count:
    /// a streaming request at an interval of one (the request's own, or the
    /// server's default when unset); one that does not stream, or streams
    /// every so many tokens, is reported at intervals. Every prompt counts
    /// as pending prefill.
    #[test]
    fn a_streaming_request_at_interval_one_is_reported_every_step() {
        use crate::protocol::tokenspeed::sampling::SamplingParams;
        let request = |stream: bool, stream_interval: Option<u32>| TokenizedGenerateReqInput {
            input_ids: vec![1, 2, 3],
            stream,
            sampling_params: SamplingParams {
                stream_interval,
                ..SamplingParams::default()
            },
            ..TokenizedGenerateReqInput::default()
        };
        assert!(TokenSpeedProtocol::reported_every_step(&request(
            true, None
        )));
        assert!(TokenSpeedProtocol::reported_every_step(&request(
            true,
            Some(1)
        )));
        assert!(!TokenSpeedProtocol::reported_every_step(&request(
            true,
            Some(8)
        )));
        assert!(!TokenSpeedProtocol::reported_every_step(&request(
            false, None
        )));
        assert!(!TokenSpeedProtocol::reported_every_step(&request(
            false,
            Some(1)
        )));
        assert_eq!(TokenSpeedProtocol::prompt_tokens(&request(false, None)), 3);
    }

    #[test]
    fn decode_batch_maps_outputs_and_finished_ids() {
        let frames = vec![Bytes::from(encode_msgpack(&slim_batch()).unwrap())];
        let decoded = TokenSpeedProtocol::decode_batch(&frames, &Default::default()).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.finished_request_ids, vec!["b".to_string()]);
        // The batch names its producing DP rank; the connector routes in-flight
        // release and scoring by it.
        assert_eq!(decoded.engine_index, 1);
        // The piggybacked snapshot surfaces as the engine-neutral load signal.
        assert_eq!(
            decoded.load,
            Some(EngineLoad {
                num_running: 2,
                num_waiting: 5,
                kv_cache_usage: 0.25,
            })
        );
    }

    #[test]
    fn decode_batch_reports_no_load_without_a_snapshot() {
        // kv_total_pages == 0 marks a pre-piggyback sender (or a snapshot the
        // engine could not take): no load, not a fabricated empty scheduler.
        let batch = BatchTokenIDOutSlim {
            kv_total_pages: 0,
            ..slim_batch()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        let decoded = TokenSpeedProtocol::decode_batch(&frames, &Default::default()).unwrap();
        assert!(decoded.load.is_none());
    }

    #[test]
    fn decode_batch_tolerates_aux_frames() {
        // Slim outputs are single-frame in practice, but the decoder must not
        // choke if the engine ever sends `[payload, aux...]`.
        let frames = vec![
            Bytes::from(encode_msgpack(&slim_batch()).unwrap()),
            Bytes::from_static(b"opaque-aux-frame"),
        ];
        let decoded = TokenSpeedProtocol::decode_batch(&frames, &Default::default()).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
    }

    #[test]
    fn request_type_frames_match_wire_contract() {
        assert_eq!(TokenSpeedProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(TokenSpeedProtocol::abort_frame().as_ref(), b"\x01");
        assert_eq!(
            TokenSpeedRequestType::from_frame(b"\x02"),
            Some(TokenSpeedRequestType::Control)
        );
        assert_eq!(TokenSpeedRequestType::Control.to_frame().as_ref(), b"\x02");
        // 3 is the encode worker's request type (its own change); 9 is unclaimed.
        assert_eq!(TokenSpeedRequestType::from_frame(b"\x09"), None);
    }

    /// `[7, "flush_cache", [1.5]]` as the scheduler's msgspec decoder of the
    /// control triple takes it.
    const FLUSH_CALL_VECTOR: &str = "9307ab666c7573685f636163686591cb3ff8000000000000";
    /// `["ControlReplySlim", 7, True, None, 0]` from the scheduler's own
    /// struct: an accepted flush on rank 0.
    const CONTROL_REPLY_ACCEPTED_VECTOR: &str = "95b0436f6e74726f6c5265706c79536c696d07c3c000";
    /// `["ControlReplySlim", 9, False, "no profiler running", 1]`: a refused
    /// profiler stop on rank 1.
    const CONTROL_REPLY_REFUSED_VECTOR: &str =
        "95b0436f6e74726f6c5265706c79536c696d09c2b36e6f2070726f66696c65722072756e6e696e6701";

    /// A control call rides the `CONTROL` frame as the positional triple the
    /// SGLang plugin takes; the reply comes back on the output socket as a
    /// utility reply under the call id; and the handshake's layout
    /// declaration is what tells a scheduler that takes control calls from
    /// one that does not.
    #[test]
    fn control_calls_ride_the_control_frame_and_replies_resolve_by_call_id() {
        use crate::protocol::{control::ControlReplySlim, positional::vector_bytes};

        let (frame, payload) =
            TokenSpeedProtocol::encode_utility(7, "flush_cache", &[OpaqueValue::F64(1.5)])
                .unwrap()
                .expect("the control-wire revision has control calls");
        assert_eq!(frame.as_ref(), b"\x02");
        let decoded: (i64, String, Vec<OpaqueValue>) = decode_msgpack(&payload).unwrap();
        assert_eq!(
            decoded,
            (7, "flush_cache".to_string(), vec![OpaqueValue::F64(1.5)])
        );
        // Byte for byte what the scheduler's msgspec decoder of the triple
        // takes, and what its `ControlReplySlim` struct emits for an accepted
        // flush and for a refused profiler stop: the captured vectors.
        assert_eq!(payload, vector_bytes(FLUSH_CALL_VECTOR));
        let accepted = TokenSpeedProtocol::decode_batch(
            &[Bytes::from(vector_bytes(CONTROL_REPLY_ACCEPTED_VECTOR))],
            &TokenSpeedLayout::default(),
        )
        .unwrap();
        assert_eq!(accepted.engine_index, 0);
        let utility = accepted.utility.expect("a utility reply");
        assert_eq!(utility.call_id, 7);
        assert_eq!(
            utility.outcome.unwrap().as_map().unwrap(),
            &[
                (OpaqueValue::from("success"), OpaqueValue::Boolean(true)),
                (OpaqueValue::from("message"), OpaqueValue::Nil),
            ]
        );
        let refused: ControlReplySlim =
            decode_msgpack(&vector_bytes(CONTROL_REPLY_REFUSED_VECTOR)).unwrap();
        assert_eq!(
            refused,
            ControlReplySlim {
                call_id: 9,
                success: false,
                message: Some("no profiler running".into()),
                engine_index: 1,
            }
        );
        let accepted_reply = ControlReplySlim {
            call_id: 7,
            success: true,
            message: None,
            engine_index: 0,
        };
        assert_eq!(
            encode_msgpack(&accepted_reply).unwrap(),
            vector_bytes(CONTROL_REPLY_ACCEPTED_VECTOR)
        );

        let reply = ControlReplySlim {
            call_id: 9,
            success: false,
            message: Some("busy".into()),
            engine_index: 1,
        };
        let frames = vec![Bytes::from(encode_msgpack(&reply).unwrap())];
        let batch =
            TokenSpeedProtocol::decode_batch(&frames, &TokenSpeedLayout::default()).unwrap();
        assert!(batch.outputs.is_empty() && batch.load.is_none());
        assert_eq!(batch.engine_index, 1);
        let utility = batch.utility.expect("a utility reply");
        assert_eq!(utility.call_id, 9);
        let map = utility.outcome.unwrap();
        assert_eq!(
            map.as_map().unwrap()[0],
            (OpaqueValue::from("success"), OpaqueValue::Boolean(false))
        );
        // The token batch still decodes through the same dispatch.
        let frames = vec![Bytes::from(encode_msgpack(&slim_batch()).unwrap())];
        assert_eq!(
            TokenSpeedProtocol::decode_batch(&frames, &TokenSpeedLayout::default())
                .unwrap()
                .outputs
                .len(),
            2
        );

        let mut layouts = BTreeMap::new();
        assert!(!wire_carries_control_calls(&layouts));
        layouts.insert(
            "ControlReplySlim".to_string(),
            ["call_id", "success", "message", "engine_index"]
                .iter()
                .map(ToString::to_string)
                .collect(),
        );
        assert!(wire_carries_control_calls(&layouts));
    }

    #[test]
    fn encode_abort_frames_rid_as_positional_array() {
        // The abort payload is a positional msgpack array of request-id strings,
        // matching the engine's `decode_abort` (`Decoder(list[str])`).
        let payload = TokenSpeedProtocol::encode_abort("req-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["req-1".to_string()]);
    }

    /// The declared control reply pins its field order: the scheduler's own
    /// order passes, a reordered one fails at connect naming the field, and
    /// a scheduler that declares nothing passes (it takes no control calls).
    #[test]
    fn check_layouts_pins_the_control_replys_field_order() {
        use crate::mock_engine::default_ready_response;

        let ready = |fields: &[&str]| {
            let mut ready = default_ready_response();
            ready.positional_layouts.insert(
                CONTROL_REPLY_SLIM_TAG.to_string(),
                fields.iter().map(ToString::to_string).collect(),
            );
            ready
        };
        assert!(TokenSpeedProtocol::check_layouts(&default_ready_response()).is_ok());
        assert!(TokenSpeedProtocol::check_layouts(&ready(&CONTROL_REPLY_WIRE_FIELDS)).is_ok());
        let error = TokenSpeedProtocol::check_layouts(&ready(&[
            "call_id",
            "message",
            "success",
            "engine_index",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("`ControlReplySlim` field 1 is `success` on this side and `message`"),
            "{error}"
        );
    }

    /// The connection's layout comes off the ready response by name, and the
    /// batch decoder reads the finish details where it points; an engine that
    /// declares nothing is read as before.
    #[test]
    fn the_layout_is_read_off_the_handshake_and_drives_the_batch_decode() {
        use crate::protocol::tokenspeed::output::{
            BASE_COLUMN_NAMES, FINISHED_MESSAGES, FINISHED_STATUS_CODES,
        };
        let mut ready = crate::mock_engine::default_ready_response();
        assert_eq!(
            TokenSpeedProtocol::read_layout(&ready),
            TokenSpeedLayout::default()
        );
        let mut names: Vec<String> = BASE_COLUMN_NAMES.iter().map(|n| n.to_string()).collect();
        names.push(FINISHED_STATUS_CODES.to_string());
        names.push(FINISHED_MESSAGES.to_string());
        ready
            .positional_layouts
            .insert(output::BATCH_TOKEN_ID_OUT_SLIM_TAG.to_string(), names);
        let layout = TokenSpeedProtocol::read_layout(&ready);
        assert_eq!(
            layout,
            TokenSpeedLayout {
                wire: WireRevision::PythonLoop,
                columns: SlimOutputLayout {
                    finish_status: Some(13),
                    finish_message: Some(14),
                },
            }
        );

        let batch = BatchTokenIDOutSlim {
            rids: vec!["edge".into()],
            output_ids: vec![vec![]],
            finished_reasons: vec!["abort".into()],
            prompt_tokens: vec![8191],
            completion_tokens: vec![0],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            finished_status_codes: vec![Some(400)],
            finished_messages: vec![Some("Invalid request: too long".into())],
            ..Default::default()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        let read = TokenSpeedProtocol::decode_batch(&frames, &layout).unwrap();
        assert_eq!(read.outputs[0].finish_status, Some(400));
        assert_eq!(
            read.outputs[0].finish_message.as_deref(),
            Some("Invalid request: too long")
        );
        assert_eq!(read.finished_request_ids, vec!["edge".to_string()]);
        let unread =
            TokenSpeedProtocol::decode_batch(&frames, &TokenSpeedLayout::default()).unwrap();
        assert_eq!(unread.outputs[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(unread.outputs[0].finish_status, None);
        assert_eq!(unread.outputs[0].finish_message, None);
    }
}
