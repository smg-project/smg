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
//! struct.

pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;

use bytes::Bytes;

use crate::{
    codec::{decode_msgpack, encode_msgpack, tensor::encode_with_aux_frames},
    error::Result,
    protocol::{
        tokenspeed::{
            output::{BatchTokenIDOutSlim, TokenSpeedOutput},
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
        },
        EngineBatch, EngineLoad, EngineProtocol,
    },
};

/// The TokenSpeed engine protocol: drives [`TokenizedGenerateReqInput`] over
/// the shared ZMQ transport and decodes [`BatchTokenIDOutSlim`] back.
pub struct TokenSpeedProtocol;

impl EngineProtocol for TokenSpeedProtocol {
    type Request = TokenizedGenerateReqInput;
    type Output = TokenSpeedOutput;

    fn add_frame() -> Bytes {
        TokenSpeedRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        TokenSpeedRequestType::Abort.to_frame()
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
    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        encode_with_aux_frames(request)
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

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
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
        let batch: BatchTokenIDOutSlim = decode_msgpack(payload)?;
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
            decode_value,
            tensor::{
                resolve_array_bytes, TensorFields, WireArrayData, WireTensor, TENSOR_AUX_MIN_BYTES,
            },
        },
        protocol::tokenspeed::multimodal::{
            mm_pad_value, TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality,
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
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request).unwrap();
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
            let (payload, aux) = TokenSpeedProtocol::encode_add(&request).unwrap();
            assert!(aux.is_empty(), "size={size}");
            assert_eq!(payload, encode_msgpack(&request).unwrap(), "size={size}");
        }
        let text = TokenizedGenerateReqInput {
            rid: "text".into(),
            input_ids: vec![1, 2, 3],
            ..TokenizedGenerateReqInput::default()
        };
        let (payload, aux) = TokenSpeedProtocol::encode_add(&text).unwrap();
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
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request).unwrap();
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
            ..Default::default()
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
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
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
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
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
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
    }

    #[test]
    fn request_type_frames_match_wire_contract() {
        assert_eq!(TokenSpeedProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(TokenSpeedProtocol::abort_frame().as_ref(), b"\x01");
    }

    #[test]
    fn encode_abort_frames_rid_as_positional_array() {
        // The abort payload is a positional msgpack array of request-id strings,
        // matching the engine's `decode_abort` (`Decoder(list[str])`).
        let payload = TokenSpeedProtocol::encode_abort("req-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["req-1".to_string()]);
    }
}
