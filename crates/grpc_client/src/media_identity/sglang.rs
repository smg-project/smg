//! The SGLang wire's identity: its batch message under the rule, and the
//! check the two sides of a disaggregated request share (a version, the
//! processor's key, the request's digest), early, before either scheduler
//! waits for its KV peer. A failed check leaves the original reference
//! request untouched so its worker can process it.
use prost::{
    bytes::{buf::UninitSlice, BufMut},
    Message,
};
use sha2::{Digest, Sha256};

use super::{Batch, Kept};
use crate::sglang_proto::{GenerateRequest, MediaIdentity, MultimodalInputs};

/// The worker label both sides advertise their processor contract under.
pub const LABEL: &str = "sglang_media_identity";
/// The contract's version; an identity of another is declined.
pub const VERSION: u32 = 1;
const MAX_IDENTITY_BYTES: usize = 4 * 1024 * 1024;

impl Batch for MultimodalInputs {
    /// The batch's one feature hash.
    fn hashed(&self) -> bool {
        self.feature_hash.is_some()
    }

    /// Every tensor on this wire is inline bytes.
    fn readable(&self, key: &str) -> bool {
        self.model_specific_tensors.contains_key(key)
    }

    /// This wire slices no tensor by a sizes tensor.
    fn sizes_key(&self, _key: &str) -> Option<&str> {
        None
    }

    fn without_pixels(&self, kept: &Kept) -> Self {
        Self {
            modalities: self.modalities.clone(),
            model_specific_tensors: self
                .model_specific_tensors
                .iter()
                .filter(|(key, _)| kept.has(key))
                .map(|(key, tensor)| (key.clone(), tensor.clone()))
                .collect(),
            im_token_id: self.im_token_id,
            mm_placeholders: self.mm_placeholders.clone(),
            feature_hash: self.feature_hash,
            ..Default::default()
        }
    }
}

/// Recognize reference-only inputs without raw media or processor tensors.
pub fn has_media_refs(req: &GenerateRequest) -> bool {
    req.extra_mm_inputs.is_empty()
        && req.mm_inputs.as_ref().is_some_and(|mm| {
            mm.pixel_values.is_none()
                && (!mm.image_urls.is_empty() || !mm.video_urls.is_empty())
                && mm.audio_urls.is_empty()
                && mm.image_data.is_empty()
                && mm.video_data.is_empty()
                && mm.audio_data.is_empty()
                && mm.model_specific_tensors.is_empty()
        })
}

/// Bind to the original prompt, ordered media references and KV rendezvous.
/// Ignore side-specific routing/sampling fields. Length prefixes avoid ambiguity.
pub fn request_digest(req: &GenerateRequest) -> Vec<u8> {
    let mut hash = Sha256::new();
    /// Hash a length-prefixed field so adjacent values cannot produce ambiguous bindings.
    fn field(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    field(&mut hash, req.request_id.as_bytes());
    if let Some(tokens) = &req.tokenized {
        field(&mut hash, &tokens.encode_to_vec());
    }
    if let Some(mm) = &req.mm_inputs {
        // Reference requests have no tensor map, so serialization is canonical.
        message_field(&mut hash, mm.as_ref());
    }
    if let Some(pd) = &req.disaggregated_params {
        field(&mut hash, &pd.encode_to_vec());
    }
    hash.finalize().to_vec()
}

/// Hash the same length-prefixed protobuf bytes without allocating a media-sized buffer.
fn message_field(hash: &mut Sha256, message: &impl Message) {
    let len = message.encoded_len();
    hash.update((len as u64).to_le_bytes());
    // Keep the contiguous encoder for small metadata; streaming pays off on media payloads.
    if len < 64 * 1024 {
        let mut encoded = Vec::with_capacity(len);
        message.encode_raw(&mut encoded);
        hash.update(encoded);
        return;
    }
    let mut output = DigestBuffer {
        hash,
        scratch: [0; 1024],
        buffered: 0,
        written: 0,
    };
    message.encode_raw(&mut output);
    output.flush();
}

/// Buffer small protobuf fields while hashing large media references directly.
struct DigestBuffer<'a> {
    hash: &'a mut Sha256,
    scratch: [u8; 1024],
    buffered: usize,
    written: usize,
}

impl DigestBuffer<'_> {
    /// Hash pending bytes before reusing the scratch buffer or finishing the message.
    fn flush(&mut self) {
        self.hash.update(&self.scratch[..self.buffered]);
        self.buffered = 0;
    }
}

// SAFETY: chunk_mut exposes initialized scratch space. Advancing consumes only that
// space, and the remaining capacity decreases by every byte accepted by the sink.
#[expect(
    unsafe_code,
    reason = "BufMut requires an unsafe implementation; this sink only exposes initialized, bounds-checked scratch bytes"
)]
unsafe impl BufMut for DigestBuffer<'_> {
    /// Report the sink's total remaining capacity across future flushes.
    fn remaining_mut(&self) -> usize {
        usize::MAX - self.written
    }

    /// Commit bytes from the current scratch chunk, flushing it once full.
    unsafe fn advance_mut(&mut self, count: usize) {
        assert!(count <= self.scratch.len() - self.buffered);
        assert!(count <= self.remaining_mut());
        self.buffered += count;
        self.written += count;
        if self.buffered == self.scratch.len() {
            self.flush();
        }
    }

    /// Expose the writable tail of the initialized scratch buffer.
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        let len = (self.scratch.len() - self.buffered).min(self.remaining_mut());
        UninitSlice::new(&mut self.scratch[self.buffered..self.buffered + len])
    }

    /// Keep short fields in order and hash large slices directly after pending bytes.
    fn put_slice(&mut self, bytes: &[u8]) {
        assert!(bytes.len() <= self.remaining_mut());
        self.written += bytes.len();
        if bytes.len() >= self.scratch.len() {
            self.flush();
            self.hash.update(bytes);
        } else {
            if bytes.len() > self.scratch.len() - self.buffered {
                self.flush();
            }
            self.scratch[self.buffered..self.buffered + bytes.len()].copy_from_slice(bytes);
            self.buffered += bytes.len();
            if self.buffered == self.scratch.len() {
                self.flush();
            }
        }
    }
}

/// Limit reuse to tokenized, single-choice PD requests with immutable media references.
pub fn reusable_request(req: &GenerateRequest) -> bool {
    has_media_refs(req)
        && req
            .mm_inputs
            .as_ref()
            .is_some_and(|mm| mm.image_urls.is_empty() || mm.video_urls.is_empty())
        && req
            .tokenized
            .as_ref()
            .is_some_and(|t| !t.input_ids.is_empty())
        && req.sampling_params.as_ref().is_none_or(|p| p.n <= 1)
        && req
            .disaggregated_params
            .as_ref()
            .is_some_and(|p| !p.bootstrap_host.is_empty() && p.bootstrap_port > 0)
        && req.input_embeds.is_empty()
}

/// Conservative structural validation, shared by router and receiving worker.
pub fn validate(identity: &MediaIdentity, req: &GenerateRequest, key: &str) -> bool {
    if key.is_empty()
        || identity.version != VERSION
        || identity.processor_key != key
        || identity.request_digest != request_digest(req)
        || !reusable_request(req)
        || identity.input_ids.is_empty()
        || identity.encoded_len() > MAX_IDENTITY_BYTES
    {
        return false;
    }
    let Some(mm) = &identity.mm_inputs else {
        return false;
    };
    let Some(token) = mm.im_token_id else {
        return false;
    };
    if mm.pixel_values.is_some()
        || mm.feature_hash.is_none()
        || !mm.image_urls.is_empty()
        || !mm.video_urls.is_empty()
        || !mm.audio_urls.is_empty()
        || !mm.image_data.is_empty()
        || !mm.video_data.is_empty()
        || !mm.audio_data.is_empty()
        || mm.mm_placeholders.is_empty()
        || mm.modalities.is_empty()
        || !(mm.modalities.iter().all(|m| m == "image")
            || mm.modalities.iter().all(|m| m == "video"))
    {
        return false;
    }
    let mut previous_end = 0;
    for range in &mm.mm_placeholders {
        let start = range.offset as usize;
        let Some(end) = start.checked_add(range.length as usize) else {
            return false;
        };
        if range.length == 0
            || start < previous_end
            || end > identity.input_ids.len()
            || identity.input_ids[start..end].iter().any(|&id| id != token)
        {
            return false;
        }
        previous_end = end;
    }
    let grid_key = if mm.modalities[0] == "video" {
        "video_grid_thw"
    } else {
        "image_grid_thw"
    };
    let Some(grid) = mm.model_specific_tensors.get(grid_key) else {
        return false;
    };
    if grid.dtype != "int64" || grid.shape.len() != 2 || grid.shape[0] == 0 || grid.shape[1] != 3 {
        return false;
    }
    // Validate all small position/grid tensors before the engine sees them.
    mm.model_specific_tensors.values().all(|t| {
        let width = match t.dtype.as_str() {
            "int64" => 8,
            "float32" | "uint32" => 4,
            "float16" | "bfloat16" => 2,
            "uint8" => 1,
            _ => return false,
        };
        t.shape
            .iter()
            .try_fold(width, |n: usize, &d| n.checked_mul(d as usize))
            == Some(t.data.len())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        media_identity::identity,
        sglang_proto::{
            DisaggregatedParams, PlaceholderRange, SamplingParams, TensorData, TokenizedInput,
        },
    };

    /// Check both encoding paths against the previous length-prefixed digest.
    #[test]
    fn media_digest_matches_length_prefixed_protobuf() {
        // Small, large and many short references exercise both paths and buffer flushes.
        for mm in [
            MultimodalInputs {
                image_urls: vec!["https://example.test/image.png".into()],
                ..Default::default()
            },
            MultimodalInputs {
                image_urls: vec!["A".repeat(64 * 1024)],
                ..Default::default()
            },
            MultimodalInputs {
                image_urls: (0..2048).map(|i| "x".repeat(i % 1000)).collect(),
                ..Default::default()
            },
            // Prost writes raw bytes through BufMut::put, which fills and advances chunks.
            MultimodalInputs {
                image_data: vec![vec![7; 70 * 1024], vec![9; 3]],
                pixel_values: Some(TensorData {
                    data: vec![1; 70 * 1024].into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ] {
            let bytes = mm.encode_to_vec();
            let mut expected = Sha256::new();
            expected.update((bytes.len() as u64).to_le_bytes());
            expected.update(bytes);
            let mut actual = Sha256::new();
            message_field(&mut actual, &mm);
            assert_eq!(actual.finalize(), expected.finalize());
        }
    }

    #[test]
    /// Mixed references must fall back to processing on both PD sides.
    fn mixed_media_declines_identity_reuse_but_retains_references() {
        let (mut req, mut identity) = fixture();
        req.mm_inputs
            .as_mut()
            .unwrap()
            .video_urls
            .push("https://example.test/v.mp4".into());
        assert!(has_media_refs(&req));
        assert!(!reusable_request(&req));
        identity.request_digest = request_digest(&req);
        assert!(!validate(&identity, &req, &identity.processor_key));
    }

    /// Construct a minimal bound image identity with valid placeholders and grid bytes.
    fn fixture() -> (GenerateRequest, MediaIdentity) {
        let req = GenerateRequest {
            request_id: "a".into(),
            tokenized: Some(TokenizedInput {
                input_ids: vec![1, 7, 2],
                original_text: "image".into(),
            }),
            mm_inputs: Some(Box::new(MultimodalInputs {
                image_urls: vec!["https://example.test/a.png".into()],
                ..Default::default()
            })),
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: "p".into(),
                bootstrap_port: 8998,
                bootstrap_room: 1,
            }),
            ..Default::default()
        };
        let mm = MultimodalInputs {
            feature_hash: Some(u64::MAX),
            im_token_id: Some(7),
            modalities: vec!["image".into()],
            mm_placeholders: vec![PlaceholderRange {
                offset: 1,
                length: 2,
            }],
            model_specific_tensors: [(
                "image_grid_thw".into(),
                TensorData {
                    dtype: "int64".into(),
                    shape: vec![1, 3],
                    data: [1i64, 2, 2]
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect::<Vec<_>>()
                        .into(),
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let identity = MediaIdentity {
            version: VERSION,
            processor_key: "key".into(),
            request_digest: request_digest(&req),
            input_ids: vec![1, 7, 7, 2],
            mm_inputs: Some(mm),
        };
        (req, identity)
    }

    /// The processed batch's identity keeps the grid alone of its tensors,
    /// with the placeholders, the token id, the modalities and the feature
    /// hash, and passes the wire's check; a batch without the feature hash
    /// has none.
    #[test]
    fn the_identity_keeps_the_grid_alone_and_passes_the_check() {
        let (req, bound) = fixture();
        let mut processed = bound.mm_inputs.clone().unwrap();
        processed.pixel_values = Some(TensorData::default());
        processed
            .model_specific_tensors
            .insert("aspect_ratios".into(), TensorData::default());
        let stripped = identity(&processed).expect("an identity");
        assert!(stripped.pixel_values.is_none());
        assert_eq!(
            stripped.model_specific_tensors.keys().collect::<Vec<_>>(),
            ["image_grid_thw"]
        );
        assert_eq!(stripped.mm_placeholders, processed.mm_placeholders);
        assert_eq!(stripped.im_token_id, Some(7));
        assert_eq!(stripped.modalities, ["image"]);
        assert_eq!(stripped.feature_hash, Some(u64::MAX));
        let checked = MediaIdentity {
            mm_inputs: Some(stripped),
            ..bound
        };
        assert!(validate(&checked, &req, "key"));

        processed.feature_hash = None;
        assert!(identity(&processed).is_none());
    }

    #[test]
    /// Reject altered bindings and incomplete metadata without consuming fallback references.
    fn identity_is_bound_to_request_room_processor_and_complete_metadata() {
        let (req, identity) = fixture();
        assert!(validate(&identity, &req, "key"));
        assert!(!validate(&identity, &req, "other"));
        for case in 0..8 {
            let mut bad = identity.clone();
            match case {
                0 => bad.version += 1,
                1 => bad.request_digest.clear(),
                2 => bad.mm_inputs.as_mut().unwrap().feature_hash = None,
                3 => bad
                    .mm_inputs
                    .as_mut()
                    .unwrap()
                    .model_specific_tensors
                    .clear(),
                4 => bad.mm_inputs.as_mut().unwrap().mm_placeholders[0].offset = u32::MAX,
                5 => bad.input_ids[1] = 8,
                6 => bad.mm_inputs.as_mut().unwrap().pixel_values = Some(TensorData::default()),
                _ => {
                    bad.mm_inputs
                        .as_mut()
                        .unwrap()
                        .model_specific_tensors
                        .get_mut("image_grid_thw")
                        .unwrap()
                        .data = vec![0].into();
                }
            }
            assert!(!validate(&bad, &req, "key"), "case {case}");
        }
        let mut changed = req.clone();
        changed
            .disaggregated_params
            .as_mut()
            .unwrap()
            .bootstrap_room += 1;
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.request_id.push('b');
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.mm_inputs.as_mut().unwrap().image_urls[0].push('b');
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.sampling_params = Some(SamplingParams {
            n: 2,
            ..Default::default()
        });
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.disaggregated_params = None;
        assert!(!validate(&identity, &changed, "key"));
        assert_eq!(
            req,
            fixture().0,
            "validation never consumes fallback inputs"
        );
    }
}
