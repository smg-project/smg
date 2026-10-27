//! Opt-in CPU identity precomputation before entering SGLang's scheduler.

use std::{num::NonZeroUsize, sync::Arc, time::Instant};

use engine_zmq_client::protocol::sglang::multimodal::SglangWireMmInputs;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tonic::Status;

/// Bounded hashing on Tokio's blocking workers, never its RPC/IO threads.
///
/// This implements SGLang's CPU tensor identity: SHA256 of contiguous raw
/// feature bytes, truncated to the first eight bytes as a big-endian integer.
/// The caller must retain scheduler hashing for engines/configurations using
/// another algorithm (in particular GPU hash buffers and skip-hash mode).
/// Padding remains engine-defined; feature bytes and dtype are never changed.
pub struct SglangCpuMmHasher {
    slots: Arc<Semaphore>,
}

impl SglangCpuMmHasher {
    /// Bound simultaneous CPU hash work independently of the asynchronous RPC runtime.
    pub fn new(concurrency: NonZeroUsize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(concurrency.get())),
        }
    }

    /// Precompute a proto feature identity on the same bounded blocking pool.
    pub async fn precompute_proto(
        &self,
        mm: &mut smg_grpc_client::sglang_proto::MultimodalInputs,
        request_id: &str,
    ) -> Result<(), Status> {
        if mm.feature_hash.is_some() {
            return Ok(());
        }
        let Some(feature) = &mm.pixel_values else {
            return Ok(());
        };
        let layout =
            smg_grpc_client::sglang_deepseek_media::layout(mm).map_err(Status::invalid_argument)?;
        let stride = if layout.is_some() {
            smg_grpc_client::sglang_deepseek_media::patch_stride(feature)
                .map_err(Status::invalid_argument)?
        } else {
            0
        };
        let bytes = feature.data.clone();
        let permit = Arc::clone(&self.slots)
            .acquire_owned()
            .await
            .map_err(|_| Status::internal("multimodal hash pool closed"))?;
        let request_id = request_id.to_owned();
        let (hash, item_hashes) = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let started = Instant::now();
            let hash = cpu_feature_hash(&bytes);
            let item_hashes = layout.map(|items| items.iter().map(|item| cpu_feature_hash(&bytes[item.patches.start * stride..item.patches.end * stride]) as i64).collect::<Vec<_>>());
            tracing::debug!(target: "smg_request_timing", request_id, hash_ms = started.elapsed().as_secs_f64()*1000.0,
                "sglang_identity_hash_timing");
            (hash, item_hashes)
        }).await.map_err(|e| Status::internal(format!("media identity hash: {e}")))?;
        mm.feature_hash = Some(hash);
        if let Some(hashes) = item_hashes {
            mm.model_specific_tensors.insert(
                smg_grpc_client::sglang_deepseek_media::HASHES.into(),
                smg_grpc_client::sglang_proto::TensorData {
                    dtype: "int64".into(),
                    shape: vec![hashes.len() as u32],
                    data: hashes
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect::<Vec<_>>()
                        .into(),
                },
            );
        }
        Ok(())
    }

    /// Fill missing large-feature hashes without copying storage or overriding existing hashes.
    pub(crate) async fn precompute(
        &self,
        mm: &mut SglangWireMmInputs,
        request_id: &str,
    ) -> Result<(), Status> {
        let features: Vec<_> = mm
            .mm_items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                if item.hash.is_some() {
                    return None;
                }
                let bytes = item.feature.as_ref()?.data.as_raw_view()?;
                (bytes.len() >= 32 * 1024).then(|| (index, bytes.clone()))
            })
            .collect();
        if features.is_empty() {
            return Ok(());
        }
        let queued = Instant::now();
        let permit = Arc::clone(&self.slots)
            .acquire_owned()
            .await
            .map_err(|_| Status::internal("multimodal hash pool closed"))?;
        let span = tracing::Span::current();
        let request_id = request_id.to_owned();
        let hashes = tokio::task::spawn_blocking(move || {
            // The task owns the permit even if its awaiting RPC is cancelled.
            let _permit = permit;
            let _entered = span.enter();
            let queue_ms = queued.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let hashes: Vec<_> = features
                .iter()
                .map(|(index, bytes)| (*index, cpu_feature_hash(bytes)))
                .collect();
            tracing::debug!(target: "smg_request_timing", request_id = %request_id, hash_queue_ms = queue_ms,
                hash_ms = started.elapsed().as_secs_f64() * 1000.0,
                items = hashes.len(), "sglang_mm_hash_timing");
            hashes
        })
        .await
        .map_err(|error| Status::internal(format!("multimodal hash task: {error}")))?;
        for (index, hash) in hashes {
            mm.mm_items[index].hash = Some(hash);
        }
        Ok(())
    }
}

/// Match the engine CPU hash: the first eight SHA256 bytes interpreted big-endian.
fn cpu_feature_hash(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Check the exact digest truncation and byte order against a known SHA256 value.
    fn sha256_cpu_identity_uses_big_endian_prefix() {
        // Python: int.from_bytes(hashlib.sha256(b"abc").digest()[:8], "big")
        assert_eq!(cpu_feature_hash(b"abc"), 0xba78_16bf_8f01_cfea);
        assert_eq!(cpu_feature_hash(b""), 0xe3b0_c442_98fc_1c14);
    }
}

#[cfg(test)]
mod transport_tests {
    use std::{collections::BTreeMap, num::NonZeroUsize};

    use bytes::Bytes;
    use engine_zmq_client::protocol::sglang::multimodal::{
        SglangWireMmInputs, SglangWireMmItem, SglangWireModality, SglangWireTensor,
    };

    use super::SglangCpuMmHasher;

    #[tokio::test]
    /// Verify hashing preserves tensor storage and leaves existing identities untouched.
    async fn bounded_prehash_preserves_feature_storage_and_existing_identity() {
        let bytes = Bytes::from(vec![0_u8; 65536]);
        let tensor = SglangWireTensor::new("bfloat16", vec![32768], bytes.clone()).unwrap();
        let item = SglangWireMmItem {
            modality: SglangWireModality::Image,
            hash: None,
            offsets: None,
            feature: Some(tensor),
            model_specific_data: BTreeMap::new(),
        };
        let mut existing = item.clone();
        existing.hash = Some(u64::MAX);
        let mut mm = SglangWireMmInputs {
            mm_items: vec![item, existing],
            im_token_id: Some(1),
            video_token_id: None,
        };
        let hasher = SglangCpuMmHasher::new(NonZeroUsize::new(1).unwrap());
        let occupied = hasher.slots.acquire().await.unwrap();
        assert!(tokio::time::timeout(
            std::time::Duration::from_millis(10),
            hasher.precompute(&mut mm, "hash-test")
        )
        .await
        .is_err());
        assert!(mm.mm_items[0].hash.is_none());
        drop(occupied);
        hasher.precompute(&mut mm, "hash-test").await.unwrap();
        // Python hashlib.sha256(bytes(65536)), first eight bytes, big-endian.
        assert_eq!(mm.mm_items[0].hash, Some(0xde2f_2560_64a0_af79));
        assert_eq!(mm.mm_items[1].hash, Some(u64::MAX));
        let actual = mm.mm_items[0].feature.as_ref().unwrap();
        assert_eq!(actual.dtype, "bfloat16");
        assert_eq!(actual.data.as_raw_view().unwrap().as_ptr(), bytes.as_ptr());
        assert_eq!(actual.data.as_raw_view().unwrap(), &bytes);
    }
}
