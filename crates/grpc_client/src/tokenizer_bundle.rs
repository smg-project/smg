//! Tokenizer bundle streaming, validation, and extraction.
//!
//! End-to-end pipeline for fetching tokenizer artifacts from gRPC workers:
//! stream chunks → validate SHA-256 → validate zip safety → extract to tempdir.

use std::{
    future::Future,
    io::{Cursor, Read, Seek},
    path::{Component, Path},
    time::Duration,
};

use futures::{Stream, StreamExt};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tracing::{debug, warn};
use zip::ZipArchive;

// ── Constants ────────────────────────────────────────────────────────────────

pub const MAX_ZIP_ENTRIES: usize = 50;
pub const MAX_UNCOMPRESSED_SIZE: u64 = 500 * 1024 * 1024;
pub const MAX_STREAM_BUNDLE_SIZE: usize = 200 * 1024 * 1024;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct StreamBundle {
    pub sha256: String,
    pub compressed_data: Vec<u8>,
}

/// Temporary extracted bundle directory with explicit cleanup support.
pub struct ExtractedArchiveDir {
    temp_dir: TempDir,
}

impl ExtractedArchiveDir {
    pub fn path(&self) -> &Path {
        self.temp_dir.path()
    }

    pub fn cleanup(self) -> Result<(), String> {
        let path = self.temp_dir.path().to_string_lossy().into_owned();
        self.temp_dir
            .close()
            .map_err(|e| format!("failed to cleanup temp dir '{path}': {e}"))
    }
}

// ── Stream collection ────────────────────────────────────────────────────────

pub async fn collect_stream_bundle<S, C, E, F>(
    stream: &mut S,
    extract: F,
) -> Result<StreamBundle, String>
where
    S: Stream<Item = Result<C, E>> + Unpin,
    E: std::fmt::Display,
    F: Fn(C) -> (Vec<u8>, String),
{
    let mut sha256 = String::new();
    let mut data = Vec::new();
    let mut saw_chunk = false;
    let mut last_chunk_had_sha = false;

    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| format!("Stream error: {e}"))?;
        let (chunk_data, chunk_sha) = extract(chunk);
        saw_chunk = true;

        last_chunk_had_sha = !chunk_sha.is_empty();
        if last_chunk_had_sha {
            sha256 = chunk_sha;
        }

        let new_total = data
            .len()
            .checked_add(chunk_data.len())
            .ok_or_else(|| "Stream bundle size overflow".to_string())?;
        if new_total > MAX_STREAM_BUNDLE_SIZE {
            return Err(format!(
                "Stream bundle exceeds maximum size limit ({new_total} bytes > {MAX_STREAM_BUNDLE_SIZE} bytes)"
            ));
        }

        data.extend_from_slice(&chunk_data);
    }

    if !saw_chunk {
        return Err("Empty stream: no chunks received".to_string());
    }

    if !last_chunk_had_sha {
        return Err("Stream ended without terminal sha256 fingerprint".to_string());
    }

    if data.is_empty() {
        return Err("Received empty stream bundle".to_string());
    }

    debug!(
        "Stream bundle received: {} bytes, sha256={}",
        data.len(),
        sha256
    );

    Ok(StreamBundle {
        sha256,
        compressed_data: data,
    })
}

/// Wraps both the RPC handshake and stream collection inside a single timeout.
///
/// This ensures the entire operation (connection + streaming) is bounded, eliminating
/// the gap where the RPC handshake could hang without any timeout.
pub async fn collect_bundle_from_rpc<S, C, E, F>(
    rpc_future: impl Future<Output = Result<tonic::Response<S>, tonic::Status>>,
    extract: F,
    timeout_duration: Duration,
) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>>
where
    S: Stream<Item = Result<C, E>> + Unpin,
    E: std::fmt::Display,
    F: Fn(C) -> (Vec<u8>, String),
{
    tokio::time::timeout(timeout_duration, async {
        let mut stream = rpc_future.await?.into_inner();
        collect_stream_bundle(&mut stream, extract)
            .await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
    })
    .await
    .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        format!(
            "get_tokenizer timed out after {}s",
            timeout_duration.as_secs()
        )
        .into()
    })?
}

// ── Validation ───────────────────────────────────────────────────────────────

// digest 0.11's `Output` is a `hybrid_array::Array` without `LowerHex`, so format bytes ourselves.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

pub fn validate_bundle_sha256(bundle: &StreamBundle) -> Result<(), String> {
    if bundle.sha256.is_empty() {
        return Ok(());
    }

    let computed = hex_lower(&Sha256::digest(&bundle.compressed_data));
    if !computed.eq_ignore_ascii_case(&bundle.sha256) {
        return Err(format!(
            "Bundle fingerprint mismatch: expected {}, got {}",
            bundle.sha256, computed
        ));
    }
    Ok(())
}

fn checked_add_uncompressed_size(total: u64, entry_size: u64) -> Result<u64, String> {
    total
        .checked_add(entry_size)
        .ok_or_else(|| "Zip archive total uncompressed size overflowed u64".to_string())
}

pub fn validate_zip_archive<R: Read + Seek>(reader: R) -> Result<ZipArchive<R>, String> {
    let mut archive =
        ZipArchive::new(reader).map_err(|e| format!("Failed to open zip archive: {e}"))?;

    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(format!(
            "Zip archive has too many entries ({} > {})",
            archive.len(),
            MAX_ZIP_ENTRIES
        ));
    }

    let mut total_uncompressed: u64 = 0;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read zip entry {i}: {e}"))?;
        let path = entry
            .name()
            .map_err(|e| format!("Failed to read the name of zip entry {i}: {e}"))?;
        let has_traversal = Path::new(&*path).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
        if has_traversal {
            return Err(format!("Zip archive contains unsafe path: {path}"));
        }
        total_uncompressed = checked_add_uncompressed_size(total_uncompressed, entry.size())?;
    }

    if total_uncompressed > MAX_UNCOMPRESSED_SIZE {
        return Err(format!(
            "Zip archive uncompressed size too large ({total_uncompressed} bytes > {MAX_UNCOMPRESSED_SIZE} bytes)"
        ));
    }

    Ok(archive)
}

// ── Extraction ───────────────────────────────────────────────────────────────

pub fn extract_bundle_to_tempdir(bundle: &StreamBundle) -> Result<ExtractedArchiveDir, String> {
    let mut archive = validate_zip_archive(Cursor::new(bundle.compressed_data.as_slice()))?;
    let dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;
    archive
        .extract(dir.path())
        .map_err(|e| format!("archive extraction failed: {e}"))?;

    Ok(ExtractedArchiveDir { temp_dir: dir })
}

pub fn with_extracted_bundle<R>(
    bundle: &StreamBundle,
    operation: impl FnOnce(&Path) -> Result<R, String>,
) -> Result<R, String> {
    let extracted = extract_bundle_to_tempdir(bundle)?;
    let result = operation(extracted.path());

    if let Err(e) = extracted.cleanup() {
        warn!("Bundle extraction tempdir cleanup failed: {}", e);
    }

    result
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use futures::{executor::block_on, stream};
    use sha2::{Digest, Sha256};
    use zip::{write::SimpleFileOptions, ZipWriter};

    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn identity(chunk: (Vec<u8>, String)) -> (Vec<u8>, String) {
        chunk
    }

    type ChunkResult = Result<(Vec<u8>, String), &'static str>;

    fn build_test_zip(entry_count: usize, payload: &[u8]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);

        for i in 0..entry_count {
            writer
                .start_file(format!("file-{i}.txt"), SimpleFileOptions::default())
                .unwrap();
            writer.write_all(payload).unwrap();
        }

        writer.finish().unwrap().into_inner()
    }

    fn make_bundle(compressed_data: Vec<u8>, sha256: String) -> StreamBundle {
        StreamBundle {
            sha256,
            compressed_data,
        }
    }

    // ── Stream collection tests ──────────────────────────────────────────

    #[test]
    fn collect_single_chunk() {
        let mut s = stream::iter(vec![
            Ok((b"zipdata".to_vec(), "abc123".to_string())) as ChunkResult
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"zipdata");
        assert_eq!(bundle.sha256, "abc123");
    }

    #[test]
    fn collect_multiple_chunks() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), String::new())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
            Ok((b"ghi".to_vec(), "sha".to_string())),
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"abcdefghi");
        assert_eq!(bundle.sha256, "sha");
    }

    #[test]
    fn collect_uses_last_non_empty_sha256() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), "sha-old".to_string())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
            Ok((b"ghi".to_vec(), "sha-new".to_string())),
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"abcdefghi");
        assert_eq!(bundle.sha256, "sha-new");
    }

    #[test]
    fn collect_rejects_missing_terminal_sha256() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), "sha-old".to_string())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
        ]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("without terminal sha256"));
    }

    #[test]
    fn collect_empty_stream() {
        let mut s = stream::iter(Vec::<ChunkResult>::new());

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("no chunks received"));
    }

    #[test]
    fn collect_stream_error() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), String::new())),
            Err("socket closed"),
        ]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("Stream error: socket closed"));
    }

    #[test]
    fn collect_exceeds_max_size() {
        let oversized = vec![0u8; MAX_STREAM_BUNDLE_SIZE + 1];
        let mut s = stream::iter(vec![Ok((oversized, String::new())) as ChunkResult]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("exceeds maximum size limit"));
    }

    #[test]
    fn collect_rejects_no_sha256() {
        let mut s = stream::iter(vec![Ok((b"data".to_vec(), String::new())) as ChunkResult]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("without terminal sha256"));
    }

    // ── SHA-256 validation tests ─────────────────────────────────────────

    #[test]
    fn validate_sha256_accepts_matching_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let sha256 = hex_lower(&Sha256::digest(&compressed_data));
        let bundle = make_bundle(compressed_data, sha256);

        validate_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn validate_sha256_accepts_uppercase_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let sha256 = hex_lower(&Sha256::digest(&compressed_data)).to_uppercase();
        let bundle = make_bundle(compressed_data, sha256);

        validate_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn validate_sha256_rejects_mismatch() {
        let bundle = make_bundle(b"test-bundle".to_vec(), "deadbeef".to_string());

        let err = validate_bundle_sha256(&bundle).unwrap_err();
        assert!(err.contains("fingerprint mismatch"));
    }

    #[test]
    fn validate_sha256_allows_missing_fingerprint() {
        let bundle = make_bundle(b"test-bundle".to_vec(), String::new());
        validate_bundle_sha256(&bundle).unwrap();
    }

    // ── Zip validation tests ─────────────────────────────────────────────

    #[test]
    fn validate_zip_accepts_valid_zip() {
        let zip_bytes = build_test_zip(1, b"hello");
        let archive = validate_zip_archive(Cursor::new(zip_bytes)).unwrap();
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn validate_zip_rejects_invalid_data() {
        let err = validate_zip_archive(Cursor::new(vec![1, 2, 3, 4])).unwrap_err();
        assert!(err.contains("Failed to open zip archive"));
    }

    #[test]
    fn validate_zip_rejects_too_many_entries() {
        let zip_bytes = build_test_zip(MAX_ZIP_ENTRIES + 1, b"x");
        let err = validate_zip_archive(Cursor::new(zip_bytes)).unwrap_err();
        assert!(err.contains("too many entries"));
    }

    #[test]
    fn validate_zip_rejects_unsafe_paths() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        writer
            .start_file("../evil.txt", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"x").unwrap();
        let zip_bytes = writer.finish().unwrap().into_inner();

        let err = validate_zip_archive(Cursor::new(zip_bytes)).unwrap_err();
        assert!(err.contains("unsafe path"));
    }

    #[test]
    fn checked_add_rejects_u64_overflow() {
        let err = checked_add_uncompressed_size(u64::MAX, 1).unwrap_err();
        assert!(err.contains("overflowed u64"));
    }

    // ── Extraction tests ─────────────────────────────────────────────────

    #[test]
    fn extract_bundle_extracts_files() {
        let zip_bytes = build_test_zip(1, b"hello");
        let sha256 = hex_lower(&Sha256::digest(&zip_bytes));
        let bundle = make_bundle(zip_bytes, sha256);

        let extracted = extract_bundle_to_tempdir(&bundle).unwrap();
        let file_path = extracted.path().join("file-0.txt");
        let content = fs::read(file_path).unwrap();
        assert_eq!(content, b"hello");
        extracted.cleanup().unwrap();
    }
}
