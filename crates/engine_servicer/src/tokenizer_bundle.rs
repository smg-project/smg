//! `GetTokenizer`: the servicer's tokenizer directory as a zip bundle,
//! streamed in chunks with the archive's sha256 on the last one. Selection,
//! entry names, compression and chunk size follow the Python servicers'
//! `tokenizer_bundle.build_tokenizer_zip` and `CHUNK_SIZE`, so the Router's
//! bundle loader sees the same thing from either implementation.

use std::{
    fs::{self, File},
    io::{self, Cursor},
    path::{Path, PathBuf},
};

use futures::stream;
use sha2::{Digest, Sha256};
use smg_grpc_client::common_proto::GetTokenizerChunk;
use tokio::task::spawn_blocking;
use tonic::Status;
use tracing::{info, warn};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::BoxStream;

/// Bytes per streamed chunk (the Python servicers' `CHUNK_SIZE`).
pub(crate) const CHUNK_SIZE: usize = 64 * 1024;

/// Exact file names, in bundle order (the Python `TOKENIZER_FILES`).
const TOKENIZER_FILES: [&str; 13] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "tiktoken.model",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];

/// Name suffixes added after the exact names, one family at a time in this
/// order (the Python `TOKENIZER_GLOBS`: `*.tiktoken`, `*.jinja`, `*.model`).
const TOKENIZER_SUFFIXES: [&str; 3] = [".tiktoken", ".jinja", ".model"];

/// What `build_tokenizer_zip` bundles from `dir`, as `(entry name, path)` in
/// bundle order: the exact names, then each suffix family (sorted by name
/// where Python takes directory order), top-level regular files only, no
/// duplicates. A missing directory selects nothing, as `Path.glob` does.
pub(super) fn select_files(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut selected: Vec<(String, PathBuf)> = TOKENIZER_FILES
        .into_iter()
        .map(|name| (name.to_string(), dir.join(name)))
        .filter(|(_, path)| path.is_file())
        .collect();
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file())
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    names.sort_unstable();
    for suffix in TOKENIZER_SUFFIXES {
        for name in names.iter().filter(|name| name.ends_with(suffix)) {
            if !selected.iter().any(|(added, _)| added == name) {
                selected.push((name.clone(), dir.join(name)));
            }
        }
    }
    selected
}

/// A built bundle: the zip bytes and their lowercase hex sha256.
#[derive(Debug)]
pub(super) struct TokenizerBundle {
    pub(super) zip: Vec<u8>,
    pub(super) sha256: String,
}

impl TokenizerBundle {
    /// Zip the selection from `dir` in memory (Deflate, names relative to
    /// `dir`, no directory entries) and fingerprint the archive.
    pub(super) fn build(dir: &Path) -> io::Result<Self> {
        let files = select_files(dir);
        if files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("No tokenizer files found in {}", dir.display()),
            ));
        }
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, path) in &files {
            writer
                .start_file(name.as_str(), options)
                .map_err(io::Error::other)?;
            io::copy(&mut File::open(path)?, &mut writer)?;
        }
        let zip = writer.finish().map_err(io::Error::other)?.into_inner();
        let sha256 = hex_lower(&Sha256::digest(&zip));
        Ok(Self { zip, sha256 })
    }

    /// The bundle as `GetTokenizerChunk` frames of `CHUNK_SIZE`, the sha256
    /// only on the last.
    pub(super) fn into_chunks(self) -> impl Iterator<Item = GetTokenizerChunk> {
        let Self { zip, sha256 } = self;
        let total = zip.len();
        let count = total.div_ceil(CHUNK_SIZE).max(1);
        (0..count).map(move |index| {
            let start = index * CHUNK_SIZE;
            let end = total.min(start + CHUNK_SIZE);
            GetTokenizerChunk {
                data: zip[start..end].to_vec(),
                sha256: if index + 1 == count {
                    sha256.clone()
                } else {
                    String::new()
                },
            }
        })
    }
}

/// digest 0.11's output has no `LowerHex`, so the hex is formatted by hand.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// `GetTokenizer`: refused without a tokenizer directory (the Python
/// servicer's precondition); otherwise bundle it off the runtime and stream.
pub(crate) async fn get_tokenizer(
    tokenizer_dir: Option<String>,
) -> Result<BoxStream<GetTokenizerChunk>, Status> {
    let Some(dir) = tokenizer_dir else {
        return Err(Status::failed_precondition(
            "Tokenizer path is not configured on this server.",
        ));
    };
    // Reading and deflating the directory is file I/O and CPU measured in
    // hundreds of milliseconds: off the runtime, as the Python servicer's
    // `to_thread`, so token streams keep flowing meanwhile.
    let bundle = spawn_blocking(move || TokenizerBundle::build(Path::new(&dir)))
        .await
        .map_err(|error| Status::internal(format!("tokenizer bundle task failed: {error}")))?
        .map_err(|error| {
            warn!(%error, "failed to build the tokenizer bundle");
            Status::internal(error.to_string())
        })?;
    info!(
        bytes = bundle.zip.len(),
        sha256 = %bundle.sha256,
        "streaming tokenizer bundle"
    );
    Ok(Box::pin(stream::iter(bundle.into_chunks().map(Ok))))
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use zip::ZipArchive;

    use super::*;

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), name).unwrap();
    }

    /// The exact names in list order, then the suffix families each sorted;
    /// weights, docs, nested files, lookalikes and directories stay out, as
    /// with the Python builder.
    #[test]
    fn selection_matches_the_python_builder() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.model",
            "zeta.jinja",
            "alpha.jinja",
            "cl100k.tiktoken",
            "extra.model",
            "model.safetensors",
            "README.md",
            ".gitattributes",
            "tokenizer.json.bak",
        ] {
            touch(dir.path(), name);
        }
        fs::create_dir(dir.path().join("original")).unwrap();
        touch(dir.path(), "original/tokenizer.model");
        fs::create_dir(dir.path().join("dir.jinja")).unwrap();

        let selected = select_files(dir.path());
        let names: Vec<&str> = selected.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.model",
                "cl100k.tiktoken",
                "alpha.jinja",
                "zeta.jinja",
                "extra.model",
            ]
        );
        for (name, path) in &selected {
            assert_eq!(path, &dir.path().join(name));
        }
    }

    /// Nothing selected (an empty or missing directory) is the Python
    /// builder's `FileNotFoundError`, not an empty archive.
    #[test]
    fn an_empty_or_missing_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        assert!(select_files(dir.path()).is_empty());
        assert!(select_files(&missing).is_empty());
        for path in [dir.path(), missing.as_path()] {
            let error = TokenizerBundle::build(path).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::NotFound);
            assert_eq!(
                error.to_string(),
                format!("No tokenizer files found in {}", path.display())
            );
        }
    }

    /// Full `CHUNK_SIZE` frames with an empty fingerprint, the remainder (or
    /// the only frame) carrying it; an exact multiple adds no empty frame.
    #[test]
    fn chunks_carry_the_fingerprint_on_the_last_frame_only() {
        let bundle = TokenizerBundle {
            zip: vec![7; 2 * CHUNK_SIZE + 5],
            sha256: "abc".to_string(),
        };
        let chunks: Vec<GetTokenizerChunk> = bundle.into_chunks().collect();
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks[..2] {
            assert_eq!(chunk.data.len(), CHUNK_SIZE);
            assert!(chunk.sha256.is_empty());
        }
        assert_eq!(chunks[2].data, vec![7; 5]);
        assert_eq!(chunks[2].sha256, "abc");

        let exact = TokenizerBundle {
            zip: vec![1; CHUNK_SIZE],
            sha256: "def".to_string(),
        };
        let chunks: Vec<GetTokenizerChunk> = exact.into_chunks().collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].data.len(), CHUNK_SIZE);
        assert_eq!(chunks[0].sha256, "def");
    }

    /// The archive is what the Router reads back: Deflate entries named
    /// relative to the directory, contents intact, fingerprint matching.
    #[test]
    fn bundle_is_a_deflate_zip_of_the_selection() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "tokenizer.json");
        touch(dir.path(), "chat_template.jinja");
        touch(dir.path(), "model.safetensors");
        let bundle = TokenizerBundle::build(dir.path()).unwrap();
        assert_eq!(bundle.sha256, hex_lower(&Sha256::digest(&bundle.zip)));
        let mut archive = ZipArchive::new(Cursor::new(bundle.zip.as_slice())).unwrap();
        assert_eq!(archive.len(), 2);
        for (index, name) in ["tokenizer.json", "chat_template.jinja"]
            .into_iter()
            .enumerate()
        {
            let mut entry = archive.by_index(index).unwrap();
            assert_eq!(entry.name(), name);
            assert!(entry.is_file());
            assert_eq!(entry.compression(), CompressionMethod::Deflated);
            let mut content = String::new();
            entry.read_to_string(&mut content).unwrap();
            assert_eq!(content, name);
        }
    }
}
