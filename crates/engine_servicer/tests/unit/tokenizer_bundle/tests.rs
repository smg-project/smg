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
        "tekken.json",
        "image_preprocessing.json",
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
            "tekken.json",
            "image_preprocessing.json",
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
        assert_eq!(entry.name().unwrap(), name);
        assert!(entry.is_file());
        assert_eq!(entry.compression(), CompressionMethod::Deflated);
        let mut content = String::new();
        entry.read_to_string(&mut content).unwrap();
        assert_eq!(content, name);
    }
}
