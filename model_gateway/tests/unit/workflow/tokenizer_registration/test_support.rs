/// A directory holding a small real tokenizer (a two-token WordLevel
/// `tokenizer.json`): a source the registration step loads from disk.
pub(crate) fn tokenizer_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
                "decoder": null, "model": {"type": "WordLevel", "vocab": {"hello": 0, "<unk>": 1},
                "unk_token": "<unk>"}}"#,
    )
    .unwrap();
    dir
}
