use super::tokenizer_source;
use crate::workflow::steps::local::test_support::labels_of;

/// A worker that names its model by id only, with neither a path label
/// nor a router flag: the id is the source (a local load that fails on
/// it falls back to the worker's bundle); an empty id is no source.
#[test]
fn tokenizer_source_is_the_model_id_when_nothing_else_names_a_path() {
    let id_only = labels_of(&[("model_id", "/engines/served")]);
    assert_eq!(
        tokenizer_source(&id_only, None, None),
        Some("/engines/served")
    );
    assert_eq!(
        tokenizer_source(&id_only, Some(""), Some("")),
        Some("/engines/served")
    );
    assert_eq!(
        tokenizer_source(&labels_of(&[("model_id", "")]), None, None),
        None
    );
    assert_eq!(tokenizer_source(&labels_of(&[]), None, None), None);
}

/// The operator's flags and the worker's path labels keep their
/// precedence over the id: `--tokenizer-path`, then `--model-path`, win
/// over an id-only worker, and a `tokenizer_path` or `model_path` label
/// wins over the flags as before.
#[test]
fn tokenizer_source_reads_the_model_id_only_after_the_flags_and_path_labels() {
    let id_only = labels_of(&[("model_id", "/engines/served")]);
    assert_eq!(
        tokenizer_source(&id_only, Some("/opt/tok"), None),
        Some("/opt/tok")
    );
    assert_eq!(
        tokenizer_source(&id_only, None, Some("/opt/model")),
        Some("/opt/model")
    );
    assert_eq!(
        tokenizer_source(&id_only, Some("/opt/tok"), Some("/opt/model")),
        Some("/opt/tok")
    );

    let with_paths = labels_of(&[
        ("model_id", "/engines/served"),
        ("model_path", "/engines/weights"),
        ("tokenizer_path", "/engines/tok"),
    ]);
    assert_eq!(
        tokenizer_source(&with_paths, Some("/opt/tok"), None),
        Some("/engines/tok")
    );
    let with_model_path = labels_of(&[
        ("model_id", "/engines/served"),
        ("model_path", "/engines/weights"),
    ]);
    assert_eq!(
        tokenizer_source(&with_model_path, Some("/opt/tok"), None),
        Some("/engines/weights")
    );
}
