use super::*;

#[test]
fn paths_resolve_against_the_repository() {
    assert_eq!(
        absolute(Path::new("/r"), Path::new("x/y")),
        PathBuf::from("/r/x/y")
    );
    assert_eq!(
        absolute(Path::new("/r"), Path::new("/x")),
        PathBuf::from("/x")
    );
}

#[test]
fn a_missing_revision_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = build_base(dir.path(), "nosuch", Path::new("out/smg"), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("git rev-parse"), "{err}");
}
