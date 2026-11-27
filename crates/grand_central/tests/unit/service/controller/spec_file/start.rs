//! The start: the file as it stands and the links of a mounted file. A
//! start refused: a file that is not there, a directory, a file over the
//! bound, a document the validator refuses. Also a start on the spec a
//! gateway writes for the manual policy.

use super::*;

/// The start reads the file as it stands: a document with no line is
/// generation 1 and a written line is the number, in the form the
/// file's name says. A file of exactly the bound starts: the bound is
/// the largest document.
#[test]
fn a_start_applies_the_file_as_it_stands() {
    let bed = bed(LEAST_LOAD.as_bytes());
    assert_eq!(bed.start_on(&bed.path), Ok(1));
    assert_eq!(bed.answered(), (1, "yaml".to_owned(), LEAST_LOAD.into()));

    let json = br#"{"apiVersion":"smg.policy/v1","generation":4,"profiles":{"default":{"pick":"random"}}}"#;
    let path = bed.dir.path().join("policy.json");
    fs::write(&path, json).expect("the file");
    assert_eq!(bed.start_on(&path), Ok(4));
    assert_eq!(bed.answered(), (4, "json".to_owned(), json.to_vec()));

    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let path = bed.dir.path().join("largest.yaml");
    fs::write(&path, padded(RANDOM, bound)).expect("the file");
    assert_eq!(bed.start_on(&path), Ok(5));
    assert_eq!(bed.answered().2.len(), bound);
}

/// A mounted ConfigMap's file is a link through `..data` into a
/// version's directory: the start follows it.
#[cfg(unix)]
#[test]
fn a_start_follows_the_links_of_a_mounted_file() {
    use std::os::unix::fs::symlink;

    let bed = bed(b"");
    let mount = bed.dir.path().join("mount");
    fs::create_dir_all(mount.join("..version_1")).expect("the version's directory");
    fs::write(mount.join("..version_1/policy.yaml"), with_line(RANDOM, 6)).expect("the file");
    symlink("..version_1", mount.join("..data")).expect("the data link");
    symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");

    assert_eq!(bed.start_on(&mount.join("policy.yaml")), Ok(6));
    assert_eq!(bed.answered().2, with_line(RANDOM, 6));
}

/// A file that is not there, a directory, a file over the bound (by its
/// size, unread) and a document the validator refuses each refuse the
/// start by name; nothing is applied and nothing counted.
#[test]
fn a_start_refuses_a_file_it_cannot_run() {
    let bed = bed(b"");
    fs::remove_file(&bed.path).expect("the file goes");
    match bed.start_on(&bed.path) {
        Err(SpecRejected::Unreadable(error)) => {
            assert!(error.contains("No such file"), "{error}");
        }
        other => panic!("started as {other:?}"),
    }
    assert!(matches!(
        bed.start_on(bed.dir.path()),
        Err(SpecRejected::Unreadable(_))
    ));

    // Five bytes over: the refusal names the file's size, which no read
    // gave it (a read stops one byte past the bound).
    let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 5;
    fs::write(&bed.path, vec![b'#'; over]).expect("the file");
    assert_eq!(
        bed.start_on(&bed.path),
        Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 5))
    );

    fs::write(&bed.path, LEAST_LOAD.replace("[healthy]", "[nowhere]")).expect("the file");
    let unknown_filter = SpecError::UnknownFilter {
        profile: "default".to_owned(),
        name: "nowhere".to_owned(),
    };
    assert_eq!(
        bed.start_on(&bed.path),
        Err(SpecRejected::Invalid(unknown_filter))
    );
    assert!(!bed.generations.bound());
    assert_eq!((bed.generation(), bed.rejected()), (0, 0));
}

/// A start takes the policy spec a gateway writes for the manual policy on
/// a pair's two sides of its encode mode. The start checks it as a
/// controller does: in the stock catalog with no stage that needs the
/// lifecycle's calls. The spec binds three sides under the manual preset's
/// rule, which tells its stages.
///
/// Under a rule that keeps what it places from the profile the same spec
/// is refused by that field's name, and the first start's generation
/// stands.
///
/// The settings write that spec:
/// `the_manual_policy_on_both_sides_of_a_pair_runs_under_three_sides` in
/// `crates/grand_central/tests/unit/document/routing_settings/tests.rs`.
#[test]
fn a_start_takes_a_sessions_rule_under_three_sides_as_a_controller_checks_it() {
    let mut bed = bed(b"");
    let catalog = Arc::new(StageRegistry::builtin().without_lifecycle());
    let store = Arc::new(MemoryStore::new());
    let validate = AppliedSpec::validate_digested;
    bed.generations = Arc::new(Generations::validating(store, catalog, validate, None));
    let manual = PolicySpec::preset("manual").expect("a preset of this build");
    let mut spec = PolicySpec::epd_presets("consistent_hashing", "manual", "manual")
        .expect("the mode's presets");
    spec.bindings[0]
        .sessions
        .clone_from(&manual.bindings[0].sessions);
    let path = bed.dir.path().join("policy.json");
    let mut document = serde_json::to_value(&spec).expect("the spec serializes");
    fs::write(&path, document.to_string()).expect("the file");
    assert_eq!(bed.start_on(&path), Ok(1));

    document["generation"] = 2.into();
    document["bindings"][0]["sessions"]["rule_placed"] = "untold".into();
    fs::write(&path, document.to_string()).expect("the file");
    match bed.start_on(&path) {
        Err(SpecRejected::Invalid(refused)) => {
            assert_eq!(refused.field_path(), "bindings[0].sessions.rule_placed");
        }
        other => panic!("started as {other:?}"),
    }
    assert_eq!(bed.generation(), 1, "the first start's generation stands");
}
