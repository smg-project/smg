//! A rollback: a generation on file runs again under the next number, or
//! is refused.

use super::{applied, bind, enforce, generations, roll_back, stored, switched, DIFFERS};
use crate::{
    document::operator::test_support::{needs_rows, standing, LEAST_LOAD, RANDOM},
    service::controller::{
        spec_store::tests::{name_of, names},
        wire,
    },
};

/// A rollback runs a generation on file again: its bytes are on file
/// under the next number and run under it, with their digest and the
/// feed an enforce answers. The standing generation, and a generation
/// whose document is the standing one's, are no change.
#[test]
fn a_rollback_runs_a_generation_on_file_again_under_the_next_number() {
    let (dir, store, generations) = stored();
    let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
    let (_, second) = enforce(&generations, RANDOM).expect("it runs");
    let ran = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));

    let back = generations.roll_back(1).expect("on file");
    assert_eq!(back.standing, (2, Some(second)));
    assert_eq!(back.ran, Ok((3, first)));
    assert_eq!(standing(&store), (3, Some(first)));
    let on_file = std::fs::read(dir.path().join(name_of(3, "yaml"))).expect("the file");
    assert_eq!(on_file, LEAST_LOAD.as_bytes());
    let needs = back.needs.map(wire::needs_diff).expect("the diff");
    assert_eq!(needs.before, Some(ran));
    let runs = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));
    assert_eq!(needs.after, Some(runs));

    for same in [3, 1] {
        assert_eq!(roll_back(&generations, same), Ok((3, first)));
    }
    assert_eq!(names(dir.path()).len(), 3);
    assert_eq!(generations.apply_counts(), applied(3, 0));
}

/// A rollback to a generation on file whose bindings differ on the
/// overload switch is refused in the refusal's words, and the standing
/// generation stays. An older build wrote that generation: this
/// build's bind lists it by its digest and passes the newest alone.
#[test]
fn a_rollback_to_a_generation_whose_bindings_differ_on_the_overload_switch_is_refused() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let older = switched(Some(false), None);
    std::fs::write(dir.path().join(name_of(1, "yaml")), older).expect("the older build's");
    std::fs::write(dir.path().join(name_of(2, "yaml")), RANDOM).expect("the newest");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    let own = standing(&store);
    assert_eq!(own.0, 2);

    let refused = roll_back(&generations, 1).expect_err("its bindings differ");
    assert_eq!(refused.to_string(), DIFFERS);
    assert_eq!(standing(&store), own, "the standing generation stays");
    assert_eq!(names(dir.path()).len(), 2, "no file is written");
}
