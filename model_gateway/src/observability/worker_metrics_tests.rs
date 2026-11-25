//! The gateway's case over the two crates' metric paths.

use std::sync::Arc;

/// The gateway's own module reaches the model labels' set and the keeping
/// switch through the external router crate's paths, this module through the
/// worker crate's: one set and one switch stand behind both.
#[test]
fn both_paths_reach_one_label_set_and_one_switch() {
    use smg_external_router::metrics as by_router;
    use smg_worker::metrics as by_worker;

    let through_router = by_router::intern_model_label("both-paths-model");
    let through_worker = by_worker::intern_model_label("both-paths-model");
    assert!(Arc::ptr_eq(&through_router, &through_worker));

    by_router::handles::force_on_this_thread(Some(true));
    assert!(by_worker::handles::keep_handles());
    by_router::handles::force_on_this_thread(Some(false));
    assert!(!by_worker::handles::keep_handles());
    by_router::handles::force_on_this_thread(None);
}
