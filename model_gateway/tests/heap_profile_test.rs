//! With the `jemalloc-profiling` feature, `heap_profile::dump` writes a
//! jemalloc heap profile once the process runs with profiling on.
//!
//! jemalloc reads its options once, at its first allocation, so the test
//! re-runs itself with `_RJEM_MALLOC_CONF=prof:true,...`: the parent process
//! (profiling off unless the environment says otherwise) checks the
//! `Inactive` answer, the child the dump and its file.

#![cfg(all(
    feature = "jemalloc-profiling",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use std::process::Command;

use smg::observability::heap_profile::{dump, HeapProfile};

#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const CHILD: &str = "SMG_HEAP_PROFILE_TEST_CHILD";
const TEST: &str = "a_dump_writes_a_heap_profile_once_profiling_is_on";

#[test]
fn a_dump_writes_a_heap_profile_once_profiling_is_on() {
    let dir = tempfile::tempdir().expect("temp dir");
    if std::env::var_os(CHILD).is_none() {
        let parent = dump(Some(dir.path()));
        assert!(
            matches!(parent, HeapProfile::Inactive | HeapProfile::Dumped { .. }),
            "profiling off in the parent: {parent:?}"
        );
        let status = Command::new(std::env::current_exe().expect("the test binary"))
            .args([TEST, "--exact", "--nocapture"])
            .env(CHILD, "1")
            .env(
                "_RJEM_MALLOC_CONF",
                "prof:true,prof_active:true,lg_prof_sample:0",
            )
            .status()
            .expect("re-run the test with profiling on");
        assert!(status.success(), "the profiling child failed: {status}");
        return;
    }

    let allocation = std::hint::black_box(vec![7_u8; 4 << 20]);
    let outcome = dump(Some(dir.path()));
    let HeapProfile::Dumped { path, prof_active } = outcome else {
        panic!("profiling on, yet no dump: {outcome:?}");
    };
    assert!(prof_active, "prof.active follows the conf");
    assert!(path.starts_with(dir.path()), "{}", path.display());
    let profile = std::fs::read_to_string(&path).expect("the dump file");
    let header = profile.lines().next().unwrap_or_default();
    assert!(
        header.starts_with("heap_v2/"),
        "jemalloc heap profile header, got {header:?}"
    );
    assert!(
        profile.contains("MAPPED_LIBRARIES:"),
        "the section jeprof resolves symbols from"
    );
    drop(allocation);
}
