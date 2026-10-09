//! An operator-triggered heap profile of this gateway process.
//!
//! `POST /heap_profile` asks the gateway's jemalloc for a `prof.dump` into
//! the directory `--jemalloc-prof-dir` names and answers the file's path. It
//! needs a binary built with the `jemalloc-profiling` feature, run with
//! profiling on (`_RJEM_MALLOC_CONF=prof:true,prof_active:true,lg_prof_sample:19`);
//! the file reads with `jeprof`. Any other build keeps the route and answers
//! 501, so a client learns why rather than meeting a 404.

use std::path::{Path, PathBuf};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use metrics::{counter, describe_counter};
use serde::Serialize;
use smg_external_router::error;

/// Whether this binary's allocator can write a profile.
pub const BUILT: bool = cfg!(all(
    feature = "jemalloc-profiling",
    not(target_env = "msvc"),
    not(target_env = "musl")
));

/// What a dump request came to.
#[derive(Debug, PartialEq, Eq)]
pub enum HeapProfile {
    /// The profile is at `path`; `prof_active` is jemalloc's sampling state
    /// at the dump (a profile taken while sampling is off holds nothing).
    Dumped { path: PathBuf, prof_active: bool },
    /// The binary was built without the `jemalloc-profiling` feature.
    NotBuilt,
    /// `--jemalloc-prof-dir` is not set.
    NoDirectory,
    /// jemalloc runs without profiling: the process did not start with
    /// `prof:true` in `_RJEM_MALLOC_CONF`.
    Inactive,
    /// The directory could not be made or jemalloc refused the dump.
    Failed(String),
}

#[derive(Serialize)]
struct DumpedBody<'a> {
    path: &'a Path,
    prof_active: bool,
}

impl IntoResponse for HeapProfile {
    fn into_response(self) -> Response {
        match self {
            Self::Dumped { path, prof_active } => Json(DumpedBody {
                path: &path,
                prof_active,
            })
            .into_response(),
            Self::NotBuilt => error::not_implemented(
                "heap_profile_not_built",
                "this gateway was built without the jemalloc-profiling feature; a build with \
                 `--features jemalloc-profiling` writes heap profiles",
            ),
            Self::NoDirectory => error::not_found(
                "heap_profile_dir_not_configured",
                "no directory for heap profiles: start the gateway with --jemalloc-prof-dir <dir>",
            ),
            Self::Inactive => error::create_error(
                StatusCode::CONFLICT,
                "heap_profile_inactive",
                "jemalloc profiling is off in this process: start it with \
                 _RJEM_MALLOC_CONF=prof:true,prof_active:true,lg_prof_sample:19",
            ),
            Self::Failed(reason) => error::internal_error("heap_profile_dump_failed", reason),
        }
    }
}

/// Write a heap profile into `dir` and count the attempt.
pub fn dump(dir: Option<&Path>) -> HeapProfile {
    if !BUILT {
        return HeapProfile::NotBuilt;
    }
    let Some(dir) = dir else {
        return HeapProfile::NoDirectory;
    };
    let outcome = jemalloc::dump(dir);
    let result = if matches!(outcome, HeapProfile::Dumped { .. }) {
        "ok"
    } else {
        "error"
    };
    counter!("smg_allocator_prof_dumps_total", "outcome" => result).increment(1);
    outcome
}

/// [`dump`] off the async runtime: jemalloc writes the whole profile before
/// it returns.
pub async fn dump_blocking(dir: Option<String>) -> HeapProfile {
    tokio::task::spawn_blocking(move || dump(dir.as_deref().map(Path::new)))
        .await
        .unwrap_or_else(|join| HeapProfile::Failed(format!("dump task: {join}")))
}

/// The dump counter at zero, with its help text, on a build that can dump.
/// Call once the recorder is installed.
pub fn init_series() {
    if !BUILT {
        return;
    }
    describe_counter!(
        "smg_allocator_prof_dumps_total",
        "Heap profile dumps asked of POST /heap_profile, by outcome (ok, error)"
    );
    for outcome in ["ok", "error"] {
        counter!("smg_allocator_prof_dumps_total", "outcome" => outcome).absolute(0);
    }
}

#[cfg(all(
    feature = "jemalloc-profiling",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
mod jemalloc {
    use std::{
        ffi::CString,
        path::Path,
        sync::atomic::{AtomicU64, Ordering},
    };

    use tikv_jemalloc_ctl::{profiling, raw};

    use super::HeapProfile;

    /// `prof.dump` and `prof.active` have no safe wrapper in
    /// tikv-jemalloc-ctl: the raw mallctl writes the path as a `*const c_char`
    /// and reads a `bool`, the types jemalloc documents for these keys.
    #[expect(
        unsafe_code,
        reason = "mallctl keys without a safe wrapper: the string behind the written pointer outlives the call and the read is bool-sized"
    )]
    pub(super) fn dump(dir: &Path) -> HeapProfile {
        match profiling::prof::read() {
            Ok(true) => {}
            Ok(false) => return HeapProfile::Inactive,
            Err(error) => return HeapProfile::Failed(format!("opt.prof: {error}")),
        }
        if let Err(error) = std::fs::create_dir_all(dir) {
            return HeapProfile::Failed(format!("{}: {error}", dir.display()));
        }
        // The UTC second, the pid and a per-process sequence number: two dumps
        // in the same second are two files.
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let path = dir.join(format!(
            "smg-heap-{}-{}-{}.heap",
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let Ok(c_path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
            return HeapProfile::Failed(format!("{}: interior NUL", path.display()));
        };
        // SAFETY: `prof.dump` takes a `const char *`; `c_path` outlives the call.
        if let Err(error) = unsafe { raw::write(b"prof.dump\0", c_path.as_ptr()) } {
            return HeapProfile::Failed(format!("prof.dump: {error}"));
        }
        // SAFETY: `prof.active` is a bool-sized read.
        let prof_active = unsafe { raw::read::<bool>(b"prof.active\0") }.unwrap_or(false);
        HeapProfile::Dumped { path, prof_active }
    }
}

#[cfg(not(all(
    feature = "jemalloc-profiling",
    not(target_env = "msvc"),
    not(target_env = "musl")
)))]
mod jemalloc {
    use std::path::Path;

    use super::HeapProfile;

    pub(super) fn dump(_dir: &Path) -> HeapProfile {
        HeapProfile::NotBuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");
        String::from_utf8(bytes.to_vec()).expect("utf-8 body")
    }

    /// Each refusal names its remedy: the feature, the flag, the conf.
    #[tokio::test]
    async fn every_refusal_names_its_remedy() {
        for (outcome, status, remedy) in [
            (
                HeapProfile::NotBuilt,
                StatusCode::NOT_IMPLEMENTED,
                "--features jemalloc-profiling",
            ),
            (
                HeapProfile::NoDirectory,
                StatusCode::NOT_FOUND,
                "--jemalloc-prof-dir",
            ),
            (HeapProfile::Inactive, StatusCode::CONFLICT, "prof:true"),
            (
                HeapProfile::Failed("disk full".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "disk full",
            ),
        ] {
            let response = outcome.into_response();
            assert_eq!(response.status(), status);
            let text = body(response).await;
            assert!(text.contains(remedy), "{status}: {text}");
        }
    }

    #[cfg(not(feature = "jemalloc-profiling"))]
    #[tokio::test]
    async fn a_build_without_the_feature_answers_501_whatever_the_flag() {
        assert_eq!(dump(Some(Path::new("/tmp"))), HeapProfile::NotBuilt);
        assert_eq!(dump(None), HeapProfile::NotBuilt);
        let response = dump_blocking(Some("/tmp".to_string()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[cfg(feature = "jemalloc-profiling")]
    #[test]
    fn a_profiling_build_without_the_flag_answers_404() {
        assert_eq!(dump(None), HeapProfile::NoDirectory);
    }
}
