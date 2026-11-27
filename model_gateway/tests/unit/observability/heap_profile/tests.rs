use super::*;
use crate::middleware::test_support::text_of;

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
        let text = text_of(response).await;
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
