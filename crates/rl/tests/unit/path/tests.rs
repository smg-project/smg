use super::*;
use crate::error::RlError;

#[test]
fn accepts_engine_route_shapes() {
    for (raw, want) in [
        ("pause_generation", "pause_generation"),
        ("/pause_generation", "pause_generation"),
        ("update_weights_from_disk", "update_weights_from_disk"),
        ("inference/v1/generate", "inference/v1/generate"),
        ("v1/chat/completions/render", "v1/chat/completions/render"),
        ("server_info", "server_info"),
        ("wake_up", "wake_up"),
    ] {
        assert_eq!(validate_engine_path(raw).unwrap(), want, "{raw}");
    }
}

#[test]
fn rejects_bad_paths() {
    for raw in [
        "",
        "/",
        "//pause",
        "../workers",
        "a/../b",
        "pause?mode=keep",
        "pause#frag",
        "a/b/c/d/e",
        "pause generation",
        "pause%20generation",
        "pause/",
    ] {
        assert!(
            matches!(
                validate_engine_path(raw),
                Err(RlError::InvalidEnginePath(_))
            ),
            "{raw:?} should be rejected"
        );
    }
}

#[test]
fn query_passthrough_drops_only_selector() {
    assert_eq!(passthrough_query(None), None);
    assert_eq!(passthrough_query(Some("")), None);
    assert_eq!(passthrough_query(Some("selector=engine%3Dsglang")), None);
    assert_eq!(
        passthrough_query(Some("mode=keep&selector=engine%3Dsglang&level=2")).as_deref(),
        Some("mode=keep&level=2")
    );
    assert_eq!(
        passthrough_query(Some("tags=weights&tags=kv_cache")).as_deref(),
        Some("tags=weights&tags=kv_cache")
    );
    assert_eq!(
        passthrough_query(Some("selectorx=1")).as_deref(),
        Some("selectorx=1")
    );
    assert_eq!(passthrough_query(Some("flag")).as_deref(), Some("flag"));
}
