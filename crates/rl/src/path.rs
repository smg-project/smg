//! Engine path and query-string handling for proxied calls.

use crate::error::RlError;

/// Maximum path segments accepted for an engine route (`inference/v1/generate` is 3).
pub const MAX_SEGMENTS: usize = 4;

/// Normalize and validate an engine route path.
pub fn validate_engine_path(raw: &str) -> Result<String, RlError> {
    let path = raw.strip_prefix('/').unwrap_or(raw);
    if path.is_empty() {
        return Err(RlError::InvalidEnginePath("path is empty".to_string()));
    }
    if path.contains('?') || path.contains('#') {
        return Err(RlError::InvalidEnginePath(
            "path must not contain `?` or `#`".to_string(),
        ));
    }
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() > MAX_SEGMENTS {
        return Err(RlError::InvalidEnginePath(format!(
            "path has more than {MAX_SEGMENTS} segments"
        )));
    }
    for seg in &segments {
        if seg.is_empty() {
            return Err(RlError::InvalidEnginePath(
                "path has an empty segment".to_string(),
            ));
        }
        if *seg == ".." || *seg == "." {
            return Err(RlError::InvalidEnginePath(
                "path must not contain `.` or `..` segments".to_string(),
            ));
        }
        if !seg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return Err(RlError::InvalidEnginePath(format!(
                "segment `{seg}` contains characters outside [A-Za-z0-9._-]"
            )));
        }
    }
    Ok(path.to_string())
}

/// Forward the caller's query string minus SMG-owned keys, without re-encoding.
pub fn passthrough_query(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let kept: Vec<&str> = raw
        .split('&')
        .filter(|pair| !pair.is_empty())
        .filter(|pair| {
            let key = pair.split('=').next().unwrap_or(pair);
            key != "selector"
        })
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join("&"))
    }
}

#[cfg(test)]
mod tests {
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
}
