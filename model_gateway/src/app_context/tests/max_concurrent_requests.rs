//! The three readings of `max_concurrent_requests`: off, the default and a value.

use super::super::{default_max_concurrent_requests, effective_max_concurrent_requests};

/// The three readings of `max_concurrent_requests`, pinned in one place
/// for the admission bucket and the start-up checks alike.
#[test]
fn the_sentinels_resolve_to_off_the_default_and_the_value() {
    assert_eq!(effective_max_concurrent_requests(0), None);
    assert_eq!(
        effective_max_concurrent_requests(-1),
        Some(default_max_concurrent_requests())
    );
    assert_eq!(effective_max_concurrent_requests(7), Some(7));
}
