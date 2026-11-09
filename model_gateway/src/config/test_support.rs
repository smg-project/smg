//! What the crate's tests configure a router with: the routing modes with
//! no worker URLs and no policies.

use super::RoutingMode;

/// Regular routing with no worker URLs.
pub(crate) fn regular_routing_mode() -> RoutingMode {
    RoutingMode::Regular {
        worker_urls: vec![],
    }
}

/// Prefill-decode routing with no worker URLs and no policies.
pub(crate) fn pd_routing_mode() -> RoutingMode {
    RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    }
}

/// Encode-prefill-decode routing with no worker URLs and no policies.
pub(crate) fn epd_routing_mode() -> RoutingMode {
    RoutingMode::EncodePrefillDecode {
        encode_urls: vec![],
        prefill_urls: vec![],
        decode_urls: vec![],
        encode_policy: None,
        prefill_policy: None,
        decode_policy: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each fixture builds the mode its name says, with no worker URLs and
    /// no policies: a test reading one configures that mode and nothing else.
    #[test]
    fn each_fixture_builds_the_mode_its_name_says() {
        assert!(matches!(
            regular_routing_mode(),
            RoutingMode::Regular { worker_urls } if worker_urls.is_empty()
        ));
        assert!(matches!(
            pd_routing_mode(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: None,
                decode_policy: None,
            } if prefill_urls.is_empty() && decode_urls.is_empty()
        ));
        assert!(matches!(
            epd_routing_mode(),
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            } if encode_urls.is_empty() && prefill_urls.is_empty() && decode_urls.is_empty()
        ));
    }
}
