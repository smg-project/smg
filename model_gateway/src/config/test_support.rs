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
