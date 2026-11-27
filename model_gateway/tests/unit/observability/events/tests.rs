use std::mem::size_of;

use super::*;

#[test]
fn test_event_sizes() {
    assert_eq!(size_of::<RequestReceivedEvent>(), 0);
    assert_eq!(size_of::<RequestSentEvent>(), 16);
    assert_eq!(size_of::<RequestPDSentEvent>(), 32);
}
