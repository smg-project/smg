use super::{FrameBudget, HeaderSet};

/// A plugin that states no budget takes the default frame, clamped to
/// the request's as a stated one is: under a request with no budget,
/// which a stage is shown as the widest, under a larger frame and under
/// a smaller one. A stated budget is the plugin's own on both sides of
/// the default, and a request whose budget is gone gets no frame.
#[test]
fn a_plugin_with_no_budget_takes_the_default_frame_clamped_to_the_requests() {
    let default = FrameBudget::DEFAULT_MICROS;
    let unset = FrameBudget {
        micros: 0,
        fuel_per_micro: 10,
    };
    for (request, frame) in [
        (u32::MAX, default),
        (default + 1, default),
        (default, default),
        (500, 500),
        (0, 0),
    ] {
        assert_eq!(unset.frame_micros(request), frame, "under {request}");
        assert_eq!(
            unset.fuel_for(request),
            u64::from(frame) * 10,
            "under {request}"
        );
    }
    for own in [200, default * 3] {
        let stated = FrameBudget {
            micros: own,
            fuel_per_micro: 10,
        };
        assert_eq!(stated.frame_micros(u32::MAX), own);
        assert_eq!(stated.frame_micros(100), 100);
    }
}

#[test]
fn a_header_set_holds_positions_over_the_whole_byte_range() {
    let set = HeaderSet::of([0, 63, 64, 255]);
    assert!(set.contains(0) && set.contains(63) && set.contains(64) && set.contains(255));
    assert!(!set.contains(1) && !set.contains(200));
    assert!(!HeaderSet::NONE.contains(0));
    assert!(HeaderSet::ALL.contains(129));
}
