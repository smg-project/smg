use smg_external_router::metrics::BoundedLabels;
use smg_worker::metrics::writers::interner_size;

use super::*;
use crate::observability::metrics::test_support::render_with_recorder;

/// Forty distinct tenant keys at a cap of 32 render 32 tenant series of
/// their own and one `other` series carrying the rest; the process-wide
/// interner, which never frees, does not grow with them.
#[test]
fn tenants_past_the_cap_count_under_other_and_grow_no_interner() {
    let labels = BoundedLabels::new(32);
    let interned_before = interner_size();
    let rendered = render_with_recorder(|| {
        for tenant in 0..40 {
            record_unknown_priority(labels.intern(&format!("header:tenant-{tenant}")));
        }
    });
    let series: Vec<&str> = rendered
        .lines()
        .filter(|line| line.starts_with(&format!("{UNKNOWN_PRIORITY_TOTAL}{{")))
        .collect();
    assert_eq!(series.len(), 33, "32 tenants and the sentinel:\n{rendered}");
    let other = series
        .iter()
        .find(|line| line.contains("tenant=\"other\""))
        .expect("the sentinel series");
    assert!(other.ends_with(" 8"), "the eight past the cap: {other}");
    // Other tests of the process intern labels of their own meanwhile;
    // forty tenant keys through the never-freed interner would be forty.
    assert!(
        interner_size().saturating_sub(interned_before) < 40,
        "the tenant keys never reach the process-wide interner"
    );
}
