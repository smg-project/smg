//! Allocation calls made while converting a pushed load record.

#![cfg(feature = "proto")]
#![expect(
    unsafe_code,
    reason = "count allocation calls through the test allocator"
)]
#![expect(clippy::print_stderr, reason = "report the measured allocation count")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use smg_worker::{
    common_proto::{EngineLoad, NamedValue},
    worker::SchedulerLoadSnapshot,
};

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the caller's layout is passed through to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the system allocator returned this pointer for this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the caller upholds the system allocator's reallocation contract.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn an_unnamed_record_converts_without_allocating() {
    let record = EngineLoad {
        running_requests: 13,
        num_used_tokens: Some(493),
        max_total_num_tokens: Some(8_317),
        ..Default::default()
    };
    let before = CALLS.with(Cell::get);
    let snapshot = SchedulerLoadSnapshot::from(std::hint::black_box(&record));
    let calls = CALLS.with(Cell::get) - before;
    assert_eq!(snapshot.num_running_reqs, 13);
    eprintln!("allocations per unnamed record conversion = {calls}");
    assert_eq!(calls, 0);
}

/// The load monitor copies a worker's report at each publish. A copy of a
/// rank's report takes the list's own block and no string: the names and
/// the units are shared.
#[test]
fn a_copy_of_a_named_report_allocates_its_list_alone() {
    let value = |name: &str, unit: &str, value: f64| NamedValue {
        name: name.into(),
        unit: unit.into(),
        value,
        age_ms: None,
    };
    let record = EngineLoad {
        running_requests: 13,
        values: vec![
            value("queue_age", "ms", 6.5),
            value("read_rate", "items/s", 0.125),
        ],
        ..Default::default()
    };
    let snapshot = SchedulerLoadSnapshot::from(&record);
    assert_eq!(snapshot.values.len(), 2);
    let before = CALLS.with(Cell::get);
    let copy = std::hint::black_box(&snapshot).clone();
    let calls = CALLS.with(Cell::get) - before;
    assert_eq!(copy.values.len(), 2);
    eprintln!("allocations per copy of a report of two named values = {calls}");
    assert_eq!(calls, 1);
}
