//! An allocation pin of the fetch tracker: what pushing the parts of one request allocates,
//! held where it stands, so a per-part rendering of the settings that comes back fails the
//! test. A separate executable with one test, so it can declare the counting allocator and so
//! the count is this test's own; the pushes run on a current-thread runtime.
//!
//! `cargo test -p llm-multimodal --test tracker_alloc_pins -- --nocapture` prints the count.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

mod common;

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use common::counting_alloc::{Counting, CALLS};
use llm_multimodal::{
    types::{MediaContentPart, Modality},
    AsyncMultiModalTracker, MediaConnector, MediaConnectorConfig,
};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes.
fn counted<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.load(Ordering::Relaxed);
    let value = work();
    (value, CALLS.load(Ordering::Relaxed) - before)
}

fn connector() -> MediaConnector {
    MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            fetch_timeout: Duration::from_secs(30),
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector")
}

/// A one-pixel PNG as a data URL: the smallest payload a part can carry, so what the tracker
/// allocates per part is the tracker's own.
const TINY_PNG_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

/// How many parts of one image a request pushes through the tracker: every part after the
/// first is the same media under the same settings, so it joins the first's fetch.
const REPEATED_PARTS: usize = 64;

/// What the tracker may allocate per repeated part: the fetch source it compares against the
/// first's, and the amortized growth of the per-modality lists. The settings key is a value
/// compared field by field, not a rendering of the settings built per part.
const CALLS_PER_REPEATED_PART: usize = 2;

/// The allocations the first part's fetch may take: the source, the slot, the task and its
/// future, the per-modality lists.
const FIRST_PART_BOUND: usize = 32;

#[test]
fn the_tracker_keys_a_repeated_part_without_allocating_for_its_settings() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let connector = Arc::new(connector());
    let parts: Vec<MediaContentPart> = (0..REPEATED_PARTS)
        .map(|_| MediaContentPart::ImageUrl {
            url: TINY_PNG_URL.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        })
        .collect();
    let (tracker, calls) = counted(|| {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&connector));
        runtime.block_on(async {
            for part in parts {
                tracker.push_part(part).expect("part");
            }
        });
        tracker
    });
    let output = runtime.block_on(tracker.finalize()).expect("finalize");
    assert_eq!(
        output.data.get(&Modality::Image).map_or(0, Vec::len),
        REPEATED_PARTS
    );
    println!(
        "tracker | {REPEATED_PARTS} parts of one image | {calls} calls | {:.1} per part",
        calls as f64 / REPEATED_PARTS as f64
    );
    // The first part's own fetch (its task, its future) is a fixed cost; every later part is
    // held to the bound.
    let repeated = REPEATED_PARTS - 1;
    assert!(
        calls <= FIRST_PART_BOUND + repeated * CALLS_PER_REPEATED_PART,
        "{calls} allocation calls for {REPEATED_PARTS} parts of one image: the tracker \
         allocates for a repeated part's settings again"
    );
}
