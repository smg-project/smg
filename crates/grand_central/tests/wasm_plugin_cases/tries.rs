//! The rules the cases share, held by cases of their own: a load the host
//! ended is made again and one refused for another reason is made once, and
//! a try reads the host's lines since it began.

use std::sync::Arc;

use smg_grand_central::plugins::wasm::LoadError;

use crate::{
    loaded, loaded_by, runtime, said, spec,
    support::wasm_guest::{self as guest, mode},
    the_frame_ended_the_rebuild, tried, LOADS,
};

/// An `init` the host ended at the frame's deadline, as a load says it.
fn ended_at_the_frame() -> LoadError {
    LoadError::Trap {
        name: "held".to_owned(),
        reason: "wasm trap: interrupt".to_owned(),
    }
}

/// A load that must succeed is made again after an `init` the host ended at
/// the frame, with no witness asked. Here every load but the last one
/// allowed is handed back as such an end, and the last hands back the
/// guest's own load. That load is made ahead, by the rule: made here as the
/// last one allowed, an `init` the host ended would have no load behind it.
#[test]
fn a_load_the_host_ended_is_made_again() {
    let runtime = runtime();
    let own = loaded(
        &runtime,
        spec("held", guest::record_guest(), mode::LEAST_IN_FLIGHT),
    );
    let mut made = 0;
    let plugin = loaded_by(|| {
        made += 1;
        if made < LOADS {
            return Err(ended_at_the_frame());
        }
        Ok(Arc::clone(&own))
    });
    assert_eq!(made, LOADS);
    assert!(Arc::ptr_eq(&plugin, &own), "the last load's own answer");
    assert_eq!(plugin.stats().traps, 0);
}

/// The last load asserts what it says: a host that ends every `init` fails
/// the case, at the last load allowed.
#[test]
fn a_host_that_ends_every_init_fails_the_load() {
    let mut made = 0;
    let load = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loaded_by(|| {
            made += 1;
            Err(ended_at_the_frame())
        })
    }));
    assert!(load.is_err(), "a load the host ended every time stood");
    assert_eq!(made, LOADS);
}

/// A load refused for any other reason is the case's failure at once: it is
/// made one time.
#[test]
fn a_load_refused_for_another_reason_is_made_once() {
    let mut made = 0;
    let load = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loaded_by(|| {
            made += 1;
            Err(LoadError::Runtime("refused".to_owned()))
        })
    }));
    assert!(load.is_err(), "a refused load stood");
    assert_eq!(made, 1);
}

/// A try reads the lines the host wrote since it began: an ended rebuild
/// that an earlier try's line tells of is not this try's, so the second
/// try here reads none.
#[test]
fn a_try_reads_the_hosts_lines_since_it_began() {
    const NAME: &str = "a-name-no-case-loads";
    let ended =
        format!("plugin=\"{NAME}\" the rebuild of a dropped instance failed: wasm trap: interrupt");
    tried(|made| {
        if made == 1 {
            said().lock().expect("the kept lines").push(ended.clone());
            assert!(
                the_frame_ended_the_rebuild(NAME),
                "the first try's own line"
            );
            return false;
        }
        assert!(
            !the_frame_ended_the_rebuild(NAME),
            "a line of the first try, read in the second"
        );
        true
    });
}
