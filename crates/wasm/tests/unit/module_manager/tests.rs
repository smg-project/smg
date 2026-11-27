use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use tokio::runtime::Builder;

use super::*;
use crate::{
    module::{MiddlewareAttachPoint, WasmModuleMeta},
    spec::smg::gateway::middleware_types::Request,
};

fn on_request() -> WasmModuleAttachPoint {
    WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest)
}

/// A registered module whose bytes are no component: the runtime answers
/// its execution with an error, after the manager's own work is done.
fn a_module(module_uuid: Uuid) -> WasmModule {
    WasmModule {
        module_uuid,
        module_meta: WasmModuleMeta {
            name: "a-module".to_string(),
            file_path: String::new(),
            sha256_hash: [7; 32],
            size_bytes: 15,
            created_at: 1,
            last_accessed_at: 0,
            access_count: 0,
            attach_points: vec![on_request()],
            wasm_bytes: Arc::new(b"not a component".to_vec()),
        },
    }
}

fn a_request() -> WasmComponentInput {
    WasmComponentInput::MiddlewareRequest(Request {
        method: "GET".to_string(),
        path: "/".to_string(),
        query: String::new(),
        headers: Vec::new(),
        body: Vec::new(),
        request_id: "r-1".to_string(),
        now_epoch_ms: 0,
    })
}

/// The five counters the route reports, after one success and one
/// failure: the totals, the running maximum, and the outcome split.
#[test]
fn the_execution_counters_add_up_over_a_success_and_a_failure() {
    let manager = WasmModuleManager::with_default_config();
    assert_eq!(manager.get_metrics(), (0, 0, 0, 0, 0));
    manager.record_execution(7, true);
    manager.record_execution(3, false);
    assert_eq!(manager.get_metrics(), (2, 1, 1, 10, 7));
}

/// Every reader of the table (one module, the list, the list by attach
/// point) sees the stamp and the count an execution bumps.
#[tokio::test]
async fn the_readers_see_the_counters_an_execution_bumps() {
    let manager = WasmModuleManager::with_default_config();
    let module_uuid = Uuid::now_v7();
    assert!(manager
        .register_module_internal(a_module(module_uuid))
        .is_ok());
    for _ in 0..2 {
        let answer = manager
            .execute_module_interface(module_uuid, on_request(), a_request())
            .await;
        assert!(
            answer.is_err(),
            "bytes that are no component execute with an error"
        );
    }
    let seen = manager
        .get_module(module_uuid)
        .ok()
        .flatten()
        .map(|module| {
            (
                module.module_meta.access_count,
                module.module_meta.last_accessed_at,
            )
        });
    assert!(matches!(seen, Some((2, stamp)) if stamp > 0), "{seen:?}");
    let listed: Vec<u64> = manager
        .get_modules()
        .ok()
        .into_iter()
        .flatten()
        .map(|module| module.module_meta.access_count)
        .collect();
    assert_eq!(listed, vec![2]);
    let by_point: Vec<u64> = manager
        .get_modules_by_attach_point(on_request())
        .ok()
        .into_iter()
        .flatten()
        .map(|module| module.module_meta.access_count)
        .collect();
    assert_eq!(by_point, vec![2]);
}

/// An execution does not wait for the table's readers: the two counters
/// it bumps live beside the record as atomics, not behind the write lock.
/// The test holds a read guard on the table while an execution runs on
/// another thread; the execution reaches the runtime (whose answer, for
/// bytes that are no component, is an error) and finishes while the guard
/// is held. Before this change the execution took the write lock for its
/// two counters and finished only after the reader let go.
#[expect(
    clippy::print_stderr,
    reason = "the numbers a pin reads are printed for the Numbers table"
)]
#[test]
fn an_execution_finishes_while_a_reader_holds_the_module_table() {
    let manager = Arc::new(WasmModuleManager::with_default_config());
    let module_uuid = Uuid::now_v7();
    assert!(manager
        .register_module_internal(a_module(module_uuid))
        .is_ok());
    let reader = manager.modules.read().ok();
    assert!(reader.is_some(), "the test holds the table's read lock");
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();
    let runner = {
        let manager = Arc::clone(&manager);
        thread::spawn(move || {
            let runtime = Builder::new_current_thread().enable_all().build().ok()?;
            let answer = runtime.block_on(manager.execute_module_interface(
                module_uuid,
                on_request(),
                a_request(),
            ));
            let _ = tx.send(started.elapsed());
            Some(answer.is_err())
        })
    };
    let bound = Duration::from_secs(5);
    let while_held = rx.recv_timeout(bound).ok();
    drop(reader);
    let errored = runner.join().ok().flatten();
    let after_release = if while_held.is_none() {
        rx.recv().ok()
    } else {
        None
    };
    assert_eq!(
        errored,
        Some(true),
        "bytes that are no component execute with an error"
    );
    match while_held {
        Some(elapsed) => eprintln!(
            "pin: an execution finished after {} ms while a reader held the module table",
            elapsed.as_millis()
        ),
        None => eprintln!(
            "pin: the execution did not finish within {} ms while a reader held the module table; it finished after {} ms in all, once the reader released",
            bound.as_millis(),
            after_release.map_or(0, |elapsed| elapsed.as_millis())
        ),
    }
    assert!(
        while_held.is_some(),
        "the execution waited for the module table's reader"
    );
}

/// The stamp never moves backward: two executions that sampled the clock
/// in one order and stored in the other leave the later stamp in place,
/// and the count has both.
#[test]
fn the_last_access_stamp_keeps_the_latest_of_two_executions() {
    let entry = ModuleEntry::new(a_module(Uuid::now_v7()));
    entry.touch(20);
    entry.touch(10);
    let seen = entry.snapshot();
    assert_eq!(
        (
            seen.module_meta.last_accessed_at,
            seen.module_meta.access_count
        ),
        (20, 2)
    );
}
