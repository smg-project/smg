use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use smg_data_connector::{MemoryResponseStorage, ResponseStorageError};
use tokio::sync::oneshot;

use super::*;
use crate::routers::grpc::common::responses::test_support::completed_response;

fn request() -> ResponsesRequest {
    serde_json::from_value(json!({
        "model": "m",
        "input": "write a story",
        "background": true,
        "max_output_tokens": 64,
        "user": "user-1"
    }))
    .unwrap()
}

fn storage() -> Arc<dyn ResponseStorage> {
    Arc::new(MemoryResponseStorage::new())
}

async fn stored_status(storage: &Arc<dyn ResponseStorage>, id: &str) -> Value {
    stored_raw(storage, id).await["status"].clone()
}

async fn stored_raw(storage: &Arc<dyn ResponseStorage>, id: &str) -> Value {
    storage
        .get_response(&ResponseId::from(id))
        .await
        .unwrap()
        .map(|s| s.raw_response)
        .unwrap_or(Value::Null)
}

async fn store_queued(storage: &Arc<dyn ResponseStorage>, id: &str) -> ResponsesRequest {
    store_queued_for(storage, id, None).await
}

async fn store_queued_for(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    tenant: Option<&str>,
) -> ResponsesRequest {
    let request = request();
    store_record(
        storage,
        &queued_response(id, &request),
        &request,
        tenant,
        None,
    )
    .await
    .unwrap();
    request
}

async fn wait_for_status(storage: &Arc<dyn ResponseStorage>, id: &str, status: &str) {
    for _ in 0..400 {
        if stored_status(storage, id).await == status {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the record never reached {status}");
}

/// Reports when the work it is held by is dropped.
struct DropFlag(Option<oneshot::Sender<()>>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

/// Work that never ends on its own; the receiver fires when it is dropped.
fn pending_work() -> (
    impl Future<Output = Result<(), Value>>,
    oneshot::Receiver<()>,
) {
    let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
    let flag = DropFlag(Some(dropped_tx));
    let work = async move {
        let _flag = flag;
        std::future::pending::<()>().await;
        Ok(())
    };
    (work, dropped_rx)
}

/// A memory store that counts its writes.
struct CountingStorage {
    inner: MemoryResponseStorage,
    stores: AtomicUsize,
}

impl CountingStorage {
    fn new() -> Self {
        Self {
            inner: MemoryResponseStorage::new(),
            stores: AtomicUsize::new(0),
        }
    }

    fn stores(&self) -> usize {
        self.stores.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ResponseStorage for CountingStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        self.stores.fetch_add(1, Ordering::SeqCst);
        self.inner.store_response(response).await
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        let outcome = self.inner.store_response_if_open(response).await?;
        if matches!(outcome, ConditionalStore::Stored) {
            self.stores.fetch_add(1, Ordering::SeqCst);
        }
        Ok(outcome)
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        let outcome = self
            .inner
            .update_response_object_if_open(id, raw_response)
            .await?;
        if matches!(outcome, ConditionalStore::Stored) {
            self.stores.fetch_add(1, Ordering::SeqCst);
        }
        Ok(outcome)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        self.inner.get_response(response_id).await
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        self.inner.delete_response(response_id).await
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        self.inner
            .list_identifier_responses(identifier, tenant, limit)
            .await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        self.inner
            .delete_identifier_responses(identifier, tenant)
            .await
    }
}

/// A memory store that answers the next `failures` reads and writes of
/// responses with a storage error, then works again; it counts its calls.
struct FlakyStorage {
    inner: MemoryResponseStorage,
    failures_left: AtomicUsize,
    calls: AtomicUsize,
    stores_landed: AtomicUsize,
    /// Answers every read with no record, as a store without history does.
    hides_records: AtomicBool,
    /// Fails every read while the writes go through, as a read that
    /// times out does.
    reads_fail: AtomicBool,
}

impl FlakyStorage {
    fn new() -> Self {
        Self {
            inner: MemoryResponseStorage::new(),
            failures_left: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            stores_landed: AtomicUsize::new(0),
            hides_records: AtomicBool::new(false),
            reads_fail: AtomicBool::new(false),
        }
    }

    fn hide_records(&self) {
        self.hides_records.store(true, Ordering::SeqCst);
    }

    fn fail_reads(&self) {
        self.reads_fail.store(true, Ordering::SeqCst);
    }

    fn fail_next(&self, failures: usize) {
        self.failures_left.store(failures, Ordering::SeqCst);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn stores_landed(&self) -> usize {
        self.stores_landed.load(Ordering::SeqCst)
    }

    fn unreachable(&self) -> Result<(), ResponseStorageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let left = self.failures_left.load(Ordering::SeqCst);
        if left > 0 {
            self.failures_left.store(left - 1, Ordering::SeqCst);
            return Err(ResponseStorageError::Unavailable(
                "connection refused".to_string(),
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl ResponseStorage for FlakyStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        self.unreachable()?;
        self.stores_landed.fetch_add(1, Ordering::SeqCst);
        self.inner.store_response(response).await
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        self.unreachable()?;
        if self.reads_fail.load(Ordering::SeqCst) {
            return Err(ResponseStorageError::Timeout("read timed out".to_string()));
        }
        if self.hides_records.load(Ordering::SeqCst) {
            return Ok(None);
        }
        self.inner.get_response(response_id).await
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        self.inner.delete_response(response_id).await
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        self.inner
            .list_identifier_responses(identifier, tenant, limit)
            .await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        self.inner
            .delete_identifier_responses(identifier, tenant)
            .await
    }
}

/// A store whose conditional writes stop at a gate once the record was
/// read: the window between a transition's read and its write, held open
/// for the test.
struct GatedStorage {
    inner: Arc<CountingStorage>,
    reached: tokio::sync::Notify,
    gate: tokio::sync::Notify,
}

#[async_trait]
impl ResponseStorage for GatedStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        self.inner.store_response(response).await
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        self.reached.notify_one();
        self.gate.notified().await;
        self.inner.store_response_if_open(response).await
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        self.reached.notify_one();
        self.gate.notified().await;
        self.inner
            .update_response_object_if_open(id, raw_response)
            .await
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        self.inner.get_response(response_id).await
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        self.inner.delete_response(response_id).await
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        self.inner
            .list_identifier_responses(identifier, tenant, limit)
            .await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        self.inner
            .delete_identifier_responses(identifier, tenant)
            .await
    }
}

/// A completed Response object under `id`, as a run's result, built from
/// the queued record.
fn completed_after_queued(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
    let mut raw = serde_json::to_value(queued_response(id, request)).unwrap();
    raw["status"] = json!("completed");
    serde_json::from_value(raw).unwrap()
}

#[tokio::test]
async fn a_run_that_ends_while_the_store_is_unreachable_is_closed_once_the_store_returns() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    // The store refuses the next three calls: the in_progress write, then
    // the terminal close's read; a work that stored nothing (its own
    // persist failed the same way) returns at once.
    flaky.fail_next(3);
    drive_watching(
        storage.clone(),
        None,
        slot,
        async { Ok(()) },
        Duration::from_millis(10),
    )
    .await;
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(
        raw["status"], "failed",
        "the record is closed once the store answers again"
    );
    assert_eq!(raw["error"]["code"], "server_error");
    assert!(
        flaky.calls() > 3,
        "the close was retried: {} calls",
        flaky.calls()
    );
}

#[tokio::test(start_paused = true)]
async fn a_background_result_that_finds_the_store_unreachable_is_stored_once_it_returns() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    let request = store_queued(&storage, "resp_1").await;
    let response = completed_after_queued("resp_1", &request);
    // Two refused calls: the is_closed read and the first persist.
    flaky.fail_next(2);
    super::super::utils::persist_response_if_needed(
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        storage.clone(),
        &response,
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert_eq!(
        stored_status(&storage, "resp_1").await,
        "completed",
        "the result landed after the retry"
    );
    assert!(
        flaky.calls() > 2,
        "the persist was retried: {} calls",
        flaky.calls()
    );
}

#[tokio::test(start_paused = true)]
async fn an_error_before_the_stream_opens_outlasts_a_store_outage() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    store_queued(&storage, "resp_1").await;
    // The store refuses the first close's read and the first retry.
    flaky.fail_next(2);
    let answer = close_before_stream(
        &storage,
        None,
        "resp_1",
        error::bad_request("bad_input", "the request was refused"),
    )
    .await;
    assert_eq!(
        answer.status(),
        StatusCode::BAD_REQUEST,
        "the answer is kept"
    );
    // Poll the store underneath the fault injection, so the test's own
    // reads do not use up the failures meant for the close and its retry.
    let mut raw = Value::Null;
    for _ in 0..2000 {
        if let Ok(Some(stored)) = flaky.inner.get_response(&ResponseId::from("resp_1")).await {
            if stored.raw_response["status"] == "failed" {
                raw = stored.raw_response;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        raw["status"], "failed",
        "the record is closed once the store answers again"
    );
    assert_eq!(raw["error"]["code"], "bad_input");
    assert_eq!(
        flaky.failures_left.load(Ordering::SeqCst),
        0,
        "the close's first read and its first retry met the injected failures"
    );
    assert!(
        flaky.calls() > 2,
        "the close was retried: {} calls",
        flaky.calls()
    );
}

#[tokio::test(start_paused = true)]
async fn a_background_result_is_not_stored_over_a_record_closed_while_the_store_was_away() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    let request = store_queued(&storage, "resp_1").await;
    let response = completed_after_queued("resp_1", &request);
    // The record is cancelled (by a client, through another gateway) while
    // the store refuses this gateway's calls.
    set_stored_status(&storage, "resp_1", "cancelled", None, None)
        .await
        .unwrap();
    let landed_before = flaky.stores_landed();
    flaky.fail_next(2);
    super::super::utils::persist_response_if_needed(
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        storage.clone(),
        &response,
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert_eq!(
        stored_status(&storage, "resp_1").await,
        "cancelled",
        "the cancel wins"
    );
    assert_eq!(
        flaky.stores_landed(),
        landed_before,
        "no late result was written"
    );
}

/// The queued record is scoped to the tenant that asked for the response.
#[tokio::test]
async fn the_queued_record_carries_the_callers_tenant() {
    let storage = storage();
    store_queued_for(&storage, "resp_t", Some("header:tenant-a")).await;
    let stored = storage
        .get_response(&ResponseId::from("resp_t"))
        .await
        .unwrap()
        .expect("stored");
    assert_eq!(stored.tenant.as_deref(), Some("header:tenant-a"));
    store_queued(&storage, "resp_n").await;
    assert_eq!(
        storage
            .get_response(&ResponseId::from("resp_n"))
            .await
            .unwrap()
            .unwrap()
            .tenant,
        None
    );
}

/// A cancel from another tenant answers as for an unknown id and stops
/// nothing: the task keeps running and the record stays open; the owner's
/// cancel then closes it.
#[tokio::test]
async fn a_cancel_from_another_tenant_answers_not_found_and_stops_nothing() {
    let storage = storage();
    store_queued_for(&storage, "resp_1", Some("header:tenant-a")).await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let token = slot.cancel_token();

    for caller in [Some("header:tenant-b"), None] {
        let answer = cancel_with(&registry, &storage, None, caller, "resp_1").await;
        assert_eq!(answer.status(), StatusCode::NOT_FOUND, "{caller:?}");
        assert!(!token.is_cancelled(), "{caller:?}: the run is untouched");
        assert_eq!(stored_status(&storage, "resp_1").await, "queued");
    }
    let missing = cancel_with(
        &registry,
        &storage,
        None,
        Some("header:tenant-b"),
        "resp_404",
    )
    .await;
    assert_eq!(
        missing.status(),
        StatusCode::NOT_FOUND,
        "the same answer as an unknown id"
    );

    // The owner's cancel fires the token; the record closes once the slot drops.
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let owner = tokio::spawn(async move {
        // The slot (the running task) ends when its token fires.
        token.cancelled().await;
        drop(slot);
    });
    let answer = cancel_with(&registry, &storage, None, Some("header:tenant-a"), "resp_1").await;
    owner.await.unwrap();
    assert_eq!(answer.status(), StatusCode::OK);
    assert_eq!(stored_status(&storage, "resp_1").await, "cancelled");
}

#[tokio::test]
async fn a_terminal_write_that_loses_the_race_to_a_close_elsewhere_is_refused() {
    // Gateway X runs the response and writes its terminal object; gateway
    // Y cancels it on the shared store between X's read and X's write.
    let counting = Arc::new(CountingStorage::new());
    let direct: Arc<dyn ResponseStorage> = counting.clone();
    store_queued(&direct, "resp_1").await;
    set_stored_status(&direct, "resp_1", "in_progress", None, None)
        .await
        .unwrap();
    let gated = Arc::new(GatedStorage {
        inner: counting.clone(),
        reached: tokio::sync::Notify::new(),
        gate: tokio::sync::Notify::new(),
    });
    let through_gate: Arc<dyn ResponseStorage> = gated.clone();
    let writes = counting.stores();

    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let x = tokio::spawn(async move {
        set_stored_status(&through_gate, "resp_1", "completed", None, None)
            .await
            .unwrap()
    });
    // X has read the open record and stands at its write.
    gated.reached.notified().await;

    // Y's cancel lands first.
    let Transition::Moved(raw) = set_stored_status(&direct, "resp_1", "cancelled", None, None)
        .await
        .unwrap()
    else {
        panic!("the open record is cancelled");
    };
    assert_eq!(raw["status"], "cancelled");

    // X's write is refused: the record stays cancelled, completed is never written.
    gated.gate.notify_one();
    let Transition::Terminal(raw) = x.await.unwrap() else {
        panic!("the late terminal write is refused");
    };
    assert_eq!(raw["status"], "cancelled");
    assert_eq!(stored_status(&direct, "resp_1").await, "cancelled");
    assert_eq!(
        counting.stores(),
        writes + 1,
        "cancelled was written once; completed never"
    );
}

#[test]
fn the_registry_admits_up_to_its_cap_and_frees_a_dropped_slot() {
    let registry = Arc::new(BackgroundResponses::new(2));
    assert!(registry.enabled());
    let first = registry.admit("resp_1").expect("first");
    let second = registry.admit("resp_2").expect("second");
    assert!(registry.admit("resp_3").is_none(), "cap reached");
    assert_eq!(registry.in_flight(), 2);
    drop(first);
    assert_eq!(registry.in_flight(), 1);
    let third = registry.admit("resp_3").expect("freed place");
    drop((second, third));
    assert_eq!(registry.in_flight(), 0);
    assert!(!BackgroundResponses::new(0).enabled());
}

#[tokio::test]
async fn cancel_fires_the_token_and_the_receiver_settles_when_the_slot_drops() {
    let registry = Arc::new(BackgroundResponses::new(8));
    assert!(registry.cancel("resp_unknown").is_none());
    let slot = registry.admit("resp_1").unwrap();
    let token = slot.cancel_token();
    let mut finished = registry.cancel("resp_1").expect("running here");
    assert!(token.is_cancelled());
    assert!(!*finished.borrow());
    drop(slot);
    finished.changed().await.unwrap();
    assert!(*finished.borrow());
    assert!(registry.cancel("resp_1").is_none(), "left the registry");
}

#[test]
fn the_queued_object_has_the_public_shape() {
    let response = queued_response("resp_q", &request());
    let wire = serde_json::to_value(&response).unwrap();
    assert_eq!(wire["id"], "resp_q");
    assert_eq!(wire["status"], "queued");
    assert_eq!(wire["background"], true);
    assert_eq!(wire["output"], json!([]));
    assert_eq!(wire["usage"], Value::Null);
    assert_eq!(wire["completed_at"], Value::Null);
    assert_eq!(wire["max_output_tokens"], 64);
    assert_eq!(wire["model"], "m");
}

#[test]
fn with_status_stamps_terminal_states_and_carries_an_error() {
    let raw = json!({"id": "resp_1", "status": "queued", "completed_at": null, "error": null});
    let in_progress = with_status(raw.clone(), "in_progress", None);
    assert_eq!(in_progress["status"], "in_progress");
    assert_eq!(in_progress["completed_at"], Value::Null);
    let cancelled = with_status(raw.clone(), "cancelled", None);
    assert_eq!(cancelled["status"], "cancelled");
    assert!(cancelled["completed_at"].is_i64());
    let failed = with_status(raw, "failed", Some(json!({"code": "x", "message": "y"})));
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["code"], "x");
}

#[tokio::test]
async fn the_queued_record_is_stored_before_the_work_and_moves_through_its_states() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let stored = storage
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .expect("pollable at once");
    assert_eq!(stored.raw_response["status"], "queued");
    assert_eq!(stored.input[0]["content"][0]["text"], "write a story");
    assert_eq!(stored.safety_identifier.as_deref(), Some("user-1"));

    let Transition::Moved(raw) = set_stored_status(&storage, "resp_1", "in_progress", None, None)
        .await
        .unwrap()
    else {
        panic!("a queued record moves to in_progress");
    };
    assert_eq!(raw["status"], "in_progress");
    assert_eq!(stored_status(&storage, "resp_1").await, "in_progress");
    assert!(matches!(
        set_stored_status(&storage, "resp_404", "cancelled", None, None)
            .await
            .unwrap(),
        Transition::Missing
    ));
}

#[tokio::test]
async fn a_terminal_record_is_never_rewritten() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let Transition::Moved(raw) = set_stored_status(&storage, "resp_1", "cancelled", None, None)
        .await
        .unwrap()
    else {
        panic!("a queued record is cancelled");
    };
    assert_eq!(raw["status"], "cancelled");

    for status in ["failed", "in_progress", "completed", "cancelled"] {
        let error = (status == "failed").then(|| json!({"code": "x", "message": "y"}));
        let Transition::Terminal(raw) = set_stored_status(&storage, "resp_1", status, error, None)
            .await
            .unwrap()
        else {
            panic!("a cancelled record stays cancelled on {status}");
        };
        assert_eq!(raw["status"], "cancelled");
    }
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "cancelled");
    assert!(raw["error"].is_null(), "{raw}");
    // A record that is gone takes no write: nothing is written under it.
    assert!(matches!(
        set_stored_status(&storage, "resp_404", "completed", None, None)
            .await
            .unwrap(),
        Transition::Missing
    ));
}

#[tokio::test]
async fn a_cancel_marks_the_record_cancelled_and_drops_the_work() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (work, dropped_rx) = pending_work();
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive(storage.clone(), None, slot, work));
    wait_for_status(&storage, "resp_1", "in_progress").await;
    let mut finished = registry.cancel("resp_1").expect("running");
    task.await.unwrap();
    dropped_rx.await.expect("the work was dropped");
    assert!(*finished.borrow_and_update());
    assert_eq!(stored_status(&storage, "resp_1").await, "cancelled");
    assert_eq!(registry.in_flight(), 0);
}

/// `DELETE /v1/responses/{id}` while the work runs, here or on another
/// gateway: the watch finds the record gone, the work is dropped and
/// nothing is written under the id again, so the delete holds.
#[tokio::test]
async fn a_record_deleted_while_the_work_runs_drops_the_work_and_stays_deleted() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (work, dropped_rx) = pending_work();
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive_watching(
        storage.clone(),
        None,
        slot,
        work,
        Duration::from_millis(10),
    ));
    wait_for_status(&storage, "resp_1", "in_progress").await;
    storage
        .delete_response(&ResponseId::from("resp_1"))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the run ends once its record is gone")
        .unwrap();
    dropped_rx.await.expect("the work was dropped");
    assert_eq!(
        stored_raw(&storage, "resp_1").await,
        Value::Null,
        "nothing is written under the deleted id"
    );
    assert_eq!(registry.in_flight(), 0);
}

/// A store that never shows the record (no history backend, or a record
/// deleted before the work began) is not read as a deletion: the work
/// runs to its end, as before, and stores nothing over the missing
/// record.
#[tokio::test]
async fn a_run_over_a_store_that_never_shows_its_record_reaches_its_end() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    store_queued(&storage, "resp_1").await;
    flaky.hide_records();
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (done_tx, done_rx) = oneshot::channel::<()>();
    let work = async move {
        // Longer than several watch ticks: a watch that read the missing
        // record as a deletion would drop this before it ends.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let _ = done_tx.send(());
        Ok(())
    };
    drive_watching(storage.clone(), None, slot, work, Duration::from_millis(10)).await;
    done_rx.await.expect("the work ran to its end");
    assert_eq!(registry.in_flight(), 0);
}

/// A store blip at the in_progress transition does not disarm the watch:
/// the run learns from its first good read that its record exists, and a
/// record deleted after that ends the run.
#[tokio::test]
async fn a_run_that_started_during_a_store_blip_still_ends_when_its_record_is_deleted() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (work, dropped_rx) = pending_work();
    // The in_progress write fails: the record stays queued and the run
    // starts without having seen it.
    flaky.fail_next(1);
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive_watching(
        storage.clone(),
        None,
        slot,
        work,
        Duration::from_millis(10),
    ));
    // A few good reads of the record, then the delete.
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(
        flaky.calls() > 2,
        "the watch read the record: {} calls",
        flaky.calls()
    );
    storage
        .delete_response(&ResponseId::from("resp_1"))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the run ends once its record is gone")
        .unwrap();
    dropped_rx.await.expect("the work was dropped");
    assert_eq!(
        stored_raw(&storage, "resp_1").await,
        Value::Null,
        "nothing is written under the deleted id"
    );
    assert_eq!(registry.in_flight(), 0);
}

/// The guard of the terminal persist is the conditional write: an open
/// record takes it, a terminal one refuses it with what it closed with, a
/// record that is gone takes nothing (the delete holds), and a store that
/// cannot be read refuses it with an error, deciding nothing.
#[tokio::test]
async fn a_record_that_is_gone_reads_closed_and_an_unreadable_store_does_not() {
    let storage = storage();
    store_queued(&storage, "resp_open").await;
    assert!(matches!(
        set_stored_status(&storage, "resp_open", "in_progress", None, None)
            .await
            .unwrap(),
        Transition::Moved(_)
    ));
    set_stored_status(&storage, "resp_open", "cancelled", None, None)
        .await
        .unwrap();
    assert!(matches!(
        set_stored_status(&storage, "resp_open", "completed", None, None)
            .await
            .unwrap(),
        Transition::Terminal(_)
    ));
    assert!(
        matches!(
            set_stored_status(&storage, "resp_gone", "completed", None, None)
                .await
                .unwrap(),
            Transition::Missing
        ),
        "a missing record is closed: nothing is written under it"
    );

    let flaky = Arc::new(FlakyStorage::new());
    let flaky_storage: Arc<dyn ResponseStorage> = flaky.clone();
    store_queued(&flaky_storage, "resp_1").await;
    flaky.fail_next(1);
    assert!(
        set_stored_status(&flaky_storage, "resp_1", "in_progress", None, None)
            .await
            .is_err(),
        "a store that cannot be read refuses the write and decides nothing"
    );
    assert!(matches!(
        set_stored_status(&flaky_storage, "resp_1", "in_progress", None, None)
            .await
            .unwrap(),
        Transition::Moved(_)
    ));
}

#[tokio::test]
async fn a_failed_work_leaves_a_failed_record_with_its_error() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    drive(storage.clone(), None, slot, async {
        Err(json!({"code": "model_not_found", "message": "no such model"}))
    })
    .await;
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["error"]["code"], "model_not_found");
    assert!(raw["completed_at"].is_i64());
}

#[tokio::test]
async fn a_work_that_stored_its_result_is_left_as_it_ended() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let store = storage.clone();
    drive(storage.clone(), None, slot, async move {
        set_stored_status(&store, "resp_1", "completed", None, None)
            .await
            .unwrap();
        Ok(())
    })
    .await;
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "completed");
    assert!(raw["error"].is_null(), "{raw}");
}

#[tokio::test]
async fn a_work_that_ends_without_a_stored_result_leaves_a_failed_record() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    // A stream whose transform ended without a terminal event and object.
    drive(storage.clone(), None, slot, async { Ok(()) }).await;
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["error"]["code"], "server_error");
    assert!(raw["completed_at"].is_i64());
    assert_eq!(registry.in_flight(), 0);
}

fn explode() -> Result<(), Value> {
    panic!("the transform exploded")
}

#[tokio::test]
async fn a_panicking_work_leaves_a_failed_record() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    drive(storage.clone(), None, slot, async { explode() }).await;
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["error"]["code"], "server_error");
    assert!(
        raw["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("the transform exploded")),
        "{raw}"
    );
    assert_eq!(
        registry.in_flight(),
        0,
        "the slot is released after a panic"
    );
}

#[tokio::test]
async fn a_cancel_on_a_gateway_that_does_not_run_the_response_closes_it_once_and_stops_the_owner() {
    // Two gateways over one store: `owner` runs the response, `other`
    // receives the cancel and has no task for the id.
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn ResponseStorage> = counting.clone();
    store_queued(&storage, "resp_1").await;
    let owner = Arc::new(BackgroundResponses::new(4));
    let other = Arc::new(BackgroundResponses::new(4));
    let slot = owner.admit("resp_1").unwrap();
    let (work, dropped_rx) = pending_work();
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive_watching(
        storage.clone(),
        None,
        slot,
        work,
        Duration::from_millis(20),
    ));
    wait_for_status(&storage, "resp_1", "in_progress").await;

    // The cancel on the other gateway: no task there, the record is closed.
    assert!(other.cancel("resp_1").is_none());
    let writes = counting.stores();
    let Transition::Moved(raw) = set_stored_status(&storage, "resp_1", "cancelled", None, None)
        .await
        .unwrap()
    else {
        panic!("the in-flight record is closed");
    };
    assert_eq!(raw["status"], "cancelled");

    // The owner reads the closed record, drops the work and writes nothing.
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the owner stops within its watch interval")
        .unwrap();
    dropped_rx.await.expect("the work was dropped");
    assert_eq!(stored_status(&storage, "resp_1").await, "cancelled");
    assert_eq!(counting.stores(), writes + 1, "cancelled was written once");
    assert_eq!(owner.in_flight(), 0);
}

#[tokio::test]
async fn a_record_closed_before_the_work_starts_is_not_run() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    set_stored_status(&storage, "resp_1", "cancelled", None, None)
        .await
        .unwrap();
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
    let flag = DropFlag(Some(dropped_tx));
    drive(storage.clone(), None, slot, async move {
        let _flag = flag;
        explode()
    })
    .await;
    dropped_rx.await.expect("the work was dropped unrun");
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "cancelled");
    assert!(raw["error"].is_null(), "{raw}");
    assert_eq!(registry.in_flight(), 0);
}

#[tokio::test]
async fn an_error_before_the_stream_opens_closes_the_record_and_keeps_the_answer() {
    let storage = storage();
    store_queued(&storage, "resp_1").await;
    let answer = close_before_stream(
        &storage,
        None,
        "resp_1",
        error::not_found("response_not_found", "no such previous response"),
    )
    .await;
    assert_eq!(answer.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(answer.into_body(), 1 << 20).await.unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "response_not_found");
    let raw = stored_raw(&storage, "resp_1").await;
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["error"]["code"], "response_not_found");
    assert_eq!(raw["error"]["message"], "no such previous response");

    // A stream that opened passes through untouched.
    store_queued(&storage, "resp_2").await;
    let answer = close_before_stream(
        &storage,
        None,
        "resp_2",
        (StatusCode::OK, "event: response.created").into_response(),
    )
    .await;
    assert_eq!(answer.status(), StatusCode::OK);
    assert_eq!(stored_status(&storage, "resp_2").await, "queued");
}

#[tokio::test]
async fn the_error_object_is_read_from_the_error_answer() {
    let response = error::bad_request("convert_request_failed", "bad input");
    let error = error_object(response).await;
    assert_eq!(error["code"], "convert_request_failed");
    assert_eq!(error["message"], "bad input");
    let plain = (StatusCode::BAD_GATEWAY, "upstream gone").into_response();
    let error = error_object(plain).await;
    assert_eq!(error["code"], "server_error");
}

#[test]
fn the_cancel_answer_follows_the_stored_status() {
    let cancelled = answer_for_closed(json!({"status": "cancelled", "background": true}));
    assert_eq!(cancelled.status(), StatusCode::OK);
    for status in ["completed", "incomplete", "failed", "in_progress"] {
        let answer = answer_for_closed(json!({"status": status}));
        assert_eq!(answer.status(), StatusCode::BAD_REQUEST, "{status}");
    }
}

/// The request of the public API's probe: `background: true` with
/// `store: false`.
fn unstored_request() -> ResponsesRequest {
    serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "background": true,
        "store": false,
        "max_output_tokens": 64
    }))
    .unwrap()
}

/// With background mode off the sweeper still runs once at start-up, for
/// the records an earlier gateway left, and then ends.
#[tokio::test]
async fn the_sweeper_runs_once_with_background_mode_off() {
    let storage = storage();
    let mut left = StoredResponse::new(None);
    left.id = ResponseId::from("resp_left");
    left.raw_response = json!({
        "id": "resp_left",
        "store": false,
        "status": "completed",
        "completed_at": chrono::Utc::now().timestamp() - 3 * UNSTORED_RETENTION.as_secs() as i64,
    });
    storage.store_response(left).await.unwrap();

    spawn_unstored_sweeper(Arc::clone(&storage), false)
        .await
        .expect("the one-shot sweeper ends");

    assert!(storage
        .get_response(&ResponseId::from("resp_left"))
        .await
        .unwrap()
        .is_none());
}

/// The sweep takes the unstored records whose window ended and leaves a
/// fresh one, a stored one and one still running.
#[tokio::test]
async fn the_sweep_retires_the_unstored_records_left_past_their_window() {
    use smg_data_connector::{MemoryResponseStorage, StoredResponse};

    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let now = chrono::Utc::now().timestamp();
    let past = now - i64::try_from(UNSTORED_RETENTION.as_secs()).unwrap() - 60;
    for (id, raw) in [
        (
            "resp_old_unstored",
            json!({"status": "completed", "store": false, "completed_at": past}),
        ),
        (
            "resp_fresh_unstored",
            json!({"status": "completed", "store": false, "completed_at": now}),
        ),
        (
            "resp_old_stored",
            json!({"status": "completed", "store": true, "completed_at": past}),
        ),
        (
            "resp_old_running_unstored",
            json!({"status": "in_progress", "store": false}),
        ),
    ] {
        let mut stored = StoredResponse::new(None);
        stored.id = ResponseId::from(id);
        stored.raw_response = raw;
        storage.store_response(stored).await.unwrap();
    }

    assert_eq!(sweep_unstored(&storage).await, 1);

    assert!(storage
        .get_response(&ResponseId::from("resp_old_unstored"))
        .await
        .unwrap()
        .is_none());
    for kept in [
        "resp_fresh_unstored",
        "resp_old_stored",
        "resp_old_running_unstored",
    ] {
        assert!(
            storage
                .get_response(&ResponseId::from(kept))
                .await
                .unwrap()
                .is_some(),
            "{kept}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn an_unstored_background_response_is_polled_through_its_window_then_retired() {
    let storage = storage();
    let request = unstored_request();
    store_record(
        &storage,
        &queued_response("resp_u", &request),
        &request,
        None,
        None,
    )
    .await
    .unwrap();
    let raw = stored_raw(&storage, "resp_u").await;
    assert_eq!(raw["status"], "queued");
    assert_eq!(raw["background"], true);
    assert_eq!(raw["store"], false);

    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_u").unwrap();
    let store = storage.clone();
    drive(storage.clone(), None, slot, async move {
        finish_unstored(
            &store,
            &completed_response("resp_u", &unstored_request()),
            None,
        )
        .await;
        Ok(())
    })
    .await;
    let stored = storage
        .get_response(&ResponseId::from("resp_u"))
        .await
        .unwrap()
        .expect("the result is polled from the record");
    assert_eq!(
        stored.raw_response["status"], "completed",
        "{}",
        stored.raw_response
    );
    assert_eq!(stored.raw_response["store"], false);
    assert_eq!(
        stored.raw_response["output"][0]["content"][0]["text"],
        "hello"
    );
    assert_eq!(stored.input[0]["content"][0]["text"], "hi");
    assert_eq!(registry.in_flight(), 0);

    tokio::time::sleep(UNSTORED_RETENTION - Duration::from_secs(1)).await;
    assert_eq!(
        stored_status(&storage, "resp_u").await,
        "completed",
        "polled inside the window"
    );
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        stored_raw(&storage, "resp_u").await,
        Value::Null,
        "retired after the window"
    );
}

#[tokio::test(start_paused = true)]
async fn a_closed_unstored_record_is_retired_and_every_other_record_is_kept() {
    let storage = storage();
    let request = unstored_request();
    for id in ["resp_cancelled", "resp_running"] {
        store_record(
            &storage,
            &queued_response(id, &request),
            &request,
            None,
            None,
        )
        .await
        .unwrap();
    }
    store_queued(&storage, "resp_stored").await;

    for id in ["resp_cancelled", "resp_stored"] {
        let Transition::Moved(raw) = set_stored_status(&storage, id, "cancelled", None, None)
            .await
            .unwrap()
        else {
            panic!("{id} is closed");
        };
        assert_eq!(raw["status"], "cancelled");
    }
    assert!(matches!(
        set_stored_status(&storage, "resp_running", "in_progress", None, None)
            .await
            .unwrap(),
        Transition::Moved(_)
    ));
    // A result that arrives after the cancel does not revive the record.
    finish_unstored(
        &storage,
        &completed_response("resp_cancelled", &request),
        None,
    )
    .await;
    assert_eq!(stored_status(&storage, "resp_cancelled").await, "cancelled");

    tokio::time::sleep(UNSTORED_RETENTION + Duration::from_secs(1)).await;
    assert_eq!(
        stored_raw(&storage, "resp_cancelled").await,
        Value::Null,
        "the closed unstored record is retired"
    );
    assert_eq!(
        stored_status(&storage, "resp_running").await,
        "in_progress",
        "a record still running is kept"
    );
    assert_eq!(
        stored_status(&storage, "resp_stored").await,
        "cancelled",
        "a stored record is kept"
    );
}

/// The work's persistence stores the terminal object first and links the
/// response's items to its conversation after it (two or more store
/// round trips later). A watch tick between the two reads `completed`:
/// the object is the work's own, the work is not dropped there and
/// finishes its linking.
#[tokio::test]
async fn a_work_finishes_its_linking_after_the_watch_reads_its_own_result() {
    let storage = storage();
    let request = store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    // The linking waits at this gate until the watch has read the result.
    let (gate_tx, gate_rx) = oneshot::channel::<()>();
    let (linked_tx, linked_rx) = oneshot::channel::<()>();
    let work = {
        let storage = storage.clone();
        async move {
            store_record(
                &storage,
                &completed_after_queued("resp_1", &request),
                &request,
                None,
                None,
            )
            .await
            .unwrap();
            if gate_rx.await.is_ok() {
                let _ = linked_tx.send(());
            }
            Ok(())
        }
    };
    let watch = Duration::from_millis(20);
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive_watching(storage.clone(), None, slot, work, watch));
    wait_for_status(&storage, "resp_1", "completed").await;
    // Several watch ticks read the result before the linking goes on.
    tokio::time::sleep(watch * 4).await;
    let _ = gate_tx.send(());
    task.await.unwrap();
    assert!(
        linked_rx.await.is_ok(),
        "the linking after the result's write ran to its end"
    );
    assert_eq!(stored_status(&storage, "resp_1").await, "completed");
    assert_eq!(registry.in_flight(), 0);
}

/// The run stays open on the work's own result for the terminal write
/// window at most: a work that never ends after its write is dropped at
/// the window's end, and the record keeps the result.
#[tokio::test]
async fn a_work_that_never_ends_after_its_result_is_dropped_at_the_windows_end() {
    let storage = storage();
    let request = store_queued(&storage, "resp_1").await;
    let registry = Arc::new(BackgroundResponses::new(4));
    let slot = registry.admit("resp_1").unwrap();
    let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
    let flag = DropFlag(Some(dropped_tx));
    let work = {
        let storage = storage.clone();
        async move {
            let _flag = flag;
            store_record(
                &storage,
                &completed_after_queued("resp_1", &request),
                &request,
                None,
                None,
            )
            .await
            .unwrap();
            std::future::pending::<()>().await;
            Ok(())
        }
    };
    let watch = Duration::from_millis(10);
    let started = tokio::time::Instant::now();
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let task = tokio::spawn(drive_watching(storage.clone(), None, slot, work, watch));
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the run ends at the window's end")
        .unwrap();
    dropped_rx.await.expect("the work was dropped");
    assert!(
        started.elapsed() >= terminal_write_window(watch),
        "dropped no sooner than the window: {:?}",
        started.elapsed()
    );
    assert_eq!(stored_status(&storage, "resp_1").await, "completed");
    assert_eq!(registry.in_flight(), 0);
}

/// [`CountingStorage`] behind a read path that is down: every read fails
/// while the writes, the conditional ones included, go through.
struct ReadFailingStorage(Arc<CountingStorage>);

#[async_trait]
impl ResponseStorage for ReadFailingStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        self.0.store_response(response).await
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        self.0.store_response_if_open(response).await
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        self.0
            .update_response_object_if_open(id, raw_response)
            .await
    }

    async fn get_response(
        &self,
        _response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        Err(ResponseStorageError::Timeout("read timed out".to_string()))
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        self.0.delete_response(response_id).await
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        self.0
            .list_identifier_responses(identifier, tenant, limit)
            .await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        self.0.delete_identifier_responses(identifier, tenant).await
    }
}

/// A store whose reads fail while its writes go through (a read that
/// times out, a read path that is down): the guard of the terminal
/// persist is the store's own conditional write, so a result that arrives
/// after a cancel or a delete is refused by the store, and no failed read
/// on the gateway's side passes for an open record.
#[tokio::test(start_paused = true)]
async fn a_late_result_is_refused_by_the_store_while_the_record_cannot_be_read() {
    let counting = Arc::new(CountingStorage::new());
    let direct: Arc<dyn ResponseStorage> = counting.clone();
    let storage: Arc<dyn ResponseStorage> = Arc::new(ReadFailingStorage(counting.clone()));
    let request = store_queued(&direct, "resp_1").await;
    let response = completed_after_queued("resp_1", &request);
    // Cancelled through another gateway; this gateway's reads fail from
    // here on.
    set_stored_status(&direct, "resp_1", "cancelled", None, None)
        .await
        .unwrap();
    let writes = counting.stores();
    super::super::utils::persist_response_if_needed(
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        storage.clone(),
        &response,
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert_eq!(
        stored_status(&direct, "resp_1").await,
        "cancelled",
        "the cancel wins"
    );
    assert_eq!(counting.stores(), writes, "no late result was written");

    // The record is deleted while the work runs: the result does not
    // re-create it.
    direct
        .delete_response(&ResponseId::from("resp_1"))
        .await
        .unwrap();
    super::super::utils::persist_response_if_needed(
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        storage,
        &response,
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert_eq!(
        stored_raw(&direct, "resp_1").await,
        Value::Null,
        "the delete holds"
    );
    assert_eq!(
        counting.stores(),
        writes,
        "nothing was written under the deleted id"
    );
}

/// A store without an atomic conditional write of its own (the trait's
/// default reads the record, then writes) whose reads fail while its
/// writes go through: the failed read is an error the persist retries
/// for the terminal write window and then gives up on, never an open
/// record; nothing is written while the record's state is unknown.
#[tokio::test(start_paused = true)]
async fn a_record_that_cannot_be_read_takes_no_late_result_without_an_atomic_check() {
    let flaky = Arc::new(FlakyStorage::new());
    let storage: Arc<dyn ResponseStorage> = flaky.clone();
    let request = store_queued(&storage, "resp_1").await;
    let response = completed_after_queued("resp_1", &request);
    set_stored_status(&storage, "resp_1", "cancelled", None, None)
        .await
        .unwrap();
    let landed_before = flaky.stores_landed();
    let calls_before = flaky.calls();
    flaky.fail_reads();
    super::super::utils::persist_response_if_needed(
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        storage,
        &response,
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert_eq!(
        flaky.stores_landed(),
        landed_before,
        "no late result was written over the record that could not be read"
    );
    assert!(
        flaky.calls() - calls_before > TERMINAL_WRITE_RETRIES as usize,
        "the read was retried for the terminal write window: {} calls",
        flaky.calls() - calls_before
    );
    let stored = flaky
        .inner
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .expect("the record is still there");
    assert_eq!(stored.raw_response["status"], "cancelled");
}
