//! Background mode of `POST /v1/responses` (`background: true`).
//!
//! A background response is answered before it is generated: its `queued`
//! Response object is stored under the response id and returned at once, the
//! work runs in a task of its own, and the client reads the stored object with
//! `GET /v1/responses/{id}` as it moves `queued` -> `in_progress` ->
//! `completed` | `incomplete` | `failed` | `cancelled`, or stops it with
//! `POST /v1/responses/{id}/cancel`. With `stream: true` the client keeps the
//! stream, which carries `response.queued` after `response.created`; the work
//! still runs behind the stored id and finishes (and is stored) when the client
//! leaves early.
//!
//! The registry here maps the in-flight response ids to their cancellation
//! tokens and bounds how many run at once ([`BackgroundResponses::admit`]);
//! the record helpers write the lifecycle states under the one id, which the
//! storage backends replace on a second store.

use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};

use axum::{
    body::to_bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::future::BoxFuture;
use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
use parking_lot::Mutex;
use serde_json::{json, Value};
use smg_data_connector::{
    with_request_context, RequestContext as StorageRequestContext, ResponseId, ResponseStorage,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::ResponsesContext;
use crate::routers::{
    common::{
        persistence_utils::{build_stored_response, extract_input_items},
        sse::{sse_channel, SseSender},
    },
    error,
};

/// How long a cancel request waits for the task to write its terminal record
/// before answering from whatever the store holds.
const CANCEL_SETTLE: Duration = Duration::from_secs(5);

/// The in-flight background responses of this gateway.
pub(crate) struct BackgroundResponses {
    running: Mutex<HashMap<String, Entry>>,
    max_in_flight: usize,
}

struct Entry {
    cancel: CancellationToken,
    finished: watch::Receiver<bool>,
}

impl BackgroundResponses {
    /// `max_in_flight` background responses at once; 0 switches the mode off
    /// (see [`Self::enabled`]).
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            running: Mutex::new(HashMap::new()),
            max_in_flight,
        }
    }

    /// Whether `background: true` runs in the background here at all; off,
    /// such requests run in the foreground like every other request.
    pub fn enabled(&self) -> bool {
        self.max_in_flight > 0
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    pub fn in_flight(&self) -> usize {
        self.running.lock().len()
    }

    /// Registers a response id; `None` when the cap is reached. The slot
    /// leaves the registry when it is dropped, at the end of the task.
    pub fn admit(self: &Arc<Self>, id: &str) -> Option<BackgroundSlot> {
        let cancel = CancellationToken::new();
        let (finished_tx, finished) = watch::channel(false);
        {
            let mut running = self.running.lock();
            if running.len() >= self.max_in_flight {
                return None;
            }
            running.insert(
                id.to_string(),
                Entry {
                    cancel: cancel.clone(),
                    finished,
                },
            );
        }
        Some(BackgroundSlot {
            registry: Arc::clone(self),
            id: id.to_string(),
            cancel,
            finished: finished_tx,
        })
    }

    /// Asks the task running `id` to stop; `None` when nothing runs under
    /// that id here. The receiver turns `true` once the task has written its
    /// terminal record and left the registry.
    pub fn cancel(&self, id: &str) -> Option<watch::Receiver<bool>> {
        let running = self.running.lock();
        let entry = running.get(id)?;
        entry.cancel.cancel();
        Some(entry.finished.clone())
    }
}

/// A registered background response; dropping it frees its place.
pub(crate) struct BackgroundSlot {
    registry: Arc<BackgroundResponses>,
    id: String,
    cancel: CancellationToken,
    finished: watch::Sender<bool>,
}

impl BackgroundSlot {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

impl Drop for BackgroundSlot {
    fn drop(&mut self) {
        self.registry.running.lock().remove(&self.id);
        self.finished.send_replace(true);
    }
}

/// The 429 a background request gets when the cap is reached.
pub(crate) fn limit_reached(max_in_flight: usize) -> Response {
    error::too_many_requests(
        "background_responses_limit_reached",
        format!(
            "This gateway already runs its maximum of {max_in_flight} background responses; retry later or run the request without 'background'."
        ),
    )
}

// ============================================================================
// Records
// ============================================================================

/// The Response object a background request is answered with: `queued`, no
/// output, no usage, the request's parameters echoed.
pub(crate) fn queued_response(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
    ResponsesResponse::builder(id, &request.model)
        .copy_from_request(request)
        .created_at(chrono::Utc::now().timestamp())
        .status(ResponseStatus::Queued)
        .background(true)
        .build()
}

/// Stores `response` under its id with the request's input items, as the
/// terminal record will carry them, so `GET /v1/responses/{id}` and its
/// `input_items` answer from the first poll.
pub(crate) async fn store_record(
    storage: &Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) -> Result<(), String> {
    let raw = serde_json::to_value(response).map_err(|e| e.to_string())?;
    let mut stored = build_stored_response(raw, request);
    stored.input = Value::Array(extract_input_items(&request.input)?);
    let store = async {
        storage
            .store_response(stored)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    match request_context {
        Some(ctx) => with_request_context(ctx, store).await,
        None => store.await,
    }
}

/// The stored Response object moved to `status`: a terminal status stamps
/// `completed_at`, a failure carries its `error`.
pub(crate) fn with_status(mut raw: Value, status: &str, error: Option<Value>) -> Value {
    let terminal = matches!(status, "completed" | "incomplete" | "failed" | "cancelled");
    if let Some(object) = raw.as_object_mut() {
        object.insert("status".into(), json!(status));
        if terminal && object.get("completed_at").is_none_or(Value::is_null) {
            object.insert("completed_at".into(), json!(chrono::Utc::now().timestamp()));
        }
        if let Some(error) = error {
            object.insert("error".into(), error);
        }
    }
    raw
}

/// Rewrites the record stored under `id` into `status` and answers the new
/// object; `None` when nothing is stored under the id.
pub(crate) async fn set_stored_status(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
) -> Result<Option<Value>, String> {
    let update = async {
        let response_id = ResponseId::from(id);
        let Some(mut stored) = storage
            .get_response(&response_id)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(None);
        };
        stored.raw_response = with_status(stored.raw_response, status, error);
        let raw = stored.raw_response.clone();
        storage
            .store_response(stored)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Some(raw))
    };
    match request_context {
        Some(ctx) => with_request_context(ctx, update).await,
        None => update.await,
    }
}

/// The `error` object of a failed record, read from the error response the
/// work answered with.
async fn error_object(response: Response) -> Value {
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let error = &parsed["error"];
    let code = error["code"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| {
            if status.is_client_error() {
                "invalid_request_error".to_string()
            } else {
                "server_error".to_string()
            }
        });
    let message = error["message"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| format!("The response failed with HTTP {status}"));
    json!({ "code": code, "message": message })
}

// ============================================================================
// Running the work
// ============================================================================

/// The work of a background response, run against a context the task owns.
pub(crate) type BackgroundWork = Box<
    dyn for<'a> FnOnce(
            &'a ResponsesContext,
            Arc<ResponsesRequest>,
        ) -> BoxFuture<'a, Result<ResponsesResponse, Response>>
        + Send,
>;

/// Runs `work` in a task of its own: the record moves to `in_progress` first;
/// the work's own persistence writes the terminal object, an error answer
/// becomes a `failed` record, and a cancel marks the record `cancelled` and
/// drops the work, which drops its engine stream with it.
#[expect(
    clippy::disallowed_methods,
    reason = "a background response outlives its request by design; the registry bounds how many run and the slot is released when the task ends"
)]
pub(crate) fn spawn(
    ctx: ResponsesContext,
    slot: BackgroundSlot,
    request: Arc<ResponsesRequest>,
    work: BackgroundWork,
) {
    tokio::spawn(async move {
        let storage = ctx.response_storage.clone();
        let request_context = ctx.request_context.clone();
        drive(storage, request_context, slot, async {
            match work(&ctx, request).await {
                Ok(_) => Ok(()),
                Err(response) => Err(error_object(response).await),
            }
        })
        .await;
    });
}

/// The lifecycle around `work` (see [`spawn`]); `work` answers `Err(error)`
/// when the response failed before its terminal object was stored.
pub(crate) async fn drive<F>(
    storage: Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    slot: BackgroundSlot,
    work: F,
) where
    F: Future<Output = Result<(), Value>>,
{
    let id = slot.id().to_string();
    let token = slot.cancel_token();
    if let Err(e) =
        set_stored_status(&storage, &id, "in_progress", None, request_context.clone()).await
    {
        warn!(response_id = %id, error = %e, "Failed to mark the background response in progress");
    }
    tokio::select! {
        biased;
        () = token.cancelled() => {
            info!(response_id = %id, "Background response cancelled");
            if let Err(e) = set_stored_status(&storage, &id, "cancelled", None, request_context).await {
                warn!(response_id = %id, error = %e, "Failed to mark the background response cancelled");
            }
        }
        outcome = work => match outcome {
            Ok(()) => debug!(response_id = %id, "Background response finished"),
            Err(error) => {
                info!(response_id = %id, error = %error, "Background response failed");
                if let Err(e) = set_stored_status(&storage, &id, "failed", Some(error), request_context).await {
                    warn!(response_id = %id, error = %e, "Failed to mark the background response failed");
                }
            }
        },
    }
    drop(slot);
}

/// A background response the client streams: the pre-stored id and the slot
/// the stream's task keeps until it ends.
pub(crate) struct BackgroundStream {
    slot: BackgroundSlot,
}

impl BackgroundStream {
    pub fn new(slot: BackgroundSlot) -> Self {
        Self { slot }
    }

    pub fn id(&self) -> &str {
        self.slot.id()
    }

    /// The sender the stream's task writes to: events reach the client while
    /// it reads and are dropped once it has left, so the work runs to its end
    /// and its terminal object is stored either way.
    #[expect(
        clippy::disallowed_methods,
        reason = "the relay lives exactly as long as the stream's task holds its sender"
    )]
    pub fn detach(client: SseSender) -> SseSender {
        let (tx, mut rx) = sse_channel();
        tokio::spawn(async move {
            let mut client = Some(client);
            while let Some(event) = rx.recv().await {
                if let Some(sender) = client.as_ref() {
                    if sender.send(event).await.is_err() {
                        client = None;
                    }
                }
            }
        });
        tx
    }

    /// The stream task's body: `work` until it ends, or until a cancel, which
    /// marks the record `cancelled` and drops the work (and the engine stream
    /// with it); the stream then ends without a terminal event, as the public
    /// API's streams have none for a cancel.
    pub async fn run<F>(
        self,
        storage: Arc<dyn ResponseStorage>,
        request_context: Option<StorageRequestContext>,
        work: F,
    ) where
        F: Future<Output = ()>,
    {
        drive(storage, request_context, self.slot, async {
            work.await;
            Ok(())
        })
        .await;
    }
}

/// `POST /v1/responses/{id}/cancel`: a response this gateway runs is stopped
/// and answered as `cancelled` once its task has written the record; a stored
/// background response no task runs here (after a restart, or on another
/// instance) is closed as `cancelled`; a response that already ended, or one
/// that never ran in the background, is refused.
pub(crate) async fn cancel(ctx: &ResponsesContext, response_id: &str) -> Response {
    if let Some(mut finished) = ctx.background.cancel(response_id) {
        let settled = tokio::time::timeout(CANCEL_SETTLE, async {
            while !*finished.borrow_and_update() {
                if finished.changed().await.is_err() {
                    break;
                }
            }
        })
        .await;
        if settled.is_err() {
            warn!(
                response_id,
                "Background response did not settle after its cancel; answering from the store"
            );
        }
    }

    let stored = match ctx
        .response_storage
        .get_response(&ResponseId::from(response_id))
        .await
    {
        Ok(Some(stored)) => stored,
        Ok(None) => {
            return error::not_found(
                "response_not_found",
                format!("Response with id '{response_id}' not found"),
            )
        }
        Err(e) => {
            return error::internal_error(
                "retrieve_response_failed",
                format!("Failed to retrieve response: {e}"),
            )
        }
    };
    let raw = stored.raw_response;
    let status = raw["status"].as_str().unwrap_or("unknown");
    let background = raw["background"].as_bool().unwrap_or(false);
    match status {
        "cancelled" => (StatusCode::OK, Json(raw)).into_response(),
        "completed" | "incomplete" => error::bad_request(
            "response_already_completed",
            "Cannot cancel completed response",
        ),
        "failed" => error::bad_request("response_already_failed", "Cannot cancel failed response"),
        "queued" | "in_progress" if background => {
            match set_stored_status(
                &ctx.response_storage,
                response_id,
                "cancelled",
                None,
                ctx.request_context.clone(),
            )
            .await
            {
                Ok(Some(raw)) => (StatusCode::OK, Json(raw)).into_response(),
                Ok(None) => error::not_found(
                    "response_not_found",
                    format!("Response with id '{response_id}' not found"),
                ),
                Err(e) => error::internal_error(
                    "cancel_response_failed",
                    format!("Failed to cancel response: {e}"),
                ),
            }
        }
        _ => error::bad_request(
            "cancellation_not_supported",
            "Only responses created with 'background': true can be cancelled.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use smg_data_connector::MemoryResponseStorage;
    use tokio::sync::oneshot;

    use super::*;

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
        storage
            .get_response(&ResponseId::from(id))
            .await
            .unwrap()
            .map(|s| s.raw_response["status"].clone())
            .unwrap_or(Value::Null)
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
        let request = request();
        let queued = queued_response("resp_1", &request);
        store_record(&storage, &queued, &request, None)
            .await
            .unwrap();
        let stored = storage
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .expect("pollable at once");
        assert_eq!(stored.raw_response["status"], "queued");
        assert_eq!(stored.input[0]["content"][0]["text"], "write a story");
        assert_eq!(stored.safety_identifier.as_deref(), Some("user-1"));

        let raw = set_stored_status(&storage, "resp_1", "in_progress", None, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(raw["status"], "in_progress");
        assert_eq!(stored_status(&storage, "resp_1").await, "in_progress");
        assert!(
            set_stored_status(&storage, "resp_404", "cancelled", None, None)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_cancel_marks_the_record_cancelled_and_drops_the_work() {
        let storage = storage();
        let request = request();
        store_record(
            &storage,
            &queued_response("resp_1", &request),
            &request,
            None,
        )
        .await
        .unwrap();
        let registry = Arc::new(BackgroundResponses::new(4));
        let slot = registry.admit("resp_1").unwrap();
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
        struct DropFlag(Option<oneshot::Sender<()>>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
            }
        }
        let flag = DropFlag(Some(dropped_tx));
        let work = async move {
            let _flag = flag;
            std::future::pending::<()>().await;
            Ok(())
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let task = tokio::spawn(drive(storage.clone(), None, slot, work));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(stored_status(&storage, "resp_1").await, "in_progress");
        let mut finished = registry.cancel("resp_1").expect("running");
        task.await.unwrap();
        dropped_rx.await.expect("the work was dropped");
        assert!(*finished.borrow_and_update());
        assert_eq!(stored_status(&storage, "resp_1").await, "cancelled");
        assert_eq!(registry.in_flight(), 0);
    }

    #[tokio::test]
    async fn a_failed_work_leaves_a_failed_record_with_its_error() {
        let storage = storage();
        let request = request();
        store_record(
            &storage,
            &queued_response("resp_1", &request),
            &request,
            None,
        )
        .await
        .unwrap();
        let registry = Arc::new(BackgroundResponses::new(4));
        let slot = registry.admit("resp_1").unwrap();
        drive(storage.clone(), None, slot, async {
            Err(json!({"code": "model_not_found", "message": "no such model"}))
        })
        .await;
        let stored = storage
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.raw_response["status"], "failed");
        assert_eq!(stored.raw_response["error"]["code"], "model_not_found");
        assert!(stored.raw_response["completed_at"].is_i64());
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
}
