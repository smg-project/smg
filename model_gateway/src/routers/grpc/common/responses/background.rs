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
//!
//! The stored record is the shared truth, also between gateways that share a
//! store: a status only moves forward ([`set_stored_status`] never rewrites a
//! terminal record: every transition is one conditional write of the stored
//! object alone,
//! `ResponseStorage::update_response_object_if_open`, and the terminal
//! object the work persists one conditional write of the row,
//! `ResponseStorage::store_response_if_open`, both refused once the record
//! is closed), the runner reads the record while the work runs and drops the
//! work once the record was closed elsewhere, and every way a run can end
//! leaves a terminal record ([`drive`]): the object the work stored, `failed`
//! with the error of a work that answered one, ended without storing a
//! result or panicked, or `cancelled`.
//!
//! A response the client asked not to store (`store: false`) runs in the
//! background the same way: its record is the run's control block and, once
//! terminal, holds the result for [`UNSTORED_RETENTION`] so the client can
//! poll it, as the public API keeps such a response for roughly ten minutes;
//! then the gateway that closed it deletes it ([`finish_unstored`]). Nothing
//! of it reaches the conversation and history side.

use std::{
    collections::HashMap, future::Future, panic::AssertUnwindSafe, sync::Arc, time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::{future::BoxFuture, FutureExt};
use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
use parking_lot::Mutex;
use serde_json::{json, Value};
use smg_data_connector::{
    with_request_context, ConditionalStore, RequestContext as StorageRequestContext, ResponseId,
    ResponseStorage, ResponseStorageError, StoredResponse, TERMINAL_RESPONSE_STATUSES,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::ResponsesContext;
use crate::routers::{
    common::{
        persistence_utils::{self, build_stored_response, extract_input_items},
        sse::{sse_channel, SseSender},
    },
    error,
    grpc::utils::isolation::panic_message,
};

/// Runs `fut`, a storage operation, under the request's storage context
/// when the request has one, so the store sees the tenant and the request
/// it serves; without one the operation runs as it is.
async fn under_context<F: Future>(
    request_context: Option<StorageRequestContext>,
    fut: F,
) -> F::Output {
    match request_context {
        Some(ctx) => with_request_context(ctx, fut).await,
        None => fut.await,
    }
}

/// How long a cancel request waits for the task to write its terminal record
/// before answering from whatever the store holds.
const CANCEL_SETTLE: Duration = Duration::from_secs(5);

/// How often the runner reads the stored record while the work runs: a
/// cancel on another gateway sharing the store closes the record, and the
/// gateway running the response drops its work within this interval.
pub(crate) const STATUS_WATCH: Duration = Duration::from_secs(2);

/// How many status-watch intervals a terminal write is retried for while the
/// store answers with an error (60 s at the default [`STATUS_WATCH`]): a store
/// that is unreachable for a moment at the end of a run must not leave the
/// record `in_progress` behind a gateway that is still alive and holds the
/// result.
const TERMINAL_WRITE_RETRIES: u32 = 30;

/// The window a terminal write is retried within, for a watch interval.
pub(crate) fn terminal_write_window(watch: Duration) -> Duration {
    watch.saturating_mul(TERMINAL_WRITE_RETRIES)
}

/// How long the record of a background response the client asked not to
/// store (`store: false`) stays once it reached a terminal status: the public
/// API keeps the response data of such a request for roughly ten minutes so
/// that it can be polled, then deletes it.
pub(crate) const UNSTORED_RETENTION: Duration = Duration::from_secs(10 * 60);

/// Whether `status` is one a response does not leave any more: the store's
/// list ([`TERMINAL_RESPONSE_STATUSES`]), so the gateway and the stores'
/// conditional writes agree on what a closed record is.
pub(crate) fn is_terminal(status: &str) -> bool {
    TERMINAL_RESPONSE_STATUSES.contains(&status)
}

/// Whether `status` is a result only the work itself stores (a run that
/// reached its end, whole or truncated): a record read in one of these while
/// the work runs is the work's own terminal write, not a close from
/// elsewhere. The persist's check for its own earlier write reads the same
/// list ([`persistence_utils::is_result_status`]).
fn is_result(status: &str) -> bool {
    persistence_utils::is_result_status(status)
}

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
/// `input_items` answer from the first poll; the record is scoped to the
/// caller's `tenant`.
pub(crate) async fn store_record(
    storage: &Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    request: &ResponsesRequest,
    tenant: Option<&str>,
    request_context: Option<StorageRequestContext>,
) -> Result<(), ResponseStorageError> {
    let raw = serde_json::to_value(response)?;
    let mut stored = build_stored_response(raw, request);
    stored.input = Value::Array(
        extract_input_items(&request.input).map_err(ResponseStorageError::StorageError)?,
    );
    stored.tenant = tenant.map(str::to_string);
    let store = async { storage.store_response(stored).await.map(|_| ()) };
    under_context(request_context, store).await
}

/// The stored Response object moved to `status`: a terminal status stamps
/// `completed_at`, a failure carries its `error`.
pub(crate) fn with_status(mut raw: Value, status: &str, error: Option<Value>) -> Value {
    let terminal = is_terminal(status);
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

/// What [`set_stored_status`] found under the id.
#[derive(Debug)]
pub(crate) enum Transition {
    /// The record moved to the status and now reads as this object.
    Moved(Value),
    /// The record had already reached a terminal status; this object stays
    /// as it is.
    Terminal(Value),
    /// Nothing is stored under the id.
    Missing,
}

/// Moves the record stored under `id` to `status`. A status only moves forward: a
/// record that is already terminal is left as it is (so a `cancelled`
/// record is written once, and a result that arrives after a cancel does
/// not revive the response), whichever gateway wrote it. The record is read
/// to build the moved object and written with one conditional write, which
/// the store refuses if the record was closed in between, on any gateway.
async fn set_stored_status(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
) -> Result<Transition, ResponseStorageError> {
    transition(storage, id, request_context, |raw| {
        with_status(raw, status, error)
    })
    .await
}

/// The end of a background response the client asked not to store: its
/// terminal object is written over the record under its id (which keeps its
/// input items), where a retained response is persisted with its conversation
/// items. The result is polled from the record for [`UNSTORED_RETENTION`],
/// then the record is retired; nothing of it reaches the conversation and
/// history side. A record that is already terminal (closed by a cancel, here
/// or on another gateway) stays as it is.
pub(crate) async fn finish_unstored(
    storage: &Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    request_context: Option<StorageRequestContext>,
) {
    let mut raw = match serde_json::to_value(response) {
        Ok(raw) => raw,
        Err(e) => {
            warn!(response_id = %response.id, error = %e, "Failed to serialize the background response");
            return;
        }
    };
    raw["store"] = json!(false);
    match transition(storage, &response.id, request_context, |_| raw).await {
        Ok(Transition::Moved(_)) => debug!(
            response_id = %response.id,
            "Unstored background response finished; its record is kept for the polling window"
        ),
        Ok(Transition::Terminal(stored)) => debug!(
            response_id = %response.id,
            stored = stored["status"].as_str().unwrap_or(""),
            "Background response record had already been closed; the result is not stored"
        ),
        Ok(Transition::Missing) => {
            warn!(response_id = %response.id, "No record under the background response id to finish");
        }
        Err(e) => warn!(
            response_id = %response.id,
            error = %e,
            "Failed to store the terminal object of the background response"
        ),
    }
}

/// Moves the record under `id` forward to the object `next` makes of the
/// stored one; a record that is already terminal stays as it is. The record
/// is read to build the moved object and written with one conditional write,
/// which the store refuses if the record was closed in between, on any
/// gateway. A record moved to a terminal status whose object says `store:
/// false` is retired after [`UNSTORED_RETENTION`] by the gateway that moved
/// it.
async fn transition(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    request_context: Option<StorageRequestContext>,
    next: impl FnOnce(Value) -> Value,
) -> Result<Transition, ResponseStorageError> {
    let update = async {
        let response_id = ResponseId::from(id);
        let Some(stored) = storage.get_response(&response_id).await? else {
            return Ok(Transition::Missing);
        };
        if is_terminal(stored.raw_response["status"].as_str().unwrap_or("")) {
            return Ok(Transition::Terminal(stored.raw_response));
        }
        let raw = next(stored.raw_response);
        match storage
            .update_response_object_if_open(&response_id, raw.clone())
            .await?
        {
            ConditionalStore::Stored => Ok(Transition::Moved(raw)),
            ConditionalStore::Closed(current) => Ok(Transition::Terminal(current.raw_response)),
            ConditionalStore::Missing => Ok(Transition::Missing),
        }
    };
    let transition: Result<Transition, ResponseStorageError> =
        under_context(request_context.clone(), update).await;
    let transition = transition?;
    if let Transition::Moved(raw) = &transition {
        retire_unstored(storage, request_context, id, raw);
    }
    Ok(transition)
}

/// Deletes the record under `id` once [`UNSTORED_RETENTION`] has passed when
/// `raw` is the terminal object of a response the client asked not to store:
/// after its polling window nothing of it is retained. A gateway restarted
/// inside the window leaves the record, whose object says `store: false`, to
/// its store's own retention.
#[expect(
    clippy::disallowed_methods,
    reason = "the record of an unstored background response outlives its request by the polling window; the task holds only the storage handle"
)]
fn retire_unstored(
    storage: &Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    id: &str,
    raw: &Value,
) {
    if raw["store"].as_bool().unwrap_or(true) || !is_terminal(raw["status"].as_str().unwrap_or(""))
    {
        return;
    }
    let storage = Arc::clone(storage);
    let id = id.to_string();
    tokio::spawn(async move {
        // A store that keeps deadlines drops the record itself should this
        // gateway be gone before the window ends; the stores that keep none
        // are swept (see [`sweep_unstored`]).
        let expire = async {
            storage
                .expire_response(&ResponseId::from(id.as_str()), UNSTORED_RETENTION)
                .await
        };
        let expired = under_context(request_context.clone(), expire).await;
        // The default answers Ok for a store that keeps no deadline: an error
        // is a store that keeps them and failed to set one, which leaves this
        // gateway's timer as the record's only retirement.
        if let Err(e) = expired {
            warn!(
                response_id = %id,
                error = %e,
                "Failed to set the store's deadline on the unstored background response record; only this gateway's timer retires it"
            );
        }
        tokio::time::sleep(UNSTORED_RETENTION).await;
        let delete = async {
            storage
                .delete_response(&ResponseId::from(id.as_str()))
                .await
        };
        let deleted = under_context(request_context, delete).await;
        match deleted {
            Ok(()) => debug!(
                response_id = %id,
                "Unstored background response record retired after its polling window"
            ),
            Err(e) => warn!(
                response_id = %id,
                error = %e,
                "Failed to retire the unstored background response record"
            ),
        }
    });
}

/// Deletes the records of unstored background responses whose polling window
/// ended with no gateway there to retire them: a gateway stopped or lost
/// inside the window, with a store that outlives the process. Returns how
/// many records went. A store that keeps no query over the stored object
/// answers 0 (see `ResponseStorage::delete_unstored_older_than`).
async fn sweep_unstored(storage: &Arc<dyn ResponseStorage>) -> usize {
    let window =
        chrono::Duration::seconds(i64::try_from(UNSTORED_RETENTION.as_secs()).unwrap_or(i64::MAX));
    match storage
        .delete_unstored_older_than(chrono::Utc::now() - window)
        .await
    {
        Ok(0) => 0,
        Ok(deleted) => {
            info!(
                deleted,
                "Retired the unstored background response records left past their polling window"
            );
            deleted
        }
        Err(e) => {
            warn!(error = %e, "Failed to sweep the unstored background response records");
            0
        }
    }
}

/// Runs [`sweep_unstored`] now, for the records an earlier gateway left in a
/// store that outlives it, whatever this gateway's background setting; then,
/// when `repeat` says so, once per polling window for the life of the gateway
/// (background mode on: new unstored records can appear; off, none can, and
/// the one sweep at start-up is all the store needs).
#[expect(
    clippy::disallowed_methods,
    reason = "a periodic sweep over the store for the gateway's lifetime; nothing waits on it at shutdown"
)]
pub(crate) fn spawn_unstored_sweeper(
    storage: Arc<dyn ResponseStorage>,
    repeat: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            sweep_unstored(&storage).await;
            if !repeat {
                return;
            }
            tokio::time::sleep(UNSTORED_RETENTION).await;
        }
    })
}

/// [`set_stored_status`] retried every `every` within `window` while the store
/// answers with an error: the write that ends a background run is not given up
/// on a store blip. The first failure is logged once; the outcome is the
/// caller's to log.
async fn set_stored_status_retrying(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
    every: Duration,
    window: Duration,
) -> Result<Transition, ResponseStorageError> {
    let deadline = tokio::time::Instant::now() + window;
    let mut attempts = 0u32;
    loop {
        match set_stored_status(storage, id, status, error.clone(), request_context.clone()).await {
            Ok(transition) => return Ok(transition),
            Err(e) if tokio::time::Instant::now() < deadline => {
                if attempts == 0 {
                    warn!(
                        response_id = %id,
                        status,
                        error = %e,
                        "Failed to write the background response record; retrying while the store is unreachable"
                    );
                } else {
                    debug!(response_id = %id, status, attempts, error = %e, "The terminal write still fails");
                }
                attempts += 1;
                tokio::time::sleep(every).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Moves the record under `id` to the terminal `status` (see
/// [`set_stored_status`]); a record that is already terminal stays. The
/// store's answer is logged here and returned for the caller that retries.
async fn close(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
) -> Result<Transition, ResponseStorageError> {
    let outcome = set_stored_status(storage, id, status, error, request_context).await;
    match &outcome {
        Ok(Transition::Moved(_)) => {}
        Ok(Transition::Terminal(raw)) => debug!(
            response_id = %id,
            status,
            stored = raw["status"].as_str().unwrap_or(""),
            "Background response record had already been closed"
        ),
        Ok(Transition::Missing) => {
            warn!(response_id = %id, status, "No record to close under the background response id");
        }
        Err(e) => warn!(
            response_id = %id,
            status,
            error = %e,
            "Failed to close the background response record"
        ),
    }
    outcome
}

/// [`close`] for the end of a run: retried every `every` within the terminal
/// write window while the store answers with an error.
async fn close_retrying(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
    every: Duration,
) {
    let window = terminal_write_window(every);
    match set_stored_status_retrying(storage, id, status, error, request_context, every, window)
        .await
    {
        Ok(Transition::Moved(_)) => {}
        Ok(Transition::Terminal(raw)) => debug!(
            response_id = %id,
            status,
            stored = raw["status"].as_str().unwrap_or(""),
            "Background response record had already been closed"
        ),
        Ok(Transition::Missing) => {
            warn!(response_id = %id, status, "No record to close under the background response id");
        }
        Err(e) => warn!(
            response_id = %id,
            status,
            error = %e,
            "Failed to close the background response record; the store stayed unreachable for the whole retry window"
        ),
    }
}

/// The `error` object of a failed record, read from the error response the
/// work answered with.
async fn error_object(response: Response) -> Value {
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    error_object_from(status, &body)
}

/// The `error` object for an error answer with `status` and `body`: its own
/// code and message when the body is the gateway's error envelope, generic
/// ones otherwise.
fn error_object_from(status: StatusCode, body: &[u8]) -> Value {
    let parsed: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
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

/// A background stream that was answered with a plain error before it
/// opened (the history or the tools could not be loaded, the engine refused
/// the request): the record, still `queued` or `in_progress`, is closed as
/// `failed` with that error, and the answer goes back to the client as it
/// was (its body is read once for the error object). A streaming answer
/// passes through untouched.
pub(crate) async fn close_before_stream(
    storage: &Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    id: &str,
    response: Response,
) -> Response {
    if response.status().is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let bytes = to_bytes(body, 1 << 20).await.unwrap_or_default();
    let error = error_object_from(parts.status, &bytes);
    info!(response_id = %id, status = %parts.status, "Background stream failed before it opened");
    if close(
        storage,
        id,
        "failed",
        Some(error.clone()),
        request_context.clone(),
    )
    .await
    .is_err()
    {
        // No run follows that would close the record, and the client is
        // waiting for this answer: the close is retried in a task of its own
        // while the store is unreachable.
        let storage = storage.clone();
        let id = id.to_string();
        #[expect(
            clippy::disallowed_methods,
            reason = "the error answer must not wait for the store; the task ends with the close, within the terminal write window"
        )]
        let _closing = tokio::spawn(async move {
            close_retrying(
                &storage,
                &id,
                "failed",
                Some(error),
                request_context,
                STATUS_WATCH,
            )
            .await;
        });
    }
    Response::from_parts(parts, Body::from(bytes))
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
/// when the response failed before its terminal object was stored. Every
/// way out leaves a terminal record: the object the work stored; `failed`
/// with its error, or with a generic one when the work ended without
/// storing a result or panicked; `cancelled` on a cancel here. A record
/// closed on another gateway sharing the store (read every
/// [`STATUS_WATCH`]) stops the work and is not written again; so does a
/// record deleted while the work ran, which is not written back either: the
/// delete holds. A record read in the work's own result (`completed` or
/// `incomplete`, which only the work writes) is the work finishing: it links
/// the response's items to its conversation after that write, so the run
/// goes on until the work ends, for the terminal write window at most.
pub(crate) async fn drive<F>(
    storage: Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    slot: BackgroundSlot,
    work: F,
) where
    F: Future<Output = Result<(), Value>>,
{
    drive_watching(storage, request_context, slot, work, STATUS_WATCH).await;
}

/// [`drive`] with the interval the stored record is read at while the work
/// runs.
async fn drive_watching<F>(
    storage: Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    slot: BackgroundSlot,
    work: F,
    watch: Duration,
) where
    F: Future<Output = Result<(), Value>>,
{
    let id = slot.id().to_string();
    let token = slot.cancel_token();
    // Whether the record was there when the work started; the watch learns
    // it from its own reads as well, so a store blip at this transition does
    // not disarm it. Only a record seen is a deletion when it is gone later:
    // a store that never shows the record (no history backend, or a record
    // deleted before the start) runs the work to its end as before and
    // stores nothing over the missing record.
    let record_seen = match set_stored_status(
        &storage,
        &id,
        "in_progress",
        None,
        request_context.clone(),
    )
    .await
    {
        Ok(Transition::Moved(_)) => true,
        Ok(Transition::Terminal(raw)) => {
            info!(
                response_id = %id,
                status = raw["status"].as_str().unwrap_or(""),
                "Background response was closed before its work started"
            );
            drop(slot);
            return;
        }
        Ok(Transition::Missing) => {
            warn!(response_id = %id, "No record stored under the background response id; its work runs and its result is not stored");
            false
        }
        Err(e) => {
            warn!(response_id = %id, error = %e, "Failed to mark the background response in progress");
            false
        }
    };
    let closed_elsewhere =
        watch_stored_status(&storage, &id, request_context.clone(), watch, record_seen);
    let work = AssertUnwindSafe(work).catch_unwind();
    tokio::select! {
        biased;
        () = token.cancelled() => {
            info!(response_id = %id, "Background response cancelled");
            close_retrying(&storage, &id, "cancelled", None, request_context, watch).await;
        }
        status = closed_elsewhere => {
            if is_result(&status) {
                warn!(response_id = %id, status = %status, "Background response stored its result but its work did not end within the terminal write window; the work is dropped");
            } else {
                info!(response_id = %id, status = %status, "Background response was closed elsewhere; its work is dropped");
            }
        }
        outcome = work => match outcome {
            Ok(Ok(())) => {
                // The work's own persistence wrote the terminal object; a work
                // that returned without one leaves `failed`, not `in_progress`
                // forever.
                let error = json!({
                    "code": "server_error",
                    "message": "The response ended without a stored result",
                });
                let window = terminal_write_window(watch);
                match set_stored_status_retrying(&storage, &id, "failed", Some(error), request_context, watch, window).await {
                    Ok(Transition::Terminal(_)) => debug!(response_id = %id, "Background response finished"),
                    Ok(Transition::Moved(_)) => {
                        warn!(response_id = %id, "Background response ended without a stored result; marked failed");
                    }
                    Ok(Transition::Missing) => {
                        info!(response_id = %id, "Background response ended with no record under its id (deleted while it ran); its result is not stored");
                    }
                    Err(e) => warn!(
                        response_id = %id,
                        error = %e,
                        "Failed to read the background response record back; the store stayed unreachable for the whole retry window"
                    ),
                }
            }
            Ok(Err(error)) => {
                info!(response_id = %id, error = %error, "Background response failed");
                close_retrying(&storage, &id, "failed", Some(error), request_context, watch).await;
            }
            Err(panic) => {
                let message = panic_message(panic.as_ref());
                warn!(response_id = %id, panic = %message, "Background response task panicked");
                let error = json!({
                    "code": "server_error",
                    "message": format!("The response's task panicked: {message}"),
                });
                close_retrying(&storage, &id, "failed", Some(error), request_context, watch).await;
            }
        },
    }
    drop(slot);
}

/// The status the watch answers for a record that is gone: the response was
/// deleted while its work ran (`DELETE /v1/responses/{id}`, here or on
/// another gateway sharing the store); the work is dropped like a cancel and
/// nothing is written under the id again, so the delete holds.
pub(crate) const DELETED: &str = "deleted";

/// Resolves with the status once the record under `id` reads terminal, or,
/// once the record was seen (`record_seen` at the start, or by a read of
/// this watch), with [`DELETED`] once it is gone. A store that never showed
/// the record is not read as a deletion; a store that cannot be read is
/// retried at the next tick. A record read in the work's own result (see
/// [`is_result`]) is the work finishing, not a close from elsewhere: the
/// work stores the terminal object first and links the response's items to
/// its conversation after it (two or more store round trips later), and a
/// run dropped between the two left a `completed` record whose conversation
/// missed the items. The watch keeps reading for the terminal write window
/// from that first read (a delete meanwhile still ends the run) and resolves
/// with the result only once the window has passed, so the run ends with the
/// work in the meantime.
async fn watch_stored_status(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    request_context: Option<StorageRequestContext>,
    every: Duration,
    mut record_seen: bool,
) -> String {
    let mut result_read_at: Option<tokio::time::Instant> = None;
    loop {
        tokio::time::sleep(every).await;
        let read = async { storage.get_response(&ResponseId::from(id)).await };
        let stored = under_context(request_context.clone(), read).await;
        match stored {
            Ok(Some(stored)) => {
                record_seen = true;
                let status = stored.raw_response["status"].as_str().unwrap_or("");
                if is_terminal(status) {
                    if !is_result(status) {
                        return status.to_string();
                    }
                    let read_at = *result_read_at.get_or_insert_with(tokio::time::Instant::now);
                    if read_at.elapsed() >= terminal_write_window(every) {
                        return status.to_string();
                    }
                }
            }
            Ok(None) if record_seen => {
                return DELETED.to_string();
            }
            Ok(None) => {
                debug!(response_id = %id, "Background response record not found while its work runs");
            }
            Err(e) => {
                debug!(response_id = %id, error = %e, "Failed to read the background response record");
            }
        }
    }
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

    /// The stream task's body: `work` until it ends (it answers `Err(error)`
    /// when the stream failed without storing a terminal object), or until a
    /// cancel, which marks the record `cancelled` and drops the work (and the
    /// engine stream with it); the stream then ends without a terminal event,
    /// as the public API's streams have none for a cancel.
    pub async fn run<F>(
        self,
        storage: Arc<dyn ResponseStorage>,
        request_context: Option<StorageRequestContext>,
        work: F,
    ) where
        F: Future<Output = Result<(), Value>>,
    {
        drive(storage, request_context, self.slot, work).await;
    }
}

/// `POST /v1/responses/{id}/cancel` for a caller of `tenant`: a response
/// this gateway runs is stopped and answered as `cancelled` once its task has
/// written the record; a `queued` or `in_progress` background record no task
/// runs here (a response of another gateway sharing the store, or one lost
/// with a restart) is closed as `cancelled`, which the gateway running it
/// reads within [`STATUS_WATCH`] and drops its work on; a response that
/// already ended, or one that never ran in the background, is refused. A
/// response of another tenant answers as an unknown id does, and nothing
/// running under its id is touched.
pub(crate) async fn cancel(
    ctx: &ResponsesContext,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    cancel_with(
        &ctx.background,
        &ctx.response_storage,
        ctx.request_context.clone(),
        tenant,
        response_id,
    )
    .await
}

/// [`cancel`] over the registry and the store it works on.
async fn cancel_with(
    registry: &BackgroundResponses,
    storage: &Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    // The record first: the registry is touched only for a response the
    // caller may see.
    match visible_record(storage, request_context.clone(), tenant, response_id).await {
        Ok(Some(_)) => {}
        Ok(None) => return response_not_found(response_id),
        Err(response) => return response,
    }

    if let Some(mut finished) = registry.cancel(response_id) {
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

    let stored = match visible_record(storage, request_context.clone(), tenant, response_id).await {
        Ok(Some(stored)) => stored,
        Ok(None) => return response_not_found(response_id),
        Err(response) => return response,
    };
    let raw = stored.raw_response;
    let background = raw["background"].as_bool().unwrap_or(false);
    match raw["status"].as_str().unwrap_or("unknown") {
        "queued" | "in_progress" if background => {
            match set_stored_status(storage, response_id, "cancelled", None, request_context).await
            {
                Ok(Transition::Moved(raw)) => {
                    info!(
                        response_id,
                        "Background response closed as cancelled; no task runs it on this gateway"
                    );
                    (StatusCode::OK, Json(raw)).into_response()
                }
                Ok(Transition::Terminal(raw)) => answer_for_closed(raw),
                Ok(Transition::Missing) => error::not_found(
                    "response_not_found",
                    format!("Response with id '{response_id}' not found"),
                ),
                Err(e) => error::storage_failure(
                    "cancel_response_failed",
                    "Failed to cancel response",
                    &e,
                ),
            }
        }
        _ => answer_for_closed(raw),
    }
}

/// The record under `id` as a caller of `tenant` may see it: `None` for an
/// id that is not stored and for one stored under another tenant alike.
async fn visible_record(
    storage: &Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    tenant: Option<&str>,
    id: &str,
) -> Result<Option<StoredResponse>, Response> {
    let read = async { storage.get_response(&ResponseId::from(id)).await };
    let stored = under_context(request_context, read).await;
    match stored {
        Ok(stored) => Ok(stored.filter(|stored| stored.is_visible_to(tenant))),
        Err(e) => Err(error::storage_failure(
            "retrieve_response_failed",
            "Failed to retrieve response",
            &e,
        )),
    }
}

fn response_not_found(id: &str) -> Response {
    error::not_found(
        "response_not_found",
        format!("Response with id '{id}' not found"),
    )
}

/// The cancel answer for a record that is not running in the background:
/// the `cancelled` object, or the refusal its status earns.
fn answer_for_closed(raw: Value) -> Response {
    match raw["status"].as_str().unwrap_or("unknown") {
        "cancelled" => (StatusCode::OK, Json(raw)).into_response(),
        "completed" | "incomplete" => error::bad_request(
            "response_already_completed",
            "Cannot cancel completed response",
        ),
        "failed" => error::bad_request("response_already_failed", "Cannot cancel failed response"),
        _ => error::bad_request(
            "cancellation_not_supported",
            "Only responses created with 'background': true can be cancelled.",
        ),
    }
}

#[cfg(test)]
mod tests;
