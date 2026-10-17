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
    any::Any, collections::HashMap, future::Future, panic::AssertUnwindSafe, sync::Arc,
    time::Duration,
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
};

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
    match request_context {
        Some(ctx) => with_request_context(ctx, store).await,
        None => store.await,
    }
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
pub(crate) async fn set_stored_status(
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
    let transition: Result<Transition, ResponseStorageError> = match request_context.clone() {
        Some(ctx) => with_request_context(ctx, update).await,
        None => update.await,
    };
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
        let expired = match request_context.clone() {
            Some(ctx) => with_request_context(ctx, expire).await,
            None => expire.await,
        };
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
        let deleted = match request_context {
            Some(ctx) => with_request_context(ctx, delete).await,
            None => delete.await,
        };
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
pub(crate) async fn sweep_unstored(storage: &Arc<dyn ResponseStorage>) -> usize {
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
pub(crate) async fn set_stored_status_retrying(
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
pub(crate) async fn drive_watching<F>(
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
        let stored = match request_context.clone() {
            Some(ctx) => with_request_context(ctx, read).await,
            None => read.await,
        };
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

/// The text of a panic's payload.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
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
pub(crate) async fn cancel_with(
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
    let stored = match request_context {
        Some(ctx) => with_request_context(ctx, read).await,
        None => read.await,
    };
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
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use async_trait::async_trait;
    use smg_data_connector::{MemoryResponseStorage, ResponseStorageError};
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

        async fn delete_response(
            &self,
            response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
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

        async fn delete_response(
            &self,
            response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
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

        async fn delete_response(
            &self,
            response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
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
        let answer =
            cancel_with(&registry, &storage, None, Some("header:tenant-a"), "resp_1").await;
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

        let Transition::Moved(raw) =
            set_stored_status(&storage, "resp_1", "in_progress", None, None)
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
            let Transition::Terminal(raw) =
                set_stored_status(&storage, "resp_1", status, error, None)
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
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
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
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
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
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
        let flag = DropFlag(Some(dropped_tx));
        let work = async move {
            let _flag = flag;
            std::future::pending::<()>().await;
            Ok(())
        };
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
    async fn a_cancel_on_a_gateway_that_does_not_run_the_response_closes_it_once_and_stops_the_owner(
    ) {
        // Two gateways over one store: `owner` runs the response, `other`
        // receives the cancel and has no task for the id.
        let counting = Arc::new(CountingStorage::new());
        let storage: Arc<dyn ResponseStorage> = counting.clone();
        store_queued(&storage, "resp_1").await;
        let owner = Arc::new(BackgroundResponses::new(4));
        let other = Arc::new(BackgroundResponses::new(4));
        let slot = owner.admit("resp_1").unwrap();
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
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

    /// The terminal object of `request` with one output message.
    fn completed_response(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
        let message = serde_json::from_value(json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }))
        .unwrap();
        ResponsesResponse::builder(id, &request.model)
            .copy_from_request(request)
            .created_at(chrono::Utc::now().timestamp())
            .completed_at(chrono::Utc::now().timestamp())
            .status(ResponseStatus::Completed)
            .add_output(message)
            .build()
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

        async fn delete_response(
            &self,
            response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
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
}
