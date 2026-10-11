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
//! terminal record), the runner reads the record while the work runs and
//! drops the work once the record was closed elsewhere, and every way a run
//! can end leaves a terminal record ([`drive`]): the object the work stored,
//! `failed` with the error of a work that answered one, ended without storing
//! a result or panicked, or `cancelled`.
//!
//! A gateway that dies mid-run leaves no such record, so every open record
//! carries a lease ([`Lease`]: the owning registry's id and an expiry), set
//! when the record is stored, renewed while the work runs, and read by the
//! reaper every gateway sharing the store runs ([`reap_expired`]): a record
//! whose lease has expired is closed as `failed` with an error naming the
//! lost gateway. The lease lives under a private key of the stored object,
//! which is served without it ([`public_object`]).

use std::{
    any::Any,
    collections::HashMap,
    future::Future,
    panic::AssertUnwindSafe,
    sync::Arc,
    time::{Duration, Instant},
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
    with_request_context, RequestContext as StorageRequestContext, ResponseId, ResponseStorage,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

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

/// How often the runner reads the stored record while the work runs: a
/// cancel on another gateway sharing the store closes the record, and the
/// gateway running the response drops its work within this interval.
const STATUS_WATCH: Duration = Duration::from_secs(2);

/// The statuses a response does not leave any more.
const TERMINAL_STATUSES: [&str; 4] = ["completed", "incomplete", "failed", "cancelled"];

/// Whether `status` is one a response does not leave any more.
pub(crate) fn is_terminal(status: &str) -> bool {
    TERMINAL_STATUSES.contains(&status)
}

/// The key under which an open background record carries its [`Lease`]
/// inside the stored Response object (the store keeps no other field per
/// record); the object is served without it and a terminal write drops it.
const LEASE_KEY: &str = "_lease";

/// The most open records one reaper pass reads.
const REAP_BATCH: usize = 1024;

/// The in-flight background responses of this gateway, and the lease it
/// holds on their records.
pub(crate) struct BackgroundResponses {
    running: Mutex<HashMap<String, Entry>>,
    max_in_flight: usize,
    /// This registry's id, the owner of the leases it writes.
    owner: String,
    /// How long a lease stays valid without a renewal; zero switches the
    /// lease and the reaper off.
    lease_ttl: Duration,
}

struct Entry {
    cancel: CancellationToken,
    finished: watch::Receiver<bool>,
}

impl BackgroundResponses {
    /// `max_in_flight` background responses at once; 0 switches the mode off
    /// (see [`Self::enabled`]). `lease_ttl` is how long the lease on a record
    /// this gateway runs stays valid without a renewal; zero writes no lease
    /// and runs no reaper.
    pub fn new(max_in_flight: usize, lease_ttl: Duration) -> Self {
        Self {
            running: Mutex::new(HashMap::new()),
            max_in_flight,
            owner: format!("gw_{}", Uuid::now_v7()),
            lease_ttl,
        }
    }

    /// The id this registry writes as the owner of its leases.
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// A fresh lease of this registry, valid for its TTL from now; `None`
    /// when leases are off.
    pub fn lease(&self) -> Option<Lease> {
        (!self.lease_ttl.is_zero()).then(|| Lease {
            owner: self.owner.clone(),
            expires_at: chrono::Utc::now().timestamp() + self.lease_ttl.as_secs() as i64,
        })
    }

    /// How often the runner renews the lease on a record it runs: a third of
    /// the TTL, so two renewals can be lost before the lease expires.
    pub fn renew_every(&self) -> Duration {
        (self.lease_ttl / 3).max(Duration::from_secs(1))
    }

    /// How often the reaper reads the open records: half the TTL, so a
    /// record of a lost gateway is closed within one and a half TTLs of its
    /// last renewal; `None` when background mode or the lease is off.
    pub fn reap_every(&self) -> Option<Duration> {
        (self.enabled() && !self.lease_ttl.is_zero())
            .then(|| (self.lease_ttl / 2).max(Duration::from_secs(1)))
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

    /// A fresh lease of the registry this slot belongs to (`None` when
    /// leases are off).
    pub fn lease(&self) -> Option<Lease> {
        self.registry.lease()
    }

    fn registry(&self) -> &BackgroundResponses {
        &self.registry
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
/// `input_items` answer from the first poll, under `lease` when this gateway
/// writes one.
pub(crate) async fn store_record(
    storage: &Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    request: &ResponsesRequest,
    lease: Option<&Lease>,
    request_context: Option<StorageRequestContext>,
) -> Result<(), String> {
    let raw = serde_json::to_value(response).map_err(|e| e.to_string())?;
    let mut stored = build_stored_response(raw, request);
    stored.input = Value::Array(extract_input_items(&request.input)?);
    if let Some(lease) = lease {
        with_lease(&mut stored.raw_response, lease);
    }
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
/// `completed_at` and drops the lease (a closed record needs none), a
/// failure carries its `error`.
pub(crate) fn with_status(mut raw: Value, status: &str, error: Option<Value>) -> Value {
    let terminal = is_terminal(status);
    if let Some(object) = raw.as_object_mut() {
        object.insert("status".into(), json!(status));
        if terminal {
            object.remove(LEASE_KEY);
            if object.get("completed_at").is_none_or(Value::is_null) {
                object.insert("completed_at".into(), json!(chrono::Utc::now().timestamp()));
            }
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

/// Moves the record stored under `id` to `status`, under `lease` when the
/// gateway moving it runs the response. A status only moves forward: a
/// record that is already terminal is left as it is (so a `cancelled`
/// record is written once, and a result that arrives after a cancel does
/// not revive the response), whichever gateway wrote it. The read and the
/// write are two round trips to the store; the window between them is the
/// one a cross-gateway race can fall into.
pub(crate) async fn set_stored_status(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    lease: Option<&Lease>,
    request_context: Option<StorageRequestContext>,
) -> Result<Transition, String> {
    let update = async {
        let response_id = ResponseId::from(id);
        let Some(mut stored) = storage
            .get_response(&response_id)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(Transition::Missing);
        };
        if is_terminal(stored.raw_response["status"].as_str().unwrap_or("")) {
            return Ok(Transition::Terminal(stored.raw_response));
        }
        stored.raw_response = with_status(stored.raw_response, status, error);
        if let Some(lease) = lease {
            with_lease(&mut stored.raw_response, lease);
        }
        let raw = stored.raw_response.clone();
        storage
            .store_response(stored)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Transition::Moved(raw))
    };
    match request_context {
        Some(ctx) => with_request_context(ctx, update).await,
        None => update.await,
    }
}

/// Whether the record under `id` has reached a terminal status (closed by a
/// cancel, here or on another gateway sharing the store); the result of a
/// work that ends after that is not stored over it.
pub(crate) async fn is_closed(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    request_context: Option<StorageRequestContext>,
) -> bool {
    let read = async { storage.get_response(&ResponseId::from(id)).await };
    let stored = match request_context {
        Some(ctx) => with_request_context(ctx, read).await,
        None => read.await,
    };
    matches!(
        stored,
        Ok(Some(stored)) if is_terminal(stored.raw_response["status"].as_str().unwrap_or(""))
    )
}

/// Moves the record under `id` to the terminal `status` (see
/// [`set_stored_status`]); a record that is already terminal stays.
async fn close(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    status: &str,
    error: Option<Value>,
    request_context: Option<StorageRequestContext>,
) {
    match set_stored_status(storage, id, status, error, None, request_context).await {
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
    close(storage, id, "failed", Some(error), request_context).await;
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
/// [`STATUS_WATCH`]) stops the work and is not written again.
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
    match set_stored_status(
        &storage,
        &id,
        "in_progress",
        None,
        slot.lease().as_ref(),
        request_context.clone(),
    )
    .await
    {
        Ok(Transition::Moved(_)) => {}
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
            warn!(response_id = %id, "No record stored under the background response id");
        }
        Err(e) => {
            warn!(response_id = %id, error = %e, "Failed to mark the background response in progress");
        }
    }
    let closed_elsewhere = watch_stored_status(
        &storage,
        &id,
        request_context.clone(),
        watch,
        slot.registry(),
    );
    let work = AssertUnwindSafe(work).catch_unwind();
    tokio::select! {
        biased;
        () = token.cancelled() => {
            info!(response_id = %id, "Background response cancelled");
            close(&storage, &id, "cancelled", None, request_context).await;
        }
        status = closed_elsewhere => {
            info!(response_id = %id, status = %status, "Background response was closed elsewhere; its work is dropped");
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
                match set_stored_status(&storage, &id, "failed", Some(error), None, request_context).await {
                    Ok(Transition::Terminal(_)) => debug!(response_id = %id, "Background response finished"),
                    Ok(Transition::Moved(_)) => {
                        warn!(response_id = %id, "Background response ended without a stored result; marked failed");
                    }
                    Ok(Transition::Missing) => {
                        warn!(response_id = %id, "Background response ended with no record under its id");
                    }
                    Err(e) => warn!(response_id = %id, error = %e, "Failed to read the background response record back"),
                }
            }
            Ok(Err(error)) => {
                info!(response_id = %id, error = %error, "Background response failed");
                close(&storage, &id, "failed", Some(error), request_context).await;
            }
            Err(panic) => {
                let message = panic_message(panic.as_ref());
                warn!(response_id = %id, panic = %message, "Background response task panicked");
                let error = json!({
                    "code": "server_error",
                    "message": format!("The response's task panicked: {message}"),
                });
                close(&storage, &id, "failed", Some(error), request_context).await;
            }
        },
    }
    drop(slot);
}

/// Resolves with the status once the record under `id` reads terminal;
/// while it is open, renews `registry`'s lease on it every
/// [`BackgroundResponses::renew_every`] (a renewal that fails is retried at
/// the next read; a lease left to expire lets the reaper close the record,
/// which this watch then reads).
async fn watch_stored_status(
    storage: &Arc<dyn ResponseStorage>,
    id: &str,
    request_context: Option<StorageRequestContext>,
    every: Duration,
    registry: &BackgroundResponses,
) -> String {
    let renew_every = registry.renew_every();
    let mut renewed = Instant::now();
    loop {
        tokio::time::sleep(every).await;
        let read = async { storage.get_response(&ResponseId::from(id)).await };
        let stored = match request_context.clone() {
            Some(ctx) => with_request_context(ctx, read).await,
            None => read.await,
        };
        match stored {
            Ok(Some(mut stored)) => {
                let status = stored.raw_response["status"].as_str().unwrap_or("");
                if is_terminal(status) {
                    return status.to_string();
                }
                let Some(lease) = registry.lease() else {
                    continue;
                };
                if renewed.elapsed() < renew_every {
                    continue;
                }
                with_lease(&mut stored.raw_response, &lease);
                let write = async { storage.store_response(stored).await };
                let written = match request_context.clone() {
                    Some(ctx) => with_request_context(ctx, write).await,
                    None => write.await,
                };
                match written {
                    Ok(_) => renewed = Instant::now(),
                    Err(e) => {
                        debug!(response_id = %id, error = %e, "Failed to renew the lease on the background response record");
                    }
                }
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

// ============================================================================
// Leases and the reaper
// ============================================================================

/// The lease a gateway holds on an open background record: the id of the
/// registry running the response and the time (unix seconds) the lease
/// expires at unless it is renewed. A record whose lease has expired belongs
/// to a gateway that stopped, and the reaper of any gateway sharing the
/// store closes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Lease {
    pub owner: String,
    pub expires_at: i64,
}

impl Lease {
    /// The lease a stored object carries, if any (a record written before
    /// leases existed carries none).
    pub fn of(raw: &Value) -> Option<Self> {
        let lease = raw.get(LEASE_KEY)?;
        Some(Self {
            owner: lease["owner"].as_str()?.to_string(),
            expires_at: lease["expires_at"].as_i64()?,
        })
    }

    /// Whether the lease has run out at `now` (unix seconds).
    pub fn expired_at(&self, now: i64) -> bool {
        self.expires_at <= now
    }
}

/// Stamps `lease` on the stored object.
fn with_lease(raw: &mut Value, lease: &Lease) {
    if let Some(object) = raw.as_object_mut() {
        object.insert(
            LEASE_KEY.into(),
            json!({ "owner": lease.owner, "expires_at": lease.expires_at }),
        );
    }
}

/// The stored object as a client sees it: without the gateway's lease.
pub(crate) fn public_object(mut raw: Value) -> Value {
    if let Some(object) = raw.as_object_mut() {
        object.remove(LEASE_KEY);
    }
    raw
}

/// The `error` of a record closed by the reaper.
fn lost_gateway_error(lease: &Lease) -> Value {
    json!({
        "code": "gateway_lost",
        "message": format!(
            "The gateway running this background response (lease owner {}) stopped renewing its lease (expired at {} unix seconds) and is presumed lost; the response was abandoned.",
            lease.owner, lease.expires_at
        ),
    })
}

/// What one reaper pass found and did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reaped {
    /// Open background records read.
    pub open: usize,
    /// Those whose lease had expired.
    pub expired: usize,
    /// Those this pass closed as `failed`.
    pub closed: usize,
}

/// One reaper pass at `now` (unix seconds): reads the open background
/// records of the store and closes each whose lease has expired as `failed`
/// with [`lost_gateway_error`]; a record that closed meanwhile is left as it
/// is ([`set_stored_status`]), and a record without a lease (written before
/// leases existed) is left to a client's cancel.
pub(crate) async fn reap_expired(
    storage: &Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    now: i64,
) -> Result<Reaped, String> {
    let list = async { storage.list_open_background_responses(REAP_BATCH).await };
    let open = match request_context.clone() {
        Some(ctx) => with_request_context(ctx, list).await,
        None => list.await,
    }
    .map_err(|e| e.to_string())?;
    let mut reaped = Reaped {
        open: open.len(),
        ..Reaped::default()
    };
    for stored in open {
        let id = stored.id.0;
        let Some(lease) = Lease::of(&stored.raw_response) else {
            debug!(response_id = %id, "Open background response record without a lease; left to a cancel");
            continue;
        };
        if !lease.expired_at(now) {
            continue;
        }
        reaped.expired += 1;
        match set_stored_status(
            storage,
            &id,
            "failed",
            Some(lost_gateway_error(&lease)),
            None,
            request_context.clone(),
        )
        .await
        {
            Ok(Transition::Moved(_)) => {
                reaped.closed += 1;
                warn!(
                    response_id = %id,
                    owner = %lease.owner,
                    expired_at = lease.expires_at,
                    "Background response abandoned by a lost gateway; closed as failed"
                );
            }
            Ok(Transition::Terminal(_) | Transition::Missing) => {}
            Err(e) => warn!(
                response_id = %id,
                error = %e,
                "Failed to close the background response record of a lost gateway"
            ),
        }
    }
    Ok(reaped)
}

/// Runs [`reap_expired`] every [`BackgroundResponses::reap_every`] for as
/// long as the gateway runs; nothing when background mode or the lease is
/// off. The start is logged with the registry's owner id, the one the
/// `gateway_lost` errors of this gateway's responses would name.
#[expect(
    clippy::disallowed_methods,
    reason = "the reaper is a maintenance loop of the gateway; it writes nothing a shutdown has to wait for"
)]
pub(crate) fn spawn_reaper(
    storage: Arc<dyn ResponseStorage>,
    request_context: Option<StorageRequestContext>,
    registry: &BackgroundResponses,
) {
    let Some(every) = registry.reap_every() else {
        return;
    };
    info!(
        owner = registry.owner(),
        lease_ttl_secs = registry.lease_ttl.as_secs(),
        every_secs = every.as_secs_f64(),
        "Background response reaper started"
    );
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(every).await;
            match reap_expired(
                &storage,
                request_context.clone(),
                chrono::Utc::now().timestamp(),
            )
            .await
            {
                Ok(reaped) if reaped.closed > 0 => {
                    info!(
                        open = reaped.open,
                        expired = reaped.expired,
                        closed = reaped.closed,
                        "Background response reaper pass"
                    );
                }
                Ok(reaped) => {
                    debug!(
                        open = reaped.open,
                        expired = reaped.expired,
                        "Background response reaper pass"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Background response reaper could not read the open records");
                }
            }
        }
    });
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

/// `POST /v1/responses/{id}/cancel`: a response this gateway runs is stopped
/// and answered as `cancelled` once its task has written the record; a
/// `queued` or `in_progress` background record no task runs here (a response
/// of another gateway sharing the store, or one lost with a restart) is
/// closed as `cancelled`, which the gateway running it reads within
/// [`STATUS_WATCH`] and drops its work on; a response that already ended, or
/// one that never ran in the background, is refused.
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
    let background = raw["background"].as_bool().unwrap_or(false);
    match raw["status"].as_str().unwrap_or("unknown") {
        "queued" | "in_progress" if background => {
            match set_stored_status(
                &ctx.response_storage,
                response_id,
                "cancelled",
                None,
                None,
                ctx.request_context.clone(),
            )
            .await
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
                Err(e) => error::internal_error(
                    "cancel_response_failed",
                    format!("Failed to cancel response: {e}"),
                ),
            }
        }
        _ => answer_for_closed(raw),
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use smg_data_connector::{MemoryResponseStorage, ResponseStorageError, StoredResponse};
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

    const LEASE: Duration = Duration::from_secs(30);

    fn lease(owner: &str, expires_at: i64) -> Lease {
        Lease {
            owner: owner.to_string(),
            expires_at,
        }
    }

    async fn store_queued_with_lease(
        storage: &Arc<dyn ResponseStorage>,
        id: &str,
        lease: Option<&Lease>,
    ) {
        let request = request();
        store_record(
            storage,
            &queued_response(id, &request),
            &request,
            lease,
            None,
        )
        .await
        .unwrap();
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
        let request = request();
        store_record(
            storage,
            &queued_response(id, &request),
            &request,
            None,
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
            limit: Option<usize>,
        ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
            self.inner
                .list_identifier_responses(identifier, limit)
                .await
        }

        async fn delete_identifier_responses(
            &self,
            identifier: &str,
        ) -> Result<usize, ResponseStorageError> {
            self.inner.delete_identifier_responses(identifier).await
        }
    }

    #[test]
    fn the_registry_admits_up_to_its_cap_and_frees_a_dropped_slot() {
        let registry = Arc::new(BackgroundResponses::new(2, LEASE));
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
        assert!(!BackgroundResponses::new(0, LEASE).enabled());
    }

    #[tokio::test]
    async fn cancel_fires_the_token_and_the_receiver_settles_when_the_slot_drops() {
        let registry = Arc::new(BackgroundResponses::new(8, LEASE));
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

    #[test]
    fn a_terminal_status_drops_the_lease_and_the_public_object_never_shows_it() {
        let mut raw = json!({"id": "resp_1", "status": "queued", "background": true});
        with_lease(&mut raw, &lease("gw_a", 1_700_000_000));
        assert_eq!(
            Lease::of(&raw),
            Some(lease("gw_a", 1_700_000_000)),
            "the lease is read back from the stored object"
        );
        let served = public_object(raw.clone());
        assert!(served.get(LEASE_KEY).is_none(), "{served}");
        assert_eq!(served["status"], "queued");

        let in_progress = with_status(raw.clone(), "in_progress", None);
        assert!(
            Lease::of(&in_progress).is_some(),
            "an open record keeps its lease"
        );
        let completed = with_status(raw, "completed", None);
        assert!(
            Lease::of(&completed).is_none(),
            "a closed record needs none"
        );
        assert!(Lease::of(&json!({"status": "queued"})).is_none());
        assert!(lease("gw_a", 10).expired_at(10));
        assert!(!lease("gw_a", 11).expired_at(10));
    }

    #[test]
    fn the_registry_derives_its_lease_and_its_periods_from_the_ttl() {
        let registry = BackgroundResponses::new(4, Duration::from_secs(30));
        let lease = registry.lease().expect("leases are on");
        assert_eq!(lease.owner, registry.owner());
        let now = chrono::Utc::now().timestamp();
        assert!(
            (now + 29..=now + 31).contains(&lease.expires_at),
            "{lease:?}"
        );
        assert_eq!(registry.renew_every(), Duration::from_secs(10));
        assert_eq!(registry.reap_every(), Some(Duration::from_secs(15)));
        assert_ne!(
            BackgroundResponses::new(4, LEASE).owner(),
            registry.owner(),
            "every registry is its own owner"
        );

        let short = BackgroundResponses::new(4, Duration::from_millis(1500));
        assert_eq!(short.renew_every(), Duration::from_secs(1));
        assert_eq!(short.reap_every(), Some(Duration::from_secs(1)));

        let off = BackgroundResponses::new(4, Duration::ZERO);
        assert!(off.lease().is_none());
        assert!(off.reap_every().is_none(), "no lease, no reaper");
        assert!(
            BackgroundResponses::new(0, LEASE).reap_every().is_none(),
            "background mode off, no reaper"
        );
    }

    #[tokio::test]
    async fn the_queued_record_carries_the_lease_through_in_progress_and_the_runner_renews_it() {
        let storage = storage();
        let registry = Arc::new(BackgroundResponses::new(4, Duration::from_secs(3)));
        let first = registry.lease().unwrap();
        store_queued_with_lease(&storage, "resp_1", Some(&first)).await;
        assert_eq!(
            Lease::of(&stored_raw(&storage, "resp_1").await),
            Some(first.clone())
        );

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
            Duration::from_millis(20),
        ));
        wait_for_status(&storage, "resp_1", "in_progress").await;
        let in_progress = Lease::of(&stored_raw(&storage, "resp_1").await).expect("leased");
        assert_eq!(in_progress.owner, registry.owner());

        // The watch renews the lease every third of the TTL (1 s here).
        let renewed = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let now = Lease::of(&stored_raw(&storage, "resp_1").await).expect("leased");
                if now.expires_at > in_progress.expires_at {
                    return now;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("the lease is renewed while the work runs");
        assert_eq!(renewed.owner, registry.owner());

        registry.cancel("resp_1").expect("running");
        task.await.unwrap();
        dropped_rx.await.expect("the work was dropped");
        let raw = stored_raw(&storage, "resp_1").await;
        assert_eq!(raw["status"], "cancelled");
        assert!(
            Lease::of(&raw).is_none(),
            "the closed record dropped its lease: {raw}"
        );
    }

    #[tokio::test]
    async fn the_reaper_closes_the_records_of_an_expired_lease_as_failed() {
        let storage = storage();
        let now = 1_700_000_000;
        store_queued_with_lease(&storage, "resp_lost", Some(&lease("gw_gone", now - 1))).await;
        set_stored_status(
            &storage,
            "resp_lost",
            "in_progress",
            None,
            Some(&lease("gw_gone", now - 1)),
            None,
        )
        .await
        .unwrap();
        store_queued_with_lease(&storage, "resp_live", Some(&lease("gw_here", now + 20))).await;
        store_queued_with_lease(&storage, "resp_old", None).await;
        store_queued_with_lease(&storage, "resp_done", Some(&lease("gw_gone", now - 1))).await;
        set_stored_status(&storage, "resp_done", "completed", None, None, None)
            .await
            .unwrap();

        let reaped = reap_expired(&storage, None, now).await.unwrap();
        assert_eq!(
            reaped,
            Reaped {
                open: 3,
                expired: 1,
                closed: 1
            }
        );
        let lost = stored_raw(&storage, "resp_lost").await;
        assert_eq!(lost["status"], "failed");
        assert_eq!(lost["error"]["code"], "gateway_lost");
        assert!(
            lost["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("gw_gone")),
            "{lost}"
        );
        assert!(lost["completed_at"].is_i64());
        assert!(Lease::of(&lost).is_none());
        assert_eq!(stored_status(&storage, "resp_live").await, "queued");
        assert_eq!(
            stored_status(&storage, "resp_old").await,
            "queued",
            "a record without a lease is left to a cancel"
        );
        assert_eq!(stored_status(&storage, "resp_done").await, "completed");

        // The next pass finds nothing to do; the live lease expires later.
        assert_eq!(
            reap_expired(&storage, None, now).await.unwrap(),
            Reaped {
                open: 2,
                expired: 0,
                closed: 0
            }
        );
        let later = reap_expired(&storage, None, now + 20).await.unwrap();
        assert_eq!(later.closed, 1);
        assert_eq!(stored_status(&storage, "resp_live").await, "failed");
    }

    #[tokio::test]
    async fn a_reaped_record_stops_the_gateway_that_still_runs_it() {
        // The owner renews its lease, but the reaper (with a clock far
        // ahead, as after a long stall of the owner) closes the record: the
        // owner reads the closed record and drops its work.
        let storage = storage();
        let owner = Arc::new(BackgroundResponses::new(4, LEASE));
        store_queued_with_lease(&storage, "resp_1", owner.lease().as_ref()).await;
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

        let far_ahead = chrono::Utc::now().timestamp() + 3600;
        let reaped = reap_expired(&storage, None, far_ahead).await.unwrap();
        assert_eq!(reaped.closed, 1);

        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the owner stops within its watch interval")
            .unwrap();
        dropped_rx.await.expect("the work was dropped");
        let raw = stored_raw(&storage, "resp_1").await;
        assert_eq!(raw["status"], "failed");
        assert_eq!(raw["error"]["code"], "gateway_lost");
        assert_eq!(owner.in_flight(), 0);
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
            set_stored_status(&storage, "resp_1", "in_progress", None, None, None)
                .await
                .unwrap()
        else {
            panic!("a queued record moves to in_progress");
        };
        assert_eq!(raw["status"], "in_progress");
        assert_eq!(stored_status(&storage, "resp_1").await, "in_progress");
        assert!(matches!(
            set_stored_status(&storage, "resp_404", "cancelled", None, None, None)
                .await
                .unwrap(),
            Transition::Missing
        ));
    }

    #[tokio::test]
    async fn a_terminal_record_is_never_rewritten() {
        let storage = storage();
        store_queued(&storage, "resp_1").await;
        let Transition::Moved(raw) =
            set_stored_status(&storage, "resp_1", "cancelled", None, None, None)
                .await
                .unwrap()
        else {
            panic!("a queued record is cancelled");
        };
        assert_eq!(raw["status"], "cancelled");
        assert!(is_closed(&storage, "resp_1", None).await);

        for status in ["failed", "in_progress", "completed", "cancelled"] {
            let error = (status == "failed").then(|| json!({"code": "x", "message": "y"}));
            let Transition::Terminal(raw) =
                set_stored_status(&storage, "resp_1", status, error, None, None)
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
        assert!(!is_closed(&storage, "resp_404", None).await);
    }

    #[tokio::test]
    async fn a_cancel_marks_the_record_cancelled_and_drops_the_work() {
        let storage = storage();
        store_queued(&storage, "resp_1").await;
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
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

    #[tokio::test]
    async fn a_failed_work_leaves_a_failed_record_with_its_error() {
        let storage = storage();
        store_queued(&storage, "resp_1").await;
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
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
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
        let slot = registry.admit("resp_1").unwrap();
        let store = storage.clone();
        drive(storage.clone(), None, slot, async move {
            set_stored_status(&store, "resp_1", "completed", None, None, None)
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
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
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
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
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
        let owner = Arc::new(BackgroundResponses::new(4, LEASE));
        let other = Arc::new(BackgroundResponses::new(4, LEASE));
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
        let Transition::Moved(raw) =
            set_stored_status(&storage, "resp_1", "cancelled", None, None, None)
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
        set_stored_status(&storage, "resp_1", "cancelled", None, None, None)
            .await
            .unwrap();
        let registry = Arc::new(BackgroundResponses::new(4, LEASE));
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
}
