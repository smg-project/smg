//! Concurrent controls and a bounded layout relay. No Accepted event is a dispatch barrier.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use futures::{Stream, StreamExt};
use smg_grpc_client::{
    sglang_managed::{
        proto::{
            self, control_event::Event, decode_command::Command,
            managed_encoder_client::ManagedEncoderClient,
            managed_sglang_client::ManagedSglangClient,
        },
        validate_manifest, Limits,
    },
    sglang_proto,
};
use tokio::{
    sync::{mpsc, Semaphore},
    task::JoinSet,
    time::{timeout, Instant},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{transport::Channel, Request, Status};

use super::planner::Plan;

/// The response guard closes the attempt, including disconnect before the first token.
pub struct ResponseStream {
    responses: ReceiverStream<Result<sglang_proto::GenerateResponse, Status>>,
    cancel: CancellationToken,
}

impl Stream for ResponseStream {
    type Item = Result<sglang_proto::GenerateResponse, Status>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.responses).poll_next(cx)
    }
}

impl Drop for ResponseStream {
    /// Client response disposal signals the independent cleanup owner, without aborting its task.
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Encode,
    Prefill,
    Decode,
}

pub struct Executor {
    supervisors: Arc<Semaphore>,
    limits: Limits,
    cleanup_timeout: Duration,
}

/// Local reservations reconcile with worker coverage and actual termination receipts.
pub trait AttemptObserver: Send + Sync {
    /// Clone the configured endpoint's multiplexed channel without connecting on the caller path.
    fn channel(&self, endpoint: &str) -> Result<Channel, Status>;
    /// An event has already passed its operation fence and monotonic sequence check.
    fn event(&self, event: &proto::ControlEvent) -> Result<(), Status>;
    /// False or missing receipts remain charged; a cancellation acknowledgement is insufficient.
    fn termination(&self, operation: &proto::OperationRef, quiesced: bool);
}

impl Executor {
    /// Supervisor permits remain owned through cleanup, bounding disconnected work too.
    pub fn new(
        max_attempts: usize,
        limits: Limits,
        cleanup_timeout: Duration,
    ) -> Result<Self, Status> {
        if max_attempts == 0 || cleanup_timeout.is_zero() {
            return Err(Status::invalid_argument(
                "invalid managed supervisor limits",
            ));
        }
        Ok(Self {
            supervisors: Arc::new(Semaphore::new(max_attempts)),
            limits,
            cleanup_timeout,
        })
    }

    /// All control tasks are registered before polling any role's response.
    pub fn execute(
        &self,
        plan: Arc<Plan>,
        prefill: proto::PrefillStart,
        decode: proto::DecodeStart,
        encode: Option<proto::EncodeRequest>,
        remaining: Duration,
        observer: Arc<dyn AttemptObserver>,
    ) -> Result<ResponseStream, Status> {
        if remaining.is_zero() {
            return Err(Status::deadline_exceeded("managed request expired"));
        }
        if prefill.binding.as_ref() != Some(plan.binding())
            || decode.binding.as_ref() != Some(plan.binding())
            || prefill.encode_plan.as_ref() != Some(plan.external())
            || encode.is_some() != plan.encoder().is_some()
        {
            return Err(Status::invalid_argument(
                "managed controls do not match frozen plan",
            ));
        }
        if let Some(request) = &encode {
            if request.plan.as_ref() != Some(plan.external())
                || request.receiver_binding.as_ref() != Some(plan.binding())
            {
                return Err(Status::invalid_argument(
                    "managed Encode does not match frozen plan",
                ));
            }
        }
        let permit = Arc::clone(&self.supervisors)
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("managed supervisors full"))?;
        let (output, responses) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let limits = self.limits;
        let cleanup_timeout = self.cleanup_timeout;
        // The permit and JoinSet live in the supervisor, never in the caller's future.
        #[expect(
            clippy::disallowed_methods,
            reason = "bounded supervisor retains cancellation ownership after client disconnect"
        )]
        tokio::spawn(async move {
            let _permit = permit;
            let deadline = Instant::now() + remaining;
            let terminations = termination_requests(&plan, &prefill, &decode, encode.as_ref());
            let result = supervise(
                &plan,
                prefill,
                decode,
                encode,
                deadline,
                limits,
                &token,
                &output,
                observer.as_ref(),
            )
            .await;
            token.cancel();
            if let Err(error) = result {
                let _ = timeout(Duration::from_secs(1), output.send(Err(error))).await;
            }
            // HTTP completion must not wait for the independent remote cleanup owner.
            drop(output);
            // The bounded supervisor remains the reconciliation owner until every
            // instance acknowledges safe release. A network timeout is never a receipt.
            futures::future::join_all(terminations.into_iter().map(
                |(role, endpoint, request)| {
                    let observer = Arc::clone(&observer);
                    async move {
                        let Some(operation) = request.operation.clone() else { return; };
                        let channel = match observer.channel(&endpoint) {
                            Ok(channel) => channel,
                            Err(error) => {
                                tracing::error!(%error, "invalid frozen cleanup endpoint");
                                return;
                            }
                        };
                        let mut backoff = Duration::from_millis(50);
                        loop {
                            let reconcile = async {
                                if role == Role::Encode {
                                    ManagedEncoderClient::new(channel.clone())
                                        .cancel(request.clone()).await.map(|reply| reply.into_inner().quiesced)
                                } else {
                                    ManagedSglangClient::new(channel.clone())
                                        .terminate(request.clone()).await.map(|reply| reply.into_inner().quiesced)
                                }
                            };
                            match timeout(cleanup_timeout, reconcile).await {
                                Ok(Ok(true)) => {
                                    observer.termination(&operation, true);
                                    break;
                                }
                                Ok(Ok(false)) => {}
                                outcome => tracing::warn!(?outcome, operation_id = operation.operation_id,
                                    "managed cleanup unconfirmed; retaining bounded reconciliation owner"),
                            }
                            tokio::time::sleep(backoff).await;
                            backoff = (backoff * 2).min(Duration::from_secs(5));
                        }
                    }
                },
            )).await;
        });
        Ok(ResponseStream {
            responses: ReceiverStream::new(responses),
            cancel,
        })
    }
}

/// Retain each target fence even if its control request has not reached the remote yet.
fn termination_requests(
    plan: &Plan,
    prefill: &proto::PrefillStart,
    decode: &proto::DecodeStart,
    encode: Option<&proto::EncodeRequest>,
) -> Vec<(Role, String, proto::Termination)> {
    let mut result = vec![
        (
            Role::Prefill,
            plan.prefill().control_endpoint.clone(),
            proto::Termination {
                operation: prefill.operation.clone(),
                original_expiry_unix_ms: prefill
                    .budget
                    .as_ref()
                    .map_or(0, |b| b.original_expiry_unix_ms),
                native_rid: plan.binding().prefill_rid.clone(),
                reason: "attempt closed".into(),
            },
        ),
        (
            Role::Decode,
            plan.decode().control_endpoint.clone(),
            proto::Termination {
                operation: plan.binding().decode.clone(),
                original_expiry_unix_ms: decode
                    .budget
                    .as_ref()
                    .map_or(0, |b| b.original_expiry_unix_ms),
                native_rid: decode
                    .generation
                    .as_ref()
                    .map_or_else(String::new, |g| g.request_id.clone()),
                reason: "attempt closed".into(),
            },
        ),
    ];
    if let (Some(worker), Some(request)) = (plan.encoder(), encode) {
        result.push((
            Role::Encode,
            worker.control_endpoint.clone(),
            proto::Termination {
                operation: request.operation.clone(),
                original_expiry_unix_ms: request
                    .budget
                    .as_ref()
                    .map_or(0, |b| b.original_expiry_unix_ms),
                native_rid: plan
                    .external()
                    .parts
                    .first()
                    .map_or_else(String::new, |p| p.native_part_rid.clone()),
                reason: "attempt closed".into(),
            },
        ));
    }
    result
}

/// Validate source-local ordering before events reach the supervisor.
async fn forward(
    role: Role,
    expected: Option<proto::OperationRef>,
    mut stream: tonic::Streaming<proto::ControlEvent>,
    events: mpsc::Sender<(Role, proto::ControlEvent)>,
) -> Result<Role, Status> {
    let mut previous = 0;
    let mut terminal = false;
    while let Some(event) = stream.next().await {
        let event = event?;
        if event.operation != expected || event.event_seq != previous + 1 {
            return Err(Status::data_loss("managed event source/sequence mismatch"));
        }
        previous = event.event_seq;
        terminal |= match (&event.event, role) {
            (Some(Event::Stage(stage)), Role::Encode) => *stage == proto::Stage::Delivered as i32,
            (Some(Event::Generation(response)), Role::Prefill | Role::Decode) => matches!(
                response.response,
                Some(sglang_proto::generate_response::Response::Complete(_))
            ),
            (Some(Event::Failed(_)), _) => true,
            _ => false,
        };
        events
            .send((role, event))
            .await
            .map_err(|_| Status::cancelled("managed supervisor closed"))?;
    }
    if terminal {
        Ok(role)
    } else {
        Err(Status::data_loss(
            "managed role ended without terminal result",
        ))
    }
}

/// Layout, output and failures progress together; controls never enter the client stream.
#[expect(
    clippy::too_many_arguments,
    reason = "one supervisor owns the three controls, frozen plan, output and cancellation"
)]
async fn supervise(
    plan: &Plan,
    prefill: proto::PrefillStart,
    decode: proto::DecodeStart,
    encode: Option<proto::EncodeRequest>,
    deadline: Instant,
    limits: Limits,
    cancel: &CancellationToken,
    output: &mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
    observer: &dyn AttemptObserver,
) -> Result<(), Status> {
    let mut tasks = JoinSet::new();
    let (events_tx, mut events) = mpsc::channel(16);
    // Start and at most one manifest; a manifest may arrive before D connects.
    let (commands, command_rx) = mpsc::channel(2);
    commands
        .try_send(proto::DecodeCommand {
            command: Some(Command::Start(decode.clone())),
        })
        .map_err(|_| Status::internal("managed Start queue failed"))?;
    let channel = observer.channel(&plan.decode().control_endpoint)?;
    let sender = events_tx.clone();
    let operation = plan.binding().decode.clone();
    tasks.spawn(async move {
        let mut client = ManagedSglangClient::new(channel);
        let mut request = Request::new(ReceiverStream::new(command_rx));
        request.set_timeout(deadline.saturating_duration_since(Instant::now()));
        let stream = client.generate_decode(request).await?.into_inner();
        forward(Role::Decode, operation, stream, sender).await
    });
    let channel = observer.channel(&plan.prefill().control_endpoint)?;
    let sender = events_tx.clone();
    tasks.spawn(async move {
        let mut client = ManagedSglangClient::new(channel);
        let operation = prefill.operation.clone();
        let mut request = Request::new(prefill);
        request.set_timeout(deadline.saturating_duration_since(Instant::now()));
        let stream = client.generate_prefill(request).await?.into_inner();
        forward(Role::Prefill, operation, stream, sender).await
    });
    if let (Some(worker), Some(request)) = (plan.encoder(), encode) {
        let channel = observer.channel(&worker.control_endpoint)?;
        let sender = events_tx.clone();
        tasks.spawn(async move {
            let mut client = ManagedEncoderClient::new(channel);
            let operation = request.operation.clone();
            let mut request = Request::new(request);
            request.set_timeout(deadline.saturating_duration_since(Instant::now()));
            let stream = client.encode(request).await?.into_inner();
            forward(Role::Encode, operation, stream, sender).await
        });
    }
    drop(events_tx);
    let dispatched = Instant::now();
    let mut metadata_published = None;
    let mut bound_digest = None;
    let mut prefill_complete = false;
    let mut encode_complete = plan.encoder().is_none();
    let mut decode_complete = None;
    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(Status::cancelled("managed client disconnected")),
            () = tokio::time::sleep_until(deadline) => return Err(Status::deadline_exceeded("managed original deadline expired")),
            event = events.recv() => {
                let Some((role, event)) = event else {
                    return Err(Status::data_loss("managed controls ended without all role completions"));
                };
                observer.event(&event)?;
                if let Some(Event::Stage(stage)) = &event.event {
                    let name = proto::Stage::try_from(*stage).map_or("UNKNOWN", |value| value.as_str_name());
                    let role_name = match role { Role::Encode => "encode", Role::Prefill => "prefill", Role::Decode => "decode" };
                    metrics::histogram!("smg_managed_epd_stage_seconds", "role" => role_name, "stage" => name).record(dispatched.elapsed().as_secs_f64());
                    tracing::info!(role = role_name, stage = name, operation = event.operation.as_ref().map(|op| op.operation_id.as_str()), elapsed_ms = dispatched.elapsed().as_millis() as u64, "managed EPD stage observed");
                    if role == Role::Decode && *stage == proto::Stage::InputBound as i32 {
                        if let Some(published) = metadata_published {
                            metrics::histogram!("smg_managed_epd_manifest_relay_seconds").record(Instant::now().saturating_duration_since(published).as_secs_f64());
                        }
                    }
                }
                match event.event {
                    Some(Event::MediaPrepared(input)) if role == Role::Prefill => {
                        validate_manifest(&input, plan.binding(), limits)?;
                        if let Some(digest) = &bound_digest {
                            if digest != &input.integrity_digest { return Err(Status::data_loss("conflicting P manifests")); }
                            continue;
                        }
                        metadata_published = Some(Instant::now());
                        bound_digest = Some(input.integrity_digest.clone());
                        commands.try_send(proto::DecodeCommand { command: Some(Command::BindPreparedInput(input)) }).map_err(|_| Status::resource_exhausted("managed input relay closed/full"))?;
                    }
                    Some(Event::Generation(response)) if role == Role::Decode => {
                        if matches!(response.response, Some(sglang_proto::generate_response::Response::Complete(_))) {
                            decode_complete = Some(response);
                        } else {
                        tokio::select! {
                            result = output.send(Ok(response)) => result.map_err(|_| Status::cancelled("managed client disconnected"))?,
                            () = cancel.cancelled() => return Err(Status::cancelled("managed client disconnected")),
                            () = tokio::time::sleep_until(deadline) => return Err(Status::deadline_exceeded("managed output deadline expired")),
                        }
                        }
                    }
                    Some(Event::Generation(response)) if role == Role::Prefill => {
                        prefill_complete |= matches!(response.response, Some(sglang_proto::generate_response::Response::Complete(_)));
                    }
                    Some(Event::Stage(stage)) if role == Role::Encode && stage == proto::Stage::Delivered as i32 => encode_complete = true,
                    Some(Event::Failed(failure)) => return Err(Status::unavailable(format!("managed {role:?}: {}: {}", failure.code, failure.message))),
                    Some(Event::Stage(_)) | Some(Event::Generation(_)) => {}
                    _ => return Err(Status::data_loss("unexpected managed control event")),
                }
                // D starts immediately and streams tokens immediately. Only its final
                // success waits for the other owners, so a late P/E failure is visible.
                if prefill_complete && encode_complete {
                    if let Some(response) = decode_complete.take() {
                        tokio::select! {
                            result = output.send(Ok(response)) => result.map_err(|_| Status::cancelled("managed client disconnected"))?,
                            () = cancel.cancelled() => return Err(Status::cancelled("managed client disconnected")),
                            () = tokio::time::sleep_until(deadline) => return Err(Status::deadline_exceeded("managed output deadline expired")),
                        }
                        return Ok(());
                    }
                }
            }
            joined = tasks.join_next(), if !tasks.is_empty() => {
                match joined {
                    Some(Ok(Ok(_))) => {}
                    Some(Ok(Err(error))) => return Err(error),
                    Some(Err(error)) => return Err(Status::internal(format!("managed role task failed: {error}"))),
                    None => {}
                }
            }
        }
    }
}
