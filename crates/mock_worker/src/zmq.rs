//! Mock ZMQ EngineCore worker: plays the vLLM `EngineCoreProc` role over ZMQ so
//! SMG's `engine-zmq-client` connector can be exercised without a GPU. Unlike
//! the HTTP/gRPC workers (which bind and are dialed by the gateway), a ZMQ mock
//! engine *dials* the frontend's `ipc://` handshake address, then registers,
//! receives `EngineCoreRequest`s, and pushes `EngineCoreOutputs` — driven by the
//! same continuous-batching [`Engine`] simulator the other protocols use.

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    future::{pending, Future},
    pin::pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
    time::Duration,
};

use engine_zmq_client::{
    codec::OpaqueValue,
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound, MockEngineOutput},
    protocol::vllm::{
        logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
        output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            StopReason, UtilityCallOutput, UtilityOutput,
        },
        stats::{PrefillStats, SchedulerStats},
    },
    EngineId,
};
use tokio::{
    sync::mpsc,
    task::{AbortHandle, JoinSet},
    time::Instant,
};

use crate::{
    config::Config,
    drain::InFlight,
    engine::{Engine, GenEvent, LoadSnapshot, NewRequest},
    script::{self, Answer, MatchedStop},
};

// zeromq queues sends and exposes no flush/linger operation. Keep both sockets
// alive briefly after the last queued output so its background writer and the
// frontend can consume it before input EOF fails the frontend's requests.
// This is best-effort transport linger, not a delivery acknowledgement; the
// process-wide drain deadline still bounds shutdown if the frontend is stuck.
const OUTPUT_LINGER: Duration = Duration::from_secs(1);

/// What a rank has pushed on its output socket: the messages that carried
/// request outputs and the outputs in them (a utility reply is neither). The
/// admin API reads them (`GET /admin/zmq/{worker}`), where a test tells a
/// step of many outputs from one message per token.
#[derive(Default)]
pub struct OutputCounts {
    batch_messages: AtomicU64,
    outputs: AtomicU64,
}

impl OutputCounts {
    /// Count one pushed message.
    fn pushed(&self, message: &EngineCoreOutputs) {
        if let Some(batch) = message.as_request_batch() {
            self.batch_messages.fetch_add(1, Ordering::Relaxed);
            self.outputs
                .fetch_add(batch.outputs.len() as u64, Ordering::Relaxed);
        }
    }

    /// The messages that carried request outputs, and the outputs in them.
    pub fn read(&self) -> (u64, u64) {
        (
            self.batch_messages.load(Ordering::Relaxed),
            self.outputs.load(Ordering::Relaxed),
        )
    }
}

static OUTPUT_COUNTS: OnceLock<Mutex<HashMap<String, Arc<OutputCounts>>>> = OnceLock::new();

/// The counts of the rank named `name` (`zmq:<index>`), created on first use.
fn output_counts(name: &str) -> Arc<OutputCounts> {
    OUTPUT_COUNTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(name.to_string())
        .or_default()
        .clone()
}

/// The counts of a rank that has served in this process, by its name.
pub fn rank_output_counts(name: &str) -> Option<Arc<OutputCounts>> {
    OUTPUT_COUNTS
        .get()?
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(name)
        .cloned()
}

/// Run one mock ZMQ EngineCore rank: connect to `handshake_address`, then serve
/// requests until the frontend disconnects.
pub async fn serve(cfg: Arc<Config>, handshake_address: String, engine_index: u32) {
    serve_until(cfg, handshake_address, engine_index, pending()).await;
}

/// [`serve`] until `shutdown` resolves as well: from then on the rank admits
/// no request and returns once the ones in flight have finished and their
/// last outputs have had a bounded transport linger (the drain of a signal), as an
/// engine process does under SIGTERM.
pub async fn serve_until(
    cfg: Arc<Config>,
    handshake_address: String,
    engine_index: u32,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    if let Err(e) = script::ensure_loaded(&cfg) {
        tracing::error!("zmq engine {engine_index} cannot load script: {e}");
        return;
    }
    // The rank's name in the process (`zmq:<index>`): what the admin API and
    // the script store address it by.
    let name = format!("zmq:{engine_index}");
    script::register(&name);
    let mut ready = default_ready_response();
    ready.data_parallel_rank = engine_index;
    ready.instance_id = format!("mock-zmq-{engine_index}");

    let mock = match connect_to_frontend(
        &handshake_address,
        EngineId::from_engine_index(engine_index),
        ready,
    )
    .await
    {
        Ok(mock) => mock,
        Err(error) => {
            tracing::error!("zmq engine {engine_index} handshake failed: {error}");
            return;
        }
    };
    tracing::info!("zmq mock engine {engine_index} connected to {handshake_address}");

    let (mut input, output) = mock.split();
    let engine = cfg
        .realistic
        .then(|| Engine::spawn_named(cfg.engine.clone(), name.clone(), true));
    // The rank's KV-event publisher, numbered after the gRPC workers and
    // stamping the batches with the rank this engine advertised.
    let publisher = engine.as_ref().and_then(|engine| {
        let index = cfg
            .grpc_count
            .checked_add(u16::try_from(engine_index).ok()?)?;
        let dp_rank = i32::try_from(engine_index).ok()?;
        cfg.kv_zmq_for(index, dp_rank)
            .map(|kv| (engine.clone(), kv))
    });
    #[expect(
        clippy::disallowed_methods,
        reason = "publisher self-terminates when the engine's event channel closes"
    )]
    let _publisher = publisher.map(|(engine, kv)| tokio::spawn(crate::kv_zmq::serve(engine, kv)));

    // A single writer owns the output PUSH socket; per-request forwarders funnel
    // their outputs here so concurrent requests serialize onto the one socket.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<EngineCoreOutputs>();
    // The frontend's aborts take their own lane to the writer: a request's
    // outputs still waiting for a step stand for steps not taken, and only
    // the frontend's abort drops them (an answer the engine itself ends with
    // `Abort` keeps its tokens, as without a period).
    let (abort_tx, mut abort_rx) = mpsc::unbounded_channel::<(String, EngineCoreOutputs)>();
    let step = (cfg.zmq_step_ms > 0).then(|| Duration::from_millis(cfg.zmq_step_ms));
    let counts = output_counts(&name);
    #[expect(
        clippy::disallowed_methods,
        reason = "the writer ends with the output and abort channels; a drain awaits it"
    )]
    let writer = tokio::spawn(async move {
        let mut output = output;
        // With a period, an engine's step clock: what the forwarders push
        // waits in the step queue and every period one message goes out with
        // the next output of every request that has one; a utility reply
        // goes as it came. Without one, every message goes as it came.
        let mut steps = step.map(|period| (StepQueue::default(), period));
        let mut next_step_at: Option<Instant> = None;
        let (mut outputs_open, mut aborts_open) = (true, true);
        while outputs_open
            || aborts_open
            || steps.as_ref().is_some_and(|(queue, _)| !queue.is_empty())
        {
            let due = async {
                match next_step_at {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => pending().await,
                }
            };
            tokio::select! {
                biased;
                () = due => {
                    if let Some((queue, period)) = steps.as_mut() {
                        match queue.next_step() {
                            Some(message) => {
                                if !push(&mut output, &counts, &message).await {
                                    return;
                                }
                                next_step_at = Some(Instant::now() + *period);
                            }
                            None => next_step_at = None,
                        }
                    }
                }
                received = out_rx.recv(), if outputs_open => match (received, steps.as_mut()) {
                    (Some(EngineCoreOutputs::RequestBatch(batch)), Some((queue, period))) => {
                        queue.push(batch);
                        if next_step_at.is_none() {
                            next_step_at = Some(Instant::now() + *period);
                        }
                    }
                    (Some(message), _) => {
                        if !push(&mut output, &counts, &message).await {
                            return;
                        }
                    }
                    (None, _) => outputs_open = false,
                },
                aborted = abort_rx.recv(), if aborts_open => match (aborted, steps.as_mut()) {
                    (Some((request_id, terminal)), Some((queue, period))) => {
                        queue.abort(&request_id);
                        match terminal {
                            EngineCoreOutputs::RequestBatch(batch) => queue.push(batch),
                            other => {
                                if !push(&mut output, &counts, &other).await {
                                    return;
                                }
                            }
                        }
                        if next_step_at.is_none() {
                            next_step_at = Some(Instant::now() + *period);
                        }
                    }
                    (Some((_, terminal)), None) => {
                        if !push(&mut output, &counts, &terminal).await {
                            return;
                        }
                    }
                    (None, _) => aborts_open = false,
                },
            }
        }
        tokio::time::sleep(OUTPUT_LINGER).await;
        drop(output);
    });

    let mut shutdown = pin!(shutdown);
    let mut stopped = false;
    // The forwarders of the admitted requests, and their abort handles by
    // request id. A stop admits nothing more and keeps reading the input
    // (an Abort still lands) until the last forwarder is done.
    let mut forwarders = JoinSet::new();
    let mut inflight: HashMap<String, AbortHandle> = HashMap::new();
    // Whether the loop ended with the last forwarder of a stop (drained) or
    // with the frontend gone.
    let drained = loop {
        let inbound = tokio::select! {
            inbound = input.recv() => inbound,
            () = &mut shutdown, if !stopped => {
                tracing::info!(
                    "zmq engine {engine_index}: stopping, admitting no more requests; {} in flight",
                    forwarders.len()
                );
                stopped = true;
                continue;
            }
            // Finished forwarders are reaped as they go; after the stop the
            // last of them ends the rank (the guard keeps an empty set from
            // yielding `None` before it).
            joined = forwarders.join_next(), if stopped || !forwarders.is_empty() => {
                match joined {
                    Some(Ok(())) => continue,
                    Some(Err(e)) if e.is_panic() => {
                        tracing::error!("zmq engine {engine_index} forwarder panicked: {e}");
                        continue;
                    }
                    // Aborted through its handle (the frontend's Abort below):
                    // the normal end of a cancelled request, not a fault.
                    Some(Err(_)) => continue,
                    None => break true,
                }
            }
        };
        match inbound {
            Ok(EngineInbound::Add(request)) if stopped => {
                // Nothing is admitted during the drain: the frontend's stream
                // for the request ends with a terminal Abort.
                let _ = out_tx.send(terminal_batch(
                    engine_index,
                    request.request_id,
                    EngineCoreFinishReason::Abort,
                ));
            }
            Ok(EngineInbound::Add(request)) => {
                inflight.retain(|_, forwarder| !forwarder.is_finished());
                let request_id = request.request_id.clone();
                let prompt = request.prompt_token_ids.clone().unwrap_or_default();
                // Honor the request's max_tokens; fall back to the worker default.
                let max_new = request
                    .sampling_params
                    .as_ref()
                    .map(|params| params.max_tokens)
                    .filter(|&max| max > 0)
                    .unwrap_or(cfg.output_tokens);

                let in_flight = InFlight::enter();
                // A scripted answer, ahead of either engine mode: the rank's
                // queue or the prompt's key decides ([`crate::script`]).
                let scripted = match script::take_ids(&cfg, &name, &prompt) {
                    Ok(scripted) => scripted,
                    Err(_) => {
                        // The reason is in the log; the request ends as one
                        // an engine failed, which the gateway reports.
                        let _ = out_tx.send(terminal_batch(
                            engine_index,
                            request_id,
                            EngineCoreFinishReason::Error,
                        ));
                        continue;
                    }
                };
                if let Some(answer) = scripted {
                    // The request's limit ends the answer as it would an
                    // engine's; the worker default is the canned path's.
                    let answer = answer.cut_to(
                        request
                            .sampling_params
                            .as_ref()
                            .map(|params| params.max_tokens)
                            .filter(|&max| max > 0),
                    );
                    // `logprobs: n` asks for the n most likely tokens per
                    // position beside the sampled one's own column; `0` for
                    // the sampled one alone; unset for none.
                    let top_logprobs = request
                        .sampling_params
                        .as_ref()
                        .and_then(|params| params.logprobs)
                        .map(|n| usize::try_from(n).unwrap_or(usize::MAX));
                    let forwarder = forwarders.spawn(scripted_request(
                        request_id.clone(),
                        engine_index,
                        prompt.len(),
                        answer,
                        top_logprobs,
                        out_tx.clone(),
                        engine.clone(),
                        in_flight,
                    ));
                    inflight.insert(request_id, forwarder);
                    continue;
                }
                let (gen_tx, gen_rx) = mpsc::unbounded_channel::<GenEvent>();
                match &engine {
                    Some(engine) => engine.submit(NewRequest {
                        request_id: request_id.clone(),
                        prompt_token_ids: prompt,
                        max_new,
                        events: gen_tx,
                    }),
                    None => {
                        let delay = cfg.gen_delay;
                        let prefix = cfg.engine.output_prefix.clone();
                        #[expect(
                            clippy::disallowed_methods,
                            reason = "canned producer self-terminates after Done"
                        )]
                        let _canned = tokio::spawn(canned_generate(gen_tx, max_new, delay, prefix));
                    }
                }

                let forwarder = forwarders.spawn(forward_request(
                    request_id.clone(),
                    engine_index,
                    gen_rx,
                    out_tx.clone(),
                    engine.clone(),
                    in_flight,
                ));
                inflight.insert(request_id, forwarder);
            }
            Ok(EngineInbound::Abort(request_ids)) => {
                for request_id in request_ids {
                    if let Some(handle) = inflight.remove(&request_id) {
                        handle.abort();
                    }
                    // A terminal Abort output ends the frontend's stream; the
                    // writer drops what the request still had waiting.
                    let terminal = terminal_batch(
                        engine_index,
                        request_id.clone(),
                        EngineCoreFinishReason::Abort,
                    );
                    let _ = abort_tx.send((request_id, terminal));
                }
            }
            Ok(EngineInbound::StartDpWave { wave, .. }) => {
                // The mock is a single independent engine, never a lockstep
                // group, so it never pauses and has no wave to start.
                tracing::debug!("zmq engine {engine_index} ignoring start of wave {wave}");
            }
            Ok(EngineInbound::Utility(call)) => {
                // A prefix-cache reset is acknowledged and leaves the simulated
                // engine's cache as it is; no other EngineCore method exists here.
                let outcome = if call.method == "reset_prefix_cache" {
                    Ok(OpaqueValue::from(true))
                } else {
                    Err(format!(
                        "Call to {} method failed: the mock engine has no such method",
                        call.method
                    ))
                };
                let _ = out_tx.send(EngineCoreOutputs::Utility(UtilityCallOutput {
                    engine_index,
                    timestamp: 0.0,
                    output: UtilityOutput::from_outcome(call.call_id, outcome),
                }));
            }
            Err(error) => {
                tracing::info!("zmq engine {engine_index} input closed: {error}");
                break false;
            }
        }
    };
    if drained {
        // The last forwarder's terminal batch is only in the funnel so far.
        // Closing the funnel queues the remaining batches and starts the
        // writer's transport linger. Keep the input alive until that finishes:
        // dropping it early makes the frontend fail even queued final outputs.
        drop(out_tx);
        drop(abort_tx);
        if let Err(e) = writer.await {
            tracing::error!("zmq engine {engine_index} output writer panicked: {e}");
        }
        return;
    }
    // The input closed with forwarders still running: they finish on their
    // own, as before (their outputs go nowhere once the writer fails).
    forwarders.detach_all();
}

/// Forward one request's generation events to the output funnel as
/// `EngineCoreOutputs`, attaching the engine's load snapshot as `scheduler_stats`.
async fn forward_request(
    request_id: String,
    engine_index: u32,
    mut gen_rx: mpsc::UnboundedReceiver<GenEvent>,
    out_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    engine: Option<Engine>,
    // Held until the forwarder is done or aborted: the request's share of
    // the in-flight count.
    _in_flight: InFlight,
) {
    while let Some(event) = gen_rx.recv().await {
        let load = engine
            .as_ref()
            .map(|engine| to_scheduler_stats(&engine.load()));
        match event {
            GenEvent::Token { token_id, .. } => {
                let batch =
                    token_batch(engine_index, request_id.clone(), vec![token_id], None, load);
                if out_tx.send(batch).is_err() {
                    return;
                }
            }
            GenEvent::Done { finish_reason, .. } => {
                let batch = token_batch(
                    engine_index,
                    request_id,
                    Vec::new(),
                    Some(map_finish(finish_reason)),
                    load,
                );
                let _ = out_tx.send(batch);
                return;
            }
        }
    }
    // Channel closed without Done (the request was aborted); nothing to send.
}

/// Push one scripted answer as `EngineCoreOutputs`: a batch per frame with
/// the answer's pause before every frame but the first, the prompt's prefill
/// stats on the first batch, the finish reason and the matched stop on the
/// last (a batch of its own when the answer has no tokens); the logprobs
/// ride along when the request asked for them, in the engine's columns
/// ([`wire_columns`]).
#[expect(
    clippy::too_many_arguments,
    reason = "one request's wire facts, handed to its forwarder task"
)]
async fn scripted_request(
    request_id: String,
    engine_index: u32,
    prompt_len: usize,
    answer: Answer,
    top_logprobs: Option<usize>,
    out_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    engine: Option<Engine>,
    // Held until the last batch is in the funnel: the request's share of
    // the in-flight count.
    _in_flight: InFlight,
) {
    let mut frames = answer.token_frames();
    if frames.is_empty() {
        frames.push(script::Frame {
            ids: Vec::new(),
            pieces: Vec::new(),
            logprobs: Vec::new(),
            last: true,
        });
    }
    let prompt_tokens = answer.prompt_tokens(prompt_len);
    let prefill = PrefillStats {
        num_prompt_tokens: prompt_tokens,
        num_computed_tokens: prompt_tokens,
        num_cached_tokens: answer.cached_tokens,
        ..Default::default()
    };
    for (n, frame) in frames.into_iter().enumerate() {
        if n > 0 && !answer.delay.is_zero() {
            tokio::time::sleep(answer.delay).await;
        }
        let load = engine
            .as_ref()
            .map(|engine| to_scheduler_stats(&engine.load()));
        let new_logprobs = top_logprobs
            .filter(|_| !frame.ids.is_empty() && frame.logprobs.len() == frame.ids.len())
            .map(|n| Logprobs {
                positions: answer
                    .top_logprobs(&frame.ids, &frame.logprobs, n.max(1))
                    .into_iter()
                    .map(|position| PositionLogprobs {
                        entries: wire_columns(&position, n),
                    })
                    .collect(),
            });
        let (finish_reason, stop_reason) = if frame.last {
            (
                Some(map_finish(&answer.finish_reason)),
                answer.matched_stop.as_ref().map(|stop| match stop {
                    MatchedStop::Token(id) => StopReason::TokenId(*id),
                    MatchedStop::Text(text) => StopReason::Text(text.clone()),
                }),
            )
        } else {
            (None, None)
        };
        let batch = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.clone(),
                new_token_ids: frame.ids,
                new_logprobs,
                finish_reason,
                stop_reason,
                prefill_stats: (n == 0).then(|| prefill.clone()),
                ..Default::default()
            }],
            scheduler_stats: load,
            finished_requests: frame.last.then(|| BTreeSet::from([request_id.clone()])),
            ..Default::default()
        });
        if out_tx.send(batch).is_err() {
            return;
        }
    }
}

/// One position's logprob columns as the engine fills them under
/// `logprobs: n`: the sampled token's own column first, at its rank among
/// the candidates (1 on the mock, whose sampled token is always the most
/// likely), then the `n` ranked columns from rank 1 up, the sampled token in
/// its ranked column as well (the gateway's shaping skips the repeat). `n`
/// of 0 leaves the own column alone. `position` holds the candidates in rank
/// order, the sampled token first: at least that one, at most `max(n, 1)`.
fn wire_columns(position: &script::TopAlternatives, n: usize) -> Vec<TokenLogprob> {
    let candidates = position
        .ids
        .iter()
        .copied()
        .zip(position.values.iter().copied());
    let own = candidates
        .clone()
        .next()
        .map(|(token_id, logprob)| TokenLogprob {
            token_id,
            logprob,
            rank: 1,
        });
    let ranked = candidates
        .take(n)
        .enumerate()
        .map(|(index, (token_id, logprob))| TokenLogprob {
            token_id,
            logprob,
            rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
        });
    own.into_iter().chain(ranked).collect()
}

/// Canned generator for non-realistic mode: emit `output_tokens` tokens (the
/// configured prefix first, then synthetic ids) then a terminal Done.
async fn canned_generate(
    gen_tx: mpsc::UnboundedSender<GenEvent>,
    output_tokens: u32,
    gen_delay: Duration,
    output_prefix: Vec<u32>,
) {
    for i in 0..output_tokens {
        if !gen_delay.is_zero() {
            tokio::time::sleep(gen_delay).await;
        }
        let event = GenEvent::Token {
            token_id: output_prefix.get(i as usize).copied().unwrap_or(100 + i),
            prompt_tokens: 1,
            cached_tokens: 0,
        };
        if gen_tx.send(event).is_err() {
            return;
        }
    }
    let _ = gen_tx.send(GenEvent::Done {
        finish_reason: "stop",
        prompt_tokens: 1,
        completion_tokens: output_tokens,
        cached_tokens: 0,
    });
}

/// Build a single-request output batch.
fn token_batch(
    engine_index: u32,
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    scheduler_stats: Option<SchedulerStats>,
) -> EngineCoreOutputs {
    let finished_requests = finish_reason.map(|_| BTreeSet::from([request_id.clone()]));
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index,
        outputs: vec![EngineCoreOutput {
            request_id,
            new_token_ids,
            finish_reason,
            ..Default::default()
        }],
        scheduler_stats,
        finished_requests,
        ..Default::default()
    })
}

/// One message on the output socket, counted; `false` once the socket has
/// failed and the writer is done.
async fn push(
    output: &mut MockEngineOutput,
    counts: &OutputCounts,
    message: &EngineCoreOutputs,
) -> bool {
    if let Err(error) = output.send_outputs(message).await {
        tracing::warn!("zmq engine output send failed: {error}");
        return false;
    }
    counts.pushed(message);
    true
}

/// An output waiting for its step, and whether the batch it came in named
/// its request finished (a finish without an output of its own, an abort's
/// acknowledgement, waits as an output-less entry).
struct Pending {
    output: Option<EngineCoreOutput>,
    finished: bool,
}

/// The step queue of a rank with a step period: the forwarders' batches are
/// taken apart into per-request queues, and every step takes the oldest
/// waiting output of every request that has one, in the order the requests
/// first appeared, as an engine's step carries one output per running
/// request. A request that produced faster than the period waits its turn,
/// one output per step; its finish rides with the output its batch named
/// (an answer the engine ends with `Abort` keeps its tokens like any other);
/// the frontend's abort ([`StepQueue::abort`]) drops the outputs still
/// waiting for its request, which stand for steps that have not happened,
/// so the terminal that follows goes out on the next step alone; the latest
/// load, KV-transfer samples and timestamp go on the next step.
#[derive(Default)]
struct StepQueue {
    waiting: HashMap<String, VecDeque<Pending>>,
    order: Vec<String>,
    /// The next step's header: the engine index, the latest load, samples
    /// and timestamp, outputs and finished set empty.
    header: RequestBatchOutputs,
}

impl StepQueue {
    /// Queue a forwarder's batch.
    fn push(&mut self, batch: RequestBatchOutputs) {
        let finished = batch.finished_requests.unwrap_or_default();
        let mut named = BTreeSet::new();
        for output in batch.outputs {
            let done = finished.contains(&output.request_id);
            named.insert(output.request_id.clone());
            self.queue_of(output.request_id.clone()).push_back(Pending {
                output: Some(output),
                finished: done,
            });
        }
        for request_id in finished.into_iter().filter(|id| !named.contains(id)) {
            self.queue_of(request_id).push_back(Pending {
                output: None,
                finished: true,
            });
        }
        self.header.engine_index = batch.engine_index;
        if batch.scheduler_stats.is_some() {
            self.header.scheduler_stats = batch.scheduler_stats;
        }
        if batch.kv_transfer.is_some() {
            self.header.kv_transfer = batch.kv_transfer;
        }
        self.header.timestamp = batch.timestamp;
    }

    /// Whether nothing waits for a step.
    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The frontend aborted `request_id`: what it still had waiting stood
    /// for steps not taken and is dropped.
    fn abort(&mut self, request_id: &str) {
        if let Some(queue) = self.waiting.get_mut(request_id) {
            queue.clear();
        }
    }

    fn queue_of(&mut self, request_id: String) -> &mut VecDeque<Pending> {
        if !self.waiting.contains_key(&request_id) {
            self.order.push(request_id.clone());
        }
        self.waiting.entry(request_id).or_default()
    }

    /// The next step, or `None` when nothing waits.
    fn next_step(&mut self) -> Option<EngineCoreOutputs> {
        if self.order.is_empty() {
            return None;
        }
        let mut outputs = Vec::with_capacity(self.order.len());
        let mut finished = BTreeSet::new();
        for request_id in &self.order {
            let Some(pending) = self
                .waiting
                .get_mut(request_id)
                .and_then(VecDeque::pop_front)
            else {
                continue;
            };
            if let Some(output) = pending.output {
                outputs.push(output);
            }
            if pending.finished {
                finished.insert(request_id.clone());
            }
        }
        let waiting = &mut self.waiting;
        self.order.retain(|request_id| {
            if waiting
                .get(request_id)
                .is_some_and(|queue| !queue.is_empty())
            {
                true
            } else {
                waiting.remove(request_id);
                false
            }
        });
        let header = std::mem::take(&mut self.header);
        self.header.engine_index = header.engine_index;
        self.header.timestamp = header.timestamp;
        Some(EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            outputs,
            finished_requests: (!finished.is_empty()).then_some(finished),
            ..header
        }))
    }
}

/// A terminal batch carrying only a finish reason (for aborts).
fn terminal_batch(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    token_batch(
        engine_index,
        request_id,
        Vec::new(),
        Some(finish_reason),
        None,
    )
}

/// Map the simulator's load snapshot onto the piggybacked scheduler stats SMG
/// uses as its DP routing signal.
fn to_scheduler_stats(snapshot: &LoadSnapshot) -> SchedulerStats {
    SchedulerStats {
        num_running_reqs: snapshot.num_running_reqs.max(0) as u64,
        num_waiting_reqs: snapshot.num_waiting_reqs.max(0) as u64,
        kv_cache_usage: snapshot.token_usage,
        ..Default::default()
    }
}

fn map_finish(reason: &str) -> EngineCoreFinishReason {
    match reason {
        "length" => EngineCoreFinishReason::Length,
        "abort" => EngineCoreFinishReason::Abort,
        "error" => EngineCoreFinishReason::Error,
        _ => EngineCoreFinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use engine_zmq_client::{
        connect_handshake, protocol::vllm::request::EngineCoreRequest, EngineCoreClient,
    };
    use futures::StreamExt;

    use super::*;
    use crate::engine::EngineParams;

    fn test_config() -> Config {
        Config {
            host: "127.0.0.1".to_string(),
            http_base_port: 0,
            http_count: 0,
            grpc_base_port: 0,
            grpc_count: 0,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: String::new(),
            gen_delay: Duration::from_millis(0),
            output_tokens: 4,
            realistic: true,
            engine: EngineParams::default(),
            ..Config::default()
        }
    }

    fn batch(
        request_id: &str,
        ids: Vec<u32>,
        finish: Option<EngineCoreFinishReason>,
        running: Option<u64>,
    ) -> RequestBatchOutputs {
        let stats = running.map(|running| SchedulerStats {
            num_running_reqs: running,
            ..Default::default()
        });
        token_batch(0, request_id.to_string(), ids, finish, stats)
            .as_request_batch()
            .cloned()
            .expect("a batch")
    }

    fn step_of(queue: &mut StepQueue) -> RequestBatchOutputs {
        queue
            .next_step()
            .as_ref()
            .and_then(EngineCoreOutputs::as_request_batch)
            .cloned()
            .expect("a step")
    }

    fn requests(step: &RequestBatchOutputs) -> Vec<&str> {
        step.outputs
            .iter()
            .map(|output| output.request_id.as_str())
            .collect()
    }

    /// A step carries the next output of every request that has one, in the
    /// order the requests first appeared; a request that produced faster
    /// than the period waits its turn, one output per step.
    #[test]
    fn a_step_carries_one_output_per_request_and_a_fast_request_waits_its_turn() {
        let mut queue = StepQueue::default();
        assert!(queue.next_step().is_none(), "nothing waits");
        queue.push(batch("a", vec![1], None, None));
        queue.push(batch("a", vec![2], None, None));
        queue.push(batch("b", vec![3], None, None));
        queue.push(batch("a", vec![4], None, None));
        queue.push(batch("c", vec![5], None, None));
        let first = step_of(&mut queue);
        assert_eq!(requests(&first), ["a", "b", "c"]);
        assert_eq!(first.outputs[0].new_token_ids, [1]);
        let second = step_of(&mut queue);
        assert_eq!(requests(&second), ["a"]);
        assert_eq!(second.outputs[0].new_token_ids, [2]);
        let third = step_of(&mut queue);
        assert_eq!(requests(&third), ["a"]);
        assert_eq!(third.outputs[0].new_token_ids, [4]);
        assert!(queue.next_step().is_none(), "every output went out");
        queue.push(batch("b", vec![6], None, None));
        assert_eq!(
            requests(&step_of(&mut queue)),
            ["b"],
            "a request may come back"
        );
    }

    /// A finish rides with the output its batch named, a finish without an
    /// output of its own ends its request in the next step, and the latest
    /// load goes on the next step and on that step only.
    #[test]
    fn a_finish_rides_with_its_output_and_the_load_with_the_next_step() {
        let mut queue = StepQueue::default();
        queue.push(batch("a", vec![1], None, Some(2)));
        queue.push(batch(
            "a",
            vec![2],
            Some(EngineCoreFinishReason::Length),
            None,
        ));
        queue.push(batch("b", vec![3], None, None));
        let first = step_of(&mut queue);
        assert_eq!(requests(&first), ["a", "b"]);
        assert_eq!(first.finished_requests, None);
        assert_eq!(
            first
                .scheduler_stats
                .as_ref()
                .map(|stats| stats.num_running_reqs),
            Some(2)
        );
        let second = step_of(&mut queue);
        assert_eq!(requests(&second), ["a"]);
        assert_eq!(
            second.finished_requests,
            Some(BTreeSet::from(["a".to_string()]))
        );
        assert_eq!(
            second.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(
            second.scheduler_stats, None,
            "the load went with the step before"
        );
        queue.push(RequestBatchOutputs {
            engine_index: 0,
            finished_requests: Some(BTreeSet::from(["b".to_string()])),
            ..Default::default()
        });
        let third = step_of(&mut queue);
        assert!(
            third.outputs.is_empty(),
            "an acknowledgement carries no output"
        );
        assert_eq!(
            third.finished_requests,
            Some(BTreeSet::from(["b".to_string()]))
        );
        assert!(queue.next_step().is_none());
    }

    /// The frontend's abort drops the outputs still waiting for its request,
    /// which stand for steps that have not happened, so the terminal that
    /// follows goes out on the next step alone; the other requests' outputs
    /// are untouched.
    #[test]
    fn the_frontends_abort_drops_the_outputs_still_waiting_for_its_request() {
        let mut queue = StepQueue::default();
        queue.push(batch("a", vec![1], None, None));
        queue.push(batch("a", vec![2], None, None));
        queue.push(batch("b", vec![3], None, None));
        queue.abort("a");
        queue.push(batch(
            "a",
            Vec::new(),
            Some(EngineCoreFinishReason::Abort),
            None,
        ));
        let first = step_of(&mut queue);
        assert_eq!(requests(&first), ["a", "b"]);
        assert_eq!(
            first.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );
        assert!(
            first.outputs[0].new_token_ids.is_empty(),
            "the waiting tokens are gone"
        );
        assert_eq!(
            first.finished_requests,
            Some(BTreeSet::from(["a".to_string()]))
        );
        assert!(
            queue.next_step().is_none(),
            "nothing of a's is left behind the abort"
        );
    }

    /// An answer the engine itself ends with `Abort` (a scripted finish
    /// reason) keeps every token it produced: the period changes the wire's
    /// shape and nothing else.
    #[test]
    fn an_abort_the_engine_answers_with_keeps_its_waiting_tokens() {
        let mut queue = StepQueue::default();
        queue.push(batch("a", vec![1], None, None));
        queue.push(batch(
            "a",
            vec![2],
            Some(EngineCoreFinishReason::Abort),
            None,
        ));
        let first = step_of(&mut queue);
        assert_eq!(first.outputs[0].new_token_ids, [1]);
        assert_eq!(first.finished_requests, None);
        let second = step_of(&mut queue);
        assert_eq!(second.outputs[0].new_token_ids, [2]);
        assert_eq!(
            second.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );
        assert_eq!(
            second.finished_requests,
            Some(BTreeSet::from(["a".to_string()]))
        );
        assert!(queue.next_step().is_none());
    }

    /// The output counts tell a step of many outputs from one message per
    /// token: a batch message counts once, its outputs by their number, a
    /// utility reply not at all.
    #[test]
    fn output_counts_read_batch_messages_and_their_outputs() {
        let counts = OutputCounts::default();
        let mut queue = StepQueue::default();
        queue.push(batch("a", vec![1], None, None));
        queue.push(batch("b", vec![2], None, None));
        queue.push(batch("c", vec![3], None, None));
        counts.pushed(&queue.next_step().expect("a step of three"));
        counts.pushed(&token_batch(0, "a".to_string(), vec![4], None, None));
        counts.pushed(&EngineCoreOutputs::Utility(UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput::from_outcome(7, Err("not answered".to_string())),
        }));
        assert_eq!(counts.read(), (2, 4));
    }

    /// End-to-end over ipc://: the engine-zmq-client connector (frontend) drives
    /// a full generate against this mock ZMQ engine.
    #[tokio::test]
    async fn end_to_end_generate_over_ipc() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (
            endpoint("hs.sock"),
            endpoint("in.sock"),
            endpoint("out.sock"),
        );

        let cfg = Arc::new(test_config());
        // Engine dials the frontend; it waits for the handshake endpoint to bind.
        #[expect(
            clippy::disallowed_methods,
            reason = "serve loops until the test drops its handles"
        )]
        let _engine = tokio::spawn(serve(cfg, handshake.clone(), 0));

        let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(10))
            .await
            .expect("frontend handshake");
        let client = EngineCoreClient::new(transport);
        assert_eq!(client.engines()[0].ready_response.data_parallel_rank, 0);

        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "t1".to_string(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                ..Default::default()
            })
            .await
            .expect("submit");

        let mut tokens = Vec::new();
        let mut finished = false;
        while let Some(item) = stream.next().await {
            let output = item.expect("output ok");
            tokens.extend(output.new_token_ids.clone());
            if output.finished() {
                finished = true;
                break;
            }
        }
        assert!(finished, "stream should reach a terminal output");
        assert_eq!(
            tokens.len(),
            4,
            "a request without max_tokens gets output_tokens tokens"
        );
    }

    /// Two mock ranks dial one socket set — the grouped-worker topology the
    /// gateway registers for `--zmq-engine-count`. Each rank must complete its
    /// own rank-pinned request off the shared output socket, tagging its
    /// outputs with its own engine index.
    #[tokio::test]
    async fn two_ranks_share_one_socket_set() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (
            endpoint("hs.sock"),
            endpoint("in.sock"),
            endpoint("out.sock"),
        );

        let cfg = Arc::new(test_config());
        for rank in 0..2 {
            #[expect(
                clippy::disallowed_methods,
                reason = "serve loops until the test drops its handles"
            )]
            let _engine = tokio::spawn(serve(cfg.clone(), handshake.clone(), rank));
        }

        let transport = connect_handshake(&handshake, 2, &input, &output, Duration::from_secs(10))
            .await
            .expect("frontend handshake");
        let client = EngineCoreClient::new(transport);
        assert_eq!(client.engines().len(), 2);

        for rank in 0..2u32 {
            let mut stream = client
                .submit(EngineCoreRequest {
                    request_id: format!("rank-{rank}"),
                    prompt_token_ids: Some(vec![1, 2, 3]),
                    data_parallel_rank: Some(rank),
                    ..Default::default()
                })
                .await
                .expect("submit");

            let mut tokens = Vec::new();
            let mut finished = false;
            while let Some(item) = stream.next().await {
                let output = item.expect("output ok");
                tokens.extend(output.new_token_ids.clone());
                if output.finished() {
                    finished = true;
                    break;
                }
            }
            assert!(finished, "rank {rank} should reach a terminal output");
            assert_eq!(
                tokens.len(),
                4,
                "rank {rank} answers with output_tokens tokens"
            );
        }
    }
}
