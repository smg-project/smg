//! Mock ZMQ EngineCore worker: plays the vLLM `EngineCoreProc` role over ZMQ so
//! SMG's `engine-zmq-client` connector can be exercised without a GPU. Unlike
//! the HTTP/gRPC workers (which bind and are dialed by the gateway), a ZMQ mock
//! engine *dials* the frontend's `ipc://` handshake address, then registers,
//! receives `EngineCoreRequest`s, and pushes `EngineCoreOutputs` — driven by the
//! same continuous-batching [`Engine`] simulator the other protocols use.

use std::{
    collections::{BTreeSet, HashMap},
    future::{pending, Future},
    pin::pin,
    sync::Arc,
    time::Duration,
};

use engine_zmq_client::{
    codec::OpaqueValue,
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
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
};

use crate::{
    config::Config,
    drain::InFlight,
    engine::{Engine, EventReceiver, GenEvent, LoadSnapshot, NewRequest},
    script::{self, Answer, MatchedStop},
};

// zeromq queues sends and exposes no flush/linger operation. Keep both sockets
// alive briefly after the last queued output so its background writer and the
// frontend can consume it before input EOF fails the frontend's requests.
// This is best-effort transport linger, not a delivery acknowledgement; the
// process-wide drain deadline still bounds shutdown if the frontend is stuck.
const OUTPUT_LINGER: Duration = Duration::from_secs(1);

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
    #[expect(
        clippy::disallowed_methods,
        reason = "the writer ends with the output channel; a drain awaits it"
    )]
    let writer = tokio::spawn(async move {
        let mut output = output;
        while let Some(outputs) = out_rx.recv().await {
            if let Err(error) = output.send_outputs(&outputs).await {
                tracing::warn!("zmq engine output send failed: {error}");
                return;
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
                    let want_logprobs = request
                        .sampling_params
                        .as_ref()
                        .is_some_and(|params| params.logprobs.is_some());
                    let forwarder = forwarders.spawn(scripted_request(
                        request_id.clone(),
                        engine_index,
                        prompt.len(),
                        answer,
                        want_logprobs,
                        out_tx.clone(),
                        engine.clone(),
                        in_flight,
                    ));
                    inflight.insert(request_id, forwarder);
                    continue;
                }
                let gen_rx = match &engine {
                    Some(engine) => {
                        let (gen_tx, gen_rx) = engine.output_channel();
                        engine.submit(NewRequest {
                            request_id: request_id.clone(),
                            prompt_token_ids: prompt,
                            max_new,
                            events: gen_tx,
                        });
                        GenerationEvents::Realistic(gen_rx)
                    }
                    None => {
                        let (gen_tx, gen_rx) = mpsc::unbounded_channel::<GenEvent>();
                        let delay = cfg.gen_delay;
                        #[expect(
                            clippy::disallowed_methods,
                            reason = "canned producer self-terminates after Done"
                        )]
                        let _canned = tokio::spawn(canned_generate(gen_tx, max_new, delay));
                        GenerationEvents::Canned(gen_rx)
                    }
                };

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
                    // Push a terminal Abort output so the frontend stream ends.
                    let _ = out_tx.send(terminal_batch(
                        engine_index,
                        request_id,
                        EngineCoreFinishReason::Abort,
                    ));
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
        if let Err(e) = writer.await {
            tracing::error!("zmq engine {engine_index} output writer panicked: {e}");
        }
        return;
    }
    // The input closed with forwarders still running: they finish on their
    // own, as before (their outputs go nowhere once the writer fails).
    forwarders.detach_all();
}

// Keep canned-mode buffering unchanged; only the shared realistic engine
// uses the bounded event contract.
enum GenerationEvents {
    Realistic(EventReceiver),
    Canned(mpsc::UnboundedReceiver<GenEvent>),
}

impl GenerationEvents {
    async fn recv(&mut self) -> Option<GenEvent> {
        match self {
            Self::Realistic(rx) => rx.recv().await,
            Self::Canned(rx) => rx.recv().await,
        }
    }
}

/// Forward one request's generation events to the output funnel as
/// `EngineCoreOutputs`, attaching the engine's load snapshot as `scheduler_stats`.
async fn forward_request(
    request_id: String,
    engine_index: u32,
    mut gen_rx: GenerationEvents,
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
            GenEvent::OutputOverflow => {
                let _ = out_tx.send(terminal_batch(
                    engine_index,
                    request_id,
                    EngineCoreFinishReason::Error,
                ));
                return;
            }
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
/// ride along when the request asked for them.
#[expect(
    clippy::too_many_arguments,
    reason = "one request's wire facts, handed to its forwarder task"
)]
async fn scripted_request(
    request_id: String,
    engine_index: u32,
    prompt_len: usize,
    answer: Answer,
    want_logprobs: bool,
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
        let new_logprobs =
            (want_logprobs && !frame.ids.is_empty() && frame.logprobs.len() == frame.ids.len())
                .then(|| Logprobs {
                    positions: frame
                        .ids
                        .iter()
                        .zip(&frame.logprobs)
                        .map(|(&token_id, &logprob)| PositionLogprobs {
                            entries: vec![TokenLogprob {
                                token_id,
                                logprob,
                                rank: 1,
                            }],
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

/// Canned generator for non-realistic mode: emit `output_tokens` synthetic
/// tokens then a terminal Done.
async fn canned_generate(
    gen_tx: mpsc::UnboundedSender<GenEvent>,
    output_tokens: u32,
    gen_delay: Duration,
) {
    for i in 0..output_tokens {
        if !gen_delay.is_zero() {
            tokio::time::sleep(gen_delay).await;
        }
        let event = GenEvent::Token {
            token_id: 100 + i,
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

    #[tokio::test]
    async fn output_overflow_becomes_a_terminal_engine_error() {
        let (tx, rx) = crate::engine::output_channel(2);
        let token = GenEvent::Token {
            token_id: 7,
            prompt_tokens: 4,
            cached_tokens: 0,
        };
        tx.send(token.clone()).unwrap();
        tx.send(token.clone()).unwrap();
        assert!(tx.send(token).is_err());
        let (out_tx, mut out_rx) = mpsc::unbounded_channel();
        forward_request(
            "slow".into(),
            0,
            GenerationEvents::Realistic(rx),
            out_tx,
            None,
            InFlight::enter(),
        )
        .await;
        let EngineCoreOutputs::RequestBatch(batch) = out_rx.recv().await.unwrap() else {
            panic!("expected request batch")
        };
        assert_eq!(batch.outputs.len(), 1);
        assert_eq!(
            batch.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Error)
        );
        assert!(batch.outputs[0].new_token_ids.is_empty());
        assert_eq!(
            batch.finished_requests,
            Some(BTreeSet::from(["slow".into()]))
        );
        assert!(out_rx.recv().await.is_none());
    }

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
