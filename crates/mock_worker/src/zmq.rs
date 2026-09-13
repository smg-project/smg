//! Mock ZMQ EngineCore worker: plays the vLLM `EngineCoreProc` role over ZMQ so
//! SMG's `engine-zmq-client` connector can be exercised without a GPU. Unlike
//! the HTTP/gRPC workers (which bind and are dialed by the gateway), a ZMQ mock
//! engine *dials* the frontend's `ipc://` handshake address, then registers,
//! receives `EngineCoreRequest`s, and pushes `EngineCoreOutputs` — driven by the
//! same continuous-batching [`Engine`] simulator the other protocols use.

use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
    time::Duration,
};

use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
    protocol::vllm::{
        output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
        },
        stats::SchedulerStats,
    },
    EngineId,
};
use tokio::{sync::mpsc, task::AbortHandle};

use crate::{
    config::Config,
    engine::{Engine, GenEvent, LoadSnapshot, NewRequest},
};

/// Run one mock ZMQ EngineCore rank: connect to `handshake_address`, then serve
/// requests until the frontend disconnects.
pub async fn serve(cfg: Arc<Config>, handshake_address: String, engine_index: u32) {
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
    let engine = cfg.realistic.then(|| Engine::spawn(cfg.engine.clone()));

    // A single writer owns the output PUSH socket; per-request forwarders funnel
    // their outputs here so concurrent requests serialize onto the one socket.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<EngineCoreOutputs>();
    #[expect(
        clippy::disallowed_methods,
        reason = "writer self-terminates when the output channel closes"
    )]
    let _writer = tokio::spawn(async move {
        let mut output = output;
        while let Some(outputs) = out_rx.recv().await {
            if let Err(error) = output.send_outputs(&outputs).await {
                tracing::warn!("zmq engine output send failed: {error}");
                break;
            }
        }
    });

    let mut inflight: HashMap<String, AbortHandle> = HashMap::new();
    loop {
        match input.recv().await {
            Ok(EngineInbound::Add(request)) => {
                let request_id = request.request_id.clone();
                let prompt = request.prompt_token_ids.clone().unwrap_or_default();
                // Honor the request's max_tokens; fall back to the worker default.
                let max_new = request
                    .sampling_params
                    .as_ref()
                    .map(|params| params.max_tokens)
                    .filter(|&max| max > 0)
                    .unwrap_or(cfg.output_tokens);

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
                        #[expect(
                            clippy::disallowed_methods,
                            reason = "canned producer self-terminates after Done"
                        )]
                        let _canned = tokio::spawn(canned_generate(gen_tx, max_new, delay));
                    }
                }

                #[expect(
                    clippy::disallowed_methods,
                    reason = "forwarder self-terminates on Done or abort"
                )]
                let forwarder = tokio::spawn(forward_request(
                    request_id.clone(),
                    engine_index,
                    gen_rx,
                    out_tx.clone(),
                    engine.clone(),
                ));
                inflight.insert(request_id, forwarder.abort_handle());
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
            Ok(EngineInbound::Other(byte)) => {
                tracing::debug!("zmq engine {engine_index} ignoring request type {byte}");
            }
            Err(error) => {
                tracing::info!("zmq engine {engine_index} input closed: {error}");
                break;
            }
        }
    }
}

/// Forward one request's generation events to the output funnel as
/// `EngineCoreOutputs`, attaching the engine's load snapshot as `scheduler_stats`.
async fn forward_request(
    request_id: String,
    engine_index: u32,
    mut gen_rx: mpsc::UnboundedReceiver<GenEvent>,
    out_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    engine: Option<Engine>,
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
        scheduler_stats: scheduler_stats.map(Box::new),
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
            tokenizer_path: "mock-model".to_string(),
            gen_delay: Duration::ZERO,
            output_tokens: 4,
            realistic: false,
            engine: EngineParams::default(),
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
        assert_eq!(tokens.len(), 4, "canned mode emits output_tokens tokens");
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
            assert_eq!(tokens.len(), 4, "rank {rank} emits output_tokens tokens");
        }
    }
}
