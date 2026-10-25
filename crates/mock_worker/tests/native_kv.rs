//! Native discovery must name a live publisher speaking the advertised wire.
use std::{sync::Arc, time::Duration};

use mock_worker::{config::Config, grpc::serve_with_listener_until, kv_zmq::Wire};
use smg_grpc_client::tokenspeed_proto as ts;
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle, time::timeout};
use tonic::{transport::Channel, Code};
use vllm_proto::{self as v, control_client::ControlClient};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

struct Worker {
    native: ControlClient<Channel>,
    scheduler: ts::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl Worker {
    async fn start(mut cfg: Config) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        cfg.host = "127.0.0.1".into();
        cfg.grpc_base_port = address.port();
        cfg.grpc_native_vllm = true;
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "joined on shutdown; aborted on assertion failure"
        )]
        let task = tokio::spawn(serve_with_listener_until(Arc::new(cfg), listener, async {
            let _ = stopped.await;
        }));
        let channel = Channel::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        Self {
            native: ControlClient::new(channel.clone()),
            scheduler: ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel),
            stop: Some(stop),
            task,
        }
    }

    async fn finish(mut self) {
        self.stop.take().unwrap().send(()).unwrap();
        (&mut self.task).await.unwrap();
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn enabled() -> Config {
    Config {
        realistic: true,
        kv_events_zmq_base_port: Some(0),
        kv_events_topic: "native-kv".into(),
        kv_events_buffer_steps: 128,
        ..Config::default()
    }
}

#[tokio::test]
async fn discovered_endpoints_deliver_live_events_and_optional_replay() {
    timeout(Duration::from_secs(20), async {
        for replay in [false, true] {
            let mut cfg = enabled();
            cfg.kv_events_replay = replay;
            let mut worker = Worker::start(cfg).await;
            // The first discovery call can only succeed after both binds and subscription.
            let sources = worker
                .native
                .get_kv_event_sources(v::GetKvEventSourcesRequest {})
                .await
                .unwrap()
                .into_inner()
                .sources;
            assert_eq!(sources.len(), 1);
            let source = &sources[0];
            assert_eq!(source.transport, "zmq");
            assert_eq!(source.encoding, "msgpack");
            assert_eq!(source.schema_version, 1);
            assert_eq!(source.data_parallel_rank, Some(0));
            assert_eq!(source.topic, "native-kv");
            assert_eq!(source.buffer_steps, 128);
            assert_eq!((source.hwm, source.max_queue_size), (100_000, 100_000));
            assert!(!source.endpoint.ends_with(":0"));
            assert_eq!(!source.replay_endpoint.is_empty(), replay);

            let mut subscriber = SubSocket::new();
            subscriber.subscribe(&source.topic).await.unwrap();
            subscriber.connect(&source.endpoint).await.unwrap();
            let mut received = None;
            // ZMQ subscription propagation is asynchronous. Produce bounded,
            // distinct requests until a real cache event arrives; drain each response.
            for i in 0..32 {
                let mut response = worker
                    .scheduler
                    .generate(ts::GenerateRequest {
                        request_id: format!("native-kv-{i}"),
                        stream: true,
                        tokenized: Some(ts::TokenizedInput {
                            input_ids: vec![100 + i; 32],
                            ..Default::default()
                        }),
                        sampling_params: Some(ts::SamplingParams {
                            max_new_tokens: Some(1),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .await
                    .unwrap()
                    .into_inner();
                let mut complete = false;
                while let Some(frame) = response.message().await.unwrap() {
                    if let Some(ts::generate_response::Response::Complete(done)) = frame.response {
                        assert_ne!(done.finish_reason, "abort");
                        assert_eq!(done.output_ids.len(), 1);
                        complete = true;
                    }
                }
                assert!(complete);
                if let Ok(message) = timeout(Duration::from_millis(50), subscriber.recv()).await {
                    received = Some(message.unwrap());
                    break;
                }
            }
            let live = received.expect("a live KV frame from an actual generation");
            assert_eq!(live.len(), 3);
            assert_eq!(live.get(0).unwrap().as_ref(), source.topic.as_bytes());
            let seq = u64::from_be_bytes(live.get(1).unwrap().as_ref().try_into().unwrap());
            let payload = live.get(2).unwrap();
            let decoded = rmpv::decode::read_value(&mut payload.as_ref()).unwrap();
            let batch = decoded.as_array().unwrap();
            assert_eq!(batch.len(), 3);
            assert_eq!(batch[2].as_u64(), Some(0));
            let events = batch[1].as_array().unwrap();
            assert!(!events.is_empty());
            assert!(events.iter().any(|event| event
                .as_map()
                .unwrap()
                .iter()
                .any(|(key, value)| key.as_str() == Some("type")
                    && value.as_str() == Some("BlockStored"))));

            if replay {
                assert_ne!(source.endpoint, source.replay_endpoint);
                assert!(!source.replay_endpoint.ends_with(":0"));
                let mut dealer = DealerSocket::new();
                dealer.connect(&source.replay_endpoint).await.unwrap();
                let mut request = ZmqMessage::from(Vec::new());
                request.push_back(seq.to_be_bytes().to_vec().into());
                dealer.send(request).await.unwrap();
                let mut replayed_live = false;
                loop {
                    let frame = dealer.recv().await.unwrap();
                    assert_eq!(frame.len(), 4);
                    if frame.get(2).unwrap().as_ref() == [0xff; 8] {
                        break;
                    }
                    assert_eq!(frame.get(1).unwrap().as_ref(), source.topic.as_bytes());
                    let replay_seq =
                        u64::from_be_bytes(frame.get(2).unwrap().as_ref().try_into().unwrap());
                    if replay_seq == seq {
                        assert_eq!(frame.get(3).unwrap(), payload);
                        replayed_live = true;
                    }
                }
                assert!(
                    replayed_live,
                    "discovered replay returns the exact live payload"
                );
            }
            worker.finish().await;
        }
    })
    .await
    .expect("native discovery/live/replay/shutdown finish within the bound");
}

#[tokio::test]
async fn disabled_unsupported_and_failed_publishers_are_distinct() {
    timeout(Duration::from_secs(15), async {
        let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
        for case in [
            "disabled",
            "canned",
            "cache-off",
            "wrong-wire",
            "bind-failed",
        ] {
            let mut cfg = enabled();
            match case {
                "disabled" => cfg.kv_events_zmq_base_port = None,
                "canned" => cfg.realistic = false,
                "cache-off" => cfg.engine.prefix_cache = false,
                "wrong-wire" => cfg.kv_events_wire = Wire::Sglang,
                "bind-failed" => {
                    cfg.kv_events_zmq_base_port = Some(occupied.local_addr().unwrap().port())
                }
                _ => unreachable!(),
            }
            let mut worker = Worker::start(cfg).await;
            let result = worker
                .native
                .get_kv_event_sources(v::GetKvEventSourcesRequest {})
                .await;
            match case {
                "wrong-wire" => assert_eq!(result.unwrap_err().code(), Code::FailedPrecondition),
                "bind-failed" => assert_eq!(result.unwrap_err().code(), Code::Unavailable),
                _ => assert!(result.unwrap().into_inner().sources.is_empty()),
            }
            worker.finish().await;
        }
    })
    .await
    .expect("every disabled/failure case drains within the bound");
}
