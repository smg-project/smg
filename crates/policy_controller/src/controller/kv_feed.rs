//! The stock controller's cases of the cache-event follow
//! ([`crate::worker_host::KvFeed`]).

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        future::Future,
        sync::Arc,
        time::{Duration, Instant},
    };

    use kv_index::compute_content_hash;
    use mock_worker::{config::Config as MockConfig, engine::EngineParams, grpc};
    use smg_grpc_client::{
        tokenspeed_proto::{GenerateRequest, SamplingParams, TokenizedInput},
        TokenSpeedSchedulerClient as SchedulerClient,
    };
    use smg_worker::kv_events::SubscriptionKind;
    use tokio::{
        net::TcpListener,
        sync::{broadcast::error::TryRecvError, oneshot, watch},
        task::JoinSet,
        time::{error::Elapsed, sleep, timeout},
    };
    use tokio_stream::StreamExt as _;

    use crate::{
        controller::{
            config::tests::valid,
            worker_objects::{ObjectsWake, WorkerObjects},
            Controller,
        },
        store::{
            Ask, GatewayId, MemoryStore, Named, Naming, Signals, ViewBuffers, WorkerFacts,
            WorkerStatus, WorkersStore,
        },
        view::{Role, WireModes, WorkerId as RowId},
        worker_host::{KvFeed, Members, Objects},
        AppliedSpec, PolicySpec,
    };

    const MODEL: &str = "m";
    const FIRST: &str = "grpc://127.0.0.1:1";
    const SECOND: &str = "grpc://127.0.0.1:2";
    const FLOOD: &str = "http://127.0.0.1:3";
    const LAST: &str = "grpc://127.0.0.1:4";
    const WAIT: Duration = Duration::from_secs(5);
    const POLL: Duration = Duration::from_millis(1);

    fn facts(url: &str, wire_modes: WireModes, events: bool) -> WorkerFacts {
        WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            models: vec![MODEL.to_owned()],
            wire_modes,
            dp_size: 1,
            signals: Signals {
                events,
                loads: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        }
    }

    struct Fixture {
        store: Arc<MemoryStore>,
        objects: Arc<Objects<Members>>,
        feed: KvFeed<Members>,
    }

    impl Fixture {
        fn new() -> Self {
            let store = Arc::new(MemoryStore::new());
            let objects = WorkerObjects::new(Arc::clone(&store), ObjectsWake::default()).holder();
            let feed = KvFeed::new(Arc::clone(&store), Arc::clone(&objects));
            Self {
                store,
                objects,
                feed,
            }
        }

        fn lay(&self, facts: WorkerFacts) -> RowId {
            let row = self.store.upsert(facts).id();
            self.objects.follow(&*self.store);
            row
        }

        fn remove(&self, row: RowId) -> bool {
            let Some(standing) = self.store.get(row) else {
                return false;
            };
            let removed = self.store.remove(row, standing.revision).is_some();
            self.objects.follow(&*self.store);
            removed
        }

        /// The feed gets its first turn before the case writes any more rows.
        async fn check(&self, case: impl Future<Output = ()>) -> Result<(), Elapsed> {
            let (stop, stopped) = watch::channel(false);
            timeout(WAIT, async {
                tokio::join!(biased; self.feed.run(stopped), async {
                    case.await;
                    stop.send_replace(true);
                });
            })
            .await
        }

        /// Wait within `check` for a new active slot, past any removal.
        async fn new_slot(&self, url: &str, before: Option<u64>) -> u64 {
            loop {
                if let (Some(id), None) = self.feed.collector().slot_ids(url).await {
                    if Some(id) != before {
                        return id;
                    }
                }
                sleep(POLL).await;
            }
        }
    }

    #[tokio::test]
    async fn a_worker_laid_before_the_feed_started_is_told_at_its_start() {
        let f = Fixture::new();
        f.lay(facts(FIRST, WireModes::GRPC, true));
        f.check(async {
            f.new_slot(FIRST, None).await;
        })
        .await
        .expect("the worker laid before the feed takes a slot");
    }

    #[tokio::test]
    async fn a_worker_laid_later_is_told_by_its_event() {
        let f = Fixture::new();
        f.lay(facts(FIRST, WireModes::GRPC, true));
        let old = f.objects.registry().get_by_url(FIRST).expect("the worker");
        f.feed
            .collector()
            .on_worker_added(&old, SubscriptionKind::Events)
            .await;
        f.lay(facts(FIRST, WireModes::GRPC, true));
        f.check(async {
            // The first read has its workers and waits for the old task to end.
            assert!(f.feed.collector().slot_ids(FIRST).await.1.is_some());
            f.lay(facts(SECOND, WireModes::GRPC, true));
            f.new_slot(SECOND, None).await;
        })
        .await
        .expect("the worker laid during the first turn takes a slot");
    }

    #[tokio::test]
    async fn a_worker_removed_frees_its_slot() {
        let f = Fixture::new();
        let row = f.lay(facts(FIRST, WireModes::GRPC, true));
        f.check(async {
            f.new_slot(FIRST, None).await;
            assert!(f.remove(row));
            f.lay(facts(SECOND, WireModes::GRPC, true));
            f.new_slot(SECOND, None).await;
            assert_eq!(f.feed.collector().slot_ids(FIRST).await, (None, None));
            assert!(f.feed.collector().task_ended(FIRST).await);
        })
        .await
        .expect("the feed starts and processes the next worker");
    }

    #[tokio::test]
    async fn a_worker_replaced_by_one_that_does_not_stream_ends_the_former_stream() {
        let f = Fixture::new();
        let row = f.lay(facts(FIRST, WireModes::GRPC, true));
        let id = f.objects.object(row).expect("the worker's id");
        f.check(async {
            f.new_slot(FIRST, None).await;
            assert!(!f.feed.collector().task_ended(FIRST).await);
            assert_eq!(f.lay(facts(FIRST, WireModes::HTTP, true)), row);
            assert_eq!(f.objects.object(row).as_ref(), Some(&id));
            f.lay(facts(SECOND, WireModes::GRPC, true));
            f.new_slot(SECOND, None).await;
            assert_eq!(f.feed.collector().slot_ids(FIRST).await, (None, None));
            assert!(f.feed.collector().task_ended(FIRST).await);
        })
        .await
        .expect("the feed starts and processes the next worker");
    }

    #[tokio::test]
    async fn a_row_that_asks_for_no_events_gets_no_events_stream() {
        let f = Fixture::new();
        f.check(async {
            f.lay(facts(FIRST, WireModes::GRPC, false));
            f.lay(facts(SECOND, WireModes::GRPC, true));
            f.new_slot(SECOND, None).await;
            assert_eq!(f.feed.collector().slot_ids(FIRST).await, (None, None));
        })
        .await
        .expect("the feed processes the worker whose row asks for events");
    }

    #[tokio::test]
    async fn a_worker_replaced_in_place_is_told_again_for_its_new_object() {
        let f = Fixture::new();
        let row = f.lay(facts(FIRST, WireModes::GRPC, true));
        let id = f.objects.object(row).expect("the worker's id");
        let old = f.objects.registry().get(&id).expect("the old worker");
        f.check(async {
            let first = f.new_slot(FIRST, None).await;
            assert_eq!(f.lay(facts(FIRST, WireModes::GRPC, true)), row);
            assert_eq!(f.objects.object(row).as_ref(), Some(&id));
            let second = f.new_slot(FIRST, Some(first)).await;
            // Hearing of the old object again leaves the new object's stream.
            f.feed.tell(&id, &old).await;
            assert_eq!(
                f.feed.collector().slot_ids(FIRST).await,
                (Some(second), None)
            );
        })
        .await
        .expect("the replacement takes a new slot");
    }

    #[tokio::test]
    async fn a_receiver_that_lagged_reads_the_registry_whole() {
        let f = Fixture::new();
        f.check(async {
            let mut row = f.lay(facts(FIRST, WireModes::GRPC, true));
            f.new_slot(FIRST, None).await;
            let mut last = None;
            for (gone, joined) in [(FIRST, SECOND), (SECOND, FIRST)] {
                let mut events = f.objects.registry().subscribe_events();
                assert!(f.remove(row));
                row = f.lay(facts(joined, WireModes::GRPC, true));
                for _ in 0..=1_024 {
                    f.lay(facts(FLOOD, WireModes::HTTP, true));
                }
                assert!(matches!(events.try_recv(), Err(TryRecvError::Lagged(_))));
                f.lay(facts(LAST, WireModes::GRPC, true));
                let slot = f.new_slot(LAST, last).await;
                // A later replacement takes its slot after the whole read ends.
                f.lay(facts(LAST, WireModes::GRPC, true));
                last = Some(f.new_slot(LAST, Some(slot)).await);
                assert_eq!(f.feed.collector().slot_ids(gone).await, (None, None));
                assert!(f.feed.collector().slot_ids(joined).await.0.is_some());
                assert_eq!(f.feed.collector().slot_ids(FLOOD).await, (None, None));
            }

            // A queued removal can name a URL that a whole read saw again.
            let mut told = HashSet::from([FIRST.to_owned()]);
            f.feed.removed(FIRST, &mut told).await;
            assert!(told.contains(FIRST));
            assert!(f.feed.collector().slot_ids(FIRST).await.0.is_some());
        })
        .await
        .expect("the feed processes the last worker after the lost events");
    }

    #[tokio::test]
    async fn a_stop_ends_every_stream() {
        let f = Fixture::new();
        let workers = [FIRST, SECOND];
        for url in workers {
            f.lay(facts(url, WireModes::GRPC, true));
        }
        f.check(async {
            for url in workers {
                f.new_slot(url, None).await;
                assert!(!f.feed.collector().task_ended(url).await);
            }
        })
        .await
        .expect("the feed ends at the stop");
        for url in workers {
            assert_eq!(f.feed.collector().slot_ids(url).await, (None, None));
            assert!(f.feed.collector().task_ended(url).await);
        }
    }

    #[tokio::test]
    async fn a_block_a_mock_engine_stores_is_named_for_a_request() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("the mock worker's listener binds");
        let addr = listener.local_addr().expect("the mock worker's address");
        let url = format!("grpc://{addr}");
        let tokens = [1, 2, 3, 4];
        let block_size = tokens.len() as u32;
        let mock = MockConfig {
            model_id: MODEL.to_owned(),
            realistic: true,
            engine: EngineParams {
                block_size,
                prefix_cache: true,
                ..EngineParams::default()
            },
            ..MockConfig::default()
        };
        let row = WorkerFacts {
            engine_kind: mock.grpc_wire.as_str().to_owned(),
            kv_block_size: Some(block_size),
            ..facts(&url, WireModes::GRPC, true)
        };
        let (stop_mock, mock_stopped) = oneshot::channel();
        let mut tasks = JoinSet::new();
        tasks.spawn(grpc::serve_with_listener_until(
            Arc::new(mock),
            listener,
            async {
                let _ = mock_stopped.await;
            },
        ));

        let controller = Controller::bind(valid())
            .await
            .expect("the controller binds");
        let store = controller.store();
        let objects = controller.workers.holder();
        let mut spec = PolicySpec::preset("cache_aware").expect("the cache-aware preset");
        spec.generation = 1;
        let applied = AppliedSpec::validate(&spec).expect("the preset validates");
        store.apply(applied).expect("the first generation");
        let (stop, stopped) = oneshot::channel();
        let serve = Box::pin(controller.serve(async {
            let _ = stopped.await;
        }));
        let case = async {
            let id = store.upsert(row).id();
            let revision = store.get(id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("the worker is ready");
            store.publish_now();

            timeout(WAIT, async {
                loop {
                    if objects
                        .registry()
                        .get_by_url(&url)
                        .and_then(|w| w.engine_connection_if_initialized())
                        .is_some()
                    {
                        break;
                    }
                    sleep(POLL).await;
                }
            })
            .await
            .expect("the feed dials through the worker object's connector");

            let hashes = [compute_content_hash(&tokens).0];
            let holders = || {
                store.deciding(|world| {
                    let gateway = GatewayId(0);
                    let ask = Ask {
                        gateway,
                        now: Instant::now(),
                        pinned: None,
                        affinity: false,
                    };
                    let mut view = ViewBuffers::default();
                    let built = world
                        .build(MODEL, Role::Single, &ask, &mut view)
                        .expect("the worker's pool");
                    let naming = Naming {
                        gateway,
                        block_hashes: &hashes,
                        seed: 0,
                        cap: 0,
                        wires: None,
                        now_micros: built.now_micros,
                    };
                    let mut named = Named::default();
                    world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
                    named
                        .holders
                        .iter()
                        .map(|h| (view.candidates[h.row as usize].id, h.overlap_blocks))
                        .collect::<Vec<_>>()
                })
            };
            assert!(holders().is_empty());
            let client = SchedulerClient::connect(&url)
                .await
                .expect("the request connects");
            let mut generated = client
                .generate(GenerateRequest {
                    request_id: "cache-a-block".to_owned(),
                    tokenized: Some(TokenizedInput {
                        input_ids: tokens.to_vec(),
                        ..TokenizedInput::default()
                    }),
                    sampling_params: Some(SamplingParams {
                        max_new_tokens: Some(1),
                        ..SamplingParams::default()
                    }),
                    ..GenerateRequest::default()
                })
                .await
                .expect("the mock worker takes the request");
            while let Some(frame) = generated.next().await {
                frame.expect("the mock worker answers");
            }
            generated.mark_completed();
            loop {
                let named = holders();
                if !named.is_empty() {
                    assert_eq!(named, [(id, 1)]);
                    break;
                }
                sleep(POLL).await;
            }
            stop.send(()).expect("the controller is still serving");
        };
        let (result, ()) = timeout(Duration::from_secs(15), async { tokio::join!(serve, case) })
            .await
            .expect("the feed names the stored block and the controller stops");
        result.expect("the controller stops cleanly");
        stop_mock
            .send(())
            .expect("the mock worker is still serving");
        timeout(WAIT, tasks.join_next())
            .await
            .expect("the mock worker stops")
            .expect("the mock worker's task")
            .expect("the mock worker stops cleanly");
    }
}
