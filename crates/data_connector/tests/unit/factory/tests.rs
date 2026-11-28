use std::{net::TcpListener, time::Instant};

use serde_json::json;

use super::*;
use crate::{
    core::{
        NewConversation, NewConversationItem, ResponseId, ResponseResult, StorageFailure,
        StoredResponse,
    },
    test_support::RefusedAddress,
};

/// A listener that completes the TCP handshake and never answers: the
/// store that accepts bytes and says nothing.
fn black_hole() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
    let port = listener.local_addr().expect("local address").port();
    (listener, port)
}

/// The store-call bound of the test configurations, the start-up
/// contact's bound: short, so the black-hole tests are quick, and under
/// the client's and pool's backstops (twice it) by construction.
const BOUND_SECS: u64 = 2;
const BOUND: Duration = Duration::from_secs(BOUND_SECS);

fn postgres_config(port: u16) -> PostgresConfig {
    PostgresConfig {
        db_url: format!("postgres://u:s3cr3t@127.0.0.1:{port}/db"),
        pool_max: 1,
        op_timeout_secs: BOUND_SECS,
        schema: None,
    }
}

#[test]
fn redacted_masks_the_password_in_the_userinfo_and_in_the_query() {
    assert_eq!(
        redacted_url("postgres://u:s3cr3t@h:5432/db"),
        "postgres://u:****@h:5432/db"
    );
    assert_eq!(
        redacted_url("postgres://u@h/db?password=s3cr3t&sslmode=require"),
        "postgres://u@h/db?password=****&sslmode=require"
    );
    assert_eq!(
        redacted_url("postgres://u:s3cr3t@h/db?Password=s3cr3t"),
        "postgres://u:****@h/db?Password=****"
    );
    assert_eq!(
        redacted_url("redis://:s3cr3t@h:6379/0"),
        "redis://:****@h:6379/0"
    );
    assert_eq!(redacted_url("not a url"), "<redacted>");
}

fn redis_config(port: u16) -> RedisConfig {
    RedisConfig {
        url: format!("redis://:s3cr3t@127.0.0.1:{port}/0"),
        pool_max: 1,
        op_timeout_secs: BOUND_SECS,
        schema: None,
        retention_days: None,
    }
}

/// An Oracle configuration with the given call bound; nothing answers at
/// its address, and no test below reaches the driver.
fn oracle_config(op_timeout_secs: u64) -> OracleConfig {
    OracleConfig {
        wallet_path: None,
        connect_descriptor: "tcps://127.0.0.1:1/s".to_string(),
        external_auth: false,
        username: "u".to_string(),
        password: "s3cr3t".to_string(),
        pool_min: OracleConfig::default_pool_min(),
        pool_max: 1,
        pool_timeout_secs: OracleConfig::default_pool_timeout_secs(),
        op_timeout_secs,
        schema: None,
    }
}

/// The tests' hang guard: a start-up verdict, the contact's own at
/// `BOUND` or a backstop's at twice it, must come within it; the guard
/// only turns a start-up that never answers into a failure. A loaded
/// runner changes the tests' duration, never their verdict.
const HANG_GUARD: Duration = Duration::from_secs(30);

/// `create_storage` for one remote backend, given up on at the hang
/// guard: `None` is a start-up that never gave a verdict.
async fn start_up(
    postgres: Option<&PostgresConfig>,
    redis: Option<&RedisConfig>,
) -> Option<Result<StorageBundle, String>> {
    let backend = if postgres.is_some() {
        HistoryBackend::Postgres
    } else {
        HistoryBackend::Redis
    };
    tokio::time::timeout(
        HANG_GUARD,
        create_storage(StorageFactoryConfig {
            backend: &backend,
            oracle: None,
            postgres,
            redis,
            memory: None,
            hook: None,
        }),
    )
    .await
    .ok()
}

#[tokio::test]
async fn a_postgres_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound() {
    let (_hole, port) = black_hole();
    let started = Instant::now();
    let verdict = start_up(Some(&postgres_config(port)), None)
        .await
        .expect("the start-up contact gives a verdict within its bound");
    let err = verdict
        .err()
        .expect("a store that never answers is refused at start-up");
    assert!(
        err.contains("Postgres history backend")
            && err.contains("did not answer within the start-up bound of 2 s"),
        "{err}"
    );
    assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
    assert!(
        started.elapsed() >= BOUND,
        "the verdict waited for the bound"
    );
}

#[tokio::test]
async fn a_redis_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound() {
    let (_hole, port) = black_hole();
    let started = Instant::now();
    let verdict = start_up(None, Some(&redis_config(port)))
        .await
        .expect("the start-up contact gives a verdict within its bound");
    let err = verdict
        .err()
        .expect("a store that never answers is refused at start-up");
    // The client's connect and the pool's create are bounded at twice the
    // store-call bound: the start-up contact's own verdict comes first,
    // the same as for Postgres. Only that verdict names the bound (a
    // backstop's reads "is unreachable at start-up: ..."), so the text
    // tells which came first; no clock on the test's side judges it.
    assert!(
        err.contains("Redis history backend")
            && err.contains("did not answer within the start-up bound of 2 s"),
        "{err}"
    );
    assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
    assert!(
        started.elapsed() >= BOUND,
        "the verdict waited for the bound"
    );
}

/// A start-up connect that never returns (a store that accepts and never
/// answers behind a driver that connects synchronously) is a verdict at
/// the bound; a connect that fails is a verdict at once; no bound is no
/// bound. The bound ends the wait, not the connect's thread, which runs
/// on, outside the runtime.
#[tokio::test]
async fn a_blocking_start_up_connect_that_never_returns_is_refused_at_the_bound() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let returned = Arc::new(AtomicBool::new(false));
    let flag = returned.clone();
    let started = Instant::now();
    let verdict = contact_blocking(Some(Duration::from_secs(1)), move || {
        std::thread::sleep(Duration::from_secs(3));
        flag.store(true, Ordering::SeqCst);
        Ok::<(), String>(())
    })
    .await;
    let elapsed = started.elapsed();
    assert!(
        matches!(verdict, Err(StoreContactError::NoAnswer)),
        "{verdict:?}"
    );
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(3),
        "verdict after {elapsed:?}; the bound is 1 s"
    );
    assert!(
        !returned.load(Ordering::SeqCst),
        "the connect's thread is still running when the verdict is given"
    );

    let failed = contact_blocking(Some(Duration::from_secs(1)), || {
        Err::<(), String>("refused".into())
    })
    .await;
    assert!(
        matches!(failed, Err(StoreContactError::Failed(ref cause)) if cause == "refused"),
        "{failed:?}"
    );
    let unbounded = contact_blocking(None, || Ok::<u8, String>(7)).await;
    assert!(matches!(unbounded, Ok(7)), "{unbounded:?}");
}

/// The connect the bound gave up on runs on a thread of its own, not in
/// the runtime's blocking pool: a runtime that shuts down after the
/// verdict does not wait for it, so the start-up error reaches the
/// operator and the process exits. A connect in the blocking pool holds
/// the runtime's drop for as long as the driver blocks, which is what this
/// test refuses: the runtime of its own, the verdict at the bound, then
/// the drop, all within a few seconds while the connect sleeps on.
#[test]
fn a_start_up_connect_the_bound_gave_up_on_does_not_hold_the_runtime_s_shutdown() {
    use std::sync::mpsc;

    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime of the test's own");
        let verdict = runtime.block_on(contact_blocking(Some(Duration::from_secs(1)), || {
            std::thread::sleep(Duration::from_secs(30));
            Ok::<(), String>(())
        }));
        assert!(
            matches!(verdict, Err(StoreContactError::NoAnswer)),
            "{verdict:?}"
        );
        drop(runtime);
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the runtime shut down while the connect's thread still ran");
}

/// The Oracle start-up against a store that accepts and never answers
/// gives a verdict within the pool wait: the bound's own where the driver
/// connects, a refusal at once where it cannot (its client library absent
/// on the test host); never a start that does not end.
#[tokio::test]
async fn an_oracle_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound() {
    let (_hole, port) = black_hole();
    let oracle = OracleConfig {
        wallet_path: None,
        connect_descriptor: format!("127.0.0.1:{port}/svc"),
        external_auth: false,
        username: "u".to_string(),
        password: "s3cr3t".to_string(),
        pool_min: 1,
        pool_max: 1,
        pool_timeout_secs: BOUND_SECS,
        op_timeout_secs: OracleConfig::default_op_timeout_secs(),
        schema: None,
    };
    let verdict = tokio::time::timeout(
        HANG_GUARD,
        create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Oracle,
            oracle: Some(&oracle),
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        }),
    )
    .await
    .expect("the start-up gives a verdict within its bound");
    let err = verdict
        .err()
        .expect("a store that never answers is refused at start-up");
    assert!(
        err.contains("Oracle history backend")
            && (err.contains("did not answer within the start-up bound of 2 s")
                || err.contains("is unreachable at start-up")),
        "{err}"
    );
    assert!(
        !err.contains("s3cr3t"),
        "the password is not in the text: {err}"
    );
}

#[tokio::test]
async fn an_unreachable_redis_store_is_refused_at_start_up_like_postgres() {
    let refused = RefusedAddress::reserve();
    let err = start_up(None, Some(&redis_config(refused.port())))
        .await
        .expect("the start-up contact gives a verdict within its bound")
        .err()
        .expect("an unreachable store is refused at start-up");
    assert!(
        err.contains("Redis history backend") && err.contains("is unreachable at start-up"),
        "{err}"
    );
    assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
}

#[tokio::test]
async fn an_unreachable_postgres_store_is_refused_at_start_up() {
    let refused = RefusedAddress::reserve();
    let err = start_up(Some(&postgres_config(refused.port())), None)
        .await
        .expect("the start-up contact gives a verdict within its bound")
        .err()
        .expect("an unreachable store is refused at start-up");
    assert!(
        err.contains("Postgres history backend") && err.contains("is unreachable at start-up"),
        "{err}"
    );
    assert!(
        err.contains("refused"),
        "the cause of the refusal is in the text the operator sees: {err}"
    );
    assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
}

/// The three stores of `bundle` take a write each and read it back.
async fn stores_and_reads_back(bundle: StorageBundle) {
    let (resp, conv, items) = (
        bundle.response_storage,
        bundle.conversation_storage,
        bundle.conversation_item_storage,
    );
    let mut response = StoredResponse::new(None);
    response.input = json!("hello");
    let id = resp.store_response(response).await.unwrap();
    assert!(resp.get_response(&id).await.unwrap().is_some());

    let conversation = conv
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
            tenant: None,
        })
        .await
        .unwrap();
    assert!(conv
        .get_conversation(&conversation.id)
        .await
        .unwrap()
        .is_some());

    let item = items
        .create_item(NewConversationItem {
            id: None,
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([]),
            status: Some("completed".to_string()),
            tenant: None,
        })
        .await
        .unwrap();
    assert!(items.get_item(&item.id).await.unwrap().is_some());
}

#[tokio::test]
async fn test_create_storage_memory() {
    let config = StorageFactoryConfig {
        backend: &HistoryBackend::Memory,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    };
    stores_and_reads_back(create_storage(config).await.unwrap()).await;
}

#[tokio::test]
async fn test_create_storage_none() {
    let config = StorageFactoryConfig {
        backend: &HistoryBackend::None,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    };
    let bundle = create_storage(config).await.unwrap();
    let (resp, conv) = (bundle.response_storage, bundle.conversation_storage);

    // NoOp storage should accept writes but return nothing on reads
    let mut response = StoredResponse::new(None);
    response.input = json!("hello");
    let id = resp.store_response(response).await.unwrap();
    assert!(resp.get_response(&id).await.unwrap().is_none());
    assert!(conv
        .get_conversation(&"nonexistent".into())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn test_create_storage_oracle_missing_config() {
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Oracle,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("should fail");
    assert!(err.contains("oracle configuration is required"));
}

/// A store config reaches the factory from every entry path (the CLI, the
/// Python bindings, a config built by hand); a bound of 0 s is refused by
/// name here, before any connection, instead of failing every store call
/// once the gateway is up.
#[tokio::test]
async fn a_zero_call_bound_is_refused_by_the_factory_before_any_connection() {
    let postgres_cfg = PostgresConfig {
        db_url: "postgres://user:pass@localhost:5432/mydb".to_string(),
        pool_max: 4,
        op_timeout_secs: 0,
        schema: None,
    };
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Postgres,
        oracle: None,
        postgres: Some(&postgres_cfg),
        redis: None,
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("a zero bound is refused");
    assert!(err.contains("op_timeout_secs"), "{err}");

    let redis_cfg = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        pool_max: 4,
        op_timeout_secs: 0,
        retention_days: Some(1),
        schema: None,
    };
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("a zero bound is refused");
    assert!(err.contains("op_timeout_secs"), "{err}");

    // The Oracle arm checks the bounds before the driver is loaded or a
    // connection is attempted: the refusal names the key.
    let oracle_cfg = oracle_config(0);
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Oracle,
        oracle: Some(&oracle_cfg),
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("a zero bound is refused");
    assert!(err.contains("op_timeout_secs"), "{err}");
}

/// The wrapper under the bound an Oracle configuration carries, with the
/// store's label: a call the store never answers fails at the configured
/// second as a timeout (which the API answers as `503 storage_unavailable`)
/// with the backend named. Building the Oracle bundle takes a live store,
/// so the factory's own wrapping of it is read, not run, here; the opt-in
/// live conformance suite goes through `create_storage`.
#[tokio::test]
async fn an_oracle_store_call_that_is_never_answered_ends_at_the_configured_bound() {
    use async_trait::async_trait;

    struct NeverAnswers;

    #[async_trait]
    impl ResponseStorage for NeverAnswers {
        async fn store_response(&self, _response: StoredResponse) -> ResponseResult<ResponseId> {
            std::future::pending().await
        }

        async fn get_response(
            &self,
            _response_id: &ResponseId,
        ) -> ResponseResult<Option<StoredResponse>> {
            std::future::pending().await
        }

        async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
            std::future::pending().await
        }

        async fn list_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
            _limit: Option<usize>,
        ) -> ResponseResult<Vec<StoredResponse>> {
            std::future::pending().await
        }

        async fn delete_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
        ) -> ResponseResult<usize> {
            std::future::pending().await
        }
    }

    let oracle_cfg = oracle_config(1);
    let bundle = bounded(
        StorageBundle {
            response_storage: Arc::new(NeverAnswers),
            conversation_storage: Arc::new(NoOpConversationStorage::new()),
            conversation_item_storage: Arc::new(NoOpConversationItemStorage::new()),
        },
        CallBound::new("oracle", oracle_cfg.op_timeout()),
    );
    let started = Instant::now();
    let err = bundle
        .response_storage
        .get_response(&ResponseId::from("resp_1"))
        .await
        .expect_err("a call the store never answers ends at the bound");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(5),
        "ended after {elapsed:?}; the bound is 1 s"
    );
    assert!(err.timed_out(), "{err}");
    assert_eq!(
        err.to_string(),
        "Storage timeout: the oracle get_response call exceeded its bound of 1 s"
    );
}

#[tokio::test]
async fn test_create_storage_postgres_missing_config() {
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Postgres,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("should fail");
    assert!(err.contains("Postgres configuration is required"));
}

#[tokio::test]
async fn test_create_storage_redis_missing_config() {
    let err = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    })
    .await
    .err()
    .expect("should fail");
    assert!(err.contains("Redis configuration is required"));
}

#[tokio::test]
async fn test_create_storage_with_hook() {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::{
        context::RequestContext,
        hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation},
    };

    struct NoOpHook;

    #[async_trait]
    impl StorageHook for NoOpHook {
        async fn before(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            Ok(BeforeHookResult::default())
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    let config = StorageFactoryConfig {
        backend: &HistoryBackend::Memory,
        oracle: None,
        postgres: None,
        redis: None,
        memory: None,
        hook: Some(Arc::new(NoOpHook)),
    };
    stores_and_reads_back(create_storage(config).await.unwrap()).await;
}
