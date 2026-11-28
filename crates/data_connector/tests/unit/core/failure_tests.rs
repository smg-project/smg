use super::*;

#[test]
fn a_redis_connection_or_io_failure_is_unavailable_and_an_answer_is_not() {
    let gone = redis::RedisError::from((redis::ErrorKind::Io, "connection reset"));
    assert!(matches!(Failure::redis(&gone), Failure::Unavailable(_)));
    let refused = redis::RedisError::from(std::io::Error::new(
        std::io::ErrorKind::ConnectionRefused,
        "refused",
    ));
    assert!(matches!(Failure::redis(&refused), Failure::Unavailable(_)));
    let wrong_type = redis::RedisError::from((redis::ErrorKind::UnexpectedReturnType, "WRONGTYPE"));
    assert!(matches!(Failure::redis(&wrong_type), Failure::Storage(_)));
    // A protocol parse failure leaves the connection unusable: the client
    // library counts it unrecoverable, and so does the classification.
    let broken = redis::RedisError::from((redis::ErrorKind::Parse, "bad frame"));
    assert!(matches!(Failure::redis(&broken), Failure::Unavailable(_)));
}

#[test]
fn a_pool_failure_keeps_its_cause_chain() {
    #[derive(Debug)]
    struct Outer(std::io::Error);
    impl Display for Outer {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str("error connecting to server")
        }
    }
    impl std::error::Error for Outer {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }
    let error = Outer(std::io::Error::new(
        std::io::ErrorKind::ConnectionRefused,
        "Connection refused (os error 111)",
    ));
    let Failure::Unavailable(text) = Failure::pool(&error) else {
        panic!("a pool failure is unavailable");
    };
    assert!(
        text.contains("error connecting to server: Connection refused (os error 111)"),
        "{text}"
    );
}

#[test]
fn only_a_timeout_or_an_unavailable_store_is_unavailable() {
    assert!(ResponseStorageError::Timeout("bound".into()).is_unavailable());
    assert!(ResponseStorageError::Timeout("bound".into()).timed_out());
    assert!(ResponseStorageError::Unavailable("gone".into()).is_unavailable());
    assert!(!ResponseStorageError::Unavailable("gone".into()).timed_out());
    assert!(!ConversationStorageError::StorageError("duplicate key".into()).is_unavailable());
    assert!(!ConversationStorageError::Rejected("tenant".into()).is_unavailable());
    assert!(
        !ConversationItemStorageError::from_failure(Failure::Storage("x".into())).is_unavailable()
    );
    assert!(
        ConversationItemStorageError::from_failure(Failure::Unavailable("x".into()))
            .is_unavailable()
    );
}
