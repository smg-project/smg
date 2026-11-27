//! Where a lease lives and the identity a replica holds it under: an identity
//! and its address, a lease's place, the process's own place, a lease client's
//! timeouts and its one try, a refused call that comes back inside the renew
//! deadline, and a lease file before its first step.

use super::*;

#[test]
fn the_identity_is_the_id_at_the_address_and_splits_back() {
    let named = identity("controller-0", Some("10.0.0.7:30100"));
    assert_eq!(named, "controller-0@10.0.0.7:30100");
    assert_eq!(split_identity(&named), ("controller-0", "10.0.0.7:30100"));
    let v6 = identity("controller-0", Some("[fd00::7]:30100"));
    assert_eq!(split_identity(&v6), ("controller-0", "[fd00::7]:30100"));
    let alone = identity("controller-0", None);
    assert_eq!(alone, "controller-0");
    assert_eq!(split_identity(&alone), ("controller-0", ""));
}

#[test]
fn a_lease_names_its_place() {
    assert_eq!(
        LeasePlace::Process.to_string(),
        "this process (one replica)"
    );
    assert!(!LeasePlace::Process.is_shared());
    let file = LeasePlace::File(PathBuf::from("/run/smg/lease"));
    assert_eq!(file.to_string(), "file /run/smg/lease");
    assert!(file.is_shared());
    #[cfg(feature = "kubernetes")]
    {
        let object = |namespace: Option<&str>| LeasePlace::Kubernetes {
            namespace: namespace.map(str::to_owned),
            name: "smg-policy".to_owned(),
        };
        assert_eq!(
            object(Some("policy")).to_string(),
            "Lease policy/smg-policy"
        );
        assert_eq!(object(None).to_string(), "Lease smg-policy");
        assert!(object(None).is_shared());
    }
}

#[tokio::test]
async fn the_process_place_makes_an_elector_alone_that_leads_at_its_take() {
    let mut alone = elector(
        &ElectionConfig::default(),
        "controller-0@127.0.0.1:1".to_owned(),
    )
    .await
    .expect("the process place needs nothing");
    let now = Instant::now();
    alone.tick(now).expect("the take");
    assert_eq!(alone.fence().leader_term(now), Some(1));
}

/// What the client library gives a client by itself is read first, so a
/// release that moves its retry's default again fails here.
#[cfg(feature = "kubernetes")]
#[test]
fn a_lease_clients_round_trip_is_bounded_by_the_renew_deadline_and_made_once() {
    let inferred = kube::Config::new("https://api.invalid".parse().expect("a url"));
    assert!(inferred.default_retry, "the library's own default");
    let deadline = Duration::from_millis(1_500);

    let lease = super::super::lease_config(inferred, deadline);

    assert_eq!(lease.connect_timeout, Some(deadline));
    assert_eq!(lease.read_timeout, Some(deadline));
    assert_eq!(lease.write_timeout, Some(deadline));
    assert!(!lease.default_retry);
}

/// The API server here refuses every call with 503 and names an hour to
/// wait. The refusal comes back inside the renew deadline, after one
/// call: the client library's own retry would hold the election thread
/// behind that call for the hour.
#[cfg(feature = "kubernetes")]
#[tokio::test(flavor = "multi_thread")]
async fn a_lease_call_the_api_server_refuses_comes_back_inside_the_renew_deadline() {
    use axum::http::{header, StatusCode};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a port of this host");
    let address = listener.local_addr().expect("the listener's address");
    let calls = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&calls);
    let refuses = axum::Router::new().fallback(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        std::future::ready((
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "3600")],
            "the server is currently unable to handle the request",
        ))
    });
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let server = tokio::spawn(async move { axum::serve(listener, refuses).await });

    let deadline = Duration::from_secs(5);
    let inferred = kube::Config::new(
        format!("http://{address}")
            .parse()
            .expect("the server's url"),
    );
    let mut lease =
        super::super::kubernetes_lease(inferred, Some("policy"), "smg-policy", deadline)
            .expect("a client of the server");
    // The election's thread is no runtime worker: a call blocks its
    // thread on the runtime.
    let read = tokio::task::spawn_blocking(move || lease.read());
    let refused = tokio::time::timeout(deadline, read)
        .await
        .expect("the refusal comes back inside the renew deadline")
        .expect("the read ran");

    assert!(refused.is_err(), "a refused read is an error: {refused:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the call was made once");
    server.abort();
}

#[tokio::test]
async fn a_lease_file_needs_its_directory_and_writes_nothing_before_the_first_step() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let at = |path: PathBuf| ElectionConfig {
        lease: LeasePlace::File(path),
        ..ElectionConfig::default()
    };
    let candidate = "controller-0@127.0.0.1:1".to_owned();

    let missing = dir.path().join("missing");
    match elector(&at(missing.join("lease")), candidate.clone()).await {
        Err(LeaseError::Directory { directory, .. }) => assert_eq!(directory, missing),
        other => panic!("a missing directory was not refused: {other:?}"),
    }

    let file = dir.path().join("file");
    std::fs::write(&file, b"").expect("a file");
    match elector(&at(file.join("lease")), candidate.clone()).await {
        Err(LeaseError::NotADirectory(directory)) => assert_eq!(directory, file),
        other => panic!("a file for a directory was not refused: {other:?}"),
    }

    let lease = dir.path().join("lease");
    let opened = elector(&at(lease.clone()), candidate.clone())
        .await
        .expect("the directory is there");
    assert!(!opened.is_holder());
    assert!(!lease.exists(), "nothing is written before the first step");

    // A file named without a directory lives where the process runs.
    let alone = PathBuf::from("a-lease-named-without-a-directory");
    let opened = elector(&at(alone.clone()), candidate)
        .await
        .expect("the process runs in a directory");
    assert!(!opened.is_holder());
    assert!(!alone.exists(), "nothing is written before the first step");
}
