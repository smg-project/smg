use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

/// A worker is found by the id the engine knows it by and written with
/// the address a request's success path shows. A registered URL that
/// carries a credential is written without it.
#[test]
fn a_worker_is_written_with_the_address_a_success_path_shows() {
    let plain = "http://127.0.0.1:8001";
    let keyed = "http://user:secret@127.0.0.1:8002";
    let workers = [plain, keyed]
        .map(|url| -> Arc<dyn Worker> { Arc::new(BasicWorkerBuilder::new(url).build()) });
    let found = addresses(&workers);
    assert_eq!(found.len(), 2);
    assert_eq!(found[&worker_id(workers[0].as_ref())], plain);
    let shown = found[&worker_id(workers[1].as_ref())];
    assert_eq!(shown, workers[1].redacted_url());
    assert!(!shown.contains("secret"), "{shown}");
    assert!(shown.contains("127.0.0.1:8002"), "{shown}");
}
