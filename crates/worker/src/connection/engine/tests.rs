//! The two calls the port derives: the health verdict from the whole
//! answer, and the KV transfer engine id from the server's labels.

use super::*;

/// A server's labels as a case writes them.
type Labels<'a> = &'a [(&'a str, &'a str)];

/// A connection that answers its health and its server labels as scripted,
/// and refuses every other call.
struct Answers {
    health: Result<HealthAnswer, tonic::Code>,
    server: Result<HashMap<String, String>, tonic::Code>,
}

fn not_scripted<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unimplemented(
        "the case scripts no such call",
    ))
}

fn scripted<T: Clone>(answer: &Result<T, tonic::Code>) -> Result<T, tonic::Status> {
    answer
        .clone()
        .map_err(|code| tonic::Status::new(code, "scripted"))
}

#[async_trait]
impl EngineConnection for Answers {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        scripted(&self.health)
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        not_scripted()
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        scripted(&self.server)
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }
}

fn answering_health(health: Result<HealthAnswer, tonic::Code>) -> Answers {
    Answers {
        health,
        server: Err(tonic::Code::Unimplemented),
    }
}

fn reporting(server: Result<Labels<'_>, tonic::Code>) -> Answers {
    Answers {
        health: Err(tonic::Code::Unimplemented),
        server: server.map(|labels| {
            labels
                .iter()
                .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                .collect()
        }),
    }
}

/// The verdict is the whole answer's, whatever the engine says beside it,
/// and a call the engine did not answer stays an error.
#[tokio::test]
async fn the_health_verdict_is_the_whole_answers() {
    for healthy in [true, false] {
        let engine = answering_health(Ok(HealthAnswer {
            healthy,
            message: "still loading".to_string(),
        }));
        assert_eq!(engine.health_check().await.ok(), Some(healthy));
    }

    let gone = answering_health(Err(tonic::Code::Unavailable));
    assert_eq!(
        gone.health_check().await.map_err(|status| status.code()),
        Err(tonic::Code::Unavailable)
    );
}

/// The id is the server's label of that name. An empty one is no id, as is
/// a server that reports none, and a refused read of the labels refuses the
/// id too.
#[tokio::test]
async fn the_kv_engine_id_is_read_from_the_servers_labels() {
    let cases: [(Labels<'_>, Option<&str>); 3] = [
        (
            &[("kv_engine_id", "engine-a"), ("version", "1")],
            Some("engine-a"),
        ),
        (&[("kv_engine_id", "")], None),
        (&[("version", "1")], None),
    ];
    for (labels, id) in cases {
        let engine = reporting(Ok(labels));
        assert_eq!(
            engine.kv_engine_id().await.ok(),
            Some(id.map(str::to_string)),
            "{labels:?}"
        );
    }

    let refused = reporting(Err(tonic::Code::Unimplemented));
    assert_eq!(
        refused.kv_engine_id().await.map_err(|status| status.code()),
        Err(tonic::Code::Unimplemented)
    );
}
