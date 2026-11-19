//! The two calls the port derives: the health verdict from the whole
//! answer, and the KV transfer engine id from the server's labels.

use super::*;
use crate::test_support::ScriptedAnswers;

/// A server's labels as a case writes them.
type Labels<'a> = &'a [(&'a str, &'a str)];

/// The verdict is the whole answer's, whatever the engine says beside it,
/// and a call the engine did not answer stays an error.
#[tokio::test]
async fn the_health_verdict_is_the_whole_answers() {
    for healthy in [true, false] {
        let engine = ScriptedAnswers::answering_health(Ok(HealthAnswer {
            healthy,
            message: "still loading".to_string(),
        }));
        assert_eq!(engine.health_check().await.ok(), Some(healthy));
    }

    let gone = ScriptedAnswers::answering_health(Err(tonic::Code::Unavailable));
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
        let engine = ScriptedAnswers::reporting(Ok(labels));
        assert_eq!(
            engine.kv_engine_id().await.ok(),
            Some(id.map(str::to_string)),
            "{labels:?}"
        );
    }

    let refused = ScriptedAnswers::reporting(Err(tonic::Code::Unimplemented));
    assert_eq!(
        refused.kv_engine_id().await.map_err(|status| status.code()),
        Err(tonic::Code::Unimplemented)
    );
}
