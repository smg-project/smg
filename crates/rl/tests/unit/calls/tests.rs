use super::*;

fn reply(success: bool, message: &str) -> RankReply {
    RankReply {
        success,
        message: message.to_string(),
    }
}

/// Every rank must succeed; a failure names the failing ranks' distinct
/// words, never a succeeding rank's.
#[test]
fn the_answer_succeeds_only_when_every_rank_did() {
    let (status, body) = route_answer(&[reply(true, ""), reply(true, "")]);
    assert_eq!(status, 200);
    assert_eq!(body, json!({"success": true, "message": ""}));

    let (status, body) = route_answer(&[
        reply(true, "ok"),
        reply(false, "a pause is already in progress"),
        reply(false, "a pause is already in progress"),
    ]);
    assert_eq!(status, 400);
    assert_eq!(
        body,
        json!({"success": false, "message": "a pause is already in progress"})
    );
}
