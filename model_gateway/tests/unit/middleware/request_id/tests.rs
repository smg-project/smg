use axum::{body::Body, routing::get, Router};
use http_body_util::BodyExt;
use tower::ServiceExt;

use super::*;

#[test]
fn generated_ids_use_endpoint_prefixes() {
    for (path, prefix) in [
        ("/v1/chat/completions", "chatcmpl-"),
        ("/v1/completions", "cmpl-"),
        ("/generate", "gnt-"),
        ("/v1/responses", "resp-"),
        ("/v1/messages", "msg_"),
        ("/v1/embeddings", "req-"),
    ] {
        assert!(
            generate_request_id(path).starts_with(prefix),
            "path {path} must generate a {prefix} id"
        );
    }
}

#[test]
fn the_inbound_id_bound_is_one_to_128_visible_ascii_characters() {
    for accepted in ["a", "!~", &"x".repeat(MAX_INBOUND_REQUEST_ID_LEN)] {
        assert!(is_acceptable_inbound_id(accepted), "{accepted:?}");
    }
    for dropped in [
        "",
        " ",
        "a b",
        "tab\there",
        "caf\u{e9}",
        "\u{7f}",
        &"x".repeat(MAX_INBOUND_REQUEST_ID_LEN + 1),
    ] {
        assert!(!is_acceptable_inbound_id(dropped), "{dropped:?}");
    }
}

/// The id the handler found in the request extensions, the first
/// `x-request-id` header value it received (`-` when absent) and how many
/// values of that header it received come back as the body,
/// `extension|header|count`; the layer echoes the id it chose in
/// `x-request-id`.
fn app() -> Router {
    app_with(vec!["x-request-id".into(), "x-correlation-id".into()])
}

/// [`app`] with its own list of configured request-id headers.
fn app_with(names: Vec<String>) -> Router {
    Router::new()
        .route("/v1/chat/completions", get(ids_seen))
        .route("/v1/messages", get(ids_seen))
        .route("/v1/models", get(ids_seen))
        .layer(RequestIdLayer::new(names))
}

/// The handler of [`app_with`]'s routes.
async fn ids_seen(request: Request) -> String {
    let extension = request
        .extensions()
        .get::<RequestId>()
        .map(|id| id.0.clone())
        .unwrap_or_default();
    let header = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-");
    let count = request.headers().get_all("x-request-id").iter().count();
    format!("{extension}|{header}|{count}")
}

/// (the echoed `x-request-id`, the id the handler saw in the extensions,
/// the first `x-request-id` header value the handler received, how many
/// values it received) for a request carrying `headers`.
async fn ids_for(headers: &[(&str, &str)]) -> (String, String, String, usize) {
    let mut request = Request::builder().uri("/v1/chat/completions");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = app()
        .oneshot(request.body(Body::empty()).expect("request"))
        .await
        .expect("response");
    let echoed = response.headers()["x-request-id"]
        .to_str()
        .expect("visible ASCII")
        .to_owned();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body = String::from_utf8(body.to_vec()).expect("utf-8");
    let mut parts = body.split('|');
    let (extension, header, count) = (
        parts.next().expect("extension"),
        parts.next().expect("header"),
        parts.next().expect("count"),
    );
    (
        echoed,
        extension.to_owned(),
        header.to_owned(),
        count.parse().expect("a count"),
    )
}

fn is_fresh_chat_id(id: &str) -> bool {
    id.strip_prefix("chatcmpl-")
        .is_some_and(|rest| rest.len() == 24 && rest.bytes().all(|b| b.is_ascii_alphanumeric()))
}

#[tokio::test]
async fn an_inbound_id_within_the_bound_is_honoured() {
    for id in [
        "req_client_1",
        "4bf92f3577b34da6a3ce929d0e0e4736",
        &"a".repeat(128),
    ] {
        let (echoed, seen, header, count) = ids_for(&[("x-request-id", id)]).await;
        assert_eq!(echoed, id, "the echoed id");
        assert_eq!(seen, id, "the id the handler saw");
        assert_eq!(
            (header.as_str(), count),
            (id, 1),
            "the header reaches the handler as is"
        );
    }
}

#[tokio::test]
async fn an_inbound_id_over_the_bound_is_replaced_by_a_fresh_one() {
    let long = "R".repeat(129);
    let (echoed, seen, header, _) = ids_for(&[("x-request-id", &long)]).await;
    assert_eq!(echoed, seen, "the handler and the client see the same id");
    assert!(is_fresh_chat_id(&echoed), "a fresh id, got {echoed:?}");
    assert!(
        !echoed.contains("RRRR"),
        "nothing of the inbound value survives"
    );
    assert_eq!(
        header, echoed,
        "the request header carries the fresh id, so a worker reading or forwarding it sees the gateway's id"
    );
    // The first configured header that is present decides; a second one
    // is not promoted in its place.
    let (echoed, _, _, _) =
        ids_for(&[("x-request-id", &long), ("x-correlation-id", "corr-1")]).await;
    assert!(is_fresh_chat_id(&echoed), "a fresh id, got {echoed:?}");
}

#[tokio::test]
async fn an_inbound_id_that_is_not_visible_ascii_is_replaced() {
    for value in ["tab\there", "two words", ""] {
        let (echoed, seen, header, _) = ids_for(&[("x-request-id", value)]).await;
        assert_eq!(echoed, seen, "the handler and the client see the same id");
        assert!(
            is_fresh_chat_id(&echoed),
            "a fresh id for {value:?}, got {echoed:?}"
        );
        assert_eq!(
            header, echoed,
            "the request header is rewritten for {value:?}"
        );
    }
}

/// A repeated request-id header forwards every value; after the layer
/// the request carries exactly one, the honoured first value or the
/// fresh id, and an unchecked second value goes no further.
#[tokio::test]
async fn a_repeated_request_id_header_keeps_only_the_chosen_id() {
    let long = "R".repeat(129);
    let (echoed, seen, header, count) =
        ids_for(&[("x-request-id", "ok-first"), ("x-request-id", &long)]).await;
    assert_eq!((echoed.as_str(), seen.as_str()), ("ok-first", "ok-first"));
    assert_eq!(
        (header.as_str(), count),
        ("ok-first", 1),
        "one value, the honoured one"
    );
    let (echoed, _, header, count) =
        ids_for(&[("x-request-id", &long), ("x-request-id", "ok-second")]).await;
    assert!(
        is_fresh_chat_id(&echoed),
        "the first value decides: {echoed:?}"
    );
    assert_eq!(
        (header.as_str(), count),
        (echoed.as_str(), 1),
        "one value, the fresh id"
    );
}

/// A request in the Messages API's shape (a Messages route, or any route
/// under the `anthropic-version` header the Anthropic SDKs send on every
/// call) is answered with the id as `request-id` as well: the header those
/// SDKs read for `_request_id` and for an error's `request_id`. Every other
/// response carries `x-request-id` alone, as before.
#[tokio::test]
async fn a_messages_api_request_is_echoed_its_id_as_request_id_too() {
    for (path, version, as_request_id) in [
        ("/v1/messages", None, true),
        ("/v1/models", Some("2023-06-01"), true),
        ("/v1/chat/completions", Some("2023-06-01"), true),
        ("/v1/models", None, false),
        ("/v1/chat/completions", None, false),
    ] {
        let mut request = Request::builder()
            .uri(path)
            .header("x-request-id", "req_client_1");
        if let Some(version) = version {
            request = request.header("anthropic-version", version);
        }
        let response = app()
            .oneshot(request.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(response.headers()["x-request-id"], "req_client_1", "{path}");
        assert_eq!(
            response
                .headers()
                .get("request-id")
                .map(|value| value.to_str().expect("visible ASCII")),
            as_request_id.then_some("req_client_1"),
            "{path} under anthropic-version {version:?}"
        );
    }

    // A fresh id is echoed under both names too, the same value.
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let echoed = &response.headers()["x-request-id"];
    assert!(echoed.to_str().expect("visible ASCII").starts_with("msg_"));
    assert_eq!(&response.headers()["request-id"], echoed);
}

/// A response that already carries a `request-id` (a proxying route
/// relaying its upstream's) keeps it; `x-request-id` still names the id
/// the gateway chose.
#[tokio::test]
async fn a_request_id_already_on_the_response_is_kept() {
    let app = Router::new()
        .route(
            "/v1/messages",
            get(|| async { ([("request-id", "req_upstream_1")], "relayed") }),
        )
        .layer(RequestIdLayer::new(vec!["x-request-id".into()]));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .header("x-request-id", "req_client_1")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.headers()["request-id"], "req_upstream_1");
    assert_eq!(response.headers()["x-request-id"], "req_client_1");
}

/// A configured name that does not parse could never match a request;
/// it is skipped and the next configured header is consulted.
#[tokio::test]
async fn a_configured_name_that_does_not_parse_is_skipped() {
    let app = app_with(vec!["bad name".into(), "x-request-id".into()]);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .header("x-request-id", "ok-1")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.headers()["x-request-id"], "ok-1");
}

/// What stands on `headers` under [`OWN_REQUEST_ID`]: the first value
/// under the name, where it is text. The engine's reader of a header
/// map takes the same first value, as its bytes.
fn restated(headers: &http::HeaderMap) -> Option<&str> {
    headers.get(&OWN_REQUEST_ID)?.to_str().ok()
}

/// The handler of [`own_for`]'s route: every value it received under
/// [`OWN_REQUEST_ID`], joined by `,`, and what [`restated`] reads of
/// the request's headers (`-` for none), as `values|read`.
async fn own_seen(request: Request) -> String {
    let values: Vec<_> = request
        .headers()
        .get_all(&OWN_REQUEST_ID)
        .iter()
        .map(|value| value.to_str().unwrap_or("?"))
        .collect();
    let read = restated(request.headers()).unwrap_or("-");
    format!("{}|{read}", values.join(","))
}

/// (the echoed `x-request-id`, what [`own_seen`] answered) for a request
/// carrying `headers`, behind the layer. Where `restating`, the write
/// stands between the layer and the route as a layer of its own, as the
/// server installs it under a policy spec that keeps a sample.
async fn own_for(restating: bool, headers: &[(&str, &str)]) -> (String, String) {
    let route = Router::new().route("/v1/models", get(own_seen));
    let route = if restating {
        route.layer(tower::util::MapRequestLayer::new(restate))
    } else {
        route
    };
    let app = route.layer(RequestIdLayer::new(vec!["x-request-id".into()]));
    let mut request = Request::builder().uri("/v1/models");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = app
        .oneshot(request.body(Body::empty()).expect("request"))
        .await
        .expect("response");
    let echoed = response.headers()["x-request-id"]
        .to_str()
        .expect("visible ASCII")
        .to_owned();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (echoed, String::from_utf8(body.to_vec()).expect("utf-8"))
}

/// Behind the layer the write restates the id the layer chose on the
/// request, once: the id it made, and a client's own.
#[tokio::test]
async fn the_write_states_the_chosen_id_on_the_request() {
    let (echoed, seen) = own_for(true, &[]).await;
    assert!(echoed.starts_with("req-"), "a fresh id, got {echoed:?}");
    assert_eq!(seen, format!("{echoed}|{echoed}"));

    let (echoed, seen) = own_for(true, &[("x-request-id", "req_client_1")]).await;
    assert_eq!(echoed, "req_client_1");
    assert_eq!(seen, "req_client_1|req_client_1");
}

/// The write replaces whatever a client sent under the private name:
/// one value or several, with an id header of the client's or without.
#[tokio::test]
async fn the_write_replaces_what_a_client_sent_under_its_name() {
    let own = OWN_REQUEST_ID.as_str();
    let (echoed, seen) = own_for(true, &[(own, "forged")]).await;
    assert!(echoed.starts_with("req-"), "a fresh id, got {echoed:?}");
    assert_eq!(
        seen,
        format!("{echoed}|{echoed}"),
        "one value, the gateway's"
    );

    let sent = [(own, "forged"), (own, "forged-too"), ("x-request-id", "ok")];
    let (echoed, seen) = own_for(true, &sent).await;
    assert_eq!(echoed, "ok");
    assert_eq!(seen, "ok|ok", "one value, the chosen id");
}

/// The layer alone writes no such header and leaves a client's value of
/// that name as it came. So it is in a gateway that places with its
/// classic policies, and under a policy spec that keeps no sample:
/// nothing reads the value there.
#[tokio::test]
async fn the_layer_alone_writes_no_header_of_its_own() {
    let (_, seen) = own_for(false, &[]).await;
    assert_eq!(seen, "|-");

    let (_, seen) = own_for(false, &[(OWN_REQUEST_ID.as_str(), "a-clients")]).await;
    assert_eq!(seen, "a-clients|a-clients");
}

/// A request that holds no chosen id keeps no value under the private
/// name behind the write: what a client sent there is removed, every
/// value of it, and not left for the reader.
#[test]
fn the_write_removes_a_clients_value_where_the_request_holds_no_id() {
    let own = OWN_REQUEST_ID.as_str();
    let request = Request::builder()
        .uri("/v1/models")
        .header(own, "forged")
        .header(own, "forged-too")
        .body(Body::empty())
        .expect("request");
    let request = restate(request);
    assert_eq!(restated(request.headers()), None);
    assert_eq!(request.headers().get_all(&OWN_REQUEST_ID).iter().count(), 0);
}

/// No route sends the private header to a worker: the routers' one
/// allow-list of request headers refuses its name. The list passes the
/// request-id headers by name and by a prefix, which this name is not
/// under.
#[test]
fn the_private_header_is_not_on_the_allow_list_of_request_headers() {
    use crate::routers::common::header_utils::should_forward_request_header;

    assert!(!should_forward_request_header(OWN_REQUEST_ID.as_str()));
    assert!(should_forward_request_header("x-request-id"));
    assert!(should_forward_request_header("x-request-id-of-a-proxy"));
}
