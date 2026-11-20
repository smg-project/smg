//! The policy document on the control plane: `GET /policy/spec` tells the
//! document in force, and `POST /policy/spec:validate` says what an apply
//! of another would do, with nothing applied.
//!
//! Both stand on the policy crate's operators' door ([`SpecDoor`]), which
//! the start leaves in the context beside the controller's store. Under
//! the classic policies no door stands, and each route refuses by one code
//! that names the setting. Nothing is applied in this process after the
//! start: the document in force is the start's, and a restart reads the
//! routing configuration again.
//!
//! A dry run validates a whole document. Its pass runs on the blocking
//! pool, one at a time, and a bounded number of calls wait for it.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::State,
    http::{
        header::{CONTENT_TYPE, ETAG, IF_MATCH, RETRY_AFTER},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::Serialize;
use smg_policy_controller::{
    operator::{
        Document, DryRun, Format, NeedsRow, SpecDoor, Turn, Unanswered, MAX_DOCUMENT_BYTES,
    },
    spec::{Digest, Needs},
    view::Phase,
};
use tracing::error;

use crate::{admin::json::refusal, middleware::SHED_RETRY_AFTER_SECS, server::AppState};

/// How many dry runs may wait for the one that runs. A call past them is
/// answered 429.
const DRY_RUNS_WAITING: usize = 32;

/// The operators' door over the controller's store, with the turn of its
/// dry runs.
pub(crate) struct Operators {
    door: Arc<SpecDoor>,
    dry_runs: Turn,
}

impl Operators {
    pub(crate) fn new(door: Arc<SpecDoor>) -> Self {
        Self {
            door,
            dry_runs: Turn::new(DRY_RUNS_WAITING),
        }
    }
}

/// The most bytes a document's body may hold: the door's own bound, kept
/// at the HTTP edge so that a longer body is never read whole.
pub(crate) fn document_bytes() -> usize {
    usize::try_from(MAX_DOCUMENT_BYTES).unwrap_or(usize::MAX)
}

/// The door, or the 409 of a gateway whose classic policies decide: it
/// holds no policy document.
fn operators(state: &AppState) -> Result<&Arc<Operators>, Response> {
    state.context.policy_operators.as_ref().ok_or_else(|| {
        refusal(
            StatusCode::CONFLICT,
            "POLICY_CONTROLLER_OFF",
            "The classic policies decide for this gateway, which holds no policy document; the \
             policy_controller setting turns the controller on"
                .to_string(),
        )
    })
}

/// A generation by its number and its digest, as each answer tells one.
#[derive(Serialize)]
struct Generation {
    generation: u64,
    /// `null` where no document stands.
    digest: Option<String>,
}

impl From<(u64, Option<Digest>)> for Generation {
    fn from((generation, digest): (u64, Option<Digest>)) -> Self {
        Self {
            generation,
            digest: digest.map(|digest| digest.to_string()),
        }
    }
}

/// The answer of `GET /policy/spec`.
#[derive(Serialize)]
struct InForce<'a> {
    generation: u64,
    digest: String,
    /// `yaml` or `json`: the form `document` is in.
    format: &'static str,
    applied_at: DateTime<Utc>,
    /// The document as its source wrote it.
    document: &'a str,
}

/// `GET /policy/spec`: the document in force as its source wrote it, with
/// its generation, its digest and when it was applied. `ETag` is the
/// digest: the guard a dry run takes in `If-Match`. 409 under the classic
/// policies.
pub(crate) async fn spec(State(state): State<Arc<AppState>>) -> Response {
    let operators = match operators(&state) {
        Ok(operators) => operators,
        Err(refused) => return refused,
    };
    // The start's document stands: nothing is applied after it.
    let held = match operators.door.document(0) {
        Ok(held) => held,
        Err(not_held) => {
            error!(%not_held, "the policy document in force is not held");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    let Document {
        generation,
        digest,
        document,
        format,
        applied,
    } = held;
    let Ok(document) = std::str::from_utf8(&document) else {
        error!(generation, "the policy document in force is no text");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let digest = digest.to_string();
    let etag = HeaderValue::from_str(&format!("\"{digest}\""));
    let mut response = Json(InForce {
        generation,
        digest,
        format: format.as_str(),
        applied_at: applied.into(),
        document,
    })
    .into_response();
    if let Ok(etag) = etag {
        response.headers_mut().insert(ETAG, etag);
    }
    response
}

/// What a host must feed for one model and phase under a generation.
#[derive(Serialize)]
struct Needed {
    model: String,
    phase: Phase,
    needs: Needs,
}

impl From<NeedsRow> for Needed {
    fn from(
        NeedsRow {
            model,
            phase,
            needs,
        }: NeedsRow,
    ) -> Self {
        Self {
            model,
            phase,
            needs,
        }
    }
}

/// What a host must feed under the standing generation and under the
/// document's.
#[derive(Serialize)]
struct Feed {
    before: Vec<Needed>,
    after: Vec<Needed>,
}

/// The answer of a dry run whose document would run.
#[derive(Serialize)]
struct WouldRun {
    /// The generation the document was checked against.
    standing: Generation,
    /// The number an apply would run the document under, with the
    /// document's digest. The standing generation, for its own document.
    would_run: Generation,
    needs: Feed,
}

/// A dry run's refusal: the control plane's envelope, with the generation
/// that stands and, for a document that would not run, the path of the
/// field the refusal is about.
#[derive(Serialize)]
struct Refused {
    error: String,
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_path: Option<String>,
    standing: Generation,
}

/// `POST /policy/spec:validate`: what an apply of the body would do
/// against the document in force, with nothing applied and no number
/// reserved. `Content-Type` names the document's form, YAML where none is
/// given. `If-Match` guards the call with the digest its caller read as
/// standing ([`guard`]).
///
/// 200 tells the number the document would run under and what a host must
/// feed before and after. 422 is a document that would not run, with its
/// field's path. 412 is a guard that names another digest. Each of the two
/// tells the generation that stands. 413 is a body over the door's bound,
/// refused at the edge and never passed. 429 says that as many dry runs
/// wait as may, with the `Retry-After` the gateway's sheds send. 409 under
/// the classic policies.
pub(crate) async fn validate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let operators = match operators(&state) {
        Ok(operators) => operators,
        Err(refused) => return refused,
    };
    let Some(format) = document_format(&headers) else {
        return refusal(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_MEDIA_TYPE",
            "A policy document is application/yaml or application/json".to_string(),
        );
    };
    let expected = match guard(&headers) {
        Ok(expected) => expected,
        Err(refused) => return refused,
    };
    let Some(turn) = operators.dry_runs.take().await else {
        let mut refused = refusal(
            StatusCode::TOO_MANY_REQUESTS,
            "TOO_MANY_DRY_RUNS",
            "As many dry runs wait as may; try again".to_string(),
        );
        // When to come again, as the gateway's sheds say it.
        let again = HeaderValue::from(SHED_RETRY_AFTER_SECS);
        refused.headers_mut().insert(RETRY_AFTER, again);
        return refused;
    };
    let door = Arc::clone(&operators.door);
    // The pass keeps its turn to its own end, off the runtime's workers: a
    // client that leaves never stops a pass. Where the control plane records
    // its mutations this handler runs on in a task of its own, so a call
    // that waits keeps its place after its client left.
    let ran = tokio::task::spawn_blocking(move || {
        let _turn = turn;
        door.dry_run(&expected, &body, format)
    })
    .await;
    match ran {
        Ok(Ok(DryRun {
            standing,
            would_run: Ok((generation, digest)),
            needs,
        })) => {
            let (before, after) = needs.unwrap_or_default();
            Json(WouldRun {
                standing: standing.into(),
                would_run: (generation, Some(digest)).into(),
                needs: Feed {
                    before: before.into_iter().map(Needed::from).collect(),
                    after: after.into_iter().map(Needed::from).collect(),
                },
            })
            .into_response()
        }
        Ok(Ok(DryRun {
            standing,
            would_run: Err(rejected),
            ..
        })) => {
            let refused = Refused {
                error: rejected.to_string(),
                code: "INVALID_DOCUMENT",
                field_path: Some(rejected.field_path()),
                standing: standing.into(),
            };
            (StatusCode::UNPROCESSABLE_ENTITY, Json(refused)).into_response()
        }
        Ok(Err(Unanswered::Guarded(generation, digest))) => {
            let refused = Refused {
                error: format!("If-Match names another digest; generation {generation} stands"),
                code: "PRECONDITION_FAILED",
                field_path: None,
                standing: (generation, digest).into(),
            };
            (StatusCode::PRECONDITION_FAILED, Json(refused)).into_response()
        }
        // A dry run names no generation on file.
        Ok(Err(unanswered @ Unanswered::NotHeld(_))) => {
            error!(%unanswered, "the policy dry run gave no verdict");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(join) => {
            error!(%join, "the policy dry run's task failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// The form of a dry run's body by its `Content-Type`: YAML where none is
/// given, else the form its whole media type names (`application/yaml`,
/// `application/x-yaml`, `text/yaml`, `application/json`). `None` for any
/// other type.
fn document_format(headers: &HeaderMap) -> Option<Format> {
    let Some(value) = headers.get(CONTENT_TYPE) else {
        return Some(Format::Yaml);
    };
    let media = value.to_str().ok()?.split(';').next()?.trim();
    match media.to_ascii_lowercase().as_str() {
        "application/yaml" | "application/x-yaml" | "text/yaml" => Some(Format::Yaml),
        "application/json" => Some(Format::Json),
        _ => None,
    }
}

/// A dry run's guard from `If-Match`: the digest its caller read as
/// standing, as `GET /policy/spec` writes it in `ETag`, quoted or bare. No
/// header guards nothing, and neither does `*`, which matches whatever
/// stands. Any other form is refused: a list, a weak tag or a value that is
/// no text would be a dry run its caller believes guarded by a digest the
/// door never compared.
fn guard(headers: &HeaderMap) -> Result<String, Response> {
    let mut values = headers.get_all(IF_MATCH).iter();
    let Some(first) = values.next() else {
        return Ok(String::new());
    };
    let one = first.to_str().ok().filter(|_| values.next().is_none());
    let digest = match one.map(str::trim) {
        Some("*") => return Ok(String::new()),
        Some(tag) => {
            let quoted = tag.strip_prefix('"').and_then(|tag| tag.strip_suffix('"'));
            quoted.unwrap_or(tag)
        }
        None => "",
    };
    if digest.is_empty() || digest.contains(['"', ',']) {
        return Err(refusal(
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            "If-Match carries one ETag, as GET /policy/spec answers it".to_string(),
        ));
    }
    Ok(digest.to_owned())
}

#[cfg(test)]
mod tests;
