//! The policy controller on the control plane: `GET /policy/spec` tells the
//! document in force, `POST /policy/spec:validate` says what an apply of
//! another would do, with nothing applied, and `GET /policy/snapshot` tells
//! the workers the policy controller decides over.
//!
//! The three read what the start left in the context ([`Source`]). A
//! gateway that decides in its own process holds the policy engine and no
//! store. Its document is the engine's, told with the YAML the start wrote,
//! a dry run runs against that document ([`dry_run_against`]), and its
//! workers are the worker registry's. A gateway that follows a controller
//! holds a policy store, and the three stand on the policy crate's
//! operators' door ([`DocumentManager`]) and the store's worker rows. Under the
//! classic policies neither stands, and each route refuses by one code that
//! names the setting. Nothing is applied in this process after the start:
//! the document in force is the start's, and a restart reads the routing
//! configuration again.
//!
//! A dry run validates a whole document and a snapshot copies rows. Each
//! runs on the blocking pool, one at a time of its kind, and a bounded
//! number of calls wait for it.

use std::{sync::Arc, time::Instant};

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
use serde::{Deserialize, Serialize};
use smg_policy_controller::{
    document::{
        operator::{
            dry_run_against,
            snapshot::{self, Rows, WorkerRowsQuery},
            Document, DocumentManager, DryRun, EnforceRefused, Format, NeedsRow, SerialTurn,
            MAX_DOCUMENT_BYTES,
        },
        Digest, Needs,
    },
    engine::Phase,
    store::{identity, MemoryStore, WorkerFacts, WorkerRow, WorkerStatus as RowStatus},
    worker_host::{facts, pairing_descriptor, registrant_of},
};
use smg_worker::{
    identity::basic::{RuntimeType, Worker},
    jobs::WorkerJobAsker,
    labels::DISCOVERY_PROVIDER_LABEL,
    worker::WorkerStatus,
};
use tracing::error;

use super::InProcess;
use crate::{
    admin::{json::refusal, params::AdminQuery},
    config::RouterConfig,
    middleware::SHED_RETRY_AFTER_SECS,
    routers::common::snapshot::WorkerRegistry,
    server::AppState,
};

/// How many dry runs may wait for the one that runs. A call past them is
/// answered 429.
const DRY_RUNS_WAITING: usize = 32;

/// How many snapshots may wait for the one that is taken. A call past
/// them is answered 429.
const SNAPSHOTS_WAITING: usize = 32;

/// The most rows a snapshot tells, and the number it tells where its
/// caller names none. A fleet past it is read by model.
const SNAPSHOT_ROWS: usize = 1_024;

/// Where the three routes read.
enum Source {
    /// A gateway that follows a controller: the operators' door over its
    /// policy store, and the store for its worker rows.
    Store {
        door: Arc<DocumentManager>,
        store: Arc<MemoryStore>,
    },
    /// A gateway that decides in its own process: what its start left, the
    /// header names it denies a document's view, and its worker registry.
    InProcess {
        started: InProcess,
        denied_headers: Vec<String>,
        registry: Arc<WorkerRegistry>,
    },
}

/// What the three routes read, each call behind the turn of its kind.
pub(crate) struct Operators {
    source: Arc<Source>,
    dry_runs: SerialTurn,
    snapshots: SerialTurn,
}

impl Operators {
    /// Over the policy store of a gateway that follows a controller, and
    /// the operators' door that applied the start's document to it.
    pub(crate) fn new(store: Arc<MemoryStore>, door: Arc<DocumentManager>) -> Self {
        Self::over(Source::Store { door, store })
    }

    /// Over what the start of a gateway that decides in its own process
    /// left, and the gateway's worker registry.
    pub(crate) fn in_process(
        started: InProcess,
        config: &RouterConfig,
        registry: Arc<WorkerRegistry>,
    ) -> Self {
        Self::over(Source::InProcess {
            started,
            denied_headers: config.denied_view_headers().map(str::to_owned).into(),
            registry,
        })
    }

    fn over(source: Source) -> Self {
        Self {
            source: Arc::new(source),
            dry_runs: SerialTurn::new(DRY_RUNS_WAITING),
            snapshots: SerialTurn::new(SNAPSHOTS_WAITING),
        }
    }
}

impl Source {
    /// The document in force as the read tells it: the one the door holds,
    /// or the engine's under the YAML and the time the start kept.
    fn document(&self) -> Result<Document, String> {
        match self {
            Self::Store { door, .. } => door.document(0).map_err(|not_held| not_held.to_string()),
            Self::InProcess { started, .. } => {
                let spec = started.engine.spec();
                Ok(Document {
                    generation: spec.generation(),
                    digest: spec.digest().ok_or("the engine's document has no digest")?,
                    document: started.written.clone().into_bytes(),
                    format: Format::Yaml,
                    applied: started.applied,
                })
            }
        }
    }

    /// What an apply of `document` would do against the document in force.
    fn dry_run(
        &self,
        expected: &str,
        document: &[u8],
        format: Format,
    ) -> Result<DryRun, EnforceRefused> {
        match self {
            Self::Store { door, .. } => door.dry_run(expected, document, format),
            Self::InProcess {
                started,
                denied_headers,
                ..
            } => {
                let denied: Vec<&str> = denied_headers.iter().map(String::as_str).collect();
                let standing = started.engine.spec();
                dry_run_against(
                    standing,
                    &started.stages,
                    &denied,
                    expected,
                    document,
                    format,
                )
            }
        }
    }

    /// The page `asked` for, taken at `now`, under the generation that runs.
    fn snapshot(&self, asked: &WorkerRowsQuery, now: Instant) -> Snapshot {
        let (generation, (rows, serving)) = match self {
            Self::Store { store, .. } => (
                store.snapshot().spec().generation(),
                store.rows_serving(&asked.models, asked.limit),
            ),
            Self::InProcess {
                started, registry, ..
            } => (
                started.engine.spec().generation(),
                workers_serving(registry, asked, now),
            ),
        };
        Snapshot {
            generation,
            page: snapshot::worker_rows_page(asked, Rows::Told(rows, serving), now),
        }
    }
}

/// The registry's workers as a page tells them: the first `asked.limit` in
/// the order of their addresses as registered, of the ones that serve a
/// model `asked` names, with the count that serve one. No model asked is
/// every worker. A model matches by an equal id among the worker's own, as
/// the store's read matches a row's, so a worker that names no model is
/// told under no model's name. A hosted provider's worker is left out: no
/// routing policy places a request on it.
fn workers_serving(
    registry: &WorkerRegistry,
    asked: &WorkerRowsQuery,
    now: Instant,
) -> (Vec<WorkerRow>, usize) {
    let named = |worker: &Arc<dyn Worker>| {
        let models = &asked.models;
        models.is_empty() || worker.models().iter().any(|card| models.contains(&card.id))
    };
    let mut serving: Vec<Arc<dyn Worker>> = registry
        .get_all()
        .into_iter()
        .filter(|worker| worker.metadata().spec.runtime_type != RuntimeType::External)
        .filter(named)
        .collect();
    let count = serving.len();
    serving.sort_by(|one, other| one.metadata().spec.url.cmp(&other.metadata().spec.url));
    serving.truncate(asked.limit);
    let rows = serving.iter().map(|worker| row_of(worker.as_ref(), now));
    (rows.collect(), count)
}

/// One worker object as a page tells a worker: the facts the object states
/// with its own pairing facts, under the id the policy crate derives from
/// its address. The registry numbers no life of a worker, so the id and
/// the incarnation are those of a first one. The revision counts the
/// object's replacements at its address. The age is the time since the
/// worker last became routable, and none for a worker that tracks none.
/// The maker is the one a registration by discovery states, for a worker
/// that carries a discovery source's provenance label.
fn row_of(worker: &dyn Worker, now: Instant) -> WorkerRow {
    let facts = WorkerFacts {
        pairing: pairing_descriptor(worker),
        ..facts(worker, None)
    };
    let labels = &worker.metadata().spec.labels;
    let maker = labels
        .contains_key(DISCOVERY_PROVIDER_LABEL)
        .then(|| registrant_of(&WorkerJobAsker::Discovery, &facts.engine).ok())
        .flatten();
    WorkerRow {
        id: identity(&facts.key, facts.incarnation),
        status: match worker.status() {
            WorkerStatus::Pending => RowStatus::Pending,
            WorkerStatus::Ready => RowStatus::Ready,
            WorkerStatus::NotReady => RowStatus::NotReady,
            WorkerStatus::Failed => RowStatus::Failed,
            WorkerStatus::Draining => RowStatus::Draining,
        },
        revision: worker.revision(),
        admitted_at: now.checked_sub(worker.admitted_age()).unwrap_or(now),
        maker: maker.unwrap_or_default(),
        facts,
    }
}

/// The most bytes a document's body may hold: the door's own bound, kept
/// at the HTTP edge so that a longer body is never read whole.
pub(crate) fn document_bytes() -> usize {
    usize::try_from(MAX_DOCUMENT_BYTES).unwrap_or(usize::MAX)
}

/// What the routes read, or the 409 of a gateway whose classic policies
/// decide: it holds no policy document.
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
struct SpecInForce<'a> {
    generation: u64,
    digest: String,
    /// `yaml` or `json`: the form `document` is in.
    format: &'static str,
    applied_at: DateTime<Utc>,
    /// The document as the gateway holds it: the YAML its start applied,
    /// its own writing of a given document.
    document: &'a str,
}

/// `GET /policy/spec`: the document in force as the gateway holds it, with
/// its generation, its digest and when it was applied: the start's time,
/// for a gateway that decides in its own process. `ETag` is the digest: the
/// guard a dry run takes in `If-Match`. 409 under the classic policies.
pub(crate) async fn spec(State(state): State<Arc<AppState>>) -> Response {
    let operators = match operators(&state) {
        Ok(operators) => operators,
        Err(refused) => return refused,
    };
    // The start's document stands: nothing is applied after it.
    let held = match operators.source.document() {
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
    let mut response = Json(SpecInForce {
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
        return full(
            "TOO_MANY_DRY_RUNS",
            "As many dry runs wait as may; try again",
        );
    };
    let source = Arc::clone(&operators.source);
    // The pass keeps its turn to its own end, off the runtime's workers: a
    // client that leaves never stops a pass. Where the control plane records
    // its mutations this handler runs on in a task of its own, so a call
    // that waits keeps its place after its client left.
    let ran = tokio::task::spawn_blocking(move || {
        let _turn = turn;
        source.dry_run(&expected, &body, format)
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
        Ok(Err(EnforceRefused::Guarded(generation, digest))) => {
            let refused = Refused {
                error: format!("If-Match names another digest; generation {generation} stands"),
                code: "PRECONDITION_FAILED",
                field_path: None,
                standing: (generation, digest).into(),
            };
            (StatusCode::PRECONDITION_FAILED, Json(refused)).into_response()
        }
        // A dry run names no generation on file.
        Ok(Err(unanswered @ EnforceRefused::NotHeld(_))) => {
            error!(%unanswered, "the policy dry run gave no verdict");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(join) => {
            error!(%join, "the policy dry run's task failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// The answer to a call past the ones of its kind that may wait: 429 under
/// `code`, with the `Retry-After` the gateway's sheds send.
fn full(code: &str, error: &str) -> Response {
    let mut refused = refusal(StatusCode::TOO_MANY_REQUESTS, code, error.to_string());
    let again = HeaderValue::from(SHED_RETRY_AFTER_SECS);
    refused.headers_mut().insert(RETRY_AFTER, again);
    refused
}

/// What a snapshot's caller asks. Any other key is refused by name, so a
/// misspelt filter never tells every row in the filtered one's place.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SnapshotQuery {
    /// Model ids, comma separated: the rows that serve one of them. Every
    /// row where none is named.
    models: Option<String>,
    /// The most rows to tell, at most [`SNAPSHOT_ROWS`].
    limit: Option<usize>,
}

/// The answer of `GET /policy/snapshot`: the page as the policy crate
/// writes one, under the generation that runs.
#[derive(Serialize)]
struct Snapshot {
    generation: u64,
    #[serde(flatten)]
    page: snapshot::WorkerRowsPage,
}

/// `GET /policy/snapshot`: the workers the policy controller decides
/// over, in registration key order, each address without its userinfo: the
/// registry's own workers where the gateway decides in its own process,
/// and the policy store's rows where it follows a controller. `models`
/// keeps the rows that serve one of the models it names, comma separated.
/// `limit` caps the rows, at most 1,024, which is the default too.
/// `truncated` says that a matching row was left out, by the cap or by
/// the policy crate's bound on a page: 2 MiB by its measure of the rows,
/// taken ahead of the JSON, whose keys and escapes may pass it. The page
/// has no cursor: a fleet past the cap is read by model, and a model that
/// more than 1,024 rows serve cannot be read whole.
///
/// 400 names the key of an ask outside the served call's bounds
/// ([`asked`]). 429 says that as many snapshots wait as may, with
/// `Retry-After`. 409 under the classic policies.
pub(crate) async fn snapshot(
    State(state): State<Arc<AppState>>,
    AdminQuery(query): AdminQuery<SnapshotQuery>,
) -> Response {
    let operators = match operators(&state) {
        Ok(operators) => operators,
        Err(refused) => return refused,
    };
    let asked = match asked(&query) {
        Ok(asked) => asked,
        Err(why) => return refusal(StatusCode::BAD_REQUEST, "BAD_REQUEST", why),
    };
    let Some(turn) = operators.snapshots.take().await else {
        return full(
            "TOO_MANY_SNAPSHOTS",
            "As many snapshots wait as may; try again",
        );
    };
    let source = Arc::clone(&operators.source);
    // The rows are copied and written as JSON off the runtime's workers,
    // the turn kept to the writing's own end.
    let taken = tokio::task::spawn_blocking(move || {
        let _turn = turn;
        serde_json::to_vec(&source.snapshot(&asked, Instant::now()))
    })
    .await;
    match taken {
        Ok(Ok(page)) => ([(CONTENT_TYPE, "application/json")], page).into_response(),
        Ok(Err(unwritten)) => {
            error!(%unwritten, "the policy snapshot did not write as JSON");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(join) => {
            error!(%join, "the policy snapshot's task failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// What `query` asks, or the refusal of a key by its name. The models pass
/// the policy crate's check of a snapshot's list, the one a served snapshot
/// reads ([`WorkerRowsQuery::refused`]). A `limit` is at least one: a served
/// call reads zero as no limit, and this route keeps its cap. A `limit` past
/// the cap is the cap, and an empty id between commas is no id.
fn asked(query: &SnapshotQuery) -> Result<WorkerRowsQuery, String> {
    let models = query.models.as_deref().unwrap_or_default().split(',');
    let models: Vec<String> = models
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_owned)
        .collect();
    if let Some(why) = WorkerRowsQuery::refused(&models) {
        return Err(format!("models: {why}"));
    }
    if query.limit == Some(0) {
        return Err(format!(
            "limit: a snapshot tells at least one row, and {SNAPSHOT_ROWS} where none is named"
        ));
    }
    Ok(WorkerRowsQuery {
        models,
        limit: query.limit.unwrap_or(SNAPSHOT_ROWS).min(SNAPSHOT_ROWS),
    })
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
