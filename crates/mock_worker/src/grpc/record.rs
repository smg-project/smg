//! The recorder (`--record-upstream URL`, with `--record PATH`): a gRPC
//! worker that stands between a gateway and an engine. Every call that
//! reaches it, of any service and method, goes to the engine as the messages
//! it came in and the engine's answer comes back as the messages it sent: the
//! calls pass as raw messages, so no proto of this crate is applied to what
//! is handed through. Beside that, each `Generate` exchange of a scheduler
//! wire is read (vLLM, TokenSpeed, SGLang) and appended to the record file as
//! lines `--script` reads, one per choice the request asked: the prompt's
//! ids, the output ids
//! with the engine's cuts (`chunks`), the finish reason, the matched stop,
//! the counts, the sampled logprobs, the scores of the token ids a scoring
//! request asked, a refusal's status with the trailers
//! that say what kind of failure it is, and under `recorded`
//! what the request asked beyond its prompt. Each gRPC worker keeps the
//! answers of `GetModelInfo` and `GetServerInfo` once each, as the bytes of
//! the message, in `PATH` with its extension replaced by `info.jsonl`; of a
//! server info answer's free-form maps it keeps what the gateway reads
//! ([`server_info`]).

use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    future::Future,
    io::{ErrorKind, Write},
    pin::Pin,
    sync::{Arc, Mutex, PoisonError},
};

use axum::{extract::State, http::uri::PathAndQuery};
use bytes::{Buf, BufMut, Bytes};
use futures::{future::BoxFuture, Stream, StreamExt};
use prost::Message;
use serde_json::{json, Map, Value};
use smg_grpc_client::{
    engine_labels::ServerInfo, sglang_proto as sg, tokenspeed_proto as ts, vllm_proto as vllm,
};
use tokio::net::TcpListener;
use tonic::{
    codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder},
    metadata::KeyAndValueRef,
    transport::{Channel, Endpoint, Server},
    Code, Request, Response, Status, Streaming,
};

use crate::{
    config::Config,
    replay::{append_owner_only, f32_json},
    script::{self, Refusal},
};

/// A message as its bytes, in either direction.
#[derive(Clone, Copy, Default)]
struct Raw;

impl Codec for Raw {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = Self;
    type Decoder = Self;

    fn encoder(&mut self) -> Self {
        Self
    }

    fn decoder(&mut self) -> Self {
        Self
    }
}

impl Encoder for Raw {
    type Item = Bytes;
    type Error = Status;

    fn encode(&mut self, item: Bytes, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        dst.put(item);
        Ok(())
    }
}

impl Decoder for Raw {
    type Item = Bytes;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Bytes>, Status> {
        Ok(Some(src.copy_to_bytes(src.remaining())))
    }
}

/// The scheduler wire a `Generate` call speaks, by its service's package.
#[derive(Clone, Copy)]
enum Wire {
    Vllm,
    TokenSpeed,
    Sglang,
}

impl Wire {
    fn name(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::TokenSpeed => "tokenspeed",
            Self::Sglang => "sglang",
        }
    }
}

/// What the recorder keeps of a call, by its path.
#[derive(Clone, Copy)]
enum Kept {
    Generate(Wire),
    ModelInfo,
    ServerInfo(Wire),
    Nothing,
}

fn kept(path: &str) -> Kept {
    let wire = if path.starts_with("/vllm.grpc.engine.") {
        Wire::Vllm
    } else if path.starts_with("/tokenspeed.grpc.scheduler.") {
        Wire::TokenSpeed
    } else if path.starts_with("/sglang.grpc.scheduler.") {
        Wire::Sglang
    } else {
        return Kept::Nothing;
    };
    match path.rsplit('/').next() {
        Some("Generate") => Kept::Generate(wire),
        Some("GetModelInfo") => Kept::ModelInfo,
        Some("GetServerInfo") => Kept::ServerInfo(wire),
        _ => Kept::Nothing,
    }
}

#[derive(Clone)]
struct Recorder {
    upstream: Channel,
    limit: usize,
    file: Arc<Mutex<File>>,
    info: Arc<Mutex<(File, HashSet<String>)>>,
}

/// Serve the recorder on `listener` until `shutdown` resolves.
pub(super) async fn serve(
    cfg: &Config,
    upstream: &str,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let recorder = match open(cfg, upstream) {
        Ok(recorder) => recorder,
        Err(e) => {
            tracing::error!("recorder: {e}");
            return;
        }
    };
    let router = axum::Router::new().fallback(forward).with_state(recorder);
    if let Err(e) = Server::builder()
        .initial_stream_window_size(Some(16 * 1024 * 1024))
        .initial_connection_window_size(Some(64 * 1024 * 1024))
        .max_frame_size(Some(1024 * 1024))
        .add_routes(tonic::service::Routes::from(router))
        .serve_with_incoming_shutdown(super::connections(listener), shutdown)
        .await
    {
        tracing::error!("recorder stopped: {e}");
    }
}

/// The parts of a recorder as `cfg` states them: the engine's endpoint, the
/// record file and the info file. What a process checks once before any
/// worker starts ([`super::recorder_starts`]).
pub(super) fn parts(cfg: &Config, upstream: &str) -> Result<(Endpoint, File, File), String> {
    let path = cfg
        .replay
        .record
        .as_ref()
        .ok_or("--record-upstream needs --record PATH")?;
    // Both files hold what the session carried: the owner's alone, as a
    // capture file is.
    let append = |path: &std::path::Path| {
        append_owner_only(path).map_err(|e| format!("{}: {e}", path.display()))
    };
    // The address may carry a credential: the error names the flag alone.
    let upstream = Endpoint::from_shared(upstream.replacen("grpc://", "http://", 1))
        .map_err(|e| format!("--record-upstream: {e}"))?;
    // An address with no scheme parses, as a bare host and port, and then
    // fails every call with nothing recorded: refused before a file is made.
    if upstream.uri().scheme().is_none() {
        return Err(
            "--record-upstream needs a scheme: http://HOST:PORT or grpc://HOST:PORT".to_string(),
        );
    }
    let upstream = upstream
        .initial_stream_window_size(16 * 1024 * 1024)
        .initial_connection_window_size(64 * 1024 * 1024)
        .tcp_nodelay(true);
    let record = append(path)?;
    Ok((
        upstream,
        record,
        append(&path.with_extension("info.jsonl"))?,
    ))
}

fn open(cfg: &Config, upstream: &str) -> Result<Recorder, String> {
    let (upstream, record, info) = parts(cfg, upstream)?;
    Ok(Recorder {
        upstream: upstream.connect_lazy(),
        limit: cfg.grpc_max_message_bytes,
        file: Arc::new(Mutex::new(record)),
        info: Arc::new(Mutex::new((info, HashSet::new()))),
    })
}

/// Every path lands here: the call as a stream of raw messages each way.
async fn forward(
    State(recorder): State<Recorder>,
    request: axum::extract::Request,
) -> axum::response::Response {
    let path = request
        .uri()
        .path_and_query()
        .cloned()
        .unwrap_or_else(|| PathAndQuery::from_static("/"));
    let limit = recorder.limit;
    let mut grpc =
        tonic::server::Grpc::new(Raw).apply_max_message_size_config(Some(limit), Some(limit));
    grpc.streaming(Call { recorder, path }, request)
        .await
        .map(axum::body::Body::new)
}

struct Call {
    recorder: Recorder,
    path: PathAndQuery,
}

type Answered = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send>>;

/// The headers of one hop: the recorder's own transport states them again.
const HOP: [&str; 5] = [
    "grpc-encoding",
    "grpc-accept-encoding",
    "content-type",
    "content-length",
    "te",
];

impl tonic::server::StreamingService<Bytes> for Call {
    type Response = Bytes;
    type ResponseStream = Answered;
    type Future = BoxFuture<'static, Result<Response<Answered>, Status>>;

    fn call(&mut self, request: Request<Streaming<Bytes>>) -> Self::Future {
        let recorder = self.recorder.clone();
        let path = self.path.clone();
        Box::pin(async move {
            let exchange = Arc::new(Exchange {
                kept: kept(path.path()),
                path: path.path().to_string(),
                recorder: recorder.clone(),
                tapped: Mutex::new(Tapped::default()),
            });
            let (mut metadata, _, incoming) = request.into_parts();
            for name in HOP {
                metadata.remove(name);
            }
            // The status of a request message that did not read: the caller
            // reads it in place of what the engine says to the cut request.
            let unread = Arc::new(Mutex::new(None::<Status>));
            let asked = Arc::clone(&exchange);
            let failed = Arc::clone(&unread);
            let mut outgoing = Request::new(incoming.scan((), move |(), message| {
                let message = match message {
                    Ok(bytes) => {
                        asked.asked(&bytes);
                        Some(bytes)
                    }
                    // The forwarded request ends here: behind a message that
                    // did not read (one over the size limit) the stream has
                    // nothing more to hand on, and every later read of it
                    // fails again.
                    Err(status) => {
                        tracing::warn!(
                            "recorder: {}: a request message did not read: {:?}: {}",
                            asked.path,
                            status.code(),
                            status.message()
                        );
                        *failed.lock().unwrap_or_else(PoisonError::into_inner) = Some(status);
                        None
                    }
                };
                std::future::ready(message)
            }));
            *outgoing.metadata_mut() = metadata;
            let mut client = tonic::client::Grpc::new(recorder.upstream.clone())
                .max_decoding_message_size(recorder.limit)
                .max_encoding_message_size(recorder.limit);
            client
                .ready()
                .await
                .map_err(|e| Status::unavailable(format!("the recorder's upstream: {e}")))?;
            let answer = match client.streaming(outgoing, path, Raw).await {
                Ok(answer) => answer,
                Err(status) => {
                    let status = taken(&unread).unwrap_or_else(|| unreached(status));
                    exchange.refused(&status);
                    return Err(status);
                }
            };
            let (mut metadata, answered, _) = answer.into_parts();
            for name in HOP {
                metadata.remove(name);
            }
            let answered = futures::stream::unfold(Some(answered), move |answered| {
                let exchange = Arc::clone(&exchange);
                let unread = Arc::clone(&unread);
                async move {
                    let mut answered = answered?;
                    let ended = match answered.message().await {
                        Ok(Some(bytes)) => {
                            exchange.answered(&bytes);
                            return Some((Ok(bytes), Some(answered)));
                        }
                        Ok(None) => None,
                        // A failure of the transport behind the engine's
                        // headers reads as one ahead of them.
                        Err(status) => Some(unreached(status)),
                    };
                    if let Some(status) = taken(&unread).or(ended) {
                        exchange.refused(&status);
                        return Some((Err(status), None));
                    }
                    // The call ended well: the trailers the engine sent with
                    // its OK go on behind the last message. A status of the
                    // code OK is written as the trailers of a call that
                    // ended well, with its metadata.
                    let mut trailers = answered.trailers().await.ok().flatten()?;
                    for name in HOP.into_iter().chain(["grpc-status", "grpc-message"]) {
                        trailers.remove(name);
                    }
                    (!trailers.is_empty())
                        .then(|| (Err(Status::with_metadata(Code::Ok, "", trailers)), None))
                }
            });
            let mut response = Response::new(Box::pin(answered) as Answered);
            *response.metadata_mut() = metadata;
            Ok(response)
        })
    }
}

/// The status kept of a request message that did not read, taken once.
fn taken(unread: &Mutex<Option<Status>>) -> Option<Status> {
    unread.lock().unwrap_or_else(PoisonError::into_inner).take()
}

/// A status that ended a call, as the caller reads it. One the client made
/// of an error of the transport (the engine could not be reached, the
/// connection failed under the call, or the engine reset the call's stream)
/// carries that error as its source: it names the recorder's upstream ahead
/// of its words and keeps its code, the one a caller of the engine itself
/// reads (`UNAVAILABLE` for a failed connect, the code of its reason for a
/// reset). Every other status carries no source and passes as it is: one
/// read off the wire, and one the client made of an answer that ended with
/// no gRPC status.
fn unreached(status: Status) -> Status {
    match std::error::Error::source(&status) {
        Some(_) => {
            let words = format!("the recorder's upstream: {}", status.message());
            Status::new(status.code(), words)
        }
        None => status,
    }
}

/// One call as the recorder saw it: written when the call is over, however
/// it ended (the engine's completion, a status, or the gateway letting go).
struct Exchange {
    kept: Kept,
    path: String,
    recorder: Recorder,
    tapped: Mutex<Tapped>,
}

#[derive(Default)]
struct Tapped {
    request: Option<Bytes>,
    choices: BTreeMap<u32, Choice>,
    status: Option<Refusal>,
    /// A frame numbered a choice at or past [`CHOICES`]: nothing is kept of
    /// it but this.
    strayed: bool,
}

/// The most choices a request asks: what a gateway sends at most.
const CHOICES: u32 = 128;

#[derive(Default)]
struct Choice {
    ids: Vec<u32>,
    chunks: Vec<usize>,
    logprobs: Vec<f32>,
    /// The prompt and the cached tokens the last chunk stated: what the line
    /// states where no completion came.
    counted: Option<(u32, u32)>,
    complete: Option<Completed>,
}

struct Completed {
    output_ids: Vec<u32>,
    finish_reason: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    matched_stop: Value,
    logprobs: Vec<f32>,
    scores: Option<(Vec<u32>, Vec<f32>)>,
}

/// The scores a completion of the SGLang wire carries for the token ids its
/// request asked, the ids and their logprobs: the one row of a prefill-only
/// scoring request. A completion with a row for each output token keeps none.
fn scores(logprobs: Option<&sg::OutputLogProbs>) -> Option<(Vec<u32>, Vec<f32>)> {
    match logprobs?.token_ids_logprobs.as_slice() {
        [row] => Some((row.token_ids.clone(), row.values.clone())),
        _ => None,
    }
}

/// The other two wires carry no scores row.
fn no_scores<T>(_: Option<&T>) -> Option<(Vec<u32>, Vec<f32>)> {
    None
}

/// One answer message of a `Generate` stream, read off its wire: the three
/// protos name these fields alike, and `$scores` reads the scores row of a
/// completion where the wire has one.
macro_rules! answered {
    ($proto:ident, $bytes:expr, $tapped:expr, $scores:expr) => {{
        use $proto::{generate_complete::MatchedStop, generate_response::Response as Piece};
        match $proto::GenerateResponse::decode($bytes)
            .ok()
            .and_then(|m| m.response)
        {
            Some(Piece::Chunk(chunk)) if chunk.index < CHOICES => {
                let choice = $tapped.choices.entry(chunk.index).or_default();
                choice.counted = Some((chunk.prompt_tokens, chunk.cached_tokens));
                choice.chunks.push(chunk.token_ids.len());
                choice.ids.extend(chunk.token_ids);
                if let Some(logprobs) = chunk.output_logprobs {
                    choice.logprobs.extend(logprobs.token_logprobs);
                }
            }
            Some(Piece::Complete(complete)) if complete.index < CHOICES => {
                let scores = $scores(complete.output_logprobs.as_ref());
                $tapped.choices.entry(complete.index).or_default().complete = Some(Completed {
                    output_ids: complete.output_ids,
                    finish_reason: complete.finish_reason,
                    prompt_tokens: complete.prompt_tokens,
                    completion_tokens: complete.completion_tokens,
                    cached_tokens: complete.cached_tokens,
                    matched_stop: match complete.matched_stop {
                        Some(MatchedStop::MatchedTokenId(id)) => json!(id),
                        Some(MatchedStop::MatchedStopStr(text)) => json!(text),
                        None => Value::Null,
                    },
                    logprobs: complete
                        .output_logprobs
                        .map(|logprobs| logprobs.token_logprobs)
                        .unwrap_or_default(),
                    scores,
                });
            }
            Some(_) => $tapped.strayed = true,
            None => {}
        }
    }};
}

impl Exchange {
    fn tapped(&self) -> std::sync::MutexGuard<'_, Tapped> {
        self.tapped.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A message of the request: the first is the request of a unary or
    /// server-streaming call.
    fn asked(&self, bytes: &Bytes) {
        if matches!(self.kept, Kept::Generate(_)) {
            self.tapped().request.get_or_insert_with(|| bytes.clone());
        }
    }

    /// The status a call ended with: its code, its message and the trailers
    /// by which a worker says what kind of failure it reports (the ones
    /// named `smg-...`: whose fault a media failure is, a constraint the
    /// engine refused); the transport's own headers stay out.
    fn refused(&self, status: &Status) {
        let trailers = status
            .metadata()
            .iter()
            .filter_map(|entry| match entry {
                KeyAndValueRef::Ascii(name, value) if name.as_str().starts_with("smg-") => {
                    Some((name.as_str().to_string(), value.to_str().ok()?.to_string()))
                }
                _ => None,
            })
            .collect();
        self.tapped().status = Some(Refusal {
            code: status.code() as i32,
            message: status.message().to_string(),
            trailers,
        });
    }

    /// An info answer, the first of its path the info file took: one line of
    /// that file. A line the file took nothing of is written by the next
    /// answer of its path; one it took a part of is not written again, so no
    /// line follows a part of a line.
    fn keep_info(&self, bytes: &[u8]) {
        let mut info = self
            .recorder
            .info
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if info.1.contains(&self.path) {
            return;
        }
        let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        // One write call a line, as a record line is written:
        // each gRPC worker appends through a handle of its own.
        let line = format!("{}\n", json!({ "path": self.path, "answer_hex": hex }));
        let mut rest = line.as_bytes();
        while !rest.is_empty() {
            match info.0.write(rest) {
                Ok(0) => {
                    tracing::error!("recorder: the info file took no more of a line");
                    break;
                }
                Ok(taken) => rest = &rest[taken..],
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    tracing::error!("recorder: the info file: {e}");
                    break;
                }
            }
        }
        if rest.len() < line.len() {
            info.1.insert(self.path.clone());
        }
    }

    fn answered(&self, bytes: &Bytes) {
        match self.kept {
            Kept::Nothing => {}
            Kept::ModelInfo => self.keep_info(bytes),
            Kept::ServerInfo(wire) => match server_info(wire, bytes) {
                Some(kept) => self.keep_info(&kept),
                None => tracing::error!(
                    "recorder: a server info answer that does not read as its wire's is not kept"
                ),
            },
            Kept::Generate(wire) => {
                let mut tapped = self.tapped();
                let bytes = bytes.clone();
                match wire {
                    Wire::Vllm => answered!(vllm, bytes, tapped, no_scores),
                    Wire::TokenSpeed => answered!(ts, bytes, tapped, no_scores),
                    Wire::Sglang => answered!(sg, bytes, tapped, scores),
                }
            }
        }
    }
}

/// A `GetServerInfo` answer as the info file keeps it. An engine hands two
/// free-form maps there, the arguments it was started with and its
/// scheduler's state, and a start argument may be a credential: of each map
/// the recorder keeps the entries the gateway reads as a label
/// ([`ServerInfo::to_labels`]) and no other, so a replay states the labels
/// the engine stated and the file holds no argument nobody reads. The vLLM
/// wire's answer has no such map and is kept as it came. `None` when the
/// bytes do not read as the wire's answer.
fn server_info(wire: Wire, bytes: &Bytes) -> Option<Vec<u8>> {
    macro_rules! read {
        ($proto:ident, $wire:ident) => {{
            let mut info = $proto::GetServerInfoResponse::decode(bytes.clone()).ok()?;
            info.server_args = read_entries(info.server_args.take(), |server_args| {
                ServerInfo::$wire(Box::new($proto::GetServerInfoResponse {
                    server_args,
                    ..Default::default()
                }))
                .to_labels()
            });
            info.scheduler_info = read_entries(info.scheduler_info.take(), |scheduler_info| {
                ServerInfo::$wire(Box::new($proto::GetServerInfoResponse {
                    scheduler_info,
                    ..Default::default()
                }))
                .to_labels()
            });
            info.encode_to_vec()
        }};
    }
    Some(match wire {
        Wire::Vllm => bytes.to_vec(),
        Wire::TokenSpeed => read!(ts, TokenSpeed),
        Wire::Sglang => read!(sg, Sglang),
    })
}

/// The entries of `map` that change what `labels` reads, each asked alone.
fn read_entries<L: PartialEq>(
    map: Option<prost_types::Struct>,
    labels: impl Fn(Option<prost_types::Struct>) -> L,
) -> Option<prost_types::Struct> {
    let mut map = map?;
    let unread = labels(None);
    map.fields.retain(|name, value| {
        let fields = [(name.clone(), value.clone())].into();
        labels(Some(prost_types::Struct { fields })) != unread
    });
    Some(map)
}

/// The prompt's ids of a `Generate` request and what it asks beyond them,
/// as the wire's own worker reads it at a replay; `None` when the bytes do
/// not read as the wire's request.
fn asked(wire: Wire, bytes: &Bytes) -> Option<(Vec<u32>, Value)> {
    let bytes = bytes.clone();
    Some(match wire {
        Wire::Vllm => {
            let request = vllm::GenerateRequest::decode(bytes).ok()?;
            let asked = super::vllm::asked(&request);
            let ids = match request.input {
                Some(vllm::generate_request::Input::Tokenized(tokenized)) => tokenized.input_ids,
                _ => Vec::new(),
            };
            (ids, asked)
        }
        Wire::TokenSpeed => {
            let request = ts::GenerateRequest::decode(bytes).ok()?;
            let asked = super::tokenspeed::asked(&request);
            (
                request.tokenized.map(|t| t.input_ids).unwrap_or_default(),
                asked,
            )
        }
        Wire::Sglang => {
            let request = sg::GenerateRequest::decode(bytes).ok()?;
            let asked = super::sglang::asked(&request);
            (
                request.tokenized.map(|t| t.input_ids).unwrap_or_default(),
                asked,
            )
        }
    })
}

impl Drop for Exchange {
    fn drop(&mut self) {
        let Kept::Generate(wire) = self.kept else {
            return;
        };
        let tapped = self
            .tapped
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        let Some((input_ids, asked)) = tapped.request.take().and_then(|b| asked(wire, &b)) else {
            return;
        };
        // What the request asks beyond its prompt, folded as a replay folds
        // it: two requests of one prompt that ask alike share their key.
        let variant = script::ask_fold(&asked);
        // A line for each choice the request asked (up to the 128 a gateway
        // sends), whatever the call lived to deliver: the calls of one key
        // hold the same count of lines, so a replay hands each call its own.
        let choices = asked.get("n").and_then(Value::as_u64);
        let choices = choices.map_or(1, |n| n.clamp(1, u64::from(CHOICES)));
        for index in 0..u32::try_from(choices).unwrap_or(1) {
            tapped.choices.entry(index).or_default();
        }
        // A choice the engine numbered past the count asked gets no line: a
        // line more than its key's other calls hold would hand every later
        // call of the key its neighbour's answer.
        let stray = tapped
            .choices
            .split_off(&u32::try_from(choices).unwrap_or(1));
        if tapped.strayed || !stray.is_empty() {
            tracing::warn!(
                "recorder: the engine numbered a choice past the {choices} its request asked: \
                 that choice is not recorded"
            );
        }
        let mut lines = Vec::new();
        for (index, choice) in &tapped.choices {
            let complete = choice.complete.as_ref();
            // The chunks carried the output where one of them held a token;
            // else the completion did, were a chunk of no tokens ahead of it.
            let chunked = !choice.ids.is_empty();
            let output_ids = match complete {
                Some(complete) if !chunked => &complete.output_ids,
                _ => &choice.ids,
            };
            let logprobs = match complete {
                Some(complete) if !chunked => &complete.logprobs,
                _ => &choice.logprobs,
            };
            let mut line = Map::new();
            line.insert("input_ids".into(), json!(input_ids));
            if output_ids.is_empty() {
                line.insert("output_text".into(), json!(""));
            } else {
                line.insert("output_ids".into(), json!(output_ids));
            }
            // The cuts of every chunk seen, were it one of no tokens. A
            // streamed call that carried its output in the completion alone
            // states that it had none: a replay streams no chunk either.
            let alone = asked.get("stream") == Some(&json!(true)) && !output_ids.is_empty();
            if !choice.chunks.is_empty() || alone {
                line.insert("chunks".into(), json!(choice.chunks));
            }
            let finish_reason = match (complete, &tapped.status) {
                (Some(complete), _) => complete.finish_reason.as_str(),
                (None, Some(_)) => "error",
                (None, None) => "abort",
            };
            line.insert("finish_reason".into(), json!(finish_reason));
            if let Some(complete) = complete {
                if !complete.matched_stop.is_null() {
                    line.insert("matched_stop".into(), complete.matched_stop.clone());
                }
                line.insert("prompt_tokens".into(), json!(complete.prompt_tokens));
                line.insert("cached_tokens".into(), json!(complete.cached_tokens));
                line.insert(
                    "completion_tokens".into(),
                    json!(complete.completion_tokens),
                );
            } else if let Some((prompt_tokens, cached_tokens)) = choice.counted {
                // No completion came: the frames of the replay state what
                // the engine's last chunk stated.
                line.insert("prompt_tokens".into(), json!(prompt_tokens));
                line.insert("cached_tokens".into(), json!(cached_tokens));
            }
            if !logprobs.is_empty() && logprobs.len() == output_ids.len() {
                let logprobs: Vec<Value> = logprobs.iter().copied().map(f32_json).collect();
                line.insert("logprobs".into(), json!(logprobs));
            }
            // The scores of the token ids the request asked, in its order:
            // what a replay hands a scoring request back.
            if let Some((ids, values)) = complete.and_then(|c| c.scores.as_ref()) {
                if asked.get("token_ids_logprob") == Some(&json!(ids)) {
                    let values: Vec<Value> = values.iter().copied().map(f32_json).collect();
                    line.insert("token_id_logprobs".into(), json!(values));
                }
            }
            if let Some(refusal) = &tapped.status {
                let mut status = json!({ "code": refusal.code, "message": refusal.message });
                if !refusal.trailers.is_empty() {
                    let trailers: Map<String, Value> = refusal
                        .trailers
                        .iter()
                        .map(|(name, value)| (name.clone(), json!(value)))
                        .collect();
                    status["trailers"] = Value::Object(trailers);
                }
                line.insert("status".into(), status);
            }
            line.insert(
                "recorded".into(),
                json!({
                    "wire": wire.name(),
                    "index": index,
                    "variant": format!("{variant:016x}"),
                    "asked": asked,
                    "complete_ids": complete.map(|c| c.output_ids.len()),
                }),
            );
            lines.push(Value::Object(line).to_string());
        }
        let mut text = lines.join("\n");
        text.push('\n');
        let mut file = self
            .recorder
            .file
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Err(e) = file.write_all(text.as_bytes()) {
            tracing::error!("recorder: the record file: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use smg_grpc_client::engine_labels::ServerInfo as Gateway;

    use super::*;

    /// A call of `path` at a recorder whose two files stand under `dir`, in
    /// front of an engine no test reaches.
    fn exchange(dir: &Path, path: &str) -> Exchange {
        let mut cfg = Config::default();
        cfg.replay.record = Some(dir.join("session.jsonl"));
        Exchange {
            kept: kept(path),
            path: path.to_string(),
            recorder: open(&cfg, "http://127.0.0.1:9").expect("a recorder on two new files"),
            tapped: Mutex::default(),
        }
    }

    /// The one answer the info file under `dir` holds, as its bytes.
    fn kept_answer(dir: &Path) -> Vec<u8> {
        let info = std::fs::read_to_string(dir.join("session.info.jsonl")).expect("the info file");
        let line: Value = serde_json::from_str(info.trim_end()).expect("one line");
        let hex = line["answer_hex"].as_str().expect("the answer in hex");
        hex.as_bytes()
            .chunks(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).expect("hex digits");
                u8::from_str_radix(pair, 16).expect("a byte in hex")
            })
            .collect()
    }

    /// The info file keeps of a server info answer the entries the gateway
    /// reads as labels and no other start argument: a key an engine was
    /// started with stays out of the file, and the labels read as before.
    #[tokio::test]
    async fn a_start_argument_nobody_reads_stays_out_of_the_info_file() {
        let text = |value: &str| prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(value.to_string())),
        };
        let args = prost_types::Struct {
            fields: [
                ("api_key".to_string(), text("a-key-of-the-engine")),
                ("mm_processor".to_string(), text("a-processor")),
            ]
            .into(),
        };

        let dir = tempfile::tempdir().expect("temp dir");
        let whole = sg::GetServerInfoResponse {
            server_args: Some(args.clone()),
            scheduler_info: Some(args.clone()),
            sglang_version: "1.2.3".to_string(),
            ..Default::default()
        };
        exchange(
            dir.path(),
            "/sglang.grpc.scheduler.SglangScheduler/GetServerInfo",
        )
        .answered(&Bytes::from(whole.encode_to_vec()));
        let kept = sg::GetServerInfoResponse::decode(kept_answer(dir.path()).as_slice())
            .expect("the kept answer reads as the wire's");
        let names = |map: &Option<prost_types::Struct>| -> Vec<String> {
            map.iter()
                .flat_map(|map| map.fields.keys().cloned())
                .collect()
        };
        assert_eq!(names(&kept.server_args), ["mm_processor"]);
        assert!(names(&kept.scheduler_info).is_empty());
        assert_eq!(kept.sglang_version, "1.2.3");
        assert_eq!(
            Gateway::Sglang(Box::new(kept)).to_labels(),
            Gateway::Sglang(Box::new(whole)).to_labels(),
            "the gateway reads the kept answer as it reads the engine's"
        );

        let dir = tempfile::tempdir().expect("temp dir");
        let whole = ts::GetServerInfoResponse {
            server_args: Some(args),
            tokenspeed_version: "4.5.6".to_string(),
            ..Default::default()
        };
        exchange(
            dir.path(),
            "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/GetServerInfo",
        )
        .answered(&Bytes::from(whole.encode_to_vec()));
        let kept = ts::GetServerInfoResponse::decode(kept_answer(dir.path()).as_slice())
            .expect("the kept answer reads as the wire's");
        assert!(!names(&kept.server_args).contains(&"api_key".to_string()));
        assert_eq!(kept.tokenspeed_version, "4.5.6");
        assert_eq!(
            Gateway::TokenSpeed(Box::new(kept)).to_labels(),
            Gateway::TokenSpeed(Box::new(whole)).to_labels(),
            "the gateway reads the kept answer as it reads the engine's"
        );
    }

    /// A frame that numbers a choice its request did not ask leaves no line
    /// (the calls of one key hold the same count of lines, on which the turn
    /// of a replay stands), and one numbered past the most a request asks
    /// leaves nothing at all.
    #[tokio::test]
    async fn a_choice_the_request_did_not_ask_gets_no_line() {
        let dir = tempfile::tempdir().expect("temp dir");
        let call = exchange(
            dir.path(),
            "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/Generate",
        );
        let request = ts::GenerateRequest {
            tokenized: Some(ts::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            ..Default::default()
        };
        call.asked(&Bytes::from(request.encode_to_vec()));
        for index in [0, 7, 4_000_000] {
            let chunk = ts::GenerateStreamChunk {
                token_ids: vec![5],
                index,
                ..Default::default()
            };
            let frame = ts::GenerateResponse {
                response: Some(ts::generate_response::Response::Chunk(chunk)),
                ..Default::default()
            };
            call.answered(&Bytes::from(frame.encode_to_vec()));
        }
        assert_eq!(
            call.tapped().choices.keys().copied().collect::<Vec<_>>(),
            [0, 7],
            "no entry for a choice past the most a request asks"
        );
        drop(call);

        let record =
            std::fs::read_to_string(dir.path().join("session.jsonl")).expect("the record file");
        let indexes: Vec<u64> = record
            .lines()
            .map(|line| {
                let line: Value = serde_json::from_str(line).expect("a line of the record");
                line["recorded"]["index"].as_u64().expect("its index")
            })
            .collect();
        assert_eq!(indexes, [0], "one line: the choice the request asked");
    }

    /// An info answer whose line the file took nothing of is written by the
    /// next answer of its path, and the answers behind that one are not.
    #[tokio::test]
    async fn an_info_answer_the_file_did_not_take_is_written_by_the_next() {
        let dir = tempfile::tempdir().expect("temp dir");
        let call = exchange(
            dir.path(),
            "/sglang.grpc.scheduler.SglangScheduler/GetModelInfo",
        );
        // A handle that takes no write stands in for a file that refuses one.
        let refusing = File::open(dir.path().join("session.info.jsonl")).expect("read only");
        let hand = |handle: File| {
            let mut info = call.recorder.info.lock().expect("the info file");
            std::mem::replace(&mut info.0, handle)
        };
        let taking = hand(refusing);
        call.keep_info(b"first");
        drop(hand(taking));
        call.keep_info(b"second");
        call.keep_info(b"third");
        assert_eq!(kept_answer(dir.path()), b"second", "one line, the second");
    }

    /// A streamed call whose chunks cut no token writes a line the script
    /// reads: the line keeps the completion's ids, and its cuts hold none of
    /// them, no cut for a completion alone and one for a chunk of no tokens
    /// ahead of it.
    #[tokio::test]
    async fn a_call_whose_chunks_cut_no_token_writes_a_line_the_script_reads() {
        use ts::generate_response::Response as Piece;
        for (empty_chunks, cuts) in [(0, vec![]), (1, vec![0])] {
            let dir = tempfile::tempdir().expect("temp dir");
            let call = exchange(
                dir.path(),
                "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/Generate",
            );
            let request = ts::GenerateRequest {
                tokenized: Some(ts::TokenizedInput {
                    input_ids: vec![1, 2, 3],
                    original_text: String::new(),
                }),
                stream: true,
                ..Default::default()
            };
            call.asked(&Bytes::from(request.encode_to_vec()));
            let frame = |response| ts::GenerateResponse {
                response: Some(response),
                ..Default::default()
            };
            for _ in 0..empty_chunks {
                let chunk = Piece::Chunk(ts::GenerateStreamChunk::default());
                call.answered(&Bytes::from(frame(chunk).encode_to_vec()));
            }
            let complete = Piece::Complete(ts::GenerateComplete {
                output_ids: vec![7, 8, 9],
                finish_reason: "stop".to_string(),
                completion_tokens: 3,
                ..Default::default()
            });
            call.answered(&Bytes::from(frame(complete).encode_to_vec()));
            drop(call);

            let record =
                std::fs::read_to_string(dir.path().join("session.jsonl")).expect("the record file");
            let line: Value = serde_json::from_str(record.trim_end()).expect("one line");
            let answer = script::Answer::parse(&line)
                .expect("a line the script reads")
                .answer;
            assert_eq!(
                answer.chunks,
                Some(cuts),
                "{empty_chunks} chunks of no tokens"
            );
            assert_eq!(answer.output_ids, [7, 8, 9]);
            assert_eq!(answer.chunk_frames().len(), empty_chunks);
            assert_eq!(answer.completion_tokens(), 3);
        }
    }
}
